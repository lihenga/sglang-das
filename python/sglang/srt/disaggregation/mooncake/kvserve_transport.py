"""Multi-request KVServe sessions with bounded Mooncake transfer credits.

Control RPCs never wait for GPU work: commit/status return pending until the
scheduler restores the destination. Wire slots stay owned until ACK, regardless
of the number of admitted model requests. Uncertain writes quarantine registered
buffers instead of handing them to a different request.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
PROTOCOL_VERSION = 2
REGIONS = ("payload", "aux", "tail")
MAX_RETIRED = 4096


class TransferCancelled(RuntimeError):
    pass


class RemoteTransferError(RuntimeError):
    def __init__(self, response):
        super().__init__(response.get("error", "KVServe peer failed"))
        self.drained = response.get("drained") is True


class SlotCredits:
    """FIFO admission; cancelled/timed-out waiters never consume a credit."""

    def __init__(self, count):
        self.condition = threading.Condition()
        self.free = deque(range(count))
        self.owned = set()
        self.waiters = deque()
        self.poisoned = None

    def acquire(self, timeout, is_active):
        ticket = object()
        deadline = time.monotonic() + timeout
        with self.condition:
            self.waiters.append(ticket)
            try:
                while True:
                    if self.poisoned:
                        raise RuntimeError(
                            f"KVServe sender quarantined: {self.poisoned}"
                        )
                    if not is_active():
                        raise TransferCancelled("KVServe source request was aborted")
                    if self.waiters[0] is ticket and self.free:
                        slot = self.free.popleft()
                        self.owned.add(slot)
                        return slot
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("KVServe source slot queue timed out")
                    self.condition.wait(min(remaining, 0.05))
            finally:
                self.waiters.remove(ticket)
                self.condition.notify_all()

    def release(self, slot):
        with self.condition:
            if slot not in self.owned:
                raise RuntimeError("KVServe source slot was released twice")
            if not self.poisoned:
                self.owned.remove(slot)
                self.free.append(slot)
            self.condition.notify_all()

    def quarantine(self, reason):
        with self.condition:
            self.poisoned = str(reason)
            self.condition.notify_all()


def validate_header(header, adapter):
    if header.get("version") != PROTOCOL_VERSION:
        raise ValueError("KVServe protocol version mismatch")
    if header.get("profile") != adapter.profile:
        raise ValueError("KVServe profile/TP/cache geometry mismatch")
    request_id = header.get("request_id")
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
        raise ValueError("Invalid KVServe chunk request id")
    if type(header.get("room")) is not int:
        raise ValueError("Invalid KVServe room")
    if type(header.get("part")) is not int or header["part"] not in (0, 1):
        raise ValueError("Invalid KV part")
    pages = header.get("pages")
    if not isinstance(pages, list) or not 1 <= len(pages) <= adapter.chunk_pages:
        raise ValueError("Invalid KVServe chunk page count")
    if any(type(p) is not int or p < 0 for p in pages) or len(set(pages)) != len(pages):
        raise ValueError("Invalid KVServe physical pages")
    tokens = header.get("valid_tokens")
    if type(tokens) is not int or not (
        (len(pages) - 1) * adapter.page_size < tokens <= len(pages) * adapter.page_size
    ):
        raise ValueError("Invalid KVServe valid-token count")
    sizes = header.get("sizes")
    if not isinstance(sizes, dict) or set(sizes) != {*REGIONS, "u8_bytes"}:
        raise ValueError("Invalid KVServe byte counts")
    for name, count in sizes.items():
        if type(count) is not int or count < 0:
            raise ValueError("Invalid KVServe byte count")
        limit = (
            adapter.receive_regions()[name][1]
            if name in REGIONS
            else adapter.max_u8_bytes
        )
        if count > limit:
            raise ValueError(f"KVServe {name} exceeds receiver capacity")
    adapter.validate_pages(pages)
    adapter.validate_sizes(header["part"], tokens, sizes)


@dataclass
class ReceiveJob:
    header: dict
    token: str = field(default_factory=lambda: uuid.uuid4().hex)
    created: float = field(default_factory=time.monotonic)
    granted_at: float | None = None
    slot_id: int | None = None
    done: threading.Event = field(default_factory=threading.Event)
    committed: bool = False
    started: bool = False
    cancelled: bool = False
    error: str | None = None


class CompressionReceiver:
    def __init__(self, adapter, is_active, fail_room, timeout):
        self.adapter = adapter
        self.is_active = is_active
        self.fail_room = fail_room
        self.timeout = timeout
        self.lock = threading.RLock()
        self.free_slots = deque(range(adapter.slot_count))
        self.jobs = {}
        self.requests = {}
        self.waiters = deque()
        self.inflight = {}
        self.retired = OrderedDict()
        self.retired_requests = {}
        self.closed_rooms = OrderedDict()
        self.page_owners = {}
        self.poisoned = None
        self.endpoint = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._startup_error = None
        self.expected = {}
        self.completed = {}

    def expect_pages(self, room, pages):
        self.adapter.validate_pages(pages)
        with self.lock:
            if room in self.expected or room in self.closed_rooms:
                raise RuntimeError("KVServe destination room was registered twice")
            if any(page in self.page_owners for page in pages):
                raise RuntimeError("KVServe destination pages belong to another room")
            self.expected[room] = tuple(pages)
            self.completed[room] = [0, 0]
            self.page_owners.update((page, room) for page in pages)

    def is_complete(self, room):
        with self.lock:
            return (
                self.poisoned is None
                and room in self.expected
                and self.completed[room] == [len(self.expected[room])] * 2
            )

    def _result(self, job):
        if job.error:
            return {
                "status": "error",
                "error": job.error,
                "drained": self.poisoned is None and job.done.is_set(),
            }
        return {"status": "done", "drained": True}

    def _retire(self, job, result):
        """Caller holds lock and has proved that no DMA/GPU work remains."""
        self.jobs.pop(job.token, None)
        request_id = job.header["request_id"]
        self.requests.pop(request_id, None)
        if job.slot_id is None:
            try:
                self.waiters.remove(job)
            except ValueError:
                pass
        key = (job.header["room"], job.header["part"])
        if self.inflight.get(key) == request_id:
            self.inflight.pop(key)
        if job.slot_id is not None and not self.poisoned:
            self.free_slots.append(job.slot_id)
        self.retired[job.token] = (job.header, result)
        self.retired_requests[request_id] = job.token
        while len(self.retired) > MAX_RETIRED:
            _, (old_header, _) = self.retired.popitem(last=False)
            self.retired_requests.pop(old_header["request_id"], None)

    def _schedule(self):
        while self.free_slots and self.waiters and not self.poisoned:
            job = self.waiters.popleft()
            if job.token not in self.jobs:
                continue
            if not self.is_active(job.header["room"]):
                job.error = "KVServe destination request is no longer active"
                job.done.set()
                self._retire(job, self._result(job))
                continue
            job.slot_id = self.free_slots.popleft()
            job.granted_at = time.monotonic()

    def _quarantine(self, reason):
        with self.lock:
            self.poisoned = str(reason)
            self.free_slots.clear()
            rooms = {job.header["room"] for job in self.jobs.values()}
            for job in self.jobs.values():
                job.error = self.poisoned
                if not job.started:
                    job.done.set()
        for room in rooms:
            self.fail_room(room, self.poisoned)
        logger.error("KVServe staging quarantined; restart worker: %s", reason)

    def forget(self, room):
        # Called on the scheduler thread, like advance(), never concurrently
        # with scatter. A committed transfer has already completed native DMA.
        with self.lock:
            pages = self.expected.pop(room, ())
            self.completed.pop(room, None)
            for page in pages:
                self.page_owners.pop(page, None)
            self.closed_rooms[room] = None
            while len(self.closed_rooms) > MAX_RETIRED:
                self.closed_rooms.popitem(last=False)
            for job in list(self.jobs.values()):
                if job.header["room"] != room:
                    continue
                job.cancelled = True
                job.error = "KVServe destination request was released"
                if job.slot_id is not None and not (job.committed or job.done.is_set()):
                    # The old sender may still write this wire slot. Retain it
                    # until its commit/drained-cancel proves DMA has stopped.
                    # It can never scatter into the released KV pages.
                    continue
                job.done.set()
                self._retire(job, self._result(job))
            self._schedule()

    def _expire_jobs(self):
        now = time.monotonic()
        failed = []
        uncertain = False
        with self.lock:
            if self.poisoned:
                return
            for job in list(self.jobs.values()):
                if now - (job.granted_at or job.created) <= self.timeout:
                    continue
                if job.done.is_set():
                    self._retire(job, self._result(job))
                elif job.slot_id is None:
                    job.error = "KVServe destination slot queue timed out"
                    job.done.set()
                    failed.append(job.header["room"])
                    self._retire(job, self._result(job))
                else:
                    uncertain = True
            if not uncertain:
                self._schedule()
        for room in failed:
            self.fail_room(room, "KVServe destination slot queue timed out")
        if uncertain:
            self._quarantine("KVServe receive lease timed out")

    def handle(self, message):
        operation = message.get("op")
        if operation == "prepare":
            header = message["header"]
            validate_header(header, self.adapter)
            with self.lock:
                if self.poisoned:
                    raise RuntimeError(f"KVServe receiver quarantined: {self.poisoned}")
                request_id = header["request_id"]
                old_token = self.retired_requests.get(request_id)
                if old_token is not None:
                    old_header, result = self.retired[old_token]
                    if old_header != header:
                        raise RuntimeError("KVServe chunk request id was reused")
                    return result
                job = self.requests.get(request_id)
                if job is not None and job.header != header:
                    raise RuntimeError("KVServe chunk request id was reused")
                if job is None:
                    room, part = header["room"], header["part"]
                    expected = self.expected.get(room)
                    if expected is None:
                        raise RuntimeError(
                            "KVServe destination pages were not registered"
                        )
                    if not self.is_active(room):
                        raise RuntimeError(
                            "KVServe destination request is no longer active"
                        )
                    start = self.completed[room][part]
                    if (
                        tuple(header["pages"])
                        != expected[start : start + len(header["pages"])]
                    ):
                        raise RuntimeError(
                            "KVServe destination page order/ownership mismatch"
                        )
                    if (room, part) in self.inflight:
                        raise RuntimeError(
                            "KVServe room/part already has an inflight chunk"
                        )
                    job = ReceiveJob(header)
                    self.jobs[job.token] = job
                    self.requests[request_id] = job
                    self.inflight[(room, part)] = request_id
                    self.waiters.append(job)
                self._schedule()
                if job.cancelled:
                    return {**self._result(job), "token": job.token}
                if job.done.is_set():
                    return self._result(job)
                if job.slot_id is None:
                    return {"status": "queued", "token": job.token}
                return {
                    "status": "ready",
                    "token": job.token,
                    "slot_id": job.slot_id,
                    "regions": self.adapter.receive_regions(job.slot_id),
                }

        with self.lock:
            token = message.get("token")
            if token in self.retired:
                if operation not in ("commit", "status", "cancel"):
                    raise ValueError("Unknown KVServe operation")
                return self.retired[token][1]
            job = self.jobs.get(token)
            if job is None:
                raise RuntimeError("Unknown/stale KVServe receive token")
            if operation == "fail" or (
                operation == "cancel" and message.get("drained") is not True
            ):
                self._quarantine(
                    "Sender could not prove compressed transfer completion"
                )
                return self._result(job)
            if operation == "cancel":
                # Only sent before issuing a native write, or after it drains.
                job.cancelled = True
                job.error = "KVServe source request was cancelled"
                if not job.started:
                    job.done.set()
            elif operation == "commit":
                if job.slot_id is None:
                    raise RuntimeError("KVServe queued chunk cannot be committed")
                job.committed = True
                if job.cancelled or not self.is_active(job.header["room"]):
                    job.error = job.error or "KVServe destination request is inactive"
                    if not job.started:
                        job.done.set()
            elif operation != "status":
                raise ValueError("Unknown KVServe operation")
            if not job.done.is_set():
                return {"status": "pending"}
            result = self._result(job)
            if not self.poisoned:
                self._retire(job, result)
                self._schedule()
            return result

    def advance(self, room):
        """Decode on the scheduler thread, before it can recycle KV pages."""
        with self.lock:
            ready = [
                job
                for job in self.jobs.values()
                if job.header["room"] == room
                and job.committed
                and not job.started
                and not job.done.is_set()
            ]
        for job in ready:
            with self.lock:
                if self.poisoned or job.cancelled or not self.is_active(room):
                    job.error = self.poisoned or "KVServe request is inactive"
                    job.done.set()
                    continue
                job.started = True
            try:
                self.adapter.restore(
                    job.header["pages"],
                    job.header["part"],
                    job.header["valid_tokens"],
                    job.header["sizes"],
                    job.slot_id,
                )
                with self.lock:
                    if not job.cancelled and not self.poisoned:
                        self.completed[room][job.header["part"]] += len(
                            job.header["pages"]
                        )
            except Exception as error:
                logger.exception("KVServe decode/scatter failed")
                self._quarantine(f"KVServe decode/scatter failed: {error}")
                if getattr(error, "cache_write_unfenced", False):
                    raise
            finally:
                job.done.set()

    def start(self, host):
        self.thread = threading.Thread(
            target=self._serve, args=(host,), name="kvserve-receiver", daemon=True
        )
        self.thread.start()
        if not self._ready.wait(self.timeout):
            raise RuntimeError("KVServe receiver startup timed out")
        if self._startup_error:
            raise RuntimeError(
                "KVServe receiver startup failed"
            ) from self._startup_error

    def advertisement(self):
        return {
            "version": PROTOCOL_VERSION,
            "profile": self.adapter.profile,
            "endpoint": self.endpoint,
        }

    def _serve(self, host):
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 20)
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        socket.setsockopt(zmq.MAXMSGSIZE, 1024 * 1024)
        try:
            if ":" in host:
                socket.setsockopt(zmq.IPV6, 1)
                host = f"[{host}]"
            port = socket.bind_to_random_port(f"tcp://{host}")
            self.endpoint = f"tcp://{host}:{port}"
            self._ready.set()
            while not self._stop.is_set():
                self._expire_jobs()
                try:
                    message = socket.recv_json()
                except zmq.Again:
                    continue
                try:
                    response = self.handle(message)
                except Exception as error:
                    logger.exception("KVServe control request rejected")
                    response = {
                        "status": "error",
                        "error": str(error),
                        "drained": False,
                    }
                socket.send_json(response)
        except Exception as error:
            self._startup_error = error
            self._ready.set()
            logger.exception("KVServe control socket stopped")
            self._quarantine(f"KVServe control socket failed: {error}")
        finally:
            socket.close()
            context.term()

    def stop(self):
        """Stop control traffic, retaining all registered accelerator buffers."""
        self._stop.set()
        self.thread.join(timeout=self.timeout + 1)


@dataclass
class SendSession:
    nonce: str = field(default_factory=lambda: uuid.uuid4().hex)
    lock: threading.Lock = field(default_factory=threading.Lock)
    users: int = 0
    sequence: int = 0


class CompressionSender:
    def __init__(self, adapter, engine, timeout, is_active=lambda _: True):
        self.adapter = adapter
        self.engine = engine
        self.timeout = timeout
        self.is_active = is_active
        self.credits = SlotCredits(adapter.slot_count)
        self.lock = threading.Lock()
        self.sessions = {}

    @property
    def poisoned(self):
        return self.credits.poisoned

    @contextmanager
    def _session(self, key, room):
        with self.lock:
            state = self.sessions.setdefault(key, SendSession())
            state.users += 1
        acquired = False
        try:
            deadline = time.monotonic() + self.timeout
            while not acquired:
                if not self.is_active(room):
                    raise TransferCancelled("KVServe source request was aborted")
                if time.monotonic() >= deadline:
                    raise TimeoutError("KVServe request session queue timed out")
                acquired = state.lock.acquire(timeout=0.05)
            yield state
        finally:
            if acquired:
                state.lock.release()
            with self.lock:
                state.users -= 1
                if state.users == 0:
                    self.sessions.pop(key)

    @contextmanager
    def _socket(self, endpoint):
        import zmq

        socket = zmq.Context.instance().socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, max(1, int(self.timeout * 1000)))
        socket.setsockopt(zmq.SNDTIMEO, max(1, int(self.timeout * 1000)))
        if "[" in endpoint:
            socket.setsockopt(zmq.IPV6, 1)
        socket.connect(endpoint)
        try:
            yield socket
        finally:
            socket.close()

    @staticmethod
    def _rpc(socket, message):
        socket.send_json(message)
        return socket.recv_json()

    def _check_active(self, room):
        if not self.is_active(room):
            raise TransferCancelled("KVServe source request was aborted")
        if self.poisoned:
            raise RuntimeError(f"KVServe sender quarantined: {self.poisoned}")

    def _cancel(self, endpoint, token, uncertain):
        # A fresh REQ socket also works after the original socket timed out.
        with self._socket(endpoint) as socket:
            operation = {
                "op": "fail" if uncertain else "cancel",
                "token": token,
                "drained": not uncertain,
            }
            response = self._rpc(socket, operation)
            deadline = time.monotonic() + self.timeout
            while response.get("status") == "pending":
                if time.monotonic() >= deadline:
                    raise TimeoutError("KVServe cancellation drain timed out")
                time.sleep(0.005)
                response = self._rpc(socket, {"op": "status", "token": token})
            return response.get("drained") is True

    def send(self, peer, session, room, source_pages, destination_pages, valid_tokens):
        if (
            not isinstance(peer, dict)
            or peer.get("version") != PROTOCOL_VERSION
            or peer.get("profile") != self.adapter.profile
        ):
            raise RuntimeError("KVServe peer profile/TP/cache geometry mismatch")
        if len(source_pages) != len(destination_pages):
            raise ValueError("KVServe source/destination page counts differ")
        if not source_pages:
            return
        if not (
            (len(source_pages) - 1) * self.adapter.page_size
            < valid_tokens
            <= len(source_pages) * self.adapter.page_size
        ):
            raise ValueError("KVServe source valid-token count is out of bounds")
        key = (peer["endpoint"], session, room)
        with (
            self._session(key, room) as state,
            self._socket(peer["endpoint"]) as socket,
        ):
            for start in range(0, len(source_pages), self.adapter.chunk_pages):
                src = source_pages[start : start + self.adapter.chunk_pages]
                dst = destination_pages[start : start + self.adapter.chunk_pages]
                tokens = min(
                    len(src) * self.adapter.page_size,
                    valid_tokens - start * self.adapter.page_size,
                )
                for part in (0, 1):
                    self._send_part(
                        socket, peer, session, room, state, src, dst, part, tokens
                    )

    def _send_part(self, socket, peer, session, room, state, src, dst, part, tokens):
        begin = time.monotonic()
        slot_id = self.credits.acquire(self.timeout, lambda: self.is_active(room))
        slot_ready = time.monotonic()
        token = None
        uncertain = False
        reusable = True
        try:
            self._check_active(room)
            encoded = self.adapter.encode(src, part, tokens, slot_id)
            after_encode = time.monotonic()
            sizes = {name: encoded[name][1] for name in REGIONS}
            sizes["u8_bytes"] = encoded["u8_bytes"]
            state.sequence += 1
            header = {
                "version": PROTOCOL_VERSION,
                "profile": self.adapter.profile,
                "request_id": f"{state.nonce}:{state.sequence}",
                "room": room,
                "pages": dst,
                "part": part,
                "valid_tokens": tokens,
                "sizes": sizes,
            }
            deadline = time.monotonic() + self.timeout
            while True:
                self._check_active(room)
                response = self._rpc(socket, {"op": "prepare", "header": header})
                token = response.get("token", token)
                if response.get("status") == "ready":
                    break
                if response.get("status") == "error":
                    raise RemoteTransferError(response)
                if response.get("status") != "queued" or time.monotonic() >= deadline:
                    raise TimeoutError("KVServe destination slot queue timed out")
                time.sleep(0.005)
            regions = response["regions"]
            sources, destinations, lengths = [], [], []
            for name in REGIONS:
                if sizes[name]:
                    if sizes[name] > regions[name][1]:
                        raise RuntimeError("KVServe remote buffer capacity mismatch")
                    sources.append(encoded[name][0])
                    destinations.append(regions[name][0])
                    lengths.append(sizes[name])
            self._check_active(room)
            before_transfer = time.monotonic()
            uncertain = True
            rc = self.engine.batch_transfer_sync(
                session, sources, destinations, lengths
            )
            if rc != 0:
                raise RuntimeError(f"Mooncake compressed write failed: {rc}")
            uncertain = False
            after_transfer = time.monotonic()
            response = self._rpc(socket, {"op": "commit", "token": token})
            deadline = time.monotonic() + self.timeout
            while response.get("status") == "pending":
                self._check_active(room)
                if time.monotonic() >= deadline:
                    raise TimeoutError("KVServe decode acknowledgement timed out")
                time.sleep(0.005)
                response = self._rpc(socket, {"op": "status", "token": token})
            if response.get("status") != "done":
                raise RemoteTransferError(response)
            token = None
            logger.info(
                "kvserve_transfer room=%s slot=%d part=%d tokens=%d raw_bytes=%d "
                "wire_bytes=%d queue_ms=%.3f encode_ms=%.3f remote_wait_ms=%.3f "
                "transfer_ms=%.3f ack_ms=%.3f",
                room,
                slot_id,
                part,
                tokens,
                self.adapter.raw_bytes(part, tokens),
                sum(lengths),
                (slot_ready - begin) * 1000,
                (after_encode - slot_ready) * 1000,
                (before_transfer - after_encode) * 1000,
                (after_transfer - before_transfer) * 1000,
                (time.monotonic() - after_transfer) * 1000,
            )
        except Exception as error:
            if token is not None:
                try:
                    reusable = self._cancel(peer["endpoint"], token, uncertain)
                except Exception:
                    logger.exception("KVServe cancellation could not establish a drain")
                    reusable = False
            if uncertain or getattr(error, "source_read_unfenced", False):
                reusable = False
            if not reusable:
                self.credits.quarantine(error)
            raise
        finally:
            if reusable:
                self.credits.release(slot_id)
