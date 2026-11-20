import logging
import threading
import time
from typing import Any

import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


UNKNOWN = "unknown"
AVAILABLE = "available"
UNAVAILABLE = "unavailable"


_HEALTH_CONFIG_SOURCES = {
    "probe_interval_s": (
        "mooncake_dfs_health_probe_interval_s",
        "SGLANG_MOONCAKE_DFS_HEALTH_PROBE_INTERVAL_S",
        5,
    ),
    "failure_threshold": (
        "mooncake_dfs_health_failure_threshold",
        "SGLANG_MOONCAKE_DFS_HEALTH_FAILURE_THRESHOLD",
        2,
    ),
    "success_threshold": (
        "mooncake_dfs_health_success_threshold",
        "SGLANG_MOONCAKE_DFS_HEALTH_SUCCESS_THRESHOLD",
        2,
    ),
    "probe_timeout_s": (
        "mooncake_dfs_health_probe_timeout_s",
        "SGLANG_MOONCAKE_DFS_HEALTH_PROBE_TIMEOUT_S",
        25,
    ),
}


def resolve_dfs_health_config(
    config: dict[str, int | None] | None = None, server_args: Any = None
) -> dict[str, int]:
    """Resolve DFS health settings consistently for every storage entry point.

    Precedence is explicit config, explicit ServerArgs/CLI value, environment,
    then the built-in default. None means that a source did not specify a value.
    """
    config = config or {}
    resolved: dict[str, int] = {}
    for key, (arg_name, env_name, default) in _HEALTH_CONFIG_SOURCES.items():
        value = config.get(key)
        if value is None and server_args is not None:
            value = getattr(server_args, arg_name, None)
        if value is None:
            raw = os.environ.get(env_name)
            value = default if raw is None else int(raw)
        value = int(value)
        if value <= 0:
            raise ValueError(f"Mooncake DFS health {key} must be positive, got {value}")
        resolved[key] = value
    return resolved


@dataclass(frozen=True)
class DfsHealthSnapshot:
    read_state: str
    write_state: str
    generation: int
    read_failures: int
    write_failures: int
    read_successes: int
    write_successes: int
    read_error: int
    write_error: int


@dataclass
class _NativeProbeCall:
    capability: str
    done: threading.Event
    started_at: float
    thread: threading.Thread | None = None
    result: int | Exception | None = None
    timed_out: bool = False


class MooncakeDfsHealthManager:
    """Owns rank-local DFS health and the native client's effective policy."""

    def __init__(
        self,
        store: Any,
        config: dict[str, int] | None = None,
        *,
        defer_policy_application: bool = False,
    ):
        self._store = store
        self._defer_policy_application = defer_policy_application
        config = resolve_dfs_health_config(config)
        self._interval_s = config["probe_interval_s"]
        self._failure_threshold = config["failure_threshold"]
        self._success_threshold = config["success_threshold"]
        self._probe_timeout_s = config["probe_timeout_s"]
        self._lock = threading.Lock()
        self._policy_lock = threading.Lock()
        self._policy_condition = threading.Condition(self._policy_lock)
        self._policy_apply_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._policy_thread: threading.Thread | None = None
        self._probe_lock = threading.Lock()
        self._probe_calls: dict[str, _NativeProbeCall | None] = {
            "read": None,
            "write": None,
        }
        self._read_state = UNKNOWN
        self._write_state = UNKNOWN
        self._generation = 0
        self._data_timeout_generation = 0
        self._read_failures = 0
        self._write_failures = 0
        self._read_successes = 0
        self._write_successes = 0
        self._read_error = 0
        self._write_error = 0
        self._effective_policy = (False, False)
        self._desired_policy = self._effective_policy
        self._timeout_policy = (True, True)
        self._policy_dirty = False
        self._policy_warning_at = float("-inf")

        for method in (
            "probe_dfs_read",
            "probe_dfs_write",
            "set_dfs_access_policy",
        ):
            if not hasattr(store, method):
                raise RuntimeError(
                    f"The installed Mooncake package lacks {method}(); "
                    "rebuild/install Mooncake with DFS health support."
                )

        # UNKNOWN is fail-closed. This call happens before SGLang warmup.
        self._apply_policy(False, False)

    def start(self) -> None:
        with self._probe_lock:
            if self._stop.is_set():
                raise RuntimeError("Mooncake DFS health manager is closing or closed")
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run, name="mooncake-dfs-health", daemon=True
            )
            self._thread.start()

    def close(self) -> None:
        """Stop scheduling; refuse owner release until all native work drains.

        Callers must retain this manager and its client on failure, refuse
        re-attachment, and retry close after recovery (or restart the process).
        Stopping health reporting does not confirm a native DFS disable.
        """
        with self._probe_lock:
            self._stop.set()
            threads = [self._thread]
            threads.extend(
                call.thread for call in self._probe_calls.values() if call is not None
            )
        with self._policy_condition:
            self._policy_condition.notify_all()
            threads.append(self._policy_thread)
        with self._lock:
            self._generation += int(self._read_state != UNKNOWN) + int(
                self._write_state != UNKNOWN
            )
            self._read_state = self._write_state = UNKNOWN

        deadline = time.monotonic() + 1.0
        for thread in threads:
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
        pending = [
            thread.name
            for thread in threads
            if thread is not None and thread.is_alive()
        ]
        # Synchronous policy callers need an owner too, even without a worker.
        if self._policy_apply_lock.acquire(blocking=False):
            self._policy_apply_lock.release()
        else:
            pending.append("native policy application")
        if pending:
            raise RuntimeError(
                "Mooncake DFS health close incomplete; retain the client and refuse "
                "detach/reattach until native operations drain: " + ", ".join(pending)
            )

    def snapshot(self) -> DfsHealthSnapshot:
        with self._lock:
            return DfsHealthSnapshot(
                read_state=self._read_state,
                write_state=self._write_state,
                generation=self._generation,
                read_failures=self._read_failures,
                write_failures=self._write_failures,
                read_successes=self._read_successes,
                write_successes=self._write_successes,
                read_error=self._read_error,
                write_error=self._write_error,
            )

    def local_policy(self) -> tuple[bool, bool]:
        """Return this rank's fail-closed candidate; DFS reads require writes."""
        with self._lock:
            if self._stop.is_set():
                return False, False
            allow_write = self._write_state == AVAILABLE
            return (
                self._read_state == AVAILABLE and allow_write,
                allow_write,
            )

    def apply_effective_policy(self, allow_read: bool, allow_write: bool) -> bool:
        """Publish a coordinated policy; True means its native update is confirmed.

        Deferred mode keeps one worker and one latest target, so neither a slow
        native call nor its retries run on the scheduler. Local probes never
        publish a policy in this mode.
        """
        allow_write = bool(allow_write)
        policy = (bool(allow_read) and allow_write, allow_write)
        with self._policy_condition:
            if self._stop.is_set():
                return False
            # A collective/probe result obtained before a business-I/O timeout
            # cannot reopen the failed capability until fresh probes recover it.
            policy = (
                policy[0] and self._timeout_policy[0],
                policy[1] and self._timeout_policy[1],
            )
            self._desired_policy = policy
            if policy == self._effective_policy and not self._policy_dirty:
                return True
            # A pending call can have applied even without a successful reply.
            self._policy_dirty = True
            if self._defer_policy_application:
                if self._policy_thread is None:
                    self._policy_thread = threading.Thread(
                        target=self._run_policy,
                        name="mooncake-dfs-policy",
                        daemon=True,
                    )
                    self._policy_thread.start()
                self._policy_condition.notify()
                return False
        self._apply_pending_policy()
        with self._policy_lock:
            return policy == self._effective_policy and not self._policy_dirty

    def _apply_pending_policy(self) -> bool:
        # Serialize native calls, but never hold the scheduler-facing state lock
        # across an RPC. A stuck call occupies this one worker until it returns.
        with self._policy_apply_lock:
            with self._policy_lock:
                if self._stop.is_set() or not self._policy_dirty:
                    return True
                policy = self._desired_policy
            try:
                self._apply_policy(*policy)
            except Exception:
                now = time.monotonic()
                with self._policy_lock:
                    warn = now - self._policy_warning_at >= self._interval_s
                    if warn:
                        self._policy_warning_at = now
                if warn:
                    logger.warning(
                        "Mooncake DFS policy application failed; native policy "
                        "is unknown and will be retried",
                        exc_info=True,
                    )
                return False
            with self._policy_condition:
                old_policy = self._effective_policy
                self._effective_policy = policy
                self._policy_dirty = self._desired_policy != policy
                self._policy_condition.notify_all()
            if old_policy != policy:
                logger.warning(
                    "Mooncake DFS effective policy transition: "
                    "old_read=%s old_write=%s new_read=%s new_write=%s",
                    old_policy[0],
                    old_policy[1],
                    policy[0],
                    policy[1],
                )
        return True

    def _run_policy(self) -> None:
        while not self._stop.is_set():
            with self._policy_condition:
                self._policy_condition.wait_for(
                    lambda: self._stop.is_set() or self._policy_dirty
                )
                if self._stop.is_set():
                    return
            if not self._apply_pending_policy():
                # Scheduler polls cannot accelerate retries of a failed RPC.
                self._stop.wait(self._interval_s)

    def probe_once(self) -> DfsHealthSnapshot:
        with self._lock:
            timeout_generation = self._data_timeout_generation
        # Launch both operations before waiting for either one. Each operation has
        # its own deadline and in-flight slot, so a stuck read cannot delay or
        # change the write result (and vice versa).
        read_call = self._start_native_probe("read", "probe_dfs_read")
        write_call = self._start_native_probe("write", "probe_dfs_write")
        read_error = self._collect_native_probe(read_call)
        write_error = self._collect_native_probe(write_call)
        read_ok = read_error == 0
        write_ok = write_error == 0

        with self._lock:
            if self._stop.is_set():
                return self.snapshot_unlocked()
            if timeout_generation != self._data_timeout_generation:
                return self.snapshot_unlocked()
            old_read = self._read_state
            old_write = self._write_state
            self._read_state, self._read_successes, self._read_failures = (
                self._next_state(
                    self._read_state,
                    read_ok,
                    self._read_successes,
                    self._read_failures,
                )
            )
            self._write_state, self._write_successes, self._write_failures = (
                self._next_state(
                    self._write_state,
                    write_ok,
                    self._write_successes,
                    self._write_failures,
                )
            )
            self._read_error = read_error
            self._write_error = write_error
            read_changed = old_read != self._read_state
            write_changed = old_write != self._write_state
            self._generation += int(read_changed) + int(write_changed)
            allow_write = self._write_state == AVAILABLE
            allow_read = self._read_state == AVAILABLE and allow_write
            snapshot = self.snapshot_unlocked()
            with self._policy_condition:
                self._timeout_policy = (allow_read, allow_write)

        if not self._defer_policy_application:
            self.apply_effective_policy(allow_read, allow_write)
        if read_changed:
            logger.warning(
                "Mooncake DFS local health transition: capability=read "
                "old_state=%s new_state=%s error=%s generation=%s",
                old_read,
                snapshot.read_state,
                read_error,
                snapshot.generation,
            )
        if write_changed:
            logger.warning(
                "Mooncake DFS local health transition: capability=write "
                "old_state=%s new_state=%s error=%s generation=%s",
                old_write,
                snapshot.write_state,
                write_error,
                snapshot.generation,
            )
        return snapshot

    def report_operation_timeout(self, capability: str, error: int) -> None:
        """Trip one DFS capability immediately after a real data-I/O timeout."""
        if capability not in ("read", "write"):
            raise ValueError(f"Unknown DFS capability: {capability}")
        with self._lock:
            if self._stop.is_set():
                return
            if capability == "read":
                changed = self._read_state != UNAVAILABLE
                self._read_state = UNAVAILABLE
                self._read_successes = 0
                self._read_failures = max(self._read_failures, self._failure_threshold)
                self._read_error = int(error)
            else:
                changed = self._write_state != UNAVAILABLE
                self._write_state = UNAVAILABLE
                self._write_successes = 0
                self._write_failures = max(
                    self._write_failures, self._failure_threshold
                )
                self._write_error = int(error)
            self._generation += int(changed)
            self._data_timeout_generation += 1
            allow_write = self._write_state == AVAILABLE
            allow_read = self._read_state == AVAILABLE and allow_write
            # Keep native policy calls serialized on the existing worker. A
            # timeout may only restrict confirmed AND pending policy, so a local
            # report never reopens a capability disabled by rank consensus.
            with self._policy_condition:
                self._timeout_policy = (allow_read, allow_write)
                self._desired_policy = (
                    self._desired_policy[0]
                    and self._effective_policy[0]
                    and allow_read,
                    self._desired_policy[1]
                    and self._effective_policy[1]
                    and allow_write,
                )
                self._policy_dirty = True
                if self._policy_thread is None:
                    self._policy_thread = threading.Thread(
                        target=self._run_policy, name="mooncake-dfs-policy", daemon=True
                    )
                    self._policy_thread.start()
                self._policy_condition.notify()
        logger.error(
            "Mooncake DFS data operation timed out; capability=%s error=%s",
            capability,
            error,
        )

    def _start_native_probe(
        self, capability: str, method_name: str
    ) -> _NativeProbeCall:
        with self._probe_lock:
            if self._stop.is_set():
                raise RuntimeError("Mooncake DFS health manager is closing or closed")
            existing = self._probe_calls[capability]
            if existing is not None:
                return existing
            call = _NativeProbeCall(
                capability=capability,
                done=threading.Event(),
                started_at=time.monotonic(),
            )

            def invoke() -> None:
                try:
                    call.result = int(getattr(self._store, method_name)())
                except Exception as exc:
                    call.result = exc
                finally:
                    call.done.set()
                    # A timed-out result is stale: release its in-flight slot,
                    # but never let a later probe cycle count it as success.
                    with self._probe_lock:
                        if (
                            call.timed_out
                            and self._probe_calls[call.capability] is call
                        ):
                            self._probe_calls[call.capability] = None

            call.thread = threading.Thread(
                target=invoke,
                name=f"mooncake-dfs-{capability}-probe",
                daemon=True,
            )
            self._probe_calls[capability] = call
            call.thread.start()
            return call

    def _collect_native_probe(self, call: _NativeProbeCall) -> int:
        remaining = max(
            0.0,
            self._probe_timeout_s - (time.monotonic() - call.started_at),
        )
        if not call.done.wait(remaining):
            with self._probe_lock:
                call.timed_out = True
                # The native call may have finished between wait() timing out
                # and acquiring the lock. Clear it here in that case; otherwise
                # invoke() will clear it when the call eventually returns.
                if call.done.is_set() and self._probe_calls[call.capability] is call:
                    self._probe_calls[call.capability] = None
            logger.warning(
                "Mooncake native DFS %s probe timed out after %ss",
                call.capability,
                self._probe_timeout_s,
            )
            return -1

        with self._probe_lock:
            result = call.result
            timed_out = call.timed_out
            if self._probe_calls[call.capability] is call:
                self._probe_calls[call.capability] = None
        if timed_out:
            return -1
        if isinstance(result, Exception):
            logger.error(
                "Mooncake native DFS %s probe raised",
                call.capability,
                exc_info=(type(result), result, result.__traceback__),
            )
            return -1
        if result is None:
            logger.error(
                "Mooncake native DFS %s probe returned no result",
                call.capability,
            )
            return -1
        return int(result)

    def snapshot_unlocked(self) -> DfsHealthSnapshot:
        return DfsHealthSnapshot(
            self._read_state,
            self._write_state,
            self._generation,
            self._read_failures,
            self._write_failures,
            self._read_successes,
            self._write_successes,
            self._read_error,
            self._write_error,
        )

    def _next_state(
        self, state: str, success: bool, successes: int, failures: int
    ) -> tuple[str, int, int]:
        if success:
            successes += 1
            failures = 0
            if successes >= self._success_threshold:
                state = AVAILABLE
        else:
            failures += 1
            successes = 0
            if failures >= self._failure_threshold:
                state = UNAVAILABLE
        return state, successes, failures

    def _apply_policy(self, allow_read: bool, allow_write: bool) -> None:
        rc = int(self._store.set_dfs_access_policy(allow_read, allow_write))
        if rc != 0:
            raise RuntimeError(
                "Mooncake set_dfs_access_policy failed: "
                f"read={allow_read}, write={allow_write}, error={rc}"
            )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.probe_once()
            except Exception:
                if self._stop.is_set():
                    return
                logger.exception("Mooncake DFS health probe failed")
            if self._stop.wait(self._interval_s):
                break
