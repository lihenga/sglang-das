"""Nonblocking page-address readback for decode transfer admission.

Only immutable page-index outputs are copied. Metadata remains unpublished
until the event completes; the scheduler can continue preparing existing work.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

_RETIRED_READBACKS = []


def _reap_retired():
    _RETIRED_READBACKS[:] = [b for b in _RETIRED_READBACKS if not b.ready()]


@dataclass(frozen=True)
class PendingArray:
    batch: "MetadataReadback"
    host: torch.Tensor
    numpy_dtype: Optional[Any] = None

    def __len__(self):
        return self.host.numel()

    def astype(self, dtype):
        return PendingArray(self.batch, self.host, np.dtype(dtype))

    def resolve(self):
        if not self.batch.ready():
            raise RuntimeError("Page metadata read before its D2H event completed")
        value = self.host.numpy()
        return value if self.numpy_dtype is None else value.astype(self.numpy_dtype)


def resolve_metadata(value):
    if isinstance(value, PendingArray):
        return value.resolve()
    if isinstance(value, tuple):
        return tuple(resolve_metadata(x) for x in value)
    if isinstance(value, list):
        return [resolve_metadata(x) for x in value]
    if isinstance(value, dict):
        return {key: resolve_metadata(x) for key, x in value.items()}
    return value


class MetadataReadback:
    def __init__(self, stream):
        _reap_retired()
        self.stream = stream
        self.copies = []
        self.event = None

    def page_indices(self, kv_indices, page_size):
        if self.event is not None:
            raise RuntimeError("Cannot append to an issued metadata readback")
        pages = kv_indices[::page_size] // page_size
        host = torch.empty(pages.shape, dtype=pages.dtype, pin_memory=True)
        self.copies.append((pages, host))
        return PendingArray(self, host)

    def issue(self):
        if self.event is not None:
            raise RuntimeError("Metadata readback was already issued")
        # The event inserted by wait_stream covers allocation and page-index
        # generation on the producer stream, including its existing WAR fence.
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for pages, host in self.copies:
                host.copy_(pages, non_blocking=True)
                pages.record_stream(self.stream)
            self.event = torch.cuda.Event()
            self.event.record()

    def ready(self):
        return self.event is not None and self.event.query()

    def retire(self):
        # A cancelled request must not free a pinned destination while a copy
        # can still write it. Reap these owners without a scheduler-side wait.
        if self.event is not None and not self.ready():
            _RETIRED_READBACKS.append(self)


class PendingMetadataSend:
    def __init__(self, readback, args, kwargs, after_send=None):
        self.readback = readback
        self.args = args
        self.kwargs = kwargs
        self.after_send = after_send
        self.sent = False
        self.cancelled = False
        self.error = None

    def try_send(self, receiver):
        if self.cancelled:
            return False
        if self.error is not None:
            raise self.error
        if self.sent:
            return True
        if not self.readback.ready():
            return False
        # Mark before invoking an external send: an exception must not cause
        # the next poll to repeat a send whose remote outcome is unknown.
        self.sent = True
        try:
            receiver.send_metadata(
                *resolve_metadata(self.args), **resolve_metadata(self.kwargs)
            )
            if self.after_send is not None:
                self.after_send()
        except Exception as exc:
            self.error = exc
            raise
        return True

    def cancel(self):
        if not self.cancelled:
            self.cancelled = True
            self.readback.retire()


class MetadataReadyGatedReceiver:
    def __init__(self, decode_req, receiver):
        self.decode_req = decode_req
        self.receiver = receiver

    def __getattr__(self, name):
        return getattr(self.receiver, name)

    def poll(self):
        from sglang.srt.disaggregation.base import KVPoll

        pending = self.decode_req.pending_metadata_send
        status = self.receiver.poll()
        if pending is None:
            return status
        if status == KVPoll.Failed:
            pending.cancel()
            self.decode_req.pending_metadata_send = None
            return status
        if not pending.try_send(self.decode_req.kv_receiver):
            return KVPoll.WaitingForInput
        self.decode_req.pending_metadata_send = None
        return self.receiver.poll()


def cancel_pending_metadata(decode_req):
    pending = decode_req.pending_metadata_send
    if pending is not None:
        pending.cancel()
        decode_req.pending_metadata_send = None
