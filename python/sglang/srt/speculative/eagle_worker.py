"""Compatibility aliases for plugins written against the pre-V2 worker path."""

from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

EAGLEWorker = EAGLEWorkerV2

__all__ = ["EAGLEWorker"]
