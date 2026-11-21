"""Opt-in glibc heap maintenance for the normal disaggregated prefill loop."""

import ctypes
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time


_MIB = 2**20
_FLAG = "SGLANG_RC2_P_IDLE_HEAP_TRIM"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IdleTrimSettings:
    minimum_idle_seconds: int = 180
    cooldown_seconds: int = 900
    minimum_growth_mib: int = 32
    startup_grace_seconds: int = 600
    check_interval_seconds: int = 5

    @classmethod
    def from_environment(cls):
        defaults = cls()
        values = {
            "minimum_idle_seconds": int(os.getenv(
                f"{_FLAG}_MIN_IDLE_S", defaults.minimum_idle_seconds
            )),
            "cooldown_seconds": int(os.getenv(
                f"{_FLAG}_COOLDOWN_S", defaults.cooldown_seconds
            )),
            "minimum_growth_mib": int(os.getenv(
                f"{_FLAG}_MIN_GROWTH_MIB", defaults.minimum_growth_mib
            )),
        }
        if min(values.values()) <= 0:
            raise ValueError("idle dwell, cooldown and anonymous growth must be positive")
        return cls(**values)


def _anonymous_rss_bytes():
    try:
        with Path("/proc/self/status").open() as status:
            for line in status:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


class PrefillIdleHeapTrim:
    def __init__(self, rank, settings, trim):
        self.rank = rank
        self.settings = settings
        self._trim = trim
        self._ready_at = time.monotonic() + settings.startup_grace_seconds
        self._idle_since = None
        self._next_check = 0.0
        self._next_trim = 0.0
        self._rss_floor = None

    def mark_busy(self):
        self._idle_since = None
        self._next_check = 0.0

    def poll(self, is_fully_idle):
        now = time.monotonic()
        if now < self._next_check:
            return
        self._next_check = now + self.settings.check_interval_seconds
        if not is_fully_idle():
            self._idle_since = None
            return

        if self._idle_since is None:
            self._idle_since = now
        eligible_at = max(
            self._ready_at,
            self._idle_since + self.settings.minimum_idle_seconds,
            self._next_trim,
        )
        if self._rss_floor is not None and now < eligible_at:
            return
        before = _anonymous_rss_bytes()
        if before is None:
            return
        if self._rss_floor is None or before <= self._rss_floor:
            self._rss_floor = before
            return
        if before - self._rss_floor < self.settings.minimum_growth_mib * _MIB:
            return
        if not is_fully_idle():
            self.mark_busy()
            return

        self._next_trim = now + self.settings.cooldown_seconds
        try:
            result = self._trim(0)
        except (OSError, ValueError) as error:
            self._rss_floor = before
            logger.warning("P idle heap trim failed on TP%d: %s", self.rank, error)
            return
        after = _anonymous_rss_bytes()
        self._rss_floor = before if after is None else after
        duration_ms = (time.monotonic() - now) * 1000
        reclaimed_mib = None if after is None else round((before - after) / _MIB, 2)
        logger.info(
            "P idle heap trim TP%d result=%d anon_before_mib=%.2f "
            "anon_after_mib=%s reclaimed_mib=%s duration_ms=%.2f idle_s=%.1f",
            self.rank, result, before / _MIB,
            None if after is None else round(after / _MIB, 2),
            reclaimed_mib, duration_ms, now - self._idle_since,
        )


def make_prefill_idle_heap_trim(rank, idle_sleeper):
    if os.getenv(_FLAG) != "1":
        return None
    if idle_sleeper is not None:
        logger.warning("P idle heap trim disabled while idle receive can block on TP%d", rank)
        return None
    try:
        settings = IdleTrimSettings.from_environment()
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
    except (AttributeError, OSError, ValueError) as error:
        logger.warning("P idle heap trim unavailable on TP%d: %s", rank, error)
        return None
    logger.info("P idle heap trim enabled on TP%d with %s", rank, settings)
    return PrefillIdleHeapTrim(rank, settings, trim)
