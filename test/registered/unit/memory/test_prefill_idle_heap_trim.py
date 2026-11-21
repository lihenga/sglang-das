import os
import unittest
from unittest.mock import Mock, patch

import importlib.util
from pathlib import Path
import sys

SRT = Path(__file__).resolve().parents[4] / "python/sglang/srt"
spec = importlib.util.spec_from_file_location("prefill_idle_heap_trim", SRT / "managers/scheduler_components/prefill_idle_heap_trim.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)



MIB = 2**20


class PolicyTest(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.rss = 100
        self.trim = Mock(return_value=1)
        self.fully_idle = Mock(return_value=True)
        clock = patch.object(module.time, "monotonic", side_effect=lambda: self.now)
        rss = patch.object(module, "_anonymous_rss_bytes", side_effect=self.read_rss)
        self.addCleanup(clock.stop)
        self.addCleanup(rss.stop)
        clock.start()
        rss.start()
        self.policy = module.PrefillIdleHeapTrim(
            2, module.IdleTrimSettings(
                minimum_idle_seconds=3, cooldown_seconds=7, minimum_growth_mib=16,
                startup_grace_seconds=5, check_interval_seconds=1,
            ), self.trim,
        )

    def read_rss(self):
        return None if self.rss is None else self.rss * MIB

    def poll(self, now, rss):
        self.now, self.rss = now, rss
        self.policy.poll(self.fully_idle)

    def test_startup_grace_dwell_and_growth_are_required(self):
        self.poll(0, 100)
        self.poll(3, 140)
        self.poll(5, 115)
        self.trim.assert_not_called()
        self.poll(6, 116)
        self.trim.assert_called_once_with(0)

    def test_busy_resets_dwell_and_cooldown_spans_episodes(self):
        self.poll(0, 100)
        self.now = 5
        self.policy.mark_busy()
        self.poll(5, 140)
        self.poll(7, 140)
        self.trim.assert_not_called()
        self.poll(8, 140)
        self.policy.mark_busy()
        self.poll(9, 180)
        self.poll(14, 180)
        self.trim.assert_called_once()
        self.poll(15, 180)
        self.assertEqual(self.trim.call_count, 2)

    def test_transfer_or_cache_work_restarts_quiet_timer(self):
        self.poll(0, 100)
        self.fully_idle.return_value = False
        self.poll(6, 160)
        self.fully_idle.return_value = True
        self.poll(7, 160)
        self.poll(9, 160)
        self.trim.assert_not_called()
        self.poll(10, 160)
        self.trim.assert_called_once()

    def test_new_busy_state_between_rss_read_and_trim_blocks_call(self):
        self.poll(0, 100)
        self.fully_idle.side_effect = [True, False]
        self.poll(6, 150)
        self.trim.assert_not_called()

    def test_cooldown_allows_new_growth_in_same_long_idle_interval(self):
        self.poll(0, 100)

        def trim(_):
            self.rss -= 30
            return 1

        self.trim.side_effect = trim
        self.poll(6, 145)
        self.poll(12, 155)
        self.trim.assert_called_once()
        self.poll(13, 155)
        self.assertEqual(self.trim.call_count, 2)

    def test_missing_rss_defers_without_advancing_the_floor(self):
        self.poll(0, 100)
        self.poll(5, None)
        self.trim.assert_not_called()
        self.poll(6, 140)
        self.trim.assert_called_once()

    def test_failed_trim_is_rate_limited_and_does_not_repeat_at_same_rss(self):
        self.poll(0, 100)
        self.trim.side_effect = OSError("test failure")
        with self.assertLogs(module.logger, "WARNING"):
            self.poll(5, 150)
        self.poll(6, 180)
        self.poll(12, 150)
        self.trim.assert_called_once()


class SetupTest(unittest.TestCase):
    def test_disabled_skips_libc_lookup(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(module.ctypes, "CDLL") as load:
            self.assertIsNone(module.make_prefill_idle_heap_trim(0, None))
            load.assert_not_called()

    def test_blocking_idle_and_invalid_settings_disable_the_feature(self):
        with patch.dict(os.environ, {module._FLAG: "1"}, clear=True):
            with self.assertLogs(module.logger, "WARNING"):
                self.assertIsNone(module.make_prefill_idle_heap_trim(0, object()))
            with patch.dict(os.environ, {f"{module._FLAG}_MIN_IDLE_S": "0"}):
                with self.assertLogs(module.logger, "WARNING"):
                    self.assertIsNone(module.make_prefill_idle_heap_trim(0, None))

    def test_missing_libc_symbol_does_not_prevent_startup(self):
        with patch.dict(os.environ, {module._FLAG: "1"}, clear=True):
            with patch.object(module.ctypes, "CDLL", return_value=object()):
                with self.assertLogs(module.logger, "WARNING"):
                    self.assertIsNone(module.make_prefill_idle_heap_trim(0, None))


if __name__ == "__main__":
    unittest.main()
