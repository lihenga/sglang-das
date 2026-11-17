"""Keep service fault tests sensitive to absent injection and absent remote reads."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.linker_load_failure_kit import LinkerLoadFailureMixin
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FailureHarness(LinkerLoadFailureMixin, unittest.TestCase):
    __test__ = False
    base_url = "http://unused"
    max_abort_probes = 4
    gsm8k_threshold = 0.93

    def __init__(self, outcomes):
        super().__init__()
        self.outcomes = iter(outcomes)

    def _generate_one(self, *args, **kwargs):
        return next(self.outcomes)

    def _flush(self):
        pass


class TestLinkerLoadFailureContract(CustomTestCase):
    def test_abort_must_be_followed_by_a_successful_request_and_live_health(self):
        harness = _FailureHarness(
            [
                ("served", {}),
                ("aborted", "external KV cache load failed"),
                ("served", {}),
                ("served", {}),
            ]
        )
        with patch(
            "requests.get", return_value=SimpleNamespace(status_code=200)
        ) as health:
            harness.test_load_failures_abort_and_the_engine_survives()
        self.assertEqual(health.call_args.args[0], "http://unused/health_generate")

    def test_missing_injection_cannot_pass_the_survival_probe(self):
        harness = _FailureHarness([("served", {})] * 4)
        with self.assertRaisesRegex(AssertionError, "No request was aborted"):
            harness.test_load_failures_abort_and_the_engine_survives()

    def test_an_unrelated_abort_is_not_evidence_of_a_failed_cache_read(self):
        harness = _FailureHarness([("served", {}), ("aborted", "unrelated failure")])
        with self.assertRaisesRegex(
            AssertionError, "some reason other than the linker"
        ):
            harness.test_load_failures_abort_and_the_engine_survives()

    def test_accuracy_without_remote_read_exposure_cannot_pass(self):
        harness = _FailureHarness([])
        harness._run_gsm8k = lambda: {"score": 1.0}
        harness._host_hit_tokens = lambda: 0
        with self.assertRaisesRegex(AssertionError, "No prefill token came"):
            harness.test_gsm8k_accuracy_under_linker_load_failure()


if __name__ == "__main__":
    unittest.main()
