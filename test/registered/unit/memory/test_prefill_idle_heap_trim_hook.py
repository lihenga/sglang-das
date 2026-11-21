"""Execute the patched normal prefill loop without starting Torch or SGLang."""

import argparse
import ast
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


LOOP_NAME = "event_loop_normal_disagg_prefill"
POLICY_MODULE = "sglang.srt.managers.scheduler_components.prefill_idle_heap_trim"


class LoopComplete(Exception):
    """Raised by the fake receiver after its finite sequence of iterations."""


def locate_loop(module):
    owners = [module] + [node for node in module.body if isinstance(node, ast.ClassDef)]
    matches = [
        (owner, node)
        for owner in owners
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == LOOP_NAME
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one normal prefill loop, found {len(matches)}")
    return matches[0]


def load_loop(path):
    source = ast.parse(path.read_text(), filename=str(path))
    _, loop = locate_loop(source)
    loop = deepcopy(loop)
    loop.decorator_list = []
    compiled = compile(ast.Module(body=[loop], type_ignores=[]), str(path), "exec")
    globals_ = {"Scheduler": object, "DAS_PREFETCH_TRACE_ENABLED": False}
    exec(compiled, globals_)
    return globals_[LOOP_NAME]


def without_loop(path):
    source = ast.parse(path.read_text(), filename=str(path))
    owner, loop = locate_loop(source)
    owner.body.remove(loop)
    return ast.dump(source)


class DisabledReceiveList(list):
    def __bool__(self):
        raise AssertionError("The disabled trim path inspected received work")


@dataclass(frozen=True)
class Iteration:
    receive: bool = False
    batch: bool = False
    paused: bool = False
    inflight_after: bool = False


class FakeScheduler:
    def __init__(self, iterations, enabled):
        self.iterations = iter(iterations)
        self.trace = []
        self.factory_args = []
        self.enabled = enabled
        self.iteration = -1
        self.last_batch_value = object()
        self.ps = SimpleNamespace(tp_rank=3)
        self.idle_sleeper = object()
        self._forward_ingress_enabled = False
        self._mooncake_final_poll_timeout_ms = 0
        self.request_receiver = SimpleNamespace(recv_requests=self.recv_requests)
        self.disagg_prefill_bootstrap_queue = SimpleNamespace(
            pop_bootstrapped=self.pop_bootstrapped
        )
        self.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=self.prepare_for_forward
        )
        self.waiting_queue = []
        self.chunked_req = None
        self.running_batch = object()
        self.enable_staging = True
        self.inflight = True

    def record(self, event, *details):
        self.trace.append((self.iteration, event, *details))

    def make_policy(self, rank, sleeper):
        self.factory_args.append((rank, sleeper))
        if not self.enabled:
            return None
        return SimpleNamespace(mark_busy=self.mark_busy, poll=self.poll)

    def mark_busy(self):
        self.record("mark_busy")

    def poll(self, is_fully_idle):
        if is_fully_idle != self.is_fully_idle:
            raise AssertionError("The policy should receive the scheduler idle predicate")
        self.record("poll", is_fully_idle())

    def is_fully_idle(self):
        return not self.inflight and self.last_batch is None

    def recv_requests(self):
        current = next(self.iterations, None)
        if current is None:
            raise LoopComplete
        self.current = current
        self.iteration += 1
        self._engine_paused = current.paused
        self.record("recv")
        receive_list = list if self.enabled else DisabledReceiveList
        return receive_list(["input"] if current.receive else [])

    def process_input_requests(self, requests):
        self.record("input", len(requests))
        requests.clear()

    def pop_bootstrapped(self):
        self.record("bootstrap")
        return []

    def get_next_disagg_prefill_batch_to_run(self, *, running_batch, last_batch):
        if running_batch is not self.running_batch or last_batch is not self.last_batch:
            raise AssertionError("The scheduler must pass its current batch state")
        self.record("plan")
        return SimpleNamespace(
            running_batch=self.running_batch,
            batch_to_run="batch" if self.current.batch else None,
        )

    def prepare_for_forward(self, batch, *, chunked_req):
        self.record("prepare")
        return batch

    def maybe_prefetch_staging_for_batch(self, batch):
        self.record("staging")

    def run_batch(self, batch):
        self.record("run")
        return "result"

    def process_batch_result(self, batch, result):
        self.record("result")

    def on_idle(self):
        self.record("idle")

    def process_disagg_prefill_inflight_queue(self):
        self.record("inflight")
        self.inflight = self.current.inflight_after

    @property
    def last_batch(self):
        return self.last_batch_value

    @last_batch.setter
    def last_batch(self, batch):
        self.record("last_batch", batch)
        self.last_batch_value = batch


class PrefillHookTest(unittest.TestCase):
    source = Path(__file__).resolve().parents[4] / "python/sglang/srt/disaggregation/prefill.py"
    baseline = None

    def run_loop(self, iterations, *, enabled, source=None):
        scheduler = FakeScheduler(iterations, enabled)
        fake_policy_module = ModuleType(POLICY_MODULE)
        fake_policy_module.make_prefill_idle_heap_trim = scheduler.make_policy
        with patch.dict(sys.modules, {POLICY_MODULE: fake_policy_module}):
            loop = load_loop(source or self.source)
            with self.assertRaises(LoopComplete):
                loop(scheduler)
        return scheduler

    def assert_factory_once(self, scheduler):
        self.assertEqual(scheduler.factory_args, [(3, scheduler.idle_sleeper)])

    def test_disabled_preserves_normal_scheduler_order(self):
        plan = [
            Iteration(),
            Iteration(receive=True, batch=True),
            Iteration(paused=True),
        ]
        scheduler = self.run_loop(plan, enabled=False)
        self.assert_factory_once(scheduler)
        idle = ["recv", "input", "bootstrap", "plan", "prepare", "idle", "inflight", "last_batch"]
        work = ["recv", "input", "bootstrap", "plan", "prepare", "staging", "run", "result",
                "inflight", "last_batch"]
        pause = ["recv", "input"]
        self.assertEqual(
            [(i, event) for i, event, *_ in scheduler.trace],
            [(i, event) for i, events in enumerate((idle, work, pause)) for event in events],
        )
        if self.baseline is not None:
            original = self.run_loop(plan, enabled=False, source=self.baseline)
            self.assertEqual(scheduler.trace, original.trace)
            self.assertEqual(original.factory_args, [])

    def test_work_and_pause_mark_busy_without_polling(self):
        scheduler = self.run_loop(
            [
                Iteration(receive=True),
                Iteration(batch=True),
                Iteration(paused=True),
                Iteration(receive=True, batch=True),
            ],
            enabled=True,
        )
        self.assert_factory_once(scheduler)
        hooks = [(i, event) for i, event, *_ in scheduler.trace if event in {"mark_busy", "poll"}]
        self.assertEqual({i for i, _ in hooks}, {0, 1, 2, 3})
        self.assertTrue(all(event == "mark_busy" for _, event in hooks))
        for i in (0, 1, 3):
            events = [event for iteration, event, *_ in scheduler.trace if iteration == i]
            self.assertLess(events.index("last_batch"), events.index("mark_busy"))
        paused_events = [event for i, event, *_ in scheduler.trace if i == 2]
        self.assertEqual(paused_events, ["recv", "input", "mark_busy"])

    def test_idle_poll_follows_inflight_processing_and_current_batch_update(self):
        scheduler = self.run_loop(
            [Iteration(inflight_after=True), Iteration()],
            enabled=True,
        )
        self.assert_factory_once(scheduler)
        for i, can_trim in enumerate((False, True)):
            events = [event for iteration, *event in scheduler.trace if iteration == i]
            self.assertEqual(events[-3:], [["inflight"], ["last_batch", None], ["poll", can_trim]])
        self.assertFalse(any(event == "mark_busy" for _, event, *_ in scheduler.trace))

    def test_patch_scope_is_only_the_normal_prefill_loop(self):
        if self.baseline is None:
            self.skipTest("Pass --baseline to compare the AST outside the normal prefill loop")
        self.assertEqual(without_loop(self.source), without_loop(self.baseline))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default=PrefillHookTest.source, type=Path, help="Path to patched sglang prefill.py")
    parser.add_argument("--baseline", type=Path, help="Path to unmodified sglang prefill.py")
    args = parser.parse_args()
    PrefillHookTest.source = args.source
    PrefillHookTest.baseline = args.baseline
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(PrefillHookTest)
    )
    sys.exit(0 if result.wasSuccessful() else 1)
