"""A replay must not bypass asynchronous direct-linker restore readiness."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.runtime_context import get_context, get_exec, get_memory
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _method(file_name, class_name, method_name):
    # Isolate the real worker boundary from model/GPU construction.
    path = Path(__file__).parents[4] / "python/sglang/srt" / file_name
    tree = ast.parse(path.read_text(encoding="utf-8"))
    worker = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    method = next(
        n
        for n in worker.body
        if isinstance(n, ast.FunctionDef) and n.name == method_name
    )
    namespace = {"get_exec": get_exec, "get_memory": get_memory}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[method.name]


class TestMooncakeGraphRestoreBarrier(CustomTestCase):
    def test_selecting_a_consumer_does_not_block_an_eager_forward(self):
        for backend, external, index, waits in (
            ("full", True, 0, True),
            ("breakable", True, 1, True),
            ("tc_piecewise", True, 0, True),
            ("disabled", True, 0, False),
            ("full", False, 0, False),
            ("full", True, -1, False),
        ):
            with self.subTest(backend=backend, external=external, index=index):
                graph = CudaGraphConfig(prefill=PhaseConfig(backend=backend))
                with get_context().override_server_args(
                    enable_unified_cache_external_linker=external,
                    cuda_graph_config=graph,
                ):
                    counter = Mock(num_layers=8)
                    owner = SimpleNamespace(hicache_layer_transfer_counter=counter)
                    _method(
                        "managers/tp_worker.py", "TpModelWorker", "set_hicache_consumer"
                    )(owner, index)
                    counter.set_consumer.assert_called_once_with(index)
                    counter.wait_until.assert_not_called()

    def test_graph_replays_wait_for_the_last_restore_group(self):
        method = _method(
            "model_executor/runner/prefill_cuda_graph_runner.py",
            "PrefillCudaGraphRunner",
            "_wait_for_external_kv_restore",
        )
        for external in (False, True):
            with (
                self.subTest(external=external),
                get_context().override_server_args(
                    enable_unified_cache_external_linker=external
                ),
            ):
                counter = Mock(num_layers=8)
                owner = SimpleNamespace(
                    model_runner=SimpleNamespace(
                        token_to_kv_pool=SimpleNamespace(layer_transfer_counter=counter)
                    )
                )
                method(owner)
                if external:
                    counter.wait_until.assert_called_once_with(7)
                else:
                    counter.wait_until.assert_not_called()


if __name__ == "__main__":
    unittest.main()
