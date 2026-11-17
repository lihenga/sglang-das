"""Exercise the scheduler -> match -> admission -> real PrefillAdder seam."""

import __future__

import ast
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler_components.mooncake_prefetch import (
    MooncakePrefetchSchedulerMixin,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    MatchResult,
    TreeLock,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.prefill_budget import PrefillBudget
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
    _SessionJob,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.runtime_context import get_context, get_schedule
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _prefill_method():
    path = Path(__file__).parents[4] / "python/sglang/srt/managers/scheduler.py"
    cls = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_get_new_batch_prefill_raw"
    )

    # Only the GPU batch assembly and stats boundary are replaced. Request
    # initialization, linker match/admission and PrefillAdder are production code.
    def batch(reqs, *args, **kwargs):
        return SimpleNamespace(reqs=reqs, prepare_for_extend=lambda: None)

    namespace = dict(
        PrefillAdder=PrefillAdder,
        AddReqResult=AddReqResult,
        DisaggregationMode=DisaggregationMode,
        TEST_RETRACT=False,
        ScheduleBatch=SimpleNamespace(init_new=batch),
        PrefillStats=SimpleNamespace(from_adder=lambda *args, **kwargs: None),
        set_time_batch=lambda *args: None,
        get_schedule=get_schedule,
    )
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[method.name]


class TestMooncakeLinkerAdmission(CustomTestCase):
    def setUp(self):
        super().setUp()
        scope = get_context().override_server_args(
            model_path="dummy", prefill_max_requests=None
        )
        scope.install()
        self.addCleanup(scope.restore)

    def _scheduler(self, *, hit=False, tokens=10000, request_slots=128):
        backend = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        backend.host_prefetch_enabled = False
        backend.host_prefetch_limit = 128
        backend.host_prefetch_worker_count = 1
        backend._init_session_runtime()
        self.addCleanup(backend.session_executor.shutdown, wait=True)
        backend.pool_group = SimpleNamespace(
            resolve_transfers=lambda transfers, **kwargs: transfers
        )
        lookups, matches = [], []

        def exists(keys, transfers):
            lookups.append(tuple(keys))
            return SimpleNamespace(
                restorable_prefix_pages=list(range(1, len(keys) + 1)) if hit else []
            )

        backend.stats = {"lookup": 0}
        backend.storage = SimpleNamespace(
            batch_exists_v2=exists,
            _batch_exist=lambda keys: [1] * len(keys),
            _get_hybrid_page_component_keys=lambda keys, _: (keys, 1),
            _tag_keys=lambda keys: list(keys),
        )
        allocator = SimpleNamespace(page_size=1, available_size=lambda: tokens)
        allocator.create_prefill_budget = lambda cache, **kwargs: PrefillBudget(
            allocator, cache, **kwargs
        )
        component = SimpleNamespace(
            build_external_linker_transfer=lambda phase, node, keys: PoolTransfer(
                name=PoolName.KV, keys=list(keys)
            )
        )
        cache = SimpleNamespace(
            page_size=1,
            tree_components={ComponentType.FULL},
            components={ComponentType.FULL: component},
            _components_tuple=(component,),
            tree_core=SimpleNamespace(is_eagle=False),
            buffer_pipeline=None,
            token_to_kv_pool_allocator=allocator,
            disable=False,
            supports_mamba=lambda: False,
            supports_prefix_sharing=lambda: False,
            swa_reprefill_tail_tokens=lambda: 0,
            evictable_size=lambda: 0,
            lock=lambda node: TreeLock(node, DecLockRefParams()),
            unlock=lambda lock: None,
            finish_storage_prefetch_admission=lambda *args, **kwargs: None,
            _all_reduce_attn_groups=lambda tensor, op: None,
            storage_prefetch_retries=None,
            ready_to_load_host_cache=lambda: -1,
        )
        wrapper = UnifiedCacheLinkerWrapper(cache, backend)

        def match(params):
            matches.append(params.req.rid)
            return wrapper.match(params.key, params.req, MatchResult(0, 0, 0, 0))

        cache.match_prefix = match
        cache.prepare_external_linker_loads = wrapper.prepare_load_admissions
        cache.discard_external_linker_load_admission = wrapper.discard_load_admission
        cache.cancel_waiting_queue_prefetch = wrapper.cancel_waiting_queue_prefetch
        owner = MooncakePrefetchSchedulerMixin()
        owner.tree_cache = cache
        owner.token_to_kv_pool_allocator = allocator
        owner.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        owner.enable_priority_preemption = owner.is_hybrid_swa = False
        owner.chunked_req = owner.min_free_slots_delayer = None
        owner.policy = SimpleNamespace(calc_priority=lambda *args, **kwargs: None)
        owner.processed_tokens_counter = 0
        owner.chunked_prefill_size = None
        owner.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=object(), prefill_aware_swa=False)
        )
        owner.page_size, owner.max_prefill_tokens, owner.max_prefill_bs = 1, 10000, 128
        owner.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        owner.max_running_requests = 128
        owner.is_mixed_chunk = owner.enable_lora = False
        owner.priority_scheduling_preemption_threshold = 0
        owner.dllm_config = None
        owner.req_to_token_pool = SimpleNamespace(available_size=lambda: request_slots)
        owner.get_num_allocatable_reqs = lambda *args, **kwargs: 128
        owner.enable_unified_cache_external_linker = True
        owner.enable_hicache_storage = owner.enable_lmcache = (
            owner.enable_hierarchical_cache
        ) = False
        owner.disaggregation_mode = DisaggregationMode.PREFILL
        owner.truncation_align_size = None
        owner.model_config, owner.spec_algorithm = None, None
        owner.enable_overlap = owner.enable_priority_scheduling = False
        owner.load_inquirer = SimpleNamespace(
            _get_num_pending_tokens=lambda **kwargs: 0
        )
        owner.waiting_queue = [
            Req(str(i), "", array("q", [1, 2, 3, 4]), SamplingParams(max_new_tokens=1))
            for i in range(65)
        ]
        running = SimpleNamespace(reqs=[], batch_is_full=False, is_empty=lambda: True)
        method = _prefill_method()

        def run():
            # The PD prefill loop clears this before each admission round.
            running.batch_is_full = False
            return method(owner, None, running)

        return owner, run, backend, wrapper, matches, lookups

    def test_full_request_pool_does_no_remote_match_and_preserves_unvisited_proofs(
        self,
    ):
        owner, run, backend, wrapper, matches, lookups = self._scheduler(
            hit=True, request_slots=0
        )
        proofs = {}
        for req in owner.waiting_queue:
            proof = _SessionJob()
            proof.set_result(True)
            key = ("refresh", backend.request_id(req.cache_request_handle))
            backend.session_jobs[key] = proof
            proofs[key] = proof
        for _ in range(2):
            self.assertIsNone(run()[0])
            self.assertEqual(matches, [])
            self.assertEqual(lookups, [])
            self.assertEqual(backend.session_jobs, proofs)
        # Give the adder space on the third round; each miss is matched once.
        owner.req_to_token_pool.available_size = lambda: 128
        backend.storage.batch_exists_v2 = lambda keys, transfers: SimpleNamespace(
            restorable_prefix_pages=[]
        )
        batch, _ = run()
        self.assertEqual(len(batch.reqs), 65)
        self.assertEqual(matches, [str(i) for i in range(65)])

    def test_skipped_first_64_requests_do_not_block_the_65th(self):
        owner, run, backend, wrapper, matches, lookups = self._scheduler()
        owner.enable_lora = True
        owner.lora_drainer = None
        owner.can_schedule_lora_req = lambda req, running: req.rid == "64"
        batch, _ = run()
        self.assertIsNotNone(batch)
        self.assertEqual([req.rid for req in batch.reqs], ["64"])
        self.assertEqual(matches, ["64"])
        self.assertEqual(len(owner.waiting_queue), 64)

    def test_kv_budget_rejection_only_retires_the_candidate_the_adder_reached(self):
        owner, run, backend, wrapper, matches, lookups = self._scheduler(
            hit=True, tokens=0
        )
        proofs = {}
        for req in owner.waiting_queue:
            proof = _SessionJob()
            proof.set_result(True)
            key = ("refresh", backend.request_id(req.cache_request_handle))
            proofs[key] = backend.session_jobs[key] = proof
        self.assertIsNone(run()[0])
        self.assertEqual(matches, ["0"])
        self.assertEqual(len(lookups), 1)
        first = (
            "refresh",
            backend.request_id(owner.waiting_queue[0].cache_request_handle),
        )
        self.assertEqual(
            backend.session_jobs,
            {key: proof for key, proof in proofs.items() if key != first},
        )
        self.assertFalse(wrapper._pending_session_cleanup)


if __name__ == "__main__":
    unittest.main()
