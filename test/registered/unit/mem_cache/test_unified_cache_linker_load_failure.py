from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
    ReadPlanLoadCounter,
)
from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import UMBPDirectLinker
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import LayerWiseLoadCounter


def test_failed_counter_drains_later_layers_before_releasing_slots():
    counter = LayerWiseLoadCounter(2)
    index = counter.update_producer()
    counter.futures[index][0].set_exception(RuntimeError("failed first layer"))
    started = Event()

    def finish():
        started.set()
        return counter.finish(index)

    with ThreadPoolExecutor(1) as pool:
        result = pool.submit(finish)
        assert started.wait(5)
        assert not result.done()
        counter.complete(index, 1)
        assert result.result(timeout=5) is False
    assert index not in counter.futures


def test_counter_failure_does_not_retain_the_callers_traceback():
    counter = LayerWiseLoadCounter(1)
    index = counter.update_producer()
    try:
        raise RuntimeError("bad KV")
    except RuntimeError as error:
        counter.fail(index, error)
        failure = counter.futures[index][0].exception()
        assert failure is not error
        assert failure.__traceback__ is None
        assert error.__traceback__ is not None
    assert counter.finish(index) is False


def test_read_plan_failure_survives_until_finish():
    counter = ReadPlanLoadCounter(2)
    index = counter.update_producer()
    counter.set_consumer(index)
    plan = Mock()
    plan.wait.side_effect = [RuntimeError("bad first layer"), None, None]
    counter.bind(index, plan)
    counter.wait_until(0)
    counter.wait_until(1)
    assert counter.finish(index) is False
    assert not counter.plans
    assert not counter.reported


def test_read_plan_run_failure_is_reported_after_the_plan_was_bound():
    counter = ReadPlanLoadCounter(1)
    index = counter.update_producer()
    counter.bind(index, Mock())
    counter.fail(index, RuntimeError("plan run failed"))
    assert counter.finish(index) is False
    assert not counter.plans


def test_backends_implement_the_private_load_completion_contract():
    assert not MooncakeDirectLinker.__abstractmethods__
    assert not UMBPDirectLinker.__abstractmethods__
