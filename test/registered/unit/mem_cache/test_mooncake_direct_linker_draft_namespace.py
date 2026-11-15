from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    _windowed_draft_storage_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_windowed_draft_keeps_user_fields_and_isolates_ring_layout():
    original = {
        "direct_storage_page_size": 128,
        "extra_backend_tag": "case",
        "other": "unchanged",
    }
    result = _windowed_draft_storage_config(original, windowed_draft=True)
    assert result == dict(original, extra_backend_tag="case_dspark_ring_v1")
    assert original["extra_backend_tag"] == "case"


def test_non_windowed_draft_keeps_original_config():
    config = {"direct_storage_page_size": 128}
    assert _windowed_draft_storage_config(config, windowed_draft=False) is config
