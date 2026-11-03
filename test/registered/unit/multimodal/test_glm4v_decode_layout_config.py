import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.glm4v import Glm4vImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

GLM4V = "sglang.srt.multimodal.processors.glm4v"
TOKENIZER_MANAGER = "sglang.srt.managers.tokenizer_manager"


def _mm_config(config):
    return SimpleNamespace(mm_process_config=config)


def _glm_processor():
    processor = Glm4vImageProcessor.__new__(Glm4vImageProcessor)
    processor.image_config = {"max_pixels": 111}
    processor.video_config = {"max_frames": 32, "size": {"longest_edge": 64}}
    processor._processor = SimpleNamespace(video_processor=None)
    processor.mm_tokens = MagicMock()
    return processor


def test_encoder_mm_config_reads_vision_config_only():
    config = {"video": {"max_frames": 32}, "vision_config": {"video": {"fps": 1}}}
    with patch(f"{GLM4V}.get_mm", return_value=_mm_config(config)):
        assert Glm4vImageProcessor._encoder_mm_config("video") == {"fps": 1}
        assert Glm4vImageProcessor._encoder_mm_config("image") == {}


def test_layout_ignores_top_level_tokenizer_config():
    processor = _glm_processor()
    loaded = SimpleNamespace(images=["image"], videos=["video"])
    captured = {}

    async def fast_load_mm_data(**kwargs):
        return loaded

    async def sample_glm_videos(base_output, *, video_configs, **kwargs):
        captured["video_configs"] = video_configs
        return [{"fps": 1}], {"max_image_tokens": 7}

    async def process_media_only(**kwargs):
        captured["processor_kwargs"] = kwargs
        return {}

    processor.fast_load_mm_data = fast_load_mm_data
    processor._sample_glm_videos = sample_glm_videos
    processor._process_media_only = process_media_only
    processor.get_mm_data = MagicMock(return_value=SimpleNamespace(mm_items=[]))
    config = {
        "image": {"max_pixels": 111},
        "video": {"max_frames": 32},
        "vision_config": {"image": {"max_pixels": 222}, "video": {"fps": 1}},
    }
    with (
        patch(f"{GLM4V}.get_mm", return_value=_mm_config(config)),
        patch(f"{GLM4V}.split_glm_video_items", return_value=(["video"], [{}])),
        patch(f"{GLM4V}.glm_processor_video_config", return_value={}),
        patch(f"{GLM4V}.glm_video_timestamps", return_value=[0]),
        patch(f"{GLM4V}._attach_layout_features"),
    ):
        asyncio.run(
            processor.process_mm_layout_async(
                ["image"], [1, 2], SimpleNamespace(video_data=["video"])
            )
        )

    assert captured["video_configs"][0].get("fps") == 1
    assert "max_frames" not in captured["video_configs"][0]
    kwargs = captured["processor_kwargs"]
    assert kwargs["processor_image_config"] == {"max_pixels": 222}
    assert kwargs["processor_video_config"] == {"max_image_tokens": 7}


def test_process_mm_data_prefers_explicit_image_config():
    processor = Glm4vImageProcessor.__new__(Glm4vImageProcessor)
    processor.image_config = {"max_pixels": 111}
    processor.disable_fast_image_processor = True
    processor.video_preprocessing_device = None
    processor.mm_feature_transport = "disabled"
    processor._tokenizer_auto_adds_specials = False
    processor.precompute_hash_before_cpu_transfer = False
    processor.FEATURE_NAMES = []
    hf_processor = MagicMock(return_value={})
    with patch.object(
        BaseMultimodalProcessor,
        "_resolve_processor",
        return_value=(hf_processor, MagicMock()),
    ):
        processor.process_mm_data("", images=["image"], processor_image_config={})

    assert "images_kwargs" not in hf_processor.call_args.kwargs


def _dispatch_decision(builds_layout):
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.disaggregation_mode = DisaggregationMode.DECODE
    manager.mm_processor = SimpleNamespace(builds_epd_layout_from_media=builds_layout)
    manager.mm_receiver = MagicMock()
    manager.mm_receiver.send_encode_request.return_value = None
    manager.enable_trace = False
    manager.tokenizer = None
    obj = MagicMock(spec=GenerateReqInput)
    obj.contains_mm_input.return_value = True
    obj.image_data = None
    disagg = SimpleNamespace(
        language_only=True,
        encoder_transfer_backend="mooncake",
        enable_adaptive_dispatch_to_encoder=False,
    )
    with (
        patch(f"{TOKENIZER_MANAGER}.get_disagg", return_value=disagg),
        patch.object(TokenizerManager, "_normalize_mm_content_hashes"),
    ):
        manager._handle_epd_disaggregation_encode_request(obj)
    assert manager.mm_receiver.send_encode_request.called is (not builds_layout)
    return obj.need_wait_for_mm_inputs


def test_decode_skips_encoder_with_media_layout():
    assert _dispatch_decision(builds_layout=True) is False


def test_decode_keeps_dispatching_without_media_layout():
    assert _dispatch_decision(builds_layout=False) is True
