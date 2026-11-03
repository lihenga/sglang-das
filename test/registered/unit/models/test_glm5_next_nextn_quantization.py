"""NextN exclusion rules must keep mixed-checkpoint BF16 experts unquantized."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.srt.layers.quantization import slimquant_w4a8
from sglang.srt.models.glm5_next_nextn import Glm5NextForConditionalGenerationNextN


class TestGlm5NextNQuantization(unittest.TestCase):
    def setUp(self):
        # Only exercise configuration resolution; do not allocate model weights.
        self.model = Glm5NextForConditionalGenerationNextN.__new__(
            Glm5NextForConditionalGenerationNextN
        )
        self.quant = SimpleNamespace(get_name=lambda: "slimquant_w4a8")

    def resolve(self, ignored):
        config = SimpleNamespace(
            num_hidden_layers=45, quantization_config={"ignore": ignored}
        )
        return self.model._resolve_nextn_quant_config(config, self.quant)

    def test_whole_nextn_block_is_unquantized(self):
        for pattern in (
            "model.layers.45.*",
            "model.language_model.layers.45.*",
            "model.layers.45",
            r"re:.*\.language_model\.layers\.45(?:\..*)?$",
            r"re:^model\.language_model\.layers\.45(?:\..*)?$",
            r"re:^model\.layers\.45(?:\..*)?$",
        ):
            with self.subTest(pattern=pattern):
                self.assertIsNone(self.resolve([pattern]))

    def test_partial_or_other_layer_exclusion_preserves_quantization(self):
        for pattern in (
            "model.layers.44.*",
            "model.layers.450.*",
            "model.layers.45.self_attn.*",
            r"re:.*\.self_attn(?:\..*)?$",
            r"re:.*\.mlp\.shared_experts(?:\..*)?$",
            r"re:.*visual(?:\..*)?$",
        ):
            with self.subTest(pattern=pattern):
                self.assertIs(self.resolve([pattern]), self.quant)

    def test_no_exclusions_preserves_quantization(self):
        self.assertIs(self.resolve([]), self.quant)


class TestMixedPrecisionDeepEPDispatch(unittest.TestCase):
    def test_bf16_layer_overrides_global_int8_dispatch(self):
        dispatcher = deepep._DeepEPDispatcherImplNormal.__new__(
            deepep._DeepEPDispatcherImplNormal
        )
        dispatcher.quant_config = {"bf16_dispatch": True}
        dispatcher.async_finish = False
        hidden = torch.ones((2, 4), dtype=torch.bfloat16)
        topk = SimpleNamespace(
            topk_weights=torch.ones((2, 1)),
            topk_ids=torch.zeros((2, 1), dtype=torch.int32),
        )
        with (
            patch.object(deepep, "use_groupgemm", True),
            patch.object(deepep, "_use_fp8_w8a8_moe", False),
            patch.object(deepep, "_use_marlin_w16a16_moe", False),
            patch.object(deepep, "per_token_quant_int8") as quantize,
        ):
            result = dispatcher.dispatch_a(hidden, topk)
            self.assertIs(result[0], hidden)
            self.assertEqual(result[1].dtype, torch.int64)
            quantize.assert_not_called()
            # The quantized target layer must still take its original path.
            dispatcher.quant_config = {}
            dispatcher.dispatch_a(hidden, topk)
            quantize.assert_called_once_with(hidden)

    def test_int8_layer_overrides_global_fp8_dispatch(self):
        dispatcher = deepep._DeepEPDispatcherImplNormal.__new__(
            deepep._DeepEPDispatcherImplNormal
        )
        dispatcher.quant_config = {"int8_dispatch": True}
        dispatcher.async_finish = False
        hidden = torch.ones((2, 4), dtype=torch.bfloat16)
        topk = SimpleNamespace(
            topk_weights=torch.ones((2, 1)),
            topk_ids=torch.zeros((2, 1), dtype=torch.int64),
        )
        with (
            patch.object(deepep, "use_groupgemm", True),
            patch.object(deepep, "_use_fp8_w8a8_moe", True),
            patch.object(deepep, "per_token_quant_int8") as int8,
            patch.object(deepep, "per_token_quant_fp8") as fp8,
        ):
            dispatcher.dispatch_a(hidden, topk)
            int8.assert_called_once_with(hidden)
            fp8.assert_not_called()

    def test_w4a8_uses_local_experts_and_existing_activation_scales(self):
        method = SimpleNamespace(
            use_deepgemm=False,
            moe_runner_config=SimpleNamespace(
                activation="silu", apply_router_weight_on_input=False
            ),
        )
        layer = SimpleNamespace(
            w13_weight=torch.empty((2, 8, 2), dtype=torch.int8),
            w2_weight=torch.empty((2, 4, 2), dtype=torch.int8),
            w13_weight_scale=torch.ones((2, 8, 1)),
            w2_weight_scale=torch.ones((2, 4, 1)),
            params_dtype=torch.bfloat16,
            num_local_experts=2,
        )
        dispatch = deepep.DeepEPNormalDispatchOutput(
            hidden_states=torch.ones((2, 4), dtype=torch.int8),
            hidden_states_scale=torch.ones((2, 1)),
            topk_ids=torch.tensor([[0, -1], [1, 0]]),
            topk_weights=torch.ones((2, 2)),
            num_recv_tokens_per_expert=[2, 1],
        )
        with (
            patch.object(slimquant_w4a8, "get_moe_cache"),
            patch.object(slimquant_w4a8, "fused_experts_impl_w4a8_triton") as kernel,
        ):
            result = slimquant_w4a8.SlimQuantW4A8Int8MoEMethod.apply_deepep_normal(
                method, layer, dispatch
            )
            self.assertIs(result, kernel.return_value)
            kwargs = kernel.call_args.kwargs
            self.assertEqual(kwargs["global_num_experts"], 2)
            self.assertIsNone(kwargs["expert_map"])
            self.assertIs(kwargs["input_scale"], dispatch.hidden_states_scale)
            self.assertEqual(kwargs["out_dtype"], torch.bfloat16)
            self.assertEqual(kwargs["routed_scaling_factor"], 1.0)


if __name__ == "__main__":
    unittest.main()
