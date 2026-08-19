from __future__ import annotations

import unittest

from roboecc.llama_block import (
    ExpandedLlamaModel,
    LlamaBlockProfile,
    LlamaStage,
)
from roboecc.network import NetworkAdjustmentPolicy
from roboecc.pool import ParameterSharingPool
from roboecc.profiles import HardwareProfile, LayerProfile, NetworkProfile
from roboecc.segmentation import enumerate_splits


class LlamaBlockTests(unittest.TestCase):
    def test_llama_7b_parameter_count_matches_decoder_layer(self) -> None:
        block = LlamaBlockProfile(
            block_index=0,
            hidden_size=4096,
            intermediate_size=11008,
            sequence_length=17,
            num_attention_heads=32,
            num_key_value_heads=32,
        )
        self.assertEqual(block.total_parameter_count, 202_383_360)
        self.assertEqual(len(block.to_layer_profiles()), 5)

    def test_internal_payload_includes_required_residual(self) -> None:
        block = LlamaBlockProfile(
            block_index=13,
            hidden_size=768,
            intermediate_size=3072,
            sequence_length=17,
            num_attention_heads=12,
            num_key_value_heads=12,
        )
        hidden_bytes = 17 * 768 * 2
        intermediate_bytes = 17 * 3072 * 2
        self.assertEqual(
            block.payload_after(LlamaStage.SELF_ATTENTION).size_bytes,
            hidden_bytes,
        )
        self.assertEqual(
            block.payload_after(LlamaStage.INPUT_NORM).size_bytes,
            2 * hidden_bytes,
        )
        self.assertEqual(
            block.payload_after(LlamaStage.MLP_EXPAND).size_bytes,
            intermediate_bytes + hidden_bytes,
        )

    def test_expanded_model_maps_local_to_global_cut(self) -> None:
        prefix = LayerProfile("vision", 1, 1, 1, 1)
        blocks = [
            LlamaBlockProfile(4, 768, 3072, 17, 12, 12),
            LlamaBlockProfile(5, 768, 3072, 17, 12, 12),
        ]
        expanded = ExpandedLlamaModel.build(
            "toy-llama", blocks, prefix_layers=(prefix,)
        )
        self.assertEqual(expanded.internal_cuts(4), (2, 3, 4, 5))
        plan = expanded.split_plan(5, cloud_stage_count=3)
        self.assertEqual(plan.global_cut, 9)
        self.assertEqual(plan.boundary_name, "after_post_attention_norm")
        self.assertEqual(plan.cloud_stages[-1], LlamaStage.POST_ATTENTION_NORM)
        self.assertEqual(plan.edge_stages[0], LlamaStage.MLP_EXPAND)

    def test_internal_cuts_work_with_pool_and_network_policy(self) -> None:
        block = LlamaBlockProfile(0, 768, 3072, 17, 12, 12)
        expanded = ExpandedLlamaModel.build("one-block", (block,))
        hardware = HardwareProfile("gpu", 1e12, 1e12)
        candidates = enumerate_splits(
            expanded.profile,
            cloud=hardware,
            edge=hardware,
            network=NetworkProfile(10_000_000),
        )
        cuts = expanded.internal_cuts(0)
        pool = ParameterSharingPool(
            expanded.profile,
            allowed_cuts=cuts,
            initial_cut=cuts[1],
        )
        pool_candidates = [candidates[cut] for cut in cuts]
        policy = NetworkAdjustmentPolicy(-1, 1)

        increased = policy.select(
            current_cut=cuts[1],
            current_bandwidth=10,
            predicted_bandwidth=12,
            pool=pool_candidates,
        )
        decreased = policy.select(
            current_cut=cuts[1],
            current_bandwidth=10,
            predicted_bandwidth=8,
            pool=pool_candidates,
        )
        self.assertEqual(
            increased.selected_cut,
            expanded.global_cut(0, 4),
        )
        self.assertEqual(
            decreased.selected_cut,
            expanded.global_cut(0, 2),
        )
        self.assertIn("llm.blocks.0.self_attention", pool.layout.shared)

    def test_block_cuts_share_the_complete_block(self) -> None:
        prefix = LayerProfile("vision", 1, 1, 7, 1)
        suffix = LayerProfile("head", 1, 1, 11, 1)
        block = LlamaBlockProfile(3, 768, 3072, 17, 12, 12)
        expanded = ExpandedLlamaModel.build(
            "one-block",
            (block,),
            prefix_layers=(prefix,),
            suffix_layers=(suffix,),
        )
        cuts = expanded.block_cuts(3)
        pool = ParameterSharingPool(
            expanded.profile,
            allowed_cuts=cuts,
            initial_cut=expanded.global_cut(3, 2),
        )
        self.assertEqual(cuts, (1, 2, 3, 4, 5, 6))
        self.assertEqual(len(pool.layout.shared), 5)
        self.assertEqual(
            pool.layout.extra_parameter_bytes,
            block.total_parameter_count * block.dtype_bytes,
        )

    def test_gqa_head_ratio_must_be_integral(self) -> None:
        with self.assertRaises(ValueError):
            LlamaBlockProfile(0, 768, 3072, 17, 12, 5)


if __name__ == "__main__":
    unittest.main()
