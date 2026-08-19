"""Inspect the legal cut points inside one illustrative LLaMA block."""

from roboecc.llama_block import ExpandedLlamaModel, LlamaBlockProfile
from roboecc.pool import ParameterSharingPool
from roboecc.profiles import HardwareProfile, NetworkProfile
from roboecc.segmentation import enumerate_splits


block = LlamaBlockProfile(
    block_index=13,
    hidden_size=768,
    intermediate_size=3072,
    sequence_length=17,
    num_attention_heads=12,
    num_key_value_heads=12,
    dtype_bytes=2,
)
expanded = ExpandedLlamaModel.build("illustrative-llama-block", (block,))

cloud = HardwareProfile("cloud", 300e12, 1_500e9, parallel_efficiency=0.35)
edge = HardwareProfile("edge", 50e12, 250e9, parallel_efficiency=0.25)
network = NetworkProfile(10_000_000, fixed_latency_ms=1.0)
candidates = enumerate_splits(
    expanded.profile,
    cloud=cloud,
    edge=edge,
    network=network,
)

internal_cuts = expanded.internal_cuts(block.block_index)
pool = ParameterSharingPool(
    expanded.profile,
    allowed_cuts=expanded.block_cuts(block.block_index),
    initial_cut=expanded.global_cut(block.block_index, 2),
)

for stage_count, cut in enumerate(internal_cuts, start=1):
    plan = expanded.split_plan(block.block_index, stage_count)
    candidate = candidates[cut]
    payload_names = ", ".join(plan.payload.tensors)
    print(
        f"cut={cut} {plan.boundary_name:<30} "
        f"payload={candidate.transfer_bytes / 1024:>6.1f} KiB "
        f"({payload_names})"
    )

print(f"shared stages: {', '.join(pool.layout.shared)}")
print(f"extra parameters: {pool.layout.extra_parameter_bytes / 1024**2:.2f} MiB")
