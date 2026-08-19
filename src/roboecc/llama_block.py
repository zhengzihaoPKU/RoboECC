"""Fine-grained, planning-only decomposition of a LLaMA decoder block."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .profiles import LayerProfile, ModelProfile


class LlamaStage(str, Enum):
    """Sequential execution stages exposed as legal planning units."""

    INPUT_NORM = "input_norm"
    SELF_ATTENTION = "self_attention"
    POST_ATTENTION_NORM = "post_attention_norm"
    MLP_EXPAND = "mlp_expand"
    MLP_DOWN = "mlp_down"


LLAMA_STAGE_ORDER = (
    LlamaStage.INPUT_NORM,
    LlamaStage.SELF_ATTENTION,
    LlamaStage.POST_ATTENTION_NORM,
    LlamaStage.MLP_EXPAND,
    LlamaStage.MLP_DOWN,
)


@dataclass(frozen=True, slots=True)
class BoundaryPayload:
    """Logical tensors that must cross an internal block boundary."""

    tensors: tuple[str, ...]
    size_bytes: int


@dataclass(frozen=True, slots=True)
class LlamaBlockSplitPlan:
    """Planning result for one LLaMA block.

    ``cloud_stage_count`` ranges from zero to five. Internal values split the
    block; endpoints place the entire block on edge or cloud respectively.
    This object describes execution and transfer only—it does not patch a
    Transformers forward method.
    """

    block_index: int
    global_cut: int
    cloud_stage_count: int
    cloud_stages: tuple[LlamaStage, ...]
    edge_stages: tuple[LlamaStage, ...]
    payload: BoundaryPayload

    @property
    def is_internal(self) -> bool:
        return 0 < self.cloud_stage_count < len(LLAMA_STAGE_ORDER)

    @property
    def boundary_name(self) -> str:
        if self.cloud_stage_count == 0:
            return "before_block"
        return f"after_{LLAMA_STAGE_ORDER[self.cloud_stage_count - 1].value}"


@dataclass(frozen=True, slots=True)
class LlamaBlockProfile:
    """Dimensions and derived costs for one LLaMA decoder block."""

    block_index: int
    hidden_size: int
    intermediate_size: int
    sequence_length: int
    num_attention_heads: int
    num_key_value_heads: int
    dtype_bytes: int = 2
    prefix: str = "llm.blocks"
    include_bias: bool = False

    def __post_init__(self) -> None:
        integer_fields = (
            "hidden_size",
            "intermediate_size",
            "sequence_length",
            "num_attention_heads",
            "num_key_value_heads",
            "dtype_bytes",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if (
            isinstance(self.block_index, bool)
            or not isinstance(self.block_index, int)
            or self.block_index < 0
        ):
            raise ValueError("block_index must be a non-negative integer")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError("num_key_value_heads must not exceed num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if not self.prefix.strip():
            raise ValueError("prefix must not be empty")

    @property
    def activation_bytes(self) -> int:
        return self.sequence_length * self.hidden_size * self.dtype_bytes

    @property
    def intermediate_activation_bytes(self) -> int:
        return self.sequence_length * self.intermediate_size * self.dtype_bytes

    @property
    def key_value_width(self) -> int:
        head_size = self.hidden_size // self.num_attention_heads
        return self.num_key_value_heads * head_size

    @property
    def stage_parameter_counts(self) -> dict[LlamaStage, int]:
        hidden = self.hidden_size
        intermediate = self.intermediate_size
        key_value = self.key_value_width
        attention = 2 * hidden * hidden + 2 * hidden * key_value
        mlp_expand = 2 * hidden * intermediate
        mlp_down = hidden * intermediate
        if self.include_bias:
            attention += 2 * hidden + 2 * key_value
            mlp_expand += 2 * intermediate
            mlp_down += hidden
        return {
            LlamaStage.INPUT_NORM: hidden,
            LlamaStage.SELF_ATTENTION: attention,
            LlamaStage.POST_ATTENTION_NORM: hidden,
            LlamaStage.MLP_EXPAND: mlp_expand,
            LlamaStage.MLP_DOWN: mlp_down,
        }

    @property
    def total_parameter_count(self) -> int:
        return sum(self.stage_parameter_counts.values())

    def payload_after(self, stage: LlamaStage | None) -> BoundaryPayload:
        """Return all state needed by the next device after ``stage``."""

        hidden = self.activation_bytes
        if stage is None:
            return BoundaryPayload(("hidden_states",), hidden)
        if stage is LlamaStage.INPUT_NORM:
            return BoundaryPayload(
                ("normalized_hidden_states", "attention_residual"),
                2 * hidden,
            )
        if stage is LlamaStage.SELF_ATTENTION:
            return BoundaryPayload(("hidden_states",), hidden)
        if stage is LlamaStage.POST_ATTENTION_NORM:
            return BoundaryPayload(
                ("normalized_hidden_states", "mlp_residual"),
                2 * hidden,
            )
        if stage is LlamaStage.MLP_EXPAND:
            return BoundaryPayload(
                ("mlp_intermediate", "mlp_residual"),
                self.intermediate_activation_bytes + hidden,
            )
        if stage is LlamaStage.MLP_DOWN:
            return BoundaryPayload(("hidden_states",), hidden)
        raise ValueError(f"unsupported LLaMA stage: {stage}")

    def _stage_flops(self) -> dict[LlamaStage, int]:
        sequence = self.sequence_length
        hidden = self.hidden_size
        intermediate = self.intermediate_size
        key_value = self.key_value_width
        attention_projection = 2 * sequence * (
            2 * hidden * hidden + 2 * hidden * key_value
        )
        attention_scores = 4 * sequence * sequence * hidden
        return {
            LlamaStage.INPUT_NORM: 5 * sequence * hidden,
            LlamaStage.SELF_ATTENTION: attention_projection + attention_scores,
            LlamaStage.POST_ATTENTION_NORM: 5 * sequence * hidden,
            LlamaStage.MLP_EXPAND: (
                4 * sequence * hidden * intermediate
                + 5 * sequence * intermediate
            ),
            LlamaStage.MLP_DOWN: (
                2 * sequence * intermediate * hidden + sequence * hidden
            ),
        }

    def to_layer_profiles(self) -> tuple[LayerProfile, ...]:
        """Expand this block into five layers understood by the core planner."""

        parameter_counts = self.stage_parameter_counts
        flops = self._stage_flops()
        layers = []
        input_payload = self.payload_after(None).size_bytes
        for stage in LLAMA_STAGE_ORDER:
            output_payload = self.payload_after(stage).size_bytes
            parameter_bytes = parameter_counts[stage] * self.dtype_bytes
            layers.append(
                LayerProfile(
                    name=f"{self.prefix}.{self.block_index}.{stage.value}",
                    compute_flops=flops[stage],
                    memory_bytes=parameter_bytes + input_payload + output_payload,
                    parameter_bytes=parameter_bytes,
                    output_bytes=output_payload,
                )
            )
            input_payload = output_payload
        return tuple(layers)

    def split_plan(
        self,
        cloud_stage_count: int,
        *,
        global_cut: int,
    ) -> LlamaBlockSplitPlan:
        cloud_stage_count = int(cloud_stage_count)
        if not 0 <= cloud_stage_count <= len(LLAMA_STAGE_ORDER):
            raise ValueError(
                f"cloud_stage_count must be in [0, {len(LLAMA_STAGE_ORDER)}]"
            )
        previous_stage = (
            None
            if cloud_stage_count == 0
            else LLAMA_STAGE_ORDER[cloud_stage_count - 1]
        )
        return LlamaBlockSplitPlan(
            block_index=self.block_index,
            global_cut=int(global_cut),
            cloud_stage_count=cloud_stage_count,
            cloud_stages=LLAMA_STAGE_ORDER[:cloud_stage_count],
            edge_stages=LLAMA_STAGE_ORDER[cloud_stage_count:],
            payload=self.payload_after(previous_stage),
        )


@dataclass(frozen=True, slots=True)
class ExpandedLlamaModel:
    """A normal ModelProfile plus mapping back to block-internal cuts."""

    profile: ModelProfile
    blocks: tuple[LlamaBlockProfile, ...]
    prefix_layer_count: int

    @classmethod
    def build(
        cls,
        name: str,
        blocks: Sequence[LlamaBlockProfile],
        *,
        prefix_layers: Sequence[LayerProfile] = (),
        suffix_layers: Sequence[LayerProfile] = (),
    ) -> ExpandedLlamaModel:
        block_tuple = tuple(blocks)
        if not block_tuple:
            raise ValueError("at least one LLaMA block is required")
        indices = [block.block_index for block in block_tuple]
        if len(indices) != len(set(indices)):
            raise ValueError("LLaMA block indices must be unique")
        expanded_layers = tuple(prefix_layers) + tuple(
            layer
            for block in block_tuple
            for layer in block.to_layer_profiles()
        ) + tuple(suffix_layers)
        return cls(
            profile=ModelProfile(name=name, layers=expanded_layers),
            blocks=block_tuple,
            prefix_layer_count=len(prefix_layers),
        )

    def global_cut(self, block_index: int, cloud_stage_count: int) -> int:
        """Convert a block-local cut into the flattened planner cut index."""

        position = self._block_position(block_index)
        if not 0 <= cloud_stage_count <= len(LLAMA_STAGE_ORDER):
            raise ValueError(
                f"cloud_stage_count must be in [0, {len(LLAMA_STAGE_ORDER)}]"
            )
        return (
            self.prefix_layer_count
            + position * len(LLAMA_STAGE_ORDER)
            + int(cloud_stage_count)
        )

    def split_plan(
        self,
        block_index: int,
        cloud_stage_count: int,
    ) -> LlamaBlockSplitPlan:
        block = self.blocks[self._block_position(block_index)]
        cut = self.global_cut(block_index, cloud_stage_count)
        return block.split_plan(cloud_stage_count, global_cut=cut)

    def internal_cuts(self, block_index: int) -> tuple[int, ...]:
        """Return legal cuts strictly inside one block."""

        return tuple(
            self.global_cut(block_index, stage_count)
            for stage_count in range(1, len(LLAMA_STAGE_ORDER))
        )

    def block_cuts(self, block_index: int) -> tuple[int, ...]:
        """Return every cut across a block, including both block boundaries.

        Using these cuts in :class:`ParameterSharingPool` keeps the complete
        block resident on both devices, matching RoboECC's parameter-sharing
        pool rather than only duplicating the stages that move.
        """

        return tuple(
            self.global_cut(block_index, stage_count)
            for stage_count in range(len(LLAMA_STAGE_ORDER) + 1)
        )

    def _block_position(self, block_index: int) -> int:
        for position, block in enumerate(self.blocks):
            if block.block_index == block_index:
                return position
        raise ValueError(f"unknown LLaMA block index: {block_index}")
