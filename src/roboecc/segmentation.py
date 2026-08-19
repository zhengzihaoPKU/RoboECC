"""Model-hardware co-aware split enumeration and selection."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .profiles import (
    HardwareProfile,
    ModelProfile,
    NetworkProfile,
)


@dataclass(frozen=True, slots=True)
class SplitCandidate:
    """Estimated cost of one cloud-prefix / edge-suffix deployment.

    ``cut`` is the number of layers placed on the cloud. Therefore, ``cut=0``
    is all-edge and ``cut=len(model.layers)`` is all-cloud.
    """

    cut: int
    cloud_layer_names: tuple[str, ...]
    edge_layer_names: tuple[str, ...]
    cloud_compute_ms: float
    edge_compute_ms: float
    transfer_ms: float
    transfer_bytes: float
    cloud_load: float
    feasible: bool

    @property
    def total_ms(self) -> float:
        return self.cloud_compute_ms + self.edge_compute_ms + self.transfer_ms


def enumerate_splits(
    model: ModelProfile,
    *,
    cloud: HardwareProfile,
    edge: HardwareProfile,
    network: NetworkProfile,
    max_cloud_load: float | None = None,
) -> tuple[SplitCandidate, ...]:
    """Estimate all split points in deterministic model order.

    The planner uses a single cut, matching the deployable cloud-to-edge path in
    the local OpenVLA and CogACT adapters. A multi-cut sandwich architecture is
    intentionally outside this core search space because it adds extra network
    round trips inside autoregressive or diffusion loops.
    """

    if max_cloud_load is not None:
        max_cloud_load = float(max_cloud_load)
        if not math.isfinite(max_cloud_load) or max_cloud_load < 0:
            raise ValueError("max_cloud_load must be finite and non-negative or None")

    layers = model.layers
    cloud_prefix_ms = [0.0]
    cloud_prefix_load = [0.0]
    for layer in layers:
        cloud_prefix_ms.append(
            cloud_prefix_ms[-1] + cloud.estimate_layer_ms(layer)
        )
        cloud_prefix_load.append(
            cloud_prefix_load[-1] + float(layer.cloud_load)
        )

    edge_suffix_ms = [0.0] * (len(layers) + 1)
    for index in range(len(layers) - 1, -1, -1):
        edge_suffix_ms[index] = (
            edge_suffix_ms[index + 1] + edge.estimate_layer_ms(layers[index])
        )

    candidates: list[SplitCandidate] = []
    for cut in range(len(layers) + 1):
        is_internal_cut = 0 < cut < len(layers)
        transfer_bytes = layers[cut - 1].output_bytes if is_internal_cut else 0.0
        cloud_load = cloud_prefix_load[cut]
        candidates.append(
            SplitCandidate(
                cut=cut,
                cloud_layer_names=tuple(layer.name for layer in layers[:cut]),
                edge_layer_names=tuple(layer.name for layer in layers[cut:]),
                cloud_compute_ms=cloud_prefix_ms[cut],
                edge_compute_ms=edge_suffix_ms[cut],
                transfer_ms=network.transfer_ms(transfer_bytes),
                transfer_bytes=transfer_bytes,
                cloud_load=cloud_load,
                feasible=(
                    max_cloud_load is None or cloud_load <= max_cloud_load
                ),
            )
        )
    return tuple(candidates)


def search_optimal_split(
    model: ModelProfile,
    *,
    cloud: HardwareProfile,
    edge: HardwareProfile,
    network: NetworkProfile,
    max_cloud_load: float | None = None,
) -> SplitCandidate:
    """Return the lowest-latency feasible candidate.

    Ties prefer the smaller cloud load, followed by the earlier cut, so repeated
    runs and JSON serializations remain stable.
    """

    feasible = [
        candidate
        for candidate in enumerate_splits(
            model,
            cloud=cloud,
            edge=edge,
            network=network,
            max_cloud_load=max_cloud_load,
        )
        if candidate.feasible
    ]
    if not feasible:
        raise ValueError("no feasible split candidate under the cloud load budget")
    return min(
        feasible,
        key=lambda candidate: (
            candidate.total_ms,
            candidate.cloud_load,
            candidate.cut,
        ),
    )
