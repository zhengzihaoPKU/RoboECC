"""Online controller connecting bandwidth prediction, policy, and deployment."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Sequence

from .network import (
    AdjustmentDecision,
    BandwidthPredictor,
    NetworkAdjustmentPolicy,
)
from .pool import ParameterSharingPool
from .segmentation import SplitCandidate


@dataclass(frozen=True, slots=True)
class RuntimeUpdate:
    observed_bandwidth: float
    predicted_bandwidth: float | None
    adjustment: AdjustmentDecision | None
    status: str


class DynamicSplitController:
    """Apply network-aware split changes between complete inference calls.

    ``apply_cut`` is the deployment-specific two-sided reconfiguration callback.
    It is called before the in-memory pool state changes, so a failed RPC leaves
    the controller on the last known-good cut.
    """

    def __init__(
        self,
        *,
        pool: ParameterSharingPool,
        candidates: Sequence[SplitCandidate],
        predictor: BandwidthPredictor,
        policy: NetworkAdjustmentPolicy,
        apply_cut: Callable[[int], None] | None = None,
        history_capacity: int | None = None,
    ) -> None:
        candidate_by_cut = {candidate.cut: candidate for candidate in candidates}
        if len(candidate_by_cut) != len(candidates):
            raise ValueError("candidates contain duplicate cuts")
        missing = set(pool.allowed_cuts) - set(candidate_by_cut)
        if missing:
            raise ValueError(f"missing split candidates for pool cuts: {sorted(missing)}")
        infeasible = [
            cut for cut in pool.allowed_cuts if not candidate_by_cut[cut].feasible
        ]
        if infeasible:
            raise ValueError(
                f"parameter-sharing pool contains infeasible cuts: {infeasible}"
            )

        minimum_history = predictor.minimum_history
        if (
            isinstance(minimum_history, bool)
            or not isinstance(minimum_history, int)
            or minimum_history <= 0
        ):
            raise ValueError("predictor.minimum_history must be a positive integer")
        capacity = (
            max(minimum_history, 128)
            if history_capacity is None
            else history_capacity
        )
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise ValueError("history_capacity must be an integer")
        if capacity < minimum_history:
            raise ValueError("history_capacity is smaller than predictor requirement")

        self.pool = pool
        self.predictor = predictor
        self.policy = policy
        self.apply_cut = apply_cut
        self._candidate_by_cut = candidate_by_cut
        self._history: deque[float] = deque(maxlen=capacity)

    @property
    def history(self) -> tuple[float, ...]:
        return tuple(self._history)

    def observe(self, bandwidth: float) -> RuntimeUpdate:
        """Record effective bandwidth and configure the split for the next call."""

        bandwidth = float(bandwidth)
        if not math.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("observed bandwidth must be finite and positive")
        self._history.append(bandwidth)

        if len(self._history) < self.predictor.minimum_history:
            return RuntimeUpdate(
                observed_bandwidth=bandwidth,
                predicted_bandwidth=None,
                adjustment=None,
                status="warming_up",
            )

        predicted = float(self.predictor.predict(self.history))
        if not math.isfinite(predicted) or predicted <= 0:
            raise ValueError("predicted bandwidth must be finite and positive")
        pool_candidates = [
            self._candidate_by_cut[cut] for cut in self.pool.allowed_cuts
        ]
        adjustment = self.policy.select(
            current_cut=self.pool.active_cut,
            current_bandwidth=bandwidth,
            predicted_bandwidth=predicted,
            pool=pool_candidates,
        )
        if adjustment.changed:
            self.pool.reconfigure(adjustment.selected_cut, self.apply_cut)
            status = "reconfigured"
        else:
            status = "stable"
        return RuntimeUpdate(
            observed_bandwidth=bandwidth,
            predicted_bandwidth=predicted,
            adjustment=adjustment,
            status=status,
        )


def effective_bandwidth_bytes_per_second(
    *,
    transferred_bytes: int,
    round_trip_ms: float,
    remote_compute_ms: float = 0.0,
    minimum_network_ms: float = 0.01,
) -> float:
    """Estimate effective throughput from an inference RPC observation.

    Serialization and queueing remain part of the effective link cost, while
    reported remote GPU compute is removed from the round-trip measurement.
    """

    if transferred_bytes <= 0:
        raise ValueError("transferred_bytes must be positive")
    if not math.isfinite(round_trip_ms) or round_trip_ms <= 0:
        raise ValueError("round_trip_ms must be finite and positive")
    if not math.isfinite(remote_compute_ms) or remote_compute_ms < 0:
        raise ValueError("remote_compute_ms must be finite and non-negative")
    if not math.isfinite(minimum_network_ms) or minimum_network_ms <= 0:
        raise ValueError("minimum_network_ms must be finite and positive")
    network_ms = max(round_trip_ms - remote_compute_ms, minimum_network_ms)
    return float(transferred_bytes) / (network_ms / 1_000.0)
