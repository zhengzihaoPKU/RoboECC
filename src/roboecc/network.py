"""Network prediction interface and paper-aligned split adjustment policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from .segmentation import SplitCandidate


class BandwidthPredictor(Protocol):
    """Interface implemented by an LSTM or a lightweight local fallback."""

    @property
    def minimum_history(self) -> int:
        """Number of observations required before prediction."""

    def predict(self, history: Sequence[float]) -> float:
        """Predict the next bandwidth in the same unit as ``history``."""


@dataclass(frozen=True, slots=True)
class MovingAveragePredictor:
    """Dependency-free smoke-test predictor; not the paper's trained LSTM."""

    window: int = 5

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be greater than zero")

    @property
    def minimum_history(self) -> int:
        return self.window

    def predict(self, history: Sequence[float]) -> float:
        if len(history) < self.minimum_history:
            raise ValueError(
                f"at least {self.minimum_history} bandwidth samples are required"
            )
        values = [float(value) for value in history[-self.window :]]
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("bandwidth samples must be finite and greater than zero")
        return sum(values) / len(values)


@dataclass(frozen=True, slots=True)
class AdjustmentDecision:
    previous_cut: int
    selected_cut: int
    bandwidth_delta: float
    reason: str

    @property
    def changed(self) -> bool:
        return self.previous_cut != self.selected_cut


@dataclass(frozen=True, slots=True)
class NetworkAdjustmentPolicy:
    """Threshold policy from RoboECC Section IV-B.

    Callers must pass only candidates backed by the parameter-sharing pool. On
    bandwidth increase the policy chooses the largest transferable activation;
    on decrease it chooses the smallest. Inside the threshold band it preserves
    the current cut.
    """

    low_delta_threshold: float
    high_delta_threshold: float

    def __post_init__(self) -> None:
        low = float(self.low_delta_threshold)
        high = float(self.high_delta_threshold)
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError("bandwidth delta thresholds must be finite")
        if low > 0 or high < 0:
            raise ValueError(
                "low_delta_threshold must be <= 0 and high_delta_threshold must be >= 0"
            )
        if low > high:
            raise ValueError(
                "low_delta_threshold must not exceed high_delta_threshold"
            )
        object.__setattr__(self, "low_delta_threshold", low)
        object.__setattr__(self, "high_delta_threshold", high)

    def select(
        self,
        *,
        current_cut: int,
        current_bandwidth: float,
        predicted_bandwidth: float,
        pool: Sequence[SplitCandidate],
    ) -> AdjustmentDecision:
        current_bandwidth = float(current_bandwidth)
        predicted_bandwidth = float(predicted_bandwidth)
        if (
            not math.isfinite(current_bandwidth)
            or not math.isfinite(predicted_bandwidth)
            or current_bandwidth <= 0
            or predicted_bandwidth <= 0
        ):
            raise ValueError(
                "current and predicted bandwidth must be finite and positive"
            )
        if not pool:
            raise ValueError("parameter-sharing pool must not be empty")

        unique_cuts = {candidate.cut for candidate in pool}
        if len(unique_cuts) != len(pool):
            raise ValueError("parameter-sharing pool contains duplicate cuts")
        if current_cut not in unique_cuts:
            raise ValueError("current_cut must be present in the parameter-sharing pool")

        delta = predicted_bandwidth - current_bandwidth
        if delta > self.high_delta_threshold:
            target = self._select_by_transfer(
                pool, current_cut=current_cut, largest=True
            )
            reason = "bandwidth_increase"
        elif delta < self.low_delta_threshold:
            target = self._select_by_transfer(
                pool, current_cut=current_cut, largest=False
            )
            reason = "bandwidth_decrease"
        else:
            target = next(
                (candidate for candidate in pool if candidate.cut == current_cut),
                None,
            )
            assert target is not None
            reason = "within_thresholds"

        return AdjustmentDecision(
            previous_cut=current_cut,
            selected_cut=target.cut,
            bandwidth_delta=delta,
            reason=reason,
        )

    @staticmethod
    def _select_by_transfer(
        pool: Sequence[SplitCandidate],
        *,
        current_cut: int,
        largest: bool,
    ) -> SplitCandidate:
        direction = -1.0 if largest else 1.0
        return min(
            pool,
            key=lambda candidate: (
                direction * candidate.transfer_bytes,
                abs(candidate.cut - current_cut),
                candidate.cut,
            ),
        )
