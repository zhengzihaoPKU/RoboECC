from __future__ import annotations

import math
import unittest

from roboecc.network import MovingAveragePredictor, NetworkAdjustmentPolicy
from roboecc.profiles import HardwareProfile, LayerProfile, ModelProfile, NetworkProfile
from roboecc.segmentation import enumerate_splits


def make_pool():
    model = ModelProfile(
        name="toy",
        layers=(
            LayerProfile("a", 1.0, 1.0, 1.0, 100.0),
            LayerProfile("b", 1.0, 1.0, 1.0, 20.0),
            LayerProfile("c", 1.0, 1.0, 1.0, 5.0),
        ),
    )
    hardware = HardwareProfile("gpu", 1.0, 1.0)
    candidates = enumerate_splits(
        model,
        cloud=hardware,
        edge=hardware,
        network=NetworkProfile(1.0),
    )
    return candidates[1:3]


class NetworkTests(unittest.TestCase):
    def test_moving_average_uses_recent_window(self) -> None:
        predictor = MovingAveragePredictor(window=2)
        self.assertEqual(predictor.predict([1.0, 4.0, 6.0]), 5.0)

    def test_bandwidth_increase_selects_largest_boundary(self) -> None:
        decision = NetworkAdjustmentPolicy(-2.0, 2.0).select(
            current_cut=2,
            current_bandwidth=10.0,
            predicted_bandwidth=13.0,
            pool=make_pool(),
        )
        self.assertEqual(decision.selected_cut, 1)
        self.assertEqual(decision.reason, "bandwidth_increase")

    def test_bandwidth_decrease_selects_smallest_boundary(self) -> None:
        decision = NetworkAdjustmentPolicy(-2.0, 2.0).select(
            current_cut=1,
            current_bandwidth=10.0,
            predicted_bandwidth=7.0,
            pool=make_pool(),
        )
        self.assertEqual(decision.selected_cut, 2)
        self.assertEqual(decision.reason, "bandwidth_decrease")

    def test_threshold_band_keeps_current_cut(self) -> None:
        decision = NetworkAdjustmentPolicy(-2.0, 2.0).select(
            current_cut=2,
            current_bandwidth=10.0,
            predicted_bandwidth=11.0,
            pool=make_pool(),
        )
        self.assertFalse(decision.changed)
        self.assertEqual(decision.reason, "within_thresholds")

    def test_non_finite_bandwidth_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MovingAveragePredictor(window=2).predict([1.0, math.nan])
        with self.assertRaises(ValueError):
            NetworkAdjustmentPolicy(-2.0, 2.0).select(
                current_cut=1,
                current_bandwidth=10.0,
                predicted_bandwidth=math.inf,
                pool=make_pool(),
            )

    def test_thresholds_must_straddle_zero(self) -> None:
        with self.assertRaises(ValueError):
            NetworkAdjustmentPolicy(1.0, 2.0)


if __name__ == "__main__":
    unittest.main()
