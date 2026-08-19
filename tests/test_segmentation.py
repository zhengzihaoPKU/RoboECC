from __future__ import annotations

import unittest

from roboecc.profiles import HardwareProfile, LayerProfile, ModelProfile, NetworkProfile
from roboecc.segmentation import enumerate_splits, search_optimal_split


def make_model() -> ModelProfile:
    return ModelProfile(
        name="toy-vla",
        layers=(
            LayerProfile("encoder", 100.0, 10.0, 4.0, 100.0),
            LayerProfile("backbone", 100.0, 10.0, 4.0, 20.0),
            LayerProfile("action_head", 100.0, 10.0, 4.0, 1.0),
        ),
    )


class SegmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = make_model()
        self.cloud = HardwareProfile("cloud", 1_000.0, 1_000.0)
        self.edge = HardwareProfile("edge", 100.0, 1_000.0)

    def test_enumerates_endpoints_and_internal_transfers(self) -> None:
        candidates = enumerate_splits(
            self.model,
            cloud=self.cloud,
            edge=self.edge,
            network=NetworkProfile(100.0),
        )
        self.assertEqual([candidate.cut for candidate in candidates], [0, 1, 2, 3])
        self.assertEqual(candidates[0].transfer_bytes, 0.0)
        self.assertEqual(candidates[1].transfer_bytes, 100.0)
        self.assertEqual(candidates[2].transfer_bytes, 20.0)
        self.assertEqual(candidates[3].transfer_bytes, 0.0)

    def test_budget_filters_fast_all_cloud_candidate(self) -> None:
        optimal = search_optimal_split(
            self.model,
            cloud=self.cloud,
            edge=self.edge,
            network=NetworkProfile(1_000_000.0),
            max_cloud_load=8.0,
        )
        self.assertEqual(optimal.cut, 2)
        self.assertEqual(optimal.cloud_load, 8.0)

    def test_zero_budget_leaves_all_edge_feasible(self) -> None:
        candidates = enumerate_splits(
            self.model,
            cloud=self.cloud,
            edge=self.edge,
            network=NetworkProfile(1_000.0),
            max_cloud_load=0.0,
        )
        self.assertEqual(
            [candidate.cut for candidate in candidates if candidate.feasible],
            [0],
        )


if __name__ == "__main__":
    unittest.main()

