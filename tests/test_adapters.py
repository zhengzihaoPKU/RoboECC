from __future__ import annotations

import math
import pickle
import unittest
from types import SimpleNamespace

from roboecc.adapters import (
    CogACTSplitConfig,
    CogACTStateReconfigurator,
    DynamicOpenVLARunner,
    OpenVLAReconfigurator,
)
from roboecc.network import MovingAveragePredictor, NetworkAdjustmentPolicy
from roboecc.pool import ParameterSharingPool
from roboecc.profiles import HardwareProfile, LayerProfile, ModelProfile, NetworkProfile
from roboecc.runtime import DynamicSplitController
from roboecc.segmentation import enumerate_splits


class FakeConnection:
    def __init__(self, response):
        self.response = response
        self.sent = []

    def send(self, message):
        self.sent.append(pickle.loads(message))

    def recv(self):
        response = (
            self.response.pop(0)
            if isinstance(self.response, list)
            else self.response
        )
        return pickle.dumps(response)


class FakeSplitter:
    def __init__(self, model, num_cloud_layers):
        self.model = model
        self.num_cloud_layers = num_cloud_layers


class AdapterTests(unittest.TestCase):
    def test_cogact_global_cut_preserves_single_cut(self) -> None:
        self.assertEqual(
            CogACTSplitConfig.from_global_cut(20),
            CogACTSplitConfig(20, vlm_edge_layers=12, dit_cloud_blocks=0),
        )
        self.assertEqual(
            CogACTSplitConfig.from_global_cut(37),
            CogACTSplitConfig(37, vlm_edge_layers=0, dit_cloud_blocks=5),
        )

    def test_cogact_updates_existing_flags(self) -> None:
        target = SimpleNamespace(vlm_edge_layers=0, dit_cloud_blocks=0)
        CogACTStateReconfigurator([target])(34)
        self.assertEqual(target.vlm_edge_layers, 0)
        self.assertEqual(target.dit_cloud_blocks, 2)

    def test_cogact_rolls_back_partial_update(self) -> None:
        first = SimpleNamespace(vlm_edge_layers=4, dit_cloud_blocks=0)

        class RejectingTarget:
            vlm_edge_layers = 4

            @property
            def dit_cloud_blocks(self):
                return 0

            @dit_cloud_blocks.setter
            def dit_cloud_blocks(self, _value):
                raise RuntimeError("cannot update")

        with self.assertRaises(RuntimeError):
            CogACTStateReconfigurator([first, RejectingTarget()])(34)
        self.assertEqual(first.vlm_edge_layers, 4)
        self.assertEqual(first.dit_cloud_blocks, 0)

    def test_openvla_reconfigures_both_sides(self) -> None:
        connection = FakeConnection(
            {"status": "success", "new_cloud_layers": 17}
        )
        server = SimpleNamespace(
            num_cloud_layers=16,
            model=object(),
            splitter=FakeSplitter(object(), 16),
            thor_connection=connection,
        )
        adapter = OpenVLAReconfigurator(
            server,
            serialize_message=pickle.dumps,
            deserialize_message=pickle.loads,
            splitter_factory=FakeSplitter,
        )
        adapter(17)
        self.assertEqual(connection.sent, [{"type": "reconfigure", "num_cloud_layers": 17}])
        self.assertEqual(server.num_cloud_layers, 17)
        self.assertEqual(server.splitter.num_cloud_layers, 17)

    def test_openvla_rejection_preserves_cloud_cut(self) -> None:
        connection = FakeConnection({"status": "error", "message": "no"})
        original = FakeSplitter(object(), 16)
        server = SimpleNamespace(
            num_cloud_layers=16,
            model=object(),
            splitter=original,
            thor_connection=connection,
        )
        adapter = OpenVLAReconfigurator(
            server,
            serialize_message=pickle.dumps,
            deserialize_message=pickle.loads,
            splitter_factory=FakeSplitter,
        )
        with self.assertRaises(RuntimeError):
            adapter(17)
        self.assertEqual(server.num_cloud_layers, 16)
        self.assertIs(server.splitter, original)

    def test_openvla_unexpected_ack_rolls_edge_back(self) -> None:
        connection = FakeConnection(
            [
                {"status": "success", "new_cloud_layers": 18},
                {"status": "success", "new_cloud_layers": 16},
            ]
        )
        original = FakeSplitter(object(), 16)
        server = SimpleNamespace(
            num_cloud_layers=16,
            model=object(),
            splitter=original,
            thor_connection=connection,
        )
        adapter = OpenVLAReconfigurator(
            server,
            serialize_message=pickle.dumps,
            deserialize_message=pickle.loads,
            splitter_factory=FakeSplitter,
        )
        with self.assertRaises(RuntimeError):
            adapter(17)
        self.assertEqual(
            connection.sent,
            [
                {"type": "reconfigure", "num_cloud_layers": 17},
                {"type": "reconfigure", "num_cloud_layers": 16},
            ],
        )
        self.assertEqual(server.num_cloud_layers, 16)
        self.assertIs(server.splitter, original)

    def test_dynamic_runner_changes_only_next_action(self) -> None:
        model = ModelProfile(
            "openvla-toy",
            (
                LayerProfile("a", 1, 1, 1, 100),
                LayerProfile("b", 1, 1, 1, 10),
            ),
        )
        hardware = HardwareProfile("gpu", 1, 1)
        candidates = enumerate_splits(
            model,
            cloud=hardware,
            edge=hardware,
            network=NetworkProfile(1),
        )
        pool = ParameterSharingPool(model, allowed_cuts=(1, 2), initial_cut=2)
        applied = []
        controller = DynamicSplitController(
            pool=pool,
            candidates=candidates,
            predictor=MovingAveragePredictor(window=2),
            policy=NetworkAdjustmentPolicy(-5, 5),
            apply_cut=applied.append,
        )

        class FakeServer:
            num_cloud_layers = 2

            def __init__(self):
                self.bandwidths = iter((30.0, 10.0))

            def predict_action_split(self):
                return "action", {
                    "effective_bandwidth_bytes_per_second": next(self.bandwidths)
                }

        runner = DynamicOpenVLARunner(FakeServer(), controller)
        _, first_timing = runner.predict_action_split()
        self.assertEqual(first_timing["dynamic_split_cut_used"], 2)
        self.assertEqual(first_timing["dynamic_split_cut_next"], 2)
        _, second_timing = runner.predict_action_split()
        self.assertEqual(second_timing["dynamic_split_cut_used"], 2)
        self.assertEqual(second_timing["dynamic_split_cut_next"], 1)
        self.assertEqual(applied, [1])

    def test_dynamic_runner_rejects_non_finite_bandwidth(self) -> None:
        model = ModelProfile("toy", (LayerProfile("a", 1, 1, 1, 1),))
        hardware = HardwareProfile("gpu", 1, 1)
        candidates = enumerate_splits(
            model,
            cloud=hardware,
            edge=hardware,
            network=NetworkProfile(1),
        )
        pool = ParameterSharingPool(model, allowed_cuts=(0,), initial_cut=0)
        controller = DynamicSplitController(
            pool=pool,
            candidates=candidates,
            predictor=MovingAveragePredictor(window=1),
            policy=NetworkAdjustmentPolicy(-1, 1),
        )
        server = SimpleNamespace(
            num_cloud_layers=0,
            predict_action_split=lambda: (
                "action",
                {"effective_bandwidth_bytes_per_second": math.nan},
            ),
        )
        with self.assertRaises(ValueError):
            DynamicOpenVLARunner(server, controller).predict_action_split()


if __name__ == "__main__":
    unittest.main()
