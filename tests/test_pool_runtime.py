from __future__ import annotations

import unittest

from roboecc.network import MovingAveragePredictor, NetworkAdjustmentPolicy
from roboecc.pool import ParameterSharingPool
from roboecc.profiles import HardwareProfile, LayerProfile, ModelProfile, NetworkProfile
from roboecc.runtime import DynamicSplitController, effective_bandwidth_bytes_per_second
from roboecc.segmentation import enumerate_splits


def make_model() -> ModelProfile:
    return ModelProfile(
        name="toy",
        layers=(
            LayerProfile("a", 1.0, 1.0, 10.0, 100.0),
            LayerProfile("b", 1.0, 1.0, 20.0, 50.0),
            LayerProfile("c", 1.0, 1.0, 30.0, 10.0),
            LayerProfile("d", 1.0, 1.0, 40.0, 1.0),
        ),
    )


class PoolRuntimeTests(unittest.TestCase):
    def test_layout_duplicates_only_switchable_layers(self) -> None:
        pool = ParameterSharingPool(
            make_model(), allowed_cuts=(1, 2, 3), initial_cut=2
        )
        self.assertEqual(pool.layout.cloud_only, ("a",))
        self.assertEqual(pool.layout.shared, ("b", "c"))
        self.assertEqual(pool.layout.edge_only, ("d",))
        self.assertEqual(pool.layout.extra_parameter_bytes, 50.0)

    def test_cannot_switch_during_inference(self) -> None:
        pool = ParameterSharingPool(
            make_model(), allowed_cuts=(1, 2), initial_cut=1
        )
        with pool.inference_session() as cut:
            self.assertEqual(cut, 1)
            with self.assertRaises(RuntimeError):
                pool.activate(2)
        self.assertTrue(pool.activate(2))

    def test_controller_applies_reconfiguration_after_warmup(self) -> None:
        model = make_model()
        hardware = HardwareProfile("gpu", 1.0, 1.0)
        candidates = enumerate_splits(
            model,
            cloud=hardware,
            edge=hardware,
            network=NetworkProfile(1.0),
        )
        pool = ParameterSharingPool(model, allowed_cuts=(1, 2), initial_cut=2)
        applied = []
        controller = DynamicSplitController(
            pool=pool,
            candidates=candidates,
            predictor=MovingAveragePredictor(window=2),
            policy=NetworkAdjustmentPolicy(-5.0, 5.0),
            apply_cut=applied.append,
        )
        self.assertEqual(controller.observe(30.0).status, "warming_up")
        update = controller.observe(10.0)
        self.assertEqual(update.status, "reconfigured")
        self.assertEqual(pool.active_cut, 1)
        self.assertEqual(applied, [1])

    def test_failed_callback_keeps_pool_cut(self) -> None:
        model = make_model()
        hardware = HardwareProfile("gpu", 1.0, 1.0)
        candidates = enumerate_splits(
            model,
            cloud=hardware,
            edge=hardware,
            network=NetworkProfile(1.0),
        )
        pool = ParameterSharingPool(model, allowed_cuts=(1, 2), initial_cut=2)

        def fail(_cut: int) -> None:
            raise RuntimeError("rpc failed")

        controller = DynamicSplitController(
            pool=pool,
            candidates=candidates,
            predictor=MovingAveragePredictor(window=2),
            policy=NetworkAdjustmentPolicy(-1.0, 1.0),
            apply_cut=fail,
        )
        controller.observe(30.0)
        with self.assertRaises(RuntimeError):
            controller.observe(10.0)
        self.assertEqual(pool.active_cut, 2)

    def test_reconfigure_callback_observes_old_cut_until_commit(self) -> None:
        pool = ParameterSharingPool(
            make_model(), allowed_cuts=(1, 2), initial_cut=1
        )
        observed = []
        pool.reconfigure(2, lambda _cut: observed.append(pool.active_cut))
        self.assertEqual(observed, [1])
        self.assertEqual(pool.active_cut, 2)

    def test_effective_bandwidth_removes_remote_compute(self) -> None:
        bandwidth = effective_bandwidth_bytes_per_second(
            transferred_bytes=1_000,
            round_trip_ms=12.0,
            remote_compute_ms=2.0,
        )
        self.assertEqual(bandwidth, 100_000.0)


if __name__ == "__main__":
    unittest.main()
