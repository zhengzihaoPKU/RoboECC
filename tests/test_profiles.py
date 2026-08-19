from __future__ import annotations

import math
import unittest

from roboecc.profiles import HardwareProfile, LayerProfile, NetworkProfile


class ProfileTests(unittest.TestCase):
    def test_roofline_uses_slower_bound(self) -> None:
        hardware = HardwareProfile(
            name="test",
            peak_flops_per_second=1_000.0,
            memory_bandwidth_bytes_per_second=100.0,
            parallel_efficiency=0.5,
        )
        layer = LayerProfile(
            name="layer",
            compute_flops=1_000.0,
            memory_bytes=100.0,
            parameter_bytes=20.0,
            output_bytes=10.0,
        )
        self.assertEqual(hardware.estimate_layer_ms(layer), 2_000.0)

    def test_network_charges_fixed_latency_only_for_payload(self) -> None:
        network = NetworkProfile(
            bandwidth_bytes_per_second=1_000.0,
            fixed_latency_ms=2.0,
        )
        self.assertEqual(network.transfer_ms(0.0), 0.0)
        self.assertEqual(network.transfer_ms(1_000.0), 1_002.0)

    def test_invalid_profiles_fail_early(self) -> None:
        with self.assertRaises(ValueError):
            HardwareProfile(
                name="bad",
                peak_flops_per_second=0.0,
                memory_bandwidth_bytes_per_second=1.0,
            )
        with self.assertRaises(ValueError):
            LayerProfile(
                name="bad",
                compute_flops=-1.0,
                memory_bytes=0.0,
                parameter_bytes=0.0,
                output_bytes=0.0,
            )
        with self.assertRaises(ValueError):
            NetworkProfile(math.nan)


if __name__ == "__main__":
    unittest.main()
