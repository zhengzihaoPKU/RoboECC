"""Validated, framework-independent profiles used by the RoboECC planner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


def _positive(value: float, field_name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be finite and greater than zero, got {value}")
    return value


def _non_negative(value: float, field_name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class LayerProfile:
    """Cost and boundary data for one sequential model layer.

    ``output_bytes`` is the activation size transmitted when the split is placed
    immediately after this layer. ``cloud_load`` is an application-defined load
    unit used by the cloud budget; it defaults to parameter bytes.
    """

    name: str
    compute_flops: float
    memory_bytes: float
    parameter_bytes: float
    output_bytes: float
    cloud_load: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("layer name must not be empty")
        for field_name in (
            "compute_flops",
            "memory_bytes",
            "parameter_bytes",
            "output_bytes",
        ):
            object.__setattr__(
                self,
                field_name,
                _non_negative(getattr(self, field_name), field_name),
            )
        if self.cloud_load is None:
            object.__setattr__(self, "cloud_load", self.parameter_bytes)
        else:
            object.__setattr__(
                self,
                "cloud_load",
                _non_negative(self.cloud_load, "cloud_load"),
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LayerProfile:
        return cls(
            name=str(value["name"]),
            compute_flops=float(value["compute_flops"]),
            memory_bytes=float(value["memory_bytes"]),
            parameter_bytes=float(value.get("parameter_bytes", 0.0)),
            output_bytes=float(value["output_bytes"]),
            cloud_load=(
                None if value.get("cloud_load") is None else float(value["cloud_load"])
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """An ordered VLA model represented as a single sequential chain."""

    name: str
    layers: tuple[LayerProfile, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("model name must not be empty")
        if not self.layers:
            raise ValueError("model profile must contain at least one layer")
        names = [layer.name for layer in self.layers]
        if len(names) != len(set(names)):
            raise ValueError("layer names must be unique")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelProfile:
        return cls(
            name=str(value["name"]),
            layers=tuple(LayerProfile.from_dict(layer) for layer in value["layers"]),
        )


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """GPU roofline inputs used to estimate per-layer execution latency."""

    name: str
    peak_flops_per_second: float
    memory_bandwidth_bytes_per_second: float
    parallel_efficiency: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("hardware name must not be empty")
        object.__setattr__(
            self,
            "peak_flops_per_second",
            _positive(self.peak_flops_per_second, "peak_flops_per_second"),
        )
        object.__setattr__(
            self,
            "memory_bandwidth_bytes_per_second",
            _positive(
                self.memory_bandwidth_bytes_per_second,
                "memory_bandwidth_bytes_per_second",
            ),
        )
        efficiency = float(self.parallel_efficiency)
        if not math.isfinite(efficiency) or not 0 < efficiency <= 1:
            raise ValueError(
                f"parallel_efficiency must be in (0, 1], got {efficiency}"
            )
        object.__setattr__(self, "parallel_efficiency", efficiency)

    def estimate_layer_ms(self, layer: LayerProfile) -> float:
        """Estimate latency with the roofline max(compute time, memory time)."""

        compute_seconds = layer.compute_flops / (
            self.peak_flops_per_second * self.parallel_efficiency
        )
        memory_seconds = (
            layer.memory_bytes / self.memory_bandwidth_bytes_per_second
        )
        return max(compute_seconds, memory_seconds) * 1_000.0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HardwareProfile:
        return cls(
            name=str(value["name"]),
            peak_flops_per_second=float(value["peak_flops_per_second"]),
            memory_bandwidth_bytes_per_second=float(
                value["memory_bandwidth_bytes_per_second"]
            ),
            parallel_efficiency=float(value.get("parallel_efficiency", 1.0)),
        )


@dataclass(frozen=True, slots=True)
class NetworkProfile:
    """One-way boundary-transfer model.

    Fixed link latency is charged only when an intermediate activation is sent.
    The two endpoint deployments (all-edge and all-cloud) intentionally have no
    boundary-transfer charge, matching the existing RoboECC sweep convention.
    """

    bandwidth_bytes_per_second: float
    fixed_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bandwidth_bytes_per_second",
            _positive(
                self.bandwidth_bytes_per_second,
                "bandwidth_bytes_per_second",
            ),
        )
        object.__setattr__(
            self,
            "fixed_latency_ms",
            _non_negative(self.fixed_latency_ms, "fixed_latency_ms"),
        )

    def transfer_ms(self, size_bytes: float) -> float:
        size_bytes = _non_negative(size_bytes, "size_bytes")
        if size_bytes == 0:
            return 0.0
        return self.fixed_latency_ms + (
            size_bytes / self.bandwidth_bytes_per_second * 1_000.0
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NetworkProfile:
        return cls(
            bandwidth_bytes_per_second=float(value["bandwidth_bytes_per_second"]),
            fixed_latency_ms=float(value.get("fixed_latency_ms", 0.0)),
        )


def total_cloud_load(layers: Iterable[LayerProfile]) -> float:
    """Return the additive cloud load for a layer collection."""

    return sum(float(layer.cloud_load) for layer in layers)
