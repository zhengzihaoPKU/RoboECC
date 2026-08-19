"""Thin adapters for the existing OpenVLA and CogACT split implementations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from .runtime import DynamicSplitController, RuntimeUpdate


def _integer(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


class SyncConnection(Protocol):
    def send(self, message: bytes) -> None: ...

    def recv(self) -> bytes: ...


@dataclass(frozen=True, slots=True)
class CogACTSplitConfig:
    """CogACT flags derived from one global cloud-prefix cut.

    This mapping assumes the revised SimplerEnv execution order: VLM cloud
    prefix, then VLM edge suffix; DiT cloud prefix, then DiT edge suffix. The
    older standalone ``CogACT/split_inference/edge_client.py`` uses the reverse
    order for a partially split DiT and is therefore not a compatible target.
    """

    global_cut: int
    vlm_edge_layers: int
    dit_cloud_blocks: int

    @classmethod
    def from_global_cut(
        cls,
        cut: int,
        *,
        total_vlm_layers: int = 32,
        total_dit_blocks: int = 12,
    ) -> CogACTSplitConfig:
        cut = _integer(cut, "cut")
        total_vlm_layers = _integer(total_vlm_layers, "total_vlm_layers")
        total_dit_blocks = _integer(total_dit_blocks, "total_dit_blocks")
        if total_vlm_layers <= 0 or total_dit_blocks <= 0:
            raise ValueError("CogACT layer and block counts must be positive")
        total = total_vlm_layers + total_dit_blocks
        if not 0 <= cut <= total:
            raise ValueError(f"CogACT cut must be in [0, {total}], got {cut}")
        if cut <= total_vlm_layers:
            return cls(
                global_cut=cut,
                vlm_edge_layers=total_vlm_layers - cut,
                dit_cloud_blocks=0,
            )
        return cls(
            global_cut=cut,
            vlm_edge_layers=0,
            dit_cloud_blocks=cut - total_vlm_layers,
        )


class CogACTStateReconfigurator:
    """Update existing CogACT policy/client objects at an action boundary.

    Every target must already own the parameters and splitter objects required
    by every allowed cut. This adapter updates flags only; it intentionally does
    not load missing model components at runtime.
    """

    def __init__(
        self,
        targets: Sequence[Any],
        *,
        total_vlm_layers: int = 32,
        total_dit_blocks: int = 12,
    ) -> None:
        if not targets:
            raise ValueError("at least one CogACT target is required")
        self.targets = tuple(targets)
        _ = CogACTSplitConfig.from_global_cut(
            0,
            total_vlm_layers=total_vlm_layers,
            total_dit_blocks=total_dit_blocks,
        )
        self.total_vlm_layers = total_vlm_layers
        self.total_dit_blocks = total_dit_blocks

    def __call__(self, cut: int) -> None:
        config = CogACTSplitConfig.from_global_cut(
            cut,
            total_vlm_layers=self.total_vlm_layers,
            total_dit_blocks=self.total_dit_blocks,
        )
        for target in self.targets:
            if not hasattr(target, "vlm_edge_layers") or not hasattr(
                target, "dit_cloud_blocks"
            ):
                raise TypeError(
                    "CogACT target must expose vlm_edge_layers and dit_cloud_blocks"
                )
        previous = [
            (target, target.vlm_edge_layers, target.dit_cloud_blocks)
            for target in self.targets
        ]
        try:
            for target, _, _ in previous:
                target.vlm_edge_layers = config.vlm_edge_layers
                target.dit_cloud_blocks = config.dit_cloud_blocks
        except Exception as error:
            rollback_errors = []
            for target, vlm_edge_layers, dit_cloud_blocks in previous:
                try:
                    target.vlm_edge_layers = vlm_edge_layers
                    target.dit_cloud_blocks = dit_cloud_blocks
                except Exception as rollback_error:
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                raise RuntimeError(
                    "CogACT flag update failed and at least one target could not "
                    "be rolled back; stop inference and rebuild all targets"
                ) from error
            raise


class OpenVLAReconfigurator:
    """Align the current cloud server and Thor edge client cut.

    This adapter matches ``openvla/split_inference/cloud_server.py`` and the
    existing edge client's ``reconfigure`` message. The cloud splitter is built
    before the edge RPC, so construction failure cannot leave the two devices
    on different cuts. A failed or ambiguous RPC triggers a best-effort rollback
    to the previous edge cut before the error is propagated.
    """

    def __init__(
        self,
        server: Any,
        *,
        serialize_message: Callable[[dict[str, Any]], bytes],
        deserialize_message: Callable[[bytes], dict[str, Any]],
        splitter_factory: Callable[..., Any] | None = None,
        total_layers: int = 32,
    ) -> None:
        self.server = server
        self.serialize_message = serialize_message
        self.deserialize_message = deserialize_message
        self.splitter_factory = splitter_factory
        self.total_layers = _integer(total_layers, "total_layers")
        if self.total_layers <= 0:
            raise ValueError("total_layers must be positive")

    def _request_remote_cut(self, connection: SyncConnection, cut: int) -> None:
        connection.send(
            self.serialize_message({"type": "reconfigure", "num_cloud_layers": cut})
        )
        response = self.deserialize_message(connection.recv())
        if response.get("status") != "success":
            raise RuntimeError(
                f"Thor rejected split {cut}: {response.get('message', response)}"
            )
        acknowledged_cut = response.get("new_cloud_layers")
        if acknowledged_cut is not None and _integer(
            acknowledged_cut, "acknowledged cut"
        ) != cut:
            raise RuntimeError(
                f"Thor acknowledged unexpected cut {acknowledged_cut}, expected {cut}"
            )

    def __call__(self, cut: int) -> None:
        cut = _integer(cut, "cut")
        if not 0 <= cut <= self.total_layers:
            raise ValueError(
                f"OpenVLA cut must be in [0, {self.total_layers}], got {cut}"
            )
        if cut == self.server.num_cloud_layers:
            return
        if self.server.model is None or self.server.splitter is None:
            raise RuntimeError("OpenVLA cloud model must be loaded before reconfiguration")
        connection: SyncConnection | None = self.server.thor_connection
        if connection is None:
            raise RuntimeError("Thor must be connected before reconfiguration")

        factory = self.splitter_factory or type(self.server.splitter)
        new_splitter = factory(self.server.model, num_cloud_layers=cut)
        previous_cut = _integer(self.server.num_cloud_layers, "server cut")
        try:
            self._request_remote_cut(connection, cut)
        except Exception as error:
            try:
                self._request_remote_cut(connection, previous_cut)
            except Exception:
                raise RuntimeError(
                    "OpenVLA reconfiguration failed and the Thor cut could not be "
                    "reconciled; stop inference and reconnect both sides"
                ) from error
            raise

        self.server.splitter = new_splitter
        self.server.num_cloud_layers = cut


class DynamicOpenVLARunner:
    """Run one OpenVLA action at a stable cut, then tune the next action."""

    def __init__(self, server: Any, controller: DynamicSplitController) -> None:
        self.server = server
        self.controller = controller

    def predict_action_split(self, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        with self.controller.pool.inference_session() as cut:
            server_cut = _integer(self.server.num_cloud_layers, "server cut")
            if server_cut != cut:
                if self.controller.apply_cut is None:
                    raise RuntimeError(
                        "server cut differs from pool and no reconfigurator is installed"
                    )
                self.controller.apply_cut(cut)
            action, timing = self.server.predict_action_split(*args, **kwargs)

        bandwidth = float(
            timing.get(
                "effective_bandwidth_bytes_per_second",
                timing.get("effective_bandwidth_bps", 0.0),
            )
        )
        if not math.isfinite(bandwidth) or bandwidth < 0:
            raise ValueError(
                "effective bandwidth timing must be finite and non-negative"
            )
        update: RuntimeUpdate | None = None
        if bandwidth > 0:
            update = self.controller.observe(bandwidth)
        timing["dynamic_split_cut_used"] = cut
        timing["dynamic_split_cut_next"] = self.controller.pool.active_cut
        timing["dynamic_split_status"] = (
            update.status if update is not None else "no_network_observation"
        )
        if update is not None and update.predicted_bandwidth is not None:
            timing["predicted_bandwidth_bytes_per_second"] = (
                update.predicted_bandwidth
            )
        return action, timing
