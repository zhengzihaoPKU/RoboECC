"""Parameter-sharing pool layout and safe online split activation."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

from .profiles import LayerProfile, ModelProfile


def _cut_index(value: int, field_name: str = "cut") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class PoolLayout:
    """Physical layer residency implied by a set of switchable cuts."""

    cloud_only: tuple[str, ...]
    shared: tuple[str, ...]
    edge_only: tuple[str, ...]
    allowed_cuts: tuple[int, ...]
    extra_parameter_bytes: float


class ParameterSharingPool:
    """Keep switchable boundary layers resident on both devices.

    For cloud-prefix / edge-suffix inference, every layer between the minimum
    and maximum allowed cuts may execute on either side. Those layers form the
    sharing pool and must be loaded on both devices. Changing ``active_cut`` is
    therefore a metadata operation and transfers no weights.

    The class also guards inference boundaries: a cut cannot change while an
    ``inference_session`` lease is active.
    """

    def __init__(
        self,
        model: ModelProfile,
        *,
        allowed_cuts: Sequence[int],
        initial_cut: int,
    ) -> None:
        raw_cuts = tuple(allowed_cuts)
        cuts = tuple(sorted(set(_cut_index(cut) for cut in raw_cuts)))
        if not cuts:
            raise ValueError("allowed_cuts must not be empty")
        if len(cuts) != len(raw_cuts):
            raise ValueError("allowed_cuts must not contain duplicates")
        if cuts[0] < 0 or cuts[-1] > len(model.layers):
            raise ValueError(
                f"allowed cuts must be in [0, {len(model.layers)}], got {cuts}"
            )
        initial_cut = _cut_index(initial_cut, "initial_cut")
        if initial_cut not in cuts:
            raise ValueError("initial_cut must be present in allowed_cuts")

        self.model = model
        self.allowed_cuts = cuts
        self._active_cut = int(initial_cut)
        self._active_sessions = 0
        self._lock = threading.RLock()

    @classmethod
    def around(
        cls,
        model: ModelProfile,
        *,
        center_cut: int,
        radius: int = 1,
    ) -> ParameterSharingPool:
        """Build a contiguous pool around an offline-optimal cut."""

        center_cut = _cut_index(center_cut, "center_cut")
        radius = _cut_index(radius, "radius")
        if radius < 0:
            raise ValueError("radius must be non-negative")
        if not 0 <= center_cut <= len(model.layers):
            raise ValueError(
                f"center_cut must be in [0, {len(model.layers)}], got {center_cut}"
            )
        lower = max(0, center_cut - radius)
        upper = min(len(model.layers), center_cut + radius)
        return cls(
            model,
            allowed_cuts=tuple(range(lower, upper + 1)),
            initial_cut=center_cut,
        )

    @property
    def active_cut(self) -> int:
        with self._lock:
            return self._active_cut

    @property
    def layout(self) -> PoolLayout:
        lower = self.allowed_cuts[0]
        upper = self.allowed_cuts[-1]
        shared_layers = self.model.layers[lower:upper]
        return PoolLayout(
            cloud_only=tuple(layer.name for layer in self.model.layers[:lower]),
            shared=tuple(layer.name for layer in shared_layers),
            edge_only=tuple(layer.name for layer in self.model.layers[upper:]),
            allowed_cuts=self.allowed_cuts,
            extra_parameter_bytes=sum(
                layer.parameter_bytes for layer in shared_layers
            ),
        )

    def activate(self, cut: int) -> bool:
        """Activate a resident split, returning whether it changed."""

        return self.reconfigure(cut)

    def reconfigure(
        self,
        cut: int,
        apply_cut: Callable[[int], None] | None = None,
    ) -> bool:
        """Atomically apply a deployment cut and commit the local pool state.

        The pool lock stays held while ``apply_cut`` runs. This prevents a new
        inference lease from observing the old local cut after the remote
        deployment has already switched. If the callback raises, the local cut
        remains unchanged.
        """

        cut = _cut_index(cut)
        with self._lock:
            if cut not in self.allowed_cuts:
                raise ValueError(
                    f"cut {cut} is not resident; allowed cuts: {self.allowed_cuts}"
                )
            if self._active_sessions:
                raise RuntimeError(
                    "cannot change split while an inference session is active"
                )
            changed = cut != self._active_cut
            if not changed:
                return False
            if apply_cut is not None:
                apply_cut(cut)
            self._active_cut = cut
            return True

    @contextmanager
    def inference_session(self) -> Iterator[int]:
        """Lease the current cut for one complete VLA action generation."""

        with self._lock:
            self._active_sessions += 1
            cut = self._active_cut
        try:
            yield cut
        finally:
            with self._lock:
                self._active_sessions -= 1

    def layers_for_cloud(self) -> tuple[LayerProfile, ...]:
        """Layers physically resident on cloud, including shared layers."""

        return self.model.layers[: self.allowed_cuts[-1]]

    def layers_for_edge(self) -> tuple[LayerProfile, ...]:
        """Layers physically resident on edge, including shared layers."""

        return self.model.layers[self.allowed_cuts[0] :]
