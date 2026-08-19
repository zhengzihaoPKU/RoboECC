"""Small many-to-one LSTM used for next-step bandwidth forecasting."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _require_positive_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class LSTMConfig:
    sequence_length: int = 20
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0

    def __post_init__(self) -> None:
        _require_positive_integer(self.sequence_length, "sequence_length")
        _require_positive_integer(self.hidden_size, "hidden_size")
        _require_positive_integer(self.num_layers, "num_layers")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    patience: int = 20
    seed: int = 7

    def __post_init__(self) -> None:
        _require_positive_integer(self.epochs, "epochs")
        _require_positive_integer(self.batch_size, "batch_size")
        _require_positive_integer(self.patience, "patience")
        if (
            not math.isfinite(self.learning_rate)
            or not math.isfinite(self.weight_decay)
            or self.learning_rate <= 0
            or self.weight_decay < 0
        ):
            raise ValueError("invalid optimizer configuration")
        if (
            not math.isfinite(self.validation_fraction)
            or not 0 < self.validation_fraction < 0.5
        ):
            raise ValueError("validation_fraction must be in (0, 0.5)")


@dataclass(frozen=True, slots=True)
class Standardizer:
    mean: float
    std: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean):
            raise ValueError("scaler mean must be finite")
        if not math.isfinite(self.std) or self.std <= 0:
            raise ValueError("scaler std must be finite and positive")

    @classmethod
    def fit(cls, values: Sequence[float]) -> Standardizer:
        tensor = torch.as_tensor(values, dtype=torch.float64)
        if tensor.numel() < 2:
            raise ValueError("at least two values are required to fit the scaler")
        if not torch.isfinite(tensor).all() or torch.any(tensor <= 0):
            raise ValueError("bandwidth values must be finite and positive")
        std = float(tensor.std(unbiased=False).item())
        return cls(mean=float(tensor.mean().item()), std=max(std, 1e-8))

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std + self.mean


class BandwidthLSTM(nn.Module):
    """Unidirectional LSTM followed by a linear next-value head."""

    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.config = config
        effective_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Linear(config.hidden_size, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim == 2:
            sequence = sequence.unsqueeze(-1)
        if sequence.ndim != 3 or sequence.shape[-1] != 1:
            raise ValueError("expected input shape [batch, sequence, 1]")
        output, _ = self.lstm(sequence)
        return self.head(output[:, -1, :]).squeeze(-1)


@dataclass(slots=True)
class TrainedBandwidthPredictor:
    model: BandwidthLSTM
    scaler: Standardizer
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.model.to(self.device)
        self.model.eval()

    @property
    def minimum_history(self) -> int:
        return self.model.config.sequence_length

    @torch.inference_mode()
    def predict(self, history: Sequence[float]) -> float:
        if len(history) < self.minimum_history:
            raise ValueError(
                f"at least {self.minimum_history} bandwidth samples are required"
            )
        values = torch.tensor(
            history[-self.minimum_history :],
            dtype=torch.float32,
            device=self.device,
        )
        if not torch.isfinite(values).all() or torch.any(values <= 0):
            raise ValueError("bandwidth history must contain finite positive values")
        prediction = self.model(self.scaler.normalize(values).unsqueeze(0))
        bandwidth = float(self.scaler.denormalize(prediction).item())
        if not math.isfinite(bandwidth):
            raise RuntimeError("bandwidth predictor produced a non-finite value")
        return max(bandwidth, 1e-8)

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        checkpoint = {
            "format_version": 1,
            "architecture": "unidirectional_many_to_one_lstm",
            "model_config": asdict(self.model.config),
            "state_dict": self.model.state_dict(),
            "scaler": asdict(self.scaler),
            "metadata": metadata or {},
        }
        torch.save(checkpoint, Path(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
    ) -> TrainedBandwidthPredictor:
        checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported bandwidth checkpoint format")
        model = BandwidthLSTM(LSTMConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["state_dict"])
        return cls(
            model=model,
            scaler=Standardizer(**checkpoint["scaler"]),
            device=device,
        )


@dataclass(frozen=True, slots=True)
class TrainingResult:
    predictor: TrainedBandwidthPredictor
    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    best_epoch: int


def _windows(values: torch.Tensor, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = []
    targets = []
    for index in range(values.numel() - sequence_length):
        inputs.append(values[index : index + sequence_length])
        targets.append(values[index + sequence_length])
    return torch.stack(inputs), torch.stack(targets)


def train_predictor(
    values: Sequence[float],
    *,
    model_config: LSTMConfig | None = None,
    training_config: TrainingConfig | None = None,
    device: str = "cpu",
) -> TrainingResult:
    """Train with a chronological train/validation split and early stopping."""

    model_config = model_config or LSTMConfig()
    training_config = training_config or TrainingConfig()
    raw = torch.as_tensor(values, dtype=torch.float32)
    minimum_values = model_config.sequence_length + 5
    if raw.numel() < minimum_values:
        raise ValueError(f"at least {minimum_values} bandwidth values are required")
    if not torch.isfinite(raw).all() or torch.any(raw <= 0):
        raise ValueError("bandwidth values must be finite and positive")

    window_count = raw.numel() - model_config.sequence_length
    validation_count = max(1, int(window_count * training_config.validation_fraction))
    train_count = window_count - validation_count
    if train_count < 2:
        raise ValueError("not enough training windows after validation split")

    scaler = Standardizer.fit(
        raw[: train_count + model_config.sequence_length].tolist()
    )
    normalized = scaler.normalize(raw)
    inputs, targets = _windows(normalized, model_config.sequence_length)
    train_inputs, validation_inputs = inputs[:train_count], inputs[train_count:]
    train_targets, validation_targets = targets[:train_count], targets[train_count:]

    torch.manual_seed(training_config.seed)
    generator = torch.Generator().manual_seed(training_config.seed)
    loader = DataLoader(
        TensorDataset(train_inputs, train_targets),
        batch_size=min(training_config.batch_size, train_count),
        shuffle=True,
        generator=generator,
    )
    model = BandwidthLSTM(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    loss_function = nn.MSELoss()

    train_history: list[float] = []
    validation_history: list[float] = []
    best_loss = math.inf
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0

    for epoch in range(training_config.epochs):
        model.train()
        epoch_loss = 0.0
        sample_count = 0
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_inputs)
            loss = loss_function(prediction, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item()) * batch_inputs.shape[0]
            sample_count += batch_inputs.shape[0]
        train_history.append(epoch_loss / sample_count)

        model.eval()
        with torch.inference_mode():
            validation_prediction = model(validation_inputs.to(device))
            validation_loss = float(
                loss_function(
                    validation_prediction,
                    validation_targets.to(device),
                ).item()
            )
        validation_history.append(validation_loss)

        if not math.isfinite(validation_loss):
            raise RuntimeError("training produced a non-finite validation loss")

        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= training_config.patience:
                break

    model.load_state_dict(best_state)
    return TrainingResult(
        predictor=TrainedBandwidthPredictor(model=model, scaler=scaler, device=device),
        train_loss=tuple(train_history),
        validation_loss=tuple(validation_history),
        best_epoch=best_epoch,
    )
