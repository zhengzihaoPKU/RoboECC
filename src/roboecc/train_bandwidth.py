"""Train a RoboECC bandwidth LSTM from a CSV trace or synthetic smoke data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import torch

from .bandwidth_lstm import LSTMConfig, TrainingConfig, train_predictor


def _read_csv(path: Path, column: str) -> list[float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"CSV must contain column {column!r}")
        values = [float(row[column]) for row in reader if row[column].strip()]
    return values


def _synthetic_trace(samples: int, seed: int) -> list[float]:
    if samples < 30:
        raise ValueError("synthetic trace requires at least 30 samples")
    random_source = random.Random(seed)
    values = []
    for index in range(samples):
        baseline = 80.0 + 18.0 * math.sin(index / 24.0)
        short_wave = 7.0 * math.sin(index / 5.0)
        dip = -35.0 if index % 137 in range(0, 8) else 0.0
        values.append(max(2.0, baseline + short_wave + dip + random_source.gauss(0, 2)))
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train next-step bandwidth LSTM")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="time-ordered bandwidth trace")
    source.add_argument(
        "--synthetic-samples",
        type=int,
        help="generate smoke-test data; do not use for paper results",
    )
    parser.add_argument("--column", default="bandwidth_bytes_per_second")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        values = (
            _read_csv(args.csv, args.column)
            if args.csv is not None
            else _synthetic_trace(args.synthetic_samples, args.seed)
        )
        device = (
            "cuda"
            if args.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if args.device == "auto"
            else args.device
        )
        model_config = LSTMConfig(
            sequence_length=args.sequence_length,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
        training_config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            seed=args.seed,
        )
        result = train_predictor(
            values,
            model_config=model_config,
            training_config=training_config,
            device=device,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"training failed: {error}") from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.predictor.save(
        args.output,
        metadata={
            "source": str(args.csv) if args.csv else "synthetic_smoke_trace",
            "bandwidth_unit": args.column,
            "training_config": asdict(training_config),
            "sample_count": len(values),
        },
    )
    summary = {
        "checkpoint": str(args.output),
        "architecture": "single-direction many-to-one LSTM + linear head",
        "sequence_length": model_config.sequence_length,
        "hidden_size": model_config.hidden_size,
        "num_layers": model_config.num_layers,
        "best_epoch": result.best_epoch + 1,
        "best_validation_mse_normalized": min(result.validation_loss),
        "epochs_ran": len(result.train_loss),
        "device": device,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
