"""Command-line entry point for inspecting a normalized RoboECC scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .profiles import HardwareProfile, ModelProfile, NetworkProfile
from .segmentation import enumerate_splits, search_optimal_split


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as error:
        raise SystemExit(f"scenario file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit("scenario root must be a JSON object")
    return value


def _format_table(candidates: tuple, optimal_cut: int) -> str:
    header = (
        "mark cut cloud_ms edge_ms transfer_ms total_ms cloud_load transfer_bytes feasible"
    )
    rows = [header]
    for candidate in candidates:
        mark = "*" if candidate.cut == optimal_cut else "-"
        rows.append(
            f"{mark:>4} {candidate.cut:>3} "
            f"{candidate.cloud_compute_ms:>8.3f} "
            f"{candidate.edge_compute_ms:>7.3f} "
            f"{candidate.transfer_ms:>11.3f} "
            f"{candidate.total_ms:>8.3f} "
            f"{candidate.cloud_load:>10.1f} "
            f"{candidate.transfer_bytes:>14.1f} "
            f"{str(candidate.feasible):>8}"
        )
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search cloud-prefix / edge-suffix RoboECC split points"
    )
    parser.add_argument("scenario", type=Path, help="normalized scenario JSON")
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit only the optimal candidate as JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenario = _load_json(args.scenario)
    try:
        model = ModelProfile.from_dict(scenario["model"])
        cloud = HardwareProfile.from_dict(scenario["hardware"]["cloud"])
        edge = HardwareProfile.from_dict(scenario["hardware"]["edge"])
        network = NetworkProfile.from_dict(scenario["network"])
        max_cloud_load = scenario.get("max_cloud_load")
        candidates = enumerate_splits(
            model,
            cloud=cloud,
            edge=edge,
            network=network,
            max_cloud_load=max_cloud_load,
        )
        optimal = search_optimal_split(
            model,
            cloud=cloud,
            edge=edge,
            network=network,
            max_cloud_load=max_cloud_load,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid scenario: {error}") from error

    if args.json:
        print(
            json.dumps(
                {
                    "model": model.name,
                    "cut": optimal.cut,
                    "cloud_layers": list(optimal.cloud_layer_names),
                    "edge_layers": list(optimal.edge_layer_names),
                    "cloud_compute_ms": optimal.cloud_compute_ms,
                    "edge_compute_ms": optimal.edge_compute_ms,
                    "transfer_ms": optimal.transfer_ms,
                    "total_ms": optimal.total_ms,
                    "cloud_load": optimal.cloud_load,
                    "transfer_bytes": optimal.transfer_bytes,
                },
                indent=2,
            )
        )
    else:
        print(f"Model: {model.name}")
        print(f"Cloud: {cloud.name}; edge: {edge.name}")
        print(_format_table(candidates, optimal.cut))
        print(f"\nOptimal cut: {optimal.cut} ({optimal.total_ms:.3f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

