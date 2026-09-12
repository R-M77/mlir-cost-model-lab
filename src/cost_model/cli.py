"""Command-line interface for the cost model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .estimator import estimate_graph
from .optimizer import optimize_graph
from .parser import GraphValidationError, load_graph


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate deterministic costs for an MLIR-inspired tensor graph; "
            "include an input target for roofline lower bounds."
        )
    )
    parser.add_argument("graph", type=Path, help="path to a graph JSON file")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="emit one-line JSON instead of the default readable form",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="emit an opt-in before/after liveness-aware optimization report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        graph = load_graph(args.graph)
        report = optimize_graph(graph) if args.optimize else estimate_graph(graph)
    except GraphValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.compact:
        print(json.dumps(report, separators=(",", ":")))
    else:
        print(json.dumps(report, indent=2))
    return 0
