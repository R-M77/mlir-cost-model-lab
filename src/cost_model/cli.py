"""Command-line interface for the cost model."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

from .estimator import estimate_graph
from .optimizer import optimize_graph
from .parser import GraphValidationError, load_graph


def _load_benchmark_cases(
    benchmark_path: Path,
) -> list[tuple[str | None, str, Path]]:
    """Load benchmark case names and graph paths from a JSON array."""
    try:
        payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GraphValidationError(
            f"could not read benchmark {benchmark_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise GraphValidationError(
            f"invalid benchmark JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
        ) from exc

    if not isinstance(payload, list):
        raise GraphValidationError("benchmark JSON must contain an array")
    if not payload:
        raise GraphValidationError("benchmark JSON must contain at least one case")

    cases: list[tuple[str | None, str, Path]] = []
    for index, raw_case in enumerate(payload):
        case_name: str | None = None
        raw_graph_path: Any = raw_case
        label = f"benchmark[{index}]"
        if isinstance(raw_case, Mapping):
            case_name = raw_case.get("name")
            raw_graph_path = raw_case.get("path")
            label = f"benchmark[{index}].path"
            if case_name is not None and (
                not isinstance(case_name, str) or not case_name.strip()
            ):
                raise GraphValidationError(
                    f"benchmark[{index}].name must be a non-empty string"
                )

        if not isinstance(raw_graph_path, str) or not raw_graph_path.strip():
            raise GraphValidationError(f"{label} must be a non-empty string")

        graph_path = Path(raw_graph_path)
        if not graph_path.is_absolute():
            graph_path = benchmark_path.parent / graph_path
        cases.append((case_name, raw_graph_path, graph_path))
    return cases


def _benchmark_report(benchmark_path: Path, *, optimize: bool) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    total_flops = 0
    total_bytes_read = 0
    total_bytes_written = 0
    estimated_times: list[float | None] = []

    for case_name, raw_graph_path, graph_path in _load_benchmark_cases(benchmark_path):
        try:
            graph = load_graph(graph_path)
        except GraphValidationError as exc:
            label = case_name or raw_graph_path
            raise GraphValidationError(f"benchmark case {label!r}: {exc}") from exc

        report = optimize_graph(graph) if optimize else estimate_graph(graph)
        cases.append(
            {
                "name": case_name or report["graph"],
                "path": raw_graph_path,
                "report": report,
            }
        )
        totals = report["optimized"]["totals"] if optimize else report["totals"]
        total_flops += totals["flops"]
        total_bytes_read += totals["bytes_read"]
        total_bytes_written += totals["bytes_written"]
        roofline = totals["roofline"]
        estimated_times.append(
            None if roofline is None else roofline["estimated_time_us"]
        )

    logical_bytes = total_bytes_read + total_bytes_written
    summary_totals: dict[str, Any] = {
        "flops": total_flops,
        "bytes_read": total_bytes_read,
        "bytes_written": total_bytes_written,
        "logical_bytes": logical_bytes,
        "arithmetic_intensity": (
            total_flops / logical_bytes if logical_bytes else 0.0
        ),
        "estimated_time_us": (
            sum(time for time in estimated_times if time is not None)
            if all(time is not None for time in estimated_times)
            else None
        ),
    }
    return {
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "totals": summary_totals,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate deterministic costs for an MLIR-inspired tensor graph; "
            "include an input target for roofline lower bounds."
        )
    )
    parser.add_argument(
        "graph",
        nargs="?",
        type=Path,
        help="path to a graph JSON file (omit when using --benchmark)",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        metavar="JSON",
        help="run a batch from a JSON array of graph paths or named cases",
    )
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
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.graph is None and args.benchmark is None:
        parser.error("a graph path or --benchmark JSON is required")
    if args.graph is not None and args.benchmark is not None:
        parser.error("graph path and --benchmark JSON are mutually exclusive")

    try:
        if args.benchmark is not None:
            report = _benchmark_report(args.benchmark, optimize=args.optimize)
        else:
            assert args.graph is not None
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
