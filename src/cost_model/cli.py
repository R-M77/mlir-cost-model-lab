"""Command-line interface for the cost model."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any, Sequence

from .estimator import estimate_graph
from .lowering import compare_matmul_lowering_graph
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
    except UnicodeDecodeError as exc:
        raise GraphValidationError(
            f"invalid benchmark UTF-8 at byte {exc.start}: {exc.reason}"
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
            unsupported_fields = sorted(set(raw_case) - {"name", "path"})
            if unsupported_fields:
                names = ", ".join(repr(field) for field in unsupported_fields)
                raise GraphValidationError(
                    f"benchmark[{index}] has unsupported field(s): {names}"
                )
            if "name" not in raw_case:
                raise GraphValidationError(
                    f"benchmark[{index}].name is required for object cases"
                )
            if "path" not in raw_case:
                raise GraphValidationError(
                    f"benchmark[{index}].path is required for object cases"
                )
            case_name = raw_case["name"]
            raw_graph_path = raw_case["path"]
            label = f"benchmark[{index}].path"
            if not isinstance(case_name, str) or not case_name.strip():
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


def _benchmark_report(
    benchmark_path: Path,
    *,
    optimize: bool,
    selected_case_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    seen_case_names: set[str] = set()

    for case_name, raw_graph_path, graph_path in _load_benchmark_cases(benchmark_path):
        try:
            graph = load_graph(graph_path)
        except GraphValidationError as exc:
            label = case_name or raw_graph_path
            raise GraphValidationError(f"benchmark case {label!r}: {exc}") from exc

        report = optimize_graph(graph) if optimize else estimate_graph(graph)
        effective_name = case_name or report["graph"]
        if effective_name in seen_case_names:
            raise GraphValidationError(
                f"benchmark case name {effective_name!r} must be unique"
            )
        seen_case_names.add(effective_name)
        cases.append(
            {
                "name": effective_name,
                "path": raw_graph_path,
                "report": report,
            }
        )

    if selected_case_names:
        requested_case_names = set(selected_case_names)
        available_case_names = {case["name"] for case in cases}
        unknown_case_names = sorted(requested_case_names - available_case_names)
        if unknown_case_names:
            if len(unknown_case_names) == 1:
                raise GraphValidationError(
                    f"unknown benchmark case name {unknown_case_names[0]!r}"
                )
            names = ", ".join(repr(name) for name in unknown_case_names)
            raise GraphValidationError(f"unknown benchmark case names: {names}")
        cases = [
            case for case in cases if case["name"] in requested_case_names
        ]

    total_flops = 0
    total_bytes_read = 0
    total_bytes_written = 0
    estimated_times: list[float | None] = []
    for case in cases:
        report = case["report"]
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


_BUDGET_METRICS = ("flops", "logical_bytes", "estimated_time_us")


def _parse_budget_metrics(value: object, label: str) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise GraphValidationError(f"invalid budget {label} must be an object")
    unknown_metrics = sorted(set(value) - set(_BUDGET_METRICS))
    if unknown_metrics:
        names = ", ".join(repr(metric) for metric in unknown_metrics)
        raise GraphValidationError(f"invalid budget {label} has unsupported metric(s): {names}")
    if not value:
        raise GraphValidationError(f"invalid budget {label} must define at least one metric")

    metrics: dict[str, int | float] = {}
    for metric in _BUDGET_METRICS:
        if metric not in value:
            continue
        limit = value[metric]
        if (
            isinstance(limit, bool)
            or not isinstance(limit, (int, float))
            or not isfinite(limit)
            or limit < 0
        ):
            raise GraphValidationError(
                f"invalid budget {label}.{metric} must be a finite non-negative number"
            )
        metrics[metric] = limit
    return metrics


def _load_budget(budget_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(budget_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GraphValidationError(
            f"invalid budget {budget_path}: could not read file: {exc}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise GraphValidationError(
            f"invalid budget UTF-8 at byte {exc.start}: {exc.reason}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise GraphValidationError(
            f"invalid budget JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise GraphValidationError("invalid budget must contain an object")
    allowed_keys = {"aggregate", "cases"}
    unknown_keys = sorted(set(payload) - allowed_keys)
    if unknown_keys:
        names = ", ".join(repr(key) for key in unknown_keys)
        raise GraphValidationError(f"invalid budget has unsupported field(s): {names}")
    if "aggregate" not in payload:
        raise GraphValidationError("invalid budget requires an aggregate object")

    aggregate = _parse_budget_metrics(payload["aggregate"], "aggregate")
    raw_cases = payload.get("cases", {})
    if not isinstance(raw_cases, Mapping):
        raise GraphValidationError("invalid budget cases must be an object")

    cases: dict[str, dict[str, int | float]] = {}
    for case_name, limits in raw_cases.items():
        if not isinstance(case_name, str) or not case_name.strip():
            raise GraphValidationError("invalid budget case names must be non-empty strings")
        cases[case_name] = _parse_budget_metrics(limits, f"cases[{case_name!r}]")
    return {"aggregate": aggregate, "cases": cases}


def _selected_case_totals(
    case_report: Mapping[str, Any], *, optimize: bool, label: str
) -> Mapping[str, Any]:
    report: object = case_report.get("optimized") if optimize else case_report
    if not isinstance(report, Mapping) or not isinstance(report.get("totals"), Mapping):
        raise GraphValidationError(f"budget {label} totals are missing from the benchmark report")
    return report["totals"]


def _metric_value(totals: Mapping[str, Any], metric: str, label: str) -> int | float:
    actual: object
    if metric == "logical_bytes":
        actual = totals.get("logical_bytes")
        if actual is None:
            bytes_read = totals.get("bytes_read")
            bytes_written = totals.get("bytes_written")
            if (
                isinstance(bytes_read, bool)
                or not isinstance(bytes_read, (int, float))
                or isinstance(bytes_written, bool)
                or not isinstance(bytes_written, (int, float))
            ):
                actual = None
            else:
                actual = bytes_read + bytes_written
    elif metric == "estimated_time_us" and "estimated_time_us" not in totals:
        roofline = totals.get("roofline")
        actual = roofline.get("estimated_time_us") if isinstance(roofline, Mapping) else None
    else:
        actual = totals.get(metric)

    if (
        actual is None
        or isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not isfinite(actual)
    ):
        raise GraphValidationError(f"budget {label}.{metric} is unavailable in the benchmark report")
    return actual


def _check_budget(
    report: Mapping[str, Any], budget: Mapping[str, Any], *, optimize: bool
) -> list[dict[str, Any]]:
    summary = report.get("summary")
    if not isinstance(summary, Mapping) or not isinstance(summary.get("totals"), Mapping):
        raise GraphValidationError("budget summary totals are missing from the benchmark report")

    violations: list[dict[str, Any]] = []

    def check_scope(
        scope: str,
        limits: Mapping[str, int | float],
        totals: Mapping[str, Any],
        label: str,
    ) -> None:
        for metric in _BUDGET_METRICS:
            if metric not in limits:
                continue
            actual = _metric_value(totals, metric, label)
            limit = limits[metric]
            if actual > limit:
                violations.append(
                    {
                        "actual": actual,
                        "limit": limit,
                        "metric": metric,
                        "scope": scope,
                    }
                )

    check_scope(
        "aggregate",
        budget["aggregate"],
        summary["totals"],
        "aggregate",
    )

    raw_case_limits = budget["cases"]
    if not raw_case_limits:
        return violations
    raw_cases = report.get("cases")
    if not isinstance(raw_cases, list):
        raise GraphValidationError("budget case reports are missing from the benchmark report")

    for case_name in sorted(raw_case_limits):
        matching_cases = [
            case
            for case in raw_cases
            if isinstance(case, Mapping) and case.get("name") == case_name
        ]
        if len(matching_cases) != 1:
            raise GraphValidationError(
                f"invalid budget case {case_name!r} must match exactly one benchmark case"
            )
        case_report = matching_cases[0].get("report")
        if not isinstance(case_report, Mapping):
            raise GraphValidationError(
                f"budget case {case_name!r} report is missing from the benchmark output"
            )
        check_scope(
            "case",
            raw_case_limits[case_name],
            _selected_case_totals(
                case_report,
                optimize=optimize,
                label=f"cases[{case_name!r}]",
            ),
            f"cases[{case_name!r}]",
        )
    return violations


def _budget_diagnostic(violations: list[dict[str, Any]]) -> str:
    return json.dumps(
        {"error": "benchmark budget exceeded", "violations": violations},
        separators=(",", ":"),
        sort_keys=True,
    )


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
        "--omit-paths",
        action="store_true",
        help="omit case input paths from benchmark JSON output",
    )
    parser.add_argument(
        "--case",
        dest="case_names",
        action="append",
        metavar="NAME",
        help="select a benchmark case by effective name; repeat to select multiple cases",
    )
    parser.add_argument(
        "--budget",
        type=Path,
        metavar="JSON",
        help="check benchmark aggregate and optional per-case metric budgets",
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
    parser.add_argument(
        "--lower-matmul",
        action="store_true",
        help="emit an opt-in before/after matmul lowering report",
    )
    parser.add_argument(
        "--lowering-mode",
        choices=("scalar", "tiled"),
        default="scalar",
        help="matmul lowering representation to compare",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=16,
        help="tiled lowering M/N/K tile extent (default: 16)",
    )
    parser.add_argument(
        "--vector-width",
        type=int,
        default=4,
        help="tiled lowering vector lane width (default: 4)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.case_names and args.benchmark is None:
        parser.error("--case requires --benchmark JSON")
    if args.omit_paths and args.benchmark is None:
        parser.error("--omit-paths requires --benchmark JSON")
    if args.lower_matmul and args.benchmark is not None:
        parser.error("--lower-matmul requires a single graph path")
    if args.lower_matmul and args.optimize:
        parser.error("--lower-matmul cannot be combined with --optimize")
    if not args.lower_matmul and args.lowering_mode != "scalar":
        parser.error("--lowering-mode requires --lower-matmul")
    if args.tile_size <= 0 or args.vector_width <= 0:
        parser.error("--tile-size and --vector-width must be positive")
    if args.graph is None and args.benchmark is None:
        parser.error("a graph path or --benchmark JSON is required")
    if args.graph is not None and args.benchmark is not None:
        parser.error("graph path and --benchmark JSON are mutually exclusive")
    if args.budget is not None and args.benchmark is None:
        parser.error("--budget requires --benchmark JSON")

    violations: list[dict[str, Any]] = []
    try:
        budget = _load_budget(args.budget) if args.budget is not None else None
        if args.benchmark is not None:
            report = _benchmark_report(
                args.benchmark,
                optimize=args.optimize,
                selected_case_names=args.case_names,
            )
            if budget is not None:
                violations = _check_budget(report, budget, optimize=args.optimize)
        else:
            assert args.graph is not None
            graph = load_graph(args.graph)
            if args.lower_matmul:
                report = compare_matmul_lowering_graph(
                    graph,
                    mode=args.lowering_mode,
                    tile_size=args.tile_size,
                    vector_width=args.vector_width,
                )
            else:
                report = optimize_graph(graph) if args.optimize else estimate_graph(graph)
    except GraphValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_report = report
    if args.omit_paths:
        output_report = {
            **report,
            "cases": [
                {key: value for key, value in case.items() if key != "path"}
                for case in report["cases"]
            ],
        }

    if args.compact:
        print(json.dumps(output_report, separators=(",", ":")))
    else:
        print(json.dumps(output_report, indent=2))
    if violations:
        print(f"error: {_budget_diagnostic(violations)}", file=sys.stderr)
        return 2
    return 0
