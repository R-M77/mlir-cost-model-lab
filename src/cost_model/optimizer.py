"""Liveness-aware, legality-checked elementwise fusion planning.

This module plans a narrowly scoped optimization over a validated :class:`Graph`.
It does not execute a graph or rewrite the IR.  The plan retains source order and
reports exact model-cost changes for live operations only.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .estimator import (
    ELEMENTWISE_OPS,
    _estimate_operations,
    _intensity,
    _roofline_metrics,
    _target_report,
    estimate_graph,
)
from .models import Graph


LEGALITY_SCOPE: dict[str, Any] = {
    "allowed_elementwise_ops": ["add", "mul", "relu"],
    "requires_adjacent_source_order": True,
    "requires_single_consumer": True,
    "excludes_graph_outputs": True,
    "assumes_pure_operations": True,
    "checks_side_effects_layout_and_aliasing": False,
}


def _legality_scope_report() -> dict[str, Any]:
    return {
        **LEGALITY_SCOPE,
        "allowed_elementwise_ops": list(LEGALITY_SCOPE["allowed_elementwise_ops"]),
    }


def _live_operation_names(graph: Graph) -> set[str]:
    """Return operations that contribute to at least one graph output."""
    required_values = set(graph.outputs)
    live_names: set[str] = set()
    for operation in reversed(graph.operations):
        if operation.output in required_values:
            live_names.add(operation.name)
            required_values.update(operation.inputs)
    return live_names


def _applied_fusions(
    graph: Graph, live_names: set[str]
) -> list[dict[str, Any]]:
    consumers: Counter[str] = Counter(
        input_name
        for operation in graph.operations
        for input_name in operation.inputs
    )
    fusions: list[dict[str, Any]] = []
    for producer, consumer in zip(graph.operations, graph.operations[1:]):
        intermediate = producer.output
        if producer.name not in live_names or consumer.name not in live_names:
            continue
        if producer.kind not in ELEMENTWISE_OPS or consumer.kind not in ELEMENTWISE_OPS:
            continue
        if intermediate in graph.outputs:
            continue
        if intermediate not in consumer.inputs or consumers[intermediate] != 1:
            continue
        fusions.append(
            {
                "producer": producer.name,
                "consumer": consumer.name,
                "intermediate": intermediate,
                "bytes_avoided": 2 * graph.values[intermediate].nbytes,
                "reason": "adjacent elementwise operations with a single-use intermediate",
            }
        )
    return fusions


def _optimized_operations(
    graph: Graph,
    live_names: set[str],
    fusions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_name = {
        operation["name"]: operation for operation in _estimate_operations(graph)
    }
    read_avoided: Counter[str] = Counter()
    write_avoided: Counter[str] = Counter()
    for fusion in fusions:
        intermediate_bytes = graph.values[fusion["intermediate"]].nbytes
        read_avoided[fusion["consumer"]] += intermediate_bytes
        write_avoided[fusion["producer"]] += intermediate_bytes

    optimized: list[dict[str, Any]] = []
    for operation in graph.operations:
        if operation.name not in live_names:
            continue
        report = dict(baseline_by_name[operation.name])
        report["bytes_read"] -= read_avoided[operation.name]
        report["bytes_written"] -= write_avoided[operation.name]
        report["arithmetic_intensity"] = _intensity(
            report["flops"], report["bytes_read"], report["bytes_written"]
        )
        report["roofline"] = _roofline_metrics(
            report["flops"],
            report["bytes_read"],
            report["bytes_written"],
            graph.target,
        )
        optimized.append(report)
    return optimized


def _cost_delta(baseline: dict[str, Any], optimized: dict[str, Any]) -> dict[str, Any]:
    baseline_totals = baseline["totals"]
    optimized_totals = optimized["totals"]
    baseline_roofline = baseline_totals["roofline"]
    optimized_roofline = optimized_totals["roofline"]
    estimated_time_us = (
        None
        if baseline_roofline is None or optimized_roofline is None
        else optimized_roofline["estimated_time_us"]
        - baseline_roofline["estimated_time_us"]
    )
    return {
        "flops": optimized_totals["flops"] - baseline_totals["flops"],
        "bytes_read": optimized_totals["bytes_read"] - baseline_totals["bytes_read"],
        "bytes_written": optimized_totals["bytes_written"]
        - baseline_totals["bytes_written"],
        "logical_bytes": (
            optimized_totals["bytes_read"]
            + optimized_totals["bytes_written"]
            - baseline_totals["bytes_read"]
            - baseline_totals["bytes_written"]
        ),
        "roofline": {"estimated_time_us": estimated_time_us},
    }


def optimize_graph(graph: Graph) -> dict[str, Any]:
    """Return a deterministic before/after optimization cost report.

    The baseline is the unchanged :func:`estimate_graph` report.  The optimized
    side removes operations unreachable from ``graph.outputs`` and plans only
    adjacent, single-consumer pairs from the explicitly listed pure
    elementwise vocabulary.  No runtime execution or IR rewrite is performed.
    """
    baseline = estimate_graph(graph)
    live_names = _live_operation_names(graph)
    dead_operations = [
        operation.name
        for operation in graph.operations
        if operation.name not in live_names
    ]
    fusions = _applied_fusions(graph, live_names)
    operations = _optimized_operations(graph, live_names, fusions)

    total_flops = sum(operation["flops"] for operation in operations)
    total_bytes_read = sum(operation["bytes_read"] for operation in operations)
    total_bytes_written = sum(operation["bytes_written"] for operation in operations)
    optimized = {
        "ops": operations,
        "totals": {
            "flops": total_flops,
            "bytes_read": total_bytes_read,
            "bytes_written": total_bytes_written,
            "arithmetic_intensity": _intensity(
                total_flops, total_bytes_read, total_bytes_written
            ),
            "roofline": _roofline_metrics(
                total_flops, total_bytes_read, total_bytes_written, graph.target
            ),
        },
        "applied_fusions": fusions,
        "eliminated_dead_operations": dead_operations,
        "bytes_avoided": sum(fusion["bytes_avoided"] for fusion in fusions),
        "graph_outputs": list(graph.outputs),
        "live_operation_order": [
            operation.name
            for operation in graph.operations
            if operation.name in live_names
        ],
        "result_order": [
            operation.output
            for operation in graph.operations
            if operation.name in live_names
        ],
        "legality_scope": _legality_scope_report(),
        "target": _target_report(graph.target),
    }
    return {
        "graph": graph.name,
        "target": _target_report(graph.target),
        "baseline": baseline,
        "optimized": optimized,
        "cost_delta": _cost_delta(baseline, optimized),
    }
