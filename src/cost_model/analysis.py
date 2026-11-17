"""Structural and target-weighted analysis of validated tensor graphs."""

from __future__ import annotations

from math import isfinite
from typing import Any

from .estimator import estimate_graph
from .models import Graph
from .parser import GraphValidationError


def analyze_weighted_critical_path(graph: Graph) -> dict[str, Any]:
    """Estimate an unlimited-parallelism schedule for a validated graph.

    Durations are the shared estimator's per-operation roofline times. Inputs
    are ready at zero; each operation waits for all operands. All operations
    are reported in source order, but only declared outputs determine the
    critical path. This neither mutates the graph nor measures runtime.
    """
    if graph.target is None:
        raise GraphValidationError("weighted critical path requires a target")
    estimates = estimate_graph(graph)["ops"]
    finishes = {name: 0.0 for name in graph.inputs}
    operations: list[dict[str, Any]] = []
    for op, estimate in zip(graph.operations, estimates):
        duration = estimate["roofline"]["estimated_time_us"]
        start = max(finishes[name] for name in op.inputs)
        finish = start + duration
        if not isfinite(finish):
            raise GraphValidationError(
                "weighted critical path arithmetic exceeds finite floating-point range"
            )
        finishes[op.output] = finish
        operations.append({
            "name": op.name,
            "op": op.kind,
            "output": op.output,
            "duration_us": duration,
            "start_us": start,
            "finish_us": finish,
        })

    return {
        "graph": graph.name,
        "operation_count": len(graph.operations),
        "critical_path_time_us": max(finishes[name] for name in graph.outputs),
        "operations": operations,
        "outputs": [
            {"value": name, "finish_us": finishes[name]}
            for name in graph.outputs
        ],
    }


def analyze_liveness(graph: Graph) -> dict[str, Any]:
    """Report live tensor storage at entry and after each source operation.

    Requires a parsed, validated graph. Future uses include all operations,
    even dead ones; graph outputs are retained through exit. Unused inputs
    and results with no future use are excluded. Each distinct live value
    contributes Tensor.nbytes once. This boundary metric excludes transient
    operation allocations, aliasing, and buffer reuse; it is not device memory.
    Names are lexicographically sorted, and the earliest boundary wins peak ties
    (peak_after=None denotes entry). The graph is not mutated or optimized.
    """
    last_use = {
        name: index
        for index, op in enumerate(graph.operations)
        for name in op.inputs
    }
    last_use.update({name: len(graph.operations) for name in graph.outputs})
    live = {name: tensor.nbytes for name, tensor in graph.inputs.items() if name in last_use}
    initial_values = sorted(live)
    initial_bytes = sum(live.values())
    peak_bytes = initial_bytes
    peak_after: str | None = None
    operations: list[dict[str, Any]] = []

    for index, op in enumerate(graph.operations):
        for name in op.inputs:
            if last_use[name] == index:
                live.pop(name, None)
        if last_use.get(op.output, -1) > index:
            live[op.output] = graph.values[op.output].nbytes
        live_bytes = sum(live.values())
        operations.append({
            "name": op.name,
            "op": op.kind,
            "output": op.output,
            "live_values": sorted(live),
            "live_bytes": live_bytes,
        })
        if live_bytes > peak_bytes:
            peak_bytes = live_bytes
            peak_after = op.name

    return {
        "graph": graph.name,
        "operation_count": len(graph.operations),
        "initial_live_values": initial_values,
        "initial_live_bytes": initial_bytes,
        "peak_live_bytes": peak_bytes,
        "peak_after": peak_after,
        "operations": operations,
    }


def analyze_dependency_depth(graph: Graph) -> dict[str, Any]:
    """Report unit-cost dependency depth for a parsed, validated graph.

    Inputs start at zero; each operation adds one to its deepest operand.
    All operations are reported, but only declared outputs determine the
    critical path. This does not estimate latency or transform the graph.
    """
    depths = {name: 0 for name in graph.inputs}
    operations: list[dict[str, Any]] = []
    for op in graph.operations:
        depth = 1 + max(depths[name] for name in op.inputs)
        depths[op.output] = depth
        operations.append({
            "name": op.name,
            "op": op.kind,
            "output": op.output,
            "dependency_depth": depth,
        })

    return {
        "graph": graph.name,
        "operation_count": len(graph.operations),
        "critical_path_depth": max(depths[name] for name in graph.outputs),
        "operations": operations,
        "outputs": [
            {"value": name, "dependency_depth": depths[name]}
            for name in graph.outputs
        ],
    }
