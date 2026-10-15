"""Deterministic cost estimation and simple fusion suggestions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from .models import Graph, Operation, Target
from .parser import parse_graph

ELEMENTWISE_OPS = frozenset({"add", "mul", "relu"})


def _intensity(flops: int, bytes_read: int, bytes_written: int) -> float:
    traffic = bytes_read + bytes_written
    return flops / traffic if traffic else 0.0


def _roofline_metrics(
    flops: int,
    bytes_read: int,
    bytes_written: int,
    target: Target | None,
) -> dict[str, Any] | None:
    """Return idealized roofline metrics, or null when no target is supplied."""
    if target is None:
        return None

    logical_bytes = bytes_read + bytes_written
    arithmetic_intensity = _intensity(flops, bytes_read, bytes_written)
    ridge_point = target.peak_compute_gflops / target.memory_bandwidth_gbps
    compute_time_us = flops / (target.peak_compute_gflops * 1_000)
    memory_time_us = logical_bytes / (target.memory_bandwidth_gbps * 1_000)
    return {
        "arithmetic_intensity": arithmetic_intensity,
        "ridge_point": ridge_point,
        "bound": "compute" if arithmetic_intensity >= ridge_point else "memory",
        "estimated_time_us": max(compute_time_us, memory_time_us),
    }


def _target_report(target: Target | None) -> dict[str, float] | None:
    if target is None:
        return None
    return {
        "peak_compute_gflops": target.peak_compute_gflops,
        "memory_bandwidth_gbps": target.memory_bandwidth_gbps,
    }


def _op_flops(operation: Operation, graph: Graph) -> int:
    output = graph.values[operation.output]
    if operation.kind == "matmul":
        lhs = graph.values[operation.inputs[0]]
        rhs = graph.values[operation.inputs[1]]
        return 2 * lhs.shape[0] * lhs.shape[1] * rhs.shape[1]
    if operation.kind == "softmax":
        width = output.shape[-1]
        rows = output.elements // width
        return rows * (5 * width - 2)
    if operation.kind == "layer_norm":
        width = output.shape[-1]
        rows = output.elements // width
        # Per row, mean costs K, variance 3K, and normalization 2K + 2
        # scalar-operation equivalents, for a total of 6K + 2.
        return rows * (6 * width + 2)
    # The model counts one scalar arithmetic/comparison per element for these ops.
    return output.elements


def _estimate_operations(graph: Graph) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for operation in graph.operations:
        output = graph.values[operation.output]
        bytes_read = sum(graph.values[name].nbytes for name in operation.inputs)
        bytes_written = output.nbytes
        flops = _op_flops(operation, graph)
        reports.append(
            {
                "name": operation.name,
                "op": operation.kind,
                "inputs": list(operation.inputs),
                "outputs": [operation.output],
                "shape": list(output.shape),
                "dtype": output.dtype,
                "flops": flops,
                "bytes_read": bytes_read,
                "bytes_written": bytes_written,
                "arithmetic_intensity": _intensity(flops, bytes_read, bytes_written),
                "roofline": _roofline_metrics(
                    flops, bytes_read, bytes_written, graph.target
                ),
            }
        )
    return reports


def _fusion_report(graph: Graph) -> dict[str, Any]:
    consumers: Counter[str] = Counter(
        input_name
        for operation in graph.operations
        for input_name in operation.inputs
    )
    candidates: list[dict[str, Any]] = []
    for producer, consumer in zip(graph.operations, graph.operations[1:]):
        intermediate = producer.output
        if producer.kind not in ELEMENTWISE_OPS or consumer.kind not in ELEMENTWISE_OPS:
            continue
        if intermediate not in consumer.inputs or consumers[intermediate] != 1:
            continue
        bytes_avoided = 2 * graph.values[intermediate].nbytes
        candidates.append(
            {
                "producer": producer.name,
                "consumer": consumer.name,
                "intermediate": intermediate,
                "bytes_avoided": bytes_avoided,
                "reason": "adjacent elementwise operations with a single-use intermediate",
            }
        )
    return {"count": len(candidates), "candidates": candidates}


def estimate_graph(graph: Graph) -> dict[str, Any]:
    """Estimate every operation in a validated graph."""
    operations = _estimate_operations(graph)
    total_flops = sum(item["flops"] for item in operations)
    total_bytes_read = sum(item["bytes_read"] for item in operations)
    total_bytes_written = sum(item["bytes_written"] for item in operations)
    totals = {
        "flops": total_flops,
        "bytes_read": total_bytes_read,
        "bytes_written": total_bytes_written,
        "arithmetic_intensity": _intensity(
            total_flops, total_bytes_read, total_bytes_written
        ),
        "roofline": _roofline_metrics(
            total_flops, total_bytes_read, total_bytes_written, graph.target
        ),
    }
    return {
        "graph": graph.name,
        "target": _target_report(graph.target),
        "ops": operations,
        "totals": totals,
        "fusion": _fusion_report(graph),
    }


def analyze_graph(
    payload: Mapping[str, Any], *, optimize: bool = False
) -> dict[str, Any]:
    """Validate and estimate a graph, optionally requesting optimization."""
    graph = parse_graph(payload)
    if optimize:
        from .optimizer import optimize_graph

        return optimize_graph(graph)
    return estimate_graph(graph)
