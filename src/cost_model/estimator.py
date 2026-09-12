"""Deterministic cost estimation and simple fusion suggestions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from .models import Graph, Operation
from .parser import parse_graph

ELEMENTWISE_OPS = frozenset({"add", "mul", "relu"})


def _intensity(flops: int, bytes_read: int, bytes_written: int) -> float:
    traffic = bytes_read + bytes_written
    return flops / traffic if traffic else 0.0


def _op_flops(operation: Operation, graph: Graph) -> int:
    output = graph.values[operation.output]
    if operation.kind == "matmul":
        lhs = graph.values[operation.inputs[0]]
        rhs = graph.values[operation.inputs[1]]
        return 2 * lhs.shape[0] * lhs.shape[1] * rhs.shape[1]
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
    }
    return {
        "graph": graph.name,
        "ops": operations,
        "totals": totals,
        "fusion": _fusion_report(graph),
    }


def analyze_graph(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and estimate a JSON-compatible graph mapping."""
    return estimate_graph(parse_graph(payload))
