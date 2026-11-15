"""Target-independent structural analysis of validated tensor graphs."""

from __future__ import annotations

from typing import Any

from .models import Graph


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
