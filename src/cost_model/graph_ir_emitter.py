"""Dependency-free MLIR-inspired graph text, not upstream MLIR."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import Graph, Tensor
from .parser import parse_graph


def _tensor_type(tensor: Tensor) -> str:
    return f"tensor<{'x'.join((*map(str, tensor.shape), tensor.dtype))}>"


def emit_graph_ir_graph(graph: Graph) -> str:
    """Render a parsed, validated graph without mutating or optimizing it."""
    names = {name: f"%arg{index}" for index, name in enumerate(sorted(graph.inputs))}
    arguments = ", ".join(
        f"{ssa}: {_tensor_type(graph.inputs[name])}" for name, ssa in names.items()
    )
    result_types = ", ".join(_tensor_type(graph.values[name]) for name in graph.outputs)
    signature = f"({result_types})" if len(graph.outputs) > 1 else result_types
    lines = ["module {", f"  func.func @main({arguments}) -> {signature} {{"]
    for index, operation in enumerate(graph.operations):
        operands = ", ".join(names[name] for name in operation.inputs)
        operand_types = ", ".join(
            _tensor_type(graph.values[name]) for name in operation.inputs
        )
        output_type = _tensor_type(graph.values[operation.output])
        ssa = f"%v{index}"
        lines.append(
            f'    {ssa} = "cost_model.{operation.kind}"({operands})'
            f" : ({operand_types}) -> {output_type}"
        )
        names[operation.output] = ssa
    results = ", ".join(names[name] for name in graph.outputs)
    lines.extend([f"    return {results} : {result_types}", "  }", "}"])
    return "\n".join(lines) + "\n"


def emit_graph_ir(payload: Mapping[str, Any]) -> str:
    """Validate a graph mapping and emit deterministic MLIR-inspired text."""
    return emit_graph_ir_graph(parse_graph(payload))
