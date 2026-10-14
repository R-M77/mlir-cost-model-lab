"""Deterministic textual MLIR emission for the standalone matmul slice."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import Graph, Tensor
from .parser import GraphValidationError, parse_graph


class MatmulMlirEmissionError(GraphValidationError):
    """Raised when a graph is outside the textual MLIR emitter's scope."""


def _tensor_type(tensor: Tensor) -> str:
    dimensions = "x".join(str(dimension) for dimension in tensor.shape)
    return f"tensor<{dimensions}x{tensor.dtype}>"


def _validate_matmul_graph(graph: Graph) -> tuple[Tensor, Tensor, Tensor]:
    if len(graph.operations) != 1:
        raise MatmulMlirEmissionError(
            "matmul MLIR emission requires exactly one operation"
        )

    operation = graph.operations[0]
    if operation.kind != "matmul":
        raise MatmulMlirEmissionError(
            "matmul MLIR emission requires a single matmul operation"
        )
    if graph.outputs != (operation.output,):
        raise MatmulMlirEmissionError(
            "matmul MLIR emission requires the matmul result to be the only graph output"
        )
    if len(operation.inputs) != 2:
        raise MatmulMlirEmissionError(
            "matmul MLIR emission requires exactly two matmul inputs"
        )

    if len(graph.inputs) not in {1, 2} or set(graph.inputs) != set(operation.inputs):
        raise MatmulMlirEmissionError(
            "matmul MLIR emission requires exactly the matmul operands as graph inputs"
        )

    try:
        lhs = graph.values[operation.inputs[0]]
        rhs = graph.values[operation.inputs[1]]
        output = graph.values[operation.output]
    except KeyError as exc:
        raise MatmulMlirEmissionError(
            f"matmul MLIR emission references unknown value {exc.args[0]!r}"
        ) from exc

    if len(lhs.shape) != 2 or len(rhs.shape) != 2:
        raise MatmulMlirEmissionError(
            "matmul MLIR emission requires rank-2 input tensors"
        )
    if lhs.shape[1] != rhs.shape[0]:
        raise MatmulMlirEmissionError(
            "matmul MLIR emission requires aligned matmul dimensions"
        )
    if output.shape != (lhs.shape[0], rhs.shape[1]):
        raise MatmulMlirEmissionError(
            "matmul MLIR emission requires the inferred matmul result shape"
        )
    if any(tensor.dtype != "f32" for tensor in (*graph.inputs.values(), lhs, rhs, output)):
        raise MatmulMlirEmissionError(
            "matmul MLIR emission only supports f32 tensors"
        )
    if lhs.dtype != rhs.dtype or output.dtype != lhs.dtype:
        raise MatmulMlirEmissionError(
            "matmul MLIR emission requires matching input and output dtypes"
        )
    return lhs, rhs, output


def emit_matmul_mlir_graph(graph: Graph) -> str:
    """Emit stable textual MLIR for one standalone rank-2 f32 matmul graph."""
    lhs, rhs, output = _validate_matmul_graph(graph)
    operation = graph.operations[0]
    lhs_type = _tensor_type(lhs)
    rhs_type = _tensor_type(rhs)
    output_type = _tensor_type(output)
    operand_ssa: dict[str, str] = {}
    argument_declarations: list[str] = []
    for input_name, input_type, argument_name in (
        (operation.inputs[0], lhs_type, "%lhs"),
        (operation.inputs[1], rhs_type, "%rhs"),
    ):
        if input_name not in operand_ssa:
            operand_ssa[input_name] = argument_name
            argument_declarations.append(f"{argument_name}: {input_type}")
    lhs_ssa = operand_ssa[operation.inputs[0]]
    rhs_ssa = operand_ssa[operation.inputs[1]]
    arguments = ", ".join(argument_declarations)
    return (
        "module {\n"
        f"  func.func @matmul({arguments}) -> {output_type} {{\n"
        f"    %init = tensor.empty() : {output_type}\n"
        "    %zero = arith.constant 0.0 : f32\n"
        f"    %zeroed = linalg.fill ins(%zero : f32) outs(%init : {output_type}) -> {output_type}\n"
        f"    %result = linalg.matmul ins({lhs_ssa}, {rhs_ssa} : "
        f"{lhs_type}, {rhs_type}) outs(%zeroed : {output_type}) -> {output_type}\n"
        f"    return %result : {output_type}\n"
        "  }\n"
        "}\n"
    )


def emit_matmul_mlir(payload: Mapping[str, Any]) -> str:
    """Validate a graph payload and emit its standalone matmul as textual MLIR."""
    return emit_matmul_mlir_graph(parse_graph(payload))
