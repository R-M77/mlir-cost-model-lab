"""Deterministic textual MLIR emission for the standalone matmul slice."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import DTYPE_BYTES, Graph, Operation, Tensor
from .parser import GraphValidationError, parse_graph


class MatmulMlirEmissionError(GraphValidationError):
    """Raised when a graph is outside the textual MLIR emitter's scope."""


def _tensor_type(tensor: Tensor) -> str:
    dimensions = "x".join(str(dimension) for dimension in tensor.shape)
    return f"tensor<{dimensions}x{tensor.dtype}>"


def _validate_matmul_graph(graph: Graph) -> tuple[Tensor, Tensor, Tensor]:
    operations = getattr(graph, "operations", None)
    if not isinstance(operations, (tuple, list)) or len(operations) != 1:
        raise MatmulMlirEmissionError(
            "matmul MLIR emission requires exactly one operation"
        )

    operation = operations[0]
    if not isinstance(operation, Operation) or any(
        not hasattr(operation, field) for field in ("kind", "inputs", "output")
    ):
        raise MatmulMlirEmissionError(
            "matmul MLIR emission requires an Operation definition"
        )
    if not isinstance(operation.kind, str) or operation.kind != "matmul":
        raise MatmulMlirEmissionError(
            "matmul MLIR emission requires a single matmul operation"
        )
    if not isinstance(operation.output, str) or not operation.output.strip():
        raise MatmulMlirEmissionError(
            "matmul MLIR emission requires a non-empty string matmul result name"
        )
    if graph.outputs != (operation.output,):
        raise MatmulMlirEmissionError(
            "matmul MLIR emission requires the matmul result to be the only graph output"
        )
    if not isinstance(operation.inputs, tuple) or len(operation.inputs) != 2:
        raise MatmulMlirEmissionError(
            "matmul MLIR emission requires exactly two matmul inputs"
        )
    if any(not isinstance(name, str) or not name.strip() for name in operation.inputs):
        raise MatmulMlirEmissionError(
            "matmul MLIR emission requires non-empty string matmul input names"
        )

    if not isinstance(graph.inputs, Mapping):
        raise MatmulMlirEmissionError(
            "matmul MLIR emission requires graph inputs to be a mapping"
        )
    if len(graph.inputs) not in {1, 2} or set(graph.inputs) != set(operation.inputs):
        raise MatmulMlirEmissionError(
            "matmul MLIR emission requires exactly the matmul operands as graph inputs"
        )

    if operation.output in graph.inputs:
        raise MatmulMlirEmissionError(
            "matmul MLIR emission requires the matmul result to be distinct from graph inputs"
        )

    values = getattr(graph, "values", None)
    if not isinstance(values, Mapping):
        raise MatmulMlirEmissionError(
            "matmul MLIR emission requires graph values to be a mapping"
        )
    try:
        lhs = values[operation.inputs[0]]
        rhs = values[operation.inputs[1]]
        output = values[operation.output]
    except KeyError as exc:
        raise MatmulMlirEmissionError(
            f"matmul MLIR emission references unknown value {exc.args[0]!r}"
        ) from exc

    # Check definitions before dataclass equality can dereference missing fields.
    for value_name, tensor in values.items():
        if not isinstance(value_name, str) or not value_name.strip():
            raise MatmulMlirEmissionError(
                "matmul MLIR emission requires non-empty string graph value names"
            )
        if not isinstance(tensor, Tensor) or any(
            not hasattr(tensor, field) for field in ("name", "shape", "dtype")
        ):
            raise MatmulMlirEmissionError(
                f"matmul MLIR emission graph value {value_name!r} "
                "requires a Tensor definition"
            )

    for input_name in operation.inputs:
        tensor = graph.inputs[input_name]
        try:
            matches_value = (
                isinstance(tensor, Tensor)
                and all(hasattr(tensor, field) for field in ("name", "shape", "dtype"))
                and bool(tensor == values[input_name])
            )
        except (AttributeError, TypeError):
            matches_value = False
        if not matches_value:
            raise MatmulMlirEmissionError(
                f"matmul MLIR emission requires graph input {input_name!r} "
                "to match its graph value definition"
            )

    for value_name, tensor in values.items():
        if value_name != tensor.name:
            raise MatmulMlirEmissionError(
                f"matmul MLIR emission graph value {value_name!r} "
                f"has mismatched tensor name {tensor.name!r}"
            )
        if not isinstance(tensor.shape, tuple):
            raise MatmulMlirEmissionError(
                f"matmul MLIR emission graph value {value_name!r} requires a tuple shape"
            )
        if any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
            for dimension in tensor.shape
        ):
            raise MatmulMlirEmissionError(
                f"matmul MLIR emission graph value {value_name!r} "
                "requires positive integer shape dimensions"
            )

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
    for value_name, tensor in values.items():
        if not isinstance(tensor.dtype, str) or tensor.dtype not in DTYPE_BYTES:
            raise MatmulMlirEmissionError(
                f"matmul MLIR emission graph value {value_name!r} "
                f"has unsupported dtype {tensor.dtype!r}"
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
