"""Validation and shape inference for JSON graphs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from json import JSONDecodeError
from math import isfinite
from pathlib import Path
from typing import Any

from .models import DTYPE_BYTES, Graph, Operation, Target, Tensor

SUPPORTED_OPS = frozenset({"add", "matmul", "mul", "relu"})


class GraphValidationError(ValueError):
    """Raised when a graph violates the input schema or inference rules."""


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GraphValidationError(f"{label} must be an object")
    return value


def _require_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GraphValidationError(f"{label} must be a non-empty string")
    return value


def _parse_shape(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise GraphValidationError(f"{label} shape must be a list")
    shape: list[int] = []
    for index, dimension in enumerate(value):
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise GraphValidationError(
                f"{label} shape dimension {index} must be a positive integer"
            )
        shape.append(dimension)
    return tuple(shape)


def _parse_tensor(name: str, value: object, label: str) -> Tensor:
    spec = _require_mapping(value, label)
    shape = _parse_shape(spec.get("shape"), label)
    dtype = spec.get("dtype")
    if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
        supported = ", ".join(sorted(DTYPE_BYTES))
        raise GraphValidationError(f"{label} unsupported dtype {dtype!r}; use one of {supported}")
    return Tensor(name=name, shape=shape, dtype=dtype)


def _parse_target_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value <= 0
    ):
        raise GraphValidationError(f"{label} must be a positive finite number")
    return float(value)


def _parse_target(value: object) -> Target | None:
    # Omitted and explicit null targets both disable roofline analysis.
    if value is None:
        return None
    spec = _require_mapping(value, "target")
    return Target(
        peak_compute_gflops=_parse_target_number(
            spec.get("peak_compute_gflops"), "target.peak_compute_gflops"
        ),
        memory_bandwidth_gbps=_parse_target_number(
            spec.get("memory_bandwidth_gbps"), "target.memory_bandwidth_gbps"
        ),
    )


def _parse_value_names(value: object, label: str, expected: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != expected:
        raise GraphValidationError(f"{label} must contain exactly {expected} value names")
    return tuple(_require_nonempty_string(item, f"{label}[{index}]") for index, item in enumerate(value))


def _broadcast_shape(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> tuple[int, ...]:
    result: list[int] = []
    for left, right in zip(reversed(lhs), reversed(rhs)):
        if left == right:
            result.append(left)
        elif left == 1:
            result.append(right)
        elif right == 1:
            result.append(left)
        else:
            raise ValueError("incompatible dimensions")
    longer = lhs if len(lhs) > len(rhs) else rhs
    result.extend(reversed(longer[: abs(len(lhs) - len(rhs))]))
    return tuple(reversed(result))


def _infer_output(kind: str, inputs: tuple[Tensor, ...], output_name: str) -> Tensor:
    if kind == "matmul":
        lhs, rhs = inputs
        if len(lhs.shape) != 2 or len(rhs.shape) != 2:
            raise GraphValidationError("matmul requires two rank-2 tensors")
        if lhs.shape[1] != rhs.shape[0]:
            raise GraphValidationError("matmul dimensions do not align")
        if lhs.dtype != rhs.dtype:
            raise GraphValidationError("matmul inputs must have the same dtype")
        return Tensor(output_name, (lhs.shape[0], rhs.shape[1]), lhs.dtype)

    if kind in {"add", "mul"}:
        lhs, rhs = inputs
        if lhs.dtype != rhs.dtype:
            raise GraphValidationError(f"{kind} inputs must have the same dtype")
        try:
            output_shape = _broadcast_shape(lhs.shape, rhs.shape)
        except ValueError as exc:
            raise GraphValidationError(f"{kind} inputs cannot be broadcast together") from exc
        return Tensor(output_name, output_shape, lhs.dtype)

    # relu is the only unary operation in this first slice.
    (source,) = inputs
    return Tensor(output_name, source.shape, source.dtype)


def parse_graph(payload: Mapping[str, Any]) -> Graph:
    """Validate a JSON-compatible mapping and infer all operation outputs."""
    root = _require_mapping(payload, "graph")
    name = _require_nonempty_string(root.get("name"), "name")
    target = _parse_target(root.get("target"))

    raw_inputs = root.get("inputs")
    if not isinstance(raw_inputs, Mapping):
        raise GraphValidationError("inputs must be an object")
    inputs: dict[str, Tensor] = {}
    for raw_name, raw_spec in raw_inputs.items():
        value_name = _require_nonempty_string(raw_name, "input name")
        if value_name in inputs:
            raise GraphValidationError(f"duplicate input value {value_name!r}")
        inputs[value_name] = _parse_tensor(value_name, raw_spec, f"input {value_name!r}")

    raw_ops = root.get("ops")
    if not isinstance(raw_ops, list):
        raise GraphValidationError("ops must be a list")
    if not raw_ops:
        raise GraphValidationError("ops must contain at least one operation")

    values = dict(inputs)
    operations: list[Operation] = []
    operation_names: set[str] = set()
    produced_values: set[str] = set()

    for index, raw_op in enumerate(raw_ops):
        op = _require_mapping(raw_op, f"ops[{index}]")
        op_name = _require_nonempty_string(op.get("name"), f"ops[{index}].name")
        if op_name in operation_names:
            raise GraphValidationError(f"duplicate operation name {op_name!r}")
        operation_names.add(op_name)

        kind = op.get("op")
        if not isinstance(kind, str) or kind not in SUPPORTED_OPS:
            supported = ", ".join(sorted(SUPPORTED_OPS))
            raise GraphValidationError(
                f"ops[{index}] unsupported operation {kind!r}; use one of {supported}"
            )
        expected_inputs = 1 if kind == "relu" else 2
        input_names = _parse_value_names(op.get("inputs"), f"ops[{index}].inputs", expected_inputs)
        output_names = _parse_value_names(op.get("outputs"), f"ops[{index}].outputs", 1)
        output_name = output_names[0]
        if output_name in values or output_name in produced_values:
            raise GraphValidationError(f"duplicate value name {output_name!r}")

        input_tensors: list[Tensor] = []
        for input_name in input_names:
            if input_name not in values:
                raise GraphValidationError(
                    f"ops[{index}] unknown input value {input_name!r}"
                )
            input_tensors.append(values[input_name])

        output_tensor = _infer_output(kind, tuple(input_tensors), output_name)
        values[output_name] = output_tensor
        produced_values.add(output_name)
        operations.append(Operation(op_name, kind, input_names, output_name))

    raw_outputs = root.get("outputs")
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise GraphValidationError("outputs must be a non-empty list")
    outputs = tuple(
        _require_nonempty_string(item, f"outputs[{index}]")
        for index, item in enumerate(raw_outputs)
    )
    for output_name in outputs:
        if output_name not in values:
            raise GraphValidationError(f"unknown graph output value {output_name!r}")

    return Graph(name, inputs, tuple(operations), outputs, values, target)


def load_graph(path: str | Path) -> Graph:
    """Read and validate one graph JSON file."""
    graph_path = Path(path)
    try:
        text = graph_path.read_text(encoding="utf-8")
        payload = json.loads(text)
    except OSError as exc:
        raise GraphValidationError(f"could not read graph {graph_path}: {exc}") from exc
    except JSONDecodeError as exc:
        raise GraphValidationError(f"invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}") from exc
    if not isinstance(payload, Mapping):
        raise GraphValidationError("graph JSON must contain an object")
    return parse_graph(payload)
