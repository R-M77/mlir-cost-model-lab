from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from cost_model import (
    MatmulMlirEmissionError,
    Tensor,
    emit_matmul_mlir,
    emit_matmul_mlir_graph,
    parse_graph,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "examples" / "matmul_lowering.json"
EXPECTED_MLIR = """module {
  func.func @matmul(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %init = tensor.empty() : tensor<2x4xf32>
    %zero = arith.constant 0.0 : f32
    %zeroed = linalg.fill ins(%zero : f32) outs(%init : tensor<2x4xf32>) -> tensor<2x4xf32>
    %result = linalg.matmul ins(%lhs, %rhs : tensor<2x3xf32>, tensor<3x4xf32>) outs(%zeroed : tensor<2x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}
"""


def _fixture_payload() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    return subprocess.run(
        [sys.executable, "-m", "cost_model", *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
        env=env,
    )


def test_emits_exact_stable_mlir_for_matmul_fixture() -> None:
    payload = _fixture_payload()

    assert emit_matmul_mlir(payload) == EXPECTED_MLIR
    assert emit_matmul_mlir_graph(parse_graph(payload)) == EXPECTED_MLIR
    assert emit_matmul_mlir(payload) == emit_matmul_mlir(payload)


@pytest.mark.parametrize("name_owner", ["graph", "operation"])
def test_emitter_ignores_graph_and_operation_names(name_owner: str) -> None:
    graph = parse_graph(_fixture_payload())
    if name_owner == "graph":
        graph = replace(graph, name="renamed graph")
    else:
        graph = replace(
            graph, operations=(replace(graph.operations[0], name="renamed operation"),)
        )

    assert emit_matmul_mlir_graph(graph) == EXPECTED_MLIR


@pytest.mark.parametrize(
    "operations",
    [
        pytest.param(None, id="none"),
        pytest.param(7, id="integer"),
        pytest.param("x", id="string"),
        pytest.param({"op": "matmul"}, id="mapping"),
        pytest.param({"matmul"}, id="set"),
        pytest.param((), id="empty-tuple"),
        pytest.param([], id="empty-list"),
        pytest.param((None, None), id="two-operations"),
    ],
)
def test_emitter_rejects_direct_operation_containers(operations: Any) -> None:
    graph = replace(parse_graph(_fixture_payload()), operations=operations)

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        "matmul MLIR emission requires exactly one operation"
    )


@pytest.mark.parametrize("operation", [None, 7, "matmul", {}, ["matmul"], object()])
def test_emitter_rejects_non_operation_definitions(operation: Any) -> None:
    graph = replace(parse_graph(_fixture_payload()), operations=(operation,))

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        "matmul MLIR emission requires an Operation definition"
    )


@pytest.mark.parametrize("kind", [None, 7, True, [], {}, "", "relu"])
def test_emitter_rejects_invalid_direct_operation_kinds(kind: Any) -> None:
    graph = parse_graph(_fixture_payload())
    graph = replace(graph, operations=(replace(graph.operations[0], kind=kind),))

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        "matmul MLIR emission requires a single matmul operation"
    )


@pytest.mark.parametrize("field", ["kind", "inputs", "output"])
def test_emitter_rejects_incomplete_operation_definitions(field: str) -> None:
    graph = parse_graph(_fixture_payload())
    object.__delattr__(graph.operations[0], field)

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        "matmul MLIR emission requires an Operation definition"
    )


def test_emitter_preserves_direct_operation_list() -> None:
    graph = parse_graph(_fixture_payload())
    graph = replace(graph, operations=list(graph.operations))

    assert emit_matmul_mlir_graph(graph) == EXPECTED_MLIR


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("operations", "requires exactly one operation"),
        ("values", "requires graph values to be a mapping"),
    ],
)
def test_emitter_rejects_missing_direct_graph_containers(field: str, message: str) -> None:
    graph = parse_graph(_fixture_payload())
    object.__delattr__(graph, field)

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == f"matmul MLIR emission {message}"


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(None, id="none"),
        pytest.param(7, id="integer"),
        pytest.param("lr", id="string"),
        pytest.param(["lhs", "rhs"], id="list"),
        pytest.param({"lhs": "lhs", "rhs": "rhs"}, id="mapping"),
        pytest.param({"lhs", "rhs"}, id="set"),
        pytest.param((), id="empty"),
        pytest.param(("lhs",), id="one-operand"),
        pytest.param(("lhs", "rhs", "lhs"), id="three-operands"),
    ],
)
def test_emitter_rejects_direct_operation_input_containers(inputs: Any) -> None:
    graph = parse_graph(_fixture_payload())
    graph = replace(graph, operations=(replace(graph.operations[0], inputs=inputs),))

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        "matmul MLIR emission requires exactly two matmul inputs"
    )


@pytest.mark.parametrize(
    "input_name",
    [
        pytest.param("", id="empty"),
        pytest.param(" \t\n", id="whitespace"),
        pytest.param(None, id="none"),
        pytest.param(7, id="integer"),
        pytest.param(True, id="boolean"),
        pytest.param(("lhs",), id="tuple"),
        pytest.param(["lhs"], id="list"),
        pytest.param({"value": "lhs"}, id="mapping"),
        pytest.param({"lhs"}, id="set"),
    ],
)
@pytest.mark.parametrize("operand_index", [0, 1])
def test_emitter_rejects_invalid_direct_operand_names(
    input_name: Any, operand_index: int
) -> None:
    graph = parse_graph(_fixture_payload())
    inputs = list(graph.operations[0].inputs)
    inputs[operand_index] = input_name
    graph = replace(
        graph, operations=(replace(graph.operations[0], inputs=tuple(inputs)),)
    )

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        "matmul MLIR emission requires non-empty string matmul input names"
    )


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(None, id="none"),
        pytest.param(7, id="integer"),
        pytest.param("lr", id="string"),
        pytest.param(("lhs", "rhs"), id="tuple"),
        pytest.param(["lhs", "rhs"], id="list"),
        pytest.param({"lhs", "rhs"}, id="set"),
        pytest.param([["lhs"], ["rhs"]], id="nested-list"),
    ],
)
def test_emitter_rejects_direct_graph_input_containers(inputs: Any) -> None:
    graph = replace(parse_graph(_fixture_payload()), inputs=inputs)

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        "matmul MLIR emission requires graph inputs to be a mapping"
    )


@pytest.mark.parametrize("input_name", ["", " \t\n", None, 7, True, ("lhs",)])
def test_emitter_rejects_invalid_direct_graph_input_keys(input_name: Any) -> None:
    graph = parse_graph(_fixture_payload())
    graph.inputs[input_name] = graph.inputs.pop("lhs")

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        "matmul MLIR emission requires exactly the matmul operands as graph inputs"
    )


@pytest.mark.parametrize("values", [None, 7, "values", [], (), {"lhs", "rhs"}])
def test_emitter_rejects_direct_graph_value_containers(values: Any) -> None:
    graph = replace(parse_graph(_fixture_payload()), values=values)

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        "matmul MLIR emission requires graph values to be a mapping"
    )


@pytest.mark.parametrize("value_name", ["", " \t\n", None, 7, True, ("extra",)])
def test_emitter_rejects_invalid_direct_graph_value_keys(value_name: Any) -> None:
    graph = parse_graph(_fixture_payload())
    graph.values[value_name] = Tensor(value_name, (), "i32")

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        "matmul MLIR emission requires non-empty string graph value names"
    )


@pytest.mark.parametrize("value_name", ["lhs", "rhs", "output", "extra"])
@pytest.mark.parametrize("definition", [None, 7, "tensor", {}, [2, 3], object()])
def test_emitter_rejects_non_tensor_graph_value_definitions(
    value_name: str, definition: Any
) -> None:
    graph = parse_graph(_fixture_payload())
    graph.values[value_name] = definition
    if value_name in graph.inputs:
        graph.inputs[value_name] = definition

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        f"matmul MLIR emission graph value {value_name!r} requires a Tensor definition"
    )


@pytest.mark.parametrize("value_name", ["lhs", "rhs", "output", "extra"])
@pytest.mark.parametrize("field", ["name", "shape", "dtype"])
def test_emitter_rejects_incomplete_tensor_definitions(
    value_name: str, field: str
) -> None:
    graph = parse_graph(_fixture_payload())
    if value_name == "extra":
        graph.values[value_name] = Tensor(value_name, (), "i32")
    object.__delattr__(graph.values[value_name], field)

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        f"matmul MLIR emission graph value {value_name!r} requires a Tensor definition"
    )


@pytest.mark.parametrize("value_name", ["lhs", "rhs", "output", "extra"])
@pytest.mark.parametrize("shape", [None, 7, "2x3", [2, 3], {"shape": [2, 3]}, {2, 3}])
def test_emitter_rejects_malformed_graph_value_shapes(
    value_name: str, shape: Any
) -> None:
    graph = parse_graph(_fixture_payload())
    tensor = Tensor(value_name, shape, "f32")
    graph.values[value_name] = tensor
    if value_name in graph.inputs:
        graph.inputs[value_name] = tensor

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        f"matmul MLIR emission graph value {value_name!r} requires a tuple shape"
    )


@pytest.mark.parametrize("value_name", ["lhs", "rhs", "output", "extra"])
@pytest.mark.parametrize("dimension", [0, -1, True, 2.0, "2", None, [2], {"size": 2}])
def test_emitter_rejects_malformed_graph_value_dimensions(
    value_name: str, dimension: Any
) -> None:
    graph = parse_graph(_fixture_payload())
    tensor = Tensor(value_name, (dimension, 3), "f32")
    graph.values[value_name] = tensor
    if value_name in graph.inputs:
        graph.inputs[value_name] = tensor

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        f"matmul MLIR emission graph value {value_name!r} "
        "requires positive integer shape dimensions"
    )


@pytest.mark.parametrize("dtype", [None, 7, True, [], {}, "", "unknown"])
@pytest.mark.parametrize("value_name", ["lhs", "rhs", "output", "extra"])
def test_emitter_rejects_malformed_graph_value_dtypes(
    value_name: str, dtype: Any
) -> None:
    graph = parse_graph(_fixture_payload())
    original = graph.values.get(value_name, Tensor(value_name, (), "i32"))
    tensor = replace(original, dtype=dtype)
    graph.values[value_name] = tensor
    if value_name in graph.inputs:
        graph.inputs[value_name] = tensor

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        f"matmul MLIR emission graph value {value_name!r} has unsupported dtype {dtype!r}"
        if value_name == "extra"
        else "matmul MLIR emission only supports f32 tensors"
    )


@pytest.mark.parametrize("readonly_values", [False, True])
def test_emitter_preserves_valid_direct_graph_value_mappings(readonly_values: bool) -> None:
    graph = parse_graph(_fixture_payload())
    values = dict(reversed(list(graph.values.items())))
    graph = replace(
        graph, values=MappingProxyType(values) if readonly_values else values
    )

    assert emit_matmul_mlir_graph(graph) == EXPECTED_MLIR


@pytest.mark.parametrize(
    ("value_name", "shape", "message"),
    [
        ("lhs", (1, 2, 3), "requires rank-2 input tensors"),
        ("rhs", (4, 4), "requires aligned matmul dimensions"),
        ("output", (2, 5), "requires the inferred matmul result shape"),
    ],
)
def test_emitter_preserves_direct_tensor_shape_errors(
    value_name: str, shape: tuple[int, ...], message: str
) -> None:
    graph = parse_graph(_fixture_payload())
    tensor = replace(graph.values[value_name], shape=shape)
    graph.values[value_name] = tensor
    if value_name in graph.inputs:
        graph.inputs[value_name] = tensor

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == f"matmul MLIR emission {message}"


@pytest.mark.parametrize("input_name", ["lhs", "rhs"])
def test_emitter_rejects_direct_operand_missing_from_graph_values(input_name: str) -> None:
    graph = parse_graph(_fixture_payload())
    del graph.values[input_name]

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        f"matmul MLIR emission references unknown value {input_name!r}"
    )


@pytest.mark.parametrize("readonly_inputs", [False, True])
@pytest.mark.parametrize("repeated_operand", [False, True])
def test_emitter_preserves_valid_direct_graph_input_mappings(
    readonly_inputs: bool, repeated_operand: bool
) -> None:
    payload = _fixture_payload()
    if repeated_operand:
        payload["inputs"] = {"lhs": {"shape": [2, 2], "dtype": "f32"}}
        payload["ops"][0]["inputs"] = ["lhs", "lhs"]
    expected = emit_matmul_mlir(payload)
    graph = parse_graph(payload)
    # Input declaration order must not change operand order or SSA argument reuse.
    inputs = {
        name: replace(tensor) for name, tensor in reversed(list(graph.inputs.items()))
    }
    graph = replace(
        graph, inputs=MappingProxyType(inputs) if readonly_inputs else inputs
    )

    assert emit_matmul_mlir_graph(graph) == expected


@pytest.mark.parametrize(
    ("input_name", "replacement"),
    [
        pytest.param("lhs", Tensor("lhs", (1, 2, 3), "f32"), id="lhs-rank"),
        pytest.param("rhs", Tensor("rhs", (1, 3, 4), "f32"), id="rhs-rank"),
        pytest.param("lhs", Tensor("lhs", (5, 3), "f32"), id="lhs-shape"),
        pytest.param("rhs", Tensor("rhs", (3, 5), "f32"), id="rhs-shape"),
        pytest.param("lhs", Tensor("lhs", (2, 3), "f16"), id="dtype"),
        pytest.param("lhs", Tensor("renamed", (2, 3), "f32"), id="name"),
        pytest.param("lhs", None, id="none"),
        pytest.param("rhs", ["rhs"], id="list"),
    ],
)
def test_emitter_rejects_inconsistent_graph_input_definitions(
    input_name: str, replacement: Any
) -> None:
    graph = parse_graph(_fixture_payload())
    original = graph.values[input_name]
    graph.inputs[input_name] = replacement
    assert graph.values[input_name] == original
    assert graph.inputs[input_name] != original

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        f"matmul MLIR emission requires graph input {input_name!r} "
        "to match its graph value definition"
    )


@pytest.mark.parametrize("input_name", ["lhs", "rhs"])
@pytest.mark.parametrize("field", ["name", "shape", "dtype"])
def test_emitter_rejects_incomplete_detached_graph_input_tensors(
    input_name: str, field: str
) -> None:
    graph = parse_graph(_fixture_payload())
    tensor = replace(graph.inputs[input_name])
    object.__delattr__(tensor, field)
    graph.inputs[input_name] = tensor
    assert hasattr(graph.values[input_name], field)

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        f"matmul MLIR emission requires graph input {input_name!r} "
        "to match its graph value definition"
    )


@pytest.mark.parametrize("input_name", ["lhs", "rhs"])
@pytest.mark.parametrize("field", ["name", "shape", "dtype"])
@pytest.mark.parametrize("error_type", [AttributeError, TypeError])
def test_emitter_rejects_uncomparable_detached_graph_input_tensor_fields(
    input_name: str, field: str, error_type: type[Exception]
) -> None:
    class InvalidField:
        def __eq__(self, other: object) -> bool:
            raise error_type("malformed tensor field")

    graph = parse_graph(_fixture_payload())
    graph.inputs[input_name] = replace(
        graph.inputs[input_name], **{field: InvalidField()}
    )

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        f"matmul MLIR emission requires graph input {input_name!r} "
        "to match its graph value definition"
    )


@pytest.mark.parametrize(
    ("value_name", "replacement"),
    [
        pytest.param("output", Tensor("stale", (2, 4), "f32"), id="output"),
        pytest.param("extra", Tensor("other", (9, 9), "f32"), id="unused-value"),
        pytest.param("lhs", Tensor("stale", (2, 3), "f32"), id="lhs-both-maps"),
        pytest.param("rhs", Tensor("stale", (3, 4), "f32"), id="rhs-both-maps"),
    ],
)
def test_emitter_rejects_graph_value_key_tensor_name_mismatches(
    value_name: str, replacement: Tensor
) -> None:
    graph = parse_graph(_fixture_payload())
    graph.values[value_name] = replacement
    if value_name in graph.inputs:
        graph.inputs[value_name] = replacement

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        f"matmul MLIR emission graph value {value_name!r} "
        f"has mismatched tensor name {replacement.name!r}"
    )


@pytest.mark.parametrize(
    "extra", [Tensor("extra", (9, 9), "f32"), Tensor("extra", (), "i32")]
)
def test_emitter_preserves_mlir_with_consistent_unused_graph_value(extra: Tensor) -> None:
    graph = parse_graph(_fixture_payload())
    graph.values[extra.name] = extra

    assert emit_matmul_mlir_graph(graph) == EXPECTED_MLIR


@pytest.mark.parametrize(
    ("operand_names", "output_name"),
    [
        pytest.param(("lhs", "rhs"), "lhs", id="lhs-alias"),
        pytest.param(("lhs", "rhs"), "rhs", id="rhs-alias"),
        pytest.param(("lhs", "lhs"), "lhs", id="repeated-input-alias"),
    ],
)
def test_emitter_rejects_matmul_result_aliasing_graph_input(
    operand_names: tuple[str, str], output_name: str
) -> None:
    payload = _fixture_payload()
    payload["inputs"] = {
        name: {"shape": [2, 2], "dtype": "f32"} for name in operand_names
    }
    payload["ops"][0]["inputs"] = list(operand_names)
    graph = parse_graph(payload)
    assert emit_matmul_mlir_graph(graph) == emit_matmul_mlir(payload)

    graph = replace(
        graph,
        operations=(replace(graph.operations[0], output=output_name),),
        outputs=(output_name,),
    )

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        "matmul MLIR emission requires the matmul result to be distinct from graph inputs"
    )


@pytest.mark.parametrize(
    "outputs",
    [
        pytest.param((), id="empty"),
        pytest.param(("output", "output"), id="duplicate"),
        pytest.param(("output", "lhs"), id="extra-output"),
        pytest.param(("lhs",), id="input-only"),
        pytest.param(("missing",), id="unknown"),
        pytest.param(["output"], id="list"),
        pytest.param("output", id="string"),
        pytest.param(None, id="none"),
        pytest.param(7, id="integer"),
        pytest.param((["output"],), id="nested-list"),
    ],
)
def test_emitter_rejects_direct_graph_output_mutations(outputs: Any) -> None:
    graph = replace(parse_graph(_fixture_payload()), outputs=outputs)

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        "matmul MLIR emission requires the matmul result to be the only graph output"
    )


@pytest.mark.parametrize("match_outputs", [False, True])
def test_emitter_rejects_unknown_direct_operation_output(match_outputs: bool) -> None:
    graph = parse_graph(_fixture_payload())
    graph = replace(
        graph,
        operations=(replace(graph.operations[0], output="missing"),),
        outputs=("missing",) if match_outputs else graph.outputs,
    )

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        "matmul MLIR emission references unknown value 'missing'"
        if match_outputs
        else "matmul MLIR emission requires the matmul result to be the only graph output"
    )


@pytest.mark.parametrize(
    "output_name",
    [
        pytest.param("", id="empty"),
        pytest.param(" \t\n", id="whitespace"),
        pytest.param(None, id="none"),
        pytest.param(7, id="integer"),
        pytest.param(True, id="boolean"),
        pytest.param(("output",), id="tuple"),
    ],
)
@pytest.mark.parametrize("register_value", [False, True])
def test_emitter_rejects_invalid_direct_result_names(
    output_name: Any, register_value: bool
) -> None:
    graph = parse_graph(_fixture_payload())
    if register_value:
        graph.values[output_name] = replace(graph.values["output"], name=output_name)
    graph = replace(
        graph,
        operations=(replace(graph.operations[0], output=output_name),),
        outputs=(output_name,),
    )

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        "matmul MLIR emission requires a non-empty string matmul result name"
    )


@pytest.mark.parametrize("output_name", [["output"], {"value": "output"}])
def test_emitter_rejects_unhashable_direct_result_names(output_name: Any) -> None:
    graph = parse_graph(_fixture_payload())
    graph = replace(
        graph,
        operations=(replace(graph.operations[0], output=output_name),),
        outputs=(output_name,),
    )

    with pytest.raises(MatmulMlirEmissionError) as exc_info:
        emit_matmul_mlir_graph(graph)

    assert str(exc_info.value) == (
        "matmul MLIR emission requires a non-empty string matmul result name"
    )


@pytest.mark.parametrize("output_name", ["renamed", " result % ! \n"])
@pytest.mark.parametrize("repeated_operand", [False, True])
def test_emitter_preserves_mlir_with_consistently_renamed_result(
    output_name: str, repeated_operand: bool
) -> None:
    payload = _fixture_payload()
    if repeated_operand:
        payload["inputs"] = {"lhs": {"shape": [2, 2], "dtype": "f32"}}
        payload["ops"][0]["inputs"] = ["lhs", "lhs"]
    graph = parse_graph(payload)
    expected = emit_matmul_mlir_graph(graph)
    # Retain the old result as a consistently named unused value.
    graph.values[output_name] = replace(graph.values["output"], name=output_name)
    graph = replace(
        graph,
        operations=(replace(graph.operations[0], output=output_name),),
        outputs=(output_name,),
    )

    assert emit_matmul_mlir_graph(graph) == expected
    assert emit_matmul_mlir_graph(graph) == expected


def test_emitter_reuses_one_argument_for_repeated_matmul_operand() -> None:
    payload = _fixture_payload()
    payload["inputs"] = {"lhs": {"shape": [2, 2], "dtype": "f32"}}
    payload["ops"][0]["inputs"] = ["lhs", "lhs"]

    emitted = emit_matmul_mlir(payload)

    assert "func.func @matmul(%lhs: tensor<2x2xf32>) -> tensor<2x2xf32>" in emitted
    assert "linalg.matmul ins(%lhs, %lhs : tensor<2x2xf32>, tensor<2x2xf32>)" in emitted


def test_emit_mlir_cli_is_raw_and_deterministic() -> None:
    first = _run_cli("--emit-mlir", str(FIXTURE_PATH))
    second = _run_cli("--emit-mlir", str(FIXTURE_PATH))

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout == EXPECTED_MLIR


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda payload: payload["ops"].append(
                {
                    "name": "relu",
                    "op": "relu",
                    "inputs": ["output"],
                    "outputs": ["final"],
                }
            ),
            "requires exactly one operation",
        ),
        (
            lambda payload: payload.update({"outputs": ["lhs"]}),
            "requires the matmul result to be the only graph output",
        ),
        (
            lambda payload: [spec.update({"dtype": "f16"}) for spec in payload["inputs"].values()],
            "only supports f32 tensors",
        ),
        (
            lambda payload: payload["inputs"].update(
                {"unused": {"shape": [1], "dtype": "f32"}}
            ),
            "exactly the matmul operands as graph inputs",
        ),
    ],
)
def test_emitter_rejects_graphs_outside_its_scope(
    change: object, message: str
) -> None:
    payload = deepcopy(_fixture_payload())
    change(payload)  # type: ignore[operator]

    with pytest.raises(MatmulMlirEmissionError, match=message):
        emit_matmul_mlir(payload)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--compact",), "--emit-mlir cannot be combined with --compact"),
        (("--omit-paths",), "--emit-mlir cannot be combined with --omit-paths"),
        (("--budget", "budget.json"), "--emit-mlir cannot be combined with --budget"),
        (("--optimize",), "--emit-mlir cannot be combined with --optimize"),
        (("--lower-matmul",), "--emit-mlir cannot be combined with --lower-matmul"),
    ],
)
def test_emit_mlir_cli_rejects_incompatible_modes(
    args: tuple[str, ...], message: str
) -> None:
    result = _run_cli("--emit-mlir", *args, str(FIXTURE_PATH))

    assert result.returncode == 2
    assert result.stdout == ""
    assert message in result.stderr


def test_emit_mlir_cli_requires_single_graph_path() -> None:
    benchmark = REPO_ROOT / "benchmarks" / "target_bounds.json"
    result = _run_cli("--emit-mlir", "--benchmark", str(benchmark))

    assert result.returncode == 2
    assert result.stdout == ""
    assert "--emit-mlir requires a single graph path" in result.stderr
