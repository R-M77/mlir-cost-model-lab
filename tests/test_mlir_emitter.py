from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
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


@pytest.mark.parametrize(
    ("input_name", "replacement"),
    [
        pytest.param("lhs", Tensor("lhs", (1, 2, 3), "f32"), id="lhs-rank"),
        pytest.param("rhs", Tensor("rhs", (1, 3, 4), "f32"), id="rhs-rank"),
        pytest.param("lhs", Tensor("lhs", (5, 3), "f32"), id="lhs-shape"),
        pytest.param("rhs", Tensor("rhs", (3, 5), "f32"), id="rhs-shape"),
        pytest.param("lhs", Tensor("lhs", (2, 3), "f16"), id="dtype"),
        pytest.param("lhs", Tensor("renamed", (2, 3), "f32"), id="name"),
    ],
)
def test_emitter_rejects_inconsistent_graph_input_definitions(
    input_name: str, replacement: Tensor
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
