from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from cost_model import (
    MatmulMlirEmissionError,
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
