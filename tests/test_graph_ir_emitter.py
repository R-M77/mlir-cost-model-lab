"""Typed graph artifacts through the public API and real module CLI."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

from cost_model import GraphValidationError, emit_graph_ir, emit_graph_ir_graph, parse_graph

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cost_model", *args],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_emits_exact_typed_graph_ir_through_cli_and_public_apis(tmp_path: Path) -> None:
    payload = {
        "name": "relu_demo",
        "inputs": {"source": {"shape": [2, 3], "dtype": "f32"}},
        "ops": [
            {"name": "activation", "op": "relu", "inputs": ["source"], "outputs": ["out"]}
        ],
        "outputs": ["out"],
    }
    expected = '''module {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %v0 = "cost_model.relu"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %v0 : tensor<2x3xf32>
  }
}
'''
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_cli("--emit-graph-ir", str(path))

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == expected

    assert emit_graph_ir(payload) == expected
    assert emit_graph_ir_graph(parse_graph(payload)) == expected


def test_multi_output_returns_preserve_declaration_order_with_scalar_result() -> None:
    payload = {
        "name": "scalar_and_input",
        "inputs": {
            "x": {"shape": [3], "dtype": "f32"},
            "unused": {"shape": [], "dtype": "bool"},
        },
        "ops": [
            {"name": "sum", "op": "reduce_sum", "inputs": ["x"], "outputs": ["sum"]}
        ],
        "outputs": ["x", "sum"],
    }

    assert emit_graph_ir(payload) == '''module {
  func.func @main(%arg0: tensor<bool>, %arg1: tensor<3xf32>) -> (tensor<3xf32>, tensor<f32>) {
    %v0 = "cost_model.reduce_sum"(%arg1) : (tensor<3xf32>) -> tensor<f32>
    return %arg1, %v0 : tensor<3xf32>, tensor<f32>
  }
}
'''


@pytest.mark.parametrize(
    "options",
    [
        ("--benchmark", "suite.json"),
        ("--optimize",),
        ("--liveness",),
        ("--critical-path",),
        ("--weighted-critical-path",),
        ("--lower-matmul",),
        ("--emit-mlir",),
        ("--compact",),
        ("--format", "markdown"),
        ("--budget", "budget.json"),
        ("--compare-baseline", "baseline.json"),
        ("--case", "example"),
        ("--omit-paths",),
        ("--lowering-mode", "tiled"),
        ("--tile-size", "8"),
        ("--vector-width", "2"),
    ],
)
def test_cli_rejects_incompatible_graph_ir_modes_before_loading(options: tuple[str, ...]) -> None:
    result = _run_cli("--emit-graph-ir", *options, "missing.json")

    assert result.returncode == 2
    assert result.stdout == ""
    assert f"--emit-graph-ir cannot be combined with {options[0]}" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_graph_ir_requires_single_graph_path() -> None:
    result = _run_cli("--emit-graph-ir")

    assert result.returncode == 2
    assert result.stdout == ""
    assert "--emit-graph-ir requires a single graph path" in result.stderr


def test_all_supported_operations_have_explicit_inferred_tensor_types() -> None:
    payload = {
        "name": "whole_vocabulary",
        "inputs": {
            "rhs": {"shape": [3, 4], "dtype": "bf16"},
            "lhs": {"shape": [1, 2, 3], "dtype": "bf16"},
            "bias": {"shape": [4], "dtype": "bf16"},
        },
        "ops": [
            {"name": "mm", "op": "matmul", "inputs": ["lhs", "rhs"], "outputs": ["product"]},
            {"name": "add", "op": "add", "inputs": ["product", "bias"], "outputs": ["biased"]},
            {"name": "square", "op": "mul", "inputs": ["biased", "biased"], "outputs": ["squared"]},
            {"name": "relu", "op": "relu", "inputs": ["squared"], "outputs": ["positive"]},
            {"name": "softmax", "op": "softmax", "inputs": ["positive"], "outputs": ["prob"]},
            {"name": "norm", "op": "layer_norm", "inputs": ["prob"], "outputs": ["norm"]},
            {"name": "sum", "op": "reduce_sum", "inputs": ["norm"], "outputs": ["sum"]},
        ],
        "outputs": ["sum"],
    }

    assert emit_graph_ir(payload) == '''module {
  func.func @main(%arg0: tensor<4xbf16>, %arg1: tensor<1x2x3xbf16>, %arg2: tensor<3x4xbf16>) -> tensor<1x2xbf16> {
    %v0 = "cost_model.matmul"(%arg1, %arg2) : (tensor<1x2x3xbf16>, tensor<3x4xbf16>) -> tensor<1x2x4xbf16>
    %v1 = "cost_model.add"(%v0, %arg0) : (tensor<1x2x4xbf16>, tensor<4xbf16>) -> tensor<1x2x4xbf16>
    %v2 = "cost_model.mul"(%v1, %v1) : (tensor<1x2x4xbf16>, tensor<1x2x4xbf16>) -> tensor<1x2x4xbf16>
    %v3 = "cost_model.relu"(%v2) : (tensor<1x2x4xbf16>) -> tensor<1x2x4xbf16>
    %v4 = "cost_model.softmax"(%v3) : (tensor<1x2x4xbf16>) -> tensor<1x2x4xbf16>
    %v5 = "cost_model.layer_norm"(%v4) : (tensor<1x2x4xbf16>) -> tensor<1x2x4xbf16>
    %v6 = "cost_model.reduce_sum"(%v5) : (tensor<1x2x4xbf16>) -> tensor<1x2xbf16>
    return %v6 : tensor<1x2xbf16>
  }
}
'''


def test_canonical_inputs_repeated_operands_and_dead_source_order_are_preserved() -> None:
    payload = {
        "name": "display label % ! \n",
        "target": {"peak_compute_gflops": 1000, "memory_bandwidth_gbps": 100},
        "inputs": {
            "z unused": {"shape": [1], "dtype": "u8"},
            "b": {"shape": [2], "dtype": "i32"},
            " a % ! \n": {"shape": [2], "dtype": "i32"},
        },
        "ops": [
            {"name": "z", "op": "mul", "inputs": [" a % ! \n", " a % ! \n"], "outputs": ["square"]},
            {"name": "y", "op": "add", "inputs": ["square", "square"], "outputs": ["sum"]},
            {"name": "a", "op": "relu", "inputs": ["b"], "outputs": ["dead"]},
        ],
        "outputs": ["sum"],
    }
    before = deepcopy(payload)
    graph = parse_graph(payload)
    graph_before = deepcopy(graph)
    expected = '''module {
  func.func @main(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>, %arg2: tensor<1xu8>) -> tensor<2xi32> {
    %v0 = "cost_model.mul"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
    %v1 = "cost_model.add"(%v0, %v0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
    %v2 = "cost_model.relu"(%arg1) : (tensor<2xi32>) -> tensor<2xi32>
    return %v1 : tensor<2xi32>
  }
}
'''

    assert emit_graph_ir(payload) == emit_graph_ir_graph(graph) == expected
    payload["inputs"] = dict(reversed(list(payload["inputs"].items())))
    assert emit_graph_ir(payload) == expected
    payload["inputs"] = before["inputs"]
    assert payload == before
    assert graph == graph_before
    payload["name"] = "renamed"
    payload.pop("target")
    for index, operation in enumerate(payload["ops"]):
        operation["name"] = f"renamed_{index}"
    assert emit_graph_ir(payload) == expected


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32", "f64", "i8", "u8", "i32", "i64", "bool"])
def test_single_scalar_output_and_scalar_operand_types(dtype: str) -> None:
    payload = {
        "name": "scalar",
        "inputs": {"x": {"shape": [], "dtype": dtype}},
        "ops": [{"name": "r", "op": "relu", "inputs": ["x"], "outputs": ["r"]}],
        "outputs": ["r"],
    }

    assert emit_graph_ir(payload) == f'''module {{
  func.func @main(%arg0: tensor<{dtype}>) -> tensor<{dtype}> {{
    %v0 = "cost_model.relu"(%arg0) : (tensor<{dtype}>) -> tensor<{dtype}>
    return %v0 : tensor<{dtype}>
  }}
}}
'''


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (b"{", "invalid JSON:"),
        (b"\xff", "invalid graph UTF-8"),
        (b"[]", "graph JSON must contain an object"),
        (b'{"name": "x", "name": "y"}', "duplicate object key 'name'"),
        (b'{"name": "x", "inputs": {}, "ops": [], "outputs": []}', "ops must contain at least one operation"),
    ],
)
def test_cli_malformed_graph_has_status_two_without_stdout(
    tmp_path: Path, contents: bytes, message: str
) -> None:
    path = tmp_path / "malformed.json"
    path.write_bytes(contents)

    result = _run_cli("--emit-graph-ir", str(path))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("error: ")
    assert message in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_missing_graph_has_concise_error(tmp_path: Path) -> None:
    result = _run_cli("--emit-graph-ir", str(tmp_path / "missing.json"))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("error: could not read graph ")
    assert "Traceback" not in result.stderr


def test_payload_api_uses_shared_graph_validation() -> None:
    with pytest.raises(GraphValidationError, match="ops must contain at least one operation"):
        emit_graph_ir({"name": "invalid", "inputs": {}, "ops": [], "outputs": []})


@pytest.mark.parametrize("entry_point", ["console", "module"])
def test_installed_graph_ir_entry_points_work_outside_checkout(
    tmp_path: Path, entry_point: str
) -> None:
    example = REPO_ROOT / "examples" / "matmul_bias_relu.json"
    expected = emit_graph_ir(json.loads(example.read_text(encoding="utf-8")))
    command = (
        [str(Path(sys.executable).with_name("cost-model"))]
        if entry_point == "console"
        else [sys.executable, "-m", "cost_model"]
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [*command, "--emit-graph-ir", str(example)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == expected
