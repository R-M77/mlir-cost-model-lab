"""Dependency depth is structural analysis, separate from cost estimates."""
from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from cost_model import parse_graph

ROOT = Path(__file__).resolve().parents[1]
CHAIN = {
    "name": "dependency_chain",
    "target": {"peak_compute_gflops": 1000, "memory_bandwidth_gbps": 100},
    "inputs": {
        "x": {"shape": [4], "dtype": "f32"},
        "y": {"shape": [4], "dtype": "f32"},
    },
    "ops": [
        {"name": "z_first", "op": "relu", "inputs": ["x"], "outputs": ["first"]},
        {"name": "a_second", "op": "add", "inputs": ["y", "first"], "outputs": ["second"]},
        {"name": "m_third", "op": "mul", "inputs": ["second", "first"], "outputs": ["result"]},
    ],
    "outputs": ["result", "first", "x"],
}
EXPECTED = {
    "graph": "dependency_chain",
    "operation_count": 3,
    "critical_path_depth": 3,
    "operations": [
        {"name": "z_first", "op": "relu", "output": "first", "dependency_depth": 1},
        {"name": "a_second", "op": "add", "output": "second", "dependency_depth": 2},
        {"name": "m_third", "op": "mul", "output": "result", "dependency_depth": 3},
    ],
    "outputs": [
        {"value": "result", "dependency_depth": 3},
        {"value": "first", "dependency_depth": 1},
        {"value": "x", "dependency_depth": 0},
    ],
}


def run_cli(*options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cost_model", *options],
        cwd=ROOT, capture_output=True, text=True, timeout=30, check=False,
    )


def test_chain_report_preserves_schema_order_and_graph() -> None:
    from cost_model import analyze_dependency_depth

    graph = parse_graph(CHAIN)
    original = deepcopy(graph)
    report = analyze_dependency_depth(graph)
    assert report == EXPECTED
    # JSON equality also checks key insertion order, not just mapping equality.
    assert json.dumps(report) == json.dumps(EXPECTED)
    assert analyze_dependency_depth(graph) == report
    assert graph == original


def test_independent_and_dead_operations_do_not_inflate_output_depth() -> None:
    from cost_model import analyze_dependency_depth

    payload = deepcopy(CHAIN)
    payload["ops"] = [
        {"name": "live", "op": "relu", "inputs": ["x"], "outputs": ["live"]},
        {"name": "branch", "op": "relu", "inputs": ["y"], "outputs": ["branch"]},
        {"name": "dead", "op": "relu", "inputs": ["branch"], "outputs": ["dead"]},
        {"name": "dead_tail", "op": "relu", "inputs": ["dead"], "outputs": ["dead_tail"]},
        {"name": "join", "op": "add", "inputs": ["x", "live"], "outputs": ["result"]},
    ]
    payload["outputs"] = ["result", "branch", "x"]
    report = analyze_dependency_depth(parse_graph(payload))
    assert report["operation_count"] == 5
    assert report["critical_path_depth"] == 2
    assert [op["name"] for op in report["operations"]] == [
        "live", "branch", "dead", "dead_tail", "join",
    ]
    assert [op["dependency_depth"] for op in report["operations"]] == [1, 1, 2, 3, 2]
    assert report["outputs"] == [
        {"value": "result", "dependency_depth": 2},
        {"value": "branch", "dependency_depth": 1},
        {"value": "x", "dependency_depth": 0},
    ]


def test_direct_input_only_output_has_zero_critical_depth() -> None:
    from cost_model import analyze_dependency_depth

    payload = deepcopy(CHAIN)
    payload["outputs"] = ["x"]
    report = analyze_dependency_depth(parse_graph(payload))
    assert report["critical_path_depth"] == 0
    assert report["outputs"] == [{"value": "x", "dependency_depth": 0}]
    assert report["operations"] == EXPECTED["operations"]


@pytest.mark.parametrize("compact", [False, True])
def test_cli_report_is_exact_and_deterministic(tmp_path: Path, compact: bool) -> None:
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(CHAIN), encoding="utf-8")
    options = [str(path), "--critical-path"]
    if compact:
        options.append("--compact")
    result = run_cli(*options)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    expected = (
        json.dumps(EXPECTED, separators=(",", ":"))
        if compact else json.dumps(EXPECTED, indent=2)
    ) + "\n"
    assert result.stdout == expected
    assert run_cli(*options).stdout == expected


@pytest.mark.parametrize("options,flag", [
    (("--benchmark", "missing.json"), "--benchmark"),
    (("--optimize",), "--optimize"),
    (("--lower-matmul",), "--lower-matmul"),
    (("--emit-mlir",), "--emit-mlir"),
    (("--format", "markdown"), "--format markdown"),
    (("--budget", "missing.json"), "--budget"),
    (("--compare-baseline", "missing.json"), "--compare-baseline"),
    (("--case", "example"), "--case"),
    (("--omit-paths",), "--omit-paths"),
])
def test_cli_rejects_incompatible_mode_before_loading(
    options: tuple[str, ...], flag: str,
) -> None:
    result = run_cli("missing-graph.json", "--critical-path", *options)
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("usage:")
    assert f"error: --critical-path cannot be combined with {flag}\n" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_requires_one_graph_path() -> None:
    result = run_cli("--critical-path")
    assert result.returncode == 2
    assert result.stdout == ""
    assert "error: --critical-path requires a single graph path\n" in result.stderr


@pytest.mark.parametrize("text,diagnostic", [
    ("{", "invalid JSON:"),
    (json.dumps({**CHAIN, "outputs": ["missing"]}), "unknown graph output value 'missing'"),
])
def test_cli_malformed_graph_uses_existing_error_path(
    tmp_path: Path, text: str, diagnostic: str,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(text, encoding="utf-8")
    result = run_cli("--critical-path", str(path))
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith(f"error: {diagnostic}")
    assert "Traceback" not in result.stderr
