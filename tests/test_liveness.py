"""Boundary liveness is separate from traffic estimates and optimization."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from cost_model import parse_graph

ROOT = Path(__file__).resolve().parents[1]
BRANCH = {
    "name": "live_storage",
    "inputs": {
        "y": {"shape": [1, 3], "dtype": "f32"},
        "x": {"shape": [2, 1], "dtype": "f32"},
    },
    "ops": [
        {"name": "z_first", "op": "mul", "inputs": ["x", "y"], "outputs": ["z"]},
        {"name": "a_second", "op": "mul", "inputs": ["z", "z"], "outputs": ["a"]},
        {"name": "m_join", "op": "add", "inputs": ["a", "z"], "outputs": ["result"]},
    ],
    "outputs": ["result", "x"],
}
EXPECTED = {
    "graph": "live_storage",
    "operation_count": 3,
    "initial_live_values": ["x", "y"],
    "initial_live_bytes": 20,
    "peak_live_bytes": 56,
    "peak_after": "a_second",
    "operations": [
        {"name": "z_first", "op": "mul", "output": "z", "live_values": ["x", "z"], "live_bytes": 32},
        {"name": "a_second", "op": "mul", "output": "a", "live_values": ["a", "x", "z"], "live_bytes": 56},
        {"name": "m_join", "op": "add", "output": "result", "live_values": ["result", "x"], "live_bytes": 32},
    ],
}


def run_cli(*options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cost_model", *options],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True, text=True, timeout=30, check=False,
    )


def test_liveness_end_to_end_reports_last_uses_and_peak(tmp_path: Path) -> None:
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(BRANCH), encoding="utf-8")
    result = run_cli("--liveness", str(path))
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == json.dumps(EXPECTED, indent=2) + "\n"
    assert run_cli("--liveness", str(path)).stdout == result.stdout

    from cost_model import analyze_liveness
    from cost_model.analysis import analyze_liveness as analysis_api

    assert analyze_liveness is analysis_api
    graph = parse_graph(BRANCH)
    original = deepcopy(graph)
    report = analyze_liveness(graph)
    assert json.dumps(report) == json.dumps(EXPECTED)
    assert analyze_liveness(graph) == report
    assert graph == original


def test_direct_input_output_retains_storage_with_dead_ops_and_unused_input() -> None:
    from cost_model import analyze_liveness

    graph = parse_graph({
        "name": "dead_branch",
        "inputs": {
            "unused": {"shape": [1000], "dtype": "f64"},
            "y": {"shape": [3], "dtype": "f64"},
            "x": {"shape": [], "dtype": "bf16"},
        },
        "ops": [
            {"name": "z_dead", "op": "relu", "inputs": ["y"], "outputs": ["z"]},
            {"name": "a_dead", "op": "relu", "inputs": ["z"], "outputs": ["a"]},
        ],
        "outputs": ["x"],
    })
    original = deepcopy(graph)
    assert analyze_liveness(graph) == {
        "graph": "dead_branch",
        "operation_count": 2,
        "initial_live_values": ["x", "y"],
        "initial_live_bytes": 26,
        "peak_live_bytes": 26,
        "peak_after": None,
        "operations": [
            {"name": "z_dead", "op": "relu", "output": "z", "live_values": ["x", "z"], "live_bytes": 26},
            {"name": "a_dead", "op": "relu", "output": "a", "live_values": ["x"], "live_bytes": 2},
        ],
    }
    assert graph == original


def test_declared_intermediate_output_survives_last_use_through_exit() -> None:
    from cost_model import analyze_liveness

    payload = deepcopy(BRANCH)
    payload["outputs"] = ["result", "x", "a"]
    report = analyze_liveness(parse_graph(payload))
    assert report["operations"][-1] == {
        "name": "m_join", "op": "add", "output": "result",
        "live_values": ["a", "result", "x"], "live_bytes": 56,
    }
    assert report["peak_after"] == "a_second"
    payload["outputs"].reverse()
    assert analyze_liveness(parse_graph(payload)) == report


def test_peak_ties_choose_earliest_operation_not_name_order() -> None:
    from cost_model import analyze_liveness

    payload = deepcopy(BRANCH)
    payload["ops"].insert(2, {
        "name": "0_tied", "op": "relu", "inputs": ["a"], "outputs": ["b"],
    })
    payload["ops"][-1]["inputs"] = ["b", "z"]
    # Input mapping order does not change the report's lexical live-value order.
    payload["inputs"] = dict(reversed(list(payload["inputs"].items())))
    report = analyze_liveness(parse_graph(payload))
    assert report["initial_live_values"] == ["x", "y"]
    assert report["peak_live_bytes"] == 56
    assert report["peak_after"] == "a_second"
    assert report["operations"][2] == {
        "name": "0_tied", "op": "relu", "output": "b",
        "live_values": ["b", "x", "z"], "live_bytes": 56,
    }


def test_repeated_last_use_counts_scalar_storage_once_and_releases_it() -> None:
    from cost_model import analyze_liveness

    report = analyze_liveness(parse_graph({
        "name": "scalar",
        "inputs": {"x": {"shape": [], "dtype": "bf16"}},
        "ops": [{"name": "square", "op": "mul", "inputs": ["x", "x"], "outputs": ["y"]}],
        "outputs": ["y"],
    }))
    assert report["initial_live_values"] == ["x"]
    assert report["initial_live_bytes"] == report["peak_live_bytes"] == 2
    assert report["peak_after"] is None
    assert report["operations"] == [{
        "name": "square", "op": "mul", "output": "y",
        "live_values": ["y"], "live_bytes": 2,
    }]


@pytest.mark.parametrize("options", [("--compact",), ("--format", "json")])
def test_cli_supports_structural_json_presentation(tmp_path: Path, options: tuple[str, ...]) -> None:
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(BRANCH), encoding="utf-8")
    result = run_cli(str(path), "--liveness", *options)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    expected = (
        json.dumps(EXPECTED, separators=(",", ":"))
        if "--compact" in options else json.dumps(EXPECTED, indent=2)
    ) + "\n"
    assert result.stdout == expected


@pytest.mark.parametrize("options,flag", [
    (("--benchmark", "missing.json"), "--benchmark"),
    (("--optimize",), "--optimize"),
    (("--critical-path",), "--critical-path"),
    (("--lower-matmul",), "--lower-matmul"),
    (("--emit-mlir",), "--emit-mlir"),
    (("--format", "markdown"), "--format markdown"),
    (("--budget", "missing.json"), "--budget"),
    (("--compare-baseline", "missing.json"), "--compare-baseline"),
    (("--case", "example"), "--case"),
    (("--omit-paths",), "--omit-paths"),
    (("--lowering-mode", "tiled"), "--lowering-mode"),
    (("--tile-size", "8"), "--tile-size"),
    (("--vector-width", "8"), "--vector-width"),
])
def test_cli_rejects_incompatible_mode_before_loading(
    tmp_path: Path, options: tuple[str, ...], flag: str,
) -> None:
    path = str(tmp_path / "missing-graph.json")
    first = run_cli(path, "--liveness", *options)
    second = run_cli(*options, "--liveness", path)
    for result in (first, second):
        assert result.returncode == 2
        assert result.stdout == ""
        assert result.stderr.startswith("usage:")
        assert f"error: --liveness cannot be combined with {flag}\n" in result.stderr
        assert "Traceback" not in result.stderr
    assert first.stderr == second.stderr


def test_cli_requires_one_graph_path() -> None:
    result = run_cli("--liveness")
    assert result.returncode == 2
    assert result.stdout == ""
    assert "error: --liveness requires a single graph path\n" in result.stderr


@pytest.mark.parametrize("text,diagnostic", [
    ("{", "invalid JSON:"),
    (json.dumps({**BRANCH, "outputs": ["missing"]}), "unknown graph output value 'missing'"),
])
def test_cli_malformed_graph_uses_existing_error_path(
    tmp_path: Path, text: str, diagnostic: str,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(text, encoding="utf-8")
    result = run_cli("--liveness", str(path))
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith(f"error: {diagnostic}")
    assert "Traceback" not in result.stderr
