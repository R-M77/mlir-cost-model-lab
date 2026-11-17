"""Target-weighted scheduling reuses costs without changing baseline reports."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from cost_model import GraphValidationError, estimate_graph, parse_graph

ROOT = Path(__file__).resolve().parents[1]
BRANCH = {
    "name": "weighted_branches",
    "target": {"peak_compute_gflops": 1, "memory_bandwidth_gbps": 1000},
    "inputs": {
        "x": {"shape": [2, 2], "dtype": "f32"},
        "large": {"shape": [100, 100], "dtype": "f32"},
    },
    "ops": [
        {"name": "z_left", "op": "relu", "inputs": ["x"], "outputs": ["left"]},
        {"name": "a_right", "op": "matmul", "inputs": ["x", "x"], "outputs": ["right"]},
        {"name": "dead", "op": "matmul", "inputs": ["large", "large"], "outputs": ["unused"]},
        {"name": "m_join", "op": "add", "inputs": ["left", "right"], "outputs": ["result"]},
    ],
    "outputs": ["result", "x", "left"],
}


def run_cli(*options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cost_model", *options],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True, text=True, timeout=30, check=False,
    )


def test_weighted_path_end_to_end_overlaps_branches_and_preserves_graph(tmp_path: Path) -> None:
    from cost_model import analyze_weighted_critical_path
    from cost_model.analysis import analyze_weighted_critical_path as analysis_api

    assert analyze_weighted_critical_path is analysis_api
    graph = parse_graph(BRANCH)
    original = deepcopy(graph)
    baseline = estimate_graph(graph)
    left, right, dead, join = [op["roofline"]["estimated_time_us"] for op in baseline["ops"]]
    assert dead > right > left
    expected = {
        "graph": "weighted_branches",
        "operation_count": 4,
        "critical_path_time_us": right + join,
        "operations": [
            {"name": "z_left", "op": "relu", "output": "left", "duration_us": left,
             "start_us": 0.0, "finish_us": left},
            {"name": "a_right", "op": "matmul", "output": "right", "duration_us": right,
             "start_us": 0.0, "finish_us": right},
            {"name": "dead", "op": "matmul", "output": "unused", "duration_us": dead,
             "start_us": 0.0, "finish_us": dead},
            {"name": "m_join", "op": "add", "output": "result", "duration_us": join,
             "start_us": right, "finish_us": right + join},
        ],
        "outputs": [
            {"value": "result", "finish_us": right + join},
            {"value": "x", "finish_us": 0.0},
            {"value": "left", "finish_us": left},
        ],
    }
    report = analyze_weighted_critical_path(graph)
    # JSON text checks key order as well as values, without time rounding.
    assert json.dumps(report) == json.dumps(expected)
    assert analyze_weighted_critical_path(graph) == report
    assert estimate_graph(graph) == baseline
    assert graph == original

    path = tmp_path / "graph.json"
    source = json.dumps(BRANCH)
    path.write_text(source, encoding="utf-8")
    result = run_cli("--weighted-critical-path", str(path))
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == json.dumps(expected, indent=2) + "\n"
    assert path.read_text(encoding="utf-8") == source


@pytest.mark.parametrize("target", [
    BRANCH["target"],
    {"peak_compute_gflops": 1000, "memory_bandwidth_gbps": 1},
])
def test_serial_dependencies_add_time_while_parallel_branches_overlap(target: dict) -> None:
    from cost_model import analyze_weighted_critical_path

    payload = deepcopy(BRANCH)
    payload["target"] = target
    parallel_graph = parse_graph(payload)
    parallel = analyze_weighted_critical_path(parallel_graph)
    payload["ops"][1]["inputs"] = ["left", "x"]
    serial_graph = parse_graph(payload)
    serial = analyze_weighted_critical_path(serial_graph)
    durations = [op["roofline"]["estimated_time_us"] for op in estimate_graph(serial_graph)["ops"]]
    assert [op["duration_us"] for op in serial["operations"]] == durations
    assert [op["duration_us"] for op in parallel["operations"]] == durations
    left, right, _, join = durations
    assert parallel["critical_path_time_us"] == max(left, right) + join
    assert serial["operations"][1]["start_us"] == left
    assert serial["operations"][-1]["start_us"] == left + right
    assert serial["critical_path_time_us"] == left + right + join
    assert serial["critical_path_time_us"] > parallel["critical_path_time_us"]


def test_direct_input_only_outputs_finish_at_zero_despite_expensive_dead_ops() -> None:
    from cost_model import analyze_weighted_critical_path

    payload = deepcopy(BRANCH)
    payload["outputs"] = ["large", "x"]
    graph = parse_graph(payload)
    original = deepcopy(graph)
    report = analyze_weighted_critical_path(graph)
    assert report["critical_path_time_us"] == 0.0
    assert report["outputs"] == [
        {"value": "large", "finish_us": 0.0},
        {"value": "x", "finish_us": 0.0},
    ]
    assert report["operation_count"] == len(report["operations"]) == len(BRANCH["ops"])
    assert all(op["finish_us"] > 0 for op in report["operations"])
    assert graph == original


@pytest.mark.parametrize("compact", [False, True])
def test_cli_json_is_exact_deterministic_and_read_only(tmp_path: Path, compact: bool) -> None:
    from cost_model import analyze_weighted_critical_path

    path = tmp_path / "graph.json"
    source = json.dumps(BRANCH)
    path.write_text(source, encoding="utf-8")
    # A read-only report must leave both user input and shipped fixtures alone.
    fixtures = [ROOT / "examples/matmul_bias_relu.json", ROOT / "benchmarks/target_bounds.expected.json"]
    before = {fixture: fixture.read_bytes() for fixture in fixtures}
    report = analyze_weighted_critical_path(parse_graph(BRANCH))
    options = [str(path), "--weighted-critical-path", "--format", "json"]
    if compact:
        options.append("--compact")
    expected = (
        json.dumps(report, separators=(",", ":")) if compact else json.dumps(report, indent=2)
    ) + "\n"
    first, second = run_cli(*options), run_cli(*options)
    for result in (first, second):
        assert result.returncode == 0, result.stderr
        assert result.stderr == ""
        assert result.stdout == expected
    assert path.read_text(encoding="utf-8") == source
    assert {fixture: fixture.read_bytes() for fixture in fixtures} == before


@pytest.mark.parametrize("explicit_null", [False, True])
def test_targetless_api_and_cli_fail_cleanly(tmp_path: Path, explicit_null: bool) -> None:
    from cost_model import analyze_weighted_critical_path

    payload = deepcopy(BRANCH)
    payload.pop("target")
    if explicit_null:
        payload["target"] = None
    graph = parse_graph(payload)
    original = deepcopy(graph)
    with pytest.raises(GraphValidationError, match="^weighted critical path requires a target$"):
        analyze_weighted_critical_path(graph)
    assert graph == original
    path = tmp_path / "targetless.json"
    source = json.dumps(payload)
    path.write_text(source, encoding="utf-8")
    result = run_cli("--weighted-critical-path", str(path))
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "error: weighted critical path requires a target\n"
    assert path.read_text(encoding="utf-8") == source


@pytest.mark.parametrize("text,diagnostic", [
    ("{", "invalid JSON:"),
    (json.dumps({**BRANCH, "outputs": ["missing"]}), "unknown graph output value 'missing'"),
])
def test_malformed_graph_cli_has_no_report_or_traceback(
    tmp_path: Path, text: str, diagnostic: str,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(text, encoding="utf-8")
    result = run_cli("--weighted-critical-path", str(path))
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith(f"error: {diagnostic}")
    assert "Traceback" not in result.stderr
    assert len(result.stderr.splitlines()) == 1


def test_incompatible_mode_rejected_before_loading_independent_of_flag_order(tmp_path: Path) -> None:
    missing = str(tmp_path / "missing-graph.json")
    first = run_cli(missing, "--weighted-critical-path", "--optimize")
    second = run_cli("--optimize", "--weighted-critical-path", missing)
    for result in (first, second):
        assert result.returncode == 2
        assert result.stdout == ""
        assert "error: --weighted-critical-path cannot be combined with --optimize\n" in result.stderr
        assert "could not read" not in result.stderr
        assert "Traceback" not in result.stderr
    assert first.stderr == second.stderr


def test_cli_requires_a_single_graph_path() -> None:
    result = run_cli("--weighted-critical-path")
    assert result.returncode == 2
    assert result.stdout == ""
    assert "error: --weighted-critical-path requires a single graph path\n" in result.stderr
    assert "Traceback" not in result.stderr
