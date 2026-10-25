from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from cost_model import GraphValidationError, analyze_graph, parse_graph


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOFLINE_ERROR = "roofline arithmetic exceeds finite floating-point range"
BATCH_TIME_ERROR = "benchmark estimated_time_us exceeds finite floating-point range"


def _relu_graph(elements: int, compute: float, bandwidth: float) -> dict[str, Any]:
    return {
        "name": "extreme_relu",
        "inputs": {"x": {"shape": [elements], "dtype": "f32"}},
        "ops": [
            {"name": "relu", "op": "relu", "inputs": ["x"], "outputs": ["output"]}
        ],
        "outputs": ["output"],
        "target": {
            "peak_compute_gflops": compute,
            "memory_bandwidth_gbps": bandwidth,
        },
    }


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cost_model", *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )


@pytest.mark.parametrize(
    ("elements", "compute", "bandwidth"),
    [
        pytest.param(10**1000, 100, 100, id="huge-shape-conversion"),
        pytest.param(1, 1e300, 1e-300, id="infinite-ridge-point"),
        pytest.param(10**100, 1e-300, 1, id="infinite-compute-time"),
        pytest.param(10**100, 1, 1e-300, id="infinite-memory-time"),
        pytest.param(1, 1e308, 1e308, id="infinite-scaled-compute-rate"),
        pytest.param(1, 1, 1e308, id="infinite-scaled-memory-rate"),
    ],
)
def test_extreme_roofline_arithmetic_fails_closed(
    tmp_path: Path, elements: int, compute: float, bandwidth: float
) -> None:
    payload = _relu_graph(elements, compute, bandwidth)
    # These are schema-valid inputs, not malformed targets or dimensions.
    parse_graph(payload)
    for optimize in (False, True):
        with pytest.raises(GraphValidationError) as error:
            analyze_graph(payload, optimize=optimize)
        assert str(error.value) == ROOFLINE_ERROR

    graph_path = tmp_path / "extreme.json"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    for options in ((), ("--compact",), ("--optimize",)):
        result = _run_cli(str(graph_path), *options)
        assert result.returncode == 2
        assert result.stdout == ""
        assert result.stderr == f"error: {ROOFLINE_ERROR}\n"


def test_unrepresentable_intensity_raises_validation_error_without_target() -> None:
    dimension = 10**1000
    payload = {
        "name": "extreme_matmul",
        "inputs": {
            "lhs": {"shape": [dimension, dimension], "dtype": "f32"},
            "rhs": {"shape": [dimension, dimension], "dtype": "f32"},
        },
        "ops": [
            {
                "name": "matmul",
                "op": "matmul",
                "inputs": ["lhs", "rhs"],
                "outputs": ["output"],
            }
        ],
        "outputs": ["output"],
    }
    parse_graph(payload)
    with pytest.raises(GraphValidationError) as error:
        analyze_graph(payload)
    assert str(error.value) == "arithmetic intensity exceeds finite floating-point range"


def test_huge_integer_counts_with_finite_intensity_remain_supported(tmp_path: Path) -> None:
    payload = _relu_graph(10**1000, 100, 100)
    del payload["target"]
    report = analyze_graph(payload)
    assert report["totals"]["flops"] == 10**1000
    assert report["totals"]["bytes_read"] == 4 * 10**1000
    assert report["totals"]["arithmetic_intensity"] == 0.125
    assert report["totals"]["roofline"] is None
    json.dumps(report, allow_nan=False)

    graph_path = tmp_path / "huge.json"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(["huge.json"]), encoding="utf-8")
    result = _run_cli("--benchmark", str(benchmark_path), "--compact")
    assert result.returncode == 0
    assert result.stderr == ""
    summary = json.loads(result.stdout)["summary"]["totals"]
    assert summary["flops"] == 10**1000
    assert summary["arithmetic_intensity"] == 0.125
    assert summary["estimated_time_us"] is None


def test_graph_total_time_overflow_rejects_finite_per_op_times() -> None:
    payload = _relu_graph(10**307, 0.001, 0.0005)
    assert analyze_graph(payload)["totals"]["roofline"]["estimated_time_us"] == 1.6e308
    payload["ops"].append(
        {"name": "again", "op": "relu", "inputs": ["output"], "outputs": ["final"]}
    )
    payload["outputs"] = ["final"]
    with pytest.raises(GraphValidationError) as error:
        analyze_graph(payload)
    assert str(error.value) == ROOFLINE_ERROR


@pytest.mark.parametrize("optimize", [False, True])
def test_batch_total_time_overflow_rejects_finite_case_times(
    tmp_path: Path, optimize: bool
) -> None:
    payload = _relu_graph(10**307, 0.001, 0.0005)
    graph_path = tmp_path / "finite.json"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            [
                {"name": "first", "path": "finite.json"},
                {"name": "second", "path": "finite.json"},
            ]
        ),
        encoding="utf-8",
    )
    options = ("--optimize",) if optimize else ()
    single = _run_cli(str(graph_path), "--compact", *options)
    assert single.returncode == 0
    assert single.stderr == ""
    report = json.loads(single.stdout)
    totals = report["optimized"]["totals"] if optimize else report["totals"]
    assert totals["roofline"]["estimated_time_us"] == 1.6e308

    for presentation in ((), ("--compact",)):
        result = _run_cli("--benchmark", str(benchmark_path), *options, *presentation)
        assert result.returncode == 2
        assert result.stdout == ""
        assert result.stderr == f"error: {BATCH_TIME_ERROR}\n"

    # Selection must still aggregate only the selected finite case.
    selected = _run_cli(
        "--benchmark", str(benchmark_path), "--case", "first", "--compact", *options
    )
    assert selected.returncode == 0
    assert selected.stderr == ""
    assert json.loads(selected.stdout)["summary"]["totals"]["estimated_time_us"] == 1.6e308

    # A target-less case disables the sum, even when the targeted subset overflows.
    del payload["target"]
    (tmp_path / "targetless.json").write_text(json.dumps(payload), encoding="utf-8")
    cases = json.loads(benchmark_path.read_text(encoding="utf-8"))
    cases.append({"name": "targetless", "path": "targetless.json"})
    benchmark_path.write_text(json.dumps(cases), encoding="utf-8")
    mixed = _run_cli("--benchmark", str(benchmark_path), "--compact", *options)
    assert mixed.returncode == 0
    assert mixed.stderr == ""
    mixed_report = json.loads(mixed.stdout)
    assert mixed_report["summary"]["totals"]["estimated_time_us"] is None
    json.dumps(mixed_report, allow_nan=False)
