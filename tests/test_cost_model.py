from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from cost_model import GraphValidationError, analyze_graph, load_graph


EXAMPLE_GRAPH = {
    "name": "matmul_bias_relu",
    "inputs": {
        "lhs": {"shape": [2, 3], "dtype": "f32"},
        "rhs": {"shape": [3, 4], "dtype": "f32"},
        "bias": {"shape": [4], "dtype": "f32"},
    },
    "ops": [
        {
            "name": "matmul",
            "op": "matmul",
            "inputs": ["lhs", "rhs"],
            "outputs": ["product"],
        },
        {
            "name": "add_bias",
            "op": "add",
            "inputs": ["product", "bias"],
            "outputs": ["biased"],
        },
        {
            "name": "relu",
            "op": "relu",
            "inputs": ["biased"],
            "outputs": ["output"],
        },
    ],
    "outputs": ["output"],
}


def test_estimates_per_op_totals_and_fusion_opportunity() -> None:
    report = analyze_graph(EXAMPLE_GRAPH)

    assert report["graph"] == "matmul_bias_relu"
    assert report["ops"] == [
        {
            "name": "matmul",
            "op": "matmul",
            "inputs": ["lhs", "rhs"],
            "outputs": ["product"],
            "shape": [2, 4],
            "dtype": "f32",
            "flops": 48,
            "bytes_read": 72,
            "bytes_written": 32,
            "arithmetic_intensity": pytest.approx(48 / 104),
            "roofline": None,
        },
        {
            "name": "add_bias",
            "op": "add",
            "inputs": ["product", "bias"],
            "outputs": ["biased"],
            "shape": [2, 4],
            "dtype": "f32",
            "flops": 8,
            "bytes_read": 48,
            "bytes_written": 32,
            "arithmetic_intensity": pytest.approx(8 / 80),
            "roofline": None,
        },
        {
            "name": "relu",
            "op": "relu",
            "inputs": ["biased"],
            "outputs": ["output"],
            "shape": [2, 4],
            "dtype": "f32",
            "flops": 8,
            "bytes_read": 32,
            "bytes_written": 32,
            "arithmetic_intensity": pytest.approx(8 / 64),
            "roofline": None,
        },
    ]
    assert report["totals"] == {
        "flops": 64,
        "bytes_read": 152,
        "bytes_written": 96,
        "arithmetic_intensity": pytest.approx(64 / 248),
        "roofline": None,
    }
    assert report["fusion"] == {
        "count": 1,
        "candidates": [
            {
                "producer": "add_bias",
                "consumer": "relu",
                "intermediate": "biased",
                "bytes_avoided": 64,
                "reason": "adjacent elementwise operations with a single-use intermediate",
            }
        ],
    }


def test_graph_can_be_loaded_from_json_file(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(EXAMPLE_GRAPH), encoding="utf-8")

    graph = load_graph(graph_path)

    assert graph.name == "matmul_bias_relu"
    assert len(graph.operations) == 3


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"ops": None}, "ops must be a list"),
        ({"inputs": {"lhs": {"shape": [2, 3], "dtype": "nope"}}}, "unsupported dtype"),
        ({"ops.0.op": "unknown"}, "unsupported operation"),
        ({"ops.0.inputs.0": "missing"}, "unknown input value"),
    ],
)
def test_invalid_graphs_raise_clear_validation_errors(change: dict, message: str) -> None:
    graph = deepcopy(EXAMPLE_GRAPH)
    change_path, replacement = next(iter(change.items()))
    target: object = graph
    path = change_path.split(".")
    for part in path[:-1]:
        target = target[int(part)] if part.isdigit() else target[part]  # type: ignore[index]
    last = path[-1]
    if isinstance(target, list):
        target[int(last)] = replacement
    else:
        target[last] = replacement  # type: ignore[index]

    with pytest.raises(GraphValidationError, match=message):
        analyze_graph(graph)


def test_invalid_matmul_shapes_are_rejected() -> None:
    graph = deepcopy(EXAMPLE_GRAPH)
    graph["inputs"]["rhs"]["shape"] = [5, 4]

    with pytest.raises(GraphValidationError, match="matmul dimensions do not align"):
        analyze_graph(graph)


def test_cli_outputs_json_and_nonzero_for_invalid_input(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(EXAMPLE_GRAPH), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": "src"}

    result = subprocess.run(
        [sys.executable, "-m", "cost_model", str(graph_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["totals"]["flops"] == 64

    graph_path.write_text("{not json", encoding="utf-8")
    invalid = subprocess.run(
        [sys.executable, "-m", "cost_model", str(graph_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert invalid.returncode == 2
    assert "error:" in invalid.stderr


def _matmul_graph_with_target(target: dict[str, object]) -> dict[str, object]:
    return {
        "name": "roofline_matmul",
        "inputs": {
            "lhs": {"shape": [64, 64], "dtype": "f32"},
            "rhs": {"shape": [64, 64], "dtype": "f32"},
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
        "target": target,
    }


@pytest.mark.parametrize(
    ("target", "bound", "expected_time_us"),
    [
        (
            {"peak_compute_gflops": 1000.0, "memory_bandwidth_gbps": 100.0},
            "compute",
            524288 / (1000 * 1000),
        ),
        (
            {"peak_compute_gflops": 1000.0, "memory_bandwidth_gbps": 10.0},
            "memory",
            49152 / (10 * 1000),
        ),
    ],
)
def test_roofline_reports_compute_and_memory_bounds(
    target: dict[str, object], bound: str, expected_time_us: float
) -> None:
    report = analyze_graph(_matmul_graph_with_target(target))

    expected = {
        "arithmetic_intensity": pytest.approx(524288 / 49152),
        "ridge_point": pytest.approx(
            10.0 if target["memory_bandwidth_gbps"] == 100.0 else 100.0
        ),
        "bound": bound,
        "estimated_time_us": pytest.approx(expected_time_us),
    }
    assert report["target"] == target
    assert report["ops"][0]["roofline"] == expected
    assert report["totals"]["roofline"] == expected


@pytest.mark.parametrize(
    "target",
    [
        {"peak_compute_gflops": 0, "memory_bandwidth_gbps": 100},
        {"peak_compute_gflops": -1, "memory_bandwidth_gbps": 100},
        {"peak_compute_gflops": 100, "memory_bandwidth_gbps": 0},
        {"peak_compute_gflops": 100, "memory_bandwidth_gbps": -1},
        {"peak_compute_gflops": "100", "memory_bandwidth_gbps": 100},
        {"peak_compute_gflops": True, "memory_bandwidth_gbps": 100},
        {"peak_compute_gflops": float("nan"), "memory_bandwidth_gbps": 100},
        {"peak_compute_gflops": 100},
    ],
)
def test_target_requires_positive_finite_compute_and_bandwidth(
    target: dict[str, object],
) -> None:
    with pytest.raises(GraphValidationError, match=r"target\."):
        analyze_graph(_matmul_graph_with_target(target))


def test_graph_without_target_has_explicit_null_roofline_analysis() -> None:
    report = analyze_graph(EXAMPLE_GRAPH)

    assert report["target"] is None
    assert all(operation["roofline"] is None for operation in report["ops"])
    assert report["totals"]["roofline"] is None
