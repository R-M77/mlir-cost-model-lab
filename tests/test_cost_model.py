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

SOFTMAX_GRAPH = {
    "name": "last_axis_softmax",
    "inputs": {"source": {"shape": [2, 4], "dtype": "f32"}},
    "ops": [
        {
            "name": "softmax",
            "op": "softmax",
            "inputs": ["source"],
            "outputs": ["result"],
        }
    ],
    "outputs": ["result"],
}

LAYER_NORM_GRAPH = {
    "name": "last_axis_layer_norm",
    "inputs": {"source": {"shape": [2, 3, 4], "dtype": "f32"}},
    "ops": [
        {
            "name": "layer_norm",
            "op": "layer_norm",
            "inputs": ["source"],
            "outputs": ["result"],
        }
    ],
    "outputs": ["result"],
}


REDUCE_SUM_GRAPH = {
    "name": "last_axis_reduce_sum",
    "inputs": {"source": {"shape": [2, 3, 4], "dtype": "f32"}},
    "ops": [
        {
            "name": "reduce_sum",
            "op": "reduce_sum",
            "inputs": ["source"],
            "outputs": ["result"],
        }
    ],
    "outputs": ["result"],
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


@pytest.mark.parametrize(
    ("dtype", "expected_bytes"),
    [("f16", 16), ("bf16", 16), ("f32", 32), ("f64", 64)],
)
def test_softmax_preserves_shape_dtype_and_reports_last_axis_cost(
    dtype: str, expected_bytes: int
) -> None:
    payload = deepcopy(SOFTMAX_GRAPH)
    payload["inputs"]["source"]["dtype"] = dtype

    report = analyze_graph(payload)

    assert report["ops"][0]["shape"] == [2, 4]
    assert report["ops"][0]["dtype"] == dtype
    assert report["ops"][0]["flops"] == 36
    assert report["ops"][0]["bytes_read"] == expected_bytes
    assert report["ops"][0]["bytes_written"] == expected_bytes
    assert report["fusion"] == {"count": 0, "candidates": []}


def test_softmax_counts_all_leading_dimensions_as_rows() -> None:
    payload = deepcopy(SOFTMAX_GRAPH)
    payload["inputs"]["source"]["shape"] = [2, 3, 4]

    report = analyze_graph(payload)

    assert report["ops"][0]["shape"] == [2, 3, 4]
    assert report["ops"][0]["flops"] == 108


@pytest.mark.parametrize("dtype", ["bool", "i8", "u8", "i32", "i64"])
def test_softmax_rejects_non_floating_input_dtypes(dtype: str) -> None:
    payload = deepcopy(SOFTMAX_GRAPH)
    payload["inputs"]["source"]["dtype"] = dtype

    with pytest.raises(
        GraphValidationError,
        match="softmax requires a floating-point input dtype",
    ):
        analyze_graph(payload)


def test_softmax_rejects_rank_zero_input() -> None:
    payload = deepcopy(SOFTMAX_GRAPH)
    payload["inputs"]["source"]["shape"] = []

    with pytest.raises(
        GraphValidationError,
        match="softmax requires an input tensor of rank at least 1",
    ):
        analyze_graph(payload)


@pytest.mark.parametrize("input_names", [[], ["source", "source"]])
def test_softmax_requires_exactly_one_input(input_names: list[str]) -> None:
    payload = deepcopy(SOFTMAX_GRAPH)
    payload["ops"][0]["inputs"] = input_names

    with pytest.raises(GraphValidationError, match="must contain exactly 1 value names"):
        analyze_graph(payload)


def test_default_cli_accepts_softmax_deterministically(tmp_path: Path) -> None:
    graph_path = tmp_path / "softmax.json"
    graph_path.write_text(json.dumps(SOFTMAX_GRAPH), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": "src"}

    first = subprocess.run(
        [sys.executable, "-m", "cost_model", str(graph_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    second = subprocess.run(
        [sys.executable, "-m", "cost_model", str(graph_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    report = json.loads(first.stdout)
    assert report["ops"][0]["shape"] == [2, 4]
    assert report["ops"][0]["flops"] == 36
    assert report["ops"][0]["bytes_read"] == 32
    assert report["ops"][0]["bytes_written"] == 32
    assert report["fusion"] == {"count": 0, "candidates": []}


@pytest.mark.parametrize(
    ("dtype", "dtype_bytes"),
    [("f16", 2), ("bf16", 2), ("f32", 4), ("f64", 8)],
)
@pytest.mark.parametrize(
    ("shape", "expected_flops"), [([4], 26), ([2, 3, 4], 156)]
)
def test_layer_norm_preserves_shape_dtype_and_reports_last_axis_cost(
    dtype: str, dtype_bytes: int, shape: list[int], expected_flops: int
) -> None:
    payload = deepcopy(LAYER_NORM_GRAPH)
    payload["inputs"]["source"]["dtype"] = dtype
    payload["inputs"]["source"]["shape"] = shape

    report = analyze_graph(payload)
    operation = report["ops"][0]
    expected_bytes = 1
    for dimension in shape:
        expected_bytes *= dimension
    expected_bytes *= dtype_bytes

    assert operation["shape"] == shape
    assert operation["dtype"] == dtype
    assert operation["flops"] == expected_flops
    assert operation["bytes_read"] == expected_bytes
    assert operation["bytes_written"] == expected_bytes
    assert report["fusion"] == {"count": 0, "candidates": []}


@pytest.mark.parametrize("dtype", ["bool", "i8", "u8", "i32", "i64"])
def test_layer_norm_rejects_non_floating_input_dtypes(dtype: str) -> None:
    payload = deepcopy(LAYER_NORM_GRAPH)
    payload["inputs"]["source"]["dtype"] = dtype

    with pytest.raises(
        GraphValidationError,
        match="layer_norm requires a floating-point input dtype",
    ):
        analyze_graph(payload)


def test_layer_norm_rejects_rank_zero_input() -> None:
    payload = deepcopy(LAYER_NORM_GRAPH)
    payload["inputs"]["source"]["shape"] = []

    with pytest.raises(
        GraphValidationError,
        match="layer_norm requires an input tensor of rank at least 1",
    ):
        analyze_graph(payload)


@pytest.mark.parametrize("input_names", [[], ["source", "source"]])
def test_layer_norm_requires_exactly_one_input(input_names: list[str]) -> None:
    payload = deepcopy(LAYER_NORM_GRAPH)
    payload["ops"][0]["inputs"] = input_names

    with pytest.raises(
        GraphValidationError,
        match="must contain exactly 1 value names",
    ):
        analyze_graph(payload)


def test_default_cli_accepts_layer_norm_deterministically(tmp_path: Path) -> None:
    graph_path = tmp_path / "layer_norm.json"
    graph_path.write_text(json.dumps(LAYER_NORM_GRAPH), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": "src"}

    first = subprocess.run(
        [sys.executable, "-m", "cost_model", str(graph_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    second = subprocess.run(
        [sys.executable, "-m", "cost_model", str(graph_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    report = json.loads(first.stdout)
    assert report["ops"][0]["shape"] == [2, 3, 4]
    assert report["ops"][0]["dtype"] == "f32"
    assert report["ops"][0]["flops"] == 156
    assert report["ops"][0]["bytes_read"] == 96
    assert report["ops"][0]["bytes_written"] == 96
    assert report["fusion"] == {"count": 0, "candidates": []}


@pytest.mark.parametrize(
    ("dtype", "dtype_bytes"),
    [
        ("f16", 2),
        ("bf16", 2),
        ("f32", 4),
        ("f64", 8),
        ("i8", 1),
        ("u8", 1),
        ("i32", 4),
        ("i64", 8),
    ],
)
def test_reduce_sum_preserves_numeric_dtype_and_reduces_final_axis(
    dtype: str, dtype_bytes: int
) -> None:
    payload = deepcopy(REDUCE_SUM_GRAPH)
    payload["inputs"]["source"]["dtype"] = dtype

    report = analyze_graph(payload)
    operation = report["ops"][0]

    assert operation["shape"] == [2, 3]
    assert operation["dtype"] == dtype
    assert operation["flops"] == 18
    assert operation["bytes_read"] == 24 * dtype_bytes
    assert operation["bytes_written"] == 6 * dtype_bytes
    assert report["fusion"] == {"count": 0, "candidates": []}


def test_reduce_sum_rank_one_input_produces_scalar_shape() -> None:
    payload = deepcopy(REDUCE_SUM_GRAPH)
    payload["inputs"]["source"] = {"shape": [5], "dtype": "i32"}

    operation = analyze_graph(payload)["ops"][0]

    assert operation["shape"] == []
    assert operation["dtype"] == "i32"
    assert operation["flops"] == 4
    assert operation["bytes_read"] == 20
    assert operation["bytes_written"] == 4


def test_reduce_sum_singleton_axis_counts_no_additions() -> None:
    payload = deepcopy(REDUCE_SUM_GRAPH)
    payload["inputs"]["source"] = {"shape": [2, 3, 1], "dtype": "f32"}

    operation = analyze_graph(payload)["ops"][0]

    assert operation["shape"] == [2, 3]
    assert operation["flops"] == 0
    assert operation["bytes_read"] == 24
    assert operation["bytes_written"] == 24


def test_reduce_sum_rejects_rank_zero_input() -> None:
    payload = deepcopy(REDUCE_SUM_GRAPH)
    payload["inputs"]["source"]["shape"] = []

    with pytest.raises(
        GraphValidationError,
        match="reduce_sum requires an input tensor of rank at least 1",
    ):
        analyze_graph(payload)


def test_reduce_sum_rejects_boolean_input_dtype() -> None:
    payload = deepcopy(REDUCE_SUM_GRAPH)
    payload["inputs"]["source"]["dtype"] = "bool"

    with pytest.raises(
        GraphValidationError,
        match="reduce_sum requires a numeric input dtype",
    ):
        analyze_graph(payload)


@pytest.mark.parametrize("input_names", [[], ["source", "source"]])
def test_reduce_sum_requires_exactly_one_input(input_names: list[str]) -> None:
    payload = deepcopy(REDUCE_SUM_GRAPH)
    payload["ops"][0]["inputs"] = input_names

    with pytest.raises(GraphValidationError, match="must contain exactly 1 value names"):
        analyze_graph(payload)


def test_default_cli_accepts_reduce_sum_deterministically(tmp_path: Path) -> None:
    graph_path = tmp_path / "reduce_sum.json"
    graph_path.write_text(json.dumps(REDUCE_SUM_GRAPH), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": "src"}

    first = subprocess.run(
        [sys.executable, "-m", "cost_model", str(graph_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    second = subprocess.run(
        [sys.executable, "-m", "cost_model", str(graph_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    report = json.loads(first.stdout)
    operation = report["ops"][0]
    assert operation["shape"] == [2, 3]
    assert operation["dtype"] == "f32"
    assert operation["flops"] == 18
    assert operation["bytes_read"] == 96
    assert operation["bytes_written"] == 24
    assert report["fusion"] == {"count": 0, "candidates": []}


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
