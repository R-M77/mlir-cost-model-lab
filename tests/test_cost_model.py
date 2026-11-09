from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import UserDict
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from cost_model import (
    GraphValidationError,
    Target,
    Tensor,
    analyze_graph,
    estimate_graph,
    load_graph,
    parse_graph,
)
from cost_model.models import DTYPE_BYTES


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


def test_target_aware_relu_infers_shape_and_reports_exact_memory_bound_cost() -> None:
    target = {"peak_compute_gflops": 2.0, "memory_bandwidth_gbps": 1.0}
    payload = {
        "name": "target_aware_relu",
        "inputs": {"source": {"shape": [2, 3], "dtype": "f32"}},
        "ops": [
            {
                "name": "relu",
                "op": "relu",
                "inputs": ["source"],
                "outputs": ["result"],
            }
        ],
        "outputs": ["result"],
        "target": target,
    }

    report = analyze_graph(payload)
    operation = report["ops"][0]

    assert operation["shape"] == [2, 3]
    assert operation["dtype"] == "f32"
    assert operation["flops"] == 6
    assert operation["bytes_read"] == 24
    assert operation["bytes_written"] == 24
    assert operation["arithmetic_intensity"] == pytest.approx(6 / 48)
    assert report["target"] == target
    assert operation["roofline"] == {
        "arithmetic_intensity": pytest.approx(6 / 48),
        "ridge_point": 2.0,
        "bound": "memory",
        "estimated_time_us": pytest.approx(0.048),
    }


def test_broadcasted_mul_infers_shape_and_reports_exact_target_aware_cost() -> None:
    target = {"peak_compute_gflops": 1, "memory_bandwidth_gbps": 1}
    payload = {
        "name": "broadcasted_mul",
        "inputs": {
            "lhs": {"shape": [2, 1, 3], "dtype": "f32"},
            "rhs": {"shape": [1, 4, 1], "dtype": "f32"},
        },
        "ops": [
            {
                "name": "mul",
                "op": "mul",
                "inputs": ["lhs", "rhs"],
                "outputs": ["result"],
            }
        ],
        "outputs": ["result"],
        "target": target,
    }

    report = analyze_graph(payload)
    operation = report["ops"][0]
    totals = report["totals"]
    expected_roofline = {
        "arithmetic_intensity": pytest.approx(24 / 136),
        "ridge_point": 1.0,
        "bound": "memory",
        "estimated_time_us": pytest.approx(0.136),
    }

    assert operation["op"] == "mul"
    assert operation["shape"] == [2, 4, 3]
    assert operation["dtype"] == "f32"
    assert operation["flops"] == 24
    assert operation["bytes_read"] == 40
    assert operation["bytes_written"] == 96
    assert operation["arithmetic_intensity"] == pytest.approx(24 / 136)
    assert totals["flops"] == operation["flops"]
    assert totals["bytes_read"] == operation["bytes_read"]
    assert totals["bytes_written"] == operation["bytes_written"]
    assert totals["arithmetic_intensity"] == pytest.approx(24 / 136)
    assert report["target"] == target
    assert operation["roofline"] == expected_roofline
    assert totals["roofline"] == expected_roofline


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


UNKNOWN_GRAPH_FIELD_CASES = [
    pytest.param((), "graph", id="graph-root"),
    pytest.param(("inputs", "lhs"), "inputs['lhs']", id="tensor-spec"),
    pytest.param(("target",), "target", id="target"),
    pytest.param(("ops", 1), "ops[1]", id="operation"),
]


@pytest.mark.parametrize(("path", "label"), UNKNOWN_GRAPH_FIELD_CASES)
@pytest.mark.parametrize("keys", [("typo",), ("z_extra", "a_extra"), ("z'quoted", "a_extra")])
@pytest.mark.parametrize("mapping_type", [dict, UserDict, MappingProxyType])
def test_graph_schema_rejects_unknown_fields_in_mappings_and_files(
    tmp_path: Path, path: tuple, label: str, keys: tuple[str, ...], mapping_type
) -> None:
    payload = deepcopy(EXAMPLE_GRAPH)
    payload["target"] = {"peak_compute_gflops": 100, "memory_bandwidth_gbps": 10}
    spec = payload
    for key in path:
        spec = spec[key]
    message = f"{label} has unknown field(s): " + ", ".join(repr(key) for key in sorted(keys))
    original = deepcopy(payload)

    # Insertion order must not affect diagnostics, nor may validation mutate input.
    for ordered_keys in (keys, tuple(reversed(keys))):
        for key in ordered_keys:
            spec[key] = None
        invalid = deepcopy(payload)
        with pytest.raises(GraphValidationError) as error:
            parse_graph(mapping_type(payload))
        assert str(error.value) == message
        for optimize in (False, True):
            with pytest.raises(GraphValidationError) as error:
                analyze_graph(mapping_type(payload), optimize=optimize)
            assert str(error.value) == message
        assert payload == invalid

        graph_path = tmp_path / "unknown_fields.json"
        graph_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(GraphValidationError) as error:
            load_graph(graph_path)
        assert str(error.value) == message
        for key in keys:
            del spec[key]
    assert payload == original


@pytest.mark.parametrize(("path", "label"), UNKNOWN_GRAPH_FIELD_CASES)
def test_graph_schema_rejects_non_string_mapping_fields_without_type_error(
    path: tuple, label: str
) -> None:
    payload = deepcopy(EXAMPLE_GRAPH)
    payload["target"] = {"peak_compute_gflops": 100, "memory_bandwidth_gbps": 10}
    spec: dict = payload
    for key in path:
        spec = spec[key]
    spec.update({1: None, "typo": None, None: None})

    with pytest.raises(GraphValidationError) as error:
        parse_graph(payload)
    assert str(error.value) == f"{label} has unknown field(s): 'typo', 1, None"


@pytest.mark.parametrize(("path", "label"), UNKNOWN_GRAPH_FIELD_CASES)
@pytest.mark.parametrize(
    "options", [[], ["--optimize"], ["--lower-matmul"], ["--emit-mlir"]]
)
def test_cli_rejects_unknown_graph_fields_deterministically(
    tmp_path: Path, path: tuple, label: str, options: list[str]
) -> None:
    payload = deepcopy(EXAMPLE_GRAPH)
    payload["target"] = {"peak_compute_gflops": 100, "memory_bandwidth_gbps": 10}
    spec = payload
    for key in path:
        spec = spec[key]
    spec.update(z_extra=None, a_extra=None)
    graph_path = tmp_path / "unknown_fields.json"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")

    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-m", "cost_model", *options, str(graph_path)],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": "src"},
        )
        assert result.returncode == 2
        assert result.stdout == ""
        assert result.stderr == f"error: {label} has unknown field(s): 'a_extra', 'z_extra'\n"


@pytest.mark.parametrize("mapping_type", [UserDict, MappingProxyType])
@pytest.mark.parametrize("target", [None, {"peak_compute_gflops": 100, "memory_bandwidth_gbps": 10}])
@pytest.mark.parametrize("input_name", ["metadata", "shape", "target", "odd['name']"])
def test_graph_schema_preserves_nested_mappings_and_dynamic_input_names(
    mapping_type, target: dict | None, input_name: str
) -> None:
    payload = deepcopy(SOFTMAX_GRAPH)
    payload["target"] = target
    payload["inputs"][input_name] = payload["inputs"].pop("source")
    payload["ops"][0]["inputs"] = [input_name]
    original = deepcopy(payload)
    expected = parse_graph(payload)
    payload["inputs"][input_name] = mapping_type(payload["inputs"][input_name])
    payload["inputs"] = mapping_type(payload["inputs"])
    payload["ops"] = [mapping_type(op) for op in payload["ops"]]
    if target is not None:
        payload["target"] = mapping_type(target)

    assert parse_graph(mapping_type(payload)) == expected
    assert analyze_graph(mapping_type(payload)) == analyze_graph(original)
    assert payload == original


def test_strict_graph_schema_preserves_dynamic_benchmark_and_budget_case_names(
    tmp_path: Path,
) -> None:
    payload = deepcopy(SOFTMAX_GRAPH)
    payload["inputs"]["metadata"] = payload["inputs"].pop("source")
    payload["ops"][0]["inputs"] = ["metadata"]
    (tmp_path / "graph.json").write_text(json.dumps(payload), encoding="utf-8")
    case_name = "arbitrary.case['metadata']"
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        json.dumps([{"name": case_name, "path": "graph.json"}]), encoding="utf-8"
    )
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(
        json.dumps({"aggregate": {"flops": 1000}, "cases": {case_name: {"flops": 1000}}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable, "-m", "cost_model", "--benchmark", str(benchmark_path),
            "--case", case_name, "--budget", str(budget_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    assert result.returncode == 0
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["cases"] == [
        {"name": case_name, "path": "graph.json", "report": analyze_graph(payload)}
    ]
    assert report["summary"]["case_count"] == 1


DUPLICATE_GRAPH_KEY_CASES = [
    pytest.param(
        '"name": "last_axis_softmax"',
        '"name": "ignored", "name": "last_axis_softmax"',
        "name",
        id="top-level-name",
    ),
    pytest.param(
        '}], "outputs": ["result"]}',
        '}], "outputs": ["result"], "outputs": ["result"]}',
        "outputs",
        id="top-level-identical-outputs",
    ),
    pytest.param(
        '"source": {"shape": [2, 4], "dtype": "f32"}',
        '"source": {"shape": [2, 4], "dtype": "f32"}, '
        '"source": {"shape": [2, 4], "dtype": "f32"}',
        "source",
        id="input-name-identical-specs",
    ),
    pytest.param(
        '"shape": [2, 4]',
        '"shape": [1, 4], "shape": [2, 4]',
        "shape",
        id="tensor-shape",
    ),
    pytest.param(
        '"dtype": "f32"',
        '"dtype": "f32", "dtype": "f32"',
        "dtype",
        id="tensor-identical-dtype",
    ),
    pytest.param(
        '"op": "softmax"',
        '"op": "relu", "op": "softmax"',
        "op",
        id="operation-kind",
    ),
    pytest.param(
        '"inputs": ["source"]',
        '"inputs": ["source"], "inputs": ["source"]',
        "inputs",
        id="operation-identical-inputs",
    ),
    pytest.param(
        '"name": "last_axis_softmax"',
        r'"name": "ignored", "na\u006de": "last_axis_softmax"',
        "name",
        id="escaped-equivalent-key",
    ),
    pytest.param(
        '"name": "last_axis_softmax"',
        '"target": {"peak_compute_gflops": 1, "peak_compute_gflops": 1, '
        '"memory_bandwidth_gbps": 1}, "name": "last_axis_softmax"',
        "peak_compute_gflops",
        id="target-identical-rate",
    ),
    pytest.param(
        '"name": "last_axis_softmax"',
        '"metadata": [{"nested": {"tag": null, "tag": null}}], '
        '"name": "last_axis_softmax"',
        "tag",
        id="deeply-nested-unused-field",
    ),
]


@pytest.mark.parametrize(("original", "replacement", "key"), DUPLICATE_GRAPH_KEY_CASES)
def test_load_graph_rejects_duplicate_json_object_keys(
    tmp_path: Path, original: str, replacement: str, key: str
) -> None:
    text = json.dumps(SOFTMAX_GRAPH)
    assert text.count(original) == 1
    graph_path = tmp_path / "duplicate_keys.json"
    graph_path.write_text(text.replace(original, replacement), encoding="utf-8")

    with pytest.raises(GraphValidationError) as error:
        load_graph(graph_path)
    assert str(error.value) == f"invalid graph JSON: duplicate object key {key!r}"


@pytest.mark.parametrize(("original", "replacement", "key"), DUPLICATE_GRAPH_KEY_CASES)
@pytest.mark.parametrize(
    "options", [[], ["--optimize"], ["--lower-matmul"], ["--emit-mlir"]]
)
def test_cli_rejects_duplicate_graph_json_object_keys_deterministically(
    tmp_path: Path, original: str, replacement: str, key: str, options: list[str]
) -> None:
    text = json.dumps(SOFTMAX_GRAPH)
    assert text.count(original) == 1
    graph_path = tmp_path / "duplicate_keys.json"
    graph_path.write_text(text.replace(original, replacement), encoding="utf-8")

    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-m", "cost_model", *options, str(graph_path)],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": "src"},
        )
        assert result.returncode == 2
        assert result.stdout == ""
        assert result.stderr == f"error: invalid graph JSON: duplicate object key {key!r}\n"


@pytest.mark.parametrize("optimize", [False, True], ids=["baseline", "optimized"])
def test_graph_json_preserves_repeated_list_values_and_unique_outputs(
    tmp_path: Path, optimize: bool
) -> None:
    payload = deepcopy(SOFTMAX_GRAPH)
    payload["inputs"]["source"]["shape"] = [2, 2]
    payload["ops"][0].update(op="mul", inputs=["source", "source"])
    payload["outputs"] = ["result", "source"]
    graph_path = tmp_path / "repeated_values.json"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")

    graph = load_graph(graph_path)
    assert graph == parse_graph(payload)
    assert graph.operations[0].inputs == ("source", "source")
    assert graph.inputs["source"].shape == (2, 2)
    assert graph.outputs == ("result", "source")

    result = subprocess.run(
        [
            sys.executable, "-m", "cost_model",
            *(["--optimize"] if optimize else []), str(graph_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    assert result.returncode == 0
    assert result.stderr == ""
    assert json.loads(result.stdout) == analyze_graph(payload, optimize=optimize)


@pytest.mark.parametrize(
    ("outputs", "duplicate"),
    [
        (["output", "output"], "output"),
        (["output", "product", "output"], "output"),
        (["biased", "product", "product", "biased"], "product"),
        (["lhs", "output", "lhs"], "lhs"),
    ],
)
def test_duplicate_graph_outputs_are_rejected_by_parser_and_api(
    outputs: list[str], duplicate: str
) -> None:
    payload = deepcopy(EXAMPLE_GRAPH)
    payload["outputs"] = outputs
    message = f"duplicate graph output value {duplicate!r}"

    with pytest.raises(GraphValidationError) as error:
        parse_graph(payload)
    assert str(error.value) == message
    for optimize in (False, True):
        with pytest.raises(GraphValidationError) as error:
            analyze_graph(payload, optimize=optimize)
        assert str(error.value) == message


@pytest.mark.parametrize(
    ("outputs", "duplicate"),
    [
        (["output", "output"], "output"),
        (["output", "product", "output"], "output"),
        (["biased", "product", "product", "biased"], "product"),
        (["lhs", "output", "lhs"], "lhs"),
    ],
)
@pytest.mark.parametrize("options", [[], ["--optimize"]])
def test_cli_rejects_duplicate_graph_outputs_deterministically(
    tmp_path: Path, outputs: list[str], duplicate: str, options: list[str]
) -> None:
    payload = deepcopy(EXAMPLE_GRAPH)
    payload["outputs"] = outputs
    graph_path = tmp_path / "duplicate_outputs.json"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")

    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-m", "cost_model", *options, str(graph_path)],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": "src"},
        )
        assert result.returncode == 2
        assert result.stdout == ""
        assert result.stderr == f"error: duplicate graph output value {duplicate!r}\n"


def test_unique_graph_outputs_preserve_declaration_order() -> None:
    payload = deepcopy(EXAMPLE_GRAPH)
    payload["outputs"] = ["output", "lhs", "biased", "product"]

    graph = parse_graph(payload)
    report = analyze_graph(payload, optimize=True)

    assert graph.outputs == ("output", "lhs", "biased", "product")
    assert report["optimized"]["graph_outputs"] == payload["outputs"]


def test_repeated_operation_operands_remain_valid() -> None:
    payload = deepcopy(SOFTMAX_GRAPH)
    payload["ops"][0].update(op="mul", inputs=["source", "source"])

    graph = parse_graph(payload)
    report = analyze_graph(payload)

    assert graph.operations[0].inputs == ("source", "source")
    assert report["ops"][0]["inputs"] == ["source", "source"]
    assert report["ops"][0]["bytes_read"] == 64


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


def test_compact_cli_classifies_exact_roofline_ridge_tie_as_compute_bound(
    tmp_path: Path,
) -> None:
    graph = {
        "name": "relu_ridge_tie",
        "inputs": {"source": {"shape": [1], "dtype": "f32"}},
        "ops": [
            {
                "name": "relu",
                "op": "relu",
                "inputs": ["source"],
                "outputs": ["result"],
            }
        ],
        "outputs": ["result"],
        "target": {
            "peak_compute_gflops": 1,
            "memory_bandwidth_gbps": 8,
        },
    }
    graph_path = tmp_path / "relu_ridge_tie.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": "src"}

    first = subprocess.run(
        [sys.executable, "-m", "cost_model", "--compact", str(graph_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    second = subprocess.run(
        [sys.executable, "-m", "cost_model", "--compact", str(graph_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    report = json.loads(first.stdout)
    assert report["ops"][0]["arithmetic_intensity"] == 0.125
    assert report["totals"]["arithmetic_intensity"] == 0.125
    for roofline in (report["ops"][0]["roofline"], report["totals"]["roofline"]):
        assert roofline["arithmetic_intensity"] == 0.125
        assert roofline["ridge_point"] == 0.125
        assert roofline["bound"] == "compute"
        assert roofline["estimated_time_us"] == 0.001


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


@pytest.mark.parametrize("field", ["peak_compute_gflops", "memory_bandwidth_gbps"])
def test_oversized_integer_target_returns_validation_and_cli_errors(
    tmp_path: Path, field: str
) -> None:
    target: dict[str, object] = {"peak_compute_gflops": 100, "memory_bandwidth_gbps": 100}
    target[field] = 10**1000
    payload = _matmul_graph_with_target(target)
    graph_path = tmp_path / "oversized_target.json"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    message = f"target.{field} must be a positive finite number"
    with pytest.raises(GraphValidationError) as error:
        load_graph(graph_path)
    assert str(error.value) == message

    result = subprocess.run(
        [sys.executable, "-m", "cost_model", str(graph_path)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == f"error: {message}\n"


def test_direct_estimate_graph_rejects_deleted_operations_container() -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    object.__delattr__(graph, "operations")

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == "graph operations must be a tuple or list"


@pytest.mark.parametrize("field", ["target", "values"])
def test_direct_estimate_graph_rejects_deleted_graph_fields(field: str) -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    object.__delattr__(graph, field)
    message = (
        "graph target must be present"
        if field == "target"
        else "graph values must be a mapping"
    )

    for _ in range(2):
        with pytest.raises(GraphValidationError) as error:
            estimate_graph(graph)
        assert str(error.value) == message
    assert field not in vars(graph)


@pytest.mark.parametrize("field", ["operations", "values"])
@pytest.mark.parametrize("container", [None, 7, "invalid", {"invalid"}])
def test_direct_estimate_graph_rejects_malformed_graph_containers(
    field: str, container: object
) -> None:
    graph = replace(parse_graph(EXAMPLE_GRAPH), **{field: container})
    message = (
        "graph operations must be a tuple or list"
        if field == "operations"
        else "graph values must be a mapping"
    )
    original = deepcopy(vars(graph))

    for _ in range(2):
        with pytest.raises(GraphValidationError) as error:
            estimate_graph(graph)
        assert str(error.value) == message
    assert vars(graph) == original
    assert getattr(graph, field) is container


@pytest.mark.parametrize("field", ["operations", "values"])
def test_direct_estimate_graph_rejects_one_shot_graph_containers_without_consuming(
    field: str,
) -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    entries = tuple(getattr(graph, field))
    container = iter(entries)
    graph = replace(graph, **{field: container})
    message = (
        "graph operations must be a tuple or list"
        if field == "operations"
        else "graph values must be a mapping"
    )

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == message
    assert tuple(container) == entries


@pytest.mark.parametrize(
    "definition", [None, {}, SimpleNamespace(kind="matmul")]
)
def test_direct_estimate_graph_rejects_non_operation_definitions(definition: object) -> None:
    graph = replace(parse_graph(EXAMPLE_GRAPH), operations=(definition,))

    for _ in range(2):
        with pytest.raises(GraphValidationError) as error:
            estimate_graph(graph)
        assert str(error.value) == "ops[0] must be an Operation definition"
    assert graph.operations[0] is definition


@pytest.mark.parametrize("field", ["name", "kind", "inputs", "output"])
def test_direct_estimate_graph_rejects_deleted_operation_fields(field: str) -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    operation = graph.operations[0]
    object.__delattr__(operation, field)
    messages = {
        "name": "ops[0].name must be a non-empty string",
        "kind": (
            "ops[0] unsupported operation None; use one of "
            "add, layer_norm, matmul, mul, reduce_sum, relu, softmax"
        ),
        "inputs": "ops[0].inputs must contain exactly 2 value names",
        "output": "ops[0].output must be a non-empty string",
    }

    for _ in range(2):
        with pytest.raises(GraphValidationError) as error:
            estimate_graph(graph)
        assert str(error.value) == messages[field]
    assert field not in vars(operation)


@pytest.mark.parametrize("field", ["name", "output"])
@pytest.mark.parametrize("value", [None, [], {}, 3, "", "   "])
def test_direct_estimate_graph_rejects_malformed_operation_names(
    field: str, value: object
) -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    object.__setattr__(graph.operations[0], field, value)
    original = deepcopy(graph)

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == f"ops[0].{field} must be a non-empty string"
    assert graph == original


@pytest.mark.parametrize("inputs", [None, "lhs", {"lhs": None}, (), ("lhs",), ("lhs",) * 3])
def test_direct_estimate_graph_rejects_malformed_operation_inputs(inputs: object) -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    object.__setattr__(graph.operations[0], "inputs", inputs)

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == "ops[0].inputs must contain exactly 2 value names"
    assert graph.operations[0].inputs is inputs


@pytest.mark.parametrize("position", [0, 1])
@pytest.mark.parametrize("name", [None, [], {}, 3, "", "   "])
def test_direct_estimate_graph_rejects_malformed_input_references(
    position: int, name: object
) -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    inputs: list[object] = list(graph.operations[0].inputs)
    inputs[position] = name
    object.__setattr__(graph.operations[0], "inputs", inputs)

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == f"ops[0].inputs[{position}] must be a non-empty string"
    assert graph.operations[0].inputs is inputs
    assert inputs[position] is name


@pytest.mark.parametrize("kind", ["matmul", "add", "mul", "relu", "softmax", "layer_norm", "reduce_sum"])
def test_direct_estimate_graph_validates_input_arity_for_each_kind(kind: str) -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    operation = replace(graph.operations[0], kind=kind, inputs=())
    graph = replace(graph, operations=(operation,))
    expected = 2 if kind in {"matmul", "add", "mul"} else 1

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == f"ops[0].inputs must contain exactly {expected} value names"


@pytest.mark.parametrize("broken_field", ["operations", "values"])
def test_direct_estimate_graph_checks_target_before_malformed_containers(broken_field: str) -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    graph = replace(graph, target=Target(0, 0), **{broken_field: None})

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == "target.peak_compute_gflops must be a positive finite number"


@pytest.mark.parametrize("field", ["name", "inputs", "output", "values"])
def test_direct_estimate_graph_checks_kind_before_malformed_fields_or_values(field: str) -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    object.__setattr__(graph.operations[0], "kind", "mystery")
    object.__delattr__(graph if field == "values" else graph.operations[0], field)

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == (
        "ops[0] unsupported operation 'mystery'; use one of "
        "add, layer_norm, matmul, mul, reduce_sum, relu, softmax"
    )


@pytest.mark.parametrize("missing", [("product", "lhs", "rhs"), ("lhs", "rhs")])
def test_direct_estimate_graph_preserves_missing_reference_order(missing: tuple[str, ...]) -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    for name in reversed(missing):
        del graph.values[name]
    values = MappingProxyType(graph.values)
    graph = replace(graph, values=values)
    original = dict(values)

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == f"ops[0] references missing value definition {missing[0]!r}"
    assert dict(values) == original
    assert graph.values is values


@pytest.mark.parametrize(("missing", "invalid_position"), [("product", 0), ("lhs", 1)])
def test_direct_estimate_graph_checks_earlier_references_before_malformed_later_names(
    missing: str, invalid_position: int
) -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    del graph.values[missing]
    inputs: list[object] = list(graph.operations[0].inputs)
    inputs[invalid_position] = []
    object.__setattr__(graph.operations[0], "inputs", inputs)

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == f"ops[0] references missing value definition {missing!r}"


@pytest.mark.parametrize("operations_type", [tuple, list])
@pytest.mark.parametrize("inputs_type", [tuple, list])
@pytest.mark.parametrize("values_type", [dict, UserDict, MappingProxyType])
def test_direct_estimate_graph_preserves_supported_structural_containers(
    operations_type, inputs_type, values_type
) -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    graph = replace(
        graph,
        operations=operations_type(
            replace(operation, inputs=inputs_type(operation.inputs))
            for operation in graph.operations
        ),
        values=values_type(graph.values),
    )
    original_operations = deepcopy(graph.operations)
    original_values = dict(graph.values)
    operations = graph.operations
    values = graph.values

    assert estimate_graph(graph) == analyze_graph(EXAMPLE_GRAPH)
    assert graph.operations == original_operations
    assert dict(values) == original_values
    assert graph.operations is operations
    assert graph.values is values


@pytest.mark.parametrize("operations", [(), []])
def test_direct_estimate_graph_checks_value_container_for_empty_graph(operations) -> None:
    graph = replace(parse_graph(EXAMPLE_GRAPH), operations=operations, values=None)

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == "graph values must be a mapping"


def test_direct_estimate_graph_rejects_unsupported_operation_kind() -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    graph = replace(
        graph,
        operations=(replace(graph.operations[0], kind="mystery"), *graph.operations[1:]),
    )

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == (
        "ops[0] unsupported operation 'mystery'; use one of "
        "add, layer_norm, matmul, mul, reduce_sum, relu, softmax"
    )


def test_direct_estimate_graph_rejects_missing_operation_value_definition() -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    del graph.values[graph.operations[0].output]

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == "ops[0] references missing value definition 'product'"


def test_direct_estimate_graph_rejects_none_output_definition() -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    graph.values["product"] = None  # type: ignore[assignment]

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == "ops[0] value definition 'product' must be a Tensor"


@pytest.mark.parametrize(
    ("name", "definition", "index"),
    [
        ("lhs", None, 0),
        pytest.param(
            "rhs",
            SimpleNamespace(name="rhs", shape=(3, 4), dtype="f32", elements=12, nbytes=48),
            0,
            id="duck-typed-input",
        ),
        ("output", {}, 2),
    ],
)
def test_direct_estimate_graph_rejects_non_tensor_definition(
    name: str, definition: object, index: int
) -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    graph.values[name] = definition  # type: ignore[assignment]
    original = dict(graph.values)

    for _ in range(2):
        with pytest.raises(GraphValidationError) as error:
            estimate_graph(graph)
        assert str(error.value) == f"ops[{index}] value definition {name!r} must be a Tensor"
    assert graph.values == original


@pytest.mark.parametrize(
    ("invalid_names", "first"),
    [(("rhs", "lhs", "product"), "product"), (("rhs", "lhs"), "lhs")],
)
def test_direct_estimate_graph_checks_definitions_in_reference_order(
    invalid_names: tuple[str, ...], first: str
) -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    for name in invalid_names:
        graph.values[name] = None  # type: ignore[assignment]

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == f"ops[0] value definition {first!r} must be a Tensor"


@pytest.mark.parametrize("earlier_error", ["target", "kind", "missing-output", "missing-input"])
def test_direct_estimate_graph_preserves_precedence_over_definition_types(
    earlier_error: str,
) -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    graph.values["lhs"] = None  # type: ignore[assignment]
    if earlier_error in {"target", "kind"}:
        graph = replace(
            graph,
            operations=(replace(graph.operations[0], kind="mystery"), *graph.operations[1:]),
        )
        del graph.values["product"]
        message = (
            "ops[0] unsupported operation 'mystery'; use one of "
            "add, layer_norm, matmul, mul, reduce_sum, relu, softmax"
        )
        if earlier_error == "target":
            graph = replace(graph, target=Target(0, 0))
            message = "target.peak_compute_gflops must be a positive finite number"
    else:
        missing = "product" if earlier_error == "missing-output" else "rhs"
        del graph.values[missing]
        message = f"ops[0] references missing value definition {missing!r}"

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == message


def test_direct_estimate_graph_rejects_unsupported_tensor_dtype() -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    object.__setattr__(graph.values["lhs"], "dtype", "wat")
    original = deepcopy(graph)

    for _ in range(2):
        with pytest.raises(GraphValidationError) as error:
            estimate_graph(graph)
        assert str(error.value) == (
            "ops[0] value definition 'lhs' dtype must be a supported string"
        )
    assert graph == original


@pytest.mark.parametrize(("name", "index"), [("product", 0), ("lhs", 0), ("output", 2)])
@pytest.mark.parametrize(
    ("field", "value", "diagnostic"),
    [
        ("shape", None, "shape must be a tuple or list"),
        ("shape", 3, "shape must be a tuple or list"),
        ("shape", "23", "shape must be a tuple or list"),
        ("shape", {2: 3}, "shape must be a tuple or list"),
        ("shape", {2, 3}, "shape must be a tuple or list"),
        ("dtype", None, "dtype must be a supported string"),
        ("dtype", "wat", "dtype must be a supported string"),
        ("dtype", [], "dtype must be a supported string"),
        ("dtype", {}, "dtype must be a supported string"),
        ("dtype", 3, "dtype must be a supported string"),
        ("dtype", True, "dtype must be a supported string"),
    ],
)
def test_direct_estimate_graph_rejects_malformed_tensor_fields(
    name: str, index: int, field: str, value: object, diagnostic: str
) -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    tensor = graph.values[name]
    object.__setattr__(tensor, field, value)
    original = deepcopy(graph)
    values = MappingProxyType(graph.values)
    graph = replace(graph, values=values)

    for _ in range(2):
        with pytest.raises(GraphValidationError) as error:
            estimate_graph(graph)
        assert str(error.value) == f"ops[{index}] value definition {name!r} {diagnostic}"
    assert dict(graph.values) == original.values
    assert graph.inputs == original.inputs
    assert graph.operations == original.operations
    assert graph.values is values
    assert graph.values[name] is tensor
    assert getattr(tensor, field) is value


@pytest.mark.parametrize("name", ["product", "lhs"])
@pytest.mark.parametrize("field", ["shape", "dtype"])
def test_direct_estimate_graph_rejects_deleted_tensor_fields(name: str, field: str) -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    tensor = graph.values[name]
    object.__delattr__(tensor, field)
    original = deepcopy(vars(tensor))
    diagnostic = (
        "shape must be a tuple or list"
        if field == "shape"
        else "dtype must be a supported string"
    )

    for _ in range(2):
        with pytest.raises(GraphValidationError) as error:
            estimate_graph(graph)
        assert str(error.value) == f"ops[0] value definition {name!r} {diagnostic}"
    assert vars(tensor) == original
    assert field not in vars(tensor)


@pytest.mark.parametrize("shape_type", [tuple, list])
@pytest.mark.parametrize("dimension", ["x", 1.5, True, False, 0, -1, None, []])
def test_direct_estimate_graph_rejects_malformed_tensor_dimensions(
    shape_type, dimension: object
) -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    shape = shape_type((2, dimension))
    object.__setattr__(graph.values["lhs"], "shape", shape)
    original = deepcopy(graph)

    for _ in range(2):
        with pytest.raises(GraphValidationError) as error:
            estimate_graph(graph)
        assert str(error.value) == (
            "ops[0] value definition 'lhs' shape dimension 1 must be a positive integer"
        )
    assert graph == original
    assert graph.values["lhs"].shape is shape


def test_direct_estimate_graph_rejects_tensor_shape_iterator_without_consuming() -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    shape = iter((2, 3))
    object.__setattr__(graph.values["lhs"], "shape", shape)

    for _ in range(2):
        with pytest.raises(GraphValidationError) as error:
            estimate_graph(graph)
        assert str(error.value) == (
            "ops[0] value definition 'lhs' shape must be a tuple or list"
        )
    assert graph.values["lhs"].shape is shape
    assert tuple(shape) == (2, 3)


@pytest.mark.parametrize(
    ("earlier_error", "message"),
    [
        ("target", "target.peak_compute_gflops must be a positive finite number"),
        ("name", "ops[0].name must be a non-empty string"),
        ("inputs", "ops[0].inputs must contain exactly 2 value names"),
        ("input-name", "ops[0].inputs[1] must be a non-empty string"),
        ("missing-input", "ops[0] references missing value definition 'rhs'"),
        ("definition-type", "ops[0] value definition 'rhs' must be a Tensor"),
    ],
)
def test_direct_estimate_graph_preserves_precedence_over_tensor_fields(
    earlier_error: str, message: str
) -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    object.__setattr__(graph.values["product"], "shape", ("x",))
    object.__setattr__(graph.values["lhs"], "dtype", "wat")
    if earlier_error == "target":
        graph = replace(graph, target=Target(0, 0))
    elif earlier_error == "name":
        object.__delattr__(graph.operations[0], "name")
    elif earlier_error == "inputs":
        object.__setattr__(graph.operations[0], "inputs", ())
    elif earlier_error == "input-name":
        object.__setattr__(graph.operations[0], "inputs", ("lhs", []))
    elif earlier_error == "missing-input":
        del graph.values["rhs"]
    else:
        graph.values["rhs"] = None  # type: ignore[assignment]

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == message


@pytest.mark.parametrize(
    ("names", "first"),
    [(("rhs", "lhs", "product"), "product"), (("rhs", "lhs"), "lhs")],
)
def test_direct_estimate_graph_checks_tensor_fields_in_reference_order(
    names: tuple[str, ...], first: str
) -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    for name in names:
        object.__setattr__(graph.values[name], "dtype", "wat")

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == (
        f"ops[0] value definition {first!r} dtype must be a supported string"
    )


def test_direct_estimate_graph_checks_tensor_shape_before_dtype() -> None:
    graph = parse_graph(EXAMPLE_GRAPH)
    object.__setattr__(graph.values["lhs"], "shape", ("x",))
    object.__setattr__(graph.values["lhs"], "dtype", [])

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == (
        "ops[0] value definition 'lhs' shape dimension 0 must be a positive integer"
    )


@pytest.mark.parametrize("shape_type", [tuple, list])
@pytest.mark.parametrize("dtype", sorted(DTYPE_BYTES))
def test_direct_estimate_graph_preserves_scalar_tensor_shapes(shape_type, dtype: str) -> None:
    payload = {
        "name": "scalar_relu",
        "inputs": {"source": {"shape": [], "dtype": dtype}},
        "ops": [
            {"name": "relu", "op": "relu", "inputs": ["source"], "outputs": ["result"]}
        ],
        "outputs": ["result"],
    }
    graph = parse_graph(payload)
    for tensor in graph.values.values():
        object.__setattr__(tensor, "shape", shape_type())
    original = deepcopy(graph)

    assert estimate_graph(graph) == analyze_graph(payload)
    assert graph == original


@pytest.mark.parametrize("repeated_operands", [False, True])
@pytest.mark.parametrize("shape_type", [tuple, list])
def test_direct_estimate_graph_preserves_tensor_subclasses_and_read_only_values(
    repeated_operands: bool, shape_type
) -> None:
    class SpecializedTensor(Tensor):
        pass

    payload = deepcopy(SOFTMAX_GRAPH if repeated_operands else EXAMPLE_GRAPH)
    if repeated_operands:
        payload["ops"][0].update(op="mul", inputs=["source", "source"])
    graph = parse_graph(payload)
    values = {
        name: SpecializedTensor(tensor.name, shape_type(tensor.shape), tensor.dtype)
        for name, tensor in graph.values.items()
    }
    values["unused"] = None  # type: ignore[assignment]
    values["unused_tensor"] = SpecializedTensor(
        "unused_tensor", ("x",), "wat"  # type: ignore[arg-type]
    )
    original = dict(values)
    read_only_values = MappingProxyType(values)
    graph = replace(graph, values=read_only_values)  # type: ignore[arg-type]
    original_inputs = deepcopy(graph.inputs)

    assert estimate_graph(graph) == analyze_graph(payload)
    assert graph.values is read_only_values
    assert values == original
    assert all(values[name] is tensor for name, tensor in original.items())
    assert graph.inputs == original_inputs


def test_direct_estimate_graph_rejects_mapping_target() -> None:
    graph = replace(
        parse_graph(EXAMPLE_GRAPH),
        target={"peak_compute_gflops": 100.0, "memory_bandwidth_gbps": 100.0},
    )

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == "graph target must be a Target definition or None"


@pytest.mark.parametrize("field", ["peak_compute_gflops", "memory_bandwidth_gbps"])
def test_direct_estimate_graph_rejects_incomplete_target(field: str) -> None:
    target = Target(100.0, 100.0)
    object.__delattr__(target, field)
    graph = replace(parse_graph(EXAMPLE_GRAPH), target=target)

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == f"target.{field} must be a positive finite number"


@pytest.mark.parametrize(
    ("target", "field"),
    [
        (Target(100.0, 0), "memory_bandwidth_gbps"),
        (Target(True, 100.0), "peak_compute_gflops"),
        (Target("100", 100.0), "peak_compute_gflops"),  # type: ignore[arg-type]
        (Target(float("nan"), 100.0), "peak_compute_gflops"),
        (Target(100.0, float("inf")), "memory_bandwidth_gbps"),
        pytest.param(Target(10**1000, 100.0), "peak_compute_gflops", id="oversized-integer"),
    ],
)
def test_direct_estimate_graph_rejects_invalid_target_rates(
    target: Target, field: str
) -> None:
    graph = replace(parse_graph(EXAMPLE_GRAPH), target=target)

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == f"target.{field} must be a positive finite number"


@pytest.mark.parametrize("field", ["peak_compute_gflops", "memory_bandwidth_gbps"])
def test_direct_estimate_graph_rejects_scaled_integer_target_overflow(field: str) -> None:
    target = replace(Target(100, 100), **{field: 10**308})
    graph = replace(parse_graph(EXAMPLE_GRAPH), target=target)

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == "roofline arithmetic exceeds finite floating-point range"


def test_direct_estimate_graph_checks_target_before_operations() -> None:
    graph = replace(parse_graph(EXAMPLE_GRAPH), target=Target(0, 0), operations=(None,))

    with pytest.raises(GraphValidationError) as error:
        estimate_graph(graph)

    assert str(error.value) == "target.peak_compute_gflops must be a positive finite number"


@pytest.mark.parametrize("target", [None, Target(100, 100), Target(100.0, 100.0)])
def test_direct_estimate_graph_preserves_valid_target_report(target: Target | None) -> None:
    graph = replace(parse_graph(EXAMPLE_GRAPH), target=target)
    original = deepcopy(graph)
    payload = deepcopy(EXAMPLE_GRAPH)
    if target is not None:
        payload["target"] = {
            "peak_compute_gflops": target.peak_compute_gflops,
            "memory_bandwidth_gbps": target.memory_bandwidth_gbps,
        }

    assert estimate_graph(graph) == analyze_graph(payload)
    assert graph == original
    assert graph.target is target


def test_graph_without_target_has_explicit_null_roofline_analysis() -> None:
    report = analyze_graph(EXAMPLE_GRAPH)

    assert report["target"] is None
    assert all(operation["roofline"] is None for operation in report["ops"])
    assert report["totals"]["roofline"] is None
