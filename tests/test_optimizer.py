from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from cost_model import analyze_graph, optimize_graph, parse_graph


EXAMPLE_GRAPH = {
    "name": "matmul_bias_relu",
    "target": {
        "peak_compute_gflops": 1000.0,
        "memory_bandwidth_gbps": 100.0,
    },
    "inputs": {
        "lhs": {"shape": [2, 3], "dtype": "f32"},
        "rhs": {"shape": [3, 4], "dtype": "f32"},
        "bias": {"shape": [4], "dtype": "f32"},
    },
    "ops": [
        {"name": "matmul", "op": "matmul", "inputs": ["lhs", "rhs"], "outputs": ["product"]},
        {"name": "add_bias", "op": "add", "inputs": ["product", "bias"], "outputs": ["biased"]},
        {"name": "relu", "op": "relu", "inputs": ["biased"], "outputs": ["output"]},
    ],
    "outputs": ["output"],
}


def _graph(ops: list[dict[str, object]], outputs: list[str], target: object = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "optimizer_test",
        "inputs": {
            "x": {"shape": [2], "dtype": "f32"},
            "y": {"shape": [2], "dtype": "f32"},
            "z": {"shape": [2], "dtype": "f32"},
        },
        "ops": ops,
        "outputs": outputs,
    }
    if target is not None:
        payload["target"] = target
    return payload


def test_example_optimizer_fuses_live_pair_and_recomputes_roofline() -> None:
    report = optimize_graph(parse_graph(EXAMPLE_GRAPH))

    assert report["baseline"]["totals"]["flops"] == 64
    assert report["baseline"]["totals"]["bytes_read"] + report["baseline"]["totals"]["bytes_written"] == 248
    optimized = report["optimized"]
    assert optimized["applied_fusions"] == [
        {
            "producer": "add_bias",
            "consumer": "relu",
            "intermediate": "biased",
            "bytes_avoided": 64,
            "reason": "adjacent elementwise operations with a single live use",
        }
    ]
    assert optimized["eliminated_dead_operations"] == []
    assert optimized["bytes_avoided"] == 64
    assert optimized["totals"]["flops"] == 64
    assert optimized["totals"]["bytes_read"] + optimized["totals"]["bytes_written"] == 184
    assert optimized["totals"]["arithmetic_intensity"] == pytest.approx(64 / 184)
    assert optimized["totals"]["roofline"] == {
        "arithmetic_intensity": pytest.approx(64 / 184),
        "ridge_point": pytest.approx(10.0),
        "bound": "memory",
        "estimated_time_us": pytest.approx(184 / (100 * 1000)),
    }
    assert report["cost_delta"] == {
        "flops": 0,
        "bytes_read": -32,
        "bytes_written": -32,
        "logical_bytes": -64,
        "roofline": {"estimated_time_us": pytest.approx(-0.00064)},
    }
    assert optimized["graph_outputs"] == ["output"]
    assert [item["name"] for item in optimized["ops"]] == ["matmul", "add_bias", "relu"]


def test_graph_output_intermediate_is_not_fused() -> None:
    payload = _graph(
        [
            {"name": "add", "op": "add", "inputs": ["x", "y"], "outputs": ["intermediate"]},
            {"name": "relu", "op": "relu", "inputs": ["intermediate"], "outputs": ["output"]},
        ],
        ["intermediate", "output"],
    )
    report = optimize_graph(parse_graph(payload))

    assert report["optimized"]["applied_fusions"] == []
    assert report["optimized"]["bytes_avoided"] == 0
    assert report["optimized"]["totals"] == report["baseline"]["totals"]


def test_multiple_consumers_are_not_fused() -> None:
    payload = _graph(
        [
            {"name": "producer", "op": "add", "inputs": ["x", "y"], "outputs": ["intermediate"]},
            {"name": "left", "op": "relu", "inputs": ["intermediate"], "outputs": ["left_out"]},
            {"name": "right", "op": "mul", "inputs": ["intermediate", "z"], "outputs": ["right_out"]},
        ],
        ["left_out", "right_out"],
    )

    report = optimize_graph(parse_graph(payload))
    assert report["optimized"]["applied_fusions"] == []


def test_dead_consumer_does_not_block_live_fusion() -> None:
    payload = _graph(
        [
            {"name": "add", "op": "add", "inputs": ["x", "y"], "outputs": ["mid"]},
            {"name": "relu", "op": "relu", "inputs": ["mid"], "outputs": ["output"]},
            {"name": "dead_use", "op": "mul", "inputs": ["mid", "z"], "outputs": ["unused"]},
        ],
        ["output"],
    )

    report = optimize_graph(parse_graph(payload))
    optimized = report["optimized"]

    assert report["baseline"]["totals"]["flops"] == 6
    assert report["baseline"]["totals"]["bytes_read"] == 40
    assert report["baseline"]["totals"]["bytes_written"] == 24
    assert optimized["eliminated_dead_operations"] == ["dead_use"]
    assert optimized["applied_fusions"] == [
        {
            "producer": "add",
            "consumer": "relu",
            "intermediate": "mid",
            "bytes_avoided": 16,
            "reason": "adjacent elementwise operations with a single live use",
        }
    ]
    assert optimized["bytes_avoided"] == 16
    assert optimized["totals"]["flops"] == 4
    assert optimized["totals"]["bytes_read"] == 16
    assert optimized["totals"]["bytes_written"] == 8


def test_repeated_live_input_references_still_block_fusion() -> None:
    payload = _graph(
        [
            {"name": "producer", "op": "add", "inputs": ["x", "y"], "outputs": ["intermediate"]},
            {"name": "consumer", "op": "mul", "inputs": ["intermediate", "intermediate"], "outputs": ["output"]},
        ],
        ["output"],
    )

    report = optimize_graph(parse_graph(payload))
    assert report["optimized"]["applied_fusions"] == []


def test_non_adjacent_elementwise_operations_are_not_fused() -> None:
    payload = _graph(
        [
            {"name": "producer", "op": "add", "inputs": ["x", "y"], "outputs": ["intermediate"]},
            {"name": "unrelated", "op": "mul", "inputs": ["y", "z"], "outputs": ["other"]},
            {"name": "consumer", "op": "relu", "inputs": ["intermediate"], "outputs": ["output"]},
        ],
        ["output", "other"],
    )

    report = optimize_graph(parse_graph(payload))
    assert report["optimized"]["applied_fusions"] == []


def test_layer_norm_is_not_an_elementwise_fusion_candidate() -> None:
    payload = _graph(
        [
            {"name": "producer", "op": "add", "inputs": ["x", "y"], "outputs": ["before_norm"]},
            {"name": "normalize", "op": "layer_norm", "inputs": ["before_norm"], "outputs": ["after_norm"]},
            {"name": "consumer", "op": "relu", "inputs": ["after_norm"], "outputs": ["output"]},
        ],
        ["output"],
    )

    report = analyze_graph(payload)
    optimized = optimize_graph(parse_graph(payload))["optimized"]

    assert report["fusion"] == {"count": 0, "candidates": []}
    assert optimized["applied_fusions"] == []
    assert optimized["bytes_avoided"] == 0


def test_reduce_sum_is_not_an_elementwise_fusion_candidate() -> None:
    payload = _graph(
        [
            {"name": "producer", "op": "add", "inputs": ["x", "y"], "outputs": ["before_sum"]},
            {"name": "sum", "op": "reduce_sum", "inputs": ["before_sum"], "outputs": ["after_sum"]},
            {"name": "consumer", "op": "relu", "inputs": ["after_sum"], "outputs": ["output"]},
        ],
        ["output"],
    )

    report = analyze_graph(payload)
    optimized = optimize_graph(parse_graph(payload))["optimized"]

    assert report["fusion"] == {"count": 0, "candidates": []}
    assert optimized["applied_fusions"] == []
    assert optimized["bytes_avoided"] == 0


def test_matmul_is_a_fusion_boundary() -> None:
    payload = {
        "name": "matmul_boundary",
        "inputs": {
            "lhs": {"shape": [2, 2], "dtype": "f32"},
            "rhs": {"shape": [2, 2], "dtype": "f32"},
            "bias": {"shape": [2, 2], "dtype": "f32"},
        },
        "ops": [
            {"name": "matmul", "op": "matmul", "inputs": ["lhs", "rhs"], "outputs": ["product"]},
            {"name": "relu", "op": "relu", "inputs": ["product"], "outputs": ["output"]},
        ],
        "outputs": ["output"],
    }

    report = optimize_graph(parse_graph(payload))
    assert report["optimized"]["applied_fusions"] == []


def test_dead_operations_do_not_contribute_to_optimized_totals() -> None:
    payload = _graph(
        [
            {"name": "dead", "op": "add", "inputs": ["x", "y"], "outputs": ["unused"]},
            {"name": "live", "op": "relu", "inputs": ["x"], "outputs": ["output"]},
        ],
        ["output"],
    )
    report = optimize_graph(parse_graph(payload))

    assert report["baseline"]["totals"]["flops"] == 4
    assert report["optimized"]["eliminated_dead_operations"] == ["dead"]
    assert report["optimized"]["totals"]["flops"] == 2
    assert report["optimized"]["totals"]["bytes_read"] + report["optimized"]["totals"]["bytes_written"] == 16
    assert [item["name"] for item in report["optimized"]["ops"]] == ["live"]
    assert report["optimized"]["bytes_avoided"] == 0
    assert report["cost_delta"] == {
        "flops": -2,
        "bytes_read": -16,
        "bytes_written": -8,
        "logical_bytes": -24,
        "roofline": {"estimated_time_us": None},
    }


def test_optimized_no_target_keeps_roofline_null() -> None:
    report = optimize_graph(parse_graph(deepcopy(EXAMPLE_GRAPH) | {"target": None}))

    assert report["baseline"]["totals"]["roofline"] is None
    assert report["optimized"]["totals"]["roofline"] is None
    assert all(item["roofline"] is None for item in report["optimized"]["ops"])
    assert report["cost_delta"]["roofline"]["estimated_time_us"] is None


def test_chained_fusions_do_not_double_count_intermediates() -> None:
    payload = _graph(
        [
            {"name": "first", "op": "add", "inputs": ["x", "y"], "outputs": ["first_out"]},
            {"name": "second", "op": "mul", "inputs": ["first_out", "y"], "outputs": ["second_out"]},
            {"name": "third", "op": "relu", "inputs": ["second_out"], "outputs": ["output"]},
        ],
        ["output"],
    )
    report = optimize_graph(parse_graph(payload))

    assert [item["producer"] for item in report["optimized"]["applied_fusions"]] == ["first", "second"]
    assert report["optimized"]["bytes_avoided"] == 32
    assert report["optimized"]["totals"]["flops"] == 6
    assert report["optimized"]["totals"]["bytes_read"] + report["optimized"]["totals"]["bytes_written"] == 32


def test_optimize_cli_is_deterministic_and_default_shape_is_unchanged(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(EXAMPLE_GRAPH), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": "src"}

    first = subprocess.run(
        [sys.executable, "-m", "cost_model", "--optimize", "--compact", str(graph_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    second = subprocess.run(
        [sys.executable, "-m", "cost_model", "--optimize", "--compact", str(graph_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    optimized_report = json.loads(first.stdout)
    assert optimized_report["optimized"]["bytes_avoided"] == 64
    assert optimized_report["cost_delta"]["flops"] == 0
    assert optimized_report["cost_delta"]["bytes_read"] == -32
    assert optimized_report["cost_delta"]["bytes_written"] == -32
    assert optimized_report["cost_delta"]["logical_bytes"] == -64
    assert optimized_report["cost_delta"]["roofline"]["estimated_time_us"] == pytest.approx(-0.00064)

    default = subprocess.run(
        [sys.executable, "-m", "cost_model", str(graph_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert default.returncode == 0
    default_report = json.loads(default.stdout)
    assert set(default_report) == {"graph", "target", "ops", "totals", "fusion"}
    assert "baseline" not in default_report


def test_analyze_graph_optimize_keyword_is_an_opt_in_api() -> None:
    report = analyze_graph(EXAMPLE_GRAPH, optimize=True)

    assert report["optimized"]["bytes_avoided"] == 64
    assert report["cost_delta"]["logical_bytes"] == -64
