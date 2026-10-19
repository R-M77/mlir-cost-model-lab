from __future__ import annotations

import pytest

from cost_model import (
    GraphValidationError,
    LoweringValidationError,
    analyze_graph,
    compare_matmul_lowering,
)


def _batched_matmul_payload() -> dict:
    return {
        "name": "broadcasted_batched_matmul",
        "target": {"peak_compute_gflops": 2.0, "memory_bandwidth_gbps": 1.0},
        "inputs": {
            "lhs": {"shape": [2, 1, 3, 5], "dtype": "f32"},
            "rhs": {"shape": [1, 4, 5, 7], "dtype": "f32"},
        },
        "ops": [
            {
                "name": "batched_matmul",
                "op": "matmul",
                "inputs": ["lhs", "rhs"],
                "outputs": ["result"],
            }
        ],
        "outputs": ["result"],
    }


def test_batched_matmul_broadcasts_batch_axes_without_expanding_input_traffic() -> None:
    report = analyze_graph(_batched_matmul_payload())

    operation = report["ops"][0]
    assert operation["shape"] == [2, 4, 3, 7]
    assert operation["dtype"] == "f32"
    expected_cost = {
        "flops": 1680,
        "bytes_read": 680,
        "bytes_written": 672,
        "arithmetic_intensity": pytest.approx(1680 / 1352),
        "roofline": {
            "arithmetic_intensity": pytest.approx(1680 / 1352),
            "ridge_point": 2.0,
            "bound": "memory",
            "estimated_time_us": pytest.approx(1.352),
        },
    }
    assert report["totals"] == expected_cost
    assert {key: operation[key] for key in expected_cost} == expected_cost
    assert report["fusion"] == {"count": 0, "candidates": []}


def test_batched_matmul_rejects_incompatible_batch_axes_as_a_graph_error() -> None:
    payload = _batched_matmul_payload()
    payload["inputs"]["lhs"]["shape"] = [2, 3, 5]
    payload["inputs"]["rhs"]["shape"] = [4, 5, 7]

    with pytest.raises(
        GraphValidationError, match="matmul batch dimensions cannot be broadcast together"
    ):
        analyze_graph(payload)


@pytest.mark.parametrize("mode", ["scalar", "tiled"])
def test_batched_matmul_stays_outside_rank_two_lowering_scope(mode: str) -> None:
    with pytest.raises(
        LoweringValidationError,
        match="matmul lowering exercise requires rank-2 input tensors",
    ):
        compare_matmul_lowering(_batched_matmul_payload(), mode=mode)
