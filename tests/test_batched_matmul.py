from __future__ import annotations

import pytest

from cost_model import (
    GraphValidationError,
    LoweringValidationError,
    MatmulMlirEmissionError,
    analyze_graph,
    compare_matmul_lowering,
    emit_matmul_mlir,
    emit_matmul_mlir_graph,
    parse_graph,
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


@pytest.mark.parametrize(
    ("lhs_shape", "rhs_shape", "bytes_read", "estimated_time_us"),
    [
        pytest.param([3, 5], [2, 4, 5, 7], 1180, 1.852, id="rank-two-lhs"),
        pytest.param([2, 4, 3, 5], [5, 7], 620, 1.292, id="rank-two-rhs"),
        pytest.param([4, 3, 5], [2, 1, 5, 7], 520, 1.192, id="shorter-lhs-batch"),
        pytest.param([2, 1, 3, 5], [4, 5, 7], 680, 1.352, id="shorter-rhs-batch"),
    ],
)
def test_matmul_broadcasts_implicit_leading_batch_axes_without_expanding_input_traffic(
    lhs_shape: list[int],
    rhs_shape: list[int],
    bytes_read: int,
    estimated_time_us: float,
) -> None:
    payload = _batched_matmul_payload()
    payload["inputs"]["lhs"]["shape"] = lhs_shape
    payload["inputs"]["rhs"]["shape"] = rhs_shape

    report = analyze_graph(payload)

    operation = report["ops"][0]
    assert operation["shape"] == [2, 4, 3, 7]
    assert operation["dtype"] == "f32"
    expected_cost = {
        "flops": 1680,
        "bytes_read": bytes_read,
        "bytes_written": 672,
        "arithmetic_intensity": pytest.approx(1680 / (bytes_read + 672)),
        "roofline": {
            "arithmetic_intensity": pytest.approx(1680 / (bytes_read + 672)),
            "ridge_point": 2.0,
            "bound": "memory",
            "estimated_time_us": pytest.approx(estimated_time_us),
        },
    }
    assert report["totals"] == expected_cost
    assert {key: operation[key] for key in expected_cost} == expected_cost


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


@pytest.mark.parametrize("as_graph", [False, True], ids=["payload", "graph"])
@pytest.mark.parametrize(
    ("lhs_shape", "rhs_shape"),
    [
        pytest.param([2, 1, 3, 5], [1, 4, 5, 7], id="explicit-batch-broadcast"),
        pytest.param([3, 5], [2, 4, 5, 7], id="rank-two-lhs"),
        pytest.param([2, 4, 3, 5], [5, 7], id="rank-two-rhs"),
        pytest.param([4, 3, 5], [2, 1, 5, 7], id="shorter-lhs-batch"),
        pytest.param([2, 1, 3, 5], [4, 5, 7], id="shorter-rhs-batch"),
    ],
)
def test_public_mlir_emitters_reject_parseable_batched_matmul(
    lhs_shape: list[int], rhs_shape: list[int], as_graph: bool
) -> None:
    payload = _batched_matmul_payload()
    payload["inputs"]["lhs"]["shape"] = lhs_shape
    payload["inputs"]["rhs"]["shape"] = rhs_shape
    graph = parse_graph(payload)
    assert graph.values["result"].shape == (2, 4, 3, 7)

    with pytest.raises(
        MatmulMlirEmissionError,
        match="^matmul MLIR emission requires rank-2 input tensors$",
    ):
        if as_graph:
            emit_matmul_mlir_graph(graph)
        else:
            emit_matmul_mlir(payload)
