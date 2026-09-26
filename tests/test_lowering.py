from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from cost_model import (
    LoweringValidationError,
    compare_matmul_lowering,
    compare_matmul_lowering_graph,
    parse_graph,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "examples" / "matmul_lowering.json"


def _fixture_payload() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    return subprocess.run(
        [sys.executable, "-m", "cost_model", *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
        env=env,
    )


def test_matmul_lowering_reports_deterministic_before_after_costs() -> None:
    report = compare_matmul_lowering(_fixture_payload())

    assert report["before"]["totals"] == {
        "flops": 48,
        "bytes_read": 72,
        "bytes_written": 32,
        "arithmetic_intensity": pytest.approx(48 / 104),
        "roofline": {
            "arithmetic_intensity": pytest.approx(48 / 104),
            "ridge_point": pytest.approx(10.0),
            "bound": "memory",
            "estimated_time_us": pytest.approx(0.00104),
        },
    }
    assert report["after"]["totals"] == {
        "flops": 48,
        "bytes_read": 192,
        "bytes_written": 32,
        "arithmetic_intensity": pytest.approx(48 / 224),
        "roofline": {
            "arithmetic_intensity": pytest.approx(48 / 224),
            "ridge_point": pytest.approx(10.0),
            "bound": "memory",
            "estimated_time_us": pytest.approx(0.00224),
        },
    }
    assert report["after"]["operation_count"] == 88
    assert report["cost_delta"] == {
        "flops": 0,
        "bytes_read": 120,
        "bytes_written": 0,
        "logical_bytes": 120,
        "operation_count": 87,
        "roofline": {"estimated_time_us": pytest.approx(0.0012)},
    }
    assert report["after"]["representation"] == {
        "kind": "scalar_reduction",
        "source_operation": "matmul",
        "loop_nest": [
            {"index": "m", "extent": 2},
            {"index": "n", "extent": 4},
            {"index": "k", "extent": 3},
        ],
        "body": [
            {"name": "init_accumulator", "op": "zero", "count": 8},
            {"name": "load_lhs", "op": "load", "count": 24},
            {"name": "load_rhs", "op": "load", "count": 24},
            {"name": "fma", "op": "fma", "count": 24},
            {"name": "store_output", "op": "store", "count": 8},
        ],
        "assumptions": {
            "accumulator_stays_in_register": True,
            "scalar_loads_are_counted_per_reduction_term": True,
            "cache_reuse_modeled": False,
            "real_mlir_required": False,
        },
    }


def test_matmul_lowering_without_target_keeps_roofline_delta_null() -> None:
    payload = _fixture_payload()
    payload.pop("target")

    report = compare_matmul_lowering(payload)

    assert report["before"]["totals"]["roofline"] is None
    assert report["after"]["totals"]["roofline"] is None
    assert report["cost_delta"]["roofline"] == {"estimated_time_us": None}


def test_zero_traffic_lowered_fma_is_compute_bound_with_null_intensity() -> None:
    report = compare_matmul_lowering(_fixture_payload())
    fma = next(
        operation for operation in report["after"]["ops"] if operation["op"] == "fma"
    )

    assert fma["flops"] == 48
    assert fma["bytes_read"] == fma["bytes_written"] == 0
    assert fma["arithmetic_intensity"] == 0.0
    assert fma["roofline"] == {
        "arithmetic_intensity": None,
        "ridge_point": pytest.approx(10.0),
        "bound": "compute",
        "estimated_time_us": pytest.approx(48 / (1000 * 1000)),
    }


def test_tiled_lowering_preserves_flops_and_models_tile_reuse() -> None:
    report = compare_matmul_lowering(
        _fixture_payload(), mode="tiled", tile_size=2, vector_width=2
    )

    assert report["before"]["totals"]["flops"] == 48
    assert report["after"]["totals"] == {
        "flops": 48,
        "bytes_read": 72,
        "bytes_written": 32,
        "arithmetic_intensity": pytest.approx(48 / 104),
        "roofline": {
            "arithmetic_intensity": pytest.approx(48 / 104),
            "ridge_point": pytest.approx(10.0),
            "bound": "memory",
            "estimated_time_us": pytest.approx(0.00104),
        },
    }
    assert report["after"]["operation_count"] == 32
    assert report["cost_delta"] == {
        "flops": 0,
        "bytes_read": 0,
        "bytes_written": 0,
        "logical_bytes": 0,
        "operation_count": 31,
        "roofline": {"estimated_time_us": pytest.approx(0.0)},
    }
    assert report["after"]["representation"]["kind"] == "tiled_vectorized"
    assert report["after"]["representation"]["tile_size"] == 2
    assert report["after"]["representation"]["vector_width"] == 2


def _compare_with_parameter(as_graph: bool, parameter: str, value: Any) -> dict[str, Any]:
    kwargs = {parameter: value}
    if as_graph:
        return compare_matmul_lowering_graph(
            parse_graph(_fixture_payload()), mode="tiled", **kwargs
        )
    return compare_matmul_lowering(_fixture_payload(), mode="tiled", **kwargs)


@pytest.mark.parametrize("as_graph", [False, True], ids=["payload", "graph"])
@pytest.mark.parametrize("parameter", ["tile_size", "vector_width"])
@pytest.mark.parametrize("value", [True, False, 2.0, -1.5, "2", None, 0, -2])
def test_matmul_lowering_rejects_invalid_parameters(
    as_graph: bool, parameter: str, value: Any
) -> None:
    with pytest.raises(LoweringValidationError, match=rf"^{parameter} must be a positive integer$"):
        _compare_with_parameter(as_graph, parameter, value)


@pytest.mark.parametrize("as_graph", [False, True], ids=["payload", "graph"])
@pytest.mark.parametrize("parameter", ["tile_size", "vector_width"])
def test_matmul_lowering_accepts_positive_integer_parameters(
    as_graph: bool, parameter: str
) -> None:
    report = _compare_with_parameter(as_graph, parameter, 1)
    assert report["after"]["representation"][parameter] == 1
    assert report["after"]["totals"]["flops"] == report["before"]["totals"]["flops"]


def _shaped_payload(m: int, k: int, n: int) -> dict[str, Any]:
    payload = _fixture_payload()
    payload["inputs"]["lhs"]["shape"] = [m, k]
    payload["inputs"]["rhs"]["shape"] = [k, n]
    return payload


def _tiled_counts(report: dict[str, Any]) -> dict[str, int]:
    return {op["name"]: op["count"] for op in report["after"]["ops"]}


def test_tiled_n_axis_vectors_and_scalar_tails_have_exact_counts_and_traffic() -> None:
    report = compare_matmul_lowering(
        _shaped_payload(3, 5, 7), mode="tiled", tile_size=4, vector_width=4
    )
    after = report["after"]
    assert set(report) == {"graph", "target", "before", "after", "cost_delta"}
    assert _tiled_counts(report) == {
        "init_accumulator_vectors": 3,
        "init_accumulator_scalar_tail": 9,
        "load_lhs_scalars": 15,
        "load_rhs_vectors": 5,
        "load_rhs_scalar_tail": 15,
        "vector_fma": 15,
        "scalar_tail_fma": 45,
        "store_output_vectors": 3,
        "store_output_scalar_tail": 9,
    }
    assert after["representation"]["body"] == [
        {"name": op["name"], "op": op["op"], "count": op["count"]}
        for op in after["ops"]
    ]
    assert after["representation"]["assumptions"]["vector_axis"] == "n"
    assert after["representation"]["assumptions"]["lhs_scalar_broadcast_across_vector_lanes"]
    assert after["representation"]["assumptions"]["incomplete_n_vectors_use_scalar_tail_operations"]
    ops = {op["name"]: op for op in after["ops"]}
    assert ops["vector_fma"]["inputs"] == ["lhs_scalar", "rhs_vector", "acc[m,n]"]
    assert ops["scalar_tail_fma"]["inputs"] == ["lhs_scalar", "rhs_scalar", "acc[m,n]"]
    assert {name: ops[name]["bytes_read"] for name in (
        "load_lhs_scalars", "load_rhs_vectors", "load_rhs_scalar_tail"
    )} == {"load_lhs_scalars": 60, "load_rhs_vectors": 80, "load_rhs_scalar_tail": 60}
    assert {name: ops[name]["flops"] for name in (
        "vector_fma", "scalar_tail_fma"
    )} == {"vector_fma": 120, "scalar_tail_fma": 90}
    assert {name: ops[name]["bytes_written"] for name in (
        "store_output_vectors", "store_output_scalar_tail"
    )} == {"store_output_vectors": 48, "store_output_scalar_tail": 36}
    assert after["totals"]["flops"] == 210
    assert after["totals"]["bytes_read"] == 200
    assert after["totals"]["bytes_written"] == 84
    assert after["operation_count"] == 119
    assert report["cost_delta"]["flops"] == 0
    assert report["cost_delta"]["bytes_read"] == 0
    assert report["cost_delta"]["bytes_written"] == 0
    assert report["cost_delta"]["operation_count"] == 118


@pytest.mark.parametrize(
    ("shape", "tile_size", "vector_width", "expected"),
    [
        ((3, 5, 7), 4, 1, (21, 0, 15, 35, 0, 105, 0, 21, 0)),
        ((3, 5, 7), 4, 8, (0, 21, 15, 0, 35, 0, 105, 0, 21)),
        ((5, 3, 9), 4, 3, (10, 15, 15, 6, 9, 30, 45, 10, 15)),
    ],
)
def test_tiled_lane_boundaries(
    shape: tuple[int, int, int], tile_size: int, vector_width: int,
    expected: tuple[int, ...],
) -> None:
    m, k, n = shape
    report = compare_matmul_lowering(
        _shaped_payload(m, k, n), mode="tiled",
        tile_size=tile_size, vector_width=vector_width,
    )
    assert tuple(_tiled_counts(report).values()) == expected
    assert report["after"]["totals"]["flops"] == 2 * m * k * n
    assert report["after"]["totals"]["bytes_read"] == (m * k + k * n) * 4
    assert report["after"]["totals"]["bytes_written"] == m * n * 4


def test_tiled_lowering_cli_accepts_mode_and_parameters() -> None:
    result = _run_cli(
        "--lower-matmul",
        "--lowering-mode",
        "tiled",
        "--tile-size",
        "2",
        "--vector-width",
        "2",
        "--compact",
        str(FIXTURE_PATH),
    )

    assert result.returncode == 0
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["after"]["representation"]["kind"] == "tiled_vectorized"
    assert report["after"]["totals"]["flops"] == report["before"]["totals"]["flops"]


def test_tiled_compact_cli_is_deterministic(tmp_path: Path) -> None:
    graph_path = tmp_path / "ragged_matmul.json"
    graph_path.write_text(json.dumps(_shaped_payload(3, 5, 7)), encoding="utf-8")
    args = (
        "--lower-matmul", "--lowering-mode", "tiled", "--tile-size", "4",
        "--vector-width", "4", "--compact", str(graph_path),
    )
    first = _run_cli(*args)
    second = _run_cli(*args)
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    assert json.loads(first.stdout)["after"]["representation"]["kind"] == "tiled_vectorized"
    assert _tiled_counts(json.loads(first.stdout))["scalar_tail_fma"] == 45


def test_lowering_mode_requires_lowering_flag() -> None:
    result = _run_cli("--lowering-mode", "tiled", str(FIXTURE_PATH))

    assert result.returncode == 2
    assert "--lowering-mode requires --lower-matmul" in result.stderr


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--tile-size", "0"),
        ("--tile-size", "-1"),
        ("--tile-size", "not-an-int"),
        ("--vector-width", "0"),
        ("--vector-width", "-1"),
        ("--vector-width", "not-an-int"),
    ],
)
def test_lowering_cli_rejects_invalid_tile_parameters_with_status_two(
    option: str, value: str
) -> None:
    result = _run_cli(
        "--lower-matmul", option, value, str(FIXTURE_PATH)
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "error:" in result.stderr


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda payload: payload["ops"].append(
                {
                    "name": "relu",
                    "op": "relu",
                    "inputs": ["output"],
                    "outputs": ["final"],
                }
            ),
            "requires exactly one operation",
        ),
        (
            lambda payload: payload["ops"][0].update({"op": "relu", "inputs": ["lhs"]}),
            "requires a single matmul operation",
        ),
        (
            lambda payload: payload.update({"outputs": ["lhs"]}),
            "requires the matmul result to be the only graph output",
        ),
    ],
)
def test_matmul_lowering_rejects_out_of_scope_graphs(
    change: object, message: str
) -> None:
    payload = deepcopy(_fixture_payload())
    change(payload)  # type: ignore[operator]

    with pytest.raises(LoweringValidationError, match=message):
        compare_matmul_lowering(payload)


def test_lowering_cli_is_opt_in_and_deterministic() -> None:
    first = _run_cli("--lower-matmul", "--compact", str(FIXTURE_PATH))
    second = _run_cli("--lower-matmul", "--compact", str(FIXTURE_PATH))

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout

    report = json.loads(first.stdout)
    assert report["before"]["totals"]["flops"] == 48
    assert report["after"]["totals"]["bytes_read"] == 192
    assert report["cost_delta"]["operation_count"] == 87


def test_lowering_cli_rejects_benchmark_and_optimizer_modes() -> None:
    benchmark = REPO_ROOT / "benchmarks" / "target_bounds.json"

    benchmark_result = _run_cli(
        "--lower-matmul", "--benchmark", str(benchmark), "--compact"
    )
    assert benchmark_result.returncode == 2
    assert "error: --lower-matmul requires a single graph path" in benchmark_result.stderr

    optimize_result = _run_cli(
        "--lower-matmul", "--optimize", str(FIXTURE_PATH), "--compact"
    )
    assert optimize_result.returncode == 2
    assert "error: --lower-matmul cannot be combined with --optimize" in optimize_result.stderr
