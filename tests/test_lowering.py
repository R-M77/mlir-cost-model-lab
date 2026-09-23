from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from cost_model import (
    LoweringValidationError,
    compare_matmul_lowering,
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
