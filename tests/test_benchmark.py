from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_BOUNDS_BENCHMARK = REPO_ROOT / "benchmarks" / "target_bounds.json"
TARGET_BOUNDS_EXPECTED = REPO_ROOT / "benchmarks" / "target_bounds.expected.json"
TARGET_BOUNDS_BUDGET = REPO_ROOT / "benchmarks" / "target_bounds.budget.json"


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


def _relu_graph(name: str) -> dict[str, object]:
    return {
        "name": name,
        "inputs": {"x": {"shape": [2], "dtype": "f32"}},
        "ops": [
            {"name": "relu", "op": "relu", "inputs": ["x"], "outputs": ["output"]}
        ],
        "outputs": ["output"],
    }


def _fusion_graph() -> dict[str, object]:
    return {
        "name": "fusion_case",
        "inputs": {
            "x": {"shape": [2], "dtype": "f32"},
            "y": {"shape": [2], "dtype": "f32"},
        },
        "ops": [
            {"name": "add", "op": "add", "inputs": ["x", "y"], "outputs": ["added"]},
            {"name": "relu", "op": "relu", "inputs": ["added"], "outputs": ["output"]},
        ],
        "outputs": ["output"],
    }


def test_benchmark_accepts_paths_and_named_cases_with_stable_summary(
    tmp_path: Path,
) -> None:
    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()
    (graph_dir / "relu.json").write_text(
        json.dumps(_relu_graph("relu_case")), encoding="utf-8"
    )
    (graph_dir / "add.json").write_text(
        json.dumps(
            {
                "name": "add_case",
                "inputs": {
                    "x": {"shape": [2], "dtype": "f32"},
                    "y": {"shape": [2], "dtype": "f32"},
                },
                "ops": [
                    {
                        "name": "add",
                        "op": "add",
                        "inputs": ["x", "y"],
                        "outputs": ["output"],
                    }
                ],
                "outputs": ["output"],
            }
        ),
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            [
                "graphs/relu.json",
                {"name": "named_add", "path": "graphs/add.json"},
            ]
        ),
        encoding="utf-8",
    )

    first = _run_cli("--benchmark", str(benchmark_path), "--compact")
    second = _run_cli("--benchmark", str(benchmark_path), "--compact")

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout

    result = json.loads(first.stdout)
    assert list(result) == ["cases", "summary"]
    assert [case["name"] for case in result["cases"]] == ["relu_case", "named_add"]
    assert [case["path"] for case in result["cases"]] == [
        "graphs/relu.json",
        "graphs/add.json",
    ]
    assert result["summary"] == {
        "case_count": 2,
        "totals": {
            "flops": 4,
            "bytes_read": 24,
            "bytes_written": 16,
            "logical_bytes": 40,
            "arithmetic_intensity": 0.1,
            "estimated_time_us": None,
        },
    }


def test_benchmark_optimize_aggregates_optimized_totals(tmp_path: Path) -> None:
    graph_path = tmp_path / "fusion.json"
    graph_path.write_text(json.dumps(_fusion_graph()), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(["fusion.json"]), encoding="utf-8")

    result = _run_cli("--benchmark", str(benchmark_path), "--optimize", "--compact")

    assert result.returncode == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["cases"][0]["report"]["optimized"]["bytes_avoided"] == 16
    assert payload["summary"]["totals"] == {
        "flops": 4,
        "bytes_read": 16,
        "bytes_written": 8,
        "logical_bytes": 24,
        "arithmetic_intensity": 4 / 24,
        "estimated_time_us": None,
    }


def test_target_bound_fixture_has_stable_classification_and_aggregate() -> None:
    expected = TARGET_BOUNDS_EXPECTED.read_text(encoding="utf-8")

    first = _run_cli("--benchmark", str(TARGET_BOUNDS_BENCHMARK), "--compact")
    second = _run_cli("--benchmark", str(TARGET_BOUNDS_BENCHMARK), "--compact")

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout == expected

    payload = json.loads(first.stdout)
    assert [case["name"] for case in payload["cases"]] == [
        "compute_bound_matmul",
        "memory_bound_relu",
    ]
    assert [
        case["report"]["totals"]["roofline"]["bound"]
        for case in payload["cases"]
    ] == ["compute", "memory"]
    assert payload["summary"] == {
        "case_count": 2,
        "totals": {
            "flops": 1_572_864,
            "bytes_read": 4_227_072,
            "bytes_written": 4_210_688,
            "logical_bytes": 8_437_760,
            "arithmetic_intensity": 1_572_864 / 8_437_760,
            "estimated_time_us": 84.410368,
        },
    }


def test_invalid_benchmark_input_returns_cli_error(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps({"path": "graph.json"}), encoding="utf-8")

    result = _run_cli("--benchmark", str(benchmark_path))

    assert result.returncode == 2
    assert result.stdout == ""
    assert "error: benchmark JSON must contain an array" in result.stderr


def _target_bounds_budget() -> dict[str, object]:
    return {
        "aggregate": {
            "flops": 1_572_864,
            "logical_bytes": 8_437_760,
            "estimated_time_us": 84.410368,
        },
        "cases": {
            "compute_bound_matmul": {
                "flops": 524_288,
                "logical_bytes": 49_152,
                "estimated_time_us": 0.524288,
            },
            "memory_bound_relu": {
                "flops": 1_048_576,
                "logical_bytes": 8_388_608,
                "estimated_time_us": 83.88608,
            },
        },
    }


def test_benchmark_budget_pass_preserves_report_and_is_deterministic() -> None:
    baseline = _run_cli("--benchmark", str(TARGET_BOUNDS_BENCHMARK), "--compact")
    first = _run_cli(
        "--benchmark",
        str(TARGET_BOUNDS_BENCHMARK),
        "--budget",
        str(TARGET_BOUNDS_BUDGET),
        "--compact",
    )
    second = _run_cli(
        "--benchmark",
        str(TARGET_BOUNDS_BENCHMARK),
        "--budget",
        str(TARGET_BOUNDS_BUDGET),
        "--compact",
    )

    assert baseline.returncode == first.returncode == second.returncode == 0
    assert baseline.stderr == first.stderr == second.stderr == ""
    assert first.stdout == second.stdout == baseline.stdout


def test_benchmark_budget_violation_returns_report_and_json_diagnostic(
    tmp_path: Path,
) -> None:
    budget = _target_bounds_budget()
    budget["aggregate"]["flops"] = 1_572_863  # type: ignore[index]
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(json.dumps(budget), encoding="utf-8")

    baseline = _run_cli("--benchmark", str(TARGET_BOUNDS_BENCHMARK), "--compact")
    result = _run_cli(
        "--benchmark",
        str(TARGET_BOUNDS_BENCHMARK),
        "--budget",
        str(budget_path),
        "--compact",
    )

    assert baseline.returncode == 0
    assert result.returncode == 2
    assert result.stdout == baseline.stdout
    assert json.loads(result.stderr.removeprefix("error: ")) == {
        "error": "benchmark budget exceeded",
        "violations": [
            {
                "actual": 1_572_864,
                "limit": 1_572_863,
                "metric": "flops",
                "scope": "aggregate",
            }
        ],
    }


def test_per_case_budget_violation_is_reported(tmp_path: Path) -> None:
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "aggregate": {"flops": 1_572_864},
                "cases": {"memory_bound_relu": {"flops": 1_048_575}},
            }
        ),
        encoding="utf-8",
    )

    result = _run_cli(
        "--benchmark",
        str(TARGET_BOUNDS_BENCHMARK),
        "--budget",
        str(budget_path),
        "--compact",
    )

    assert result.returncode == 2
    assert json.loads(result.stderr.removeprefix("error: "))["violations"] == [
        {
            "actual": 1_048_576,
            "limit": 1_048_575,
            "metric": "flops",
            "scope": "case",
        }
    ]


def test_invalid_benchmark_budget_schema_fails_closed(tmp_path: Path) -> None:
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(
        json.dumps({"aggregate": {"flops": "not-a-number"}}),
        encoding="utf-8",
    )

    result = _run_cli(
        "--benchmark",
        str(TARGET_BOUNDS_BENCHMARK),
        "--budget",
        str(budget_path),
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "error: invalid budget aggregate.flops must be a finite non-negative number" in result.stderr


def test_targetless_estimated_time_is_optional_but_cannot_be_checked(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(_relu_graph("targetless")), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(["graph.json"]), encoding="utf-8")

    without_time = tmp_path / "without_time.json"
    without_time.write_text(
        json.dumps({"aggregate": {"flops": 2, "logical_bytes": 16}}),
        encoding="utf-8",
    )
    allowed = _run_cli(
        "--benchmark",
        str(benchmark_path),
        "--budget",
        str(without_time),
        "--compact",
    )
    assert allowed.returncode == 0
    assert allowed.stderr == ""
    assert json.loads(allowed.stdout)["summary"]["totals"]["estimated_time_us"] is None

    with_time = tmp_path / "with_time.json"
    with_time.write_text(
        json.dumps({"aggregate": {"estimated_time_us": 1.0}}),
        encoding="utf-8",
    )
    rejected = _run_cli(
        "--benchmark",
        str(benchmark_path),
        "--budget",
        str(with_time),
    )
    assert rejected.returncode == 2
    assert rejected.stdout == ""
    assert "error: budget aggregate.estimated_time_us is unavailable" in rejected.stderr
