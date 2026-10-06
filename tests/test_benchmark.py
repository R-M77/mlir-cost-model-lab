from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_BOUNDS_BENCHMARK = REPO_ROOT / "benchmarks" / "target_bounds.json"
TARGET_BOUNDS_EXPECTED = REPO_ROOT / "benchmarks" / "target_bounds.expected.json"
TARGET_BOUNDS_BUDGET = REPO_ROOT / "benchmarks" / "target_bounds.budget.json"
LOWERING_GRAPH = REPO_ROOT / "examples" / "matmul_lowering.json"


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


def test_benchmark_case_selection_is_repeatable_and_preserves_original_order() -> None:
    result = _run_cli(
        "--benchmark",
        str(TARGET_BOUNDS_BENCHMARK),
        "--case",
        "memory_bound_relu",
        "--case",
        "compute_bound_matmul",
        "--compact",
    )

    assert result.returncode == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert [case["name"] for case in payload["cases"]] == [
        "compute_bound_matmul",
        "memory_bound_relu",
    ]
    assert payload["summary"]["case_count"] == 2
    assert payload["summary"]["totals"]["flops"] == 1_572_864


def test_benchmark_case_selection_recomputes_summary_and_budget(tmp_path: Path) -> None:
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "flops": 1_048_576,
                    "logical_bytes": 8_388_608,
                    "estimated_time_us": 83.88608,
                },
                "cases": {
                    "memory_bound_relu": {
                        "flops": 1_048_576,
                        "logical_bytes": 8_388_608,
                        "estimated_time_us": 83.88608,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = _run_cli(
        "--benchmark",
        str(TARGET_BOUNDS_BENCHMARK),
        "--case",
        "memory_bound_relu",
        "--budget",
        str(budget_path),
        "--compact",
    )

    assert result.returncode == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert [case["name"] for case in payload["cases"]] == ["memory_bound_relu"]
    assert payload["summary"] == {
        "case_count": 1,
        "totals": {
            "flops": 1_048_576,
            "bytes_read": 4_194_304,
            "bytes_written": 4_194_304,
            "logical_bytes": 8_388_608,
            "arithmetic_intensity": 0.125,
            "estimated_time_us": 83.88608,
        },
    }


def test_benchmark_case_selection_rejects_unknown_name() -> None:
    result = _run_cli(
        "--benchmark",
        str(TARGET_BOUNDS_BENCHMARK),
        "--case",
        "missing_case",
        "--compact",
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == "error: unknown benchmark case name 'missing_case'"


def test_case_selection_requires_benchmark(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(_relu_graph("single_case")), encoding="utf-8")

    result = _run_cli("--case", "single_case", str(graph_path))

    assert result.returncode == 2
    assert result.stdout == ""
    assert "error: --case requires --benchmark JSON" in result.stderr


def test_graph_path_and_benchmark_are_mutually_exclusive(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(_relu_graph("single_case")), encoding="utf-8")

    result = _run_cli(str(graph_path), "--benchmark", str(TARGET_BOUNDS_BENCHMARK))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.endswith(
        "error: graph path and --benchmark JSON are mutually exclusive\n"
    )


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


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        pytest.param(
            [],
            "benchmark JSON must contain at least one case",
            id="empty-array",
        ),
        pytest.param(
            [42],
            "benchmark[0] must be a non-empty string",
            id="non-string-path",
        ),
        pytest.param(
            ["   "],
            "benchmark[0] must be a non-empty string",
            id="blank-path",
        ),
        pytest.param(
            [{"name": "named_case", "path": None}],
            "benchmark[0].path must be a non-empty string",
            id="non-string-object-path",
        ),
        pytest.param(
            [{"name": "named_case", "path": " "}],
            "benchmark[0].path must be a non-empty string",
            id="blank-object-path",
        ),
    ],
)
def test_benchmark_manifest_cardinality_and_path_types_return_cli_error(
    tmp_path: Path, payload: object, message: str
) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_cli("--benchmark", str(benchmark_path))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == f"error: {message}\n"


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ({"path": "graph.json"}, "benchmark[0].name is required for object cases"),
        ({"name": "named_case"}, "benchmark[0].path is required for object cases"),
        (
            {"name": None, "path": "graph.json"},
            "benchmark[0].name must be a non-empty string",
        ),
        (
            {"name": "named_case", "path": "graph.json", "description": "extra"},
            "benchmark[0] has unsupported field(s): 'description'",
        ),
    ],
)
def test_malformed_benchmark_object_case_returns_cli_error(
    tmp_path: Path, case: dict[str, object], message: str
) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps([case]), encoding="utf-8")

    result = _run_cli("--benchmark", str(benchmark_path))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == f"error: {message}"


def test_malformed_benchmark_json_returns_cli_error(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text("{", encoding="utf-8")

    result = _run_cli("--benchmark", str(benchmark_path))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("error: invalid benchmark JSON: ")
    assert result.stderr.endswith("at line 1 column 2\n")


def test_invalid_utf8_benchmark_json_returns_cli_error(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_bytes(b"\xff")

    result = _run_cli("--benchmark", str(benchmark_path))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "error: invalid benchmark UTF-8 at byte 0: invalid start byte\n"


def test_unreadable_benchmark_path_returns_cli_error(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark-directory"
    benchmark_path.mkdir()

    result = _run_cli("--benchmark", str(benchmark_path))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith(
        f"error: could not read benchmark {benchmark_path}: "
    )


def test_missing_graph_path_in_benchmark_returns_cli_error(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    graph_path = tmp_path / "missing.json"
    benchmark_path.write_text(
        json.dumps([{"name": "missing_case", "path": graph_path.name}]),
        encoding="utf-8",
    )

    result = _run_cli("--benchmark", str(benchmark_path))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith(
        f"error: benchmark case 'missing_case': could not read graph {graph_path}: "
    )


def test_malformed_graph_json_in_benchmark_returns_cli_error(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    graph_path = tmp_path / "malformed.json"
    graph_path.write_text("{", encoding="utf-8")
    benchmark_path.write_text(
        json.dumps([{"name": "malformed_case", "path": graph_path.name}]),
        encoding="utf-8",
    )

    result = _run_cli("--benchmark", str(benchmark_path))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith(
        "error: benchmark case 'malformed_case': invalid JSON: "
    )
    assert result.stderr.endswith("at line 1 column 2\n")


def test_invalid_utf8_graph_json_in_benchmark_returns_cli_error(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    graph_path = tmp_path / "invalid.json"
    graph_path.write_bytes(b"\xff")
    benchmark_path.write_text(
        json.dumps([{"name": "invalid_case", "path": graph_path.name}]),
        encoding="utf-8",
    )

    result = _run_cli("--benchmark", str(benchmark_path))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == (
        "error: benchmark case 'invalid_case': invalid graph UTF-8 at byte 0: "
        "invalid start byte\n"
    )


def test_missing_positional_graph_path_returns_cli_error(tmp_path: Path) -> None:
    graph_path = tmp_path / "missing.json"

    result = _run_cli(str(graph_path))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith(f"error: could not read graph {graph_path}: ")


def test_malformed_positional_graph_json_returns_cli_error(tmp_path: Path) -> None:
    graph_path = tmp_path / "malformed.json"
    graph_path.write_text("{", encoding="utf-8")

    result = _run_cli(str(graph_path))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("error: invalid JSON: ")
    assert result.stderr.endswith("at line 1 column 2\n")


def test_invalid_utf8_positional_graph_json_returns_cli_error(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "invalid.json"
    graph_path.write_bytes(b"\xff")

    result = _run_cli(str(graph_path))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "error: invalid graph UTF-8 at byte 0: invalid start byte\n"


def test_benchmark_rejects_duplicate_effective_case_names(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(_relu_graph("same_case")), encoding="utf-8")

    for benchmark in (
        ["graph.json", "graph.json"],
        [
            {"name": "same_case", "path": "graph.json"},
            {"name": "same_case", "path": "graph.json"},
        ],
    ):
        benchmark_path = tmp_path / "benchmark.json"
        benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")

        result = _run_cli("--benchmark", str(benchmark_path), "--compact")

        assert result.returncode == 2
        assert result.stdout == ""
        assert result.stderr.strip() == (
            "error: benchmark case name 'same_case' must be unique"
        )


def test_benchmark_case_selection_does_not_skip_duplicate_validation(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(_relu_graph("same_case")), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            [
                {"name": "selected_case", "path": "graph.json"},
                {"name": "same_case", "path": "graph.json"},
                {"name": "same_case", "path": "graph.json"},
            ]
        ),
        encoding="utf-8",
    )

    result = _run_cli(
        "--benchmark",
        str(benchmark_path),
        "--case",
        "selected_case",
        "--compact",
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == (
        "error: benchmark case name 'same_case' must be unique"
    )


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


def test_optimized_fusion_budget_uses_aggregate_and_case_totals(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "fusion.json"
    graph_path.write_text(json.dumps(_fusion_graph()), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(["fusion.json"]), encoding="utf-8")
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "aggregate": {"logical_bytes": 24},
                "cases": {"fusion_case": {"logical_bytes": 24}},
            }
        ),
        encoding="utf-8",
    )

    result = _run_cli(
        "--benchmark",
        str(benchmark_path),
        "--budget",
        str(budget_path),
        "--optimize",
        "--compact",
    )

    assert result.returncode == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["summary"]["totals"]["logical_bytes"] == 24
    optimized_totals = payload["cases"][0]["report"]["optimized"]["totals"]
    assert optimized_totals["bytes_read"] + optimized_totals["bytes_written"] == 24


def test_optimized_fusion_per_case_budget_violation_preserves_report(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "fusion.json"
    graph_path.write_text(json.dumps(_fusion_graph()), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(["fusion.json"]), encoding="utf-8")
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "aggregate": {"logical_bytes": 24},
                "cases": {"fusion_case": {"logical_bytes": 23}},
            }
        ),
        encoding="utf-8",
    )

    baseline = _run_cli(
        "--benchmark", str(benchmark_path), "--optimize", "--compact"
    )
    first = _run_cli(
        "--benchmark",
        str(benchmark_path),
        "--budget",
        str(budget_path),
        "--optimize",
        "--compact",
    )
    second = _run_cli(
        "--benchmark",
        str(benchmark_path),
        "--budget",
        str(budget_path),
        "--optimize",
        "--compact",
    )

    assert baseline.returncode == 0
    assert first.returncode == second.returncode == 2
    assert first.stdout == second.stdout == baseline.stdout
    assert first.stderr == second.stderr == (
        'error: {"error":"benchmark budget exceeded","violations":'
        '[{"actual":24,"limit":23,"metric":"logical_bytes","scope":"case"}]}\n'
    )


def _two_fusion_case_benchmark(tmp_path: Path) -> Path:
    graph_path = tmp_path / "fusion.json"
    graph_path.write_text(json.dumps(_fusion_graph()), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            [
                {"name": "selected_fusion", "path": "fusion.json"},
                {"name": "unselected_fusion", "path": "fusion.json"},
            ]
        ),
        encoding="utf-8",
    )
    return benchmark_path


def test_optimized_case_selection_budget_uses_selected_optimized_totals(
    tmp_path: Path,
) -> None:
    benchmark_path = _two_fusion_case_benchmark(tmp_path)
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "aggregate": {"logical_bytes": 24},
                "cases": {"selected_fusion": {"logical_bytes": 24}},
            }
        ),
        encoding="utf-8",
    )

    result = _run_cli(
        "--benchmark",
        str(benchmark_path),
        "--case",
        "selected_fusion",
        "--budget",
        str(budget_path),
        "--optimize",
        "--compact",
    )

    assert result.returncode == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert [case["name"] for case in payload["cases"]] == ["selected_fusion"]
    assert payload["summary"]["case_count"] == 1
    assert payload["summary"]["totals"]["logical_bytes"] == 24
    optimized = payload["cases"][0]["report"]["optimized"]
    assert optimized["bytes_avoided"] == 16
    assert (
        optimized["totals"]["bytes_read"] + optimized["totals"]["bytes_written"]
        == 24
    )


def test_optimized_case_selection_budget_violation_preserves_report(
    tmp_path: Path,
) -> None:
    benchmark_path = _two_fusion_case_benchmark(tmp_path)
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "aggregate": {"logical_bytes": 24},
                "cases": {"selected_fusion": {"logical_bytes": 23}},
            }
        ),
        encoding="utf-8",
    )

    baseline = _run_cli(
        "--benchmark",
        str(benchmark_path),
        "--case",
        "selected_fusion",
        "--optimize",
        "--compact",
    )
    first = _run_cli(
        "--benchmark",
        str(benchmark_path),
        "--case",
        "selected_fusion",
        "--budget",
        str(budget_path),
        "--optimize",
        "--compact",
    )
    second = _run_cli(
        "--benchmark",
        str(benchmark_path),
        "--case",
        "selected_fusion",
        "--budget",
        str(budget_path),
        "--optimize",
        "--compact",
    )

    assert baseline.returncode == 0
    assert first.returncode == second.returncode == 2
    assert first.stdout == second.stdout == baseline.stdout
    assert first.stderr == second.stderr == (
        'error: {"error":"benchmark budget exceeded","violations":'
        '[{"actual":24,"limit":23,"metric":"logical_bytes","scope":"case"}]}\n'
    )


def test_optimized_case_selection_rejects_unselected_case_budget_deterministically(
    tmp_path: Path,
) -> None:
    benchmark_path = _two_fusion_case_benchmark(tmp_path)
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "aggregate": {"logical_bytes": 24},
                "cases": {"unselected_fusion": {"logical_bytes": 24}},
            }
        ),
        encoding="utf-8",
    )

    first = _run_cli(
        "--benchmark",
        str(benchmark_path),
        "--case",
        "selected_fusion",
        "--budget",
        str(budget_path),
        "--optimize",
        "--compact",
    )
    second = _run_cli(
        "--benchmark",
        str(benchmark_path),
        "--case",
        "selected_fusion",
        "--budget",
        str(budget_path),
        "--optimize",
        "--compact",
    )

    assert first.returncode == second.returncode == 2
    assert first.stdout == second.stdout == ""
    assert first.stderr == second.stderr == (
        "error: invalid budget case 'unselected_fusion' must match exactly one benchmark case\n"
    )


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


def test_unknown_per_case_budget_name_returns_cli_error(tmp_path: Path) -> None:
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "aggregate": {"flops": 1_572_864},
                "cases": {"missing_case": {"flops": 1}},
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
    assert result.stdout == ""
    assert result.stderr == (
        "error: invalid budget case 'missing_case' must match exactly one benchmark case\n"
    )


def test_budget_requires_benchmark_when_a_graph_is_given(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(_relu_graph("single_case")), encoding="utf-8")
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(json.dumps({"aggregate": {"flops": 2}}), encoding="utf-8")

    result = _run_cli(str(graph_path), "--budget", str(budget_path))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.endswith("error: --budget requires --benchmark JSON\n")


@pytest.mark.parametrize(
    ("args", "message"),
    [
        pytest.param(
            (),
            "a graph path or --benchmark JSON is required",
            id="no-source",
        ),
        pytest.param(
            ("--budget", str(TARGET_BOUNDS_BUDGET)),
            "a graph path or --benchmark JSON is required",
            id="budget-without-source",
        ),
        pytest.param(
            ("--lower-matmul",),
            "a graph path or --benchmark JSON is required",
            id="lowering-without-graph",
        ),
    ],
)
def test_cli_source_selection_rejection_matrix(
    args: tuple[str, ...], message: str
) -> None:
    result = _run_cli(*args)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.endswith(f"error: {message}\n")


@pytest.mark.parametrize(
    ("args", "message"),
    [
        pytest.param(
            (
                "--case",
                "missing_case",
                "--budget",
                str(TARGET_BOUNDS_BUDGET),
                str(LOWERING_GRAPH),
            ),
            "--case requires --benchmark JSON",
            id="case-before-budget-on-single-graph",
        ),
        pytest.param(
            (
                "--case",
                "missing_case",
                "--lowering-mode",
                "tiled",
                str(LOWERING_GRAPH),
            ),
            "--case requires --benchmark JSON",
            id="case-before-lowering-mode-on-single-graph",
        ),
        pytest.param(
            ("--case", "missing_case", "--lower-matmul", str(LOWERING_GRAPH)),
            "--case requires --benchmark JSON",
            id="case-before-lowering-on-single-graph",
        ),
        pytest.param(
            (
                "--budget",
                str(TARGET_BOUNDS_BUDGET),
                "--lowering-mode",
                "tiled",
                str(LOWERING_GRAPH),
            ),
            "--lowering-mode requires --lower-matmul",
            id="lowering-mode-before-budget-on-single-graph",
        ),
        pytest.param(
            ("--budget", str(TARGET_BOUNDS_BUDGET), "--lower-matmul", str(LOWERING_GRAPH)),
            "--budget requires --benchmark JSON",
            id="budget-before-lowering-on-single-graph",
        ),
        pytest.param(
            (
                "--case",
                "missing_case",
                "--budget",
                str(TARGET_BOUNDS_BUDGET),
                "--lower-matmul",
                str(LOWERING_GRAPH),
            ),
            "--case requires --benchmark JSON",
            id="case-before-budget-and-lowering-on-single-graph",
        ),
        pytest.param(
            (
                "--lower-matmul",
                "--benchmark",
                str(TARGET_BOUNDS_BENCHMARK),
                "--case",
                "compute_bound_matmul",
                "--budget",
                str(TARGET_BOUNDS_BUDGET),
            ),
            "--lower-matmul requires a single graph path",
            id="lowering-before-valid-benchmark-options",
        ),
        pytest.param(
            (
                "--lower-matmul",
                "--optimize",
                "--benchmark",
                str(TARGET_BOUNDS_BENCHMARK),
                "--case",
                "compute_bound_matmul",
                "--budget",
                str(TARGET_BOUNDS_BUDGET),
            ),
            "--lower-matmul requires a single graph path",
            id="benchmark-before-lowering-optimizer-conflict",
        ),
        pytest.param(
            (
                "--lowering-mode",
                "tiled",
                "--benchmark",
                str(TARGET_BOUNDS_BENCHMARK),
                "--case",
                "compute_bound_matmul",
                "--budget",
                str(TARGET_BOUNDS_BUDGET),
            ),
            "--lowering-mode requires --lower-matmul",
            id="lowering-mode-before-valid-benchmark-options",
        ),
        pytest.param(
            (
                "--lower-matmul",
                "--optimize",
                "--budget",
                str(TARGET_BOUNDS_BUDGET),
                str(LOWERING_GRAPH),
            ),
            "--lower-matmul cannot be combined with --optimize",
            id="lowering-optimizer-before-budget-on-single-graph",
        ),
        pytest.param(
            (
                "--lower-matmul",
                "--tile-size",
                "0",
                "--budget",
                str(TARGET_BOUNDS_BUDGET),
                str(LOWERING_GRAPH),
            ),
            "--tile-size and --vector-width must be positive",
            id="lowering-parameter-before-budget-on-single-graph",
        ),
    ],
)
def test_cli_rejection_precedence_matrix(
    args: tuple[str, ...], message: str
) -> None:
    result = _run_cli(*args)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.endswith(f"error: {message}\n")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        pytest.param(
            [],
            "invalid budget must contain an object",
            id="non-object-payload",
        ),
        pytest.param(
            {},
            "invalid budget requires an aggregate object",
            id="missing-aggregate",
        ),
        pytest.param(
            {"aggregate": []},
            "invalid budget aggregate must be an object",
            id="non-object-aggregate",
        ),
        pytest.param(
            {"aggregate": {}},
            "invalid budget aggregate must define at least one metric",
            id="empty-aggregate",
        ),
        pytest.param(
            {"aggregate": {"flops": 1}, "unexpected": True},
            "invalid budget has unsupported field(s): 'unexpected'",
            id="unsupported-top-level-field",
        ),
    ],
)
def test_invalid_budget_structure_schema_fails_closed(
    tmp_path: Path, payload: object, message: str
) -> None:
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_cli(
        "--benchmark",
        str(TARGET_BOUNDS_BENCHMARK),
        "--budget",
        str(budget_path),
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == f"error: {message}\n"


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


def test_unreadable_budget_path_returns_cli_error(tmp_path: Path) -> None:
    budget_path = tmp_path / "budget-directory"
    budget_path.mkdir()

    result = _run_cli(
        "--benchmark",
        str(TARGET_BOUNDS_BENCHMARK),
        "--budget",
        str(budget_path),
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith(
        f"error: invalid budget {budget_path}: could not read file: "
    )


@pytest.mark.parametrize(
    ("case_limits", "message"),
    [
        ([], "invalid budget cases['memory_bound_relu'] must be an object"),
        (
            {},
            "invalid budget cases['memory_bound_relu'] must define at least one metric",
        ),
        (
            {"unknown_metric": 1},
            "invalid budget cases['memory_bound_relu'] has unsupported metric(s): 'unknown_metric'",
        ),
    ],
)
def test_invalid_budget_case_entry_schema_fails_closed(
    tmp_path: Path, case_limits: object, message: str
) -> None:
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "aggregate": {"flops": 1_572_864},
                "cases": {"memory_bound_relu": case_limits},
            }
        ),
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
    assert result.stderr.strip() == f"error: {message}"


def test_invalid_utf8_budget_json_returns_cli_error(tmp_path: Path) -> None:
    budget_path = tmp_path / "budget.json"
    budget_path.write_bytes(b"\xff")

    result = _run_cli(
        "--benchmark",
        str(TARGET_BOUNDS_BENCHMARK),
        "--budget",
        str(budget_path),
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "error: invalid budget UTF-8 at byte 0: invalid start byte\n"


@pytest.mark.parametrize("optimize", [False, True], ids=["baseline", "optimized"])
def test_targetless_estimated_time_is_optional_but_cannot_be_checked(
    tmp_path: Path, optimize: bool
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
    optimize_args = ("--optimize",) if optimize else ()
    allowed = _run_cli(
        "--benchmark",
        str(benchmark_path),
        *optimize_args,
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
        *optimize_args,
        "--budget",
        str(with_time),
    )
    assert rejected.returncode == 2
    assert rejected.stdout == ""
    assert rejected.stderr == (
        "error: budget aggregate.estimated_time_us is unavailable in the benchmark report\n"
    )

    per_case_time = tmp_path / "per_case_time.json"
    per_case_time.write_text(
        json.dumps(
            {
                "aggregate": {"flops": 2},
                "cases": {"targetless": {"estimated_time_us": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    per_case_rejected = _run_cli(
        "--benchmark",
        str(benchmark_path),
        *optimize_args,
        "--budget",
        str(per_case_time),
    )
    assert per_case_rejected.returncode == 2
    assert per_case_rejected.stdout == ""
    assert per_case_rejected.stderr == (
        "error: budget cases['targetless'].estimated_time_us is unavailable in the benchmark report\n"
    )


@pytest.mark.parametrize(
    ("budget", "message"),
    [
        pytest.param(
            {"aggregate": {"flops": True}},
            "invalid budget aggregate.flops must be a finite non-negative number",
            id="boolean-true-metric",
        ),
        pytest.param(
            {"aggregate": {"flops": False}},
            "invalid budget aggregate.flops must be a finite non-negative number",
            id="boolean-false-metric",
        ),
        pytest.param(
            {"aggregate": {"flops": -1}},
            "invalid budget aggregate.flops must be a finite non-negative number",
            id="negative-metric",
        ),
        pytest.param(
            {"aggregate": {"flops": float("nan")}},
            "invalid budget aggregate.flops must be a finite non-negative number",
            id="nan-metric",
        ),
        pytest.param(
            {"aggregate": {"flops": float("inf")}},
            "invalid budget aggregate.flops must be a finite non-negative number",
            id="positive-infinity-metric",
        ),
        pytest.param(
            {"aggregate": {"flops": float("-inf")}},
            "invalid budget aggregate.flops must be a finite non-negative number",
            id="negative-infinity-metric",
        ),
        pytest.param(
            {"aggregate": {"flops": 1}, "cases": []},
            "invalid budget cases must be an object",
            id="non-object-cases",
        ),
        pytest.param(
            {"aggregate": {"flops": 1}, "cases": {" ": {"flops": 1}}},
            "invalid budget case names must be non-empty strings",
            id="blank-case-name",
        ),
    ],
)
def test_invalid_budget_validation_edges_fail_closed(
    tmp_path: Path, budget: dict[str, object], message: str
) -> None:
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(json.dumps(budget), encoding="utf-8")

    result = _run_cli(
        "--benchmark",
        str(TARGET_BOUNDS_BENCHMARK),
        "--budget",
        str(budget_path),
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == f"error: {message}\n"
