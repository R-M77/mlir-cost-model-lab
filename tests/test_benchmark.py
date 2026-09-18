from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_invalid_benchmark_input_returns_cli_error(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps({"path": "graph.json"}), encoding="utf-8")

    result = _run_cli("--benchmark", str(benchmark_path))

    assert result.returncode == 2
    assert result.stdout == ""
    assert "error: benchmark JSON must contain an array" in result.stderr
