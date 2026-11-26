"""Regression tests for the original-graph fusion model and its explanation."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from cost_model import analyze_graph, estimate_graph, load_graph, parse_graph
from cost_model.fusion_analysis import analyze_fusion

REPO_ROOT = Path(__file__).resolve().parents[1]


def _op(name: str, kind: str, inputs: list[str], output: str) -> dict:
    return {"name": name, "op": kind, "inputs": inputs, "outputs": [output]}


def _graph(ops: list[dict], outputs: list[str]) -> dict:
    return {
        "name": "synthetic",
        "inputs": {"x": {"shape": [3], "dtype": "f16"}},
        "ops": ops,
        "outputs": outputs,
    }


def test_default_estimator_excludes_retained_intermediate() -> None:
    ops = [_op("first", "relu", ["x"], "a"), _op("second", "relu", ["a"], "b")]
    unretained = analyze_graph(_graph(ops, ["b"]))
    retained = analyze_graph(_graph(ops, ["b", "a"]))

    assert unretained["fusion"] == {
        "count": 1,
        "candidates": [
            {
                "producer": "first",
                "consumer": "second",
                "intermediate": "a",
                "bytes_avoided": 12,
                "reason": "adjacent elementwise operations with a single-use intermediate",
            }
        ],
    }
    assert retained["fusion"] == {"count": 0, "candidates": []}
    assert retained["ops"] == unretained["ops"]
    assert retained["totals"] == unretained["totals"]


def _run_cli(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable, "-W", "error", "-X", "int_max_str_digits=4300",
            "-m", "cost_model.fusion_analysis", str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )


def test_cli_reports_eligible_edge_and_agrees_with_callable(tmp_path: Path) -> None:
    ops = [_op("first", "relu", ["x"], "a"), _op("second", "mul", ["x", "a"], "b")]
    payload = _graph(ops, ["b"])
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_cli(path)

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    expected = {
        "graph": "synthetic",
        "edges": [
            {
                "producer": "first",
                "consumer": "second",
                "intermediate": "a",
                "use_count": 1,
                "blocking_reasons": [],
                "eligible": True,
                "bytes_avoided": 12,
            }
        ],
        "total_bytes_avoided": 12,
    }
    assert json.loads(result.stdout) == expected
    assert _run_cli(path).stdout == result.stdout

    assert analyze_fusion(load_graph(path)) == expected


def test_explains_only_actual_edges_in_producer_then_consumer_order() -> None:
    graph = parse_graph(_graph([
        _op("normalize", "softmax", ["x"], "a"),
        _op("activate", "relu", ["a"], "b"),
        _op("offset", "add", ["b", "x"], "c"),
        _op("normalize_again", "softmax", ["a"], "d"),
        _op("square", "mul", ["c", "c"], "e"),
        _op("merge", "add", ["e", "d"], "f"),
        _op("finish", "relu", ["f"], "g"),
        _op("dead_use", "relu", ["f"], "dead"),
        _op("unused", "relu", ["x"], "unused_value"),
    ], ["a", "g"]))
    original = deepcopy(graph)
    report = analyze_fusion(graph)

    assert [
        (edge["producer"], edge["consumer"], edge["intermediate"], edge["use_count"],
         edge["blocking_reasons"], edge["eligible"], edge["bytes_avoided"])
        for edge in report["edges"]
    ] == [
        ("normalize", "activate", "a", 2,
         ["graph_output", "multiple_uses", "non_elementwise_producer"], False, 0),
        ("normalize", "normalize_again", "a", 2,
         ["graph_output", "multiple_uses", "non_adjacent", "non_elementwise_producer",
          "non_elementwise_consumer"], False, 0),
        ("activate", "offset", "b", 1, [], True, 12),
        ("offset", "square", "c", 2, ["multiple_uses", "non_adjacent"], False, 0),
        ("normalize_again", "merge", "d", 1, ["non_adjacent", "non_elementwise_producer"], False, 0),
        ("square", "merge", "e", 1, [], True, 12),
        ("merge", "finish", "f", 2, ["multiple_uses"], False, 0),
        ("merge", "dead_use", "f", 2, ["multiple_uses", "non_adjacent"], False, 0),
    ]
    assert report["total_bytes_avoided"] == 24
    assert graph == original
    assert analyze_fusion(graph) == report

    fields = ("producer", "consumer", "intermediate", "bytes_avoided")
    fusion = estimate_graph(graph)["fusion"]
    eligible = [{key: edge[key] for key in fields} for edge in report["edges"] if edge["eligible"]]
    assert fusion["count"] == len(eligible)
    assert [{key: row[key] for key in fields} for row in fusion["candidates"]] == eligible


@pytest.mark.parametrize(
    ("intervening", "consumer_kind", "operands", "outputs", "uses", "reason"),
    [
        (False, "relu", ["a"], ["b", "a"], 1, "graph_output"),
        (False, "mul", ["a", "a"], ["b"], 2, "multiple_uses"),
        (True, "relu", ["a"], ["b"], 1, "non_adjacent"),
        (False, "softmax", ["a"], ["b"], 1, "non_elementwise_consumer"),
    ],
)
def test_independent_blockers(
    intervening: bool, consumer_kind: str, operands: list[str], outputs: list[str],
    uses: int, reason: str,
) -> None:
    ops = [_op("first", "relu", ["x"], "a")]
    if intervening:
        ops.append(_op("independent", "relu", ["x"], "unused"))
    ops.append(_op("last", consumer_kind, operands, "b"))
    graph = parse_graph(_graph(ops, outputs))

    report = analyze_fusion(graph)

    (edge,) = report["edges"]
    assert edge["blocking_reasons"] == [reason]
    assert edge["use_count"] == uses
    assert edge["eligible"] is False
    assert edge["bytes_avoided"] == report["total_bytes_avoided"] == 0
    assert estimate_graph(graph)["fusion"] == {"count": 0, "candidates": []}


def test_empty_edge_report_does_not_include_graph_inputs_or_unused_results() -> None:
    graph = parse_graph(_graph([
        _op("first", "mul", ["x", "x"], "a"),
        _op("second", "relu", ["x"], "b"),
    ], ["x", "b"]))

    assert analyze_fusion(graph) == {"graph": "synthetic", "edges": [], "total_bytes_avoided": 0}


@pytest.mark.parametrize(
    ("content", "diagnostic"),
    [
        (None, "could not read graph"),
        (b"\xff", "invalid graph UTF-8"),
        (b"{", "invalid JSON"),
        (b"[]", "graph JSON must contain an object"),
        (b'{"name": "invalid", "inputs": {}, "ops": []}', "ops must contain at least one operation"),
        (b'{"name": ' + b"9" * 4301 + b"}", "integer string conversion"),
    ],
    ids=["missing", "utf8", "json", "root", "schema", "integer-limit"],
)
def test_cli_rejects_bad_input(tmp_path: Path, content: bytes | None, diagnostic: str) -> None:
    path = tmp_path / "graph.json"
    if content is not None:
        path.write_bytes(content)

    result = _run_cli(path)

    assert result.returncode == 2, result.stderr
    assert result.stdout == ""
    assert result.stderr.startswith("error: ")
    assert diagnostic in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_rejects_deep_invalid_json_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "graph.json"
    path.write_bytes(b"[" * 2000 + b"0" + b"]" * 2000)

    result = _run_cli(path)

    assert result.returncode == 2, result.stderr
    assert result.stdout == ""
    assert result.stderr.startswith("error: ")
    assert "Traceback" not in result.stderr


def test_cli_handles_report_integer_conversion_limit(tmp_path: Path) -> None:
    ops = [_op("first", "relu", ["x"], "a"), _op("second", "relu", ["a"], "b")]
    payload = _graph(ops, ["b"])
    payload["inputs"]["x"]["shape"] = ["BIG_DIM", "BIG_DIM"]
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(payload).replace('"BIG_DIM"', "9" * 2200), encoding="utf-8")

    result = _run_cli(path)

    assert result.returncode == 2, result.stderr
    assert result.stdout == ""
    assert result.stderr.startswith("error: ")
    assert "integer string conversion" in result.stderr
    assert "Traceback" not in result.stderr
