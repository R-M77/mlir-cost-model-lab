from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.test_benchmark import _fusion_graph, _relu_graph, _run_cli


def _batch(tmp_path: Path, sizes: tuple[int, int]) -> Path:
    for name, size in zip(("alpha", "beta"), sizes):
        graph: Any = _relu_graph(name)
        graph["inputs"]["x"]["shape"] = [size]
        (tmp_path / f"{name}.json").write_text(json.dumps(graph), encoding="utf-8")
    manifest = tmp_path / "batch.json"
    manifest.write_text(json.dumps(["alpha.json", "beta.json"]), encoding="utf-8")
    return manifest


def _save_report(tmp_path: Path, manifest: Path, *options: str) -> Path:
    result = _run_cli("--benchmark", str(manifest), *options, "--compact")
    assert result.returncode == 0, result.stderr
    saved = tmp_path / "saved.json"
    saved.write_text(result.stdout, encoding="utf-8")
    return saved


def test_saved_benchmark_versions_gate_comparison_before_cases(tmp_path: Path) -> None:
    manifest = _batch(tmp_path, (2, 4))
    saved = _save_report(tmp_path, manifest)
    payload = json.loads(saved.read_text())
    versions = {"benchmark_schema_version": 1, "cost_model_version": 1}
    assert {field: payload.get(field) for field in versions} == versions
    assert all(type(payload[field]) is int for field in versions)
    repeated = _run_cli("--benchmark", str(manifest), "--compact")
    assert repeated.returncode == 0, repeated.stderr
    assert repeated.stdout == saved.read_text()

    compatible = _run_cli(
        "--benchmark", str(manifest), "--compare-baseline", str(saved), "--compact"
    )
    assert compatible.returncode == 0, compatible.stderr
    compared = json.loads(compatible.stdout)
    assert compared.pop("comparison")["passed"] is True
    assert compared == payload

    for field in versions:
        for value in (None, 2):
            incompatible = {**payload, "cases": []}
            if value is None:
                del incompatible[field]
            else:
                incompatible[field] = value
            saved.write_text(json.dumps(incompatible), encoding="utf-8")
            result = _run_cli(
                "--benchmark", str(manifest), "--compare-baseline", str(saved)
            )
            assert result.returncode == 2
            assert result.stdout == ""
            assert result.stderr == f"error: incompatible baseline {field}: expected integer 1\n"


def test_case_regression_fails_even_when_aggregate_is_unchanged(tmp_path: Path) -> None:
    manifest = _batch(tmp_path, (2, 4))
    saved = _save_report(tmp_path, manifest)
    _batch(tmp_path, (3, 3))

    result = _run_cli(
        "--benchmark", str(manifest), "--compare-baseline", str(saved), "--compact"
    )

    assert result.returncode == 2
    assert result.stdout, result.stderr
    payload = json.loads(result.stdout)
    assert payload["summary"] == json.loads(saved.read_text())["summary"]
    assert payload["comparison"] == {
        "mode": "baseline",
        "passed": False,
        "cases": [
            {
                "name": "alpha",
                "baseline": {"flops": 2, "logical_bytes": 16, "estimated_time_us": None},
                "current": {"flops": 3, "logical_bytes": 24, "estimated_time_us": None},
                "delta": {"flops": 1, "logical_bytes": 8, "estimated_time_us": None},
            },
            {
                "name": "beta",
                "baseline": {"flops": 4, "logical_bytes": 32, "estimated_time_us": None},
                "current": {"flops": 3, "logical_bytes": 24, "estimated_time_us": None},
                "delta": {"flops": -1, "logical_bytes": -8, "estimated_time_us": None},
            },
        ],
    }
    assert json.loads(result.stderr.removeprefix("error: ")) == {
        "error": "benchmark baseline exceeded",
        "violations": [
            {"case": "alpha", "metric": "flops", "baseline": 2, "current": 3, "delta": 1},
            {"case": "alpha", "metric": "logical_bytes", "baseline": 16, "current": 24, "delta": 8},
        ],
    }


@pytest.mark.parametrize("options", [(), ("examples/matmul_lowering.json",),
                                    ("--emit-mlir", "examples/matmul_lowering.json"),
                                    ("--lower-matmul", "examples/matmul_lowering.json")])
def test_comparison_requires_benchmark(options: tuple[str, ...]) -> None:
    result = _run_cli(*options, "--compare-baseline", "saved.json")
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.endswith("error: --compare-baseline requires --benchmark JSON\n")


@pytest.mark.parametrize("saved_optimized", [False, True])
def test_comparison_rejects_different_modes(tmp_path: Path, saved_optimized: bool) -> None:
    manifest = _batch(tmp_path, (2, 4))
    saved = _save_report(tmp_path, manifest, *(("--optimize",) if saved_optimized else ()))
    result = _run_cli(
        "--benchmark", str(manifest), "--compare-baseline", str(saved),
        *(("--optimize",) if not saved_optimized else ()),
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "error: baseline case 'alpha': incompatible report mode\n"


@pytest.mark.parametrize("selection", [(), ("--case", "alpha")])
def test_comparison_requires_exact_selected_case_set(tmp_path: Path, selection: tuple[str, ...]) -> None:
    manifest = _batch(tmp_path, (2, 4))
    saved_selection = ("--case", "alpha") if not selection else ()
    saved = _save_report(tmp_path, manifest, *saved_selection)
    result = _run_cli("--benchmark", str(manifest), *selection, "--compare-baseline", str(saved))
    assert result.returncode == 2
    assert result.stdout == ""
    missing, extra = (["beta"], []) if not selection else ([], ["beta"])
    assert result.stderr == f"error: incompatible baseline case set: missing={missing!r}, extra={extra!r}\n"


TARGET = {"peak_compute_gflops": 1000.0, "memory_bandwidth_gbps": 100.0}


def _set_target(tmp_path: Path, target: object) -> None:
    for name in ("alpha", "beta"):
        graph_path = tmp_path / f"{name}.json"
        graph = json.loads(graph_path.read_text())
        graph["target"] = target
        graph_path.write_text(json.dumps(graph), encoding="utf-8")


@pytest.mark.parametrize("options", [(), ("--optimize",)])
@pytest.mark.parametrize("field", ["peak_compute_gflops", "memory_bandwidth_gbps", "presence"])
def test_comparison_rejects_different_target_ceilings(
    tmp_path: Path, options: tuple[str, ...], field: str
) -> None:
    manifest = _batch(tmp_path, (2, 4))
    _set_target(tmp_path, TARGET)
    saved = _save_report(tmp_path, manifest, *options)
    _set_target(tmp_path, None if field == "presence" else {**TARGET, field: TARGET[field] * 2})
    result = _run_cli("--benchmark", str(manifest), *options, "--compare-baseline", str(saved))
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "error: baseline case 'alpha': incompatible target ceilings\n"


@pytest.mark.parametrize(("kind", "message"), [
    ("missing", "invalid baseline: could not read file:"),
    ("directory", "invalid baseline: could not read file:"),
    ("json", "invalid baseline JSON:"),
    ("utf8", "invalid baseline UTF-8:"),
    ("duplicate", "invalid baseline JSON: duplicate object key 'cases'"),
])
def test_comparison_reports_unreadable_or_invalid_json(
    tmp_path: Path, kind: str, message: str
) -> None:
    manifest = _batch(tmp_path, (2, 4))
    saved = tmp_path / "saved.json"
    if kind == "directory":
        saved.mkdir()
    elif kind == "json":
        saved.write_text("{", encoding="utf-8")
    elif kind == "utf8":
        saved.write_bytes(b"\xff")
    elif kind == "duplicate":
        saved.write_text('{"cases": [], "cases": []}', encoding="utf-8")
    result = _run_cli("--benchmark", str(manifest), "--compare-baseline", str(saved))
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith(f"error: {message}")
    assert "Traceback" not in result.stderr


def test_comparison_rejects_json_integer_digit_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _batch(tmp_path, (2, 4))
    saved = tmp_path / "saved.json"
    saved.write_text('{"private_marker": ' + "7" * 4301 + "}", encoding="utf-8")
    # Pin the child decoder limit independently of the caller's environment.
    monkeypatch.setenv("PYTHONINTMAXSTRDIGITS", "4300")

    result = _run_cli("--benchmark", str(manifest), "--compare-baseline", str(saved))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "error: invalid baseline JSON: numeric value exceeds decoder limits\n"
    assert "Traceback" not in result.stderr
    assert "private_marker" not in result.stderr
    assert "7" * 100 not in result.stderr


def test_comparison_rejects_excessive_json_nesting(tmp_path: Path) -> None:
    manifest = _batch(tmp_path, (2, 4))
    saved = tmp_path / "saved.json"
    saved.write_text(
        '{"benchmark_schema_version": 1, "cost_model_version": 1, "private_marker": '
        + "[" * 2000 + "0" + "]" * 2000 + "}", encoding="utf-8"
    )

    result = _run_cli("--benchmark", str(manifest), "--compare-baseline", str(saved))

    assert result.returncode == 2
    assert result.stdout == ""
    # Decoder nesting limits differ across supported Python versions. A decoder
    # that accepts this input must still reject its missing benchmark cases.
    assert result.stderr in {
        "error: invalid baseline JSON: nesting exceeds decoder limits\n",
        "error: invalid baseline cases must be a non-empty array\n",
    }
    assert "Traceback" not in result.stderr
    assert "private_marker" not in result.stderr


@pytest.mark.parametrize(("mutation", "message"), [
    ("root", "invalid baseline must contain a benchmark object"),
    ("missing-cases", "invalid baseline cases must be a non-empty array"),
    ("empty-cases", "invalid baseline cases must be a non-empty array"),
    ("object-cases", "invalid baseline cases must be a non-empty array"),
    ("case", "invalid baseline cases[0] must be an object"),
    ("name", "invalid baseline cases[0].name must be a non-empty string"),
    ("blank-name", "invalid baseline cases[0].name must be a non-empty string"),
    ("list-name", "invalid baseline cases[0].name must be a non-empty string"),
    ("duplicate-name", "invalid baseline case name 'alpha' must be unique"),
    ("report", "baseline case 'alpha': report must be an object"),
    ("missing-report", "baseline case 'alpha': report must be an object"),
])
def test_comparison_rejects_malformed_case_structure(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest = _batch(tmp_path, (2, 4))
    saved = _save_report(tmp_path, manifest)
    payload = json.loads(saved.read_text())
    if mutation == "root":
        payload = []
    elif mutation == "missing-cases":
        del payload["cases"]
    elif mutation == "empty-cases":
        payload["cases"] = []
    elif mutation == "object-cases":
        payload["cases"] = {}
    elif mutation == "case":
        payload["cases"][0] = None
    elif mutation == "name":
        del payload["cases"][0]["name"]
    elif mutation == "blank-name":
        payload["cases"][0]["name"] = " "
    elif mutation == "list-name":
        payload["cases"][0]["name"] = []
    elif mutation == "duplicate-name":
        payload["cases"][1]["name"] = "alpha"
    elif mutation == "report":
        payload["cases"][0]["report"] = []
    elif mutation == "missing-report":
        del payload["cases"][0]["report"]
    saved.write_text(json.dumps(payload), encoding="utf-8")
    result = _run_cli("--benchmark", str(manifest), "--compare-baseline", str(saved))
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == f"error: {message}\n"


@pytest.mark.parametrize("options", [(), ("--optimize",)])
@pytest.mark.parametrize("metric", ["flops", "bytes_read", "bytes_written"])
@pytest.mark.parametrize("value", ["missing", None, True, -1, 1.5, "2", float("nan"), float("inf")])
def test_comparison_rejects_missing_or_malformed_counts(
    tmp_path: Path, options: tuple[str, ...], metric: str, value: object
) -> None:
    manifest = _batch(tmp_path, (2, 4))
    saved = _save_report(tmp_path, manifest, *options)
    payload = json.loads(saved.read_text())
    report = payload["cases"][0]["report"]
    totals = (report["optimized"] if options else report)["totals"]
    if value == "missing":
        del totals[metric]
    else:
        totals[metric] = value
    saved.write_text(json.dumps(payload), encoding="utf-8")
    result = _run_cli("--benchmark", str(manifest), *options, "--compare-baseline", str(saved))
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == f"error: baseline case 'alpha': totals.{metric} must be a non-negative integer\n"


@pytest.mark.parametrize(("field", "value"), [
    ("roofline", "missing"), ("roofline", None), ("roofline", []),
    ("estimated_time_us", "missing"), ("estimated_time_us", None),
    ("estimated_time_us", True), ("estimated_time_us", -1),
    ("estimated_time_us", "0.1"), ("estimated_time_us", float("nan")),
    ("estimated_time_us", float("inf")),
    pytest.param("estimated_time_us", 10**1000, id="oversized-time"),
])
def test_comparison_requires_valid_targeted_time(tmp_path: Path, field: str, value: object) -> None:
    manifest = _batch(tmp_path, (2, 4))
    _set_target(tmp_path, TARGET)
    saved = _save_report(tmp_path, manifest)
    payload = json.loads(saved.read_text())
    totals = payload["cases"][0]["report"]["totals"]
    container = totals if field == "roofline" else totals["roofline"]
    if value == "missing":
        del container[field]
    else:
        container[field] = value
    saved.write_text(json.dumps(payload), encoding="utf-8")
    result = _run_cli("--benchmark", str(manifest), "--compare-baseline", str(saved))
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == (
        "error: baseline case 'alpha': totals.roofline.estimated_time_us "
        "must be a finite non-negative number for a targeted case\n"
    )


@pytest.mark.parametrize("value", ["missing", [], {"estimated_time_us": 0.1}])
def test_comparison_requires_explicit_null_roofline_without_target(tmp_path: Path, value: object) -> None:
    manifest = _batch(tmp_path, (2, 4))
    saved = _save_report(tmp_path, manifest)
    payload = json.loads(saved.read_text())
    totals = payload["cases"][0]["report"]["totals"]
    if value == "missing":
        del totals["roofline"]
    else:
        totals["roofline"] = value
    saved.write_text(json.dumps(payload), encoding="utf-8")
    result = _run_cli("--benchmark", str(manifest), "--compare-baseline", str(saved))
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "error: baseline case 'alpha': totals.roofline must be null for a target-less case\n"


@pytest.mark.parametrize("options", [(), ("--optimize",)])
@pytest.mark.parametrize("value", ["missing", None, []])
def test_comparison_requires_totals_object(tmp_path: Path, options: tuple[str, ...], value: object) -> None:
    manifest = _batch(tmp_path, (2, 4))
    saved = _save_report(tmp_path, manifest, *options)
    payload = json.loads(saved.read_text())
    report = payload["cases"][0]["report"]
    selected = report["optimized"] if options else report
    if value == "missing":
        del selected["totals"]
    else:
        selected["totals"] = value
    saved.write_text(json.dumps(payload), encoding="utf-8")
    result = _run_cli("--benchmark", str(manifest), *options, "--compare-baseline", str(saved))
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "error: baseline case 'alpha': totals must be an object\n"


@pytest.mark.parametrize(("field", "value"), [
    ("baseline", "missing"), ("optimized", "missing"),
    ("baseline", None), ("optimized", []), ("totals", {}),
])
def test_comparison_rejects_ambiguous_or_incomplete_optimized_mode(
    tmp_path: Path, field: str, value: object
) -> None:
    manifest = _batch(tmp_path, (2, 4))
    saved = _save_report(tmp_path, manifest, "--optimize")
    payload = json.loads(saved.read_text())
    report = payload["cases"][0]["report"]
    if value == "missing":
        del report[field]
    else:
        report[field] = value
    saved.write_text(json.dumps(payload), encoding="utf-8")
    result = _run_cli("--benchmark", str(manifest), "--optimize", "--compare-baseline", str(saved))
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "error: baseline case 'alpha': incompatible report mode\n"


@pytest.mark.parametrize("location", ["root", "baseline", "optimized"])
@pytest.mark.parametrize("mutation", ["missing", "inconsistent"])
def test_comparison_rejects_missing_or_inconsistent_target_metadata(
    tmp_path: Path, location: str, mutation: str
) -> None:
    manifest = _batch(tmp_path, (2, 4))
    saved = _save_report(tmp_path, manifest, "--optimize")
    payload = json.loads(saved.read_text())
    report = payload["cases"][0]["report"]
    target_report = report if location == "root" else report[location]
    if mutation == "missing":
        del target_report["target"]
        field = "target" if location == "root" else f"{location}.target"
        message = f"{field} must be present"
    else:
        target_report["target"] = TARGET
        message = "inconsistent target ceilings"
    saved.write_text(json.dumps(payload), encoding="utf-8")
    result = _run_cli("--benchmark", str(manifest), "--optimize", "--compare-baseline", str(saved))
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == f"error: baseline case 'alpha': {message}\n"


@pytest.mark.parametrize("field", ["peak_compute_gflops", "memory_bandwidth_gbps"])
def test_comparison_does_not_treat_boolean_target_as_numeric_one(tmp_path: Path, field: str) -> None:
    manifest = _batch(tmp_path, (2, 4))
    _set_target(tmp_path, {"peak_compute_gflops": 1, "memory_bandwidth_gbps": 1})
    saved = _save_report(tmp_path, manifest)
    payload = json.loads(saved.read_text())
    payload["cases"][0]["report"]["target"][field] = True
    saved.write_text(json.dumps(payload), encoding="utf-8")
    result = _run_cli("--benchmark", str(manifest), "--compare-baseline", str(saved))
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == f"error: baseline case 'alpha': target.{field} must be a positive finite number\n"


@pytest.mark.parametrize("options", [(), ("--optimize",)])
@pytest.mark.parametrize("sizes", [(2, 4), (1, 3)])
def test_equality_or_improvement_passes_independent_of_order_and_paths(
    tmp_path: Path, options: tuple[str, ...], sizes: tuple[int, int]
) -> None:
    manifest = _batch(tmp_path, (2, 4))
    # Mixed targets: compare beta's available time even though summary time is null.
    beta_path = tmp_path / "beta.json"
    beta = json.loads(beta_path.read_text())
    beta["target"] = TARGET
    beta_path.write_text(json.dumps(beta), encoding="utf-8")
    saved = _save_report(tmp_path, manifest, *options)
    payload = json.loads(saved.read_text())
    payload["cases"].reverse()
    del payload["cases"][0]["path"]
    payload["cases"][1]["path"] = "another/location.json"
    saved.write_text(json.dumps(payload), encoding="utf-8")
    _batch(tmp_path, sizes)
    beta = json.loads(beta_path.read_text())
    beta["target"] = TARGET
    beta_path.write_text(json.dumps(beta), encoding="utf-8")
    manifest.write_text('["beta.json", "alpha.json"]', encoding="utf-8")
    args = ("--benchmark", str(manifest), *options, "--case", "alpha", "--case", "beta",
            "--case", "alpha", "--omit-paths", "--compact")
    plain = _run_cli(*args)
    first = _run_cli(*args, "--compare-baseline", str(saved))
    second = _run_cli(*args, "--compare-baseline", str(saved))
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    output = json.loads(first.stdout)
    comparison = output.pop("comparison")
    assert output == json.loads(plain.stdout)
    assert comparison["passed"] is True
    assert comparison["mode"] == ("optimized" if options else "baseline")
    assert [case["name"] for case in comparison["cases"]] == ["alpha", "beta"]
    assert all("path" not in case for case in output["cases"])
    assert output["summary"]["totals"]["estimated_time_us"] is None
    for case, old_size, size in zip(comparison["cases"], (2, 4), sizes):
        assert case["delta"]["flops"] == size - old_size
        assert case["delta"]["logical_bytes"] == (size - old_size) * 8
    assert comparison["cases"][0]["delta"]["estimated_time_us"] is None
    assert comparison["cases"][1]["delta"]["estimated_time_us"] == sizes[1] * 8 / 100000 - 0.00032


def test_selected_optimized_comparison_ignores_unselected_and_unoptimized_costs(tmp_path: Path) -> None:
    manifest = _batch(tmp_path, (2, 4))
    fusion: Any = _fusion_graph()
    fusion["target"] = TARGET
    graph_path = tmp_path / "beta.json"
    graph_path.write_text(json.dumps(fusion), encoding="utf-8")
    # Effective case name is the graph name, not the path stem.
    options = ("--case", "fusion_case", "--optimize", "--omit-paths")
    saved = _save_report(tmp_path, manifest, *options)
    fusion["ops"].append({"name": "dead", "op": "relu", "inputs": ["x"], "outputs": ["dead_result"]})
    graph_path.write_text(json.dumps(fusion), encoding="utf-8")
    result = _run_cli("--benchmark", str(manifest), *options, "--compare-baseline", str(saved))
    assert result.returncode == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["cases"][0]["report"]["baseline"]["totals"]["flops"] == 6
    assert payload["comparison"] == {
        "mode": "optimized", "passed": True, "cases": [{
            "name": "fusion_case",
            "baseline": {"flops": 4, "logical_bytes": 24, "estimated_time_us": 0.00024},
            "current": {"flops": 4, "logical_bytes": 24, "estimated_time_us": 0.00024},
            "delta": {"flops": 0, "logical_bytes": 0, "estimated_time_us": 0.0},
        }],
    }


@pytest.mark.parametrize("options", [(), ("--optimize",)])
def test_time_increase_alone_fails_with_case_diagnostic(tmp_path: Path, options: tuple[str, ...]) -> None:
    manifest = _batch(tmp_path, (2, 4))
    _set_target(tmp_path, TARGET)
    saved = _save_report(tmp_path, manifest, *options)
    payload = json.loads(saved.read_text())
    report = payload["cases"][0]["report"]
    selected = report["optimized"] if options else report
    selected["totals"]["roofline"]["estimated_time_us"] = 0.00008
    saved.write_text(json.dumps(payload), encoding="utf-8")
    result = _run_cli("--benchmark", str(manifest), *options, "--compare-baseline", str(saved))
    assert result.returncode == 2
    assert json.loads(result.stderr.removeprefix("error: ")) == {
        "error": "benchmark baseline exceeded", "violations": [{
            "case": "alpha", "metric": "estimated_time_us", "baseline": 0.00008,
            "current": 0.00016, "delta": 0.00008,
        }],
    }
    case = json.loads(result.stdout)["comparison"]["cases"][0]
    assert case["delta"] == {"flops": 0, "logical_bytes": 0, "estimated_time_us": 0.00008}



def test_large_integral_saved_time_keeps_one_microsecond_regression(tmp_path: Path) -> None:
    manifest = _batch(tmp_path, (2**53 + 4, 1))
    _set_target(tmp_path, {**TARGET, "memory_bandwidth_gbps": 0.008})
    saved = _save_report(tmp_path, manifest)
    payload = json.loads(saved.read_text())
    payload["cases"][0]["report"]["totals"]["roofline"]["estimated_time_us"] = 2**53 + 3
    saved.write_text(json.dumps(payload), encoding="utf-8")
    result = _run_cli("--benchmark", str(manifest), "--compare-baseline", str(saved))
    assert result.returncode == 2
    assert json.loads(result.stdout)["comparison"]["cases"][0]["delta"]["estimated_time_us"] == 1
    assert json.loads(result.stderr.removeprefix("error: "))["violations"][0]["case"] == "alpha"


@pytest.mark.parametrize(("sizes", "budget_flops", "errors"), [
    ((3, 3), 6, ["benchmark baseline exceeded"]),
    ((2, 4), 5, ["benchmark budget exceeded"]),
    ((3, 3), 5, ["benchmark budget exceeded", "benchmark baseline exceeded"]),
])
def test_budget_and_comparison_gates_are_both_enforced(
    tmp_path: Path, sizes: tuple[int, int], budget_flops: int, errors: list[str]
) -> None:
    manifest = _batch(tmp_path, (2, 4))
    saved = _save_report(tmp_path, manifest)
    _batch(tmp_path, sizes)
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps({"aggregate": {"flops": budget_flops}}), encoding="utf-8")
    result = _run_cli(
        "--benchmark", str(manifest), "--compare-baseline", str(saved),
        "--budget", str(budget), "--omit-paths", "--compact",
    )
    assert result.returncode == 2
    assert [json.loads(line.removeprefix("error: "))["error"] for line in result.stderr.splitlines()] == errors
    payload = json.loads(result.stdout)
    assert payload["comparison"]["passed"] == (sizes == (2, 4))
    assert all("path" not in case for case in payload["cases"])


def test_explicit_case_names_not_graph_labels_define_comparison_identity(tmp_path: Path) -> None:
    manifest = _batch(tmp_path, (2, 4))
    manifest.write_text(json.dumps([
        {"name": "named_beta", "path": "beta.json"},
        {"name": "named_alpha", "path": "alpha.json"},
    ]), encoding="utf-8")
    saved = _save_report(tmp_path, manifest)
    for name in ("alpha", "beta"):
        graph_path = tmp_path / f"{name}.json"
        graph = json.loads(graph_path.read_text())
        graph["name"] = "changed_label"
        graph_path.write_text(json.dumps(graph), encoding="utf-8")
    result = _run_cli("--benchmark", str(manifest), "--compare-baseline", str(saved))
    assert result.returncode == 0
    assert result.stderr == ""
    comparison = json.loads(result.stdout)["comparison"]
    assert comparison["passed"] is True
    assert [case["name"] for case in comparison["cases"]] == ["named_alpha", "named_beta"]


def test_saved_report_rejects_nested_duplicate_metric_keys(tmp_path: Path) -> None:
    manifest = _batch(tmp_path, (2, 4))
    saved = _save_report(tmp_path, manifest)
    saved.write_text(saved.read_text().replace('"flops":2', '"flops":2,"flops":2', 1), encoding="utf-8")
    result = _run_cli("--benchmark", str(manifest), "--compare-baseline", str(saved))
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "error: invalid baseline JSON: duplicate object key 'flops'\n"
