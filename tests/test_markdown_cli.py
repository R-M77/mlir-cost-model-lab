"""Exercise Markdown through the real module CLI, not a mocked formatter."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "benchmarks/target_bounds.json"
BASELINE = "benchmarks/target_bounds.expected.json"


def run_cli(*options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cost_model", *options],
        cwd=ROOT, capture_output=True, text=True, timeout=30, check=False,
    )


def test_markdown_normal_batch_is_a_deterministic_cost_estimate(tmp_path: Path) -> None:
    result = run_cli("--benchmark", MANIFEST, "--format", "markdown")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.startswith("# Benchmark cost estimates\n")
    assert "Mode: normal | Benchmark schema: 2 | Cost model: 1" in result.stdout
    assert "not measured runtime" in result.stdout
    assert "| Case | FLOPs | Logical bytes | estimated_time_us |" in result.stdout
    expected = json.loads((ROOT / BASELINE).read_text())
    for case in expected["cases"]:
        totals = case["report"]["totals"]
        row = f'| {case["name"].replace("_", "&#95;")} | {totals["flops"]} | {totals["bytes_read"] + totals["bytes_written"]} | {totals["roofline"]["estimated_time_us"]} |'
        assert row in result.stdout
        assert case["path"] not in result.stdout
    assert result.stdout.index("| compute&#95;bound&#95;matmul |") < result.stdout.index("| memory&#95;bound&#95;relu |")
    artifact = tmp_path / "summary.md"
    artifact.write_text(result.stdout, encoding="utf-8")
    assert artifact.read_text() == run_cli("--benchmark", MANIFEST, "--format", "markdown").stdout


def write_batch(tmp_path: Path, *, name: str = "fusion", targeted: bool = True) -> Path:
    graph = json.loads((ROOT / "examples/matmul_bias_relu.json").read_text())
    if not targeted:
        graph.pop("target", None)
    (tmp_path / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    manifest = tmp_path / "batch.json"
    manifest.write_text(json.dumps([{"name": name, "path": "graph.json"}]), encoding="utf-8")
    return manifest


def test_markdown_optimized_uses_selected_costs(tmp_path: Path) -> None:
    manifest = write_batch(tmp_path)
    result = run_cli("--benchmark", str(manifest), "--optimize", "--format", "markdown")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert "Mode: optimized" in result.stdout
    assert "| fusion | 64 | 184 | 0.00184 |" in result.stdout


@pytest.mark.parametrize("options", [(), ("--optimize",)])
def test_markdown_marks_missing_time_unavailable(tmp_path: Path, options: tuple[str, ...]) -> None:
    manifest = write_batch(tmp_path, targeted=False)
    result = run_cli("--benchmark", str(manifest), "--format", "markdown", *options)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    logical_bytes = 184 if options else 248
    assert f"| fusion | 64 | {logical_bytes} | unavailable |" in result.stdout


@pytest.mark.parametrize("options,metric,adjustment,status", [
    ((), "flops", 0, "PASS"),
    (("--optimize",), "flops", 1, "PASS"),
    ((), "flops", -1, "FAIL"),
    (("--optimize",), "bytes_read", -1, "FAIL"),
    ((), "estimated_time_us", -1e-18, "FAIL"),
])
def test_markdown_baseline_preserves_gate_and_per_case_results(
    tmp_path: Path, options: tuple[str, ...], metric: str, adjustment: float, status: str,
) -> None:
    manifest = write_batch(tmp_path)
    baseline_result = run_cli("--benchmark", str(manifest), *options)
    assert baseline_result.returncode == 0, baseline_result.stderr
    saved = json.loads(baseline_result.stdout)
    selected = saved["cases"][0]["report"]
    if options:
        selected = selected["optimized"]
    totals = selected["totals"]
    scope = totals["roofline"] if metric == "estimated_time_us" else totals
    scope[metric] += adjustment
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(saved), encoding="utf-8")
    args = ("--benchmark", str(manifest), *options, "--compare-baseline", str(baseline))
    result = run_cli(*args, "--format", "markdown")
    json_result = run_cli(*args)
    assert result.returncode == json_result.returncode == (2 if status == "FAIL" else 0)
    assert result.stderr == json_result.stderr
    assert f"## Baseline comparison: {status}" in result.stdout
    assert f"Gates: {status}" in result.stdout
    assert f"| fusion | {status} | flops |" in result.stdout
    assert f"| fusion | {status} | logical_bytes |" in result.stdout
    assert f"| fusion | {status} | estimated_time_us |" in result.stdout
    assert "Delta = current - baseline; positive is worse." in result.stdout
    if status == "FAIL":
        assert "benchmark baseline exceeded" in result.stderr


@pytest.mark.parametrize("baseline_status,budget_status", [
    (None, "PASS"), (None, "FAIL"),
    ("PASS", "PASS"), ("PASS", "FAIL"), ("FAIL", "PASS"), ("FAIL", "FAIL"),
])
def test_markdown_budget_and_combined_status_are_truthful(
    tmp_path: Path, baseline_status: str | None, budget_status: str,
) -> None:
    manifest = write_batch(tmp_path)
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps({"aggregate": {"flops": 64 if budget_status == "PASS" else 63}}))
    args = ["--benchmark", str(manifest), "--budget", str(budget)]
    if baseline_status is not None:
        saved_result = run_cli("--benchmark", str(manifest))
        assert saved_result.returncode == 0, saved_result.stderr
        saved = json.loads(saved_result.stdout)
        if baseline_status == "FAIL":
            saved["cases"][0]["report"]["totals"]["flops"] -= 1
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps(saved))
        args.extend(["--compare-baseline", str(baseline)])
    result = run_cli(*args, "--format", "markdown")
    json_result = run_cli(*args)
    failed = "FAIL" in (baseline_status, budget_status)
    assert result.returncode == json_result.returncode == (2 if failed else 0)
    assert result.stderr == json_result.stderr
    assert f"Gates: {'FAIL' if failed else 'PASS'}" in result.stdout
    assert f"## Budget: {budget_status}" in result.stdout
    if baseline_status is not None:
        assert f"## Baseline comparison: {baseline_status}" in result.stdout
    if budget_status == "FAIL":
        assert "| aggregate | flops | 64 | 63 |" in result.stdout
        assert "benchmark budget exceeded" in result.stderr
    if baseline_status == "FAIL" and budget_status == "FAIL":
        assert len(result.stderr.splitlines()) == 2


@pytest.mark.parametrize("options,diagnostic", [
    ((), "--format markdown requires --benchmark"),
    (("examples/matmul_bias_relu.json",), "--format markdown requires --benchmark"),
    (("--benchmark", MANIFEST, "--compact"), "--format markdown cannot be combined with --compact"),
    (("--benchmark", MANIFEST, "--emit-mlir"), "--emit-mlir"),
    (("--benchmark", MANIFEST, "--lower-matmul"), "--lower-matmul"),
    (("--benchmark", MANIFEST, "examples/matmul_bias_relu.json"), "mutually exclusive"),
    (("--benchmark", MANIFEST, "--lowering-mode", "tiled"), "--lowering-mode"),
    (("--benchmark", MANIFEST, "--tile-size", "8"), "require tiled lowering"),
])
def test_markdown_rejects_incompatible_flags_cleanly(options: tuple[str, ...], diagnostic: str) -> None:
    result = run_cli(*options, "--format", "markdown")
    assert result.returncode == 2
    assert result.stdout == ""
    assert diagnostic in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("incompatibility,diagnostic", [
    ("fingerprint", "incompatible graph content"),
    ("mode", "incompatible report mode"),
    ("schema", "incompatible baseline benchmark_schema_version"),
    ("target", "incompatible target ceilings"),
])
def test_markdown_comparison_preflight_errors_emit_no_report(
    tmp_path: Path, incompatibility: str, diagnostic: str,
) -> None:
    options = ("--optimize",) if incompatibility == "mode" else ()
    saved_result = run_cli("--benchmark", MANIFEST, *options)
    assert saved_result.returncode == 0, saved_result.stderr
    saved = json.loads(saved_result.stdout)
    if incompatibility == "fingerprint":
        saved["cases"][0]["graph_fingerprint"] = "sha256:" + "0" * 64
    elif incompatibility == "schema":
        saved["benchmark_schema_version"] = 1
    elif incompatibility == "target":
        saved["cases"][0]["report"]["target"]["peak_compute_gflops"] *= 2
    baseline = tmp_path / "incompatible.json"
    baseline.write_text(json.dumps(saved))
    args = ("--benchmark", MANIFEST, "--compare-baseline", str(baseline))
    result = run_cli(*args, "--format", "markdown")
    json_result = run_cli(*args)
    assert result.returncode == json_result.returncode == 2
    assert result.stdout == json_result.stdout == ""
    assert result.stderr == json_result.stderr
    assert diagnostic in result.stderr
    assert "Traceback" not in result.stderr


def test_default_json_bytes_and_budget_fixture_stay_exact() -> None:
    expected = (ROOT / BASELINE).read_text(encoding="utf-8")
    for options in [(), ("--format", "json"), ("--budget", "benchmarks/target_bounds.budget.json")]:
        result = run_cli("--benchmark", MANIFEST, "--compact", *options)
        assert result.returncode == 0, result.stderr
        assert result.stderr == ""
        assert result.stdout == expected
    readable = run_cli("--benchmark", MANIFEST)
    assert readable.returncode == 0, readable.stderr
    assert readable.stdout == json.dumps(json.loads(expected), indent=2) + "\n"


@pytest.mark.parametrize("name,safe", [
    ("_PASS_", "&#95;PASS&#95;"),
    ("__FAIL__", "&#95;&#95;FAIL&#95;&#95;"),
    ("www.example.com", "www&#46;example&#46;com"),
])
def test_markdown_literal_labels_keep_raw_selection_and_comparison(
    tmp_path: Path, name: str, safe: str,
) -> None:
    manifest = write_batch(tmp_path, name=name)
    before = manifest.read_bytes()
    raw = run_cli("--benchmark", str(manifest), "--case", name)
    assert raw.returncode == 0, raw.stderr
    assert json.loads(raw.stdout)["cases"][0]["name"] == name
    baseline = tmp_path / "baseline.json"
    baseline.write_text(raw.stdout)
    result = run_cli(
        "--benchmark", str(manifest), "--case", name,
        "--compare-baseline", str(baseline), "--format", "markdown",
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.count(f"| {safe} |") == 4
    assert f"| {safe} | PASS | flops | 64 | 64 | 0 |" in result.stdout
    assert "Gates: PASS" in result.stdout
    assert baseline.read_text() == raw.stdout
    assert manifest.read_bytes() == before


def test_markdown_escapes_names_without_changing_selection_or_baseline(tmp_path: Path) -> None:
    name = "x|`<b>&\\[link](url)\r\n::error::\x1b"
    manifest = write_batch(tmp_path, name=name, targeted=False)
    before = manifest.read_bytes()
    raw = run_cli("--benchmark", str(manifest))
    assert raw.returncode == 0, raw.stderr
    assert json.loads(raw.stdout)["cases"][0]["name"] == name
    baseline = tmp_path / "baseline.json"
    baseline.write_text(raw.stdout)
    args = ("--benchmark", str(manifest), "--case", name, "--compare-baseline", str(baseline), "--format", "markdown")
    result = run_cli(*args)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    safe = "x&#124;&#96;&lt;b&gt;&amp;&#92;&#91;link&#93;&#40;url&#41;&#92;u000d&#92;u000a&#58;&#58;error&#58;&#58;&#92;u001b"
    assert result.stdout.count(f"| {safe} |") == 4
    assert f"| {safe} | PASS | estimated_time_us | unavailable | unavailable | unavailable |" in result.stdout
    assert "Gates: PASS" in result.stdout
    assert raw.stdout == baseline.read_text()
    assert manifest.read_bytes() == before
    assert result.stdout == run_cli(*args, "--omit-paths").stdout


@pytest.mark.parametrize("options,actual", [((), 248), (("--optimize",), 184)])
def test_markdown_per_case_budget_failure_cannot_look_green(
    tmp_path: Path, options: tuple[str, ...], actual: int,
) -> None:
    manifest = write_batch(tmp_path)
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps({"aggregate": {"flops": 64}, "cases": {"fusion": {"logical_bytes": actual - 1}}}))
    result = run_cli("--benchmark", str(manifest), *options, "--budget", str(budget), "--format", "markdown")
    assert result.returncode == 2
    assert "Gates: FAIL" in result.stdout
    assert "## Budget: FAIL" in result.stdout
    assert f"| case | logical&#95;bytes | {actual} | {actual - 1} |" in result.stdout
    assert "benchmark budget exceeded" in result.stderr


def test_markdown_unavailable_budget_metric_fails_before_stdout(tmp_path: Path) -> None:
    manifest = write_batch(tmp_path, targeted=False)
    budget = tmp_path / "budget.json"
    budget.write_text('{"aggregate":{"estimated_time_us":1}}')
    result = run_cli("--benchmark", str(manifest), "--budget", str(budget), "--format", "markdown")
    assert result.returncode == 2
    assert result.stdout == ""
    assert "estimated_time_us is unavailable" in result.stderr
    assert "Traceback" not in result.stderr


def test_optimized_markdown_budget_passes_at_optimized_limits(tmp_path: Path) -> None:
    manifest = write_batch(tmp_path)
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps({"aggregate": {"logical_bytes": 184}, "cases": {"fusion": {"logical_bytes": 184}}}))
    result = run_cli("--benchmark", str(manifest), "--optimize", "--budget", str(budget), "--format", "markdown")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert "Gates: PASS" in result.stdout
    assert "## Budget: PASS" in result.stdout


def test_invalid_format_is_a_clean_usage_error() -> None:
    result = run_cli("--benchmark", MANIFEST, "--format", "yaml")
    assert result.returncode == 2
    assert result.stdout == ""
    assert "invalid choice" in result.stderr
    assert "Traceback" not in result.stderr
