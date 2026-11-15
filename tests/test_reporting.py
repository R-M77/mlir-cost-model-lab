"""Pure renderer contracts: presentation must not mutate computed reports."""
from __future__ import annotations

from copy import deepcopy
from html import unescape

import pytest

from cost_model.reporting import render_markdown


def batch(*names: str) -> dict:
    return {
        "benchmark_schema_version": 2,
        "cost_model_version": 1,
        "cases": [
            {"name": name, "path": "must-not-be-rendered.json", "report": {"totals": {
                "flops": 10**100 + 1, "bytes_read": 3, "bytes_written": 5,
                "roofline": {"estimated_time_us": 1e-200},
            }}} for name in names
        ],
    }


def test_names_cannot_inject_table_rows_html_links_or_controls() -> None:
    name = "a|`<script>&\\[x](url)\r\n::error::\x00\t\x7f\x85\u2028\u202e"
    report = batch("zeta", name, "beta")
    before = deepcopy(report)
    rendered = render_markdown(report, optimize=False)
    safe = (
        "a&#124;&#96;&lt;script&gt;&amp;&#92;&#91;x&#93;&#40;url&#41;"
        "&#92;u000d&#92;u000a&#58;&#58;error&#58;&#58;&#92;u0000"
        "&#92;u0009&#92;u007f&#92;u0085&#92;u2028&#92;u202e"
    )
    assert f"| {safe} |" in rendered
    assert rendered.index(f"| {safe} |") < rendered.index("| beta |") < rendered.index("| zeta |")
    assert len([line for line in rendered.splitlines() if line.startswith("| ")]) == 5
    assert not any(char in rendered for char in "<>`\r\x00\t\x7f\x85\u2028\u202e")
    assert "must-not-be-rendered.json" not in rendered
    assert str(10**100 + 1) in rendered
    assert "1e-200" in rendered
    assert report == before
    assert render_markdown(report, optimize=False) == rendered


@pytest.mark.parametrize("name,safe", [
    ("_PASS_", "&#95;PASS&#95;"),
    ("__FAIL__", "&#95;&#95;FAIL&#95;&#95;"),
    ("www.example.com", "www&#46;example&#46;com"),
])
def test_labels_escape_emphasis_and_bare_autolinks(name: str, safe: str) -> None:
    report = batch(name)
    before = deepcopy(report)
    violations = [{"scope": name, "metric": name, "actual": 2, "limit": 1}]
    rendered = render_markdown(report, optimize=False, budget_violations=violations)
    assert f"| {safe} | {10**100 + 1} | 8 | 1e-200 |" in rendered
    assert f"| {safe} | {safe} | 2 | 1 |" in rendered
    assert unescape(safe) == name
    assert report == before


def test_comparison_keeps_exact_integers_and_tiny_signed_deltas() -> None:
    report = batch("zeta", "alpha")
    report["comparison"] = {"mode": "baseline", "passed": False, "cases": [
        {"name": "zeta", "baseline": {"flops": 10**100, "logical_bytes": 8, "estimated_time_us": 0.0},
         "current": {"flops": 10**100 + 1, "logical_bytes": 8, "estimated_time_us": 1e-200},
         "delta": {"flops": 1, "logical_bytes": 0, "estimated_time_us": 1e-200}},
        {"name": "alpha", "baseline": {"flops": 2, "logical_bytes": 8, "estimated_time_us": 1e-200},
         "current": {"flops": 1, "logical_bytes": 8, "estimated_time_us": 0.0},
         "delta": {"flops": -1, "logical_bytes": 0, "estimated_time_us": -1e-200}},
    ]}
    before = deepcopy(report)
    rendered = render_markdown(report, optimize=False)
    assert f"| zeta | FAIL | flops | {10**100} | {10**100 + 1} | +1 |" in rendered
    assert "| zeta | FAIL | estimated_time_us | 0.0 | 1e-200 | +1e-200 |" in rendered
    assert "| alpha | PASS | estimated_time_us | 1e-200 | 0.0 | -1e-200 |" in rendered
    assert rendered.index("| alpha | PASS |") < rendered.index("| zeta | FAIL |")
    assert report == before
