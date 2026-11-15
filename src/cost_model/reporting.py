"""Pure presentation of computed benchmark cost estimates."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from string import punctuation
from typing import Any
from unicodedata import category


def _cell(text: str) -> str:
    """Keep labels on one line and escape HTML/Markdown punctuation."""
    escaped = []
    for char in text:
        if category(char).startswith("C") or char in "\u2028\u2029":
            escaped.append(f"&#92;u{ord(char):04x}")
        elif char in "<>&":
            escaped.append({"<": "&lt;", ">": "&gt;", "&": "&amp;"}[char])
        elif char in punctuation and char != "-":
            escaped.append(f"&#{ord(char)};")
        else:
            escaped.append(char)
    return "".join(escaped)


def _number(value: int | float | None, *, signed: bool = False) -> str:
    """Use exact integers and Python's round-trip float text, never fixed rounding."""
    if value is None:
        return "unavailable"
    return ("+" if signed and value > 0 else "") + str(value)


def render_markdown(
    report: Mapping[str, Any], *, optimize: bool,
    budget_violations: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Render computed costs/gates; None means no budget, an empty sequence PASS."""
    comparison = report.get("comparison")
    statuses = []
    if comparison is not None:
        statuses.append(comparison["passed"])
    if budget_violations is not None:
        statuses.append(not budget_violations)
    gates = ("PASS" if all(statuses) else "FAIL") if statuses else "not requested"
    lines = [
        "# Benchmark cost estimates",
        "",
        f"Mode: {'optimized' if optimize else 'normal'} | "
        f"Benchmark schema: {report['benchmark_schema_version']} | "
        f"Cost model: {report['cost_model_version']}",
        "",
        "Logical-model costs and idealized roofline estimates; not measured runtime.",
        "",
        f"Gates: {gates}",
        "",
        "| Case | FLOPs | Logical bytes | estimated_time_us |",
        "| --- | ---: | ---: | ---: |",
    ]
    for case in sorted(report["cases"], key=lambda case: case["name"]):
        selected = case["report"]["optimized"] if optimize else case["report"]
        totals = selected["totals"]
        time = (
            None if totals["roofline"] is None
            else totals["roofline"]["estimated_time_us"]
        )
        lines.append(
            f"| {_cell(case['name'])} | {totals['flops']} | "
            f"{totals['bytes_read'] + totals['bytes_written']} | {_number(time)} |"
        )
    if comparison is not None:
        lines.extend([
            "", f"## Baseline comparison: {'PASS' if comparison['passed'] else 'FAIL'}", "",
            "Delta = current - baseline; positive is worse.", "",
            "| Case | Result | Metric | Baseline | Current | Delta |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ])
        for case in sorted(comparison["cases"], key=lambda case: case["name"]):
            metrics = ("flops", "logical_bytes", "estimated_time_us")
            failed = any(
                case["current"][metric] is not None
                and case["current"][metric] > case["baseline"][metric]
                for metric in metrics
            )
            for metric in metrics:
                lines.append(
                    f"| {_cell(case['name'])} | {'FAIL' if failed else 'PASS'} | {metric} | "
                    f"{_number(case['baseline'][metric])} | {_number(case['current'][metric])} | "
                    f"{_number(case['delta'][metric], signed=True)} |"
                )
    if budget_violations is not None:
        lines.extend(["", f"## Budget: {'FAIL' if budget_violations else 'PASS'}"])
        if budget_violations:
            lines.extend([
                "", "| Scope | Metric | Actual | Limit |",
                "| --- | --- | ---: | ---: |",
            ])
            for violation in budget_violations:
                lines.append(
                    f"| {_cell(violation['scope'])} | {_cell(violation['metric'])} | "
                    f"{_number(violation['actual'])} | {_number(violation['limit'])} |"
                )
    return "\n".join(lines) + "\n"
