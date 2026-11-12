"""Per-case regression comparison for saved benchmark reports."""

from __future__ import annotations

import json
from fractions import Fraction
from math import isfinite
from pathlib import Path
from typing import Any

from .parser import GraphValidationError, _parse_target, _unique_json_object


BENCHMARK_SCHEMA_VERSION = 1
COST_MODEL_VERSION = 1


def _check_targets(report: dict[str, Any], *, optimize: bool, label: str) -> None:
    scopes = [("", report)]
    if optimize:
        scopes.extend((name + ".", report[name]) for name in ("baseline", "optimized"))
    for prefix, scope in scopes:
        if "target" not in scope:
            raise GraphValidationError(f"{label}: {prefix}target must be present")
        try:
            _parse_target(scope["target"])
        except GraphValidationError as exc:
            raise GraphValidationError(f"{label}: {prefix}{exc}") from exc
    if any(scope["target"] != report["target"] for _, scope in scopes):
        raise GraphValidationError(f"{label}: inconsistent target ceilings")


def _metrics(report: dict[str, Any], *, optimize: bool, label: str) -> dict[str, Any]:
    selected = report["optimized"] if optimize else report
    totals = selected.get("totals")
    if not isinstance(totals, dict):
        raise GraphValidationError(f"{label}: totals must be an object")
    for metric in ("flops", "bytes_read", "bytes_written"):
        value = totals.get(metric)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GraphValidationError(f"{label}: totals.{metric} must be a non-negative integer")
    roofline = totals.get("roofline")
    time = roofline.get("estimated_time_us") if isinstance(roofline, dict) else None
    if selected["target"] is not None:
        valid_time = False
        if not isinstance(time, bool) and isinstance(time, (int, float)):
            try:
                valid_time = isfinite(time) and time >= 0
            except OverflowError:
                pass
        if not valid_time:
            raise GraphValidationError(
                f"{label}: totals.roofline.estimated_time_us must be a finite "
                "non-negative number for a targeted case"
            )
    elif "roofline" not in totals or roofline is not None:
        raise GraphValidationError(f"{label}: totals.roofline must be null for a target-less case")
    return {
        "flops": totals["flops"],
        "logical_bytes": totals["bytes_read"] + totals["bytes_written"],
        "estimated_time_us": time,
    }


def compare_baseline(
    current: dict[str, Any], baseline_path: Path, *, optimize: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare the consumed fields of the CLI's versioned batch schema.

    Schema/model versions, case identity, mode, targets, and selected totals
    establish compatibility; paths and descriptive/aggregate report fields are
    not comparison inputs.
    """
    try:
        saved = json.loads(
            baseline_path.read_text(encoding="utf-8"),
            object_pairs_hook=lambda pairs: _unique_json_object(pairs, label="baseline"),
        )
    except OSError as exc:
        raise GraphValidationError(f"invalid baseline: could not read file: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise GraphValidationError(
            f"invalid baseline UTF-8: byte {exc.start}: {exc.reason}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise GraphValidationError(
            f"invalid baseline JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
        ) from exc
    except GraphValidationError:
        # Preserve duplicate-key diagnostics from the object-pairs hook.
        raise
    except ValueError as exc:
        raise GraphValidationError(
            "invalid baseline JSON: numeric value exceeds decoder limits"
        ) from exc
    except RecursionError as exc:
        raise GraphValidationError(
            "invalid baseline JSON: nesting exceeds decoder limits"
        ) from exc
    if not isinstance(saved, dict):
        raise GraphValidationError("invalid baseline must contain a benchmark object")
    for field in ("benchmark_schema_version", "cost_model_version"):
        version = saved.get(field)
        expected = current[field]
        if type(version) is not int or version != expected:
            raise GraphValidationError(
                f"incompatible baseline {field}: expected integer {expected}"
            )
    raw_cases = saved.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise GraphValidationError("invalid baseline cases must be a non-empty array")
    baseline_cases: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(raw_cases):
        if not isinstance(case, dict):
            raise GraphValidationError(f"invalid baseline cases[{index}] must be an object")
        name = case.get("name")
        if not isinstance(name, str) or not name.strip():
            raise GraphValidationError(
                f"invalid baseline cases[{index}].name must be a non-empty string"
            )
        if name in baseline_cases:
            raise GraphValidationError(f"invalid baseline case name {name!r} must be unique")
        if not isinstance(case.get("report"), dict):
            raise GraphValidationError(f"baseline case {name!r}: report must be an object")
        baseline_cases[name] = case["report"]
    current_names = {case["name"] for case in current["cases"]}
    if current_names != baseline_cases.keys():
        missing = sorted(current_names - baseline_cases.keys())
        extra = sorted(baseline_cases.keys() - current_names)
        raise GraphValidationError(
            f"incompatible baseline case set: missing={missing!r}, extra={extra!r}"
        )
    cases = []
    violations = []
    for case in sorted(current["cases"], key=lambda case: case["name"]):
        name = case["name"]
        saved_report = baseline_cases[name]
        if optimize:
            valid_mode = (
                isinstance(saved_report.get("baseline"), dict)
                and isinstance(saved_report.get("optimized"), dict)
                and "totals" not in saved_report
            )
        else:
            valid_mode = not ({"baseline", "optimized"} & saved_report.keys())
        if not valid_mode:
            raise GraphValidationError(f"baseline case {name!r}: incompatible report mode")
        _check_targets(saved_report, optimize=optimize, label=f"baseline case {name!r}")
        if saved_report["target"] != case["report"]["target"]:
            raise GraphValidationError(f"baseline case {name!r}: incompatible target ceilings")
        baseline = _metrics(saved_report, optimize=optimize, label=f"baseline case {name!r}")
        actual = _metrics(case["report"], optimize=optimize, label=f"current case {name!r}")
        delta = {}
        for metric, value in baseline.items():
            if value is None:
                delta[metric] = None
            elif isinstance(value, int) and isinstance(actual[metric], float):
                # Avoid rounding an integral saved time before subtraction.
                delta[metric] = float(Fraction(actual[metric]) - value)
            else:
                delta[metric] = actual[metric] - value
        cases.append({"name": name, "baseline": baseline, "current": actual, "delta": delta})
        for metric, change in delta.items():
            if change is not None and actual[metric] > baseline[metric]:
                violations.append(
                    {
                        "case": name,
                        "metric": metric,
                        "baseline": baseline[metric],
                        "current": actual[metric],
                        "delta": change,
                    }
                )
    return {
        "mode": "optimized" if optimize else "baseline",
        "passed": not violations,
        "cases": cases,
    }, violations
