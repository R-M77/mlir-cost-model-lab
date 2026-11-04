from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import UserDict
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from cost_model import (
    LoweringValidationError,
    compare_matmul_lowering,
    compare_matmul_lowering_graph,
    parse_graph,
)
from cost_model.models import DTYPE_BYTES, Graph, Tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "examples" / "matmul_lowering.json"


def _fixture_payload() -> dict[str, Any]:
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


def test_zero_traffic_lowered_fma_is_compute_bound_with_null_intensity() -> None:
    report = compare_matmul_lowering(_fixture_payload())
    fma = next(
        operation for operation in report["after"]["ops"] if operation["op"] == "fma"
    )

    assert fma["flops"] == 48
    assert fma["bytes_read"] == fma["bytes_written"] == 0
    assert fma["arithmetic_intensity"] == 0.0
    assert fma["roofline"] == {
        "arithmetic_intensity": None,
        "ridge_point": pytest.approx(10.0),
        "bound": "compute",
        "estimated_time_us": pytest.approx(48 / (1000 * 1000)),
    }


def test_tiled_lowering_preserves_flops_and_models_tile_reuse() -> None:
    report = compare_matmul_lowering(
        _fixture_payload(), mode="tiled", tile_size=2, vector_width=2
    )

    assert report["before"]["totals"]["flops"] == 48
    assert report["after"]["totals"] == {
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
    assert report["after"]["operation_count"] == 32
    assert report["cost_delta"] == {
        "flops": 0,
        "bytes_read": 0,
        "bytes_written": 0,
        "logical_bytes": 0,
        "operation_count": 31,
        "roofline": {"estimated_time_us": pytest.approx(0.0)},
    }
    assert report["after"]["representation"]["kind"] == "tiled_vectorized"
    assert report["after"]["representation"]["tile_size"] == 2
    assert report["after"]["representation"]["vector_width"] == 2


def _compare_with_parameters(as_graph: bool, **kwargs: Any) -> dict[str, Any]:
    if as_graph:
        return compare_matmul_lowering_graph(
            parse_graph(_fixture_payload()), **kwargs
        )
    return compare_matmul_lowering(_fixture_payload(), **kwargs)


@pytest.mark.parametrize("as_graph", [False, True], ids=["payload", "graph"])
@pytest.mark.parametrize("mode", ["scalar", "tiled"])
@pytest.mark.parametrize("parameter", ["tile_size", "vector_width"])
@pytest.mark.parametrize("value", [True, False, 2.0, -1.5, "2", None, 0, -2])
def test_matmul_lowering_rejects_invalid_parameters(
    as_graph: bool, mode: str, parameter: str, value: Any
) -> None:
    with pytest.raises(LoweringValidationError, match=rf"^{parameter} must be a positive integer$"):
        _compare_with_parameters(as_graph, mode=mode, **{parameter: value})


@pytest.mark.parametrize("as_graph", [False, True], ids=["payload", "graph"])
@pytest.mark.parametrize("mode_kwargs", [{}, {"mode": "scalar"}], ids=["implicit", "explicit"])
@pytest.mark.parametrize(
    "parameters",
    [{"tile_size": 1}, {"vector_width": 1}, {"tile_size": 32, "vector_width": 8}],
)
def test_scalar_lowering_rejects_non_default_parameters_deterministically(
    as_graph: bool, mode_kwargs: dict[str, Any], parameters: dict[str, int]
) -> None:
    for _ in range(2):
        with pytest.raises(LoweringValidationError) as exc_info:
            _compare_with_parameters(as_graph, **mode_kwargs, **parameters)
        assert str(exc_info.value) == "non-default tile_size/vector_width require tiled lowering"


@pytest.mark.parametrize("as_graph", [False, True], ids=["payload", "graph"])
@pytest.mark.parametrize("mode_kwargs", [{}, {"mode": "scalar"}], ids=["implicit", "explicit"])
@pytest.mark.parametrize(
    "parameters",
    [{"tile_size": 16}, {"vector_width": 4}, {"tile_size": 16, "vector_width": 4}],
)
def test_scalar_lowering_accepts_explicit_default_parameters(
    as_graph: bool, mode_kwargs: dict[str, Any], parameters: dict[str, int]
) -> None:
    assert _compare_with_parameters(as_graph, **mode_kwargs, **parameters) == (
        compare_matmul_lowering(_fixture_payload())
    )


@pytest.mark.parametrize("as_graph", [False, True], ids=["payload", "graph"])
@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"tile_size": 1, "vector_width": False}, "vector_width must be a positive integer"),
        ({"tile_size": 0, "vector_width": 1}, "tile_size must be a positive integer"),
        ({"tile_size": 0, "vector_width": 0}, "tile_size must be a positive integer"),
    ],
)
def test_scalar_lowering_validates_positive_integers_before_mode(
    as_graph: bool, parameters: dict[str, Any], message: str
) -> None:
    with pytest.raises(LoweringValidationError) as exc_info:
        _compare_with_parameters(as_graph, mode="scalar", **parameters)
    assert str(exc_info.value) == message


@pytest.mark.parametrize("as_graph", [False, True], ids=["payload", "graph"])
@pytest.mark.parametrize("parameter", ["tile_size", "vector_width"])
def test_matmul_lowering_accepts_positive_integer_parameters(
    as_graph: bool, parameter: str
) -> None:
    report = _compare_with_parameters(as_graph, mode="tiled", **{parameter: 1})
    assert report["after"]["representation"][parameter] == 1
    assert report["after"]["totals"]["flops"] == report["before"]["totals"]["flops"]


def _shaped_payload(m: int, k: int, n: int) -> dict[str, Any]:
    payload = _fixture_payload()
    payload["inputs"]["lhs"]["shape"] = [m, k]
    payload["inputs"]["rhs"]["shape"] = [k, n]
    return payload


def _tiled_counts(report: dict[str, Any]) -> dict[str, int]:
    return {op["name"]: op["count"] for op in report["after"]["ops"]}


def test_tiled_n_axis_vectors_and_scalar_tails_have_exact_counts_and_traffic() -> None:
    report = compare_matmul_lowering(
        _shaped_payload(3, 5, 7), mode="tiled", tile_size=4, vector_width=4
    )
    after = report["after"]
    assert set(report) == {"graph", "target", "before", "after", "cost_delta"}
    assert _tiled_counts(report) == {
        "init_accumulator_vectors": 3,
        "init_accumulator_scalar_tail": 9,
        "load_lhs_scalars": 15,
        "load_rhs_vectors": 5,
        "load_rhs_scalar_tail": 15,
        "vector_fma": 15,
        "scalar_tail_fma": 45,
        "store_output_vectors": 3,
        "store_output_scalar_tail": 9,
    }
    assert after["representation"]["body"] == [
        {"name": op["name"], "op": op["op"], "count": op["count"]}
        for op in after["ops"]
    ]
    assert after["representation"]["assumptions"]["vector_axis"] == "n"
    assert after["representation"]["assumptions"]["lhs_scalar_broadcast_across_vector_lanes"]
    assert after["representation"]["assumptions"]["incomplete_n_vectors_use_scalar_tail_operations"]
    ops = {op["name"]: op for op in after["ops"]}
    assert ops["vector_fma"]["inputs"] == ["lhs_scalar", "rhs_vector", "acc[m,n]"]
    assert ops["scalar_tail_fma"]["inputs"] == ["lhs_scalar", "rhs_scalar", "acc[m,n]"]
    assert {name: ops[name]["bytes_read"] for name in (
        "load_lhs_scalars", "load_rhs_vectors", "load_rhs_scalar_tail"
    )} == {"load_lhs_scalars": 60, "load_rhs_vectors": 80, "load_rhs_scalar_tail": 60}
    assert {name: ops[name]["flops"] for name in (
        "vector_fma", "scalar_tail_fma"
    )} == {"vector_fma": 120, "scalar_tail_fma": 90}
    assert {name: ops[name]["bytes_written"] for name in (
        "store_output_vectors", "store_output_scalar_tail"
    )} == {"store_output_vectors": 48, "store_output_scalar_tail": 36}
    assert after["totals"]["flops"] == 210
    assert after["totals"]["bytes_read"] == 200
    assert after["totals"]["bytes_written"] == 84
    assert after["operation_count"] == 119
    assert report["cost_delta"]["flops"] == 0
    assert report["cost_delta"]["bytes_read"] == 0
    assert report["cost_delta"]["bytes_written"] == 0
    assert report["cost_delta"]["operation_count"] == 118


@pytest.mark.parametrize(
    ("shape", "tile_size", "vector_width", "expected"),
    [
        ((3, 5, 7), 4, 1, (21, 0, 15, 35, 0, 105, 0, 21, 0)),
        ((3, 5, 7), 4, 8, (0, 21, 15, 0, 35, 0, 105, 0, 21)),
        ((5, 3, 9), 4, 3, (10, 15, 15, 6, 9, 30, 45, 10, 15)),
    ],
)
def test_tiled_lane_boundaries(
    shape: tuple[int, int, int], tile_size: int, vector_width: int,
    expected: tuple[int, ...],
) -> None:
    m, k, n = shape
    report = compare_matmul_lowering(
        _shaped_payload(m, k, n), mode="tiled",
        tile_size=tile_size, vector_width=vector_width,
    )
    assert tuple(_tiled_counts(report).values()) == expected
    assert report["after"]["totals"]["flops"] == 2 * m * k * n
    assert report["after"]["totals"]["bytes_read"] == (m * k + k * n) * 4
    assert report["after"]["totals"]["bytes_written"] == m * n * 4


def test_tiled_lowering_cli_accepts_mode_and_parameters() -> None:
    result = _run_cli(
        "--lower-matmul",
        "--lowering-mode",
        "tiled",
        "--tile-size",
        "2",
        "--vector-width",
        "2",
        "--compact",
        str(FIXTURE_PATH),
    )

    assert result.returncode == 0
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["after"]["representation"]["kind"] == "tiled_vectorized"
    assert report["after"]["totals"]["flops"] == report["before"]["totals"]["flops"]


def test_tiled_compact_cli_is_deterministic(tmp_path: Path) -> None:
    graph_path = tmp_path / "ragged_matmul.json"
    graph_path.write_text(json.dumps(_shaped_payload(3, 5, 7)), encoding="utf-8")
    args = (
        "--lower-matmul", "--lowering-mode", "tiled", "--tile-size", "4",
        "--vector-width", "4", "--compact", str(graph_path),
    )
    first = _run_cli(*args)
    second = _run_cli(*args)
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    assert json.loads(first.stdout)["after"]["representation"]["kind"] == "tiled_vectorized"
    assert _tiled_counts(json.loads(first.stdout))["scalar_tail_fma"] == 45


@pytest.mark.parametrize(
    "mode_args",
    [
        ("--lower-matmul",),
        ("--lower-matmul", "--lowering-mode", "scalar"),
        (),
        ("--optimize",),
        ("--emit-mlir",),
        ("--benchmark",),
    ],
    ids=["implicit-scalar", "explicit-scalar", "estimate", "optimize", "mlir", "benchmark"],
)
@pytest.mark.parametrize(
    "parameters",
    [("--tile-size", "1"), ("--vector-width", "1"),
     ("--tile-size", "32", "--vector-width", "8")],
)
def test_lowering_cli_rejects_non_tiled_parameters_deterministically(
    mode_args: tuple[str, ...], parameters: tuple[str, ...]
) -> None:
    path = (
        REPO_ROOT / "benchmarks" / "target_bounds.json"
        if "--benchmark" in mode_args else FIXTURE_PATH
    )
    args = (*parameters, *mode_args, str(path))
    first = _run_cli(*args)
    second = _run_cli(*args)

    assert first.returncode == second.returncode == 2
    assert first.stdout == second.stdout == ""
    assert first.stderr == second.stderr == (
        "error: non-default tile_size/vector_width require tiled lowering\n"
    )


@pytest.mark.parametrize("mode_args", [(), ("--lowering-mode", "scalar")])
@pytest.mark.parametrize(
    "parameters",
    [("--tile-size", "16"), ("--vector-width", "4"),
     ("--tile-size", "16", "--vector-width", "4")],
)
def test_scalar_lowering_cli_accepts_explicit_defaults(
    mode_args: tuple[str, ...], parameters: tuple[str, ...]
) -> None:
    baseline = _run_cli("--lower-matmul", "--compact", str(FIXTURE_PATH))
    explicit = _run_cli(
        "--lower-matmul", *mode_args, *parameters, "--compact", str(FIXTURE_PATH)
    )

    assert baseline.returncode == explicit.returncode == 0
    assert baseline.stderr == explicit.stderr == ""
    assert baseline.stdout == explicit.stdout


def test_lowering_mode_requires_lowering_flag() -> None:
    result = _run_cli("--lowering-mode", "tiled", str(FIXTURE_PATH))

    assert result.returncode == 2
    assert "--lowering-mode requires --lower-matmul" in result.stderr


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--tile-size", "0"),
        ("--tile-size", "-1"),
        ("--tile-size", "not-an-int"),
        ("--vector-width", "0"),
        ("--vector-width", "-1"),
        ("--vector-width", "not-an-int"),
    ],
)
@pytest.mark.parametrize("mode", ["scalar", "tiled"])
def test_lowering_cli_rejects_invalid_tile_parameters_with_status_two(
    option: str, value: str, mode: str
) -> None:
    result = _run_cli(
        "--lower-matmul", "--lowering-mode", mode, option, value, str(FIXTURE_PATH)
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "error:" in result.stderr
    if value != "not-an-int":
        assert result.stderr.endswith("error: --tile-size and --vector-width must be positive\n")


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


def _assert_direct_lowering_error(graph: Graph, message: str) -> None:
    for mode in ("scalar", "tiled"):
        for _ in range(2):
            with pytest.raises(LoweringValidationError) as exc_info:
                compare_matmul_lowering_graph(graph, mode=mode)
            assert str(exc_info.value) == f"matmul lowering exercise {message}"


@pytest.mark.parametrize(
    "operations", [None, 7, "x", {}, {0: None}, {"op"}, (), [], (None, None)]
)
def test_lowering_rejects_direct_operation_containers(operations: Any) -> None:
    graph = replace(parse_graph(_fixture_payload()), operations=operations)
    _assert_direct_lowering_error(graph, "requires exactly one operation")


@pytest.mark.parametrize("field", ["operations", "values", "outputs"])
def test_lowering_rejects_missing_direct_graph_fields(field: str) -> None:
    graph = deepcopy(parse_graph(_fixture_payload()))
    object.__delattr__(graph, field)
    messages = {
        "operations": "requires exactly one operation",
        "values": "requires graph values to be a mapping",
        "outputs": "requires the matmul result to be the only graph output",
    }
    _assert_direct_lowering_error(graph, messages[field])


@pytest.mark.parametrize("definition", [None, 7, "matmul", {}, object()])
def test_lowering_rejects_non_operation_definitions(definition: Any) -> None:
    graph = replace(parse_graph(_fixture_payload()), operations=(definition,))
    _assert_direct_lowering_error(graph, "requires an Operation definition")


@pytest.mark.parametrize("field", ["name", "kind", "inputs", "output"])
def test_lowering_rejects_incomplete_operation_definitions(field: str) -> None:
    graph = deepcopy(parse_graph(_fixture_payload()))
    object.__delattr__(graph.operations[0], field)
    _assert_direct_lowering_error(graph, "requires an Operation definition")


@pytest.mark.parametrize("kind", [None, 7, True, [], {}, "", "MATMUL", "relu"])
def test_lowering_rejects_invalid_direct_matmul_kind(kind: Any) -> None:
    graph = parse_graph(_fixture_payload())
    graph = replace(graph, operations=(replace(graph.operations[0], kind=kind),))
    _assert_direct_lowering_error(graph, "requires a single matmul operation")


@pytest.mark.parametrize("name", [None, 7, True, [], {}, "", " \t\n"])
def test_lowering_rejects_invalid_direct_operation_names(name: Any) -> None:
    graph = parse_graph(_fixture_payload())
    graph = replace(graph, operations=(replace(graph.operations[0], name=name),))
    _assert_direct_lowering_error(graph, "requires a non-empty string operation name")


@pytest.mark.parametrize(
    "inputs", [None, 7, "lr", {}, {0: "lhs", 1: "rhs"}, {"lhs", "rhs"}, (), [],
               ("lhs",), ["lhs"], ("lhs", "rhs", "lhs")]
)
def test_lowering_rejects_malformed_direct_operand_containers(inputs: Any) -> None:
    graph = parse_graph(_fixture_payload())
    graph = replace(graph, operations=(replace(graph.operations[0], inputs=inputs),))
    _assert_direct_lowering_error(graph, "requires exactly two matmul inputs")


@pytest.mark.parametrize("operand_index", [0, 1])
@pytest.mark.parametrize("name", ["", " \t\n", None, 7, True, ("lhs",), ["lhs"], {}])
def test_lowering_rejects_invalid_direct_operand_names(name: Any, operand_index: int) -> None:
    graph = parse_graph(_fixture_payload())
    inputs = list(graph.operations[0].inputs)
    inputs[operand_index] = name
    graph = replace(graph, operations=(replace(graph.operations[0], inputs=tuple(inputs)),))
    _assert_direct_lowering_error(graph, "requires non-empty string matmul input names")


@pytest.mark.parametrize("name", ["", " \t\n", None, 7, True, ("output",), ["output"], {}])
def test_lowering_rejects_invalid_direct_result_names(name: Any) -> None:
    graph = parse_graph(_fixture_payload())
    graph = replace(
        graph, operations=(replace(graph.operations[0], output=name),), outputs=(name,)
    )
    _assert_direct_lowering_error(graph, "requires a non-empty string matmul result name")


@pytest.mark.parametrize(
    "outputs", [(), ("output", "output"), ("output", "lhs"), ("lhs",),
                ["output"], "output", None, 7, (["output"],)]
)
def test_lowering_preserves_direct_graph_output_errors(outputs: Any) -> None:
    graph = replace(parse_graph(_fixture_payload()), outputs=outputs)
    _assert_direct_lowering_error(
        graph, "requires the matmul result to be the only graph output"
    )


@pytest.mark.parametrize("values", [None, 7, "values", [], (), {"lhs", "rhs"}])
def test_lowering_rejects_direct_value_containers(values: Any) -> None:
    graph = replace(parse_graph(_fixture_payload()), values=values)
    _assert_direct_lowering_error(graph, "requires graph values to be a mapping")


@pytest.mark.parametrize("value_name", ["lhs", "rhs", "output"])
def test_lowering_rejects_unknown_direct_value_references(value_name: str) -> None:
    graph = parse_graph(_fixture_payload())
    del graph.values[value_name]
    _assert_direct_lowering_error(graph, f"references unknown value {value_name!r}")


@pytest.mark.parametrize("value_name", ["", " \t\n", None, 7, True, ("extra",)])
def test_lowering_rejects_invalid_direct_value_keys(value_name: Any) -> None:
    graph = parse_graph(_fixture_payload())
    graph.values[value_name] = Tensor(value_name, (), "i32")
    _assert_direct_lowering_error(graph, "requires non-empty string graph value names")


@pytest.mark.parametrize("value_name", ["lhs", "rhs", "output", "extra"])
@pytest.mark.parametrize("definition", [None, 7, "tensor", {}, [2, 3], object()])
def test_lowering_rejects_non_tensor_value_definitions(value_name: str, definition: Any) -> None:
    graph = parse_graph(_fixture_payload())
    graph.values[value_name] = definition
    _assert_direct_lowering_error(graph, f"graph value {value_name!r} requires a Tensor definition")


@pytest.mark.parametrize("value_name", ["lhs", "rhs", "output", "extra"])
@pytest.mark.parametrize("field", ["name", "shape", "dtype"])
def test_lowering_rejects_incomplete_tensor_definitions(value_name: str, field: str) -> None:
    graph = deepcopy(parse_graph(_fixture_payload()))
    if value_name == "extra":
        graph.values[value_name] = Tensor(value_name, (), "i32")
    object.__delattr__(graph.values[value_name], field)
    _assert_direct_lowering_error(graph, f"graph value {value_name!r} requires a Tensor definition")


@pytest.mark.parametrize("value_name", ["lhs", "rhs", "output", "extra"])
@pytest.mark.parametrize("name", ["stale", "", None, [], {}])
def test_lowering_rejects_value_key_tensor_name_mismatches(value_name: str, name: Any) -> None:
    graph = parse_graph(_fixture_payload())
    graph.values[value_name] = Tensor(name, (2, 3), "f32")
    _assert_direct_lowering_error(
        graph, f"graph value {value_name!r} has mismatched tensor name {name!r}"
    )


@pytest.mark.parametrize("value_name", ["lhs", "rhs", "output", "extra"])
@pytest.mark.parametrize("shape", [None, 7, "2x3", {"shape": [2, 3]}, {2, 3}])
def test_lowering_rejects_malformed_direct_tensor_shapes(value_name: str, shape: Any) -> None:
    graph = parse_graph(_fixture_payload())
    graph.values[value_name] = Tensor(value_name, shape, "f32")
    _assert_direct_lowering_error(
        graph, f"graph value {value_name!r} requires a tuple or list shape"
    )


@pytest.mark.parametrize("value_name", ["lhs", "rhs", "output", "extra"])
@pytest.mark.parametrize("dimension", [0, -1, True, False, 2.0, "2", None, [2], {}])
def test_lowering_rejects_malformed_direct_tensor_dimensions(
    value_name: str, dimension: Any
) -> None:
    graph = parse_graph(_fixture_payload())
    graph.values[value_name] = Tensor(value_name, (dimension, 3), "f32")
    _assert_direct_lowering_error(
        graph, f"graph value {value_name!r} requires positive integer shape dimensions"
    )


@pytest.mark.parametrize("value_name", ["lhs", "rhs", "output", "extra"])
@pytest.mark.parametrize("dtype", [None, 7, True, [], {}, "", "unknown"])
def test_lowering_rejects_malformed_direct_tensor_dtypes(value_name: str, dtype: Any) -> None:
    graph = parse_graph(_fixture_payload())
    original = graph.values.get(value_name, Tensor(value_name, (), "i32"))
    graph.values[value_name] = replace(original, dtype=dtype)
    _assert_direct_lowering_error(
        graph, f"graph value {value_name!r} has unsupported dtype {dtype!r}"
    )


@pytest.mark.parametrize(
    ("value_name", "shape", "message"),
    [
        ("lhs", (), "requires rank-2 input tensors"),
        ("rhs", (3,), "requires rank-2 input tensors"),
        ("lhs", (1, 2, 3), "requires rank-2 input tensors"),
        ("rhs", (1, 3, 4), "requires rank-2 input tensors"),
        ("rhs", (4, 4), "requires aligned matmul dimensions"),
        ("output", (), "requires the inferred matmul result shape"),
        ("output", (2, 5), "requires the inferred matmul result shape"),
        ("output", (1, 2, 4), "requires the inferred matmul result shape"),
    ],
)
def test_lowering_validates_direct_matmul_shapes(
    value_name: str, shape: tuple[int, ...], message: str
) -> None:
    graph = parse_graph(_fixture_payload())
    graph.values[value_name] = replace(graph.values[value_name], shape=shape)
    _assert_direct_lowering_error(graph, message)


@pytest.mark.parametrize("value_name", ["lhs", "rhs", "output"])
def test_lowering_rejects_inconsistent_supported_dtypes(value_name: str) -> None:
    graph = parse_graph(_fixture_payload())
    graph.values[value_name] = replace(graph.values[value_name], dtype="f16")
    _assert_direct_lowering_error(graph, "requires matching input and output dtypes")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda g: replace(g, operations=(None, None), values=None),
         "requires exactly one operation"),
        (lambda g: replace(g, operations=(replace(g.operations[0], kind="relu", inputs=None),)),
         "requires a single matmul operation"),
        (lambda g: replace(g, operations=(replace(g.operations[0], output=[]),), outputs=()),
         "requires a non-empty string matmul result name"),
        (lambda g: replace(g, outputs=(), values=None),
         "requires the matmul result to be the only graph output"),
        (lambda g: replace(g, operations=(replace(g.operations[0], inputs=None),), values=None),
         "requires exactly two matmul inputs"),
        (lambda g: replace(g, values={}), "references unknown value 'lhs'"),
        (lambda g: replace(g, values={"lhs": None}), "references unknown value 'rhs'"),
        (lambda g: replace(g, values={"lhs": None, "rhs": None}),
         "references unknown value 'output'"),
        (lambda g: replace(g, values={"lhs": None, "rhs": None, "output": None}),
         "graph value 'lhs' requires a Tensor definition"),
        (lambda g: replace(g, values={
            **g.values, "lhs": replace(g.values["lhs"], shape=(1, 2, 3), dtype=[])
        }),
         "requires rank-2 input tensors"),
    ],
)
def test_lowering_direct_validation_precedence(mutation: Any, message: str) -> None:
    graph = mutation(parse_graph(_fixture_payload()))
    _assert_direct_lowering_error(graph, message)
    # Graph diagnostics must also precede unrelated lowering-option validation.
    with pytest.raises(LoweringValidationError) as exc_info:
        compare_matmul_lowering_graph(graph, mode="invalid", tile_size=0)
    assert str(exc_info.value) == f"matmul lowering exercise {message}"


@pytest.mark.parametrize("mode", ["scalar", "tiled"])
@pytest.mark.parametrize("container", [tuple, list])
@pytest.mark.parametrize("mapping", [dict, MappingProxyType, UserDict])
@pytest.mark.parametrize("repeated_operand", [False, True])
def test_lowering_preserves_valid_direct_containers(
    mode: str, container: Any, mapping: Any, repeated_operand: bool
) -> None:
    payload = _fixture_payload()
    if repeated_operand:
        payload["inputs"] = {"lhs": {"shape": [2, 2], "dtype": "f32"}}
        payload["ops"][0]["inputs"] = ["lhs", "lhs"]
    # Lowering, unlike textual emission, permits unused graph inputs.
    payload["inputs"]["unused"] = {"shape": [], "dtype": "i32"}
    expected = compare_matmul_lowering(payload, mode=mode)
    graph = parse_graph(payload)
    graph.values["extra"] = Tensor("extra", (), "bool")
    graph = replace(
        graph,
        operations=container([replace(graph.operations[0], inputs=container(graph.operations[0].inputs))]),
        inputs=mapping(dict(reversed(list(graph.inputs.items())))),
        values=mapping({
            name: replace(tensor, shape=container(tensor.shape))
            for name, tensor in reversed(list(graph.values.items()))
        }),
    )
    assert compare_matmul_lowering_graph(graph, mode=mode) == expected
    assert compare_matmul_lowering_graph(graph, mode=mode) == expected


@pytest.mark.parametrize("mode", ["scalar", "tiled"])
@pytest.mark.parametrize("dtype", sorted(DTYPE_BYTES))
def test_lowering_preserves_all_supported_matmul_dtypes(mode: str, dtype: str) -> None:
    payload = _fixture_payload()
    for spec in payload["inputs"].values():
        spec["dtype"] = dtype
    graph = parse_graph(payload)
    original = deepcopy(graph)
    report = compare_matmul_lowering_graph(graph, mode=mode)
    assert graph == original
    assert report == compare_matmul_lowering(payload, mode=mode)
    assert report["before"]["totals"]["flops"] == report["after"]["totals"]["flops"] == 48
    assert report["after"]["ops"][0]["dtype"] == dtype


@pytest.mark.parametrize("name", ["renamed", " label % ! \n"])
def test_lowering_preserves_direct_labels_and_result_names(name: str) -> None:
    payload = _fixture_payload()
    payload["name"] = payload["ops"][0]["name"] = name
    payload["ops"][0]["outputs"] = payload["outputs"] = [name]
    graph = parse_graph(payload)
    assert compare_matmul_lowering_graph(graph) == compare_matmul_lowering(payload)
    assert compare_matmul_lowering_graph(graph)["after"]["representation"]["source_operation"] == name


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
