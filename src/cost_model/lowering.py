"""An opt-in, dependency-free matmul lowering exercise.

The lowering is intentionally a target-independent representation rather than
real MLIR.  It expands one rank-2 ``matmul`` into a three-deep scalar loop nest
with explicit loads, accumulator initialization, FMAs, and an output store.
The loop body is represented with counted operation templates so large shapes
do not require materializing one object per scalar operation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .estimator import _intensity, _roofline_metrics, _target_report, estimate_graph
from .models import DTYPE_BYTES, Graph, Operation, Target, Tensor
from .parser import GraphValidationError, parse_graph


class LoweringValidationError(GraphValidationError):
    """Raised when a graph is valid but outside the matmul exercise scope."""


_LOWERING_KIND = "scalar_reduction"


def _lowered_roofline_metrics(
    flops: int,
    bytes_read: int,
    bytes_written: int,
    target: Target | None,
) -> dict[str, Any] | None:
    """Return lowered-op roofline metrics with a JSON-safe zero-traffic case.

    Arithmetic intensity is undefined for nonzero FLOPs with no logical
    traffic.  Lowered register-resident templates are nevertheless
    compute-bound, so their intensity is emitted as ``None`` (JSON ``null``)
    instead of inheriting the estimator's zero-traffic ``0.0`` convention.
    All other lowered operations retain the shared estimator behavior.
    """
    metrics = _roofline_metrics(flops, bytes_read, bytes_written, target)
    if (
        metrics is not None
        and flops > 0
        and bytes_read == 0
        and bytes_written == 0
    ):
        return {**metrics, "arithmetic_intensity": None, "bound": "compute"}
    return metrics


def _validate_matmul_exercise(graph: Graph) -> tuple[Operation, Tensor, Tensor, Tensor]:
    if len(graph.operations) != 1:
        raise LoweringValidationError(
            "matmul lowering exercise requires exactly one operation"
        )

    operation = graph.operations[0]
    if operation.kind != "matmul":
        raise LoweringValidationError(
            "matmul lowering exercise requires a single matmul operation"
        )

    if graph.outputs != (operation.output,):
        raise LoweringValidationError(
            "matmul lowering exercise requires the matmul result to be the only graph output"
        )

    lhs = graph.values[operation.inputs[0]]
    rhs = graph.values[operation.inputs[1]]
    output = graph.values[operation.output]
    return operation, lhs, rhs, output


def _lowered_operation(
    *,
    name: str,
    kind: str,
    count: int,
    dtype: str,
    inputs: list[str],
    outputs: list[str],
    flops: int,
    bytes_read: int,
    bytes_written: int,
    target: Target | None,
) -> dict[str, Any]:
    return {
        "name": name,
        "op": kind,
        "count": count,
        "inputs": inputs,
        "outputs": outputs,
        "shape": [],
        "dtype": dtype,
        "flops": flops,
        "bytes_read": bytes_read,
        "bytes_written": bytes_written,
        "arithmetic_intensity": _intensity(flops, bytes_read, bytes_written),
        "roofline": _lowered_roofline_metrics(
            flops, bytes_read, bytes_written, target
        ),
    }


def _lowered_representation(
    operation: Operation,
    lhs: Tensor,
    rhs: Tensor,
    output: Tensor,
    target: Target | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    m, k = lhs.shape
    _, n = rhs.shape
    scalar_bytes = DTYPE_BYTES[lhs.dtype]
    reduction_terms = m * n * k
    output_elements = m * n

    # The loop order is explicit: each output accumulator reloads both scalar
    # operands for every reduction term, while the accumulator stays in a
    # register.  This is deliberately simple and target-independent.
    lowered_ops = [
        _lowered_operation(
            name="init_accumulator",
            kind="zero",
            count=output_elements,
            dtype=output.dtype,
            inputs=[],
            outputs=["acc[m,n]"],
            flops=0,
            bytes_read=0,
            bytes_written=0,
            target=target,
        ),
        _lowered_operation(
            name="load_lhs",
            kind="load",
            count=reduction_terms,
            dtype=lhs.dtype,
            inputs=["lhs[m,k]"],
            outputs=["lhs_scalar"],
            flops=0,
            bytes_read=reduction_terms * scalar_bytes,
            bytes_written=0,
            target=target,
        ),
        _lowered_operation(
            name="load_rhs",
            kind="load",
            count=reduction_terms,
            dtype=rhs.dtype,
            inputs=["rhs[k,n]"],
            outputs=["rhs_scalar"],
            flops=0,
            bytes_read=reduction_terms * scalar_bytes,
            bytes_written=0,
            target=target,
        ),
        _lowered_operation(
            name="fma",
            kind="fma",
            count=reduction_terms,
            dtype=output.dtype,
            inputs=["lhs_scalar", "rhs_scalar", "acc[m,n]"],
            outputs=["acc[m,n]"],
            flops=2 * reduction_terms,
            bytes_read=0,
            bytes_written=0,
            target=target,
        ),
        _lowered_operation(
            name="store_output",
            kind="store",
            count=output_elements,
            dtype=output.dtype,
            inputs=["acc[m,n]"],
            outputs=["output[m,n]"],
            flops=0,
            bytes_read=0,
            bytes_written=output.nbytes,
            target=target,
        ),
    ]
    representation = {
        "kind": _LOWERING_KIND,
        "source_operation": operation.name,
        "loop_nest": [
            {"index": "m", "extent": m},
            {"index": "n", "extent": n},
            {"index": "k", "extent": k},
        ],
        "body": [
            {
                "name": item["name"],
                "op": item["op"],
                "count": item["count"],
            }
            for item in lowered_ops
        ],
        "assumptions": {
            "accumulator_stays_in_register": True,
            "scalar_loads_are_counted_per_reduction_term": True,
            "cache_reuse_modeled": False,
            "real_mlir_required": False,
        },
    }
    return representation, lowered_ops


def _lowered_report(
    graph: Graph, representation: dict[str, Any], operations: list[dict[str, Any]]
) -> dict[str, Any]:
    total_flops = sum(operation["flops"] for operation in operations)
    total_bytes_read = sum(operation["bytes_read"] for operation in operations)
    total_bytes_written = sum(operation["bytes_written"] for operation in operations)
    return {
        "graph": graph.name,
        "target": _target_report(graph.target),
        "ops": operations,
        "totals": {
            "flops": total_flops,
            "bytes_read": total_bytes_read,
            "bytes_written": total_bytes_written,
            "arithmetic_intensity": _intensity(
                total_flops, total_bytes_read, total_bytes_written
            ),
            "roofline": _roofline_metrics(
                total_flops,
                total_bytes_read,
                total_bytes_written,
                graph.target,
            ),
        },
        "operation_count": sum(operation["count"] for operation in operations),
        "representation": representation,
    }


def _cost_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_totals = before["totals"]
    after_totals = after["totals"]
    before_roofline = before_totals["roofline"]
    after_roofline = after_totals["roofline"]
    estimated_time_us = (
        None
        if before_roofline is None or after_roofline is None
        else after_roofline["estimated_time_us"]
        - before_roofline["estimated_time_us"]
    )
    return {
        "flops": after_totals["flops"] - before_totals["flops"],
        "bytes_read": after_totals["bytes_read"] - before_totals["bytes_read"],
        "bytes_written": after_totals["bytes_written"]
        - before_totals["bytes_written"],
        "logical_bytes": (
            after_totals["bytes_read"]
            + after_totals["bytes_written"]
            - before_totals["bytes_read"]
            - before_totals["bytes_written"]
        ),
        "operation_count": after["operation_count"] - len(before["ops"]),
        "roofline": {"estimated_time_us": estimated_time_us},
    }


def compare_matmul_lowering_graph(graph: Graph) -> dict[str, Any]:
    """Compare a validated single-matmul graph with its scalar lowering."""
    operation, lhs, rhs, output = _validate_matmul_exercise(graph)
    before = estimate_graph(graph)
    representation, lowered_operations = _lowered_representation(
        operation, lhs, rhs, output, graph.target
    )
    after = _lowered_report(graph, representation, lowered_operations)
    return {
        "graph": graph.name,
        "target": _target_report(graph.target),
        "before": before,
        "after": after,
        "cost_delta": _cost_delta(before, after),
    }


def compare_matmul_lowering(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and compare one graph before and after scalar matmul lowering."""
    return compare_matmul_lowering_graph(parse_graph(payload))
