"""An opt-in, dependency-free matmul lowering exercise.

The lowering is intentionally a target-independent representation rather than
real MLIR. It expands one rank-2 ``matmul`` into scalar or N-axis tiled/vector
operation templates for loads, accumulator initialization, FMAs, and stores.
Counted templates avoid materializing one object per instruction.
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
_TILED_LOWERING_KIND = "tiled_vectorized"


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


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
    if len(lhs.shape) != 2 or len(rhs.shape) != 2:
        raise LoweringValidationError(
            "matmul lowering exercise requires rank-2 input tensors"
        )
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


def _tiled_vectorized_representation(
    operation: Operation,
    lhs: Tensor,
    rhs: Tensor,
    output: Tensor,
    target: Target | None,
    *,
    tile_size: int,
    vector_width: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build a cache-reuse-aware tiled/vectorized matmul representation."""
    m, k = lhs.shape
    _, n = rhs.shape
    scalar_bytes = DTYPE_BYTES[lhs.dtype]
    zero_vectors = zero_scalar_tail = 0
    load_lhs_scalars = load_rhs_vectors = load_rhs_scalar_tail = 0
    vector_fmas = scalar_tail_fmas = 0
    store_vectors = store_scalar_tail = 0

    for m0 in range(0, m, tile_size):
        rows = min(tile_size, m - m0)
        for n0 in range(0, n, tile_size):
            cols = min(tile_size, n - n0)
            full_vectors, tail_lanes = divmod(cols, vector_width)
            zero_vectors += rows * full_vectors
            zero_scalar_tail += rows * tail_lanes
            store_vectors += rows * full_vectors
            store_scalar_tail += rows * tail_lanes
    for m0 in range(0, m, tile_size):
        rows = min(tile_size, m - m0)
        for k0 in range(0, k, tile_size):
            depth = min(tile_size, k - k0)
            # One scalar LHS value is loaded for each (m, k) and broadcast
            # across N lanes; it is reused across all N tiles.
            load_lhs_scalars += rows * depth
    for k0 in range(0, k, tile_size):
        depth = min(tile_size, k - k0)
        for n0 in range(0, n, tile_size):
            cols = min(tile_size, n - n0)
            full_vectors, tail_lanes = divmod(cols, vector_width)
            load_rhs_vectors += depth * full_vectors
            load_rhs_scalar_tail += depth * tail_lanes
    for m0 in range(0, m, tile_size):
        rows = min(tile_size, m - m0)
        for n0 in range(0, n, tile_size):
            cols = min(tile_size, n - n0)
            full_vectors, tail_lanes = divmod(cols, vector_width)
            for k0 in range(0, k, tile_size):
                depth = min(tile_size, k - k0)
                vector_fmas += rows * full_vectors * depth
                scalar_tail_fmas += rows * tail_lanes * depth

    lowered_ops = [
        _lowered_operation(
            name="init_accumulator_vectors", kind="zero", count=zero_vectors,
            dtype=output.dtype, inputs=[], outputs=["acc[m,n]"], flops=0,
            bytes_read=0, bytes_written=0, target=target,
        ),
        _lowered_operation(
            name="init_accumulator_scalar_tail", kind="zero", count=zero_scalar_tail,
            dtype=output.dtype, inputs=[], outputs=["acc[m,n]"], flops=0,
            bytes_read=0, bytes_written=0, target=target,
        ),
        _lowered_operation(
            name="load_lhs_scalars", kind="load", count=load_lhs_scalars,
            dtype=lhs.dtype, inputs=["lhs[m,k]"], outputs=["lhs_scalar"], flops=0,
            bytes_read=load_lhs_scalars * scalar_bytes, bytes_written=0, target=target,
        ),
        _lowered_operation(
            name="load_rhs_vectors", kind="load", count=load_rhs_vectors,
            dtype=rhs.dtype, inputs=["rhs[k,n]"], outputs=["rhs_vector"], flops=0,
            bytes_read=load_rhs_vectors * vector_width * scalar_bytes,
            bytes_written=0, target=target,
        ),
        _lowered_operation(
            name="load_rhs_scalar_tail", kind="load", count=load_rhs_scalar_tail,
            dtype=rhs.dtype, inputs=["rhs[k,n]"], outputs=["rhs_scalar"], flops=0,
            bytes_read=load_rhs_scalar_tail * scalar_bytes,
            bytes_written=0, target=target,
        ),
        _lowered_operation(
            name="vector_fma", kind="fma", count=vector_fmas,
            dtype=output.dtype, inputs=["lhs_scalar", "rhs_vector", "acc[m,n]"],
            outputs=["acc[m,n]"], flops=2 * vector_fmas * vector_width,
            bytes_read=0, bytes_written=0, target=target,
        ),
        _lowered_operation(
            name="scalar_tail_fma", kind="fma", count=scalar_tail_fmas,
            dtype=output.dtype, inputs=["lhs_scalar", "rhs_scalar", "acc[m,n]"],
            outputs=["acc[m,n]"], flops=2 * scalar_tail_fmas,
            bytes_read=0, bytes_written=0, target=target,
        ),
        _lowered_operation(
            name="store_output_vectors", kind="store", count=store_vectors,
            dtype=output.dtype, inputs=["acc[m,n]"], outputs=["output[m,n]"], flops=0,
            bytes_read=0, bytes_written=store_vectors * vector_width * scalar_bytes,
            target=target,
        ),
        _lowered_operation(
            name="store_output_scalar_tail", kind="store", count=store_scalar_tail,
            dtype=output.dtype, inputs=["acc[m,n]"], outputs=["output[m,n]"], flops=0,
            bytes_read=0, bytes_written=store_scalar_tail * scalar_bytes, target=target,
        ),
    ]
    representation = {
        "kind": _TILED_LOWERING_KIND,
        "source_operation": operation.name,
        "tile_size": tile_size,
        "vector_width": vector_width,
        "tile_grid": {"m": _ceil_div(m, tile_size), "n": _ceil_div(n, tile_size), "k": _ceil_div(k, tile_size)},
        "body": [
            {"name": item["name"], "op": item["op"], "count": item["count"]}
            for item in lowered_ops
        ],
        "assumptions": {
            "input_tiles_reused_across_output_tiles": True,
            "vector_axis": "n",
            "lhs_scalar_broadcast_across_vector_lanes": True,
            "vector_width_is_an_instruction_laning_factor": True,
            "incomplete_n_vectors_use_scalar_tail_operations": True,
            "cache_capacity_and_alignment_modeled": False,
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


def compare_matmul_lowering_graph(
    graph: Graph,
    *,
    mode: str = "scalar",
    tile_size: int = 16,
    vector_width: int = 4,
) -> dict[str, Any]:
    """Compare a validated single-matmul graph with a selected lowering."""
    operation, lhs, rhs, output = _validate_matmul_exercise(graph)
    if mode not in {"scalar", "tiled"}:
        raise LoweringValidationError("lowering mode must be 'scalar' or 'tiled'")
    for name, value in (("tile_size", tile_size), ("vector_width", vector_width)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise LoweringValidationError(f"{name} must be a positive integer")
    before = estimate_graph(graph)
    if mode == "scalar":
        representation, lowered_operations = _lowered_representation(
            operation, lhs, rhs, output, graph.target
        )
    else:
        representation, lowered_operations = _tiled_vectorized_representation(
            operation, lhs, rhs, output, graph.target,
            tile_size=tile_size, vector_width=vector_width,
        )
    after = _lowered_report(graph, representation, lowered_operations)
    return {
        "graph": graph.name,
        "target": _target_report(graph.target),
        "before": before,
        "after": after,
        "cost_delta": _cost_delta(before, after),
    }


def compare_matmul_lowering(
    payload: Mapping[str, Any],
    *,
    mode: str = "scalar",
    tile_size: int = 16,
    vector_width: int = 4,
) -> dict[str, Any]:
    """Validate and compare one graph before and after matmul lowering."""
    return compare_matmul_lowering_graph(
        parse_graph(payload), mode=mode, tile_size=tile_size, vector_width=vector_width
    )
