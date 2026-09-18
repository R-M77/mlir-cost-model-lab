# MLIR Cost Model Lab

A small, dependency-light Python 3.11+ project for making tensor graph costs explicit. It accepts a compact JSON graph inspired by MLIR's value-and-operation model and produces deterministic estimates that are easy to inspect in a code review.

This is a portfolio slice, not a hardware benchmark. It models static operation and logical tensor traffic counts. When an optional target is supplied, it also computes an idealized roofline lower bound; that estimate is not measured or predicted wall-clock runtime.

## Problem

Compiler and kernel decisions often need a quick first-order answer to questions such as:

- Which operation dominates scalar work?
- How much tensor data does each operation read and write?
- Which adjacent elementwise operations could share an intermediate?

`mlir-cost-model-lab` answers those questions for a deliberately small graph vocabulary without depending on MLIR, NumPy, or a runtime.

## Architecture

The package under `src/cost_model` is split into five small layers:

- `models.py` — immutable tensor, operation, and graph data structures.
- `parser.py` — JSON schema validation, value lookup, and static shape inference.
- `estimator.py` — FLOP, byte-traffic, arithmetic-intensity, roofline, and baseline fusion calculations.
- `optimizer.py` — opt-in liveness-aware elementwise fusion planning and before/after reporting.
- `cli.py` / `__main__.py` — stable `python -m cost_model` interface, batch benchmark mode, and readable JSON output.

The parser processes operations in source order, so an operation can only use an input or an earlier result. Elementwise `add` and `mul` use trailing-dimension broadcasting; `matmul` is rank-2 only. An optional validated target is carried with the graph. The estimator preserves operation order and emits JSON-friendly integer counts plus floating-point intensity and roofline values.

## Quickstart

Run the example without installing anything beyond Python:

```bash
PYTHONPATH=src python -m cost_model examples/matmul_bias_relu.json
```

Use compact output for scripts:

```bash
PYTHONPATH=src python -m cost_model --compact examples/matmul_bias_relu.json
```

Request the opt-in optimization report (the unfused baseline remains embedded for auditability):

```bash
PYTHONPATH=src python -m cost_model --optimize examples/matmul_bias_relu.json
```

Run several graph files as one deterministic batch. The `--benchmark` file is a non-empty JSON array. Each item can be a graph path string, or an object with an explicit case name and graph path:

```json
[
  "examples/matmul_bias_relu.json",
  {"name": "named_case", "path": "examples/matmul_bias_relu.json"}
]
```

Graph paths are resolved relative to the benchmark JSON file, and a path-only case uses the graph's own `name`. Emit one-line JSON for scripts with:

```bash
PYTHONPATH=src python -m cost_model --benchmark benchmark.json --compact
```

Batch output keeps each report under `cases[*].report` and adds an aggregate `summary`:

```json
{
  "cases": [
    {"name": "matmul_bias_relu", "path": "examples/matmul_bias_relu.json", "report": {"...": "..."}}
  ],
  "summary": {
    "case_count": 1,
    "totals": {
      "flops": 64,
      "bytes_read": 152,
      "bytes_written": 96,
      "logical_bytes": 248,
      "arithmetic_intensity": 0.25806451612903225,
      "estimated_time_us": 0.00248
    }
  }
}
```

The summary totals use each case's normal report, or its `optimized.totals` when `--optimize` is also supplied. `estimated_time_us` is the sum of per-case roofline times when every case has a target; it is `null` for a batch with any target-less case. A positional graph path and `--benchmark` are mutually exclusive; omitting both is an error. Existing single-graph commands and their JSON shape are unchanged.

The target-bound regression fixture runs a compute-heavy matmul and a
memory-heavy ReLU against the same idealized target:

```bash
PYTHONPATH=src python -m cost_model --benchmark benchmarks/target_bounds.json --compact
```

The first case must classify as `compute`-bound and the second as `memory`-bound.
`benchmarks/target_bounds.expected.json` stores the deterministic compact batch
output; CI compares a fresh run with that file so classification, per-case
reports, and aggregate totals cannot drift silently.

Install the project and test extras in a virtual environment when desired:

```bash
python -m pip install -e '.[test]'
pytest
```

The command exits with status `2` and prints a concise `error:` message for invalid JSON or invalid graph input.

## Input format

A graph has a name, named tensor inputs, a source-ordered list of single-output operations, graph outputs, and an optional target for roofline analysis:

```json
{
  "name": "matmul_bias_relu",
  "target": {
    "peak_compute_gflops": 1000.0,
    "memory_bandwidth_gbps": 100.0
  },
  "inputs": {
    "lhs": {"shape": [2, 3], "dtype": "f32"},
    "rhs": {"shape": [3, 4], "dtype": "f32"},
    "bias": {"shape": [4], "dtype": "f32"}
  },
  "ops": [
    {"name": "matmul", "op": "matmul", "inputs": ["lhs", "rhs"], "outputs": ["product"]},
    {"name": "add_bias", "op": "add", "inputs": ["product", "bias"], "outputs": ["biased"]},
    {"name": "relu", "op": "relu", "inputs": ["biased"], "outputs": ["output"]}
  ],
  "outputs": ["output"]
}
```

Supported operations are `matmul`, `add`, `mul`, and `relu`. Supported dtypes include `f16`, `bf16`, `f32`, `f64`, `bool`, `i8`, `u8`, `i32`, and `i64`.

The optional `target` values are decimal rates: `peak_compute_gflops` is peak compute throughput in GFLOP/s, and `memory_bandwidth_gbps` is memory bandwidth in GB/s. Both fields are required when `target` is an object and must be positive, finite JSON numbers; booleans, strings, zero, negative values, `NaN`, and infinity are rejected. An omitted or explicit `null` target disables roofline analysis. In that case the report emits `"target": null` and `"roofline": null` for every operation and for `totals`; it never invents hardware defaults.

## Cost definitions

- **Matmul FLOPs:** `2 × M × K × N`, counting one multiply and one add per output reduction term.
- **Elementwise FLOPs:** one scalar operation per output element. `relu` counts its scalar comparison as one operation so it remains visible in the report; this is a modeling convention, not a hardware claim.
- **Bytes read:** the sum of the storage size of each input value referenced by the operation.
- **Bytes written:** the storage size of the operation's output value.
- **Arithmetic intensity:** `FLOPs / (bytes read + bytes written)`, in FLOPs/byte. Zero traffic would produce `0.0` rather than a division error.
- **Roofline ridge point:** `peak_compute_gflops / memory_bandwidth_gbps`, in FLOPs/byte. A per-op or total analysis is `compute`-bound when its arithmetic intensity is at least the ridge point (including equality); otherwise it is `memory`-bound.
- **Roofline estimated time:** `max(compute_time_us, memory_time_us)`, in microseconds. `compute_time_us = FLOPs / (peak_compute_gflops × 1000)` and `memory_time_us = (bytes read + bytes written) / (memory_bandwidth_gbps × 1000)`. The `1000` conversion follows from decimal GFLOP/s and GB/s and converts seconds to microseconds. This is an idealized lower-bound calculation from logical traffic, not a measured runtime.
- **Fusion candidates:** the default report's informational candidates are adjacent `add`, `mul`, or `relu` operations that share a single-use intermediate. `bytes_avoided` estimates the intermediate write plus the later read that a fusion could remove; it does not change default totals.

With a target, each entry in `ops` and `totals` contains a `roofline` object with `arithmetic_intensity` (FLOPs/byte), `ridge_point` (FLOPs/byte), `bound`, and `estimated_time_us` (microseconds). The report also echoes the validated target. Without a target, those `roofline` values are explicit `null` rather than assumptions.

The report contains `ops`, `totals`, a `fusion` section, and the target/roofline analysis described above. Counts are exact for the model above; roofline times are transparent lower bounds and are not measurements of a particular accelerator, cache, compiler, or runtime.

## Opt-in optimization report

`--optimize` and `analyze_graph(payload, optimize=True)` return a deterministic report with this shape:

- `baseline` — the complete, unchanged report that the default command returns. It includes every source operation, including dead operations, and its original informational `fusion` candidates.
- `optimized` — the liveness-aware plan and recomputed costs. Its `ops` list contains only live operations in original source order; fusion-local reads and writes are removed from the affected entries so its totals add up exactly.
- `cost_delta` — signed optimized-minus-baseline deltas for `flops`, `bytes_read`, `bytes_written`, `logical_bytes` (read plus written bytes), and `roofline.estimated_time_us`. Negative values mean the optimized report is lower; positive values mean it is higher. With no target, `roofline.estimated_time_us` is `null`.
- `optimized.applied_fusions` — ordered `{producer, consumer, intermediate, bytes_avoided, reason}` records for every legal fusion.
- `optimized.eliminated_dead_operations` — operation names, in original source order, that cannot reach a graph output.
- `optimized.bytes_avoided` — the sum of the applied fusion records. Each single-use intermediate contributes its logical write plus the later logical read once, including across a chained fusion sequence.
- `optimized.graph_outputs`, `live_operation_order`, and `result_order` — explicit preservation of graph output names and the remaining operation/result order.
- `optimized.legality_scope` — the assumptions and checks used by this pass.

The optimizer fuses only adjacent source-order pairs where both operations are pure elementwise `add`, `mul`, or `relu`, the producer result is not a graph output, and that result has exactly one consumer. It never fuses across a non-adjacent operation, `matmul`, a graph output, or a multi-consumer intermediate. Liveness is computed backward from `Graph.outputs`, so dead operations contribute to the baseline for auditability but not to optimized totals or optimized roofline metrics. The pass plans a legal cost transformation; it does not execute the graph or rewrite an IR.

The legality scope is intentionally explicit: this IR has no side-effect, layout, stride, or alias metadata. The pass therefore assumes the listed elementwise operations are pure and does not claim a general compiler legality proof. The optimized totals reuse the existing FLOP, logical byte-traffic, intensity, and roofline definitions exactly. With the example graph, the baseline is 64 FLOPs and 248 bytes, while the optimized result is 64 FLOPs and 184 bytes after one `add_bias` → `relu` fusion avoids 64 bytes; with the example target its optimized roofline is memory-bound with estimated total time `184 / (100 × 1000)` microseconds. The `cost_delta.roofline.estimated_time_us` value is an idealized lower-bound model delta, not a measured speedup.

## Limitations

- Static, fully known positive dimensions only; no symbolic shapes or runtime shape guards.
- `matmul` accepts rank-2 tensors and requires matching dtypes.
- Elementwise broadcasting follows a small trailing-dimension rule; there is no layout or stride model.
- Every operation has exactly one output, and operations are already topologically ordered.
- Byte traffic is logical tensor storage traffic. It does not model cache reuse, tiling, alignment, transfers, or aliasing; the roofline memory term uses the full logical read-plus-write total.
- Default fusion reporting is an informational local opportunity heuristic; only the opt-in optimizer applies liveness and the narrower legality checks documented above.
- A supplied target provides idealized peak compute and bandwidth ceilings only. Roofline `estimated_time_us` values omit latency, utilization, vectorization, parallelism, overlap, and device-specific instruction costs, so they are lower-bound estimates rather than measured performance.

## Development

The project has no runtime dependencies. Focused tests cover exact estimates, target-aware compute- and memory-bound roofline cases, target validation, no-target null behavior, JSON file loading, default and optimized CLI behavior, deterministic batch benchmark output, optimizer legality/liveness, and invalid schema/reference/shape inputs. CI runs the test suite on Python 3.11 and 3.12.
