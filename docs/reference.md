# Reference guide

Full input, CLI, output, and model reference for [MLIR Cost Model Lab](../README.md). Run the commands below from the repository root.

This is a portfolio slice, not a hardware benchmark. It models static operation and logical tensor traffic counts. When an optional target is supplied, it also computes an idealized roofline lower bound; that estimate is not measured or predicted wall-clock runtime.

## Problem

Compiler and kernel decisions often need a quick first-order answer to questions such as:

- Which operation dominates scalar work?
- How much tensor data does each operation read and write?
- Which adjacent elementwise operations could share an intermediate?

`mlir-cost-model-lab` answers those questions for a deliberately small graph vocabulary without depending on MLIR, NumPy, or a runtime.

## Architecture

The package under `src/cost_model` is split into small layers:

- `models.py` — frozen tensor, operation, and graph dataclasses (graph maps remain mutable).
- `parser.py` — JSON schema validation, value lookup, and static shape inference.
- `estimator.py` — FLOP, byte-traffic, arithmetic-intensity, roofline, and baseline fusion calculations.
- `optimizer.py` — opt-in liveness-aware elementwise fusion planning and before/after reporting.
- [`lowering.py`](../src/cost_model/lowering.py) — an opt-in, target-independent scalar or tiled/vector matmul lowering exercise and cost comparison.
- [`mlir_emitter.py`](../src/cost_model/mlir_emitter.py) — deterministic textual MLIR emission for one standalone rank-2 f32 matmul.
- `cli.py` / `__main__.py` — stable `python -m cost_model` interface, batch benchmark mode, deterministic budget gate, and readable JSON output.

The parser processes operations in source order, so an operation can only use an input or an earlier result. Elementwise `add` and `mul` use trailing-dimension broadcasting; `matmul` accepts rank-2-or-higher operands and broadcasts their batch prefixes using the same right-aligned rule. Lowering and textual MLIR emission remain rank-2 only. An optional validated target is carried with the graph. The estimator preserves operation order and emits JSON-friendly integer counts plus floating-point intensity and roofline values.

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

Run the opt-in matmul-lowering exercise:

```bash
PYTHONPATH=src python -m cost_model --lower-matmul examples/matmul_lowering.json
```

Run several graph files as one deterministic batch. The `--benchmark` file is a non-empty JSON array. Each item can be a graph path string, or an object with an explicit case name and graph path:

```json
[
  "examples/matmul_bias_relu.json",
  {"name": "named_case", "path": "examples/matmul_bias_relu.json"}
]
```

Graph paths are resolved relative to the benchmark JSON file, and a path-only case uses the graph's own `name`. Effective case names must be unique across the batch; duplicate names are rejected before any report is emitted. Emit one-line JSON for scripts with:

```bash
PYTHONPATH=src python -m cost_model --benchmark benchmark.json --compact
```

Select one or more effective case names with repeatable `--case` options. Selection preserves the benchmark file's original order and is applied only after every case has been loaded and validated, so invalid or duplicate unselected cases still fail the command:

```bash
PYTHONPATH=src python -m cost_model \
  --benchmark benchmark.json \
  --case named_case \
  --case matmul_bias_relu \
  --compact
```

`--case` is valid only with `--benchmark`; names must match exactly, and an unknown name is an error. When `--budget` is also supplied, aggregate and per-case limits are checked against the selected report, so budget case entries must name selected cases.

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

Use the opt-in benchmark regression gate with an absolute upper-bound budget:

```bash
PYTHONPATH=src python -m cost_model \
  --benchmark benchmarks/target_bounds.json \
  --budget benchmarks/target_bounds.budget.json \
  --compact
```

The budget JSON requires a non-empty `aggregate` object and accepts optional
per-case limits keyed by the case name:

```json
{
  "aggregate": {
    "flops": 1572864,
    "logical_bytes": 8437760,
    "estimated_time_us": 84.410368
  },
  "cases": {
    "compute_bound_matmul": {
      "flops": 524288,
      "logical_bytes": 49152,
      "estimated_time_us": 0.524288
    }
  }
}
```

Supported metrics are `flops`, `logical_bytes`, and `estimated_time_us`. Every
limit is an absolute upper bound: the gate fails only when `actual > limit`, so
equality passes. A metric may be omitted when it is not applicable. In
particular, `estimated_time_us` is unavailable for a target-less case (and for
a batch containing one), so requesting that limit fails closed with a clear
error instead of inventing a value. Unknown fields, unsupported metrics,
non-finite values, missing aggregate limits, and unknown case names are errors.
On a violation the command keeps the normal report on stdout, emits a
single-line `error: {...}` JSON diagnostic on stderr, and exits with status `2`.
Without `--budget`, existing single-graph and batch commands and their JSON
output are unchanged.

## Matmul lowering exercise

`--lower-matmul` and `compare_matmul_lowering(payload)` accept one graph containing
exactly one rank-2 `matmul` whose result is the only graph output. They return a
deterministic report with:

- `before` — the normal `analyze_graph`-style cost report for the high-level `matmul`.
- `after` — a cost report for a counted scalar-reduction or tiled/vector representation with explicit `load`, `zero`, `fma`, and `store` operation templates.
- `cost_delta` — signed after-minus-before changes for FLOPs, byte traffic, expanded operation count, and target roofline time.

Both `scalar` and `tiled` lowering intentionally reject batched operands with
`LoweringValidationError`, even when those operands are valid for parsing and
cost estimation. `compare_matmul_lowering_graph(graph)` has the same scope.

The representation is deliberately not MLIR and has no MLIR dependency. For
`examples/matmul_lowering.json`, the high-level graph models 48 FLOPs and 104
logical bytes. The lowered representation keeps 48 FLOPs but models 224
logical bytes and 88 expanded scalar operations, because each reduction term
reloads both scalar operands while the accumulator stays in a register. Its
target-aware modeled time changes from `0.00104` to `0.00224` microseconds.
This is a deterministic exercise in how lowering choices affect the model, not
a runtime prediction; the assumptions are included in `after.representation`.
The register-resident `fma` template has nonzero FLOPs and zero logical bytes,
so its roofline arithmetic intensity is undefined. The lowered operation emits
that value as JSON `null`, classifies the template as `compute`-bound, and keeps
its compute-only estimated time. This is a narrow lowering-report convention;
the shared estimator and aggregate before/after metrics retain their existing
zero-traffic and total-cost behavior.
The flag is single-graph only and cannot be combined with `--benchmark` or
`--optimize`. Select the tiled/vectorized lowering with explicit tile and lane
parameters:

```bash
PYTHONPATH=src python -m cost_model \
  --lower-matmul \
  --lowering-mode tiled \
  --tile-size 16 \
  --vector-width 4 \
  examples/matmul_lowering.json
```

The tiled representation assumes ideal global reuse: every element of each
input is read once across all output tiles, regardless of tile size. Its total
logical bytes and roofline estimate therefore equal the high-level baseline.
Tiles span M, N, and K, including ragged final tiles. Vectorization is only along
N and is computed independently within each N tile: `divmod(tile_columns, vector_width)`
gives full-width vectors and individual scalar-tail operations. Vectors never
cross tile boundaries; masked or partial-width vectors are not counted. A width
of 1 yields one-lane vectors without tails; a width larger than an N tile yields
only scalar tails in that tile. LHS loads are scalar and broadcast across N
lanes; RHS loads, accumulator initializations, FMAs, and stores have separate
vector and scalar-tail templates, including zero-count categories.
Tile size and vector width change these counts, not the modeled total FLOPs or
logical bytes, and do not produce a measured speedup. Cache capacity, eviction,
alignment, register pressure, and generated machine code are not modeled. The same `before`, `after`, and `cost_delta` contract is returned
for both `scalar` and `tiled` modes.

Install the project and test extras in a virtual environment when desired:

```bash
python -m pip install -e '.[test]'
pytest
```

Handled JSON, graph-validation, budget-validation and budget-violation errors
return status `2` with an `error:` diagnostic. Extreme numeric and malformed
encoding cases have additional robustness limits; see [privacy guidance](privacy.md).

## Textual MLIR emission

```bash
PYTHONPATH=src python -m cost_model --emit-mlir examples/matmul_lowering.json
```

`emit_matmul_mlir(payload)` and `emit_matmul_mlir_graph(graph)` emit deterministic
textual MLIR for exactly one rank-2 f32 `matmul`, with its result as the only
graph output and exactly its operands as graph inputs (one named input is valid
when both operand references are the same). Batched operands are intentionally
unsupported and raise `MatmulMlirEmissionError`, even when parsing and estimation
succeed. Other dtypes accepted by the estimator are also outside this emitter's
scope.

The emitted `module` contains `func.func`, `tensor.empty`, an `arith.constant`
zero, `linalg.fill` to initialize the output, `linalg.matmul`, and a return.
This is separate from the counted lowering representations: emitting text does
not run a lowering pipeline, compile or execute a kernel, or verify the text
with MLIR tooling. No MLIR dependency is required.

The CLI prints raw MLIR rather than a JSON cost report. `--emit-mlir` requires a
single graph path and cannot be combined with `--benchmark`, `--optimize`,
`--lower-matmul`, `--compact`, `--omit-paths`, or `--budget`; `--case` is also
unavailable because it requires benchmark mode.

## Input format

A graph has a name, named tensor inputs, a source-ordered list of single-output operations, graph outputs, and an optional target for roofline analysis:

Top-level `outputs` must contain unique value names; declaration order is preserved. Repeated operation operands, such as `mul(x, x)`, remain valid.

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

Supported operations are `matmul`, `add`, `mul`, `relu`, `reduce_sum`, `softmax`, and `layer_norm`. Supported dtypes include `f16`, `bf16`, `f32`, `f64`, `bool`, `i8`, `u8`, `i32`, and `i64`.

The optional `target` values are decimal rates: `peak_compute_gflops` is peak compute throughput in GFLOP/s, and `memory_bandwidth_gbps` is memory bandwidth in GB/s. Both fields are required when `target` is an object and must be positive, finite JSON numbers; booleans, strings, zero, negative values, `NaN`, and infinity are rejected. An omitted or explicit `null` target disables roofline analysis. In that case the report emits `"target": null` and `"roofline": null` for every operation and for `totals`; it never invents hardware defaults.

### Matmul shapes and broadcasting

For operands shaped `[..., M, K]` and `[..., K, N]`, both ranks must be at least
2, their dtypes must match, and the contraction dimensions K must be equal.
Only the leading batch prefixes broadcast: align them from the right, treating
missing leading axes as 1. Each aligned pair must be equal or have one dimension
equal to 1; otherwise validation fails. The output shape is
`broadcast(lhs.shape[:-2], rhs.shape[:-2]) + (M, N)`, with the shared input dtype.
Rank-2 operands have empty batch prefixes and can broadcast across a batched
operand. Rank-0/rank-1 operands are rejected; there is no vector promotion.

For a runnable example:

```bash
PYTHONPATH=src python -m cost_model --compact examples/batched_matmul.json
```

Its `[2, 1, 3, 5]` and `[1, 4, 5, 7]` f32 inputs produce `[2, 4, 3, 7]`:
1,680 FLOPs, 680 bytes read, and 672 bytes written (1,352 logical bytes).
Broadcasting expands the output work, not the stored input sizes used for
logical read accounting. See the formulas below and the implementations in
[`parser.py`](../src/cost_model/parser.py) and
[`estimator.py`](../src/cost_model/estimator.py).

## Cost definitions

- **Matmul FLOPs:** `2 × B × M × K × N` (equivalently `2 × output.elements × K`), counting one multiply and one add per output reduction term. B is the product of the broadcasted batch dimensions, or 1 for rank-2 matmul.
- **Elementwise FLOPs:** one scalar operation per output element. `relu` counts its scalar comparison as one operation so it remains visible in the report; this is a modeling convention, not a hardware claim.
- **Bytes read:** the sum of the storage size of each input reference in the operation. Matmul reads `dtype_bytes × (product(lhs.shape) + product(rhs.shape))`: each operand's stored size is counted once per reference, not expanded to the broadcasted batch shape or multiplied by reduction reuse. Referencing the same value as both operands counts its storage twice.
- **Bytes written:** the storage size of the operation's output value. For matmul this is `dtype_bytes × B × M × N`, including the full broadcasted output.
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
- `matmul` accepts rank-2-or-higher tensors with matching dtypes, aligned contraction dimensions, and broadcast-compatible batch prefixes; rank-0/rank-1 inputs are unsupported.
- Elementwise broadcasting follows a small trailing-dimension rule; there is no layout or stride model.
- Every operation has exactly one output, and operations are already topologically ordered.
- Baseline byte traffic is logical tensor storage traffic, without cache, layout, transfer, or alias analysis. The tiled lowering separately assumes ideal global input reuse; it does not simulate a cache or prove that the reuse is feasible.
- Default fusion reporting is an informational local opportunity heuristic; only the opt-in optimizer applies liveness and the narrower legality checks documented above.
- The lowering exercise supports one standalone rank-2 `matmul`; it is a target-independent representation, not an MLIR parser or lowering pass.
- The separate textual MLIR emitter supports one standalone rank-2 f32 `matmul` only. Neither lowering nor emission supports batched operands; estimation support does not imply backend support.
- Scalar lowering counts both operand loads per reduction term. Tiled lowering models ideal reuse and vector template counts. Neither representation generates or executes code.
- A supplied target provides idealized peak compute and bandwidth ceilings only. Roofline `estimated_time_us` values omit latency, utilization, vectorization, parallelism, overlap, and device-specific instruction costs, so they are lower-bound estimates rather than measured performance.

## Development

The project has no runtime dependencies. Focused tests cover exact estimates, target-aware compute- and memory-bound roofline cases, target validation, no-target null behavior, JSON file loading, default and optimized CLI behavior, deterministic batch benchmark output, optimizer legality/liveness, the opt-in matmul lowering exercise, and invalid schema/reference/shape inputs. CI runs the test suite on Python 3.11 and 3.12.

## Handling private inputs and reports

See [privacy guidance](privacy.md) before sharing results from non-example inputs.
With `--benchmark`, `--omit-paths` removes each `cases[*].path` key from JSON
stdout. It leaves names, tensor details, model counts, targets, and stderr
diagnostics unchanged. It is not an anonymizer. The flag is rejected without
`--benchmark`; cost calculations, case selection, and budget semantics are
unchanged.

## Additional model conventions

`flops` is a uniform scalar-work counter, including integer/Boolean operations
and ReLU comparisons; dtype support sets storage size, not hardware throughput.
Graph roofline totals apply the formula once to aggregate work and traffic; they
do not sum per-operation times. Batch times sum per-graph totals. Lower-bound
interpretation is conditional on the chosen logical-traffic assumptions.
The tiled model counts full-width N-axis vectors and individual scalar tails
within each tile, never vectors that cross tile boundaries. Ceiling division
is used for tile-grid extents, not to count partial vectors. Positive tile/lane CLI options
are meaningful only for tiled lowering; they may otherwise be ignored.
