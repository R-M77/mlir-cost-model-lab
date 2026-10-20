# MLIR Cost Model Lab

**Explain tensor graph costs, compare fusion plans, and inspect lowering tradeoffs in pure Python.**

A Python tool with no runtime dependencies for a small, statically shaped JSON
graph vocabulary. It validates value references and shapes, counts scalar work
and logical tensor traffic, and emits deterministic reports that can be reviewed
and regression-tested.

This is an educational compiler-modeling project inspired by MLIR's values and
operations. It does not parse MLIR, compile kernels, execute graphs, or measure
hardware performance. Roofline times are idealized model estimates based on the
ceilings you supply.

## What it demonstrates

- **Graph validation and shape inference:** rank-2 and batched matmul,
  trailing-dimension broadcasting, last-axis reductions and normalization,
  source-ordered values, dtype and reference checks
- **Explainable cost accounting:** per-operation and aggregate work, reads,
  writes, arithmetic intensity, and optional roofline classification
- **Liveness-aware optimization planning:** remove dead operations from the
  modeled plan and account for adjacent single-use elementwise fusion,
  preserving graph outputs and an unchanged baseline report
- **Lowering representations:** compare one rank-2 matmul with counted scalar or
  tiled/vector templates, with explicit scalar tails and signed cost deltas
- **Textual MLIR emission:** deterministic MLIR for one standalone rank-2 f32
  matmul, without requiring MLIR tooling
- **Deterministic regression checks:** batch cases, exact JSON fixtures, case
  selection, and aggregate/per-case budget gates with nonzero failure status

## Run a reproducible example

Requires Python 3.11+. From the repository root, no installation is needed:

```bash
PYTHONPATH=src python -m cost_model examples/matmul_bias_relu.json
PYTHONPATH=src python -m cost_model --optimize examples/matmul_bias_relu.json
```

The included graph is a `2×3` by `3×4` matmul, followed by bias addition and ReLU.
At the example's synthetic target of 1,000 GFLOP/s and 100 GB/s:

| Model result | Baseline | Fusion plan |
| --- | ---: | ---: |
| Scalar work (`flops`) | 64 | 64 |
| Logical bytes read + written | 248 | 184 |
| Roofline bound | memory | memory |
| Idealized time, µs | 0.00248 | 0.00184 |

One `add_bias` → `relu` fusion removes a modeled intermediate write and read:
**64 bytes, or 25.8% of this graph's logical traffic**. No graph is executed or
rewritten, and this result is not a measured speedup. ReLU comparisons count as
one scalar operation each under the model's `flops` convention.

### Batched matmul

```bash
PYTHONPATH=src python -m cost_model --compact examples/batched_matmul.json
```

The inputs `[2, 1, 3, 5]` and `[1, 4, 5, 7]` produce `[2, 4, 3, 7]`:
eight `3 × 5 × 7` matrix products, totaling 1,680 FLOPs, 680 bytes read,
and 672 bytes written. The logical read model counts each stored operand
once, without materializing broadcast copies. At the example's target,
the roofline lower bound is memory-bound at `1.352` microseconds—not a
measured runtime.

The final two input axes are `(M, K)` and `(K, N)`; leading axes broadcast
using the trailing-dimension rule. FLOPs are `2 × B × M × K × N`, where `B`
is the product of the broadcasted batch dimensions (1 for rank-2 matmul).
Rank-2 inputs may also broadcast across a batched operand. Rank-0/rank-1
inputs, mismatched dtypes, unaligned contraction dimensions, and incompatible
batch axes are rejected. Lowering and textual MLIR emission remain rank-2-only.

### Compare matmul representations

```bash
PYTHONPATH=src python -m cost_model --lower-matmul examples/matmul_lowering.json
PYTHONPATH=src python -m cost_model --lower-matmul \
  --lowering-mode tiled --tile-size 16 --vector-width 4 \
  examples/matmul_lowering.json
```

| Representation | Scalar work | Logical bytes | Counted template instances |
| --- | ---: | ---: | ---: |
| High-level matmul | 48 | 104 | 1 graph operation |
| Scalar reduction | 48 | 224 | 88 |
| Tiled/vector model | 48 | 104 | 15 |

Scalar lowering reloads both operands per reduction term while retaining the
accumulator in a register. Tiled lowering assumes **ideal global input reuse**:
each input element is read once across all output tiles. Its bytes and roofline
total therefore equal the high-level model; tile/lane choices change
instruction-template counts, not measured runtime.

Tiles span M, N, and K, including ragged final tiles. Vectorization is **only
along N**, independently within each N tile: `divmod(tile_columns, vector_width)`
gives full-width vector operations and individual scalar-tail operations.
Vectors never cross tile boundaries; masked or partial-width vectors are not
counted. A width of 1 produces one-lane vectors and no tails, while a width
larger than an N tile produces only scalar tails. LHS values are scalar loads
broadcast into N lanes; RHS loads, initializations, FMAs, and stores have
separate vector and scalar-tail templates, including zero-count categories.
Cache feasibility, alignment, register pressure, instruction scheduling, and
generated machine-code quality are not modeled.

### Emit textual MLIR

```bash
PYTHONPATH=src python -m cost_model --emit-mlir examples/matmul_lowering.json
```

`emit_matmul_mlir(payload)` and `emit_matmul_mlir_graph(graph)` also expose this
operation in Python. The graph must contain exactly one rank-2 `matmul`, with
its result as the only graph output and its two f32 operands as the only graph
inputs. Stable function and SSA names accompany `module`, `func.func`,
`tensor.empty`, and `linalg.matmul`. The CLI prints raw MLIR; it cannot be combined
with `--benchmark`, `--optimize`, `--lower-matmul`, `--compact`, `--omit-paths`,
or `--budget`.

## Batch reports and CLI

```bash
# Baseline JSON; --compact produces one line
PYTHONPATH=src python -m cost_model --compact examples/matmul_bias_relu.json

# Select a batch case and omit its file-path field from stdout
PYTHONPATH=src python -m cost_model --benchmark benchmarks/target_bounds.json \
  --case compute_bound_matmul --omit-paths

PYTHONPATH=src python -m cost_model --help
```

A benchmark manifest is a non-empty JSON array of graph path strings or objects
containing exactly `name` and `path`. Relative graph paths resolve against the
manifest's directory; path-only entries use the graph's name. Effective case
names must be unique. Repeatable `--case` selects exact names in original manifest
order, but every case is loaded and validated before selection. A positional
graph and `--benchmark` are mutually exclusive.

Each case report appears under `cases[*].report`; `summary` aggregates normal
totals or `optimized.totals` with `--optimize`. Batch `estimated_time_us` sums
per-case roofline times, or is `null` if any selected case has no target.

`--budget` is batch-only and requires a non-empty `aggregate` object, with
optional per-case limits keyed by selected case names. Supported metrics are
`flops`, `logical_bytes`, and `estimated_time_us`. Limits are absolute upper
bounds: equality passes. Invalid budgets or unavailable requested metrics fail
closed. A violation keeps the report on stdout, emits a one-line `error: {...}`
JSON diagnostic on stderr, and exits with status `2`.

`--omit-paths` removes only `cases[*].path` from batch JSON stdout, in readable
or compact output, including when a budget fails. It leaves names, reports,
input files, validation, selection, optimization, and budget checks unchanged.
Error diagnostics are not redacted. This is not anonymization; review the
[privacy guidance](docs/privacy.md) before sharing.

See the [reference quickstart](docs/reference.md#quickstart) for manifest and
budget examples, [input format](docs/reference.md#input-format) for the base JSON
schema, and [output and cost definitions](docs/reference.md#cost-definitions).
The batched-matmul, scalar-tail, additional-operation, and MLIR sections below
summarize current behavior; the reference provides full input and cost definitions.

## Run tests and regression gates

The package has no runtime dependencies; pytest is a development extra.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q

mkdir -p outputs
python -m cost_model --benchmark benchmarks/target_bounds.json --compact \
  > outputs/target_bounds.actual.json
diff -u benchmarks/target_bounds.expected.json outputs/target_bounds.actual.json
python -m cost_model --benchmark benchmarks/target_bounds.json \
  --budget benchmarks/target_bounds.budget.json --compact
```

The batch includes one compute-bound matmul and one memory-bound ReLU. These are
model regression fixtures. CI runs the tests, exact fixture comparison, and budget
gate on Python 3.11 and 3.12.

## Design and scope

The parser processes operations in source order, so an operation can only use an
input or an earlier result. Supported operations are `matmul`, `add`, `mul`,
`relu`, last-axis `softmax`, last-axis `layer_norm`, and last-axis `reduce_sum`.
Elementwise `add` and `mul` use trailing-dimension broadcasting. Supported dtypes
are `f16`, `bf16`, `f32`, `f64`, `bool`, `i8`, `u8`, `i32`, and `i64`.

Softmax and layer norm preserve shape/dtype and require rank at least 1 with a
floating-point dtype. Layer norm has no affine scale/bias inputs or additional
attributes. For last-axis width `K`, softmax counts `5 × K - 2` scalar-operation
equivalents per row; layer norm counts `6 × K + 2`. Each exponential or square
root contributes one equivalent, not a literal hardware FLOP cost. Reduce sum
accepts numeric dtypes except `bool`, counts `K - 1` additions per row, preserves
dtype, and removes the last axis (a rank-1 input produces scalar shape `[]`).

| Layer | Responsibility |
| --- | --- |
| `models.py` | Tensor, operation, target and graph records |
| `parser.py` | Input validation, value lookup and shape inference |
| `estimator.py` | Baseline costs, roofline estimates and local fusion hints |
| `optimizer.py` | Liveness and legality-scoped fusion cost planning |
| `lowering.py` | Scalar and tiled/vector matmul representations |
| `mlir_emitter.py` | Deterministic standalone matmul MLIR emission |
| `cli.py` / `__main__.py` | Single/batch reports, selection, budgets and output controls |

The opt-in optimizer retains an unchanged `baseline`, reports `optimized` costs,
and emits signed after-minus-before `cost_delta` values. It removes operations
that cannot reach graph outputs. Fusion is restricted to adjacent source-order
`add`, `mul`, or `relu` pairs whose intermediate is not a graph output and has
exactly one use across live operations; repeated references count as repeated
uses. It never fuses across non-adjacent operations, matmul, graph outputs, or
multiple live uses. This is a cost plan, not execution or an IR rewrite.

Static positive dimensions and topologically ordered, single-output operations
keep the model inspectable. It has no symbolic shapes, side-effect/layout/alias
analysis, calibrated device model, numerical execution or general compiler
legality proof. Default fusion hints are only local opportunities; the opt-in
plan additionally protects graph outputs and accounts for liveness.

Logical traffic counts stored operands, not materialized broadcast copies or
measured DRAM traffic. Target ceilings omit latency, utilization, caches,
scheduling and device-specific instructions. The same scalar-work convention is
used for all supported dtypes. Inputs should be trusted and reasonably sized;
extreme numbers and huge tiled workloads are outside the robustness coverage
described in [privacy guidance](docs/privacy.md).

## License

Licensing for this project is undecided. No LICENSE file is included in the
current source tree.
