# MLIR Cost Model Lab

**Explain tensor graph costs, compare fusion plans, and inspect lowering tradeoffs in pure Python.**

A Python tool with no runtime dependencies for a small, statically shaped JSON
graph vocabulary.
It validates value references and shapes, counts scalar work and logical tensor
traffic, and emits deterministic reports that can be reviewed and regression-tested.

This is an educational compiler-modeling project inspired by MLIR's values and
operations. It does not parse MLIR, compile kernels, execute graphs, or measure
hardware performance. Roofline times are idealized model estimates based on the
ceilings you supply.

## What it demonstrates

- **Graph validation and shape inference:** source-ordered values, rank-2 matmul,
  trailing-dimension broadcasting, dtype and reference checks
- **Explainable cost accounting:** per-operation and aggregate work, reads,
  writes, arithmetic intensity, and optional roofline classification
- **Liveness-aware optimization planning:** remove dead operations from the
  modeled plan and account for adjacent single-consumer elementwise fusion,
  preserving graph outputs and an unchanged baseline report
- **Lowering representations:** compare one matmul with counted scalar or
  tiled/vector templates, with assumptions and signed cost deltas in the output
- **Deterministic regression checks:** batch cases, exact JSON fixtures, case
  selection, and aggregate/per-case budget gates with nonzero failure status

## Run a reproducible example

Requires Python 3.11+. From the repository root, no installation is needed:

```bash
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

Scalar lowering reloads both operands per reduction term. Tiled lowering assumes
**ideal global input reuse**: each input element is read once across all output
tiles. Its bytes and roofline total therefore equal the high-level model;
tile/lane choices change instruction-template counts. This does not establish
cache feasibility, generated machine-code quality, or runtime gains.

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
gate on Python 3.11 and 3.12. Budget equality passes; exceeding any supplied limit
returns status `2`, keeps the report on stdout, and writes a diagnostic to stderr.

## Explore the interface

```bash
# Baseline JSON; --compact produces one line
PYTHONPATH=src python -m cost_model --compact examples/matmul_bias_relu.json

# Select a batch case and omit its file-path field from stdout
PYTHONPATH=src python -m cost_model --benchmark benchmarks/target_bounds.json \
  --case compute_bound_matmul --omit-paths

PYTHONPATH=src python -m cost_model --help
```

Supported operations: `matmul`, `add`, `mul`, `relu`. Supported dtypes:
`f16`, `bf16`, `f32`, `f64`, `bool`, `i8`, `u8`, `i32`, `i64`.

- [Full reference](docs/reference.md): input schema, CLI options, output fields,
  formulas, optimization legality and lowering assumptions
- [Privacy guidance](docs/privacy.md): private inputs, generated reports and the
  narrow scope of `--omit-paths`

## Design and scope

| Layer | Responsibility |
| --- | --- |
| `models.py` | Tensor, operation, target and graph records |
| `parser.py` | Input validation, value lookup and shape inference |
| `estimator.py` | Baseline costs, roofline estimates and local fusion hints |
| `optimizer.py` | Liveness and legality-scoped fusion cost planning |
| `lowering.py` | Scalar and tiled/vector matmul representations |
| `cli.py` | Single/batch reports, selection, budget gates and output controls |

Static positive dimensions and topologically ordered, single-output operations
keep the model inspectable. It has no symbolic shapes, side-effect/layout/alias
analysis, calibrated device model, numerical execution or general compiler
legality proof. Default fusion hints are only local opportunities; the opt-in
plan additionally protects graph outputs and accounts for liveness.

Logical traffic is not measured DRAM traffic. Target ceilings omit latency,
utilization, caches, scheduling and device-specific instructions. The same
scalar-work convention is used for all supported dtypes. Inputs should be trusted
and reasonably sized; extreme numbers and huge tiled workloads are outside the
robustness coverage described in [privacy guidance](docs/privacy.md).

## License

The project metadata declares MIT. The copyright holder and a corresponding
license file should be confirmed before distribution.
