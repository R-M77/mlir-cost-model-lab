# MLIR Cost Model Lab

A small, dependency-light Python 3.11+ project for making tensor graph costs explicit. It accepts a compact JSON graph inspired by MLIR's value-and-operation model and produces deterministic estimates that are easy to inspect in a code review.

This is a portfolio slice, not a hardware benchmark. It models static operation and tensor traffic counts; it does not claim to predict wall-clock runtime.

## Problem

Compiler and kernel decisions often need a quick first-order answer to questions such as:

- Which operation dominates scalar work?
- How much tensor data does each operation read and write?
- Which adjacent elementwise operations could share an intermediate?

`mlir-cost-model-lab` answers those questions for a deliberately small graph vocabulary without depending on MLIR, NumPy, or a runtime.

## Architecture

The package under `src/cost_model` is split into four small layers:

- `models.py` — immutable tensor, operation, and graph data structures.
- `parser.py` — JSON schema validation, value lookup, and static shape inference.
- `estimator.py` — FLOP, byte-traffic, arithmetic-intensity, and fusion calculations.
- `cli.py` / `__main__.py` — stable `python -m cost_model` interface and readable JSON output.

The parser processes operations in source order, so an operation can only use an input or an earlier result. Elementwise `add` and `mul` use trailing-dimension broadcasting; `matmul` is rank-2 only. The estimator preserves operation order and emits JSON-friendly integer counts plus floating-point intensity values.

## Quickstart

Run the example without installing anything beyond Python:

```bash
PYTHONPATH=src python -m cost_model examples/matmul_bias_relu.json
```

Use compact output for scripts:

```bash
PYTHONPATH=src python -m cost_model --compact examples/matmul_bias_relu.json
```

Install the project and test extras in a virtual environment when desired:

```bash
python -m pip install -e '.[test]'
pytest
```

The command exits with status `2` and prints a concise `error:` message for invalid JSON or invalid graph input.

## Input format

A graph has a name, named tensor inputs, a source-ordered list of single-output operations, and graph outputs:

```json
{
  "name": "matmul_bias_relu",
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

## Cost definitions

- **Matmul FLOPs:** `2 × M × K × N`, counting one multiply and one add per output reduction term.
- **Elementwise FLOPs:** one scalar operation per output element. `relu` counts its scalar comparison as one operation so it remains visible in the report; this is a modeling convention, not a hardware claim.
- **Bytes read:** the sum of the storage size of each input value referenced by the operation.
- **Bytes written:** the storage size of the operation's output value.
- **Arithmetic intensity:** `FLOPs / (bytes read + bytes written)`. Zero traffic would produce `0.0` rather than a division error.
- **Fusion candidates:** adjacent `add`, `mul`, or `relu` operations that share a single-use intermediate. `bytes_avoided` estimates the intermediate write plus the later read that a fusion could remove.

The report contains `ops`, `totals`, and a `fusion` section. Counts are exact for the model above; they are not measurements of a particular accelerator, cache, compiler, or runtime.

## Limitations

- Static, fully known positive dimensions only; no symbolic shapes or runtime shape guards.
- `matmul` accepts rank-2 tensors and requires matching dtypes.
- Elementwise broadcasting follows a small trailing-dimension rule; there is no layout or stride model.
- Every operation has exactly one output, and operations are already topologically ordered.
- Byte traffic is logical tensor storage traffic. It does not model cache reuse, tiling, alignment, transfers, or aliasing.
- Fusion reporting is a local opportunity heuristic, not a legality proof or an optimizer.
- The model intentionally omits latency, bandwidth, vectorization, parallelism, and device-specific instruction costs.

## Development

The project has no runtime dependencies. Focused tests cover exact estimates, JSON file loading, CLI behavior, and invalid schema/reference/shape inputs. CI runs the test suite on Python 3.11 and 3.12.
