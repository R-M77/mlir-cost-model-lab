# Handling private inputs and reports

The bundled fixtures are small synthetic graphs. Real graph descriptions can
reveal workload structure even when they contain no tensor values.

## What the program reads and emits

The CLI reads the graph, benchmark, and budget files you supply. Benchmark
entries can reference relative or absolute paths, including paths outside the
benchmark directory. Read and trust a benchmark manifest before running it.
All cases are loaded and validated before `--case` filtering.

The package has no runtime network calls or telemetry. It writes JSON to stdout
and diagnostics to stderr; shell redirection determines where reports are saved.
Reports include graph, operation, tensor and case names, shapes, dtypes, target
parameters and cost counts. Benchmark output also includes each path exactly as
written in the manifest, which can contain a username or private directory name.
Read failures and validation errors can expose file paths or input names.

## Keep local data out of commits

Use `private/` for your own graphs and manifests, and `outputs/` or `reports/`
for generated files. Those directories, `.env` files, common private-key file
extensions and logs are ignored. The checked-in examples and expected benchmark
JSON remain versioned so tests are reproducible.

```bash
mkdir -p outputs
PYTHONPATH=src python -m cost_model \
  --benchmark benchmarks/target_bounds.json --omit-paths \
  > outputs/target-bounds.json
```

`--omit-paths` removes only the `cases[*].path` fields from batch JSON stdout,
including when a budget is exceeded. It keeps cost calculations, case selection
and budget behavior unchanged. It does **not** redact names, tensor details,
targets, counts or error diagnostics, and it is rejected outside batch mode.
Treat it as a small output control, not an anonymization or security boundary.

`.gitignore` does not protect files already tracked, files saved elsewhere,
force-added files, Git history, terminal logs or CI artifacts. Before committing
or sharing, inspect `git status`, `git diff --cached`, the output itself and any
captured stderr. Use synthetic inputs for public examples. Do not include
credentials, employer/client graphs, datasets or artifacts unless you have the
right to share them.

## Input trust and modeling boundaries

Use this educational tool with trusted, reasonably sized inputs. It is not a
sandbox or a hardened service for arbitrary uploaded JSON. Extremely large
integers or extreme floating-point targets can overflow, underflow or produce
non-finite results; tiled enumeration can be expensive for huge shapes and small
tiles. Normal fixture coverage does not establish correctness for those cases.
Names and paths are not sanitized for display in another application: escape
them before embedding a report in HTML or other markup.
