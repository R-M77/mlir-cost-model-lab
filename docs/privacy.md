# Privacy and benchmark paths

Benchmark JSON output includes the original input path in each `cases[*].path` field. Paths can expose local usernames, checkout or project names, and private filenames.

`python -m cost_model --benchmark benchmark.json --omit-paths` removes exactly those `cases[*].path` fields from JSON stdout. It leaves case names, graph reports, all other fields, and input files unchanged. Validation and budget checks still run normally. Error diagnostics are not redacted and may include paths.

This is not anonymization. Other report content—including case and graph names—may identify people, projects, or systems. Review the complete output before sharing it.
