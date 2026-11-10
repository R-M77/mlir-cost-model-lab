"""Behavioral coverage for the installed console entry point."""

import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_installed_cost_model_emits_deterministic_json() -> None:
    executable = Path(sys.executable).with_name("cost-model")
    example = REPO_ROOT / "examples" / "matmul_bias_relu.json"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    outputs = []
    for _ in range(2):
        result = subprocess.run(
            [str(executable), "--compact", str(example)],
            cwd=example.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stderr == ""
        report = json.loads(result.stdout)
        assert report["graph"] == "matmul_bias_relu"
        outputs.append(result.stdout)

    assert outputs[0] == outputs[1]
