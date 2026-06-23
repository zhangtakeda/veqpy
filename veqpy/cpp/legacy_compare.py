from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class LegacyCompareConfig:
    repeat: int = 1
    warmup: int = 0
    cxx_backend: str = "nanobind"
    module_dir: Path | None = None


def benchmark_legacy_veqpy_comparison(
    *,
    config: LegacyCompareConfig | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run the existing VEQPy-vs-VEQlib PF/psin/uniform comparison script.

    This is the P5 bridge while the new ``VEQlibSolver`` is still an MVP wrapper around the
    benchmark PF kernel. It keeps legacy VEQPy numerical and latency evidence in the same Python
    package surface as the new lifecycle benchmark.
    """

    config = config or LegacyCompareConfig()
    repo_root = _default_repo_root() if repo_root is None else repo_root.resolve()
    script = repo_root / "veqlib" / "benchmark_pf_psin_uniform_compare.py"
    if not script.exists():
        raise FileNotFoundError(f"legacy comparison script not found: {script}")

    command = [
        sys.executable,
        str(script),
        "--cxx-backend",
        config.cxx_backend,
        "--repeat",
        str(config.repeat),
        "--warmup",
        str(config.warmup),
        "--no-write",
        "--quiet",
    ]
    if config.module_dir is not None:
        command.extend(["--module-dir", str(config.module_dir)])
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "legacy VEQPy comparison failed with exit code "
            f"{completed.returncode}: {completed.stderr or completed.stdout}"
        )
    report = json.loads(completed.stdout)
    if not isinstance(report, dict):
        raise RuntimeError("legacy VEQPy comparison did not return a JSON object")
    return report


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
