from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_example(
    script_name: str,
    output_dir: Path,
    *,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["VEQPY_OUTPUT_DIR"] = str(output_dir)
    env.setdefault("MPLCONFIGDIR", str(output_dir / "mplconfig"))
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples" / script_name)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


@pytest.mark.slow
@pytest.mark.examples
def test_minimal_equilibrium_example_runs_with_tmp_output(tmp_path: Path) -> None:
    outdir = tmp_path / "minimal"
    result = _run_example("minimal_equilibrium.py", outdir)

    assert result.returncode == 0, result.stderr + result.stdout
    assert (outdir / "demo_flux_surfaces.png").is_file()
    assert (outdir / "demo_equilibrium.json").is_file()


@pytest.mark.slow
@pytest.mark.examples
def test_geqdsk_workflow_example_runs_with_tmp_output(tmp_path: Path) -> None:
    outdir = tmp_path / "geqdsk"
    result = _run_example("geqdsk_workflow.py", outdir)

    assert result.returncode == 0, result.stderr + result.stdout
    assert (outdir / "demo_geqdsk_workflow.png").is_file()
    assert (outdir / "demo_geqdsk_equilibrium.json").is_file()
