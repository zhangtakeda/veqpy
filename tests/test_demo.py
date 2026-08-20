from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from veqpy import Geqdsk

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
@pytest.mark.demo
def test_root_demo_runs_with_tmp_output(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not existing_pythonpath
        else os.pathsep.join((str(PROJECT_ROOT), existing_pythonpath))
    )

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "demo.py")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "VEQPy minimal Kernel demo" in result.stdout
    assert "success: True" in result.stdout
    assert (tmp_path / "demo_init.png").is_file()
    assert (tmp_path / "demo_result.png").is_file()
    assert (tmp_path / "demo_equilibrium.json").is_file()


@pytest.mark.slow
@pytest.mark.demo
def test_geqdsk_demo_writes_geqdsk_and_comparison(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not existing_pythonpath
        else os.pathsep.join((str(PROJECT_ROOT), existing_pythonpath))
    )

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "demo_geqdsk.py")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "VEQPy Numba GEQDSK demo" in result.stdout
    assert "success: True" in result.stdout
    assert "comparison:" in result.stdout

    output = PROJECT_ROOT / "data" / "solovev-veqpy.geqdsk"
    assert output.is_file()
    exported = Geqdsk(output)
    exported.check()
    reference = Geqdsk(PROJECT_ROOT / "data" / "SOLOVEV.geqdsk")
    assert (exported.NR, exported.NZ) == (reference.NR, reference.NZ)
    assert exported.boundary.shape == (65, 2)
    assert exported.boundary[-1].tolist() == exported.boundary[0].tolist()

    figure = PROJECT_ROOT / "data" / "solovev-veqpy-comparison.png"
    assert figure.is_file()
    assert figure.stat().st_size > 0
