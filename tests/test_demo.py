from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

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
