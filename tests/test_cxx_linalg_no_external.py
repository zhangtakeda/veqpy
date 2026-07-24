from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = REPO_ROOT / "veqpy" / "kernels" / "cxx_kernel" / "core"
DRIVER_SOURCE = REPO_ROOT / "tests" / "cxx" / "linalg_no_external_cases.cpp"


def _gcem_include() -> Path:
    candidates = (
        Path(os.environ.get("VEQPY_CXX_GCEM_ROOT", "")) / "include",
        Path.home() / "opt" / "gcem-install" / "include",
        Path("/usr/local/include"),
        Path("/usr/include"),
    )
    for candidate in candidates:
        if candidate and (candidate / "gcem.hpp").is_file():
            return candidate
    pytest.skip("GCEM headers are unavailable")


def test_forced_fixed_size_linalg_compiles_and_runs_without_blas_or_lapack(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("clang++ is unavailable")

    output = tmp_path / "no-external-driver"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-DVEQPY_CXX_FORCE_INTERNAL_LINALG=1",
            "-I",
            str(CORE_DIR),
            "-isystem",
            str(_gcem_include()),
            str(DRIVER_SOURCE),
            "-o",
            str(output),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    completed = subprocess.run(
        [str(output)],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.stdout == "fixed-size linalg has no external runtime dependency\n"
