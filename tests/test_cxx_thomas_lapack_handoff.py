from __future__ import annotations

import ctypes.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = REPO_ROOT / "veqpy" / "kernels" / "cxx_kernel" / "core"
DRIVER_SOURCE = REPO_ROOT / "tests" / "cxx" / "thomas_lapack_handoff.cpp"
BRIDGE_SOURCE = CORE_DIR / "linalg.cpp"


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


def _lapacke_available() -> bool:
    has_headers_or_library = Path("/usr/include/lapacke.h").is_file() or (
        ctypes.util.find_library("lapacke") is not None
    )
    return has_headers_or_library and all(
        ctypes.util.find_library(name) is not None for name in ("lapack", "openblas")
    )


def test_remote_thomas_lapack_handoff_compiles_and_solves(tmp_path: Path) -> None:
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("clang++ is unavailable")
    if not _lapacke_available():
        pytest.skip("LAPACKE/LAPACK/OpenBLAS development libraries are unavailable")

    output = tmp_path / "thomas-lapack-handoff"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-I",
            str(CORE_DIR),
            "-isystem",
            str(_gcem_include()),
            str(DRIVER_SOURCE),
            str(BRIDGE_SOURCE),
            "-o",
            str(output),
            "-llapacke",
            "-llapack",
            "-lopenblas",
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
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
    )
    assert completed.stdout == "Thomas LAPACKE handoff passed\n"
