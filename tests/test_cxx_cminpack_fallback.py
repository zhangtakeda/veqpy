from __future__ import annotations

import ctypes.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = REPO_ROOT / "veqpy" / "kernels" / "cxx_kernel" / "core"
DRIVER_SOURCE = REPO_ROOT / "tests" / "cxx" / "cminpack_fallback_cases.cpp"
BRIDGE_SOURCE = CORE_DIR / "nonlinear.cpp"
MINPACK_SOURCES = (
    "dogleg.c",
    "dpmpar.c",
    "enorm.c",
    "lmpar.c",
    "qform.c",
    "qrfac.c",
    "r1mpyq.c",
    "r1updt.c",
    "qrsolv.c",
)


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


def _cminpack_include() -> Path:
    candidates = (Path("/usr/include/cminpack-1"), Path("/usr/local/include"), Path("/usr/include"))
    for candidate in candidates:
        if (candidate / "cminpack.h").is_file():
            return candidate
    pytest.skip("CMINPACK headers are unavailable")


def test_remote_cminpack_fallback_compiles_and_solves_all_driver_variants(tmp_path: Path) -> None:
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("clang++ is unavailable")
    if ctypes.util.find_library("cminpack") is None:
        pytest.skip("CMINPACK development library is unavailable")

    output = tmp_path / "cminpack-fallback"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-DVEQPY_CXX_CMINPACK_FALLBACK_MIN_DIMENSION=2",
            "-I",
            str(CORE_DIR),
            "-isystem",
            str(_gcem_include()),
            "-isystem",
            str(_cminpack_include()),
            str(DRIVER_SOURCE),
            str(BRIDGE_SOURCE),
            *(str(CORE_DIR / "minpack" / source) for source in MINPACK_SOURCES),
            "-o",
            str(output),
            "-lcminpack",
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
    records = {
        name: (int(info), int(evaluations), int(jacobian_evaluations), float(norm))
        for name, info, evaluations, jacobian_evaluations, norm in (
            line.split() for line in completed.stdout.splitlines()
        )
    }
    assert records.keys() == {"powell_fd", "powell_jac", "lm_fd", "lm_jac"}
    assert all(
        info > 0 and evaluations > 0 and norm <= 1.0e-10
        for info, evaluations, _, norm in records.values()
    )
    assert records["powell_jac"][2] > 0
    assert records["lm_jac"][2] > 0
