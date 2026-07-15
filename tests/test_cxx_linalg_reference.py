from __future__ import annotations

import ctypes.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = REPO_ROOT / "veqpy" / "kernels" / "cxx_kernel" / "core"
DRIVER_SOURCE = REPO_ROOT / "tests" / "cxx" / "linalg_reference_cases.cpp"


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


@pytest.fixture(scope="module")
def linalg_reference_driver(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("clang++ is unavailable")
    if not _lapacke_available():
        pytest.skip("LAPACKE/LAPACK/OpenBLAS development libraries are unavailable")

    output = tmp_path_factory.mktemp("cxx-linalg-reference") / "reference-driver"
    command = [
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
        "-llapacke",
        "-llapack",
        "-lopenblas",
    ]
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    return output


def test_fixed_size_linalg_matches_lapacke_on_reference_cases(
    linalg_reference_driver: Path,
) -> None:
    completed = subprocess.run(
        [str(linalg_reference_driver)],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
    )
    rows = [line.split() for line in completed.stdout.splitlines()]
    assert len(rows) == 28
    for name, internal_error, lapack_error, delta in rows:
        assert name in {
            "doolittle",
            "cholesky",
            "bunch_kaufman",
            "householder",
            "doolittle_subnormal",
            "cholesky_lower_storage",
            "cholesky_non_positive",
            "bunch_kaufman_two_by_two",
            "bunch_kaufman_one_by_one_swap",
            "householder_subnormal",
            "golub_reinsch",
            "golub_reinsch_rank_deficient",
        }
        assert float(internal_error) <= 1.0e-11
        assert float(lapack_error) <= 1.0e-11
        assert float(delta) <= 1.0e-11
