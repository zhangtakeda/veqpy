from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = REPO_ROOT / "veqpy" / "kernels" / "cxx_kernel" / "core"
DRIVER_SOURCE = REPO_ROOT / "tests" / "cxx" / "nonlinear_reference_cases.cpp"

EXPECTED_CMINPACK_1_3_6 = {
    "powell_fd_rosenbrock": (1, 27, 0, (1.0, 1.0, 0.0)),
    "powell_jac_rosenbrock": (1, 23, 2, (1.0, 1.0, 0.0)),
    "lm_fd_rosenbrock": (2, 54, 0, (1.0, 1.0, 0.0)),
    "lm_jac_rosenbrock": (4, 21, 16, (1.0, 1.0, 0.0)),
    "powell_immediate": (1, 4, 0, (1.0, 1.0, 0.0)),
    "lm_immediate": (4, 1, 1, (1.0, 1.0, 0.0)),
    "powell_budget": (
        -1,
        3,
        0,
        (-1.1987999999999999, 1.0, 4.893080499109669),
    ),
    "lm_budget": (
        -1,
        3,
        0,
        (-1.1999999821186065, 1.0, 4.919349158656231),
    ),
    "powell_scaled": (5, 15, 0, (0.0, 0.0, 1.0e150)),
    "lm_rank_deficient": (4, 2, 2, (2.0, 0.0, 0.0)),
}


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
    for candidate in (Path("/usr/include/cminpack-1"), Path("/usr/local/include")):
        if (candidate / "cminpack.h").is_file():
            return candidate
    pytest.skip("CMINPACK headers are unavailable")


@pytest.fixture(scope="module")
def nonlinear_reference_driver(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("clang++ is unavailable")
    output = tmp_path_factory.mktemp("cxx-nonlinear-reference") / "reference-driver"
    command = [
        compiler,
        "-std=c++20",
        "-O2",
        "-I",
        str(CORE_DIR),
        "-I",
        str(_gcem_include()),
        "-I",
        str(_cminpack_include()),
        str(DRIVER_SOURCE),
        str(CORE_DIR / "nonlinear.cpp"),
        *(
            str(CORE_DIR / "minpack" / source)
            for source in (
                "dogleg.c",
                "dpmpar.c",
                "enorm.c",
                "fdjac1.c",
                "fdjac2.c",
                "hybrd.c",
                "hybrj.c",
                "lmder.c",
                "lmdif.c",
                "lmpar.c",
                "qform.c",
                "qrfac.c",
                "r1mpyq.c",
                "r1updt.c",
                "qrsolv.c",
            )
        ),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    return output


def _records(driver: Path) -> dict[str, tuple[int, int, int, tuple[float, ...]]]:
    completed = subprocess.run(
        [str(driver)],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    records: dict[str, tuple[int, int, int, tuple[float, ...]]] = {}
    for line in completed.stdout.splitlines():
        name, info, evaluations, jacobian_evaluations, *values = line.split()
        records[name] = (
            int(info),
            int(evaluations),
            int(jacobian_evaluations),
            tuple(float(value) for value in values),
        )
    return records


def test_cminpack_reference_cases_are_deterministic(nonlinear_reference_driver: Path) -> None:
    first = _records(nonlinear_reference_driver)
    second = _records(nonlinear_reference_driver)
    assert second == first
    assert first == EXPECTED_CMINPACK_1_3_6
