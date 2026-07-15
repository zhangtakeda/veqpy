from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MINPACK_ACKNOWLEDGEMENT = (
    "This product includes software developed by the University of Chicago, "
    "as Operator of Argonne National Laboratory."
)


def test_minpack_attribution_and_wheel_inputs_are_retained() -> None:
    readme = (REPO_ROOT / "README.md").read_text()
    license_text = (
        REPO_ROOT / "veqpy" / "kernels" / "cxx_kernel" / "core" / "minpack" / "CopyrightMINPACK.txt"
    ).read_text()
    manifest = (REPO_ROOT / "MANIFEST.in").read_text()
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        package_data = tomllib.load(stream)["tool"]["setuptools"]["package-data"]

    assert MINPACK_ACKNOWLEDGEMENT in " ".join(readme.split())
    assert MINPACK_ACKNOWLEDGEMENT in " ".join(license_text.split())
    assert "https://netlib.org/minpack/" in readme
    assert "https://github.com/devernay/cminpack/tree/v1.3.11" in readme
    assert "*.c" in manifest
    assert "*.txt" in manifest
    assert "core/minpack/*.c" in package_data["veqpy.kernels.cxx_kernel"]
    assert "core/minpack/*.h" in package_data["veqpy.kernels.cxx_kernel"]
    assert "core/minpack/*.txt" in package_data["veqpy.kernels.cxx_kernel"]
