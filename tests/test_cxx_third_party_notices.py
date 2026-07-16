from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTICE_PATH = REPO_ROOT / "veqpy" / "kernels" / "cxx_kernel" / "core" / "THIRD_PARTY_NOTICES"
MINPACK_ACKNOWLEDGEMENT = (
    "This product includes software developed by the University of Chicago, "
    "as Operator of Argonne National Laboratory."
)


def test_cxx_third_party_notices_and_distribution_inputs_are_retained() -> None:
    readme = (REPO_ROOT / "README.md").read_text()
    notices = NOTICE_PATH.read_text()
    minpack_license = (
        REPO_ROOT / "veqpy" / "kernels" / "cxx_kernel" / "core" / "minpack" / "CopyrightMINPACK.txt"
    ).read_text()
    lapack_license = (
        REPO_ROOT / "veqpy" / "kernels" / "cxx_kernel" / "core" / "lapack" / "CopyrightLAPACK.txt"
    ).read_text()
    manifest = (REPO_ROOT / "MANIFEST.in").read_text()
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        package_data = tomllib.load(stream)["tool"]["setuptools"]["package-data"]

    assert "### Code references" not in readme
    assert MINPACK_ACKNOWLEDGEMENT not in readme
    assert MINPACK_ACKNOWLEDGEMENT in notices
    assert "https://netlib.org/minpack/" in notices
    assert "https://github.com/devernay/cminpack/tree/v1.3.11" in notices
    assert "minpack/CopyrightMINPACK.txt" in notices
    assert "https://github.com/Reference-LAPACK/lapack/tree/v3.12.1" in notices
    assert "BSD 3-Clause" in notices
    assert "lapack/CopyrightLAPACK.txt" in notices
    assert MINPACK_ACKNOWLEDGEMENT in " ".join(minpack_license.split())
    assert "Copyright (c) 1992-2023 The University of Tennessee" in lapack_license
    assert "Redistribution and use in source and binary forms" in lapack_license
    assert "core/THIRD_PARTY_NOTICES" in manifest
    assert "core/THIRD_PARTY_NOTICES" in package_data["veqpy.kernels.cxx_kernel"]
    assert "core/minpack/*.c" in package_data["veqpy.kernels.cxx_kernel"]
    assert "core/minpack/*.txt" in package_data["veqpy.kernels.cxx_kernel"]
    assert "core/lapack/*.txt" in package_data["veqpy.kernels.cxx_kernel"]
