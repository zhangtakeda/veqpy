from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_reference_lapack_attribution_and_wheel_input_are_retained() -> None:
    readme = (REPO_ROOT / "README.md").read_text()
    license_text = (
        REPO_ROOT / "veqpy" / "kernels" / "cxx_kernel" / "core" / "lapack" / "CopyrightLAPACK.txt"
    ).read_text()
    manifest = (REPO_ROOT / "MANIFEST.in").read_text()
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        package_data = tomllib.load(stream)["tool"]["setuptools"]["package-data"]

    assert "https://github.com/Reference-LAPACK/lapack/tree/v3.12.1" in readme
    assert "BSD-3-Clause" in readme
    assert "Copyright (c) 1992-2023 The University of Tennessee" in license_text
    assert "Redistribution and use in source and binary forms" in license_text
    assert "*.txt" in manifest
    assert "core/lapack/*.txt" in package_data["veqpy.kernels.cxx_kernel"]
