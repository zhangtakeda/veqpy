from __future__ import annotations

import importlib
import subprocess
import sys

import veqpy

ROOT_EXPORTS = {
    "Reactive",
    "Registry",
    "Serial",
    "depends_on",
    "read_serializer",
    "write_serializer",
    "Equilibrium",
    "Geqdsk",
    "Grid",
    "Profile",
    "Kernel",
    "KernelBoundary",
    "KernelConfig",
    "KernelRecipe",
    "KernelSource",
    "KernelTopology",
    "ParetoResult",
    "ParetoSample",
    "SolveResult",
    "build",
    "fit",
    "pareto",
    "solve",
}

KERNEL_EXPORTS = {
    "Kernel",
    "KernelBoundary",
    "KernelConfig",
    "KernelRecipe",
    "KernelSource",
    "KernelTopology",
    "ParetoResult",
    "ParetoSample",
    "SolveResult",
    "config_with_overrides",
}

MODEL_EXPORTS = {"Equilibrium", "Geqdsk", "Grid", "Profile"}

NUMERICS_EXPORTS = {
    "DEFAULT_CALCULUS",
    "DEFAULT_LOCAL_BARYCENTRIC_STENCIL",
    "DEFAULT_QUADRATURE",
    "R_AXIS",
    "SOURCE_INTERP_DEFAULT",
    "THETA_AXIS",
    "apply_accumulation",
    "apply_differentiation",
    "barycentric_log_weights",
    "build_uniform_source_interpolation_coefficients",
    "build_uniform_source_interpolation_matrix",
    "interpolation_matrix",
    "make_calculus",
    "make_quadrature",
    "normalize_source_interpolation_kind",
    "source_interpolation_kind_is_barycentric",
}


def test_package_roots_export_current_public_contracts() -> None:
    packages = {
        "veqpy": ROOT_EXPORTS,
        "veqpy.kernels": KERNEL_EXPORTS,
        "veqpy.model": MODEL_EXPORTS,
        "veqpy.numerics": NUMERICS_EXPORTS,
    }

    for package_name, expected in packages.items():
        package = importlib.import_module(package_name)
        exported = set(package.__all__)
        assert expected <= exported
        assert all(hasattr(package, name) for name in expected)


def test_api_module_exposes_function_entrypoints() -> None:
    api = importlib.import_module("veqpy.api")

    assert api.__all__ == ["build", "fit", "pareto", "solve"]
    assert api.build is veqpy.build
    assert api.fit is veqpy.fit
    assert api.pareto is veqpy.pareto
    assert api.solve is veqpy.solve


def test_direct_module_imports_are_available() -> None:
    modules = [
        "veqpy.api",
        "veqpy.base.reactive",
        "veqpy.base.serial",
        "veqpy.kernels.types",
        "veqpy.kernels.abi.source_semantics",
        "veqpy.kernels.cxx_kernel.builder",
        "veqpy.kernels.numba_kernel.packed_layout",
        "veqpy.model.profile",
        "veqpy.model.grid",
        "veqpy.numerics.interpolate",
    ]

    for module_name in modules:
        assert importlib.import_module(module_name)


def test_model_import_does_not_eagerly_load_matplotlib() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, veqpy; assert 'matplotlib' not in sys.modules",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
