from __future__ import annotations

import ast
import importlib
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("veqpy", "veqlib")
INTERNAL_CROSS_MODULE_IMPORTS = {
    "veqlib.facade.source_semantics",
    "veqlib.source_semantics",
}

PUBLIC_EXPORTS = {
    "veqpy.base": {"depends_on"},
    "veqpy.engine": {
        "COORDINATE_NAMES",
        "PSIN_COORDINATE",
        "RHO_AXIS",
        "RHO_COORDINATE",
        "THETA_AXIS",
        "full_differentiation",
        "full_integration",
    },
    "veqpy.math": {
        "barycentric_log_weights",
        "interpolation_matrix",
    },
    "veqpy.model": {
        "Boundary",
        "Geqdsk",
    },
    "veqpy.kernel": {
        "NumbaKernel",
    },
    "veqpy.facade": {
        "Kernel",
        "KernelBoundary",
        "KernelConfig",
        "KernelRecipe",
        "KernelSource",
        "KernelTopology",
        "NumbaKernel",
        "SolveResult",
        "build",
    },
    "veqlib.facade": {
        "CleanResult",
        "Kernel",
        "KernelBoundary",
        "KernelConfig",
        "KernelLoadError",
        "KernelRecipe",
        "KernelRegistry",
        "KernelSource",
        "KernelTopology",
        "LoadedKernel",
        "PrepareError",
        "PrepareResult",
        "SolveResult",
        "SolverThreadError",
        "TopologyError",
        "VEQlibSolver",
        "build",
        "clean",
        "prepare",
        "solve",
    },
}


def _source_files() -> list[Path]:
    return sorted(
        path
        for root in SOURCE_ROOTS
        for path in (REPO_ROOT / root).rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _module_name(path: Path) -> str:
    parts = path.relative_to(REPO_ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


PACKAGE_ROOTS = {
    _module_name(path)
    for root in SOURCE_ROOTS
    for path in (REPO_ROOT / root).rglob("__init__.py")
}


def _submodule_name(module: str) -> str | None:
    parts = module.split(".")
    if not parts or parts[0] not in SOURCE_ROOTS:
        return None
    if len(parts) < 2:
        return parts[0]
    return ".".join(parts[:2])


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _cross_package_root_imports() -> dict[str, set[str]]:
    imports: dict[str, set[str]] = defaultdict(set)
    for path in _source_files():
        importer = _module_name(path)
        importer_submodule = _submodule_name(importer)
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_submodule = _submodule_name(node.module)
                if imported_submodule is None or imported_submodule == importer_submodule:
                    continue
                if node.module in PACKAGE_ROOTS and node.module == imported_submodule:
                    imports[imported_submodule].update(
                        alias.name for alias in node.names if alias.name != "*"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_submodule = _submodule_name(alias.name)
                if imported_submodule is None or imported_submodule == importer_submodule:
                    continue
                if alias.name in PACKAGE_ROOTS and alias.name == imported_submodule:
                    imported_name = alias.asname or alias.name.rsplit(".", 1)[-1]
                    imports[imported_submodule].add(imported_name)
    return imports


def test_source_imports_respect_submodule_boundaries() -> None:
    violations: list[str] = []
    for path in _source_files():
        importer = _module_name(path)
        importer_submodule = _submodule_name(importer)
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_submodule = _submodule_name(node.module)
                if imported_submodule is None or importer_submodule is None:
                    continue
                if (
                    imported_submodule != importer_submodule
                    and node.module != imported_submodule
                    and node.module not in INTERNAL_CROSS_MODULE_IMPORTS
                ):
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno} imports "
                        f"{node.module} across submodule boundary"
                    )
                if (
                    imported_submodule == importer_submodule
                    and node.module == imported_submodule
                    and importer != imported_submodule
                ):
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno} imports same "
                        f"submodule through package root {node.module}"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_submodule = _submodule_name(alias.name)
                    if imported_submodule is None or importer_submodule is None:
                        continue
                    is_cross_leaf_import = (
                        imported_submodule != importer_submodule
                        and alias.name != imported_submodule
                    )
                    if is_cross_leaf_import:
                        violations.append(
                            f"{path.relative_to(REPO_ROOT)}:{node.lineno} imports "
                            f"{alias.name} across submodule boundary"
                        )
                    if (
                        imported_submodule == importer_submodule
                        and alias.name == imported_submodule
                        and importer != imported_submodule
                    ):
                        violations.append(
                            f"{path.relative_to(REPO_ROOT)}:{node.lineno} imports same "
                            f"submodule through package root {alias.name}"
                        )
    assert violations == []


def test_only_package_roots_declare_all() -> None:
    violations: list[str] = []
    for path in _source_files():
        if path.name == "__init__.py":
            continue
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            ):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "__all__"
            ):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert violations == []


def test_veqpy_all_exports_match_cross_contract_plus_public_api() -> None:
    cross_imports = _cross_package_root_imports()
    checked_packages = set(PUBLIC_EXPORTS) | set(cross_imports)
    for package_name in sorted(checked_packages):
        package = importlib.import_module(package_name)
        exported = set(package.__all__)
        required = cross_imports.get(package_name, set())
        allowed = required | PUBLIC_EXPORTS.get(package_name, set())
        assert required <= exported, package_name
        assert exported <= allowed, package_name
