from __future__ import annotations

from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "docs" / "jax_static_dynamic_manifest.md"


def _manifest_text() -> str:
    return MANIFEST_PATH.read_text(encoding="utf-8")


def test_jax_static_dynamic_manifest_exists() -> None:
    assert MANIFEST_PATH.exists()


def test_jax_static_dynamic_manifest_required_sections() -> None:
    text = _manifest_text()
    required_sections = (
        "## Scope",
        "## Categories",
        "## OperatorBuildPlan Fields",
        "## Workspace-Derived JAX Lowering Arrays",
        "## Route And Capability Keys",
        "## Dynamic Inputs And Outputs",
        "## Backend Options",
        "## Host Publication And Unsupported State",
        "## Public Operator Method Matrix",
        "## Unsupported Or Not Yet Lowered",
    )

    for section in required_sections:
        assert section in text


def test_jax_static_dynamic_manifest_does_not_mark_large_arrays_static() -> None:
    text = _manifest_text()
    large_arrays = (
        "grid_workspace.radial_fields",
        "grid_workspace.poloidal_fields",
        "grid_workspace.weights",
        "grid_workspace.differentiator",
        "grid_workspace.accumulator",
        "profile_workspace.profile_fields",
        "profile_workspace.profile_rp_fields",
        "profile_workspace.profile_env_fields",
        "profile_workspace.c_family_fields",
        "profile_workspace.s_family_fields",
        "geometry_workspace.surface_fields",
        "geometry_workspace.radial_fields",
        "source_workspace.array_scratch",
        "source_workspace.matrix_scratch",
        "residual_workspace.root_fields",
        "residual_workspace.surface_fields",
        "residual_workspace.pack_scratch",
    )

    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip(" `") for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        name, category = cells[0], cells[1].lower()
        if name in large_arrays:
            assert category != "static metadata"


def test_jax_static_dynamic_manifest_public_operator_method_matrix() -> None:
    text = _manifest_text()
    methods = (
        "residual_var",
        "residual_var_into",
        "residual_collocation",
        "residual_collocation_into",
        "stage_a_profile",
        "stage_b_geometry",
        "stage_c_source",
        "stage_d_residual",
        "build_equilibrium",
        "replace_problem",
        "replace_case",
        "alpha1",
        "alpha2",
    )

    for method in methods:
        assert f"`{method}`" in text
