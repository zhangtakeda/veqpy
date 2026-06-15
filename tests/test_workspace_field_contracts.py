from __future__ import annotations

import numpy as np
from helpers import tiny_boundary, tiny_grid
from numpy.testing import assert_allclose

import veqpy.workspace.field_rows as rows
from veqpy.engine.backend_abi import SourceExecutionABI, build_fused_source_eval_abi
from veqpy.engine.numba_geometry import update_geometry_hot
from veqpy.operator import Operator, OperatorCase
from veqpy.workspace.geometry_workspace import GeometryWorkspace
from veqpy.workspace.grid_workspace import GridWorkspace
from veqpy.workspace.profile_workspace import ProfileWorkspace
from veqpy.workspace.residual_workspace import ResidualWorkspace
from veqpy.workspace.source_workspace import SourceWorkspace


def test_geometry_workspace_properties_alias_field_rows() -> None:
    workspace = GeometryWorkspace(nr=3, nt=2)
    workspace.surface_fields.fill(0.0)
    workspace.radial_fields.fill(0.0)

    surface_views = (
        (workspace.sin_tb_surface, rows.GEOMETRY_SURFACE_SIN_TB),
        (workspace.R_surface, rows.GEOMETRY_SURFACE_R),
        (workspace.R_t_surface, rows.GEOMETRY_SURFACE_R_T),
        (workspace.Z_t_surface, rows.GEOMETRY_SURFACE_Z_T),
        (workspace.J_surface, rows.GEOMETRY_SURFACE_J),
        (workspace.JdivR_surface, rows.GEOMETRY_SURFACE_JDIVR),
        (workspace.grtdivJR_t_surface, rows.GEOMETRY_SURFACE_GRTDIVJR_T),
        (workspace.gttdivJR_surface, rows.GEOMETRY_SURFACE_GTTDIVJR),
        (workspace.gttdivJR_r_surface, rows.GEOMETRY_SURFACE_GTTDIVJR_R),
    )
    for value, (view, row) in enumerate(surface_views, start=1):
        assert np.shares_memory(view, workspace.surface_fields[row])
        view.fill(float(value))
        assert_allclose(workspace.surface_fields[row], float(value))

    radial_views = (
        (workspace.S_r, rows.GEOMETRY_RADIAL_S_R),
        (workspace.V_r, rows.GEOMETRY_RADIAL_V_R),
        (workspace.Kn, rows.GEOMETRY_RADIAL_KN),
        (workspace.Kn_r, rows.GEOMETRY_RADIAL_KN_R),
        (workspace.Ln_r, rows.GEOMETRY_RADIAL_LN_R),
    )
    for value, (view, row) in enumerate(radial_views, start=10):
        assert np.shares_memory(view, workspace.radial_fields[row])
        view.fill(float(value))
        assert_allclose(workspace.radial_fields[row], float(value))


def test_residual_workspace_properties_alias_field_rows() -> None:
    radial_weights = np.ones(4, dtype=np.float64)
    workspace = ResidualWorkspace(nr=4, nt=3, x_size=5, radial_weights=radial_weights)
    workspace.root_fields.fill(0.0)
    workspace.surface_fields.fill(0.0)

    root_views = (
        (workspace.psin, rows.RESIDUAL_ROOT_PSIN),
        (workspace.psin_r, rows.RESIDUAL_ROOT_PSIN_R),
        (workspace.psin_rr, rows.RESIDUAL_ROOT_PSIN_RR),
        (workspace.FFn_psin, rows.RESIDUAL_ROOT_FFN_PSIN),
        (workspace.Pn_psin, rows.RESIDUAL_ROOT_PN_PSIN),
    )
    for value, (view, row) in enumerate(root_views, start=1):
        assert np.shares_memory(view, workspace.root_fields[row])
        view.fill(float(value))
        assert_allclose(workspace.root_fields[row], float(value))

    surface_views = (
        (workspace.G, rows.RESIDUAL_SURFACE_G),
        (workspace.Gpsin_R, rows.RESIDUAL_SURFACE_GPSIN_R),
        (workspace.Gpsin_Z, rows.RESIDUAL_SURFACE_GPSIN_Z),
        (workspace.Gpsin_R_sin_tb, rows.RESIDUAL_SURFACE_GPSIN_R_SIN_TB),
    )
    for value, (view, row) in enumerate(surface_views, start=10):
        assert np.shares_memory(view, workspace.surface_fields[row])
        view.fill(float(value))
        assert_allclose(workspace.surface_fields[row], float(value))


def test_source_workspace_scratch_names_are_vocabulary_aligned() -> None:
    source_execution = SourceExecutionABI(
        route_key=("PF", "rho", "grid"),
        psin_active_length=0,
        f_active_length=0,
        requires_optimized_psin_profile=False,
        requires_optimized_f_profile=False,
        requires_psin_query_workspace=False,
        requires_source_parameter_query=False,
        requires_target_root_fields=False,
    )
    workspace = SourceWorkspace(nr=4, nt=3, source_execution=source_execution)

    assert workspace.array_scratch.shape == (11, 4)
    assert workspace.matrix_scratch.shape == (1, 4, 3)
    assert not hasattr(workspace, "scratch_1d")
    assert not hasattr(workspace, "scratch_2d")


def test_grid_workspace_properties_alias_static_field_rows() -> None:
    workspace = GridWorkspace.from_grid(tiny_grid())

    assert np.shares_memory(workspace.rho, workspace.radial_fields[rows.GRID_RADIAL_RHO])
    assert np.shares_memory(workspace.x, workspace.radial_fields[rows.GRID_RADIAL_X])
    assert np.shares_memory(workspace.y, workspace.radial_fields[rows.GRID_RADIAL_Y])
    assert np.shares_memory(
        workspace.rho_powers,
        workspace.radial_fields[
            rows.GRID_RADIAL_RHO_POWERS_START : rows.GRID_RADIAL_RHO_POWERS_START
            + workspace.K_max
            + 2
        ],
    )
    assert np.shares_memory(workspace.theta, workspace.poloidal_fields[rows.GRID_POLOIDAL_THETA])
    assert np.shares_memory(
        workspace.cos_mtheta,
        workspace.poloidal_fields[
            rows.GRID_POLOIDAL_COS_MTHETA_START : rows.GRID_POLOIDAL_COS_MTHETA_START
            + workspace.M_max
            + 1
        ],
    )
    assert not workspace.radial_fields.flags.writeable
    assert not workspace.poloidal_fields.flags.writeable


def test_rho_source_eval_bindings_use_grid_radial_fields() -> None:
    grid = tiny_grid()

    def make_operator(route: str) -> Operator:
        current_input = np.ones(grid.Nr, dtype=np.float64)
        if route in {"PI", "PJ1", "PJ2"}:
            current_input = np.full(grid.Nr, 1.0e6, dtype=np.float64)
        profile_coeffs = {
            "h": [0.0, 0.0],
            "k": [0.0, 0.0],
            "s1": [0.0, 0.0],
        }
        if route == "PJ2":
            profile_coeffs["F"] = [0.0, 0.0]
        return Operator(
            grid,
            OperatorCase(
                route=route,
                coordinate="rho",
                nodes="grid",
                profile_coeffs=profile_coeffs,
                boundary=tiny_boundary(),
                heat_input=np.full(grid.Nr, 1.0e6, dtype=np.float64),
                current_input=current_input,
            ),
        )

    for route in ("PF", "PP", "PI", "PJ1", "PJ2", "PQ"):
        operator = make_operator(route)
        binding = build_fused_source_eval_abi(
            source_plan=operator.plan.source_plan,
            grid_workspace=operator.plan.grid_workspace,
            geometry_workspace=operator.geometry_workspace,
            source_workspace=operator.source_workspace,
            B0=operator.case.boundary.B0,
            fix_rho=operator.fix_rho,
        )

        assert binding.scratch_source_kernel is not None
        assert np.shares_memory(
            binding.grid_radial_fields,
            operator.plan.grid_workspace.radial_fields,
        )
        assert np.shares_memory(binding.rho, operator.plan.grid_workspace.rho)


def test_update_geometry_hot_accepts_grid_field_slabs() -> None:
    grid_workspace = GridWorkspace.from_grid(tiny_grid())
    geometry_workspace = GeometryWorkspace(nr=grid_workspace.Nr, nt=grid_workspace.Nt)
    h_fields = np.zeros((3, grid_workspace.Nr), dtype=np.float64)
    v_fields = np.zeros((3, grid_workspace.Nr), dtype=np.float64)
    k_fields = np.zeros((3, grid_workspace.Nr), dtype=np.float64)
    k_fields[rows.PROFILE_VALUE] = 1.0
    c_fields = np.zeros((grid_workspace.M_max + 1, 3, grid_workspace.Nr), dtype=np.float64)
    s_fields = np.zeros_like(c_fields)

    update_geometry_hot(
        geometry_workspace.surface_fields,
        geometry_workspace.radial_fields,
        0.5,
        1.5,
        0.0,
        grid_workspace.radial_fields,
        grid_workspace.poloidal_fields,
        h_fields,
        v_fields,
        k_fields,
        c_fields,
        s_fields,
        0,
        0,
    )

    assert np.all(np.isfinite(geometry_workspace.surface_fields))
    assert np.all(np.isfinite(geometry_workspace.radial_fields))


def test_profile_workspace_row_accessors_alias_profile_fields() -> None:
    workspace = ProfileWorkspace(
        nr=4,
        m_max=1,
        profile_names=("F",),
        profile_index={"F": 0},
        active_profile_ids=np.empty(0, dtype=np.int64),
        profile_L=np.array([0], dtype=np.int64),
    )
    workspace.profile_fields[0] = np.arange(12, dtype=np.float64).reshape(3, 4)

    assert np.shares_memory(
        workspace.values_for("F"), workspace.profile_fields[0, rows.PROFILE_VALUE]
    )
    assert np.shares_memory(
        workspace.radial_derivative_for("F"), workspace.profile_fields[0, rows.PROFILE_R]
    )
    assert np.shares_memory(
        workspace.radial_second_derivative_for("F"), workspace.profile_fields[0, rows.PROFILE_RR]
    )
