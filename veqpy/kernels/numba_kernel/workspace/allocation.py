"""
Module: veqpy.kernels.numba_kernel.workspace.allocation

Role:
- Allocate kernel stage workspaces.
- Keep workspace construction coordinated from one entrypoint.

Public API:
- allocate_runtime_state

Notes:
- Workspace classes own memory for their respective stages.
- Kernel runtime and layout modules own orchestration and executable callables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from veqpy.kernels.numba_kernel.workspace.geometry_workspace import GeometryWorkspace
from veqpy.kernels.numba_kernel.workspace.profile_workspace import ProfileWorkspace
from veqpy.kernels.numba_kernel.workspace.residual_workspace import ResidualWorkspace
from veqpy.kernels.numba_kernel.workspace.source_workspace import SourceWorkspace

if TYPE_CHECKING:
    from veqpy.kernels.numba_kernel.backend_abi import SourceExecutionABI
    from veqpy.kernels.numba_kernel.workspace.grid_workspace import GridWorkspace


def allocate_runtime_state(
    *,
    grid_workspace: GridWorkspace,
    source_execution: SourceExecutionABI,
    profile_names: tuple[str, ...],
    profile_index: dict[str, int],
    active_profile_ids: np.ndarray,
    profile_L: np.ndarray,
    x_size: int,
) -> tuple[
    ProfileWorkspace,
    GeometryWorkspace,
    SourceWorkspace,
    ResidualWorkspace,
]:
    """Build kernel runtime state through stage workspace constructors."""

    nr = grid_workspace.Nr
    nt = grid_workspace.Nt
    m_max = grid_workspace.M_max

    profile_workspace = ProfileWorkspace(
        nr=nr,
        m_max=m_max,
        profile_names=profile_names,
        profile_index=profile_index,
        active_profile_ids=active_profile_ids,
        profile_L=profile_L,
        active_slot_capacity=len(profile_names),
        active_coeff_capacity=int(grid_workspace.L_max) + 1,
    )
    x_capacity = max(int(x_size), len(profile_names) * (int(grid_workspace.L_max) + 1))
    geometry_workspace = GeometryWorkspace(
        nr=nr,
        nt=nt,
    )
    source_workspace = SourceWorkspace(
        nr=nr,
        nt=nt,
        source_execution=source_execution,
    )
    residual_workspace = ResidualWorkspace(
        nr=nr,
        nt=nt,
        x_size=x_capacity,
        radial_weights=np.asarray(grid_workspace.weights, dtype=np.float64),
        active_residual_block_count=len(profile_names),
    )
    return (
        profile_workspace,
        geometry_workspace,
        source_workspace,
        residual_workspace,
    )
