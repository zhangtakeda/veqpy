"""
Module: engine.numba_residual

Role:
- Update residual surface workspace.
- Assemble precomputed residual fields into a packed residual.

Public API:
- update_residual_compact
- write_weighted_scaled_g_collocation_field_into

Notes:
- Keep only the minimal interface required by the numba hot path.
- The old staged/binder residual API has been removed.
- Packed residual block codes are layout ABI, not local magic numbers:
  0=h, 1=v, 2=k, 3=c0, 4=c_m, 5=s_m, 6=psin, 7=F.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from veqpy.model.numerics import (
    indexed_matvec_into,
    rowwise_sum_into,
    rowwise_weighted_sum_into,
)
from veqpy.model.numerics import (
    GEOMETRY_SURFACE_GRTDIVJR_T,
    GEOMETRY_SURFACE_GTTDIVJR,
    GEOMETRY_SURFACE_GTTDIVJR_R,
    GEOMETRY_SURFACE_J,
    GEOMETRY_SURFACE_JDIVR,
    GEOMETRY_SURFACE_R,
    GEOMETRY_SURFACE_R_T,
    GEOMETRY_SURFACE_SIN_TB,
    GEOMETRY_SURFACE_Z_T,
    GRID_POLOIDAL_COS_MTHETA_START,
    GRID_RADIAL_RHO_POWERS_START,
    GRID_RADIAL_Y,
    RESIDUAL_ROOT_FFN_PSIN,
    RESIDUAL_ROOT_PN_PSIN,
    RESIDUAL_ROOT_PSIN_R,
    RESIDUAL_ROOT_PSIN_RR,
    RESIDUAL_SURFACE_G,
    RESIDUAL_SURFACE_GPSIN_R,
    RESIDUAL_SURFACE_GPSIN_R_SIN_TB,
    RESIDUAL_SURFACE_GPSIN_Z,
)


@njit(cache=True, inline="always")
def _residual_grid_radial_views(
    grid_radial_fields: np.ndarray,
    grid_k_max: int,
    grid_l_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho_start = GRID_RADIAL_RHO_POWERS_START
    rho_powers = grid_radial_fields[rho_start : rho_start + grid_k_max + 2]
    T_start = rho_start + grid_k_max + 2
    T = grid_radial_fields[T_start : T_start + grid_l_max + 1]
    y = grid_radial_fields[GRID_RADIAL_Y]
    return rho_powers, y, T


@njit(cache=True, inline="always")
def _residual_grid_poloidal_views(
    grid_poloidal_fields: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    block = (grid_poloidal_fields.shape[0] - GRID_POLOIDAL_COS_MTHETA_START) // 6
    cos_start = GRID_POLOIDAL_COS_MTHETA_START
    sin_start = cos_start + block
    cos_mtheta = grid_poloidal_fields[cos_start:sin_start]
    sin_mtheta = grid_poloidal_fields[sin_start : sin_start + block]
    return sin_mtheta, cos_mtheta


@njit(cache=True, fastmath=True, nogil=True)
def update_residual_compact(
    out_workspace: np.ndarray,
    alpha1: float,
    alpha2: float,
    root_fields: np.ndarray,
    geometry_surface_fields: np.ndarray,
) -> None:
    """Update residual-related 2D fields in place from compact geometry fields."""
    out_G = out_workspace[RESIDUAL_SURFACE_G]
    out_Gpsin_R = out_workspace[RESIDUAL_SURFACE_GPSIN_R]
    out_Gpsin_Z = out_workspace[RESIDUAL_SURFACE_GPSIN_Z]
    out_Gpsin_R_sin_tb = out_workspace[RESIDUAL_SURFACE_GPSIN_R_SIN_TB]
    sin_tb_surface = geometry_surface_fields[GEOMETRY_SURFACE_SIN_TB]
    R_surface = geometry_surface_fields[GEOMETRY_SURFACE_R]
    R_t_surface = geometry_surface_fields[GEOMETRY_SURFACE_R_T]
    Z_t_surface = geometry_surface_fields[GEOMETRY_SURFACE_Z_T]
    J_surface = geometry_surface_fields[GEOMETRY_SURFACE_J]
    JdivR_surface = geometry_surface_fields[GEOMETRY_SURFACE_JDIVR]
    grtdivJR_t_surface = geometry_surface_fields[GEOMETRY_SURFACE_GRTDIVJR_T]
    gttdivJR_surface = geometry_surface_fields[GEOMETRY_SURFACE_GTTDIVJR]
    gttdivJR_r_surface = geometry_surface_fields[GEOMETRY_SURFACE_GTTDIVJR_R]

    psin_r = root_fields[RESIDUAL_ROOT_PSIN_R]
    psin_rr = root_fields[RESIDUAL_ROOT_PSIN_RR]
    FFn_psin = root_fields[RESIDUAL_ROOT_FFN_PSIN]
    Pn_psin = root_fields[RESIDUAL_ROOT_PN_PSIN]

    nr, nt = out_G.shape
    for i in range(nr):
        psin_r_i = psin_r[i]
        psin_rr_i = psin_rr[i]
        FFn_psin_i = FFn_psin[i]
        Pn_psin_i = Pn_psin[i]
        for j in range(nt):
            inv_J = 1.0 / J_surface[i, j]
            psin_R = -Z_t_surface[i, j] * inv_J * psin_r_i
            psin_Z = R_t_surface[i, j] * inv_J * psin_r_i

            R_ij = R_surface[i, j]
            G1n = JdivR_surface[i, j] * (FFn_psin_i + R_ij * R_ij * Pn_psin_i)
            G2n = (
                gttdivJR_surface[i, j] * psin_rr_i
                + (gttdivJR_r_surface[i, j] - grtdivJR_t_surface[i, j]) * psin_r_i
            )
            G_ij = alpha1 * G1n + alpha2 * G2n
            out_G[i, j] = G_ij
            # Variational residual blocks project G against shape derivatives.
            # Cache the repeated G*grad(psin) products once per surface point.
            Gpsin_R = G_ij * psin_R
            out_Gpsin_R[i, j] = Gpsin_R
            out_Gpsin_Z[i, j] = G_ij * psin_Z
            out_Gpsin_R_sin_tb[i, j] = Gpsin_R * sin_tb_surface[i, j]


@njit(cache=True, fastmath=True, nogil=True)
def _project_scaled2(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    T: np.ndarray,
    collapsed: np.ndarray,
    weight_a: np.ndarray,
    weight_b: np.ndarray,
    scalar: float,
) -> None:
    # ``collapsed`` is scratch owned by the caller.  Scale in place, then project
    # onto the active coefficient basis indices for this residual block.
    for i in range(collapsed.shape[0]):
        collapsed[i] *= weight_a[i] * weight_b[i] * scalar
    indexed_matvec_into(out_packed, coeff_indices, T, collapsed)


@njit(cache=True, fastmath=True, nogil=True)
def _project_scaled3(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    T: np.ndarray,
    collapsed: np.ndarray,
    weight_a: np.ndarray,
    weight_b: np.ndarray,
    weight_c: np.ndarray,
    scalar: float,
) -> None:
    for i in range(collapsed.shape[0]):
        collapsed[i] *= weight_a[i] * weight_b[i] * weight_c[i] * scalar
    indexed_matvec_into(out_packed, coeff_indices, T, collapsed)


@njit(cache=True, fastmath=True, nogil=True)
def _copy_row_into(out: np.ndarray, row: np.ndarray) -> None:
    for i in range(out.shape[0]):
        out[i] = row[i]


@njit(cache=True, fastmath=True, nogil=True)
def run_residual_blocks_packed_precomputed(
    out_packed: np.ndarray,
    scratch: np.ndarray,
    block_codes: np.ndarray,
    block_orders: np.ndarray,
    block_radial_powers: np.ndarray,
    coeff_index_rows: np.ndarray,
    lengths: np.ndarray,
    residual_workspace: np.ndarray,
    grid_radial_fields: np.ndarray,
    grid_poloidal_fields: np.ndarray,
    grid_k_max: int,
    grid_l_max: int,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    G = residual_workspace[RESIDUAL_SURFACE_G]
    Gpsin_R = residual_workspace[RESIDUAL_SURFACE_GPSIN_R]
    Gpsin_Z = residual_workspace[RESIDUAL_SURFACE_GPSIN_Z]
    Gpsin_R_sin_tb = residual_workspace[RESIDUAL_SURFACE_GPSIN_R_SIN_TB]
    rho_powers, y, T = _residual_grid_radial_views(
        grid_radial_fields, grid_k_max, grid_l_max
    )
    sin_mtheta, cos_mtheta = _residual_grid_poloidal_views(grid_poloidal_fields)
    sin_theta = sin_mtheta[1]
    rho = rho_powers[1]
    rho2 = rho_powers[2]
    nt = G.shape[1]
    base_scale = 2.0 * np.pi / nt
    for slot in range(block_codes.shape[0]):
        # block_codes are packed-layout metadata: 0/1/2/3/4/5 project shape
        # families, while 6/7 project psin and F source-prefix profiles.
        coeff_indices = coeff_index_rows[slot, : lengths[slot]]
        code = block_codes[slot]
        order = block_orders[slot]
        radial_power = block_radial_powers[slot]
        if code == 0:
            # h and v are low-order shape translations; they project G*grad(psin)
            # against the edge envelope y and the radial Chebyshev basis.
            rowwise_sum_into(scratch, Gpsin_R)
            _project_scaled2(out_packed, coeff_indices, T, scratch, y, weights, base_scale * a)
        elif code == 1:
            rowwise_sum_into(scratch, Gpsin_Z)
            _project_scaled2(out_packed, coeff_indices, T, scratch, y, weights, base_scale * a)
        elif code == 2:
            # k modifies vertical elongation through rho*sin(theta), hence the
            # extra radial rho factor and theta sine weighting.
            rowwise_weighted_sum_into(scratch, Gpsin_Z, sin_theta)
            _project_scaled3(
                out_packed, coeff_indices, T, scratch, rho, y, weights, base_scale * (-a)
            )
        elif code == 3:
            # c0 is the axisymmetric theta_bar shift.  It uses the c-family
            # residual form but no explicit Fourier cosine factor.
            rowwise_sum_into(scratch, Gpsin_R_sin_tb)
            _project_scaled3(
                out_packed, coeff_indices, T, scratch, rho, y, weights, base_scale * (-a)
            )
        elif code == 4:
            # Higher cosine/sine shape modes carry their regularity power through
            # block_radial_powers; residual projection uses power+1 because the
            # boundary variation contributes one additional rho factor.
            rowwise_weighted_sum_into(scratch, Gpsin_R_sin_tb, cos_mtheta[order])
            _project_scaled3(
                out_packed,
                coeff_indices,
                T,
                scratch,
                rho_powers[radial_power + 1],
                y,
                weights,
                base_scale * (-a),
            )
        elif code == 5:
            rowwise_weighted_sum_into(scratch, Gpsin_R_sin_tb, sin_mtheta[order])
            _project_scaled3(
                out_packed,
                coeff_indices,
                T,
                scratch,
                rho_powers[radial_power + 1],
                y,
                weights,
                base_scale * (-a),
            )
        elif code == 6:
            # psin coefficients project the strong-form G block itself, with
            # rho**2 regularity matching the psin profile convention.
            rowwise_sum_into(scratch, G)
            _project_scaled3(out_packed, coeff_indices, T, scratch, rho2, y, weights, base_scale)
        elif code == 7:
            # F is represented by normalized F**2 profile coefficients.  The
            # projection scale restores the physical edge magnitude (R0*B0)**2.
            rowwise_sum_into(scratch, G)
            _project_scaled3(
                out_packed,
                coeff_indices,
                T,
                scratch,
                y,
                y,
                weights,
                base_scale * (R0 * B0) * (R0 * B0),
            )
        else:
            raise ValueError("Unknown residual block code")


@njit(cache=True, fastmath=True, nogil=True)
def run_residual_blocks_packed_precomputed_auto(
    out_packed: np.ndarray,
    scratch: np.ndarray,
    scratch_rows: np.ndarray,
    block_codes: np.ndarray,
    block_orders: np.ndarray,
    block_radial_powers: np.ndarray,
    coeff_index_rows: np.ndarray,
    lengths: np.ndarray,
    residual_workspace: np.ndarray,
    grid_radial_fields: np.ndarray,
    grid_poloidal_fields: np.ndarray,
    grid_k_max: int,
    grid_l_max: int,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    block_count = block_codes.shape[0]
    fourier_count = 0
    need_gpsin_r = False
    need_gpsin_z = False
    need_k = False
    need_c0 = False
    need_g = False
    for slot in range(block_count):
        code = block_codes[slot]
        if code == 0:
            need_gpsin_r = True
        elif code == 1:
            need_gpsin_z = True
        elif code == 2:
            need_k = True
        elif code == 3:
            need_c0 = True
        elif code == 4 or code == 5:
            fourier_count += 1
        elif code == 6 or code == 7:
            need_g = True

    if block_count < 8 and fourier_count < 4:
        run_residual_blocks_packed_precomputed(
            out_packed,
            scratch,
            block_codes,
            block_orders,
            block_radial_powers,
            coeff_index_rows,
            lengths,
            residual_workspace,
            grid_radial_fields,
            grid_poloidal_fields,
            grid_k_max,
            grid_l_max,
            weights,
            a,
            R0,
            B0,
        )
        return

    G = residual_workspace[RESIDUAL_SURFACE_G]
    Gpsin_R = residual_workspace[RESIDUAL_SURFACE_GPSIN_R]
    Gpsin_Z = residual_workspace[RESIDUAL_SURFACE_GPSIN_Z]
    Gpsin_R_sin_tb = residual_workspace[RESIDUAL_SURFACE_GPSIN_R_SIN_TB]
    rho_powers, y, T = _residual_grid_radial_views(
        grid_radial_fields, grid_k_max, grid_l_max
    )
    sin_mtheta, cos_mtheta = _residual_grid_poloidal_views(grid_poloidal_fields)
    sin_theta = sin_mtheta[1]
    rho = rho_powers[1]
    rho2 = rho_powers[2]
    nt = G.shape[1]
    base_scale = 2.0 * np.pi / nt

    if need_gpsin_r:
        rowwise_sum_into(scratch_rows[0], Gpsin_R)
    if need_gpsin_z:
        rowwise_sum_into(scratch_rows[1], Gpsin_Z)
    if need_k:
        rowwise_weighted_sum_into(scratch_rows[2], Gpsin_Z, sin_theta)
    if need_c0:
        rowwise_sum_into(scratch_rows[3], Gpsin_R_sin_tb)
    if need_g:
        rowwise_sum_into(scratch_rows[4], G)

    for slot in range(block_count):
        code = block_codes[slot]
        order = block_orders[slot]
        if code == 4:
            rowwise_weighted_sum_into(scratch_rows[5 + slot], Gpsin_R_sin_tb, cos_mtheta[order])
        elif code == 5:
            rowwise_weighted_sum_into(scratch_rows[5 + slot], Gpsin_R_sin_tb, sin_mtheta[order])

    for slot in range(block_count):
        coeff_indices = coeff_index_rows[slot, : lengths[slot]]
        code = block_codes[slot]
        radial_power = block_radial_powers[slot]
        if code == 0:
            _copy_row_into(scratch, scratch_rows[0])
            _project_scaled2(out_packed, coeff_indices, T, scratch, y, weights, base_scale * a)
        elif code == 1:
            _copy_row_into(scratch, scratch_rows[1])
            _project_scaled2(out_packed, coeff_indices, T, scratch, y, weights, base_scale * a)
        elif code == 2:
            _copy_row_into(scratch, scratch_rows[2])
            _project_scaled3(
                out_packed, coeff_indices, T, scratch, rho, y, weights, base_scale * (-a)
            )
        elif code == 3:
            _copy_row_into(scratch, scratch_rows[3])
            _project_scaled3(
                out_packed, coeff_indices, T, scratch, rho, y, weights, base_scale * (-a)
            )
        elif code == 4:
            _copy_row_into(scratch, scratch_rows[5 + slot])
            _project_scaled3(
                out_packed,
                coeff_indices,
                T,
                scratch,
                rho_powers[radial_power + 1],
                y,
                weights,
                base_scale * (-a),
            )
        elif code == 5:
            _copy_row_into(scratch, scratch_rows[5 + slot])
            _project_scaled3(
                out_packed,
                coeff_indices,
                T,
                scratch,
                rho_powers[radial_power + 1],
                y,
                weights,
                base_scale * (-a),
            )
        elif code == 6:
            _copy_row_into(scratch, scratch_rows[4])
            _project_scaled3(out_packed, coeff_indices, T, scratch, rho2, y, weights, base_scale)
        elif code == 7:
            _copy_row_into(scratch, scratch_rows[4])
            _project_scaled3(
                out_packed,
                coeff_indices,
                T,
                scratch,
                y,
                y,
                weights,
                base_scale * (R0 * B0) * (R0 * B0),
            )
        else:
            raise ValueError("Unknown residual block code")

@njit(cache=True, fastmath=True, nogil=True)
def write_weighted_scaled_g_collocation_field_into(
    out: np.ndarray,
    G: np.ndarray,
    geometry_surface_fields: np.ndarray,
    sqrt_weights: np.ndarray,
    offset: int,
) -> None:
    """Write collocation-scaled Grad-Shafranov residual samples into ``out``."""
    R_surface = geometry_surface_fields[GEOMETRY_SURFACE_R]
    J_surface = geometry_surface_fields[GEOMETRY_SURFACE_J]
    nr, nt = G.shape
    cursor = offset
    for i in range(nr):
        weight_i = sqrt_weights[i]
        for j in range(nt):
            # Collocation writes pointwise R/J * G with sqrt quadrature weights
            # so least_squares minimizes a discrete L2 norm over the surface grid.
            out[cursor] = weight_i * (R_surface[i, j] / J_surface[i, j]) * G[i, j]
            cursor += 1
