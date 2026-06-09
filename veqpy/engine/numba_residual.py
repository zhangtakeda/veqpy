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
"""

from __future__ import annotations

import numpy as np
from numba import njit

from veqpy.math.fast import (
    indexed_matvec_into,
    rowwise_sum_into,
    rowwise_weighted_sum_into,
)


@njit(cache=True, fastmath=True, nogil=True)
def update_residual_compact(
    out_workspace: np.ndarray,
    alpha1: float,
    alpha2: float,
    root_fields: np.ndarray,
    geometry_surface_fields: np.ndarray,
) -> None:
    """Update residual-related 2D fields in place from compact geometry fields."""
    out_G = out_workspace[0]
    out_Gpsin_R = out_workspace[1]
    out_Gpsin_Z = out_workspace[2]
    out_Gpsin_R_sin_tb = out_workspace[3]
    sin_tb_surface = geometry_surface_fields[0]
    R_surface = geometry_surface_fields[1]
    R_t_surface = geometry_surface_fields[2]
    Z_t_surface = geometry_surface_fields[3]
    J_surface = geometry_surface_fields[4]
    JdivR_surface = geometry_surface_fields[5]
    grtdivJR_t_surface = geometry_surface_fields[6]
    gttdivJR_surface = geometry_surface_fields[7]
    gttdivJR_r_surface = geometry_surface_fields[8]

    psin_r = root_fields[1]
    psin_rr = root_fields[2]
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]

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
def _run_residual_blocks_packed_precomputed(
    out_packed: np.ndarray,
    scratch: np.ndarray,
    block_codes: np.ndarray,
    block_orders: np.ndarray,
    block_radial_powers: np.ndarray,
    coeff_index_rows: np.ndarray,
    lengths: np.ndarray,
    residual_workspace: np.ndarray,
    sin_mtheta: np.ndarray,
    cos_mtheta: np.ndarray,
    rho_powers: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    G = residual_workspace[0]
    Gpsin_R = residual_workspace[1]
    Gpsin_Z = residual_workspace[2]
    Gpsin_R_sin_tb = residual_workspace[3]
    sin_theta = sin_mtheta[1]
    rho = rho_powers[1]
    rho2 = rho_powers[2]
    nt = G.shape[1]
    base_scale = 2.0 * np.pi / nt
    for slot in range(block_codes.shape[0]):
        coeff_indices = coeff_index_rows[slot, : lengths[slot]]
        code = block_codes[slot]
        order = block_orders[slot]
        radial_power = block_radial_powers[slot]
        if code == 0:
            rowwise_sum_into(scratch, Gpsin_R)
            _project_scaled2(out_packed, coeff_indices, T, scratch, y, weights, base_scale * a)
        elif code == 1:
            rowwise_sum_into(scratch, Gpsin_Z)
            _project_scaled2(out_packed, coeff_indices, T, scratch, y, weights, base_scale * a)
        elif code == 2:
            rowwise_weighted_sum_into(scratch, Gpsin_Z, sin_theta)
            _project_scaled3(
                out_packed, coeff_indices, T, scratch, rho, y, weights, base_scale * (-a)
            )
        elif code == 3:
            rowwise_sum_into(scratch, Gpsin_R_sin_tb)
            _project_scaled3(
                out_packed, coeff_indices, T, scratch, rho, y, weights, base_scale * (-a)
            )
        elif code == 4:
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
            rowwise_sum_into(scratch, G)
            _project_scaled3(out_packed, coeff_indices, T, scratch, rho2, y, weights, base_scale)
        elif code == 7:
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


run_residual_blocks_packed_precomputed = _run_residual_blocks_packed_precomputed


@njit(cache=True, fastmath=True, nogil=True)
def write_weighted_scaled_g_collocation_field_into(
    out: np.ndarray,
    G: np.ndarray,
    geometry_surface_fields: np.ndarray,
    sqrt_weights: np.ndarray,
    offset: int,
) -> None:
    """Write collocation-scaled Grad-Shafranov residual samples into ``out``."""
    R_surface = geometry_surface_fields[1]
    J_surface = geometry_surface_fields[4]
    nr, nt = G.shape
    cursor = offset
    for i in range(nr):
        weight_i = sqrt_weights[i]
        for j in range(nt):
            out[cursor] = weight_i * (R_surface[i, j] / J_surface[i, j]) * G[i, j]
            cursor += 1
