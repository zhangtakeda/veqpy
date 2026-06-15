"""
Module: engine.numba_geometry

Role:
- Materialize geometry fields.
- Update geometry integrals at the same time.

Public API:
- update_geometry

Notes:
- Inputs and outputs use packed-field semantics.
- Operator staging is not handled here.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from veqpy.workspace.field_rows import (
    GEOMETRY_RADIAL_KN,
    GEOMETRY_RADIAL_KN_R,
    GEOMETRY_RADIAL_LN_R,
    GEOMETRY_RADIAL_S_R,
    GEOMETRY_RADIAL_V_R,
    GEOMETRY_SURFACE_GRTDIVJR_T,
    GEOMETRY_SURFACE_GTTDIVJR,
    GEOMETRY_SURFACE_GTTDIVJR_R,
    GEOMETRY_SURFACE_J,
    GEOMETRY_SURFACE_JDIVR,
    GEOMETRY_SURFACE_R,
    GEOMETRY_SURFACE_R_T,
    GEOMETRY_SURFACE_SIN_TB,
    GEOMETRY_SURFACE_Z_T,
    PROFILE_R,
    PROFILE_RR,
    PROFILE_VALUE,
)


@njit(cache=True, fastmath=True, nogil=True)
def update_geometry_hot(
    surface_fields: np.ndarray,
    radial_fields: np.ndarray,
    a: float,
    R0: float,
    Z0: float,
    rho: np.ndarray,
    theta: np.ndarray,
    cos_mtheta: np.ndarray,
    sin_mtheta: np.ndarray,
    m_cos_mtheta: np.ndarray,
    m_sin_mtheta: np.ndarray,
    m2_cos_mtheta: np.ndarray,
    m2_sin_mtheta: np.ndarray,
    h_fields: np.ndarray,
    v_fields: np.ndarray,
    k_fields: np.ndarray,
    c_fields: np.ndarray,
    s_fields: np.ndarray,
    c_active_order: int,
    s_active_order: int,
) -> None:
    """Materialize only the geometry fields and integrals required by the fused solve hot path."""
    sin_tb = surface_fields[GEOMETRY_SURFACE_SIN_TB]
    R_surface = surface_fields[GEOMETRY_SURFACE_R]
    R_t_surface = surface_fields[GEOMETRY_SURFACE_R_T]
    Z_t_surface = surface_fields[GEOMETRY_SURFACE_Z_T]
    J_surface = surface_fields[GEOMETRY_SURFACE_J]
    JdivR_surface = surface_fields[GEOMETRY_SURFACE_JDIVR]
    grtdivJR_t_surface = surface_fields[GEOMETRY_SURFACE_GRTDIVJR_T]
    gttdivJR_surface = surface_fields[GEOMETRY_SURFACE_GTTDIVJR]
    gttdivJR_r_surface = surface_fields[GEOMETRY_SURFACE_GTTDIVJR_R]
    S_r = radial_fields[GEOMETRY_RADIAL_S_R]
    V_r = radial_fields[GEOMETRY_RADIAL_V_R]
    Kn = radial_fields[GEOMETRY_RADIAL_KN]
    Kn_r = radial_fields[GEOMETRY_RADIAL_KN_R]
    Ln_r = radial_fields[GEOMETRY_RADIAL_LN_R]
    nr = rho.shape[0]
    nt = theta.shape[0]
    theta_scale = 2.0 * np.pi / nt
    mean_scale = 1.0 / nt
    two_pi = 2.0 * np.pi
    c_limit = min(c_active_order + 1, c_fields.shape[0], cos_mtheta.shape[0])
    s_limit = min(s_active_order + 1, s_fields.shape[0], sin_mtheta.shape[0])
    for i in range(nr):
        rho_i = rho[i]
        h_i = h_fields[PROFILE_VALUE, i]
        h_r_i = h_fields[PROFILE_R, i]
        h_rr_i = h_fields[PROFILE_RR, i]
        v_r_i = v_fields[PROFILE_R, i]
        v_rr_i = v_fields[PROFILE_RR, i]
        k_i = k_fields[PROFILE_VALUE, i]
        k_r_i = k_fields[PROFILE_R, i]
        k_rr_i = k_fields[PROFILE_RR, i]
        c0_i = c_fields[0, PROFILE_VALUE, i]
        c0_r_i = c_fields[0, PROFILE_R, i]
        c0_rr_i = c_fields[0, PROFILE_RR, i]

        sum_J = 0.0
        sum_JR = 0.0
        sum_gttdivJR = 0.0
        sum_gttdivJR_r = 0.0
        sum_JdivR = 0.0

        for j in range(nt):
            sin_t = sin_mtheta[1, j]
            cos_t = cos_mtheta[1, j]

            tb_ij = theta[j] + c0_i
            tb_r_ij = c0_r_i
            tb_t_ij = 1.0
            tb_rr_ij = c0_rr_i
            tb_rt_ij = 0.0
            tb_tt_ij = 0.0

            # theta_bar is the Fourier-distorted poloidal angle used by the R
            # surface only; Z keeps the explicit elongation/sine form below.
            for order in range(1, c_limit):
                cos_kt = cos_mtheta[order, j]
                k_sin_kt = m_sin_mtheta[order, j]
                k2_cos_kt = m2_cos_mtheta[order, j]
                c_i = c_fields[order, PROFILE_VALUE, i]
                c_r_i = c_fields[order, PROFILE_R, i]
                c_rr_i = c_fields[order, PROFILE_RR, i]

                tb_ij += c_i * cos_kt
                tb_r_ij += c_r_i * cos_kt
                tb_t_ij -= c_i * k_sin_kt
                tb_rr_ij += c_rr_i * cos_kt
                tb_rt_ij -= c_r_i * k_sin_kt
                tb_tt_ij -= c_i * k2_cos_kt

            for order in range(1, s_limit):
                sin_kt = sin_mtheta[order, j]
                k_cos_kt = m_cos_mtheta[order, j]
                k2_sin_kt = m2_sin_mtheta[order, j]
                s_i = s_fields[order, PROFILE_VALUE, i]
                s_r_i = s_fields[order, PROFILE_R, i]
                s_rr_i = s_fields[order, PROFILE_RR, i]

                tb_ij += s_i * sin_kt
                tb_r_ij += s_r_i * sin_kt
                tb_t_ij += s_i * k_cos_kt
                tb_rr_ij += s_rr_i * sin_kt
                tb_rt_ij += s_r_i * k_cos_kt
                tb_tt_ij -= s_i * k2_sin_kt

            cos_tb_ij = np.cos(tb_ij)
            sin_tb_ij = np.sin(tb_ij)

            R_ij = R0 + a * (h_i + rho_i * cos_tb_ij)
            if R_ij < 1e-6:
                # Downstream source/residual formulas divide by R and J; clamp
                # only the singular denominator, not the user-facing profile.
                R_ij = 1e-6

            R_r_ij = a * (h_r_i + cos_tb_ij - rho_i * sin_tb_ij * tb_r_ij)
            R_t_ij = -a * rho_i * sin_tb_ij * tb_t_ij
            R_rr_ij = a * (
                h_rr_i
                - 2.0 * sin_tb_ij * tb_r_ij
                - rho_i * (cos_tb_ij * tb_r_ij * tb_r_ij + sin_tb_ij * tb_rr_ij)
            )
            R_rt_ij = -a * (
                sin_tb_ij * tb_t_ij + rho_i * (cos_tb_ij * tb_r_ij * tb_t_ij + sin_tb_ij * tb_rt_ij)
            )
            R_tt_ij = -a * rho_i * (cos_tb_ij * tb_t_ij * tb_t_ij + sin_tb_ij * tb_tt_ij)

            Z_r_ij = a * (v_r_i - (k_i + rho_i * k_r_i) * sin_t)
            Z_t_ij = -a * rho_i * k_i * cos_t
            Z_rr_ij = a * (v_rr_i - (2.0 * k_r_i + rho_i * k_rr_i) * sin_t)
            Z_rt_ij = -a * (k_i + rho_i * k_r_i) * cos_t
            Z_tt_ij = a * rho_i * k_i * sin_t

            J_ij = R_t_ij * Z_r_ij - R_r_ij * Z_t_ij
            if J_ij < 1e-6:
                # A non-positive Jacobian means the surface map folded.  The
                # hot path keeps running with a finite penalty instead of
                # injecting NaNs into the nonlinear optimizer.
                J_ij = 1e-6

            # Store only metric combinations consumed by source/residual stages;
            # full R/Z derivative tensors would add memory traffic without a
            # current caller.
            J_r_ij = -(R_rr_ij * Z_t_ij - R_rt_ij * Z_r_ij + R_r_ij * Z_rt_ij - R_t_ij * Z_rr_ij)
            J_t_ij = -(R_rt_ij * Z_t_ij - R_tt_ij * Z_r_ij + R_r_ij * Z_tt_ij - R_t_ij * Z_rt_ij)
            JR_ij = J_ij * R_ij
            JR_r_ij = J_r_ij * R_ij + J_ij * R_r_ij
            JR_t_ij = J_t_ij * R_ij + J_ij * R_t_ij
            JdivR_ij = J_ij / R_ij

            grt_ij = R_r_ij * R_t_ij + Z_r_ij * Z_t_ij
            grt_t_ij = R_rt_ij * R_t_ij + R_r_ij * R_tt_ij + Z_rt_ij * Z_t_ij + Z_r_ij * Z_tt_ij
            gtt_ij = R_t_ij * R_t_ij + Z_t_ij * Z_t_ij
            gtt_r_ij = 2.0 * (R_t_ij * R_rt_ij + Z_t_ij * Z_rt_ij)
            inv_JR = 1.0 / JR_ij
            grtdivJR_t_ij = (grt_t_ij - grt_ij * JR_t_ij * inv_JR) * inv_JR
            gttdivJR_ij = gtt_ij * inv_JR
            gttdivJR_r_ij = gtt_r_ij * inv_JR - gtt_ij * JR_r_ij * inv_JR * inv_JR

            sin_tb[i, j] = sin_tb_ij
            R_surface[i, j] = R_ij
            R_t_surface[i, j] = R_t_ij
            Z_t_surface[i, j] = Z_t_ij
            J_surface[i, j] = J_ij
            JdivR_surface[i, j] = JdivR_ij
            grtdivJR_t_surface[i, j] = grtdivJR_t_ij
            gttdivJR_surface[i, j] = gttdivJR_ij
            gttdivJR_r_surface[i, j] = gttdivJR_r_ij

            sum_J += J_ij
            sum_JR += JR_ij
            sum_gttdivJR += gttdivJR_ij
            sum_gttdivJR_r += gttdivJR_r_ij
            sum_JdivR += JdivR_ij

        # Radial fields are theta integrals or means paired with the compact
        # surface-field row layout documented by GeometryWorkspace.
        S_r[i] = sum_J * theta_scale
        V_r[i] = sum_JR * theta_scale * two_pi
        Kn[i] = sum_gttdivJR * mean_scale
        Kn_r[i] = sum_gttdivJR_r * mean_scale
        Ln_r[i] = sum_JdivR * mean_scale
