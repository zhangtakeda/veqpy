"""
Module: veqpy.model.equilibrium

Role:
- Hold an equilibrium snapshot on one grid.
- Re-derive geometry and diagnostics from root fields.
- Provide plotting, resampling, and other inspection capabilities.

Public API:
- Equilibrium

Notes:
- `Equilibrium` is a snapshot, not a solver runtime container.
- Physical fields are materialized lazily by self-contained model-side Numba
  kernels; packed solve state and persistent workspaces remain outside the model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
from numba import njit

from veqpy.base import Reactive, Serial
from veqpy.model.geqdsk import Geqdsk
from veqpy.model.grid import Grid
from veqpy.model.profile import Profile
from veqpy.numerics import interpolation_matrix

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from rich.tree import Tree

MU0 = 4e-7 * np.pi


def _regularize_axis_linear_profile(
    values: np.ndarray,
    rho: np.ndarray,
    *,
    copy: bool = False,
) -> np.ndarray:
    values = np.array(values, dtype=np.float64, copy=copy)
    rho = np.asarray(rho, dtype=np.float64)
    if values.ndim != 1 or rho.ndim != 1 or values.shape != rho.shape:
        raise ValueError(
            f"Expected values/rho to share a 1D shape, got {values.shape} and {rho.shape}"
        )
    if values.size < 3 or abs(rho[0]) >= 1e-10:
        return values

    # Diagnostics such as q and j are singular-looking at rho=0 even when the
    # physical limit is finite.  Reconstruct only the axis sample from the first
    # two off-axis values so plotted/exported profiles stay regular.
    rho1 = float(rho[1])
    rho2 = float(rho[2])
    if abs(rho2 - rho1) < 1e-14:
        return values
    if not np.isfinite(values[1]) or not np.isfinite(values[2]):
        return values

    slope = (values[2] - values[1]) / (rho2 - rho1)
    values[0] = values[1] + slope * (rho[0] - rho1)
    return values


class Equilibrium(Reactive, Serial):
    """Equilibrium snapshot object on one grid."""

    root_properties = {
        "R0",
        "Z0",
        "B0",
        "a",
        "grid",
        "shape_profiles",
        "FFn_psin",
        "Pn_psin",
        "psin",
        "psin_r",
        "psin_rr",
        "p0",
        "alpha1",
        "alpha2",
    }

    def __init__(
        self,
        R0: float,
        Z0: float,
        B0: float,
        a: float,
        grid: Grid,
        *,
        shape_profiles: dict[str, Profile],
        FFn_psin: np.ndarray,
        Pn_psin: np.ndarray,
        psin: np.ndarray,
        psin_r: np.ndarray,
        psin_rr: np.ndarray,
        p0: float = 0.0,
        alpha1: float = 1.0,
        alpha2: float = 1.0,
    ) -> None:
        """Initialize the equilibrium snapshot object."""
        super().__init__()

        self.R0 = R0
        self.Z0 = Z0
        self.B0 = B0
        self.a = a
        self.grid = grid
        self.shape_profiles = _normalize_shape_profiles(shape_profiles)

        self.psin = np.asarray(psin, dtype=np.float64)
        # Source profiles are stored as derivatives with respect to psin.  Their
        # axis values are used by diagnostics/export, so repair only the snapshot
        # copy and leave the solver root arrays untouched.
        self.FFn_psin = _regularize_axis_linear_profile(FFn_psin, grid.rho, copy=True)
        self.Pn_psin = _regularize_axis_linear_profile(Pn_psin, grid.rho, copy=True)
        self.psin_r = np.asarray(psin_r, dtype=np.float64)
        self.psin_rr = np.asarray(psin_rr, dtype=np.float64)
        self.p0 = float(p0)
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def __rich__(self) -> Tree:
        from .equilibrium_plot import _build_equilibrium_rich_tree

        return _build_equilibrium_rich_tree(self)

    def __str__(self) -> str:
        from .equilibrium_plot import _equilibrium_to_string

        return _equilibrium_to_string(self)

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def serial_attributes(cls) -> dict[str, type]:
        """Declare serializable construction root state."""
        attrs: dict[str, type] = {
            "R0": float,
            "Z0": float,
            "B0": float,
            "a": float,
            "grid": Grid,
            "shape_profiles": dict[str, Profile],
            "psin": np.ndarray,
            "FFn_psin": np.ndarray,
            "Pn_psin": np.ndarray,
            "psin_r": np.ndarray,
            "psin_rr": np.ndarray,
            "p0": float,
            "alpha1": float,
            "alpha2": float,
        }
        return attrs

    @property
    def rho(self) -> np.ndarray:
        """Radial grid nodes inherited from ``grid``."""
        return self.grid.rho

    @property
    def theta(self) -> np.ndarray:
        """Poloidal grid angles inherited from ``grid``."""
        return self.grid.theta

    @property
    def cos_theta(self) -> np.ndarray:
        """First-harmonic cosine table, ``cos(theta)``."""
        return self.grid.cos_mtheta[1]

    @property
    def sin_theta(self) -> np.ndarray:
        """First-harmonic sine table, ``sin(theta)``."""
        return self.grid.sin_mtheta[1]

    @property
    def _h_fields(self) -> np.ndarray:
        return _materialize_profile_fields(self.shape_profiles.get("h"), self.grid)

    @property
    def _v_fields(self) -> np.ndarray:
        return _materialize_profile_fields(self.shape_profiles.get("v"), self.grid)

    @property
    def _kappa_fields(self) -> np.ndarray:
        return _materialize_profile_fields(self.shape_profiles.get("k"), self.grid)

    @property
    def _fourier_profile_fields(self) -> tuple[np.ndarray, np.ndarray]:
        return _materialize_fourier_profile_fields(self.shape_profiles, self.grid)

    @property
    def h(self) -> np.ndarray:
        """Normalized horizontal/Shafranov-shift profile on ``rho``."""
        return self._h_fields[0]

    @property
    def v(self) -> np.ndarray:
        """Normalized vertical-shift profile on ``rho``."""
        return self._v_fields[0]

    @property
    def kappa(self) -> np.ndarray:
        """Elongation profile; public name for shape profile ``k``."""
        return self._kappa_fields[0]

    @property
    def Rc(self) -> np.ndarray:
        """Flux-surface major-radius center in metres."""
        out = np.empty_like(self.rho)
        update_rc(out, self.R0, self.a, self.h)
        return _readonly_owned(out)

    @property
    def epsilon(self) -> np.ndarray:
        """Local inverse aspect ratio ``a*rho/Rc``."""
        out = np.empty_like(self.rho)
        update_epsilon(out, self.a, self.rho, self.Rc)
        return _readonly_owned(out)

    @property
    def ftrap(self) -> np.ndarray:
        """Trapped-particle fraction from the full flux-surface magnetic field."""
        out = np.empty_like(self.rho)
        invalid = update_ftrap(
            out,
            self.R,
            self.J,
            self.gttdivJR,
            self.F,
            _validated_radial_root(self.psin_r, self.grid, "psin_r"),
            self.alpha2,
            self.rho,
        )
        if invalid:
            raise ValueError("trapped-particle fraction contains non-finite values")
        return _readonly_owned(out)

    @property
    def _R_geometry_fields(self) -> np.ndarray:
        grid = self.grid
        _validate_equilibrium_grid_tables(grid)
        out = np.empty((18, grid.Nr, grid.Nt), dtype=np.float64)
        c_fields, s_fields = self._fourier_profile_fields
        update_r_coordinates(
            out,
            self.a,
            self.R0,
            grid.rho,
            grid.theta,
            grid.cos_mtheta,
            grid.sin_mtheta,
            grid.m_cos_mtheta,
            grid.m_sin_mtheta,
            grid.m2_cos_mtheta,
            grid.m2_sin_mtheta,
            self._h_fields,
            c_fields,
            s_fields,
        )
        return _readonly_owned(out)

    @property
    def _Z_geometry_fields(self) -> np.ndarray:
        grid = self.grid
        _validate_equilibrium_grid_tables(grid)
        out = np.empty((18, grid.Nr, grid.Nt), dtype=np.float64)
        update_z_coordinates(
            out,
            self.a,
            self.Z0,
            grid.rho,
            grid.sin_mtheta[1],
            grid.cos_mtheta[1],
            self._v_fields,
            self._kappa_fields,
        )
        return _readonly_owned(out)

    @property
    def _metric_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        surface, radial, invalid = materialize_metric_geometry(
            self._R_geometry_fields,
            self._Z_geometry_fields,
            self.rho,
        )
        if invalid:
            raise ValueError(
                "Equilibrium metric geometry is singular away from the magnetic axis "
                f"at radial index {invalid - 1}"
            )
        return _readonly_owned(surface), _readonly_owned(radial)

    @property
    def _materialized_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility view of the lazily materialized metric group."""
        return self._metric_geometry

    @property
    def surface_fields(self) -> np.ndarray:
        """Packed two-dimensional geometry fields on ``(rho, theta)``."""
        return self._metric_geometry[0]

    @property
    def radial_fields(self) -> np.ndarray:
        """Packed one-dimensional radial geometry integrals."""
        return self._metric_geometry[1]

    @property
    def R(self) -> np.ndarray:
        """Major-radius surface coordinates on ``(rho, theta)``."""
        return self._R_geometry_fields[R]

    @property
    def Z(self) -> np.ndarray:
        """Vertical surface coordinates on ``(rho, theta)``."""
        return self._Z_geometry_fields[Z]

    @property
    def Z_t(self) -> np.ndarray:
        """Poloidal derivative of ``Z``."""
        return self._Z_geometry_fields[Z_T]

    @property
    def J(self) -> np.ndarray:
        """Surface Jacobian field."""
        return self.surface_fields[J]

    @property
    def JdivR(self) -> np.ndarray:
        """Jacobian divided by major radius."""
        return self.surface_fields[JDIVR]

    @property
    def gttdivJR(self) -> np.ndarray:
        """Metric coefficient ``g_tt / (J*R)``."""
        return self.surface_fields[GTTDIVJR]

    @property
    def gttdivJR_r(self) -> np.ndarray:
        """Radial derivative of ``g_tt / (J*R)``."""
        return self.surface_fields[GTTDIVJR_R]

    @property
    def grtdivJR_t(self) -> np.ndarray:
        """Poloidal derivative of the mixed metric coefficient ``g_rt/(J*R)``."""
        return self.surface_fields[GRTDIVJR_T]

    @property
    def S(self) -> np.ndarray:
        """Flux-surface area S = -int R*Z_t dtheta."""
        out = np.empty_like(self.rho)
        update_surface_area(out, self.R, self.Z_t)
        return _readonly_owned(out)

    @property
    def S_r(self) -> np.ndarray:
        """Flux-surface area derivative S_r = int J dtheta."""
        return self.radial_fields[S_R]

    @property
    def V(self) -> np.ndarray:
        """Flux-surface volume V = -pi*int R**2*Z_t dtheta."""
        out = np.empty_like(self.rho)
        update_volume(out, self.R, self.Z_t)
        return _readonly_owned(out)

    @property
    def V_r(self) -> np.ndarray:
        """Flux-surface volume derivative V_r = 2pi * int J*R dtheta."""
        return self.radial_fields[V_R]

    @property
    def Kn(self) -> np.ndarray:
        """Normalized geometry factor Kn = int gttdivJR dtheta/(2pi)."""
        return self.radial_fields[KN]

    @property
    def Kn_r(self) -> np.ndarray:
        """Radial derivative of Kn."""
        return self.radial_fields[KN_R]

    @property
    def Ln_r(self) -> np.ndarray:
        """Normalized geometry factor Ln_r = int JdivR dtheta/(2pi)."""
        return self.radial_fields[LN_R]

    @property
    def FF_r(self) -> np.ndarray:
        """Physical F*F' profile, model-side diagnostic."""
        out = np.empty_like(self.rho)
        update_scaled_copy(out, self.FFn_r, self.alpha1 * self.alpha2)
        return _readonly_owned(out)

    @property
    def FFn_r(self) -> np.ndarray:
        """Radial derivative of the normalized ``F*F'`` source profile."""
        out = np.empty_like(self.rho)
        update_scaled_product(
            out,
            _validated_radial_root(self.FFn_psin, self.grid, "FFn_psin"),
            _validated_radial_root(self.psin_r, self.grid, "psin_r"),
            1.0,
        )
        return _readonly_owned(out)

    @property
    def F2(self) -> np.ndarray:
        """Physical F^2 profile."""
        out = np.empty_like(self.rho)
        update_f2(
            out,
            self.FF_r,
            self.grid.accumulator,
            self.grid.weights,
            (self.R0 * self.B0) ** 2,
        )
        return _readonly_owned(out)

    @property
    def F(self) -> np.ndarray:
        """Signed poloidal current function ``F = R * B_phi``."""
        out = np.empty_like(self.rho)
        invalid = update_f(out, self.F2, self.R0 * self.B0)
        if invalid:
            raise ValueError("Negative F2 encountered, cannot compute F")
        return _readonly_owned(out)

    @property
    def P_r(self) -> np.ndarray:
        """Physical pressure gradient P', model-side diagnostic."""
        out = np.empty_like(self.rho)
        update_scaled_copy(out, self.Pn_r, self.alpha1 * self.alpha2 / MU0)
        return _readonly_owned(out)

    @property
    def Pn_r(self) -> np.ndarray:
        """Radial derivative of the normalized pressure source profile."""
        out = np.empty_like(self.rho)
        update_scaled_product(
            out,
            _validated_radial_root(self.Pn_psin, self.grid, "Pn_psin"),
            _validated_radial_root(self.psin_r, self.grid, "psin_r"),
            1.0,
        )
        return _readonly_owned(out)

    @property
    def P(self) -> np.ndarray:
        """Physical pressure profile P."""
        out = np.empty_like(self.rho)
        update_pressure(
            out,
            self.P_r,
            self.grid.accumulator,
            self.grid.weights,
            self.p0,
        )
        return _readonly_owned(out)

    @property
    def beta_t(self) -> float:
        """Toroidal beta beta_t = 2*mu0*<P> / B0^2."""
        return float(update_beta_t(self.P, self.V_r, self.grid.weights, self.B0))

    @property
    def Gn1(self) -> np.ndarray:
        """Normalized source term before alpha1 in the GS operator."""
        out = np.empty((self.grid.Nr, self.grid.Nt), dtype=np.float64)
        update_gn1(
            out,
            self.R,
            self.JdivR,
            _validated_radial_root(self.FFn_psin, self.grid, "FFn_psin"),
            _validated_radial_root(self.Pn_psin, self.grid, "Pn_psin"),
        )
        return _readonly_owned(out)

    @property
    def Gn2(self) -> np.ndarray:
        """Normalized geometry term before alpha2 in the GS operator."""
        out = np.empty((self.grid.Nr, self.grid.Nt), dtype=np.float64)
        update_gn2(
            out,
            self.gttdivJR,
            self.gttdivJR_r,
            self.grtdivJR_t,
            _validated_radial_root(self.psin_r, self.grid, "psin_r"),
            _validated_radial_root(self.psin_rr, self.grid, "psin_rr"),
        )
        return _readonly_owned(out)

    @property
    def G(self) -> np.ndarray:
        """GS operator residual field G = alpha1 * Gn1 + alpha2 * Gn2."""
        out = np.empty((self.grid.Nr, self.grid.Nt), dtype=np.float64)
        update_linear_combination_2d(
            out,
            self.Gn1,
            self.Gn2,
            self.alpha1,
            self.alpha2,
        )
        return _readonly_owned(out)

    @property
    def Ip(self) -> float:
        """Total plasma current Ip (Amps)."""
        return float(update_ip(self.Gn1, self.grid.weights, self.alpha1))

    @property
    def q(self) -> np.ndarray:
        """Safety factor q, model-side diagnostic."""
        out = np.empty_like(self.rho)
        invalid = update_q(
            out,
            self.F,
            self.Ln_r,
            self.alpha2,
            _validated_radial_root(self.psin_r, self.grid, "psin_r"),
            self.rho,
        )
        if invalid:
            raise ValueError(
                "q contains a non-finite value away from a removable magnetic-axis "
                f"singularity at radial index {invalid - 1}"
            )
        return _readonly_owned(out)

    @property
    def s(self) -> np.ndarray:
        """Magnetic shear s, model-side diagnostic."""
        out = np.empty_like(self.rho)
        update_shear(
            out,
            self.q,
            self.rho,
            self.grid.differentiator,
        )
        return _readonly_owned(out)

    @property
    def Itor(self) -> np.ndarray:
        """Toroidal current distribution I_tor(rho), model-side diagnostic."""
        out = np.empty_like(self.rho)
        update_itor(
            out,
            self.Kn,
            self.alpha2,
            _validated_radial_root(self.psin_r, self.grid, "psin_r"),
        )
        return _readonly_owned(out)

    @property
    def jtor(self) -> np.ndarray:
        """Toroidal current density j_phi, model-side diagnostic."""
        out = np.empty_like(self.rho)
        invalid = update_jtor(
            out,
            _validated_radial_root(self.FFn_psin, self.grid, "FFn_psin"),
            _validated_radial_root(self.Pn_psin, self.grid, "Pn_psin"),
            self.Ln_r,
            self.S_r,
            self.V_r,
            self.Itor,
            self.S,
            self.alpha1,
            self.rho,
        )
        if invalid:
            raise ValueError(
                "jtor contains a non-finite value away from a removable magnetic-axis "
                f"singularity at radial index {invalid - 1}"
            )
        return _readonly_owned(out)

    @property
    def jpara(self) -> np.ndarray:
        """PJ2 current profile ``<J·B> / (F * <R^-2>)``.

        This is not the IMAS parallel-current convention ``<J·B> / B0``.
        With ``gm1 = <R^-2> = (2*pi)^2 * Ln_r / V_r``, the corresponding
        IMAS total-current profile is ``jpara * F * gm1 / B0``.
        """
        out = np.empty_like(self.rho)
        invalid = update_jpara(
            out,
            self.F,
            self.Kn,
            self.Kn_r,
            self.Ln_r,
            _validated_radial_root(self.psin_r, self.grid, "psin_r"),
            _validated_radial_root(self.psin_rr, self.grid, "psin_rr"),
            self.alpha2,
            self.rho,
            self.grid.differentiator,
        )
        if invalid:
            raise ValueError(
                "jpara contains a non-finite value away from a removable magnetic-axis "
                f"singularity at radial index {invalid - 1}"
            )
        return _readonly_owned(out)

    @property
    def jtotal(self) -> np.ndarray:
        """IMAS total parallel-current convention ``<J·B> / B0``.

        ``jpara`` retains VEQ/PJ2's ``<J·B> / (F <R^-2>)`` convention;
        this property applies ``gm1 = <R^-2> = (2*pi)^2 Ln_r / V_r`` so
        callers can compare directly with ``equilibrium.time_slice[].profiles_1d.j_total``.
        """

        out = np.empty_like(self.rho)
        invalid = update_jtotal(
            out,
            self.jpara,
            self.F,
            self.Ln_r,
            self.V_r,
            self.B0,
            self.rho,
        )
        if invalid:
            raise ValueError(
                "jtotal contains a non-finite value away from a removable magnetic-axis "
                f"singularity at radial index {invalid - 1}"
            )
        return _readonly_owned(out)

    @property
    def jphi(self) -> np.ndarray:
        """Local toroidal current density j_phi(R, Z)."""
        out = np.empty((self.grid.Nr, self.grid.Nt), dtype=np.float64)
        update_jphi(
            out,
            self.R,
            _validated_radial_root(self.FFn_psin, self.grid, "FFn_psin"),
            _validated_radial_root(self.Pn_psin, self.grid, "Pn_psin"),
            self.alpha1,
        )
        return _readonly_owned(out)

    @property
    def Psi(self) -> np.ndarray:
        """Physical poloidal flux Psi."""
        out = np.empty_like(self.rho)
        update_scaled_copy(
            out,
            _validated_radial_root(self.psin, self.grid, "psin"),
            2.0 * np.pi * self.alpha2,
        )
        return _readonly_owned(out)

    @property
    def Phi_r(self) -> np.ndarray:
        """Derivative of toroidal flux ``Phi`` with respect to VEQ ``rho``."""
        out = np.empty_like(self.rho)
        update_scaled_product(out, self.F, self.Ln_r, 2.0 * np.pi)
        return _readonly_owned(out)

    @property
    def Phi(self) -> np.ndarray:
        """Toroidal flux Phi."""
        out = np.empty_like(self.rho)
        update_phi(
            out,
            self.F,
            self.Ln_r,
            self.grid.accumulator,
        )
        return _readonly_owned(out)

    @property
    def _toroidal_flux_coordinates(self) -> np.ndarray:
        out = np.empty((4, self.grid.Nr), dtype=np.float64)
        phi_edge = self.grid.full_integral(self.Phi_r)
        rho_tor_edge_squared = phi_edge / (np.pi * self.B0)
        if rho_tor_edge_squared <= 0.0 or not np.isfinite(rho_tor_edge_squared):
            raise ValueError("IMAS toroidal-flux coordinate has a non-physical LCFS value")
        invalid = update_toroidal_flux_coordinates(
            out,
            self.Phi,
            self.F,
            self.Ln_r,
            self.B0,
            np.sqrt(rho_tor_edge_squared),
            self.rho,
        )
        if invalid:
            raise ValueError(
                "IMAS toroidal-flux coordinates contain a non-finite or "
                f"non-physical value at radial index {invalid - 1}"
            )
        return _readonly_owned(out)

    @property
    def rho_tor(self) -> np.ndarray:
        """IMAS toroidal-flux coordinate ``sqrt(Phi / (pi * B0))`` [m]."""
        return self._toroidal_flux_coordinates[RHO_TOR]

    @property
    def rho_tor_norm(self) -> np.ndarray:
        """IMAS toroidal-flux coordinate normalized from axis to boundary."""
        return self._toroidal_flux_coordinates[RHO_TOR_NORM]

    @property
    def rho_tor_r(self) -> np.ndarray:
        """Derivative of physical ``rho_tor`` with respect to VEQ ``rho`` [m]."""
        return self._toroidal_flux_coordinates[RHO_TOR_R]

    @property
    def rho_tor_norm_r(self) -> np.ndarray:
        """Derivative of ``rho_tor_norm`` with respect to VEQ ``rho``."""
        return self._toroidal_flux_coordinates[RHO_TOR_NORM_R]

    @property
    def _gm_fields(self) -> np.ndarray:
        out = np.empty((9, self.grid.Nr), dtype=np.float64)
        invalid = update_gm(
            out,
            self.R,
            self.J,
            self.gttdivJR,
            self.F,
            _validated_radial_root(self.psin_r, self.grid, "psin_r"),
            self.Ln_r,
            self.S_r,
            self.V_r,
            self.rho_tor_r,
            self.alpha2,
            self.rho,
        )
        if invalid:
            raise ValueError(
                "IMAS gm geometry contains a non-finite or non-physical value "
                f"at radial index {invalid - 1}"
            )
        return _readonly_owned(out)

    @property
    def gm1(self) -> np.ndarray:
        """IMAS ``gm1 = <1/R^2>`` [m^-2]."""
        return self._gm_fields[GM1]

    @property
    def gm2(self) -> np.ndarray:
        """IMAS ``gm2 = <|grad rho_tor|^2/R^2>`` [m^-2]."""
        return self._gm_fields[GM2]

    @property
    def gm3(self) -> np.ndarray:
        """IMAS ``gm3 = <|grad rho_tor|^2>``."""
        return self._gm_fields[GM3]

    @property
    def gm4(self) -> np.ndarray:
        """IMAS ``gm4 = <1/B^2>`` [T^-2]."""
        return self._gm_fields[GM4]

    @property
    def gm5(self) -> np.ndarray:
        """IMAS ``gm5 = <B^2>`` [T^2]."""
        return self._gm_fields[GM5]

    @property
    def gm6(self) -> np.ndarray:
        """IMAS ``gm6 = <|grad rho_tor|^2/B^2>`` [T^-2]."""
        return self._gm_fields[GM6]

    @property
    def gm7(self) -> np.ndarray:
        """IMAS ``gm7 = <|grad rho_tor|>``."""
        return self._gm_fields[GM7]

    @property
    def gm8(self) -> np.ndarray:
        """IMAS ``gm8 = <R>`` [m]."""
        return self._gm_fields[GM8]

    @property
    def gm9(self) -> np.ndarray:
        """IMAS ``gm9 = <1/R>`` [m^-1]."""
        return self._gm_fields[GM9]

    def plot(
        self,
        outpath: str | None = None,
        *,
        show: bool = False,
        plot_residual: bool = False,
        grid: Grid | None = None,
        plot_all: bool = True,
    ) -> Figure:
        """Render a compact equilibrium summary figure."""
        from .equilibrium_plot import _plot_equilibrium

        return _plot_equilibrium(
            self,
            outpath=outpath,
            show=show,
            plot_residual=plot_residual,
            grid=grid,
            plot_all=plot_all,
        )

    def resample(
        self,
        grid: Grid,
    ) -> Self:
        """Interpolate the current equilibrium snapshot to a target grid."""
        return _build_resampled_equilibrium(
            self,
            grid=grid,
        )

    def to_geqdsk(
        self,
        *,
        R_range: tuple[float, float],
        Z_range: tuple[float, float],
        NR: int,
        NZ: int | None = None,
        header: str = "",
        limiter: np.ndarray | None = None,
        psi_axis: float = 0.0,
        psi_outside: float | None = None,
    ) -> Geqdsk:
        """Export a GEQDSK snapshot written in physical psi."""
        R_nodes, Z_nodes, Rmin, Rmax, Zmin, Zmax = _build_geqdsk_rectilinear_grid(
            R_range=R_range,
            Z_range=Z_range,
            NR=NR,
            NZ=NZ,
        )
        psin_uniform = np.linspace(0.0, 1.0, int(NR), dtype=np.float64)
        psi_axis = float(psi_axis)
        psi_scale = float(self.alpha2)
        if abs(psi_scale) <= 1.0e-14:
            raise ValueError("alpha2 is zero")
        # GEQDSK stores physical psi on a rectangular R/Z grid.  Internally this
        # snapshot carries normalized psin, so alpha2 supplies the physical span.
        psi_bound = psi_axis + psi_scale
        psi_outside_value = psi_bound if psi_outside is None else float(psi_outside)
        # GEQDSK owns the magnetic axis and LCFS explicitly.  A native Legendre
        # solve grid owns neither, so first lower the snapshot to an
        # endpoint-inclusive view instead of aliasing its first/last interior
        # surfaces to those physical boundaries.
        export_equilibrium = _endpoint_inclusive_equilibrium(self)
        R = export_equilibrium.R
        Z = export_equilibrium.Z
        boundary = np.column_stack((R[-1], Z[-1])).astype(np.float64, copy=False)
        limiter_points = _coerce_optional_point_array(limiter, name="limiter")

        geqdsk = Geqdsk(
            header=str(header),
            NR=int(NR),
            NZ=int(Z_nodes.size),
            R0=float(self.R0),
            Z0=float(self.Z0),
            Rmin=Rmin,
            Rmax=Rmax,
            Zmin=Zmin,
            Zmax=Zmax,
            boundary=boundary.copy(),
            limiter=limiter_points,
            Bt0=float(self.B0),
            Raxis=float(R[0, 0]),
            Zaxis=float(Z[0, 0]),
            Ip=float(self.Ip),
            psi_axis=psi_axis,
            psi_bound=psi_bound,
            # Profile arrays in GEQDSK are sampled on uniform normalized psin,
            # not on the solver's rho grid.  Sorting/deduplicating psin below
            # protects exports from slightly nonuniform optimized psin profiles.
            F=_sample_profile_on_uniform_psin(
                export_equilibrium.psin,
                export_equilibrium.F,
                psin_uniform,
            ),
            P=_sample_profile_on_uniform_psin(
                export_equilibrium.psin,
                export_equilibrium.P,
                psin_uniform,
            ),
            FF_psi=_sample_profile_on_uniform_psin(
                export_equilibrium.psin,
                export_equilibrium.alpha1 * export_equilibrium.FFn_psin,
                psin_uniform,
            ),
            P_psi=_sample_profile_on_uniform_psin(
                export_equilibrium.psin,
                export_equilibrium.alpha1 * export_equilibrium.Pn_psin / MU0,
                psin_uniform,
            ),
            q=_sample_profile_on_uniform_psin(
                export_equilibrium.psin,
                export_equilibrium.q,
                psin_uniform,
            ),
            psi=_interpolate_psin_to_rectilinear_grid(
                R,
                Z,
                export_equilibrium.psin,
                np.square(np.asarray(export_equilibrium.rho, dtype=np.float64)),
                R_nodes=R_nodes,
                Z_nodes=Z_nodes,
                psi_axis=psi_axis,
                psi_scale=psi_scale,
                psi_outside=psi_outside_value,
            ),
        )
        geqdsk.dR = float(R_nodes[1] - R_nodes[0]) if R_nodes.size > 1 else 0.0
        geqdsk.dZ = float(Z_nodes[1] - Z_nodes[0]) if Z_nodes.size > 1 else 0.0
        return geqdsk


def _normalize_shape_profiles(
    shape_profiles: dict[str, Profile],
) -> dict[str, Profile]:
    if not isinstance(shape_profiles, dict):
        raise TypeError(
            f"shape_profiles must be dict[str, Profile], got {type(shape_profiles).__name__}"
        )
    normalized: dict[str, Profile] = {}
    for name, profile in shape_profiles.items():
        if not isinstance(name, str):
            raise TypeError(f"shape profile names must be str, got {type(name).__name__}")
        if not isinstance(profile, Profile):
            raise TypeError(f"shape profile {name!r} must be Profile, got {type(profile).__name__}")
        normalized[name] = profile.copy()
    return normalized


def _validate_equilibrium_grid_tables(grid: Grid) -> None:
    """Validate the root-grid tables consumed by model-side Numba kernels."""

    expected_radial = (grid.Nr,)
    expected_radial_matrix = (grid.Nr, grid.Nr)
    expected_basis = (grid.L_max + 1, grid.Nr)
    expected_trig = (grid.M_max + 1, grid.Nt)
    for name in ("rho", "weights"):
        value = getattr(grid, name)
        if value.shape != expected_radial:
            raise ValueError(f"grid.{name} must have shape {expected_radial}, got {value.shape}")
    for name in ("accumulator", "differentiator"):
        value = getattr(grid, name)
        if value.shape != expected_radial_matrix:
            raise ValueError(
                f"grid.{name} must have shape {expected_radial_matrix}, got {value.shape}"
            )
    for name in ("T", "T_r", "T_rr"):
        value = getattr(grid, name)
        if value.shape != expected_basis:
            raise ValueError(f"grid.{name} must have shape {expected_basis}, got {value.shape}")
    for name in (
        "cos_mtheta",
        "sin_mtheta",
        "m_cos_mtheta",
        "m_sin_mtheta",
        "m2_cos_mtheta",
        "m2_sin_mtheta",
    ):
        value = getattr(grid, name)
        if value.shape != expected_trig:
            raise ValueError(f"grid.{name} must have shape {expected_trig}, got {value.shape}")


def _validated_profile_coefficients(
    profile: Profile,
    grid: Grid,
) -> tuple[np.ndarray, int]:
    coeff = profile.coeff
    if coeff is None:
        return np.empty(0, dtype=np.float64), 0
    count = int(coeff.size)
    limit = grid.L_max + 1
    if count > limit:
        raise ValueError(f"shape profile coefficient count {count} exceeds grid basis size {limit}")
    if np.any(~np.isfinite(coeff)):
        raise ValueError("shape profile coefficients must be finite")
    return coeff, count


def _materialize_profile_fields(
    profile: Profile | None,
    grid: Grid,
) -> np.ndarray:
    """Materialize one profile into a new Reactive cache value via Numba."""

    _validate_equilibrium_grid_tables(grid)
    out = np.empty((3, grid.Nr), dtype=np.float64)
    if profile is None:
        scale = 1.0
        power = 0
        envelope_power = 1
        amplitude_power = 1.0
        offset = 0.0
        coeff = np.empty(0, dtype=np.float64)
        coeff_count = 0
    else:
        coeff, coeff_count = _validated_profile_coefficients(profile, grid)
        scale = profile.scale
        power = profile.power
        envelope_power = profile.envelope_power
        amplitude_power = profile.amplitude_power
        offset = profile.offset
    update_profile_fields(
        out,
        grid.rho,
        grid.T,
        grid.T_r,
        grid.T_rr,
        scale,
        power,
        envelope_power,
        amplitude_power,
        offset,
        coeff,
        coeff_count,
    )
    return _readonly_owned(out)


def _materialize_fourier_profile_fields(
    shape_profiles: dict[str, Profile],
    grid: Grid,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize the Fourier profile families without persistent buffers."""

    fields = np.zeros((2, grid.M_max + 1, 3, grid.Nr), dtype=np.float64)
    for family, prefix in enumerate(("c", "s")):
        first_order = 0 if prefix == "c" else 1
        for order in range(first_order, grid.M_max + 1):
            profile = shape_profiles.get(f"{prefix}{order}")
            if profile is None:
                continue
            fields[family, order] = _materialize_profile_fields(profile, grid)
    _readonly_owned(fields)
    return fields[0], fields[1]


def _validated_radial_root(value: np.ndarray, grid: Grid, name: str) -> np.ndarray:
    expected = (grid.Nr,)
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
    return value


def _readonly_owned(array: np.ndarray) -> np.ndarray:
    """Freeze a newly allocated property result without copying it."""

    array.flags.writeable = False
    return array


def _const_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.flags.writeable:
        array = array.copy()
        array.flags.writeable = False
    return array


def _build_resampled_equilibrium(
    equilibrium: Equilibrium,
    *,
    grid: Grid | None,
) -> Equilibrium:
    source_grid = equilibrium.grid
    plot_grid = grid or Grid(
        Nr=64,
        Nt=64,
        quadrature_scheme="uniform",
        L_max=source_grid.L_max,
        M_max=source_grid.M_max,
        K_max=source_grid.K_max,
    )

    psin, psin_r, psin_rr, FFn_psin, Pn_psin = _resample_equilibrium_root_fields(
        source_grid=source_grid,
        target_grid=plot_grid,
        psin=equilibrium.psin,
        psin_r=equilibrium.psin_r,
        FFn_psin=equilibrium.FFn_psin,
        Pn_psin=equilibrium.Pn_psin,
    )

    return Equilibrium(
        R0=equilibrium.R0,
        Z0=equilibrium.Z0,
        B0=equilibrium.B0,
        a=equilibrium.a,
        grid=plot_grid,
        shape_profiles=equilibrium.shape_profiles,
        psin=psin,
        FFn_psin=FFn_psin,
        Pn_psin=Pn_psin,
        psin_r=psin_r,
        psin_rr=psin_rr,
        p0=equilibrium.p0,
        alpha1=equilibrium.alpha1,
        alpha2=equilibrium.alpha2,
    )


def _endpoint_inclusive_equilibrium(equilibrium: Equilibrium) -> Equilibrium:
    """Return a snapshot view that explicitly owns ``rho=0`` and ``rho=1``."""

    rho = np.asarray(equilibrium.rho, dtype=np.float64)
    if abs(float(rho[0])) <= 1.0e-14 and abs(float(rho[-1]) - 1.0) <= 1.0e-14:
        return equilibrium
    source_grid = equilibrium.grid
    return _build_resampled_equilibrium(
        equilibrium,
        grid=Grid(
            Nr=source_grid.Nr + 2,
            Nt=source_grid.Nt,
            quadrature_scheme="lobatto",
            calculus_scheme=source_grid.calculus_scheme,
            L_max=source_grid.L_max,
            M_max=source_grid.M_max,
            K_max=source_grid.K_max,
        ),
    )


def _resample_equilibrium_root_fields(
    *,
    source_grid: Grid,
    target_grid: Grid,
    psin: np.ndarray,
    psin_r: np.ndarray,
    FFn_psin: np.ndarray,
    Pn_psin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample snapshot-owned radial fields without materializing geometry."""

    source_rho = source_grid.rho
    target_rho = target_grid.rho
    if float(source_rho[0]) > 1.0e-14 or float(source_rho[-1]) < 1.0 - 1.0e-14:
        # Native solve grids are polynomial collocation grids.  Their physical
        # endpoint values are evaluations of the same nodal polynomial, not
        # constant extensions of the first/last interior samples.
        remap = interpolation_matrix(source_rho, target_rho)
        psin_out = remap @ psin
        psin_r_out = remap @ psin_r
        FFn_out = remap @ FFn_psin
        Pn_out = remap @ Pn_psin

        # Differentiate before repairing the unresolved axis cell so the local
        # endpoint extension cannot contaminate the resolved interior through a
        # second global spectral operation.
        psin_rr_out = target_grid.differentiate(psin_r_out)
        _extend_psin_to_missing_axis(
            source_rho,
            target_rho,
            psin,
            psin_r,
            psin_out,
            psin_r_out,
            psin_rr_out,
        )
    else:
        psin_out = _resample_profile_linear(source_rho, psin, target_rho)
        psin_r_out = _resample_profile_linear(source_rho, psin_r, target_rho)
        FFn_out = _resample_profile_linear(source_rho, FFn_psin, target_rho)
        Pn_out = _resample_profile_linear(source_rho, Pn_psin, target_rho)
        psin_rr_out = target_grid.differentiate(psin_r_out)
    return (
        psin_out,
        psin_r_out,
        psin_rr_out,
        FFn_out,
        Pn_out,
    )


def _extend_psin_to_missing_axis(
    source_rho: np.ndarray,
    target_rho: np.ndarray,
    source_psin: np.ndarray,
    source_psin_r: np.ndarray,
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
) -> None:
    """Extend a collocation flux coordinate to the axis with its regular limit.

    Gauss and Radau solve grids may not own ``rho=0``.  Evaluating their global
    nodal polynomial at a substantially closer-to-axis output node is poorly
    conditioned for ``psin_r = O(rho)``: a tiny absolute extrapolation error is
    amplified by diagnostics containing ``1 / psin_r``.  On only the unresolved
    cell spanning the physical axis and the first two source nodes,
    reconstruct the smooth even ratio ``psin_r / rho`` linearly in ``rho**2``
    from those two resolved anchors.  Analytically integrate that representation
    from the axis and shift the remaining primitive by one constant to keep it
    continuous at the repaired-cell boundary.  No resolved derivative anchor or
    source profile is filtered or overwritten.
    """

    if source_rho.size == 0 or float(source_rho[0]) <= 1.0e-14:
        return
    rho0 = float(source_rho[0])
    if source_rho.size < 2:
        return
    rho1 = float(source_rho[1])
    axis = target_rho < rho1
    if not np.any(axis):
        return
    psin1 = float(source_psin[1])
    psin_r0 = float(source_psin_r[0])
    psin_r1 = float(source_psin_r[1])
    if (
        not np.isfinite(psin1)
        or not np.isfinite(psin_r0)
        or not np.isfinite(psin_r1)
        or rho1 <= rho0
    ):
        return

    rho0_sq = rho0 * rho0
    rho1_sq = rho1 * rho1
    ratio0 = psin_r0 / rho0
    ratio1 = psin_r1 / rho1
    ratio_gradient = (ratio1 - ratio0) / (rho1_sq - rho0_sq)
    ratio_axis = ratio0 - ratio_gradient * rho0_sq
    rho_axis = target_rho[axis]
    rho_axis_sq = rho_axis * rho_axis
    out_psin[axis] = rho_axis_sq * (
        0.5 * ratio_axis + 0.25 * ratio_gradient * rho_axis_sq
    )
    out_psin_r[axis] = rho_axis * (
        ratio_axis + ratio_gradient * rho_axis_sq
    )
    out_psin_rr[axis] = ratio_axis + 3.0 * ratio_gradient * rho_axis_sq

    psin_at_rho1 = rho1_sq * (
        0.5 * ratio_axis + 0.25 * ratio_gradient * rho1_sq
    )
    out_psin[~axis] += psin_at_rho1 - psin1


def _resample_profile_linear(
    rho_src: np.ndarray,
    y_src: np.ndarray,
    rho_eval: np.ndarray,
    *,
    left: float | None = None,
    right: float | None = None,
) -> np.ndarray:
    rho_src = np.asarray(rho_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    rho_eval = np.asarray(rho_eval, dtype=np.float64)
    if rho_src.ndim != 1 or y_src.ndim != 1 or rho_eval.ndim != 1 or rho_src.shape != y_src.shape:
        raise ValueError("Expected 1D rho_src/y_src/rho_eval with matching source shape")
    left_val = float(y_src[0]) if left is None else float(left)
    right_val = float(y_src[-1]) if right is None else float(right)
    return np.interp(rho_eval, rho_src, y_src, left=left_val, right=right_val)


def _build_geqdsk_rectilinear_grid(
    *,
    R_range: tuple[float, float],
    Z_range: tuple[float, float],
    NR: int,
    NZ: int | None,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    NR = int(NR)
    if NR < 2:
        raise ValueError(f"NR must be at least 2, got {NR}")

    Zmin, Zmax = map(float, Z_range)
    Rmin, Rmax = map(float, R_range)

    if not np.isfinite(Rmin) or not np.isfinite(Rmax) or Rmax <= Rmin:
        raise ValueError(f"R_range must be finite and increasing, got {R_range!r}")

    if not np.isfinite(Zmin) or not np.isfinite(Zmax) or Zmax <= Zmin:
        raise ValueError(f"Z_range must be finite and increasing, got {Z_range!r}")

    if NZ is None:
        NZ = NR

    if NZ < 2:
        raise ValueError(f"NZ must be at least 2, got {NZ}")

    R_nodes = np.linspace(Rmin, Rmax, NR, dtype=np.float64)
    Z_nodes = np.linspace(Zmin, Zmax, NZ, dtype=np.float64)
    return R_nodes, Z_nodes, Rmin, Rmax, Zmin, Zmax


def _prepare_profile_interp_axis(
    psin_src: np.ndarray, values_src: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    psin_arr = np.asarray(psin_src, dtype=np.float64)
    values_arr = np.asarray(values_src, dtype=np.float64)
    if psin_arr.ndim != 1 or values_arr.ndim != 1 or psin_arr.shape != values_arr.shape:
        raise ValueError("psin_src and values_src must be 1D arrays with the same shape")

    order = np.argsort(psin_arr)
    psin_sorted = psin_arr[order]
    values_sorted = values_arr[order]
    # np.interp requires a strictly increasing axis.  Optimized psin can contain
    # duplicate edge/axis samples after clipping or regularization, so keep the
    # first value for each distinct psin sample.
    psin_unique, unique_indices = np.unique(psin_sorted, return_index=True)
    values_unique = values_sorted[unique_indices]
    if psin_unique.size < 2:
        raise ValueError("Need at least two distinct psin samples to export Geqdsk profiles")
    return psin_unique, values_unique


def _sample_profile_on_uniform_psin(
    psin_src: np.ndarray,
    values_src: np.ndarray,
    psin_eval: np.ndarray,
) -> np.ndarray:
    psin_axis, values_axis = _prepare_profile_interp_axis(psin_src, values_src)
    psin_eval = np.asarray(psin_eval, dtype=np.float64)
    return np.interp(
        psin_eval, psin_axis, values_axis, left=float(values_axis[0]), right=float(values_axis[-1])
    )


def _interpolate_psin_to_rectilinear_grid(
    R_surfaces: np.ndarray,
    Z_surfaces: np.ndarray,
    psin: np.ndarray,
    rho2_src: np.ndarray,
    *,
    R_nodes: np.ndarray,
    Z_nodes: np.ndarray,
    psi_axis: float,
    psi_scale: float,
    psi_outside: float,
) -> np.ndarray:
    R_surfaces = np.asarray(R_surfaces, dtype=np.float64)
    Z_surfaces = np.asarray(Z_surfaces, dtype=np.float64)
    psin = np.asarray(psin, dtype=np.float64)
    rho2_src = np.asarray(rho2_src, dtype=np.float64)
    if R_surfaces.shape != Z_surfaces.shape:
        raise ValueError(
            f"Equilibrium R/Z shape mismatch: {R_surfaces.shape} vs {Z_surfaces.shape}"
        )
    if psin.ndim != 1 or psin.shape[0] != R_surfaces.shape[0]:
        raise ValueError(f"psin must have shape ({R_surfaces.shape[0]},), got {psin.shape}")
    if rho2_src.ndim != 1 or rho2_src.shape[0] != R_surfaces.shape[0]:
        raise ValueError(f"rho2_src must have shape ({R_surfaces.shape[0]},), got {rho2_src.shape}")

    R_grid, Z_grid = np.meshgrid(R_nodes, Z_nodes, indexing="ij")
    # The flux mesh is monotone in rho**2, which gives a better-conditioned
    # interpolation coordinate near the magnetic axis than rho itself.
    rho2_grid = _interpolate_rho2_to_rectilinear_grid(
        R_surfaces,
        Z_surfaces,
        rho2_src,
        R_grid,
        Z_grid,
    )

    psi_grid = np.full(R_grid.shape, float(psi_outside), dtype=np.float64)
    inside = np.isfinite(rho2_grid)
    if np.any(inside):
        # First locate each R/Z cell in the flux-surface mesh, then convert the
        # recovered rho**2 back to the solver's psin profile.
        psi_grid[inside] = float(psi_axis) + float(psi_scale) * np.interp(
            rho2_grid[inside], rho2_src, psin
        )
    return psi_grid


def _interpolate_rho2_to_rectilinear_grid(
    R_surfaces: np.ndarray,
    Z_surfaces: np.ndarray,
    rho2_surfaces: np.ndarray,
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
) -> np.ndarray:
    if R_surfaces.ndim != 2 or Z_surfaces.ndim != 2 or R_surfaces.shape != Z_surfaces.shape:
        raise ValueError(
            f"Expected R_surfaces/Z_surfaces to share a 2D shape, "
            f"got {R_surfaces.shape} and {Z_surfaces.shape}"
        )
    if rho2_surfaces.ndim != 1 or rho2_surfaces.shape[0] != R_surfaces.shape[0]:
        raise ValueError(
            f"rho2_surfaces must have shape ({R_surfaces.shape[0]},), got {rho2_surfaces.shape}"
        )

    points_R, points_Z, point_values, triangles = _build_flux_mesh_triangulation(
        R_surfaces, Z_surfaces, rho2_surfaces
    )
    triangle_mask = _build_degenerate_triangle_mask(points_R, points_Z, triangles)
    rho2_grid = np.full(R_grid.shape, np.nan, dtype=np.float64)
    R_nodes = np.asarray(R_grid[:, 0], dtype=np.float64)
    Z_nodes = np.asarray(Z_grid[0, :], dtype=np.float64)

    for tri, masked in zip(triangles, triangle_mask, strict=True):
        if bool(masked):
            continue
        _rasterize_triangle_to_grid(
            rho2_grid,
            R_grid,
            Z_grid,
            R_nodes,
            Z_nodes,
            points_R[tri],
            points_Z[tri],
            point_values[tri],
        )
    return rho2_grid


def _build_flux_mesh_triangulation(
    R_surfaces: np.ndarray,
    Z_surfaces: np.ndarray,
    rho2_surfaces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nr, nt = R_surfaces.shape
    if nr < 2 or nt < 3:
        raise ValueError(f"Need at least a 2x3 flux mesh, got {(nr, nt)}")

    point_count = 1 + (nr - 1) * nt
    points_R = np.empty(point_count, dtype=np.float64)
    points_Z = np.empty(point_count, dtype=np.float64)
    point_values = np.empty(point_count, dtype=np.float64)

    # All theta samples collapse to one magnetic-axis point.  Store that once,
    # then index later rings with periodic theta wrapping.
    points_R[0] = float(R_surfaces[0, 0])
    points_Z[0] = float(Z_surfaces[0, 0])
    point_values[0] = float(rho2_surfaces[0])

    for i in range(1, nr):
        start = 1 + (i - 1) * nt
        end = start + nt
        points_R[start:end] = R_surfaces[i]
        points_Z[start:end] = Z_surfaces[i]
        point_values[start:end] = float(rho2_surfaces[i])

    triangle_count = nt + 2 * (nr - 2) * nt
    triangles = np.empty((triangle_count, 3), dtype=np.int32)
    cursor = 0

    def vertex_index(i: int, j: int) -> int:
        if i == 0:
            return 0
        return 1 + (i - 1) * nt + (j % nt)

    for j in range(nt):
        # The first ring is triangulated as a fan from the single axis point.
        triangles[cursor] = [vertex_index(0, 0), vertex_index(1, j), vertex_index(1, j + 1)]
        cursor += 1

    for i in range(1, nr - 1):
        for j in range(nt):
            # Each annular quad is split consistently across the periodic theta
            # seam; vertex_index wraps j+1 back to zero.
            triangles[cursor] = [
                vertex_index(i, j),
                vertex_index(i + 1, j),
                vertex_index(i + 1, j + 1),
            ]
            cursor += 1
            triangles[cursor] = [
                vertex_index(i, j),
                vertex_index(i + 1, j + 1),
                vertex_index(i, j + 1),
            ]
            cursor += 1

    return points_R, points_Z, point_values, triangles


def _build_degenerate_triangle_mask(
    points_R: np.ndarray,
    points_Z: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    p0 = triangles[:, 0]
    p1 = triangles[:, 1]
    p2 = triangles[:, 2]
    twice_area = (points_R[p1] - points_R[p0]) * (points_Z[p2] - points_Z[p0]) - (
        points_R[p2] - points_R[p0]
    ) * (points_Z[p1] - points_Z[p0])
    scale = np.maximum(
        np.maximum(np.abs(points_R[p0]), np.abs(points_R[p1])),
        np.maximum(np.abs(points_Z[p0]), np.abs(points_Z[p1])),
    )
    scale = np.maximum(scale, 1.0)
    # Degenerate triangles can appear near the magnetic axis or on highly
    # compressed surfaces.  Masking them avoids unstable barycentric weights in
    # the GEQDSK rasterizer.
    return np.abs(twice_area) <= 1.0e-14 * scale * scale


def _rasterize_triangle_to_grid(
    rho2_grid: np.ndarray,
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
    R_nodes: np.ndarray,
    Z_nodes: np.ndarray,
    tri_R: np.ndarray,
    tri_Z: np.ndarray,
    tri_values: np.ndarray,
) -> None:
    r_min = float(np.min(tri_R))
    r_max = float(np.max(tri_R))
    z_min = float(np.min(tri_Z))
    z_max = float(np.max(tri_Z))
    i0 = int(np.searchsorted(R_nodes, r_min, side="left"))
    i1 = int(np.searchsorted(R_nodes, r_max, side="right"))
    j0 = int(np.searchsorted(Z_nodes, z_min, side="left"))
    j1 = int(np.searchsorted(Z_nodes, z_max, side="right"))
    if i0 >= i1 or j0 >= j1:
        return

    # Restrict barycentric evaluation to the triangle bounding box; full-grid
    # evaluation would dominate GEQDSK export cost for moderate NR/NZ.
    x0, x1, x2 = map(float, tri_R)
    y0, y1, y2 = map(float, tri_Z)
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) <= 1.0e-20:
        return

    sub_R = R_grid[i0:i1, j0:j1]
    sub_Z = Z_grid[i0:i1, j0:j1]
    l0 = ((y1 - y2) * (sub_R - x2) + (x2 - x1) * (sub_Z - y2)) / denom
    l1 = ((y2 - y0) * (sub_R - x2) + (x0 - x2) * (sub_Z - y2)) / denom
    l2 = 1.0 - l0 - l1
    inside = (l0 >= -1.0e-12) & (l1 >= -1.0e-12) & (l2 >= -1.0e-12)
    if not np.any(inside):
        return

    values = l0 * float(tri_values[0]) + l1 * float(tri_values[1]) + l2 * float(tri_values[2])
    target = rho2_grid[i0:i1, j0:j1]
    # Later triangles may touch the same grid node on shared edges.  The
    # interpolated rho**2 is identical up to roundoff, so last writer is fine.
    target[inside] = values[inside]


def _coerce_optional_point_array(value, *, name: str) -> np.ndarray:
    arr = np.asarray(
        value if value is not None else np.empty((0, 2), dtype=np.float64), dtype=np.float64
    )
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {arr.shape}")
    return arr.copy()


# Equilibrium model-side Numba kernels
# ------------------------------------
# These private kernels live with their sole owning model so the Equilibrium
# implementation can migrate independently of solver runtimes and adapters.

_AMPLITUDE_POWER_FLOOR = 1.0e-10

# Public packed geometry rows keep the historical ``surface_fields`` contract.
SIN_TB = 0
R = 1
R_T = 2
Z_T = 3
J = 4
JDIVR = 5
GRTDIVJR_T = 6
GTTDIVJR = 7
GTTDIVJR_R = 8

# Private coordinate intermediates share the same allocation.
R_R = 9
R_RR = 10
R_RT = 11
R_TT = 12
Z = 13
Z_R = 14
Z_RR = 15
Z_RT = 16
Z_TT = 17

S_R = 0
V_R = 1
KN = 2
KN_R = 3
LN_R = 4

# IMAS equilibrium ``profiles_1d`` geometric coefficients.
GM1 = 0
GM2 = 1
GM3 = 2
GM4 = 3
GM5 = 4
GM6 = 5
GM7 = 6
GM8 = 7
GM9 = 8

RHO_TOR = 0
RHO_TOR_NORM = 1
RHO_TOR_R = 2
RHO_TOR_NORM_R = 3


@njit(cache=True, nogil=True, inline="always")
def _power_terms_at(rho: float, power: int) -> tuple[float, float, float]:
    if power == 0:
        return 1.0, 0.0, 0.0
    value = rho**power
    first = power * rho ** (power - 1)
    second = 0.0 if power == 1 else power * (power - 1) * rho ** (power - 2)
    return value, first, second


@njit(cache=True, nogil=True, inline="always")
def _envelope_terms_at(rho: float, envelope_power: int) -> tuple[float, float, float]:
    if envelope_power == 0:
        return 1.0, 0.0, 0.0
    y = 1.0 - rho * rho
    if envelope_power == 1:
        return y, -2.0 * rho, -2.0
    value = y**envelope_power
    first = -2.0 * envelope_power * rho * y ** (envelope_power - 1)
    second = -2.0 * envelope_power * y ** (envelope_power - 1)
    second += 4.0 * envelope_power * (envelope_power - 1) * rho * rho * y ** (envelope_power - 2)
    return value, first, second


@njit(cache=True, nogil=True, inline="always")
def _amplitude_terms(
    value: float,
    first: float,
    second: float,
    amplitude_power: float,
) -> tuple[float, float, float]:
    if amplitude_power == 1.0:
        return value, first, second
    safe = max(value, _AMPLITUDE_POWER_FLOOR)
    if amplitude_power == 0.5:
        powered = np.sqrt(safe)
        inv = 1.0 / powered
        return (
            powered,
            0.5 * first * inv,
            0.5 * second * inv - 0.25 * first * first * inv / safe,
        )
    powered = safe**amplitude_power
    powered_r = amplitude_power * safe ** (amplitude_power - 1.0) * first
    powered_rr = amplitude_power * safe ** (amplitude_power - 1.0) * second
    powered_rr += (
        amplitude_power * (amplitude_power - 1.0) * safe ** (amplitude_power - 2.0) * first * first
    )
    return powered, powered_r, powered_rr


@njit(cache=True, nogil=True)
def update_profile_fields(
    out: np.ndarray,
    rho: np.ndarray,
    T: np.ndarray,
    T_r: np.ndarray,
    T_rr: np.ndarray,
    scale: float,
    power: int,
    envelope_power: int,
    amplitude_power: float,
    offset: float,
    coeff: np.ndarray,
    coeff_count: int,
) -> None:
    """Evaluate one profile and two radial derivatives without temporaries."""

    for i in range(rho.shape[0]):
        rp, rp_r, rp_rr = _power_terms_at(rho[i], power)
        if coeff_count == 0:
            amp, amp_r, amp_rr = _amplitude_terms(offset, 0.0, 0.0, amplitude_power)
        else:
            series = 0.0
            series_r = 0.0
            series_rr = 0.0
            for k in range(coeff_count):
                coefficient = coeff[k]
                series += coefficient * T[k, i]
                series_r += coefficient * T_r[k, i]
                series_rr += coefficient * T_rr[k, i]
            env, env_r, env_rr = _envelope_terms_at(rho[i], envelope_power)
            base = env * series
            base_r = env_r * series + env * series_r
            base_rr = env_rr * series + 2.0 * env_r * series_r + env * series_rr
            amp, amp_r, amp_rr = _amplitude_terms(offset + base, base_r, base_rr, amplitude_power)

        out[0, i] = scale * rp * amp
        out[1, i] = scale * (rp_r * amp + rp * amp_r)
        out[2, i] = scale * (rp_rr * amp + 2.0 * rp_r * amp_r + rp * amp_rr)


@njit(cache=True, nogil=True)
def update_r_coordinates(
    surface: np.ndarray,
    a: float,
    R0: float,
    rho: np.ndarray,
    theta: np.ndarray,
    cos_mtheta: np.ndarray,
    sin_mtheta: np.ndarray,
    m_cos_mtheta: np.ndarray,
    m_sin_mtheta: np.ndarray,
    m2_cos_mtheta: np.ndarray,
    m2_sin_mtheta: np.ndarray,
    h: np.ndarray,
    c: np.ndarray,
    s: np.ndarray,
) -> None:
    c_limit = min(c.shape[0], cos_mtheta.shape[0])
    s_limit = min(s.shape[0], sin_mtheta.shape[0])
    for i in range(rho.shape[0]):
        rho_i = rho[i]
        for j in range(theta.shape[0]):
            tb = theta[j] + c[0, 0, i]
            tb_r = c[0, 1, i]
            tb_t = 1.0
            tb_rr = c[0, 2, i]
            tb_rt = 0.0
            tb_tt = 0.0
            for order in range(1, c_limit):
                ci = c[order, 0, i]
                cir = c[order, 1, i]
                cirr = c[order, 2, i]
                tb += ci * cos_mtheta[order, j]
                tb_r += cir * cos_mtheta[order, j]
                tb_t -= ci * m_sin_mtheta[order, j]
                tb_rr += cirr * cos_mtheta[order, j]
                tb_rt -= cir * m_sin_mtheta[order, j]
                tb_tt -= ci * m2_cos_mtheta[order, j]
            for order in range(1, s_limit):
                si = s[order, 0, i]
                sir = s[order, 1, i]
                sirr = s[order, 2, i]
                tb += si * sin_mtheta[order, j]
                tb_r += sir * sin_mtheta[order, j]
                tb_t += si * m_cos_mtheta[order, j]
                tb_rr += sirr * sin_mtheta[order, j]
                tb_rt += sir * m_cos_mtheta[order, j]
                tb_tt -= si * m2_sin_mtheta[order, j]

            cos_tb = np.cos(tb)
            sin_tb = np.sin(tb)
            radius = R0 + a * (h[0, i] + rho_i * cos_tb)
            if radius < 1.0e-6:
                radius = 1.0e-6
            surface[SIN_TB, i, j] = sin_tb
            surface[R, i, j] = radius
            surface[R_R, i, j] = a * (h[1, i] + cos_tb - rho_i * sin_tb * tb_r)
            surface[R_T, i, j] = -a * rho_i * sin_tb * tb_t
            surface[R_RR, i, j] = a * (
                h[2, i] - 2.0 * sin_tb * tb_r - rho_i * (cos_tb * tb_r * tb_r + sin_tb * tb_rr)
            )
            surface[R_RT, i, j] = -a * (
                sin_tb * tb_t + rho_i * (cos_tb * tb_r * tb_t + sin_tb * tb_rt)
            )
            surface[R_TT, i, j] = -a * rho_i * (cos_tb * tb_t * tb_t + sin_tb * tb_tt)


@njit(cache=True, nogil=True)
def update_z_coordinates(
    surface: np.ndarray,
    a: float,
    Z0: float,
    rho: np.ndarray,
    sin_theta: np.ndarray,
    cos_theta: np.ndarray,
    v: np.ndarray,
    kappa: np.ndarray,
) -> None:
    for i in range(rho.shape[0]):
        rho_i = rho[i]
        for j in range(sin_theta.shape[0]):
            sin_t = sin_theta[j]
            cos_t = cos_theta[j]
            surface[Z, i, j] = Z0 + a * (v[0, i] - rho_i * kappa[0, i] * sin_t)
            surface[Z_R, i, j] = a * (v[1, i] - (kappa[0, i] + rho_i * kappa[1, i]) * sin_t)
            surface[Z_T, i, j] = -a * rho_i * kappa[0, i] * cos_t
            surface[Z_RR, i, j] = a * (v[2, i] - (2.0 * kappa[1, i] + rho_i * kappa[2, i]) * sin_t)
            surface[Z_RT, i, j] = -a * (kappa[0, i] + rho_i * kappa[1, i]) * cos_t
            surface[Z_TT, i, j] = a * rho_i * kappa[0, i] * sin_t


@njit(cache=True, nogil=True, inline="always")
def _axis_even_rho2_limit(value_1: float, value_2: float, rho: np.ndarray) -> float:
    """Extrapolate an even finite scalar from the first two off-axis rows."""

    x0 = rho[0] * rho[0]
    x1 = rho[1] * rho[1]
    x2 = rho[2] * rho[2]
    denominator = x2 - x1
    if denominator == 0.0 or not np.isfinite(value_1) or not np.isfinite(value_2):
        return np.nan
    return value_1 + (value_2 - value_1) / denominator * (x0 - x1)


@njit(cache=True, nogil=True, inline="always")
def _axis_linear_rho_limit(value_1: float, value_2: float, rho: np.ndarray) -> float:
    """Extrapolate a local surface quantity linearly in ``rho``."""

    denominator = rho[2] - rho[1]
    if denominator == 0.0 or not np.isfinite(value_1) or not np.isfinite(value_2):
        return np.nan
    return value_1 + (value_2 - value_1) / denominator * (rho[0] - rho[1])


@njit(cache=True, nogil=True, inline="always")
def _axis_leading_rho_coefficient(value_1: float, value_2: float, rho: np.ndarray) -> float:
    """Recover the leading coefficient of ``value = rho*(A + O(rho))``."""

    if rho[1] == 0.0 or rho[2] == 0.0:
        return np.nan
    return _axis_linear_rho_limit(value_1 / rho[1], value_2 / rho[2], rho)


@njit(cache=True, nogil=True)
def update_metric_geometry(
    surface: np.ndarray,
    radial: np.ndarray,
    rho: np.ndarray,
) -> int:
    nr = surface.shape[1]
    nt = surface.shape[2]
    has_axis = nr >= 3 and abs(rho[0]) < 1.0e-10
    theta_scale = 2.0 * np.pi / nt
    mean_scale = 1.0 / nt
    for i in range(nr):
        sum_j = 0.0
        sum_jr = 0.0
        sum_gtt = 0.0
        sum_gtt_r = 0.0
        sum_jdivr = 0.0
        for j in range(nt):
            radius = surface[R, i, j]
            rr = surface[R_R, i, j]
            rt = surface[R_T, i, j]
            rrr = surface[R_RR, i, j]
            rrt = surface[R_RT, i, j]
            rtt = surface[R_TT, i, j]
            zr = surface[Z_R, i, j]
            zt = surface[Z_T, i, j]
            zrr = surface[Z_RR, i, j]
            zrt = surface[Z_RT, i, j]
            ztt = surface[Z_TT, i, j]

            jac = rt * zr - rr * zt
            jac_r = -(rrr * zt - rrt * zr + rr * zrt - rt * zrr)
            jac_t = -(rrt * zt - rtt * zr + rr * ztt - rt * zrt)
            jr = jac * radius
            jr_r = jac_r * radius + jac * rr
            jr_t = jac_t * radius + jac * rt
            grt = rr * rt + zr * zt
            grt_t = rrt * rt + rr * rtt + zrt * zt + zr * ztt
            gtt = rt * rt + zt * zt
            gtt_r = 2.0 * (rt * rrt + zt * zrt)
            jdivr = jac / radius
            if has_axis and i == 0:
                # The public Jacobian remains the raw coordinate Jacobian.  The
                # three metric ratios have removable axis singularities and are
                # reconstructed after all off-axis rows are available.
                grtdivjr_t = np.nan
                gttdivjr = np.nan
                gttdivjr_r = np.nan
            else:
                if jr == 0.0 or not np.isfinite(jr):
                    return i + 1
                inv_jr = 1.0 / jr
                grtdivjr_t = (grt_t - grt * jr_t * inv_jr) * inv_jr
                gttdivjr = gtt * inv_jr
                gttdivjr_r = gtt_r * inv_jr - gtt * jr_r * inv_jr * inv_jr

            surface[J, i, j] = jac
            surface[JDIVR, i, j] = jdivr
            surface[GRTDIVJR_T, i, j] = grtdivjr_t
            surface[GTTDIVJR, i, j] = gttdivjr
            surface[GTTDIVJR_R, i, j] = gttdivjr_r
            sum_j += jac
            sum_jr += jr
            if not (has_axis and i == 0):
                sum_gtt += gttdivjr
                sum_gtt_r += gttdivjr_r
            sum_jdivr += jdivr

        radial[S_R, i] = sum_j * theta_scale
        radial[V_R, i] = sum_jr * theta_scale * 2.0 * np.pi
        radial[KN, i] = sum_gtt * mean_scale
        radial[KN_R, i] = sum_gtt_r * mean_scale
        radial[LN_R, i] = sum_jdivr * mean_scale

    if has_axis:
        for j in range(nt):
            leading_gtt = _axis_leading_rho_coefficient(
                surface[GTTDIVJR, 1, j],
                surface[GTTDIVJR, 2, j],
                rho,
            )
            surface[GTTDIVJR, 0, j] = rho[0] * leading_gtt
            surface[GTTDIVJR_R, 0, j] = leading_gtt
            surface[GRTDIVJR_T, 0, j] = _axis_linear_rho_limit(
                surface[GRTDIVJR_T, 1, j],
                surface[GRTDIVJR_T, 2, j],
                rho,
            )
        sum_gtt = 0.0
        sum_gtt_r = 0.0
        for j in range(nt):
            sum_gtt += surface[GTTDIVJR, 0, j]
            sum_gtt_r += surface[GTTDIVJR_R, 0, j]
        radial[KN, 0] = sum_gtt * mean_scale
        radial[KN_R, 0] = sum_gtt_r * mean_scale

    for i in range(nr):
        for j in range(nt):
            if not np.isfinite(surface[J, i, j]):
                return i + 1
            if not np.isfinite(surface[JDIVR, i, j]):
                return i + 1
            if not np.isfinite(surface[GRTDIVJR_T, i, j]):
                return i + 1
            if not np.isfinite(surface[GTTDIVJR, i, j]):
                return i + 1
            if not np.isfinite(surface[GTTDIVJR_R, i, j]):
                return i + 1
        for field in range(radial.shape[0]):
            if not np.isfinite(radial[field, i]):
                return i + 1
    return 0


@njit(cache=True, nogil=True)
def materialize_metric_geometry(
    r_surface: np.ndarray,
    z_surface: np.ndarray,
    rho: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Combine independent coordinate stages and materialize metric outputs."""

    surface = np.empty_like(r_surface)
    radial = np.empty((5, rho.shape[0]), dtype=np.float64)
    r_rows = (SIN_TB, R, R_R, R_T, R_RR, R_RT, R_TT)
    z_rows = (Z, Z_R, Z_T, Z_RR, Z_RT, Z_TT)
    for row in r_rows:
        for i in range(surface.shape[1]):
            for j in range(surface.shape[2]):
                surface[row, i, j] = r_surface[row, i, j]
    for row in z_rows:
        for i in range(surface.shape[1]):
            for j in range(surface.shape[2]):
                surface[row, i, j] = z_surface[row, i, j]
    invalid = update_metric_geometry(surface, radial, rho)
    return surface[:9], radial, invalid


@njit(cache=True, nogil=True)
def update_rc(out: np.ndarray, R0: float, a: float, h: np.ndarray) -> None:
    for i in range(out.shape[0]):
        out[i] = R0 + a * h[i]


@njit(cache=True, nogil=True)
def update_epsilon(out: np.ndarray, a: float, rho: np.ndarray, rc: np.ndarray) -> None:
    for i in range(out.shape[0]):
        out[i] = a * rho[i] / rc[i]


@njit(cache=True, nogil=True)
def update_surface_area(out: np.ndarray, radius: np.ndarray, z_t: np.ndarray) -> None:
    scale = -2.0 * np.pi / radius.shape[1]
    for i in range(radius.shape[0]):
        total = 0.0
        for j in range(radius.shape[1]):
            total += radius[i, j] * z_t[i, j]
        out[i] = scale * total


@njit(cache=True, nogil=True)
def update_volume(out: np.ndarray, radius: np.ndarray, z_t: np.ndarray) -> None:
    scale = -2.0 * np.pi * np.pi / radius.shape[1]
    for i in range(radius.shape[0]):
        total = 0.0
        for j in range(radius.shape[1]):
            total += radius[i, j] * radius[i, j] * z_t[i, j]
        out[i] = scale * total


@njit(cache=True, nogil=True)
def update_scaled_product(
    out: np.ndarray, left: np.ndarray, right: np.ndarray, scale: float
) -> None:
    for i in range(out.shape[0]):
        out[i] = scale * left[i] * right[i]


@njit(cache=True, nogil=True)
def update_scaled_copy(out: np.ndarray, source: np.ndarray, scale: float) -> None:
    for i in range(out.shape[0]):
        out[i] = scale * source[i]


@njit(cache=True, nogil=True)
def update_f2(
    out: np.ndarray,
    ff_r: np.ndarray,
    accumulator: np.ndarray,
    weights: np.ndarray,
    edge_f2: float,
) -> None:
    edge = 0.0
    for k in range(out.shape[0]):
        edge += weights[k] * ff_r[k]
    for i in range(out.shape[0]):
        total = 0.0
        for k in range(out.shape[0]):
            total += accumulator[i, k] * ff_r[k]
        out[i] = edge_f2 + 2.0 * (total - edge)


@njit(cache=True, nogil=True)
def update_f(out: np.ndarray, f2: np.ndarray, edge_f: float) -> int:
    sign = -1.0 if edge_f < 0.0 else 1.0
    for i in range(out.shape[0]):
        if f2[i] < 1.0e-6:
            return i + 1
        out[i] = sign * np.sqrt(f2[i])
    return 0


@njit(cache=True, nogil=True)
def update_pressure(
    out: np.ndarray,
    p_r: np.ndarray,
    accumulator: np.ndarray,
    weights: np.ndarray,
    p0: float,
) -> None:
    edge_integral = 0.0
    for k in range(out.shape[0]):
        edge_integral += weights[k] * p_r[k]
    for i in range(out.shape[0]):
        prefix = 0.0
        for k in range(out.shape[0]):
            prefix += accumulator[i, k] * p_r[k]
        out[i] = p0 + prefix - edge_integral


@njit(cache=True, nogil=True)
def update_beta_t(
    pressure: np.ndarray,
    volume_r: np.ndarray,
    weights: np.ndarray,
    B0: float,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for i in range(pressure.shape[0]):
        weighted_volume = weights[i] * volume_r[i]
        numerator += weighted_volume * pressure[i]
        denominator += weighted_volume
    return 2.0 * MU0 * numerator / (denominator * B0 * B0)


@njit(cache=True, nogil=True)
def update_gn1(
    out: np.ndarray,
    radius: np.ndarray,
    jdivr: np.ndarray,
    ffn_psin: np.ndarray,
    pn_psin: np.ndarray,
) -> None:
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            r = radius[i, j]
            out[i, j] = jdivr[i, j] * (ffn_psin[i] + r * r * pn_psin[i])


@njit(cache=True, nogil=True)
def update_gn2(
    out: np.ndarray,
    gttdivjr: np.ndarray,
    gttdivjr_r: np.ndarray,
    grtdivjr_t: np.ndarray,
    psin_r: np.ndarray,
    psin_rr: np.ndarray,
) -> None:
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = gttdivjr[i, j] * psin_rr[i]
            out[i, j] += (gttdivjr_r[i, j] - grtdivjr_t[i, j]) * psin_r[i]


@njit(cache=True, nogil=True)
def update_linear_combination_2d(
    out: np.ndarray, left: np.ndarray, right: np.ndarray, a: float, b: float
) -> None:
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = a * left[i, j] + b * right[i, j]


@njit(cache=True, nogil=True)
def update_ip(gn1: np.ndarray, weights: np.ndarray, alpha1: float) -> float:
    total = 0.0
    for i in range(gn1.shape[0]):
        row_sum = 0.0
        for j in range(gn1.shape[1]):
            row_sum += gn1[i, j]
        total += weights[i] * row_sum
    return -alpha1 * (2.0 * np.pi / gn1.shape[1]) * total / MU0


@njit(cache=True, nogil=True, inline="always")
def _regularize_axis_rho2_1d(values: np.ndarray, rho: np.ndarray) -> None:
    if values.shape[0] < 3 or abs(rho[0]) >= 1.0e-10:
        return
    values[0] = _axis_even_rho2_limit(values[1], values[2], rho)


@njit(cache=True, nogil=True, inline="always")
def _regularize_axis_sqrt_rho_1d(values: np.ndarray, rho: np.ndarray) -> None:
    """Recover a quantity with leading ``sqrt(rho)`` axis behavior."""

    if values.shape[0] < 3 or abs(rho[0]) >= 1.0e-10:
        return
    if rho[1] <= 0.0 or rho[2] <= 0.0:
        values[0] = np.nan
        return
    coefficient = _axis_linear_rho_limit(
        values[1] / np.sqrt(rho[1]),
        values[2] / np.sqrt(rho[2]),
        rho,
    )
    values[0] = np.sqrt(rho[0]) * coefficient


@njit(cache=True, nogil=True, inline="always")
def _first_nonfinite_1d(values: np.ndarray) -> int:
    for i in range(values.shape[0]):
        if not np.isfinite(values[i]):
            return i + 1
    return 0


@njit(cache=True, nogil=True)
def update_q(
    out: np.ndarray,
    f: np.ndarray,
    ln_r: np.ndarray,
    alpha2: float,
    psin_r: np.ndarray,
    rho: np.ndarray,
) -> int:
    for i in range(out.shape[0]):
        denominator = alpha2 * psin_r[i]
        out[i] = np.nan if denominator == 0.0 else f[i] * ln_r[i] / denominator
    _regularize_axis_rho2_1d(out, rho)
    return _first_nonfinite_1d(out)


@njit(cache=True, nogil=True)
def update_shear(
    out: np.ndarray,
    q: np.ndarray,
    rho: np.ndarray,
    differentiator: np.ndarray,
) -> None:
    for i in range(out.shape[0]):
        derivative = 0.0
        for k in range(out.shape[0]):
            derivative += differentiator[i, k] * q[k]
        out[i] = rho[i] * derivative / q[i]


@njit(cache=True, nogil=True)
def update_itor(out: np.ndarray, kn: np.ndarray, alpha2: float, psin_r: np.ndarray) -> None:
    scale = 2.0 * np.pi * alpha2 / MU0
    for i in range(out.shape[0]):
        out[i] = scale * kn[i] * psin_r[i]


@njit(cache=True, nogil=True)
def update_jtor(
    out: np.ndarray,
    ffn_psin: np.ndarray,
    pn_psin: np.ndarray,
    ln_r: np.ndarray,
    s_r: np.ndarray,
    v_r: np.ndarray,
    itor: np.ndarray,
    surface_area: np.ndarray,
    alpha1: float,
    rho: np.ndarray,
) -> int:
    for i in range(out.shape[0]):
        if s_r[i] == 0.0:
            out[i] = np.nan
        else:
            out[i] = (
                -alpha1
                / (MU0 * s_r[i])
                * (2.0 * np.pi * ffn_psin[i] * ln_r[i] + v_r[i] * pn_psin[i] / (2.0 * np.pi))
            )
    if out.shape[0] >= 3 and abs(rho[0]) < 1.0e-10:
        # Close the removable axis limit with the two conservative primitives
        # owned by this Equilibrium.  Both I_tor and poloidal cross-section S
        # vanish as rho**2, so I_tor/S tends to dI_tor/dS = j_phi.  Extrapolating
        # this finite ratio in rho**2 is much better conditioned than either the
        # source-form 0/0 quotient or a second spectral derivative at the axis.
        if surface_area[1] == 0.0 or surface_area[2] == 0.0:
            out[0] = np.nan
        else:
            out[0] = _axis_even_rho2_limit(
                itor[1] / surface_area[1],
                itor[2] / surface_area[2],
                rho,
            )
    return _first_nonfinite_1d(out)


@njit(cache=True, nogil=True)
def update_jpara(
    out: np.ndarray,
    f: np.ndarray,
    kn: np.ndarray,
    kn_r: np.ndarray,
    ln_r: np.ndarray,
    psin_r: np.ndarray,
    psin_rr: np.ndarray,
    alpha2: float,
    rho: np.ndarray,
    differentiator: np.ndarray,
) -> int:
    for i in range(out.shape[0]):
        if f[i] == 0.0 or ln_r[i] == 0.0:
            out[i] = np.nan
        else:
            derivative = 0.0
            for k in range(out.shape[0]):
                derivative += differentiator[i, k] * f[k]
            term = kn_r[i] * psin_r[i] / f[i]
            term += kn[i] * psin_rr[i] / f[i]
            term -= kn[i] * psin_r[i] * derivative / (f[i] * f[i])
            out[i] = alpha2 / MU0 * f[i] / ln_r[i] * term
    _regularize_axis_rho2_1d(out, rho)
    return _first_nonfinite_1d(out)


@njit(cache=True, nogil=True)
def update_jtotal(
    out: np.ndarray,
    jpara: np.ndarray,
    f: np.ndarray,
    ln_r: np.ndarray,
    volume_r: np.ndarray,
    B0: float,
    rho: np.ndarray,
) -> int:
    """Write the IMAS convention ``<J·B>/B0`` from the PJ2 current."""

    gm1_scale = (2.0 * np.pi) ** 2
    for i in range(out.shape[0]):
        if volume_r[i] == 0.0 or B0 == 0.0:
            out[i] = np.nan
        else:
            gm1 = gm1_scale * ln_r[i] / volume_r[i]
            out[i] = jpara[i] * f[i] * gm1 / B0
    _regularize_axis_rho2_1d(out, rho)
    return _first_nonfinite_1d(out)


@njit(cache=True, nogil=True)
def update_jphi(
    out: np.ndarray,
    radius: np.ndarray,
    ffn_psin: np.ndarray,
    pn_psin: np.ndarray,
    alpha1: float,
) -> None:
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            r = radius[i, j]
            out[i, j] = -alpha1 / (MU0 * r) * (ffn_psin[i] + r * r * pn_psin[i])


@njit(cache=True, nogil=True)
def update_phi(
    out: np.ndarray,
    f: np.ndarray,
    ln_r: np.ndarray,
    accumulator: np.ndarray,
) -> None:
    for i in range(out.shape[0]):
        total = 0.0
        for k in range(out.shape[0]):
            total += accumulator[i, k] * f[k] * ln_r[k]
        out[i] = 2.0 * np.pi * total


@njit(cache=True, nogil=True)
def update_toroidal_flux_coordinates(
    out: np.ndarray,
    phi: np.ndarray,
    f: np.ndarray,
    ln_r: np.ndarray,
    B0: float,
    rho_tor_edge: float,
    rho: np.ndarray,
) -> int:
    """Materialize IMAS ``rho_tor`` coordinates and radial derivatives."""

    if B0 == 0.0 or not np.isfinite(B0) or rho_tor_edge <= 0.0 or not np.isfinite(rho_tor_edge):
        return 1

    for i in range(phi.shape[0]):
        rho_tor_squared = phi[i] / (np.pi * B0)
        if rho_tor_squared < 0.0 or not np.isfinite(rho_tor_squared):
            return i + 1
        rho_tor = np.sqrt(rho_tor_squared)
        out[RHO_TOR, i] = rho_tor
        if rho_tor == 0.0:
            out[RHO_TOR_R, i] = np.nan
        else:
            out[RHO_TOR_R, i] = f[i] * ln_r[i] / (B0 * rho_tor)

    _regularize_axis_rho2_1d(out[RHO_TOR_R], rho)
    for i in range(phi.shape[0]):
        out[RHO_TOR_NORM, i] = out[RHO_TOR, i] / rho_tor_edge
        out[RHO_TOR_NORM_R, i] = out[RHO_TOR_R, i] / rho_tor_edge

    for row in range(out.shape[0]):
        invalid = _first_nonfinite_1d(out[row])
        if invalid:
            return invalid
    return 0


@njit(cache=True, nogil=True)
def update_gm(
    out: np.ndarray,
    radius: np.ndarray,
    jacobian: np.ndarray,
    gttdivjr: np.ndarray,
    f: np.ndarray,
    psin_r: np.ndarray,
    ln_r: np.ndarray,
    surface_area_r: np.ndarray,
    volume_r: np.ndarray,
    rho_tor_r: np.ndarray,
    alpha2: float,
    rho: np.ndarray,
) -> int:
    """Materialize the IMAS gm1--gm9 flux-surface geometry coefficients."""

    has_axis = out.shape[1] >= 3 and abs(rho[0]) < 1.0e-10
    gm1_scale = (2.0 * np.pi) ** 2
    gm9_scale = 2.0 * np.pi
    for i in range(out.shape[1]):
        if has_axis and i == 0:
            for field in range(out.shape[0]):
                out[field, i] = np.nan
            continue
        if volume_r[i] == 0.0 or not np.isfinite(volume_r[i]):
            return i + 1

        weight_sum = 0.0
        gm2_sum = 0.0
        gm3_sum = 0.0
        gm4_sum = 0.0
        gm5_sum = 0.0
        gm6_sum = 0.0
        gm7_sum = 0.0
        gm8_sum = 0.0
        for j in range(radius.shape[1]):
            r = radius[i, j]
            jac = jacobian[i, j]
            jr = jac * r
            if r == 0.0 or jac == 0.0 or not np.isfinite(jr):
                return i + 1

            bphi2 = (f[i] / r) ** 2
            bp2 = (alpha2 * psin_r[i]) ** 2 * gttdivjr[i, j] / jr
            grad_rho_tor2 = rho_tor_r[i] * rho_tor_r[i] * gttdivjr[i, j] * r / jac
            b2 = bphi2 + bp2
            if (
                b2 <= 0.0
                or grad_rho_tor2 < 0.0
                or not np.isfinite(b2)
                or not np.isfinite(grad_rho_tor2)
            ):
                return i + 1

            weight_sum += jr
            gm2_sum += jr * grad_rho_tor2 / (r * r)
            gm3_sum += jr * grad_rho_tor2
            gm4_sum += jr / b2
            gm5_sum += jr * b2
            gm6_sum += jr * grad_rho_tor2 / b2
            gm7_sum += jr * np.sqrt(grad_rho_tor2)
            gm8_sum += jr * r

        if weight_sum == 0.0 or not np.isfinite(weight_sum):
            return i + 1
        inv_weight = 1.0 / weight_sum
        out[GM1, i] = gm1_scale * ln_r[i] / volume_r[i]
        out[GM2, i] = gm2_sum * inv_weight
        out[GM3, i] = gm3_sum * inv_weight
        out[GM4, i] = gm4_sum * inv_weight
        out[GM5, i] = gm5_sum * inv_weight
        out[GM6, i] = gm6_sum * inv_weight
        out[GM7, i] = gm7_sum * inv_weight
        out[GM8, i] = gm8_sum * inv_weight
        out[GM9, i] = gm9_scale * surface_area_r[i] / volume_r[i]

    if has_axis:
        for field in range(out.shape[0]):
            _regularize_axis_rho2_1d(out[field], rho)

    for field in range(out.shape[0]):
        invalid = _first_nonfinite_1d(out[field])
        if invalid:
            return invalid
    return 0


@njit(cache=True, nogil=True)
def update_ftrap(
    out: np.ndarray,
    radius: np.ndarray,
    jacobian: np.ndarray,
    gttdivjr: np.ndarray,
    f: np.ndarray,
    psin_r: np.ndarray,
    alpha2: float,
    rho: np.ndarray,
) -> int:
    has_axis = out.shape[0] >= 3 and abs(rho[0]) < 1.0e-10
    for i in range(out.shape[0]):
        if has_axis and i == 0:
            out[i] = np.nan
            continue
        weight_sum = 0.0
        b_sum = 0.0
        b2_sum = 0.0
        bmax = 0.0
        for j in range(radius.shape[1]):
            r = radius[i, j]
            jac = jacobian[i, j]
            bphi2 = (f[i] / r) ** 2
            bp2 = (alpha2 * psin_r[i]) ** 2 * gttdivjr[i, j] / (jac * r)
            magnetic = np.sqrt(bphi2 + bp2)
            weight = jac * r
            weight_sum += weight
            b_sum += magnetic * weight
            b2_sum += magnetic * magnetic * weight
            bmax = max(bmax, magnetic)
        if weight_sum == 0.0 or bmax == 0.0:
            out[i] = np.nan
            continue
        h = b_sum / (weight_sum * bmax)
        h2 = b2_sum / (weight_sum * bmax * bmax)
        hf_sum = 0.0
        for j in range(radius.shape[1]):
            r = radius[i, j]
            jac = jacobian[i, j]
            bphi2 = (f[i] / r) ** 2
            bp2 = (alpha2 * psin_r[i]) ** 2 * gttdivjr[i, j] / (jac * r)
            x = np.sqrt(bphi2 + bp2) / bmax
            one_minus_x = 1.0 - x
            if one_minus_x < 0.0 and one_minus_x > -1.0e-14:
                one_minus_x = 0.0
            if one_minus_x < 0.0 or x == 0.0:
                hf_sum = np.nan
                break
            integrand = (1.0 - np.sqrt(one_minus_x) * (1.0 + 0.5 * x)) / (x * x)
            hf_sum += integrand * jac * r
        hf = hf_sum / weight_sum
        ftu = 1.0 - h2 / (h * h) * (1.0 - np.sqrt(1.0 - h) * (1.0 + 0.5 * h))
        ftl = 1.0 - h2 * hf
        out[i] = 0.75 * ftu + 0.25 * ftl
    _regularize_axis_sqrt_rho_1d(out, rho)
    return _first_nonfinite_1d(out)
