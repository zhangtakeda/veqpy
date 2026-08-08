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

from pathlib import Path
from typing import Self

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rich.console import Console
from rich.text import Text
from rich.tree import Tree

from veqpy.base import Reactive, Serial
from veqpy.model.geqdsk import Geqdsk
from veqpy.model.grid import Grid
from veqpy.model.profile import Profile

from . import _equilibrium_numba as eqnb

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "lines.linewidth": 1.5,
        "legend.frameon": False,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "black",
        "lines.markersize": 4,
        "legend.labelspacing": 0.3,
        "legend.columnspacing": 0.6,
    }
)
BLACK = "black"
BLUE = mcolors.TABLEAU_COLORS["tab:blue"]
ORANGE = mcolors.TABLEAU_COLORS["tab:orange"]
GREEN = mcolors.TABLEAU_COLORS["tab:green"]
RED = mcolors.TABLEAU_COLORS["tab:red"]
PURPLE = mcolors.TABLEAU_COLORS["tab:purple"]

PLOT_FONT_FAMILY = "DejaVu Sans"
PLOT_BASE_FONT_SIZE = 10
PLOT_MATH_FONTSET = "dejavusans"
TITLE_FONT_SIZE = 9
AXIS_LABEL_FONT_SIZE = 9
TICK_LABEL_FONT_SIZE = 8
LEGEND_FONT_SIZE = 8
SINGLE_COLUMN_WIDTH = 4.75
DOUBLE_COLUMN_WIDTH = 10.0

EQUILIBRIUM_PLOT_WIDTH = DOUBLE_COLUMN_WIDTH
EQUILIBRIUM_PLOT_HEIGHT = 4.0
EQUILIBRIUM_COMPACT_PLOT_WIDTH = SINGLE_COLUMN_WIDTH
EQUILIBRIUM_COMPACT_PLOT_HEIGHT = 4.0

GRID_SPEC_NROWS = 3
GRID_SPEC_NCOLS = 2
GRID_SPEC_WIDTH_RATIOS = [1, 1]
GRID_SPEC_HEIGHT_RATIOS = [1, 1, 1]
GRID_SPEC_HSPACE = 0.25
GRID_SPEC_WSPACE = 0.45
GRID_SPEC_TOP = 0.94
GRID_SPEC_BOTTOM = 0.12
GRID_SPEC_LEFT = 0.10
GRID_SPEC_RIGHT = 0.985

SUBPLOT_TITLE_FONTSIZE = TITLE_FONT_SIZE

GRID_ALPHA = 0.25
GRID_LINESTYLE = "-"
GRID_LINE_WIDTH = 0.5
LINE_WIDTH = 1.5
LEGEND_COLUMN_SPACING = 0.6
LEGEND_LABEL_SPACING = 0.1
PLOT_TICK_DIRECTION = "in"
PLOT_TICK_TOP = True
PLOT_TICK_RIGHT = True
PLOT_TICK_BOTTOM = True
PLOT_TICK_LEFT = True
PLOT_LABEL_TOP = False
PLOT_LABEL_RIGHT = False
TOP_SPINE_VISIBLE = True
RIGHT_SPINE_VISIBLE = True
COLORBAR_TICK_DIRECTION = "out"
COLORBAR_TICK_RIGHT = True
COLORBAR_TICK_LEFT = False
COLORBAR_HEIGHT_FRACTION = 0.68
COLORBAR_Y0_FRACTION = 0.16

R_LABEL = r"$R$ [m]"
Z_LABEL = r"$Z$ [m]"
RHO_LABEL = r"$\rho$"
PROFILE_LABEL = "value"
SOURCE_LABEL = "source"
CURRENT_LABEL = "current [MA]"

PANEL_A_TITLE = ""
PANEL_B_TITLE = ""
PANEL_C_TITLE = ""
PANEL_D_TITLE = ""
PANEL_E_TITLE = ""
PANEL_F_TITLE = ""
PANEL_G_TITLE = ""

SURFACE_RAY_COLOR = "#9aa0a6"
SURFACE_RAY_LINE_WIDTH = 0.8
SURFACE_RAY_ALPHA = 0.55
SURFACE_CURVE_LINE_WIDTH = LINE_WIDTH
SURFACE_COLORBAR_SIZE = "6.5%"
SURFACE_COLORBAR_PAD = 0.1
SURFACE_COLORBAR_NBINS = 2
SURFACE_CMAP_NAME = "inferno"
SURFACE_CMAP_MIN = 0.15
SURFACE_CMAP_MAX = 0.92
SURFACE_AXIS_MARKER_COLOR = plt.cm.inferno(SURFACE_CMAP_MIN)
SURFACE_AXIS_MARKER_SIZE = 6
SURFACE_AXIS_MARKER_LINE_WIDTH = 1.2

PSI_LEVEL_COUNT = 128
PSI_COLORBAR_SIZE = "5%"
PSI_COLORBAR_PAD = 0.1
PSI_COLORBAR_NBINS = 5

SOURCE_FF_COLOR = BLUE
SOURCE_PRESSURE_COLOR = ORANGE
SOURCE_FF_STYLE = "-"
SOURCE_PRESSURE_STYLE = "--"
SOURCE_LINE_WIDTH = LINE_WIDTH
SOURCE_TOP_HEADROOM = 0.3
SOURCE_LEGEND_LOC = "upper left"

CURRENT_IP_COLOR = BLACK
CURRENT_IP_STYLE = "--"
CURRENT_IP_LINE_WIDTH = LINE_WIDTH
CURRENT_IP_XMIN = 0.75
CURRENT_IP_XMAX = 1.0
CURRENT_ITOR_COLOR = BLUE
CURRENT_ITOR_STYLE = "-"
CURRENT_JTOR_COLOR = ORANGE
CURRENT_JTOR_STYLE = "--"
CURRENT_JPARA_COLOR = GREEN
CURRENT_JPARA_STYLE = (0, (5, 1, 1, 1, 1, 1))
CURRENT_LINE_WIDTH = LINE_WIDTH
CURRENT_TOP_HEADROOM = 0.35
CURRENT_LEGEND_LOC = "upper left"
CURRENT_LEGEND_NCOLS = 2

SAFETY_Q_COLOR = RED
SAFETY_Q_STYLE = "-"
SAFETY_S_COLOR = PURPLE
SAFETY_S_STYLE = "--"
SAFETY_LINE_WIDTH = LINE_WIDTH
SAFETY_TOP_HEADROOM = 0.2
SAFETY_LEGEND_LOC = "upper left"

SHAPE_PROFILE_PLOT_META = {
    "h": {"color": "#1f77b4", "label": r"$h$", "linestyle": "-", "marker": None},
    "v": {"color": "#ff7f0e", "label": r"$v$", "linestyle": "-", "marker": None},
    "k": {"color": "#2ca02c", "label": r"$\kappa$", "linestyle": "-", "marker": None},
}
SHAPE_PROFILE_NAMES = tuple(SHAPE_PROFILE_PLOT_META)
EXTRA_SHAPE_PROFILE_COLORS = (
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


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
        tree = Tree("[bold blue]Equilibrium[/]")
        tree.add(self.grid)
        tree.add(Text(f"a: {self.a:.3f} [m]"))
        tree.add(Text(f"R0: {self.R0:.3f} [m]"))
        tree.add(Text(f"Z0: {self.Z0:.3f} [m]"))
        tree.add(f"B0: {self.B0:.3f} [T]")
        tree.add(f"Ip: {float(self.Ip):.3e} [A]")
        tree.add(f"beta_t: {float(self.beta_t):.3e}")
        tree.add(f"p0: {self.p0:.6e} [Pa]")
        tree.add(f"alpha1: {self.alpha1:.6f}")
        tree.add(f"alpha2: {self.alpha2:.6f}")
        return tree

    def __str__(self) -> str:
        console = Console(
            color_system=None, force_terminal=False, width=120, record=True, soft_wrap=False
        )
        with console.capture() as capture:
            console.print(self.__rich__())
        return capture.get().rstrip()

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
        eqnb.update_rc(out, self.R0, self.a, self.h)
        return _readonly_owned(out)

    @property
    def epsilon(self) -> np.ndarray:
        """Local inverse aspect ratio ``a*rho/Rc``."""
        out = np.empty_like(self.rho)
        eqnb.update_epsilon(out, self.a, self.rho, self.Rc)
        return _readonly_owned(out)

    @property
    def ftrap(self) -> np.ndarray:
        """Trapped-particle fraction from the full flux-surface magnetic field."""
        out = np.empty_like(self.rho)
        invalid = eqnb.update_ftrap(
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
        eqnb.update_r_coordinates(
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
        eqnb.update_z_coordinates(
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
        surface, radial, invalid = eqnb.materialize_metric_geometry(
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
        return self._R_geometry_fields[eqnb.R]

    @property
    def Z(self) -> np.ndarray:
        """Vertical surface coordinates on ``(rho, theta)``."""
        return self._Z_geometry_fields[eqnb.Z]

    @property
    def Z_t(self) -> np.ndarray:
        """Poloidal derivative of ``Z``."""
        return self._Z_geometry_fields[eqnb.Z_T]

    @property
    def J(self) -> np.ndarray:
        """Surface Jacobian field."""
        return self.surface_fields[eqnb.J]

    @property
    def JdivR(self) -> np.ndarray:
        """Jacobian divided by major radius."""
        return self.surface_fields[eqnb.JDIVR]

    @property
    def gttdivJR(self) -> np.ndarray:
        """Metric coefficient ``g_tt / (J*R)``."""
        return self.surface_fields[eqnb.GTTDIVJR]

    @property
    def gttdivJR_r(self) -> np.ndarray:
        """Radial derivative of ``g_tt / (J*R)``."""
        return self.surface_fields[eqnb.GTTDIVJR_R]

    @property
    def grtdivJR_t(self) -> np.ndarray:
        """Poloidal derivative of the mixed metric coefficient ``g_rt/(J*R)``."""
        return self.surface_fields[eqnb.GRTDIVJR_T]

    @property
    def S(self) -> np.ndarray:
        """Flux-surface area S = -int R*Z_t dtheta."""
        out = np.empty_like(self.rho)
        eqnb.update_surface_area(out, self.R, self.Z_t)
        return _readonly_owned(out)

    @property
    def S_r(self) -> np.ndarray:
        """Flux-surface area derivative S_r = int J dtheta."""
        return self.radial_fields[eqnb.S_R]

    @property
    def V(self) -> np.ndarray:
        """Flux-surface volume V = -pi*int R**2*Z_t dtheta."""
        out = np.empty_like(self.rho)
        eqnb.update_volume(out, self.R, self.Z_t)
        return _readonly_owned(out)

    @property
    def V_r(self) -> np.ndarray:
        """Flux-surface volume derivative V_r = 2pi * int J*R dtheta."""
        return self.radial_fields[eqnb.V_R]

    @property
    def Kn(self) -> np.ndarray:
        """Normalized geometry factor Kn = int gttdivJR dtheta/(2pi)."""
        return self.radial_fields[eqnb.KN]

    @property
    def Kn_r(self) -> np.ndarray:
        """Radial derivative of Kn."""
        return self.radial_fields[eqnb.KN_R]

    @property
    def Ln_r(self) -> np.ndarray:
        """Normalized geometry factor Ln_r = int JdivR dtheta/(2pi)."""
        return self.radial_fields[eqnb.LN_R]

    @property
    def FF_r(self) -> np.ndarray:
        """Physical F*F' profile, model-side diagnostic."""
        out = np.empty_like(self.rho)
        eqnb.update_scaled_copy(out, self.FFn_r, self.alpha1 * self.alpha2)
        return _readonly_owned(out)

    @property
    def FFn_r(self) -> np.ndarray:
        """Radial derivative of the normalized ``F*F'`` source profile."""
        out = np.empty_like(self.rho)
        eqnb.update_scaled_product(
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
        eqnb.update_f2(
            out,
            self.FF_r,
            self.grid.accumulator,
            (self.R0 * self.B0) ** 2,
        )
        return _readonly_owned(out)

    @property
    def F(self) -> np.ndarray:
        """Signed poloidal current function ``F = R * B_phi``."""
        out = np.empty_like(self.rho)
        invalid = eqnb.update_f(out, self.F2, self.R0 * self.B0)
        if invalid:
            raise ValueError("Negative F2 encountered, cannot compute F")
        return _readonly_owned(out)

    @property
    def P_r(self) -> np.ndarray:
        """Physical pressure gradient P', model-side diagnostic."""
        out = np.empty_like(self.rho)
        eqnb.update_scaled_copy(out, self.Pn_r, self.alpha1 * self.alpha2 / MU0)
        return _readonly_owned(out)

    @property
    def Pn_r(self) -> np.ndarray:
        """Radial derivative of the normalized pressure source profile."""
        out = np.empty_like(self.rho)
        eqnb.update_scaled_product(
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
        eqnb.update_pressure(
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
        return float(eqnb.update_beta_t(self.P, self.V_r, self.grid.weights, self.B0))

    @property
    def Gn1(self) -> np.ndarray:
        """Normalized source term before alpha1 in the GS operator."""
        out = np.empty((self.grid.Nr, self.grid.Nt), dtype=np.float64)
        eqnb.update_gn1(
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
        eqnb.update_gn2(
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
        eqnb.update_linear_combination_2d(
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
        return float(eqnb.update_ip(self.Gn1, self.grid.weights, self.alpha1))

    @property
    def q(self) -> np.ndarray:
        """Safety factor q, model-side diagnostic."""
        out = np.empty_like(self.rho)
        invalid = eqnb.update_q(
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
        eqnb.update_shear(
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
        eqnb.update_itor(
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
        invalid = eqnb.update_jtor(
            out,
            _validated_radial_root(self.FFn_psin, self.grid, "FFn_psin"),
            _validated_radial_root(self.Pn_psin, self.grid, "Pn_psin"),
            self.Ln_r,
            self.S_r,
            self.V_r,
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
        invalid = eqnb.update_jpara(
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
        invalid = eqnb.update_jtotal(
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
        eqnb.update_jphi(
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
        eqnb.update_scaled_copy(
            out,
            _validated_radial_root(self.psin, self.grid, "psin"),
            2.0 * np.pi * self.alpha2,
        )
        return _readonly_owned(out)

    @property
    def Phi_r(self) -> np.ndarray:
        """Derivative of toroidal flux ``Phi`` with respect to VEQ ``rho``."""
        out = np.empty_like(self.rho)
        eqnb.update_scaled_product(out, self.F, self.Ln_r, 2.0 * np.pi)
        return _readonly_owned(out)

    @property
    def Phi(self) -> np.ndarray:
        """Toroidal flux Phi."""
        out = np.empty_like(self.rho)
        eqnb.update_phi(
            out,
            self.F,
            self.Ln_r,
            self.grid.accumulator,
        )
        return _readonly_owned(out)

    @property
    def _toroidal_flux_coordinates(self) -> np.ndarray:
        out = np.empty((4, self.grid.Nr), dtype=np.float64)
        invalid = eqnb.update_toroidal_flux_coordinates(
            out,
            self.Phi,
            self.F,
            self.Ln_r,
            self.B0,
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
        return self._toroidal_flux_coordinates[eqnb.RHO_TOR]

    @property
    def rho_tor_norm(self) -> np.ndarray:
        """IMAS toroidal-flux coordinate normalized from axis to boundary."""
        return self._toroidal_flux_coordinates[eqnb.RHO_TOR_NORM]

    @property
    def rho_tor_r(self) -> np.ndarray:
        """Derivative of physical ``rho_tor`` with respect to VEQ ``rho`` [m]."""
        return self._toroidal_flux_coordinates[eqnb.RHO_TOR_R]

    @property
    def rho_tor_norm_r(self) -> np.ndarray:
        """Derivative of ``rho_tor_norm`` with respect to VEQ ``rho``."""
        return self._toroidal_flux_coordinates[eqnb.RHO_TOR_NORM_R]

    @property
    def _gm_fields(self) -> np.ndarray:
        out = np.empty((10, self.grid.Nr), dtype=np.float64)
        invalid = eqnb.update_gm(
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
        return self._gm_fields[eqnb.GM1]

    @property
    def gm2(self) -> np.ndarray:
        """IMAS ``gm2 = <|grad rho_tor|^2/R^2>`` [m^-2]."""
        return self._gm_fields[eqnb.GM2]

    @property
    def gm3(self) -> np.ndarray:
        """IMAS ``gm3 = <|grad rho_tor|^2>``."""
        return self._gm_fields[eqnb.GM3]

    @property
    def gm4(self) -> np.ndarray:
        """IMAS ``gm4 = <1/B^2>`` [T^-2]."""
        return self._gm_fields[eqnb.GM4]

    @property
    def gm5(self) -> np.ndarray:
        """IMAS ``gm5 = <B^2>`` [T^2]."""
        return self._gm_fields[eqnb.GM5]

    @property
    def gm6(self) -> np.ndarray:
        """IMAS ``gm6 = <|grad rho_tor|^2/B^2>`` [T^-2]."""
        return self._gm_fields[eqnb.GM6]

    @property
    def gm7(self) -> np.ndarray:
        """IMAS ``gm7 = <|grad rho_tor|>``."""
        return self._gm_fields[eqnb.GM7]

    @property
    def gm8(self) -> np.ndarray:
        """IMAS ``gm8 = <R>`` [m]."""
        return self._gm_fields[eqnb.GM8]

    @property
    def gm9(self) -> np.ndarray:
        """IMAS ``gm9 = <1/R>`` [m^-1]."""
        return self._gm_fields[eqnb.GM9]

    @property
    def gm10(self) -> np.ndarray:
        """IMAS.jl/FUSE extension ``gm10 = <R^2>`` [m^2]."""
        return self._gm_fields[eqnb.GM10]

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
        R = self.R
        Z = self.Z
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
            F=_sample_profile_on_uniform_psin(self.psin, self.F, psin_uniform),
            P=_sample_profile_on_uniform_psin(self.psin, self.P, psin_uniform),
            FF_psi=_sample_profile_on_uniform_psin(
                self.psin, self.alpha1 * self.FFn_psin, psin_uniform
            ),
            P_psi=_sample_profile_on_uniform_psin(
                self.psin, self.alpha1 * self.Pn_psin / MU0, psin_uniform
            ),
            q=_sample_profile_on_uniform_psin(self.psin, self.q, psin_uniform),
            psi=_interpolate_psin_to_rectilinear_grid(
                R,
                Z,
                self.psin,
                np.square(np.asarray(self.rho, dtype=np.float64)),
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
        raise ValueError(
            f"shape profile coefficient count {count} exceeds grid basis size {limit}"
        )
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
    eqnb.update_profile_fields(
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


def _shape_profile_plot_meta(name: str) -> dict[str, str | None]:
    meta = SHAPE_PROFILE_PLOT_META.get(name)
    if meta is not None:
        return meta

    if name.startswith("c") and name[1:].isdigit():
        label = rf"$c_{int(name[1:])}$"
        style = {"linestyle": "--", "marker": None}
    elif name.startswith("s") and name[1:].isdigit():
        label = rf"$s_{int(name[1:])}$"
        style = {"linestyle": "-", "marker": "x"}
    else:
        label = name
        style = {"linestyle": "-", "marker": None}
    color = EXTRA_SHAPE_PROFILE_COLORS[
        sum(ord(ch) for ch in name) % len(EXTRA_SHAPE_PROFILE_COLORS)
    ]
    return {"color": color, "label": label, **style}


def _plot_equilibrium(
    equilibrium: Equilibrium,
    outpath: str | Path | None = None,
    *,
    show: bool = False,
    plot_residual: bool = False,
    grid: Grid | None = None,
    plot_all: bool = True,
) -> Figure:
    """Render one model-side equilibrium using the paper-style summary layout."""
    surface_equilibrium = _build_resampled_equilibrium(equilibrium, grid=grid)
    fig = _render_equilibrium_summary(
        surface_equilibrium=surface_equilibrium,
        profile_equilibrium=equilibrium,
        plot_residual=plot_residual,
        plot_all=plot_all,
    )

    if outpath is not None:
        fig.savefig(Path(outpath), dpi=300, facecolor="white")
    if show:
        plt.show()
    elif outpath is not None:
        plt.close(fig)

    return fig


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
    psin_out = _resample_profile_linear(source_rho, psin, target_rho)
    psin_r_out = _resample_profile_linear(
        source_rho,
        psin_r,
        target_rho,
        left=0.0,
    )
    FFn_out = _resample_profile_linear(
        source_rho,
        _regularize_axis_linear_profile(FFn_psin, source_rho, copy=True),
        target_rho,
        right=0.0,
    )
    Pn_out = _resample_profile_linear(
        source_rho,
        _regularize_axis_linear_profile(Pn_psin, source_rho, copy=True),
        target_rho,
        right=0.0,
    )
    return (
        psin_out,
        psin_r_out,
        target_grid.differentiate(psin_r_out),
        FFn_out,
        Pn_out,
    )


def _render_equilibrium_summary(
    *,
    surface_equilibrium: Equilibrium,
    profile_equilibrium: Equilibrium | None = None,
    plot_residual: bool = False,
    plot_all: bool = True,
):
    if profile_equilibrium is None:
        profile_equilibrium = surface_equilibrium

    # Surface panels may use a dense resampled grid, while 1D profile panels stay
    # on the original solve grid so diagnostics match the exported snapshot.
    _apply_equilibrium_plot_style()
    if plot_all or plot_residual:
        fig = plt.figure(figsize=(EQUILIBRIUM_PLOT_WIDTH, EQUILIBRIUM_PLOT_HEIGHT))
        left_gs, right_gs = _build_equilibrium_plot_grids(fig, blocks=2)
    else:
        fig = plt.figure(figsize=(EQUILIBRIUM_COMPACT_PLOT_WIDTH, EQUILIBRIUM_COMPACT_PLOT_HEIGHT))
        (left_gs,) = _build_equilibrium_plot_grids(fig, blocks=1)
        right_gs = None

    panel_a = _build_surface_panel_data(surface_equilibrium)
    panel_c = _build_source_panel_data(profile_equilibrium)
    panel_e = _build_current_panel_data(profile_equilibrium)
    panel_f = _build_safety_panel_data(profile_equilibrium)

    _render_panel_a_surfaces(fig.add_subplot(left_gs[:, 0]), fig, panel_a)
    ax_c = fig.add_subplot(left_gs[0, 1])
    ax_e = fig.add_subplot(left_gs[1, 1], sharex=ax_c)
    ax_f = fig.add_subplot(left_gs[2, 1], sharex=ax_c)
    _render_panel_c_sources(ax_c, panel_c)
    _render_panel_e_current_1d(ax_e, panel_e)
    _render_panel_f_safety(ax_f, panel_f)
    for ax in (ax_c, ax_e):
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    if plot_all and right_gs is not None:
        panel_b = _build_shape_panel_data(profile_equilibrium)
        panel_d = _build_jphi_panel_data(surface_equilibrium)
        _render_panel_d_jphi(fig.add_subplot(right_gs[:, 0]), fig, panel_d, panel_a["boundary"])
        shape_spec = right_gs[:2, 1] if plot_residual else right_gs[:, 1]
        shape_gs = shape_spec.subgridspec(3, 1, hspace=GRID_SPEC_HSPACE)
        _render_panel_b_shape_families(
            [fig.add_subplot(shape_gs[row, 0]) for row in range(3)],
            panel_b,
        )
        if plot_residual:
            panel_g = _build_gs_residual_panel_data(surface_equilibrium)
            _render_panel_g_gs_residual(
                fig.add_subplot(right_gs[2, 1]),
                fig,
                panel_g,
                panel_a["boundary"],
            )
    else:
        if plot_residual and right_gs is not None:
            panel_g = _build_gs_residual_panel_data(surface_equilibrium)
            _render_panel_g_gs_residual(
                fig.add_subplot(right_gs[:, 0]),
                fig,
                panel_g,
                panel_a["boundary"],
            )
    return fig


def _apply_equilibrium_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": PLOT_FONT_FAMILY,
            "font.size": PLOT_BASE_FONT_SIZE,
            "mathtext.fontset": PLOT_MATH_FONTSET,
            "axes.unicode_minus": False,
        }
    )


def _build_equilibrium_plot_grids(fig: Figure, *, blocks: int) -> tuple[GridSpec, ...]:
    figure_width, _ = fig.get_size_inches()
    if blocks == 1:
        block_offsets = [0.0]
    elif blocks == 2:
        block_gap = float(figure_width) - 2.0 * SINGLE_COLUMN_WIDTH
        block_offsets = [0.0, SINGLE_COLUMN_WIDTH + block_gap]
    else:
        raise ValueError(f"Expected one or two plot blocks, got {blocks}")

    grids: list[GridSpec] = []
    for offset in block_offsets:
        grids.append(
            GridSpec(
                GRID_SPEC_NROWS,
                GRID_SPEC_NCOLS,
                figure=fig,
                width_ratios=GRID_SPEC_WIDTH_RATIOS,
                height_ratios=GRID_SPEC_HEIGHT_RATIOS,
                hspace=GRID_SPEC_HSPACE,
                wspace=GRID_SPEC_WSPACE,
                top=GRID_SPEC_TOP,
                bottom=GRID_SPEC_BOTTOM,
                left=(offset + GRID_SPEC_LEFT * SINGLE_COLUMN_WIDTH) / figure_width,
                right=(offset + GRID_SPEC_RIGHT * SINGLE_COLUMN_WIDTH) / figure_width,
            )
        )
    return tuple(grids)


def _build_surface_panel_data(equilibrium: Equilibrium) -> dict:
    R = equilibrium.R
    Z = equilibrium.Z
    rho = equilibrium.rho
    Nt = equilibrium.grid.Nt

    sample_rho = np.linspace(0.0, 1.0, 12)
    surfaces = []
    for rho_value in sample_rho:
        idx = int(np.argmin(np.abs(rho - rho_value)))
        if rho[idx] <= 0.0:
            continue
        # Close only at plotting data level; R/Z storage stays Nt-periodic
        # without duplicating theta=0 in the model arrays.
        surfaces.append(
            {
                "rho": float(rho[idx]),
                "R": _close_periodic_curve(R[idx, :]),
                "Z": _close_periodic_curve(Z[idx, :]),
            }
        )

    theta_count = min(max(Nt, 1), 16)
    theta_indices = np.unique(np.linspace(0, Nt - 1, theta_count, dtype=int))
    rays = []
    for theta_idx in theta_indices:
        rays.append(
            {
                "theta_index": int(theta_idx),
                "R": np.asarray(R[:, theta_idx], dtype=np.float64),
                "Z": np.asarray(Z[:, theta_idx], dtype=np.float64),
            }
        )

    return {
        "surfaces": surfaces,
        "rays": rays,
        "axis": {"R": float(R[0, 0]), "Z": float(Z[0, 0])},
        "center": {"R": float(equilibrium.R0), "Z": float(equilibrium.Z0)},
        "boundary": {"R": _close_periodic_curve(R[-1, :]), "Z": _close_periodic_curve(Z[-1, :])},
    }


def _close_periodic_curve(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D periodic curve, got {arr.shape}")
    return np.concatenate([arr, arr[:1]])


def _build_shape_panel_data(equilibrium: Equilibrium) -> dict:
    values = {
        key: _evaluate_profile_fields(profile, equilibrium.grid)[0]
        for key, profile in equilibrium.shape_profiles.items()
        if _include_shape_panel_profile(key)
    }
    return {"shape": {"rho": equilibrium.rho, "values": values}}


def _include_shape_panel_profile(name: str) -> bool:
    if name in SHAPE_PROFILE_PLOT_META:
        return True
    if name.startswith("c") and name[1:].isdigit():
        return int(name[1:]) <= 2
    if name.startswith("s") and name[1:].isdigit():
        return int(name[1:]) <= 3
    return False


def _build_source_panel_data(equilibrium: Equilibrium) -> dict:
    # alpha1 converts normalized source derivatives into the physical GEQDSK-like
    # source convention used in the summary panel.
    return {
        "rho": equilibrium.rho,
        "FF_psi": equilibrium.alpha1 * equilibrium.FFn_psin.copy(),
        "mu0_P_psi": equilibrium.alpha1 * equilibrium.Pn_psin.copy(),
    }


def _build_jphi_panel_data(surface_equilibrium: Equilibrium) -> dict:
    R = surface_equilibrium.R
    Z = surface_equilibrium.Z
    return {
        "R": np.hstack([R, R[:, :1]]),
        "Z": np.hstack([Z, Z[:, :1]]),
        "jphi": np.hstack([surface_equilibrium.jphi, surface_equilibrium.jphi[:, :1]]) / 1e6,
    }


def _build_gs_residual_panel_data(surface_equilibrium: Equilibrium) -> dict:
    R = surface_equilibrium.R
    Z = surface_equilibrium.Z
    return {
        "R": np.hstack([R, R[:, :1]]),
        "Z": np.hstack([Z, Z[:, :1]]),
        "G": np.hstack([surface_equilibrium.G, surface_equilibrium.G[:, :1]]),
    }


def _build_current_panel_data(equilibrium: Equilibrium) -> dict:
    return {
        "rho": equilibrium.rho,
        "itor": equilibrium.Itor.copy() / 1e6,
        "jtor": equilibrium.jtor.copy() / 1e6,
        "jpara": equilibrium.jpara.copy() / 1e6,
        "Ip": float(equilibrium.Ip) / 1e6,
    }


def _build_safety_panel_data(equilibrium: Equilibrium) -> dict:
    return {"rho": equilibrium.rho, "q": equilibrium.q.copy(), "s": equilibrium.s.copy()}


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


def _evaluate_profile_fields(profile: Profile, grid: Grid) -> np.ndarray:
    return profile.with_grid(grid).fields


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


def _apply_rz_limits(ax: plt.Axes, boundary_data: dict):
    R_bnd, Z_bnd = boundary_data["R"], boundary_data["Z"]
    R_margin = (R_bnd.max() - R_bnd.min()) * 0.1
    Z_margin = (Z_bnd.max() - Z_bnd.min()) * 0.1
    ax.set_xlim(R_bnd.min() - R_margin, R_bnd.max() + R_margin)
    ax.set_ylim(Z_bnd.min() - Z_margin, Z_bnd.max() + Z_margin)
    ax.set_aspect("equal")


def _get_trunc_inferno() -> mcolors.LinearSegmentedColormap:
    cmap = plt.get_cmap(SURFACE_CMAP_NAME)
    return mcolors.LinearSegmentedColormap.from_list(
        "trunc_inferno", cmap(np.linspace(SURFACE_CMAP_MIN, SURFACE_CMAP_MAX, 256))
    )


def _get_gs_residual_cmap() -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "gs_residual",
        [
            (0.0, "#2166ac"),
            (0.5, "#f7f7f7"),
            (1.0, "#b2182b"),
        ],
    )


def _add_top_headroom(ax: plt.Axes, ratio: float) -> None:
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    if ratio > 0.0:
        ax.set_ylim(y0, y1 + ratio * span)
    else:
        ax.set_ylim(y0 + ratio * span, y1)


def _style_axis(
    ax: plt.Axes,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    grid: bool = True,
) -> None:
    ax.set_title(title, fontsize=SUBPLOT_TITLE_FONTSIZE, fontweight="normal")
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE)
    if grid:
        ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINE_WIDTH, linestyle=GRID_LINESTYLE)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(
        direction=PLOT_TICK_DIRECTION,
        top=PLOT_TICK_TOP,
        right=PLOT_TICK_RIGHT,
        bottom=PLOT_TICK_BOTTOM,
        left=PLOT_TICK_LEFT,
        labeltop=PLOT_LABEL_TOP,
        labelright=PLOT_LABEL_RIGHT,
        labelsize=TICK_LABEL_FONT_SIZE,
    )
    ax.spines["top"].set_visible(TOP_SPINE_VISIBLE)
    ax.spines["right"].set_visible(RIGHT_SPINE_VISIBLE)


def _style_legend(ax: plt.Axes, *, loc: str = "upper left", ncols: int = 1) -> None:
    ax.legend(
        frameon=False,
        loc=loc,
        ncols=ncols,
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=LEGEND_COLUMN_SPACING,
        labelspacing=LEGEND_LABEL_SPACING,
    )


def _style_colorbar(cbar, *, label: str) -> None:
    cbar.ax.set_title(label, fontsize=AXIS_LABEL_FONT_SIZE, pad=6.0)
    cbar.ax.tick_params(
        which="both",
        direction=COLORBAR_TICK_DIRECTION,
        right=COLORBAR_TICK_RIGHT,
        left=COLORBAR_TICK_LEFT,
        labelsize=TICK_LABEL_FONT_SIZE,
    )


def _render_panel_a_surfaces(ax: plt.Axes, fig: plt.Figure, data: dict):
    _style_axis(ax, xlabel=R_LABEL, ylabel=Z_LABEL, title=PANEL_A_TITLE, grid=False)
    for ray in data.get("rays", []):
        ax.plot(
            ray["R"],
            ray["Z"],
            color=SURFACE_RAY_COLOR,
            linewidth=SURFACE_RAY_LINE_WIDTH,
            alpha=SURFACE_RAY_ALPHA,
            zorder=1,
        )

    surfaces = data["surfaces"]
    colors = plt.cm.inferno(np.linspace(0.0, 1.0, max(len(surfaces), 1)) * 0.77 + 0.15)
    for ci, surf in enumerate(surfaces):
        ax.plot(
            surf["R"],
            surf["Z"],
            color=colors[min(ci, len(colors) - 1)],
            linewidth=SURFACE_CURVE_LINE_WIDTH,
            zorder=2,
        )

    axis = data["axis"]
    ax.scatter(
        [axis["R"]],
        [axis["Z"]],
        color=SURFACE_AXIS_MARKER_COLOR,
        marker="o",
        s=SURFACE_AXIS_MARKER_SIZE,
        linewidths=SURFACE_AXIS_MARKER_LINE_WIDTH,
        zorder=3,
    )
    _apply_rz_limits(ax, data["boundary"])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=SURFACE_COLORBAR_SIZE, pad=SURFACE_COLORBAR_PAD)
    cax.set_axis_off()
    cbar_ax = cax.inset_axes([0.0, COLORBAR_Y0_FRACTION, 1.0, COLORBAR_HEIGHT_FRACTION])
    sm = plt.cm.ScalarMappable(
        cmap=_get_trunc_inferno(), norm=mcolors.Normalize(vmin=0.0, vmax=1.0)
    )
    cbar = fig.colorbar(sm, cax=cbar_ax)
    _style_colorbar(cbar, label=RHO_LABEL)
    cbar.locator = ticker.MaxNLocator(nbins=SURFACE_COLORBAR_NBINS)
    cbar.update_ticks()


def _render_panel_b_shape_families(axes: list[plt.Axes], data: dict) -> None:
    if len(axes) != 3:
        raise ValueError(f"Expected three shape family axes, got {len(axes)}")
    shape = data["shape"]
    values = shape["values"]
    groups = [
        [key for key in SHAPE_PROFILE_NAMES if key in values],
        sorted(
            [key for key in values if key.startswith("c") and key[1:].isdigit()],
            key=lambda key: int(key[1:]),
        ),
        sorted(
            [key for key in values if key.startswith("s") and key[1:].isdigit()],
            key=lambda key: int(key[1:]),
        ),
    ]
    for index, (ax, keys) in enumerate(zip(axes, groups, strict=True)):
        _style_axis(
            ax,
            xlabel=RHO_LABEL,
            ylabel=PROFILE_LABEL if index == 0 else "",
            title=PANEL_B_TITLE,
        )
        _plot_shape_profile_group(ax, shape, keys, linestyle="-")
        if keys:
            _style_legend(ax, loc="best")
        if index < 2:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)


def _plot_shape_profile_group(
    ax: plt.Axes,
    shape: dict,
    keys: list[str],
    *,
    linestyle: str | tuple | None = None,
) -> None:
    for key in keys:
        vals = shape["values"][key]
        meta = _shape_profile_plot_meta(key)
        ax.plot(
            shape["rho"],
            vals,
            linestyle=meta["linestyle"] if linestyle is None else linestyle,
            marker=meta["marker"],
            color=meta["color"],
            linewidth=LINE_WIDTH,
            label=meta["label"],
        )


def _render_panel_c_sources(ax: plt.Axes, data: dict):
    _style_axis(ax, xlabel=RHO_LABEL, ylabel=SOURCE_LABEL, title=PANEL_C_TITLE)
    rho = data["rho"]
    ax.plot(
        rho,
        data["FF_psi"],
        SOURCE_FF_STYLE,
        color=SOURCE_FF_COLOR,
        linewidth=SOURCE_LINE_WIDTH,
        label=r"$FF_\psi$",
    )
    ax.plot(
        rho,
        data["mu0_P_psi"],
        SOURCE_PRESSURE_STYLE,
        color=SOURCE_PRESSURE_COLOR,
        linewidth=SOURCE_LINE_WIDTH,
        label=r"$\mu_0 P_\psi$",
    )

    _add_top_headroom(ax, ratio=SOURCE_TOP_HEADROOM)
    _style_legend(ax, loc=SOURCE_LEGEND_LOC)


def _render_panel_d_jphi(ax: plt.Axes, fig: plt.Figure, data: dict, boundary: dict):
    _style_axis(ax, xlabel=R_LABEL, ylabel=Z_LABEL, title=PANEL_D_TITLE, grid=False)
    R_plot, Z_plot, j_plot = data["R"], data["Z"], data["jphi"]
    cmap = _get_trunc_inferno()
    vmin = min(float(np.nanmin(j_plot)), 0.0)
    vmax = float(np.nanmax(j_plot))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    ax.set_facecolor(cmap(norm(0.0)))
    pcm = ax.contourf(
        R_plot, Z_plot, j_plot, levels=np.linspace(vmin, vmax, 128), cmap=cmap, norm=norm
    )
    _apply_rz_limits(ax, boundary)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=PSI_COLORBAR_SIZE, pad=PSI_COLORBAR_PAD)
    cax.set_axis_off()
    cbar_ax = cax.inset_axes([0.0, COLORBAR_Y0_FRACTION, 1.0, COLORBAR_HEIGHT_FRACTION])
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    _style_colorbar(cbar, label=r"$j_\phi$")
    cbar.locator = ticker.MaxNLocator(nbins=PSI_COLORBAR_NBINS)
    cbar.update_ticks()


def _render_panel_g_gs_residual(ax: plt.Axes, fig: plt.Figure, data: dict, boundary: dict):
    _style_axis(ax, xlabel=R_LABEL, ylabel=Z_LABEL, title=PANEL_G_TITLE, grid=False)
    R_plot, Z_plot, G_plot = data["R"], data["Z"], data["G"]
    finite_abs = np.abs(G_plot[np.isfinite(G_plot)])
    vmax = float(np.quantile(finite_abs, 0.99)) if finite_abs.size else 0.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    vmin = -vmax
    cmap = _get_gs_residual_cmap()
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    ax.set_facecolor(cmap(norm(0.0)))
    pcm = ax.contourf(
        R_plot, Z_plot, G_plot, levels=np.linspace(vmin, vmax, 129), cmap=cmap, norm=norm
    )
    _apply_rz_limits(ax, boundary)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=PSI_COLORBAR_SIZE, pad=PSI_COLORBAR_PAD)
    cax.set_axis_off()
    cbar_ax = cax.inset_axes([0.0, COLORBAR_Y0_FRACTION, 1.0, COLORBAR_HEIGHT_FRACTION])
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    _style_colorbar(cbar, label=r"$G$")
    cbar.locator = ticker.MaxNLocator(nbins=PSI_COLORBAR_NBINS)
    cbar.update_ticks()


def _render_panel_e_current_1d(ax: plt.Axes, data: dict):
    _style_axis(ax, xlabel=RHO_LABEL, ylabel=CURRENT_LABEL, title=PANEL_E_TITLE)
    rho = data["rho"]
    ax.axhline(
        data["Ip"],
        xmin=CURRENT_IP_XMIN,
        xmax=CURRENT_IP_XMAX,
        color=CURRENT_IP_COLOR,
        linestyle=CURRENT_IP_STYLE,
        linewidth=CURRENT_IP_LINE_WIDTH,
        label=r"$I_p$",
    )
    ax.plot(
        rho,
        data["itor"],
        CURRENT_ITOR_STYLE,
        color=CURRENT_ITOR_COLOR,
        linewidth=CURRENT_LINE_WIDTH,
        label=r"$I_{\mathrm{tor}}$",
    )
    ax.plot(
        rho,
        data["jtor"],
        CURRENT_JTOR_STYLE,
        color=CURRENT_JTOR_COLOR,
        linewidth=CURRENT_LINE_WIDTH,
        label=r"$j_{\mathrm{tor}}$",
    )
    ax.plot(
        rho,
        data["jpara"],
        color=CURRENT_JPARA_COLOR,
        linestyle=CURRENT_JPARA_STYLE,
        linewidth=CURRENT_LINE_WIDTH,
        label=r"$j_{\parallel}$",
    )

    _add_top_headroom(ax, ratio=CURRENT_TOP_HEADROOM)
    _style_legend(ax, loc=CURRENT_LEGEND_LOC, ncols=CURRENT_LEGEND_NCOLS)


def _render_panel_f_safety(ax: plt.Axes, data: dict):
    _style_axis(ax, xlabel=RHO_LABEL, ylabel=PROFILE_LABEL, title=PANEL_F_TITLE)
    rho = data["rho"]
    ax.plot(
        rho,
        data["q"],
        SAFETY_Q_STYLE,
        color=SAFETY_Q_COLOR,
        linewidth=SAFETY_LINE_WIDTH,
        label=r"$q$",
    )
    ax.plot(
        rho,
        data["s"],
        SAFETY_S_STYLE,
        color=SAFETY_S_COLOR,
        linewidth=SAFETY_LINE_WIDTH,
        label=r"$s$",
    )
    _add_top_headroom(ax, ratio=SAFETY_TOP_HEADROOM)
    _style_legend(ax, loc=SAFETY_LEGEND_LOC)
