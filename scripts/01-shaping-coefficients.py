"""Build Figure 01 shaping-coefficient basis panels.

The script visualizes the radial and Fourier basis terms used by VEQPy's
boundary and profile parameterizations.  Outputs are controlled by ``PNG_PATH``
and ``PDF_PATH`` so the same script can be used for manuscript regeneration or
local smoke checks.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
from typing import Dict, List

import matplotlib.pyplot as plt
from config import (
    AXIS_LABEL_FONT_SIZE,
    DOUBLE_COLUMN_WIDTH,
    LEGEND_FONT_SIZE,
    PLOT_FONT_FAMILY,
    PLOT_LABEL_RIGHT,
    PLOT_LABEL_TOP,
    PLOT_TICK_BOTTOM,
    PLOT_TICK_DIRECTION,
    PLOT_TICK_LEFT,
    PLOT_TICK_RIGHT,
    PLOT_TICK_TOP,
    SAVE_DPI,
    SAVE_TRANSPARENT,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    apply_plot_style,
    figure_path,
    save_figure_outputs,
    scaled_font_size,
)
from matplotlib.collections import LineCollection

FIGURE_SIZE = (DOUBLE_COLUMN_WIDTH, 6)
SUBPLOTS_ADJUST = {"bottom": 0.28, "hspace": 0.25, "wspace": 0.25}
PNG_PATH = figure_path("01-shaping-coefficients.png")
PDF_PATH = None
SUBPLOT_ROWS = 2
SUBPLOT_COLS = 4

SUBPLOT_TITLES = [
    r"$\bf{(a)}$ All Parameters",
    r"$\bf{(b)}$ Horizontal Shift $h$",
    r"$\bf{(c)}$ Vertical Shift $v$",
    r"$\bf{(d)}$ Elongation $\kappa$",
    r"$\bf{(e)}$ Tilt $c_0$",
    r"$\bf{(f)}$ Ovality $c_1$",
    r"$\bf{(g)}$ Triangularity $s_1$",
    r"$\bf{(h)}$ Squareness $s_2$",
]
X_LABEL = r"$R$ [m]"
Y_LABEL = r"$Z$ [m]"

GRID_ALPHA = 0.25
GRID_LINESTYLE = "-"
GRID_LINE_WIDTH = 0.5
TOP_SPINE_VISIBLE = True
RIGHT_SPINE_VISIBLE = True
LEGEND_FRAME_ON = False
ALL_PARAMS_LEGEND_LOC = "upper right"
REFERENCE_LINE_WIDTH = 1.0
ALL_PARAMS_LINE_WIDTH = 1.25
SINGLE_PARAM_LINE_WIDTH = 1.25
AXIS_MARKER_SIZE = 30

REFERENCE_LINE_COLOR = "gray"
REFERENCE_LINE_STYLE = "--"
EFFECT_LINE_COLOR = "C0"
AXIS_MARKER_COLOR = "red"
AXIS_MARKER_STYLE = "x"
REFERENCE_LABEL = "Reference"
EFFECT_LABEL = "With effect"
INITIAL_PARAMS = {
    "h": 0.15,
    "v": -0.15,
    "kappa": 1.2,
    "c0": 0.3,
    "c1": 0.3,
    "s1": 0.3,
    "s2": 0.2,
}


def _style_axis(
    ax: plt.Axes,
    *,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Apply shared plot-axis styling for the surface panels."""
    ax.set_title(title, fontsize=scaled_font_size(TITLE_FONT_SIZE), fontweight="normal")
    ax.set_xlabel(X_LABEL, fontsize=scaled_font_size(AXIS_LABEL_FONT_SIZE))
    ax.set_ylabel(Y_LABEL, fontsize=scaled_font_size(AXIS_LABEL_FONT_SIZE))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINE_WIDTH, linestyle=GRID_LINESTYLE)
    ax.set_axisbelow(True)
    ax.tick_params(
        direction=PLOT_TICK_DIRECTION,
        top=PLOT_TICK_TOP,
        right=PLOT_TICK_RIGHT,
        bottom=PLOT_TICK_BOTTOM,
        left=PLOT_TICK_LEFT,
        labeltop=PLOT_LABEL_TOP,
        labelright=PLOT_LABEL_RIGHT,
        labelsize=scaled_font_size(TICK_LABEL_FONT_SIZE),
    )
    ax.spines["top"].set_visible(TOP_SPINE_VISIBLE)
    ax.spines["right"].set_visible(RIGHT_SPINE_VISIBLE)


def _style_legend(ax: plt.Axes, *, loc: str) -> None:
    """Apply shared legend styling."""
    ax.legend(
        loc=loc,
        fontsize=scaled_font_size(LEGEND_FONT_SIZE),
        frameon=LEGEND_FRAME_ON,
    )


def chebyshev_ploy(order: int, x: np.ndarray) -> np.ndarray:
    """Chebyshev polynomial of the first kind T_l(x)."""
    if order == 0:
        return np.ones_like(x)
    elif order == 1:
        return x.copy()
    else:
        T_prev2 = np.ones_like(x)
        T_prev1 = x.copy()
        for _ in range(2, order + 1):
            T_curr = 2 * x * T_prev1 - T_prev2
            T_prev2 = T_prev1
            T_prev1 = T_curr
        return T_prev1


def basis_u(order: int, rho: np.ndarray) -> np.ndarray:
    """Radial basis function u_l(rho) = (1 - rho^2) * T_l(2*rho^2 - 1)."""
    x = 2 * rho**2 - 1
    return (1 - rho**2) * chebyshev_ploy(order, x)


def expand_profile(
    coeffs: List[float], rho: np.ndarray, has_boundary: bool = False, power: int = 0
) -> np.ndarray:
    """
    Expand a profile using Chebyshev basis.

    For profiles like h, v:
        f(rho) = sum_{l=0}^L f_l * u_l(rho)

    For profiles like kappa, c0 (with boundary constant):
        f(rho) = f_a + sum_{l=0}^L f_l * u_l(rho)

    For profiles like c_m, s_n (with rho^m or rho^n factor):
        f(rho) = rho^power * (f_a + sum_{l=0}^L f_l * u_l(rho))

    Args:
        coeffs: List of coefficients [f_a, f_0, f_1, ...] or [f_0, f_1, ...]
        rho: Radial coordinate array
        has_boundary: If True, first coeff is boundary constant f_a
        power: Power of rho factor (for c_m, s_n)

    Returns:
        Profile values at rho points
    """
    result = np.zeros_like(rho)

    if has_boundary:
        if len(coeffs) == 0:
            return result
        result += coeffs[0]  # boundary constant
        cheb_coeffs = coeffs[1:]
    else:
        cheb_coeffs = coeffs

    for idx, coeff in enumerate(cheb_coeffs):
        result += coeff * basis_u(idx, rho)

    if power > 0:
        result *= rho**power

    return result


class FluxSurface:
    """
    Magnetic flux surface calculator.

    Computes R(rho, theta) and Z(rho, theta) based on spectral expansion.
    """

    def __init__(self, R0: float, Z0: float, a: float, params: Dict[str, List[float]]):
        """
        Initialize flux surface calculator.

        Args:
            R0: Major radius of magnetic axis
            Z0: Vertical position of magnetic axis
            a: Minor radius
            params: Dictionary of shape parameters, e.g.:
                {
                    "h": [h0, h1, h2],           # Horizontal Shift (no boundary)
                    "v": [v0, v1],               # Vertical Shift (no boundary)
                    "kappa": [ka, k0, k1],       # Elongation (with boundary)
                    "c0": [c0a, c00, c01],       # Tilt (with boundary)
                    "c1": [c1a, c10],            # Ovality m=1 (with boundary + rho^1)
                    "c2": [c2a],                 # Ovality m=2 (with boundary + rho^2)
                    "s1": [s1a, s10, s11],       # Triangularity (with boundary + rho^1)
                    "s2": [s2a, s20],            # Squareness (with boundary + rho^2)
                    ...
                }
        """
        self.R0 = R0
        self.Z0 = Z0
        self.a = a
        self.params = params

    def _get_profile(self, name: str, rho: np.ndarray) -> np.ndarray:
        """Get profile values for a given parameter name."""
        if name not in self.params or self.params[name] is None:
            return np.zeros_like(rho)

        coeffs = self.params[name]
        if len(coeffs) == 0:
            return np.zeros_like(rho)

        # h, v: no boundary constant, no rho power
        if name in ["h", "v"]:
            return expand_profile(coeffs, rho, has_boundary=False, power=0)

        # kappa, c0: with boundary constant, no rho power
        if name in ["kappa", "c0"]:
            return expand_profile(coeffs, rho, has_boundary=True, power=0)

        # c_m (m >= 1): with boundary constant, rho^m power
        if name.startswith("c") and len(name) > 1:
            m = int(name[1:])
            return expand_profile(coeffs, rho, has_boundary=True, power=m)

        # s_n (n >= 1): with boundary constant, rho^n power
        if name.startswith("s") and len(name) > 1:
            n = int(name[1:])
            return expand_profile(coeffs, rho, has_boundary=True, power=n)

        return np.zeros_like(rho)

    def compute_theta_bar(self, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute modified poloidal angle theta_bar.

        theta_bar = theta + c0 + sum_m(c_m * cos(m*theta)) + sum_n(s_n * sin(n*theta))
        """
        # rho and theta should be 2D arrays of same shape
        theta_bar = theta.copy()

        # Add c0 (tilt)
        c0 = self._get_profile("c0", rho)
        theta_bar += c0

        # Add c_m terms (ovality)
        for key in self.params:
            if key.startswith("c") and key != "c0" and self.params[key] is not None:
                m = int(key[1:])
                c_m = self._get_profile(key, rho)
                theta_bar += c_m * np.cos(m * theta)

        # Add s_n terms (triangularity, squareness)
        for key in self.params:
            if key.startswith("s") and self.params[key] is not None:
                n = int(key[1:])
                s_n = self._get_profile(key, rho)
                theta_bar += s_n * np.sin(n * theta)

        return theta_bar

    def compute_RZ(self, rho: np.ndarray, theta: np.ndarray) -> tuple:
        """
        Compute R and Z coordinates.

        R = R0 + a * (h + rho * cos(theta_bar))
        Z = Z0 + a * (v - rho * kappa * sin(theta))

        Args:
            rho: 2D array of radial coordinates
            theta: 2D array of poloidal angles

        Returns:
            (R, Z) tuple of 2D arrays
        """
        h = self._get_profile("h", rho)
        v = self._get_profile("v", rho)
        kappa = self._get_profile("kappa", rho)

        # Default kappa to 1 if not specified
        if "kappa" not in self.params or self.params["kappa"] is None:
            kappa = np.ones_like(rho)

        theta_bar = self.compute_theta_bar(rho, theta)

        R = self.R0 + self.a * (h + rho * np.cos(theta_bar))
        Z = self.Z0 + self.a * (v - rho * kappa * np.sin(theta))

        return R, Z

    def compute_on_grid(self, rho_1d: np.ndarray, theta_1d: np.ndarray) -> tuple:
        """
        Compute R, Z on a meshgrid of rho and theta.

        Args:
            rho_1d: 1D array of radial coordinates (0 to 1)
            theta_1d: 1D array of poloidal angles (0 to 2*pi)

        Returns:
            (R, Z) tuple of 2D arrays with shape (len(rho_1d), len(theta_1d))
        """
        rho_2d, theta_2d = np.meshgrid(rho_1d, theta_1d, indexing="ij")
        return self.compute_RZ(rho_2d, theta_2d)


def plot(save: bool = True, font_family: str = PLOT_FONT_FAMILY):
    apply_plot_style(font_family)

    # Base parameters
    R0 = 1.8
    Z0 = 0.0
    a = 0.58

    param_names = ["h", "v", "kappa", "c0", "c1", "s1", "s2"]

    n_theta = 64  # Reduced for smoother interaction
    theta_1d = np.linspace(0, 2 * np.pi, n_theta)
    rho_all = np.linspace(0.125, 1, 6)
    rho_sub = np.linspace(0.25, 1.0, 4)

    # Fixed axis limits
    R_min, R_max = R0 - a - 0.15, R0 + a + 0.15
    Z_min, Z_max = -a * 1.8 - 0.15, a * 1.8 + 0.15

    # Pre-compute reference surfaces
    fs_ref = FluxSurface(R0, Z0, a, {})
    ref_all = {}
    for rho_val in rho_all:
        R, Z = fs_ref.compute_on_grid(np.array([rho_val]), theta_1d)
        ref_all[rho_val] = (np.append(R[0, :], R[0, 0]), np.append(Z[0, :], Z[0, 0]))
    ref_sub = {}
    for rho_val in rho_sub:
        R, Z = fs_ref.compute_on_grid(np.array([rho_val]), theta_1d)
        ref_sub[rho_val] = (np.append(R[0, :], R[0, 0]), np.append(Z[0, :], Z[0, 0]))

    # Create figure
    fig, axes = plt.subplots(SUBPLOT_ROWS, SUBPLOT_COLS, figsize=FIGURE_SIZE)
    plt.subplots_adjust(**SUBPLOTS_ADJUST)
    axes = axes.flatten()

    # Initialize LineCollection objects for fast batch updates
    collections = {}

    def setup_axes():
        """Setup static elements of axes."""
        for idx, ax in enumerate(axes):
            _style_axis(
                ax,
                title=SUBPLOT_TITLES[idx],
                xlim=(R_min, R_max),
                ylim=(Z_min, Z_max),
            )
            ax.scatter(
                [R0],
                [Z0],
                marker=AXIS_MARKER_STYLE,
                color=AXIS_MARKER_COLOR,
                s=AXIS_MARKER_SIZE,
                zorder=10,
            )

            # Use LineCollection for batch updates
            if idx == 0:
                for i, rho_val in enumerate(rho_all):
                    R_surf, Z_surf = ref_all[rho_val]
                    ax.plot(
                        R_surf,
                        Z_surf,
                        color=REFERENCE_LINE_COLOR,
                        linewidth=REFERENCE_LINE_WIDTH,
                        linestyle=REFERENCE_LINE_STYLE,
                        label=REFERENCE_LABEL if i == 0 else None,
                    )
                # 8 curves for all params subplot
                lc = LineCollection(
                    [],
                    colors=EFFECT_LINE_COLOR,
                    linewidths=ALL_PARAMS_LINE_WIDTH,
                    label=EFFECT_LABEL,
                )
                ax.add_collection(lc)
                collections[idx] = lc
                _style_legend(ax, loc=ALL_PARAMS_LEGEND_LOC)
            else:
                # Draw static reference lines
                for i, rho_val in enumerate(rho_sub):
                    R_surf, Z_surf = ref_sub[rho_val]
                    ax.plot(
                        R_surf,
                        Z_surf,
                        color=REFERENCE_LINE_COLOR,
                        linewidth=REFERENCE_LINE_WIDTH,
                        linestyle=REFERENCE_LINE_STYLE,
                        label=REFERENCE_LABEL if i == 0 else None,
                    )
                # 4 curves for single param subplot
                lc = LineCollection(
                    [],
                    colors=EFFECT_LINE_COLOR,
                    linewidths=SINGLE_PARAM_LINE_WIDTH,
                )
                ax.add_collection(lc)
                collections[idx] = lc

    # Pre-compute meshgrids and trigonometric values for fast updates
    rho_2d_all, theta_2d_all = np.meshgrid(rho_all, theta_1d, indexing="ij")
    rho_2d_sub, theta_2d_sub = np.meshgrid(rho_sub, theta_1d, indexing="ij")
    cos_theta_all = np.cos(theta_2d_all)
    sin_theta_all = np.sin(theta_2d_all)
    sin_2theta_all = np.sin(2 * theta_2d_all)
    cos_theta_sub = np.cos(theta_2d_sub)
    sin_theta_sub = np.sin(theta_2d_sub)
    sin_2theta_sub = np.sin(2 * theta_2d_sub)

    # Pre-compute reference circle (no parameters)
    R_ref_sub = R0 + a * rho_2d_sub * cos_theta_sub
    Z_ref_sub = Z0 - a * rho_2d_sub * sin_theta_sub

    def update_all_params(h, v, kappa, c0, c1, s1, s2):
        """Update first subplot with all parameters."""
        theta_bar = (
            theta_2d_all + c0 + c1 * cos_theta_all + s1 * sin_theta_all + s2 * sin_2theta_all
        )
        R = R0 + a * (h + rho_2d_all * np.cos(theta_bar))
        Z = Z0 + a * (v - rho_2d_all * kappa * sin_theta_all)
        # Build segments for LineCollection: list of (N, 2) arrays
        segments = []
        for i in range(len(rho_all)):
            pts = np.column_stack([np.append(R[i, :], R[i, 0]), np.append(Z[i, :], Z[i, 0])])
            segments.append(pts)
        collections[0].set_segments(segments)

    def update_single_param(idx, param_name, value):
        """Update single parameter subplot with simplified formulas."""
        if param_name == "h":
            # R = R0 + a*(h + rho*cos(theta)), Z = Z0 - a*rho*sin(theta)
            R = R_ref_sub + a * value
            Z = Z_ref_sub
        elif param_name == "v":
            # R = R0 + a*rho*cos(theta), Z = Z0 + a*(v - rho*sin(theta))
            R = R_ref_sub
            Z = Z_ref_sub + a * value
        elif param_name == "kappa":
            # R = R0 + a*rho*cos(theta), Z = Z0 - a*rho*kappa*sin(theta)
            R = R_ref_sub
            Z = Z0 - a * rho_2d_sub * value * sin_theta_sub
        elif param_name == "c0":
            # theta_bar = theta + c0
            theta_bar = theta_2d_sub + value
            R = R0 + a * rho_2d_sub * np.cos(theta_bar)
            Z = Z_ref_sub
        elif param_name == "c1":
            # theta_bar = theta + c1*cos(theta)
            theta_bar = theta_2d_sub + value * cos_theta_sub
            R = R0 + a * rho_2d_sub * np.cos(theta_bar)
            Z = Z_ref_sub
        elif param_name == "s1":
            # theta_bar = theta + s1*sin(theta)
            theta_bar = theta_2d_sub + value * sin_theta_sub
            R = R0 + a * rho_2d_sub * np.cos(theta_bar)
            Z = Z_ref_sub
        elif param_name == "s2":
            # theta_bar = theta + s2*sin(2*theta)
            theta_bar = theta_2d_sub + value * sin_2theta_sub
            R = R0 + a * rho_2d_sub * np.cos(theta_bar)
            Z = Z_ref_sub
        else:
            return
        # Build segments for LineCollection
        segments = []
        for i in range(len(rho_sub)):
            pts = np.column_stack([np.append(R[i, :], R[i, 0]), np.append(Z[i, :], Z[i, 0])])
            segments.append(pts)
        collections[idx].set_segments(segments)

    def update_lines(params):
        """Update all subplots."""
        h, v, kappa = params["h"][0], params["v"][0], params["kappa"][0]
        c0, c1, s1, s2 = (
            params["c0"][0],
            params["c1"][0],
            params["s1"][0],
            params["s2"][0],
        )
        update_all_params(h, v, kappa, c0, c1, s1, s2)
        for idx, name in enumerate(param_names, 1):
            update_single_param(idx, name, params[name][0])

    setup_axes()

    # Static plot
    params = {k: [v] for k, v in INITIAL_PARAMS.items()}
    update_lines(params)
    plt.tight_layout()
    if save:
        save_figure_outputs(
            fig,
            png_path=PNG_PATH,
            pdf_path=PDF_PATH,
            dpi=SAVE_DPI,
            transparent=SAVE_TRANSPARENT,
        )
    return fig


if __name__ == "__main__":
    plot()
