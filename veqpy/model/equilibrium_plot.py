"""Visualization helpers for veqpy.model.Equilibrium.

Matplotlib and Rich stay behind Equilibrium's lazy display methods so importing
the physical model does not load visualization dependencies.
"""

from __future__ import annotations

from pathlib import Path

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

from veqpy.model.grid import Grid
from veqpy.model.profile import Profile

from .equilibrium import Equilibrium, _build_resampled_equilibrium

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


def _build_equilibrium_rich_tree(equilibrium: Equilibrium) -> Tree:
    tree = Tree("[bold blue]Equilibrium[/]")
    tree.add(equilibrium.grid)
    tree.add(Text(f"a: {equilibrium.a:.3f} [m]"))
    tree.add(Text(f"R0: {equilibrium.R0:.3f} [m]"))
    tree.add(Text(f"Z0: {equilibrium.Z0:.3f} [m]"))
    tree.add(f"B0: {equilibrium.B0:.3f} [T]")
    tree.add(f"Ip: {float(equilibrium.Ip):.3e} [A]")
    tree.add(f"beta_t: {float(equilibrium.beta_t):.3e}")
    tree.add(f"p0: {equilibrium.p0:.6e} [Pa]")
    tree.add(f"alpha1: {equilibrium.alpha1:.6f}")
    tree.add(f"alpha2: {equilibrium.alpha2:.6f}")
    return tree


def _equilibrium_to_string(equilibrium: Equilibrium) -> str:
    console = Console(
        color_system=None, force_terminal=False, width=120, record=True, soft_wrap=False
    )
    with console.capture() as capture:
        console.print(_build_equilibrium_rich_tree(equilibrium))
    return capture.get().rstrip()


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

def _evaluate_profile_fields(profile: Profile, grid: Grid) -> np.ndarray:
    return profile.with_grid(grid).fields

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
