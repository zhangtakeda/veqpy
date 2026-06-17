"""Downstream heat-transport geometry check.

This script is a deliberately small downstream-operator test.  It compares
the Pareto-selected Low/Medium/High reduced VEQ equilibria against the
direct GEQDSK-file target curve.  The transport problem is not intended as
predictive modelling; it only propagates geometry differences through a fixed
1-D steady heat-diffusion operator.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from config import (
    AXIS_LABEL_FONT_SIZE,
    CASE_KEYS,
    CASE_LABELS,
    CASE_LINE_COLORS,
    CASE_REFERENCE_GFILES,
    DOUBLE_COLUMN_WIDTH,
    LEGEND_FONT_SIZE,
    LEVEL_LINESTYLES,
    MU0,
    PLOT_LABEL_RIGHT,
    PLOT_LABEL_TOP,
    PLOT_TICK_BOTTOM,
    PLOT_TICK_DIRECTION,
    PLOT_TICK_LEFT,
    PLOT_TICK_RIGHT,
    PLOT_TICK_TOP,
    REDUCED_CONFIG_LABELS,
    REFERENCE_LABELS,
    SAVE_DPI,
    SAVE_TRANSPARENT,
    SCIENTIFIC_DECIMALS,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    apply_plot_style,
    figure_path,
    load_equilibrium_json,
    load_reduced_equilibrium_manifest,
    manifest_entry,
    metadata_float,
    metadata_int,
    read_geqdsk,
    reduced_equilibrium_json_path,
    save_figure_outputs,
    scaled_font_size,
)
from scipy.interpolate import RegularGridInterpolator

from veqpy.model import Grid

PNG_PATH = figure_path("09-downstream-check.png")
PDF_PATH = None

LEVEL_LABELS = REDUCED_CONFIG_LABELS
LEGEND_LABEL_SPACING = 0.15
RADIAL_LINE_WIDTH = 1.4
EXTERNAL_RADIAL_LINE_WIDTH = 0.75 * RADIAL_LINE_WIDTH
EXTERNAL_RADIAL_MARKER_SIZE = 3.0 * RADIAL_LINE_WIDTH
EXTERNAL_RADIAL_ALPHA = 1.0
EXTERNAL_RADIAL_MARKER_COUNT = 10
COLUMN_TITLES = (
    r"$\mathbf{(a)}$ Temperature",
    r"$\mathbf{(b)}$ $T$ error",
    r"$\mathbf{(c)}$ $V'$ error",
    r"$\mathbf{(d)}$ $V'\langle|\nabla\hat{\psi}|^2\rangle$ error",
)

EVAL_POINTS = 128
ANALYTIC_THETA_POINTS = 384
VEQ_RHO_POINTS = 257
VEQ_THETA_POINTS = 512
X_FLOOR = 1.0e-4


@dataclass(frozen=True)
class TransportGeometry:
    label: str
    x: np.ndarray
    vprime: np.ndarray
    metric_weight: np.ndarray
    q: np.ndarray
    ip: float
    b0: float
    params: int | None = None
    elapsed_ms: float | None = None
    table5_time_ms: float | None = None


@dataclass(frozen=True)
class TransportResult:
    label: str
    geometry: TransportGeometry
    source: np.ndarray
    temperature: np.ndarray
    thermal_energy: float
    beta_proxy: float
    q95: float
    ip: float


def cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=np.float64)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def reverse_integral_to_edge(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return cumtrapz(y[::-1], x[::-1])[::-1] * -1.0


def interp_unique(x_src: np.ndarray, y_src: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x_src, dtype=np.float64)
    y_arr = np.asarray(y_src, dtype=np.float64)
    order = np.argsort(x_arr, kind="mergesort")
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]
    x_unique, unique_idx = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_idx]
    return np.interp(x_eval, x_unique, y_unique, left=float(y_unique[0]), right=float(y_unique[-1]))


def rel_rms(reference: np.ndarray, current: np.ndarray, x: np.ndarray) -> float:
    ref_all = np.asarray(reference, dtype=np.float64)
    cur_all = np.asarray(current, dtype=np.float64)
    x_all = np.asarray(x, dtype=np.float64)
    n = min(ref_all.size, cur_all.size, x_all.size)
    if n == 0:
        return float("nan")
    ref = ref_all[:n]
    cur = cur_all[:n]
    mask = np.isfinite(x_all[:n]) & np.isfinite(ref) & np.isfinite(cur)
    if not np.any(mask):
        return float("nan")
    ref = ref[mask]
    cur = cur[mask]
    scale = max(float(np.max(np.abs(ref))), 1.0e-14)
    return float(np.sqrt(np.mean((cur - ref) ** 2)) / scale)


def rel_abs(reference: float, current: float) -> float:
    return float(abs(current - reference) / max(abs(reference), 1.0e-14))


def format_sci(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if value == 0.0:
        return "$0$"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return rf"${mantissa:.{SCIENTIFIC_DECIMALS}f}\times 10^{{{exponent}}}$"


def close_curve(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return pts
    if np.allclose(pts[0], pts[-1], rtol=1.0e-12, atol=1.0e-12):
        return pts
    return np.vstack([pts, pts[:1]])


def point_in_polygon(poly: np.ndarray, R: float, Z: float) -> bool:
    """Return whether point ``(R, Z)`` lies inside a closed R-Z polygon."""

    pts = np.asarray(poly, dtype=np.float64)
    if pts.shape[0] < 3:
        return False
    inside = False
    j = pts.shape[0] - 1
    for i in range(pts.shape[0]):
        Ri, Zi = float(pts[i, 0]), float(pts[i, 1])
        Rj, Zj = float(pts[j, 0]), float(pts[j, 1])
        if (Zi > Z) != (Zj > Z):
            denom = Zj - Zi
            if abs(denom) < 1.0e-300:
                j = i
                continue
            crossing_R = (Rj - Ri) * (Z - Zi) / denom + Ri
            if R < crossing_R:
                inside = not inside
        j = i
    return inside


def select_flux_contour(
    candidates: list[np.ndarray], *, axis_center: tuple[float, float]
) -> np.ndarray | None:
    """Pick the longest contour enclosing the magnetic axis, falling back to the longest segment."""

    selected = None
    selected_length = -1
    for curve in candidates:
        arr = np.asarray(curve, dtype=np.float64)
        if arr.shape[0] < 8:
            continue
        if not point_in_polygon(close_curve(arr), axis_center[0], axis_center[1]):
            continue
        if arr.shape[0] > selected_length:
            selected = arr.copy()
            selected_length = arr.shape[0]
    if selected is not None:
        return selected
    if candidates:
        return max((np.asarray(curve, dtype=np.float64) for curve in candidates), key=len)
    return None


def geqdsk_contours(geqdsk, levels: np.ndarray) -> dict[float, np.ndarray]:
    """Extract GEQDSK flux-surface contours at normalized-flux levels."""

    psi = np.asarray(geqdsk.psi, dtype=np.float64)
    psi_span = float(geqdsk.psi_bound) - float(geqdsk.psi_axis)
    if psi_span == 0.0:
        raise ValueError("GEQDSK psi_axis and psi_bound are identical.")
    psin_grid = (psi.T - float(geqdsk.psi_axis)) / psi_span
    R = np.linspace(float(geqdsk.Rmin), float(geqdsk.Rmax), int(geqdsk.NR), dtype=np.float64)
    Z = np.linspace(float(geqdsk.Zmin), float(geqdsk.Zmax), int(geqdsk.NZ), dtype=np.float64)
    axis_center = (float(geqdsk.Raxis), float(geqdsk.Zaxis))
    surfaces: dict[float, np.ndarray] = {}

    finite = np.isfinite(psin_grid)
    if not np.any(finite):
        return surfaces
    grid_min = float(np.nanmin(psin_grid[finite]))
    grid_max = float(np.nanmax(psin_grid[finite]))
    contour_levels = [
        float(level)
        for level in np.asarray(levels, dtype=np.float64)
        if grid_min < float(level) < min(grid_max, 1.0 - 1.0e-12)
    ]
    if contour_levels:
        fig, ax = plt.subplots()
        contour = ax.contour(R, Z, psin_grid, levels=contour_levels)
        plt.close(fig)
        for idx, level in enumerate(contour_levels):
            selected = select_flux_contour(contour.allsegs[idx], axis_center=axis_center)
            if selected is not None:
                surfaces[level] = selected

    if np.any(np.isclose(np.asarray(levels, dtype=np.float64), 1.0, rtol=0.0, atol=1.0e-12)):
        surfaces[1.0] = np.asarray(geqdsk.boundary, dtype=np.float64)
    return surfaces


def fill_missing_profile(x: np.ndarray, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(arr)
    if np.all(finite):
        return arr
    if not np.any(finite):
        return np.full_like(arr, np.nan, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    arr[~finite] = np.interp(x_arr[~finite], x_arr[finite], arr[finite])
    return arr


def geqdsk_transport_geometry(geqdsk, x_eval: np.ndarray, *, label: str) -> TransportGeometry:
    """Compute downstream geometry factors directly from GEQDSK flux contours."""

    x = np.asarray(x_eval, dtype=np.float64)
    psi = np.asarray(geqdsk.psi, dtype=np.float64)
    psi_span = float(geqdsk.psi_bound) - float(geqdsk.psi_axis)
    if psi_span == 0.0:
        raise ValueError("GEQDSK psi_axis and psi_bound are identical.")
    psin = (psi - float(geqdsk.psi_axis)) / psi_span
    R_axis = np.linspace(float(geqdsk.Rmin), float(geqdsk.Rmax), int(geqdsk.NR), dtype=np.float64)
    Z_axis = np.linspace(float(geqdsk.Zmin), float(geqdsk.Zmax), int(geqdsk.NZ), dtype=np.float64)
    dR = float(R_axis[1] - R_axis[0])
    dZ = float(Z_axis[1] - Z_axis[0])
    psin_R, psin_Z = np.gradient(psin, dR, dZ, edge_order=2)
    grad_R_interp = RegularGridInterpolator(
        (R_axis, Z_axis), psin_R, bounds_error=False, fill_value=np.nan
    )
    grad_Z_interp = RegularGridInterpolator(
        (R_axis, Z_axis), psin_Z, bounds_error=False, fill_value=np.nan
    )

    surfaces = geqdsk_contours(geqdsk, x)
    vprime = np.full_like(x, np.nan, dtype=np.float64)
    metric_weight = np.full_like(x, np.nan, dtype=np.float64)
    for idx, level in enumerate(x):
        contour = surfaces.get(float(level))
        if contour is None:
            continue
        loop = close_curve(contour)
        if loop.shape[0] < 3:
            continue
        start = loop[:-1]
        stop = loop[1:]
        mid = 0.5 * (start + stop)
        ds = np.sqrt(np.sum((stop - start) ** 2, axis=1))
        points = np.column_stack((mid[:, 0], mid[:, 1]))
        grad_r = grad_R_interp(points)
        grad_z = grad_Z_interp(points)
        grad = np.sqrt(grad_r * grad_r + grad_z * grad_z)
        mask = (
            np.isfinite(ds) & np.isfinite(mid[:, 0]) & np.isfinite(grad) & (ds > 0.0) & (grad > 0.0)
        )
        if not np.any(mask):
            continue
        R_mid = np.maximum(mid[mask, 0], 1.0e-14)
        ds_masked = ds[mask]
        grad_masked = np.maximum(grad[mask], 1.0e-14)
        vprime[idx] = 2.0 * np.pi * float(np.sum(R_mid * ds_masked / grad_masked))
        metric_weight[idx] = 2.0 * np.pi * float(np.sum(R_mid * grad_masked * ds_masked))

    vprime = fill_missing_profile(x, vprime)
    metric_weight = fill_missing_profile(x, metric_weight)
    q_axis = np.linspace(0.0, 1.0, len(geqdsk.q), dtype=np.float64)
    return TransportGeometry(
        label=label,
        x=x,
        vprime=vprime,
        metric_weight=metric_weight,
        q=np.interp(x, q_axis, np.asarray(geqdsk.q, dtype=np.float64)),
        ip=float(geqdsk.Ip),
        b0=float(geqdsk.Bt0),
        params=None,
        elapsed_ms=None,
    )


def veq_transport_geometry(
    equilibrium,
    x_eval: np.ndarray,
    *,
    label: str,
    params: int | None,
    elapsed_ms: float | None,
):
    grid = Grid(
        Nr=VEQ_RHO_POINTS,
        Nt=VEQ_THETA_POINTS,
        quadrature_scheme="uniform",
        L_max=int(equilibrium.grid.L_max),
        M_max=int(equilibrium.grid.M_max),
    )
    eq = equilibrium.resample(grid=grid)
    psin = np.asarray(eq.psin, dtype=np.float64)
    psin_r = np.asarray(eq.psin_r, dtype=np.float64)
    vprime_rho = np.asarray(eq.V_r, dtype=np.float64) / np.maximum(psin_r, 1.0e-14)
    # M = V' <|grad psin|^2> = 2*pi*psin_r*int R*gtt/J dtheta.
    r_gtt_over_j = (
        np.asarray(eq.gttdivJR, dtype=np.float64) * np.asarray(eq.R, dtype=np.float64) ** 2
    )
    metric_rho = (
        2.0 * np.pi * psin_r * np.asarray(eq.grid.integrate(r_gtt_over_j, axis=1), dtype=np.float64)
    )

    return TransportGeometry(
        label=label,
        x=np.asarray(x_eval, dtype=np.float64),
        vprime=interp_unique(psin, vprime_rho, x_eval),
        metric_weight=interp_unique(psin, metric_rho, x_eval),
        q=interp_unique(psin, np.asarray(eq.q, dtype=np.float64), x_eval),
        ip=float(eq.Ip),
        b0=float(eq.B0),
        params=None if params is None else int(params),
        elapsed_ms=None if elapsed_ms is None else float(elapsed_ms),
        table5_time_ms=None,
    )


def solve_heat_profile(geometry: TransportGeometry, source: np.ndarray) -> TransportResult:
    x = geometry.x
    rhs = geometry.vprime * source
    cumulative_power = cumtrapz(rhs, x)
    diffusivity_weight = np.maximum(geometry.metric_weight, 1.0e-14)
    temperature = reverse_integral_to_edge(cumulative_power / diffusivity_weight, x)
    thermal_energy = 1.5 * float(np.trapezoid(geometry.vprime * temperature, x))
    volume = max(float(np.trapezoid(geometry.vprime, x)), 1.0e-14)
    beta_proxy = float(
        2.0 * MU0 * np.trapezoid(geometry.vprime * temperature, x) / (volume * geometry.b0**2)
    )
    q95 = float(np.interp(0.95, geometry.x, geometry.q))
    return TransportResult(
        label=geometry.label,
        geometry=geometry,
        source=source,
        temperature=temperature,
        thermal_energy=thermal_energy,
        beta_proxy=beta_proxy,
        q95=q95,
        ip=float(geometry.ip),
    )


def make_source(x: np.ndarray, reference_vprime: np.ndarray) -> np.ndarray:
    raw = np.exp(-(((x - 0.18) / 0.33) ** 2)) * (1.0 - 0.15 * x)
    raw = np.maximum(raw, 0.0)
    norm = max(float(np.trapezoid(reference_vprime * raw, x)), 1.0e-14)
    return raw / norm


def latex_error_table(case_results: list[tuple[str, list[TransportResult]]]) -> str:
    lines = [
        r"\hline",
        (
            r"Case & \(\mathrm{RMS}(\delta_T)\) & \(\mathrm{RMS}(\delta_{V'})\) & "
            r"\(\mathrm{RMS}(\delta_{V'\langle|\nabla\hat{\psi}|^2\rangle})\) & "
            r"\(\Delta_W\) & \(\Delta_{\beta_t}\) \\"
        ),
        r"\hline",
    ]
    for _case_key, results in case_results:
        reference = results[0]
        x = reference.geometry.x
        for result in results[1:]:
            geom = result.geometry
            lines.append(
                f"{result.label} & "
                f"{format_sci(rel_rms(reference.temperature, result.temperature, x))} & "
                f"{format_sci(rel_rms(reference.geometry.vprime, geom.vprime, x))} & "
                f"{format_sci(rel_rms(reference.geometry.metric_weight, geom.metric_weight, x))}"
                f" & {format_sci(rel_abs(reference.thermal_energy, result.thermal_energy))} & "
                f"{format_sci(rel_abs(reference.beta_proxy, result.beta_proxy))} \\\\"
            )
    lines.extend([r"\hline"])
    return "\n".join(lines)


def style_axis(ax: plt.Axes) -> None:
    ax.title.set_fontsize(scaled_font_size(TITLE_FONT_SIZE))
    ax.xaxis.label.set_fontsize(scaled_font_size(AXIS_LABEL_FONT_SIZE))
    ax.yaxis.label.set_fontsize(scaled_font_size(AXIS_LABEL_FONT_SIZE))
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
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.45)


def level_line_style(
    case_key: str, level_label: str
) -> tuple[str | tuple[int, tuple[float, ...]], str]:
    colors = CASE_LINE_COLORS[case_key]
    color_by_label = {
        "Ref": colors[-1],
        "Low": colors[-4],
        "Medium": colors[-3],
        "High": colors[-2],
    }
    return LEVEL_LINESTYLES[level_label], color_by_label[level_label]


def level_display_label(level_label: str, result: TransportResult) -> str:
    params = result.geometry.params
    if params is None:
        return level_label
    return f"{level_label} ({params:d})"


def marker_indices(length: int, count: int = EXTERNAL_RADIAL_MARKER_COUNT) -> list[int]:
    n = int(length)
    if n <= 0:
        return []
    if n <= int(count):
        return list(range(n))
    return np.unique(np.linspace(0, n - 1, int(count), dtype=int)).tolist()


def relative_profile_error(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference_arr = np.asarray(reference, dtype=np.float64)
    scale = max(float(np.nanmax(np.abs(reference_arr))), 1.0e-14)
    return np.abs(np.asarray(current, dtype=np.float64) - reference_arr) / scale


def plot_relative_error_family(
    ax: plt.Axes,
    *,
    x: np.ndarray,
    reference_profile: np.ndarray,
    level_results: list[tuple[str, TransportResult]],
    profile_getter,
    case_key: str,
    show_legend: bool = False,
    legend_loc: str = "upper right",
) -> None:
    for level_label, result in reversed(level_results):
        linestyle, color = level_line_style(case_key, level_label)
        rel = relative_profile_error(profile_getter(result), reference_profile)
        ax.semilogy(
            x,
            rel,
            color=color,
            linestyle=linestyle,
            linewidth=RADIAL_LINE_WIDTH,
            label=level_display_label(level_label, result),
        )
    ax.set_ylim(1.0e-8, 1.0e0)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[::-1],
            labels[::-1],
            loc=legend_loc,
            frameon=False,
            fontsize=scaled_font_size(LEGEND_FONT_SIZE),
            labelspacing=LEGEND_LABEL_SPACING,
        )


def plot_results(
    case_results: list[tuple[str, list[TransportResult]]],
    output_png: str | None,
    output_pdf: str | None,
) -> list[str]:
    apply_plot_style()
    fig, axes = plt.subplots(
        len(case_results),
        4,
        figsize=(DOUBLE_COLUMN_WIDTH, 6.15),
        constrained_layout=True,
        sharex="col",
    )
    axes = np.atleast_2d(axes)

    for row_idx, (case_key, results) in enumerate(case_results):
        reference = results[0]
        x = reference.geometry.x
        ax_t, ax_terr, ax_vprime, ax_metric = axes[row_idx]
        case_label = CASE_LABELS[case_key]
        show_xlabel = row_idx == len(case_results) - 1

        ax_t.plot(
            x,
            reference.temperature,
            label=reference.label,
            color="black",
            linestyle="-",
            linewidth=EXTERNAL_RADIAL_LINE_WIDTH,
            alpha=EXTERNAL_RADIAL_ALPHA,
            marker="o",
            markersize=EXTERNAL_RADIAL_MARKER_SIZE,
            markevery=marker_indices(len(x)),
            markerfacecolor="black",
            markeredgecolor="black",
        )
        level_results = list(zip(LEVEL_LABELS, results[1:], strict=True))
        for level_label, result in reversed(level_results):
            linestyle, color = level_line_style(case_key, level_label)
            ax_t.plot(
                x,
                result.temperature,
                label=level_display_label(level_label, result),
                color=color,
                linestyle=linestyle,
                linewidth=RADIAL_LINE_WIDTH,
            )
        ax_t.set_xlabel(r"$\hat{\psi}$" if show_xlabel else "")
        ax_t.set_ylabel(f"{case_label}\n" + r"$T$ [arb.]")
        if row_idx == 0:
            ax_t.set_title(COLUMN_TITLES[0])
        handles, labels = ax_t.get_legend_handles_labels()
        ax_t.legend(
            handles[::-1],
            labels[::-1],
            frameon=False,
            fontsize=scaled_font_size(LEGEND_FONT_SIZE),
            labelspacing=LEGEND_LABEL_SPACING,
        )

        plot_relative_error_family(
            ax_terr,
            x=x,
            reference_profile=reference.temperature,
            level_results=level_results,
            profile_getter=lambda result: result.temperature,
            case_key=case_key,
            show_legend=True,
            legend_loc="upper right",
        )
        ax_terr.set_xlabel(r"$\hat{\psi}$" if show_xlabel else "")
        ax_terr.set_ylabel("rel. error")
        if row_idx == 0:
            ax_terr.set_title(COLUMN_TITLES[1])

        plot_relative_error_family(
            ax_vprime,
            x=x,
            reference_profile=reference.geometry.vprime,
            level_results=level_results,
            profile_getter=lambda result: result.geometry.vprime,
            case_key=case_key,
        )
        ax_vprime.set_xlabel(r"$\hat{\psi}$" if show_xlabel else "")
        ax_vprime.set_ylabel("rel. error")
        if row_idx == 0:
            ax_vprime.set_title(COLUMN_TITLES[2])

        plot_relative_error_family(
            ax_metric,
            x=x,
            reference_profile=reference.geometry.metric_weight,
            level_results=level_results,
            profile_getter=lambda result: result.geometry.metric_weight,
            case_key=case_key,
        )
        ax_metric.set_xlabel(r"$\hat{\psi}$" if show_xlabel else "")
        ax_metric.set_ylabel("rel. error")
        if row_idx == 0:
            ax_metric.set_title(COLUMN_TITLES[3])

        style_axis(ax_t)
        style_axis(ax_terr)
        style_axis(ax_vprime)
        style_axis(ax_metric)

    saved_paths = save_figure_outputs(
        fig,
        png_path=output_png,
        pdf_path=output_pdf,
        dpi=SAVE_DPI,
        transparent=SAVE_TRANSPARENT,
    )
    plt.close(fig)
    return saved_paths


def prepare_case_reference(case_key: str, x_eval: np.ndarray) -> TransportGeometry:
    geqdsk = read_geqdsk(CASE_REFERENCE_GFILES[case_key])
    return geqdsk_transport_geometry(geqdsk, x_eval, label=REFERENCE_LABELS[case_key])


def load_reduced_transport_geometries(
    manifest: dict[tuple[str, str], dict[str, object]],
    case_key: str,
    x_eval: np.ndarray,
) -> list[TransportGeometry]:
    geometries: list[TransportGeometry] = []
    for config_label in LEVEL_LABELS:
        metadata = manifest_entry(manifest, case_key, config_label)
        default_path = reduced_equilibrium_json_path(case_key, config_label)
        path = str(metadata.get("path", default_path))
        try:
            equilibrium = load_equilibrium_json(path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Missing {CASE_LABELS[case_key]} {config_label} "
                f"reduced equilibrium JSON at {path}. "
                "Run `python scripts/07-pareto-analysis.py` first so Figure 09 uses the same "
                "representative equilibria as Figures 07/08."
            ) from exc

        params = metadata_int(metadata, "parameter_count")
        label = (
            f"{CASE_LABELS[case_key]}({params:d})"
            if params is not None
            else f"{CASE_LABELS[case_key]} {config_label}"
        )
        geometries.append(
            veq_transport_geometry(
                equilibrium,
                x_eval,
                label=label,
                params=params,
                elapsed_ms=metadata_float(metadata, "elapsed_ms"),
            )
        )
    return geometries


def solve_case_results(
    manifest: dict[tuple[str, str], dict[str, object]],
    case_key: str,
    x_eval: np.ndarray,
) -> list[TransportResult]:
    reference_geometry = prepare_case_reference(case_key, x_eval)
    veq_geometries = load_reduced_transport_geometries(manifest, case_key, x_eval)
    source = make_source(x_eval, reference_geometry.vprime)
    return [solve_heat_profile(reference_geometry, source)] + [
        solve_heat_profile(geometry, source) for geometry in veq_geometries
    ]


def main() -> None:
    manifest = load_reduced_equilibrium_manifest()

    x_eval = np.linspace(0.0, 1.0, EVAL_POINTS, dtype=np.float64)
    x_eval[0] = X_FLOOR

    case_results = [
        (case_key, solve_case_results(manifest, case_key, x_eval)) for case_key in CASE_KEYS
    ]
    saved_paths = plot_results(case_results, PNG_PATH, PDF_PATH)

    for path in saved_paths:
        print(f"wrote {path}")
    print(latex_error_table(case_results))


if __name__ == "__main__":
    main()
