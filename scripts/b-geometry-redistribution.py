"""Appendix-B single-column VAR/collocation force-redistribution diagnostic.

This standalone script uses the same benchmark cases, reduced active levels,
warm-started point-collocation polish, and Figure-08 radial-line style
vocabulary as the appendix comparison, but plots only the radial ratio between
the collocation-polished and variational sampled force residuals.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from _cases import (
    CASE_COLORS,
    CASE_KEYS,
    CASE_LABELS,
    CASE_LINE_COLORS,
    CASE_REFERENCE_GFILES,
    CASE_REFERENCE_PROFILE_LENGTHS,
    CASE_SOLVER_METHODS,
    CONFIG_LABELS,
    CONFIG_LINE_COLORS,
    REDUCED_CONFIG_LABELS,
)
from _common import ensure_parent_dir, figure_path, save_figure_outputs
from _kernel_cases import (
    build_pf_case,
    build_pf_reference_case,
    load_equilibrium_json,
    load_pf_benchmark,
    load_reduced_equilibrium_manifest,
    normalize_signature,
    read_geqdsk,
    reduced_equilibrium_json_path,
    signature_from_metadata,
)
from _plotting import (
    AXIS_LABEL_SIZE,
    LEGEND_FONT_SIZE,
    PLOT_TICK_BOTTOM,
    PLOT_TICK_DIRECTION,
    PLOT_TICK_LEFT,
    PLOT_TICK_RIGHT,
    PLOT_TICK_TOP,
    SAVE_DPI,
    SAVE_TRANSPARENT,
    SINGLE_COLUMN_WIDTH,
    TICK_LABEL_SIZE,
    TITLE_FONT_SIZE,
    apply_plot_style,
    scaled_font_size,
)
from _reporting import (
    SCRIPT_CONSOLE,
    format_script_sci,
    make_script_table,
    print_output_table,
    print_script_config,
    print_script_table,
    script_progress,
)
from matplotlib import ticker
from matplotlib.lines import Line2D
from scipy.optimize import least_squares

PNG_PATH = figure_path("b-geometry-redistribution.png")
PDF_PATH = None
COMPACT_PNG_PATH = figure_path("b-geometry-redistribution-compact.png")
FIGURE_HEIGHT = 4.45
PANEL_TITLE_PAD = 0.030
EPS = 1.0e-300
DISPLAY_CONFIG_LABELS = ("Low", "Medium", "High", "Ref")
REFERENCE_PROFILE_LENGTHS = CASE_REFERENCE_PROFILE_LENGTHS
DEFAULT_COLLOCATION_METHOD = "lm"
DEFAULT_COLLOCATION_WEIGHT = 1.0
DEFAULT_MAX_NFEV = 500
DEFAULT_REPEAT_COUNT = 5
DEFAULT_SOLVE_NR = 32
DEFAULT_SOLVE_NT = 32
BACKEND = "numba"
CASE_KEYS_TO_RUN = "all"
INITIAL_SOLVE_TIMEOUT_S = 30.0
MAX_RESIDUAL = 1.0e-8
SAVE_TABLE_PATH = None
PRINT_TABLE = True
STANDARD_GS_JDIVER_FLOOR = 1.0e-30
RESIDUAL_LOG_YLIM = (-2.0, 2.0)
EXTERNAL_SHAPE_SURFACE_NR = 128
EXTERNAL_SHAPE_SURFACE_NT = 256
EXTERNAL_SHAPE_THETA_SAMPLE_COUNT = 16
EXTERNAL_SHAPE_PSIN_LEVELS = tuple(np.linspace(0.0, 1.0, 11, dtype=np.float64))

CONFIG_LINE_COLORS = {
    **CONFIG_LINE_COLORS,
    "Ref": "#111111",
}
EXTERNAL_RADIAL_MARKER_SIZE = 4.2
EXTERNAL_RADIAL_MARKER_COUNT = 15
RADIAL_LINE_WIDTH = 1.4
GRID_ALPHA = 0.25
GRID_LINE_WIDTH = 0.5
GRID_LINESTYLE = "-"
HEATMAP_X_TICKS = {
    "solovev": (4.0, 8.0),
    "chease": (0.5, 1.5),
    "efit": (1.0, 2.0),
}
HEATMAP_X_TICK_BINS = 2
HEATMAP_Y_TICK_BINS = 4
SHAPE_SURFACE_LEVELS = (0.2, 0.4, 0.6, 0.8, 1.0)
SHAPE_LINE_WIDTH = 1.0
SHAPE_BOUNDARY_LINE_WIDTH = 1.35
SHAPE_TARGET_LINESTYLE = (0, (4.0, 2.0))
SHAPE_TARGET_LINE_WIDTH_SCALE = 1.5
COMPACT_CONFIG_LABELS = ("Low", "Medium", "High", "Ref")
COMPACT_COLUMN_LABELS = ("Low", "Medium", "High", "Ref")
COMPACT_FIGURE_WIDTH = SINGLE_COLUMN_WIDTH
COMPACT_ROW_HEIGHT = 1.55
COMPACT_LEFT = 0.14
COMPACT_RIGHT = 0.98
COMPACT_BOTTOM = 0.08
COMPACT_TOP = 0.88
COMPACT_WSPACE = 0.04
COMPACT_HSPACE = 0.18


@dataclass(frozen=True)
class ResidualSample:
    case_key: str
    config_label: str
    signature: dict[str, int]
    parameter_count: int | None
    elapsed_ms: float
    solver_residual_norm: float
    rho: np.ndarray
    psin: np.ndarray
    R: np.ndarray
    Z: np.ndarray
    G: np.ndarray
    radial_rms: np.ndarray


@dataclass(frozen=True)
class GeometryRedistributionSample:
    weak: ResidualSample
    collocation: ResidualSample
    weak_force_radial: np.ndarray
    collocation_force_radial: np.ndarray
    force_radial_ratio: np.ndarray
    shape_rms_over_a: float
    shape_max_over_a: float
    weak_external_shape_error_over_a: float
    collocation_external_shape_error_over_a: float
    external_shape_error_ratio: float
    force_rms_ratio: float
    nfev: int
    success: bool


class InitialSolveTimeoutError(RuntimeError):
    def __init__(self, *, elapsed_ms: float, timeout_s: float):
        self.elapsed_ms = float(elapsed_ms)
        self.timeout_s = float(timeout_s)
        super().__init__(
            f"Initial solve exceeded {self.timeout_s:.2f} s ({self.elapsed_ms:.2f} ms)"
        )


def residual_vector_radial_rms(vector: np.ndarray, *, nr: int, nt: int) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64).reshape((int(nr), int(nt)))
    return np.sqrt(np.nanmean(values * values, axis=1))


def finite_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    out = np.full(np.broadcast_shapes(num.shape, den.shape), np.nan, dtype=np.float64)
    mask = np.isfinite(num) & np.isfinite(den) & (den > 0.0)
    out[mask] = num[mask] / den[mask]
    return out


def marker_indices(length: int, count: int = EXTERNAL_RADIAL_MARKER_COUNT) -> list[int]:
    n = int(length)
    if n <= 0:
        return []
    if n <= int(count):
        return list(range(n))
    return np.unique(np.linspace(0, n - 1, int(count), dtype=int)).tolist()


def standard_gs_residual_from_equilibrium(equilibrium) -> np.ndarray:
    G = np.asarray(equilibrium.G, dtype=np.float64)
    JdivR = np.asarray(equilibrium.JdivR, dtype=np.float64)
    G_std = np.full_like(G, np.nan, dtype=np.float64)
    np.divide(
        G,
        JdivR,
        out=G_std,
        where=np.isfinite(G)
        & np.isfinite(JdivR)
        & (np.abs(JdivR) > STANDARD_GS_JDIVER_FLOOR),
    )
    return G_std


def sample_legend_label(sample: ResidualSample) -> str:
    if sample.parameter_count is None:
        return str(sample.config_label)
    return f"{sample.config_label} ({int(sample.parameter_count)})"


def surface_at_psin_level(sample: ResidualSample, level: float) -> tuple[np.ndarray, np.ndarray]:
    psin = np.asarray(sample.psin, dtype=np.float64)
    if psin.ndim != 1 or psin.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    valid = np.isfinite(psin)
    if not np.any(valid):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    order = np.argsort(psin[valid])
    psin_sorted = psin[valid][order]
    R_sorted = np.asarray(sample.R, dtype=np.float64)[valid, :][order, :]
    Z_sorted = np.asarray(sample.Z, dtype=np.float64)[valid, :][order, :]
    if float(level) <= psin_sorted[0]:
        R, Z = R_sorted[0], Z_sorted[0]
    elif float(level) >= psin_sorted[-1]:
        R, Z = R_sorted[-1], Z_sorted[-1]
    else:
        R = np.asarray(
            [
                np.interp(float(level), psin_sorted, R_sorted[:, idx])
                for idx in range(R_sorted.shape[1])
            ]
        )
        Z = np.asarray(
            [
                np.interp(float(level), psin_sorted, Z_sorted[:, idx])
                for idx in range(Z_sorted.shape[1])
            ]
        )
    return np.r_[R, R[0]], np.r_[Z, Z[0]]


def build_equilibrium_surface_from_psin(equilibrium, level: float) -> np.ndarray:
    psin = np.asarray(equilibrium.psin, dtype=np.float64)
    rho = np.asarray(equilibrium.rho, dtype=np.float64)
    order = np.argsort(psin)
    psin_unique, unique_idx = np.unique(psin[order], return_index=True)
    rho_level = float(np.interp(float(level), psin_unique, rho[order][unique_idx]))
    R = np.asarray(
        [np.interp(rho_level, rho, equilibrium.R[:, idx]) for idx in range(equilibrium.grid.Nt)],
        dtype=np.float64,
    )
    Z = np.asarray(
        [np.interp(rho_level, rho, equilibrium.Z[:, idx]) for idx in range(equilibrium.grid.Nt)],
        dtype=np.float64,
    )
    return np.column_stack((R, Z))


def point_in_polygon(points: np.ndarray, R: float, Z: float) -> bool:
    vertices = np.asarray(points, dtype=np.float64)
    inside = False
    j = vertices.shape[0] - 1
    for i in range(vertices.shape[0]):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if (yi > Z) != (yj > Z):
            dy = yj - yi
            if abs(dy) < 1.0e-14:
                dy = 1.0e-14 if dy >= 0.0 else -1.0e-14
            x_cross = (xj - xi) * (Z - yi) / dy + xi
            if R < x_cross:
                inside = not inside
        j = i
    return inside


def select_gfile_contour(
    candidates: list[np.ndarray], *, axis_center: tuple[float, float]
) -> np.ndarray | None:
    selected = None
    selected_length = -1
    for curve in candidates:
        arr = np.asarray(curve, dtype=np.float64)
        if arr.shape[0] < 8:
            continue
        if not point_in_polygon(arr, axis_center[0], axis_center[1]):
            continue
        if arr.shape[0] > selected_length:
            selected = arr.copy()
            selected_length = arr.shape[0]
    if selected is not None:
        return selected
    if candidates:
        return max((np.asarray(curve, dtype=np.float64) for curve in candidates), key=len)
    return None


def extract_gfile_surfaces(geqdsk, levels: tuple[float, ...]) -> dict[float, np.ndarray]:
    psi_span = float(geqdsk.psi_bound) - float(geqdsk.psi_axis)
    psin_grid = (np.asarray(geqdsk.psi.T, dtype=np.float64) - float(geqdsk.psi_axis)) / psi_span
    R = np.linspace(float(geqdsk.Rmin), float(geqdsk.Rmax), int(geqdsk.NR), dtype=np.float64)
    Z = np.linspace(float(geqdsk.Zmin), float(geqdsk.Zmax), int(geqdsk.NZ), dtype=np.float64)
    axis_center = (float(geqdsk.Raxis), float(geqdsk.Zaxis))
    surfaces: dict[float, np.ndarray] = {}
    contour_levels = [float(level) for level in levels if level < 1.0 - 1.0e-12]
    if contour_levels:
        fig, ax = plt.subplots()
        contour = ax.contour(R, Z, psin_grid, levels=contour_levels)
        plt.close(fig)
        for idx, level in enumerate(contour_levels):
            selected = select_gfile_contour(contour.allsegs[idx], axis_center=axis_center)
            if selected is not None:
                surfaces[level] = selected
    if any(abs(float(level) - 1.0) <= 1.0e-12 for level in levels):
        surfaces[1.0] = np.asarray(geqdsk.boundary, dtype=np.float64)
    return surfaces


def axis_position_error(
    reference_axis: tuple[float, float], equilibrium_axis: tuple[float, float]
) -> float:
    reference = np.asarray(reference_axis, dtype=np.float64)
    current = np.asarray(equilibrium_axis, dtype=np.float64)
    return float(np.linalg.norm(current - reference))


def sample_curve_at_theta(
    points: np.ndarray, *, center: tuple[float, float], theta_eval: np.ndarray
) -> np.ndarray:
    curve = np.asarray(points, dtype=np.float64)
    theta = np.mod(np.arctan2(curve[:, 1] - center[1], curve[:, 0] - center[0]), 2.0 * np.pi)
    order = np.argsort(theta, kind="mergesort")
    theta_sorted = theta[order]
    curve_sorted = curve[order]
    theta_periodic = np.concatenate((theta_sorted, [theta_sorted[0] + 2.0 * np.pi]))
    R_periodic = np.concatenate((curve_sorted[:, 0], [curve_sorted[0, 0]]))
    Z_periodic = np.concatenate((curve_sorted[:, 1], [curve_sorted[0, 1]]))
    theta_target = np.mod(np.asarray(theta_eval, dtype=np.float64), 2.0 * np.pi)
    return np.column_stack(
        (
            np.interp(theta_target, theta_periodic, R_periodic),
            np.interp(theta_target, theta_periodic, Z_periodic),
        )
    )


def radial_profile_from_surface(
    points: np.ndarray,
    *,
    center: tuple[float, float],
    theta_eval: np.ndarray,
) -> np.ndarray:
    sampled = sample_curve_at_theta(points, center=center, theta_eval=theta_eval)
    center_arr = np.asarray(center, dtype=np.float64)
    return np.sqrt(np.sum((sampled - center_arr[None, :]) ** 2, axis=1))


def external_shape_error_over_a(equilibrium, *, case_key: str) -> float:
    grid = equilibrium.grid.__class__(
        Nr=max(int(EXTERNAL_SHAPE_SURFACE_NR), int(equilibrium.grid.Nr)),
        Nt=max(int(EXTERNAL_SHAPE_SURFACE_NT), int(equilibrium.grid.Nt)),
        quadrature_scheme="uniform",
        L_max=int(equilibrium.grid.L_max),
        M_max=int(equilibrium.grid.M_max),
    )
    plot_equilibrium = equilibrium.resample(grid=grid)
    geqdsk = read_geqdsk(CASE_REFERENCE_GFILES[case_key])
    theta_eval = np.linspace(
        0.0,
        2.0 * np.pi,
        int(EXTERNAL_SHAPE_THETA_SAMPLE_COUNT),
        endpoint=False,
        dtype=np.float64,
    )
    reference_center = (float(geqdsk.Raxis), float(geqdsk.Zaxis))
    equilibrium_center = (
        float(plot_equilibrium.R[0, 0]),
        float(plot_equilibrium.Z[0, 0]),
    )
    gfile_levels = tuple(
        float(level) for level in EXTERNAL_SHAPE_PSIN_LEVELS if float(level) > 1.0e-12
    )
    surfaces = extract_gfile_surfaces(geqdsk, gfile_levels)
    rms_values: list[float] = []
    for level in EXTERNAL_SHAPE_PSIN_LEVELS:
        level = float(level)
        if abs(level) <= 1.0e-12:
            rms_values.append(axis_position_error(reference_center, equilibrium_center))
            continue
        if level not in surfaces:
            continue
        reference_surface = np.asarray(surfaces[level], dtype=np.float64)
        equilibrium_surface = build_equilibrium_surface_from_psin(plot_equilibrium, level)
        reference_r = radial_profile_from_surface(
            reference_surface, center=reference_center, theta_eval=theta_eval
        )
        equilibrium_r = radial_profile_from_surface(
            equilibrium_surface, center=equilibrium_center, theta_eval=theta_eval
        )
        rms_values.append(float(np.sqrt(np.mean((equilibrium_r - reference_r) ** 2))))
    values = np.asarray(rms_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    error = float("nan") if values.size == 0 else float(np.sqrt(np.mean(values * values)))
    return error / max(float(equilibrium.a), 1.0e-12)


def rz_limits(samples: list[ResidualSample]) -> tuple[tuple[float, float], tuple[float, float]]:
    R_all = np.concatenate([np.ravel(sample.R) for sample in samples])
    Z_all = np.concatenate([np.ravel(sample.Z) for sample in samples])
    rmin, rmax = float(np.nanmin(R_all)), float(np.nanmax(R_all))
    zmin, zmax = float(np.nanmin(Z_all)), float(np.nanmax(Z_all))
    rpad = max(0.06 * (rmax - rmin), 1.0e-6)
    zpad = max(0.06 * (zmax - zmin), 1.0e-6)
    return (rmin - rpad, rmax + rpad), (zmin - zpad, zmax + zpad)


def style_rz_axis(ax: plt.Axes, *, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    apply_box_ticks(ax)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=HEATMAP_X_TICK_BINS))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=HEATMAP_Y_TICK_BINS))


def apply_box_ticks(ax: plt.Axes, **kwargs) -> None:
    ax.tick_params(
        which="both",
        direction=PLOT_TICK_DIRECTION,
        top=PLOT_TICK_TOP,
        right=PLOT_TICK_RIGHT,
        bottom=PLOT_TICK_BOTTOM,
        left=PLOT_TICK_LEFT,
        labelsize=scaled_font_size(TICK_LABEL_SIZE),
        **kwargs,
    )


def residual_norm_row(sample: ResidualSample) -> dict[str, object]:
    residual_rms_interior = region_rms(sample, upper=0.8)
    residual_rms_edge = region_rms(sample, lower=0.8)
    return {
        "case_key": sample.case_key,
        "case": CASE_LABELS[sample.case_key],
        "case_params": (
            f"{CASE_LABELS[sample.case_key]} {sample.config_label}"
            if sample.parameter_count is None
            else f"{CASE_LABELS[sample.case_key]} ({int(sample.parameter_count)})"
        ),
        "config": sample.config_label,
        "params": None if sample.parameter_count is None else int(sample.parameter_count),
        "epsilon_proj": float(sample.solver_residual_norm),
        "G_rms": rms(sample.G),
        "G_rms_interior": residual_rms_interior,
        "G_rms_edge": residual_rms_edge,
        "G_max": float(np.nanmax(np.abs(sample.G))),
    }


def rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite * finite)))


def region_rms(
    sample: ResidualSample, *, lower: float | None = None, upper: float | None = None
) -> float:
    psin = np.asarray(sample.psin, dtype=np.float64)
    values = np.asarray(sample.G, dtype=np.float64)
    if psin.ndim != 1 or values.ndim == 0 or values.shape[0] != psin.shape[0]:
        return float("nan")
    mask = np.ones(psin.shape, dtype=bool)
    if lower is not None:
        mask &= psin >= float(lower)
    if upper is not None:
        mask &= psin < float(upper)
    if not np.any(mask):
        return float("nan")
    return rms(values[mask, ...])


def minor_radius(sample: ResidualSample) -> float:
    boundary_r, _ = surface_at_psin_level(sample, 1.0)
    if boundary_r.size:
        finite = boundary_r[np.isfinite(boundary_r)]
        if finite.size:
            return max(0.5 * float(np.nanmax(finite) - np.nanmin(finite)), 1.0e-30)
    return max(0.5 * float(np.nanmax(sample.R[-1]) - np.nanmin(sample.R[-1])), 1.0e-30)


def shape_mismatch_over_a(
    weak: ResidualSample,
    collocation: ResidualSample,
) -> tuple[float, float]:
    """RMS and maximum displacement of equal-psin surfaces, normalized by minor radius."""

    d2_blocks: list[np.ndarray] = []
    for level in SHAPE_SURFACE_LEVELS:
        weak_r, weak_z = surface_at_psin_level(weak, level)
        coll_r, coll_z = surface_at_psin_level(collocation, level)
        if weak_r.size == 0 or coll_r.size == 0:
            continue
        # Drop the duplicated closing point and compare the shared computational
        # theta sampling used by both states.
        count = min(weak_r.size, coll_r.size)
        if (
            count > 1
            and np.allclose(weak_r[0], weak_r[count - 1])
            and np.allclose(weak_z[0], weak_z[count - 1])
        ):
            count -= 1
        weak_r = weak_r[:count]
        weak_z = weak_z[:count]
        coll_r = coll_r[:count]
        coll_z = coll_z[:count]
        d2 = (coll_r - weak_r) * (coll_r - weak_r) + (coll_z - weak_z) * (coll_z - weak_z)
        d2_blocks.append(d2[np.isfinite(d2)])
    if not d2_blocks:
        return float("nan"), float("nan")
    d2_all = np.concatenate(d2_blocks)
    scale = minor_radius(weak)
    return float(np.sqrt(np.mean(d2_all)) / scale), float(np.sqrt(np.max(d2_all)) / scale)


def selected_signature_map(case_keys: tuple[str, ...]) -> dict[str, list[dict[str, int]]]:
    """Return the Low/Medium/High/Ref signatures used by Figure 08."""

    manifest = load_reduced_equilibrium_manifest()
    signatures_by_case: dict[str, list[dict[str, int]]] = {}
    for case_key in case_keys:
        signatures: list[dict[str, int]] = []
        for config_label in REDUCED_CONFIG_LABELS:
            metadata = manifest.get((case_key, config_label), {})
            signature = signature_from_metadata(metadata)
            if not signature:
                default_path = reduced_equilibrium_json_path(case_key, config_label)
                path = str(metadata.get("path", default_path))
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Missing {CASE_LABELS[case_key]} {config_label} "
                        f"reduced equilibrium JSON at {path}. "
                        "Run `python scripts/07-pareto-analysis.py` first."
                    )
                _ = load_equilibrium_json(path)
                signature = {}
            signatures.append(signature)
        signatures.append(normalize_signature(REFERENCE_PROFILE_LENGTHS[case_key]))
        signatures_by_case[case_key] = signatures
    return signatures_by_case


def make_residual_sample(
    *,
    case_key: str,
    config_label: str,
    signature: dict[str, int],
    parameter_count: int,
    elapsed_ms: float,
    packed_residual_norm: float,
    equilibrium,
) -> ResidualSample:
    G = standard_gs_residual_from_equilibrium(equilibrium)
    radial_rms = np.sqrt(np.nanmean(G * G, axis=1))
    return ResidualSample(
        case_key=case_key,
        config_label=config_label,
        signature=dict(signature),
        parameter_count=int(parameter_count),
        elapsed_ms=float(elapsed_ms),
        solver_residual_norm=float(packed_residual_norm),
        rho=np.asarray(equilibrium.rho, dtype=np.float64),
        psin=np.asarray(equilibrium.psin, dtype=np.float64),
        R=np.asarray(equilibrium.R, dtype=np.float64),
        Z=np.asarray(equilibrium.Z, dtype=np.float64),
        G=G,
        radial_rms=radial_rms,
    )


def solve_variational_with_average_timing(
    benchmark,
    case,
    grid,
    repeat_count: int,
    *,
    method: str,
    solver_config,
    initial_solve_timeout_s: float,
):
    del grid, method
    kernel = benchmark.Kernel(
        topology=case.topology,
        recipe=benchmark.KernelRecipe(backend=benchmark.BACKEND),
        config=solver_config,
    )

    probe_started = time.perf_counter()
    result = kernel.solve(case.boundary, case.source, config=solver_config)
    probe_elapsed_ms = (time.perf_counter() - probe_started) * 1000.0
    probe_elapsed_ms = max(probe_elapsed_ms, float(result.elapsed_ms))
    if probe_elapsed_ms > float(initial_solve_timeout_s) * 1000.0:
        kernel.close()
        raise InitialSolveTimeoutError(
            elapsed_ms=probe_elapsed_ms,
            timeout_s=initial_solve_timeout_s,
        )

    elapsed_values: list[float] = []
    final_result = None
    for _ in range(max(int(repeat_count), 1)):
        final_result = kernel.solve(case.boundary, case.source, config=solver_config)
        elapsed_values.append(float(final_result.elapsed_ms))
    if final_result is None:
        kernel.close()
        raise RuntimeError("No variational solver result returned")

    return final_result, kernel.build_equilibrium(), float(np.mean(elapsed_values)), kernel


def solve_collocation_with_average_timing(
    kernel,
    case,
    weak_x: np.ndarray,
    weak_elapsed_ms: float,
    args: SimpleNamespace,
    *,
    solver_config,
):
    """Time only the extra mixed-collocation polish and report weak+postprocess time."""

    del solver_config
    # Appendix-B is a backend diagnostic: collocation is not part of the public
    # Kernel API, so the script binds directly to the Numba runtime runner here.
    runtime = kernel._impl._solver.runtime
    weak_x_eval = runtime.coerce_x(weak_x).copy()
    collocation_weight = float(args.collocation_weight)

    def collocation_residual(x_eval: np.ndarray) -> np.ndarray:
        out = np.empty(int(case.topology.Nr) * int(case.topology.Nt), dtype=np.float64)
        runtime.layout.run_collocation_into(runtime.coerce_x(x_eval), out)
        if collocation_weight != 1.0:
            out *= collocation_weight
        return out

    if collocation_weight <= 0.0:
        residual = collocation_residual(weak_x_eval)
        result = SimpleNamespace(
            x=weak_x_eval.copy(),
            success=True,
            message="collocation_weight=0; no post-processing polish",
            function_evaluations=0,
            jacobian_evaluations=0,
            iterations=0,
            residual_norm_final=float(np.linalg.norm(residual)),
            elapsed=0.0,
        )
        return result, kernel.build_equilibrium(weak_x_eval), float(weak_elapsed_ms), 0.0

    def run_one_polish():
        opt = least_squares(
            collocation_residual,
            weak_x_eval,
            method=str(args.collocation_method),
            ftol=float(args.max_residual),
            xtol=float(args.max_residual),
            gtol=float(args.max_residual),
            max_nfev=int(args.max_nfev),
        )
        return opt

    # Warm-up: triggers any least-squares/Jacobian setup but is not included in
    # the reported extra post-processing time.
    run_one_polish()

    elapsed_values: list[float] = []
    final_opt = None
    for _ in range(max(int(args.collocation_repeat_count), 1)):
        started = time.perf_counter()
        final_opt = run_one_polish()
        elapsed_values.append((time.perf_counter() - started) * 1000.0)
    if final_opt is None:
        raise RuntimeError("Mixed-collocation post-process did not produce a result")

    x_final = runtime.coerce_x(final_opt.x).copy()
    residual_norm = float(np.linalg.norm(collocation_residual(x_final)))
    postprocess_elapsed_ms = float(np.mean(elapsed_values))
    total_elapsed_ms = float(weak_elapsed_ms) + postprocess_elapsed_ms
    result = SimpleNamespace(
        x=x_final,
        success=bool(final_opt.success),
        message=str(final_opt.message),
        function_evaluations=int(getattr(final_opt, "nfev", 0) or 0),
        jacobian_evaluations=int(getattr(final_opt, "njev", 0) or 0),
        iterations=int(getattr(final_opt, "nit", 0) or 0),
        residual_norm_final=float(residual_norm),
        elapsed=postprocess_elapsed_ms * 1000.0,
    )
    return result, kernel.build_equilibrium(result.x), total_elapsed_ms, postprocess_elapsed_ms


def result_count(result, solver_name: str, optimize_name: str) -> int:
    value = getattr(result, solver_name, getattr(result, optimize_name, 0))
    if value is None:
        return 0
    return int(value)


def solve_geometry_redistribution_sample(
    benchmark,
    reference,
    *,
    case_key: str,
    config_label: str,
    signature: dict[str, int],
    args: SimpleNamespace,
) -> GeometryRedistributionSample:
    grid = benchmark.Grid(
        Nr=int(args.solve_nr),
        Nt=int(args.solve_nt),
        quadrature_scheme="legendre",
        L_max=int(benchmark.REFERENCE_GRID.L_max),
        M_max=int(benchmark.REFERENCE_GRID.M_max),
    )
    case = build_pf_case(benchmark, reference, signature, grid)

    weak_result, weak_equilibrium, weak_elapsed_ms, kernel = solve_variational_with_average_timing(
        benchmark,
        case,
        grid,
        int(args.weak_repeat_count),
        method=CASE_SOLVER_METHODS[case_key],
        solver_config=benchmark.CONFIG,
        initial_solve_timeout_s=float(args.initial_solve_timeout_s),
    )

    try:
        # Use the same backend diagnostic runner as the collocation polish above.
        runtime = kernel._impl._solver.runtime

        def collocation_vector(x_eval: np.ndarray) -> np.ndarray:
            out = np.empty(int(case.topology.Nr) * int(case.topology.Nt), dtype=np.float64)
            runtime.layout.run_collocation_into(runtime.coerce_x(x_eval), out)
            return out

        weak_x = runtime.coerce_x(weak_result.x).copy()
        weak_packed = np.asarray(
            kernel.residual(weak_x, case.boundary, case.source),
            dtype=np.float64,
        )
        weak_force_vector = np.asarray(collocation_vector(weak_x), dtype=np.float64)

        collocation_result, collocation_equilibrium, elapsed_ms, _postprocess_elapsed_ms = (
            solve_collocation_with_average_timing(
                kernel,
                case,
                weak_x,
                weak_elapsed_ms,
                args,
                solver_config=benchmark.CONFIG,
            )
        )

        x_final = runtime.coerce_x(collocation_result.x)
        final_packed = np.asarray(
            kernel.residual(x_final, case.boundary, case.source),
            dtype=np.float64,
        )
        collocation_force_vector = np.asarray(collocation_vector(x_final), dtype=np.float64)
    finally:
        kernel.close()

    weak_sample = make_residual_sample(
        case_key=case_key,
        config_label=config_label,
        signature=signature,
        parameter_count=weak_x.size,
        elapsed_ms=weak_elapsed_ms,
        packed_residual_norm=float(np.linalg.norm(weak_packed)),
        equilibrium=weak_equilibrium,
    )
    collocation_sample = make_residual_sample(
        case_key=case_key,
        config_label=config_label,
        signature=signature,
        parameter_count=x_final.size,
        elapsed_ms=elapsed_ms,
        packed_residual_norm=float(np.linalg.norm(final_packed)),
        equilibrium=collocation_equilibrium,
    )

    weak_force_radial = residual_vector_radial_rms(weak_force_vector, nr=grid.Nr, nt=grid.Nt)
    collocation_force_radial = residual_vector_radial_rms(
        collocation_force_vector, nr=grid.Nr, nt=grid.Nt
    )
    shape_rms, shape_max = shape_mismatch_over_a(weak_sample, collocation_sample)
    weak_external_shape_error = external_shape_error_over_a(weak_equilibrium, case_key=case_key)
    collocation_external_shape_error = external_shape_error_over_a(
        collocation_equilibrium, case_key=case_key
    )
    weak_force_rms = rms(weak_force_vector)
    collocation_force_rms = rms(collocation_force_vector)

    return GeometryRedistributionSample(
        weak=weak_sample,
        collocation=collocation_sample,
        weak_force_radial=weak_force_radial,
        collocation_force_radial=collocation_force_radial,
        force_radial_ratio=finite_ratio(collocation_force_radial, weak_force_radial),
        shape_rms_over_a=shape_rms,
        shape_max_over_a=shape_max,
        weak_external_shape_error_over_a=weak_external_shape_error,
        collocation_external_shape_error_over_a=collocation_external_shape_error,
        external_shape_error_ratio=scalar_ratio(
            collocation_external_shape_error, weak_external_shape_error
        ),
        force_rms_ratio=collocation_force_rms / weak_force_rms
        if weak_force_rms > 0.0
        else float("nan"),
        nfev=result_count(collocation_result, "function_evaluations", "nfev"),
        success=bool(collocation_result.success),
    )


def solve_case_samples(
    benchmark,
    *,
    case_key: str,
    signatures: list[dict[str, int]],
    args: SimpleNamespace,
    progress=None,
    task=None,
) -> list[GeometryRedistributionSample]:
    reference = build_pf_reference_case(case_key)
    samples: list[GeometryRedistributionSample] = []
    for config_label, signature in zip(CONFIG_LABELS, signatures, strict=True):
        if config_label not in DISPLAY_CONFIG_LABELS:
            continue
        current = f"{CASE_LABELS[case_key]} {config_label}"
        if progress is not None and task is not None:
            progress.update(task, current=current, phase="[cyan]solve[/]")
        samples.append(
            solve_geometry_redistribution_sample(
                benchmark,
                reference,
                case_key=case_key,
                config_label=config_label,
                signature=signature,
                args=args,
            )
        )
        if progress is not None and task is not None:
            progress.update(task, advance=1, current=current, phase="[cyan]solve[/]")
    return samples


def line_style(case_key: str, config_label: str) -> tuple[object, str, str | None, float]:
    colors = CASE_LINE_COLORS[case_key]
    styles = {
        "Low": ((0, (5, 1.6, 1.2, 1.6, 1.2, 1.6)), colors[-4], None, 0.0),
        "Medium": ("--", colors[-3], None, 0.0),
        "High": ("-", colors[-2], None, 0.0),
        "Ref": ("-", colors[-1], "x", 1.15 * EXTERNAL_RADIAL_MARKER_SIZE),
    }
    return styles.get(config_label, ("-", colors[-2], None, 0.0))


def plot_variational_collocation_overlay(
    ax: plt.Axes,
    item: GeometryRedistributionSample,
    *,
    case_key: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    _, collocation_color, _, _ = line_style(case_key, item.collocation.config_label)
    for level in SHAPE_SURFACE_LEVELS:
        linewidth = SHAPE_LINE_WIDTH if level < 1.0 - 1.0e-12 else SHAPE_BOUNDARY_LINE_WIDTH
        weak_r, weak_z = surface_at_psin_level(item.weak, level)
        coll_r, coll_z = surface_at_psin_level(item.collocation, level)
        if weak_r.size:
            ax.plot(
                weak_r,
                weak_z,
                color="#111111",
                lw=linewidth * SHAPE_TARGET_LINE_WIDTH_SCALE,
                ls=SHAPE_TARGET_LINESTYLE,
                alpha=1.0,
                zorder=2,
            )
        if coll_r.size:
            ax.plot(
                coll_r,
                coll_z,
                color=collocation_color,
                lw=linewidth,
                ls="-",
                alpha=1.0,
                zorder=3,
            )
    style_rz_axis(ax, xlim=xlim, ylim=ylim)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))


def add_group_title(
    fig: plt.Figure, axes: list[plt.Axes], title: str, *, pad: float = PANEL_TITLE_PAD
) -> None:
    boxes = [ax.get_position().frozen() for ax in axes]
    x0 = min(float(box.x0) for box in boxes)
    x1 = max(float(box.x1) for box in boxes)
    y1 = max(float(box.y1) for box in boxes)
    fig.text(
        0.5 * (x0 + x1),
        y1 + pad,
        title,
        ha="center",
        va="bottom",
        fontsize=scaled_font_size(TITLE_FONT_SIZE),
    )


def plot_overlay_grid(
    fig: plt.Figure,
    subspec,
    samples_by_case: dict[str, list[GeometryRedistributionSample]],
) -> list[plt.Axes]:
    axes: list[plt.Axes] = []
    case_keys = list(samples_by_case)
    grid = subspec.subgridspec(
        nrows=len(case_keys),
        ncols=len(DISPLAY_CONFIG_LABELS),
        wspace=0.05,
        hspace=0.20,
    )
    for row, case_key in enumerate(case_keys):
        items = [
            item
            for item in samples_by_case[case_key]
            if item.collocation.config_label in DISPLAY_CONFIG_LABELS
        ]
        sample_limits = [sample for item in items for sample in (item.weak, item.collocation)]
        xlim, ylim = rz_limits(sample_limits)
        by_label = {item.collocation.config_label: item for item in items}
        for col, config_label in enumerate(DISPLAY_CONFIG_LABELS):
            item = by_label[config_label]
            ax = fig.add_subplot(grid[row, col])
            axes.append(ax)
            plot_variational_collocation_overlay(ax, item, case_key=case_key, xlim=xlim, ylim=ylim)
            ax.set_xticks(HEATMAP_X_TICKS[case_key])
            ax.set_anchor("C")
            if row == 0:
                ax.set_title(
                    config_label,
                    fontsize=scaled_font_size(TITLE_FONT_SIZE),
                    fontweight="normal",
                )
            if col == 0:
                ax.set_ylabel(
                    f"{CASE_LABELS[case_key]}\nZ [m]",
                    fontsize=scaled_font_size(AXIS_LABEL_SIZE),
                )
            else:
                ax.set_yticklabels([])
            if row == len(case_keys) - 1:
                ax.set_xlabel("R [m]", fontsize=scaled_font_size(AXIS_LABEL_SIZE))
            else:
                ax.set_xlabel("")
    return axes


def compact_overlay_items(
    items: list[GeometryRedistributionSample],
) -> list[GeometryRedistributionSample]:
    by_label = {item.collocation.config_label: item for item in items}
    return [by_label[label] for label in COMPACT_CONFIG_LABELS]


def plot_var_coll_shape_panel(
    ax: plt.Axes,
    item: GeometryRedistributionSample,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    for level in SHAPE_SURFACE_LEVELS:
        linewidth = SHAPE_LINE_WIDTH if level < 1.0 - 1.0e-12 else SHAPE_BOUNDARY_LINE_WIDTH
        weak_r, weak_z = surface_at_psin_level(item.weak, level)
        collocation_r, collocation_z = surface_at_psin_level(item.collocation, level)
        if weak_r.size:
            ax.plot(
                weak_r,
                weak_z,
                color="#111111",
                lw=linewidth * SHAPE_TARGET_LINE_WIDTH_SCALE,
                ls=SHAPE_TARGET_LINESTYLE,
                alpha=1.0,
                zorder=2,
            )
        if collocation_r.size:
            ax.plot(
                collocation_r,
                collocation_z,
                color="#d62728",
                lw=linewidth * 1.15,
                ls="-",
                alpha=1.0,
                zorder=3,
            )
    style_rz_axis(ax, xlim=xlim, ylim=ylim)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))


def build_compact_overlay_figure(
    samples_by_case: dict[str, list[GeometryRedistributionSample]],
    *,
    collocation_weight: float,
) -> plt.Figure:
    """Build the 3x4 VAR--COLL magnetic-surface overlay companion figure."""

    apply_plot_style()
    case_keys = list(samples_by_case)
    fig = plt.figure(figsize=(COMPACT_FIGURE_WIDTH, COMPACT_ROW_HEIGHT * len(case_keys)))
    grid = fig.add_gridspec(
        nrows=len(case_keys),
        ncols=len(COMPACT_CONFIG_LABELS),
        left=COMPACT_LEFT,
        right=COMPACT_RIGHT,
        bottom=COMPACT_BOTTOM,
        top=COMPACT_TOP,
        wspace=COMPACT_WSPACE,
        hspace=COMPACT_HSPACE,
    )
    for row, case_key in enumerate(case_keys):
        items = compact_overlay_items(samples_by_case[case_key])
        samples_for_limits = [sample for item in items for sample in (item.weak, item.collocation)]
        xlim, ylim = rz_limits(samples_for_limits)
        for col, item in enumerate(items):
            ax = fig.add_subplot(grid[row, col])
            plot_var_coll_shape_panel(ax, item, xlim=xlim, ylim=ylim)
            ax.set_xticks(HEATMAP_X_TICKS[case_key])
            ax.set_anchor("C")
            ax.set_title(
                COMPACT_COLUMN_LABELS[col] if row == 0 else "",
                fontsize=scaled_font_size(TITLE_FONT_SIZE),
                fontweight="normal",
            )
            if col == 0:
                ax.set_ylabel(
                    f"{CASE_LABELS[case_key]}\nZ [m]",
                    fontsize=scaled_font_size(AXIS_LABEL_SIZE),
                )
            else:
                ax.set_yticklabels([])
            if row == len(case_keys) - 1:
                ax.set_xlabel("R [m]", fontsize=scaled_font_size(AXIS_LABEL_SIZE))
            else:
                ax.set_xlabel("")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#111111",
            lw=SHAPE_LINE_WIDTH * SHAPE_TARGET_LINE_WIDTH_SCALE,
            ls=SHAPE_TARGET_LINESTYLE,
            label="VAR",
        ),
        Line2D(
            [0],
            [0],
            color="#d62728",
            lw=SHAPE_LINE_WIDTH * 1.15,
            ls="-",
            label=rf"COLL ($\lambda={float(collocation_weight):g}$)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.995),
        ncol=2,
        frameon=False,
        fontsize=scaled_font_size(LEGEND_FONT_SIZE),
        handlelength=1.8,
        columnspacing=1.0,
    )
    return fig


def plot_shape_mismatch(
    ax: plt.Axes,
    samples_by_case: dict[str, list[GeometryRedistributionSample]],
) -> None:
    for case_key, items in samples_by_case.items():
        x = np.asarray([item.collocation.parameter_count for item in items], dtype=np.float64)
        y = np.asarray([item.shape_rms_over_a for item in items], dtype=np.float64)
        color = CASE_COLORS[case_key]
        ax.semilogy(
            x,
            np.maximum(y, 1.0e-12),
            color=color,
            lw=RADIAL_LINE_WIDTH,
            marker="o",
            markersize=3.2,
            label=CASE_LABELS[case_key],
        )
    ax.set_title(
        r"$\bf{(b)}$ Geometry mismatch",
        pad=PANEL_TITLE_PAD,
        fontsize=scaled_font_size(TITLE_FONT_SIZE),
        fontweight="normal",
    )
    ax.set_xlabel("")
    ax.set_ylabel(r"$\delta_{\rm shape}/a$", fontsize=scaled_font_size(AXIS_LABEL_SIZE))
    ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINE_WIDTH, linestyle=GRID_LINESTYLE)
    ax.legend(
        loc="best",
        frameon=False,
        fontsize=scaled_font_size(LEGEND_FONT_SIZE),
        handlelength=1.8,
        labelspacing=0.2,
    )
    apply_box_ticks(ax)


def residual_log_limits(
    samples_by_case: dict[str, list[GeometryRedistributionSample]],
) -> tuple[float, float]:
    _ = samples_by_case
    return RESIDUAL_LOG_YLIM


def plot_residual_redistribution(
    axes: list[plt.Axes],
    samples_by_case: dict[str, list[GeometryRedistributionSample]],
) -> None:
    ylimits = residual_log_limits(samples_by_case)
    case_keys = list(samples_by_case)
    for row, (ax, case_key) in enumerate(zip(axes, case_keys, strict=True)):
        items = [
            item
            for item in samples_by_case[case_key]
            if item.collocation.config_label in DISPLAY_CONFIG_LABELS
        ]
        for item in items:
            linestyle, color, marker, marker_size = line_style(
                case_key, item.collocation.config_label
            )
            ratio = item.force_radial_ratio
            y = np.full_like(ratio, np.nan, dtype=np.float64)
            mask = np.isfinite(ratio) & (ratio > 0.0)
            y[mask] = np.log10(ratio[mask])
            ax.plot(
                item.weak.rho,
                y,
                color=color,
                ls=linestyle,
                lw=RADIAL_LINE_WIDTH,
                marker=marker,
                markersize=marker_size,
                markevery=marker_indices(len(item.weak.rho)) if marker is not None else None,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.9 if marker == "x" else 0.6,
                label=sample_legend_label(item.collocation),
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(*ylimits)
        ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINE_WIDTH, linestyle=GRID_LINESTYLE)
        panel_letter = chr(ord("a") + row)
        ax.set_title(
            rf"$\bf{{({panel_letter})}}$ {CASE_LABELS[case_key]}",
            pad=PANEL_TITLE_PAD,
            fontsize=scaled_font_size(TITLE_FONT_SIZE),
            fontweight="normal",
        )
        ax.legend(
            loc="upper right",
            ncol=2,
            frameon=False,
            fontsize=scaled_font_size(LEGEND_FONT_SIZE),
            handlelength=1.7,
            columnspacing=0.7,
            labelspacing=0.15,
        )
        if row == 1:
            ax.set_ylabel(
                r"$\log_{10}\,E_{\rm coll}(\rho)/E_{\rm var}(\rho)$",
                fontsize=scaled_font_size(AXIS_LABEL_SIZE),
            )
        else:
            ax.set_ylabel("")
        if row == len(case_keys) - 1:
            ax.set_xlabel(r"$\rho$", fontsize=scaled_font_size(AXIS_LABEL_SIZE))
        else:
            ax.set_xlabel("")
            apply_box_ticks(ax, labelbottom=False)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        apply_box_ticks(ax)


def build_figure(samples_by_case: dict[str, list[GeometryRedistributionSample]]) -> plt.Figure:
    apply_plot_style()
    fig = plt.figure(figsize=(COMPACT_FIGURE_WIDTH, FIGURE_HEIGHT))
    grid = fig.add_gridspec(
        nrows=len(samples_by_case),
        ncols=1,
        left=0.18,
        right=0.97,
        bottom=0.12,
        top=0.94,
        hspace=0.34,
    )
    redistribution_axes = [fig.add_subplot(grid[idx]) for idx in range(len(samples_by_case))]
    plot_residual_redistribution(redistribution_axes, samples_by_case)
    return fig


def scalar_ratio(numerator: float, denominator: float) -> float:
    return (
        float(numerator) / float(denominator)
        if np.isfinite(numerator) and np.isfinite(denominator) and float(denominator) > 0.0
        else float("nan")
    )


def geometry_redistribution_table_rows(
    samples_by_case: dict[str, list[GeometryRedistributionSample]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for items in samples_by_case.values():
        for item in items:
            var_row = residual_norm_row(item.weak)
            collocation_row = residual_norm_row(item.collocation)
            rows.append(
                {
                    "case": collocation_row["case"],
                    "config": collocation_row["config"],
                    "params": collocation_row["params"],
                    "ratio_rms_all": scalar_ratio(
                        float(collocation_row["G_rms"]), float(var_row["G_rms"])
                    ),
                    "ratio_rms_interior": scalar_ratio(
                        float(collocation_row["G_rms_interior"]),
                        float(var_row["G_rms_interior"]),
                    ),
                    "ratio_rms_edge": scalar_ratio(
                        float(collocation_row["G_rms_edge"]), float(var_row["G_rms_edge"])
                    ),
                    "ratio_max": scalar_ratio(
                        float(collocation_row["G_max"]), float(var_row["G_max"])
                    ),
                    "ratio_e_geqdsk": float(item.external_shape_error_ratio),
                    "nfev": int(item.nfev),
                    "success": bool(item.success),
                }
            )
    return rows


def build_geometry_redistribution_latex_table(rows: list[dict[str, object]]) -> str:
    indent = "        "
    header = [
        "Case (Params)",
        r"$r_{\mathrm{RMS,all}}$",
        r"$r_{\mathrm{RMS},<0.8}$",
        r"$r_{\mathrm{RMS},\geq0.8}$",
        r"$r_{|\mathcal{G}_{\mathrm{std}}|_{\max}}$",
        r"$r_{E_{\mathrm{gqdsk}}}$",
    ]
    table_rows: list[list[str]] = []
    for row in rows:

        def fmt_ratio(key: str) -> str:
            value = float(row[key])
            return f"${value:.3f}$" if np.isfinite(value) else "--"

        table_rows.append(
            [
                str(row["case"])
                if row["params"] is None
                else f"{row['case']} ({int(row['params'])})",
                fmt_ratio("ratio_rms_all"),
                fmt_ratio("ratio_rms_interior"),
                fmt_ratio("ratio_rms_edge"),
                fmt_ratio("ratio_max"),
                fmt_ratio("ratio_e_geqdsk"),
            ]
        )

    widths = [max(len(row[i]) for row in [header, *table_rows]) for i in range(len(header))]

    def fmt(row: list[str]) -> str:
        return " & ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + r" \\"

    lines = [r"\hline", fmt(header), r"\hline", *(fmt(row) for row in table_rows), r"\hline"]
    return "\n".join(indent + line for line in lines)


def write_text(path: str, text: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)
        file.write("\n")


def print_summary(
    samples_by_case: dict[str, list[GeometryRedistributionSample]], args: SimpleNamespace
) -> None:
    table = make_script_table(
        "geometry redistribution summary",
        [
            ("case", "left"),
            ("config", "left"),
            ("params", "right"),
            ("shape rms/a", "right"),
            ("shape max/a", "right"),
            ("E_gqdsk ratio", "right"),
            ("force ratio", "right"),
            ("nfev", "right"),
            ("status", "left"),
        ],
    )
    for case_key, items in samples_by_case.items():
        for item in items:
            table.add_row(
                CASE_LABELS[case_key],
                item.collocation.config_label,
                str(int(item.collocation.parameter_count)),
                format_script_sci(item.shape_rms_over_a),
                format_script_sci(item.shape_max_over_a),
                format_script_sci(item.external_shape_error_ratio),
                format_script_sci(item.force_rms_ratio),
                str(int(item.nfev)),
                "passed" if item.success else "failed",
            )
    print_script_table(SCRIPT_CONSOLE, table)


def main() -> None:
    args = SimpleNamespace(
        backend=BACKEND,
        case=CASE_KEYS_TO_RUN,
        solve_nr=DEFAULT_SOLVE_NR,
        solve_nt=DEFAULT_SOLVE_NT,
        weak_repeat_count=DEFAULT_REPEAT_COUNT,
        initial_solve_timeout_s=INITIAL_SOLVE_TIMEOUT_S,
        collocation_method=DEFAULT_COLLOCATION_METHOD,
        collocation_weight=DEFAULT_COLLOCATION_WEIGHT,
        collocation_repeat_count=DEFAULT_REPEAT_COUNT,
        max_residual=MAX_RESIDUAL,
        max_nfev=DEFAULT_MAX_NFEV,
        save_png=PNG_PATH,
        save_pdf=PDF_PATH,
        save_compact_png=COMPACT_PNG_PATH,
        save_table=SAVE_TABLE_PATH,
        no_print_table=not PRINT_TABLE,
    )
    selected_cases = CASE_KEYS if args.case == "all" else (args.case,)
    signatures_by_case = selected_signature_map(selected_cases)
    benchmark = load_pf_benchmark(args.backend)

    print_script_config(
        SCRIPT_CONSOLE,
        "appendix b: geometry redistribution",
        (
            ("backend", args.backend),
            ("cases", len(selected_cases)),
            ("configs", ", ".join(DISPLAY_CONFIG_LABELS)),
            ("grid", f"{args.solve_nr}x{args.solve_nt}"),
            ("method", args.collocation_method),
        )
    )

    solve_count = sum(
        1
        for case_key in selected_cases
        for config_label, _signature in zip(
            CONFIG_LABELS, signatures_by_case[case_key], strict=True
        )
        if config_label in DISPLAY_CONFIG_LABELS
    )
    with script_progress(SCRIPT_CONSOLE) as progress:
        total = solve_count + 2 + (1 if args.save_compact_png else 0)
        task = progress.add_task("", total=total, current="solve samples", phase="[cyan]solve[/]")
        samples_by_case: dict[str, list[GeometryRedistributionSample]] = {}
        for case_key in selected_cases:
            samples_by_case[case_key] = solve_case_samples(
                benchmark,
                case_key=case_key,
                signatures=signatures_by_case[case_key],
                args=args,
                progress=progress,
                task=task,
            )

        progress.update(task, current="main figure", phase="[cyan]run[/]")
        fig = build_figure(samples_by_case)
        saved_paths = save_figure_outputs(
            fig,
            png_path=args.save_png,
            pdf_path=args.save_pdf,
            dpi=SAVE_DPI,
            transparent=SAVE_TRANSPARENT,
        )
        plt.close(fig)
        progress.update(task, advance=1, current="compact figure", phase="[cyan]run[/]")
        compact_saved_paths: list[str] = []
        if args.save_compact_png:
            compact_fig = build_compact_overlay_figure(
                samples_by_case,
                collocation_weight=float(args.collocation_weight),
            )
            compact_saved_paths = save_figure_outputs(
                compact_fig,
                png_path=args.save_compact_png,
                pdf_path=None,
                dpi=SAVE_DPI,
                transparent=SAVE_TRANSPARENT,
            )
            plt.close(compact_fig)
            progress.update(task, advance=1, current="table", phase="[cyan]run[/]")

        table_body = build_geometry_redistribution_latex_table(
            geometry_redistribution_table_rows(samples_by_case)
        )
        if args.save_table:
            write_text(args.save_table, table_body)
        progress.update(task, advance=1, current="table", phase="[green]done[/]")

    if not args.no_print_table:
        SCRIPT_CONSOLE.print(table_body, markup=False)
    print_summary(samples_by_case, args)
    output_rows = [
        ("Appendix B figure", path, "VAR/collocation force redistribution")
        for path in saved_paths
    ]
    output_rows.extend(
        ("Appendix B compact", path, "Compact geometry redistribution overlay")
        for path in compact_saved_paths
    )
    if args.save_table:
        output_rows.append(("Appendix B table", args.save_table, "LaTeX table body"))
    print_output_table(SCRIPT_CONSOLE, output_rows)


if __name__ == "__main__":
    main()
