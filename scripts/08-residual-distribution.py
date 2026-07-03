"""Plot variational and collocation residual distributions.

The script compares full and reduced reconstructions against reference GEQDSK
cases, then writes the residual distribution figures and associated LaTeX
summary tables used by the manuscript.
"""

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from config import (
    AXIS_LABEL_SIZE,
    CASE_KEYS,
    CASE_LABELS,
    CASE_LINE_COLORS,
    CASE_REFERENCE_EQUILIBRIUM_JSONS,
    CASE_REFERENCE_GFILES,
    CONFIG_LABELS,
    DOUBLE_COLUMN_WIDTH,
    FIXED_DECIMALS,
    LEGEND_FONT_SIZE,
    PLOT_TICK_BOTTOM,
    PLOT_TICK_DIRECTION,
    PLOT_TICK_LEFT,
    PLOT_TICK_RIGHT,
    PLOT_TICK_TOP,
    SAVE_DPI,
    SAVE_TRANSPARENT,
    SCIENTIFIC_DECIMALS,
    SCRIPT_CONSOLE,
    SINGLE_COLUMN_WIDTH,
    TICK_LABEL_SIZE,
    TITLE_FONT_SIZE,
    apply_plot_style,
    data_path,
    figure_path,
    format_script_sci,
    load_equilibrium_json,
    load_reduced_equilibrium_manifest,
    load_reference_equilibrium_manifest,
    make_script_table,
    manifest_entry,
    metadata_float,
    metadata_int,
    normalize_signature,
    print_output_table,
    print_script_config,
    print_script_table,
    read_geqdsk,
    reduced_equilibrium_json_path,
    reference_manifest_entry,
    save_figure_outputs,
    scaled_font_size,
    script_progress,
    signature_from_metadata,
)
from matplotlib import colors, ticker
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator

PNG_PATH = figure_path("08-residual-distribution.png")
PDF_PATH = None
COMPACT_PNG_PATH = figure_path("08-residual-distribution-compact.png")
DEFAULT_CACHE_PATH = data_path("09-residual-cache.npz")
RESIDUAL_CACHE_VERSION = 10
CASE_KEYS_TO_RUN = "all"

EXTERNAL_REFERENCE_LABELS = {
    "solovev": "G-EQDSK",
    "chease": "G-EQDSK",
    "efit": "G-EQDSK",
}
CONFIG_CMAP = "magma"
LOG_RESIDUAL_FLOOR = -5.0
LOG_RESIDUAL_CEIL = 0.0
# Manuscript diagnostics use the standard cylindrical strong-form residual.
# Set this to False to reproduce the transformed coordinate-density residual.
USE_STANDARD_GS_RESIDUAL = True
STANDARD_GS_JDIVER_FLOOR = 1.0e-30

FIGURE_WIDTH = DOUBLE_COLUMN_WIDTH
ROW_HEIGHT = 2.45
FIGURE_MAX_HEIGHT = 5
FIGURE_LEFT = 0.06
FIGURE_RIGHT = 0.96
FIGURE_BOTTOM = 0.08
FIGURE_TOP = 0.90
FIGURE_GRID_WSPACE = -0.05
FIGURE_GRID_HSPACE = 0.2
FIGURE_GRID_WIDTH_RATIOS = (0.6, 0.6, 0.6, 0.6, 0.6, -0.18, 0.05, 0.42, 1.25)
HEATMAP_GRID_COLS = (0, 1, 2, 3, 4)
COLORBAR_GRID_COL = 6
RADIAL_GRID_COL = 8
PANEL_TITLE_PAD = 6
HEATMAP_COLUMN_GAP = 0.018
COLORBAR_MAX_WIDTH = 0.012
COLORBAR_HEIGHT_FRACTION = 0.64
COLORBAR_LEFT_PAD = 0.030
RADIAL_LEFT_PAD = 0.160
HEATMAP_LEVEL_COUNT = 129
HEATMAP_X_TICK_BINS = 2
HEATMAP_Y_TICK_BINS = 4
HEATMAP_X_TICKS = {
    "solovev": (4.0, 8.0),
    "chease": (0.5, 1.5),
    "efit": (1.0, 2.0),
}
BOUNDARY_LINE_WIDTH = 0.8
RADIAL_LINE_WIDTH = 1.4
EXTERNAL_RADIAL_LINE_WIDTH = 0.75 * RADIAL_LINE_WIDTH
EXTERNAL_RADIAL_MARKER_SIZE = 3.0 * RADIAL_LINE_WIDTH
EXTERNAL_RADIAL_MARKER_COUNT = 15
EXTERNAL_RADIAL_ALPHA = 1.0
RADIAL_YMIN = 1.0e-8
RADIAL_YMAX = 100.0
GRID_ALPHA = 0.25
GRID_LINE_WIDTH = 0.5
GRID_LINESTYLE = "-"
COMPACT_COLUMN_LABELS = ("Low", "Medium", "High", "Ref")
COMPACT_CONFIG_LABELS = ("Low", "Medium", "High", "Ref")
COMPACT_FIGURE_WIDTH = SINGLE_COLUMN_WIDTH
COMPACT_ROW_HEIGHT = 1.55
COMPACT_LEFT = 0.14
COMPACT_RIGHT = 0.98
COMPACT_BOTTOM = 0.08
COMPACT_TOP = 0.88
COMPACT_WSPACE = 0.04
COMPACT_HSPACE = 0.18
SHAPE_SURFACE_LEVELS = (0.2, 0.4, 0.6, 0.8, 1.0)
COMPACT_REFERENCE_COLOR = "#111111"
COMPACT_VEQ_COLOR = "#d62728"
SHAPE_TARGET_LINESTYLE = (0, (4.0, 2.0))
SHAPE_LINE_WIDTH = 1.0
SHAPE_BOUNDARY_LINE_WIDTH = 1.35
SHAPE_TARGET_LINE_WIDTH_SCALE = 1.5


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


def active_fourier_order(signature: dict[str, int]) -> int:
    order = 0
    for name, length in signature.items():
        if int(length) <= 0:
            continue
        if len(name) >= 2 and name[0] in {"c", "s"} and name[1:].isdigit():
            order = max(order, int(name[1:]))
    return order


def active_radial_length(signature: dict[str, int]) -> int:
    return max((int(length) for length in signature.values() if int(length) > 0), default=0)


def format_config_title(sample: ResidualSample) -> str:
    return ""


def sample_legend_label(sample: ResidualSample) -> str:
    if sample.parameter_count is None:
        return str(sample.config_label)
    return f"{sample.config_label} ({int(sample.parameter_count)})"


def marker_indices(length: int, count: int = EXTERNAL_RADIAL_MARKER_COUNT) -> list[int]:
    n = int(length)
    if n <= 0:
        return []
    if n <= int(count):
        return list(range(n))
    return np.unique(np.linspace(0, n - 1, int(count), dtype=int)).tolist()


def residual_latex_symbol() -> str:
    if USE_STANDARD_GS_RESIDUAL:
        return r"\mathcal{G}_{\mathrm{std}}"
    return r"\mathcal{G}"


def residual_tilde_latex_symbol() -> str:
    if USE_STANDARD_GS_RESIDUAL:
        return r"\tilde{\mathcal{G}}_{\mathrm{std}}"
    return r"\tilde{\mathcal{G}}"


def diagnostic_residual_from_equilibrium(equilibrium) -> np.ndarray:
    G = np.asarray(equilibrium.G, dtype=np.float64)
    if not USE_STANDARD_GS_RESIDUAL:
        return G

    JdivR = np.asarray(equilibrium.JdivR, dtype=np.float64)
    G_std = np.full_like(G, np.nan, dtype=np.float64)
    np.divide(
        G,
        JdivR,
        out=G_std,
        where=np.isfinite(G) & np.isfinite(JdivR) & (np.abs(JdivR) > STANDARD_GS_JDIVER_FLOOR),
    )
    return G_std


def geqdsk_standard_balance_grid(case_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the standard R-Z GS balance on the exported GEQDSK grid.

    The returned field is

        Delta*psi + mu0 R^2 p'(psi) + F F'(psi)

    on the second-order finite-difference interior of the rectangular grid.
    It is not yet multiplied by the flux-coordinate factor J/R.
    """

    geqdsk = read_geqdsk(CASE_REFERENCE_GFILES[case_key])
    psi = np.asarray(geqdsk.psi, dtype=np.float64)
    if psi.shape != (int(geqdsk.NR), int(geqdsk.NZ)):
        raise ValueError(f"Unexpected GEQDSK psi shape for {case_key}: {psi.shape}")
    if int(geqdsk.NR) < 3 or int(geqdsk.NZ) < 3:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, np.empty((0, 0), dtype=np.float64)

    R = np.linspace(float(geqdsk.Rmin), float(geqdsk.Rmax), int(geqdsk.NR), dtype=np.float64)
    Z = np.linspace(float(geqdsk.Zmin), float(geqdsk.Zmax), int(geqdsk.NZ), dtype=np.float64)
    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])
    if dR == 0.0 or dZ == 0.0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, np.empty((0, 0), dtype=np.float64)

    psi_center = psi[1:-1, 1:-1]
    psi_R = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dR)
    psi_RR = (psi[2:, 1:-1] - 2.0 * psi_center + psi[:-2, 1:-1]) / (dR * dR)
    psi_ZZ = (psi[1:-1, 2:] - 2.0 * psi_center + psi[1:-1, :-2]) / (dZ * dZ)
    R_center = R[1:-1, None]

    psi_span = float(geqdsk.psi_bound) - float(geqdsk.psi_axis)
    if psi_span == 0.0:
        return R[1:-1], Z[1:-1], np.full_like(psi_center, np.nan, dtype=np.float64)
    psin = (psi_center - float(geqdsk.psi_axis)) / psi_span
    source_axis = np.linspace(0.0, 1.0, len(geqdsk.P_psi), dtype=np.float64)
    p_psi = np.interp(
        psin.ravel(), source_axis, np.asarray(geqdsk.P_psi, dtype=np.float64)
    ).reshape(psin.shape)
    ff_psi = np.interp(
        psin.ravel(), source_axis, np.asarray(geqdsk.FF_psi, dtype=np.float64)
    ).reshape(psin.shape)

    gs_balance = (
        psi_RR - psi_R / R_center + psi_ZZ + (4.0e-7 * np.pi) * R_center**2 * p_psi + ff_psi
    )
    return R[1:-1], Z[1:-1], gs_balance


def external_reference_residual_sample(reference_equilibrium, *, case_key: str) -> ResidualSample:
    """Build the GEQDSK-file residual sample used in Figure 08.

    For the analytic D-shaped file this is the finite-difference residual of the
    generated analytic exchange grid.  For the GEQDSK-based cases, the standard
    R-Z balance residual is evaluated on the exported grid, interpolated to the
    same flux-coordinate nodes used by the VEQ diagnostics.  The manuscript
    reports this standard balance directly; the switch above can instead
    multiply by the mapped J/R factor to recover the coordinate-form residual
    density ``\\mathcal{G}`` as in the VEQ rows.
    """

    equilibrium = reference_equilibrium
    R_axis, Z_axis, gs_balance = geqdsk_standard_balance_grid(case_key)
    equilibrium_R = np.asarray(equilibrium.R, dtype=np.float64)
    equilibrium_Z = np.asarray(equilibrium.Z, dtype=np.float64)
    if R_axis.size == 0 or Z_axis.size == 0:
        G = np.full_like(equilibrium_R, np.nan)
    else:
        interpolator = RegularGridInterpolator(
            (R_axis, Z_axis),
            gs_balance,
            bounds_error=False,
            fill_value=np.nan,
        )
        points = np.column_stack(
            (
                equilibrium_R.ravel(),
                equilibrium_Z.ravel(),
            )
        )
        gs_on_flux_grid = interpolator(points).reshape(equilibrium_R.shape)
        if USE_STANDARD_GS_RESIDUAL:
            G = gs_on_flux_grid
        else:
            G = np.asarray(equilibrium.JdivR, dtype=np.float64) * gs_on_flux_grid
    radial_rms = np.sqrt(np.nanmean(G * G, axis=1))
    return ResidualSample(
        case_key=case_key,
        config_label=EXTERNAL_REFERENCE_LABELS[case_key],
        signature={},
        parameter_count=None,
        elapsed_ms=float("nan"),
        solver_residual_norm=float("nan"),
        rho=np.asarray(equilibrium.rho, dtype=np.float64),
        psin=np.asarray(equilibrium.psin, dtype=np.float64),
        R=equilibrium_R,
        Z=equilibrium_Z,
        G=G,
        radial_rms=radial_rms,
    )


def sample_from_equilibrium(
    equilibrium,
    *,
    case_key: str,
    config_label: str,
    signature: dict[str, int] | None = None,
    parameter_count: int | None = None,
    elapsed_ms: float = float("nan"),
    solver_residual_norm: float | None = None,
) -> ResidualSample:
    G = diagnostic_residual_from_equilibrium(equilibrium)
    radial_rms = np.sqrt(np.nanmean(G * G, axis=1))
    if solver_residual_norm is None or not np.isfinite(float(solver_residual_norm)):
        solver_residual_norm = _rms(G)
    return ResidualSample(
        case_key=case_key,
        config_label=config_label,
        signature={} if signature is None else normalize_signature(signature),
        parameter_count=parameter_count,
        elapsed_ms=float(elapsed_ms),
        solver_residual_norm=float(solver_residual_norm),
        rho=np.asarray(equilibrium.rho, dtype=np.float64),
        psin=np.asarray(equilibrium.psin, dtype=np.float64),
        R=np.asarray(equilibrium.R, dtype=np.float64),
        Z=np.asarray(equilibrium.Z, dtype=np.float64),
        G=G,
        radial_rms=radial_rms,
    )


def required_metadata_float(entry: dict[str, object], key: str, *, source: str) -> float:
    value = metadata_float(entry, key, None)
    if value is None:
        raise ValueError(f"Missing finite `{key}` in {source}. Regenerate the upstream metadata.")
    return float(value)


def load_case_samples_from_equilibrium_jsons(
    case_keys: tuple[str, ...],
) -> dict[str, list[ResidualSample]]:
    """Load Figure 08 inputs directly from Figure 06/07 data products.

    Inputs are the three GEQDSK files, the three Figure 06 reference
    equilibrium JSON files, and the 3x3 representative reduced equilibrium
    JSON files listed by the reduced-equilibrium manifest.
    """

    reduced_manifest = load_reduced_equilibrium_manifest()
    reference_manifest = load_reference_equilibrium_manifest()
    case_samples: dict[str, list[ResidualSample]] = {}
    for case_key in case_keys:
        samples: list[ResidualSample] = []
        for config_label in CONFIG_LABELS[:-1]:
            metadata = manifest_entry(reduced_manifest, case_key, config_label)
            default_path = reduced_equilibrium_json_path(case_key, config_label)
            path = str(metadata.get("path", default_path))
            try:
                equilibrium = load_equilibrium_json(path)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Missing {CASE_LABELS[case_key]} {config_label} "
                    f"reduced equilibrium JSON at {path}. "
                    "Generate the reduced-equilibrium manifest before running this figure."
                ) from exc
            samples.append(
                sample_from_equilibrium(
                    equilibrium,
                    case_key=case_key,
                    config_label=config_label,
                    signature=signature_from_metadata(metadata),
                    parameter_count=metadata_int(metadata, "parameter_count"),
                    elapsed_ms=metadata_float(metadata, "elapsed_ms", float("nan")),
                    solver_residual_norm=required_metadata_float(
                        metadata,
                        "residual_norm_final",
                        source=f"{CASE_LABELS[case_key]} {config_label} reduced manifest entry",
                    ),
                )
            )

        reference_metadata = reference_manifest_entry(reference_manifest, case_key)
        reference_path = str(
            reference_metadata.get("path", CASE_REFERENCE_EQUILIBRIUM_JSONS[case_key])
        )
        reference_equilibrium = load_equilibrium_json(reference_path)
        samples.append(
            sample_from_equilibrium(
                reference_equilibrium,
                case_key=case_key,
                config_label="Ref",
                signature=signature_from_metadata(reference_metadata),
                parameter_count=metadata_int(reference_metadata, "parameter_count"),
                elapsed_ms=metadata_float(reference_metadata, "elapsed_ms", float("nan")),
                solver_residual_norm=required_metadata_float(
                    reference_metadata,
                    "residual_norm_final",
                    source=f"{CASE_LABELS[case_key]} Ref reference manifest entry",
                ),
            )
        )
        samples.append(external_reference_residual_sample(reference_equilibrium, case_key=case_key))
        case_samples[case_key] = samples
    return case_samples


def residual_scale(samples: list[ResidualSample]) -> float:
    values = np.concatenate(
        [np.ravel(np.abs(sample.G[np.isfinite(sample.G)])) for sample in samples]
    )
    if values.size == 0:
        return 1.0
    scale = float(np.nanmax(values))
    return max(scale, 1.0e-30)


def log_residual_field(sample: ResidualSample, *, scale: float) -> np.ndarray:
    normalized = np.abs(sample.G) / max(float(scale), 1.0e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_field = np.log10(normalized + 10.0 ** (LOG_RESIDUAL_FLOOR - 1.0))
    return np.clip(log_field, LOG_RESIDUAL_FLOOR, LOG_RESIDUAL_CEIL)


def _finite_abs_values(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = np.abs(finite[np.isfinite(finite)])
    return finite


def _rms(values: np.ndarray) -> float:
    finite = _finite_abs_values(values)
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite * finite)))


def _max(values: np.ndarray) -> float:
    finite = _finite_abs_values(values)
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def _region_rms(
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
    return _rms(values[mask, ...])


def residual_norm_row(sample: ResidualSample) -> dict[str, object]:
    residual_rms = _rms(sample.G)
    residual_max = _max(sample.G)
    residual_rms_interior = _region_rms(sample, upper=0.8)
    residual_rms_edge = _region_rms(sample, lower=0.8)
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
        # The assembled weak/projected residual is the nonlinear system that VEQ
        # actually solves.  We report the final assembled norm directly instead
        # of normalizing by the zero-coefficient initial residual, because some
        # strongly shaped cases have a singular or near-singular zero geometry
        # that makes the initial residual scale non-physical.
        "epsilon_proj": float(sample.solver_residual_norm),
        "G_rms": residual_rms,
        "G_rms_interior": residual_rms_interior,
        "G_rms_edge": residual_rms_edge,
        "G_max": residual_max,
        "solver_residual_norm": float(sample.solver_residual_norm),
    }


def residual_norm_rows(case_samples: dict[str, list[ResidualSample]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case_key in case_samples:
        rows.extend(residual_norm_row(sample) for sample in case_samples[case_key])
    return rows


def _format_scientific(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(numeric):
        return "--"
    return f"{numeric:.{SCIENTIFIC_DECIMALS}e}"


def _format_tex_number(
    value: object,
    *,
    fixed_precision: int = FIXED_DECIMALS,
    scientific_precision: int = SCIENTIFIC_DECIMALS,
    fixed_min_magnitude: float = 1.0e-2,
) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(numeric):
        return "--"
    if numeric == 0.0:
        return f"${numeric:.{fixed_precision}f}$"
    magnitude = abs(numeric)
    if float(fixed_min_magnitude) <= magnitude < 1.0e3:
        return f"${numeric:.{fixed_precision}f}$"
    exponent = int(np.floor(np.log10(magnitude)))
    mantissa = numeric / (10.0**exponent)
    return rf"${mantissa:.{scientific_precision}f}\times 10^{{{exponent}}}$"


def build_residual_norm_latex_table(rows: list[dict[str, object]]) -> str:
    indent = "              "
    residual_symbol = residual_latex_symbol()
    header = [
        "Case",
        r"$\varepsilon_{\mathrm{proj}}$",
        rf"$\mathrm{{RMS}}_{{\mathrm{{all}}}}({residual_symbol})$",
        rf"$\mathrm{{RMS}}_{{<0.8}}({residual_symbol})$",
        rf"$\mathrm{{RMS}}_{{\geq0.8}}({residual_symbol})$",
        rf"$|{residual_symbol}|_{{\max}}$",
    ]
    table_rows: list[list[str]] = []
    for row in rows:
        table_rows.append(
            [
                str(row["case_params"]),
                _format_tex_number(row["epsilon_proj"]),
                _format_tex_number(row["G_rms"], fixed_min_magnitude=1.0e-1),
                _format_tex_number(row["G_rms_interior"], fixed_min_magnitude=1.0e-1),
                _format_tex_number(row["G_rms_edge"], fixed_min_magnitude=1.0e-1),
                _format_tex_number(row["G_max"]),
            ]
        )
    column_widths = [
        max(len(row[column_index]) for row in [header, *table_rows])
        for column_index in range(len(header))
    ]

    def format_row(row: list[str]) -> str:
        return (
            " & ".join(cell.ljust(column_widths[index]) for index, cell in enumerate(row)) + r" \\"
        )

    return "\n".join(
        indent + line
        for line in [
            r"\hline",
            format_row(header),
            r"\hline",
            *(format_row(row) for row in table_rows),
            r"\hline",
        ]
    )


def print_residual_norm_latex_table(rows: list[dict[str, object]]) -> None:
    print(build_residual_norm_latex_table(rows))


def print_residual_norm_summary(rows: list[dict[str, object]]) -> None:
    residual_symbol = "G_std" if USE_STANDARD_GS_RESIDUAL else "G"
    table = make_script_table(
        "projected and pointwise Grad-Shafranov residual diagnostics",
        [
            ("Case", "left"),
            ("epsilon_proj", "right"),
            (f"RMS_all({residual_symbol})", "right"),
            (f"RMS_<0.8({residual_symbol})", "right"),
            (f"RMS_>=0.8({residual_symbol})", "right"),
            (f"|{residual_symbol}|_max", "right"),
        ],
    )
    for row in rows:
        table.add_row(
            str(row["case_params"]),
            format_script_sci(row["epsilon_proj"]),
            format_script_sci(row["G_rms"]),
            format_script_sci(row["G_rms_interior"]),
            format_script_sci(row["G_rms_edge"]),
            format_script_sci(row["G_max"]),
        )
    print_script_table(SCRIPT_CONSOLE, table)


def periodic_surface_arrays(
    sample: ResidualSample, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.hstack([sample.R, sample.R[:, :1]]),
        np.hstack([sample.Z, sample.Z[:, :1]]),
        np.hstack([values, values[:, :1]]),
    )


def rz_limits(
    samples: list[ResidualSample],
) -> tuple[tuple[float, float], tuple[float, float]]:
    R_all = np.concatenate([np.ravel(sample.R) for sample in samples])
    Z_all = np.concatenate([np.ravel(sample.Z) for sample in samples])
    rmin, rmax = float(np.nanmin(R_all)), float(np.nanmax(R_all))
    zmin, zmax = float(np.nanmin(Z_all)), float(np.nanmax(Z_all))
    rpad = max(0.06 * (rmax - rmin), 1.0e-6)
    zpad = max(0.06 * (zmax - zmin), 1.0e-6)
    return (rmin - rpad, rmax + rpad), (zmin - zpad, zmax + zpad)


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


def style_rz_axis(ax: plt.Axes, *, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    apply_box_ticks(ax)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=HEATMAP_X_TICK_BINS))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=HEATMAP_Y_TICK_BINS))


def plot_heatmap_panel(
    ax: plt.Axes,
    sample: ResidualSample,
    *,
    scale: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> matplotlib.contour.QuadContourSet:
    log_field = log_residual_field(sample, scale=scale)
    R_plot, Z_plot, G_plot = periodic_surface_arrays(sample, log_field)
    levels = np.linspace(LOG_RESIDUAL_FLOOR, LOG_RESIDUAL_CEIL, HEATMAP_LEVEL_COUNT)
    contour = ax.contourf(
        R_plot,
        Z_plot,
        G_plot,
        levels=levels,
        cmap=CONFIG_CMAP,
        norm=colors.Normalize(vmin=LOG_RESIDUAL_FLOOR, vmax=LOG_RESIDUAL_CEIL),
        extend="min",
    )
    ax.plot(
        np.r_[sample.R[-1], sample.R[-1, 0]],
        np.r_[sample.Z[-1], sample.Z[-1, 0]],
        color="white",
        lw=BOUNDARY_LINE_WIDTH,
        alpha=0.95,
    )
    ax.set_title(
        format_config_title(sample),
        fontsize=scaled_font_size(TITLE_FONT_SIZE),
        fontweight="normal",
    )
    style_rz_axis(ax, xlim=xlim, ylim=ylim)
    return contour


def add_centered_panel_title(
    fig: plt.Figure,
    left_ax: plt.Axes,
    right_ax: plt.Axes,
    title: str,
    *,
    pad: float,
) -> None:
    left_bbox = left_ax.get_position()
    right_bbox = right_ax.get_position()
    center_x = 0.5 * (left_bbox.x0 + right_bbox.x1)
    y = max(left_bbox.y1, right_bbox.y1) + (pad / 72.0) / fig.get_figheight()
    fig.text(
        center_x,
        y,
        title,
        ha="center",
        va="bottom",
        fontsize=scaled_font_size(TITLE_FONT_SIZE),
    )


def align_heatmap_columns(
    fig: plt.Figure,
    heatmap_rows: list[list[plt.Axes]],
    *,
    gap: float = HEATMAP_COLUMN_GAP,
) -> float:
    """Align the four heatmap columns while preserving equal-aspect panel boxes."""
    if not heatmap_rows:
        return 0.0
    bboxes_by_row = [[ax.get_position().frozen() for ax in row] for row in heatmap_rows]
    ncols = len(bboxes_by_row[0])
    column_widths = [max(float(row[col].width) for row in bboxes_by_row) for col in range(ncols)]
    x0 = min(float(row[0].x0) for row in bboxes_by_row)
    centers: list[float] = []
    x = x0
    for width in column_widths:
        centers.append(x + 0.5 * width)
        x += width + float(gap)

    for axes, row_bboxes in zip(heatmap_rows, bboxes_by_row, strict=True):
        for col, (ax, bbox) in enumerate(zip(axes, row_bboxes, strict=True)):
            ax.set_position([centers[col] - 0.5 * bbox.width, bbox.y0, bbox.width, bbox.height])
    return centers[-1] + 0.5 * column_widths[-1]


def plot_radial_panel(
    ax: plt.Axes,
    samples: list[ResidualSample],
    *,
    scale: float,
    case_key: str,
    show_xlabel: bool,
    legend_loc: str = "upper left",
) -> None:
    style_by_label = {
        "Ref": (
            "-",
            CASE_LINE_COLORS[case_key][-1],
            RADIAL_LINE_WIDTH,
            1.0,
            "x",
            1.15 * EXTERNAL_RADIAL_MARKER_SIZE,
        ),
        "High": ("-", CASE_LINE_COLORS[case_key][-2], RADIAL_LINE_WIDTH, 1.0, None, 0.0),
        "Medium": ("--", CASE_LINE_COLORS[case_key][-3], RADIAL_LINE_WIDTH, 1.0, None, 0.0),
        "Low": (
            (0, (5, 1.6, 1.2, 1.6, 1.2, 1.6)),
            CASE_LINE_COLORS[case_key][-4],
            RADIAL_LINE_WIDTH,
            1.0,
            None,
            0.0,
        ),
        "G-EQDSK": (
            "-",
            "#000000",
            EXTERNAL_RADIAL_LINE_WIDTH,
            EXTERNAL_RADIAL_ALPHA,
            "o",
            EXTERNAL_RADIAL_MARKER_SIZE,
        ),
    }
    zorder_by_label = {
        "G-EQDSK": 1.0,
        "Ref": 1.5,
        "Low": 2.0,
        "Medium": 2.5,
        "High": 3.0,
    }
    for sample in reversed(samples):
        _ = scale
        linestyle, line_color, line_width, line_alpha, marker, marker_size = style_by_label.get(
            sample.config_label,
            ("-", "#111111", RADIAL_LINE_WIDTH, 1.0, None, 0.0),
        )
        y = sample.radial_rms
        ax.semilogy(
            sample.rho,
            np.maximum(y, 1.0e-12),
            color=line_color,
            ls=linestyle,
            lw=line_width,
            alpha=line_alpha,
            marker=marker,
            markersize=marker_size,
            markevery=marker_indices(len(sample.rho)) if marker is not None else None,
            markerfacecolor=line_color,
            markeredgecolor=line_color,
            markeredgewidth=0.9 if marker == "x" else 0.6,
            zorder=zorder_by_label.get(sample.config_label, 2.0),
            label=sample_legend_label(sample),
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(RADIAL_YMIN, RADIAL_YMAX)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylabel(
        rf"$\varepsilon_{{{residual_latex_symbol()}}}(\rho)$",
        fontsize=scaled_font_size(AXIS_LABEL_SIZE),
    )
    if show_xlabel:
        ax.set_xlabel(r"$\rho$", fontsize=scaled_font_size(AXIS_LABEL_SIZE))
    else:
        ax.set_xlabel("")
    ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINE_WIDTH, linestyle=GRID_LINESTYLE)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc=legend_loc,
        ncol=3,
        frameon=False,
        fontsize=scaled_font_size(LEGEND_FONT_SIZE),
        handlelength=1.7,
        columnspacing=0.7,
        labelspacing=0.2,
    )
    apply_box_ticks(ax, labelbottom=show_xlabel)


def build_figure(case_samples: dict[str, list[ResidualSample]]) -> plt.Figure:
    case_keys = list(case_samples)
    figure_height = min(max(ROW_HEIGHT * len(case_keys), ROW_HEIGHT), FIGURE_MAX_HEIGHT)
    fig = plt.figure(figsize=(FIGURE_WIDTH, figure_height))
    grid = fig.add_gridspec(
        nrows=len(case_keys),
        ncols=len(FIGURE_GRID_WIDTH_RATIOS),
        width_ratios=FIGURE_GRID_WIDTH_RATIOS,
        left=FIGURE_LEFT,
        right=FIGURE_RIGHT,
        bottom=FIGURE_BOTTOM,
        top=FIGURE_TOP,
        wspace=FIGURE_GRID_WSPACE,
        hspace=FIGURE_GRID_HSPACE,
    )
    mappable = None
    max_heatmap_right = 0.0
    heatmap_rows: list[list[plt.Axes]] = []
    radial_axes: list[plt.Axes] = []
    for row, case_key in enumerate(case_keys):
        samples = case_samples[case_key]
        scale = residual_scale(samples)
        xlim, ylim = rz_limits(samples)
        heatmap_anchors = ("C", "C", "C", "C", "C")
        heatmap_axes = []
        for col, (grid_col, sample) in enumerate(zip(HEATMAP_GRID_COLS, samples, strict=True)):
            ax = fig.add_subplot(grid[row, grid_col])
            heatmap_axes.append(ax)
            mappable = plot_heatmap_panel(ax, sample, scale=scale, xlim=xlim, ylim=ylim)
            ax.set_xticks(HEATMAP_X_TICKS[case_key])
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
            ax.set_anchor(heatmap_anchors[col])
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
        heatmap_rows.append(heatmap_axes)
        radial_ax = fig.add_subplot(grid[row, RADIAL_GRID_COL])
        radial_axes.append(radial_ax)
        plot_radial_panel(
            radial_ax,
            samples,
            scale=scale,
            case_key=case_key,
            show_xlabel=row == len(case_keys) - 1,
            legend_loc="upper left" if row == 0 else "lower left",
        )
        if row == 0:
            radial_ax.set_title(
                rf"$\bf{{(b)}}$ $\varepsilon_{{{residual_latex_symbol()}}}(\rho)$",
                pad=PANEL_TITLE_PAD,
                fontsize=scaled_font_size(TITLE_FONT_SIZE),
                fontweight="normal",
            )

    max_heatmap_right = align_heatmap_columns(fig, heatmap_rows)
    if heatmap_rows:
        add_centered_panel_title(
            fig,
            heatmap_rows[0][0],
            heatmap_rows[0][-1],
            r"$\bf{(a)}$ Residual distribution",
            pad=PANEL_TITLE_PAD,
        )

    radial_left = min(max_heatmap_right + RADIAL_LEFT_PAD, FIGURE_RIGHT - 0.25)
    for radial_ax, heatmap_axes in zip(radial_axes, heatmap_rows, strict=True):
        row_bboxes = [ax.get_position().frozen() for ax in heatmap_axes]
        y0 = min(float(bbox.y0) for bbox in row_bboxes)
        y1 = max(float(bbox.y1) for bbox in row_bboxes)
        radial_ax.set_position([radial_left, y0, FIGURE_RIGHT - radial_left, y1 - y0])

    if mappable is not None:
        cax = fig.add_subplot(grid[:, COLORBAR_GRID_COL])
        cbar_bbox = cax.get_position()
        cbar_width = min(COLORBAR_MAX_WIDTH, cbar_bbox.width)
        cbar_height = COLORBAR_HEIGHT_FRACTION * cbar_bbox.height
        cbar_y = cbar_bbox.y0 + 0.5 * (cbar_bbox.height - cbar_height)
        cbar_x = max_heatmap_right + COLORBAR_LEFT_PAD
        cax.set_position([cbar_x, cbar_y, cbar_width, cbar_height])
        cbar = fig.colorbar(
            mappable,
            cax=cax,
        )
        cbar_ticks = np.arange(LOG_RESIDUAL_FLOOR, LOG_RESIDUAL_CEIL + 0.5, 1.0)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([rf"$10^{{{int(tick)}}}$" for tick in cbar_ticks])
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.ax.yaxis.set_label_position("right")
        cbar.set_label("")

        # Tilde residual colorbar label.
        cbar.ax.set_title(
            f"${residual_tilde_latex_symbol()}$",
            fontsize=scaled_font_size(AXIS_LABEL_SIZE),
            pad=6.0,
        )
        cbar.ax.tick_params(
            which="both", direction="out", labelsize=scaled_font_size(TICK_LABEL_SIZE)
        )
    return fig


def compact_case_samples(samples: list[ResidualSample]) -> list[ResidualSample]:
    by_label = {sample.config_label: sample for sample in samples}
    return [by_label[label] for label in COMPACT_CONFIG_LABELS]


def target_sample(samples: list[ResidualSample]) -> ResidualSample:
    return next((sample for sample in samples if sample.parameter_count is None), samples[-1])


def surface_at_psin_level(sample: ResidualSample, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the R-Z curve at a normalized-flux level.

    The compact 08-1/c-1 panels compare each VEQ equilibrium directly with
    GEQDSK target contours, so both curves must be selected at the same psin
    level.  Using the shared Legendre rho grid here would mix radial labels
    because psin(rho) is an active solved profile.
    """

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
        R = R_sorted[0]
        Z = Z_sorted[0]
    elif float(level) >= psin_sorted[-1]:
        R = R_sorted[-1]
        Z = Z_sorted[-1]
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
    if not (np.any(np.isfinite(R)) and np.any(np.isfinite(Z))):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    return np.r_[R, R[0]], np.r_[Z, Z[0]]


def plot_shape_comparison_panel(
    ax: plt.Axes,
    sample: ResidualSample,
    reference: ResidualSample,
    *,
    case_key: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    show_legend: bool,
) -> None:
    _ = case_key
    for level in SHAPE_SURFACE_LEVELS:
        linewidth = SHAPE_LINE_WIDTH if level < 1.0 - 1.0e-12 else SHAPE_BOUNDARY_LINE_WIDTH
        reference_r, reference_z = surface_at_psin_level(reference, level)
        sample_r, sample_z = surface_at_psin_level(sample, level)
        if reference_r.size:
            ax.plot(
                reference_r,
                reference_z,
                color=COMPACT_REFERENCE_COLOR,
                lw=linewidth * SHAPE_TARGET_LINE_WIDTH_SCALE,
                ls=SHAPE_TARGET_LINESTYLE,
                alpha=1.0,
                zorder=2,
                label="G-EQDSK" if show_legend and level == SHAPE_SURFACE_LEVELS[-1] else None,
            )
        if sample_r.size:
            ax.plot(
                sample_r,
                sample_z,
                color=COMPACT_VEQ_COLOR,
                lw=linewidth,
                ls="-",
                alpha=1.0,
                zorder=3,
                label=sample_legend_label(sample)
                if show_legend and level == SHAPE_SURFACE_LEVELS[-1]
                else None,
            )
    style_rz_axis(ax, xlim=xlim, ylim=ylim)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    if show_legend:
        ax.legend(
            loc="upper left",
            frameon=False,
            fontsize=scaled_font_size(LEGEND_FONT_SIZE),
            handlelength=1.8,
            labelspacing=0.2,
        )


def _point_in_polygon(poly: np.ndarray, R: float, Z: float) -> bool:
    """Ray-casting test: whether (R, Z) lies inside a closed polygon."""
    pts = np.asarray(poly, dtype=np.float64)
    n = pts.shape[0]
    inside = False
    j = n - 1
    for i in range(n):
        yi, zi = pts[i, 1], pts[i, 0]
        yj, zj = pts[j, 1], pts[j, 0]
        if (zi > Z) != (zj > Z):
            intersect = (yj - yi) * (Z - zi) / (zj - zi) + yi
            if R < intersect:
                inside = not inside
        j = i
    return inside


def _select_contour(
    candidates: list[np.ndarray], *, axis_center: tuple[float, float]
) -> np.ndarray | None:
    """Pick the longest closed contour segment that encloses the magnetic axis."""
    selected = None
    selected_length = -1
    for curve in candidates:
        arr = np.asarray(curve, dtype=np.float64)
        if arr.shape[0] < 8:
            continue
        if not _point_in_polygon(arr, axis_center[0], axis_center[1]):
            continue
        if arr.shape[0] > selected_length:
            selected = arr.copy()
            selected_length = arr.shape[0]
    if selected is not None:
        return selected
    if candidates:
        return max((np.asarray(c, dtype=np.float64) for c in candidates), key=len)
    return None


def _contour_flux_surfaces(geqdsk, levels: tuple[float, ...]) -> dict[float, np.ndarray]:
    """Extract flux-surface (R,Z) contours from a GEQDSK psi grid at given psin levels."""
    psi = np.asarray(geqdsk.psi, dtype=np.float64)
    psi_span = float(geqdsk.psi_bound - geqdsk.psi_axis)
    psin_grid = (psi.T - float(geqdsk.psi_axis)) / psi_span
    R = np.linspace(float(geqdsk.Rmin), float(geqdsk.Rmax), int(geqdsk.NR), dtype=np.float64)
    Z = np.linspace(float(geqdsk.Zmin), float(geqdsk.Zmax), int(geqdsk.NZ), dtype=np.float64)
    axis_center = (float(geqdsk.Raxis), float(geqdsk.Zaxis))
    surfaces: dict[float, np.ndarray] = {}
    subsurf = [lv for lv in levels if lv < 1.0 - 1e-12]
    if subsurf:
        fig, ax = plt.subplots()
        contour = ax.contour(R, Z, psin_grid, levels=subsurf)
        plt.close(fig)
        for idx, level in enumerate(subsurf):
            seg = _select_contour(contour.allsegs[idx], axis_center=axis_center)
            if seg is not None:
                surfaces[level] = seg
    if any(abs(lv - 1.0) <= 1e-12 for lv in levels):
        surfaces[1.0] = np.asarray(geqdsk.boundary, dtype=np.float64)
    return surfaces


def _resample_contour_sequential(points: np.ndarray, n_theta: int) -> np.ndarray:
    """Resample a closed contour to *n_theta* points by cumulative chord length."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return np.full((n_theta, 2), np.nan, dtype=np.float64)
    closed = np.allclose(pts[0], pts[-1], rtol=1e-12, atol=1e-12)
    loop = pts if closed else np.vstack([pts, pts[:1]])
    d = np.sqrt(np.sum(np.diff(loop, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    if total < 1e-30:
        return np.full((n_theta, 2), np.nan, dtype=np.float64)
    s_uniform = np.linspace(0.0, total, int(n_theta), endpoint=False, dtype=np.float64)
    out = np.empty((n_theta, 2), dtype=np.float64)
    out[:, 0] = np.interp(s_uniform, s, loop[:, 0])
    out[:, 1] = np.interp(s_uniform, s, loop[:, 1])
    return out


def build_true_target_sample(case_key: str, *, n_theta: int = 256) -> ResidualSample:
    """Build a shape-only sample whose R,Z surfaces come directly from GEQDSK contours."""
    geqdsk = read_geqdsk(CASE_REFERENCE_GFILES[case_key])
    surfaces = _contour_flux_surfaces(geqdsk, SHAPE_SURFACE_LEVELS)
    n_levels = len(SHAPE_SURFACE_LEVELS)
    R = np.full((n_levels, n_theta), np.nan, dtype=np.float64)
    Z = np.full((n_levels, n_theta), np.nan, dtype=np.float64)
    for idx, level in enumerate(SHAPE_SURFACE_LEVELS):
        contour = surfaces.get(level)
        if contour is not None:
            resampled = _resample_contour_sequential(contour, n_theta)
            R[idx, :] = resampled[:, 0]
            Z[idx, :] = resampled[:, 1]
    return ResidualSample(
        case_key=case_key,
        config_label=EXTERNAL_REFERENCE_LABELS[case_key],
        signature={},
        parameter_count=None,
        elapsed_ms=float("nan"),
        solver_residual_norm=float("nan"),
        rho=np.array(SHAPE_SURFACE_LEVELS, dtype=np.float64),
        psin=np.array(SHAPE_SURFACE_LEVELS, dtype=np.float64),
        R=R,
        Z=Z,
        G=np.full((n_levels, n_theta), np.nan, dtype=np.float64),
        radial_rms=np.full(n_levels, np.nan, dtype=np.float64),
    )


def build_compact_figure(case_samples: dict[str, list[ResidualSample]]) -> plt.Figure:
    case_keys = list(case_samples)
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
        samples = compact_case_samples(case_samples[case_key])
        target = build_true_target_sample(case_key)
        xlim, ylim = rz_limits([*samples, target])
        for col, sample in enumerate(samples):
            ax = fig.add_subplot(grid[row, col])
            plot_shape_comparison_panel(
                ax,
                sample,
                target,
                case_key=case_key,
                xlim=xlim,
                ylim=ylim,
                show_legend=False,
            )
            ax.set_xticks(HEATMAP_X_TICKS[case_key])
            ax.set_anchor("C")
            ax.set_title(
                COMPACT_COLUMN_LABELS[col] if row == 0 else "",
                fontsize=scaled_font_size(TITLE_FONT_SIZE),
                fontweight="normal",
            )
            if col == 0:
                ax.set_ylabel(
                    f"{CASE_LABELS[case_key]}\nZ [m]", fontsize=scaled_font_size(AXIS_LABEL_SIZE)
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
            color=COMPACT_REFERENCE_COLOR,
            lw=SHAPE_LINE_WIDTH * SHAPE_TARGET_LINE_WIDTH_SCALE,
            ls=SHAPE_TARGET_LINESTYLE,
            label="G-EQDSK",
        ),
        Line2D(
            [0],
            [0],
            color=COMPACT_VEQ_COLOR,
            lw=SHAPE_LINE_WIDTH,
            ls="-",
            label="VEQ",
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


def main() -> None:
    apply_plot_style()
    selected_cases = CASE_KEYS if CASE_KEYS_TO_RUN == "all" else tuple(CASE_KEYS_TO_RUN)
    print_script_config(
        SCRIPT_CONSOLE,
        "figure 08 / table 06: residual distribution",
        (
            ("cases", len(selected_cases)),
            ("configs", ", ".join(CONFIG_LABELS)),
            ("residual", "standard GS" if USE_STANDARD_GS_RESIDUAL else "transformed"),
        ),
    )
    with script_progress(SCRIPT_CONSOLE) as progress:
        total = 4 if COMPACT_PNG_PATH is not None else 3
        task = progress.add_task("", total=total, current="load samples", phase="[cyan]run[/]")
        case_samples = load_case_samples_from_equilibrium_jsons(
            selected_cases,
        )
        progress.update(task, advance=1, current="main figure", phase="[cyan]run[/]")
        fig = build_figure(case_samples)
        saved_paths = save_figure_outputs(
            fig,
            png_path=PNG_PATH,
            pdf_path=PDF_PATH,
            dpi=SAVE_DPI,
            transparent=SAVE_TRANSPARENT,
        )
        plt.close(fig)
        progress.update(task, advance=1, current="compact figure", phase="[cyan]run[/]")
        compact_saved_paths: list[str] = []
        if COMPACT_PNG_PATH is not None:
            compact_fig = build_compact_figure(case_samples)
            compact_saved_paths = save_figure_outputs(
                compact_fig,
                png_path=COMPACT_PNG_PATH,
                pdf_path=None,
                dpi=SAVE_DPI,
                transparent=SAVE_TRANSPARENT,
            )
            plt.close(compact_fig)
            progress.update(task, advance=1, current="table", phase="[cyan]run[/]")
        rows = residual_norm_rows(case_samples)
        progress.update(task, advance=1, current="table", phase="[green]done[/]")
    print_residual_norm_summary(rows)
    output_rows = [
        ("Figure 08", path, "Sampled standard-form residual distribution")
        for path in saved_paths
    ]
    output_rows.extend(
        ("Figure 08 compact", path, "Compact G-EQDSK vs VEQ shape comparison")
        for path in compact_saved_paths
    )
    print_output_table(SCRIPT_CONSOLE, output_rows)


if __name__ == "__main__":
    main()
