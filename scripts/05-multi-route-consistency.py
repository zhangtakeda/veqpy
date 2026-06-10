"""Compare VEQPy source routes on one shared demo equilibrium.

The script solves the same physical setup through multiple route/coordinate
choices and plots route-to-route profile and surface consistency diagnostics.
"""

import os
from dataclasses import dataclass
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from config import (
    AXIS_LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    PLOT_LABEL_RIGHT,
    PLOT_LABEL_TOP,
    PLOT_TICK_BOTTOM,
    PLOT_TICK_DIRECTION,
    PLOT_TICK_LEFT,
    PLOT_TICK_RIGHT,
    PLOT_TICK_TOP,
    SAVE_DPI,
    SAVE_TRANSPARENT,
    SINGLE_COLUMN_WIDTH,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    apply_plot_style,
    profile_interp,
    save_figure_outputs,
    scaled_font_size,
)
from config import (
    demo_psin_reference_profiles as pf_reference_profiles,
)
from matplotlib.ticker import LogLocator, NullFormatter

MU0 = 4.0e-7 * np.pi

FIGURE_SIZE = (SINGLE_COLUMN_WIDTH, 4.5)
FIGURE_NROWS = 2
FIGURE_NCOLS = 2
FIGURE_CONSTRAINED_LAYOUT = True
FIGURE_WSPACE = 0.06
FIGURE_HSPACE = 0.08
PNG_PATH = "05-multi-route-consistency.png"
PDF_PATH = None

TOP_SPINE_VISIBLE = True
RIGHT_SPINE_VISIBLE = True
GRID_ALPHA = 0.28
GRID_LINE_WIDTH = 0.8
GRID_LINESTYLE = "-"

LINE_WIDTH = 1.0
MARKER_SIZE = 4
Y_MIN_FLOOR = 1.0e-16
Y_MAX = 1.0
LOG_BASE = 10.0
LOG_MINOR_SUBS = np.arange(2, 10, dtype=np.float64) * 0.1

NR_LABEL = r"$N_\rho$"
VALUE_LABEL = "error"
LEGEND_LOC = "upper center"
LEGEND_NCOLS = 3
LEGEND_FRAME_ON = False
LEGEND_BBOX_TO_ANCHOR = (0.5, 0.98)
LEGEND_BORDER_AXES_PAD = 0.2
Q95_PSIN = 0.95

METRIC_SPECS = (
    ("shape_error", r"$\bf{(a)}$ $E_{\mathrm{coeff}}$"),
    ("ip_rel_error", r"$\bf{(b)}$ $\Delta_{I_p}$"),
    ("beta_rel_error", r"$\bf{(c)}$ $\Delta_{\beta_t}$"),
    ("q95_rel_error", r"$\bf{(d)}$ $\Delta_{q_{95}}$"),
)

REFERENCE_NR = 64
REFERENCE_NT = 64
TEST_NT = 32
DEFAULT_GRID_SIZES = tuple(range(12, 64, 2))
BACKEND = "numba"
TABLE_NR = 32
ROUTES = ("PF", "PP", "PI", "PJ1", "PJ2", "PQ")
REFERENCE_CONSTRAINT = "null"
REFERENCE_SOURCE_SAMPLE_COUNT = 51
TEST_SOURCE_SAMPLE_COUNT = 51
REFERENCE_IP = 3.0e6
REFERENCE_MU0_IP = MU0 * REFERENCE_IP
REFERENCE_BOUNDARY_A = 1.05 / 1.85
REFERENCE_BOUNDARY_R0 = 1.05
REFERENCE_BOUNDARY_Z0 = 0.0
REFERENCE_BOUNDARY_B0 = 3.0
REFERENCE_BOUNDARY_KA = 2.2
REFERENCE_BOUNDARY_S_OFFSETS = np.array([0.0, float(np.arcsin(0.5))], dtype=np.float64)
BASE_COEFFS = {
    "h": [0.0] * 3,
    "k": [0.0] * 5,
    "s1": [0.0] * 3,
}
_UNIFORM_SOURCE_AXIS = np.linspace(0.0, 1.0, TEST_SOURCE_SAMPLE_COUNT, dtype=np.float64)


def route_constraint(route: str) -> str:
    return REFERENCE_CONSTRAINT


def reference_mu0_ip(benchmark) -> float:
    return float(getattr(benchmark, "REFERENCE_MU0_IP", MU0 * float(benchmark.REFERENCE_IP)))


ROUTE_COLORS = {
    "PF": "#1b9e77",
    "PP": "#d95f02",
    "PI": "#7570b3",
    "PJ1": "#e7298a",
    "PJ2": "#66a61e",
    "PQ": "#e6ab02",
}
ROUTE_MARKERS = {
    "PF": "o",
    "PP": "s",
    "PI": "^",
    "PJ1": "D",
    "PJ2": "v",
    "PQ": "P",
}


@dataclass(frozen=True)
class ReferenceData:
    grid: object
    result: object
    equilibrium: object
    ref_profiles: dict[str, np.ndarray | float]
    shape_x: np.ndarray
    rho_axis: np.ndarray
    target_beta_t: float
    target_q95: float


@dataclass(frozen=True)
class RegressionRow:
    route: str
    nr: int
    nt: int
    shape_error: float
    ip_rel_error: float
    beta_rel_error: float
    q95_rel_error: float
    elapsed_us: float
    nfev: int
    nit: int
    residual_norm_final: float


def build_pf_reference_profiles(equilibrium) -> dict[str, np.ndarray | float]:
    psin_r = np.asarray(equilibrium.psin_r, dtype=np.float64).copy()
    psin_r_safe = np.where(np.abs(psin_r) > 1e-14, psin_r, 1e-14)

    psi_r = np.asarray(equilibrium.alpha2 * psin_r, dtype=np.float64)
    psi_r_safe = np.where(np.abs(psi_r) > 1e-14, psi_r, 1e-14)

    FFn_r = np.asarray(equilibrium.FFn_r, dtype=np.float64).copy()
    Pn_r = np.asarray(equilibrium.Pn_r, dtype=np.float64).copy()
    FF_r = np.asarray(equilibrium.FF_r, dtype=np.float64).copy()
    P_r = np.asarray(equilibrium.P_r, dtype=np.float64).copy()
    Itor = np.asarray(equilibrium.Itor, dtype=np.float64).copy()
    jtor = np.asarray(equilibrium.jtor, dtype=np.float64).copy()
    jpara = np.asarray(equilibrium.jpara, dtype=np.float64).copy()
    q = np.asarray(equilibrium.q, dtype=np.float64).copy()
    mu0_P_r = MU0 * P_r
    mu0_P_psi = mu0_P_r / psi_r_safe
    mu0_Itor = MU0 * Itor
    mu0_jtor = MU0 * jtor
    mu0_jpara = MU0 * jpara

    return {
        "psin_r": psin_r,
        "psi_r": psi_r,
        "FFn_r": FFn_r,
        "Pn_r": Pn_r,
        "FFn_psin": FFn_r / psin_r_safe,
        "Pn_psin": Pn_r / psin_r_safe,
        "setup_Pn_r": Pn_r / MU0,
        "setup_Pn_psin": (Pn_r / psin_r_safe) / MU0,
        "FF_r": FF_r,
        "P_r": P_r,
        "mu0_P_r": mu0_P_r,
        "FF_psi": FF_r / psi_r_safe,
        "P_psi": P_r / psi_r_safe,
        "mu0_P_psi": mu0_P_psi,
        "Itorn": mu0_Itor,
        "Itor": Itor,
        "mu0_Itor": mu0_Itor,
        "jtorn": mu0_jtor,
        "jtor": jtor,
        "mu0_jtor": mu0_jtor,
        "jparan": mu0_jpara,
        "jpara": jpara,
        "mu0_jpara": mu0_jpara,
        "qn": q * 0.1,
        "q": q,
        "mu0_Ip": float(MU0 * equilibrium.Ip),
        "beta_constraint": float(equilibrium.beta_t),
    }


def _constraint_route_domains(constraint: str) -> tuple[str, str]:
    if constraint == "Ip_beta":
        return "normalized", "normalized"
    if constraint == "Ip":
        return "normalized", "physical"
    if constraint == "beta":
        return "physical", "normalized"
    return "physical", "physical"


def _pressure_keys_for_coordinate(coordinate: str) -> tuple[str, str]:
    if coordinate == "rho":
        return "setup_Pn_r", "P_r"
    return "setup_Pn_psin", "P_psi"


def _pick_ref_profile(
    ref: dict[str, np.ndarray | float],
    normalized_key: str,
    physical_key: str,
    normalized: bool,
) -> np.ndarray:
    key = normalized_key if normalized else physical_key
    return np.asarray(ref[key], dtype=np.float64)


def _build_mode_init_kwargs(
    mode: str,
    coordinate: str,
    constraint: str,
    ref: dict[str, np.ndarray | float],
) -> dict[str, np.ndarray]:
    pressure_keys = _pressure_keys_for_coordinate(coordinate)

    if mode == "PF":
        use_normalized = constraint in {"Ip", "beta"}
        driver_keys = ("FFn_r", "FF_r") if coordinate == "rho" else ("FFn_psin", "FF_psi")
        return {
            "current_input": _pick_ref_profile(ref, driver_keys[0], driver_keys[1], use_normalized),
            "heat_input": _pick_ref_profile(
                ref, pressure_keys[0], pressure_keys[1], use_normalized
            ),
        }

    if mode == "PP":
        driver_normalized = constraint in {"Ip_beta", "Ip"}
        pressure_normalized = constraint in {"Ip_beta", "beta"}
        return {
            "current_input": _pick_ref_profile(ref, "psin_r", "psi_r", driver_normalized),
            "heat_input": _pick_ref_profile(
                ref, pressure_keys[0], pressure_keys[1], pressure_normalized
            ),
        }

    driver_domain, pressure_domain = _constraint_route_domains(constraint)
    driver_keys = {
        "PI": ("Itor", "Itor"),
        "PJ1": ("jtor", "jtor"),
        "PJ2": ("jpara", "jpara"),
        "PQ": ("qn", "q"),
    }[mode]
    return {
        "current_input": _pick_ref_profile(
            ref, driver_keys[0], driver_keys[1], driver_domain == "normalized"
        ),
        "heat_input": _pick_ref_profile(
            ref, pressure_keys[0], pressure_keys[1], pressure_domain == "normalized"
        ),
    }


def _profile_coeffs_for_case(
    mode: str,
    coordinate: str,
    input_kind: str,
    *,
    constraint: str | None = None,
) -> dict[str, list[float] | None]:
    del coordinate, input_kind, constraint
    coeffs = {name: list(values) for name, values in BASE_COEFFS.items()}
    if mode in {"PJ2"}:
        coeffs["F"] = [0.0] * 5
    return coeffs


def load_benchmark_module(backend: str):
    """Return the small benchmark API this figure needs, without importing tests/."""

    os.environ["VEQPY_BACKEND"] = str(backend)
    from veqpy.model import Boundary, Grid
    from veqpy.operator import (
        Operator,
        OperatorCase,
        build_profile_index,
        build_profile_layout,
        build_profile_names,
        build_shape_profile_names,
    )
    from veqpy.solver import Solver, SolverConfig

    boundary = Boundary(
        a=REFERENCE_BOUNDARY_A,
        R0=REFERENCE_BOUNDARY_R0,
        Z0=REFERENCE_BOUNDARY_Z0,
        B0=REFERENCE_BOUNDARY_B0,
        ka=REFERENCE_BOUNDARY_KA,
        s_offsets=REFERENCE_BOUNDARY_S_OFFSETS.copy(),
    )
    config = SolverConfig(
        method="hybr",
        enable_verbose=False,
        enable_warmstart=False,
        enable_history=False,
    )

    def _reference_pf_case() -> OperatorCase:
        rho_src = np.linspace(0.0, 1.0, REFERENCE_SOURCE_SAMPLE_COUNT)
        psin_src = rho_src * rho_src
        FFn_psin_src, Pn_psin_src = pf_reference_profiles(psin_src)
        FFn_r_src = FFn_psin_src * (2.0 * rho_src)
        Pn_r_src = Pn_psin_src * (2.0 * rho_src)
        return OperatorCase(
            route="PF",
            coordinate="rho",
            nodes="uniform",
            profile_coeffs=BASE_COEFFS,
            boundary=boundary,
            heat_input=Pn_r_src / MU0,
            current_input=FFn_r_src,
            Ip=REFERENCE_IP,
        )

    return SimpleNamespace(
        Grid=Grid,
        Operator=Operator,
        OperatorCase=OperatorCase,
        Solver=Solver,
        SolverConfig=SolverConfig,
        CONFIG=config,
        BASE_COEFFS=BASE_COEFFS,
        BOUNDARY=boundary,
        REFERENCE_IP=REFERENCE_IP,
        REFERENCE_MU0_IP=REFERENCE_MU0_IP,
        _UNIFORM_SOURCE_AXIS=_UNIFORM_SOURCE_AXIS,
        build_profile_index=build_profile_index,
        build_profile_layout=build_profile_layout,
        build_profile_names=build_profile_names,
        build_shape_profile_names=build_shape_profile_names,
        build_pf_reference_profiles=build_pf_reference_profiles,
        _reference_pf_case=_reference_pf_case,
        _build_mode_init_kwargs=_build_mode_init_kwargs,
        _profile_coeffs_for_case=_profile_coeffs_for_case,
    )


def _relative_rms_error(reference_values: np.ndarray, current_values: np.ndarray) -> float:
    reference_values = np.asarray(reference_values, dtype=np.float64)
    current_values = np.asarray(current_values, dtype=np.float64)
    n = min(reference_values.shape[0], current_values.shape[0])
    if n == 0:
        return 0.0
    diff = current_values[:n] - reference_values[:n]
    scale = max(float(np.max(np.abs(reference_values[:n]))), 1.0e-12)
    return float(np.sqrt(np.mean(diff * diff)) / scale)


def _extract_shape_x(
    benchmark, profile_coeffs: dict[str, list[float] | None], x: np.ndarray, m_max: int
) -> np.ndarray:
    profile_names = benchmark.build_profile_names(m_max)
    profile_index = benchmark.build_profile_index(profile_names)
    _, coeff_index, _ = benchmark.build_profile_layout(profile_coeffs, profile_names=profile_names)
    shape_profile_names = benchmark.build_shape_profile_names(m_max)
    shape_values: list[float] = []
    x = np.asarray(x, dtype=np.float64)
    for k in range(coeff_index.shape[1]):
        for name in shape_profile_names:
            idx = int(coeff_index[profile_index[name], k])
            if idx >= 0:
                shape_values.append(float(x[idx]))
    return np.asarray(shape_values, dtype=np.float64)


def _shape_error(reference_shape_x: np.ndarray, current_shape_x: np.ndarray) -> float:
    n = min(reference_shape_x.shape[0], current_shape_x.shape[0])
    if n == 0:
        return 0.0
    diff = current_shape_x[:n] - reference_shape_x[:n]
    return float(np.sqrt(np.mean(diff * diff)))


def q_at_psin(equilibrium, psin_query: float = Q95_PSIN) -> float:
    """Interpolate q at a target normalized flux coordinate."""
    psin = np.asarray(equilibrium.psin, dtype=np.float64)
    q = np.asarray(equilibrium.q, dtype=np.float64)
    if psin.ndim != 1 or q.ndim != 1 or psin.size != q.size:
        raise ValueError(
            f"Expected 1D psin/q arrays with equal length, got {psin.shape} and {q.shape}"
        )

    order = np.argsort(psin)
    psin_sorted = psin[order]
    q_sorted = q[order]
    psin_unique, unique_idx = np.unique(psin_sorted, return_index=True)
    q_unique = q_sorted[unique_idx]
    psin_clamped = float(np.clip(psin_query, psin_unique[0], psin_unique[-1]))
    return float(np.interp(psin_clamped, psin_unique, q_unique))


def _style_axis(
    ax: plt.Axes,
    *,
    title: str,
    row: int,
    col: int,
    nrows: int,
    ncols: int,
    test_nt: int,
) -> None:
    """Apply shared plot-axis styling."""
    show_bottom = True
    show_left = col == 0
    ax.set_title(title, fontsize=scaled_font_size(TITLE_FONT_SIZE), fontweight="normal")
    ax.set_xlabel(
        rf"{NR_LABEL} ($N_\theta={test_nt}$)" if show_bottom else "",
        fontsize=scaled_font_size(AXIS_LABEL_FONT_SIZE),
    )
    ax.set_ylabel(VALUE_LABEL if show_left else "", fontsize=scaled_font_size(AXIS_LABEL_FONT_SIZE))
    ax.tick_params(
        direction=PLOT_TICK_DIRECTION,
        top=PLOT_TICK_TOP,
        right=PLOT_TICK_RIGHT,
        bottom=PLOT_TICK_BOTTOM,
        left=PLOT_TICK_LEFT,
        labeltop=PLOT_LABEL_TOP,
        labelright=PLOT_LABEL_RIGHT,
        labelbottom=show_bottom,
        labelleft=show_left,
        labelsize=scaled_font_size(TICK_LABEL_FONT_SIZE),
    )
    ax.spines["top"].set_visible(TOP_SPINE_VISIBLE)
    ax.spines["right"].set_visible(RIGHT_SPINE_VISIBLE)
    ax.grid(True, which="major", alpha=GRID_ALPHA, lw=GRID_LINE_WIDTH, linestyle=GRID_LINESTYLE)


def _style_legend(ax: plt.Axes, handles, labels) -> None:
    """Apply shared legend styling to the first subplot."""
    ax.legend(
        handles,
        labels,
        loc=LEGEND_LOC,
        ncols=LEGEND_NCOLS,
        frameon=LEGEND_FRAME_ON,
        fontsize=scaled_font_size(LEGEND_FONT_SIZE),
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        borderaxespad=LEGEND_BORDER_AXES_PAD,
    )


def solve_with_benchmark_config(solver, config) -> None:
    solver.solve(
        method=config.method,
        max_residual=getattr(config, "max_residual", getattr(config, "rtol", None)),
        max_evaluations=getattr(config, "max_evaluations", getattr(config, "root_maxfev", None)),
        enable_verbose=False,
        enable_history=False,
        enable_warmstart=False,
    )


def build_reference(benchmark) -> ReferenceData:
    reference_grid = benchmark.Grid(Nr=REFERENCE_NR, Nt=REFERENCE_NT, quadrature_scheme="legendre")
    source_reference_case = benchmark._reference_pf_case()
    reference_case = benchmark.OperatorCase(
        route="PF",
        coordinate="rho",
        nodes="grid",
        profile_coeffs={name: list(values) for name, values in benchmark.BASE_COEFFS.items()},
        boundary=benchmark.BOUNDARY,
        heat_input=profile_interp(
            benchmark._UNIFORM_SOURCE_AXIS,
            source_reference_case.heat_input,
            np.asarray(reference_grid.rho, dtype=np.float64),
        )
        / MU0,
        current_input=profile_interp(
            benchmark._UNIFORM_SOURCE_AXIS,
            source_reference_case.current_input,
            np.asarray(reference_grid.rho, dtype=np.float64),
        ),
        Ip=float(getattr(benchmark, "REFERENCE_IP", source_reference_case.Ip / MU0)),
    )
    solver = benchmark.Solver(
        operator=benchmark.Operator(reference_grid, reference_case),
        config=benchmark.CONFIG,
    )
    solve_with_benchmark_config(solver, benchmark.CONFIG)
    result = solver.result
    equilibrium = solver.build_equilibrium()
    return ReferenceData(
        grid=reference_grid,
        result=result,
        equilibrium=equilibrium,
        ref_profiles=benchmark.build_pf_reference_profiles(equilibrium),
        shape_x=_extract_shape_x(
            benchmark,
            reference_case.profile_coeffs,
            result.x,
            reference_grid.M_max,
        ),
        rho_axis=np.asarray(equilibrium.rho, dtype=np.float64),
        target_beta_t=float(equilibrium.beta_t),
        target_q95=q_at_psin(equilibrium),
    )


def build_case(benchmark, reference: ReferenceData, route: str, nr: int, nt: int):
    grid = benchmark.Grid(
        Nr=nr,
        Nt=nt,
        quadrature_scheme="legendre",
        L_max=reference.grid.L_max,
        M_max=reference.grid.M_max,
    )
    constraint = route_constraint(route)
    init_kwargs = benchmark._build_mode_init_kwargs(
        route, "rho", constraint, reference.ref_profiles
    )
    if hasattr(benchmark, "_split_benchmark_inputs"):
        heat_profile, current_profile = benchmark._split_benchmark_inputs(init_kwargs)
    else:
        heat_profile = init_kwargs["heat_input"]
        current_profile = init_kwargs["current_input"]
    grid_axis = np.asarray(grid.rho, dtype=np.float64)
    heat_input = profile_interp(reference.rho_axis, heat_profile, grid_axis)
    current_input = profile_interp(reference.rho_axis, current_profile, grid_axis)
    case = benchmark.OperatorCase(
        route=route,
        coordinate="rho",
        nodes="grid",
        profile_coeffs=benchmark._profile_coeffs_for_case(
            route, "rho", "grid", constraint=constraint
        ),
        boundary=benchmark.BOUNDARY,
        heat_input=heat_input,
        current_input=current_input,
        Ip=float(getattr(benchmark, "REFERENCE_IP", REFERENCE_IP))
        if constraint in {"Ip", "Ip_beta"}
        else None,
        beta=(
            float(reference.ref_profiles["beta_constraint"])
            if constraint in {"beta", "Ip_beta"}
            else None
        ),
    )
    return grid, case


def solve_case(benchmark, reference: ReferenceData, route: str, nr: int, nt: int) -> RegressionRow:
    grid, case = build_case(benchmark, reference, route, nr, nt)
    solver = benchmark.Solver(operator=benchmark.Operator(grid, case), config=benchmark.CONFIG)
    solve_with_benchmark_config(solver, benchmark.CONFIG)
    result = solver.result
    if result is None:
        raise RuntimeError(f"{route} Nr={nr}, Nt={nt} produced no SolverResult")
    if not bool(result.success):
        raise RuntimeError(
            f"{route} Nr={nr}, Nt={nt} failed to converge: "
            f"{result.message} (residual={result.residual_norm_final:.3e})"
        )
    equilibrium = solver.build_equilibrium()

    current_shape_x = _extract_shape_x(
        benchmark, case.profile_coeffs, result.x, reference.grid.M_max
    )
    ip_reference = float(reference.equilibrium.Ip)
    beta_reference = float(reference.target_beta_t)
    q95_reference = float(reference.target_q95)
    ip_current = float(equilibrium.Ip)
    beta_current = float(equilibrium.beta_t)
    q95_current = q_at_psin(equilibrium)

    ip_scale = max(abs(ip_reference), 1.0e-12)
    beta_scale = max(abs(beta_reference), 1.0e-12)
    q95_scale = max(abs(q95_reference), 1.0e-12)

    return RegressionRow(
        route=route,
        nr=nr,
        nt=nt,
        shape_error=_shape_error(reference.shape_x, current_shape_x),
        ip_rel_error=float(abs(ip_current - ip_reference) / ip_scale),
        beta_rel_error=float(abs(beta_current - beta_reference) / beta_scale),
        q95_rel_error=float(abs(q95_current - q95_reference) / q95_scale),
        elapsed_us=float(result.elapsed),
        nfev=int(getattr(result, "nfev", result.function_evaluations)),
        nit=int(getattr(result, "nit", result.iterations)),
        residual_norm_final=float(result.residual_norm_final),
    )


def run_regression(
    benchmark, nr_list: list[int], test_nt: int
) -> tuple[ReferenceData, list[RegressionRow]]:
    reference = build_reference(benchmark)
    rows: list[RegressionRow] = []
    for route in ROUTES:
        for nr in nr_list:
            try:
                row = solve_case(benchmark, reference, route, nr, test_nt)
            except RuntimeError as exc:
                print(f"[{route}] Nr={nr:>2d}, Nt={test_nt:>2d}: skipped ({exc})")
                continue
            rows.append(row)
            print(
                f"[{route}] Nr={nr:>2d}, Nt={test_nt:>2d}: "
                f"elapsed={row.elapsed_us / 1000.0:.3f} ms | "
                f"shape={row.shape_error:.3e} | "
                f"Ip={row.ip_rel_error:.3e} | "
                f"beta={row.beta_rel_error:.3e} | "
                f"q95={row.q95_rel_error:.3e} | "
                f"nfev={row.nfev:>3d}"
            )
    return reference, rows


def _plot_values(values: list[float]) -> list[float]:
    return [max(value, 1.0e-16) for value in values]


def _format_tex_number(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "--"
    abs_value = abs(value)
    if abs_value == 0.0:
        return "$0$"
    if 1.0e-2 <= abs_value < 1.0e2:
        return f"${value:.3f}$"
    mantissa, exponent_text = f"{value:.3e}".split("e")
    exponent = int(exponent_text)
    return rf"${float(mantissa):.3f}\times 10^{{{exponent}}}$"


def build_route_consistency_latex_table(
    rows: list[RegressionRow], *, table_nr: int, test_nt: int
) -> str:
    selected = [row for row in rows if int(row.nr) == int(table_nr) and int(row.nt) == int(test_nt)]
    selected.sort(key=lambda row: ROUTES.index(row.route))
    if not selected:
        available = sorted({(row.nr, row.nt) for row in rows})
        raise ValueError(
            f"No converged rows available for Nr={table_nr}, Nt={test_nt}. "
            f"Available grids: {available}"
        )

    indent = "              "
    header = [
        "Route",
        r"$E_{\mathrm{coeff}}$",
        r"$\Delta_{I_p}$",
        r"$\Delta_{\beta_t}$",
        r"$\Delta_{q_{95}}$",
    ]
    table_rows = [
        [
            row.route,
            _format_tex_number(row.shape_error),
            _format_tex_number(row.ip_rel_error),
            _format_tex_number(row.beta_rel_error),
            _format_tex_number(row.q95_rel_error),
        ]
        for row in selected
    ]
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


def print_route_consistency_latex_table(
    rows: list[RegressionRow], *, table_nr: int, test_nt: int
) -> None:
    print(build_route_consistency_latex_table(rows, table_nr=table_nr, test_nt=test_nt))


def build_route_regression_figure(rows: list[RegressionRow], *, test_nt: int) -> plt.Figure:
    apply_plot_style()
    fig, axes = plt.subplots(
        FIGURE_NROWS,
        FIGURE_NCOLS,
        figsize=FIGURE_SIZE,
        constrained_layout=FIGURE_CONSTRAINED_LAYOUT,
        sharex=False,
        sharey=True,
    )

    grouped: dict[str, list[RegressionRow]] = {route: [] for route in ROUTES}
    for row in rows:
        grouped[row.route].append(row)
    for route_rows in grouped.values():
        route_rows.sort(key=lambda row: row.nr)

    flat_axes = np.ravel(axes)
    for idx, (ax, (metric_name, panel_label)) in enumerate(
        zip(flat_axes, METRIC_SPECS, strict=True)
    ):
        row_idx = idx // FIGURE_NCOLS
        col_idx = idx % FIGURE_NCOLS
        for route in ROUTES:
            route_rows = grouped[route]
            x_values = [row.nr for row in route_rows]
            y_values = _plot_values([float(getattr(row, metric_name)) for row in route_rows])
            ax.semilogy(
                x_values,
                y_values,
                linestyle="-",
                linewidth=LINE_WIDTH,
                color=ROUTE_COLORS[route],
                marker=ROUTE_MARKERS[route],
                markersize=MARKER_SIZE,
                label=route,
            )
        _style_axis(
            ax,
            title=panel_label,
            row=row_idx,
            col=col_idx,
            nrows=FIGURE_NROWS,
            ncols=FIGURE_NCOLS,
            test_nt=test_nt,
        )
        ax.set_ylim(1e-08, Y_MAX)
        ax.yaxis.set_major_locator(LogLocator(base=LOG_BASE))
        ax.yaxis.set_minor_locator(LogLocator(base=LOG_BASE, subs=LOG_MINOR_SUBS))
        ax.yaxis.set_minor_formatter(NullFormatter())

    handles, labels = flat_axes[0].get_legend_handles_labels()
    _style_legend(flat_axes[0], handles, labels)
    return fig


def main() -> None:
    nr_list = sorted({int(size) for size in DEFAULT_GRID_SIZES})
    if not nr_list:
        raise ValueError("At least one Nr value is required")
    if any(size <= 0 for size in nr_list):
        raise ValueError("Nr values must be positive")
    if TEST_NT <= 0:
        raise ValueError("test_nt must be positive")
    if TABLE_NR <= 0:
        raise ValueError("table_nr must be positive")
    if int(TABLE_NR) not in nr_list:
        nr_list = sorted({*nr_list, int(TABLE_NR)})

    benchmark = load_benchmark_module(BACKEND)
    _, rows = run_regression(benchmark, nr_list, TEST_NT)
    print_route_consistency_latex_table(rows, table_nr=int(TABLE_NR), test_nt=TEST_NT)

    fig = build_route_regression_figure(rows, test_nt=TEST_NT)
    saved_paths = save_figure_outputs(
        fig,
        png_path=PNG_PATH,
        pdf_path=PDF_PATH,
        dpi=SAVE_DPI,
        transparent=SAVE_TRANSPARENT,
    )
    plt.close(fig)

    for path in saved_paths:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
