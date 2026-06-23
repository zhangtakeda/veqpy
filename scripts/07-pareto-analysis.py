"""Run Pareto sweeps over active coefficient layouts.

This manuscript benchmark searches reduced VEQ representations for each
reference GEQDSK case, records timing/error tradeoffs, and writes the JSON
artifacts consumed by later diagnostic scripts.
"""

import concurrent.futures
import itertools
import json
import multiprocessing
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from config import (
    AXIS_LABEL_FONT_SIZE,
    BOUNDARY_MAXTOL,
    CASE_BOUNDARY_FIT_M,
    CASE_BOUNDARY_FIT_N,
    CASE_COLORS,
    CASE_KEYS,
    CASE_LABELS,
    CASE_LINESTYLES,
    CASE_REFERENCE_EQUILIBRIUM_JSONS,
    CASE_REFERENCE_GFILES,
    CASE_REFERENCE_PROFILE_LENGTHS,
    CASE_SOLVER_METHODS,
    DEFAULT_JSON_STEM,
    DOUBLE_COLUMN_WIDTH,
    FIXED_DECIMALS,
    LEGEND_FONT_SIZE,
    MU0,
    PLOT_LABEL_RIGHT,
    PLOT_LABEL_TOP,
    PLOT_TICK_BOTTOM,
    PLOT_TICK_DIRECTION,
    PLOT_TICK_LEFT,
    PLOT_TICK_RIGHT,
    PLOT_TICK_TOP,
    REDUCED_CONFIG_LABELS,
    REDUCED_EQUILIBRIUM_JSON_TEMPLATE,
    REDUCED_EQUILIBRIUM_MANIFEST_PATH,
    REFERENCE_SOLVER_MAXFEV,
    SAVE_DPI,
    SAVE_TRANSPARENT,
    SCIENTIFIC_DECIMALS,
    SOLVER_INITIAL_POLICY,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    active_profiles_from_coeffs,
    apply_plot_style,
    build_geqdsk_boundary,
    coefficients_from_profile_coeffs,
    figure_path,
    read_geqdsk,
    save_figure_outputs,
    scaled_font_size,
)
from config import (
    as_float64_array as _as_float64_array,
)
from config import (
    load_veqpy_components as _load_veqpy_components,
)
from config import (
    prepare_interp_axis as _prepare_interp_axis,
)
from config import (
    profile_interp as _profile_interp,
)
from matplotlib.ticker import (
    FixedLocator,
    FormatStrFormatter,
    LogFormatterMathtext,
    MaxNLocator,
    NullLocator,
)

FIGURE_SIZE = (DOUBLE_COLUMN_WIDTH, 6)
FIGURE_LEFT_RIGHT_WIDTH_RATIOS = (1.0, 0.75)
FIGURE_PARETO_HSPACE = 0.14
FIGURE_SURFACE_WSPACE = -0.06
FIGURE_SURFACE_HSPACE = 0.06
FIGURE_SURFACE_COLUMN_GAP = 0.018
FIGURE_CONSTRAINED_LAYOUT = True
FIGURE_WSPACE = 0.5
FIGURE_HSPACE = 0.08
PNG_PATH = figure_path("07-pareto-analysis.png")
PDF_PATH = None
RUN_BACKEND = "numba"
RUN_TEST_NR = 32
RUN_TEST_NT = 32
RUN_REPEAT_COUNT = 10
# Script entrypoint mode.  ``DEFAULT_SWEEP_MODE`` below only applies to helper
# calls, while ``main()`` reads this value directly.
RUN_SWEEP_MODE = "full"
RUN_RECOMPUTE_FULL = False
RUN_CASE_WORKERS = 4
RUN_CASE_WORKER_INNER_THREADS = 1
RUN_FRONTIER_MIN_REL_IMPROVEMENT = 0.05
RUN_RANDOM_SIGNATURE_COUNT = 5
RUN_RANDOM_SIGNATURE_SEED = 20260407
# Selected reduced rows target normalized shape errors E_ref/a.  D-shape has
# enough low-order accuracy to use a tighter Medium/High ladder than the
# GEQDSK-derived H-mode and X-point cases.
CASE_FASTEST_CONFIG_ERROR_THRESHOLDS = {
    "solovev": (1.0e-2, 1.0e-3, 1.0e-4),
    "chease": (1.0e-2, 5.0e-3, 1.0e-3),
    "efit": (1.0e-2, 5.0e-3, 1.0e-3),
}

GRID_ALPHA = 0.2
GRID_LINESTYLE = "-"
GRID_LINE_WIDTH = 0.5
TOP_SPINE_VISIBLE = True
RIGHT_SPINE_VISIBLE = True
LEGEND_FRAME_ON = False
LEGEND_LOC = "upper right"
LEGEND_NCOL = 1
LEGEND_COLUMN_SPACING = 0.8
LEGEND_LABEL_SPACING = 0.15

BACKGROUND_MARKER_SIZE = 5
BACKGROUND_MARKER_ALPHA = 0.22
FRONTIER_LINE_WIDTH = 1.75
CONFIG_MARKER_SIZE = 30
CONFIG_MARKER_EDGE_WIDTH = 0.7
CONFIG_MARKER_FONT_SIZE = 7.0
TIME_CONFIG_MARKER_LABEL_OFFSETS = {
    ("solovev", "Low"): (-20, 0),
    ("solovev", "Medium"): (-16, -12),
    ("solovev", "High"): (-18, -7),
    ("chease", "Low"): (-22, 46),
    ("chease", "Medium"): (3, 46),
    ("chease", "High"): (18, -16),
    ("efit", "Low"): (-22, -3),
    ("efit", "Medium"): (-19, -10),
    ("efit", "High"): (-28, -5),
}
REPRESENTATIVE_ERROR_THRESHOLD = 1.0e-3
ERROR_AXIS_YMIN = 1.0e-5
ERROR_AXIS_YMAX = 1.0
PARAMETER_COUNT_PLOT_MAX = 110
SURFACE_COMPARE_REFERENCE_COLOR = "black"
SURFACE_COMPARE_REFERENCE_STYLE = "--"
SURFACE_COMPARE_REPRESENTATIVE_STYLE = "-"
SURFACE_COMPARE_LINE_WIDTH = 1.0
SURFACE_COMPARE_BOUNDARY_LINE_WIDTH = 1.35
SURFACE_COMPARE_REFERENCE_SCALE = 1.5
SURFACE_COMPARE_PAD_FRACTION = 0.07
SURFACE_COMPARE_XPAD_EXTRA_FRACTION = 0.12
SURFACE_COMPARE_X_TICK_BINS = 2
SURFACE_COMPARE_Y_TICK_BINS = 4
SURFACE_COMPARE_X_TICKS = {
    "solovev": (4.0, 8.0),
    "chease": (0.5, 1.5),
    "efit": (1.0, 2.0),
}
TIME_XMIN = 0.5
TIME_XMAX = 2.0e2
REFERENCE_SOLVE_NR = 32
REFERENCE_SOLVE_NT = 32
REFERENCE_LAYOUT_NR = REFERENCE_SOLVE_NR
REFERENCE_LAYOUT_NT = REFERENCE_SOLVE_NT
REFERENCE_LAYOUT_M_MAX = 10
ERROR_UNIT = "hybr"
# The aggregate error is the theta-resolved geometric flux-surface shape RMS in metres.
AGGREGATE_METRIC_NAME = "theta-surface-shape-rms-meters-resampled-ref"
REFERENCE_VALIDATION_ATOL = 1.0e-8
D_SHAPE_MIN_LENGTHS = {
    "psin": 1,
    "h": 1,
    "k": 1,
    "s1": 1,
}
H_MODE_MIN_LENGTHS = {
    "psin": 1,
    "h": 1,
    "k": 1,
    "v": 1,
    "c0": 1,
    "s1": 1,
}
X_POINT_MIN_LENGTHS = {
    "psin": 1,
    "h": 1,
    "k": 1,
    "v": 1,
    "c0": 1,
    "s1": 1,
}
CASE_MIN_LENGTHS = {
    "solovev": D_SHAPE_MIN_LENGTHS,
    "chease": H_MODE_MIN_LENGTHS,
    "efit": X_POINT_MIN_LENGTHS,
}
CASE_FRONTIER_CURVE_SAMPLES = 24
SURFACE_LEVELS = tuple(np.linspace(0.1, 1.0, 10))
SURFACE_COMPARE_LEVELS = (0.2, 0.4, 0.6, 0.8, 1.0)
SHAPE_RMS_PSIN_LEVELS = tuple(np.linspace(0.0, 1.0, 11, dtype=np.float64))
REFERENCE_SURFACE_NR = 128
REFERENCE_SURFACE_NT = 256
ERROR_SURFACE_NR = 128
ERROR_SURFACE_NT = 256
ERROR_THETA_SAMPLE_COUNT = 16
CORE_FAMILIES = ("psin", "h", "k", "v")
FOURIER_FAMILIES = (
    "c0",
    "s1",
    "c1",
    "s2",
    "c2",
    "s3",
    "c3",
    "s4",
    "c4",
    "s5",
    "c5",
    "s6",
)
PLOT_EPS = 1.0e-12
FRONTIER_MIN_REL_ERROR_IMPROVEMENT = 0.05
FRONTIER_MIN_REL_ERROR = 1.0e-5
FRONTIER_MAX_REL_ERROR = 1.0e-1
RANDOM_SIGNATURE_COUNT = 5
RANDOM_SIGNATURE_SEED = 20260407
MIN_FOURIER_PREFIX = 2
INITIAL_SOLVE_TIMEOUT_S = 1.0
FULL_SWEEP_SIGNATURE_VERSION = "representative-pruned-v2"
FULL_SWEEP_MAX_CONFIGS_PER_CASE = 10000
D_SHAPE_FULL_SINE_SAMPLES_PER_CORE = 8
GENERAL_REF_PRUNE_CORE_SAMPLE_COUNT = 4800
GENERAL_REF_PRUNE_FOURIER_SAMPLES_PER_CORE = 2
REF_PRUNE_HALTON_BASES = (
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
)
DEFAULT_CASE_WORKERS = 4
DEFAULT_CASE_WORKER_INNER_THREADS = 1
REFERENCE_CHECK_SUPPRESS_ENV = "VEQPY_FIG07_SUPPRESS_REFERENCE_CHECK_PRINT"

STRATEGY_LABELS = {
    "balanced_unlock": "balanced-unlock",
    "core_first": "core-first",
    "fourier_first": "fourier-first",
    "core_shells": "core-shells",
    "harmonic_pairs": "harmonic-pairs",
    "shape_first": "shape-first",
    "random_collision": "random-collision",
}

SWEEP_MODES = ("partial", "full")
DEFAULT_REPEAT_COUNT = 10
DEFAULT_SWEEP_MODE = "partial"

# ``partial`` mode is intentionally not a cache-backed sweep.  It always reruns
# only the three selected representative configurations for each case so the
# tabulated rows can be refreshed cheaply without reading the full Pareto cache
# or writing a partial Pareto cache.
TABLE05_SELECTED_SIGNATURES: dict[str, tuple[dict[str, int], ...]] = {
    "solovev": (
        {"psin": 1, "h": 1, "k": 1, "s1": 1},
        {"psin": 1, "h": 1, "k": 2, "s1": 1},
        {"psin": 4, "h": 2, "k": 2, "s1": 2},
    ),
    "chease": (
        {
            "psin": 6,
            "h": 6,
            "k": 4,
            "v": 1,
            "c0": 4,
            "s1": 3,
            "c1": 1,
            "s2": 1,
            "c2": 1,
            "s3": 1,
        },
        {
            "psin": 3,
            "h": 8,
            "k": 5,
            "v": 5,
            "c0": 4,
            "s1": 4,
            "c1": 2,
            "s2": 2,
            "c2": 1,
            "s3": 1,
            "c3": 1,
            "s4": 1,
            "c4": 1,
            "s5": 1,
        },
        {
            "psin": 8,
            "h": 7,
            "k": 6,
            "v": 7,
            "c0": 6,
            "s1": 6,
            "c1": 5,
            "s2": 5,
            "c2": 3,
            "s3": 3,
            "c3": 2,
            "s4": 2,
            "c4": 2,
            "s5": 2,
            "c5": 1,
            "s6": 1,
        },
    ),
    "efit": (
        {
            "psin": 4,
            "h": 5,
            "k": 3,
            "v": 2,
            "c0": 2,
            "s1": 2,
            "c1": 1,
            "s2": 1,
        },
        {
            "psin": 3,
            "h": 4,
            "k": 4,
            "v": 5,
            "c0": 2,
            "s1": 2,
            "c1": 2,
            "s2": 2,
            "c2": 2,
            "s3": 2,
            "c3": 1,
            "s4": 1,
        },
        {
            "psin": 7,
            "h": 8,
            "k": 9,
            "v": 7,
            "c0": 9,
            "s1": 9,
            "c1": 5,
            "s2": 5,
            "c2": 5,
            "s3": 5,
            "c3": 5,
            "s4": 5,
            "c4": 5,
            "s5": 5,
            "c5": 2,
            "s6": 2,
            "c6": 1,
            "s7": 1,
        },
    ),
}


# This H-mode branch has low radial-shape RMS in the Pareto data, but its
# solved normalized-flux mapping is non-monotone after resampling
# (min psin_r <= 0). Exclude it from the standard Low/Medium/High rows
# used by downstream diagnostics while keeping the raw Pareto cloud intact.
EXCLUDED_STANDARD_SIGNATURES: dict[str, tuple[dict[str, int], ...]] = {
    "chease": (
        {
            "psin": 5,
            "h": 7,
            "k": 6,
            "v": 3,
            "c0": 3,
            "s1": 5,
            "c1": 2,
            "s2": 5,
            "c2": 2,
            "s3": 3,
        },
    ),
}


def is_excluded_standard_signature(case_key: str, signature: dict[str, int]) -> bool:
    signature_tuple = signature_key(
        {name: int(value) for name, value in signature.items() if int(value) > 0}
    )
    return any(
        signature_tuple
        == signature_key({name: int(value) for name, value in excluded.items() if int(value) > 0})
        for excluded in EXCLUDED_STANDARD_SIGNATURES.get(case_key, ())
    )


FULL_SELECTED_BRANCHES: dict[str, tuple[dict[str, tuple[int, ...]], ...]] = {
    "chease": (
        {
            "psin": (5,),
            "h": (6, 7, 8),
            "k": (6, 7, 8),
            "v": (2, 3, 4),
            "c0": (3,),
            "s1": (5,),
            "c1": (1, 2),
            "s2": (5,),
            "c2": (1, 2),
            "s3": (3,),
        },
    ),
    "efit": (
        {
            "psin": (2,),
            "h": (7, 8, 9),
            "k": (4, 5, 6),
            "v": (9, 10),
            "c0": (5,),
            "s1": (5,),
            "c1": (5,),
            "s2": (5,),
            "c2": (5,),
            "s3": (5,),
            "c3": (4, 5),
            "s4": (4, 5),
            "c4": (4, 5),
            "s5": (1, 2, 3),
            "c5": (1, 2),
            "s6": (4, 5),
        },
    ),
}


@dataclass(frozen=True)
class ReferenceCase:
    case_key: str
    label: str
    boundary: object
    geqdsk: object
    equilibrium: object
    surface_equilibrium: object
    exact_equilibrium: object
    ref_profiles: dict[str, np.ndarray | float]
    rho_interp_axis: object
    psin_interp_axis: object
    exact_shape_x: np.ndarray
    exact_elapsed_ms: float
    exact_nfev: int
    exact_nit: int
    exact_residual_norm_final: float
    reference_a: float


@dataclass(frozen=True)
class PrecomputedReference:
    path: str
    equilibrium: object
    profile_lengths: dict[str, int]
    shape_x: np.ndarray
    validation_max_abs_error: float


@dataclass(frozen=True)
class Sample:
    case_key: str
    parameter_count: int
    elapsed_ms: float
    aggregate_rel_error: float
    shape_rel_error: float
    surface_rel_rms_error: float
    psi_r_rel_rms_error: float
    ff_psi_rel_rms_error: float
    mu0_p_psi_rel_rms_error: float
    q_rel_rms_error: float
    nfev: int
    nit: int
    residual_norm_final: float
    strategy_name: str
    strategy_names: list[str]
    signature: dict[str, int]
    sweep_step: str
    is_exact_reference: bool


@dataclass(frozen=True)
class SignatureRecord:
    strategy_name: str
    strategy_names: tuple[str, ...]
    sweep_step: str
    signature: dict[str, int]


class InitialSolveTimeoutError(RuntimeError):
    def __init__(self, *, elapsed_ms: float, timeout_s: float):
        self.elapsed_ms = float(elapsed_ms)
        self.timeout_s = float(timeout_s)
        super().__init__(
            f"Initial solve exceeded {self.timeout_s:.{FIXED_DECIMALS}f} s "
            f"({self.elapsed_ms:.{FIXED_DECIMALS}f} ms)"
        )


@dataclass
class CaseProgressState:
    label: str
    total: int = 0
    completed: int = 0
    skipped: int = 0
    active: bool = False
    finished: bool = False
    message: str = "pending"
    started_monotonic: float | None = None
    elapsed_seconds: float | None = None


class ParetoProgressDisplay:
    def __init__(self, case_keys: tuple[str, ...]):
        self.case_keys = tuple(case_keys)
        self.states = {
            case_key: CaseProgressState(label=CASE_LABELS[case_key]) for case_key in self.case_keys
        }
        self._interactive = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._rendered_once = False
        self._last_render_monotonic = 0.0
        self._min_render_interval_s = 0.75

    def preparing(self, case_key: str) -> None:
        state = self.states[case_key]
        state.active = True
        state.message = "preparing reference"
        state.started_monotonic = time.monotonic()
        state.elapsed_seconds = 0.0

    def start_case(
        self,
        case_key: str,
        *,
        total: int,
        completed: int = 0,
    ) -> None:
        state = self.states[case_key]
        state.active = True
        state.total = int(total)
        state.completed = int(completed)
        state.skipped = 0
        state.finished = False
        state.message = "running" if state.completed < state.total else "done"
        state.started_monotonic = time.monotonic()
        state.elapsed_seconds = 0.0
        self._render(force=True)

    def set_total(self, case_key: str, *, total: int) -> None:
        state = self.states[case_key]
        state.total = int(total)
        if self._rendered_once:
            self._render(force=False)

    def update(
        self,
        case_key: str,
        *,
        elapsed_ms: float | None,
        aggregate_rel_error: float | None,
        skipped: bool = False,
    ) -> None:
        state = self.states[case_key]
        state.completed += 1
        if skipped:
            state.skipped += 1
            state.message = "skip >1s"
        else:
            state.message = "running"
        if state.started_monotonic is not None:
            state.elapsed_seconds = max(0.0, time.monotonic() - float(state.started_monotonic))
        self._render(force=False)

    def finish_case(self, case_key: str) -> None:
        state = self.states[case_key]
        state.active = False
        state.finished = True
        state.message = "done"
        if state.started_monotonic is not None:
            state.elapsed_seconds = max(0.0, time.monotonic() - float(state.started_monotonic))
        self._render(force=True)

    def close(self) -> None:
        if self._rendered_once:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _render(self, *, force: bool) -> None:
        if not self._interactive and not force:
            return
        now = time.monotonic()
        if (
            self._interactive
            and not force
            and self._rendered_once
            and (now - self._last_render_monotonic) < self._min_render_interval_s
        ):
            return
        lines = [self._format_line(case_key) for case_key in self.case_keys]
        if self._interactive and self._rendered_once:
            sys.stdout.write(f"\x1b[{len(lines)}F")
        if self._interactive:
            sys.stdout.write("".join(f"\x1b[2K{line}\n" for line in lines))
        else:
            sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        self._rendered_once = True
        self._last_render_monotonic = now

    def _format_line(self, case_key: str) -> str:
        state = self.states[case_key]
        total = max(int(state.total), 1)
        progress = min(max(state.completed / total, 0.0), 1.0) if state.total else 0.0
        width = 24
        filled = int(round(progress * width))
        bar = "#" * filled + "-" * (width - filled)
        count_text = f"{state.completed:>4d}/{state.total:<4d}" if state.total else "   0/0   "
        suffix_parts = [state.message]
        if state.total:
            suffix_parts.append(f"{progress * 100.0:5.{FIXED_DECIMALS}f}%")
        if state.skipped:
            suffix_parts.append(f"skip {state.skipped}")
        elapsed_seconds = state.elapsed_seconds
        if state.active and state.started_monotonic is not None:
            elapsed_seconds = max(
                float(elapsed_seconds or 0.0),
                time.monotonic() - float(state.started_monotonic),
            )
        if elapsed_seconds is not None:
            suffix_parts.append(f"{elapsed_seconds:6.{FIXED_DECIMALS}f} s")
        status = " | ".join(suffix_parts)
        return f"{state.label:<8} [{bar}] {count_text} {status}"


def sample_signature_key(case_key: str, signature: dict[str, int]) -> str:
    ordered = ",".join(f"{name}:{int(length)}" for name, length in sorted(signature.items()))
    return f"{case_key}|{ordered}"


@dataclass(frozen=True)
class BenchmarkCaseSpec:
    mode: str
    coordinate: str
    constraint: str
    input_kind: str

    @property
    def case_name(self) -> str:
        return f"{self.mode}_{self.coordinate}_{self.input_kind}_{self.constraint}"


def _extract_shape_x(profile_specs: dict[str, object], x: np.ndarray) -> np.ndarray:
    components = _load_veqpy_components()
    profile_names = components["build_profile_names"](REFERENCE_LAYOUT_M_MAX)
    profile_index = components["build_profile_index"](profile_names)
    _, coeff_index, _ = components["build_profile_layout"](
        active_profiles_from_coeffs(profile_specs), profile_names=profile_names
    )
    shape_values: list[float] = []
    for k in range(coeff_index.shape[1]):
        for name in SHAPE_PROFILE_NAMES:
            idx = int(coeff_index[profile_index[name], k])
            if idx >= 0:
                shape_values.append(float(x[idx]))
    return np.asarray(shape_values, dtype=np.float64)


def build_pf_reference_profiles(equilibrium) -> dict[str, np.ndarray | float]:
    psin_r = _as_float64_array(equilibrium.psin_r, copy=True)
    psin_r_safe = np.where(np.abs(psin_r) > 1e-14, psin_r, 1e-14)

    psi_r = _as_float64_array(equilibrium.alpha2 * psin_r)
    psi_r_safe = np.where(np.abs(psi_r) > 1e-14, psi_r, 1e-14)

    FFn_r = _as_float64_array(equilibrium.FFn_r, copy=True)
    Pn_r = _as_float64_array(equilibrium.Pn_r, copy=True)
    FF_r = _as_float64_array(equilibrium.FF_r, copy=True)
    P_r = _as_float64_array(equilibrium.P_r, copy=True)
    Itor = _as_float64_array(equilibrium.Itor, copy=True)
    jtor = _as_float64_array(equilibrium.jtor, copy=True)
    jpara = _as_float64_array(equilibrium.jpara, copy=True)
    q = _as_float64_array(equilibrium.q, copy=True)
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
        "scaled_Ip": float(MU0 * equilibrium.Ip),
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
    return ref[key]


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
    driver_normalized = driver_domain == "normalized"
    pressure_normalized = pressure_domain == "normalized"
    return {
        "current_input": _pick_ref_profile(ref, driver_keys[0], driver_keys[1], driver_normalized),
        "heat_input": _pick_ref_profile(
            ref, pressure_keys[0], pressure_keys[1], pressure_normalized
        ),
    }


def load_benchmark(backend: str):
    """Return the small benchmark API that this figure script needs.

    The small benchmark helper API is kept local so this paper script is
    runnable from ``~/veqpy-Zhang2026`` without relying on another checkout's
    regression-suite layout.
    """
    os.environ["VEQPY_BACKEND"] = str(backend)
    components = _load_veqpy_components()
    reference_grid = components["Grid"](
        Nr=REFERENCE_LAYOUT_NR, Nt=REFERENCE_LAYOUT_NT, quadrature_scheme="legendre"
    )
    config = components["SolverConfig"](
        method="hybr",
        max_evaluations=REFERENCE_SOLVER_MAXFEV,
        initial_policy=SOLVER_INITIAL_POLICY,
        enable_verbose=False,
        enable_fallback=False,
        enable_history=False,
    )
    return SimpleNamespace(
        Grid=components["Grid"],
        Operator=components["Operator"],
        Problem=components["Problem"],
        Solver=components["Solver"],
        BenchmarkCaseSpec=BenchmarkCaseSpec,
        CONFIG=config,
        REFERENCE_GRID=reference_grid,
        SHAPE_PROFILE_NAMES=SHAPE_PROFILE_NAMES,
        _extract_shape_x=_extract_shape_x,
        build_pf_reference_profiles=build_pf_reference_profiles,
        _prepare_interp_axis=_prepare_interp_axis,
        _profile_interp=_profile_interp,
        _build_mode_init_kwargs=_build_mode_init_kwargs,
    )


def reference_a_value(equilibrium) -> float:
    return max(float(equilibrium.a), 1.0e-12)


def profile_coeffs_from_lengths(lengths: dict[str, int]) -> dict[str, list[float]]:
    return {name: [0.0] * int(length) for name, length in lengths.items() if int(length) > 0}


def reference_profile_lengths_for_case(case_key: str) -> dict[str, int]:
    try:
        lengths = CASE_REFERENCE_PROFILE_LENGTHS[str(case_key)]
    except KeyError as exc:
        raise KeyError(
            f"Missing Figure 06 reference profile lengths for case {case_key!r}"
        ) from exc
    return {str(name): int(length) for name, length in lengths.items() if int(length) > 0}


def shape_profile_coefficients(equilibrium) -> dict[str, np.ndarray]:
    profiles = getattr(equilibrium, "shape_profiles", None)
    if not profiles:
        return {}
    coeffs: dict[str, np.ndarray] = {}
    for name, profile in dict(profiles).items():
        coeff = None if profile is None else getattr(profile, "coeff", None)
        if coeff is None:
            continue
        coeffs[str(name)] = np.asarray(coeff, dtype=np.float64)
    return coeffs


def infer_reference_profile_lengths_from_equilibrium(
    equilibrium, case_key: str | None = None
) -> dict[str, int]:
    if case_key is not None:
        return reference_profile_lengths_for_case(case_key)

    lengths: dict[str, int] = {}
    # ``Equilibrium`` stores solved shape profiles as coefficient-bearing
    # ``Profile`` snapshots.  Figure 06 reference cases still use the explicit
    # high-order length map above via ``case_key`` so the sweep does not infer
    # active topology from runtime/source fields.
    lengths["psin"] = int(equilibrium.grid.L_max) // 2
    for name, coeff in shape_profile_coefficients(equilibrium).items():
        coeff_size = int(coeff.size)
        if coeff_size > 0:
            lengths[name] = coeff_size
    if len(lengths) <= 1:
        raise AttributeError(
            "Equilibrium snapshot does not contain coefficient-bearing shape_profiles; "
            "pass case_key for Figure 06 reference cases."
        )
    return lengths


def load_precomputed_reference_json(case_key: str) -> tuple[str, object, dict[str, int]]:
    try:
        path = CASE_REFERENCE_EQUILIBRIUM_JSONS[case_key]
    except KeyError as exc:
        raise KeyError(f"Missing Figure 06 equilibrium JSON path for case {case_key!r}") from exc
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing precomputed Figure 06 reference equilibrium JSON: {path}. "
            "Run `python scripts/06-high-order-reconstructions.py` first; "
            "Figure 07 does not regenerate these references."
        )
    equilibrium = _load_veqpy_components()["Equilibrium"].load(path)
    lengths = infer_reference_profile_lengths_from_equilibrium(equilibrium, case_key=case_key)
    if (
        int(equilibrium.grid.Nr) != REFERENCE_SOLVE_NR
        or int(equilibrium.grid.Nt) != REFERENCE_SOLVE_NT
    ):
        raise ValueError(
            f"Precomputed Figure 06 equilibrium {path} uses grid "
            f"{int(equilibrium.grid.Nr)}x{int(equilibrium.grid.Nt)}, expected "
            f"{REFERENCE_SOLVE_NR}x{REFERENCE_SOLVE_NT}."
        )
    return path, equilibrium, lengths


def equilibrium_max_abs_difference(reference, candidate) -> float:
    differences: list[float] = []
    for attr in ("R0", "Z0", "B0", "a", "alpha1", "alpha2"):
        differences.append(abs(float(getattr(reference, attr)) - float(getattr(candidate, attr))))
    for attr in ("psin", "psin_r", "psin_rr", "FFn_psin", "Pn_psin"):
        ref_arr = np.asarray(getattr(reference, attr), dtype=np.float64)
        cur_arr = np.asarray(getattr(candidate, attr), dtype=np.float64)
        if ref_arr.shape != cur_arr.shape:
            return float("inf")
        differences.append(float(np.max(np.abs(ref_arr - cur_arr))))
    reference_coeffs = shape_profile_coefficients(reference)
    candidate_coeffs = shape_profile_coefficients(candidate)
    for name in sorted(set(reference_coeffs) | set(candidate_coeffs)):
        ref_arr = reference_coeffs.get(name)
        cur_arr = candidate_coeffs.get(name)
        if ref_arr is None or cur_arr is None:
            return float("inf")
        if ref_arr.shape != cur_arr.shape:
            return float("inf")
        differences.append(float(np.max(np.abs(ref_arr - cur_arr))))
    return max(differences, default=0.0)


def recompute_reference_equilibrium(
    case_key: str, lengths: dict[str, int]
) -> tuple[object, np.ndarray]:
    geqdsk = read_geqdsk(CASE_REFERENCE_GFILES[case_key])
    boundary, _ = build_boundary(
        geqdsk,
        fit_m=CASE_BOUNDARY_FIT_M[case_key],
        fit_n=CASE_BOUNDARY_FIT_N[case_key],
    )
    solver_case = build_solver_case(
        boundary,
        geqdsk,
        profile_coeffs=profile_coeffs_from_lengths(lengths),
    )
    solver, equilibrium, _surface_equilibrium = solve_equilibrium(
        solver_case,
        method=CASE_SOLVER_METHODS[case_key],
    )
    if solver.result is None:
        raise RuntimeError(f"Reference validation solve failed for {case_key}")
    shape_x = _extract_shape_x(profile_coeffs_from_lengths(lengths), solver.result.x)
    return equilibrium, shape_x


@lru_cache(maxsize=None)
def load_validated_precomputed_reference(case_key: str) -> PrecomputedReference:
    path, equilibrium, lengths = load_precomputed_reference_json(case_key)
    recomputed, shape_x = recompute_reference_equilibrium(case_key, lengths)
    validation_error = equilibrium_max_abs_difference(equilibrium, recomputed)
    if not np.isfinite(validation_error) or validation_error > REFERENCE_VALIDATION_ATOL:
        raise ValueError(
            f"Figure 06 reference JSON {path} failed validation for {case_key}: "
            f"max |delta|={validation_error:.{SCIENTIFIC_DECIMALS}e}, "
            f"tolerance={REFERENCE_VALIDATION_ATOL:.{SCIENTIFIC_DECIMALS}e}."
        )
    if os.environ.get(REFERENCE_CHECK_SUPPRESS_ENV) != "1":
        print(
            f"[pareto] Figure 06 reference check {CASE_LABELS[case_key]}: "
            f"max |delta|={validation_error:.{SCIENTIFIC_DECIMALS}e}"
        )
    return PrecomputedReference(
        path=path,
        equilibrium=equilibrium,
        profile_lengths=lengths,
        shape_x=shape_x,
        validation_max_abs_error=float(validation_error),
    )


def build_reference_profile_coeffs(case_key: str) -> dict[str, list[float]]:
    reference = load_validated_precomputed_reference(case_key)
    return profile_coeffs_from_lengths(reference.profile_lengths)


def load_precomputed_reference_equilibrium(case_key: str):
    return load_validated_precomputed_reference(case_key).equilibrium


def extract_shape_x_from_equilibrium(
    equilibrium, profile_coeffs: dict[str, list[float]]
) -> np.ndarray:
    equilibrium_coeffs = shape_profile_coefficients(equilibrium)
    if not equilibrium_coeffs:
        raise AttributeError(
            "Equilibrium snapshot does not contain coefficient-bearing shape_profiles; "
            "use PrecomputedReference.shape_x for Figure 06 references."
        )
    components = _load_veqpy_components()
    profile_names = components["build_profile_names"](REFERENCE_LAYOUT_M_MAX)
    profile_index = components["build_profile_index"](profile_names)
    _, coeff_index, _ = components["build_profile_layout"](
        active_profiles_from_coeffs(profile_coeffs), profile_names=profile_names
    )
    shape_values: list[float] = []
    for k in range(coeff_index.shape[1]):
        for name in SHAPE_PROFILE_NAMES:
            idx = int(coeff_index[profile_index[name], k])
            if idx < 0:
                continue
            coeff = equilibrium_coeffs.get(name)
            if coeff is None:
                raise ValueError(
                    f"Precomputed Figure 06 equilibrium is missing coefficients for {name!r}"
                )
            coeff_arr = np.asarray(coeff, dtype=np.float64)
            if k >= coeff_arr.size:
                raise ValueError(
                    f"Precomputed Figure 06 equilibrium has only {coeff_arr.size} coefficients "
                    f"for {name!r}, expected at least {k + 1}"
                )
            shape_values.append(float(coeff_arr[k]))
    return np.asarray(shape_values, dtype=np.float64)


def _shape_profile_names() -> tuple[str, ...]:
    return _load_veqpy_components()["build_shape_profile_names"](REFERENCE_LAYOUT_M_MAX)


SHAPE_PROFILE_NAMES = _shape_profile_names()


def build_boundary(geqdsk, *, fit_m: int, fit_n: int, boundary_maxtol: float = BOUNDARY_MAXTOL):
    if float(boundary_maxtol) != BOUNDARY_MAXTOL:
        components = _load_veqpy_components()
        fit = components["fit_boundary_params"](
            geqdsk,
            M=int(fit_m),
            N=int(fit_n),
            maxtol=float(boundary_maxtol),
            R0=None,
            Z0=None,
            a=None,
            ka=None,
        )
        boundary = components["Boundary"](
            a=float(fit["a"]),
            R0=float(fit["R0"]),
            Z0=float(fit["Z0"]),
            B0=float(geqdsk.Bt0),
            ka=float(fit["ka"]),
            c_offsets=np.asarray(fit["c_offsets"], dtype=np.float64),
            s_offsets=np.asarray(fit["s_offsets"], dtype=np.float64),
        )
        return boundary, fit
    return build_geqdsk_boundary(
        geqdsk,
        fit_m=fit_m,
        fit_n=fit_n,
        return_fit=True,
    )


def build_solver_case(boundary, geqdsk, *, profile_coeffs: dict[str, list[float]]):
    modules = _load_veqpy_components()
    return modules["Problem"](
        route="PF",
        coordinate="psin",
        nodes="uniform",
        active_profiles=active_profiles_from_coeffs(profile_coeffs),
        boundary=boundary,
        heat_input=np.asarray(geqdsk.P_psi, dtype=np.float64),
        current_input=np.asarray(geqdsk.FF_psi, dtype=np.float64),
        Ip=float(geqdsk.Ip),
    )


def resample_surface_equilibrium(equilibrium):
    modules = _load_veqpy_components()
    plot_grid = modules["Grid"](
        Nr=max(int(REFERENCE_SURFACE_NR), int(equilibrium.grid.Nr)),
        Nt=max(int(REFERENCE_SURFACE_NT), int(equilibrium.grid.Nt)),
        quadrature_scheme="uniform",
        L_max=int(equilibrium.grid.L_max),
        M_max=int(equilibrium.grid.M_max),
    )
    return equilibrium.resample(grid=plot_grid)


def solve_equilibrium(case, *, method: str):
    modules = _load_veqpy_components()
    solve_grid = modules["Grid"](
        Nr=REFERENCE_SOLVE_NR,
        Nt=REFERENCE_SOLVE_NT,
        quadrature_scheme="legendre",
    )
    solver = modules["Solver"](
        operator=modules["Operator"](solve_grid, case),
        config=modules["SolverConfig"](
            method=str(method),
            max_evaluations=REFERENCE_SOLVER_MAXFEV,
            initial_policy=SOLVER_INITIAL_POLICY,
            enable_fallback=False,
            enable_verbose=False,
            enable_history=False,
        ),
    )
    solver.solve(
        enable_verbose=False,
        enable_history=False,
        initial_policy=SOLVER_INITIAL_POLICY,
        enable_fallback=False,
    )
    equilibrium = solver.build_equilibrium()
    return solver, equilibrium, resample_surface_equilibrium(equilibrium)


def result_nfev(result) -> int:
    if hasattr(result, "nfev"):
        return int(result.nfev)
    return int(getattr(result, "function_evaluations", 0))


def result_nit(result) -> int:
    if hasattr(result, "nit"):
        return int(result.nit)
    return int(getattr(result, "iterations", 0))


def build_surface_from_psin(equilibrium, level: float) -> np.ndarray:
    psin = np.asarray(equilibrium.psin, dtype=np.float64)
    rho = np.asarray(equilibrium.rho, dtype=np.float64)
    order = np.argsort(psin)
    psin_unique, unique_idx = np.unique(psin[order], return_index=True)
    rho_level = float(np.interp(level, psin_unique, rho[order][unique_idx]))
    R = np.array(
        [np.interp(rho_level, rho, equilibrium.R[:, idx]) for idx in range(equilibrium.grid.Nt)],
        dtype=np.float64,
    )
    Z = np.array(
        [np.interp(rho_level, rho, equilibrium.Z[:, idx]) for idx in range(equilibrium.grid.Nt)],
        dtype=np.float64,
    )
    return np.column_stack((R, Z))


def axis_position_error(
    reference_axis: tuple[float, float], veqpy_axis: tuple[float, float]
) -> float:
    ref_axis = np.asarray(reference_axis, dtype=np.float64)
    vq_axis = np.asarray(veqpy_axis, dtype=np.float64)
    return float(np.linalg.norm(vq_axis - ref_axis))


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


def get_case_solver_config(benchmark, case_key: str):
    return benchmark.CONFIG


def build_reference_case(benchmark, case_key: str) -> ReferenceCase:
    precomputed_reference = load_validated_precomputed_reference(case_key)
    equilibrium = precomputed_reference.equilibrium
    gfile_path = CASE_REFERENCE_GFILES[case_key]
    geqdsk = read_geqdsk(gfile_path)
    boundary, _ = build_boundary(
        geqdsk,
        fit_m=CASE_BOUNDARY_FIT_M[case_key],
        fit_n=CASE_BOUNDARY_FIT_N[case_key],
    )
    surface_equilibrium = resample_surface_equilibrium(equilibrium)
    exact_shape_x = np.asarray(precomputed_reference.shape_x, dtype=np.float64)

    return ReferenceCase(
        case_key=case_key,
        label=CASE_LABELS[case_key],
        boundary=boundary,
        geqdsk=geqdsk,
        equilibrium=equilibrium,
        surface_equilibrium=surface_equilibrium,
        exact_equilibrium=surface_equilibrium,
        ref_profiles=benchmark.build_pf_reference_profiles(equilibrium),
        rho_interp_axis=benchmark._prepare_interp_axis(
            np.asarray(equilibrium.rho, dtype=np.float64)
        ),
        psin_interp_axis=benchmark._prepare_interp_axis(
            np.asarray(equilibrium.psin, dtype=np.float64)
        ),
        exact_shape_x=exact_shape_x,
        exact_elapsed_ms=0.0,
        exact_nfev=0,
        exact_nit=0,
        exact_residual_norm_final=0.0,
        reference_a=reference_a_value(equilibrium),
    )


def build_exact_shape_x(benchmark, active_coeffs: dict[str, list[float]]) -> np.ndarray:
    shape_values: list[float] = []
    lengths = {name: len(values) for name, values in active_coeffs.items()}
    max_len = max(lengths.values(), default=0)
    for idx in range(max_len):
        for name in benchmark.SHAPE_PROFILE_NAMES:
            values = active_coeffs.get(name)
            if values is None or idx >= len(values):
                continue
            shape_values.append(float(values[idx]))
    return np.asarray(shape_values, dtype=np.float64)


def build_exact_reference_sample(benchmark, reference: ReferenceCase) -> Sample:
    case_key = reference.case_key
    exact_lengths = get_max_lengths(case_key)
    exact_count = int(sum(exact_lengths.values()))
    aggregate, shape_err, surf_err, psi_r_err, ff_psi_err, mu0_p_psi_err, q_err = compute_metrics(
        benchmark,
        reference,
        reference.exact_equilibrium,
        reference.exact_shape_x,
    )
    return Sample(
        case_key=case_key,
        parameter_count=exact_count,
        elapsed_ms=float(reference.exact_elapsed_ms),
        aggregate_rel_error=float(aggregate),
        shape_rel_error=float(shape_err),
        surface_rel_rms_error=float(surf_err),
        psi_r_rel_rms_error=float(psi_r_err),
        ff_psi_rel_rms_error=float(ff_psi_err),
        mu0_p_psi_rel_rms_error=float(mu0_p_psi_err),
        q_rel_rms_error=float(q_err),
        nfev=int(reference.exact_nfev),
        nit=int(reference.exact_nit),
        residual_norm_final=float(reference.exact_residual_norm_final),
        strategy_name="exact_reference",
        strategy_names=["exact_reference"],
        signature={name: int(length) for name, length in exact_lengths.items()},
        sweep_step="exact-reference",
        is_exact_reference=True,
    )


def get_max_lengths(case_key: str) -> dict[str, int]:
    return dict(load_validated_precomputed_reference(case_key).profile_lengths)


def normalize_length_dict(lengths: dict[str, int], *, label: str) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for name, value in lengths.items():
        family = str(name)
        count = int(value)
        if count < 0:
            raise ValueError(f"{label} contains a negative length for {family!r}")
        if count == 0:
            continue
        normalized[family] = count
    return normalized


def get_case_length_bounds(case_key: str) -> tuple[dict[str, int], dict[str, int]]:
    try:
        raw_min = CASE_MIN_LENGTHS[case_key]
    except KeyError as exc:
        raise KeyError(f"Missing configured min-length bounds for case {case_key!r}") from exc
    raw_max = get_max_lengths(case_key)
    min_lengths = normalize_length_dict(raw_min, label=f"{case_key} min-lengths")
    max_lengths = normalize_length_dict(raw_max, label=f"{case_key} max-lengths")
    present_core = [family for family in CORE_FAMILIES if family in max_lengths]
    if not present_core:
        raise ValueError(
            f"{case_key} max-lengths must include at least one core family "
            f"from {list(CORE_FAMILIES)}"
        )
    extra_min = sorted(set(min_lengths) - set(max_lengths))
    if extra_min:
        raise ValueError(
            f"{case_key} min-lengths contain families missing from max-lengths: {extra_min}"
        )
    for family, min_count in min_lengths.items():
        max_count = max_lengths[family]
        if min_count > max_count:
            raise ValueError(
                f"{case_key} has min-length {min_count} > max-length {max_count} "
                f"for family {family!r}"
            )
    return dict(min_lengths), dict(max_lengths)


def merge_active_coeffs(
    base_profile_coeffs: dict[str, list[float] | None],
    active_coeffs: dict[str, list[float]],
) -> dict[str, list[float] | None]:
    merged = {
        name: (None if values is None else list(values))
        for name, values in base_profile_coeffs.items()
    }
    for name, values in active_coeffs.items():
        merged[name] = list(values)
    return merged


def reconstruct_reference_equilibrium(
    benchmark,
    *,
    solver_case,
    active_coeffs: dict[str, list[float]],
    nr: int,
    nt: int,
    scheme: str,
):
    base_profile_coeffs = {
        name: [0.0] * int(length) for name, length in solver_case.active_profiles.items()
    }
    merged_coeffs = merge_active_coeffs(
        base_profile_coeffs,
        active_coeffs,
    )
    exact_case = solver_case.replace(active_profiles=active_profiles_from_coeffs(merged_coeffs))
    grid = benchmark.Grid(
        Nr=int(nr),
        Nt=int(nt),
        quadrature_scheme=str(scheme),
        L_max=int(benchmark.REFERENCE_GRID.L_max),
        M_max=int(benchmark.REFERENCE_GRID.M_max),
    )
    operator = benchmark.Operator(grid, exact_case)
    x_exact = operator.pack_coefficients(coefficients_from_profile_coeffs(merged_coeffs))
    return operator.build_equilibrium(x_exact)


def baseline_signature(max_lengths: dict[str, int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for family in ("psin", "h", "k", "v", "c0", "s1"):
        if family in max_lengths:
            counts[family] = 1
    return counts


def signature_key(signature: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((name, int(length)) for name, length in signature.items()))


def append_signature(
    records: list[tuple[str, dict[str, int]]], step_name: str, counts: dict[str, int]
) -> None:
    records.append((step_name, dict(counts)))


def add_strategy_records(
    catalog: dict[tuple[tuple[str, int], ...], dict[str, object]],
    ordered_keys: list[tuple[tuple[str, int], ...]],
    strategy_name: str,
    records: list[tuple[str, dict[str, int]]],
) -> None:
    for step_name, signature in records:
        key = signature_key(signature)
        if key not in catalog:
            catalog[key] = {
                "strategy_name": strategy_name,
                "strategy_names": [strategy_name],
                "sweep_step": step_name,
                "signature": dict(signature),
            }
            ordered_keys.append(key)
            continue
        strategy_names = catalog[key]["strategy_names"]
        if strategy_name not in strategy_names:
            strategy_names.append(strategy_name)


def fourier_shells(max_lengths: dict[str, int]) -> list[tuple[str, ...]]:
    shells: list[tuple[str, ...]] = []
    idx = 0
    while True:
        c_name = f"c{idx}"
        s_name = f"s{idx + 1}"
        shell = tuple(name for name in (c_name, s_name) if name in max_lengths)
        if not shell:
            break
        shells.append(shell)
        idx += 1
    return shells


def available_fourier(max_lengths: dict[str, int]) -> list[str]:
    families: list[str] = []
    for shell in fourier_shells(max_lengths):
        families.extend(shell)
    return families


def radical_inverse(index: int, base: int) -> float:
    inverse = 0.0
    fraction = 1.0 / float(base)
    current = int(index)
    while current > 0:
        inverse += float(current % int(base)) * fraction
        current //= int(base)
        fraction /= float(base)
    return inverse


def stratified_count(index: int, base: int, lower: int, upper: int) -> int:
    lower_int = int(lower)
    upper_int = int(upper)
    if upper_int < lower_int:
        raise ValueError(f"invalid count range {lower_int}..{upper_int}")
    span = upper_int - lower_int + 1
    value = lower_int + int(np.floor(radical_inverse(index, base) * span))
    return min(upper_int, max(lower_int, value))


def positive_signature(signature: dict[str, int]) -> dict[str, int]:
    return {name: int(length) for name, length in signature.items() if int(length) > 0}


def append_signature_record(
    records: list[SignatureRecord],
    seen: set[tuple[tuple[str, int], ...]],
    *,
    strategy_name: str,
    sweep_step: str,
    signature: dict[str, int],
) -> None:
    normalized = positive_signature(signature)
    key = signature_key(normalized)
    if key in seen:
        return
    seen.add(key)
    records.append(
        SignatureRecord(
            strategy_name=strategy_name,
            strategy_names=(strategy_name,),
            sweep_step=sweep_step,
            signature=normalized,
        )
    )


def d_shape_sine_ref_pruning_signature(
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
    *,
    sample_index: int,
) -> dict[str, int]:
    sine_families = [
        f"s{idx}" for idx in range(1, 1 + REFERENCE_LAYOUT_M_MAX) if f"s{idx}" in max_lengths
    ]
    if not sine_families:
        return {}

    active_depth = stratified_count(
        sample_index,
        REF_PRUNE_HALTON_BASES[0],
        1,
        len(sine_families),
    )
    s1_name = sine_families[0]
    s1_length = stratified_count(
        sample_index,
        REF_PRUNE_HALTON_BASES[1],
        int(min_lengths.get(s1_name, 1)),
        int(max_lengths[s1_name]),
    )
    signature = {s1_name: s1_length}
    if active_depth <= 1:
        return signature

    tail_values: list[int] = []
    tail_cap = max(
        1, min(s1_length, max(int(max_lengths[name]) for name in sine_families[1:active_depth]))
    )
    for offset, name in enumerate(sine_families[1:active_depth], start=2):
        upper = min(tail_cap, int(max_lengths[name]))
        tail_values.append(stratified_count(sample_index, REF_PRUNE_HALTON_BASES[offset], 1, upper))
    tail_values.sort(reverse=True)

    previous = s1_length
    for name, length in zip(sine_families[1:active_depth], tail_values, strict=True):
        previous = min(previous, int(length), int(max_lengths[name]))
        if previous <= 0:
            break
        signature[name] = previous
    return signature


def generate_d_shape_ref_pruning_signatures(
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
) -> list[SignatureRecord]:
    """Deterministically sample D-shape configs by pruning prefixes of the VEQ-ref solution.

    The D-shape reference contains independent radial/source families
    (psin, h, k) and a contiguous sine-harmonic prefix.  A full Cartesian
    monotone-pruning sweep is too large, so this uses the complete 10x10x10
    radial grid and nine low-discrepancy sine-prefix samples per radial point.
    The result stays in the 1e3--1e4 range while covering the min-to-ref box
    evenly and retaining the exact min/ref endpoints.
    """

    required_core = ("psin", "h", "k")
    missing_core = [name for name in required_core if name not in max_lengths]
    if missing_core:
        raise ValueError(
            f"D-shape VEQ-ref pruning is missing required core families: {missing_core}"
        )

    records: list[SignatureRecord] = []
    seen: set[tuple[tuple[str, int], ...]] = set()

    append_signature_record(
        records,
        seen,
        strategy_name="veq_ref_prune_full",
        sweep_step="d-shape-ref-prune-min",
        signature=dict(min_lengths),
    )

    core_ranges = [
        range(int(min_lengths.get(name, 1)), int(max_lengths[name]) + 1) for name in required_core
    ]
    sample_index = 1
    for psin_length, h_length, k_length in itertools.product(*core_ranges):
        core_signature = {
            "psin": int(psin_length),
            "h": int(h_length),
            "k": int(k_length),
        }
        for local_index in range(D_SHAPE_FULL_SINE_SAMPLES_PER_CORE):
            sine_signature = d_shape_sine_ref_pruning_signature(
                min_lengths,
                max_lengths,
                sample_index=sample_index,
            )
            append_signature_record(
                records,
                seen,
                strategy_name="veq_ref_prune_full",
                sweep_step=f"d-shape-ref-prune-core-{psin_length}-{h_length}-{k_length}-s{local_index}",
                signature={**core_signature, **sine_signature},
            )
            sample_index += 1

    append_signature_record(
        records,
        seen,
        strategy_name="veq_ref_prune_full",
        sweep_step="d-shape-ref-prune-ref",
        signature=dict(max_lengths),
    )
    append_representative_neighborhood_records(
        records,
        seen,
        case_key="solovev",
        min_lengths=min_lengths,
        max_lengths=max_lengths,
    )
    return records


def ref_pruning_core_signature(
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
    *,
    sample_index: int,
) -> dict[str, int]:
    core_families = [family for family in CORE_FAMILIES if family in max_lengths]
    signature: dict[str, int] = {}
    for offset, family in enumerate(core_families):
        signature[family] = stratified_count(
            sample_index,
            REF_PRUNE_HALTON_BASES[offset],
            int(min_lengths.get(family, 1)),
            int(max_lengths[family]),
        )
    return signature


def required_fourier_shell_count(min_lengths: dict[str, int], max_lengths: dict[str, int]) -> int:
    required_count = 0
    for idx, shell in enumerate(fourier_shells(max_lengths), start=1):
        if any(int(min_lengths.get(name, 0)) > 0 for name in shell):
            required_count = idx
    return required_count


def paired_fourier_ref_pruning_signature(
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
    *,
    sample_index: int,
) -> dict[str, int]:
    shells = fourier_shells(max_lengths)
    if not shells:
        return {}

    min_active_depth = max(1, required_fourier_shell_count(min_lengths, max_lengths))
    active_depth = stratified_count(
        sample_index,
        REF_PRUNE_HALTON_BASES[4],
        min_active_depth,
        len(shells),
    )

    signature: dict[str, int] = {}
    previous_cap = max(int(max_lengths[name]) for name in shells[0])
    for shell_idx, shell in enumerate(shells[:active_depth]):
        floor = max(1, max(int(min_lengths.get(name, 0)) for name in shell))
        cap = min(previous_cap, min(int(max_lengths[name]) for name in shell))
        if floor > cap:
            break
        length = stratified_count(
            sample_index,
            REF_PRUNE_HALTON_BASES[5 + shell_idx],
            floor,
            cap,
        )
        for name in shell:
            count = min(int(length), int(max_lengths[name]))
            if count > 0:
                signature[name] = count
        previous_cap = int(length)
    return signature


def append_ref_pruning_selected_branch_records(
    records: list[SignatureRecord],
    seen: set[tuple[tuple[str, int], ...]],
    *,
    case_key: str,
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
) -> None:
    append_representative_neighborhood_records(
        records,
        seen,
        case_key=case_key,
        min_lengths=min_lengths,
        max_lengths=max_lengths,
    )


def normalize_representative_signature(
    case_key: str,
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
    signature: dict[str, int],
) -> dict[str, int]:
    normalized = {family: int(floor) for family, floor in min_lengths.items() if int(floor) > 0}
    for family, raw_count in signature.items():
        if family not in max_lengths:
            raise ValueError(
                f"Representative signature for {case_key!r} uses unknown family {family!r}"
            )
        count = int(raw_count)
        floor = int(min_lengths.get(family, 0))
        ceiling = int(max_lengths[family])
        if count < floor or count > ceiling:
            raise ValueError(
                f"Representative signature for {case_key!r} "
                f"uses invalid {family}={count}; expected {floor}..{ceiling}"
            )
        if count > 0:
            normalized[family] = count
    for family, floor in min_lengths.items():
        if int(normalized.get(family, 0)) < int(floor):
            raise ValueError(
                f"Representative signature for {case_key!r} misses required family {family!r}"
            )
    return normalized


def nearby_count_values(
    *,
    current: int,
    family: str,
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
) -> tuple[int, ...]:
    floor = max(1 if str(family).startswith(("c", "s")) else 0, int(min_lengths.get(family, 0)))
    ceiling = int(max_lengths[family])
    values = {
        min(max(int(current) - 1, floor), ceiling),
        min(max(int(current) + 1, floor), ceiling),
    }
    values.discard(int(current))
    return tuple(sorted(values))


def append_representative_neighborhood_records(
    records: list[SignatureRecord],
    seen: set[tuple[tuple[str, int], ...]],
    *,
    case_key: str,
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
) -> None:
    """Add compact single-step neighborhoods around selected representative rows.

    The broad Halton/pruning sweep gives global coverage.  These deterministic
    local increments keep the known Low/Medium/High representatives in the
    run set and spend a small extra budget on adjacent core/profile lengths
    that can reveal a better time-error tradeoff without restoring the old
    thousand-scale selected Cartesian branches.
    """

    anchors = [
        normalize_representative_signature(case_key, min_lengths, max_lengths, signature)
        for signature in TABLE05_SELECTED_SIGNATURES.get(case_key, ())
    ]
    for anchor_idx, anchor in enumerate(anchors):
        append_signature_record(
            records,
            seen,
            strategy_name="veq_ref_prune_full_selected",
            sweep_step=f"{case_key}-representative-{anchor_idx}",
            signature=anchor,
        )
        active_families = tuple(
            family
            for family in (*CORE_FAMILIES, *available_fourier(max_lengths))
            if family in max_lengths
        )
        for family in active_families:
            current = int(anchor.get(family, min_lengths.get(family, 0)))
            if current <= 0:
                continue
            for value in nearby_count_values(
                current=current,
                family=family,
                min_lengths=min_lengths,
                max_lengths=max_lengths,
            ):
                variant = dict(anchor)
                variant[family] = int(value)
                append_signature_record(
                    records,
                    seen,
                    strategy_name="veq_ref_prune_full_selected_nearby",
                    sweep_step=f"{case_key}-representative-{anchor_idx}-{family}-{value}",
                    signature=variant,
                )

        core_families = tuple(
            family for family in CORE_FAMILIES if family in anchor and family in max_lengths
        )
        for delta in (-1, 1):
            variant = dict(anchor)
            changed = False
            for family in core_families:
                floor = int(min_lengths.get(family, 0))
                ceiling = int(max_lengths[family])
                value = min(max(int(anchor[family]) + delta, floor), ceiling)
                if value != int(anchor[family]):
                    variant[family] = value
                    changed = True
            if changed:
                append_signature_record(
                    records,
                    seen,
                    strategy_name="veq_ref_prune_full_selected_nearby",
                    sweep_step=f"{case_key}-representative-{anchor_idx}-core{delta:+d}",
                    signature=variant,
                )

        for shell_idx, shell in enumerate(fourier_shells(max_lengths)):
            if not any(family in anchor for family in shell):
                continue
            current = max(int(anchor.get(family, 0)) for family in shell)
            for value in nearby_count_values(
                current=current,
                family=shell[0],
                min_lengths=min_lengths,
                max_lengths=max_lengths,
            ):
                variant = dict(anchor)
                for family in shell:
                    variant[family] = min(int(value), int(max_lengths[family]))
                append_signature_record(
                    records,
                    seen,
                    strategy_name="veq_ref_prune_full_selected_nearby",
                    sweep_step=f"{case_key}-representative-{anchor_idx}-shell{shell_idx}-{value}",
                    signature=variant,
                )


def generate_general_ref_pruning_signatures(
    case_key: str,
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
) -> list[SignatureRecord]:
    """Sample H-mode/X-point configs by pruning the Figure 06 VEQ reference.

    Unlike the D-shape case, these references include four core profiles and
    paired cosine/sine shaping shells.  A complete Cartesian pruning grid would
    be too large, so the core lengths and active shaping-shell prefix are
    sampled with a deterministic Halton-style sequence.  The generated set is
    sized above 10k per case, includes the min/ref endpoints, and keeps known
    selected neighborhoods available without reverting to the older
    shell-recursion sweep.  Candidate families remain bounded by the nonzero
    Figure 06 reference lengths, and Fourier families are sampled as contiguous
    monotone prefixes so the sweep does not spend most of its budget on high
    isolated harmonics that are unlikely to define the Pareto frontier.
    """

    records: list[SignatureRecord] = []
    seen: set[tuple[tuple[str, int], ...]] = set()

    append_signature_record(
        records,
        seen,
        strategy_name="veq_ref_prune_full",
        sweep_step=f"{case_key}-ref-prune-min",
        signature=dict(min_lengths),
    )

    sample_index = 1
    for core_sample_index in range(1, GENERAL_REF_PRUNE_CORE_SAMPLE_COUNT + 1):
        core_signature = ref_pruning_core_signature(
            min_lengths,
            max_lengths,
            sample_index=core_sample_index,
        )
        for local_index in range(GENERAL_REF_PRUNE_FOURIER_SAMPLES_PER_CORE):
            fourier_signature = paired_fourier_ref_pruning_signature(
                min_lengths,
                max_lengths,
                sample_index=sample_index,
            )
            append_signature_record(
                records,
                seen,
                strategy_name="veq_ref_prune_full",
                sweep_step=f"{case_key}-ref-prune-core-{core_sample_index}-f{local_index}",
                signature={**core_signature, **fourier_signature},
            )
            sample_index += 1

    append_signature_record(
        records,
        seen,
        strategy_name="veq_ref_prune_full",
        sweep_step=f"{case_key}-ref-prune-ref",
        signature=dict(max_lengths),
    )
    append_ref_pruning_selected_branch_records(
        records,
        seen,
        case_key=case_key,
        min_lengths=min_lengths,
        max_lengths=max_lengths,
    )
    return records


def strategy_balanced_unlock(
    max_lengths: dict[str, int],
) -> list[tuple[str, dict[str, int]]]:
    counts = baseline_signature(max_lengths)
    records: list[tuple[str, dict[str, int]]] = [("baseline", dict(counts))]
    for family in available_fourier(max_lengths)[2:]:
        counts[family] = 1
        append_signature(records, f"{family}-L1", counts)
    all_families = [
        family
        for family in (*CORE_FAMILIES, *available_fourier(max_lengths))
        if family in max_lengths
    ]
    for target_length in range(2, 6):
        for family in all_families:
            if counts.get(family, 0) < min(target_length, max_lengths[family]):
                counts[family] = counts.get(family, 0) + 1
                append_signature(records, f"{family}-L{counts[family]}", counts)
    for target_length in range(6, 11):
        for family in CORE_FAMILIES:
            if family in max_lengths and counts.get(family, 0) < min(
                target_length, max_lengths[family]
            ):
                counts[family] = counts.get(family, 0) + 1
                append_signature(records, f"{family}-L{counts[family]}", counts)
    return records


def strategy_core_first(
    max_lengths: dict[str, int],
) -> list[tuple[str, dict[str, int]]]:
    counts = baseline_signature(max_lengths)
    records: list[tuple[str, dict[str, int]]] = [("baseline", dict(counts))]
    for target_length in range(2, 11):
        for family in CORE_FAMILIES:
            if family in max_lengths and counts.get(family, 0) < min(
                target_length, max_lengths[family]
            ):
                counts[family] = counts.get(family, 0) + 1
                append_signature(records, f"{family}-L{counts[family]}", counts)
    for family in available_fourier(max_lengths)[2:]:
        counts[family] = 1
        append_signature(records, f"{family}-L1", counts)
    for target_length in range(2, 6):
        for family in available_fourier(max_lengths):
            if counts.get(family, 0) < min(target_length, max_lengths[family]):
                counts[family] = counts.get(family, 0) + 1
                append_signature(records, f"{family}-L{counts[family]}", counts)
    return records


def strategy_fourier_first(
    max_lengths: dict[str, int],
) -> list[tuple[str, dict[str, int]]]:
    counts = baseline_signature(max_lengths)
    records: list[tuple[str, dict[str, int]]] = [("baseline", dict(counts))]
    for family in available_fourier(max_lengths)[2:]:
        counts[family] = 1
        append_signature(records, f"{family}-L1", counts)
    for target_length in range(2, 6):
        for family in available_fourier(max_lengths):
            if counts.get(family, 0) < min(target_length, max_lengths[family]):
                counts[family] = counts.get(family, 0) + 1
                append_signature(records, f"{family}-L{counts[family]}", counts)
    for target_length in range(2, 11):
        for family in CORE_FAMILIES:
            if family in max_lengths and counts.get(family, 0) < min(
                target_length, max_lengths[family]
            ):
                counts[family] = counts.get(family, 0) + 1
                append_signature(records, f"{family}-L{counts[family]}", counts)
    return records


def strategy_core_shells(
    max_lengths: dict[str, int],
) -> list[tuple[str, dict[str, int]]]:
    counts = baseline_signature(max_lengths)
    records: list[tuple[str, dict[str, int]]] = [("baseline", dict(counts))]
    fourier = available_fourier(max_lengths)
    for target_length in range(2, 6):
        for family in CORE_FAMILIES:
            if family in max_lengths and counts.get(family, 0) < min(
                target_length, max_lengths[family]
            ):
                counts[family] = target_length
                append_signature(records, f"{family}-L{counts[family]}", counts)
        for family in fourier[2:]:
            if family not in counts:
                counts[family] = 1
                append_signature(records, f"{family}-L1", counts)
        for family in fourier:
            if counts.get(family, 0) < min(target_length, max_lengths[family]):
                counts[family] = target_length
                append_signature(records, f"{family}-L{counts[family]}", counts)
    for target_length in range(6, 11):
        for family in CORE_FAMILIES:
            if family in max_lengths and counts.get(family, 0) < min(
                target_length, max_lengths[family]
            ):
                counts[family] = target_length
                append_signature(records, f"{family}-L{counts[family]}", counts)
    return records


def strategy_harmonic_pairs(
    max_lengths: dict[str, int],
) -> list[tuple[str, dict[str, int]]]:
    counts = baseline_signature(max_lengths)
    records: list[tuple[str, dict[str, int]]] = [("baseline", dict(counts))]
    fourier = available_fourier(max_lengths)
    harmonic_pairs = [fourier[idx : idx + 2] for idx in range(0, len(fourier), 2)]
    for pair in harmonic_pairs[1:]:
        for family in pair:
            counts[family] = 1
        append_signature(records, f"{'+'.join(pair)}-L1", counts)
    for target_length in range(2, 6):
        for pair in harmonic_pairs:
            changed = False
            for family in pair:
                if counts.get(family, 0) < min(target_length, max_lengths[family]):
                    counts[family] = counts.get(family, 0) + 1
                    changed = True
            if changed:
                append_signature(records, f"{'+'.join(pair)}-L{target_length}", counts)
        for family in ("psin", "k", "h", "v"):
            if family in max_lengths and counts.get(family, 0) < min(
                target_length, max_lengths[family]
            ):
                counts[family] = counts.get(family, 0) + 1
                append_signature(records, f"{family}-L{counts[family]}", counts)
    for target_length in range(6, 11):
        for family in ("psin", "k", "h", "v"):
            if family in max_lengths and counts.get(family, 0) < min(
                target_length, max_lengths[family]
            ):
                counts[family] = counts.get(family, 0) + 1
                append_signature(records, f"{family}-L{counts[family]}", counts)
    return records


def strategy_shape_first(
    max_lengths: dict[str, int],
) -> list[tuple[str, dict[str, int]]]:
    counts = baseline_signature(max_lengths)
    records: list[tuple[str, dict[str, int]]] = [("baseline", dict(counts))]
    fourier = available_fourier(max_lengths)
    for family in fourier[2:]:
        counts[family] = 1
        append_signature(records, f"{family}-L1", counts)
    priority = ("k", "h", "psin", "v")
    for target_length in range(2, 6):
        for family in fourier:
            if counts.get(family, 0) < min(target_length, max_lengths[family]):
                counts[family] = counts.get(family, 0) + 1
                append_signature(records, f"{family}-L{counts[family]}", counts)
        for family in priority:
            if family in max_lengths and counts.get(family, 0) < min(
                target_length, max_lengths[family]
            ):
                counts[family] = counts.get(family, 0) + 1
                append_signature(records, f"{family}-L{counts[family]}", counts)
    for target_length in range(6, 11):
        for family in priority:
            if family in max_lengths and counts.get(family, 0) < min(
                target_length, max_lengths[family]
            ):
                counts[family] = counts.get(family, 0) + 1
                append_signature(records, f"{family}-L{counts[family]}", counts)
    return records


def random_admissible_signature(max_lengths: dict[str, int], rng: random.Random) -> dict[str, int]:
    counts = baseline_signature(max_lengths)
    fourier = available_fourier(max_lengths)
    if rng.random() < 0.65:
        prefix = rng.randint(MIN_FOURIER_PREFIX, len(fourier))
        for family in fourier[:prefix]:
            counts.setdefault(family, 1)
        global_level = rng.randint(1, 5)
        for family in (*CORE_FAMILIES, *fourier[:prefix]):
            if family not in max_lengths:
                continue
            lower = max(1, global_level - 1)
            upper = min(global_level, max_lengths[family])
            counts[family] = rng.randint(lower, upper)
    else:
        for family in fourier:
            counts[family] = min(5, max_lengths[family])
        core_level = rng.randint(6, 10)
        for family in CORE_FAMILIES:
            if family not in max_lengths:
                continue
            lower = max(5, core_level - 1)
            upper = min(core_level, max_lengths[family])
            counts[family] = rng.randint(lower, upper)
    return {name: int(length) for name, length in counts.items()}


def add_full_selected_branch_records(
    records: list[SignatureRecord],
    *,
    case_key: str,
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
) -> None:
    branches = FULL_SELECTED_BRANCHES.get(case_key, ())
    if not branches:
        return

    seen = {signature_key(record.signature) for record in records}
    for branch_idx, options in enumerate(branches):
        if not options:
            continue
        for family, values in options.items():
            if family not in max_lengths:
                raise ValueError(
                    f"Full selected branch for {case_key!r} uses unknown family {family!r}"
                )
            if not values:
                raise ValueError(
                    f"Full selected branch for {case_key!r} has no values for family {family!r}"
                )
            for value in values:
                count = int(value)
                if count < int(min_lengths.get(family, 0)) or count > int(max_lengths[family]):
                    raise ValueError(
                        f"Full selected branch for {case_key!r} uses invalid {family}={count}; "
                        f"expected {int(min_lengths.get(family, 0))}..{int(max_lengths[family])}"
                    )

        missing_required = [
            family
            for family, floor in min_lengths.items()
            if int(floor) > 0 and family not in options
        ]
        if missing_required:
            raise ValueError(
                f"Full selected branch for {case_key!r} misses "
                f"required families: {missing_required}"
            )

        families = tuple(options)
        for values in itertools.product(
            *(tuple(int(value) for value in options[family]) for family in families)
        ):
            signature = {family: int(value) for family, value in zip(families, values, strict=True)}
            key = signature_key(signature)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                SignatureRecord(
                    strategy_name="scheme_a_full_selected",
                    strategy_names=("scheme_a_full_selected",),
                    sweep_step=f"full-selected_{case_key}_{branch_idx}_{len(records)}",
                    signature=signature,
                )
            )

    record_keys = {signature_key(record.signature) for record in records}
    for target_signature in TABLE05_SELECTED_SIGNATURES.get(case_key, ()):
        target_key = signature_key(target_signature)
        if target_key not in record_keys:
            raise AssertionError(
                f"Full selected branches for {case_key!r} do not include {target_signature}"
            )


def generate_strategy_signatures(
    case_key: str,
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
    random_signature_count: int,
    random_seed: int,
) -> list[SignatureRecord]:
    del random_signature_count, random_seed
    if case_key == "solovev":
        return generate_d_shape_ref_pruning_signatures(min_lengths, max_lengths)
    return generate_general_ref_pruning_signatures(case_key, min_lengths, max_lengths)


def add_partial_selected_signature(
    records: list[SignatureRecord],
    seen: set[tuple[tuple[str, int], ...]],
    *,
    case_key: str,
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
    signature: dict[str, int],
) -> None:
    normalized = {name: int(length) for name, length in min_lengths.items() if int(length) > 0}
    for name, length in signature.items():
        if name not in max_lengths:
            return
        value = int(length)
        if value < int(min_lengths.get(name, 0)) or value > int(max_lengths[name]):
            return
        if value > 0:
            normalized[name] = value
    for name, floor in min_lengths.items():
        if int(normalized.get(name, 0)) < int(floor):
            return
    key = signature_key(normalized)
    if key in seen:
        return
    seen.add(key)
    records.append(
        SignatureRecord(
            strategy_name="table05_selected",
            strategy_names=("table05_selected",),
            sweep_step=f"table05-selected_{case_key}_{len(records)}",
            signature=normalized,
        )
    )


def generate_partial_strategy_signatures(
    case_key: str,
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
) -> list[SignatureRecord]:
    records: list[SignatureRecord] = []
    seen: set[tuple[tuple[str, int], ...]] = set()

    for signature in TABLE05_SELECTED_SIGNATURES.get(case_key, ()):
        add_partial_selected_signature(
            records,
            seen,
            case_key=case_key,
            min_lengths=min_lengths,
            max_lengths=max_lengths,
            signature=signature,
        )

    return records


def generate_case_signatures(
    case_key: str,
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
    random_signature_count: int,
    random_seed: int,
    *,
    sweep_mode: str,
) -> list[SignatureRecord]:
    sweep_mode = str(sweep_mode)
    if sweep_mode == "full":
        records = generate_strategy_signatures(
            case_key,
            min_lengths,
            max_lengths,
            random_signature_count=random_signature_count,
            random_seed=random_seed,
        )
        if len(records) >= FULL_SWEEP_MAX_CONFIGS_PER_CASE:
            raise ValueError(
                f"{CASE_LABELS[case_key]} full sweep generated {len(records)} configs; "
                f"expected fewer than {FULL_SWEEP_MAX_CONFIGS_PER_CASE}"
            )
        return records
    if sweep_mode == "partial":
        del random_signature_count, random_seed
        return generate_partial_strategy_signatures(case_key, min_lengths, max_lengths)
    raise ValueError(f"Unsupported sweep mode {sweep_mode!r}; expected one of {SWEEP_MODES}")


def make_profile_coeffs(
    signature: dict[str, int],
    *,
    max_lengths: dict[str, int],
) -> dict[str, list[float] | None]:
    profile_coeffs: dict[str, list[float] | None] = {name: None for name in max_lengths}
    for name, length in signature.items():
        coeff_length = int(length)
        if coeff_length <= 0:
            continue
        profile_coeffs[name] = [0.0] * coeff_length
    return profile_coeffs


def build_pf_case(benchmark, reference: ReferenceCase, grid, signature: dict[str, int]):
    _, max_lengths = get_case_length_bounds(reference.case_key)
    return benchmark.Problem(
        route="PF",
        coordinate="psin",
        nodes="uniform",
        active_profiles=active_profiles_from_coeffs(
            make_profile_coeffs(signature, max_lengths=max_lengths)
        ),
        boundary=reference.boundary,
        heat_input=np.asarray(reference.geqdsk.P_psi, dtype=np.float64),
        current_input=np.asarray(reference.geqdsk.FF_psi, dtype=np.float64),
        Ip=float(reference.geqdsk.Ip),
        beta=None,
    )


def solve_with_timing(
    benchmark,
    case,
    grid,
    repeat_count: int,
    method: str,
    solver_config,
    *,
    initial_solve_timeout_s: float = INITIAL_SOLVE_TIMEOUT_S,
):
    solver = benchmark.Solver(
        operator=benchmark.Operator(grid, case),
        config=solver_config,
    )
    solve_kwargs = {
        "method": str(method),
        "max_residual": float(getattr(solver_config, "max_residual", 1.0e-6)),
        "max_evaluations": int(getattr(solver_config, "max_evaluations", REFERENCE_SOLVER_MAXFEV)),
        "enable_verbose": False,
        "enable_history": False,
        "initial_policy": SOLVER_INITIAL_POLICY,
        "enable_fallback": False,
    }
    probe_started = time.perf_counter()
    solver.solve(**solve_kwargs)
    probe_elapsed_ms = (time.perf_counter() - probe_started) * 1000.0
    if solver.result is not None:
        probe_elapsed_ms = max(
            probe_elapsed_ms,
            float(solver.result.elapsed) / 1000.0,
        )
    if probe_elapsed_ms > float(initial_solve_timeout_s) * 1000.0:
        raise InitialSolveTimeoutError(
            elapsed_ms=probe_elapsed_ms,
            timeout_s=initial_solve_timeout_s,
        )
    elapsed_values: list[float] = []
    final_result = None
    for _ in range(max(int(repeat_count), 1)):
        solver.solve(**solve_kwargs)
        final_result = solver.result
        elapsed_values.append(float(final_result.elapsed) / 1000.0)
    if final_result is None:
        raise RuntimeError("No solver result returned")
    final_eq = solver.build_equilibrium()
    return final_result, final_eq, float(np.median(elapsed_values))


_CASE_WORKER_STATE: dict[str, object] = {}


def set_inner_thread_environment(inner_threads: int) -> None:
    thread_count = max(int(inner_threads), 1)
    value = str(thread_count)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[name] = value


def clamp_case_workers(case_workers: int, inner_threads: int) -> int:
    requested = max(int(case_workers), 1)
    if requested <= 1:
        return 1
    cpu_count = os.cpu_count() or 1
    per_worker_threads = max(int(inner_threads), 1)
    safe_cap = max(1, int(cpu_count) // per_worker_threads)
    return max(1, min(requested, safe_cap))


def _case_worker_initializer(
    backend: str,
    case_key: str,
    test_nr: int,
    test_nt: int,
    repeat_count: int,
    inner_threads: int,
) -> None:
    os.environ[REFERENCE_CHECK_SUPPRESS_ENV] = "1"
    set_inner_thread_environment(inner_threads)
    try:
        import numba

        numba.set_num_threads(max(int(inner_threads), 1))
    except Exception:
        pass

    benchmark = load_benchmark(str(backend))
    reference = build_reference_case(benchmark, str(case_key))
    grid = benchmark.Grid(
        Nr=int(test_nr),
        Nt=int(test_nt),
        quadrature_scheme="legendre",
        L_max=int(benchmark.REFERENCE_GRID.L_max),
        M_max=int(benchmark.REFERENCE_GRID.M_max),
    )
    _CASE_WORKER_STATE.clear()
    _CASE_WORKER_STATE.update(
        {
            "benchmark": benchmark,
            "case_key": str(case_key),
            "reference": reference,
            "grid": grid,
            "repeat_count": int(repeat_count),
            "solver_config": get_case_solver_config(benchmark, str(case_key)),
            "method": CASE_SOLVER_METHODS[str(case_key)],
        }
    )


def solve_signature_record_worker(
    task: tuple[int, SignatureRecord],
) -> tuple[int, Sample | None, float | None, bool, str | None]:
    idx, record = task
    benchmark = _CASE_WORKER_STATE["benchmark"]
    case_key = str(_CASE_WORKER_STATE["case_key"])
    reference = _CASE_WORKER_STATE["reference"]
    grid = _CASE_WORKER_STATE["grid"]
    repeat_count = int(_CASE_WORKER_STATE["repeat_count"])
    solver_config = _CASE_WORKER_STATE["solver_config"]
    method = str(_CASE_WORKER_STATE["method"])

    sweep_step = record.sweep_step
    signature = record.signature
    case = build_pf_case(benchmark, reference, grid, signature)
    try:
        result, equilibrium, elapsed_ms = solve_with_timing(
            benchmark,
            case,
            grid,
            repeat_count,
            method=method,
            solver_config=solver_config,
        )
        shape_x = benchmark._extract_shape_x(case.active_profiles, result.x)
        aggregate, shape_err, surf_err, psi_r_err, ff_psi_err, mu0_p_psi_err, q_err = (
            compute_metrics(
                benchmark,
                reference,
                equilibrium,
                shape_x,
            )
        )
        sample = Sample(
            case_key=case_key,
            parameter_count=int(np.size(result.x)),
            elapsed_ms=float(elapsed_ms),
            aggregate_rel_error=float(aggregate),
            shape_rel_error=float(shape_err),
            surface_rel_rms_error=float(surf_err),
            psi_r_rel_rms_error=float(psi_r_err),
            ff_psi_rel_rms_error=float(ff_psi_err),
            mu0_p_psi_rel_rms_error=float(mu0_p_psi_err),
            q_rel_rms_error=float(q_err),
            nfev=result_nfev(result),
            nit=result_nit(result),
            residual_norm_final=float(result.residual_norm_final),
            strategy_name=str(record.strategy_name),
            strategy_names=list(record.strategy_names),
            signature=dict(signature),
            sweep_step=sweep_step,
            is_exact_reference=False,
        )
        return int(idx), sample, float(sample.elapsed_ms), False, None
    except InitialSolveTimeoutError as exc:
        return int(idx), None, float(exc.elapsed_ms), True, None
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return int(idx), None, None, True, message


def finite_metric(benchmark, x_ref, y_ref, x_cur, y_cur, rho_min: float = 0.0) -> float:
    axis = np.asarray(x_cur, dtype=np.float64)
    values = np.asarray(y_cur, dtype=np.float64)
    mask = np.isfinite(axis) & np.isfinite(values)
    if rho_min > 0.0:
        mask &= axis >= float(rho_min)
    if int(np.count_nonzero(mask)) < 2:
        return float("inf")
    axis = axis[mask]
    values = values[mask]
    reference_on_current = np.asarray(
        benchmark._profile_interp(x_ref, y_ref, axis),
        dtype=np.float64,
    )
    valid = np.isfinite(reference_on_current) & np.isfinite(values)
    if int(np.count_nonzero(valid)) < 2:
        return float("inf")
    ref = reference_on_current[valid]
    cur = values[valid]
    scale = max(float(np.max(np.abs(ref))), 1.0e-12)
    diff = cur - ref
    return float(np.sqrt(np.mean(diff * diff)) / scale)


def theta_resolved_surface_metric(reference: ReferenceCase, equilibrium) -> float:
    grid = equilibrium.grid.__class__(
        Nr=max(
            int(reference.exact_equilibrium.grid.Nr),
            int(equilibrium.grid.Nr),
            int(ERROR_SURFACE_NR),
        ),
        Nt=max(
            int(reference.exact_equilibrium.grid.Nt),
            int(equilibrium.grid.Nt),
            int(ERROR_SURFACE_NT),
        ),
        quadrature_scheme="uniform",
        L_max=max(int(reference.exact_equilibrium.grid.L_max), int(equilibrium.grid.L_max)),
        M_max=max(int(reference.exact_equilibrium.grid.M_max), int(equilibrium.grid.M_max)),
    )
    reference_equilibrium = reference.exact_equilibrium.resample(grid=grid)
    equilibrium = equilibrium.resample(grid=grid)
    theta_eval = np.linspace(
        0.0, 2.0 * np.pi, int(ERROR_THETA_SAMPLE_COUNT), endpoint=False, dtype=np.float64
    )
    ref_center = (
        float(reference_equilibrium.R[0, 0]),
        float(reference_equilibrium.Z[0, 0]),
    )
    cur_center = (
        float(equilibrium.R[0, 0]),
        float(equilibrium.Z[0, 0]),
    )
    rms_values_m = [axis_position_error(ref_center, cur_center)]
    for level in SURFACE_LEVELS:
        ref_surface = build_surface_from_psin(reference_equilibrium, float(level))
        cur_surface = build_surface_from_psin(equilibrium, float(level))
        ref_r = radial_profile_from_surface(ref_surface, center=ref_center, theta_eval=theta_eval)
        cur_r = radial_profile_from_surface(cur_surface, center=cur_center, theta_eval=theta_eval)
        diff = cur_r - ref_r
        rms_values_m.append(float(np.sqrt(np.mean(diff * diff))))
    return max(float(np.sqrt(np.mean(np.square(rms_values_m)))), PLOT_EPS)


def compute_metrics(
    benchmark,
    reference: ReferenceCase,
    equilibrium,
    shape_x: np.ndarray,
) -> tuple[float, float, float, float, float, float, float]:
    shape_scale = max(float(np.max(np.abs(reference.exact_shape_x))), 1.0e-12)
    shape_n = min(int(reference.exact_shape_x.shape[0]), int(shape_x.shape[0]))
    if shape_n == 0:
        shape_err = float("inf")
    else:
        shape_diff = shape_x[:shape_n] - reference.exact_shape_x[:shape_n]
        shape_err = float(np.sqrt(np.mean(shape_diff * shape_diff)) / shape_scale)
    psi_r_err = finite_metric(
        benchmark,
        reference.rho_interp_axis,
        reference.ref_profiles["psi_r"],
        equilibrium.rho,
        equilibrium.alpha2 * equilibrium.psin_r,
    )
    ff_psi_err = finite_metric(
        benchmark,
        reference.rho_interp_axis,
        reference.ref_profiles["FF_psi"],
        equilibrium.rho,
        equilibrium.alpha1 * equilibrium.FFn_psin,
    )
    mu0_p_psi_err = finite_metric(
        benchmark,
        reference.rho_interp_axis,
        (4.0e-7 * np.pi) * reference.ref_profiles["P_psi"],
        equilibrium.rho,
        equilibrium.alpha1 * equilibrium.Pn_psin,
    )
    try:
        q_values = equilibrium.q
    except Exception:
        q_values = None
    q_err = (
        float("inf")
        if q_values is None
        else finite_metric(
            benchmark,
            reference.rho_interp_axis,
            reference.ref_profiles["q"],
            equilibrium.rho,
            q_values,
            rho_min=0.05,
        )
    )
    surf_err = theta_resolved_surface_metric(reference, equilibrium)
    return surf_err, shape_err, surf_err, psi_r_err, ff_psi_err, mu0_p_psi_err, q_err


def resample_for_external_shape_metric(equilibrium):
    grid = equilibrium.grid.__class__(
        Nr=max(int(ERROR_SURFACE_NR), int(equilibrium.grid.Nr)),
        Nt=max(int(ERROR_SURFACE_NT), int(equilibrium.grid.Nt)),
        quadrature_scheme="uniform",
        L_max=int(equilibrium.grid.L_max),
        M_max=int(equilibrium.grid.M_max),
    )
    return equilibrium.resample(grid=grid)


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


def extract_gfile_surfaces(geqdsk, levels: list[float]) -> dict[float, np.ndarray]:
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
    if any(abs(level - 1.0) <= 1.0e-12 for level in levels):
        surfaces[1.0] = np.asarray(geqdsk.boundary, dtype=np.float64)
    return surfaces


def external_reference_shape_error(reference: ReferenceCase, equilibrium) -> float:
    """RMS radial shape error against the external GEQDSK target."""
    plot_equilibrium = resample_for_external_shape_metric(equilibrium)
    theta_eval = np.linspace(
        0.0,
        2.0 * np.pi,
        int(ERROR_THETA_SAMPLE_COUNT),
        endpoint=False,
        dtype=np.float64,
    )
    ref_center = (float(reference.geqdsk.Raxis), float(reference.geqdsk.Zaxis))
    cur_center = (float(plot_equilibrium.R[0, 0]), float(plot_equilibrium.Z[0, 0]))
    gfile_levels = [float(level) for level in SHAPE_RMS_PSIN_LEVELS if level > 1.0e-12]
    surfaces = extract_gfile_surfaces(reference.geqdsk, gfile_levels)
    rms_values: list[float] = []
    for level in SHAPE_RMS_PSIN_LEVELS:
        if abs(float(level)) <= 1.0e-12:
            rms_values.append(axis_position_error(ref_center, cur_center))
            continue
        if float(level) not in surfaces:
            continue
        reference_surface = np.asarray(surfaces[float(level)], dtype=np.float64)
        veqpy_surface = build_surface_from_psin(plot_equilibrium, float(level))
        ref_r = radial_profile_from_surface(
            reference_surface, center=ref_center, theta_eval=theta_eval
        )
        veqpy_r = radial_profile_from_surface(
            veqpy_surface, center=cur_center, theta_eval=theta_eval
        )
        rms_values.append(float(np.sqrt(np.mean((veqpy_r - ref_r) ** 2))))
    values = np.asarray(rms_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float("nan") if values.size == 0 else float(np.sqrt(np.mean(values * values)))


def solve_selected_equilibrium_for_shape_metric(
    benchmark,
    reference: ReferenceCase,
    sample: Sample,
    *,
    test_nr: int,
    test_nt: int,
):
    grid = benchmark.Grid(
        Nr=int(test_nr),
        Nt=int(test_nt),
        quadrature_scheme="legendre",
        L_max=int(benchmark.REFERENCE_GRID.L_max),
        M_max=int(benchmark.REFERENCE_GRID.M_max),
    )
    case = build_pf_case(benchmark, reference, grid, sample.signature)
    solver_config = get_case_solver_config(benchmark, reference.case_key)
    solver = benchmark.Solver(
        operator=benchmark.Operator(grid, case),
        config=solver_config,
    )
    solver.solve(
        method=CASE_SOLVER_METHODS[reference.case_key],
        max_residual=float(getattr(solver_config, "max_residual", 1.0e-6)),
        max_evaluations=int(getattr(solver_config, "max_evaluations", REFERENCE_SOLVER_MAXFEV)),
        enable_verbose=False,
        enable_history=False,
        initial_policy=SOLVER_INITIAL_POLICY,
        enable_fallback=False,
    )
    if solver.result is None:
        raise RuntimeError(f"Selected reduced solve failed for {reference.case_key}")
    return solver.build_equilibrium()


def compute_selected_external_shape_errors(
    samples_by_case: dict[str, list[Sample]],
    *,
    backend: str,
    test_nr: int,
    test_nt: int,
    reference_a_by_case: dict[str, float] | None = None,
    selected_config_rows: list[tuple[str, float, Sample | None]] | None = None,
) -> dict[str, float]:
    benchmark = load_benchmark(backend)
    references: dict[str, ReferenceCase] = {}
    if reference_a_by_case is None:
        reference_a_by_case = compute_reference_a_by_case(backend=backend)
    errors: dict[str, float] = {}
    config_rows = selected_config_rows or fastest_standard_config_rows(
        samples_by_case, reference_a_by_case
    )
    for case_key, _threshold, sample in config_rows:
        if sample is None:
            continue
        key = sample_signature_key(case_key, sample.signature)
        if key in errors:
            continue
        reference = references.get(case_key)
        if reference is None:
            reference = build_reference_case(benchmark, case_key)
            references[case_key] = reference
        equilibrium = solve_selected_equilibrium_for_shape_metric(
            benchmark,
            reference,
            sample,
            test_nr=test_nr,
            test_nt=test_nt,
        )
        errors[key] = external_reference_shape_error(reference, equilibrium)
        normalized_error = float(errors[key]) / max(float(reference.reference_a), 1.0e-12)
        print(
            f"[pareto] {CASE_LABELS[case_key]} GEQDSK-file shape error E_gqdsk/a: "
            f"{normalized_error:.{SCIENTIFIC_DECIMALS}e}"
        )
    return errors


def selected_representative_rows_by_case(
    samples_by_case: dict[str, list[Sample]],
    reference_a_by_case: dict[str, float],
    selected_config_rows: list[tuple[str, float, Sample | None]] | None = None,
) -> dict[str, list[tuple[str, Sample]]]:
    rows_by_case: dict[str, list[tuple[str, Sample]]] = {case_key: [] for case_key in CASE_KEYS}
    label_index_by_case = {case_key: 0 for case_key in CASE_KEYS}
    config_rows = selected_config_rows or fastest_standard_config_rows(
        samples_by_case, reference_a_by_case
    )
    for case_key, _threshold, sample in config_rows:
        if sample is None:
            raise RuntimeError(
                f"No representative reduced configuration found for {CASE_LABELS[case_key]}."
            )
        label_index = label_index_by_case[case_key]
        if label_index >= len(REDUCED_CONFIG_LABELS):
            raise RuntimeError(
                f"Too many representative reduced configurations for {CASE_LABELS[case_key]}."
            )
        rows_by_case[case_key].append((REDUCED_CONFIG_LABELS[label_index], sample))
        label_index_by_case[case_key] = label_index + 1
    for case_key, rows in rows_by_case.items():
        if len(rows) != len(REDUCED_CONFIG_LABELS):
            raise RuntimeError(
                f"Expected {len(REDUCED_CONFIG_LABELS)} representative reduced configurations for "
                f"{CASE_LABELS[case_key]}, got {len(rows)}."
            )
    return rows_by_case


def write_representative_reduced_equilibria(
    samples_by_case: dict[str, list[Sample]],
    *,
    backend: str,
    test_nr: int,
    test_nt: int,
    sweep_mode: str = DEFAULT_SWEEP_MODE,
    reference_a_by_case: dict[str, float] | None = None,
    selected_config_rows: list[tuple[str, float, Sample | None]] | None = None,
    manifest_path: str = REDUCED_EQUILIBRIUM_MANIFEST_PATH,
) -> dict[str, object]:
    """Write the 3x3 representative reduced equilibria selected from Pareto cache rows."""

    if reference_a_by_case is None:
        reference_a_by_case = compute_reference_a_by_case(backend=backend)
    rows_by_case = selected_representative_rows_by_case(
        samples_by_case,
        reference_a_by_case,
        selected_config_rows=selected_config_rows,
    )
    benchmark = load_benchmark(backend)
    references: dict[str, ReferenceCase] = {}
    entries: list[dict[str, object]] = []
    for case_key in CASE_KEYS:
        reference = references.get(case_key)
        if reference is None:
            reference = build_reference_case(benchmark, case_key)
            references[case_key] = reference
        for config_label, sample in rows_by_case[case_key]:
            equilibrium = solve_selected_equilibrium_for_shape_metric(
                benchmark,
                reference,
                sample,
                test_nr=test_nr,
                test_nt=test_nt,
            )
            output_path = reduced_equilibrium_json_path(case_key, config_label)
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            equilibrium.write_json(output_path)
            entries.append(
                {
                    "case_key": case_key,
                    "case_label": CASE_LABELS[case_key],
                    "config_label": config_label,
                    "path": output_path,
                    "signature": dict(sample.signature),
                    "parameter_count": int(sample.parameter_count),
                    "elapsed_ms": float(sample.elapsed_ms),
                    "aggregate_rel_error": float(sample.aggregate_rel_error),
                    "residual_norm_final": float(sample.residual_norm_final),
                    "source": sample_signature_key(case_key, sample.signature),
                }
            )
            print(
                f"[pareto] wrote {CASE_LABELS[case_key]} {config_label} "
                f"reduced equilibrium: {output_path}"
            )

    manifest = {
        "signature_version": FULL_SWEEP_SIGNATURE_VERSION
        if str(sweep_mode) == "full"
        else "partial-table05",
        "sweep_mode": str(sweep_mode),
        "backend": str(backend),
        "test_nr": int(test_nr),
        "test_nt": int(test_nt),
        "config_labels": list(REDUCED_CONFIG_LABELS),
        "reference_equilibrium_jsons": dict(CASE_REFERENCE_EQUILIBRIUM_JSONS),
        "reference_gfiles": dict(CASE_REFERENCE_GFILES),
        "entries": entries,
    }
    manifest_dir = os.path.dirname(manifest_path)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[pareto] wrote reduced-equilibrium manifest: {manifest_path}")
    return manifest


def write_representative_reduced_equilibria_from_cache(
    json_stem: str,
    sweep_mode: str,
    *,
    backend: str,
    test_nr: int,
    test_nt: int,
) -> dict[str, object]:
    samples_by_case, _frontiers = load_plot_data_bundle(json_stem, sweep_mode)
    if samples_by_case is None:
        expected = ", ".join(
            json_output_path(json_stem, case_key, sweep_mode) for case_key in CASE_KEYS
        )
        raise FileNotFoundError(
            f"Cannot write reduced equilibria because Pareto cache is missing or stale: {expected}"
        )
    reference_a_by_case = compute_reference_a_by_case(backend=backend)
    return write_representative_reduced_equilibria(
        samples_by_case,
        backend=backend,
        test_nr=test_nr,
        test_nt=test_nt,
        sweep_mode=sweep_mode,
        reference_a_by_case=reference_a_by_case,
    )


def compute_reference_a_by_case(*, backend: str) -> dict[str, float]:
    benchmark = load_benchmark(backend)
    return {
        case_key: build_reference_case(benchmark, case_key).reference_a for case_key in CASE_KEYS
    }


def sample_case(
    benchmark,
    case_key: str,
    backend: str,
    test_nr: int,
    test_nt: int,
    repeat_count: int,
    random_signature_count: int,
    random_signature_seed: int,
    signature_records: list[SignatureRecord] | None = None,
    progress: ParetoProgressDisplay | None = None,
    sweep_mode: str = DEFAULT_SWEEP_MODE,
    case_workers: int = 1,
    case_worker_inner_threads: int = 1,
) -> list[Sample]:
    min_lengths, max_lengths = get_case_length_bounds(case_key)
    case_seed = int(random_signature_seed) + sum(ord(ch) for ch in case_key)
    if signature_records is None:
        signature_records = generate_case_signatures(
            case_key,
            min_lengths,
            max_lengths,
            random_signature_count=random_signature_count,
            random_seed=case_seed,
            sweep_mode=sweep_mode,
        )
    else:
        signature_records = list(signature_records)
    total_signature_count = len(signature_records)
    samples_by_index: list[Sample | None] = [None] * total_signature_count
    missing_records = list(enumerate(signature_records))

    reference: ReferenceCase | None = None

    if progress is not None:
        progress.preparing(case_key)
        progress.start_case(
            case_key,
            total=total_signature_count,
            completed=0,
        )

    if missing_records:
        workers = clamp_case_workers(case_workers, case_worker_inner_threads)
        reference = None
        solver_config = None
        grid = None

        def ensure_serial_context():
            nonlocal reference, solver_config, grid
            if reference is None:
                reference = build_reference_case(benchmark, case_key)
            if solver_config is None:
                solver_config = get_case_solver_config(benchmark, case_key)
            if grid is None:
                grid = benchmark.Grid(
                    Nr=int(test_nr),
                    Nt=int(test_nt),
                    quadrature_scheme="legendre",
                    L_max=int(benchmark.REFERENCE_GRID.L_max),
                    M_max=int(benchmark.REFERENCE_GRID.M_max),
                )
            return reference, solver_config, grid

        def solve_record_serial(idx: int, record: SignatureRecord) -> None:
            ref, config, solve_grid = ensure_serial_context()
            case = build_pf_case(benchmark, ref, solve_grid, record.signature)
            try:
                result, equilibrium, elapsed_ms = solve_with_timing(
                    benchmark,
                    case,
                    solve_grid,
                    repeat_count,
                    method=CASE_SOLVER_METHODS[case_key],
                    solver_config=config,
                )
            except InitialSolveTimeoutError as exc:
                if progress is not None:
                    progress.update(
                        case_key,
                        elapsed_ms=exc.elapsed_ms,
                        aggregate_rel_error=None,
                        skipped=True,
                    )
                return
            shape_x = benchmark._extract_shape_x(case.active_profiles, result.x)
            aggregate, shape_err, surf_err, psi_r_err, ff_psi_err, mu0_p_psi_err, q_err = (
                compute_metrics(
                    benchmark,
                    ref,
                    equilibrium,
                    shape_x,
                )
            )
            sample = Sample(
                case_key=case_key,
                parameter_count=int(np.size(result.x)),
                elapsed_ms=float(elapsed_ms),
                aggregate_rel_error=float(aggregate),
                shape_rel_error=float(shape_err),
                surface_rel_rms_error=float(surf_err),
                psi_r_rel_rms_error=float(psi_r_err),
                ff_psi_rel_rms_error=float(ff_psi_err),
                mu0_p_psi_rel_rms_error=float(mu0_p_psi_err),
                q_rel_rms_error=float(q_err),
                nfev=result_nfev(result),
                nit=result_nit(result),
                residual_norm_final=float(result.residual_norm_final),
                strategy_name=str(record.strategy_name),
                strategy_names=list(record.strategy_names),
                signature=dict(record.signature),
                sweep_step=record.sweep_step,
                is_exact_reference=False,
            )
            samples_by_index[idx] = sample
            if progress is not None:
                progress.update(
                    case_key,
                    elapsed_ms=float(sample.elapsed_ms),
                    aggregate_rel_error=float(sample.aggregate_rel_error),
                )

        if workers > 1:
            set_inner_thread_environment(case_worker_inner_threads)
            context = multiprocessing.get_context("spawn")
            pending_records = dict(missing_records)
            try:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=context,
                    initializer=_case_worker_initializer,
                    initargs=(
                        str(backend),
                        str(case_key),
                        int(test_nr),
                        int(test_nt),
                        int(repeat_count),
                        int(case_worker_inner_threads),
                    ),
                ) as executor:
                    futures = {
                        executor.submit(solve_signature_record_worker, task): task
                        for task in missing_records
                    }
                    for future in concurrent.futures.as_completed(futures):
                        idx, sample, elapsed_ms, skipped, error_message = future.result()
                        _, record = futures[future]
                        pending_records.pop(int(idx), None)
                        if error_message is not None:
                            print(
                                f"[pareto] worker skipped {case_key} "
                                f"{record.sweep_step}: {error_message}"
                            )
                        if sample is not None:
                            samples_by_index[idx] = sample
                        if progress is not None:
                            progress.update(
                                case_key,
                                elapsed_ms=elapsed_ms,
                                aggregate_rel_error=None
                                if sample is None
                                else float(sample.aggregate_rel_error),
                                skipped=bool(skipped),
                            )
            except concurrent.futures.process.BrokenProcessPool as exc:
                print(
                    f"[pareto] {case_key} process pool failed ({exc}); "
                    f"falling back to serial for {len(pending_records)} remaining cases"
                )
                for idx, record in sorted(pending_records.items()):
                    solve_record_serial(idx, record)
        else:
            for idx, record in missing_records:
                solve_record_serial(idx, record)

    samples: list[Sample] = [sample for sample in samples_by_index if sample is not None]
    if reference is None:
        reference = build_reference_case(benchmark, case_key)
    samples.append(build_exact_reference_sample(benchmark, reference))
    if progress is not None:
        progress.finish_case(case_key)
    return samples


def pareto_frontier(samples: list[Sample], min_rel_improvement: float) -> list[Sample]:
    valid = [
        sample
        for sample in samples
        if np.isfinite(sample.elapsed_ms)
        and np.isfinite(sample.aggregate_rel_error)
        and sample.elapsed_ms > 0.0
        and sample.aggregate_rel_error >= FRONTIER_MIN_REL_ERROR
        and sample.aggregate_rel_error <= FRONTIER_MAX_REL_ERROR
        and not sample.is_exact_reference
        and not is_excluded_standard_signature(sample.case_key, sample.signature)
    ]
    valid.sort(
        key=lambda sample: (
            sample.elapsed_ms,
            sample.aggregate_rel_error,
            sample.parameter_count,
        )
    )
    frontier: list[Sample] = []
    best_error = float("inf")
    for sample in valid:
        rel_gain = 1.0 - (sample.aggregate_rel_error / max(best_error, PLOT_EPS))
        if sample.aggregate_rel_error < best_error * (1.0 - 1.0e-12) and (
            not frontier or rel_gain >= float(min_rel_improvement)
        ):
            frontier.append(sample)
            best_error = sample.aggregate_rel_error
    return frontier


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h = np.diff(x)
    delta = np.diff(y) / h
    n = x.size
    slopes = np.zeros(n, dtype=np.float64)
    if n == 2:
        slopes[:] = delta[0]
        return slopes
    for idx in range(1, n - 1):
        if delta[idx - 1] * delta[idx] <= 0.0:
            slopes[idx] = 0.0
            continue
        w1 = 2.0 * h[idx] + h[idx - 1]
        w2 = h[idx] + 2.0 * h[idx - 1]
        slopes[idx] = (w1 + w2) / ((w1 / delta[idx - 1]) + (w2 / delta[idx]))
    left = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if left * delta[0] <= 0.0:
        slopes[0] = 0.0
    elif delta[0] * delta[1] < 0.0 and abs(left) > abs(3.0 * delta[0]):
        slopes[0] = 3.0 * delta[0]
    else:
        slopes[0] = left
    right = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if right * delta[-1] <= 0.0:
        slopes[-1] = 0.0
    elif delta[-1] * delta[-2] < 0.0 and abs(right) > abs(3.0 * delta[-1]):
        slopes[-1] = 3.0 * delta[-1]
    else:
        slopes[-1] = right
    return slopes


def blend_with_white(color: str, blend: float) -> tuple[float, float, float]:
    rgb = np.asarray(matplotlib.colors.to_rgb(color), dtype=np.float64)
    weight = min(max(float(blend), 0.0), 1.0)
    mixed = rgb * (1.0 - weight) + weight
    return tuple(float(value) for value in mixed)


def smooth_frontier_curve(
    x_values,
    y_values,
    *,
    samples_per_segment: int,
    log_x: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if int(np.count_nonzero(valid)) < 2:
        return None
    x = x[valid]
    y = y[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_unique, unique_idx = np.unique(x, return_index=True)
    y = y[unique_idx]
    if x_unique.size < 2:
        return None
    x_domain = np.log10(x_unique) if log_x else x_unique
    log_y = np.log10(y)
    slopes = _pchip_slopes(x_domain, log_y)
    x_segments: list[np.ndarray] = []
    y_segments: list[np.ndarray] = []
    for idx in range(x_domain.size - 1):
        x0 = x_domain[idx]
        x1 = x_domain[idx + 1]
        if abs(x1 - x0) < 1.0e-12:
            continue
        y0 = log_y[idx]
        y1 = log_y[idx + 1]
        m0 = slopes[idx]
        m1 = slopes[idx + 1]
        t = np.linspace(
            0.0,
            1.0,
            max(int(samples_per_segment), 24),
            endpoint=(idx == x_domain.size - 2),
        )
        h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
        h10 = t**3 - 2.0 * t**2 + t
        h01 = -2.0 * t**3 + 3.0 * t**2
        h11 = t**3 - t**2
        x_seg = x0 + t * (x1 - x0)
        y_seg = h00 * y0 + h10 * (x1 - x0) * m0 + h01 * y1 + h11 * (x1 - x0) * m1
        x_segments.append(x_seg)
        y_segments.append(y_seg)
    if not x_segments:
        return None
    x_curve = np.concatenate(x_segments)
    if log_x:
        x_curve = np.power(10.0, x_curve)
    return x_curve, np.power(10.0, np.concatenate(y_segments))


def parameter_error_frontier(samples: list[Sample]) -> list[Sample]:
    valid = [
        sample
        for sample in samples
        if np.isfinite(sample.aggregate_rel_error)
        and sample.aggregate_rel_error >= FRONTIER_MIN_REL_ERROR
        and sample.aggregate_rel_error <= FRONTIER_MAX_REL_ERROR
        and int(sample.parameter_count) <= PARAMETER_COUNT_PLOT_MAX
        and not sample.is_exact_reference
    ]
    best_by_count: dict[int, Sample] = {}
    for sample in valid:
        count = int(sample.parameter_count)
        incumbent = best_by_count.get(count)
        if incumbent is None or sample.aggregate_rel_error < incumbent.aggregate_rel_error:
            best_by_count[count] = sample
    frontier: list[Sample] = []
    best_error = float("inf")
    for count in sorted(best_by_count):
        sample = best_by_count[count]
        if sample.aggregate_rel_error < best_error * (1.0 - 1.0e-12):
            frontier.append(sample)
            best_error = sample.aggregate_rel_error
    return frontier


def frontier_plot_samples(samples: list[Sample], *, normalizer_m: float = 1.0) -> list[Sample]:
    return [
        sample
        for sample in samples
        if np.isfinite(normalized_shape_error(sample, normalizer_m))
        and FRONTIER_MIN_REL_ERROR
        <= normalized_shape_error(sample, normalizer_m)
        <= FRONTIER_MAX_REL_ERROR
    ]


def format_frontier_x(value: float, *, kind: str) -> str:
    if kind == "time_ms":
        return f"{value:.{FIXED_DECIMALS}f} ms"
    if kind == "parameter_count":
        return f"{int(round(value))}"
    raise ValueError(f"Unsupported frontier x kind: {kind!r}")


def normalized_shape_error(sample: Sample, normalizer_m: float) -> float:
    normalizer = float(normalizer_m)
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        normalizer = 1.0
    return float(sample.aggregate_rel_error) / normalizer


def format_frontier_y(value: float) -> str:
    return f"{float(value):.{SCIENTIFIC_DECIMALS}e}"


def format_signature(signature: dict[str, int]) -> str:
    return (
        "{" + ", ".join(f"{name}: {int(value)}" for name, value in sorted(signature.items())) + "}"
    )


def _signature_tuple(signature: dict[str, int], names: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(int(signature.get(name, 0)) for name in names)


def signature_core_tuple(signature: dict[str, int]) -> tuple[int, int, int, int]:
    return _signature_tuple(signature, ("h", "v", "k", "psin"))


def signature_family_tuple(
    signature: dict[str, int], prefix: str, *, start: int, stop: int
) -> tuple[int, ...]:
    return tuple(int(signature.get(f"{prefix}{idx}", 0)) for idx in range(start, stop + 1))


def trim_trailing_zeros(values: tuple[int, ...]) -> tuple[int, ...]:
    end = len(values)
    while end > 0 and int(values[end - 1]) == 0:
        end -= 1
    return values[:end]


def format_tuple_tex(values: tuple[int, ...]) -> str:
    if not values:
        return "--"
    return "$(" + ",".join(str(int(value)) for value in values) + ")$"


def signature_group_tuples(
    sample: Sample,
) -> tuple[tuple[int, int, int, int], tuple[int, ...], tuple[int, ...]]:
    max_lengths = get_max_lengths(sample.case_key)
    c_indices = [
        int(name[1:])
        for name in max_lengths
        if name.startswith("c") and len(name) > 1 and name[1:].isdigit()
    ]
    s_indices = [
        int(name[1:])
        for name in max_lengths
        if name.startswith("s") and len(name) > 1 and name[1:].isdigit()
    ]
    c_tuple = (
        trim_trailing_zeros(
            signature_family_tuple(sample.signature, "c", start=0, stop=max(c_indices))
        )
        if c_indices
        else ()
    )
    s_tuple = (
        trim_trailing_zeros(
            signature_family_tuple(sample.signature, "s", start=1, stop=max(s_indices))
        )
        if s_indices
        else ()
    )
    return signature_core_tuple(sample.signature), c_tuple, s_tuple


def format_tex_scientific(value: float, *, precision: int = SCIENTIFIC_DECIMALS) -> str:
    if not np.isfinite(value):
        return "--"
    if value == 0.0:
        return "$0$"
    exponent = int(np.floor(np.log10(abs(float(value)))))
    mantissa = float(value) / (10.0**exponent)
    if abs(mantissa - 1.0) < 0.5 * 10.0 ** (-precision):
        return rf"$10^{{{exponent}}}$"
    return rf"${mantissa:.{precision}f}\times 10^{{{exponent}}}$"


def format_normalized_error(value_m: float, normalizer_m: float) -> str:
    if not np.isfinite(value_m) or not np.isfinite(normalizer_m) or normalizer_m <= 0.0:
        return "--"
    return format_tex_scientific(
        float(value_m) / float(normalizer_m), precision=SCIENTIFIC_DECIMALS
    )


def same_standard_config(left: Sample | None, right: Sample | None) -> bool:
    if left is None or right is None:
        return False
    return left.case_key == right.case_key and left.signature == right.signature


def first_frontier_index_below_error(
    frontier: list[Sample],
    *,
    threshold: float,
    normalizer_m: float,
    after_index: int = -1,
) -> int | None:
    for index, sample in enumerate(frontier):
        if index <= int(after_index):
            continue
        if float(sample.aggregate_rel_error) / float(normalizer_m) <= float(threshold):
            return index
    return None


def closest_frontier_index_above_error(
    frontier: list[Sample],
    *,
    threshold: float,
    normalizer_m: float,
    after_index: int = -1,
) -> int | None:
    candidates: list[tuple[float, float, int, int]] = []
    for index, sample in enumerate(frontier):
        if index <= int(after_index):
            continue
        normalized_error = float(sample.aggregate_rel_error) / float(normalizer_m)
        if normalized_error > float(threshold):
            candidates.append(
                (
                    normalized_error - float(threshold),
                    float(sample.elapsed_ms),
                    int(sample.parameter_count),
                    index,
                )
            )
    if not candidates:
        return None
    return min(candidates)[-1]


def frontier_index_for_target_error(
    frontier: list[Sample],
    *,
    threshold: float,
    normalizer_m: float,
    after_index: int = -1,
) -> int | None:
    selected_index = first_frontier_index_below_error(
        frontier,
        threshold=threshold,
        normalizer_m=normalizer_m,
        after_index=after_index,
    )
    if selected_index is not None:
        return selected_index

    selected_index = first_frontier_index_below_error(
        frontier,
        threshold=threshold,
        normalizer_m=normalizer_m,
    )
    if selected_index is not None:
        return selected_index

    selected_index = closest_frontier_index_above_error(
        frontier,
        threshold=threshold,
        normalizer_m=normalizer_m,
        after_index=after_index,
    )
    if selected_index is not None:
        return selected_index

    return closest_frontier_index_above_error(
        frontier,
        threshold=threshold,
        normalizer_m=normalizer_m,
    )


def fastest_standard_config_rows(
    samples_by_case: dict[str, list[Sample]],
    reference_a_by_case: dict[str, float] | None = None,
) -> list[tuple[str, float, Sample | None]]:
    rows: list[tuple[str, float, Sample | None]] = []
    reference_a_by_case = reference_a_by_case or {}
    for case_key in CASE_KEYS:
        thresholds = tuple(
            float(threshold) for threshold in CASE_FASTEST_CONFIG_ERROR_THRESHOLDS[case_key]
        )
        normalizer = float(reference_a_by_case.get(case_key, 1.0))
        if not np.isfinite(normalizer) or normalizer <= 0.0:
            normalizer = 1.0
        samples = [
            sample
            for sample in samples_by_case.get(case_key, [])
            if not is_excluded_standard_signature(case_key, sample.signature)
        ]
        frontier = pareto_frontier(samples, min_rel_improvement=0.0)
        selected_indices: list[int | None] = []
        previous_index = -1
        for threshold in thresholds:
            selected_index = frontier_index_for_target_error(
                frontier,
                threshold=threshold,
                normalizer_m=normalizer,
                after_index=previous_index,
            )
            selected_indices.append(selected_index)
            if selected_index is not None:
                previous_index = int(selected_index)
        selected = [None if index is None else frontier[index] for index in selected_indices]
        rows.extend(
            (case_key, threshold, sample)
            for threshold, sample in zip(thresholds, selected, strict=True)
        )
    return rows


def selected_table05_config_rows(
    samples_by_case: dict[str, list[Sample]],
) -> list[tuple[str, float, Sample | None]]:
    """Return the manuscript's fixed 3x3 representative rows in Low/Medium/High order.

    Partial mode is a cheap rerun of the already selected rows, not a new
    timing-sensitive Pareto selection.  Keeping the row order fixed prevents
    normal timing noise from dropping a latency-oriented row that is dominated
    by another selected row in a short partial run.
    """

    rows: list[tuple[str, float, Sample | None]] = []
    for case_key in CASE_KEYS:
        thresholds = tuple(
            float(threshold) for threshold in CASE_FASTEST_CONFIG_ERROR_THRESHOLDS[case_key]
        )
        signatures = TABLE05_SELECTED_SIGNATURES.get(case_key, ())
        if len(signatures) != len(thresholds):
            raise RuntimeError(
                f"Expected {len(thresholds)} selected signatures "
                f"for {CASE_LABELS[case_key]}, got {len(signatures)}."
            )
        samples_by_signature = {
            signature_key(sample.signature): sample
            for sample in samples_by_case.get(case_key, [])
            if not sample.is_exact_reference
        }
        for threshold, signature in zip(thresholds, signatures, strict=True):
            sample = samples_by_signature.get(signature_key(signature))
            if sample is None:
                raise RuntimeError(
                    f"Missing selected {CASE_LABELS[case_key]} "
                    f"partial signature: {signature_key(signature)}"
                )
            rows.append((case_key, threshold, sample))
    return rows


def build_fastest_config_latex_table_body(
    samples_by_case: dict[str, list[Sample]],
    external_shape_errors: dict[str, float] | None = None,
    reference_a_by_case: dict[str, float] | None = None,
    selected_config_rows: list[tuple[str, float, Sample | None]] | None = None,
) -> str:
    indent = "              "
    header = [
        "Case (Params)",
        "Time [ms]",
        r"$n_{\mathrm{fev}}$",
        r"$E_{\mathrm{ref}}/a$",
        r"$E_{\mathrm{gqdsk}}/a$",
        "Core",
        "Cos",
        "Sin",
    ]
    external_shape_errors = external_shape_errors or {}
    reference_a_by_case = reference_a_by_case or {}
    rows: list[list[str]] = []
    config_rows = selected_config_rows or fastest_standard_config_rows(
        samples_by_case, reference_a_by_case
    )
    for case_key, _threshold, sample in config_rows:
        case_label = CASE_LABELS[case_key]
        if sample is None:
            rows.append([case_label, "--", "--", "--", "--", "--", "--", "--"])
            continue
        core_tuple, c_tuple, s_tuple = signature_group_tuples(sample)
        external_error = external_shape_errors.get(
            sample_signature_key(case_key, sample.signature), float("nan")
        )
        normalizer = reference_a_by_case.get(case_key, float("nan"))
        rows.append(
            [
                f"{case_label} ({int(sample.parameter_count)})",
                f"${float(sample.elapsed_ms):.{FIXED_DECIMALS}f}$",
                f"${int(sample.nfev)}$",
                format_normalized_error(float(sample.aggregate_rel_error), normalizer),
                format_normalized_error(float(external_error), normalizer),
                format_tuple_tex(core_tuple),
                format_tuple_tex(c_tuple),
                format_tuple_tex(s_tuple),
            ]
        )

    column_widths = [
        max(len(row[column_index]) for row in [header, *rows])
        for column_index in range(len(header))
    ]

    def format_row(row: list[str]) -> str:
        return (
            " & ".join(cell.ljust(column_widths[index]) for index, cell in enumerate(row)) + r" \\"
        )

    lines = [
        r"\hline",
        format_row(header),
        r"\hline",
        *(format_row(row) for row in rows),
        r"\hline",
    ]
    return "\n".join(indent + line for line in lines)


def build_fastest_config_latex_table(
    samples_by_case: dict[str, list[Sample]],
    external_shape_errors: dict[str, float] | None = None,
    reference_a_by_case: dict[str, float] | None = None,
    selected_config_rows: list[tuple[str, float, Sample | None]] | None = None,
) -> str:
    return build_fastest_config_latex_table_body(
        samples_by_case,
        external_shape_errors,
        reference_a_by_case,
        selected_config_rows=selected_config_rows,
    )


def print_fastest_config_table_body(
    samples_by_case: dict[str, list[Sample]],
    external_shape_errors: dict[str, float] | None = None,
    reference_a_by_case: dict[str, float] | None = None,
    selected_config_rows: list[tuple[str, float, Sample | None]] | None = None,
) -> None:
    print(
        build_fastest_config_latex_table(
            samples_by_case,
            external_shape_errors,
            reference_a_by_case,
            selected_config_rows=selected_config_rows,
        )
    )


def describe_frontier(
    samples: list[Sample],
    *,
    x_kind: str,
    normalizer_m: float = 1.0,
) -> str:
    if not samples:
        return "none"
    head = samples[0]
    tail = samples[-1]
    head_x = float(head.elapsed_ms) if x_kind == "time_ms" else float(head.parameter_count)
    tail_x = float(tail.elapsed_ms) if x_kind == "time_ms" else float(tail.parameter_count)
    return (
        f"head=({format_frontier_x(head_x, kind=x_kind)}, "
        f"{format_frontier_y(normalized_shape_error(head, normalizer_m))}), "
        f"tail=({format_frontier_x(tail_x, kind=x_kind)}, "
        f"{format_frontier_y(normalized_shape_error(tail, normalizer_m))})"
    )


def standard_case_for_target_error(
    samples: list[Sample],
    *,
    threshold: float,
    normalizer_m: float = 1.0,
) -> Sample | None:
    frontier = pareto_frontier(samples, min_rel_improvement=0.0)
    selected_index = frontier_index_for_target_error(
        frontier,
        threshold=threshold,
        normalizer_m=normalizer_m,
    )
    if selected_index is None:
        return None
    return frontier[selected_index]


def _style_axis(
    ax: plt.Axes,
    *,
    title: str,
    xlabel: str,
    ylabel: str = "",
    yscale: str = "log",
) -> None:
    ax.set_title(title, fontsize=scaled_font_size(TITLE_FONT_SIZE), fontweight="normal")
    ax.set_xlabel(xlabel, fontsize=scaled_font_size(AXIS_LABEL_FONT_SIZE))
    ax.set_ylabel(ylabel, fontsize=scaled_font_size(AXIS_LABEL_FONT_SIZE))
    ax.set_yscale(yscale)
    ax.grid(
        True, which="both", alpha=GRID_ALPHA, linewidth=GRID_LINE_WIDTH, linestyle=GRID_LINESTYLE
    )
    ax.set_axisbelow(True)
    ax.tick_params(
        which="both",
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


def remove_minor_ticks(ax: plt.Axes) -> None:
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())


def _style_case_config_legend(
    ax: plt.Axes,
    line_handles: dict[str, object],
) -> None:
    handles: list[object] = []
    labels: list[str] = []
    for case_key in CASE_KEYS:
        line_handle = line_handles.get(case_key)
        if line_handle is None:
            continue
        handles.append(line_handle)
        labels.append(CASE_LABELS[case_key])
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        loc=LEGEND_LOC,
        frameon=LEGEND_FRAME_ON,
        fontsize=scaled_font_size(LEGEND_FONT_SIZE),
        ncol=LEGEND_NCOL,
        columnspacing=LEGEND_COLUMN_SPACING,
        labelspacing=LEGEND_LABEL_SPACING,
    )


def _scatter_representative_marker(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    case_key: str,
    config_label: str,
    label_offsets: dict[tuple[str, str], tuple[int, int]],
):
    color = CASE_COLORS[case_key]
    label_offset = label_offsets[(case_key, config_label)]
    marker_artist = ax.scatter(
        [float(x)],
        [float(y)],
        s=CONFIG_MARKER_SIZE,
        marker="o",
        facecolors=color,
        edgecolors="white",
        linewidths=CONFIG_MARKER_EDGE_WIDTH,
        alpha=1.0,
        zorder=5,
    )
    text_artist = ax.annotate(
        config_label,
        xy=(float(x), float(y)),
        xytext=label_offset,
        textcoords="offset points",
        color=color,
        ha="center",
        va="center",
        fontsize=scaled_font_size(CONFIG_MARKER_FONT_SIZE),
        fontweight="bold",
        arrowprops={
            "arrowstyle": "-",
            "color": color,
            "linewidth": 0.7,
            "shrinkA": 1.5,
            "shrinkB": 3.5,
        },
        zorder=6,
    )
    return marker_artist, text_artist


def _scatter_representative_markers(
    ax: plt.Axes,
    *,
    representative_rows: list[tuple[str, Sample]],
    x_value,
    normalizer_m: float,
    label_offsets: dict[tuple[str, str], tuple[int, int]],
):
    first_marker = None
    for config_label, sample in representative_rows:
        marker_artist, _text_artist = _scatter_representative_marker(
            ax,
            x=float(x_value(sample)),
            y=normalized_shape_error(sample, normalizer_m),
            case_key=sample.case_key,
            config_label=config_label,
            label_offsets=label_offsets,
        )
        if first_marker is None:
            first_marker = marker_artist
    return first_marker


def _scatter_representative_points(
    ax: plt.Axes,
    *,
    representative_rows: list[tuple[str, Sample]],
    x_value,
    normalizer_m: float,
) -> None:
    if not representative_rows:
        return
    case_key = representative_rows[0][1].case_key
    ax.scatter(
        [float(x_value(sample)) for _config_label, sample in representative_rows],
        [
            normalized_shape_error(sample, normalizer_m)
            for _config_label, sample in representative_rows
        ],
        s=CONFIG_MARKER_SIZE,
        marker="o",
        facecolors=CASE_COLORS[case_key],
        edgecolors="white",
        linewidths=CONFIG_MARKER_EDGE_WIDTH,
        alpha=1.0,
        zorder=5,
    )


def _plot_selected_frontier_line(
    ax: plt.Axes,
    *,
    representative_rows: list[tuple[str, Sample]],
    x_value,
    normalizer_m: float,
    label: str | None = None,
) -> object | None:
    if not representative_rows:
        return None
    case_key = representative_rows[0][1].case_key
    (line_handle,) = ax.plot(
        [float(x_value(sample)) for _config_label, sample in representative_rows],
        [
            normalized_shape_error(sample, normalizer_m)
            for _config_label, sample in representative_rows
        ],
        color=CASE_COLORS[case_key],
        linestyle=CASE_LINESTYLES[case_key],
        linewidth=FRONTIER_LINE_WIDTH,
        label=label,
        alpha=0.95,
        zorder=4,
    )
    return line_handle


def _sample_plot_key(sample: Sample) -> tuple[int, float, float]:
    return (
        int(sample.parameter_count),
        float(sample.elapsed_ms),
        float(sample.aggregate_rel_error),
    )


def _scatter_background(
    ax: plt.Axes,
    *,
    x_values: list[float],
    y_values: list[float],
    color: str,
) -> None:
    if not x_values:
        return
    ax.scatter(
        x_values,
        y_values,
        s=BACKGROUND_MARKER_SIZE,
        color=color,
        alpha=BACKGROUND_MARKER_ALPHA,
        linewidths=0.0,
        zorder=1,
    )


def _print_case_frontier_summary(
    case_key: str,
    *,
    time_frontier: list[Sample],
    parameter_frontier: list[Sample],
    representative: Sample | None,
    normalizer_m: float,
) -> None:
    print(
        f"[pareto] {CASE_LABELS[case_key]} time frontier: "
        f"{describe_frontier(time_frontier, x_kind='time_ms', normalizer_m=normalizer_m)}"
    )
    if representative is None:
        print(
            f"[pareto] {CASE_LABELS[case_key]} representative case "
            f"(target E_ref/a={REPRESENTATIVE_ERROR_THRESHOLD:.0e}): none"
        )
    else:
        representative_error = normalized_shape_error(representative, normalizer_m)
        relation = (
            "<=" if representative_error <= REPRESENTATIVE_ERROR_THRESHOLD else "closest above"
        )
        print(
            f"[pareto] {CASE_LABELS[case_key]} representative case "
            f"(target E_ref/a={REPRESENTATIVE_ERROR_THRESHOLD:.0e}, {relation}): "
            f"({format_frontier_x(float(representative.elapsed_ms), kind='time_ms')}, "
            f"{format_frontier_y(representative_error)}), "
            f"parameters={int(representative.parameter_count)}, "
            f"signature={format_signature(representative.signature)}"
        )
    print(
        f"[pareto] {CASE_LABELS[case_key]} parameter frontier: ",
        describe_frontier(
            parameter_frontier,
            x_kind="parameter_count",
            normalizer_m=normalizer_m,
        ),
    )


def _plot_time_case(
    ax: plt.Axes,
    *,
    case_key: str,
    samples: list[Sample],
    time_frontier: list[Sample],
    representative_rows: list[tuple[str, Sample]],
    normalizer_m: float,
) -> object | None:
    del time_frontier
    representative_keys = {
        _sample_plot_key(sample) for _config_label, sample in representative_rows
    }
    time_background = [
        sample for sample in samples if _sample_plot_key(sample) not in representative_keys
    ]
    background_color = blend_with_white(CASE_COLORS[case_key], 0.42)
    _scatter_background(
        ax,
        x_values=[float(sample.elapsed_ms) for sample in time_background],
        y_values=[normalized_shape_error(sample, normalizer_m) for sample in time_background],
        color=background_color,
    )

    line_handle = _plot_selected_frontier_line(
        ax,
        representative_rows=representative_rows,
        x_value=lambda sample: sample.elapsed_ms,
        normalizer_m=normalizer_m,
        label=CASE_LABELS[case_key],
    )
    _scatter_representative_markers(
        ax,
        representative_rows=representative_rows,
        x_value=lambda sample: sample.elapsed_ms,
        normalizer_m=normalizer_m,
        label_offsets=TIME_CONFIG_MARKER_LABEL_OFFSETS,
    )
    return line_handle


def _plot_parameter_case(
    ax: plt.Axes,
    *,
    case_key: str,
    samples: list[Sample],
    parameter_frontier: list[Sample],
    representative_rows: list[tuple[str, Sample]],
    normalizer_m: float,
) -> None:
    del parameter_frontier
    samples = [
        sample for sample in samples if int(sample.parameter_count) <= PARAMETER_COUNT_PLOT_MAX
    ]
    background_color = blend_with_white(CASE_COLORS[case_key], 0.42)
    _scatter_background(
        ax,
        x_values=[float(sample.parameter_count) for sample in samples],
        y_values=[normalized_shape_error(sample, normalizer_m) for sample in samples],
        color=background_color,
    )
    visible_representative_rows = [
        (config_label, sample)
        for config_label, sample in representative_rows
        if int(sample.parameter_count) <= PARAMETER_COUNT_PLOT_MAX
    ]
    _plot_selected_frontier_line(
        ax,
        representative_rows=visible_representative_rows,
        x_value=lambda sample: sample.parameter_count,
        normalizer_m=normalizer_m,
    )
    _scatter_representative_points(
        ax,
        representative_rows=visible_representative_rows,
        x_value=lambda sample: sample.parameter_count,
        normalizer_m=normalizer_m,
    )


def json_output_path(json_stem: str, case_key: str, sweep_mode: str) -> str:
    json_dir = os.path.dirname(json_stem)
    json_name = os.path.basename(json_stem)
    case_name = f"{json_name}_{str(sweep_mode)}_{case_key}.json"
    return os.path.join(json_dir, case_name) if json_dir else case_name


def reduced_equilibrium_json_path(case_key: str, config_label: str) -> str:
    return REDUCED_EQUILIBRIUM_JSON_TEMPLATE.format(
        case_key=str(case_key),
        config_label=str(config_label).lower(),
    )


def close_curve(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    return np.vstack((arr, arr[:1]))


def compute_rz_limits(
    curves: list[np.ndarray], *, pad_fraction: float = SURFACE_COMPARE_PAD_FRACTION
) -> tuple[tuple[float, float], tuple[float, float]]:
    stacked = np.vstack(
        [np.asarray(curve, dtype=np.float64) for curve in curves if np.asarray(curve).size]
    )
    r_min = float(np.min(stacked[:, 0]))
    r_max = float(np.max(stacked[:, 0]))
    z_min = float(np.min(stacked[:, 1]))
    z_max = float(np.max(stacked[:, 1]))
    r_pad = max((r_max - r_min) * pad_fraction, 1.0e-3)
    z_pad = max((z_max - z_min) * pad_fraction, 1.0e-3)
    return (r_min - r_pad, r_max + r_pad), (z_min - z_pad, z_max + z_pad)


def build_surfaces_from_equilibrium(
    equilibrium, levels: tuple[float, ...]
) -> dict[float, np.ndarray]:
    return {float(level): build_surface_from_psin(equilibrium, float(level)) for level in levels}


def finite_non_reference_samples(samples: list[Sample]) -> list[Sample]:
    return [
        sample
        for sample in samples
        if np.isfinite(sample.elapsed_ms)
        and np.isfinite(sample.aggregate_rel_error)
        and sample.elapsed_ms > 0.0
        and sample.aggregate_rel_error > 0.0
        and not sample.is_exact_reference
        and not is_excluded_standard_signature(sample.case_key, sample.signature)
    ]


def exact_reference_parameter_count(samples: list[Sample], case_key: str) -> int:
    exact = next((sample for sample in samples if sample.is_exact_reference), None)
    if exact is not None:
        return int(exact.parameter_count)
    return int(sum(get_max_lengths(case_key).values()))


def build_surface_comparison_data(
    all_samples: dict[str, list[Sample]],
    *,
    selected_rows_by_case: dict[str, list[tuple[str, Sample]]],
    backend: str,
    test_nr: int,
    test_nt: int,
) -> dict[str, dict[str, object]]:
    benchmark = load_benchmark(backend)
    data: dict[str, dict[str, object]] = {}
    for case_key in CASE_KEYS:
        reference = build_reference_case(benchmark, case_key)
        rows = selected_rows_by_case.get(case_key, [])
        if not rows:
            continue
        grid = benchmark.Grid(
            Nr=int(test_nr),
            Nt=int(test_nt),
            quadrature_scheme="legendre",
            L_max=int(benchmark.REFERENCE_GRID.L_max),
            M_max=int(benchmark.REFERENCE_GRID.M_max),
        )
        reduced: dict[str, dict[str, object]] = {}
        for config_label, sample in rows:
            representative_case = build_pf_case(benchmark, reference, grid, sample.signature)
            _, representative_equilibrium, _ = solve_with_timing(
                benchmark,
                representative_case,
                grid,
                repeat_count=1,
                method=CASE_SOLVER_METHODS[case_key],
                solver_config=get_case_solver_config(benchmark, case_key),
            )
            reduced[str(config_label)] = {
                "equilibrium": resample_surface_equilibrium(representative_equilibrium),
                "parameter_count": int(sample.parameter_count),
                "sample": sample,
            }
        data[case_key] = {
            "reference_equilibrium": reference.exact_equilibrium,
            "reference_parameter_count": exact_reference_parameter_count(
                all_samples[case_key], case_key
            ),
            "reduced": reduced,
        }
    return data


def render_surface_comparison_grid(
    axes: list[list[plt.Axes]],
    surface_data: dict[str, dict[str, object]],
) -> None:
    surface_limits_by_case: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
    high_surfaces_by_case: dict[str, dict[float, np.ndarray]] = {}
    reduced_surfaces_by_case: dict[str, dict[str, dict[float, np.ndarray]]] = {}
    for case_key in CASE_KEYS:
        case_data = surface_data.get(case_key)
        if case_data is None:
            continue
        high_surfaces = build_surfaces_from_equilibrium(
            case_data["reference_equilibrium"], SURFACE_COMPARE_LEVELS
        )
        reduced_surfaces: dict[str, dict[float, np.ndarray]] = {}
        curves = list(high_surfaces.values())
        reduced_block = case_data.get("reduced", {})
        if isinstance(reduced_block, dict):
            for config_label, reduced_data in reduced_block.items():
                if not isinstance(reduced_data, dict):
                    continue
                reduced_equilibrium = reduced_data.get("equilibrium")
                if reduced_equilibrium is None:
                    continue
                label_surfaces = build_surfaces_from_equilibrium(
                    reduced_equilibrium, SURFACE_COMPARE_LEVELS
                )
                reduced_surfaces[str(config_label)] = label_surfaces
                curves.extend(label_surfaces.values())
        high_surfaces_by_case[case_key] = high_surfaces
        reduced_surfaces_by_case[case_key] = reduced_surfaces
        surface_limits_by_case[case_key] = compute_rz_limits(curves)

    for row, case_key in enumerate(CASE_KEYS):
        for col, config_label in enumerate(REDUCED_CONFIG_LABELS):
            ax = axes[row][col]
            case_data = surface_data.get(case_key)
            reduced_surfaces = reduced_surfaces_by_case.get(case_key, {}).get(config_label)
            if case_data is None or reduced_surfaces is None:
                ax.set_axis_off()
                continue
            reduced_data = case_data["reduced"][config_label]
            high_label = f"{CASE_LABELS[case_key]} ({int(case_data['reference_parameter_count'])})"
            representative_label = (
                f"{CASE_LABELS[case_key]} ({int(reduced_data['parameter_count'])})"
            )
            _plot_surface_pair(
                ax,
                case_key=case_key,
                high_surfaces=high_surfaces_by_case[case_key],
                representative_surfaces=reduced_surfaces,
                high_label=high_label,
                representative_label=representative_label,
            )
            ylabel = "Z [m]" if col == 0 else ""
            if col == 0:
                ylabel = f"{CASE_LABELS[case_key]}\n{ylabel}"
            _style_axis(
                ax,
                title=surface_compare_title(config_label) if row == 0 else "",
                xlabel="R [m]" if row == len(CASE_KEYS) - 1 else "",
                ylabel=ylabel,
                yscale="linear",
            )
            limits = surface_limits_by_case[case_key]
            x0, x1 = limits[0]
            xpad_extra_fraction = SURFACE_COMPARE_XPAD_EXTRA_FRACTION
            ax.set_xlim(x0, x1 + (x1 - x0) * xpad_extra_fraction)
            ax.set_ylim(*limits[1])
            style_surface_compare_axis(ax, case_key=case_key)
            if col > 0:
                ax.tick_params(labelleft=False)


def surface_compare_title(config_label: str) -> str:
    if str(config_label) == "Medium":
        return r"$\bf{(c)}$ Reduced vs Ref" + "\n\n" + str(config_label)
    return str(config_label)


def style_surface_compare_axis(ax: plt.Axes, *, case_key: str) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=SURFACE_COMPARE_X_TICK_BINS))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=SURFACE_COMPARE_Y_TICK_BINS))
    ax.set_xticks(SURFACE_COMPARE_X_TICKS[case_key])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def align_surface_columns(
    fig: plt.Figure,
    surface_axes: list[list[plt.Axes]],
    *,
    gap: float = FIGURE_SURFACE_COLUMN_GAP,
) -> None:
    if not surface_axes:
        return
    fig.canvas.draw()
    bboxes_by_row = [[ax.get_position().frozen() for ax in row] for row in surface_axes]
    ncols = len(bboxes_by_row[0])
    column_widths = [max(float(row[col].width) for row in bboxes_by_row) for col in range(ncols)]
    x0 = min(float(row[0].x0) for row in bboxes_by_row)
    centers: list[float] = []
    x = x0
    for width in column_widths:
        centers.append(x + 0.5 * width)
        x += width + float(gap)
    for axes, row_bboxes in zip(surface_axes, bboxes_by_row, strict=True):
        for col, (ax, bbox) in enumerate(zip(axes, row_bboxes, strict=True)):
            ax.set_position([centers[col] - 0.5 * bbox.width, bbox.y0, bbox.width, bbox.height])


def _plot_surface_pair(
    ax: plt.Axes,
    *,
    case_key: str,
    high_surfaces: dict[float, np.ndarray],
    representative_surfaces: dict[float, np.ndarray],
    high_label: str,
    representative_label: str,
) -> None:
    for idx, level in enumerate(sorted(high_surfaces)):
        linewidth = (
            SURFACE_COMPARE_LINE_WIDTH
            if level < 1.0 - 1.0e-12
            else SURFACE_COMPARE_BOUNDARY_LINE_WIDTH
        )
        high_curve = close_curve(high_surfaces[level])
        rep_curve = close_curve(representative_surfaces[level])
        ax.plot(
            high_curve[:, 0],
            high_curve[:, 1],
            color=SURFACE_COMPARE_REFERENCE_COLOR,
            linestyle=SURFACE_COMPARE_REFERENCE_STYLE,
            linewidth=linewidth * SURFACE_COMPARE_REFERENCE_SCALE,
            label=(high_label if idx == 0 else None),
        )
        ax.plot(
            rep_curve[:, 0],
            rep_curve[:, 1],
            color=CASE_COLORS[case_key],
            linestyle=SURFACE_COMPARE_REPRESENTATIVE_STYLE,
            linewidth=linewidth,
            label=(representative_label if idx == 0 else None),
        )


def render(
    all_samples: dict[str, list[Sample]],
    frontiers: dict[str, list[Sample]],
    reference_a_by_case: dict[str, float],
    png_path: str | None,
    pdf_path: str | None,
    test_nr: int,
    test_nt: int,
    backend: str,
    selected_config_rows: list[tuple[str, float, Sample | None]] | None = None,
) -> None:
    apply_plot_style()
    selected_rows_by_case = selected_representative_rows_by_case(
        all_samples,
        reference_a_by_case,
        selected_config_rows=selected_config_rows,
    )
    high_representative_by_case = {
        case_key: sample
        for case_key, rows in selected_rows_by_case.items()
        for label, sample in rows
        if label == "High"
    }
    surface_data = build_surface_comparison_data(
        all_samples,
        selected_rows_by_case=selected_rows_by_case,
        backend=backend,
        test_nr=test_nr,
        test_nt=test_nt,
    )
    fig = plt.figure(
        figsize=FIGURE_SIZE,
        constrained_layout=FIGURE_CONSTRAINED_LAYOUT,
    )
    grid_spec = fig.add_gridspec(
        1,
        2,
        width_ratios=FIGURE_LEFT_RIGHT_WIDTH_RATIOS,
        wspace=FIGURE_WSPACE,
    )
    pareto_spec = grid_spec[0, 0].subgridspec(2, 1, hspace=FIGURE_PARETO_HSPACE)
    surface_spec = grid_spec[0, 1].subgridspec(
        len(CASE_KEYS),
        len(REDUCED_CONFIG_LABELS),
        wspace=FIGURE_SURFACE_WSPACE,
        hspace=FIGURE_SURFACE_HSPACE,
    )
    ax_time = fig.add_subplot(pareto_spec[0, 0])
    ax_param = fig.add_subplot(pareto_spec[1, 0], sharey=ax_time)
    surface_axes = [
        [fig.add_subplot(surface_spec[row, col]) for col in range(len(REDUCED_CONFIG_LABELS))]
        for row in range(len(CASE_KEYS))
    ]
    fig.set_constrained_layout_pads(wspace=FIGURE_WSPACE, hspace=FIGURE_HSPACE)
    max_parameter_count = 1
    time_legend_line_handles: dict[str, object] = {}
    for case_key in CASE_KEYS:
        normalizer = float(reference_a_by_case.get(case_key, 1.0))
        if not np.isfinite(normalizer) or normalizer <= 0.0:
            normalizer = 1.0
        samples = [
            sample
            for sample in all_samples[case_key]
            if np.isfinite(normalized_shape_error(sample, normalizer))
            and normalized_shape_error(sample, normalizer) > 0.0
            and not sample.is_exact_reference
        ]
        time_frontier = frontiers[case_key]
        parameter_frontier = parameter_error_frontier(all_samples[case_key])
        if not samples:
            continue
        samples_for_parameter_axis = [
            sample for sample in samples if int(sample.parameter_count) <= PARAMETER_COUNT_PLOT_MAX
        ]
        if samples_for_parameter_axis:
            max_parameter_count = max(
                max_parameter_count,
                max(int(sample.parameter_count) for sample in samples_for_parameter_axis),
            )

        representative = high_representative_by_case.get(case_key)
        representative_rows = selected_rows_by_case.get(case_key, [])
        _print_case_frontier_summary(
            case_key,
            time_frontier=time_frontier,
            parameter_frontier=parameter_frontier,
            representative=representative,
            normalizer_m=normalizer,
        )
        line_handle = _plot_time_case(
            ax_time,
            case_key=case_key,
            samples=samples,
            time_frontier=time_frontier,
            representative_rows=representative_rows,
            normalizer_m=normalizer,
        )
        if line_handle is not None:
            time_legend_line_handles[case_key] = line_handle
        _plot_parameter_case(
            ax_param,
            case_key=case_key,
            samples=samples,
            parameter_frontier=parameter_frontier,
            representative_rows=representative_rows,
            normalizer_m=normalizer,
        )

    ax_time.set_xscale("log")
    ax_time.xaxis.set_major_locator(FixedLocator([1.0, 10.0, 100.0]))
    ax_time.xaxis.set_major_formatter(LogFormatterMathtext())
    ax_time.set_xticks(
        [1.0, 10.0, 100.0],
        labels=[r"$10^0$", r"$10^1$", r"$10^2$"],
    )
    ax_time.set_xlim(TIME_XMIN, TIME_XMAX)
    ax_time.set_ylim(bottom=ERROR_AXIS_YMIN, top=ERROR_AXIS_YMAX)
    _style_axis(
        ax_time,
        title=r"$\bf{(a)}$ Time vs Error",
        xlabel="solve-only median time [ms]",
        ylabel=r"$E_{\mathrm{ref}}/a$",
    )
    remove_minor_ticks(ax_time)
    _style_case_config_legend(ax_time, time_legend_line_handles)

    ax_param.set_ylim(bottom=ERROR_AXIS_YMIN, top=ERROR_AXIS_YMAX)
    right_limit = min(
        PARAMETER_COUNT_PLOT_MAX,
        max_parameter_count + max(2, int(round(max_parameter_count * 0.04))),
    )
    ax_param.set_xlim(0.0, float(right_limit))
    ax_param.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    _style_axis(
        ax_param,
        title=r"$\bf{(b)}$ Parameter Count vs Error",
        xlabel="parameter count",
        ylabel=r"$E_{\mathrm{ref}}/a$",
    )
    remove_minor_ticks(ax_param)
    render_surface_comparison_grid(surface_axes, surface_data)
    align_surface_columns(fig, surface_axes)

    label_anchor_path = os.environ.get("FIG07_LABEL_ANCHORS_JSON")
    if label_anchor_path:
        fig.canvas.draw()
        canvas_to_save = float(SAVE_DPI) / float(fig.dpi)
        image_width = int(round(float(fig.get_figwidth()) * float(SAVE_DPI)))
        image_height = int(round(float(fig.get_figheight()) * float(SAVE_DPI)))
        anchors: dict[str, dict[str, object]] = {}
        for case_key, rows in selected_rows_by_case.items():
            normalizer = float(reference_a_by_case.get(case_key, 1.0))
            if not np.isfinite(normalizer) or normalizer <= 0.0:
                normalizer = 1.0
            for config_label, sample in rows:
                x_px, y_px_bottom = ax_time.transData.transform(
                    (
                        float(sample.elapsed_ms),
                        normalized_shape_error(sample, normalizer),
                    )
                )
                anchors[f"{case_key}:{config_label}"] = {
                    "case_key": case_key,
                    "case_label": CASE_LABELS[case_key],
                    "config_label": config_label,
                    "color": CASE_COLORS[case_key],
                    "x_px": float(x_px) * canvas_to_save,
                    "y_px": float(image_height) - float(y_px_bottom) * canvas_to_save,
                }
        label_anchor_dir = os.path.dirname(label_anchor_path)
        if label_anchor_dir:
            os.makedirs(label_anchor_dir, exist_ok=True)
        with open(label_anchor_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "image_width": image_width,
                    "image_height": image_height,
                    "point_scale": float(SAVE_DPI) / 72.0,
                    "anchors": anchors,
                },
                handle,
                indent=2,
            )

    save_figure_outputs(
        fig,
        png_path=png_path,
        pdf_path=pdf_path,
        dpi=SAVE_DPI,
        transparent=SAVE_TRANSPARENT,
    )
    plt.close(fig)


def write_json(
    all_samples: dict[str, list[Sample]],
    frontiers: dict[str, list[Sample]],
    json_stem: str,
    test_nr: int,
    test_nt: int,
    backend: str,
    sweep_mode: str,
) -> None:
    def build_payload(
        samples_by_case: dict[str, list[Sample]],
        frontiers_by_case: dict[str, list[Sample]],
    ) -> dict[str, object]:
        return {
            "sweep": {
                "route": "PF",
                "coordinate": "psin",
                "input_kind": "uniform",
                "constraint": "Ip",
                "aggregate_metric": AGGREGATE_METRIC_NAME,
                "error_unit": ERROR_UNIT,
                "test_nr": int(test_nr),
                "test_nt": int(test_nt),
                "backend": backend,
                "sweep_mode": str(sweep_mode),
                "signature_version": FULL_SWEEP_SIGNATURE_VERSION
                if str(sweep_mode) == "full"
                else "partial-table05",
                "max_configs_per_case": FULL_SWEEP_MAX_CONFIGS_PER_CASE
                if str(sweep_mode) == "full"
                else None,
                "progression_rule": (
                    "Selected representative configurations only: three reduced-order rows per case"
                    if str(sweep_mode) == "partial"
                    else (
                        "D-shape uses a pruned VEQ-ref lattice with independent radial/core levels "
                        "and stratified contiguous sine-prefix samples; H-mode/X-point use a <10k "
                        "VEQ-ref-bounded Halton core sweep with contiguous monotone "
                        "Fourier prefixes plus compact one-step neighborhoods "
                        "around the selected representative rows"
                    )
                ),
                "strategies": (
                    ["table05_selected"]
                    if str(sweep_mode) == "partial"
                    else [
                        "veq_ref_prune_full",
                        "veq_ref_prune_full_selected",
                        "veq_ref_prune_full_selected_nearby",
                    ]
                ),
                "length_bounds": {
                    case_key: {
                        "label": CASE_LABELS[case_key],
                        "min": dict(get_case_length_bounds(case_key)[0]),
                        "max": dict(get_case_length_bounds(case_key)[1]),
                    }
                    for case_key in samples_by_case
                },
            },
            "samples": {
                case_key: [asdict(sample) for sample in samples]
                for case_key, samples in samples_by_case.items()
            },
            "frontier": {
                case_key: [asdict(sample) for sample in frontiers_by_case.get(case_key, [])]
                for case_key in samples_by_case
            },
        }

    for case_key in all_samples:
        case_json_path = json_output_path(json_stem, case_key, sweep_mode)
        case_json_dir = os.path.dirname(case_json_path)
        if case_json_dir:
            os.makedirs(case_json_dir, exist_ok=True)
        case_payload = build_payload(
            {case_key: list(all_samples.get(case_key, []))},
            {case_key: list(frontiers.get(case_key, []))},
        )
        with open(case_json_path, "w", encoding="utf-8") as f:
            json.dump(case_payload, f, indent=2)


def _load_sample_list(entries: object) -> list[Sample] | None:
    if not isinstance(entries, list):
        return None
    samples: list[Sample] = []
    for entry in entries:
        if not isinstance(entry, dict):
            return None
        try:
            samples.append(Sample(**entry))
        except TypeError:
            return None
    return samples


def load_plot_data_bundle(
    json_stem: str,
    sweep_mode: str,
) -> tuple[dict[str, list[Sample]] | None, dict[str, list[Sample]] | None]:
    all_samples: dict[str, list[Sample]] = {}
    all_frontiers: dict[str, list[Sample]] = {}
    for case_key in CASE_KEYS:
        path = json_output_path(json_stem, case_key, sweep_mode)
        if not os.path.exists(path):
            return None, None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None, None
        if not isinstance(payload, dict):
            return None, None
        sweep_block = payload.get("sweep")
        if not isinstance(sweep_block, dict):
            return None, None
        if (
            str(sweep_mode) == "full"
            and sweep_block.get("signature_version") != FULL_SWEEP_SIGNATURE_VERSION
        ):
            return None, None
        samples_block = payload.get("samples")
        frontier_block = payload.get("frontier")
        if not isinstance(samples_block, dict) or not isinstance(frontier_block, dict):
            return None, None
        samples = _load_sample_list(samples_block.get(case_key))
        frontiers = _load_sample_list(frontier_block.get(case_key))
        if samples is None:
            return None, None
        if not samples:
            return None, None
        if frontiers is None:
            frontiers = pareto_frontier(
                samples, min_rel_improvement=FRONTIER_MIN_REL_ERROR_IMPROVEMENT
            )
        all_samples[case_key] = samples
        all_frontiers[case_key] = frontiers
    return all_samples, all_frontiers


def main() -> None:
    json_stem = str(DEFAULT_JSON_STEM)
    sweep_mode = str(RUN_SWEEP_MODE)
    cache_enabled = sweep_mode == "full"
    if sweep_mode == "full" and not bool(RUN_RECOMPUTE_FULL):
        all_samples, frontiers = load_plot_data_bundle(json_stem, sweep_mode)
        if all_samples is not None and frontiers is not None:
            print(
                "[pareto] loaded plot data from "
                f"{json_stem}_{sweep_mode}_{{solovev,chease,efit}}.json"
            )
            reference_a_by_case = compute_reference_a_by_case(backend=RUN_BACKEND)
            external_shape_errors = compute_selected_external_shape_errors(
                all_samples,
                backend=RUN_BACKEND,
                test_nr=RUN_TEST_NR,
                test_nt=RUN_TEST_NT,
                reference_a_by_case=reference_a_by_case,
            )
            print_fastest_config_table_body(all_samples, external_shape_errors, reference_a_by_case)
            render(
                all_samples,
                frontiers,
                reference_a_by_case,
                PNG_PATH,
                PDF_PATH,
                RUN_TEST_NR,
                RUN_TEST_NT,
                RUN_BACKEND,
            )
            write_representative_reduced_equilibria_from_cache(
                json_stem,
                sweep_mode,
                backend=RUN_BACKEND,
                test_nr=RUN_TEST_NR,
                test_nt=RUN_TEST_NT,
            )
            return
        expected = ", ".join(
            json_output_path(json_stem, case_key, sweep_mode) for case_key in CASE_KEYS
        )
        print(f"[pareto] missing or stale full Pareto JSON bundle; recomputing: {expected}")
    elif sweep_mode == "partial":
        print("[pareto] partial mode reruns selected 3x3 configs without Pareto cache I/O")

    benchmark = load_benchmark(RUN_BACKEND)
    progress = ParetoProgressDisplay(CASE_KEYS)
    case_signature_records: dict[str, list[SignatureRecord]] = {}
    for case_key in CASE_KEYS:
        min_lengths, max_lengths = get_case_length_bounds(case_key)
        case_seed = int(RUN_RANDOM_SIGNATURE_SEED) + sum(ord(ch) for ch in case_key)
        signature_records = generate_case_signatures(
            case_key,
            min_lengths,
            max_lengths,
            random_signature_count=RUN_RANDOM_SIGNATURE_COUNT,
            random_seed=case_seed,
            sweep_mode=sweep_mode,
        )
        case_signature_records[case_key] = signature_records
        progress.set_total(case_key, total=len(signature_records))
    all_samples: dict[str, list[Sample]] = {}
    try:
        for case_key in CASE_KEYS:
            all_samples[case_key] = sample_case(
                benchmark,
                case_key=case_key,
                backend=RUN_BACKEND,
                test_nr=RUN_TEST_NR,
                test_nt=RUN_TEST_NT,
                repeat_count=RUN_REPEAT_COUNT,
                random_signature_count=RUN_RANDOM_SIGNATURE_COUNT,
                random_signature_seed=RUN_RANDOM_SIGNATURE_SEED,
                signature_records=case_signature_records[case_key],
                progress=progress,
                sweep_mode=sweep_mode,
                case_workers=RUN_CASE_WORKERS,
                case_worker_inner_threads=RUN_CASE_WORKER_INNER_THREADS,
            )
            partial_frontiers = {
                key: pareto_frontier(samples, min_rel_improvement=RUN_FRONTIER_MIN_REL_IMPROVEMENT)
                for key, samples in all_samples.items()
            }
            if cache_enabled:
                write_json(
                    all_samples,
                    partial_frontiers,
                    json_stem,
                    RUN_TEST_NR,
                    RUN_TEST_NT,
                    RUN_BACKEND,
                    sweep_mode,
                )
    finally:
        progress.close()
    frontiers = {
        case_key: pareto_frontier(samples, min_rel_improvement=RUN_FRONTIER_MIN_REL_IMPROVEMENT)
        for case_key, samples in all_samples.items()
    }
    reference_a_by_case = compute_reference_a_by_case(backend=RUN_BACKEND)
    selected_config_rows = (
        selected_table05_config_rows(all_samples) if sweep_mode == "partial" else None
    )
    external_shape_errors = compute_selected_external_shape_errors(
        all_samples,
        backend=RUN_BACKEND,
        test_nr=RUN_TEST_NR,
        test_nt=RUN_TEST_NT,
        reference_a_by_case=reference_a_by_case,
        selected_config_rows=selected_config_rows,
    )
    print_fastest_config_table_body(
        all_samples,
        external_shape_errors,
        reference_a_by_case,
        selected_config_rows=selected_config_rows,
    )
    render(
        all_samples,
        frontiers,
        reference_a_by_case,
        PNG_PATH,
        PDF_PATH,
        RUN_TEST_NR,
        RUN_TEST_NT,
        RUN_BACKEND,
        selected_config_rows=selected_config_rows,
    )
    if cache_enabled:
        write_json(
            all_samples,
            frontiers,
            json_stem,
            RUN_TEST_NR,
            RUN_TEST_NT,
            RUN_BACKEND,
            sweep_mode,
        )
    write_representative_reduced_equilibria(
        all_samples,
        backend=RUN_BACKEND,
        test_nr=RUN_TEST_NR,
        test_nt=RUN_TEST_NT,
        sweep_mode=sweep_mode,
        reference_a_by_case=reference_a_by_case,
        selected_config_rows=selected_config_rows,
    )


if __name__ == "__main__":
    main()
