"""GEQDSK-backed benchmark and regression-driving script.

This file is intentionally script-oriented rather than pytest-oriented. It
loads one GEQDSK truth case, builds a calibrated PF reference solve, projects
that reference into the benchmark route/constraint matrix, and writes comparison
artifacts under ``tests/benchmark/geqdsk/``.

Note: The first run may be slower due to JIT compilation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Matplotlib is needed for robust GEQDSK contour extraction.  Use a headless
# backend so this benchmark can run in CI and non-interactive shells.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/veqpy_mplconfig")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


def _find_project_root(start: Path) -> Path:
    """Find the repository root when this file is run from tests/."""
    for candidate in (
        start.parent,
        *start.parents,
        Path.cwd().resolve(),
        *Path.cwd().resolve().parents,
    ):
        if (candidate / "pyproject.toml").exists() and (candidate / "veqpy").is_dir():
            return candidate
    # When the file is located at tests/benchmark_with_geqdsk.py, parents[1]
    # is the repository root.  This fallback also keeps the artifact importable
    # outside an actual checkout.
    try:
        return start.parents[1]
    except IndexError:
        return Path.cwd().resolve()


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
if (PROJECT_ROOT / "veqpy").is_dir() and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from veqpy.model.boundary import Boundary  # noqa: E402
from veqpy.model.geqdsk import Geqdsk  # noqa: E402
from veqpy.model.grid import Grid  # noqa: E402
from veqpy.operator import Operator, OperatorCase  # noqa: E402
from veqpy.solver import Solver, SolverConfig  # noqa: E402

try:  # Optional internal ABI metadata.  The benchmark works without it.
    import veqpy.engine.backend_abi as backend_abi
except Exception:  # pragma: no cover - only exercised on incompatible installs.
    backend_abi = None  # type: ignore[assignment]


MU0 = 4.0e-7 * math.pi
SCHEMA_VERSION = 1
DEFAULT_GEQDSK_NAME = "SOLOVEV.geqdsk"
PLOT = True


def _discover_project_root() -> Path:
    """Locate the VEQPy project root when run from tests/ or copied elsewhere."""

    script_path = Path(__file__).resolve()
    candidates: list[Path] = []
    for candidate in (
        script_path.parent,
        *script_path.parents,
        Path.cwd().resolve(),
        *Path.cwd().resolve().parents,
    ):
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "veqpy").exists():
            return candidate
        if (candidate / "data" / DEFAULT_GEQDSK_NAME).is_file() and (candidate / "tests").exists():
            return candidate

    # Expected placement is tests/benchmark_with_geqdsk.py; keep that case working
    # even in stripped-down checkouts where pyproject.toml is unavailable.
    if script_path.parent.name == "tests":
        return script_path.parent.parent
    return Path.cwd().resolve()


PROJECT_ROOT = _discover_project_root()
DEFAULT_GEQDSK_PATH = PROJECT_ROOT / "data" / DEFAULT_GEQDSK_NAME
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tests" / "benchmark" / "geqdsk"
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "tests" / "baselines"

BENCHMARK_MODES = ("PF", "PP", "PI", "PJ1", "PJ2", "PQ")
BENCHMARK_COORDINATES = ("rho", "psin")
BENCHMARK_INPUT_KINDS = ("uniform",)
BENCHMARK_MODE_CONSTRAINTS: Mapping[str, tuple[str, ...]] = {
    "PF": ("null", "Ip", "beta"),
    "PP": ("Ip_beta", "Ip", "beta", "null"),
    "PI": ("Ip_beta", "Ip", "beta", "null"),
    "PJ1": ("Ip_beta", "Ip", "beta", "null"),
    "PJ2": ("Ip_beta", "Ip", "beta", "null"),
    "PQ": ("Ip_beta", "Ip", "beta", "null"),
}

BOUNDARY_FIT_M = 10
BOUNDARY_FIT_N = 10
BOUNDARY_MAXTOL = 1.0

REFERENCE_NR = 64
REFERENCE_NT = 32
SOLVE_NR = 32
SOLVE_NT = 32
METRIC_NR = 96
METRIC_NT = 192
SOURCE_SAMPLE_COUNT = 51
PJ2_PSIN_SOURCE_SAMPLE_COUNT = 51

SOLVER_METHOD = "lm"
SOLVER_MAX_RESIDUAL = 1.0e-7
SOLVER_MAX_EVALUATIONS = 2000
SOLVER_INITIAL_POLICY = "homothetic"
ROUTE_SOLVER_INITIAL_POLICY: str | None = None
EQUIVALENCE_SHAPE_REL_RMS_MAX = 1.0e-2
EQUIVALENCE_PSI_R_REL_RMS_MAX = 5.0e-2
EQUIVALENCE_FF_PSI_REL_RMS_MAX = 5.0e-2

SURFACE_LEVELS = tuple(float(x) for x in np.linspace(0.1, 1.0, 10))
PSI_R_TRUTH_CONTOUR_LEVELS = tuple(float(x) for x in np.linspace(0.02, 1.0, 64))
PROFILE_RHO_EVAL = tuple(float(x) for x in np.linspace(0.05, 0.95, 64))
PROFILE_PSIN_EVAL = tuple(float(x) for x in np.linspace(0.05, 1.0, 64))
SHAPE_THETA_SAMPLE_COUNT = 256
SIGN_CHANGE_WINDOW = 12
EPS = 1.0e-14

# High-order but zero-valued shape coefficients, matching the GEQDSK workflow's
# philosophy while allowing the solver to represent realistic fixed boundaries.
GEQDSK_PROFILE_COEFFS: dict[str, list[float]] = {
    "psin": [0.0] * 10,
    "h": [0.0] * 10,
    "k": [0.0] * 10,
    "v": [0.0] * 10,
    "c0": [0.0] * 10,
    "c1": [0.0] * 5,
    "c2": [0.0] * 5,
    "c3": [0.0] * 5,
    "c4": [0.0] * 5,
    "c5": [0.0] * 5,
    "c6": [0.0] * 5,
    "c7": [0.0] * 5,
    "s1": [0.0] * 10,
    "s2": [0.0] * 5,
    "s3": [0.0] * 5,
    "s4": [0.0] * 5,
    "s5": [0.0] * 5,
    "s6": [0.0] * 5,
    "s7": [0.0] * 5,
    "s8": [0.0] * 5,
}


@dataclass(frozen=True)
class PreparedAxis:
    """Strictly increasing interpolation axis and aligned source values."""

    axis: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class GeqdskTruthBundle:
    geqdsk_path: Path
    geqdsk: Geqdsk
    boundary: Boundary
    psi_span: float
    r_nodes: np.ndarray
    z_nodes: np.ndarray
    psin_grid: np.ndarray
    surface_levels: tuple[float, ...]
    surfaces: dict[float, np.ndarray]
    rho_geom_axis: np.ndarray
    psin_geom_axis: np.ndarray
    psin_profile_axis: np.ndarray
    ff_psi_profile: np.ndarray
    minor_radius_reference: float


@dataclass(frozen=True)
class RouteSourceBundle:
    canonical_result: Any
    canonical_equilibrium: Any
    reference_grid: Grid
    profile_coeffs: dict[str, list[float]]
    profiles: dict[str, np.ndarray | float]
    rho_axis: np.ndarray
    psin_axis: np.ndarray


@dataclass(frozen=True)
class GeqdskBenchmarkSpec:
    mode: str
    coordinate: str
    constraint: str
    input_kind: str = "uniform"

    @property
    def case_name(self) -> str:
        return f"{self.mode}_{self.coordinate}_{self.input_kind}_{self.constraint}"


@dataclass(frozen=True)
class GeqdskBenchmarkResult:
    spec: GeqdskBenchmarkSpec
    success: bool
    solver_success: bool | None
    equivalence_success: bool | None
    error_type: str | None
    message: str | None
    traceback: str | None
    residual_norm_final: float | None
    function_evaluations: int | None
    iterations: int | None
    state_size: int | None
    avg_ms: float | None
    std_ms: float | None
    shape_rel_rms_error: float | None
    shape_rel_max_error: float | None
    lcfs_rel_rms_error: float | None
    axis_rel_error: float | None
    psi_r_rel_rms_error: float | None
    psi_r_rel_max_error: float | None
    ff_psi_rel_rms_error: float | None
    ff_psi_rel_max_error: float | None
    psi_r_head_sign_changes: int | None
    psi_r_tail_sign_changes: int | None
    ff_psi_head_sign_changes: int | None
    ff_psi_tail_sign_changes: int | None


# ---------------------------------------------------------------------------
# Small numeric and serialization helpers
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_json(value: Any) -> Any:
    """Convert numpy/dataclass-adjacent objects to standard JSON values."""

    if isinstance(value, np.ndarray):
        return [_safe_json(x) for x in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return value


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def _relative_rms_max(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    denominator: float | None = None,
) -> tuple[float | None, float | None]:
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    mask = np.isfinite(actual) & np.isfinite(reference)
    if mask.sum() == 0:
        return None, None
    diff = actual[mask] - reference[mask]
    if denominator is None:
        denominator = float(np.max(np.abs(reference[mask]))) if mask.any() else 0.0
    normalizer = max(abs(float(denominator)), EPS)
    return _as_float(_rms(diff) / normalizer), _as_float(np.max(np.abs(diff)) / normalizer)


def _sign_change_count(
    values: Sequence[float], *, head: bool, window: int = SIGN_CHANGE_WINDOW
) -> int:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return 0
    if arr.size > window:
        arr = arr[:window] if head else arr[-window:]
    diffs = np.diff(arr)
    scale = max(float(np.max(np.abs(arr))), 1.0)
    diffs = diffs[np.abs(diffs) > 1.0e-12 * scale]
    if diffs.size < 2:
        return 0
    signs = np.sign(diffs)
    return int(np.count_nonzero(signs[1:] * signs[:-1] < 0.0))


def _prepare_axis(axis: Sequence[float], values: Sequence[float]) -> PreparedAxis:
    axis_arr = np.asarray(axis, dtype=np.float64).reshape(-1)
    values_arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if axis_arr.size != values_arr.size:
        raise ValueError(f"axis/value length mismatch: {axis_arr.size} != {values_arr.size}")
    mask = np.isfinite(axis_arr) & np.isfinite(values_arr)
    axis_arr = axis_arr[mask]
    values_arr = values_arr[mask]
    if axis_arr.size < 2:
        raise ValueError("at least two finite interpolation points are required")
    order = np.argsort(axis_arr, kind="mergesort")
    axis_arr = axis_arr[order]
    values_arr = values_arr[order]

    unique_axis: list[float] = []
    unique_values: list[float] = []
    start = 0
    while start < axis_arr.size:
        stop = start + 1
        while stop < axis_arr.size and abs(axis_arr[stop] - axis_arr[start]) <= 1.0e-13:
            stop += 1
        unique_axis.append(float(np.mean(axis_arr[start:stop])))
        unique_values.append(float(np.mean(values_arr[start:stop])))
        start = stop

    prepared_axis = np.asarray(unique_axis, dtype=np.float64)
    prepared_values = np.asarray(unique_values, dtype=np.float64)
    if prepared_axis.size < 2:
        raise ValueError("interpolation axis collapsed to fewer than two unique points")
    return PreparedAxis(axis=prepared_axis, values=prepared_values)


def _profile_interp(
    axis: Sequence[float], values: Sequence[float], targets: Sequence[float]
) -> np.ndarray:
    prepared = _prepare_axis(axis, values)
    x = np.asarray(targets, dtype=np.float64)
    if prepared.axis.size >= 3:
        interpolant = PchipInterpolator(prepared.axis, prepared.values, extrapolate=True)
        return np.asarray(interpolant(x), dtype=np.float64)
    return np.interp(x, prepared.axis, prepared.values).astype(np.float64)


def _profile_derivative(
    axis: Sequence[float], values: Sequence[float], targets: Sequence[float]
) -> np.ndarray:
    prepared = _prepare_axis(axis, values)
    x = np.asarray(targets, dtype=np.float64)
    if prepared.axis.size >= 3:
        interpolant = PchipInterpolator(prepared.axis, prepared.values, extrapolate=True)
        return np.asarray(interpolant.derivative()(x), dtype=np.float64)
    slope = (prepared.values[-1] - prepared.values[0]) / (prepared.axis[-1] - prepared.axis[0])
    return np.full_like(x, slope, dtype=np.float64)


def _safe_divisor(values: Sequence[float], *, floor: float = EPS) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    sign = np.where(arr < 0.0, -1.0, 1.0)
    return np.where(np.abs(arr) > floor, arr, sign * floor)


def _truth_flux_orientation(truth: GeqdskTruthBundle, equilibrium: Any) -> float:
    psi_span = float(truth.psi_span)
    alpha2 = float(equilibrium.alpha2)
    if not math.isfinite(psi_span) or not math.isfinite(alpha2):
        return 1.0
    if abs(psi_span) <= EPS or abs(alpha2) <= EPS:
        return 1.0
    return 1.0 if psi_span * alpha2 > 0.0 else -1.0


def _ff_psi_from_equilibrium(equilibrium: Any) -> np.ndarray:
    psi_r = float(equilibrium.alpha2) * np.asarray(equilibrium.psin_r, dtype=np.float64)
    return np.asarray(equilibrium.FF_r, dtype=np.float64) / _safe_divisor(psi_r)


def _p_psi_from_equilibrium(equilibrium: Any) -> np.ndarray:
    psi_r = float(equilibrium.alpha2) * np.asarray(equilibrium.psin_r, dtype=np.float64)
    return np.asarray(equilibrium.P_r, dtype=np.float64) / _safe_divisor(psi_r)


def _pf_flux_direction_is_underdetermined(spec: "GeqdskBenchmarkSpec") -> bool:
    """Return whether a PF case lacks source data that fixes flux direction."""
    if spec.mode != "PF":
        return False
    if spec.constraint == "beta":
        return True
    return spec.coordinate == "rho" and spec.constraint == "null"


def _close_curve(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("curve points must have shape (N, 2)")
    if points.shape[0] == 0:
        return points
    if np.linalg.norm(points[0] - points[-1]) > 1.0e-10:
        points = np.vstack([points, points[0]])
    return points


def _polygon_signed_area(points: np.ndarray) -> float:
    points = _close_curve(points)
    if points.shape[0] < 4:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _polygon_area(points: np.ndarray) -> float:
    return abs(_polygon_signed_area(points))


def _point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    polygon = _close_curve(np.asarray(polygon, dtype=np.float64))
    if polygon.shape[0] < 4:
        return False
    x, y = point
    px = polygon[:, 0]
    py = polygon[:, 1]
    inside = False
    j = len(px) - 1
    for i in range(len(px)):
        yi, yj = py[i], py[j]
        xi, xj = px[i], px[j]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + EPS) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def _radial_curve_on_angles(
    points: np.ndarray,
    center: tuple[float, float],
    theta_eval: np.ndarray,
) -> np.ndarray:
    """Return radius r(theta) for a closed curve around ``center``.

    The curve is converted to polar coordinates around ``center``.  If the polar
    angle wraps around ±pi, the curve is unwrapped before interpolation.  This is
    more stable for VEQ/GEQDSK flux-surface comparison than nearest-neighbour
    distances and is sufficient for fixed-boundary tokamak surfaces.
    """

    points = _close_curve(np.asarray(points, dtype=np.float64))
    if points.shape[0] < 4:
        return np.full(theta_eval.shape, np.nan, dtype=np.float64)
    dx = points[:, 0] - center[0]
    dz = points[:, 1] - center[1]
    theta = np.unwrap(np.arctan2(dz, dx))
    radius = np.sqrt(dx * dx + dz * dz)
    mask = np.isfinite(theta) & np.isfinite(radius)
    theta = theta[mask]
    radius = radius[mask]
    if theta.size < 3:
        return np.full(theta_eval.shape, np.nan, dtype=np.float64)
    order = np.argsort(theta, kind="mergesort")
    theta = theta[order]
    radius = radius[order]

    # Remove repeated angles by averaging radii.
    prepared = _prepare_axis(theta, radius)
    theta = prepared.axis
    radius = prepared.values
    if theta.size < 3:
        return np.full(theta_eval.shape, np.nan, dtype=np.float64)

    period = 2.0 * math.pi
    base_min = theta[0]
    target = np.asarray(theta_eval, dtype=np.float64)
    while np.min(target) < base_min:
        target = target + period
    while np.max(target) > theta[-1]:
        target = target - period

    theta_ext = np.concatenate([theta - period, theta, theta + period])
    radius_ext = np.concatenate([radius, radius, radius])
    return _profile_interp(theta_ext, radius_ext, target)


def _resolve_path(path_like: Path | str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (Path.cwd() / path, PROJECT_ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _sanitize_case_key(path: Path) -> str:
    stem = path.stem or "GEQDSK"
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return sanitized or "GEQDSK"


# ---------------------------------------------------------------------------
# GEQDSK truth construction
# ---------------------------------------------------------------------------


def read_geqdsk(path: Path | str) -> Geqdsk:
    path = _resolve_path(path)
    geqdsk = Geqdsk()
    geqdsk.read_geqdsk(str(path))
    geqdsk.check()
    return geqdsk


def _extract_psin_contour(
    geqdsk: Geqdsk,
    r_nodes: np.ndarray,
    z_nodes: np.ndarray,
    psin_grid: np.ndarray,
    level: float,
) -> np.ndarray:
    if np.isclose(level, 1.0, rtol=0.0, atol=1.0e-12):
        return _close_curve(np.asarray(geqdsk.boundary, dtype=np.float64))

    min_level = float(np.nanmin(psin_grid))
    max_level = float(np.nanmax(psin_grid))
    if not (min_level <= level <= max_level):
        raise ValueError(
            f"psin level {level:.6g} is outside GEQDSK grid range "
            f"[{min_level:.6g}, {max_level:.6g}]"
        )

    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        # Matplotlib's contour expects Z with shape (len(y), len(x)); GEQDSK psi
        # is stored as (NR, NZ), therefore transpose it for R/Z plotting.
        contours = ax.contour(r_nodes, z_nodes, psin_grid.T, levels=[float(level)])
        segments = contours.allsegs[0] if contours.allsegs else []
    finally:
        plt.close(fig)

    cleaned: list[np.ndarray] = []
    for segment in segments:
        segment = np.asarray(segment, dtype=np.float64)
        if segment.ndim == 2 and segment.shape[0] >= 4 and segment.shape[1] == 2:
            segment = segment[np.all(np.isfinite(segment), axis=1)]
            if segment.shape[0] >= 4:
                cleaned.append(_close_curve(segment))
    if not cleaned:
        raise ValueError(f"no valid contour found for psin={level:.6g}")

    axis_point = (float(geqdsk.Raxis), float(geqdsk.Zaxis))
    containing_axis = [seg for seg in cleaned if _point_in_polygon(axis_point, seg)]
    candidates = containing_axis if containing_axis else cleaned
    return max(candidates, key=_polygon_area)


def _build_geometry_rho_axis(
    geqdsk: Geqdsk,
    r_nodes: np.ndarray,
    z_nodes: np.ndarray,
    psin_grid: np.ndarray,
    contour_levels: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    boundary = _close_curve(np.asarray(geqdsk.boundary, dtype=np.float64))
    lcfs_area = _polygon_area(boundary)
    if not math.isfinite(lcfs_area) or lcfs_area <= 0.0:
        raise ValueError("GEQDSK boundary area is not positive")

    rho_values: list[float] = [0.0]
    psin_values: list[float] = [0.0]

    for level in contour_levels:
        if level <= 0.0:
            continue
        try:
            contour = _extract_psin_contour(geqdsk, r_nodes, z_nodes, psin_grid, float(level))
        except Exception:
            continue
        area = _polygon_area(contour)
        if not math.isfinite(area) or area <= 0.0:
            continue
        rho = math.sqrt(max(area / lcfs_area, 0.0))
        if math.isfinite(rho):
            rho_values.append(float(np.clip(rho, 0.0, 1.0)))
            psin_values.append(float(np.clip(level, 0.0, 1.0)))

    # Ensure the edge is represented exactly even if contour extraction skipped it.
    rho_values.append(1.0)
    psin_values.append(1.0)

    prepared = _prepare_axis(rho_values, psin_values)
    rho = np.clip(prepared.axis, 0.0, 1.0)
    psin = np.clip(prepared.values, 0.0, 1.0)

    # PCHIP requires a strictly increasing axis; after clipping, uniqueness may be
    # damaged near the edge.  Prepare once more to guarantee that invariant.
    prepared = _prepare_axis(rho, psin)
    rho = prepared.axis
    psin = prepared.values

    # Keep the physical endpoints pinned and suppress small contour-noise
    # reversals before differentiating psin(rho).
    psin = np.maximum.accumulate(psin)
    if rho[0] > 0.0:
        rho = np.insert(rho, 0, 0.0)
        psin = np.insert(psin, 0, 0.0)
    else:
        rho[0] = 0.0
        psin[0] = 0.0
    if rho[-1] < 1.0:
        rho = np.append(rho, 1.0)
        psin = np.append(psin, 1.0)
    else:
        rho[-1] = 1.0
        psin[-1] = 1.0
    return rho.astype(np.float64), psin.astype(np.float64)


def build_geqdsk_truth_bundle(
    geqdsk_path: Path | str,
    *,
    boundary_fit_m: int = BOUNDARY_FIT_M,
    boundary_fit_n: int = BOUNDARY_FIT_N,
    boundary_maxtol: float = BOUNDARY_MAXTOL,
    surface_levels: Sequence[float] = SURFACE_LEVELS,
    psi_r_truth_contour_levels: Sequence[float] = PSI_R_TRUTH_CONTOUR_LEVELS,
) -> GeqdskTruthBundle:
    path = _resolve_path(geqdsk_path)
    geqdsk = read_geqdsk(path)
    boundary = Boundary.from_geqdsk(
        geqdsk,
        M=int(boundary_fit_m),
        N=int(boundary_fit_n),
        maxtol=float(boundary_maxtol),
    )

    psi_span = float(geqdsk.psi_bound - geqdsk.psi_axis)
    if not math.isfinite(psi_span) or abs(psi_span) <= EPS:
        raise ValueError("GEQDSK psi_bound - psi_axis is zero or non-finite")

    r_nodes = np.linspace(float(geqdsk.Rmin), float(geqdsk.Rmax), int(geqdsk.NR))
    z_nodes = np.linspace(float(geqdsk.Zmin), float(geqdsk.Zmax), int(geqdsk.NZ))
    psin_grid = (np.asarray(geqdsk.psi, dtype=np.float64) - float(geqdsk.psi_axis)) / psi_span

    surfaces: dict[float, np.ndarray] = {}
    for level in surface_levels:
        level = float(level)
        surfaces[level] = _extract_psin_contour(geqdsk, r_nodes, z_nodes, psin_grid, level)

    rho_geom_axis, psin_geom_axis = _build_geometry_rho_axis(
        geqdsk,
        r_nodes,
        z_nodes,
        psin_grid,
        psi_r_truth_contour_levels,
    )

    boundary_points = np.asarray(geqdsk.boundary, dtype=np.float64)
    minor_radius_reference = 0.5 * float(
        np.nanmax(boundary_points[:, 0]) - np.nanmin(boundary_points[:, 0])
    )
    if not math.isfinite(minor_radius_reference) or minor_radius_reference <= 0.0:
        minor_radius_reference = max(float(getattr(boundary, "a", 1.0)), 1.0)

    return GeqdskTruthBundle(
        geqdsk_path=path,
        geqdsk=geqdsk,
        boundary=boundary,
        psi_span=psi_span,
        r_nodes=r_nodes,
        z_nodes=z_nodes,
        psin_grid=psin_grid,
        surface_levels=tuple(float(x) for x in surface_levels),
        surfaces=surfaces,
        rho_geom_axis=rho_geom_axis,
        psin_geom_axis=psin_geom_axis,
        psin_profile_axis=np.linspace(0.0, 1.0, int(geqdsk.NR), dtype=np.float64),
        ff_psi_profile=np.asarray(geqdsk.FF_psi, dtype=np.float64).copy(),
        minor_radius_reference=minor_radius_reference,
    )


# ---------------------------------------------------------------------------
# Canonical PF reference and route input generation
# ---------------------------------------------------------------------------


def _copy_operator_case(case: OperatorCase) -> OperatorCase:
    copy_method = getattr(case, "copy", None)
    if callable(copy_method):
        return copy_method()
    return case


def _solver_config(
    max_evaluations: int = SOLVER_MAX_EVALUATIONS,
    *,
    initial_policy: str | None = SOLVER_INITIAL_POLICY,
) -> SolverConfig:
    return SolverConfig(
        method=SOLVER_METHOD,
        max_residual=SOLVER_MAX_RESIDUAL,
        max_evaluations=int(max_evaluations),
        initial_policy=initial_policy,
        enable_fallback=False,
        enable_verbose=False,
        enable_history=False,
    )


def _solve_operator_case(
    operator_case: OperatorCase,
    grid: Grid,
    *,
    max_evaluations: int = SOLVER_MAX_EVALUATIONS,
    initial_policy: str | None = SOLVER_INITIAL_POLICY,
) -> Solver:
    solver = Solver(
        operator=Operator(grid, _copy_operator_case(operator_case)),
        config=_solver_config(max_evaluations=max_evaluations, initial_policy=initial_policy),
    )
    solver.solve(
        enable_verbose=False,
        enable_history=False,
        initial_policy=initial_policy,
        enable_fallback=False,
    )
    return solver


def _grid(
    nr: int,
    nt: int,
    *,
    quadrature_scheme: str = "legendre",
    like: Grid | None = None,
) -> Grid:
    kwargs: dict[str, Any] = {
        "Nr": int(nr),
        "Nt": int(nt),
        "quadrature_scheme": quadrature_scheme,
    }
    if like is not None:
        for name in ("L_max", "M_max", "K_max"):
            if hasattr(like, name):
                kwargs[name] = getattr(like, name)
    return Grid(**kwargs)


def build_route_source_bundle(
    truth: GeqdskTruthBundle,
    *,
    reference_nr: int = REFERENCE_NR,
    reference_nt: int = REFERENCE_NT,
    max_evaluations: int = SOLVER_MAX_EVALUATIONS,
) -> RouteSourceBundle:
    reference_grid = _grid(reference_nr, reference_nt, quadrature_scheme="legendre")
    canonical_case = OperatorCase(
        route="PF",
        coordinate="psin",
        nodes="uniform",
        profile_coeffs=dict(GEQDSK_PROFILE_COEFFS),
        boundary=truth.boundary,
        heat_input=np.asarray(truth.geqdsk.P_psi, dtype=np.float64),
        current_input=np.asarray(truth.geqdsk.FF_psi, dtype=np.float64),
        Ip=float(truth.geqdsk.Ip),
    )
    canonical_solver = _solve_operator_case(
        canonical_case,
        reference_grid,
        max_evaluations=max_evaluations,
    )
    if canonical_solver.result is None:
        raise RuntimeError("canonical PF(psin)+Ip solve produced no SolverResult")

    canonical_state = _result_field(canonical_solver.result, "x", "state", "solution")
    if canonical_state is None:
        raise RuntimeError("canonical PF(psin)+Ip solve produced no packed state")
    canonical_profile_coeffs = canonical_solver.operator.build_coeffs(
        np.asarray(canonical_state, dtype=np.float64),
        include_none=False,
    )

    equilibrium = canonical_solver.build_equilibrium()
    rho_axis = np.asarray(equilibrium.rho, dtype=np.float64)
    psin_axis = np.asarray(equilibrium.psin, dtype=np.float64)

    psin_r = np.asarray(equilibrium.psin_r, dtype=np.float64)
    psin_r_safe = _safe_divisor(psin_r)
    psi_r = float(equilibrium.alpha2) * psin_r

    ffn_r = np.asarray(equilibrium.FFn_r, dtype=np.float64)
    pn_r = np.asarray(equilibrium.Pn_r, dtype=np.float64)
    ffn_psin = ffn_r / psin_r_safe
    pn_psin = pn_r / psin_r_safe

    ff_psi = _ff_psi_from_equilibrium(equilibrium)
    p_psi = _p_psi_from_equilibrium(equilibrium)

    q = np.asarray(equilibrium.q, dtype=np.float64)
    qn = q * 0.1

    profiles: dict[str, np.ndarray | float] = {
        "rho": rho_axis,
        "psin": psin_axis,
        "psin_r": psin_r,
        "psi_r": psi_r,
        "FFn_psin": ffn_psin,
        "FF_psi": ff_psi,
        "FFn_r": ffn_r,
        "FF_r": np.asarray(equilibrium.FF_r, dtype=np.float64),
        "setup_Pn_psin": pn_psin / MU0,
        "setup_Pn_r": pn_r / MU0,
        "P_psi": p_psi,
        "P_r": np.asarray(equilibrium.P_r, dtype=np.float64),
        "Itor": np.asarray(equilibrium.Itor, dtype=np.float64),
        "jtor": np.asarray(equilibrium.jtor, dtype=np.float64),
        "jpara": np.asarray(equilibrium.jpara, dtype=np.float64),
        "q": q,
        "qn": qn,
        "Ip_constraint": float(equilibrium.Ip),
        "beta_constraint": float(equilibrium.beta_t),
    }

    return RouteSourceBundle(
        canonical_result=canonical_solver.result,
        canonical_equilibrium=equilibrium,
        reference_grid=reference_grid,
        profile_coeffs=canonical_profile_coeffs,
        profiles=profiles,
        rho_axis=rho_axis,
        psin_axis=psin_axis,
    )


def _constraint_route_domains(constraint: str) -> tuple[str, str]:
    if constraint == "Ip_beta":
        return "normalized", "normalized"
    if constraint == "Ip":
        return "normalized", "physical"
    if constraint == "beta":
        return "physical", "normalized"
    if constraint == "null":
        return "physical", "physical"
    raise ValueError(f"unsupported constraint: {constraint!r}")


def _pressure_keys_for_coordinate(coordinate: str) -> tuple[str, str]:
    if coordinate == "rho":
        return "setup_Pn_r", "P_r"
    if coordinate == "psin":
        return "setup_Pn_psin", "P_psi"
    raise ValueError(f"unsupported coordinate: {coordinate!r}")


def _profile_coeffs_for_case(
    spec: GeqdskBenchmarkSpec,
    initial_coeffs: dict[str, list[float]] | None = None,
) -> dict[str, list[float]]:
    coeffs = {key: list(values) for key, values in GEQDSK_PROFILE_COEFFS.items()}

    # ``psin`` is not universally accepted as an active profile.  Only non-PJ2
    # psin/uniform routes need it to query source samples at the current optimized
    # flux coordinate; psin/grid inputs are already materialized on operator nodes.
    # Use backend ABI metadata when available and fall back to the known develop
    # route set otherwise.
    route_key = (spec.mode, spec.coordinate, spec.input_kind)
    if backend_abi is not None:
        psin_required_keys = getattr(
            backend_abi,
            "PROFILE_OWNED_PSIN_ROUTE_KEYS",
            frozenset(),
        )
    else:  # pragma: no cover - defensive compatibility path.
        psin_required_keys = frozenset(
            {
                ("PF", "psin", "uniform"),
                ("PP", "psin", "uniform"),
                ("PI", "psin", "uniform"),
                ("PJ1", "psin", "uniform"),
                ("PQ", "psin", "uniform"),
            }
        )
    if route_key in psin_required_keys:
        coeffs.setdefault("psin", [0.0] * 10)
    else:
        coeffs.pop("psin", None)

    if spec.mode == "PJ2":
        coeffs.setdefault("F", [0.0] * 5)
    if initial_coeffs:
        for name in tuple(coeffs):
            values = initial_coeffs.get(name)
            if values is not None and len(values) == len(coeffs[name]):
                coeffs[name] = list(values)
    return coeffs


def _build_mode_input_profiles(
    source: RouteSourceBundle,
    spec: GeqdskBenchmarkSpec,
) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
    profiles = source.profiles
    pressure_norm_key, pressure_phys_key = _pressure_keys_for_coordinate(spec.coordinate)

    if spec.mode == "PF":
        # PF has no simultaneous Ip+beta branch in the synthetic benchmark.  For
        # single constraints, the constrained scale is carried by Ip/beta; for
        # the unconstrained branch the input profiles must carry the physical
        # source scale.
        use_setup_inputs = spec.constraint in {"Ip", "beta"}
        if spec.coordinate == "rho":
            current_key = "FFn_r" if use_setup_inputs else "FF_r"
        elif spec.coordinate == "psin":
            current_key = "FFn_psin" if use_setup_inputs else "FF_psi"
        else:
            raise ValueError(f"unsupported coordinate: {spec.coordinate!r}")
        heat_key = pressure_norm_key if use_setup_inputs else pressure_phys_key
    else:
        driver_domain, pressure_domain = _constraint_route_domains(spec.constraint)
        pressure_key = pressure_norm_key if pressure_domain == "normalized" else pressure_phys_key
        if spec.mode == "PP":
            current_key = "psin_r" if driver_domain == "normalized" else "psi_r"
        elif spec.mode == "PI":
            current_key = "Itor"
        elif spec.mode == "PJ1":
            current_key = "jtor"
        elif spec.mode == "PJ2":
            current_key = "jpara"
        elif spec.mode == "PQ":
            current_key = "qn" if driver_domain == "normalized" else "q"
        else:
            raise ValueError(f"unsupported mode: {spec.mode!r}")
        heat_key = pressure_key

    heat_profile = np.asarray(profiles[heat_key], dtype=np.float64)
    current_profile = np.asarray(profiles[current_key], dtype=np.float64)

    if spec.constraint in {"Ip", "Ip_beta"}:
        ip = float(profiles["Ip_constraint"])
    else:
        ip = None
    if spec.constraint in {"beta", "Ip_beta"}:
        beta = float(profiles["beta_constraint"])
    else:
        beta = None
    return heat_profile, current_profile, ip, beta


def _uniform_source_axis(spec: GeqdskBenchmarkSpec, sample_count: int) -> np.ndarray:
    if spec.input_kind != "uniform":
        raise ValueError(f"unsupported input_kind: {spec.input_kind!r}")
    axis = np.linspace(0.0, 1.0, int(sample_count), dtype=np.float64)
    # The PP/psin source route uses sqrt(psin) as its source parameter; sampling
    # uniformly in that parameter corresponds to psin = x^2.
    if spec.mode == "PP" and spec.coordinate == "psin":
        return axis * axis
    return axis


def _source_sample_count_for_case(spec: GeqdskBenchmarkSpec, sample_count: int) -> int:
    if spec.mode == "PJ2" and spec.coordinate == "psin" and spec.constraint in {"Ip", "Ip_beta"}:
        return max(int(sample_count), PJ2_PSIN_SOURCE_SAMPLE_COUNT)
    return int(sample_count)


def build_operator_case_from_source(
    truth: GeqdskTruthBundle,
    source: RouteSourceBundle,
    spec: GeqdskBenchmarkSpec,
    *,
    source_sample_count: int = SOURCE_SAMPLE_COUNT,
) -> OperatorCase:
    heat_profile, current_profile, ip, beta = _build_mode_input_profiles(source, spec)
    source_axis = source.rho_axis if spec.coordinate == "rho" else source.psin_axis
    case_sample_count = _source_sample_count_for_case(spec, source_sample_count)
    target_axis = _uniform_source_axis(spec, case_sample_count)

    heat_input = _profile_interp(source_axis, heat_profile, target_axis)
    current_input = _profile_interp(source_axis, current_profile, target_axis)

    return OperatorCase(
        route=spec.mode,
        coordinate=spec.coordinate,
        nodes=spec.input_kind,
        profile_coeffs=_profile_coeffs_for_case(spec, source.profile_coeffs),
        boundary=truth.boundary,
        heat_input=heat_input,
        current_input=current_input,
        Ip=ip,
        beta=beta,
    )


def iter_benchmark_specs() -> Iterable[GeqdskBenchmarkSpec]:
    for mode in BENCHMARK_MODES:
        for coordinate in BENCHMARK_COORDINATES:
            for input_kind in BENCHMARK_INPUT_KINDS:
                for constraint in BENCHMARK_MODE_CONSTRAINTS[mode]:
                    yield GeqdskBenchmarkSpec(
                        mode=mode,
                        coordinate=coordinate,
                        input_kind=input_kind,
                        constraint=constraint,
                    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _equilibrium_surface_at_psin(equilibrium: Any, psin_level: float) -> np.ndarray:
    rho_axis = np.asarray(equilibrium.rho, dtype=np.float64)
    psin_axis = np.asarray(equilibrium.psin, dtype=np.float64)
    if rho_axis.size < 2 or psin_axis.size != rho_axis.size:
        raise ValueError("equilibrium rho/psin axes are invalid")

    if psin_level <= float(np.nanmin(psin_axis)):
        rho_level = float(rho_axis[np.nanargmin(psin_axis)])
    elif psin_level >= float(np.nanmax(psin_axis)):
        rho_level = float(rho_axis[np.nanargmax(psin_axis)])
    else:
        rho_level = float(_profile_interp(psin_axis, rho_axis, [psin_level])[0])
    rho_level = float(np.clip(rho_level, np.nanmin(rho_axis), np.nanmax(rho_axis)))

    r = np.asarray(equilibrium.R, dtype=np.float64)
    z = np.asarray(equilibrium.Z, dtype=np.float64)
    if r.ndim != 2 or z.ndim != 2 or r.shape != z.shape:
        raise ValueError("equilibrium R/Z arrays must be two-dimensional and aligned")

    surface_r = np.empty(r.shape[1], dtype=np.float64)
    surface_z = np.empty(z.shape[1], dtype=np.float64)
    for j in range(r.shape[1]):
        surface_r[j] = _profile_interp(rho_axis, r[:, j], [rho_level])[0]
        surface_z[j] = _profile_interp(rho_axis, z[:, j], [rho_level])[0]
    return _close_curve(np.column_stack([surface_r, surface_z]))


def compute_shape_metrics(
    truth: GeqdskTruthBundle,
    equilibrium: Any,
    *,
    theta_sample_count: int = SHAPE_THETA_SAMPLE_COUNT,
) -> dict[str, float | None]:
    theta_eval = np.linspace(-math.pi, math.pi, int(theta_sample_count), endpoint=False)
    center = (float(truth.geqdsk.Raxis), float(truth.geqdsk.Zaxis))
    normalizer = max(float(truth.minor_radius_reference), EPS)

    level_rms: list[float] = []
    level_max: list[float] = []
    lcfs_rms: float | None = None

    for level in truth.surface_levels:
        try:
            reference_points = truth.surfaces[float(level)]
            actual_points = _equilibrium_surface_at_psin(equilibrium, float(level))
            reference_radius = _radial_curve_on_angles(reference_points, center, theta_eval)
            actual_radius = _radial_curve_on_angles(actual_points, center, theta_eval)
            mask = np.isfinite(reference_radius) & np.isfinite(actual_radius)
            if mask.sum() == 0:
                continue
            diff = actual_radius[mask] - reference_radius[mask]
            rms = _rms(diff)
            max_abs = float(np.max(np.abs(diff)))
            level_rms.append(rms)
            level_max.append(max_abs)
            if np.isclose(level, 1.0, rtol=0.0, atol=1.0e-12):
                lcfs_rms = rms
        except Exception:
            continue

    if level_rms:
        shape_rms = float(np.sqrt(np.mean(np.asarray(level_rms) ** 2)) / normalizer)
        shape_max = float(np.max(level_max) / normalizer)
    else:
        shape_rms = float("nan")
        shape_max = float("nan")

    try:
        r_axis_actual = float(np.asarray(equilibrium.R, dtype=np.float64)[0, 0])
        z_axis_actual = float(np.asarray(equilibrium.Z, dtype=np.float64)[0, 0])
        axis_rel = (
            math.hypot(
                r_axis_actual - float(truth.geqdsk.Raxis),
                z_axis_actual - float(truth.geqdsk.Zaxis),
            )
            / normalizer
        )
    except Exception:
        axis_rel = float("nan")

    return {
        "shape_rel_rms_error": _as_float(shape_rms),
        "shape_rel_max_error": _as_float(shape_max),
        "lcfs_rel_rms_error": _as_float(None if lcfs_rms is None else lcfs_rms / normalizer),
        "axis_rel_error": _as_float(axis_rel),
    }


def compute_profile_metrics(
    truth: GeqdskTruthBundle,
    equilibrium: Any,
    *,
    truth_orientation: float,
    align_flux_orientation: bool = False,
    rho_eval: Sequence[float] = PROFILE_RHO_EVAL,
    psin_eval: Sequence[float] = PROFILE_PSIN_EVAL,
) -> dict[str, float | int | None]:
    rho_eval_arr = np.asarray(rho_eval, dtype=np.float64)
    psin_eval_arr = np.asarray(psin_eval, dtype=np.float64)

    psi_r_actual_axis = np.asarray(equilibrium.rho, dtype=np.float64)
    psi_r_actual_profile = float(equilibrium.alpha2) * np.asarray(
        equilibrium.psin_r, dtype=np.float64
    )
    psi_r_actual = _profile_interp(psi_r_actual_axis, psi_r_actual_profile, rho_eval_arr)

    ff_psi_actual_axis = np.asarray(equilibrium.psin, dtype=np.float64)
    ff_psi_actual_profile = _ff_psi_from_equilibrium(equilibrium)
    ff_psi_actual = _profile_interp(ff_psi_actual_axis, ff_psi_actual_profile, psin_eval_arr)

    if align_flux_orientation:
        current_orientation = float(truth_orientation)
        flipped_orientation = -current_orientation
        current_score = _profile_orientation_error_score(
            truth,
            psi_r_actual,
            ff_psi_actual,
            orientation=current_orientation,
            rho_eval=rho_eval_arr,
            psin_eval=psin_eval_arr,
        )
        flipped_score = _profile_orientation_error_score(
            truth,
            psi_r_actual,
            ff_psi_actual,
            orientation=flipped_orientation,
            rho_eval=rho_eval_arr,
            psin_eval=psin_eval_arr,
        )
        if flipped_score < current_score:
            truth_orientation = flipped_orientation

    psi_r_truth = (
        truth_orientation
        * truth.psi_span
        * _profile_derivative(
            truth.rho_geom_axis,
            truth.psin_geom_axis,
            rho_eval_arr,
        )
    )
    psi_r_rms, psi_r_max = _relative_rms_max(psi_r_actual, psi_r_truth)

    ff_psi_truth = truth_orientation * _profile_interp(
        truth.psin_profile_axis,
        truth.ff_psi_profile,
        psin_eval_arr,
    )
    ff_psi_rms, ff_psi_max = _relative_rms_max(ff_psi_actual, ff_psi_truth)

    return {
        "psi_r_rel_rms_error": psi_r_rms,
        "psi_r_rel_max_error": psi_r_max,
        "ff_psi_rel_rms_error": ff_psi_rms,
        "ff_psi_rel_max_error": ff_psi_max,
        "psi_r_head_sign_changes": _sign_change_count(psi_r_actual_profile, head=True),
        "psi_r_tail_sign_changes": _sign_change_count(psi_r_actual_profile, head=False),
        "ff_psi_head_sign_changes": _sign_change_count(ff_psi_actual_profile, head=True),
        "ff_psi_tail_sign_changes": _sign_change_count(ff_psi_actual_profile, head=False),
    }


def _profile_orientation_error_score(
    truth: GeqdskTruthBundle,
    psi_r_actual: np.ndarray,
    ff_psi_actual: np.ndarray,
    *,
    orientation: float,
    rho_eval: np.ndarray,
    psin_eval: np.ndarray,
) -> float:
    psi_r_truth = (
        float(orientation)
        * truth.psi_span
        * _profile_derivative(truth.rho_geom_axis, truth.psin_geom_axis, rho_eval)
    )
    ff_psi_truth = float(orientation) * _profile_interp(
        truth.psin_profile_axis,
        truth.ff_psi_profile,
        psin_eval,
    )
    psi_r_rms, _ = _relative_rms_max(psi_r_actual, psi_r_truth)
    ff_psi_rms, _ = _relative_rms_max(ff_psi_actual, ff_psi_truth)
    return float(psi_r_rms) + float(ff_psi_rms)


def compute_all_metrics(
    truth: GeqdskTruthBundle,
    equilibrium: Any,
    solve_grid: Grid,
    *,
    truth_orientation: float,
    align_flux_orientation: bool = False,
    metric_nr: int = METRIC_NR,
    metric_nt: int = METRIC_NT,
) -> dict[str, float | int | None]:
    metric_grid = _grid(
        metric_nr,
        metric_nt,
        quadrature_scheme="uniform",
        like=solve_grid,
    )
    metric_equilibrium = equilibrium.resample(metric_grid)
    metrics: dict[str, float | int | None] = {}
    metrics.update(compute_shape_metrics(truth, metric_equilibrium))
    metrics.update(
        compute_profile_metrics(
            truth,
            equilibrium,
            truth_orientation=truth_orientation,
            align_flux_orientation=align_flux_orientation,
        )
    )
    return metrics


def _metric_within_limit(value: float | int | None, limit: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric <= limit


def _equivalence_success(metrics: Mapping[str, float | int | None]) -> bool:
    return (
        _metric_within_limit(
            metrics.get("shape_rel_rms_error"),
            EQUIVALENCE_SHAPE_REL_RMS_MAX,
        )
        and _metric_within_limit(
            metrics.get("psi_r_rel_rms_error"),
            EQUIVALENCE_PSI_R_REL_RMS_MAX,
        )
        and _metric_within_limit(
            metrics.get("ff_psi_rel_rms_error"),
            EQUIVALENCE_FF_PSI_REL_RMS_MAX,
        )
    )


def _equivalence_failure_message(metrics: Mapping[str, float | int | None]) -> str:
    return (
        "equivalence metric threshold exceeded: "
        f"shape<={EQUIVALENCE_SHAPE_REL_RMS_MAX:.1e}, "
        f"psi_r<={EQUIVALENCE_PSI_R_REL_RMS_MAX:.1e}, "
        f"FF_psi<={EQUIVALENCE_FF_PSI_REL_RMS_MAX:.1e}; "
        f"got shape={_format_float(metrics.get('shape_rel_rms_error'), 3)}, "
        f"psi_r={_format_float(metrics.get('psi_r_rel_rms_error'), 3)}, "
        f"FF_psi={_format_float(metrics.get('ff_psi_rel_rms_error'), 3)}"
    )


# ---------------------------------------------------------------------------
# Case execution and plotting
# ---------------------------------------------------------------------------


def _result_field(result: Any, *names: str) -> Any:
    for name in names:
        if hasattr(result, name):
            return getattr(result, name)
    return None


def _elapsed_ms(result: Any) -> float | None:
    elapsed = _result_field(result, "elapsed_ms", "elapsed_milliseconds")
    if elapsed is not None:
        return _as_float(elapsed)
    elapsed_raw = _result_field(result, "elapsed")
    if elapsed_raw is None:
        return None
    # Current VEQPy benchmark reports result.elapsed / 1000 as milliseconds.
    return _as_float(float(elapsed_raw) / 1000.0)


def _plot_case_comparison(
    truth: GeqdskTruthBundle,
    equilibrium: Any,
    solve_grid: Grid,
    spec: GeqdskBenchmarkSpec,
    output_path: Path,
    *,
    truth_orientation: float,
    align_flux_orientation: bool = False,
    metric_nr: int = METRIC_NR,
    metric_nt: int = METRIC_NT,
) -> None:
    metric_grid = _grid(metric_nr, metric_nt, quadrature_scheme="uniform", like=solve_grid)
    metric_equilibrium = equilibrium.resample(metric_grid)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax_shape, ax_psi_r, ax_ff = axes

    for level in truth.surface_levels:
        ref = truth.surfaces[float(level)]
        act = _equilibrium_surface_at_psin(metric_equilibrium, float(level))
        ax_shape.plot(ref[:, 0], ref[:, 1], linewidth=1.0, alpha=0.6)
        ax_shape.plot(act[:, 0], act[:, 1], linewidth=0.8, linestyle="--", alpha=0.8)
    ax_shape.set_aspect("equal", adjustable="box")
    ax_shape.set_title("GEQDSK / VEQ surfaces")
    ax_shape.set_xlabel("R")
    ax_shape.set_ylabel("Z")

    rho_eval = np.asarray(PROFILE_RHO_EVAL, dtype=np.float64)
    psi_r_actual = _profile_interp(
        np.asarray(equilibrium.rho, dtype=np.float64),
        float(equilibrium.alpha2) * np.asarray(equilibrium.psin_r, dtype=np.float64),
        rho_eval,
    )
    psin_eval = np.asarray(PROFILE_PSIN_EVAL, dtype=np.float64)
    ff_actual = _profile_interp(
        np.asarray(equilibrium.psin, dtype=np.float64),
        _ff_psi_from_equilibrium(equilibrium),
        psin_eval,
    )
    if align_flux_orientation:
        current_score = _profile_orientation_error_score(
            truth,
            psi_r_actual,
            ff_actual,
            orientation=truth_orientation,
            rho_eval=rho_eval,
            psin_eval=psin_eval,
        )
        flipped_score = _profile_orientation_error_score(
            truth,
            psi_r_actual,
            ff_actual,
            orientation=-truth_orientation,
            rho_eval=rho_eval,
            psin_eval=psin_eval,
        )
        if flipped_score < current_score:
            truth_orientation = -truth_orientation

    psi_r_truth = (
        truth_orientation
        * truth.psi_span
        * _profile_derivative(
            truth.rho_geom_axis,
            truth.psin_geom_axis,
            rho_eval,
        )
    )
    ax_psi_r.plot(rho_eval, psi_r_truth, linewidth=1.2, label="GEQDSK")
    ax_psi_r.plot(rho_eval, psi_r_actual, linewidth=1.0, linestyle="--", label="VEQ")
    ax_psi_r.set_title("psi_r(rho)")
    ax_psi_r.set_xlabel("rho")
    ax_psi_r.legend(loc="best", fontsize=8)

    ff_truth = truth_orientation * _profile_interp(
        truth.psin_profile_axis,
        truth.ff_psi_profile,
        psin_eval,
    )
    ax_ff.plot(psin_eval, ff_truth, linewidth=1.2, label="GEQDSK")
    ax_ff.plot(psin_eval, ff_actual, linewidth=1.0, linestyle="--", label="VEQ")
    ax_ff.set_title("FF_psi(psin)")
    ax_ff.set_xlabel("psin")
    ax_ff.legend(loc="best", fontsize=8)

    fig.suptitle(spec.case_name)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_benchmark_case(
    truth: GeqdskTruthBundle,
    source: RouteSourceBundle,
    spec: GeqdskBenchmarkSpec,
    *,
    solve_nr: int = SOLVE_NR,
    solve_nt: int = SOLVE_NT,
    metric_nr: int = METRIC_NR,
    metric_nt: int = METRIC_NT,
    source_sample_count: int = SOURCE_SAMPLE_COUNT,
    max_evaluations: int = SOLVER_MAX_EVALUATIONS,
    timing_repeats: int = 0,
    plots_dir: Path | None = None,
) -> GeqdskBenchmarkResult:
    solve_grid = _grid(solve_nr, solve_nt, quadrature_scheme="legendre")
    timing_values: list[float] = []

    try:
        operator_case = build_operator_case_from_source(
            truth,
            source,
            spec,
            source_sample_count=source_sample_count,
        )

        solver = _solve_operator_case(
            operator_case,
            solve_grid,
            max_evaluations=max_evaluations,
            initial_policy=ROUTE_SOLVER_INITIAL_POLICY,
        )
        result = solver.result
        if result is None:
            raise RuntimeError("Solver produced no SolverResult")
        elapsed = _elapsed_ms(result)
        if elapsed is not None:
            timing_values.append(elapsed)
        equilibrium = solver.build_equilibrium()
        truth_orientation = _truth_flux_orientation(truth, source.canonical_equilibrium)
        align_flux_orientation = _pf_flux_direction_is_underdetermined(spec)

        # Optional timing repeats are independent solves so warm-start leakage does
        # not hide regressions.  Metrics are computed from the first successful
        # run to keep the payload deterministic.
        for _ in range(max(int(timing_repeats), 0)):
            repeat_solver = _solve_operator_case(
                operator_case,
                solve_grid,
                max_evaluations=max_evaluations,
                initial_policy=ROUTE_SOLVER_INITIAL_POLICY,
            )
            repeat_elapsed = _elapsed_ms(repeat_solver.result)
            if repeat_elapsed is not None:
                timing_values.append(repeat_elapsed)

        metrics = compute_all_metrics(
            truth,
            equilibrium,
            solve_grid,
            truth_orientation=truth_orientation,
            align_flux_orientation=align_flux_orientation,
            metric_nr=metric_nr,
            metric_nt=metric_nt,
        )

        if plots_dir is not None:
            _plot_case_comparison(
                truth,
                equilibrium,
                solve_grid,
                spec,
                plots_dir / f"{spec.case_name}_compare.png",
                truth_orientation=truth_orientation,
                align_flux_orientation=align_flux_orientation,
                metric_nr=metric_nr,
                metric_nt=metric_nt,
            )

        state = _result_field(result, "x", "state", "solution")
        state_size = len(state) if state is not None and hasattr(state, "__len__") else None
        avg_ms = float(np.mean(timing_values)) if timing_values else None
        std_ms = float(np.std(timing_values)) if len(timing_values) > 1 else None

        solver_success = bool(_result_field(result, "success"))
        equivalence_success = _equivalence_success(metrics) if solver_success else False
        case_success = solver_success and equivalence_success
        solver_message = str(_result_field(result, "message") or "")
        return GeqdskBenchmarkResult(
            spec=spec,
            success=case_success,
            solver_success=solver_success,
            equivalence_success=equivalence_success,
            error_type=(
                None
                if case_success
                else ("SolverResultFailure" if not solver_success else "EquivalenceMetricFailure")
            ),
            message=(
                solver_message
                if not solver_success or case_success
                else _equivalence_failure_message(metrics)
            ),
            traceback=None,
            residual_norm_final=_as_float(
                _result_field(result, "residual_norm_final", "residual_norm")
            ),
            function_evaluations=_as_int(
                _result_field(result, "function_evaluations", "nfev", "n_function_evaluations")
            ),
            iterations=_as_int(_result_field(result, "iterations", "nit", "n_iterations")),
            state_size=_as_int(state_size),
            avg_ms=_as_float(avg_ms),
            std_ms=_as_float(std_ms),
            shape_rel_rms_error=_as_float(metrics.get("shape_rel_rms_error")),
            shape_rel_max_error=_as_float(metrics.get("shape_rel_max_error")),
            lcfs_rel_rms_error=_as_float(metrics.get("lcfs_rel_rms_error")),
            axis_rel_error=_as_float(metrics.get("axis_rel_error")),
            psi_r_rel_rms_error=_as_float(metrics.get("psi_r_rel_rms_error")),
            psi_r_rel_max_error=_as_float(metrics.get("psi_r_rel_max_error")),
            ff_psi_rel_rms_error=_as_float(metrics.get("ff_psi_rel_rms_error")),
            ff_psi_rel_max_error=_as_float(metrics.get("ff_psi_rel_max_error")),
            psi_r_head_sign_changes=_as_int(metrics.get("psi_r_head_sign_changes")),
            psi_r_tail_sign_changes=_as_int(metrics.get("psi_r_tail_sign_changes")),
            ff_psi_head_sign_changes=_as_int(metrics.get("ff_psi_head_sign_changes")),
            ff_psi_tail_sign_changes=_as_int(metrics.get("ff_psi_tail_sign_changes")),
        )
    except Exception as exc:
        return GeqdskBenchmarkResult(
            spec=spec,
            success=False,
            solver_success=None,
            equivalence_success=None,
            error_type=type(exc).__name__,
            message=str(exc),
            traceback=traceback.format_exc(),
            residual_norm_final=None,
            function_evaluations=None,
            iterations=None,
            state_size=None,
            avg_ms=None,
            std_ms=None,
            shape_rel_rms_error=None,
            shape_rel_max_error=None,
            lcfs_rel_rms_error=None,
            axis_rel_error=None,
            psi_r_rel_rms_error=None,
            psi_r_rel_max_error=None,
            ff_psi_rel_rms_error=None,
            ff_psi_rel_max_error=None,
            psi_r_head_sign_changes=None,
            psi_r_tail_sign_changes=None,
            ff_psi_head_sign_changes=None,
            ff_psi_tail_sign_changes=None,
        )


# ---------------------------------------------------------------------------
# Payload, reports, baseline helper
# ---------------------------------------------------------------------------


def benchmark_result_to_dict(
    result: GeqdskBenchmarkResult, *, include_timing: bool = True
) -> dict[str, Any]:
    row = {
        "case_name": result.spec.case_name,
        "mode": result.spec.mode,
        "coordinate": result.spec.coordinate,
        "constraint": result.spec.constraint,
        "input_kind": result.spec.input_kind,
        "success": result.success,
        "solver_success": result.solver_success,
        "equivalence_success": result.equivalence_success,
        "error_type": result.error_type,
        "message": result.message,
        "residual_norm_final": result.residual_norm_final,
        "function_evaluations": result.function_evaluations,
        "iterations": result.iterations,
        "state_size": result.state_size,
        "shape_rel_rms_error": result.shape_rel_rms_error,
        "shape_rel_max_error": result.shape_rel_max_error,
        "lcfs_rel_rms_error": result.lcfs_rel_rms_error,
        "axis_rel_error": result.axis_rel_error,
        "psi_r_rel_rms_error": result.psi_r_rel_rms_error,
        "psi_r_rel_max_error": result.psi_r_rel_max_error,
        "ff_psi_rel_rms_error": result.ff_psi_rel_rms_error,
        "ff_psi_rel_max_error": result.ff_psi_rel_max_error,
        "psi_r_head_sign_changes": result.psi_r_head_sign_changes,
        "psi_r_tail_sign_changes": result.psi_r_tail_sign_changes,
        "ff_psi_head_sign_changes": result.ff_psi_head_sign_changes,
        "ff_psi_tail_sign_changes": result.ff_psi_tail_sign_changes,
    }
    if include_timing:
        row["avg_ms"] = result.avg_ms
        row["std_ms"] = result.std_ms
    if not result.success:
        row["traceback"] = result.traceback
    return _safe_json(row)


def _best_or_worst_case(
    cases: Sequence[dict[str, Any]],
    key: str,
    *,
    largest: bool = True,
) -> dict[str, Any] | None:
    candidates = [
        case for case in cases if case.get("solver_success") and case.get(key) is not None
    ]
    if not candidates:
        return None
    return (
        max(candidates, key=lambda c: float(c[key]))
        if largest
        else min(candidates, key=lambda c: float(c[key]))
    )


def _canonical_summary(source: RouteSourceBundle) -> dict[str, Any]:
    result = source.canonical_result
    equilibrium = source.canonical_equilibrium
    state = _result_field(result, "x", "state", "solution")
    return _safe_json(
        {
            "grid": {
                "Nr": int(source.reference_grid.Nr),
                "Nt": int(source.reference_grid.Nt),
                "quadrature_scheme": str(source.reference_grid.quadrature_scheme),
            },
            "success": bool(_result_field(result, "success")),
            "message": str(_result_field(result, "message") or ""),
            "residual_norm_final": _as_float(
                _result_field(result, "residual_norm_final", "residual_norm")
            ),
            "function_evaluations": _as_int(
                _result_field(result, "function_evaluations", "nfev", "n_function_evaluations")
            ),
            "state_size": len(state) if state is not None and hasattr(state, "__len__") else None,
            "Ip": _as_float(equilibrium.Ip),
            "beta_constraint": _as_float(equilibrium.beta_t),
            "alpha1": _as_float(equilibrium.alpha1),
            "alpha2": _as_float(equilibrium.alpha2),
        }
    )


def _truth_summary(truth: GeqdskTruthBundle) -> dict[str, Any]:
    geqdsk = truth.geqdsk
    return _safe_json(
        {
            "path": str(truth.geqdsk_path),
            "stem": truth.geqdsk_path.stem,
            "header": str(getattr(geqdsk, "header", "")),
            "NR": int(geqdsk.NR),
            "NZ": int(geqdsk.NZ),
            "R0": _as_float(geqdsk.R0),
            "Z0": _as_float(geqdsk.Z0),
            "Raxis": _as_float(geqdsk.Raxis),
            "Zaxis": _as_float(geqdsk.Zaxis),
            "Bt0": _as_float(geqdsk.Bt0),
            "Ip": _as_float(geqdsk.Ip),
            "psi_axis": _as_float(geqdsk.psi_axis),
            "psi_bound": _as_float(geqdsk.psi_bound),
            "psi_span": _as_float(truth.psi_span),
            "minor_radius_reference": _as_float(truth.minor_radius_reference),
            "boundary_points": int(np.asarray(geqdsk.boundary).shape[0]),
            "limiter_points": int(np.asarray(geqdsk.limiter).shape[0]),
            "rho_geom_axis_count": int(truth.rho_geom_axis.size),
        }
    )


def _payload_summary(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    failure_count = sum(1 for case in cases if not case.get("success"))
    solver_not_success_count = sum(1 for case in cases if case.get("solver_success") is False)
    equivalence_not_success_count = sum(
        1 for case in cases if case.get("equivalence_success") is False
    )
    worst_shape = _best_or_worst_case(cases, "shape_rel_rms_error")
    worst_psi_r = _best_or_worst_case(cases, "psi_r_rel_rms_error")
    worst_ff = _best_or_worst_case(cases, "ff_psi_rel_rms_error")
    return _safe_json(
        {
            "failure_count": failure_count,
            "solver_not_success_count": solver_not_success_count,
            "equivalence_not_success_count": equivalence_not_success_count,
            "worst_shape_case": None if worst_shape is None else worst_shape["case_name"],
            "worst_shape_rel_rms_error": None
            if worst_shape is None
            else worst_shape["shape_rel_rms_error"],
            "worst_psi_r_case": None if worst_psi_r is None else worst_psi_r["case_name"],
            "worst_psi_r_rel_rms_error": None
            if worst_psi_r is None
            else worst_psi_r["psi_r_rel_rms_error"],
            "worst_ff_psi_case": None if worst_ff is None else worst_ff["case_name"],
            "worst_ff_psi_rel_rms_error": None
            if worst_ff is None
            else worst_ff["ff_psi_rel_rms_error"],
        }
    )


def build_payload(
    truth: GeqdskTruthBundle,
    source: RouteSourceBundle,
    results: Sequence[GeqdskBenchmarkResult],
    *,
    include_timing: bool,
    source_sample_count: int,
    solve_nr: int,
    solve_nt: int,
    metric_nr: int,
    metric_nt: int,
    boundary_fit_m: int,
    boundary_fit_n: int,
    boundary_maxtol: float,
    max_evaluations: int = SOLVER_MAX_EVALUATIONS,
) -> dict[str, Any]:
    cases = [benchmark_result_to_dict(result, include_timing=include_timing) for result in results]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "case_count": len(cases),
        "geqdsk": _truth_summary(truth),
        "boundary_fit": {
            "M": int(boundary_fit_m),
            "N": int(boundary_fit_n),
            "maxtol": float(boundary_maxtol),
            "a": _as_float(getattr(truth.boundary, "a", None)),
            "b": _as_float(getattr(truth.boundary, "b", None)),
            "R0": _as_float(getattr(truth.boundary, "R0", None)),
            "Z0": _as_float(getattr(truth.boundary, "Z0", None)),
        },
        "reference_strategy": {
            "truth": "GEQDSK_contours_psi_r_from_area_FF_psi_profile",
            "route_input_generator": "canonical_PF_psin_Ip_solve",
        },
        "canonical": _canonical_summary(source),
        "test": {
            "grid": {"Nr": int(solve_nr), "Nt": int(solve_nt), "quadrature_scheme": "legendre"},
            "metric_grid": {
                "Nr": int(metric_nr),
                "Nt": int(metric_nt),
                "quadrature_scheme": "uniform",
            },
            "source_sample_count": int(source_sample_count),
            "route_source_sample_count_overrides": {
                "PJ2_psin_Ip": _source_sample_count_for_case(
                    GeqdskBenchmarkSpec("PJ2", "psin", "Ip"),
                    source_sample_count,
                ),
                "PJ2_psin_Ip_beta": _source_sample_count_for_case(
                    GeqdskBenchmarkSpec("PJ2", "psin", "Ip_beta"),
                    source_sample_count,
                ),
            },
            "surface_levels": [float(x) for x in truth.surface_levels],
            "profile_rho_eval_min": float(min(PROFILE_RHO_EVAL)),
            "profile_rho_eval_max": float(max(PROFILE_RHO_EVAL)),
            "profile_psin_eval_min": float(min(PROFILE_PSIN_EVAL)),
            "profile_psin_eval_max": float(max(PROFILE_PSIN_EVAL)),
            "mode_constraints": {
                key: list(values) for key, values in BENCHMARK_MODE_CONSTRAINTS.items()
            },
            "solver": {
                "method": SOLVER_METHOD,
                "max_residual": SOLVER_MAX_RESIDUAL,
                "max_evaluations": int(max_evaluations),
                "canonical_initial_policy": SOLVER_INITIAL_POLICY,
                "route_initial_policy": ROUTE_SOLVER_INITIAL_POLICY,
            },
            "equivalence_thresholds": {
                "shape_rel_rms_error": EQUIVALENCE_SHAPE_REL_RMS_MAX,
                "psi_r_rel_rms_error": EQUIVALENCE_PSI_R_REL_RMS_MAX,
                "ff_psi_rel_rms_error": EQUIVALENCE_FF_PSI_REL_RMS_MAX,
            },
        },
        "summary": _payload_summary(cases),
        "cases": cases,
    }
    return _safe_json(payload)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_safe_json(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _format_float(value: Any, width: int = 10) -> str:
    if value is None:
        return "-".rjust(width)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value).rjust(width)
    if not math.isfinite(v):
        return "-".rjust(width)
    return f"{v:{width}.3e}"


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    cases = list(payload["cases"])
    summary = payload["summary"]
    geqdsk = payload["geqdsk"]
    canonical = payload["canonical"]

    solver_success_cases = [case for case in cases if case.get("solver_success")]
    failures = [case for case in cases if not case.get("success")]

    lines: list[str] = []
    lines.append(f"GEQDSK benchmark: {geqdsk['path']}")
    lines.append(f"case_count                : {payload['case_count']}")
    lines.append("truth                     : GEQDSK contour / psi_r(area) / FF_psi")
    lines.append("route_input_generator     : canonical PF(psin)+Ip")
    lines.append(f"canonical_success         : {canonical.get('success')}")
    canonical_residual = _format_float(canonical.get("residual_norm_final"), 12).strip()
    lines.append(f"canonical_residual        : {canonical_residual}")
    lines.append(
        f"canonical_beta            : {_format_float(canonical.get('beta_constraint'), 12).strip()}"
    )
    lines.append(f"failure_count             : {summary.get('failure_count')}")
    lines.append(f"solver_not_success_count  : {summary.get('solver_not_success_count')}")
    lines.append(f"equivalence_fail_count    : {summary.get('equivalence_not_success_count')}")
    lines.append(f"worst_shape_case          : {summary.get('worst_shape_case')}")
    lines.append(f"worst_psi_r_case          : {summary.get('worst_psi_r_case')}")
    lines.append(f"worst_FF_psi_case         : {summary.get('worst_ff_psi_case')}")
    lines.append("")

    lines.append("Case results")
    lines.append(
        "case".ljust(30)
        + " | shape_rms | psi_r_rms | FF_psi_rms | residual   | evals | solver | equiv | ok"
    )
    lines.append("-" * 115)
    for case in cases:
        lines.append(
            str(case["case_name"]).ljust(30)
            + " | "
            + _format_float(case.get("shape_rel_rms_error"), 9)
            + " | "
            + _format_float(case.get("psi_r_rel_rms_error"), 9)
            + " | "
            + _format_float(case.get("ff_psi_rel_rms_error"), 10)
            + " | "
            + _format_float(case.get("residual_norm_final"), 10)
            + " | "
            + str(case.get("function_evaluations") or "-").rjust(5)
            + " | "
            + str(case.get("solver_success")).rjust(6)
            + " | "
            + str(case.get("equivalence_success")).rjust(5)
            + " | "
            + str(case.get("success"))
        )
    lines.append("")

    for title, key in (
        ("Largest shape RMS error ranking", "shape_rel_rms_error"),
        ("Largest psi_r RMS error ranking", "psi_r_rel_rms_error"),
        ("Largest FF_psi RMS error ranking", "ff_psi_rel_rms_error"),
    ):
        lines.append(title)
        ranked = sorted(
            [case for case in solver_success_cases if case.get(key) is not None],
            key=lambda item: float(item[key]),
            reverse=True,
        )[:10]
        for idx, case in enumerate(ranked, start=1):
            lines.append(f"  {idx:2d}. {case['case_name']:<30} {key}={case[key]:.6e}")
        if not ranked:
            lines.append("  -")
        lines.append("")

    lines.append("Route failures")
    if failures:
        for case in failures:
            lines.append(f"  {case['case_name']}: {case.get('error_type')}: {case.get('message')}")
    else:
        lines.append("  none")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _case_progress_status(result: GeqdskBenchmarkResult) -> str:
    if result.success:
        return "OK"
    if result.solver_success is False:
        return "SOLVER_FAIL"
    if result.equivalence_success is False:
        return "METRIC_FAIL"
    return "EXCEPTION"


def _short_progress_message(result: GeqdskBenchmarkResult, *, limit: int = 90) -> str:
    if result.error_type and result.solver_success is not False:
        message = f"{result.error_type}: {result.message or ''}"
    else:
        message = result.message or ""
    message = " ".join(message.split())
    if not message:
        return "-"
    if len(message) <= limit:
        return message
    return f"{message[: limit - 3]}..."


def _format_case_progress_line(
    index: int,
    total: int,
    result: GeqdskBenchmarkResult,
) -> str:
    evals = "-" if result.function_evaluations is None else str(result.function_evaluations)
    return (
        f"[{index:02d}/{total:02d}] {result.spec.case_name:<30} "
        f"{_case_progress_status(result):<11} "
        f"solver={result.solver_success!s:<5} "
        f"equiv={result.equivalence_success!s:<5} "
        f"evals={evals:>5} "
        f"resid={_format_float(result.residual_norm_final).strip():>10} "
        f"shape={_format_float(result.shape_rel_rms_error).strip():>10} "
        f"psi_r={_format_float(result.psi_r_rel_rms_error).strip():>10} "
        f"FF={_format_float(result.ff_psi_rel_rms_error).strip():>10} "
        f"msg={_short_progress_message(result)}"
    )


def _strip_timing(payload: Mapping[str, Any]) -> dict[str, Any]:
    stripped = json.loads(json.dumps(_safe_json(payload)))
    for case in stripped.get("cases", []):
        case.pop("avg_ms", None)
        case.pop("std_ms", None)
    return stripped


def run_geqdsk_benchmark(
    *,
    geqdsk_path: Path | str = DEFAULT_GEQDSK_PATH,
    output_dir: Path | str | None = None,
    include_timing: bool = True,
    timing_repeats: int = 0,
    enable_plots: bool = PLOT,
    write_artifacts: bool = True,
    write_baseline: bool = False,
    source_sample_count: int = SOURCE_SAMPLE_COUNT,
    reference_nr: int = REFERENCE_NR,
    reference_nt: int = REFERENCE_NT,
    solve_nr: int = SOLVE_NR,
    solve_nt: int = SOLVE_NT,
    metric_nr: int = METRIC_NR,
    metric_nt: int = METRIC_NT,
    boundary_fit_m: int = BOUNDARY_FIT_M,
    boundary_fit_n: int = BOUNDARY_FIT_N,
    boundary_maxtol: float = BOUNDARY_MAXTOL,
    max_evaluations: int = SOLVER_MAX_EVALUATIONS,
    show_progress: bool = True,
) -> dict[str, Any]:
    geqdsk_path = _resolve_path(geqdsk_path)
    case_key = _sanitize_case_key(geqdsk_path)
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_ROOT / case_key
    output_dir = Path(output_dir).expanduser().resolve()
    plots_dir = output_dir / "plots" if enable_plots else None
    if show_progress and plots_dir is not None:
        print(f"[plots] writing comparison plots to {plots_dir}", flush=True)

    if show_progress:
        print(f"[geqdsk] reading truth case: {geqdsk_path}", flush=True)
    truth = build_geqdsk_truth_bundle(
        geqdsk_path,
        boundary_fit_m=boundary_fit_m,
        boundary_fit_n=boundary_fit_n,
        boundary_maxtol=boundary_maxtol,
    )

    if show_progress:
        print("[canonical] solving PF(psin)+Ip reference", flush=True)
    source = build_route_source_bundle(
        truth,
        reference_nr=reference_nr,
        reference_nt=reference_nt,
        max_evaluations=max_evaluations,
    )

    specs = list(iter_benchmark_specs())
    results: list[GeqdskBenchmarkResult] = []
    for index, spec in enumerate(specs, start=1):
        result = run_benchmark_case(
            truth,
            source,
            spec,
            solve_nr=solve_nr,
            solve_nt=solve_nt,
            metric_nr=metric_nr,
            metric_nt=metric_nt,
            source_sample_count=source_sample_count,
            max_evaluations=max_evaluations,
            timing_repeats=timing_repeats,
            plots_dir=plots_dir,
        )
        results.append(result)
        if show_progress:
            print(_format_case_progress_line(index, len(specs), result), flush=True)

    payload = build_payload(
        truth,
        source,
        results,
        include_timing=include_timing,
        source_sample_count=source_sample_count,
        solve_nr=solve_nr,
        solve_nt=solve_nt,
        metric_nr=metric_nr,
        metric_nt=metric_nt,
        boundary_fit_m=boundary_fit_m,
        boundary_fit_n=boundary_fit_n,
        boundary_maxtol=boundary_maxtol,
        max_evaluations=max_evaluations,
    )

    if write_artifacts:
        write_json(output_dir / "benchmark_with_geqdsk_summary.json", payload)
        write_report(output_dir / "benchmark_with_geqdsk_compare.txt", payload)
        write_json(
            output_dir / "truth_summary.json",
            {
                "schema_version": SCHEMA_VERSION,
                "geqdsk": payload["geqdsk"],
                "boundary_fit": payload["boundary_fit"],
                "truth_rho_geom_axis": truth.rho_geom_axis,
                "truth_psin_geom_axis": truth.psin_geom_axis,
                "surface_levels": truth.surface_levels,
            },
        )
        write_json(
            output_dir / "canonical_equilibrium.json",
            {
                "schema_version": SCHEMA_VERSION,
                "canonical": payload["canonical"],
                "rho": source.rho_axis,
                "psin": source.psin_axis,
                "psin_r": source.profiles["psin_r"],
                "psi_r": source.profiles["psi_r"],
                "FF_psi": source.profiles["FF_psi"],
                "P_psi": source.profiles["P_psi"],
                "q": source.profiles["q"],
            },
        )

    if write_baseline:
        baseline_path = DEFAULT_BASELINE_DIR / f"benchmark_with_geqdsk_{case_key}_non_timing.json"
        write_json(baseline_path, _strip_timing(payload))
        if show_progress:
            print(f"[baseline] wrote {baseline_path}", flush=True)

    return payload


def build_geqdsk_benchmark_baseline_payload(
    *,
    geqdsk_path: Path | str = DEFAULT_GEQDSK_PATH,
    source_sample_count: int = SOURCE_SAMPLE_COUNT,
    reference_nr: int = REFERENCE_NR,
    reference_nt: int = REFERENCE_NT,
    solve_nr: int = SOLVE_NR,
    solve_nt: int = SOLVE_NT,
    metric_nr: int = METRIC_NR,
    metric_nt: int = METRIC_NT,
    boundary_fit_m: int = BOUNDARY_FIT_M,
    boundary_fit_n: int = BOUNDARY_FIT_N,
    boundary_maxtol: float = BOUNDARY_MAXTOL,
    max_evaluations: int = SOLVER_MAX_EVALUATIONS,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Return the deterministic non-timing payload intended for baseline tests."""

    payload = run_geqdsk_benchmark(
        geqdsk_path=geqdsk_path,
        output_dir=None,
        include_timing=False,
        timing_repeats=0,
        enable_plots=False,
        write_artifacts=False,
        write_baseline=False,
        source_sample_count=source_sample_count,
        reference_nr=reference_nr,
        reference_nt=reference_nt,
        solve_nr=solve_nr,
        solve_nt=solve_nt,
        metric_nr=metric_nr,
        metric_nt=metric_nt,
        boundary_fit_m=boundary_fit_m,
        boundary_fit_n=boundary_fit_n,
        boundary_maxtol=boundary_maxtol,
        max_evaluations=max_evaluations,
        show_progress=show_progress,
    )
    return _strip_timing(payload)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GEQDSK-driven VEQPy route/constraint benchmark.",
    )
    parser.add_argument(
        "--geqdsk",
        type=Path,
        default=DEFAULT_GEQDSK_PATH,
        help=f"GEQDSK file to use. Default: {DEFAULT_GEQDSK_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: tests/benchmark/geqdsk/<GEQDSK stem>",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Also write tests/baselines/benchmark_with_geqdsk_<case>_non_timing.json.",
    )
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plots",
        dest="plots",
        action="store_true",
        help="Write per-case comparison plots.",
    )
    plot_group.add_argument(
        "--no-plots",
        dest="plots",
        action="store_false",
        help="Skip per-case comparison plots.",
    )
    parser.set_defaults(plots=PLOT)
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=0,
        help="Extra independent solves per case for timing statistics. Default: 0.",
    )
    parser.add_argument("--source-sample-count", type=int, default=SOURCE_SAMPLE_COUNT)
    parser.add_argument("--reference-nr", type=int, default=REFERENCE_NR)
    parser.add_argument("--reference-nt", type=int, default=REFERENCE_NT)
    parser.add_argument("--solve-nr", type=int, default=SOLVE_NR)
    parser.add_argument("--solve-nt", type=int, default=SOLVE_NT)
    parser.add_argument("--metric-nr", type=int, default=METRIC_NR)
    parser.add_argument("--metric-nt", type=int, default=METRIC_NT)
    parser.add_argument("--boundary-fit-m", type=int, default=BOUNDARY_FIT_M)
    parser.add_argument("--boundary-fit-n", type=int, default=BOUNDARY_FIT_N)
    parser.add_argument("--boundary-maxtol", type=float, default=BOUNDARY_MAXTOL)
    parser.add_argument("--max-evaluations", type=int, default=SOLVER_MAX_EVALUATIONS)
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.perf_counter()
    payload = run_geqdsk_benchmark(
        geqdsk_path=args.geqdsk,
        output_dir=args.output_dir,
        include_timing=True,
        timing_repeats=args.timing_repeats,
        enable_plots=args.plots,
        write_artifacts=True,
        write_baseline=args.write_baseline,
        source_sample_count=args.source_sample_count,
        reference_nr=args.reference_nr,
        reference_nt=args.reference_nt,
        solve_nr=args.solve_nr,
        solve_nt=args.solve_nt,
        metric_nr=args.metric_nr,
        metric_nt=args.metric_nt,
        boundary_fit_m=args.boundary_fit_m,
        boundary_fit_n=args.boundary_fit_n,
        boundary_maxtol=args.boundary_maxtol,
        max_evaluations=args.max_evaluations,
        show_progress=not args.quiet,
    )
    elapsed = time.perf_counter() - started
    if not args.quiet:
        failure_count = payload["summary"].get("failure_count")
        print(
            f"[done] cases={payload['case_count']} failures={failure_count} elapsed_s={elapsed:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
