from __future__ import annotations

import importlib.util
import json
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/veqpy-mpl")))

import numpy as np

from veqlib.facade import KernelResult, pinned_cpu

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
VEQLIB_ROOT = REPO_ROOT / "veqlib"
CORE_DIR = VEQLIB_ROOT / "core"
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MU0 = 4.0e-7 * np.pi

CASE_KEYS = ("solovev", "chease", "efit")
CASE_LABELS = {
    "solovev": "D-shape",
    "chease": "H-mode",
    "efit": "X-point",
}
CONFIG_LABELS = ("Low", "Medium", "High", "Ref")

CASE_REFERENCE_GFILES = {
    "solovev": str(REPO_ROOT / "data" / "SOLOVEV.geqdsk"),
    "chease": str(REPO_ROOT / "data" / "CHEASE.geqdsk"),
    "efit": str(REPO_ROOT / "data" / "EFIT.geqdsk"),
}
CASE_REFERENCE_EQUILIBRIUM_JSONS = {
    "solovev": str(REPO_ROOT / "data" / "solovev-equilibrium.json"),
    "chease": str(REPO_ROOT / "data" / "chease-equilibrium.json"),
    "efit": str(REPO_ROOT / "data" / "efit-equilibrium.json"),
}
REFERENCE_EQUILIBRIUM_MANIFEST_PATH = str(REPO_ROOT / "data" / "reference_equilibria.json")
REDUCED_EQUILIBRIUM_JSON_TEMPLATE = str(
    REPO_ROOT / "data" / "pareto_reduced_{case_key}_{config_label}.json"
)
REDUCED_EQUILIBRIUM_MANIFEST_PATH = str(REPO_ROOT / "data" / "pareto_reduced_equilibria.json")

CASE_REFERENCE_PROFILE_LENGTHS = {
    "solovev": {
        "psin": 10,
        "h": 10,
        "k": 10,
        "s1": 10,
        "s2": 5,
        "s3": 5,
        "s4": 5,
        "s5": 5,
        "s6": 5,
        "s7": 5,
        "s8": 5,
    },
    "chease": {
        "psin": 10,
        "h": 10,
        "k": 10,
        "v": 10,
        "c0": 10,
        "c1": 5,
        "c2": 5,
        "c3": 5,
        "c4": 5,
        "c5": 5,
        "c6": 5,
        "c7": 5,
        "s1": 10,
        "s2": 5,
        "s3": 5,
        "s4": 5,
        "s5": 5,
        "s6": 5,
        "s7": 5,
        "s8": 5,
    },
    "efit": {
        "psin": 10,
        "h": 10,
        "k": 10,
        "v": 10,
        "c0": 10,
        "c1": 5,
        "c2": 5,
        "c3": 5,
        "c4": 5,
        "c5": 5,
        "c6": 5,
        "c7": 5,
        "s1": 10,
        "s2": 5,
        "s3": 5,
        "s4": 5,
        "s5": 5,
        "s6": 5,
        "s7": 5,
        "s8": 5,
    },
}

REFERENCE_LAYOUT_NR = 32
REFERENCE_LAYOUT_NT = 32
REFERENCE_SOLVER_MAXFEV = 2000
SOLVER_INITIAL_POLICY = "auto"
BOUNDARY_MAXTOL = 1.0
CASE_BOUNDARY_FIT_M = {
    "solovev": 10,
    "chease": 10,
    "efit": 10,
}
CASE_BOUNDARY_FIT_N = {
    "solovev": 10,
    "chease": 10,
    "efit": 10,
}


@dataclass(frozen=True, slots=True)
class PreparedInterpAxis:
    unique_axis: np.ndarray
    order: np.ndarray
    unique_index: np.ndarray


@dataclass(frozen=True, slots=True)
class PfReferenceCase:
    case_key: str
    boundary: object
    geqdsk: object
    equilibrium: object
    ref_profiles: dict[str, np.ndarray | float]
    psin_interp_axis: PreparedInterpAxis


@lru_cache(maxsize=1)
def load_veqpy_components() -> dict[str, object]:
    from veqpy.model import Boundary, Equilibrium, Geqdsk, Grid, Problem
    from veqpy.model.boundary import _fit_boundary_params
    from veqpy.operator import Operator
    from veqpy.solver import Solver, SolverConfig

    return {
        "Boundary": Boundary,
        "Equilibrium": Equilibrium,
        "Geqdsk": Geqdsk,
        "Grid": Grid,
        "Problem": Problem,
        "fit_boundary_params": _fit_boundary_params,
        "Operator": Operator,
        "Solver": Solver,
        "SolverConfig": SolverConfig,
    }


def temp_cache(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cpu_affinity() -> list[int] | None:
    if hasattr(os, "sched_getaffinity"):
        return sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    return None


def runtime_env() -> dict[str, str | None]:
    keys = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    return {key: os.environ.get(key) for key in keys}


def quantile(values: list[float], q: float) -> float:
    values_sorted = sorted(values)
    return float(values_sorted[int((len(values_sorted) - 1) * q)])


def float_stats(values: list[float], *, samples: bool = True, prefix: str = "") -> dict[str, Any]:
    if not values:
        return {
            f"{prefix}median_ms": float("nan"),
            f"{prefix}mean_ms": float("nan"),
            f"{prefix}min_ms": float("nan"),
            f"{prefix}max_ms": float("nan"),
            f"{prefix}p05_ms": float("nan"),
            f"{prefix}p95_ms": float("nan"),
            "count": 0,
            **({"samples_ms": []} if samples else {}),
        }
    payload: dict[str, Any] = {
        f"{prefix}median_ms": float(statistics.median(values)),
        f"{prefix}mean_ms": float(statistics.mean(values)),
        f"{prefix}min_ms": float(min(values)),
        f"{prefix}max_ms": float(max(values)),
        f"{prefix}p05_ms": quantile(values, 0.05),
        f"{prefix}p95_ms": quantile(values, 0.95),
        "count": len(values),
    }
    if samples:
        payload["samples_ms"] = [float(value) for value in values]
    return payload


def int_stats(values: list[int], *, samples: bool = True) -> dict[str, Any]:
    if not values:
        return {
            "median": 0,
            "mean": 0.0,
            "min": 0,
            "max": 0,
            **({"samples": []} if samples else {}),
        }
    payload: dict[str, Any] = {
        "median": int(statistics.median(values)),
        "mean": float(statistics.mean(values)),
        "min": int(min(values)),
        "max": int(max(values)),
    }
    if samples:
        payload["samples"] = [int(value) for value in values]
    return payload


def max_abs(lhs: Any, rhs: Any) -> float:
    lhs_arr = np.asarray(lhs, dtype=np.float64)
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    if lhs_arr.shape != rhs_arr.shape:
        return float("inf")
    if lhs_arr.size == 0:
        return 0.0
    return float(np.max(np.abs(lhs_arr - rhs_arr)))


def profile_count(profile_coeffs: dict[str, Any], name: str) -> int:
    values = profile_coeffs.get(name)
    return 0 if values is None else int(np.asarray(values, dtype=np.float64).size)


def family_counts(profile_coeffs: dict[str, Any], prefix: str, first: int) -> tuple[int, ...]:
    orders = [
        int(name[1:])
        for name, values in profile_coeffs.items()
        if values is not None
        and len(name) > 1
        and name[0] == prefix
        and name[1:].isdigit()
        and profile_count(profile_coeffs, name) > 0
    ]
    if not orders:
        return ()
    counts = [
        profile_count(profile_coeffs, f"{prefix}{order}")
        for order in range(first, max(orders) + 1)
    ]
    while counts and counts[-1] == 0:
        counts.pop()
    return tuple(counts)


def finite_or_nan(value: Any) -> float:
    parsed = float(value)
    return parsed if np.isfinite(parsed) else float("nan")


def as_float64_array(values: Any, *, copy: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr.copy() if copy else arr


def prepare_interp_axis(axis: np.ndarray) -> PreparedInterpAxis:
    axis_f64 = as_float64_array(axis)
    order = np.argsort(axis_f64)
    axis_sorted = axis_f64[order]
    unique_axis, unique_index = np.unique(axis_sorted, return_index=True)
    return PreparedInterpAxis(unique_axis=unique_axis, order=order, unique_index=unique_index)


def prepare_interp_values(values: np.ndarray, prepared_axis: PreparedInterpAxis) -> np.ndarray:
    values_f64 = as_float64_array(values)
    return values_f64[prepared_axis.order][prepared_axis.unique_index]


def profile_interp(
    axis: np.ndarray | PreparedInterpAxis, values: np.ndarray, x_new: np.ndarray
) -> np.ndarray:
    from scipy.interpolate import PchipInterpolator

    prepared_axis = axis if isinstance(axis, PreparedInterpAxis) else prepare_interp_axis(axis)
    unique_axis = prepared_axis.unique_axis
    unique_values = prepare_interp_values(values, prepared_axis)
    x_new = as_float64_array(x_new)
    if unique_axis.size < 2:
        fill_value = float(unique_values[0] if unique_values.size else 0.0)
        return np.full_like(x_new, fill_value, dtype=np.float64)
    if unique_axis.size < 3:
        return np.interp(x_new, unique_axis, unique_values).astype(np.float64, copy=False)
    return as_float64_array(PchipInterpolator(unique_axis, unique_values, extrapolate=True)(x_new))


def active_profiles_from_coeffs(profile_coeffs: Mapping[str, object]) -> dict[str, int]:
    active_profiles: dict[str, int] = {}
    for name, coeff in profile_coeffs.items():
        if coeff is None:
            continue
        if isinstance(coeff, (int, np.integer)):
            length = int(coeff)
        else:
            length = int(np.asarray(coeff, dtype=np.float64).size)
        if length > 0:
            active_profiles[str(name)] = length
    return active_profiles


def read_geqdsk(path: str):
    geqdsk = load_veqpy_components()["Geqdsk"]()
    geqdsk.read_geqdsk(str(path))
    return geqdsk


def load_equilibrium_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing equilibrium JSON: {path}")
    return load_veqpy_components()["Equilibrium"].load(path)


def build_pf_reference_profiles(equilibrium: Any) -> dict[str, np.ndarray | float]:
    psin_r = as_float64_array(equilibrium.psin_r, copy=True)
    psin_r_safe = np.where(np.abs(psin_r) > 1.0e-14, psin_r, 1.0e-14)
    pn_psin = as_float64_array(equilibrium.Pn_r, copy=True) / psin_r_safe
    return {
        "psin": as_float64_array(equilibrium.psin, copy=True),
        "FFn_psin": as_float64_array(equilibrium.FFn_r, copy=True) / psin_r_safe,
        "Pn_psin": pn_psin,
        "setup_Pn_psin": pn_psin / MU0,
    }


def build_geqdsk_boundary(geqdsk: Any, *, fit_m: int, fit_n: int, return_fit: bool = False):
    components = load_veqpy_components()
    fit = components["fit_boundary_params"](
        geqdsk,
        M=int(fit_m),
        N=int(fit_n),
        maxtol=BOUNDARY_MAXTOL,
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
    return (boundary, fit) if return_fit else boundary


def load_pf_benchmark(backend: str):
    os.environ["VEQPY_BACKEND"] = str(backend)
    components = load_veqpy_components()
    reference_grid = components["Grid"](
        Nr=REFERENCE_LAYOUT_NR,
        Nt=REFERENCE_LAYOUT_NT,
        quadrature_scheme="legendre",
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
        CONFIG=config,
        REFERENCE_GRID=reference_grid,
    )


def build_pf_reference_case(case_key: str) -> PfReferenceCase:
    equilibrium = load_equilibrium_json(CASE_REFERENCE_EQUILIBRIUM_JSONS[case_key])
    geqdsk = read_geqdsk(CASE_REFERENCE_GFILES[case_key])
    boundary = build_geqdsk_boundary(
        geqdsk,
        fit_m=CASE_BOUNDARY_FIT_M[case_key],
        fit_n=CASE_BOUNDARY_FIT_N[case_key],
    )
    return PfReferenceCase(
        case_key=case_key,
        boundary=boundary,
        geqdsk=geqdsk,
        equilibrium=equilibrium,
        ref_profiles=build_pf_reference_profiles(equilibrium),
        psin_interp_axis=prepare_interp_axis(np.asarray(equilibrium.psin, dtype=np.float64)),
    )


def make_profile_coeffs(
    signature: dict[str, int],
    *,
    max_lengths: dict[str, int],
) -> dict[str, list[float] | None]:
    profile_coeffs: dict[str, list[float] | None] = {name: None for name in max_lengths}
    for name, length in signature.items():
        coeff_length = int(length)
        if coeff_length > 0:
            profile_coeffs[name] = [0.0] * coeff_length
    return profile_coeffs


def build_pf_case(benchmark: Any, reference: PfReferenceCase, signature: dict[str, int]):
    return benchmark.Problem(
        route="PF",
        coordinate="psin",
        nodes="uniform",
        active_profiles=active_profiles_from_coeffs(
            make_profile_coeffs(
                signature,
                max_lengths=CASE_REFERENCE_PROFILE_LENGTHS[reference.case_key],
            )
        ),
        boundary=reference.boundary,
        heat_input=np.asarray(reference.geqdsk.P_psi, dtype=np.float64),
        current_input=np.asarray(reference.geqdsk.FF_psi, dtype=np.float64),
        Ip=float(reference.geqdsk.Ip),
        beta=None,
    )


def reduced_equilibrium_json_path(case_key: str, config_label: str) -> str:
    return REDUCED_EQUILIBRIUM_JSON_TEMPLATE.format(
        case_key=str(case_key),
        config_label=str(config_label).lower(),
    )


def load_reduced_equilibrium_manifest(
    path: str | None = None,
) -> dict[tuple[str, str], dict[str, object]]:
    path = REDUCED_EQUILIBRIUM_MANIFEST_PATH if path is None else str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing reduced-equilibrium manifest: {path}. "
            "Run `python scripts/07-pareto-analysis.py` first."
        )
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    manifest: dict[tuple[str, str], dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        case_key = str(entry.get("case_key", ""))
        config_label = str(entry.get("config_label", ""))
        if case_key and config_label:
            manifest[(case_key, config_label)] = entry
    return manifest


def load_reference_equilibrium_manifest(
    path: str | None = None,
) -> dict[tuple[str, str], dict[str, object]]:
    path = REFERENCE_EQUILIBRIUM_MANIFEST_PATH if path is None else str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing reference-equilibrium manifest: {path}. "
            "Run `python scripts/06-high-order-reconstructions.py` first."
        )
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    manifest: dict[tuple[str, str], dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        case_key = str(entry.get("case_key", ""))
        config_label = str(entry.get("config_label", ""))
        if case_key and config_label:
            manifest[(case_key, config_label)] = entry
    return manifest


def manifest_entry(
    manifest: dict[tuple[str, str], dict[str, object]],
    case_key: str,
    config_label: str,
) -> dict[str, object]:
    entry = manifest.get((case_key, config_label))
    if entry is None:
        raise FileNotFoundError(
            f"Missing {CASE_LABELS[case_key]} {config_label} "
            f"entry in {REDUCED_EQUILIBRIUM_MANIFEST_PATH}. "
            "Run `python scripts/07-pareto-analysis.py` first."
        )
    return entry


def reference_manifest_entry(
    manifest: dict[tuple[str, str], dict[str, object]], case_key: str
) -> dict[str, object]:
    entry = manifest.get((case_key, "Ref"))
    if entry is None:
        raise FileNotFoundError(
            f"Missing {CASE_LABELS[case_key]} Ref entry in "
            f"{REFERENCE_EQUILIBRIUM_MANIFEST_PATH}. "
            "Run `python scripts/06-high-order-reconstructions.py` first."
        )
    return entry


def normalize_signature(signature: dict[str, int]) -> dict[str, int]:
    return {str(name): int(length) for name, length in sorted(signature.items()) if int(length) > 0}


def signature_from_metadata(entry: dict[str, object]) -> dict[str, int]:
    signature = entry.get("signature", {})
    return normalize_signature(signature) if isinstance(signature, dict) else {}


@dataclass(frozen=True, slots=True)
class NativeTiming:
    result: KernelResult
    wall_ms: list[float]
    inner_ms: list[float]
    success: list[bool]
    nfev: list[int]
    njev: list[int]
    jacobian_component_evaluations: list[int]

    def compact(self) -> dict[str, Any]:
        return {
            "success_all": all(self.success),
            "info": int(self.result.info),
            "timing": float_stats(self.wall_ms),
            "inner_timing": float_stats(self.inner_ms),
            "nfev": int_stats(self.nfev),
            "njev": int_stats(self.njev),
            "jacobian_component_evaluations": int_stats(self.jacobian_component_evaluations),
            "raw_norm": float(self.result.raw_norm),
            "scaled_norm": float(self.result.scaled_norm),
            "x": self.result.x.tolist(),
            "raw": self.result.raw.tolist(),
            "alpha": self.result.alpha.tolist(),
        }


def measure_native_solver(
    solver: Any,
    configure_runtime: Callable[[], None],
    *,
    warmup: int,
    repeat: int,
) -> NativeTiming:
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    with pinned_cpu():
        configure_runtime()
        for _ in range(warmup):
            solver.solve_direct()

        wall_ms: list[float] = []
        inner_ms: list[float] = []
        success: list[bool] = []
        nfev: list[int] = []
        njev: list[int] = []
        jaccomp: list[int] = []
        final_result: Any | None = None
        for _ in range(repeat):
            start_ns = time.perf_counter_ns()
            final_result = solver.solve_direct()
            wall_ms.append(float(time.perf_counter_ns() - start_ns) / 1.0e6)
            inner_ms.append(float(final_result[0]))
            success.append(bool(final_result[1]))
            nfev.append(int(final_result[3]))
            njev.append(int(final_result[4]))
            jaccomp.append(int(final_result[6]))

    if final_result is None:
        raise RuntimeError("native timing loop did not run")
    return NativeTiming(
        result=KernelResult.from_solve_direct(final_result),
        wall_ms=wall_ms,
        inner_ms=inner_ms,
        success=success,
        nfev=nfev,
        njev=njev,
        jacobian_component_evaluations=jaccomp,
    )
