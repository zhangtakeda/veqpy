"""Shared GEQDSK benchmark fixtures for the VEQPy 2.x public Module."""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Sequence

import numpy as np
from fusionprime_base import Current, Equilibrium, Flux, Geometry, Kinetic, Plasma, Source
from fusionprime_base.io import Geqdsk, load_geqdsk

from veqpy import build

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = REPO_ROOT / "veqpy" / "kernels" / "cxx_kernel" / "core"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
DATA_DIR = REPO_ROOT / "data"

CASE_KEYS = ("solovev", "chease", "efit")
CONFIG_LABELS = ("Low", "Medium", "High", "Ref")
CASE_REFERENCE_GFILES = {
    "solovev": DATA_DIR / "SOLOVEV.geqdsk",
    "chease": DATA_DIR / "CHEASE.geqdsk",
    "efit": DATA_DIR / "EFIT.geqdsk",
}
REFERENCE_LAYOUT_NR = 32
REFERENCE_LAYOUT_NT = 32
REFERENCE_SOLVER_MAXFEV = 2000
BACKENDS = ("numba", "cxx-strict", "cxx-relaxed", "cxx-enzyme")
RUNNABLE_BACKENDS = ("numba", "cxx-strict", "cxx-relaxed")
ENZYME_SKIP_REASON = "cxx-enzyme is intentionally deferred and is not implemented in this task"
HISTORICAL_SPEEDUP_RANGE = (5.0, 11.0)

# These are the historical manuscript signatures restored from main.  The
# values are coefficient counts, not user-facing ABI field names.
GEQDSK_CONFIG_SIGNATURES: dict[tuple[str, str], dict[str, int]] = {
    ("solovev", "Low"): {"psin": 1, "h": 1, "k": 1, "s1": 1},
    ("solovev", "Medium"): {"psin": 1, "h": 1, "k": 2, "s1": 1},
    ("solovev", "High"): {"psin": 3, "h": 2, "k": 2, "s1": 2},
    ("solovev", "Ref"): {
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
    ("chease", "Low"): {
        "psin": 4,
        "h": 6,
        "k": 2,
        "v": 3,
        "c0": 3,
        "s1": 3,
        "c1": 2,
        "s2": 2,
        "c2": 1,
        "s3": 1,
    },
    ("chease", "Medium"): {
        "psin": 5,
        "h": 10,
        "k": 4,
        "v": 3,
        "c0": 3,
        "s1": 3,
        "c1": 3,
        "s2": 3,
        "c2": 1,
        "s3": 1,
    },
    ("chease", "High"): {
        "psin": 8,
        "h": 9,
        "k": 7,
        "v": 8,
        "c0": 6,
        "s1": 6,
        "c1": 4,
        "s2": 4,
        "c2": 4,
        "s3": 4,
    },
    ("chease", "Ref"): {
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
    ("efit", "Low"): {
        "psin": 3,
        "h": 5,
        "k": 3,
        "v": 2,
        "c0": 2,
        "s1": 2,
        "c1": 1,
        "s2": 1,
    },
    ("efit", "Medium"): {
        "psin": 3,
        "h": 4,
        "k": 4,
        "v": 5,
        "c0": 2,
        "s1": 2,
        "c1": 2,
        "s2": 2,
        "c2": 2,
        "s3": 1,
        "c3": 1,
        "s4": 1,
    },
    ("efit", "High"): {
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
        "s4": 4,
        "c4": 5,
        "s5": 5,
        "c5": 2,
        "s6": 2,
        "c6": 1,
        "s7": 1,
    },
    ("efit", "Ref"): {
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

# Frozen fits restored from the historical benchmark.  GEQDSK contains a
# scatter LCFS, while the base Geometry contract consumes the fitted phase
# harmonics. Geometry and the VEQ Kernel both use ``+ s_m sin(m theta)``, so
# the stored sine coefficients pass through without a sign conversion.
BOUNDARY_FITS: dict[str, dict[str, Any]] = {
    "solovev": {
        "a": 1.999991815361528,
        "R0": 6.199980064550139,
        "Z0": -4.0265979758802695e-05,
        "kappa": 1.6999963725270892,
        "c": (
            -5.2020273089148361e-06,
            7.1653084700073520e-05,
            9.0046463771700606e-06,
            -9.7789815345071059e-06,
            -2.4974793964430963e-05,
            6.9834184194999052e-05,
            7.2729999976377730e-06,
            -1.0542591490004266e-05,
            -2.4922126347913156e-05,
            6.9894200998597234e-05,
            -6.1688729054668674e-06,
        ),
        "s": (
            3.3418610987342229e-01,
            1.1666700286207601e-03,
            -2.0450964699343364e-03,
            -1.5833379968858402e-04,
            -3.8596504114596517e-05,
            -4.1901619591027170e-05,
            1.4968468916007242e-05,
            2.6075948416621554e-05,
            -4.8303378655747770e-05,
            -3.4317843409700719e-05,
        ),
        "fit_rms": 3.896377390478151e-04,
        "fit_max_curve_error": 1.0740887012215919e-02,
    },
    "chease": {
        "a": 0.6504127010781183,
        "R0": 0.9999628382309164,
        "Z0": 0.00016201215320952212,
        "kappa": 1.8353512314259297,
        "c": (
            -0.10093500947178713,
            0.09953753013381786,
            0.00263797964542851,
            0.0002364648141952,
            -0.00187058163436749,
            -0.00015749468335108,
            0.00250455340197678,
            -0.00021010577314975,
            -0.00138185135101046,
            0.00046348834914139,
            0.0010925753359023,
        ),
        "s": (
            0.39741190586126168,
            0.30000064059401577,
            -0.19752465029931532,
            0.00012375584334904224,
            -0.0028918496083633863,
            0.00055728321961535845,
            0.0020885844545475403,
            -0.00065686603212885782,
            -0.0016236488464970841,
            0.0010090121427006112,
        ),
        "fit_rms": 6.90058644014939e-04,
        "fit_max_curve_error": 1.218376784194946e-02,
    },
    "efit": {
        "a": 0.6171676117603371,
        "R0": 1.6613762798644713,
        "Z0": -0.08363305046404519,
        "kappa": 1.7821260070974403,
        "c": (
            0.07852828254669175,
            0.06312715933508059,
            -0.07905163910660493,
            -0.01769724397809446,
            0.02910897204124359,
            0.02433733959900054,
            -0.0055909769586503,
            -0.00879177546686944,
            0.00550482646737737,
            0.00504000048961428,
            0.00162409293766496,
        ),
        "s": (
            0.6133689729358927,
            0.04214392213264707,
            -0.12629878943384087,
            0.01888688896953439,
            0.02390779724976376,
            0.02772395371056949,
            -0.00865560279811922,
            -0.00225943350361505,
            0.00283741776166376,
            0.004692674167182,
        ),
        "fit_rms": 2.810771174407942e-04,
        "fit_max_curve_error": 1.729559104966416e-03,
    },
}


@dataclass(frozen=True, slots=True)
class GeqdskBenchmarkCase:
    case_key: str
    config_label: str
    geqdsk: Geqdsk
    plasma: Plasma
    topology: dict[str, Any]
    solver: dict[str, Any]
    signature: dict[str, int]
    boundary_fit: dict[str, float]

    @property
    def label(self) -> str:
        return f"{self.case_key}:{self.config_label}"

    @property
    def geqdsk_path(self) -> Path:
        return CASE_REFERENCE_GFILES[self.case_key]


def benchmark_result_dir(name: str) -> Path:
    return RESULTS_DIR / name


def benchmark_result_path(name: str, filename: str | None = None) -> Path:
    return benchmark_result_dir(name) / (filename or f"{name}.json")


def default_kernel_cache_root() -> Path:
    override = os.environ.get("VEQPY_KERNEL_CACHE")
    return Path(override).expanduser() if override else REPO_ROOT / ".veqpy-kernel-cache"


def runtime_env() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    }


def cpu_affinity() -> list[int] | None:
    getter = getattr(os, "sched_getaffinity", None)
    return None if getter is None else sorted(int(cpu) for cpu in getter(0))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def profile_counts_from_signature(signature: dict[str, int]) -> dict[str, Any]:
    def family(prefix: str, start: int) -> tuple[int, ...]:
        values: list[int] = []
        order = start
        while f"{prefix}{order}" in signature:
            values.append(int(signature[f"{prefix}{order}"]))
            order += 1
        while values and values[-1] == 0:
            values.pop()
        return tuple(values)

    return {
        "h_count": int(signature.get("h", 0)),
        "v_count": int(signature.get("v", 0)),
        "kappa_count": int(signature.get("k", 0)),
        "psin_count": int(signature.get("psin", 0)),
        "F_count": int(signature.get("F", 0)),
        "c_counts": family("c", 0),
        "s_counts": family("s", 1),
    }


def _geqdsk_axis(size: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, int(size), dtype=np.float64)


def _resample_profile(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    axis = _geqdsk_axis(source.size)
    if source.size < 2 or not np.all(np.isfinite(source)):
        return np.zeros_like(target)
    return np.interp(target, axis, source).astype(np.float64, copy=False)


def _make_plasma(geqdsk: Geqdsk, case_key: str, *, nr: int, nt: int) -> Plasma:
    fit = BOUNDARY_FITS[case_key]
    radial_order = 10
    boundary_order = 10
    geometry = Geometry(
        Nr=nr,
        Nt=nt,
        radial_rule="uniform",
        radial_calculus="spectral",
        K_max=20,
        R0=float(fit["R0"]),
        Z0=float(fit["Z0"]),
        a=float(fit["a"]),
        kappa_lcfs=float(fit["kappa"]),
        c_lcfs=np.asarray(fit["c"], dtype=np.float64),
        # Geometry and VEQ both use +s_m sin(m*theta) in the poloidal phase.
        s_lcfs=np.asarray(fit["s"], dtype=np.float64),
        h_coeffs=np.zeros(radial_order + 1, dtype=np.float64),
        v_coeffs=np.zeros(radial_order + 1, dtype=np.float64),
        kappa_coeffs=np.zeros(radial_order + 1, dtype=np.float64),
        c_coeffs=np.zeros((boundary_order + 1, radial_order + 1), dtype=np.float64),
        s_coeffs=np.zeros((boundary_order, radial_order + 1), dtype=np.float64),
    )
    r = np.asarray(geometry.r, dtype=np.float64)
    ff_psi = _resample_profile(geqdsk.FF_psi, r)
    p_psi = _resample_profile(geqdsk.P_psi, r)
    q = _resample_profile(geqdsk.q, r)
    if np.any(q == 0.0):
        q = np.where(q == 0.0, 1.0, q)
    equilibrium = Equilibrium(
        geometry=geometry,
        FF_psi=ff_psi,
        P_psi=p_psi,
        # A constant physical psi_r keeps psin=r on the uniform Plasma grid,
        # while its integral retains the GEQDSK axis-to-LCFS flux span.  The
        # Adapter therefore supplies the historical d/dpsin source values
        # rather than silently treating d/dpsi as d/dpsin.
        psi_r=np.full(
            nr,
            float(geqdsk.psi_bound - geqdsk.psi_axis),
            dtype=np.float64,
        ),
        B0=float(geqdsk.Bt0) if geqdsk.Bt0 != 0.0 else 1.0,
        P0=float(geqdsk.P[-1]) if geqdsk.P.size else 0.0,
    )
    rho = np.linspace(0.0, 1.0, nr, dtype=np.float64)
    q_rho = np.gradient(q, rho)
    current_total = abs(float(geqdsk.Ip))
    itor = current_total * rho
    current_density = np.full(nr, max(current_total, 1.0), dtype=np.float64)
    current = Current(
        rho=rho,
        q=q,
        q_rho=q_rho,
        Itor=itor,
        jtor=current_density,
        jtotal=current_density,
        ellpara=np.full(nr, 20.0, dtype=np.float64),
        etapara=np.full(nr, 1.0e-8, dtype=np.float64),
        jbootstrap=np.zeros(nr, dtype=np.float64),
        jdriven=np.zeros(nr, dtype=np.float64),
    )
    kinetic = Kinetic(
        rho=rho,
        ion_names=("D",),
        Aion=np.array([2.014], dtype=np.float64),
        Znuc=np.array([1], dtype=np.int64),
        Zion=np.ones((1, nr), dtype=np.float64),
        Z2ion=np.ones((1, nr), dtype=np.float64),
        ni=np.full((1, nr), 2.0e19, dtype=np.float64),
        Ti=np.full((1, nr), 5.0e3, dtype=np.float64),
        Te=np.full(nr, 5.0e3, dtype=np.float64),
        omega=np.zeros((1, nr), dtype=np.float64),
    )
    return Plasma(
        equilibrium=equilibrium,
        kinetic=kinetic,
        current=current,
        flux=Flux(rho=rho, ion_names=("D",)),
        source=Source(rho=rho, ion_names=("D",)),
    ).freeze()


def geqdsk_signature(case_key: str, config_label: str) -> dict[str, int]:
    try:
        signature = GEQDSK_CONFIG_SIGNATURES[(case_key, config_label)]
    except KeyError as error:
        raise KeyError(f"missing historical signature for {case_key}:{config_label}") from error
    return {str(name): int(value) for name, value in signature.items() if int(value) > 0}


def geqdsk_kernel_case(
    case_key: str,
    config_label: str,
    *,
    nr: int = REFERENCE_LAYOUT_NR,
    nt: int = REFERENCE_LAYOUT_NT,
    max_evaluations: int = REFERENCE_SOLVER_MAXFEV,
) -> GeqdskBenchmarkCase:
    geqdsk = load_geqdsk(CASE_REFERENCE_GFILES[case_key])
    signature = geqdsk_signature(case_key, config_label)
    counts = profile_counts_from_signature(signature)
    all_counts = [
        counts["h_count"],
        counts["v_count"],
        counts["kappa_count"],
        counts["psin_count"],
        counts["F_count"],
        *counts["c_counts"],
        *counts["s_counts"],
    ]
    l_max = max(1, max(all_counts, default=2) - 1)
    topology = {
        **counts,
        "Nr": int(nr),
        "Nt": int(nt),
        "L_max": int(l_max),
        "M_max": 10,
        "K_max": 10,
        "quadrature": "legendre",
        "calculus": "spectral",
        "route": "PF",
        "coordinate": "psin",
        "constraint": "ip",
    }
    solver = {
        "method": "powell",
        "max_residual": 1.0e-6,
        "max_evaluations": int(max_evaluations),
        "initial": "cold",
        "continuation": "cold",
        "norm": "fast",
    }
    fit = BOUNDARY_FITS[case_key]
    return GeqdskBenchmarkCase(
        case_key=case_key,
        config_label=config_label,
        geqdsk=geqdsk,
        # The Module topology remains the requested 32x32 solve layout.  The
        # external Plasma deliberately retains the GEQDSK profile grid so the
        # benchmark exercises runtime explicit-source counts and capacity
        # growth (128 for CHEASE, 257 for SOLOVEV/EFIT).
        plasma=_make_plasma(
            geqdsk,
            case_key,
            nr=int(geqdsk.P_psi.size),
            nt=nt,
        ),
        topology=topology,
        solver=solver,
        signature=signature,
        boundary_fit={
            "fit_rms": float(fit["fit_rms"]),
            "fit_max_curve_error": float(fit["fit_max_curve_error"]),
            "fit_c_order": 10.0,
            "fit_s_order": 10.0,
        },
    )


def selected_cases(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return CASE_KEYS
    selected = tuple(str(value).lower() for value in values)
    unknown = sorted(set(selected) - set(CASE_KEYS))
    if unknown:
        raise ValueError(f"unknown case(s): {', '.join(unknown)}")
    return selected


def selected_configs(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return CONFIG_LABELS
    by_lower = {value.lower(): value for value in CONFIG_LABELS}
    try:
        return tuple(by_lower[str(value).lower()] for value in values)
    except KeyError as error:
        raise ValueError(f"unknown config {error.args[0]!r}") from error


def build_module(case: GeqdskBenchmarkCase, backend: str, *, artifact_dir: Path):
    return build(
        topology=case.topology,
        solver=case.solver,
        backend=backend,
        artifact_dir=artifact_dir,
        materialize=False,
        verbose=False,
        report=False,
    )


def prepare_metadata(module: Any, backend: str) -> dict[str, Any]:
    implementation = module._kernel._impl
    prepared = implementation.prepare()
    artifact = prepared.artifact
    payload: dict[str, Any] = {
        "backend": backend,
        "recipe_backend": str(implementation.recipe.backend),
        "recipe_build": str(implementation.recipe.build),
        "artifact_dir": str(getattr(module, "_artifact_dir", "")),
    }
    if artifact is None:
        payload["artifact"] = "python-numba"
    else:
        payload["artifact"] = {
            "artifact_id": str(artifact.artifact_id),
            "reused": bool(artifact.reused),
            "built": bool(artifact.built),
            "root_dir": str(artifact.root_dir),
            "shared_library": str(artifact.shared_library_path),
            "compiler": artifact.metadata.get("compiler"),
            "cmake": artifact.metadata.get("cmake"),
        }
    return payload


def statistics_payload(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        return {"count": 0, "min_ms": None, "p25_ms": None, "median_ms": None, "p75_ms": None, "max_ms": None}
    return {
        "count": len(samples),
        "min_ms": float(min(samples)),
        "p25_ms": float(np.percentile(samples, 25)),
        "median_ms": float(statistics.median(samples)),
        "p75_ms": float(np.percentile(samples, 75)),
        "max_ms": float(max(samples)),
    }


def integer_statistics(values: Sequence[int]) -> dict[str, Any]:
    samples = [int(value) for value in values]
    return {
        "count": len(samples),
        "min": min(samples) if samples else None,
        "median": float(statistics.median(samples)) if samples else None,
        "max": max(samples) if samples else None,
    }


def monotonic_interleave(items: Sequence[str], iteration: int) -> tuple[str, ...]:
    values = tuple(items)
    if not values:
        return ()
    offset = int(iteration) % len(values)
    return values[offset:] + values[:offset]


def time_call(function) -> tuple[Any, float]:
    started = perf_counter_ns()
    value = function()
    elapsed_ms = (perf_counter_ns() - started) / 1.0e6
    return value, float(elapsed_ms)


__all__ = [
    "BACKENDS",
    "BOUNDARY_FITS",
    "CASE_KEYS",
    "CASE_REFERENCE_GFILES",
    "CONFIG_LABELS",
    "CORE_DIR",
    "ENZYME_SKIP_REASON",
    "GeqdskBenchmarkCase",
    "HISTORICAL_SPEEDUP_RANGE",
    "REFERENCE_LAYOUT_NR",
    "REFERENCE_LAYOUT_NT",
    "REFERENCE_SOLVER_MAXFEV",
    "REPO_ROOT",
    "RUNNABLE_BACKENDS",
    "benchmark_result_path",
    "build_module",
    "cpu_affinity",
    "default_kernel_cache_root",
    "geqdsk_kernel_case",
    "geqdsk_signature",
    "integer_statistics",
    "monotonic_interleave",
    "prepare_metadata",
    "runtime_env",
    "selected_cases",
    "selected_configs",
    "statistics_payload",
    "time_call",
    "write_json",
]
