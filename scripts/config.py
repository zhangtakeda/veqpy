"""Shared visualization configuration for manuscript figure scripts.

The constants in this module are intentionally value-for-value copies of the
settings that were previously duplicated across the individual plotting scripts.
Keeping them here makes paper-wide typography, output resolution, and one-/two-
column sizing explicit without changing any generated figures.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from rich import box
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def repo_path(*parts: str) -> str:
    return os.fspath(REPO_ROOT.joinpath(*parts))


def data_path(filename: str) -> str:
    return repo_path("data", filename)


def figure_path(filename: str) -> str:
    return repo_path("figures", filename)


# Typography shared by all Matplotlib figures.
PLOT_FONT_FAMILY = "DejaVu Sans"
PLOT_BASE_FONT_SIZE = 10
PLOT_MATH_FONTSET = "dejavusans"
FONT_SCALE = 1.0

TITLE_FONT_SIZE = 9
AXIS_LABEL_FONT_SIZE = 9
TICK_LABEL_FONT_SIZE = 8
LEGEND_FONT_SIZE = 8

# Short aliases used by the residual diagnostic scripts.
AXIS_LABEL_SIZE = AXIS_LABEL_FONT_SIZE
TICK_LABEL_SIZE = TICK_LABEL_FONT_SIZE

# Manuscript layout widths in inches.
SINGLE_COLUMN_WIDTH = 4.75
DOUBLE_COLUMN_WIDTH = 10.0

# Output options shared by all figure exporters.
SAVE_DPI = 330
SAVE_TRANSPARENT = False
FIGURE_FACE_COLOR = "white"


def ensure_parent_dir(path: str | os.PathLike[str] | None) -> None:
    if path is None:
        return
    parent = os.path.dirname(os.fspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_figure_outputs(
    fig: Any,
    *,
    png_path: str | os.PathLike[str] | None,
    pdf_path: str | os.PathLike[str] | None = None,
    dpi: int = SAVE_DPI,
    transparent: bool = SAVE_TRANSPARENT,
    facecolor: str | None = None,
) -> list[str]:
    saved_paths: list[str] = []
    for output_path in (png_path, pdf_path):
        if output_path is None:
            continue
        ensure_parent_dir(output_path)
        save_kwargs: dict[str, Any] = {
            "dpi": dpi,
            "transparent": transparent,
        }
        if facecolor is not None:
            save_kwargs["facecolor"] = facecolor
        path_text = os.fspath(output_path)
        fig.savefig(path_text, **save_kwargs)
        saved_paths.append(path_text)
    return saved_paths


# Numeric precision shared by generated LaTeX tables.
FIXED_DECIMALS = 2
SCIENTIFIC_DECIMALS = 2

# Rich command-line reporting shared by manuscript scripts.  The table box
# matches the benchmark entry points so figure/table regeneration has one
# console vocabulary.
REPORT_TABLE_BOX = box.Box("    \n    \n ── \n    \n ── \n ── \n    \n ── \n")
SCRIPT_CONSOLE = Console()


def script_progress(console: Console = SCRIPT_CONSOLE, *, quiet: bool = False):
    """Return the benchmark-style progress context used by figure scripts."""
    if quiet:
        return nullcontext(None)
    return Progress(
        TextColumn("[dim]{task.fields[current]:<28.28}[/]"),
        BarColumn(
            bar_width=48,
            complete_style="cyan",
            finished_style="green",
            pulse_style="cyan",
        ),
        MofNCompleteColumn(),
        TextColumn("{task.fields[phase]:>10}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def script_display_path(path: str | os.PathLike[str]) -> str:
    path_obj = Path(path)
    try:
        return os.fspath(path_obj.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return os.fspath(path_obj)


def print_script_config(
    console: Console,
    title: str,
    rows: list[tuple[str, object]] | tuple[tuple[str, object], ...],
) -> None:
    console.print(Text(f"[{title}]", style="bold cyan"))
    for index, (name, value) in enumerate(rows):
        branch = "└──" if index == len(rows) - 1 else "├──"
        console.print(f"  {branch} {name}: [green]{value}[/]")


def make_script_table(title: str, columns: list[tuple[str | Text, str]]) -> Table:
    table = Table(
        title=title,
        box=REPORT_TABLE_BOX,
        show_lines=False,
        expand=False,
        padding=(0, 1),
    )
    for column_title, justify in columns:
        table.add_column(column_title, justify=justify)
    return table


def print_script_table(console: Console, table: Table) -> None:
    console.print(table)


def print_output_table(
    console: Console,
    rows: list[tuple[str, str | os.PathLike[str], str | None]],
) -> None:
    if not rows:
        return
    table = make_script_table(
        "outputs",
        [("artifact", "left"), ("path", "left"), ("detail", "left")],
    )
    for artifact, path, detail in rows:
        table.add_row(str(artifact), f"[green]{script_display_path(path)}[/]", detail or "")
    console.print(table)


def format_script_float(value: float | None, *, decimals: int = FIXED_DECIMALS) -> str:
    if value is None:
        return "--"
    value = float(value)
    if not np.isfinite(value):
        return "--"
    return f"{value:.{decimals}f}"


def format_script_sci(value: float | None, *, decimals: int = SCIENTIFIC_DECIMALS) -> str:
    if value is None:
        return "--"
    value = float(value)
    if not np.isfinite(value):
        return "--"
    return f"{value:.{decimals}e}"


# Common tick styling defaults.
PLOT_TICK_DIRECTION = "in"
PLOT_TICK_TOP = True
PLOT_TICK_RIGHT = True
PLOT_TICK_BOTTOM = True
PLOT_TICK_LEFT = True
PLOT_LABEL_TOP = False
PLOT_LABEL_RIGHT = False

LINESTYLE_A = "-"
LINESTYLE_B = "--"
LINESTYLE_C = (0, (5, 1, 1, 1, 1, 1))

# Shared benchmark/data cases used by the manuscript diagnostics.
MU0 = 4.0e-7 * 3.141592653589793
CASE_KEYS = ("solovev", "chease", "efit")
CASE_LABELS = {
    "solovev": "D-shaped",
    "chease": "H-mode",
    "efit": "X-point",
}
REFERENCE_LABELS = {case_key: "GEQDSK" for case_key in CASE_KEYS}
CASE_COLORS = {
    "solovev": "#1f77b4",
    "chease": "#ff7f0e",
    "efit": "#2ca02c",
}
CASE_LINESTYLES = {
    "solovev": "-",
    "chease": "--",
    "efit": LINESTYLE_C,
}
CASE_LINE_COLORS = {
    "solovev": ("#101010", "#777777", "#74a9cf", "#1f77b4", "#08306b"),
    "chease": ("#101010", "#777777", "#fdb863", "#ff7f0e", "#7f2704"),
    "efit": ("#101010", "#777777", "#74c476", "#2ca02c", "#00441b"),
}
CONFIG_LABELS = ("Low", "Medium", "High", "Ref")
REDUCED_CONFIG_LABELS = ("Low", "Medium", "High")
CONFIG_LINE_COLORS = {
    "Low": "#777777",
    "Medium": "#999999",
    "High": "#555555",
    "Ref": "#111111",
}
LOW_LINESTYLE = (0, (5, 1.6, 1.2, 1.6, 1.2, 1.6))
LEVEL_LINESTYLES = {
    "Ref": ":",
    "Low": LOW_LINESTYLE,
    "Medium": "--",
    "High": "-",
}

CASE_REFERENCE_GFILES = {
    "solovev": data_path("SOLOVEV.geqdsk"),
    "chease": data_path("CHEASE.geqdsk"),
    "efit": data_path("EFIT.geqdsk"),
}
CASE_REFERENCE_EQUILIBRIUM_JSONS = {
    "solovev": data_path("solovev-equilibrium.json"),
    "chease": data_path("chease-equilibrium.json"),
    "efit": data_path("efit-equilibrium.json"),
}
REFERENCE_EQUILIBRIUM_MANIFEST_PATH = data_path("reference_equilibria.json")
CASE_REFERENCE_PROFILE_LENGTHS = {
    "demo(psin)": {
        "psin": [0.0] * 6,
        "h": [0.0] * 3,
        "k": [0.0] * 6,
        "s1": [0.0] * 3,
    },
    "demo(rho)": {
        "h": [0.0] * 3,
        "k": [0.0] * 6,
        "s1": [0.0] * 3,
    },
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
REDUCED_EQUILIBRIUM_JSON_TEMPLATE = data_path("pareto_reduced_{case_key}_{config_label}.json")
REDUCED_EQUILIBRIUM_MANIFEST_PATH = data_path("pareto_reduced_equilibria.json")
DEFAULT_JSON_STEM = data_path("pareto")

REFERENCE_LAYOUT_NR = 32
REFERENCE_LAYOUT_NT = 32
REFERENCE_SOLVER_MAXFEV = 2000
SOLVER_INITIAL_POLICY = "auto"
TEST_SOURCE_SAMPLE_COUNT = 51
BOUNDARY_MAXTOL = 1.0
CASE_SOLVER_METHODS = {
    "solovev": "hybr",
    "efit": "hybr",
    "chease": "hybr",
}
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

# Figure 03/04 demo case stored as plain data so both scripts can construct
# their own VEQPy objects without importing each other.
DEMO_GRID = {
    "Nr": 64,
    "Nt": 64,
    "quadrature_scheme": "legendre",
}
DEMO_SNAPSHOT_GRID = {
    "Nr": 128,
    "Nt": 256,
    "quadrature_scheme": "uniform",
}
DEMO_BOUNDARY = {
    "a": 1.05 / 1.85,
    "R0": 1.05,
    "Z0": 0.0,
    "B0": 3.0,
    "ka": 2.2,
    "s_offsets": (0.0, 0.5235987755982989),
}
DEMO_SOLVER_CONFIG = {
    "method": "hybr",
    "enable_verbose": False,
}
DEMO_PROFILE_COEFFS = {
    "psin": [0.0] * 5,
    "h": [0.0] * 3,
    "k": [0.0] * 5,
    "s1": [0.0] * 3,
}
DEMO_SOURCE_SAMPLE_COUNT = 128
DEMO_ROUTE = "PF"
DEMO_COORDINATE = "psin"
DEMO_NODES = "uniform"
DEMO_IP = 3.0e6


@dataclass(frozen=True)
class PreparedInterpAxis:
    unique_axis: np.ndarray
    order: np.ndarray
    unique_index: np.ndarray


@dataclass(frozen=True)
class PfReferenceCase:
    case_key: str
    boundary: object
    geqdsk: object
    equilibrium: object
    ref_profiles: dict[str, np.ndarray | float]
    psin_interp_axis: PreparedInterpAxis


def scaled_font_size(size: float) -> float:
    """Scale figure text sizes with the paper-wide font scale."""
    return size * FONT_SCALE


def plot_style_rcparams(
    font_family: str = PLOT_FONT_FAMILY,
    *,
    font_size: float = PLOT_BASE_FONT_SIZE,
    math_fontset: str = PLOT_MATH_FONTSET,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return shared Matplotlib rcParams without importing pyplot at module load."""
    rcparams: dict[str, Any] = {
        "font.family": font_family,
        "font.size": font_size,
        "mathtext.fontset": math_fontset,
        "axes.unicode_minus": False,
    }
    if extra:
        rcparams.update(extra)
    return rcparams


def apply_plot_style(
    font_family: str = PLOT_FONT_FAMILY,
    *,
    font_size: float = PLOT_BASE_FONT_SIZE,
    math_fontset: str = PLOT_MATH_FONTSET,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Apply the shared Matplotlib style used by manuscript figure scripts."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        plot_style_rcparams(
            font_family,
            font_size=font_size,
            math_fontset=math_fontset,
            extra=extra,
        )
    )


def demo_psin_reference_profiles(psin):
    """Return the PF source profiles shared by Figures 03 and 04."""
    import numpy as np

    psin = np.asarray(psin, dtype=np.float64)
    beta0 = 0.75
    alpha_p, alpha_f = 5.0, 3.32
    exp_ap = np.exp(alpha_p)
    exp_af = np.exp(alpha_f)
    den_p = 1.0 + exp_ap * (alpha_p - 1.0)
    den_f = 1.0 + exp_af * (alpha_f - 1.0)

    current_input = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f
    heat_input = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p
    return current_input, heat_input


@lru_cache(maxsize=1)
def load_veqpy_components() -> dict[str, object]:
    from veqpy.model import Boundary, Equilibrium, Geqdsk, Grid, Problem
    from veqpy.model.boundary import _fit_boundary_params
    from veqpy.operator import (
        Operator,
        build_profile_index,
        build_profile_layout,
        build_profile_names,
        build_shape_profile_names,
    )
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
        "build_profile_index": build_profile_index,
        "build_profile_layout": build_profile_layout,
        "build_profile_names": build_profile_names,
        "build_shape_profile_names": build_shape_profile_names,
    }


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


def coefficients_from_profile_coeffs(profile_coeffs: Mapping[str, object]) -> dict[str, np.ndarray]:
    coefficients: dict[str, np.ndarray] = {}
    for name, coeff in profile_coeffs.items():
        if coeff is None:
            continue
        if isinstance(coeff, (int, np.integer)):
            length = int(coeff)
            if length <= 0:
                continue
            coeff_array = np.zeros(length, dtype=np.float64)
        else:
            coeff_array = np.asarray(coeff, dtype=np.float64)
            if coeff_array.size <= 0:
                continue
        coefficients[str(name)] = coeff_array
    return coefficients


def read_geqdsk(path: str):
    geqdsk = load_veqpy_components()["Geqdsk"]()
    geqdsk.read_geqdsk(str(path))
    return geqdsk


def load_equilibrium_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing equilibrium JSON: {path}")
    return load_veqpy_components()["Equilibrium"].load(path)


def as_float64_array(values, *, copy: bool = False) -> np.ndarray:
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


def build_pf_reference_profiles(equilibrium) -> dict[str, np.ndarray | float]:
    psin_r = as_float64_array(equilibrium.psin_r, copy=True)
    psin_r_safe = np.where(np.abs(psin_r) > 1.0e-14, psin_r, 1.0e-14)
    pn_psin = as_float64_array(equilibrium.Pn_r, copy=True) / psin_r_safe
    return {
        "psin": as_float64_array(equilibrium.psin, copy=True),
        "FFn_psin": as_float64_array(equilibrium.FFn_r, copy=True) / psin_r_safe,
        "Pn_psin": pn_psin,
        "setup_Pn_psin": pn_psin / MU0,
    }


def build_geqdsk_boundary(geqdsk, *, fit_m: int, fit_n: int, return_fit: bool = False):
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


def build_pf_case(benchmark, reference: PfReferenceCase, signature: dict[str, int]):
    return benchmark.Problem(
        route="PF",
        coordinate="psin",
        nodes="uniform",
        active_profiles=active_profiles_from_coeffs(
            make_profile_coeffs(
                signature, max_lengths=CASE_REFERENCE_PROFILE_LENGTHS[reference.case_key]
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


def metadata_float(
    entry: dict[str, object], key: str, default: float | None = None
) -> float | None:
    value = entry.get(key)
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    import math

    return parsed if math.isfinite(parsed) else default


def metadata_int(entry: dict[str, object], key: str) -> int | None:
    value = entry.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_signature(signature: dict[str, int]) -> dict[str, int]:
    return {str(name): int(length) for name, length in sorted(signature.items()) if int(length) > 0}


def signature_from_metadata(entry: dict[str, object]) -> dict[str, int]:
    signature = entry.get("signature", {})
    return normalize_signature(signature) if isinstance(signature, dict) else {}
