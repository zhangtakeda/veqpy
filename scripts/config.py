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
from functools import lru_cache
from pathlib import Path
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
REDUCED_EQUILIBRIUM_JSON_TEMPLATE = data_path("pareto_reduced_{case_key}_{config_label}.json")
REDUCED_EQUILIBRIUM_MANIFEST_PATH = data_path("pareto_reduced_equilibria.json")


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
    from veqpy.model import Equilibrium, Geqdsk

    return {
        "Equilibrium": Equilibrium,
        "Geqdsk": Geqdsk,
    }


def read_geqdsk(path: str):
    geqdsk = load_veqpy_components()["Geqdsk"]()
    geqdsk.read_geqdsk(str(path))
    return geqdsk


def load_equilibrium_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing equilibrium JSON: {path}")
    return load_veqpy_components()["Equilibrium"].load(path)


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
            "Generate the reduced-equilibrium manifest before running this figure."
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
            "Generate the reference-equilibrium manifest before running this figure."
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
            "Generate the reduced-equilibrium manifest before running this figure."
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
            "Generate the reference-equilibrium manifest before running this figure."
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
