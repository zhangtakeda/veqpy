"""Rich console reporting helpers for manuscript scripts.

This mirrors the benchmark entry points: each script can print a compact config
tree, use a shared progress bar, and finish with an outputs table.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from _common import REPO_ROOT
from rich import box
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

FIXED_DECIMALS = 2
SCIENTIFIC_DECIMALS = 2

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
