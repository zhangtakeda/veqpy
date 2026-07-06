"""Shared terminal reporting helpers for benchmark entry points."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from rich import box
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

REPORT_TABLE_BOX = box.Box("    \n    \n ── \n    \n ── \n ── \n    \n ── \n")


def console() -> Console:
    return Console(highlight=False)


def progress_context(console: Console, *, quiet: bool, width: int = 24) -> Any:
    if quiet:
        return nullcontext(None)
    return Progress(
        TextColumn(f"[dim]{{task.fields[current]:<{width}.{width}}}[/]"),
        BarColumn(bar_width=48, complete_style="cyan", finished_style="green", pulse_style="cyan"),
        MofNCompleteColumn(),
        TextColumn("{task.fields[phase]:>8}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def print_config_tree(console: Console, lines: tuple[str, ...]) -> None:
    console.print(Text("[config]", style="bold cyan"))
    for index, line in enumerate(lines):
        branch = "└──" if index == len(lines) - 1 else "├──"
        console.print(f"  {branch} {line}")


def print_outputs_tree(console: Console, outputs: dict[str, Path], *, repo_root: Path) -> None:
    if not outputs:
        return
    console.print(Text("[outputs]", style="bold cyan"))
    paths: list[Path] = []
    for path in outputs.values():
        try:
            display_path = path.resolve().relative_to(repo_root)
        except ValueError:
            display_path = path
        paths.append(display_path)
    for index, path in enumerate(paths):
        branch = "└──" if index == len(paths) - 1 else "├──"
        console.print(f"  {branch} [green]{path}[/]")


def status_cell(status: object) -> str:
    text = str(status)
    if text == "passed":
        return "[green]passed[/]"
    if text == "failed":
        return "[red]failed[/]"
    if text == "not_requested":
        return "[blue]not requested[/]"
    if text.startswith("blocked"):
        return "[yellow]blocked[/]"
    if text.startswith("skipped") or text == "invalid":
        return "[yellow]skipped[/]"
    if text == "planned":
        return "[blue]planned[/]"
    return text


def progress_phase(status: object) -> str:
    text = str(status)
    if text == "passed":
        return "[green]passed[/]"
    if text == "failed":
        return "[red]failed[/]"
    if text == "not_requested":
        return "[blue]skip[/]"
    if text.startswith("blocked"):
        return "[yellow]blocked[/]"
    if text.startswith("skipped"):
        return "[yellow]skipped[/]"
    if text == "planned":
        return "[blue]plan[/]"
    return "[dim]done[/]"


def format_optional_float(value: object, *, precision: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{precision}f}"


def format_optional_sci(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.2e}"


def format_optional_speedup(py_ms: object, cxx_ms: object) -> str:
    try:
        py_value = float(py_ms)
        cxx_value = float(cxx_ms)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(py_value) or not np.isfinite(cxx_value) or cxx_value <= 0.0:
        return "n/a"
    return f"{py_value / cxx_value:.3f}x"


def nfev_median(engine: dict[str, Any] | None) -> str:
    if engine is None:
        return "n/a"
    nfev = engine.get("nfev")
    if not isinstance(nfev, dict):
        return "n/a"
    value = nfev.get("median")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return str(int(number)) if number.is_integer() else f"{number:.3g}"


def runtime_engine_payload(runtime: dict[str, Any], label: str) -> dict[str, Any] | None:
    engines = runtime.get("engines")
    if not isinstance(engines, dict):
        return None
    engine = engines.get(label)
    return engine if isinstance(engine, dict) else None


def timing_median_ms(engine: dict[str, Any] | None) -> float:
    if engine is None:
        return float("nan")
    timing = engine.get("timing")
    if not isinstance(timing, dict):
        return float("nan")
    return float(timing.get("median_ms", float("nan")))


def print_runtime_summary(console: Console, summary: dict[str, int], keys: tuple[str, ...]) -> None:
    counts = Table(box=REPORT_TABLE_BOX, show_lines=False, expand=False, padding=(0, 1))
    counts.add_column("summary")
    counts.add_column("count", justify="right")
    for key in keys:
        counts.add_row(key.replace("_", " "), str(summary.get(key, 0)))
    console.print(counts)


def print_runtime_failures(console: Console, rows: list[dict[str, Any]]) -> None:
    failed = [row for row in rows if row.get("runtime", {}).get("status") == "failed"]
    if not failed:
        return
    console.print()
    tree = Tree(Text("[failures]", style="bold red"))
    for row in failed:
        runtime = row.get("runtime", {})
        detail = runtime.get("failure_reason") or runtime.get("error") or "failed"
        tree.add(f"{row.get('case', 'n/a')}: {detail}")
    console.print(tree)
    console.print()
