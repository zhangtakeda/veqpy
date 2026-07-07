"""Shared filesystem and figure-output helpers for manuscript scripts.

The module is intentionally small: it owns repository-relative paths and the
Matplotlib save wrapper used by the figure entry points. Plot styling, case
definitions, and Kernel helpers live in narrower private script modules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def repo_path(*parts: str) -> str:
    return os.fspath(REPO_ROOT.joinpath(*parts))


def data_path(filename: str) -> str:
    return repo_path("data", filename)


def figure_path(filename: str) -> str:
    return repo_path("figures", filename)


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
    dpi: int,
    transparent: bool,
    facecolor: str | None = None,
) -> list[str]:
    saved_paths: list[str] = []
    for output_path in (png_path, pdf_path):
        if output_path is None:
            continue
        ensure_parent_dir(output_path)
        save_kwargs: dict[str, Any] = {
            "dpi": int(dpi),
            "transparent": bool(transparent),
        }
        if facecolor is not None:
            save_kwargs["facecolor"] = facecolor
        path_text = os.fspath(output_path)
        fig.savefig(path_text, **save_kwargs)
        saved_paths.append(path_text)
    return saved_paths
