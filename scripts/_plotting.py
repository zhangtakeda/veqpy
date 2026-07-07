"""Shared Matplotlib style constants for manuscript scripts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Typography shared by all Matplotlib figures.
PLOT_FONT_FAMILY = "DejaVu Sans"
PLOT_BASE_FONT_SIZE = 10
PLOT_MATH_FONTSET = "dejavusans"
FONT_SCALE = 1.0

TITLE_FONT_SIZE = 9
AXIS_LABEL_FONT_SIZE = 9
TICK_LABEL_FONT_SIZE = 8
LEGEND_FONT_SIZE = 8

# Short aliases used by residual diagnostic scripts.
AXIS_LABEL_SIZE = AXIS_LABEL_FONT_SIZE
TICK_LABEL_SIZE = TICK_LABEL_FONT_SIZE

# Manuscript layout widths in inches.
SINGLE_COLUMN_WIDTH = 4.75
DOUBLE_COLUMN_WIDTH = 10.0

# Output options shared by all figure exporters.
SAVE_DPI = 330
SAVE_TRANSPARENT = False
FIGURE_FACE_COLOR = "white"

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
