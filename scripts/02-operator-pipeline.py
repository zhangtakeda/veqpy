"""Render the operator-pipeline SVG into publication raster/vector outputs.

This script is intentionally limited to format conversion.  The source diagram
stays in ``figures/02-operator-pipeline.svg`` while ``PNG_PATH`` and
``PDF_PATH`` control which derived artifacts are written.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cairosvg
from config import (
    SCRIPT_CONSOLE,
    figure_path,
    print_output_table,
    print_script_config,
    script_progress,
)
from PIL import Image

FIGURE_WIDTH_IN = 6.5
SAVE_DPI = 500
SOURCE_SVG_PATH = Path(figure_path("02-operator-pipeline.svg"))
PNG_PATH = SOURCE_SVG_PATH.with_suffix(".png")
PDF_PATH = None


def render_svg_outputs(
    source_svg: Path,
    *,
    png_path: str | Path | None,
    pdf_path: str | Path | None,
    width_in: float,
    dpi: int,
) -> list[tuple[str, str | Path, str | None]]:
    rows: list[tuple[str, str | Path, str | None]] = []
    if png_path is None and pdf_path is None:
        return rows
    if not source_svg.is_file():
        raise FileNotFoundError(f"Missing source SVG: {source_svg}")

    output_width_px = round(width_in * dpi)
    if png_path is not None:
        png_path = Path(png_path)
        png_bytes = cairosvg.svg2png(url=str(source_svg), output_width=output_width_px)

        png_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(BytesIO(png_bytes)) as image:
            image.save(png_path, dpi=(dpi, dpi))

        with Image.open(png_path) as image:
            width_px, height_px = image.size
        rows.append(
            (
                "Figure 02 PNG",
                png_path,
                f"{width_px}x{height_px} px ({width_px / dpi:.3f}x{height_px / dpi:.3f} in)",
            )
        )

    if pdf_path is not None:
        pdf_path = Path(pdf_path)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        cairosvg.svg2pdf(
            url=str(source_svg),
            write_to=str(pdf_path),
            output_width=output_width_px,
        )
        rows.append(("Figure 02 PDF", pdf_path, f"width {width_in:.3f} in"))
    return rows


def main() -> None:
    print_script_config(
        SCRIPT_CONSOLE,
        "figure 02: operator pipeline",
        (
            ("source", SOURCE_SVG_PATH),
            ("width", f"{FIGURE_WIDTH_IN:.3f} in"),
            ("dpi", SAVE_DPI),
        ),
    )
    with script_progress(SCRIPT_CONSOLE) as progress:
        task = progress.add_task(
            "",
            total=1,
            current="convert SVG",
            phase="[cyan]run[/]",
        )
        rows = render_svg_outputs(
            SOURCE_SVG_PATH,
            png_path=PNG_PATH,
            pdf_path=PDF_PATH,
            width_in=FIGURE_WIDTH_IN,
            dpi=SAVE_DPI,
        )
        progress.update(task, advance=1, current="convert SVG", phase="[green]done[/]")
    print_output_table(SCRIPT_CONSOLE, rows)


if __name__ == "__main__":
    main()
