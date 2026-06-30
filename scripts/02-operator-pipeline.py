"""Render the operator-pipeline SVG into publication raster/vector outputs.

This script is intentionally limited to format conversion.  The source diagram
stays in ``figures/02-operator-pipeline.svg`` while ``PNG_PATH`` and
``PDF_PATH`` control which derived artifacts are written.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cairosvg
from config import figure_path
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
) -> None:
    if png_path is None and pdf_path is None:
        return
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
        print(f"saved: {png_path}")
        print(
            f"size: {width_px}x{height_px} px "
            f"({width_px / dpi:.3f}x{height_px / dpi:.3f} in @ {dpi} dpi)"
        )

    if pdf_path is not None:
        pdf_path = Path(pdf_path)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        cairosvg.svg2pdf(
            url=str(source_svg),
            write_to=str(pdf_path),
            output_width=output_width_px,
        )
        print(f"saved: {pdf_path}")


def main() -> None:
    render_svg_outputs(
        SOURCE_SVG_PATH,
        png_path=PNG_PATH,
        pdf_path=PDF_PATH,
        width_in=FIGURE_WIDTH_IN,
        dpi=SAVE_DPI,
    )


if __name__ == "__main__":
    main()
