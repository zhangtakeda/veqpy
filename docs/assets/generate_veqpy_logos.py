#!/usr/bin/env python3

import math
from pathlib import Path

FONT_FAMILY = "Inter, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
VEQ_ATTRS = 'fill="#0f172a"'


def fmt(value):
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def d_shape_points(
    cx,
    cy,
    a,
    kappa,
    delta,
    start=0,
    end=math.tau,
    n=220,
):
    pts = []
    for i in range(n + 1):
        theta = start + (end - start) * i / n
        x = cx + a * (math.cos(theta) - delta * math.sin(theta) ** 2)
        y = cy - a * kappa * math.sin(theta)
        pts.append((x, y))
    return pts


def pts_to_path(points, close=False):
    coords = points[:-1] if close else points
    path = "M" + " ".join(f"{fmt(x)},{fmt(y)}" for x, y in coords)
    return path + ("Z" if close else "")


def svg_defs():
    return (
        '<defs><linearGradient id="p" x1="0%" y1="10%" x2="100%" y2="90%">'
        '<stop offset="0%" stop-color="#0f172a"/><stop offset="48%" '
        'stop-color="#2563eb"/><stop offset="100%" stop-color="#06b6d4"/>'
        '</linearGradient><linearGradient id="w"><stop offset="0%" '
        'stop-color="#2563eb"/><stop offset="100%" stop-color="#06b6d4"/>'
        "</linearGradient></defs>"
    )


def opacity_attr(opacity):
    return "" if opacity == 1 else f' opacity="{fmt(opacity)}"'


def make_flux_mark_group(
    cx,
    cy,
    max_a,
    stroke_width_outer=11,
    kappa=1.42,
    delta=0.34,
):
    parts = [
        "<g>",
        '<g fill="none" stroke="url(#p)" stroke-linejoin="round" stroke-linecap="round">',
    ]

    for t, width_scale, opacity in (
        (1, 1.12, 1),
        (0.68, 0.54, 0.78),
        (0.38, 0.31, 0.58),
    ):
        path = pts_to_path(
            d_shape_points(cx, cy, max_a * t, kappa, delta),
            close=True,
        )
        parts.append(
            f'<path d="{path}" '
            f'stroke-width="{fmt(stroke_width_outer * width_scale)}"{opacity_attr(opacity)}/>'
        )

    parts.append('</g><g fill="none" stroke="#06b6d4" stroke-linecap="round">')
    for t, width_scale, opacity, start_deg, end_deg in (
        (1, 1, 0.9, 318, 382),
        (0.68, 0.58, 0.62, 324, 376),
    ):
        path = pts_to_path(
            d_shape_points(
                cx,
                cy,
                max_a * t,
                kappa,
                delta,
                math.radians(start_deg),
                math.radians(end_deg),
                n=70,
            )
        )
        parts.append(
            f'<path d="{path}" '
            f'stroke-width="{fmt(stroke_width_outer * width_scale)}"{opacity_attr(opacity)}/>'
        )

    parts.append("</g></g>")
    return "".join(parts)


def make_banner_svg(width=1100, height=340):
    cy = height / 2
    title_dy = 12
    subtitle_dy = 20
    mark = make_flux_mark_group(
        131.36,
        cy,
        92,
        stroke_width_outer=9.2,
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">{svg_defs()}{mark}<g '
        f'transform="translate(283.36)" font-family="{FONT_FAMILY}" letter-spacing="0">'
        f'<text y="{fmt(cy + 8 + title_dy)}" font-size="116" font-weight="800">'
        f'<tspan {VEQ_ATTRS}>VEQ</tspan><tspan fill="url(#w)">Py</tspan></text>'
        f'<text x="6" y="{fmt(cy + 66 + subtitle_dy)}" font-size="30" font-weight="500" '
        'letter-spacing=".2" fill="#64748b">Fixed-boundary Grad-Shafranov solver'
        "</text></g></svg>\n"
    )


def make_icon_svg(size=512):
    mark = make_flux_mark_group(
        256,
        190,
        104,
        stroke_width_outer=11.7,
        kappa=1.38,
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">{svg_defs()}{mark}<text x="256" y="442" '
        f'text-anchor="middle" font-family="{FONT_FAMILY}" font-size="104" '
        f'font-weight="850" letter-spacing="0"><tspan {VEQ_ATTRS}>VEQ</tspan>'
        '<tspan fill="url(#w)">Py</tspan></text></svg>\n'
    )


def generate_veqpy_logos(out_dir="docs/assets"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, content in {
        "veqpy_banner.svg": make_banner_svg(),
        "veqpy_icon.svg": make_icon_svg(),
    }.items():
        path = out / name
        path.write_text(content, encoding="utf-8")
        print(f"saved: {path}")


def main():
    generate_veqpy_logos()


if __name__ == "__main__":
    main()
else:
    raise RuntimeError("This script is not meant to be imported.")
