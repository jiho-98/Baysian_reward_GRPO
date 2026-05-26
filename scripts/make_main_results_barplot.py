#!/usr/bin/env python3
"""Create publication-ready bar plots for the main controlled comparison.

This script intentionally uses only the Python standard library so it can run on
servers without matplotlib/cairo/inkscape. It writes both a vector PDF for LaTeX
and an SVG preview.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


OUT_DIR = Path("outputs/paper_figures")
PDF_PATH = OUT_DIR / "main_results_barplot.pdf"
SVG_PATH = OUT_DIR / "main_results_barplot.svg"


BENCHMARKS = ["GSM8K", "MATH 500", "Minerva", "Olympiad", "Avg."]
METHODS = ["Base", "GRPO", "Dr.GRPO", "BPR-GRPO"]

DATA = {
    "Qwen3-1.7B": {
        "Base": [76.99, 58.60, 13.97, 26.41, 43.99],
        "GRPO": [77.56, 59.60, 15.62, 27.23, 45.00],
        "Dr.GRPO": [77.18, 59.00, 15.07, 27.45, 44.68],
        "BPR-GRPO": [79.87, 61.20, 15.81, 29.34, 46.55],
    },
    "Qwen3-4B": {
        "Base": [88.99, 70.15, 24.54, 40.02, 55.93],
        "GRPO": [90.01, 69.45, 24.91, 39.21, 55.89],
        "Dr.GRPO": [90.07, 71.40, 25.37, 36.65, 55.87],
        "BPR-GRPO": [90.56, 73.05, 25.86, 41.89, 57.84],
    },
}

COLORS = {
    "Base": "#B8C0CC",
    "GRPO": "#6FA8DC",
    "Dr.GRPO": "#93C47D",
    "BPR-GRPO": "#CC4C4C",
}


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float


def hex_to_rgb01(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def esc_pdf(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def pdf_text(x: float, y: float, text: str, size: float = 8, bold: bool = False, color: str = "#111111") -> str:
    r, g, b = hex_to_rgb01(color)
    font = "/F2" if bold else "/F1"
    return f"{r:.3f} {g:.3f} {b:.3f} rg BT {font} {size:.2f} Tf {x:.2f} {y:.2f} Td ({esc_pdf(text)}) Tj ET\n"


def pdf_centered_text(x: float, y: float, text: str, size: float = 8, bold: bool = False, color: str = "#111111") -> str:
    # Helvetica average width is close enough for short labels in this figure.
    est_w = len(text) * size * 0.49
    return pdf_text(x - est_w / 2, y, text, size=size, bold=bold, color=color)


def pdf_rect(rect: Rect, color: str, stroke: str | None = None, width: float = 0.5) -> str:
    r, g, b = hex_to_rgb01(color)
    s = f"{r:.3f} {g:.3f} {b:.3f} rg "
    if stroke:
        sr, sg, sb = hex_to_rgb01(stroke)
        s += f"{sr:.3f} {sg:.3f} {sb:.3f} RG {width:.2f} w "
        s += f"{rect.x:.2f} {rect.y:.2f} {rect.w:.2f} {rect.h:.2f} re B\n"
    else:
        s += f"{rect.x:.2f} {rect.y:.2f} {rect.w:.2f} {rect.h:.2f} re f\n"
    return s


def pdf_line(x1: float, y1: float, x2: float, y2: float, color: str = "#444444", width: float = 0.4) -> str:
    r, g, b = hex_to_rgb01(color)
    return f"{r:.3f} {g:.3f} {b:.3f} RG {width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S\n"


def draw_panel_pdf(x0: float, y0: float, w: float, h: float, model: str) -> str:
    left_pad, right_pad, top_pad, bottom_pad = 34, 8, 26, 34
    plot = Rect(x0 + left_pad, y0 + bottom_pad, w - left_pad - right_pad, h - top_pad - bottom_pad)
    out = []
    out.append(pdf_centered_text(x0 + w / 2, y0 + h - 12, model, size=10, bold=True))

    # Grid and axis labels.
    for tick in range(0, 101, 20):
        yy = plot.y + plot.h * tick / 100.0
        out.append(pdf_line(plot.x, yy, plot.x + plot.w, yy, color="#E3E6EA", width=0.35))
        out.append(pdf_text(x0 + 4, yy - 2.5, str(tick), size=6.8, color="#333333"))
    out.append(pdf_line(plot.x, plot.y, plot.x + plot.w, plot.y, color="#333333", width=0.55))
    out.append(pdf_line(plot.x, plot.y, plot.x, plot.y + plot.h, color="#333333", width=0.55))
    out.append(pdf_text(x0 + 3, y0 + h - 28, "Acc. (%)", size=7.2, color="#333333"))

    group_w = plot.w / len(BENCHMARKS)
    bar_w = group_w * 0.16
    gap = group_w * 0.035
    total_bar_w = len(METHODS) * bar_w + (len(METHODS) - 1) * gap

    for bi, bench in enumerate(BENCHMARKS):
        cx = plot.x + group_w * bi + group_w / 2
        start_x = cx - total_bar_w / 2
        # Match the table annotations: BPR minus the Dr.GRPO baseline.
        bpr = DATA[model]["BPR-GRPO"][bi]
        delta = bpr - DATA[model]["Dr.GRPO"][bi]
        for mi, method in enumerate(METHODS):
            val = DATA[model][method][bi]
            bh = plot.h * val / 100.0
            bx = start_x + mi * (bar_w + gap)
            by = plot.y
            stroke = "#7A1F1F" if method == "BPR-GRPO" else None
            out.append(pdf_rect(Rect(bx, by, bar_w, bh), COLORS[method], stroke=stroke, width=0.35))
        if delta >= 0:
            out.append(pdf_centered_text(cx + group_w * 0.20, plot.y + plot.h * bpr / 100.0 + 4, f"+{delta:.2f}", size=6.4, bold=True, color="#9C2F2F"))
        else:
            out.append(pdf_centered_text(cx + group_w * 0.20, plot.y + plot.h * bpr / 100.0 + 4, f"{delta:.2f}", size=6.4, bold=True, color="#9C2F2F"))
        # Split long benchmark labels.
        if bench == "MATH 500":
            out.append(pdf_centered_text(cx, y0 + 15, "MATH", size=7.0))
            out.append(pdf_centered_text(cx, y0 + 7, "500", size=7.0))
        elif bench == "Olympiad":
            out.append(pdf_centered_text(cx, y0 + 15, "Olympiad", size=7.0))
            out.append(pdf_centered_text(cx, y0 + 7, "Bench", size=7.0))
        else:
            out.append(pdf_centered_text(cx, y0 + 10, bench, size=7.0))

    return "".join(out)


def make_pdf() -> None:
    width, height = 540.0, 285.0
    margin = 18.0
    panel_gap = 18.0
    legend_h = 25.0
    panel_w = (width - 2 * margin - panel_gap) / 2
    panel_h = height - 2 * margin - legend_h

    content = []
    content.append(pdf_text(margin, height - 14, "Controlled local comparison", size=10.5, bold=True))
    content.append(pdf_text(margin + 148, height - 14, "(BPR annotations show improvement over Dr.GRPO, in accuracy points)", size=7.2, color="#444444"))
    content.append(draw_panel_pdf(margin, margin + legend_h, panel_w, panel_h, "Qwen3-1.7B"))
    content.append(draw_panel_pdf(margin + panel_w + panel_gap, margin + legend_h, panel_w, panel_h, "Qwen3-4B"))

    # Legend.
    lx = margin
    ly = 12.0
    for method in METHODS:
        content.append(pdf_rect(Rect(lx, ly, 8.0, 8.0), COLORS[method], stroke="#555555", width=0.25))
        label = method
        if method == "Dr.GRPO":
            label = "Dr.GRPO (baseline)"
        elif method == "BPR-GRPO":
            label = "BPR-GRPO (Ours)"
        content.append(pdf_text(lx + 12.0, ly + 1.0, label, size=7.4))
        lx += 98.0 if method != "BPR-GRPO" else 124.0

    stream = "".join(content).encode("latin-1", errors="replace")
    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.2f} {height:.2f}] "
        f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>".encode()
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream")

    offsets = []
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    for i, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        pdf.extend(f"{off:010d} 00000 n \n".encode())
    pdf.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    PDF_PATH.write_bytes(pdf)


def svg_text(x: float, y: float, text: str, size: float = 10, weight: str = "400", fill: str = "#111111", anchor: str = "start") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size:.2f}" font-family="Arial, Helvetica, sans-serif" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{text}</text>\n'


def svg_rect(rect: Rect, fill: str, stroke: str | None = None, width: float = 0.5) -> str:
    stroke_part = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    return f'<rect x="{rect.x:.2f}" y="{rect.y:.2f}" width="{rect.w:.2f}" height="{rect.h:.2f}" fill="{fill}"{stroke_part}/>\n'


def svg_line(x1: float, y1: float, x2: float, y2: float, stroke: str = "#444444", width: float = 0.4) -> str:
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"/>\n'


def draw_panel_svg(x0: float, y0: float, w: float, h: float, model: str) -> str:
    left_pad, right_pad, top_pad, bottom_pad = 34, 8, 26, 34
    plot = Rect(x0 + left_pad, y0 + top_pad, w - left_pad - right_pad, h - top_pad - bottom_pad)
    out = [svg_text(x0 + w / 2, y0 + 12, model, size=10, weight="700", anchor="middle")]

    for tick in range(0, 101, 20):
        yy = plot.y + plot.h * (1 - tick / 100.0)
        out.append(svg_line(plot.x, yy, plot.x + plot.w, yy, stroke="#E3E6EA", width=0.35))
        out.append(svg_text(x0 + 4, yy + 2.5, str(tick), size=6.8, fill="#333333"))
    out.append(svg_line(plot.x, plot.y + plot.h, plot.x + plot.w, plot.y + plot.h, stroke="#333333", width=0.55))
    out.append(svg_line(plot.x, plot.y, plot.x, plot.y + plot.h, stroke="#333333", width=0.55))
    out.append(svg_text(x0 + 3, y0 + 28, "Acc. (%)", size=7.2, fill="#333333"))

    group_w = plot.w / len(BENCHMARKS)
    bar_w = group_w * 0.16
    gap = group_w * 0.035
    total_bar_w = len(METHODS) * bar_w + (len(METHODS) - 1) * gap
    for bi, bench in enumerate(BENCHMARKS):
        cx = plot.x + group_w * bi + group_w / 2
        start_x = cx - total_bar_w / 2
        bpr = DATA[model]["BPR-GRPO"][bi]
        delta = bpr - DATA[model]["Dr.GRPO"][bi]
        for mi, method in enumerate(METHODS):
            val = DATA[model][method][bi]
            bh = plot.h * val / 100.0
            bx = start_x + mi * (bar_w + gap)
            by = plot.y + plot.h - bh
            stroke = "#7A1F1F" if method == "BPR-GRPO" else None
            out.append(svg_rect(Rect(bx, by, bar_w, bh), COLORS[method], stroke=stroke, width=0.35))
        out.append(svg_text(cx + group_w * 0.20, plot.y + plot.h - plot.h * bpr / 100.0 - 4, f"{delta:+.2f}", size=6.4, weight="700", fill="#9C2F2F", anchor="middle"))
        if bench == "MATH 500":
            out.append(svg_text(cx, y0 + h - 18, "MATH", size=7.0, anchor="middle"))
            out.append(svg_text(cx, y0 + h - 9, "500", size=7.0, anchor="middle"))
        elif bench == "Olympiad":
            out.append(svg_text(cx, y0 + h - 18, "Olympiad", size=7.0, anchor="middle"))
            out.append(svg_text(cx, y0 + h - 9, "Bench", size=7.0, anchor="middle"))
        else:
            out.append(svg_text(cx, y0 + h - 12, bench, size=7.0, anchor="middle"))
    return "".join(out)


def make_svg() -> None:
    width, height = 540.0, 285.0
    margin, panel_gap, legend_h = 18.0, 18.0, 25.0
    panel_w = (width - 2 * margin - panel_gap) / 2
    panel_h = height - 2 * margin - legend_h
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>\n',
        svg_text(margin, 14, "Controlled local comparison", size=10.5, weight="700"),
        svg_text(margin + 148, 14, "(BPR annotations show improvement over Dr.GRPO, in accuracy points)", size=7.2, fill="#444444"),
        draw_panel_svg(margin, margin, panel_w, panel_h, "Qwen3-1.7B"),
        draw_panel_svg(margin + panel_w + panel_gap, margin, panel_w, panel_h, "Qwen3-4B"),
    ]
    lx, ly = margin, height - 16.0
    for method in METHODS:
        out.append(svg_rect(Rect(lx, ly - 8.0, 8.0, 8.0), COLORS[method], stroke="#555555", width=0.25))
        label = method
        if method == "Dr.GRPO":
            label = "Dr.GRPO (baseline)"
        elif method == "BPR-GRPO":
            label = "BPR-GRPO (Ours)"
        out.append(svg_text(lx + 12.0, ly - 1.0, label, size=7.4))
        lx += 98.0 if method != "BPR-GRPO" else 124.0
    out.append("</svg>\n")
    SVG_PATH.write_text("".join(out), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_pdf()
    make_svg()
    print(f"Wrote {PDF_PATH}")
    print(f"Wrote {SVG_PATH}")


if __name__ == "__main__":
    main()
