#!/usr/bin/env python3
"""Create a polished publication-style bar plot for the main result table.

The script uses only the Python standard library and emits vector PDF + SVG.
The design is intentionally quiet: muted baselines, one emphasized method, thin
rules, direct panel labels, and compact delta annotations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


OUT_DIR = Path("outputs/paper_figures")
PDF_PATH = OUT_DIR / "main_results_barplot_publication.pdf"
SVG_PATH = OUT_DIR / "main_results_barplot_publication.svg"

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
    "Base": "#C4CAD3",
    "GRPO": "#7EA6CF",
    "Dr.GRPO": "#8FB996",
    "BPR-GRPO": "#B64242",
}
TEXT = "#222222"
MUTED = "#666D75"
GRID = "#E7E9ED"


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


def text_width(text: str, size: float, bold: bool = False) -> float:
    # Good enough for Helvetica labels used here.
    factor = 0.52 if bold else 0.49
    return len(text) * size * factor


def pdf_text(x: float, y: float, text: str, size: float = 8, bold: bool = False, color: str = TEXT) -> str:
    r, g, b = hex_to_rgb01(color)
    font = "/F2" if bold else "/F1"
    return f"{r:.3f} {g:.3f} {b:.3f} rg BT {font} {size:.2f} Tf {x:.2f} {y:.2f} Td ({esc_pdf(text)}) Tj ET\n"


def pdf_center(x: float, y: float, text: str, size: float = 8, bold: bool = False, color: str = TEXT) -> str:
    return pdf_text(x - text_width(text, size, bold) / 2, y, text, size, bold, color)


def pdf_rect(rect: Rect, fill: str, stroke: str | None = None, width: float = 0.4) -> str:
    r, g, b = hex_to_rgb01(fill)
    cmd = f"{r:.3f} {g:.3f} {b:.3f} rg "
    if stroke:
        sr, sg, sb = hex_to_rgb01(stroke)
        cmd += f"{sr:.3f} {sg:.3f} {sb:.3f} RG {width:.2f} w {rect.x:.2f} {rect.y:.2f} {rect.w:.2f} {rect.h:.2f} re B\n"
    else:
        cmd += f"{rect.x:.2f} {rect.y:.2f} {rect.w:.2f} {rect.h:.2f} re f\n"
    return cmd


def pdf_line(x1: float, y1: float, x2: float, y2: float, color: str = GRID, width: float = 0.35) -> str:
    r, g, b = hex_to_rgb01(color)
    return f"{r:.3f} {g:.3f} {b:.3f} RG {width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S\n"


def draw_panel_pdf(x0: float, y0: float, w: float, h: float, model: str) -> str:
    left, right, top, bottom = 34.0, 7.0, 26.0, 33.0
    plot = Rect(x0 + left, y0 + bottom, w - left - right, h - top - bottom)
    out: list[str] = []

    out.append(pdf_text(x0 + 1, y0 + h - 10, model, size=10.2, bold=True, color=TEXT))
    out.append(pdf_text(x0 + 1, y0 + h - 21, "deterministic pass@1 accuracy", size=6.8, color=MUTED))

    # Minimal y-axis with four gridlines.
    for tick in [0, 25, 50, 75, 100]:
        yy = plot.y + plot.h * tick / 100.0
        out.append(pdf_line(plot.x, yy, plot.x + plot.w, yy, GRID, 0.32))
        out.append(pdf_text(x0 + 11, yy - 2.0, str(tick), size=6.6, color=MUTED))
    out.append(pdf_line(plot.x, plot.y, plot.x + plot.w, plot.y, "#333333", 0.55))
    out.append(pdf_line(plot.x, plot.y, plot.x, plot.y + plot.h, "#333333", 0.55))

    group_w = plot.w / len(BENCHMARKS)
    bar_w = group_w * 0.145
    gap = group_w * 0.042
    total_w = len(METHODS) * bar_w + (len(METHODS) - 1) * gap
    for i, bench in enumerate(BENCHMARKS):
        cx = plot.x + group_w * i + group_w / 2
        start = cx - total_w / 2
        if bench == "Avg.":
            # Subtle divider before the macro-average column.
            out.append(pdf_line(plot.x + group_w * i + 1.0, plot.y, plot.x + group_w * i + 1.0, plot.y + plot.h, "#D5D8DD", 0.45))

        bpr = DATA[model]["BPR-GRPO"][i]
        delta = bpr - DATA[model]["Dr.GRPO"][i]
        for j, method in enumerate(METHODS):
            val = DATA[model][method][i]
            bh = plot.h * val / 100.0
            bx = start + j * (bar_w + gap)
            stroke = "#7B2727" if method == "BPR-GRPO" else None
            out.append(pdf_rect(Rect(bx, plot.y, bar_w, bh), COLORS[method], stroke=stroke, width=0.35))

        # Compact delta annotation above BPR.
        bpr_x = start + 3 * (bar_w + gap) + bar_w / 2
        bpr_y = plot.y + plot.h * bpr / 100.0
        out.append(pdf_center(bpr_x, min(plot.y + plot.h + 2, bpr_y + 5.5), f"+{delta:.2f}", size=6.3, bold=True, color="#9D2F2F"))

        # Benchmark labels.
        if bench == "MATH 500":
            out.append(pdf_center(cx, y0 + 15.0, "MATH", size=6.8, color=TEXT))
            out.append(pdf_center(cx, y0 + 7.5, "500", size=6.8, color=TEXT))
        elif bench == "Olympiad":
            out.append(pdf_center(cx, y0 + 15.0, "Olympiad", size=6.8, color=TEXT))
            out.append(pdf_center(cx, y0 + 7.5, "Bench", size=6.8, color=TEXT))
        else:
            out.append(pdf_center(cx, y0 + 10.0, bench, size=6.8, color=TEXT))

    return "".join(out)


def make_pdf() -> None:
    width, height = 520.0, 265.0
    margin = 16.0
    legend_h = 24.0
    panel_gap = 16.0
    panel_w = (width - margin * 2 - panel_gap) / 2
    panel_h = height - margin * 2 - legend_h

    content: list[str] = []
    content.append(draw_panel_pdf(margin, margin + legend_h, panel_w, panel_h, "Qwen3-1.7B"))
    content.append(draw_panel_pdf(margin + panel_w + panel_gap, margin + legend_h, panel_w, panel_h, "Qwen3-4B"))

    # Legend centered at the bottom.
    legend = [
        ("Base", "Base"),
        ("GRPO", "GRPO"),
        ("Dr.GRPO", "Dr.GRPO baseline"),
        ("BPR-GRPO", "BPR-GRPO (ours)"),
    ]
    x = 82.0
    y = 12.0
    for method, label in legend:
        content.append(pdf_rect(Rect(x, y, 8.0, 8.0), COLORS[method], stroke="#555555", width=0.25))
        content.append(pdf_text(x + 12.0, y + 1.1, label, size=7.2, color=TEXT))
        x += 87.0 if method in {"Base", "GRPO"} else 121.0
    content.append(pdf_text(350.0, 22.4, r"red labels: BPR $-$ Dr.GRPO", size=6.6, color=MUTED))

    stream = "".join(content).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.2f} {height:.2f}] "
        f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for idx, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        pdf.extend(f"{off:010d} 00000 n \n".encode())
    pdf.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    PDF_PATH.write_bytes(pdf)


def svg_text(x: float, y: float, text: str, size: float = 8, weight: str = "400", fill: str = TEXT, anchor: str = "start") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" font-size="{size:.2f}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{text}</text>\n'


def svg_rect(rect: Rect, fill: str, stroke: str | None = None, width: float = 0.4) -> str:
    stroke_part = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    return f'<rect x="{rect.x:.2f}" y="{rect.y:.2f}" width="{rect.w:.2f}" height="{rect.h:.2f}" fill="{fill}"{stroke_part}/>\n'


def svg_line(x1: float, y1: float, x2: float, y2: float, color: str = GRID, width: float = 0.35) -> str:
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"/>\n'


def draw_panel_svg(x0: float, y0: float, w: float, h: float, model: str) -> str:
    left, right, top, bottom = 34.0, 7.0, 26.0, 33.0
    plot = Rect(x0 + left, y0 + top, w - left - right, h - top - bottom)
    out: list[str] = []
    out.append(svg_text(x0 + 1, y0 + 10, model, 10.2, "700", TEXT))
    out.append(svg_text(x0 + 1, y0 + 21, "deterministic pass@1 accuracy", 6.8, "400", MUTED))
    for tick in [0, 25, 50, 75, 100]:
        yy = plot.y + plot.h * (1 - tick / 100.0)
        out.append(svg_line(plot.x, yy, plot.x + plot.w, yy, GRID, 0.32))
        out.append(svg_text(x0 + 11, yy + 2.4, str(tick), 6.6, "400", MUTED))
    out.append(svg_line(plot.x, plot.y + plot.h, plot.x + plot.w, plot.y + plot.h, "#333333", 0.55))
    out.append(svg_line(plot.x, plot.y, plot.x, plot.y + plot.h, "#333333", 0.55))

    group_w = plot.w / len(BENCHMARKS)
    bar_w = group_w * 0.145
    gap = group_w * 0.042
    total_w = len(METHODS) * bar_w + (len(METHODS) - 1) * gap
    for i, bench in enumerate(BENCHMARKS):
        cx = plot.x + group_w * i + group_w / 2
        start = cx - total_w / 2
        if bench == "Avg.":
            out.append(svg_line(plot.x + group_w * i + 1.0, plot.y, plot.x + group_w * i + 1.0, plot.y + plot.h, "#D5D8DD", 0.45))
        bpr = DATA[model]["BPR-GRPO"][i]
        delta = bpr - DATA[model]["Dr.GRPO"][i]
        for j, method in enumerate(METHODS):
            val = DATA[model][method][i]
            bh = plot.h * val / 100.0
            bx = start + j * (bar_w + gap)
            by = plot.y + plot.h - bh
            stroke = "#7B2727" if method == "BPR-GRPO" else None
            out.append(svg_rect(Rect(bx, by, bar_w, bh), COLORS[method], stroke, 0.35))
        bpr_x = start + 3 * (bar_w + gap) + bar_w / 2
        bpr_y = plot.y + plot.h - plot.h * bpr / 100.0
        out.append(svg_text(bpr_x, max(plot.y + 5, bpr_y - 5.0), f"+{delta:.2f}", 6.3, "700", "#9D2F2F", "middle"))
        if bench == "MATH 500":
            out.append(svg_text(cx, y0 + h - 18.0, "MATH", 6.8, "400", TEXT, "middle"))
            out.append(svg_text(cx, y0 + h - 9.5, "500", 6.8, "400", TEXT, "middle"))
        elif bench == "Olympiad":
            out.append(svg_text(cx, y0 + h - 18.0, "Olympiad", 6.8, "400", TEXT, "middle"))
            out.append(svg_text(cx, y0 + h - 9.5, "Bench", 6.8, "400", TEXT, "middle"))
        else:
            out.append(svg_text(cx, y0 + h - 12.0, bench, 6.8, "400", TEXT, "middle"))
    return "".join(out)


def make_svg() -> None:
    width, height = 520.0, 265.0
    margin = 16.0
    legend_h = 24.0
    panel_gap = 16.0
    panel_w = (width - margin * 2 - panel_gap) / 2
    panel_h = height - margin * 2 - legend_h
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>\n',
        draw_panel_svg(margin, margin, panel_w, panel_h, "Qwen3-1.7B"),
        draw_panel_svg(margin + panel_w + panel_gap, margin, panel_w, panel_h, "Qwen3-4B"),
    ]
    legend = [
        ("Base", "Base"),
        ("GRPO", "GRPO"),
        ("Dr.GRPO", "Dr.GRPO baseline"),
        ("BPR-GRPO", "BPR-GRPO (ours)"),
    ]
    x, y = 82.0, height - 12.0
    for method, label in legend:
        out.append(svg_rect(Rect(x, y - 8.0, 8.0, 8.0), COLORS[method], "#555555", 0.25))
        out.append(svg_text(x + 12.0, y - 1.0, label, 7.2, "400", TEXT))
        x += 87.0 if method in {"Base", "GRPO"} else 121.0
    out.append(svg_text(350.0, 22.4, "red labels: BPR - Dr.GRPO", 6.6, "400", MUTED))
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
