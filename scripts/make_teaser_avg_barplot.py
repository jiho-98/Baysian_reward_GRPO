#!/usr/bin/env python3
"""Small first-page teaser bar plot for macro-average accuracy.

Designed to resemble compact ML paper teaser figures: two small panels, direct
method labels, value labels, no legend, tight margins. Uses only the Python
standard library and emits vector PDF + SVG.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


OUT_DIR = Path("outputs/paper_figures")
PDF_PATH = OUT_DIR / "main_avg_teaser_barplot.pdf"
SVG_PATH = OUT_DIR / "main_avg_teaser_barplot.svg"

MODELS = ["Qwen3-1.7B", "Qwen3-4B"]
METHODS = ["Base", "GRPO", "Dr.GRPO", "BPR"]
VALUES = {
    "Qwen3-1.7B": [43.99, 45.00, 44.68, 46.55],
    "Qwen3-4B": [55.93, 55.89, 55.87, 57.84],
}
FULL_LABELS = {
    "Base": "Base",
    "GRPO": "GRPO",
    "Dr.GRPO": "Dr.GRPO",
    "BPR": "BPR-GRPO",
}

COLORS = ["#AEBBD3", "#F0A58E", "#91D2B4", "#D98DB9"]
TEXT = "#222222"
MUTED = "#666666"
GRID = "#E8E8E8"


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float


def rgb(c: str) -> tuple[float, float, float]:
    c = c.lstrip("#")
    return tuple(int(c[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def tw(s: str, size: float, bold: bool = False) -> float:
    return len(s) * size * (0.52 if bold else 0.49)


def pdf_text(x: float, y: float, s: str, size: float = 7, bold: bool = False, color: str = TEXT) -> str:
    r, g, b = rgb(color)
    font = "/F2" if bold else "/F1"
    return f"{r:.3f} {g:.3f} {b:.3f} rg BT {font} {size:.2f} Tf {x:.2f} {y:.2f} Td ({esc(s)}) Tj ET\n"


def pdf_center(x: float, y: float, s: str, size: float = 7, bold: bool = False, color: str = TEXT) -> str:
    return pdf_text(x - tw(s, size, bold) / 2, y, s, size, bold, color)


def pdf_rect(rct: Rect, fill: str, stroke: str | None = None, width: float = 0.35) -> str:
    r, g, b = rgb(fill)
    out = f"{r:.3f} {g:.3f} {b:.3f} rg "
    if stroke:
        sr, sg, sb = rgb(stroke)
        out += f"{sr:.3f} {sg:.3f} {sb:.3f} RG {width:.2f} w {rct.x:.2f} {rct.y:.2f} {rct.w:.2f} {rct.h:.2f} re B\n"
    else:
        out += f"{rct.x:.2f} {rct.y:.2f} {rct.w:.2f} {rct.h:.2f} re f\n"
    return out


def pdf_line(x1: float, y1: float, x2: float, y2: float, color: str = GRID, width: float = 0.3) -> str:
    r, g, b = rgb(color)
    return f"{r:.3f} {g:.3f} {b:.3f} RG {width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S\n"


def panel_pdf(x0: float, y0: float, w: float, h: float, model: str, ymin: float, ymax: float, ticks: list[int]) -> str:
    left, right, top, bottom = 25.0, 5.0, 14.0, 16.0
    plot = Rect(x0 + left, y0 + bottom, w - left - right, h - top - bottom)
    vals = VALUES[model]
    out: list[str] = []
    out.append(pdf_center(x0 + w / 2, y0 + h - 7.0, f"Average Accuracy on {model}", 6.7, False, TEXT))

    for t in ticks:
        yy = plot.y + plot.h * (t - ymin) / (ymax - ymin)
        out.append(pdf_line(plot.x, yy, plot.x + plot.w, yy, GRID, 0.25))
        out.append(pdf_text(x0 + 4.0, yy - 2.0, str(t), 5.6, False, MUTED))
    out.append(pdf_line(plot.x, plot.y, plot.x + plot.w, plot.y, "#333333", 0.45))
    out.append(pdf_line(plot.x, plot.y, plot.x, plot.y + plot.h, "#333333", 0.45))

    bar_w = plot.w / 6.1
    gap = bar_w * 0.42
    total = 4 * bar_w + 3 * gap
    start = plot.x + plot.w / 2 - total / 2
    for i, (method, val) in enumerate(zip(METHODS, vals)):
        bh = plot.h * (val - ymin) / (ymax - ymin)
        bx = start + i * (bar_w + gap)
        stroke = "#8B2B63" if method == "BPR" else None
        out.append(pdf_rect(Rect(bx, plot.y, bar_w, bh), COLORS[i], stroke, 0.3))
        out.append(pdf_center(bx + bar_w / 2, plot.y + bh + 3.0, f"{val:.2f}", 5.4, False, TEXT))
        out.append(pdf_center(bx + bar_w / 2, y0 + 4.8, method, 5.8, False, TEXT))
    return "".join(out)


def make_pdf() -> None:
    width, height = 292.0, 112.0
    content: list[str] = []
    content.append(panel_pdf(0.0, 0.0, 144.0, height, "Qwen3-1.7B", 42.0, 48.0, [42, 44, 46, 48]))
    content.append(panel_pdf(148.0, 0.0, 144.0, height, "Qwen3-4B", 54.0, 59.0, [54, 56, 58]))
    stream = "".join(content).encode("latin-1", errors="replace")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.2f} {height:.2f}] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        pdf.extend(f"{off:010d} 00000 n \n".encode())
    pdf.extend(f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    PDF_PATH.write_bytes(pdf)


def svg_text(x: float, y: float, s: str, size: float = 7, weight: str = "400", fill: str = TEXT, anchor: str = "start") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" font-size="{size:.2f}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{s}</text>\n'


def svg_rect(rct: Rect, fill: str, stroke: str | None = None, width: float = 0.35) -> str:
    sp = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    return f'<rect x="{rct.x:.2f}" y="{rct.y:.2f}" width="{rct.w:.2f}" height="{rct.h:.2f}" fill="{fill}"{sp}/>\n'


def svg_line(x1: float, y1: float, x2: float, y2: float, color: str = GRID, width: float = 0.3) -> str:
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"/>\n'


def panel_svg(x0: float, y0: float, w: float, h: float, model: str, ymin: float, ymax: float, ticks: list[int]) -> str:
    left, right, top, bottom = 25.0, 5.0, 14.0, 16.0
    plot = Rect(x0 + left, y0 + top, w - left - right, h - top - bottom)
    out = [svg_text(x0 + w / 2, y0 + 8.0, f"Average Accuracy on {model}", 6.7, "400", TEXT, "middle")]
    for t in ticks:
        yy = plot.y + plot.h * (1 - (t - ymin) / (ymax - ymin))
        out.append(svg_line(plot.x, yy, plot.x + plot.w, yy, GRID, 0.25))
        out.append(svg_text(x0 + 4.0, yy + 2.0, str(t), 5.6, "400", MUTED))
    out.append(svg_line(plot.x, plot.y + plot.h, plot.x + plot.w, plot.y + plot.h, "#333333", 0.45))
    out.append(svg_line(plot.x, plot.y, plot.x, plot.y + plot.h, "#333333", 0.45))
    bar_w = plot.w / 6.1
    gap = bar_w * 0.42
    total = 4 * bar_w + 3 * gap
    start = plot.x + plot.w / 2 - total / 2
    for i, (method, val) in enumerate(zip(METHODS, VALUES[model])):
        bh = plot.h * (val - ymin) / (ymax - ymin)
        bx = start + i * (bar_w + gap)
        by = plot.y + plot.h - bh
        stroke = "#8B2B63" if method == "BPR" else None
        out.append(svg_rect(Rect(bx, by, bar_w, bh), COLORS[i], stroke, 0.3))
        out.append(svg_text(bx + bar_w / 2, by - 3.0, f"{val:.2f}", 5.4, "400", TEXT, "middle"))
        out.append(svg_text(bx + bar_w / 2, y0 + h - 5.0, method, 5.8, "400", TEXT, "middle"))
    return "".join(out)


def make_svg() -> None:
    width, height = 292.0, 112.0
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>\n',
        panel_svg(0.0, 0.0, 144.0, height, "Qwen3-1.7B", 42.0, 48.0, [42, 44, 46, 48]),
        panel_svg(148.0, 0.0, 144.0, height, "Qwen3-4B", 54.0, 59.0, [54, 56, 58]),
        "</svg>\n",
    ]
    SVG_PATH.write_text("".join(svg), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_pdf()
    make_svg()
    print(f"Wrote {PDF_PATH}")
    print(f"Wrote {SVG_PATH}")
    for method in METHODS:
        print(f"{method}: {FULL_LABELS[method]}")


if __name__ == "__main__":
    main()
