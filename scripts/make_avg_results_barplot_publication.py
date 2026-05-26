#!/usr/bin/env python3
"""Create a compact publication-style bar plot for macro-average accuracy.

The figure is designed as a simple visual summary for the main paper: two model
groups, four methods, and direct BPR-vs-Dr.GRPO delta annotations.
It uses only the Python standard library and writes vector PDF + SVG.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


OUT_DIR = Path("outputs/paper_figures")
PDF_PATH = OUT_DIR / "main_avg_barplot_publication.pdf"
SVG_PATH = OUT_DIR / "main_avg_barplot_publication.svg"

MODELS = ["Qwen3-1.7B", "Qwen3-4B"]
METHODS = ["Base", "GRPO", "Dr.GRPO", "BPR-GRPO"]
AVG = {
    "Qwen3-1.7B": {"Base": 43.99, "GRPO": 45.00, "Dr.GRPO": 44.68, "BPR-GRPO": 46.55},
    "Qwen3-4B": {"Base": 55.93, "GRPO": 55.89, "Dr.GRPO": 55.87, "BPR-GRPO": 57.84},
}

COLORS = {
    "Base": "#C6CBD3",
    "GRPO": "#7EA6CF",
    "Dr.GRPO": "#8FB996",
    "BPR-GRPO": "#B64242",
}
TEXT = "#202326"
MUTED = "#626A73"
GRID = "#E6E8EC"


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float


def rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def approx_width(text: str, size: float, bold: bool = False) -> float:
    return len(text) * size * (0.52 if bold else 0.49)


def pdf_text(x: float, y: float, text: str, size: float = 8, bold: bool = False, color: str = TEXT) -> str:
    r, g, b = rgb(color)
    font = "/F2" if bold else "/F1"
    return f"{r:.3f} {g:.3f} {b:.3f} rg BT {font} {size:.2f} Tf {x:.2f} {y:.2f} Td ({esc(text)}) Tj ET\n"


def pdf_center(x: float, y: float, text: str, size: float = 8, bold: bool = False, color: str = TEXT) -> str:
    return pdf_text(x - approx_width(text, size, bold) / 2, y, text, size, bold, color)


def pdf_rect(rect: Rect, fill: str, stroke: str | None = None, width: float = 0.4) -> str:
    r, g, b = rgb(fill)
    out = f"{r:.3f} {g:.3f} {b:.3f} rg "
    if stroke:
        sr, sg, sb = rgb(stroke)
        out += f"{sr:.3f} {sg:.3f} {sb:.3f} RG {width:.2f} w {rect.x:.2f} {rect.y:.2f} {rect.w:.2f} {rect.h:.2f} re B\n"
    else:
        out += f"{rect.x:.2f} {rect.y:.2f} {rect.w:.2f} {rect.h:.2f} re f\n"
    return out


def pdf_line(x1: float, y1: float, x2: float, y2: float, color: str = GRID, width: float = 0.35) -> str:
    r, g, b = rgb(color)
    return f"{r:.3f} {g:.3f} {b:.3f} RG {width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S\n"


def draw_pdf() -> bytes:
    width, height = 380.0, 230.0
    plot = Rect(47.0, 47.0, 305.0, 138.0)
    content: list[str] = []
    content.append(pdf_text(18, 213, "Macro-average accuracy", 11.0, True))
    content.append(pdf_text(18, 201, "Unweighted average over GSM8K, MATH 500, MinervaMath, and OlympiadBench", 7.0, False, MUTED))

    # Axis.
    for tick in [40, 45, 50, 55, 60]:
        y = plot.y + plot.h * (tick - 40) / 20.0
        content.append(pdf_line(plot.x, y, plot.x + plot.w, y, GRID, 0.35))
        content.append(pdf_text(25, y - 2.2, str(tick), 6.8, False, MUTED))
    content.append(pdf_line(plot.x, plot.y, plot.x + plot.w, plot.y, "#333333", 0.55))
    content.append(pdf_line(plot.x, plot.y, plot.x, plot.y + plot.h, "#333333", 0.55))

    group_w = plot.w / len(MODELS)
    bar_w = 17.0
    gap = 7.0
    total = len(METHODS) * bar_w + (len(METHODS) - 1) * gap
    scale_min, scale_max = 40.0, 60.0
    for gi, model in enumerate(MODELS):
        cx = plot.x + group_w * gi + group_w / 2
        start = cx - total / 2
        for mi, method in enumerate(METHODS):
            val = AVG[model][method]
            h = plot.h * (val - scale_min) / (scale_max - scale_min)
            x = start + mi * (bar_w + gap)
            stroke = "#762626" if method == "BPR-GRPO" else None
            content.append(pdf_rect(Rect(x, plot.y, bar_w, h), COLORS[method], stroke, 0.4))
            # Label the exact avg above BPR only.
            if method == "BPR-GRPO":
                content.append(pdf_center(x + bar_w / 2, plot.y + h + 4.5, f"{val:.2f}", 7.0, True, "#8E2929"))
        delta = AVG[model]["BPR-GRPO"] - AVG[model]["Dr.GRPO"]
        content.append(pdf_center(cx, plot.y + plot.h + 11.0, f"BPR vs Dr.GRPO: +{delta:.2f}", 7.2, True, "#8E2929"))
        content.append(pdf_center(cx, 29.0, model, 8.6, True))

    # Legend.
    x, y = 57.0, 12.0
    labels = {"Base": "Base", "GRPO": "GRPO", "Dr.GRPO": "Dr.GRPO", "BPR-GRPO": "BPR-GRPO"}
    for method in METHODS:
        content.append(pdf_rect(Rect(x, y, 8.5, 8.5), COLORS[method], "#555555", 0.25))
        content.append(pdf_text(x + 12, y + 1.2, labels[method], 7.2))
        x += 72 if method != "BPR-GRPO" else 88

    stream = "".join(content).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.2f} {height:.2f}] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
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
    return bytes(pdf)


def svg_text(x: float, y: float, text: str, size: float = 8, weight: str = "400", fill: str = TEXT, anchor: str = "start") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" font-size="{size:.2f}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{text}</text>\n'


def svg_rect(rect: Rect, fill: str, stroke: str | None = None, width: float = 0.4) -> str:
    s = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    return f'<rect x="{rect.x:.2f}" y="{rect.y:.2f}" width="{rect.w:.2f}" height="{rect.h:.2f}" fill="{fill}"{s}/>\n'


def svg_line(x1: float, y1: float, x2: float, y2: float, color: str = GRID, width: float = 0.35) -> str:
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"/>\n'


def draw_svg() -> str:
    width, height = 380.0, 230.0
    plot = Rect(47.0, 45.0, 305.0, 138.0)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>\n',
        svg_text(18, 17, "Macro-average accuracy", 11.0, "700"),
        svg_text(18, 29, "Unweighted average over GSM8K, MATH 500, MinervaMath, and OlympiadBench", 7.0, "400", MUTED),
    ]
    for tick in [40, 45, 50, 55, 60]:
        y = plot.y + plot.h * (1 - (tick - 40) / 20.0)
        out.append(svg_line(plot.x, y, plot.x + plot.w, y, GRID, 0.35))
        out.append(svg_text(25, y + 2.4, str(tick), 6.8, "400", MUTED))
    out.append(svg_line(plot.x, plot.y + plot.h, plot.x + plot.w, plot.y + plot.h, "#333333", 0.55))
    out.append(svg_line(plot.x, plot.y, plot.x, plot.y + plot.h, "#333333", 0.55))

    group_w = plot.w / len(MODELS)
    bar_w, gap = 17.0, 7.0
    total = len(METHODS) * bar_w + (len(METHODS) - 1) * gap
    scale_min, scale_max = 40.0, 60.0
    for gi, model in enumerate(MODELS):
        cx = plot.x + group_w * gi + group_w / 2
        start = cx - total / 2
        for mi, method in enumerate(METHODS):
            val = AVG[model][method]
            h = plot.h * (val - scale_min) / (scale_max - scale_min)
            x = start + mi * (bar_w + gap)
            y = plot.y + plot.h - h
            stroke = "#762626" if method == "BPR-GRPO" else None
            out.append(svg_rect(Rect(x, y, bar_w, h), COLORS[method], stroke, 0.4))
            if method == "BPR-GRPO":
                out.append(svg_text(x + bar_w / 2, y - 4.5, f"{val:.2f}", 7.0, "700", "#8E2929", "middle"))
        delta = AVG[model]["BPR-GRPO"] - AVG[model]["Dr.GRPO"]
        out.append(svg_text(cx, plot.y - 11.0, f"BPR vs Dr.GRPO: +{delta:.2f}", 7.2, "700", "#8E2929", "middle"))
        out.append(svg_text(cx, 201.0, model, 8.6, "700", TEXT, "middle"))

    x, y = 57.0, 217.0
    labels = {"Base": "Base", "GRPO": "GRPO", "Dr.GRPO": "Dr.GRPO", "BPR-GRPO": "BPR-GRPO"}
    for method in METHODS:
        out.append(svg_rect(Rect(x, y - 8.5, 8.5, 8.5), COLORS[method], "#555555", 0.25))
        out.append(svg_text(x + 12, y - 1.0, labels[method], 7.2))
        x += 72 if method != "BPR-GRPO" else 88
    out.append("</svg>\n")
    return "".join(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_PATH.write_bytes(draw_pdf())
    SVG_PATH.write_text(draw_svg(), encoding="utf-8")
    print(f"Wrote {PDF_PATH}")
    print(f"Wrote {SVG_PATH}")


if __name__ == "__main__":
    main()
