#!/usr/bin/env python3
"""Create a compact offline prior-strength sensitivity plot.

The plot is based on stored BPR reward-debug artifacts already summarized in the
experiment notes. It is an offline reward-allocation analysis, not downstream
accuracy. Uses only the Python standard library and emits vector PDF + SVG.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


OUT_DIR = Path("outputs/paper_figures")
PDF_PATH = OUT_DIR / "prior_lambda_sensitivity.pdf"
SVG_PATH = OUT_DIR / "prior_lambda_sensitivity.svg"

LAMBDAS = [0.0, 0.5, 0.7, 1.0, 1.5, 2.0]
CORRECT_MASS = [0.7811, 0.7807, 0.7803, 0.7794, 0.7775, 0.7752]
TOP1 = [1.0000, 1.0000, 0.9994, 0.9953, 0.9943, 0.9646]
ENTROPY = [0.9082, 0.9012, 0.8951, 0.8832, 0.8580, 0.8292]

SERIES = [
    ("Correct reward mass", CORRECT_MASS, "#4C78A8"),
    ("Top-1 correct", TOP1, "#59A14F"),
    ("Entropy", ENTROPY, "#B64242"),
]

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


def text_width(s: str, size: float, bold: bool = False) -> float:
    return len(s) * size * (0.52 if bold else 0.49)


def pdf_text(x: float, y: float, s: str, size: float = 7, bold: bool = False, color: str = TEXT) -> str:
    r, g, b = rgb(color)
    font = "/F2" if bold else "/F1"
    return f"{r:.3f} {g:.3f} {b:.3f} rg BT {font} {size:.2f} Tf {x:.2f} {y:.2f} Td ({esc(s)}) Tj ET\n"


def pdf_center(x: float, y: float, s: str, size: float = 7, bold: bool = False, color: str = TEXT) -> str:
    return pdf_text(x - text_width(s, size, bold) / 2, y, s, size, bold, color)


def pdf_line(x1: float, y1: float, x2: float, y2: float, color: str = GRID, width: float = 0.4) -> str:
    r, g, b = rgb(color)
    return f"{r:.3f} {g:.3f} {b:.3f} RG {width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S\n"


def pdf_rect(r: Rect, color: str) -> str:
    cr, cg, cb = rgb(color)
    return f"{cr:.3f} {cg:.3f} {cb:.3f} rg {r.x:.2f} {r.y:.2f} {r.w:.2f} {r.h:.2f} re f\n"


def map_xy(plot: Rect, x: float, y: float) -> tuple[float, float]:
    x_min, x_max = 0.0, 2.0
    y_min, y_max = 0.75, 1.01
    px = plot.x + plot.w * (x - x_min) / (x_max - x_min)
    py = plot.y + plot.h * (y - y_min) / (y_max - y_min)
    return px, py


def path_for(plot: Rect, values: list[float], color: str) -> str:
    pts = [map_xy(plot, x, y) for x, y in zip(LAMBDAS, values)]
    r, g, b = rgb(color)
    out = f"{r:.3f} {g:.3f} {b:.3f} RG 1.15 w "
    x0, y0 = pts[0]
    out += f"{x0:.2f} {y0:.2f} m "
    for x, y in pts[1:]:
        out += f"{x:.2f} {y:.2f} l "
    out += "S\n"
    for x, y in pts:
        out += pdf_rect(Rect(x - 1.5, y - 1.5, 3.0, 3.0), color)
    return out


def make_pdf() -> None:
    width, height = 255.0, 150.0
    plot = Rect(34.0, 33.0, 165.0, 84.0)
    content: list[str] = []
    content.append(pdf_text(16, 136, "Offline prior-strength sensitivity", 9.3, True))
    content.append(pdf_text(16, 125, "reward allocation only; no retraining", 6.6, False, MUTED))
    for y in [0.75, 0.85, 0.95, 1.00]:
        px1, py = map_xy(plot, 0.0, y)
        px2, _ = map_xy(plot, 2.0, y)
        content.append(pdf_line(px1, py, px2, py, GRID, 0.3))
        content.append(pdf_text(13, py - 2.0, f"{y:.2f}", 5.8, False, MUTED))
    for x in [0.0, 0.5, 1.0, 1.5, 2.0]:
        px, _ = map_xy(plot, x, 0.75)
        content.append(pdf_center(px, 20.5, f"{x:g}", 6.0, False, MUTED))
    content.append(pdf_line(plot.x, plot.y, plot.x + plot.w, plot.y, "#333333", 0.5))
    content.append(pdf_line(plot.x, plot.y, plot.x, plot.y + plot.h, "#333333", 0.5))
    content.append(pdf_center(plot.x + plot.w / 2, 9.5, "prior strength lambda", 6.4, False, TEXT))
    for _, vals, color in SERIES:
        content.append(path_for(plot, vals, color))
    # Direct labels at the right.
    lx = 207.0
    for label, vals, color in SERIES:
        _, y = map_xy(plot, 2.0, vals[-1])
        content.append(pdf_rect(Rect(lx, y - 2.0, 4.0, 4.0), color))
        content.append(pdf_text(lx + 7.0, y - 2.2, label, 5.8, False, TEXT))

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


def svg_line(x1: float, y1: float, x2: float, y2: float, color: str = GRID, width: float = 0.4) -> str:
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"/>\n'


def svg_rect(r: Rect, color: str) -> str:
    return f'<rect x="{r.x:.2f}" y="{r.y:.2f}" width="{r.w:.2f}" height="{r.h:.2f}" fill="{color}"/>\n'


def map_xy_svg(plot: Rect, x: float, y: float) -> tuple[float, float]:
    x_min, x_max = 0.0, 2.0
    y_min, y_max = 0.75, 1.01
    px = plot.x + plot.w * (x - x_min) / (x_max - x_min)
    py = plot.y + plot.h * (1 - (y - y_min) / (y_max - y_min))
    return px, py


def make_svg() -> None:
    width, height = 255.0, 150.0
    plot = Rect(34.0, 33.0, 165.0, 84.0)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>\n',
        svg_text(16, 14, "Offline prior-strength sensitivity", 9.3, "700"),
        svg_text(16, 25, "reward allocation only; no retraining", 6.6, "400", MUTED),
    ]
    for yv in [0.75, 0.85, 0.95, 1.00]:
        x1, y = map_xy_svg(plot, 0.0, yv)
        x2, _ = map_xy_svg(plot, 2.0, yv)
        out.append(svg_line(x1, y, x2, y, GRID, 0.3))
        out.append(svg_text(13, y + 2.0, f"{yv:.2f}", 5.8, "400", MUTED))
    for xv in [0.0, 0.5, 1.0, 1.5, 2.0]:
        x, _ = map_xy_svg(plot, xv, 0.75)
        out.append(svg_text(x, 130.0, f"{xv:g}", 6.0, "400", MUTED, "middle"))
    out.append(svg_line(plot.x, plot.y + plot.h, plot.x + plot.w, plot.y + plot.h, "#333333", 0.5))
    out.append(svg_line(plot.x, plot.y, plot.x, plot.y + plot.h, "#333333", 0.5))
    out.append(svg_text(plot.x + plot.w / 2, 143.0, "prior strength lambda", 6.4, "400", TEXT, "middle"))
    for label, vals, color in SERIES:
        pts = [map_xy_svg(plot, x, y) for x, y in zip(LAMBDAS, vals)]
        d = " ".join([f"{'M' if i == 0 else 'L'} {x:.2f} {y:.2f}" for i, (x, y) in enumerate(pts)])
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.15"/>\n')
        for x, y in pts:
            out.append(svg_rect(Rect(x - 1.5, y - 1.5, 3.0, 3.0), color))
        lx = 207.0
        _, y = map_xy_svg(plot, 2.0, vals[-1])
        out.append(svg_rect(Rect(lx, y - 2.0, 4.0, 4.0), color))
        out.append(svg_text(lx + 7.0, y + 2.0, label, 5.8))
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
