import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Data from main table
# =========================
models = ["Qwen3-1.7B", "Qwen3-4B"]
methods = ["Base", "GRPO", "Dr.GRPO", "BPR-GRPO (Ours)"]

avg_scores = np.array([
    [43.97, 44.98, 44.68, 46.65],  # Qwen3-1.7B
    [55.94, 55.97, 55.87, 57.88],  # Qwen3-4B
])

# =========================
# Publication style
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

colors = [
    "#D9EAF7",  # Base: very light blue
    "#A9D3F0",  # GRPO: light blue
    "#6FAED6",  # Dr.GRPO: medium blue
    "#1F5FA8",  # Ours: strong blue
]

hatches = ["", "//", "\\\\", ""]

# One-column friendly size for top-right paper figure
fig, ax = plt.subplots(figsize=(3.35, 2.15), dpi=300)

x = np.arange(len(models))
bar_width = 0.18
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width

for i, method in enumerate(methods):
    bars = ax.bar(
        x + offsets[i],
        avg_scores[:, i],
        width=bar_width,
        label=method,
        color=colors[i],
        edgecolor="black",
        linewidth=0.55,
        hatch=hatches[i],
        zorder=3,
    )

    # Highlight only our method with value labels
    if method == "BPR-GRPO (Ours)":
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.35,
                f"{b.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )

# Improvement over the best baseline for each model
for j in range(len(models)):
    ours = avg_scores[j, -1]
    best_baseline = np.max(avg_scores[j, :-1])
    gain = ours - best_baseline

    ax.text(
        x[j] + offsets[-1],
        ours + 1.35,
        f"+{gain:.2f}",
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        color="#1F5FA8",
    )

# =========================
# Axes / labels
# =========================
ax.set_ylabel("Average accuracy (%)", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=8)
ax.tick_params(axis="y", labelsize=7)

# For compact paper teaser.
# If you want a non-truncated axis, change this to ax.set_ylim(0, 60).
ax.set_ylim(42, 60)

ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.45, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title("BPR-GRPO improves average reasoning accuracy", fontsize=8.5, pad=6)

legend = ax.legend(
    frameon=False,
    fontsize=6.5,
    ncol=2,
    loc="upper left",
    bbox_to_anchor=(-0.02, 1.02),
    handlelength=1.3,
    columnspacing=0.8,
)

plt.tight_layout(pad=0.35)

# =========================
# Save
# =========================
out_dir = Path("figures")
out_dir.mkdir(exist_ok=True)

plt.savefig(out_dir / "bpr_avg_bar_teaser.pdf", bbox_inches="tight")
plt.savefig(out_dir / "bpr_avg_bar_teaser.png", dpi=600, bbox_inches="tight")
plt.show()
