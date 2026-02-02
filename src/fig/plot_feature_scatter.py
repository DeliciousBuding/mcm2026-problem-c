from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox

COLOR_PRIMARY = "#0072B2"
COLOR_GRAY = "#7A7A7A"

FIG_DIR = Path("paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = FIG_DIR / "fig_feature_scatter.pdf"
DATA_PATH = Path("outputs/feature_effects.csv")


def set_plot_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.figsize": (5.4, 4.2),
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.30,
        "legend.frameon": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
    })


def pretty_label(raw: str) -> str:
    label = str(raw)
    label = label.replace("C(celebrity_industry)", "ind")
    label = label.replace("C(ballroom_partner)", "pro")
    label = label.replace("celebrity_industry_", "ind:")
    label = label.replace("ballroom_partner_", "pro:")
    label = label.replace("[T.", ":")
    label = label.replace("]", "")
    label = label.replace("C(", "").replace(")", "")
    return label


def generate_offsets() -> List[Tuple[int, int]]:
    offsets: List[Tuple[int, int]] = []
    radii = [6, 10, 14, 18, 22, 26]
    angles = [45, -45, 135, -135, 0, 90, 180, -90]
    seen = set()
    for r in radii:
        for ang in angles:
            rad = math.radians(ang)
            dx = int(round(r * math.cos(rad)))
            dy = int(round(r * math.sin(rad)))
            if (dx, dy) in seen:
                continue
            seen.add((dx, dy))
            offsets.append((dx, dy))
    return offsets


def rank_offsets(x: float, y: float, offsets: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    sx = 1 if x >= 0 else -1
    sy = 1 if y >= 0 else -1

    def score(offset: Tuple[int, int]) -> Tuple[int, int]:
        dx, dy = offset
        sign_score = 0
        if dx * sx > 0:
            sign_score -= 2
        if dy * sy > 0:
            sign_score -= 2
        dist_score = abs(dx) + abs(dy)
        return (sign_score, dist_score)

    return sorted(offsets, key=score)


def bbox_inside(inner: Bbox, outer: Bbox, pad: float = 1.0) -> bool:
    return (
        inner.x0 >= outer.x0 + pad
        and inner.x1 <= outer.x1 - pad
        and inner.y0 >= outer.y0 + pad
        and inner.y1 <= outer.y1 - pad
    )


def point_bbox(ax: plt.Axes, x: float, y: float, size: float = 4.0) -> Bbox:
    px, py = ax.transData.transform((x, y))
    return Bbox.from_bounds(px - size, py - size, size * 2, size * 2)


def place_label(
    ax: plt.Axes,
    fig: plt.Figure,
    x: float,
    y: float,
    label: str,
    offsets: Sequence[Tuple[int, int]],
    placed_bboxes: List[Bbox],
    point_bboxes: Iterable[Bbox],
    renderer,
    ax_bbox: Bbox,
) -> None:
    for dx, dy in offsets:
        if dx > 0:
            ha = "left"
        elif dx < 0:
            ha = "right"
        else:
            ha = "center"
        if dy > 0:
            va = "bottom"
        elif dy < 0:
            va = "top"
        else:
            va = "center"
        ann = ax.annotate(
            label,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7,
            alpha=0.88,
            ha=ha,
            va=va,
            arrowprops=dict(arrowstyle="-", color=COLOR_GRAY, lw=0.6, shrinkA=0, shrinkB=4),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
            zorder=5,
        )
        fig.canvas.draw()
        bbox = ann.get_window_extent(renderer)
        if not bbox_inside(bbox, ax_bbox):
            ann.remove()
            continue
        if any(bbox.overlaps(b) for b in placed_bboxes):
            ann.remove()
            continue
        if any(bbox.overlaps(pb) for pb in point_bboxes):
            ann.remove()
            continue
        placed_bboxes.append(bbox)
        return

    # Fallback: use the first offset even if it overlaps.
    dx, dy = offsets[0]
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=7,
        alpha=0.88,
        ha="left" if dx >= 0 else "right",
        va="bottom" if dy >= 0 else "top",
        arrowprops=dict(arrowstyle="-", color=COLOR_GRAY, lw=0.6, shrinkA=0, shrinkB=4),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
        zorder=5,
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"Missing {DATA_PATH}. Run the pipeline to generate feature_effects.csv.")

    set_plot_style()
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["effect_j", "effect_f", "feature"])
    if df.empty:
        raise SystemExit("feature_effects.csv is empty.")

    x = df["effect_j"].to_numpy()
    y = df["effect_f"].to_numpy()
    diffs = np.abs(y - x)
    top_idx = np.argsort(diffs)[-5:][::-1]

    fig, ax = plt.subplots()
    ax.scatter(x, y, color=COLOR_PRIMARY, s=26, alpha=0.9)
    lim = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), 0.1)
    pad = 0.08 * lim
    ax.plot([-lim - pad, lim + pad], [-lim - pad, lim + pad], color=COLOR_GRAY, linestyle="--", linewidth=1.0)
    ax.set_xlim(-lim - pad, lim + pad)
    ax.set_ylim(-lim - pad, lim + pad)
    ax.set_xlabel("Judge effect")
    ax.set_ylabel("Fan effect")
    ax.set_title("Feature impacts: judges vs fans")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer)
    placed_bboxes: List[Bbox] = []
    offsets = generate_offsets()

    point_bboxes = [point_bbox(ax, float(x[i]), float(y[i]), size=4.0) for i in top_idx]
    for i in top_idx:
        label = pretty_label(df["feature"].iloc[i])
        ranked_offsets = rank_offsets(float(x[i]), float(y[i]), offsets)
        place_label(
            ax,
            fig,
            float(x[i]),
            float(y[i]),
            label,
            ranked_offsets,
            placed_bboxes,
            point_bboxes,
            renderer,
            ax_bbox,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    print(f"Written {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
