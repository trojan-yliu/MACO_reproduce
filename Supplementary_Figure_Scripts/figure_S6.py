#!/usr/bin/env python3
"""Figure S6. Species Venn diagrams at three contamination gates.

Infomap and ImputeCC. Each row is one gate, from 1% to 5%. ONT is the
left half and PacBio the right half. A species enters a method when it
is the majority genome of a bin with completeness >= 50% and
contamination at or below the row's gate. Circle order is MACO, NormCC,
bin3C, as in Figure S3.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Main_Figures_Scripts"))
from figure_2 import (
    COLORS,
    METHODS,
    PLATFORMS,
    load_labels,
    load_ref_lengths,
)
from figure_S3 import (
    COMP_MIN,
    _style,
    _venn_centres,
    _venn_fill,
    _venn_limits,
    score_bins,
    venn_regions,
)


HERE = Path(__file__).resolve().parent
GATES = (1.0, 2.0, 5.0)
PANEL_FONT = 26
COUNT_FONT = 14

# Square Venns. The gutter between ONT and PacBio is wider than the
# gap between clusterers, so each half reads as one panel.
VENN = 2.20
GAP = 0.22
SIDE_GAP = 1.15
ROW_GAP = 0.48
LEFT = 1.28
RIGHT = 0.18
TOP = 1.16
BOTTOM = 0.78


def _load_rows():
    cached = {}
    for platform, label in PLATFORMS:
        genome, _length = load_labels(platform)
        ref_len = load_ref_lengths(platform, genome)
        cached[label] = {}
        for cluster, cluster_label in (
            ("metacc", "Leiden"),
            ("infomap", "Infomap"),
            ("imputecc", "ImputeCC"),
        ):
            cached[label][cluster_label] = {}
            for method in METHODS:
                cached[label][cluster_label][method] = score_bins(
                    cluster, platform, method, genome, ref_len,
                )
    return cached


def _sets(cached, platform, cluster, field, cont_max):
    found = {}
    for method in METHODS:
        names = set()
        for row in cached[platform][cluster][method]:
            if row["contamination"] > cont_max or row["completeness"] < COMP_MIN:
                continue
            name = row[field].strip()
            if name:
                names.add(name)
        found[method] = names
    return found


def _draw_venn(ax, sets):
    """Same circles and region colors as Figure S3."""
    box = ax.get_position().frozen()
    regions = venn_regions(sets)
    centres = _venn_centres()
    radius = 1.08
    limits = _venn_limits(centres, radius)
    _venn_fill(ax, centres, radius, limits)

    def outward(key, scale):
        centre = centres[key]
        return centre + centre / np.linalg.norm(centre) * scale

    def lens(left, right, third, scale):
        mid = (centres[left] + centres[right]) / 2.0
        away = mid - centres[third]
        return mid + away / np.linalg.norm(away) * scale

    labels = {
        "A": outward("A", 0.55),
        "B": outward("B", 0.55),
        "C": outward("C", 0.55),
        "AB": lens("A", "B", "C", 0.30),
        "AC": lens("A", "C", "B", 0.30),
        "BC": lens("B", "C", "A", 0.30),
        "ABC": (centres["A"] + centres["B"] + centres["C"]) / 3.0,
    }
    for key, xy in labels.items():
        ax.text(
            xy[0],
            xy[1],
            str(regions[key]),
            ha="center",
            va="center",
            fontsize=COUNT_FONT,
            fontweight="bold",
            color="black",
            zorder=4,
        )
    left, right, bottom, top = limits
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("auto")
    ax.set_position(box)
    ax.axis("off")
    return regions


def _legend(fig):
    handles = [
        Patch(facecolor=COLORS[method], edgecolor="black", linewidth=0.7, label=method)
        for method in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.012),
        bbox_transform=fig.transFigure,
        ncol=3,
        frameon=False,
        fontsize=13,
        handlelength=1.35,
        handleheight=0.90,
        columnspacing=1.35,
        handletextpad=0.50,
        borderaxespad=0.0,
    )


def draw(cached, path, field, clusters):
    _style()
    n_col = len(clusters)
    n_row = len(GATES)
    side_w = n_col * VENN + (n_col - 1) * GAP
    fig_w = LEFT + 2 * side_w + SIDE_GAP + RIGHT
    fig_h = TOP + n_row * VENN + (n_row - 1) * ROW_GAP + BOTTOM
    fig = plt.figure(figsize=(fig_w, fig_h))
    platforms = [label for _key, label in PLATFORMS]
    letters = "abcdef"
    anchors = []
    panel = 0
    for row, cont in enumerate(GATES):
        y_in = BOTTOM + (n_row - 1 - row) * (VENN + ROW_GAP)
        fig.text(
            (LEFT - 0.55) / fig_w,
            (y_in + 0.5 * VENN) / fig_h,
            "≤ {0:g}%".format(cont),
            fontsize=14,
            ha="right",
            va="center",
        )
        for side, platform in enumerate(platforms):
            x0 = LEFT + side * (side_w + SIDE_GAP)
            letter = letters[panel]
            panel += 1
            fig.text(
                (x0 - 0.10) / fig_w,
                (y_in + VENN) / fig_h,
                letter,
                fontsize=PANEL_FONT,
                fontweight="bold",
                ha="right",
                va="top",
            )
            for col, cluster in enumerate(clusters):
                x_in = x0 + col * (VENN + GAP)
                ax = fig.add_axes(
                    [x_in / fig_w, y_in / fig_h, VENN / fig_w, VENN / fig_h]
                )
                anchors.append((ax, ax.get_position().frozen()))
                sets = _sets(cached, platform, cluster, field, cont)
                regions = _draw_venn(ax, sets)
                print(
                    "[VENN] {0} {1} ≤{2:g}% {3}  {4}".format(
                        path.stem,
                        platform,
                        cont,
                        cluster,
                        " ".join(
                            "{0}={1}".format(key, regions[key])
                            for key in ("A", "B", "C", "AB", "AC", "BC", "ABC")
                        ),
                    ),
                    flush=True,
                )
    y_top = BOTTOM + (n_row - 1) * (VENN + ROW_GAP) + VENN
    for side, platform in enumerate(platforms):
        x0 = LEFT + side * (side_w + SIDE_GAP)
        fig.text(
            (x0 + 0.5 * side_w) / fig_w,
            (y_top + 0.78) / fig_h,
            platform,
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
        for col, cluster in enumerate(clusters):
            fig.text(
                (x0 + col * (VENN + GAP) + 0.5 * VENN) / fig_w,
                (y_top + 0.36) / fig_h,
                cluster,
                fontsize=13,
                ha="center",
                va="bottom",
            )
    _legend(fig)
    fig.canvas.draw()
    for ax, anchor in anchors:
        ax.set_position(anchor)
    fig.canvas.draw()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    plt.close(fig)
    print("[FIG] {0}".format(path), flush=True)


def main():
    draw(
        _load_rows(),
        HERE.parent / "Supplementary_Figures" / "Supplementary_Figure_S6.png",
        "species",
        ("Infomap", "ImputeCC"),
    )


if __name__ == "__main__":
    main()
