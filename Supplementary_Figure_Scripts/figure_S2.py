#!/usr/bin/env python3
"""Figure S2. Long-read precision and recall on all MOCK1 bins.

Precision and recall use the same definitions as Figure 2. Precision is the
length-weighted majority purity of every bin. Recall keeps, for each genome,
only the longest bin in which that genome is the majority, divided by all
labeled bases of that genome. Values are computed from the MOCK1 tables
in Main_Figures_Scripts/data.

Panel a is ONT and panel b is PacBio. Each cluster block holds Precision and
Recall. Bar width matches Figure 2. The axis starts at 0.4 because ImputeCC
recall on ONT falls to 0.42.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Main_Figures_Scripts"))
from figure_2 import (
    AX,
    CLUSTERS,
    COLORS,
    METHODS,
    PLATFORMS,
    _style,
    load_labels,
    read_bins,
    score,
)


HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "Supplementary_Figures" / "Supplementary_Figure_S2.png"

METRICS = ("Precision", "Recall")
Y_MIN = 0.4
Y_MAX = 1.0
Y_TICKS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

LEFT = 0.48
GAP = 0.55
RIGHT = 0.08
TOP = 0.42
BOTTOM = 0.68


def _bar_width(ax_w):
    """Same bar width as the 27-bar layout in figure_2._layout."""
    side = 0.04
    metric_gap = 0.055
    cluster_gap = 0.11
    usable = ax_w - 2.0 * side
    return (usable - 6.0 * metric_gap - 2.0 * cluster_gap) / 27.0


def _layout(ax_w):
    """Two metric pairs per cluster. Bars keep the Figure 2 width."""
    bar = _bar_width(ax_w)
    metric_gap = 0.18
    cluster_gap = 0.24
    block = 2.0 * (3.0 * bar) + metric_gap
    content = len(CLUSTERS) * block + (len(CLUSTERS) - 1) * cluster_gap
    pos = (ax_w - content) / 2.0
    groups = []
    for _key, cluster in CLUSTERS:
        metrics = []
        block_left = pos
        for metric_index, metric in enumerate(METRICS):
            left = pos
            center = left + 1.5 * bar
            metrics.append((metric, left, center, bar))
            pos += 3.0 * bar
            if metric_index != len(METRICS) - 1:
                pos += metric_gap
        block_right = pos
        groups.append((cluster, metrics, block_left, block_right))
        pos = block_right + cluster_gap
    return groups, ax_w


def compute():
    scores = {}
    lowest = 1.0
    for platform, label in PLATFORMS:
        genome, length = load_labels(platform)
        scores[label] = {}
        for cluster, cluster_label in CLUSTERS:
            column = {metric: [] for metric in METRICS}
            for method in METHODS:
                result = score(genome, length, read_bins(cluster, platform, method))
                for metric in METRICS:
                    column[metric].append(result[metric])
                    lowest = min(lowest, result[metric])
                print(
                    "[SCORE] {0} {1} {2}  P={3:.6f}  R={4:.6f}".format(
                        label,
                        cluster_label,
                        method,
                        result["Precision"],
                        result["Recall"],
                    ),
                    flush=True,
                )
            scores[label][cluster_label] = column
    if lowest < Y_MIN:
        raise SystemExit("[ERROR] score {0:.3f} falls below the axis".format(lowest))
    return scores


def _panel(ax, platform, scores, groups, x_right, letter, ax_h):
    overlap = 0.003
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    metric_y = -0.048 / ax_h
    rule_y = -0.185 / ax_h
    name_y = -0.235 / ax_h
    for cluster, metrics, block_left, block_right in groups:
        for metric, left, center, bar in metrics:
            values = scores[platform][cluster][metric]
            for index, method in enumerate(METHODS):
                ax.bar(
                    left + index * bar,
                    values[index] - Y_MIN,
                    width=bar + overlap,
                    bottom=Y_MIN,
                    align="edge",
                    color=COLORS[method],
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=3,
                )
            ax.text(
                center,
                metric_y,
                metric,
                transform=trans,
                ha="center",
                va="top",
                fontsize=7,
                color="black",
                clip_on=False,
            )
        ax.plot(
            [block_left, block_right],
            [rule_y, rule_y],
            transform=trans,
            color="black",
            lw=2.4,
            clip_on=False,
            solid_capstyle="butt",
            zorder=4,
        )
        ax.text(
            0.5 * (block_left + block_right),
            name_y,
            cluster,
            transform=trans,
            ha="center",
            va="top",
            fontsize=8,
            color="black",
            clip_on=False,
        )
    ax.set_xlim(0, x_right)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_yticks(list(Y_TICKS))
    ax.set_yticklabels(["{0:.1f}".format(tick) for tick in Y_TICKS])
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color("black")
    ax.tick_params(
        axis="y",
        which="both",
        direction="out",
        top=False,
        right=False,
        left=True,
        width=0.7,
        length=3.0,
        pad=2.0,
    )
    ax.tick_params(axis="x", which="both", bottom=False, top=False, length=0)
    ax.set_axisbelow(False)
    ax.text(
        0.0,
        1.012,
        letter,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def draw(scores, path):
    _style()
    fig_w = LEFT + AX + GAP + AX + RIGHT
    fig_h = TOP + AX + BOTTOM
    groups, x_lim = _layout(AX)
    fig = plt.figure(figsize=(fig_w, fig_h))
    width = AX / fig_w
    height = AX / fig_h
    x_left = LEFT / fig_w
    x_right = (LEFT + AX + GAP) / fig_w
    y_lo = BOTTOM / fig_h
    placed = (
        (x_left, y_lo, "a", "ONT"),
        (x_right, y_lo, "b", "PacBio"),
    )
    for x, y, letter, platform in placed:
        ax = fig.add_axes([x, y, width, height])
        _panel(ax, platform, scores, groups, x_lim, letter, AX)
    handles = [
        Patch(facecolor=COLORS[method], edgecolor="black", linewidth=0.6, label=method)
        for method in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.010),
        bbox_transform=fig.transFigure,
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=0.9,
        handleheight=0.62,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    plt.close(fig)
    print("[FIG] {0}".format(path), flush=True)


if __name__ == "__main__":
    draw(compute(), OUT)
