#!/usr/bin/env python3
"""Figure S1. Number of bins and bin length on all MOCK1 bins.

Panels a and c count every bin. Panels b and d draw the sequence length
of every bin. Lengths run from 0.17 Mb to 58 Mb, so the axis is log10 of
megabases. Zero on that axis is 1 Mb.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Main_Figures_Scripts"))
from figure_2 import (
    AX,
    BOTTOM,
    CLUSTERS,
    COLORS,
    GAP,
    LEFT,
    METHODS,
    MID,
    PLATFORMS,
    RIGHT,
    TOP,
    _boxplot,
    _quantile,
    _jitter,
    _muted,
    _style,
    bin_records,
)


HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "Supplementary_Figures" / "Supplementary_Figure_S1.png"

# Compared with log10(bp) and log10(Kb). Those axes are the same shape
# shifted by 6 and by 3. Megabases put 1 Mb on the zero line and keep the
# tick labels on a bacterial-genome scale, so that is the axis used here.
UNIT = 1.0e6
UNIT_NAME = "Mb"
LENGTH_TICKS = (-1, 0, 1, 2)
LENGTH_Y_MIN = -1.20
LENGTH_Y_MAX = 2.15

COUNT_TICKS = (0, 20, 40, 60, 80, 100)
COUNT_Y_MIN = 0
COUNT_Y_MAX = 100

# Three touching bars per cluster, on the same centers as the boxes.
BAR = 0.228
SIDE = 0.20
BOX_W = 0.168


def _layout(ax_w):
    block = BAR * len(METHODS)
    cluster_gap = (ax_w - 2.0 * SIDE - block * len(CLUSTERS)) / (len(CLUSTERS) - 1)
    groups = []
    pos = SIDE
    for _key, cluster in CLUSTERS:
        block_left = pos
        centers = [pos + (index + 0.5) * BAR for index in range(len(METHODS))]
        block_right = pos + block
        groups.append((cluster, centers, block_left, block_right))
        pos = block_right + cluster_gap
    return groups, ax_w


def bin_lengths(cluster, platform, method):
    lengths = []
    for bid, records in bin_records(cluster, platform, method):
        total = sum(size for _cid, size in records)
        if total <= 0:
            raise SystemExit("[ERROR] empty bin {0}".format(bid))
        lengths.append(total)
    return lengths


def compute():
    counts = {}
    lengths = {}
    for platform, label in PLATFORMS:
        counts[label] = {}
        lengths[label] = {}
        for cluster, cluster_label in CLUSTERS:
            counts[label][cluster_label] = []
            lengths[label][cluster_label] = {}
            for method in METHODS:
                values = bin_lengths(cluster, platform, method)
                counts[label][cluster_label].append(len(values))
                lengths[label][cluster_label][method] = values
                median = _quantile(values, 0.5)
                print(
                    "[BIN] {0} {1} {2}  n={3}  median_{4}={5:.3f}".format(
                        label,
                        cluster_label,
                        method,
                        len(values),
                        UNIT_NAME,
                        median / UNIT,
                    ),
                    flush=True,
                )
    return counts, lengths


def _log_length(bp):
    return math.log10(bp / UNIT)


def _frame(ax, letter):
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
    ax.set_xticks([])
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


def _cluster_labels(ax, groups, ax_h):
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    rule_y = -0.055 / ax_h
    name_y = -0.108 / ax_h
    for cluster, _centers, block_left, block_right in groups:
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


def _count_panel(ax, platform, counts, groups, x_right, letter, ax_h):
    overlap = 0.003
    for cluster, centers, _block_left, _block_right in groups:
        values = counts[platform][cluster]
        for index, method in enumerate(METHODS):
            left = centers[index] - 0.5 * BAR
            ax.bar(
                left,
                values[index],
                width=BAR + overlap,
                bottom=0,
                align="edge",
                color=COLORS[method],
                edgecolor="black",
                linewidth=0.6,
                zorder=3,
            )
    _cluster_labels(ax, groups, ax_h)
    ax.set_xlim(0, x_right)
    ax.set_ylim(COUNT_Y_MIN, COUNT_Y_MAX)
    ax.set_yticks(list(COUNT_TICKS))
    ax.set_ylabel("Number of bins")
    _frame(ax, letter)


def _length_panel(ax, platform, lengths, groups, x_right, letter, ax_h):
    cap = 0.40 * BOX_W
    ax.axhline(0.0, color="0.78", lw=0.4, linestyle=(0, (1.3, 1.5)), zorder=1)
    for cluster_index, (cluster, centers, _block_left, _block_right) in enumerate(groups):
        for method_index, method in enumerate(METHODS):
            values = [_log_length(bp) for bp in lengths[platform][cluster][method]]
            x = centers[method_index]
            q1, median, q3, whisker_low, whisker_high, _outliers = _boxplot(values)
            tone = _muted(COLORS[method])
            height = q3 - q1
            if height < 0.02:
                y0 = median - 0.01
                height = 0.02
            else:
                y0 = q1
            y1 = y0 + height
            if whisker_low < y0 - 0.008:
                ax.plot(
                    [x, x],
                    [whisker_low, y0],
                    color="black",
                    lw=0.6,
                    solid_capstyle="butt",
                    zorder=2,
                )
                ax.plot(
                    [x - cap, x + cap],
                    [whisker_low, whisker_low],
                    color="black",
                    lw=0.6,
                    solid_capstyle="butt",
                    zorder=2,
                )
            if whisker_high > y1 + 0.008:
                ax.plot(
                    [x, x],
                    [y1, whisker_high],
                    color="black",
                    lw=0.6,
                    solid_capstyle="butt",
                    zorder=2,
                )
                ax.plot(
                    [x - cap, x + cap],
                    [whisker_high, whisker_high],
                    color="black",
                    lw=0.6,
                    solid_capstyle="butt",
                    zorder=2,
                )
            ax.scatter(
                [
                    x + _jitter(index + 1000 * cluster_index + 80 * method_index) * 0.92 * BOX_W
                    for index, _value in enumerate(values)
                ],
                values,
                s=11,
                c=[tone + (0.82,)] * len(values),
                edgecolors="none",
                linewidths=0,
                zorder=4,
                clip_on=True,
            )
            ax.add_patch(
                Rectangle(
                    (x - 0.5 * BOX_W, y0),
                    BOX_W,
                    height,
                    facecolor=tone + (0.12,),
                    edgecolor="black",
                    linewidth=1.35,
                    zorder=3,
                )
            )
            ax.plot(
                [x - 0.5 * BOX_W, x + 0.5 * BOX_W],
                [median, median],
                color="black",
                lw=1.25,
                solid_capstyle="butt",
                zorder=5,
            )
    _cluster_labels(ax, groups, ax_h)
    ax.set_xlim(0, x_right)
    ax.set_ylim(LENGTH_Y_MIN, LENGTH_Y_MAX)
    ax.set_yticks(list(LENGTH_TICKS))
    ax.set_ylabel(r"$\log_{10}$ (bin length / " + UNIT_NAME + ")")
    _frame(ax, letter)


def draw(counts, lengths, path):
    _style()
    fig_w = LEFT + AX + GAP + AX + RIGHT
    fig_h = TOP + AX + MID + AX + BOTTOM
    groups, x_lim = _layout(AX)
    fig = plt.figure(figsize=(fig_w, fig_h))
    width = AX / fig_w
    height = AX / fig_h
    x_left = LEFT / fig_w
    x_box = (LEFT + AX + GAP) / fig_w
    y_lo = BOTTOM / fig_h
    y_hi = (BOTTOM + AX + MID) / fig_h
    placed = (
        (x_left, y_hi, "a", "ONT", "count"),
        (x_box, y_hi, "b", "ONT", "length"),
        (x_left, y_lo, "c", "PacBio", "count"),
        (x_box, y_lo, "d", "PacBio", "length"),
    )
    for x, y, letter, platform, kind in placed:
        ax = fig.add_axes([x, y, width, height])
        if kind == "count":
            _count_panel(ax, platform, counts, groups, x_lim, letter, AX)
        else:
            _length_panel(ax, platform, lengths, groups, x_lim, letter, AX)
    handles = [
        Patch(facecolor=COLORS[method], edgecolor="black", linewidth=0.6, label=method)
        for method in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        bbox_transform=fig.transFigure,
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=1.05,
        handleheight=0.72,
        columnspacing=1.15,
        borderaxespad=0.0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    plt.close(fig)
    print("[FIG] {0}".format(path), flush=True)


if __name__ == "__main__":
    draw(*compute(), OUT)
