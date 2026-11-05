#!/usr/bin/env python3
"""Figure 3. Real-data bins at contamination <= 1%, stacked by completeness.

The stacks follow Figure S3. A bin is kept when CheckM2 completeness is
at least 50% and contamination is at most 1%. From the axis outward the
bands are >90%, 80-90%, 70-80%, 60-70% and 50-60%. Each normalization
keeps its own color; a lighter segment is a lower completeness band.

Panel a is the first row: Mount Melbourne and Mount Rittmann. Panel b is
the mouse fecal pellet. Panel c is the sheep gut. The legend sits below
the four panels.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory


HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "Main_Figures" / "Main_Figure_3.png"
TABLE = HERE.parent / "Supplementary_Tables" / "Supplementary_Table_S3.xlsx"

METHODS = ("MACO", "NormCC", "bin3C")
METHOD_DIR = {"MACO": "maconi", "NormCC": "normcc", "bin3C": "bin3c"}
CLUSTERS = (
    ("metacc", "Leiden"),
    ("infomap", "Infomap"),
    ("imputecc", "ImputeCC"),
)
COLORS = {"MACO": "#E64B35", "NormCC": "#00A087", "bin3C": "#3C5488"}

# Left to right. The first band is the solid method color.
BANDS = (
    (90.0, None, ">90%"),
    (80.0, 90.0, "80–90%"),
    (70.0, 80.0, "70–80%"),
    (60.0, 70.0, "60–70%"),
    (50.0, 60.0, "50–60%"),
)
BAND_WHITE = (0.0, 0.32, 0.50, 0.66, 0.80)
COMP_MIN = 50.0
CONT_MAX = 1.0

# Bars in one cluster share an edge. The gap is only between clusters.
BAR_THICK = 0.42
BAR_OVERLAP = 0.012
GROUP_GAP = 0.78

BAR_H = 3.28
BAR_W = 3.35
# Room for the cluster names, and for panel letters on the left of those names.
LABEL_W = 2.35
COL_GAP = 0.22
ROW_GAP = 1.02
RIGHT = 0.16
TOP = 0.58
BOTTOM = 1.32


def _font():
    names = {font.name for font in font_manager.fontManager.ttflist}
    for cand in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"):
        if cand in names:
            return cand
    return "DejaVu Sans"


def _rgb(hex_color):
    text = hex_color.lstrip("#")
    return tuple(int(text[index : index + 2], 16) / 255.0 for index in (0, 2, 4))


def _tint(hex_color, white):
    red, green, blue = _rgb(hex_color)
    return (
        red + (1.0 - red) * white,
        green + (1.0 - green) * white,
        blue + (1.0 - blue) * white,
    )


def _style():
    font = _font()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font, "Arial", "Helvetica", "DejaVu Sans"],
            "font.weight": "normal",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.labelweight": "normal",
            "axes.titlesize": 15,
            "axes.titleweight": "normal",
            "axes.linewidth": 1.05,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
        }
    )


def _in_band(completeness, contamination, low, high):
    if contamination > CONT_MAX or completeness < COMP_MIN:
        return False
    if completeness < low:
        return False
    if high is None:
        return True
    return completeness < high


def band_counts(rows):
    return [
        sum(1 for row in rows if _in_band(row["completeness"], row["contamination"], low, high))
        for low, high, _label in BANDS
    ]


def load_bins(path):
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append(
                {
                    "completeness": float(row["completeness"]),
                    "contamination": float(row["contamination"]),
                }
            )
    if not rows:
        raise SystemExit("[ERROR] no bins in {0}".format(path))
    return rows


def _table_rows():
    import pandas as pd

    frame = pd.read_excel(TABLE, header=2)
    frame = frame[frame["Dataset"].isin(
        ["Mount Melbourne", "Mount Rittmann", "Mouse fecal pellet", "Sheep gut"]
    )]
    return frame


def compute():
    """Four panels. The first two share panel letter a."""
    frame = _table_rows()
    panels = (
        ("a", "Mount Melbourne", "Mount Melbourne"),
        ("", "Mount Rittmann", "Mount Rittmann"),
        ("b", "Mouse fecal", "Mouse fecal pellet"),
        ("c", "Sheep gut", "Sheep gut"),
    )
    stacks = []
    for letter, title, dataset in panels:
        by_cluster = {}
        for _cluster, cluster_label in CLUSTERS:
            by_cluster[cluster_label] = {}
            for method in METHODS:
                sub = frame[
                    (frame["Dataset"] == dataset)
                    & (frame["Clustering"] == cluster_label)
                    & (frame["Normalization"] == method)
                ]
                rows = [
                    {"completeness": float(comp), "contamination": float(cont)}
                    for comp, cont in zip(sub["Completeness (%)a"], sub["Contamination (%)"])
                ]
                counts = band_counts(rows)
                by_cluster[cluster_label][method] = counts
                print(
                    "[BAND] {0} {1} {2}  {3}  total={4}".format(
                        title,
                        cluster_label,
                        method,
                        " ".join(
                            "{0}={1}".format(name, count)
                            for (_lo, _hi, name), count in zip(BANDS, counts)
                        ),
                        sum(counts),
                    ),
                    flush=True,
                )
        stacks.append((letter, title, by_cluster))
    return stacks


def _bar_geometry():
    """ImputeCC at the bottom, Leiden at the top. MACO is the top row of a group."""
    clusters = [label for _key, label in CLUSTERS]
    pad = 0.62
    pos = pad + BAR_THICK / 2.0
    rows = []
    spans = {}
    for index, cluster in enumerate(reversed(clusters)):
        ys = []
        for method in reversed(METHODS):
            rows.append((pos, cluster, method))
            ys.append(pos)
            pos += BAR_THICK
        spans[cluster] = (min(ys), max(ys))
        if index < len(clusters) - 1:
            pos += GROUP_GAP
    y_max = pos - BAR_THICK / 2.0 + pad
    return rows, spans, y_max


def _count_axis(peak):
    if peak > 200:
        step = 100.0
    elif peak > 80:
        step = 20.0
    elif peak > 40:
        step = 10.0
    elif peak > 15:
        step = 5.0
    else:
        step = 5.0
    top = step * np.ceil((peak * 1.08) / step)
    if top <= peak:
        top += step
    ticks = [float(tick) for tick in np.arange(0.0, top + 0.01, step)]
    return float(top), ticks


def _frame(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.05)
        ax.spines[side].set_color("black")
    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        top=False,
        right=False,
        width=1.0,
        length=5.0,
        pad=3.0,
        labelsize=12,
    )
    ax.set_axisbelow(False)


def _bar_panel(ax, title, stacks, x_max, x_ticks):
    rows, spans, y_max = _bar_geometry()
    for y_pos, cluster, method in rows:
        left = 0.0
        for band_index, count in enumerate(stacks[cluster][method]):
            if count <= 0:
                continue
            ax.barh(
                y_pos,
                count,
                left=left,
                height=BAR_THICK + BAR_OVERLAP,
                color=_tint(COLORS[method], BAND_WHITE[band_index]),
                edgecolor="black",
                linewidth=0.7,
                zorder=3,
                align="center",
            )
            left += count
    ax.set_yticks([y_pos for y_pos, _cluster, _method in rows])
    ax.set_yticklabels([method for _y, _cluster, method in rows], fontsize=12)
    ax.set_ylim(0.0, y_max)
    ax.set_xlim(0.0, x_max)
    ax.set_xticks(x_ticks)
    ax.set_xlabel("Number of bins (contamination < 1%)", fontsize=14, labelpad=6)
    ax.set_title(title, fontsize=15, pad=8)
    ax.tick_params(axis="y", length=0, pad=3)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    rule_x = -0.95 / BAR_W
    half = BAR_THICK / 2.0
    for cluster, (low, high) in spans.items():
        ax.plot(
            [rule_x, rule_x],
            [low - half, high + half],
            transform=trans,
            color="black",
            lw=2.4,
            clip_on=False,
            solid_capstyle="butt",
            zorder=4,
        )
        label = ax.text(
            rule_x - 0.08 / BAR_W,
            0.5 * (low + high),
            cluster,
            transform=trans,
            ha="right",
            va="center",
            fontsize=13,
            color="black",
            clip_on=False,
        )
        label.set_gid("cluster-label")
    _frame(ax)


def _legend(fig):
    """One row: method colors, a gap, then the completeness bands. Same squares."""
    square = 1.05
    text = 13
    edge = 0.7
    methods = [
        Patch(facecolor=COLORS[method], edgecolor="black", linewidth=edge, label=method)
        for method in METHODS
    ]
    bands = [
        Patch(
            facecolor=_tint(COLORS["MACO"], white),
            edgecolor="black",
            linewidth=edge,
            label=label,
        )
        for (_low, _high, label), white in zip(BANDS, BAND_WHITE)
    ]
    shared = {
        "loc": "lower left",
        "bbox_transform": fig.transFigure,
        "frameon": False,
        "fontsize": text,
        "handlelength": square,
        "handleheight": square,
        "handletextpad": 0.45,
        "columnspacing": 1.05,
        "borderaxespad": 0.0,
        "borderpad": 0.2,
    }
    method_legend = fig.legend(
        handles=methods,
        bbox_to_anchor=(0.0, 0.02),
        ncol=len(METHODS),
        **shared,
    )
    band_legend = fig.legend(
        handles=bands,
        bbox_to_anchor=(0.0, 0.02),
        ncol=len(BANDS),
        **shared,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_box = fig.bbox
    method_width = method_legend.get_window_extent(renderer).width / fig_box.width
    band_width = band_legend.get_window_extent(renderer).width / fig_box.width
    gap = 0.045
    left = 0.5 - (method_width + gap + band_width) / 2.0
    y = 0.016
    method_legend.set_bbox_to_anchor((left, y), transform=fig.transFigure)
    band_legend.set_bbox_to_anchor((left + method_width + gap, y), transform=fig.transFigure)


def _letters(fig, drawn):
    """Put a, b and c left of the cluster names. Rittmann stays inside panel a."""
    renderer = fig.canvas.get_renderer()
    for ax, letter in drawn:
        if not letter:
            continue
        left = None
        for text in ax.texts:
            if text.get_gid() != "cluster-label":
                continue
            box = text.get_window_extent(renderer)
            left = box.x0 if left is None else min(left, box.x0)
        ax_x, _ax_y = ax.transAxes.inverted().transform((left, 0.0))
        ax.text(
            ax_x - 0.03,
            1.02,
            letter,
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            ha="right",
            va="bottom",
            clip_on=False,
        )


def draw(stacks, path):
    _style()
    fig_w = LABEL_W + BAR_W + COL_GAP + LABEL_W + BAR_W + RIGHT
    fig_h = BOTTOM + BAR_H + ROW_GAP + BAR_H + TOP
    fig = plt.figure(figsize=(fig_w, fig_h))
    width = BAR_W / fig_w
    height = BAR_H / fig_h
    x_left = LABEL_W / fig_w
    x_right = (LABEL_W + BAR_W + COL_GAP + LABEL_W) / fig_w
    y_top = (BOTTOM + BAR_H + ROW_GAP) / fig_h
    y_bottom = BOTTOM / fig_h
    origins = (
        (x_left, y_top),
        (x_right, y_top),
        (x_left, y_bottom),
        (x_right, y_bottom),
    )
    drawn = []
    anchors = []
    for (x, y), (letter, title, by_cluster) in zip(origins, stacks):
        ax = fig.add_axes([x, y, width, height])
        anchors.append(ax.get_position().frozen())
        peak = max(
            sum(by_cluster[cluster][method])
            for _key, cluster in CLUSTERS
            for method in METHODS
        )
        x_max, x_ticks = _count_axis(peak)
        _bar_panel(ax, title, by_cluster, x_max, x_ticks)
        drawn.append((ax, letter))
    _legend(fig)
    fig.canvas.draw()
    for ax, box in zip((item[0] for item in drawn), anchors):
        ax.set_position(box)
    fig.canvas.draw()
    _letters(fig, drawn)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    plt.close(fig)
    print("[FIG] {0}".format(path), flush=True)


if __name__ == "__main__":
    draw(compute(), OUT)
