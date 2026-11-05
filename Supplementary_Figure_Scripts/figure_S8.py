#!/usr/bin/env python3
"""Figure S8. Real-data log10(completeness / contamination), same boxes as figure 2.

Panels sit in the figure 3 arrangement. The first row is panel a: Mount
Melbourne, then Mount Rittmann. The second row is panel b, the mouse fecal
pellet, and panel c, the sheep gut. Each panel is a Tukey box plot: one box per contact method and three
clusterers, with the same pale box, black median and jittered bins as
figure 2. Whiskers stop at the last point inside 1.5 IQR.

A bin is kept when CheckM2 completeness is at least 50% and contamination
is at most 10%. The shared axis starts at 0.5, just under log10(50/10).
A contamination of 0 is entered as 0.01. Every panel draws every kept bin.
The box outline and median sit above the points.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory


HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "Supplementary_Figures" / "Supplementary_Figure_S8.png"
TABLE = HERE.parent / "Supplementary_Tables" / "Supplementary_Table_S3.xlsx"

METHODS = ("MACO", "NormCC", "bin3C")
METHOD_DIR = {"MACO": "maconi", "NormCC": "normcc", "bin3C": "bin3c"}
CLUSTERS = (
    ("metacc", "Leiden"),
    ("infomap", "Infomap"),
    ("imputecc", "ImputeCC"),
)
# Present only so the cluster brackets fall where they do in figure 2.
METRICS = ("F1", "ARI", "NMI")
COLORS = {"MACO": "#E64B35", "NormCC": "#00A087", "bin3C": "#3C5488"}

COMP_MIN = 50.0
CONT_MAX = 10.0
# log10(50/10) is 0.70, so 0.5 leaves a small margin under every kept bin.
BOX_Y_MIN = 0.5
BOX_Y_MAX = 4.65
BOX_TICKS = (1, 2, 3, 4)

# Square panels, same data width as figure 2, in the figure 3 grid.
PANEL = 3.05
LEFT = 0.78
COL_GAP = 0.78
ROW_GAP = 1.12
RIGHT = 0.16
TOP = 0.62
BOTTOM = 1.28


def _font():
    names = {font.name for font in font_manager.fontManager.ttflist}
    for cand in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"):
        if cand in names:
            return cand
    return "DejaVu Sans"


def _style():
    font = _font()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font, "Arial", "Helvetica", "DejaVu Sans"],
            "font.weight": "normal",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.labelweight": "normal",
            "axes.titlesize": 15,
            "axes.titleweight": "normal",
            "axes.linewidth": 0.7,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 13,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
        }
    )


def _rgb(hex_color):
    text = hex_color.lstrip("#")
    return tuple(int(text[index : index + 2], 16) / 255.0 for index in (0, 2, 4))


def _muted(hex_color):
    """Keep the bar hue, with only a slight softening for the strip points."""
    red, green, blue = _rgb(hex_color)
    gray = 0.30 * red + 0.59 * green + 0.11 * blue
    chroma = 0.92
    red = gray + (red - gray) * chroma
    green = gray + (green - gray) * chroma
    blue = gray + (blue - gray) * chroma
    return red, green, blue


def _jitter(index):
    raw = math.sin(index * 12.9898) * 43758.5453
    return raw - math.floor(raw) - 0.5


def _quantile(values, probability):
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return ordered[low]
    weight = index - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _boxplot(values):
    """Tukey box. Whiskers stop at the last point inside 1.5 IQR."""
    q1 = _quantile(values, 0.25)
    median = _quantile(values, 0.5)
    q3 = _quantile(values, 0.75)
    iqr = q3 - q1
    low_fence = q1 - 1.5 * iqr
    high_fence = q3 + 1.5 * iqr
    inside = [value for value in values if low_fence <= value <= high_fence]
    outliers = [value for value in values if value < low_fence or value > high_fence]
    return q1, median, q3, min(inside), max(inside), outliers


def _layout(ax_w):
    """Same cluster brackets as figure 2. The metric slots are not drawn."""
    side = 0.04
    metric_gap = 0.055
    cluster_gap = 0.11
    usable = ax_w - 2.0 * side
    bar = (usable - 6.0 * metric_gap - 2.0 * cluster_gap) / 27.0
    groups = []
    pos = side
    for _key, cluster in CLUSTERS:
        block_left = pos
        for _metric in METRICS:
            pos += 3.0 * bar + metric_gap
        block_right = pos - metric_gap
        groups.append((cluster, block_left, block_right))
        pos = block_right + cluster_gap
    return groups, ax_w


def load_ratios(path):
    """log10(completeness / contamination) for the 50% / 10% bins.

    Contamination of 0 is entered as 0.01.
    """
    values = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            completeness = float(row["completeness"])
            contamination = float(row["contamination"])
            if completeness < COMP_MIN or contamination > CONT_MAX:
                continue
            values.append(math.log10(completeness / max(contamination, 0.01)))
    if not values:
        raise SystemExit("[ERROR] no bins in {0}".format(path))
    return values


def _ratios(frame):
    """log10(completeness / contamination). Contamination of 0 is entered as 0.01."""
    import math

    values = []
    for comp, cont in zip(frame["Completeness (%)a"], frame["Contamination (%)"]):
        completeness = float(comp)
        contamination = float(cont)
        if completeness < COMP_MIN or contamination > CONT_MAX:
            continue
        values.append(math.log10(completeness / max(contamination, 0.01)))
    return values


def compute():
    import pandas as pd

    frame = pd.read_excel(TABLE, header=2)
    frame = frame[frame["Dataset"].isin(
        ["Mount Melbourne", "Mount Rittmann", "Mouse fecal pellet", "Sheep gut"]
    )]
    panels = (
        ("a", "Mount Melbourne", "Mount Melbourne"),
        ("", "Mount Rittmann", "Mount Rittmann"),
        ("b", "Mouse fecal", "Mouse fecal pellet"),
        ("c", "Sheep gut", "Sheep gut"),
    )
    packed = []
    for letter, title, dataset in panels:
        by_cluster = {}
        for _key, cluster_label in CLUSTERS:
            by_cluster[cluster_label] = {}
        for _cluster, cluster_label in CLUSTERS:
            for method in METHODS:
                sub = frame[
                    (frame["Dataset"] == dataset)
                    & (frame["Clustering"] == cluster_label)
                    & (frame["Normalization"] == method)
                ]
                values = _ratios(sub)
                if not values:
                    raise SystemExit("[ERROR] no bins for {0} {1} {2}".format(title, cluster_label, method))
                by_cluster[cluster_label][method] = values
                ordered = sorted(values)
                print(
                    "[BOX] {0} {1} {2}  n={3}  median={4:.2f}  min={5:.2f}  max={6:.2f}".format(
                        title,
                        cluster_label,
                        method,
                        len(values),
                        ordered[len(ordered) // 2],
                        ordered[0],
                        ordered[-1],
                    ),
                    flush=True,
                )
        packed.append((letter, title, by_cluster))
    return packed


def _ratio_panel(ax, ratios, groups, x_right, ax_h, title):
    """One box per method, on the figure 2 cluster brackets."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    rule_y = -0.185 / ax_h
    name_y = -0.235 / ax_h
    box_w = 0.168
    step = 0.228
    cap = 0.40 * box_w
    for cluster_index, (cluster, block_left, block_right) in enumerate(groups):
        block_center = 0.5 * (block_left + block_right)
        origin = block_center - 0.5 * step * (len(METHODS) - 1)
        for method_index, method in enumerate(METHODS):
            values = ratios[cluster][method]
            x = origin + method_index * step
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
                    zorder=6,
                )
                ax.plot(
                    [x - cap, x + cap],
                    [whisker_low, whisker_low],
                    color="black",
                    lw=0.6,
                    solid_capstyle="butt",
                    zorder=6,
                )
            if whisker_high > y1 + 0.008:
                ax.plot(
                    [x, x],
                    [y1, whisker_high],
                    color="black",
                    lw=0.6,
                    solid_capstyle="butt",
                    zorder=6,
                )
                ax.plot(
                    [x - cap, x + cap],
                    [whisker_high, whisker_high],
                    color="black",
                    lw=0.6,
                    solid_capstyle="butt",
                    zorder=6,
                )
            ax.scatter(
                [
                    x + _jitter(index + 1000 * cluster_index + 80 * method_index) * 0.92 * box_w
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
                    (x - 0.5 * box_w, y0),
                    box_w,
                    height,
                    facecolor=tone + (0.12,),
                    edgecolor="none",
                    linewidth=0,
                    zorder=3,
                )
            )
            ax.add_patch(
                Rectangle(
                    (x - 0.5 * box_w, y0),
                    box_w,
                    height,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=1.35,
                    zorder=7,
                )
            )
            ax.plot(
                [x - 0.5 * box_w, x + 0.5 * box_w],
                [median, median],
                color="black",
                lw=1.25,
                solid_capstyle="butt",
                zorder=8,
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
        label = ax.text(
            block_center,
            name_y,
            cluster,
            transform=trans,
            ha="center",
            va="top",
            fontsize=11,
            color="black",
            clip_on=False,
        )
        label.set_gid("cluster-label")
    ax.set_xlim(0, x_right)
    ax.set_ylim(BOX_Y_MIN, BOX_Y_MAX)
    ax.set_yticks(list(BOX_TICKS))
    ax.set_xticks([])
    ax.set_ylabel(r"$\log_{10}$ (completeness / contamination)")
    ax.set_title(title, fontsize=15, pad=8)
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


def _legend(fig):
    """One centered row of method squares, clear of the lower cluster names."""
    square = 1.05
    handles = [
        Patch(facecolor=COLORS[method], edgecolor="black", linewidth=0.7, label=method)
        for method in METHODS
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.016),
        bbox_transform=fig.transFigure,
        ncol=len(METHODS),
        frameon=False,
        fontsize=13,
        handlelength=square,
        handleheight=square,
        handletextpad=0.45,
        columnspacing=1.15,
        borderaxespad=0.0,
        borderpad=0.2,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width = legend.get_window_extent(renderer).width / fig.bbox.width
    legend.set_bbox_to_anchor((0.5 - width / 2.0, 0.016), transform=fig.transFigure)


def _letters(fig, drawn):
    """a, b and c at the upper left. Rittmann stays inside panel a."""
    for ax, letter in drawn:
        if not letter:
            continue
        ax.text(
            -0.01,
            1.01,
            letter,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            ha="right",
            va="bottom",
            clip_on=False,
        )


def draw(packed, path):
    _style()
    fig_w = LEFT + PANEL + COL_GAP + PANEL + RIGHT
    fig_h = BOTTOM + PANEL + ROW_GAP + PANEL + TOP
    fig = plt.figure(figsize=(fig_w, fig_h))
    width = PANEL / fig_w
    height = PANEL / fig_h
    x_left = LEFT / fig_w
    x_right = (LEFT + PANEL + COL_GAP) / fig_w
    y_top = (BOTTOM + PANEL + ROW_GAP) / fig_h
    y_bottom = BOTTOM / fig_h
    origins = (
        (x_left, y_top),
        (x_right, y_top),
        (x_left, y_bottom),
        (x_right, y_bottom),
    )
    groups, x_data = _layout(PANEL)
    drawn = []
    anchors = []
    for (x, y), (letter, title, ratios) in zip(origins, packed):
        ax = fig.add_axes([x, y, width, height])
        anchors.append(ax.get_position().frozen())
        _ratio_panel(ax, ratios, groups, x_data, PANEL, title)
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
