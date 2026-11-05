#!/usr/bin/env python3
"""Figure S4. Same panels as Figure S3, at a 2% contamination gate.

S4 keeps bins with contamination <= 2%. Completeness stays >= 50%.
From the axis outward the bands are >90%, 80-90%, 70-80%, 60-70% and
50-60%. Each method keeps its own color; a lighter segment is a lower
completeness band.

Panels b and e are three-set species Venn diagrams for Leiden only.
A species enters a method when it is the majority genome of a bin in
that same quality gate. The geometry follows the sheep-gut Venn.

Panels c and f follow the sheep-gut comparison scatter. A species or
genus counts only inside that gate, and only when the other two
normalization methods do not recover it. Color is the method. Shape is
the clustering algorithm. Markers that share a coordinate stay as one
point, and the covered markers are drawn in a callout.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Patch
from matplotlib.transforms import blended_transform_factory

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Main_Figures_Scripts"))
from figure_2 import (
    CLUSTERS,
    COLORS,
    METHODS,
    PLATFORMS,
    _font,
    _rgb,
    bin_records,
    load_labels,
    load_ref_lengths,
)


HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "Supplementary_Figures" / "Supplementary_Figure_S4.png"
CONT_MAX = 2.0

# The bar panels are square. Labels sit in the left margin.
BAR_H = 3.72
BAR_W = BAR_H
VENN = 3.72
SCAT = 3.72
LEFT = 2.28
GAP = 0.48
GAP_SCAT = 1.12
RIGHT = 0.14
TOP = 0.56
PANEL_FONT = 26
MID = 1.18
BOTTOM = 1.62

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

SHAPES = {"Leiden": "o", "Infomap": "^", "ImputeCC": "*"}
# Stars draw smaller than circles at the same nominal size, so ImputeCC is larger.
MARKER_SIZE = {"Leiden": 14.0, "Infomap": 15.0, "ImputeCC": 18.0}
LEGEND_MARKER = {"Leiden": 11.0, "Infomap": 12.0, "ImputeCC": 14.0}
# The callout is an annotation. Keep it smaller than the plotted points.
CALLOUT_MARKER = {"Leiden": 7.2, "Infomap": 7.6, "ImputeCC": 9.0}
# Bars in one cluster share an edge. The gap is only between clusters.
BAR_THICK = 0.42
BAR_OVERLAP = 0.012
GROUP_GAP = 0.78


def taxon_of(genome):
    """Genus is the first token. Species keeps the specific epithet."""
    parts = [part for part in genome.split("_") if part]
    if not parts:
        return "", ""
    genus = parts[0]
    if len(parts) >= 2 and parts[1] in {"sp.", "sp"}:
        return genus + " sp.", genus
    if len(parts) >= 2:
        return genus + " " + parts[1], genus
    return genus, genus


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


def score_bins(cluster, platform, method, genome, ref_len):
    rows = []
    grouped = bin_records(cluster, platform, method)
    for _name, records in grouped:
        total = sum(size for _cid, size in records)
        by_genome = defaultdict(int)
        for cid, size in records:
            name_g = genome.get(cid)
            if name_g:
                by_genome[name_g] += size
        if total <= 0 or not by_genome:
            continue
        majority = max(by_genome, key=by_genome.get)
        labeled = by_genome[majority]
        reference = ref_len[majority]
        if reference <= 0:
            continue
        species, genus = taxon_of(majority)
        rows.append(
            {
                "species": species,
                "genus": genus,
                "completeness": 100.0 * labeled / float(reference),
                "contamination": 100.0 * (total - labeled) / float(total),
            }
        )
    return rows


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


def _names(rows, field):
    found = set()
    for row in rows:
        if row["contamination"] > CONT_MAX or row["completeness"] < COMP_MIN:
            continue
        name = row[field].strip()
        if name:
            found.add(name)
    return found


def species_sets(rows_by_method):
    """Species of bins at completeness >= 50% and contamination <= CONT_MAX."""
    found = {}
    for method in METHODS:
        names = set()
        for row in rows_by_method[method]:
            if row["contamination"] > CONT_MAX or row["completeness"] < COMP_MIN:
                continue
            name = row["species"].strip()
            if name:
                names.add(name)
        found[method] = names
    return found


def venn_regions(sets):
    """MACO, NormCC, bin3C as A, B, C. Same partition as the sheep-gut Venn."""
    left, mid, right = (sets[name] for name in METHODS)
    return {
        "A": len(left - mid - right),
        "B": len(mid - left - right),
        "C": len(right - left - mid),
        "AB": len((left & mid) - right),
        "AC": len((left & right) - mid),
        "BC": len((mid & right) - left),
        "ABC": len(left & mid & right),
    }


def unique_counts(rows_by_method):
    species = {method: _names(rows_by_method[method], "species") for method in METHODS}
    genus = {method: _names(rows_by_method[method], "genus") for method in METHODS}
    counts = {}
    for method in METHODS:
        other_species = set().union(*(species[other] for other in METHODS if other != method))
        other_genus = set().union(*(genus[other] for other in METHODS if other != method))
        counts[method] = (
            len(species[method] - other_species),
            len(genus[method] - other_genus),
        )
    return counts


def compute():
    stacks = {}
    taxa = {}
    species = {}
    for platform, label in PLATFORMS:
        genome, _length = load_labels(platform)
        ref_len = load_ref_lengths(platform, genome)
        stacks[label] = {}
        taxa[label] = {}
        for cluster, cluster_label in CLUSTERS:
            stacks[label][cluster_label] = {}
            rows_by_method = {}
            for method in METHODS:
                rows = score_bins(cluster, platform, method, genome, ref_len)
                rows_by_method[method] = rows
                counts = band_counts(rows)
                stacks[label][cluster_label][method] = counts
                print(
                    "[BAND] {0} {1} {2}  {3}  total={4}".format(
                        label,
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
            taxa[label][cluster_label] = unique_counts(rows_by_method)
            if cluster_label == "Leiden":
                species[label] = species_sets(rows_by_method)
                regions = venn_regions(species[label])
                print(
                    "[VENN] {0} Leiden  {1}  totals={2}".format(
                        label,
                        " ".join(
                            "{0}={1}".format(key, regions[key])
                            for key in ("A", "B", "C", "AB", "AC", "BC", "ABC")
                        ),
                        " ".join(
                            "{0}={1}".format(method, len(species[label][method]))
                            for method in METHODS
                        ),
                    ),
                    flush=True,
                )
            for method in METHODS:
                n_species, n_genus = taxa[label][cluster_label][method]
                print(
                    "[TAXA] {0} {1} {2}  species={3}  genera={4}".format(
                        label, cluster_label, method, n_species, n_genus
                    ),
                    flush=True,
                )
    return stacks, taxa, species


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


def _frame(ax, letter):
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
    ax.text(
        -0.02,
        1.02,
        letter,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def _bar_panel(ax, platform, stacks, x_max, x_ticks):
    rows, spans, y_max = _bar_geometry()
    for y_pos, cluster, method in rows:
        left = 0.0
        for band_index, count in enumerate(stacks[platform][cluster][method]):
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
    ax.set_xlabel(
        "Bins (contamination ≤ {0:g}%)".format(CONT_MAX),
        fontsize=14,
        labelpad=6,
    )
    ax.tick_params(axis="y", length=0, pad=3)
    # One thick rule per cluster, the same stroke figure 2 uses under its groups.
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
    _frame(ax, "")
    return rows


def _venn_centres():
    """Same circle centres as the sheep-gut species Venn."""
    side = 1.05
    height = side * np.sqrt(3.0) / 2.0
    return {
        "A": np.array([-side / 2.0, height / 3.0]),
        "B": np.array([side / 2.0, height / 3.0]),
        "C": np.array([0.0, -2.0 * height / 3.0]),
    }


# Same hues as the method legend, lifted so the lenses stay readable.
# Pairwise colors are chosen rather than averaged, so red and teal do not turn gray.
VENN_REGION = {
    "A": "#F08B7E",
    "B": "#4EC4B4",
    "C": "#8EA6D6",
    "AB": "#E6C0A6",
    "AC": "#E0AFC6",
    "BC": "#7FBFD0",
    "ABC": "#C9A9BE",
}


def _venn_limits(centres, radius):
    xs = [float(centres[key][0]) for key in centres]
    ys = [float(centres[key][1]) for key in centres]
    pad = 0.30
    left = min(xs) - radius - pad
    right = max(xs) + radius + pad
    bottom = min(ys) - radius - pad
    top = max(ys) + radius + pad
    span = max(right - left, top - bottom)
    cx = 0.5 * (left + right)
    cy = 0.5 * (bottom + top)
    half = span / 2.0
    return (cx - half, cx + half, cy - half, cy + half)


def _venn_fill(ax, centres, radius, limits):
    """Paint each of the seven regions. Alpha blending turns the lenses gray."""
    left, right, bottom, top = limits
    n = 1700
    xs = np.linspace(left, right, n)
    ys = np.linspace(bottom, top, n)
    xx, yy = np.meshgrid(xs, ys)
    inside = {
        key: (xx - float(centres[key][0])) ** 2 + (yy - float(centres[key][1])) ** 2 <= (radius * radius)
        for key in ("A", "B", "C")
    }
    masks = {
        "A": inside["A"] & ~inside["B"] & ~inside["C"],
        "B": inside["B"] & ~inside["A"] & ~inside["C"],
        "C": inside["C"] & ~inside["A"] & ~inside["B"],
        "AB": inside["A"] & inside["B"] & ~inside["C"],
        "AC": inside["A"] & inside["C"] & ~inside["B"],
        "BC": inside["B"] & inside["C"] & ~inside["A"],
        "ABC": inside["A"] & inside["B"] & inside["C"],
    }
    image = np.ones((n, n, 4), dtype=np.float32)
    image[..., 3] = 0.0
    for key, mask in masks.items():
        red, green, blue = _rgb(VENN_REGION[key])
        image[mask, 0] = red
        image[mask, 1] = green
        image[mask, 2] = blue
        image[mask, 3] = 1.0
    ax.imshow(
        image,
        extent=(left, right, bottom, top),
        origin="lower",
        interpolation="bilinear",
        zorder=1,
        aspect="auto",
    )
    for key in ("A", "B", "C"):
        ax.add_patch(
            Circle(
                centres[key],
                radius,
                facecolor="none",
                edgecolor="black",
                linewidth=2.05,
                zorder=2,
            )
        )


def _venn_panel(ax, sets):
    """Three-set species Venn. Circle order is MACO, NormCC, bin3C."""
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
            fontsize=15,
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


def _shown_index(group):
    """MACO stays on a shared coordinate. Otherwise Leiden does."""
    cluster_rank = {label: index for index, (_key, label) in enumerate(CLUSTERS)}
    method_rank = {method: index for index, method in enumerate(METHODS)}

    def sort_key(index):
        cluster, method = group[index][0], group[index][1]
        return (
            0 if method == "MACO" else 1,
            0 if cluster == "Leiden" else 1,
            cluster_rank[cluster],
            method_rank[method],
        )

    return min(range(len(group)), key=sort_key)


def _callout_layout(n_points):
    pitch_x = 0.072
    pitch_y = 0.068
    pad_x = 0.018
    pad_y = 0.014
    ncol = n_points if n_points <= 4 else int(np.ceil(n_points / 2.0))
    nrow = int(np.ceil(n_points / float(ncol)))
    width = pad_x * 2.0 + ncol * pitch_x
    height = pad_y * 2.0 + nrow * pitch_y
    return width, height, ncol, nrow, pitch_x, pitch_y, pad_x, pad_y


def _rects_overlap(left, right, gap):
    ax0, ay0, aw, ah = left
    bx0, by0, bw, bh = right
    return ax0 < bx0 + bw + gap and ax0 + aw + gap > bx0 and ay0 < by0 + bh + gap and ay0 + ah + gap > by0


def _box_anchor(rect, target):
    x0, y0, width, height = rect
    cx = x0 + width / 2.0
    cy = y0 + height / 2.0
    dx = target[0] - cx
    dy = target[1] - cy
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return cx, cy
    scales = []
    if dx != 0.0:
        scales.append(((x0 if dx < 0.0 else x0 + width) - cx) / dx)
    if dy != 0.0:
        scales.append(((y0 if dy < 0.0 else y0 + height) - cy) / dy)
    scale = min(value for value in scales if value > 0.0)
    return cx + dx * scale, cy + dy * scale


def _segment_distance(point, start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length2 = dx * dx + dy * dy
    if length2 < 1e-12:
        return float(np.hypot(point[0] - start[0], point[1] - start[1]))
    weight = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length2
    weight = min(1.0, max(0.0, weight))
    return float(np.hypot(point[0] - (start[0] + weight * dx), point[1] - (start[1] + weight * dy)))


def _place_overflow_box(anchor, width, height, blocked, avoid):
    best = None
    best_dist = None
    x_stops = np.linspace(0.04, max(0.04, 0.96 - width), 26)
    y_stops = np.linspace(0.06, max(0.06, 0.90 - height), 26)
    for left in x_stops:
        for bottom in y_stops:
            rect = (float(left), float(bottom), width, height)
            if any(_rects_overlap(rect, other, 0.015) for other in blocked):
                continue
            edge = _box_anchor(rect, anchor)
            if any(_segment_distance(point, edge, anchor) < 0.07 for point in avoid):
                continue
            cx = left + width / 2.0
            cy = bottom + height / 2.0
            dist = float(np.hypot(cx - anchor[0], cy - anchor[1]))
            if best_dist is None or dist < best_dist:
                best = rect
                best_dist = dist
    if best is not None:
        return best
    return (0.08, max(0.08, 0.88 - height), width, height)


def _draw_overflow_box(ax, anchor_data, anchor_axes, hidden, rect):
    x0, y0, width, height = rect
    _width, _height, ncol, _nrow, pitch_x, pitch_y, pad_x, pad_y = _callout_layout(len(hidden))
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle="round,pad=0,rounding_size=0.012",
            transform=ax.transAxes,
            facecolor="#F2F2F2",
            edgecolor="#8A8A8A",
            linewidth=0.8,
            clip_on=False,
            zorder=5,
        )
    )
    edge_x, edge_y = _box_anchor(rect, anchor_axes)
    ax.annotate(
        "",
        xy=anchor_data,
        xycoords="data",
        xytext=(edge_x, edge_y),
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#8A8A8A",
            "lw": 0.9,
            "mutation_scale": 6,
            "shrinkA": 0.0,
            "shrinkB": 5.0,
        },
        zorder=4,
        clip_on=False,
    )
    for index, (cluster, method, _x, _y) in enumerate(hidden):
        col = index % ncol
        row = index // ncol
        cx = x0 + pad_x + (col + 0.5) * pitch_x
        cy = y0 + height - pad_y - (row + 0.5) * pitch_y
        ax.plot(
            [cx],
            [cy],
            linestyle="none",
            marker=SHAPES[cluster],
            transform=ax.transAxes,
            color=COLORS[method],
            markerfacecolor=COLORS[method],
            markeredgecolor="black",
            markeredgewidth=0.45,
            markersize=CALLOUT_MARKER[cluster],
            clip_on=False,
            zorder=6,
        )


def _scatter_panel(ax, platform, taxa, x_lim, y_lim):
    x_lo, x_hi, x_ticks = x_lim
    y_lo, y_hi, y_ticks = y_lim
    points = []
    for _key, cluster in CLUSTERS:
        for method in METHODS:
            n_species, n_genus = taxa[platform][cluster][method]
            points.append((cluster, method, float(n_species), float(n_genus)))
    keyed = defaultdict(list)
    for item in points:
        keyed[(round(item[2], 5), round(item[3], 5))].append(item)
    placed = []
    overflows = []
    for group in keyed.values():
        keep = _shown_index(group)
        cluster, method, x_mean, y_mean = group[keep]
        on_edge = x_mean <= x_lo + 0.05 or y_mean <= y_lo + 0.05
        ax.plot(
            [x_mean],
            [y_mean],
            linestyle="none",
            marker=SHAPES[cluster],
            color=COLORS[method],
            markerfacecolor=COLORS[method],
            markeredgecolor="black",
            markeredgewidth=0.9,
            markersize=MARKER_SIZE[cluster],
            clip_on=not on_edge,
            zorder=4,
        )
        placed.append((x_mean, y_mean))
        if len(group) > 1:
            hidden = [item for index, item in enumerate(group) if index != keep]
            cluster_rank = {label: index for index, (_key, label) in enumerate(CLUSTERS)}
            method_rank = {method: index for index, method in enumerate(METHODS)}
            hidden.sort(key=lambda item: (cluster_rank[item[0]], method_rank[item[1]]))
            overflows.append(((x_mean, y_mean), hidden))
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xlabel("Unique species", fontsize=14, labelpad=6)
    ax.set_ylabel("Unique genera", fontsize=14, labelpad=6)
    _frame(ax, "")
    return placed, overflows


def _callouts(ax, placed, overflows):
    blocked = []
    marker_axes = []
    for x_mean, y_mean in placed:
        fx, fy = ax.transAxes.inverted().transform(ax.transData.transform((x_mean, y_mean)))
        marker_axes.append((float(fx), float(fy)))
        blocked.append((float(fx) - 0.05, float(fy) - 0.05, 0.10, 0.10))
    for anchor_data, hidden in overflows:
        width, height, _ncol, _nrow, _px, _py, _padx, _pady = _callout_layout(len(hidden))
        fx, fy = ax.transAxes.inverted().transform(ax.transData.transform(anchor_data))
        anchor_axes = (float(fx), float(fy))
        avoid = [point for point in marker_axes if float(np.hypot(point[0] - fx, point[1] - fy)) > 0.04]
        rect = _place_overflow_box(anchor_axes, width, height, blocked, avoid)
        _draw_overflow_box(ax, anchor_data, anchor_axes, hidden, rect)
        blocked.append(rect)


def _taxon_limits(values):
    low = 0.0
    high = max(values) if values else 1.0
    span = max(high - low, 1.0)
    if span >= 40.0:
        step = 10.0
    elif span >= 12.0:
        step = 5.0
    else:
        step = 2.0
    hi = step * np.ceil((high + max(span * 0.08, 1.0)) / step)
    if hi <= high:
        hi += step
    ticks = [float(tick) for tick in np.arange(0.0, hi + step * 0.01, step)]
    return 0.0, float(hi), ticks


def _count_axis(peak):
    step = 10.0 if peak > 25 else 5.0
    top = step * np.ceil((peak * 1.08) / step)
    if top <= peak:
        top += step
    ticks = [float(tick) for tick in np.arange(0.0, top + 0.01, step)]
    return float(top), ticks


def _legend(fig):
    methods = [
        Patch(facecolor=COLORS[method], edgecolor="black", linewidth=0.7, label=method)
        for method in METHODS
    ]
    shapes = [
        Line2D(
            [0],
            [0],
            marker=SHAPES[cluster],
            color="none",
            markerfacecolor="0.25",
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=LEGEND_MARKER[cluster],
            linestyle="None",
            label=cluster,
        )
        for _key, cluster in CLUSTERS
    ]
    bands = [
        Patch(
            facecolor=_tint(COLORS["MACO"], white),
            edgecolor="black",
            linewidth=0.6,
            label=label,
        )
        for (_low, _high, label), white in zip(BANDS, BAND_WHITE)
    ]
    method_legend = fig.legend(
        handles=methods + shapes,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.048),
        bbox_transform=fig.transFigure,
        ncol=6,
        frameon=False,
        fontsize=13,
        handlelength=1.15,
        handleheight=0.85,
        columnspacing=1.15,
        handletextpad=0.45,
        borderaxespad=0.0,
    )
    fig.add_artist(method_legend)
    fig.legend(
        handles=bands,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        bbox_transform=fig.transFigure,
        ncol=len(BANDS),
        frameon=False,
        fontsize=12,
        handlelength=1.35,
        handleheight=0.85,
        columnspacing=0.95,
        borderaxespad=0.0,
    )


def _letters(fig, axes):
    for ax, letter in zip(axes, ("a", "b", "c", "d", "e", "f")):
        if letter in ("a", "d"):
            continue
        ax.text(
            -0.02,
            1.03,
            letter,
            transform=ax.transAxes,
            fontsize=PANEL_FONT,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )


def _bar_letters(fig, drawn):
    """Put a and d at the left of the cluster names, not the axes spine."""
    renderer = fig.canvas.get_renderer()
    for ax, letter in drawn:
        if letter not in ("a", "d"):
            continue
        left = None
        for text in ax.texts:
            if text.get_gid() != "cluster-label":
                continue
            box = text.get_window_extent(renderer)
            left = box.x0 if left is None else min(left, box.x0)
        ax_x, _ax_y = ax.transAxes.inverted().transform((left, 0.0))
        ax.text(
            ax_x - 0.02,
            1.03,
            letter,
            transform=ax.transAxes,
            fontsize=PANEL_FONT,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )


def draw(stacks, taxa, species, path):
    _style()
    fig_w = LEFT + BAR_W + GAP + VENN + GAP_SCAT + SCAT + RIGHT
    fig_h = TOP + BAR_H + MID + BAR_H + BOTTOM
    peak = max(
        sum(stacks[label][cluster][method])
        for label in stacks
        for cluster in stacks[label]
        for method in stacks[label][cluster]
    )
    x_max, x_ticks = _count_axis(peak)
    species_values = []
    genus_values = []
    for label in taxa:
        for cluster in taxa[label]:
            for method in taxa[label][cluster]:
                n_species, n_genus = taxa[label][cluster][method]
                species_values.append(n_species)
                genus_values.append(n_genus)
    x_lim = _taxon_limits(species_values)
    y_lim = _taxon_limits(genus_values)
    fig = plt.figure(figsize=(fig_w, fig_h))
    bar_width = BAR_W / fig_w
    venn_width = VENN / fig_w
    scat_width = SCAT / fig_w
    height = BAR_H / fig_h
    x_bar = LEFT / fig_w
    x_venn = (LEFT + BAR_W + GAP) / fig_w
    x_scat = (LEFT + BAR_W + GAP + VENN + GAP_SCAT) / fig_w
    y_lo = BOTTOM / fig_h
    y_hi = (BOTTOM + BAR_H + MID) / fig_h
    panels = (
        (x_bar, y_hi, bar_width, "a", "ONT", "bar"),
        (x_venn, y_hi, venn_width, "b", "ONT", "venn"),
        (x_scat, y_hi, scat_width, "c", "ONT", "taxa"),
        (x_bar, y_lo, bar_width, "d", "PacBio", "bar"),
        (x_venn, y_lo, venn_width, "e", "PacBio", "venn"),
        (x_scat, y_lo, scat_width, "f", "PacBio", "taxa"),
    )
    drawn = []
    scatters = []
    anchors = {}
    for x, y, width, letter, platform, kind in panels:
        ax = fig.add_axes([x, y, width, height])
        anchors[letter] = ax.get_position().frozen()
        if kind == "bar":
            _bar_panel(ax, platform, stacks, x_max, x_ticks)
        elif kind == "venn":
            _venn_panel(ax, species[platform])
        else:
            placed, overflows = _scatter_panel(ax, platform, taxa, x_lim, y_lim)
            scatters.append((ax, placed, overflows))
        drawn.append((ax, letter))
    _letters(fig, [ax for ax, _letter in drawn])
    _legend(fig)
    fig.canvas.draw()
    for ax, letter in drawn:
        ax.set_position(anchors[letter])
    fig.canvas.draw()
    _bar_letters(fig, drawn)
    for ax, placed, overflows in scatters:
        _callouts(ax, placed, overflows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    plt.close(fig)
    print("[FIG] {0}".format(path), flush=True)


if __name__ == "__main__":
    print("[GATE] contamination <= {0:g}%".format(CONT_MAX), flush=True)
    draw(*compute(), OUT)
