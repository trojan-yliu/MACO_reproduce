#!/usr/bin/env python3
"""Figure 2. Long-read scores and per-bin quality on all MOCK1 bins.

Scores and ratios are computed from the MOCK1 tables in data/. Nothing in
the figure is a typed-in number.

Panels a and c: precision is the length-weighted majority purity of every bin.
Recall keeps, for each genome, only the longest bin in which that genome is
the majority, divided by all labeled bases of that genome. F1 is the harmonic
mean. ARI is the Hubert-Arabie index after that same projection: the selected
piece keeps its bin, and unrecovered bases are singletons. NMI uses one shared
unrecovered state instead of those singletons.

Panels b and d: every labeled bin is drawn. Completeness is
the majority genome's labeled length divided by its reference length.
Contamination is the remaining fraction of the bin. Both are percentages.
The axis is log10 of their ratio. A contamination of 0 is entered as 0.01.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE.parent / "Main_Figures" / "Main_Figure_2.png"
_PLATFORM = {"ont": "ONT", "pacbio": "PacBio"}
_CLUSTER = {"metacc": "Leiden", "infomap": "Infomap", "imputecc": "ImputeCC"}
_DATA = None

METHODS = ("MACO", "NormCC", "bin3C")
CLUSTERS = (
    ("metacc", "Leiden"),
    ("infomap", "Infomap"),
    ("imputecc", "ImputeCC"),
)
PLATFORMS = (("ont", "ONT"), ("pacbio", "PacBio"))
METRICS = ("F1", "ARI", "NMI")
COLORS = {"MACO": "#E64B35", "NormCC": "#00A087", "bin3C": "#3C5488"}

Y_MIN = 0.5
Y_MAX = 1.0
BOX_Y_MIN = -1.55
BOX_Y_MAX = 4.65
BOX_TICKS = (-1, 0, 1, 2, 3, 4)

# Each panel is its own square. Two rows, bars on the left and boxes on the right.
AX = 3.05
LEFT = 0.52
GAP = 0.66
RIGHT = 0.06
TOP = 0.42
MID = 0.78
BOTTOM = 0.78


def _font():
    names = {font.name for font in font_manager.fontManager.ttflist}
    for cand in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"):
        if cand in names:
            return cand
    return "DejaVu Sans"


def _data():
    """Contig labels, reference lengths, and bin membership from data/."""
    global _DATA
    if _DATA is not None:
        return _DATA
    paths = {
        "labels": DATA / "contig_labels.csv",
        "refs": DATA / "reference_lengths.csv",
        "bins": DATA / "bin_membership.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise SystemExit("[ERROR] missing {0}".format(path))
    labels = {name: {} for name in _PLATFORM.values()}
    with paths["labels"].open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels[row["Platform"]][row["Contig"]] = (
                int(row["Length (bp)"]),
                row["Genome"].strip(),
                row["Status"].strip(),
            )
    refs = {}
    with paths["refs"].open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            refs[row["Genome"]] = int(row["Length (bp)"])
    bins = {}
    with paths["bins"].open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["Platform"], row["Clustering"], row["Normalization"])
            bins.setdefault(key, {}).setdefault(row["Bin"], []).append(row["Contig"])
    _DATA = (labels, refs, bins)
    return _DATA


def load_labels(platform):
    labels, _refs, _bins = _data()
    table = labels[_PLATFORM[platform]]
    genome = {}
    length = {}
    for cid, (size, name, status) in table.items():
        length[cid] = size
        if status == "ok" and name:
            genome[cid] = name
    if not genome:
        raise SystemExit("[ERROR] no labeled contigs for {0}".format(platform))
    return genome, length


def load_ref_lengths(platform, genome):
    """Reference genome lengths. The same 66 genomes are used for ONT and PacBio."""
    _labels, refs, _bins = _data()
    if not refs:
        raise SystemExit("[ERROR] no reference lengths in {0}".format(DATA / "reference_lengths.csv"))
    missing = sorted({name for name in set(genome.values()) if name not in refs})
    if missing:
        raise SystemExit(
            "[ERROR] {0} genomes missing reference lengths: {1}".format(
                platform, ", ".join(missing[:8])
            )
        )
    return refs


def bin_records(cluster, platform, method):
    """(bin id, [(contig, length), ...]) for one clustering and normalization."""
    labels, _refs, bins = _data()
    key = (_PLATFORM[platform], _CLUSTER[cluster], method)
    grouped = bins.get(key)
    if not grouped:
        raise SystemExit("[ERROR] no bins for {0} {1} {2}".format(platform, cluster, method))
    table = labels[_PLATFORM[platform]]
    records = []
    for bid in sorted(grouped):
        contigs = []
        for cid in grouped[bid]:
            if cid not in table:
                raise SystemExit("[ERROR] contig absent from labels: {0}".format(cid))
            contigs.append((cid, table[cid][0]))
        records.append((bid, contigs))
    return records


def read_bins(cluster, platform, method):
    return [(bid, [cid for cid, _length in contigs]) for bid, contigs in bin_records(cluster, platform, method)]


def bin_quality(cluster, platform, method, genome, ref_len):
    """log10(completeness / contamination) for every labeled bin. No quality gate."""
    values = []
    for _bid, records in bin_records(cluster, platform, method):
        total = sum(length for _cid, length in records)
        by_genome = defaultdict(int)
        for cid, length in records:
            name = genome.get(cid)
            if name:
                by_genome[name] += length
        if total <= 0 or not by_genome:
            continue
        majority = max(by_genome, key=by_genome.get)
        labeled = by_genome[majority]
        reference = ref_len[majority]
        completeness = 100.0 * labeled / float(reference)
        contamination = 100.0 * (total - labeled) / float(total)
        values.append(math.log10(completeness / max(contamination, 0.01)))
    if not values:
        raise SystemExit("[ERROR] no labeled bins for {0} {1} {2}".format(platform, cluster, method))
    return values


def score(genome, length, bin_list):
    labeled = defaultdict(int)
    for cid, name in genome.items():
        labeled[name] += length[cid]
    pieces = []
    seen = set()
    for bid, contigs in bin_list:
        lam = defaultdict(int)
        total = 0
        for cid in contigs:
            if cid in seen:
                raise SystemExit("[ERROR] contig in two bins: {0}".format(cid))
            seen.add(cid)
            if cid not in length:
                raise SystemExit("[ERROR] contig absent from ground truth: {0}".format(cid))
            total += length[cid]
            name = genome.get(cid)
            if name:
                lam[name] += length[cid]
        if not lam or total <= 0:
            if total > 0:
                pieces.append((bid, None, 0, total))
            continue
        majority = min((-lam[name], name) for name in lam)[1]
        pieces.append((bid, majority, lam[majority], total))

    best = {}
    best_bid = {}
    for bid, majority, pure, _total in pieces:
        if majority is None:
            continue
        previous = best.get(majority, -1)
        if pure > previous or (pure == previous and bid < best_bid.get(majority, "~")):
            best[majority] = pure
            best_bid[majority] = bid

    sum_labeled = sum(labeled.values())
    recovered = {}
    for name, genome_len in labeled.items():
        recovered[name] = min(best.get(name, 0), genome_len)
    sum_pure = sum(pure for _bid, majority, pure, _total in pieces if majority is not None)
    sum_total = sum(total for _bid, _majority, _pure, total in pieces)
    precision = (sum_pure / sum_total) if sum_total else 0.0
    recall = (sum(recovered.values()) / sum_labeled) if sum_labeled else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    gold_pairs = 0
    true_pairs = 0
    for name, genome_len in labeled.items():
        gold_pairs += genome_len * (genome_len - 1) // 2
        piece = recovered[name]
        true_pairs += piece * (piece - 1) // 2
    mass = defaultdict(int)
    for name, piece in recovered.items():
        if piece > 0:
            mass[best_bid[name]] += piece
    pred_pairs = 0
    for size in mass.values():
        pred_pairs += size * (size - 1) // 2
    choose_all = sum_labeled * (sum_labeled - 1) // 2
    numer = 2 * true_pairs * choose_all - 2 * gold_pairs * pred_pairs
    denom = (gold_pairs + pred_pairs) * choose_all - 2 * gold_pairs * pred_pairs
    if denom == 0:
        ari = 1.0 if true_pairs == gold_pairs == pred_pairs and gold_pairs > 0 else 0.0
    else:
        ari = numer / denom

    unrecovered = sum_labeled - sum(recovered.values())
    h_genome = 0.0
    for genome_len in labeled.values():
        if genome_len:
            p = genome_len / sum_labeled
            h_genome -= p * math.log(p)
    h_bin = 0.0
    if unrecovered:
        p = unrecovered / sum_labeled
        h_bin -= p * math.log(p)
    for size in mass.values():
        if size:
            p = size / sum_labeled
            h_bin -= p * math.log(p)
    mutual = 0.0
    for name, genome_len in labeled.items():
        p_genome = genome_len / sum_labeled
        piece = recovered[name]
        if piece:
            joint = piece / sum_labeled
            p_bin = mass[best_bid[name]] / sum_labeled
            mutual += joint * math.log(joint / (p_genome * p_bin))
        left = genome_len - piece
        if left and unrecovered:
            joint = left / sum_labeled
            p_sink = unrecovered / sum_labeled
            mutual += joint * math.log(joint / (p_genome * p_sink))
    nmi = (2 * mutual / (h_genome + h_bin)) if (h_genome + h_bin) else 0.0
    return {
        "F1": f1,
        "ARI": ari,
        "NMI": nmi,
        "Precision": precision,
        "Recall": recall,
        "bins": len(pieces),
    }


def _toy():
    genome = {"a": "A", "b": "B"}
    length = {"a": 10, "b": 10, "u": 5}
    perfect = score(genome, length, [("b0", ["a"]), ("b1", ["b"])])
    if abs(perfect["F1"] - 1) > 1e-12 or abs(perfect["ARI"] - 1) > 1e-12:
        raise SystemExit("[ERROR] perfect-partition check failed")
    dirty = score(genome, length, [("b0", ["a", "u"]), ("b1", ["b"])])
    if abs(dirty["F1"] - (2 * 0.8 * 1 / 1.8)) > 1e-12:
        raise SystemExit("[ERROR] contamination check failed")
    empty = score(genome, length, [])
    if empty["F1"] != 0 or abs(empty["ARI"]) > 1e-12 or abs(empty["NMI"]) > 1e-12:
        raise SystemExit("[ERROR] empty-binning check failed")


def compute():
    _toy()
    scores = {}
    ratios = {}
    for platform, label in PLATFORMS:
        genome, length = load_labels(platform)
        ref_len = load_ref_lengths(platform, genome)
        scores[label] = {}
        ratios[label] = {}
        for cluster, cluster_label in CLUSTERS:
            column = {metric: [] for metric in METRICS}
            ratios[label][cluster_label] = {}
            for method in METHODS:
                result = score(genome, length, read_bins(cluster, platform, method))
                quality = bin_quality(cluster, platform, method, genome, ref_len)
                for metric in METRICS:
                    column[metric].append(result[metric])
                ratios[label][cluster_label][method] = quality
                print(
                    "[SCORE] {0} {1} {2}  F1={3:.6f}  ARI={4:.6f}  NMI={5:.6f}  bins={6}  quality_n={7}".format(
                        label,
                        cluster_label,
                        method,
                        result["F1"],
                        result["ARI"],
                        result["NMI"],
                        result["bins"],
                        len(quality),
                    ),
                    flush=True,
                )
            scores[label][cluster_label] = column
    return scores, ratios


def _style():
    font = _font()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font, "Arial", "Helvetica", "DejaVu Sans"],
            "font.weight": "normal",
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.labelweight": "normal",
            "axes.linewidth": 0.6,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
        }
    )


def _layout(ax_w):
    """Fill the square. Bars in one triplet share an edge."""
    side = 0.04
    metric_gap = 0.055
    cluster_gap = 0.11
    usable = ax_w - 2.0 * side
    bar = (usable - 6.0 * metric_gap - 2.0 * cluster_gap) / 27.0
    groups = []
    pos = side
    for _key, cluster in CLUSTERS:
        metrics = []
        block_left = pos
        for metric in METRICS:
            left = pos
            center = left + 1.5 * bar
            metrics.append((metric, left, center, bar))
            pos += 3.0 * bar + metric_gap
        block_right = pos - metric_gap
        groups.append((cluster, metrics, block_left, block_right))
        pos = block_right + cluster_gap
    return groups, ax_w


def _panel(ax, platform, scores, groups, x_right, letter, ax_h):
    # Fills overlap by a hair so the shared edge has no white slit.
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
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
    ax.set_xticks([])
    ax.set_ylabel("Score")
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


def _quantile(values, probability):
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return ordered[low]
    weight = index - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _jitter(index):
    raw = math.sin(index * 12.9898) * 43758.5453
    return raw - math.floor(raw) - 0.5


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


def _ratio_panel(ax, platform, ratios, groups, x_right, letter, ax_h):
    """One box per method, centered on the same cluster brackets as the bars."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    rule_y = -0.185 / ax_h
    name_y = -0.235 / ax_h
    box_w = 0.168
    step = 0.228
    cap = 0.40 * box_w
    ax.axhline(0.0, color="0.78", lw=0.4, linestyle=(0, (1.3, 1.5)), zorder=1)
    for cluster_index, (cluster, _metrics, block_left, block_right) in enumerate(groups):
        block_center = 0.5 * (block_left + block_right)
        origin = block_center - 0.5 * step * (len(METHODS) - 1)
        for method_index, method in enumerate(METHODS):
            values = ratios[platform][cluster][method]
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
                    edgecolor="black",
                    linewidth=1.35,
                    zorder=3,
                )
            )
            ax.plot(
                [x - 0.5 * box_w, x + 0.5 * box_w],
                [median, median],
                color="black",
                lw=1.25,
                solid_capstyle="butt",
                zorder=5,
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
            block_center,
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
    ax.set_ylim(BOX_Y_MIN, BOX_Y_MAX)
    ax.set_yticks(list(BOX_TICKS))
    ax.set_xticks([])
    ax.set_ylabel(r"$\log_{10}$ (completeness / contamination)")
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


def draw(scores, ratios, path):
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
        (x_left, y_hi, "a", "ONT", "bar"),
        (x_box, y_hi, "b", "ONT", "ratio"),
        (x_left, y_lo, "c", "PacBio", "bar"),
        (x_box, y_lo, "d", "PacBio", "ratio"),
    )
    for x, y, letter, platform, kind in placed:
        ax = fig.add_axes([x, y, width, height])
        if kind == "bar":
            _panel(ax, platform, scores, groups, x_lim, letter, AX)
        else:
            _ratio_panel(ax, platform, ratios, groups, x_lim, letter, AX)
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
