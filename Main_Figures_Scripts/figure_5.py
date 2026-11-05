#!/usr/bin/env python3
"""Main Figure 5. Host range of the sheep-gut plasmidome.

Reads Supplementary Table S6 and writes Main_Figure_5.png.
Paths are relative to this file.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import fontManager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle, Wedge

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "Main_Figures" / "Main_Figure_5.png"
TABLE = HERE.parent / "Supplementary_Tables" / "Supplementary_Table_S6.xlsx"

ALGOS = ("Leiden", "Infomap", "ImputeCC")
CLUSTER_COLOR = {"Leiden": "#E69F00", "Infomap": "#56B4E9", "ImputeCC": "#CC79A7"}
PHYLUM_COLOR = {
    "Bacillota": "#6A51A3",
    "Bacteroidota": "#2CA25F",
    "Pseudomonadota": "#D94801",
    "Campylobacterota": "#CE1256",
    "Spirochaetota": "#4292C6",
    "Other": "#969696",
}
PHYLUM_GROUP = {"Bacillota_A": "Bacillota", "Bacillota_I": "Bacillota", "Bacillota_C": "Bacillota"}
MAG_COMP = 50.0
MAG_CONT = 5.0
INK = "#242424"
IN_ALL = "#2C3E66"
ONE_ONLY = "#E8927C"
ACCENT = "#E8961E"
ACCENT_DK = "#B86E07"
SHOWCASE = ("Escherichia coli", "Ruminococcus sp017513795", "Lawsonibacter phocaeensis_A")
# Share of the sheep-gut assembly stated in the published figure. The assembly
# length itself is not a column of Supplementary Table S6.
ASSEMBLY_PERCENT = 1.8

FIG_W = 18.7
RING_R0 = 1.02
RING_BAR = 1.28
RING_PAD = 0.28
RING_R = RING_R0 + RING_BAR + RING_PAD
GAP_CENTER = 106.0
GAP_DEG = 18.0
DISC_LEFT = 0.36
DISC_SIZE = 12.15
DISC_BOTTOM = 0.52
DISC_TOP = DISC_BOTTOM + DISC_SIZE
FIG_H = DISC_TOP + 0.56
BOX_A = (DISC_LEFT, DISC_BOTTOM, DISC_SIZE, DISC_SIZE)
_COL_TOP = DISC_TOP
_COL_BOT = DISC_BOTTOM
RIGHT_L = DISC_LEFT + DISC_SIZE + 0.78
RIGHT_W = FIG_W - RIGHT_L - 0.16
_COL_GAP = 0.16
_COL_H = (_COL_TOP - _COL_BOT - 2 * _COL_GAP) / 3
BOX_D = (RIGHT_L, _COL_BOT, RIGHT_W, _COL_H)
BOX_C = (RIGHT_L, _COL_BOT + _COL_H + _COL_GAP, RIGHT_W, _COL_H)
BOX_B = (RIGHT_L, _COL_BOT + 2 * (_COL_H + _COL_GAP), RIGHT_W, _COL_H)
A_LEGEND_TOP = 1.66
D_LEGEND_TOP = DISC_BOTTOM - 0.14
EDGE_LW = 0.6


def _font():
    installed = {font.name for font in fontManager.ttflist}
    for name in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"):
        if name in installed:
            return name
    return "DejaVu Sans"


def _style():
    face = _font()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [face, "DejaVu Sans"],
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.linewidth": 1.05,
        "axes.edgecolor": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "legend.frameon": False,
        "hatch.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "mathtext.fontset": "custom",
        "mathtext.it": face + ":italic",
        "mathtext.rm": face,
        "mathtext.bf": face + ":bold",
        "mathtext.sf": face,
    })


def _rgb(hex_color):
    text = hex_color.lstrip("#")
    return tuple(int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _tint(hex_color, white):
    red, green, blue = _rgb(hex_color)
    return (
        red + (1.0 - red) * white,
        green + (1.0 - green) * white,
        blue + (1.0 - blue) * white,
    )


def _box(spec):
    x, y, width, height = spec
    return (x / FIG_W, y / FIG_H, width / FIG_W, height / FIG_H)


def _polar(radius, degrees):
    theta = math.radians(degrees)
    return radius * math.cos(theta), radius * math.sin(theta)


def _bezier(p0, c0, c1, p1, n=48):
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0, c0, c1, p1 = (np.asarray(p, float) for p in (p0, c0, c1, p1))
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * c0 + 3 * (1 - t) * (t ** 2) * c1 + (t ** 3) * p1


def _ribbon(ax, p0, c0, c1, p1, w0, w1, color, alpha=0.5, zorder=1, edge="none", elw=0.4):
    p0, c0, c1, p1 = (np.asarray(p, float) for p in (p0, c0, c1, p1))

    def perp(a, b):
        delta = np.asarray(b, float) - np.asarray(a, float)
        length = math.hypot(delta[0], delta[1]) or 1.0
        return np.array([-delta[1] / length, delta[0] / length])

    n0 = perp(p0, c0)
    n1 = perp(c1, p1)
    upper = _bezier(p0 + n0 * w0, c0 + n0 * w0, c1 + n1 * w1, p1 + n1 * w1)
    lower = _bezier(p1 - n1 * w1, c1 - n1 * w1, c0 - n0 * w0, p0 - n0 * w0)
    ax.add_patch(plt.Polygon(
        np.vstack([upper, lower]), closed=True, facecolor=color, edgecolor=edge,
        linewidth=elw, alpha=alpha, zorder=zorder, joinstyle="round"))


def _italic(species):
    if not species:
        return ""
    parts = species.split(" ")
    base, _sep, suffix = parts[0].partition("_")
    out = "$\\mathit{" + base + "}$" + (("_" + suffix) if suffix else "")
    if len(parts) > 1:
        rest = parts[1]
        if rest and rest[0].islower() and not any(ch.isdigit() for ch in rest):
            out += " $\\mathit{" + rest + "}$ " + " ".join(parts[2:])
        else:
            out += " " + " ".join(parts[1:])
    return out.rstrip()


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    if text.lower() in ("", "nan", "none", "-"):
        return ""
    return text


def _column(frame, prefix):
    hits = [name for name in frame.columns if str(name).startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit("[ERROR] column {0} matched {1}".format(prefix, hits))
    return hits[0]


def _phylum_name(raw):
    name = PHYLUM_GROUP.get(raw, raw)
    return name if name in PHYLUM_COLOR else "Other"


def _is_mag(completeness, contamination):
    return completeness >= MAG_COMP and 0 < contamination <= MAG_CONT


def load():
    """Sheep-gut plasmid contigs in Supplementary Table S6."""
    frame = pd.read_excel(TABLE, sheet_name="Table S6", header=2)
    frame = frame[frame["Dataset"] == "Sheep gut"].copy()
    if frame.empty:
        raise SystemExit("[ERROR] Supplementary Table S6 has no Sheep gut rows")
    length_col = _column(frame, "Length (bp)")
    cluster_col = _column(frame, "MOB-suite cluster")
    mash_col = _column(frame, "Mash distance")
    relax_col = _column(frame, "Relaxase")
    nbin_col = _column(frame, "Clusterers with a bin")
    agreed_col = _column(frame, "Agreed host species")
    fields = {}
    for algo in ALGOS:
        fields[algo] = {
            "species": _column(frame, algo + " species"),
            "phylum": _column(frame, algo + " phylum"),
            "completeness": _column(frame, algo + " completeness"),
            "contamination": _column(frame, algo + " contamination"),
            "call": _column(frame, algo + " Hi-C call"),
            "top": _column(frame, algo + " top species"),
            "share": _column(frame, algo + " normalized-contact share"),
        }

    bp = {algo: defaultdict(float) for algo in ALGOS}
    core = defaultdict(float)
    phy_bp = {algo: defaultdict(lambda: defaultdict(float)) for algo in ALGOS}
    quality = defaultdict(lambda: defaultdict(float))
    ann = []
    linked = Counter()
    contact = Counter()
    total_bp = 0.0
    binned = 0
    all_three = 0
    agreed = 0
    for record in frame.to_dict("records"):
        length = _num(record[length_col])
        total_bp += length
        n_bin = int(_num(record[nbin_col]))
        if n_bin >= 1:
            binned += 1
        if n_bin == 3:
            all_three += 1
        if _clean(record[agreed_col]):
            agreed += 1
        species = {}
        calls = {}
        fracs = {}
        for algo in ALGOS:
            col = fields[algo]
            name = _clean(record[col["species"]])
            species[algo] = name
            call = _clean(record[col["call"]])
            calls[algo] = call
            if call == "linked":
                linked[algo] += 1
            elif call == "contact_host":
                contact[algo] += 1
            top = _clean(record[col["top"]])
            share = record[col["share"]]
            share_v = None if _clean(share) == "" else _num(share)
            if call == "linked" and share_v is not None:
                fracs[algo] = share_v / 100.0
            elif name and name == top and share_v is not None:
                fracs[algo] = share_v / 100.0
            else:
                fracs[algo] = None
            if name:
                bp[algo][name] += length
                phy_bp[algo][name][_phylum_name(_clean(record[col["phylum"]]))] += length
                if algo == "Leiden":
                    quality[name][(_num(record[col["completeness"]]), _num(record[col["contamination"]], 1.0e9))] += length
        names = [species[algo] for algo in ALGOS]
        if names[0] and names[0] == names[1] == names[2]:
            core[names[0]] += length
        ann.append({
            "length": length,
            "cluster": _clean(record[cluster_col]),
            "mash": _num(record[mash_col]),
            "relaxase": _clean(record[relax_col]),
            "species": species,
            "call": calls,
            "frac": fracs,
        })

    species_rows = []
    phylum = {}
    names = set(core) | set().union(*(bp[algo] for algo in ALGOS))
    for name in names:
        union = max(bp[algo][name] for algo in ALGOS)
        if union <= 0:
            continue
        pool = phy_bp["Leiden"][name]
        if not pool:
            pool = Counter()
            for algo in ALGOS:
                pool.update(phy_bp[algo][name])
        phy = max(pool, key=pool.get) if pool else "Other"
        phylum[name] = phy
        if quality[name]:
            (completeness, contamination), _weight = max(quality[name].items(), key=lambda item: item[1])
        else:
            completeness, contamination = 0.0, 1.0e9
        species_rows.append({
            "species": name,
            "bp_in_all_three": core[name],
            "completeness": completeness,
            "contamination": contamination,
            **{algo + "_bp": bp[algo][name] for algo in ALGOS},
        })
    summary = {
        "plasmid_contigs": len(ann),
        "plasmid_contigs_in_any_bin": binned,
        "host_agreed_n": agreed,
        "all_three": all_three,
        "total_bp": total_bp,
    }
    for algo in ALGOS:
        summary[algo + "_hic_linked"] = linked[algo]
        summary[algo + "_hic_contact_host"] = contact[algo]
    return {"summary": summary, "species": species_rows, "ann": ann, "phylum": phylum}


def _host_species(data):
    rows = []
    for row in data["species"]:
        name = row["species"].strip()
        if not name:
            continue
        union = max(_num(row.get(algo + "_bp")) for algo in ALGOS)
        if union <= 0:
            continue
        core = _num(row.get("bp_in_all_three"))
        rows.append({
            "name": name,
            "core": core / 1.0e6,
            "union": union / 1.0e6,
            "phylum": data["phylum"].get(name, "Other"),
            "mag": _is_mag(_num(row.get("completeness")), _num(row.get("contamination"), 1.0e9)),
        })
    order = list(PHYLUM_COLOR)
    rows.sort(key=lambda row: (order.index(row["phylum"]), -row["union"], row["name"]))
    return rows


def _broad_cluster(data):
    """Non-orphan MOB cluster linked to the most host species."""
    by = {}
    for row in data["ann"]:
        cluster = row["cluster"]
        if cluster in ("", "-"):
            continue
        bucket = by.setdefault(cluster, {"n": 0, "orphan": 0, "novel": 0, "rows": [], "hosts": defaultdict(dict)})
        bucket["n"] += 1
        bucket["rows"].append(row)
        if row["mash"] >= 1.0:
            bucket["orphan"] += 1
        if row["mash"] > 0.1:
            bucket["novel"] += 1
        for algo in ALGOS:
            if row["call"][algo] != "linked":
                continue
            species = row["species"][algo]
            if not species:
                continue
            frac = row["frac"][algo]
            bucket["hosts"][species][algo] = 0.0 if frac is None else frac
    best = None
    for cluster, bucket in by.items():
        if bucket["n"] and bucket["orphan"] / bucket["n"] > 0.5:
            continue
        if len(bucket["hosts"]) < 2:
            continue
        if best is not None and len(bucket["hosts"]) <= len(by[best]["hosts"]):
            continue
        best = cluster
    if best is None:
        return None, None
    bucket = by[best]
    bucket["lengths"] = [row["length"] for row in bucket["rows"]]
    anchors = {}
    for species in bucket["hosts"]:
        pick = None
        pick_fr = -1.0
        for row in bucket["rows"]:
            for algo in ALGOS:
                if row["call"][algo] != "linked" or row["species"][algo] != species:
                    continue
                frac = 0.0 if row["frac"][algo] is None else row["frac"][algo]
                if frac > pick_fr:
                    pick, pick_fr = row, frac
        if pick is None:
            continue
        anchors[species] = {
            "frac": float(np.median(list(bucket["hosts"][species].values()))),
            "mobv": "MOBV" in pick["relaxase"],
            "algs": sorted(bucket["hosts"][species]),
        }
    bucket["anchors"] = anchors
    return best, bucket


def _host_contigs(data, host):
    out = []
    for row in data["ann"]:
        hit = [algo for algo in ALGOS if row["species"][algo] == host]
        if not hit:
            continue
        fracs = [row["frac"][algo] for algo in hit if row["frac"][algo] is not None]
        out.append({
            "length": row["length"],
            "all": len(hit) == 3,
            "frac": float(np.median(fracs)) if fracs else 0.0,
        })
    out.sort(key=lambda item: item["frac"])
    return out


def _chords(ax, data, angle_of):
    _cluster, bundle = _broad_cluster(data)
    if not bundle:
        return
    hosts = [(name, info["frac"]) for name, info in bundle["anchors"].items() if name in angle_of]
    if len(hosts) < 2:
        return
    radius = 0.92
    for i in range(len(hosts)):
        for j in range(i + 1, len(hosts)):
            left, fi = hosts[i]
            right, fj = hosts[j]
            p0 = _polar(radius, angle_of[left])
            p1 = _polar(radius, angle_of[right])
            c0 = _polar(radius * 0.08, angle_of[left])
            c1 = _polar(radius * 0.08, angle_of[right])
            _ribbon(ax, p0, c0, c1, p1, 0.015 + 0.034 * fi, 0.015 + 0.034 * fj,
                     ACCENT, alpha=0.42, zorder=1, edge=ACCENT_DK, elw=0.4)
    for name, _frac in hosts:
        x, y = _polar(radius, angle_of[name])
        ax.plot([x], [y], marker="o", markersize=5.2, markerfacecolor=ACCENT,
                markeredgecolor="white", markeredgewidth=0.8, zorder=6)


def _panel_host_map(ax, data):
    rows = _host_species(data)
    _cluster, bundle = _broad_cluster(data)
    bold = set(bundle["anchors"]) if bundle else set()
    n = len(rows)
    gap = GAP_DEG
    step = (360.0 - gap) / n
    start = GAP_CENTER + gap / 2.0 + step / 2.0
    top = max(row["union"] for row in rows) * 1.06
    angles = [start + i * step for i in range(n)]
    angle_of = {row["name"]: angle for angle, row in zip(angles, rows)}
    _chords(ax, data, angle_of)

    by_phylum = defaultdict(list)
    for angle, row in zip(angles, rows):
        by_phylum[row["phylum"]].append(angle)
    for phy, group in by_phylum.items():
        t1 = group[0] - step / 2.0
        t2 = group[-1] + step / 2.0
        if t2 <= t1:
            t2 += 360.0
        ax.add_patch(Wedge((0, 0), RING_R0 - 0.02, t1, t2, width=0.1,
                            facecolor=PHYLUM_COLOR[phy], edgecolor="white", linewidth=0.8, zorder=2))

    for angle, row in zip(angles, rows):
        color = PHYLUM_COLOR[row["phylum"]]
        lo = angle - step / 2.0 + step * 0.13
        hi = angle + step / 2.0 - step * 0.13
        r_union = RING_R0 + RING_BAR * row["union"] / top
        r_core = RING_R0 + RING_BAR * row["core"] / top
        ax.add_patch(Wedge((0, 0), r_union, lo, hi, width=r_union - RING_R0,
                            facecolor=_tint(color, 0.64), edgecolor="none", zorder=3))
        if row["core"] > 0:
            ax.add_patch(Wedge((0, 0), r_core, lo, hi, width=r_core - RING_R0,
                                facecolor=color, edgecolor="white", linewidth=0.4, zorder=4))
        if row["mag"]:
            x, y = _polar(r_union + 0.05, angle)
            ax.plot([x], [y], marker="*", markersize=8.0, markerfacecolor="#FFC800",
                    markeredgecolor="#4D4D4D", markeredgewidth=0.5, zorder=6)
        linked = row["name"] in bold
        label_r = r_union + (0.12 if row["mag"] else 0.065)
        x, y = _polar(label_r, angle)
        wrapped = angle % 360.0
        right = math.cos(math.radians(wrapped)) >= 0
        ax.text(x, y, _italic(row["name"]), rotation=wrapped if right else wrapped + 180.0,
                rotation_mode="anchor", ha="left" if right else "right", va="center",
                fontsize=7.8 if linked else 6.9, fontweight="bold" if linked else "normal",
                color=ACCENT_DK if linked else INK, zorder=6)

    scale_label = dict(boxstyle="round,pad=0.04", fc="white", ec="none")
    gap_lo = GAP_CENTER - gap / 2.0
    gap_hi = GAP_CENTER + gap / 2.0
    for mb in (0.1, 0.2, 0.3):
        radius = RING_R0 + RING_BAR * mb / top
        ax.add_patch(Wedge((0, 0), radius, gap_hi, gap_lo + 360.0, width=0.0015,
                            facecolor="none", edgecolor="#C4C4C4", linewidth=0.7, zorder=1))
        x, y = _polar(radius, GAP_CENTER)
        ax.text(x, y, "{0:.1f}".format(mb), ha="center", va="center", fontsize=6.6,
                color="#8A8A8A", zorder=6, bbox=scale_label)
    x, y = _polar(RING_R0 + RING_BAR + 0.02, GAP_CENTER)
    ax.text(x, y, "Mb", ha="center", va="bottom", fontsize=6.8, color="#8A8A8A", zorder=6)

    ax.add_patch(Circle((0, 0), 0.78, facecolor="white", edgecolor="none", zorder=6))
    summary = data["summary"]
    mb = summary["total_bp"] / 1.0e6
    ax.text(0, 0.155, "{0:,}".format(summary["plasmid_contigs"]), ha="center", va="center",
            fontsize=28, fontweight="bold", zorder=7)
    ax.text(0, -0.015, "plasmid contigs", ha="center", va="center", fontsize=12.5,
            fontweight="bold", zorder=7)
    ax.text(0, -0.155, "{0:.1f} Mb  ·  {1:.1f}% of the assembly".format(mb, ASSEMBLY_PERCENT),
            ha="center", va="center", fontsize=10.4, fontweight="bold", color=INK, zorder=7)
    ax.set_xlim(-RING_R, RING_R)
    ax.set_ylim(-RING_R, RING_R)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def _panel_overview(ax, data):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    summary = data["summary"]
    total = int(summary["plasmid_contigs"])
    stages = [
        (total, "Plasmid contigs", "#C9CEDC"),
        (int(summary["plasmid_contigs_in_any_bin"]), "In a bin, ≥ 1 algorithm", "#9AA6C6"),
        (int(summary["all_three"]), "In a bin, all three", "#6B7BA8"),
        (int(summary["host_agreed_n"]), "Unanimous host name", "#E64B35"),
    ]
    left, right = 0.28, 0.72
    mid = 0.5 * (left + right)
    span = right - left
    weights = [math.sqrt(count) for count, _label, _color in stages]
    ax.text(mid, 0.985, "From contigs to a reproducible host map", ha="center", va="top",
            fontsize=10.4, fontweight="bold")
    thick, gap, pitch = 0.068, 0.052, 0.12
    y = 0.9
    prev = None
    for (count, label, color), weight in zip(stages, weights):
        half = 0.5 * span * weight / weights[0]
        if prev is not None:
            ax.fill([mid - prev, mid + prev, mid + half, mid - half],
                    [y + gap, y + gap, y, y], color="#ECEEF3", lw=0, zorder=1)
        ax.add_patch(Rectangle((mid - half, y - thick), 2 * half, thick, facecolor=color,
                               edgecolor="white", linewidth=0.8, zorder=3))
        ink = INK if color in ("#C9CEDC", "#9AA6C6") else "white"
        ax.text(mid, y - thick * 0.5, "{0:,}".format(count), ha="center", va="center",
                fontsize=11, fontweight="bold", color=ink, zorder=4)
        ax.text(mid, y - thick - 0.012, label, ha="center", va="top", fontsize=7.2, color=INK, zorder=4)
        pct = 100.0 * count / total
        ax.text(right + 0.02, y - thick * 0.5, "{0:.0f}%".format(pct) if pct >= 1 else "{0:.1f}%".format(pct),
                ha="left", va="center", fontsize=8.2, color=INK, zorder=4)
        prev = half
        y -= pitch

    linked = [int(summary[algo + "_hic_linked"]) for algo in ALGOS]
    contact = [int(summary[algo + "_hic_contact_host"]) for algo in ALGOS]
    vmax = max(a + b for a, b in zip(linked, contact)) or 1
    for algo, lk, ct, yy in zip(ALGOS, linked, contact, (0.355, 0.245, 0.135)):
        x1 = left + span * lk / (vmax * 1.02)
        x2 = left + span * (lk + ct) / (vmax * 1.02)
        ax.add_patch(Rectangle((left, yy - 0.055 / 2), x1 - left, 0.055, facecolor=CLUSTER_COLOR[algo],
                               edgecolor="black", linewidth=EDGE_LW, zorder=3))
        ax.add_patch(Rectangle((x1, yy - 0.055 / 2), max(x2 - x1, 0), 0.055,
                               facecolor=_tint(CLUSTER_COLOR[algo], 0.55), edgecolor="black",
                               linewidth=EDGE_LW, hatch="////", zorder=3))
        ax.text(left - 0.015, yy, algo, ha="right", va="center", fontsize=8.8, fontweight="bold")
        ax.text(right + 0.02, yy, "{0} + {1}".format(lk, ct), ha="left", va="center", fontsize=8.0)
    _panel_b_legend(ax)


def _panel_b_legend(ax):
    items = [("#4D4D4D", None, "Linked"), (_tint("#4D4D4D", 0.45), "////", "Contact host")]
    items.extend((CLUSTER_COLOR[algo], None, algo) for algo in ALGOS)
    artists = []
    x = 0.0
    for color, hatch, label in items:
        rect = Rectangle((x, 0.0), 0.03, 0.022, facecolor=color, edgecolor="black", linewidth=0.5,
                         hatch=hatch, zorder=3, clip_on=False)
        ax.add_patch(rect)
        text = ax.text(x + 0.038, 0.011, label, ha="left", va="center", fontsize=8, color=INK,
                       zorder=4, clip_on=False)
        artists.append((rect, text))
        x += 0.16
    ax._b_legend_items = artists


def _align_b_legend(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[1]
    artists = getattr(ax, "_b_legend_items", None)
    if not artists:
        return
    def span(artist):
        box = artist.get_window_extent(renderer)
        return box.x0, box.x1

    x0 = min(min(span(rect)[0], span(text)[0]) for rect, text in artists)
    x1 = max(max(span(rect)[1], span(text)[1]) for rect, text in artists)
    inv = ax.transData.inverted()
    x0, _y0 = inv.transform((x0, 0))
    x1, _y1 = inv.transform((x1, 0))
    shift = 0.50 - (x0 + x1) / 2.0
    for rect, text in artists:
        xy = rect.get_xy()
        rect.set_xy((xy[0] + shift, xy[1]))
        text.set_x(text.get_position()[0] + shift)


def _panel_map(ax, data):
    aspect = BOX_C[2] / BOX_C[3]
    ax.set_xlim(0, aspect)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    cluster, bundle = _broad_cluster(data)
    if not bundle:
        ax.text(aspect / 2, 0.5, "no broad-host plasmid cluster", ha="center", va="center")
        return
    med_kb = float(np.median(bundle["lengths"])) / 1000.0
    novel = 100.0 * bundle["novel"] / max(bundle["n"], 1)
    meta = {row["species"].strip(): row for row in data["species"]}
    hosts = sorted(bundle["anchors"].items(), key=lambda item: -item[1]["frac"])
    ax.text(aspect / 2.0, 0.985, "A novel plasmid type bridges three genera", ha="center", va="top",
            fontsize=10.5, fontweight="bold")
    hub = (aspect * 0.36, 0.44)
    hr = 0.175
    ax.add_patch(Circle(hub, hr, facecolor="#2C3E66", edgecolor="white", linewidth=1.4, zorder=5))
    ax.text(hub[0], hub[1] + 0.055, cluster, ha="center", va="center", fontsize=12,
            fontweight="bold", color="white", zorder=6)
    ax.text(hub[0], hub[1] - 0.012, "{0} contigs".format(bundle["n"]), ha="center", va="center",
            fontsize=7.4, color="#E4E7F0", zorder=6)
    ax.text(hub[0], hub[1] - 0.052, "median {0:.0f} kb".format(med_kb), ha="center", va="center",
            fontsize=7.4, color="#E4E7F0", zorder=6)
    ax.text(hub[0], hub[1] - 0.092, "{0:.0f}% novel".format(novel), ha="center", va="center",
            fontsize=7.4, fontweight="bold", color="white", zorder=6)
    node_x = aspect * 0.64
    nr = 0.07
    for (species, info), y in zip(hosts, (0.78, 0.46, 0.14)):
        row = meta.get(species, {})
        is_mag = _is_mag(_num(row.get("completeness")), _num(row.get("contamination"), 1.0e9))
        ang = math.atan2(y - hub[1], node_x - hub[0])
        p0 = (hub[0] + (hr + 0.008) * math.cos(ang), hub[1] + (hr + 0.008) * math.sin(ang))
        p1 = (node_x - nr * math.cos(ang), y - nr * math.sin(ang))
        c0 = ((p0[0] + p1[0]) / 2.0, p0[1])
        c1 = ((p0[0] + p1[0]) / 2.0, p1[1])
        half = 0.008 + 0.022 * info["frac"]
        _ribbon(ax, p0, c0, c1, p1, half, half * 0.65, "#6B7BA8", alpha=0.5, zorder=2)
        mx = p0[0] * 0.28 + p1[0] * 0.72
        my = p0[1] * 0.28 + p1[1] * 0.72
        if abs(y - hub[1]) < 0.12:
            my += 0.055
        ax.text(mx, my, "{0:.0f}%".format(round(100 * info["frac"])), ha="center", va="center",
                fontsize=9.2, fontweight="bold", color=INK, zorder=7,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
        phy = PHYLUM_COLOR[data["phylum"].get(species, "Other")]
        ax.add_patch(Circle((node_x, y), nr, facecolor=phy, edgecolor="white", linewidth=1.2, zorder=5))
        ax.text(node_x + nr + 0.03, y + 0.028, _italic(species), ha="left", va="center",
                fontsize=8.2, zorder=6, gid="mag-star" if is_mag else None)
        for k, algo in enumerate(ALGOS):
            on = algo in info["algs"]
            ax.plot([node_x + nr + 0.045 + k * 0.055], [y - 0.038], marker="o", markersize=5.6,
                    linestyle="none", markerfacecolor=CLUSTER_COLOR[algo] if on else "white",
                    markeredgecolor=CLUSTER_COLOR[algo], markeredgewidth=1.0, zorder=6)
        if info["mobv"]:
            ax.text(node_x + nr + 0.045 + 0.165 + 0.02, y - 0.038, "MOBV", ha="left", va="center",
                    fontsize=7.0, fontweight="bold", color=ACCENT_DK, zorder=6)


def _separate(values, tol=0.04):
    offsets = [0.0] * len(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    groups = []
    current = []
    for index in order:
        if current and values[index] - values[current[-1]] > tol:
            groups.append(current)
            current = []
        current.append(index)
    if current:
        groups.append(current)
    for group in groups:
        n = len(group)
        if n == 1:
            continue
        if n == 2:
            shifts = [-0.07, 0.07]
        elif n == 3:
            shifts = [-0.07, 0.0, 0.07]
        else:
            shifts = list(np.linspace(-0.07, 0.07, n))
        for shift, index in zip(shifts, group):
            offsets[index] = shift
    return offsets


def _panel_anchors(ax, data):
    ax.axis("off")
    meta = {row["species"].strip(): row for row in data["species"]}
    per_host = [_host_contigs(data, host) for host in SHOWCASE]
    ax.set_xlim(-8, 118)
    ax.set_ylim(-0.52, 3.22)
    for x, label in ((0, "0"), (50, "50"), (100, "100")):
        ax.text(x, -0.06, label, ha="center", va="top", fontsize=8.0, color=INK)
    ax.text(50, -0.24, "Contact on the host chromosome (%)", ha="center", va="top",
            fontsize=8.2, fontweight="bold", color=INK)
    for idx, (host, contigs) in enumerate(zip(SHOWCASE, per_host)):
        base = (len(SHOWCASE) - 1 - idx) * 1.0
        axis_y = base + 0.28
        line_y = axis_y - 0.1
        row = meta.get(host, {})
        is_mag = _is_mag(_num(row.get("completeness")), _num(row.get("contamination"), 1.0e9))
        med = float(np.median([item["frac"] for item in contigs])) if contigs else 0.0
        ax.plot([0, 100], [line_y, line_y], color=INK, lw=2.2, zorder=1, solid_capstyle="butt")
        ax.plot([100 * med, 100 * med], [line_y - 0.1, line_y], color=INK, lw=1.5, zorder=2,
                solid_capstyle="butt")
        ax.text(100 * med, line_y - 0.12, "{0:.0f}%".format(round(100 * med)), ha="center", va="top",
                fontsize=8.0, fontweight="bold", color=INK, zorder=5)
        jitter = _separate([item["frac"] for item in contigs])
        for item, dy in zip(contigs, jitter):
            color = IN_ALL if item["all"] else ONE_ONLY
            ax.plot([100 * item["frac"]], [axis_y + dy], marker="o", markersize=8.6,
                    markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.7, zorder=4)
        ax.text(0, base + 0.68, _italic(host), ha="left", va="center", fontsize=10.2,
                fontweight="bold", zorder=5, gid="mag-star" if is_mag else None)


def _legends(fig, data):
    present = []
    seen = set()
    for row in _host_species(data):
        if row["phylum"] not in seen:
            seen.add(row["phylum"])
            present.append(row["phylum"])
    phyla = [Patch(facecolor=PHYLUM_COLOR[name], edgecolor="white", label=name) for name in present]
    first = PHYLUM_COLOR[present[0]]
    second = [
        Patch(facecolor=first, edgecolor="white", label="All three algorithms"),
        Patch(facecolor=_tint(first, 0.64), edgecolor="white", label="Union of algorithms"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#FFC800",
               markeredgecolor="#4D4D4D", markersize=14, label="MAG"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ACCENT,
               markeredgecolor="white", markersize=11, label="AG093, shared"),
    ]
    kw = dict(fontsize=12.0, handlelength=1.45, handleheight=1.15, handletextpad=0.45,
              columnspacing=1.25, borderaxespad=0.0, frameon=False)
    y1 = A_LEGEND_TOP / FIG_H
    leg1 = fig.legend(handles=phyla, loc="upper left", bbox_to_anchor=(0.05, y1), ncol=6, **kw)
    leg2 = fig.legend(handles=second, loc="upper left", bbox_to_anchor=(0.05, y1), ncol=3, **kw)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    box = leg1.get_window_extent(renderer)
    x0, y0 = inv.transform((box.x0, box.y0))
    x1, y_top = inv.transform((box.x1, box.y1))
    left = (BOX_A[0] + BOX_A[2] / 2.0) / FIG_W - (x1 - x0) / 2.0
    leg1.set_bbox_to_anchor((left, y1))
    leg2.set_bbox_to_anchor((left, y1 - (y_top - y0) - 0.002))
    d_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=IN_ALL, markeredgecolor="white",
               markersize=8, label="Co-binned by all three"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ONE_ONLY, markeredgecolor="white",
               markersize=8, label="Not in every algorithm"),
        Line2D([0], [0], marker="|", color=INK, markersize=11, markeredgewidth=1.5, linestyle="none",
               label="Median"),
    ]
    fig.legend(handles=d_handles, loc="upper center",
               bbox_to_anchor=((BOX_D[0] + BOX_D[2] / 2.0) / FIG_W, D_LEGEND_TOP / FIG_H),
               ncol=3, fontsize=8.6, handlelength=1.2, handleheight=0.9, handletextpad=0.4,
               columnspacing=1.0, borderaxespad=0.0, frameon=False)


def _mag_stars(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for text in list(ax.texts):
            if text.get_gid() != "mag-star":
                continue
            box = text.get_window_extent(renderer)
            inv = ax.transData.inverted()
            x, y = inv.transform((box.x1 + 2.0, (box.y0 + box.y1) / 2.0))
            ax.plot([x], [y], marker="*", markersize=11.0, linestyle="none",
                    markerfacecolor="#FFC800", markeredgecolor="#4D4D4D", markeredgewidth=0.45,
                    zorder=7, clip_on=False)


def draw(data, path):
    _style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    _panel_host_map(fig.add_axes(_box(BOX_A)), data)
    _panel_overview(fig.add_axes(_box(BOX_B)), data)
    _panel_map(fig.add_axes(_box(BOX_C)), data)
    _panel_anchors(fig.add_axes(_box(BOX_D)), data)
    _legends(fig, data)
    _align_b_legend(fig)
    _mag_stars(fig)
    gutter = BOX_A[0] + BOX_A[2] + 0.04
    letters = (
        ("a", 0.04, _COL_TOP + 0.02),
        ("b", gutter, _COL_TOP + 0.02),
        ("c", gutter, BOX_C[1] + BOX_C[3] + 0.02),
        ("d", gutter, BOX_D[1] + BOX_D[3] + 0.02),
    )
    for letter, x, y in letters:
        fig.text(x / FIG_W, y / FIG_H, letter, fontsize=21, fontweight="bold", ha="left", va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    plt.close(fig)
    print("[FIG] {0}".format(path), flush=True)


def report(data):
    summary = data["summary"]
    print("[PLASMID] contigs={0}  binned={1}  all_three={2}  agreed={3}  Mb={4:.1f}".format(
        summary["plasmid_contigs"], summary["plasmid_contigs_in_any_bin"], summary["all_three"],
        summary["host_agreed_n"], summary["total_bp"] / 1.0e6), flush=True)
    for algo in ALGOS:
        print("[HIC] {0:<8} linked={1}  contact_host={2}".format(
            algo, summary[algo + "_hic_linked"], summary[algo + "_hic_contact_host"]), flush=True)
    cluster, bundle = _broad_cluster(data)
    if bundle:
        print("[CLUSTER] {0} n={1} novel={2:.0f}% median_kb={3:.0f}".format(
            cluster, bundle["n"], 100.0 * bundle["novel"] / bundle["n"],
            float(np.median(bundle["lengths"])) / 1000.0), flush=True)
        for name, info in sorted(bundle["anchors"].items(), key=lambda item: -item[1]["frac"]):
            print("[HOST] {0}  {1:.0f}%  {2}{3}".format(
                name, 100 * info["frac"], ",".join(info["algs"]), "  MOBV" if info["mobv"] else ""), flush=True)
    for host in SHOWCASE:
        contigs = _host_contigs(data, host)
        med = float(np.median([item["frac"] for item in contigs])) if contigs else 0.0
        print("[STRIP] {0} n={1} median={2:.0f}%".format(host, len(contigs), 100 * med), flush=True)


def main():
    data = load()
    report(data)
    draw(data, OUT)


if __name__ == "__main__":
    main()
