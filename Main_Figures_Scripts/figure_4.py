#!/usr/bin/env python3
"""Figure 4. Lineages recovered from the sheep gut and the fate of each genome across contact normalizations.

    python figure_4.py

Output: Main_Figures/Main_Figure_4.png at 600 dpi.

Input: figures/cache_fig5/<dataset>_<cluster>_<arm>/, written on the
cluster by sheep_gut/HARMONI/Scripts/figure_5.py and copied here. For
every dataset, clustering algorithm (metacc is Leiden) and normalization
(arms harmoni, normcc and bin3c for MACO, NormCC and bin3C) it holds
    bins.tsv.gz         contig id and length of every bin
    quality_report.tsv  the CheckM2 scores of these bins
    hq_bins.tsv         the GTDB-Tk table of the high-quality bins
    source.tsv          cluster paths of the FASTA, CheckM2 and GTDB-Tk inputs
The script stops when CheckM2 disagrees with the contigs of a bin, when
hq_bins.tsv disagrees with CheckM2, when one contig sits in two bins of
a run, or when one contig identifier has two lengths in the runs of one
dataset and clustering algorithm.

A clean genome is a bin with completeness >= 50% and contamination <= 1%.
A high-quality bin has completeness >= 50% and contamination <= 10%.

figure_S9.py, figure_S10.py and figure_S11.py draw the same figure for
Mount Melbourne, Mount Rittmann and the mouse fecal pellet.

Panel a. Sheep gut only. GTDB cladogram of every genus with a clean bin
in any of the nine runs (three clustering algorithms, three
normalizations). Bins without a genus name form one leaf per family
(open stars). From the leaves outward the rings are Leiden vs NormCC,
Leiden vs bin3C, Infomap vs NormCC, Infomap vs bin3C, ImputeCC vs NormCC
and ImputeCC vs bin3C. Each cell compares the clean bins of MACO in the
genus with those of that normalization under the same clustering
algorithm: MACO only, more in MACO, equal, fewer in MACO, or other only.
The bars below count the genera of each class for every ring and give
the clean bins of MACO and of the other normalization. Black dots mark
genera with a species that MACO recovered as two or more clean bins
under Leiden.

Panel b. Every clean genome of one run is traced into the run of another
normalization with the same clustering algorithm, in both directions:
MACO into NormCC and bin3C, and NormCC and bin3C into MACO. Each contig
of the genome is looked up by its FASTA identifier. The primary target
bin is the one holding the most base pairs of the genome. The fate is
the first that applies:
    unbinned    at least half of the genome's base pairs are in no target bin
    split       no target bin holds half of the genome's base pairs
    merged      the primary bin has contamination > 10% and also holds at
                least half of the base pairs of another high-quality bin
                of the query run
    >10%        the primary bin has contamination > 10%
    incomplete  the primary bin has completeness < 50%
    1-10%       the primary bin has contamination > 1% and <= 10%
    <=1%        otherwise
Split and incomplete are drawn together. Bars are percentages of the
clean genomes of the query run, pooled over the four datasets.

Panel c. NormCC and bin3C bins under Leiden that hold at least half of
the base pairs of two or more high-quality MACO bins, at least one of
them clean, and have contamination > 10%. Segments are the base pairs
of each MACO genome in the bin, colored by phylum as in a; the gray
segment is the rest of the bin. Taxa are the GTDB genera of the MACO
genomes (family when the genus is unnamed). At most C_ROWS bins are
drawn, ordered by the number of MACO genomes and then by their base
pairs.
"""

from __future__ import annotations

import csv
import gzip
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, Wedge
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory


HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "Main_Figures"
TABLES = HERE.parent / "Supplementary_Tables"
OUT = FIG_DIR / "Main_Figure_4.png"

DATASETS = {"melbourne": "Mount Melbourne", "rittmann": "Mount Rittmann", "mouse": "Mouse fecal pellet",
            "sheep": "Sheep gut"}
DATASET_SHORT = {"melbourne": "Melbourne", "rittmann": "Rittmann", "mouse": "Mouse", "sheep": "Sheep"}

METHODS = ("MACO", "NormCC", "bin3C")
ARM = {"MACO": "harmoni", "NormCC": "normcc", "bin3C": "bin3c"}
CLUSTERS = (
    ("metacc", "Leiden"),
    ("infomap", "Infomap"),
    ("imputecc", "ImputeCC"),
)
# (query, target)
DIRECTIONS = (("MACO", "NormCC"), ("MACO", "bin3C"), ("NormCC", "MACO"), ("bin3C", "MACO"))
COLORS = {"MACO": "#E64B35", "NormCC": "#00A087", "bin3C": "#3C5488"}

COMP_MIN = 50.0
CLEAN_MAX = 1.0
HQ_MAX = 10.0
MAJORITY = 0.5
FASTA_SUFFIXES = (".fasta.gz", ".fna.gz", ".fa.gz", ".fasta", ".fna", ".fa")

FATES = ("clean", "light", "composite", "merged", "incomplete", "split", "unbinned")
PLOT_FATES = (
    ("clean", "≤1%"),
    ("light", "1–10%"),
    ("composite", ">10%"),
    ("merged", "Merged with another genome"),
    ("broken", "Split or <50% complete"),
    ("unbinned", "Unbinned"),
)
FATE_WHITE = {"clean": 0.0, "light": 0.45, "composite": 0.74, "merged": 0.74}
FATE_FILL = {"broken": "#D9D9D9", "unbinned": "#FFFFFF"}

# Pastel shades for the most frequent bacterial phyla of the tree, in order.
PASTELS = ("#FFD8A8", "#C3E6B4", "#D9CCE8", "#BEE3EE", "#F8C8DC", "#E6E2A3", "#DCC7AE", "#C6D7EE")
ARCHAEA_COLOR = "#F4B6AE"
OTHER_COLORS = ("#E6E6E6", "#D2D2D2")
OTHER_LABEL = "Other bacterial phyla"
REST_COLOR = "#BDBDBD"

# Clean bins of MACO against those of one other normalization, per genus and clustering algorithm.
RIVALS = ("NormCC", "bin3C")
COMPARE = (
    ("maco_only", "MACO only", "#E64B35"),
    ("maco_more", "More in MACO", "#F6B2A6"),
    ("equal", "Equal", "#D4D4D4"),
    ("maco_fewer", "Fewer in MACO", "#BCC6DA"),
    ("other_only", "Other only", "#5F7194"),
)
EMPTY = "#F4F4F4"

RANK_RADII = (0.0, 0.07, 0.16, 0.27, 0.38, 0.49, 0.60)
R_LEAF = RANK_RADII[-1]
RING = 0.045
RING_GAP = 0.005
GROUP_GAP = 0.022
GAP_DEG = 30.0
DOMAIN_GAP = 2.0
LABEL_DY = 0.088
LABELS = ("Methanocorpusculum", "Methanoprimaticola", "Mailhella", "Cryptobacteroides", "UMGS1668", "Tidjanibacter")
MAX_LABELS = 14

C_ROWS = 10

FIG_W = 16.4
FIG_H = 11.6
# left, bottom, width, height in inches; panel a spans the same height as panels b and c
TREE_BOX = (0.25, 4.15, 8.75, 7.1)
TREE_LIM = 1.42
SUMMARY_BOX = (2.35, 1.0, 4.2, 1.45)
LEGEND_X = 0.45
LEGEND_TOP = 4.1
LEGEND_ROW = 0.215
LEGEND_NCOL = 6
LEGEND_FS = 10
KEY_W, KEY_H, KEY_PAD = 0.16, 0.13, 0.07
EDGE_LW = 0.5
LABEL_FS = 13
BRACKET_LW = 1.6
CLUSTER_FS = 12
FATE_BOX = (11.55, 6.55, 4.05, 4.15)
JOIN_BOX = (11.55, 1.0, 4.05, 4.3)


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
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.linewidth": 1.05,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.frameon": False,
            "hatch.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "mathtext.fontset": "custom",
            "mathtext.it": font + ":italic",
            "mathtext.rm": font,
            "mathtext.bf": font + ":bold",
            "mathtext.sf": font,
            "mathtext.cal": font,
        }
    )


# ---------------------------------------------------------------- inputs

def _stem(name):
    for suffix in FASTA_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _read_bins_cache(path):
    """(source directory, bins) from a bins.tsv.gz cache; source is None when there is no cache."""
    if not path.is_file():
        return None, None
    bins = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        source = handle.readline().rstrip("\n")[1:].split("\t")[0]
        for row in csv.reader(handle, delimiter="\t"):
            bins[row[0]].append((row[1], int(row[2])))
    return source, dict(bins)


def _read_checkm(directory):
    reports = sorted(directory.glob("chunk_*/quality_report.tsv"))
    if not reports and (directory / "quality_report.tsv").is_file():
        reports = [directory / "quality_report.tsv"]
    if not reports:
        return None
    quality = {}
    for report in reports:
        with report.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                quality[_stem(row["Name"].strip())] = {
                    "completeness": float(row["Completeness"]),
                    "contamination": float(row["Contamination"]),
                    "size": int(float(row["Genome_Size"])),
                    "contigs": int(float(row["Total_Contigs"])),
                    "longest": int(float(row["Max_Contig_Length"])),
                }
    return quality


def _checkm_problems(quality, bins):
    problems = []
    missing = sorted(set(bins) - set(quality))
    extra = sorted(set(quality) - set(bins))
    if missing or extra:
        problems.append("{0} FASTA bins without a score, {1} scores without a FASTA, e.g. {2}".format(
            len(missing), len(extra), (missing + extra)[:4]))
        return problems
    for bin_id, contigs in bins.items():
        info = quality[bin_id]
        lengths = [length for _contig, length in contigs]
        if (info["contigs"], info["size"], info["longest"]) != (len(lengths), sum(lengths), max(lengths)):
            problems.append("{0}: CheckM2 contigs/size/longest {1}/{2}/{3}, FASTA {4}/{5}/{6}".format(
                bin_id, info["contigs"], info["size"], info["longest"], len(lengths), sum(lengths), max(lengths)))
            if len(problems) >= 4:
                break
    return problems


def _ranks(classification):
    ranks = dict.fromkeys("dpcofgs", "")
    for part in classification.split(";"):
        part = part.strip()
        if len(part) >= 3 and part[1:3] == "__":
            ranks[part[0]] = part[3:]
    return ranks


def read_taxonomy(path, quality):
    """GTDB ranks of every high-quality bin. hq_bins.tsv must agree with CheckM2."""
    if not path.is_file():
        raise SystemExit("[ERROR] missing {0}; run analyze_scripts/run_bins_compare.sh first".format(path))
    taxonomy = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            bin_id = _stem(row["bin_id"].strip())
            info = quality.get(bin_id)
            if info is None or abs(info["completeness"] - float(row["completeness"])) > 0.006 \
                    or abs(info["contamination"] - float(row["contamination"])) > 0.006:
                raise SystemExit("[ERROR] {0} row {1} does not match CheckM2; rerun run_bins_compare.sh".format(
                    path, bin_id))
            taxonomy[bin_id] = _ranks(row["classification"])
    expected = {bin_id for bin_id, info in quality.items() if _is_hq(info)}
    if set(taxonomy) != expected:
        raise SystemExit("[ERROR] {0} lists {1} bins, CheckM2 has {2} high-quality bins; rerun run_bins_compare.sh"
                         .format(path, len(taxonomy), len(expected)))
    return taxonomy


def _is_hq(info):
    return info["completeness"] >= COMP_MIN and info["contamination"] <= HQ_MAX


def _is_clean(info):
    return info["completeness"] >= COMP_MIN and info["contamination"] <= CLEAN_MAX


def _load_cached_run(run_dir, bins):
    quality = _read_checkm(run_dir)
    if quality is None:
        raise SystemExit("[ERROR] {0} has no quality_report.tsv; rerun figure_5.py on the cluster".format(run_dir))
    problems = _checkm_problems(quality, bins)
    if problems:
        raise SystemExit("[ERROR] cached CheckM2 does not match the cached bins in {0}\n  {1}".format(
            run_dir, "\n  ".join(problems)))
    taxonomy = read_taxonomy(run_dir / "hq_bins.tsv", quality)
    with (run_dir / "source.tsv").open(encoding="utf-8", newline="") as handle:
        source = dict(csv.reader(handle, delimiter="\t"))
    return quality, taxonomy, source


def load_run(dataset, cluster, method):
    """One run from figures/cache_fig5."""
    run_dir = CACHE_DIR / "{0}_{1}_{2}".format(dataset, cluster, ARM[method])
    source, bins = _read_bins_cache(run_dir / "bins.tsv.gz")
    if source is None:
        raise SystemExit("[ERROR] missing {0}; copy cache_fig5 from the cluster".format(run_dir / "bins.tsv.gz"))
    quality, taxonomy, paths = _load_cached_run(run_dir, bins)
    directory = paths["fasta"]
    contig_bin = {}
    length = {}
    for bin_id, contigs in bins.items():
        for contig, size in contigs:
            if contig in contig_bin:
                raise SystemExit("[ERROR] contig {0} is in {1} and {2} of {3}".format(
                    contig, contig_bin[contig], bin_id, directory))
            contig_bin[contig] = bin_id
            length[contig] = size
    print("[RUN] {0:<9} {1:<8} {2:<6} bins={3:<5} clean={4:<4} hq={5:<4} {6}".format(
        dataset, cluster, method, len(bins), sum(_is_clean(q) for q in quality.values()), len(taxonomy),
        run_dir.name), flush=True)
    return {"bins": bins, "quality": quality, "taxonomy": taxonomy, "contig_bin": contig_bin, "length": length,
            "size": {bin_id: sum(size for _c, size in contigs) for bin_id, contigs in bins.items()}}


def check_assembly(runs, dataset, cluster):
    """Runs of one dataset and clustering algorithm must share contig ids and lengths."""
    reference = runs["MACO"]
    for method in METHODS[1:]:
        other = runs[method]
        shared = 0
        for contig, size in other["length"].items():
            ref = reference["length"].get(contig)
            if ref is None:
                continue
            if ref != size:
                raise SystemExit("[ERROR] contig {0} is {1} bp in MACO and {2} bp in {3} ({4} {5})".format(
                    contig, ref, size, method, dataset, cluster))
            shared += size
        total = sum(reference["length"].values())
        fraction = shared / total if total else 0.0
        print("[ASM] {0:<9} {1:<8} MACO binned bp also binned by {2}: {3:.1%}".format(
            dataset, cluster, method, fraction), flush=True)
        if fraction < 0.01:
            raise SystemExit("[ERROR] MACO and {0} share no contig ids in {1} {2}".format(method, dataset, cluster))


# ---------------------------------------------------------------- tracing

def _old_fate(query, target, contigs):
    """Fate under the earlier rule: the bin whose longest contig is the genome's longest contig."""
    contig, size = max(contigs, key=lambda item: (item[1], item[0]))
    hit = target["contig_bin"].get(contig)
    if hit is None:
        return "missing"
    info = target["quality"][hit]
    if info["longest"] != size and not (len(contigs) == 1 and info["contigs"] == 2):
        return "missing"
    if info["contamination"] > HQ_MAX:
        return "composite"
    if info["completeness"] < COMP_MIN:
        return "partial"
    if info["contamination"] > CLEAN_MAX:
        return "light"
    return "clean"


def _agreement(query_tax, target_tax):
    if target_tax is None:
        return "target not high quality"
    if query_tax["s"] and target_tax["s"]:
        return "same species" if query_tax["s"] == target_tax["s"] else "different species"
    if query_tax["g"] and target_tax["g"]:
        return "same genus" if query_tax["g"] == target_tax["g"] else "different genus"
    return "same family" if query_tax["f"] == target_tax["f"] else "different family"


def trace(dataset, cluster, query_method, target_method, query, target):
    """Fate of every clean query genome in the target run, and the target bins that merge query genomes."""
    target_of = target["contig_bin"]
    held = defaultdict(Counter)
    for query_bin, contigs in query["bins"].items():
        for contig, size in contigs:
            target_bin = target_of.get(contig)
            if target_bin is not None:
                held[target_bin][query_bin] += size
    hq = {bin_id for bin_id, info in query["quality"].items() if _is_hq(info)}
    rows = []
    for query_bin in sorted(query["bins"]):
        info = query["quality"][query_bin]
        if not _is_clean(info):
            continue
        contigs = query["bins"][query_bin]
        size = query["size"][query_bin]
        shares = Counter()
        unbinned = 0
        for contig, length in contigs:
            target_bin = target_of.get(contig)
            if target_bin is None:
                unbinned += length
            else:
                shares[target_bin] += length
        primary = min(shares, key=lambda name: (-shares[name], name)) if shares else None
        in_primary = shares[primary] / size if primary else 0.0
        partners = []
        if primary:
            partners = sorted(other for other, bp in held[primary].items()
                              if other != query_bin and other in hq and bp >= MAJORITY * query["size"][other])
        p_info = target["quality"][primary] if primary else None
        if unbinned >= MAJORITY * size:
            fate = "unbinned"
        elif in_primary < MAJORITY:
            fate = "split"
        elif p_info["contamination"] > HQ_MAX and partners:
            fate = "merged"
        elif p_info["contamination"] > HQ_MAX:
            fate = "composite"
        elif p_info["completeness"] < COMP_MIN:
            fate = "incomplete"
        elif p_info["contamination"] > CLEAN_MAX:
            fate = "light"
        else:
            fate = "clean"
        q_tax = query["taxonomy"][query_bin]
        rows.append({
            "dataset": dataset, "clustering": cluster, "query": query_method, "target": target_method,
            "query_bin": query_bin, "query_completeness": info["completeness"],
            "query_contamination": info["contamination"], "query_size_bp": size, "query_contigs": len(contigs),
            "query_longest_share": max(length for _c, length in contigs) / size,
            "target_bins_holding_query": len(shares),
            "primary_bin": primary or "", "primary_share_of_query": round(in_primary, 4),
            "unbinned_share_of_query": round(unbinned / size, 4),
            "query_share_of_primary": round(shares[primary] / target["size"][primary], 4) if primary else "",
            "primary_completeness": p_info["completeness"] if p_info else "",
            "primary_contamination": p_info["contamination"] if p_info else "",
            "primary_size_bp": target["size"][primary] if primary else "",
            "other_query_genomes_in_primary": ";".join(partners),
            "fate": fate, "longest_contig_fate": _old_fate(query, target, contigs),
            "query_classification": _classification(q_tax),
            "primary_classification": _classification(target["taxonomy"].get(primary)) if primary else "",
            "taxonomy_agreement": _agreement(q_tax, target["taxonomy"].get(primary)) if primary else "",
        })
    merged = []
    for target_bin, parts in held.items():
        t_info = target["quality"][target_bin]
        if t_info["contamination"] <= HQ_MAX:
            continue
        members = [bin_id for bin_id, bp in parts.items() if bin_id in hq and bp >= MAJORITY * query["size"][bin_id]]
        if len(members) < 2 or not any(_is_clean(query["quality"][bin_id]) for bin_id in members):
            continue
        members.sort(key=lambda bin_id: (-parts[bin_id], bin_id))
        merged.append({
            "dataset": dataset, "clustering": cluster, "query": query_method, "target": target_method,
            "bin": target_bin, "info": t_info, "size": target["size"][target_bin],
            "members": [(bin_id, parts[bin_id], query["quality"][bin_id], query["size"][bin_id],
                         query["taxonomy"][bin_id]) for bin_id in members],
        })
    return rows, merged


def _classification(tax):
    if not tax:
        return ""
    return ";".join("{0}__{1}".format(rank, tax[rank]) for rank in "dpcofgs")


# ---------------------------------------------------------------- compute

def _leaf(tax):
    family = tax["f"] or "{0} (unnamed family)".format(tax["o"] or tax["c"] or tax["p"])
    genus = tax["g"] or "unnamed genus"
    return (tax["d"], tax["p"], tax["c"] or "unnamed class", tax["o"] or "unnamed order", family, genus)


_FATE = {
    "Clean (≤1%)": "clean",
    "1–10% contamination": "light",
    "Merged with another genome": "merged",
    ">10% contamination": "composite",
    "<50% complete": "incomplete",
    "Unbinned": "unbinned",
    "Split": "split",
}
_DATA_KEY = {label: key for key, label in DATASETS.items()}


def _gtdb(text):
    tax = {rank: "" for rank in "dpcofgs"}
    if not isinstance(text, str):
        return tax
    for part in text.split(";"):
        if "__" not in part:
            continue
        rank, name = part.split("__", 1)
        if rank in tax:
            tax[rank] = name.strip()
    return tax


def _sheet(name, sheet):
    import pandas as pd

    frame = pd.read_excel(TABLES / name, sheet_name=sheet, header=2)
    frame = frame[frame["Dataset"].isin(DATASETS.values())]
    return frame


def compute(datasets, tree_dataset):
    """Panel a from tree_dataset; panels b and c pooled over datasets.

    Completeness, contamination and GTDB labels come from Supplementary Table S3.
    Tracing and merged bins come from the two sheets of Supplementary Table S5.
    """
    bins = _sheet("Supplementary_Table_S3.xlsx", "Table S3")
    traces = _sheet("Supplementary_Table_S5.xlsx", "Genome fates")
    joins = _sheet("Supplementary_Table_S5.xlsx", "Merged bins")
    tree_name = DATASETS[tree_dataset]
    wanted = {DATASETS[key] for key in datasets}
    counts = defaultdict(lambda: defaultdict(int))
    paired = set()
    species = Counter()
    tree = bins[bins["Dataset"] == tree_name]
    for row in tree.itertuples(index=False):
        completeness = float(row[4])
        contamination = float(row[5])
        if not _is_clean({"completeness": completeness, "contamination": contamination}):
            continue
        tax = _gtdb(row[11])
        if not tax["d"]:
            continue
        leaf = _leaf(tax)
        counts[leaf][(row[1], row[2])] += 1
        if row[1] == "Leiden" and row[2] == "MACO" and tax["s"]:
            species[(tax["s"], leaf)] += 1
    paired.update(leaf for (_name, leaf), n in species.items() if n >= 2)
    if not counts:
        raise SystemExit("[ERROR] panel a needs the {0} dataset".format(tree_dataset))

    fates = defaultdict(Counter)
    trace_rows = []
    subset = traces[traces["Dataset"].isin(wanted)]
    for row in subset.itertuples(index=False):
        label = _FATE.get(row[17])
        if label is None:
            raise SystemExit("[ERROR] unknown tracing class {0}".format(row[17]))
        fates[(row[1], row[2], row[8])][label] += 1
        trace_rows.append({"clustering": row[1], "query": row[2], "target": row[8], "fate": label})

    merged = []
    grouped = {}
    for row in joins.itertuples(index=False):
        if row[0] not in wanted:
            continue
        key = (row[0], row[1], row[2], row[3], str(row[4]))
        grouped.setdefault(key, []).append(row)
    for (dataset, clustering, query, target, bin_id), members in grouped.items():
        first = members[0]
        packed = []
        for member in members:
            tax = _gtdb(member[16])
            packed.append((
                str(member[9]),
                float(member[13]),
                {"completeness": float(member[10]), "contamination": float(member[11])},
                float(member[12]),
                tax,
            ))
        packed.sort(key=lambda item: (-item[1], item[0]))
        merged.append({
            "dataset": _DATA_KEY[dataset],
            "clustering": clustering,
            "query": query,
            "target": target,
            "bin": bin_id,
            "info": {"completeness": float(first[5]), "contamination": float(first[6])},
            "size": float(first[7]),
            "members": packed,
        })
    print("[TREE] {0} leaves={1}  phyla={2}  paired genera={3}".format(
        tree_dataset, len(counts), len({leaf[1] for leaf in counts}), len(paired)), flush=True)
    return {"counts": counts, "paired": paired, "rows": trace_rows, "fates": fates, "merged": merged,
            "datasets": datasets, "tree_dataset": tree_dataset}


def report(data):
    for _key, cluster in CLUSTERS:
        for query, target in DIRECTIONS:
            table = data["fates"][(cluster, query, target)]
            total = sum(table.values())
            text = "  ".join("{0}={1}".format(key, table[key]) for key in FATES)
            print("[FATE] {0:<8} {1:>6} -> {2:<6} n={3:<4} {4}".format(cluster, query, target, total, text), flush=True)
    joined = Counter((item["clustering"], item["query"], item["target"]) for item in data["merged"])
    for (cluster, query, target), n in sorted(joined.items()):
        print("[MERGED] {0:<8} {1} bins holding >=2 {2} genomes: {3}".format(cluster, target, query, n), flush=True)


# ---------------------------------------------------------------- source data

# ---------------------------------------------------------------- panel a

def _order(leaves):
    """Leaves in tree order: domains, then clades by size, then name."""
    tree = {}
    for leaf in leaves:
        node = tree
        for name in leaf:
            node = node.setdefault(name, {})

    def size(node):
        return 1 if not node else sum(size(child) for child in node.values())

    def walk(node, path):
        if not node:
            yield tuple(path)
            return
        domain_rank = {"Archaea": 0, "Bacteria": 1}
        keys = sorted(node, key=lambda key: (domain_rank.get(key, 2), -size(node[key]), key))
        for key in keys:
            yield from walk(node[key], path + [key])

    return list(walk(tree, []))


def _angles(ordered):
    """Clockwise from the right edge of the top gap. A small gap separates the domains."""
    n = len(ordered)
    n_domains = len({leaf[0] for leaf in ordered})
    span = 360.0 - GAP_DEG - DOMAIN_GAP * (n_domains - 1)
    step = span / n
    start = 90.0 - GAP_DEG / 2.0 - step / 2.0
    angles = {}
    offset = 0.0
    previous = None
    for index, leaf in enumerate(ordered):
        if previous is not None and leaf[0] != previous:
            offset += DOMAIN_GAP
        angles[leaf] = start - index * step - offset
        previous = leaf[0]
    return angles, step


def _polar(radius, degrees):
    theta = math.radians(degrees)
    return radius * math.cos(theta), radius * math.sin(theta)


def _arc(radius, low, high, n=None):
    n = n or max(2, int(abs(high - low) / 0.8) + 2)
    return [_polar(radius, value) for value in np.linspace(low, high, n)]


def phylum_palette(ordered):
    """Phylum -> color. The most frequent bacterial phyla get pastels, the rest alternate grays."""
    sizes = Counter(leaf[1] for leaf in ordered if leaf[0] == "Bacteria")
    named = [name for name, _n in sorted(sizes.items(), key=lambda item: (-item[1], item[0]))][: len(PASTELS)]
    palette = {name: color for name, color in zip(named, PASTELS)}
    others = [leaf[1] for leaf in ordered if leaf[0] == "Bacteria" and leaf[1] not in palette]
    for index, name in enumerate(dict.fromkeys(others)):
        palette[name] = OTHER_COLORS[index % 2]
    return palette, named


def _phylum_color(tax_or_leaf, palette):
    domain, phylum = (tax_or_leaf[0], tax_or_leaf[1]) if isinstance(tax_or_leaf, tuple) else \
        (tax_or_leaf["d"], tax_or_leaf["p"])
    if domain == "Archaea":
        return ARCHAEA_COLOR
    return palette.get(phylum, OTHER_COLORS[0])


def _ring_radii():
    """(cluster, rival) -> (inner, outer) radius; per clustering algorithm, NormCC inside and bin3C outside."""
    rings = {}
    pos = R_LEAF + GROUP_GAP
    for _key, cluster in CLUSTERS:
        for rival in RIVALS:
            rings[(cluster, rival)] = (pos, pos + RING)
            pos += RING + RING_GAP
        pos += GROUP_GAP - RING_GAP
    return rings, pos - GROUP_GAP


def _compare(counts, cluster, rival):
    """Class of one genus: MACO clean bins against those of the rival; None when both are zero."""
    maco = counts.get((cluster, "MACO"), 0)
    other = counts.get((cluster, rival), 0)
    if not maco and not other:
        return None
    if not other:
        return "maco_only"
    if not maco:
        return "other_only"
    return "maco_more" if maco > other else "equal" if maco == other else "maco_fewer"


def _clade_lines(ordered, angles):
    """Radial and arc segments of the rectangular circular cladogram."""
    segments = []

    def visit(prefix):
        depth = len(prefix)
        children = sorted({leaf[: depth + 1] for leaf in ordered if leaf[:depth] == prefix},
                          key=lambda key: -min(angles[leaf] for leaf in ordered if leaf[: depth + 1] == key))
        child_angles = []
        for child in children:
            angle = angles[child] if len(child) == len(ordered[0]) else visit(child)
            child_angles.append(angle)
            segments.append([_polar(RANK_RADII[depth], angle), _polar(RANK_RADII[depth + 1], angle)])
        if len(child_angles) > 1:
            segments.append(_arc(RANK_RADII[depth], min(child_angles), max(child_angles)))
        return 0.5 * (min(child_angles) + max(child_angles))

    visit(())
    return segments


def _panel_tree(ax, data, palette, labels):
    counts, paired = data["counts"], data["paired"]
    ordered = _order(list(counts))
    angles, step = _angles(ordered)

    phylum_span = defaultdict(list)
    for leaf in ordered:
        phylum_span[leaf[:2]].append(angles[leaf])
    wedges, colors = [], []
    for key, values in phylum_span.items():
        wedges.append(Wedge((0, 0), R_LEAF + 0.012, min(values) - step / 2.0, max(values) + step / 2.0,
                            width=R_LEAF + 0.012 - RANK_RADII[2] + 0.02))
        colors.append(_phylum_color(key, palette))
    ax.add_collection(PatchCollection(wedges, facecolors=colors, edgecolors="none", zorder=1))
    ax.add_collection(LineCollection(_clade_lines(ordered, angles), colors="#3A3A3A", linewidths=0.42,
                                     capstyle="round", zorder=3))

    rings, outer = _ring_radii()
    palette_compare = {key: color for key, _label, color in COMPARE}
    cells, cell_colors = [], []
    for (cluster, rival), (low, high) in rings.items():
        for leaf in ordered:
            cells.append(Wedge((0, 0), high, angles[leaf] - step / 2.0, angles[leaf] + step / 2.0, width=high - low))
            cell_colors.append(palette_compare.get(_compare(counts[leaf], cluster, rival), EMPTY))
    ax.add_collection(PatchCollection(cells, facecolors=cell_colors, edgecolors="white", linewidths=0.15,
                                      zorder=2))

    radius = outer + 0.022
    for leaf in ordered:
        if leaf in paired:
            x, y = _polar(radius, angles[leaf])
            ax.plot([x], [y], marker="o", markersize=3.8, color="black", markeredgewidth=0, zorder=5)
        if leaf[-1] == "unnamed genus":
            x, y = _polar(radius + (0.028 if leaf in paired else 0.0), angles[leaf])
            ax.plot([x], [y], marker="*", markersize=6.0, markerfacecolor="white", markeredgecolor="black",
                    markeredgewidth=0.55, zorder=5)

    for (cluster, rival), (low, high) in rings.items():
        ax.text(0.0, 0.5 * (low + high), "{0} vs {1}".format(cluster, rival), ha="center", va="center",
                fontsize=7.4, zorder=6)

    _genus_labels(ax, ordered, angles, outer + 0.05, labels)
    height = TREE_LIM * TREE_BOX[3] / TREE_BOX[2]
    ax.set_xlim(-TREE_LIM, TREE_LIM)
    ax.set_ylim(-height, height + 0.04)
    ax.set_aspect("equal")
    ax.axis("off")
    return ordered


def _panel_summary(ax, counts):
    """Genera per comparison class for every clustering algorithm and rival, with the clean bins of both."""
    rows, spans, pos = [], {}, 0.0
    for _key, cluster in CLUSTERS:
        start = pos
        for rival in RIVALS:
            rows.append((pos, cluster, rival))
            pos -= 1.0
        spans[cluster] = (pos + 1.0, start)
        pos -= 0.45
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    width = 0
    shown = min(10, max(2, round(0.045 * len(counts))))
    for y_pos, cluster, rival in rows:
        classes = Counter(_compare(per_leaf, cluster, rival) for per_leaf in counts.values())
        left = 0
        for key, _label, color in COMPARE:
            n = classes[key]
            ax.barh(y_pos, n, left=left, height=0.72, color=color, edgecolor="black", linewidth=EDGE_LW)
            if n >= shown:
                ax.text(left + n / 2.0, y_pos, str(n), ha="center", va="center", fontsize=9,
                        color="white" if key in ("maco_only", "other_only") else "black")
            left += n
        width = max(width, left)
        maco = sum(per_leaf.get((cluster, "MACO"), 0) for per_leaf in counts.values())
        other = sum(per_leaf.get((cluster, rival), 0) for per_leaf in counts.values())
        ax.text(1.03, y_pos, str(maco), transform=trans, ha="left", va="center", fontsize=10,
                color=COLORS["MACO"], fontweight="bold")
        ax.text(1.115, y_pos, ":", transform=trans, ha="center", va="center", fontsize=10)
        ax.text(1.14, y_pos, str(other), transform=trans, ha="left", va="center", fontsize=10, color=COLORS[rival])
    for cluster, (low, high) in spans.items():
        ax.plot([-0.25, -0.25], [low - 0.36, high + 0.36], transform=trans, color="black", lw=BRACKET_LW,
                clip_on=False, solid_capstyle="butt")
        ax.text(-0.27, 0.5 * (low + high), cluster, transform=trans, ha="right", va="center", fontsize=CLUSTER_FS,
                clip_on=False)
    ax.text(1.115, rows[0][0] + 0.55, "Clean bins", transform=trans, ha="center", va="bottom", fontsize=10,
            fontweight="bold")
    ax.set_yticks([row[0] for row in rows])
    ax.set_yticklabels(["vs " + row[2] for row in rows])
    ax.set_ylim(rows[-1][0] - 0.55, rows[0][0] + 0.55)
    ax.set_xlim(0, width)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Genera with a clean bin", fontsize=LABEL_FS, labelpad=6)
    _frame(ax)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", length=0, labelsize=10)

def _is_latin(genus):
    return bool(genus) and not any(ch.isdigit() for ch in genus.split("_")[0]) and "-" not in genus


def tree_labels(data):
    """Sheep gut: fixed genera, then those of panel c. Other datasets: genera of panel c, then the genera
    where MACO leads in the most rings. At most MAX_LABELS."""
    counts = data["counts"]
    names = list(LABELS) if data["tree_dataset"] == "sheep" else []
    present = {leaf[5] for leaf in counts} | {leaf[4] for leaf in counts if leaf[5] == "unnamed genus"}
    for item in _panel_c_items(data):
        if item["dataset"] != data["tree_dataset"]:
            continue
        for _bin_id, _bp, _info, _size, tax in item["members"]:
            name = tax["g"] or tax["f"]
            if name in present:
                names.append(name)
    if data["tree_dataset"] != "sheep":
        lead = []
        for leaf, per_leaf in counts.items():
            if leaf[5] == "unnamed genus":
                continue
            wins = sum(_compare(per_leaf, cluster, rival) in ("maco_only", "maco_more")
                       for _key, cluster in CLUSTERS for rival in RIVALS)
            lead.append((-wins, -per_leaf.get(("Leiden", "MACO"), 0), not _is_latin(leaf[5]), leaf[5]))
        names += [item[3] for item in sorted(lead) if item[0] < 0][:MAX_LABELS]
    return list(dict.fromkeys(name for name in names if name))[:MAX_LABELS if data["tree_dataset"] == "sheep" else 10]


def _genus_labels(ax, ordered, angles, radius, labels):
    """Callouts on each side of the circle, spread to a minimum vertical spacing."""
    r_label = radius + 0.11
    sides = {True: [], False: []}
    for name in labels:
        hits = [leaf for leaf in ordered if leaf[5] == name or (leaf[4] == name and leaf[5] == "unnamed genus")]
        if not hits:
            print("[WARN] no sheep gut leaf for label {0}".format(name), flush=True)
            continue
        leaf = hits[0]
        angle = angles[leaf]
        right = math.cos(math.radians(angle)) >= 0
        italic = leaf[5] == name and _is_latin(name)
        sides[right].append((angle, name, italic, r_label * math.sin(math.radians(angle))))
    for right, items in sides.items():
        items.sort(key=lambda item: -item[3])
        ys = [item[3] for item in items]
        for _ in range(500):
            moved = False
            for index in range(1, len(ys)):
                gap = ys[index - 1] - ys[index]
                if gap < LABEL_DY:
                    shift = (LABEL_DY - gap) / 2.0
                    ys[index - 1] += shift
                    ys[index] -= shift
                    moved = True
            if not moved:
                break
        sign = 1.0 if right else -1.0
        for (angle, text, italic, _start), y_label in zip(items, ys):
            y_label = max(-TREE_LIM + 0.05, min(TREE_LIM - 0.05, y_label))
            x_label = sign * max(math.sqrt(max(r_label ** 2 - y_label ** 2, 0.0)), 0.16)
            x0, y0 = _polar(radius, angle)
            x1, y1 = _polar(radius + 0.035, angle)
            x2 = x_label + sign * 0.035
            ax.plot([x0, x1, x_label, x2], [y0, y1, y_label, y_label], color="#4D4D4D", lw=0.7, zorder=4,
                    solid_capstyle="round", solid_joinstyle="round")
            ax.text(x2 + sign * 0.015, y_label, text, ha="left" if right else "right", va="center",
                    fontsize=11.5, style="italic" if italic else "normal", zorder=6)


# ---------------------------------------------------------------- panels b and c

def _frame(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.05)
        ax.spines[side].set_color("black")
    ax.tick_params(axis="both", which="both", direction="out", top=False, right=False, width=1.0, length=5.0,
                   pad=3.0, labelsize=12)


def _fate_color(method, key):
    if key in FATE_WHITE:
        return _tint(COLORS[method], FATE_WHITE[key])
    return FATE_FILL[key]


def _plot_counts(table):
    counts = {key: table[key] for key in ("clean", "light", "composite", "merged", "unbinned")}
    counts["broken"] = table["split"] + table["incomplete"]
    return counts


def _panel_fates(ax, fates):
    thick = 0.62
    pitch = 0.74
    rows = []
    pos = 0.0
    spans = {}
    for _key, cluster in reversed(CLUSTERS):
        ys = []
        for index, (query, target) in enumerate(reversed(DIRECTIONS)):
            if index == 2:
                pos += 0.22
            rows.append((pos, cluster, query, target))
            ys.append(pos)
            pos += pitch
        spans[cluster] = (min(ys), max(ys))
        pos += 0.5
    for y_pos, cluster, query, target in rows:
        table = fates[(cluster, query, target)]
        total = sum(table.values())
        counts = _plot_counts(table)
        left = 0.0
        for key, _label in PLOT_FATES:
            value = 100.0 * counts[key] / total if total else 0.0
            if value <= 0:
                continue
            ax.barh(y_pos, value, left=left, height=thick, color=_fate_color(target, key), edgecolor="black",
                    linewidth=EDGE_LW, hatch="////" if key == "merged" else None, zorder=3)
            if key == "clean" and value >= 9:
                ax.text(left + value / 2.0, y_pos, "{0:.0f}%".format(value), ha="center", va="center", fontsize=9.5,
                        color="white", zorder=4)
            left += value
        ax.text(101.5, y_pos, "n = {0}".format(total), ha="left", va="center", fontsize=9.5, clip_on=False)
    ax.set_yticks([row[0] for row in rows])
    ax.set_yticklabels(["{0} → {1}".format(row[2], row[3]) for row in rows], fontsize=10)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_ylim(-0.5, pos - 0.5 - pitch + 0.35)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Clean genomes of the query run (%)", fontsize=LABEL_FS, labelpad=6)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for cluster, (low, high) in spans.items():
        ax.plot([-0.37, -0.37], [low - thick / 2.0, high + thick / 2.0], transform=trans, color="black", lw=BRACKET_LW,
                clip_on=False, solid_capstyle="butt")
        ax.text(-0.395, 0.5 * (low + high), cluster, transform=trans, ha="right", va="center", fontsize=CLUSTER_FS,
                clip_on=False)
    _frame(ax)
    ax.tick_params(axis="y", labelsize=10)


def _genus_groups(taxa):
    """Genera of the members in order of first appearance, with counts; family when the genus is unnamed."""
    counts = Counter()
    for tax in taxa:
        counts[(tax["g"], tax["f"] or tax["o"] or tax["p"])] += 1
    out = []
    for genus, family in counts:
        if genus and _is_latin(genus):
            base, _sep, suffix = genus.partition("_")
            text = r"$\mathit{" + base + "}$" + ("_" + suffix if suffix else "")
        else:
            text = genus or family
        k = counts[(genus, family)]
        out.append(text + (r" $\times$" + str(k) if k > 1 else ""))
    return out


def _panel_c_items(data):
    items = [item for item in data["merged"]
             if item["clustering"] == "Leiden" and item["query"] == "MACO"]
    items.sort(key=lambda item: (-len(item["members"]), -sum(part[1] for part in item["members"]),
                                 item["target"], item["dataset"], item["bin"]))
    return items


def _panel_joined(ax, data, palette):
    items = _panel_c_items(data)
    shown = sorted(items[:C_ROWS], key=lambda item: (METHODS.index(item["target"]), -len(item["members"]),
                                                     -item["info"]["contamination"]))
    thick, pitch = 0.46, 1.0
    n = len(shown)
    if not n:
        ax.text(0.5, 0.9, "No NormCC or bin3C bin under Leiden holds\ntwo or more high-quality MACO genomes",
                transform=ax.transAxes, ha="center", va="center", fontsize=11)
        ax.axis("off")
        return
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    xmax = max([item["size"] / 1e6 for item in shown] + [1.0])
    for index, item in enumerate(shown):
        y_pos = (n - 1 - index) * pitch
        left = 0.0
        for _bin_id, bp, _info, _size, tax in item["members"]:
            width = bp / 1e6
            ax.barh(y_pos, width, left=left, height=thick, color=_phylum_color(tax, palette), edgecolor="black",
                    linewidth=EDGE_LW, zorder=3)
            left += width
        rest = (item["size"] - sum(part[1] for part in item["members"])) / 1e6
        if rest > 0:
            ax.barh(y_pos, rest, left=left, height=thick, color=REST_COLOR, edgecolor="black", linewidth=EDGE_LW,
                    zorder=3)
        ax.text(-0.03, y_pos + 0.13, "{0} {1}".format(item["target"], item["bin"]), transform=trans, ha="right",
                va="center", fontsize=10, color=COLORS[item["target"]], clip_on=False)
        quality = "{0:.1f}% / {1:.1f}%".format(item["info"]["completeness"], item["info"]["contamination"])
        if len(data["datasets"]) > 1:
            quality += " · " + DATASET_SHORT[item["dataset"]]
        ax.text(-0.03, y_pos - 0.22, quality, transform=trans, ha="right", va="center", fontsize=8.8, clip_on=False)
        groups = _genus_groups([part[4] for part in item["members"]])
        text = " + ".join(groups[:3]) + (" + {0} more".format(len(groups) - 3) if len(groups) > 3 else "")
        ax.text(0.0, y_pos + thick / 2.0 + 0.04, text, ha="left", va="bottom", fontsize=9.3, zorder=4)
    top = (n - 1) * pitch + 0.62
    ax.text(-0.03, top, "Bin (completeness / contamination)", transform=trans, ha="right", va="bottom",
            fontsize=10, fontweight="bold", clip_on=False)
    ax.text(0.0, top, "MACO genomes in the bin (GTDB genus)", ha="left", va="bottom", fontsize=10,
            fontweight="bold", clip_on=False)
    ax.set_ylim(-0.45 - max(6 - n, 0) * pitch, (n - 1) * pitch + 0.6)
    ax.set_xlim(0, xmax * 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.set_yticks([])
    ax.set_xlabel("Base pairs in the bin (Mb)", fontsize=LABEL_FS, labelpad=6)
    _frame(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


# ---------------------------------------------------------------- figure

def _legend_key(fig, x, y, kind, color, label):
    """One legend entry at (x, y) inches from the lower left of the figure: key, then label."""
    trans = fig.dpi_scale_trans
    if kind == "patch":
        fig.add_artist(Rectangle((x, y - KEY_H / 2.0), KEY_W, KEY_H, transform=trans, facecolor=color,
                                 edgecolor="black", linewidth=EDGE_LW))
    elif kind == "dot":
        fig.add_artist(Line2D([x + KEY_W / 2.0], [y], transform=trans, marker="o", color="none",
                              markerfacecolor="black", markeredgewidth=0, markersize=5.2))
    else:
        fig.add_artist(Line2D([x + KEY_W / 2.0], [y], transform=trans, marker="*", color="none",
                              markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.6, markersize=9.0))
    fig.text(x + KEY_W + KEY_PAD, y, label, transform=trans, ha="left", va="center", fontsize=LEGEND_FS)


def _legends(fig, ordered, palette, named):
    """Phylum, mark and comparison keys below panel a on one column grid with a common left edge."""
    phyla = []
    if any(leaf[0] == "Archaea" for leaf in ordered):
        phyla.append((ARCHAEA_COLOR, "Archaea"))
    phyla += [(palette[name], name) for name in named]
    if any(leaf[0] == "Bacteria" and leaf[1] not in named for leaf in ordered):
        phyla.append((OTHER_COLORS[0], OTHER_LABEL))
    classes = [(color, label) for _key, label, color in COMPARE]

    renderer = fig.canvas.get_renderer()
    widest = 0.0
    for _color, label in phyla + classes:
        if label == OTHER_LABEL:
            continue
        probe = fig.text(0, 0, label, fontsize=LEGEND_FS)
        widest = max(widest, probe.get_window_extent(renderer).width / fig.dpi)
        probe.remove()
    x0 = LEGEND_X
    span = TREE_BOX[0] + TREE_BOX[2] - x0
    ncol = max(len(classes), min(LEGEND_NCOL, int(span // (KEY_W + KEY_PAD + widest + 0.12))))
    col = span / ncol
    trans = fig.dpi_scale_trans

    y = LEGEND_TOP
    fig.text(x0, y, "Phylum (GTDB)", transform=trans, ha="left", va="center", fontsize=LEGEND_FS,
             fontweight="bold")
    for index, (color, label) in enumerate(phyla):
        if index % ncol == 0:
            y -= LEGEND_ROW
        _legend_key(fig, x0 + (index % ncol) * col, y, "patch", color, label)
    y -= LEGEND_ROW + 0.06
    _legend_key(fig, x0, y, "dot", None, "Species with ≥2 MACO genomes (Leiden)")
    _legend_key(fig, x0 + 3 * col, y, "star", None, "No genus name")
    y -= LEGEND_ROW + 0.12
    fig.text(x0, y, "Clean bins per genus in MACO compared with NormCC or bin3C (rings and bars)", transform=trans,
             ha="left", va="center", fontsize=LEGEND_FS, fontweight="bold")
    y -= LEGEND_ROW
    for index, (color, label) in enumerate(classes):
        _legend_key(fig, x0 + index * col, y, "patch", color, label)


def _fate_legend(fig, ax):
    handles = []
    for key, label in PLOT_FATES:
        color = _tint("#555555", FATE_WHITE[key]) if key in FATE_WHITE else FATE_FILL[key]
        handles.append(Patch(facecolor=color, edgecolor="black", linewidth=EDGE_LW, label=label,
                             hatch="////" if key == "merged" else None))
    box = ax.get_position()
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5 * (box.x0 + box.x1) - 0.04, box.y1 + 0.01),
               ncol=3, fontsize=10.5, handlelength=1.2, handleheight=1.0, handletextpad=0.4, columnspacing=0.9,
               borderaxespad=0.0, frameon=False)


def _box(spec):
    left, bottom, width, height = spec
    return [left / FIG_W, bottom / FIG_H, width / FIG_W, height / FIG_H]


def draw(data, path):
    _style()
    ordered = _order(list(data["counts"]))
    palette, named = phylum_palette(ordered)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_tree = fig.add_axes(_box(TREE_BOX))
    _panel_tree(ax_tree, data, palette, tree_labels(data))
    _panel_summary(fig.add_axes(_box(SUMMARY_BOX)), data["counts"])
    ax_fate = fig.add_axes(_box(FATE_BOX))
    _panel_fates(ax_fate, data["fates"])
    ax_join = fig.add_axes(_box(JOIN_BOX))
    _panel_joined(ax_join, data, palette)
    _legends(fig, ordered, palette, named)
    _fate_legend(fig, ax_fate)
    top = FATE_BOX[1] + FATE_BOX[3] + 0.85
    letters = (
        ("a", TREE_BOX[0] + 0.1, top),
        ("b", 9.3, top),
        ("c", 9.3, JOIN_BOX[1] + JOIN_BOX[3] + 0.5),
    )
    for letter, x, y in letters:
        fig.text(x / FIG_W, y / FIG_H, letter, fontsize=20, fontweight="bold", ha="left", va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    plt.close(fig)
    print("[FIG] {0}".format(path), flush=True)


def main():
    data = compute(list(DATASETS), "sheep")
    report(data)
    draw(data, OUT)


if __name__ == "__main__":
    main()
