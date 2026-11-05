<p align="center">
  <img src="logo.png" alt="MACO" width="960">
</p>

This repository stores the figures of [MACO](https://github.com/trojan-yliu/MACO) and the tables they are drawn from. `Main_Figures/` holds Main Figures 1–5. `Supplementary_Figures/` holds Supplementary Figures S1–S11. `Supplementary_Tables/` holds Supplementary Tables S1–S7. Main Figure 2 and Supplementary Figures S1–S7 are computed from three MOCK1 tables in `Main_Figures_Scripts/data/`: `contig_labels.csv` (contig, length, reference genome, and status), `reference_lengths.csv` (reference genome and length), and `bin_membership.csv` (which contig sits in which bin). The scripts redraw the figures from these tables. They do not assemble reads, normalize contacts, or bin contigs.

## Contents

- [Clone the repository](#clone-the-repository)
- [Install](#install)
- [Redraw the figures](#redraw-the-figures)
- [Inputs](#inputs)
- [Output](#output)
- [License](#license)
- [Contact](#contact)

## Clone the repository

```bash
git clone https://github.com/trojan-yliu/MACO_reproduce.git
cd MACO_reproduce
```

Keep `Main_Figures`, `Main_Figures_Scripts`, `Supplementary_Figures`, `Supplementary_Figure_Scripts`, and `Supplementary_Tables` side by side. A script finds its tables from its own path.

```text
Main_Figures/                  Main_Figure_1.png through Main_Figure_5.png
Main_Figures_Scripts/          figure_2.py through figure_5.py
Main_Figures_Scripts/data/     MOCK1 labels, reference lengths, and bin membership
Supplementary_Figures/         Supplementary_Figure_S1.png through S11.png
Supplementary_Figure_Scripts/  figure_S1.py through figure_S11.py
Supplementary_Tables/          Supplementary_Table_S1.xlsx through S7.xlsx
```

## Install

```bash
conda env create -f maco_reproduce.yaml
conda activate maco_reproduce
```

The environment is named `maco_reproduce`. It contains Python, matplotlib, numpy, pandas, and openpyxl, and it is separate from `maco_env` and `maco_prep` in the MACO repository. pandas reads the workbooks with openpyxl. When Arial is installed, the figures use it. Otherwise they use Helvetica, Liberation Sans, or DejaVu Sans. DejaVu Sans comes with matplotlib.

## Redraw the figures

Each script takes no arguments, writes one PNG at 600 dpi, and replaces the file of the same name. The working directory does not matter. Main Figure 1 is a schematic stored as `Main_Figures/Main_Figure_1.png`. No script draws it.

```bash
python Main_Figures_Scripts/figure_2.py
```

From this directory, redraw every scripted figure:

```bash
for script in Main_Figures_Scripts/figure_*.py Supplementary_Figure_Scripts/figure_S*.py
do
    python "$script"
done
```

| Script | Figure | Source |
|--------|--------|--------|
| `figure_2.py` | Main Figure 2. MOCK1 precision, recall, F1, ARI, NMI, and the per-bin quality ratio. | `data/` |
| `figure_3.py` | Main Figure 3. Real-data bins with contamination at most 1%, stacked by completeness. | Supplementary Table S3 |
| `figure_4.py` | Main Figure 4. Sheep-gut lineages, and tracing of clean genomes across normalizations. | Supplementary Tables S3 and S5 |
| `figure_5.py` | Main Figure 5. Host range of the sheep-gut plasmidome. | Supplementary Table S6 |
| `figure_S1.py` | Supplementary Figure S1. MOCK1 bin counts and bin lengths. | `data/` |
| `figure_S2.py` | Supplementary Figure S2. MOCK1 precision and recall. | `data/` |
| `figure_S3.py` | Supplementary Figure S3. MOCK1 panels at contamination at most 1%. | `data/` |
| `figure_S4.py` | Supplementary Figure S4. The same panels at contamination at most 2%. | `data/` |
| `figure_S5.py` | Supplementary Figure S5. The same panels at contamination at most 5%. The drawing code is `figure_S4.py`. | `data/` |
| `figure_S6.py` | Supplementary Figure S6. Species Venn diagrams for Infomap and ImputeCC, at 1%, 2%, and 5%. | `data/` |
| `figure_S7.py` | Supplementary Figure S7. Genus Venn diagrams for Leiden, Infomap, and ImputeCC. The drawing code is `figure_S6.py`. | `data/` |
| `figure_S8.py` | Supplementary Figure S8. Real-data log10(completeness / contamination). | Supplementary Table S3 |
| `figure_S9.py` | Supplementary Figure S9. Figure 4 for Mount Melbourne. | Supplementary Tables S3 and S5, through `figure_4.py` |
| `figure_S10.py` | Supplementary Figure S10. Figure 4 for Mount Rittmann. | Supplementary Tables S3 and S5, through `figure_4.py` |
| `figure_S11.py` | Supplementary Figure S11. Figure 4 for the mouse fecal pellet. | Supplementary Tables S3 and S5, through `figure_4.py` |

`figure_S1.py` and `figure_S2.py` import `figure_2.py`. For S3, S4, and S5, completeness stays at or above 50%, and only the contamination gate changes.

## Inputs

### MOCK1 tables

Main Figure 2 and Supplementary Figures S1 through S7 read `Main_Figures_Scripts/data/`.

| File | Content |
|------|---------|
| `contig_labels.csv` | Platform, contig, length, reference genome, and status. A genome is used only when the status is `ok`. |
| `reference_lengths.csv` | Reference genome and its length. |
| `bin_membership.csv` | Platform, clustering, normalization, bin, and contig. |

Bin lengths come from `contig_labels.csv`. The scripts do not read bin FASTA files.

### Supplementary tables

`figure_4.py` reads sheet `Table S3`, plus sheets `Genome fates` and `Merged bins` in Supplementary Table S5. `figure_5.py` reads sheet `Table S6`. `figure_3.py` and `figure_S8.py` read the first sheet of Supplementary Table S3. `figure_S9.py`, `figure_S10.py`, and `figure_S11.py` use those same S3 and S5 sheets through `figure_4.py`. In every workbook the header is row 3. Leave those file names and sheet names as they are.

Supplementary Tables S1, S2, S4, and S7 are stored with the others. No drawing script reads them.

## Output

A script replaces the PNG of the same name. Nothing is written outside this repository.

```text
Main_Figures/Main_Figure_2.png through Main_Figure_5.png
Supplementary_Figures/Supplementary_Figure_S1.png through Supplementary_Figure_S11.png
```

## License

Released under the [MIT License](LICENSE).

## Contact

Yangyang Liu (yliu9184@usc.edu), Department of Quantitative and Computational Biology, University of Southern California.
