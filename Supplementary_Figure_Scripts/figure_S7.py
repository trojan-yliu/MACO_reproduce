#!/usr/bin/env python3
"""Figure S7. Genus Venn diagrams at three contamination gates.

Leiden, Infomap and ImputeCC. Each row is one gate, from 1% to 5%.
ONT is the left half and PacBio the right half. A genus enters a method
when it is the majority genome of a bin with completeness >= 50% and
contamination at or below the row's gate. Circle order is MACO, NormCC,
bin3C, as in Figure S3.
"""

from pathlib import Path

import figure_S6 as drawing

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "Supplementary_Figures" / "Supplementary_Figure_S7.png"


def main():
    drawing.draw(
        drawing._load_rows(),
        OUT,
        "genus",
        ("Leiden", "Infomap", "ImputeCC"),
    )


if __name__ == "__main__":
    main()
