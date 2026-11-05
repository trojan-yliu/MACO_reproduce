#!/usr/bin/env python3
"""Figure S10. Figure 4 for Mount Rittmann.

Reads Supplementary Table S3 and Supplementary Table S5 through figure_4.py.
Writes Supplementary_Figure_S10.png at 600 dpi.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Main_Figures_Scripts"))
import figure_4 as drawing  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "Supplementary_Figures" / "Supplementary_Figure_S10.png"


def main():
    data = drawing.compute(["rittmann"], "rittmann")
    drawing.report(data)
    drawing.draw(data, OUT)


if __name__ == "__main__":
    main()
