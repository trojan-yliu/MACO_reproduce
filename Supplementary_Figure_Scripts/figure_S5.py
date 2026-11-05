#!/usr/bin/env python3
"""Figure S5. Same panels as Figure S4, at a 5% contamination gate.

S5 keeps bins with contamination <= 5%. Completeness stays >= 50%.
Panels, colors and callouts match Figure S4.
"""

from pathlib import Path

import figure_S4 as drawing

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "Supplementary_Figures" / "Supplementary_Figure_S5.png"


def main():
    drawing.CONT_MAX = 5.0
    print("[GATE] contamination <= {0:g}%".format(drawing.CONT_MAX), flush=True)
    drawing.draw(*drawing.compute(), OUT)


if __name__ == "__main__":
    main()
