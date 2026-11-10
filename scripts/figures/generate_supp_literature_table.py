#!/usr/bin/env python
"""Write Supplementary Table S1, the surveyed replay literature's parameters.

Reads the 57-paper survey that ``ripple_detection`` packages and writes
``paper/tables/table_s1_literature.tex`` (the table ``paper/main.tex`` inputs)
and ``paper/tables/table_s1_literature_parameters.csv`` (the full survey,
Supplementary Data 1). Needs no lab data.

Usage::

    uv run python scripts/figures/generate_supp_literature_table.py
"""

from __future__ import annotations

from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.literature import write_table

OUT_DIR = ROOT_DIR / "paper" / "tables"


def main() -> None:
    for path in write_table(OUT_DIR):
        print(f"wrote {path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
