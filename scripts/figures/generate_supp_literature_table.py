#!/usr/bin/env python
"""Write Supplementary Table S1, the surveyed replay literature's parameters.

Reads the 57-paper survey, ``paper/tables/table_s1_literature_parameters.csv``
(Supplementary Data 1, the paper's one copy), and writes
``paper/tables/table_s1_literature.tex``, the table ``paper/main.tex`` inputs.
Needs no lab data.

Usage::

    uv run python scripts/figures/generate_supp_literature_table.py
"""

from __future__ import annotations

from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.literature import write_table


def main() -> None:
    print(f"wrote {write_table().relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
