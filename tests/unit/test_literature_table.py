"""Supplementary Table S1 shows the literature survey without changing it.

The table is rendered from the survey CSV, ``paper/tables/table_s1_literature_parameters.csv``.
These tests check the display rules on small cases, that every surveyed paper
appears once, a few rows checked by hand against the CSV, that the survey's
bounds are ordered, and that the committed table is what the generator writes
from the CSV, so a corrected value cannot leave a stale table in the paper.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from continuum_swr_replay.reported import literature, part0

PAPER_TABLES = Path(__file__).resolve().parents[2] / "paper" / "tables"


@pytest.fixture(scope="module")
def survey() -> pd.DataFrame:
    return literature.load_survey()


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        ("150", "250", "150--250"),
        ("50", "#N/A", "≥50"),
        ("#N/A", "500", "≤500"),
        ("#N/A", "#N/A", "--"),
        ("Not reported", "Not reported", "n.r."),
        ("Not reported", "250", "n.r.--250"),
        ("Not reported", "#N/A", "n.r."),
        ("#N/A", "Not reported", "n.r."),
    ],
)
def test_range_cells(low, high, expected):
    assert literature._range(low, high) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("#N/A", "--"),
        ("Not reported", "n.r."),
        ("UMAP\nBayes", "UMAP; Bayes"),
        ("\nTemplate Matching\nBayes", "Template Matching; Bayes"),
        ("3", "3"),
    ],
)
def test_cell_markers_and_line_breaks(text, expected):
    assert literature._cell(text) == expected


def test_escape_specials_and_superscripts():
    assert literature._escape("15% of >5 & #2") == r"15\% of \textgreater{}5 \& \#2"
    assert literature._escape("R^2 significance") == "R$^{2}$ significance"
    assert literature._escape("≥50") == r"$\geq$50"
    assert literature._escape("150--250") == "150--250"


def test_every_paper_appears_once_in_survey_order(survey):
    table = literature.display_table(survey)
    assert len(table) == len(survey) == 57
    assert table["Study"].is_unique
    assert all(
        s.startswith(f"{a} {y}")
        for s, a, y in zip(
            table["Study"], survey["First Author"], survey["Year"], strict=True
        )
    )
    assert table["Last author"].tolist() == survey["Last Author"].tolist()


# Rows of the table, written by hand from the CSV's cells: a minimum-only
# duration, a maximum-only one with many inapplicable cells and a multi-line
# Decoding cell, and a paper told apart by its journal with unreported cells.
HAND_CHECKED_ROWS = [
    {
        "Study": "Mallory 2025",
        "Last author": "Foster",
        "Trigger": "MUA, Decoding",
        "SWR z (SD)": "3",
        "MUA z (SD)": "3",
        "Speed (cm/s)": "5",
        "Min. cells": "10",
        "SWR band (Hz)": "150--250",
        "Duration (ms)": "≥50",
        "SWR electrodes": "1",
        "Decoding": "Bayes; rank-order correlation (track control)",
        "Time bin (ms)": "20 (track); 80 (arena)",
        "Position bin (cm)": "2",
        "Trajectory": "Weighted correlation (track); posterior center of mass (arena)",
    },
    {
        "Study": "Ólafsdóttir 2015",
        "Last author": "Spiers",
        "Trigger": "MUA",
        "SWR z (SD)": "--",
        "MUA z (SD)": "--",
        "Speed (cm/s)": "--",
        "Min. cells": "15% of template cells",
        "SWR band (Hz)": "--",
        "Duration (ms)": "≤300",
        "SWR electrodes": "--",
        "Decoding": "Template Matching; Bayes",
        "Time bin (ms)": "5",
        "Position bin (cm)": "1",
        "Trajectory": "Radon transform",
    },
    {
        "Study": "Farooq 2019 (Neuron)",
        "Last author": "Dragoi",
        "Trigger": "MUA",
        "SWR z (SD)": "n.r.",
        "MUA z (SD)": "2",
        "Speed (cm/s)": "1",
        "Min. cells": "5",
        "SWR band (Hz)": "175--225",
        "Duration (ms)": "100--800",
        "SWR electrodes": "n.r.",
        "Decoding": "Bayes",
        "Time bin (ms)": "20",
        "Position bin (cm)": "1",
        "Trajectory": "Weighted correlation, Radon transform",
    },
]


@pytest.mark.parametrize("row", HAND_CHECKED_ROWS, ids=lambda row: row["Study"])
def test_rows_checked_by_hand(survey, row):
    table = literature.display_table(survey).set_index("Study", drop=False)
    assert table.loc[row["Study"]].to_dict() == row


def test_shown_cells_are_not_blank(survey):
    shown = survey[list(literature.SHOWN_COLUMNS.values())]
    blank = shown.apply(lambda column: column.str.strip() == "")
    assert not blank.any().any(), shown.columns[blank.any()].tolist()


@pytest.mark.parametrize(
    ("low", "high"),
    [
        ("SWR Low Band (Hz)", "SWR High Band (Hz)"),
        ("Min. Duration (ms)", "Max Duration (ms)"),
    ],
)
def test_survey_lower_bounds_are_below_upper_bounds(survey, low, high):
    numbers = part0.numeric(survey[[low, high]]).dropna()
    assert len(numbers) > 0
    swapped = numbers.index[numbers[low] >= numbers[high]]
    papers = survey.loc[swapped, ["First Author", "Year"]]
    assert papers.empty, papers.to_dict("records")


def test_rendered_table_has_one_row_per_paper(survey):
    tex = literature.table_tex(literature.display_table(survey), len(survey))
    body = tex.split(r"\endlastfoot")[1].split(r"\end{longtable}")[0]
    rows = [line for line in body.splitlines() if line.endswith(r"\\")]
    assert len(rows) == len(survey)
    assert r"\label{tab:literature}" in tex


def test_committed_table_is_current(tmp_path):
    fresh = literature.write_table(tex_path=tmp_path / "table.tex")
    committed = PAPER_TABLES / "table_s1_literature.tex"
    assert committed.read_text() == fresh.read_text(), (
        "paper/tables/table_s1_literature.tex is stale: run "
        "scripts/figures/generate_supp_literature_table.py"
    )
