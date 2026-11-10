"""Supplementary Table S1 shows the packaged literature survey without changing it.

The table is rendered from the survey ``ripple_detection`` packages. These tests
check the display rules on small cases, that every surveyed paper appears once
with its values, that the supplementary CSV is the packaged file byte for byte,
and that the committed table is what the generator writes from the installed
package, so a package update cannot leave a stale table in the paper.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from continuum_swr_replay.reported import literature

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


def test_shown_values_are_the_survey_values(survey):
    table = literature.display_table(survey)
    for header, column in literature.SHOWN_COLUMNS.items():
        expected = survey[column].map(literature._cell)
        assert table[header].tolist() == expected.tolist(), header


def test_rendered_table_has_one_row_per_paper(survey):
    tex = literature.table_tex(literature.display_table(survey), len(survey), "pkg 1.0")
    body = tex.split(r"\endlastfoot")[1].split(r"\end{longtable}")[0]
    rows = [line for line in body.splitlines() if line.endswith(r"\\")]
    assert len(rows) == len(survey)
    assert r"\label{tab:literature}" in tex
    assert "pkg 1.0" in tex


def test_write_table_copies_the_packaged_survey(tmp_path):
    tex_path, csv_path = literature.write_table(tmp_path)
    assert csv_path.read_bytes() == literature.source_path().read_bytes()
    assert re.search(r"\\begin\{longtable\}", tex_path.read_text())


def test_committed_table_is_current(tmp_path):
    tex_path, csv_path = literature.write_table(tmp_path)
    committed_tex = PAPER_TABLES / tex_path.name
    committed_csv = PAPER_TABLES / csv_path.name
    assert committed_tex.read_text() == tex_path.read_text(), (
        "paper/tables/ is stale: run scripts/figures/generate_supp_literature_table.py"
    )
    assert committed_csv.read_bytes() == csv_path.read_bytes()
