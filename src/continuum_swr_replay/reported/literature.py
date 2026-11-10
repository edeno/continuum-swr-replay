"""The surveyed replay literature: Supplementary Table S1 and its source.

The table is the 57-paper survey of candidate-event detection and decoding
parameters, kept in this repository as ``paper/tables/table_s1_literature_parameters.csv``
(Supplementary Data 1). That file is the one copy the paper reads: correct a
value there. This module renders a selection of its columns as the
manuscript's LaTeX table. It changes no value.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from continuum_swr_replay.paths import ROOT_DIR

SURVEY_PATH = ROOT_DIR / "paper" / "tables" / "table_s1_literature_parameters.csv"
"""The survey, one row per paper: the paper's only copy."""

#: Columns shown in the PDF table: header -> source column. The study, the band
#: and the durations combine source columns and are built in :func:`display_table`.
SHOWN_COLUMNS: dict[str, str] = {
    "Last author": "Last Author",
    "Trigger": "Detection",
    "SWR z (SD)": "SWR Z-score Thresh. (STD)",
    "MUA z (SD)": "MUA Z-score Thresh. (STD)",
    "Speed (cm/s)": "Animal Speed (cm/s)",
    "Min. cells": "Min. Cells (#)",
    "SWR electrodes": "SWR electrodes (#)",
    "Decoding": "Decoding",
    "Time bin (ms)": "Time bin (ms)",
    "Position bin (cm)": "Position bin (cm)",
    "Trajectory": "Trajectory",
}

NOT_APPLICABLE = "#N/A"
NOT_REPORTED = "Not reported"

_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    ">": r"\textgreater{}",
    "<": r"\textless{}",
    "≥": r"$\geq$",
    "≤": r"$\leq$",
}


def load_survey(path: Path = SURVEY_PATH) -> pd.DataFrame:
    """The survey with every cell as its literal text.

    Parameters
    ----------
    path : Path
        The survey CSV; :data:`SURVEY_PATH` by default.

    Returns
    -------
    survey : pd.DataFrame, shape (n_papers, 35)
        One row per paper, in the file's order (newest first). ``#N/A`` and
        ``Not reported`` are kept as written: the first means the parameter does
        not apply to the paper's method, the second that the paper does not
        state it.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _cell(text: str) -> str:
    """One cell's display text: missing markers abbreviated, line breaks joined."""
    text = "; ".join(part.strip() for part in text.splitlines() if part.strip())
    if text == NOT_APPLICABLE:
        return "--"
    if text == NOT_REPORTED:
        return "n.r."
    return text


def _range(low: str, high: str) -> str:
    """Two bounds as one cell.

    ``150--250`` when both are given, ``≥50`` or ``≤500`` when only one applies,
    and the shared marker when neither is a value.
    """
    low, high = _cell(low), _cell(high)
    if low == high and low in ("--", "n.r."):
        return low
    if high == "--":
        return f"≥{low}"
    if low == "--":
        return f"≤{high}"
    return f"{low}--{high}"


def display_table(survey: pd.DataFrame) -> pd.DataFrame:
    """The columns the PDF shows, as display text (not yet LaTeX-escaped).

    Parameters
    ----------
    survey : pd.DataFrame, shape (n_papers, 35)
        From :func:`load_survey`.

    Returns
    -------
    table : pd.DataFrame, shape (n_papers, 14)
        ``Study`` (first author and year, with the journal when two papers
        share both), the :data:`SHOWN_COLUMNS`, the SWR
        band and the minimum and maximum event durations, in survey order.
    """
    study = survey["First Author"] + " " + survey["Year"]
    # Two papers share a first author and year (Farooq & Dragoi 2019); name their journals.
    shared = study.duplicated(keep=False)
    study[shared] = study[shared] + " (" + survey.loc[shared, "Journal"] + ")"
    out = pd.DataFrame({"Study": study})
    for header, column in SHOWN_COLUMNS.items():
        out[header] = survey[column].map(_cell)
    out.insert(
        out.columns.get_loc("SWR electrodes"),
        "SWR band (Hz)",
        [
            _range(lo, hi)
            for lo, hi in zip(
                survey["SWR Low Band (Hz)"], survey["SWR High Band (Hz)"], strict=True
            )
        ],
    )
    out.insert(
        out.columns.get_loc("SWR electrodes"),
        "Duration (ms)",
        [
            _range(lo, hi)
            for lo, hi in zip(
                survey["Min. Duration (ms)"], survey["Max Duration (ms)"], strict=True
            )
        ],
    )
    return out


_SUPERSCRIPT = re.compile(r"\^(\d+)")


def _escape(text: str) -> str:
    """Escape LaTeX's special characters; ``--`` is left as an en dash, ``^2`` a superscript."""
    pieces = _SUPERSCRIPT.split(text)
    out = []
    for i, piece in enumerate(pieces):
        if i % 2:
            out.append(f"$^{{{piece}}}$")
        else:
            out.append("".join(_LATEX_SPECIALS.get(ch, ch) for ch in piece))
    return "".join(out)


#: Column widths of the landscape table (the text columns wrap).
_COLUMN_SPEC = (
    r">{\raggedright\arraybackslash}p{1.8cm}"  # Study
    r">{\raggedright\arraybackslash}p{1.8cm}"  # Last author
    r">{\raggedright\arraybackslash}p{1.3cm}"  # Trigger
    r">{\raggedright\arraybackslash}p{1.1cm}"  # SWR z
    r">{\raggedright\arraybackslash}p{0.8cm}"  # MUA z
    r">{\raggedright\arraybackslash}p{0.9cm}"  # Speed
    r">{\raggedright\arraybackslash}p{1.2cm}"  # Min. cells
    r">{\raggedright\arraybackslash}p{1.2cm}"  # SWR band
    r">{\raggedright\arraybackslash}p{1.3cm}"  # Duration
    r">{\raggedright\arraybackslash}p{1.2cm}"  # SWR electrodes
    r">{\raggedright\arraybackslash}p{2.1cm}"  # Decoding
    r">{\raggedright\arraybackslash}p{1.3cm}"  # Time bin
    r">{\raggedright\arraybackslash}p{1.2cm}"  # Position bin
    r">{\raggedright\arraybackslash}p{2.6cm}"  # Trajectory
)


def table_tex(table: pd.DataFrame, n_papers: int) -> str:
    """The LaTeX of Supplementary Table S1, a ``longtable``.

    Parameters
    ----------
    table : pd.DataFrame, shape (n_papers, 14)
        From :func:`display_table`.
    n_papers : int
        Number of surveyed papers, for the caption.

    Returns
    -------
    tex : str
        The ``longtable``, for a ``landscape`` page; needs the ``longtable``,
        ``array`` and ``booktabs`` packages.
    """
    header = " & ".join(rf"\textbf{{{_escape(h)}}}" for h in table.columns) + r" \\"
    caption = (
        rf"\caption{{Candidate-event detection and decoding parameters in {n_papers} "
        r"published replay studies. Trigger: the signal that defines candidate events "
        r"(SWR, ripple-band LFP power; MUA, multiunit activity; Decoding, the decoded "
        r"position itself). Thresholds are in standard deviations above the mean. "
        r"Duration is the minimum--maximum event duration ($\geq$, a minimum only; $\leq$, a "
        r"maximum only). --, not applicable to the "
        r"study's method; n.r., not reported. The full table, with each study's DOI, "
        r"detection and decoding notes, smoothing, merging, shuffle and significance "
        r"settings, is Supplementary Data~1 (\texttt{table\_s1\_literature\_parameters.csv}). "
        r"Each value was checked against the study's methods, supplement or released code.}"
    )
    rows = [
        " & ".join(_escape(str(v)) for v in row) + r" \\"
        for row in table.itertuples(index=False)
    ]
    lines = [
        "% Generated by scripts/figures/generate_supp_literature_table.py --- do not edit by hand.",
        r"\scriptsize",
        r"\setlength{\LTcapwidth}{\linewidth}",
        r"\setlength{\tabcolsep}{2.5pt}",
        rf"\begin{{longtable}}{{{_COLUMN_SPEC}}}",
        caption + r"\label{tab:literature} \\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
        *rows,
        r"\end{longtable}",
        "",
    ]
    return "\n".join(lines)


TABLE_PATH = ROOT_DIR / "paper" / "tables" / "table_s1_literature.tex"
"""Where :func:`write_table` writes the LaTeX table by default."""


def write_table(survey_path: Path = SURVEY_PATH, tex_path: Path = TABLE_PATH) -> Path:
    """Write Supplementary Table S1's LaTeX from the survey.

    Parameters
    ----------
    survey_path : Path
        The survey CSV; :data:`SURVEY_PATH` by default.
    tex_path : Path
        Where to write the table; :data:`TABLE_PATH` by default.

    Returns
    -------
    tex_path : Path
    """
    survey = load_survey(survey_path)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(table_tex(display_table(survey), len(survey)))
    return tex_path
