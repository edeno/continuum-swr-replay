"""The surveyed replay literature: Supplementary Table S1 and its source.

The table is the 57-paper survey of candidate-event detection and decoding
parameters that ``ripple_detection`` packages
(``ripple_detection/data/literature_detection_parameters.csv``), audited there
against each paper's methods, supplement and released code. This module reads
that file, renders a selection of its columns as the manuscript's LaTeX table,
and copies the full file beside it as supplementary data. It changes no value.
"""

from __future__ import annotations

import re
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import pandas as pd

SOURCE_PACKAGE = "ripple_detection"
SOURCE_FILE = "literature_detection_parameters.csv"

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


def source_path() -> Path:
    """Path of the packaged survey CSV."""
    return Path(str(files(f"{SOURCE_PACKAGE}.data") / SOURCE_FILE))


def source_version() -> str:
    """Installed version of the package the survey comes from."""
    return version(SOURCE_PACKAGE.replace("_", "-"))


def load_survey() -> pd.DataFrame:
    """The survey with every cell as its literal text.

    Returns
    -------
    survey : pd.DataFrame, shape (n_papers, 35)
        One row per paper, in the package's order (newest first). ``#N/A`` and
        ``Not reported`` are kept as written: the first means the parameter does
        not apply to the paper's method, the second that the paper does not
        state it.
    """
    return pd.read_csv(
        source_path(), dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )


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


def table_tex(table: pd.DataFrame, n_papers: int, source: str) -> str:
    """The LaTeX of Supplementary Table S1, a ``longtable``.

    Parameters
    ----------
    table : pd.DataFrame, shape (n_papers, 14)
        From :func:`display_table`.
    n_papers : int
        Number of surveyed papers, for the caption.
    source : str
        The package and version the survey comes from, for the caption.

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
        r"settings, is Supplementary Data~1 (\texttt{table\_s1\_literature\_parameters.csv}); "
        rf"it is the survey packaged in {_escape(source)}, where each value's source in "
        r"the paper, supplement or released code is recorded.}"
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


def write_table(out_dir: Path) -> tuple[Path, Path]:
    """Write Supplementary Table S1 and its full CSV into `out_dir`.

    Parameters
    ----------
    out_dir : Path
        Usually ``paper/tables/``; created if missing.

    Returns
    -------
    tex_path, csv_path : Path
        The LaTeX table and the copy of the full survey.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    survey = load_survey()
    source = f"{SOURCE_PACKAGE} {source_version()}"
    tex_path = out_dir / "table_s1_literature.tex"
    csv_path = out_dir / "table_s1_literature_parameters.csv"
    tex_path.write_text(table_tex(display_table(survey), len(survey), source))
    csv_path.write_bytes(source_path().read_bytes())
    return tex_path, csv_path
