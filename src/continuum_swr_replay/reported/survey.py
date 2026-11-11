"""The survey summary: the surveyed literature the introduction describes.

The introduction summarizes how published replay studies detect candidate
events and decode them, from the survey in Supplementary Table S1
(``continuum_swr_replay.reported.literature``). The values here are computed
from that survey, so a correction to it reaches the text.

A numeric parameter keeps a cell only when it is a bare number. The other forms
a cell takes are listed in :data:`EXCLUDED_FORMS` -- a missing marker
(``#N/A``, ``Not reported``), a bound (``>4``), a range (``1.8-1.9``), a share
(``15% of ensemble``), ``Adaptive`` and the few cells that carry several values
or a qualifier, each listed in :data:`ANNOTATED_CELLS` -- and are left out; a
cell of any other form (``4 cm/s``, ``5 (cm/s)``, a blank) is an error.
Each mode, minimum and maximum is over the papers whose cell is a number. A tie
for the mode is an error rather than a silent choice. The trigger shares read
the Detection column as a set of signals, each one of :data:`TRIGGERS`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported import literature
from continuum_swr_replay.reported.values import Summary, Tags, Value, file_record

TAGS = {"survey": Tags(("0.1",), ("paper",))}
"""The questions every survey value answers, and where it is used."""

#: The parameters the introduction quotes: key stem -> (columns pooled, unit, what it is).
PARAMETERS: dict[str, tuple[tuple[str, ...], str, str]] = {
    "threshold": (
        ("SWR Z-score Thresh. (STD)", "MUA Z-score Thresh. (STD)"),
        "SD",
        "candidate-event threshold on z-scored ripple power or multiunit "
        "activity, the ripple and multiunit columns pooled (a paper with "
        "both contributes both)",
    ),
    "band_low": (("SWR Low Band (Hz)",), "Hz", "lower edge of the ripple band"),
    "band_high": (("SWR High Band (Hz)",), "Hz", "upper edge of the ripple band"),
    "speed": (
        ("Animal Speed (cm/s)",),
        "cm/s",
        "running-speed limit for candidate events",
    ),
    "cells": (
        ("Min. Cells (#)",),
        "cells",
        "minimum number of active cells per event",
    ),
    "shortest": (("Min. Duration (ms)",), "ms", "minimum event duration"),
    "longest": (("Max Duration (ms)",), "ms", "maximum event duration"),
    "time_bin": (("Time bin (ms)",), "ms", "decoding time bin"),
    "position_bin": (("Position bin (cm)",), "cm", "decoding position bin"),
}

PARAMETER_COLUMNS: tuple[str, ...] = tuple(
    column for columns, _, _ in PARAMETERS.values() for column in columns
)
"""The survey columns :data:`PARAMETERS` reads, in order."""

_NUMBER = r"\d+(\.\d+)?"
_BARE_NUMBER = re.compile(_NUMBER)

ANNOTATED_CELLS: tuple[str, ...] = (
    "7 x SD (not z-scored)",
    "20 (track); 80 (arena)",
    "40-600 (adaptive sequence window)",
    "5, 6.7, 10 (analysis variants)",
)
"""Cells with several values or a qualifier, read and left out one by one. A new
annotated cell is an error until it is added here, so that it is looked at."""

#: The forms of a parameter cell that are not a bare number and are left out.
#: Each pattern must match the whole (stripped) cell.
EXCLUDED_FORMS: dict[str, re.Pattern[str]] = {
    "missing marker": re.compile(
        f"{re.escape(literature.NOT_APPLICABLE)}|{re.escape(literature.NOT_REPORTED)}"
    ),
    "bound": re.compile(rf"[<>≥≤]\s*{_NUMBER}"),
    "range": re.compile(rf"{_NUMBER}\s*-\s*{_NUMBER}"),
    "share": re.compile(rf"{_NUMBER}%( of .+)?|\d+/\d+ of .+"),
    "adaptive": re.compile(r"Adaptive( \(.+\))?"),
    "annotated": re.compile("|".join(map(re.escape, ANNOTATED_CELLS))),
}

TRIGGERS = frozenset({"SWR", "MUA", "Decoding"})
"""The signals a Detection cell may name, comma-separated."""

_BAYES = re.compile(r"(?<!\w)(?<!non-)bayes", re.IGNORECASE)
"""``Bayes`` or ``Bayesian`` as a word (``naive-Bayes`` too); ``non-Bayesian`` does not match."""


@dataclass(frozen=True)
class SurveyInputs:
    """The survey file the summary reads."""

    survey: Path


def default_inputs() -> SurveyInputs:
    """The survey of Supplementary Table S1."""
    return SurveyInputs(survey=literature.SURVEY_PATH)


def _mode(values: pd.Series, what: str) -> float:
    """The single most common value; raises on a tie."""
    counts = values.value_counts()
    top = counts[counts == counts.max()]
    if len(top) != 1:
        raise ValueError(f"{what}: the mode is tied between {sorted(top.index)}")
    return float(top.index[0])


def _is_excluded(text: str) -> bool:
    return any(form.fullmatch(text) for form in EXCLUDED_FORMS.values())


def numeric(survey: pd.DataFrame) -> pd.DataFrame:
    """The parameter cells as numbers, NaN where the cell is a listed excluded form.

    Parameters
    ----------
    survey : pd.DataFrame, shape (n_papers, n_columns)
        Parameter columns of the survey, as literal text.

    Returns
    -------
    numbers : pd.DataFrame, shape (n_papers, n_columns)
        Floats; NaN where the cell is one of :data:`EXCLUDED_FORMS`.

    Raises
    ------
    ValueError
        If a cell is neither a bare number nor an excluded form, so that a
        malformed or blank cell cannot silently leave the counts.
    """
    stripped = survey.apply(lambda column: column.str.strip())
    bare = stripped.apply(lambda column: column.str.fullmatch(_BARE_NUMBER))
    unknown = [
        (column, row, text)
        for column in stripped.columns
        for row, text in stripped.loc[~bare[column], column].items()
        if not _is_excluded(text)
    ]
    if unknown:
        raise ValueError(
            "survey cells that are neither a bare number nor an excluded form "
            f"(column, row, text): {unknown}"
        )
    return stripped.where(bare).astype(float)


def _quantity(
    key: str, value: float, unit: str, description: str, source: str
) -> Value:
    """A parameter value: printed whole when it is whole, else to two figures."""
    style = "int" if float(value).is_integer() else "sig2"
    return Value(
        key=key,
        value=float(value),
        kind="quantity",
        style=style,
        unit=unit,
        description=description,
        questions=TAGS["survey"].questions,
        used_in=TAGS["survey"].used_in,
        source=source,
    )


def parameter_values(numeric: pd.DataFrame, source: str) -> list[Value]:
    """Mode, minimum and maximum of each parameter the introduction quotes.

    Parameters
    ----------
    numeric : pd.DataFrame, shape (n_papers, n_columns)
        The survey's numbers (:func:`numeric`), with every column of
        :data:`PARAMETER_COLUMNS`.
    source : str
        Where the values come from, for each value's record.

    Returns
    -------
    values : list of Value

    Raises
    ------
    ValueError
        If a parameter has no numeric cell, or its mode is tied.
    """
    out: list[Value] = []
    for stem, (columns, unit, what) in PARAMETERS.items():
        pooled = pd.concat([numeric[c] for c in columns]).dropna()
        if pooled.empty:
            raise ValueError(f"{stem}: no numeric cells in {list(columns)}")
        where = f"over the {len(pooled)} numeric cells of {', '.join(columns)}"
        out += [
            _quantity(
                f"{stem}_mode",
                _mode(pooled, stem),
                unit,
                f"Most common {what}, {where}",
                source,
            ),
            _quantity(
                f"{stem}_min", pooled.min(), unit, f"Smallest {what}, {where}", source
            ),
            _quantity(
                f"{stem}_max", pooled.max(), unit, f"Largest {what}, {where}", source
            ),
        ]
    return out


def triggers(detection: pd.Series) -> pd.Series:
    """Each paper's candidate-event triggers, as a set of signal names.

    Parameters
    ----------
    detection : pd.Series of str, shape (n_papers,)
        The survey's Detection column, e.g. ``"SWR, MUA"``.

    Returns
    -------
    triggers : pd.Series of frozenset of str, shape (n_papers,)

    Raises
    ------
    ValueError
        If a cell names a signal outside :data:`TRIGGERS`, or is blank.
    """
    sets = detection.map(lambda cell: frozenset(t.strip() for t in cell.split(",")))
    unknown = sorted({t for s in sets for t in s} - TRIGGERS)
    if unknown:
        raise ValueError(
            f"Detection names {unknown}, not one of {sorted(TRIGGERS)}: "
            "add the signal to TRIGGERS or correct the survey"
        )
    return sets


def method_values(survey: pd.DataFrame, source: str) -> list[Value]:
    """Counts and shares of the candidate-event triggers and the decoders.

    Parameters
    ----------
    survey : pd.DataFrame, shape (n_papers, 35)
        The survey as literal text (``literature.load_survey``).
    source : str
        Where the values come from, for each value's record.

    Returns
    -------
    values : list of Value
    """
    n = len(survey)
    trigger = triggers(survey["Detection"])
    shares: dict[str, tuple[pd.Series, str]] = {
        "mua_only": (
            trigger == frozenset({"MUA"}),
            "whose candidate events are triggered by multiunit activity alone (Detection names MUA only)",
        ),
        "swr_only": (
            trigger == frozenset({"SWR"}),
            "whose candidate events are triggered by ripple power alone (Detection names SWR only)",
        ),
        "swr_and_mua": (
            trigger == frozenset({"SWR", "MUA"}),
            "that use both ripple power and multiunit activity (Detection names SWR and MUA only)",
        ),
        "bayes": (
            survey["Decoding"].str.contains(_BAYES),
            "whose decoding uses a Bayesian decoder, alone or with another method "
            "(Decoding names 'Bayes' or 'Bayesian'; 'non-Bayesian' does not count)",
        ),
    }
    out = [
        Value(
            key="papers",
            value=n,
            kind="count",
            style="count",
            unit="",
            description="Papers in the survey",
            questions=TAGS["survey"].questions,
            used_in=TAGS["survey"].used_in,
            source=source,
        )
    ]
    for stem, (mask, what) in shares.items():
        out.append(
            Value(
                key=f"{stem}_fraction",
                value=float(mask.sum()) / n,
                kind="fraction",
                style="pctnum",
                unit="",
                description=f"Share of the {n} surveyed papers {what}",
                questions=TAGS["survey"].questions,
                used_in=TAGS["survey"].used_in,
                source=source,
            )
        )
    return out


def summarize(inputs: SurveyInputs, produced_by: str, commit: str) -> Summary:
    """The survey summary: the survey's parameter values and method shares.

    Parameters
    ----------
    inputs : SurveyInputs
        The survey to read.
    produced_by : str
        The command that builds the summary, for its record.
    commit : str
        The commit the summary is built at.

    Returns
    -------
    summary : Summary
        Every survey value, with the survey's path and checksum as its input.
    """
    summary = Summary(name="survey", produced_by=produced_by, commit=commit)
    source = "the replay-literature survey of Supplementary Table S1 (paper/tables/table_s1_literature_parameters.csv)"
    survey = literature.load_survey(inputs.survey)
    summary.add(*parameter_values(numeric(survey[list(PARAMETER_COLUMNS)]), source))
    summary.add(*method_values(survey, source))
    summary.inputs = [file_record(inputs.survey, ROOT_DIR)]
    return summary
