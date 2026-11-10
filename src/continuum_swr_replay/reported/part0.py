"""Part 0's summary: the surveyed literature the introduction describes.

The introduction summarizes how published replay studies detect candidate
events and decode them, from the 57-paper survey in Supplementary Table S1
(``continuum_swr_replay.reported.literature``). The values here are computed
from that survey, so a correction to it reaches the text.

A numeric parameter keeps a cell only when it is a bare number, the rule
``ripple_detection.load_literature_parameters`` uses: a range (``1-3``), a bound
(``>4``), a share (``15% of ensemble``), ``Adaptive``, ``Not reported`` and
``#N/A`` are left out. Each mode, minimum and maximum is over the papers whose
cell is a number. A tie for the mode is an error rather than a silent choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported import literature
from continuum_swr_replay.reported.values import Summary, Value, file_record

CLAIMS = ("introduction",)
USED_IN = ("paper",)


@dataclass(frozen=True)
class Part0Inputs:
    """The survey file the summary reads."""

    survey: Path


def default_inputs() -> Part0Inputs:
    """The survey the installed ``ripple_detection`` packages."""
    return Part0Inputs(survey=literature.source_path())


def _mode(values: pd.Series, what: str) -> float:
    """The single most common value; raises on a tie."""
    counts = values.value_counts()
    top = counts[counts == counts.max()]
    if len(top) != 1:
        raise ValueError(f"{what}: the mode is tied between {sorted(top.index)}")
    return float(top.index[0])


def numeric(survey: pd.DataFrame) -> pd.DataFrame:
    """The survey with every cell that is not a bare number set to NaN.

    Parameters
    ----------
    survey : pd.DataFrame, shape (n_papers, 35)
        The survey as literal text.

    Returns
    -------
    numbers : pd.DataFrame, shape (n_papers, 35)
        Floats; NaN where the cell is a range, bound, phrase or missing marker.
    """
    return survey.apply(
        lambda column: pd.to_numeric(column.str.strip(), errors="coerce")
    )


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
        claims=CLAIMS,
        used_in=USED_IN,
        source=source,
    )


def parameter_values(numeric: pd.DataFrame, source: str) -> list[Value]:
    """Mode, minimum and maximum of each parameter the introduction quotes.

    Parameters
    ----------
    numeric : pd.DataFrame, shape (n_papers, 35)
        The survey's numbers (:func:`numeric`).
    source : str
        Where the values come from, for each value's record.

    Returns
    -------
    values : list of Value
    """
    # key stem -> (columns pooled, unit, what it is)
    parameters: dict[str, tuple[tuple[str, ...], str, str]] = {
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
    out: list[Value] = []
    for stem, (columns, unit, what) in parameters.items():
        pooled = pd.concat([numeric[c] for c in columns]).dropna()
        n = len(pooled)
        where = f"over the {n} numeric cells of {', '.join(columns)}"
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
    trigger = survey["Detection"].str.strip()
    decoding = survey["Decoding"]
    shares: dict[str, tuple[pd.Series, str]] = {
        "mua_only": (
            trigger == "MUA",
            "whose candidate events are triggered by multiunit activity alone (Detection is exactly 'MUA')",
        ),
        "swr_only": (
            trigger == "SWR",
            "whose candidate events are triggered by ripple power alone (Detection is exactly 'SWR')",
        ),
        "swr_and_mua": (
            trigger == "SWR, MUA",
            "that use both ripple power and multiunit activity (Detection is 'SWR, MUA')",
        ),
        "bayes": (
            decoding.str.contains("Bayes", case=False),
            "whose decoding uses a Bayesian decoder, alone or with another method (Decoding names 'Bayes')",
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
            claims=CLAIMS,
            used_in=USED_IN,
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
                claims=CLAIMS,
                used_in=USED_IN,
                source=source,
            )
        )
    return out


def summarize(inputs: Part0Inputs, produced_by: str, commit: str) -> Summary:
    """Part 0's summary from ``inputs``."""
    summary = Summary(part=0, produced_by=produced_by, commit=commit)
    source = (
        f"the replay-literature survey packaged in {literature.SOURCE_PACKAGE} "
        f"{literature.source_version()} (Supplementary Table S1)"
    )
    survey = pd.read_csv(
        inputs.survey, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    summary.add(*parameter_values(numeric(survey), source))
    summary.add(*method_values(survey, source))
    summary.inputs = [file_record(inputs.survey, ROOT_DIR)]
    return summary
