"""Statistics of the published replay studies surveyed in Supplementary Table S1.

The survey is a table with one row per paper. Its numeric parameter cells are
read into numbers, and its Detection column into sets of trigger signals, by
the survey summary's builder (``continuum_swr_replay.reported.survey``), which
knows the forms a cell takes. The functions here compute what the introduction
quotes from those: the most common, smallest and largest value of a parameter
over the papers that report it as a number, and the share of papers that use a
trigger or a decoder.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import pandas as pd

BAYES = re.compile(r"(?<!\w)(?<!non-)bayes", re.IGNORECASE)
"""``Bayes`` or ``Bayesian`` as a word (``naive-Bayes`` too); ``non-Bayesian`` does not match."""


def mode(values: pd.Series, what: str) -> float:
    """The single most common value of ``values``.

    Parameters
    ----------
    values : pd.Series of float, shape (n_values,)
    what : str
        What the values are, for the error.

    Returns
    -------
    mode : float

    Raises
    ------
    ValueError
        If two or more values are equally most common: a tie is an error
        rather than a silent choice.
    """
    counts = values.value_counts()
    top = counts[counts == counts.max()]
    if len(top) != 1:
        raise ValueError(f"{what}: the mode is tied between {sorted(top.index)}")
    return float(top.index[0])


def parameter_statistics(
    numbers: pd.DataFrame, columns: Sequence[str], what: str
) -> dict[str, float | int]:
    """The mode, minimum and maximum of one parameter, over its numeric cells.

    Parameters
    ----------
    numbers : pd.DataFrame, shape (n_papers, n_columns)
        The survey's parameter cells as numbers, NaN where a cell is not a
        bare number.
    columns : sequence of str
        The columns pooled for the parameter: a paper with a number in two of
        them contributes both.
    what : str
        The parameter's name, for the errors.

    Returns
    -------
    statistics : dict
        ``n_cells`` (the numeric cells pooled), ``mode`` (:func:`mode`),
        ``min`` and ``max``.

    Raises
    ------
    ValueError
        If no cell of ``columns`` is a number, or the mode is tied.
    """
    pooled = pd.concat([numbers[c] for c in columns]).dropna()
    if pooled.empty:
        raise ValueError(f"{what}: no numeric cells in {list(columns)}")
    return {
        "n_cells": len(pooled),
        "mode": mode(pooled, what),
        "min": pooled.min(),
        "max": pooled.max(),
    }


def method_shares(triggers: pd.Series, decoding: pd.Series) -> dict[str, float | int]:
    """The share of papers that trigger candidate events on each signal, and that
    decode with a Bayesian decoder.

    Parameters
    ----------
    triggers : pd.Series of frozenset of str, shape (n_papers,)
        Each paper's candidate-event triggers (``"SWR"``, ``"MUA"``,
        ``"Decoding"``).
    decoding : pd.Series of str, shape (n_papers,)
        Each paper's Decoding cell.

    Returns
    -------
    shares : dict
        ``n_papers``; ``mua_only_fraction``, ``swr_only_fraction`` and
        ``swr_and_mua_fraction``, the papers whose triggers are exactly
        ``{"MUA"}``, ``{"SWR"}`` and ``{"SWR", "MUA"}`` (a paper that also
        names ``"Decoding"`` is in none of them); ``bayes_fraction``, the papers
        whose Decoding cell names a Bayesian decoder, alone or with another
        method (:data:`BAYES`).
    """
    n = len(triggers)
    masks = {
        "mua_only_fraction": triggers == frozenset({"MUA"}),
        "swr_only_fraction": triggers == frozenset({"SWR"}),
        "swr_and_mua_fraction": triggers == frozenset({"SWR", "MUA"}),
        "bayes_fraction": decoding.str.contains(BAYES),
    }
    return {"n_papers": n, **{name: float(m.sum()) / n for name, m in masks.items()}}
