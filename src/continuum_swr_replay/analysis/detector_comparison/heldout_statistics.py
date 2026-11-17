"""Summary statistics of the held-out check on simulated data.

The held-out check asks whether cells left out of a decode fire where an
event's decoded content says they should. On simulated data, where the truth of
every event is known, it is run in three folds of discovery and evaluation cells
(``scripts/detector_comparison/sweep_memoryless_heldout.py``,
``heldout_sweep``), and its long-format summary
(``heldout.heldout_summary``) gives how often support falls on events with
spatial content and on events without it. Each function here takes that summary
and returns named numbers, which the detection summary reports
(``continuum_swr_replay.reported.detection``).

A rate or share is a fraction in [0, 1].
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.detection.memoryless_sequence.sweep_statistics import (
    whole_count,
)

__all__ = [
    "heldout_event_classes",
    "heldout_fold_statistics",
    "lowest_estimator_agreement",
    "placeless_event_statistics",
    "pooled_heldout_rate",
]


def pooled_heldout_rate(
    heldout: pd.DataFrame,
    fold: str,
    classes: Sequence[str],
    truth: str,
    measure: str = "support_rate",
    suffix: str = "",
    estimator: str = "state_space",
) -> dict[str, float]:
    """A held-out rate over the state-space detector's events of one fold,
    pooled over event classes.

    Each event is counted once, as the state-space detector bounds it. A rate
    over informative events is weighted by how many are informative, a share
    of all events by how many there are.

    Parameters
    ----------
    heldout : pd.DataFrame, shape (n_rows, n_columns)
        The held-out summary, with ``summary`` (the block), ``measure``,
        ``fold``, ``estimator``, ``detector``, ``event_class``, ``truth`` and
        ``value``.
    fold : str
        The fold of discovery and evaluation cells.
    classes : sequence of str
        The event classes pooled (``"ours_only"``, ``"shared"``).
    truth : str
        What the events truly are (``"continuous_replay"``, ``"non_local"``,
        ``"no_content"``).
    measure : str, optional
        ``"support_rate"``, the share of informative events with held-out
        support, or ``"informative_share"``, the share of all events the
        evaluation cells can judge.
    suffix : str, optional
        The coverage floor of the informative events: ``""`` for the main
        floor, ``"_floor2"`` for 2 expected spikes.
    estimator : str, optional
        The content estimate scored: ``"state_space"`` or ``"sliding_window"``.

    Returns
    -------
    statistics : dict
        - ``rate``: the pooled rate, a share of ``n_events``; NaN when no event
          counts (the fold has no event of these classes, or none informative).
        - ``n_events``: the events the rate rests on, its denominator: the
          informative events for a support rate, all events for an
          informative share.
        - ``n_counted``: of those, the events the rate counts, its numerator:
          the events with held-out support, or the informative ones.

    Raises
    ------
    ValueError
        If the by-class block has no row of the fold, estimator, truth, a
        class or a measure read; two rows for one class and measure; a class
        with no number of events, or with events and no rate; or a class whose
        rate is not a whole number of its events.
    """
    rate_name = f"{measure}{suffix}"
    weight = "n_events_all" if measure == "informative_share" else f"n_events{suffix}"
    by_class = heldout[heldout["summary"] == "by_class"]
    for column, wanted in (
        ("fold", fold),
        ("estimator", estimator),
        ("truth", truth),
        ("measure", rate_name),
        ("measure", weight),
        *(("event_class", event_class) for event_class in classes),
    ):
        if wanted not in set(by_class[column]):
            raise ValueError(
                f"held-out summary: no by-class row with {column} {wanted!r}"
            )
    rows = by_class[
        (by_class["fold"] == fold)
        & (by_class["estimator"] == estimator)
        & (by_class["detector"] == "state_space")
        & by_class["event_class"].isin(classes)
        & (by_class["truth"] == truth)
        & by_class["measure"].isin([rate_name, weight])
    ]
    where = f"held-out summary, fold {fold}, {estimator}, {truth}"
    if rows.duplicated(["event_class", "measure"]).any():
        raise ValueError(f"{where}: several rows for one class and measure")
    rows = rows.pivot(index="event_class", columns="measure", values="value").reindex(
        columns=[rate_name, weight]
    )
    n, rates = rows[weight], rows[rate_name]
    counted = n > 0
    if n.isna().any() or rates[counted].isna().any():
        raise ValueError(
            f"{where}: a class has no {weight}, or has events and no {rate_name}"
        )
    if not counted.any():
        return {"rate": float("nan"), "n_events": 0, "n_counted": 0}
    n, rates = n[counted], rates[counted]
    return {
        "rate": float((rates * n).sum() / n.sum()),
        "n_events": int(n.sum()),
        "n_counted": sum(
            whole_count(rate, events) for rate, events in zip(rates, n, strict=True)
        ),
    }


def heldout_fold_statistics(
    heldout: pd.DataFrame,
    folds: Sequence[str],
    classes: Sequence[str],
    truth: str,
    measure: str = "support_rate",
    suffix: str = "",
    estimator: str = "state_space",
) -> dict[str, float]:
    """A pooled held-out rate over several folds.

    Parameters
    ----------
    heldout : pd.DataFrame, shape (n_rows, n_columns)
        The held-out summary (:func:`pooled_heldout_rate`).
    folds : sequence of str
        The folds.
    classes, truth, measure, suffix, estimator
        As in :func:`pooled_heldout_rate`, which gives each fold's rate and the
        events it rests on.

    Returns
    -------
    statistics : dict
        - ``rate_min``, ``rate_max``: the lowest and highest of the folds'
          rates; NaN when a fold has no rate (no event that counts).
        - ``rate_spread``: the highest minus the lowest; NaN when a fold has no
          rate.
        - ``n_events``: the events the rates rest on, summed over the folds.
        - ``n_counted``: of those, the events the rates count (with held-out
          support, or informative), summed over the folds.
        - ``n_folds_with_support``: the folds whose rate is above zero.
    """
    pooled = [
        pooled_heldout_rate(heldout, fold, classes, truth, measure, suffix, estimator)
        for fold in folds
    ]
    rates = [fold["rate"] for fold in pooled]
    return {
        "rate_min": float(np.min(rates)),
        "rate_max": float(np.max(rates)),
        "rate_spread": float(np.ptp(rates)),
        "n_events": sum(fold["n_events"] for fold in pooled),
        "n_counted": sum(fold["n_counted"] for fold in pooled),
        "n_folds_with_support": sum(fold["rate"] > 0 for fold in pooled),
    }


def placeless_event_statistics(
    heldout: pd.DataFrame, folds: Sequence[str]
) -> dict[str, int]:
    """Place-less events in the held-out check, and how many have held-out support.

    A place-less event is an event whose truth is ``"no_content"``, of either
    class the state-space detector reported (``"ours_only"``, ``"shared"``).
    Support for one is false support: it represents no place.

    Parameters
    ----------
    heldout : pd.DataFrame, shape (n_rows, n_columns)
        The held-out summary (:func:`pooled_heldout_rate`).
    folds : sequence of str
        The folds.

    Returns
    -------
    statistics : dict
        - ``n_events``: the informative reports on place-less events, summed over
          the folds, under the state-space content estimate
          (:func:`heldout_fold_statistics`).
        - ``n_supported``: of those, the events with held-out support.
        - ``n_events_sliding_window``, ``n_supported_sliding_window``: the same
          two counts under the sliding window's content estimate.

    Raises
    ------
    ValueError
        If a row read is missing (:func:`pooled_heldout_rate`).
    """
    state_space, sliding_window = (
        heldout_fold_statistics(
            heldout, folds, ("ours_only", "shared"), "no_content", estimator=estimator
        )
        for estimator in ("state_space", "sliding_window")
    )
    return {
        "n_events": state_space["n_events"],
        "n_supported": state_space["n_counted"],
        "n_events_sliding_window": sliding_window["n_events"],
        "n_supported_sliding_window": sliding_window["n_counted"],
    }


def lowest_estimator_agreement(heldout: pd.DataFrame) -> float:
    """The lowest agreement of the two content estimates, over folds.

    Parameters
    ----------
    heldout : pd.DataFrame, shape (n_rows, n_columns)
        The held-out summary (:func:`pooled_heldout_rate`). Its ``agreement``
        block holds, per fold, the share of the events informative under both
        content estimates on which the two verdicts agree.

    Returns
    -------
    agreement : float
        The lowest fold's share.

    Raises
    ------
    ValueError
        If the summary holds no fold's agreement, or a fold's is missing.
    """
    agreement = heldout[
        (heldout["summary"] == "agreement")
        & (heldout["measure"] == "estimator_agreement")
    ]["value"]
    if agreement.empty or agreement.isna().any():
        raise ValueError(
            "held-out summary: no estimator agreement, or a fold without one"
        )
    return float(agreement.min())


def heldout_event_classes(heldout: pd.DataFrame) -> set[str]:
    """The classes of event the held-out summary holds: none that only the
    sliding window reported, or the summary is refused.

    Parameters
    ----------
    heldout : pd.DataFrame, shape (n_rows, n_columns)
        The held-out summary (:func:`pooled_heldout_rate`).

    Returns
    -------
    classes : set of str
        The ``event_class`` of its by-class rows: ``"shared"`` (both detectors
        reported the event) or ``"ours_only"``.

    Raises
    ------
    ValueError
        If a by-class row's class is ``"comparator_only"``: an event the
        sliding window reported and the state-space detector did not.
    """
    by_class = heldout[heldout["summary"] == "by_class"]
    classes = set(by_class["event_class"])
    if "comparator_only" in classes:
        raise ValueError(
            "held-out summary: expected no event only the sliding window reported; "
            "it holds comparator-only events"
        )
    return classes
