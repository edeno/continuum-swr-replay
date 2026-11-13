"""Sessions pooled within an animal: the one rule of the comparison's animal tables.

An animal pools its sessions by summing counts and seconds, and a proportion is
taken from those sums, never by averaging the sessions' proportions. A session
where the comparison was unavailable is not summed; it is listed instead
(:func:`pool_sessions`).

Every per-animal table of the inventory comparison's report
(``scripts/detector_comparison/report_inventory_comparison.py``) and of the event
features report (``scripts/detector_comparison/report_event_features.py``) is pooled by
:func:`pool_sessions`. Three of them are built here whole, because the events
summary reads them: from the run's tables (:func:`session_tables`), which
``nonlocal_comparison.left_out_statistics`` computes too, each detector's
counts (:func:`counts_by_animal`, the non-local detector's exposure) and the
cells (:func:`cells_by_animal`); and the directional agreement
(:func:`directional_by_animal`), which ``nonlocal_comparison`` checks its
memberships against.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

import numpy as np
import pandas as pd

from continuum_swr_replay.decoding.identifiers import nwb_file_name, session_name
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS_DAY

SESSION = list(SESSION_COLUMNS_DAY)
"""The columns naming a session: ``animal``, ``day`` and ``epoch``."""

MEMBERSHIPS = ("matched", "unmatched", "unknown")
"""An event's membership against another inventory; ``unknown`` is never
counted as ``unmatched``."""

COUNT_SUMS = (
    "n_native",
    "n_eligible",
    "n_moving",
    "n_set_aside",
    "exposure_s",
    "n_in_exposure",
)
"""The runner's count columns an animal sums (:func:`counts_by_animal`)."""


def session_names(frame: pd.DataFrame) -> pd.Series:
    """``{animal}{day}_{epoch}`` of each row (``identifiers.session_name``)."""
    return pd.Series(
        [
            session_name(nwb_file_name(animal, day), epoch)
            for animal, day, epoch in zip(
                frame.animal, frame.day, frame.epoch, strict=True
            )
        ],
        index=frame.index,
        dtype=object,
    )


def ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """``numerator / denominator``, NaN where the denominator is not positive."""
    denominator = denominator.astype(np.float64)
    return numerator.astype(np.float64) / denominator.where(denominator > 0)


def pool_sessions(
    frame: pd.DataFrame,
    *,
    by: Sequence[str],
    sums: Sequence[str],
    available: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Each animal's sums over its available sessions, by ``by``.

    Parameters
    ----------
    frame : DataFrame
        One row per session and ``by`` group, with ``animal``, ``day`` and
        ``epoch``.
    by : sequence of str
        The columns to pool within, besides ``animal``.
    sums : sequence of str
        The count and seconds columns to sum.
    available : boolean array, shape (len(frame),)
        Whether a row's session enters the sums.

    Returns
    -------
    DataFrame
        One row per animal and ``by`` group in ``frame``: ``n_sessions`` (the
        sessions summed), the sums (zero where no session was available), and
        ``unavailable_sessions``, the other sessions' names joined by ``"; "``.
    """
    keys = ["animal", *by]
    work = frame.assign(
        _session=session_names(frame), _available=np.asarray(available, dtype=bool)
    )
    every = work.groupby(keys).size().index
    kept = work[work._available].groupby(keys)
    pooled = pd.concat(
        [kept._session.nunique().rename("n_sessions"), kept[list(sums)].sum()], axis=1
    ).reindex(every)
    listed = (
        work[~work._available]
        .groupby(keys)
        ._session.agg(lambda names: "; ".join(sorted(set(names))))
        .rename("unavailable_sessions")
        .reindex(every)
    )
    pooled = pooled.fillna(0).astype(
        {"n_sessions": np.int64} | {column: frame[column].dtype for column in sums}
    )
    return pd.concat([pooled, listed.fillna("")], axis=1).reset_index()


def summed(
    by_session: pd.DataFrame, by: Sequence[str], sums: Sequence[str]
) -> pd.DataFrame:
    """Sums over every session of each animal, none left out."""
    return pool_sessions(
        by_session, by=by, sums=sums, available=np.ones(len(by_session), dtype=bool)
    ).drop(columns="unavailable_sessions")


def detector_availability(observations: pd.DataFrame) -> pd.DataFrame:
    """Per session and detector, whether it observed any of the epoch.

    Returns
    -------
    DataFrame
        ``animal``, ``day``, ``epoch``, ``detector``, ``available``, and
        ``reason``: the observation table's reasons where nothing was
        observed, else empty.
    """
    keys = [*SESSION, "detector"]
    available = observations.groupby(keys).available.any().rename("available")
    reasons = (
        observations[~observations.available]
        .groupby(keys)
        .reason.agg(lambda reason: "; ".join(sorted(set(reason))))
    )
    out = pd.concat([available, reasons.reindex(available.index)], axis=1)
    out["reason"] = out.reason.where(~out.available, "").fillna("")
    return out.reset_index()


def counts_by_session(counts: pd.DataFrame, availability: pd.DataFrame) -> pd.DataFrame:
    """The runner's counts with each detector's session availability.

    Raises
    ------
    ValueError
        A detector has counts in a session but no observation row there.
    """
    merged = counts.merge(
        availability[[*SESSION, "detector", "available"]],
        on=[*SESSION, "detector"],
        how="left",
        validate="one_to_one",
    )
    unobserved = merged.available.isna()
    if unobserved.any():
        missing = merged[unobserved]
        raise ValueError(
            "No observation row for "
            + ", ".join(
                f"{detector} in {session}"
                for detector, session in zip(
                    missing.detector, session_names(missing), strict=True
                )
            )
            + "; its availability is unknown."
        )
    return merged.astype({"available": np.bool_})


def counts_by_animal(by_session: pd.DataFrame) -> pd.DataFrame:
    """Each detector's counts summed over each animal's available sessions."""
    pooled = pool_sessions(
        by_session, by=["detector"], sums=COUNT_SUMS, available=by_session.available
    )
    notes = by_session.groupby("detector").native_note.first()
    return pooled.assign(
        rate_per_s=ratio(pooled.n_in_exposure, pooled.exposure_s),
        native_note=pooled.detector.map(notes).fillna(""),
    )


def cells_by_session(
    cells: pd.DataFrame,
    availability: pd.DataFrame,
    sources: Mapping[str, Collection[str]],
) -> pd.DataFrame:
    """Each cell's memberships counted per session and source detector.

    One row per session, cell and source detector, zeros where the detector
    found no events. ``n_unknown`` is counted apart from ``n_unmatched``.

    Raises
    ------
    ValueError
        A membership is not one of :data:`MEMBERSHIPS`, or a cell's events
        are of a session or detector the observations do not name.
    """
    strange = set(cells.membership) - set(MEMBERSHIPS)
    if strange:
        raise ValueError(f"Memberships {sorted(strange)} are not {MEMBERSHIPS}.")
    keys = [*SESSION, "cell", "detector"]
    skeleton = availability[[*SESSION, "detector", "available"]].merge(
        pd.DataFrame(
            [(cell, d) for cell, detectors in sources.items() for d in detectors],
            columns=["cell", "detector"],
        ),
        on="detector",
    )
    counted = (
        cells.assign(
            **{f"n_{m}": cells.membership == m for m in MEMBERSHIPS},
        )
        .groupby(keys)
        .agg(
            n_events=("event_key", "size"),
            **{f"n_{m}": (f"n_{m}", "sum") for m in MEMBERSHIPS},
        )
        .reset_index()
    )
    out = skeleton.merge(counted, on=keys, how="outer", indicator=True)
    if (out._merge == "right_only").any():
        raise ValueError("Cell events of a session or detector no observation names.")
    counts = ["n_events", *(f"n_{m}" for m in MEMBERSHIPS)]
    out[counts] = out[counts].fillna(0).astype(np.int64)
    return out.drop(columns="_merge")[[*keys, "available", *counts]].sort_values(
        keys, ignore_index=True
    )


def session_tables(
    manifest: Mapping, tables: Mapping[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """The per-session tables the animal tables pool, from an inventory run.

    Parameters
    ----------
    manifest : mapping
        The inventory run's manifest; its ``settings["cells"]`` name each
        cell's source detectors.
    tables : mapping of str to DataFrame
        The run's ``observations``, ``counts`` and ``cells`` tables over its
        ``ok`` epochs.

    Returns
    -------
    dict of str to DataFrame
        ``availability`` (:func:`detector_availability`), ``counts``
        (:func:`counts_by_session`) and ``cells`` (:func:`cells_by_session`).
    """
    sources = {
        name: cell["sources"] for name, cell in manifest["settings"]["cells"].items()
    }
    availability = detector_availability(tables["observations"])
    return {
        "availability": availability,
        "counts": counts_by_session(tables["counts"], availability),
        "cells": cells_by_session(tables["cells"], availability, sources),
    }


def cells_by_animal(by_session: pd.DataFrame) -> pd.DataFrame:
    """Each cell's counts summed over the sessions its source detector observed."""
    return pool_sessions(
        by_session,
        by=["cell", "detector"],
        sums=["n_events", *(f"n_{m}" for m in MEMBERSHIPS)],
        available=by_session.available,
    )


DIRECTIONAL_SUMS = (
    "shared_exposure_s",
    "n_source_events",
    "n_eligible",
    "n_excluded",
    "n_matched",
)
"""The directional table's columns an animal sums (:func:`directional_by_animal`)."""


def directional_by_animal(
    directional: pd.DataFrame, by: Sequence[str] = ()
) -> pd.DataFrame:
    """Directional agreement pooled per animal and ordered pair.

    ``proportion_matched`` is the summed ``n_matched`` over the summed
    ``n_eligible`` of the sessions whose pair was available.
    """
    pooled = pool_sessions(
        directional,
        by=[*by, "source_detector", "target_detector"],
        sums=DIRECTIONAL_SUMS,
        available=directional.available,
    )
    return pooled.assign(proportion_matched=ratio(pooled.n_matched, pooled.n_eligible))
