"""The populations of compared events, from the comparison's stored tables.

The comparison runner (``scripts/detection/run_inventory_comparison.py``) stores each
epoch's observations, set-aside events, inventory, pairs, directional
agreement and cells, but not the per-event memberships. :func:`rebuild_matches`
recomputes them (``event_matches``) from the stored tables, and refuses an
epoch whose rebuilt memberships do not reproduce the stored ``directional``
and ``cells`` tables exactly.

The populations, by label (:func:`population_rows`):

- ``all``: every compared event of the detector (speed-eligible, not set
  aside).
- ``matched_by:<B>``, ``unmatched_by:<B>``: the event's pairwise membership
  against detector B, as ``event_matches`` defines it. ``matched_by:<B>`` are
  the events shared with B.
- ``unknown_by:<B>``: neither, counted apart and never as unmatched, with the
  reason in ``unknown_reason``: ``no_shared_exposure`` when the pair shares
  no exposure in the epoch (B unavailable there, or the two exposures
  disjoint; the ``directional`` row reads unavailable), or
  ``outside_shared_exposure`` when the event does not lie wholly inside the
  pair's shared exposure.
- ``exactly_one_detector``, ``swr_matched_by_no_decoder``,
  ``sequence_matched_by_no_trigger`` (or any cell's name): the events of the
  stored cell with membership ``unmatched``. ``<cell>:matched`` is the
  reference beside it, and ``<cell>:unknown`` is counted apart.
- ``baseline`` (:data:`BASELINE`): the matched baseline windows. Only runs
  written before the baselines were dropped (2026-10-18) hold such rows, as
  detector ``baseline`` in the features table; the report drops them, and
  :data:`BASELINE` names them for that. They are not compared events and
  never enter a detector's populations; :func:`population_rows` makes none.

Each method's unmatched events are ``unmatched_by:<B>`` and the exactly-one
cell; sequences matched by no SWR or MUA event are
``sequence_matched_by_no_trigger`` (Carey counts as an SWR definition).
"""

from collections.abc import Collection, Mapping, Sequence
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.inventory_comparison.availability import (
    stored_coverage,
)
from continuum_swr_replay.analysis.inventory_comparison.cells import (
    membership_against,
)
from continuum_swr_replay.analysis.inventory_comparison.overlap import (
    PAIR_DTYPES,
    UNKNOWN,
    UNMATCHED,
    Coverage,
    directional_agreement,
    event_matches,
    flag_pairs,
)
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS_DAY

__all__ = [
    "ALL",
    "BASELINE",
    "NO_SHARED_EXPOSURE",
    "OUTSIDE_SHARED_EXPOSURE",
    "POPULATION_DTYPES",
    "event_populations",
    "population_rows",
    "rebuild_matches",
    "without_baseline_rows",
]

ALL = "all"
"""Every compared event of a detector."""
BASELINE = "baseline"
"""The matched baseline windows' population and detector name."""
NO_SHARED_EXPOSURE = "no_shared_exposure"
"""Unknown against B: the pair shares no exposure in the epoch."""
OUTSIDE_SHARED_EXPOSURE = "outside_shared_exposure"
"""Unknown against B: the event is not wholly inside the pair's shared
exposure."""

POPULATION_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "detector": object,
        "event_key": object,
        "population": object,
        "unknown_reason": object,
    }
)
"""One row per event and population it belongs to. ``unknown_reason`` is
empty except on ``unknown_by:<B>`` rows."""


def _require_same(
    name: str, rebuilt: pd.DataFrame, stored: pd.DataFrame, keys: list[str]
) -> None:
    """Refuse a rebuilt table that is not the stored one, row order aside."""
    rebuilt = rebuilt[list(stored.columns)].sort_values(keys, ignore_index=True)
    stored = stored.sort_values(keys, ignore_index=True)
    if rebuilt.shape == stored.shape and rebuilt.equals(stored):
        return
    if rebuilt.shape == stored.shape:
        try:
            differs = (rebuilt != stored) & ~(rebuilt.isna() & stored.isna())
        except TypeError:  # values of dtypes that cannot be compared
            detail = "the columns differ in dtype"
        else:
            detail = (
                f"{int(differs.any(axis=1).sum())} of {len(stored)} rows differ, "
                f"in {list(stored.columns[differs.any(axis=0).to_numpy()])} (or "
                "in dtype)"
            )
    else:
        detail = f"{len(rebuilt)} rows rebuilt against {len(stored)} stored"
    raise ValueError(
        f"The memberships rebuilt from the stored tables do not reproduce the "
        f"stored {name} table: {detail}. The coverage, roster or cells differ "
        "from the comparison's."
    )


def rebuild_matches(
    tables: Mapping[str, pd.DataFrame],
    still: Mapping[tuple, NDArray],
    *,
    detectors: Collection[str],
    cells: Mapping[str, tuple[Sequence[str], Sequence[str]]],
) -> tuple[dict[tuple, Coverage], pd.DataFrame]:
    """An epoch's coverage and memberships, rebuilt from its stored tables.

    The coverage is ``stored_coverage`` of the stored observations and
    set-aside events; the memberships are ``event_matches`` over the stored
    compared events (the ``inventory`` rows with
    ``whole_event_speed_eligible``) and pairs. Both are checked: the
    ``flag_pairs``, ``directional_agreement`` and ``membership_against`` they
    give must equal the stored ``pairs``, ``directional`` and ``cells`` tables
    exactly, rows in any order.

    Parameters
    ----------
    tables : mapping
        The epoch's stored tables by name: ``observations``, ``set_aside``,
        ``inventory``, ``pairs``, ``directional`` and ``cells``.
    still : mapping
        ``(animal, day, epoch) -> ndarray (n, 2)``, the still intervals the
        comparison read, as for ``stored_coverage``.
    detectors : collection of str
        The comparison's roster, in the comparison's order.
    cells : mapping
        Each cell's name -> (its source detectors, the detectors they are
        tested against), as the comparison defined them.

    Returns
    -------
    coverage : dict
        ``(animal, day, epoch, detector) -> Coverage``.
    matches : DataFrame
        Columns and dtypes as ``overlap.MATCH_DTYPES``, one row per compared
        event and other detector of the roster.

    Raises
    ------
    ValueError
        If the rebuilt pair flags, directional agreement or cells differ from
        the stored ones, or (from ``event_matches``) a compared event lies
        outside its own detector's rebuilt coverage.

    """
    coverage = stored_coverage(tables["observations"], tables["set_aside"], still)
    inventory = tables["inventory"]
    compared = inventory[inventory.whole_event_speed_eligible].reset_index(drop=True)
    pairs = tables["pairs"][list(PAIR_DTYPES)]
    matches = event_matches(compared, pairs, coverage, detectors=detectors)
    # each pair's within-exposure flags, so the memberships are checked event
    # by event and not only through the tables counted from them
    _require_same(
        "pairs",
        flag_pairs(pairs, matches),
        tables["pairs"],
        [*SESSION_COLUMNS_DAY, "event_key_a", "event_key_b"],
    )
    _require_same(
        "directional",
        directional_agreement(compared, pairs, coverage, detectors=detectors),
        tables["directional"],
        [*SESSION_COLUMNS_DAY, "source_detector", "target_detector"],
    )
    _require_same(
        "cells",
        pd.concat(
            [
                membership_against(
                    compared, matches, sources=sources, against=against
                ).assign(cell=name)
                for name, (sources, against) in cells.items()
            ],
            ignore_index=True,
        ),
        tables["cells"],
        ["cell", "event_key"],
    )
    return coverage, matches


def population_rows(
    events: pd.DataFrame,
    matches: pd.DataFrame,
    cells: pd.DataFrame,
    directional: pd.DataFrame,
) -> pd.DataFrame:
    """Every population each compared event belongs to, by the module's labels.

    Parameters
    ----------
    events : DataFrame
        The compared events, in the common schema.
    matches : DataFrame
        ``event_matches`` over them, as ``rebuild_matches`` returns it.
    cells : DataFrame
        The stored ``cells`` table: ``cell`` and ``cells.CELL_DTYPES``.
    directional : DataFrame
        The stored ``directional`` table, whose ``available`` says whether
        each ordered pair shares exposure.

    Returns
    -------
    rows : DataFrame
        Columns and dtypes as ``POPULATION_DTYPES``: per event one ``all``
        row, one pairwise row per other detector, and one row per cell it is
        a source of.

    Raises
    ------
    ValueError
        If the matches are not over these events, a cell holds an event not
        among them, or a membership has no directional row for its pair.

    """
    if set(events.event_key) != set(matches.event_key):
        raise ValueError(
            "The matches are not over these events; pass the compared events "
            "the matches were computed over."
        )
    if not cells.event_key.isin(events.event_key).all():
        raise ValueError(
            "A row of the cells table is not one of these events; pass the "
            "cells of the compared events."
        )
    ids = [*SESSION_COLUMNS_DAY, "detector", "event_key"]
    everyone = events[ids].assign(population=ALL, unknown_reason="")

    pair = [*SESSION_COLUMNS_DAY, "detector", "other_detector"]
    shared = directional[
        [*SESSION_COLUMNS_DAY, "source_detector", "target_detector", "available"]
    ].rename(
        columns={"source_detector": "detector", "target_detector": "other_detector"}
    )
    pairwise = matches.merge(shared, on=pair, how="left", validate="many_to_one")
    if pairwise.available.isna().any():
        raise ValueError("A membership's pair has no row in the directional table.")
    unknown = (pairwise.membership == UNKNOWN).to_numpy()
    available = pairwise.available.astype(bool).to_numpy()
    pairwise = pairwise[ids].assign(
        population=(pairwise.membership + "_by:" + pairwise.other_detector).to_numpy(),
        unknown_reason=np.where(
            unknown,
            np.where(available, OUTSIDE_SHARED_EXPOSURE, NO_SHARED_EXPOSURE),
            "",
        ),
    )

    unmatched = (cells.membership == UNMATCHED).to_numpy()
    in_cells = cells[ids].assign(
        population=np.where(unmatched, cells.cell, cells.cell + ":" + cells.membership),
        unknown_reason="",
    )
    rows = pd.concat([everyone, pairwise, in_cells], ignore_index=True)
    return rows[list(POPULATION_DTYPES)].astype(dict(POPULATION_DTYPES))


def without_baseline_rows(features: pd.DataFrame) -> pd.DataFrame:
    """``features`` less the baseline windows (detector ``baseline``) that a
    run written before the baselines were dropped holds."""
    kept: pd.DataFrame = features[(features.detector != BASELINE).to_numpy()]
    return kept.reset_index(drop=True)


def event_populations(
    features: pd.DataFrame,
    pair_membership: pd.DataFrame,
    cells: pd.DataFrame,
    directional: pd.DataFrame,
) -> pd.DataFrame:
    """Each compared event once per population it belongs to.

    Returns
    -------
    DataFrame
        Columns and dtypes as :data:`POPULATION_DTYPES`.
    """
    rows = population_rows(features, pair_membership, cells, directional)
    return rows[list(POPULATION_DTYPES)].astype(dict(POPULATION_DTYPES))
