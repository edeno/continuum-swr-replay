"""Duration-matched non-event control windows for the engagement test.

Draws one control window per non-local event from the same fitted
immobility bout, clear of every real and artifact-dropped event, with a
per-session seed.
"""

import hashlib
import logging
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.event_definition.eligibility import (
    EXCLUDED_TIME_REASONS,
    DropReason,
)
from continuum_swr_replay.analysis.pfc.event_sets import EVENT_CLASSES
from continuum_swr_replay.intervals import (
    contiguous_true_runs,
    merge_intervals,
)

logger = logging.getLogger(__name__)


#: The class matched controls are drawn one-for-one against.
MATCHED_CONTROL_SOURCE_CLASS = "non_local"


#: Placement semantics written to every bundle.  The duration-matching window
#: for each control must avoid real events, but two controls may have
#: overlapping windows because the fitted regressor contains only their
#: distinct onsets.  Treating those auxiliary windows as occupied time makes
#: an exact count-matched null impossible in dense bouts even when hundreds of
#: distinct event-free onset bins exist.
MATCHED_CONTROL_PLACEMENT_METHOD = (
    "same_bout_real_event_free_distinct_onsets_windows_may_overlap"
)


#: Whole-draw placement attempts before the best incomplete matching is kept.
#: Each attempt visits source events in a new random order and samples directly
#: from their legal onset bins; it is not rejection sampling over illegal bins.
DEFAULT_MATCHED_CONTROL_ATTEMPTS = 50


def matched_control_seed(nwb_file_name: str, epoch_name: str) -> int:
    """Per-session seed for the matched-control draw.

    Derived from the session key rather than taken from a counter so a session
    refit on its own -- to repair one bundle, say -- redraws exactly the control
    population the batch drew, and two sessions never share a draw.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name, e.g. ``"wilbur20210329_.nwb"``.
    epoch_name : str
        Epoch name, e.g. ``"02_r1"``.

    Returns
    -------
    int
        The low 64 bits of ``sha256(f"{nwb_file_name}|{epoch_name}")``.
    """
    digest = hashlib.sha256(f"{nwb_file_name}|{epoch_name}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _bout_containing(runs: NDArray, index: int) -> int:
    """Row of ``runs`` whose half-open bounds contain ``index``, or ``-1``.

    Parameters
    ----------
    runs : NDArray of int, shape (n_runs, 2)
        Ascending, non-overlapping half-open bounds, as
        :func:`contiguous_true_runs` returns them.
    index : int
        Sample index to locate.

    Returns
    -------
    int
        Row index into ``runs``, or ``-1`` when ``index`` is in no run.
    """
    if len(runs) == 0 or index < 0:
        return -1
    row = int(np.searchsorted(runs[:, 0], index, side="right")) - 1
    if row < 0 or index >= int(runs[row, 1]):
        return -1
    return row


#: The decode-frame cohorts whose dropped events the PFC path counts and keeps
#: its controls clear of: the two it models (non-local events and SWRs).
ARTIFACT_DROP_COLUMNS = ("non_local_event", "SWR_event")


def artifact_dropped_events(drops: pd.DataFrame) -> pd.DataFrame:
    """The non-local events and SWRs the analysed-time rule dropped.

    Parameters
    ----------
    drops : pandas.DataFrame
        ``analysed_time.artifact_event_drops`` of the session's decode frame.

    Returns
    -------
    pandas.DataFrame
        ``start_time`` and ``end_time`` (seconds) of every non-local event or
        SWR dropped for excluded time (``artifact``, ``near_artifact``) or for
        overlapping such an event (``overlaps_dropped_event``), sorted by start.
    """
    reasons = {*EXCLUDED_TIME_REASONS, DropReason.OVERLAPS_DROPPED_EVENT.value}
    selected = drops["event_col"].isin(ARTIFACT_DROP_COLUMNS) & drops[
        "drop_reason"
    ].isin(reasons)
    return (
        drops.loc[selected, ["start_time", "end_time"]]
        .sort_values("start_time")
        .reset_index(drop=True)
    )


class MatchedControlDraw(NamedTuple):
    """One draw of the matched non-event control, and why events went unplaced.

    The two shortfall counts are kept apart because they mean different things
    about a session. An onset outside every fitted bout is a *cohort* fact --
    the event is kept by its onset bin, which can sit outside the fit mask --
    and says nothing about how crowded the session is. A saturated bout is a
    *density* fact: that bout is too short or too full of real events to offer
    a legal onset for this event's duration, or the finite whole-draw matching
    could not give every source a distinct onset. Control duration windows may
    overlap one another; only their onsets enter the fitted design, and those
    onsets are distinct.

    Attributes
    ----------
    times : pandas.DataFrame
        Columns ``start_time``, ``end_time`` in seconds, one row per placed
        control.
    source_indices : NDArray of int, shape (n_placed,)
        Positional indices of the source events represented by ``times``, in
        the same order. This makes a placement shortfall auditable and lets a
        caller restrict both the observed statistic and subsequent null draws
        to one fixed, jointly matchable event subset.
    n_onset_outside_fitted_bout : int
        Source events whose onset bin belongs to no fitted immobility bout.
    n_bout_saturated : int
        Source events whose bout could not supply a distinct event-free onset
        for a window of their duration after ``max_attempts`` whole-draw
        matchings.
    """

    times: pd.DataFrame
    source_indices: NDArray
    n_onset_outside_fitted_bout: int
    n_bout_saturated: int

    @property
    def n_unplaced(self) -> int:
        """Source events with no control window, for either reason."""
        return self.n_onset_outside_fitted_bout + self.n_bout_saturated


def make_matched_control_times(
    event_times: Mapping[str, pd.DataFrame],
    is_immobile: pd.Series,
    rng: np.random.Generator,
    max_attempts: int = DEFAULT_MATCHED_CONTROL_ATTEMPTS,
    avoid: pd.DataFrame | None = None,
) -> MatchedControlDraw:
    """Time-shifted-onset matched non-event control.

    The non-SWR replay regressor asks whether a neuron responds to replay. On
    its own it cannot say whether the response is to replay or to *being in the
    stretch of immobility where replay happens* -- events are not uniform over
    a stop, and neither is firing. This draws, for every real non-local event, a
    window of the same duration inside the same fitted immobility bout that
    holds no real event at all. Entered as its own event class, it turns the
    replay question into a nested one: does replay add beyond a duration-matched
    non-event baseline drawn from the same bins.

    For each source event the control onset is drawn uniformly from the legal
    sample times of its own bout, where a legal onset has these properties:

    - the control window lies entirely inside that bout (so every bin it covers
      is a bin the models are fit on),
    - it overlaps no window of any real event class, ``SWR`` included, nor any
      event in `avoid` (the events the analysed-time rule dropped), and
    - no other control in the draw uses that exact onset bin.

    The duration-matching windows may overlap one another. Duration constrains
    where an onset may be drawn, but the GLM design is onset-only: it does not
    contain an event-in-progress box. Requiring the auxiliary windows to be
    mutually disjoint can therefore make a count-matched null impossible in a
    dense bout even when many distinct, individually event-free onsets exist.
    Directly enumerating legal onsets also avoids stochastic shortfalls from
    missing a narrow valid interval during rejection sampling.

    An event is dropped -- never placed somewhere else -- when its onset falls
    in no fitted bout at all (the cohort keeps an event by onset, and an onset
    bin can still be outside the fit mask) or when no distinct legal onset can
    be assigned in ``max_attempts`` whole-draw matchings. The returned count is
    therefore ``<=`` the number of source events, and the caller must record
    the shortfall rather than assume a one-to-one match.

    Parameters
    ----------
    event_times : Mapping[str, pandas.DataFrame]
        The event sets :func:`build_session_event_sets` returns, keyed
        ``non_local_no_SWR``, ``non_local_SWR``, ``non_local``, ``SWR``, each
        with ``start_time`` / ``end_time`` in seconds. ``non_local`` supplies
        the source events; every class supplies windows a control may not
        overlap.
    is_immobile : pandas.Series of bool, shape (n_time,)
        The bins a control may be drawn from, indexed by the session's sample
        times in seconds. Callers pass the *fit mask*
        (:func:`build_session_fit_mask`'s second return), not raw immobility:
        the control has to sample the same bins the models are fit on. A "bout"
        is a contiguous True run of this mask.
    rng : numpy.random.Generator
        Seeded generator; see :func:`matched_control_seed`.
    max_attempts : int
        Random source-order matchings attempted before the best incomplete
        matching is kept. Zero deliberately places nothing (used by the
        incomplete-null regression test).
    avoid : pandas.DataFrame, optional
        Further windows a control may not overlap (``start_time`` /
        ``end_time``, seconds): the events :func:`artifact_dropped_events`
        returns.

    Returns
    -------
    MatchedControlDraw
        ``times`` holds columns ``start_time``, ``end_time`` in seconds, one
        row per *placed* control, in source-event order. Rows are deliberately
        not re-sorted by time; ``source_indices[k]`` identifies the source
        event matched by row ``k``. ``n_onset_outside_fitted_bout`` and
        ``n_bout_saturated`` split the shortfall by its two causes.

    Raises
    ------
    KeyError
        If ``event_times`` lacks one of the four event classes.
    """
    times = np.asarray(is_immobile.index, dtype=float)
    runs = contiguous_true_runs(np.asarray(is_immobile, dtype=bool))

    source = event_times[MATCHED_CONTROL_SOURCE_CLASS]
    windows = [event_times[name] for name in EVENT_CLASSES]
    if avoid is not None:
        windows.append(avoid)
    real_starts, real_ends = merge_intervals(
        np.concatenate([w.start_time.to_numpy(dtype=float) for w in windows]),
        np.concatenate([w.end_time.to_numpy(dtype=float) for w in windows]),
    ).T

    if max_attempts < 0:
        raise ValueError(f"max_attempts must be non-negative, got {max_attempts}")

    source_starts = source.start_time.to_numpy(dtype=float)
    source_ends = source.end_time.to_numpy(dtype=float)
    candidate_indices: list[NDArray] = []
    onset_outside = np.zeros(len(source), dtype=bool)
    for source_index, (start, end) in enumerate(
        zip(source_starts, source_ends, strict=True)
    ):
        duration = end - start
        # The bin holding the onset, and the bout that bin belongs to.
        onset_index = int(np.searchsorted(times, start, side="right")) - 1
        run_row = _bout_containing(runs, onset_index)
        if run_row < 0:
            onset_outside[source_index] = True
            candidate_indices.append(np.empty(0, dtype=np.int64))
            continue
        run_start, run_stop = int(runs[run_row, 0]), int(runs[run_row, 1])
        # Onsets late enough in the bout that the window would run past its
        # last sample are not candidates.
        last_offset = (
            int(
                np.searchsorted(
                    times[run_start:run_stop],
                    times[run_stop - 1] - duration,
                    side="right",
                )
            )
            - 1
        )
        if last_offset < 0:
            candidate_indices.append(np.empty(0, dtype=np.int64))
            continue

        indices = np.arange(run_start, run_start + last_offset + 1, dtype=np.int64)
        starts = times[indices]
        ends = starts + duration
        # For each candidate, the only merged real window that can overlap is
        # the last one starting before candidate_end.  This vectorized search
        # enumerates all legal onset bins without a candidate x event matrix.
        n_starting_before = np.searchsorted(real_starts, ends, side="left")
        overlaps = np.zeros(len(indices), dtype=bool)
        has_predecessor = n_starting_before > 0
        overlaps[has_predecessor] = (
            real_ends[n_starting_before[has_predecessor] - 1] > starts[has_predecessor]
        )
        candidate_indices.append(indices[~overlaps])

    # This is a randomized matching between source events and legal onset
    # bins.  Windows are allowed to overlap, but onset bins are not: duplicate
    # onsets would be summed by ``event_onset_counts`` and would not represent
    # the requested number of events.  Retrying source orders avoids a greedy
    # early choice starving a later event whose legal set is narrower.
    target = int(sum(len(candidates) > 0 for candidates in candidate_indices))
    best: dict[int, int] = {}
    for _ in range(max_attempts):
        used = np.zeros(len(times), dtype=bool)
        placed: dict[int, int] = {}
        for source_index in rng.permutation(len(source)):
            candidates = candidate_indices[int(source_index)]
            available = candidates[~used[candidates]]
            if not len(available):
                continue
            chosen = int(available[int(rng.integers(len(available)))])
            placed[int(source_index)] = chosen
            used[chosen] = True
        if len(placed) > len(best):
            best = placed
        if len(best) == target:
            break

    placed_source_indices = sorted(best)
    control_starts = [float(times[best[index]]) for index in placed_source_indices]
    control_ends = [
        control_start + (source_ends[index] - source_starts[index])
        for index, control_start in zip(
            placed_source_indices, control_starts, strict=True
        )
    ]
    n_onset_outside_fitted_bout = int(onset_outside.sum())
    n_bout_saturated = (
        len(source) - n_onset_outside_fitted_bout - len(placed_source_indices)
    )

    if n_onset_outside_fitted_bout or n_bout_saturated:
        logger.warning(
            "%d of %d matched controls could not be placed in %d attempts: "
            "%d events have an onset in no fitted bout, %d sit in a bout with "
            "no free window of the event's duration",
            n_onset_outside_fitted_bout + n_bout_saturated,
            len(source),
            max_attempts,
            n_onset_outside_fitted_bout,
            n_bout_saturated,
        )
    return MatchedControlDraw(
        times=pd.DataFrame(
            {"start_time": control_starts, "end_time": control_ends}, dtype=float
        ),
        source_indices=np.asarray(placed_source_indices, dtype=np.int64),
        n_onset_outside_fitted_bout=n_onset_outside_fitted_bout,
        n_bout_saturated=n_bout_saturated,
    )
