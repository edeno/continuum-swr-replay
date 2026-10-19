"""The (stop x bin) point-process design and the bases it is built on.

Everything the per-bin event-rate models consume is assembled here:
:func:`aggregate_to_timebins` expands the stop table into one row per
(stop, bin) with its exposure, response count, per-bin clocks and history
columns, and the raised-cosine bases give the reward- and self-history blocks
a smooth low-dimensional parameterization (plus the inverse, used to read a
kernel back out of fitted coefficients). :func:`attach_is_swr_to_onsets` and
``POPULATIONS`` prepare the onset stream that supplies the response.

Its imports are the design constants, the SWR source (for each session's
excluded time) and the artifact mask's interval rule. It must not import
:mod:`..covariates`, which imports the bases from here.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.hippocampus.artifacts.masks import (
    overlaps_intervals,
)
from continuum_swr_replay.analysis.hippocampus.swr_source import (
    excluded_intervals_by_session,
)
from continuum_swr_replay.session_keys import SESSION_COLUMNS

from .design_constants import (
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_INCOMPLETE,
    REWARD_HISTORY_N_BASIS,
    REWARD_HISTORY_N_LAGS,
)

__all__ = [
    "NO_EXCLUDED_TIME",
    "POPULATIONS",
    "aggregate_to_timebins",
    "attach_is_swr_to_onsets",
    "raised_cosine_history_basis",
    "reconstruct_reward_kernels",
    "reconstruct_self_history_kernels",
    "reward_basis_from_lags",
    "reward_kernel_rate_ratios",
    "reward_kernel_weights",
    "self_history_rate_ratios",
]

logger = logging.getLogger(__name__)

NO_EXCLUDED_TIME = pd.DataFrame(
    {
        **{column: pd.Series(dtype=object) for column in SESSION_COLUMNS},
        "start_time": pd.Series(dtype=float),
        "end_time": pd.Series(dtype=float),
    }
)
"""``excluded_intervals`` for :func:`aggregate_to_timebins` that masks nothing.

For synthetic stop tables, whose sessions the SWR source does not serve. The
default (``None``) reads each session's excluded time from the source instead.

One frame shared by every caller: do not mutate it (add rows, columns or
``attrs``). Pass it as it is, or pass a copy you change.
"""

_REWARD_LAG_COLUMNS = tuple(
    f"known_reward_outcome_lag_{j}" for j in range(1, REWARD_HISTORY_N_LAGS + 1)
)
#: Stop columns needed to place a bin on the arrival / outcome-availability clocks.
_ARRIVAL_CLOCK_STOP_COLUMNS = (
    "stop_start",
    "last_well_arrival_time_s",
    "outcome_known_time_s",
)
#: Extra stop columns needed to re-derive the reward history at the outcome boundary.
_OUTCOME_SHIFT_STOP_COLUMNS = (
    "stop_start",
    "outcome_known_time_s",
    "completed_trial_reward",
    *_REWARD_LAG_COLUMNS,
)
#: Stop columns read only at STOP level, to find the bouts that span an arrival.
#: They are never carried per bin (the per-bin design has its own clocks).
_ARRIVAL_SPAN_STOP_COLUMNS = (
    "stop_start",
    "stop_end",
    "last_well_arrival_time_s",
)


def attach_is_swr_to_onsets(
    onset_df: pd.DataFrame, event_table: pd.DataFrame
) -> pd.DataFrame:
    """Add a canonical ``is_swr`` column to the non-local onset rows.

    First freezes the intersection of natural session keys carried by both
    tables, then matches each non-local onset to its ``event_table`` event by
        exact ``(session, stop_id, onset_time == event_start_s)`` identity. Within
    every retained session the sources must be a complete one-to-one partition.
    Sessions present in only one artifact are excluded as an explicit
    regeneration/build-coverage gap rather than blocking valid shared sessions.
    SWR/MUA onset rows get ``is_swr = NaN``.

    Parameters
    ----------
    onset_df : pd.DataFrame
        Event-onset rows. Must carry ``stop_id``, ``event_type``, and
        ``onset_time``.
    event_table : pd.DataFrame
        Per-event table. Must carry ``stop_id``, ``event_start_s``, ``is_swr``.

    Returns
    -------
    pd.DataFrame
        Shared-session copy of ``onset_df`` with an added ``is_swr`` column (the
        matched boolean for ``event_type == "non_local"`` rows, ``NaN``
        elsewhere). ``attrs["event_session_alignment"]`` records shared and
        excluded session keys and row counts.
    """
    session_presence = {
        "onset": [column in onset_df for column in SESSION_COLUMNS],
        "event": [column in event_table for column in SESSION_COLUMNS],
    }
    for name, presence in session_presence.items():
        if any(presence) and not all(presence):
            missing = [
                column
                for column, present in zip(SESSION_COLUMNS, presence, strict=True)
                if not present
            ]
            raise KeyError(
                f"{name} table has a partial natural session key; missing {missing}"
            )
    has_session_keys = all(session_presence["onset"]) and all(session_presence["event"])
    if all(session_presence["onset"]) != all(session_presence["event"]):
        raise KeyError(
            "onset and event tables must either both carry natural session keys "
            f"{tuple(SESSION_COLUMNS)} or both be single-session fixtures"
        )

    if has_session_keys:
        if (
            onset_df[list(SESSION_COLUMNS)].isna().any().any()
            or event_table[list(SESSION_COLUMNS)].isna().any().any()
        ):
            raise ValueError("onset/event natural session keys must not be missing")
        onset_sessions = pd.MultiIndex.from_frame(
            onset_df[list(SESSION_COLUMNS)].drop_duplicates()
        )
        event_sessions = pd.MultiIndex.from_frame(
            event_table[list(SESSION_COLUMNS)].drop_duplicates()
        )
        shared_sessions = onset_sessions.intersection(event_sessions, sort=False)
        onset_only = onset_sessions.difference(event_sessions, sort=False)
        event_only = event_sessions.difference(onset_sessions, sort=False)
        if len(shared_sessions) == 0 and (len(onset_df) or len(event_table)):
            raise ValueError(
                "onset and event tables share no natural session keys; they are "
                "from incompatible cohorts or builds"
            )
        onset_keys = pd.MultiIndex.from_frame(onset_df[list(SESSION_COLUMNS)])
        event_keys = pd.MultiIndex.from_frame(event_table[list(SESSION_COLUMNS)])
        out = onset_df.loc[onset_keys.isin(shared_sessions)].copy()
        event_shared = event_table.loc[event_keys.isin(shared_sessions)].copy()
        merge_keys = [*SESSION_COLUMNS, "stop_id"]
    else:
        # Small direct unit fixtures historically omit natural keys. Treat both
        # complete frames as one implicit session and retain strict whole-frame
        # equality; never mix a keyed frame with an unkeyed one above.
        out = onset_df.copy()
        event_shared = event_table.copy()
        shared_sessions = (
            ["<implicit-single-session>"] if len(out) or len(event_shared) else []
        )
        onset_only = []
        event_only = []
        merge_keys = ["stop_id"]

    alignment_audit = {
        "n_onset_sessions": int(len(shared_sessions) + len(onset_only)),
        "n_event_sessions": int(len(shared_sessions) + len(event_only)),
        "n_shared_sessions": int(len(shared_sessions)),
        "onset_only_sessions": tuple(map(tuple, onset_only)),
        "event_only_sessions": tuple(map(tuple, event_only)),
        "n_onset_rows_input": int(len(onset_df)),
        "n_event_rows_input": int(len(event_table)),
        "n_onset_rows_retained": int(len(out)),
        "n_event_rows_retained": int(len(event_shared)),
    }
    if len(onset_only) or len(event_only):
        message = (
            "attach_is_swr_to_onsets excluded non-shared artifact sessions; "
            f"onset_only={alignment_audit['onset_only_sessions']}, "
            f"event_only={alignment_audit['event_only_sessions']}. Regenerate "
            "the missing sessions before full-corpus inference."
        )
        # The warning is the API-level signal, but every rate runner installs
        # ``filterwarnings("ignore", category=RuntimeWarning)`` at import, so a
        # warning alone would drop these sessions with no trace in the log. The
        # cohort actually fitted is a scientific fact, not a nuisance.
        logger.error(
            "%s Fitting %d of %d onset sessions (%d of %d onset rows).",
            message,
            alignment_audit["n_shared_sessions"],
            alignment_audit["n_onset_sessions"],
            alignment_audit["n_onset_rows_retained"],
            alignment_audit["n_onset_rows_input"],
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    # object dtype holds the matched booleans alongside NaN for SWR/MUA rows
    # without the float coercion pandas now warns about.
    out["is_swr"] = pd.Series(np.nan, index=out.index, dtype="object")
    # Join on the onset time itself, not on within-stop ordinal rank. A
    # non-local onset IS its event's start, so the two are exactly equal by
    # construction. Ranking instead makes one missing onset shift every later
    # rank in that stop, relabelling events that were present and correct --
    # and because the shifted ranks still all find *a* row, the match-rate
    # guard below stays silent through it. Matching on time contains the
    # failure to the row that is actually missing.
    ev = event_shared[[*merge_keys, "event_start_s", "is_swr"]].copy()
    nl = out[out["event_type"] == "non_local"].copy()
    if has_session_keys:
        onset_counts = nl.groupby(list(SESSION_COLUMNS), sort=False).size()
        event_counts = ev.groupby(list(SESSION_COLUMNS), sort=False).size()
        counts = pd.concat(
            [onset_counts.rename("onsets"), event_counts.rename("events")], axis=1
        ).fillna(0)
        bad_counts = counts[counts["onsets"] != counts["events"]]
    else:
        bad_counts = (
            pd.DataFrame()
            if len(nl) == len(ev)
            else pd.DataFrame(
                {"onsets": [len(nl)], "events": [len(ev)]}, index=["implicit"]
            )
        )
    if len(bad_counts):
        raise ValueError(
            "attach_is_swr_to_onsets requires an exact non-local onset/event "
            "partition within every shared session; mismatched counts: "
            f"{bad_counts.astype(int).to_dict(orient='index')}"
        )
    nl["_onset_order"] = np.arange(len(nl))
    merged = nl.merge(
        ev.rename(columns={"event_start_s": "onset_time"}),
        on=[*merge_keys, "onset_time"],
        how="left",
        suffixes=("", "_ev"),
        # Duplicate (stop_id, onset_time) keys would expand rows and misalign
        # the positional assignment below.
        validate="one_to_one",
    )
    # Any mismatch means the onset table and event_table were built from
    # divergent non-local/speed thresholds or decodes -- fail loud rather than
    # silently leave is_swr unlabelled downstream. On the shipped artifacts the
    # time join matches exactly; unmatched rows would make SWR and non-SWR fail
    # to partition the pooled non-local population.
    merged = merged.sort_values("_onset_order", kind="stable")
    matched = merged["is_swr_ev"].notna()
    if len(merged) and not matched.all():
        raise ValueError(
            "attach_is_swr_to_onsets requires every non-local onset to be exactly "
            f"matched; {int((~matched).sum())}/{len(matched)} were unmatched by "
            "(session, stop_id, onset_time). The onset table and "
            "event_table must be built from the same non-local/speed thresholds "
            "and decodes."
        )
    out.loc[nl.index, "is_swr"] = merged["is_swr_ev"].to_numpy()
    out.attrs["event_session_alignment"] = alignment_audit
    return out


def raised_cosine_history_basis(n_lags: int, n_basis: int) -> np.ndarray:
    """Log-spaced raised-cosine self-history basis (Pillow et al. 2008).

    Smooth, overlapping bumps over within-stop lags ``1..n_lags`` used to encode
    the self-history block. Projecting the ``onset_lag{j}`` counts onto these
    ``n_basis`` bumps replaces ``n_lags`` collinear lag dummies that otherwise
    perfectly separate on the refractory bins (coefficients -> -inf) and wreck
    held-out prediction.

    Parameters
    ----------
    n_lags : int
        Number of within-stop lags spanned (the history window in bins).
    n_basis : int
        Number of raised-cosine bumps.

    Returns
    -------
    np.ndarray, shape (n_basis, n_lags)
        ``B[k, j]`` is the weight of lag ``j+1`` in basis function ``k``.
    """
    x = np.log(np.arange(1, n_lags + 1))
    centers = np.linspace(x.min(), x.max(), n_basis)
    spacing = centers[1] - centers[0] if n_basis > 1 else 1.0
    B = np.zeros((n_basis, n_lags))
    for k, c in enumerate(centers):
        arg = np.clip((x - c) * np.pi / (2 * spacing), -np.pi, np.pi)
        B[k] = 0.5 * (1 + np.cos(arg))
    return B


def reward_basis_from_lags(lags: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project a lag matrix onto raised-cosine bumps, normalized per bump.

    Each output is the bump-weighted mean over the *available* (finite) lags,
    ``sum_j B[k, j] * lags[:, j] / sum_j B[k, j]``, so a query with only part of
    the history is still on the reward-fraction scale rather than being shrunk
    toward zero by the missing lags.

    Parameters
    ----------
    lags : np.ndarray, shape (n_query, n_lags)
        Per-lag reward outcomes (0/1), ``NaN`` where the lag is not known.
    basis : np.ndarray, shape (n_basis, n_lags)
        Bump weights; ``basis[k, j]`` is the weight of lag ``j + 1`` in bump
        ``k``.

    Returns
    -------
    np.ndarray, shape (n_query, n_basis)
        Normalized projections; ``NaN`` where no lag in the bump's support is
        finite.
    """
    finite = np.isfinite(lags)
    values = np.where(finite, lags, 0.0)  # (n_query, n_lags)
    num = values @ basis.T  # (n_query, n_basis)
    den = finite.astype(float) @ basis.T  # (n_query, n_basis)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den <= 0] = np.nan
    return out


def reward_kernel_weights(n_lags: int, n_basis: int) -> np.ndarray:
    """Per-lag weight each reward-history basis coefficient contributes.

    The persisted features are bump-weighted reward *fractions* (each bump is
    normalized by the sum of its weights over the lags it spans), so an outcome
    at lag ``j`` contributes ``sum_k coef_k * B[k, j] / sum_j' B[k, j']`` to the
    log rate. This returns that row-normalized basis.

    Parameters
    ----------
    n_lags : int
        Number of reward lags the basis spans.
    n_basis : int
        Number of raised-cosine bumps.

    Returns
    -------
    np.ndarray, shape (n_basis, n_lags)
        Row-normalized raised-cosine basis.
    """
    B = raised_cosine_history_basis(n_lags, n_basis)
    return B / B.sum(axis=1, keepdims=True)


def reconstruct_reward_kernels(
    coefs: np.ndarray, n_lags: int, n_basis: int
) -> np.ndarray:
    """Per-session reward kernels over lags ``1..n_lags``, on the log-rate scale.

    Parameters
    ----------
    coefs : np.ndarray, shape (n_sessions, n_basis)
        Fitted basis coefficients, one row per session. Rows with any NaN are
        dropped, so a session that did not fit the block contributes nothing.
    n_lags, n_basis : int
        Shape of the basis the coefficients were fit against.

    Returns
    -------
    np.ndarray, shape (n_fitted_sessions, n_lags)
        Log-rate contribution of a reward at each lag, per session.
    """
    coefs = np.atleast_2d(np.asarray(coefs, dtype=float))
    coefs = coefs[~np.isnan(coefs).any(axis=1)]
    return coefs @ reward_kernel_weights(n_lags, n_basis)


def reconstruct_self_history_kernels(
    coefs: np.ndarray, n_lags: int, n_basis: int
) -> np.ndarray:
    """Per-session self-history kernels over lags ``1..n_lags``, log-rate scale.

    The reward and self-history blocks share the raised-cosine bumps but not the
    projection. Reward features are bump-weighted *fractions* over the available
    lags, so their kernel is read off the ROW-NORMALIZED basis
    (:func:`reward_kernel_weights`); ``hist_rcos_k`` is the plain projection of
    the ``onset_lag{j}`` counts onto the raw bumps, so an onset ``j`` bins back
    contributes ``sum_k coef_k * B[k, j]`` with ``B`` un-normalized. Using the
    reward weights here would divide every lag by the bump's span.

    Parameters
    ----------
    coefs : np.ndarray, shape (n_sessions, n_basis)
        Fitted ``coef_hist_rcos_k`` values, one row per session. Rows with any
        NaN are dropped, so a session that did not fit the block contributes
        nothing.
    n_lags, n_basis : int
        Shape of the basis the coefficients were fit against.

    Returns
    -------
    np.ndarray, shape (n_fitted_sessions, n_lags)
        Log-rate contribution of an onset at each lag, per session.
    """
    coefs = np.atleast_2d(np.asarray(coefs, dtype=float))
    coefs = coefs[~np.isnan(coefs).any(axis=1)]
    return coefs @ raised_cosine_history_basis(n_lags, n_basis)


def reward_kernel_rate_ratios(sessions: pd.DataFrame) -> np.ndarray:
    """Across-session median of the per-session reconstructed reward kernel.

    Each session's kernel is reconstructed from that session's own basis
    coefficients and the median is then taken per lag. Reconstructing from
    across-session median coefficients instead would summarize a session that
    was never fit: the bumps are strongly collinear, so their marginal medians
    describe no fitted kernel.

    Parameters
    ----------
    sessions : pd.DataFrame
        One row per session for a single (model, outcome), carrying
        ``coef_reward_history_rcos_*``. Sessions with any NaN basis coefficient
        contribute nothing.

    Returns
    -------
    np.ndarray, shape (REWARD_HISTORY_N_LAGS,)
        Median rate ratio at each lag ``1..REWARD_HISTORY_N_LAGS``.
    """
    columns = [f"coef_{c}" for c in REWARD_HISTORY_COLUMNS]
    kernels = reconstruct_reward_kernels(
        sessions[columns].to_numpy(dtype=float),
        REWARD_HISTORY_N_LAGS,
        REWARD_HISTORY_N_BASIS,
    )
    return np.median(np.exp(kernels), axis=0)


def self_history_rate_ratios(sessions: pd.DataFrame) -> np.ndarray:
    """Across-session median of the per-session reconstructed self-history kernel.

    ``hist_rcos_k`` is the plain projection of the ``onset_lag{j}`` counts onto
    the raw raised-cosine bumps, so an onset ``j`` bins back contributes
    ``sum_k coef_k * B[k, j]`` to the log rate -- the un-normalized basis,
    unlike the reward block's bump-weighted fractions.

    Parameters
    ----------
    sessions : pd.DataFrame
        One row per session for a single (model, outcome), carrying
        ``coef_hist_rcos_*``. Sessions with any NaN basis coefficient
        contribute nothing.

    Returns
    -------
    np.ndarray, shape (N_HISTORY_LAGS,)
        Median rate ratio at each lag ``1..N_HISTORY_LAGS``.
    """
    columns = [f"coef_hist_rcos_{k}" for k in range(N_HISTORY_BASIS)]
    kernels = reconstruct_self_history_kernels(
        sessions[columns].to_numpy(dtype=float), N_HISTORY_LAGS, N_HISTORY_BASIS
    )
    return np.median(np.exp(kernels), axis=0)


def _arrival_spanning_stop_ids(stops: pd.DataFrame) -> np.ndarray:
    """``stop_id`` of the bouts that contain a later well arrival.

    A stop spans an arrival iff the NEXT stop of the same session -- ordering
    by ``stop_start`` within ``animal, session_date, epoch`` -- reports a
    ``last_well_arrival_time_s`` strictly inside ``(stop_start, stop_end]`` of
    this bout. Well arrivals are monotone within a session, so only the next
    stop's arrival can fall inside this one and the test is exact. A stop with
    no successor, or whose successor has no arrival clock, does not span.

    Parameters
    ----------
    stops : pd.DataFrame
        Per-stop frame (unique ``stop_id``) carrying ``animal``,
        ``session_date``, ``epoch``, ``stop_start``, ``stop_end`` and
        ``last_well_arrival_time_s``.

    Returns
    -------
    np.ndarray
        The spanning stops' ``stop_id`` values (possibly empty).
    """
    ordered = stops.sort_values([*SESSION_COLUMNS, "stop_start"], kind="stable")
    next_arrival = ordered.groupby(list(SESSION_COLUMNS), sort=False, observed=True)[
        "last_well_arrival_time_s"
    ].shift(-1)
    spans = (next_arrival > ordered["stop_start"]) & (
        next_arrival <= ordered["stop_end"]
    )
    return ordered.loc[spans.fillna(False).to_numpy(), "stop_id"].to_numpy()


def aggregate_to_timebins(
    stop_table: pd.DataFrame,
    onset_times: pd.DataFrame,
    *,
    bin_s: float,
    tmax: float,
    reward_basis: bool = False,
    self_history_lags: int = 0,
    self_history_basis: int = 0,
    cross_onset_times: pd.DataFrame | None = None,
    cross_history_lags: int = 0,
    cross_history_basis: int = 0,
    stop_covariates: tuple[str, ...] = (),
    excluded_intervals: pd.DataFrame | None = None,
    history_window_s: float = 0.0,
) -> pd.DataFrame:
    """One row per (stop, bin) up to ``tmax``, at ``bin_s`` resolution.

    Builds the canonical (stop x bin) point-process design used by the per-bin
    event-rate models. Bins are built over each stop's span, ``stop_end -
    stop_start`` (``immobility_duration_s`` when the table has no bounds; the
    two are equal unless excluded time took seconds out of the stop),
    independent of any dense 4 ms grid, so non-4-ms widths (e.g. 15 ms) are
    exact. The last bin's ``exposure`` is the partial remainder. A stop with no
    exposure (``immobility_duration_s <= 0``) has no bins.

    Excluded time (the artifact mask widened by 0.3 s) is not analysed: a bin is
    dropped when ``[left - history_window_s, left + bin]`` (its history clipped
    at the stop's start) touches it. The drop comes last, so every kept bin's
    history columns are those of the design without the mask.

    Parameters
    ----------
    stop_table : pd.DataFrame
        Per-stop table (unique ``stop_id``). Must carry ``animal``,
        ``session_date``, ``epoch``, ``stop_id``, ``immobility_duration_s``,
        plus the reward-history basis columns (when ``reward_basis``) and any
        ``stop_covariates`` columns requested.
    onset_times : pd.DataFrame
        Event-onset rows for ONE population (already filtered). Must carry
        ``stop_id`` and ``t_since_stop_onset``.
    bin_s : float
        Bin width in seconds.
    tmax : float
        Right-truncation; bins beyond ``tmax`` are dropped.
    reward_basis : bool
        Carry the reward-history basis features ``REWARD_HISTORY_COLUMNS`` and
        the structural-missingness indicator ``REWARD_HISTORY_INCOMPLETE`` from
        ``stop_table``.
    self_history_lags : int
        Number of ``onset_lag{j}`` self-history columns to build (same-
        population onset counts ``j`` bins back, within stop, reset at the
        stop boundary).
    self_history_basis : int
        If > 0, also add ``hist_rcos_0..self_history_basis-1`` columns: the
        ``onset_lag`` counts projected onto a log-spaced raised-cosine basis
        (see :func:`raised_cosine_history_basis`). Requires ``self_history_lags
        > 0`` (the basis spans that window). Used by the ``*_histcos`` models,
        which generalize where the lag dummies overfit the refractory bins.
    stop_covariates : tuple[str, ...]
        Extra per-stop columns carried from ``stop_table`` by ``stop_id``.
    excluded_intervals : pd.DataFrame, optional
        Each session's excluded time: ``animal``, ``session_date``,
        ``epoch``, ``start_time`` and ``end_time`` (absolute seconds, closed).
        None (the default) reads it for ``stop_table``'s sessions from the SWR
        source, ``swr_source.excluded_intervals_by_session(stop_table)``; with
        no artifact run configured that has no rows. A frame with no rows (such
        as :data:`NO_EXCLUDED_TIME`, for synthetic sessions the source does not
        serve) masks nothing and leaves ``attrs`` without the excluded counts.
        Requires ``stop_start`` in ``stop_table`` when it has rows.
    history_window_s : float
        Seconds the ladder's longest history covariate looks back
        (``model_specs.longest_history_window_s``); 0 drops only the bins that
        touch the mask.

    Returns
    -------
    pd.DataFrame
        Columns: ``animal``, ``session_date``, ``epoch``, ``stop_id``, ``binj``,
        ``tc`` (bin-center s = time since stop onset), ``exposure`` (s of
        immobility in the bin), the reward-history basis columns (if
        ``reward_basis``), each name in
        ``stop_covariates``, ``count`` (population onsets in the bin),
        ``onset_lag1..self_history_lags``, and (if ``self_history_basis``)
        ``hist_rcos_0..self_history_basis-1``. When ``stop_table`` carries
        ``stop_start``, ``last_well_arrival_time_s`` and
        ``outcome_known_time_s``, two further per-bin clocks are emitted:
        ``time_since_last_well_arrival_s`` and ``time_since_outcome_known_s``
        (bin centre minus the absolute clock; ``NaN`` where the clock is
        ``NaN``, and negative for bins that precede the outcome boundary).
        When excluded time has rows, ``attrs`` counts the bins dropped for
        touching it (``n_bins_excluded``), those dropped for their
        history window alone (``n_bins_excluded_history``) and the onsets
        those bins held (``n_onsets_in_excluded_bins``).

    Notes on the outcome boundary
    -----------------------------
    A stop's reward history is evaluated at bout start, so a stop that is still
    in progress when the current trip's reward or omission becomes observable
    would carry a history missing that outcome for its later bins. When
    ``reward_basis`` is set and ``stop_table`` additionally carries
    ``stop_end``, ``last_well_arrival_time_s``, ``completed_trial_reward`` and
    the ``known_reward_outcome_lag_{1..10}`` columns, bins whose LEFT EDGE is at
    or past ``outcome_known_time_s`` (for a stop whose boundary falls inside it)
    get the lag vector shifted by one --
    ``[completed_trial_reward, lag_1, ..., lag_9]`` -- and the basis features and
    ``reward_history_incomplete`` re-derived from it by the same functions the
    stop level uses. Bins before the boundary keep the bout-start values. The
    one exception is a bout that spans a well arrival -- the next stop of the
    session arrived inside ``(stop_start, stop_end]`` (see
    :func:`_arrival_spanning_stop_ids`) -- which changes which trip is the
    completed one partway through, so it is never shifted;
    ``out.attrs["n_stops_arrival_spanning"]`` counts how many such stops have at
    least one bin at or after the boundary (a boundary that falls inside a stop
    but beyond its last retained bin leaves no bin to shift and is not counted).

    The absolute stop clocks are carried only to derive these per-bin columns
    and are dropped before return unless explicitly requested through
    ``stop_covariates``; an absolute session clock is never a model predictor.

    Raises
    ------
    ValueError
        If the binned counts do not reconcile with the onsets that fall inside
        the stops' exposed windows (``t_since_stop_onset < min(tmax,
        immobility_duration_s)``) AND land in a bin the exposure filter keeps --
        e.g. an onset at or past the end of its stop's immobility, which is
        binned into the stop's last bin but earns no exposure there.
    KeyError
        If excluded time has rows and ``stop_table`` has no ``stop_start``.
    SwrSourceError
        With the default ``excluded_intervals``, if an artifact run is
        configured but does not serve one of ``stop_table``'s sessions (as
        ``swr_source.artifact_intervals``).

    Notes
    -----
    A stop whose immobility ends a sliver past a bin edge has a final bin with
    ``exposure <= 1e-4``, which is dropped from the design; an onset in that
    sliver is therefore not counted anywhere. That is a real (tiny) loss rather
    than an inconsistency, so it does not raise -- it is counted in
    ``out.attrs["n_onsets_in_dropped_bins"]`` so it stays visible.
    """
    if self_history_basis and not self_history_lags:
        raise ValueError("self_history_basis requires self_history_lags > 0")
    if cross_history_basis and not cross_history_lags:
        raise ValueError("cross_history_basis requires cross_history_lags > 0")
    if cross_history_lags and cross_onset_times is None:
        raise ValueError("cross_history_lags requires cross_onset_times")
    reward_cols = (
        [*REWARD_HISTORY_COLUMNS, REWARD_HISTORY_INCOMPLETE] if reward_basis else []
    )
    # Optional per-bin clocks / outcome-boundary shift: both are gated on the
    # stop frame actually carrying their inputs, so a stop table built before
    # the arrival clocks existed produces exactly the design it produced before.
    emit_clocks = all(c in stop_table.columns for c in _ARRIVAL_CLOCK_STOP_COLUMNS)
    shift_at_outcome = reward_basis and all(
        c in stop_table.columns
        for c in (*_OUTCOME_SHIFT_STOP_COLUMNS, *_ARRIVAL_SPAN_STOP_COLUMNS)
    )
    if excluded_intervals is None:
        excluded_intervals = excluded_intervals_by_session(stop_table)
    # an empty mask masks nothing, so a stop table without bounds still builds
    mask_excluded = len(excluded_intervals) > 0
    if mask_excluded and "stop_start" not in stop_table.columns:
        raise KeyError("excluded_intervals needs stop_start in stop_table")
    internal_cols = list(
        dict.fromkeys(
            (_ARRIVAL_CLOCK_STOP_COLUMNS if emit_clocks else ())
            + (_OUTCOME_SHIFT_STOP_COLUMNS if shift_at_outcome else ())
            + (("stop_start",) if mask_excluded else ())
        )
    )
    carry = list(dict.fromkeys([*reward_cols, *internal_cols, *stop_covariates]))
    # Carried only to derive the per-bin columns; the absolute session clocks in
    # particular must never reach a model as predictors (see covariate_timing).
    drop_internal = [c for c in internal_cols if c not in stop_covariates]

    s = stop_table[stop_table["immobility_duration_s"] > 0].copy()
    # The bins' clock is the bout's: the mask takes time out of a stop's
    # exposure (immobility_duration_s), not off its end, so the bins span the
    # whole stop and the masked ones are dropped below.
    if "stop_start" in s.columns and "stop_end" in s.columns:
        s["_span_s"] = s["stop_end"].to_numpy() - s["stop_start"].to_numpy()
    else:
        s["_span_s"] = s["immobility_duration_s"].to_numpy()
    # ceil, not int: when tmax is not a multiple of bin_s, int() drops the final
    # partial window [floor(tmax/bin_s)*bin_s, tmax) so onsets there are lost.
    nmax = int(np.ceil(tmax / bin_s))
    nbin = np.minimum(np.ceil(s["_span_s"].to_numpy() / bin_s), nmax).astype(int)
    idx = np.repeat(np.arange(len(s)), nbin)
    # position of each bin within its stop: 0..nbin-1, stop after stop
    binj = np.arange(len(idx)) - np.repeat(np.cumsum(nbin) - nbin, nbin)
    cols = [*SESSION_COLUMNS, "stop_id", "_span_s", *carry]
    out = s.iloc[idx][cols].reset_index(drop=True)
    out["binj"] = binj
    out["tc"] = (binj + 0.5) * bin_s
    if emit_clocks:
        # Bin CENTRE on two absolute session clocks, as elapsed seconds. The
        # arrival clock is >= 0 for an at-well stop (the animal arrived before
        # the bout began); the outcome clock is NEGATIVE for bins that precede
        # the reward/omission becoming observable.
        bin_center_abs = out["stop_start"].to_numpy(dtype=float) + (binj + 0.5) * bin_s
        out["time_since_last_well_arrival_s"] = bin_center_abs - out[
            "last_well_arrival_time_s"
        ].to_numpy(dtype=float)
        out["time_since_outcome_known_s"] = bin_center_abs - out[
            "outcome_known_time_s"
        ].to_numpy(dtype=float)
    # bin width, clipped by both the stop's immobility end and tmax (the final
    # bin of a stop longer than tmax is exposed only up to tmax, not bin_s).
    out["exposure"] = np.minimum(bin_s, np.minimum(out["_span_s"], tmax) - binj * bin_s)
    out = out[out["exposure"] > 1e-4]

    src = onset_times[onset_times["t_since_stop_onset"] < tmax]
    # floor with a tolerance, not int(): int() truncates the quotient, and an
    # onset exactly on a bin edge whose binary-float quotient lands just below
    # the integer (0.3 / 0.1 -> 2.9999999999999996) would fall one bin early,
    # shifting the estimated time course backwards. The clip keeps an onset in
    # the final partial window [(nmax-1)*bin_s, tmax) in the last bin rather
    # than a phantom bin that the merge below would silently drop.
    src_binj = np.clip(
        np.floor(src["t_since_stop_onset"].to_numpy() / bin_s + 1e-9).astype(int),
        0,
        nmax - 1,
    )
    cnt = (
        pd.DataFrame({"stop_id": src["stop_id"].to_numpy(), "binj": src_binj})
        .groupby(["stop_id", "binj"])
        .size()
        .rename("count")
    )
    out = out.merge(cnt, on=["stop_id", "binj"], how="left")
    out["count"] = out["count"].fillna(0)
    # Every onset inside a stop's exposed window whose bin survives the exposure
    # filter must land on exactly one row of the design; a mismatch means the
    # response and the exposure disagree, which a Poisson fit cannot detect.
    # The one legitimate loss is an onset in a final sliver bin that the filter
    # removed (immobility ending just past a bin edge): it is counted, not
    # raised on, so the design stays consistent and the loss stays visible.
    expected = src[["stop_id", "t_since_stop_onset"]].merge(
        s[["stop_id", "_span_s"]], on="stop_id", how="inner"
    )
    t_expected = expected["t_since_stop_onset"].to_numpy(dtype=float)
    span = expected["_span_s"].to_numpy(dtype=float)
    stop_end = np.minimum(span, tmax)
    in_window = t_expected < stop_end
    expected_binj = np.clip(
        np.floor(t_expected / bin_s + 1e-9).astype(int), 0, nmax - 1
    )
    bin_kept = (stop_end - expected_binj * bin_s) > 1e-4
    n_expected = int((in_window & bin_kept).sum())
    n_onsets_in_dropped_bins = int((in_window & ~bin_kept).sum())
    n_binned = int(out["count"].sum())
    if n_binned != n_expected:
        raise ValueError(
            f"aggregate_to_timebins binned {n_binned} onset(s) but {n_expected} "
            "fall inside a stop's exposed window "
            "(t_since_stop_onset < min(tmax, the stop's span)) and in a "
            "retained bin; the (stop x bin) design's counts and exposure disagree"
        )

    if self_history_lags:
        out = out.sort_values(["stop_id", "binj"])
        grouped = out.groupby("stop_id")["count"]
        for j in range(1, self_history_lags + 1):
            out[f"onset_lag{j}"] = grouped.shift(j).fillna(0)

    if self_history_basis:
        B = raised_cosine_history_basis(self_history_lags, self_history_basis)
        lag_cols = [f"onset_lag{j}" for j in range(1, self_history_lags + 1)]
        proj = out[lag_cols].to_numpy() @ B.T
        for k in range(self_history_basis):
            out[f"hist_rcos_{k}"] = proj[:, k]

    # Cross-population history: lagged onset counts of a different event stream
    # (e.g. with-SWR non-local onsets as predictors of the non-SWR rate). Binned
    # on the SAME grid as the response, shifted within stop (lag >= 1 -> only the
    # cross stream's past predicts the current bin), reset at the stop boundary.
    if cross_history_lags:
        xsrc = cross_onset_times[cross_onset_times["t_since_stop_onset"] < tmax]
        xbinj = np.clip(
            np.floor(xsrc["t_since_stop_onset"].to_numpy() / bin_s + 1e-9).astype(int),
            0,
            nmax - 1,
        )
        xcnt = (
            pd.DataFrame({"stop_id": xsrc["stop_id"].to_numpy(), "binj": xbinj})
            .groupby(["stop_id", "binj"])
            .size()
            .rename("cross_count")
        )
        out = out.merge(xcnt, on=["stop_id", "binj"], how="left")
        out["cross_count"] = out["cross_count"].fillna(0)
        # A left merge on keys unique in ``xcnt`` keeps the rows in order, so
        # the self-history block's sort still holds.
        if not self_history_lags:
            out = out.sort_values(["stop_id", "binj"])
        xgrouped = out.groupby("stop_id")["cross_count"]
        for j in range(1, cross_history_lags + 1):
            out[f"cross_lag{j}"] = xgrouped.shift(j).fillna(0)
        if cross_history_basis:
            Bx = raised_cosine_history_basis(cross_history_lags, cross_history_basis)
            xlag_cols = [f"cross_lag{j}" for j in range(1, cross_history_lags + 1)]
            xproj = out[xlag_cols].to_numpy() @ Bx.T
            for k in range(cross_history_basis):
                out[f"cross_hist_rcos_{k}"] = xproj[:, k]
        out = out.drop(columns=["cross_count"])

    n_stops_arrival_spanning: int | None = None
    if shift_at_outcome:
        # The reward history a stop carries is evaluated at BOUT START, so bins
        # after the current trip's reward/omission became observable would
        # otherwise be modelled with a history that omits the outcome the animal
        # already has. Shift the lag vector by one at the boundary and re-derive
        # the basis features with the same functions the stop level uses, so the
        # two levels stay one kernel rather than two.
        stop_start = out["stop_start"].to_numpy(dtype=float)
        boundary = out["outcome_known_time_s"].to_numpy(dtype=float)
        left_edge_abs = stop_start + out["binj"].to_numpy(dtype=float) * bin_s
        # NaN boundary -> False, so a stop with no known outcome is untouched.
        # ``stop_start < boundary`` excludes stops already past their outcome at
        # bout start (their bout-start history already contains it); the bins of
        # a stop whose boundary sits beyond its last bin never satisfy ``post``.
        inside = (stop_start < boundary) & (left_edge_abs >= boundary)
        # A bout that spans a well arrival changes WHICH trip is the completed
        # one partway through, so ``completed_trial_reward`` is not the outcome
        # that becomes known at the boundary; those stops are left on their
        # bout-start history and counted instead of silently mis-shifted. A bout
        # spans an arrival iff the NEXT stop of the same session (ordered by
        # ``stop_start``) arrived at its well inside ``(stop_start, stop_end]``;
        # arrivals are monotone in the session, so the next stop's is the only
        # one that can fall there and the test is exact.
        spans_arrival = out["stop_id"].isin(_arrival_spanning_stop_ids(s)).to_numpy()
        crosses = inside & ~spans_arrival
        n_stops_arrival_spanning = int(
            pd.unique(out["stop_id"].to_numpy()[inside & spans_arrival]).size
        )

        lags = out[list(_REWARD_LAG_COLUMNS)].to_numpy(dtype=float)
        shifted = lags.copy()
        shifted[crosses, 1:] = lags[crosses, :-1]
        shifted[crosses, 0] = out["completed_trial_reward"].to_numpy(dtype=float)[
            crosses
        ]
        raw = reward_basis_from_lags(
            shifted,
            raised_cosine_history_basis(REWARD_HISTORY_N_LAGS, REWARD_HISTORY_N_BASIS),
        )
        out[REWARD_HISTORY_INCOMPLETE] = np.isnan(raw).any(axis=1).astype(float)
        for k, col in enumerate(REWARD_HISTORY_COLUMNS):
            out[col] = np.nan_to_num(raw[:, k], nan=0.0)

    excluded_counts: dict[str, int] = {}
    if mask_excluded:
        out, excluded_counts = _drop_excluded_bins(
            out, excluded_intervals, bin_s=bin_s, history_window_s=history_window_s
        )
    out = out.drop(columns=["_span_s", *drop_internal]).reset_index(drop=True)
    out.attrs["n_onsets_in_dropped_bins"] = n_onsets_in_dropped_bins
    out.attrs.update(excluded_counts)
    if n_stops_arrival_spanning is not None:
        out.attrs["n_stops_arrival_spanning"] = n_stops_arrival_spanning
    return out


def _drop_excluded_bins(
    design: pd.DataFrame,
    excluded_intervals: pd.DataFrame,
    *,
    bin_s: float,
    history_window_s: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop the bins that touch a session's excluded time, or whose history
    reaches it.

    Bin ``binj`` of a stop is ``[left, left + exposure]`` with ``left =
    stop_start + binj * bin_s``; its history looks back ``history_window_s``
    from ``left``, never before ``stop_start`` (the lags reset at the stop
    boundary). Intervals are closed, so touching counts.

    Returns
    -------
    kept : pd.DataFrame
        The other rows, in order.
    counts : dict
        ``n_bins_excluded``, ``n_bins_excluded_history`` and
        ``n_onsets_in_excluded_bins``.
    """
    if not (np.isfinite(history_window_s) and history_window_s >= 0):
        raise ValueError(
            f"history_window_s must be finite and >= 0: {history_window_s}"
        )
    stop_start = design["stop_start"].to_numpy(dtype=float)
    left = stop_start + design["binj"].to_numpy(dtype=float) * bin_s
    right = left + design["exposure"].to_numpy(dtype=float)
    look_from = np.maximum(left - history_window_s, stop_start)
    in_mask = np.zeros(len(design), dtype=bool)
    reached = np.zeros(len(design), dtype=bool)
    keys = list(SESSION_COLUMNS)
    if len(design) and len(excluded_intervals):
        session_of_bin = pd.MultiIndex.from_frame(design[keys])
        for key, intervals in excluded_intervals.groupby(keys, sort=False):
            rows = session_of_bin.isin([key])
            in_mask[rows] = overlaps_intervals(left[rows], right[rows], intervals)
            reached[rows] = overlaps_intervals(look_from[rows], right[rows], intervals)
    counts = {
        "n_bins_excluded": int(in_mask.sum()),
        "n_bins_excluded_history": int((reached & ~in_mask).sum()),
        "n_onsets_in_excluded_bins": int(design.loc[reached, "count"].sum()),
    }
    return design.loc[~reached], counts


# given onset_df already passed through attach_is_swr_to_onsets:
POPULATIONS = {
    "swr": lambda o: o[o.event_type == "SWR"],
    "non_local": lambda o: o[o.event_type == "non_local"],
    "non_local_swr": lambda o: o[
        (o.event_type == "non_local") & (o.is_swr == True)  # noqa: E712
    ],
    "non_local_no_swr": lambda o: o[
        (o.event_type == "non_local") & (o.is_swr == False)  # noqa: E712
    ],
}
