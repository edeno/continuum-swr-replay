"""The paper's non-local event detector applied to a decode, on one time-indexed frame.

This module aggregates decoding results, posterior statistics, event detections, and
behavioral data into comprehensive time-indexed DataFrames ready for visualization
and statistical analysis. It serves as the bridge between raw decoding output and
downstream analyses.

The main function, get_data_over_time, combines:
- Decoding state probabilities (local, non-local)
- Posterior statistics (mean position, variance, HPD coverage)
- Spatial variables (ahead/behind distance, MAP speed)
- Event labels (ripples, confident events, movement)
- Behavioral data (position, speed, task variables)

Main Function
-------------
get_data_over_time
    Primary aggregation function that combines all analysis components into a single
    time-aligned DataFrame. Returns both the aggregated DataFrame and the conditional
    non-local posterior distribution.

Aggregated Data
---------------
The output DataFrame contains time-indexed columns for:

Event labels (integer event numbers):
    - non_local_event: Non-local event number (0 if not in event)
    - SWR_event: Sharp-wave ripple event number, from the stored Spyglass
      ``RippleTimesV1`` ripples. This is the decode's own provenance: analyses
      read a saved frame through ``analysed_time.read_dot_with_swr_source``, which
      replaces it with the paper's SWR source (the direct Kay inventory). An
      HPC decode saves its frame with the source's labels
      (``analysed_time.with_swr_source(..., drop_masked_events=False)``): that
      frame keeps the SWRs in the 0.3-s reach of the artifact mask, while its
      ``swr_observed`` already excludes the reach, so both are provenance and
      analyses re-read the frame through the source.
    - swr_observed: True where SWR detection was observed and is analysed (the
      stored band's stretches less excluded time); added to the saved frame
      with the source's labels, and again by every read through the source.
      In the saved frame it already excludes the 0.3-s reach whose SWR labels
      the frame keeps.
    - in_excluded_time: True where the row's time lies in the epoch's
      excluded time (the artifact mask widened by 0.3 s,
      ``analysed_time.excluded_intervals``, closed); added by
      ``analysed_time.with_swr_source`` in both its modes, so by every read
      through the source; all False with no artifact run. A reader that
      summarises bins leaves these rows out.
    - MUA_event: Multiunit activity high-synchrony event number

State probabilities (0-1):
    - Local_prob: Probability of local state
    - No-Spike_prob: Probability of no-spike state
    - Non-Local_Continuous_prob: Probability of continuous non-local state
    - Non-Local_Fragmented_prob: Probability of fragmented non-local state

Spatial statistics:
    - map_estimate: Maximum a posteriori position estimate (cm)
    - map_speed: Speed of MAP position estimate (cm/s)
    - ahead_behind_distance: Signed distance from actual to decoded position (cm)
    - hpd: Highest posterior density threshold value
    - run_speed_baseline: The epoch's run-speed baseline (cm/s), a percentile
      of head speed (``epoch_normalizers.run_speed_baseline``), one value per
      epoch. Taken over every bin by ``get_data_over_time``;
      ``analysed_time.with_swr_source`` (every read, and a decoder saving its
      frame) takes it again over the bins outside the artifact mask.
    - max_coverage: Maximum HPD spatial coverage
    - max_distance: Maximum distance in posterior
    - max_linear_position: Maximum linear position

Neural signals (z-scored or phase):
    - theta_phase: Theta oscillation phase (radians)
    - theta_power_zscore: Z-scored theta power. Z-scored over every bin by
      ``get_data_over_time``; ``analysed_time.with_swr_source`` (every read, and
      a decoder saving its frame) re-z-scores it over the bins outside the
      artifact mask and sets it to NaN inside the mask. ``SWR_zscore`` is also
      NaN inside the mask; ``MUA_zscore`` leaves the mask out of its
      normalization but stays finite inside it.
    - SWR_zscore: Z-scored ripple consensus trace
    - MUA_zscore: Z-scored multiunit firing rate

Event counts (session-level; the ``*SWR*`` counts are of the stored ripples,
and every count includes events the analysed-time rule drops, so all of them
are dropped by ``analysed_time.with_swr_source``):
    - n_non_local, n_SWR, n_MUA: Whole-event-speed-eligible event counts
    - n_non_local_SWR, n_non_local_no_SWR: Non-local overlaps with/without SWR
    - n_non_local_MUA, n_non_local_no_MUA: Non-local overlaps with/without MUA
    - n_SWR_MUA, n_SWR_no_MUA: SWR overlaps with/without MUA

Session metadata:
    - animal: Animal name (str)
    - day: Recording date (int, YYYYMMDD)
    - epoch: Epoch name (str)
    - nwb_file_name: NWB file name (str)

Position and behavioral data:
    - All columns from position_info DataFrame (position, speed, track segment, etc.)

Filtering and Processing
-------------------------
- Non-Local, SWR, and MUA cohorts require every event speed sample to be
  finite and strictly below 4 cm/s; an ineligible event is removed atomically
- Missing values are handled via nearest-neighbor reindexing with tolerance
- Z-scoring is applied to relevant continuous variables
- Event overlap detection uses interval-based logic
- All data is aligned to the same temporal resolution (default 500 Hz)

Notes
-----
- All data is time-indexed in seconds
- Non-local threshold typically set to 0.8 for classification
- Ahead/behind distance uses graph shortest path (can be negative)
- MAP speed is smoothed before computation to reduce noise
- Missing behavioral data is handled via nearest-neighbor alignment

See Also
--------
continuum_swr_replay.analysis.decode_quality.posterior_analysis : Posterior statistics
continuum_swr_replay.analysis.event_definition.nonlocal_event_rule : Event detection functions
continuum_swr_replay.data_loaders.bandit_task : Session data loading

Examples
--------
>>> from continuum_swr_replay.decoding.results import load_results
>>> from continuum_swr_replay.analysis.detection.nonlocal_events import get_data_over_time
>>> results, detector, env = load_results("chimi20200212_.nwb", "02_r1")
>>> data_df, posterior = get_data_over_time(
...     data, results, env, "chimi20200212_.nwb", "02_r1",
...     non_local_threshold=0.80
... )
>>> # Access aggregated data
>>> data_df[["non_local_event", "ahead_behind_distance", "map_speed"]].head()
>>> # Access state probabilities
>>> data_df[["Local_prob", "Non-Local_Continuous_prob"]].head()

"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from non_local_detector.environment import Environment
from scipy.stats import zscore

from continuum_swr_replay.analysis.event_definition.analysed_time import (
    read_dot_with_swr_source,
)
from continuum_swr_replay.analysis.event_definition.epoch_normalizers import (
    run_speed_baseline as epoch_run_speed_baseline,
)
from continuum_swr_replay.analysis.event_definition.immobility import (
    EventDefinitionError,
    build_speed_eligible_event_cohort,
)
from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    detect_nonlocal_events,
    event_strict_overlap_keys,
    get_overlap_times,
    nonlocal_probability,
    nonlocal_state_probability,
)
from continuum_swr_replay.data_loaders.epoch_register import analysed_decode_files
from continuum_swr_replay.decoding.files import data_over_time_suffix
from continuum_swr_replay.decoding.identifiers import parse_nwb_filename
from continuum_swr_replay.parameters import (
    DEFAULT_NON_LOCAL_THRESHOLD,
    SAMPLING_FREQUENCY,
)

from ...data_types import SessionData
from ..decode_quality.posterior_analysis import (
    get_conditional_non_local_posterior,
    get_map_step_distance,
    get_posterior_stats,
)


def theta_power_zscore(theta_power: np.ndarray) -> np.ndarray:
    """Z-score a theta-power trace, ignoring NaN entries.

    ``_nearest``-interpolated traces can leave NaN at any timebin
    where no LFP sample fell within tolerance. The default
    ``scipy.stats.zscore(...)`` would propagate that NaN and zero out
    the entire session's z-score, silently suppressing theta event
    classification downstream in
    :func:`quality_metrics.compute_non_local_event_context`. Pass
    ``nan_policy="omit"`` so finite values keep finite z-scores.
    Shared between :func:`get_data_over_time` (this module) and the
    ContFrag aggregator so a regression in either path fails the
    same regression test.
    """
    return np.asarray(zscore(np.asarray(theta_power), nan_policy="omit"))


def validate_required_session_data_keys(
    data: Mapping[str, object],
    required_keys: list[str],
) -> None:
    """Refuse to aggregate when ``data`` is missing required keys.

    Both ``get_data_over_time`` (this module) and
    :func:`continuum_swr_replay.analysis.contfrag.data_aggregation.get_contfrag_data_over_time`
    require the same set of session-data keys (position, theta,
    ripple, MUA, …). Routing both call sites through this helper
    keeps the contract — and the error message — in one place so a
    test can pin the validator behavior directly without
    instantiating the full aggregation pipeline.
    """
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise KeyError(
            f"Missing required data keys: {missing_keys}. "
            "Ensure load_data() was called with "
            "DataLoadingPreset.DECODING or equivalent."
        )


def validate_results_position_time_alignment(
    results_time: np.ndarray,
    position_index: pd.Index,
) -> None:
    """Refuse to attach decoder outputs to position rows on a
    different timebase.

    The aggregations in this module and in
    :mod:`continuum_swr_replay.analysis.contfrag.data_aggregation`
    build ``data_over_time`` on ``position_info.index`` while pulling
    column values from ``results.time``-indexed arrays. A length-only
    check passes when the two axes share row count but are time-
    shifted, silently attaching event labels / probabilities / MAP
    stats to the wrong position rows. Pinning the stricter
    ``np.abs(results.time - position.index).max() > 1e-6`` guard
    here — and calling it from both aggregators — keeps the two paths
    in lockstep so a regression on either path fails the same
    regression test (mirrors the alignment check used in
    ``analysis.event_tables._common.align_posterior_to_dot``).
    """
    if len(results_time) != len(position_index):
        raise ValueError(
            f"results.time ({len(results_time)}) and position_info "
            f"({len(position_index)}) have different lengths; "
            "they must be aligned 1-to-1."
        )
    drift = float(
        np.abs(
            np.asarray(results_time, dtype=float) - position_index.to_numpy(dtype=float)
        ).max()
    )
    if drift > 1e-6:
        raise ValueError(
            f"results.time drifts from position_info.index by up to "
            f"{drift:.3g}s — same-length axes do not imply same time "
            "origin; refusing to attach decoder outputs to position "
            "rows that don't share their timebase."
        )


def align_results_to_position(
    results: xr.Dataset,
    position_info: pd.DataFrame,
    *,
    max_dropped_seconds_results: float = 600.0,
    max_dropped_seconds_position: float = 0.1,
    atol: float = 1e-6,
) -> tuple[xr.Dataset, pd.DataFrame, dict[str, Any]]:
    """Restrict a decode and a position frame to the timestamps they share.

    Decodes saved before ``get_position_info`` was trimmed (April 2026) carry a
    few more time bins than ``load_data`` now returns for the same epoch, so
    :func:`validate_results_position_time_alignment` — which refuses any length
    difference — rejects the pair at load. That strictness is right for the
    aggregations that attach decoder columns to position rows; what it cannot do
    is tell a head/tail trim of a handful of samples apart from two axes on
    different timebases.

    This helper resolves only the first case. It intersects the two axes on
    exact timestamps (equal within ``atol``), and keeps the result **only** if
    the shared timestamps form one contiguous run on both axes — i.e. the
    mismatch is a trim at the head and/or tail, not an interleaving — and if the
    dropped span on each side is at most ``max_dropped_seconds``. Both objects
    are then restricted to that run and handed to the strict validator, so
    drift inside the shared run still raises. Anything else raises rather than
    silently re-aligning data that does not share a timebase.

    **What the trim removed, on the mPFC corpus** (27 of 71 sessions trimmed,
    always at the tail; measured 2026-09-14): no non-local event begins
    in a dropped tail and none straddles its leading edge (0 of either, in all
    27); the stops overlapping the two largest tails -- 135.2 s in
    ``wilbur20210329_16_r8`` and 76.1 s in ``j1620210706_10_r5`` -- are
    ``in_transit``, so outside the fit cohort in any case; and across the whole
    corpus 36 dropped timestamps (72 ms) fall inside a well-anchored stop, in
    four ``mixed`` stops. Those 36 are timestamps the position frame does not
    carry -- which is why the decode bin had no partner and was dropped -- so
    they were never fittable; the trim removes the decode side of a gap the
    position frame already had.

    Parameters
    ----------
    results : xarray.Dataset
        Decoding results with a sorted ``time`` coordinate (seconds).
    position_info : pandas.DataFrame
        Time-indexed position frame (seconds).
    max_dropped_seconds_results : float
        Largest span the decode may lose. Decodes were run on the untrimmed
        epoch, so their tail past the trimmed position frame is pre/post
        trial time (0.002-135 s in practice) and is safe to drop: no
        non-local event begins in it or straddles its edge, and the stops
        it covers are ``in_transit`` or (for 36 timestamps corpus-wide)
        ``mixed`` stops whose position samples are missing anyway. See the
        paragraph above. The budget only catches a wrong file.
    max_dropped_seconds_position : float
        Largest span the position frame may lose. Position rows without a
        decode would be fit bins with no event evidence, so this stays
        tight (0.1 s, fifty 2 ms bins); a larger mismatch is a different
        timebase and must be looked at, not trimmed.
    atol : float
        Absolute tolerance (seconds) for calling two timestamps the same.

    Returns
    -------
    results : xarray.Dataset
        ``results`` restricted to the shared run.
    position_info : pandas.DataFrame
        ``position_info`` restricted to the shared run.
    counts : dict
        ``n_results_bins_dropped``, ``n_position_rows_dropped``,
        ``dropped_seconds_results``, ``dropped_seconds_position`` and
        ``dropped_at`` (``"head"`` / ``"tail"`` / ``"both"`` / ``"none"``).

    Raises
    ------
    ValueError
        If either axis is not strictly increasing, if the shared timestamps are
        empty or not contiguous on both axes, if more than
        ``max_dropped_seconds`` would be dropped from either side, or if the
        restricted pair still fails the strict alignment check.
    """
    results_time = np.asarray(results.time.values, dtype=float)
    position_time = position_info.index.to_numpy(dtype=float)

    for name, axis in (
        ("results.time", results_time),
        ("position_info", position_time),
    ):
        if axis.size == 0 or not np.all(np.diff(axis) > 0):
            raise ValueError(
                f"{name} is not strictly increasing; alignment by shared "
                "timestamps needs both axes sorted."
            )

    # Nearest-neighbor match on the sorted axes (no float-key merge): for each
    # results bin, the closest position row is at searchsorted or one before it.
    insert = np.searchsorted(position_time, results_time)
    left = np.clip(insert - 1, 0, position_time.size - 1)
    right = np.clip(insert, 0, position_time.size - 1)
    nearest = np.where(
        np.abs(position_time[left] - results_time)
        <= np.abs(position_time[right] - results_time),
        left,
        right,
    )
    is_shared_results = np.isclose(
        position_time[nearest], results_time, rtol=0.0, atol=atol
    )
    shared_results_ind = np.flatnonzero(is_shared_results)
    shared_position_ind = nearest[is_shared_results]

    def _first_divergent_time() -> float:
        unmatched = [
            axis[~mask][0]
            for axis, mask in (
                (results_time, is_shared_results),
                (
                    position_time,
                    np.isin(np.arange(position_time.size), shared_position_ind),
                ),
            )
            if (~mask).any()
        ]
        return float(min(unmatched)) if unmatched else float("nan")

    def _raise(reason: str) -> None:
        raise ValueError(
            f"{reason}: results.time ({results_time.size}) and position_info "
            f"({position_time.size}); first divergent timestamp "
            f"{_first_divergent_time():.6f}s. Refusing to align axes whose "
            "mismatch is not a head/tail trim."
        )

    if shared_results_ind.size == 0:
        _raise("results and position_info share no timestamps")
    is_contiguous_run = (
        np.all(np.diff(shared_results_ind) == 1)
        and np.all(np.diff(shared_position_ind) == 1)
        and shared_position_ind.size == np.unique(shared_position_ind).size
    )
    if not is_contiguous_run:
        _raise("shared timestamps are interleaved, not a head/tail trim")

    results_slice = slice(int(shared_results_ind[0]), int(shared_results_ind[-1]) + 1)
    position_slice = slice(
        int(shared_position_ind[0]), int(shared_position_ind[-1]) + 1
    )

    dropped: dict[str, float] = {}
    dropped_at_ends: set[str] = set()
    for side, axis, sl, budget in (
        ("results", results_time, results_slice, max_dropped_seconds_results),
        ("position", position_time, position_slice, max_dropped_seconds_position),
    ):
        head_seconds = float(axis[sl.start] - axis[0])
        tail_seconds = float(axis[-1] - axis[sl.stop - 1])
        dropped[side] = head_seconds + tail_seconds
        if sl.start > 0:
            dropped_at_ends.add("head")
        if sl.stop < axis.size:
            dropped_at_ends.add("tail")
        if dropped[side] > budget:
            raise ValueError(
                f"aligning results.time ({results_time.size}) to position_info "
                f"({position_time.size}) would drop {dropped[side]:.3f}s from "
                f"the {side} axis, over the {budget}s budget; "
                "refusing to trim this much."
            )

    aligned_results = results.isel(time=results_slice)
    aligned_position = position_info.iloc[position_slice]
    validate_results_position_time_alignment(
        np.asarray(aligned_results.time.values), aligned_position.index
    )

    if not dropped_at_ends:
        dropped_at = "none"
    elif dropped_at_ends == {"head", "tail"}:
        dropped_at = "both"
    else:
        dropped_at = dropped_at_ends.pop()

    counts = {
        "n_results_bins_dropped": int(
            results_time.size - (results_slice.stop - results_slice.start)
        ),
        "n_position_rows_dropped": int(
            position_time.size - (position_slice.stop - position_slice.start)
        ),
        "dropped_seconds_results": dropped["results"],
        "dropped_seconds_position": dropped["position"],
        "dropped_at": dropped_at,
    }
    return aligned_results, aligned_position, counts


def validate_dot_event_contract(
    dot: pd.DataFrame,
    *,
    event_cols: tuple[str, ...] = ("non_local_event", "SWR_event", "MUA_event"),
) -> dict[str, int]:
    """Check a ``data_over_time`` frame against the table-regen input contract.

    The table builders assume each ``data_over_time`` (``dot``) frame holds a
    single session, has a finite strictly increasing time axis, and labels each event as one
    contiguous run of timebins (``label > 0``; the builders take an event's span
    from the first and last labelled bin). A violation silently corrupts the
    regenerated tables — wrong bout assignment, an event split across a gap, or
    rows attributed to the wrong session — so this **fails closed**.

    Parameters
    ----------
    dot : pd.DataFrame
        Time-indexed per-bin frame (``get_data_over_time`` output) carrying
        ``animal`` / ``day`` / ``epoch`` and the event-label columns.
    event_cols : tuple of str
        Event-label columns to check for contiguity (absent columns are skipped).

    Returns
    -------
    dict[str, int]
        Violation counts (empty when clean). Always raises before returning if
        any fatal violation is present.

    Raises
    ------
    EventDefinitionError
        If the frame spans multiple sessions, has a non-finite/non-increasing
        time axis, or any event label is non-contiguous.
    """
    violations: dict[str, int] = {}
    for col in ("animal", "day", "epoch"):
        if col in dot.columns and dot[col].nunique(dropna=False) > 1:
            violations[f"multiple_{col}"] = int(dot[col].nunique(dropna=False))

    t = dot.index.to_numpy(dtype=float)
    n_nonfinite = int((~np.isfinite(t)).sum())
    if n_nonfinite:
        violations["nonfinite_time"] = n_nonfinite
    # Keep the established ``unsorted_time`` diagnostic key, but include equal
    # adjacent timestamps: duplicate samples are just as unsafe as reversals
    # because builders would double-count their content and event rows.
    n_unsorted = int((np.diff(t) <= 0).sum()) if t.size > 1 else 0
    if n_unsorted:
        violations["unsorted_time"] = n_unsorted

    for col in event_cols:
        if col not in dot.columns:
            continue
        lab = dot[col].to_numpy()
        # One pass per column: group positions of labelled bins by label. The
        # positions are ascending (flatnonzero), so per label first()/last() are
        # its min/max bin index; a contiguous run has last - first + 1 == count.
        pos = np.flatnonzero(lab > 0)
        if pos.size:
            grp = pd.Series(pos).groupby(lab[pos])
            span = grp.last() - grp.first() + 1
            noncontig = int((span.to_numpy() != grp.size().to_numpy()).sum())
        else:
            noncontig = 0
        if noncontig:
            violations[f"noncontiguous_{col}"] = noncontig

    if violations:
        raise EventDefinitionError(
            f"validate_dot_event_contract: input-contract violations {violations}"
        )
    return violations


def get_data_over_time(
    data: SessionData,
    results: xr.Dataset,
    env: Environment,
    nwb_file_name: str,
    epoch_name: str,
    brain_area: str = "HPC",
    sampling_frequency: int = SAMPLING_FREQUENCY,
    non_local_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
) -> tuple[pd.DataFrame, xr.DataArray]:
    """Aggregate all analysis data into time-indexed DataFrame.

    Combines decoding results, event detection, posterior statistics,
    and behavioral data into a single DataFrame aligned by time.

    The frame's ``SWR_event`` labels and its SWR counts (``n_SWR``,
    ``n_non_local_SWR``, ``n_non_local_no_SWR``, ``n_SWR_MUA``,
    ``n_SWR_no_MUA``) come from ``data["ripple_times"]``, the stored Spyglass
    ``RippleTimesV1`` ripples, not from the paper's pinned SWR inventory. Read
    through ``analysed_time.with_swr_source`` (or, for a saved file,
    ``analysed_time.read_dot_with_swr_source``), ``SWR_event`` is replaced by
    the pinned inventory's labels and every decode count column is dropped.
    Callers must read the frame that way; its own ``SWR_event`` and ``n_SWR``
    are provenance only.

    Parameters
    ----------
    data : SessionData
        Data dictionary from load_data() containing position, spikes, etc.
    results : xr.Dataset
        Decoding results
    env : Environment
        Environment configuration from detector
    nwb_file_name : str
        Name of the NWB file (e.g., "chimi20200212_.nwb")
    epoch_name : str
        Name of the epoch (e.g., "02_r1")
    brain_area : str, optional
        Brain area for analysis, by default "HPC"
    sampling_frequency : int, optional
        Samples per second, by default ``SAMPLING_FREQUENCY`` (500)
    non_local_threshold : float, optional
        Probability threshold for non-local classification, by default
        ``DEFAULT_NON_LOCAL_THRESHOLD`` (0.8)

    Returns
    -------
    data_over_time : pd.DataFrame
        Time-indexed DataFrame containing:
        - Event labels (non_local_event, SWR_event, MUA_event)
        - State probabilities (Local_prob, No-Spike_prob, etc.)
        - Posterior statistics (map_estimate, map_speed, step_distance,
          ahead_behind_distance, hpd)
        - Neural signals (theta_phase, theta_power_zscore, SWR_zscore,
          MUA_zscore)
        - Per-session run-speed baseline (run_speed_baseline, cm/s)
        - Event counts (n_non_local, n_SWR, n_MUA, etc.)
        - Session metadata (animal, day, epoch, nwb_file_name)
        - Position and behavioral variables (from position_info)
    posterior : xr.DataArray
        Conditional non-local posterior distribution

    """
    validate_required_session_data_keys(
        data,
        required_keys=[
            "position_info",
            "ripple_times",
            "hse_times",
            "theta_phase",
            "theta_power",
            "zscored_ripple_consensus_trace",
            "multiunit_rate_zscore",
        ],
    )

    # Decoder values and position samples are paired row-for-row. Validate that
    # relationship, then use the position axis for detection, speed gating,
    # label writing, and the final data-over-time index. One canonical axis
    # prevents sub-tolerance drift from changing endpoint membership between
    # those operations.
    validate_results_position_time_alignment(
        np.asarray(results.time.values), data["position_info"].index
    )
    position_time = data["position_info"].index
    head_speed = data["position_info"].head_speed.to_numpy()

    non_local_prob = nonlocal_state_probability(results.acausal_state_probabilities)

    posterior = get_conditional_non_local_posterior(results)

    non_local_times, non_local_labels = detect_nonlocal_events(
        non_local_prob, head_speed, position_time, non_local_threshold
    )
    swr_times, swr_labels = build_speed_eligible_event_cohort(
        data["ripple_times"],
        speed=head_speed,
        time=position_time,
        event_col="SWR_event",
    )
    mua_times, mua_labels = build_speed_eligible_event_cohort(
        data["hse_times"][brain_area],
        speed=head_speed,
        time=position_time,
        event_col="MUA_event",
    )

    non_local_swr_times, non_local_no_swr_times = get_overlap_times(
        non_local_times, swr_times
    )
    non_local_mua_times, non_local_no_mua_times = get_overlap_times(
        non_local_times, mua_times
    )

    swr_mua_times, swr_no_mua_times = get_overlap_times(swr_times, mua_times)
    (
        map_estimate,
        map_speed,
        ahead_behind_distance,
        hpd,
        max_coverage,
        max_distance,
        max_linear_position,
        posterior,
    ) = get_posterior_stats(
        data, env, sampling_frequency=sampling_frequency, posterior=posterior
    )
    # Raw per-timebin track-graph displacement of the MAP position (cm). Kept
    # unsmoothed so downstream summaries can sum it (path length) or take
    # robust order statistics (jumpiness); ``map_speed`` is the smoothed rate.
    step_distance = get_map_step_distance(
        posterior.values,
        env.track_graph_with_bin_centers_edges_,
        np.asarray(env.place_bin_centers_nodes_df_.node_id),
    )
    # Per-session "typical run speed" baseline (cm/s): a robust, threshold-free
    # percentile of head speed over every bin, intended as a per-session speed
    # normalizer for downstream replay-speed/compression measures. Constant
    # across the session. The read through the SWR source takes it again over
    # the bins outside the artifact mask (``analysed_time.with_swr_source``).
    run_speed_baseline = epoch_run_speed_baseline(
        data["position_info"].head_speed.to_numpy()
    )
    animal, day = parse_nwb_filename(nwb_file_name)

    def _nearest(series):
        # Return a positional ndarray. ``reindex`` returns a Series
        # indexed by ``results.time``; embedding that Series into the
        # DataFrame below would make pandas re-align it against
        # ``position_info.index``, producing NaNs whenever the two
        # axes differ by even epsilon despite identical lengths. The
        # length guard above ensures the two are 1-to-1, so a positional
        # ndarray is safe and matches the bare-numpy columns used here.
        return series.reindex(
            results.time, method="nearest", tolerance=0.004
        ).to_numpy()

    data_over_time = pd.DataFrame(
        {
            "non_local_event": non_local_labels.to_numpy(),
            "SWR_event": swr_labels.to_numpy(),
            "MUA_event": mua_labels.to_numpy(),
            "Local_prob": results.sel(
                states="Local"
            ).acausal_state_probabilities.squeeze(),
            "No-Spike_prob": results.sel(
                states="No-Spike"
            ).acausal_state_probabilities.squeeze(),
            "Non-Local_Continuous_prob": results.sel(
                states="Non-Local Continuous"
            ).acausal_state_probabilities.squeeze(),
            "Non-Local_Fragmented_prob": results.sel(
                states="Non-Local Fragmented"
            ).acausal_state_probabilities.squeeze(),
            "map_estimate": map_estimate.squeeze(),
            "map_speed": map_speed.squeeze(),
            "step_distance": step_distance.squeeze(),
            "ahead_behind_distance": ahead_behind_distance.squeeze(),
            "hpd": hpd.squeeze(),
            "run_speed_baseline": run_speed_baseline,
            "max_coverage": max_coverage,
            "max_distance": max_distance,
            "max_linear_position": max_linear_position,
            "theta_phase": _nearest(data["theta_phase"]).squeeze(),
            "theta_power_zscore": theta_power_zscore(
                _nearest(data["theta_power"].squeeze()).squeeze()
            ),
            "SWR_zscore": _nearest(data["zscored_ripple_consensus_trace"]).squeeze(),
            "MUA_zscore": _nearest(data["multiunit_rate_zscore"][brain_area].squeeze()),
            "n_non_local": len(non_local_times),
            "n_SWR": len(swr_times),
            "n_MUA": len(mua_times),
            "n_non_local_SWR": len(non_local_swr_times),
            "n_non_local_no_SWR": len(non_local_no_swr_times),
            "n_non_local_MUA": len(non_local_mua_times),
            "n_non_local_no_MUA": len(non_local_no_mua_times),
            "n_SWR_MUA": len(swr_mua_times),
            "n_SWR_no_MUA": len(swr_no_mua_times),
            "animal": animal,
            "day": day,
            "epoch": epoch_name,
            "nwb_file_name": nwb_file_name,
        },
        index=data["position_info"].index,
    )

    return pd.concat((data_over_time, data["position_info"]), axis=1), posterior


def load_data_over_time(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate the analysed epochs' NonLocal processed data.

    This loads data from the 4-state NonLocal detector only. For
    ContFrag (2-state) data, use
    :func:`continuum_swr_replay.analysis.contfrag.load_contfrag_data_over_time`.

    The epochs are the epoch register's analysed ones
    (``epoch_register.analysed_decode_files``), never a glob of `data_dir`, so a
    pending or excluded epoch's decode there is not read. Each is read through
    the SWR source (``analysed_time.read_dot_with_swr_source``): ``SWR_event`` is
    the direct Kay inventory's labels, not the decode's stored ripples.

    Parameters
    ----------
    data_dir : Path
        Directory containing the ``{epoch_id}_80_data_over_time.pkl`` files.

    Returns
    -------
    pd.DataFrame
        Concatenated data with derived Non-Local_prob and distance columns,
        and the source's ``SWR_event`` and ``swr_observed``.

    Raises
    ------
    FileNotFoundError
        If an analysed epoch's file is not in `data_dir`.
    SwrSourceError
        If the SWR source cannot serve an analysed epoch.

    """
    files = analysed_decode_files(data_dir, data_over_time_suffix(), require_all=True)
    data_over_time = pd.concat([read_dot_with_swr_source(p) for p in files])
    data_over_time["Non-Local_prob"] = nonlocal_probability(data_over_time)
    data_over_time["distance"] = np.abs(data_over_time["ahead_behind_distance"])
    return data_over_time


def get_event_types(
    data_over_time: pd.DataFrame,
    event_col: str = "non_local_event",
) -> dict:
    """Classify events by overlap with other detectors (SWR, MUA).

    Works for any primary event column: ``non_local_event`` (model-
    detected), ``SWR_event`` (ripple-detected), or ``MUA_event``
    (high-synchrony-detected). The two "other" detector columns are
    inferred from the remaining pair among {SWR_event, MUA_event,
    non_local_event}.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Aggregated data with ``event_col``, ``SWR_event``,
        ``MUA_event``, and ``head_speed`` columns, read through the SWR
        source (``load_data_over_time``): its labels already hold only
        events the whole-event speed rule admits, and its Non-Local labels
        meet the probability contract.
    event_col : str, optional
        Primary event column, by default ``"non_local_event"``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys use ``event_col`` as prefix, e.g. for
        ``event_col="SWR_event"``:
        ``SWR_event_immobile``, ``SWR_event_immobile_overlap_MUA_event``,
        etc. Each value is a row subset of the input. ``_immobile`` means
        the canonical whole-event rule (every sampled speed finite and
        strictly below the threshold), not the retired endpoint rule; the
        read applied it.

    """
    all_event_cols = {"non_local_event", "SWR_event", "MUA_event"}
    if event_col not in all_event_cols:
        raise ValueError(
            f"event_col must be one of {sorted(all_event_cols)}; got {event_col!r}"
        )
    other_cols = sorted(all_event_cols - {event_col})

    immobile = data_over_time.loc[data_over_time[event_col] > 0]

    result: dict[str, pd.DataFrame] = {
        f"{event_col}_immobile": immobile,
    }

    # Canonically eligible events for each "other" detector.
    for oc in other_cols:
        if oc in data_over_time.columns:
            result[f"{oc}_events"] = data_over_time.loc[data_over_time[oc] > 0]

    # Overlap / no-overlap with each "other" detector. Use strict
    # positive-duration interval overlap rather than per-timebin label
    # presence; otherwise events whose intervals only touch at a
    # boundary timestamp (where ``get_labels`` writes both labels
    # inclusively) get classified as overlapping. Matches the strict
    # semantics used by ``classify_swr_overlap`` /
    # ``get_overlap_times``.
    def _strict_overlap_mask(df: pd.DataFrame, primary: str, other: str) -> np.ndarray:
        keys = event_strict_overlap_keys(data_over_time, primary, other)
        if not keys:
            return np.zeros(len(df), dtype=bool)
        return np.array(
            [
                (a, d, e, p) in keys
                for a, d, e, p in zip(
                    df["animal"].to_numpy(),
                    df["day"].to_numpy(),
                    df["epoch"].to_numpy(),
                    df[primary].to_numpy(),
                    strict=True,
                )
            ]
        )

    for oc in other_cols:
        if oc not in data_over_time.columns:
            continue
        overlap_mask = _strict_overlap_mask(immobile, event_col, oc)
        result[f"{event_col}_immobile_overlap_{oc}"] = immobile.loc[overlap_mask]
        result[f"{event_col}_immobile_no_overlap_{oc}"] = immobile.loc[~overlap_mask]

    # No overlap with ANY other detector
    if all(oc in data_over_time.columns for oc in other_cols):
        any_overlap = np.zeros(len(immobile), dtype=bool)
        for oc in other_cols:
            any_overlap |= _strict_overlap_mask(immobile, event_col, oc)
        result[f"{event_col}_immobile_no_overlap"] = immobile.loc[~any_overlap]

    # Other-detector events that do NOT overlap the primary
    for oc in other_cols:
        if oc not in data_over_time.columns:
            continue
        oc_events = data_over_time.loc[data_over_time[oc] > 0]
        if len(oc_events) == 0:
            result[f"{oc}_events_no_{event_col}"] = oc_events
            continue
        overlap_keys = event_strict_overlap_keys(data_over_time, oc, event_col)
        if not overlap_keys:
            result[f"{oc}_events_no_{event_col}"] = oc_events
            continue
        oc_keys = list(
            zip(
                oc_events["animal"].to_numpy(),
                oc_events["day"].to_numpy(),
                oc_events["epoch"].to_numpy(),
                oc_events[oc].to_numpy(),
                strict=True,
            )
        )
        no_overlap_mask = np.array([k not in overlap_keys for k in oc_keys])
        result[f"{oc}_events_no_{event_col}"] = oc_events.loc[no_overlap_mask]

    return result


def compute_swr_nl_event_counts(
    data_over_time: pd.DataFrame,
    event_col_a: str = "non_local_event",
    event_col_b: str = "SWR_event",
    *,
    immobile_col_a: bool = True,
) -> dict[str, int]:
    """Event-level totals and overlap for two event detectors.

    Designed as the data source for a 2-circle Venn diagram. Reports
    four numbers: total count for each detector and how many of each
    overlap with the other.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Aggregated data with both event columns, ``head_speed``, and
        session group columns, read through the SWR source
        (``load_data_over_time``), which holds every event label to the
        whole-event speed rule and the Non-Local probability contract.
    event_col_a : str, optional
        First event column, by default ``"non_local_event"``.
    event_col_b : str, optional
        Second event column, by default ``"SWR_event"``.
    immobile_col_a : bool, optional
        Deprecated compatibility argument. It must remain ``True`` because all
        event sources now share the canonical whole-event speed criterion.

    Raises
    ------
    ValueError
        When ``immobile_col_a`` is ``False``.

    Returns
    -------
    dict[str, int]
        Keys: ``n_a``, ``n_a_with_b``, ``n_b``, ``n_b_with_a``. Both totals
        are speed-eligible counts, not raw detector counts.

    """
    if not immobile_col_a:
        raise ValueError(
            "immobile_col_a=False is incompatible with the unified event "
            "definition; Non-Local, SWR, and MUA events all require whole-event "
            "speed < 4 cm/s"
        )

    sess_cols = ["animal", "day", "epoch"]

    # Strict positive-duration interval overlap on per-event bounds
    # derived from the timebin labels. The previous form used
    # ``(g[other] > 0).any()`` over timebins, which counted
    # boundary-touch contact (a single shared timestamp from the
    # inclusive ``get_labels`` writer) as overlap — diverging from
    # the strict semantics enforced by ``classify_swr_overlap`` and
    # ``get_overlap_times``.
    #
    # Both totals and both overlap directions use the same globally eligible
    # populations; an ineligible event cannot reappear as the secondary
    # detector in the reverse overlap calculation.
    a_active = data_over_time.loc[data_over_time[event_col_a] > 0]
    n_a = (
        int(a_active.groupby([*sess_cols, event_col_a], sort=False).ngroups)
        if len(a_active)
        else 0
    )
    b_active = data_over_time.loc[data_over_time[event_col_b] > 0]
    n_b = (
        int(b_active.groupby([*sess_cols, event_col_b], sort=False).ngroups)
        if len(b_active)
        else 0
    )

    if n_a == 0 or n_b == 0:
        return {"n_a": n_a, "n_a_with_b": 0, "n_b": n_b, "n_b_with_a": 0}

    a_with_b_keys = event_strict_overlap_keys(data_over_time, event_col_a, event_col_b)
    n_a_with_b = len(a_with_b_keys)

    b_with_a_keys = event_strict_overlap_keys(data_over_time, event_col_b, event_col_a)
    n_b_with_a = len(b_with_a_keys)

    return {
        "n_a": n_a,
        "n_a_with_b": n_a_with_b,
        "n_b": n_b,
        "n_b_with_a": n_b_with_a,
    }
