"""Aggregation of analysis results over time.

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
    - SWR_event: Sharp-wave ripple event number
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
    - max_coverage: Maximum HPD spatial coverage
    - max_distance: Maximum distance in posterior
    - max_linear_position: Maximum linear position

Neural signals (z-scored or phase):
    - theta_phase: Theta oscillation phase (radians)
    - theta_power_zscore: Z-scored theta power
    - SWR_zscore: Z-scored ripple consensus trace
    - MUA_zscore: Z-scored multiunit firing rate

Event counts (session-level):
    - n_non_local, n_SWR, n_MUA: Total event counts
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
- Events exceeding max speed threshold are filtered out
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
continuum_swr_replay.analysis.hippocampus.posterior_analysis : Posterior statistics
continuum_swr_replay.analysis.hippocampus.event_detection : Event detection functions
continuum_swr_replay.data_loaders.bandit_task : Session data loading

Examples
--------
>>> from continuum_swr_replay.analysis.hippocampus import load_results, get_data_over_time
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

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from non_local_detector.environment import Environment
from scipy.stats import zscore

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.data_loaders import get_labels
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename
from continuum_swr_replay.parameters import MAX_SPEED_THRESHOLD_CM_PER_SEC

from ...types import SessionData
from .event_analysis import get_immobile_events
from .event_detection import get_confident_times, get_overlap_times
from .posterior_analysis import get_conditional_non_local_posterior, get_posterior_stats


def get_data_over_time(
    data: SessionData,
    results: xr.Dataset,
    env: Environment,
    nwb_file_name: str,
    epoch_name: str,
    brain_area: str = "HPC",
    sampling_frequency: int = 500,
    non_local_threshold: float = 0.8,
) -> tuple[pd.DataFrame, xr.DataArray]:
    """Aggregate all analysis data into time-indexed DataFrame.

    Combines decoding results, event detection, posterior statistics,
    and behavioral data into a single DataFrame aligned by time.

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
        Samples per second, by default 500
    non_local_threshold : float, optional
        Probability threshold for non-local classification, by default 0.8

    Returns
    -------
    data_over_time : pd.DataFrame
        Time-indexed DataFrame containing:
        - Event labels (non_local_event, SWR_event, MUA_event)
        - State probabilities (Local_prob, No-Spike_prob, etc.)
        - Posterior statistics (map_estimate, map_speed, ahead_behind_distance, hpd)
        - Neural signals (theta_phase, theta_power_zscore, SWR_zscore, MUA_zscore)
        - Event counts (n_non_local, n_SWR, n_MUA, etc.)
        - Session metadata (animal, day, epoch, nwb_file_name)
        - Position and behavioral variables (from position_info)
    posterior : xr.DataArray
        Conditional non-local posterior distribution

    """
    # Validate required data keys
    required_keys = [
        "position_info", "ripple_times", "hse_times",
        "theta_phase", "theta_power",
        "zscored_ripple_consensus_trace", "multiunit_rate_zscore",
    ]
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise KeyError(
            f"Missing required data keys: {missing_keys}. "
            f"Ensure load_data() was called with DataLoadingPreset.DECODING or equivalent."
        )

    non_local_prob = results.acausal_state_probabilities.sel(
        states="Non-Local Continuous"
    ) + results.acausal_state_probabilities.sel(states="Non-Local Fragmented")

    posterior = get_conditional_non_local_posterior(results)

    non_local_times = get_confident_times(
        non_local_prob,
        data["position_info"].head_speed.to_numpy(),
        data["position_info"].index.to_numpy(),
        non_local_threshold,
    )
    non_local_times = non_local_times[
        non_local_times.max_speed < MAX_SPEED_THRESHOLD_CM_PER_SEC
    ]
    non_local_labels = get_labels(non_local_times, results.time)
    swr_labels = get_labels(data["ripple_times"], results.time)
    mua_labels = get_labels(data["hse_times"][brain_area], results.time)

    non_local_swr_times, non_local_no_swr_times = get_overlap_times(
        non_local_times, data["ripple_times"], results
    )
    non_local_mua_times, non_local_no_mua_times = get_overlap_times(
        non_local_times, data["hse_times"][brain_area], results
    )

    swr_mua_times, swr_no_mua_times = get_overlap_times(
        data["ripple_times"], data["hse_times"][brain_area], results
    )
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
    animal, day = parse_nwb_filename(nwb_file_name)

    def _nearest(series):
        # Return a positional ndarray. ``reindex`` returns a Series
        # indexed by ``results.time``; embedding that Series into the
        # DataFrame below would make pandas re-align it against
        # ``position_info.index``, producing NaNs whenever the two
        # axes differ by even epsilon despite identical lengths. The
        # length guard above ensures the two are 1-to-1, so a positional
        # ndarray is safe and matches the bare-numpy columns used here.
        return (
            series.reindex(results.time, method="nearest", tolerance=0.004).to_numpy()
        )

    # Build with the position_info time index so axis=1 concat aligns by
    # time instead of falling back to a default RangeIndex (which would
    # produce a 2N-row frame full of NaNs against the time-indexed
    # position frame). Length is necessary but not sufficient — same-
    # length but time-shifted axes would silently attach event labels
    # to the wrong rows. Mirror the stricter drift check used in
    # ``analysis.replay._helpers._align_posterior_to_dot``.
    if len(results.time) != len(data["position_info"]):
        raise ValueError(
            f"results.time ({len(results.time)}) and position_info "
            f"({len(data['position_info'])}) have different lengths; "
            "they must be aligned 1-to-1."
        )
    drift = float(
        np.abs(
            np.asarray(results.time.values, dtype=float)
            - data["position_info"].index.to_numpy(dtype=float)
        ).max()
    )
    if drift > 1e-6:
        raise ValueError(
            f"results.time drifts from position_info.index by up to "
            f"{drift:.3g}s — same-length axes do not imply same time "
            "origin; refusing to attach decoder outputs to position "
            "rows that don't share their timebase."
        )

    data_over_time = pd.DataFrame(
        {
            "non_local_event": non_local_labels.squeeze(),
            "SWR_event": swr_labels.squeeze(),
            "MUA_event": mua_labels.squeeze(),
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
            "ahead_behind_distance": ahead_behind_distance.squeeze(),
            "hpd": hpd.squeeze(),
            "max_coverage": max_coverage,
            "max_distance": max_distance,
            "max_linear_position": max_linear_position,
            "theta_phase": _nearest(data["theta_phase"]).squeeze(),
            "theta_power_zscore": zscore(
                _nearest(data["theta_power"].squeeze()).squeeze()
            ),
            "SWR_zscore": _nearest(data["zscored_ripple_consensus_trace"]).squeeze(),
            "MUA_zscore": _nearest(
                data["multiunit_rate_zscore"][brain_area].squeeze()
            ),
            "n_non_local": len(non_local_times),
            "n_SWR": len(data["ripple_times"]),
            "n_MUA": len(data["hse_times"][brain_area]),
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
    """Load and concatenate all NonLocal processed session data.

    This loads data from the 4-state NonLocal detector only. For
    ContFrag (2-state) data, use
    :func:`continuum_swr_replay.analysis.contfrag.load_contfrag_data_over_time`.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``*_HPC_80_data_over_time.pkl`` files.

    Returns
    -------
    pd.DataFrame
        Concatenated data with derived Non-Local_prob and distance columns.

    """
    # Sort and check explicitly so an empty directory raises a
    # targeted FileNotFoundError instead of pandas' generic
    # "No objects to concatenate" — matches the contfrag loader.
    files = sorted(data_dir.glob("*_HPC_80_data_over_time.pkl"))
    if not files:
        raise FileNotFoundError(
            f"No *_HPC_80_data_over_time.pkl files found in {data_dir}"
        )
    data_over_time = pd.concat([pd.read_pickle(p) for p in files])
    data_over_time["Non-Local_prob"] = (
        data_over_time["Non-Local_Continuous_prob"]
        + data_over_time["Non-Local_Fragmented_prob"]
    )
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
        ``MUA_event``, and ``head_speed`` columns.
    event_col : str, optional
        Primary event column, by default ``"non_local_event"``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys use ``event_col`` as prefix, e.g. for
        ``event_col="SWR_event"``:
        ``SWR_event_immobile``, ``SWR_event_immobile_overlap_MUA_event``,
        etc. Each value is a filtered subset of the input DataFrame.

    """
    all_event_cols = {"non_local_event", "SWR_event", "MUA_event"}
    if event_col not in all_event_cols:
        raise ValueError(
            f"event_col must be one of {sorted(all_event_cols)}; got {event_col!r}"
        )
    other_cols = sorted(all_event_cols - {event_col})

    immobile = get_immobile_events(
        data_over_time,
        event_col=event_col,
        speed_col="head_speed",
        speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
        method="endpoints",
    )

    group_keys = ["animal", "day", "epoch", event_col]

    result: dict[str, pd.DataFrame] = {
        f"{event_col}_immobile": immobile,
    }

    # Raw events for each "other" detector
    for oc in other_cols:
        if oc in data_over_time.columns:
            result[f"{oc}_events"] = data_over_time.loc[
                data_over_time[oc] > 0
            ]

    # Overlap / no-overlap with each "other" detector
    for oc in other_cols:
        if oc not in data_over_time.columns:
            continue
        result[f"{event_col}_immobile_overlap_{oc}"] = immobile.groupby(
            group_keys
        ).filter(lambda g, c=oc: (g[c] > 0).any())
        result[f"{event_col}_immobile_no_overlap_{oc}"] = immobile.groupby(
            group_keys
        ).filter(lambda g, c=oc: (g[c] == 0).all())

    # No overlap with ANY other detector
    if all(oc in data_over_time.columns for oc in other_cols):
        result[f"{event_col}_immobile_no_overlap"] = immobile.groupby(
            group_keys
        ).filter(
            lambda g: all((g[oc] == 0).all() for oc in other_cols)
        )

    # Other-detector events that do NOT overlap the primary
    for oc in other_cols:
        if oc not in data_over_time.columns:
            continue
        oc_events = data_over_time.loc[data_over_time[oc] > 0]
        result[f"{oc}_events_no_{event_col}"] = oc_events.groupby(
            ["animal", "day", "epoch", oc]
        ).filter(lambda g, ec=event_col: (g[ec] == 0).all())

    return result


def compute_swr_overlap_class_counts(
    data_over_time: pd.DataFrame,
    event_col: str = "non_local_event",
    overlap_col: str = "SWR_event",
) -> dict[str, int]:
    """Count immobile events by SWR-overlap class.

    Per-session: derives non-local-event and SWR intervals from the
    timebin labels in ``data_over_time`` and delegates to
    :func:`classify_swr_overlap` for per-event classification. Counts
    are aggregated across all sessions.

    Uses the event-level definition of "contained" (does any single
    SWR fully span the event) so the counts match the example-panel
    classification in Figure 2. The alternative timebin-level
    definition (does every timebin overlap some SWR) labels events
    tiled by two adjacent SWRs as ``swr_contained``, which the
    event-level rule calls ``swr_partial``.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Aggregated data containing ``event_col``, ``overlap_col``,
        ``head_speed``, ``animal``, ``day``, and ``epoch`` columns.
        Time must be the DataFrame index (production) or present as a
        column named ``"time"`` (tests).
    event_col : str, optional
        Primary event column, by default ``"non_local_event"``.
    overlap_col : str, optional
        Column to check overlap against, by default ``"SWR_event"``.

    Returns
    -------
    dict[str, int]
        Mapping ``{"swr_contained": n, "swr_partial": n,
        "non_swr": n, "total": n}``.

    """
    from .event_detection import classify_swr_overlap

    immobile = get_immobile_events(
        data_over_time,
        event_col=event_col,
        speed_col="head_speed",
        speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
        method="endpoints",
    )

    # Normalize time to a column called "time". Production has time as
    # the DataFrame index; tests sometimes have it as a column.
    if "time" not in immobile.columns:
        immobile = immobile.copy()
        immobile["time"] = immobile.index

    counts = {"swr_contained": 0, "swr_partial": 0, "non_swr": 0}

    for _session_key, session_df in immobile.groupby(
        ["animal", "day", "epoch"], sort=False
    ):
        event_intervals = _intervals_from_labels(session_df, event_col)
        if len(event_intervals) == 0:
            continue
        swr_intervals = _intervals_from_labels(session_df, overlap_col)

        classified = classify_swr_overlap(event_intervals, swr_intervals)
        session_counts = classified["overlap_class"].value_counts()
        for cls in counts:
            counts[cls] += int(session_counts.get(cls, 0))

    return {**counts, "total": sum(counts.values())}


def _intervals_from_labels(session_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Per-label (start_time, end_time) table from a timebin-labeled frame.

    Expects a ``time`` column; positive values in ``label_col`` mark
    timebins belonging to distinct events, with 0 treated as background.
    """
    return (
        session_df.loc[session_df[label_col] > 0]
        .groupby(label_col)["time"]
        .agg(start_time="min", end_time="max")
        .reset_index(drop=True)
    )


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
        session group columns.
    event_col_a : str, optional
        First event column (immobility-filtered if ``immobile_col_a``),
        by default ``"non_local_event"``.
    event_col_b : str, optional
        Second event column (not speed-filtered), by default ``"SWR_event"``.
    immobile_col_a : bool, optional
        Whether to apply immobility filter to event_col_a, by default True.
        Set to True for model-detected events (non_local_event) where
        immobility filtering is standard. Set to False for externally-
        detected events (SWR_event, MUA_event) that are not speed-gated.

    Returns
    -------
    dict[str, int]
        Keys: ``n_a``, ``n_a_with_b``, ``n_b``, ``n_b_with_a``.

    """
    # Event A (optionally immobility-filtered)
    if immobile_col_a:
        a_data = get_immobile_events(
            data_over_time,
            event_col=event_col_a,
            speed_col="head_speed",
            speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
            method="endpoints",
        )
    else:
        a_data = data_over_time.loc[data_over_time[event_col_a] > 0]

    a_groups = a_data.loc[a_data[event_col_a] > 0].groupby(
        ["animal", "day", "epoch", event_col_a],
    )
    a_with_b = a_groups.apply(lambda g: (g[event_col_b] > 0).any())
    n_a = len(a_with_b)
    n_a_with_b = int(a_with_b.sum())

    # Event B (not speed-filtered)
    b_groups = data_over_time.loc[data_over_time[event_col_b] > 0].groupby(
        ["animal", "day", "epoch", event_col_b],
    )
    b_with_a = b_groups.apply(lambda g: (g[event_col_a] > 0).any())
    n_b = len(b_with_a)
    n_b_with_a = int(b_with_a.sum())

    return {
        "n_a": n_a,
        "n_a_with_b": n_a_with_b,
        "n_b": n_b,
        "n_b_with_a": n_b_with_a,
    }
