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
- Missing values are forward-filled to maintain continuity
- Z-scoring is applied to relevant continuous variables
- Event overlap detection uses interval-based logic
- All data is aligned to the same temporal resolution (default 500 Hz)

Notes
-----
- All data is time-indexed in seconds
- Non-local threshold typically set to 0.8 for classification
- Ahead/behind distance uses graph shortest path (can be negative)
- MAP speed is smoothed before computation to reduce noise
- Missing behavioral data is handled gracefully with forward-fill

See Also
--------
continuum_swr_replay.analysis.hippocampus.posterior_analysis : Posterior statistics
continuum_swr_replay.analysis.hippocampus.event_detection : Event detection functions
continuum_swr_replay.data_loaders.bandit_task : Session data loading

Examples
--------
>>> from continuum_swr_replay.analysis.hippocampus import load_results, get_data_over_time
>>> results, detector, env = load_results("chimi20200212_02_r1_HPC")
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
            "theta_phase": data["theta_phase"]
            .reindex(results.time, method="nearest", tolerance=0.004)
            .squeeze(),
            "theta_power_zscore": zscore(
                data["theta_power"]
                .squeeze()
                .reindex(results.time, method="nearest", tolerance=0.004)
                .squeeze()
            ),
            "SWR_zscore": data["zscored_ripple_consensus_trace"]
            .reindex(results.time, method="nearest", tolerance=0.004)
            .squeeze(),
            "MUA_zscore": data["multiunit_rate_zscore"][brain_area].squeeze(),
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
        }
    )

    return pd.concat((data_over_time, data["position_info"]), axis=1), posterior


def load_data_over_time(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate all processed session data.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``*_HPC_80_data_over_time.pkl`` files.

    Returns
    -------
    pd.DataFrame
        Concatenated data with derived Non-Local_prob and distance columns.

    """
    data_path = data_dir.glob("*_HPC_80_data_over_time.pkl")
    data_over_time = pd.concat([pd.read_pickle(p) for p in data_path])
    data_over_time["Non-Local_prob"] = (
        data_over_time["Non-Local_Continuous_prob"]
        + data_over_time["Non-Local_Fragmented_prob"]
    )
    data_over_time["distance"] = np.abs(data_over_time["ahead_behind_distance"])
    return data_over_time


def get_event_types(data_over_time: pd.DataFrame) -> dict:
    """Classify non-local events by overlap with SWR and MUA.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Aggregated data with ``non_local_event``, ``SWR_event``,
        ``MUA_event``, and ``head_speed`` columns.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ``non_local_immobile``, ``swr_events``, ``mua_events``,
        ``non_local_immobile_overlap_swr``, ``non_local_immobile_overlap_mua``,
        ``non_local_immobile_no_overlap``, ``non_local_immobile_no_overlap_swr``,
        ``non_local_immobile_no_overlap_mua``, ``swr_events_no_non_local``,
        ``mua_events_no_non_local``. Each value is a filtered subset of
        the input DataFrame.

    """
    non_local_immobile = get_immobile_events(
        data_over_time,
        event_col="non_local_event",
        speed_col="head_speed",
        speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
        method="endpoints",
    )

    swr_events = data_over_time.loc[data_over_time["SWR_event"] > 0]
    mua_events = data_over_time.loc[data_over_time["MUA_event"] > 0]

    non_local_immobile_overlap_swr = non_local_immobile.groupby(
        ["animal", "day", "epoch", "non_local_event"]
    ).filter(lambda g: (g["SWR_event"] > 0).any())
    non_local_immobile_overlap_mua = non_local_immobile.groupby(
        ["animal", "day", "epoch", "non_local_event"]
    ).filter(lambda g: (g["MUA_event"] > 0).any())

    non_local_immobile_no_overlap = non_local_immobile.groupby(
        ["animal", "day", "epoch", "non_local_event"]
    ).filter(lambda g: (g["SWR_event"] == 0).all() & (g["MUA_event"] == 0).all())
    non_local_immobile_no_overlap_swr = non_local_immobile.groupby(
        ["animal", "day", "epoch", "non_local_event"]
    ).filter(lambda g: (g["SWR_event"] == 0).all())
    non_local_immobile_no_overlap_mua = non_local_immobile.groupby(
        ["animal", "day", "epoch", "non_local_event"]
    ).filter(lambda g: (g["MUA_event"] == 0).all())
    swr_events_no_non_local = swr_events.groupby(
        ["animal", "day", "epoch", "SWR_event"]
    ).filter(lambda g: (g["non_local_event"] == 0).all())
    mua_events_no_non_local = mua_events.groupby(
        ["animal", "day", "epoch", "MUA_event"]
    ).filter(lambda g: (g["non_local_event"] == 0).all())

    return {
        "non_local_immobile": non_local_immobile,
        "swr_events": swr_events,
        "mua_events": mua_events,
        "non_local_immobile_overlap_swr": non_local_immobile_overlap_swr,
        "non_local_immobile_overlap_mua": non_local_immobile_overlap_mua,
        "non_local_immobile_no_overlap": non_local_immobile_no_overlap,
        "non_local_immobile_no_overlap_swr": non_local_immobile_no_overlap_swr,
        "non_local_immobile_no_overlap_mua": non_local_immobile_no_overlap_mua,
        "swr_events_no_non_local": swr_events_no_non_local,
        "mua_events_no_non_local": mua_events_no_non_local,
    }


def compute_swr_overlap_class_counts(
    data_over_time: pd.DataFrame,
) -> dict[str, int]:
    """Count immobile non-local events by SWR-overlap geometry.

    Uses a timepoint-level rule equivalent to the interval-based
    :func:`classify_swr_overlap`: if every timepoint of a non-local
    event has ``SWR_event > 0`` the event is "contained", if some but
    not all do it "extends" beyond the SWR, and if none do it is
    "none".

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Aggregated data containing ``non_local_event``, ``SWR_event``,
        ``head_speed``, ``animal``, ``day``, and ``epoch`` columns.

    Returns
    -------
    dict[str, int]
        Mapping ``{"contained": n, "extends": n, "none": n,
        "total": n}``. ``total`` is the number of immobile non-local
        events (sum of the three class counts).

    Notes
    -----
    This aggregates across all sessions. For per-session or
    per-animal breakdowns, call this on a filtered DataFrame.

    """
    non_local_immobile = get_immobile_events(
        data_over_time,
        event_col="non_local_event",
        speed_col="head_speed",
        speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
        method="endpoints",
    )

    def _classify(group: pd.DataFrame) -> str:
        has_swr = group["SWR_event"] > 0
        if not has_swr.any():
            return "none"
        if has_swr.all():
            return "contained"
        return "extends"

    classes = non_local_immobile.groupby(
        ["animal", "day", "epoch", "non_local_event"],
    ).apply(_classify)
    counts = classes.value_counts()
    return {
        "contained": int(counts.get("contained", 0)),
        "extends": int(counts.get("extends", 0)),
        "none": int(counts.get("none", 0)),
        "total": int(counts.sum()),
    }


def compute_swr_nl_event_counts(
    data_over_time: pd.DataFrame,
) -> dict[str, int]:
    """Event-level totals and overlap for the SWR / immobile non-local pair.

    Designed as the data source for a 2-circle Venn (Figure 3 panel b).
    Reports four numbers:

    - ``n_swr``                — total SWR events (any speed)
    - ``n_swr_with_nl``        — SWR events overlapping at least one NL event
    - ``n_nl_immobile``        — total immobile non-local events
    - ``n_nl_immobile_with_swr`` — immobile NL events overlapping at least one SWR

    The two "with" counts may differ by a few percent because a single
    SWR can overlap multiple NL events, or vice versa. For a Venn
    diagram, the natural "shared" region is the mean of the two counts
    (both refer to the same underlying overlap interval set but are
    counted from different sides).

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Aggregated data with ``non_local_event``, ``SWR_event``,
        ``head_speed``, and session group columns.

    Returns
    -------
    dict[str, int]
        Counts dict suitable for :func:`plot_swr_nl_venn_2circle`.

    """
    # Immobile non-local events
    nl_immobile = get_immobile_events(
        data_over_time,
        event_col="non_local_event",
        speed_col="head_speed",
        speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
        method="endpoints",
    )
    nl_groups = nl_immobile.loc[nl_immobile["non_local_event"] > 0].groupby(
        ["animal", "day", "epoch", "non_local_event"],
    )
    nl_with_swr = nl_groups.apply(lambda g: (g["SWR_event"] > 0).any())
    n_nl = len(nl_with_swr)
    n_nl_with_swr = int(nl_with_swr.sum())

    # All SWR events (not filtered by speed — ripple detection is speed-agnostic)
    swr_groups = data_over_time.loc[data_over_time["SWR_event"] > 0].groupby(
        ["animal", "day", "epoch", "SWR_event"],
    )
    swr_with_nl = swr_groups.apply(lambda g: (g["non_local_event"] > 0).any())
    n_swr = len(swr_with_nl)
    n_swr_with_nl = int(swr_with_nl.sum())

    return {
        "n_swr": n_swr,
        "n_swr_with_nl": n_swr_with_nl,
        "n_nl_immobile": n_nl,
        "n_nl_immobile_with_swr": n_nl_with_swr,
    }
