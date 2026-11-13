"""Aggregation of ContFrag decoding results over time.

Mirrors ``decode_quality.data_aggregation`` but for the 2-state
Continuous-Fragmented classifier. Key differences:

- State probability columns: ``Continuous_prob``, ``Fragmented_prob``
  (2 columns, not 4)
- No ``non_local_event`` column (events are externally defined via SWR/MUA)
- Posterior stats computed on the full (combined) posterior since both
  states decode meaningful spatial content
- Event overlap counts are SWR vs MUA only (no model-detected events)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from non_local_detector.environment import Environment

from continuum_swr_replay.analysis.decode_quality.data_aggregation import (
    theta_power_zscore,
    validate_required_session_data_keys,
    validate_results_position_time_alignment,
)
from continuum_swr_replay.analysis.event_definition.analysed_time import (
    read_dot_with_swr_source,
)
from continuum_swr_replay.analysis.event_definition.epoch_normalizers import (
    run_speed_baseline as epoch_run_speed_baseline,
)
from continuum_swr_replay.analysis.event_definition.immobility import (
    build_speed_eligible_event_cohort,
)
from continuum_swr_replay.analysis.event_definition.nonlocal_detection import (
    get_overlap_times,
)
from continuum_swr_replay.data_loaders.epoch_register import analysed_decode_files
from continuum_swr_replay.decoding.files import data_over_time_suffix
from continuum_swr_replay.decoding.identifiers import parse_nwb_filename
from continuum_swr_replay.parameters import SAMPLING_FREQUENCY

from ...data_types import SessionData
from ..decode_quality.posterior_analysis import (
    get_map_step_distance,
    get_posterior_stats,
)
from .posterior_analysis import get_contfrag_posterior


def get_contfrag_data_over_time(
    data: SessionData,
    results: xr.Dataset,
    env: Environment,
    nwb_file_name: str,
    epoch_name: str,
    brain_area: str = "HPC",
    sampling_frequency: float = SAMPLING_FREQUENCY,
) -> tuple[pd.DataFrame, xr.DataArray]:
    """Aggregate ContFrag decoding results into time-indexed DataFrame.

    Parameters
    ----------
    data : SessionData
        Data dictionary from ``load_data()``.
    results : xr.Dataset
        ContFrag decoding results with states=["Continuous", "Fragmented"].
    env : Environment
        Environment configuration from detector.
    nwb_file_name : str
        Name of the NWB file.
    epoch_name : str
        Name of the epoch.
    brain_area : str, optional
        Brain area for MUA lookup, by default "HPC".
    sampling_frequency : int, optional
        Samples per second, by default 500.

    Returns
    -------
    data_over_time : pd.DataFrame
        Time-indexed DataFrame with state probabilities, posterior stats,
        event labels, neural signals, and behavioral data.
    posterior : xr.DataArray
        Combined (Continuous + Fragmented) posterior distribution.
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

    validate_results_position_time_alignment(
        np.asarray(results.time.values), data["position_info"].index
    )
    position_time = data["position_info"].index
    head_speed = data["position_info"].head_speed.to_numpy()

    # Get combined posterior (both states decode position)
    posterior = get_contfrag_posterior(results, state="combined")

    # External detector intervals become analysis events only when every
    # sampled head-speed value in the complete interval is finite and < 4 cm/s.
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

    # SWR vs MUA overlap
    swr_mua_times, swr_no_mua_times = get_overlap_times(swr_times, mua_times)

    # Posterior stats on the full posterior
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
    # Raw per-timebin track-graph displacement of the MAP position (cm),
    # unsmoothed (cf. the smoothed ``map_speed``) so downstream summaries can
    # sum it (path length) or take robust order statistics (jumpiness).
    step_distance = get_map_step_distance(
        posterior.values,
        env.track_graph_with_bin_centers_edges_,
        np.asarray(env.place_bin_centers_nodes_df_.node_id),
    )
    # Per-session "typical run speed" baseline (cm/s): robust, threshold-free
    # percentile of head speed over every bin, a per-session speed normalizer
    # for downstream replay-speed/compression measures. The read through the
    # SWR source takes it again over the bins outside the artifact mask.
    run_speed_baseline = epoch_run_speed_baseline(
        data["position_info"].head_speed.to_numpy()
    )

    animal, day = parse_nwb_filename(nwb_file_name)

    def _nearest(series):
        # Return a positional ndarray; otherwise the Series — indexed
        # by ``results.time`` — would be re-aligned against
        # ``position_info.index`` when packed into the DataFrame
        # below, silently introducing NaNs whenever the two axes
        # differ by epsilon despite identical lengths.
        return series.reindex(
            results.time, method="nearest", tolerance=0.004
        ).to_numpy()

    data_over_time = pd.DataFrame(
        {
            "SWR_event": swr_labels.to_numpy(),
            "MUA_event": mua_labels.to_numpy(),
            "Continuous_prob": results.sel(
                states="Continuous"
            ).acausal_state_probabilities.squeeze(),
            "Fragmented_prob": results.sel(
                states="Fragmented"
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
            "n_SWR": len(swr_times),
            "n_MUA": len(mua_times),
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


def load_contfrag_data_over_time(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate the analysed epochs' ContFrag processed data.

    The epochs are the epoch register's analysed ones
    (``epoch_register.analysed_decode_files``), never a glob of `data_dir`. Each
    is read through the SWR source (``analysed_time.read_dot_with_swr_source``):
    ``SWR_event`` is the direct Kay inventory's labels.

    Parameters
    ----------
    data_dir : Path
        Directory containing the ``{epoch_id}_contfrag_data_over_time.pkl`` files.

    Returns
    -------
    pd.DataFrame
        Concatenated data with derived distance column, and the source's
        ``SWR_event`` and ``swr_observed``.

    Raises
    ------
    FileNotFoundError
        If an analysed epoch's file is not in `data_dir`.
    SwrSourceError
        If the SWR source cannot serve an analysed epoch.
    """
    files = analysed_decode_files(
        data_dir, f"_contfrag{data_over_time_suffix(None)}", require_all=True
    )
    data_over_time = pd.concat([read_dot_with_swr_source(p) for p in files])
    data_over_time["distance"] = np.abs(data_over_time["ahead_behind_distance"])
    return data_over_time
