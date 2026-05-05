"""Aggregation of ContFrag decoding results over time.

Mirrors ``hippocampus.data_aggregation`` but for the 2-state
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
from scipy.stats import zscore

from continuum_swr_replay.data_loaders import get_labels
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename

from ...types import SessionData
from ..hippocampus.event_detection import get_overlap_times
from ..hippocampus.posterior_analysis import get_posterior_stats
from .posterior_analysis import get_contfrag_posterior


def get_contfrag_data_over_time(
    data: SessionData,
    results: xr.Dataset,
    env: Environment,
    nwb_file_name: str,
    epoch_name: str,
    brain_area: str = "HPC",
    sampling_frequency: float = 500.0,
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
    required_keys = [
        "position_info",
        "ripple_times",
        "hse_times",
        "theta_phase",
        "theta_power",
        "zscored_ripple_consensus_trace",
        "multiunit_rate_zscore",
    ]
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise KeyError(
            f"Missing required data keys: {missing_keys}. "
            f"Ensure load_data() was called with DataLoadingPreset.DECODING or equivalent."
        )

    # Get combined posterior (both states decode position)
    posterior = get_contfrag_posterior(results, state="combined")

    # Event labels from external detectors
    swr_labels = get_labels(data["ripple_times"], results.time)
    mua_labels = get_labels(data["hse_times"][brain_area], results.time)

    # SWR vs MUA overlap
    swr_mua_times, swr_no_mua_times = get_overlap_times(
        data["ripple_times"], data["hse_times"][brain_area], results
    )

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

    animal, day = parse_nwb_filename(nwb_file_name)

    def _nearest(series):
        return series.reindex(results.time, method="nearest", tolerance=0.004)

    # Length guard runs first so we can construct on position_info.index
    # safely; without that index the axis=1 concat below would align a
    # default RangeIndex against a time index and silently produce a
    # 2N-row NaN-filled frame.
    if len(results.time) != len(data["position_info"]):
        raise ValueError(
            f"results.time ({len(results.time)}) and position_info "
            f"({len(data['position_info'])}) have different lengths; "
            "they must be aligned 1-to-1."
        )

    data_over_time = pd.DataFrame(
        {
            "SWR_event": swr_labels.squeeze(),
            "MUA_event": mua_labels.squeeze(),
            "Continuous_prob": results.sel(
                states="Continuous"
            ).acausal_state_probabilities.squeeze(),
            "Fragmented_prob": results.sel(
                states="Fragmented"
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
            "n_SWR": len(data["ripple_times"]),
            "n_MUA": len(data["hse_times"][brain_area]),
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
    """Load and concatenate all ContFrag processed session data.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``*_contfrag_data_over_time.pkl`` files.

    Returns
    -------
    pd.DataFrame
        Concatenated data with derived distance column.
    """
    files = sorted(data_dir.glob("*_contfrag_data_over_time.pkl"))
    if not files:
        raise FileNotFoundError(
            f"No *_contfrag_data_over_time.pkl files found in {data_dir}"
        )
    data_over_time = pd.concat([pd.read_pickle(p) for p in files])
    data_over_time["distance"] = np.abs(data_over_time["ahead_behind_distance"])
    return data_over_time
