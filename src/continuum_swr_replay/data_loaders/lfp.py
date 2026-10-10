"""LFP, ripple, theta, and multiunit data loading.

This module provides functions for loading local field potential (LFP) data from
Spyglass/NWB files, including theta oscillations, ripple events, and multiunit
activity. All data is extracted from the Spyglass DataJoint pipeline and returned
as time-indexed pandas DataFrames or TypedDicts.

Main Functions
--------------
get_theta
    Load theta-band (5-11 Hz) filtered LFP with phase and power estimates.
get_ripple
    Load ripple-band (150-250 Hz) filtered LFP and detected ripple events.
get_ripple_band_inputs
    Read the stored CA1 ripple band and the referenced raw LFP behind it,
    read-only, for direct SWR detection.
get_spyglass_kay_inputs
    Read the inputs ``RippleTimesV1.make`` gave Kay, read-only, for the SWR
    runner's version check.
get_multiunit
    Compute multiunit high-synchrony events (HSE) from spike times across brain areas.

Helper Functions
----------------
get_multiunit_hse
    Compute HSE detection for a single brain area's spike data.
filter_ripple_lfp_time
    Filter ripple data to match position time bounds.
check_band_electrodes
    Check that a ripple band's electrodes are its ripple selection's.

Data Sources
------------
The module queries the following Spyglass DataJoint tables:
- LFPBandV1, LFPBandSelection: Band-pass filtered LFP data and its electrodes
- LFPOutput: Raw LFP traces
- RippleTimesV1, RippleLFPSelection: Detected ripple events using Kay detector,
  and the electrodes it read
- FirFilterParameters: The band filter's coefficients

Notes
-----
- All time indices are in seconds
- Theta phase is computed from the first reference electrode
- Ripple consensus trace combines multiple electrodes for robust detection
- HSE detection uses the ripple_detection package with default thresholds
- Z-scored traces handle constant values to avoid division by zero

See Also
--------
continuum_swr_replay.types : ThetaData, RippleData, RippleBandInputs,
    MultiunitData TypedDicts
continuum_swr_replay.data_loaders.spikes : Spike data loading

Examples
--------
>>> theta_data = get_theta("chimi20200212_.nwb", "02_r1")
>>> ripple_data = get_ripple("chimi20200212_.nwb", "02_r1", "Kay")
>>> multiunit_data = get_multiunit(spike_times, position_info)

"""

import warnings
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from ripple_detection import (
    get_Kay_ripple_consensus_trace,
    get_multiunit_population_firing_rate,
    multiunit_HSE_detector,
    normalize_signal,
)
from scipy.stats import zscore

# ``spyglass.{lfp,ripple}`` imports open a DataJoint connection at
# import time. Defer them into the DB-loading function bodies so that
# pure helpers (``get_multiunit``, ``get_multiunit_hse``,
# ``filter_ripple_lfp_time``) — and the unit tests that exercise them
# — stay offline-importable on machines without VPN.
from ..parameters import (
    HSE_CLOSE_EVENT_THRESHOLD_SEC,
    HSE_MINIMUM_DURATION_SEC,
    HSE_ZSCORE_THRESHOLD,
    MAX_SPEED_THRESHOLD_CM_PER_SEC,
)
from ..types import MultiunitData, RippleBandInputs, RippleData, ThetaData

KAY_SMOOTHING_SIGMA_S = 0.004
"""The Kay consensus's Gaussian smoothing, in seconds: the SWR runner's
``swr_inventories.DETECTOR_SETTINGS["kay"]["smoothing_sigma"]`` (a test holds
the two equal; ``data_loaders`` may not import ``analysis``)."""

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pynwb")
warnings.filterwarnings("ignore", category=UserWarning, module="hdmf")
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint")


def get_theta(nwb_file_name: str, epoch_name: str) -> ThetaData:
    """Load theta-band LFP data.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name
    epoch_name : str
        Epoch name

    Returns
    -------
    ThetaData
        TypedDict containing:
        - theta_filtered_lfp: pd.DataFrame with filtered LFP
        - theta_phase: pd.Series with theta phase
        - theta_power: pd.Series with theta power

    """
    from spyglass.lfp.analysis.v1.lfp_band import LFPBandV1

    theta_lfp_key = {
        "nwb_file_name": nwb_file_name,
        "filter_name": "theta_5_11",
        "filter_sampling_rate": 1000,
        "lfp_band_sampling_rate": 1000,
        "target_interval_list_name": epoch_name + " noPrePostTrialTimes",
    }
    theta_table = LFPBandV1 & theta_lfp_key
    theta_filtered_lfp = theta_table.fetch1_dataframe()
    # Pick the first reference electrode
    electrode_list = [theta_table.fetch_nwb()[0]["lfp_band"].electrodes.data[0]]
    theta_phase = theta_table.compute_signal_phase(electrode_list=electrode_list)
    theta_power = theta_table.compute_signal_power(electrode_list=electrode_list)
    return {
        "theta_filtered_lfp": theta_filtered_lfp,
        "theta_phase": theta_phase,
        "theta_power": theta_power,
    }


def _ripple_keys(nwb_file_name: str, epoch_name: str) -> tuple[dict, dict]:
    """Return the restrictions that select an epoch's CA1 ripples and band.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name
    epoch_name : str
        Epoch name

    Returns
    -------
    ripple_key : dict
        Restriction of ``RippleTimesV1`` (and ``LFPBandSelection``) to the
        epoch's CA1 ripples, detected with the Kay parameter set "default".
    lfp_band_restriction : dict
        Restriction of ``LFPBandV1`` to the epoch's 1000 Hz ripple band.

    """
    # NOTE: the primary-key field is `ripple_param_name`, not `ripple_params`.
    # DataJoint silently drops dict keys absent from a table's heading, so the
    # old `ripple_params` key left the parameter dimension unrestricted (it only
    # worked because `fetch1_dataframe` requires a single matching row).
    ripple_key = {
        "nwb_file_name": nwb_file_name,
        "target_interval_list_name": epoch_name + " noPrePostTrialTimes",
        "group_name": "CA1",
        "ripple_param_name": "default",
        "filter_name": "ripple_150_250",
        "filter_sampling_rate": 1000,
    }
    lfp_band_restriction = {
        "nwb_file_name": nwb_file_name,
        "filter_name": ripple_key["filter_name"],
        "target_interval_list_name": ripple_key["target_interval_list_name"],
        "lfp_band_sampling_rate": ripple_key["filter_sampling_rate"],
    }
    return ripple_key, lfp_band_restriction


def get_ripple(
    nwb_file_name: str, epoch_name: str, ripple_detector_name: str
) -> RippleData:
    """Load ripple detection data.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name
    epoch_name : str
        Epoch name
    ripple_detector_name : str
        Ripple detector name (currently only "Kay" is supported)

    Returns
    -------
    RippleData
        TypedDict containing:
        - ripple_times: pd.DataFrame with ripple start/end times
        - ripple_filtered_lfps: pd.DataFrame with ripple-band filtered LFP
        - ripple_lfps: pd.DataFrame with raw ripple LFP
        - ripple_consensus_trace: pd.Series with ripple consensus trace
        - zscored_ripple_consensus_trace: pd.Series with z-scored consensus trace

    """
    from spyglass.lfp.analysis.v1.lfp_band import LFPBandSelection, LFPBandV1
    from spyglass.lfp.lfp_merge import LFPOutput
    from spyglass.ripple.v1 import RippleTimesV1

    # Ripples. The Kay detector's parameter set, "default", is the only one.
    if ripple_detector_name != "Kay":
        raise ValueError("Invalid ripple detector name")

    ripple_key, lfp_band_restriction = _ripple_keys(nwb_file_name, epoch_name)
    ripple_param_name = ripple_key["ripple_param_name"]
    lfp_band_key = (LFPBandV1 & lfp_band_restriction).fetch1("KEY")

    # Fail loudly and self-descriptively when a session has no matching CA1
    # ripple extraction, rather than the opaque `fetch1` "0 tuples found" error.
    # The epochs with none (wilbur20210404, a depletion session, and those
    # pending a populate) are not analysed: see docs/epoch_register.md.
    ripple_query = RippleTimesV1 & ripple_key
    if len(ripple_query) != 1:
        available_group_names = sorted(
            set(
                (
                    RippleTimesV1
                    & {
                        "nwb_file_name": nwb_file_name,
                        "target_interval_list_name": ripple_key[
                            "target_interval_list_name"
                        ],
                    }
                ).fetch("group_name")
            )
        )
        raise ValueError(
            f"Expected exactly one RippleTimesV1 row for {nwb_file_name} "
            f"{epoch_name} (group_name='CA1', "
            f"ripple_param_name='{ripple_param_name}'), found {len(ripple_query)}. "
            f"group_name(s) present for this session: {available_group_names}. "
            "An epoch with no CA1 ripple extraction is not analysed; the epoch "
            "register (docs/epoch_register.md) records why."
        )
    ripple_times = ripple_query.fetch1_dataframe()
    ripple_filtered_lfp = (LFPBandV1 & lfp_band_key).fetch1_dataframe()
    lfp_merge_id = (LFPBandSelection & ripple_key).fetch1("lfp_merge_id")
    lfp_df = (LFPOutput & {"merge_id": lfp_merge_id}).fetch1_dataframe()

    ripple_consensus_trace = RippleTimesV1.get_Kay_ripple_consensus_trace(
        ripple_filtered_lfp, sampling_frequency=ripple_key["filter_sampling_rate"]
    )
    zscored_ripple_consensus_trace = zscore(ripple_consensus_trace, nan_policy="omit")
    zscored_ripple_consensus_trace = pd.Series(
        zscored_ripple_consensus_trace.squeeze(),
        index=ripple_consensus_trace.index,
        name="zscored_ripple_consensus_trace",
    )

    return {
        "ripple_times": ripple_times,
        "ripple_filtered_lfps": ripple_filtered_lfp,
        "ripple_lfps": lfp_df,
        "ripple_consensus_trace": ripple_consensus_trace,
        "zscored_ripple_consensus_trace": zscored_ripple_consensus_trace,
    }


def ripple_band_electrode_ids(nwb_file_name: str, epoch_name: str) -> NDArray[np.int64]:
    """The stored CA1 ripple band's electrodes, in its column order.

    The order of ``get_ripple``'s ``ripple_filtered_lfps`` columns: ascending
    electrode id, as ``get_ripple_band_inputs`` checks the band holds them.

    Parameters
    ----------
    nwb_file_name, epoch_name : str

    Returns
    -------
    NDArray of int, shape (n_channels,)
    """
    from spyglass.lfp.analysis.v1.lfp_band import LFPBandSelection, LFPBandV1

    _, lfp_band_restriction = _ripple_keys(nwb_file_name, epoch_name)
    band_key = (LFPBandV1 & lfp_band_restriction).fetch1("KEY")
    electrode_ids = (LFPBandSelection.LFPBandElectrode & band_key).fetch("electrode_id")
    return np.sort(np.asarray(electrode_ids, dtype=np.int64))


def kay_ripple_power_z(
    time: NDArray[np.float64],
    ripple_band: NDArray[np.float64],
    sampling_frequency: float,
    unobserved: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The Kay detector's consensus trace and its z-score, the mask left out.

    ``ripple_detection``'s own ``get_Kay_ripple_consensus_trace`` (``sqrt`` of
    the Gaussian-smoothed sum of squared envelopes, within each gap-free block)
    and ``normalize_signal`` (mean and SD over the samples it is given), as
    ``Kay_ripple_detector`` computes and normalizes them: the trace the SWR
    inventory's events are thresholds of. The normalization reads the finite
    samples outside `unobserved`; the z-score inside it is NaN (no value).

    Parameters
    ----------
    time : NDArray, shape (n_time,)
        Seconds, ascending.
    ripple_band : NDArray, shape (n_time, n_channels)
        The stored 150-250 Hz band of the tetrodes to combine (rule R-T's kept
        ones).
    sampling_frequency : float
        Hz.
    unobserved : NDArray of bool, shape (n_time,)
        The artifact mask on `time`.

    Returns
    -------
    consensus : NDArray, shape (n_time,)
        NaN where a channel is not finite.
    zscore : NDArray, shape (n_time,)
        NaN there and inside `unobserved`.
    """
    unobserved = np.asarray(unobserved, dtype=bool)
    consensus = get_Kay_ripple_consensus_trace(
        ripple_band, sampling_frequency, KAY_SMOOTHING_SIGMA_S, time=time
    )
    normalized = np.asarray(
        normalize_signal(
            consensus,
            method="zscore",
            normalization_mask=np.isfinite(consensus) & ~unobserved,
        ),
        dtype=np.float64,
    )
    normalized[unobserved] = np.nan
    return np.asarray(consensus, dtype=np.float64), normalized


def with_ripple_power(
    ripple_data: RippleData,
    ripple_power: Mapping[str, object],
    *,
    band_electrode_ids: NDArray[np.int64],
) -> RippleData:
    """`ripple_data` with both consensus traces the SWR source's covariate.

    The traces become :func:`kay_ripple_power_z` over the tetrodes rule R-T
    kept for the configured SWR inventory, on the band as given (already
    within the position span, as the SWR runner reads it), with the artifact
    mask out of the normalization and NaN inside it. Everything else is as
    given.

    Parameters
    ----------
    ripple_data : RippleData
        As ``filter_ripple_lfp_time`` returns it.
    ripple_power : Mapping
        ``swr_source.ripple_power_inputs``: ``band_electrode_ids`` and
        ``kept_electrode_ids`` (the inventory's), ``sampling_frequency`` (Hz)
        and ``artifact_intervals`` (closed, seconds).
    band_electrode_ids : NDArray of int, shape (n_channels,)
        The electrodes of `ripple_data`'s band, in column order
        (:func:`ripple_band_electrode_ids`).

    Returns
    -------
    RippleData

    Raises
    ------
    ValueError
        If the band is not the inventory's band, or a kept electrode is not
        in it.
    """
    band = ripple_data["ripple_filtered_lfps"]
    band_ids = np.asarray(band_electrode_ids, dtype=np.int64)
    inventory_ids = np.asarray(ripple_power["band_electrode_ids"], dtype=np.int64)
    if band.shape[1] != band_ids.size or not np.array_equal(band_ids, inventory_ids):
        raise ValueError(
            f"The ripple band holds electrodes {band_ids.tolist()} in "
            f"{band.shape[1]} columns, but the SWR inventory read the band "
            f"{inventory_ids.tolist()}."
        )
    kept = np.isin(band_ids, np.asarray(ripple_power["kept_electrode_ids"]))
    if kept.sum() != len(ripple_power["kept_electrode_ids"]):
        raise ValueError("A tetrode the SWR inventory kept is not in the ripple band.")
    time = band.index.to_numpy(dtype=np.float64)
    consensus, z = kay_ripple_power_z(
        time,
        band.to_numpy(dtype=np.float64)[:, kept],
        float(ripple_power["sampling_frequency"]),
        _in_intervals(time, ripple_power["artifact_intervals"]),
    )
    return {
        **ripple_data,
        "ripple_consensus_trace": pd.Series(
            consensus, index=band.index, name="ripple_consensus_trace"
        ),
        "zscored_ripple_consensus_trace": pd.Series(
            z, index=band.index, name="zscored_ripple_consensus_trace"
        ),
    }


def get_spyglass_kay_inputs(
    nwb_file_name: str, epoch_name: str
) -> tuple[pd.Series, pd.DataFrame, float, dict]:
    """The inputs and parameters ``RippleTimesV1.make`` gave Kay, read only.

    ``RippleTimesV1.get_ripple_lfps_and_position_info``'s steps
    (``spyglass/ripple/v1/ripple.py``) on the epoch's full ``RippleTimesV1``
    key, with one change: the position of the key's ``pos_merge_id`` is read
    by ``get_merge_position_info``, from the source the merge master names, as
    ``load_data`` reads a position. Spyglass's own read,
    ``(PositionOutput & {"merge_id": ...}).fetch1_dataframe()``, refuses an
    entry held in two parts (j1620210710 ``02_r1`` and ``04_r2``); for any
    other entry the two reads return the same frame. The steps:

    - the key's ``RippleParameters`` row;
    - the stored ``LFPBandV1`` band's columns of the
      ``RippleLFPSelection.RippleLFPElectrode`` electrodes, in the band's
      dtype, indexed by its timestamps;
    - the key's target interval intersected with the position's first and
      last time, and the band and the position restricted to it;
    - the position interpolated linearly onto the band's times
      (``interpolate_to_new_time``), and its ``speed_name`` column.

    Parameters
    ----------
    nwb_file_name, epoch_name : str

    Returns
    -------
    speed : pd.Series
        Head speed in cm/s on the LFP times.
    ripple_lfps : pd.DataFrame, shape (n_time, n_channels)
        The stored ripple band, indexed by time in seconds.
    sampling_frequency : float
        The band's nominal rate in Hz.
    ripple_param_dict : dict
        The stored parameters: ``speed_name``, ``ripple_detection_algorithm``
        and ``ripple_detection_params``.
    """
    from spyglass.common import IntervalList
    from spyglass.lfp.analysis.v1.lfp_band import LFPBandV1
    from spyglass.ripple.v1 import RippleLFPSelection, RippleParameters, RippleTimesV1
    from spyglass.ripple.v1.ripple import interpolate_to_new_time
    from spyglass.utils.nwb_helper_fn import get_electrode_indices

    from .position import get_merge_position_info

    ripple_key, _ = _ripple_keys(nwb_file_name, epoch_name)
    key = (RippleTimesV1 & ripple_key).fetch1("KEY")
    ripple_param_dict: dict = (
        RippleParameters & {"ripple_param_name": key["ripple_param_name"]}
    ).fetch1("ripple_param_dict")

    electrode_keys = (RippleLFPSelection.RippleLFPElectrode() & key).fetch(
        "electrode_id"
    )
    ripple_lfp_nwb = (LFPBandV1 & key).fetch_nwb()[0]
    band = ripple_lfp_nwb["lfp_band"]
    band_electrodes = band.electrodes.data[:]
    is_selected = np.zeros_like(band_electrodes, dtype=bool)
    selected = [
        electrode for electrode in electrode_keys if electrode in band_electrodes
    ]
    is_selected[get_electrode_indices(band, selected)] = True
    ripple_lfp = pd.DataFrame(
        band.data, index=pd.Index(band.timestamps, name="time")
    ).loc[:, is_selected]
    sampling_frequency = ripple_lfp_nwb["lfp_band_sampling_rate"]

    target_times = (
        IntervalList
        & {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": key["target_interval_list_name"],
        }
    ).fetch_interval()
    position_info = get_merge_position_info(key["pos_merge_id"])
    valid_times = target_times.intersect(
        np.array([position_info.index[0], position_info.index[-1]])
    ).times
    position_info = pd.concat(
        [position_info.loc[slice(start, end)] for start, end in valid_times], axis=0
    )
    ripple_lfps = pd.concat(
        [ripple_lfp.loc[slice(start, end)] for start, end in valid_times], axis=0
    )
    position_info = interpolate_to_new_time(position_info, ripple_lfps.index)
    return (
        position_info[ripple_param_dict["speed_name"]],
        ripple_lfps,
        sampling_frequency,
        ripple_param_dict,
    )


def check_band_electrodes(
    band_electrode_ids: Iterable[int], selection_electrode_ids: Iterable[int]
) -> None:
    """Raise unless a ripple band's electrodes are its ripple selection's.

    Parameters
    ----------
    band_electrode_ids : Iterable[int], shape (n_band,)
        Electrodes of the stored ripple band, from
        ``LFPBandSelection.LFPBandElectrode``.
    selection_electrode_ids : Iterable[int], shape (n_selection,)
        Electrodes the ripple detection reads for the same band key, from
        ``RippleLFPSelection.RippleLFPElectrode``.

    Raises
    ------
    ValueError
        If the two sets differ. The message names the selection's electrodes
        missing from the band and the band's electrodes the selection lacks.

    """
    band = {int(electrode_id) for electrode_id in band_electrode_ids}
    selection = {int(electrode_id) for electrode_id in selection_electrode_ids}
    if band != selection:
        raise ValueError(
            "The ripple band's electrodes differ from its ripple selection's: "
            f"missing from the band: {sorted(selection - band)}; "
            f"extra in the band: {sorted(band - selection)}."
        )


def get_ripple_band_inputs(nwb_file_name: str, epoch_name: str) -> RippleBandInputs:
    """Read an epoch's stored CA1 ripple band and the inputs that made it.

    The fetch only reads: it inserts, populates and deletes nothing. It reads:

    - the stored ``LFPBandV1`` band that ``_ripple_keys`` selects, with its
      electrodes, references and electrode groups from
      ``LFPBandSelection.LFPBandElectrode``;
    - the raw LFP behind it, through ``LFPBandV1.make``'s own steps:
      ``_compute_interval_times`` gives the interval make filtered over, and
      ``_load_lfp_data_with_referencing`` maps each electrode to its LFP column
      through the LFP's electrode table and subtracts its reference in the
      stored dtype;
    - the band filter's coefficients, and the name, units and conversion of
      the LFP behind it.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name
    epoch_name : str
        Epoch name

    Returns
    -------
    RippleBandInputs
        ``ripple_band`` and ``raw_referenced`` share the band's timestamps and
        hold ``electrode_ids`` in that order. ``raw_referenced_native`` is the
        referenced LFP as make passes it to ``filter_data``, every column of
        it; ``lfp_band_elect_index`` picks the same electrodes from it.

    Raises
    ------
    ValueError
        If the band's electrodes differ from ``RippleLFPSelection``'s CA1
        electrodes for the same band key, if the stored band's columns are not
        its electrodes in ascending order, if the stored band does not have one
        column per electrode, or if the band's timestamps are not among the
        referenced raw LFP's timestamps.

    """
    from spyglass.common import FirFilterParameters
    from spyglass.lfp.analysis.v1.lfp_band import LFPBandSelection, LFPBandV1
    from spyglass.lfp.lfp_merge import LFPOutput
    from spyglass.ripple.v1 import RippleLFPSelection
    from spyglass.utils.nwb_helper_fn import get_electrode_indices

    ripple_key, lfp_band_restriction = _ripple_keys(nwb_file_name, epoch_name)
    band_key = (LFPBandV1 & lfp_band_restriction).fetch1("KEY")

    # The band's electrodes, references and electrode groups, ascending by
    # electrode: the order LFPBandV1._fetch_and_sort_electrodes gives make. The
    # group is part of the electrode's key, so the same fetch carries it.
    electrode_ids, reference_ids, electrode_group_names = (
        LFPBandSelection.LFPBandElectrode & band_key
    ).fetch("electrode_id", "reference_elect_id", "electrode_group_name")
    sort_order = np.argsort(electrode_ids)
    electrode_ids = np.asarray(electrode_ids, dtype=np.int64)[sort_order]
    reference_ids = np.asarray(reference_ids, dtype=np.int64)[sort_order]
    electrode_group_names = np.asarray(electrode_group_names, dtype=str)[sort_order]
    # RippleLFPSelection is keyed on LFPBandV1, so it is restricted by the full
    # band key: by less, it would union other epochs' selections.
    check_band_electrodes(
        electrode_ids,
        (
            RippleLFPSelection.RippleLFPElectrode
            & {**band_key, "group_name": ripple_key["group_name"]}
        ).fetch("electrode_id"),
    )

    band_series = (LFPBandV1 & band_key).fetch_nwb()[0]["lfp_band"]
    band_columns = [int(c) for c in get_electrode_indices(band_series, electrode_ids)]
    if band_columns != list(range(electrode_ids.size)):
        raise ValueError(
            "The stored ripple band's columns do not hold electrodes "
            f"{electrode_ids.tolist()} in order: they sit in columns {band_columns}."
        )
    time = np.asarray(band_series.timestamps, dtype=np.float64)
    ripple_band_native = np.asarray(band_series.data)
    # The electrode table maps electrode i to column i, so the data must hold
    # exactly one column per electrode; otherwise band_reproduction_error would
    # compare arrays of different widths.
    n_columns = ripple_band_native.shape[1]
    if n_columns != electrode_ids.size:
        raise ValueError(
            f"The stored ripple band has {n_columns} columns, not one for each of "
            f"its {electrode_ids.size} electrodes {electrode_ids.tolist()}."
        )

    # LFPBandV1.make's own steps, reused read-only.
    band_table = LFPBandV1()
    lfp_key = {"merge_id": band_key["lfp_merge_id"]}
    lfp_object = (LFPOutput & lfp_key).fetch_nwb()[0]["lfp"]
    _, band_valid_interval, _, lfp_band_sampling_rate, decimation = (
        band_table._compute_interval_times(band_key, lfp_key)
    )
    raw_timestamps, raw_referenced_native, lfp_band_elect_index = (
        band_table._load_lfp_data_with_referencing(
            lfp_object, electrode_ids, reference_ids, band_valid_interval
        )
    )
    raw_timestamps = np.asarray(raw_timestamps, dtype=np.float64)
    lfp_band_elect_index = np.asarray(lfp_band_elect_index, dtype=np.int64)

    # The same samples with no reference subtracted, to count where the
    # stored-dtype subtraction wrapped: for an integer dtype, exactly the
    # samples where it differs from a float64 subtraction.
    _, raw_unreferenced, _ = band_table._load_lfp_data_with_referencing(
        lfp_object, electrode_ids, np.full_like(reference_ids, -1), band_valid_interval
    )
    is_referenced = reference_ids != -1
    referenced_columns = lfp_band_elect_index[is_referenced]
    reference_columns = np.asarray(
        get_electrode_indices(lfp_object, reference_ids[is_referenced]), dtype=np.int64
    )
    exact = raw_unreferenced[:, referenced_columns].astype(np.float64)
    exact -= raw_unreferenced[:, reference_columns]
    reference_overflow_count = int(
        np.count_nonzero(raw_referenced_native[:, referenced_columns] != exact)
    )

    # make took the band's timestamps from these raw timestamps, so each band
    # sample has its raw sample; the float64 copy is restricted to them.
    is_band_time = np.isin(raw_timestamps, time)
    if not np.array_equal(raw_timestamps[is_band_time], time):
        n_missing = int(np.count_nonzero(~np.isin(time, raw_timestamps)))
        raise ValueError(
            f"{n_missing} of the ripple band's {time.size} timestamps are not "
            "among the referenced raw LFP's timestamps, or the raw timestamps "
            "repeat or fall out of order at them."
        )
    raw_referenced = raw_referenced_native[
        np.ix_(is_band_time, lfp_band_elect_index)
    ].astype(np.float64)

    band_filter_coeff = (
        FirFilterParameters
        & {
            "filter_name": band_key["filter_name"],
            "filter_sampling_rate": band_key["filter_sampling_rate"],
        }
    ).fetch1("filter_coeff")
    lfp_filter_name = LFPOutput.merge_get_parent(lfp_key).fetch1("filter_name")

    return {
        "time": time,
        "ripple_band": ripple_band_native.astype(np.float64),
        "ripple_band_dtype": str(ripple_band_native.dtype),
        "raw_referenced": raw_referenced,
        "raw_referenced_native": raw_referenced_native,
        "raw_timestamps": raw_timestamps,
        "reference_overflow_count": reference_overflow_count,
        "electrode_ids": electrode_ids,
        "reference_ids": reference_ids,
        "electrode_group_names": electrode_group_names,
        "lfp_band_elect_index": lfp_band_elect_index,
        "band_valid_interval": band_valid_interval,
        "decimation": int(decimation),
        "sampling_frequency": float(lfp_band_sampling_rate),
        "band_filter_coeff": np.asarray(band_filter_coeff, dtype=np.float64),
        "band_filter_name": band_key["filter_name"],
        "lfp_filter_name": lfp_filter_name,
        "lfp_units": lfp_object.unit,
        "lfp_conversion": float(lfp_object.conversion),
    }


def _in_intervals(
    time: NDArray[np.float64], intervals: pd.DataFrame
) -> NDArray[np.bool_]:
    """True at each time inside any closed ``[start_time, end_time]`` interval.

    Parameters
    ----------
    time : NDArray, shape (n_time,)
        Ascending seconds.
    intervals : pd.DataFrame
        ``start_time`` and ``end_time`` in seconds; any order, may overlap.

    Returns
    -------
    NDArray of bool, shape (n_time,)
    """
    first = np.searchsorted(
        time, intervals["start_time"].to_numpy(dtype=float), side="left"
    )
    stop = np.searchsorted(time, intervals["end_time"].to_numpy(dtype=float), "right")
    coverage = np.zeros(time.size + 1, dtype=np.int64)
    np.add.at(coverage, first, 1)
    np.add.at(coverage, stop, -1)
    inside: NDArray[np.bool_] = np.cumsum(coverage[:-1]) > 0
    return inside


def _overlap_any(events: pd.DataFrame, intervals: pd.DataFrame) -> NDArray[np.bool_]:
    """Events whose closed interval shares any time with a closed interval."""
    starts = events["start_time"].to_numpy(dtype=float)[:, np.newaxis]
    ends = events["end_time"].to_numpy(dtype=float)[:, np.newaxis]
    overlaps: NDArray[np.bool_] = (
        (starts <= intervals["end_time"].to_numpy(dtype=float)[np.newaxis, :])
        & (intervals["start_time"].to_numpy(dtype=float)[np.newaxis, :] <= ends)
    ).any(axis=1)
    return overlaps


def get_multiunit_hse(
    spike_times: list[NDArray[np.float64]],
    speed: NDArray[np.float64],
    bin_time: NDArray[np.float64],
    artifact_intervals: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute multiunit high synchrony events (HSE).

    Parameters
    ----------
    spike_times : list[NDArray[np.float64]]
        List of spike time arrays for all units
    speed : NDArray[np.float64]
        Speed of the animal in cm/s
    bin_time : NDArray[np.float64]
        Time bins for spike counting in seconds
    artifact_intervals : pd.DataFrame, optional
        Closed ``[start_time, end_time]`` intervals (seconds) whose spikes are
        not to be trusted, such as the artifact mask. Their bins
        are left out of the detector's normalization (``normalization_mask``)
        and of the z-scored rate's mean and SD, and an event that shares any
        time with one is dropped, the rest renumbered 1..n. The spike counts
        themselves are not changed. None, or no rows, masks nothing.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        - hse_times: pd.DataFrame with HSE start/end times; empty, with
          ``start_time``, ``end_time`` and ``duration`` columns, when the
          firing rate is constant over the unmasked bins (for example, an
          area with no spikes)
        - multiunit_firing_rate: pd.DataFrame with population firing rate
        - multiunit_rate_zscore: pd.DataFrame with z-scored firing rate

    """
    sampling_frequency = 1 / np.median(np.diff(bin_time))
    start_time, end_time = bin_time[[0, -1]]
    # ``np.digitize(t, bin_time[1:])`` assigns spikes to bin index
    # 0..N-1 where bin ``i`` covers ``[bin_time[i], bin_time[i+1])``
    # (and bin N-1 also catches ``t == bin_time[-1]``). The previous
    # ``bin_time[1:-1]`` form produced indices 0..N-2 only, so the
    # final output bin was unreachable — verified with
    # ``bin_time=[0,1,2,3]``: spikes at 2.5 and 3.0 both landed in
    # bin 2 and bin 3 stayed at zero.
    multiunit_spikes = np.stack(
        [
            np.bincount(
                np.digitize(
                    unit_spike_times[
                        np.logical_and(
                            unit_spike_times >= start_time,
                            unit_spike_times <= end_time,
                        )
                    ],
                    bin_time[1:],
                ),
                minlength=bin_time.shape[0],
            )
            for unit_spike_times in spike_times
        ],
        axis=1,
    )

    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_population_firing_rate(multiunit_spikes, sampling_frequency),
        index=bin_time,
        columns=["firing_rate"],
    )
    is_artifact = (
        np.zeros(bin_time.shape, dtype=bool)
        if artifact_intervals is None
        else _in_intervals(bin_time, artifact_intervals)
    )
    observed_rate = multiunit_firing_rate.to_numpy()[~is_artifact]
    mean = np.nanmean(observed_rate) if observed_rate.size else np.nan
    std = np.nanstd(observed_rate) if observed_rate.size else np.nan

    # Avoid division by near-zero if firing rate is constant
    if std < 1e-10 or np.isnan(std):
        multiunit_rate_zscore = pd.DataFrame(
            np.zeros_like(multiunit_firing_rate),
            index=bin_time,
            columns=["firing_rate"],
        )
        # A constant rate holds no HSE. ripple_detection 2.0 raises on a rate
        # with no scale, where 1.7.0 returned no events, so skip the detector.
        hse_times = pd.DataFrame(
            {"start_time": [], "end_time": [], "duration": []},
            index=pd.Index([], dtype=np.int64, name="event_number"),
            dtype=float,
        )
        return hse_times, multiunit_firing_rate, multiunit_rate_zscore

    multiunit_rate_zscore = (multiunit_firing_rate - mean) / std

    hse_times = multiunit_HSE_detector(
        bin_time,
        multiunit_spikes,
        speed,
        sampling_frequency,
        speed_threshold=MAX_SPEED_THRESHOLD_CM_PER_SEC,
        minimum_duration=HSE_MINIMUM_DURATION_SEC,
        zscore_threshold=HSE_ZSCORE_THRESHOLD,
        close_event_threshold=HSE_CLOSE_EVENT_THRESHOLD_SEC,
        normalization_mask=None if not is_artifact.any() else ~is_artifact,
    )
    if artifact_intervals is not None and len(artifact_intervals):
        hse_times = hse_times.loc[~_overlap_any(hse_times, artifact_intervals)]
        hse_times.index = pd.RangeIndex(1, len(hse_times) + 1, name="event_number")

    return hse_times, multiunit_firing_rate, multiunit_rate_zscore


def get_multiunit(
    spike_times: dict[str, list[NDArray[np.float64]]],
    position_info: pd.DataFrame,
    artifact_intervals: Mapping[str, pd.DataFrame] | None = None,
) -> MultiunitData:
    """Get multiunit HSE data for all brain areas.

    Parameters
    ----------
    spike_times : dict[str, list[NDArray[np.float64]]]
        Dictionary mapping brain area to list of spike time arrays
    position_info : pd.DataFrame
        Position info with head_speed column and time index
    artifact_intervals : Mapping[str, pd.DataFrame], optional
        By brain area, the intervals ``get_multiunit_hse`` masks; an area not
        named is not masked.

    Returns
    -------
    MultiunitData
        TypedDict containing:
        - hse_times: dict mapping brain area to HSE times
        - multiunit_firing_rate: dict mapping brain area to firing rates
        - multiunit_rate_zscore: dict mapping brain area to z-scored rates

    """
    # Get multiunit HSE
    hse_times = {}
    multiunit_firing_rate = {}
    multiunit_rate_zscore = {}

    for brain_area in spike_times:
        (
            hse_times[brain_area],
            multiunit_firing_rate[brain_area],
            multiunit_rate_zscore[brain_area],
        ) = get_multiunit_hse(
            spike_times[brain_area],
            position_info.head_speed.to_numpy(),
            position_info.index.to_numpy(),
            artifact_intervals=(artifact_intervals or {}).get(brain_area),
        )

    return {
        "hse_times": hse_times,
        "multiunit_firing_rate": multiunit_firing_rate,
        "multiunit_rate_zscore": multiunit_rate_zscore,
    }


def filter_ripple_lfp_time(
    ripple_data: RippleData, position_time: NDArray[np.float64]
) -> RippleData:
    """Filter ripple LFP data to match position time bounds.

    Parameters
    ----------
    ripple_data : RippleData
        Dictionary with ripple data
    position_time : NDArray[np.float64]
        Position time array in seconds (defines time bounds)

    Returns
    -------
    RippleData
        New dictionary with filtered ripple data matching position time bounds

    """
    position_time_slice = slice(position_time[0], position_time[-1])

    ripple_times = ripple_data["ripple_times"]
    ripple_times = ripple_times[
        (ripple_times["start_time"] >= position_time[0])
        & (ripple_times["end_time"] <= position_time[-1])
    ]

    return {
        "ripple_times": ripple_times,
        "ripple_filtered_lfps": ripple_data["ripple_filtered_lfps"].loc[
            position_time_slice
        ],
        "ripple_lfps": ripple_data["ripple_lfps"].loc[position_time_slice],
        "ripple_consensus_trace": ripple_data["ripple_consensus_trace"].loc[
            position_time_slice
        ],
        "zscored_ripple_consensus_trace": ripple_data[
            "zscored_ripple_consensus_trace"
        ].loc[position_time_slice],
    }
