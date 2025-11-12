import itertools
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from datajoint import DataJointError
from ripple_detection import (
    get_multiunit_population_firing_rate,
    multiunit_HSE_detector,
)
from scipy.interpolate import interp1d
from scipy.ndimage import label
from scipy.stats import zscore
from spyglass.common import (
    DIOEvents,
    IntervalList,
    Nwbfile,
    PositionIntervalMap,
    Session,
)
from spyglass.common.common_position import IntervalPositionInfo
from spyglass.decoding.v0.clusterless import UnitMarks
from spyglass.lfp.analysis.v1.lfp_band import LFPBandSelection, LFPBandV1
from spyglass.lfp.lfp_merge import LFPOutput
from spyglass.linearization.v0.main import IntervalLinearizedPosition, TrackGraph
from spyglass.position import PositionOutput
from spyglass.ripple.v1 import RippleTimesV1
from spyglass.spikesorting.v0 import CuratedSpikeSorting, SortGroup
from spyglass.utils.nwb_helper_fn import get_nwb_file
from track_linearization import get_linearized_position

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pynwb")
warnings.filterwarnings("ignore", category=UserWarning, module="hdmf")
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint")

NWB_FILES = [
    "chimi20200212_.nwb",
    "chimi20200213_.nwb",
    "chimi20200214_.nwb",
    "chimi20200215_.nwb",
    "chimi20200216_.nwb",
    "chimi20200217_.nwb",
    "chimi20200218_.nwb",
    "chimi20200219_.nwb",
    "chimi20200220_.nwb",
    "chimi20200221_.nwb",
    "chimi20200222_.nwb",
    "chimi20200223_.nwb",
    "chimi20200224_.nwb",
    "chimi20200225_.nwb",
    "chimi20200226_.nwb",
    "chimi20200227_.nwb",
    "chimi20200228_.nwb",
    "chimi20200301_.nwb",
    "chimi20200302_.nwb",
    "chimi20200303_.nwb",
    "chimi20200304_.nwb",
    "chimi20200305_.nwb",
    "chimi20200306_.nwb",
    "chimi20200307_.nwb",
    "chimi20200308_.nwb",
    "chimi20200310_.nwb",
    "chimi20200311_.nwb",
    "chimi20200312_.nwb",
    "chimi20200313_.nwb",
    "j1620210706_.nwb",
    "j1620210707_.nwb",
    "j1620210708_.nwb",
    "j1620210709_.nwb",
    "j1620210710_.nwb",
    "j1620210711_.nwb",
    "j1620210712_.nwb",
    "j1620210713_.nwb",
    "j1620210714_.nwb",
    "j1620210715_.nwb",
    "j1620210716_.nwb",
    "j1620210717_.nwb",
    "j1620210718_.nwb",
    "j1620210719_.nwb",
    "j1620210720_.nwb",
    "j1620210721_.nwb",
    "peanut20201124_.nwb",
    "peanut20201125_.nwb",
    "peanut20201126_.nwb",
    "peanut20201127_.nwb",
    "peanut20201128_.nwb",
    "peanut20201129_.nwb",
    "peanut20201130_.nwb",
    "peanut20201201_.nwb",
    "peanut20201202_.nwb",
    "peanut20201203_.nwb",
    "peanut20201204_.nwb",
    "peanut20201205_.nwb",
    "peanut20201206_.nwb",
    "peanut20201207_.nwb",
    "peanut20201208_.nwb",
    "peanut20201209_.nwb",
    "senor20201027_.nwb",
    "senor20201028_.nwb",
    "senor20201029_.nwb",
    "senor20201030_.nwb",
    "senor20201031_.nwb",
    "senor20201101_.nwb",
    "senor20201102_.nwb",
    "senor20201103_.nwb",
    "senor20201104_.nwb",
    "senor20201105_.nwb",
    "senor20201106_.nwb",
    "senor20201107_.nwb",
    "senor20201108_.nwb",
    "senor20201109_.nwb",
    "senor20201110_.nwb",
    "senor20201111_.nwb",
    "senor20201112_.nwb",
    "senor20201113_.nwb",
    "senor20201114_.nwb",
    "senor20201115_.nwb",
    "senor20201116_.nwb",
    "senor20201117_.nwb",
    "senor20201118_.nwb",
    "senor20201119_.nwb",
    "senor20201120_.nwb",
    "senor20201121_.nwb",
    "wilbur20210326_.nwb",
    "wilbur20210327_.nwb",
    "wilbur20210328_.nwb",
    "wilbur20210329_.nwb",
    "wilbur20210330_.nwb",
    "wilbur20210331_.nwb",
    "wilbur20210401_.nwb",
    "wilbur20210402_.nwb",
    "wilbur20210403_.nwb",
    "wilbur20210404_.nwb",
    "wilbur20210405_.nwb",
    "wilbur20210406_.nwb",
    "wilbur20210407_.nwb",
    "wilbur20210408_.nwb",
    "wilbur20210409_.nwb",
    "wilbur20210410_.nwb",
    "wilbur20210411_.nwb",
    "wilbur20210412_.nwb",
    "wilbur20210413_.nwb",
    "wilbur20210414_.nwb",
    "wilbur20210415_.nwb",
    "wilbur20210416_.nwb",
]

WELL_PATCH_MAPPING = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 3,
    5: 3,
}

TRACK_SEGMENT_TO_PATCH = {
    0: 1,
    1: 1,
    6: 1,
    2: 2,
    3: 2,
    7: 2,
    4: 3,
    5: 3,
    8: 3,
}
TRACK_SEGMENT_TO_EDGE_ID = {
    0: 1,
    1: 2,
    2: 4,
    3: 5,
    4: 7,
    5: 8,
    6: 0,
    7: 3,
    8: 6,
}


def get_epoch_info() -> pd.DataFrame:
    epoch_info = (
        pd.DataFrame(
            (
                Session
                & [
                    {
                        "nwb_file_name": nwbf,
                        "session_description": "Spatial bandit task (regular)",
                    }
                    for nwbf in NWB_FILES
                ]
            )
            * PositionIntervalMap
        )
        .set_index(["nwb_file_name", "interval_list_name"])
        .sort_index()
    )

    # only use run epochs
    epoch_info = epoch_info[
        epoch_info.index.get_level_values("interval_list_name").str.match(
            r"^[0-9]+_r[0-9]+$"
        )
    ]

    animal, date = zip(
        *[
            (
                file_name.replace("_.nwb", "")[:-8],
                int(file_name.replace("_.nwb", "")[-8:]),
            )
            for file_name in epoch_info.index.get_level_values(
                "nwb_file_name"
            ).to_numpy()
        ]
    )
    epoch_info["animal"] = animal
    epoch_info["date"] = date

    epoch, epoch_tag = zip(
        *epoch_info.index.get_level_values("interval_list_name").str.split("_")
    )
    epoch_info["epoch"] = np.asarray(epoch, dtype=int)
    epoch_info["epoch_tag"] = epoch_tag

    def _count_exposure(df):
        df["exposure"] = np.arange(len(df)) + 1
        return df

    epoch_info = (
        epoch_info.sort_values(["animal", "date", "epoch"])
        .groupby("animal", group_keys=False)
        .apply(_count_exposure)
    )

    interval_times = np.stack(
        (
            IntervalList
            & epoch_info.reset_index()
            .loc[:, ("nwb_file_name", "interval_list_name")]
            .to_dict(orient="records")
        ).fetch("valid_times")
    ).squeeze()

    epoch_info["start_time"] = pd.to_datetime(interval_times[:, 0], unit="s")
    epoch_info["end_time"] = pd.to_datetime(interval_times[:, 1], unit="s")

    return epoch_info


def _get_interpolated_position_info(
    position_info: pd.DataFrame,
    time: np.ndarray,
    track_graph: nx.Graph,
    edge_order: list[tuple[int, int]],
    edge_spacing: list[float],
    position_columns: list[str] = ["head_position_x", "head_position_y"],
) -> pd.DataFrame:
    new_index = pd.Index(
        np.unique(np.concatenate((position_info.index, time))),
        name="time",
    )

    interpolated_position_info = (
        position_info.reindex(index=new_index)
        .interpolate(method="linear")
        .reindex(index=time)
    )

    linear_position_info = get_linearized_position(
        position=interpolated_position_info[position_columns].to_numpy(),
        track_graph=track_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    ).set_index(interpolated_position_info.index)

    return pd.concat(
        (
            interpolated_position_info,
            linear_position_info,
        ),
        axis=1,
    )


def _get_position_info(nwb_file_name: str, epoch_name: str, pos_name: str) -> dict:
    position_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": pos_name,
        "position_info_param_name": "default_decoding",
    }

    linearization_key = {
        "position_info_param_name": "default_decoding",
        "nwb_file_name": nwb_file_name,
        "interval_list_name": pos_name,
        "linearization_param_name": "default",
    }

    try:
        track_graph_name = (IntervalLinearizedPosition() & linearization_key).fetch1(
            "track_graph_name"
        )
    except DataJointError:
        track_graph_name = nwb_file_name.split("_")[0]

    track_graph = (
        TrackGraph() & {"track_graph_name": track_graph_name}
    ).get_networkx_track_graph()
    track_graph_params = (
        TrackGraph() & {"track_graph_name": track_graph_name}
    ).fetch1()
    linear_edge_order = track_graph_params["linear_edge_order"]
    linear_edge_spacing = track_graph_params["linear_edge_spacing"]

    try:
        epoch = int(epoch_name.split("_")[0])
        pos_merge_id = str(
            (
                PositionOutput().merge_restrict()
                & {
                    "nwb_file_name": nwb_file_name,
                    "source": "DLCPosV1",
                    "epoch": epoch,
                }
            ).fetch1("merge_id")
        )
        position_info = (
            (
                (PositionOutput() & {"merge_id": pos_merge_id})
                .fetch1_dataframe()
                .dropna()
            )
            .drop(columns="video_frame_ind")
            .add_prefix("head_")
        )
        time = (IntervalPositionInfo() & position_key).fetch1_dataframe().dropna().index

    except DataJointError:
        position_info = (
            (IntervalPositionInfo() & position_key).fetch1_dataframe().dropna()
        )
        time = position_info.index
    try:
        valid_interval_times = (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": epoch_name + " noPrePostTrialTimes",
            }
        ).fetch1("valid_times")
        position_info = position_info.loc[
            valid_interval_times[0][0] : valid_interval_times[-1][1]
        ]
    except DataJointError:
        pass

    position_info = _get_interpolated_position_info(
        position_info,
        time,
        track_graph,
        linear_edge_order,
        linear_edge_spacing,
    )
    position_info["patch_id"] = position_info["track_segment_id"].map(
        TRACK_SEGMENT_TO_PATCH
    )

    return {
        "position_info": position_info,
        "linear_edge_order": linear_edge_order,
        "linear_edge_spacing": linear_edge_spacing,
        "track_graph": track_graph,
    }


def _get_theta(nwb_file_name: str, epoch_name: str) -> dict:
    theta_lfp_key = {
        "nwb_file_name": nwb_file_name,
        "filter_name": "theta_5_11",
        "filter_sampling_rate": 1000,
        "target_interval_list_name": epoch_name + " noPrePostTrialTimes",
    }
    theta_filtered_lfp = (LFPBandV1 & theta_lfp_key).fetch1_dataframe()
    # Pick the first reference electrode
    electrode_list = [
        (LFPBandV1 & theta_lfp_key).fetch_nwb()[0]["lfp_band"].electrodes.data[0]
    ]
    theta_phase = (LFPBandV1 & theta_lfp_key).compute_signal_phase(
        electrode_list=electrode_list
    )
    theta_power = (LFPBandV1 & theta_lfp_key).compute_signal_power(
        electrode_list=electrode_list
    )
    return {
        "theta_filtered_lfp": theta_filtered_lfp,
        "theta_phase": theta_phase,
        "theta_power": theta_power,
    }


def _get_electrode_group_info(nwb_file_name: str) -> pd.DataFrame:
    nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
    nwbf = get_nwb_file(nwb_file_abspath)
    electrode_group_df = []
    for electrode_group in nwbf.electrode_groups.values():
        cur = {
            "electrode_group_name": electrode_group.name,
            "description": electrode_group.description,
        }
        cur.update(electrode_group.fields)
        electrode_group_df.append(cur)

    electrode_group_df = (
        pd.DataFrame(electrode_group_df)
        .drop(columns=["device"])
        .set_index("electrode_group_name")
    )

    is_CA1 = electrode_group_df.targeted_location.str.contains("CA1") & (
        electrode_group_df.location != "CorpusCallosum"
    )
    electrode_group_df.loc[is_CA1, "targeted_location"] = "CA1"

    is_mPFC = electrode_group_df.targeted_location.str.contains("mPFC") & (
        electrode_group_df.location != "CorpusCallosum"
    )
    electrode_group_df.loc[is_mPFC, "targeted_location"] = "mPFC"

    is_OFC = electrode_group_df.targeted_location.str.contains("OFC") & (
        electrode_group_df.location != "CorpusCallosum"
    )
    electrode_group_df.loc[is_OFC, "targeted_location"] = "OFC"

    return electrode_group_df


def _detect_coincident_spikes(
    spike_times: list,
    spike_closeness_threshold: float = 0.00004,
    max_coincident_fraction: float = 0.33,
) -> tuple:
    # Handle empty spike times
    if not spike_times or all(len(st) == 0 for st in spike_times):
        return [], []

    # Concatenate all spike times
    concat_spike_times = np.concatenate(spike_times)
    # Create group IDs for each spike time
    sort_group_id = np.concatenate(
        [
            np.ones(len(spike_time), dtype=int) * i
            for i, spike_time in enumerate(spike_times)
        ]
    )
    time_bin_ind = np.concatenate(
        [np.arange(len(spike_time), dtype=int) for spike_time in spike_times]
    )

    # Sort spike times and group IDs based on the spike times
    sort_ind = np.argsort(concat_spike_times)
    sorted_spike_times = concat_spike_times[sort_ind]
    sort_group_id = sort_group_id[sort_ind]
    time_bin_ind = time_bin_ind[sort_ind]

    # Find differences and label close spikes
    is_close = np.diff(sorted_spike_times) < spike_closeness_threshold
    is_close = np.concatenate([[False], is_close])
    labels, _ = label(is_close)

    # Create a DataFrame for further analysis
    df = pd.DataFrame(
        {
            "labels": labels,
            "sort_group_id": sort_group_id,
            "time_bin_ind": time_bin_ind,
            "spike_times": sorted_spike_times,
        },
    )
    # Calculate the fraction of each group and the median spike times
    n_sort_groups = len(spike_times)
    frac = (
        df.loc[df.labels > 0].groupby("labels").sort_group_id.nunique() / n_sort_groups
    )
    frac = frac[frac > max_coincident_fraction]

    df = df.loc[~df.labels.isin(frac.index)]

    filtered_spike_times = (
        df.groupby("sort_group_id").spike_times.apply(np.array).tolist()
    )
    filtered_time_bin_ind = (
        df.groupby("sort_group_id").time_bin_ind.apply(np.array).tolist()
    )

    return filtered_spike_times, filtered_time_bin_ind


def _get_hpc_marks(nwb_file_name: str) -> tuple:
    try:
        restriction = {
            "nwb_file_name": nwb_file_name,
            "curation_id": 0,
            "sort_interval_name": "runs_noPrePostTrialTimes raw data valid times",
            "preproc_params_name": "franklab_tetrode_hippocampus",
            "team_name": "ac_em_xs",
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "clusterless_fixed",
            "mark_param_name": "default",
        }
        marks = (UnitMarks & restriction).fetch_dataframe()
        marks = [(mark.index.to_numpy(), mark.to_numpy()) for mark in marks]
        spike_times, spike_waveform_features = zip(*marks)
    except ValueError:
        restriction = {
            "nwb_file_name": nwb_file_name,
            "curation_id": 0,
            "sort_interval_name": "runs_noPrePostTrialTimes raw data valid times",
            "preproc_params_name": "franklab_tetrode_hippocampus",
            "team_name": "ac_em_xs",
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default_clusterless",
            "mark_param_name": "default",
        }
        marks = (UnitMarks & restriction).fetch_dataframe()
        marks = [(mark.index.to_numpy(), mark.to_numpy()) for mark in marks]
        spike_times, spike_waveform_features = zip(*marks)

    spike_times, filtered_time_bin_ind = _detect_coincident_spikes(spike_times)
    spike_waveform_features = [
        features[ind]
        for ind, features in zip(filtered_time_bin_ind, spike_waveform_features)
    ]

    return spike_times, spike_waveform_features


def _get_pfc_spike_times(nwb_file_name: str, brain_area: str) -> list:
    restriction = {
        "nwb_file_name": nwb_file_name,
        "preproc_params_name": "default",
        "sort_interval_name": "sleeps_runs_noPrePostTrialTimes raw data valid times",
        "sorter_params_name": "franklab_probe_ctx_30KHz_115rad_new_mountainsort2",
        "team_name": "ac_em_xs",
        "sorter": "mountainsort4",
    }

    # Get the latest curation_id
    max_curation_id = (CuratedSpikeSorting & restriction).fetch("curation_id").max()
    restriction.update({"curation_id": max_curation_id})
    # “ampl_2000_z_30_prop_075_1ms” for artifact_params_name

    # Find brain area
    curated_spikes_info = pd.DataFrame(
        (CuratedSpikeSorting & restriction) * SortGroup.SortGroupElectrode
    )
    electrode_group_df = _get_electrode_group_info(nwb_file_name)
    curated_spikes_info = pd.merge(
        curated_spikes_info, electrode_group_df, on="electrode_group_name"
    )
    spikesorting_keys = pd.merge(
        pd.DataFrame(CuratedSpikeSorting() & restriction),
        curated_spikes_info.groupby("sort_group_id").targeted_location.first(),
        on="sort_group_id",
    )
    spikesorting_keys = spikesorting_keys.loc[
        spikesorting_keys.targeted_location == brain_area
    ].to_dict(orient="records")

    nwb_pfc = (CuratedSpikeSorting() & spikesorting_keys).fetch_nwb()

    return list(
        itertools.chain.from_iterable(
            [
                file["units"]["spike_times"].to_list()
                for file in nwb_pfc
                if "units" in file
            ]
        )
    )


def _get_spike_data(nwb_file_name: str) -> dict:
    spike_times = dict()
    spike_waveform_features = dict()

    try:
        spike_times["HPC"], spike_waveform_features["HPC"] = _get_hpc_marks(
            nwb_file_name
        )
    except ValueError:
        pass

    for brain_area in ["mPFC", "OFC"]:
        try:
            spike_times[brain_area] = _get_pfc_spike_times(nwb_file_name, brain_area)
            if len(spike_times[brain_area]) < 1:
                del spike_times[brain_area]
        except ValueError:
            pass

    return {
        "spike_times": spike_times,
        "spike_waveform_features": spike_waveform_features,
    }


def _get_ripple(nwb_file_name: str, epoch_name: str, ripple_detector_name: str) -> dict:
    # Ripples
    if ripple_detector_name == "Kay":
        ripple_params = "default"
    else:
        raise ValueError("Invalid ripple detector name")

    ripple_key = {
        "nwb_file_name": nwb_file_name,
        "target_interval_list_name": epoch_name + " noPrePostTrialTimes",
        "group_name": "CA1",
        "ripple_params": ripple_params,
        "filter_name": "ripple_150_250",
        "filter_sampling_rate": 1000,
    }
    lfp_band_restriction = {
        "nwb_file_name": nwb_file_name,
        "filter_name": ripple_key["filter_name"],
        "target_interval_list_name": ripple_key["target_interval_list_name"],
        "lfp_band_sampling_rate": ripple_key["filter_sampling_rate"],
    }
    lfp_band_key = (LFPBandV1 & lfp_band_restriction).fetch1("KEY")
    ripple_times = (RippleTimesV1 & ripple_key).fetch1_dataframe()
    ripple_filtered_lfp = (LFPBandV1 & lfp_band_key).fetch1_dataframe()
    lfp_merge_id = (LFPBandSelection & ripple_key).fetch1("lfp_merge_id")
    lfp_df = (LFPOutput & {"merge_id": lfp_merge_id}).fetch1_dataframe()

    ripple_consensus_trace = RippleTimesV1.get_Kay_ripple_consensus_trace(
        ripple_filtered_lfp, sampling_frequency=ripple_key["filter_sampling_rate"]
    )
    zscored_ripple_consensus_trace = zscore(ripple_consensus_trace, nan_policy="omit")

    return {
        "ripple_times": ripple_times,
        "ripple_filtered_lfps": ripple_filtered_lfp,
        "ripple_lfps": lfp_df,
        "ripple_consensus_trace": ripple_consensus_trace,
        "zscored_ripple_consensus_trace": zscored_ripple_consensus_trace,
    }


def _get_multiunit_HSE(
    spike_times: dict, speed: np.ndarray, bin_time: np.ndarray
) -> tuple:
    sampling_frequency = 1 / np.median(np.diff(bin_time))
    start_time, end_time = bin_time[[0, -1]]
    multiunit_spikes = np.stack(
        [
            np.bincount(
                np.digitize(
                    spike_times[
                        np.logical_and(
                            spike_times >= start_time, spike_times <= end_time
                        )
                    ],
                    bin_time[1:-1],
                ),
                minlength=bin_time.shape[0],
            )
            for spike_times in spike_times
        ],
        axis=1,
    )

    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_population_firing_rate(multiunit_spikes, sampling_frequency),
        index=bin_time,
        columns=["firing_rate"],
    )
    mean = np.nanmean(multiunit_firing_rate)
    std = np.nanstd(multiunit_firing_rate)

    # Avoid division by zero if firing rate is constant
    if std == 0 or np.isnan(std):
        multiunit_rate_zscore = pd.DataFrame(
            np.zeros_like(multiunit_firing_rate),
            index=bin_time,
            columns=["firing_rate"]
        )
    else:
        multiunit_rate_zscore = (multiunit_firing_rate - mean) / std

    hse_times = multiunit_HSE_detector(
        bin_time,
        multiunit_spikes,
        speed,
        sampling_frequency,
        speed_threshold=4.0,
        minimum_duration=0.015,
        zscore_threshold=2.0,
        close_event_threshold=0.000,
        use_speed_threshold_for_zscore=False,
    )

    return hse_times, multiunit_firing_rate, multiunit_rate_zscore


def _get_multiunit(spike_times: dict, position_info: pd.DataFrame) -> dict:
    # Get multiunit HSE
    hse_times = {}
    multiunit_firing_rate = {}
    multiunit_rate_zscore = {}

    for brain_area in spike_times:
        (
            hse_times[brain_area],
            multiunit_firing_rate[brain_area],
            multiunit_rate_zscore[brain_area],
        ) = _get_multiunit_HSE(
            spike_times[brain_area],
            position_info.head_speed.to_numpy(),
            position_info.index.to_numpy(),
        )

    return {
        "hse_times": hse_times,
        "multiunit_firing_rate": multiunit_firing_rate,
        "multiunit_rate_zscore": multiunit_rate_zscore,
    }


def _filter_spike_times(
    spike_times: dict, spike_waveform_features: dict, position_time: np.ndarray
) -> None:
    for brain_area, brain_area_spike_times in spike_times.items():
        filtered_spike_times = []
        filtered_spike_waveform_features = []
        for sort_ind, sort_group_spike_times in enumerate(brain_area_spike_times):

            is_in_bounds = np.logical_and(
                sort_group_spike_times >= position_time[0],
                sort_group_spike_times <= position_time[-1],
            )
            filtered_spike_times.append(sort_group_spike_times[is_in_bounds])

            try:
                filtered_spike_waveform_features.append(
                    spike_waveform_features[brain_area][sort_ind][is_in_bounds]
                )
            except KeyError:
                pass

        spike_times[brain_area] = filtered_spike_times

        if filtered_spike_waveform_features:
            spike_waveform_features[brain_area] = filtered_spike_waveform_features


def _filter_ripple_lfp_time(ripple_data: dict, position_time: np.ndarray) -> None:
    position_time_slice = slice(position_time[0], position_time[-1])
    ripple_data["ripple_filtered_lfps"] = ripple_data["ripple_filtered_lfps"].loc[
        position_time_slice
    ]
    ripple_data["ripple_lfps"] = ripple_data["ripple_lfps"].loc[position_time_slice]
    ripple_data["ripple_consensus_trace"] = ripple_data["ripple_consensus_trace"].loc[
        position_time_slice
    ]
    ripple_data["zscored_ripple_consensus_trace"] = ripple_data[
        "zscored_ripple_consensus_trace"
    ].loc[position_time_slice]


def _convert_dio_events_to_start_stop_times(
    event: pd.DataFrame, event_name: str
) -> pd.DataFrame:
    start_ind = event.astype(bool).to_numpy().nonzero()[0]
    end_ind = start_ind + 1
    end_ind[end_ind >= len(event)] = len(event) - 1

    event_times = pd.DataFrame(
        {
            "start_time": event.iloc[start_ind].index,
            "end_time": event.iloc[end_ind].index,
            "event_name": event_name,
            "event_number": np.arange(len(start_ind)) + 1,
        },
    )
    return event_times.set_index(["event_name", "event_number"])


def _load_dios(nwb_file_name: str, valid_times: tuple) -> dict:
    beam_breaks = {
        nwb_file["dio_event_name"]: pd.DataFrame(nwb_file["dio"].data)
        .set_index(pd.Index(nwb_file["dio"].timestamps, name="time"))
        .loc[valid_times[0] : valid_times[1]]
        for nwb_file in (
            DIOEvents & {"nwb_file_name": nwb_file_name} & 'dio_event_name LIKE "Poke%"'
        ).fetch_nwb()
    }
    beam_breaks = pd.concat(
        [
            _convert_dio_events_to_start_stop_times(event, event_name)
            for event_name, event in beam_breaks.items()
        ],
        axis=0,
    )

    pump_events = {
        nwb_file["dio_event_name"]: pd.DataFrame(nwb_file["dio"].data)
        .set_index(pd.Index(nwb_file["dio"].timestamps, name="time"))
        .loc[valid_times[0] : valid_times[1]]
        for nwb_file in (
            DIOEvents & {"nwb_file_name": nwb_file_name} & 'dio_event_name LIKE "Pump%"'
        ).fetch_nwb()
    }
    pump_events = pd.concat(
        [
            _convert_dio_events_to_start_stop_times(event, event_name)
            for event_name, event in pump_events.items()
        ],
        axis=0,
    )

    light_events = {
        nwb_file["dio_event_name"]: pd.DataFrame(nwb_file["dio"].data)
        .set_index(pd.Index(nwb_file["dio"].timestamps, name="time"))
        .loc[valid_times[0] : valid_times[1]]
        for nwb_file in (
            DIOEvents
            & {"nwb_file_name": nwb_file_name}
            & 'dio_event_name LIKE "Light%"'
        ).fetch_nwb()
    }
    light_events = pd.concat(
        [
            _convert_dio_events_to_start_stop_times(event, event_name)
            for event_name, event in light_events.items()
        ],
        axis=0,
    )

    return {
        "beam_breaks": beam_breaks,
        "pump_events": pump_events,
        "light_events": light_events,
    }


def _get_poke_in_poke_out_times(
    beam_breaks: pd.DataFrame, max_inter_event_interval: float = 1.0
) -> pd.DataFrame:
    merged_events = []
    for _, df in beam_breaks.groupby("event_name"):
        interevent_interval = (df.start_time.shift(-1) - df.end_time).fillna(0)
        labels = (
            (interevent_interval > max_inter_event_interval).shift(1).fillna(0).cumsum()
        )
        new_events = df.groupby(labels).agg({"start_time": "first", "end_time": "last"})
        merged_events.append(new_events)

    return (
        pd.concat(merged_events)
        .sort_values("start_time")
        .assign(
            duration=lambda x: x.end_time - x.start_time,
        )
    )


def _get_run_intervals(poke_times: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "start_time": poke_times["end_time"].to_numpy()[:-1],
                "end_time": poke_times["start_time"].shift(-1).to_numpy()[:-1],
            }
        )
        .dropna()
        .assign(
            duration=lambda x: x.end_time - x.start_time,
        )
    )


def _get_start_end_node_id(
    position_info: pd.DataFrame, track_graph: nx.Graph, times: pd.DataFrame
) -> dict:
    PROJECTED_COLS = ["projected_x_position", "projected_y_position"]

    start_time_pos = position_info.iloc[
        position_info.index.searchsorted(times.start_time.values, side="right")
    ][PROJECTED_COLS].to_numpy()
    end_time_pos = position_info.iloc[
        position_info.index.searchsorted(times.end_time.values, side="right")
    ][PROJECTED_COLS].to_numpy()

    node_attrs = nx.get_node_attributes(track_graph, "pos")
    node_pos = np.array(list(node_attrs.values()))
    node_ids = np.array(list(node_attrs.keys()))

    start_node_ids = node_ids[
        np.argmin(
            np.linalg.norm(start_time_pos[:, None, :] - node_pos[None, :, :], axis=2),
            axis=1,
        )
    ]
    end_node_ids = node_ids[
        np.argmin(
            np.linalg.norm(end_time_pos[:, None, :] - node_pos[None, :, :], axis=2),
            axis=1,
        )
    ]

    return {
        "from_well": start_node_ids,
        "to_well": end_node_ids,
    }


def _make_well_visits_df(
    beam_breaks: pd.DataFrame,
    pump_events: pd.DataFrame,
    position_info: pd.DataFrame,
    track_graph: nx.Graph,
) -> pd.DataFrame:
    poke_times = _get_poke_in_poke_out_times(beam_breaks)
    poke_times = pd.concat(
        (
            poke_times,
            pd.DataFrame(
                {
                    "start_time": [
                        position_info.index[0],
                        position_info.index[-1] - 1e-4,
                    ],
                    "end_time": [
                        position_info.index[0],
                        position_info.index[-1] - 1e-4,
                    ],
                    "duration": [0.0, 0.0],
                },
                index=[0, 0],
            ),
        )
    ).sort_values("start_time")
    poke_times = poke_times.assign(
        **_get_start_end_node_id(position_info, track_graph, poke_times)
    ).set_index(["from_well", "to_well"])

    run_times = _get_run_intervals(poke_times)
    run_times = run_times.assign(
        **_get_start_end_node_id(position_info, track_graph, run_times)
    ).set_index(["from_well", "to_well"])

    well_visits = (
        pd.concat((poke_times, run_times)).sort_values("start_time").reset_index()
    )
    well_visits["is_run"] = well_visits["from_well"] != well_visits["to_well"]
    well_visits["is_reward"] = [
        pump_events.start_time.between(row.start_time, row.end_time).any()
        for row in well_visits.itertuples()
    ]
    well_visits["is_reward"] = (
        well_visits["is_run"] & well_visits["is_reward"].shift(-1)
    ) | well_visits["is_reward"]
    well_visits["from_patch"] = well_visits["from_well"].map(WELL_PATCH_MAPPING)
    well_visits["to_patch"] = well_visits["to_well"].map(WELL_PATCH_MAPPING)
    well_visits["is_patch_change"] = (
        well_visits["from_patch"] != well_visits["to_patch"]
    )

    trial_number = np.full((len(well_visits),), np.nan)
    trial_number[well_visits.is_run] = np.arange(1, well_visits.is_run.sum() + 1)
    trial_number = (
        pd.DataFrame(trial_number)
        .fillna(method="ffill")
        .fillna(method="bfill")
        .astype(int)
        .to_numpy()
        .squeeze()
    )
    well_visits["trial_number"] = trial_number

    return well_visits


def _path_progress_for_run_segment(
    pos_info: pd.DataFrame,
    track_graph: nx.Graph,
    start_node: int,
    end_node: int,
) -> np.ndarray:

    path = nx.shortest_path(track_graph, source=start_node, target=end_node)
    path_edges = list(zip(path[:-1], path[1:]))
    path_edge_ids = [track_graph.edges[edge]["edge_id"] for edge in path_edges]

    dist = np.full(pos_info.shape[0], np.nan)
    time = pos_info.index
    add = 0.0

    if len(path_edges) == 0:
        return np.zeros(pos_info.shape[0])

    for (start_node, end_node), edge_id in zip(path_edges, path_edge_ids):
        is_edge = pos_info.track_segment_id == edge_id
        proj_pos = pos_info.loc[
            is_edge, ["projected_x_position", "projected_y_position"]
        ].to_numpy()
        start_node_pos = np.array(track_graph.nodes[start_node]["pos"])[None, :]
        dist[is_edge] = np.linalg.norm(proj_pos - start_node_pos, axis=1)
        dist[is_edge] += add
        add += np.linalg.norm(
            np.array(track_graph.nodes[start_node]["pos"])
            - np.array(track_graph.nodes[end_node]["pos"])
        )

    is_nan = np.isnan(dist)
    if np.sum(~is_nan) >= 2:
        interpolater = interp1d(
            time[~is_nan], dist[~is_nan], fill_value="extrapolate", kind="previous"
        )
        dist[is_nan] = interpolater(time[is_nan])

    # Avoid division by zero if start and end nodes are the same
    max_dist = dist.max()
    if max_dist == 0 or np.isnan(max_dist):
        return np.zeros(pos_info.shape[0])
    return dist / max_dist


def _calc_path_progress(
    well_visits: pd.DataFrame,
    position_info: pd.DataFrame,
    track_graph: nx.Graph,
) -> pd.DataFrame:
    path_progress = np.full((position_info.shape[0]), np.nan)

    for row in well_visits.itertuples():
        is_in_row = (position_info.index >= row.start_time) & (
            position_info.index < row.end_time
        )
        pos_info = position_info.loc[is_in_row]
        path_progress[is_in_row] = _path_progress_for_run_segment(
            pos_info,
            track_graph,
            row.from_well,
            row.to_well,
        )

    return pd.DataFrame(
        path_progress, index=position_info.index, columns=["path_progress"]
    ).fillna(method="ffill")


def _make_goal_node_id(well_visits, position_info):
    time = position_info.index
    goal_node_ids = np.full_like(time, fill_value=np.nan, dtype=float)
    for _, trial_df in well_visits.groupby("trial_number"):
        is_in_trial = (time >= trial_df.iloc[0].start_time) & (
            time <= trial_df.iloc[-1].end_time
        )
        goal_node_ids[is_in_trial] = trial_df.iloc[-1].to_well

    return pd.Series(goal_node_ids).fillna(method="ffill").to_numpy().astype(int)


def _make_dist_to_goal(well_visits, position_info, track_graph):

    track_segment_id_to_edge = {
        track_graph.edges[edge]["edge_id"]: edge for edge in track_graph.edges
    }

    time = position_info.index
    dist_to_goal = np.full_like(time, fill_value=np.nan, dtype=float)
    goal_node_ids = _make_goal_node_id(well_visits, position_info)

    for time_ind, (goal_node_id, row) in enumerate(
        zip(goal_node_ids, position_info.itertuples())
    ):
        track_segment_id = row.track_segment_id
        projected_position = row.projected_x_position, row.projected_y_position
        node1, node2 = track_segment_id_to_edge[track_segment_id]
        copy_graph = track_graph.copy()
        copy_graph.add_node("cur_pos", pos=projected_position)
        x1, y1 = copy_graph.nodes[node1]["pos"]
        x2, y2 = copy_graph.nodes[node2]["pos"]
        x3, y3 = copy_graph.nodes["cur_pos"]["pos"]
        left_distance = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        right_distance = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        nx.add_path(copy_graph, [node1, "cur_pos"], distance=left_distance)
        nx.add_path(copy_graph, ["cur_pos", node2], distance=right_distance)
        dist_to_goal[time_ind] = nx.shortest_path_length(
            copy_graph,
            source="cur_pos",
            target=goal_node_id,
            weight="distance",
        )

    return dist_to_goal


def _make_time_to_goal(well_visits, position_info, sampling_frequency=500):
    time = position_info.index
    n_time = len(time)

    time_to_goal = np.zeros((n_time,))

    for _, df in well_visits.loc[well_visits.is_run].groupby("trial_number"):
        is_in_trial = (time >= df.iloc[0].start_time) & (time < df.iloc[-1].end_time)
        n_time_in_trial = np.sum(is_in_trial)
        time_to_goal[is_in_trial] = np.arange(n_time_in_trial) / sampling_frequency

    return time_to_goal


def _make_turn_direction(well_visits, position_info, track_graph):
    track_edges = np.array(track_graph.edges)
    turn_direction = []
    position_info["turn_direction"] = np.full(
        (len(position_info),),
        pd.NA,
    )
    for trial in (
        well_visits.groupby("trial_number")
        .agg(
            {
                "start_time": "min",
                "end_time": "max",
                "from_well": "first",
                "to_well": "last",
            }
        )
        .itertuples()
    ):
        trial_segment_ids = position_info.loc[
            trial.start_time : trial.end_time
        ].track_segment_id.unique()  # pandas unique is not sorted
        is_bad = (
            position_info.loc[trial.start_time : trial.end_time]
            .groupby("track_segment_id")
            .size()
            < 50
        )
        trial_segment_ids = trial_segment_ids[
            ~np.isin(trial_segment_ids, is_bad.index[is_bad])
        ]

        # Ensure that the trial is in the correct traversal direction
        # track_segment_id does not contain node order information
        start_well, end_well = trial.from_well, trial.to_well
        trial_nodes = track_edges[trial_segment_ids]

        if trial_nodes[0][0] != start_well:
            trial_nodes = trial_nodes[::-1]

        for ind, (edge1, edge2) in enumerate(zip(trial_nodes[:-1], trial_nodes[1:])):
            if edge1[-1] != edge2[0]:
                trial_nodes[ind + 1] = trial_nodes[ind + 1][::-1]

        # Verify trial nodes end at the expected well
        if trial_nodes[-1][-1] != end_well:
            raise ValueError(
                f"Trial node path validation failed: expected end well {end_well}, "
                f"but got {trial_nodes[-1][-1]} for trial from well {trial.from_well} "
                f"to well {trial.to_well}. This indicates data corruption or "
                f"incorrect track graph topology."
            )

        vectors = np.array(
            [
                np.array(track_graph.nodes[node2]["pos"])
                - np.array(track_graph.nodes[node1]["pos"])
                for (node1, node2) in trial_nodes
            ]
        )
        trial_turns = np.where(
            np.cross(vectors[:-1], vectors[1:], axis=1) > 0, "left", "right"
        )
        trial_turns = "-".join(trial_turns)
        position_info.loc[trial.start_time : trial.end_time, "turn_direction"] = (
            trial_turns
        )
        turn_direction.append(trial_turns)

    turn_direction = np.array(turn_direction)


def _make_trials_df_from_well_visits(well_visits: pd.DataFrame) -> pd.DataFrame:
    trials = well_visits.groupby("trial_number").agg(
        is_reward=("is_reward", lambda x: x.sum() > 0),
        start_time=("start_time", "min"),
        end_time=("end_time", "max"),
        is_patch_change=("is_patch_change", lambda x: x.sum() > 0),
        from_well=("from_well", "first"),
        to_well=("to_well", "last"),
        from_patch=("from_patch", "first"),
        to_patch=("to_patch", "last"),
        duration=("duration", "sum"),
    )
    goal_time = (
        well_visits.loc[well_visits.from_well != well_visits.to_well]
        .groupby("trial_number")
        .agg(
            goal_arrival_time=("end_time", "min"),
        )
    )
    return trials.merge(goal_time, left_index=True, right_index=True)


def _dist_to_goal_per_time(
    track_segment_id,
    projected_x_position,
    projected_y_position,
    goal_node_id,
    track_graph,
    track_segment_id_to_edge,
):
    node1, node2 = track_segment_id_to_edge[track_segment_id]
    copy_graph = track_graph.copy()
    copy_graph.add_node("cur_pos", pos=(projected_x_position, projected_y_position))
    x1, y1 = copy_graph.nodes[node1]["pos"]
    x2, y2 = copy_graph.nodes[node2]["pos"]
    x3, y3 = copy_graph.nodes["cur_pos"]["pos"]
    left_distance = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
    right_distance = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    nx.add_path(copy_graph, [node1, "cur_pos"], distance=left_distance)
    nx.add_path(copy_graph, ["cur_pos", node2], distance=right_distance)
    return nx.shortest_path_length(
        copy_graph,
        source="cur_pos",
        target=goal_node_id,
        weight="distance",
    )


def _dist_to_goal(position_info, trial, track_graph, track_segment_id_to_edge):
    return np.array(
        [
            _dist_to_goal_per_time(
                row.track_segment_id,
                row.projected_x_position,
                row.projected_y_position,
                trial.to_well,
                track_graph,
                track_segment_id_to_edge,
            )
            for row in position_info.itertuples()
        ]
    )


def _turn_direction(trial, position_info, track_graph, track_edges):
    trial_segment_ids = position_info.loc[
        trial.start_time : trial.end_time
    ].track_segment_id.unique()  # pandas unique is not sorted
    is_bad = (
        position_info.loc[trial.start_time : trial.end_time]
        .groupby("track_segment_id")
        .size()
        < 50
    )
    trial_segment_ids = trial_segment_ids[
        ~np.isin(trial_segment_ids, is_bad.index[is_bad])
    ]

    # Ensure that the trial is in the correct traversal direction
    # track_segment_id does not contain node order information
    trial_nodes = track_edges[trial_segment_ids]

    if trial_nodes[0][0] != trial.from_well:
        trial_nodes = trial_nodes[::-1]

    for ind, (edge1, edge2) in enumerate(zip(trial_nodes[:-1], trial_nodes[1:])):
        if edge1[-1] != edge2[0]:
            trial_nodes[ind + 1] = trial_nodes[ind + 1][::-1]

    vectors = np.array(
        [
            np.array(track_graph.nodes[node2]["pos"])
            - np.array(track_graph.nodes[node1]["pos"])
            for (node1, node2) in trial_nodes
        ]
    )
    trial_turns = np.where(
        np.cross(vectors[:-1], vectors[1:], axis=1) > 0, "left", "right"
    )
    return "-".join(trial_turns)


def _make_task_variables(
    position_info: pd.DataFrame,
    trials_df: pd.DataFrame,
    track_graph: nx.Graph,
    sampling_frequency: float = 500.0,
    omission_shifts: list[int] = range(1, 6),
    patch_change_shifts: list[int] = range(-5, 6),
) -> pd.DataFrame:
    time = position_info.index
    n_time = len(time)

    trial_number = np.full((n_time,), np.nan)
    time_to_goal = np.full((n_time), np.nan)
    start_node_id = np.full((n_time), np.nan)
    goal_node_id = np.full((n_time), np.nan)
    dist_to_goal = np.full((n_time), np.nan)
    path_progress = np.full((n_time), np.nan)
    turn_direction = np.full((n_time), pd.NA)
    n_shifts = len(omission_shifts)
    is_prev_reward_omission_pos = np.full((n_time, n_shifts), False)

    track_edges = np.array(track_graph.edges)

    track_segment_id_to_edge = {
        track_graph.edges[edge]["edge_id"]: edge for edge in track_graph.edges
    }
    is_prev_reward_omission = pd.concat(
        [
            (~trials_df.is_reward)
            .shift(shift)
            .fillna(False)
            .rename(f"is_no_reward_{shift}")
            for shift in omission_shifts
        ],
        axis=1,
    )
    is_patch_change = pd.concat(
        [
            trials_df.is_patch_change.shift(shift)
            .fillna(False)
            .rename(f"is_patch_change_{shift}")
            for shift in patch_change_shifts
        ],
        axis=1,
    )
    n_shifts = len(patch_change_shifts)
    is_patch_change_pos = np.full((n_time, n_shifts), False)

    for trial in trials_df.itertuples():
        is_in_trial = (time >= trial.start_time) & (time < trial.end_time)
        from_start_time_to_goal = (time >= trial.start_time) & (
            time < trial.goal_arrival_time
        )
        from_goal_to_next_movement = (time >= trial.goal_arrival_time) & (
            time < trial.end_time
        )
        trial_number[is_in_trial] = trial.Index

        time_to_goal[from_start_time_to_goal] = (
            np.arange(np.sum(from_start_time_to_goal)) / sampling_frequency
        )
        time_to_goal[from_goal_to_next_movement] = 0.0

        start_node_id[is_in_trial] = trial.from_well
        goal_node_id[is_in_trial] = trial.to_well
        turn_direction[is_in_trial] = _turn_direction(
            trial, position_info.loc[is_in_trial], track_graph, track_edges
        )
        path_progress[from_start_time_to_goal] = _path_progress_for_run_segment(
            position_info.loc[from_start_time_to_goal],
            track_graph,
            trial.from_well,
            trial.to_well,
        )
        path_progress[from_goal_to_next_movement] = 0.0

        dist_to_goal[is_in_trial] = _dist_to_goal(
            position_info.loc[is_in_trial],
            trial,
            track_graph,
            track_segment_id_to_edge,
        )
        is_prev_reward_omission_pos[is_in_trial] = is_prev_reward_omission.loc[
            trial.Index
        ].to_numpy()
        is_patch_change_pos[is_in_trial] = is_patch_change.loc[trial.Index].to_numpy()

    return pd.DataFrame(
        {
            "trial_number": trial_number,
            "time_to_goal": time_to_goal,
            "start_node_id": start_node_id,
            "goal_node_id": goal_node_id,
            "dist_to_goal": dist_to_goal,
            "path_progress": path_progress,
            "turn_direction": turn_direction,
            **{
                f"is_no_reward_{shift}": is_reward_omission
                for is_reward_omission, shift in zip(
                    is_prev_reward_omission_pos.T, omission_shifts
                )
            },
            **{
                f"is_patch_change{shift}": is_patch_change
                for is_patch_change, shift in zip(
                    is_patch_change_pos.T, patch_change_shifts
                )
            },
        },
        index=time,
    ).fillna(method="ffill")


def load_data(
    nwb_file_name: str,
    epoch_name: str,
    ripple_detector_name: str = "Kay",
) -> dict[pd.DataFrame]:

    electrode_group_info = _get_electrode_group_info(nwb_file_name)

    pos_name = (
        PositionIntervalMap
        & {"nwb_file_name": nwb_file_name, "interval_list_name": epoch_name}
    ).fetch1("position_interval_name")
    position_data = _get_position_info(nwb_file_name, epoch_name, pos_name)

    spike_data = _get_spike_data(nwb_file_name)
    # Filter spike times by the start and end time of the position data
    _filter_spike_times(
        spike_data["spike_times"],
        spike_data["spike_waveform_features"],
        position_data["position_info"].index,
    )

    ripple_data = _get_ripple(nwb_file_name, epoch_name, ripple_detector_name)
    _filter_ripple_lfp_time(ripple_data, position_data["position_info"].index)

    dio_events = _load_dios(
        nwb_file_name, position_data["position_info"].index.to_numpy()[[0, -1]]
    )
    well_visits = _make_well_visits_df(
        dio_events["beam_breaks"],
        dio_events["pump_events"],
        position_data["position_info"],
        track_graph=position_data["track_graph"],
    )
    trials_df = _make_trials_df_from_well_visits(well_visits)

    task_variables_df = _make_task_variables(
        position_info=position_data["position_info"],
        trials_df=trials_df,
        track_graph=position_data["track_graph"],
    )

    return {
        **position_data,
        **spike_data,
        **ripple_data,
        **_get_theta(nwb_file_name, epoch_name),
        **_get_multiunit(spike_data["spike_times"], position_data["position_info"]),
        **dio_events,
        "electrode_group_info": electrode_group_info,
        "well_visits": well_visits,
        "trials": trials_df,
        "task_variables": task_variables_df,
    }


def get_labels(times: pd.DataFrame, time: pd.Index) -> pd.DataFrame:
    event_labels = pd.DataFrame(
        np.zeros_like(time, dtype=int), index=time, columns=["event_number"]
    )
    # This means the label will be +1 because label count starts at 0
    for i, row in enumerate(times.itertuples()):
        event_labels.loc[row.start_time : row.end_time, :] = i + 1

    return event_labels


def get_training_timepoints(
    data: dict[pd.DataFrame], training_type: str = "no_ripple"
) -> np.ndarray:
    if training_type == "all":
        n_time = len(data["position_info"].index)
        return np.ones((n_time,), dtype=bool)

    elif training_type == "no_ripple":
        ripple_labels = get_labels(
            data["ripple_times"], data["position_info"].index.to_numpy()
        )
        is_ripple = ripple_labels > 0
        not_ripple = ~np.asarray(is_ripple).squeeze()

        return not_ripple
