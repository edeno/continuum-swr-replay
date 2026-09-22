"""The whole memoryless detector over one epoch, from spikes to table.

Arrays in, tables out, with no file access, so the same function runs under
the tests and under the per-epoch runner. The order is fixed here rather than
left to callers: decode every window under a uniform prior, gate and assemble
sequences, shuffle every candidate that meets the replay criteria, as the
reference tests every such candidate, and write one row per candidate.

The design is in ``docs/design/replay_detection_comparison/02-memoryless-sequence.md``,
section "Data flow, per epoch".
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.detect import (
    CandidateSequences,
    detect_candidate_sequences,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.event_table import (
    MEMORYLESS_EVENT_TABLE_DTYPES,
    SequenceIdentity,
    build_memoryless_event_table,
    build_shuffle_draw_table,
    shuffle_eligible,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    clusterless_epoch_log_likelihood,
    interior_bin_distances_cm,
    window_midpoints_s,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.shuffle import (
    DEFAULT_N_SHUFFLES,
    ShuffleSeed,
    shuffle_candidate,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.statistics import (
    aggregate_windows,
    normalize_windows,
)

__all__ = ["EpochDetection", "run_epoch"]


@dataclass(frozen=True, eq=False)
class EpochDetection:
    """What one epoch yields.

    Attributes
    ----------
    identity : SequenceIdentity
        Which epoch, and which run; kept here so an epoch with no candidate
        is still identifiable.
    candidates : CandidateSequences
        Every window's statistics and gates, and the runs before and after
        merging: the diagnostics that explain each rejection.
    table : DataFrame
        One row per candidate, from ``build_memoryless_event_table``. Row
        ``i`` is candidate ``i``.
    shuffle_draws : DataFrame
        Every shuffled draw's statistics, from ``build_shuffle_draw_table``;
        one row per draw of every shuffled candidate, keyed on the table.

    """

    identity: SequenceIdentity
    candidates: CandidateSequences
    table: pd.DataFrame
    shuffle_draws: pd.DataFrame

    def __post_init__(self) -> None:
        if len(self.table) != len(self.candidates.runs):
            raise ValueError(
                f"{len(self.table)} table rows for {len(self.candidates.runs)} "
                "candidates; row i must be candidate i."
            )
        if list(self.table.columns) != list(MEMORYLESS_EVENT_TABLE_DTYPES):
            raise ValueError("The table does not carry the declared columns in order.")
        keyed = set(self.shuffle_draws.ms_event_id)
        if not keyed <= set(self.table.ms_event_id):
            raise ValueError("A shuffle draw is keyed to an event not in the table.")
        drawn = self.shuffle_draws.groupby("ms_event_id").size()
        declared = self.table.set_index("ms_event_id").n_shuffles
        if not drawn.eq(declared.loc[drawn.index]).all():
            raise ValueError(
                "The draws per event disagree with the table's n_shuffles."
            )


def run_epoch(
    time: NDArray,
    linear_position: NDArray,
    speed_cm_per_s: NDArray,
    spike_times: list[NDArray],
    spike_waveform_features: list[NDArray],
    encoding_model: dict,
    identity: SequenceIdentity,
    *,
    swr_intervals: pd.DataFrame | None,
    swr_observed_intervals: NDArray | None,
    swr_definition: str,
    seed: ShuffleSeed | None,
    config: MemorylessSequenceConfig,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
) -> EpochDetection:
    """Detect, test and tabulate the candidate sequences of one epoch.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        The decode's uniform grid.
    linear_position : ndarray, shape (n_time,) or (n_time, 1)
    speed_cm_per_s : ndarray, shape (n_time,)
        The animal's speed on the same grid. Its value at each window midpoint
        gates the window; every sample between a sequence's endpoints decides
        whether the sequence is still.
    spike_times, spike_waveform_features : list of ndarray
        Per electrode, in the encoding model's order.
    encoding_model : dict
        From ``load_clusterless_encoding_model``.
    identity : SequenceIdentity
    swr_intervals, swr_observed_intervals, swr_definition
        As for ``build_memoryless_event_table``.
    seed : ShuffleSeed or None
        Its ``session`` must be ``identity.epoch_id``, so a manifest that says
        the draws are keyed on the epoch is telling the truth. None runs no
        shuffle, and the table then records every candidate as untested.
    config : MemorylessSequenceConfig
    n_shuffles : int, default 100

    Returns
    -------
    detection : EpochDetection

    """
    time = np.asarray(time, dtype=np.float64)
    speed = np.asarray(speed_cm_per_s, dtype=np.float64)
    if speed.shape != time.shape:
        raise ValueError(
            f"speed_cm_per_s {speed.shape} must be on the decode grid {time.shape}."
        )
    if seed is not None and seed.session != identity.epoch_id:
        raise ValueError(
            f"The shuffle seed names session {seed.session!r} but the epoch is "
            f"{identity.epoch_id!r}; draws must be keyed on the epoch they test."
        )

    likelihood = clusterless_epoch_log_likelihood(
        time,
        linear_position,
        spike_times,
        spike_waveform_features,
        encoding_model,
        config,
    )
    posterior = normalize_windows(
        aggregate_windows(
            likelihood.log_likelihood,
            config.window_n_samples,
            config.stride_n_samples,
        )
    )
    midpoint_s = window_midpoints_s(likelihood.time, config)
    bin_distance_cm = interior_bin_distances_cm(encoding_model["environment"])
    candidates = detect_candidate_sequences(
        posterior,
        midpoint_s,
        bin_distance_cm,
        np.interp(midpoint_s, time, speed),
        config,
    )

    if seed is None:
        shuffle_results = None
    else:
        shuffle_results = {
            int(index): shuffle_candidate(
                int(candidates.runs[index, 0]),
                int(candidates.runs[index, 1]),
                time,
                spike_times,
                spike_waveform_features,
                encoding_model,
                bin_distance_cm,
                config,
                seed=seed,
                n_shuffles=n_shuffles,
            )
            for index in shuffle_eligible(candidates)
        }

    table = build_memoryless_event_table(
        candidates,
        identity,
        speed_time_s=time,
        speed_cm_per_s=speed,
        bin_distance_cm=bin_distance_cm,
        swr_intervals=swr_intervals,
        swr_observed_intervals=swr_observed_intervals,
        swr_definition=swr_definition,
        shuffle_results=shuffle_results,
    )
    return EpochDetection(
        identity=identity,
        candidates=candidates,
        table=table,
        shuffle_draws=build_shuffle_draw_table(table, shuffle_results or {}),
    )
