"""The examples' memoryless windows, recomputed and checked against the stored ones:
a center may move only to a tie, and a tie does not move a candidate's dispersion."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.inventory_comparison import example_windows


def _one_candidate_epoch(monkeypatch, middle_posterior):
    """An epoch of four windows on a straight 5-bin track (bins 2 cm apart), one
    stored candidate over windows 0-2, and the recomputation's decode replaced
    by the given posteriors. The stored decisions are the recomputed ones with
    window 1's center moved to the other bin of the middle posterior's pair, as
    another device's rounding would; the stored dispersion reads that center."""
    from continuum_swr_replay.analysis.detection.memoryless_sequence.config import (
        MemorylessSequenceConfig,
    )
    from continuum_swr_replay.analysis.detection.memoryless_sequence.detect import (
        WindowDecisions,
        forward_jumps_cm,
        sequence_dispersion_cm,
        window_decisions,
    )

    position = np.arange(5) * 2.0
    distance = np.abs(position[:, None] - position[None, :])
    posterior = np.zeros((4, 5))
    posterior[0, 0] = 1.0
    posterior[1] = middle_posterior
    posterior[2:, 2] = 1.0
    config = MemorylessSequenceConfig()
    computed = window_decisions(posterior, distance, np.zeros(4), config)
    pair = np.flatnonzero(middle_posterior)
    center = computed.center_index.copy()
    center[1] = pair[pair != center[1]][0]
    stored = WindowDecisions(
        center_index=center,
        spread_cm=computed.spread_cm,
        jump_cm=forward_jumps_cm(center, distance),
        gates=computed.gates,
    )
    runs = np.array([[0, 2]])
    candidates = SimpleNamespace(
        config=config,
        midpoint_s=np.arange(4) * config.stride_s,
        decisions=stored,
        runs=runs,
        pre_merge_runs=runs,
        pieces=(np.array([0]),),
        dispersion_cm=sequence_dispersion_cm(runs, stored.gates.kept, center, distance),
    )
    inputs = SimpleNamespace(
        identifier="sim20200101_02_r1_HPC",
        candidates=candidates,
        time=np.arange(4) * config.stride_s,
        dot=pd.DataFrame({"head_speed": np.zeros(4)}),
        spike_times=None,
        spike_waveform_features=None,
        encoding_model=None,
        bin_distance_cm=distance,
        missing_windows=np.zeros(4, dtype=bool),
    )
    with np.errstate(divide="ignore"):
        log_posterior = np.log(posterior)
    monkeypatch.setattr(
        example_windows,
        "candidate_window_log_likelihood",
        lambda first, stop, *args: log_posterior[first : stop + 1].copy(),
    )
    return inputs


def test_a_center_moved_to_a_tie_does_not_move_the_candidates_dispersion(
    monkeypatch,
) -> None:
    """Window 1's posterior is split evenly over bins 1 and 2, so either is its
    center; the stored run took the other one. The dispersion it reads is the
    stored center's, so the candidate checks out and the window counts as a
    tie."""
    inputs = _one_candidate_epoch(monkeypatch, np.array([0, 0.5, 0.5, 0, 0]))

    _, _, check = example_windows.recompute_windows(inputs, 0, 2)

    assert check["center_ties"] == 1
    (record,) = check["candidates"]
    assert record["dispersion_cm"][0] == record["dispersion_cm"][1]


def test_a_center_moved_off_a_tie_still_refuses_the_figure(monkeypatch) -> None:
    """Window 1's posterior favors bin 1 (0.6 against 0.4), so a stored center
    at bin 2 is no tie: the recomputation is refused."""
    inputs = _one_candidate_epoch(monkeypatch, np.array([0, 0.6, 0.4, 0, 0]))

    with pytest.raises(ValueError, match="1 centers moved off a tie"):
        example_windows.recompute_windows(inputs, 0, 2)
