"""Trial folds of the mPFC fit bins, with the kernel's training margin."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.heldout import assign_trial_folds
from continuum_swr_replay.analysis.regression.folds import (
    task_segment_starts,
    trial_folds,
)

STEP_S = 0.002
KERNEL_HALF_WIDTH_S = 0.25


@pytest.fixture
def session():
    """Ten minutes at 500 Hz; a trial every 12 s to one of three wells, and one
    trial of 80 s; the animal is fit-eligible in the last 4 s of each trial
    (its stay at the well), broken by a 0.3 s gap in the middle of the stay,
    and for the last 70 s of the long trial. The long stay is cut into pieces
    (``MAX_SEGMENT_S``), which may fall in different folds: only there is a
    training bin within a kernel's reach of a held-out bin."""
    time = np.arange(300_000) * STEP_S
    departures = np.r_[np.arange(2.0, 300.0, 12.0), np.arange(374.0, 590.0, 12.0)]
    trials = pd.DataFrame(
        {
            "movement_start_time": departures,
            "end_time": departures + 12.0,
            "to_well": np.arange(departures.size) % 3,
        }
    )
    trials.loc[trials.index[-1], "end_time"] = 598.0
    fit = np.zeros(time.size, dtype=bool)
    for end in departures + 12.0:
        fit |= (time >= end - 4.0) & (time < end)
        fit &= ~((time >= end - 2.15) & (time < end - 1.85))
    fit |= (time >= 304.0) & (time < 374.0)
    return time, fit, trials


def test_every_fit_bin_is_in_one_fold(session):
    time, fit, trials = session
    result = assign_trial_folds(time, fit, trials)
    assert result.folds.shape == (fit.sum(),)
    assert set(np.unique(result.folds)) == set(range(5))
    assert sum(result.counts["fold_sizes"]) == fit.sum()


def test_the_folds_are_the_shared_rule_on_the_fit_bins(session):
    time, fit, trials = session
    starts, classes = task_segment_starts(
        time,
        trials["movement_start_time"].to_numpy(),
        trials["to_well"].to_numpy(),
        float(trials["end_time"].iloc[-1]),
    )
    np.testing.assert_array_equal(
        assign_trial_folds(time, fit, trials).folds,
        trial_folds(time[fit], starts, classes),
    )


def test_a_symmetric_kernel_on_a_training_bin_sees_no_held_out_bin(session):
    time, fit, trials = session
    result = assign_trial_folds(time, fit, trials, margin_s=KERNEL_HALF_WIDTH_S)
    fit_time = time[fit]
    for fold, train in enumerate(result.training):
        held_out = np.sort(fit_time[result.folds == fold])
        trained = fit_time[train]
        # held-out bins inside [t - half width, t + half width] of each training bin
        first = np.searchsorted(held_out, trained - KERNEL_HALF_WIDTH_S, side="left")
        last = np.searchsorted(held_out, trained + KERNEL_HALF_WIDTH_S, side="right")
        assert (last == first).all(), fold
        assert not (train & (result.folds == fold)).any()


def test_the_margin_removes_only_the_bins_near_the_fold(session):
    time, fit, trials = session
    result = assign_trial_folds(time, fit, trials)
    assert result.margin_bins == 125
    fit_time = time[fit]
    for fold, train in enumerate(result.training):
        held_out = np.sort(fit_time[result.folds == fold])
        left_out = (result.folds != fold) & ~train
        assert result.counts["n_margin_bins"][fold] == left_out.sum()
        if not left_out.any():
            continue
        # every bin left out lies within the margin of a held-out bin
        nearest = np.minimum(
            np.abs(
                fit_time[left_out]
                - held_out[
                    np.clip(np.searchsorted(held_out, fit_time[left_out]) - 1, 0, None)
                ]
            ),
            np.abs(
                held_out[
                    np.clip(
                        np.searchsorted(held_out, fit_time[left_out]),
                        0,
                        held_out.size - 1,
                    )
                ]
                - fit_time[left_out]
            ),
        )
        assert (nearest <= 125 * STEP_S + 1e-9).all()
    # the long stay's pieces fall in different folds, so the margin binds
    assert sum(result.counts["n_margin_bins"]) > 0


def test_the_folds_need_no_seed(session):
    time, fit, trials = session
    first = assign_trial_folds(time, fit, trials)
    again = assign_trial_folds(time, fit, trials.sample(frac=1.0, random_state=0))
    np.testing.assert_array_equal(first.folds, again.folds)
