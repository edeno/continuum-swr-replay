"""Tests for the held-out tetrode score's pieces."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence import tetrode_holdout as th

STATES = ["Local", "No-Spike", "Non-Local Continuous", "Non-Local Fragmented"]
N_BINS = 3
# Local (3 bins), No-Spike (1), Continuous (3), Fragmented (3)
STATE_IND = np.array([0, 0, 0, 1, 2, 2, 2, 3, 3, 3])


def test_time_blocks_are_contiguous_and_of_equal_duration():
    time = np.arange(100) * 0.01
    blocks = th.time_blocks(time, 5)
    assert blocks[0] == 0 and blocks[-1] == 4
    assert (np.diff(blocks) >= 0).all()
    np.testing.assert_array_equal(np.bincount(blocks), [20, 20, 20, 20, 20])


def test_fold_training_masks_leave_out_the_block_and_its_margin():
    blocks = np.repeat([0, 1, 2], 10)
    is_training = np.ones(30, dtype=bool)
    is_training[25] = False
    masks = th.fold_training_masks(is_training, blocks, 3, margin_bins=2)
    expected = np.ones(30, dtype=bool)
    expected[8:22] = False  # block 1 is bins 10-19, two bins either side
    expected[25] = False
    np.testing.assert_array_equal(masks[1], expected)
    # block 0's mask keeps block 1 and 2 beyond the margin
    assert not masks[0][:12].any() and masks[0][12:25].all()


def _terms() -> th.FoldTerms:
    """Two blocks of two bins, two tetrodes, three place bins."""
    ground = np.array(
        [
            [[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]],  # block 0
            [[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]],  # block 1
        ]
    )
    log_intensity = np.array([[0.1, 0.2, 0.3], [1.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    return th.FoldTerms(
        ground=ground,
        log_intensity=log_intensity,
        time_bin=np.array([0, 0, 3]),
        tetrode=np.array([0, 1, 0]),
        block=np.array([0, 0, 1, 1]),
    )


def test_fold_terms_use_each_bins_block_and_sum_to_the_total():
    terms = _terms()
    tetrode_0 = terms.tetrode_terms(0)
    # bin 0 (block 0): -ground + its spike; bin 3 (block 1): -ground + its spike
    np.testing.assert_allclose(tetrode_0[0], [-0.9, -1.8, -2.7])
    np.testing.assert_allclose(tetrode_0[3], [-2.0, -2.0, 3.0])
    np.testing.assert_allclose(tetrode_0[1], [-1.0, -2.0, -3.0])
    np.testing.assert_allclose(
        terms.tetrode_terms(0) + terms.tetrode_terms(1), terms.total()
    )
    np.testing.assert_array_equal(
        terms.spike_counts(), [[1, 1], [0, 0], [0, 0], [1, 0]]
    )


def test_state_log_likelihood_shares_the_spatial_term_and_adds_the_kernel_to_local():
    spatial = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    kernel = np.full((2, 3), 0.5)
    out = th.state_log_likelihood(
        spatial,
        np.array([-1.0, -2.0]),
        kernel,
        STATE_IND,
        STATES,
        np.array([False, True]),
    )
    np.testing.assert_allclose(out[0], [1.5, 2.5, 3.5, -1.0, 1, 2, 3, 1, 2, 3])
    np.testing.assert_array_equal(out[1], 0.0)  # missing


def test_spatial_distribution_sums_the_spatial_states_and_leaves_out_no_spike():
    posterior = np.array(
        [
            # Local at bin 0, No-Spike 0.5, Continuous at bin 2
            [0.25, 0, 0, 0.5, 0, 0, 0.25, 0, 0, 0],
            # all No-Spike
            [0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0],
        ]
    )
    distribution = th.spatial_distribution(posterior, STATE_IND, STATES)
    np.testing.assert_allclose(distribution[0], [0.5, 0.0, 0.5])
    assert np.isnan(distribution[1]).all()


def test_predictive_log_score_averages_the_likelihood_over_the_weights():
    terms = np.log(np.array([[0.2, 0.4, 0.8]]))
    score = th.predictive_log_score(terms, np.array([[0.5, 0.0, 0.5]]))
    assert score[0] == pytest.approx(np.log(0.5 * 0.2 + 0.5 * 0.8))
    assert np.isnan(th.predictive_log_score(terms, np.full((1, 3), np.nan)))[0]


def test_window_rows_and_sums():
    rows = th.window_rows(np.array([2, 5, 3]), np.array([4, 6, 3]))
    np.testing.assert_array_equal(rows, [2, 3, 4, 5, 6])
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(
        th.window_sums(values, rows, np.array([2, 5, 3]), np.array([4, 6, 3])),
        [6.0, 9.0, 2.0],
    )


def test_window_scores_sum_over_tetrodes_and_divide_by_the_windows_spikes():
    rows = pd.DataFrame(
        {
            "epoch_id": "e1",
            "animal": "a",
            "window_id": [0, 0, 1, 1],
            "event_id": [7, 7, 7, 7],
            "kind": ["event", "event", "control", "control"],
            "group": "no SWR",
            "gain": [3.0, 1.0, 0.5, -0.5],
            "gain_silent": [-1.0, 0.0, 0.0, 0.0],
            "agreement": [1.0, 1.0, 0.0, 0.0],
            "agreement_silent": [0.0, 0.0, 0.0, 0.0],
            "n_spikes": [3, 1, 0, 0],
            "spread_cm": [40.0, 60.0, 5.0, 7.0],
            "nonlocal_prob": [0.9, 1.0, 0.0, 0.0],
        }
    )
    windows = th.window_scores(rows).set_index("window_id")
    event = windows.loc[0]
    assert event["gain_per_spike"] == pytest.approx(4.0 / 4)
    # the spiking bins' gain: 4 - (-1) = 5 over 4 spikes
    assert event["spiking_gain_per_spike"] == pytest.approx(5.0 / 4)
    assert event["agreement_per_spike"] == pytest.approx(0.5)
    assert event["spread_cm"] == pytest.approx(50.0)
    assert np.isnan(windows.loc[1, "gain_per_spike"])
    assert windows.loc[1, "gain"] == pytest.approx(0.0)
