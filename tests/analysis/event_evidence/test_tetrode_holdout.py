"""Tests for the held-out tetrode score's pieces."""

import numpy as np
import pandas as pd
import pytest
import sklearn

from continuum_swr_replay.analysis.event_evidence import tetrode_holdout as th
from continuum_swr_replay.analysis.regression import folds as fo
from tests.support._time_blocks import TIME_BLOCK_MARGIN_BINS, time_blocks

STATES = ["Local", "No-Spike", "Non-Local Continuous", "Non-Local Fragmented"]
N_BINS = 3
# Local (3 bins), No-Spike (1), Continuous (3), Fragmented (3)
STATE_IND = np.array([0, 0, 0, 1, 2, 2, 2, 3, 3, 3])


def test_fold_blocks_are_the_given_folds_checked():
    time = np.arange(10) * 0.5
    folds = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    np.testing.assert_array_equal(th.fold_blocks(time, 2, folds), folds)
    for bad in (folds[:-1], np.zeros(10, dtype=int), folds + 1):
        with pytest.raises(ValueError, match="folds must give each of the"):
            th.fold_blocks(time, 2, bad)


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


def _convolved_masks(is_training, blocks, n_folds, margin_bins):
    """The masks as ``fold_training_masks`` computed them before it took
    interleaved folds: a convolution of each block's indicator."""
    width = 2 * margin_bins + 1
    return [
        is_training
        & ~(
            np.convolve(
                (blocks == b).astype(np.int64), np.ones(width, np.int64), "same"
            )
            > 0
        )
        for b in range(n_folds)
    ]


@pytest.mark.parametrize("margin_bins", [0, 2, 7])
def test_fold_training_masks_match_the_earlier_convolution(margin_bins):
    rng = np.random.default_rng(0)
    is_training = rng.random(500) < 0.7
    for blocks in (time_blocks(np.arange(500.0), 5), rng.integers(0, 5, 500)):
        for got, expected in zip(
            th.fold_training_masks(is_training, blocks, 5, margin_bins),
            _convolved_masks(is_training, blocks, 5, margin_bins),
            strict=True,
        ):
            np.testing.assert_array_equal(got, expected)


def test_no_training_bin_of_an_interleaved_fold_is_within_the_margin_of_it():
    """Folds of many short segments: every training bin a fold keeps is more
    than the margin from each of the fold's bins, and every other training
    bin outside the fold is kept."""
    rng = np.random.default_rng(1)
    lengths = rng.integers(5, 40, 60)
    blocks = np.repeat(rng.integers(0, 4, lengths.size), lengths)
    is_training = rng.random(blocks.size) < 0.8
    margin = 6
    index = np.arange(blocks.size)
    for b, mask in enumerate(th.fold_training_masks(is_training, blocks, 4, margin)):
        distance = np.abs(index[:, None] - index[blocks == b][None, :]).min(axis=1)
        np.testing.assert_array_equal(mask, is_training & (distance > margin))


def test_margin_in_bins_rounds_seconds_up_to_bins():
    time = 100.0 + np.arange(1000) / 500.0
    assert th.margin_in_bins(1.0, time) == 500
    assert th.margin_in_bins(0.003, time) == 2


def _trials(lengths_s, classes, bin_s=0.1):
    """A time axis and the segment starts of back-to-back trials."""
    starts = np.r_[0.0, np.cumsum(lengths_s)[:-1]]
    time = np.arange(0.0, float(np.sum(lengths_s)), bin_s)
    return time, starts, np.asarray(classes)


def test_place_sampling_counts_the_folds_training_seconds_at_the_place():
    # two folds of three bins; places 0, 1, 1 | 1, 2, -1 (unknown)
    place = np.array([0, 1, 1, 1, 2, -1])
    blocks = np.array([0, 0, 0, 1, 1, 1])
    is_training = np.array([True, True, True, True, True, False])
    sampling = th.place_sampling(
        place, th.fold_training_masks(is_training, blocks, 2, 0), blocks, 0.5
    )
    assert sampling.dtype == np.float32
    # fold 0 is fit on bins 3-4 (places 1, 2); fold 1 on bins 0-2 (0, 1, 1)
    np.testing.assert_array_equal(sampling[:5], [0.0, 0.5, 0.5, 1.0, 0.0])
    assert np.isnan(sampling[5])
    # a margin of one bin drops bin 3 from fold 0's fit and bin 2 from fold 1's
    np.testing.assert_array_equal(
        th.place_sampling(
            place, th.fold_training_masks(is_training, blocks, 2, 1), blocks, 0.5
        )[:5],
        [0.0, 0.0, 0.0, 0.5, 0.0],
    )


def test_trial_folds_sample_a_patch_that_contiguous_blocks_leave_unsampled():
    """Why the folds are made of trials. The animal runs between the wells of
    one patch (places 0-9) for the first fifth of the epoch and of another
    (places 10-19) for the rest. Each contiguous block is fit on the other
    blocks, so the first block's encoding never saw the first patch; trial
    folds, balanced by destination, put first-patch trials in every fold."""
    bin_s = 0.1
    n_trials = 50
    lengths = np.full(n_trials, 10.0)
    classes = np.where(np.arange(n_trials) < 10, 0, 1)  # destination patch
    time, starts, _ = _trials(lengths, classes, bin_s)
    segment = fo.trial_segments(time, starts)
    # within each trial the animal sweeps its patch's ten places
    within = ((time - starts[segment]) / 10.0 * 10).astype(np.int64).clip(0, 9)
    place = within + 10 * classes[segment]
    is_training = np.ones(time.size, dtype=bool)
    first_patch = place < 10

    blocks = time_blocks(time, 5)
    contiguous = th.place_sampling(
        place,
        th.fold_training_masks(is_training, blocks, 5, TIME_BLOCK_MARGIN_BINS),
        blocks,
        bin_s,
    )
    assert (contiguous[first_patch] == 0).all()

    folds = fo.trial_folds(time, starts, classes, 5)
    margin = th.margin_in_bins(th.TRIAL_MARGIN_S, time)
    trial = th.place_sampling(
        place, th.fold_training_masks(is_training, folds, 5, margin), folds, bin_s
    )
    assert (trial > 0).all()
    # eight first-patch trials outside each fold, about 1 s at each place
    assert trial[first_patch].min() >= 5.0


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


def test_add_rows_is_np_add_at_to_the_bit():
    rng = np.random.default_rng(0)
    index = rng.integers(0, 50, size=2000)
    values = rng.normal(size=(2000, 7)).astype(np.float32)
    expected = rng.normal(size=(50, 7))
    out = expected.copy()
    np.add.at(expected, index, values)
    th._add_rows(out, index, values)
    np.testing.assert_array_equal(out, expected)


def test_the_leakage_check_passes_time_blocks_and_trial_folds():
    rng = np.random.default_rng(3)
    time = np.arange(0.0, 300.0, 0.002)
    position = rng.uniform(0, 100, time.size)
    is_training = rng.random(time.size) < 0.9
    blocks = time_blocks(time, 5)
    th.check_no_fold_leakage(
        time,
        position,
        th.fold_training_masks(is_training, blocks, 5, TIME_BLOCK_MARGIN_BINS),
        blocks,
    )
    starts = np.arange(0.0, 300.0, 7.0)
    folds = fo.trial_folds(time, starts, np.arange(starts.size) % 3, 5)
    margin = th.margin_in_bins(th.TRIAL_MARGIN_S, time)
    th.check_no_fold_leakage(
        time, position, th.fold_training_masks(is_training, folds, 5, margin), folds
    )


class _Stop(Exception):
    """Raised once the encodings are fit: the masks are all the test needs."""


def _fit_masks(monkeypatch, is_training, **options) -> list[np.ndarray]:
    """The training mask of each fold's encoding fit in ``holdout_scores``,
    on a stand-in decoder (three place bins), stopped after the fits."""
    from types import SimpleNamespace

    from continuum_swr_replay.decoding.results import ENCODING_KEY

    environment = SimpleNamespace(
        is_track_interior_=np.ones(N_BINS, dtype=bool),
        place_bin_centers_=np.arange(N_BINS, dtype=float)[:, None],
    )
    detector = SimpleNamespace(
        encoding_model_={ENCODING_KEY: {"environment": environment}},
        state_ind_=STATE_IND,
        is_track_interior_state_bins_=np.ones(STATE_IND.size, dtype=bool),
        state_names=STATES,
    )
    masks = []

    def refit(det, time, position, spike_times, marks, mask):
        masks.append(np.asarray(mask).copy())
        return {}

    def fold_terms(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(th, "fold_terms", fold_terms)
    n_time = is_training.size
    time = np.arange(n_time) * 0.002
    with pytest.raises(_Stop):
        th.holdout_scores(
            detector,
            time,
            np.linspace(0.0, 2.0, n_time)[:, None],
            np.linspace(0.0, 2.0, n_time),
            [np.array([0.01])],
            [np.ones((1, 4))],
            is_training,
            np.zeros(n_time, dtype=bool),
            pd.DataFrame({"window_id": [0], "s": [0], "e": [1]}),
            refit=refit,
            log=lambda message: None,
            **options,
        )
    return masks


def test_holdout_scores_fit_each_fold_on_the_folds_and_margin_it_is_given(
    monkeypatch,
):
    """Interleaved folds (segments of 40 bins dealt to the five folds in turn)
    and a margin of 7 bins reach the encodings' fit; other folds and margin
    (five time blocks, a margin of two bins) give other fits."""
    rng = np.random.default_rng(4)
    is_training = rng.random(1000) < 0.8
    folds = (np.arange(1000) // 40) % fo.N_FOLDS
    given = _fit_masks(monkeypatch, is_training, folds=folds, margin_bins=7)
    expected = th.fold_training_masks(is_training, folds, fo.N_FOLDS, 7)
    assert len(given) == fo.N_FOLDS
    for got, mask in zip(given, expected, strict=True):
        np.testing.assert_array_equal(got, mask)

    blocks = time_blocks(np.arange(1000) * 0.002)
    other_fits = _fit_masks(
        monkeypatch, is_training, folds=blocks, margin_bins=TIME_BLOCK_MARGIN_BINS
    )
    expected = th.fold_training_masks(
        is_training, blocks, fo.N_FOLDS, TIME_BLOCK_MARGIN_BINS
    )
    assert len(other_fits) == fo.N_FOLDS
    for got, mask, other in zip(other_fits, expected, given, strict=True):
        np.testing.assert_array_equal(got, mask)
        assert not np.array_equal(got, other)


def test_the_leakage_check_raises_where_the_library_would_reach_a_fold():
    time = np.arange(100) * 0.002
    position = np.linspace(0, 50, 100)
    is_training = np.ones(100, dtype=bool)
    # interleaved folds with no margin: the library pads a run by one bin, so
    # each run takes spikes from the bin of the other fold before it
    blocks = (np.arange(100) // 10) % 2
    with pytest.raises(ValueError, match="fold 0's encoding would train on"):
        th.check_no_fold_leakage(
            time, position, th.fold_training_masks(is_training, blocks, 2, 0), blocks
        )
    # a margin of two bins is enough
    th.check_no_fold_leakage(
        time, position, th.fold_training_masks(is_training, blocks, 2, 2), blocks
    )
    # but a gap without a position across the fold and its margin is bridged:
    # the run spans it, and the fold's bins with it
    blocks = np.zeros(100, dtype=np.int64)
    blocks[45:55] = 1
    gap = position.copy()
    gap[43:57] = np.nan
    masks = th.fold_training_masks(is_training, blocks, 2, 2)
    with pytest.raises(ValueError, match="fold 1's encoding would train on 10"):
        th.check_no_fold_leakage(time, gap, masks, blocks)
    th.check_no_fold_leakage(time, position, masks, blocks)


def test_the_header_names_the_fold_design_and_its_margin():
    assert th.fold_settings() == {
        "fold_design": "trials",
        "fold_classes": "to_well",
        "margin_s": 1.0,
        "max_segment_s": fo.MAX_SEGMENT_S,
        "min_trial_bin_share": fo.MIN_TRIAL_BIN_SHARE,
        "sklearn_version": sklearn.__version__,
    }


def test_epoch_folds_deal_the_trials_and_the_time_after_them_into_folds():
    """Six trials of 1.5 s, to wells 2 and 3 in turn, from 0.5 s into a 10 s
    epoch, the last ending at 9.5 s: eight segments (the time before the first
    trial, the six trials, the time after the last), none longer than the cap."""
    time = np.arange(0.0, 10.0, 0.002)
    trials = pd.DataFrame(
        {
            # out of order: the folds sort the trials by departure
            "movement_start_time": [2.0, 0.5, 3.5, 5.0, 6.5, 8.0],
            "end_time": [3.5, 2.0, 5.0, 6.5, 8.0, 9.5],
            "to_well": [3, 2, 2, 3, 2, 3],
        }
    )
    folds, margin_bins, counts = th.epoch_folds(time, trials)
    starts = np.array([0.0, 0.5, 2.0, 3.5, 5.0, 6.5, 8.0, 9.5])
    classes = np.r_[fo.BEFORE_FIRST_TRIAL, 2, 3, 2, 3, 2, 3, fo.AFTER_LAST_TRIAL]
    np.testing.assert_array_equal(
        folds, fo.trial_folds(time, starts, classes, fo.N_FOLDS)
    )
    assert margin_bins == 500
    assert counts == {
        "n_segments": 8,
        "n_groups": 8,
        "trial_bin_share": pytest.approx(0.9),
    }
