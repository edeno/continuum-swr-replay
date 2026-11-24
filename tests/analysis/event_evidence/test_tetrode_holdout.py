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


def _window_rows() -> pd.DataFrame:
    """Two windows of two tetrodes each, with the columns
    :func:`~continuum_swr_replay.analysis.event_evidence.tetrode_holdout.window_scores`
    reads: window 0 has donor window 1, window 1 (a control) none."""
    return pd.DataFrame(
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
            "gain_rate": [2.0, 0.0, 0.5, -0.5],
            "agreement_rate": [0.5, -0.5, 0.0, 0.0],
            "hpd_cm": [30.0, 50.0, 4.0, 6.0],
            "agreement_null": [-1.0, -2.0, np.nan, np.nan],
            "agreement_null_silent": [0.0, -0.5, np.nan, np.nan],
            "agreement_null_rate": [-1.0, -1.0, np.nan, np.nan],
            "n_null_bins": [10, 10, 0, 0],
            "n_null_spikes": [2, 1, 0, 0],
            "spread_null_cm": [80.0, 100.0, np.nan, np.nan],
            "hpd_null_cm": [90.0, 110.0, np.nan, np.nan],
            "null_window_id": [1, 1, -1, -1],
            "mark_information": [3.0, 1.0, 0.0, 0.0],
            "spike_information": [4.0, 2.0, 0.0, 0.0],
            "n_information_spikes": [3, 1, 0, 0],
        }
    )


def test_window_scores_sum_over_tetrodes_and_divide_by_the_windows_spikes():
    windows = th.window_scores(_window_rows()).set_index("window_id")
    event = windows.loc[0]
    assert event["gain_per_spike"] == pytest.approx(4.0 / 4)
    # the spiking bins' gain: 4 - (-1) = 5 over 4 spikes
    assert event["spiking_gain_per_spike"] == pytest.approx(5.0 / 4)
    # the silent bins' gain: -1 over 4 spikes; the two parts add to the whole
    assert event["silent_gain_per_spike"] == pytest.approx(-1.0 / 4)
    assert event["spiking_gain_per_spike"] + event["silent_gain_per_spike"] == (
        pytest.approx(event["gain_per_spike"])
    )
    assert event["agreement_per_spike"] == pytest.approx(0.5)
    assert event["spread_cm"] == pytest.approx(50.0)
    assert np.isnan(windows.loc[1, "gain_per_spike"])
    assert windows.loc[1, "gain"] == pytest.approx(0.0)
    # the extra scores: rate part 2 of the gain 4, mark part 2, over 4 spikes
    assert event["gain_rate_per_spike"] == pytest.approx(0.5)
    assert event["gain_mark_per_spike"] == pytest.approx(0.5)
    assert event["agreement_rate_per_spike"] == pytest.approx(0.0)
    assert event["agreement_mark_per_spike"] == pytest.approx(0.5)
    assert event["agreement_null"] == -3.0 and event["n_null_spikes"] == 3
    assert event["agreement_null_per_spike"] == pytest.approx(-1.0)
    assert event["hpd_cm"] == pytest.approx(40.0)
    assert event["hpd_null_cm"] == pytest.approx(100.0)
    assert event["null_window_id"] == 1
    assert event["mark_information_per_spike"] == pytest.approx(1.0)
    assert event["spike_information_per_spike"] == pytest.approx(1.5)
    assert np.isnan(windows.loc[1, "mark_information_per_spike"])
    # no donor: the reference is NaN, not 0
    assert np.isnan(windows.loc[1, "agreement_null"])
    assert np.isnan(windows.loc[1, "agreement_null_per_spike"])


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


def test_a_table_without_a_column_of_the_run_is_refused_naming_the_runner():
    """A table that lacks a column the computation reads is refused, not
    half-read; the reader's contract is every column the runner writes."""
    short = _window_rows().drop(columns=["hpd_cm", "agreement_null"])
    with pytest.raises(ValueError, match=r"score_tetrode_holdout\.py") as error:
        th.window_scores(short)
    assert "hpd_cm" in str(error.value) and "agreement_null" in str(error.value)
    with pytest.raises(ValueError, match="'population'"):
        th.require_run_columns(_window_rows(), "the run")
    th.require_run_columns(_window_rows(), "a window", ["hpd_cm"])


def test_the_mark_part_is_the_marks_score_given_the_count():
    """``log sum exp(l) w - log sum exp(r) w = log sum exp(m) w'``, ``w'``
    proportional to ``exp(r) w``: the mark part is the predictive score of the
    marks under the weights updated by the count."""
    rng = np.random.default_rng(0)
    ground = rng.uniform(0.001, 0.2, (6, 5))
    counts = np.array([0, 1, 2, 0, 3, 1])
    marks = rng.normal(0, 2, (6, 5)) * (counts[:, None] > 0)
    rate = th.rate_terms(ground, counts)
    np.testing.assert_allclose(rate, counts[:, None] * np.log(ground) - ground)
    total = rate + marks
    weights = rng.dirichlet(np.ones(5), size=6)
    mark_part = th.predictive_log_score(total, weights) - th.predictive_log_score(
        rate, weights
    )
    updated = np.exp(rate) * weights
    updated /= updated.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(mark_part, th.predictive_log_score(marks, updated))
    # silent bins: the rate term is the bin term, the mark part exactly 0
    silent = counts == 0
    np.testing.assert_array_equal(rate[silent], -ground[silent])
    assert (mark_part[silent] == 0).all()


def _environment(interior):
    from types import SimpleNamespace

    interior = np.asarray(interior, dtype=bool)
    return SimpleNamespace(
        is_track_interior_=interior,
        place_bin_centers_=2.0 * np.arange(interior.size)[:, None],
    )


def test_hpd_lengths_by_hand():
    """Four real bins of 2 cm and a junction placeholder: the 95 % set of
    (0.6, 0.3, 0.06, 0.04) is its three largest bins, 6 cm; a point mass is one
    bin; a uniform distribution all four (8 cm); a row with a NaN is NaN."""
    environment = _environment([True, True, False, True, True])
    distribution = np.array(
        [
            [0.04, 0.6, 0.3, 0.06],
            [0.0, 1.0, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [np.nan] * 4,
        ]
    )
    widths = th.hpd_lengths(distribution, environment, chunk_rows=2)
    np.testing.assert_allclose(widths[:3], [6.0, 2.0, 8.0])
    assert np.isnan(widths[3])


def _donor_windows() -> tuple[pd.DataFrame, np.ndarray]:
    """Windows over 100 bins in two folds (bins 0-49 fold 0, 50-99 fold 1):
    three no-SWR events in fold 0, one in fold 1, two SWR events in fold 1, an
    event across the folds, and a control in fold 0."""
    folds = np.repeat([0, 1], 50)
    windows = pd.DataFrame(
        {
            "window_id": np.arange(8),
            "kind": ["event"] * 7 + ["control"],
            "group": ["no SWR"] * 3 + ["no SWR", "SWR", "SWR", "no SWR", "no SWR"],
            "s": [0, 10, 20, 60, 70, 80, 45, 30],
            "e": [4, 17, 22, 64, 72, 89, 54, 34],
        }
    )
    return windows, folds


def test_donors_stay_in_group_and_fold_and_are_never_the_window_itself():
    windows, folds = _donor_windows()
    donors = th.null_donors(windows, folds, seed=3)
    # the three no-SWR events of fold 0 give to each other, a cycle
    assert sorted(donors[:3]) == [0, 1, 2] and all(donors[:3] != [0, 1, 2])
    # the two SWR events of fold 1 swap
    assert list(donors[4:6]) == [5, 4]
    # alone in its group and fold, across the folds, a control: no donor
    assert donors[3] == -1 and donors[6] == -1 and donors[7] == -1
    for w, d in enumerate(donors):
        if d >= 0:
            assert d != w
            assert windows["group"][d] == windows["group"][w]
            assert folds[windows["s"][d]] == folds[windows["s"][w]]
    # the same seed, the same donors; another seed may differ
    np.testing.assert_array_equal(donors, th.null_donors(windows, folds, seed=3))
    seeds = {tuple(th.null_donors(windows, folds, seed=s)[:3]) for s in range(20)}
    assert len(seeds) == 2  # the two cycles of three


def test_null_pairs_match_bins_by_offset_and_wrap():
    windows, folds = _donor_windows()
    donors = np.full(len(windows), -1)
    donors[1] = 2  # window 1 (8 bins) takes window 2 (3 bins)
    rows = th.window_rows(windows["s"].to_numpy(), windows["e"].to_numpy())
    receiver, donor, owner = th.null_pairs(windows, donors, rows)
    np.testing.assert_array_equal(rows[receiver], np.arange(10, 18))
    np.testing.assert_array_equal(rows[donor], [20, 21, 22, 20, 21, 22, 20, 21])
    assert (owner == 1).all()
    empty = th.null_pairs(windows, np.full(len(windows), -1), rows)
    assert all(part.size == 0 for part in empty)


def test_spike_information_by_hand():
    """One fold, two tetrodes, four place bins; tetrode 0's ground is at the
    floor at bin 3, tetrode 1's at every bin but one. Spike 0 (tetrode 0) has
    the same conditional mark density at bins 0-2: no mark information; spike
    1 (tetrode 0) points to bin 1: about log 3; spike 2 is of tetrode 1, with
    one place above the floor: not counted."""
    from non_local_detector.likelihoods.common import EPS

    floor = np.float32(EPS)  # as the encoding stores it
    ground = np.array([[[0.1, 0.2, 0.4, floor], [0.1, floor, floor, floor]]])
    log_ground = np.log(ground[0, 0])
    log_intensity = np.array(
        [
            [*(np.log(0.5) + log_ground[:3]), np.log(EPS)],
            [-60.0, np.log(0.3), -60.0, np.log(EPS)],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    terms = th.FoldTerms(
        ground=ground,
        log_intensity=log_intensity,
        time_bin=np.array([0, 1, 1]),
        tetrode=np.array([0, 0, 1]),
        block=np.zeros(2, dtype=np.int64),
    )
    out = th.spike_information(terms, chunk_spikes=2)
    np.testing.assert_array_equal(out["n_places"], [[3, 1]])
    np.testing.assert_array_equal(out["counted"], [True, True, False])
    assert abs(out["mark"][0]) < 1e-6
    assert out["mark"][1] == pytest.approx(np.log(3), abs=1e-6)
    assert np.isnan(out["mark"][2]) and np.isnan(out["spike"][2])
    # the whole term of spike 0 follows the ground (0.1, 0.2, 0.4, ~0)
    p = np.array([0.1, 0.2, 0.4]) / 0.7
    assert out["spike"][0] == pytest.approx(np.log(4) + (p * np.log(p)).sum(), abs=1e-6)
    assert 0 < out["mark"][1] <= np.log(3) + 1e-9
