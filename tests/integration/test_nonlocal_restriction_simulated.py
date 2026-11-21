"""The restricted decoders' scores end to end, on clusterless recordings with
known content (``tests.support.tetrode_holdout_simulation``).

Three decoders share the fitted detector and differ only in when the non-local
states are allowed: never (``none``), exactly inside SWRs and nowhere else
(``swr``), always (``full``). Checked, for the filter score and the two-sided
score:

- the fold fit: each fold's whole decoder is fit without the fold's bins
  (no spike of the fold reaches its encoding, transitions or initial
  conditions), each bin is scored by its own fold's fit, and the decode's
  missing bins score 0;
- on a likelihood held out by contiguous time blocks
  (``tetrode_holdout.heldout_terms``, every tetrode in the model), the models
  (``model_scores``) behave as follows. The filter scores add up to the library's marginal log-likelihood, and the
  library's covariate-dependent ``_predict`` agrees with the ``swr`` model;
- A, replay only inside SWRs (the SWRs are the remote events): ``swr`` gains
  over ``none``, and ``full`` gains nothing over ``swr`` outside SWRs;
- B, replay outside SWRs too: ``full`` gains over ``swr`` outside SWRs, in the
  bins of the remote events no SWR marks;
- D, SWRs with no non-local content (plain rest and local bursts marked as
  SWRs beside the remote events): the forced state costs ``swr`` inside those
  SWRs, and ``full`` still gains nothing over ``swr`` outside SWRs, so the
  split by category puts the cost inside SWRs;
- C, bursts with no shared content (each tetrode its own random place each bin):
  ``full`` gains over ``none`` almost as much as for content the tetrodes share.
  At 2 ms a bin holds about one spike, and a non-local state can follow each
  spike's place from bin to bin, so this score cannot tell a burst without
  shared spatial content from fragmented replay. Local bursts and plain rest
  gain nothing.

Rest bins are labeled by whether an SWR marks them; running bins are
``moving`` and reported beside them, never in them.
"""

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence import nonlocal_restriction as nr
from continuum_swr_replay.analysis.event_evidence import tetrode_holdout as th
from continuum_swr_replay.analysis.event_evidence.tetrode_holdout import (
    fold_training_masks,
)
from continuum_swr_replay.decoding.results import ENCODING_KEY
from tests.support import tetrode_holdout_simulation as sim
from tests.support._restriction_tables import restriction_contrasts
from tests.support._time_blocks import TIME_BLOCK_MARGIN_BINS, time_blocks

pytestmark = [pytest.mark.slow, pytest.mark.integration]

BIN_S = 0.002
NOT_REMOTE = ("local_burst", "silent")


def _event_mask(recording: sim.Recording, events: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(recording.time.size, dtype=bool)
    for s, e in zip(events.s, events.e, strict=True):
        mask[s : e + 1] = True
    return mask


def _rest_id(recording: sim.Recording) -> np.ndarray:
    """Each bin's rest period (counting from 1 at the first rest)."""
    rest = (~recording.is_running).astype(int)
    return np.cumsum(np.r_[0, np.diff(rest)] == 1)


def _plain_rest_bins(recording: sim.Recording) -> np.ndarray:
    """The rest bins 250 bins (500 ms) or more from every event."""
    near = np.zeros(recording.time.size, dtype=bool)
    for s, e in zip(recording.events.s, recording.events.e, strict=True):
        near[max(0, s - 250) : e + 250] = True
    return ~recording.is_running & ~near


def _plain_rest(recording: sim.Recording, n_bins: int = 60) -> np.ndarray:
    """A stretch of ``n_bins`` of plain rest (:func:`_plain_rest_bins`) in the
    middle of each rest."""
    plain = _plain_rest_bins(recording)
    rest = _rest_id(recording)
    mask = np.zeros(recording.time.size, dtype=bool)
    for r in np.unique(rest[plain]):
        bins = np.flatnonzero(plain & (rest == r))
        start = bins[bins.size // 2]
        if plain[start : start + n_bins].all():
            mask[start : start + n_bins] = True
    return mask


def swr_masks(recording: sim.Recording) -> dict[str, np.ndarray]:
    """``in_swr`` of scenario A (every remote event), B (the remote events of
    every other rest) and D (every remote event, every local burst and a
    stretch of plain rest in each rest); B's remote events without an SWR, and
    D's SWRs with no non-local content."""
    remote = recording.events[~recording.events.kind.isin(NOT_REMOTE)]
    marked = _rest_id(recording)[remote.s.to_numpy()] % 2 == 0
    local_bursts = recording.events[recording.events.kind == "local_burst"]
    decoys = _event_mask(recording, local_bursts) | _plain_rest(recording)
    return {
        "A": _event_mask(recording, remote),
        "B": _event_mask(recording, remote[marked]),
        "B_unmarked": _event_mask(recording, remote[~marked]),
        "D": _event_mask(recording, remote) | decoys,
        "D_decoys": decoys,
    }


def _model_scores(detector, log_likelihood, in_swr, models) -> dict:
    """``nr.model_scores`` of ``models``, its library checks asserted."""
    scores, record = nr.model_scores(
        detector, log_likelihood, in_swr, models=models, log=sim.quiet
    )
    for model in models:
        assert (
            record[f"filter_identity_deviation_{model}"] <= nr.FILTER_IDENTITY_TOLERANCE
        ), record
    return scores


def gains(scores: pd.DataFrame, mask: np.ndarray) -> dict[str, float]:
    """``full - none``, ``swr - none`` and ``full - swr`` summed over ``mask``,
    both families."""
    out = {}
    for suffix in ("", "_two_sided"):
        full, swr, none = (
            scores[f"{m}{suffix}"].to_numpy(np.float64)[mask]
            for m in ("full", "swr", "none")
        )
        out[f"full_minus_none{suffix}"] = float((full - none).sum())
        out[f"swr_minus_none{suffix}"] = float((swr - none).sum())
        out[f"full_minus_swr{suffix}"] = float((full - swr).sum())
    return out


def _heldout_likelihood(recording, detector) -> np.ndarray:
    """The decoder's four-state log-likelihood, shape (n_time, n_state_bins),
    each contiguous time block under encodings fit on the other blocks
    (``tetrode_holdout.heldout_terms``, every tetrode in the model, then
    ``tetrode_holdout.state_log_likelihood``): the likelihood the models are
    scored on below."""
    terms, no_spike, kernel, _ = th.heldout_terms(
        detector,
        recording.time,
        recording.position[:, None],
        recording.spike_times,
        recording.marks,
        recording.is_running,
        folds=time_blocks(recording.time),
        margin_bins=TIME_BLOCK_MARGIN_BINS,
        log=sim.quiet,
    )
    state_ind = np.asarray(detector.state_ind_)[detector.is_track_interior_state_bins_]
    return th.state_log_likelihood(
        terms.total(),
        no_spike.sum(axis=1),
        kernel,
        state_ind,
        list(detector.state_names),
        np.zeros(recording.time.size, dtype=bool),
    )


@pytest.fixture(scope="module")
def recording_and_detector():
    recording = sim.simulate_recording(seed=0)
    return recording, sim.fit_detector(recording)


@pytest.fixture(scope="module")
def simulated(recording_and_detector):
    """The recording, its detector and the held-out likelihood (contiguous
    time blocks)."""
    recording, detector = recording_and_detector
    return recording, detector, _heldout_likelihood(recording, detector)


@pytest.fixture(scope="module")
def scenarios(simulated):
    """Each scenario's SWR mask and scores. The scenarios share the likelihood,
    and ``none`` and ``full`` do not depend on ``in_swr``, so they are scored
    once and ``swr`` per scenario."""
    recording, detector, log_likelihood, *_ = simulated
    masks = {**swr_masks(recording), "running": recording.is_running}
    shared = _model_scores(detector, log_likelihood, masks["A"], ("none", "full"))
    return masks, {
        name: pd.DataFrame(
            {
                **shared,
                **_model_scores(detector, log_likelihood, masks[name], ("swr",)),
                "in_swr": masks[name],
            }
        )
        for name in "ABD"
    }


@pytest.fixture(scope="module")
def chance():
    recording = sim.simulate_recording(
        kinds=sim.CHANCE_KINDS, boundary_event=False, seed=1
    )
    detector = sim.fit_detector(recording)
    no_swr = np.zeros(recording.time.size, dtype=bool)
    scores, record = nr.model_scores(
        detector, _heldout_likelihood(recording, detector), no_swr, log=sim.quiet
    )
    return recording, pd.DataFrame(scores), record


def test_the_filter_scores_add_up_to_the_librarys_marginal(simulated, scenarios):
    """The tolerance is the library's float32 running total (its per-bin terms
    agree to about 1e-5 nats; its sum of 93,000 of them is off by about 60).
    ``full`` allows every state at every bin, so its scores are the same in
    every scenario."""
    recording, detector, log_likelihood, *_ = simulated
    full = scenarios[1]["A"]
    smoothed = detector._predict(
        time=recording.time,
        log_likelihoods=log_likelihood,
        is_missing=np.zeros(recording.time.size, dtype=bool),
    )
    total = full["full"].to_numpy(np.float64).sum()
    # the library adds its terms into a float32 total: 1.8e-4 of it here
    assert abs(smoothed[2] - total) <= 5e-3 * abs(total)
    # bin by bin, against the library's own predictive posterior
    predicted = np.asarray(smoothed[7], dtype=np.float64)
    ll = log_likelihood.astype(np.float64)
    top = ll.max(axis=1, keepdims=True)
    per_bin = np.log((predicted * np.exp(ll - top)).sum(axis=1)) + top[:, 0]
    np.testing.assert_allclose(full["full"], per_bin, atol=1e-4)


def test_the_librarys_covariate_dependent_predict_is_the_swr_model(
    simulated, scenarios
):
    """``_predict`` with a (n_time, S, S) discrete matrix pairs index t with
    the move into t + 1; given the restricted likelihood (forbidden bins at
    -inf) and D~ of bin t + 1 at index t, its per-bin terms are the ``swr``
    scores. (Its initial distribution is unrestricted, which changes nothing
    here: the first bin is outside SWRs and ``pi`` is all on Local.)"""
    recording, detector, log_likelihood, *_ = simulated
    masks, scores = scenarios
    interior = detector.is_track_interior_state_bins_
    state_ind = np.asarray(detector.state_ind_)[interior]
    allowed = nr.allowed_states(detector.state_names, masks["B"])["swr"]
    patterns = np.array([allowed[~masks["B"]][0], allowed[masks["B"]][0]])
    restricted = nr.restricted_discrete_transitions(
        detector.discrete_state_transitions_, patterns
    )
    index = masks["B"].astype(int)
    masked = np.where(allowed[:, state_ind], log_likelihood, -np.inf)
    smoothed = detector._predict(
        time=recording.time,
        log_likelihoods=masked,
        is_missing=np.zeros(recording.time.size, dtype=bool),
        discrete_state_transitions=restricted[np.r_[index[1:], index[-1]]],
    )
    predicted = np.asarray(smoothed[7], dtype=np.float64)
    with np.errstate(divide="ignore"):
        terms = np.log(predicted) + log_likelihood.astype(np.float64)
    top = terms.max(axis=1, keepdims=True)
    per_bin = np.log(np.exp(terms - top).sum(axis=1)) + top[:, 0]
    np.testing.assert_allclose(scores["B"]["swr"], per_bin, atol=1e-4)


def _contrasts(scenarios, name: str) -> tuple[dict, dict, dict]:
    """Scenario ``name``'s contrasts inside SWRs, outside them (rest bins
    only), and its share of the non-local gain outside SWRs."""
    _, scores = scenarios
    sums = nr.restriction_sums(scores[name], _labels(scenarios, name))
    inside = restriction_contrasts(sums, [nr.IMMOBILE_IN_SWR], BIN_S)
    outside = restriction_contrasts(sums, [nr.IMMOBILE_OUTSIDE_SWR], BIN_S)
    return inside, outside, nr.share_outside_swr(sums)


@pytest.mark.parametrize("suffix", ["", "_two_sided"])
def test_replay_only_inside_swrs_gains_nothing_outside_them(scenarios, suffix):
    inside, outside, share = _contrasts(scenarios, "A")
    assert inside[f"swr_minus_none{suffix}"] > 5000
    # allowing the non-local states where nothing is non-local costs a little
    assert outside[f"full_minus_swr{suffix}"] < 0.01 * inside[f"swr_minus_none{suffix}"]
    assert share[f"share_outside_swr{suffix}"] < 0.01


@pytest.mark.parametrize("suffix", ["", "_two_sided"])
def test_replay_outside_swrs_is_what_the_full_model_gains(scenarios, suffix):
    masks, scores = scenarios
    inside, outside, share = _contrasts(scenarios, "B")
    assert inside[f"swr_minus_none{suffix}"] > 3000
    assert outside[f"full_minus_swr{suffix}"] > 3000
    # the gain is in the remote events no SWR marks
    unmarked = gains(scores["B"], masks["B_unmarked"])
    assert unmarked[f"full_minus_swr{suffix}"] > outside[f"full_minus_swr{suffix}"]
    assert 0.3 < share[f"share_outside_swr{suffix}"] < 0.7


@pytest.mark.parametrize("suffix", ["", "_two_sided"])
def test_swrs_without_non_local_content_cost_the_swr_model_inside_them(
    scenarios, suffix
):
    """The forced state's cost lands in the bins inside SWRs, not outside."""
    masks, scores = scenarios
    inside, outside, _ = _contrasts(scenarios, "D")
    decoys = gains(scores["D"], masks["D_decoys"])
    # about 0.1 nats a bin over 720 bins of plain rest and local bursts
    assert decoys[f"swr_minus_none{suffix}"] < -25
    assert decoys[f"full_minus_swr{suffix}"] > 25
    # the remote events still carry the SWR model's gain
    assert inside[f"swr_minus_none{suffix}"] > 3000
    assert outside[f"full_minus_swr{suffix}"] < 0.01 * inside[f"swr_minus_none{suffix}"]


def _labels(scenarios, name: str) -> np.ndarray:
    """Running bins ``moving``; rest bins by scenario ``name``'s SWRs."""
    masks, _ = scenarios
    return np.where(
        masks["running"],
        "moving",
        np.where(masks[name], nr.IMMOBILE_IN_SWR, nr.IMMOBILE_OUTSIDE_SWR),
    )


def test_the_library_record_checks_hold(chance):
    _, scores, record = chance
    assert np.isfinite(scores[list(nr.SCORE_COLUMNS)].to_numpy()).all()
    # the checks' keys are the real record's
    assert nr.check_failures(record) == []
    for model in nr.MODELS:
        assert record[f"filter_identity_deviation_{model}"] < 1e-4
        assert record[f"causal_mass_deviation_{model}"] < 1e-3
    assert record["continuous_block_deviation"] < 1e-6
    # no SWRs: the swr model is the none model
    np.testing.assert_array_equal(scores["swr"], scores["none"])
    np.testing.assert_array_equal(scores["swr_two_sided"], scores["none_two_sided"])


@pytest.mark.parametrize("suffix", ["", "_two_sided"])
def test_bursts_without_shared_content_are_credited_as_non_local(chance, suffix):
    """A caveat of the score, not a success: a burst whose tetrodes each
    follow their own random place gains over ``none`` about as much as one
    whose tetrodes share a random place, because at 2 ms a bin holds about one
    spike and a non-local state can follow each spike's place. Local bursts
    and plain rest gain nothing."""
    recording, scores, _ = chance
    column = f"full{suffix}"
    gain = scores[column].to_numpy(np.float64) - scores[f"none{suffix}"].to_numpy(
        np.float64
    )
    events = recording.events
    per_event = {
        kind: np.array([gain[s : e + 1].sum() for s, e in zip(g.s, g.e, strict=True)])
        for kind, g in events.groupby("kind")
    }
    assert (per_event["nonspatial"] > 100).all()
    assert (per_event["shared_random"] > 100).all()
    assert np.median(per_event["nonspatial"]) > 0.5 * np.median(
        per_event["shared_random"]
    )
    assert (np.abs(per_event["local_burst"]) < 1).all()
    plain = _plain_rest_bins(recording)
    assert abs(gain[plain].sum()) / (plain.sum() * BIN_S) < 1.0  # nats per s


# The fold fit: the whole decoder refit by EM on each fold's training folds.


def _fold_spikes_replaced(recording, mine, seed=0):
    """The recording's spikes with every spike the decoder bins into a bin of
    ``mine`` replaced: those removed, and as many again, with random marks, put
    in random bins of ``mine``."""
    rng = np.random.default_rng(seed)
    time = recording.time
    rows = np.flatnonzero(mine)
    spike_times, marks = [], []
    for st, mk in zip(recording.spike_times, recording.marks, strict=True):
        inside = mine[np.digitize(st, time[1:-1])]
        n = 2 * int(inside.sum()) + 5
        new_times = time[rng.choice(rows, n)] + 0.25 * BIN_S
        new_marks = rng.uniform(mk.min(), mk.max(), (n, mk.shape[1]))
        order = np.argsort(np.r_[st[~inside], new_times], kind="stable")
        spike_times.append(np.r_[st[~inside], new_times][order])
        marks.append(np.concatenate([mk[~inside], new_marks])[order])
    return spike_times, marks


@pytest.fixture(scope="module")
def fold_fit():
    """A shorter recording, its decoder, the fold-fit scores with trial folds,
    each fold's fit as ``fold_fit_scores`` asked for it (its training and
    missing bins, the fitted detector), the decode's masks, and each fold's
    likelihood as ``fold_fit_scores`` computed it (``fold_log_likelihood``).

    The decode has missing bins, as 85 of the analyzed epochs do: the first
    150 bins of the epoch, and 150 bins inside a stretch of fold 0 and of fold
    3; they are not training bins, as the decode's masks leave them out."""
    recording = sim.simulate_recording(
        kinds=("sweep", "stationary", "local_burst"),
        n_laps=4,
        boundary_event=False,
        seed=2,
    )
    canonical = sim.fit_detector(recording)
    starts, classes = sim.trials(recording)
    folds = th.trial_folds(recording.time, starts, classes, nr.N_FOLDS)
    margin = th.margin_in_bins(th.TRIAL_MARGIN_S, recording.time)
    remote = recording.events[~recording.events.kind.isin(NOT_REMOTE)]
    in_swr = _event_mask(recording, remote)
    is_missing = np.zeros(recording.time.size, dtype=bool)
    is_missing[:150] = True
    for k in (0, 3):
        start = np.flatnonzero(folds == k)[0]
        assert (folds[start : start + 500] == k).all()
        is_missing[start + 200 : start + 350] = True
    is_training = recording.is_running & ~is_missing
    fits = []

    likelihoods = []

    def recording_fit(*args, **kwargs):
        detector, record = nr.fit_fold_detector(*args, **kwargs)
        fits.append((args[5].copy(), args[6].copy(), detector))
        return detector, record

    def recording_likelihood(*args, **kwargs):
        likelihood = fold_log_likelihood(*args, **kwargs)
        likelihoods.append(np.asarray(likelihood))
        return likelihood

    fold_log_likelihood = nr.fold_log_likelihood
    with mock.patch.object(nr, "fold_log_likelihood", recording_likelihood):
        table, record = nr.fold_fit_scores(
            canonical,
            recording.time,
            recording.position[:, None],
            recording.spike_times,
            recording.marks,
            is_training,
            is_missing,
            in_swr,
            folds=folds,
            margin_bins=margin,
            linear_position=recording.position,
            fit=recording_fit,
            log=sim.quiet,
        )
    return (
        recording,
        canonical,
        folds,
        margin,
        in_swr,
        table,
        record,
        fits,
        is_training,
        is_missing,
        likelihoods,
    )


def test_each_fold_is_fit_without_its_bins(fold_fit):
    """Fold k's EM is given the encoding training bins of fold k's held-out
    encoding (none in the fold or its margin, none missing) and fold k's bins
    and the decode's missing bins as missing; every fold's pass passes the
    model checks."""
    _, _, folds, margin, _, table, record, fits, is_training, is_missing, _ = fold_fit
    masks = fold_training_masks(is_training, folds, nr.N_FOLDS, margin)
    assert len(fits) == nr.N_FOLDS
    for k, (training, missing, detector) in enumerate(fits):
        np.testing.assert_array_equal(training, masks[k])
        np.testing.assert_array_equal(missing, is_missing | (folds == k))
        assert not (training & (folds == k)).any()
        assert not (training & is_missing).any()
        np.testing.assert_allclose(
            np.asarray(detector.discrete_state_transitions_).sum(axis=1), 1.0
        )
    assert nr.fold_check_failures(record) == []
    assert [f["fold"] for f in record["folds"]] == list(range(nr.N_FOLDS))
    assert all(1 <= f["em_iterations"] <= 20 for f in record["folds"])
    assert np.isfinite(table[list(nr.SCORE_COLUMNS)].to_numpy()).all()
    probability = table[list(nr.STATE_PROBABILITY_COLUMNS.values())].to_numpy()
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-5)
    assert (table[nr.NONLOCAL_PLACE_BIN_COLUMN] >= 0).all()
    np.testing.assert_array_equal(table["block"], folds)


def test_the_fold_fit_table_has_its_columns_and_place_sampling(fold_fit):
    """The table's columns, in order, and each bin's seconds of training data
    at the animal's place under its own fold's encoding (the trial folds and
    their margin)."""
    recording, canonical, folds, margin, _, table, _, _, is_training, _, _ = fold_fit
    assert list(table.columns) == [
        "time",
        "none",
        "swr",
        "full",
        "none_two_sided",
        "swr_two_sided",
        "full_two_sided",
        "in_swr",
        "block",
        "place_sampling_s",
        *nr.STATE_PROBABILITY_COLUMNS.values(),
        nr.NONLOCAL_PLACE_BIN_COLUMN,
    ]
    environment = canonical.encoding_model_[ENCODING_KEY]["environment"]
    np.testing.assert_array_equal(
        table["place_sampling_s"],
        th.place_sampling(
            th.animal_place_bins(environment, recording.position),
            th.fold_training_masks(is_training, folds, nr.N_FOLDS, margin),
            folds,
            float(np.median(np.diff(recording.time))),
        ),
    )


def test_each_bin_is_scored_by_its_own_folds_fit(fold_fit):
    """The table's rows of fold k are fold k's pass: its fitted detector's
    likelihood of every bin, then the three models. Fold 1's pass scores fold
    0's bins differently, so the test fails if the folds are stitched wrong."""
    recording, _, folds, _, in_swr, table, _, fits, _, is_missing, _ = fold_fit
    columns = [
        *nr.SCORE_COLUMNS,
        *nr.STATE_PROBABILITY_COLUMNS.values(),
        nr.NONLOCAL_PLACE_BIN_COLUMN,
    ]
    passes = {}
    for k in (0, 1):
        detector = fits[k][2]
        likelihood = nr.fold_log_likelihood(
            detector,
            recording.time,
            recording.position[:, None],
            recording.spike_times,
            recording.marks,
            is_missing,
        )
        passes[k], _ = nr.model_scores(
            detector, likelihood, in_swr, summary_bins=folds == k, log=sim.quiet
        )
        mine = folds == k
        for column in columns:
            np.testing.assert_array_equal(
                table[column].to_numpy()[mine], passes[k][column][mine]
            )
    zero = folds == 0
    assert np.abs(passes[1]["full"][zero] - table["full"].to_numpy()[zero]).max() > 1e-3


def test_a_folds_spikes_reach_none_of_its_fit(fold_fit):
    """Replacing every spike in fold 0's bins leaves fold 0's fitted encoding,
    discrete transitions, initial conditions and EM trace as they were, and its
    likelihood of every bin outside fold 0; the likelihood of fold 0's bins
    (and so their scores) changes, as it must.

    That is the whole invariant: fold 0's model sees nothing of fold 0. The
    other folds' models are fit on fold 0's bins (they are training data
    there), and every pass's smoother runs over the whole epoch, so the rows of
    the other folds may change with fold 0's spikes."""
    (
        recording,
        canonical,
        folds,
        margin,
        _,
        _,
        record,
        fits,
        is_training,
        is_missing,
        likelihoods,
    ) = fold_fit
    zero = folds == 0
    masks = fold_training_masks(is_training, folds, nr.N_FOLDS, margin)
    spike_times, marks = _fold_spikes_replaced(recording, zero)
    changed_spikes = sum(
        int((~np.isin(new, old)).sum())
        for new, old in zip(spike_times, recording.spike_times, strict=True)
    )
    assert changed_spikes > 100
    detector, refit = nr.fit_fold_detector(
        canonical,
        recording.time,
        recording.position[:, None],
        spike_times,
        marks,
        masks[0],
        is_missing | zero,
    )
    original = fits[0][2]
    np.testing.assert_array_equal(
        detector.discrete_state_transitions_, original.discrete_state_transitions_
    )
    np.testing.assert_array_equal(
        detector.initial_conditions_, original.initial_conditions_
    )
    np.testing.assert_array_equal(
        refit["marginal_log_likelihoods"],
        record["folds"][0]["marginal_log_likelihoods"],
    )
    new, old = (
        detector.encoding_model_[ENCODING_KEY],
        original.encoding_model_[ENCODING_KEY],
    )
    for key in ("occupancy", "summed_ground_process_intensity", "mean_rates"):
        np.testing.assert_array_equal(np.asarray(new[key]), np.asarray(old[key]))
    for key in ("encoding_positions", "encoding_spike_waveform_features"):
        for a, b in zip(new[key], old[key], strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def likelihood(spikes, marks):
        return np.asarray(
            nr.fold_log_likelihood(
                original,
                recording.time,
                recording.position[:, None],
                spikes,
                marks,
                is_missing,
            )
        )

    # fold 0's likelihood of the recorded spikes, as the fold fit computed it
    before = likelihoods[0]
    after = likelihood(spike_times, marks)
    # float32 rounding only: the library's KDE batches the spikes, so the
    # batches change shape with the spikes (observed: 32 of 12 million values
    # off, at most 1.7e-7 of the value)
    np.testing.assert_allclose(after[~zero], before[~zero], rtol=1e-6, atol=1e-5)
    assert np.abs(after[zero & ~is_missing] - before[zero & ~is_missing]).max() > 1.0


def test_the_decodes_missing_bins_score_zero_in_every_fold(fold_fit):
    """The decode's missing bins, at the very start of the epoch and inside the
    held-out folds 0 and 3, have a likelihood of 0 under every fold's fit and
    score 0 in every column (to float32 rounding of the posterior's mass); the
    held-out folds' other bins score as usual."""
    recording, _, folds, _, _, table, _, fits, _, is_missing, likelihoods = fold_fit
    assert is_missing[0] and (is_missing & (folds == 0)).sum() >= 150
    assert (is_missing & (folds == 3)).sum() >= 150
    assert len(likelihoods) == len(fits) == nr.N_FOLDS
    for likelihood in likelihoods:
        assert (likelihood[is_missing] == 0).all()
        assert (likelihood[~is_missing] != 0).any(axis=1).all()
    scores = table[list(nr.SCORE_COLUMNS)].to_numpy(np.float64)
    # float32 rounding of the posterior's mass: here up to 1e-6 (filter) and
    # 7e-6 (two-sided)
    filter_columns = [nr.SCORE_COLUMNS.index(m) for m in nr.MODELS]
    two_sided_columns = [nr.SCORE_COLUMNS.index(f"{m}_two_sided") for m in nr.MODELS]
    np.testing.assert_allclose(
        scores[is_missing][:, filter_columns], 0.0, rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(
        scores[is_missing][:, two_sided_columns], 0.0, rtol=0, atol=1e-4
    )
    observed = scores[~is_missing & np.isin(folds, [0, 3])]
    assert np.isfinite(observed).all() and (observed != 0).all(axis=1).mean() > 0.99
