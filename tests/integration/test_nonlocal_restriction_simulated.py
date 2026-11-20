"""The restricted decoders' scores end to end, on clusterless recordings with
known content (``tests.support.tetrode_holdout_simulation``).

Three decoders share the fitted detector and differ only in when the non-local
states are allowed: never (``none``), exactly inside SWRs and nowhere else
(``swr``), always (``full``). Checked, for the filter score and the two-sided
score:

- the likelihood is held out: each block is scored with the encoding of its
  own fold, and no fold's encoding is fit on its block, with contiguous time
  blocks and with folds of whole trials (each lap and the rest after it);
- the filter scores add up to the library's marginal log-likelihood, and the
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

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence import nonlocal_restriction as nr
from continuum_swr_replay.analysis.event_evidence import tetrode_holdout as th
from continuum_swr_replay.analysis.event_evidence.state_evidence.terms import (
    refit_encoding,
)
from continuum_swr_replay.analysis.event_evidence.tetrode_holdout import (
    fold_training_masks,
)
from continuum_swr_replay.decoding.results import ENCODING_KEY
from tests.support import tetrode_holdout_simulation as sim

pytestmark = pytest.mark.integration

BIN_S = 0.002
NOT_REMOTE = ("local_burst", "silent")


def _quiet(message: str) -> None:
    pass


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
        detector, log_likelihood, in_swr, models=models, log=_quiet
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


def _heldout(recording, detector, refit, folds=None, margin_bins=th.MARGIN_BINS):
    return nr.heldout_log_likelihood(
        detector,
        recording.time,
        recording.position[:, None],
        recording.spike_times,
        recording.marks,
        recording.is_running,
        np.zeros(recording.time.size, dtype=bool),
        folds=folds,
        margin_bins=margin_bins,
        refit=refit,
        log=_quiet,
    )


def _recorded_heldout(recording, detector, folds=None, margin_bins=th.MARGIN_BINS):
    """The held-out likelihood and its blocks, and each fold's training mask
    and encoding as ``heldout_log_likelihood`` asked for them."""
    fits = []

    def recording_refit(detector, time, position, spike_times, marks, is_training):
        encoding = refit_encoding(
            detector, time, position, spike_times, marks, is_training
        )
        fits.append((np.asarray(is_training, dtype=bool).copy(), encoding))
        return encoding

    log_likelihood, blocks, _ = _heldout(
        recording, detector, recording_refit, folds, margin_bins
    )
    return log_likelihood, blocks, fits


@pytest.fixture(scope="module")
def recording_and_detector():
    recording = sim.simulate_recording(seed=0)
    return recording, sim.fit_detector(recording)


@pytest.fixture(scope="module")
def simulated(recording_and_detector):
    """The recording, its detector, the held-out likelihood (contiguous time
    blocks) and its blocks, each fold's training mask and encoding, and the
    folds' design (None: contiguous) and margin."""
    recording, detector = recording_and_detector
    log_likelihood, blocks, fits = _recorded_heldout(recording, detector)
    return recording, detector, log_likelihood, blocks, fits, None, th.MARGIN_BINS


@pytest.fixture(scope="module")
def simulated_trials(recording_and_detector):
    """As :func:`simulated`, with trial folds (each lap and the rest after it,
    balanced by the end it runs to, cut by the segment cap) and their margin."""
    recording, detector = recording_and_detector
    starts, classes = sim.trials(recording)
    folds = th.trial_folds(recording.time, starts, classes, nr.N_FOLDS)
    margin = th.margin_in_bins(th.TRIAL_MARGIN_S, recording.time)
    log_likelihood, blocks, fits = _recorded_heldout(recording, detector, folds, margin)
    return recording, detector, log_likelihood, blocks, fits, folds, margin


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


FOLD_DESIGNS = ["simulated", "simulated_trials"]


@pytest.mark.parametrize("design", FOLD_DESIGNS)
def test_no_fold_is_fit_on_its_own_block(design, request):
    """Fails if every fold's encoding is fit on all training bins, or on a
    training bin within the margin of its own bins."""
    recording, _, _, blocks, fits, folds, margin = request.getfixturevalue(design)
    assert len(fits) == nr.N_FOLDS
    if folds is None:
        np.testing.assert_array_equal(blocks, th.time_blocks(recording.time))
    else:
        np.testing.assert_array_equal(blocks, folds)
        # whole groups (each 31 s lap cut into two pieces by the 30 s cap):
        # every group's bins in one fold, and the folds interleaved
        segment = th.trial_segments(recording.time, sim.trials(recording)[0])
        group = th.capped_groups(recording.time, segment)
        assert group[-1] + 1 == 2 * sim.N_LAPS
        for g in np.unique(group):
            assert np.unique(blocks[group == g]).size == 1
        assert (np.diff(blocks) != 0).sum() >= nr.N_FOLDS
    expected = fold_training_masks(recording.is_running, blocks, nr.N_FOLDS, margin)
    index = np.arange(blocks.size)
    for k, (mask, _) in enumerate(fits):
        assert not (mask & (blocks == k)).any()
        np.testing.assert_array_equal(mask, expected[k])
        own = index[blocks == k]
        nearest = own[np.clip(np.searchsorted(own, index[mask]), 0, own.size - 1)]
        before = own[np.clip(np.searchsorted(own, index[mask]) - 1, 0, own.size - 1)]
        distance = np.minimum(
            np.abs(index[mask] - nearest), np.abs(index[mask] - before)
        )
        assert distance.min() > margin


@pytest.mark.parametrize("design", FOLD_DESIGNS)
def test_each_block_is_scored_with_its_own_folds_encoding(design, request):
    """Fold 0's encoding replaced by one fit on every training bin changes the
    likelihood of block 0 and of no other block. Fails if a block is scored
    with another block's encoding."""
    recording, detector, log_likelihood, blocks, fits, folds, margin = (
        request.getfixturevalue(design)
    )
    replacement = refit_encoding(
        detector,
        recording.time,
        recording.position[:, None],
        recording.spike_times,
        recording.marks,
        recording.is_running,
    )
    encodings = [replacement] + [encoding for _, encoding in fits[1:]]
    calls = iter(encodings)
    changed, _, _ = _heldout(
        recording, detector, lambda *args: next(calls), folds, margin
    )
    own = blocks == 0
    np.testing.assert_allclose(changed[~own], log_likelihood[~own], rtol=0, atol=1e-6)
    assert np.abs(changed[own] - log_likelihood[own]).max() > 0.1


@pytest.fixture(scope="module")
def chance():
    recording = sim.simulate_recording(
        kinds=sim.CHANCE_KINDS, boundary_event=False, seed=1
    )
    detector = sim.fit_detector(recording)
    no_swr = np.zeros(recording.time.size, dtype=bool)
    scores, record = nr.restriction_scores(
        detector,
        recording.time,
        recording.position[:, None],
        recording.spike_times,
        recording.marks,
        recording.is_running,
        no_swr,
        no_swr,
        linear_position=recording.position,
        log=_quiet,
    )
    return recording, detector, scores, record


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
    inside = nr.restriction_contrasts(sums, [nr.IMMOBILE_IN_SWR], BIN_S)
    outside = nr.restriction_contrasts(sums, [nr.IMMOBILE_OUTSIDE_SWR], BIN_S)
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
    recording, detector, scores, record = chance
    assert np.isfinite(scores[list(nr.SCORE_COLUMNS)].to_numpy()).all()
    # the checks' keys are the real record's
    assert nr.check_failures(record) == []
    for model in nr.MODELS:
        assert record[f"filter_identity_deviation_{model}"] < 1e-4
        assert record[f"causal_mass_deviation_{model}"] < 1e-3
    assert record["continuous_block_deviation"] < 1e-6
    assert list(scores.columns) == [
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
    ]
    # the seconds each block's encoding saw at the animal's place bin
    environment = detector.encoding_model_[ENCODING_KEY]["environment"]
    np.testing.assert_array_equal(
        scores["place_sampling_s"],
        th.place_sampling(
            th.animal_place_bins(environment, recording.position),
            recording.is_running,
            th.time_blocks(recording.time),
            nr.N_FOLDS,
            th.MARGIN_BINS,
            BIN_S,
        ),
    )
    # every place of the track is run over in every block
    assert (scores["place_sampling_s"] > 0).all()
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
    recording, _, scores, _ = chance
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
