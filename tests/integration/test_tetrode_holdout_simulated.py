"""The held-out tetrode score end to end, on clusterless recordings with known
content (``tests.support.tetrode_holdout_simulation``).

The score's two contrasts are checked on every kind of content:

- gain (population minus local coding) is positive for remote content of every
  kind the detector finds, grows with distance from the animal, and is about
  zero for local content and rest;
- agreement (population minus no place preferred) is positive when the
  tetrodes share content and negative for a burst whose tetrodes each follow
  their own random place, which the gain and the in-sample score still credit.

Also: the rebuilt likelihood is the decoder's, and with a tetrode left out it
is the likelihood (and the smoothed states) of a decoder that never had that
tetrode; no scored spike is in the training spikes of the encoding that scores
it, even when every bin is a training bin; a tetrode with no field at
either place contributes nothing; an event across a time-block boundary, a
window with missing bins and a silent window are scored as specified.
"""

import copy

import numpy as np
import pandas as pd
import pytest
from non_local_detector.likelihoods.no_spike import predict_no_spike_log_likelihood

from continuum_swr_replay.analysis.event_evidence import tetrode_holdout as th
from continuum_swr_replay.analysis.event_evidence.state_evidence.terms import (
    refit_encoding,
    training_spike_bins,
)
from continuum_swr_replay.decoding.results import ENCODING_KEY
from tests.support import tetrode_holdout_simulation as sim

pytestmark = pytest.mark.integration

REMOTE = ("sweep", "stationary", "fragmented", "close_20", "close_40", "close_80")
N_MISSING = 10


def _quiet(message: str) -> None:
    pass


def _rest_windows(recording: sim.Recording, n: int, length: int) -> pd.DataFrame:
    rest = ~recording.is_running
    for s, e in zip(recording.events.s, recording.events.e, strict=True):
        rest[max(0, s - 250) : e + 250] = False
    starts = np.flatnonzero(np.convolve(rest, np.ones(length, int), "valid") == length)
    chosen = np.random.default_rng(1).choice(starts, n, replace=False)
    return pd.DataFrame({"kind": "rest", "s": chosen, "e": chosen + length - 1})


def _score(detector, recording, windows, is_missing=None, **folds) -> pd.DataFrame:
    """The held-out scores of ``windows``; ``folds`` passes ``folds`` and
    ``margin_bins`` to ``holdout_scores`` (by default its time blocks)."""
    windows = windows.reset_index(drop=True).rename_axis("window_id").reset_index()
    missing = (
        np.zeros(recording.time.size, dtype=bool) if is_missing is None else is_missing
    )
    scores, _ = th.holdout_scores(
        detector,
        recording.time,
        recording.position[:, None],
        recording.position,
        recording.spike_times,
        recording.marks,
        recording.is_running,
        missing,
        windows,
        log=_quiet,
        **folds,
    )
    return scores.join(windows.set_index("window_id")["kind"], on="window_id")


def _per_window(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.groupby(["window_id", "kind"]).agg(
        gain=("gain", "sum"),
        agreement=("agreement", "sum"),
        n_spikes=("n_spikes", "sum"),
        n_bins=("n_bins", "first"),
        n_scored_bins=("n_scored_bins", "first"),
    )
    spikes = out["n_spikes"].where(out["n_spikes"] > 0)
    out["gain_per_spike"] = out["gain"] / spikes
    out["agreement_per_spike"] = out["agreement"] / spikes
    return out.reset_index()


@pytest.fixture(scope="module")
def recording():
    recording = sim.simulate_recording(seed=0)
    return recording, sim.fit_detector(recording)


def _inserted_windows(recording: sim.Recording):
    """The inserted events and 12 rest windows, and the decode's missing bins:
    ``N_MISSING`` bins inside the first stationary event (its window id)."""
    length = int(sim.EVENT_S * 500)
    windows = pd.concat(
        [recording.events, _rest_windows(recording, 12, length)], ignore_index=True
    )
    is_missing = np.zeros(recording.time.size, dtype=bool)
    first = recording.events[recording.events.kind == "stationary"].iloc[0]
    is_missing[first.s + 10 : first.s + 10 + N_MISSING] = True
    return windows, is_missing, int(first.name)


@pytest.fixture(scope="module")
def inserted(recording):
    recording, detector = recording
    windows, is_missing, first = _inserted_windows(recording)
    scores = _score(detector, recording, windows, is_missing)
    return scores, _per_window(scores), first


@pytest.fixture(scope="module")
def inserted_trials(recording):
    """As :func:`inserted`, with the runner's trial folds: each lap and the
    rest after it, balanced by the end it runs to and cut by the 30 s cap into
    12 groups, and the 1 s margin."""
    recording, detector = recording
    windows, is_missing, first = _inserted_windows(recording)
    starts, classes = sim.trials(recording)
    folds = th.trial_folds(recording.time, starts, classes, th.N_FOLDS)
    margin = th.margin_in_bins(th.TRIAL_MARGIN_S, recording.time)
    scores = _score(
        detector, recording, windows, is_missing, folds=folds, margin_bins=margin
    )
    return scores, _per_window(scores), first, folds


@pytest.fixture(scope="module")
def detected(recording):
    recording, detector = recording
    events = sim.detected_events(detector, recording)
    return events, _per_window(_score(detector, recording, events))


@pytest.fixture(scope="module")
def chance():
    recording = sim.simulate_recording(
        kinds=sim.CHANCE_KINDS, boundary_event=False, seed=1
    )
    detector = sim.fit_detector(recording)
    events = sim.detected_events(detector, recording)
    return events, _per_window(_score(detector, recording, events))


def test_the_rebuilt_likelihood_is_the_decoders(recording):
    recording, detector = recording
    encoding = detector.encoding_model_[ENCODING_KEY]
    environment = encoding["environment"]
    interior = np.asarray(environment.is_track_interior_).ravel()
    one_block = np.zeros(recording.time.size, dtype=np.int64)
    terms = th.fold_terms(
        [encoding],
        recording.spike_times,
        recording.marks,
        recording.time,
        np.asarray(environment.place_bin_centers_)[interior],
        one_block,
    )
    position = recording.position[:, None]
    no_spike = np.asarray(
        predict_no_spike_log_likelihood(
            recording.time, recording.spike_times, detector.no_spike_rate
        )
    )[:, 0]
    kernel = np.asarray(
        detector._compute_local_position_kernel(
            recording.time, recording.time, position, environment
        )
    )
    rebuilt = th.state_log_likelihood(
        terms.total(),
        no_spike,
        kernel,
        np.asarray(detector.state_ind_)[detector.is_track_interior_state_bins_],
        detector.state_names,
        np.zeros(recording.time.size, dtype=bool),
    )
    decoders = np.asarray(
        detector.compute_log_likelihood(
            recording.time,
            recording.time,
            position,
            recording.spike_times,
            recording.marks,
        )
    )
    np.testing.assert_allclose(rebuilt, decoders, atol=1e-3)


def test_every_remote_event_is_detected_and_scores_on_both_contrasts(detected):
    events, per_window = detected
    for kind in REMOTE:
        assert (events.kind == kind).sum() == 6, kind
    assert (events.kind == "none").sum() == 0
    remote = per_window[per_window["kind"].isin([*REMOTE, "short"])]
    assert (remote["gain_per_spike"] > 0.3).all()
    assert (remote["agreement_per_spike"] > 0.5).all()


def test_the_gain_grows_with_distance_from_the_animal(inserted):
    _, per_window, _ = inserted
    mean = per_window.groupby("kind")["gain_per_spike"].mean()
    assert mean["close_20"] < mean["close_40"] < mean["close_80"]


def test_local_content_and_rest_gain_nothing(inserted):
    _, per_window, _ = inserted
    local = per_window[per_window["kind"].isin(["local_burst", "rest", "silent"])]
    assert len(local) == 24
    assert (local["gain"].abs() < 0.15).all()


def test_a_silent_window_has_a_score_but_no_score_per_spike(inserted):
    _, per_window, _ = inserted
    silent = per_window[per_window["kind"] == "silent"]
    assert (silent["n_spikes"] == 0).all()
    assert np.isfinite(silent["gain"]).all()
    assert silent["gain_per_spike"].isna().all()


def test_a_tetrode_with_no_field_at_either_place_contributes_nothing(inserted):
    scores, _, _ = inserted
    stationary = scores[scores["kind"] == "stationary"]
    region = stationary[stationary["tetrode"] == sim.N_TETRODES]
    others = stationary[stationary["tetrode"] != sim.N_TETRODES]
    assert region["gain"].abs().max() < 0.01
    assert others["gain"].mean() > 10


def test_an_event_across_a_block_boundary_is_scored_whole(inserted):
    _, per_window, _ = inserted
    boundary = per_window[per_window["kind"] == "boundary"]
    assert len(boundary) >= 1
    assert (boundary["n_scored_bins"] == boundary["n_bins"]).all()
    assert (boundary["gain_per_spike"] > 1).all()


def test_missing_bins_are_not_scored(inserted):
    _, per_window, missing_window = inserted
    row = per_window[per_window["window_id"] == missing_window].iloc[0]
    assert row["n_scored_bins"] == row["n_bins"] - N_MISSING


def test_bursts_without_shared_content_gain_but_do_not_agree(chance):
    events, per_window = chance
    assert (events.kind == "nonspatial").sum() >= 12
    nonspatial = per_window[per_window["kind"] == "nonspatial"]
    shared = per_window[per_window["kind"] == "shared_random"]
    # the gain credits any non-local activity the population covers...
    assert (nonspatial["gain_per_spike"] > 1).all()
    # ...the agreement only content the tetrodes share
    assert (nonspatial["agreement_per_spike"] < 0).all()
    assert (shared["agreement_per_spike"] > 0).all()


def _pieces(detector, recording, spike_times):
    """The decoder's No-Spike term and Local kernel for ``spike_times``."""
    environment = detector.encoding_model_[ENCODING_KEY]["environment"]
    no_spike = np.asarray(
        predict_no_spike_log_likelihood(
            recording.time, spike_times, detector.no_spike_rate
        )
    )[:, 0]
    kernel = np.asarray(
        detector._compute_local_position_kernel(
            recording.time, recording.time, recording.position[:, None], environment
        )
    )
    return no_spike, kernel


def test_leaving_a_tetrode_out_is_the_decoder_without_it(recording):
    recording, detector = recording
    k = 2
    keep = [j for j in range(len(recording.spike_times)) if j != k]
    spikes = [recording.spike_times[j] for j in keep]
    marks = [recording.marks[j] for j in keep]
    position = recording.position[:, None]
    encoding = detector.encoding_model_[ENCODING_KEY]
    environment = encoding["environment"]
    interior = np.asarray(environment.is_track_interior_).ravel()
    terms = th.fold_terms(
        [encoding],
        recording.spike_times,
        recording.marks,
        recording.time,
        np.asarray(environment.place_bin_centers_)[interior],
        np.zeros(recording.time.size, dtype=np.int64),
    )
    no_spike_all, kernel = _pieces(detector, recording, recording.spike_times)
    no_spike_k, _ = _pieces(detector, recording, [recording.spike_times[k]])
    state_ind = np.asarray(detector.state_ind_)[detector.is_track_interior_state_bins_]
    mine = th.state_log_likelihood(
        terms.total() - terms.tetrode_terms(k),
        no_spike_all - no_spike_k,
        kernel,
        state_ind,
        detector.state_names,
        np.zeros(recording.time.size, dtype=bool),
    )
    # a decoder whose encoding never saw tetrode k
    reduced = copy.deepcopy(detector)
    reduced.encoding_model_[ENCODING_KEY] = refit_encoding(
        detector, recording.time, position, spikes, marks, recording.is_running
    )
    theirs = np.asarray(
        reduced.compute_log_likelihood(
            recording.time, recording.time, position, spikes, marks
        )
    )
    np.testing.assert_allclose(mine, theirs, atol=1e-3)
    states_mine = np.asarray(
        detector._predict(time=recording.time, log_likelihoods=mine)[1]
    )
    states_theirs = reduced.predict(
        spike_times=spikes,
        spike_waveform_features=marks,
        time=recording.time,
        position=position,
        position_time=recording.time,
    )["acausal_state_probabilities"].values
    np.testing.assert_allclose(states_mine, states_theirs, atol=1e-3)


def test_no_scored_spike_trained_the_encoding_that_scores_it(recording, monkeypatch):
    recording, detector = recording
    # every bin trains, rest included (the real decodes train on all non-SWR
    # time), so only the folds keep a scored spike out of its encoding
    is_training = np.ones(recording.time.size, dtype=bool)
    position = recording.position[:, None]
    masks, scored = {}, []

    def refit(det, time, pos, spike_times, marks, mask):
        encoding = refit_encoding(det, time, pos, spike_times, marks, mask)
        masks[id(encoding)] = mask
        return encoding

    library_terms = th.per_tetrode_terms

    def per_tetrode_terms(encoding, spike_times, marks, time, centers):
        scored.append((id(encoding), np.concatenate(spike_times)))
        return library_terms(encoding, spike_times, marks, time, centers)

    monkeypatch.setattr(th, "per_tetrode_terms", per_tetrode_terms)
    window = recording.events[recording.events.kind == "stationary"].iloc[:1]
    th.holdout_scores(
        detector,
        recording.time,
        position,
        recording.position,
        recording.spike_times,
        recording.marks,
        is_training,
        np.zeros(recording.time.size, dtype=bool),
        window.assign(window_id=0),
        refit=refit,
        log=_quiet,
    )
    assert len(masks) == th.N_FOLDS and len(scored) == th.N_FOLDS
    n_scored = 0
    for encoding_id, spike_times in scored:
        trained = training_spike_bins(recording.time, position, masks[encoding_id])
        bins = np.digitize(spike_times, recording.time[1:-1])
        assert not trained[bins].any()
        n_scored += bins.size
    assert n_scored == sum(s.size for s in recording.spike_times)


def test_trial_folds_change_only_the_encodings(inserted, inserted_trials):
    """With trial folds (interleaved, a 1 s margin) the run completes on the
    same windows. What does not depend on the encodings (each window and
    tetrode's bins, scored bins and spikes) is unchanged. The scores move a
    little, as other training bins fit the encodings: by at most 0.15 nats per
    spike for every window with spikes (agreement: every window but rest,
    whose few spikes make it large), and the contrasts keep their signs:
    remote content gains and agrees, the gain grows with distance, and local
    content and rest gain under 0.2 nats (0.17 at most here, 0.11 with
    contiguous blocks), a local burst under 0.02 nats per spike."""
    scores, contiguous, _ = inserted
    trial_scores, per_window, _, folds = inserted_trials
    assert (np.diff(folds) != 0).sum() > th.N_FOLDS - 1  # interleaved, not blocks
    key = ["window_id", "tetrode", "n_bins", "n_scored_bins", "n_spikes"]
    pd.testing.assert_frame_equal(scores[key], trial_scores[key])
    assert not np.allclose(scores["gain"], trial_scores["gain"])
    pd.testing.assert_series_equal(per_window["kind"], contiguous["kind"])
    change = (per_window["gain_per_spike"] - contiguous["gain_per_spike"]).abs()
    assert (change.dropna() < 0.15).all()
    agreement = per_window["agreement_per_spike"] - contiguous["agreement_per_spike"]
    assert (agreement[per_window["kind"] != "rest"].dropna().abs() < 0.15).all()

    remote = per_window[per_window["kind"].isin(REMOTE)]
    assert (remote["gain_per_spike"] > 0.3).all()
    assert (remote["agreement_per_spike"] > 0.5).all()
    mean = per_window.groupby("kind")["gain_per_spike"].mean()
    assert mean["close_20"] < mean["close_40"] < mean["close_80"]
    local = per_window[per_window["kind"].isin(["local_burst", "rest", "silent"])]
    assert len(local) == 24
    assert (local["gain"].abs() < 0.2).all()
    bursts = per_window[per_window["kind"] == "local_burst"]
    assert (bursts["gain_per_spike"].abs() < 0.02).all()
