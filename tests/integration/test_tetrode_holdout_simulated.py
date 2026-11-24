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
from continuum_swr_replay.analysis.regression import folds as fo
from continuum_swr_replay.decoding.results import ENCODING_KEY
from tests.support import tetrode_holdout_simulation as sim
from tests.support._tetrode_holdout_tables import SCORE_COLUMNS
from tests.support._time_blocks import TIME_BLOCK_MARGIN_BINS, time_blocks

pytestmark = [pytest.mark.slow, pytest.mark.integration]

REMOTE = ("sweep", "stationary", "fragmented", "close_20", "close_40", "close_80")
N_MISSING = 10


def _rest_windows(recording: sim.Recording, n: int, length: int) -> pd.DataFrame:
    rest = ~recording.is_running
    for s, e in zip(recording.events.s, recording.events.e, strict=True):
        rest[max(0, s - 250) : e + 250] = False
    starts = np.flatnonzero(np.convolve(rest, np.ones(length, int), "valid") == length)
    chosen = np.random.default_rng(1).choice(starts, n, replace=False)
    return pd.DataFrame({"kind": "rest", "s": chosen, "e": chosen + length - 1})


def _holdout_scores(
    detector,
    recording,
    windows,
    *,
    is_missing=None,
    folds=None,
    margin_bins=TIME_BLOCK_MARGIN_BINS,
    marks=None,
) -> tuple[pd.DataFrame, dict]:
    """``holdout_scores`` of ``windows`` (numbered in order, ``window_id``)
    under ``folds`` and ``margin_bins`` (by default the recording's five time
    blocks, a margin of two bins), the reference's donors drawn with seed 0.
    Its columns are the first of the run's, in order."""
    missing = (
        np.zeros(recording.time.size, dtype=bool) if is_missing is None else is_missing
    )
    scores, record = th.holdout_scores(
        detector,
        recording.time,
        recording.position[:, None],
        recording.position,
        recording.spike_times,
        recording.marks if marks is None else marks,
        recording.is_running,
        missing,
        windows,
        folds=time_blocks(recording.time) if folds is None else folds,
        margin_bins=margin_bins,
        null_seed=0,
        log=sim.quiet,
    )
    assert tuple(scores.columns) == SCORE_COLUMNS
    return scores, record


def _score(
    detector,
    recording,
    windows,
    is_missing=None,
    folds=None,
    margin_bins=TIME_BLOCK_MARGIN_BINS,
) -> pd.DataFrame:
    """The held-out scores of ``windows`` (:func:`_holdout_scores`) with each
    window's ``kind``. No window is an event (``kind`` is the simulated
    content), so none has a donor for the reference."""
    windows = windows.reset_index(drop=True).rename_axis("window_id").reset_index()
    windows = windows.assign(group="no SWR")
    scores, _ = _holdout_scores(
        detector,
        recording,
        windows,
        is_missing=is_missing,
        folds=folds,
        margin_bins=margin_bins,
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
def recording(simulated_recording):
    return simulated_recording


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
    folds = fo.trial_folds(recording.time, starts, classes, fo.N_FOLDS)
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
def chance(chance_recording):
    recording, detector = chance_recording
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
        window.assign(window_id=0, group="no SWR"),
        folds=time_blocks(recording.time),
        margin_bins=TIME_BLOCK_MARGIN_BINS,
        refit=refit,
        log=sim.quiet,
    )
    assert len(masks) == fo.N_FOLDS and len(scored) == fo.N_FOLDS
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
    assert (np.diff(folds) != 0).sum() > fo.N_FOLDS - 1  # interleaved, not blocks
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


# The extra scores: the HPD length, the rate and mark parts, and the
# shuffled-window reference.


def _score_extra(detector, recording, windows, *, is_missing=None, marks=None):
    """:func:`_holdout_scores` of ``windows`` (each with the simulated
    ``content`` and its ``group``; every window an event); the per-window
    scores (``window_scores``) with ``content``, and the record."""
    windows = windows.reset_index(drop=True).rename_axis("window_id").reset_index()
    windows = windows.assign(kind="event", event_id=windows["window_id"])
    scores, record = _holdout_scores(
        detector, recording, windows, is_missing=is_missing, marks=marks
    )
    keyed = windows.set_index("window_id")[["event_id", "kind", "group", "content"]]
    scores = scores.join(keyed, on="window_id").assign(epoch_id="e", animal="a")
    per_window = th.window_scores(scores.drop(columns="content")).join(
        keyed["content"], on="window_id"
    )
    return scores, per_window, record


@pytest.fixture(scope="module")
def extra(recording):
    """The inserted events (and the missing bins) of :func:`inserted`, scored: the
    remote events in one group, the local ones (local
    bursts and silent windows) in the other, so the remote events' donors are
    other remote events of another place or time."""
    recording, detector = recording
    windows, is_missing, _ = _inserted_windows(recording)
    windows = windows[windows["kind"] != "rest"].rename(columns={"kind": "content"})
    windows["group"] = np.where(
        windows["content"].isin(["local_burst", "silent"]), "SWR", "no SWR"
    )
    out = _score_extra(detector, recording, windows, is_missing=is_missing)
    return (*out, windows, is_missing)


@pytest.fixture(scope="module")
def extra_chance(chance_recording):
    """The chance recording's inserted bursts, scored: the bursts whose
    tetrodes each follow their own random place (``nonspatial``) one group, the
    bursts they share (``shared_random``) the other."""
    recording, detector = chance_recording
    windows = recording.events[
        recording.events.kind.isin(["nonspatial", "shared_random"])
    ].rename(columns={"kind": "content"})
    windows["group"] = np.where(windows["content"] == "nonspatial", "no SWR", "SWR")
    return _score_extra(detector, recording, windows)


def test_the_rate_and_mark_parts_add_up_and_silence_is_all_rate(extra):
    scores, per_window, *_ = extra
    # a tetrode silent through a window: every bin's term is its rate term
    # (exactly, bin by bin: tests/analysis/event_evidence/test_tetrode_holdout.py);
    # the window sums are differences of cumulative sums over every scored
    # bin, so they agree to their rounding (here up to 5e-11 nats)
    silent = scores[scores["n_spikes"] == 0]
    assert len(silent) > 50
    np.testing.assert_allclose(silent["gain_rate"], silent["gain"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        silent["agreement_rate"], silent["agreement"], rtol=0, atol=1e-8
    )
    # the parts per spike add to the scores per spike
    spiking = per_window.dropna(subset=["gain_per_spike"])
    np.testing.assert_allclose(
        spiking["gain_rate_per_spike"] + spiking["gain_mark_per_spike"],
        spiking["gain_per_spike"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        spiking["agreement_rate_per_spike"] + spiking["agreement_mark_per_spike"],
        spiking["agreement_per_spike"],
        rtol=1e-12,
    )


def test_place_specific_marks_agree_where_the_tetrodes_share_a_place(extra):
    """The simulated marks name the cell, and each cell has its own place, so
    the marks tell where a tetrode points: for remote content the tetrodes
    share, the mark part of the agreement is positive."""
    _, per_window, *_ = extra
    remote = per_window[per_window["content"].isin(REMOTE)]
    assert (remote["agreement_mark_per_spike"] > 0).mean() > 0.9
    assert remote["agreement_mark_per_spike"].median() > 0.2


def test_marks_without_place_information_carry_no_part_of_the_scores(recording, extra):
    """With every spike's mark drawn from one distribution, whatever its cell,
    the marks say nothing about place: the mark parts of the gain and the
    agreement are small beside the rate parts and beside the mark parts of the
    same events with the simulated marks, which name the cell. Without place
    information in the marks the agreement is small too (the spike count
    alone says little about place). Not 0: each
    place's mark density is estimated from the training spikes near it, and
    the log of a noisy estimate varies with place (observed: the gain's mark
    part 0.13 of its rate part)."""
    _, informative, *_ = extra
    informative = informative[informative["content"].isin(REMOTE)]
    recording, detector = recording
    rng = np.random.default_rng(4)
    marks = [rng.normal(50.0, 20.0, m.shape) for m in recording.marks]
    windows = recording.events[recording.events.kind.isin(REMOTE)].rename(
        columns={"kind": "content"}
    )
    windows["group"] = "no SWR"
    _, per_window, _ = _score_extra(detector, recording, windows, marks=marks)
    # the marks' own place information: near 0 here, positive with the
    # simulated marks, below log(n_place_bins) either way
    n_places = np.log(
        detector.encoding_model_[ENCODING_KEY]["environment"].is_track_interior_.sum()
    )
    flat = per_window["mark_information_per_spike"].median()
    named = informative["mark_information_per_spike"].median()
    print(f"mark information: {flat:.3f} flat, {named:.3f} named, bound {n_places:.3f}")
    assert abs(flat) < 0.1 * named
    assert 0 < named < n_places
    assert (informative["spike_information_per_spike"] > 0).all()
    for score in ("gain", "agreement"):
        mark = per_window[f"{score}_mark_per_spike"].abs().median()
        rate = per_window[f"{score}_rate_per_spike"].abs().median()
        named = informative[f"{score}_mark_per_spike"].abs().median()
        print(f"{score}: mark {mark:.3f}, rate {rate:.3f}; named marks {named:.3f}")
        # observed: gain mark 0.094, rate 0.711, named 5.4; agreement mark
        # 0.006, rate 0.044, named 1.0 (nats per spike, medians)
        assert rate > 0.01, (score, rate)
        assert mark < 0.25 * rate, (score, mark, rate)
        assert mark < 0.25 * named, (score, mark, named)


def test_hpd_is_the_event_tables_hpd(recording):
    """The decode frame's ``hpd`` (``get_posterior_stats``: the library's HPD
    on the decode's conditional non-local posterior over every place bin, the
    junction placeholders NaN) is ``hpd_lengths`` of the same posterior over
    the real place bins, to the bit."""
    from non_local_detector.model_checking.highest_posterior_density import (
        get_highest_posterior_threshold,
        get_HPD_spatial_coverage,
    )

    from continuum_swr_replay.analysis.decode_quality.posterior_analysis import (
        get_conditional_non_local_posterior,
    )
    from continuum_swr_replay.analysis.event_evidence.state_evidence.terms import (
        interior_bin_mask,
    )

    recording, detector = recording
    keep = slice(0, 20_000)
    results = detector.predict(
        spike_times=recording.spike_times,
        spike_waveform_features=recording.marks,
        time=recording.time[keep],
        position=recording.position[keep, None],
        position_time=recording.time[keep],
    )
    posterior = get_conditional_non_local_posterior(results)
    event_table = get_HPD_spatial_coverage(
        posterior, get_highest_posterior_threshold(posterior, coverage=0.95)
    )
    environment = detector.encoding_model_[ENCODING_KEY]["environment"]
    interior = interior_bin_mask(environment)
    mine = th.hpd_lengths(np.asarray(posterior)[:, interior], environment)
    np.testing.assert_array_equal(mine, event_table)
    assert np.ptp(mine) > 0


def test_the_reference_is_below_shared_content(extra):
    """Remote content the tetrodes share agrees more with its own window's
    distribution than with another window's (another place or time)."""
    _, per_window, *_ = extra
    remote = per_window[per_window["content"].isin(REMOTE)].dropna(
        subset=["agreement_null_per_spike"]
    )
    assert len(remote) > 20
    assert (
        remote["agreement_per_spike"] > remote["agreement_null_per_spike"]
    ).mean() > 0.9
    assert remote["agreement_null_per_spike"].median() < 0


def test_unrelated_places_agree_at_the_reference_below_zero(extra_chance):
    """Bursts whose tetrodes each follow their own random place agree with
    their own window's distribution as with another burst's: the two are
    exchangeable (the scored tetrode is not in the decode that makes the
    distribution, and each tetrode's places are drawn on their own), so their
    paired difference is 0 up to its sampling error, and both are below 0
    (Jensen: a distribution unrelated to the spikes loses to no preference).
    The per-window differences are large (observed: real -0.87 and reference
    -0.75 nats per spike on average over 36 bursts, the real one lower in 61 %
    of them, a difference of about 2 standard errors). Bursts the tetrodes
    share are above the reference."""
    _, per_window, _ = extra_chance
    known = per_window.dropna(
        subset=["agreement_null_per_spike", "agreement_per_spike"]
    )
    nonspatial = known[known["content"] == "nonspatial"]
    shared = known[known["content"] == "shared_random"]
    assert len(nonspatial) >= 20 and len(shared) >= 6
    real = nonspatial["agreement_per_spike"].to_numpy()
    null = nonspatial["agreement_null_per_spike"].to_numpy()
    difference = real - null
    error = difference.std(ddof=1) / np.sqrt(difference.size)
    print(
        f"unrelated places: real {real.mean():.3f}, reference {null.mean():.3f}, "
        f"difference {difference.mean():.3f} +/- {error:.3f} (standard error)"
    )
    assert real.mean() < 0 and null.mean() < 0
    assert abs(difference.mean()) < 3 * error
    assert (shared["agreement_per_spike"] > shared["agreement_null_per_spike"]).all()


def test_donors_are_other_windows_of_the_group_and_fold_and_seeded(recording, extra):
    recording, _ = recording
    scores, per_window, record, windows, *_ = extra
    folds = time_blocks(recording.time)
    by_id = per_window.set_index("window_id")
    with_donor = by_id[by_id["null_window_id"] >= 0]
    assert len(with_donor) == record["n_null_windows"] > 20
    starts = windows.reset_index(drop=True)["s"]
    for w, row in with_donor.iterrows():
        d = int(row["null_window_id"])
        assert d != w
        assert by_id.loc[d, "group"] == row["group"]
        assert folds[starts[d]] == folds[starts[w]]
    # without a donor the reference is NaN
    without = by_id[by_id["null_window_id"] < 0]
    assert without["agreement_null"].isna().all()
    # the same seed, the same donors and the same reference
    keyed = windows.reset_index(drop=True).rename_axis("window_id").reset_index()
    keyed = keyed.assign(kind="event")
    np.testing.assert_array_equal(
        th.null_donors(keyed, folds, 0), th.null_donors(keyed, folds, 0)
    )
    donors = th.null_donors(keyed, folds, 0)
    expected = np.where(donors >= 0, keyed["window_id"].to_numpy()[donors], -1)
    np.testing.assert_array_equal(by_id["null_window_id"].to_numpy(), expected)


def test_place_information_is_a_property_of_the_spike_and_its_encoding(
    recording, extra
):
    """Scored with only the first four tetrodes (another decode, other
    distributions), those tetrodes' spikes carry the same place information,
    window by window, while their agreement changes."""
    recording, detector = recording
    scores, _, _, windows, is_missing = extra
    subset = copy.deepcopy(detector)
    fewer, _, _ = _score_extra(
        subset,
        sim.Recording(
            time=recording.time,
            position=recording.position,
            is_running=recording.is_running,
            spike_times=recording.spike_times[:4],
            marks=recording.marks[:4],
            events=recording.events,
            region_tetrode=recording.region_tetrode,
        ),
        windows,
        is_missing=is_missing,
    )
    key = ["window_id", "tetrode"]
    mine = scores[scores["tetrode"] < 4].set_index(key).sort_index()
    theirs = fewer.set_index(key).sort_index()
    same = mine["n_scored_bins"] == theirs["n_scored_bins"]
    assert same.mean() > 0.95
    for column in ("mark_information", "spike_information", "n_information_spikes"):
        np.testing.assert_allclose(
            theirs.loc[same, column], mine.loc[same, column], rtol=1e-6, atol=1e-9
        )
    assert not np.allclose(theirs["agreement"], mine["agreement"])
