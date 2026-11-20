"""The restricted decoders' scores end to end, on clusterless recordings with
known content (``tests.support.tetrode_holdout_simulation``).

Three decoders share the fitted detector and differ only in when the non-local
states are allowed: never (``none``), inside SWRs (``swr_only``), always
(``full``). Checked, for the filter score and the two-sided score:

- the filter scores add up to the library's marginal log-likelihood, and the
  library's covariate-dependent ``_predict`` agrees with the ``swr_only`` model;
- replay only inside SWRs: ``swr_only`` gains over ``none`` and ``full`` gains
  nothing over ``swr_only``;
- replay outside SWRs too: ``full`` gains over ``swr_only``, in the bins of the
  remote events no SWR marks;
- bursts with no shared content (each tetrode its own random place each bin):
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
from tests.support import tetrode_holdout_simulation as sim

pytestmark = pytest.mark.integration

BIN_S = 0.002
IMMOBILE = ["immobile_in_swr", "immobile_outside_swr"]
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


def swr_masks(recording: sim.Recording) -> dict[str, np.ndarray]:
    """``in_swr`` of scenario A (every remote event) and B (the remote events
    of every other rest), and B's remote events without an SWR."""
    remote = recording.events[~recording.events.kind.isin(NOT_REMOTE)]
    marked = _rest_id(recording)[remote.s.to_numpy()] % 2 == 0
    return {
        "A": _event_mask(recording, remote),
        "B": _event_mask(recording, remote[marked]),
        "B_unmarked": _event_mask(recording, remote[~marked]),
    }


def model_scores(detector, log_likelihood, in_swr, models=nr.MODELS) -> pd.DataFrame:
    """Every bin's scores under ``models``, on a shared likelihood."""
    interior = detector.is_track_interior_state_bins_
    state_ind = np.asarray(detector.state_ind_)[interior]
    columns = {}
    for model, allowed in nr.allowed_states(detector.state_names, in_swr).items():
        if model not in models:
            continue
        filter_score, two_sided, record = nr.predictive_log_scores(
            log_likelihood,
            np.asarray(detector.initial_conditions_)[interior],
            np.asarray(detector.continuous_state_transitions_)[
                np.ix_(interior, interior)
            ],
            np.asarray(detector.discrete_state_transitions_),
            state_ind,
            allowed,
        )
        assert record["marginal_agrees"], record
        columns[model] = filter_score
        columns[f"{model}_two_sided"] = two_sided
    return pd.DataFrame({**columns, "in_swr": in_swr})


def gains(scores: pd.DataFrame, mask: np.ndarray) -> dict[str, float]:
    """``full - none``, ``swr_only - none`` and ``full - swr_only`` summed over
    ``mask``, both families."""
    out = {}
    for suffix in ("", "_two_sided"):
        full, swr, none = (
            scores[f"{m}{suffix}"].to_numpy(np.float64)[mask]
            for m in ("full", "swr_only", "none")
        )
        out[f"full_minus_none{suffix}"] = float((full - none).sum())
        out[f"swr_only_minus_none{suffix}"] = float((swr - none).sum())
        out[f"full_minus_swr_only{suffix}"] = float((full - swr).sum())
    return out


@pytest.fixture(scope="module")
def simulated():
    recording = sim.simulate_recording(seed=0)
    detector = sim.fit_detector(recording)
    log_likelihood, _, _ = nr.heldout_log_likelihood(
        detector,
        recording.time,
        recording.position[:, None],
        recording.spike_times,
        recording.marks,
        recording.is_running,
        np.zeros(recording.time.size, dtype=bool),
        log=_quiet,
    )
    return recording, detector, log_likelihood


@pytest.fixture(scope="module")
def scenarios(simulated):
    recording, detector, log_likelihood = simulated
    masks = {**swr_masks(recording), "running": recording.is_running}
    return masks, {
        name: model_scores(detector, log_likelihood, masks[name]) for name in "AB"
    }


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
        log=_quiet,
    )
    return recording, scores, record


def test_the_filter_scores_add_up_to_the_librarys_marginal(simulated):
    """The tolerance is the library's float32 running total (its per-bin terms
    agree to about 1e-5 nats; its sum of 93,000 of them is off by about 60)."""
    recording, detector, log_likelihood = simulated
    full = model_scores(
        detector, log_likelihood, np.zeros(recording.time.size, bool), ["full"]
    )
    smoothed = detector._predict(
        time=recording.time,
        log_likelihoods=log_likelihood,
        is_missing=np.zeros(recording.time.size, dtype=bool),
    )
    total = full["full"].to_numpy(np.float64).sum()
    assert abs(smoothed[2] - total) <= nr.MARGINAL_TOLERANCE * abs(total)
    # bin by bin, against the library's own predictive posterior
    predicted = np.asarray(smoothed[7], dtype=np.float64)
    ll = log_likelihood.astype(np.float64)
    top = ll.max(axis=1, keepdims=True)
    per_bin = np.log((predicted * np.exp(ll - top)).sum(axis=1)) + top[:, 0]
    np.testing.assert_allclose(full["full"], per_bin, atol=1e-4)


def test_the_librarys_covariate_dependent_predict_is_the_swr_only_model(
    simulated, scenarios
):
    """``_predict`` with a (n_time, S, S) discrete matrix pairs index t with
    the move into t + 1; given the restricted likelihood (forbidden bins at
    -inf) and D~ of bin t + 1 at index t, its per-bin terms are the
    ``swr_only`` scores. (Its initial distribution is unrestricted, which
    changes nothing here: the first bin is outside SWRs and ``pi`` is all on
    Local.)"""
    recording, detector, log_likelihood = simulated
    masks, scores = scenarios
    interior = detector.is_track_interior_state_bins_
    state_ind = np.asarray(detector.state_ind_)[interior]
    allowed = nr.allowed_states(detector.state_names, masks["B"])["swr_only"]
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
    np.testing.assert_allclose(scores["B"]["swr_only"], per_bin, atol=1e-4)


@pytest.mark.parametrize("suffix", ["", "_two_sided"])
def test_replay_only_inside_swrs_gains_nothing_outside_them(scenarios, suffix):
    masks, scores = scenarios
    sums = nr.restriction_sums(scores["A"], _labels(scenarios, "A"))
    immobile = nr.restriction_contrasts(sums, IMMOBILE, BIN_S)
    assert immobile[f"swr_only_minus_none{suffix}"] > 5000
    # allowing the non-local states where nothing is non-local costs a little
    assert (
        immobile[f"full_minus_swr_only{suffix}"]
        < 0.01 * (immobile[f"swr_only_minus_none{suffix}"])
    )


@pytest.mark.parametrize("suffix", ["", "_two_sided"])
def test_replay_outside_swrs_is_what_the_full_model_gains(scenarios, suffix):
    masks, scores = scenarios
    sums = nr.restriction_sums(scores["B"], _labels(scenarios, "B"))
    immobile = nr.restriction_contrasts(sums, IMMOBILE, BIN_S)
    assert immobile[f"swr_only_minus_none{suffix}"] > 3000
    assert immobile[f"full_minus_swr_only{suffix}"] > 3000
    # the gain is in the remote events no SWR marks
    unmarked = gains(scores["B"], masks["B_unmarked"])
    assert (
        unmarked[f"full_minus_swr_only{suffix}"]
        > (immobile[f"full_minus_swr_only{suffix}"])
    )
    assert 0.3 < immobile[f"share_outside_swr{suffix}"] < 0.7


def _labels(scenarios, name: str) -> np.ndarray:
    """Running bins ``moving``; rest bins by scenario ``name``'s SWRs."""
    masks, _ = scenarios
    return np.where(
        masks["running"],
        "moving",
        np.where(masks[name], "immobile_in_swr", "immobile_outside_swr"),
    )


def test_the_library_record_checks_hold(chance):
    _, scores, record = chance
    assert record["all_finite"]
    for model in nr.MODELS:
        assert record[f"marginal_agrees_{model}"]
        assert record[f"causal_mass_deviation_{model}"] < 1e-3
    assert record["continuous_block_deviation"] < 1e-6
    assert list(scores.columns) == [
        "time",
        "none",
        "swr_only",
        "full",
        "none_two_sided",
        "swr_only_two_sided",
        "full_two_sided",
        "in_swr",
        "block",
    ]


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
    near = np.zeros(recording.time.size, dtype=bool)
    for s, e in zip(events.s, events.e, strict=True):
        near[max(0, s - 250) : e + 250] = True
    plain = ~recording.is_running & ~near
    assert abs(gain[plain].sum()) / (plain.sum() * BIN_S) < 1.0  # nats per s
