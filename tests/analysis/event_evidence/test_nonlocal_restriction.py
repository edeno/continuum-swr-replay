"""Tests for the restricted decoders' predictive scores and their sums."""

import itertools

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence import nonlocal_restriction as nr
from continuum_swr_replay.decoding.transitions import nonlocal_transition_settings

# three discrete states with 1, 2 and 2 state bins
STATE_IND = np.array([0, 1, 1, 2, 2])
N_TIME = 6


def _toy_model(seed: int = 0):
    """Random ``pi``, ``C`` (each destination block's rows sum to 1), ``D`` and
    log-likelihoods."""
    rng = np.random.default_rng(seed)
    n = STATE_IND.size
    continuous = rng.uniform(0.1, 1.0, (n, n))
    for j in np.unique(STATE_IND):
        block = STATE_IND == j
        continuous[:, block] /= continuous[:, block].sum(axis=1, keepdims=True)
    discrete = rng.uniform(0.1, 1.0, (3, 3))
    discrete /= discrete.sum(axis=1, keepdims=True)
    initial = rng.uniform(0.1, 1.0, n)
    initial /= initial.sum()
    log_likelihood = rng.normal(0.0, 2.0, (N_TIME, n))
    return initial, continuous, discrete, log_likelihood


def _brute_force(initial, continuous, discrete, log_likelihood, allowed):
    """``log p(y_1..y_t)`` for every t and ``log p(y_t | y_{-t})`` for every t,
    from sums over every path of state bins of the restricted model, built from
    its definition (not from a filter's recursion).

    Returns
    -------
    prefix : ndarray, shape (n_time,)
        ``log p(y_1..y_t)``.
    two_sided : ndarray, shape (n_time,)
        ``log p(y_1..y_T) - log p(y_1..y_T with bin t's likelihood set to 1)``.
    """
    n = STATE_IND.size
    n_time = log_likelihood.shape[0]

    def joint(t):
        # T~_t[s', s] = C[s', s] D[d(s'), d(s)] a_t[d(s)] / sum_k D[d(s'), k] a_t[k]
        out = np.zeros((n, n))
        for source in range(n):
            i = STATE_IND[source]
            norm = sum(discrete[i, k] * allowed[t, k] for k in range(3))
            for destination in range(n):
                j = STATE_IND[destination]
                out[source, destination] = (
                    continuous[source, destination]
                    * discrete[i, j]
                    * allowed[t, j]
                    / norm
                )
        return out

    start = initial * allowed[0, STATE_IND]
    start = start / start.sum()
    transitions = [None] + [joint(t) for t in range(1, n_time)]

    def log_probability(likelihood, last):
        """log of the sum over paths z_0..z_last of
        p(z_0) prod_k T~_k(z_{k-1}, z_k) prod_k likelihood[k, z_k]."""
        paths = np.array(list(itertools.product(range(n), repeat=last + 1)))
        p = start[paths[:, 0]] * likelihood[0, paths[:, 0]]
        for k in range(1, last + 1):
            p = p * transitions[k][paths[:, k - 1], paths[:, k]]
            p = p * likelihood[k, paths[:, k]]
        return np.log(p.sum())

    likelihood = np.exp(log_likelihood)
    prefix = np.array([log_probability(likelihood, t) for t in range(n_time)])
    two_sided = []
    for t in range(n_time):
        without = likelihood.copy()
        without[t] = 1.0
        two_sided.append(prefix[-1] - log_probability(without, n_time - 1))
    return prefix, np.array(two_sided)


ALLOWED = {
    "everything": np.ones((N_TIME, 3), dtype=bool),
    "state 2 never": np.tile([True, True, False], (N_TIME, 1)),
    "time-varying": np.array(
        [
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    # the strict model's kind: state 2 alone, or states 0 and 1 alone
    "forced": np.array(
        [
            [1, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=bool,
    ),
}


def _scores(
    log_likelihood, initial, continuous, discrete, allowed, chunk_bins=nr.CHUNK_BINS
):
    return nr.predictive_log_scores(
        log_likelihood,
        initial,
        continuous,
        discrete,
        STATE_IND,
        allowed,
        chunk_bins=chunk_bins,
    )


@pytest.mark.parametrize("chunk_bins", [1, 2, nr.CHUNK_BINS])
@pytest.mark.parametrize("name", list(ALLOWED))
def test_the_scores_match_the_brute_force_sums_over_paths(name, chunk_bins):
    """The library's index convention (transition t is the move into t + 1)
    and the two-sided formula, against the model's definition; batches of 1
    and 2 bins put a batch boundary of the two-sided step at every bin."""
    initial, continuous, discrete, log_likelihood = _toy_model()
    allowed = ALLOWED[name]
    filter_score, two_sided, record = _scores(
        log_likelihood, initial, continuous, discrete, allowed, chunk_bins
    )
    prefix, expected_two_sided = _brute_force(
        initial, continuous, discrete, log_likelihood, allowed
    )
    np.testing.assert_allclose(np.cumsum(filter_score), prefix, rtol=0, atol=1e-5)
    np.testing.assert_allclose(two_sided, expected_two_sided, rtol=0, atol=1e-5)
    np.testing.assert_allclose(record["marginal"], prefix[-1], atol=1e-5)
    assert record["filter_identity_deviation"] < 1e-5
    assert record["causal_mass_deviation"] < 1e-5
    assert record["predicted_mass_deviation"] < 1e-5


def test_a_forbidden_state_holds_no_mass():
    """A state's likelihood cannot change the scores at bins where it is
    forbidden, so it has no predicted (or smoothed) mass there.

    The change is far beyond float32's ``exp`` range, so the test also fails if
    the likelihood of forbidden bins reaches the library's filter, which
    subtracts each bin's largest likelihood (the allowed bins would underflow).
    """
    initial, continuous, discrete, log_likelihood = _toy_model(2)
    allowed = ALLOWED["time-varying"]
    filter_score, two_sided, _ = _scores(
        log_likelihood, initial, continuous, discrete, allowed
    )
    changed = log_likelihood.copy()
    forbidden = ~allowed[:, STATE_IND]
    changed[forbidden] += 500.0
    rescored, two_sided_rescored, record = _scores(
        changed, initial, continuous, discrete, allowed
    )
    np.testing.assert_allclose(rescored, filter_score, atol=1e-5)
    np.testing.assert_allclose(two_sided_rescored, two_sided, atol=1e-5)
    assert record["causal_mass_deviation"] < 1e-5
    # where it is allowed, a change does move the scores
    changed = log_likelihood.copy()
    changed[~forbidden] += 5.0 * np.random.default_rng(0).random(
        int((~forbidden).sum())
    )
    assert not np.allclose(
        _scores(changed, initial, continuous, discrete, allowed)[0], filter_score
    )


def test_the_first_bins_likelihood_counts_only_where_the_initial_mass_is():
    """All the initial mass on state 0, and the first bin's likelihood 70 nats
    higher on every other state bin: the library's filter subtracts the
    largest likelihood of the bin before exponentiating and adds 1e-15 to the
    normalizer (non_local_detector 2d730c3a), so without leaving out the state
    bins with no initial mass the first bin's filtered mass collapses to about
    1e-15 and its scores are off by its log."""
    _, continuous, discrete, log_likelihood = _toy_model(4)
    initial = np.zeros(STATE_IND.size)
    initial[0] = 1.0
    log_likelihood[0] = [-70.0, 0.0, 0.0, 0.0, 0.0]
    allowed = ALLOWED["everything"]
    filter_score, two_sided, record = _scores(
        log_likelihood, initial, continuous, discrete, allowed
    )
    prefix, expected_two_sided = _brute_force(
        initial, continuous, discrete, log_likelihood, allowed
    )
    assert record["predicted_mass_deviation"] < 1e-5
    assert record["causal_mass_deviation"] < 1e-5
    assert record["filter_identity_deviation"] < 1e-5
    np.testing.assert_allclose(filter_score[0], -70.0, atol=1e-5)
    np.testing.assert_allclose(np.cumsum(filter_score), prefix, rtol=0, atol=1e-4)
    np.testing.assert_allclose(two_sided, expected_two_sided, rtol=0, atol=1e-4)


def test_a_missing_bin_scores_zero():
    initial, continuous, discrete, log_likelihood = _toy_model(3)
    log_likelihood[[0, 3, 5]] = 0.0
    for allowed in ALLOWED.values():
        filter_score, two_sided, _ = _scores(
            log_likelihood, initial, continuous, discrete, allowed
        )
        np.testing.assert_allclose(filter_score[[0, 3, 5]], 0.0, atol=1e-6)
        np.testing.assert_allclose(two_sided[[0, 3, 5]], 0.0, atol=1e-6)


def test_filter_identity_deviation_by_hand():
    predicted = np.array([[0.5, 0.5, 0.0], [0.2, 0.8, 0.0], [0.0, 0.0, 1.0]])
    log_likelihood = np.array(
        [[np.log(3.0), 0.0, -np.inf], [0.0, 0.0, -np.inf], [0.0, 0.0, -np.inf]]
    )
    causal = np.array([[0.75, 0.25, 0.0], [0.2, 0.8, 0.0], [0.0, 0.0, 1.0]])
    # the third bin's predicted mass is all on a bin with no likelihood: NaN
    assert np.isnan(nr.filter_identity_deviation(causal, predicted, log_likelihood))
    # the first bin updated, the second a missing bin (causal = predicted)
    assert nr.filter_identity_deviation(
        causal[:2], predicted[:2], log_likelihood[:2]
    ) == pytest.approx(0.0, abs=1e-15)
    # the arrays swapped: the first bin's update of (0.75, 0.25) is (0.9, 0.1)
    assert nr.filter_identity_deviation(
        predicted[:2], causal[:2], log_likelihood[:2]
    ) == pytest.approx(0.4)


def test_a_row_with_no_allowed_destination_raises():
    initial, continuous, _, log_likelihood = _toy_model()
    discrete = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="discrete state 2"):
        _scores(log_likelihood, initial, continuous, discrete, ALLOWED["time-varying"])


def test_the_inputs_must_agree():
    initial, continuous, discrete, log_likelihood = _toy_model()
    allowed = ALLOWED["everything"]
    with pytest.raises(ValueError, match="bins"):
        _scores(log_likelihood, initial, continuous, discrete, allowed[:-1])
    with pytest.raises(ValueError, match="2-D"):
        _scores(log_likelihood, initial, continuous, discrete[None], allowed)
    with pytest.raises(ValueError, match="in_swr"):
        nr.fold_fit_scores(
            None, np.arange(5.0), None, [], [], None, None, np.zeros(4, bool)
        )


CANONICAL_NAMES = ["Local", "No-Spike", "Non-Local Continuous", "Non-Local Fragmented"]


def _canonical_discrete() -> np.ndarray:
    """The decoders' discrete transitions before EM (P30-B), whose No-Spike row
    EM leaves as it is."""
    return np.asarray(nonlocal_transition_settings()["discrete_transition_type"].values)


def test_the_canonical_transitions_give_a_forced_entry_from_every_row():
    """Inside an SWR the ``swr`` model must be non-local: every row, the frozen
    No-Spike row too, enters Continuous or Fragmented with probability 1, and
    leaves them for the local states outside SWRs."""
    discrete = _canonical_discrete()
    allowed = nr.allowed_states(CANONICAL_NAMES, np.array([False, True]))["swr"]
    outside, inside = nr.restricted_discrete_transitions(discrete, allowed)
    np.testing.assert_allclose(inside.sum(axis=1), 1.0)
    assert (inside[:, :2] == 0).all() and (inside[:, 2:] > 0).all()
    np.testing.assert_allclose(outside.sum(axis=1), 1.0)
    assert (outside[:, 2:] == 0).all() and (outside[:, :2] > 0).all()
    # the No-Spike row's exits to the two non-local states, renormalized
    np.testing.assert_allclose(inside[1, 2:], [0.5, 0.5])


def test_a_row_that_cannot_reach_a_non_local_state_raises():
    discrete = _canonical_discrete()
    discrete[1] = [0.02, 0.98, 0.0, 0.0]
    allowed = nr.allowed_states(CANONICAL_NAMES, np.array([True]))["swr"]
    with pytest.raises(ValueError, match="discrete state 1"):
        nr.restricted_discrete_transitions(discrete, allowed)


def test_restricted_discrete_transitions_by_hand():
    discrete = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8]])
    patterns = np.array([[1, 1, 1], [1, 1, 0]], dtype=bool)
    restricted = nr.restricted_discrete_transitions(discrete, patterns)
    np.testing.assert_allclose(restricted[0], discrete)
    np.testing.assert_allclose(
        restricted[1], [[0.625, 0.375, 0.0], [0.25, 0.75, 0.0], [0.5, 0.5, 0.0]]
    )


def test_continuous_block_deviation_by_hand():
    continuous = np.array(
        [[1.0, 0.5, 0.5], [1.0, 0.25, 0.75], [1.0, 0.1, 0.8]]
    )  # last row's state-1 block sums to 0.9
    assert nr.continuous_block_deviation(continuous, [0, 1, 1]) == pytest.approx(0.1)


def test_allowed_states_by_model():
    names = ["Local", "No-Spike", "Non-Local Continuous", "Non-Local Fragmented"]
    allowed = nr.allowed_states(names, np.array([False, True, False]))
    np.testing.assert_array_equal(allowed["none"], [[1, 1, 0, 0]] * 3)
    np.testing.assert_array_equal(
        allowed["swr"], [[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0]]
    )
    np.testing.assert_array_equal(allowed["full"], [[1, 1, 1, 1]] * 3)


def _table() -> pd.DataFrame:
    """Five bins; two-sided scores are the filter scores plus 10."""
    table = pd.DataFrame(
        {
            "none": [-1.0, -2.0, -3.0, -1.0, -5.0],
            "swr": [-1.5, -1.0, -2.0, -1.0, -5.0],
            "full": [-0.5, -1.0, -2.0, -0.5, -4.0],
            "in_swr": [False, True, True, False, False],
        }
    )
    for model in nr.MODELS:
        table[f"{model}_two_sided"] = table[model] + 10.0
    return table


LABELS = ["outside", "inside", "inside", "outside", "moving"]


def test_restriction_sums_by_hand():
    sums = nr.restriction_sums(_table(), LABELS)
    assert sums.loc["inside"].to_dict() == {
        "n_bins": 2,
        "none": -5.0,
        "swr": -3.0,
        "full": -3.0,
        "none_two_sided": 15.0,
        "swr_two_sided": 17.0,
        "full_two_sided": 17.0,
    }
    assert sums.loc["outside", "full"] == -1.0
    assert sums.loc["moving", "n_bins"] == 1
    # with every bin labeled, the categories add up to the all-bin sums
    total = _table()[sums.columns[1:]].sum()
    pd.testing.assert_series_equal(sums.sum()[1:], total, check_names=False)
    assert sums["n_bins"].sum() == 5


def test_an_unlabeled_bin_is_in_no_category():
    sums = nr.restriction_sums(_table(), ["outside", None, "inside", np.nan, "moving"])
    assert sums["n_bins"].to_dict() == {"inside": 1, "moving": 1, "outside": 1}
    assert sums.loc["inside", "none"] == -3.0


def test_a_labeled_bin_with_a_score_that_is_not_finite_raises():
    table = _table()
    table.loc[1, "swr"] = np.nan
    table.loc[3, "full_two_sided"] = -np.inf
    with pytest.raises(ValueError, match="'swr': 1, 'full_two_sided': 1"):
        nr.restriction_sums(table, LABELS)
    # an unlabeled bin's score is not summed, so it may be anything
    sums = nr.restriction_sums(table, ["outside", None, "inside", None, "moving"])
    assert sums.loc["inside", "swr"] == -2.0


def test_restriction_contrasts_by_hand():
    sums = nr.restriction_sums(_table(), LABELS)
    inside = nr.restriction_contrasts(sums, ["inside"], bin_s=0.5)
    assert inside["n_bins"] == 2 and inside["seconds"] == 1.0
    assert inside["swr_minus_none"] == 2.0
    assert inside["swr_minus_none_per_s"] == 2.0
    assert inside["full_minus_swr"] == 0.0
    assert inside["swr_minus_none_two_sided"] == 2.0
    immobile = nr.restriction_contrasts(sums, ["inside", "outside"], bin_s=0.5)
    assert immobile["seconds"] == 2.0
    assert immobile["swr_minus_none"] == 1.5
    assert immobile["full_minus_swr"] == 1.5
    assert immobile["full_minus_none"] == 3.0
    assert immobile["full_minus_none_per_s"] == 1.5
    assert immobile["full_minus_swr_two_sided"] == 1.5
    # moving is its own category, not part of the immobile result
    moving = nr.restriction_contrasts(sums, ["moving"], bin_s=0.5)
    assert moving["full_minus_none"] == 1.0 and moving["full_minus_swr"] == 1.0


def test_share_outside_swr_by_hand():
    """full - none is 2 nats inside (bins 1, 2) and 1 outside (bins 0, 3)."""
    sums = nr.restriction_sums(_table(), LABELS)
    share = nr.share_outside_swr(sums, "inside", "outside")
    assert share == {
        "share_outside_swr": pytest.approx(1 / 3),
        "share_outside_swr_two_sided": pytest.approx(1 / 3),
    }
    # the moving bin's gain is in neither category
    assert nr.share_outside_swr(sums, "inside", "moving")["share_outside_swr"] == (
        pytest.approx(1 / 3)
    )
    assert nr.share_outside_swr(sums, "excluded", "outside")["share_outside_swr"] == 1


def test_the_share_is_nan_without_a_full_model_gain():
    table = _table()
    table["full"] = table["none"]
    table["full_two_sided"] = table["none_two_sided"]
    sums = nr.restriction_sums(table, LABELS)
    share = nr.share_outside_swr(sums, "inside", "outside")
    assert np.isnan(share["share_outside_swr"])
    assert np.isnan(share["share_outside_swr_two_sided"])
    empty = nr.restriction_contrasts(sums, ["excluded"], bin_s=0.002)
    assert empty["n_bins"] == 0
    assert np.isnan(empty["full_minus_none_per_s"])


def _frame(speed, excluded):
    return pd.DataFrame(
        {
            "head_speed": np.asarray(speed, dtype=float),
            "in_excluded_time": np.asarray(excluded, dtype=bool),
        },
        index=pd.Index(np.arange(len(speed)) * 0.002, name="time"),
    )


def test_each_bin_gets_its_category():
    # bins: excluded (slow, in SWR), excluded (fast), immobile in SWR,
    # immobile outside SWR, at the threshold (moving), fast (moving),
    # no speed (moving), immobile outside SWR
    frame = _frame(
        [1.0, 10.0, 0.5, 3.9, 4.0, 20.0, np.nan, 0.0],
        [True, True, False, False, False, False, False, False],
    )
    in_swr = np.array([True, False, True, False, True, True, True, False])
    labels = nr.bin_categories(frame, in_swr, speed_threshold=4.0)
    assert list(labels) == [
        "excluded",
        "excluded",
        "immobile_in_swr",
        "immobile_outside_swr",
        "moving",
        "moving",
        "moving",
        "immobile_outside_swr",
    ]
    assert list(labels.categories) == list(nr.CATEGORIES)


def test_categories_reject_a_mismatched_swr_mask():
    with pytest.raises(ValueError, match="shape"):
        nr.bin_categories(_frame([1.0, 2.0], [False, False]), np.array([True]))


def _record(**changes):
    record = {"continuous_block_deviation": 1e-12}
    for model in ("none", "swr", "full"):
        record |= {
            f"filter_identity_deviation_{model}": 1e-7,
            f"marginal_log_likelihood_{model}": -10.0,
            f"marginal_difference_{model}": 0.0,
            f"causal_mass_deviation_{model}": 1e-6,
            f"predicted_mass_deviation_{model}": 1e-6,
        }
    return record | changes


def test_a_clean_record_passes_the_checks():
    assert nr.check_failures(_record()) == []


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"filter_identity_deviation_full": 0.5}, "full: the filtered posterior"),
        ({"filter_identity_deviation_swr": float("nan")}, "swr: the filtered"),
        ({"causal_mass_deviation_none": 2e-3}, "none: filtered posterior mass"),
        ({"causal_mass_deviation_swr": float("nan")}, "swr: filtered"),
        ({"predicted_mass_deviation_full": 4e-4}, "full: predictive posterior mass"),
        ({"predicted_mass_deviation_none": float("nan")}, "none: predictive"),
        ({"continuous_block_deviation": 1e-3}, "continuous transitions' blocks"),
    ],
)
def test_each_failed_check_is_named(changes, expected):
    failures = nr.check_failures(_record(**changes))
    assert len(failures) == 1 and expected in failures[0]


def test_place_sampling_uses_the_folds_and_margin_it_is_given(monkeypatch):
    """``fold_fit_scores``' ``place_sampling_s`` comes from the folds and
    margin it is given, not the contiguous defaults. The fit, the likelihood,
    the models and the place bins are stubbed."""
    from types import SimpleNamespace

    from continuum_swr_replay.analysis.event_evidence import tetrode_holdout as th
    from continuum_swr_replay.decoding.results import ENCODING_KEY

    n = 2000
    time = np.arange(n) * 0.002
    rng = np.random.default_rng(5)
    folds = np.repeat(rng.permutation(np.arange(20) % 5), n // 20)
    place = (np.arange(n) // 7) % 30  # the animal's place bin
    is_training = rng.random(n) < 0.9
    margin = 13
    environment = object()
    canonical = SimpleNamespace(
        encoding_model_={ENCODING_KEY: {"environment": environment}},
        discrete_state_transitions_=np.eye(4),
        discrete_initial_conditions=np.array([1.0, 0, 0, 0]),
    )
    linear_position = np.arange(n, dtype=float)

    def fit(c, t, position, spikes, marks, training, missing, estimate_encoding_model):
        return SimpleNamespace(), {
            "discrete_transition": np.eye(4),
            "initial_discrete": np.array([1.0, 0, 0, 0]),
            "em_iterations": 3,
            "converged": True,
            "fit_s": 1.0,
        }

    def scores(detector, likelihood, in_swr, summary_bins, log):
        out = {c: np.zeros(n, dtype=np.float32) for c in nr.SCORE_COLUMNS}
        out |= {
            c: np.zeros(n, dtype=np.float32)
            for c in nr.STATE_PROBABILITY_COLUMNS.values()
        }
        out[nr.NONLOCAL_PLACE_BIN_COLUMN] = np.zeros(n, dtype=np.int32)
        return out, {}

    def places(env, position):
        assert env is environment
        np.testing.assert_array_equal(position, linear_position)
        return place

    monkeypatch.setattr(nr, "fold_log_likelihood", lambda *args: None)
    monkeypatch.setattr(nr, "model_scores", scores)
    monkeypatch.setattr(nr, "check_place_grid", lambda detector, c: None)
    monkeypatch.setattr(nr, "animal_place_bins", places)
    table, _ = nr.fold_fit_scores(
        canonical,
        time,
        np.zeros((n, 2)),
        [],
        [],
        is_training,
        np.zeros(n, dtype=bool),
        np.zeros(n, dtype=bool),
        folds=folds,
        margin_bins=margin,
        linear_position=linear_position,
        estimate_encoding_model=False,
        fit=fit,
        log=lambda message: None,
    )
    expected = th.place_sampling(place, is_training, folds, 5, margin, 0.002)
    np.testing.assert_array_equal(table["place_sampling_s"], expected)
    np.testing.assert_array_equal(table["block"], folds)
    # the contiguous defaults would give something else
    contiguous = th.place_sampling(
        place, is_training, th.time_blocks(time), 5, th.MARGIN_BINS, 0.002
    )
    assert not np.array_equal(expected, contiguous)


def test_smoothed_summary_by_hand():
    """States 0 (one bin) and 1, 2 (two place bins each): each state's mass,
    and the place bin with the most mass of states 1 and 2 together."""
    smoothed = np.array(
        [
            [0.1, 0.5, 0.0, 0.1, 0.3],  # place 0: 0.6, place 1: 0.3
            [0.2, 0.1, 0.2, 0.0, 0.5],  # place 0: 0.1, place 1: 0.7
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    probability, place = nr.smoothed_summary(
        smoothed, STATE_IND, [1, 2], np.array([0, 1]), chunk_bins=1
    )
    np.testing.assert_allclose(probability, [[0.1, 0.5, 0.4], [0.2, 0.3, 0.5]])
    np.testing.assert_array_equal(place, [0, 1])
    assert probability.dtype == np.float32 and place.dtype == np.int32
    with pytest.raises(ValueError, match="place bins"):
        nr.smoothed_summary(smoothed, np.array([0, 1, 1, 1, 2]), [1, 2], [0])


def test_the_summary_changes_no_score_and_is_the_smoothed_posterior():
    """With ``summary_rows`` the scores are the same to the bit, and the state
    probabilities are the library smoother's, summed by state."""
    from non_local_detector.core import chunked_filter_smoother_covariate_dependent

    initial, continuous, discrete, log_likelihood = _toy_model(3)
    allowed = ALLOWED["time-varying"]
    plain = _scores(log_likelihood, initial, continuous, discrete, allowed)
    rows = np.array([0, 2, 5])
    summarized = nr.predictive_log_scores(
        log_likelihood,
        initial,
        continuous,
        discrete,
        STATE_IND,
        allowed,
        summary_rows=rows,
        summed_states=[1, 2],
    )
    np.testing.assert_array_equal(plain[0], summarized[0])
    np.testing.assert_array_equal(plain[1], summarized[1])
    patterns, pattern = np.unique(allowed, axis=0, return_inverse=True)
    pattern = pattern.ravel()
    restricted = nr.restricted_discrete_transitions(discrete, patterns)
    first = initial * patterns[pattern[0]][STATE_IND]
    masked = np.where(patterns[pattern][:, STATE_IND], log_likelihood, -np.inf)
    smoothed = np.asarray(
        chunked_filter_smoother_covariate_dependent(
            time=np.arange(N_TIME),
            state_ind=STATE_IND,
            initial_distribution=first / first.sum(),
            discrete_transition_matrix=restricted[np.r_[pattern[1:], pattern[-1]]],
            continuous_transition_matrix=continuous,
            log_likelihood_func=None,
            log_likelihood_args=(),
            log_likelihoods=masked,
        )[0]
    )
    expected = np.stack([smoothed[:, STATE_IND == s].sum(axis=1) for s in range(3)], 1)
    np.testing.assert_allclose(
        summarized[2]["state_probability"], expected[rows], atol=1e-6
    )
    nonlocal_mass = smoothed[:, STATE_IND == 1] + smoothed[:, STATE_IND == 2]
    np.testing.assert_array_equal(
        summarized[2]["place_bin"], nonlocal_mass[rows].argmax(axis=1)
    )


def test_em_record_reads_the_librarys_loop():
    """The trace holds each iteration's E-step marginal, then the final one;
    the loop stops when the last two iterations' change is below the
    tolerance relative to their mean."""
    record = nr.em_record(np.array([-200.0, -100.0, -99.999, -99.99]), 1e-4)
    assert record == {
        "em_iterations": 3,
        "converged": True,
        "marginal_log_likelihoods": [-200.0, -100.0, -99.999, -99.99],
    }
    # stopped by max_iter, still changing
    record = nr.em_record(np.array([-200.0, -100.0, -90.0]), 1e-4)
    assert record["em_iterations"] == 2 and record["converged"] is False


def test_em_settings_are_the_librarys_defaults():
    from non_local_detector import NonLocalClusterlessDetector

    settings = nr.em_settings(False)
    assert settings["estimate_encoding_model"] is False
    assert settings["max_iter"] == 20 and settings["tolerance"] == 1e-4
    import inspect

    defaults = inspect.signature(NonLocalClusterlessDetector.estimate_parameters)
    assert defaults.parameters["estimate_encoding_model"].default is True
    assert set(settings) == {*nr.EM_SETTING_NAMES, "estimate_encoding_model"}


def _configured_detector(**changes):
    from non_local_detector import NonLocalClusterlessDetector
    from non_local_detector.environment import Environment

    from continuum_swr_replay.decoding.config import NonLocalDecodingConfig
    from continuum_swr_replay.decoding.fit import algorithm_params
    from continuum_swr_replay.decoding.transitions import nonlocal_detector_kwargs
    from continuum_swr_replay.parameters import PLACE_BIN_SIZE_1D

    config = NonLocalDecodingConfig()
    return NonLocalClusterlessDetector(
        environments=Environment(place_bin_size=PLACE_BIN_SIZE_1D),
        **{
            **algorithm_params(config, "clusterless"),
            **nonlocal_detector_kwargs(config),
            **changes,
        },
    )


def test_a_fresh_detector_never_starts_from_the_fitted_initial_distribution():
    """The library's EM writes its fitted initial distribution into the
    detector's ``discrete_initial_conditions`` setting; a fresh detector has
    the configured one."""
    configured = _configured_detector()
    canonical = _configured_detector()
    canonical.discrete_initial_conditions = np.array([0.9, 0.05, 0.05, 0.0])
    fresh = nr.fresh_detector(canonical)
    assert fresh is not canonical
    np.testing.assert_array_equal(
        fresh.discrete_initial_conditions, configured.discrete_initial_conditions
    )
    assert fresh.environments[0] is not canonical.environments[0]


def test_a_changed_configuration_raises():
    with pytest.raises(ValueError, match="local_position_std"):
        nr.fresh_detector(_configured_detector(local_position_std=4.0))
    with pytest.raises(ValueError, match="discrete_transition_concentration"):
        nr.fresh_detector(_configured_detector(discrete_transition_concentration=2.0))


def _fold(k, converged=True, iterations=4, **changes):
    return {
        "fold": k,
        "converged": converged,
        "em_iterations": iterations,
        **_record(**changes),
    }


def test_each_fold_failure_is_named_by_its_fold():
    record = {"folds": [_fold(0), _fold(1, causal_mass_deviation_swr=2e-3)]}
    failures = nr.fold_check_failures(record)
    assert len(failures) == 1 and failures[0].startswith("fold 1: swr: filtered")
    assert nr.fold_check_failures({"folds": [_fold(0)]}) == []


def test_a_fold_whose_em_did_not_converge_fails_the_check():
    record = {"folds": [_fold(0), _fold(1), _fold(2, converged=False, iterations=20)]}
    assert nr.fold_check_failures(record) == [
        "fold 2: EM did not converge in 20 iterations"
    ]


def test_a_fold_on_another_place_grid_raises():
    from types import SimpleNamespace

    from continuum_swr_replay.decoding.results import ENCODING_KEY

    def detector(centers, interior):
        environment = SimpleNamespace(
            place_bin_centers_=np.asarray(centers, dtype=float)[:, None],
            is_track_interior_=np.asarray(interior, dtype=bool),
        )
        return SimpleNamespace(
            encoding_model_={ENCODING_KEY: {"environment": environment}}
        )

    canonical = detector([1.0, 3.0, 5.0], [True, False, True])
    nr.check_place_grid(detector([1.0, 3.0, 5.0], [True, False, True]), canonical)
    with pytest.raises(ValueError, match="place_bin_centers_"):
        nr.check_place_grid(detector([1.0, 3.0, 5.5], [True, False, True]), canonical)
    with pytest.raises(ValueError, match="place_bin_centers_"):
        nr.check_place_grid(detector([1.0, 3.0], [True, False]), canonical)
    with pytest.raises(ValueError, match="is_track_interior_"):
        nr.check_place_grid(detector([1.0, 3.0, 5.0], [True, True, True]), canonical)


def test_fold_fit_scores_fits_each_fold_without_its_bins_and_stitches(monkeypatch):
    """Each fold's fit is given its encoding's training bins (the decode's,
    outside the fold and its margin) and the fold's bins as missing; each bin's
    columns come from its own fold's pass. The fit, likelihood and models are
    stubbed: pass k scores every bin k (state probabilities k / 10, place bin
    k)."""
    from types import SimpleNamespace

    from continuum_swr_replay.analysis.event_evidence import tetrode_holdout as th

    n = 3000
    time = np.arange(n) * 0.002
    rng = np.random.default_rng(2)
    folds = np.repeat(rng.permutation(np.arange(30) % 5), n // 30)
    is_training = rng.random(n) < 0.8
    is_missing = np.zeros(n, dtype=bool)
    is_missing[100:110] = True
    margin = 7
    canonical = SimpleNamespace(
        discrete_state_transitions_=np.eye(4),
        discrete_initial_conditions=np.array([1.0, 0, 0, 0]),
    )
    fits = []

    def fit(c, t, position, spikes, marks, training, missing, estimate_encoding_model):
        assert c is canonical and estimate_encoding_model is False
        fits.append((training.copy(), missing.copy()))
        k = len(fits) - 1
        return SimpleNamespace(k=k), {
            "discrete_transition": np.eye(4) * (1 - k / 10),
            "initial_discrete": np.array([1.0, 0, 0, 0]),
            "em_iterations": 3,
            "converged": True,
            "fit_s": 1.0,
        }

    def likelihood(detector, t, position, spikes, marks, missing):
        np.testing.assert_array_equal(missing, is_missing)
        return detector.k

    def scores(detector, k, in_swr, summary_bins, log):
        assert detector.k == k
        np.testing.assert_array_equal(summary_bins, folds == k)
        out = {c: np.full(n, k, dtype=np.float32) for c in nr.SCORE_COLUMNS}
        out |= {
            c: np.full(n, k / 10, dtype=np.float32)
            for c in nr.STATE_PROBABILITY_COLUMNS.values()
        }
        out[nr.NONLOCAL_PLACE_BIN_COLUMN] = np.full(n, k, dtype=np.int32)
        return out, {"continuous_block_deviation": 0.0}

    monkeypatch.setattr(nr, "fold_log_likelihood", likelihood)
    monkeypatch.setattr(nr, "model_scores", scores)
    monkeypatch.setattr(nr, "check_place_grid", lambda detector, c: None)
    table, record = nr.fold_fit_scores(
        canonical,
        time,
        np.zeros((n, 2)),
        [],
        [],
        is_training,
        is_missing,
        np.zeros(n, dtype=bool),
        folds=folds,
        margin_bins=margin,
        estimate_encoding_model=False,
        fit=fit,
        log=lambda message: None,
    )
    masks = th.fold_training_masks(is_training, folds, 5, margin)
    assert len(fits) == 5
    for k, (training, missing) in enumerate(fits):
        np.testing.assert_array_equal(training, masks[k])
        np.testing.assert_array_equal(missing, is_missing | (folds == k))
    for column in nr.SCORE_COLUMNS:
        np.testing.assert_array_equal(table[column], folds)
        assert table[column].dtype == np.float32
    np.testing.assert_array_equal(table["block"], folds)
    np.testing.assert_allclose(table["full_prob_local"], folds / 10)
    np.testing.assert_array_equal(table[nr.NONLOCAL_PLACE_BIN_COLUMN], folds)
    assert table[nr.NONLOCAL_PLACE_BIN_COLUMN].dtype == np.int32
    assert [f["fold"] for f in record["folds"]] == list(range(5))
    assert [f["n_bins"] for f in record["folds"]] == np.bincount(folds).tolist()
    assert record["folds"][2]["discrete_transition_difference"] == pytest.approx(0.2)
    assert record["n_missing"] == 10 and record["fit_s"] == 5.0
