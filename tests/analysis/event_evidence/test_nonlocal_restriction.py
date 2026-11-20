"""Tests for the restricted decoders' predictive scores and their sums."""

import itertools

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence import nonlocal_restriction as nr

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
}


def _scores(log_likelihood, initial, continuous, discrete, allowed):
    return nr.predictive_log_scores(
        log_likelihood, initial, continuous, discrete, STATE_IND, allowed
    )


@pytest.mark.parametrize("name", list(ALLOWED))
def test_the_scores_match_the_brute_force_sums_over_paths(name):
    """The library's index convention (transition t is the move into t + 1)
    and the two-sided formula, against the model's definition."""
    initial, continuous, discrete, log_likelihood = _toy_model()
    allowed = ALLOWED[name]
    filter_score, two_sided, record = _scores(
        log_likelihood, initial, continuous, discrete, allowed
    )
    prefix, expected_two_sided = _brute_force(
        initial, continuous, discrete, log_likelihood, allowed
    )
    np.testing.assert_allclose(np.cumsum(filter_score), prefix, rtol=0, atol=1e-5)
    np.testing.assert_allclose(two_sided, expected_two_sided, rtol=0, atol=1e-5)
    np.testing.assert_allclose(record["marginal"], prefix[-1], atol=1e-5)
    assert record["marginal_agrees"]
    assert record["causal_mass_deviation"] < 1e-5


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


def test_a_missing_bin_scores_zero():
    initial, continuous, discrete, log_likelihood = _toy_model(3)
    log_likelihood[[0, 3, 5]] = 0.0
    for allowed in ALLOWED.values():
        filter_score, two_sided, _ = _scores(
            log_likelihood, initial, continuous, discrete, allowed
        )
        np.testing.assert_allclose(filter_score[[0, 3, 5]], 0.0, atol=1e-6)
        np.testing.assert_allclose(two_sided[[0, 3, 5]], 0.0, atol=1e-6)


def test_a_row_with_no_allowed_destination_raises():
    initial, continuous, _, log_likelihood = _toy_model()
    discrete = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="discrete state 2"):
        _scores(log_likelihood, initial, continuous, discrete, ALLOWED["time-varying"])


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
        allowed["swr_only"], [[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0]]
    )
    np.testing.assert_array_equal(allowed["full"], [[1, 1, 1, 1]] * 3)


def _table() -> pd.DataFrame:
    """Five bins; two-sided scores are the filter scores plus 10."""
    table = pd.DataFrame(
        {
            "none": [-1.0, -2.0, -3.0, -1.0, -5.0],
            "swr_only": [-1.0, -1.0, -2.0, -1.0, -5.0],
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
        "swr_only": -3.0,
        "full": -3.0,
        "none_two_sided": 15.0,
        "swr_only_two_sided": 17.0,
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


def test_restriction_contrasts_by_hand():
    sums = nr.restriction_sums(_table(), LABELS)
    inside = nr.restriction_contrasts(sums, ["inside"], bin_s=0.5)
    assert inside["n_bins"] == 2 and inside["seconds"] == 1.0
    assert inside["swr_only_minus_none"] == 2.0
    assert inside["swr_only_minus_none_per_s"] == 2.0
    assert inside["full_minus_swr_only"] == 0.0
    assert inside["swr_only_minus_none_two_sided"] == 2.0
    immobile = nr.restriction_contrasts(sums, ["inside", "outside"], bin_s=0.5)
    assert immobile["seconds"] == 2.0
    assert immobile["swr_only_minus_none"] == 2.0
    assert immobile["full_minus_swr_only"] == 1.0
    assert immobile["full_minus_none"] == 3.0
    assert immobile["full_minus_none_per_s"] == 1.5
    assert immobile["share_outside_swr"] == pytest.approx(1 / 3)
    assert immobile["share_outside_swr_two_sided"] == pytest.approx(1 / 3)
    # moving is its own category, not part of the immobile result
    moving = nr.restriction_contrasts(sums, ["moving"], bin_s=0.5)
    assert moving["full_minus_none"] == 1.0 and moving["full_minus_swr_only"] == 1.0


def test_the_share_is_nan_without_a_full_model_gain():
    table = _table()
    table["full"] = table["none"]
    sums = nr.restriction_sums(table, LABELS)
    contrasts = nr.restriction_contrasts(sums, ["inside", "outside"], bin_s=0.002)
    assert np.isnan(contrasts["share_outside_swr"])
    empty = nr.restriction_contrasts(sums, ["excluded"], bin_s=0.002)
    assert empty["n_bins"] == 0
    assert np.isnan(empty["full_minus_none_per_s"])


def test_an_unknown_score_family_raises():
    with pytest.raises(ValueError, match="score family"):
        nr.score_columns("smoothed")
