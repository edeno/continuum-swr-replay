"""Invariant and correctness tests for analysis/behavior/tables/summaries.py.

Two styles of test:

- **Simple numpy** — hand-constructed per-event arrays probe the arithmetic
  invariants (sum invariants, bincount correctness, threshold behavior).
- **SyntheticEnv** — reuses the env constructed by
  ``simulation.make_synthetic_env`` to probe the junction-bin exclusion
  in ``posterior_to_arm_marginal`` end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest

from continuum_swr_replay.analysis.behavior.content.simulation import make_synthetic_env
from continuum_swr_replay.analysis.behavior.tables.summaries import (
    arm_count_hard,
    arm_count_soft,
    arm_fraction_from_event_total,
    arm_map_from_posterior,
    arm_posterior_mass,
    n_nonlocal_bins,
    nonlocal_weight_sum,
    posterior_mass_total,
    posterior_to_arm_marginal,
)


def test_arm_fraction_from_event_total_uses_matching_support() -> None:
    np.testing.assert_allclose(
        arm_fraction_from_event_total(
            np.array([2.0, 1.0, np.nan, 0.0]),
            np.array([4.0, 4.0, 4.0, 0.0]),
        ),
        [0.5, 0.25, np.nan, np.nan],
        equal_nan=True,
    )


def test_arm_fraction_from_event_total_rejects_shape_and_negative_values() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        arm_fraction_from_event_total(np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="nonnegative"):
        arm_fraction_from_event_total(np.array([-1.0]), np.array([1.0]))


# ── posterior_to_arm_marginal ──


class TestPosteriorToArmMarginal:
    @pytest.mark.parametrize(
        "posterior",
        [
            np.array([0.5, 0.5]),
            np.ones((1, 2, 1)) / 2,
        ],
    )
    def test_rejects_non_matrix_posterior(self, posterior):
        """A malformed axis layout must not fail later with an incidental error."""
        with pytest.raises(ValueError, match="must be 2-D"):
            posterior_to_arm_marginal(posterior, np.array([0, 1]), n_arms=2)

    def test_rejects_non_vector_edge_ids(self):
        posterior = np.array([[0.5, 0.5]])
        with pytest.raises(ValueError, match="edge_id must be 1-D"):
            posterior_to_arm_marginal(posterior, np.array([[0, 1]]), n_arms=2)

    @pytest.mark.parametrize(
        ("posterior", "message"),
        [
            (np.array([[np.nan, 1.0]]), "non-finite values on real-arm bins"),
            (np.array([[np.inf, 0.0]]), "infinite values"),
            (np.array([[1.1, -0.1]]), "negative probabilities"),
        ],
    )
    def test_rejects_invalid_probability_values(self, posterior, message):
        """Invalid values previously produced plausible but invalid arm masses."""
        with pytest.raises(ValueError, match=message):
            posterior_to_arm_marginal(posterior, np.array([0, 1]), n_arms=2)

    def test_allows_nan_only_on_junction_placeholders(self):
        """Real decoder artifacts store NaN at structural junction positions."""
        posterior = np.array([[0.75, 0.25, np.nan]])
        arm_post = posterior_to_arm_marginal(posterior, np.array([0, 1, -1]), n_arms=2)
        np.testing.assert_allclose(arm_post, [[0.75, 0.25]])

    def test_rejects_improper_probability_row_mass(self):
        """Do not silently turn an unnormalised score row into probabilities."""
        posterior = np.array([[0.2, 0.2]])
        with pytest.raises(ValueError, match="rows must sum to 1"):
            posterior_to_arm_marginal(posterior, np.array([0, 1]), n_arms=2)

    def test_accepts_small_probability_row_roundoff(self):
        """Float32-scale summation error is not a scientific contract failure."""
        posterior = np.array([[0.500004, 0.500004]])
        arm_post = posterior_to_arm_marginal(posterior, np.array([0, 1]), n_arms=2)
        np.testing.assert_allclose(arm_post, [[0.5, 0.5]])

    @pytest.mark.parametrize(
        ("edge_id", "message"),
        [
            (np.array([0.0, 1.5]), "finite integers"),
            (np.array([0, -2]), "-1 for junctions"),
            (np.array([0, 2]), "outside the canonical range"),
        ],
    )
    def test_rejects_invalid_edge_ids(self, edge_id, message):
        """Bad edge labels must not be truncated, dropped, or renormalized away."""
        posterior = np.array([[0.5, 0.5]])
        with pytest.raises(ValueError, match=message):
            posterior_to_arm_marginal(posterior, edge_id, n_arms=2)

    @pytest.mark.parametrize("n_arms", [0, -1, 2.5, True])
    def test_rejects_invalid_n_arms(self, n_arms):
        with pytest.raises(ValueError, match="positive integer"):
            posterior_to_arm_marginal(
                np.array([[0.5, 0.5]]), np.array([0, 1]), n_arms=n_arms
            )

    def test_sums_to_one_per_timebin(self):
        # 5 timebins, 10 position bins, 3 arms (edge_ids 0,0,0,1,1,1,2,2,-1,-1)
        edge_id = np.array([0, 0, 0, 1, 1, 1, 2, 2, -1, -1])
        rng = np.random.default_rng(0)
        posterior = rng.dirichlet(np.ones(10), size=5)  # (5, 10), rows sum to 1
        arm_post = posterior_to_arm_marginal(posterior, edge_id, n_arms=3)
        assert arm_post.shape == (5, 3)
        np.testing.assert_allclose(arm_post.sum(axis=1), 1.0)

    def test_all_junction_timebin_raises(self):
        """A timebin with no real-arm mass carries no arm information.

        This previously filled the row with a uniform ``1/n_arms`` and the test
        asserted that fill. The assertion described the implementation rather
        than specifying anything: uniform claims *equal evidence for every arm*
        where the truth is *no evidence*, and it does not stay contained --
        ``arm_map_from_posterior`` argmaxes the row and returns arm 0 (ties
        break low), giving the bin a confident and systematically biased arm,
        while the row credits ``1/n_arms`` of a real bin's mass to every arm in
        the content sums. Junction bins are interpolation placeholders, so a
        posterior sitting entirely on them means the decode is unusable for that
        bin and should say so.

        Fires on 0 of the shipped corpus's 2,971,614 in-event bins
        (``len(event_sequence_table)`` == the sum of ``n_nonlocal_bins``).
        The 65,811 quoted in the 2026-08-08 audit report is one session's bin
        count, not the corpus total.
        """
        edge_id = np.array([0, 0, 1, 1, -1, -1])
        posterior = np.zeros((2, 6))
        posterior[:, 4:] = 0.5  # all mass on junction bins
        with pytest.raises(ValueError, match="entire conditional posterior"):
            posterior_to_arm_marginal(posterior, edge_id, n_arms=2)

    def test_junction_bins_contribute_zero_when_real_mass_exists(self):
        """Junction mass is dropped and the row renormalizes over real arms."""
        edge_id = np.array([0, 0, 1, 1, -1, -1])
        posterior = np.zeros((2, 6))
        posterior[:, 0] = 0.25  # arm 0
        posterior[:, 2] = 0.25  # arm 1
        posterior[:, 4:] = 0.25  # junction, must be discarded
        arm_post = posterior_to_arm_marginal(posterior, edge_id, n_arms=2)
        np.testing.assert_allclose(arm_post, 0.5)
        np.testing.assert_allclose(arm_post.sum(axis=1), 1.0)

    def test_single_bin_per_arm(self):
        edge_id = np.array([0, 1, 2])
        posterior = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]], dtype=float)
        arm_post = posterior_to_arm_marginal(posterior, edge_id, n_arms=3)
        np.testing.assert_allclose(arm_post, posterior)

    def test_raises_when_all_bins_are_junction(self):
        edge_id = np.array([-1, -1, -1])
        posterior = np.ones((2, 3)) / 3
        with pytest.raises(ValueError, match="No real arm bins"):
            posterior_to_arm_marginal(posterior, edge_id)

    def test_synthetic_env_round_trip(self):
        """Delta posterior placed in arm k's region should collapse to arm k."""
        env = make_synthetic_env()
        edge_id = env.place_bin_centers_nodes_df_["edge_id"].to_numpy()
        n_bins = len(edge_id)
        # One delta per arm, placed at the first real bin of that arm.
        posterior = np.zeros((9, n_bins))
        for arm in range(9):
            first = np.flatnonzero(edge_id == arm)[0]
            posterior[arm, first] = 1.0
        arm_post = posterior_to_arm_marginal(posterior, edge_id, n_arms=9)
        # Should be an identity matrix (row arm → mass on arm column).
        np.testing.assert_allclose(arm_post, np.eye(9))


# ── Length measures ──


class TestLengthMeasures:
    def test_n_nonlocal_bins_counts_above_threshold(self):
        nonlocal_prob = np.array([0.9, 0.5, 0.85, 0.81, 0.2, 0.95])
        event_id = np.array([1, 1, 1, 2, 2, 2])
        event_ids, counts = n_nonlocal_bins(nonlocal_prob, event_id, threshold=0.8)
        np.testing.assert_array_equal(event_ids, [1, 2])
        # Event 1: 0.9, 0.85 above; 0.5 below → count 2
        # Event 2: 0.81, 0.95 above; 0.2 below → count 2
        np.testing.assert_array_equal(counts, [2, 2])

    def test_nonlocal_weight_sum_matches_groupby_sum(self):
        nonlocal_prob = np.array([0.9, 0.5, 0.85, 0.81, 0.2, 0.95])
        event_id = np.array([1, 1, 1, 2, 2, 2])
        event_ids, sums = nonlocal_weight_sum(nonlocal_prob, event_id)
        np.testing.assert_allclose(sums, [0.9 + 0.5 + 0.85, 0.81 + 0.2 + 0.95])

    def test_posterior_mass_total_equals_nonlocal_weight_sum(self):
        """Under the renormalization convention, these are identical."""
        nonlocal_prob = np.array([0.9, 0.85, 0.81, 0.95])
        event_id = np.array([1, 1, 2, 2])
        _, total = posterior_mass_total(nonlocal_prob, event_id)
        _, weight = nonlocal_weight_sum(nonlocal_prob, event_id)
        np.testing.assert_allclose(total, weight)

    def test_threshold_variants(self):
        nonlocal_prob = np.array([0.5, 0.7, 0.8, 0.9])
        event_id = np.array([0, 0, 0, 0])
        for th, expected in [(0.0, 4), (0.5, 3), (0.7, 2), (0.8, 1), (0.9, 0)]:
            _, counts = n_nonlocal_bins(nonlocal_prob, event_id, threshold=th)
            assert counts[0] == expected, f"threshold={th}"


# ── arm_count_hard and arm_count_soft ──


class TestArmCountHard:
    def test_sum_invariant(self):
        """sum(arm_count_hard) over arms == n_nonlocal_bins per event."""
        nonlocal_prob = np.array([0.9, 0.5, 0.85, 0.81, 0.2, 0.95])
        arm_map = np.array([0, 1, 2, 3, 4, 5])
        event_id = np.array([1, 1, 1, 2, 2, 2])
        _, counts = arm_count_hard(
            nonlocal_prob, arm_map, event_id, n_arms=9, threshold=0.8
        )
        _, n_bins = n_nonlocal_bins(nonlocal_prob, event_id, threshold=0.8)
        np.testing.assert_array_equal(counts.sum(axis=1), n_bins)

    def test_two_different_map_arms_in_same_event(self):
        """Scatter must correctly accumulate on distinct arms within one event."""
        nonlocal_prob = np.array([0.9, 0.9, 0.9, 0.9])
        arm_map = np.array([3, 5, 3, 7])  # three distinct arms, arm 3 twice
        event_id = np.array([1, 1, 1, 1])
        _, counts = arm_count_hard(
            nonlocal_prob, arm_map, event_id, n_arms=9, threshold=0.8
        )
        expected = np.zeros(9, dtype=int)
        expected[3] = 2
        expected[5] = 1
        expected[7] = 1
        np.testing.assert_array_equal(counts[0], expected)

    def test_placement(self):
        """Above-threshold timebins land on their MAP arm; below-threshold drop."""
        nonlocal_prob = np.array([0.9, 0.5, 0.85])
        arm_map = np.array([2, 5, 2])
        event_id = np.array([7, 7, 7])
        _, counts = arm_count_hard(
            nonlocal_prob, arm_map, event_id, n_arms=9, threshold=0.8
        )
        expected = np.zeros(9, dtype=int)
        expected[2] = 2  # timebins 0 and 2
        np.testing.assert_array_equal(counts[0], expected)

    def test_empty_event_gets_zero_row(self):
        """Event with zero above-threshold timebins has zero arm counts."""
        nonlocal_prob = np.array([0.5, 0.5])
        arm_map = np.array([0, 1])
        event_id = np.array([3, 3])
        _, counts = arm_count_hard(
            nonlocal_prob, arm_map, event_id, n_arms=9, threshold=0.8
        )
        np.testing.assert_array_equal(counts[0], np.zeros(9, dtype=int))


class TestArmCountSoft:
    def test_sum_invariant(self):
        """sum(arm_count_soft) over arms == nonlocal_weight_sum per event."""
        nonlocal_prob = np.array([0.9, 0.5, 0.85, 0.81, 0.2, 0.95])
        arm_map = np.array([0, 1, 2, 3, 4, 5])
        event_id = np.array([1, 1, 1, 2, 2, 2])
        _, soft = arm_count_soft(nonlocal_prob, arm_map, event_id, n_arms=9)
        _, weight = nonlocal_weight_sum(nonlocal_prob, event_id)
        np.testing.assert_allclose(soft.sum(axis=1), weight)

    def test_agrees_with_hard_at_threshold_zero_all_w_one(self):
        """When w_t = 1 everywhere and threshold = 0, hard and soft match."""
        nonlocal_prob = np.ones(6)
        arm_map = np.array([0, 1, 0, 1, 2, 2])
        event_id = np.array([1, 1, 2, 2, 2, 1])
        _, hard = arm_count_hard(
            nonlocal_prob, arm_map, event_id, n_arms=3, threshold=0.0
        )
        _, soft = arm_count_soft(nonlocal_prob, arm_map, event_id, n_arms=3)
        np.testing.assert_allclose(hard, soft)


# ── arm_posterior_mass ──


class TestArmPosteriorMass:
    def test_sum_invariant(self):
        """sum(arm_posterior_mass) over arms == posterior_mass_total."""
        rng = np.random.default_rng(42)
        T, n_arms = 20, 5
        arm_post = rng.dirichlet(np.ones(n_arms), size=T)  # rows sum to 1
        nonlocal_prob = rng.uniform(0.2, 1.0, size=T)
        event_id = np.repeat([10, 11, 12, 13], T // 4)
        _, mass = arm_posterior_mass(nonlocal_prob, arm_post, event_id)
        _, total = posterior_mass_total(nonlocal_prob, event_id)
        np.testing.assert_allclose(mass.sum(axis=1), total)

    def test_delta_posterior_matches_soft_counts(self):
        """With a delta posterior on MAP arm, posterior_mass equals soft count."""
        T, n_arms = 6, 9
        arm_map = np.array([3, 5, 3, 7, 7, 3])
        arm_post = np.zeros((T, n_arms))
        arm_post[np.arange(T), arm_map] = 1.0
        nonlocal_prob = np.array([0.9, 0.5, 0.85, 0.81, 0.2, 0.95])
        event_id = np.array([0, 0, 0, 1, 1, 1])

        _, soft = arm_count_soft(nonlocal_prob, arm_map, event_id, n_arms=n_arms)
        _, mass = arm_posterior_mass(nonlocal_prob, arm_post, event_id)
        np.testing.assert_allclose(mass, soft)


# ── arm_map_from_posterior ──


class TestArmMapFromPosterior:
    def test_argmax(self):
        arm_post = np.array([[0.1, 0.7, 0.2], [0.4, 0.3, 0.3], [0.1, 0.1, 0.8]])
        np.testing.assert_array_equal(arm_map_from_posterior(arm_post), [1, 0, 2])
