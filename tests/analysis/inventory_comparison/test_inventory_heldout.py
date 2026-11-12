"""Held-out evidence: an event's discovery content scored on evaluation cells."""

import numpy as np
import pytest

from continuum_swr_replay.analysis.inventory_comparison.heldout import (
    event_classes,
    expected_log_likelihood,
    heldout_score,
    interleaved_folds,
    nearest_window_content,
    poor_coverage_fold,
)

N_BINS = 40


def tuning(n_cells: int = 20, peak: float = 0.2, floor: float = 1e-4) -> np.ndarray:
    """Per-sample fields tiling 40 bins, a Gaussian each."""
    centers = np.linspace(0, N_BINS - 1, n_cells)
    bins = np.arange(N_BINS)
    return floor + peak * np.exp(-0.5 * ((bins[None, :] - centers[:, None]) / 2.0) ** 2)


def spikes_at(place: np.ndarray, fields: np.ndarray, seed: int = 0) -> np.ndarray:
    """Poisson counts from each cell's field at the given place per time bin."""
    return np.random.default_rng(seed).poisson(fields[:, place].T)


def at(place: np.ndarray) -> np.ndarray:
    """A content certain of the given place at each time bin."""
    content = np.zeros((place.size, N_BINS))
    content[np.arange(place.size), place] = 1.0
    return content


class TestExpectedLogLikelihood:
    def test_is_the_poisson_log_likelihood_split_in_two(self):
        content = np.array([[0.25, 0.75, 0.0], [0.0, 0.5, 0.5]])
        counts = np.array([[1, 0], [2, 1]])
        fields = np.array([[0.1, 0.2, 0.4], [0.3, 0.1, 0.2]])

        spike, silence = expected_log_likelihood(content, counts, fields)

        expected_spike = sum(
            content[t, x] * sum(counts[t, c] * np.log(fields[c, x]) for c in range(2))
            for t in range(2)
            for x in range(3)
        )
        expected_silence = -sum(
            content[t, x] * fields[:, x].sum() for t in range(2) for x in range(3)
        )
        assert spike == pytest.approx(expected_spike)
        assert silence == pytest.approx(expected_silence)

    def test_silence_where_fields_predict_firing_lowers_the_silence_term(self):
        fields = tuning()
        quiet_place = at(np.full(10, 0))
        busy_place = at(np.full(10, 20))
        silent = np.zeros((10, fields.shape[0]))

        _, quiet = expected_log_likelihood(quiet_place, silent, fields)
        _, busy = expected_log_likelihood(busy_place, silent, fields)

        # every cell's field is larger somewhere in the middle than at bin 0's edge
        assert busy < quiet


class TestHeldoutScore:
    fields = tuning()
    replay = np.repeat(np.arange(5, 35, 2), 4)  # a path across the track
    animal = np.full(replay.size, 38)  # the animal sits at the far end

    def permutations(self, n: int = 100) -> list[np.ndarray]:
        rng = np.random.default_rng(1)
        return [rng.permutation(self.fields.shape[0]) for _ in range(n)]

    def test_content_where_the_spikes_come_from_is_supported(self):
        counts = spikes_at(self.replay, self.fields)

        score = heldout_score(
            at(self.replay), counts, self.fields, self.animal, self.permutations()
        )

        assert score.gain_vs_animal > 0
        assert score.p_shuffle <= 0.05
        assert score.supported

    def test_content_somewhere_else_is_not(self):
        counts = spikes_at(self.replay, self.fields)
        elsewhere = (self.replay + 20) % N_BINS

        score = heldout_score(
            at(elsewhere), counts, self.fields, self.animal, self.permutations()
        )

        assert score.gain_vs_animal < 0 or score.p_shuffle > 0.05
        assert not score.supported

    def test_local_coding_does_not_beat_the_animal(self):
        counts = spikes_at(self.animal, self.fields)

        score = heldout_score(
            at(self.animal), counts, self.fields, self.animal, self.permutations()
        )

        assert score.gain_vs_animal == pytest.approx(0.0)
        assert not score.supported

    def test_permuted_fields_score_near_the_observed_only_by_chance(self):
        """Spikes with no relation to the content leave the p-value spread out."""
        rng = np.random.default_rng(4)
        p_values = [
            heldout_score(
                at(self.replay),
                spikes_at(rng.integers(0, N_BINS, self.replay.size), self.fields, seed),
                self.fields,
                self.animal,
                self.permutations(40),
            ).p_shuffle
            for seed in range(30)
        ]

        assert np.mean(np.array(p_values) <= 0.05) < 0.2

    def test_coverage_is_the_expected_number_of_evaluation_spikes(self):
        counts = spikes_at(self.replay, self.fields)

        score = heldout_score(
            at(self.replay), counts, self.fields, self.animal, self.permutations(5)
        )

        assert score.coverage == pytest.approx(self.fields[:, self.replay].sum())
        assert score.informative(1.0) and not score.informative(1e6)


class TestNearestWindowContent:
    def test_each_bin_takes_the_window_whose_midpoint_is_nearest(self):
        posterior = np.eye(3)
        midpoint_s = np.array([0.0, 0.1, 0.2])
        time = np.array([-0.05, 0.04, 0.06, 0.19, 0.3])

        content = nearest_window_content(posterior, midpoint_s, time)

        np.testing.assert_array_equal(content.argmax(axis=1), [0, 0, 1, 2, 2])


class TestEventClasses:
    def test_a_positive_overlap_is_shared_and_touching_is_not(self):
        ours = [(0.0, 1.0), (2.0, 3.0), (5.0, 6.0)]
        theirs = [(0.5, 1.5), (3.0, 4.0)]

        ours_classes, theirs_classes = event_classes(ours, theirs)

        assert ours_classes == ["shared", "ours_only", "ours_only"]
        assert theirs_classes == ["shared", "comparator_only"]


class TestFolds:
    def test_interleaved_folds_are_disjoint_and_swap_roles(self):
        folds = interleaved_folds(25)
        discovery, evaluation = folds["A"]

        assert not np.intersect1d(discovery, evaluation).size
        assert np.union1d(discovery, evaluation).size == 25
        np.testing.assert_array_equal(folds["B"][0], evaluation)
        np.testing.assert_array_equal(folds["B"][1], discovery)

    def test_interleaved_folds_each_cover_the_track(self):
        for discovery, evaluation in interleaved_folds(25).values():
            for cells in (discovery, evaluation):
                assert cells.min() <= 1 and cells.max() >= 23

    def test_the_poor_coverage_fold_evaluates_only_the_lower_half(self):
        discovery, evaluation = poor_coverage_fold(25)["C"]

        assert evaluation.max() < 12
        assert not np.intersect1d(discovery, evaluation).size
        assert discovery.min() == 1 and discovery.max() == 24
