"""One scoring rule for every detector, against a replay of known shape."""

import numpy as np
import pytest

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.scoring import (
    InjectionScore,
    best_scored,
    score_against_injection,
)

BOUNDS = (1.0, 1.2)
PATH_CM = 60.0
TOLERANCE_CM = 20.0


def injected_path(at):
    """A 60 cm path from 40 cm to 100 cm over the event, held at its ends outside it."""
    fraction = np.clip((np.asarray(at) - BOUNDS[0]) / (BOUNDS[1] - BOUNDS[0]), 0, 1)
    return 40.0 + PATH_CM * fraction


def score(times, decoded):
    return score_against_injection(
        np.asarray(times),
        np.asarray(decoded),
        injected_path,
        BOUNDS,
        PATH_CM,
        TOLERANCE_CM,
    )


class TestTheRule:
    def test_an_event_that_tracks_the_path_and_stays_on_it_matches(self):
        times = np.linspace(1.0, 1.2, 11)

        result = score(times, injected_path(times) + 3.0)

        assert result.matches
        assert result.position_error_cm == pytest.approx(3.0)
        assert result.span_cm == pytest.approx(PATH_CM)

    def test_an_event_somewhere_else_does_not_match(self):
        times = np.linspace(1.0, 1.2, 11)

        result = score(times, injected_path(times) + 50.0)

        assert not result.matches
        assert result.position_error_cm == pytest.approx(50.0)

    def test_the_whole_event_is_read_not_the_overlap(self):
        """An absorbed sequence tracks the injection during the injection and
        then runs off to the animal's own place. Scored on the overlap alone
        it would pass; scored on the whole event it fails on both counts."""
        during = np.linspace(1.0, 1.2, 11)
        after = np.linspace(1.21, 1.6, 40)
        times = np.concatenate([during, after])
        decoded = np.concatenate([injected_path(during), np.full(after.size, 170.0)])

        whole = score(times, decoded)
        overlap_only = score(during, injected_path(during))

        assert overlap_only.matches
        assert not whole.matches
        assert whole.position_error_cm > TOLERANCE_CM
        assert whole.span_cm > PATH_CM + 2 * TOLERANCE_CM

    def test_running_on_in_time_at_the_path_s_end_still_matches(self):
        """The window floor: a short replay reported longer than it is, with
        its decoded position parked where the replay stopped. That is the
        injected event, mis-measured in time, and must count."""
        during = np.linspace(1.0, 1.2, 11)
        after = np.linspace(1.21, 1.5, 30)
        times = np.concatenate([during, after])
        decoded = np.concatenate([injected_path(during), np.full(after.size, 100.0)])

        result = score(times, decoded)

        assert result.matches
        assert result.position_error_cm == pytest.approx(0.0)

    def test_a_short_path_absorbed_into_nearby_local_coding_does_not_match(self):
        """The span allowance is two tolerances, which for a 10 cm path is
        looser than the path itself; the position test over the whole event
        is what catches a sequence that is mostly local coding 30 cm away."""
        short_bounds, short_path = (1.0, 1.1), 10.0

        def short(at):
            f = np.clip((np.asarray(at) - 1.0) / 0.1, 0, 1)
            return 120.0 + short_path * f

        during = np.linspace(1.0, 1.1, 6)
        before = np.linspace(0.6, 0.99, 40)
        times = np.concatenate([before, during])
        decoded = np.concatenate([np.full(before.size, 150.0), short(during)])

        result = score_against_injection(
            times, decoded, short, short_bounds, short_path, TOLERANCE_CM
        )

        assert result.span_cm <= short_path + 2 * TOLERANCE_CM
        assert not result.matches

    def test_an_event_with_no_decoded_position_inside_the_injection_cannot_match(
        self,
    ):
        times = np.linspace(1.3, 1.5, 5)

        result = score(times, np.full(5, 70.0))

        assert not result.matches
        assert np.isnan(result.position_error_cm)
        assert result.span_cm == 0.0

    def test_an_empty_event_has_no_span(self):
        result = score(np.zeros(0), np.zeros(0))

        assert not result.matches
        assert np.isnan(result.span_cm)

    def test_mismatched_times_and_positions_raise(self):
        with pytest.raises(ValueError, match="decoded"):
            score(np.zeros(3), np.zeros(4))

    def test_the_span_allowance_is_one_tolerance_at_each_end(self):
        times = np.linspace(1.0, 1.2, 11)
        decoded = injected_path(times)
        decoded[0] -= TOLERANCE_CM
        decoded[-1] += TOLERANCE_CM

        assert score(times, decoded).matches
        decoded[-1] += 1.0
        assert not score(times, decoded).matches


class TestBestScored:
    def test_a_match_beats_an_earlier_non_match(self):
        """Scoring only the first overlapping event would let an absorbed
        sequence that starts earlier hide the true detection behind it."""
        scores = [
            InjectionScore(False, 5.0, 200.0),
            InjectionScore(True, 8.0, 60.0),
        ]

        assert best_scored(scores) == 1

    def test_among_matches_the_smaller_error_wins(self):
        scores = [
            InjectionScore(True, 8.0, 60.0),
            InjectionScore(True, 2.0, 60.0),
        ]

        assert best_scored(scores) == 1

    def test_a_finite_error_beats_a_missing_one(self):
        scores = [
            InjectionScore(False, float("nan"), 10.0),
            InjectionScore(False, 40.0, 10.0),
        ]

        assert best_scored(scores) == 1

    def test_nothing_to_choose_from_raises(self):
        with pytest.raises(ValueError):
            best_scored([])
