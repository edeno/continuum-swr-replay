"""The injected-replay sweep's scoring of the state-space detector's events."""

import numpy as np
import pytest

from continuum_swr_replay.analysis.detection.memoryless_sequence import (
    sweep,
    sweep_replays,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.sweep_replays import (
    score_state_space,
)

CENTERS = np.arange(0.0, 171.0, 10.0)
RECORDING = {
    "place_bin_centers_cm": CENTERS,
    "bin_distance_cm": np.abs(CENTERS[:, None] - CENTERS[None, :]),
}
BOUNDS = (1.0, 1.2)
PATH_CM = 60.0


def true_position_cm(at):
    """A 60 cm replay from 40 cm to 100 cm over 1.0-1.2 s."""
    return 40.0 + PATH_CM * np.clip((np.asarray(at) - 1.0) / 0.2, 0.0, 1.0)


def event(start_s, end_s, decoded=None):
    times = np.linspace(start_s, end_s, 11)
    positions = true_position_cm(times) if decoded is None else np.full(11, decoded)
    return sweep.ReportedEvent(start_s, end_s, times, positions)


ON_PATH = event(1.0, 1.2)
"""The whole replay: 200 ms, dispersion about 19 cm, so it meets the published
definition (longer than 100 ms, dispersion above 12 cm)."""
SHORT = event(1.05, 1.12)
"""On the path but 70 ms: it is the event, but not a replay by the definition."""
ELSEWHERE = event(1.0, 1.2, decoded=160.0)
"""Overlaps the injection but sits 60 cm or more from its path."""
AFTER = event(1.4, 1.6)
"""Does not overlap the injection."""


def run(*events):
    time = np.round(np.arange(0.5, 1.71, 0.01), 2)
    p_continuous = np.full(time.size, 0.05)
    p_fragmented = np.full(time.size, 0.05)
    p_continuous[time == 1.1], p_fragmented[time == 1.1] = 0.6, 0.3
    # higher, but outside the injection
    p_continuous[time == 1.5], p_fragmented[time == 1.5] = 0.9, 0.05
    return sweep.StateSpaceRun(
        time=time,
        p_continuous=p_continuous,
        p_fragmented=p_fragmented,
        canonical=list(events),
        continuous_only=[],
        non_local_posterior=np.zeros((time.size, CENTERS.size)),
    )


def score(*events):
    return score_state_space(
        RECORDING, run(*events), BOUNDS, BOUNDS, true_position_cm, PATH_CM
    )


def test_the_matching_event_answers_for_the_cell_whatever_comes_first():
    out = score(ELSEWHERE, SHORT)

    assert out["state_space_found"] and out["state_space_n_intervals"] == 2
    assert out["state_space_matches"]
    assert out["state_space_duration_s"] == pytest.approx(0.07)
    assert out["state_space_start_error_s"] == pytest.approx(0.05)
    assert out["state_space_end_error_s"] == pytest.approx(-0.08)


def test_a_match_too_short_for_the_published_definition_is_not_a_replay_match():
    assert not score(ELSEWHERE, SHORT)["state_space_replay_matches"]
    assert score(ELSEWHERE, ON_PATH)["state_space_replay_matches"]


def test_the_peak_probability_is_read_inside_the_injection_only():
    assert score(ON_PATH)["state_space_peak_probability"] == pytest.approx(0.9)


def test_an_event_outside_the_injection_is_not_found():
    out = score(AFTER)

    assert not out["state_space_found"]
    assert out["state_space_n_intervals"] == 0
    assert not out["state_space_matches"]
    assert np.isnan(out["state_space_position_error_cm"])


def test_planned_cells_run_every_combination_in_each_direction():
    intervals = np.array([[5.0, 8.0], [18.0, 21.0]])
    cells = sweep_replays.planned_cells(intervals, n_seeds=2)
    grid = sweep.replay_grid()
    held_still = sum(path_cm == 0 for path_cm, *_ in grid)

    # a path of 0 cm has no direction and is planned once
    assert len(cells) == 2 * (2 * len(grid) - held_still)
    assert not any(cell["path_cm"] == 0 for cell in cells if cell["direction"] == -1)
    # the same plan every time, each placement inside a usable stretch
    assert cells == sweep_replays.planned_cells(intervals, n_seeds=2)
    for cell in cells:
        assert any(
            start <= cell["at_s"] and cell["at_s"] + cell["duration_s"] <= end
            for start, end in intervals
        )


def test_every_planned_cell_has_its_own_placement_and_firing_noise():
    intervals = np.array([[5.0, 8.0], [18.0, 21.0]])
    cells = sweep_replays.planned_cells(intervals, n_seeds=2)

    assert len({cell["injection_seed"] for cell in cells}) == len(cells)
    assert len({cell["at_s"] for cell in cells}) == len(cells)


def test_a_planned_cell_does_not_depend_on_the_other_cells():
    intervals = np.array([[5.0, 8.0], [18.0, 21.0]])
    by_name = {
        n_seeds: {
            (
                cell["path_cm"],
                cell["midpoint_cm"],
                cell["duration_s"],
                cell["rate_hz"],
                cell["direction"],
                cell["seed"],
            ): cell
            for cell in sweep_replays.planned_cells(intervals, n_seeds=n_seeds)
        }
        for n_seeds in (1, 3)
    }
    # a cell planned among three replicates is the cell planned alone
    for name, cell in by_name[1].items():
        assert by_name[3][name] == cell
