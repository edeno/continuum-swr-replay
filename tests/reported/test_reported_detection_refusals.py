"""The detection summary refuses inputs that contradict a fact its caption states.

The sweeps' summaries kept in the repository are accepted as they are; each test
changes one of them so that one stated fact no longer holds.
"""

from __future__ import annotations

import pandas as pd
import pytest

from continuum_swr_replay.analysis.detection.memoryless_sequence import (
    sweep_statistics,
)
from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported import detection


def _summary(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT_DIR / detection.SWEEP_DIR / f"detection_{name}_summary.csv")


@pytest.fixture(scope="module")
def sweep() -> pd.DataFrame:
    return _summary("sweep")


@pytest.fixture(scope="module")
def controls() -> pd.DataFrame:
    return _summary("controls")


@pytest.fixture(scope="module")
def heldout() -> pd.DataFrame:
    return _summary("heldout")


def _with_value(table: pd.DataFrame, value: float, **where: str) -> pd.DataFrame:
    """``table`` with ``value`` in the first row matching ``where``."""
    rows = pd.Series(True, index=table.index)
    for column, wanted in where.items():
        rows &= table[column] == wanted
    changed = table.copy()
    changed.loc[rows[rows].index[0], "value"] = value
    return changed


def test_the_repositorys_summaries_are_accepted(sweep, controls, heldout):
    for build, table in (
        (detection.sweep_values, sweep),
        (detection.controls_values, controls),
        (detection.heldout_values, heldout),
    ):
        values = build(table, "source")
        assert values
        assert {value.function for value in values} == {build.__name__}


@pytest.mark.parametrize(
    ("edge", "error_ms"), [("start", 1.0), ("start", 0.0), ("end", -1.0), ("end", 0.0)]
)
def test_a_sliding_window_that_does_not_start_early_and_end_late_is_refused(
    sweep, edge, error_ms
):
    changed = _with_value(
        sweep,
        error_ms,
        summary="timing_by_duration",
        measure=f"sliding_{edge}_error_ms_median",
    )
    with pytest.raises(ValueError, match="starts early and ends late"):
        detection.sweep_values(changed, "source")


def test_jump_pass_rates_and_timing_errors_at_other_durations_are_refused(sweep):
    # without its 40 ms rows, the shortest replays of the pass rates are 60 ms
    # ones, and those of the timing errors still 40 ms ones
    shortest = (sweep["summary"] == "shuffle_pass_rate_by_duration") & (
        sweep["duration_s"] == 0.04
    )
    assert shortest.sum() == 2
    with pytest.raises(ValueError, match="not of the same durations"):
        detection.sweep_values(sweep[~shortest], "source")


def test_the_sweep_descriptions_state_the_durations_the_summary_holds(sweep):
    # every true duration doubled: 80 to 1000 ms in place of 40 to 500 ms
    doubled = sweep.assign(duration_s=2 * sweep["duration_s"])
    description = {
        value.key: value.description
        for value in detection.sweep_values(doubled, "source")
    }
    assert "shortest (80 ms)" in description["sweep_jump_half_pass_shortest_fraction"]
    assert "longest (1000 ms)" in description["sweep_jump_half_pass_longest_fraction"]
    assert "shortest (80 ms)" in description["sweep_shortest_start_early_ms"]
    assert "shortest (80 ms)" in description["sweep_shortest_reported_duration_ms"]
    assert "(80 to 1000 ms)" in description["sweep_sliding_end_late_max_ms"]


def test_an_event_where_there_was_no_replay_is_refused(controls):
    changed = _with_value(controls, 1.0, summary="no_replay", measure="n_events")
    with pytest.raises(ValueError, match="no output reported an event"):
        detection.controls_values(changed, "source")


def test_a_controls_summary_without_the_no_replay_counts_is_refused(controls):
    counts = (controls["summary"] == "no_replay") & (controls["measure"] == "n_events")
    with pytest.raises(ValueError, match="no output reported an event"):
        detection.controls_values(controls[~counts], "source")


def _with_sliding_trials(
    controls: pd.DataFrame, condition: str, n_trials: float
) -> pd.DataFrame:
    return _with_value(
        controls,
        n_trials,
        summary="parts",
        measure="n_trials",
        condition=condition,
        detector="sliding_published",
    )


# each jump condition has 96 trials for both detectors
@pytest.mark.parametrize("n_trials", [7.0, 100.0])
def test_detectors_scored_on_different_trials_are_refused(controls, n_trials):
    changed = _with_sliding_trials(controls, "jump_120ms", n_trials)
    with pytest.raises(ValueError, match="different multi-part trials"):
        detection.controls_values(changed, "source")


def test_detectors_scored_on_as_many_trials_of_other_conditions_are_refused(controls):
    # 100 and 92 trials of the two jump conditions: the same 192 in all
    changed = _with_sliding_trials(
        _with_sliding_trials(controls, "jump_120ms", 100.0), "jump_200ms", 92.0
    )
    with pytest.raises(ValueError, match="different multi-part trials"):
        detection.controls_values(changed, "source")


SHARED_REPLAYS = {
    "summary": "by_class",
    "estimator": "state_space",
    "detector": "state_space",
    "event_class": "shared",
    "truth": "continuous_replay",
}


# the informative shared replays of each fold; all have held-out support
@pytest.mark.parametrize(("fold", "n_events"), [("A", 137), ("B", 173), ("C", 23)])
def test_shared_replays_whose_support_differs_between_folds_are_refused(
    heldout, fold, n_events
):
    assert (
        sweep_statistics.pooled_heldout_rate(
            heldout, fold, ("shared",), "continuous_replay"
        )["n_events"]
        == n_events
    )
    changed = _with_value(
        heldout,
        (n_events - 1) / n_events,
        measure="support_rate",
        fold=fold,
        **SHARED_REPLAYS,
    )
    with pytest.raises(ValueError, match="one held-out support rate"):
        detection.heldout_values(changed, "source")


def test_a_fold_with_no_informative_shared_replay_is_refused(heldout):
    # with no rate in fold C there is no one rate "the same in every fold"
    changed = _with_value(
        _with_value(heldout, 0.0, measure="n_events", fold="C", **SHARED_REPLAYS),
        float("nan"),
        measure="support_rate",
        fold="C",
        **SHARED_REPLAYS,
    )
    with pytest.raises(ValueError, match="one held-out support rate"):
        detection.heldout_values(changed, "source")


@pytest.mark.parametrize("estimator", ["state_space", "sliding_window"])
def test_a_placeless_burst_with_heldout_support_is_refused(heldout, estimator):
    # 1 of fold A's 10 informative place-less events
    changed = _with_value(
        heldout,
        1 / 10,
        summary="by_class",
        measure="support_rate",
        fold="A",
        estimator=estimator,
        detector="state_space",
        event_class="ours_only",
        truth="no_content",
    )
    with pytest.raises(ValueError, match="no informative place-less burst"):
        detection.heldout_values(changed, "source")


def test_an_event_only_the_sliding_window_reported_is_refused(heldout):
    row = heldout[heldout["summary"] == "by_class"].iloc[[0]]
    changed = pd.concat(
        [heldout, row.assign(event_class="comparator_only")], ignore_index=True
    )
    with pytest.raises(ValueError, match="comparator-only events"):
        detection.heldout_values(changed, "source")


def _pilot(decided_call_passes: bool) -> pd.DataFrame:
    """Three replay calls, the second decided by its outside window, and one
    candidate that is no replay call."""
    return pd.DataFrame(
        {
            "is_replay": [True, True, True, False],
            "n_kept_windows_outside_pieces": [0, 1, 1, 1],
            "pieces_dispersion_cm": [20.0, 5.0, 30.0, 5.0],
            "shuffle_pass": [True, decided_call_passes, True, False],
        }
    )


def test_the_pilot_values_count_calls_and_those_an_outside_window_decides():
    values = detection.pilot_stray_window_values(_pilot(True), 12.0, "source")
    assert {value.key: value.value for value in values} == {
        "pilot_replay_calls": 3,
        "pilot_calls_decided_by_outside_window": 1,
        "pilot_calls_decided_by_outside_window_passing": 1,
    }


def test_a_decided_pilot_call_that_fails_the_significance_test_is_counted():
    """The caption reports how many decided calls pass; it no longer asserts that
    every one does (the lead's rewording, 2026-11-14), so a failing one is
    counted, not refused."""
    values = detection.pilot_stray_window_values(_pilot(False), 12.0, "source")
    assert {value.key: value.value for value in values}[
        "pilot_calls_decided_by_outside_window_passing"
    ] == 0
