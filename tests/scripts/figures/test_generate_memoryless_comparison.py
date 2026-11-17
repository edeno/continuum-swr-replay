"""The memoryless comparison figure's checks of what it commits to."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from scripts.figures import generate_memoryless_comparison as figure  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    """The committed sweep summary the figure reads."""
    return pd.read_csv(REPO / figure.SUMMARY_PATH)


def test_the_legend_settings_are_the_codes():
    figure.check_caption_settings(figure.CONFIG)


def test_committed_sweep_summary_passes(summary):
    figure.check_summary_claims(summary)


def test_a_sliding_window_report_of_a_replay_held_still_is_refused(summary):
    held_still = (
        (summary["summary"] == "detection_rate_by_path_and_duration")
        & (summary["measure"] == "sliding_matches")
        & (summary["path_cm"] == 0.0)
    )
    assert held_still.sum() > 0
    changed = summary.copy()
    changed.loc[held_still, "value"] = 0.1
    with pytest.raises(RuntimeError, match="no replay held"):
        figure.check_summary_claims(changed)


def test_the_examples_name_distinct_events_and_end_with_the_control():
    events = [example.event for example in figure.EXAMPLES]
    assert len(set(events)) == len(events)
    # the replays both detectors report come first, the control neither does last
    expected = [example.expected for example in figure.EXAMPLES]
    assert expected == sorted(expected, key=["found", "declined"].index)
    assert figure.EXAMPLES[-1].event == figure.SILENCE_EVENT
