"""The event rule sweep refuses counts whose rule settings are not the
canonical events."""

import pandas as pd
import pytest

import scripts.event_properties.sweep_event_rule as runner


def _events() -> pd.DataFrame:
    """Epoch a1 of animal a: 3 events, 1 with no SWR."""
    return pd.DataFrame(
        {
            "animal": "a",
            "session_date": 20200212,
            "epoch": "02_r1",
            "is_swr": [True, True, False],
        }
    )


def _counts(n_events: int, n_no_swr: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "epoch_id": "a20200212_02_r1_HPC",
            "setting": ["probability_threshold", "probability_threshold"],
            "value": [0.8, 0.9],
            "n_events": [n_events, 1],
            "n_no_swr": [n_no_swr, 0],
        }
    )


def test_counts_at_the_rule_equal_to_the_table_pass():
    runner.check_rule_counts(_counts(3, 1), _events())


@pytest.mark.parametrize(("n_events", "n_no_swr"), [(4, 1), (3, 2)])
def test_counts_at_the_rule_that_differ_from_the_table_are_refused(n_events, n_no_swr):
    with pytest.raises(ValueError, match="1 epochs' counts are not the canonical"):
        runner.check_rule_counts(_counts(n_events, n_no_swr), _events())
