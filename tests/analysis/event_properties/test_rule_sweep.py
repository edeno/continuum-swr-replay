"""The event rule's sensitivity check, on counts small enough to check by hand."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_properties import rule_sweep as sweep


def test_the_rule_is_among_the_swept_values():
    assert sweep.RULE == {
        "probability_threshold": 0.8,
        "minimum_samples": 7,
        "swr_threshold_z": 2.0,
    }
    for name, values in sweep.SETTINGS.items():
        assert sweep.RULE[name] in values
    # 0 to 6 z by 0.5, the earlier 1, 1.5, 2, 2.5, 3 and 4 among them
    assert sweep.SWR_THRESHOLDS_Z[0] == 0.0 and sweep.SWR_THRESHOLDS_Z[-1] == 6.0
    assert {1.0, 1.5, 2.0, 2.5, 3.0, 4.0} <= set(sweep.SWR_THRESHOLDS_Z)
    assert len(sweep.SWR_THRESHOLDS_Z) == 13


def _events(starts, is_swr) -> pd.DataFrame:
    starts = np.asarray(starts, dtype=float)
    return pd.DataFrame(
        {"event_start": starts, "event_end": starts + 0.02, "is_swr": is_swr}
    )


def test_epoch_counts_detect_again_at_each_setting_and_count_those_in_a_visit(
    monkeypatch,
):
    """Each setting's events (one starts outside the visit, 0 to 10 s, and is
    not counted); the rule's events are detected once; the criterion counts the
    events below each SWR threshold and how often its 2 z label agrees."""
    calls = []

    def detect(raw, probability_threshold, minimum_samples):
        calls.append((probability_threshold, minimum_samples))
        n = int(100 * probability_threshold) // 10  # 0.5 -> 5 ... 0.95 -> 9
        is_swr = np.arange(n) % 2 == 0
        return _events([*np.arange(n) + 0.5, 50.0], [*is_swr, False])

    monkeypatch.setattr(sweep, "detect_classified_events", detect)
    # the rule's 8 events: criteria 0 to 3.5 by 0.5
    monkeypatch.setattr(
        sweep, "kay_criterion", lambda z, first, last: np.arange(8) * 0.5
    )
    raw = pd.DataFrame({"SWR_zscore": 0.0}, index=np.arange(0, 60, 0.002))
    counts = sweep.epoch_sweep_counts(raw, np.array([0.0]), np.array([10.0]))
    assert len(calls) == len(set(calls)) == 6 + 5  # the rule detected once
    by_value = counts.set_index(["setting", "value"])
    assert by_value.loc[("probability_threshold", 0.5), "n_events"] == 5
    assert by_value.loc[("probability_threshold", 0.8), "n_events"] == 8
    assert by_value.loc[("probability_threshold", 0.8), "n_no_swr"] == 4
    assert by_value.loc[("minimum_samples", 7.0), "n_events"] == 8
    # SWR thresholds: criteria below 1 z are 0, 0.5; below 2 z, 4 of them
    assert by_value.loc[("swr_threshold_z", 1.0), "n_no_swr"] == 2
    assert by_value.loc[("swr_threshold_z", 2.0), "n_no_swr"] == 4
    assert by_value.loc[("swr_threshold_z", 4.0), "n_no_swr"] == 8
    assert by_value.loc[("swr_threshold_z", 0.0), "n_no_swr"] == 0
    assert by_value.loc[("swr_threshold_z", 6.0), "n_no_swr"] == 8
    # at 2 z the criterion calls events 4-7 SWR; the table calls 0, 2, 4, 6
    assert (by_value.xs("swr_threshold_z")["n_criterion_agrees"] == 4).all()


def _counts() -> pd.DataFrame:
    """Two animals: a with epochs e1 (10 events at the rule, 5 no-SWR) and e2
    (10, 3), b with e1 (20, 12); at probability 0.9 a has 8/4 and 6/2, b
    16/10; e2 of a has no event at 25 samples."""
    rows = []
    for animal, epoch, rule, high in (
        ("a", "e1", (10, 5), (8, 4)),
        ("a", "e2", (10, 3), (6, 2)),
        ("b", "e1", (20, 12), (16, 10)),
    ):
        for setting, values in sweep.SETTINGS.items():
            for value in values:
                n, k = high if value in (0.9, 0.95, 15, 25) else rule
                if (animal, epoch, setting, value) == (
                    "a",
                    "e2",
                    "minimum_samples",
                    25,
                ):
                    n, k = 0, 0
                rows.append(
                    {
                        "epoch_id": f"{animal}{epoch}", "animal": animal,
                        "session_date": 1, "epoch": epoch, "setting": setting,
                        "value": float(value), "n_events": n, "n_no_swr": k,
                        "n_criterion_agrees": n - 1
                        if setting == "swr_threshold_z" else np.nan,
                    }
                )  # fmt: skip
    return pd.DataFrame(rows)


def test_sweep_statistics_pool_the_events_and_average_each_animals_epochs():
    stats = sweep.sweep_statistics(_counts()).set_index(["setting", "value"])
    assert len(stats) == sum(len(v) for v in sweep.SETTINGS.values())
    rule = stats.loc[("probability_threshold", 0.8)]
    assert rule["is_rule"] and rule["n_events"] == 40
    assert rule["no_swr_fraction"] == pytest.approx(20 / 40)
    # a: mean of 0.5 and 0.3; b: 0.6
    np.testing.assert_allclose(rule["animals"], [0.4, 0.6])
    assert rule["no_swr_fraction_animal_mean"] == pytest.approx(0.5)
    assert list(rule["animal_names"]) == ["a", "b"]
    assert (rule["animal_min"], rule["animal_max"]) == pytest.approx((0.4, 0.6))
    high = stats.loc[("probability_threshold", 0.9)]
    assert not high["is_rule"]
    assert high["events_vs_rule"] == pytest.approx(30 / 40)
    assert high["no_swr_fraction"] == pytest.approx(16 / 30)
    # a's e2 has no event at 25 samples: a's share is e1's alone
    longest = stats.loc[("minimum_samples", 25.0)]
    np.testing.assert_allclose(longest["animals"], [0.5, 10 / 16])
    assert longest["no_swr_fraction_animal_mean"] == pytest.approx((0.5 + 10 / 16) / 2)
    assert sweep.criterion_agreement(_counts()) == pytest.approx(37 / 40)


def test_sweep_statistics_refuse_counts_made_with_other_settings():
    """Counts without a row at a swept value (made before it was swept) are
    refused, naming the value and the runner, not read as no events."""
    counts = _counts()
    old = counts[
        ~((counts["setting"] == "minimum_samples") & (counts["value"] == 50.0))
    ]
    with pytest.raises(ValueError, match="minimum_samples 50.*sweep_event_rule"):
        sweep.sweep_statistics(old)
