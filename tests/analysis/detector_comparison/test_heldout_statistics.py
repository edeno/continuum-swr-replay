"""The statistics of the held-out check's summary, on summaries small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.detector_comparison import (
    heldout_statistics as stats,
)


def _heldout(rows: list[tuple]) -> pd.DataFrame:
    """A held-out summary from (block, measure, fold, estimator, detector,
    event_class, truth, value) rows."""
    return pd.DataFrame(
        rows,
        columns=[
            "summary",
            "measure",
            "fold",
            "estimator",
            "detector",
            "event_class",
            "truth",
            "value",
        ],
    )


def _by_class(
    fold: str,
    event_class: str,
    truth: str,
    measures: dict[str, float],
    estimator: str = "state_space",
    detector: str = "state_space",
) -> list[tuple]:
    return [
        ("by_class", measure, fold, estimator, detector, event_class, truth, value)
        for measure, value in measures.items()
    ]


REPLAY = "continuous_replay"
HELDOUT_ROWS = [
    # fold A: 2 of 4 informative ours-only replays and 12 of 16 shared ones have
    # support; 4 of 10 and 16 of 20 are informative
    *_by_class(
        "A",
        "ours_only",
        REPLAY,
        {
            "n_events_all": 10,
            "informative_share": 0.4,
            "n_events": 4,
            "support_rate": 0.5,
        },
    ),
    *_by_class(
        "A",
        "shared",
        REPLAY,
        {
            "n_events_all": 20,
            "informative_share": 0.8,
            "n_events": 16,
            "support_rate": 0.75,
        },
    ),
    # fold B: 3 of 5 and 12 of 15; 5 of 10 and 15 of 30 informative
    *_by_class(
        "B",
        "ours_only",
        REPLAY,
        {
            "n_events_all": 10,
            "informative_share": 0.5,
            "n_events": 5,
            "support_rate": 0.6,
        },
    ),
    *_by_class(
        "B",
        "shared",
        REPLAY,
        {
            "n_events_all": 30,
            "informative_share": 0.5,
            "n_events": 15,
            "support_rate": 0.8,
        },
    ),
    # fold C: no ours-only replay is informative
    *_by_class(
        "C",
        "ours_only",
        REPLAY,
        {
            "n_events_all": 3,
            "informative_share": 0.0,
            "n_events": 0,
            "support_rate": np.nan,
        },
    ),
    # the sliding window's content of the same events: 1 of 4 and 8 of 16 in
    # fold A, 2 of 5 and 3 of 15 in fold B
    *_by_class(
        "A",
        "ours_only",
        REPLAY,
        {"n_events": 4, "support_rate": 0.25},
        "sliding_window",
    ),
    *_by_class(
        "A", "shared", REPLAY, {"n_events": 16, "support_rate": 0.5}, "sliding_window"
    ),
    *_by_class(
        "B", "ours_only", REPLAY, {"n_events": 5, "support_rate": 0.4}, "sliding_window"
    ),
    *_by_class(
        "B", "shared", REPLAY, {"n_events": 15, "support_rate": 0.2}, "sliding_window"
    ),
    # the shared events as the sliding window bounds them are not counted
    *_by_class(
        "A",
        "shared",
        REPLAY,
        {"n_events": 100, "support_rate": 0.1},
        detector="sliding_window",
    ),
    # another truth: place-less events, 0 of 3 with support in fold A and 1 of 2
    # in fold B; at the floor of 2 spikes, 1 of 49 and 2 of 4 (as floats, 1 / 49
    # times 49 is just short of 1)
    *_by_class(
        "A",
        "ours_only",
        "no_content",
        {
            "n_events": 3,
            "support_rate": 0.0,
            "n_events_floor2": 49,
            "support_rate_floor2": 1 / 49,
        },
    ),
    *_by_class(
        "B",
        "ours_only",
        "no_content",
        {
            "n_events": 2,
            "support_rate": 0.5,
            "n_events_floor2": 4,
            "support_rate_floor2": 0.5,
        },
    ),
    (
        "error_rates",
        "false_support_rate",
        "A",
        "state_space",
        "state_space",
        "all",
        "",
        0.0,
    ),
    ("agreement", "estimator_agreement", "A", "both", "both", "all", "", 0.99),
    ("agreement", "estimator_agreement", "B", "both", "both", "all", "", 0.9),
    ("agreement", "estimator_agreement", "C", "both", "both", "all", "", 1.0),
    ("agreement", "n_informative_under_both", "A", "both", "both", "all", "", 0.5),
]
HELDOUT = _heldout(HELDOUT_ROWS)
EITHER = ("ours_only", "shared")


def test_pooled_heldout_rate_weights_each_class_by_its_informative_events():
    result = stats.pooled_heldout_rate(HELDOUT, "A", EITHER, REPLAY)
    # (2 + 12) of (4 + 16), not the mean of 0.5 and 0.75
    assert result["rate"] == pytest.approx(0.7)
    assert result["n_events"] == 20
    assert result["n_counted"] == 14


def test_pooled_heldout_rate_of_one_class_is_that_class_rate():
    result = stats.pooled_heldout_rate(HELDOUT, "B", ("shared",), REPLAY)
    assert result["rate"] == pytest.approx(0.8)
    assert result["n_events"] == 15


def test_pooled_informative_share_weights_each_class_by_all_its_events():
    result = stats.pooled_heldout_rate(
        HELDOUT, "A", EITHER, REPLAY, measure="informative_share"
    )
    # (4 + 16) of (10 + 20)
    assert result["rate"] == pytest.approx(2 / 3)
    assert result["n_events"] == 30
    assert result["n_counted"] == 20


def test_pooled_heldout_rate_reads_the_named_estimator_and_floor():
    sliding = stats.pooled_heldout_rate(
        HELDOUT, "A", EITHER, REPLAY, estimator="sliding_window"
    )
    # (1 + 8) of (4 + 16)
    assert sliding["rate"] == pytest.approx(0.45)
    loose = stats.pooled_heldout_rate(
        HELDOUT, "A", ("ours_only",), "no_content", suffix="_floor2"
    )
    assert loose["rate"] == pytest.approx(1 / 49)
    assert loose["n_events"] == 49
    assert loose["n_counted"] == 1


def test_pooled_heldout_rate_is_nan_when_no_event_counts():
    none_informative = stats.pooled_heldout_rate(HELDOUT, "C", ("ours_only",), REPLAY)
    assert np.isnan(none_informative["rate"])
    assert (none_informative["n_events"], none_informative["n_counted"]) == (0, 0)
    no_rows = stats.pooled_heldout_rate(HELDOUT, "C", ("shared",), REPLAY)
    assert np.isnan(no_rows["rate"])
    assert (no_rows["n_events"], no_rows["n_counted"]) == (0, 0)


@pytest.mark.parametrize(
    ("which", "message"),
    [
        ({"fold": "D"}, "fold 'D'"),
        ({"truth": "replay"}, "truth 'replay'"),
        ({"estimator": "sliding"}, "estimator 'sliding'"),
        ({"classes": ("ours_only", "ours")}, "event_class 'ours'"),
        ({"measure": "support"}, "measure 'support'"),
        ({"suffix": "_floor3"}, "measure 'support_rate_floor3'"),
    ],
)
def test_pooled_heldout_rate_refuses_a_name_the_summary_does_not_hold(which, message):
    # a name no row has would select no event, and read as "no event counts"
    arguments = {"fold": "A", "classes": EITHER, "truth": REPLAY} | which
    with pytest.raises(ValueError, match=message):
        stats.pooled_heldout_rate(HELDOUT, **arguments)


def _jumping(measures: dict[str, float]) -> pd.DataFrame:
    """``HELDOUT`` with fold A's ours-only and shared jumping events: 1 of 2
    ours-only with support, and the shared ones as given."""
    return _heldout(
        HELDOUT_ROWS
        + _by_class("A", "ours_only", "non_local", {"n_events": 2, "support_rate": 0.5})
        + _by_class("A", "shared", "non_local", measures)
    )


def test_pooled_heldout_rate_pools_the_jumping_events():
    result = stats.pooled_heldout_rate(
        _jumping({"n_events": 6, "support_rate": 0.5}), "A", EITHER, "non_local"
    )
    # (1 + 3) of (2 + 6)
    assert result == {"rate": 0.5, "n_events": 8, "n_counted": 4}


def test_pooled_heldout_rate_refuses_two_rows_for_one_class_and_measure():
    # averaged, the two rates would read as one
    twice = pd.concat(
        [
            _jumping({"n_events": 6, "support_rate": 0.5}),
            _heldout(_by_class("A", "shared", "non_local", {"support_rate": 1.0})),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="several rows for one class and measure"):
        stats.pooled_heldout_rate(twice, "A", EITHER, "non_local")


@pytest.mark.parametrize(
    "measures",
    [
        # events and no rate, which is not "none of them has support"
        {"n_events": 6, "support_rate": np.nan},
        {"n_events": 6},
        # a rate and no number of events
        {"n_events": np.nan, "support_rate": 0.5},
        {"support_rate": 0.5},
    ],
)
def test_pooled_heldout_rate_refuses_a_class_without_its_events_or_its_rate(measures):
    with pytest.raises(ValueError, match="has no n_events, or has events and no"):
        stats.pooled_heldout_rate(_jumping(measures), "A", EITHER, "non_local")


def test_pooled_heldout_rate_refuses_a_rate_that_is_no_whole_count_of_events():
    # half of 5 events
    with pytest.raises(ValueError, match="not a whole count"):
        stats.pooled_heldout_rate(
            _jumping({"n_events": 5, "support_rate": 0.5}), "A", EITHER, "non_local"
        )


def test_heldout_fold_statistics_range_over_folds_and_count_events():
    result = stats.heldout_fold_statistics(HELDOUT, ("A", "B"), EITHER, REPLAY)
    # fold A: 14 of 20; fold B: 15 of 20
    assert result["rate_min"] == pytest.approx(0.7)
    assert result["rate_max"] == pytest.approx(0.75)
    assert result["rate_spread"] == pytest.approx(0.05)
    assert result["n_events"] == 40
    assert result["n_counted"] == 29
    assert result["n_folds_with_support"] == 2


def test_heldout_fold_statistics_find_the_lowest_rate_in_a_later_fold():
    result = stats.heldout_fold_statistics(
        HELDOUT, ("A", "B"), EITHER, REPLAY, estimator="sliding_window"
    )
    # fold A: 9 of 20; fold B: 5 of 20
    assert result["rate_min"] == pytest.approx(0.25)
    assert result["rate_max"] == pytest.approx(0.45)


def test_heldout_fold_statistics_count_supported_events_and_folds():
    main = stats.heldout_fold_statistics(
        HELDOUT, ("A", "B"), ("ours_only",), "no_content"
    )
    assert (main["n_events"], main["n_counted"]) == (5, 1)
    # fold A's three informative events have no support
    assert main["n_folds_with_support"] == 1
    loose = stats.heldout_fold_statistics(
        HELDOUT, ("A", "B"), ("ours_only",), "no_content", suffix="_floor2"
    )
    # 1 of 49 and 2 of 4
    assert (loose["n_events"], loose["n_counted"]) == (53, 3)


def test_heldout_fold_statistics_count_informative_events():
    result = stats.heldout_fold_statistics(
        HELDOUT, ("A", "B"), EITHER, REPLAY, measure="informative_share"
    )
    # fold A: 20 of 30; fold B: 20 of 40
    assert (result["n_events"], result["n_counted"]) == (70, 40)
    assert result["rate_min"] == pytest.approx(0.5)
    assert result["rate_max"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("folds", [("A", "B", "C"), ("C", "A", "B")])
def test_heldout_fold_statistics_have_no_range_over_a_fold_with_no_rate(folds):
    result = stats.heldout_fold_statistics(HELDOUT, folds, ("ours_only",), REPLAY)
    # fold C has no informative event: its counts are zero and it has no rate
    assert (result["n_events"], result["n_counted"]) == (9, 5)
    assert result["n_folds_with_support"] == 2
    assert np.isnan(result["rate_min"])
    assert np.isnan(result["rate_max"])
    assert np.isnan(result["rate_spread"])


def _placeless(
    state_space_rate_b: float = 0.0, sliding_rate_b: float = 0.0
) -> pd.DataFrame:
    """Place-less events with no support: 3 ours-only and 2 shared informative in
    fold A, 4 ours-only in fold B; the sliding window's content judges 3 and 4
    ours-only ones."""
    return _heldout(
        [
            *_by_class(
                "A", "ours_only", "no_content", {"n_events": 3, "support_rate": 0.0}
            ),
            *_by_class(
                "A", "shared", "no_content", {"n_events": 2, "support_rate": 0.0}
            ),
            *_by_class(
                "B",
                "ours_only",
                "no_content",
                {"n_events": 4, "support_rate": state_space_rate_b},
            ),
            *_by_class(
                "A",
                "ours_only",
                "no_content",
                {"n_events": 3, "support_rate": 0.0},
                "sliding_window",
            ),
            *_by_class(
                "B",
                "ours_only",
                "no_content",
                {"n_events": 4, "support_rate": sliding_rate_b},
                "sliding_window",
            ),
        ]
    )


def test_placeless_event_statistics_count_both_classes_under_each_estimate():
    result = stats.placeless_event_statistics(_placeless(), ("A", "B"))
    # 3 + 2 in fold A and 4 in fold B; the sliding window's estimate judges 3 + 4
    assert result == {
        "n_events": 9,
        "n_supported": 0,
        "n_events_sliding_window": 7,
        "n_supported_sliding_window": 0,
    }


@pytest.mark.parametrize(
    ("state_space_rate_b", "sliding_rate_b", "supported"),
    [(0.25, 0.0, (1, 0)), (0.0, 0.25, (0, 1))],
)
def test_placeless_event_statistics_count_a_supported_event(
    state_space_rate_b, sliding_rate_b, supported
):
    # 1 of fold B's 4 informative place-less events, under one estimate
    result = stats.placeless_event_statistics(
        _placeless(state_space_rate_b, sliding_rate_b), ("A", "B")
    )
    assert (result["n_supported"], result["n_supported_sliding_window"]) == supported


def test_placeless_event_statistics_ignore_a_fold_with_no_informative_event():
    # fold B's four events are not informative: no rate, which is no support
    table = _placeless()
    in_b = (table["fold"] == "B") & (table["estimator"] == "state_space")
    table.loc[in_b & (table["measure"] == "n_events"), "value"] = 0.0
    table.loc[in_b & (table["measure"] == "support_rate"), "value"] = np.nan
    result = stats.placeless_event_statistics(table, ("A", "B"))
    assert (result["n_events"], result["n_supported"]) == (5, 0)


def test_lowest_estimator_agreement_is_over_folds():
    assert stats.lowest_estimator_agreement(HELDOUT) == 0.9


def test_lowest_estimator_agreement_refuses_a_fold_without_one():
    rows = [
        (*row[:7], np.nan) if row[1] == "estimator_agreement" and row[2] == "B" else row
        for row in HELDOUT_ROWS
    ]
    with pytest.raises(ValueError, match="a fold without one"):
        stats.lowest_estimator_agreement(_heldout(rows))
    with pytest.raises(ValueError, match="no estimator agreement"):
        stats.lowest_estimator_agreement(HELDOUT[HELDOUT["summary"] == "by_class"])


def test_heldout_event_classes_are_those_of_the_by_class_rows():
    assert stats.heldout_event_classes(HELDOUT) == {"ours_only", "shared"}
    # a class in another block is not one of them
    other_block = _heldout(
        [*HELDOUT_ROWS, ("error_rates", "n", "A", "x", "x", "comparator_only", "", 1.0)]
    )
    assert stats.heldout_event_classes(other_block) == {"ours_only", "shared"}


def test_heldout_event_classes_refuse_an_event_only_the_sliding_window_reported():
    with_comparator = _heldout(
        HELDOUT_ROWS + _by_class("A", "comparator_only", REPLAY, {"n_events_all": 1})
    )
    with pytest.raises(ValueError, match="comparator-only events"):
        stats.heldout_event_classes(with_comparator)
