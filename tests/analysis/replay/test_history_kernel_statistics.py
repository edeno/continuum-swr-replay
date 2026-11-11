"""The history-kernel statistics of the duration-aware fits, on kernels small
enough to check by hand."""

from __future__ import annotations

import numpy as np
import pytest

from continuum_swr_replay.analysis.replay import history_kernel_statistics as stats

LOG = np.log
RUNG = "Mself_dur"
CROSS_RUNG = "Mselfcross_long_dur"


def test_a_lag_index_is_one_bin_less_than_its_lag():
    assert stats.lag_ms(0) == 15
    assert stats.lag_ms(6) == 105
    assert stats.lag_index(15) == 0
    assert stats.lag_index(45) == 2


def _record(animal, outcome, end, onset=None, *, ok=True, converged=True):
    """A pooled result whose rung carries an end-locked kernel and, when given,
    an onset-locked one. A kernel is (rate ratios, SEs, supports), one per lag."""
    kernels, support = {}, {}
    for block, kernel in (("self_offset", end), ("self_onset", onset)):
        if kernel is not None:
            rate_ratios, se, expected = kernel
            kernels[block] = {"log_rr": list(LOG(rate_ratios)), "se": list(se)}
            support[block] = list(expected)
    return {
        "animal": animal,
        "outcome": outcome,
        "support": support,
        "duration_aware": {
            RUNG: {"ok": ok, "converged": converged, "kernels": kernels}
        },
    }


# --- one kernel -----------------------------------------------------------------------


def test_a_lag_is_measurable_with_enough_support_and_a_small_enough_se():
    record = _record(
        "a",
        "swr",
        # supports 2 (too few) and 3 (enough); SEs 0.5 (allowed) and 0.6 (too wide)
        ([2.0, 2.0, 2.0, 2.0], [0.1, 0.1, 0.5, 0.6], [2.0, 3.0, 10.0, 10.0]),
    )
    kernel = stats.measurable_kernel(record, RUNG, "self_offset")
    assert list(kernel["ok"]) == [False, True, True, False]
    np.testing.assert_allclose(kernel["log_rr"], LOG(2.0))
    assert list(kernel["se"]) == [0.1, 0.1, 0.5, 0.6]


def test_a_kernel_of_a_failed_or_missing_fit_is_not_read():
    end = ([2.0], [0.1], [10.0])
    assert stats.measurable_kernel(_record("a", "swr", end), RUNG, "self_onset") is None
    assert (
        stats.measurable_kernel(_record("a", "swr", end), "other", "self_offset")
        is None
    )
    for failure in ({"ok": False}, {"converged": False}):
        record = _record("a", "swr", end, **failure)
        assert stats.measurable_kernel(record, RUNG, "self_offset") is None


def _kernel(log_rr, se, ok):
    return {
        "log_rr": np.asarray(log_rr, dtype=float),
        "se": np.asarray(se, dtype=float),
        "ok": np.asarray(ok, dtype=bool),
    }


def test_the_history_peak_interval_is_widened_for_the_lags_searched():
    # z = 2.2 at the largest of 8 measurable lags: above 1.96, below the
    # Bonferroni threshold for 8 lags (about 2.73).
    log_rr = np.r_[0.22, np.zeros(9)]
    se = np.full(10, 0.1)
    many = _kernel(log_rr, se, [True] * 8 + [False] * 2)
    one = _kernel(log_rr, se, [True] + [False] * 9)

    assert stats.kernel_peak(many)["above_one"] is False
    assert stats.kernel_peak(one)["above_one"] is True
    # A large peak still clears the widened interval.
    assert stats.kernel_peak(
        _kernel(np.r_[0.5, np.zeros(9)], se, [True] * 8 + [False] * 2)
    )["above_one"]


def test_the_peak_is_the_largest_measurable_lag():
    # the largest rate ratio (5, at 30 ms) is not measurable: the peak is 3 at 45 ms
    kernel = _kernel(LOG([2.0, 5.0, 3.0, 1.0]), [0.1] * 4, [True, False, True, True])
    assert stats.kernel_peak(kernel) == {
        "rate_ratio": pytest.approx(3.0),
        "lag_ms": 45,
        "above_one": True,
    }
    assert stats.kernel_peak(_kernel([1.0, 2.0], [0.1, 0.1], [False, False])) is None


# --- the self-history over animals ----------------------------------------------------

SUPPORTED = [10.0, 10.0, 10.0]
PRECISE = [0.1, 0.1, 0.1]


def _pooled_records() -> list[dict]:
    """Two populations.

    swr, end-locked (15, 30, 45 ms): a 0.5, 2, 4 (suppressed first lag, peak 4
    at 45 ms); b 3, 2, 1 (raised first lag, peak 3 at 15 ms); c 1.1, 6, 2 with
    an imprecise first lag (peak 6 at 30 ms). Onset-locked: a 9, 1.5, 1.2 with
    no support at the first lag (peak 1.5 at 30 ms, clear of 1); b 1, 1.1, 1
    (peak 1.1 at 30 ms, not clear of 1); c has none.

    non_local, end-locked: a 2, 1, 1 (raised, peak 2 at 15 ms); b 0.9, 1.05, 1
    and c 1.1, 1, 1, whose first lags are below and above 1 with intervals that
    hold 1 (neither suppressed nor raised; peaks 1.05 at 30 ms and 1.1 at
    15 ms, not clear of 1). Onset-locked: a 1.5, 1, 1; b and c have none.
    """
    return [
        _record(
            "a",
            "swr",
            ([0.5, 2.0, 4.0], PRECISE, SUPPORTED),
            ([9.0, 1.5, 1.2], PRECISE, [0.0, 10.0, 10.0]),
        ),
        _record(
            "b",
            "swr",
            ([3.0, 2.0, 1.0], PRECISE, SUPPORTED),
            ([1.0, 1.1, 1.0], PRECISE, SUPPORTED),
        ),
        _record("c", "swr", ([1.1, 6.0, 2.0], [0.6, 0.1, 0.1], SUPPORTED)),
        _record(
            "a",
            "non_local",
            ([2.0, 1.0, 1.0], PRECISE, SUPPORTED),
            ([1.5, 1.0, 1.0], PRECISE, SUPPORTED),
        ),
        _record("b", "non_local", ([0.9, 1.05, 1.0], PRECISE, SUPPORTED)),
        _record("c", "non_local", ([1.1, 1.0, 1.0], PRECISE, SUPPORTED)),
    ]


def test_self_history_statistics_have_one_row_per_population():
    table = stats.self_history_statistics(_pooled_records(), RUNG)
    assert table.index.name == "outcome"
    assert set(table.index) == {"swr", "non_local"}
    assert table.loc["swr"].to_dict() == {
        "n_fits": 3,
        "n_first_lag_measurable": 2,
        "first_lag_rate_ratio_median": pytest.approx(1.75),  # of 0.5 and 3
        "n_first_lag_suppressed": 1,
        "n_first_lag_raised": 1,
        "end_peak_rate_ratio_median": pytest.approx(4.0),  # of 4, 3 and 6
        "end_peak_lag_ms_median": 30.0,  # of 45, 15 and 30
        "n_end_peaks_above_one": 3,
        "onset_peak_rate_ratio_median": pytest.approx(1.3),  # of 1.5 and 1.1
        "onset_peak_lag_ms_median": 30.0,
        "n_onset_peaks_above_one": 1,
    }
    assert table.loc["non_local"].to_dict() == {
        "n_fits": 3,
        "n_first_lag_measurable": 3,
        "first_lag_rate_ratio_median": pytest.approx(1.1),  # of 2, 0.9 and 1.1
        "n_first_lag_suppressed": 0,
        "n_first_lag_raised": 1,
        "end_peak_rate_ratio_median": pytest.approx(1.1),  # of 2, 1.05 and 1.1
        "end_peak_lag_ms_median": 15.0,  # of 15, 30 and 15
        "n_end_peaks_above_one": 1,
        "onset_peak_rate_ratio_median": pytest.approx(1.5),
        "onset_peak_lag_ms_median": 15.0,
        "n_onset_peaks_above_one": 1,
    }


def test_the_self_history_ranges_are_over_the_populations():
    table = stats.self_history_statistics(_pooled_records(), RUNG)
    assert stats.self_history_range_statistics(table) == {
        "n_fits": 6,
        "n_first_lag_measurable": 5,
        "n_first_lag_suppressed": 1,
        "n_first_lag_raised": 2,
        "first_lag_rate_ratio_low": pytest.approx(1.1),
        "first_lag_rate_ratio_high": pytest.approx(1.75),
        "end_peak_rate_ratio_low": pytest.approx(1.1),
        "end_peak_rate_ratio_high": pytest.approx(4.0),
        "end_peak_lag_ms_low": 15.0,
        "end_peak_lag_ms_high": 30.0,
        "n_end_peaks_above_one_min": 1,
    }


def test_self_history_refuses_an_unconverged_end_locked_fit():
    records = _pooled_records()
    records[1] = _record(
        "b", "swr", ([3.0, 2.0, 1.0], PRECISE, SUPPORTED), converged=False
    )
    with pytest.raises(ValueError, match="swr/b: Mself_dur failed or did not converge"):
        stats.self_history_statistics(records, RUNG)


def test_self_history_refuses_a_population_with_no_measurable_first_lag():
    records = [_record("c", "swr", ([1.1, 6.0, 2.0], [0.6, 0.1, 0.1], SUPPORTED))]
    with pytest.raises(ValueError, match="swr: no animal's 15 ms lag is measurable"):
        stats.self_history_statistics(records, RUNG)


# --- the cross-history over animals ---------------------------------------------------


def _ridge_record(
    animal, outcome, rate_ratio, sd, support, gain, *, full_converged=True, n_bad=0
):
    """A ridge result whose end-locked cross kernel has ``rate_ratio`` at 45 ms
    (the third lag), with that lag's posterior SD and expected onsets."""
    kernel = {"log_rr": [9.0, 9.0, LOG(rate_ratio), 9.0], "sd": [0.1, 0.1, sd, 0.1]}
    fit = {
        "full_converged": full_converged,
        "cv": {"n_nonconverged": n_bad},
        "kernels": {"cross_offset": kernel},
    }
    return {
        "animal": animal,
        "outcome": outcome,
        "gain_per_event": gain,
        "support": {"cross_offset": [20.0, 20.0, support, 20.0]},
        "models": {
            "Mself_long_dur": {"full_converged": True, "cv": {"n_nonconverged": 0}},
            CROSS_RUNG: fit,
        },
    }


def _ridge_records() -> list[dict]:
    return [
        _ridge_record("a", "non_local_swr", 2.0, 0.1, 20.0, 0.01),
        # a wide posterior: 3 is not clear of 1
        _ridge_record("b", "non_local_swr", 3.0, 1.0, 20.0, 0.02),
        _ridge_record("c", "non_local_swr", 0.5, 0.1, 20.0, -0.03),
        # exactly the least support that is read
        _ridge_record("d", "non_local_swr", 4.0, 0.1, 3.0, 0.04),
        # too little support: its rate ratio is not read, its gain is
        _ridge_record("e", "non_local_swr", 50.0, 0.1, 1.0, 0.0),
        _ridge_record("a", "non_local_no_swr", 7.0, 0.1, 20.0, 0.5, n_bad=1),
    ]


def test_cross_history_statistics_are_of_one_response_population():
    result = stats.cross_history_statistics(
        _ridge_records(), "non_local_swr", CROSS_RUNG, 45
    )
    assert result == {
        "n_animals": 5,
        "n_measurable": 4,
        "rate_ratio_median": pytest.approx(2.5),  # of 2, 3, 0.5 and 4
        "n_animals_above_one": 2,  # a and d
        "heldout_gain_median": 0.01,  # of 0.01, 0.02, -0.03, 0.04 and 0
        "n_animals_gain_positive": 3,
    }


def test_cross_history_refuses_an_unconverged_or_absent_fit():
    records = _ridge_records()
    with pytest.raises(
        ValueError, match="non_local_no_swr/a: ridge .* did not converge"
    ):
        stats.cross_history_statistics(records, "non_local_no_swr", CROSS_RUNG, 45)
    records[0] = _ridge_record(
        "a", "non_local_swr", 2.0, 0.1, 20.0, 0.01, full_converged=False
    )
    with pytest.raises(ValueError, match="non_local_swr/a: ridge .* did not converge"):
        stats.cross_history_statistics(records, "non_local_swr", CROSS_RUNG, 45)
    with pytest.raises(ValueError, match="swr: no ridge cross-history fit"):
        stats.cross_history_statistics(records, "swr", CROSS_RUNG, 45)


def test_cross_history_refuses_a_population_with_no_measurable_lag():
    records = [_ridge_record("e", "non_local_swr", 50.0, 0.1, 1.0, 0.0)]
    with pytest.raises(ValueError, match="no animal's 45 ms lag is measurable"):
        stats.cross_history_statistics(records, "non_local_swr", CROSS_RUNG, 45)
