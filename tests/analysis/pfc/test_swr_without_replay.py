"""The extended event set: SWRs that overlap no non-local event."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.design import _make_design_matrices
from continuum_swr_replay.analysis.pfc.event_sets import (
    EVENT_CLASSES,
    EXTENDED_EVENT_CLASSES,
    SWR_NO_NON_LOCAL,
    swr_without_non_local,
)
from continuum_swr_replay.analysis.pfc.matched_control import (
    make_matched_control_times,
)
from continuum_swr_replay.analysis.pfc.modulation import (
    CURVE_SOURCES,
    EXTENDED_CURVE_SOURCES,
    NESTED_MODEL_PAIRS,
    THREE_CLASS_MODEL,
    curve_coefficients,
)


def _frame(pairs):
    return pd.DataFrame(pairs, columns=["start_time", "end_time"], dtype=float)


def test_swr_without_non_local_partitions_swrs():
    swrs = _frame([(1.0, 1.1), (2.0, 2.1), (3.0, 3.1), (4.0, 4.1)])
    non_local = _frame(
        [(1.05, 1.3), (2.1, 2.4)]
    )  # overlaps the first; abuts the second
    dropped = _frame([(3.05, 3.2)])

    kept, counts = swr_without_non_local(swrs, non_local, dropped)

    np.testing.assert_array_equal(kept["start_time"], [2.0, 4.0])
    assert counts == {
        "n_swr_with_non_local": 1,
        "n_swr_no_non_local": 2,
        "n_swr_no_non_local_unknown": 1,
    }
    assert sum(counts.values()) == len(swrs)


def test_replay_outside_the_cohort_still_counts():
    # The SWR is in the cohort; the non-local event it overlaps starts before
    # the stop (so it is not a cohort event), but it is still replay.
    swrs = _frame([(10.0, 10.1)])
    non_local_all_session = _frame([(9.9, 10.05)])

    kept, counts = swr_without_non_local(swrs, non_local_all_session, _frame([]))

    assert kept.empty and counts["n_swr_with_non_local"] == 1


def test_empty_inputs():
    kept, counts = swr_without_non_local(_frame([]), _frame([]), _frame([]))

    assert kept.empty and counts["n_swr_no_non_local"] == 0


@pytest.fixture
def session():
    time = np.arange(0, 60, 1 / 500.0)
    position_info = pd.DataFrame(index=pd.Index(time, name="time"))
    onsets = np.arange(1.0, 58.0, 1.7)
    events = _frame([(t, t + 0.08) for t in onsets])
    times = {
        "non_local": events,
        "non_local_SWR": events.iloc[::2].reset_index(drop=True),
        "non_local_no_SWR": events.iloc[1::2].reset_index(drop=True),
        "SWR": _frame([(t + 0.6, t + 0.68) for t in onsets[::3]]),
        "matched_control": _frame([(t + 0.9, t + 0.98) for t in onsets[::2]]),
    }
    return position_info, times


def test_canonical_event_set_builds_exactly_the_canonical_ladder(session):
    position_info, times = session
    lags = np.arange(-125, 126)

    designs, _, _ = _make_design_matrices(times, position_info, lags=lags)

    assert list(designs) == [
        "constant",
        "non_local_no_SWR",
        "non_local_SWR",
        "SWR",
        "non_local_SWR + non_local_no_SWR",
        "SWR + non_local_no_SWR",
        "non_local",
        "matched_control",
        "matched_control + non_local_no_SWR",
    ]
    assert all(SWR_NO_NON_LOCAL not in model for model, _ in NESTED_MODEL_PAIRS)
    assert all(model in designs for model, _ in CURVE_SOURCES.values())


def test_extended_design_block_order(session):
    position_info, times = session
    times = {**times, SWR_NO_NON_LOCAL: times["SWR"].iloc[::2].reset_index(drop=True)}
    lags = np.arange(-125, 126)

    designs, basis, _ = _make_design_matrices(times, position_info, lags=lags)
    canonical, _, _ = _make_design_matrices(
        {k: v for k, v in times.items() if k != SWR_NO_NON_LOCAL},
        position_info,
        lags=lags,
    )

    n_basis = basis.shape[1]
    three = designs[THREE_CLASS_MODEL]
    assert three.shape[1] == 1 + 3 * n_basis
    np.testing.assert_array_equal(
        three[:, 1 : 1 + 2 * n_basis],
        canonical["non_local_SWR + non_local_no_SWR"][:, 1:],
    )
    np.testing.assert_array_equal(
        three[:, 1 + 2 * n_basis :], designs[SWR_NO_NON_LOCAL][:, 1:]
    )
    for name in canonical:
        np.testing.assert_array_equal(designs[name], canonical[name])
    model, block = curve_coefficients(SWR_NO_NON_LOCAL, n_basis, EXTENDED_CURVE_SOURCES)
    assert model == THREE_CLASS_MODEL and block == slice(
        1 + 2 * n_basis, 1 + 3 * n_basis
    )
    _, block = curve_coefficients(
        "non_local_SWR_3class", n_basis, EXTENDED_CURVE_SOURCES
    )
    assert block == slice(1 + n_basis, 1 + 2 * n_basis)


def test_matched_control_unchanged_by_the_extra_class(session):
    position_info, times = session
    fit_mask = pd.Series(True, index=position_info.index)
    canonical = {k: times[k] for k in EVENT_CLASSES}
    extended = {
        **canonical,
        SWR_NO_NON_LOCAL: times["SWR"].iloc[::2].reset_index(drop=True),
    }

    first = make_matched_control_times(canonical, fit_mask, np.random.default_rng(7))
    second = make_matched_control_times(extended, fit_mask, np.random.default_rng(7))

    pd.testing.assert_frame_equal(first.times, second.times)
    assert set(EXTENDED_EVENT_CLASSES) - set(EVENT_CLASSES) == {SWR_NO_NON_LOCAL}
