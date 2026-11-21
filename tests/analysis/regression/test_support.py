"""The support rule and its share over sessions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.regression.constants import MAX_FAILING_SHARE
from continuum_swr_replay.analysis.regression.support import (
    MIN_UNITS,
    Support,
    binary_support,
    design_support,
    slope_support,
    terms_failing_share,
)
from continuum_swr_replay.analysis.regression.terms import (
    Slope,
    Smooth,
    build_design,
)

from .conftest import ROLE

FEW = f"fewer than MIN_UNITS = {MIN_UNITS} units with an event"


def test_binary_term_counts() -> None:
    """Hand-built visits: the units with an event on a row where the 0/1 term
    is 1 and those where it is 0, and the pass flag (both at least 5)."""
    # Visits 0-6 rewarded (term 1), 7-12 not; events in visits 0-5 and 7-9.
    visit = np.repeat(np.arange(13), 3)
    term = (visit <= 6).astype(float)
    count = np.zeros(visit.size)
    count[
        np.isin(visit, [0, 1, 2, 3, 4, 5, 7, 8, 9]) & (np.arange(visit.size) % 3 == 1)
    ] = 1
    count[0] = 2  # a second event in visit 0 counts it once
    support = binary_support(term, visit, count > 0)
    assert support == Support(n_low=3, n_high=6, reason=FEW)
    count[np.isin(visit, [10, 11]) & (np.arange(visit.size) % 3 == 0)] = 1
    assert binary_support(term, visit, count > 0) == Support(5, 6)
    assert Support(5, 6).passes and not support.passes
    assert MIN_UNITS == 5


def test_term_changing_within_a_unit_counts_each_side() -> None:
    """A unit whose term changes within it (an outcome known mid-visit) counts
    on the side of each row that holds an event."""
    visit = np.array([0, 0, 1, 1])
    term = np.array([0.0, 1.0, 0.0, 1.0])
    events = np.array([True, True, False, True])
    assert binary_support(term, visit, events) == Support(1, 2, FEW)


def test_role_slope_counts_and_constant() -> None:
    """A role slope: events above and below the median of the role's rows
    (ties to neither); a slope constant within the role fails."""
    event = np.arange(12)
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    every = np.ones(12, dtype=bool)
    # median 0.6: five below, five above, two ties
    assert slope_support(x, event, every) == Support(5, 5)
    assert slope_support(np.full(12, 0.4), event, every) == Support(0, 0, FEW)
    assert slope_support(x, event, np.zeros(12, dtype=bool)) == Support(0, 0, FEW)


def test_design_support_reads_each_columns_rows(content_sessions) -> None:
    """In a content design, a role level counts the events with a candidate in
    the role (and those with a candidate in another); a slope by a role is
    split at the median of that role's rows only; smooths have no entry."""
    pooled = pd.concat(content_sessions.values(), ignore_index=True)
    design = build_design(
        [
            ROLE,
            Slope("reward_fraction", by=ROLE, levels=("current", "previous")),
            Smooth("clock_s", 0.0, 20.0, 8, 2.0),
        ],
        pooled,
        intercept=False,
    )
    rows = next(iter(content_sessions.values())).copy()
    rows = rows[~((rows["role"] == "previous") & (rows["event_id"] >= 4))]
    rows = rows.reset_index(drop=True)
    built = design.build(rows)
    unit = rows["event_id"].to_numpy()
    every = np.ones(len(rows), dtype=bool)
    support = design_support(design, built.X, built.level_rows, unit, every)
    assert list(support) == [
        "role[current]",
        "role[previous]",
        "role[trunk]",
        "reward_fraction:role=current",
        "reward_fraction:role=previous",
    ]
    n_events = rows["event_id"].nunique()
    assert support["role[current]"] == Support(n_events, n_events)
    assert support["role[previous]"] == Support(n_events, 4, FEW)
    previous = rows["role"] == "previous"
    assert support["reward_fraction:role=previous"] == slope_support(
        rows.loc[previous, "reward_fraction"].to_numpy(),
        unit[previous.to_numpy()],
        np.ones(int(previous.sum()), dtype=bool),
    )
    assert not support["reward_fraction:role=previous"].passes


def test_failing_share_rule() -> None:
    """A column failing in 21 % of the sessions with a fit is flagged, one
    failing in 20 % is not; a session without a fit does not count."""

    def record(status: str, a: bool, b: bool) -> SimpleNamespace:
        return SimpleNamespace(
            status=status,
            support={"a": {"passes": a}, "b": {"passes": b}},
        )

    records = [record("converged", i >= 21, i >= 20) for i in range(100)]
    records.append(record("no finite maximum", False, False))
    assert MAX_FAILING_SHARE == 0.20
    assert terms_failing_share(records) == {"a": 0.21}
