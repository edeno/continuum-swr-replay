"""The support rule and its share over sessions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.regression.constants import MAX_FAILING_SHARE
from continuum_swr_replay.analysis.regression.likelihoods import GroupedSoftmax
from continuum_swr_replay.analysis.regression.sessions import fit_sessions
from continuum_swr_replay.analysis.regression.support import (
    MIN_UNITS,
    Removed,
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

from .conftest import ROLE, by_session

FEW = Removed.SUPPORT


def test_binary_term_counts_rows_not_events() -> None:
    """Hand-built visits: the units with a row where the 0/1 term is 1 and
    those with a row where it is 0, and the pass flag (both at least 5). The
    rule reads the design only: where the events are does not enter."""
    # Visits 0-6 rewarded (term 1), 7-10 not.
    visit = np.repeat(np.arange(11), 3)
    term = (visit <= 6).astype(float)
    support = binary_support(term, visit)
    assert support == Support(n_low=4, n_high=7, reason=FEW)
    visit = np.repeat(np.arange(13), 3)
    term = (visit <= 6).astype(float)
    assert binary_support(term, visit) == Support(6, 7)
    assert Support(6, 7).passes and not support.passes
    assert MIN_UNITS == 5


def test_term_changing_within_a_unit_counts_each_side() -> None:
    """A unit whose term changes within it (an outcome known mid-visit) counts
    on each side it has a row on."""
    visit = np.array([0, 0, 1, 1, 2])
    term = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    assert binary_support(term, visit) == Support(2, 3, FEW)


def test_role_slope_counts_and_constant() -> None:
    """A slope: units with a row above and below the median of its rows (ties
    to neither); a constant column fails, and so do no rows."""
    event = np.arange(12)
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    # median 0.6: five below, five above, two ties
    assert slope_support(x, event) == Support(5, 5)
    assert slope_support(np.full(12, 0.4), event) == Support(0, 0, FEW)
    assert slope_support(x[:0], event[:0]) == Support(0, 0, FEW)


def test_design_support_reads_each_columns_rows(content_sessions) -> None:
    """In a content design, a role level counts the events with a candidate in
    the role (and those with a candidate in another); a slope by a role is
    split at the median of that role's rows only; a smooth has no entry."""
    pooled = pd.concat(content_sessions.values(), ignore_index=True)
    design = build_design(
        [
            ROLE,
            Slope("reward_fraction", "none", by=ROLE, levels=("current", "previous")),
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
    support = design_support(design, built.X, built.level_rows, unit)
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
    )
    assert not support["reward_fraction:role=previous"].passes


def test_softmax_counts_are_those_of_the_event_rule(content_sessions) -> None:
    """Under the grouped softmax every candidate row belongs to an event, so
    the counts on rows are those of the event-based rule they replace (lead,
    2026-11-23): the support records of two content sessions equal the counts
    that rule gave at commit ec1ae69c, for every simple term."""
    frames = list(content_sessions.values())
    first = frames[0]
    first = first[~((first["role"] == "previous") & (first["event_id"] >= 7))]
    frames[0] = first.assign(
        flag=((first["role"] == "current") & (first["event_id"] < 9)) * 1.0
    ).reset_index(drop=True)
    for k, rows in enumerate(frames[1:], start=1):
        on = (rows["role"] == "current") & (rows["event_id"] % 2 == 0)
        frames[k] = rows.assign(flag=on * 1.0)
    sessions = by_session(frames)
    design = build_design(
        [
            ROLE,
            Slope(
                "reward_fraction",
                "none",
                by=ROLE,
                levels=("current", "previous", "other"),
            ),
            Slope("clock_s", "none"),
            Slope("flag", "none"),
        ],
        pd.concat(sessions.values(), ignore_index=True),
        intercept=False,
    )
    records = fit_sessions(
        design,
        sessions,
        GroupedSoftmax(),
        model="m",
        kind="k",
        response="share",
        unit="event_id",
        table_sha256={},
    )
    before = [
        {
            "role[current]": (150, 150, ""),
            "role[previous]": (150, 7, ""),
            "role[trunk]": (150, 150, ""),
            "reward_fraction:role=current": (75, 75, ""),
            "reward_fraction:role=previous": (3, 3, FEW),
            "reward_fraction:role=other": (75, 75, ""),
            "clock_s": (131, 126, ""),
            "flag": (150, 9, ""),
        },
        {
            "role[current]": (150, 150, ""),
            "role[previous]": (150, 150, ""),
            "role[trunk]": (150, 150, ""),
            "reward_fraction:role=current": (75, 75, ""),
            "reward_fraction:role=previous": (75, 75, ""),
            "reward_fraction:role=other": (75, 75, ""),
            "clock_s": (135, 140, ""),
            "flag": (150, 75, ""),
        },
    ]
    for record, expected in zip(records[:2], before, strict=True):
        assert {
            name: (entry["n_low"], entry["n_high"], entry["reason"])
            for name, entry in record.support.items()
        } == expected


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
