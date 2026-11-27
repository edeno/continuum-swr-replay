"""The role content estimands per session, their summaries and the model-free
references."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay_models import declared_tests as dt
from continuum_swr_replay.analysis.replay_models.content import role_content as rc
from continuum_swr_replay.analysis.replay_models.content import (
    role_content_statistics as rs,
)
from continuum_swr_replay.analysis.shared_statistics import estimand_summary
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

SESSIONS = [
    ("a1", "20200101", "02_r1"),
    ("a1", "20200101", "04_r2"),
    ("a1", "20200102", "02_r1"),
    ("a2", "20200103", "02_r1"),
    ("a2", "20200104", "02_r1"),
    ("a3", "20200105", "02_r1"),
]


def _per_session(values: dict[str, list[float]], kept=None) -> pd.DataFrame:
    rows = []
    for kind, estimates in values.items():
        for i, (key, estimate) in enumerate(zip(SESSIONS, estimates, strict=True)):
            rows.append(
                dict(zip(SESSION_COLUMNS, key, strict=True))
                | {
                    "model": "role_content",
                    "variant": "unadjusted",
                    "kind": kind,
                    "estimand_id": "Q3a",
                    "estimate": estimate,
                    "variance": 0.01,
                    "kept": True if kept is None else kept[kind][i],
                }
            )
    return pd.DataFrame(rows)


def test_summaries_for_each_kind_and_the_difference() -> None:
    swr = [0.5, 0.7, 0.2, 0.9, 1.1, 0.4]
    no_swr = [0.1, 0.3, 0.0, 0.2, 0.6, 0.5]
    kept = {"SWR": [True] * 6, "No SWR": [True, True, True, True, True, False]}
    per = _per_session({"SWR": swr, "No SWR": no_swr}, kept)
    out = rs.estimand_summaries(per, n_draws=200, seed=1).set_index("tested")
    assert list(out.index) == list(dt.TESTED)
    labels = [pd.Series([k[i] for k in SESSIONS]) for i in range(3)]
    expected = estimand_summary(pd.Series(swr), *labels, n_draws=200, seed=1)
    assert out.loc["SWR", "estimate"] == pytest.approx(expected["estimate"])
    assert out.loc["SWR", "low"] == pytest.approx(expected["low"])
    # The difference is over the sessions with both kinds kept.
    diff = pd.Series(np.subtract(swr, no_swr)[:5])
    expected = estimand_summary(
        diff, *[s.iloc[:5].reset_index(drop=True) for s in labels], n_draws=200, seed=1
    )
    assert out.loc[dt.DIFFERENCE, "estimate"] == pytest.approx(expected["estimate"])
    assert out.loc[dt.DIFFERENCE, "n_sessions"] == 5
    assert out.loc[dt.DIFFERENCE, "n_sessions_left_out"] == 1
    assert out.loc["No SWR", "n_sessions_left_out"] == 1
    assert out.loc["No SWR", "n_sessions"] == 5


def _event_rows(event, session, role_shares, kind="No SWR", **columns):
    roles = ["current", "previous", *["other_leaf"] * 4, *["trunk"] * 3]
    shares = np.zeros(9)
    for role, share in role_shares.items():
        on = [i for i, r in enumerate(roles) if r == role]
        shares[on] = share / len(on)
    frame = pd.DataFrame(
        {
            "animal": session[0],
            "session_date": session[1],
            "epoch": session[2],
            rc.KIND: kind,
            rc.UNIT: event,
            rc.ARM: np.arange(9),
            rc.ROLE: roles,
            rc.RESPONSE: shares,
        }
    )
    for name, value in columns.items():
        frame[name] = value
    return frame


def test_role_shares_by_hand() -> None:
    rows = pd.concat(
        [
            _event_rows(1, SESSIONS[0], {"current": 0.6, "other_leaf": 0.4}),
            _event_rows(2, SESSIONS[0], {"current": 0.2, "trunk": 0.8}),
            _event_rows(3, SESSIONS[3], {"previous": 1.0}),
        ]
    )
    out = rs.role_shares(rows).set_index(["animal", rc.ROLE])
    assert out.loc[("a1", "current"), "total"] == pytest.approx(0.4)
    assert out.loc[("a1", "other_leaf"), "per_arm"] == pytest.approx(0.05)
    assert out.loc[("a1", "trunk"), "total"] == pytest.approx(0.4)
    assert out.loc[("a2", "previous"), "total"] == pytest.approx(1.0)


def test_within_arm_slope_by_hand() -> None:
    """Share = 0.1 * x on one other-leaf arm: its slope per pooled SD is
    0.1 * SD of x over the leaf rows."""
    rng = np.random.default_rng(0)
    frames = []
    for event in range(40):
        frame = _event_rows(event, SESSIONS[0], {"current": 0.5, "other_leaf": 0.5})
        x = rng.uniform(0, 1)
        frame[rc.VALUE] = np.where(frame[rc.ARM] == 2, x, 0.5)
        frame.loc[frame[rc.ARM] == 2, rc.RESPONSE] = 0.1 * x
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    out = rs.within_arm_slopes(rows, (rc.VALUE,), min_rows=10)
    other = out[out[rc.ROLE] == "other_leaf"]["slope"].iloc[0]
    leaf = rows[rows[rc.ROLE].isin(rc.LEAF_ROLES)]
    # Only arm 2 varies in x; the session's value is that arm's slope.
    assert other == pytest.approx(0.1 * leaf[rc.VALUE].std(ddof=0))


def test_current_share_by_time_until_leaving_by_hand() -> None:
    rows = pd.concat(
        [
            _event_rows(
                1, SESSIONS[0], {"current": 0.9, "other_leaf": 0.1}, **{rc.LEAVING: 0.1}
            ),
            _event_rows(
                2, SESSIONS[0], {"current": 0.5, "other_leaf": 0.5}, **{rc.LEAVING: 0.2}
            ),
            _event_rows(
                3, SESSIONS[0], {"current": 0.2, "other_leaf": 0.8}, **{rc.LEAVING: 1.5}
            ),
        ]
    )
    out = rs.current_share_by_time_until_leaving(rows).set_index("bin")["share"]
    assert out["[0.0, 0.25)"] == pytest.approx(0.7)
    assert out["[1.0, 2.0)"] == pytest.approx(0.2)
