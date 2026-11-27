"""The role content estimands per session, their summaries and the model-free
references."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.regression.contrasts import evaluate_contrast
from continuum_swr_replay.analysis.regression.units import unit_summary
from continuum_swr_replay.analysis.replay_models import declared_tests as dt
from continuum_swr_replay.analysis.replay_models.content import role_content as rc
from continuum_swr_replay.analysis.replay_models.content import (
    role_content_scenarios as sc,
)
from continuum_swr_replay.analysis.replay_models.content import (
    role_content_statistics as rs,
)
from continuum_swr_replay.analysis.shared_statistics import estimand_summary
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS, fit_by_columns

from .role_tables import draw_shares, synthetic_event_arm_table

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
                    "fit_by": "session",
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
    assert out.loc[dt.DIFFERENCE, "n_units_left_out"] == 1
    assert out.loc["No SWR", "n_units_left_out"] == 1
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


@pytest.fixture(scope="module")
def unit_rows():
    """The No SWR rows of four sessions (a day of two, a day of one, a second
    animal), shares drawn around role levels and arm effects."""
    keys = (
        ("a1", "20200101", "02_r1"),
        ("a1", "20200101", "04_r2"),
        ("a1", "20200102", "02_r1"),
        ("a2", "20200103", "04_r2"),
    )
    cohort = rc.role_content_cohort(
        synthetic_event_arm_table(keys, n_visits=150, swr_share=0.3, seed=3)
    )
    truth = sc.Truth(arm=sc.arm_effects(), role={"current": 0.6, "previous": -0.4})
    reference = sc.ScenarioReference(cohort.rows, cohort.reference_log_duration)
    rows = cohort.model_rows("role_content")
    rows = rows[rows[rc.KIND] == "No SWR"]
    sessions = {
        key: draw_shares(part, sc.expected_shares(part, truth, reference), 40, i)
        for i, (key, part) in enumerate(rc.session_rows(rows).items())
    }
    return cohort, sessions


@pytest.mark.parametrize(
    ("fit_by", "n_units"), [("session", 4), ("day", 3), ("animal", 2)]
)
def test_unit_estimands_at_each_unit(unit_rows, fit_by, n_units) -> None:
    """One row per unit and estimand, keyed by the unit; each value is the
    declared contrast mapped onto the unit's coefficients by name (the arm
    copies are not read); the summary is ``unit_summary`` at the unit, the
    kind difference taken unit by unit; values of two units are refused."""
    cohort, sessions = unit_rows
    settings = rc.RoleContentSettings(fit_by=fit_by)
    fit = rc.fit_role_content(
        sessions, model="role_content", variant="unadjusted", kind="No SWR",
        reference_log_duration=cohort.reference_log_duration, settings=settings,
        scale_rows=cohort.model_rows("role_content"),
    )  # fmt: skip
    values = rs.unit_estimands(fit)
    assert len(values) == n_units * len(fit.contrasts)
    assert (values["fit_by"] == fit_by).all() and values["kept"].all()
    record = fit.records[-1]
    got = values[
        (values["animal"] == record.animal)
        & (values["session_date"] == record.session_date)
        & (values["epoch"] == record.epoch)
    ].set_index("estimand_id")
    for estimand_id, contrast in fit.contrasts.items():
        on_fit = fit.design.contrast_on(contrast, record.coef_names)
        expected = evaluate_contrast(
            on_fit, np.asarray(record.coef), np.asarray(record.vcov)
        )
        assert got.loc[estimand_id, "estimate"] == pytest.approx(
            expected.estimate, rel=1e-12
        )
    shift = values.assign(kind="SWR", estimate=values["estimate"] + 0.2)
    summary = rs.estimand_summaries(
        pd.concat([values, shift]), n_draws=100, seed=1
    ).set_index(["estimand_id", "tested"])
    assert (summary["fit_by"] == fit_by).all()
    q3a = values[values["estimand_id"] == "Q3a"]
    columns = list(fit_by_columns(fit_by))
    index = (
        pd.Index(q3a[columns[0]].to_numpy(), name=columns[0])
        if len(columns) == 1
        else pd.MultiIndex.from_frame(q3a[columns])
    )
    expected = unit_summary(
        pd.Series(q3a["estimate"].to_numpy(), index=index), fit_by, n_draws=100, seed=1
    )
    assert summary.loc[("Q3a", "No SWR"), "estimate"] == pytest.approx(
        expected["estimate"]
    )
    assert summary.loc[("Q3a", dt.DIFFERENCE), "estimate"] == pytest.approx(0.2)
    assert summary.loc[("Q3a", "No SWR"), "n_units"] == n_units
    other = "animal" if fit_by != "animal" else "day"
    with pytest.raises(ValueError, match="fit_by"):
        rs.estimand_summaries(
            pd.concat([values, shift.assign(fit_by=other)]), n_draws=100, seed=1
        )
