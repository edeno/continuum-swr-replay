"""Tests for analysis/replay/learning.py.

Stage 6 asks whether replay content modulates the stay/leave decision
after reward vs. omission. Two pieces:

- :func:`stop_replay_summary` collapses per-event arm-fractions to one
  row per stop, picking out the fraction that landed on each
  task-relevant role (current, previous, last-rewarded, best by recent
  reward rate).
- :func:`fit_is_stay_logistic` fits a per-session logistic:
  ``is_stay ~ is_after_omission * frac_on_<role>`` and reports the
  interaction coefficient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.learning import (
    fit_is_stay_logistic_per_session,
    fit_is_stay_pooled_gee,
    stop_replay_summary,
)


def _make_tables(
    events: list[dict],
    stops: list[dict],
    *,
    broadcast_role_cols_to_event_arm: bool = True,
) -> tuple[pd.DataFrame, ...]:
    """Build minimal event_table / event_arm_table / stop_table.

    Each event in ``events`` must include: stop_id, event_id,
    current_arm, prev_arm, last_rewarded_arm, and ``arm_fracs`` (a
    list of 9 floats summing to 1).
    Each stop in ``stops`` must include: stop_id, current_arm,
    last_rewarded_arm, is_after_reward, is_after_omission, is_stay.

    When ``broadcast_role_cols_to_event_arm`` is True (the production
    schema), the role columns ``current_arm``/``prev_arm``/
    ``last_rewarded_arm`` are duplicated onto every row of the
    event_arm_table — this exercises the suffix-collision-avoidance
    path. When False, the event_arm_table omits them and the merge
    falls back to pulling them from event_table.
    """
    sess = {"animal": "test", "day": 1, "epoch": "01_r1"}

    ev_rows = []
    ea_rows = []
    for e in events:
        ev_rows.append(
            {
                **sess,
                "stop_id": e["stop_id"],
                "event_id": e["event_id"],
                "current_arm": e["current_arm"],
                "prev_arm": e["prev_arm"],
                "last_rewarded_arm": e["last_rewarded_arm"],
                "is_swr": e.get("is_swr", False),
                "is_mua": e.get("is_mua", False),
            }
        )
        for arm in range(9):
            ea_row = {
                **sess,
                "stop_id": e["stop_id"],
                "event_id": e["event_id"],
                "arm": arm,
                "arm_fraction": e["arm_fracs"][arm],
                "reward_rate_arm_k10": e.get("rrr", [np.nan] * 9)[arm],
                "is_swr": e.get("is_swr", False),
                "is_mua": e.get("is_mua", False),
            }
            if broadcast_role_cols_to_event_arm:
                ea_row["current_arm"] = e["current_arm"]
                ea_row["prev_arm"] = e["prev_arm"]
                ea_row["last_rewarded_arm"] = e["last_rewarded_arm"]
            ea_rows.append(ea_row)
    st_rows = []
    for s in stops:
        st_rows.append({**sess, **s})

    return (
        pd.DataFrame(ev_rows),
        pd.DataFrame(ea_rows),
        pd.DataFrame(st_rows),
    )


# ─────────────────────────────────────────────────────────────────────
# stop_replay_summary
# ─────────────────────────────────────────────────────────────────────


class TestStopReplaySummary:
    def test_basic_aggregation(self):
        ev, ea, st = _make_tables(
            events=[
                {
                    "stop_id": 1,
                    "event_id": 100,
                    "current_arm": 0,
                    "prev_arm": 1,
                    "last_rewarded_arm": 1,
                    "arm_fracs": [0.50, 0.20, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02],
                    "rrr": [0.8, 0.2, 0.0, 0.4, 0.0, 0.6, np.nan, np.nan, np.nan],
                },
                {
                    "stop_id": 1,
                    "event_id": 101,
                    "current_arm": 0,
                    "prev_arm": 1,
                    "last_rewarded_arm": 1,
                    "arm_fracs": [0.30, 0.40, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02],
                    "rrr": [0.8, 0.2, 0.0, 0.4, 0.0, 0.6, np.nan, np.nan, np.nan],
                },
            ],
            stops=[
                {
                    "stop_id": 1,
                    "current_arm": 0,
                    "prev_arm": 1,
                    "last_rewarded_arm": 1,
                    "is_after_reward": 1,
                    "is_after_omission": 0,
                    "is_after_stay": 0,
                    "is_after_switch": 0,
                    "is_stay": 1,
                    "is_swr_stop": False,
                    "is_mua_stop": False,
                    "n_events": 2,
                    "immobility_time": 5.0,
                },
            ],
        )
        out = stop_replay_summary(st, ev, ea)
        assert len(out) == 1
        row = out.iloc[0]
        # Mean of [0.50, 0.30] on current arm 0
        assert row["frac_on_current"] == pytest.approx(0.40)
        # Mean of [0.20, 0.40] on prev arm 1
        assert row["frac_on_prev"] == pytest.approx(0.30)
        # last_rewarded == prev here
        assert row["frac_on_last_rewarded"] == pytest.approx(0.30)
        # Best arm by recent_reward_rate is arm 0 (rrr=0.8)
        assert row["frac_on_best"] == pytest.approx(0.40)
        # PATCH-LEVEL fractions. Patch 1 = arms {0, 1, 6}; current_arm=0
        # → current_patch=1; arm_fracs sum on patch 1 per event:
        #   event 100: 0.50 + 0.20 + 0.05 = 0.75
        #   event 101: 0.30 + 0.40 + 0.05 = 0.75
        # Mean across the two events = 0.75.
        assert row["frac_on_current_patch"] == pytest.approx(0.75)
        # prev_arm=1, prev_patch=1 → same as current_patch = 0.75
        assert row["frac_on_prev_patch"] == pytest.approx(0.75)
        # last_rewarded_arm=1, patch=1 → 0.75
        assert row["frac_on_last_rewarded_patch"] == pytest.approx(0.75)
        # best_arm=0, patch=1 → 0.75
        assert row["frac_on_best_patch"] == pytest.approx(0.75)
        # Behavioral pass-throughs
        assert int(row["is_after_reward"]) == 1
        assert int(row["is_stay"]) == 1
        assert int(row["n_events_with_replay"]) == 2

    def test_zero_event_stops_have_nan_summaries(self):
        ev, ea, st = _make_tables(
            events=[],
            stops=[
                {
                    "stop_id": 1,
                    "current_arm": 0,
                    "prev_arm": 1,
                    "last_rewarded_arm": 1,
                    "is_after_reward": 1,
                    "is_after_omission": 0,
                    "is_after_stay": 0,
                    "is_after_switch": 0,
                    "is_stay": 1,
                    "is_swr_stop": False,
                    "is_mua_stop": False,
                    "n_events": 0,
                    "immobility_time": 5.0,
                },
            ],
        )
        out = stop_replay_summary(st, ev, ea)
        assert len(out) == 1
        # No replay events → all four fraction columns are NaN.
        for col in (
            "frac_on_current",
            "frac_on_prev",
            "frac_on_last_rewarded",
            "frac_on_best",
        ):
            assert np.isnan(out[col].iloc[0])
        assert int(out["n_events_with_replay"].iloc[0]) == 0

    def test_swr_mua_stratified_fractions(self):
        # Two events: one SWR, one non-SWR. The SWR-only fraction
        # should equal the SWR-event arm_fraction at that role.
        ev, ea, st = _make_tables(
            events=[
                {
                    "stop_id": 1,
                    "event_id": 100,
                    "current_arm": 0,
                    "prev_arm": 1,
                    "last_rewarded_arm": 1,
                    "is_swr": True,
                    "arm_fracs": [0.80, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01],
                    "rrr": [0.8, 0.2, 0.0, 0.4, 0.0, 0.6, np.nan, np.nan, np.nan],
                },
                {
                    "stop_id": 1,
                    "event_id": 101,
                    "current_arm": 0,
                    "prev_arm": 1,
                    "last_rewarded_arm": 1,
                    "is_swr": False,
                    "arm_fracs": [0.10, 0.50, 0.10, 0.10, 0.05, 0.05, 0.05, 0.03, 0.02],
                    "rrr": [0.8, 0.2, 0.0, 0.4, 0.0, 0.6, np.nan, np.nan, np.nan],
                },
            ],
            stops=[
                {
                    "stop_id": 1,
                    "current_arm": 0,
                    "prev_arm": 1,
                    "last_rewarded_arm": 1,
                    "is_after_reward": 1,
                    "is_after_omission": 0,
                    "is_after_stay": 0,
                    "is_after_switch": 0,
                    "is_stay": 1,
                    "is_swr_stop": True,
                    "is_mua_stop": False,
                    "n_events": 2,
                    "immobility_time": 5.0,
                },
            ],
        )
        out = stop_replay_summary(st, ev, ea)
        row = out.iloc[0]
        assert row["frac_on_current_swr"] == pytest.approx(0.80)
        assert row["frac_on_current_non_swr"] == pytest.approx(0.10)

    def test_best_arm_ties_break_by_lowest_arm_id(self):
        """Best-arm lookup should be deterministic, not row-order dependent."""
        ev, ea, st = _make_tables(
            events=[
                {
                    "stop_id": 1,
                    "event_id": 100,
                    "current_arm": 0,
                    "prev_arm": 1,
                    "last_rewarded_arm": 1,
                    # Arm 5 appears first below and has larger replay mass,
                    # but arm 0 ties it on recent_reward_rate and should win.
                    "arm_fracs": [0.10, 0.0, 0.0, 0.0, 0.0, 0.90, 0.0, 0.0, 0.0],
                    "rrr": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, np.nan, np.nan, np.nan],
                },
            ],
            stops=[
                {
                    "stop_id": 1,
                    "current_arm": 0,
                    "prev_arm": 1,
                    "last_rewarded_arm": 1,
                    "is_after_reward": 0,
                    "is_after_omission": 0,
                    "is_after_stay": 0,
                    "is_after_switch": 0,
                    "is_stay": 1,
                    "is_swr_stop": False,
                    "is_mua_stop": False,
                    "n_events": 1,
                    "immobility_time": 5.0,
                },
            ],
        )
        event100 = ea[ea["event_id"] == 100]
        ea = pd.concat(
            [
                event100[event100["arm"] == 5],
                event100[event100["arm"] != 5],
            ],
            ignore_index=True,
        )

        out = stop_replay_summary(st, ev, ea)
        assert out["frac_on_best"].iloc[0] == pytest.approx(0.10)

    def test_non_swr_complement_excludes_unknown_swr_flag(self):
        # When is_swr is NaN/None for some events, those events count
        # toward neither the SWR nor the non-SWR stratum (conservative
        # interpretation: missing flag = unknown). A pandas BooleanDtype
        # with NA must not crash the mask construction.
        ev, ea, st = _make_tables(
            events=[
                {
                    "stop_id": 1,
                    "event_id": 100,
                    "current_arm": 0,
                    "prev_arm": 1,
                    "last_rewarded_arm": 1,
                    "is_swr": True,
                    "arm_fracs": [0.80, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01],
                    "rrr": [0.8, 0.2, 0.0, 0.4, 0.0, 0.6, np.nan, np.nan, np.nan],
                },
                {
                    "stop_id": 1,
                    "event_id": 101,
                    "current_arm": 0,
                    "prev_arm": 1,
                    "last_rewarded_arm": 1,
                    "is_swr": False,
                    "arm_fracs": [0.10, 0.50, 0.10, 0.10, 0.05, 0.05, 0.05, 0.03, 0.02],
                    "rrr": [0.8, 0.2, 0.0, 0.4, 0.0, 0.6, np.nan, np.nan, np.nan],
                },
            ],
            stops=[
                {
                    "stop_id": 1,
                    "current_arm": 0,
                    "prev_arm": 1,
                    "last_rewarded_arm": 1,
                    "is_after_reward": 1,
                    "is_after_omission": 0,
                    "is_after_stay": 0,
                    "is_after_switch": 0,
                    "is_stay": 1,
                    "is_swr_stop": True,
                    "is_mua_stop": False,
                    "n_events": 2,
                    "immobility_time": 5.0,
                },
            ],
        )
        # Cast is_swr to nullable BooleanDtype on event_arm_table and
        # set the second event's flag to NA. The summary must complete
        # without raising and must exclude the NA event from both
        # frac_on_current_swr and frac_on_current_non_swr.
        ea = ea.copy()
        ea["is_swr"] = ea["is_swr"].astype("boolean")
        ea.loc[ea["event_id"] == 101, "is_swr"] = pd.NA
        ev = ev.copy()
        ev["is_swr"] = ev["is_swr"].astype("boolean")
        ev.loc[ev["event_id"] == 101, "is_swr"] = pd.NA

        out = stop_replay_summary(st, ev, ea)
        row = out.iloc[0]
        # Only event 100 has known is_swr=True → current_swr = 0.80
        assert row["frac_on_current_swr"] == pytest.approx(0.80)
        # event 101 has is_swr=NA → excluded from non_swr; nothing left
        # in the non_swr stratum, so frac_on_current_non_swr is NaN.
        assert np.isnan(row["frac_on_current_non_swr"])

    def test_partial_nan_event_propagates_nan_to_stop_summary(self):
        """Regression: stop-level ``frac_on_<role>`` must be strict.

        Pre-fix, the per-role aggregator was ``"mean"`` (pandas
        ``skipna=True``), so a stop with two events — one finite, one
        with NaN ``arm_fraction`` for the current_arm — silently
        emitted ``frac_on_current = <surviving event's value>`` while
        ``n_events_with_replay`` still reported the full event count.
        That biased Stage-6 stop-level fractions toward the subset of
        events that happened to have a finite arm_fraction for this
        role.

        Post-fix uses a strict ``skipna=False`` mean so a stop with
        any missing event-level role fraction is surfaced as missing
        for that role.
        """
        ev, ea, st = _make_tables(
            events=[
                {
                    "stop_id": 1,
                    "event_id": 100,
                    "current_arm": 0,
                    "prev_arm": 1,
                    "last_rewarded_arm": 1,
                    "arm_fracs": [0.50, 0.20, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02],
                    "rrr": [0.8, 0.2, 0.0, 0.4, 0.0, 0.6, np.nan, np.nan, np.nan],
                },
                {
                    "stop_id": 1,
                    "event_id": 101,
                    "current_arm": 0,
                    "prev_arm": 1,
                    "last_rewarded_arm": 1,
                    # arm 0 (= current_arm) is NaN for this event.
                    # Other arms keep finite values so prev / last_rewarded
                    # roles still see two finite contributors.
                    "arm_fracs": [
                        np.nan, 0.40, 0.10, 0.10, 0.10, 0.10, 0.10, 0.06, 0.04,
                    ],
                    "rrr": [0.8, 0.2, 0.0, 0.4, 0.0, 0.6, np.nan, np.nan, np.nan],
                },
            ],
            stops=[
                {
                    "stop_id": 1,
                    "current_arm": 0,
                    "prev_arm": 1,
                    "last_rewarded_arm": 1,
                    "is_after_reward": 1,
                    "is_after_omission": 0,
                    "is_after_stay": 0,
                    "is_after_switch": 0,
                    "is_stay": 1,
                    "is_swr_stop": False,
                    "is_mua_stop": False,
                    "n_events": 2,
                    "immobility_time": 5.0,
                },
            ],
        )
        out = stop_replay_summary(st, ev, ea)
        row = out.iloc[0]
        # frac_on_current must propagate NaN — pre-fix would have
        # emitted 0.50 (the surviving event 100's value).
        assert np.isnan(row["frac_on_current"]), (
            f"frac_on_current={row['frac_on_current']!r} — strict "
            "aggregator should propagate NaN when any event has NaN "
            "for this role; pre-fix silently emitted the surviving "
            "event's value."
        )
        # Roles whose arms ARE finite in both events stay finite.
        assert row["frac_on_prev"] == pytest.approx(0.30)
        assert row["frac_on_last_rewarded"] == pytest.approx(0.30)
        # n_events_with_replay still reports the total event count
        # (it's the per-stop event count, not a per-role contributor
        # count). The per-role NaN above is the missing-data signal.
        assert int(row["n_events_with_replay"]) == 2

    def test_event_arm_table_without_role_columns_falls_back(self):
        # When the event_arm_table omits the role columns (legacy
        # fixtures, or future tables built without the broadcast),
        # stop_replay_summary must pull them from event_table.
        ev, ea, st = _make_tables(
            broadcast_role_cols_to_event_arm=False,
            events=[
                {
                    "stop_id": 1,
                    "event_id": 100,
                    "current_arm": 0,
                    "prev_arm": 1,
                    "last_rewarded_arm": 1,
                    "arm_fracs": [0.50, 0.20, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02],
                    "rrr": [0.8, 0.2, 0.0, 0.4, 0.0, 0.6, np.nan, np.nan, np.nan],
                },
            ],
            stops=[
                {
                    "stop_id": 1,
                    "current_arm": 0,
                    "prev_arm": 1,
                    "last_rewarded_arm": 1,
                    "is_after_reward": 1,
                    "is_after_omission": 0,
                    "is_after_stay": 0,
                    "is_after_switch": 0,
                    "is_stay": 1,
                    "is_swr_stop": False,
                    "is_mua_stop": False,
                    "n_events": 1,
                    "immobility_time": 5.0,
                },
            ],
        )
        # Sanity: confirm the fixture really lacks the role columns.
        assert "current_arm" not in ea.columns
        out = stop_replay_summary(st, ev, ea)
        row = out.iloc[0]
        assert row["frac_on_current"] == pytest.approx(0.50)
        assert row["frac_on_prev"] == pytest.approx(0.20)


# ─────────────────────────────────────────────────────────────────────
# fit_is_stay_logistic_per_session
# ─────────────────────────────────────────────────────────────────────


class TestFitIsStayLogistic:
    def test_recovers_simulated_interaction_sign(self):
        """If is_stay = sigmoid(2 * is_after_omission * frac_on_last_rewarded),
        the fit should return a clearly positive coefficient on the
        is_after_omission × frac_on_last_rewarded interaction."""
        rng = np.random.default_rng(0)
        n_stops = 200
        is_after_omission = rng.integers(0, 2, n_stops)
        frac = rng.uniform(0, 1, n_stops)
        # True linear predictor
        beta_omission = -0.5
        beta_frac = 0.0
        beta_inter = 3.0
        eta = (
            beta_omission * is_after_omission
            + beta_frac * frac
            + (beta_inter * is_after_omission * frac)
        )
        p = 1 / (1 + np.exp(-eta))
        is_stay = (rng.uniform(0, 1, n_stops) < p).astype(int)

        df = pd.DataFrame(
            {
                "animal": "test",
                "day": 1,
                "epoch": "01_r1",
                "stop_id": np.arange(n_stops),
                "is_after_omission": is_after_omission,
                "frac_on_last_rewarded": frac,
                "is_stay": is_stay,
            }
        )

        coefs = fit_is_stay_logistic_per_session(
            df,
            content_col="frac_on_last_rewarded",
        )
        assert len(coefs) == 1
        row = coefs.iloc[0]
        # Recover the truth (β_inter = 3.0) within ~3 SE — the previous
        # ``1.0 < β < 5.0`` window accepts a 30% bias silently.
        assert np.isfinite(row["se_inter"]) and row["se_inter"] > 0
        z = (row["beta_inter"] - 3.0) / row["se_inter"]
        assert abs(z) < 3, (
            f"beta_inter={row['beta_inter']:.3f}, "
            f"se={row['se_inter']:.3f}, z={z:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────
# fit_is_stay_pooled_gee
# ─────────────────────────────────────────────────────────────────────


class TestFitIsStayPooledGEE:
    def test_recovers_pooled_interaction_with_session_clustering(self):
        """Multi-session simulation: every session has the same
        interaction effect, but session-level random intercepts
        added on top. The pooled GEE should recover the interaction
        with a tight cluster-robust SE."""
        rng = np.random.default_rng(0)
        rows = []
        n_sessions = 30
        n_per = 60
        true_inter = 2.5
        for s in range(n_sessions):
            session_intercept = rng.normal(0, 0.8)
            for _ in range(n_per):
                outcome = int(rng.integers(0, 2))
                frac = float(rng.uniform(0, 1))
                eta = session_intercept - 0.3 * outcome + true_inter * outcome * frac
                p = 1 / (1 + np.exp(-eta))
                y = int(rng.uniform() < p)
                rows.append(
                    {
                        "animal": f"a{s:03d}",
                        "day": 1,
                        "epoch": "01_r1",
                        "stop_id": len(rows),
                        "is_after_omission": outcome,
                        "frac_on_last_rewarded": frac,
                        "is_stay": y,
                    }
                )
        df = pd.DataFrame(rows)

        result = fit_is_stay_pooled_gee(df, content_col="frac_on_last_rewarded")
        assert result["status"] == "ok"
        assert result["n_sessions"] == n_sessions
        # Recover the truth (β_inter = 2.5) within ~3 cluster-robust SE.
        assert np.isfinite(result["se_inter"]) and result["se_inter"] > 0
        z = (result["beta_inter"] - true_inter) / result["se_inter"]
        assert abs(z) < 3, (
            f"beta_inter={result['beta_inter']:.3f}, "
            f"se_inter={result['se_inter']:.3f}, z={z:.2f}"
        )
        # Should be highly significant given the strong simulated effect.
        assert result["p_inter"] < 1e-4

    def test_constant_target_returns_status(self):
        df = pd.DataFrame(
            {
                "animal": "a",
                "day": 1,
                "epoch": "01_r1",
                "stop_id": np.arange(100),
                "is_after_omission": np.tile([0, 1], 50),
                "frac_on_last_rewarded": np.linspace(0, 1, 100),
                "is_stay": np.ones(100, dtype=int),
            }
        )
        result = fit_is_stay_pooled_gee(df, content_col="frac_on_last_rewarded")
        assert result["status"] == "constant_target_or_outcome"

    def test_constant_content_col_rejected_as_rank_deficient(self):
        """Regression: a constant ``content_col`` makes the design
        ``[1, outcome, content, outcome*content]`` rank-deficient
        (column 2 collinear with intercept; column 3 collinear with
        outcome). Pre-fix, GEE silently returned ``status='ok'`` with
        ``se_inter`` ~ 1e-17 and a deceptively significant ``p_inter``.
        The fit must instead be rejected with
        ``status='rank_deficient_design'`` and NaN coefficients/SEs/
        p-values, so downstream FDR / reporting cannot pick up a
        mathematically unidentified interaction.
        """
        rng = np.random.default_rng(0)
        rows = []
        n_sessions = 30
        for s in range(n_sessions):
            for stop_id in range(40):
                rows.append(
                    {
                        "animal": f"a{s:03d}",
                        "day": 1,
                        "epoch": "01_r1",
                        "stop_id": stop_id,
                        "is_stay": int(rng.integers(0, 2)),
                        "is_after_omission": int(rng.integers(0, 2)),
                        # Constant across every stop ⇒ rank-deficient
                        # design.
                        "frac_on_last_rewarded": 0.5,
                    }
                )
        df = pd.DataFrame(rows)
        result = fit_is_stay_pooled_gee(df, content_col="frac_on_last_rewarded")
        assert result["status"] == "rank_deficient_design", (
            f"status={result['status']!r}, beta_inter="
            f"{result['beta_inter']}, se_inter={result['se_inter']}, "
            f"p_inter={result['p_inter']} — a constant content_col "
            "must be rejected before GEE fits a non-identifiable model."
        )
        # Coefficients/SEs/p-values must remain NaN — no fit happened.
        for col in (
            "beta_intercept",
            "beta_outcome",
            "beta_content",
            "beta_inter",
            "se_outcome",
            "se_content",
            "se_inter",
            "p_outcome",
            "p_content",
            "p_inter",
        ):
            assert np.isnan(result[col]), (
                f"{col}={result[col]} — should be NaN under "
                "rank_deficient_design, otherwise a downstream "
                "FDR loop could mistake it for a real fit."
            )
