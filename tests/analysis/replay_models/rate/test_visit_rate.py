"""The visit rate model (``replay_models/rate/visit_rate.py``): its terms, the
cohort step, the fit and the third clock's rule.

The fits run on synthetic in-visit bins (``conftest.synthetic_visit_rows``)
whose counts are drawn from a planted rate (``visit_rate_scenarios``); the
cohort step runs on the hand-built session of ``test_visit_design``. One test
reads one real session (``real_data``).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    excluded_intervals_by_session,
    read_swr_table,
)
from continuum_swr_replay.analysis.regression.contrasts import range_grid
from continuum_swr_replay.analysis.regression.newton import CONVERGED
from continuum_swr_replay.analysis.regression.terms import (
    LinearFunctional,
    Slope,
    Smooth,
    build_design,
    log_penalty_from_prior_sd,
)
from continuum_swr_replay.analysis.replay_models.rate import visit_rate as vr
from continuum_swr_replay.analysis.replay_models.rate import (
    visit_rate_scenarios as scenarios,
)
from continuum_swr_replay.analysis.replay_models.rate.visit_rate_statistics import (
    lag_curves,
    session_estimands,
)
from continuum_swr_replay.analysis.replay_models.visit_design import (
    REWARD_LAG_COLUMNS as LAG_COLUMNS,
)
from continuum_swr_replay.paths import table_path
from scripts.replay_models.rate.describe_visit_clocks import session_port_experiences
from tests.analysis.replay_models.test_visit_design import (
    BIN_S,
    EXCLUDED,
    _events,
    _port_experiences,
    _stop_table,
)

#: Planted clocks: a rise over the first seconds of a visit and a drop in the
#: last half second before leaving.
CLOCKS = scenarios.PlantedClocks(
    arrival=lambda s: 1.5 * (1.0 - np.exp(-np.asarray(s) / 1.5)),
    leaving=lambda u: -1.5 * np.exp(-np.asarray(u) / 0.3),
)
LOG_RATIO = float(np.log(1.5))
DEFAULT = vr.VisitRateSettings()


def _planted_cohort(
    make_visit_rows,
    cohort_of,
    scenario,
    *,
    n_sessions: int,
    n_visits: int,
    total: float,
    seed: int,
    kind: str = "SWR",
    **rows_kwargs,
) -> tuple[dict, dict]:
    """Sessions whose counts are drawn from ``scenario(rows, total=...)``, and
    each session's planted values."""
    rng = np.random.default_rng(seed)
    cohort, planted = {}, {}
    for i in range(n_sessions):
        key = (f"rat{i % 3}", f"2020010{i}", "02_r1")
        rows = make_visit_rows(key, n_visits=n_visits, seed=seed + i, **rows_kwargs)
        truth = scenario(rows, total=total)
        rows["count"] = rng.poisson(truth.expected).astype(float)
        cohort[key] = cohort_of(rows, kind)
        planted[key] = truth.pooled
    return cohort, planted


# --- Terms and the guard -------------------------------------------------------


def test_model_terms_follow_the_settings() -> None:
    terms_a = vr.model_terms("A", "SWR", DEFAULT)
    smooths = [t for t in terms_a if isinstance(t, Smooth)]
    assert [t.column for t in smooths] == [vr.ARRIVAL, vr.LEAVING, vr.STRETCH]
    assert [(t.lo, t.hi, t.n_basis) for t in smooths] == [
        (0.0, 20.0, 43),
        (0.0, 3.0, 33),
        (0.0, 1.5, 18),
    ]
    assert [t.penalty for t in smooths] == [
        vr.CLOCK_LOG_PENALTIES["SWR"]["declared"][c]
        for c in (vr.ARRIVAL, vr.LEAVING, vr.STRETCH)
    ]
    (reward,) = [t for t in terms_a if isinstance(t, Slope)]
    assert reward.column == vr.REWARD
    assert reward.penalty == pytest.approx(log_penalty_from_prior_sd(2.0))

    capped = vr.VisitRateSettings(
        max_time_since_arrival_s=10.0, include_stretch_clock=False
    )
    arrival = vr.model_terms("A", "No SWR", capped)[1]
    assert (arrival.hi, arrival.n_basis) == (10.0, 23)
    assert vr.STRETCH not in [
        getattr(t, "column", None) for t in vr.model_terms("A", "SWR", capped)
    ]

    terms_d = vr.model_terms("D", "SWR", DEFAULT)
    (lag,) = [t for t in terms_d if isinstance(t, LinearFunctional)]
    assert lag.lags == tuple(range(1, 11)) and lag.n_basis == 8
    assert lag.hi == pytest.approx(np.log(10.0))
    assert not any(isinstance(t, Slope) for t in terms_d)


def test_the_weak_penalty_is_decided() -> None:
    """The reward coefficient's prior SD is the decided 2 (log penalty
    log(1/4)), with 1 and 4 beside it; the default reward level is the
    session."""
    assert vr.WEAK_PRIOR_SD == 2.0
    assert vr.WEAK_PRIOR_SD_SENSITIVITY == (1.0, 4.0)
    assert DEFAULT.weak_prior_sd == vr.WEAK_PRIOR_SD
    assert DEFAULT.weak_log_penalty == pytest.approx(np.log(1.0 / 4.0))
    assert DEFAULT.reward_level == "session"


@pytest.mark.parametrize(
    "bad",
    [
        {"clock_penalty": "tight"},
        {"max_time_since_arrival_s": 5.0},
        {"reward_form": "spline"},
        {"q1_reference_s": (0.0, 2.0)},
        {"weak_prior_sd": 0.0},
    ],
)
def test_settings_outside_the_allowed_values_are_refused(bad: dict) -> None:
    with pytest.raises(ValueError):
        vr.VisitRateSettings(**bad)


def test_guard_refuses_an_unpenalized_tested_coefficient(make_visit_rows) -> None:
    rows = make_visit_rows(("rat", "20200101", "02_r1"), n_visits=30, seed=0)
    terms = [
        Slope(t.column, "none") if isinstance(t, Slope) else t
        for t in vr.model_terms("A", "SWR", DEFAULT)
    ]
    design = build_design(terms, rows, intercept=True)
    with pytest.raises(ValueError, match="last_outcome_rewarded"):
        vr.require_declared_penalized(design, "A", DEFAULT)
    terms_d = [
        dataclasses.replace(t, penalty="none") if isinstance(t, LinearFunctional) else t
        for t in vr.model_terms("D", "SWR", DEFAULT)
    ]
    design_d = build_design(terms_d, rows, intercept=True)
    with pytest.raises(ValueError, match="lf\\(known_reward_outcome_lag_1"):
        vr.require_declared_penalized(design_d, "D", DEFAULT)
    # The model's own design passes.
    vr.build_visit_rate_design("A", "SWR", rows, DEFAULT)


# --- The declared contrasts ----------------------------------------------------


@pytest.mark.parametrize(
    ("reference", "q1"), [((0.0, 1.0), 3.0 - 0.5), ((1.0, 2.0), 3.0 - 1.5)]
)
def test_declared_contrasts_equal_a_hand_calculation(
    make_visit_rows, reference, q1
) -> None:
    """With the arrival curve ``f(s) = s`` and the leaving curve ``f(u) =
    u`` (up to their constants), Q1 is ``mean(s on [2, 4]) - mean(s on the
    reference)`` and Q7a ``mean(u on [0, 0.25]) - mean(u on [1, 2]) = 0.125 -
    1.5``; Q2 reads the reward coefficient alone."""
    rows = make_visit_rows(("rat", "20200101", "02_r1"), n_visits=40, seed=1)
    settings = vr.VisitRateSettings(q1_reference_s=reference)
    design = vr.build_visit_rate_design("A", "SWR", rows, settings)
    beta = np.zeros(len(design.coef_names))
    for clock, hi in ((vr.ARRIVAL, 20.0), (vr.LEAVING, 3.0)):
        term = f"s({clock})"
        columns = design.by_name[term].columns
        grid = np.linspace(0.0, hi, 401)
        basis = vr.curve_rows(design, term)(grid)[:, columns]
        gamma, *_ = np.linalg.lstsq(
            np.column_stack([basis, np.ones_like(grid)]), grid, rcond=None
        )
        assert np.allclose(basis @ gamma[:-1] + gamma[-1], grid, atol=1e-9)
        beta[columns] = gamma[:-1]
    beta[design.coef_names.index(vr.REWARD)] = 0.7
    c = vr.declared_contrasts(design, settings)
    assert set(c) == {"Q1", "Q2", "Q7a"}
    assert c["Q1"] @ beta == pytest.approx(q1, abs=1e-9)
    assert c["Q7a"] @ beta == pytest.approx(0.125 - 1.5, abs=1e-9)
    assert c["Q2"] @ beta == pytest.approx(0.7)
    assert np.count_nonzero(c["Q2"]) == 1


# --- The cohort step -----------------------------------------------------------


def _cohort(kind: str, settings=DEFAULT, stops=None, events=None, port=None):
    return vr.cohort_rows(
        _stop_table() if stops is None else stops,
        _events() if events is None else events,
        _port_experiences() if port is None else port,
        kind=kind,
        settings=settings,
        excluded_intervals=EXCLUDED,
        bin_s=BIN_S,
    )


def test_cohort_counts_on_the_hand_built_session() -> None:
    """The hand-built session of ``test_visit_design`` in 0.1 s bins.

    SWR: 49 in-visit bins (5 in stop 0's visit part; 9 in stop 1, whose bin
    11.3 lies inside an event; 4 in stop 2's, whose bin 12.9 lies inside one;
    31 in stop 4, less two bins inside the event 20.5 - 20.7), holding the 4
    starts of the default set. No outcome is known before 10.25 s, so the
    bins 10.0, 10.1 and 10.2 have no last outcome: 46 bins for model A, with
    the 4 starts. No bin has ten known outcomes: model D has none.

    No SWR: 52 in-visit bins (stop 0's bin 10.2 lies inside an event) with 2
    starts; the bins 10.0 and 10.1 have no last outcome, and the start at
    10.2 s is counted in the bin that ends there, 10.1: model A has 50 bins
    and 1 start.
    """
    swr = _cohort("SWR")
    counts = swr.counts
    assert counts["bins_visit_context"]["in_visit"] == {"n": 49, "kept": True}
    assert counts["bins_visit_context"]["after_visit_at_well"]["kept"] is False
    assert counts["events_visit_context"]["in_visit"]["n"] == 3
    assert counts["events_visit_context"]["spans_visit_end"]["n"] == 1
    assert counts["events_visit_context"]["track_leaf_arm"] == {"n": 1, "kept": False}
    assert (counts["bins_selected"], counts["events_selected"]) == (49, 4)
    assert (counts["bins_beyond_time_range"], counts["events_beyond_time_range"]) == (
        0,
        0,
    )
    assert counts["bins_missing"][vr.REWARD] == 3
    assert (counts["bins_missing_any"], counts["events_missing_any"]) == (3, 0)
    assert (counts["bins_A"], counts["events_A"]) == (46, 4)
    assert (counts["bins_without_ten_lags"], counts["events_without_ten_lags"]) == (
        46,
        4,
    )
    assert (counts["bins_D"], counts["events_D"]) == (0, 0)
    assert len(swr.rows["A"]) == 46 and swr.rows["D"].empty
    assert (swr.rows["A"]["visit_context"] == "in_visit").all()
    assert swr.rows["A"][vr.REWARD].isin([0.0, 1.0]).all()
    assert counts["events_on_stop_first_sample"] == 0

    no_swr = _cohort("No SWR").counts
    assert (no_swr["bins_selected"], no_swr["events_selected"]) == (52, 2)
    assert (no_swr["bins_missing_any"], no_swr["events_missing_any"]) == (2, 1)
    assert (no_swr["bins_A"], no_swr["events_A"]) == (50, 1)


def test_cohort_counts_the_bins_beyond_the_time_range() -> None:
    """Visit 1 made 8 s longer (its last stop to 31 s, the poke-out at 32 s)
    and an SWR event at 30.5 s: with the 10 s cap the bins whose center is at
    least 30 s, the last 10 bins of the stop, leave, with the event. All of
    them lie within 2 s of the poke-out (Q7a's windows): one visit loses its
    last 2 s."""
    stops = _stop_table()
    stops.loc[stops["stop_id"] == 104, "stop_end"] = 31.0
    stops.loc[stops["overlapping_visit_id"] == 1, "visit_last_poke_out_s"] = 32.0
    port = _port_experiences()
    port.loc[1, "departure_time"] = 32.0
    events = _events()
    events.loc[events["well_visit_id"] == 1, "time_until_visit_end_s"] += 8.0
    late = events.iloc[[5]].assign(
        event_start_s=30.5,
        event_end_s=30.55,
        time_since_visit_start_s=10.5,
        time_until_visit_end_s=1.5,
    )
    events = pd.concat([events, late], ignore_index=True)
    every = _cohort("SWR", stops=stops, events=events, port=port).counts
    capped = _cohort(
        "SWR",
        vr.VisitRateSettings(max_time_since_arrival_s=10.0),
        stops=stops,
        events=events,
        port=port,
    )
    assert every["bins_beyond_time_range"] == 0
    assert every["visits_beyond_time_range_before_leaving"] == 0
    counts = capped.counts
    assert (counts["bins_beyond_time_range"], counts["events_beyond_time_range"]) == (
        10,
        1,
    )
    assert (
        counts["bins_beyond_time_range_before_leaving"],
        counts["events_beyond_time_range_before_leaving"],
        counts["visits_beyond_time_range_before_leaving"],
    ) == (10, 1, 1)
    assert counts["bins_A"] == every["bins_A"] - 10
    assert (capped.rows["A"][vr.ARRIVAL] < 10.0).all()


def _patched_bins(monkeypatch, change) -> None:
    """Make the cohort step read ``visit_timebins``' bins changed by
    ``change(bins)`` (in place)."""
    original = vr.visit_timebins

    def patched(*args, **kwargs):
        bins = original(*args, **kwargs)
        change(bins)
        return bins

    monkeypatch.setattr(vr, "visit_timebins", patched)


def test_a_missing_arrival_time_is_a_missing_covariate(monkeypatch) -> None:
    """A bin of the visit with no time since arrival is counted as missing,
    with the cap as without it, and not as beyond the cap."""

    def blank(bins: pd.DataFrame) -> None:
        in_visit = np.flatnonzero(bins["visit_context"] == "in_visit")
        bins.loc[in_visit[-1], vr.ARRIVAL] = np.nan

    _patched_bins(monkeypatch, blank)
    every = _cohort("SWR").counts
    capped = _cohort("SWR", vr.VisitRateSettings(max_time_since_arrival_s=10.0)).counts
    for counts in (every, capped):
        assert counts["bins_beyond_time_range"] == 0
        assert counts["bins_missing"][vr.ARRIVAL] == 1
        assert counts["bins_missing_any"] == 3 + 1
        assert counts["bins_A"] == 46 - 1


@pytest.mark.parametrize("constant", [True, False])
def test_model_d_leaves_a_session_whose_outcomes_cannot_identify_the_lag_curve(
    monkeypatch, constant
) -> None:
    """Every bin of the hand-built session given ten known outcomes: all the
    same (the lag curve is collinear with the intercept), model D has no row
    and the bins are counted as ``lag_curve_not_identified``; outcomes that
    vary keep every bin."""
    rng = np.random.default_rng(0)

    def ten_lags(bins: pd.DataFrame) -> None:
        values = (
            np.ones((len(bins), len(LAG_COLUMNS)))
            if constant
            else rng.integers(0, 2, (len(bins), len(LAG_COLUMNS)))
        )
        bins[list(LAG_COLUMNS)] = values.astype(float)

    _patched_bins(monkeypatch, ten_lags)
    rows = _cohort("SWR")
    counts = rows.counts
    assert (counts["bins_A"], counts["events_A"]) == (49, 4)
    assert counts["bins_without_ten_lags"] == 0
    if constant:
        assert (
            counts["bins_lag_curve_not_identified"],
            counts["events_lag_curve_not_identified"],
        ) == (49, 4)
        assert counts["bins_D"] == 0 and rows.rows["D"].empty
    else:
        assert counts["bins_lag_curve_not_identified"] == 0
        assert counts["bins_D"] == 49 and len(rows.rows["D"]) == 49


def test_cohort_records_an_event_on_a_stop_first_sample() -> None:
    """A No SWR event from 11.0 (stop 1's first sample) to 11.2 s fills the
    stop's first two bins: it has no at-risk time before it, is not counted,
    and the counts name it."""
    events = _events()
    extra = events.iloc[[0]].assign(
        event_start_s=11.0,
        event_end_s=11.2,
        time_since_visit_start_s=1.0,
        time_until_visit_end_s=2.0,
    )
    events = pd.concat([events, extra], ignore_index=True)
    counts = _cohort("No SWR", events=events).counts
    assert counts["events_on_stop_first_sample"] == 1
    assert counts["starts_not_at_risk_by_reason"]["stop_start"] == 1
    assert counts["events_not_at_risk_in_default_set"] == 1
    assert counts["events_selected"] == 2


# --- The fit -------------------------------------------------------------------


def test_planted_clocks_and_reward_ratio_recovered(make_visit_rows, cohort_of) -> None:
    """Model A on four sessions of 150 visits and 3000 expected events each,
    a planted rate ratio of 1.5 after a reward. Tolerances: the mean over
    sessions of each of Q1, Q2 and Q7a is within 3 standard errors of the
    mean of its planted value, and each fitted clock curve, both centered on
    the grid, is within a tolerance of the planted one: 0.25 for arrival on
    0.5 to 6 s, 0.5 for leaving on 0 to 2 s (simulation tolerances; the
    planted drop of 1.5 in the last 0.3 s is smoothed by the penalty, most at
    the poke-out). The model has no stretch clock here: in a visit's last
    stretch the stretch clock is the leaving clock, so with ``f_m`` in the
    model a planted drop before leaving is shared between ``f_u`` and ``f_m``
    (``visit_rate_statistics.third_clock_q7a`` shows how much)."""
    cohort, planted = _planted_cohort(
        make_visit_rows,
        cohort_of,
        lambda rows, total: scenarios.constant_ratio(
            rows, CLOCKS, log_ratio=LOG_RATIO, total=total
        ),
        n_sessions=4,
        n_visits=150,
        total=3000.0,
        seed=10,
    )
    settings = vr.VisitRateSettings(include_stretch_clock=False)
    fit = vr.fit_visit_rate(
        cohort, model="A", kind="SWR", settings=settings, allow_provisional="test"
    )
    assert [r.status for r in fit.records] == [CONVERGED] * 4
    assert all(r.edf[vr.REWARD] > 0.99 for r in fit.records)
    values = session_estimands(fit)
    for estimand_id, group in values.groupby("estimand_id"):
        truth = np.mean([p[estimand_id] for p in planted.values()])
        se = np.sqrt(group["variance"].mean() / len(group))
        assert abs(group["estimate"].mean() - truth) < 3 * se, estimand_id
    for clock, curve, (lo, hi), tolerance in (
        (vr.ARRIVAL, CLOCKS.arrival, (0.5, 6.0), 0.25),
        (vr.LEAVING, CLOCKS.leaving, (0.0, 2.0), 0.5),
    ):
        grid = range_grid(lo, hi)
        rows = vr.curve_rows(fit.design, f"s({clock})")(grid)
        for record in fit.records:
            fitted = rows @ np.nan_to_num(np.asarray(record.coef))
            truth = curve(grid)
            deviation = (fitted - fitted.mean()) - (truth - truth.mean())
            assert np.max(np.abs(deviation)) < tolerance, clock


#: Planted clocks for the convergence checks: smooth enough that the declared
#: penalties barely bend them on a large session.
SMOOTH_CLOCKS = scenarios.PlantedClocks(
    arrival=lambda s: 1.5 * (1.0 - np.exp(-np.asarray(s) / 1.5)),
    leaving=lambda u: -1.0 * np.exp(-np.asarray(u) / 0.5),
)


def _noiseless_fit(rows, truth, reward_form: str, cohort_of) -> dict[str, float]:
    """Model A of one form on one session whose counts are the expected
    counts (a large session without noise), its Q1, Q2 and Q7a."""
    key = tuple(rows[["animal", "session_date", "epoch"]].iloc[0])
    rows = rows.assign(count=truth.expected)
    settings = vr.VisitRateSettings(
        reward_form=reward_form, include_stretch_clock=False
    )
    fit = vr.fit_visit_rate_session(
        key,
        cohort_of(rows, "SWR"),
        model="A",
        settings=settings,
        allow_provisional="test",
    )
    assert fit.records[0].status == CONVERGED
    return session_estimands(fit).set_index("estimand_id")["estimate"].to_dict()


def test_each_reward_form_converges_to_its_truth(make_visit_rows, cohort_of) -> None:
    """A reward effect that fades from log 2.5 to 0 between 1 and 4 s after
    arrival, in visits shorter after an omission (1 to 4 s) than after a
    reward (2 to 14 s), on one large session without noise. The coefficient
    form's Q1, Q2 and Q7a converge to the pooled truth, which is far from the
    omission side's: the shared curves take the part of the effect the
    coefficient cannot hold. The time-varying form's Q1 and Q7a converge to
    the omission side's. Tolerance 0.01, a simulation tolerance for the
    declared penalties and the projection's cells."""
    rows = make_visit_rows(
        ("rat", "20200101", "02_r1"),
        n_visits=600,
        seed=0,
        two_stretch_share=0.0,
        length_s=(2.0, 14.0),
        omission_length_s=(1.0, 4.0),
    )
    truth = scenarios.fading_ratio(
        rows, SMOOTH_CLOCKS, log_ratio=0.9, fade_start_s=1.0, fade_end_s=4.0, total=2e5
    )
    coefficient = _noiseless_fit(rows, truth, "coefficient", cohort_of)
    for estimand_id in vr.RATE_ESTIMANDS:
        assert coefficient[estimand_id] == pytest.approx(
            truth.pooled[estimand_id], abs=0.01
        ), estimand_id
    assert truth.pooled["Q1"] < truth.omission_side["Q1"] - 0.3
    assert truth.pooled["Q7a"] > truth.omission_side["Q7a"] + 0.05
    time_varying = _noiseless_fit(rows, truth, "time_varying", cohort_of)
    for estimand_id in ("Q1", "Q7a"):
        assert time_varying[estimand_id] == pytest.approx(
            truth.omission_side[estimand_id], abs=0.01
        ), estimand_id


def test_time_varying_q2_is_the_mean_effect_on_the_rewarded_rows(
    make_visit_rows, cohort_of
) -> None:
    """The same fading effect in visits of the same lengths after either
    outcome, so that every rewarded row's time since arrival has omission
    rows: the time-varying form's Q2 converges to the mean of the effect over
    the rewarded rows (the omission side's Q2), the coefficient form's to the
    pooled Q2. Tolerance 0.005, a simulation tolerance."""
    rows = make_visit_rows(
        ("rat", "20200101", "02_r1"),
        n_visits=600,
        seed=0,
        two_stretch_share=0.0,
        length_s=(1.5, 14.0),
    )
    truth = scenarios.fading_ratio(
        rows, SMOOTH_CLOCKS, log_ratio=0.9, fade_start_s=1.0, fade_end_s=4.0, total=2e5
    )
    assert truth.omission_side["Q2"] > truth.pooled["Q2"] + 0.02
    for form, planted in (
        ("time_varying", truth.omission_side),
        ("coefficient", truth.pooled),
    ):
        assert _noiseless_fit(rows, truth, form, cohort_of)["Q2"] == pytest.approx(
            planted["Q2"], abs=0.005
        ), form


def test_model_d_recovers_a_lag_one_effect(make_visit_rows, cohort_of) -> None:
    """A reward effect on the last outcome only (lag 1, ratio 1.5): model D's
    band at lag 1 covers it and holds 0 at lags 3 to 10."""
    cohort, _ = _planted_cohort(
        make_visit_rows,
        cohort_of,
        lambda rows, total: scenarios.constant_ratio(
            rows, CLOCKS, log_ratio=LOG_RATIO, total=total
        ),
        n_sessions=2,
        n_visits=150,
        total=3000.0,
        seed=20,
    )
    fit = vr.fit_visit_rate(
        cohort, model="D", kind="SWR", settings=DEFAULT, allow_provisional="test"
    )
    assert [r.status for r in fit.records] == [CONVERGED] * 2
    curves = lag_curves(fit, seed=0)
    for _, curve in curves.groupby(["animal", "session_date"]):
        curve = curve.set_index("lag")
        assert curve.loc[1, "low"] <= LOG_RATIO <= curve.loc[1, "high"]
        later = curve.loc[3:]
        assert ((later["low"] <= 0.0) & (later["high"] >= 0.0)).all()


SETTINGS = (
    [
        vr.VisitRateSettings(weak_prior_sd=sd)
        for sd in (vr.WEAK_PRIOR_SD, *vr.WEAK_PRIOR_SD_SENSITIVITY)
    ]
    + [vr.VisitRateSettings(clock_penalty=p) for p in vr.CLOCK_PENALTY_SETTINGS]
    + [
        vr.VisitRateSettings(max_time_since_arrival_s=10.0),
        vr.VisitRateSettings(reward_form="time_varying"),
        vr.VisitRateSettings(q1_reference_s=(1.0, 2.0)),
        vr.VisitRateSettings(include_stretch_clock=False),
    ]
)


@pytest.mark.parametrize("settings", SETTINGS, ids=repr)
@pytest.mark.parametrize("kind", ["SWR", "No SWR"])
def test_each_open_setting_builds_and_fits(
    make_visit_rows, cohort_of, settings, kind
) -> None:
    """Every allowed value of every open setting (one at a time) builds both
    models, fits two sessions, records its penalties and gives finite
    estimands."""
    cohort, _ = _planted_cohort(
        make_visit_rows,
        cohort_of,
        lambda rows, total: scenarios.constant_ratio(
            rows, CLOCKS, log_ratio=LOG_RATIO, total=total
        ),
        n_sessions=2,
        n_visits=60,
        total=600.0,
        seed=30,
        kind=kind,
        length_s=(1.5, 14.0),
    )
    if settings.max_time_since_arrival_s is not None:
        cohort = {
            key: cohort_of(
                rows.rows["A"][rows.rows["A"][vr.ARRIVAL] < 10.0].reset_index(
                    drop=True
                ),
                kind,
            )
            for key, rows in cohort.items()
        }
    for model in vr.MODELS:
        fit = vr.fit_visit_rate(
            cohort, model=model, kind=kind, settings=settings, allow_provisional="test"
        )
        assert [r.status for r in fit.records] == [CONVERGED] * 2
        record = fit.records[0]
        assert record.model == vr.MODEL_NAMES[model] and record.kind == kind
        assert np.isfinite(record.coef).all()
        penalties = record.log_penalty
        for clock in settings.clocks():
            assert penalties[f"s({clock})"]["log_penalty"] == pytest.approx(
                settings.log_penalty(kind, clock)
            )
        if model == "A":
            assert penalties[vr.REWARD]["log_penalty"] == pytest.approx(
                settings.weak_log_penalty
            )
            assert (vr.REWARDED_ARRIVAL_TERM in penalties) == (
                settings.reward_form == "time_varying"
            )
            values = session_estimands(fit)
            assert len(values) == 2 * len(vr.RATE_ESTIMANDS)
            assert np.isfinite(values[["estimate", "variance"]]).all().all()
        else:
            assert penalties[vr.LAG_TERM]["log_penalty"] == pytest.approx(
                settings.log_penalty(kind, vr.LAG_TERM)
            )


def test_cohort_counts_join_each_record(make_visit_rows) -> None:
    rows = make_visit_rows(("rat", "20200101", "02_r1"), n_visits=40, seed=2)
    rows["count"] = np.random.default_rng(0).poisson(0.05, len(rows)).astype(float)
    cohort = vr.CohortRows("SWR", {"A": rows, "D": rows}, {"bins_A": len(rows)})
    fit = vr.fit_visit_rate_session(
        ("rat", "20200101", "02_r1"),
        cohort,
        model="A",
        settings=DEFAULT,
        allow_provisional="test",
    )
    (record,) = fit.records
    assert record.cohort_counts["bins_A"] == len(rows)
    assert record.cohort_counts["rows_missing_values"] == 0
    with pytest.raises(ValueError, match="not No SWR"):
        vr.fit_visit_rate(
            {("rat", "20200101", "02_r1"): cohort},
            model="A",
            kind="No SWR",
            settings=DEFAULT,
            allow_provisional="test",
        )


def test_model_d_skips_a_session_with_constant_outcomes(
    make_visit_rows, cohort_of
) -> None:
    """A session rewarded at every visit has no model D rows: its record says
    it was not fitted, its counts say why, the other sessions are fitted, and
    the lag curves keep one row per session and lag, NaN for it."""
    rng = np.random.default_rng(6)
    cohort = {}
    for i, p_reward in enumerate((0.6, 0.6, 1.0)):
        key = (f"rat{i}", "20200101", "02_r1")
        rows = make_visit_rows(key, n_visits=80, seed=i, p_reward=p_reward)
        rows["count"] = rng.poisson(0.2 * rows["exposure"].to_numpy()).astype(float)
        cohort[key] = cohort_of(rows, "SWR")
    constant = cohort[("rat2", "20200101", "02_r1")]
    assert constant.rows["D"].empty
    assert constant.counts["bins_lag_curve_not_identified"] == len(constant.rows["A"])
    settings = vr.VisitRateSettings(include_stretch_clock=False)
    fit = vr.fit_visit_rate(
        cohort, model="D", kind="SWR", settings=settings, allow_provisional="test"
    )
    assert [r.status for r in fit.records[:2]] == [CONVERGED] * 2
    assert fit.records[2].status != CONVERGED and fit.records[2].no_fit
    assert fit.records[2].cohort_counts["bins_lag_curve_not_identified"] > 0
    curves = lag_curves(fit, seed=0)
    assert len(curves) == 3 * len(vr.LAGS)
    assert curves.loc[curves["animal"] == "rat2", "estimate"].isna().all()
    assert curves.loc[curves["animal"] != "rat2", "estimate"].notna().all()


# --- Provisional and calibrated penalties ----------------------------------------


def test_a_real_fit_refuses_provisional_penalties(make_visit_rows, cohort_of) -> None:
    """The module's penalties are provisional: a fit refuses them unless it
    says it is a test or a simulation. With a calibrated penalty for every
    curve it runs as a real fit, at the calibrated values."""
    key = ("rat", "20200101", "02_r1")
    rows = make_visit_rows(key, n_visits=40, seed=2)
    rows["count"] = np.random.default_rng(0).poisson(0.05, len(rows)).astype(float)
    cohort = {key: cohort_of(rows, "SWR")}
    assert vr.CLOCK_LOG_PENALTIES_ARE_PROVISIONAL
    assert DEFAULT.provisional_curves("A", "SWR") == DEFAULT.clocks()
    assert DEFAULT.provisional_curves("D", "SWR") == (*DEFAULT.clocks(), vr.LAG_TERM)
    for model in vr.MODELS:
        with pytest.raises(ValueError, match="provisional"):
            vr.fit_visit_rate(cohort, model=model, kind="SWR", settings=DEFAULT)
    with pytest.raises(ValueError, match="allow_provisional"):
        vr.fit_visit_rate(
            cohort, model="A", kind="SWR", settings=DEFAULT, allow_provisional="real"
        )
    for use in vr.PROVISIONAL_USES:
        fit = vr.fit_visit_rate(
            cohort, model="A", kind="SWR", settings=DEFAULT, allow_provisional=use
        )
        assert fit.records[0].status == CONVERGED

    record = {"tables": {"stop_table": "sha"}, "commit": "abc", "n_sessions": 1}
    calibrated = {
        ("SWR", curve): vr.CalibratedPenalty(1.5 + i, 6.0, record)
        for i, curve in enumerate(DEFAULT.clocks())
    }
    settings = dataclasses.replace(DEFAULT, calibrated_penalties=calibrated)
    assert settings.provisional_curves("A", "SWR") == ()
    assert settings.provisional_curves("A", "No SWR") == DEFAULT.clocks()
    assert settings.provisional_curves("D", "SWR") == (vr.LAG_TERM,)
    fit = vr.fit_visit_rate(cohort, model="A", kind="SWR", settings=settings)
    penalties = fit.records[0].log_penalty
    for (_, curve), value in calibrated.items():
        assert penalties[f"s({curve})"]["log_penalty"] == pytest.approx(
            value.log_penalty
        )
    # A raw override has no record: provisional again.
    overridden = dataclasses.replace(settings, log_penalties={vr.ARRIVAL: 0.0})
    assert overridden.provisional_curves("A", "SWR") == (vr.ARRIVAL,)
    with pytest.raises(ValueError, match="record"):
        vr.CalibratedPenalty(1.0, 6.0, {})


# --- The third clock's rule ----------------------------------------------------


@pytest.mark.parametrize(("two_stretch_share", "keep"), [(0.0, False), (1.0, True)])
def test_third_clock_rule(make_visit_rows, cohort_of, two_stretch_share, keep) -> None:
    """With one stretch per visit the stretch clock is the leaving clock in
    every bin: the rule drops ``f_m``. With two stretches in every visit
    (about half the at-risk time before a small movement) it keeps it."""
    cohort, _ = _planted_cohort(
        make_visit_rows,
        cohort_of,
        lambda rows, total: scenarios.no_effect(rows, CLOCKS, total=total),
        n_sessions=3,
        n_visits=80,
        total=600.0,
        seed=40,
        two_stretch_share=two_stretch_share,
    )
    fit = vr.fit_visit_rate(
        cohort, model="A", kind="SWR", settings=DEFAULT, allow_provisional="test"
    )
    decision = vr.third_clock_decision(fit.records)
    assert decision.keep is keep
    assert decision.n_sessions == 3
    assert decision.share == (0.0 if keep else 1.0)


@pytest.mark.parametrize(("n_flagged", "keep"), [(1, True), (2, False)])
def test_third_clock_rule_at_its_boundary(
    make_visit_rows, cohort_of, n_flagged, keep
) -> None:
    """Five sessions, ``n_flagged`` of them with ``f_m`` not separable: a
    share of exactly 20 % keeps the clock, 40 % removes it."""
    cohort, _ = _planted_cohort(
        make_visit_rows,
        cohort_of,
        lambda rows, total: scenarios.no_effect(rows, CLOCKS, total=total),
        n_sessions=1,
        n_visits=60,
        total=400.0,
        seed=41,
    )
    fit = vr.fit_visit_rate(
        cohort, model="A", kind="SWR", settings=DEFAULT, allow_provisional="test"
    )
    (record,) = fit.records
    stretch, leaving = f"s({vr.STRETCH})", f"s({vr.LEAVING})"

    def measured(name: str, estimate: float) -> dict:
        return {**record.collinearity[name], "estimate": estimate}

    records = [
        dataclasses.replace(
            record,
            collinearity={
                **record.collinearity,
                stretch: measured(stretch, 0.9 if i < n_flagged else 0.1),
                leaving: measured(leaving, 0.1),
            },
        )
        for i in range(5)
    ]
    decision = vr.third_clock_decision(records)
    assert decision.share == pytest.approx(n_flagged / 5)
    assert decision.keep is keep and decision.n_sessions == 5


# --- Calibration ---------------------------------------------------------------


def test_calibration_reaches_its_target(make_visit_rows, cohort_of) -> None:
    cohort, _ = _planted_cohort(
        make_visit_rows,
        cohort_of,
        lambda rows, total: scenarios.no_effect(rows, CLOCKS, total=total),
        n_sessions=3,
        n_visits=60,
        total=600.0,
        seed=50,
    )
    settings = vr.VisitRateSettings(include_stretch_clock=False)
    found = vr.calibrate_log_penalty(
        cohort,
        model="A",
        kind="SWR",
        curve=vr.LEAVING,
        target_edf=5.0,
        settings=settings,
    )
    assert found.median_edf == pytest.approx(5.0, abs=0.01)
    refit = vr.fit_visit_rate(
        cohort,
        model="A",
        kind="SWR",
        settings=dataclasses.replace(
            settings, log_penalties={vr.LEAVING: found.log_penalty}
        ),
        allow_provisional="test",
    )
    edf = [r.edf[f"s({vr.LEAVING})"] for r in refit.records]
    assert np.median(edf) == pytest.approx(5.0, abs=0.01)


# --- One real session ----------------------------------------------------------


@pytest.mark.real_data
def test_one_real_session_of_each_kind() -> None:
    """One epoch of the canonical tables, each kind: the cohort step's counts
    add up, and models A and D fit the session (shapes, statuses and counts
    only; no estimate is asserted)."""
    session = {"animal": "chimi", "session_date": "20200212", "epoch": "02_r1"}
    key = tuple(session.values())

    def _read(name: str) -> pd.DataFrame:
        table = read_swr_table(table_path(name))
        keep = np.logical_and.reduce(
            [table[column].astype(str) == value for column, value in session.items()]
        )
        return table.loc[keep].reset_index(drop=True)

    stops, events = _read("stop_table"), _read("event_table")
    excluded = excluded_intervals_by_session(stops)
    experiences = session_port_experiences(**session)
    for kind in ("SWR", "No SWR"):
        rows = vr.cohort_rows(
            stops,
            events,
            experiences,
            kind=kind,
            settings=DEFAULT,
            excluded_intervals=excluded,
        )
        counts = rows.counts
        for unit in ("bins", "events"):
            assert counts[f"{unit}_selected"] == (
                counts[f"{unit}_beyond_time_range"]
                + counts[f"{unit}_missing_any"]
                + counts[f"{unit}_A"]
            )
            assert counts[f"{unit}_A"] == (
                counts[f"{unit}_without_ten_lags"]
                + counts[f"{unit}_lag_curve_not_identified"]
                + counts[f"{unit}_D"]
            )
        assert counts["bins_selected"] == counts["bins_visit_context"]["in_visit"]["n"]
        default_set = sum(
            label["n"]
            for label in counts["events_visit_context"].values()
            if label["kept"]
        )
        assert default_set == (
            counts["events_selected"] + counts["events_not_at_risk_in_default_set"]
        )
        assert counts["bins_A"] > 0 and counts["bins_D"] > 0
        for model in vr.MODELS:
            fit = vr.fit_visit_rate_session(
                key, rows, model=model, settings=DEFAULT, allow_provisional="test"
            )
            (record,) = fit.records
            assert record.status == CONVERGED and not record.no_fit
            assert len(record.coef) == len(fit.design.coef_names)
            assert record.n_rows == counts[f"bins_{model}"]
            assert record.n_events == counts[f"events_{model}"]
            assert record.cohort_counts["bins_selected"] == counts["bins_selected"]
            if model == "A":
                assert session_estimands(fit).shape[0] == len(vr.RATE_ESTIMANDS)
            else:
                assert lag_curves(fit, seed=0).shape[0] == len(vr.LAGS)
