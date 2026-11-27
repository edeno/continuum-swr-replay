"""The visit rate model (``replay_models/rate/visit_rate.py``): its terms, the
cohort step and the fit.

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
    unit_estimands,
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
DEFAULT = vr.VisitRateSettings(fit_by="session")


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
        planted[key] = truth.by_outcome
    return cohort, planted


# --- Terms and the guard -------------------------------------------------------


def test_model_terms_follow_the_settings() -> None:
    terms_a = vr.model_terms("A", "SWR", DEFAULT)
    smooths = [t for t in terms_a if isinstance(t, Smooth)]
    assert [t.column for t in smooths] == [vr.ARRIVAL, vr.ARRIVAL, vr.LEAVING]
    assert [(t.lo, t.hi, t.n_basis) for t in smooths] == [
        (0.0, 20.0, 43),
        (0.0, 20.0, 43),
        (0.0, 3.0, 33),
    ]
    assert [t.penalty for t in smooths] == [
        vr.CLOCK_LOG_PENALTIES["SWR"]["declared"][c]
        for c in (vr.ARRIVAL, vr.ARRIVAL, vr.LEAVING)
    ]
    assert DEFAULT.clocks() == (vr.ARRIVAL, vr.LEAVING)
    assert "time_until_stretch_end_s" not in [
        getattr(t, "column", None) for t in terms_a
    ]
    (reward,) = [t for t in terms_a if isinstance(t, Slope)]
    assert reward.column == vr.REWARD
    assert reward.penalty == pytest.approx(log_penalty_from_prior_sd(2.0))

    capped = vr.VisitRateSettings(fit_by="session", max_time_since_arrival_s=10.0)
    for arrival in vr.model_terms("A", "No SWR", capped)[1:3]:
        assert (arrival.column, arrival.hi, arrival.n_basis) == (vr.ARRIVAL, 10.0, 23)

    terms_d = vr.model_terms("D", "SWR", DEFAULT)
    (lag,) = [t for t in terms_d if isinstance(t, LinearFunctional)]
    assert lag.lags == tuple(range(1, 11)) and lag.n_basis == 8
    assert lag.hi == pytest.approx(np.log(10.0))
    assert not any(isinstance(t, Slope) for t in terms_d)


def test_the_weak_penalty_is_decided() -> None:
    """The reward coefficient's prior SD is the decided 2 (log penalty
    log(1/4)), with 1 and 4 beside it."""
    assert vr.WEAK_PRIOR_SD == 2.0
    assert vr.WEAK_PRIOR_SD_SENSITIVITY == (1.0, 4.0)
    assert DEFAULT.weak_prior_sd == vr.WEAK_PRIOR_SD
    assert DEFAULT.weak_log_penalty == pytest.approx(np.log(1.0 / 4.0))


@pytest.mark.parametrize(
    "bad",
    [
        {"clock_penalty": "tight"},
        {"max_time_since_arrival_s": 5.0},
        {"reward_form": "spline"},
        {"reward_form": "time_varying"},
        {"q1_reference_s": (0.0, 2.0)},
        {"weak_prior_sd": 0.0},
    ],
)
def test_settings_outside_the_allowed_values_are_refused(bad: dict) -> None:
    with pytest.raises(ValueError):
        vr.VisitRateSettings(fit_by="session", **bad)


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
    """With the arrival curve after an omission ``f_0(s) = s``, after a
    reward ``f_1(s) = 2 s`` and the leaving curve ``f(u) = u`` (each up to
    its constant ``k``), Q1 is ``mean(s on [2, 4]) - mean(s on the
    reference)``, Q7a ``mean(u on [0, 0.25]) - mean(u on [1, 2]) = 0.125 -
    1.5``, and Q2 and Q2-onset the reward coefficient plus the mean of ``s``
    over their windows (7 and 3) plus ``k_0 - k_1``. The coefficient form
    reads the reward coefficient alone."""
    rows = make_visit_rows(("rat", "20200101", "02_r1"), n_visits=40, seed=1)
    settings = vr.VisitRateSettings(fit_by="session", q1_reference_s=reference)
    design = vr.build_visit_rate_design("A", "SWR", rows, settings)
    beta = np.zeros(len(design.coef_names))
    constants = {}
    for term, hi, slope in (
        (vr.OMITTED_ARRIVAL_TERM, 20.0, 1.0),
        (vr.REWARDED_ARRIVAL_TERM, 20.0, 2.0),
        (f"s({vr.LEAVING})", 3.0, 1.0),
    ):
        columns = design.by_name[term].columns
        grid = np.linspace(0.0, hi, 401)
        basis = vr.curve_rows(design, term)(grid)[:, columns]
        gamma, *_ = np.linalg.lstsq(
            np.column_stack([basis, np.ones_like(grid)]), slope * grid, rcond=None
        )
        assert np.allclose(basis @ gamma[:-1] + gamma[-1], slope * grid, atol=1e-9)
        beta[columns] = gamma[:-1]
        constants[term] = gamma[-1]
    beta[design.coef_names.index(vr.REWARD)] = 0.7
    shift = constants[vr.REWARDED_ARRIVAL_TERM] - constants[vr.OMITTED_ARRIVAL_TERM]
    c = vr.declared_contrasts(design, settings)
    assert set(c) == {"Q1", "Q2", "Q2-onset", "Q7a"}
    assert c["Q1"] @ beta == pytest.approx(q1, abs=1e-9)
    assert c["Q7a"] @ beta == pytest.approx(0.125 - 1.5, abs=1e-9)
    assert c["Q2"] @ beta == pytest.approx(0.7 + 7.0 - shift, abs=1e-9)
    assert c["Q2-onset"] @ beta == pytest.approx(0.7 + 3.0 - shift, abs=1e-9)
    coefficient = dataclasses.replace(settings, reward_form="coefficient")
    design_c = vr.build_visit_rate_design("A", "SWR", rows, coefficient)
    (secondary,) = vr.declared_contrasts(design_c, coefficient).values()
    assert np.count_nonzero(secondary) == 1
    assert secondary[design_c.coef_names.index(vr.REWARD)] == 1.0


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
    assert "bins_D" not in counts  # model D's rows are counted at the fit
    assert vr.lag_curve_counts([swr.rows["D"]])["bins_D"] == 0
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
        vr.VisitRateSettings(fit_by="session", max_time_since_arrival_s=10.0),
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
    capped = _cohort(
        "SWR", vr.VisitRateSettings(fit_by="session", max_time_since_arrival_s=10.0)
    ).counts
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
    same (the lag curve is collinear with the intercept), the session alone
    (a unit of one session) has no model D row and the bins are counted as
    ``lag_curve_not_identified``; outcomes that vary keep every bin."""
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
    assert len(rows.rows["D"]) == 49
    unit = vr.lag_curve_counts([rows.rows["D"]])
    if constant:
        assert (
            unit["bins_lag_curve_not_identified"],
            unit["events_lag_curve_not_identified"],
        ) == (49, 4)
        assert unit["bins_D"] == 0
    else:
        assert unit["bins_lag_curve_not_identified"] == 0
        assert unit["bins_D"] == 49


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
    sessions of each declared estimand is within 3 standard errors of the
    mean of its planted value, and each fitted clock curve (each outcome's
    arrival curve, and the leaving curve), both centered on
    the grid, is within a tolerance of the planted one: 0.25 for arrival on
    0.5 to 6 s, 0.5 for leaving on 0 to 2 s (simulation tolerances; the
    planted drop of 1.5 in the last 0.3 s is smoothed by the penalty, most at
    the poke-out)."""
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
    settings = vr.VisitRateSettings(fit_by="session")
    fit = vr.fit_visit_rate(
        cohort, model="A", kind="SWR", settings=settings, allow_provisional="test"
    )
    assert [r.status for r in fit.records] == [CONVERGED] * 4
    assert all(r.edf[vr.REWARD] > 0.99 for r in fit.records)
    values = unit_estimands(fit)
    # Visits last 1.5 to 8 s: Q2's 8 to 10 s part has no row, so Q2 has no
    # value (the window support rule); the others are read.
    q2 = values[values["estimand_id"] == "Q2"]
    assert q2["estimate"].isna().all() and not q2["supported"].any()
    values = values[values["estimand_id"] != "Q2"]
    for estimand_id, group in values.groupby("estimand_id"):
        truth = np.mean([p[estimand_id] for p in planted.values()])
        se = np.sqrt(group["variance"].mean() / len(group))
        assert abs(group["estimate"].mean() - truth) < 3 * se, estimand_id
    for term, curve, (lo, hi), tolerance in (
        (vr.OMITTED_ARRIVAL_TERM, CLOCKS.arrival, (0.5, 6.0), 0.25),
        (vr.REWARDED_ARRIVAL_TERM, CLOCKS.arrival, (0.5, 6.0), 0.25),
        (f"s({vr.LEAVING})", CLOCKS.leaving, (0.0, 2.0), 0.5),
    ):
        grid = range_grid(lo, hi)
        rows = vr.curve_rows(fit.design, term)(grid)
        for record in fit.records:
            fitted = rows @ np.nan_to_num(np.asarray(record.coef))
            truth = curve(grid)
            deviation = (fitted - fitted.mean()) - (truth - truth.mean())
            assert np.max(np.abs(deviation)) < tolerance, term


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
    settings = vr.VisitRateSettings(fit_by="session", reward_form=reward_form)
    fit = vr.fit_visit_rate_session(
        key,
        cohort_of(rows, "SWR"),
        model="A",
        settings=settings,
        allow_provisional="test",
    )
    assert fit.records[0].status == CONVERGED
    return unit_estimands(fit).set_index("estimand_id")["estimate"].to_dict()


def test_each_reward_form_converges_to_its_truth(make_visit_rows, cohort_of) -> None:
    """A reward effect that fades from log 2.5 to 0 between 1 and 4 s after
    arrival, in visits shorter after an omission (1 to 4 s) than after a
    reward (2 to 14 s), on one large session without noise. The coefficient
    form's reward coefficient converges to the pooled truth; the by-outcome
    form's Q1 (the arrival curve after an omission) and Q7a converge to the
    omission side's, which the pooled curves miss, and its Q2 has no value:
    no visit after an omission reaches 4 s, so Q2's window lacks support. Tolerance 0.01, a
    simulation tolerance for the declared penalties and the projection's
    cells."""
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
    assert coefficient[vr.SECONDARY_ESTIMAND] == pytest.approx(
        truth.pooled[vr.SECONDARY_ESTIMAND], abs=0.01
    )
    by_outcome = _noiseless_fit(rows, truth, "by_outcome", cohort_of)
    # No visit after an omission lasts past 4 s: Q2's window lacks support.
    assert np.isnan(by_outcome["Q2"])
    for estimand_id in ("Q1", "Q7a"):
        assert by_outcome[estimand_id] == pytest.approx(
            truth.by_outcome[estimand_id], abs=0.01
        ), estimand_id


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
        vr.VisitRateSettings(fit_by="session", weak_prior_sd=sd)
        for sd in (vr.WEAK_PRIOR_SD, *vr.WEAK_PRIOR_SD_SENSITIVITY)
    ]
    + [
        vr.VisitRateSettings(fit_by="session", clock_penalty=p)
        for p in vr.CLOCK_PENALTY_SETTINGS
    ]
    + [
        vr.VisitRateSettings(fit_by="session", max_time_since_arrival_s=10.0),
        vr.VisitRateSettings(fit_by="session", reward_form="coefficient"),
        vr.VisitRateSettings(fit_by="session", q1_reference_s=(1.0, 2.0)),
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
        for curve in settings.curves(model):
            term = curve if curve.startswith(("s(", "lf(")) else f"s({curve})"
            assert penalties[term]["log_penalty"] == pytest.approx(
                settings.log_penalty(kind, curve)
            )
        if model == "A":
            assert penalties[vr.REWARD]["log_penalty"] == pytest.approx(
                settings.weak_log_penalty
            )
            by_outcome = settings.reward_form == "by_outcome"
            for term in vr.ARRIVAL_BY_OUTCOME.values():
                assert (term in penalties) == by_outcome
            values = unit_estimands(fit)
            n_estimands = len(vr.RATE_ESTIMANDS) if by_outcome else 1
            assert len(values) == 2 * n_estimands
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
    """At ``fit_by="session"`` a session rewarded at every visit has no model
    D rows: its record says it was not fitted, its counts say why, the other
    sessions are fitted, and the lag curves keep one row per session and lag,
    NaN for it."""
    rng = np.random.default_rng(6)
    cohort = {}
    for i, p_reward in enumerate((0.6, 0.6, 1.0)):
        key = (f"rat{i}", "20200101", "02_r1")
        rows = make_visit_rows(key, n_visits=80, seed=i, p_reward=p_reward)
        rows["count"] = rng.poisson(0.2 * rows["exposure"].to_numpy()).astype(float)
        cohort[key] = cohort_of(rows, "SWR")
    constant = cohort[("rat2", "20200101", "02_r1")]
    assert len(constant.rows["D"]) == len(constant.rows["A"])
    settings = vr.VisitRateSettings(fit_by="session")
    fit = vr.fit_visit_rate(
        cohort, model="D", kind="SWR", settings=settings, allow_provisional="test"
    )
    assert [r.status for r in fit.records[:2]] == [CONVERGED] * 2
    assert fit.records[2].status != CONVERGED and fit.records[2].no_fit
    assert fit.records[2].cohort_counts["bins_lag_curve_not_identified"] == len(
        constant.rows["D"]
    )
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
    assert DEFAULT.provisional_curves("A", "SWR") == DEFAULT.curves("A")
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
    curves = DEFAULT.curves("A")
    calibrated = {
        ("SWR", curve): vr.CalibratedPenalty(1.5 + i, 6.0, record)
        for i, curve in enumerate(curves)
    }
    settings = dataclasses.replace(DEFAULT, calibrated_penalties=calibrated)
    assert settings.provisional_curves("A", "SWR") == ()
    assert settings.provisional_curves("A", "No SWR") == curves
    assert settings.provisional_curves("D", "SWR") == (vr.ARRIVAL, vr.LAG_TERM)
    fit = vr.fit_visit_rate(cohort, model="A", kind="SWR", settings=settings)
    penalties = fit.records[0].log_penalty
    for (_, curve), value in calibrated.items():
        term = curve if curve in vr.ARRIVAL_BY_OUTCOME.values() else f"s({curve})"
        assert penalties[term]["log_penalty"] == pytest.approx(value.log_penalty)
    # A raw override has no record: provisional again.
    overridden = dataclasses.replace(
        settings, log_penalties={vr.OMITTED_ARRIVAL_TERM: 0.0}
    )
    assert overridden.provisional_curves("A", "SWR") == (vr.OMITTED_ARRIVAL_TERM,)
    with pytest.raises(ValueError, match="record"):
        vr.CalibratedPenalty(1.0, 6.0, {})


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
    settings = vr.VisitRateSettings(fit_by="session")
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
            unit_d = vr.lag_curve_counts([rows.rows["D"]])
            assert counts[f"{unit}_A"] == (
                counts[f"{unit}_without_ten_lags"]
                + unit_d[f"{unit}_lag_curve_not_identified"]
                + unit_d[f"{unit}_D"]
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
        assert counts["bins_A"] > 0 and len(rows.rows["D"]) > 0
        for model in vr.MODELS:
            fit = vr.fit_visit_rate_session(
                key, rows, model=model, settings=DEFAULT, allow_provisional="test"
            )
            (record,) = fit.records
            assert record.status == CONVERGED and not record.no_fit
            assert len(record.coef) == len(fit.design.coef_names)
            assert record.n_rows == record.cohort_counts[f"bins_{model}"]
            assert record.n_events == record.cohort_counts[f"events_{model}"]
            assert record.cohort_counts["bins_selected"] == counts["bins_selected"]
            if model == "A":
                assert unit_estimands(fit).shape[0] == len(vr.RATE_ESTIMANDS)
            else:
                assert lag_curves(fit, seed=0).shape[0] == len(vr.LAGS)


# --- The unit of the fit ---------------------------------------------------------


def _two_session_days(make_visit_rows, cohort_of, *, p_rewards=(0.6, 0.6, 0.6, 0.6)):
    """Two animals, each one day of two sessions (epochs 02_r1 and 04_r2),
    counts drawn from a constant ratio of 1.5."""
    rng = np.random.default_rng(70)
    cohort = {}
    for i, p_reward in enumerate(p_rewards):
        key = (f"rat{i // 2}", "20200101", ("02_r1", "04_r2")[i % 2])
        rows = make_visit_rows(key, n_visits=60, seed=70 + i, p_reward=p_reward)
        truth = scenarios.constant_ratio(rows, CLOCKS, log_ratio=LOG_RATIO, total=400.0)
        rows["count"] = rng.poisson(truth.expected).astype(float)
        cohort[key] = cohort_of(rows, "SWR")
    return cohort


def test_fit_by_from_the_one_setting(make_visit_rows, cohort_of) -> None:
    """The setting's units are ``fit_by.BEHAVIOR_FIT_BY`` (the first is the
    default); "session" stays a value of the mechanism; another is refused.
    The fit's records are of the setting's unit."""
    from continuum_swr_replay.analysis.replay_models.fit_by import BEHAVIOR_FIT_BY

    assert vr.VisitRateSettings().fit_by == BEHAVIOR_FIT_BY[0]
    with pytest.raises(ValueError, match="fit_by"):
        vr.VisitRateSettings(fit_by="week")
    cohort = _two_session_days(make_visit_rows, cohort_of)
    for fit_by, n_units in (("session", 4), ("day", 2), ("animal", 2)):
        settings = vr.VisitRateSettings(fit_by=fit_by)
        fit = vr.fit_visit_rate(
            cohort, model="A", kind="SWR", settings=settings, allow_provisional="test"
        )
        assert [r.fit_by for r in fit.records] == [fit_by] * n_units
        assert all(r.status == CONVERGED for r in fit.records)


def test_day_fit_has_one_intercept_per_session(make_visit_rows, cohort_of) -> None:
    """At ``fit_by="day"`` the intercept is the baseline: one per session of
    the day, the curves and the reward coefficient shared; the cohort counts
    are the sum of the day's sessions' counts."""
    cohort = _two_session_days(make_visit_rows, cohort_of)
    settings = vr.VisitRateSettings(fit_by="day")
    fit = vr.fit_visit_rate(
        cohort, model="A", kind="SWR", settings=settings, allow_provisional="test"
    )
    assert fit.design.baseline == ("(Intercept)",)
    first = fit.records[0]
    intercepts = [n for n in first.coef_names if n.startswith("(Intercept)")]
    assert intercepts == [
        "(Intercept)@rat0/20200101/02_r1",
        "(Intercept)@rat0/20200101/04_r2",
    ]
    assert first.coef_names.count(vr.REWARD) == 1
    assert [s["epoch"] for s in first.sessions] == ["02_r1", "04_r2"]
    keys = [k for k in cohort if k[0] == "rat0"]
    assert first.cohort_counts["bins_without_ten_lags"] == sum(
        cohort[k].counts["bins_without_ten_lags"] for k in keys
    )
    assert first.cohort_counts["bins_D"] == sum(len(cohort[k].rows["D"]) for k in keys)


def test_model_d_identified_on_the_unit(make_visit_rows, cohort_of) -> None:
    """A session rewarded at every visit cannot identify the lag curve alone,
    but can inside a day with a session of mixed outcomes: at
    ``fit_by="session"`` it is not fitted (its rows counted as
    ``lag_curve_not_identified``), at ``fit_by="day"`` its rows join the day's
    fit. A day whose sessions are all rewarded at every visit is not fitted
    at either unit."""
    cohort = _two_session_days(
        make_visit_rows, cohort_of, p_rewards=(0.6, 1.0, 1.0, 1.0)
    )
    settings = vr.VisitRateSettings(fit_by="session")
    by_session = vr.fit_visit_rate(
        cohort, model="D", kind="SWR", settings=settings, allow_provisional="test"
    )
    assert [r.status == CONVERGED for r in by_session.records] == [
        True,
        False,
        False,
        False,
    ]
    constant = by_session.records[1].cohort_counts
    rows_d = cohort[("rat0", "20200101", "04_r2")].rows["D"]
    assert len(rows_d) > 0
    assert constant["bins_lag_curve_not_identified"] == len(rows_d)
    assert constant["bins_D"] == 0
    by_day = vr.fit_visit_rate(
        cohort,
        model="D",
        kind="SWR",
        settings=dataclasses.replace(settings, fit_by="day"),
        allow_provisional="test",
    )
    mixed, constant_day = by_day.records
    assert mixed.status == CONVERGED
    assert mixed.cohort_counts["bins_lag_curve_not_identified"] == 0
    assert [s["n_rows"] > 0 for s in mixed.sessions] == [True, True]
    assert constant_day.no_fit
    assert constant_day.cohort_counts["bins_D"] == 0
    # Model A's records carry the same counts of model D's rows.
    by_day_a = vr.fit_visit_rate(
        cohort,
        model="A",
        kind="SWR",
        settings=dataclasses.replace(settings, fit_by="day"),
        allow_provisional="test",
    )
    for a, d in zip(by_day_a.records, by_day.records, strict=True):
        assert a.cohort_counts["bins_D"] == d.cohort_counts["bins_D"]


def test_calibration_at_the_unit(make_visit_rows, cohort_of) -> None:
    """The calibration fits at the setting's unit: one edf per day."""
    cohort = _two_session_days(make_visit_rows, cohort_of)
    settings = vr.VisitRateSettings(fit_by="day")
    found = vr.calibrate_log_penalty(
        cohort,
        model="A",
        kind="SWR",
        curve=vr.LEAVING,
        target_edf=5.0,
        settings=settings,
    )
    assert list(found.edf.index) == [("rat0", "20200101"), ("rat1", "20200101")]
    assert found.median_edf == pytest.approx(5.0, abs=0.01)


# --- The by-outcome form of the reward effect -------------------------------------


def test_the_by_outcome_form_is_the_declared_form() -> None:
    """The default form is an arrival curve for each last outcome (the
    twenty-sixth round), each a smooth by the outcome with its own penalty
    block, beside the shared leaving curve and the outcome coefficient."""
    assert vr.REWARD_FORMS == ("by_outcome", "coefficient")
    assert DEFAULT.reward_form == "by_outcome"
    terms = vr.model_terms("A", "SWR", DEFAULT)
    arrival = [t for t in terms if isinstance(t, Smooth) and t.column == vr.ARRIVAL]
    assert [t.levels for t in arrival] == [(0,), (1,)]
    assert all(t.by.column == vr.REWARD for t in arrival)
    assert [t.column for t in terms if isinstance(t, Smooth)].count(vr.LEAVING) == 1
    assert DEFAULT.curves("A") == (
        vr.OMITTED_ARRIVAL_TERM,
        vr.REWARDED_ARRIVAL_TERM,
        vr.LEAVING,
    )


def test_each_arrival_curve_has_its_own_penalty_and_constraint(make_visit_rows) -> None:
    """Two penalty blocks, one per outcome's curve, each at its own declared
    value; each curve sums to zero over its outcome's pooled rows."""
    rows = make_visit_rows(("rat", "20200101", "02_r1"), n_visits=60, seed=3)
    settings = dataclasses.replace(
        DEFAULT,
        log_penalties={vr.OMITTED_ARRIVAL_TERM: 0.5, vr.REWARDED_ARRIVAL_TERM: 2.5},
    )
    design = vr.build_visit_rate_design("A", "SWR", rows, settings)
    record = design.log_penalty_record()
    assert record[vr.OMITTED_ARRIVAL_TERM]["log_penalty"] == pytest.approx(0.5)
    assert record[vr.REWARDED_ARRIVAL_TERM]["log_penalty"] == pytest.approx(2.5)
    blocks = {design.by_name[t].block for t in vr.ARRIVAL_BY_OUTCOME.values()}
    assert len(blocks) == 2 and None not in blocks
    X = design.build(rows).X
    for level, term in vr.ARRIVAL_BY_OUTCOME.items():
        columns = design.by_name[term].columns
        on = rows[vr.REWARD].to_numpy() == level
        np.testing.assert_allclose(X[on][:, columns].sum(axis=0), 0.0, atol=1e-8)
        assert not X[~on][:, columns].any()


def test_the_by_outcome_form_recovers_a_later_onset(make_visit_rows, cohort_of) -> None:
    """A later onset after an omission (the arrival curve 1 s later) and no
    ratio, on one large session without noise, visits of the same lengths
    after either outcome: the fitted difference curve, the outcome
    coefficient plus the rewarded curve less the omitted one, follows the
    planted difference from 1 to 8 s, and Q2 and the onset test are its
    means over their windows (tolerance 0.05 on the curve, 0.02 on the
    windows: simulation tolerances of the declared penalties)."""
    rows = make_visit_rows(
        ("rat", "20200101", "02_r1"),
        n_visits=800,
        seed=4,
        two_stretch_share=0.0,
        length_s=(1.0, 14.0),
    )

    def delayed(s):
        return SMOOTH_CLOCKS.arrival(np.maximum(np.asarray(s) - 1.0, 0.0))

    def planted(s):
        return SMOOTH_CLOCKS.arrival(np.asarray(s)) - delayed(s)

    truth = scenarios.later_onset_after_omission(
        rows, SMOOTH_CLOCKS, delay_s=1.0, total=2e5
    )
    key = ("rat", "20200101", "02_r1")
    fit = vr.fit_visit_rate_session(
        key,
        cohort_of(rows.assign(count=truth.expected), "SWR"),
        model="A",
        settings=DEFAULT,
        allow_provisional="test",
    )
    (record,) = fit.records
    assert record.status == CONVERGED
    beta = np.asarray(record.coef)
    grid = range_grid(1.0, 8.0)
    reward = fit.design.coef_names.index(vr.REWARD)
    difference = (
        beta[reward]
        + vr.curve_rows(fit.design, vr.REWARDED_ARRIVAL_TERM)(grid) @ beta
        - vr.curve_rows(fit.design, vr.OMITTED_ARRIVAL_TERM)(grid) @ beta
    )
    assert np.max(np.abs(difference - planted(grid))) < 0.05
    values = unit_estimands(fit).set_index("estimand_id")["estimate"]
    for estimand_id, window in (("Q2", (4.0, 10.0)), ("Q2-onset", (2.0, 4.0))):
        assert values[estimand_id] == pytest.approx(
            np.mean(planted(range_grid(*window))), abs=0.02
        ), estimand_id


@pytest.mark.parametrize("n_omitted", [0, 3])
def test_a_unit_with_few_omission_visits(make_visit_rows, cohort_of, n_omitted) -> None:
    """A day whose visits after an omission are none or three (below
    ``MIN_UNITS``): the outcome coefficient fails the support rule and is
    left out, so Q2 and the onset test have no value; with none the curve
    after an omission is left out too (its columns are zero), and the unit
    is still fitted (no shared arrival curve to be collinear with the
    rewarded one); with three it stays in the fit (the known issue
    ``docs/known_issues/smooth_by_a_thin_level.md``)."""
    key = ("rat", "20200101", "02_r1")
    rows = make_visit_rows(key, n_visits=60, seed=5, p_reward=1.0)
    omitted_visits = rows["well_visit_id"].unique()[:n_omitted]
    on = rows["well_visit_id"].isin(omitted_visits)
    rows.loc[on, vr.REWARD] = 0.0
    rows["count"] = (
        np.random.default_rng(0).poisson(0.6 * rows["exposure"]).astype(float)
    )
    other = make_visit_rows(("rat", "20200102", "02_r1"), n_visits=60, seed=6)
    other["count"] = (
        np.random.default_rng(1).poisson(0.6 * other["exposure"]).astype(float)
    )
    cohort = {
        key: cohort_of(rows, "SWR"),
        ("rat", "20200102", "02_r1"): cohort_of(other, "SWR"),
    }
    fit = vr.fit_visit_rate(
        cohort, model="A", kind="SWR", settings=DEFAULT, allow_provisional="test"
    )
    thin = fit.records[0]
    assert thin.status == CONVERGED and not thin.no_fit
    assert thin.left_out[vr.REWARD] == "support"
    assert (vr.OMITTED_ARRIVAL_TERM + ".0" in thin.left_out) == (n_omitted == 0)
    values = unit_estimands(fit)
    values = values[values["session_date"] == "20200101"].set_index("estimand_id")
    assert values.loc[["Q2", "Q2-onset"], "estimate"].isna().all()
    assert np.isfinite(values.loc["Q7a", "estimate"])


def test_the_window_tests_are_declared_per_animal() -> None:
    """One rule in one place: Q2 and the onset test are declared tests at the
    animal unit only (the twenty-sixth round: "the window tests are read per
    animal, not per day"); Q1 and Q7a at every unit; Q1b and the secondary
    reward coefficient at none."""
    assert vr.WINDOW_TEST_UNIT == "animal"
    assert vr.WINDOW_TESTS == ("Q2", "Q2-onset")
    for fit_by in ("session", "day", "animal"):
        for estimand_id in ("Q2", "Q2-onset"):
            assert vr.is_declared_test(estimand_id, fit_by) == (fit_by == "animal")
        assert vr.is_declared_test("Q1", fit_by)
        assert vr.is_declared_test("Q7a", fit_by)
        assert not vr.is_declared_test("Q1b", fit_by)
        assert not vr.is_declared_test(vr.SECONDARY_ESTIMAND, fit_by)
