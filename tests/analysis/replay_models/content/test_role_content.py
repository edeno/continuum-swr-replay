"""The role content models: cohort step, design, contrasts and fit."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.regression.pspline import PSplineBasis
from continuum_swr_replay.analysis.regression.support import require_tested_penalized
from continuum_swr_replay.analysis.regression.terms import (
    Category,
    Slope,
    build_design,
)
from continuum_swr_replay.analysis.replay_models.content import role_content as rc
from continuum_swr_replay.analysis.replay_models.content import (
    role_content_scenarios as sc,
)
from continuum_swr_replay.analysis.replay_models.content import (
    role_content_statistics as rs,
)

from .role_tables import draw_shares, synthetic_event_arm_table

SESSIONS = (
    ("a1", "20200101", "02_r1"),
    ("a1", "20200102", "02_r1"),
    ("a2", "20200103", "04_r2"),
)
KIND = "No SWR"


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return synthetic_event_arm_table(SESSIONS, n_visits=150, swr_share=0.3, seed=3)


@pytest.fixture(scope="module")
def cohort(table) -> rc.RoleContentCohort:
    return rc.role_content_cohort(table)


@pytest.fixture(scope="module")
def big_cohort() -> rc.RoleContentCohort:
    """Larger sessions, one kind, for recovery."""
    table = synthetic_event_arm_table(
        SESSIONS, n_visits=500, events_per_visit=3.0, swr_share=0.0, seed=11
    )
    return rc.role_content_cohort(table)


def _simulate(cohort, truth, seed, n_bins=60):
    rows = cohort.rows[cohort.rows[rc.KIND] == KIND]
    reference = sc.ScenarioReference(rows, cohort.reference_log_duration)
    sessions = {
        key: draw_shares(
            part, sc.expected_shares(part, truth, reference), n_bins, seed + i
        )
        for i, (key, part) in enumerate(rc.session_rows(rows).items())
    }
    return sessions, reference


def _fit(cohort, sessions, model, variant="unadjusted", settings=None):
    return rc.fit_role_content(
        sessions,
        model=model,
        variant=variant,
        kind=KIND,
        reference_log_duration=cohort.reference_log_duration,
        settings=settings,
    )


def _pooled(fit) -> pd.DataFrame:
    """Each estimand's mean over sessions and the SE of that mean."""
    per = rs.session_estimands(fit)
    assert per["kept"].all()
    grouped = per.groupby("estimand_id")
    return pd.DataFrame(
        {
            "estimate": grouped["estimate"].mean(),
            "se": np.sqrt(grouped["variance"].sum()) / grouped.size(),
        }
    )


def _assert_recovered(pooled, planted, spread) -> None:
    """Each estimand's pooled estimate within its simulation tolerance of the
    planted value: 4 times ``spread`` (the SD of the pooled estimate measured
    over replicate draws) or 20 % of the planted value, whichever is larger.

    The model's own standard error is not the tolerance: under the share
    response (each event one categorical draw) it is about ten times the
    spread measured here, wide enough to pass a slope set to 0.
    """
    for eid, sd in spread.items():
        est = pooled.loc[eid, "estimate"]
        tol = max(4.0 * sd, 0.2 * abs(planted[eid]))
        assert abs(est - planted[eid]) < tol, (eid, est, planted[eid], tol)


# ---------------------------------------------------------------------------
# The cohort step
# ---------------------------------------------------------------------------


def test_roles_exclusive_and_complete(cohort) -> None:
    rows = cohort.rows
    assert set(rows[rc.ROLE]) <= set(rc.ROLES)
    per_event = rows.groupby(rc.UNIT)[rc.ROLE]
    assert (per_event.apply(lambda r: (r == "current").sum()) == 1).all()
    assert (per_event.apply(lambda r: (r == "previous").sum()) <= 1).all()
    assert (per_event.size() == 9).all()
    trunk = rows[rows[rc.ARM].isin((6, 7, 8))]
    assert (trunk[rc.ROLE] == "trunk").all()
    leaf_columns = [rc.REWARD_FRACTION, rc.VALUE, rc.BELIEF_VARIANCE, rc.SAMPLE_COUNT]
    assert (trunk[leaf_columns] == 0).all().all()
    # The recency and never-visited flag are read on the other-leaf rows only.
    not_other = rows[rows[rc.ROLE] != rc.REFERENCE_ROLE]
    assert (not_other[[rc.RECENCY, rc.NEVER_VISITED]] == 0).all().all()


def test_previous_only_when_it_differs_from_current() -> None:
    role = rc.assign_roles(
        np.array([1, 2, 1, 2, 7]),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([2.0, 2.0, 1.0, np.nan, 7.0]),
    )
    assert role.tolist() == ["current", "previous", "current", "other_leaf", "trunk"]


def test_spans_visit_end_events_in_cohort(table, cohort) -> None:
    spans = table.loc[table["visit_context"] == "spans_visit_end"]
    kept = cohort.rows[cohort.rows[rc.UNIT].isin(spans[rc.UNIT])]
    assert kept[rc.UNIT].nunique() == spans[rc.UNIT].nunique() > 0
    current = kept[kept[rc.ROLE] == "current"].set_index(rc.UNIT)[rc.ARM]
    well_arm = spans.groupby(rc.UNIT)["occupancy_well_arm"].first()
    assert (current.sort_index() == well_arm.loc[current.index].sort_index()).all()
    outside = table.loc[table["visit_context"] == "after_visit_at_well", rc.UNIT]
    assert not cohort.rows[rc.UNIT].isin(outside).any()
    counts = cohort.counts["visit_context"]
    assert counts["spans_visit_end"] == {
        "n": spans[rc.UNIT].nunique(),
        "kept": True,
    }
    assert counts["after_visit_at_well"]["kept"] is False


def test_missing_covariate_drops_the_event_and_is_counted(table) -> None:
    table = table.copy()
    leaf = table[rc.ARM] < 6
    event = table.loc[leaf & (table["visit_context"] == "in_visit"), rc.UNIT].iloc[0]
    kind = "SWR" if table.loc[table[rc.UNIT] == event, "is_swr"].iloc[0] else "No SWR"
    row = table.index[(table[rc.UNIT] == event) & leaf][0]
    table.loc[row, "reward_belief_variance_candidate_arm_preoutcome"] = np.nan
    cohort = rc.role_content_cohort(table)
    assert event not in set(cohort.rows[rc.UNIT])
    counts = cohort.counts["by_kind"][kind]
    assert counts["events_dropped"] == 1
    assert counts["events_missing"][rc.BELIEF_VARIANCE] == 1
    assert counts["rows_dropped"] == 9
    assert counts["events_kept"] == counts["events_selected"] - 1


def test_a_value_on_a_row_that_does_not_read_it_drops_nothing(table) -> None:
    # Trunk rows carry NaN beliefs on the real table; they are not read.
    assert (
        table.loc[table[rc.ARM] >= 6, "reward_belief_mean_candidate_arm_preoutcome"]
        .isna()
        .all()
    )
    cohort = rc.role_content_cohort(table)
    for counts in cohort.counts["by_kind"].values():
        assert counts["events_dropped"] == 0


def test_reference_log_duration_is_the_pooled_median(table, cohort) -> None:
    kept = table[table["visit_context"].isin(["in_visit", "spans_visit_end"])]
    durations = kept.groupby(rc.UNIT)["event_duration_s"].first()
    assert cohort.reference_log_duration == pytest.approx(np.median(np.log(durations)))


@pytest.mark.parametrize(
    "change, match",
    [
        (
            lambda t: t.assign(candidate_arm=t[rc.ARM].replace({8: -1})),
            "outside the nine",
        ),
        (
            lambda t: t[~((t[rc.UNIT] == t[rc.UNIT].iloc[0]) & (t[rc.ARM] == 3))],
            "exactly one row",
        ),
        (lambda t: t.assign(occupancy_well_arm=6.0), "leaf arm of the visited well"),
        (lambda t: t.assign(arm_fraction=0.2), "sum to one"),
    ],
)
def test_the_track_rules_are_checked(table, change, match) -> None:
    """A junction place bin is not an arm; each well touches one leaf arm."""
    with pytest.raises(ValueError, match=match):
        rc.role_content_cohort(change(table))


# ---------------------------------------------------------------------------
# The design and its contrasts
# ---------------------------------------------------------------------------


def _design(cohort, model, variant, settings=None):
    rows = cohort.rows[cohort.rows[rc.KIND] == KIND]
    reference = rc.ContrastReference.from_rows(
        rows, cohort.reference_log_duration, model
    )
    design = rc.role_design(rows, model, variant, settings, reference=reference)
    return design, reference, rows


def test_unadjusted_coefficient_contrasts_are_the_declared_coefficients(cohort) -> None:
    for model, names in (
        ("role_content", {"Q3a": "role[current]", "Q3b": "role[previous]", "Q4b": "value:role=other_leaf"}),
        ("role_content_uncertainty", {"Q5b-var": "belief_variance:role=other_leaf", "Q5b-n": "sample_count:role=other_leaf"}),
    ):  # fmt: skip
        design, reference, _ = _design(cohort, model, "unadjusted")
        contrasts = rc.estimand_contrasts(design, model, "unadjusted", reference)
        for eid, name in names.items():
            expected = np.zeros(len(design.coef_names))
            expected[design.coef_names.index(name)] = 1.0
            np.testing.assert_allclose(contrasts[eid], expected, atol=1e-12)


def _basis_rows(design, term, x):
    """A smooth term's constrained basis at ``x``, by hand from its record."""
    record = design.basis_record()[term]
    lo, hi = record["range"]
    n_basis = len(record["knots"]) - record["degree"] - 1
    basis = PSplineBasis(lo, hi, n_basis, record["degree"])
    return basis.design(np.asarray(x, dtype=float)) @ np.asarray(record["constraint"])


def test_curve_contrasts_by_hand(cohort, big_cohort) -> None:
    design, reference, rows = _design(cohort, "role_content", "unadjusted")
    contrasts = rc.estimand_contrasts(design, "role_content", "unadjusted", reference)
    names = np.asarray(design.coef_names)
    # Q4a: f(m + sd/2) - f(m - sd/2) on the current rows' pooled mean and SD.
    term = "s(reward_fraction):role=current"
    current = rows.loc[rows[rc.ROLE] == "current", rc.REWARD_FRACTION]
    m, sd = current.mean(), current.std(ddof=0)
    by_hand = _basis_rows(design, term, [m + sd / 2, m - sd / 2])
    columns = [j for j, n in enumerate(names) if n.startswith(term + ".")]
    np.testing.assert_allclose(
        contrasts["Q4a"][columns], by_hand[0] - by_hand[1], atol=1e-12
    )
    assert np.count_nonzero(contrasts["Q4a"]) <= len(columns)
    # Q7b: the mean over [0, 0.25] s less the mean over [1, 2] s, 0.01 s apart.
    term = "s(time_until_visit_end_s):role=current"
    columns = [j for j, n in enumerate(names) if n.startswith(term + ".")]
    first = _basis_rows(design, term, np.linspace(0, 0.25, 26)).mean(axis=0)
    second = _basis_rows(design, term, np.linspace(1, 2, 101)).mean(axis=0)
    np.testing.assert_allclose(contrasts["Q7b"][columns], first - second, atol=1e-12)
    # Q5a: the recency curve at log 20 less at log 3.
    design, reference, _ = _design(cohort, "role_content_uncertainty", "unadjusted")
    contrasts = rc.estimand_contrasts(
        design, "role_content_uncertainty", "unadjusted", reference
    )
    term = "s(log_trials_since_last_visit):role=other_leaf"
    names = np.asarray(design.coef_names)
    columns = [j for j, n in enumerate(names) if n.startswith(term + ".")]
    by_hand = _basis_rows(design, term, np.log([20.0, 3.0]))
    np.testing.assert_allclose(
        contrasts["Q5a"][columns], by_hand[0] - by_hand[1], atol=1e-12
    )


def test_length_adjusted_role_contrast_by_hand(cohort) -> None:
    """At L0 the role's length slope is at (0 - its pooled reference) / SD;
    each product with the length, centered at the covariate's pooled mean, is
    0 there, so only its own pooled centering remains."""
    design, reference, _ = _design(cohort, "role_content", "length_adjusted")
    c = rc.estimand_contrasts(design, "role_content", "length_adjusted", reference)[
        "Q3a"
    ]
    std = design.standardization_record()
    expected = np.zeros(len(design.coef_names))
    index = design.coef_names.index
    expected[index("role[current]")] = 1.0
    entry = std[rc.LOG_DURATION_REL]
    expected[index("log_duration_rel:role=current")] = -entry["reference"] / entry["sd"]
    entry = std["value_x_log_duration_rel"]
    expected[index("value_x_log_duration_rel:role=current")] = (
        -entry["reference"] / entry["sd"]
    )
    expected[index("value_x_log_duration_rel:role=other_leaf")] = (
        entry["reference"] / entry["sd"]
    )
    np.testing.assert_allclose(c, expected, atol=1e-12)
    # The slope of Q4b at L0 is its own coefficient.
    c = rc.estimand_contrasts(design, "role_content", "length_adjusted", reference)[
        "Q4b"
    ]
    expected = np.zeros(len(design.coef_names))
    expected[index("value:role=other_leaf")] = 1.0
    np.testing.assert_allclose(c, expected, atol=1e-12)


def test_the_guard_refuses_an_unpenalized_tested_coefficient(cohort) -> None:
    design, reference, rows = _design(cohort, "role_content", "unadjusted")
    contrasts = rc.estimand_contrasts(design, "role_content", "unadjusted", reference)
    require_tested_penalized(design, rc.tested_coefficients(design, contrasts))
    terms = [
        replace(t, penalty="none")
        if isinstance(t, Category) and t.column == rc.ROLE
        else t
        for t in rc.role_terms("role_content", "unadjusted")
    ]
    unpenalized = build_design(terms, rows, intercept=False)
    tested = rc.tested_coefficients(
        unpenalized,
        rc.estimand_contrasts(unpenalized, "role_content", "unadjusted", reference),
    )
    with pytest.raises(ValueError, match="role\\[current\\].*role\\[previous\\]"):
        require_tested_penalized(unpenalized, tested)


def test_arm_effects_are_baselines_and_role_slopes_penalized(cohort) -> None:
    design, _, _ = _design(cohort, "role_content_uncertainty", "length_adjusted")
    penalties = design.log_penalty_record()
    weak = -2.0 * np.log(rc.DEFAULT_PRIOR_SD)
    assert penalties[rc.ARM]["set"] == "none"
    assert penalties[rc.NEVER_VISITED]["set"] == "none"
    assert penalties["log_duration_rel:role=trunk"]["set"] == "none"
    for name in ("role", "value:role=current", "belief_variance:role=other_leaf",
                 "trial_number:role=previous", "log_duration_rel:role=current",
                 "sample_count_x_log_duration_rel:role=other_leaf"):  # fmt: skip
        assert penalties[name] == {"set": "fixed", "log_penalty": pytest.approx(weak)}
    for covariate, value in rc.DEFAULT_SMOOTH_LOG_PENALTIES.items():
        smooths = [n for n in penalties if n.startswith(f"s({covariate})")]
        assert smooths and all(penalties[n]["log_penalty"] == value for n in smooths)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def test_planted_role_levels_and_opposite_slopes_are_recovered(big_cohort) -> None:
    truth = sc.Truth(
        arm=sc.arm_effects(),
        role={"current": 1.0, "previous": -0.5},
        slopes={(rc.VALUE, "current"): -0.3, (rc.VALUE, rc.REFERENCE_ROLE): 0.3},
    )
    sessions, reference = _simulate(big_cohort, truth, seed=0)
    fit = _fit(big_cohort, sessions, "role_content")
    pooled = _pooled(fit)
    planted = sc.planted_values(truth, reference)
    # Simulation tolerance: the SD of the pooled estimate over 8 replicate
    # draws (seeds 100 to 107) was 0.0036, 0.0067, 0.016, 0.019 and 0.026; the
    # model's pooled SE was 0.038, 0.063, 0.13, 0.19 and 0.18.
    spread = {"Q3a": 0.004, "Q3b": 0.007, "Q4b": 0.017, "Q4a": 0.02, "Q7b": 0.026}
    _assert_recovered(pooled, planted, spread)
    record = fit.records[0]
    beta = dict(zip(record.coef_names, record.coef, strict=True))
    assert beta["value:role=current"] < -0.1


def test_pooled_slope_hides_opposite_roles(big_cohort) -> None:
    """With opposite true slopes by role, one pooled slope is about 0."""
    truth = sc.Truth(
        arm=sc.arm_effects(),
        role={"current": 1.0},
        slopes={(rc.VALUE, "current"): -0.4, (rc.VALUE, rc.REFERENCE_ROLE): 0.4},
    )
    sessions, _ = _simulate(big_cohort, truth, seed=5)
    rows = pd.concat(sessions.values(), ignore_index=True)
    leaf = rows[rc.ROLE].isin(rc.LEAF_ROLES)
    rows["value_leaf"] = np.where(leaf, rows[rc.VALUE], 0.0)
    role = Category(rc.ROLE, rc.ROLES, reference=rc.REFERENCE_ROLE, penalty=0.0)
    pooled_terms = [
        Category(rc.ARM, tuple(range(9)), reference=0, penalty="none"),
        role,
        Slope("value_leaf", "none"),
    ]
    from continuum_swr_replay.analysis.regression.likelihoods import GroupedSoftmax
    from continuum_swr_replay.analysis.regression.sessions import fit_sessions

    design = build_design(pooled_terms, rows, intercept=False)
    records = fit_sessions(
        design, {("all", "0", "0"): rows}, GroupedSoftmax(), model="pooled",
        kind=KIND, response=rc.RESPONSE, unit=rc.UNIT, table_sha256={},
    )  # fmt: skip
    j = design.coef_names.index("value_leaf")
    beta, se = records[0].coef[j], np.sqrt(records[0].vcov[j][j])
    # The value covariate is standardized on all leaf rows; the role slopes
    # of the model recover both signs.
    assert abs(beta) < 0.15
    fit = _fit(big_cohort, sessions, "role_content")
    estimates = rs.session_estimands(fit)
    assert (estimates.loc[estimates.estimand_id == "Q4b", "estimate"] > 0.15).all()
    assert se > 0


def test_curves_are_recovered(big_cohort) -> None:
    truth = sc.Truth(
        arm=sc.arm_effects(),
        curves={
            (rc.REWARD_FRACTION, "current"): sc.reward_curve(1.0),
            (rc.LEAVING, "current"): sc.leaving_curve(1.0),
            (rc.RECENCY, rc.REFERENCE_ROLE): sc.recency_curve(1.0),
        },
    )
    sessions, reference = _simulate(big_cohort, truth, seed=1)
    planted = sc.planted_values(truth, reference)
    # The reward-fraction smooth at log penalty -4: at the placeholder 0 its
    # penalty shrinks this S-shaped curve toward a line, and Q4a came back at
    # 0.74 of its planted value (mean of 8 draws, seeds 100 to 107), at -4 at
    # 1.01 (the smooth penalties are open).
    settings = rc.RoleContentSettings(
        smooth_log_penalties={
            **rc.DEFAULT_SMOOTH_LOG_PENALTIES,
            rc.REWARD_FRACTION: -4.0,
        }
    )
    pooled = pd.concat(
        [_pooled(_fit(big_cohort, sessions, m, settings=settings)) for m in rc.MODELS]
    )
    pooled = pooled[~pooled.index.duplicated()]
    # Simulation tolerance: the SD over 8 replicate draws (seeds 100 to 107)
    # was 0.031, 0.035, 0.045, 0.0063 and 0.014.
    spread = {"Q4a": 0.031, "Q7b": 0.035, "Q5a": 0.045, "Q3a": 0.007, "Q5b-var": 0.014}
    _assert_recovered(pooled, planted, spread)
    assert planted["Q4a"] > 0.3 and planted["Q7b"] > 0.3 and planted["Q5a"] > 0.5


def test_belief_variance_and_sample_count_recovered_in_one_model(big_cohort) -> None:
    truth = sc.Truth(
        arm=sc.arm_effects(),
        curves={(rc.RECENCY, rc.REFERENCE_ROLE): sc.recency_curve(0.6)},
        slopes={(rc.BELIEF_VARIANCE, rc.REFERENCE_ROLE): 0.4,
                (rc.SAMPLE_COUNT, rc.REFERENCE_ROLE): -0.4},
    )  # fmt: skip
    sessions, reference = _simulate(big_cohort, truth, seed=2)
    pooled = _pooled(_fit(big_cohort, sessions, "role_content_uncertainty"))
    planted = sc.planted_values(truth, reference)
    # Simulation tolerance: the SD over 8 replicate draws (seeds 100 to 107)
    # was 0.013, 0.016 and 0.030.
    spread = {"Q5b-var": 0.014, "Q5b-n": 0.017, "Q5a": 0.031}
    _assert_recovered(pooled, planted, spread)


def test_length_adjusted_at_reference(big_cohort) -> None:
    """With a planted length interaction, the adjusted estimate is the truth
    at L0."""
    truth = sc.Truth(
        arm=sc.arm_effects(), role={"current": 0.8}, length={"current": 0.8}
    )
    sessions, reference = _simulate(big_cohort, truth, seed=4)
    pooled = _pooled(_fit(big_cohort, sessions, "role_content", "length_adjusted"))
    # Simulation tolerance: the SD over 8 replicate draws (seeds 100 to 107)
    # was 0.0042.
    _assert_recovered(pooled, {"Q3a": 0.8}, {"Q3a": 0.005})
    assert np.isnan(sc.planted_values(truth, reference)["Q3a"])
    assert sc.planted_values(truth, reference, "length_adjusted")["Q3a"] == 0.8


def test_a_thin_previous_level_is_dropped_with_its_rows(table) -> None:
    table = table.copy()
    session = (table["animal"] == "a1") & (table["session_date"] == 20200101)
    candidates = (
        session
        & (table["visit_context"] == "in_visit")
        & ~table["is_swr"]
        & table["prev_arm"].notna()
    )
    events = table.loc[candidates, rc.UNIT].unique()
    keep_previous = set(events[:3])
    thin = session & ~table[rc.UNIT].isin(keep_previous)
    table.loc[thin, "prev_arm"] = np.nan
    cohort = rc.role_content_cohort(table)
    rows = cohort.rows[cohort.rows[rc.KIND] == KIND]
    reference = sc.ScenarioReference(rows, cohort.reference_log_duration)
    truth = sc.Truth(arm=sc.arm_effects(), role={"current": 1.0})
    sessions = {
        key: draw_shares(part, sc.expected_shares(part, truth, reference), 40, i)
        for i, (key, part) in enumerate(rc.session_rows(rows).items())
    }
    fit = _fit(cohort, sessions, "role_content")
    by_key = {(r.animal, r.session_date, r.epoch): r for r in fit.records}
    thin_record = by_key[("a1", "20200101", "02_r1")]
    assert thin_record.status == "converged"
    assert thin_record.left_out["role[previous]"] == "support"
    assert thin_record.left_out["value:role=previous"] == "with_level"
    assert thin_record.cohort_counts["rows_dropped"] == {"role[previous]": 3}
    assert thin_record.support["role[previous]"]["n_high"] == 3
    estimates = rs.session_estimands(fit).set_index(["session_date", "estimand_id"])
    assert not estimates.loc[("20200101", "Q3b"), "kept"]
    assert estimates.loc[("20200101", "Q3a"), "kept"]
    other = by_key[("a1", "20200102", "02_r1")]
    assert "role[previous]" not in other.left_out


@pytest.mark.parametrize(
    "settings",
    [
        *(rc.RoleContentSettings(prior_sd=sd) for sd in rc.PRIOR_SD_CANDIDATES),
        rc.RoleContentSettings(
            smooth_log_penalties={
                "reward_fraction": 2.0,
                "log_trials_since_last_visit": -2.0,
                "time_until_visit_end_s": 4.0,
            }
        ),
        rc.RoleContentSettings(trial_number="completed_trial_id"),
        rc.RoleContentSettings(q4a_reading="full_range"),
        rc.RoleContentSettings(q5a_trials=(40.0, 5.0)),
    ],
    ids=lambda s: repr(s)[:60],
)
def test_each_open_setting_builds_and_fits(table, settings) -> None:
    table = table.assign(completed_trial_id=table["n_visits_arrived"] - 1)
    cohort = rc.role_content_cohort(table, settings)
    rows = cohort.rows[cohort.rows[rc.KIND] == KIND]
    reference = sc.ScenarioReference(rows, cohort.reference_log_duration)
    truth = sc.Truth(arm=sc.arm_effects(), role={"current": 1.0})
    sessions = {
        key: draw_shares(part, sc.expected_shares(part, truth, reference), 40, 0)
        for key, part in rc.session_rows(rows).items()
    }
    for model in rc.MODELS:
        fit = _fit(cohort, sessions, model, "length_adjusted", settings)
        assert all(r.status == "converged" for r in fit.records)
        estimates = rs.session_estimands(fit)
        assert estimates["kept"].all() and np.isfinite(estimates["estimate"]).all()
        assert set(estimates["estimand_id"]) == {
            e.estimand_id for e in rc.model_estimands(model)
        }


def test_open_settings_refuse_values_outside_their_candidates() -> None:
    with pytest.raises(ValueError, match="candidate width"):
        rc.RoleContentSettings(prior_sd=3.0)
    with pytest.raises(ValueError, match="trial_number"):
        rc.RoleContentSettings(trial_number="trial")
    with pytest.raises(ValueError, match="q4a_reading"):
        rc.RoleContentSettings(q4a_reading="slope")
    with pytest.raises(ValueError, match="smooth_log_penalties"):
        rc.RoleContentSettings(smooth_log_penalties={"reward_fraction": 0.0})


def test_records_carry_the_cohort_counts_and_tested_collinearity(cohort) -> None:
    truth = sc.Truth(arm=sc.arm_effects())
    sessions, _ = _simulate(cohort, truth, seed=9, n_bins=30)
    fit = rc.fit_role_content(
        sessions,
        model="role_content",
        variant="unadjusted",
        kind=KIND,
        reference_log_duration=cohort.reference_log_duration,
        cohort_counts=cohort.counts,
    )
    for record in fit.records:
        assert record.cohort_counts["cohort"]["by_kind"] == cohort.counts["by_kind"]
        assert record.likelihood == "grouped_softmax"
        assert record.model == "role_content:unadjusted"
        assert {"role", "s(reward_fraction):role=current"} <= set(record.collinearity)
    collinearity = next(iter(fit.tested_collinearity.values()))
    assert set(collinearity) == {
        "role[current]",
        "role[previous]",
        "value:role=other_leaf",
    }
    assert all(0 <= v["r2"] < 1 for v in collinearity.values())


def test_one_session_fit(cohort) -> None:
    sessions, _ = _simulate(cohort, sc.Truth(arm=sc.arm_effects()), seed=7)
    key, rows = next(iter(sessions.items()))
    fit = rc.fit_role_content_session(
        key, rows, model="role_content", variant="unadjusted", kind=KIND,
        reference_log_duration=cohort.reference_log_duration,
    )  # fmt: skip
    assert len(fit.records) == 1 and fit.records[0].status == "converged"


def test_a_session_of_another_kind_is_refused(cohort) -> None:
    rows = cohort.rows
    with pytest.raises(ValueError, match="another kind"):
        rc.fit_role_content(
            rc.session_rows(rows), model="role_content", variant="unadjusted",
            kind=KIND, reference_log_duration=cohort.reference_log_duration,
        )  # fmt: skip


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------


@pytest.mark.real_data
def test_one_real_session_of_each_kind_end_to_end() -> None:
    from continuum_swr_replay.paths import table_path
    from continuum_swr_replay.tables.io import read_table

    path = table_path("event_arm_table")
    keys = read_table(
        path, columns=["animal", "session_date", "epoch"]
    ).drop_duplicates()
    animal, date, epoch = keys.sort_values(list(keys.columns)).iloc[0]
    table = read_table(
        path,
        filters=[
            ("animal", "==", animal),
            ("session_date", "==", date),
            ("epoch", "==", epoch),
        ],
    )
    cohort = rc.role_content_cohort(table)
    rows = cohort.rows
    assert len(rows) % 9 == 0 and rows[rc.UNIT].nunique() * 9 == len(rows)
    for kind in ("SWR", "No SWR"):
        counts = cohort.counts["by_kind"][kind]
        assert (
            counts["events_kept"] + counts["events_dropped"]
            == counts["events_selected"]
        )
        of_kind = rc.session_rows(rows[rows[rc.KIND] == kind])
        assert len(of_kind) == 1
        for model in rc.MODELS:
            for variant in rc.VARIANTS:
                fit = rc.fit_role_content(
                    of_kind, model=model, variant=variant, kind=kind,
                    reference_log_duration=cohort.reference_log_duration,
                    cohort_counts=cohort.counts,
                )  # fmt: skip
                (record,) = fit.records
                assert record.status == "converged", record.no_fit
                assert record.n_units == counts[
                    "events_kept"
                ] - record.cohort_counts.get("events_without_response", 0)
                assert np.asarray(record.vcov).shape == (len(record.coef),) * 2
                estimates = rs.session_estimands(fit)
                assert len(estimates) == len(rc.model_estimands(model))
