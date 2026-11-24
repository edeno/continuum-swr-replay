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
from continuum_swr_replay.analysis.replay_models import declared_tests as dt
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
    """Each session's rows of KIND with drawn shares, and the reference of the
    scale rows (both kinds, the default pooling)."""
    rows = cohort.rows[cohort.rows[rc.KIND] == KIND]
    reference = sc.ScenarioReference(cohort.rows, cohort.reference_log_duration)
    sessions = {
        key: draw_shares(
            part, sc.expected_shares(part, truth, reference), n_bins, seed + i
        )
        for i, (key, part) in enumerate(rc.session_rows(rows).items())
    }
    return sessions, reference


def _fit(cohort, sessions, model, variant="unadjusted", settings=None, kind=KIND):
    return rc.fit_role_content(
        sessions,
        model=model,
        variant=variant,
        kind=kind,
        reference_log_duration=cohort.reference_log_duration,
        settings=settings,
        scale_rows=cohort.model_rows(model),
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


def _with_missing(table, column):
    """The table with ``column`` missing on one leaf row of one in-visit
    event; the event and its kind."""
    table = table.copy()
    leaf = table[rc.ARM] < 6
    event = table.loc[leaf & (table["visit_context"] == "in_visit"), rc.UNIT].iloc[0]
    kind = "SWR" if table.loc[table[rc.UNIT] == event, "is_swr"].iloc[0] else "No SWR"
    row = table.index[(table[rc.UNIT] == event) & leaf][0]
    table.loc[row, column] = np.nan
    return table, event, kind


def test_a_value_only_the_uncertainty_model_reads_drops_the_event_there_only(
    table,
) -> None:
    table, event, kind = _with_missing(
        table, "reward_belief_variance_candidate_arm_preoutcome"
    )
    cohort = rc.role_content_cohort(table)
    assert event in set(cohort.model_rows("role_content")[rc.UNIT])
    assert event not in set(cohort.model_rows("role_content_uncertainty")[rc.UNIT])
    plain = cohort.counts["by_model"]["role_content"][kind]
    assert plain["events_dropped"] == 0 and plain["rows_dropped"] == 0
    assert rc.BELIEF_VARIANCE not in plain["events_missing"]
    counts = cohort.counts["by_model"]["role_content_uncertainty"][kind]
    assert counts["events_dropped"] == 1
    assert counts["events_missing"][rc.BELIEF_VARIANCE] == 1
    assert counts["rows_dropped"] == 9
    assert counts["events_kept"] == counts["events_selected"] - 1
    assert plain["events_kept"] == counts["events_kept"] + 1
    # The model's rows are finite in every column it reads; the plain model's
    # rows keep the missing value, which it does not read.
    for model in rc.MODELS:
        values = cohort.model_rows(model)[list(rc.MODEL_COLUMNS[model])]
        assert np.isfinite(values.to_numpy(dtype=float)).all()
    assert cohort.model_rows("role_content")[rc.BELIEF_VARIANCE].isna().sum() == 1
    # Handing the uncertainty model rows with the missing value is refused.
    rows = cohort.rows[cohort.rows[rc.KIND] == kind]
    with pytest.raises(ValueError, match="missing values in a column"):
        _fit(cohort, rc.session_rows(rows), "role_content_uncertainty", kind=kind)


def test_a_value_both_models_read_drops_the_event_from_both(table) -> None:
    table, event, kind = _with_missing(
        table, "reward_belief_mean_candidate_arm_preoutcome"
    )
    cohort = rc.role_content_cohort(table)
    assert event not in set(cohort.rows[rc.UNIT])
    for model in rc.MODELS:
        counts = cohort.counts["by_model"][model][kind]
        assert counts["events_dropped"] == 1
        assert counts["events_missing"][rc.VALUE] == 1
        assert counts["rows_dropped"] == 9


def test_a_value_on_a_row_that_does_not_read_it_drops_nothing(table) -> None:
    # Trunk rows carry NaN beliefs on the real table; they are not read.
    assert (
        table.loc[table[rc.ARM] >= 6, "reward_belief_mean_candidate_arm_preoutcome"]
        .isna()
        .all()
    )
    cohort = rc.role_content_cohort(table)
    for by_kind in cohort.counts["by_model"].values():
        for counts in by_kind.values():
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
        rc.RoleContentSettings(scale_pooling="per_kind"),
        rc.RoleContentSettings(event_weight="n_nonlocal_bins"),
    ],
    ids=lambda s: repr(s)[:60],
)
def test_each_open_setting_builds_and_fits(table, settings) -> None:
    table = table.assign(
        completed_trial_id=table["n_visits_arrived"] - 1,
        n_nonlocal_bins=np.round(table["event_duration_s"] / 0.002),
    )
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
    with pytest.raises(ValueError, match="scale_pooling"):
        rc.RoleContentSettings(scale_pooling="pooled")
    with pytest.raises(ValueError, match="event_weight"):
        rc.RoleContentSettings(event_weight="n_bins")
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
        scale_rows=cohort.rows,
        cohort_counts=cohort.counts,
    )
    for record in fit.records:
        assert record.cohort_counts["cohort"]["by_model"] == cohort.counts["by_model"]
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


def _drawn(rows, truth, reference, n_bins=40, seed=0):
    return {
        key: draw_shares(
            part, sc.expected_shares(part, truth, reference), n_bins, seed + i
        )
        for i, (key, part) in enumerate(rc.session_rows(rows).items())
    }


@pytest.mark.parametrize("pooling", rc.SCALE_POOLINGS)
def test_scales_and_reference_points_by_the_pooling_setting(cohort, pooling) -> None:
    """Under ``both_kinds`` the two kinds' designs use one set of scales,
    constraints and reference points, those of the rows of both kinds; under
    ``per_kind`` each kind's are its own."""
    settings = rc.RoleContentSettings(scale_pooling=pooling)
    model, variant = "role_content_uncertainty", "length_adjusted"
    truth = sc.Truth(arm=sc.arm_effects(), role={"current": 1.0})
    fits = {}
    for kind in ("SWR", "No SWR"):
        rows = cohort.rows[cohort.rows[rc.KIND] == kind]
        reference = sc.ScenarioReference(rows, cohort.reference_log_duration)
        fits[kind] = _fit(
            cohort, _drawn(rows, truth, reference), model, variant, settings, kind
        )
    swr, no_swr = fits["SWR"], fits["No SWR"]
    if pooling == "both_kinds":
        expected = rc.ContrastReference.from_rows(
            cohort.rows, cohort.reference_log_duration, model
        )
        assert swr.reference == no_swr.reference == expected
        assert swr.design.standardization_record() == (
            no_swr.design.standardization_record()
        )
        assert swr.design.basis_record() == no_swr.design.basis_record()
        for eid, c in swr.contrasts.items():
            np.testing.assert_array_equal(c, no_swr.contrasts[eid])
    else:
        assert swr.reference.covariate_means != no_swr.reference.covariate_means
        assert swr.design.standardization_record() != (
            no_swr.design.standardization_record()
        )
    # L0 is pooled over both kinds under either option.
    assert swr.reference.reference_log_duration == cohort.reference_log_duration
    assert no_swr.reference.reference_log_duration == cohort.reference_log_duration


@pytest.fixture(scope="module")
def trunk_session():
    """One session's fit of the plain model (the reviewer's check)."""
    table = synthetic_event_arm_table(
        (SESSIONS[0],), n_visits=400, events_per_visit=3.0, swr_share=0.0, seed=5
    )
    cohort = rc.role_content_cohort(table)
    truth = sc.Truth(arm=sc.arm_effects(), role={"current": 0.5, "previous": -0.3})
    reference = sc.ScenarioReference(cohort.rows, cohort.reference_log_duration)
    sessions = _drawn(cohort.rows, truth, reference, n_bins=50, seed=1)
    return cohort, sessions, _fit(cohort, sessions, "role_content")


def test_trunk_level_is_fitted_at_zero_and_changes_no_tested_coefficient(
    trunk_session,
) -> None:
    from continuum_swr_replay.analysis.regression.likelihoods import GroupedSoftmax
    from continuum_swr_replay.analysis.regression.sessions import fit_sessions
    from continuum_swr_replay.analysis.regression.terms import (
        log_penalty_from_prior_sd,
    )

    cohort, sessions, fit = trunk_session
    (record,) = fit.records
    names = list(fit.design.coef_names)
    assert record.status == "converged"
    assert abs(record.coef[names.index(rc.TRUNK_LEVEL)]) < 1e-6
    # The same model with no trunk level: the trunk rows join the reference
    # level of the role category only (every term by role still reads role).
    weak = log_penalty_from_prior_sd(rc.DEFAULT_PRIOR_SD)
    terms = [
        Category("role_no_trunk", rc.LEAF_ROLES[::-1], reference=rc.REFERENCE_ROLE,
                 penalty=weak)
        if isinstance(t, Category) and t.column == rc.ROLE
        else t
        for t in rc.role_terms("role_content", "unadjusted")
    ]  # fmt: skip
    ((key, rows),) = sessions.items()
    rows = rows.assign(role_no_trunk=rows[rc.ROLE].replace("trunk", rc.REFERENCE_ROLE))
    scale = cohort.rows.assign(
        role_no_trunk=cohort.rows[rc.ROLE].replace("trunk", rc.REFERENCE_ROLE)
    )
    design = build_design(terms, scale, intercept=False)
    (without,) = fit_sessions(
        design, {key: rows}, GroupedSoftmax(), model="no_trunk", kind=KIND,
        response=rc.RESPONSE, unit=rc.UNIT, table_sha256={},
    )  # fmt: skip
    other = list(design.coef_names)
    tested = rc.tested_coefficients(fit.design, fit.contrasts)
    assert tested
    index = [names.index(n) for n in tested]
    index_without = [other.index(n.replace("role[", "role_no_trunk[")) for n in tested]
    np.testing.assert_allclose(
        np.asarray(record.coef)[index],
        np.asarray(without.coef)[index_without],
        rtol=1e-6, atol=1e-8,
    )  # fmt: skip
    vcov = np.asarray(record.vcov)[np.ix_(index, index)]
    vcov_without = np.asarray(without.vcov)[np.ix_(index_without, index_without)]
    np.testing.assert_allclose(vcov, vcov_without, rtol=1e-6, atol=1e-12)


def test_the_collinearity_rule_reads_the_tested_coefficients(trunk_session) -> None:
    from continuum_swr_replay.analysis.regression.collinearity import (
        MAX_EXPLAINED,
        separable,
    )

    _, _, fit = trunk_session
    (record,) = fit.records
    key = (record.animal, record.session_date, record.epoch)
    # The record reports the trunk level's exact collinearity with the arm
    # effects in the term's worst case, while the term's rule reads its
    # estimate, which passes: the record's term ``role`` is not the rule here.
    assert record.collinearity["role"]["worst"] > 1 - 1e-6
    assert separable(record.collinearity)["role"]
    tested = fit.tested_collinearity[key]
    assert set(fit.separable[key]) == set(tested) == {
        "role[current]", "role[previous]", "value:role=other_leaf"
    }  # fmt: skip
    assert rc.TRUNK_LEVEL not in fit.separable[key]
    assert fit.separable[key] == {n: m["r2"] < MAX_EXPLAINED for n, m in tested.items()}
    # The role levels are separable; on these synthetic rows the value slope
    # is not (the reward fraction and the belief mean share their outcomes).
    assert fit.separable[key]["role[current]"] and fit.separable[key]["role[previous]"]


def test_an_event_weight_counts_as_repeated_events(table) -> None:
    rng = np.random.default_rng(0)
    events = table[rc.UNIT].unique()
    weight = pd.Series(rng.integers(1, 4, events.size), index=events)
    table = table.assign(n_nonlocal_bins=table[rc.UNIT].map(weight).astype(float))
    weighted = rc.RoleContentSettings(event_weight="n_nonlocal_bins")
    cohort = rc.role_content_cohort(table, weighted)
    rows = cohort.rows[cohort.rows[rc.KIND] == KIND]
    key = SESSIONS[0]
    rows = rc.session_rows(rows)[key]
    truth = sc.Truth(arm=sc.arm_effects(), role={"current": 1.0})
    reference = sc.ScenarioReference(rows, cohort.reference_log_duration)
    rows = draw_shares(rows, sc.expected_shares(rows, truth, reference), 40, 3)
    assert set(rows[rc.EVENT_WEIGHT]) == {1.0, 2.0, 3.0}
    # Each event repeated as many times as its weight, each copy weight 1.
    copies = []
    for copy in range(3):
        part = rows[rows[rc.EVENT_WEIGHT] > copy]
        copies.append(part.assign(**{rc.UNIT: part[rc.UNIT] * 10 + copy}))
    repeated = pd.concat(copies, ignore_index=True).assign(**{rc.EVENT_WEIGHT: 1.0})
    fits = [
        rc.fit_role_content(
            {key: data},
            model="role_content",
            variant="unadjusted",
            kind=KIND,
            reference_log_duration=cohort.reference_log_duration,
            settings=settings,
            scale_rows=rows,
        )  # fmt: skip
        for data, settings in ((rows, weighted), (repeated, None))
    ]
    (a,), (b,) = (f.records for f in fits)
    assert a.n_units == rows[rc.UNIT].nunique()
    assert b.n_units == int(weight.loc[rows[rc.UNIT].unique()].sum())
    np.testing.assert_allclose(a.coef, b.coef, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(a.vcov, b.vcov, rtol=1e-6, atol=1e-12)


def test_event_weights_are_checked_and_a_missing_one_drops_the_event(table) -> None:
    settings = rc.RoleContentSettings(event_weight="n_nonlocal_bins")
    with pytest.raises(ValueError, match="n_nonlocal_bins"):
        rc.role_content_cohort(table, settings)
    bins = table.assign(n_nonlocal_bins=20.0)
    first = (
        bins[rc.UNIT] == bins.loc[bins["visit_context"] == "in_visit", rc.UNIT].iloc[0]
    )
    with pytest.raises(ValueError, match="the same on an event's nine rows"):
        rc.role_content_cohort(
            bins.assign(
                n_nonlocal_bins=np.where(first & (bins[rc.ARM] == 0), 3.0, 20.0)
            ),
            settings,
        )
    with pytest.raises(ValueError, match="positive"):
        rc.role_content_cohort(bins.assign(n_nonlocal_bins=0.0), settings)
    missing = bins.assign(n_nonlocal_bins=np.where(first, np.nan, 20.0))
    cohort = rc.role_content_cohort(missing, settings)
    assert not cohort.rows[rc.UNIT].isin(bins.loc[first, rc.UNIT]).any()
    kind = "SWR" if bins.loc[first, "is_swr"].iloc[0] else "No SWR"
    for model in rc.MODELS:
        counts = cohort.counts["by_model"][model][kind]
        assert counts["events_missing"][rc.EVENT_WEIGHT] == 1
    # One unit per event, the default, reads no column.
    assert (rc.role_content_cohort(table).rows[rc.EVENT_WEIGHT] == 1.0).all()


def test_penalized_edf_reports_the_shrinkage(cohort) -> None:
    truth = sc.Truth(arm=sc.arm_effects(), role={"current": 1.0, "previous": -0.5})
    sessions, _ = _simulate(cohort, truth, seed=12, n_bins=30)
    per = {
        sd: rs.penalized_edf(
            _fit(cohort, sessions, "role_content",
                 settings=rc.RoleContentSettings(prior_sd=sd))
        )
        for sd in (1.0, 8.0)
    }  # fmt: skip
    narrow, wide = per[1.0], per[8.0]
    assert set(narrow["coefficient"]) == {
        "role[current]", "role[previous]", "value:role=current",
        "value:role=previous", "value:role=other_leaf",
    }  # fmt: skip
    tested = narrow.set_index("coefficient")["tested"].groupby(level=0).first()
    assert tested["role[current]"] and tested["value:role=other_leaf"]
    assert not tested["value:role=current"]
    assert ((narrow["edf"] > 0) & (narrow["edf"] <= 1 + 1e-9)).all()
    # A narrower prior shrinks more: a lower edf for every coefficient.
    assert (narrow["edf"].to_numpy() < wide["edf"].to_numpy()).all()
    n_events = {
        (k[0], k[1], k[2]): part[rc.UNIT].nunique() for k, part in sessions.items()
    }
    for row in narrow.itertuples():
        assert row.n_events == n_events[(row.animal, row.session_date, row.epoch)]
    summary = rs.penalized_edf_summary(pd.concat([narrow, wide]))
    assert len(summary) == 2 * narrow["coefficient"].nunique()
    assert (summary["n_sessions"] == len(sessions)).all()
    assert (summary["edf_min"] <= summary["edf_median"]).all()


def test_one_session_fit(cohort) -> None:
    sessions, _ = _simulate(cohort, sc.Truth(arm=sc.arm_effects()), seed=7)
    key, rows = next(iter(sessions.items()))
    both = rc.session_rows(cohort.rows)[key]
    fit = rc.fit_role_content_session(
        key, rows, model="role_content", variant="unadjusted", kind=KIND,
        reference_log_duration=cohort.reference_log_duration, scale_rows=both,
    )  # fmt: skip
    assert len(fit.records) == 1 and fit.records[0].status == "converged"
    with pytest.raises(ValueError, match="needs scale_rows"):
        rc.fit_role_content_session(
            key, rows, model="role_content", variant="unadjusted", kind=KIND,
            reference_log_duration=cohort.reference_log_duration,
        )  # fmt: skip


def test_a_session_of_another_kind_is_refused(cohort) -> None:
    rows = cohort.rows
    with pytest.raises(ValueError, match="another kind"):
        rc.fit_role_content(
            rc.session_rows(rows), model="role_content", variant="unadjusted",
            kind=KIND, reference_log_duration=cohort.reference_log_duration,
        )  # fmt: skip


# ---------------------------------------------------------------------------
# The rows of the recency curve (decided: the other-leaf rows)
# ---------------------------------------------------------------------------


def _recency_fixed_case():
    """Three synthetic sessions of No SWR events with a current and a previous
    level and a recency curve planted, drawn with one seed."""
    table = synthetic_event_arm_table(SESSIONS, n_visits=150, swr_share=0.3, seed=3)
    truth = sc.Truth(
        arm=sc.arm_effects(),
        role={"current": 1.0, "previous": -0.5},
        curves={(rc.RECENCY, rc.REFERENCE_ROLE): sc.recency_curve(0.6)},
    )
    return table, truth


def test_the_recency_curve_is_on_the_other_leaf_rows_only() -> None:
    # The estimates of this fixed case from the fit on the other-leaf rows
    # (commit 66f49d00), to 1e-8; no setting chooses the rows any more.
    pinned = {
        ("role_content", "20200101", "Q3a"): 1.0046419949810848,
        ("role_content", "20200101", "Q3b"): -0.538315217883582,
        ("role_content", "20200102", "Q3a"): 1.0288708193328473,
        ("role_content", "20200102", "Q3b"): -0.5016731624117258,
        ("role_content", "20200103", "Q3a"): 0.9683642571771983,
        ("role_content", "20200103", "Q3b"): -0.468569118545361,
        ("role_content_uncertainty", "20200101", "Q5a"): 0.43670292665890204,
        ("role_content_uncertainty", "20200102", "Q5a"): 0.38477520910167545,
        ("role_content_uncertainty", "20200103", "Q5a"): 0.7331946525204704,
    }
    assert not hasattr(rc.RoleContentSettings(), "recency_rows")
    table, truth = _recency_fixed_case()
    cohort = rc.role_content_cohort(table)
    rows = cohort.rows[cohort.rows[rc.KIND] == KIND]
    reference = sc.ScenarioReference(cohort.rows, cohort.reference_log_duration)
    sessions = {
        key: draw_shares(part, sc.expected_shares(part, truth, reference), 40, 5)
        for key, part in rc.session_rows(rows).items()
    }
    for model in rc.MODELS:
        fit = _fit(cohort, sessions, model)
        recency = [t.name for t in fit.design.terms if t.covariate == rc.RECENCY]
        assert recency == [dt.estimand("Q5a").contrast.term]
        per = rs.session_estimands(fit).set_index(["session_date", "estimand_id"])[
            "estimate"
        ]
        for (m, date, eid), value in pinned.items():
            if m == model:
                assert per[(date, eid)] == pytest.approx(value, abs=1e-8)


def _within_event_null_space(design, rows):
    """The null space of the design centered within each event: the
    directions the grouped softmax cannot see. Shape ``(p, k)``."""
    X, _ = design.session(rc.design_rows(rows, "role_content", "unadjusted", None))
    groups = pd.factorize(rows[rc.UNIT])[0]
    means = np.zeros((groups.max() + 1, X.shape[1]))
    np.add.at(means, groups, X)
    means /= np.bincount(groups)[:, None]
    _, s, vt = np.linalg.svd(X - means[groups], full_matrices=False)
    return vt[s < 1e-9 * s[0]].T


def _identified(c, null):
    """True when the functional ``c`` is orthogonal to the null space: the
    likelihood fixes ``c' beta``."""
    return bool(np.abs(c @ null).max(initial=0.0) < 1e-8)


def test_the_role_levels_are_identified_beside_the_recency_curve(table) -> None:
    """On events whose other-leaf arms are all at least 3 trials since their
    last visit, the current and previous levels are identified by the
    unpenalized likelihood and Q5a is too; the trunk level is not."""
    model = "role_content"
    cohort = rc.role_content_cohort(table)
    rows = cohort.model_rows(model)
    rows = rows[rows[rc.KIND] == KIND]
    other = rows[rows[rc.ROLE] == rc.REFERENCE_ROLE]
    early = other.loc[other[rc.RECENCY] < np.log(3.0) - 1e-9, rc.UNIT]
    rows = rows[~rows[rc.UNIT].isin(early)].reset_index(drop=True)
    reference = rc.ContrastReference.from_rows(
        rows, cohort.reference_log_duration, model
    )
    design = rc.role_design(rows, model, "unadjusted", reference=reference)
    null = _within_event_null_space(design, rows)
    names = design.coef_names
    unit = np.eye(len(names))
    assert not _identified(unit[names.index(rc.TRUNK_LEVEL)], null)
    for role in ("current", "previous"):
        assert _identified(unit[names.index(f"role[{role}]")], null)
    term = dt.estimand("Q5a").contrast.term
    columns = [j for j, n in enumerate(names) if n.startswith(term + ".")]
    at = _basis_rows(design, term, np.log([20.0, 3.0]))
    q5a = np.zeros(len(names))
    q5a[columns] = at[0] - at[1]
    assert _identified(q5a, null)


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
    assert len(cohort.rows) % 9 == 0
    assert cohort.rows[rc.UNIT].nunique() * 9 == len(cohort.rows)
    for model in rc.MODELS:
        rows = cohort.model_rows(model)
        for kind in ("SWR", "No SWR"):
            counts = cohort.counts["by_model"][model][kind]
            assert (
                counts["events_kept"] + counts["events_dropped"]
                == counts["events_selected"]
            )
            assert (
                rows.loc[rows[rc.KIND] == kind, rc.UNIT].nunique()
                == (counts["events_kept"])
            )
            of_kind = rc.session_rows(rows[rows[rc.KIND] == kind])
            assert len(of_kind) == 1
            for variant in rc.VARIANTS:
                fit = rc.fit_role_content(
                    of_kind, model=model, variant=variant, kind=kind,
                    reference_log_duration=cohort.reference_log_duration,
                    scale_rows=rows, cohort_counts=cohort.counts,
                )  # fmt: skip
                (record,) = fit.records
                assert record.status == "converged", record.no_fit
                assert record.n_units == counts[
                    "events_kept"
                ] - record.cohort_counts.get("events_without_response", 0)
                assert np.asarray(record.vcov).shape == (len(record.coef),) * 2
                estimates = rs.session_estimands(fit)
                assert len(estimates) == len(rc.model_estimands(model))
