"""Design-column, standardization, and cohort contracts of the SWR-contrast content runner.

The runner's pure helpers are exercised directly on synthetic frames; ``main`` is
never executed (it reads the canonical tables). Three contracts are pinned:

* the event-length control (``z(log n_nonlocal_bins)`` crossed with the current /
  previous arm-or-patch indicator) equals the product of its two factors;
* the persisted standardization reference reproduces every ``<col>_z`` the fit used;
* the outcome strata partition the events whose current outcome is known, and events
  with an unknown outcome are excluded from both strata and counted by ``is_swr``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior import contrast_aggregation
from continuum_swr_replay.analysis.behavior.contrast_aggregation import (
    aggregate_contrast,
)
from continuum_swr_replay.analysis.behavior.model_specs import (
    SWR_CONTRAST_CONTENT_ARM_LADDER as ARM_LADDER,
)
from continuum_swr_replay.analysis.behavior.model_specs import (
    SWR_CONTRAST_CONTENT_PATCH_LADDER as PATCH_LADDER,
)
from continuum_swr_replay.tables.standardization import attach_standardized
from scripts.behavior.content import fit_swr_contrast_content as mod


def _z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - np.nanmean(values)) / np.nanstd(values)


@pytest.fixture
def arm_frame() -> pd.DataFrame:
    """Six events x three arms carrying every raw source the arm design needs."""
    rng = np.random.default_rng(0)
    n_events, n_arms = 6, 3
    events = np.repeat(np.arange(n_events), n_arms)
    arms = np.tile(np.arange(n_arms), n_events)
    frame = pd.DataFrame(
        {
            "event_id": events,
            "candidate_arm": arms,
            "n_nonlocal_bins": np.repeat([12, 45, 3, 1, 80, 25], n_arms),
            "time_since_stop_start_s": np.repeat(
                [0.5, 2.0, 7.5, 1.0, 3.25, 0.75], n_arms
            ),
            "is_current_arm": (arms == 0).astype(float),
            "is_prev_arm": (arms == 1).astype(float),
            "is_swr": np.repeat([True, False, True, False, True, False], n_arms),
            "is_periswitch_post": np.repeat([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], n_arms),
            "reward_belief_mean_candidate_arm_preoutcome": rng.normal(
                size=n_events * n_arms
            ),
            "reward_belief_variance_candidate_arm_preoutcome_z": rng.normal(
                size=n_events * n_arms
            ),
        }
    )
    for column in contrast_aggregation._ZSCALARS:
        frame[column] = np.repeat(rng.normal(size=n_events), n_arms)
    return frame


@pytest.fixture
def patch_frame() -> pd.DataFrame:
    """Six events x three patches carrying every raw source the patch design needs."""
    rng = np.random.default_rng(1)
    n_events, n_patches = 6, 3
    events = np.repeat(np.arange(n_events), n_patches)
    patches = np.tile(np.arange(n_patches), n_events)
    return pd.DataFrame(
        {
            "event_id": events,
            "patch": patches,
            "n_nonlocal_bins": np.repeat([9, 30, 4, 61, 15, 2], n_patches),
            "time_since_stop_start_s": np.repeat(
                [1.5, 0.25, 4.0, 2.5, 6.0, 0.5], n_patches
            ),
            "is_current_patch": (patches == 0).astype(float),
            "is_prev_patch": (patches == 2).astype(float),
            "is_swr": np.repeat([True, False, True, False, True, False], n_patches),
            "periswitch_index_trials": np.repeat(rng.normal(size=n_events), n_patches),
            "patch_switch_fraction_last5_completed_trips": np.repeat(
                rng.uniform(size=n_events), n_patches
            ),
        }
    )


def test_arm_length_products_are_the_standardized_log_length_times_the_indicator(
    arm_frame,
):
    designed, _reference = contrast_aggregation._attach_arm_design_columns(arm_frame)

    expected_z = _z(np.log(arm_frame["n_nonlocal_bins"].to_numpy(dtype=float)))
    np.testing.assert_allclose(
        designed["log_n_nonlocal_bins_z"].to_numpy(), expected_z, atol=1e-12
    )
    np.testing.assert_allclose(
        designed["log_n_nonlocal_bins_z_x_is_current_arm"].to_numpy(),
        expected_z * arm_frame["is_current_arm"].to_numpy(dtype=float),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        designed["log_n_nonlocal_bins_z_x_is_prev_arm"].to_numpy(),
        expected_z * arm_frame["is_prev_arm"].to_numpy(dtype=float),
        atol=1e-12,
    )
    # The control is only nonzero on the arm it modulates.
    assert (
        designed.loc[
            arm_frame["is_current_arm"] == 0, "log_n_nonlocal_bins_z_x_is_current_arm"
        ]
        == 0
    ).all()


def test_patch_length_products_are_the_standardized_log_length_times_the_indicator(
    patch_frame,
):
    designed, _reference = contrast_aggregation._attach_patch_design_columns(
        patch_frame
    )

    expected_z = _z(np.log(patch_frame["n_nonlocal_bins"].to_numpy(dtype=float)))
    np.testing.assert_allclose(
        designed["log_n_nonlocal_bins_z_x_is_current_patch"].to_numpy(),
        expected_z * patch_frame["is_current_patch"].to_numpy(dtype=float),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        designed["log_n_nonlocal_bins_z_x_is_prev_patch"].to_numpy(),
        expected_z * patch_frame["is_prev_patch"].to_numpy(dtype=float),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        designed["log1p_time_since_stop_start_s_z_x_is_current_patch"].to_numpy(),
        _z(np.log1p(patch_frame["time_since_stop_start_s"].to_numpy(dtype=float)))
        * patch_frame["is_current_patch"].to_numpy(dtype=float),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("helper", "frame_name", "sources"),
    [
        (
            "_attach_arm_design_columns",
            "arm_frame",
            contrast_aggregation._ARM_Z_SOURCES,
        ),
        (
            "_attach_patch_design_columns",
            "patch_frame",
            contrast_aggregation._PATCH_Z_SOURCES,
        ),
    ],
)
def test_persisted_reference_reproduces_every_standardized_column(
    helper, frame_name, sources, request
):
    frame = request.getfixturevalue(frame_name)
    designed, reference = getattr(contrast_aggregation, helper)(frame.copy())

    assert set(reference) == set(sources)
    for source in sources:
        assert reference[source]["support"] == "fit_cohort_all_rows"
    replayed, _ = attach_standardized(
        designed.loc[:, list(sources)].copy(),
        list(sources),
        reference=reference,
        support_label="fit_cohort_all_rows",
    )
    for source in sources:
        np.testing.assert_array_equal(
            replayed[f"{source}_z"].to_numpy(), designed[f"{source}_z"].to_numpy()
        )


@pytest.mark.parametrize("bad", [0, -1])
def test_event_length_below_one_bin_is_refused(arm_frame, bad):
    arm_frame.loc[3, "n_nonlocal_bins"] = bad
    with pytest.raises(ValueError, match="n_nonlocal_bins"):
        contrast_aggregation._attach_arm_design_columns(arm_frame)


def test_event_length_missing_is_refused(arm_frame):
    arm_frame.loc[3, "n_nonlocal_bins"] = np.nan
    with pytest.raises(ValueError, match="n_nonlocal_bins"):
        contrast_aggregation._attach_arm_design_columns(arm_frame)


@pytest.fixture
def outcome_frame() -> pd.DataFrame:
    """Five events x two arms: two rewarded, two omitted, one unknown outcome."""
    return pd.DataFrame(
        {
            "event_id": np.repeat([10, 11, 12, 13, 14], 2),
            "candidate_arm": np.tile([0, 1], 5),
            "is_rewarded_outcome": np.repeat([1.0, 0.0, 1.0, np.nan, 0.0], 2),
            "is_swr": np.repeat([True, False, True, True, False], 2),
        }
    )


def test_outcome_strata_partition_the_known_outcome_events(outcome_frame):
    rewarded, _ = contrast_aggregation.select_outcome_stratum(outcome_frame, "rewarded")
    omitted, _ = contrast_aggregation.select_outcome_stratum(outcome_frame, "omitted")
    everything, _ = contrast_aggregation.select_outcome_stratum(outcome_frame, "all")

    rewarded_events = set(rewarded["event_id"])
    omitted_events = set(omitted["event_id"])
    assert rewarded_events == {10, 12}
    assert omitted_events == {11, 14}
    assert not rewarded_events & omitted_events
    # The unknown-outcome event is in neither stratum but survives "all".
    assert rewarded_events | omitted_events == {10, 11, 12, 14}
    assert set(everything["event_id"]) == {10, 11, 12, 13, 14}
    assert len(rewarded) == 4
    assert len(everything) == len(outcome_frame)


def test_outcome_stratum_audit_counts_unknown_outcome_by_swr(outcome_frame):
    _, audit = contrast_aggregation.select_outcome_stratum(outcome_frame, "rewarded")

    assert audit["stratum"] == "rewarded"
    assert audit["n_events_in"] == 5
    assert audit["n_events_kept"] == 2
    assert audit["n_events_unknown_outcome"] == 1
    assert audit["n_events_unknown_outcome_swr"] == 1
    assert audit["n_events_unknown_outcome_non_swr"] == 0

    _, all_audit = contrast_aggregation.select_outcome_stratum(outcome_frame, "all")
    assert all_audit["n_events_kept"] == 5
    assert all_audit["n_events_unknown_outcome"] == 1


def test_unknown_outcome_stratum_is_refused(outcome_frame):
    with pytest.raises(ValueError, match="outcome stratum"):
        contrast_aggregation.select_outcome_stratum(outcome_frame, "rewarded_only")


@pytest.mark.parametrize(
    ("max_per_animal", "estimator", "response", "stratum", "length", "expected"),
    [
        (None, "conditional_poisson", "fraction", "all", False, ""),
        (None, "conditional_poisson", "fraction", "rewarded", False, "_rewarded"),
        (None, "conditional_poisson", "fraction", "omitted", False, "_omitted"),
        (
            None,
            "conditional_poisson",
            "hard_count",
            "rewarded",
            False,
            "_hard_count_rewarded",
        ),
        (
            1,
            "conditional_poisson",
            "hard_count",
            "omitted",
            False,
            "_smoke_hard_count_omitted",
        ),
        # The Dirichlet-multinomial has only one response, so naming it would say
        # the same thing twice; ``_dm`` comes before the stratum.
        (None, "dirichlet_multinomial", "hard_count", "all", False, "_dm"),
        (None, "dirichlet_multinomial", "hard_count", "omitted", False, "_dm_omitted"),
        (
            1,
            "dirichlet_multinomial",
            "hard_count",
            "rewarded",
            False,
            "_smoke_dm_rewarded",
        ),
        # The length-adjusted concentration is a different estimand of
        # log_phi_is_swr, so it gets its own artifact name.
        (None, "dirichlet_multinomial", "hard_count", "all", True, "_dm_len"),
        (
            None,
            "dirichlet_multinomial",
            "hard_count",
            "omitted",
            True,
            "_dm_len_omitted",
        ),
    ],
)
def test_artifact_suffix_records_smoke_estimator_response_and_stratum(
    max_per_animal, estimator, response, stratum, length, expected
):
    import argparse

    args = argparse.Namespace(
        max_per_animal=max_per_animal,
        estimator=estimator,
        outcome_stratum=stratum,
        concentration_length=length,
    )
    assert mod._artifact_suffix(args, response) == expected


# --- Cohort-wide per-SD scales -------------------------------------------------


def _outcome_split_cohorts() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Arm/patch cohorts whose rewarded and omitted events have different spreads."""
    rewarded_value = [0.50, 0.52, 0.48, 0.51]  # tight
    omitted_value = [0.10, 0.90, 0.20, 0.80]  # wide
    outcome = [1.0] * 4 + [0.0] * 4 + [np.nan]
    value = rewarded_value + omitted_value + [0.5]
    n_events, n_arms = 9, 2
    ev = pd.DataFrame(
        {
            "animal": ["a"] * n_events * n_arms,
            "session_date": [1] * n_events * n_arms,
            "epoch": ["01_r1"] * n_events * n_arms,
            "event_id": np.repeat(np.arange(n_events), n_arms),
            "candidate_arm": np.tile([0, 1], n_events),
            "is_swr": np.repeat([True, False] * 5, n_arms)[: n_events * n_arms],
            "is_rewarded_outcome": np.repeat(outcome, n_arms),
            "reward_belief_mean_candidate_arm_preoutcome": np.repeat(value, n_arms),
            "reward_fraction_candidate_arm_last10_visits": np.repeat(value, n_arms),
        }
    )
    pt = ev.loc[:, ["animal", "session_date", "epoch", "event_id"]].copy()
    return ev, pt


@pytest.mark.parametrize("stratum", ["all", "rewarded", "omitted"])
def test_per_sd_scale_is_frozen_over_the_full_cohort(stratum):
    """A stratified refit must report per-SD effects in the pooled cohort's unit."""
    ev, pt = _outcome_split_cohorts()
    full_sd = float(
        np.nanstd(ev["reward_belief_mean_candidate_arm_preoutcome"].to_numpy(float))
    )

    selected, _selected_pt, scale, _audit = contrast_aggregation.prepare_cohorts(
        ev.copy(), pt.copy(), outcome_stratum=stratum
    )

    assert scale["arm_cohort_support"] == "leaf_arms"
    assert scale["patch_cohort_support"] == "fit_cohort_all_rows"
    assert set(scale["arm_cohort"]) == {
        "reward_belief_mean_candidate_arm_preoutcome",
        "reward_fraction_candidate_arm_last10_visits",
    }
    assert scale["arm_cohort"]["reward_belief_mean_candidate_arm_preoutcome"] == full_sd
    assert scale["patch_cohort"] == {}
    # The stratum really does have a different spread -- so computing the scale
    # after the filter (the bug this pins) would have changed the unit.
    stratum_sd = float(
        np.nanstd(
            selected["reward_belief_mean_candidate_arm_preoutcome"].to_numpy(float)
        )
    )
    if stratum != "all":
        assert stratum_sd != pytest.approx(full_sd)


def test_per_sd_scale_survives_the_smoke_subset():
    ev, pt = _outcome_split_cohorts()
    _, _, full_scale, _ = contrast_aggregation.prepare_cohorts(
        ev.copy(), pt.copy(), outcome_stratum="all"
    )
    smoke_ev, _, smoke_scale, _ = contrast_aggregation.prepare_cohorts(
        ev.copy(), pt.copy(), outcome_stratum="rewarded", max_per_animal=1
    )

    assert smoke_scale == full_scale
    assert set(smoke_ev["event_id"]) == {0, 1, 2, 3}


def test_per_sd_scale_ignores_trunk_arms():
    """The per-SD unit is the spread on the leaf arms the rescaled rungs are fit on.

    Every raw per-arm covariate the aggregation rescales is a structural zero on the
    three trunk arms, so trunk rows contribute a spread the fitted rows never see.
    """
    ev, pt = _outcome_split_cohorts()
    leaf_sd = float(
        np.nanstd(ev["reward_belief_mean_candidate_arm_preoutcome"].to_numpy(float))
    )
    # Same events, but each also carries a trunk row with a wildly different value.
    trunk = ev.copy()
    trunk["candidate_arm"] = 6
    trunk["reward_belief_mean_candidate_arm_preoutcome"] = 100.0
    trunk["reward_fraction_candidate_arm_last10_visits"] = 100.0
    with_trunks = pd.concat([ev, trunk], ignore_index=True)
    all_rows_sd = float(
        np.nanstd(
            with_trunks["reward_belief_mean_candidate_arm_preoutcome"].to_numpy(float)
        )
    )
    assert all_rows_sd > 10 * leaf_sd

    _, _, scale, _ = contrast_aggregation.prepare_cohorts(
        with_trunks, pt.copy(), outcome_stratum="all"
    )

    assert scale["arm_cohort_support"] == "leaf_arms"
    assert scale["arm_cohort"][
        "reward_belief_mean_candidate_arm_preoutcome"
    ] == pytest.approx(leaf_sd)
    assert scale["arm_cohort"][
        "reward_fraction_candidate_arm_last10_visits"
    ] == pytest.approx(leaf_sd)


def test_provenance_records_the_frozen_per_sd_scale():
    import argparse

    ev, pt = _outcome_split_cohorts()
    _, _, scale, audit = contrast_aggregation.prepare_cohorts(
        ev, pt, outcome_stratum="rewarded"
    )
    prov = mod._provenance(
        argparse.Namespace(
            max_per_animal=None,
            response="fraction",
            estimator="conditional_poisson",
            min_events_session_phi=40,
            outcome_stratum="rewarded",
        ),
        n_sessions=1,
        outcome_stratum_audit=audit,
        per_sd_scale=scale,
    )

    assert prov["per_sd_scale"] == scale
    assert prov["outcome_stratum"] == "rewarded"
    assert prov["outcome_stratum_audit"]["n_events_kept"] == 4


# --- Ladder/design name agreement ---------------------------------------------

# The interaction products the runner materializes. Spelled out here independently
# of both the assemblers and the ladders, so a rename on either side fails: the
# runner must build every one of these, and the ladders must reference only these
# plus columns the canonical tables supply.
_RUNNER_BUILT_ARM_PRODUCTS = (
    "rpe_postoutcome_z_x_is_current_arm",
    "relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm",
    "reward_belief_differential_entropy_leaf_mean_preoutcome_z_x_is_current_arm",
    "reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm",
    "periswitch_index_trials_z_x_is_current_arm",
    "is_periswitch_post_x_is_current_arm",
    "periswitch_index_trials_z_x_is_periswitch_post_x_is_current_arm",
    "trials_since_block_change_z_x_is_current_arm",
    "patch_switch_fraction_last5_completed_trips_z_x_is_current_arm",
    "log1p_time_since_stop_start_s_z_x_is_current_arm",
    "log_n_nonlocal_bins_z_x_is_current_arm",
    "log_n_nonlocal_bins_z_x_is_prev_arm",
    "swr_x_reward_belief_mean_candidate_arm_preoutcome",
    "swr_x_reward_belief_variance_candidate_arm_preoutcome_z",
)
_RUNNER_BUILT_PATCH_PRODUCTS = (
    "periswitch_index_trials_z_x_is_current_patch",
    "patch_switch_fraction_last5_completed_trips_z_x_is_current_patch",
    "log1p_time_since_stop_start_s_z_x_is_current_patch",
    "log_n_nonlocal_bins_z_x_is_current_patch",
    "log_n_nonlocal_bins_z_x_is_prev_patch",
    "swr_x_is_current_patch",
)


def _ladder_predictors(ladder: dict) -> set[str]:
    return {p for preds, _scope in ladder.values() for p in preds}


@pytest.mark.parametrize(
    ("helper", "frame_name", "ladder", "built"),
    [
        (
            "_attach_arm_design_columns",
            "arm_frame",
            ARM_LADDER,
            _RUNNER_BUILT_ARM_PRODUCTS,
        ),
        (
            "_attach_patch_design_columns",
            "patch_frame",
            PATCH_LADDER,
            _RUNNER_BUILT_PATCH_PRODUCTS,
        ),
    ],
)
def test_every_ladder_predictor_reaches_the_assembled_design(
    helper, frame_name, ladder, built, request
):
    """A product renamed on one side would empty a rung through a silent skip."""
    frame = request.getfixturevalue(frame_name).copy()
    predictors = _ladder_predictors(ladder)
    # Seed only what a canonical table supplies; the runner must build the rest.
    for column in sorted(predictors - set(built)):
        if column not in frame.columns:
            frame[column] = 0.0
    assert not set(built) & set(frame.columns), "a product was seeded, not built"

    designed, _reference = getattr(contrast_aggregation, helper)(frame)

    assert set(built) <= set(designed.columns)
    missing = predictors - set(designed.columns)
    assert not missing, f"ladder predictors absent from the design: {sorted(missing)}"


# --- Leaf-only rungs: response support ----------------------------------------


def _nine_arm_cohort() -> pd.DataFrame:
    """Two events x nine arms; each spends a third of its mass on a trunk arm."""
    rows = []
    for index, event in enumerate(("a", "b")):
        counts = np.zeros(9)
        counts[index] = 6.0
        counts[index + 1] = 2.0
        counts[6] = 4.0
        for arm in range(9):
            rows.append(
                {
                    "animal": "chimi",
                    "session_date": "20200212",
                    "epoch": "02_r1",
                    "event_id": event,
                    "candidate_arm": arm,
                    "arm_fraction": counts[arm] / counts.sum(),
                    "is_swr": index == 0,
                    "reward_belief_mean_candidate_arm_preoutcome_z": float(arm),
                }
            )
    return pd.DataFrame(rows)


def _fit_row() -> dict:
    return {
        "coef_reward_belief_mean_candidate_arm_preoutcome_z": 0.1,
        "bse_reward_belief_mean_candidate_arm_preoutcome_z": 0.05,
        "sd_reward_belief_mean_candidate_arm_preoutcome_z": 1.0,
        "pval_reward_belief_mean_candidate_arm_preoutcome_z": 0.5,
        "separated_terms": "",
        "n_events": 2,
        "n_params": 1,
        "estimator": "conditional_poisson_fraction",
        "converged": True,
        "separated": False,
        "n_convergence_warnings": 0,
    }


def test_fit_rows_carry_the_optimizer_diagnostics(monkeypatch):
    """The refit bundle records whether each fit reached its optimum, and which
    terms its session could not inform, so its gates can check them."""
    diagnostics = {
        "llf": -12.5,
        "newton_decrement": 3e-12,
        "max_abs_grad": 1e-7,
        "n_newton_steps": 1,
        "unidentified_terms": "is_candidate_arm_0:separated_to_minus_inf",
    }

    def _fit(g, predictors, *, response_col, n_categories):
        fit = {**_fit_row(), **diagnostics}
        return fit, mod._audit(2, 0, 0, 0, 0, 2, "", g.iloc[0:0])

    monkeypatch.setattr(mod, "_fit_session", _fit)
    term = "reward_belief_mean_candidate_arm_preoutcome_z"

    res, _audits = mod._run(
        _nine_arm_cohort(),
        {"C1_nine_arm": ([term], "pooled")},
        response_col="arm_fraction",
        n_categories=9,
    )

    row = res.iloc[0]
    for column, value in diagnostics.items():
        assert row[column] == value
    # The existing columns are kept.
    assert row[f"coef_{term}"] == pytest.approx(0.1)
    assert bool(row["converged"]) is True


def test_leaf_only_rungs_are_fit_on_leaf_renormalized_fractions(monkeypatch):
    """The trunk share is not balanced across the contrast, so it cannot set the
    weight an event carries: leaf-only rungs get unit mass over the six leaves,
    while the nine-arm rungs in the same run keep the table's own fractions."""
    seen = []

    def _capture(g, predictors, *, response_col, n_categories):
        seen.append((n_categories, g.copy()))
        return _fit_row(), mod._audit(2, 0, 0, 0, 0, 2, "", g.iloc[0:0])

    monkeypatch.setattr(mod, "_fit_session", _capture)
    term = "reward_belief_mean_candidate_arm_preoutcome_z"
    ladder = {
        "C4_armvalue": ([term], "pooled"),
        "C1_nine_arm": ([term], "pooled"),
    }

    res, _audits = mod._run(
        _nine_arm_cohort(), ladder, response_col="arm_fraction", n_categories=9
    )

    fitted = dict(seen)
    assert set(fitted) == {6, 9}
    leaf = fitted[6]
    assert sorted(leaf["candidate_arm"].unique()) == [0, 1, 2, 3, 4, 5]
    assert np.allclose(
        leaf.groupby("event_id")["arm_fraction"].sum().to_numpy(), 1.0, atol=1e-12
    )
    # Event "a" held 6/12 and 2/12 on leaves and 4/12 on a trunk.
    a_leaf = leaf[leaf["event_id"].eq("a")].set_index("candidate_arm")["arm_fraction"]
    assert a_leaf[0] == pytest.approx(0.75)
    assert a_leaf[1] == pytest.approx(0.25)

    nine = fitted[9]
    assert sorted(nine["candidate_arm"].unique()) == list(range(9))
    a_nine = nine[nine["event_id"].eq("a")].set_index("candidate_arm")["arm_fraction"]
    assert a_nine[0] == pytest.approx(0.5)
    assert a_nine[6] == pytest.approx(1 / 3)

    renormalized = res.set_index("model")["response_renormalized"]
    assert bool(renormalized["C4_armvalue"]) is True
    assert bool(renormalized["C1_nine_arm"]) is False


def test_leaf_only_count_rungs_keep_their_timebin_counts(monkeypatch):
    """A count response is already weighted by decoded timebins; leave it alone."""
    seen = []

    def _capture(g, predictors, *, response_col, n_categories):
        seen.append(g.copy())
        return None, mod._audit(2, 0, 0, 0, 2, 0, "too_few", g.iloc[0:0])

    monkeypatch.setattr(mod, "_fit_session", _capture)
    cohort = _nine_arm_cohort()
    cohort["arm_count_hard"] = cohort["arm_fraction"] * 12.0
    term = "reward_belief_mean_candidate_arm_preoutcome_z"

    mod._run(
        cohort,
        {"C4_armvalue": ([term], "pooled")},
        response_col="arm_count_hard",
        n_categories=9,
    )

    leaf = seen[0]
    expected = cohort[cohort["candidate_arm"] < 6]["arm_count_hard"].to_numpy()
    assert np.array_equal(leaf["arm_count_hard"].to_numpy(), expected)


def test_provenance_names_only_the_rescaled_rungs_that_produced_rows():
    """A rung nothing was fitted for cannot be claimed as rescaled."""
    import argparse

    fitted = pd.DataFrame(
        {
            "model": ["C4_armvalue", "C4_armvalue", "C1_nine_arm"],
            "response_renormalized": [True, True, False],
        }
    )
    empty = pd.DataFrame(columns=["model", "response_renormalized"])

    assert mod._renormalized_rungs(fitted, empty) == ["C4_armvalue"]
    assert mod._renormalized_rungs(empty) == []
    # A count run rescales nothing, so nothing is named.
    counts = fitted.assign(response_renormalized=False)
    assert mod._renormalized_rungs(counts) == []

    args = argparse.Namespace(
        max_per_animal=None,
        response="fraction",
        estimator="conditional_poisson",
        min_events_session_phi=40,
        outcome_stratum="all",
    )
    prov = mod._provenance(
        args, n_sessions=1, renormalized_rungs=mod._renormalized_rungs(fitted)
    )
    assert prov["response_renormalized_rungs"] == ["C4_armvalue"]
    assert mod._provenance(args, n_sessions=1)["response_renormalized_rungs"] == []


def _provenance_args():
    import argparse

    return argparse.Namespace(
        max_per_animal=None,
        response="fraction",
        estimator="conditional_poisson",
        min_events_session_phi=40,
        outcome_stratum="all",
    )


def test_provenance_records_the_commit_this_process_loaded(monkeypatch):
    """The audited record names the code that ran, as the fit sidecars do.

    A content refit runs for hours in a checkout that other work commits in, so
    the HEAD at write time can be newer than the code that ran, and a bare
    ``git rev-parse HEAD`` drops the ``-dirty`` marker of an edited tree.
    """
    import subprocess

    loaded = "0123456789abcdef0123456789abcdef01234567-dirty"
    monkeypatch.setattr(mod, "loaded_commit", lambda: loaded)

    def read_at_write_time(*args, **kwargs):
        raise AssertionError("_provenance read the checkout at write time")

    monkeypatch.setattr(subprocess, "run", read_at_write_time)

    assert mod._provenance(_provenance_args(), n_sessions=1)["git_hash"] == loaded


def _leaf_session_with_a_trunk_only_event(n_events: int = 20) -> pd.DataFrame:
    """One session; event ``e0`` spends its whole allocation on the trunks."""
    rows = []
    for index in range(n_events):
        event = f"e{index}"
        current = index % 6
        counts = np.zeros(9)
        if index == 0:
            counts[6] = 8.0
        else:
            counts[current] = 6.0
            counts[(current + 1) % 6] = 2.0
        for arm in range(9):
            rows.append(
                {
                    "animal": "chimi",
                    "session_date": "20200212",
                    "epoch": "02_r1",
                    "event_id": event,
                    "candidate_arm": arm,
                    "arm_fraction": counts[arm] / counts.sum(),
                    "is_current_arm": float(arm == current),
                }
            )
    return pd.DataFrame(rows)


def test_event_left_without_leaf_mass_is_audited_as_a_response_drop():
    """Its response is undefined, not its predictors; the mechanism must say so."""
    session = _leaf_session_with_a_trunk_only_event()
    leaves = mod.renormalize_fraction_over_candidates(
        session[session["candidate_arm"] < 6], response_col="arm_fraction"
    )

    result, audit = mod._fit_session(
        leaves, ["is_current_arm"], response_col="arm_fraction", n_categories=6
    )

    assert audit["n_drop_zero_response"] == 1
    assert audit["n_drop_own_predictor_missing"] == 0
    assert audit["n_drop_qvals_gap"] == 0
    assert audit["n_modeled"] == 19
    assert result is not None
    drops = (
        audit["n_drop_qvals_gap"]
        + audit["n_drop_own_predictor_missing"]
        + audit["n_drop_zero_response"]
        + audit["n_excluded_session"]
    )
    assert audit["n_in"] - drops == audit["n_modeled"]


# --- Dirichlet-multinomial estimator -------------------------------------------


def _dm_contrast_cohort(n_events: int = 60, *, epoch: str, seed: int) -> pd.DataFrame:
    """One session of nine-arm counts drawn at a known within-event concentration."""
    from continuum_swr_replay.analysis.behavior.content.simulation import (
        simulate_dirichlet_multinomial_counts,
    )

    df = simulate_dirichlet_multinomial_counts(
        {"is_current_arm": 0.8},
        log_phi_intercept=1.2,
        log_phi_is_swr=-0.8,
        n_events=n_events,
        L_e=20,
        seed=seed,
    )
    df["animal"] = "chimi"
    df["session_date"] = "20200212"
    df["epoch"] = epoch
    df["event_id"] = [f"{epoch}-{index}" for index in df["event_id"]]
    df["is_swr"] = df["is_swr"].astype(bool)
    df.attrs.clear()
    return df


def test_dm_run_reports_a_concentration_contrast_on_every_rung():
    """Every rung also answers "does an SWR event spread its mass more widely"."""
    cohort = pd.concat(
        [
            _dm_contrast_cohort(epoch="02_r1", seed=0),
            _dm_contrast_cohort(epoch="04_r2", seed=1),
        ],
        ignore_index=True,
    )
    ladder = {
        "C0_base": (["is_current_arm"], "pooled"),
        "C1_split": (["is_current_arm"], "split"),
    }

    res, audits = mod._run(
        cohort,
        ladder,
        response_col="arm_count_hard",
        n_categories=9,
        estimator="dirichlet_multinomial",
        n_jobs=2,
        min_events_session_phi=40,
    )

    assert set(res["model"]) == {"C0_base", "C1_split"}
    assert res["estimator"].unique().tolist() == ["dirichlet_multinomial"]
    assert bool(res["llf_includes_constants"].all())
    assert res["phi_source"].unique().tolist() == ["session"]

    # The events behind the modifier travel with it: a contrast resting on one
    # or two SWR events is finite and unflagged, and nothing else would show it.
    pooled_counts = res[res["population"] == "all"]
    assert (pooled_counts["n_events_swr"] > 0).all()
    assert (
        pooled_counts["n_events_swr"] + pooled_counts["n_events_nonswr"]
        == pooled_counts["n_events"]
    ).all()
    assert res.loc[res["population"] == "SWR", "n_events_swr"].isna().all()

    pooled = res[res["population"] == "all"]
    # The pooled rows carry the contrast; both kinds of event are present there.
    assert pooled["coef_log_phi_is_swr"].notna().all()
    assert pooled["bse_log_phi_is_swr"].notna().all()
    # Generated at log_phi_is_swr = -0.8: SWR events are the more concentrated.
    assert pooled["coef_log_phi_is_swr"].median() < 0.0

    # Within an SWR split the flag is constant, so there is no contrast to fit.
    split = res[res["population"] == "SWR"]
    assert len(split) == 2
    assert split["coef_log_phi_is_swr"].isna().all()

    assert {record["model"] for record in audits} == {"C0_base", "C1_split"}


def _with_event_length(cohort: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Attach a z-scored log event length that is constant within each event."""
    rng = np.random.default_rng(seed)
    events = sorted(cohort["event_id"].unique())
    # SWR events run longer, which is the whole reason the adjustment exists.
    is_swr = (
        cohort.drop_duplicates("event_id").set_index("event_id")["is_swr"].astype(bool)
    )
    values = pd.Series(
        rng.normal(size=len(events)) + 0.6 * is_swr.loc[events].to_numpy(),
        index=events,
    )
    values = (values - values.mean()) / values.std(ddof=0)
    return cohort.assign(log_n_nonlocal_bins_z=cohort["event_id"].map(values))


class TestLengthAdjustedConcentration:
    """``--concentration-length`` fits the length-adjusted twin of log_phi_is_swr."""

    def test_the_length_term_joins_the_design_only_where_is_swr_is_identified(self):
        cohort = _with_event_length(_dm_contrast_cohort(epoch="02_r1", seed=0), seed=0)

        assert mod.resolve_concentration_cols(cohort, with_event_length=False) == (
            "is_swr",
        )
        assert mod.resolve_concentration_cols(cohort, with_event_length=True) == (
            "is_swr",
            "log_n_nonlocal_bins_z",
        )
        # Inside an SWR split there is no concentration contrast to decompose.
        swr_only = cohort[cohort["is_swr"]]
        assert mod.resolve_concentration_cols(swr_only, with_event_length=True) == ()

    def test_a_cohort_without_the_length_column_is_refused(self):
        """A silently dropped term would publish ``_dm_len`` fits that are not."""
        cohort = _dm_contrast_cohort(epoch="02_r1", seed=0)
        with pytest.raises(ValueError, match="log_n_nonlocal_bins_z"):
            mod.resolve_concentration_cols(cohort, with_event_length=True)

    def test_the_run_reports_both_concentration_terms(self):
        cohort = pd.concat(
            [
                _with_event_length(_dm_contrast_cohort(epoch="02_r1", seed=0), seed=0),
                _with_event_length(_dm_contrast_cohort(epoch="04_r2", seed=1), seed=1),
            ],
            ignore_index=True,
        )
        ladder = {"C0_base": (["is_current_arm"], "pooled")}
        kwargs = {
            "response_col": "arm_count_hard",
            "n_categories": 9,
            "estimator": "dirichlet_multinomial",
            "n_jobs": 2,
            "min_events_session_phi": 40,
        }

        total, _ = mod._run(cohort, ladder, concentration_length=False, **kwargs)
        adjusted, _ = mod._run(cohort, ladder, concentration_length=True, **kwargs)

        assert "coef_log_phi_log_n_nonlocal_bins_z" not in total.columns
        assert adjusted["coef_log_phi_log_n_nonlocal_bins_z"].notna().all()
        assert adjusted["bse_log_phi_log_n_nonlocal_bins_z"].notna().all()
        # The effect-scale unit travels with the coefficient (the column is
        # already z-scored, so its SD is ~1).
        assert (adjusted["sd_log_phi_log_n_nonlocal_bins_z"] > 0).all()
        # One more free parameter than the unadjusted twin.
        assert (
            adjusted["n_params"].to_numpy() == total["n_params"].to_numpy() + 1
        ).all()

    def test_the_aggregation_reports_the_length_term_next_to_the_contrast(self):
        cohort = pd.concat(
            [
                _with_event_length(
                    _dm_contrast_cohort(epoch=f"{index:02d}_r1", seed=index), seed=index
                )
                for index in range(6)
            ],
            ignore_index=True,
        )
        ladder = {"C0_base": (["is_current_arm"], "pooled")}
        res, _ = mod._run(
            cohort,
            ladder,
            response_col="arm_count_hard",
            n_categories=9,
            estimator="dirichlet_multinomial",
            n_jobs=2,
            min_events_session_phi=40,
            concentration_length=True,
        )
        agg = aggregate_contrast(res, mod._concentration_specs(ladder), "candidate_arm")

        assert set(agg["term"]) == {
            "log_phi_is_swr",
            "log_phi_log_n_nonlocal_bins_z",
        }
        # A z-scored column is reported per SD, not per category.
        length = agg[agg["term"] == "log_phi_log_n_nonlocal_bins_z"].iloc[0]
        assert length["std_unit"] == "per_SD"


def test_concentration_specs_cover_every_rung_and_population():
    specs = mod._concentration_specs(
        {"C0_base": ([], "pooled"), "C1_split": ([], "split")}
    )
    # Both concentration terms on every (rung, population): the length term is
    # only fitted under --concentration-length, and on a run without it the
    # coefficient column is absent so aggregate_contrast emits no row for it.
    assert specs == [
        ("C0_base", "all", "log_phi_is_swr"),
        ("C0_base", "all", "log_phi_log_n_nonlocal_bins_z"),
        ("C1_split", "all", "log_phi_is_swr"),
        ("C1_split", "all", "log_phi_log_n_nonlocal_bins_z"),
        ("C1_split", "SWR", "log_phi_is_swr"),
        ("C1_split", "SWR", "log_phi_log_n_nonlocal_bins_z"),
        ("C1_split", "nonSWR", "log_phi_is_swr"),
        ("C1_split", "nonSWR", "log_phi_log_n_nonlocal_bins_z"),
    ]


def test_a_refused_dm_session_is_named_in_the_run_audit():
    """A session the estimator refuses is named in the record, not just counted.

    Its events land in ``n_excluded_session`` like any unfitted session, so the
    audit's ``drop_notes`` is the only place a reader learns which session went
    missing from the rung and why.
    """
    good = _dm_contrast_cohort(epoch="02_r1", seed=0)
    bad = _dm_contrast_cohort(epoch="04_r2", seed=1).assign(is_prev_arm=0.0)
    cohort = pd.concat([good, bad], ignore_index=True)

    res, audits = mod._run(
        cohort,
        {"C0_base": (["is_current_arm", "is_prev_arm"], "pooled")},
        response_col="arm_count_hard",
        n_categories=9,
        estimator="dirichlet_multinomial",
        n_jobs=2,
        min_events_session_phi=40,
    )

    # The refused session contributes no row.
    assert res["epoch"].tolist() == ["02_r1"]

    assert len(audits) == 1
    record = audits[0]
    notes = record["drop_notes"]
    assert "no fit: dm_design" in notes
    assert "estimator skipped chimi/20200212/04_r2 (dm_design:" in notes
    # Counted as a session-level exclusion, not as dropped events.
    assert record["n_drop_own_predictor_missing"] == 0
    assert record["n_excluded_session"] == 60
    assert record["n_modeled"] == 60


def test_the_summary_prints_a_refused_term_with_the_gates_reason(capsys) -> None:
    """Four sessions are below the gate's five, so the term is refused; six
    are not, and the term gets its numbers."""

    def fits(model: str, n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "model": model,
                "population": "all",
                "animal": [("a", "b", "c", "d", "e")[i % 5] for i in range(n)],
                "session_date": [f"d{i}" for i in range(n)],
                "epoch": "02_r1",
                "separated_terms": "",
                "coef_is_swr": np.linspace(0.1, 0.6, n),
            }
        )

    ladder = {"few": ([], "pooled"), "enough": ([], "pooled")}
    res = pd.concat([fits("few", 4), fits("enough", 6)], ignore_index=True)

    mod._summ(res, ladder, {"few": "is_swr", "enough": "is_swr"})

    lines = capsys.readouterr().out.splitlines()
    refused = next(line for line in lines if line.startswith("few"))
    assert refused.split()[1:] == [
        "all",
        "is_swr",
        "(not",
        "reported:",
        "too_few_sessions)",
    ]
    kept = next(line for line in lines if line.startswith("enough"))
    assert kept.split()[3] == "6"
    # a term longer than its 30-character column stays apart from the reason
    long_term = "is_swr_x_" + "a" * 30
    mod._summ(
        fits("few", 4).rename(columns={"coef_is_swr": f"coef_{long_term}"}),
        {"few": ([], "pooled")},
        {"few": long_term},
    )
    assert f"{long_term} (not reported: too_few_sessions)" in capsys.readouterr().out
