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
    renormalized_rungs,
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


def test_provenance_names_only_the_rescaled_rungs_that_produced_rows():
    """A rung nothing was fitted for cannot be claimed as rescaled."""
    fitted = pd.DataFrame(
        {
            "model": ["C4_armvalue", "C4_armvalue", "C1_nine_arm"],
            "response_renormalized": [True, True, False],
        }
    )
    args = _provenance_args()
    prov = mod._provenance(
        args, n_sessions=1, renormalized_rungs=renormalized_rungs(fitted)
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
