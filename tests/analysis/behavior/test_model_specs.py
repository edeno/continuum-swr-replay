"""Contracts for the shipped rate-model specifications.

The rate ladders model reward history with the raised-cosine basis
(``reward_history_rcos_0..2``) plus the structural-missingness indicator
``reward_history_incomplete``; the per-lag dummies
(``known_reward_outcome_lag_{j}``) and the window-10 fraction
(``reward_fraction_last10_known_trials``) stay persisted as stop metadata and
are never ladder predictors. These tests pin that, pin the indicator's
placement outside every ``is_swr`` expansion, and pin that each per-bin
formula's terms are exactly the columns the fitter asks the design for.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior import model_specs as ms
from continuum_swr_replay.analysis.behavior.contrast_aggregation import (
    BINARY_TERMS,
    aggregate_contrast,
    specs_from_ladder,
)
from continuum_swr_replay.analysis.behavior.rate.per_bin_rate_model import (
    _design_columns,
)

RETIRED_PREDICTORS = (
    "known_reward_outcome_lag_",
    "reward_fraction_last10_known_trials",
)

# Every per-bin spec this module ships: the Family B ladder and the auxiliary
# per-bin models fit by their own runners.
PER_BIN_LADDERS = (
    ms.MODEL_PER_BIN_RATE,
    ms.MODEL_UNCERTAINTY_RATE,
    ms.MODEL_CROSS_HISTORY_RATE,
)
ALL_PER_BIN_SPECS = {
    name: spec for ladder in PER_BIN_LADDERS for name, spec in ladder.items()
}


def test_the_per_bin_ladders_share_no_rung_name() -> None:
    """Merging them into ``ALL_PER_BIN_SPECS`` must drop no rung."""
    assert len(ALL_PER_BIN_SPECS) == sum(len(ladder) for ladder in PER_BIN_LADDERS)


def _all_shipped_formulas() -> dict[str, str]:
    """Every fitted-model expression this module ships, keyed by rung name."""
    out = {
        name: " + ".join(spec.predictors)
        for name, spec in ms.MODEL_PER_STOP_RATE.items()
    }
    out.update({name: spec.formula() for name, spec in ALL_PER_BIN_SPECS.items()})
    out.update({n: f for n, (f, _) in ms.SWR_CONTRAST_RATE_G_SPECS.items()})
    out.update({n: f for n, (f, _) in ms.SWR_CONTRAST_RATE_S_SPECS.items()})
    return out


def test_no_shipped_rate_spec_uses_the_retired_reward_encodings() -> None:
    for name, formula in _all_shipped_formulas().items():
        for retired in RETIRED_PREDICTORS:
            assert retired not in formula, f"{name}: {retired}"


def test_no_shipped_need_list_asks_for_the_retired_reward_encodings() -> None:
    ladders = (ms.SWR_CONTRAST_RATE_G_SPECS, ms.SWR_CONTRAST_RATE_S_SPECS)
    need_lists = {n: need for lad in ladders for n, (_, need) in lad.items()}
    need_lists.update({n: _design_columns(s) for n, s in ALL_PER_BIN_SPECS.items()})
    need_lists["STOP_COVARIATES"] = list(ms.STOP_COVARIATES)
    need_lists["SWR_CONTRAST_RATE_STOP_COVARS"] = list(ms.SWR_CONTRAST_RATE_STOP_COVARS)
    for name, need in need_lists.items():
        for col in need:
            for retired in RETIRED_PREDICTORS:
                assert retired not in col, f"{name}: {col}"


def test_per_stop_ladder_is_the_null_plus_the_basis_rung() -> None:
    assert list(ms.MODEL_PER_STOP_RATE) == ["R0", "R1"]
    assert ms.MODEL_PER_STOP_RATE["R0"].predictors == ()
    assert ms.MODEL_PER_STOP_RATE["R1"].predictors == (
        *ms.REWARD_HISTORY_COLUMNS,
        ms.REWARD_HISTORY_INCOMPLETE,
    )
    assert ms.MODEL_PER_STOP_RATE["R1"].parent == "R0"


def test_per_bin_ladder_reward_rungs_carry_the_basis() -> None:
    basis_terms = " + ".join([*ms.REWARD_HISTORY_COLUMNS, ms.REWARD_HISTORY_INCOMPLETE])
    reward_rungs = {
        "M1_rew": "M0_const",
        "M7r_time_histcos_rew": "M6r_time_histcos",
        "M9r_rew_x_time": "M7r_time_histcos_rew",
        "M7ra_time_arrival_histcos_rew": "M5a_time_arrival",
    }
    for name, parent in reward_rungs.items():
        spec = ms.MODEL_PER_BIN_RATE[name]
        assert spec.reward_basis == ms.REWARD_HISTORY_N_BASIS, name
        assert spec.parent == parent, name
        assert basis_terms in spec.formula(), name
    for gone in ("M1_rew1", "M2_rewhist", "M2r_rewrate"):
        assert gone not in ms.MODEL_PER_BIN_RATE
    for name, spec in ms.MODEL_PER_BIN_RATE.items():
        if name not in reward_rungs:
            assert spec.reward_basis == 0, name


def test_the_reward_block_has_a_time_plus_self_history_null() -> None:
    """``M7r - M6r`` is the reward block alone, not reward plus self-history."""
    m6r = ms.MODEL_PER_BIN_RATE["M6r_time_histcos"]
    m7r = ms.MODEL_PER_BIN_RATE["M7r_time_histcos_rew"]
    assert m6r.parent == "M5_time"
    assert m6r.reward_basis == 0
    assert m7r.parent == "M6r_time_histcos"
    # Identical apart from the reward block, so the step is nested and clean.
    reward_terms = [*ms.REWARD_HISTORY_COLUMNS, ms.REWARD_HISTORY_INCOMPLETE]
    m6r_terms = [t.strip() for t in m6r.formula().split("~", 1)[1].split("+")]
    m7r_terms = [t.strip() for t in m7r.formula().split("~", 1)[1].split("+")]
    assert [t for t in m7r_terms if t not in reward_terms] == m6r_terms


def _rhs_terms(spec: ms.PerBinRateModelSpec) -> list[str]:
    return [t.strip() for t in re.split(r"\+(?![^(]*\))", spec.formula().split("~")[1])]


def test_mu_uncertainty_is_m7r_plus_the_uncertainty_block() -> None:
    mu = ms.MODEL_UNCERTAINTY_RATE["MU_uncertainty"]
    m7r = ms.MODEL_PER_BIN_RATE[mu.parent]
    assert mu.parent == "M7r_time_histcos_rew"
    assert _rhs_terms(mu) == [*_rhs_terms(m7r), *ms.MU_UNCERTAINTY_COVARIATES]
    assert ms.MU_UNCERTAINTY_COVARIATES == (
        *ms.MU_UNCERTAINTY_CURATED_COVARIATES,
        *ms.MU_UNCERTAINTY_MODEL_VALUE_COVARIATES,
        *ms.MU_UNCERTAINTY_RL_COVARIATES,
    )


def test_cross_history_pair_differs_only_by_the_cross_basis() -> None:
    self_only = ms.MODEL_CROSS_HISTORY_RATE["Mself"]
    with_cross = ms.MODEL_CROSS_HISTORY_RATE["Mselfcross"]
    cross_terms = [f"cross_hist_rcos_{k}" for k in range(ms.N_HISTORY_BASIS)]
    assert _rhs_terms(with_cross) == [*_rhs_terms(self_only), *cross_terms]
    # Same self-history window, so one history window serves both designs.
    assert with_cross.cross_history_lags == self_only.self_history_lags


def test_reward_time_interaction_uses_the_immediate_component() -> None:
    formula = ms.MODEL_PER_BIN_RATE["M9r_rew_x_time"].formula()
    assert f"{ms.REWARD_HISTORY_COLUMNS[0]}:{ms._SPLINE}" in formula


def test_every_per_bin_formula_term_resolves_through_design_columns() -> None:
    """Each formula term is a column the fitter requests (``tc`` via the spline)."""
    for name, spec in ALL_PER_BIN_SPECS.items():
        cols = set(_design_columns(spec))
        assert "count" in cols
        rhs = spec.formula().split("~", 1)[1]
        # Split on '+' outside the spline's own argument list, then on ':' so
        # each interaction factor is checked on its own.
        terms = [t.strip() for t in re.split(r"\+(?![^(]*\))", rhs)]
        factors = [f.strip() for t in terms for f in t.split(":")]
        for factor in factors:
            if factor == "1":  # intercept-only rung
                continue
            if factor == ms._SPLINE:
                assert "tc" in cols, name
                continue
            if factor == ms._ARRIVAL_SPLINE:
                assert ms.ARRIVAL_CLOCK_COLUMN in cols, name
                continue
            assert factor in cols, f"{name}: {factor}"
        # and nothing is requested that the formula never mentions. An arrival
        # rung requests its clock whether or not it splines it: the runner
        # windows the rung on that column, so a bin without one is outside the
        # rung's support and has to leave the ladder's common cohort.
        clock_cols = {"tc"}
        if spec.time_clock == "arrival":
            clock_cols.add(ms.ARRIVAL_CLOCK_COLUMN)
        for col in cols - {"count"} - clock_cols:
            assert col in rhs, f"{name}: unused design column {col}"


def test_swr_contrast_specs_place_the_indicator_outside_the_interaction() -> None:
    indicator = ms.REWARD_HISTORY_INCOMPLETE
    for name, (formula, _) in ms.SWR_CONTRAST_RATE_S_SPECS.items():
        assert formula.count(indicator) == 1, name
        assert formula.endswith(f" + {indicator}"), name
        # nothing inside a parenthesized is_swr expansion mentions it
        for group in re.findall(r"\(([^()]*)\)", formula):
            assert indicator not in group, name


def test_swr_contrast_need_lists_carry_the_indicator() -> None:
    for ladder in (ms.SWR_CONTRAST_RATE_G_SPECS, ms.SWR_CONTRAST_RATE_S_SPECS):
        for name, (formula, need) in ladder.items():
            assert ms.REWARD_HISTORY_INCOMPLETE in need, name
            assert ms.REWARD_HISTORY_INCOMPLETE in formula, name
            for col in ms.REWARD_HISTORY_COLUMNS:
                assert col in need, f"{name}: {col}"


def test_swr_contrast_base_block_is_spline_basis_and_self_history() -> None:
    base = ms.SWR_CONTRAST_RATE_G_SPECS["G0_base"][0]
    expected = (
        f"{ms._SPLINE} + "
        + " + ".join(ms.REWARD_HISTORY_COLUMNS)
        + " + "
        + " + ".join(ms.PER_BIN_HISTORY_COLUMNS)
        + f" + {ms.REWARD_HISTORY_INCOMPLETE}"
    )
    assert base == expected


def test_immediate_component_contrast_is_the_s1_rung() -> None:
    formula, _ = ms.SWR_CONTRAST_RATE_S_SPECS["S1_rew"]
    assert f"is_swr:{ms.REWARD_HISTORY_COLUMNS[0]}" in formula
    assert "S1_rew1" not in ms.SWR_CONTRAST_RATE_S_SPECS


def test_binary_terms_no_longer_claim_the_reward_lag_dummy() -> None:
    assert not any("known_reward_outcome_lag" in t for t in BINARY_TERMS)
    # the basis features are continuous fractions -> standardized, not per-category
    for col in ms.REWARD_HISTORY_COLUMNS:
        assert col not in BINARY_TERMS
        assert f"is_swr:{col}" not in BINARY_TERMS


def test_aggregation_never_emits_a_row_for_the_missingness_indicator() -> None:
    """The indicator is a nuisance main effect: fit, never reported."""
    rng = np.random.default_rng(0)
    res = pd.DataFrame(
        {
            "model": ["S1_rew"] * 20,
            "population": ["nonSWR"] * 20,
            "animal": ["a"] * 10 + ["b"] * 10,
            f"coef_{ms.REWARD_HISTORY_INCOMPLETE}": rng.normal(0.5, 0.1, 20),
            f"bse_{ms.REWARD_HISTORY_INCOMPLETE}": np.full(20, 0.05),
            f"pval_{ms.REWARD_HISTORY_INCOMPLETE}": np.full(20, 0.01),
            "coef_is_swr:reward_history_rcos_0": rng.normal(0.3, 0.1, 20),
            "bse_is_swr:reward_history_rcos_0": np.full(20, 0.05),
            "pval_is_swr:reward_history_rcos_0": np.full(20, 0.01),
        }
    )
    key_term = {"S1_rew": "is_swr:reward_history_rcos_0"}
    specs = specs_from_ladder({"S1_rew": ((), "split")}, key_term)
    assert all(term != ms.REWARD_HISTORY_INCOMPLETE for _, _, term in specs)
    agg = aggregate_contrast(res, specs, level="rate")
    assert ms.REWARD_HISTORY_INCOMPLETE not in set(agg["term"])
    assert set(agg["term"]) == {"is_swr:reward_history_rcos_0"}


# --- Content ladder: event-length and within-stop-time controls ---

_ARM_CONTROLS = (
    "log1p_time_since_stop_start_s_z_x_is_current_arm",
    "log_n_nonlocal_bins_z_x_is_current_arm",
    "log_n_nonlocal_bins_z_x_is_prev_arm",
)
_PATCH_CONTROLS = (
    "log1p_time_since_stop_start_s_z_x_is_current_patch",
    "log_n_nonlocal_bins_z_x_is_current_patch",
    "log_n_nonlocal_bins_z_x_is_prev_patch",
)


_NT_TWIN = re.compile(r"^CP?S\d+nt_")


def _rungs_built_on(ladder: dict, base: list[str]) -> dict[str, list[str]]:
    """Ladder rungs whose predictors extend ``base`` (the spatial-baseline rungs
    C0/C1 do not, and the ``nt`` twins are the deliberately uncontrolled ones)."""
    return {
        name: preds
        for name, (preds, _scope) in ladder.items()
        if set(base) <= set(preds) and not _NT_TWIN.match(name)
    }


def test_content_arm_rungs_carry_the_length_and_time_controls() -> None:
    rungs = _rungs_built_on(ms.SWR_CONTRAST_CONTENT_ARM_LADDER, ms.BASEc_NT)
    assert "CS1_current" in rungs and "CS5_armvalue" in rungs and "C2_base" in rungs
    for name, preds in rungs.items():
        for control in _ARM_CONTROLS:
            assert control in preds, f"{name} is missing {control}"
        assert len(preds) == len(set(preds)), f"{name} repeats a predictor"


def test_content_patch_rungs_carry_the_length_and_time_controls() -> None:
    rungs = _rungs_built_on(ms.SWR_CONTRAST_CONTENT_PATCH_LADDER, ms.BASEc_PATCH_NT)
    assert "CPS1_current" in rungs and "CP0_base" in rungs
    for name, preds in rungs.items():
        for control in _PATCH_CONTROLS:
            assert control in preds, f"{name} is missing {control}"
        assert len(preds) == len(set(preds)), f"{name} repeats a predictor"


def test_uncontrolled_twins_keep_the_base_without_the_controls() -> None:
    arm_nt = ms.SWR_CONTRAST_CONTENT_ARM_LADDER["CS1nt_current"][0]
    patch_nt = ms.SWR_CONTRAST_CONTENT_PATCH_LADDER["CPS1nt_current"][0]

    assert arm_nt == [*ms.BASEc_NT_SWR, "swr_x_is_current_arm"]
    assert patch_nt == [*ms.BASEc_PATCH_NT, "swr_x_is_current_patch"]
    for control in _ARM_CONTROLS:
        assert control not in arm_nt
    for control in _PATCH_CONTROLS:
        assert control not in patch_nt


def test_every_nt_twin_mirrors_its_controlled_rung() -> None:
    """Each ``nt`` twin swaps ``BASEc``/``BASEc_PATCH`` for the uncontrolled base
    and is otherwise term-for-term its controlled rung: same added terms, same
    scope, same key term, same leaf-only membership. That is what makes the pair
    a length-adjusted (direct) vs total SWR-effect contrast rather than two
    unrelated fits."""
    ladders = (
        (
            ms.SWR_CONTRAST_CONTENT_ARM_LADDER,
            ms.SWR_CONTRAST_CONTENT_ARM_KEY_TERM,
            ms.BASEc,
            ms.BASEc_NT,
            _ARM_CONTROLS,
        ),
        (
            ms.SWR_CONTRAST_CONTENT_PATCH_LADDER,
            ms.SWR_CONTRAST_CONTENT_PATCH_KEY_TERM,
            ms.BASEc_PATCH,
            ms.BASEc_PATCH_NT,
            _PATCH_CONTROLS,
        ),
    )
    seen = 0
    for ladder, key_term, base, base_nt, controls in ladders:
        for name, (preds, scope) in ladder.items():
            if not _NT_TWIN.match(name):
                continue
            seen += 1
            twin_of = name.replace("nt_", "_", 1)
            assert twin_of in ladder, f"{name} has no controlled rung {twin_of}"
            base_preds, base_scope = ladder[twin_of]
            assert preds == [*base_nt, *[p for p in base_preds if p not in base]]
            assert scope == base_scope
            assert key_term[name] == key_term[twin_of]
            assert (name in ms.LEAF_ONLY_CONTRAST_RUNGS) == (
                twin_of in ms.LEAF_ONLY_CONTRAST_RUNGS
            )
            for control in controls:
                assert control not in preds
    assert seen == 5


def test_base_minus_the_controls_is_the_uncontrolled_base() -> None:
    assert [p for p in ms.BASEc if p not in _ARM_CONTROLS] == ms.BASEc_NT
    assert ms.BASEc == [*ms.BASEc_NT, *_ARM_CONTROLS]
    assert [p for p in ms.BASEc_PATCH if p not in _PATCH_CONTROLS] == ms.BASEc_PATCH_NT
    assert ms.BASEc_PATCH == [*ms.BASEc_PATCH_NT, *_PATCH_CONTROLS]


def test_a_comparison_rung_pulls_in_the_parent_it_is_differenced_against() -> None:
    selected = ms.resolve_comparison_rungs(["CS5_armvalue"])
    names = [rung.name for rung in selected]
    assert names == ["C2_base_swr_leaf", "CS5_armvalue"]
    # The parent carries everything but the two value terms, so the rung's
    # delta_aic prices only those.
    parent, rung = selected
    assert set(rung.predictors) - set(parent.predictors) == {
        "reward_belief_mean_candidate_arm_preoutcome",
        "swr_x_reward_belief_mean_candidate_arm_preoutcome",
    }
    assert set(parent.predictors) <= set(rung.predictors)
    # A leaf-only rung must be differenced against a leaf-only parent: a
    # six-category likelihood and a nine-arm one are not comparable.
    assert all(rung.leaf_only for rung in selected)
    assert selected[-1].n_categories == 6


def test_an_unknown_comparison_rung_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown rung"):
        ms.resolve_comparison_rungs(["not_a_rung"])


def test_every_comparison_rung_names_a_parent_on_its_own_candidate_set() -> None:
    catalog = {rung.name: rung for rung in ms.CONTENT_ESTIMATOR_COMPARISON_RUNGS}
    for rung in ms.CONTENT_ESTIMATOR_COMPARISON_RUNGS:
        if rung.parent is None:
            continue
        assert catalog[rung.parent].leaf_only == rung.leaf_only
        assert set(catalog[rung.parent].predictors) <= set(rung.predictors)
