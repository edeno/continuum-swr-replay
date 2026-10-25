"""Interaction products: one declaration builds them and answers timing and qvals-gap.

``covariates.interactions.INTERACTION_FACTORS`` replaced three hand-kept copies
(at 436c4538): the products written out one by one in ``contrast_aggregation``,
the product entries of ``covariate_timing.DERIVED_COVARIATE_SOURCES``, and the
``_x_`` name split in ``cohort_audit``. These tests pin that the declaration
reproduces all three:

* the built columns, bit for bit, against copies of the old builders
  (synthetic frames here; the canonical tables under ``-m real_data``);
* every covariate-timing verdict and every qvals answer, for every column of the
  table-parity slice's tables and fit cohorts, every ladder predictor and every
  catalogued covariate, against ``data/interaction_provenance_answers.json`` --
  the answers the code gave at 436c4538;
* the one intended difference: ``timing_of`` now lists every factor of a
  product, so eleven products whose old entry left out the arm or patch
  indicator gain it. The indicators are pre-outcome, so no verdict moves.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay import cohort_audit as coh
from continuum_swr_replay.analysis.replay import contrast_aggregation as ca
from continuum_swr_replay.analysis.replay import covariate_timing as ct
from continuum_swr_replay.analysis.replay.covariates.interactions import (
    INTERACTION_FACTORS,
    factor_source,
)
from continuum_swr_replay.tables.io import read_table
from tests.unit.test_covariate_timing import _all_ladder_predictors

ANSWERS_436C4538 = (
    Path(__file__).parent / "data" / "interaction_provenance_answers.json"
)

# Products whose old timing entry listed only the covariate factor; the
# declaration lists the indicator factor too.
INDICATOR_ADDED = {
    "is_periswitch_post_x_is_current_arm": "is_current_arm",
    "patch_switch_fraction_last5_completed_trips_z_x_is_current_arm": "is_current_arm",
    "patch_switch_fraction_last5_completed_trips_z_x_is_current_patch": (
        "is_current_patch"
    ),
    "periswitch_index_trials_z_x_is_current_arm": "is_current_arm",
    "periswitch_index_trials_z_x_is_current_patch": "is_current_patch",
    "periswitch_index_trials_z_x_is_periswitch_post_x_is_current_arm": (
        "is_current_arm"
    ),
    "relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm": (
        "is_current_arm"
    ),
    "reward_belief_differential_entropy_leaf_mean_preoutcome_z_x_is_current_arm": (
        "is_current_arm"
    ),
    "reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm": (
        "is_current_arm"
    ),
    "rpe_postoutcome_z_x_is_current_arm": "is_current_arm",
    "trials_since_block_change_z_x_is_current_arm": "is_current_arm",
}


# --- The builders' product blocks at 436c4538, constants spelled out ----------


def _old_arm_products(ev: pd.DataFrame) -> pd.DataFrame:
    cur = ev["is_current_arm"].to_numpy(dtype=float)
    prev = ev["is_prev_arm"].to_numpy(dtype=float)
    swr = ev["is_swr"].to_numpy(dtype=float)
    ev["rpe_postoutcome_z_x_is_current_arm"] = cur * ev["rpe_postoutcome_z"]
    ev["relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm"] = (
        cur * ev["relative_value_switch_minus_stay_preoutcome_z"]
    )
    ev["reward_belief_differential_entropy_leaf_mean_preoutcome_z_x_is_current_arm"] = (
        cur * ev["reward_belief_differential_entropy_leaf_mean_preoutcome_z"]
    )
    ev["reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm"] = (
        cur * ev["reward_belief_mean_update_l1_all_ports_postoutcome_z"]
    )
    ev["periswitch_index_trials_z_x_is_current_arm"] = (
        cur * ev["periswitch_index_trials_z"]
    )
    ev["is_periswitch_post_x_is_current_arm"] = cur * ev["is_periswitch_post"].to_numpy(
        dtype=float
    )
    ev["periswitch_index_trials_z_x_is_periswitch_post_x_is_current_arm"] = (
        cur
        * ev["periswitch_index_trials_z"]
        * ev["is_periswitch_post"].to_numpy(dtype=float)
    )
    ev["trials_since_block_change_z_x_is_current_arm"] = (
        cur * ev["trials_since_block_change_z"]
    )
    ev["patch_switch_fraction_last5_completed_trips_z_x_is_current_arm"] = (
        cur * ev["patch_switch_fraction_last5_completed_trips_z"]
    )
    ev["log1p_time_since_stop_start_s_z_x_is_current_arm"] = (
        cur * ev["log1p_time_since_stop_start_s_z"]
    )
    ev["log_n_nonlocal_bins_z_x_is_current_arm"] = cur * ev["log_n_nonlocal_bins_z"]
    ev["log_n_nonlocal_bins_z_x_is_prev_arm"] = prev * ev["log_n_nonlocal_bins_z"]
    ev["swr_x_reward_belief_mean_candidate_arm_preoutcome"] = swr * ev[
        "reward_belief_mean_candidate_arm_preoutcome"
    ].to_numpy(dtype=float)
    ev["swr_x_reward_belief_variance_candidate_arm_preoutcome_z"] = swr * ev[
        "reward_belief_variance_candidate_arm_preoutcome_z"
    ].to_numpy(dtype=float)
    return ev


def _old_patch_products(pt: pd.DataFrame) -> pd.DataFrame:
    cp = pt["is_current_patch"].to_numpy(float)
    pp = pt["is_prev_patch"].to_numpy(float)
    swr = pt["is_swr"].to_numpy(float)
    pt["periswitch_index_trials_z_x_is_current_patch"] = (
        cp * pt["periswitch_index_trials_z"]
    )
    pt["log1p_time_since_stop_start_s_z_x_is_current_patch"] = (
        cp * pt["log1p_time_since_stop_start_s_z"]
    )
    pt["log_n_nonlocal_bins_z_x_is_current_patch"] = cp * pt["log_n_nonlocal_bins_z"]
    pt["log_n_nonlocal_bins_z_x_is_prev_patch"] = pp * pt["log_n_nonlocal_bins_z"]
    pt["patch_switch_fraction_last5_completed_trips_z_x_is_current_patch"] = (
        cp * pt["patch_switch_fraction_last5_completed_trips_z"]
    )
    pt["swr_x_is_current_patch"] = swr * cp
    return pt


# --- Helpers --------------------------------------------------------------------


def _assert_identical(old: pd.DataFrame, new: pd.DataFrame) -> None:
    """Same columns, order, dtypes and values; float columns bit for bit."""
    pd.testing.assert_frame_equal(old, new, check_exact=True)
    for column in old.columns:
        if old[column].dtype == np.float64:
            assert np.array_equal(
                old[column].to_numpy().view(np.uint64),
                new[column].to_numpy().view(np.uint64),
            ), column


def _old_and_new(builder, frame: pd.DataFrame, old_products, monkeypatch):
    """``builder``'s output with the old product block and with the declaration."""
    real = ca.attach_interactions
    captured = {}

    def capture(design: pd.DataFrame, names):
        captured["pre"] = design.copy()
        return real(design, names)

    monkeypatch.setattr(ca, "attach_interactions", capture)
    new, _reference = builder(frame)
    old = old_products(captured["pre"])
    return old, new


def _verdict(name: str, retrospective: bool, descriptive: bool) -> str:
    """The timing verdict as recorded in the answers file."""
    try:
        ct.assert_predictor_timing(
            [name], retrospective=retrospective, descriptive=descriptive
        )
    except ValueError as error:
        message = str(error)
        if "absolute session clock" in message:
            return "absolute_clock"
        if "unclassified timing atom" in message:
            unknown = sorted(set(ct.timing_of(name)) - ct.KNOWN_ATOMS)
            return f"unclassified:{','.join(unknown)}"
        if "future-referencing" in message:
            return "future_referencing"
        if "post-outcome" in message:
            return "post_outcome"
        raise
    return "ok"


def _synthetic_arm_frame() -> pd.DataFrame:
    """Events x arms with negative values, NaN factors and off-well (NaN) rows."""
    rng = np.random.default_rng(4)
    n_events, n_arms = 40, 9
    n = n_events * n_arms
    arms = np.tile(np.arange(n_arms), n_events)
    frame = pd.DataFrame(
        {
            "event_id": np.repeat(np.arange(n_events), n_arms),
            "candidate_arm": arms,
            "n_nonlocal_bins": np.repeat(rng.integers(1, 90, n_events), n_arms),
            "time_since_stop_start_s": np.repeat(rng.uniform(0, 9, n_events), n_arms),
            "is_current_arm": (arms == 0).astype(float),
            "is_prev_arm": (arms == 1).astype(int),
            "is_swr": np.repeat(rng.uniform(size=n_events) < 0.5, n_arms),
            "is_periswitch_post": np.repeat(
                rng.choice([0.0, 1.0, np.nan], n_events), n_arms
            ),
            "reward_belief_mean_candidate_arm_preoutcome": rng.normal(size=n),
            "reward_belief_variance_candidate_arm_preoutcome_z": rng.normal(size=n),
        }
    )
    for source in ca._ZSCALARS:
        values = np.repeat(rng.normal(size=n_events), n_arms)
        values[rng.uniform(size=n) < 0.1] = np.nan
        frame[source] = values
    frame.loc[frame["event_id"] == 3, "is_current_arm"] = np.nan
    frame.loc[5, "reward_belief_mean_candidate_arm_preoutcome"] = np.nan
    return frame


def _synthetic_patch_frame() -> pd.DataFrame:
    """Events x patches carrying the canonical integer ``swr_x_is_current_patch``."""
    rng = np.random.default_rng(5)
    n_events, n_patches = 40, 3
    patches = np.tile(np.arange(n_patches), n_events)
    is_swr = np.repeat(rng.uniform(size=n_events) < 0.5, n_patches)
    frame = pd.DataFrame(
        {
            "event_id": np.repeat(np.arange(n_events), n_patches),
            "patch": patches,
            "n_nonlocal_bins": np.repeat(rng.integers(1, 90, n_events), n_patches),
            "time_since_stop_start_s": np.repeat(
                rng.uniform(0, 9, n_events), n_patches
            ),
            "is_current_patch": (patches == 0).astype(float),
            "is_prev_patch": (patches == 2).astype(int),
            "is_swr": is_swr,
            "swr_x_is_current_patch": is_swr.astype(int) * (patches == 0),
            "periswitch_index_trials": np.repeat(rng.normal(size=n_events), n_patches),
            "patch_switch_fraction_last5_completed_trips": np.repeat(
                rng.uniform(size=n_events), n_patches
            ),
        }
    )
    frame.loc[frame["event_id"] == 7, "is_current_patch"] = np.nan
    frame.loc[frame["event_id"] == 9, "periswitch_index_trials"] = np.nan
    return frame


# --- The declaration ------------------------------------------------------------


def test_each_product_name_spells_its_declared_factors():
    for name, factors in INTERACTION_FACTORS.items():
        spelled = "_x_".join("swr" if f == "is_swr" else f for f in factors)
        assert spelled == name


def test_every_fitted_product_is_declared():
    products = {p for p in _all_ladder_predictors() if "_x_" in p}
    assert products - set(INTERACTION_FACTORS) == set()
    built = (*ca._ARM_INTERACTIONS, *ca._PATCH_INTERACTIONS)
    assert len(set(built)) == len(built)
    assert set(built) <= set(INTERACTION_FACTORS)


def test_every_factor_resolves_to_a_catalogued_covariate():
    for factors in INTERACTION_FACTORS.values():
        for factor in factors:
            assert factor in ct.SAFE_MODIFIERS or factor_source(factor) in (
                ct.KNOWN_ATOMS
            ), factor


# --- Built columns --------------------------------------------------------------


def test_arm_products_match_the_old_builder(monkeypatch):
    old, new = _old_and_new(
        ca._attach_arm_design_columns,
        _synthetic_arm_frame(),
        _old_arm_products,
        monkeypatch,
    )
    _assert_identical(old, new)
    assert new["rpe_postoutcome_z_x_is_current_arm"].isna().any()


def test_patch_products_match_the_old_builder(monkeypatch):
    old, new = _old_and_new(
        ca._attach_patch_design_columns,
        _synthetic_patch_frame(),
        _old_patch_products,
        monkeypatch,
    )
    _assert_identical(old, new)
    # The fit-time build still overwrites the canonical integer column in place.
    assert new["swr_x_is_current_patch"].dtype == np.float64
    assert list(new.columns).index("swr_x_is_current_patch") == 7


# --- Timing and qvals answers ---------------------------------------------------


def test_timing_and_qvals_answers_match_436c4538():
    frozen = json.loads(ANSWERS_436C4538.read_text())
    assert set(INDICATOR_ADDED) <= set(frozen)
    for name, old in frozen.items():
        new_atoms = ct.timing_of(name)
        if name in INDICATOR_ADDED:
            assert new_atoms == [*old["timing_of"], INDICATOR_ADDED[name]], name
            assert INDICATOR_ADDED[name] in ct.PRE_OUTCOME_SAFE
        else:
            assert new_atoms == old["timing_of"], name
        verdicts = [_verdict(name, r, d) for r, d in product([False, True], repeat=2)]
        assert verdicts == old["verdicts"], name
        assert coh.is_qvals_sourced_column(name) is old["qvals_sourced"], name
        assert coh.is_qvals_gap_column(name) is old["qvals_gap"], name


# --- Real data ------------------------------------------------------------------


@pytest.mark.real_data
def test_products_match_the_old_builders_on_the_canonical_tables(monkeypatch):
    """The canonical fit's cohorts, built with the declaration and the old code."""
    inputs = ca.CONTENT_FIT_INPUTS
    real = ca.attach_interactions
    captured = {}

    def capture(design: pd.DataFrame, names):
        captured.setdefault("pre", []).append(design.copy())
        return real(design, names)

    monkeypatch.setattr(ca, "attach_interactions", capture)
    arm, _ = ca.assemble_arm(inputs)
    patch, _ = ca.assemble_patch(arm, inputs)
    arm_pre, patch_pre = captured["pre"]
    _assert_identical(_old_arm_products(arm_pre), arm)
    _assert_identical(_old_patch_products(patch_pre), patch)

    # The canonical tables' own swr_x_* columns are the products of the factors
    # declared for them.
    for table in ("event_arm_table", "event_patch_table"):
        frame = read_table(inputs[table])
        declared = [c for c in frame.columns if c in INTERACTION_FACTORS]
        assert declared, table
        for name in declared:
            expected = np.prod(
                [frame[f].to_numpy(dtype=float) for f in INTERACTION_FACTORS[name]],
                axis=0,
            )
            np.testing.assert_array_equal(
                frame[name].to_numpy(dtype=float), expected, err_msg=name
            )
