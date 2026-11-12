"""Unit conventions and session-drop rule of the shared contrast aggregation.

``std_factor`` resolves three mutually exclusive labels, in order: binary
per-category predictors, the reward-history basis features (weighted reward
fractions, so their coefficients are per unit of the fraction), and continuous
covariates scaled by their pooled SD.

``aggregate_contrast`` drops a session fit for a term only when THAT term is
unidentified: the fitters name the separated terms on the effect scale, with a
magnitude backstop read on that same scale (``|coef| * sd >= 10``, using the
fit's own persisted predictor SD; the raw ``|coef| >= 10`` only for legacy
frames that carry no ``sd_`` column). Precision alone is not a drop criterion.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay import contrast_aggregation
from continuum_swr_replay.analysis.replay.contrast_aggregation import (
    MAX_UNIDENTIFIED_SHARE,
    MIN_SESSIONS_PER_TERM,
    aggregate_contrast,
    gate_term_sessions,
    load_per_sd_scale,
    session_se,
    std_factor,
    typical_session_se,
)
from continuum_swr_replay.analysis.replay.model_specs import REWARD_HISTORY_COLUMNS

SD = {"rpe_postoutcome": 2.0, "reward_history_rcos_0": 0.25}

# Eight identified per-session coefficients; median 0.315.
REWARD_COEFS = [0.30, 0.35, 0.25, 0.40, 0.20, 0.45, 0.28, 0.33]


def test_binary_predictor_is_per_category() -> None:
    assert std_factor("is_current_arm", SD) == (1.0, "per_category")
    assert std_factor("swr_x_is_current_arm", SD) == (1.0, "per_category")


def test_reward_history_basis_is_per_unit_even_when_an_sd_is_available() -> None:
    """The features are fractions in [0, 1]; a stray SD entry must not rescale them."""
    for column in REWARD_HISTORY_COLUMNS:
        assert std_factor(column, SD) == (1.0, "per_unit")
        assert std_factor(f"is_swr:{column}", SD) == (1.0, "per_unit")


def test_continuous_covariate_is_scaled_by_its_pooled_sd() -> None:
    assert std_factor("rpe_postoutcome", SD) == (2.0, "per_SD")
    assert std_factor("is_swr:rpe_postoutcome", SD) == (2.0, "per_SD")
    assert std_factor("swr_x_rpe_postoutcome", SD) == (2.0, "per_SD")


@pytest.mark.parametrize(
    "term",
    [
        "rpe_postoutcome_z_x_is_current_arm",
        "periswitch_index_trials_z_x_is_current_patch",
        "reward_belief_variance_candidate_arm_preoutcome_z",
        "swr_x_reward_belief_variance_candidate_arm_preoutcome_z",
        "log_phi_log_n_nonlocal_bins_z",
    ],
)
def test_a_term_z_scored_at_fit_time_is_already_per_sd(term) -> None:
    """A ``_z`` column, an SWR product of one, or a ``_z_x_`` product."""
    assert std_factor(term, SD) == (1.0, "per_SD")


@pytest.mark.parametrize(
    "term",
    [
        "trials_since_block_change",
        "is_swr:trials_since_block_change",
        # An SWR product of a raw covariate is not exempt just for its ``_x_``.
        "swr_x_reward_belief_mean_candidate_arm_preoutcome",
    ],
)
def test_a_raw_term_without_an_sd_is_refused(term) -> None:
    """Without its SD it would be reported at raw scale under a per-SD label."""
    with pytest.raises(KeyError, match=term.replace(":", ".")):
        std_factor(term, SD)


def test_a_raw_term_without_an_sd_is_refused_by_the_aggregation() -> None:
    fits = _session_fits()

    with pytest.raises(KeyError, match="rpe_postoutcome"):
        aggregate_contrast(fits, SPECS, level="rate", sd={})


def test_the_last_rewarded_arm_indicator_is_per_category() -> None:
    """A 0/1 indicator, like its SWR product and its patch analogue."""
    assert std_factor("is_last_rewarded_noncurrent_arm", SD) == (1.0, "per_category")


def _session_fits(**overrides) -> pd.DataFrame:
    """Eight identified sessions across two animals for one (model, population)."""
    n = 8
    fits = pd.DataFrame(
        {
            "model": ["G5_rpe"] * n,
            "population": ["non_local"] * n,
            "animal": ["a"] * 4 + ["b"] * 4,
            "coef_rpe_postoutcome": [0.30, 0.35, 0.25, 0.40, 0.20, 0.45, 0.28, 0.33],
            "bse_rpe_postoutcome": [0.10] * n,
            "pval_rpe_postoutcome": [0.01] * n,
            "separated": [False] * n,
            "converged": [True] * n,
        }
    )
    for column, values in overrides.items():
        fits[column] = values
    return fits


SPECS = [("G5_rpe", "non_local", "rpe_postoutcome")]
# Resolves every synthetic term the session-gate tests aggregate. Those tests are
# about which sessions a term keeps; ``std_factor`` refuses a raw term with no SD.
_GATE_SD = {**SD, "q_arm": 1.0, "is_swr": 1.0, "hist_rcos_2": 1.0, "h": 1.0}


def test_imprecise_but_identified_session_is_kept() -> None:
    """A near-zero coefficient with an ordinary SE is a small effect, not separation.

    coef 0.01 with SE 0.5 satisfies the retired ``SE > 10 * |coef|`` rule, which
    dropped every session whose effect happened to be near zero -- selection on
    the estimate, biasing the surviving median away from zero.
    """
    fits = _session_fits()
    fits.loc[0, "coef_rpe_postoutcome"] = 0.01
    fits.loc[0, "bse_rpe_postoutcome"] = 0.5

    out = aggregate_contrast(fits, SPECS, level="rate", sd=_GATE_SD)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_sessions"] == 8
    assert row["n_dropped"] == 0
    assert row["coef_session_median"] == pytest.approx(0.305)


def test_separated_session_is_dropped() -> None:
    """Result frames written before the term-level flag carry only ``separated``;
    for those the fit-level flag remains the identifiability rule."""
    fits = _session_fits()
    fits.loc[0, "separated"] = True

    out = aggregate_contrast(fits, SPECS, level="rate", sd=_GATE_SD)

    row = out.iloc[0]
    assert row["n_sessions"] == 7
    assert row["n_dropped"] == 1
    assert row["n_sessions_flagged_separated"] == 1
    assert row["coef_session_median"] == pytest.approx(0.33)


def test_legacy_frame_without_predictor_sds_uses_the_raw_coefficient() -> None:
    """A frame with no ``sd_`` column has only the raw coefficient to judge on.

    These are artifacts written before the fitters persisted per-term predictor
    SDs; for them ``|coef| >= 10`` remains the backstop.
    """
    fits = _session_fits()
    assert "sd_rpe_postoutcome" not in fits.columns
    fits.loc[0, "coef_rpe_postoutcome"] = 42.0

    out = aggregate_contrast(fits, SPECS, level="rate", sd=_GATE_SD)

    row = out.iloc[0]
    assert row["n_sessions"] == 7
    assert row["n_dropped"] == 1


def test_large_coefficient_on_a_tiny_sd_predictor_is_kept() -> None:
    """A steep per-unit slope on a narrow covariate is identified, not runaway.

    ``|coef| = 12`` over an SD of 0.01 is a log-rate swing of 0.12 across one SD
    of the predictor -- an ordinary effect expressed in small units. Dropping it
    on the raw coefficient would be selection ON the estimate.
    """
    fits = _session_fits(sd_rpe_postoutcome=[0.01] * 8)
    fits.loc[0, "coef_rpe_postoutcome"] = 12.0

    out = aggregate_contrast(fits, SPECS, level="rate", sd=_GATE_SD)

    row = out.iloc[0]
    assert row["n_sessions"] == 8
    assert row["n_dropped"] == 0
    # 0.20, 0.25, 0.28, 0.33, 0.35, 0.40, 0.45, 12.0 -> median (0.33 + 0.35) / 2.
    assert row["coef_session_median"] == pytest.approx(0.34)


def test_large_coefficient_on_a_unit_sd_predictor_is_dropped() -> None:
    """``|coef| = 12`` over an SD of 1.0 is a runaway swing on the effect scale."""
    fits = _session_fits(sd_rpe_postoutcome=[1.0] * 8)
    fits.loc[0, "coef_rpe_postoutcome"] = 12.0

    out = aggregate_contrast(fits, SPECS, level="rate", sd=_GATE_SD)

    row = out.iloc[0]
    assert row["n_sessions"] == 7
    assert row["n_dropped"] == 1
    assert row["coef_session_median"] == pytest.approx(0.33)


def test_product_backstop_reads_the_products_own_sd() -> None:
    """An ``is_swr:<c>`` product is judged against its OWN persisted SD.

    The ``is_swr`` main effect's SD is 1.0 here and the product's is 0.01, so a
    coefficient of 12 on the product is an ordinary effect while the same
    coefficient on ``is_swr`` is a runaway one.
    """
    fits = _product_fits([""] * 12)
    fits["sd_is_swr:q_arm"] = 0.01
    fits["sd_is_swr"] = 1.0
    fits.loc[0, "coef_is_swr:q_arm"] = 12.0
    fits.loc[0, "coef_is_swr"] = 12.0

    out = aggregate_contrast(
        fits,
        [
            ("S4_value", "contrast", "is_swr:q_arm"),
            ("S4_value", "contrast", "is_swr"),
        ],
        level="rate",
        sd=_GATE_SD,
    ).set_index("term")

    assert out.loc["is_swr:q_arm", "n_dropped"] == 0
    assert out.loc["is_swr:q_arm", "n_sessions"] == 12
    assert out.loc["is_swr", "n_dropped"] == 1
    assert out.loc["is_swr", "n_sessions"] == 11


def _two_term_fits() -> pd.DataFrame:
    """Eight sessions carrying both a self-history and a reward-basis coefficient."""
    n = 8
    return pd.DataFrame(
        {
            "model": ["S1_rew"] * n,
            "population": ["contrast"] * n,
            "animal": ["a"] * 4 + ["b"] * 4,
            "coef_hist_rcos_2": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
            "bse_hist_rcos_2": [0.10] * n,
            "pval_hist_rcos_2": [0.01] * n,
            "coef_reward_history_rcos_0": REWARD_COEFS,
            "bse_reward_history_rcos_0": [0.10] * n,
            "pval_reward_history_rcos_0": [0.01] * n,
            "separated": [False] * n,
            "converged": [True] * n,
            "separated_terms": [""] * n,
        }
    )


def test_separation_drops_only_the_term_it_holds_for() -> None:
    """A session separated on ``hist_rcos_2`` still contributes its reward coefficient.

    The fit-level flag dropped the session for every term, which is what emptied
    the audited is_swr-contrast table.
    """
    fits = _two_term_fits()
    fits.loc[0, "separated"] = True
    fits.loc[0, "separated_terms"] = "hist_rcos_2"

    out = aggregate_contrast(
        fits,
        [
            ("S1_rew", "contrast", "hist_rcos_2"),
            ("S1_rew", "contrast", "reward_history_rcos_0"),
        ],
        level="rate",
        sd=_GATE_SD,
    ).set_index("term")

    assert out.loc["hist_rcos_2", "n_sessions"] == 7
    assert out.loc["hist_rcos_2", "n_dropped"] == 1
    assert out.loc["hist_rcos_2", "coef_session_median"] == pytest.approx(0.50)
    assert out.loc["reward_history_rcos_0", "n_sessions"] == 8
    assert out.loc["reward_history_rcos_0", "n_dropped"] == 0
    assert out.loc["reward_history_rcos_0", "coef_session_median"] == pytest.approx(
        0.315
    )
    # The fit-level diagnostic still counts the flagged fit, for both terms.
    assert set(out["n_sessions_flagged_separated"]) == {1}


def test_a_term_a_session_cannot_inform_is_counted_as_unidentified() -> None:
    """The fitter reports such a term with a NaN coefficient, infinite SEs and
    the term named in ``unidentified_terms``.

    That session drops out of the term's aggregate and is counted in
    ``n_unidentified`` (``n_dropped`` stays the count of separation drops); it
    still contributes its other terms.
    """
    fits = _two_term_fits()
    fits["bse_cluster_hist_rcos_2"] = [0.10] * len(fits)
    fits.loc[0, "coef_hist_rcos_2"] = float("nan")
    fits.loc[0, "pval_hist_rcos_2"] = float("nan")
    fits.loc[0, "bse_hist_rcos_2"] = float("inf")
    fits.loc[0, "bse_cluster_hist_rcos_2"] = float("inf")
    fits["unidentified_terms"] = ["hist_rcos_2:separated_to_minus_inf"] + [""] * (
        len(fits) - 1
    )

    out = aggregate_contrast(
        fits,
        [
            ("S1_rew", "contrast", "hist_rcos_2"),
            ("S1_rew", "contrast", "reward_history_rcos_0"),
        ],
        level="rate",
        sd=_GATE_SD,
    ).set_index("term")

    assert out.loc["hist_rcos_2", "n_sessions"] == 7
    assert out.loc["hist_rcos_2", "n_unidentified"] == 1
    assert out.loc["hist_rcos_2", "n_dropped"] == 0
    assert out.loc["hist_rcos_2", "coef_session_median"] == pytest.approx(0.50)
    assert out.loc["hist_rcos_2", "typical_session_se"] == pytest.approx(0.10)
    assert out.loc["reward_history_rcos_0", "n_sessions"] == 8
    assert out.loc["reward_history_rcos_0", "n_unidentified"] == 0


def test_a_term_half_the_sessions_cannot_inform_is_withheld(caplog) -> None:
    """The refusal gate counts unidentified sessions as it counts separated ones.

    A headline that 20 of 40 sessions cannot inform describes the identified
    half only, exactly as when those 20 were separated (the review's probe 2E).
    """
    n = 40
    fits = pd.DataFrame(
        {
            "model": "CS1",
            "population": "all",
            "animal": [animal for animal in "abcde" for _ in range(n // 5)],
            "coef_h": [0.3] * n,
            "bse_h": [0.1] * n,
            "bse_cluster_h": [0.1] * n,
            "pval_h": [0.01] * n,
            "sd_h": [1.0] * n,
            "separated_terms": [""] * n,
            "separated": [False] * n,
            "converged": [True] * n,
            "unidentified_terms": ["h:no_within_event_variation"] * (n // 2)
            + [""] * (n // 2),
        }
    )
    fits.loc[: n // 2 - 1, ["coef_h", "pval_h"]] = float("nan")
    fits.loc[: n // 2 - 1, ["bse_h", "bse_cluster_h"]] = float("inf")

    with caplog.at_level("WARNING"):
        out = aggregate_contrast(
            fits, [("CS1", "all", "h")], "candidate_arm", sd=_GATE_SD
        )

    assert out.empty
    assert any("20 of 40" in r.getMessage() for r in caplog.records)


def _product_fits(separated_terms: list[str]) -> pd.DataFrame:
    """Twelve sessions carrying both ``is_swr`` and its product with ``q_arm``."""
    n = len(separated_terms)
    return pd.DataFrame(
        {
            "model": ["S4_value"] * n,
            "population": ["contrast"] * n,
            "animal": ["a"] * (n // 2) + ["b"] * (n - n // 2),
            "coef_is_swr:q_arm": [0.1 * (i + 1) for i in range(n)],
            "bse_is_swr:q_arm": [0.10] * n,
            "pval_is_swr:q_arm": [0.01] * n,
            "coef_is_swr": [0.5] * n,
            "bse_is_swr": [0.10] * n,
            "pval_is_swr": [0.01] * n,
            "separated": [bool(t) for t in separated_terms],
            "converged": [True] * n,
            "separated_terms": separated_terms,
        }
    )


def test_product_term_does_not_inherit_its_factors_separation() -> None:
    """``is_swr:<c>`` is judged on its own coefficient, not on ``is_swr``'s.

    A large SE on the ``is_swr`` main effect says the two streams' baseline rates
    are hard to tell apart; it says nothing about whether the covariate's slope
    differs between them, which is what the product estimates.
    """
    fits = _product_fits(["", "q_arm", "is_swr"] + [""] * 9)

    out = aggregate_contrast(
        fits,
        [
            ("S4_value", "contrast", "is_swr:q_arm"),
            ("S4_value", "contrast", "is_swr"),
        ],
        level="rate",
        sd=_GATE_SD,
    ).set_index("term")

    # The product keeps every session: neither factor's separation touches it.
    assert out.loc["is_swr:q_arm", "n_sessions"] == 12
    assert out.loc["is_swr:q_arm", "n_dropped"] == 0
    assert out.loc["is_swr:q_arm", "coef_session_median"] == pytest.approx(0.65)
    # The ``is_swr`` term itself is still dropped where it separated.
    assert out.loc["is_swr", "n_sessions"] == 11
    assert out.loc["is_swr", "n_dropped"] == 1
    # The fit-level diagnostic still counts both flagged fits, for both terms.
    assert set(out["n_sessions_flagged_separated"]) == {2}


def test_product_term_is_dropped_when_the_product_itself_separates() -> None:
    """Separation ON the product is what drops the session for the product."""
    fits = _product_fits(["", "is_swr:q_arm", ""] + [""] * 9)

    out = aggregate_contrast(
        fits, [("S4_value", "contrast", "is_swr:q_arm")], level="rate", sd=_GATE_SD
    )

    assert out.iloc[0]["n_sessions"] == 11
    assert out.iloc[0]["n_dropped"] == 1
    # 0.2 is gone; the remaining eleven run 0.1, 0.3, 0.4, ..., 1.2 (median 0.7).
    assert out.iloc[0]["coef_session_median"] == pytest.approx(0.70)


def test_unrelated_separated_term_never_drops_a_session() -> None:
    """The nuisance/self-history terms that flagged 55% of fits are inert here."""
    fits = _two_term_fits()
    fits["separated"] = True
    fits["separated_terms"] = "hist_rcos_0|hist_rcos_1"

    out = aggregate_contrast(
        fits, [("S1_rew", "contrast", "reward_history_rcos_0")], level="rate"
    )

    assert out.iloc[0]["n_sessions"] == 8
    assert out.iloc[0]["n_dropped"] == 0
    assert out.iloc[0]["n_sessions_flagged_separated"] == 8


# ── Which standard error the audited row reports ──


def test_typical_session_se_prefers_the_event_clustered_se() -> None:
    """With a fraction response the model SE describes a design that no longer
    exists (one draw per event, none of the within-event mixing), so the
    event-clustered SE is the one the audited row must report."""
    fits = _session_fits(bse_cluster_rpe_postoutcome=[0.30] * 8)

    out = aggregate_contrast(fits, SPECS, level="rate", sd=SD)

    row = out.iloc[0]
    assert row["typical_session_se"] == pytest.approx(0.30)
    assert row["std_typical_session_se"] == pytest.approx(0.60)  # sd = 2.0
    assert row["se_source"] == "cluster"


def test_typical_session_se_falls_back_to_the_model_se() -> None:
    """Frames written before the clustered SE existed carry only ``bse_``."""
    fits = _session_fits()
    assert "bse_cluster_rpe_postoutcome" not in fits.columns

    out = aggregate_contrast(fits, SPECS, level="rate", sd=SD)

    row = out.iloc[0]
    assert row["typical_session_se"] == pytest.approx(0.10)
    assert row["se_source"] == "model"


def test_session_without_a_finite_clustered_se_uses_its_own_model_se() -> None:
    """A single non-finite sandwich must not void the whole row, nor be filled
    with a value from another session: that session reports its model SE and the
    row is labeled by the source the majority of sessions used."""
    cluster = [0.30] * 8
    cluster[0] = float("nan")
    fits = _session_fits(bse_cluster_rpe_postoutcome=cluster)

    out = aggregate_contrast(fits, SPECS, level="rate", sd=SD)

    row = out.iloc[0]
    # Seven clustered 0.30 and one model 0.10 -> median 0.30.
    assert row["typical_session_se"] == pytest.approx(0.30)
    assert row["se_source"] == "cluster"


def test_se_source_is_model_when_most_sessions_have_no_clustered_se() -> None:
    cluster = [float("nan")] * 8
    cluster[0] = 0.30
    fits = _session_fits(bse_cluster_rpe_postoutcome=cluster)

    out = aggregate_contrast(fits, SPECS, level="rate", sd=SD)

    row = out.iloc[0]
    # One clustered 0.30 and seven model 0.10 -> median 0.10.
    assert row["typical_session_se"] == pytest.approx(0.10)
    assert row["se_source"] == "model"


# ── Which SE the per-session significance count is read on ──


def test_pct_sig_session_is_read_on_the_clustered_se() -> None:
    """The count answers to the same SE ``typical_session_se`` reports.

    Coefficients 0.30..0.45 against a clustered SE of 0.20 give |z| between 1.0
    and 2.25, so only the two largest clear 1.96 -- while the model p-values
    say all eight are significant. Reporting the model count next to a
    clustered SE would describe two different designs in one row.
    """
    fits = _session_fits(bse_cluster_rpe_postoutcome=[0.20] * 8)

    row = aggregate_contrast(fits, SPECS, level="rate", sd=SD).iloc[0]

    assert row["se_source"] == "cluster"
    assert row["pct_sig_session"] == pytest.approx(2 / 8)


def test_pct_sig_session_falls_back_to_the_model_pvalue() -> None:
    """A frame with no SE column at all keeps the fitted p-value count."""
    fits = _session_fits().drop(columns=["bse_rpe_postoutcome"])
    fits.loc[fits.index[:3], "pval_rpe_postoutcome"] = 0.20

    row = aggregate_contrast(fits, SPECS, level="rate", sd=SD).iloc[0]

    assert row["se_source"] == "model"
    assert row["pct_sig_session"] == pytest.approx(5 / 8)


def test_pct_sig_session_uses_the_model_se_when_no_cluster_column_exists() -> None:
    """Legacy frames carry ``bse_`` only; the Wald test is read on that."""
    fits = _session_fits()
    assert "bse_cluster_rpe_postoutcome" not in fits.columns

    row = aggregate_contrast(fits, SPECS, level="rate", sd=SD).iloc[0]

    # SE 0.10 against coefficients 0.20-0.45: every |z| clears 1.96.
    assert row["se_source"] == "model"
    assert row["pct_sig_session"] == pytest.approx(1.0)


# ── The session signed-rank p and its correction across rungs ──


def test_a_rung_with_too_few_differences_is_left_out_of_the_correction() -> None:
    """Eight positive sessions give p = 2 / 2 ** 8. The second rung keeps six
    sessions, one with a coefficient of exactly zero: five differences, too few
    for a p, so the first rung's family is itself alone and its q is its p."""
    fits = _session_fits()
    second = _session_fits(model=["G6_rpe"] * 8).iloc[:6].copy()
    second.loc[second.index[0], "coef_rpe_postoutcome"] = 0.0
    specs = [*SPECS, ("G6_rpe", "non_local", "rpe_postoutcome")]

    out = aggregate_contrast(pd.concat([fits, second]), specs, level="rate", sd=SD)

    assert list(out["n_sessions"]) == [8, 6]
    assert out["session_signrank_p"].iloc[0] == pytest.approx(2 / 2**8)
    assert out["session_signrank_p"].isna().iloc[1]
    assert out["session_signrank_fdr_q"].iloc[0] == pytest.approx(2 / 2**8)
    assert out["session_signrank_fdr_q"].isna().iloc[1]


# ── Per-SD units come from the fit, not from a reader's own recomputation ──

_ARM_UNITS = {
    "reward_belief_mean_candidate_arm_preoutcome": 0.11898572753538102,
    "reward_fraction_candidate_arm_last10_visits": 0.29973092444891863,
    "task_reward_probability_candidate_arm": 0.23976509313715935,
}
_CONTRAST_FIGURE = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "figures"
    / "plot_swr_contrast_coefs.py"
)


def _provenance_file(tmp_path: Path, record: dict) -> Path:
    path = tmp_path / "swr_contrast_content_provenance.json"
    path.write_text(json.dumps(record))
    return path


def test_persisted_units_are_returned_and_used_verbatim(tmp_path: Path) -> None:
    """The reader's standardization factor IS the fit's frozen SD, to the bit."""
    path = _provenance_file(
        tmp_path,
        {"per_sd_scale": {"arm_cohort": _ARM_UNITS, "patch_cohort": {}}},
    )

    sd = load_per_sd_scale(path, "arm_cohort")

    assert sd == _ARM_UNITS
    for covariate, unit in _ARM_UNITS.items():
        assert std_factor(covariate, sd) == (unit, "per_SD")
        assert std_factor(f"swr_x_{covariate}", sd) == (unit, "per_SD")
    assert load_per_sd_scale(path, "patch_cohort") == {}


def test_a_record_without_the_frozen_units_is_refused(tmp_path: Path) -> None:
    """A fit in an unknown unit must fail, not fall back to a recomputed SD."""
    with pytest.raises(KeyError, match="per_sd_scale"):
        load_per_sd_scale(
            _provenance_file(tmp_path, {"per_sd_scale": {}}), "arm_cohort"
        )
    with pytest.raises(KeyError, match="per_sd_scale"):
        load_per_sd_scale(_provenance_file(tmp_path, {}), "arm_cohort")
    with pytest.raises(FileNotFoundError):
        load_per_sd_scale(tmp_path / "absent.json", "arm_cohort")


_RATE_UNITS = {
    "periswitch_index_trials": 11.012232609424599,
    "patch_switch_fraction_last5_completed_trips": 0.12789700553254071,
}


def test_rate_units_are_returned_verbatim(tmp_path: Path) -> None:
    """The rate meta's frozen SDs are the reader's factors, to the bit."""
    path = tmp_path / "swr_contrast_rate_meta.json"
    path.write_text(json.dumps({"fit_time_sd_reference": _RATE_UNITS}))

    sd = contrast_aggregation.load_fit_time_sd_reference(path)

    assert sd == _RATE_UNITS
    for covariate, unit in _RATE_UNITS.items():
        assert std_factor(covariate, sd) == (unit, "per_SD")
        assert std_factor(f"is_swr:{covariate}", sd) == (unit, "per_SD")


@pytest.mark.parametrize("record", [{}, {"fit_time_sd_reference": {}}])
def test_a_rate_meta_without_the_frozen_units_is_refused(
    record, tmp_path: Path
) -> None:
    path = tmp_path / "swr_contrast_rate_meta.json"
    path.write_text(json.dumps(record))

    with pytest.raises(KeyError, match="fit_time_sd_reference"):
        contrast_aggregation.load_fit_time_sd_reference(path)


def test_an_absent_rate_meta_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        contrast_aggregation.load_fit_time_sd_reference(tmp_path / "absent.json")


def test_the_contrast_figure_takes_every_unit_from_the_fits(
    replay_content_root,
) -> None:
    """No SD is recomputed in the figure, for either ladder.

    A recomputed per-arm unit pools over all nine candidate arms (the trunks are
    structural zeros), 2.1x the audited per-leaf-arm one; a recomputed rate unit
    pools over all stops rather than the at-well cohort the rate fit names, up
    to ~7 % off the audited table's.
    """
    source = _CONTRAST_FIGURE.read_text()
    tree = ast.parse(source)
    called = {
        getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }

    assert {"load_per_sd_scale", "load_fit_time_sd_reference"} <= called
    assert not called & {"_global_sd", "nanstd", "std", "read_table"}
    from scripts.figures import plot_swr_contrast_coefs as figure

    assert figure.fit_artifact_paths()["rate_meta"] == (
        replay_content_root / "models_per_bin_rate" / "swr_contrast_rate_meta.json"
    )


# --- Dirichlet-multinomial concentration terms ---------------------------------


def test_concentration_terms_are_per_category() -> None:
    """``log_phi_is_swr`` is a coefficient on a 0/1 flag, not on an SD of one."""
    assert std_factor("log_phi_is_swr", SD) == (1.0, "per_category")
    assert std_factor("log_phi_intercept", SD) == (1.0, "per_category")
    # Even if a caller hands in an SD for the flag, the per-category rule wins.
    assert std_factor("log_phi_is_swr", {"log_phi_is_swr": 0.5}) == (
        1.0,
        "per_category",
    )


def _dm_rows(n_sessions: int = 6) -> pd.DataFrame:
    """Per-session Dirichlet-multinomial fits: model SEs, no clustered column."""
    return pd.DataFrame(
        {
            "model": ["C0"] * n_sessions,
            "population": ["all"] * n_sessions,
            "animal": [f"a{index % 3}" for index in range(n_sessions)],
            "coef_log_phi_is_swr": [-0.8 - 0.01 * index for index in range(n_sessions)],
            "bse_log_phi_is_swr": [0.2] * n_sessions,
            "pval_log_phi_is_swr": [0.01] * n_sessions,
            "separated_terms": [""] * n_sessions,
            "converged": [True] * n_sessions,
            "separated": [False] * n_sessions,
        }
    )


def test_dm_rows_report_the_model_se_they_actually_carry() -> None:
    """The DM SE is already event-level, so there is no sandwich to prefer."""
    rows = _dm_rows()
    se, source = session_se(rows, "log_phi_is_swr")

    assert source == "model"
    assert se.tolist() == [0.2] * 6
    assert typical_session_se(rows, "log_phi_is_swr") == (0.2, "model")

    aggregated = aggregate_contrast(rows, [("C0", "all", "log_phi_is_swr")], "arm")
    assert len(aggregated) == 1
    row = aggregated.iloc[0]
    assert row["se_source"] == "model"
    assert row["typical_session_se"] == pytest.approx(0.2)
    assert row["std_unit"] == "per_category"
    assert row["std_factor"] == 1.0
    assert row["std_coef_session_median"] == pytest.approx(row["coef_session_median"])


# --- The per-term session gate, shared by the aggregation and the figure -------


def _gate_fits(n_kept: int, n_separated: int) -> pd.DataFrame:
    """``n_kept`` identified sessions and ``n_separated`` flagged as separated."""
    n = n_kept + n_separated
    return pd.DataFrame(
        {
            "model": "CS1_current",
            "population": "all",
            "animal": [f"a{i % 5}" for i in range(n)],
            "coef_swr_x_is_current_arm": [0.1 * (i + 1) for i in range(n)],
            "separated_terms": [""] * n_kept + ["swr_x_is_current_arm"] * n_separated,
        }
    )


@pytest.mark.parametrize(
    ("n_kept", "n_separated", "refusal"),
    [
        (8, 2, None),  # 20 % lost: at the limit, emitted
        (8, 3, "unidentified_share"),  # 27 % lost
        (MIN_SESSIONS_PER_TERM - 1, 0, "too_few_sessions"),
    ],
)
def test_the_gate_refuses_exactly_what_the_audited_table_omits(
    n_kept, n_separated, refusal
) -> None:
    fits = _gate_fits(n_kept, n_separated)
    gate = gate_term_sessions(fits, "swr_x_is_current_arm")
    audited = aggregate_contrast(
        fits, [("CS1_current", "all", "swr_x_is_current_arm")], "candidate_arm"
    )

    assert gate.refusal == refusal
    assert gate.n_kept == n_kept and gate.n_dropped == n_separated
    assert (gate.drop_share > MAX_UNIDENTIFIED_SHARE) == (
        refusal == "unidentified_share"
    )
    assert audited.empty == (refusal is not None)
    if refusal is None:
        assert audited.loc[0, "n_sessions"] == gate.n_kept


def _assembly_tables(duplicated: str) -> dict[str, pd.DataFrame]:
    ev = pd.DataFrame({"event_id": [1, 2], "stop_id": [10, 11]})
    stop = pd.DataFrame(
        {
            "stop_id": [10, 11],
            "known_reward_outcome_lag_1": [1.0, 0.0],
            "patch_switch_fraction_last5_completed_trips": [0.1, 0.2],
        }
    )
    q = pd.DataFrame(
        {
            "stop_id": [10, 11],
            **{
                c: [0.0, 1.0]
                for c in (
                    "rpe_postoutcome",
                    "relative_value_switch_minus_stay_preoutcome",
                    "reward_belief_differential_entropy_leaf_mean_preoutcome",
                    "reward_belief_mean_update_l1_all_ports_postoutcome",
                    "periswitch_index_trials",
                    "is_periswitch_post",
                    "trials_since_block_change",
                )
            },
        }
    )
    tables = {"event_arm": ev, "stop": stop, "qvals": q}
    if duplicated:
        frame = tables[duplicated]
        tables[duplicated] = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    return tables


@pytest.mark.parametrize("duplicated", ["", "stop", "qvals"])
def test_arm_assembly_refuses_a_duplicated_stop_id(
    duplicated, tmp_path: Path, monkeypatch
) -> None:
    """A duplicated stop would silently duplicate the event-arm rows it joins."""
    tables = _assembly_tables(duplicated)
    ca = contrast_aggregation
    monkeypatch.setattr(ca, "read_table", lambda key: tables[key].copy())
    for name in (
        "require_covariate_schema",
        "require_swr_source",
        "assert_standardized_matches_reference",
        "load_reference",
        "add_arm_fixed_effects",
        "_add_swr_arm_fixed_effects",
        "validate_uncertainty_inputs",
        "_cohort_log",
    ):
        monkeypatch.setattr(ca, name, lambda *a, **k: None)
    monkeypatch.setattr(ca, "select_event_cohort", lambda f, cohort: (f, {}))
    monkeypatch.setattr(ca, "_attach_arm_design_columns", lambda f: (f, {}))
    reference = tmp_path / "standardization_reference.json"
    reference.write_text("{}")
    fit_inputs = {
        "event_arm_table": "event_arm",
        "stop_table": "stop",
        "qvals_table": "qvals",
    }

    if duplicated:
        with pytest.raises(pd.errors.MergeError, match="many-to-one"):
            ca.assemble_arm(fit_inputs, reference)
    else:
        ev, _ = ca.assemble_arm(fit_inputs, reference)
        assert len(ev) == 2


def test_arm_assembly_flags_the_stops_with_a_qvals_row(
    tmp_path: Path, monkeypatch
) -> None:
    """The qvals join says which stops it matched; the audit books the gap from it."""
    tables = _assembly_tables("")
    tables["qvals"] = tables["qvals"].iloc[:1]  # stop 11 has no qvals row
    ca = contrast_aggregation
    monkeypatch.setattr(ca, "read_table", lambda key: tables[key].copy())
    for name in (
        "require_covariate_schema",
        "require_swr_source",
        "assert_standardized_matches_reference",
        "load_reference",
        "add_arm_fixed_effects",
        "_add_swr_arm_fixed_effects",
        "validate_uncertainty_inputs",
        "_cohort_log",
    ):
        monkeypatch.setattr(ca, name, lambda *a, **k: None)
    monkeypatch.setattr(ca, "select_event_cohort", lambda f, cohort: (f, {}))
    monkeypatch.setattr(ca, "_attach_arm_design_columns", lambda f: (f, {}))
    reference = tmp_path / "standardization_reference.json"
    reference.write_text("{}")

    ev, _ = ca.assemble_arm(
        {"event_arm_table": "event_arm", "stop_table": "stop", "qvals_table": "qvals"},
        reference,
    )

    assert ev["has_qvals_row"].dtype == bool
    assert ev.set_index("stop_id")["has_qvals_row"].to_dict() == {10: True, 11: False}
