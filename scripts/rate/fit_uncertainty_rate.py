#!/usr/bin/env python
"""Fit the per-bin replay rate vs behavioral-uncertainty / value, by population.

Tests whether non-local replay *rate* is differentially modulated by the
animal's behavioral uncertainty and the task's nominal value structure across
SWR-coupled vs non-SWR non-local events, controlling for the within-stop time
course, raised-cosine self-history, the reward kernel, and the recent reward
rate. Rate-only -- replay direction is out of scope.

Per session: load ``stop_table``; validate its canonical ``recent_switch_rate``
and derive stop-local summaries (:func:`add_uncertainty_covariates`); inner-merge
the nominal value + real
behavioral-model covariates from ``qvals_table.pkl`` on ``stop_id`` (one exactly
poke_out-aligned source -- no difflib CSV); drop the first (exposure) session per
animal; attach the canonical ``is_swr`` to the onsets.
Because the fitted block contains current-trip RPE/value updates, retain only
physical response bins whose left edge is at or after ``outcome_known_time``
(reward-on for rewarded trials; conservatively poke-out for omissions).
Per population in ``{swr, non_local_swr, non_local_no_swr}`` build the per-bin
design (carrying the uncertainty + value covariates as ``stop_covariates``) and
fit a single spec -- ``MU_uncertainty`` -- instantiated HERE (not added to
``MODEL_PER_BIN_RATE`` / ``all_models.pkl``). The spec is M7r (time spline +
raised-cosine self-history + reward kernel + reward rate) plus the
uncertainty/value block, so the block is tested *beyond* those confounds.

Collect one tidy row per (session, population) with every covariate's
``coef_`` / ``bse_`` / ``pval_`` / ``rr_`` ->
``Processed-Data/replay_content/models_per_bin_rate/uncertainty_rate.pkl``.

Usage::

    uv run python scripts/rate/fit_uncertainty_rate.py                    # full
    uv run python scripts/rate/fit_uncertainty_rate.py --max-per-animal 1 # smoke
"""

from __future__ import annotations

import os

# Pin BLAS to one thread BEFORE numpy import (matches the Family B orchestrator).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from continuum_swr_replay.analysis.replay import (  # noqa: E402
    BIN_S,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    N_REWARD_LAGS,
    STOP_COVARIATES,
    TMAX,
    PerBinRateModelSpec,
    add_uncertainty_covariates,
    aggregate_to_timebins,
    attach_is_swr_to_onsets,
    fit_per_bin_rate_model,
)
from continuum_swr_replay.analysis.replay._helpers import POPULATIONS  # noqa: E402
from continuum_swr_replay.analysis.replay.cohort_audit import (  # noqa: E402
    cohort_record,
    stop_own_missing,
    write_cohort_audit,
)
from continuum_swr_replay.analysis.replay.covariate_timing import (  # noqa: E402
    assert_predictor_timing,
)
from continuum_swr_replay.analysis.replay.covariates import (  # noqa: E402
    CANONICAL_SWITCH_WINDOW,
)
from continuum_swr_replay.analysis.replay.per_bin_rate_model import (  # noqa: E402
    _design_columns,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR  # noqa: E402

# separated history coefs legitimately overflow exp(); convergence/separation are
# recorded per fit as flags in per_bin_rate_model (analysis.replay._fit_diagnostics),
# so only the benign RuntimeWarning is silenced (not a blanket ignore).
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SESSION_COLS = ["animal", "day", "epoch"]
RC = PROCESSED_DATA_DIR / "replay_content"

# Nominal schedule covariates, now sourced from qvals_table (derived from Alison's
# p_rew_leaf/contingency_count, exactly poke_out-aligned) rather than the difflib
# contingency_table_csv -- so every covariate the model fits shares one alignment.
VALUE_COVARIATES = (
    "p_reward_current",
    "value_diff_patch",
    "trials_since_block_change",
)

# Nominal / model-free block (per-unit; these covariates are O(1) and are the
# task schedule + raw behavior, NOT the behavioral model). We prefer the real
# model over stand-ins, so the leaky-Q q_current is replaced by the model DeltaV
# (switch_advantage) and the nominal rpe by rpe_model below. value_diff_patch is
# KEPT: it is the nominal best-minus-second patch value from the schedule and is
# orthogonal to the model DeltaV (r=-0.04), so it carries independent information
# (an earlier drop assuming redundancy was wrong). Excluded for collinearity:
# wsls_deviation (r=-0.70 with rpe), omission_streak (r=-0.66 with reward_lag1).
CURATED_COVARIATES = (
    "p_reward_current",
    "value_diff_patch",
    "recent_switch_rate",
    "trials_since_block_change",
)

# Real behavioral-model reward-prediction error, used INSTEAD of the nominal rpe
# stand-in (rpe_model vs old nominal rpe: r=0.71 -- the model version matters).
# q_chosen (model chosen-port value) is deliberately NOT fit: it is r=0.90
# collinear with the DeltaV decision variable (switch_advantage), which carries
# the model's value signal here. O(1) scale, so per-unit like the nominal block.
MODEL_VALUE_COVARIATES = (
    "rpe_model",  # reward - q_chosen (model reward-prediction error)
)

# Real SpatialBanditTask decision variables (qvals_table.pkl, beta_stable_withleaf;
# see docs/behavior_model_covariates.md). The paper's covariates: DeltaV, belief
# entropy, absolute value-update magnitudes. z-scored below (disparate/tiny scale)
# -> per-SD rate ratios.
RL_COVARIATES = (
    "switch_advantage_mu_preoutcome_sess",  # DeltaV = Switch - Stay value
    "beta_reward_entropy",  # belief uncertainty (mean per-leaf Beta differential entropy)
    "global_mu_update_abs_postoutcome_sess",  # |value update| summed over all ports
    "unchosen_patches_mu_update_abs_postoutcome_sess",  # |update| over alternative patches
)

# What the model actually fits + reports (nominal + model value + RL DVs).
FITTED_COVARIATES = (*CURATED_COVARIATES, *MODEL_VALUE_COVARIATES, *RL_COVARIATES)

# The three non-local-vs-SWR populations to dissociate (pooled non_local is the
# null and is not fit here).
POPULATIONS_TO_FIT = ("swr", "non_local_swr", "non_local_no_swr")

OUTCOME_TIME_SEMANTICS = "reward_on_ts for rewarded trials; poke_out_ts for omissions"
POST_OUTCOME_GATE_RULE = "stop_start + binj * BIN_S >= outcome_known_time"

# M7r + the curated uncertainty/value block, instantiated in the runner so the
# shipped MODEL_PER_BIN_RATE ladder / all_models.pkl schema is untouched.
# reward_rate is OFF: reward_rate_trial_k10 (the reward_rate covariate) is exactly the mean of
# reward_lag1..10 (corr 1.000, VIF inf), so including it with the reward kernel
# makes the reward block rank-deficient; the kernel alone carries reward history.
MU_UNCERTAINTY = PerBinRateModelSpec(
    "MU_uncertainty",
    True,  # time spline
    N_REWARD_LAGS,  # reward kernel
    False,  # recent reward rate OFF (perfectly collinear with the kernel)
    N_HISTORY_LAGS,
    False,  # no switch term (recent_switch_rate carries the switch signal)
    False,  # no reward x time interaction
    "curated uncertainty/value beyond time/reward/history",
    "M7r_time_histcos_rew",
    self_history_basis=N_HISTORY_BASIS,
    extra_covariates=FITTED_COVARIATES,
)


def _drop_first_session_per_animal(stop_table: pd.DataFrame) -> pd.DataFrame:
    """Drop each animal's first (exposure) session by (day, epoch) order."""
    first = (
        stop_table[SESSION_COLS]
        .drop_duplicates()
        .sort_values(SESSION_COLS)
        .groupby("animal", sort=False)
        .head(1)
    )
    merged = stop_table.merge(first, on=SESSION_COLS, how="left", indicator=True)
    return stop_table[merged["_merge"].to_numpy() == "left_only"]


def _subset_max_per_animal(stop_table: pd.DataFrame, n: int) -> pd.DataFrame:
    keep = (
        stop_table[SESSION_COLS]
        .drop_duplicates()
        .sort_values(SESSION_COLS)
        .groupby("animal", sort=False)
        .head(n)
    )
    return stop_table.merge(keep, on=SESSION_COLS)


def _gate_bins_after_outcome(
    design: pd.DataFrame,
    *,
    bin_s: float = BIN_S,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Keep fit-complete bins whose left edge is at/after the current outcome.

    ``binj`` is zero-based, so a bin's physical observation time is
    ``stop_start + binj * bin_s``. Equality is eligible: information known at
    the boundary can be associated with responses beginning at that boundary.
    Non-finite timing rows are counted separately and never pass the gate. The
    outcome-balance guard is evaluated only after applying the same listwise
    column selection and positive-exposure filter as
    :func:`fit_per_bin_rate_model`.
    """
    required = (
        "stop_id",
        "stop_start",
        "binj",
        "outcome_known_time",
        "trial_reward",
    )
    missing = [column for column in required if column not in design.columns]
    if missing:
        raise KeyError(
            "Post-outcome uncertainty-rate gating requires physical timing "
            f"column(s): {missing}. Regenerate qvals_table.pkl and stop_table.pkl."
        )

    timing = design.groupby("stop_id", sort=False)[
        ["stop_start", "outcome_known_time", "trial_reward"]
    ]
    inconsistent = timing.nunique(dropna=False).gt(1).any(axis=1)
    if inconsistent.any():
        bad_ids = tuple(inconsistent[inconsistent].index)
        raise ValueError(
            "Post-outcome uncertainty-rate gating requires stop_start, "
            "outcome_known_time, and trial_reward to be constant within stop_id; "
            "invalid stop_id(s): "
            f"{bad_ids}"
        )

    fit_columns = [*_design_columns(MU_UNCERTAINTY), "exposure"]
    missing_fit = [column for column in fit_columns if column not in design.columns]
    if missing_fit:
        raise KeyError(
            "Post-outcome uncertainty-rate gating requires exact fit column(s): "
            f"{missing_fit}"
        )

    output = design.copy()
    stop_start = output["stop_start"].to_numpy(dtype=float)
    binj = output["binj"].to_numpy(dtype=float)
    boundary = output["outcome_known_time"].to_numpy(dtype=float)
    trial_reward = output["trial_reward"].to_numpy(dtype=float)
    if (
        not np.isfinite(trial_reward).all()
        or not np.isin(trial_reward, (0.0, 1.0)).all()
    ):
        raise ValueError("trial_reward must contain finite binary 0/1 values")
    response_time = stop_start + binj * float(bin_s)
    finite = np.isfinite(response_time) & np.isfinite(boundary)
    eligible = finite & (response_time >= boundary)
    output["response_time"] = response_time
    # Match fit_per_bin_rate_model exactly: listwise drop over formula columns
    # plus exposure, followed by exposure > 0. Keeping this selection here
    # prevents the sole row class from disappearing inside the fitter after an
    # apparently balanced physical gate.
    # Treat +/-inf as unavailable too. Statsmodels may fail deep in fitting or
    # emit non-finite estimates if an infinite design value is allowed through;
    # the scientific complete-case contract is finite, not merely non-null.
    listwise_complete = np.isfinite(output[fit_columns].to_numpy(dtype=float)).all(
        axis=1
    )
    positive_exposure = output["exposure"].to_numpy(dtype=float) > 0
    fit_complete = listwise_complete & positive_exposure
    candidate = eligible & fit_complete
    rewarded = trial_reward == 1.0
    omitted = trial_reward == 0.0
    reward_after = int((eligible & rewarded).sum())
    omission_after = int((eligible & omitted).sum())
    reward_complete = int((candidate & rewarded).sum())
    omission_complete = int((candidate & omitted).sum())
    outcome_balance_valid = reward_complete > 0 and omission_complete > 0
    if outcome_balance_valid:
        balance_status = "both_outcomes_retained"
    else:
        missing = []
        if reward_complete == 0:
            missing.append("rewarded")
        if omission_complete == 0:
            missing.append("omission")
        balance_status = "missing_after_complete_case:" + ",".join(missing)
    audit = {
        "n_bins_before_post_outcome_gate": int(len(output)),
        "n_bins_after_post_outcome_gate": int(eligible.sum()),
        "n_rewarded_bins_before_post_outcome_gate": int(rewarded.sum()),
        "n_rewarded_bins_after_post_outcome_gate": reward_after,
        "n_omission_bins_before_post_outcome_gate": int(omitted.sum()),
        "n_omission_bins_after_post_outcome_gate": omission_after,
        "n_bins_before_complete_case": int(eligible.sum()),
        "n_rewarded_bins_before_complete_case": reward_after,
        "n_omission_bins_before_complete_case": omission_after,
        "n_bins_after_complete_case": int(candidate.sum()),
        "n_rewarded_bins_after_complete_case": reward_complete,
        "n_omission_bins_after_complete_case": omission_complete,
        "n_bins_excluded_missing_timing": int((~finite).sum()),
        "n_bins_excluded_pre_outcome": int((finite & ~eligible).sum()),
        "n_bins_excluded_incomplete_fit": int((eligible & ~listwise_complete).sum()),
        "n_bins_excluded_nonpositive_exposure": int(
            (eligible & listwise_complete & ~positive_exposure).sum()
        ),
        "outcome_balance_valid": outcome_balance_valid,
        "outcome_balance_status": balance_status,
        "n_bins_after_outcome_balance_guard": int(candidate.sum())
        if outcome_balance_valid
        else 0,
    }
    # A one-outcome survivor set changes the estimand and makes current-outcome
    # terms uninterpretable. Keep the audit, but do not fit that cell.
    selected = candidate if outcome_balance_valid else np.zeros(len(output), dtype=bool)
    return output.loc[selected].reset_index(drop=True), audit


def fit_uncertainty_rate(
    stop_table: pd.DataFrame,
    onsets: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (session, population): MU_uncertainty fit coefficients.

    ``stop_table`` must already carry the uncertainty + value covariates and the
    reward lags; ``onsets`` must already carry ``is_swr``.
    """
    assert_predictor_timing(FITTED_COVARIATES, retrospective=True)
    onset_groups = dict(tuple(onsets.groupby(SESSION_COLS, sort=False)))
    stop_covariates = (
        *STOP_COVARIATES,
        *FITTED_COVARIATES,
        "stop_start",
        "outcome_known_time",
        "trial_reward",
    )

    rows: list[dict] = []
    gate_audit_rows: list[dict] = []
    attempted = dict.fromkeys(POPULATIONS_TO_FIT, 0)
    n_unfit = dict.fromkeys(POPULATIONS_TO_FIT, 0)
    for keys, stops_g in stop_table.groupby(SESSION_COLS, sort=False):
        if keys not in onset_groups:
            continue
        onsets_g = onset_groups[keys]
        for outcome in POPULATIONS_TO_FIT:
            attempted[outcome] += 1
            design = aggregate_to_timebins(
                stops_g,
                POPULATIONS[outcome](onsets_g),
                bin_s=BIN_S,
                tmax=TMAX,
                reward_lags=N_REWARD_LAGS,
                self_history_lags=N_HISTORY_LAGS,
                self_history_basis=N_HISTORY_BASIS,
                stop_covariates=stop_covariates,
            )
            design, gate_audit = _gate_bins_after_outcome(design)
            gate_audit_rows.append(
                {
                    **dict(zip(SESSION_COLS, keys, strict=True)),
                    "outcome": outcome,
                    "post_outcome_gate": True,
                    "post_outcome_gate_rule": POST_OUTCOME_GATE_RULE,
                    "outcome_time_semantics": OUTCOME_TIME_SEMANTICS,
                    **gate_audit,
                }
            )
            if not gate_audit["outcome_balance_valid"]:
                n_unfit[outcome] += 1
                logger.warning(
                    "  %s / %s: skipped (%s)",
                    "_".join(map(str, keys)),
                    outcome,
                    gate_audit["outcome_balance_status"],
                )
                continue
            fit = fit_per_bin_rate_model(design, MU_UNCERTAINTY)
            if fit is None:
                n_unfit[outcome] += 1
                logger.info("  %s / %s: unfit", "_".join(map(str, keys)), outcome)
                continue
            rows.append(
                {
                    "outcome": outcome,
                    "model": MU_UNCERTAINTY.name,
                    "post_outcome_gate": True,
                    "post_outcome_gate_rule": POST_OUTCOME_GATE_RULE,
                    "outcome_time_semantics": OUTCOME_TIME_SEMANTICS,
                    **gate_audit,
                    **dict(zip(SESSION_COLS, keys, strict=True)),
                    **{k: v for k, v in fit.items() if k != "predicted"},
                }
            )
    # Surface survivor bias: if many (session x population) cells failed to fit,
    # the median RRs for that population come from a biased survivor subset.
    for outcome in POPULATIONS_TO_FIT:
        att, bad = attempted[outcome], n_unfit[outcome]
        if att and bad:
            frac = bad / att
            (logger.warning if frac > 0.2 else logger.info)(
                "population %s: %d/%d session cells unfit (%.0f%%)",
                outcome,
                bad,
                att,
                100 * frac,
            )
    result = pd.DataFrame(rows)
    result.attrs["post_outcome_gate_audit"] = gate_audit_rows
    return result


def _cohort_records(
    models: pd.DataFrame,
    stop_fitted: pd.DataFrame,
    *,
    n_postjoin: int,
    n_sessions_postjoin: int,
    qvals_gap: int,
) -> list[dict]:
    """Stop-grain cohort record per population for the MU-uncertainty ladder.

    ``n_in = n_postjoin + qvals_gap`` (== pre-join stops on a full run); the inner
    qvals-join drop is the mechanism-B ``n_drop_qvals_gap``; ``immobility<=0`` / NaN
    stop-level predictors on ``stop_fitted`` are ``n_drop_own_predictor_missing``; the
    first-session-per-animal drop, subset, and per-bin gates fall into
    ``n_excluded_session``. ``sessions_in`` is the post-join (pre-first-drop) session
    count so it matches ``n_in``'s stop population. Reconciles by construction.
    """
    stop_level = [
        c for c in _design_columns(MU_UNCERTAINTY) if c in stop_fitted.columns
    ]
    n_own = int(stop_own_missing(stop_fitted, stop_level).sum())
    records = []
    for outcome in POPULATIONS_TO_FIT:
        sub = (
            models[(models.model == MU_UNCERTAINTY.name) & (models.outcome == outcome)]
            if len(models)
            else models
        )
        n_modeled = int(sub["n_stops"].sum()) if "n_stops" in sub else 0
        fit_keys = (
            {tuple(r) for r in sub[SESSION_COLS].to_numpy()} if len(sub) else set()
        )
        records.append(
            cohort_record(
                MU_UNCERTAINTY.name,
                outcome,
                "stop",
                n_in=n_postjoin + int(qvals_gap),
                n_modeled=n_modeled,
                sessions_in=n_sessions_postjoin,
                sessions_modeled=len(fit_keys),
                animals_modeled=len({k[0] for k in fit_keys}),
                drop_notes=(
                    "qvals inner-join -> qvals_gap (mechanism B); immobility<=0 / NaN "
                    "stop-level predictor -> own; first-session/subset/physical "
                    "post-outcome bin gate -> excluded"
                ),
                n_drop_qvals_gap=int(qvals_gap),
                n_drop_own_predictor_missing=n_own,
                n_excluded_session=n_postjoin - n_own - n_modeled,
                n_sessions_flagged_nonconverged=(
                    int((sub["converged"] == False).sum()) if "converged" in sub else 0  # noqa: E712
                ),
                n_sessions_flagged_separated=(
                    int((sub["separated"] == True).sum()) if "separated" in sub else 0  # noqa: E712
                ),
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-animal",
        type=int,
        default=None,
        help="Smoke mode: keep only the first N (post-exposure) sessions per animal.",
    )
    parser.add_argument(
        "--switch-window",
        type=int,
        choices=(CANONICAL_SWITCH_WINDOW,),
        default=CANONICAL_SWITCH_WINDOW,
        help=(
            "Completed-trial window for recent_switch_rate. The canonical "
            "tables currently support only 5; other values require rebuilding "
            "from the full trial stream."
        ),
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    stop = pd.read_pickle(RC / "stop_table.pkl")
    onsets = attach_is_swr_to_onsets(
        pd.read_pickle(RC / "event_onset_table.pkl"),
        pd.read_pickle(RC / "event_table.pkl"),
    )

    # add_uncertainty_covariates validates the upstream recent_switch_rate (the
    # only model-free covariate still fit) and derives stop-local summaries. Its
    # leaky_q_value_diff / omission_streak / wsls_deviation stand-ins are no
    # longer used here.
    stop = add_uncertainty_covariates(stop, switch_window=args.switch_window)
    # Nominal schedule + real model covariates, ALL from qvals_table -- one
    # exactly poke_out-aligned source (retires the difflib contingency_table_csv
    # for this model). qvals' `rpe` (R - q_chosen) -> rpe_model. Inner join: keep
    # only stops mapped to a trial (post-final-trial tails ~1-3% drop out); model
    # DVs are NaN on the first trial of a session -> listwise-dropped at fit.
    qvals = pd.read_pickle(RC / "models_per_bin_rate" / "qvals_table.pkl")
    qcols = qvals[
        [
            "stop_id",
            "outcome_known_time",
            "trial_reward",
            *VALUE_COVARIATES,
            "rpe",
            *RL_COVARIATES,
        ]
    ].rename(columns={"rpe": "rpe_model"})
    n_before = len(stop)
    stop = stop.merge(qcols, on="stop_id", how="inner", validate="many_to_one")
    n_postjoin = len(stop)  # inner-join (qvals-gap, mechanism B) drop = n_before - this
    # Session count BEFORE the first-session-per-animal drop, so the audit's
    # sessions_in matches n_in's stop population (n_in includes the first-session
    # stops as excluded, not removed).
    n_sessions_postjoin = stop.groupby(SESSION_COLS).ngroups
    frac = len(stop) / n_before
    if frac < 0.5:
        raise ValueError(
            f"qvals join kept only {frac:.1%} of stops -- check qvals_table stop_id convention"
        )
    logger.info(
        "qvals join: %d -> %d stops (%.0f%% kept)", n_before, len(stop), 100 * frac
    )
    stop = _drop_first_session_per_animal(stop)
    if args.max_per_animal is not None:
        stop = _subset_max_per_animal(stop, args.max_per_animal)
    # z-score the RL decision variables AFTER the drop/subset so the per-SD scale
    # is the SD of exactly the rows that get fit (disparate/tiny scale: DeltaV
    # ~0.2, updates ~0.05). The nominal block + rpe_model stay per-unit.
    for c in RL_COVARIATES:
        frac_valid = stop[c].notna().mean()
        if frac_valid < 0.7:
            logger.warning(
                "RL covariate %r is non-NaN on only %.0f%% of fitted rows; its SD "
                "(hence the per-SD rate ratio) is set by that minority and rows with "
                "NaN drop from every model listwise",
                c,
                100 * frac_valid,
            )
        sd = stop[c].std()
        if pd.isna(sd) or sd == 0:
            raise ValueError(
                f"RL covariate {c!r} has degenerate SD ({sd}); cannot z-score"
            )
        stop[c] = (stop[c] - stop[c].mean()) / sd

    n_sessions = stop.groupby(SESSION_COLS).ngroups
    logger.info(
        "Fitting %s over %d sessions x %s populations (BIN_S=%.3f, "
        "switch_window=%d)...",
        MU_UNCERTAINTY.name,
        n_sessions,
        list(POPULATIONS_TO_FIT),
        BIN_S,
        args.switch_window,
    )
    models = fit_uncertainty_rate(stop, onsets)
    gate_audit = pd.DataFrame(models.attrs["post_outcome_gate_audit"])

    out_dir = RC / "models_per_bin_rate"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (
        "uncertainty_rate_smoke.pkl" if args.max_per_animal else "uncertainty_rate.pkl"
    )
    csfx = "_smoke" if args.max_per_animal else ""
    out = out_dir / fname
    models.to_pickle(out)
    gate_audit.to_csv(
        out_dir / f"uncertainty_rate_post_outcome_gate_audit{csfx}.csv",
        index=False,
    )

    # Cohort audit (stop grain). The inner qvals join is the mechanism-B qvals-gap;
    # immobility<=0 / NaN stop-level predictors on the fitted frame -> own; the
    # first-session-per-animal drop, subset, and per-bin gates -> excluded.
    audit_qvals = 0 if args.max_per_animal else (n_before - n_postjoin)  # smoke: N/A
    audit_records = _cohort_records(
        models,
        stop,
        n_postjoin=n_postjoin,
        n_sessions_postjoin=n_sessions_postjoin,
        qvals_gap=audit_qvals,
    )
    write_cohort_audit(
        audit_records, out_dir / f"uncertainty_rate_cohort_audit{csfx}.csv"
    )
    if models.empty:
        raise RuntimeError(
            "No uncertainty-rate cell retained both rewarded and omission bins "
            "after the physical post-outcome and complete-case gates; audits "
            "were written, but a one-outcome survivor model was not fit."
        )
    logger.info(
        "Saved %d rows (%d sessions, populations=%s) to %s in %.1fs",
        len(models),
        models.groupby(SESSION_COLS).ngroups if len(models) else 0,
        sorted(models["outcome"].unique()) if len(models) else [],
        out,
        time.perf_counter() - t0,
    )
    if len(models):
        for cov in FITTED_COVARIATES:
            col = f"rr_{cov}"
            if col not in models:
                continue
            summary = models.groupby("outcome")[col].median().to_dict()
            logger.info("median RR  %-26s %s", cov, summary)


if __name__ == "__main__":
    main()
