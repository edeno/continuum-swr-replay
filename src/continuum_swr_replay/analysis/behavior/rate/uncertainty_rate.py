"""The per-bin replay rate vs behavioral uncertainty and value, by population.

``MU_uncertainty`` (``model_specs.MODEL_UNCERTAINTY_RATE``: M7r plus the
uncertainty/value block) is fit per session to the detected-SWR, non-local SWR
and non-local no-SWR onsets, on the at-risk bins at or after each stop's
outcome is known (:func:`fit_uncertainty_rate`). The stop table it is fit on is
the stop cohort joined to the ``qvals_table`` value covariates
(:func:`join_uncertainty_qvals`), without the register's first-exposure
sessions (:func:`drop_exposure_sessions`), with the RL decision variables
z-scored over the fitted rows (:func:`standardize_rl_covariates`).
:func:`uncertainty_cohort_records` is its stop-grain cohort audit.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import OUTCOME_TIME_SEMANTICS
from continuum_swr_replay.data_loaders.epoch_register import first_exposure_epochs
from continuum_swr_replay.decoding.identifiers import parse_nwb_filename
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS
from continuum_swr_replay.tables.standardization import Reference, attach_standardized

from ..cohort_audit import fitted_cohort_record, stop_own_missing
from ..covariate_timing import assert_predictor_timing
from ..model_specs import (
    BIN_S,
    MODEL_UNCERTAINTY_RATE,
    MU_UNCERTAINTY_COVARIATES,
    MU_UNCERTAINTY_RL_COVARIATES,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    PER_BIN_DESIGN,
    STOP_COVARIATES,
    TMAX,
    longest_history_window_s,
)
from ..per_bin_design import POPULATIONS, aggregate_to_timebins
from .duration_aware_history import at_risk_events
from .per_bin_rate_model import (
    _design_columns,
    fit_per_bin_rate_model,
    freeze_per_bin_cohort,
)
from .post_outcome import POST_OUTCOME_GATE_RULE, gate_bins_after_outcome

logger = logging.getLogger(__name__)

# The model_specs blocks the uncertainty rate fit fits and standardizes.
FITTED_COVARIATES = MU_UNCERTAINTY_COVARIATES
RL_COVARIATES = MU_UNCERTAINTY_RL_COVARIATES

# Nominal schedule covariates, now sourced from qvals_table (derived from Alison's
# p_rew_leaf/contingency_count, exactly stop-start-aligned) rather than the difflib
# contingency_table_csv -- so every covariate the model fits shares one alignment.
VALUE_COVARIATES = (
    "task_reward_probability_chosen_arm",
    "task_reward_probability_patch_mean_gap_best_vs_second",
    "trials_since_block_change",
)

# Fit-cohort standardization support: the moments are the fitted rows' own.
FIT_SUPPORT_LABEL = "fit_cohort_all_rows"

# The three non-local-vs-SWR populations to dissociate (pooled non_local is the
# null and is not fit here).
POPULATIONS_TO_FIT = ("swr", "non_local_swr", "non_local_no_swr")

MU_UNCERTAINTY = MODEL_UNCERTAINTY_RATE["MU_uncertainty"]


def drop_exposure_sessions(stop_table: pd.DataFrame) -> pd.DataFrame:
    """Drop the epoch register's first-exposure sessions (``first_exposure_epochs``).

    All six ports share one p(R) there (1.0, then 0.5), so the value covariates
    have no contrast between ports.
    """
    exposure = pd.MultiIndex.from_tuples(
        [(*parse_nwb_filename(name), epoch) for name, epoch in first_exposure_epochs()],
        names=list(SESSION_COLUMNS),
    )
    sessions = pd.MultiIndex.from_frame(stop_table[list(SESSION_COLUMNS)])
    return stop_table[~sessions.isin(exposure)]


def fit_uncertainty_rate(
    stop_table: pd.DataFrame,
    onsets: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (session, population): MU_uncertainty fit coefficients.

    ``stop_table`` must already carry the uncertainty + value covariates and the
    reward lags; ``onsets`` must already carry ``is_swr``.
    """
    assert_predictor_timing(FITTED_COVARIATES, retrospective=True)
    onset_groups = dict(tuple(onsets.groupby(list(SESSION_COLUMNS), sort=False)))
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
    for keys, stops_g in stop_table.groupby(list(SESSION_COLUMNS), sort=False):
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
                reward_basis=True,
                self_history_lags=N_HISTORY_LAGS,
                self_history_basis=N_HISTORY_BASIS,
                stop_covariates=stop_covariates,
                history_window_s=longest_history_window_s([MU_UNCERTAINTY]),
                at_risk_events=at_risk_events(onsets_g, outcome),
                offset_history=True,
            )
            design, gate_audit = gate_bins_after_outcome(design, MU_UNCERTAINTY)
            gate_audit_rows.append(
                {
                    **dict(zip(SESSION_COLUMNS, keys, strict=True)),
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
            # One cohort for the fitted rung, frozen before the fit: the
            # fitter refuses an incomplete design rather than subsetting it.
            # The physical post-outcome gate above already selected the rows
            # that are complete for exactly these columns, so this must remove
            # nothing -- a removal would mean the two selections disagree.
            design, freeze = freeze_per_bin_cohort(design, [MU_UNCERTAINTY])
            if freeze["n_bins_after"] != freeze["n_bins_before"]:
                raise ValueError(
                    f"{'_'.join(map(str, keys))} / {outcome}: the cohort freeze "
                    f"removed {freeze['n_bins_before'] - freeze['n_bins_after']} "
                    "bins the post-outcome gate had kept; the gate's complete-case "
                    "selection and freeze_per_bin_cohort must agree."
                )
            fit = fit_per_bin_rate_model(design, MU_UNCERTAINTY)
            if fit is None:
                n_unfit[outcome] += 1
                logger.info("  %s / %s: unfit", "_".join(map(str, keys)), outcome)
                continue
            rows.append(
                {
                    "outcome": outcome,
                    "model": MU_UNCERTAINTY.name,
                    "per_bin_design": PER_BIN_DESIGN,
                    "post_outcome_gate": True,
                    "post_outcome_gate_rule": POST_OUTCOME_GATE_RULE,
                    "outcome_time_semantics": OUTCOME_TIME_SEMANTICS,
                    **gate_audit,
                    **dict(zip(SESSION_COLUMNS, keys, strict=True)),
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


def uncertainty_cohort_records(
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
    first-exposure drop, subset, and per-bin gates fall into
    ``n_excluded_session``. ``sessions_in`` is the post-join (pre-exposure-drop)
    session count so it matches ``n_in``'s stop population. Reconciles by
    construction.
    """
    stop_level = [
        c for c in _design_columns(MU_UNCERTAINTY) if c in stop_fitted.columns
    ]
    own_missing = stop_own_missing(stop_fitted, stop_level)
    records = []
    for outcome in POPULATIONS_TO_FIT:
        sub = (
            models[(models.model == MU_UNCERTAINTY.name) & (models.outcome == outcome)]
            if len(models)
            else models
        )
        records.append(
            fitted_cohort_record(
                MU_UNCERTAINTY.name,
                outcome,
                "stop",
                sub,
                count_column="n_stops",
                n_in=n_postjoin + int(qvals_gap),
                sessions_in=n_sessions_postjoin,
                own_missing=own_missing,
                n_qvals_gap=int(qvals_gap),
                drop_notes=(
                    "qvals inner-join -> qvals_gap (mechanism B); immobility<=0 / NaN "
                    "stop-level predictor -> own; first-exposure/subset/physical "
                    "post-outcome bin gate -> excluded"
                ),
            )
        )
    return records


# The qvals_table columns joined onto the stop table.
QVALS_COLUMNS = [
    "stop_id",
    "outcome_known_time",
    "trial_reward",
    *VALUE_COVARIATES,
    "rpe_postoutcome",
    *RL_COVARIATES,
]


def join_uncertainty_qvals(stop: pd.DataFrame, qvals: pd.DataFrame) -> pd.DataFrame:
    """Inner-join the value covariates of ``qvals`` (:data:`QVALS_COLUMNS`) onto ``stop``.

    Keep only stops mapped to a trial (post-final-trial tails ~1-3% drop out);
    model DVs are NaN on the first trial of a session.

    Raises
    ------
    ValueError
        If the join keeps fewer than half of the stops (a ``stop_id``
        convention mismatch).
    """
    n_before = len(stop)
    stop = stop.merge(
        qvals[QVALS_COLUMNS], on="stop_id", how="inner", validate="many_to_one"
    )
    frac = len(stop) / n_before
    if frac < 0.5:
        raise ValueError(
            f"qvals join kept only {frac:.1%} of stops -- check qvals_table stop_id convention"
        )
    logger.info(
        "qvals join: %d -> %d stops (%.0f%% kept)", n_before, len(stop), 100 * frac
    )
    return stop


def standardize_rl_covariates(stop: pd.DataFrame) -> tuple[pd.DataFrame, Reference]:
    """z-score the RL decision variables over exactly the rows that get fit.

    Applied after the first-exposure drop and any subset, so the per-SD scale
    is the sample SD (``ddof=1``) of the fitted rows (disparate/tiny scale:
    DeltaV ~0.2, updates ~0.05). The nominal block and RPE stay per-unit. The
    fitted design carries the standardized values under the raw names.

    Returns
    -------
    stop : pd.DataFrame
        ``stop`` with :data:`RL_COVARIATES` replaced by their z-scores.
    z_reference : Reference
        The moments used, for ``<model>.z_reference.json``.

    Raises
    ------
    ValueError
        If an RL covariate's SD is not finite or is zero.
    """
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
    # ddof=1: these coefficients are reported per *sample* SD of the fitted rows.
    stop, z_reference = attach_standardized(
        stop, RL_COVARIATES, support_label=FIT_SUPPORT_LABEL, ddof=1
    )
    for c in RL_COVARIATES:
        sd = z_reference[c]["std"]
        if not np.isfinite(sd) or sd == 0:
            raise ValueError(
                f"RL covariate {c!r} has degenerate SD ({sd}); cannot z-score"
            )
        # The fitted design carries the standardized values under the raw name.
        stop[c] = stop.pop(f"{c}_z")
    return stop, z_reference
