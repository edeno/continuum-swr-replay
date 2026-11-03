#!/usr/bin/env python
"""Fit the per-bin replay rate vs behavioral-uncertainty / value, by population.

Tests whether non-local replay *rate* is differentially modulated by the
animal's behavioral uncertainty and the task's nominal value structure across
SWR-coupled vs non-SWR non-local events, controlling for the within-stop time
course, raised-cosine self-history, the reward kernel, and the recent reward
rate. Rate-only -- replay direction is out of scope.

Per session: load ``stop_table``; validate its canonical
``patch_switch_fraction_last5_completed_trips``
and derive stop-local summaries (:func:`add_uncertainty_covariates`); inner-merge
the nominal value + real
behavioral-model covariates from ``qvals_table`` on ``stop_id`` (one exactly
stop-start-aligned source -- no difflib CSV); drop the first (exposure) session per
animal; attach the canonical ``is_swr`` to the onsets.
Because the fitted block contains current-trip RPE/value updates, retain only
physical response bins whose left edge is at or after ``outcome_known_time``
(reward onset for rewarded trials; latest possible pump latency, capped at
the collapsed task visit's final departure, for omissions).
Per population in ``{swr, non_local_swr, non_local_no_swr}`` build the per-bin
design (carrying the uncertainty + value covariates as ``stop_covariates``) on
the at-risk bins shared with ``fit_per_bin_rate_models.py`` (a bin inside an
ongoing event of the population's detector is dropped) and fit a single spec -- ``MU_uncertainty``, defined in
``model_specs.MODEL_UNCERTAINTY_RATE`` (not in ``MODEL_PER_BIN_RATE`` /
``all_models.pkl``). The spec is M7r (time spline +
raised-cosine self-history from event ends + reward kernel) plus the
uncertainty/value block, so the block is tested *beyond* those confounds.

Collect one tidy row per (session, population) with every covariate's
``coef_`` / ``bse_`` / ``pval_`` / ``rr_`` ->
``Processed-Data/replay_content/models_per_bin_rate/uncertainty_rate.pkl``.

Usage::

    uv run python scripts/rate/fit_uncertainty_rate.py                    # full
    uv run python scripts/rate/fit_uncertainty_rate.py --max-per-animal 1 # smoke
"""

from __future__ import annotations

from collections.abc import Sequence

from continuum_swr_replay.runs.threads import pin_blas_threads

# Pin BLAS to one thread BEFORE numpy import (matches the Family B orchestrator).
pin_blas_threads()

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from continuum_swr_replay.analysis.constants import OUTCOME_TIME_SEMANTICS  # noqa: E402
from continuum_swr_replay.analysis.events.analysed_time import (  # noqa: E402
    require_swr_source,
)
from continuum_swr_replay.analysis.events.cohort_selection import (  # noqa: E402
    STOP_COHORTS,
    select_stop_cohort,
    stop_cohort_suffix,
)
from continuum_swr_replay.analysis.events.swr_source import (  # noqa: E402
    swr_source_name,
)
from continuum_swr_replay.analysis.replay import (  # noqa: E402
    BIN_S,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    STOP_COVARIATES,
    TMAX,
    add_uncertainty_covariates,
    aggregate_to_timebins,
    attach_is_swr_to_onsets,
    fit_per_bin_rate_model,
    freeze_per_bin_cohort,
)
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
    assert_qvals_aligned,
)
from continuum_swr_replay.analysis.replay.duration_aware_history import (  # noqa: E402
    at_risk_events,
    attach_event_ends,
)
from continuum_swr_replay.analysis.replay.fit_provenance import (  # noqa: E402
    record_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.analysis.replay.model_specs import (  # noqa: E402
    MODEL_UNCERTAINTY_RATE,
    MU_UNCERTAINTY_COVARIATES,
    MU_UNCERTAINTY_RL_COVARIATES,
    PER_BIN_DESIGN,
    longest_history_window_s,
)
from continuum_swr_replay.analysis.replay.per_bin_design import (  # noqa: E402
    POPULATIONS,
)
from continuum_swr_replay.analysis.replay.per_bin_rate_model import (  # noqa: E402
    _design_columns,
)
from continuum_swr_replay.paths import (  # noqa: E402
    model_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.tables.covariate_registry import (  # noqa: E402
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table  # noqa: E402
from continuum_swr_replay.tables.session_keys import (  # noqa: E402
    SESSION_COLUMNS,
    subset_max_per_animal,
)
from continuum_swr_replay.tables.standardization import (  # noqa: E402
    Reference,
    attach_standardized,
    save_reference,
    z_reference_path,
)

# This runner's names for the model_specs blocks it fits and standardizes.
FITTED_COVARIATES = MU_UNCERTAINTY_COVARIATES
RL_COVARIATES = MU_UNCERTAINTY_RL_COVARIATES

# separated history coefs legitimately overflow exp(); convergence/separation are
# recorded per fit as flags in per_bin_rate_model (analysis.replay._fit_diagnostics),
# so only the benign RuntimeWarning is silenced (not a blanket ignore).
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def fit_input_paths() -> dict[str, Path]:
    """The tables `main` reads, recorded in the artifact's provenance sidecar;
    under ``paths.replay_content_dir()``, read when called."""
    rc = replay_content_dir()
    return {
        "stop_table": table_path("stop_table", rc),
        "event_onset_table": table_path("event_onset_table", rc),
        "event_table": table_path("event_table", rc),
        "qvals_table": table_path("qvals_table", rc),
    }


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

POST_OUTCOME_GATE_RULE = "stop_start + binj * BIN_S >= outcome_known_time"

MU_UNCERTAINTY = MODEL_UNCERTAINTY_RATE["MU_uncertainty"]


def _drop_first_session_per_animal(stop_table: pd.DataFrame) -> pd.DataFrame:
    """Drop each animal's first (exposure) session by (session_date, epoch) order."""
    first = (
        stop_table[list(SESSION_COLUMNS)]
        .drop_duplicates()
        .sort_values(list(SESSION_COLUMNS))
        .groupby("animal", sort=False)
        .head(1)
    )
    merged = stop_table.merge(
        first, on=list(SESSION_COLUMNS), how="left", indicator=True
    )
    return stop_table[merged["_merge"].to_numpy() == "left_only"]


def _retained_time_course_support(
    time_in_stop: pd.Series,
    retained: np.ndarray,
    rewarded: np.ndarray,
    omitted: np.ndarray,
) -> dict[str, float]:
    """Where the surviving within-stop time course starts, by outcome class.

    The spline's boundary knots are fixed at ``(0, TMAX)``, but the gate
    removes early bins -- and can remove different amounts by outcome class
    because rewarded and omitted outcomes have different physical cues. The low-``tc``
    region is then populated mainly by rewarded stops, so the basis is not
    orthogonal to outcome and the time-course and reward coefficients are
    partially confounded. Reporting the retained support makes the size of
    that asymmetry visible per session instead of assumed away.

    Returns
    -------
    dict
        Minimum and median retained ``tc`` for each outcome class, plus the
        rewarded-minus-omission gap in minimum retained ``tc``. NaN where a
        class retains no bin.
    """
    values = time_in_stop.to_numpy(dtype=float)
    support: dict[str, float] = {}
    for label, mask in (("rewarded", rewarded), ("omission", omitted)):
        kept = values[retained & mask]
        kept = kept[np.isfinite(kept)]
        support[f"min_retained_tc_{label}"] = (
            float(kept.min()) if kept.size else float("nan")
        )
        support[f"median_retained_tc_{label}"] = (
            float(np.median(kept)) if kept.size else float("nan")
        )
    support["min_retained_tc_omission_minus_rewarded"] = (
        support["min_retained_tc_omission"] - support["min_retained_tc_rewarded"]
    )
    return support


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
            f"column(s): {missing}. Regenerate qvals_table and stop_table."
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
        # Where the retained within-stop time course starts, per outcome
        # class. An omission's latency boundary and a reward's pump-onset
        # boundary can truncate the two classes by different amounts
        # and the time spline is no longer orthogonal to outcome. Counts
        # alone cannot show that; these do. See docs/rate_models.md.
        **_retained_time_course_support(output["tc"], candidate, rewarded, omitted),
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
            design, gate_audit = _gate_bins_after_outcome(design)
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
            {tuple(r) for r in sub[list(SESSION_COLUMNS)].to_numpy()}
            if len(sub)
            else set()
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


def _write_outputs(
    out_dir: Path,
    *,
    models: pd.DataFrame,
    z_reference: Reference,
    gate_audit: pd.DataFrame,
    cohort_records: list[dict],
    fit_inputs: dict,
    argv: list[str],
    smoke: bool,
    stop_cohort: str,
) -> Path:
    """Write the fitted rows, their z reference, the provenance, and the audits.

    The fit-cohort standardization moments go beside the model as
    ``<stem>.z_reference.json``, so a per-SD coefficient can be converted back to
    raw units without refitting. ``stop_cohort`` picks the artifact suffix (the
    canonical at-well bundle keeps the unsuffixed names) and is recorded in the
    provenance sidecar. Returns the model path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = stop_cohort_suffix(stop_cohort, smoke=smoke)
    out = out_dir / f"uncertainty_rate{sfx}.pkl"
    models.to_pickle(out)
    save_reference(z_reference, z_reference_path(out))
    write_fit_provenance(
        out,
        fit_inputs,
        "poisson_glm",
        argv,
        extra={"stop_cohort": stop_cohort, "swr_source": swr_source_name()},
    )
    gate_audit.to_csv(
        out_dir / f"uncertainty_rate_post_outcome_gate_audit{sfx}.csv",
        index=False,
    )
    write_cohort_audit(
        cohort_records, out_dir / f"uncertainty_rate_cohort_audit{sfx}.csv"
    )
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-animal",
        type=int,
        default=None,
        help="Smoke mode: keep only the first N (post-exposure) sessions per animal.",
    )
    parser.add_argument(
        "--stop-cohort",
        choices=STOP_COHORTS,
        default="at_well",
        help=(
            "Stop population to fit. 'at_well' (default) is the canonical cohort "
            "shared with the content family and publishes the unsuffixed "
            "artifacts; 'all_stops' and 'in_transit' are sensitivity bundles "
            "published under _allstops / _intransit."
        ),
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
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(argv)
    # `--max-per-animal 0` is still smoke mode, so test the flag, not its value.
    smoke = args.max_per_animal is not None

    t0 = time.perf_counter()
    inputs = fit_input_paths()
    stop = read_table(inputs["stop_table"])
    onset_table = read_table(inputs["event_onset_table"])
    event_table = read_table(inputs["event_table"])
    require_covariate_schema(stop, table="stop")
    require_covariate_schema(onset_table, table="event_onset")
    require_covariate_schema(event_table, table="event")
    # The bins are dropped by the configured source's excluded time, so the
    # tables must come from that source (its artifact run included).
    require_swr_source(stop, what=str(inputs["stop_table"]))
    require_swr_source(onset_table, what=str(inputs["event_onset_table"]))
    require_swr_source(event_table, what=str(inputs["event_table"]))
    onsets = attach_event_ends(
        attach_is_swr_to_onsets(onset_table, event_table), event_table
    )

    # add_uncertainty_covariates validates the upstream recent_switch_rate (the
    # only model-free covariate still fit) and derives stop-local summaries. Its
    # leaky_q_value_diff / omission_streak / wsls_deviation stand-ins are no
    # longer used here.
    stop = add_uncertainty_covariates(stop, switch_window=args.switch_window)
    # Nominal schedule + real model covariates, ALL from qvals_table -- one
    # exactly stop-start-aligned source (retires the difflib contingency_table_csv
    # for this model). Keep qvals' canonical RPE name. Inner join: keep
    # only stops mapped to a trial (post-final-trial tails ~1-3% drop out); model
    # DVs are NaN on the first trial of a session -> listwise-dropped at fit.
    qvals = read_table(inputs["qvals_table"])
    require_covariate_schema(qvals, table="qvals")
    # Hash the inputs now, not at write time: the fit runs long enough that a
    # table could be regenerated mid-run and be certified as this fit's input.
    fit_inputs = record_inputs(inputs)
    qcols = qvals[
        [
            "stop_id",
            "outcome_known_time",
            "trial_reward",
            *VALUE_COVARIATES,
            "rpe_postoutcome",
            *RL_COVARIATES,
        ]
    ]
    # Guard the stop_id join before the inner join hides its casualties
    # (see assert_qvals_aligned). It must see the WHOLE stop table: it claims
    # every qvals row falling inside the epoch's span, so checking it against a
    # cohort-restricted table would read the cohort's own stops as missing.
    assert_qvals_aligned(stop, qvals)
    # Cohort next, so the qvals gap below is measured within the population
    # actually fit rather than across one this run never touches.
    stop, cohort_report = select_stop_cohort(stop, cohort=args.stop_cohort)
    logger.info(
        "stop cohort = %s (%s): %d -> %d stops (%.1f%% retained); by state %s",
        cohort_report["cohort"],
        cohort_report["rule"],
        cohort_report["n_stops_before"],
        cohort_report["n_stops_after"],
        100 * cohort_report["n_stops_after"] / max(cohort_report["n_stops_before"], 1),
        cohort_report["counts_before"],
    )
    n_before = len(stop)
    stop = stop.merge(qcols, on="stop_id", how="inner", validate="many_to_one")
    n_postjoin = len(stop)  # inner-join (qvals-gap, mechanism B) drop = n_before - this
    # Session count BEFORE the first-session-per-animal drop, so the audit's
    # sessions_in matches n_in's stop population (n_in includes the first-session
    # stops as excluded, not removed).
    n_sessions_postjoin = stop.groupby(list(SESSION_COLUMNS)).ngroups
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
        stop = subset_max_per_animal(stop, args.max_per_animal, sort_sessions=True)
    # z-score the RL decision variables AFTER the drop/subset so the per-SD scale
    # is the SD of exactly the rows that get fit (disparate/tiny scale: DeltaV
    # ~0.2, updates ~0.05). The nominal block + RPE stay per-unit.
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

    n_sessions = stop.groupby(list(SESSION_COLUMNS)).ngroups
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

    # Cohort audit (stop grain). The inner qvals join is the mechanism-B qvals-gap;
    # immobility<=0 / NaN stop-level predictors on the fitted frame -> own; the
    # first-session-per-animal drop, subset, and per-bin gates -> excluded.
    audit_qvals = 0 if smoke else (n_before - n_postjoin)  # smoke: N/A
    audit_records = _cohort_records(
        models,
        stop,
        n_postjoin=n_postjoin,
        n_sessions_postjoin=n_sessions_postjoin,
        qvals_gap=audit_qvals,
    )
    # Which population every row was fit on, carried on the row itself: a
    # coefficient read out of a concatenated table would otherwise not say, and
    # the model comparison invalidates a cross-cohort step on this column.
    models["stop_cohort"] = args.stop_cohort
    out = _write_outputs(
        model_dir("per_bin_rate"),
        models=models,
        z_reference=z_reference,
        gate_audit=gate_audit,
        cohort_records=audit_records,
        fit_inputs=fit_inputs,
        argv=[sys.argv[0], *argv],
        smoke=smoke,
        stop_cohort=args.stop_cohort,
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
        models.groupby(list(SESSION_COLUMNS)).ngroups if len(models) else 0,
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
