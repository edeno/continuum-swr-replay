#!/usr/bin/env python
"""Per-event SWR-coupling strength (ripple power) vs behavioral covariates.

For each non-local event, the SWR-coupling strength is summarized by
``ripple_power_z`` (the event-mean of the z-scored consensus ripple power -- the
same signal the Kay detector thresholds, so ``is_swr`` is a thresholded version of
this continuous quantity). This script asks, across non-local events:

  1. Distribution shape of the SWR power -- bimodal (categorical SWR vs non-SWR)
     or unimodal (the SWR/non-SWR split is a cut on a continuum)?
  2. Duration confound -- is the (event-mean) power correlated with event duration?
  3. Power model -- does reward/value/exploration predict the SWR power of an
     event (power ~ covariates), i.e. acting on SWR-COUPLING strength rather than
     on whether an event occurs (occurrence is the separate per-bin rate model)?

The model contains the current trial's RPE, so its fit cohort is restricted to
events beginning at or after ``outcome_known_time`` (reward onset for rewarded
trials; latest possible pump latency, capped at the collapsed task visit's final
departure, for omissions).
Distribution and duration
descriptives remain summaries of all non-local events.

Scope / caveat: this fits only the *conditional power-vs-covariates* relationship
by per-event OLS pooled across sessions. It is NOT a full marked point process --
it does not jointly model the event intensity, does not test whether the power
depends on event history (the conditional-independence assumption), and has no
shared-latent coupling between occurrence and power. Those are deferred to a joint
treatment. ``ripple_power_z`` here is the event-mean; a peri-onset peak (the
detector's actual criterion) is the better summary and is left as a follow-up.

Output: ``models_per_bin_rate/swr_power_model.pkl`` (per-session coefficients).

Usage::

    uv run python scripts/rate/fit_swr_power_model.py                    # full
    uv run python scripts/rate/fit_swr_power_model.py --max-per-animal 1 # smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, wilcoxon

from continuum_swr_replay.analysis.constants import OUTCOME_TIME_SEMANTICS
from continuum_swr_replay.analysis.events.swr_source import require_swr_source
from continuum_swr_replay.analysis.replay.cohort_audit import (
    cohort_record,
    write_cohort_audit,
)
from continuum_swr_replay.analysis.replay.covariate_timing import (
    assert_predictor_timing,
)
from continuum_swr_replay.analysis.replay.covariates import add_uncertainty_covariates
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.tables.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.tables.session_keys import (
    SESSION_COLUMNS,
    subset_max_per_animal,
)
from continuum_swr_replay.tables.standardization import (
    Reference,
    attach_standardized,
    save_reference,
    z_reference_path,
)

RC = PROCESSED_DATA_DIR / "replay_content"
SC = list(SESSION_COLUMNS)
POWER_COL = "ripple_power_z"  # event-mean z-scored consensus ripple power
PREDICTORS = [
    "known_reward_outcome_lag_1",
    "task_reward_probability_chosen_arm",
    "rpe_postoutcome",
    "patch_switch_fraction_last5_completed_trips",
    "task_reward_probability_patch_mean_gap_best_vs_second",
    "reward_belief_mean_chosen_arm_preoutcome",
    "time_since_stop_start_s",
]
POWER_FIT_COLUMNS = (*PREDICTORS, POWER_COL)
POST_OUTCOME_GATE_RULE = "event_start_s >= outcome_known_time"
FIT_SUPPORT_LABEL = "fit_cohort_all_rows"


def _bimodality_coef(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient; > 0.555 suggests bimodal (uniform ref)."""
    n = len(x)
    g = skew(x)
    k = (np.mean((x - x.mean()) ** 4) / x.var() ** 2) - 3.0  # excess kurtosis
    denom = k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return (g**2 + 1) / denom


def _assemble() -> pd.DataFrame:
    ev = read_table(RC / "event_table.pkl")
    stop = read_table(RC / "stop_table.pkl")
    # qvals_table (exact stop-start alignment). These are the behavioral
    # model's experienced value and RPE, replacing retired nominal/leaky-Q
    # stand-ins. Keep their canonical names through fitting and persistence.
    cont = read_table(RC / "models_per_bin_rate" / "qvals_table.pkl")
    require_covariate_schema(ev, table="event")
    require_covariate_schema(stop, table="stop")
    require_covariate_schema(cont, table="qvals")
    # is_swr and ripple_power_z must come from the configured SWR source
    require_swr_source(ev, what=str(RC / "event_table.pkl"))
    require_swr_source(stop, what=str(RC / "stop_table.pkl"))
    stop = add_uncertainty_covariates(stop)
    # Many events per stop, one row per stop on both right-hand tables.
    # Without validate= a duplicated stop_id would silently multiply
    # events into the per-event OLS instead of raising.
    df = ev.merge(
        stop[
            [
                "stop_id",
                "known_reward_outcome_lag_1",
                "patch_switch_fraction_last5_completed_trips",
            ]
        ],
        on="stop_id",
        how="left",
        validate="many_to_one",
    )
    df = df.merge(
        cont[
            [
                "stop_id",
                "outcome_known_time",
                "trial_reward",
                "task_reward_probability_chosen_arm",
                "rpe_postoutcome",
                "task_reward_probability_patch_mean_gap_best_vs_second",
                "reward_belief_mean_chosen_arm_preoutcome",
            ]
        ],
        on="stop_id",
        how="left",
        validate="many_to_one",
    )
    return df


def _post_outcome_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(finite, eligible)`` row masks for the post-outcome gate.

    The single definition of the gate rule, shared by the fit and the
    cohort audit so the two cannot drift apart. Equality is eligible.

    Parameters
    ----------
    df : pd.DataFrame
        Event-grain frame carrying ``event_start_s`` and ``outcome_known_time``.

    Returns
    -------
    finite : np.ndarray, shape (n_events,)
        Both timestamps present.
    eligible : np.ndarray, shape (n_events,)
        Finite *and* starting at/after the outcome boundary.
    """
    required = (*SC, "event_start_s", "outcome_known_time", "trial_reward")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(
            "Post-outcome SWR-power gating requires physical timing column(s): "
            f"{missing}. Regenerate event_table.pkl and qvals_table.pkl."
        )

    event_start = df["event_start_s"].to_numpy(dtype=float)
    boundary = df["outcome_known_time"].to_numpy(dtype=float)
    raw_trial_reward = df["trial_reward"]
    trial_reward = pd.to_numeric(raw_trial_reward, errors="coerce").to_numpy(
        dtype=float
    )
    if (raw_trial_reward.notna() & ~np.isfinite(trial_reward)).any():
        raise ValueError("trial_reward must contain binary 0/1 values or NaN")
    known_outcome = np.isfinite(trial_reward)
    if not np.isin(trial_reward[known_outcome], (0.0, 1.0)).all():
        raise ValueError("trial_reward must contain binary 0/1 values or NaN")
    # A left-join qvals gap legitimately has no outcome label. It cannot pass
    # the physical gate and is attributed to qvals_gap by the cohort audit,
    # rather than aborting the entire paper-scale fit.
    finite = np.isfinite(event_start) & np.isfinite(boundary) & known_outcome
    return finite, finite & (event_start >= boundary)


def _power_fit_complete_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows retained by the OLS model's exact listwise-finite filter."""
    missing = [column for column in POWER_FIT_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"SWR-power fit requires model column(s): {missing}")
    return np.isfinite(df[list(POWER_FIT_COLUMNS)].to_numpy(dtype=float)).all(axis=1)


def _balanced_post_outcome_selection(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Return physical and fit-complete outcome-balanced masks plus an audit."""
    finite, eligible = _post_outcome_masks(df)
    fit_complete = _power_fit_complete_mask(df)
    trial_reward = df["trial_reward"].to_numpy(dtype=float)
    selected = np.zeros(len(df), dtype=bool)
    audit_rows = []
    for keys, positions in df.groupby(SC, sort=False).indices.items():
        pos = np.asarray(positions, dtype=int)
        finite_g = finite[pos]
        eligible_g = eligible[pos]
        fit_complete_g = fit_complete[pos]
        candidate_g = eligible_g & fit_complete_g
        rewarded_g = trial_reward[pos] == 1.0
        omitted_g = trial_reward[pos] == 0.0
        reward_after = int((eligible_g & rewarded_g).sum())
        omission_after = int((eligible_g & omitted_g).sum())
        reward_complete = int((candidate_g & rewarded_g).sum())
        omission_complete = int((candidate_g & omitted_g).sum())
        balance_valid = reward_complete > 0 and omission_complete > 0
        if balance_valid:
            selected[pos] = candidate_g
            balance_status = "both_outcomes_retained"
        else:
            missing = []
            if reward_complete == 0:
                missing.append("rewarded")
            if omission_complete == 0:
                missing.append("omission")
            balance_status = "missing_after_complete_case:" + ",".join(missing)
        audit_rows.append(
            {
                **dict(zip(SC, keys, strict=True)),
                "post_outcome_gate": True,
                "post_outcome_gate_rule": POST_OUTCOME_GATE_RULE,
                "outcome_time_semantics": OUTCOME_TIME_SEMANTICS,
                "n_events_before_post_outcome_gate": int(len(pos)),
                "n_events_after_post_outcome_gate": int(eligible_g.sum()),
                "n_rewarded_events_before_post_outcome_gate": int(rewarded_g.sum()),
                "n_rewarded_events_after_post_outcome_gate": reward_after,
                "n_omission_events_before_post_outcome_gate": int(omitted_g.sum()),
                "n_omission_events_after_post_outcome_gate": omission_after,
                "n_events_before_complete_case": int(eligible_g.sum()),
                "n_rewarded_events_before_complete_case": reward_after,
                "n_omission_events_before_complete_case": omission_after,
                "n_events_after_complete_case": int(candidate_g.sum()),
                "n_rewarded_events_after_complete_case": reward_complete,
                "n_omission_events_after_complete_case": omission_complete,
                "n_events_excluded_missing_timing": int((~finite_g).sum()),
                "n_events_excluded_pre_outcome": int((finite_g & ~eligible_g).sum()),
                "n_events_excluded_incomplete_fit": int(
                    (eligible_g & ~fit_complete_g).sum()
                ),
                "outcome_balance_valid": balance_valid,
                "outcome_balance_status": balance_status,
                "n_events_after_outcome_balance_guard": int(candidate_g.sum())
                if balance_valid
                else 0,
            }
        )
    return finite, eligible, selected, pd.DataFrame(audit_rows)


def _gate_events_after_outcome(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep fit-complete events whose start is at/after the current outcome.

    Equality is eligible. Rows with non-finite event or boundary timestamps do
    not pass. Outcome balance is evaluated after the OLS listwise-finite mask;
    the audit distinguishes timing, physical-gate, and complete-case losses.
    """
    _, _, selected, audit = _balanced_post_outcome_selection(df)
    return df.loc[selected].copy(), audit


def fit_power_model(df: pd.DataFrame) -> pd.DataFrame:
    """Per-session OLS: ripple_power_z ~ standardized covariates. One row/session."""
    assert_predictor_timing(PREDICTORS, retrospective=True)
    gated, gate_audit = _gate_events_after_outcome(df)
    gate_by_session = {
        tuple(row[column] for column in SC): row
        for row in gate_audit.to_dict(orient="records")
    }
    rows = []
    z_reference: dict[str, Reference] = {}
    for keys, g in gated.groupby(SC, sort=False):
        # ``_balanced_post_outcome_selection`` already applied this exact
        # listwise-finite column set before checking outcome balance.
        g = g[list(POWER_FIT_COLUMNS)]
        if len(g) < 30:
            continue
        # Per-session standardization, so a coefficient reads per 1 SD of that
        # session's own covariate spread. ``zero_std="center"`` keeps a
        # within-session constant predictor as an all-zero (unidentified,
        # coefficient 0) column rather than NaN-ing the whole session's fit.
        standardized, session_reference = attach_standardized(
            g[list(PREDICTORS)].copy(),
            PREDICTORS,
            support_label=FIT_SUPPORT_LABEL,
            zero_std="center",
        )
        z_reference["|".join(str(key) for key in keys)] = session_reference
        X = standardized[[f"{c}_z" for c in PREDICTORS]].to_numpy(dtype=float)
        y = g[POWER_COL].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(g)), X], y, rcond=None)
        gate_metadata = gate_by_session[keys]
        rows.append(
            {
                **dict(zip(SC, keys, strict=True)),
                "n_events": len(g),
                **{key: value for key, value in gate_metadata.items() if key not in SC},
                **dict(zip(PREDICTORS, beta[1:], strict=True)),
            }
        )
    result = pd.DataFrame(rows)
    result.attrs["post_outcome_gate_audit"] = gate_audit.to_dict(orient="records")
    result.attrs["z_reference"] = z_reference
    return result


# Predictors sourced (via left join) from qvals_table -- NaN on the mechanism-B tail.
_QVALS_PREDICTORS = (
    "outcome_known_time",
    "trial_reward",
    "task_reward_probability_chosen_arm",
    "rpe_postoutcome",
    "task_reward_probability_patch_mean_gap_best_vs_second",
    "reward_belief_mean_chosen_arm_preoutcome",
)


def _cohort_records(df: pd.DataFrame, coefs: pd.DataFrame) -> list[dict]:
    """Event-grain cohort record. The per-session ``dropna`` over PREDICTORS silently
    removed the qvals-tail events (qvals-derived predictors NaN on the left join); this
    attributes them to ``n_drop_qvals_gap`` (mechanism B), the other predictor/response
    NaNs to own, and the <30-event session gate to excluded."""
    n_in = len(df)
    # Raises if the timing columns are absent, so ``event_start_s`` below
    # needs no membership guard: reaching this point means it exists.
    _, eligible, selected, _ = _balanced_post_outcome_selection(df)
    qp = [c for c in _QVALS_PREDICTORS if c in df.columns]
    op = [
        c
        for c in (*PREDICTORS, POWER_COL, "event_start_s")
        if c not in qp and c in df.columns
    ]
    qmask = (
        ~np.isfinite(df[qp].to_numpy(dtype=float)).all(axis=1)
        if qp
        else pd.Series(False, index=df.index)
    )
    qmask = pd.Series(qmask, index=df.index)
    omask = (
        ~np.isfinite(df[op].to_numpy(dtype=float)).all(axis=1)
        if op
        else pd.Series(False, index=df.index)
    )
    omask = pd.Series(omask, index=df.index) & ~qmask
    complete = ~(qmask | omask)
    post_outcome = complete & pd.Series(selected, index=df.index)
    valid = df[post_outcome]
    fit_keys = {tuple(r) for r in coefs[SC].to_numpy()} if len(coefs) else set()
    vbs = valid.groupby(SC, sort=False).size()
    n_modeled = int(sum(int(vbs.get(k, 0)) for k in fit_keys))
    n_qvals, n_own = int(qmask.sum()), int(omask.sum())
    records = [
        cohort_record(
            "swr_power",
            "all",
            "event",
            n_in=n_in,
            n_modeled=n_modeled,
            sessions_in=df.groupby(SC).ngroups,
            sessions_modeled=len(fit_keys),
            animals_modeled=len({k[0] for k in fit_keys}),
            drop_notes=(
                "qvals-derived predictor NaN (left join) -> qvals_gap (mechanism B); "
                "other predictor/response/timing NaN -> own; physical pre-outcome "
                "event or <30-event session -> excluded"
            ),
            n_drop_qvals_gap=n_qvals,
            n_drop_own_predictor_missing=n_own,
            n_excluded_session=n_in - n_qvals - n_own - n_modeled,
        )
    ]
    records[0].update(
        post_outcome_gate=True,
        post_outcome_gate_rule=POST_OUTCOME_GATE_RULE,
        outcome_time_semantics=OUTCOME_TIME_SEMANTICS,
        n_events_excluded_pre_outcome=int(
            (complete & ~pd.Series(eligible, index=df.index)).sum()
        ),
        n_events_excluded_outcome_imbalance=int(
            (
                complete
                & pd.Series(eligible, index=df.index)
                & ~pd.Series(selected, index=df.index)
            ).sum()
        ),
    )
    return records


def _write_outputs(
    out_dir: Path,
    *,
    coefs: pd.DataFrame,
    z_reference: dict[str, Reference],
    gate_audit: pd.DataFrame,
    cohort_records: list[dict],
    smoke: bool,
) -> Path:
    """Write the fitted coefficients, their z reference, and the two audits.

    The per-session standardization moments go beside the model as
    ``<stem>.z_reference_by_session.json``, so a standardized coefficient can be
    converted back to raw units without refitting. Returns the model path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = "_smoke" if smoke else ""
    out = out_dir / f"swr_power_model{sfx}.pkl"
    coefs.to_pickle(out)
    save_reference(z_reference, z_reference_path(out, by_session=True))
    gate_audit.to_csv(
        out_dir / f"swr_power_post_outcome_gate_audit{sfx}.csv",
        index=False,
    )
    write_cohort_audit(cohort_records, out_dir / f"swr_power_cohort_audit{sfx}.csv")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-animal",
        type=int,
        default=None,
        help=(
            "Smoke mode: keep only the first N sessions per animal and write "
            "`_smoke`-suffixed artifacts, never the canonical ones."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RC / "models_per_bin_rate",
        help="Directory for the fitted model, its z reference, and the audits.",
    )
    args = parser.parse_args()
    # `--max-per-animal 0` is still smoke mode, so test the flag, not its value.
    smoke = args.max_per_animal is not None

    df = _assemble()
    if smoke:
        df = subset_max_per_animal(df, args.max_per_animal, sort_sessions=True)
    m = df[POWER_COL].to_numpy()
    print(f"=== SWR power distribution ({POWER_COL}, n={len(df)} non-local events) ===")
    print(
        "  pctiles 5/25/50/75/95:", np.round(np.percentile(m, [5, 25, 50, 75, 95]), 2)
    )
    print(f"  Sarle bimodality coef = {_bimodality_coef(m):.3f}  (>0.555 => bimodal)")
    for lab, sub in [("non-SWR events", df[~df.is_swr]), ("SWR events", df[df.is_swr])]:
        s = sub[POWER_COL]
        print(f"  {lab:16s} n={len(s):6d}  mean={s.mean():+.2f}  sd={s.std():.2f}")
    lo, hi = np.percentile(m, [1, 99])
    h, edges = np.histogram(m, bins=np.linspace(lo, hi, 21))
    print("  histogram (1-99 pctile):")
    for c, cnt in zip((edges[:-1] + edges[1:]) / 2, h, strict=True):
        print(f"    {c:+5.1f} | {'#' * int(40 * cnt / h.max())}")

    d = df[["event_duration_s", POWER_COL]].dropna()
    print(
        f"\n=== duration confound: corr(power, event_duration_s) ="
        f" {np.corrcoef(d.event_duration_s, d[POWER_COL])[0, 1]:+.3f}"
        f"  (log-dur {np.corrcoef(np.log(d.event_duration_s), d[POWER_COL])[0, 1]:+.3f}) ==="
    )

    coefs = fit_power_model(df)
    gate_audit = pd.DataFrame(coefs.attrs["post_outcome_gate_audit"])
    out = _write_outputs(
        args.output_dir,
        coefs=coefs,
        z_reference=coefs.attrs["z_reference"],
        gate_audit=gate_audit,
        cohort_records=_cohort_records(df, coefs),
        smoke=smoke,
    )
    if coefs.empty:
        raise RuntimeError(
            "No SWR-power session retained both rewarded and omission events "
            "after the physical post-outcome and complete-case gates; audits "
            "were written, but a one-outcome survivor model was not fit."
        )
    print(f"\n=== power model: {POWER_COL} ~ standardized covariates (per session) ===")
    print(
        f"  {len(coefs)} sessions, {coefs.animal.nunique()} animals; saved -> {out}\n"
    )
    print(f"  {'covariate':<22}{'med coef':>10}{'%pos':>7}{'animals':>9}{'p':>10}")
    for c in PREDICTORS:
        s = coefs[c].dropna()
        n_an = (coefs.groupby("animal")[c].median() > 0).sum()
        print(
            f"  {c:<22}{s.median():>+10.3f}{(s > 0).mean() * 100:>6.0f}%"
            f"{f'{n_an}/5':>9}{wilcoxon(s)[1]:>10.1e}"
        )


if __name__ == "__main__":
    main()
