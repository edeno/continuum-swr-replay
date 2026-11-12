"""Replay content × reward outcome → behavioral change.

Two stop-level analyses:

1. :func:`stop_replay_summary` — collapse per-event ``arm_fraction`` to
   one row per stop with the fraction that fell on each task-relevant
   role: current, previous, last-rewarded, and "best" (arm with highest
   ``reward_fraction_candidate_arm_last10_visits`` at event time). All four
   also have SWR-only
   and MUA-only variants for the SWR/MUA-as-moderator analyses.
2. :func:`fit_is_stay_logistic_per_session` — per-session logistic
   regression of ``is_stay`` on ``is_omitted_outcome * frac_on_<role>``.
   Returns coefficients suitable for cross-session aggregation
   (Wilcoxon, sign agreement).
3. :func:`cross_session_role_summary` — that aggregation: per role, the
   Wilcoxon on the per-session interaction, BH-FDR across the roles.
4. :func:`pooled_gee_role_summary` — the pooled GEE's interaction p,
   BH-FDR across the roles; :func:`smallest_adjusted_p` reads the smallest
   adjusted p over strata from either summary.

The k-trial visit-rate-change analysis is intentionally not in this
module yet; it requires trial-level joins not present in the canonical
tables.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.shared_statistics import (
    benjamini_hochberg,
    signed_rank_p,
)
from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH
from continuum_swr_replay.tables.session_keys import (
    SESSION_COLUMNS,
    present_session_columns,
)

from ._helpers import strict_mean, strict_sum

__all__ = [
    "StopSummaryConfig",
    "fit_is_stay_logistic_per_session",
    "fit_is_stay_pooled_gee",
    "stop_replay_summary",
]

logger = logging.getLogger(__name__)

# Roles whose ``arm_fraction`` we track at the stop level. Each role
# maps to the column on ``event_table`` that names the arm to query.
_ROLES: dict[str, str] = {
    "current": "occupancy_well_arm",
    "prev": "prev_arm",
    "last_rewarded": "last_rewarded_arm",
}


@dataclass(frozen=True)
class StopSummaryConfig:
    """Configuration for :func:`stop_replay_summary`.

    Attributes
    ----------
    well_arms : tuple[int, ...]
        Arm IDs eligible for "best" lookup. Trunk arms are excluded
        because they have no candidate-arm reward-fraction history.
    """

    well_arms: tuple[int, ...] = (0, 1, 2, 3, 4, 5)


_DEFAULT_CFG = StopSummaryConfig()


def _select_arm_fractions(
    event_arm_table: pd.DataFrame,
    *,
    arm_col: str,
) -> pd.DataFrame:
    """For each event, select ``arm_fraction`` at ``arm == arm_col`` value.

    ``arm_col`` is a column on ``event_arm_table`` (and also on
    ``event_table``, which is where it originates) telling us which
    arm to look up per event.
    """
    sess_keys = present_session_columns(event_arm_table)
    keep = event_arm_table[
        [*sess_keys, "event_id", "candidate_arm", arm_col, "arm_fraction"]
    ]
    matched = keep[keep["candidate_arm"] == keep[arm_col]]
    return matched[[*sess_keys, "event_id", "arm_fraction"]].rename(
        columns={"arm_fraction": f"frac_on_{arm_col}"}
    )


def _per_event_best_arm(
    event_arm_table: pd.DataFrame, well_arms: tuple[int, ...]
) -> pd.DataFrame:
    """Per-event arm with the highest prior-visit reward fraction.

    Only well arms are considered (trunk arms have no rate). Ties
    resolved by taking the lowest arm id.
    """
    sess_keys = list(SESSION_COLUMNS)
    well_only = event_arm_table[event_arm_table["candidate_arm"].isin(well_arms)]
    # Keep only events where at least one well arm has a non-NaN rate.
    valid = well_only.dropna(subset=["reward_fraction_candidate_arm_last10_visits"])
    if valid.empty:
        return pd.DataFrame(columns=[*sess_keys, "event_id", "best_arm"])
    best = (
        valid.sort_values(
            [
                *sess_keys,
                "event_id",
                "reward_fraction_candidate_arm_last10_visits",
                "candidate_arm",
            ],
            ascending=[True, True, True, True, False, True],
        )
        .groupby([*sess_keys, "event_id"], sort=False)
        .head(1)[[*sess_keys, "event_id", "candidate_arm"]]
        .rename(columns={"candidate_arm": "best_arm"})
    )
    return best.reset_index(drop=True)


def _frac_on_best(event_arm_table: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """Per-event ``arm_fraction`` at ``arm == best_arm``."""
    sess_keys = list(SESSION_COLUMNS)
    if best.empty:
        return pd.DataFrame(columns=[*sess_keys, "event_id", "frac_on_best"])
    merged = event_arm_table.merge(best, on=[*sess_keys, "event_id"], how="inner")
    matched = merged[merged["candidate_arm"] == merged["best_arm"]]
    return matched[[*sess_keys, "event_id", "arm_fraction"]].rename(
        columns={"arm_fraction": "frac_on_best"}
    )


def stop_replay_summary(
    stop_table: pd.DataFrame,
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    *,
    cfg: StopSummaryConfig = _DEFAULT_CFG,
) -> pd.DataFrame:
    """One row per stop with replay-content summaries.

    For each role in ``{current, prev, last_rewarded, best}``, the
    output has:

    - ``frac_on_<role>``: strict mean ``arm_fraction`` across events
      in the stop (NaN if the stop has zero events OR any event in
      the stop has NaN ``arm_fraction`` for this role — see the
      ``_strict_mean`` aggregator below for why a NaN-skipping mean
      would silently bias the stop toward surviving events).
    - ``frac_on_<role>_swr``: same, restricted to events with
      ``is_swr == True``.
    - ``frac_on_<role>_non_swr``: same, restricted to ``is_swr == False``.
    - ``frac_on_<role>_mua``: same, restricted to ``is_mua == True``.

    Plus pass-through behavioral columns from ``stop_table``. The source event
    count is ``n_selected_events``; the summary exposes the more specific
    derived name ``n_events_with_replay``.
    """
    sess_keys = list(SESSION_COLUMNS)

    # Short-circuit when there are no events at all: stop_table flows
    # straight through with all frac_on_<role> columns as NaN.
    role_labels = (
        "current",
        "prev",
        "last_rewarded",
        "best",
        "current_patch",
        "prev_patch",
        "last_rewarded_patch",
        "best_patch",
    )
    if event_table.empty or event_arm_table.empty:
        out = stop_table[
            [
                *sess_keys,
                "stop_id",
                "occupancy_well_arm",
                "prev_arm",
                "last_rewarded_arm",
                "is_rewarded_outcome",
                "did_last_completed_trip_switch_patch",
                "is_stay",
                "is_swr_stop",
                "is_mua_stop",
                "n_selected_events",
                "immobility_duration_s",
            ]
        ].copy()
        for label in ("", "_swr", "_non_swr", "_mua"):
            for r in role_labels:
                out[f"frac_on_{r}{label}"] = np.nan
        out["n_events_with_replay"] = out["n_selected_events"].astype("Int64")
        out["is_omitted_outcome"] = 1.0 - out["is_rewarded_outcome"]
        return out

    # ── Per-event arm fractions for each role.
    # event_table holds the role→arm mapping. ``event_arm_table`` may
    # also carry the role columns broadcast across its 9 rows per
    # event; in that case we trust the broadcast values and only need
    # event_table for the stop_id linkage and as a fallback for any
    # role columns the event-arm table happens to be missing.
    role_cols = list(_ROLES.values())
    available_role_cols = [c for c in role_cols if c in event_table.columns]
    if not available_role_cols and not event_table.empty:
        raise ValueError(
            "event_table is missing all role columns "
            f"({role_cols}); cannot compute stop_replay_summary"
        )
    # Only pull role columns from event_table that are NOT already on
    # event_arm_table — otherwise the merge would create a `<col>_ev`
    # suffix and force a fragile coalesce. event_arm_table's broadcast
    # is already correct for production tables.
    role_cols_to_pull = [
        c for c in available_role_cols if c not in event_arm_table.columns
    ]
    pull_cols = [*sess_keys, "event_id", "stop_id", *role_cols_to_pull]
    ev_role = event_table[pull_cols]
    ea = event_arm_table.merge(ev_role, on=[*sess_keys, "event_id"], how="left")

    per_event_fracs = []
    for role, arm_col in _ROLES.items():
        if arm_col not in ea.columns:
            continue
        df = _select_arm_fractions(ea, arm_col=arm_col)
        df = df.rename(columns={f"frac_on_{arm_col}": f"frac_on_{role}_event"})
        per_event_fracs.append(df)

    best = _per_event_best_arm(ea, cfg.well_arms)
    fb = _frac_on_best(ea, best).rename(columns={"frac_on_best": "frac_on_best_event"})
    per_event_fracs.append(fb)

    # ── Per-event PATCH-level fractions for each role.
    # Sum arm_fraction over arms whose patch label equals the role's
    # arm's patch. The patch-level signal is much less alternation-
    # confounded than the arm-level signal: in this task next_arm ≈
    # prev_arm in ~90% of trials, but next_patch == current_patch in
    # ~88% of trials, leaving the cross-patch trials uncontaminated.
    ea_p = ea.copy()
    ea_p["_arm_patch"] = ea_p["candidate_arm"].map(TRACK_SEGMENT_TO_PATCH)
    # Use a NaN-aware sum (``skipna=False``) so a per-event-per-patch
    # group whose target arms have any NaN ``arm_fraction`` propagates
    # NaN to the patch-level total. The default ``.sum()`` is
    # ``skipna=True`` and silently fabricates a real zero from
    # missing data — the user reproduced ``frac_on_current_patch ==
    # 0.0`` for an event whose target-patch arm fractions were all
    # NaN, indistinguishable from a confidently-zero replay.
    patch_sums = (
        ea_p.dropna(subset=["_arm_patch"])
        .groupby([*sess_keys, "event_id", "_arm_patch"], as_index=False)["arm_fraction"]
        .agg(strict_sum)
        .rename(columns={"arm_fraction": "_patch_frac", "_arm_patch": "_patch"})
    )
    patch_sums["_patch"] = patch_sums["_patch"].astype(int)

    # For each role, resolve the role's patch label per event and look
    # up the corresponding patch_frac.
    for role, arm_col in _ROLES.items():
        if arm_col not in ea.columns:
            continue
        # role_patch per event
        sub = (
            ea[[*sess_keys, "event_id", arm_col]]
            .drop_duplicates(subset=[*sess_keys, "event_id"])
            .copy()
        )
        sub["_role_patch"] = sub[arm_col].map(
            lambda a: float(TRACK_SEGMENT_TO_PATCH.get(int(a), np.nan))
            if pd.notna(a)
            else np.nan
        )
        sub = sub.dropna(subset=["_role_patch"])
        sub["_role_patch"] = sub["_role_patch"].astype(int)
        merged = sub.merge(
            patch_sums,
            left_on=[*sess_keys, "event_id", "_role_patch"],
            right_on=[*sess_keys, "event_id", "_patch"],
            how="left",
        )
        merged = merged[[*sess_keys, "event_id", "_patch_frac"]].rename(
            columns={"_patch_frac": f"frac_on_{role}_patch_event"}
        )
        per_event_fracs.append(merged)

    # frac_on_best_patch_event: the patch of the best arm.
    if not best.empty:
        b = best.copy()
        b["_best_patch"] = b["best_arm"].map(
            lambda a: float(TRACK_SEGMENT_TO_PATCH.get(int(a), np.nan))
            if pd.notna(a)
            else np.nan
        )
        b = b.dropna(subset=["_best_patch"])
        b["_best_patch"] = b["_best_patch"].astype(int)
        merged = b.merge(
            patch_sums,
            left_on=[*sess_keys, "event_id", "_best_patch"],
            right_on=[*sess_keys, "event_id", "_patch"],
            how="left",
        )
        merged = merged[[*sess_keys, "event_id", "_patch_frac"]].rename(
            columns={"_patch_frac": "frac_on_best_patch_event"}
        )
        per_event_fracs.append(merged)

    # ── Merge per-event fractions back onto event_table for SWR/MUA flags.
    ev_keys = event_table[[*sess_keys, "event_id", "stop_id", "is_swr", "is_mua"]]
    pe = ev_keys.copy()
    for df in per_event_fracs:
        pe = pe.merge(df, on=[*sess_keys, "event_id"], how="left")

    # ── Aggregate to stop level.
    # Coerce the SWR/MUA flags to a non-null bool mask. NA/None values
    # default to False so they're excluded from the *positive* stratum
    # (e.g., NA-is_swr does not count as an SWR event) and from the
    # negative complement (we only complement the rows we'd have
    # included). This is the conservative interpretation — a missing
    # flag is treated as "unknown, exclude from both is_swr and
    # is_non_swr selections."
    is_swr_mask = pe["is_swr"].fillna(False).astype(bool)
    is_mua_mask = pe["is_mua"].fillna(False).astype(bool)
    is_swr_known = pe["is_swr"].notna()
    arm_role_cols = ["current", "prev", "last_rewarded", "best"]
    patch_role_cols = [f"{r}_patch" for r in arm_role_cols]
    role_cols = arm_role_cols + patch_role_cols
    agg_dfs = []
    for label, mask in (
        ("", None),
        ("_swr", is_swr_mask),
        ("_non_swr", (~is_swr_mask) & is_swr_known),
        ("_mua", is_mua_mask),
    ):
        sub = pe if mask is None else pe[mask]
        if sub.empty:
            agg = pd.DataFrame(
                {f"frac_on_{r}{label}": pd.Series(dtype=float) for r in role_cols}
            )
            agg[[*sess_keys, "stop_id"]] = pd.DataFrame(columns=[*sess_keys, "stop_id"])
        else:
            # ``strict_mean`` keeps a stop with any NaN event-level
            # role fraction surfaced as NaN — the default ``skipna=True``
            # mean would silently emit the surviving event's value.
            agg_cols = {
                f"frac_on_{r}{label}": (f"frac_on_{r}_event", strict_mean)
                for r in role_cols
                if f"frac_on_{r}_event" in sub.columns
            }
            agg = sub.groupby([*sess_keys, "stop_id"], as_index=False).agg(**agg_cols)
        agg_dfs.append(agg)

    summary = stop_table[
        [
            *sess_keys,
            "stop_id",
            "occupancy_well_arm",
            "prev_arm",
            "last_rewarded_arm",
            "is_rewarded_outcome",
            "did_last_completed_trip_switch_patch",
            "is_stay",
            "is_swr_stop",
            "is_mua_stop",
            "n_selected_events",
            "immobility_duration_s",
        ]
    ].copy()
    for agg in agg_dfs:
        summary = summary.merge(agg, on=[*sess_keys, "stop_id"], how="left")

    # n_events_with_replay = total events in the stop. Per-role
    # contribution counts are NOT exposed; if any event had NaN
    # ``arm_fraction`` for a given role, the stop's ``frac_on_<role>``
    # is NaN (strict aggregation above). Treat ``frac_on_<role>``
    # being NaN as the per-role missing-data signal — don't try to
    # back out a per-role contributor count from this column.
    summary["n_events_with_replay"] = summary["n_selected_events"].astype("Int64")
    summary["is_omitted_outcome"] = 1.0 - summary["is_rewarded_outcome"]

    return summary


# ─────────────────────────────────────────────────────────────────────
# is_stay logistic
# ─────────────────────────────────────────────────────────────────────


def fit_is_stay_logistic_per_session(
    stop_summary: pd.DataFrame,
    *,
    content_col: str = "frac_on_last_rewarded",
    outcome_col: str = "is_omitted_outcome",
    target_col: str = "is_stay",
    min_stops: int = 30,
) -> pd.DataFrame:
    """Per-session logistic: ``is_stay ~ outcome * content_col``.

    Causal-hygiene note: ``outcome_col`` defaults to
    ``is_omitted_outcome`` — a POST_OUTCOME covariate — but that is **valid here
    and deliberately unchanged**: the modeled unit is the stay/leave *decision*,
    which happens AFTER the current trial's outcome by construction (the animal
    decides where to go next having already seen the reward/omission). So this is
    a genuinely post-outcome regression, not a leak; the covariate-timing gate
    that governs the *replay-rate/content* ladders does not apply to it.

    The interaction coefficient (``beta_inter``) is the headline
    statistic: positive ⇒ replay content on the role amplifies the
    omission effect on staying; negative ⇒ counteracts.

    Sessions with fewer than ``min_stops`` valid stops, or constant
    ``is_stay`` / ``outcome`` / ``content_col``, are skipped (returned
    with NaN coefficients flagged by ``status``).

    Parameters
    ----------
    stop_summary : pd.DataFrame
        Output of :func:`stop_replay_summary`, possibly concatenated
        across sessions.
    content_col : str
        Name of the replay-content column to interact with the outcome.
    outcome_col : str
        Binary 0/1 column (e.g., ``is_omitted_outcome``).
    target_col : str
        Binary 0/1 column to predict (default ``is_stay``).
    min_stops : int
        Minimum number of valid stops per session.

    Returns
    -------
    pd.DataFrame
        One row per session with:
        ``animal, day, epoch, n_stops, beta_outcome, beta_content,
        beta_inter, se_outcome, se_content, se_inter, status``.
    """
    import statsmodels.api as sm

    sess_keys = list(SESSION_COLUMNS)
    out_rows = []
    for sess_key, g in stop_summary.groupby(sess_keys, dropna=False):
        animal, day, epoch = sess_key
        sub = g.dropna(subset=[content_col, outcome_col, target_col]).copy()
        sub = sub[sub[content_col].notna()]
        n = len(sub)

        row: dict = dict(zip(sess_keys, sess_key, strict=True))
        row.update(
            {
                "n_stops": n,
                "beta_outcome": np.nan,
                "beta_content": np.nan,
                "beta_inter": np.nan,
                "se_outcome": np.nan,
                "se_content": np.nan,
                "se_inter": np.nan,
                "status": "",
            }
        )

        if n < min_stops:
            row["status"] = f"too_few_stops_{n}"
            out_rows.append(row)
            continue
        # Constant target / outcome / content → un-fittable.
        if sub[target_col].nunique() < 2:
            row["status"] = "target_constant"
            out_rows.append(row)
            continue
        if sub[outcome_col].nunique() < 2:
            row["status"] = "outcome_constant"
            out_rows.append(row)
            continue

        outcome = sub[outcome_col].astype(float).to_numpy()
        content = sub[content_col].astype(float).to_numpy()
        y = sub[target_col].astype(float).to_numpy()
        X = np.column_stack(
            [
                np.ones(n),
                outcome,
                content,
                outcome * content,
            ]
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = sm.Logit(y, X).fit(disp=False, maxiter=200)
        except Exception as exc:  # noqa: BLE001
            row["status"] = f"fit_error_{type(exc).__name__}"
            out_rows.append(row)
            continue

        if not model.mle_retvals.get("converged", False):
            row["status"] = "not_converged"

        params = model.params
        bse = model.bse if model.bse is not None else np.full_like(params, np.nan)
        row["beta_outcome"] = float(params[1])
        row["beta_content"] = float(params[2])
        row["beta_inter"] = float(params[3])
        row["se_outcome"] = float(bse[1])
        row["se_content"] = float(bse[2])
        row["se_inter"] = float(bse[3])
        if row["status"] == "":
            row["status"] = "ok"
        out_rows.append(row)

    return pd.DataFrame(out_rows)


# ─────────────────────────────────────────────────────────────────────
# Pooled GEE (Bernoulli + cluster-robust SE on session)
# ─────────────────────────────────────────────────────────────────────


def fit_is_stay_pooled_gee(
    stop_summary: pd.DataFrame,
    *,
    content_col: str,
    outcome_col: str = "is_omitted_outcome",
    target_col: str = "is_stay",
) -> dict:
    """Pooled cross-session GEE: ``is_stay ~ outcome * content_col``.

    Uses :class:`statsmodels.genmod.generalized_estimating_equations.GEE`
    with Binomial family and an independence working correlation,
    clustering on session (``(animal, day, epoch)``). The
    independence working correlation gives a population-averaged
    logistic regression fit; cluster-robust ("sandwich") SEs absorb
    the cross-session noise that the per-session ``Logit`` could not.

    This is the right tool for the Phase E gate: it returns one set
    of population-level coefficients with valid inference, instead
    of 8–50 per-session fits whose variance overwhelms the trend.

    Parameters
    ----------
    stop_summary : pd.DataFrame
        Output of :func:`stop_replay_summary` (concatenated across
        sessions).
    content_col : str
        Replay-content column to interact with the outcome
        (e.g., ``frac_on_last_rewarded`` or ``frac_on_current_patch``).
    outcome_col, target_col : str
        Binary 0/1 columns.

    Returns
    -------
    dict
        Keys ``content_col``, ``outcome_col``, ``target_col``,
        ``n_stops``, ``n_sessions``, ``beta_intercept``,
        ``beta_outcome``, ``beta_content``, ``beta_inter``,
        ``se_outcome``, ``se_content``, ``se_inter``,
        ``p_outcome``, ``p_content``, ``p_inter``,
        ``status`` (``"ok"``/error code).
    """
    from statsmodels.genmod.cov_struct import Independence
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.generalized_estimating_equations import GEE

    sess_keys = list(SESSION_COLUMNS)
    sub = stop_summary.dropna(subset=[content_col, outcome_col, target_col]).copy()

    base = {
        "content_col": content_col,
        "outcome_col": outcome_col,
        "target_col": target_col,
        "n_stops": int(len(sub)),
        "n_sessions": int(sub.groupby(sess_keys).ngroups) if not sub.empty else 0,
        "beta_intercept": np.nan,
        "beta_outcome": np.nan,
        "beta_content": np.nan,
        "beta_inter": np.nan,
        "se_outcome": np.nan,
        "se_content": np.nan,
        "se_inter": np.nan,
        "p_outcome": np.nan,
        "p_content": np.nan,
        "p_inter": np.nan,
        "status": "",
    }
    if len(sub) < 50:
        base["status"] = "too_few_stops"
        return base
    if sub[target_col].nunique() < 2 or sub[outcome_col].nunique() < 2:
        base["status"] = "constant_target_or_outcome"
        return base

    # Build a stable session-id grouping key for GEE clusters.
    sub = sub.sort_values(sess_keys).reset_index(drop=True)
    groups = sub[sess_keys].astype(str).agg("|".join, axis=1).factorize()[0]

    outcome = sub[outcome_col].astype(float).to_numpy()
    content = sub[content_col].astype(float).to_numpy()
    y = sub[target_col].astype(float).to_numpy()
    X = np.column_stack(
        [
            np.ones(len(sub)),
            outcome,
            content,
            outcome * content,
        ]
    )

    # Rank check on the design matrix. The GEE backend silently
    # pseudo-inverts a rank-deficient X^T W X and returns parameter
    # estimates with vanishing standard errors — a constant
    # ``content_col`` (e.g. ``frac_on_last_rewarded == 0.5`` for every
    # stop) makes column 2 collinear with the intercept and column 3
    # (``outcome * content``) collinear with ``outcome``, so the
    # interaction is mathematically unidentified, but ``status`` came
    # back as ``"ok"`` with ``se_inter`` ~ 1e-17 and a deceptively
    # significant ``p_inter``. Reject the design before fitting so
    # downstream FDR / reporting only sees fits with a unique
    # solution.
    if np.linalg.matrix_rank(X) < X.shape[1]:
        base["status"] = "rank_deficient_design"
        return base

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GEE(
                y,
                X,
                groups=groups,
                family=Binomial(),
                cov_struct=Independence(),
            ).fit()
    except Exception as exc:  # noqa: BLE001
        base["status"] = f"fit_error_{type(exc).__name__}"
        return base

    params = model.params
    bse = model.bse if model.bse is not None else np.full_like(params, np.nan)
    pvals = model.pvalues if model.pvalues is not None else np.full_like(params, np.nan)
    base["beta_intercept"] = float(params[0])
    base["beta_outcome"] = float(params[1])
    base["beta_content"] = float(params[2])
    base["beta_inter"] = float(params[3])
    base["se_outcome"] = float(bse[1])
    base["se_content"] = float(bse[2])
    base["se_inter"] = float(bse[3])
    base["p_outcome"] = float(pvals[1])
    base["p_content"] = float(pvals[2])
    base["p_inter"] = float(pvals[3])
    base["status"] = "ok"
    return base


# The content roles Stage 6 reads the stay decision on: four arm roles, and the
# same four at the patch grain.
ARM_ROLES = ("current", "prev", "last_rewarded", "best")
PATCH_ROLES = ("current_patch", "prev_patch", "last_rewarded_patch", "best_patch")
ROLES = ARM_ROLES + PATCH_ROLES


def cross_session_role_summary(coefs: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon on each role's per-session interaction, FDR-corrected across roles.

    This is the statistic Stage 6 is read on, so it lives in one function: the
    run logs it, and an audit re-derives it from a saved bundle -- including a
    bundle fit on another population -- with the same code rather than a second
    implementation that can drift.

    Parameters
    ----------
    coefs : pandas.DataFrame
        Per-session fits as ``fit_is_stay_logistic_per_session`` returns them,
        concatenated over roles: ``role``, ``status`` and ``beta_inter``.

    Returns
    -------
    pandas.DataFrame
        One row per role in :data:`ROLES`, in that order:
        ``role``, ``n_sessions`` (converged sessions with a finite
        ``beta_inter``), ``median_beta_inter``, ``n_pos``, ``n_neg``, ``p``,
        ``p_fdr``, and ``status`` (``"ok"`` or ``"too_few_sessions"``). A role
        whose sessions give no signed-rank p (fewer than six with a non-zero
        ``beta_inter``,
        :func:`~continuum_swr_replay.analysis.shared_statistics.signed_rank_p`) is
        ``"too_few_sessions"``, has no median, and is left out of the
        correction as well as the test.
    """
    converged = coefs[coefs["status"] == "ok"]
    rows = []
    for role in ROLES:
        beta = converged.loc[converged["role"] == role, "beta_inter"].dropna()
        p = signed_rank_p(beta)
        tested = bool(np.isfinite(p))
        rows.append(
            {
                "role": role,
                "n_sessions": len(beta),
                "median_beta_inter": float(beta.median()) if tested else np.nan,
                "n_pos": int((beta > 0).sum()),
                "n_neg": int((beta < 0).sum()),
                "p": p,
                "status": "ok" if tested else "too_few_sessions",
            }
        )
    summary = pd.DataFrame(rows)
    summary["p_fdr"] = benjamini_hochberg(summary["p"])
    return summary


def pooled_gee_role_summary(pooled: pd.DataFrame) -> pd.DataFrame:
    """The pooled GEE's interaction p of each role, FDR-corrected across roles.

    Parameters
    ----------
    pooled : pandas.DataFrame
        One row per role as :func:`fit_is_stay_pooled_gee` returns them:
        ``role``, ``status`` and ``p_inter``.

    Returns
    -------
    pandas.DataFrame
        One row per role whose fit converged (none when no fit did): ``role``
        and ``p_fdr``, its ``p_inter`` Benjamini-Hochberg corrected across
        those roles. A converged fit without a p-value is left out of the
        correction, and its ``p_fdr`` is NaN.
    """
    converged = pooled[pooled["status"] == "ok"]
    return pd.DataFrame(
        {
            "role": converged["role"].to_numpy(),
            "p_fdr": benjamini_hochberg(converged["p_inter"]),
        }
    )


def smallest_adjusted_p(summaries: Mapping[str, pd.DataFrame]) -> dict[str, object]:
    """The smallest FDR-adjusted p over the roles of several strata.

    Parameters
    ----------
    summaries : mapping of str to pandas.DataFrame
        By stratum, a role summary with ``role`` and ``p_fdr``
        (:func:`cross_session_role_summary`, :func:`pooled_gee_role_summary`).
        A role without a finite ``p_fdr`` is left out.

    Returns
    -------
    dict
        ``p_fdr``, the smallest adjusted p, and the ``stratum`` and ``role``
        that have it (on a tie, the first stratum in alphabetical order, then
        the first role).

    Raises
    ------
    ValueError
        If no role of any stratum has a finite ``p_fdr``.
    """
    candidates = [
        (p, stratum, role)
        for stratum, summary in summaries.items()
        for p, role in zip(summary["p_fdr"], summary["role"], strict=True)
        if np.isfinite(p)
    ]
    if not candidates:
        raise ValueError("no role of any stratum has a finite adjusted p")
    p_fdr, stratum, role = min(candidates)
    return {"p_fdr": p_fdr, "stratum": stratum, "role": role}


def fit_status_counts(fits: pd.DataFrame) -> dict[str, int]:
    """How many stay/leave fits there are rows for and how many converged.

    Parameters
    ----------
    fits : pandas.DataFrame
        The per-session fits (:func:`fit_is_stay_logistic_per_session`,
        concatenated over roles): one row per session and role, with
        ``status``. It is ``"ok"`` when the fit converged; any other status is
        a session skipped before fitting (too few stops, a constant target or
        outcome) or a fit that failed or did not converge.

    Returns
    -------
    dict
        ``n_fits``, the rows (sessions by roles, skipped sessions included),
        and ``n_converged``, the rows whose fit converged.
    """
    return {"n_fits": len(fits), "n_converged": int((fits["status"] == "ok").sum())}
