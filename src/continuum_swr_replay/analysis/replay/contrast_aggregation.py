"""Shared audited aggregation for the is_swr contrast ladders (rate + content).

Both the content contrast ladder (C/CS, CP/CPS) and the rate contrast ladder
(G/S) aggregate per-session conditional/Poisson coefficients the SAME way, so the
two ladders' summary tables mirror each other. This module is the single home for
that logic.

PRIMARY population evidence is at the **animal** level (n=5): ``coef_animal_median``
(median of per-animal medians -- balanced so animals with more sessions do not
dominate) and ``n_animals_pos`` / ``n_animals`` (sign consistency). With only 5
animals a frequentist p is barely meaningful, so the sign consistency IS the claim.
``session_signrank_p`` is a DESCRIPTIVE across-session consistency statistic (sessions
within an animal are not independent), with a BH-FDR across the rungs sharing a
population (``session_signrank_fdr_q``). ``typical_session_se`` is the median
per-session SE, not the aggregate's uncertainty. Continuous-predictor coefficients
also carry a standardized (per-1-SD) form (``std_*``); binary predictors are
per-category.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# Above this share of unidentified session fits, a term's aggregate is a
# statement about the identified minority and is not emitted at all.
MAX_UNIDENTIFIED_SHARE = 0.20

# The per-row output schema, in emit order. Kept as a constant so an
# all-gated-out run still returns a correctly-shaped (empty) frame rather than a
# column-less one that KeyErrors on column selection. ``session_signrank_fdr_q``
# is appended after the BH-FDR pass, so it is added by the caller/return, not here.
_AGG_COLUMNS = (
    "level", "model", "population", "term", "n_sessions", "n_animals",
    "n_dropped", "n_sessions_flagged_nonconverged", "n_sessions_flagged_separated",
    "coef_animal_median", "n_animals_pos", "coef_session_median",
    "typical_session_se", "std_unit", "std_factor", "std_coef_animal_median",
    "std_coef_session_median", "std_typical_session_se", "pct_positive_session",
    "pct_sig_session", "session_signrank_p",
)

logger = logging.getLogger(__name__)

# Binary predictors (per-category; not standardized). Full term names, including the
# swr_x_ / is_swr: interaction forms used by the content and rate ladders.
BINARY_TERMS = frozenset(
    {
        "is_current_arm",
        "swr_x_is_current_arm",
        "swr_x_is_prev_arm",
        "swr_x_is_last_rewarded_noncurrent_arm",
        "is_current_patch",
        "is_last_rewarded_noncurrent_patch",
        "swr_x_is_current_patch",
        "swr_x_is_last_rewarded_noncurrent_patch",
        "known_reward_outcome_lag_1",
        "is_swr:known_reward_outcome_lag_1",
    }
)


def std_factor(term: str, sd: dict) -> tuple[float, str]:
    """SD multiplier + unit label to convert a raw coef to standardized units.

    Continuous predictor -> (SD of the predictor, "per_SD"); the term may be the bare
    covariate, its ``swr_x_<cov>`` product, or its ``is_swr:<cov>`` product. Binary ->
    (1.0, "per_category"). A term whose covariate is not in ``sd`` is assumed already
    z-scored at fit time -> (1.0, "per_SD").
    """
    if term in BINARY_TERMS:
        return 1.0, "per_category"
    for cov, s in sd.items():
        if term in (cov, f"swr_x_{cov}", f"is_swr:{cov}"):
            return (s, "per_SD") if s > 0 else (1.0, "per_SD")
    return 1.0, "per_SD"


def aggregate_contrast(
    res: pd.DataFrame, specs: list[tuple[str, str, str]], level: str, sd: dict | None = None
) -> pd.DataFrame:
    """Aggregate per-session coefficients into one audited row per spec.

    Parameters
    ----------
    res : pd.DataFrame
        Per-session fits: columns ``model``, ``population``, ``animal`` and
        ``coef_<term>`` (optionally ``bse_<term>``, ``pval_<term>``).
    specs : list of (model, population, term)
        One aggregated row per spec (in order).
    level : str
        Value for the ``level`` column ("candidate_arm" / "patch" / "rate").
    sd : dict, optional
        Covariate -> global SD, for the standardized (per-SD) columns.
    """
    sd = sd or {}
    rows = []
    for model, pop, term in specs:
        factor, unit = std_factor(term, sd)
        ccol, scol, pcol = f"coef_{term}", f"bse_{term}", f"pval_{term}"
        s_all = res[(res.model == model) & (res.population == pop)]
        if ccol not in s_all.columns:
            continue
        present = s_all[ccol].notna()
        # Identifiability filter. The |coef| bound alone is selection ON the
        # estimate: over a set of unidentified fits it keeps exactly the ones
        # whose runaway estimate happened to land under the threshold, which is
        # how CS3_lastrew came to report -2.72 next to a typical session SE of
        # 2,428 (the identified sessions give -0.77). Drop a session when the
        # fit is flagged separated, or when the standard error is large relative
        # to the estimate -- a coefficient known only to within an order of
        # magnitude carries no information about its own sign.
        finite_se = s_all[scol] if scol in s_all.columns else None
        unstable = (
            (finite_se > 10 * s_all[ccol].abs()) & finite_se.notna()
            if finite_se is not None
            else pd.Series(False, index=s_all.index)
        )
        flagged = (
            s_all["separated"] == True  # noqa: E712 (NaN must not count)
            if "separated" in s_all.columns
            else pd.Series(False, index=s_all.index)
        )
        kept = present & (s_all[ccol].abs() < 10) & ~unstable & ~flagged
        n_dropped = int((present & ~kept).sum())
        s = s_all[kept]
        if len(s) < 5:
            continue
        # Refuse to publish a term that is unidentified in a large share of the
        # sessions that were fit: the surviving median describes the identified
        # minority, not the corpus, and reads as if it described the corpus.
        drop_share = n_dropped / max(int(present.sum()), 1)
        if drop_share > MAX_UNIDENTIFIED_SHARE:
            logging.warning(
                "%s/%s/%s: %.0f%% of fitted sessions unidentified (%d of %d); "
                "not emitting an aggregate row",
                model, pop, term, 100 * drop_share, n_dropped, int(present.sum()),
            )
            continue
        coef = s[ccol]
        per_animal = s.groupby("animal")[ccol].median()
        srp = wilcoxon(coef)[1] if len(coef) > 5 else np.nan
        # Session-level fit-diagnostics counts (informational; NOT drops). Counted over
        # ALL modeled session-fits for this (model, population), independent of the
        # term-level |coef|>=10 quasi-separation filter above. NaN/absent flags (older
        # res without diagnostics) are treated as not-flagged.
        n_nonconverged = (
            int((s_all["converged"] == False).sum())  # noqa: E712 (NaN must not count)
            if "converged" in s_all.columns else 0
        )
        n_separated = (
            int((s_all["separated"] == True).sum())  # noqa: E712
            if "separated" in s_all.columns else 0
        )
        rows.append({
            "level": level,
            "model": model,
            "population": pop,
            "term": term,
            "n_sessions": len(s),
            "n_animals": int(per_animal.size),
            "n_dropped": n_dropped,
            "n_sessions_flagged_nonconverged": n_nonconverged,
            "n_sessions_flagged_separated": n_separated,
            "coef_animal_median": float(per_animal.median()),
            "n_animals_pos": int((per_animal > 0).sum()),
            "coef_session_median": float(coef.median()),
            "typical_session_se": float(s[scol].median()) if scol in s else np.nan,
            "std_unit": unit,
            "std_factor": factor,
            "std_coef_animal_median": float(per_animal.median()) * factor,
            "std_coef_session_median": float(coef.median()) * factor,
            "std_typical_session_se": (
                float(s[scol].median()) * factor if scol in s else np.nan
            ),
            "pct_positive_session": float((coef > 0).mean()),
            "pct_sig_session": float((s[pcol] < 0.05).mean()) if pcol in s else np.nan,
            "session_signrank_p": srp,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        # Every spec was gated out (all terms unidentified in too large a share
        # of sessions). Return the full schema so callers that select columns
        # (e.g. ``agg["population"].map(...)``) get an empty frame, not a
        # KeyError -- a mostly-separated corpus is a valid, reportable "nothing
        # identified", not a crash.
        return pd.DataFrame(columns=[*_AGG_COLUMNS, "session_signrank_fdr_q"])
    n_dropped_total = int(df["n_dropped"].sum())
    if n_dropped_total:
        logger.info(
            "%s: dropped %d quasi-separated/NaN session-fits before aggregating",
            level, n_dropped_total,
        )
    # BH-FDR of the (descriptive) session signed-rank p, across rungs per population.
    df["session_signrank_fdr_q"] = np.nan
    for _, g in df.groupby("population"):
        v = g["session_signrank_p"].notna()
        if v.sum():
            df.loc[g.index[v], "session_signrank_fdr_q"] = multipletests(
                g.loc[v, "session_signrank_p"], method="fdr_bh"
            )[1]
    return df


def specs_from_ladder(ladder: dict, key_term: dict) -> list[tuple[str, str, str]]:
    """Expand a content ladder (rung -> (predictors, scope)) into aggregation specs:
    pooled rungs -> the ``all`` population; split rungs -> all / SWR / nonSWR."""
    specs = []
    for name in ladder:
        term = key_term.get(name)
        if term is None:
            continue
        scope = ladder[name][1]
        for pop in (["all"] if scope == "pooled" else ["all", "SWR", "nonSWR"]):
            specs.append((name, pop, term))
    return specs
