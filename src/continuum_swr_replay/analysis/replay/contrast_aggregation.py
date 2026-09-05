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
per-session SE, not the aggregate's uncertainty; it is the event-clustered SE
wherever the fitters wrote one (``se_source``). Continuous-predictor coefficients
also carry a standardized (per-1-SD) form (``std_*``); binary predictors are
per-category; and the reward-history basis features are weighted reward fractions
in [0, 1], so their coefficients are reported per unit of the fraction (factor 1.0,
``per_unit``) rather than per SD.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from ._fit_diagnostics import SEPARATION_COEF_ABS
from .model_specs import REWARD_HISTORY_COLUMNS

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
    "typical_session_se", "se_source", "std_unit", "std_factor",
    "std_coef_animal_median", "std_coef_session_median",
    "std_typical_session_se", "pct_positive_session",
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
    }
)

# Predictors reported per unit of the predictor itself, not per SD: the
# reward-history basis features are weighted reward fractions in [0, 1], so a
# per-SD rescaling would make "one SD of a fraction" the unit of an effect whose
# natural unit is "all of the recent outcomes rewarded vs none of them".
PER_UNIT_TERMS = frozenset(
    [*REWARD_HISTORY_COLUMNS, *(f"is_swr:{c}" for c in REWARD_HISTORY_COLUMNS)]
)


def std_factor(term: str, sd: dict) -> tuple[float, str]:
    """SD multiplier + unit label to convert a raw coef to standardized units.

    Continuous predictor -> (SD of the predictor, "per_SD"); the term may be the bare
    covariate, its ``swr_x_<cov>`` product, or its ``is_swr:<cov>`` product. Binary ->
    (1.0, "per_category"). Reward-history basis feature -> (1.0, "per_unit"), checked
    before the ``sd`` lookup so a pooled SD computed for some other purpose cannot
    silently rescale a fraction-valued coefficient. A term whose covariate is not in
    ``sd`` is assumed already z-scored at fit time -> (1.0, "per_SD").
    """
    if term in BINARY_TERMS:
        return 1.0, "per_category"
    if term in PER_UNIT_TERMS:
        return 1.0, "per_unit"
    for cov, s in sd.items():
        if term in (cov, f"swr_x_{cov}", f"is_swr:{cov}"):
            return (s, "per_SD") if s > 0 else (1.0, "per_SD")
    return 1.0, "per_SD"


def separated_for_term(fits: pd.DataFrame, term: str) -> pd.Series:
    """Per-fit mask: is ``term`` itself unidentified in this fit?

    Separation is a property of a coefficient, not of a model. A fit whose
    ``reward_history_incomplete`` or self-history coefficient ran away still
    identifies its scientific terms perfectly well, so dropping the session for
    every term is what emptied the audited table. The per-term rule reads the
    ``separated_terms`` column written by the fitters (``"|"``-joined names).

    A product term (``is_swr:<covariate>``, ``swr_x_<covariate>``) is judged on
    its OWN coefficient and nothing else. Its ``bse * sd`` already measures how
    well the interaction itself is pinned down, so inheriting a factor's
    separation would drop sessions in which the product is perfectly well
    estimated: an ``is_swr`` main effect with a large SE is a statement about the
    baseline rate difference between the two streams, not about whether the
    covariate's slope differs between them.

    Parameters
    ----------
    fits : pd.DataFrame
        Per-session fits for one (model, population).
    term : str
        The aggregated term.

    Returns
    -------
    pd.Series of bool
        Indexed like ``fits``. Frames written before the term-level flag carry no
        ``separated_terms`` column; for those the fit-level ``separated`` flag is
        the only information available and is used unchanged.
    """
    if "separated_terms" not in fits.columns:
        if "separated" not in fits.columns:
            return pd.Series(False, index=fits.index)
        return fits["separated"] == True  # noqa: E712 (NaN must not count)
    return fits["separated_terms"].map(
        lambda v: isinstance(v, str) and term in v.split("|")
    )


def typical_session_se(fits: pd.DataFrame, term: str) -> tuple[float, str]:
    """Median per-session SE for ``term``, and which SE that median is made of.

    The event-clustered SE (``bse_cluster_<term>``) is preferred wherever the
    fitters wrote a finite one. On a fraction response the model SE describes a
    design in which each event contributed one draw and nothing within it
    varied, so it is the clustered SE that reflects the events' own
    variability. A session whose sandwich is not finite keeps its own model SE
    rather than being dropped or filled from another session.

    Parameters
    ----------
    fits : pd.DataFrame
        The per-session fits already kept for this (model, population, term).
    term : str
        The aggregated term.

    Returns
    -------
    (float, str)
        Median SE (NaN when neither column is present), and ``"cluster"`` when
        more than half of the sessions contributed a clustered SE, else
        ``"model"``. In practice the mix is all-or-nothing: a frame either
        predates the clustered column or carries it for every session.
    """
    model = fits.get(f"bse_{term}")
    cluster = fits.get(f"bse_cluster_{term}")
    if cluster is None:
        return (np.nan, "model") if model is None else (float(model.median()), "model")
    from_cluster = np.isfinite(cluster.to_numpy(dtype=float))
    chosen = (
        cluster.where(from_cluster)
        if model is None
        else cluster.where(from_cluster, model)
    )
    source = "cluster" if from_cluster.sum() > len(fits) / 2 else "model"
    return float(chosen.median()), source


def aggregate_contrast(
    res: pd.DataFrame, specs: list[tuple[str, str, str]], level: str, sd: dict | None = None
) -> pd.DataFrame:
    """Aggregate per-session coefficients into one audited row per spec.

    Parameters
    ----------
    res : pd.DataFrame
        Per-session fits: columns ``model``, ``population``, ``animal`` and
        ``coef_<term>`` (optionally ``bse_<term>``, ``pval_<term>``,
        ``sd_<term>``). A session is excluded from a term's aggregate when the
        fitters named that term in ``separated_terms``, or when its coefficient
        ran away past the magnitude backstop without tripping that flag. The
        backstop is read on the effect scale, ``|coef_<term>| * sd_<term> >=
        SEPARATION_COEF_ABS``, using the fit's own persisted predictor SD --
        including for an ``is_swr:<covariate>`` product, which carries its own
        ``sd_`` column. Frames with no ``sd_<term>`` column (legacy artifacts)
        fall back to ``|coef_<term>| >= SEPARATION_COEF_ABS``.
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
        ccol, pcol = f"coef_{term}", f"pval_{term}"
        s_all = res[(res.model == model) & (res.population == pop)]
        if ccol not in s_all.columns:
            continue
        present = s_all[ccol].notna()
        # Identifiability filter, applied PER TERM. Separation is measured on the
        # effect scale (SE times the predictor's own SD), which is what
        # non-identification actually shows up in, and the fitters name the terms
        # it holds for. The magnitude backstop catches an estimate that ran away
        # without tripping that flag, and it is read on the SAME effect scale:
        # ``|coef| * sd`` is the log-rate swing the predictor actually produces
        # over one SD of itself, so a large coefficient on a tiny-SD covariate --
        # identified, just steep per unit -- is kept, while a genuinely runaway
        # fit is dropped. Judging ``|coef|`` alone was selection ON the estimate:
        # it dropped identified sessions for having small-SD predictors, and it
        # kept whichever unidentified fits happened to land under the threshold,
        # which is how CS3_lastrew came to report -2.72 next to a typical session
        # SE of 2,428 (the identified sessions give -0.77). Precision is
        # deliberately NOT a criterion: the retired ``SE > 10 * |coef|`` rule
        # dropped identified sessions whose effect was near zero, biasing the
        # surviving median away from zero.
        flagged = separated_for_term(s_all, term)
        sdcol = f"sd_{term}"
        if sdcol in s_all.columns:
            runaway = s_all[ccol].abs() * s_all[sdcol] >= SEPARATION_COEF_ABS
        else:
            # Legacy artifacts predate the persisted per-fit predictor SDs; the
            # raw coefficient is the only scale available for them.
            runaway = s_all[ccol].abs() >= SEPARATION_COEF_ABS
        kept = present & ~runaway & ~flagged
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
        session_se, se_source = typical_session_se(s, term)
        per_animal = s.groupby("animal")[ccol].median()
        srp = wilcoxon(coef)[1] if len(coef) > 5 else np.nan
        # Session-level fit-diagnostics counts (informational; NOT drops). These stay
        # FIT-level -- they answer "how many fits had any unidentified term", which is
        # a different question from the per-term drop above. Counted over ALL modeled
        # session-fits for this (model, population). NaN/absent flags (older res
        # without diagnostics) are treated as not-flagged.
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
            "typical_session_se": session_se,
            "se_source": se_source,
            "std_unit": unit,
            "std_factor": factor,
            "std_coef_animal_median": float(per_animal.median()) * factor,
            "std_coef_session_median": float(coef.median()) * factor,
            "std_typical_session_se": session_se * factor,
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
