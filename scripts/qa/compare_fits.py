#!/usr/bin/env python
"""Compare a backed-up rate fit against the current one and write a markdown audit.

The reward block of both rate families changed from per-lag indicators to a
three-bump raised-cosine basis, so individual lag coefficients have no
counterpart in the new fits and are deliberately **not** compared. What is
comparable is the *joint* evidence the reward block buys, the goodness-of-fit
of the reward-free time model, and the size of the fitted cohort:

* Family B (per-bin conditional intensity) reward step, twice: the reward rung
  against its reward-free parent, as a per-session AIC difference
  (``AIC(reward rung) - AIC(parent)``; negative = the reward block helps).
  Old ``M2_rewhist`` / new ``M1_rew`` share the parent ``M0_const``; old and
  new ``M7_time_hist_rew`` share the parent ``M6_time_hist``.
* Family B time-only rung ``M5_time``: median KS statistic and the share of
  sessions inside the KS band — a reward-free rung, so a shift here means the
  cohort or the binning moved, not the reward respecification.
* ``n_bins`` per (session, outcome): the frozen-cohort size.
* Family A (per-stop Poisson) reward step: the new ``R1`` against ``R0``,
  read beside the old ``R2`` and ``Rlag`` against ``R0``.
* What the per-term separation flag changed: flagged-fit share, cross-validation
  validity, per-term separated share, audited aggregate rows, and — per emitted
  aggregate row — the split between the two mechanisms that exclude a session
  (the separation flag vs the effect-scale ``|coef| * SD >= 10`` magnitude
  backstop). That
  comparison has its own "before" — the immediately preceding refit of the SAME
  specification, fit under the fit-level flag.

Usage:
    uv run python scripts/qa/compare_fits.py
    uv run python scripts/qa/compare_fits.py --out docs/audits/my-comparison.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from continuum_swr_replay.analysis.replay._fit_diagnostics import (
    SEPARATION_COEF_ABS,
)
from continuum_swr_replay.analysis.replay._helpers import reconstruct_reward_kernels
from continuum_swr_replay.analysis.replay.contrast_aggregation import (
    separated_for_term,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_BASIS,
    REWARD_HISTORY_N_LAGS,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.session_keys import SESSION_COLUMNS

RC = PROCESSED_DATA_DIR / "replay_content"
REPO = Path(__file__).resolve().parents[2]
SC = list(SESSION_COLUMNS)
KEY = SC + ["outcome"]

DEFAULT_OLD_PER_BIN = (
    RC / "_pre_fix_backup_20260904" / "models_per_bin_rate" / "all_models.pkl"
)
DEFAULT_NEW_PER_BIN = RC / "models_per_bin_rate" / "all_models.pkl"
DEFAULT_OLD_PER_STOP = (
    RC
    / "_pre_v5_regression_refit_backup_20260902"
    / "models_per_stop_rate"
    / "all_models.pkl"
)
DEFAULT_NEW_PER_STOP = RC / "models_per_stop_rate" / "all_models.pkl"

# The term-level separation section compares the SAME specification fit under the
# fit-level flag and under the per-term flag, so its "before" is the immediately
# preceding refit, not the pre-reward-basis one used above.
TERMLEVEL_BACKUP_ROOT = RC / "_pre_termlevel_backup_20260904"
TERMLEVEL_BACKUP = TERMLEVEL_BACKUP_ROOT / "models_per_bin_rate"
DEFAULT_OLD_TERMLEVEL_PER_STOP = (
    TERMLEVEL_BACKUP_ROOT / "models_per_stop_rate" / "all_models.pkl"
)
DEFAULT_OLD_CONTRAST = TERMLEVEL_BACKUP / "swr_contrast_rate.pkl"
DEFAULT_NEW_CONTRAST = RC / "models_per_bin_rate" / "swr_contrast_rate.pkl"
DEFAULT_OLD_AUDITED = TERMLEVEL_BACKUP / "swr_contrast_rate_audited.pkl"
DEFAULT_NEW_AUDITED = RC / "models_per_bin_rate" / "swr_contrast_rate_audited.pkl"
DEFAULT_OLD_TERMLEVEL_PER_BIN = TERMLEVEL_BACKUP / "all_models.pkl"

# The reward step of each family: (rung, its reward-free parent).
OLD_PER_BIN_STEPS = {
    "base": ("M2_rewhist", "M0_const"),
    "full": ("M7_time_hist_rew", "M6_time_hist"),
}
NEW_PER_BIN_STEPS = {
    "base": ("M1_rew", "M0_const"),
    "full": ("M7_time_hist_rew", "M6_time_hist"),
}
# Family A: the new single reward rung, and the two old rungs it replaced.
OLD_PER_STOP_STEPS = {"R2 vs R0": ("R2", "R0"), "Rlag vs R0": ("Rlag", "R0")}
NEW_PER_STOP_STEPS = {"R1 vs R0": ("R1", "R0")}

OUTCOME_ORDER = ["swr", "non_local", "non_local_swr", "non_local_no_swr"]
# The rung both reward quantities below are read from.
FULL_MODEL = "M7_time_hist_rew"
IMMEDIATE_COEF = f"coef_{REWARD_HISTORY_COLUMNS[0]}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def delta_aic(
    models: pd.DataFrame, rung: str, parent: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Per (session, outcome) ``AIC(rung) - AIC(parent)``, and what pairing cost.

    A session contributes a difference only when both rungs were fit there and
    both carry a finite AIC. Sessions that reach only one side of the pair are
    silently unusable for this comparison, so they are counted rather than
    dropped without trace.

    Parameters
    ----------
    models : pd.DataFrame
        Tidy per-session fits with ``model``, ``outcome``, the session keys and
        ``aic``.
    rung, parent : str
        The reward rung and its reward-free parent.

    Returns
    -------
    paired : pd.DataFrame
        One row per (session, outcome) with ``delta_aic``.
    unpaired : dict of str to int
        Per outcome, sessions carrying either rung that did not survive the
        pairing (present on one side only, or with a non-finite AIC).
    """
    have = set(models["model"].unique())
    if rung not in have or parent not in have:
        return pd.DataFrame(columns=KEY + ["delta_aic"]), {}
    a = models.loc[models.model == rung, KEY + ["aic"]]
    b = models.loc[models.model == parent, KEY + ["aic"]]
    merged = a.merge(b, on=KEY, suffixes=("_rung", "_parent"))
    merged["delta_aic"] = merged["aic_rung"] - merged["aic_parent"]
    paired = merged[KEY + ["delta_aic"]].dropna(subset=["delta_aic"])
    either = pd.concat([a[KEY], b[KEY]]).drop_duplicates()
    n_either = either.groupby("outcome", observed=True).size()
    n_paired = paired.groupby("outcome", observed=True).size()
    unpaired = n_either.subtract(n_paired, fill_value=0).astype(int).to_dict()
    return paired, unpaired


def _median_table(
    old: pd.DataFrame,
    new: pd.DataFrame,
    value: str,
    old_unpaired: dict[str, int] | None = None,
    new_unpaired: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Per-outcome median of ``value`` in each fit, with session counts.

    ``old_unpaired`` / ``new_unpaired`` add the sessions each side lost to the
    rung-against-parent pairing, so a shrinking median cannot be read as a
    shifted effect when it is really a shrinking denominator.
    """
    rows = []
    for outcome in OUTCOME_ORDER:
        o = (
            old.loc[old.outcome == outcome, value]
            if len(old)
            else pd.Series(dtype=float)
        )
        n = (
            new.loc[new.outcome == outcome, value]
            if len(new)
            else pd.Series(dtype=float)
        )
        if o.empty and n.empty:
            continue
        row = {
            "outcome": outcome,
            "n_sessions_before": int(o.notna().sum()),
            "median_before": o.median(),
            "n_sessions_after": int(n.notna().sum()),
            "median_after": n.median(),
        }
        if old_unpaired is not None or new_unpaired is not None:
            row["n_unpaired_before"] = int((old_unpaired or {}).get(outcome, 0))
            row["n_unpaired_after"] = int((new_unpaired or {}).get(outcome, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def _per_animal_median_table(
    old: pd.DataFrame, new: pd.DataFrame, value: str
) -> pd.DataFrame:
    """Per (outcome, animal) median of ``value`` in each fit."""
    rows = []
    animals = sorted(
        set(old.get("animal", pd.Series(dtype=object)))
        | set(new.get("animal", pd.Series(dtype=object)))
    )
    for outcome in OUTCOME_ORDER:
        for animal in animals:
            o = (
                old[(old.outcome == outcome) & (old.animal == animal)][value]
                if len(old)
                else pd.Series(dtype=float)
            )
            n = (
                new[(new.outcome == outcome) & (new.animal == animal)][value]
                if len(new)
                else pd.Series(dtype=float)
            )
            if o.empty and n.empty:
                continue
            rows.append(
                {
                    "outcome": outcome,
                    "animal": animal,
                    "n_before": int(o.notna().sum()),
                    "median_before": o.median(),
                    "n_after": int(n.notna().sum()),
                    "median_after": n.median(),
                }
            )
    return pd.DataFrame(rows)


def n_bins_table(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Per-outcome cohort size before/after, from any rung (all rungs share it)."""
    rows = []
    for outcome in OUTCOME_ORDER:
        o = old[(old.outcome == outcome) & (old.model == "M0_const")]
        n = new[(new.outcome == outcome) & (new.model == "M0_const")]
        if o.empty and n.empty:
            continue
        merged = o[SC + ["n_bins"]].merge(
            n[SC + ["n_bins"]], on=SC, suffixes=("_before", "_after")
        )
        diff = merged["n_bins_after"] - merged["n_bins_before"]
        rows.append(
            {
                "outcome": outcome,
                "n_sessions_before": len(o),
                "n_sessions_after": len(n),
                "n_sessions_paired": len(merged),
                "median_n_bins_before": o["n_bins"].median(),
                "median_n_bins_after": n["n_bins"].median(),
                "median_delta": diff.median() if len(merged) else np.nan,
                "n_sessions_changed": int((diff != 0).sum()) if len(merged) else 0,
            }
        )
    return pd.DataFrame(rows)


def reward_quantities(new_bin: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two reward quantities the refit is usually summarized by.

    They are computed from different things and must not be quoted as one
    clause. The first is the **reconstructed kernel**: each session's three
    basis coefficients are mapped through the row-normalized basis and the
    per-lag rate ratios are summarized by their median across sessions. The
    second is a **paired test on one coefficient**, the immediate basis
    component, comparing the SWR-coupled and non-SWR non-local populations
    within session.

    Parameters
    ----------
    new_bin : pd.DataFrame
        Tidy per-session per-bin fits.

    Returns
    -------
    kernel : pd.DataFrame
        Median reconstructed rate ratio at lags 1-3 per outcome.
    paired : pd.DataFrame
        One row: the within-session paired comparison of the immediate
        component between the two non-local populations.
    """
    full = new_bin[new_bin.model == FULL_MODEL]
    columns = [f"coef_{c}" for c in REWARD_HISTORY_COLUMNS]
    rows = []
    for outcome in OUTCOME_ORDER:
        fitted = full.loc[full.outcome == outcome, columns].dropna()
        if fitted.empty:
            continue
        kernels = np.exp(
            reconstruct_reward_kernels(
                fitted.to_numpy(dtype=float),
                REWARD_HISTORY_N_LAGS,
                REWARD_HISTORY_N_BASIS,
            )
        )
        median = np.median(kernels, axis=0)
        rows.append(
            {
                "outcome": outcome,
                "n_sessions": len(fitted),
                "rr_lag1": median[0],
                "rr_lag2": median[1],
                "rr_lag3": median[2],
            }
        )
    kernel = pd.DataFrame(rows)

    a = full.loc[full.outcome == "non_local_swr", SC + [IMMEDIATE_COEF]]
    b = full.loc[full.outcome == "non_local_no_swr", SC + [IMMEDIATE_COEF]]
    merged = a.merge(b, on=SC, suffixes=("_swr", "_no_swr")).dropna()
    if len(merged) < 2:
        return kernel, pd.DataFrame()
    difference = merged[f"{IMMEDIATE_COEF}_swr"] - merged[f"{IMMEDIATE_COEF}_no_swr"]
    per_animal = merged.assign(d=difference).groupby("animal")["d"].median()
    paired = pd.DataFrame(
        [
            {
                "comparison": "non_local_swr − non_local_no_swr",
                "n_paired_sessions": len(merged),
                "pct_sessions_positive": 100 * float((difference > 0).mean()),
                "animals_positive": f"{int((per_animal > 0).sum())}/{len(per_animal)}",
                # Pre-formatted: a p-value of this magnitude is unreadable
                # under the table's shared fixed-point float format.
                "signrank_p": "{:.1e}".format(
                    wilcoxon(
                        merged[f"{IMMEDIATE_COEF}_swr"],
                        merged[f"{IMMEDIATE_COEF}_no_swr"],
                    ).pvalue
                ),
            }
        ]
    )
    return kernel, paired


def rung_session_counts(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Sessions fitted per (rung, outcome) in each fit; rungs absent show as 0."""

    def _counts(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["model", "outcome", f"n_sessions{suffix}"])
        out = (
            df.groupby(["model", "outcome"], sort=False)
            .size()
            .rename(f"n_sessions{suffix}")
        )
        return out.reset_index()

    merged = _counts(old, "_before").merge(
        _counts(new, "_after"), on=["model", "outcome"], how="outer"
    )
    merged[["n_sessions_before", "n_sessions_after"]] = (
        merged[["n_sessions_before", "n_sessions_after"]].fillna(0).astype(int)
    )
    merged["outcome"] = pd.Categorical(merged["outcome"], OUTCOME_ORDER, ordered=True)
    return merged.sort_values(["model", "outcome"]).reset_index(drop=True)


def _md(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    """Markdown table; NaN renders as an em dash."""
    if df.empty:
        return "_(no rows)_\n"
    shown = df.copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(
                lambda v: "—" if pd.isna(v) else floatfmt.format(v)
            )
    header = "| " + " | ".join(shown.columns) + " |"
    rule = "| " + " | ".join("---" for _ in shown.columns) + " |"
    body = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in shown.itertuples(index=False)
    )
    return f"{header}\n{rule}\n{body}\n"


def cv_validity_table(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Per Family-B rung: share of session-fits flagged separated and CV-scored.

    A fit-level separation flag voids the whole cross-validation metric for the
    rung, so the two shares move together: the fraction of fits carrying the flag
    and the fraction whose ``cv_bits_per_event`` survived as a finite number.

    Parameters
    ----------
    old, new : pd.DataFrame
        Tidy per-session per-bin fits, before and after.

    Returns
    -------
    pd.DataFrame
        One row per rung present in either fit.
    """

    def _shares(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["model", f"n_fits{suffix}"])
        grouped = df.groupby("model", sort=True)
        out = pd.DataFrame(
            {
                f"n_fits{suffix}": grouped.size(),
                f"pct_separated{suffix}": 100
                * grouped["separated"].apply(lambda s: (s == True).mean()),  # noqa: E712
                f"pct_cv_scored{suffix}": 100
                * grouped["cv_bits_per_event"].apply(
                    lambda s: float(np.isfinite(s).mean())
                ),
            }
        )
        return out.reset_index()

    merged = _shares(old, "_before").merge(
        _shares(new, "_after"), on="model", how="outer"
    )
    return merged.sort_values("model").reset_index(drop=True)


def per_stop_validity_table(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Family A: flagged-fit and CV-valid share per rung, before and after.

    The per-stop cross-validation is invalidated by the same rule as Family B's:
    any fold whose fit carries a separated non-nuisance term voids
    ``cv_ll_per_stop_mean`` for the whole rung.

    Parameters
    ----------
    old, new : pd.DataFrame
        Tidy per-session per-stop fits, before and after.

    Returns
    -------
    pd.DataFrame
        One row per rung, plus an ``all`` row over every rung.
    """

    def _shares(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["model", f"n_fits{suffix}"])
        rows = []
        for label, sub in [*df.groupby("model", sort=True), ("all", df)]:
            rows.append(
                {
                    "model": label,
                    f"n_fits{suffix}": len(sub),
                    f"pct_separated{suffix}": 100
                    * float((sub["separated"] == True).mean()),  # noqa: E712
                    f"pct_cv_scored{suffix}": 100
                    * float(np.isfinite(sub["cv_ll_per_stop_mean"]).mean()),
                }
            )
        return pd.DataFrame(rows)

    return _shares(old, "_before").merge(
        _shares(new, "_after"), on="model", how="outer"
    )


def contrast_fit_flag_table(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Share of is_swr-contrast session-fits carrying the fit-level flag."""

    def _shares(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        grouped = df.groupby(["population", "cohort"], sort=True)
        return pd.DataFrame(
            {
                f"n_fits{suffix}": grouped.size(),
                f"pct_separated{suffix}": 100
                * grouped["separated"].apply(lambda s: (s == True).mean()),  # noqa: E712
            }
        ).reset_index()

    return (
        _shares(old, "_before")
        .merge(_shares(new, "_after"), on=["population", "cohort"], how="outer")
        .sort_values(["population", "cohort"])
        .reset_index(drop=True)
    )


def term_separation_table(new: pd.DataFrame, min_share: float = 0.0) -> pd.DataFrame:
    """Per term, the share of is_swr-contrast session-fits that name it separated.

    Read from the ``separated_terms`` column (``"|"``-joined names). Only fits that
    actually carry the term in their design are counted, so a term that only a few
    rungs use is not diluted by the rungs without it: a fit carries the term when
    it has a ``coef_<term>`` value.

    Parameters
    ----------
    new : pd.DataFrame
        Tidy per-session is_swr-contrast fits carrying ``separated_terms``.
    min_share : float
        Drop terms whose share is at or below this (as a fraction).

    Returns
    -------
    pd.DataFrame
        ``term``, ``n_fits_with_term``, ``pct_separated``, sorted worst first.
    """
    if "separated_terms" not in new.columns:
        return pd.DataFrame(columns=["term", "n_fits_with_term", "pct_separated"])
    named = (
        new["separated_terms"]
        .fillna("")
        .map(lambda v: set(v.split("|")) - {""} if isinstance(v, str) else set())
    )
    terms = sorted({t for names in named for t in names})
    rows = []
    for term in terms:
        ccol = f"coef_{term}"
        carries = (
            new[ccol].notna()
            if ccol in new.columns
            else pd.Series(True, index=new.index)
        )
        n_carry = int(carries.sum())
        if n_carry == 0:
            continue
        share = float((named.map(lambda s, t=term: t in s) & carries).sum()) / n_carry
        if share > min_share:
            rows.append(
                {
                    "term": term,
                    "n_fits_with_term": n_carry,
                    "pct_separated": 100 * share,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values("pct_separated", ascending=False)
        .reset_index(drop=True)
    )


def audited_row_table(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Audited is_swr-contrast rows per (population, cohort), before and after."""

    def _counts(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["population", "cohort", f"n_rows{suffix}"])
        return (
            df.groupby(["population", "cohort"], sort=True)
            .size()
            .rename(f"n_rows{suffix}")
            .reset_index()
        )

    merged = _counts(old, "_before").merge(
        _counts(new, "_after"), on=["population", "cohort"], how="outer"
    )
    if merged.empty:
        return merged
    for column in ("n_rows_before", "n_rows_after"):
        if column in merged.columns:
            merged[column] = merged[column].astype(float).fillna(0).astype(int)
    return merged.sort_values(["population", "cohort"]).reset_index(drop=True)


def audited_term_drop_table(audited: pd.DataFrame) -> pd.DataFrame:
    """Per emitted aggregate row, the share of fitted sessions dropped as separated.

    ``n_dropped`` is now a per-term count: sessions whose fit named THIS term (or
    a factor of it) as unidentified, plus the effect-scale
    ``|coef| * SD >= 10`` backstop. A spec
    whose share exceeds the unidentified-share gate emits no row at all and so
    does not appear here.

    Parameters
    ----------
    audited : pd.DataFrame
        The audited aggregate, one row per (model, population, term).

    Returns
    -------
    pd.DataFrame
        ``model``, ``population``, ``cohort``, ``term``, ``n_fitted``,
        ``n_dropped``, ``pct_dropped``, worst first.
    """
    if audited.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "population",
                "cohort",
                "term",
                "n_fitted",
                "n_dropped",
                "pct_dropped",
            ]
        )
    out = audited[
        ["model", "population", "cohort", "term", "n_sessions", "n_dropped"]
    ].copy()
    out["n_fitted"] = out["n_sessions"] + out["n_dropped"]
    out["pct_dropped"] = 100 * out["n_dropped"] / out["n_fitted"]
    return (
        out.drop(columns=["n_sessions"])
        .sort_values("pct_dropped", ascending=False)
        .reset_index(drop=True)[
            [
                "model",
                "population",
                "cohort",
                "term",
                "n_fitted",
                "n_dropped",
                "pct_dropped",
            ]
        ]
    )


# The audited aggregate relabels the stratified populations to match the content
# table; the per-session fits keep the fitter's own names.
_AUDITED_POP_TO_FIT = {
    "all": "non_local",
    "SWR": "non_local_swr",
    "nonSWR": "non_local_no_swr",
    "contrast": "contrast",
}


def drop_mechanism_table(contrast: pd.DataFrame, audited: pd.DataFrame) -> pd.DataFrame:
    """Split each emitted row's excluded sessions into the two mechanisms.

    ``aggregate_contrast`` excludes a session from a term's aggregate for either
    of two reasons, and they mean different things. (i) The fitters named the
    term in ``separated_terms`` — the effect-scale separation flag, i.e. the
    coefficient is not identified. (ii) The coefficient ran away past the
    magnitude backstop without tripping that flag, read on the same effect
    scale: ``|coef| * sd >= 10`` with the fit's own persisted predictor SD
    (``|coef| >= 10`` for legacy frames that carry no ``sd_`` column). Reporting
    only the total invites reading a backstop exclusion as a separation
    diagnosis, so the two are counted separately here, recomputed from the
    per-session fits with the same predicates the aggregation uses.

    Parameters
    ----------
    contrast : pd.DataFrame
        Per-session is_swr-contrast fits (``model``, ``population``, ``cohort``,
        ``coef_<term>``, ``sd_<term>``, ``separated_terms``).
    audited : pd.DataFrame
        The audited aggregate, one row per (model, population, cohort, term).

    Returns
    -------
    pd.DataFrame
        ``model``, ``population``, ``cohort``, ``term``, ``n_flag_only``,
        ``n_both``, ``n_backstop_only``.

    Raises
    ------
    ValueError
        If a row's term is absent from the fits, or the recomputed total
        disagrees with the audited ``n_dropped`` — either means this table is
        describing different sessions from the aggregate it annotates.
    """
    columns = [
        "model",
        "population",
        "cohort",
        "term",
        "n_flag_only",
        "n_both",
        "n_backstop_only",
    ]
    if audited.empty or contrast.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for row in audited.itertuples(index=False):
        pop = _AUDITED_POP_TO_FIT.get(row.population, row.population)
        sub = contrast[
            (contrast["model"] == row.model)
            & (contrast["population"] == pop)
            & (contrast["cohort"] == row.cohort)
        ]
        ccol = f"coef_{row.term}"
        if ccol not in sub.columns:
            raise ValueError(
                f"{row.model}/{row.population}/{row.cohort}: the per-session fits "
                f"carry no {ccol}"
            )
        present = sub[ccol].notna()
        flagged = separated_for_term(sub, row.term) & present
        sdcol = f"sd_{row.term}"
        if sdcol in sub.columns:
            runaway = sub[ccol].abs() * sub[sdcol] >= SEPARATION_COEF_ABS
        else:
            runaway = sub[ccol].abs() >= SEPARATION_COEF_ABS
        backstop = runaway & present
        counts = {
            "n_flag_only": int((flagged & ~backstop).sum()),
            "n_both": int((flagged & backstop).sum()),
            "n_backstop_only": int((backstop & ~flagged).sum()),
        }
        total = sum(counts.values())
        if total != int(row.n_dropped):
            raise ValueError(
                f"{row.model}/{row.population}/{row.cohort}/{row.term}: "
                f"recomputed {total} excluded sessions but the audited row "
                f"reports {row.n_dropped}"
            )
        rows.append(
            {
                "model": row.model,
                "population": row.population,
                "cohort": row.cohort,
                "term": row.term,
                **counts,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _drop_mechanism_prose(mechanism: pd.DataFrame) -> str:
    """One paragraph naming which emitted rows the separation flag drove.

    Parameters
    ----------
    mechanism : pd.DataFrame
        Output of :func:`drop_mechanism_table`.

    Returns
    -------
    str
        Markdown paragraph.
    """
    if mechanism.empty:
        return "No aggregate row was emitted, so there is nothing to attribute.\n"
    dropped = mechanism[
        mechanism[["n_flag_only", "n_both", "n_backstop_only"]].sum(axis=1) > 0
    ]
    if dropped.empty:
        return (
            "**Which mechanism produced these exclusions.** No emitted row lost a "
            "single session: neither the separation flag nor the "
            f"`|coef| x SD >= {SEPARATION_COEF_ABS:.0f}` backstop fired.\n"
        )
    by_flag = dropped[(dropped["n_flag_only"] + dropped["n_both"]) > 0]
    by_backstop = dropped[(dropped["n_flag_only"] + dropped["n_both"]) == 0]
    parts = [
        "**Which mechanism produced these exclusions.** Of the "
        f"{len(dropped)} emitted rows that lost at least one session, "
        f"{len(by_flag)} are driven by the per-term separation flag"
    ]
    if len(by_flag):
        named = ", ".join(
            f"`{r.model}`/{r.population} "
            f"({r.n_flag_only + r.n_both} of "
            f"{r.n_flag_only + r.n_both + r.n_backstop_only})"
            for r in by_flag.itertuples(index=False)
        )
        parts.append(f": {named}")
    parts.append(
        f". The other {len(by_backstop)} are **entirely** the "
        f"`|coef| x SD >= {SEPARATION_COEF_ABS:.0f}` backstop"
    )
    if len(by_backstop):
        models = ", ".join(f"`{m}`" for m in sorted(set(by_backstop["model"])))
        parts.append(f" ({models})")
    parts.append(
        " — runaway magnitudes the fitters did not flag. Read `n_dropped` as "
        '"excluded from the aggregate", not as "the fitters judged this term '
        'separated".\n'
    )
    return "".join(parts)


def _termlevel_section(frames: dict[str, pd.DataFrame]) -> str:
    """The markdown section describing what the per-term separation flag changed.

    Parameters
    ----------
    frames : dict of str to pd.DataFrame
        ``old_bin`` / ``new_bin`` (Family B), ``old_stop`` / ``new_stop``
        (Family A), ``old_contrast`` / ``new_contrast`` (is_swr-contrast session
        fits) and ``old_audited`` / ``new_audited`` (the audited aggregate), the
        "old" side being the same specification fit under the fit-level flag.

    Returns
    -------
    str
        Markdown.
    """
    lines: list[str] = []
    add = lines.append
    add("\n## What the per-term separation flag changed\n")
    add(
        "Separation is now judged per coefficient, and the structural nuisance "
        "terms (`Intercept`, `reward_history_incomplete`) are excluded from the "
        "judgment. Both sides below are the SAME specification and the same "
        "cohort; only the flag differs. Under the fit-level flag one blown-up "
        "nuisance or self-history coefficient voided the whole fit — it voided "
        "the cross-validation metric and dropped the session for every term in "
        "the aggregation.\n"
    )
    add(
        "\n**Two mechanisms can exclude a session from a term's aggregate, and "
        "they are not interchangeable.** (i) The **per-term separation flag** "
        "(`analysis.replay._fit_diagnostics`): the fitters name a coefficient in "
        "`separated_terms` when its standard error is non-finite or above `1e3`, "
        "or when `SE × the predictor's own SD` reaches `1.0` — the standard error "
        "measured on the effect scale, so the threshold is `SE >= 1 / SD` — with "
        f"`|coef| >= {SEPARATION_COEF_ABS:.0f}` substituted for a column of zero "
        "SD, which cannot be judged on that scale. (ii) The "
        f"**`|coef| x SD >= {SEPARATION_COEF_ABS:.0f}` backstop** in "
        "`aggregate_contrast`: a coefficient that ran away without tripping the "
        "flag is excluded on its magnitude, read on the same effect scale using "
        "the fit's own persisted predictor SD, so a large coefficient on a "
        "tiny-SD covariate is kept. The per-term table below "
        "reports mechanism (i); the per-emitted-term table at the end of this "
        "section reports both, decomposed.\n"
    )
    add("\n### Family B — flagged fits and CV validity per rung\n")
    add(
        "`pct_cv_scored` is the share of session-fits whose `cv_bits_per_event` "
        "survived as a finite number:\n"
    )
    add(_md(cv_validity_table(frames["old_bin"], frames["new_bin"]), "{:.1f}"))
    add("\n### Family A — flagged fits and CV validity per rung\n")
    add(
        "The per-stop family carries the same rule; `pct_cv_scored` is the share "
        "of session-fits whose `cv_ll_per_stop_mean` is finite:\n"
    )
    add(
        _md(
            per_stop_validity_table(frames["old_stop"], frames["new_stop"]),
            "{:.1f}",
        )
    )
    add("\n### is_swr-contrast — flagged fits per population\n")
    add(
        _md(
            contrast_fit_flag_table(frames["old_contrast"], frames["new_contrast"]),
            "{:.1f}",
        )
    )
    add("\n### is_swr-contrast — separated share per term (after)\n")
    add(
        "Share of the session-fits that carry each term in which THAT term is "
        "flagged by mechanism (i), the per-term separation flag; the "
        f"`|coef| x SD >= {SEPARATION_COEF_ABS:.0f}` backstop is not counted "
        "here. "
        "Terms never flagged are omitted. Before the change there was no "
        "per-term information: a fit was flagged as a whole.\n"
    )
    add(_md(term_separation_table(frames["new_contrast"]), "{:.1f}"))
    add("\n### is_swr-contrast — audited aggregate rows\n")
    add(
        "One row per (model, population, term) that survived the "
        "unidentified-share gate. A product term is judged on its OWN "
        "coefficient: an `is_swr:<covariate>` interaction does not inherit the "
        "`is_swr` main effect's separation, because a poorly identified "
        "baseline-rate difference between the two streams says nothing about "
        "whether the covariate's slope differs between them:\n"
    )
    add(_md(audited_row_table(frames["old_audited"], frames["new_audited"])))
    add("\n### is_swr-contrast — sessions excluded per emitted term (after)\n")
    mechanism = drop_mechanism_table(frames["new_contrast"], frames["new_audited"])
    add(
        "Per aggregated term, the share of the sessions fitted for it that were "
        "excluded from the aggregate by either mechanism, and the split between "
        "them: `n_flag_only` and `n_both` carry the separation flag, "
        "`n_backstop_only` is the magnitude backstop alone. A term whose share "
        "exceeds the unidentified-share gate emits no row and is absent here.\n"
    )
    add("\n" + _drop_mechanism_prose(mechanism))
    drops = audited_term_drop_table(frames["new_audited"])
    if not drops.empty and not mechanism.empty:
        drops = drops.merge(
            mechanism, on=["model", "population", "cohort", "term"], how="left"
        )
    add(_md(drops, "{:.1f}"))
    return "\n".join(lines)


def build_report(
    old_bin: pd.DataFrame,
    new_bin: pd.DataFrame,
    old_stop: pd.DataFrame,
    new_stop: pd.DataFrame,
    inputs: dict[str, Path],
    per_stop_note: str,
    termlevel: dict[str, pd.DataFrame] | None = None,
) -> str:
    """Assemble the markdown audit.

    Parameters
    ----------
    old_bin, new_bin, old_stop, new_stop : pd.DataFrame
        Tidy per-session fits for the two rate families, before and after.
    inputs : dict of str to Path
        Artifacts to list with their hashes.
    per_stop_note : str
        Provenance caveat printed under the input table.
    termlevel : dict of str to pd.DataFrame, optional
        ``old_bin`` / ``new_bin`` / ``old_stop`` / ``new_stop`` /
        ``old_contrast`` / ``new_contrast`` / ``old_audited`` / ``new_audited``
        for the term-level separation section. Omitted when those artifacts are
        not available.
    """
    lines: list[str] = []
    add = lines.append
    add("# Rate refit comparison: pre-fix fits vs the reward-basis refit\n")
    add(
        "Generated by `scripts/qa/compare_fits.py`. The reward block changed from "
        "per-lag reward indicators to a three-bump raised-cosine basis, so lag "
        "coefficients are **not** compared term for term. Comparable quantities "
        "are the joint reward step (ΔAIC of a reward rung against its reward-free "
        "parent, negative = reward block helps), the reward-free `M5_time` "
        "goodness-of-fit, the frozen-cohort size, and the per-rung session counts.\n"
    )
    add("## Inputs\n")
    rows = []
    for name, path in inputs.items():
        rows.append(
            {
                "artifact": name,
                "path": str(path.relative_to(REPO))
                if path.is_relative_to(REPO)
                else str(path),
                "sha256[:12]": _sha256(path)[:12],
                "modified": dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(
                    timespec="seconds"
                ),
            }
        )
    add(_md(pd.DataFrame(rows)))
    add(f"\n{per_stop_note}\n")

    kernel, paired = reward_quantities(new_bin)
    add("\n## Reward effect after the refit — two distinct quantities\n")
    add(
        "These two are routinely quoted together and are **not** the same "
        f"measurement. Both are read from `{FULL_MODEL}` in the new fit.\n"
    )
    add(
        f"\n**(a) Reconstructed kernel.** Each session's "
        f"{REWARD_HISTORY_N_BASIS} basis coefficients are mapped through the "
        "row-normalized basis to a rate ratio per reward lag; the table is the "
        "median of those per-session kernels. It describes the size of the "
        "reward effect at a given lag.\n"
    )
    add(_md(kernel, floatfmt="{:.2f}"))
    add(
        "\n**(b) Paired coefficient test.** A within-session comparison of the "
        f"single immediate basis coefficient (`{IMMEDIATE_COEF}`) between the "
        "two non-local populations. It describes how *consistently* the two "
        "populations differ, not how large the kernel is. The session share, "
        "the animal count and the p-value all come from this test and from no "
        "other quantity.\n"
    )
    add(_md(paired, floatfmt="{:.1f}"))

    add("\n## Family B — joint reward step (per-bin conditional intensity)\n")
    for label, title in [
        (
            "base",
            "Reward rung vs constant baseline (old `M2_rewhist` − `M0_const`; new `M1_rew` − `M0_const`)",
        ),
        (
            "full",
            "Full model vs time+history (old and new `M7_time_hist_rew` − `M6_time_hist`)",
        ),
    ]:
        old_d, old_unpaired = delta_aic(old_bin, *OLD_PER_BIN_STEPS[label])
        new_d, new_unpaired = delta_aic(new_bin, *NEW_PER_BIN_STEPS[label])
        add(f"\n### {title}\n")
        add(
            "Per-session ΔAIC, median across sessions. `n_unpaired_*` counts "
            "sessions that carried one of the two rungs but did not survive the "
            "pairing, and so contribute to neither median:\n"
        )
        add(_md(_median_table(old_d, new_d, "delta_aic", old_unpaired, new_unpaired)))
        add("\nPer-animal medians:\n")
        add(_md(_per_animal_median_table(old_d, new_d, "delta_aic")))

    add("\n## Family B — `M5_time` goodness of fit (reward-free rung)\n")
    old_m5 = old_bin[old_bin.model == "M5_time"]
    new_m5 = new_bin[new_bin.model == "M5_time"]
    add("KS statistic (median across sessions):\n")
    add(_md(_median_table(old_m5, new_m5, "ks_statistic"), floatfmt="{:.4f}"))
    add("\nShare of sessions inside the KS band (`ks_within_band` mean):\n")
    band = _median_table(
        old_m5.assign(ks_within_band=old_m5["ks_within_band"].astype(float)),
        new_m5.assign(ks_within_band=new_m5["ks_within_band"].astype(float)),
        "ks_within_band",
    )
    for df, src in [(old_m5, "median_before"), (new_m5, "median_after")]:
        band[src.replace("median", "share")] = [
            df.loc[df.outcome == oc, "ks_within_band"].astype(float).mean()
            for oc in band["outcome"]
        ]
    add(_md(band.drop(columns=["median_before", "median_after"]), floatfmt="{:.3f}"))

    add("\n## Family B — frozen cohort size (`n_bins`, `M0_const`)\n")
    add(_md(n_bins_table(old_bin, new_bin), floatfmt="{:.1f}"))

    add("\n## Family A — per-stop reward step\n")
    add(
        "The new ladder carries one reward rung (`R1`: three basis features plus "
        "the structural-missingness indicator, 5 parameters); the old ladder "
        "carried `R1` (the lag-1 indicator alone, 2 parameters), `R2` (lag-1 + "
        "the ten-trial reward fraction, 3) and `Rlag` (ten lag indicators, 11). "
        "The rung name `R1` therefore denotes **different specifications** "
        "before and after — read the new step beside all three old steps, not "
        "as a term-for-term replacement.\n"
    )
    for label, (rung, parent) in {**OLD_PER_STOP_STEPS, **NEW_PER_STOP_STEPS}.items():
        old_d, old_unpaired = delta_aic(old_stop, rung, parent)
        new_d, new_unpaired = delta_aic(new_stop, rung, parent)
        add(f"\n### {label}\n")
        add(_md(_median_table(old_d, new_d, "delta_aic", old_unpaired, new_unpaired)))

    if termlevel is not None:
        add(_termlevel_section(termlevel))

    add("\n## Sessions fitted per rung\n")
    add("### Family B\n")
    add(_md(rung_session_counts(old_bin, new_bin)))
    add("\n### Family A\n")
    add(_md(rung_session_counts(old_stop, new_stop)))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-per-bin", type=Path, default=DEFAULT_OLD_PER_BIN)
    parser.add_argument("--new-per-bin", type=Path, default=DEFAULT_NEW_PER_BIN)
    parser.add_argument("--old-per-stop", type=Path, default=DEFAULT_OLD_PER_STOP)
    parser.add_argument("--new-per-stop", type=Path, default=DEFAULT_NEW_PER_STOP)
    parser.add_argument(
        "--old-termlevel-per-bin",
        type=Path,
        default=DEFAULT_OLD_TERMLEVEL_PER_BIN,
        help="Family B fit under the fit-level separation flag (same spec).",
    )
    parser.add_argument(
        "--old-termlevel-per-stop",
        type=Path,
        default=DEFAULT_OLD_TERMLEVEL_PER_STOP,
        help="Family A fit under the fit-level separation flag (same spec).",
    )
    parser.add_argument("--old-contrast", type=Path, default=DEFAULT_OLD_CONTRAST)
    parser.add_argument("--new-contrast", type=Path, default=DEFAULT_NEW_CONTRAST)
    parser.add_argument("--old-audited", type=Path, default=DEFAULT_OLD_AUDITED)
    parser.add_argument("--new-audited", type=Path, default=DEFAULT_NEW_AUDITED)
    parser.add_argument(
        "--per-stop-note",
        default=(
            "**Family A caveat.** The per-stop fit that preceded this refit was "
            'overwritten in place before it could be archived, so the "before" '
            "artifact here is one refit older than the lost 2026-09-02 fit; same "
            "spec family (R0/R1/R2/Rlag)."
        ),
        help="Provenance caveat printed under the input table.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO
        / "docs"
        / "audits"
        / f"{dt.date.today().isoformat()}-rate-refit-comparison.md",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    inputs = {
        "per-bin (before)": args.old_per_bin,
        "per-bin (after)": args.new_per_bin,
        "per-stop (before)": args.old_per_stop,
        "per-stop (after)": args.new_per_stop,
    }
    missing = [str(p) for p in inputs.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing fit artifact(s): {missing}")
    termlevel_paths = {
        "old_bin": args.old_termlevel_per_bin,
        "new_bin": args.new_per_bin,
        "old_stop": args.old_termlevel_per_stop,
        "new_stop": args.new_per_stop,
        "old_contrast": args.old_contrast,
        "new_contrast": args.new_contrast,
        "old_audited": args.old_audited,
        "new_audited": args.new_audited,
    }
    termlevel_missing = [str(p) for p in termlevel_paths.values() if not p.exists()]
    if termlevel_missing:
        print(f"skipping the term-level section: missing {termlevel_missing}")
        termlevel = None
    else:
        termlevel = {k: pd.read_pickle(v) for k, v in termlevel_paths.items()}
        inputs.update(
            {
                "term-level Family B (before)": args.old_termlevel_per_bin,
                "term-level Family A (before)": args.old_termlevel_per_stop,
                "is_swr-contrast (before)": args.old_contrast,
                "is_swr-contrast (after)": args.new_contrast,
                "is_swr-contrast audited (before)": args.old_audited,
                "is_swr-contrast audited (after)": args.new_audited,
            }
        )
    report = build_report(
        pd.read_pickle(args.old_per_bin),
        pd.read_pickle(args.new_per_bin),
        pd.read_pickle(args.old_per_stop),
        pd.read_pickle(args.new_per_stop),
        inputs,
        args.per_stop_note,
        termlevel,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
