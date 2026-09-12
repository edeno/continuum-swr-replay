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
  Old ``M2_rewhist`` / new ``M1_rew`` share the parent ``M0_const``: that is
  the reward block's MARGINAL step. The conditional step is the reward block
  given time and self-history — old ``M7_time_hist_rew`` over ``M6_time_hist``,
  new ``M7r_time_histcos_rew`` over ``M6r_time_histcos`` — the same three
  blocks with a raised-cosine self-history in place of the ``onset_lag``
  dummies, not a term-for-term counterpart.
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

There is also a **cohort-contrast** mode (``--cohort-contrast``), which does
not compare a refit to its predecessor at all: it places the SAME ladder fit on
the three stop populations (well-anchored ``at_well``, ``all_stops``,
``in_transit``) side by side, so a number that moves between them is moving
because the population moved. Every likelihood-scale step it prints is taken
WITHIN one cohort.

Usage:
    uv run python scripts/qa/compare_fits.py
    uv run python scripts/qa/compare_fits.py --out docs/audits/my-comparison.md
    uv run python scripts/qa/compare_fits.py --cohort-contrast
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import wilcoxon

from continuum_swr_replay.analysis.replay._fit_diagnostics import (
    SEPARATION_COEF_ABS,
)
from continuum_swr_replay.analysis.replay.contrast_aggregation import (
    separated_for_term,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    ARRIVAL_CLOCK_COLUMN,
    ARRIVAL_SPLINE_KNOTS,
    BIN_S,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_BASIS,
    REWARD_HISTORY_N_LAGS,
    SPLINE_KNOTS,
    SWR_CONTRAST_CONTENT_ARM_KEY_TERM,
    SWR_CONTRAST_CONTENT_PATCH_KEY_TERM,
    TMAX,
    TMAX_ARRIVAL,
)
from continuum_swr_replay.analysis.replay.per_bin_design import (
    reconstruct_reward_kernels,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.session_keys import SESSION_COLUMNS
from continuum_swr_replay.table_io import read_table, table_exists

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
    "full": ("M7r_time_histcos_rew", "M6r_time_histcos"),
}
# Family A: the new single reward rung, and the two old rungs it replaced.
OLD_PER_STOP_STEPS = {"R2 vs R0": ("R2", "R0"), "Rlag vs R0": ("Rlag", "R0")}
NEW_PER_STOP_STEPS = {"R1 vs R0": ("R1", "R0")}

OUTCOME_ORDER = ["swr", "non_local", "non_local_swr", "non_local_no_swr"]
# Current-outcome strata the SWR-content contrast runner publishes.
CONTRAST_OUTCOME_STRATA = ("all", "rewarded", "omitted")
# The rung both reward quantities below are read from.
FULL_MODEL = "M7r_time_histcos_rew"
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


# ── Arrival-clock sub-ladder and the outcome-boundary reward shift ────────────
#
# ``M5a_time_arrival`` re-anchors the same time-course spline block on the
# arrival clock and is fit on the arrival window, a strict subset of the frozen
# cohort's rows. Its likelihood-scale numbers therefore live on a different row
# set from ``M5_time``'s and are never differenced. Two things ARE readable side
# by side: the held-out ``cv_bits_per_event``, which is a per-event rate rather
# than a total, and the SHAPE of the two fitted time courses, each evaluated on
# its own window and compared on a common grid of RELATIVE time (fraction of the
# window). Neither is a nested test; both are descriptive.
STOP_TIME_RUNG = "M5_time"
ARRIVAL_TIME_RUNG = "M5a_time_arrival"
# The rung whose reward block moves when the per-bin reward-history features are
# recomputed at the outcome boundary instead of held at their bout-start values.
BOUNDARY_REWARD_RUNG = "M7r_time_histcos_rew"
TIME_COURSE_GRID_POINTS = 401


def _spline_coef_columns(fits: pd.DataFrame, clock: str) -> list[str]:
    """Fitted spline coefficients of the time-course block, in basis order.

    The block's patsy term carries its own knots and bounds in the column name,
    so the arrival and stop clocks are told apart by the variable inside
    ``bs(...)``. Interaction columns (``reward:bs(...)``) carry a colon and are
    excluded: they are a different block.
    """
    prefix = f"coef_bs({clock},"
    cols = [
        c
        for c in fits.columns
        if isinstance(c, str)
        and c.startswith(prefix)
        and c.endswith("]")
        and ":" not in c
    ]
    return sorted(cols, key=lambda c: int(c.rsplit("[", 1)[1][:-1]))


def _support_window(knots: Sequence[float]) -> tuple[float, float]:
    """The part of a spline's window the summary is allowed to read.

    A cubic B-spline is only pinned by data between its knots; outside the
    outermost interior knot the fitted curve is an extrapolation held up by very
    little exposure, and it can run away there. Two independent quantities are
    read off these curves — where the rate peaks, and how the two clocks'
    normalized shapes differ — and both are destroyed by an unsupported edge
    excursion: the argmax lands on the boundary grid point and the normalizing
    integral is dominated by mass that almost no bin contributes to.

    The exposure per grid cell is the honest support measure, but it lives in
    the per-bin design, which this script does not read — it reads fitted
    coefficients only. So the window is a structural proxy: the curve is read
    **between the outermost interior knots** — ``[1, 3]`` s on the stop clock
    (knots 1/2/3 over ``[0, 4]``) and ``[1, 6]`` s on the arrival clock (knots
    1/2/4/6 over ``[0, 8]``). Each end drops one boundary-knot span, the region
    where the basis is carried by a single extrapolating B-spline. It is a rule
    of thumb, not a measurement, and the audit says so beside the table.
    """
    return float(knots[0]), float(knots[-1])


def _fitted_time_courses(
    fits: pd.DataFrame,
    model: str,
    outcome: str,
    clock: str,
    knots: Sequence[float],
    tmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-session fitted rate over the clock's SUPPORTED interior.

    The knots are fixed (not data-driven), so the basis is a deterministic
    function of the clock and the curve is reconstructable from the saved
    coefficients: ``rate(t) = exp(intercept + B(t) @ spline_coef)``, with every
    other block at its reference value. The basis is built over the rung's real
    bounds and then evaluated only on :func:`_support_window`, so a coefficient
    is never re-fit — the curve is simply not read where it is unsupported.

    A session is dropped when its fit is flagged ``separated`` (an unidentified
    non-nuisance term makes its coefficients meaningless, not merely uncertain)
    or when its reconstructed curve is not everywhere finite (``exp`` of a
    runaway coefficient overflows).

    Returns
    -------
    grid : ndarray, shape (TIME_COURSE_GRID_POINTS,)
        Clock values in seconds spanning the supported interior.
    curves : ndarray, shape (n_sessions, TIME_COURSE_GRID_POINTS)
        One finite fitted curve per retained session.
    """
    grid = np.linspace(*_support_window(knots), TIME_COURSE_GRID_POINTS)
    columns = _spline_coef_columns(fits, clock)
    sub = fits[(fits.model == model) & (fits.outcome == outcome)]
    if not columns or sub.empty or "coef_Intercept" not in sub:
        return grid, np.empty((0, grid.size))
    beta = sub[columns].to_numpy(dtype=float)
    intercept = sub["coef_Intercept"].to_numpy(dtype=float)
    ok = np.isfinite(beta).all(axis=1) & np.isfinite(intercept)
    if "separated" in sub:
        ok &= ~(sub["separated"] == True).to_numpy()  # noqa: E712
    if not ok.any():
        return grid, np.empty((0, grid.size))
    basis = np.asarray(
        dmatrix(
            f"bs(x, knots={tuple(knots)}, lower_bound=0.0, upper_bound={tmax}) - 1",
            {"x": grid},
            return_type="dataframe",
        )
    )
    if basis.shape[1] != beta.shape[1]:
        raise ValueError(
            f"{model}: the saved fit carries {beta.shape[1]} spline coefficients "
            f"but the basis rebuilt from knots={tuple(knots)} over [0, {tmax}] "
            f"has {basis.shape[1]} columns. The fit was made with different "
            "knots than the ones this comparison reconstructs."
        )
    with np.errstate(over="ignore"):
        curves = np.exp(intercept[ok, None] + beta[ok] @ basis.T)
    return grid, curves[np.isfinite(curves).all(axis=1)]


def _density_cdf(grid: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """``curve`` read as a density over its own window, as a CDF on that window.

    Normalizing removes the baseline rate, which differs between the two clocks
    because they are fit to different row sets. What survives is the shape.
    """
    increments = np.diff(grid) * 0.5 * (curve[1:] + curve[:-1])
    cdf = np.concatenate([[0.0], np.cumsum(increments)])
    total = cdf[-1]
    if not np.isfinite(total) or total <= 0:
        return np.full(cdf.shape, np.nan)
    return cdf / total


def arrival_clock_cv_table(fits: pd.DataFrame) -> pd.DataFrame:
    """Held-out bits per event for the stop-clock and arrival-clock time rungs.

    ``cv_bits_per_event`` is normalized per held-out event, so the two rungs'
    values are on one scale even though they are fit to different numbers of
    bins. The paired column is a within-session difference over the sessions
    where both are finite; it is descriptive, not a nested-model test.
    """
    rows = []
    for outcome in OUTCOME_ORDER:
        sides = {}
        for label, rung in (("stop", STOP_TIME_RUNG), ("arrival", ARRIVAL_TIME_RUNG)):
            side = fits[(fits.model == rung) & (fits.outcome == outcome)]
            sides[label] = side[SC + ["cv_bits_per_event"]] if len(side) else side
        if all(len(side) == 0 for side in sides.values()):
            continue
        row = {"outcome": outcome}
        for label, side in sides.items():
            finite = (
                np.isfinite(side["cv_bits_per_event"])
                if len(side)
                else pd.Series(dtype=bool)
            )
            row[f"n_fits_{label}"] = int(len(side))
            row[f"pct_cv_finite_{label}"] = (
                100 * float(finite.mean()) if len(side) else np.nan
            )
            row[f"median_cv_bits_{label}"] = (
                side.loc[finite, "cv_bits_per_event"].median() if len(side) else np.nan
            )
        merged = (
            sides["stop"].merge(sides["arrival"], on=SC, suffixes=("_stop", "_arrival"))
            if len(sides["stop"]) and len(sides["arrival"])
            else pd.DataFrame()
        )
        if len(merged):
            difference = (
                (merged["cv_bits_per_event_arrival"] - merged["cv_bits_per_event_stop"])
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
        else:
            difference = pd.Series(dtype=float)
        row["n_paired_cv_finite"] = int(len(difference))
        row["median_delta_bits_arrival_minus_stop"] = (
            difference.median() if len(difference) else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _separated_share(fits: pd.DataFrame, model: str, outcome: str) -> float:
    """Share of a rung's session-fits flagged ``separated`` for one population."""
    sub = fits[(fits.model == model) & (fits.outcome == outcome)]
    if not len(sub) or "separated" not in sub:
        return np.nan
    return 100 * float((sub["separated"] == True).mean())  # noqa: E712


def arrival_clock_shape_table(fits: pd.DataFrame) -> pd.DataFrame:
    """Shape of the two fitted time courses: peak position and KS distance.

    Both quantities are read on each rung's SUPPORTED interior only
    (:func:`_support_window`): the peak is the argmax of the across-session
    median curve there, and the KS distance compares the two curves after each
    is normalized to a density over its own supported interior and read on a
    common grid of relative position within it. Reading either quantity over the
    full window lets an unsupported edge excursion decide it — the argmax lands
    on the boundary grid point and the normalizing integral is dominated by mass
    almost no bin contributes to.

    The peak is reported both in seconds on the rung's own clock and as a
    fraction of the supported interval, because the two intervals differ in
    length. ``pct_separated`` is the share of that rung's session-fits excluded
    for carrying an unidentified non-nuisance term, reported so a shrinking
    ``n_sessions`` is attributable.
    """
    rows = []
    for outcome in OUTCOME_ORDER:
        stop_grid, stop_curves = _fitted_time_courses(
            fits, STOP_TIME_RUNG, outcome, "tc", SPLINE_KNOTS, TMAX
        )
        arr_grid, arr_curves = _fitted_time_courses(
            fits,
            ARRIVAL_TIME_RUNG,
            outcome,
            ARRIVAL_CLOCK_COLUMN,
            ARRIVAL_SPLINE_KNOTS,
            TMAX_ARRIVAL,
        )
        if not len(stop_curves) and not len(arr_curves):
            continue
        row: dict[str, object] = {"outcome": outcome}
        cdfs = {}
        for label, rung, grid, curves in (
            ("stop", STOP_TIME_RUNG, stop_grid, stop_curves),
            ("arrival", ARRIVAL_TIME_RUNG, arr_grid, arr_curves),
        ):
            row[f"n_sessions_{label}"] = int(len(curves))
            row[f"pct_separated_{label}"] = _separated_share(fits, rung, outcome)
            row[f"support_s_{label}"] = f"{grid[0]:.1f}-{grid[-1]:.1f}"
            if not len(curves):
                row[f"peak_s_{label}"] = np.nan
                row[f"peak_frac_support_{label}"] = np.nan
                cdfs[label] = None
                continue
            median_curve = np.median(curves, axis=0)
            peak = float(grid[int(np.argmax(median_curve))])
            row[f"peak_s_{label}"] = peak
            row[f"peak_frac_support_{label}"] = (peak - grid[0]) / (grid[-1] - grid[0])
            cdfs[label] = _density_cdf(grid, median_curve)
        if cdfs["stop"] is None or cdfs["arrival"] is None:
            row["ks_distance"] = np.nan
        else:
            row["ks_distance"] = float(
                np.nanmax(np.abs(cdfs["stop"] - cdfs["arrival"]))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def cv_nan_share_table(fits: pd.DataFrame) -> pd.DataFrame:
    """Share of session-fits whose ``cv_bits_per_event`` is not finite, per rung.

    A fit whose cross-validation is NaN carries no held-out evidence at all, and
    a rung that is NaN on nearly every session cannot take part in the ladder's
    held-out comparison. The share is reported per population as well as pooled,
    because the four populations differ in how sparse their events are and a
    rung can be scoreable on one and not another.

    Parameters
    ----------
    fits : pd.DataFrame
        Tidy per-session per-bin fits carrying ``model``, ``outcome`` and
        ``cv_bits_per_event``.

    Returns
    -------
    pd.DataFrame
        One row per rung: the pooled fit count and NaN share, then the NaN share
        within each population.
    """
    rows = []
    for model in sorted(fits["model"].unique()):
        rung = fits[fits.model == model]
        row: dict[str, object] = {
            "model": model,
            "n_fits": int(len(rung)),
            "pct_cv_nan_pooled": 100
            * float((~np.isfinite(rung["cv_bits_per_event"])).mean()),
        }
        for outcome in OUTCOME_ORDER:
            side = rung.loc[rung.outcome == outcome, "cv_bits_per_event"]
            row[f"pct_cv_nan_{outcome}"] = (
                100 * float((~np.isfinite(side)).mean()) if len(side) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def arrival_cohort_table(fits: pd.DataFrame) -> pd.DataFrame:
    """What requiring the arrival clock cost, and what its window then kept.

    Read from ``fits.attrs["arrival_clock_cohort"]``, which the runner attaches
    per population: the bins and stops the arrival column removed from the
    COMMON cohort (they leave every rung, not only the arrival ones), then the
    share of that frozen cohort the ``[0, TMAX_ARRIVAL]`` window retained. The
    event-bin share is the one that matters for reading the arrival ladder: the
    populations do not retain the same fraction of their events, so an
    arrival-ladder contrast between two populations is partly a contrast between
    two windows.
    """
    cohort = fits.attrs.get("arrival_clock_cohort")
    if not cohort:
        return pd.DataFrame()
    rows = []
    for outcome in OUTCOME_ORDER:
        counts = cohort.get(outcome)
        if not counts or not counts.get("n_bins_frozen"):
            continue
        before_freeze = counts["n_bins_frozen"] + counts["n_bins_dropped_arrival_clock"]
        rows.append(
            {
                "outcome": outcome,
                "n_bins_dropped_arrival_clock": int(
                    counts["n_bins_dropped_arrival_clock"]
                ),
                "pct_bins_dropped_arrival_clock": 100
                * counts["n_bins_dropped_arrival_clock"]
                / before_freeze,
                "n_stops_dropped_arrival_clock": int(
                    counts["n_stops_dropped_arrival_clock"]
                ),
                "n_bins_frozen": int(counts["n_bins_frozen"]),
                "n_bins_arrival_window": int(counts["n_bins_arrival_window"]),
                "retained_share_arrival_bins": counts["n_bins_arrival_window"]
                / counts["n_bins_frozen"],
                "n_event_bins_frozen": int(counts["n_event_bins_frozen"]),
                "n_event_bins_arrival_window": int(
                    counts["n_event_bins_arrival_window"]
                ),
                "retained_share_arrival_events": counts["n_event_bins_arrival_window"]
                / counts["n_event_bins_frozen"],
            }
        )
    return pd.DataFrame(rows)


def outcome_boundary_shift_cohort(stop_path: Path) -> pd.DataFrame:
    """Which stops the outcome-boundary shift applies to, on the canonical table.

    Two predicates, counted on the stop table the fits were built from:
    a stop **crosses** its outcome boundary when
    ``stop_start < outcome_known_time_s <= stop_end``, and a stop **spans an
    arrival** when the next stop of the same session (ordered by ``stop_start``)
    arrived inside ``(stop_start, stop_end]``. The shift applies to the crossing
    stops except the arrival-spanning ones, whose completed trip changes partway
    through the bout — and only where the design still has a bin to shift, i.e.
    a retained ``BIN_S`` bin whose left edge is at or past the boundary within
    the ``TMAX`` truncation (``n_shifted``).

    Parameters
    ----------
    stop_path : Path
        Canonical stop table.

    Returns
    -------
    pd.DataFrame
        One row per ``occupancy_state`` plus an ``all`` row, with the stop
        counts and the crossing share.
    """
    stops = read_table(stop_path)
    crosses = (
        (stops["stop_start"] < stops["outcome_known_time_s"])
        & (stops["outcome_known_time_s"] <= stops["stop_end"])
    ).fillna(False)
    ordered = stops.sort_values(SC + ["stop_start"], kind="stable")
    next_arrival = ordered.groupby(SC, sort=False, observed=True)[
        "last_well_arrival_time_s"
    ].shift(-1)
    spans = (
        (next_arrival > ordered["stop_start"]) & (next_arrival <= ordered["stop_end"])
    ).fillna(False)
    spans = spans.reindex(stops.index)
    # A crossing stop only moves if a retained bin starts at or past the
    # boundary: the first such bin is ceil(delta / BIN_S), and it survives the
    # exposure filter only inside min(immobility, TMAX).
    delta = (stops["outcome_known_time_s"] - stops["stop_start"]).to_numpy(float)
    exposed = np.minimum(stops["immobility_duration_s"].to_numpy(float), float(TMAX))
    with np.errstate(invalid="ignore"):
        has_bin = (exposed - np.ceil(delta / BIN_S) * BIN_S) > 1e-4
    has_bin = pd.Series(np.nan_to_num(has_bin, nan=False).astype(bool), stops.index)
    shifts = crosses & ~spans & has_bin
    rows = []
    groups = list(stops.groupby("occupancy_state", observed=True).groups.items())
    for state, index in [*groups, ("all", stops.index)]:
        c, s = crosses.loc[index], spans.loc[index]
        rows.append(
            {
                "occupancy_state": str(state),
                "n_stops": int(len(index)),
                "n_crossing_boundary": int(c.sum()),
                "pct_crossing": 100.0 * float(c.mean()),
                "n_spanning_arrival": int(s.sum()),
                "n_crossing_and_spanning": int((c & s).sum()),
                "n_shifted": int(shifts.loc[index].sum()),
            }
        )
    return pd.DataFrame(rows)


def reward_boundary_shift_table(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """The reward-history coefficients before and after the boundary shift.

    Before, every bin of a stop carried the reward features as of the bout
    start; after, a stop that spans the moment its trial's outcome becomes known
    carries the recomputed features from that bin on. Same rung, same basis,
    same lag matrix — only the per-bin values move — so the three coefficients
    are compared term for term, paired within session.
    """
    rows = []
    for outcome in OUTCOME_ORDER:
        for term in REWARD_HISTORY_COLUMNS:
            column = f"coef_{term}"
            sides = []
            for fits in (old, new):
                side = fits[
                    (fits.model == BOUNDARY_REWARD_RUNG) & (fits.outcome == outcome)
                ]
                sides.append(
                    side[SC + [column]]
                    if len(side) and column in side
                    else pd.DataFrame()
                )
            before, after = sides
            if not len(before) and not len(after):
                continue
            merged = (
                before.merge(after, on=SC, suffixes=("_before", "_after"))
                if len(before) and len(after)
                else pd.DataFrame()
            )
            row: dict[str, object] = {
                "outcome": outcome,
                "term": term,
                "n_sessions_before": int(before[column].notna().sum())
                if len(before)
                else 0,
                "median_before": before[column].median() if len(before) else np.nan,
                "n_sessions_after": int(after[column].notna().sum())
                if len(after)
                else 0,
                "median_after": after[column].median() if len(after) else np.nan,
            }
            difference = (
                (merged[f"{column}_after"] - merged[f"{column}_before"]).dropna()
                if len(merged)
                else pd.Series(dtype=float)
            )
            row["n_paired"] = int(len(difference))
            row["median_delta"] = difference.median() if len(difference) else np.nan
            row["pct_sessions_moved"] = (
                100 * float((difference.abs() > 1e-9).mean())
                if len(difference)
                else np.nan
            )
            row["signrank_p"] = (
                f"{wilcoxon(difference).pvalue:.1e}"
                if len(difference) >= 2 and float(difference.abs().max()) > 0
                else "—"
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _suffixed_side(
    df: pd.DataFrame, keys: Sequence[str], cols: Sequence[str], suffix: str
) -> pd.DataFrame:
    """``df`` restricted to ``cols``, with every non-key column suffixed.

    The join half of every before/after table in this module: keep the join keys
    unsuffixed so the two sides merge on them, and tag everything else with which
    fit it came from. Columns absent from ``df`` are simply not carried, so a
    frame that predates a column joins without a KeyError.

    An EMPTY side still contributes its suffixed columns, with no rows, so the
    join leaves them all-NaN rather than dropping them: a side that produced
    nothing is a result to display (that stratum or fit is empty), and a table
    that silently loses its columns hides it.
    """
    keys = list(keys)
    if df.empty:
        return pd.DataFrame(
            columns=[*keys, *(f"{c}{suffix}" for c in cols if c not in keys)]
        )
    have = [c for c in cols if c in df.columns]
    return df[have].rename(columns={c: f"{c}{suffix}" for c in have if c not in keys})


def session_counts_by(
    old: pd.DataFrame, new: pd.DataFrame, keys: Sequence[str]
) -> pd.DataFrame:
    """Sessions fitted per ``keys`` group in each fit; a group absent shows as 0."""
    keys = list(keys)

    def _counts(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=keys + [f"n_sessions{suffix}"])
        return (
            df.groupby(keys, sort=False)
            .size()
            .rename(f"n_sessions{suffix}")
            .reset_index()
        )

    merged = _counts(old, "_before").merge(_counts(new, "_after"), on=keys, how="outer")
    count_columns = ["n_sessions_before", "n_sessions_after"]
    # An empty side contributes an object-dtype column, so coerce before filling:
    # a bare fillna on object dtype is a deprecated silent downcast.
    merged[count_columns] = (
        merged[count_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )
    return merged


def rung_session_counts(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Sessions fitted per (rung, outcome) in each fit; rungs absent show as 0."""
    merged = session_counts_by(old, new, ["model", "outcome"])
    merged["outcome"] = pd.Categorical(merged["outcome"], OUTCOME_ORDER, ordered=True)
    return merged.sort_values(["model", "outcome"]).reset_index(drop=True)


# Column-name stems whose values are counts, not measurements. An outer join
# floats them (a group missing on one side is NaN), and rendering a session count
# as "218.0000" beside a coefficient invites reading it as a measurement.
COUNT_COLUMN_STEMS = (
    "n_sessions",
    "n_animals",
    "n_stops",
    "n_bins",
    "n_event",
    "n_dropped",
    "n_paired",
    "n_fit",
    "n_flagged",
    "n_runaway",
    "child_better_aic",
    "parent_better_aic",
    "child_better_bic",
)


def _is_count_column(name: object) -> bool:
    """Whether a column name is a count (rendered without decimals)."""
    return isinstance(name, str) and name.startswith(COUNT_COLUMN_STEMS)


def _md(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    """Markdown table; NaN renders as an em dash, count columns as integers."""
    if df.empty:
        return "_(no rows)_\n"
    shown = df.copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            fmt = "{:.0f}" if _is_count_column(col) else floatfmt
            shown[col] = shown[col].map(
                lambda v, f=fmt: "—" if pd.isna(v) else f.format(v)
            )
        else:
            # An outer join puts NaN into label columns too (a spec present on
            # only one side). str() would render it as a literal "nan", which
            # reads as a value rather than as "this side has no row".
            shown[col] = shown[col].map(lambda v: "—" if pd.isna(v) else v)
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


def _invocation() -> str:
    """This run's command line, relative to the repo root and shell-quoted."""
    script = Path(sys.argv[0])
    try:
        script = script.resolve().relative_to(REPO)
    except ValueError:
        pass
    return shlex.join(["uv", "run", "python", str(script), *sys.argv[1:]])


def build_report(
    old_bin: pd.DataFrame,
    new_bin: pd.DataFrame,
    old_stop: pd.DataFrame,
    new_stop: pd.DataFrame,
    inputs: dict[str, Path],
    per_stop_note: str,
    termlevel: dict[str, pd.DataFrame] | None = None,
    cv_audit_bin: pd.DataFrame | None = None,
    command: str | None = None,
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
    cv_audit_bin : pd.DataFrame, optional
        A per-bin fit whose per-rung, per-population cross-validation validity
        is tabulated. Use it to report rungs that are absent from ``new_bin``
        because that measurement retired them; defaults to ``new_bin``.
    command : str, optional
        The invocation that produced the report, printed in the header so the
        audit records which artifacts it was pointed at and can be reproduced.
    """
    lines: list[str] = []
    add = lines.append
    add("# Rate refit comparison\n")
    add(
        "Generated by `scripts/qa/compare_fits.py`, comparing a backed-up rate "
        "fit against the current one. Rungs whose **specification** differs "
        "between the two sides are not compared term for term; what is compared "
        "is the joint reward step (ΔAIC of a reward rung against its reward-free "
        "parent, negative = reward block helps), the reward-free `M5_time` "
        "goodness-of-fit, the frozen-cohort size, and the per-rung session "
        "counts. Where the two sides share a rung and a basis and only the "
        "per-bin feature VALUES moved — the outcome-boundary recomputation — the "
        "coefficients are paired within session and compared directly. The "
        "arrival-clock rungs are fit on a subset of the frozen cohort's rows and "
        "are never differenced against a stop-clock rung; they are reported by "
        "held-out bits per event and by the shape of the fitted time course.\n"
    )
    if command:
        add(f"Command: `{command}`\n")
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
            "Reward block, marginal (old `M2_rewhist` − `M0_const`; new "
            "`M1_rew` − `M0_const`)",
        ),
        (
            "full",
            "Reward block given time + self-history (old `M7_time_hist_rew` − "
            "`M6_time_hist`; new `M7r_time_histcos_rew` − `M6r_time_histcos`)",
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

    add(
        "\n## Family B — reward history at the outcome boundary "
        f"(`{BOUNDARY_REWARD_RUNG}`)\n"
    )
    add(
        "Before, every bin of a stop carried the reward-history features as of "
        "the **bout start**; after, a stop that spans the moment its trial's "
        "outcome becomes known carries features **recomputed at that boundary** "
        "from the shifted lag matrix. Same rung, same basis, same lag matrix — "
        "only the per-bin values move — so the three coefficients are compared "
        "term for term, paired within session. `pct_sessions_moved` is the share "
        "of paired sessions whose coefficient changed at all: a session with no "
        "boundary-spanning stop is expected to be unchanged.\n"
    )
    add(_md(reward_boundary_shift_table(old_bin, new_bin), floatfmt="{:.3f}"))

    stop_path = RC / "stop_table.pkl"
    if table_exists(stop_path):
        add(
            "\nWhich stops the shift applies to, on the canonical stop table. A "
            "stop **crosses** the boundary when `stop_start < "
            "outcome_known_time_s <= stop_end`; it **spans an arrival** when the "
            "next stop of the session arrived inside `(stop_start, stop_end]`, "
            "which is the only exclusion (the completed trip changes partway "
            "through the bout). `occupancy_state` is not part of the rule:\n"
        )
        add(_md(outcome_boundary_shift_cohort(stop_path), floatfmt="{:.2f}"))

    add(
        f"\n## Family B — arrival clock vs stop clock "
        f"(`{ARRIVAL_TIME_RUNG}` vs `{STOP_TIME_RUNG}`)\n"
    )
    add(
        f"`{ARRIVAL_TIME_RUNG}` is fit on the arrival window "
        f"`[0, {TMAX_ARRIVAL}]` s, a strict subset of the frozen cohort's rows, "
        f"so its AIC / deviance / log-likelihood are on a different row set from "
        f"`{STOP_TIME_RUNG}`'s and are **never differenced**. The two quantities "
        "below are read side by side, not as a nested test.\n"
    )
    add(
        "\n**(a) Held-out bits per event.** Normalized per held-out event, so "
        "the two rungs are on one scale despite the different bin counts. "
        "`pct_cv_finite` is the share of fits whose cross-validation survived; "
        "the paired column is the within-session difference over the sessions "
        "where both are finite.\n"
    )
    add(_md(arrival_clock_cv_table(new_bin), floatfmt="{:.4f}"))
    stop_lo, stop_hi = _support_window(SPLINE_KNOTS)
    arr_lo, arr_hi = _support_window(ARRIVAL_SPLINE_KNOTS)
    add(
        "\n**(b) Time-course shape.** Both quantities are read on each rung's "
        "**supported interior** only — **between the outermost interior "
        "knots**, so the stop clock is read on "
        f"[{stop_lo:g}, {stop_hi:g}] s (knots "
        f"{'/'.join(f'{k:g}' for k in SPLINE_KNOTS)}) and the arrival clock on "
        f"[{arr_lo:g}, {arr_hi:g}] s (knots "
        f"{'/'.join(f'{k:g}' for k in ARRIVAL_SPLINE_KNOTS)}). "
        "Outside the outermost interior knot a "
        "cubic B-spline is extrapolating over very little exposure and can run "
        "away: read over the full window, the argmax lands on a boundary grid "
        "point and the normalizing integral is dominated by mass that almost no "
        "bin contributes to. **This window is a rule of thumb, not a measured "
        "support** — the exposure per grid cell lives in the per-bin design, "
        "which this script does not read — so a peak sitting near either edge of "
        "the interval should still be checked against the exposure before it is "
        "quoted. Within the interval, each rung's across-session median fitted "
        "curve is normalized to a density and turned into a CDF; the two CDFs "
        "are read on a common grid of relative position and `ks_distance` is "
        "their maximum absolute difference. Session-fits flagged `separated` "
        "are excluded (`pct_separated` reports how many), as are curves that "
        "overflow to infinity.\n"
    )
    add(_md(arrival_clock_shape_table(new_bin), floatfmt="{:.3f}"))

    cohort = arrival_cohort_table(new_bin)
    add("\n### What the arrival clock cost the cohort, and what its window kept\n")
    add(
        "The arrival column is part of the union the per-bin cohort freeze runs "
        "over, so a bin with no arrival clock (a stop with no completed trial) "
        "leaves the frozen cohort for **every** rung, not only the arrival ones "
        "— that is `*_dropped_arrival_clock`. The arrival rungs then apply the "
        f"`[0, {TMAX_ARRIVAL}]` s window to what is left. "
        "`retained_share_arrival_events` is **not equal across populations**: "
        "the arrival ladder reads a different fraction of each population's "
        "events, so a between-population contrast on that ladder is partly a "
        "contrast between windows.\n"
    )
    add(_md(cohort, floatfmt="{:.3f}"))

    add("\n## Held-out cross-validation validity per rung and population\n")
    add(
        "Share of session-fits whose `cv_bits_per_event` is NOT finite. A fit "
        "carrying a separated non-nuisance term has its whole cross-validation "
        "voided, so a rung that is NaN on nearly every session carries no "
        "held-out evidence at all and cannot take part in the ladder's held-out "
        "comparison.\n"
    )
    add(
        _md(
            cv_nan_share_table(new_bin if cv_audit_bin is None else cv_audit_bin),
            floatfmt="{:.2f}",
        )
    )

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


# ── Content mode ──────────────────────────────────────────────────────────────
#
# The content ladders changed RESPONSE, not specification: the per-bin timebin
# count (``arm_count_hard`` / ``patch_count_hard``) was replaced by the per-event
# allocation fraction, so every event contributes one observation of unit mass
# and the fits also carry event-clustered SEs. Coefficients are therefore
# comparable term for term, but the likelihood-scale numbers are NOT: an old
# ``aic`` is stated per timebin and a new one per event, so ΔAIC before and after
# are on different scales and are printed side by side only as a scale change.

# Bundles a full content refit publishes, and the backup subdirectory each is
# compared against. The leaf-only belief chain is a partial ladder, so it lands
# in a rung-keyed directory that has no pre-fix counterpart.
CONTENT_BUNDLES = (
    "models",
    "models_patch",
    "models_swr_T",
    "models_swr_F",
    "models_patch_swr_T",
    "models_patch_swr_F",
    # The leaf-only belief chain. A partial ladder, so it publishes to a
    # rung-keyed directory named after the documented invocation
    # (--leaf-only --models M0,M0a,M1,M2,M3,M4,Mv,Mu,Mvu) and has no
    # count-response counterpart to compare against.
    "models_leaf_M0-M0a-M1-M2-M3-M4-Mv-Mu-Mvu",
)
# Bundles deliberately NOT refit, with the reason, so their absence from the
# comparison is a recorded decision rather than an oversight.
SUPERSEDED_CONTENT_BUNDLES = {
    "models_event_grain": (
        "`--grain event` collapses each event to its dominant arm and fits a "
        "conditional LOGIT, using the response column only to take that argmax. "
        "`arm_fraction` is a within-event monotone rescaling of "
        "`arm_count_hard`, so the argmax -- and therefore the whole fit -- is "
        "unchanged by the response switch: refitting would reproduce the backed "
        "up coefficients. Its purpose, an honest per-session n at the event "
        "grain, is now served by the default fraction ladder, which keeps the "
        "within-event distribution as well. Backed up, not refit."
    ),
}
# Why a particular bundle's two sides hold no rung in common. This is bundle
# history, not something the rung sets themselves can establish, so it is stated
# here and passed to the (otherwise neutral) note builder.
DISJOINT_BUNDLE_NOTES = {
    "models": (
        "A partial run had published into this canonical directory before the "
        "partial-ladder guard existed, so there is no count-response fit of the "
        "refitted rungs to compare against."
    ),
}


class ContentFraming(NamedTuple):
    """What a content comparison is about, in the prose the report opens with.

    A content comparison can be a RESPONSE change (count -> per-event fraction)
    or a COHORT change (the same response refit on a different event
    population). The tables are built the same way either way; only the reading
    instructions differ, so they live here rather than hard-coded in the report.
    """

    title: str
    change: str
    scales: str
    units: str  # format template with {covariate}, {before}, {after}


CONTENT_FRAMINGS = {
    "response": ContentFraming(
        title="Content refit comparison: count response vs per-event fraction response",
        change=(
            "Generated by `scripts/qa/compare_fits.py --content`. The content "
            "ladders changed RESPONSE, not specification: the per-timebin count "
            "(`arm_count_hard` / `patch_count_hard`) was replaced by the "
            "per-event allocation fraction (`arm_fraction` / `patch_fraction`), "
            "so every event contributes one observation of unit mass, and the "
            "fits now also carry event-clustered standard errors."
        ),
        scales=(
            "**Read the scales, not just the numbers.** Coefficients are "
            "comparable term for term. Standard errors are not: the *before* "
            "column is the model SE of a design that treated a median ~32 "
            "autocorrelated timebins per event as independent draws; the "
            "*after* column the aggregation reads is the event-clustered "
            "sandwich. Likelihood-scale quantities (AIC, ΔAIC, BIC) are stated "
            "per timebin before and per event (quasi-likelihood) after, so the "
            "before/after ΔAIC columns are **different scales** and must never "
            "be differenced or read as an effect change."
        ),
        units=(
            "**The per-SD unit also changed for the arm belief terms.** Their "
            "SDs are now taken over the leaf arms only -- the trunks are a "
            "structural zero for every one of them, so nine-arm rows measured a "
            "spread the fitted rows never see. `{covariate}` is stated per "
            "{before} (all rows) before and per {after} (leaf arms) after, read "
            "from the fits' own `std_factor`. A `std_coef_*_before` and a "
            "`std_coef_*_after` are therefore stated per DIFFERENT SD: the "
            "`std_unit` / `std_factor` columns below carry each side's unit, "
            "and the std_coef columns **must not be differenced across the "
            "change**. Compare the raw `coef_*` columns instead."
        ),
    ),
    "cohort": ContentFraming(
        title=(
            "Content contrast comparison: the same ladder before and after the "
            "well-anchored fit cohort"
        ),
        change=(
            "Generated by `scripts/qa/compare_fits.py --content`. These fits "
            "changed COHORT, not specification and not response: the "
            "`is_swr`-contrast content runner now states the at-well event "
            "cohort it previously held only implicitly, through the "
            "complete-case drop on `is_current_arm`. Every rung is now fit on "
            "one population, and the per-SD units are taken on that population."
        ),
        scales=(
            "**Read what moved.** Both sides carry the same per-event fraction "
            "response and the same event-clustered standard errors, so the "
            "columns are on one scale and coefficients are comparable term for "
            "term. What differs is the rows: a rung carrying `is_current_arm` "
            "was already confined to at-well events and barely moves, while a "
            "rung without it was fit on off-well events the rungs it is a null "
            "for never saw. Likelihood-scale quantities (AIC, ΔAIC, BIC) are "
            "computed over different row sets on the two sides and must not be "
            "differenced across them."
        ),
        units=(
            "**The per-SD unit moved with the cohort.** `{covariate}` is stated "
            "per {before} before and per {after} after, read from the fits' own "
            "`std_factor`: the SD is taken over the rows actually fit, and "
            "those rows changed. A `std_coef_*_before` and a `std_coef_*_after` "
            "are therefore stated per DIFFERENT SD -- the `std_unit` / "
            "`std_factor` columns below carry each side's unit. Compare the raw "
            "`coef_*` columns, or requote the per-SD value from the new side."
        ),
    ),
}
DEFAULT_CONTENT_BACKUP_ROOT = RC / "_pre_fix_backup_20260905"
# Terms carried by every rung above M0a purely to absorb each arm's static
# occupancy. Sorted last so the scientific terms lead the coefficient table.
CONTENT_NUISANCE_PREFIX = "is_candidate_arm_"


def _content_terms(*frames: pd.DataFrame) -> list[str]:
    """Fitted term names present in any of ``frames``, scientific terms first."""
    terms = {
        col[len("coef_") :]
        for frame in frames
        for col in frame.columns
        if col.startswith("coef_")
    }
    return sorted(terms, key=lambda t: (t.startswith(CONTENT_NUISANCE_PREFIX), t))


def _n_animals_pos(fits: pd.DataFrame, term: str) -> tuple[int, int]:
    """Animals whose median ``coef_<term>`` is positive, and animals contributing.

    Mirrors the audited aggregation's per-animal sign count, which is the primary
    population evidence for the content ladders: a median over sessions within an
    animal first, so an animal with many sessions does not outvote the others.
    """
    col = f"coef_{term}"
    if col not in fits.columns or "animal" not in fits.columns:
        return 0, 0
    per_animal = fits.dropna(subset=[col]).groupby("animal")[col].median()
    return int((per_animal > 0).sum()), int(per_animal.size)


def _rung_frame(fits: pd.DataFrame, model: str) -> pd.DataFrame:
    """One rung's fits, with the columns no session of that rung carries removed.

    A bundle's frame is the union of every rung's design, so a rung is joined to
    columns belonging to other rungs, all NaN for its own rows. Dropping them
    keeps a per-rung term list to the terms that rung was actually fit with.
    """
    if fits.empty or "model" not in fits.columns:
        return fits
    return fits[fits["model"] == model].dropna(axis=1, how="all")


def _paired_coefficients(
    old: pd.DataFrame, new: pd.DataFrame, term: str
) -> tuple[float, float, int]:
    """Medians of ``coef_<term>`` over the sessions BOTH fits reached.

    The per-side medians are taken over each side's own fitted sessions, which
    differ (e.g. 197 vs 201 sessions on ``models_swr_T``), so a shift between
    them can be a cohort change rather than an effect change. This restricts to
    the intersection on (session, rung) -- the brief's join -- so the two
    medians describe the same sessions. Returns NaNs and 0 when the rung has no
    session in common, which is what a bundle whose rung sets are disjoint looks
    like.
    """
    col = f"coef_{term}"
    keys = [*SC]
    if any(
        col not in frame.columns or not set(keys) <= set(frame.columns)
        for frame in (old, new)
    ):
        return np.nan, np.nan, 0
    merged = old[keys + [col]].merge(
        new[keys + [col]], on=keys, suffixes=("_before", "_after")
    )
    merged = merged.dropna(subset=[f"{col}_before", f"{col}_after"])
    if merged.empty:
        return np.nan, np.nan, 0
    return (
        float(merged[f"{col}_before"].median()),
        float(merged[f"{col}_after"].median()),
        len(merged),
    )


def content_term_table(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Per (model, term): median coefficient, typical SE and animal signs, both fits.

    Two things are deliberately asymmetric and labelled as such.

    The SE columns: the old count-response fits carry only the model SE
    (``bse_<term>``), computed as if an event's ~32 autocorrelated timebins were
    independent draws. The new fraction-response fits also carry the
    event-clustered sandwich SE (``bse_cluster_<term>``), which is the one the
    aggregation reads. Comparing the two says how much the old SE understated the
    events' own variability; it is not a like-for-like SE change.

    The medians: ``coef_median_before`` / ``_after`` are over each side's own
    fitted sessions, which are not the same set, while
    ``coef_median_paired_before`` / ``_after`` restrict to the ``n_paired``
    sessions both fits reached. Read the paired pair when asking whether the
    coefficient moved; the unpaired pair still reports a rung that only one side
    fitted at all.

    Parameters
    ----------
    old, new : pd.DataFrame
        Per-session content fits from the backup (count response) and from the
        current bundle (fraction response).

    Returns
    -------
    pd.DataFrame
        One row per (model, term) present in either fit.
    """
    rows = []
    models = sorted(
        set(old.get("model", pd.Series(dtype=object)))
        | set(new.get("model", pd.Series(dtype=object)))
    )
    for model in models:
        o = _rung_frame(old, model)
        n = _rung_frame(new, model)
        for term in _content_terms(o, n):
            col = f"coef_{term}"
            o_coef = o[col] if col in o.columns else pd.Series(dtype=float)
            n_coef = n[col] if col in n.columns else pd.Series(dtype=float)
            o_pos, o_animals = _n_animals_pos(o, term)
            n_pos, n_animals = _n_animals_pos(n, term)
            o_se = o.get(f"bse_{term}")
            n_model_se = n.get(f"bse_{term}")
            n_cluster_se = n.get(f"bse_cluster_{term}")
            paired_before, paired_after, n_paired = _paired_coefficients(o, n, term)
            rows.append(
                {
                    "model": model,
                    "term": term,
                    "n_sessions_before": int(o_coef.notna().sum()),
                    "n_sessions_after": int(n_coef.notna().sum()),
                    "n_paired": n_paired,
                    "coef_median_before": o_coef.median() if len(o_coef) else np.nan,
                    "coef_median_after": n_coef.median() if len(n_coef) else np.nan,
                    "coef_median_paired_before": paired_before,
                    "coef_median_paired_after": paired_after,
                    "bse_model_median_before": (
                        o_se.median() if o_se is not None else np.nan
                    ),
                    "bse_model_median_after": (
                        n_model_se.median() if n_model_se is not None else np.nan
                    ),
                    "bse_cluster_median_after": (
                        n_cluster_se.median() if n_cluster_se is not None else np.nan
                    ),
                    "n_animals_pos_before": f"{o_pos}/{o_animals}",
                    "n_animals_pos_after": f"{n_pos}/{n_animals}",
                }
            )
    return pd.DataFrame(rows)


def content_scale_table(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Estimator, response column and cohort size each fit was produced with.

    A frame that names neither predates the integrity columns; its response is
    then only recoverable from a provenance sidecar, so it is reported as
    unstated rather than assumed to be the count.
    """

    def _describe(frame: pd.DataFrame, when: str) -> dict:
        return {
            "fit": when,
            "estimator": (
                "|".join(sorted(map(str, frame["estimator"].dropna().unique())))
                if "estimator" in frame.columns
                else "(unstated)"
            ),
            "response_col": (
                "|".join(sorted(map(str, frame["response_col"].dropna().unique())))
                if "response_col" in frame.columns
                else "(unstated)"
            ),
            "n_sessions": int(frame.groupby(SC).ngroups) if len(frame) else 0,
            "n_rungs": int(frame["model"].nunique()) if len(frame) else 0,
        }

    return pd.DataFrame([_describe(old, "before"), _describe(new, "after")])


def content_rung_session_counts(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Sessions fitted per rung in each fit; a rung absent on one side shows as 0."""
    return (
        session_counts_by(old, new, ["model"])
        .sort_values("model")
        .reset_index(drop=True)
    )


def content_delta_aic_table(
    old_cmp: pd.DataFrame, new_cmp: pd.DataFrame
) -> pd.DataFrame:
    """Per nested step: median per-session ΔAIC before and after, on DIFFERENT scales.

    Positive = the child rung wins. The two columns are not commensurable: the
    count-response ΔAIC is a sum over an event's timebins (a median 32 of them,
    treated as independent draws), the fraction-response ΔAIC is one unit-mass
    observation per event on a quasi-likelihood scale. A collapse in magnitude
    from before to after is the expected consequence of the response change, not
    evidence that a rung stopped helping; read the sign and the per-session /
    per-animal win counts instead.
    """
    keys = ["parent", "child"]
    cols = keys + [
        "n_sessions",
        "delta_aic_median",
        "child_better_aic",
        "n_animals_child_better",
    ]

    return (
        _suffixed_side(old_cmp, keys, cols, "_before")
        .merge(_suffixed_side(new_cmp, keys, cols, "_after"), on=keys, how="outer")
        .reset_index(drop=True)
    )


def content_separation_table(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Per (model, term): session fits the identifiability backstops would exclude.

    Two mechanisms, counted separately because they answer different questions:
    the fitters' per-term separation flag (``separated_terms``), and the
    effect-scale magnitude backstop ``|coef| * sd >= SEPARATION_COEF_ABS`` that
    ``aggregate_contrast`` applies on top of it. Rows where neither fit excludes
    anything are dropped; an excluded share above ``MAX_UNIDENTIFIED_SHARE`` is
    what suppresses a term's aggregate row entirely.
    """
    rows = []
    models = sorted(
        set(old.get("model", pd.Series(dtype=object)))
        | set(new.get("model", pd.Series(dtype=object)))
    )
    for model in models:
        o = _rung_frame(old, model)
        n = _rung_frame(new, model)
        for term in _content_terms(o, n):
            record: dict = {"model": model, "term": term}
            excluded_any = False
            for frame, suffix in ((o, "_before"), (n, "_after")):
                ccol, sdcol = f"coef_{term}", f"sd_{term}"
                if ccol not in frame.columns:
                    record[f"n_fit{suffix}"] = 0
                    record[f"n_flagged{suffix}"] = 0
                    record[f"n_runaway{suffix}"] = 0
                    continue
                present = frame[ccol].notna()
                flagged = separated_for_term(frame, term) & present
                if sdcol in frame.columns:
                    runaway = (
                        frame[ccol].abs() * frame[sdcol] >= SEPARATION_COEF_ABS
                    ) & present
                else:
                    # Legacy artifacts predate the persisted per-fit predictor
                    # SDs; the raw coefficient is the only scale available.
                    runaway = (frame[ccol].abs() >= SEPARATION_COEF_ABS) & present
                record[f"n_fit{suffix}"] = int(present.sum())
                record[f"n_flagged{suffix}"] = int(flagged.sum())
                record[f"n_runaway{suffix}"] = int((runaway & ~flagged).sum())
                excluded_any |= bool((flagged | runaway).any())
            if excluded_any:
                rows.append(record)
    return pd.DataFrame(rows)


_NT_TWIN_RUNG = re.compile(r"^(CP?S\d+)nt(_.*)$")


def rung_sort_key(model: str) -> tuple[str, int]:
    """Sort key placing an ``nt`` twin immediately after its controlled rung.

    A twin's ``swr_x_`` coefficient is the TOTAL SWR effect and its controlled
    rung's is the length-adjusted (direct) one; the pair is only readable side
    by side, and plain alphabetical order separates them (``CS5b`` sorts between
    ``CS5_armvalue`` and ``CS5nt_armvalue``).

    Parameters
    ----------
    model : str
        Rung name.

    Returns
    -------
    (str, int)
        The controlled rung's name, and 0 for it / 1 for its twin.
    """
    match = _NT_TWIN_RUNG.match(model)
    if match is None:
        return (model, 0)
    return (f"{match.group(1)}{match.group(2)}", 1)


def contrast_audited_table(
    old: pd.DataFrame, new: pd.DataFrame, terms: set[str] | None = None
) -> pd.DataFrame:
    """Audited SWR-content rows before and after, joined on (level, model, pop, term).

    ``typical_session_se`` carries its own ``se_source`` in each fit, so the SE
    columns say which SE the median is made of rather than implying that the two
    are the same quantity.
    """
    keys = ["level", "model", "population", "term"]
    cols = keys + [
        "n_sessions",
        "n_animals",
        "n_dropped",
        "coef_animal_median",
        "n_animals_pos",
        "coef_session_median",
        "typical_session_se",
        "se_source",
        # The per-SD unit travels with the standardized coefficient. The arm
        # belief terms' SD moved onto leaf-arm support, so a std_coef before and
        # a std_coef after are stated per DIFFERENT SD and must not be
        # differenced without reading these two columns; the sizes are in the
        # std_factor columns themselves, not repeated here.
        "std_unit",
        "std_factor",
        "std_coef_animal_median",
        "session_signrank_fdr_q",
    ]

    merged = _suffixed_side(old, keys, cols, "_before").merge(
        _suffixed_side(new, keys, cols, "_after"), on=keys, how="outer"
    )
    if terms is not None:
        merged = merged[merged["term"].isin(terms)]
    # Each rung next to its ``nt`` twin: the direct and total SWR effects of one
    # term are read as a pair, not from two ends of the table.
    return (
        merged.assign(_rung_order=merged["model"].map(rung_sort_key))
        .sort_values(["level", "_rung_order", "population", "term"])
        .drop(columns="_rung_order")
        .reset_index(drop=True)
    )


def contrast_design_delta_table(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Per rung: terms fitted on only one side of the contrast comparison.

    The ladder bundles changed response and nothing else, but the ``is_swr``
    contrast ladder also gained design terms in the same round (the event-length
    products, and rungs that did not exist before). A before/after difference on
    a rung listed here is therefore NOT attributable to the response change
    alone, and the audit must say so rather than let the reader assume one cause.
    Rungs whose design is identical on both sides are omitted.
    """
    rows = []
    models = sorted(
        set(old.get("model", pd.Series(dtype=object)))
        | set(new.get("model", pd.Series(dtype=object)))
    )
    for model in models:
        o = _rung_frame(old, model)
        n = _rung_frame(new, model)
        o_terms = set(_content_terms(o)) if len(o) else set()
        n_terms = set(_content_terms(n)) if len(n) else set()
        if not len(o):
            rows.append(
                {"model": model, "only_before": "(rung not fit)", "only_after": ""}
            )
            continue
        if not len(n):
            rows.append(
                {"model": model, "only_before": "", "only_after": "(rung not fit)"}
            )
            continue
        if o_terms == n_terms:
            continue
        rows.append(
            {
                "model": model,
                "only_before": ", ".join(sorted(o_terms - n_terms)) or "—",
                "only_after": ", ".join(sorted(n_terms - o_terms)) or "—",
            }
        )
    return pd.DataFrame(rows)


def contrast_stratum_table(
    strata: dict[str, pd.DataFrame], terms: set[str]
) -> pd.DataFrame:
    """Key-term audited rows for each outcome stratum, side by side."""
    keys = ["level", "model", "population", "term"]
    # std_factor comes along so a reader can see that every stratum shares one
    # per-SD unit (frozen over the pooled cohort) before comparing std_coef.
    wanted = keys + [
        "n_sessions",
        "coef_animal_median",
        "n_animals_pos",
        "std_unit",
        "std_factor",
    ]
    merged: pd.DataFrame | None = None
    for name, df in strata.items():
        # An empty stratum keeps its (all-NaN) columns rather than vanishing from
        # the table: "this stratum fitted nothing" is the finding.
        rows = df if df.empty else df[df["term"].isin(terms)]
        side = _suffixed_side(rows, keys, wanted, f"_{name}")
        merged = side if merged is None else merged.merge(side, on=keys, how="outer")
    if merged is None:
        return pd.DataFrame()
    # Same twin-adjacent order as the audited table above.
    return (
        merged.assign(_rung_order=merged["model"].map(rung_sort_key))
        .sort_values(["level", "_rung_order", "population", "term"])
        .drop(columns="_rung_order")
        .reset_index(drop=True)
    )


def _disjoint_rungs_note(old: pd.DataFrame, new: pd.DataFrame, note: str = "") -> str:
    """Prose when a bundle's two sides share no rung, naming what each side holds.

    The note states only what the frames show: which rungs each side holds and
    that nothing in the bundle is a before/after comparison. WHY they are
    disjoint is a fact about a particular bundle's history, not something the
    rung sets can establish, so the caller supplies it.

    Parameters
    ----------
    old, new : pandas.DataFrame
        The two sides' per-session fits; ``model`` names the rung.
    note : str, optional
        Bundle-specific explanation of the disjointness, appended verbatim.

    Returns
    -------
    str
        The note, or ``""`` when the sides share a rung (or either is empty).
    """
    if old.empty or new.empty:
        return ""
    old_rungs = set(old.get("model", pd.Series(dtype=object)))
    new_rungs = set(new.get("model", pd.Series(dtype=object)))
    if not old_rungs or not new_rungs or (old_rungs & new_rungs):
        return ""
    return (
        "**No rung is present on both sides of this bundle**, so every row below "
        "reads `0 before` or `0 after` and nothing here is a before/after "
        f"comparison: the two sides fitted disjoint rung sets. The pre-fix bundle "
        f"holds {', '.join(sorted(old_rungs))}; the refit holds "
        f"{', '.join(sorted(new_rungs))}." + (f" {note}" if note else "") + "\n"
    )


def _stratum_power_note(table: pd.DataFrame, strata: Sequence[str]) -> str:
    """Prose naming how many sessions each stratum's printed rows rest on.

    A stratified refit can succeed and still be far too thin to read: the
    ``omitted`` stratum keeps a small minority of events, so most of its rungs
    are fitted on a handful of sessions.
    """
    parts = []
    for name in strata:
        col = f"n_sessions_{name}"
        if col not in table.columns:
            continue
        counts = pd.to_numeric(table[col], errors="coerce").dropna()
        if counts.empty:
            # The stratum's column is present but carries no fitted row at all.
            parts.append(f"`{name}` 0 sessions")
            continue
        parts.append(
            f"`{name}` {int(counts.min())}-{int(counts.max())} "
            f"(median {int(counts.median())})"
        )
    if not parts:
        return ""
    return (
        "**Sessions behind each stratum's rows** (min-max across the rows below): "
        + "; ".join(parts)
        + ". A stratum whose rungs rest on a handful of sessions is reported for "
        "completeness and is not population evidence.\n"
    )


# The arm covariate whose per-SD unit moved onto leaf-arm support. Named once so
# the header prose reads the sizes from the fits rather than restating them.
_ARM_BELIEF_COVARIATE = "reward_belief_mean_candidate_arm_preoutcome"


def _per_sd_units(frame: pd.DataFrame, covariate: str) -> str:
    """The per-SD unit(s) ``frame`` states ``covariate``'s coefficients in.

    Read from the audited rows' own ``std_factor`` so the prose cannot state a
    unit the fits do not use.

    Parameters
    ----------
    frame : pandas.DataFrame
        An audited contrast table (``term``, ``std_factor``).
    covariate : str
        The bare covariate; its ``swr_x_`` product is matched too.

    Returns
    -------
    str
        The distinct factors, formatted; ``"not fitted"`` when the covariate has
        no row (an empty side, or a fit that never carried the term).
    """
    if frame.empty or "std_factor" not in frame.columns:
        return "not fitted"
    rows = frame[frame["term"].isin({covariate, f"swr_x_{covariate}"})]
    factors = pd.to_numeric(rows["std_factor"], errors="coerce").dropna().unique()
    if not len(factors):
        return "not fitted"
    return ", ".join(f"{value:.4f}" for value in sorted(factors))


def build_content_report(
    bundles: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    contrast: dict[str, pd.DataFrame],
    inputs: dict[str, Path],
    missing_note: str,
    framing: ContentFraming = CONTENT_FRAMINGS["response"],
    command: str = "",
) -> str:
    """Assemble the content-refit markdown audit.

    Parameters
    ----------
    bundles : dict
        Bundle name -> ``(old_fits, new_fits, old_comparison, new_comparison)``.
    contrast : dict
        ``audited_before`` / ``audited_after`` plus one ``stratum_<name>`` entry
        per outcome stratum; any of them may be an empty frame.
    inputs : dict of str to Path
        Artifacts to list with their hashes.
    missing_note : str
        Prose naming artifacts that had no "before" or no "after" side.
    framing : ContentFraming
        Whether the two sides differ by response or by fit cohort, which sets
        the title and the reading instructions.
    command : str
        The invocation that wrote the report, recorded so it can be rerun.
    """
    lines: list[str] = []
    add = lines.append
    add(f"# {framing.title}\n")
    add(f"{framing.change}\n")
    if command:
        add(f"Command: `{command}`\n")
    add(f"{framing.scales}\n")
    before_unit = _per_sd_units(
        contrast.get("audited_before", pd.DataFrame()), _ARM_BELIEF_COVARIATE
    )
    after_unit = _per_sd_units(
        contrast.get("audited_after", pd.DataFrame()), _ARM_BELIEF_COVARIATE
    )
    add(
        framing.units.format(
            covariate=_ARM_BELIEF_COVARIATE, before=before_unit, after=after_unit
        )
        + "\n"
    )
    add("## Inputs\n")
    rows = []
    for name, path in inputs.items():
        rows.append(
            {
                "artifact": name,
                "path": (
                    str(path.relative_to(REPO))
                    if path.is_relative_to(REPO)
                    else str(path)
                ),
                "sha256[:12]": _sha256(path)[:12],
                "modified": dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(
                    timespec="seconds"
                ),
            }
        )
    add(_md(pd.DataFrame(rows)))
    if missing_note:
        add(f"\n{missing_note}\n")
    if not bundles:
        add(
            "\n**Contrast bundles only.** No ladder bundle was named, so this "
            "report compares the `is_swr`-contrast aggregates alone. A ladder "
            "bundle that was never moved into the backup root was not refit "
            "and has nothing to compare.\n"
        )
    else:
        for name, reason in SUPERSEDED_CONTENT_BUNDLES.items():
            add(f"\n**`{name}` was not refit.** {reason}\n")

    for name, (old, new, old_cmp, new_cmp) in bundles.items():
        add(f"\n## Bundle `{name}`\n")
        disjoint = _disjoint_rungs_note(
            old, new, note=DISJOINT_BUNDLE_NOTES.get(name, "")
        )
        if disjoint:
            add(disjoint)
        add("### Response and cohort\n")
        add(_md(content_scale_table(old, new)))
        add("\n### Sessions fitted per rung\n")
        add(_md(content_rung_session_counts(old, new)))
        add("\n### Per-session ΔAIC against the nested parent (different scales)\n")
        add(
            "`delta_aic_median_before` is per timebin; `delta_aic_median_after` "
            "is per event on a quasi-likelihood scale. Positive = child wins.\n"
        )
        add(_md(content_delta_aic_table(old_cmp, new_cmp)))
        add("\n### Coefficients, SEs and per-animal signs\n")
        add(_md(content_term_table(old, new), floatfmt="{:.4f}"))
        add(
            "\n### Identifiability backstops "
            "(session fits a term's aggregate would drop)\n"
        )
        add(
            f"Effect-scale backstop `|coef| * sd >= {SEPARATION_COEF_ABS}`, "
            "counted beside the fitters' own per-term separation flag. Rows "
            "where neither fit excludes anything are omitted.\n"
        )
        add(_md(content_separation_table(old, new), floatfmt="{:.0f}"))

    add("\n## SWR-content contrast (audited aggregate)\n")
    design_delta = contrast_design_delta_table(
        contrast.get("fits_before", pd.DataFrame()),
        contrast.get("fits_after", pd.DataFrame()),
    )
    if not design_delta.empty:
        add(
            "**The contrast ladder also changed DESIGN, not only response.** The "
            "rungs below were fit on different term sets before and after, so a "
            "before/after difference on any of them is not attributable to the "
            "response change alone. Read them as a refit of a changed model.\n"
        )
        add(_md(design_delta))
    add(
        "\nThe key terms of the `is_swr` contrast ladder, before and after. "
        "`typical_session_se` is annotated with `se_source`: `model` for a "
        "count-response fit that had no clustered SE, `cluster` where the "
        "event-clustered sandwich was available.\n"
    )
    key_terms = set(SWR_CONTRAST_CONTENT_ARM_KEY_TERM.values()) | set(
        SWR_CONTRAST_CONTENT_PATCH_KEY_TERM.values()
    )
    add(
        _md(
            contrast_audited_table(
                contrast.get("audited_before", pd.DataFrame()),
                contrast.get("audited_after", pd.DataFrame()),
                key_terms,
            ),
            floatfmt="{:.4f}",
        )
    )
    strata = {
        k[len("stratum_") :]: v for k, v in contrast.items() if k.startswith("stratum_")
    }
    if strata:
        add("\n### Outcome strata, side by side\n")
        add(
            "Each stratum is a refit of the same ladder within one current "
            "outcome; the per-SD units are frozen over the pooled cohort, so the "
            "columns are on one unit.\n"
        )
        stratum_table = contrast_stratum_table(strata, key_terms)
        power_note = _stratum_power_note(stratum_table, list(strata))
        if power_note:
            add(power_note)
        add(_md(stratum_table, floatfmt="{:.4f}"))
    return "\n".join(lines)


def _read_optional(path: Path) -> pd.DataFrame:
    """Read a fit pickle, or an empty frame when the artifact is absent."""
    return pd.read_pickle(path) if path.exists() else pd.DataFrame()


def collect_content_inputs(
    old_root: Path, new_root: Path, bundle_names: Sequence[str]
) -> tuple[dict, dict, dict, list[str]]:
    """Load every content artifact the report needs, naming the ones absent.

    A bundle with only one side is still reported (the missing side shows as an
    empty frame) so a refit that has not run yet is visible in the audit rather
    than silently omitted.

    Returns
    -------
    (bundles, contrast, inputs, missing)
        Ladder bundles, contrast frames, the artifacts to hash in the header,
        and one prose line per artifact that had no file.
    """
    bundles: dict = {}
    inputs: dict[str, Path] = {}
    missing: list[str] = []
    for name in bundle_names:
        old_dir, new_dir = old_root / name, new_root / name
        old_fits, new_fits = old_dir / "all_models.pkl", new_dir / "all_models.pkl"
        if not old_fits.exists():
            missing.append(f"`{name}`: no pre-fix bundle at `{old_fits}`")
        if not new_fits.exists():
            missing.append(f"`{name}`: no refit bundle at `{new_fits}`")
        if not (old_fits.exists() or new_fits.exists()):
            continue
        bundles[name] = (
            _read_optional(old_fits),
            _read_optional(new_fits),
            _read_optional(old_dir / "model_comparison.pkl"),
            _read_optional(new_dir / "model_comparison.pkl"),
        )
        for path, label in ((old_fits, "before"), (new_fits, "after")):
            if path.exists():
                inputs[f"{name} ({label})"] = path

    contrast: dict[str, pd.DataFrame] = {}
    audited = "swr_contrast_content_audited.pkl"
    for root, label in ((old_root, "before"), (new_root, "after")):
        path = root / "models" / audited
        contrast[f"audited_{label}"] = _read_optional(path)
        if path.exists():
            inputs[f"swr-content audited ({label})"] = path
        else:
            missing.append(f"SWR-content audited ({label}): `{path}` absent")
        # The per-session fits, for the design-delta check: the audited
        # aggregate carries one row per term and cannot say which terms a rung
        # was fit with.
        fits = root / "models" / "swr_contrast_content_arm.pkl"
        contrast[f"fits_{label}"] = _read_optional(fits)
        if fits.exists():
            inputs[f"swr-content arm fits ({label})"] = fits
    for stratum in CONTRAST_OUTCOME_STRATA:
        sfx = "" if stratum == "all" else f"_{stratum}"
        path = new_root / "models" / f"swr_contrast_content_audited{sfx}.pkl"
        if path.exists():
            contrast[f"stratum_{stratum}"] = pd.read_pickle(path)
            inputs[f"swr-content audited ({stratum})"] = path
        else:
            missing.append(f"outcome stratum `{stratum}`: `{path}` absent")
    return bundles, contrast, inputs, missing


def run_content_mode(args: argparse.Namespace) -> None:
    """Build and write the content-refit comparison."""
    bundles, contrast, inputs, missing = collect_content_inputs(
        args.old_content_root, args.new_content_root, args.content_bundles
    )
    if not inputs:
        raise SystemExit(
            f"no content artifacts found under {args.old_content_root} or "
            f"{args.new_content_root}"
        )
    note = (
        "**Artifacts with only one side.** " + "; ".join(missing) + "."
        if missing
        else ""
    )
    report = build_content_report(
        bundles,
        contrast,
        inputs,
        note,
        CONTENT_FRAMINGS[args.content_framing],
        _invocation(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"wrote {args.out}")


# ── Cohort-contrast mode ──────────────────────────────────────────────────────
#
# The rate ladders can be fit on three stop populations, and only one of them is
# the population the content family describes. `at_well` (well-anchored stops)
# is that canonical cohort; `all_stops` and `in_transit` are separately
# published sensitivity bundles. Reading one bundle's numbers against another's
# is the whole point of this mode: every quantity below is the SAME estimator on
# the SAME rung, so a difference between the columns is a difference between the
# populations and nothing else. The three bundles are never pooled, and no
# ΔAIC / likelihood quantity is ever differenced ACROSS cohorts -- those live on
# different row sets. What is compared across cohorts is a per-session
# coefficient summary, a within-cohort nested step, a scale-free curve feature,
# and the cohort's own size.
COHORT_ORDER = ("at_well", "all_stops", "in_transit")
COHORT_RULE_NOTE = {
    "at_well": "well-anchored stops (`occupancy_state` at_well or mixed) — canonical",
    "all_stops": "every immobility stop — sensitivity bundle",
    "in_transit": "in-transit stops only — sensitivity bundle",
}
DEFAULT_COHORT_PER_BIN = {
    "at_well": RC / "models_per_bin_rate" / "all_models.pkl",
    "all_stops": RC / "models_per_bin_rate" / "all_models_allstops.pkl",
    "in_transit": RC / "models_per_bin_rate" / "all_models_intransit.pkl",
}
DEFAULT_COHORT_CONTRAST = {
    "at_well": RC / "models_per_bin_rate" / "swr_contrast_rate.pkl",
    "all_stops": RC / "models_per_bin_rate" / "swr_contrast_rate_allstops.pkl",
}
# The nested per-bin reward steps, child first. Both are read WITHIN a cohort.
COHORT_AIC_STEPS = (
    ("M7r_time_histcos_rew", "M6r_time_histcos"),
    ("M1_rew", "M0_const"),
)
# The is_swr-contrast bundle fits each rung on its own primary/post-outcome
# cohort; only the primary one spans the whole ladder, so the rung table reads it.
CONTRAST_PRIMARY_COHORT = "primary"
#: Rungs the primary cohort does NOT carry: they need a post-outcome covariate
#: (`rpe_postoutcome`, `reward_belief_mean_update_l1_all_ports_postoutcome`),
#: which is undefined on a stop that starts before its trial's outcome is known,
#: so they are fit on the gated cohort alongside a refit of their ladder's null.
CONTRAST_POST_OUTCOME_RUNGS = ("G5_rpe", "G7_update", "S7_rpe", "S9_update")
#: Rungs whose reward block is interacted with the within-stop time spline. The
#: main effect `coef_reward_history_rcos_0` is then the reward slope at the point
#: where EVERY spline column is zero, which for a B-spline over [0, TMAX] is off
#: the fitted window -- a boundary extrapolation, not the reward contrast. It is
#: blanked rather than printed (it produced a 39.8x cell in an earlier draft).
REWARD_X_TIME_RUNGS = ("G8_rewxtime", "S10_rewxtime")
#: The is_swr interaction on the immediate reward component. On an S rung the
#: main effect is the NON-SWR reference-level slope and this is the multiplier
#: that turns it into the SWR one, so the ladder reads as a contrast.
SWR_IMMEDIATE_COEF = f"coef_is_swr:{REWARD_HISTORY_COLUMNS[0]}"
#: Ladder letter + rung number, so ``S10_rewxtime`` sorts after ``S9_update``
#: instead of between ``S0_rate`` and ``S1_rew``.
_RATE_RUNG_NUMBER = re.compile(r"^([A-Za-z]+)(\d+)(.*)$")


def rate_rung_sort_key(model: str) -> tuple[str, int, str]:
    """Order a rate rung by its ladder letter and its NUMBER, not as text."""
    match = _RATE_RUNG_NUMBER.match(model)
    if match is None:
        return (model, 0, "")
    return (match.group(1), int(match.group(2)), match.group(3))


def _animal_balanced(frame: pd.DataFrame, column: str) -> tuple[float, int, int]:
    """Median of the per-animal medians of ``column``, and the animal sign count.

    The animal is the replication unit: a median within each animal first, so an
    animal contributing many sessions does not outvote the others. Mirrors the
    audited aggregation the ladders are read on.

    Returns
    -------
    median : float
        Median across animals of each animal's median, NaN when nothing is fit.
    n_positive, n_animals : int
        Animals whose own median is positive, and animals contributing.
    """
    if column not in frame.columns or "animal" not in frame.columns:
        return np.nan, 0, 0
    per_animal = frame.dropna(subset=[column]).groupby("animal")[column].median()
    if per_animal.empty:
        return np.nan, 0, 0
    return (
        float(per_animal.median()),
        int((per_animal > 0).sum()),
        int(per_animal.size),
    )


def _by_cohort(fits: dict[str, pd.DataFrame]) -> list[tuple[str, pd.DataFrame]]:
    """The supplied cohorts in the canonical order, skipping the absent ones."""
    return [(c, fits[c]) for c in COHORT_ORDER if c in fits and len(fits[c])]


def cohort_size_table(fits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Each cohort's fitted size per population, from the common null rung.

    Every rung of a population shares one frozen cohort, so ``M0_const`` carries
    it. ``n_event_onsets`` is the response total -- the events the intensity
    model is fit to -- not the number of non-local events in the corpus.
    """
    rows = []
    for outcome in OUTCOME_ORDER:
        for cohort, frame in _by_cohort(fits):
            sub = frame[(frame.model == "M0_const") & (frame.outcome == outcome)]
            if sub.empty:
                continue
            rows.append(
                {
                    "outcome": outcome,
                    "stop_cohort": cohort,
                    "n_sessions": len(sub),
                    "n_stops": float(sub["n_stops"].sum()),
                    "n_bins": float(sub["n_bins"].sum()),
                    "n_event_onsets": float(sub["n_events"].sum()),
                }
            )
    return pd.DataFrame(rows)


def reward_kernel_lag1(frame: pd.DataFrame) -> pd.Series:
    """Per-session reconstructed reward kernel at lag 1, indexed by animal.

    This is the PUBLISHED functional. Each session's three basis coefficients
    are mapped through the ROW-NORMALIZED reward basis and read at lag 1, whose
    blend over the three bumps is ``[0.325, 0.066, 0]`` -- so it is a weighted
    mixture of the whole block, not one coefficient. ``docs/rate_models.md``
    and the 19c claim row quote this number; they do not quote
    ``coef_reward_history_rcos_0``.

    Parameters
    ----------
    frame : pandas.DataFrame
        One rung's per-session fits, carrying ``animal`` and the three
        ``coef_reward_history_rcos_*`` columns.

    Returns
    -------
    pandas.Series
        Log-rate kernel value at lag 1 per fitted session, indexed by animal.
        Empty when the block was not fit.
    """
    columns = [f"coef_{c}" for c in REWARD_HISTORY_COLUMNS]
    if not set(columns) <= set(frame.columns) or "animal" not in frame.columns:
        return pd.Series(dtype=float)
    fitted = frame.dropna(subset=columns)
    if fitted.empty:
        return pd.Series(dtype=float)
    kernels = reconstruct_reward_kernels(
        fitted[columns].to_numpy(dtype=float),
        REWARD_HISTORY_N_LAGS,
        REWARD_HISTORY_N_BASIS,
    )
    return pd.Series(kernels[:, 0], index=fitted["animal"].to_numpy())


def cohort_reward_rr_table(fits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Two DIFFERENT reward rate ratios per population and cohort.

    They are not interchangeable and must never be quoted as one number:

    * ``rr_immediate`` -- ``exp`` of the animal-balanced median
      ``coef_reward_history_rcos_0``, the full 0 -> 1 contrast on the first
      raised-cosine bump. The cohort audit reads it because it is one
      coefficient with one definition in all three cohorts.
    * ``rr_lag1_*`` -- ``exp`` of the reconstructed kernel at lag 1
      (:func:`reward_kernel_lag1`), the published functional, given both
      session-pooled and animal-balanced because the rate docs report the first
      and the audited aggregates the second.

    Medians commute with ``exp``, so exponentiating the median kernel is the
    same number as the median of the exponentiated kernels.
    """
    rows = []
    for outcome in OUTCOME_ORDER:
        for cohort, frame in _by_cohort(fits):
            sub = frame[(frame.model == FULL_MODEL) & (frame.outcome == outcome)]
            if sub.empty:
                continue
            median, n_positive, n_animals = _animal_balanced(sub, IMMEDIATE_COEF)
            lag1 = reward_kernel_lag1(sub)
            per_animal = (
                lag1.groupby(level=0).median() if len(lag1) else pd.Series(dtype=float)
            )
            rows.append(
                {
                    "outcome": outcome,
                    "stop_cohort": cohort,
                    "n_sessions": (
                        int(sub[IMMEDIATE_COEF].notna().sum())
                        if IMMEDIATE_COEF in sub
                        else 0
                    ),
                    "rr_immediate": float(np.exp(median)),
                    "animals_pos_immediate": f"{n_positive}/{n_animals}",
                    "rr_lag1_session": (
                        float(np.exp(lag1.median())) if len(lag1) else np.nan
                    ),
                    "rr_lag1_animal": (
                        float(np.exp(per_animal.median()))
                        if len(per_animal)
                        else np.nan
                    ),
                    "animals_pos_lag1": (
                        f"{int((per_animal > 0).sum())}/{per_animal.size}"
                        if len(per_animal)
                        else "—"
                    ),
                }
            )
    return pd.DataFrame(rows)


def cohort_paired_swr_table(fits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The SWR-coupled vs non-SWR immediate-coefficient test, once per cohort.

    Reuses :func:`reward_quantities`, the same within-session paired comparison
    the refit audit reports, so the three cohorts are read with one estimator.
    """
    rows = []
    for cohort, frame in _by_cohort(fits):
        _, paired = reward_quantities(frame)
        if paired.empty:
            continue
        rows.append({"stop_cohort": cohort, **paired.iloc[0].to_dict()})
    return pd.DataFrame(rows)


def cohort_delta_aic_table(fits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Each nested reward step, WITHIN each cohort (negative = child better).

    The difference is always taken between two rungs of one cohort, which share
    rows; the table simply places the three within-cohort steps side by side.
    """
    rows = []
    for rung, parent in COHORT_AIC_STEPS:
        for outcome in OUTCOME_ORDER:
            for cohort, frame in _by_cohort(fits):
                paired, _ = delta_aic(frame, rung, parent)
                sub = paired[paired.outcome == outcome] if len(paired) else paired
                if not len(sub):
                    continue
                rows.append(
                    {
                        "step": f"{rung} − {parent}",
                        "outcome": outcome,
                        "stop_cohort": cohort,
                        "n_sessions": len(sub),
                        "median_delta_aic": float(sub["delta_aic"].median()),
                        "pct_sessions_child_better": 100
                        * float((sub["delta_aic"] < 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def cohort_burst_peak_table(fits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Where the smooth within-stop time course peaks, per population and cohort.

    The peak of the across-session median curve of the reward-free time rung,
    read only on the spline's supported interior (:func:`_support_window`). It
    is a scale-free feature of one cohort's own fit, so it is comparable across
    cohorts even though their likelihoods are not.
    """
    rows = []
    for outcome in OUTCOME_ORDER:
        for cohort, frame in _by_cohort(fits):
            grid, curves = _fitted_time_courses(
                frame, STOP_TIME_RUNG, outcome, "tc", SPLINE_KNOTS, TMAX
            )
            if not len(curves):
                continue
            median_curve = np.median(curves, axis=0)
            peak = float(grid[int(np.argmax(median_curve))])
            rows.append(
                {
                    "outcome": outcome,
                    "stop_cohort": cohort,
                    "n_sessions": int(len(curves)),
                    "support_s": f"{grid[0]:.1f}–{grid[-1]:.1f}",
                    "burst_peak_s": peak,
                    "peak_frac_support": (peak - grid[0]) / (grid[-1] - grid[0]),
                }
            )
    return pd.DataFrame(rows)


def cohort_arrival_retention_table(fits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """``retained_share_arrival_events`` per population and cohort.

    The arrival-clock window keeps a different share of each population's event
    bins, and moving the stop cohort moves that share again -- an arrival-ladder
    contrast between two cohorts is partly a contrast between two windows.
    """
    rows = []
    per_cohort = {
        cohort: arrival_cohort_table(frame) for cohort, frame in _by_cohort(fits)
    }
    for outcome in OUTCOME_ORDER:
        for cohort, table in per_cohort.items():
            if table.empty:
                continue
            sub = table[table.outcome == outcome]
            if sub.empty:
                continue
            row = sub.iloc[0]
            rows.append(
                {
                    "outcome": outcome,
                    "stop_cohort": cohort,
                    "n_event_bins_frozen": float(row["n_event_bins_frozen"]),
                    "n_event_bins_arrival_window": float(
                        row["n_event_bins_arrival_window"]
                    ),
                    "retained_share_arrival_events": float(
                        row["retained_share_arrival_events"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def cohort_contrast_rung_table(contrast: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per rung and population, what each cohort's is_swr-contrast bundle holds.

    Restricted to the bundle's PRIMARY fitting cohort, so the four rungs that
    need a post-outcome covariate (:data:`CONTRAST_POST_OUTCOME_RUNGS`) are
    absent: they are fit on the gated cohort with their own null and are not a
    second copy of the ladder.

    ``rr_immediate`` is the estimand the rung actually defines, which is not the
    same statement on every rung:

    * on a ``G`` rung it is that population's immediate reward slope;
    * on an ``S`` rung the ladder carries ``is_swr`` as a gating modifier, so the
      main effect is the **non-SWR reference-level** slope and
      ``rr_swr_x_immediate`` is the multiplier that turns it into the SWR one;
    * on a reward-x-time rung (:data:`REWARD_X_TIME_RUNGS`) it is neither -- the
      main effect there is the slope where every spline column is zero, off the
      fitted window -- so it is left blank rather than printed.
    """
    rows = []
    frames = list(_by_cohort(contrast))
    models = sorted(
        {m for _, frame in frames for m in frame["model"].unique()},
        key=rate_rung_sort_key,
    )
    populations = sorted(
        {p for _, frame in frames for p in frame["population"].unique()}
    )
    for model in models:
        boundary = model in REWARD_X_TIME_RUNGS
        for population in populations:
            for cohort, frame in frames:
                sub = frame[
                    (frame.model == model)
                    & (frame.population == population)
                    & (frame.cohort == CONTRAST_PRIMARY_COHORT)
                ]
                if sub.empty:
                    continue
                median, n_positive, n_animals = _animal_balanced(sub, IMMEDIATE_COEF)
                swr_median, _, swr_animals = _animal_balanced(sub, SWR_IMMEDIATE_COEF)
                rows.append(
                    {
                        "model": model,
                        "population": population,
                        "stop_cohort": cohort,
                        "n_sessions": len(sub),
                        "n_stops": float(sub["n_stops"].sum()),
                        "n_bins": float(sub["n_bins"].sum()),
                        "rr_immediate": (np.nan if boundary else float(np.exp(median))),
                        "animals_positive": (
                            "—"
                            if boundary or not n_animals
                            else f"{n_positive}/{n_animals}"
                        ),
                        "rr_swr_x_immediate": (
                            np.nan
                            if boundary or not swr_animals
                            else float(np.exp(swr_median))
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _cohort_label_note(fits: dict[str, pd.DataFrame]) -> str:
    """What each artifact says its own cohort is, so a mislabelled input shows."""
    parts = []
    for cohort, frame in _by_cohort(fits):
        if "stop_cohort" not in frame.columns:
            parts.append(f"`{cohort}`: no `stop_cohort` column (pre-flag fit)")
            continue
        recorded = sorted(map(str, frame["stop_cohort"].dropna().unique()))
        parts.append(f"`{cohort}`: rows say {recorded}")
    return "; ".join(parts)


def _cohort_headline(fits: dict[str, pd.DataFrame]) -> str:
    """The two numbers the cohort choice actually moves, read off the tables.

    Written from the tables below rather than typed, so the prose cannot drift
    from the artifacts. It names the non-SWR non-local reward ratio and the
    paired SWR-vs-non-SWR test, which are the quantities the population choice
    was suspected of deciding.
    """
    rr = cohort_reward_rr_table(fits)
    paired = cohort_paired_swr_table(fits)
    parts = []
    non_swr = rr[rr.outcome == "non_local_no_swr"]
    if len(non_swr):
        stated = "; ".join(
            f"`{row.stop_cohort}` {row.rr_immediate:.2f}x "
            f"({row.animals_pos_immediate} animals, {int(row.n_sessions)} sessions)"
            for row in non_swr.itertuples()
        )
        parts.append(
            "Immediate reward component (`rr_immediate`, the 0 -> 1 contrast on "
            "bump 0) on the non-SWR non-local population "
            f"({FULL_MODEL}): {stated}."
        )
        stated = "; ".join(
            f"`{row.stop_cohort}` {row.rr_lag1_session:.3f}x session-pooled / "
            f"{row.rr_lag1_animal:.3f}x animal-balanced "
            f"({row.animals_pos_lag1} animals)"
            for row in non_swr.itertuples()
        )
        parts.append(
            "The published functional on the same population and rung "
            "(`rr_lag1`, the reconstructed kernel at lag 1 — a different "
            f"quantity, not a restatement): {stated}."
        )
    if len(paired):
        stated = "; ".join(
            f"`{row.stop_cohort}` {row.pct_sessions_positive:.1f}% of "
            f"{int(row.n_paired_sessions)} sessions, {row.animals_positive} "
            f"animals, p={row.signrank_p}"
            for row in paired.itertuples()
        )
        parts.append(
            "SWR-coupled minus non-SWR immediate coefficient, paired within "
            f"session: {stated}."
        )
    sizes = cohort_size_table(fits)
    sizes = sizes[sizes.outcome == "non_local"]
    if len(sizes):
        stated = "; ".join(
            f"`{row.stop_cohort}` {int(row.n_stops):,} stops over "
            f"{int(row.n_sessions)} sessions"
            for row in sizes.itertuples()
        )
        parts.append(f"Modelled non-local cohort: {stated}.")
    return "\n\n".join(parts) if parts else "_(nothing fitted)_"


def _short_git_hash(recorded: str) -> str:
    """Abbreviate a recorded git hash without dropping its ``-dirty`` marker.

    Sidecars record the full 40-character sha plus ``-dirty`` when the tree
    had uncommitted changes at write time; that suffix is the whole point of
    printing the hash in a provenance table, so only the hex part is cut.
    """
    hexpart, sep, suffix = recorded.partition("-")
    return hexpart[:12] + (sep + suffix if sep else "")


def cohort_provenance_table(paths: dict[str, Path]) -> pd.DataFrame:
    """What each artifact's sidecar records about the code that produced it.

    Read rather than asserted, because the two facts a reader needs here are
    both easy to get wrong from memory: whether the tree was clean when the fit
    ran, and which artifacts were re-derived after a later commit.
    """
    rows = []
    for role, path in paths.items():
        sidecar = Path(str(path).replace(".pkl", ".provenance.json"))
        if not sidecar.exists():
            continue
        record = json.loads(sidecar.read_text())
        rows.append(
            {
                "artifact": path.name,
                "role": role,
                "git_hash": _short_git_hash(str(record.get("git_hash", ""))),
                "stop_cohort": str(record.get("stop_cohort", "—")),
                "written_at": str(record.get("written_at", ""))[:19],
            }
        )
    return pd.DataFrame(rows)


def _audited_sibling(path: Path) -> Path:
    """The audited aggregate published beside a per-session contrast bundle."""
    return path.with_name(
        path.name.replace("swr_contrast_rate", "swr_contrast_rate_audited", 1)
    )


def in_transit_power_note(fits: dict[str, pd.DataFrame]) -> str:
    """How many sessions the in-transit bundle actually reaches, per population.

    Derived rather than asserted, because it is the caveat the in-transit column
    cannot be read without: an in-transit stop is short and sparse, so most
    sessions never clear the per-bin minimum-sample gates and the population
    that is rarest there -- SWR-coupled non-local events -- clears them least
    often. A cell that shows no effect in that column is an absence of evidence.
    """
    frame = fits.get("in_transit")
    if frame is None or not len(frame):
        return ""
    reference = fits.get("at_well")
    corpus = (
        (reference if reference is not None and len(reference) else frame)
        .groupby(SC)
        .ngroups
    )
    fitted = frame.groupby(SC).ngroups
    nulls = frame[frame.model == "M0_const"]
    per_population = ", ".join(
        f"{outcome} {int((nulls.outcome == outcome).sum())}"
        for outcome in OUTCOME_ORDER
        if (nulls.outcome == outcome).any()
    )
    return (
        f"**The in-transit column is thin.** Only **{fitted} of {corpus}** sessions "
        "clear the per-bin minimum-sample gates on any population "
        f"({per_population}), against {corpus} on the canonical cohort. An "
        "in-transit cell that shows no effect is an absence of evidence, not "
        "evidence of absence, and its paired tests have no power."
    )


def _flag_support_edge(table: pd.DataFrame) -> pd.DataFrame:
    """Star a burst peak that sits on the edge of the spline's supported interior.

    An argmax at ``peak_frac_support`` 0 or 1 is the curve still rising (or
    falling) where the summary stops looking, so the peak is a bound on where
    the burst is, not a location. Starring is done at render time so the table
    itself keeps a numeric column.
    """
    if table.empty or "burst_peak_s" not in table.columns:
        return table
    shown = table.copy()
    edge = shown["peak_frac_support"].isin((0.0, 1.0))
    shown["burst_peak_s"] = [
        f"{value:.3f}*" if flag else f"{value:.3f}"
        for value, flag in zip(shown["burst_peak_s"], edge, strict=True)
    ]
    return shown


def build_cohort_report(
    fits: dict[str, pd.DataFrame],
    contrast: dict[str, pd.DataFrame],
    inputs: dict[str, Path],
    command: str | None = None,
    provenance_extra: dict[str, Path] | None = None,
) -> str:
    """Assemble the at-well / all-stops / in-transit comparison.

    ``provenance_extra`` names artifacts the audit does not read but whose
    sidecars belong in its provenance table -- the audited is_swr-contrast
    aggregates, which are what a reader quotes from.
    """
    provenance_extra = provenance_extra or {}
    lines = [
        "# Rate ladders across the three stop cohorts",
        "",
        f"Generated {dt.date.today().isoformat()} by `scripts/qa/compare_fits.py "
        "--cohort-contrast`.",
        "",
    ]
    if command:
        lines += [f"Command: `{command}`", ""]
    lines += [
        "The same ladder, the same estimator, three stop populations. "
        "`at_well` is the canonical cohort the content family is also fit on; "
        "the other two are sensitivity bundles and are never pooled with it. "
        "No likelihood-scale quantity is differenced across cohorts — the "
        "ΔAIC steps below are each taken WITHIN one cohort, between two rungs "
        "that share rows.",
        "",
        "## What moved",
        "",
        _cohort_headline(fits),
        "",
        "## Cohorts",
        "",
        _md(
            pd.DataFrame(
                [
                    {
                        "stop_cohort": c,
                        "fit_supplied": "yes" if c in fits and len(fits[c]) else "no",
                        "rule": COHORT_RULE_NOTE[c],
                    }
                    for c in COHORT_ORDER
                ]
            )
        ),
        "",
        f"What each artifact records about itself: {_cohort_label_note(fits)}.",
        "",
        in_transit_power_note(fits),
        "",
        "## Inputs",
        "",
        _md(
            pd.DataFrame(
                [
                    {"role": role, "path": str(path), "sha256": _sha256(path)[:12]}
                    for role, path in inputs.items()
                ]
            )
        ),
        "",
        "### Provenance of the artifacts read",
        "",
        _md(cohort_provenance_table({**inputs, **provenance_extra})),
        "",
        "The hash is taken when the artifact is WRITTEN, not when its run started, "
        "so a long fit that spanned a commit records the later one — which is why "
        "the rows above do not all agree even though they describe one sitting. A "
        "`-dirty` suffix means the tree carried uncommitted edits at that moment; "
        "here those edits were to this audit script and to nothing the fit runners "
        "import. The audited is_swr-contrast bundles are the one place where the "
        "recorded hash PREDATES the code that wrote them: they were re-derived with "
        "`--aggregate-only` after the commit that gave the audited table its "
        "`stop_cohort` label and fixed the input name its freshness check resolves, "
        "and the hash records the tree that re-derivation ran from. The per-session "
        "fits beneath them, which carry every number this audit reads, are "
        "unaffected.",
        "",
        "## Cohort size (per-bin ladder)",
        "",
        _md(cohort_size_table(fits)),
        "",
        "The `in_transit` rows above are a whole-corpus total over the few "
        "sessions that reach the ladder, not over the whole corpus. "
        + in_transit_power_note(fits),
        "",
        "## Reward rate ratios, `M7r_time_histcos_rew`",
        "",
        "Two different quantities, never to be quoted as one. "
        f"**`rr_immediate`** is `exp` of the animal-balanced median `{IMMEDIATE_COEF}` "
        "— the full 0 → 1 contrast on the first raised-cosine bump, i.e. one "
        "coefficient. **`rr_lag1_*`** is `exp` of the reconstructed reward kernel at "
        "lag 1, the blend `[0.325, 0.066, 0]` over the three bumps, which is the "
        "published functional `docs/rate_models.md` and claims row 19c quote; it is "
        "given session-pooled (`rr_lag1_session`, how the rate docs report it) and "
        "animal-balanced (`rr_lag1_animal`, how the audited aggregates report it). "
        "`animals_pos_*` counts animals whose own median is positive for each.",
        "",
        _md(cohort_reward_rr_table(fits), floatfmt="{:.3f}"),
        "",
        "## SWR-coupled vs non-SWR, paired within session",
        "",
        _md(cohort_paired_swr_table(fits), floatfmt="{:.3f}"),
        "",
        "## Nested reward steps, within cohort (negative = child better)",
        "",
        _md(cohort_delta_aic_table(fits)),
        "",
        "## Burst peak of the smooth time course (`M5_time`)",
        "",
        _md(_flag_support_edge(cohort_burst_peak_table(fits)), floatfmt="{:.3f}"),
        "",
        "`*` marks a peak sitting on the edge of the spline's supported interior "
        "(`peak_frac_support` 0 or 1): the median curve is still rising or falling "
        "where the summary stops reading it, so the value bounds the burst rather "
        "than locating it.",
        "",
        "## Arrival-window retention",
        "",
        _md(cohort_arrival_retention_table(fits), floatfmt="{:.3f}"),
        "",
    ]
    if contrast:
        lines += [
            "## is_swr-contrast rate bundle, per rung and population",
            "",
            "Primary fitting cohort only, so `"
            + "`, `".join(CONTRAST_POST_OUTCOME_RUNGS)
            + "` are absent: they need a post-outcome covariate and are fit on the "
            "gated cohort with their own null, which exists for one AIC comparison "
            "and is not a second copy of the ladder.",
            "",
            "`rr_immediate` is not the same statement on every rung. On a `G` rung "
            "it is that population's immediate reward slope, fit on that "
            "population's own rows. On an `S` rung that splits the reward block by "
            "`is_swr` it is the **non-SWR reference-level** slope, and "
            f"`rr_swr_x_immediate` (`exp` of the animal-balanced "
            f"`{SWR_IMMEDIATE_COEF}`) is the multiplier that turns it into the SWR "
            "one — so those two cells are a contrast, not two estimates of one "
            "thing. `rr_swr_x_immediate` is blank wherever the rung carries no such "
            "interaction: on every `G` rung, and on the ladder null `S0_rate`, "
            "where `is_swr` enters only as an intercept shift and the reward slope "
            "is shared by both populations. On `"
            + "`, `".join(REWARD_X_TIME_RUNGS)
            + "` it is blank: the reward block is interacted with the time spline "
            "there, so the main effect is the slope at the point where every spline "
            "column is zero — off the fitted window, a boundary extrapolation rather "
            "than a reward contrast.",
            "",
            _md(cohort_contrast_rung_table(contrast), floatfmt="{:.3f}"),
            "",
        ]
    return "\n".join(lines)


def collect_cohort_inputs(
    per_bin: dict[str, Path], contrast: dict[str, Path]
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Path]]:
    """Read whichever cohort bundles exist, and record what was read.

    A missing sensitivity bundle is skipped rather than fatal -- the mode is
    still meaningful with two cohorts -- but the canonical at-well per-bin fit
    is required, since it is what the other columns are read against.
    """
    fits: dict[str, pd.DataFrame] = {}
    contrast_fits: dict[str, pd.DataFrame] = {}
    inputs: dict[str, Path] = {}
    for cohort in COHORT_ORDER:
        path = per_bin.get(cohort)
        if path is not None and path.exists():
            fits[cohort] = pd.read_pickle(path)
            inputs[f"per-bin ({cohort})"] = path
        path = contrast.get(cohort)
        if path is not None and path.exists():
            contrast_fits[cohort] = pd.read_pickle(path)
            inputs[f"is_swr-contrast rate ({cohort})"] = path
    if "at_well" not in fits:
        raise SystemExit(
            f"the canonical at-well per-bin fit is required: {per_bin.get('at_well')}"
        )
    return fits, contrast_fits, inputs


def run_cohort_contrast_mode(args: argparse.Namespace) -> None:
    """Build and write the cohort comparison."""
    fits, contrast, inputs = collect_cohort_inputs(
        {
            "at_well": args.at_well_per_bin,
            "all_stops": args.all_stops_per_bin,
            "in_transit": args.in_transit_per_bin,
        },
        {
            "at_well": args.at_well_contrast,
            "all_stops": args.all_stops_contrast,
        },
    )
    audited = {
        f"is_swr-contrast audited ({cohort})": _audited_sibling(path)
        for cohort, path in (
            ("at_well", args.at_well_contrast),
            ("all_stops", args.all_stops_contrast),
        )
        if _audited_sibling(path).exists()
    }
    report = build_cohort_report(fits, contrast, inputs, _invocation(), audited)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"wrote {args.out}")


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
    parser.add_argument(
        "--per-bin-cv-audit",
        type=Path,
        default=None,
        help=(
            "Per-bin fit whose per-rung, per-population cross-validation "
            "validity is tabulated. Point it at the fit that still carried a "
            "rung the CV measurement then retired; defaults to --new-per-bin."
        ),
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
        "--content",
        action="store_true",
        help=(
            "Compare the CONTENT ladders (count response against the per-event "
            "fraction response) instead of the rate families."
        ),
    )
    parser.add_argument(
        "--old-content-root",
        type=Path,
        default=DEFAULT_CONTENT_BACKUP_ROOT,
        help="Root holding the pre-fix content bundles (--content).",
    )
    parser.add_argument(
        "--new-content-root",
        type=Path,
        default=RC,
        help="Root holding the current content bundles (--content).",
    )
    parser.add_argument(
        "--content-bundles",
        type=lambda v: tuple(part.strip() for part in v.split(",") if part.strip()),
        default=CONTENT_BUNDLES,
        help="Comma-separated content bundle directory names to compare.",
    )
    parser.add_argument(
        "--content-framing",
        choices=sorted(CONTENT_FRAMINGS),
        default="response",
        help=(
            "What the two content sides differ by: `response` (count vs "
            "per-event fraction) or `cohort` (the same response refit on a "
            "different event population)."
        ),
    )
    parser.add_argument(
        "--cohort-contrast",
        action="store_true",
        help=(
            "Compare the rate ladders fit on the three stop cohorts (at-well, "
            "all-stops, in-transit) instead of comparing a refit to its "
            "predecessor."
        ),
    )
    parser.add_argument(
        "--at-well-per-bin",
        type=Path,
        default=DEFAULT_COHORT_PER_BIN["at_well"],
        help="Canonical at-well per-bin fit (--cohort-contrast).",
    )
    parser.add_argument(
        "--all-stops-per-bin",
        type=Path,
        default=DEFAULT_COHORT_PER_BIN["all_stops"],
        help="All-stops per-bin sensitivity fit (--cohort-contrast).",
    )
    parser.add_argument(
        "--in-transit-per-bin",
        type=Path,
        default=DEFAULT_COHORT_PER_BIN["in_transit"],
        help="In-transit per-bin sensitivity fit (--cohort-contrast).",
    )
    parser.add_argument(
        "--at-well-contrast",
        type=Path,
        default=DEFAULT_COHORT_CONTRAST["at_well"],
        help="Canonical at-well is_swr-contrast rate bundle (--cohort-contrast).",
    )
    parser.add_argument(
        "--all-stops-contrast",
        type=Path,
        default=DEFAULT_COHORT_CONTRAST["all_stops"],
        help="All-stops is_swr-contrast rate bundle (--cohort-contrast).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Report path. Defaults to docs/audits/<today>-rate-refit-comparison.md, "
            "<today>-content-refit-comparison.md under --content, or "
            "<today>-rate-cohort-atwell.md under --cohort-contrast."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.content and args.cohort_contrast:
        raise SystemExit("--content and --cohort-contrast are separate modes")
    if args.out is None:
        today = dt.date.today().isoformat()
        if args.cohort_contrast:
            args.out = REPO / "docs" / "audits" / f"{today}-rate-cohort-atwell.md"
        else:
            kind = "content" if args.content else "rate"
            args.out = REPO / "docs" / "audits" / f"{today}-{kind}-refit-comparison.md"
    if args.cohort_contrast:
        run_cohort_contrast_mode(args)
        return
    if args.content:
        run_content_mode(args)
        return
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
    cv_audit_bin = None
    if args.per_bin_cv_audit is not None:
        if not args.per_bin_cv_audit.exists():
            raise SystemExit(f"missing fit artifact: {args.per_bin_cv_audit}")
        cv_audit_bin = pd.read_pickle(args.per_bin_cv_audit)
        inputs["per-bin (cross-validation audit)"] = args.per_bin_cv_audit
    report = build_report(
        pd.read_pickle(args.old_per_bin),
        pd.read_pickle(args.new_per_bin),
        pd.read_pickle(args.old_per_stop),
        pd.read_pickle(args.new_per_stop),
        inputs,
        args.per_stop_note,
        termlevel,
        cv_audit_bin,
        _invocation(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
