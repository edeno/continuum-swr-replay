#!/usr/bin/env python
"""Plot replay rate vs uncertainty/value by population from the tidy fit output.

Reads the per-session ``MU_uncertainty`` coefficients in
``models_per_bin_rate/uncertainty_rate.pkl`` (produced by
``fit_uncertainty_rate.py``) -- no re-fitting here -- and renders one panel per
uncertainty/value covariate showing the three populations' per-session
coefficient distributions on the rate-ratio scale, with the pre-registered
paired dissociation tests annotated:

* ``non_local_no_swr`` vs ``swr`` and vs ``non_local_swr`` (paired Wilcoxon over
  sessions present in both).

Results are *reported*, not asserted to a magnitude -- a null is a valid result.
Also plots the peri-block-change latency anchor: per-population mean coefficient
contribution of ``trials_since_block_change`` is a single linear term, so the
latency view here is the rate-ratio per unit trials-since-change per population.

Usage::

    uv run python scripts/rate/plot_uncertainty_rate.py            # full fit
    uv run python scripts/rate/plot_uncertainty_rate.py --smoke    # smoke fit
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from continuum_swr_replay.parameters import PROCESSED_DATA_DIR  # noqa: E402
from continuum_swr_replay.session_keys import SESSION_COLUMNS

# Single source of truth: the curated covariate set + populations are defined in
# the fitter; import them so the plot axes match exactly what was fit.
_FIT_SPEC = importlib.util.spec_from_file_location(
    "fit_uncertainty_rate", Path(__file__).resolve().with_name("fit_uncertainty_rate.py")
)
_fit = importlib.util.module_from_spec(_FIT_SPEC)
_FIT_SPEC.loader.exec_module(_fit)
POPULATIONS = _fit.POPULATIONS_TO_FIT

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SESSION_COLS = list(SESSION_COLUMNS)
COVARIATES = _fit.CURATED_COVARIATES


def _paired_wilcoxon(
    models: pd.DataFrame, covariate: str, pop_a: str, pop_b: str
) -> tuple[float, int]:
    """Paired Wilcoxon of ``coef_{covariate}`` between two populations.

    Pairs on the sessions present in both populations. Returns ``(p, n_pairs)``;
    ``(nan, n)`` if fewer than 6 pairs or all differences are zero.
    """
    col = f"coef_{covariate}"
    a = models[models["outcome"] == pop_a][[*SESSION_COLS, col]]
    b = models[models["outcome"] == pop_b][[*SESSION_COLS, col]]
    merged = a.merge(b, on=SESSION_COLS, suffixes=("_a", "_b")).dropna()
    if len(merged) < 6 or np.allclose(merged[f"{col}_a"], merged[f"{col}_b"]):
        return np.nan, len(merged)
    stat, p = wilcoxon(merged[f"{col}_a"], merged[f"{col}_b"])
    return float(p), len(merged)


def report(models: pd.DataFrame) -> None:
    """Log per-covariate per-population RR summary + dissociation Wilcoxon."""
    for cov in COVARIATES:
        rr = f"rr_{cov}"
        coef = f"coef_{cov}"
        if rr not in models.columns:
            continue
        logger.info("\n=== %s ===", cov)
        for pop in POPULATIONS:
            sub = models[models["outcome"] == pop][coef].dropna()
            if sub.empty:
                continue
            pos = float((sub > 0).mean())
            n_animal = models[(models["outcome"] == pop) & models[coef].notna()][
                "animal"
            ].nunique()
            logger.info(
                "  %-18s median RR=%.3f  %%pos=%.0f%%  n=%d  animals=%d/5",
                pop,
                float(np.exp(sub.median())),
                100 * pos,
                len(sub),
                n_animal,
            )
        for other in ("swr", "non_local_swr"):
            p, n = _paired_wilcoxon(models, cov, "non_local_no_swr", other)
            logger.info(
                "  paired non_local_no_swr vs %-14s p=%s (n=%d)",
                other,
                f"{p:.3g}" if np.isfinite(p) else "n/a",
                n,
            )


def plot(models: pd.DataFrame, out_path) -> None:
    """One panel per covariate; per-population per-session coefficient boxplots."""
    n = len(COVARIATES)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
    pop_colors = {
        "swr": "#4477AA",
        "non_local_swr": "#CCBB44",
        "non_local_no_swr": "#EE6677",
    }
    for ax, cov in zip(axes.ravel(), COVARIATES, strict=False):
        coef = f"coef_{cov}"
        data, labels, colors = [], [], []
        for pop in POPULATIONS:
            vals = np.exp(models[models["outcome"] == pop][coef].dropna().to_numpy())
            data.append(vals)
            labels.append(pop.replace("non_local", "NL"))
            colors.append(pop_colors[pop])
        bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors, strict=True):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_title(cov, fontsize=9)
        ax.set_ylabel("rate ratio")
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        # annotate the headline dissociation p (NL-no-SWR vs SWR)
        p, npair = _paired_wilcoxon(models, cov, "non_local_no_swr", "swr")
        ax.text(
            0.5,
            0.97,
            f"NLnoSWR vs SWR p={p:.2g} (n={npair})"
            if np.isfinite(p)
            else "p n/a",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7,
        )
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="Read the smoke fit output."
    )
    args = parser.parse_args()

    rc = PROCESSED_DATA_DIR / "replay_content" / "models_per_bin_rate"
    fname = "uncertainty_rate_smoke.pkl" if args.smoke else "uncertainty_rate.pkl"
    models = pd.read_pickle(rc / fname)
    logger.info("Loaded %d rows from %s", len(models), rc / fname)

    report(models)

    out_dir = Path(__file__).resolve().parents[2] / "paper" / "figures" / "rate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot(models, out_dir / "uncertainty_rate_by_population.png")


if __name__ == "__main__":
    main()
