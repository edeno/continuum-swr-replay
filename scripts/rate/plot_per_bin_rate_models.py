#!/usr/bin/env python
"""Plot the per-bin conditional-intensity ladder (Family B) from its tidy output.

Reads the persisted per-session coefficients in
``models_per_bin_rate/all_models.pkl`` (produced by
``fit_per_bin_rate_models.py``) and renders the two headline figures -- no
re-fitting happens here. Within-animal / across-session summaries (medians,
IQR, paired Wilcoxon) are computed at read time.

Both figures read the full model ``M7_time_hist_rew`` (time spline +
self-history + reward kernel), so every effect is shown controlled for the
within-stop time course and the same-event clustering:

* ``reward_kernel_per_bin_rate.png`` -- the reward rate-ratio kernel
  reconstructed from the three raised-cosine basis coefficients per
  population, and the immediate-component dissociation between SWR-coupled and
  non-SWR non-local events.
* ``selfhistory_autocorr.png`` -- the within-stop self-history rate ratio
  (onset_lag1..10): refractory (<1) at short lags, bursty (>1) at ~100 ms.

Usage:
    uv run python scripts/rate/plot_per_bin_rate_models.py
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from continuum_swr_replay.analysis.replay._helpers import (  # noqa: E402
    raised_cosine_history_basis,
)
from continuum_swr_replay.analysis.replay.model_specs import (  # noqa: E402
    BIN_S,
    N_HISTORY_LAGS,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_BASIS,
    REWARD_HISTORY_N_LAGS,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR as P  # noqa: E402
from continuum_swr_replay.session_keys import SESSION_COLUMNS

RC = P / "replay_content"
FIGS = P.parent / "paper" / "figures" / "rate"
FIGS.mkdir(parents=True, exist_ok=True)
SC = list(SESSION_COLUMNS)
MODEL = "M7_time_hist_rew"

# outcome key -> (display label, colour)
POP_STYLE = {
    "swr": ("detected SWR", "k"),
    "non_local": ("non-local (all)", "C2"),
    "non_local_swr": ("non-local w/ SWR", "C3"),
    "non_local_no_swr": ("non-local NO SWR", "C0"),
}


def _kernel(
    df: pd.DataFrame, prefix: str, n: int
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-lag median + IQR rate ratios (exp coef) across sessions, by outcome."""
    out = {}
    for oc in POP_STYLE:
        d = df[df.outcome == oc]
        med = np.array([np.exp(d[f"coef_{prefix}{j}"].median()) for j in range(1, n + 1)])
        lo = np.array([np.exp(d[f"coef_{prefix}{j}"].quantile(0.25)) for j in range(1, n + 1)])
        hi = np.array([np.exp(d[f"coef_{prefix}{j}"].quantile(0.75)) for j in range(1, n + 1)])
        out[oc] = (med, lo, hi)
    return out


IMMEDIATE_COEF = f"coef_{REWARD_HISTORY_COLUMNS[0]}"


def _reward_kernel_weights() -> np.ndarray:
    """Per-lag weight each basis coefficient contributes, shape (n_basis, n_lags).

    The fitted features are bump-weighted reward *fractions* (each bump is
    normalized by the sum of its weights over the known lags), so the log-rate
    contribution of an outcome at lag ``j`` is
    ``sum_k coef_k * B[k, j] / sum_j' B[k, j']``.
    """
    B = raised_cosine_history_basis(REWARD_HISTORY_N_LAGS, REWARD_HISTORY_N_BASIS)
    return B / B.sum(axis=1, keepdims=True)


def _reconstructed_kernels(m7: pd.DataFrame, outcome: str) -> np.ndarray:
    """Per-session reconstructed log-rate kernel over lags 1..10, shape (n_sess, n_lags)."""
    coefs = m7[m7.outcome == outcome][
        [f"coef_{c}" for c in REWARD_HISTORY_COLUMNS]
    ].to_numpy(dtype=float)
    coefs = coefs[~np.isnan(coefs).any(axis=1)]
    return coefs @ _reward_kernel_weights()


def plot_reward_kernel(m7: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    lags = np.arange(1, REWARD_HISTORY_N_LAGS + 1)
    for i, oc in enumerate(["swr", "non_local_swr", "non_local_no_swr"]):
        label, col = POP_STYLE[oc]
        kernels = np.exp(_reconstructed_kernels(m7, oc))
        med = np.median(kernels, axis=0)
        lo, hi = np.percentile(kernels, [25, 75], axis=0)
        ax.errorbar(lags + 0.05 * (i - 1), med, yerr=[med - lo, hi - med],
                    marker="o", color=col, label=label, capsize=3, lw=2)
    ax.axhline(1, color="0.5", ls=":")
    ax.set_xlabel("reward lag (trials back)")
    ax.set_ylabel("rate ratio (per-session median, IQR)")
    ax.set_title("Reward kernel (per-bin, time + self-history controlled)\n"
                 "reconstructed from the raised-cosine basis coefficients")
    ax.set_xticks(lags)
    # A few near-separated sessions give extreme whiskers; clip the display so
    # the kernel shape is readable. The robust per-session medians are unaffected.
    ax.set_ylim(0, 6)
    ax.legend(frameon=False)

    ax = axes[1]
    order = ["swr", "non_local_swr", "non_local_no_swr"]
    data = [np.exp(m7[m7.outcome == oc][IMMEDIATE_COEF].dropna()) for oc in order]
    bp = ax.boxplot(data, labels=["SWR", "NL\nw/SWR", "NL\nNO SWR"],
                    showfliers=False, patch_artist=True)
    for patch, oc in zip(bp["boxes"], order, strict=True):
        patch.set_facecolor(POP_STYLE[oc][1])
        patch.set_alpha(0.4)
    ax.axhline(1, color="0.5", ls=":")
    ax.set_ylabel("immediate reward effect — RR per unit (per session)")
    ax.set_ylim(0, 5)

    # paired dissociation annotation: non_local_swr vs non_local_no_swr
    a = m7[m7.outcome == "non_local_swr"][SC + [IMMEDIATE_COEF]]
    b = m7[m7.outcome == "non_local_no_swr"][SC + [IMMEDIATE_COEF]]
    mrg = a.merge(b, on=SC, suffixes=("_s", "_n")).dropna()
    if len(mrg) < 2:
        raise SystemExit(
            f"Only {len(mrg)} paired sessions for the SWR/no-SWR dissociation; "
            "run fit_per_bin_rate_models.py over more sessions first."
        )
    diff = mrg[f"{IMMEDIATE_COEF}_s"] - mrg[f"{IMMEDIATE_COEF}_n"]
    pa = mrg.assign(d=diff).groupby("animal")["d"].median()
    pval = wilcoxon(mrg[f"{IMMEDIATE_COEF}_s"], mrg[f"{IMMEDIATE_COEF}_n"]).pvalue
    ax.set_title(f"Immediate reward effect by population\n"
                 f"NL-SWR > NL-no-SWR: {(diff > 0).mean():.0%} of {len(mrg)}, "
                 f"{(pa > 0).sum()}/{mrg.animal.nunique()} animals, p={pval:.1e}")

    fig.tight_layout()
    out = FIGS / "reward_kernel_per_bin_rate.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
    for oc in order:
        v = m7[m7.outcome == oc][IMMEDIATE_COEF].dropna()
        print(f"  {POP_STYLE[oc][0]:18s}: immediate RR={np.exp(v.median()):.2f} (n={len(v)})")


def plot_selfhistory(m7: pd.DataFrame) -> None:
    ker = _kernel(m7, "onset_lag", N_HISTORY_LAGS)
    fig, ax = plt.subplots(figsize=(8, 5))
    lagms = np.arange(1, N_HISTORY_LAGS + 1) * BIN_S * 1000
    for oc in ["swr", "non_local", "non_local_swr", "non_local_no_swr"]:
        label, col = POP_STYLE[oc]
        med, lo, hi = ker[oc]
        ax.errorbar(lagms, med, yerr=[med - lo, hi - med], marker="o",
                    color=col, label=label, capsize=3)
    ax.axhline(1, color="0.5", ls=":")
    ax.set_xlabel("time lag (ms)")
    ax.set_ylabel("self-history rate ratio (per-session median, IQR)")
    ax.set_title("Within-stop event autocorrelation\n(<1 = refractory, >1 = bursty)")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = FIGS / "selfhistory_autocorr.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    df = pd.read_pickle(RC / "models_per_bin_rate" / "all_models.pkl")
    m7 = df[df.model == MODEL]
    if m7.empty:
        raise SystemExit(
            f"No {MODEL} rows in all_models.pkl; run fit_per_bin_rate_models.py first."
        )
    FIGS.mkdir(parents=True, exist_ok=True)
    plot_reward_kernel(m7)
    plot_selfhistory(m7)


if __name__ == "__main__":
    main()
