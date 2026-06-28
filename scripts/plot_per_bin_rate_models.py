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
  (reward_lag1..10) per population and the lag-1 dissociation between
  SWR-coupled and non-SWR non-local events.
* ``selfhistory_autocorr.png`` -- the within-stop self-history rate ratio
  (onset_lag1..10): refractory (<1) at short lags, bursty (>1) at ~100 ms.

Usage:
    uv run python scripts/plot_per_bin_rate_models.py
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from continuum_swr_replay.analysis.replay.model_specs import (  # noqa: E402
    BIN_S,
    N_HISTORY_LAGS,
    N_REWARD_LAGS,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR as P  # noqa: E402

RC = P / "replay_content"
FIGS = P.parent / "paper" / "figures"
SC = ["animal", "day", "epoch"]
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


def plot_reward_kernel(m7: pd.DataFrame) -> None:
    ker = _kernel(m7, "reward_lag", N_REWARD_LAGS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    lags = np.arange(1, N_REWARD_LAGS + 1)
    for i, oc in enumerate(["swr", "non_local_swr", "non_local_no_swr"]):
        label, col = POP_STYLE[oc]
        med, lo, hi = ker[oc]
        ax.errorbar(lags + 0.05 * (i - 1), med, yerr=[med - lo, hi - med],
                    marker="o", color=col, label=label, capsize=3, lw=2)
    ax.axhline(1, color="0.5", ls=":")
    ax.set_xlabel("reward lag (trials back)")
    ax.set_ylabel("rate ratio (per-session median, IQR)")
    ax.set_title("Reward kernel (per-bin, time + self-history controlled)\n"
                 "lag1 dissociates; lags 2+ shared (RPE baseline)")
    ax.set_xticks(lags)
    # A few near-separated sessions give extreme lag-1 IQR whiskers; clip the
    # display so the kernel shape (lag-1 boost, lags 2+ dip) is readable. The
    # robust per-session medians are unaffected.
    ax.set_ylim(0, 6)
    ax.legend(frameon=False)

    ax = axes[1]
    order = ["swr", "non_local_swr", "non_local_no_swr"]
    data = [np.exp(m7[m7.outcome == oc]["coef_reward_lag1"].dropna()) for oc in order]
    bp = ax.boxplot(data, labels=["SWR", "NL\nw/SWR", "NL\nNO SWR"],
                    showfliers=False, patch_artist=True)
    for patch, oc in zip(bp["boxes"], order, strict=True):
        patch.set_facecolor(POP_STYLE[oc][1])
        patch.set_alpha(0.4)
    ax.axhline(1, color="0.5", ls=":")
    ax.set_ylabel("immediate reward effect — lag1 RR (per session)")
    ax.set_ylim(0, 5)

    # paired dissociation annotation: non_local_swr vs non_local_no_swr
    a = m7[m7.outcome == "non_local_swr"][SC + ["coef_reward_lag1"]]
    b = m7[m7.outcome == "non_local_no_swr"][SC + ["coef_reward_lag1"]]
    mrg = a.merge(b, on=SC, suffixes=("_s", "_n")).dropna()
    if len(mrg) < 2:
        raise SystemExit(
            f"Only {len(mrg)} paired sessions for the SWR/no-SWR dissociation; "
            "run fit_per_bin_rate_models.py over more sessions first."
        )
    diff = mrg.coef_reward_lag1_s - mrg.coef_reward_lag1_n
    pa = mrg.assign(d=diff).groupby("animal")["d"].median()
    pval = wilcoxon(mrg.coef_reward_lag1_s, mrg.coef_reward_lag1_n).pvalue
    ax.set_title(f"Immediate reward effect by population\n"
                 f"NL-SWR > NL-no-SWR: {(diff > 0).mean():.0%} of {len(mrg)}, "
                 f"{(pa > 0).sum()}/{mrg.animal.nunique()} animals, p={pval:.1e}")

    fig.tight_layout()
    out = FIGS / "reward_kernel_per_bin_rate.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
    for oc in order:
        v = m7[m7.outcome == oc]["coef_reward_lag1"].dropna()
        print(f"  {POP_STYLE[oc][0]:18s}: lag1 RR={np.exp(v.median()):.2f} (n={len(v)})")


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
