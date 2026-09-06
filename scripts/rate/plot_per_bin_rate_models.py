#!/usr/bin/env python
"""Plot the per-bin conditional-intensity ladder (Family B) from its tidy output.

Reads the persisted per-session coefficients in
``models_per_bin_rate/all_models.pkl`` (produced by
``fit_per_bin_rate_models.py``) and renders the two headline figures -- no
re-fitting happens here. Within-animal / across-session summaries (medians,
IQR, paired Wilcoxon) are computed at read time.

Both figures read the full model ``M7r_time_histcos_rew`` (time spline +
raised-cosine self-history + reward kernel), so every effect is shown
controlled for the within-stop time course and the same-event clustering:

* ``reward_kernel_basis.png`` -- the reward rate-ratio kernel over lags 1-10
  reconstructed from the three raised-cosine basis coefficients per
  population (per-session median and IQR, with the per-animal medians drawn
  through them), and the immediate-component dissociation between
  SWR-coupled and non-SWR non-local events.
* ``selfhistory_autocorr.png`` -- the within-stop self-history rate ratio
  over lags 1..10, reconstructed from the four raised-cosine basis
  coefficients: refractory (<1) at short lags, bursty (>1) at ~100 ms.

Usage:
    uv run python scripts/rate/plot_per_bin_rate_models.py
    uv run python scripts/rate/plot_per_bin_rate_models.py --allow-stale
"""

from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from continuum_swr_replay.analysis.replay.fit_provenance import (  # noqa: E402
    require_fresh_inputs,
)
from continuum_swr_replay.analysis.replay.model_specs import (  # noqa: E402
    BIN_S,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_BASIS,
    REWARD_HISTORY_N_LAGS,
)
from continuum_swr_replay.analysis.replay.per_bin_design import (  # noqa: E402
    reconstruct_reward_kernels,
    reconstruct_self_history_kernels,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR as P  # noqa: E402
from continuum_swr_replay.session_keys import SESSION_COLUMNS

RC = P / "replay_content"
ALL_MODELS = RC / "models_per_bin_rate" / "all_models.pkl"
# The tables fit_per_bin_rate_models.py fit all_models.pkl on. Checked against
# the artifact's provenance sidecar before plotting, so a figure cannot be
# rendered from coefficients that predate the tables on disk.
ALL_MODELS_INPUTS = {
    "stop_table": RC / "stop_table.pkl",
    "event_onset_table": RC / "event_onset_table.pkl",
    "event_table": RC / "event_table.pkl",
}
FIGS = P.parent / "paper" / "figures" / "rate"
FIGS.mkdir(parents=True, exist_ok=True)
SC = list(SESSION_COLUMNS)
MODEL = "M7r_time_histcos_rew"

# outcome key -> (display label, colour)
POP_STYLE = {
    "swr": ("detected SWR", "k"),
    "non_local": ("non-local (all)", "C2"),
    "non_local_swr": ("non-local w/ SWR", "C3"),
    "non_local_no_swr": ("non-local NO SWR", "C0"),
}


def _self_history_kernel(
    df: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-lag median + IQR self-history rate ratios across sessions, by outcome.

    Each session's kernel is reconstructed from that session's own four
    raised-cosine coefficients and the quantiles are taken per lag. Summarizing
    the coefficients first and reconstructing once would describe a session that
    does not exist: the basis coefficients are collinear, so their marginal
    medians are not any fitted kernel.
    """
    out = {}
    columns = [f"coef_hist_rcos_{k}" for k in range(N_HISTORY_BASIS)]
    for oc in POP_STYLE:
        fitted = df.loc[df.outcome == oc, columns].to_numpy(dtype=float)
        kernels = np.exp(
            reconstruct_self_history_kernels(fitted, N_HISTORY_LAGS, N_HISTORY_BASIS)
        )
        med = np.median(kernels, axis=0)
        lo, hi = np.percentile(kernels, [25, 75], axis=0)
        out[oc] = (med, lo, hi)
    return out


IMMEDIATE_COEF = f"coef_{REWARD_HISTORY_COLUMNS[0]}"


def _reconstructed_kernels(
    m7: pd.DataFrame, outcome: str
) -> tuple[np.ndarray, np.ndarray]:
    """Per-session reconstructed log-rate kernel over lags 1..10 for one population.

    Parameters
    ----------
    m7 : pd.DataFrame
        Tidy per-session rows of the full model, one row per (outcome, session).
    outcome : str
        Event-type population key.

    Returns
    -------
    animals : np.ndarray, shape (n_sessions,)
        Animal of each retained session, for the per-animal medians.
    kernels : np.ndarray, shape (n_sessions, n_lags)
        Log-rate contribution of a reward at each lag 1..10.
    """
    columns = [f"coef_{c}" for c in REWARD_HISTORY_COLUMNS]
    fitted = m7.loc[m7.outcome == outcome, ["animal", *columns]].dropna()
    kernels = reconstruct_reward_kernels(
        fitted[columns].to_numpy(dtype=float),
        REWARD_HISTORY_N_LAGS,
        REWARD_HISTORY_N_BASIS,
    )
    return fitted["animal"].to_numpy(), kernels


def _animal_median_kernels(
    animals: np.ndarray, kernels: np.ndarray
) -> dict[str, np.ndarray]:
    """Median kernel per animal, so each animal weighs once in the display."""
    return {
        animal: np.median(kernels[animals == animal], axis=0)
        for animal in np.unique(animals)
    }


def plot_reward_kernel(m7: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    lags = np.arange(1, REWARD_HISTORY_N_LAGS + 1)
    shown = []  # every rate ratio drawn, for the y-limits
    for i, oc in enumerate(["swr", "non_local_swr", "non_local_no_swr"]):
        label, col = POP_STYLE[oc]
        animals, log_kernels = _reconstructed_kernels(m7, oc)
        kernels = np.exp(log_kernels)
        offset = 0.05 * (i - 1)
        # Per-animal medians first, so the across-session summary reads on top.
        for animal_kernel in _animal_median_kernels(animals, log_kernels).values():
            animal_ratio = np.exp(animal_kernel)
            shown.append(animal_ratio)
            ax.plot(
                lags + offset,
                animal_ratio,
                color=col,
                lw=0.8,
                alpha=0.35,
                zorder=1,
            )
        med = np.median(kernels, axis=0)
        lo, hi = np.percentile(kernels, [25, 75], axis=0)
        shown += [lo, hi]
        ax.errorbar(
            lags + offset,
            med,
            yerr=[med - lo, hi - med],
            marker="o",
            color=col,
            label=label,
            capsize=3,
            lw=2,
            zorder=3,
        )
    ax.axhline(1, color="0.5", ls=":")
    ax.set_xlabel("reward lag (trials back)")
    ax.set_ylabel("rate ratio (per-session median, IQR)")
    ax.set_title(
        "Reward kernel (time + self-history controlled)\n"
        "bold: per-session median (IQR); thin: per-animal medians"
    )
    ax.set_xticks(lags)
    # Fit the limits to what is drawn (medians, IQR, per-animal medians). A few
    # near-separated sessions have extreme individual kernels, but none of them
    # reaches the display: every drawn curve is a median or a quartile.
    drawn = np.concatenate(shown)
    span = drawn.max() - drawn.min()
    ax.set_ylim(drawn.min() - 0.1 * span, drawn.max() + 0.1 * span)
    ax.legend(frameon=False)

    ax = axes[1]
    order = ["swr", "non_local_swr", "non_local_no_swr"]
    data = [np.exp(m7[m7.outcome == oc][IMMEDIATE_COEF].dropna()) for oc in order]
    bp = ax.boxplot(
        data,
        labels=["SWR", "NL\nw/SWR", "NL\nNO SWR"],
        showfliers=False,
        patch_artist=True,
    )
    for patch, oc in zip(bp["boxes"], order, strict=True):
        patch.set_facecolor(POP_STYLE[oc][1])
        patch.set_alpha(0.4)
    ax.axhline(1, color="0.5", ls=":")
    ax.set_ylabel("immediate reward effect — RR per unit (per session)")
    # The immediate component is a rate ratio per unit of a 0-1 reward fraction,
    # so the per-session spread is multiplicative and covers orders of
    # magnitude; a linear axis truncates the upper boxes.
    ax.set_yscale("log")

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
    ax.set_title(
        f"Immediate reward effect by population\n"
        f"NL-SWR > NL-no-SWR: {(diff > 0).mean():.0%} of {len(mrg)}, "
        f"{(pa > 0).sum()}/{mrg.animal.nunique()} animals, p={pval:.1e}"
    )

    fig.tight_layout()
    out = FIGS / "reward_kernel_basis.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
    for oc in order:
        v = m7[m7.outcome == oc][IMMEDIATE_COEF].dropna()
        print(
            f"  {POP_STYLE[oc][0]:18s}: immediate RR={np.exp(v.median()):.2f} (n={len(v)})"
        )


def plot_selfhistory(m7: pd.DataFrame) -> None:
    ker = _self_history_kernel(m7)
    fig, ax = plt.subplots(figsize=(8, 5))
    lagms = np.arange(1, N_HISTORY_LAGS + 1) * BIN_S * 1000
    for oc in ["swr", "non_local", "non_local_swr", "non_local_no_swr"]:
        label, col = POP_STYLE[oc]
        med, lo, hi = ker[oc]
        ax.errorbar(
            lagms,
            med,
            yerr=[med - lo, hi - med],
            marker="o",
            color=col,
            label=label,
            capsize=3,
        )
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


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser (also the provenance gate's opt-out)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help=(
            "Plot the fit even if its input tables have changed since it was "
            "fit (or carry no provenance sidecar at all)."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_fresh_inputs(ALL_MODELS, ALL_MODELS_INPUTS, allow_stale=args.allow_stale)
    df = pd.read_pickle(ALL_MODELS)
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
