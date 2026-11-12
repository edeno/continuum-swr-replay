#!/usr/bin/env python
"""Plot the per-bin conditional-intensity ladder (Family B) from its tidy output.

Reads the persisted per-session coefficients in
``models_per_bin_rate/all_models.pkl`` (produced by
``fit_per_bin_rate_models.py``) and renders the two headline figures -- no
re-fitting happens here. The summaries drawn (medians, IQR, per-animal medians,
the paired signed-rank test) are ``rate_fit_statistics``' and
``coefficient_statistics``', over the sessions the summary reads them over.

Both figures read the full model ``M7r_time_histcos_rew`` (time spline +
raised-cosine self-history + reward kernel), so every effect is shown
controlled for the within-stop time course and the same-event clustering:

* ``reward_kernel_basis.png`` -- the reward rate-ratio kernel over lags 1-10
  reconstructed from the three raised-cosine basis coefficients per
  population, over the sessions whose three reward terms pass the per-term
  gate (per-session median and IQR, with the per-animal medians drawn through
  them; ``rate_fit_statistics.reward_kernel_by_lag``), and the dissociation between SWR-coupled and non-SWR
  non-local events on the first reward-history basis coefficient
  (``reward_history_rcos_0``, a recent-reward fraction weighted toward the
  newest outcomes: about 0.33 / 0.26 / 0.17 / 0.11 on lags 1-4), over the
  sessions the per-term gate keeps.
* ``selfhistory_autocorr.png`` -- the within-stop self-history rate ratio
  over lags 1..10 after the previous event's END (the at-risk design counts
  history from event ends), reconstructed from the four raised-cosine basis
  coefficients.

Usage:
    uv run python scripts/behavior/rate/plot_per_bin_rate_models.py
    uv run python scripts/behavior/rate/plot_per_bin_rate_models.py --allow-stale
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from continuum_swr_replay.analysis.behavior.coefficient_statistics import (  # noqa: E402
    fit_rows,
    gated_coefficient_statistics,
    paired_coefficient_statistics,
)
from continuum_swr_replay.analysis.behavior.contrast_aggregation import (  # noqa: E402
    gate_term_sessions,
)
from continuum_swr_replay.analysis.behavior.fit_provenance import (  # noqa: E402
    require_fresh_inputs,
)
from continuum_swr_replay.analysis.behavior.model_specs import (  # noqa: E402
    BIN_S,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_LAGS,
)
from continuum_swr_replay.analysis.behavior.rate.rate_fit_statistics import (  # noqa: E402
    reward_kernel_animal_medians,
    reward_kernel_by_lag,
    self_history_kernel_by_lag,
)
from continuum_swr_replay.paths import (
    figure_dir,
    model_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.cli import fit_figure_parser  # noqa: E402


def all_models_path() -> Path:
    """The per-bin ladder's fitted bundle, ``all_models.pkl``."""
    return model_dir("per_bin_rate") / "all_models.pkl"


def all_models_input_paths() -> dict[str, Path]:
    """The tables fit_per_bin_rate_models.py fit all_models.pkl on. Checked against
    the artifact's provenance sidecar before plotting, so a figure cannot be
    rendered from coefficients that predate the tables on disk. Under
    ``paths.replay_content_dir()``, read when called.
    """
    rc = replay_content_dir()
    return {
        "stop_table": table_path("stop_table", rc),
        "event_onset_table": table_path("event_onset_table", rc),
        "event_table": table_path("event_table", rc),
    }


MODEL = "M7r_time_histcos_rew"

# outcome key -> (display label, colour)
POP_STYLE = {
    "swr": ("detected SWR", "k"),
    "non_local": ("non-local (all)", "C2"),
    "non_local_swr": ("non-local w/ SWR", "C3"),
    "non_local_no_swr": ("non-local NO SWR", "C0"),
}


# The first raised-cosine reward-history basis coefficient. Its feature is a
# reward fraction weighted toward the newest outcomes (about 0.33 / 0.26 / 0.17 /
# 0.11 on lags 1-4), not the lag-1 outcome alone; the "IMMEDIATE" name is legacy.
IMMEDIATE_TERM = REWARD_HISTORY_COLUMNS[0]
IMMEDIATE_COEF = f"coef_{IMMEDIATE_TERM}"


def plot_reward_kernel(m7: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    lags = np.arange(1, REWARD_HISTORY_N_LAGS + 1)
    shown = []  # every rate ratio drawn, for the y-limits
    for i, oc in enumerate(["swr", "non_local_swr", "non_local_no_swr"]):
        label, col = POP_STYLE[oc]
        # the sessions the reward gate keeps, as the summary reads the kernel
        fits = fit_rows(m7, outcome=oc)
        kernel = reward_kernel_by_lag(fits)
        offset = 0.05 * (i - 1)
        # Per-animal medians first, so the across-session summary reads on top.
        for animal_ratio in reward_kernel_animal_medians(fits).to_numpy():
            shown.append(animal_ratio)
            ax.plot(
                lags + offset,
                animal_ratio,
                color=col,
                lw=0.8,
                alpha=0.35,
                zorder=1,
            )
        med, lo, hi = (
            kernel[column].to_numpy()
            for column in ("median", "lower_quartile", "upper_quartile")
        )
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
        "Reward kernel (time + self-history controlled), reward-gated sessions\n"
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
    # the sessions the per-term gate keeps, as the summary reads the coefficient
    gated = {
        oc: fits[gate_term_sessions(fits, IMMEDIATE_TERM).kept]
        for oc in order
        for fits in [fit_rows(m7, outcome=oc)]
    }
    data = [np.exp(gated[oc][IMMEDIATE_COEF]) for oc in order]
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
    ax.set_ylabel(
        "recent-reward effect (first history basis) — RR per unit (per session)"
    )
    # The first basis coefficient is a rate ratio per unit of a 0-1 reward fraction,
    # so the per-session spread is multiplicative and covers orders of
    # magnitude; a linear axis truncates the upper boxes.
    ax.set_yscale("log")

    # paired dissociation annotation: non_local_swr vs non_local_no_swr, over
    # the sessions in which the gate keeps the coefficient in both
    paired = paired_coefficient_statistics(
        gated["non_local_swr"], gated["non_local_no_swr"], IMMEDIATE_COEF
    )
    if paired["n_sessions"] < 2:
        raise SystemExit(
            f"Only {paired['n_sessions']} paired sessions for the SWR/no-SWR "
            "dissociation; run fit_per_bin_rate_models.py over more sessions first."
        )
    ax.set_title(
        f"Recent-reward effect (first history basis) by population\n"
        f"NL-SWR > NL-no-SWR: {paired['positive_fraction']:.0%} of "
        f"{paired['n_sessions']}, {paired['n_animals_positive']}/"
        f"{paired['n_animals']} animals, p={paired['signed_rank_p']:.1e}"
    )

    fig.tight_layout()
    out = figure_dir("rate") / "reward_kernel_basis.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
    for oc in order:
        coefficient = gated_coefficient_statistics(
            fit_rows(m7, outcome=oc), IMMEDIATE_TERM
        )
        print(
            f"  {POP_STYLE[oc][0]:18s}: first-basis RR="
            f"{coefficient['exp_median']:.2f} (n={coefficient['n_sessions']})"
        )


def plot_selfhistory(m7: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for oc in ["swr", "non_local", "non_local_swr", "non_local_no_swr"]:
        label, col = POP_STYLE[oc]
        kernel = self_history_kernel_by_lag(fit_rows(m7, outcome=oc))
        med, lo, hi = (
            kernel[column].to_numpy()
            for column in ("median", "lower_quartile", "upper_quartile")
        )
        ax.errorbar(
            kernel.index * BIN_S * 1000,
            med,
            yerr=[med - lo, hi - med],
            marker="o",
            color=col,
            label=label,
            capsize=3,
        )
    ax.axhline(1, color="0.5", ls=":")
    ax.set_xlabel("time since the previous event ended (ms)")
    ax.set_ylabel("self-history rate ratio (per-session median, IQR)")
    ax.set_title("Within-stop event history, from event ends\n(<1 = fewer, >1 = more)")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = figure_dir("rate") / "selfhistory_autocorr.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser (also the provenance gate's opt-out)."""
    return fit_figure_parser(__doc__)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    all_models, inputs = all_models_path(), all_models_input_paths()
    require_fresh_inputs(all_models, inputs, allow_stale=args.allow_stale)
    df = pd.read_pickle(all_models)
    m7 = df[df.model == MODEL]
    if m7.empty:
        raise SystemExit(
            f"No {MODEL} rows in all_models.pkl; run fit_per_bin_rate_models.py first."
        )
    figure_dir("rate").mkdir(parents=True, exist_ok=True)
    plot_reward_kernel(m7)
    plot_selfhistory(m7)


if __name__ == "__main__":
    main()
