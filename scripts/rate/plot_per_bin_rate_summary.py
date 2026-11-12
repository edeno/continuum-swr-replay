#!/usr/bin/env python
"""Six-panel summary figure for the per-bin rate ladder (Family B).

The visual companion to the per-bin ladder's reported values (question 3.3) — reads
``models_per_bin_rate/all_models.pkl`` and renders, per event-type population:
the descriptive and the fitted (M5_time) within-stop time course,
baseline rate (M0), the recent-reward dissociation on the first reward-history
basis coefficient (M7), the
reconstructed reward kernel and the self-history kernel (M7), the
model-comparison ladder (ΔAIC + CV bits/event), and the slow reward-history
component by population (M7).

Every panel draws the statistic the behavior summary reports, from the function
that computes it (``rate_fit_statistics``, ``stop_statistics``,
``coefficient_statistics``), over the same sessions: a reward term over the
sessions the per-term gate keeps, AIC steps differenced within session, the time
courses' support over the well-anchored stops.

Usage:
    uv run python scripts/rate/plot_per_bin_rate_summary.py
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

from continuum_swr_replay.analysis.events.analysed_time import (  # noqa: E402
    require_swr_source,
)
from continuum_swr_replay.analysis.events.cohort_selection import (  # noqa: E402
    select_stop_cohort,
)
from continuum_swr_replay.analysis.replay import (  # noqa: E402
    POPULATIONS,
    attach_is_swr_to_onsets,
)
from continuum_swr_replay.analysis.replay.coefficient_statistics import (  # noqa: E402
    aic_difference_statistics,
    fit_rows,
    gated_coefficient_statistics,
)
from continuum_swr_replay.analysis.replay.contrast_aggregation import (  # noqa: E402
    gate_term_sessions,
)
from continuum_swr_replay.analysis.replay.fit_provenance import (  # noqa: E402
    require_fresh_inputs,
)
from continuum_swr_replay.analysis.replay.model_specs import (  # noqa: E402
    BIN_S,
    MODEL_PER_BIN_RATE,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_LAGS,
    TMAX,
)
from continuum_swr_replay.analysis.replay.rate_fit_statistics import (  # noqa: E402
    baseline_rate_statistics,
    fitted_time_course,
    ladder_fit_statistics,
    reward_kernel_by_lag,
    self_history_kernel_by_lag,
)
from continuum_swr_replay.analysis.replay.stop_statistics import (  # noqa: E402
    onset_rate_time_course,
    still_immobile_fraction,
)
from continuum_swr_replay.figure_utilities import save_figure  # noqa: E402
from continuum_swr_replay.paths import (
    figure_dir,
    model_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.tables.covariate_registry import (  # noqa: E402
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table


def all_models_path() -> Path:
    """The per-bin ladder's fitted bundle, ``all_models.pkl``."""
    return model_dir("per_bin_rate") / "all_models.pkl"


def all_models_input_paths() -> dict[str, Path]:
    """The tables fit_per_bin_rate_models.py fit all_models.pkl on; checked against
    the artifact's provenance sidecar before any panel is drawn. Under
    ``paths.replay_content_dir()``, read when called.
    """
    rc = replay_content_dir()
    return {
        "stop_table": table_path("stop_table", rc),
        "event_onset_table": table_path("event_onset_table", rc),
        "event_table": table_path("event_table", rc),
    }


STYLE = {  # outcome -> (label, color)
    "swr": ("detected SWR", "k"),
    "non_local": ("non-local (all)", "C2"),
    "non_local_swr": ("NL w/ SWR", "C3"),
    "non_local_no_swr": ("NL no-SWR", "C0"),
}
# The stop-clock ladder, in ladder order, taken from the executable spec so a
# new rung cannot be silently missing from panel E. The arrival-clock rungs are
# fit on a subset of the frozen cohort's rows, so their AIC and held-out
# likelihood are not on this ladder's scale and they are deliberately excluded.
LADDER = [n for n, s in MODEL_PER_BIN_RATE.items() if s.time_clock == "stop"]
POPS4 = ["swr", "non_local", "non_local_swr", "non_local_no_swr"]
POPS3 = ["swr", "non_local_swr", "non_local_no_swr"]
SHORT = ["SWR", "NL\nw/SWR", "NL\nno-SWR"]
FULL = "M7r_time_histcos_rew"
# The stop-clock time rung whose fitted time course the summary reads its peak from.
TIME_RUNG = "M5_time"
# First reward-history basis coefficient (a recent-reward fraction weighted
# toward the newest outcomes, not lag 1 alone); the "IMMEDIATE" name is legacy.
IMMEDIATE_TERM = REWARD_HISTORY_COLUMNS[0]
SLOW_TERM = REWARD_HISTORY_COLUMNS[-1]
# The descriptive time course's bin, in seconds.
DESCRIPTIVE_BIN_S = 0.1
# Panel B's margin beyond the lowest and highest whisker: a factor in RR, applied
# as its log on the beta axis.
RR_MARGIN_FACTOR = 1.15


def _gated_coefficient(df, oc, term):
    """``coef_<term>`` of the full rung over the sessions the per-term gate keeps,
    the sessions the summary reads the coefficient over."""
    fits = fit_rows(df, model=FULL, outcome=oc)
    return fits.loc[gate_term_sessions(fits, term).kept, f"coef_{term}"]


def _survival(stop, tmax=TMAX, dt=DESCRIPTIVE_BIN_S):
    """Share of stops still immobile at each tc (the support of the curve), on
    the stops' own clock (``stop_statistics.still_immobile_fraction``)."""
    tc = np.arange(dt / 2, tmax, dt)
    return tc, np.array([still_immobile_fraction(stop, t) for t in tc])


def _shade_low_support(ax, stop, thresh=0.2, tmax=TMAX):
    """Grey the tc region where <thresh of stops survive (curve is a thin subset)."""
    tc, surv = _survival(stop, tmax)
    below = tc[surv < thresh]
    if below.size:
        ax.axvspan(below.min(), tmax, color="0.85", zorder=0)
        ax.text(
            below.min(),
            ax.get_ylim()[1] * 0.97,
            f" <{thresh:.0%} stops survive",
            fontsize=6,
            va="top",
            color="0.4",
        )


def _panel_timecourse(ax, stop, onsets):
    """Descriptive onset rate (events per second of immobility) against time
    since the stop began (``stop_statistics.onset_rate_time_course``). The
    exposure makes it a rate whatever the stops' durations; what changes with
    time is the support (fewer, longer, more-rewarded stops), shown by the
    survival overlay."""
    for oc in POPS4:
        course = onset_rate_time_course(
            stop, POPULATIONS[oc](onsets), TMAX, DESCRIPTIVE_BIN_S
        )
        ax.plot(
            course.index, course["rate"], color=STYLE[oc][1], label=STYLE[oc][0], lw=1.8
        )
    ax.set_xlabel("time since stop onset (s)")
    ax.set_ylabel("onset rate (events / s)")
    ax.set_title("A0. Within-stop time course (descriptive, well-anchored stops)")
    ax.set_xlim(0, 4)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    _shade_low_support(ax, stop)
    ax2 = ax.twinx()
    stc, surv = _survival(stop)
    ax2.plot(stc, surv, color="0.5", ls="--", lw=1.0)
    ax2.set_ylabel(
        "fraction of well-anchored stops still immobile", color="0.5", fontsize=8
    )
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", colors="0.5")


def _panel_fitted_timecourse(ax, df, stop):
    """The median fitted time course of the stop-clock time rung, per population.

    ``rate_fit_statistics.fitted_time_course``: the curve whose peak the
    summary reports, on the same grid over the spline's interior knots, with
    separated sessions left out.
    """
    for oc in POPS4:
        grid, curve, _ = fitted_time_course(
            fit_rows(df, model=TIME_RUNG, outcome=oc), "stop"
        )
        ax.plot(grid, curve, color=STYLE[oc][1], label=STYLE[oc][0], lw=1.8)
    ax.set_xlabel("time since stop onset (s)")
    ax.set_ylabel("fitted rate (events / s)")
    ax.set_title(f"A1. Fitted time course ({TIME_RUNG}, between its interior knots)")
    ax.set_xlim(0, 4)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    _shade_low_support(ax, stop)


def _panel_baseline(ax, df):
    for i, oc in enumerate(POPS4):
        rate = baseline_rate_statistics(fit_rows(df, model="M0_const", outcome=oc))
        median = rate["median"]
        err = [[median - rate["lower_quartile"]], [rate["upper_quartile"] - median]]
        ax.bar(i, median, color=STYLE[oc][1], alpha=0.7, yerr=err, capsize=4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(
        [STYLE[o][0] for o in POPS4], rotation=20, ha="right", fontsize=8
    )
    ax.set_ylabel("baseline rate (events / s)")
    ax.set_title("A. Baseline rate (M0_const)")


def _panel_reward1(ax, df):
    # The boxes are of beta = log(RR), the coefficient itself, so the quartiles and
    # the 1.5 IQR whiskers are taken in log space. On RR they are not: a rate ratio
    # is skewed, the whisker runs to the minimum, and one near-separated session
    # stretched the axis over five decades. The axis is linear in beta, with
    # ticks at the log of plain RR values and labelled with the RR; RR = 1 is
    # beta = 0.
    data = [_gated_coefficient(df, oc, IMMEDIATE_TERM) for oc in POPS3]
    bp = ax.boxplot(data, labels=SHORT, showfliers=False, patch_artist=True)
    for patch, oc in zip(bp["boxes"], POPS3, strict=True):
        patch.set_facecolor(STYLE[oc][1])
        patch.set_alpha(0.4)
    ax.axhline(0, color="0.5", ls=":")
    # Fliers are off, so the data end at the lowest and highest whisker or cap;
    # the limits are those, less and plus a margin. RR = 1 stays in view.
    ends = np.concatenate([line.get_ydata() for line in bp["whiskers"] + bp["caps"]])
    low, high = min(ends.min(), 0.0), max(ends.max(), 0.0)
    margin = np.log(RR_MARGIN_FACTOR)
    ax.set_ylim(low - margin, high + margin)
    rate_ratios = [
        m * 10.0**e
        for e in range(-6, 7)
        for m in (1, 2, 5)
        if low <= np.log(m * 10.0**e) <= high
    ]
    ax.set_yticks(np.log(rate_ratios), labels=[f"{r:g}" for r in rate_ratios])
    ax.set_ylabel("recent-reward RR, first history basis (per session, log scale)")
    ax.set_title("B. First reward-history basis component (M7r, controlled)")


def _panel_reward_kernel(ax, df):
    """Reward kernel reconstructed from the raised-cosine basis coefficients.

    The median over the sessions the reward gate keeps of each session's kernel
    (``rate_fit_statistics.reward_kernel_by_lag``), the sessions the summary
    reads the kernel over.
    """
    lags = np.arange(1, REWARD_HISTORY_N_LAGS + 1)
    for oc in POPS4:
        kernel = reward_kernel_by_lag(fit_rows(df, model=FULL, outcome=oc))
        ax.plot(
            kernel.index,
            kernel["median"],
            marker="o",
            color=STYLE[oc][1],
            label=STYLE[oc][0],
            lw=1.8,
            ms=4,
        )
    ax.axhline(1, color="0.5", ls=":")
    ax.set_xticks(lags)
    ax.set_ylim(0, 4.3)
    ax.set_xlabel("reward lag (trials back)")
    ax.set_ylabel("rate ratio")
    ax.set_title("C. Reward kernel (M7r, reconstructed from the basis)")
    ax.legend(frameon=False, fontsize=7)


def _panel_selfhistory(ax, df):
    """Self-history kernel from event ends, reconstructed per session from the
    raised-cosine coefficients: the median over sessions
    (``rate_fit_statistics.self_history_kernel_by_lag``).
    """
    for oc in POPS4:
        kernel = self_history_kernel_by_lag(fit_rows(df, model=FULL, outcome=oc))
        ax.plot(
            kernel.index * BIN_S * 1000,
            kernel["median"],
            marker="o",
            color=STYLE[oc][1],
            label=STYLE[oc][0],
            lw=1.8,
            ms=4,
        )
    ax.axhline(1, color="0.5", ls=":")
    ax.set_xlabel("time since the previous event ended (ms)")
    ax.set_ylabel("rate ratio")
    ax.set_title("D. Self-history from event ends (M7r)")
    ax.legend(frameon=False, fontsize=7)


def _panel_ladder(ax, df):
    """Each stop-clock rung against M0_const for the non-local population: the
    median within-session AIC difference (``aic_difference_statistics``) and
    the median held-out bits per event over the sessions with a finite score
    (``rate_fit_statistics.ladder_fit_statistics``)."""
    g = fit_rows(df, outcome="non_local")
    null = fit_rows(g, model="M0_const")
    daic = [
        aic_difference_statistics(fit_rows(g, model=m), null)["median"] for m in LADDER
    ]
    ladder = ladder_fit_statistics(g)
    x = np.arange(len(LADDER))
    ax.bar(x, daic, color="0.6")
    ax.set_xticks(x)
    ax.set_xticklabels(LADDER, rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("median within-session ΔAIC vs M0 (lower=better)")
    ax.set_title("E. Model ladder (non-local): ΔAIC + CV bits/event")
    ax2 = ax.twinx()
    cvb = [ladder.loc[(m, "non_local"), "cv_bits_median"] for m in LADDER]
    ax2.plot(x, cvb, marker="D", color="C1", lw=1.5, ms=4)
    ax2.axhline(0, color="C1", ls=":", alpha=0.5)
    ax2.set_ylabel("CV bits/event (orange, higher=better)", color="C1")
    ax2.tick_params(axis="y", colors="C1")


def _panel_covar(ax, df):
    w = 0.35
    blocks = [(FULL, SLOW_TERM)]
    for j, (model, term) in enumerate(blocks):
        for i, oc in enumerate(POPS3):
            rr = gated_coefficient_statistics(
                fit_rows(df, model=model, outcome=oc), term
            )["exp_median"]
            ax.bar(
                i + j * (w + 0.05) - 0.2,
                rr,
                width=w,
                color=STYLE[oc][1],
                alpha=0.5 + 0.5 * (1 - j),
                edgecolor="k",
                linewidth=0.5,
            )
    ax.axhline(1, color="0.5", ls=":")
    ax.set_xticks(range(3))
    ax.set_xticklabels(SHORT)
    ax.set_ylabel("rate ratio")
    ax.set_title("F. Slow reward-history component by type (M7r)")


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


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    all_models, inputs = all_models_path(), all_models_input_paths()
    require_fresh_inputs(all_models, inputs, allow_stale=args.allow_stale)
    df = pd.read_pickle(all_models)
    stop = read_table(inputs["stop_table"])
    onset_table = read_table(inputs["event_onset_table"])
    event_table = read_table(inputs["event_table"])
    require_covariate_schema(stop, table="stop")
    require_covariate_schema(onset_table, table="event_onset")
    require_covariate_schema(event_table, table="event")
    # the time course leaves out the configured source's excluded time
    require_swr_source(stop, what=str(inputs["stop_table"]))
    require_swr_source(onset_table, what=str(inputs["event_onset_table"]))
    require_swr_source(event_table, what=str(inputs["event_table"]))
    onsets = attach_is_swr_to_onsets(onset_table, event_table)
    figure_dir("rate").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplot_mosaic(
        [["time", "rate", "rew1", "ladder"], ["timefit", "rewk", "hist", "covar"]],
        figsize=(21, 9),
    )
    # The time courses and their support are read on the well-anchored stops,
    # the cohort the ladder is fit on and the summary reads the survival over.
    anchored, _ = select_stop_cohort(stop, cohort="at_well")
    anchored_onsets = onsets[onsets["stop_id"].isin(anchored["stop_id"])]
    _panel_timecourse(ax["time"], anchored, anchored_onsets)
    _panel_fitted_timecourse(ax["timefit"], df, anchored)
    _panel_baseline(ax["rate"], df)
    _panel_reward1(ax["rew1"], df)
    _panel_reward_kernel(ax["rewk"], df)
    _panel_selfhistory(ax["hist"], df)
    _panel_ladder(ax["ladder"], df)
    _panel_covar(ax["covar"], df)
    # Derive the cohort from the data actually plotted. A hardcoded count
    # silently misreports the corpus whenever a session fails to regenerate
    # (attach_is_swr_to_onsets excludes non-shared sessions).
    n_sessions = onsets[["animal", "session_date", "epoch"]].drop_duplicates().shape[0]
    fig.suptitle(
        "Per-bin replay-rate ladder (Family B): fitted effects by event type, "
        f"{n_sessions} sessions",
        y=1.0,
    )
    fig.tight_layout()
    out = figure_dir("rate") / "per_bin_rate_ladder_summary.png"
    save_figure(
        fig, out.stem, base_dir=out.parent, dpi=150, formats=["png"], transparent=False
    )
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
