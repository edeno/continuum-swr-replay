#!/usr/bin/env python
"""Six-panel summary figure for the per-bin rate ladder (Family B).

The visual companion to ``docs/per_bin_rate_results.md`` — reads
``models_per_bin_rate/all_models.pkl`` and renders, per event-type population:
baseline rate (M0), the recent-reward dissociation on the first reward-history
basis coefficient (M7), the
reconstructed reward kernel and the self-history kernel (M7), the
model-comparison ladder (ΔAIC + CV bits/event), and the slow reward-history
component by population (M7).

Usage:
    uv run python scripts/rate/plot_per_bin_rate_summary.py
"""

from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import pandas as pd
from patsy import dmatrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from continuum_swr_replay.analysis.events.analysed_time import (  # noqa: E402
    excluded_intervals_by_session,
    require_swr_source,
)
from continuum_swr_replay.analysis.replay import (  # noqa: E402
    POPULATIONS,
    attach_is_swr_to_onsets,
)
from continuum_swr_replay.analysis.replay.fit_provenance import (  # noqa: E402
    require_fresh_inputs,
)
from continuum_swr_replay.analysis.replay.model_specs import (  # noqa: E402
    MODEL_PER_BIN_RATE,
    N_HISTORY_LAGS,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_LAGS,
    SPLINE_KNOTS,
    TMAX,
)
from continuum_swr_replay.analysis.replay.per_bin_design import (  # noqa: E402
    reward_kernel_rate_ratios,
    self_history_rate_ratios,
)
from continuum_swr_replay.intervals import (  # noqa: E402
    covered_seconds,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR  # noqa: E402
from continuum_swr_replay.paths import (
    REPLAY_CONTENT_DIR,  # noqa: E402
    model_dir,
    table_path,
)
from continuum_swr_replay.tables.covariate_registry import (  # noqa: E402
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table

RC = REPLAY_CONTENT_DIR
ALL_MODELS = model_dir("per_bin_rate", RC) / "all_models.pkl"
# The tables fit_per_bin_rate_models.py fit all_models.pkl on; checked against the
# artifact's provenance sidecar before any panel is drawn.
ALL_MODELS_INPUTS = {
    "stop_table": table_path("stop_table", RC, fmt="pkl"),
    "event_onset_table": table_path("event_onset_table", RC, fmt="pkl"),
    "event_table": table_path("event_table", RC, fmt="pkl"),
}
# Beside the data root, not parameters.FIGURE_DIR: by default the two are the
# same folder, but a run on another root (CSR_PROCESSED_DATA_DIR, e.g. a
# run_refits sensitivity run) must not overwrite the paper's rate figures.
FIGS = PROCESSED_DATA_DIR.parent / "paper" / "figures" / "rate"
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
# First reward-history basis coefficient (a recent-reward fraction weighted
# toward the newest outcomes, not lag 1 alone); the "IMMEDIATE" name is legacy.
IMMEDIATE_COEF = f"coef_{REWARD_HISTORY_COLUMNS[0]}"
SLOW_COEF = f"coef_{REWARD_HISTORY_COLUMNS[-1]}"
# Panel B's margin beyond the lowest and highest whisker: a factor in RR, applied
# as its log on the beta axis.
RR_MARGIN_FACTOR = 1.15


def _coef(df, model, oc, coef):
    return df[(df.model == model) & (df.outcome == oc)][coef].dropna()


def _lag_exposure(stop, edges):
    """Immobility seconds per lag bin since stop onset, excluded time left out.

    The clock is the stop's own (``stop_end - stop_start``): a stop covers lag
    bin ``[e, e + dt)`` from ``stop_start + e`` to its end, less the excluded
    seconds there (``analysed_time.excluded_intervals_by_session``). Its
    ``immobility_duration_s`` is already net of excluded time, so it is not a
    length on this clock.
    """
    excluded = excluded_intervals_by_session(stop)
    keys = ["animal", "session_date", "epoch"]
    denom = np.zeros(len(edges) - 1)
    for session, stops_g in stop.groupby(keys, sort=False):
        session_excluded = excluded.loc[
            (excluded[keys] == pd.Series(session, index=keys)).all(axis=1)
        ]
        start = stops_g["stop_start"].to_numpy(dtype=float)
        span = stops_g["stop_end"].to_numpy(dtype=float) - start
        for k, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            lo, hi = start + low, start + np.minimum(span, high)
            inside = hi > lo
            denom[k] += (hi - lo)[inside].sum() - covered_seconds(
                lo[inside], hi[inside], session_excluded
            ).sum()
    return denom


def _timecourse(stop, onsets, oc, dt=0.1, tmax=4.0):
    """Descriptive onset rate (events / s of immobility) vs time since stop onset.

    Numerator = onset-count histogram over t_since_stop_onset; denominator =
    immobility-seconds per t-bin (:func:`_lag_exposure`: each stop's span in
    the bin, excluded time left out). The denominator already makes this a
    proper rate despite varying stop durations; what changes with tc is the
    *support* (fewer, longer, more-rewarded stops survive) — see the survival
    overlay.
    """
    t = POPULATIONS[oc](onsets)["t_since_stop_onset"].to_numpy()
    t = t[(t >= 0) & (t < tmax)]
    edges = np.arange(0, tmax + dt, dt)
    num, _ = np.histogram(t, edges)
    denom = _lag_exposure(stop, edges)
    rate = num / np.where(denom > 0, denom, np.nan)
    return edges[:-1] + dt / 2, rate


def _survival(stop, tmax=4.0, dt=0.1):
    """Fraction of stops still immobile at each tc (the support of the curve),
    on the stops' own clock (``stop_end - stop_start``)."""
    span = (stop["stop_end"] - stop["stop_start"]).to_numpy()
    span = span[span > 0]
    tc = np.arange(dt / 2, tmax, dt)
    return tc, np.array([(span > t).mean() for t in tc])


def _shade_low_support(ax, stop, thresh=0.2, tmax=4.0):
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
    for oc in POPS4:
        tc, rate = _timecourse(stop, onsets, oc)
        ax.plot(tc, rate, color=STYLE[oc][1], label=STYLE[oc][0], lw=1.8)
    ax.set_xlabel("time since stop onset (s)")
    ax.set_ylabel("onset rate (events / s)")
    ax.set_title("A0. Within-stop time course (descriptive)")
    ax.set_xlim(0, 4)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    _shade_low_support(ax, stop)
    ax2 = ax.twinx()
    stc, surv = _survival(stop)
    ax2.plot(stc, surv, color="0.5", ls="--", lw=1.0)
    ax2.set_ylabel("fraction of stops still immobile", color="0.5", fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", colors="0.5")


def _spline_basis(tc_grid):
    """Fixed-knot spline basis at tc_grid (same basis used in the fits)."""
    f = f"bs(tc, knots={SPLINE_KNOTS}, lower_bound=0.0, upper_bound={TMAX}) - 1"
    return np.asarray(dmatrix(f, {"tc": tc_grid}, return_type="dataframe"))


def _panel_fitted_timecourse(ax, df, stop):
    """Reconstructed M7 spline time course (reward/history at reference), per pop.

    Now reconstructable because the spline knots are fixed (data-independent):
    rate(tc) = exp(intercept + B(tc) @ spline_coef), averaged across sessions.
    """
    grid = np.linspace(0, TMAX, 81)
    B = _spline_basis(grid)
    scols = sorted(
        c
        for c in df.columns
        if c.startswith("coef_bs(tc, knots=")
        and c.endswith("]")
        and f"{REWARD_HISTORY_COLUMNS[0]}:" not in c
    )
    for oc in POPS4:
        s = df[(df.model == FULL) & (df.outcome == oc)]
        beta = s[scols].to_numpy(float)
        b0 = s["coef_Intercept"].to_numpy(float)
        ok = ~np.isnan(beta).any(axis=1) & ~np.isnan(b0)
        if ok.sum() == 0:
            continue
        curves = np.exp(b0[ok, None] + beta[ok] @ B.T)  # (n_sess, n_grid)
        ax.plot(
            grid,
            np.median(curves, axis=0),
            color=STYLE[oc][1],
            label=STYLE[oc][0],
            lw=1.8,
        )
    ax.set_xlabel("time since stop onset (s)")
    ax.set_ylabel("fitted rate (events / s)")
    ax.set_title("A1. Fitted time course (M7r spline, reward/history at ref)")
    ax.set_xlim(0, 4)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    _shade_low_support(ax, stop)


def _panel_baseline(ax, df):
    for i, oc in enumerate(POPS4):
        v = np.exp(_coef(df, "M0_const", oc, "coef_Intercept"))
        err = [[v.median() - v.quantile(0.25)], [v.quantile(0.75) - v.median()]]
        ax.bar(i, v.median(), color=STYLE[oc][1], alpha=0.7, yerr=err, capsize=4)
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
    data = [_coef(df, FULL, oc, IMMEDIATE_COEF) for oc in POPS3]
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

    The fitted features are bump-weighted reward fractions, so an outcome at lag
    ``j`` contributes ``sum_k coef_k * B[k, j] / sum_j' B[k, j']`` to the log rate.
    """
    lags = np.arange(1, REWARD_HISTORY_N_LAGS + 1)
    for oc in POPS4:
        sessions = df[(df.model == FULL) & (df.outcome == oc)]
        ax.plot(
            lags,
            reward_kernel_rate_ratios(sessions),
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
    """Self-history kernel reconstructed from the raised-cosine coefficients.

    ``hist_rcos_k`` is the plain projection of the ``onset_lag{j}`` counts onto
    the raw bumps, so an onset ``j`` bins back contributes
    ``sum_k coef_k * B[k, j]`` to the log rate.
    """
    lags = np.arange(1, N_HISTORY_LAGS + 1)
    for oc in POPS4:
        sessions = df[(df.model == FULL) & (df.outcome == oc)]
        ax.plot(
            lags * 15,
            self_history_rate_ratios(sessions),
            marker="o",
            color=STYLE[oc][1],
            label=STYLE[oc][0],
            lw=1.8,
            ms=4,
        )
    ax.axhline(1, color="0.5", ls=":")
    ax.set_xlabel("self-history lag (ms)")
    ax.set_ylabel("rate ratio")
    ax.set_title("D. Self-history (M7r): refractory then bursty ~100 ms")
    ax.legend(frameon=False, fontsize=7)


def _panel_ladder(ax, df):
    g = df[df.outcome == "non_local"]
    aic0 = g[g.model == "M0_const"]["aic"].median()
    daic = [g[g.model == m]["aic"].median() - aic0 for m in LADDER]
    x = np.arange(len(LADDER))
    ax.bar(x, daic, color="0.6")
    ax.set_xticks(x)
    ax.set_xticklabels(LADDER, rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("median ΔAIC vs M0 (lower=better)")
    ax.set_title("E. Model ladder (non-local): ΔAIC + CV bits/event")
    ax2 = ax.twinx()
    cvb = [g[g.model == m]["cv_bits_per_event"].median() for m in LADDER]
    ax2.plot(x, cvb, marker="D", color="C1", lw=1.5, ms=4)
    ax2.axhline(0, color="C1", ls=":", alpha=0.5)
    ax2.set_ylabel("CV bits/event (orange, higher=better)", color="C1")
    ax2.tick_params(axis="y", colors="C1")


def _panel_covar(ax, df):
    w = 0.35
    blocks = [(FULL, SLOW_COEF)]
    for j, (model, coef) in enumerate(blocks):
        for i, oc in enumerate(POPS3):
            rr = np.exp(_coef(df, model, oc, coef).median())
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


def main() -> None:
    args = build_parser().parse_args()
    require_fresh_inputs(ALL_MODELS, ALL_MODELS_INPUTS, allow_stale=args.allow_stale)
    df = pd.read_pickle(ALL_MODELS)
    stop = read_table(table_path("stop_table", RC, fmt="pkl"))
    onset_table = read_table(table_path("event_onset_table", RC, fmt="pkl"))
    event_table = read_table(table_path("event_table", RC, fmt="pkl"))
    require_covariate_schema(stop, table="stop")
    require_covariate_schema(onset_table, table="event_onset")
    require_covariate_schema(event_table, table="event")
    # the time course leaves out the configured source's excluded time
    require_swr_source(stop, what=str(ALL_MODELS_INPUTS["stop_table"]))
    require_swr_source(onset_table, what=str(ALL_MODELS_INPUTS["event_onset_table"]))
    require_swr_source(event_table, what=str(ALL_MODELS_INPUTS["event_table"]))
    onsets = attach_is_swr_to_onsets(onset_table, event_table)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplot_mosaic(
        [["time", "rate", "rew1", "ladder"], ["timefit", "rewk", "hist", "covar"]],
        figsize=(21, 9),
    )
    _panel_timecourse(ax["time"], stop, onsets)
    _panel_fitted_timecourse(ax["timefit"], df, stop)
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
    out = FIGS / "per_bin_rate_ladder_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
