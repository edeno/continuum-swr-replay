#!/usr/bin/env python
"""Point-process model checking for the per-bin ladder (Family B).

Treats the replay event train as a point process and checks it as one. The
scalar goodness-of-fit (``ks_statistic``, ``ks_within_band``, ``dispersion``,
``cv_bits_per_event``) for every fit already lives in
``models_per_bin_rate/all_models.pkl`` (emitted per fit by ``per_bin_rate_gof``);
this script *reads* those to tell the model-checking story up the ladder, and
re-fits a couple of example sessions inline to render the diagnostic quartet
(KS, Q-Q, rescaled-ISI autocorrelation, cumulative point-process residuals).

All time-rescaling math is imported from
``non_local_detector.model_checking.sorted_spikes`` -- nothing is re-implemented
here.

Usage:
    uv run python scripts/rate/plot_model_checking.py
"""

from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from non_local_detector.model_checking.sorted_spikes import (  # noqa: E402
    TimeRescaling,
    point_process_residuals,
)

from continuum_swr_replay.analysis.hippocampus.swr_source import (  # noqa: E402
    excluded_intervals_by_session,
    require_swr_source,
)
from continuum_swr_replay.analysis.replay import (  # noqa: E402
    BIN_S,
    MODEL_PER_BIN_RATE,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    POPULATIONS,
    STOP_COVARIATES,
    TMAX,
    aggregate_to_timebins,
    attach_is_swr_to_onsets,
    fit_per_bin_rate_model,
    freeze_per_bin_cohort,
)
from continuum_swr_replay.analysis.replay.covariate_registry import (  # noqa: E402
    require_covariate_schema,
)
from continuum_swr_replay.analysis.replay.fit_provenance import (  # noqa: E402
    require_fresh_inputs,
)
from continuum_swr_replay.analysis.replay.model_specs import (  # noqa: E402
    longest_history_window_s,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR as P  # noqa: E402
from continuum_swr_replay.session_keys import SESSION_COLUMNS
from continuum_swr_replay.table_io import read_table

RC = P / "replay_content"
ALL_MODELS = RC / "models_per_bin_rate" / "all_models.pkl"
# The tables fit_per_bin_rate_models.py fit all_models.pkl on; checked against the
# artifact's provenance sidecar before the fitted GOF columns are read.
ALL_MODELS_INPUTS = {
    "stop_table": RC / "stop_table.pkl",
    "event_onset_table": RC / "event_onset_table.pkl",
    "event_table": RC / "event_table.pkl",
}
FIGS = P.parent / "paper" / "figures" / "rate"
FIGS.mkdir(parents=True, exist_ok=True)
SC = list(SESSION_COLUMNS)
# The two clocks are two ladders, not one. An arrival rung is fit to the
# arrival window — a strict subset of the frozen cohort's rows — so its AIC,
# deviance, dispersion and held-out likelihood are on a different row set from
# every stop-clock rung's. Drawn on one continuous axis they read as further
# steps of the same ladder, which is exactly the comparison that is invalid, so
# they are plotted after a marked gap with the two groups labelled.
STOP_LADDER = [n for n, s in MODEL_PER_BIN_RATE.items() if s.time_clock == "stop"]
ARRIVAL_LADDER = [n for n, s in MODEL_PER_BIN_RATE.items() if s.time_clock == "arrival"]
# Gap in x units between the last stop rung and the first arrival rung.
CLOCK_GAP = 1.0
FULL = "M7r_time_histcos_rew"
# The arrival rungs are fit on the arrival window, which is not a neutral
# subset: it keeps a larger share of the SWR onsets than of the non-SWR ones,
# so a BETWEEN-population arrival comparison is confounded by retention. The
# note gives the warning in words, with no share. It used to quote two shares
# as constants and name the cohort audit (``per_bin_rate_cohort_audit.csv``) as
# their source; that audit counts stops, one row per (model, population), so it
# holds no onset retention to compute them from. A share belongs in the note
# only if it is computed when the figure is made.
ARRIVAL_RETENTION_NOTE = (
    "arrival rungs are read on a different row set — the arrival window keeps a "
    "larger share of SWR than of non-SWR non-local onsets, so between-population "
    "arrival differences are partly retention"
)


def _ladder_positions() -> tuple[np.ndarray, np.ndarray, float]:
    """x positions for the stop rungs, the arrival rungs, and the gap between."""
    stop_x = np.arange(len(STOP_LADDER), dtype=float)
    boundary = len(STOP_LADDER) - 1 + CLOCK_GAP / 2
    arrival_x = (
        len(STOP_LADDER) - 1 + CLOCK_GAP + np.arange(len(ARRIVAL_LADDER), dtype=float)
    )
    return stop_x, arrival_x, boundary


def plot_gof_ladder(df: pd.DataFrame) -> None:
    """GOF metrics up the ladder, one line per population.

    The stop-clock and arrival-clock rungs are drawn as two SEPARATE line
    segments either side of a labelled gap: they are fit to different rows, so a
    line joining them would assert a ladder step that does not exist.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    stop_x, arrival_x, boundary = _ladder_positions()
    metrics = [
        ("ks_within_band", "fraction KS within band", "mean"),
        ("ks_statistic", "median KS statistic", "median"),
        ("dispersion", "median dispersion (Poisson=1)", "median"),
        ("cv_bits_per_event", "median CV bits/event vs M0", "median"),
    ]
    for ax, (col, label, agg) in zip(axes.flat, metrics, strict=True):
        for i, oc in enumerate(
            ["swr", "non_local", "non_local_swr", "non_local_no_swr"]
        ):
            d = df[df.outcome == oc]
            for rungs, xs in ((STOP_LADDER, stop_x), (ARRIVAL_LADDER, arrival_x)):
                vals = [
                    getattr(d[d.model == m][col], agg)()
                    if (d.model == m).any()
                    else np.nan
                    for m in rungs
                ]
                ax.plot(
                    xs,
                    vals,
                    marker="o",
                    color=f"C{i}",
                    label=oc if rungs is STOP_LADDER else None,
                )
        ax.axvline(boundary, color="0.4", ls="--", lw=1.0)
        ax.set_xticks([*stop_x, *arrival_x])
        ax.set_xticklabels(
            [*STOP_LADDER, *ARRIVAL_LADDER], rotation=60, ha="right", fontsize=7
        )
        ax.set_ylabel(label)
        # Above the axes frame (x in data units, y in axes units) so the labels
        # cannot land on top of a line: the dispersion panel is nearly flat and
        # its data reach the top of the axes.
        for xpos, text in (
            (stop_x[0], "stop clock"),
            (
                arrival_x[0],
                "arrival clock (different rows — not a further ladder step)",
            ),
        ):
            ax.text(
                xpos,
                1.01,
                text,
                transform=ax.get_xaxis_transform(),
                fontsize=7,
                va="bottom",
                color="0.4",
                clip_on=False,
            )
        if col == "dispersion":
            ax.axhline(1, color="0.5", ls=":")
        if col == "cv_bits_per_event":
            ax.axhline(0, color="0.5", ls=":")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Point-process GOF up the per-bin ladder\n" + ARRIVAL_RETENTION_NOTE,
        y=1.06,
        fontsize=10,
    )
    fig.tight_layout()
    out = FIGS / "model_checking_ladder.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def _refit_session(
    stop: pd.DataFrame, onsets: pd.DataFrame, keys: tuple, outcome: str
) -> tuple[pd.DataFrame, dict | None]:
    """Rebuild one session's design and full-model fit for the diagnostic plots.

    Carries ``stop_start`` as an extra covariate (ignored by the fit) so the
    drift panel can order bins by true within-session time -- ``stop_id`` is a
    hash, not a temporal key.
    """
    mask = pd.Series(True, index=stop.index)
    for c, v in zip(SC, keys, strict=True):
        mask &= stop[c] == v
    stops_g = stop[mask]
    om = pd.Series(True, index=onsets.index)
    for c, v in zip(SC, keys, strict=True):
        om &= onsets[c] == v
    pop = POPULATIONS[outcome](onsets[om])
    # the bins the published fits used: excluded time and the bins whose
    # history window (the ladder's longest) reaches it are not analysed
    design = aggregate_to_timebins(
        stops_g,
        pop,
        bin_s=BIN_S,
        tmax=TMAX,
        reward_basis=True,
        self_history_lags=N_HISTORY_LAGS,
        self_history_basis=N_HISTORY_BASIS,
        stop_covariates=(*STOP_COVARIATES, "stop_start"),
        excluded_intervals=excluded_intervals_by_session(stops_g),
        history_window_s=longest_history_window_s(MODEL_PER_BIN_RATE.values()),
    )
    # Freeze the cohort before fitting (the fitter refuses an incomplete
    # design). This design is built for one rung, not the whole ladder, so the
    # freeze covers that rung's columns only.
    design, _ = freeze_per_bin_cohort(design, [MODEL_PER_BIN_RATE[FULL]])
    fit = fit_per_bin_rate_model(design, MODEL_PER_BIN_RATE[FULL])
    return design, fit


def isi_autocorrelation_window(
    autocorrelation: np.ndarray, lag_max: int
) -> tuple[np.ndarray, np.ndarray]:
    """Lags ``-(lag_max - 1) ... lag_max - 1`` of a rescaled-ISI autocorrelation.

    Parameters
    ----------
    autocorrelation : np.ndarray, shape (2 * n_spikes - 1,)
        ``TimeRescaling.rescaled_ISI_autocorrelation()``; lag 0 is at index
        ``n_spikes - 1``.
    lag_max : int
        The largest lag shown, exclusive. A value above ``n_spikes`` keeps every
        lag.

    Returns
    -------
    lags : np.ndarray, shape (2 * min(lag_max, n_spikes) - 1,)
    values : np.ndarray, shape (2 * min(lag_max, n_spikes) - 1,)
    """
    n_spikes = autocorrelation.size // 2 + 1
    half = min(lag_max, n_spikes) - 1
    lags = np.arange(-half, half + 1)
    return lags, autocorrelation[n_spikes - 1 + lags]


def plot_rescaled_isi_autocorrelation(
    ax: plt.Axes, tr: TimeRescaling, lag_max: int
) -> None:
    """Scatter the rescaled-ISI autocorrelation with its +/- 1.96 / sqrt(n) lines.

    The values, the bounds and the labels are the ones non_local_detector's own
    ``plot_rescaled_ISI_autocorrelation`` draws; only the lag selection differs.
    """
    autocorrelation = tr.rescaled_ISI_autocorrelation()
    lags, values = isi_autocorrelation_window(autocorrelation, lag_max)
    n_spikes = autocorrelation.size // 2 + 1
    ci = 1.96 / np.sqrt(n_spikes)
    ax.scatter(lags, values)
    ax.axhline(ci, linestyle="--", color="red")
    ax.axhline(-ci, linestyle="--", color="red")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")


def plot_example_quartet(
    design: pd.DataFrame, fit: dict, title: str, fname: str
) -> None:
    """KS / Q-Q / rescaled-ISI autocorr / cumulative-residual diagnostic quartet."""
    predicted = fit["predicted"]
    # Order by true within-session time (stop_start, then bin): stop_id is a
    # hash, so sorting by it would scramble the cross-stop residual drift. Each
    # stop_start is unique, so same-stop bins stay contiguous for TimeRescaling.
    sub = design.loc[predicted.index].sort_values(["stop_start", "binj"])
    ci = predicted.reindex(sub.index).to_numpy(dtype=float)
    is_spike = sub["count"].to_numpy() > 0
    trial_id = sub["stop_id"].to_numpy()
    tr = TimeRescaling(ci, is_spike, trial_id, adjust_for_short_trials=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    tr.plot_ks(ax=axes[0, 0])
    axes[0, 0].set_title(f"KS (stat={tr.ks_statistic():.3f}, n={int(is_spike.sum())})")
    tr.plot_qq(ax=axes[0, 1])
    axes[0, 1].set_title("Q-Q rescaled ISIs")
    # Drawn here, not by ``tr.plot_rescaled_ISI_autocorrelation``: the library's
    # lag slice is empty (LorenFrankLab/non_local_detector#64). Return to the
    # library call once #64 is fixed.
    plot_rescaled_isi_autocorrelation(axes[1, 0], tr, lag_max=50)
    axes[1, 0].set_title("Rescaled-ISI autocorrelation (|lag| < 50)")

    # Cumulative point-process residual vs within-session time (stop order):
    # systematic drift signals structure the model misses (e.g. slow session
    # drift -> the state-space term). point_process_residuals already returns
    # the cumulative (cumsum) martingale residual.
    resid = point_process_residuals(ci, is_spike)
    axes[1, 1].plot(np.arange(resid.size), resid, color="C3")
    axes[1, 1].axhline(0, color="0.5", ls=":")
    axes[1, 1].set_xlabel("bin index (within-session time order)")
    axes[1, 1].set_ylabel("cumulative point-process residual")
    axes[1, 1].set_title("Cumulative residual vs time (drift check)")

    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    out = FIGS / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def _print_ladder_table(df: pd.DataFrame) -> None:
    print("\n=== GOF up the ladder (outcome=non_local) ===")
    g = df[df.outcome == "non_local"]
    for header, rungs in (
        ("stop clock", STOP_LADDER),
        (
            "arrival clock (fit on the arrival window — not comparable with "
            "the rows above)",
            ARRIVAL_LADDER,
        ),
    ):
        print(f"-- {header}")
        print(
            f"{'model':30s} {'ks_band':>8s} {'ks_med':>8s} "
            f"{'disp_med':>9s} {'bits_med':>9s}"
        )
        for m in rungs:
            s = g[g.model == m]
            if len(s):
                print(
                    f"{m:30s} {s.ks_within_band.mean():8.2f} "
                    f"{s.ks_statistic.median():8.3f} "
                    f"{s.dispersion.median():9.3f} "
                    f"{s.cv_bits_per_event.median():9.3f}"
                )


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
    FIGS.mkdir(parents=True, exist_ok=True)

    plot_gof_ladder(df)
    _print_ladder_table(df)

    # --- GOF sanity: assert only what the data actually supports ---
    g = df[df.outcome == "non_local"]
    ks0 = g[g.model == "M0_const"].ks_within_band.mean()
    ksf = g[g.model == FULL].ks_within_band.mean()
    assert ks0 < ksf, f"expected M0 KS-band frac < {FULL} ({ks0:.2f} !< {ksf:.2f})"
    print(
        f"\nKS calibration improves up the ladder: M0_const={ks0:.2f} -> {FULL}={ksf:.2f}"
    )

    # Honest reporting of where the textbook narrative does NOT hold:
    bits = {
        m: g[g.model == m].cv_bits_per_event.median()
        for m in (*STOP_LADDER, *ARRIVAL_LADDER)
    }
    print(
        "CV bits/event: time block helps (M5_time={:.3f}); the raised-cosine "
        "self-history block costs held-out likelihood (M3r_histcos={:.3f}) -- "
        "in-sample KS improves but cv_bits drops. Dispersion stays ~{:.2f} "
        "(underdispersed at sparse 15 ms bins), so the dispersion->1 "
        "expectation does not apply here.".format(
            bits["M5_time"], bits["M3r_histcos"], g[g.model == FULL].dispersion.median()
        )
    )

    # --- per-event-type model-checking face of the dissociation ---
    print(f"\n=== fit quality by population (model={FULL}) ===")
    for oc in ["swr", "non_local_swr", "non_local_no_swr"]:
        s = df[(df.outcome == oc) & (df.model == FULL)]
        print(
            f"  {oc:18s}: ks_band={s.ks_within_band.mean():.2f} "
            f"bits/event_med={s.cv_bits_per_event.median():.3f} n={len(s)}"
        )

    # --- example-session diagnostic quartets (inline refit) ---
    stop = read_table(RC / "stop_table.pkl")
    onset_table = read_table(RC / "event_onset_table.pkl")
    event_table = read_table(RC / "event_table.pkl")
    require_covariate_schema(stop, table="stop")
    require_covariate_schema(onset_table, table="event_onset")
    require_covariate_schema(event_table, table="event")
    require_swr_source(stop, what=str(ALL_MODELS_INPUTS["stop_table"]))
    require_swr_source(onset_table, what=str(ALL_MODELS_INPUTS["event_onset_table"]))
    require_swr_source(event_table, what=str(ALL_MODELS_INPUTS["event_table"]))
    onsets = attach_is_swr_to_onsets(onset_table, event_table)
    full_nl = df[(df.outcome == "non_local") & (df.model == FULL)].dropna(
        subset=["n_events"]
    )
    if full_nl.empty:
        print("No fitted non_local M7 sessions; skipping example quartets.")
        return
    ordered = full_nl.sort_values("n_events")
    examples = {"rich": tuple(ordered.iloc[-1][SC])}
    # ~17th percentile as a genuinely sparser (but still fitted) example; fall
    # back to a distinct session if the run is tiny.
    sparse_pos = min(len(ordered) // 6, len(ordered) - 2) if len(ordered) > 1 else 0
    sparse_keys = tuple(ordered.iloc[sparse_pos][SC])
    if sparse_keys != examples["rich"]:
        examples["sparse"] = sparse_keys
    for tag, keys in examples.items():
        design, fit = _refit_session(stop, onsets, keys, "non_local")
        if fit is None:
            print(f"  ({tag} example {keys} did not fit; skipped)")
            continue
        label = "_".join(map(str, keys))
        plot_example_quartet(
            design,
            fit,
            f"{FULL} GOF — non_local — {label} ({tag})",
            f"model_checking_example_{tag}.png",
        )


if __name__ == "__main__":
    main()
