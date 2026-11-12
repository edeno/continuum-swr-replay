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
    uv run python scripts/behavior/rate/plot_model_checking.py
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
from non_local_detector.model_checking.sorted_spikes import (  # noqa: E402
    TimeRescaling,
    point_process_residuals,
)

from continuum_swr_replay.analysis.behavior import (  # noqa: E402
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
from continuum_swr_replay.analysis.behavior.fit_provenance import (  # noqa: E402
    require_fresh_inputs,
)
from continuum_swr_replay.analysis.behavior.model_specs import (  # noqa: E402
    longest_history_window_s,
)
from continuum_swr_replay.analysis.behavior.rate.duration_aware_history import (  # noqa: E402
    at_risk_events,
    attach_event_ends,
)
from continuum_swr_replay.analysis.behavior.rate.rate_fit_statistics import (  # noqa: E402
    arrival_window_statistics,
    ladder_fit_statistics,
)
from continuum_swr_replay.analysis.events.analysed_time import (  # noqa: E402
    require_swr_source,
)
from continuum_swr_replay.paths import (
    figure_dir,
    model_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.cli import fit_figure_parser  # noqa: E402
from continuum_swr_replay.tables.covariate_registry import (  # noqa: E402
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS
from continuum_swr_replay.visualization.figure_utilities import (
    save_figure,  # noqa: E402
)


def all_models_path() -> Path:
    """The per-bin ladder's fitted bundle, ``all_models.pkl``."""
    return model_dir("per_bin_rate") / "all_models.pkl"


def all_models_input_paths() -> dict[str, Path]:
    """The tables fit_per_bin_rate_models.py fit all_models.pkl on; checked against
    the artifact's provenance sidecar before the fitted GOF columns are read. Under
    ``paths.replay_content_dir()``, read when called.
    """
    rc = replay_content_dir()
    return {
        "stop_table": table_path("stop_table", rc),
        "event_onset_table": table_path("event_onset_table", rc),
        "event_table": table_path("event_table", rc),
    }


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
# subset: it need not keep the same share of each population's event bins, so a
# BETWEEN-population arrival comparison is confounded by retention. The order
# differs between fits (the in-transit cohort keeps more non-SWR bins).
# The shares are the runner's own tallies, ``fits.attrs["arrival_clock_cohort"]``:
# per population, the event bins (bins with ``count > 0``) of the frozen cohort
# that the window kept. ``per_bin_rate_cohort_audit.csv`` cannot give them: it
# counts stops. A fit without the tallies gets the warning in words only.
_ARRIVAL_RETENTION_TEMPLATE = (
    "arrival rungs are read on a different row set — the arrival window keeps "
    "{shares}, so between-population arrival differences are partly retention"
)
ARRIVAL_RETENTION_NOTE = (
    "arrival rungs are read on a different row set — the arrival window can keep "
    "different shares of SWR and of non-SWR non-local event bins, so "
    "between-population arrival differences can be partly retention"
)


def arrival_retention_note(fits: pd.DataFrame) -> str:
    """The arrival-window retention warning, with the SWR and non-SWR shares.

    Parameters
    ----------
    fits : pd.DataFrame
        ``all_models.pkl``; ``attrs["arrival_clock_cohort"]`` holds, per
        population, the arrival-clock tallies
        ``rate_fit_statistics.arrival_window_statistics`` reads.

    Returns
    -------
    str
        "... keeps 95 % of SWR event bins vs 88 % of non-SWR non-local event
        bins ...", the
        shares of the ``swr`` and ``non_local_no_swr`` populations in whole
        percent; the words-only :data:`ARRIVAL_RETENTION_NOTE` when the fit
        carries no tallies.
    """
    cohort = fits.attrs.get("arrival_clock_cohort")
    if not cohort:
        return ARRIVAL_RETENTION_NOTE
    share = {
        population: 100 * arrival_window_statistics(cohort[population])["event_share"]
        for population in ("swr", "non_local_no_swr")
    }
    return _ARRIVAL_RETENTION_TEMPLATE.format(
        shares=(
            f"{share['swr']:.0f} % of SWR event bins vs "
            f"{share['non_local_no_swr']:.0f} % of non-SWR non-local event bins"
        )
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
    line joining them would assert a ladder step that does not exist. Each
    point is ``rate_fit_statistics.ladder_fit_statistics``' summary of the
    rung's fits.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    stop_x, arrival_x, boundary = _ladder_positions()
    ladder = ladder_fit_statistics(df)
    metrics = [
        ("ks_band_fraction", "fraction KS within band"),
        ("ks_statistic_median", "median KS statistic"),
        ("dispersion_median", "median dispersion (Poisson=1)"),
        ("cv_bits_median", "median CV bits/event vs M0 (finite scores)"),
    ]
    for ax, (col, label) in zip(axes.flat, metrics, strict=True):
        for i, oc in enumerate(
            ["swr", "non_local", "non_local_swr", "non_local_no_swr"]
        ):
            for rungs, xs in ((STOP_LADDER, stop_x), (ARRIVAL_LADDER, arrival_x)):
                vals = [
                    ladder.loc[(m, oc), col] if (m, oc) in ladder.index else np.nan
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
        if col == "dispersion_median":
            ax.axhline(1, color="0.5", ls=":")
        if col == "cv_bits_median":
            ax.axhline(0, color="0.5", ls=":")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Point-process GOF up the per-bin ladder\n" + arrival_retention_note(df),
        y=1.06,
        fontsize=10,
    )
    fig.tight_layout()
    out = figure_dir("rate") / "model_checking_ladder.png"
    save_figure(
        fig, out.stem, base_dir=out.parent, dpi=150, formats=["png"], transparent=False
    )
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
    # history window (the ladder's longest) reaches it are not analysed, nor
    # the bins inside an ongoing event (the at-risk design)
    design = aggregate_to_timebins(
        stops_g,
        pop,
        bin_s=BIN_S,
        tmax=TMAX,
        reward_basis=True,
        self_history_lags=N_HISTORY_LAGS,
        self_history_basis=N_HISTORY_BASIS,
        stop_covariates=(*STOP_COVARIATES, "stop_start"),
        history_window_s=longest_history_window_s(MODEL_PER_BIN_RATE.values()),
        at_risk_events=at_risk_events(onsets[om], outcome),
        offset_history=True,
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
    out = figure_dir("rate") / fname
    save_figure(
        fig, out.stem, base_dir=out.parent, dpi=150, formats=["png"], transparent=False
    )
    plt.close(fig)
    print(f"saved {out}")


def _print_ladder_table(ladder: pd.DataFrame) -> None:
    print("\n=== GOF up the ladder (outcome=non_local) ===")
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
            if (m, "non_local") in ladder.index:
                row = ladder.loc[(m, "non_local")]
                print(
                    f"{m:30s} {row.ks_band_fraction:8.2f} "
                    f"{row.ks_statistic_median:8.3f} "
                    f"{row.dispersion_median:9.3f} "
                    f"{row.cv_bits_median:9.3f}"
                )


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser (also the provenance gate's opt-out)."""
    return fit_figure_parser(__doc__)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    all_models, inputs = all_models_path(), all_models_input_paths()
    require_fresh_inputs(all_models, inputs, allow_stale=args.allow_stale)
    df = pd.read_pickle(all_models)
    figure_dir("rate").mkdir(parents=True, exist_ok=True)

    plot_gof_ladder(df)
    ladder = ladder_fit_statistics(df)
    _print_ladder_table(ladder)

    # --- GOF sanity: assert only what the data actually supports ---
    ks0 = ladder.loc[("M0_const", "non_local"), "ks_band_fraction"]
    ksf = ladder.loc[(FULL, "non_local"), "ks_band_fraction"]
    assert ks0 < ksf, f"expected M0 KS-band frac < {FULL} ({ks0:.2f} !< {ksf:.2f})"
    print(
        f"\nKS calibration improves up the ladder: M0_const={ks0:.2f} -> {FULL}={ksf:.2f}"
    )

    # Honest reporting of where the textbook narrative does NOT hold:
    bits = ladder.xs("non_local", level="outcome")["cv_bits_median"]
    print(
        "CV bits/event: time block helps (M5_time={:.3f}); the raised-cosine "
        "self-history block costs held-out likelihood (M3r_histcos={:.3f}) -- "
        "in-sample KS improves but cv_bits drops. Dispersion stays ~{:.2f} "
        "(underdispersed at sparse 15 ms bins), so the dispersion->1 "
        "expectation does not apply here.".format(
            bits["M5_time"],
            bits["M3r_histcos"],
            ladder.loc[(FULL, "non_local"), "dispersion_median"],
        )
    )

    # --- per-event-type model-checking face of the dissociation ---
    print(f"\n=== fit quality by population (model={FULL}) ===")
    for oc in ["swr", "non_local_swr", "non_local_no_swr"]:
        row = ladder.loc[(FULL, oc)]
        print(
            f"  {oc:18s}: ks_band={row.ks_band_fraction:.2f} "
            f"bits/event_med={row.cv_bits_median:.3f} n={int(row.n_fitted)}"
        )

    # --- example-session diagnostic quartets (inline refit) ---
    stop = read_table(inputs["stop_table"])
    onset_table = read_table(inputs["event_onset_table"])
    event_table = read_table(inputs["event_table"])
    require_covariate_schema(stop, table="stop")
    require_covariate_schema(onset_table, table="event_onset")
    require_covariate_schema(event_table, table="event")
    require_swr_source(stop, what=str(inputs["stop_table"]))
    require_swr_source(onset_table, what=str(inputs["event_onset_table"]))
    require_swr_source(event_table, what=str(inputs["event_table"]))
    onsets = attach_event_ends(
        attach_is_swr_to_onsets(onset_table, event_table), event_table
    )
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
