"""Driver for the 2026-10-26 duration-aware history review. Simulation only.

Fits the shipped cross-history design and its duration-aware twin to simulated
sessions (``analysis/replay/duration_history_simulation.py``) and writes
``results.json`` and the figures next to this file. Every number in the
review's report comes from ``results.json``.

Three parts:

1. pooled: one long simulated session per scenario (bias / recovery);
2. sessions: many session-sized simulations per scenario, summarised as
   ``scripts/rate/fit_cross_history_rate.py`` summarises real sessions;
3. identifiability: the fragmentation and onset-rhythm scenarios with the
   duration spread shrunk to zero (constant durations).

Usage::

    uv run python .claude/docs/reviews/2026-10-26-duration-aware-history-simulation/run_simulation.py
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from continuum_swr_replay.analysis.replay.design_constants import (  # noqa: E402
    BIN_S,
    N_HISTORY_LAGS,
)
from continuum_swr_replay.analysis.replay.duration_history_simulation import (  # noqa: E402
    HISTORY_BLOCKS,
    SCENARIOS,
    STREAMS,
    fit_history_direction,
    simulate_interval_streams,
    summarize_direction,
    true_lag_kernel,
)

OUT = Path(__file__).resolve().parent
LAG_MS = np.arange(1, N_HISTORY_LAGS + 1) * BIN_S * 1000
#: A lag's kernel value is reported only with this many expected onsets.
MIN_SUPPORT = 10.0
SIGMAS = (0.0, 0.1, 0.2, 0.45)


def _kernel_record(result: dict, rung: str, scenario) -> dict:
    """The rung's fitted kernels with truth and support, per block."""
    out = {}
    for block, kernel in result["kernels"][rung].items():
        _, anchor, rel = HISTORY_BLOCKS[block]
        source = result["response"] if rel == "self" else result["cross"]
        truth = true_lag_kernel(
            scenario, target=result["response"], source=source, anchor=anchor
        )
        support = result["support"][block]
        supported = support >= MIN_SUPPORT
        fitted = np.where(supported, kernel, np.nan)
        peak = int(np.nanargmax(fitted)) + 1 if supported.any() else None
        out[block] = {
            "log_rr": kernel.round(4).tolist(),
            "truth_log_rr": truth.round(4).tolist(),
            "support": support.round(2).tolist(),
            "peak_lag": peak,
            "peak_rr": float(np.exp(fitted[peak - 1])) if peak else None,
            "min_supported_log_rr": float(np.nanmin(fitted)) if peak else None,
            "separated": any(
                t.startswith(HISTORY_BLOCKS[block][0])
                for t in result["fits"][rung]["separated_terms"].split("|")
                if t
            ),
        }
    return out


def _direction_record(result: dict, scenario) -> dict:
    rec = {
        **summarize_direction(result),
        "n_bins": result["n_bins"],
        "n_events": result["n_events"],
        "aic": {k: round(v, 2) for k, v in result["aic"].items()},
        "collinearity": result["collinearity"],
        "design_attrs": {
            k: v for k, v in result["design_attrs"].items() if "risk" in k
        },
        "kernels_best": _kernel_record(result, result["best"], scenario),
    }
    full = "self+cross" if not result["duration_aware"] else "self_both+cross_both"
    if full in result["kernels"]:
        rec["kernels_full"] = _kernel_record(result, full, scenario)
    if not result["duration_aware"]:
        fit = result["fits"]["self+cross"]
        rec["coef_cross_hist_rcos_0"] = fit["coef_cross_hist_rcos_0"]
        rec["coef_hist_rcos_0"] = fit["coef_hist_rcos_0"]
        rec["separated_terms"] = fit["separated_terms"]
    return rec


def _fit_session(scenario, n_stops: int, seed: int) -> dict:
    stops, events = simulate_interval_streams(scenario, n_stops=n_stops, seed=seed)
    rec = {
        "n_stops": n_stops,
        "seed": seed,
        "n_events": events["stream"].value_counts().to_dict(),
        "median_duration_s": events.groupby("stream")["duration_s"]
        .median()
        .round(4)
        .to_dict(),
        "directions": {},
    }
    for response in STREAMS:
        for design in ("current", "duration_aware"):
            result = fit_history_direction(
                stops, events, response=response, duration_aware=design != "current"
            )
            rec["directions"][f"{response}/{design}"] = _direction_record(
                result, scenario
            )
    return rec


def _session_summary(sessions: list[dict]) -> dict:
    out = {}
    for key in sessions[0]["directions"]:
        rows = [s["directions"][key] for s in sessions]
        d_cross = np.array([r["delta_aic_cross"] for r in rows])
        d_self = np.array([r["delta_aic_self"] for r in rows])
        summ = {
            "n_sessions": len(rows),
            "delta_aic_cross_median": float(np.median(d_cross)),
            "delta_aic_cross_pct_positive": float(100 * np.mean(d_cross > 0)),
            "delta_aic_self_median": float(np.median(d_self)),
            "delta_aic_self_pct_positive": float(100 * np.mean(d_self > 0)),
            "best": {
                b: sum(r["best"] == b for r in rows) for b in {r["best"] for r in rows}
            },
        }
        if key.endswith("/current"):
            c0 = np.array([r["coef_cross_hist_rcos_0"] for r in rows])
            summ["coef_cross_hist_rcos_0_median"] = float(np.median(c0))
            summ["coef_cross_hist_rcos_0_pct_negative"] = float(100 * np.mean(c0 < 0))
            summ["pct_cross_rcos_0_separated"] = float(
                100
                * np.mean(["cross_hist_rcos_0" in r["separated_terms"] for r in rows])
            )
            # As the real summary reads it (``self_history_rate_ratios``): the
            # across-session median rate ratio per lag, unmasked.
            for block in ("self_onset", "cross_onset"):
                rr = np.median(
                    np.exp([r["kernels_full"][block]["log_rr"] for r in rows]), axis=0
                )
                summ[f"{block}_median_rr_by_lag"] = rr.round(3).tolist()
                summ[f"{block}_median_rr_peak_lag_ms"] = float(
                    (np.argmax(rr) + 1) * BIN_S * 1000
                )
                summ[f"{block}_median_rr_peak"] = float(rr.max())
        else:
            summ["self_offset_vs_onset_median"] = float(
                np.median([r["self_offset_vs_onset"] for r in rows])
            )
            summ["cross_offset_vs_onset_median"] = float(
                np.median([r["cross_offset_vs_onset"] for r in rows])
            )
        out[key] = summ
    return out


def _plot_kernels(pooled: dict, response: str) -> Path:
    names = list(pooled)
    fig, axes = plt.subplots(
        len(names), 2, figsize=(8.5, 2.3 * len(names)), sharex=True, squeeze=False
    )
    colors = {"onset": "#0072B2", "offset": "#D55E00"}
    for row, name in enumerate(names):
        for col, design in enumerate(("current", "duration_aware")):
            ax = axes[row, col]
            rec = pooled[name]["directions"][f"{response}/{design}"]
            kernels = (
                rec["kernels_full"] if design == "current" else rec["kernels_best"]
            )
            for block, k in kernels.items():
                _, anchor, rel = HISTORY_BLOCKS[block]
                fit = np.array(k["log_rr"])
                ok = np.array(k["support"]) >= MIN_SUPPORT
                style = "-" if rel == "self" else "--"
                ax.plot(
                    LAG_MS[ok],
                    np.exp(np.clip(fit[ok], -3, 3)),
                    style,
                    marker="o",
                    ms=3,
                    color=colors[anchor],
                    label=f"{rel} {anchor}",
                )
                truth = np.exp(np.array(k["truth_log_rr"]))
                if np.any(truth != 1):
                    ax.plot(
                        LAG_MS,
                        truth,
                        style,
                        color="0.5",
                        lw=1,
                        label=f"truth {rel} {anchor}",
                    )
            ax.axhline(1, color="0.8", lw=0.8)
            ax.set_yscale("log")
            ax.set_ylim(0.05, 20)
            label = (
                "current design"
                if design == "current"
                else f"duration-aware ({rec['best']})"
            )
            ax.set_title(f"{name}: {label}", fontsize=8)
            ax.legend(fontsize=6, frameon=False, loc="upper right")
            if col == 0:
                ax.set_ylabel("rate ratio")
    for ax in axes[-1]:
        ax.set_xlabel("lag (ms; onset- or offset-locked as labelled)")
    fig.suptitle(
        f"Response stream {response}: fitted history kernels (lags with "
        f"< {MIN_SUPPORT:g} expected onsets not drawn; grey = truth)",
        fontsize=9,
    )
    fig.tight_layout()
    path = OUT / f"kernels_{response}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_identifiability(ident: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for name, rows in ident.items():
        sig = [r["sigma"] for r in rows]
        axes[0].plot(sig, [r["self_offset_vs_onset"] for r in rows], "o-", label=name)
        axes[1].plot(sig, [r["canonical_corr_self"] for r in rows], "o-", label=name)
    axes[0].axhline(0, color="0.7", lw=0.8)
    axes[0].set_ylabel("AIC(onset-only) - AIC(offset-only)")
    axes[1].set_ylabel("canonical corr., self onset vs offset block")
    for ax in axes:
        ax.set_xlabel("duration log-SD (both streams)")
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path = OUT / "identifiability.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pooled-stops", type=int, default=6000)
    p.add_argument("--sessions", type=int, default=10)
    p.add_argument("--session-stops", type=int, default=350)
    p.add_argument("--ident-stops", type=int, default=4000)
    args = p.parse_args()
    warnings.filterwarnings("ignore")
    t0 = time.time()
    results: dict = {"args": vars(args), "min_support": MIN_SUPPORT}
    results["scenarios"] = {
        name: {
            "description": sc.description,
            "base_rate_hz": sc.base_rate_hz,
            "duration_median_s": sc.duration_median_s,
            "duration_sigma": sc.duration_sigma,
            "kernels": [
                {
                    "target": k.target,
                    "source": k.source,
                    "anchor": k.anchor,
                    "rate_ratio_peak": float(np.exp(k.log_gain)),
                    "center_ms": k.center_s * 1000,
                    "width_ms": k.width_s * 1000,
                }
                for k in sc.kernels
            ],
        }
        for name, sc in SCENARIOS.items()
    }

    results["pooled"] = {}
    for i, (name, sc) in enumerate(SCENARIOS.items()):
        results["pooled"][name] = _fit_session(sc, args.pooled_stops, seed=100 + i)
        print(f"pooled {name} done ({time.time() - t0:.0f}s)", flush=True)

    results["sessions"] = {}
    for i, (name, sc) in enumerate(SCENARIOS.items()):
        sessions = [
            _fit_session(sc, args.session_stops, seed=1000 + 100 * i + s)
            for s in range(args.sessions)
        ]
        results["sessions"][name] = _session_summary(sessions)
        print(f"sessions {name} done ({time.time() - t0:.0f}s)", flush=True)

    results["identifiability"] = {}
    for name in ("b_fragmentation", "c_onset_rhythm"):
        rows = []
        for sigma in SIGMAS:
            sc = SCENARIOS[name].with_durations(sigma)
            stops, events = simulate_interval_streams(
                sc, n_stops=args.ident_stops, seed=2000
            )
            r = fit_history_direction(
                stops, events, response="no_swr", duration_aware=True
            )
            s = summarize_direction(r)
            rows.append(
                {
                    "sigma": sigma,
                    **{k: s[k] for k in ("self_offset_vs_onset", "best_self", "best")},
                    "delta_aic_self": s["delta_aic_self"],
                    **r["collinearity"],
                    "kernels_best": _kernel_record(r, r["best"], sc),
                }
            )
            print(
                f"ident {name} sigma={sigma} done ({time.time() - t0:.0f}s)", flush=True
            )
        results["identifiability"][name] = rows

    results["figures"] = [_plot_kernels(results["pooled"], r).name for r in STREAMS] + [
        _plot_identifiability(results["identifiability"]).name
    ]
    results["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"wrote results.json ({results['runtime_s']} s)")


if __name__ == "__main__":
    main()
