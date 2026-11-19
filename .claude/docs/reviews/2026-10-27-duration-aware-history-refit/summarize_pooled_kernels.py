"""Summarize the per-animal pooled kernels (``pooled_kernels_at_well.json``).

A lag of a kernel is *measurable* when the animal's pooled support (expected
onsets at that lag, ``lag_support``) is >= 3 and the kernel's Wald SE on the
log scale is <= 0.5; unmeasurable lags are neither summarized nor drawn.

Per population, over the five animals:

- pooled AIC contrasts with session fixed effects (the same contrasts as
  ``summarize_session_contrasts.py``), with the per-animal LR p for nested ones
  and animals positive / significant of 5;
- for each kernel (shipped ``Mself`` onset kernel; duration-aware onset-only,
  offset-only, and both-anchor kernels; cross kernels for the non-local
  streams): per animal the peak rate ratio and its lag over measurable lags,
  the smallest measurable rate ratio, and whether the peak's 95 % CI excludes
  1; across animals the median peak and the count with CI > 1. The shipped
  kernel's first three lags are listed (the "refractory" read).

Writes ``pooled_kernels_summary.json``, ``pooled_kernels_self.png`` and
``pooled_kernels_cross.png``.

Usage::

    uv run python .claude/docs/reviews/2026-10-27-duration-aware-history-refit/summarize_pooled_kernels.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import chi2  # noqa: E402

OUT = Path(__file__).resolve().parent
MIN_SUPPORT = 3.0
MAX_SE = 0.5
LAG_MS = np.arange(1, 11) * 15
CONTRASTS = {
    "self_both": ("Mnohist_dur", "Mself_dur", True),
    "self_onset_only": ("Mnohist_dur", "Mself_onset_dur", True),
    "self_offset_only": ("Mnohist_dur", "Mself_offset_dur", True),
    "self_offset_adds_to_onset": ("Mself_onset_dur", "Mself_dur", True),
    "self_onset_adds_to_offset": ("Mself_offset_dur", "Mself_dur", True),
    "self_offset_vs_onset": ("Mself_onset_dur", "Mself_offset_dur", False),
    "cross_both": ("Mself_dur", "Mselfcross_dur", True),
    "cross_onset_only": ("Mself_dur", "Mselfcross_onset_dur", True),
    "cross_offset_only": ("Mself_dur", "Mselfcross_offset_dur", True),
    "cross_offset_vs_onset": ("Mselfcross_onset_dur", "Mselfcross_offset_dur", False),
}
#: label -> (design, rung, block)
KERNELS = {
    "shipped_self_onset": ("shipped", "Mself", "self_onset"),
    "dur_self_onset_only": ("duration_aware", "Mself_onset_dur", "self_onset"),
    "dur_self_offset_only": ("duration_aware", "Mself_offset_dur", "self_offset"),
    "dur_both_self_onset": ("duration_aware", "Mself_dur", "self_onset"),
    "dur_both_self_offset": ("duration_aware", "Mself_dur", "self_offset"),
    "shipped_cross_onset": ("shipped", "Mselfcross", "cross_onset"),
    "dur_cross_onset_only": ("duration_aware", "Mselfcross_onset_dur", "cross_onset"),
    "dur_cross_offset_only": (
        "duration_aware",
        "Mselfcross_offset_dur",
        "cross_offset",
    ),
    "dur_both_cross_onset": ("duration_aware", "Mselfcross_dur", "cross_onset"),
    "dur_both_cross_offset": ("duration_aware", "Mselfcross_dur", "cross_offset"),
}


def _kernel(rec: dict, design: str, rung: str, block: str) -> dict | None:
    fit = rec[design].get(rung)
    if fit is None or not fit["ok"] or block not in fit["kernels"]:
        return None
    k = fit["kernels"][block]
    support = rec["shipped_support" if design == "shipped" else "support"][block]
    log_rr, se = np.array(k["log_rr"]), np.array(k["se"])
    ok = (np.array(support) >= MIN_SUPPORT) & (se <= MAX_SE)
    out = {
        "log_rr": log_rr.round(4).tolist(),
        "se": se.round(4).tolist(),
        "measurable": ok.tolist(),
        "support": np.round(support, 2).tolist(),
        "rr_first_three_lags": np.exp(log_rr[:3]).round(3).tolist(),
    }
    if ok.any():
        masked = np.where(ok, log_rr, np.nan)
        j = int(np.nanargmax(masked))
        out |= {
            "peak_rr": float(np.exp(log_rr[j])),
            "peak_lag_ms": int(LAG_MS[j]),
            "peak_ci_low": float(np.exp(log_rr[j] - 1.96 * se[j])),
            "peak_ci_excludes_1": bool(log_rr[j] - 1.96 * se[j] > 0),
            "min_measurable_rr": float(np.exp(np.nanmin(masked))),
            "first_measurable_lag_ms": int(LAG_MS[np.argmax(ok)]),
        }
    return out


def main() -> None:
    data = json.loads((OUT / "pooled_kernels_at_well.json").read_text())
    by_pop: dict[str, dict] = {}
    for rec in data["results"]:
        by_pop.setdefault(rec["outcome"], {})[rec["animal"]] = rec
    summary: dict = {
        "cohort": data["cohort"],
        "measurable_rule": f"support >= {MIN_SUPPORT} and SE <= {MAX_SE}",
        "populations": {},
    }
    for pop, animals in sorted(by_pop.items()):
        ps: dict = {
            "n_animals": len(animals),
            "contrasts": {},
            "kernels": {},
            "failed_fits": {},
        }
        for name, (simple, rich, nested) in CONTRASTS.items():
            per = {}
            for a, rec in animals.items():
                d = rec["duration_aware"]
                if simple in d and rich in d and d[simple]["ok"] and d[rich]["ok"]:
                    v = {"d_aic": d[simple]["aic"] - d[rich]["aic"]}
                    if nested:
                        dk = d[rich]["n_params"] - d[simple]["n_params"]
                        v["p"] = float(chi2.sf(v["d_aic"] + 2 * dk, dk))
                    per[a] = v
            if per:
                ps["contrasts"][name] = {
                    "animals": per,
                    "median_d_aic": float(
                        np.median([v["d_aic"] for v in per.values()])
                    ),
                    "animals_positive": int(sum(v["d_aic"] > 0 for v in per.values())),
                    **(
                        {
                            "animals_significant": int(
                                sum(v["p"] < 0.05 for v in per.values())
                            )
                        }
                        if nested
                        else {}
                    ),
                }
        for label, (design, rung, block) in KERNELS.items():
            per = {a: _kernel(rec, design, rung, block) for a, rec in animals.items()}
            per = {a: k for a, k in per.items() if k is not None}
            if not per:
                continue
            peaks = [k for k in per.values() if "peak_rr" in k]
            ps["kernels"][label] = {
                "animals": per,
                "n_animals_measurable": len(peaks),
                "median_peak_rr": float(np.median([k["peak_rr"] for k in peaks]))
                if peaks
                else None,
                "median_peak_lag_ms": float(
                    np.median([k["peak_lag_ms"] for k in peaks])
                )
                if peaks
                else None,
                "animals_peak_ci_above_1": int(
                    sum(k["peak_ci_excludes_1"] for k in peaks)
                ),
                "median_min_measurable_rr": float(
                    np.median([k["min_measurable_rr"] for k in peaks])
                )
                if peaks
                else None,
                "median_rr_first_three_lags": np.median(
                    [k["rr_first_three_lags"] for k in per.values()], axis=0
                )
                .round(3)
                .tolist(),
            }
        ps["failed_fits"] = {
            f"{design}/{rung}": sum(
                not rec[design][rung]["ok"]
                for rec in animals.values()
                if rung in rec[design]
            )
            for design in ("duration_aware", "shipped")
            for rung in next(iter(animals.values()))[design]
        }
        summary["populations"][pop] = ps
    (OUT / "pooled_kernels_summary.json").write_text(json.dumps(summary, indent=1))

    def _panel(ax, pop, label, title):
        for a, k in (
            summary["populations"][pop]["kernels"]
            .get(label, {})
            .get("animals", {})
            .items()
        ):
            ok = np.array(k["measurable"])
            lr, se = np.array(k["log_rr"]), np.array(k["se"])
            if not ok.any():
                continue
            x = LAG_MS[ok]
            ax.plot(x, np.exp(lr[ok]), "o-", ms=3, lw=1, label=a)
            ax.fill_between(
                x,
                np.exp(lr[ok] - 1.96 * se[ok]),
                np.exp(lr[ok] + 1.96 * se[ok]),
                alpha=0.12,
            )
        ax.axhline(1, color="0.7", lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(0.05, 20)
        ax.set_xlim(0, 160)
        ax.set_title(title, fontsize=7)

    pops = ["swr", "non_local", "non_local_swr", "non_local_no_swr"]
    cols = [
        ("shipped_self_onset", "shipped design: onset"),
        ("dur_self_onset_only", "at-risk: onset only"),
        ("dur_self_offset_only", "at-risk: end only"),
        ("dur_both_self_onset", "at-risk both: onset"),
        ("dur_both_self_offset", "at-risk both: end"),
    ]
    fig, axes = plt.subplots(
        len(pops), len(cols), figsize=(16, 11), sharex=True, sharey=True
    )
    for i, pop in enumerate(pops):
        for j, (label, title) in enumerate(cols):
            _panel(axes[i, j], pop, label, f"{pop} | {title}")
        axes[i, 0].set_ylabel("rate ratio")
    axes[0, 0].legend(fontsize=6, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("lag since previous onset / end (ms)")
    fig.suptitle(
        "Self-history kernels pooled within animal (at-well; measurable lags only, 95 % CI)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(OUT / "pooled_kernels_self.png", dpi=130)
    plt.close(fig)

    cpops = ["non_local_swr", "non_local_no_swr"]
    ccols = [
        ("shipped_cross_onset", "shipped design: other stream's onset"),
        ("dur_cross_onset_only", "at-risk: other's onset only"),
        ("dur_cross_offset_only", "at-risk: other's end only"),
        ("dur_both_cross_onset", "at-risk both: other's onset"),
        ("dur_both_cross_offset", "at-risk both: other's end"),
    ]
    fig, axes = plt.subplots(2, len(ccols), figsize=(16, 6), sharex=True, sharey=True)
    for i, pop in enumerate(cpops):
        for j, (label, title) in enumerate(ccols):
            _panel(axes[i, j], pop, label, f"{pop} | {title}")
        axes[i, 0].set_ylabel("rate ratio")
    axes[0, 0].legend(fontsize=6, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("lag since the other stream's onset / end (ms)")
    fig.suptitle("Cross-history kernels pooled within animal (at-well)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "pooled_kernels_cross.png", dpi=130)
    plt.close(fig)

    for pop, ps in summary["populations"].items():
        print(
            f"\n## {pop}  failed={ {k: v for k, v in ps['failed_fits'].items() if v} }"
        )
        for name, c in ps["contrasts"].items():
            print(
                f"  {name:27s} med dAIC={c['median_d_aic']:+8.1f} +{c['animals_positive']}/5 sig={c.get('animals_significant', '-')}"
            )
        for label, k in ps["kernels"].items():
            print(
                f"  {label:24s} meas={k['n_animals_measurable']} peak={k['median_peak_rr']} "
                f"@{k['median_peak_lag_ms']} CI>1:{k['animals_peak_ci_above_1']} "
                f"min={k['median_min_measurable_rr']} first3={k['median_rr_first_three_lags']}"
            )


if __name__ == "__main__":
    main()
