"""Model-free hazards and per-session spot checks for the duration-aware refit.

1. Empirical hazard of the next onset against the time since the previous
   event ENDED (renewal-style, at-well stops, 15 ms bins to 300 ms): for each
   event, the gap to the next onset of the same detector in its stop, censored
   at the stop's end. ``h(b) = onsets in bin / at-risk seconds in bin``, divided
   by the stream's rate over all at-risk time after 300 ms (the late
   baseline). Split by the next event's population (detected SWR; non-local with
   / without SWR) and, for non-local, by the previous event's population. The
   same against the time since the previous event's ONSET (at-risk only after
   it ended). Per animal and pooled.
2. Three sessions' per-session kernels from ``duration_aware_history_models.pkl``
   (one-anchor rungs, every lag drawn, with its support) beside that session's
   gap histogram.

Writes ``hazard.json``, ``hazard_by_animal.png`` and ``spotcheck_sessions.png``.

Usage::

    CSR_PROCESSED_DATA_DIR=/cumulus/edeno/continuum-swr-replay/Processed-Data \\
    PYTHONPATH=. uv run python \\
        .claude/docs/reviews/2026-10-27-duration-aware-history-refit/hazard_and_spotcheck.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from continuum_swr_replay.analysis.events.cohort_selection import (  # noqa: E402
    select_stop_cohort,
)
from continuum_swr_replay.analysis.replay import attach_is_swr_to_onsets  # noqa: E402
from continuum_swr_replay.analysis.replay.design_constants import (  # noqa: E402
    BIN_S,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    TMAX,
)
from continuum_swr_replay.analysis.replay.duration_aware_history import (  # noqa: E402
    END_COLUMN,
    attach_event_ends,
)
from continuum_swr_replay.analysis.replay.per_bin_design import (  # noqa: E402
    raised_cosine_history_basis,
)
from continuum_swr_replay.paths import model_dir, table_path  # noqa: E402
from continuum_swr_replay.tables.io import read_table  # noqa: E402
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS  # noqa: E402

OUT = Path(__file__).resolve().parent
EDGES = np.arange(0, 0.3 + 1e-9, BIN_S)  # 20 bins of 15 ms
LATE_S = 0.3


def _label(o: pd.DataFrame) -> pd.Series:
    lab = pd.Series("swr", index=o.index)
    nl = o["event_type"] == "non_local"
    lab[nl & (o["is_swr"] == True)] = "non_local_swr"  # noqa: E712
    lab[nl & (o["is_swr"] == False)] = "non_local_no_swr"  # noqa: E712
    return lab


def _intervals(
    onsets: pd.DataFrame, stops: pd.DataFrame, detector: str
) -> pd.DataFrame:
    """Consecutive event pairs of one detector within stops: previous event's
    onset/end, population, the next onset (or censoring at the stop's end)."""
    ev = onsets[onsets["event_type"] == detector].copy()
    ev["pop"] = _label(ev)
    span = (
        stops.set_index("stop_id")["stop_end"]
        - stops.set_index("stop_id")["stop_start"]
    )
    ev["horizon"] = ev["stop_id"].map(span).clip(upper=TMAX)
    ev = ev.dropna(subset=["horizon"]).sort_values(
        [*SESSION_COLUMNS, "stop_id", "t_since_stop_onset"]
    )
    nxt = ev.groupby([*SESSION_COLUMNS, "stop_id"], sort=False)
    out = pd.DataFrame(
        {
            "animal": ev["animal"].to_numpy(),
            "prev_pop": ev["pop"].to_numpy(),
            "prev_onset": ev["t_since_stop_onset"].to_numpy(),
            "prev_end": ev[END_COLUMN].to_numpy(),
            "next_onset": nxt["t_since_stop_onset"].shift(-1).to_numpy(),
            "next_pop": nxt["pop"].shift(-1).to_numpy(),
            "horizon": ev["horizon"].to_numpy(),
        }
    )
    return out[out["prev_end"] < out["horizon"]]


def _hazard(iv: pd.DataFrame, anchor: str, next_pop: str | None) -> dict:
    """Hazard ratio by time since the previous event's ``anchor``."""
    t0 = iv["prev_end"].to_numpy()
    ref = iv["prev_end"] if anchor == "end" else iv["prev_onset"]
    ref = ref.to_numpy()
    stop_at = np.where(
        iv["next_onset"].notna(), iv["next_onset"], iv["horizon"]
    ).astype(float)
    stop_at = np.minimum(stop_at, iv["horizon"].to_numpy())
    hit = iv["next_onset"].notna().to_numpy() & (
        iv["next_onset"].to_numpy() < iv["horizon"].to_numpy()
    )
    if next_pop is not None:
        hit &= iv["next_pop"].to_numpy() == next_pop
    # at-risk interval in "time since ref": [t0 - ref, stop_at - ref)
    lo, hi = t0 - ref, stop_at - ref
    exposure = np.array(
        [
            np.clip(np.minimum(hi, b1) - np.maximum(lo, b0), 0, None).sum()
            for b0, b1 in zip(EDGES[:-1], EDGES[1:], strict=True)
        ]
    )
    tau = (stop_at - ref)[hit]
    counts, _ = np.histogram(tau, bins=EDGES)
    late_exp = np.clip(hi - np.maximum(lo, LATE_S), 0, None).sum()
    late_n = int((tau >= LATE_S).sum())
    base = late_n / late_exp if late_exp > 0 else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = counts / exposure / base
    return {
        "ratio": np.round(ratio, 3).tolist(),
        "counts": counts.tolist(),
        "exposure_s": np.round(exposure, 2).tolist(),
        "late_rate_hz": float(base),
        "n_late": late_n,
    }


def _spotcheck(models: pd.DataFrame, onsets: pd.DataFrame, sessions: list) -> None:
    basis = raised_cosine_history_basis(N_HISTORY_LAGS, N_HISTORY_BASIS)
    lag_ms = np.arange(1, N_HISTORY_LAGS + 1) * BIN_S * 1000
    fig, axes = plt.subplots(
        len(sessions), 3, figsize=(12, 3 * len(sessions)), squeeze=False
    )
    for row, keys in enumerate(sessions):
        sel = (
            models[list(SESSION_COLUMNS)] == pd.Series(keys, index=SESSION_COLUMNS)
        ).all(axis=1)
        m = models[sel]
        for col, outcome in enumerate(("swr", "non_local_no_swr")):
            ax = axes[row, col]
            for rung, prefix, block, color in (
                ("Mself_onset_dur", "hist_rcos_", "self_onset", "#0072B2"),
                ("Mself_offset_dur", "offhist_rcos_", "self_offset", "#D55E00"),
            ):
                r = m[(m["outcome"] == outcome) & (m["model"] == rung)].iloc[0]
                if not r["fit_ok"]:
                    continue
                k = (
                    np.array([r[f"coef_{prefix}{j}"] for j in range(N_HISTORY_BASIS)])
                    @ basis
                )
                sup = np.array(r[f"support_{block}"])
                ax.plot(
                    lag_ms,
                    np.exp(np.clip(k, -4, 4)),
                    "o-",
                    color=color,
                    ms=3,
                    label=f"{block} ({rung})",
                )
                for x, s in zip(lag_ms, sup, strict=True):
                    ax.text(
                        x,
                        0.06 if block == "self_onset" else 0.03,
                        f"{s:.1f}",
                        fontsize=5,
                        color=color,
                        ha="center",
                    )
            ax.axhline(1, color="0.8", lw=0.8)
            ax.set_yscale("log")
            ax.set_ylim(0.02, 50)
            ax.set_title(f"{'_'.join(map(str, keys))} {outcome}", fontsize=8)
            ax.legend(fontsize=6, frameon=False)
            ax.set_xlabel("lag (ms); numbers = expected onsets per lag")
        ax = axes[row, 2]
        o = onsets[
            (
                onsets[list(SESSION_COLUMNS)] == pd.Series(keys, index=SESSION_COLUMNS)
            ).all(axis=1)
        ]
        nl = o[o["event_type"] == "non_local"].sort_values(
            ["stop_id", "t_since_stop_onset"]
        )
        same = nl["stop_id"].to_numpy()[1:] == nl["stop_id"].to_numpy()[:-1]
        gaps = (
            nl["t_since_stop_onset"].to_numpy()[1:] - nl[END_COLUMN].to_numpy()[:-1]
        )[same]
        ax.hist(gaps * 1000, bins=np.arange(0, 500, 10), color="0.4")
        ax.set_xlabel("non-local gap: end to next onset (ms)")
        ax.set_title(f"n gaps = {len(gaps)}", fontsize=8)
    fig.suptitle(
        "Per-session kernels (one-anchor rungs, every lag drawn) and gaps", fontsize=9
    )
    fig.tight_layout()
    fig.savefig(OUT / "spotcheck_sessions.png", dpi=130)
    plt.close(fig)


def main() -> None:
    stop = read_table(table_path("stop_table"))
    onset_table = read_table(table_path("event_onset_table"))
    event_table = read_table(table_path("event_table"))
    onsets = attach_event_ends(
        attach_is_swr_to_onsets(onset_table, event_table), event_table
    )
    stop, _ = select_stop_cohort(stop, cohort="at_well")
    onsets = onsets[onsets["stop_id"].isin(stop["stop_id"])]
    result: dict = {
        "edges_ms": (EDGES * 1000).round(1).tolist(),
        "late_s": LATE_S,
        "hazards": {},
    }
    cases = [
        ("SWR", "swr", None, "end"),
        ("SWR", "swr", None, "onset"),
        ("non_local", "non_local_swr", None, "end"),
        ("non_local", "non_local_swr", None, "onset"),
        ("non_local", "non_local_no_swr", None, "end"),
        ("non_local", "non_local_no_swr", None, "onset"),
        ("non_local", "non_local_no_swr", "non_local_swr", "end"),
        ("non_local", "non_local_swr", "non_local_no_swr", "end"),
        ("non_local", "non_local_swr", "non_local_swr", "end"),
        ("non_local", "non_local_no_swr", "non_local_no_swr", "end"),
    ]
    ivs = {d: _intervals(onsets, stop, d) for d in ("SWR", "non_local")}
    for detector, next_pop, prev_pop, anchor in cases:
        iv = ivs[detector]
        if prev_pop is not None:
            iv = iv[iv["prev_pop"] == prev_pop]
        npop = None if detector == "SWR" else next_pop
        name = f"{next_pop}|prev={prev_pop or 'any'}|since={anchor}"
        result["hazards"][name] = {
            "pooled": _hazard(iv, anchor, npop),
            "animals": {a: _hazard(g, anchor, npop) for a, g in iv.groupby("animal")},
        }
    (OUT / "hazard.json").write_text(json.dumps(result, indent=1))

    centers = (EDGES[:-1] + EDGES[1:]) / 2 * 1000
    fig, axes = plt.subplots(2, 5, figsize=(17, 6), sharey=True)
    for ax, (name, h) in zip(axes.ravel(), result["hazards"].items(), strict=True):
        for a, ha in h["animals"].items():
            ax.plot(centers, ha["ratio"], lw=0.8, alpha=0.6, label=a)
        ax.plot(centers, h["pooled"]["ratio"], "k-", lw=2, label="pooled")
        ax.axhline(1, color="0.7", lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(0.05, 20)
        ax.set_title(name.replace("|", "\n"), fontsize=7)
        ax.set_xlabel("ms")
    axes[0, 0].set_ylabel("hazard / late (>300 ms) rate")
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle("Model-free hazard of the next onset (at-well stops)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "hazard_by_animal.png", dpi=130)
    plt.close(fig)

    models = pd.read_pickle(
        model_dir("per_bin_rate") / "duration_aware_history_models.pkl"
    )
    n_events = (
        models[
            (models["model"] == "Mnohist_dur")
            & (models["outcome"] == "non_local_no_swr")
        ]
        .groupby("animal")
        .apply(lambda g: g.loc[g["n_events"].idxmax(), list(SESSION_COLUMNS)])
    )
    picks = [tuple(n_events.loc[a]) for a in ("chimi", "j16", "wilbur")]
    _spotcheck(models, onsets, picks)
    print(
        json.dumps(
            {k: v["pooled"]["ratio"][:8] for k, v in result["hazards"].items()},
            indent=0,
        )
    )


if __name__ == "__main__":
    main()
