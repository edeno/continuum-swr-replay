#!/usr/bin/env python
"""Exploratory: lagged-reward kernel rate model (replaces is_after_omission +
recent_trial_reward_rate with per-lag reward outcomes).

For each stop, reward_lag{j} = 1 if the j-th most recent trial before the stop
was rewarded. Fits a per-stop Poisson (log-immobility offset) with
reward_lag1..K + is_after_switch, per session, for detected-SWR and
non-SWR-non-local counts, and reports the median rate-ratio kernel.

Key question: lag-1 boosts (RR>1, immediate reward) while lags >=2 suppress
(RR<1, the recent baseline) = RPE shape; and lags >=2 are dwell-clean (past
trials don't set the current stop's duration).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from continuum_swr_replay.analysis.replay import (  # noqa: E402
    trial_outcome_times,
)
from continuum_swr_replay.analysis.replay.per_stop_rate_model import (  # noqa: E402
    fit_per_stop_rate_model,
)
from continuum_swr_replay.analysis.replay.prepare_data import (  # noqa: E402
    find_sessions,
)
from continuum_swr_replay.data_loaders import load_data  # noqa: E402
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR  # noqa: E402
from continuum_swr_replay.types import SessionDataKeys  # noqa: E402

K = 5
RC = PROCESSED_DATA_DIR / "replay_content"


def lagged_rewards(query_times: np.ndarray, trials: pd.DataFrame, k: int) -> np.ndarray:
    """(n_query, k): reward at lag 1..k before each query time; NaN if absent."""
    query_times = np.asarray(query_times, dtype=float)
    out = np.full((query_times.size, k), np.nan)
    if len(trials) == 0:
        return out
    # Reward lags become available at feedback, not merely at destination
    # arrival. Older trial frames conservatively fall back to end_time.
    cutoff = trial_outcome_times(trials)
    order = np.argsort(cutoff)
    cutoff_sorted = cutoff[order]
    is_reward = trials["is_reward"].to_numpy().astype(float)[order]
    counts = np.searchsorted(cutoff_sorted, query_times, side="right")
    for j in range(1, k + 1):
        idx = counts - j
        valid = idx >= 0
        out[valid, j - 1] = is_reward[idx[valid]]
    return out


def main() -> None:
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    stop = pd.read_pickle(RC / "stop_table.pkl")
    sessions = find_sessions(max_per_animal=n_per)
    preds = [f"reward_lag{j}" for j in range(1, K + 1)] + ["is_after_switch"]
    rows = []
    for s in sessions:
        adate = Path(s["dot_file"]).name.split("_")[0]
        animal, day = adate[:-8], int(adate[-8:])
        grp = stop[
            (stop.animal == animal)
            & (stop.day == day)
            & (stop.epoch == s["epoch_name"])
        ].copy()
        if len(grp) < 30:
            continue
        try:
            data = load_data(
                s["nwb_file_name"],
                s["epoch_name"],
                include=frozenset({SessionDataKeys.TRIALS}),
            )
            trials = data.get("trials")
        except Exception:
            continue
        if trials is None or "is_reward" not in trials:
            continue
        lags = lagged_rewards(grp["stop_start"].to_numpy(), trials, K)
        for j in range(K):
            grp[f"reward_lag{j + 1}"] = lags[:, j]
        for outcome, col in [
            ("swr", "n_SWR"),
            ("non_local_no_swr", "n_non_local_no_SWR"),
        ]:
            res = fit_per_stop_rate_model(grp, preds, response_col=col)
            if res is not None:
                rows.append({"outcome": outcome, "animal": animal, **res})
    df = pd.DataFrame(rows)
    print(f"\nfit {df.groupby('outcome').size().to_dict()} (sessions x outcome), K={K}")
    print("\nmedian rate-ratio kernel (RR>1 = reward at that lag INCREASES rate):")
    for outcome in ["swr", "non_local_no_swr"]:
        d = df[df.outcome == outcome]
        print(f"\n[{outcome}]  n={len(d)} sessions")
        for j in range(1, K + 1):
            c = f"coef_reward_lag{j}"
            v = d[c].dropna()
            rr = np.exp(v.median())
            print(
                f"  reward_lag{j}: RR={rr:.2f}  (median coef {v.median():+.3f}, "
                f"{(v > 0).sum()}+/{(v < 0).sum()}-)"
            )
        sw = d["coef_is_after_switch"].dropna()
        print(f"  is_after_switch: RR={np.exp(sw.median()):.2f}")


if __name__ == "__main__":
    main()
