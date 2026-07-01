#!/usr/bin/env python
"""Does reward / value / exploration drive current-arm content representation?

TEMPORARY / exploratory. Adds behavioral covariates to the content (arm-hierarchy)
model as interactions with ``is_current`` -- the only way scalar per-stop covariates
enter a conditional/multinomial-over-arms model (a constant-across-arms term cancels
in the event normalization). Each interaction asks whether the current-arm content
bias *sharpens* with the behavioral variable:

  rew_x_current    = is_current * reward_lag1          (within-well outcome contrast)
  value_x_current  = is_current * p_reward_current     (current-arm nominal value)
  switch_x_current = is_current * recent_switch_rate   (exploration)

The within-well outcome contrast (rew_x_current) is the occupancy-confound-clean
test: it compares rewarded vs omitted visits to the SAME current well (identifiable
because the schedule is probabilistic), isolating the reward OUTCOME from the well's
location and from the animal being there.

Base = M5 content predictors (arm FE + is_current + is_prev + recent_reward_rate +
visit_recency_demeaned + is_last_rewarded_elsewhere). Fit per session (conditional
Poisson), SWR-split (all / with-SWR / non-SWR). Reports the interaction coefficients.

Usage: python scripts/_content_reward_interactions.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from continuum_swr_replay.analysis.replay._helpers import (
    add_arm_fixed_effects,
    add_uncertainty_covariates,
)
from continuum_swr_replay.analysis.replay.fitting import fit_conditional_poisson
from continuum_swr_replay.analysis.replay.model_specs import MODEL_HIERARCHY
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

RC = PROCESSED_DATA_DIR / "replay_content"
SC = ["animal", "day", "epoch"]
SCALARS = ["reward_lag1", "p_reward_current", "recent_switch_rate"]
INTERACTIONS = ["rew_x_current", "value_x_current", "switch_x_current"]
REPORT = ["is_current", *INTERACTIONS]


def _assemble() -> pd.DataFrame:
    ev = pd.read_pickle(RC / "event_arm_table.pkl")
    add_arm_fixed_effects(ev)
    stop = add_uncertainty_covariates(pd.read_pickle(RC / "stop_table.pkl"))
    cont = pd.read_pickle(RC / "models_per_bin_rate" / "qvals_table.pkl")
    ev = ev.merge(stop[["stop_id", "reward_lag1", "recent_switch_rate"]],
                  on="stop_id", how="left")
    ev = ev.merge(cont[["stop_id", "p_reward_current"]], on="stop_id", how="left")
    # scalars are constant within an event (one stop per event), so dropping
    # NaN-scalar rows drops whole events cleanly.
    ev = ev.dropna(subset=SCALARS).reset_index(drop=True)
    cur = ev["is_current"].to_numpy(dtype=float)
    ev["rew_x_current"] = cur * ev["reward_lag1"].to_numpy(dtype=float)
    ev["value_x_current"] = cur * ev["p_reward_current"].to_numpy(dtype=float)
    ev["switch_x_current"] = cur * ev["recent_switch_rate"].to_numpy(dtype=float)
    # duration control: within-stop time on the current arm (z(log1p), so the
    # 0-105 s range and the long-stop outliers don't dominate the conditioning).
    tsc = np.log1p(ev["time_since_stop"].to_numpy(dtype=float))
    tsc = (tsc - tsc.mean()) / tsc.std()
    ev["tsc_x_current"] = cur * tsc
    return ev


def _fit(ev: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in ev.groupby(SC):
        if len(g) < 18:
            continue
        r = fit_conditional_poisson(
            g, predictors=predictors, response_col="arm_count_hard",
            event_col="event_id", n_categories=9,
        )
        if r is None:
            continue
        rows.append({**dict(zip(SC, keys, strict=True)),
                     **{k: v for k, v in r.items() if k.startswith("coef_")}})
    return pd.DataFrame(rows)


def _line(mdf: pd.DataFrame, col: str) -> None:
    c = f"coef_{col}"
    if c not in mdf:
        print(f"  {col:<20}{'(unfit)':>10}")
        return
    s = mdf[c].dropna()
    if len(s) < 5:
        print(f"  {col:<20}{'(n<5)':>10}")
        return
    n_an = (mdf.groupby("animal")[c].median() > 0).sum()
    print(f"  {col:<20}{s.median():>+10.3f}{(s > 0).mean() * 100:>6.0f}%"
          f"{f'{n_an}/5':>9}{wilcoxon(s)[1]:>10.1e}{f'  n={len(s)}':>8}")


def main() -> None:
    ev = _assemble()
    base = list(MODEL_HIERARCHY["M5"].predictors)
    # The three is_current interactions are mutually collinear (each is
    # is_current * an event-constant scalar), so add ONE at a time on top of M5.
    print(f"M5 + ONE current-arm interaction at a time; {len(ev)} arm-rows, "
          f"{ev['event_id'].nunique()} events\n")
    for label, sub in [
        ("ALL non-local", ev),
        ("with-SWR", ev[ev.is_swr]),
        ("non-SWR", ev[~ev.is_swr]),
    ]:
        print(f"=== {label} ===")
        print(f"  {'term':<20}{'med coef':>10}{'%pos':>7}{'animals':>9}{'p':>10}")
        # base M5 + within-stop-time control: gives is_current and the drift term
        base_tsc = _fit(sub, [*base, "tsc_x_current"])
        _line(base_tsc, "is_current")
        _line(base_tsc, "tsc_x_current")  # within-stop content drift off current
        # each behavioral interaction, CONTROLLED for within-stop time
        for inter in INTERACTIONS:
            _line(_fit(sub, [*base, "tsc_x_current", inter]), inter)
        print()


if __name__ == "__main__":
    main()
