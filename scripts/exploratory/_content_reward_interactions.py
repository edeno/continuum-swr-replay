#!/usr/bin/env python
"""Does reward / value / exploration drive current-arm content representation?

TEMPORARY / exploratory. Adds behavioral covariates to the content (arm-hierarchy)
model as interactions with ``is_current_arm`` -- the only way scalar per-stop covariates
enter a conditional/multinomial-over-arms model (a constant-across-arms term cancels
in the event normalization). Each interaction asks whether the current-arm content
bias *sharpens* with the behavioral variable:

  known_reward_outcome_lag_1_x_is_current_arm
  task_reward_probability_chosen_arm_x_is_current_arm
  patch_switch_fraction_last5_completed_trips_x_is_current_arm

The explicit product names prevent collisions with the retired immediate-switch
interaction and make every fitted factor visible in coefficient tables.

The within-well known-outcome interaction is the occupancy-confound-clean
test: it compares rewarded vs omitted visits to the SAME current well (identifiable
because the schedule is probabilistic), isolating the reward OUTCOME from the well's
location and from the animal being there.

Base = M5 content predictors (arm FE + is_current + is_prev + recent_reward_rate +
visit_recency_trials_event_leaf_centered + is_last_rewarded_elsewhere). Fit per session (conditional
Poisson), SWR-split (all / with-SWR / non-SWR). Reports the interaction coefficients.

Usage: python scripts/exploratory/_content_reward_interactions.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from continuum_swr_replay.analysis.replay._helpers import (
    add_arm_fixed_effects,
)
from continuum_swr_replay.analysis.replay.covariates import add_uncertainty_covariates
from continuum_swr_replay.analysis.replay.fitting import fit_conditional_poisson
from continuum_swr_replay.analysis.replay.model_specs import MODEL_HIERARCHY
from continuum_swr_replay.paths import REPLAY_CONTENT_DIR, table_path
from continuum_swr_replay.tables.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

RC = REPLAY_CONTENT_DIR
SC = list(SESSION_COLUMNS)
SCALARS = [
    "known_reward_outcome_lag_1",
    "task_reward_probability_chosen_arm",
    "patch_switch_fraction_last5_completed_trips",
]
INTERACTIONS = [
    "known_reward_outcome_lag_1_x_is_current_arm",
    "task_reward_probability_chosen_arm_x_is_current_arm",
    "patch_switch_fraction_last5_completed_trips_x_is_current_arm",
]
REPORT = ["is_current_arm", *INTERACTIONS]


def _assemble() -> pd.DataFrame:
    ev = read_table(table_path("event_arm_table", RC, fmt="pkl"))
    require_covariate_schema(ev, table="event_arm")
    add_arm_fixed_effects(ev)
    stop = read_table(table_path("stop_table", RC, fmt="pkl"))
    require_covariate_schema(stop, table="stop")
    stop = add_uncertainty_covariates(stop)
    cont = read_table(table_path("qvals_table", RC, fmt="pkl"))
    require_covariate_schema(cont, table="qvals")
    ev = ev.merge(stop[["stop_id", "known_reward_outcome_lag_1", "patch_switch_fraction_last5_completed_trips"]],
                  on="stop_id", how="left")
    ev = ev.merge(cont[["stop_id", "task_reward_probability_chosen_arm"]], on="stop_id", how="left")
    # scalars are constant within an event (one stop per event), so dropping
    # NaN-scalar rows drops whole events cleanly.
    ev = ev.dropna(subset=SCALARS).reset_index(drop=True)
    cur = ev["is_current_arm"].to_numpy(dtype=float)
    ev["known_reward_outcome_lag_1_x_is_current_arm"] = (
        cur * ev["known_reward_outcome_lag_1"].to_numpy(dtype=float)
    )
    # `task_reward_probability_chosen_arm` is the *scheduled* contingency the animal never
    # observes (covariate_dictionary F.0), so calling it "value" invites a
    # belief reading it does not support.
    ev["task_reward_probability_chosen_arm_x_is_current_arm"] = (
        cur * ev["task_reward_probability_chosen_arm"].to_numpy(dtype=float)
    )
    # recent_switch_rate is a 5-trial fraction that is exactly 0 on 64% of stops
    # and exactly 0.2 on 28%, so a per-unit coefficient extrapolates 5x beyond
    # the modal step; read it per 0.2, not per 1.
    ev["patch_switch_fraction_last5_completed_trips_x_is_current_arm"] = (
        cur * ev["patch_switch_fraction_last5_completed_trips"].to_numpy(dtype=float)
    )
    # duration control: within-stop time on the current arm (z(log1p), so the
    # 0-105 s range and the long-stop outliers don't dominate the conditioning).
    tsc = np.log1p(ev["time_since_stop_start_s"].to_numpy(dtype=float))
    tsc = (tsc - tsc.mean()) / tsc.std()
    time_interaction = "log1p_time_since_stop_start_s_z_x_is_current_arm"
    ev[time_interaction] = cur * tsc
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
        time_interaction = "log1p_time_since_stop_start_s_z_x_is_current_arm"
        base_tsc = _fit(sub, [*base, time_interaction])
        _line(base_tsc, "is_current_arm")
        _line(base_tsc, time_interaction)
        # each behavioral interaction, CONTROLLED for within-stop time
        for inter in INTERACTIONS:
            _line(_fit(sub, [*base, time_interaction, inter]), inter)
        print()


if __name__ == "__main__":
    main()
