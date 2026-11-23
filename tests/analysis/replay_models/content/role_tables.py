"""Synthetic event-arm tables for the role content tests.

Each session is a walk over the six wells (never the same well twice in a
row) with a few events at each visit. The candidate-arm columns follow the
event-arm table's rules: the current arm 1 trial since its last visit, the
previous one 2, a leaf arm not yet visited carries the censored count and the
never-visited flag, trunk arms carry zeros; the belief variance climbs with the
trials since the arm's last visit, as on the real table. The shares are
placeholders (``1/9``): a test draws them from a scenario's expected shares.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay_models.content import role_content as rc

N_ARMS = 9


def synthetic_event_arm_table(
    sessions: tuple[tuple[str, str, str], ...] = (("a1", "20200101", "02_r1"),),
    *,
    n_visits: int = 120,
    events_per_visit: float = 2.5,
    swr_share: float = 0.5,
    seed: int = 0,
    outside_share: float = 0.05,
) -> pd.DataFrame:
    """A synthetic event-arm table, nine rows per event (module docstring).

    ``outside_share`` of the events start outside a visit
    (``after_visit_at_well``), and as many again cross the final poke-out
    (``spans_visit_end``).
    """
    rng = np.random.default_rng(seed)
    p_reward = rng.uniform(0.2, 0.9, 6)
    frames = []
    event_id = 0
    for animal, date, epoch in sessions:
        wells = [int(rng.integers(6))]
        for _ in range(n_visits - 1):
            wells.append(int(rng.choice([w for w in range(6) if w != wells[-1]])))
        rewarded = rng.random(n_visits) < p_reward[wells]
        rows = []
        for v, well in enumerate(wells):
            history = wells[:v]  # visits before this one
            for _ in range(rng.poisson(events_per_visit)):
                event_id += 1
                context = rng.choice(
                    ["in_visit", "spans_visit_end", "after_visit_at_well"],
                    p=[1 - 2 * outside_share, outside_share, outside_share],
                )
                is_swr = bool(rng.random() < swr_share)
                duration = float(np.exp(rng.normal(np.log(0.08), 0.5)))
                until_end = float(rng.uniform(0.02, 5.0))
                for arm in range(N_ARMS):
                    row = {
                        "animal": animal,
                        "session_date": int(date),
                        "epoch": epoch,
                        "event_id": event_id,
                        "candidate_arm": arm,
                        "arm_fraction": 1.0 / N_ARMS,
                        "is_swr": is_swr,
                        "visit_context": context,
                        "occupancy_well_arm": float(well),
                        "prev_arm": float(history[-1]) if history else np.nan,
                        "n_visits_arrived": v + 1,
                        "time_until_visit_end_s": until_end,
                        "event_duration_s": duration,
                    }
                    if arm >= 6:
                        row |= {
                            "visit_recency_trials": 0.0,
                            "is_candidate_arm_never_visited": 0.0,
                            "reward_fraction_candidate_arm_last10_visits": 0.0,
                            "reward_belief_mean_candidate_arm_preoutcome": np.nan,
                            "reward_belief_variance_candidate_arm_preoutcome": np.nan,
                            "log1p_n_known_visits_candidate_arm": 0.0,
                        }
                    else:
                        visits = [i for i, w in enumerate(wells[: v + 1]) if w == arm]
                        known = [i for i in visits if i < v]
                        outcomes = rewarded[known][-10:]
                        recency = (v + 1) - visits[-1] if visits else v + 2
                        a = 1 + outcomes.sum() * 0.9 ** (recency - 1)
                        b = 1 + (len(outcomes) - outcomes.sum()) * 0.9 ** (recency - 1)
                        row |= {
                            "visit_recency_trials": float(recency),
                            "is_candidate_arm_never_visited": float(not visits),
                            "reward_fraction_candidate_arm_last10_visits": float(
                                outcomes.mean() if len(outcomes) else 0.0
                            ),
                            "reward_belief_mean_candidate_arm_preoutcome": a / (a + b),
                            "reward_belief_variance_candidate_arm_preoutcome": a
                            * b
                            / ((a + b) ** 2 * (a + b + 1)),
                            "log1p_n_known_visits_candidate_arm": float(
                                np.log1p(len(known))
                            ),
                        }
                    rows.append(row)
        frames.append(pd.DataFrame(rows))
    table = pd.concat(frames, ignore_index=True)
    table["event_id"] = table["event_id"].astype(np.uint64)
    return table


def draw_shares(
    rows: pd.DataFrame, expected: np.ndarray, n_bins: int, seed: int
) -> pd.DataFrame:
    """``rows`` with ``arm_fraction`` drawn as multinomial counts over
    ``n_bins`` bins per event, divided by ``n_bins``."""
    rng = np.random.default_rng(seed)
    out = rows.copy()
    share = np.empty(len(rows))
    for _, index in rows.groupby(rc.UNIT, sort=False).indices.items():
        p = expected[index] / expected[index].sum()
        share[index] = rng.multinomial(n_bins, p) / n_bins
    out[rc.RESPONSE] = share
    return out
