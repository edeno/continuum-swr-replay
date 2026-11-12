"""Benchmark: throughput of the vectorized per-arm history helpers.

Run with (as a module, so the shared test fixtures import):
    python -m tests.benchmarks.bench_per_arm_helpers
"""

from __future__ import annotations

import time

import networkx as nx
import numpy as np
import pandas as pd


def _bandit_graph() -> nx.Graph:
    g = nx.Graph()
    for u, v, eid in (
        (0, 6, 0),
        (1, 6, 1),
        (2, 7, 2),
        (3, 7, 3),
        (4, 8, 4),
        (5, 8, 5),
        (6, 9, 6),
        (7, 9, 7),
        (8, 9, 8),
    ):
        g.add_edge(u, v, edge_id=eid, distance=1.0)
    return g


def _make_trials(n_trials: int, rng: np.random.Generator) -> pd.DataFrame:
    end_times = np.sort(rng.uniform(0, 100, n_trials))
    start_times = end_times - rng.uniform(0.5, 2.0, n_trials)
    return pd.DataFrame(
        {
            "movement_start_time": start_times,
            "end_time": end_times,
            "to_well": rng.choice([0, 1, 2, 3, 4, 5], n_trials).astype(float),
            "from_well": rng.choice([0, 1, 2, 3, 4, 5], n_trials).astype(float),
            "is_reward": rng.choice([True, False], n_trials),
            "is_patch_change": rng.choice([True, False], n_trials),
        }
    )


def main():
    from continuum_swr_replay.analysis.behavior._helpers import (
        build_well_to_arm_map,
    )
    from continuum_swr_replay.analysis.behavior.covariates import (
        recent_reward_rate_all_events,
        rolling_reward_count_all_events,
        visit_recency_all_events,
    )
    from continuum_swr_replay.data_loaders.constants import ALL_ARMS
    from tests.support._port_experience_fixtures import port_experiences_from_trials

    rng = np.random.default_rng(42)
    graph = _bandit_graph()
    trials = _make_trials(80, rng)
    experiences = port_experiences_from_trials(trials, prepend_initial=True)
    event_times = np.sort(rng.uniform(0, 100, 500))
    w2a = build_well_to_arm_map(graph)

    funcs = [
        ("visit_recency_trials", visit_recency_all_events),
        (
            "reward_count_candidate_arm_last10_known_trials",
            rolling_reward_count_all_events,
        ),
        ("reward_fraction_candidate_arm_last10_visits", recent_reward_rate_all_events),
    ]

    for name, batch_fn in funcs:
        t0 = time.perf_counter()
        values = batch_fn(event_times, experiences, w2a)
        elapsed = time.perf_counter() - t0

        assert values.shape == (len(event_times), len(ALL_ARMS)), name
        per_event_us = 1e6 * elapsed / len(event_times)
        print(
            f"{name:30s}  batch={elapsed:.4f}s"
            f"  ({per_event_us:.1f} us/event over {len(event_times)} events)"
        )


if __name__ == "__main__":
    main()
