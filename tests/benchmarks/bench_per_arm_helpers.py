"""Benchmark: vectorized vs scalar per-arm helpers.

Run with:
    python tests/benchmarks/bench_per_arm_helpers.py
"""
from __future__ import annotations

import time

import networkx as nx
import numpy as np
import pandas as pd


def _bandit_graph() -> nx.Graph:
    g = nx.Graph()
    for u, v, eid in (
        (0, 6, 0), (1, 6, 1), (2, 7, 2), (3, 7, 3),
        (4, 8, 4), (5, 8, 5), (6, 9, 6), (7, 9, 7), (8, 9, 8),
    ):
        g.add_edge(u, v, edge_id=eid, distance=1.0)
    return g


def _make_trials(n_trials: int, rng: np.random.Generator) -> pd.DataFrame:
    end_times = np.sort(rng.uniform(0, 100, n_trials))
    start_times = end_times - rng.uniform(0.5, 2.0, n_trials)
    return pd.DataFrame({
        "start_time": start_times,
        "end_time": end_times,
        "to_well": rng.choice([0, 1, 2, 3, 4, 5], n_trials).astype(float),
        "from_well": rng.choice([0, 1, 2, 3, 4, 5], n_trials).astype(float),
        "is_reward": rng.choice([True, False], n_trials),
        "is_patch_change": rng.choice([True, False], n_trials),
    })


def _run_scalar(func, event_times, trials, track_graph):
    """Run the scalar (per-event) version, collect results as 2D array."""
    from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH
    ALL_ARMS = sorted(TRACK_SEGMENT_TO_PATCH.keys())
    n_arms = len(ALL_ARMS)
    out = np.full((len(event_times), n_arms), np.nan)
    for i, t in enumerate(event_times):
        result = func(float(t), trials, track_graph)
        for j, arm in enumerate(ALL_ARMS):
            out[i, j] = result[arm]
    return out


def main():
    from continuum_swr_replay.analysis.replay._helpers import (
        build_well_to_arm_map,
        cumulative_reward_all_events,
        cumulative_reward_per_arm,
        recent_reward_rate_all_events,
        recent_reward_rate_per_arm,
        visit_recency_all_events,
        visit_recency_per_arm,
    )

    rng = np.random.default_rng(42)
    graph = _bandit_graph()
    trials = _make_trials(80, rng)
    event_times = np.sort(rng.uniform(0, 100, 500))
    w2a = build_well_to_arm_map(graph)

    funcs = [
        ("visit_recency", visit_recency_per_arm, visit_recency_all_events),
        ("cumulative_reward", cumulative_reward_per_arm, cumulative_reward_all_events),
        ("recent_reward_rate", recent_reward_rate_per_arm, recent_reward_rate_all_events),
    ]

    for name, scalar_fn, batch_fn in funcs:
        # --- Scalar (old) ---
        t0 = time.perf_counter()
        old = _run_scalar(scalar_fn, event_times, trials, graph)
        t_old = time.perf_counter() - t0

        # --- Batch (new) ---
        t0 = time.perf_counter()
        new = batch_fn(event_times, trials, w2a)
        t_new = time.perf_counter() - t0

        # Verify identical output
        np.testing.assert_array_equal(
            old, new,
            err_msg=f"{name}: vectorized output differs from scalar",
        )

        speedup = t_old / t_new if t_new > 0 else float("inf")
        print(f"{name:30s}  scalar={t_old:.4f}s  batch={t_new:.4f}s  speedup={speedup:.1f}x")

    print("\nAll outputs match exactly.")


if __name__ == "__main__":
    main()
