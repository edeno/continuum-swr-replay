"""Benchmark: vectorized vs loop-based _held_out_conditional_ll.

Run with:
    python tests/benchmarks/bench_conditional_ll.py
"""

from __future__ import annotations

import time

import numpy as np


def _held_out_conditional_ll_loop(y, X, groups, params):
    """Original loop-based implementation for comparison."""
    eta = X @ params
    total_ll = 0.0
    for event_id in np.unique(groups):
        mask = groups == event_id
        eta_e = eta[mask]
        c_e = y[mask]
        L_e = c_e.sum()
        if L_e == 0:
            continue
        max_eta = eta_e.max()
        log_sum_exp = max_eta + np.log(np.sum(np.exp(eta_e - max_eta)))
        total_ll += np.sum(c_e * eta_e) - L_e * log_sum_exp
    return float(total_ll)


def main():
    from continuum_swr_replay.analysis.behavior.content.model_checking import (
        _held_out_conditional_ll,
    )

    rng = np.random.default_rng(42)

    n_events = 5000
    n_arms = 9
    n_predictors = 6
    n_rows = n_events * n_arms

    groups = np.repeat(np.arange(n_events), n_arms)
    X = rng.standard_normal((n_rows, n_predictors))
    params = rng.standard_normal(n_predictors) * 0.5
    y = rng.poisson(3, n_rows).astype(float)

    # --- Loop (old) ---
    t0 = time.perf_counter()
    for _ in range(3):
        old = _held_out_conditional_ll_loop(y, X, groups, params)
    t_old = (time.perf_counter() - t0) / 3

    # --- Vectorized (new) ---
    t0 = time.perf_counter()
    for _ in range(3):
        new = _held_out_conditional_ll(y, X, groups, params)
    t_new = (time.perf_counter() - t0) / 3

    # Verify identical output
    np.testing.assert_allclose(
        old,
        new,
        rtol=1e-10,
        err_msg="_held_out_conditional_ll: vectorized differs from loop",
    )

    speedup = t_old / t_new if t_new > 0 else float("inf")
    print(
        f"_held_out_conditional_ll  loop={t_old:.4f}s  vectorized={t_new:.4f}s  speedup={speedup:.1f}x"
    )
    print(f"  LL value: {new:.6f}")
    print("\nOutputs match (rtol=1e-10).")


if __name__ == "__main__":
    main()
