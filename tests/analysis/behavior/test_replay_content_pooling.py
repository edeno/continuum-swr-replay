"""Session/animal pooling tests for the replay-content GLM.

The production fit pools events across animals via a shared
``ConditionalPoisson`` regression. The model conditions on ``event_id``,
but it has no per-animal intercept or random effect. We probe two
distinct concerns:

1. **Bias when animals share β**: simulating multiple animals with
   identical generative coefficients should recover those coefficients
   unbiased, regardless of the per-animal sample size.
2. **Pooled-fit bias / SE under-coverage when animals differ**: when
   per-animal β's vary, the pooled fit estimates a population average β.
   Is that average unbiased w.r.t. the between-animal mean? And does
   the nominal ``bse`` capture the true between-fit variability, or does
   it under-cover (the cluster-by-animal analogue of the cluster-by-event
   autocorrelation finding)?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.content.simulation import (
    simulate_counts,
)
from continuum_swr_replay.analysis.behavior.fitting import (
    fit_conditional_poisson,
)


def _pool_animals(per_animal_betas, *, n_events_per_animal, L_e, seed):
    """Concatenate Tier-A simulations from multiple animals into one
    long-format frame with globally unique ``event_id``s."""
    rng = np.random.default_rng(seed)
    parts = []
    offset = 0
    for i, beta in enumerate(per_animal_betas):
        n = (
            n_events_per_animal[i]
            if isinstance(n_events_per_animal, (list, tuple, np.ndarray))
            else n_events_per_animal
        )
        df = simulate_counts(beta, n_events=n, L_e=L_e, seed=int(rng.integers(0, 1e9)))
        df["event_id"] = df["event_id"].astype(int) + offset
        df["animal"] = f"a{i}"
        offset = int(df["event_id"].max()) + 1
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


class TestPoolingWhenAnimalsShareTruth:
    """If every animal has the same generative β, pooling should give an
    unbiased estimate, and equal-size pooling should match per-animal
    fits in mean."""

    def test_pooled_recovery_with_shared_beta(self):
        beta = {"is_current_arm": 0.5, "is_prev_arm": 0.7}
        df = _pool_animals(
            per_animal_betas=[beta, beta, beta],
            n_events_per_animal=300,
            L_e=20,
            seed=0,
        )
        fit = fit_conditional_poisson(df, predictors=list(beta))
        assert fit is not None
        for p, truth in beta.items():
            z = (fit[f"coef_{p}"] - truth) / fit[f"bse_{p}"]
            assert abs(z) < 3, f"{p} z={z:.2f}"

    def test_pooled_se_smaller_than_single_animal(self):
        """Pooling 3 animals' worth of data should yield ~sqrt(3)× tighter
        SEs than a single-animal fit, since events are independent under
        the shared truth."""
        beta = {"is_current_arm": 0.7}
        single = simulate_counts(beta, n_events=300, L_e=20, seed=11)
        pooled = _pool_animals(
            per_animal_betas=[beta] * 3,
            n_events_per_animal=300,
            L_e=20,
            seed=11,
        )
        fit_single = fit_conditional_poisson(single, predictors=list(beta))
        fit_pooled = fit_conditional_poisson(pooled, predictors=list(beta))
        ratio = fit_pooled["bse_is_current_arm"] / fit_single["bse_is_current_arm"]
        # Expect ~1/sqrt(3) ≈ 0.58. Allow generous tolerance.
        assert 0.45 < ratio < 0.75, (
            f"pooled/single SE ratio = {ratio:.2f}; expected ~0.58"
        )


class TestPooledFitRecoversWeightedAverage:
    """When animals have different β, the pooled fit estimates a
    Fisher-information-weighted average. With equal sample sizes and L_e,
    the weights are roughly equal, so the recovered β should track the
    arithmetic mean of the per-animal β's."""

    def test_recovers_arithmetic_mean_with_equal_sample_sizes(self):
        per_animal = [
            {"is_current_arm": 0.3},
            {"is_current_arm": 0.6},
            {"is_current_arm": 0.9},
        ]
        df = _pool_animals(
            per_animal_betas=per_animal,
            n_events_per_animal=400,
            L_e=20,
            seed=23,
        )
        fit = fit_conditional_poisson(df, predictors=["is_current_arm"])
        assert fit is not None
        true_mean = float(np.mean([b["is_current_arm"] for b in per_animal]))
        # Should be close to the mean (0.6) within a few SE.
        z = (fit["coef_is_current_arm"] - true_mean) / fit["bse_is_current_arm"]
        assert abs(z) < 3, (
            f"pooled β = {fit['coef_is_current_arm']:.3f}, "
            f"animal mean = {true_mean:.3f}, z = {z:.2f}"
        )


class TestPooledSEUnderHeterogeneity:
    """The most important pooling check, analogous to Tier B's
    autocorrelation finding: when per-animal β truly differs but the
    pooled fit ignores that, the nominal ``bse`` understates the true
    between-fit variability of the population-average estimate.

    Construction: across many "experiment replicates", we draw fresh
    per-animal β's from N(μ, τ²) and then simulate events from each
    animal. The empirical SE of the pooled β̂ across experiments has
    two components: (i) within-experiment Fisher information (which the
    nominal ``bse`` captures) and (ii) between-experiment variance from
    the random per-animal β draws (which it does NOT). The empirical
    SE should therefore exceed the mean nominal ``bse``.
    """

    @pytest.mark.slow
    def test_nominal_se_under_covers_with_random_animal_effects(self):
        rng = np.random.default_rng(0)
        n_replicates = 30
        n_animals = 4
        mu = 0.6
        tau = 0.3  # between-animal SD of true β

        coefs, bses = [], []
        for _r in range(n_replicates):
            per_animal = [
                {"is_current_arm": float(rng.normal(mu, tau))} for _ in range(n_animals)
            ]
            df = _pool_animals(
                per_animal_betas=per_animal,
                n_events_per_animal=200,
                L_e=20,
                seed=int(rng.integers(0, 1e9)),
            )
            fit = fit_conditional_poisson(df, predictors=["is_current_arm"])
            if fit is None:
                continue
            coefs.append(fit["coef_is_current_arm"])
            bses.append(fit["bse_is_current_arm"])

        coefs = np.asarray(coefs)
        bses = np.asarray(bses)
        emp_se = float(coefs.std(ddof=1))
        mean_bse = float(bses.mean())
        ratio = mean_bse / emp_se

        # Expect ratio < 1: nominal SE under-covers because between-animal
        # variance is unmodeled.
        assert ratio < 0.85, (
            f"Expected nominal SE under-coverage; got ratio = {ratio:.2f}. "
            f"emp_se = {emp_se:.3f}, mean_bse = {mean_bse:.3f}. "
            f"This means nominal CIs would be too tight under random "
            f"animal effects of SD {tau}."
        )
        # Sanity: the recovered population-average β should still be
        # close to μ.
        assert abs(coefs.mean() - mu) < 3 * emp_se / np.sqrt(n_replicates)
