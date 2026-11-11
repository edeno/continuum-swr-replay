"""The behavior summary's duration-aware history and horse-race values, on synthetic inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.history_kernel_statistics import (
    HISTORY_MAX_SE,
)
from continuum_swr_replay.reported.behavior import (
    duration_aware_history_values,
    horserace_values,
)

ANIMALS = ("a", "b", "c", "d", "e")
POPS = ("swr", "non_local", "non_local_swr", "non_local_no_swr")


def _kernel(log_rr, se=0.1):
    se = [se] * 10 if np.isscalar(se) else list(se)
    return {"log_rr": list(log_rr), "se": se, "coef": [0.0] * 4}


def _pooled(
    onset_peak_rr: float = 2.0,
    end_first_rr: float = 3.0,
    *,
    imprecise_first_lag: tuple[str, str] | None = None,
) -> dict:
    """Every animal: an end-locked kernel peaking at 30 ms and an onset-locked one
    peaking at 105 ms (rate ratio ``onset_peak_rr``). Onset lags 1-3 have no
    support (not measurable).

    ``imprecise_first_lag`` = (population, animal) whose 15 ms end-locked lag
    has a Wald SE above ``HISTORY_MAX_SE`` (and an extreme rate ratio that must
    not be read)."""
    end = np.full(10, 0.0)
    end[0], end[1] = np.log(end_first_rr), np.log(4.0)
    onset = np.full(10, 0.1)
    onset[6] = np.log(onset_peak_rr)
    onset[0] = 5.0  # unsupported: must not be read
    support = {
        "self_onset": [0.0, 1.0, 2.0] + [10.0] * 7,
        "self_offset": [20.0] * 10,
    }
    results = []
    for pop in POPS:
        for animal in ANIMALS:
            end_k = _kernel(end)
            if (pop, animal) == imprecise_first_lag:
                lr = end.copy()
                lr[0] = np.log(50.0)
                end_k = _kernel(lr, [HISTORY_MAX_SE + 0.01] + [0.1] * 9)
            fits = {
                "Mself_dur": {
                    "ok": True,
                    "converged": True,
                    "kernels": {
                        "self_offset": end_k,
                        "self_onset": _kernel(onset),
                    },
                }
            }
            results.append(
                {
                    "animal": animal,
                    "outcome": pop,
                    "support": support,
                    "duration_aware": fits,
                }
            )
    return {"stop_cohort": "at_well", "results": results}


def _ridge(
    gains: tuple[float, ...] = (0.01, 0.02, 0.03, -0.01, 0.04),
    *,
    cross_rr: float = 2.5,
    unsupported: tuple[str, str] | None = None,
    unconverged: tuple[str, str] | None = None,
) -> dict:
    """Per animal and non-local stream: a held-out gain from ``gains`` and a 1 s
    end-locked cross kernel of rate ratio ``cross_rr`` at 45 ms (posterior SD
    0.1). ``unsupported`` = (stream, animal) whose 45 ms lag has fewer than 3
    expected onsets (an extreme rate ratio that must not be read);
    ``unconverged`` = (stream, animal) whose self+cross fit did not converge."""
    results = []
    for pop in ("non_local_swr", "non_local_no_swr"):
        for animal, gain in zip(ANIMALS, gains, strict=True):
            lr = np.zeros(67)
            lr[2] = np.log(cross_rr)
            support = np.full(67, 20.0)
            if (pop, animal) == unsupported:
                lr[2], support[2] = np.log(50.0), 1.0
            kernel = {"log_rr": lr.tolist(), "sd": [0.1] * 67, "coef": [0.0] * 6}
            models = {
                rung: {
                    "full_converged": True,
                    "cv": {"n_nonconverged": 0},
                    "kernels": {"cross_offset": kernel},
                }
                for rung in ("Mself_long_dur", "Mselfcross_long_dur")
            }
            if (pop, animal) == unconverged:
                models["Mselfcross_long_dur"]["cv"]["n_nonconverged"] = 1
            results.append(
                {
                    "animal": animal,
                    "outcome": pop,
                    "gain_per_event": gain,
                    "support": {"cross_offset": support.tolist()},
                    "models": models,
                }
            )
    return {"stop_cohort": "at_well", "results": results}


def test_history_values_read_measurable_lags_and_count_animals():
    values = {v.key: v for v in duration_aware_history_values(_ridge(), _pooled(), "x")}
    # the onset peak is read at 105 ms, not at the unsupported first lag
    assert values["history_onset_peak_swr"].value == pytest.approx(2.0)
    assert values["history_onset_peak_lag_ms_swr"].value == 105
    assert values["history_onset_animals_swr"].value == 5
    assert values["history_end_peak_high"].value == pytest.approx(4.0)
    assert values["history_end_peak_lag_ms_low"].value == 30
    assert values["history_end_peak_animals_min"].value == 5
    assert values["history_end_first_lag_rr_low"].value == pytest.approx(3.0)
    assert values["history_end_first_lag_rr_high"].value == pytest.approx(3.0)
    assert values["history_end_first_lag_suppressed_fits"].value == 0
    assert values["history_end_first_lag_raised_fits"].value == 20
    assert values["history_fits"].value == 20
    assert values["history_cross_end_rr_non_swr_from_swr"].value == pytest.approx(2.5)
    assert values["history_cross_end_animals_swr_from_non_swr"].value == 5
    assert values["history_cross_end_rr_non_swr_from_swr"].pending is None
    assert values["history_cross_heldout_gain_non_swr_from_swr"].value == pytest.approx(
        0.02
    )
    assert values["history_cross_heldout_animals_positive_swr_from_non_swr"].value == 4


def test_history_values_count_a_suppression_after_the_end():
    values = {
        v.key: v
        for v in duration_aware_history_values(_ridge(), _pooled(end_first_rr=0.2), "x")
    }
    assert values["history_end_first_lag_suppressed_fits"].value == 20
    assert values["history_end_first_lag_raised_fits"].value == 0


def test_history_values_leave_out_an_imprecise_first_lag():
    # wilbur-like: one fit's 15 ms lag has SE above HISTORY_MAX_SE; its extreme
    # rate ratio must not be read, and the denominator drops to 19.
    pooled = _pooled(imprecise_first_lag=("non_local_no_swr", "e"))
    values = {v.key: v for v in duration_aware_history_values(_ridge(), pooled, "x")}
    assert values["history_fits"].value == 19
    assert values["history_end_first_lag_raised_fits"].value == 19
    assert values["history_end_first_lag_rr_high"].value == pytest.approx(3.0)
    assert "19 of 20" in values["history_end_first_lag_rr_low"].description


def test_cross_values_leave_out_an_unsupported_lag():
    ridge = _ridge(unsupported=("non_local_swr", "a"))
    values = {v.key: v for v in duration_aware_history_values(ridge, _pooled(), "x")}
    assert values["history_cross_end_rr_swr_from_non_swr"].value == pytest.approx(2.5)
    assert values["history_cross_end_animals_swr_from_non_swr"].value == 4
    assert (
        "4 of 5 animals" in values["history_cross_end_rr_swr_from_non_swr"].description
    )
    # the held-out gain still counts every animal
    assert (
        "5 animals" in values["history_cross_heldout_gain_swr_from_non_swr"].description
    )


def test_cross_values_refuse_an_unconverged_ridge_fit():
    ridge = _ridge(unconverged=("non_local_no_swr", "b"))
    with pytest.raises(ValueError, match="did not converge"):
        duration_aware_history_values(ridge, _pooled(), "x")


def test_history_values_drop_an_unconverged_end_locked_fit_loudly():
    pooled = _pooled()
    pooled["results"][0]["duration_aware"]["Mself_dur"]["converged"] = False
    with pytest.raises(ValueError, match="did not converge"):
        duration_aware_history_values(_ridge(), pooled, "x")


def test_horserace_values_name_each_population_and_predictor():
    predictors = {
        "reward_belief_variance_candidate_arm_preoutcome_z": 0.03,
        "reward_belief_mean_candidate_arm_preoutcome_z": -0.001,
        "distance_to_candidate_arm_cm_z": -0.01,
        "visit_recency_trials_z": 0.002,
    }
    summary = pd.DataFrame(
        [
            {
                "population": pop,
                "predictor": p,
                "median": m,
                "n_animals_consistent": 4,
                "median_cond_number": 3.0 + i,
            }
            for i, pop in enumerate(("all", "SWR", "non_SWR"))
            for p, m in predictors.items()
        ]
    )
    values = {v.key: v for v in horserace_values(summary, "x")}
    assert values["horserace_uncertainty_swr"].value == pytest.approx(0.03)
    assert values["horserace_value_non_swr_animals"].value == 4
    assert values["horserace_condition_number_non_swr"].value == pytest.approx(5.0)
    assert len(values) == 3 * (4 * 2 + 1)
