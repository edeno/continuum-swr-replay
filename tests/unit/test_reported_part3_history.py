"""Part 3's duration-aware history and horse-race values, on synthetic inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.reported.part3 import (
    HISTORY_MAX_SE,
    HISTORY_SUPERSEDED,
    duration_aware_history_values,
    horserace_values,
    mark_superseded,
)
from continuum_swr_replay.reported.values import Summary, Value

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
    imprecise_cross_lag: tuple[str, str] | None = None,
    unconverged_cross: tuple[str, str] | None = None,
) -> dict:
    """Every animal: an end-locked kernel peaking at 30 ms, an onset-locked one
    peaking at 105 ms (rate ratio ``onset_peak_rr``), and a cross end kernel of
    rate ratio 2.5 at 45 ms. Onset lags 1-3 have no support (not measurable).

    ``imprecise_first_lag`` / ``imprecise_cross_lag`` = (population, animal)
    whose 15 ms end-locked / 45 ms cross lag has a Wald SE above
    ``HISTORY_MAX_SE`` (and an extreme rate ratio that must not be read);
    ``unconverged_cross`` = (population, animal) whose cross rung did not
    converge (its kernel is extreme and must not be read)."""
    end = np.full(10, 0.0)
    end[0], end[1] = np.log(end_first_rr), np.log(4.0)
    onset = np.full(10, 0.1)
    onset[6] = np.log(onset_peak_rr)
    onset[0] = 5.0  # unsupported: must not be read
    cross = np.zeros(10)
    cross[2] = np.log(2.5)
    support = {
        "self_onset": [0.0, 1.0, 2.0] + [10.0] * 7,
        "self_offset": [20.0] * 10,
        "cross_offset": [20.0] * 10,
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
            if pop in ("non_local_swr", "non_local_no_swr"):
                cross_k = _kernel(cross)
                if (pop, animal) == imprecise_cross_lag:
                    lr = cross.copy()
                    lr[2] = np.log(50.0)
                    cross_k = _kernel(
                        lr, [0.1] * 2 + [HISTORY_MAX_SE + 0.01] + [0.1] * 7
                    )
                converged = (pop, animal) != unconverged_cross
                if not converged:
                    cross_k = _kernel(np.full(10, np.log(50.0)))
                fits["Mselfcross_dur"] = {
                    "ok": True,
                    "converged": converged,
                    "kernels": {"cross_offset": cross_k},
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


def _sessions(gain: float, unusable_gain: float | None = None) -> pd.DataFrame:
    """Three sessions per animal and stream whose cross gain is ``gain``; with
    ``unusable_gain``, a fourth per animal whose cross rung did not converge and
    a fifth whose Mself_dur is separated, both of gain ``unusable_gain``."""
    rows = []
    for pop in ("non_local_swr", "non_local_no_swr"):
        for animal in ANIMALS:
            for s in range(3 if unusable_gain is None else 5):
                key = {
                    "animal": animal,
                    "session_date": s,
                    "epoch": "02_r1",
                    "outcome": pop,
                    "fit_ok": True,
                }
                g = gain if s < 3 else unusable_gain
                rows.append(
                    {
                        **key,
                        "model": "Mself_dur",
                        "aic": 100.0,
                        "converged": True,
                        "separated": s == 4,
                    }
                )
                rows.append(
                    {
                        **key,
                        "model": "Mselfcross_dur",
                        "aic": 100.0 - g,
                        "converged": s != 3,
                        "separated": False,
                    }
                )
    return pd.DataFrame(rows)


def test_history_values_read_measurable_lags_and_count_animals():
    values = {
        v.key: v for v in duration_aware_history_values(_sessions(-6.0), _pooled(), "x")
    }
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
    # negative gain: the cross block does not pay for its parameters
    assert values["history_cross_session_gain_non_swr_from_swr"].value == pytest.approx(
        -6.0
    )
    assert values["history_cross_session_animals_positive_non_swr_from_swr"].value == 0


def test_history_values_count_a_suppression_after_the_end():
    values = {
        v.key: v
        for v in duration_aware_history_values(
            _sessions(1.0), _pooled(end_first_rr=0.2), "x"
        )
    }
    assert values["history_end_first_lag_suppressed_fits"].value == 20
    assert values["history_end_first_lag_raised_fits"].value == 0
    assert values["history_cross_session_animals_positive_swr_from_non_swr"].value == 5


def test_history_values_leave_out_an_imprecise_first_lag():
    # wilbur-like: one fit's 15 ms lag has SE above HISTORY_MAX_SE; its extreme
    # rate ratio must not be read, and the denominator drops to 19.
    pooled = _pooled(imprecise_first_lag=("non_local_no_swr", "e"))
    values = {
        v.key: v for v in duration_aware_history_values(_sessions(-6.0), pooled, "x")
    }
    assert values["history_fits"].value == 19
    assert values["history_end_first_lag_raised_fits"].value == 19
    assert values["history_end_first_lag_rr_high"].value == pytest.approx(3.0)
    assert "19 of 20" in values["history_end_first_lag_rr_low"].description


def test_history_values_leave_out_an_imprecise_cross_lag():
    pooled = _pooled(imprecise_cross_lag=("non_local_swr", "a"))
    values = {
        v.key: v for v in duration_aware_history_values(_sessions(-6.0), pooled, "x")
    }
    assert values["history_cross_end_rr_swr_from_non_swr"].value == pytest.approx(2.5)
    assert values["history_cross_end_animals_swr_from_non_swr"].value == 4
    assert (
        "4 of 5 animals" in values["history_cross_end_rr_swr_from_non_swr"].description
    )
    assert values["history_cross_end_rr_swr_from_non_swr"].pending is None


def test_history_values_pend_a_direction_with_an_unconverged_cross_fit():
    pooled = _pooled(unconverged_cross=("non_local_no_swr", "b"))
    values = {
        v.key: v for v in duration_aware_history_values(_sessions(-6.0), pooled, "x")
    }
    rr = values["history_cross_end_rr_non_swr_from_swr"]
    assert rr.value == pytest.approx(2.5)  # the unconverged kernel is not read
    assert values["history_cross_end_animals_non_swr_from_swr"].value == 4
    assert rr.pending is not None and "did not converge" in rr.pending
    assert values["history_cross_end_animals_non_swr_from_swr"].pending == rr.pending
    # the other direction and the per-session gain are unaffected
    assert values["history_cross_end_rr_swr_from_non_swr"].pending is None
    assert values["history_cross_session_gain_non_swr_from_swr"].pending is None


def test_history_values_drop_an_unconverged_end_locked_fit_loudly():
    pooled = _pooled()
    pooled["results"][0]["duration_aware"]["Mself_dur"]["converged"] = False
    with pytest.raises(ValueError, match="did not converge"):
        duration_aware_history_values(_sessions(-6.0), pooled, "x")


def test_cross_session_gain_reads_only_usable_sessions():
    # Unconverged or separated sessions carry a gain of +50 that must not count.
    values = {
        v.key: v
        for v in duration_aware_history_values(_sessions(-6.0, 50.0), _pooled(), "x")
    }
    gain = values["history_cross_session_gain_non_swr_from_swr"]
    assert gain.value == pytest.approx(-6.0)
    assert "15 of 25 sessions" in gain.description
    assert values["history_cross_session_animals_positive_non_swr_from_swr"].value == 0


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


def test_mark_superseded_points_old_history_values_to_the_refit():
    summary = Summary(part=3, produced_by="x", commit="x")
    common = {
        "kind": "quantity",
        "style": "dec2",
        "unit": "",
        "description": "d",
        "claims": ("19c",),
        "used_in": ("paper",),
        "source": "s",
    }
    summary.add(
        Value(key="rate_burst_peak_swr", value=2.0, **common),
        Value(key="rate_refractory_ms_swr", value=60, **common),
        Value(key="rate_kernel_lag_one_swr", value=1.4, **common),
    )
    mark_superseded(summary)
    assert summary.values["rate_burst_peak_swr"].pending == HISTORY_SUPERSEDED
    assert summary.values["rate_refractory_ms_swr"].pending == HISTORY_SUPERSEDED
    assert summary.values["rate_kernel_lag_one_swr"].pending is None
