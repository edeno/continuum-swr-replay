"""The behavior summary's duration-aware history values, on synthetic inputs."""

from __future__ import annotations

import numpy as np
import pytest

from continuum_swr_replay.analysis.behavior.rate.history_kernel_statistics import (
    HISTORY_MAX_SE,
)
from continuum_swr_replay.reported.behavior.history import duration_aware_history_values

ANIMALS = ("a", "b", "c", "d", "e")
# Per population: the end-locked first-lag rate ratio (as a multiple of
# ``end_first_rr``), the end-locked peak and its lag index, the onset-locked
# peak and its lag index.
POOLED = {
    "swr": (1.0, 4.0, 1, 2.0, 6),
    "non_local": (1.1, 3.8, 1, 3.0, 8),
    "non_local_swr": (1.2, 6.0, 1, 4.0, 6),
    "non_local_no_swr": (0.9, 7.0, 3, 5.0, 6),
}


def _kernel(log_rr, se=0.1):
    se = [se] * 10 if np.isscalar(se) else list(se)
    return {"log_rr": list(log_rr), "se": se, "coef": [0.0] * 4}


def _pooled(
    end_first_rr: float = 3.0,
    *,
    imprecise_first_lag: tuple[str, str] | None = None,
) -> dict:
    """Every animal of a population: an end-locked kernel and an onset-locked
    one with that population's peaks (``POOLED``). Onset lags 1-3 have no
    support (not measurable).

    Three fits differ. non_local, animals a and b: every end-locked lag past the
    first has SE 0.5, so the peak of 3.8 (2.67 SE above 1) does not clear the
    interval widened for ten lags (2.81 SE). swr, animal a: no onset-locked lag
    has support. swr, animal b: every onset-locked SE is 0.5, so its peak of 2
    is not clear of 1.

    ``imprecise_first_lag`` = (population, animal) whose 15 ms end-locked lag
    has a Wald SE above ``HISTORY_MAX_SE`` (and an extreme rate ratio that must
    not be read)."""
    results = []
    for pop, (first, end_peak, end_at, onset_peak, onset_at) in POOLED.items():
        end = np.full(10, 0.0)
        end[0], end[end_at] = np.log(end_first_rr * first), np.log(end_peak)
        onset = np.full(10, 0.1)
        onset[onset_at] = np.log(onset_peak)
        onset[0] = 5.0  # unsupported: must not be read
        for animal in ANIMALS:
            end_se = [0.1] * 10
            if pop == "non_local" and animal in ("a", "b"):
                end_se = [0.1] + [0.5] * 9
            end_k = _kernel(end, end_se)
            if (pop, animal) == imprecise_first_lag:
                lr = end.copy()
                lr[0] = np.log(50.0)
                end_k = _kernel(lr, [HISTORY_MAX_SE + 0.01] + [0.1] * 9)
            onset_support = [0.0, 1.0, 2.0] + [10.0] * 7
            if (pop, animal) == ("swr", "a"):
                onset_support = [0.0] * 10
            onset_se = 0.5 if (pop, animal) == ("swr", "b") else 0.1
            fits = {
                "Mself_dur": {
                    "ok": True,
                    "converged": True,
                    "kernels": {
                        "self_offset": end_k,
                        "self_onset": _kernel(onset, onset_se),
                    },
                }
            }
            results.append(
                {
                    "animal": animal,
                    "outcome": pop,
                    "support": {
                        "self_onset": onset_support,
                        "self_offset": [20.0] * 10,
                    },
                    "duration_aware": fits,
                }
            )
    return {"stop_cohort": "at_well", "results": results}


# Per response stream: the cross kernel's rate ratio at 45 ms and each animal's
# held-out gain.
RIDGE = {
    "non_local_swr": (2.5, (0.01, 0.02, 0.03, -0.01, 0.04)),
    "non_local_no_swr": (1.5, (-0.01, 0.0, 0.01, -0.03, 0.02)),
}


def _ridge(
    *,
    unsupported: tuple[str, str] | None = None,
    unconverged: tuple[str, str] | None = None,
) -> dict:
    """Per animal and non-local stream (``RIDGE``): a held-out gain and a 1 s
    end-locked cross kernel with the stream's rate ratio at 45 ms (posterior SD
    0.1); the self-only rung has no cross kernel. ``unsupported`` = (stream,
    animal) whose 45 ms lag has fewer than 3 expected onsets (an extreme rate
    ratio that must not be read); ``unconverged`` = (stream, animal) whose
    self+cross fit did not converge."""
    results = []
    for pop, (cross_rr, gains) in RIDGE.items():
        for animal, gain in zip(ANIMALS, gains, strict=True):
            lr = np.zeros(67)
            lr[2] = np.log(cross_rr)
            support = np.full(67, 20.0)
            if (pop, animal) == unsupported:
                lr[2], support[2] = np.log(50.0), 1.0
            kernel = {"log_rr": lr.tolist(), "sd": [0.1] * 67, "coef": [0.0] * 6}
            models = {
                "Mself_long_dur": {
                    "full_converged": True,
                    "cv": {"n_nonconverged": 0},
                    "kernels": {},
                },
                "Mselfcross_long_dur": {
                    "full_converged": True,
                    "cv": {"n_nonconverged": 0},
                    "kernels": {"cross_offset": kernel},
                },
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
    assert values["history_onset_peak_non_local"].value == pytest.approx(3.0)
    assert values["history_onset_peak_lag_ms_non_local"].value == 135
    assert values["history_onset_peak_non_local_no_swr"].value == pytest.approx(5.0)
    # swr: animal a has no measurable onset-locked lag, and animal b's peak is
    # not clear of 1
    assert values["history_onset_animals_swr"].value == 3
    assert "of the 4 of 5 animals" in values["history_onset_animals_swr"].description
    assert "over the 4 of 5 animals" in values["history_onset_peak_swr"].description
    assert (
        "over the 4 of 5 animals" in values["history_onset_peak_lag_ms_swr"].description
    )
    assert values["history_onset_animals_non_local"].value == 5
    assert "the 5 of 5 animals" in values["history_onset_peak_non_local"].description
    assert values["history_end_peak_low"].value == pytest.approx(3.8)
    assert values["history_end_peak_high"].value == pytest.approx(7.0)
    assert values["history_end_peak_lag_ms_low"].value == 30
    assert values["history_end_peak_lag_ms_high"].value == 60
    # non_local: the end-locked peaks of animals a and b are not clear of 1
    assert values["history_end_peak_animals_min"].value == 3
    for key in ("low", "high", "lag_ms_low", "lag_ms_high", "animals_min"):
        assert "20 of the 20" in values[f"history_end_peak_{key}"].description
    assert values["history_end_first_lag_rr_low"].value == pytest.approx(2.7)
    assert values["history_end_first_lag_rr_high"].value == pytest.approx(3.6)
    assert values["history_end_first_lag_suppressed_fits"].value == 0
    assert values["history_end_first_lag_raised_fits"].value == 20
    assert values["history_fits"].value == 20
    # the rate after SWR-coupled events is the non-SWR stream's response
    assert values["history_cross_end_rr_non_swr_from_swr"].value == pytest.approx(1.5)
    assert values["history_cross_end_rr_swr_from_non_swr"].value == pytest.approx(2.5)
    assert values["history_cross_end_animals_swr_from_non_swr"].value == 5
    assert values["history_cross_end_rr_non_swr_from_swr"].pending is None
    assert values["history_cross_heldout_gain_non_swr_from_swr"].value == pytest.approx(
        0.0
    )
    assert values["history_cross_heldout_gain_swr_from_non_swr"].value == pytest.approx(
        0.02
    )
    assert values["history_cross_heldout_animals_positive_swr_from_non_swr"].value == 4
    assert values["history_cross_heldout_animals_positive_non_swr_from_swr"].value == 2


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
    # its population's median is over the other four animals
    assert values["history_end_first_lag_rr_low"].value == pytest.approx(2.7)
    assert values["history_end_first_lag_rr_high"].value == pytest.approx(3.6)
    assert "19 of 20" in values["history_end_first_lag_rr_low"].description
    # its other lags are measurable, so it still has an end-locked peak
    assert "20 of the 20" in values["history_end_peak_low"].description


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
