"""Planted-kernel recovery of the marked family with per-stop baseline terms.

One generator serves every test: stops of fitted bins at alternating rewarded
and unrewarded outcomes, replay events of both classes inside them, and a
log-rate that is a constant baseline, optionally raised by a stationary factor
on rewarded stops, plus a planted per-SD kernel on one class's mark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.design import event_basis_functions
from continuum_swr_replay.analysis.pfc.heldout import assign_blocked_folds
from continuum_swr_replay.analysis.pfc.marked_glm import (
    EVENT_DURATION_MARK,
    MARKED_EVENT_CLASSES,
    build_baseline_columns,
)
from continuum_swr_replay.analysis.pfc.marked_model_family import (
    CURRENT_ARM_FRACTION,
    REWARDED_CONTEXT,
    MarkedFamilySpec,
    build_marked_family_ladder,
    fit_marked_family_ladder,
)

SAMPLING_FREQUENCY = 500.0
DRIVEN_CLASS = "non_local_no_SWR"
PLANTED_PEAK = 0.9


@dataclass(frozen=True)
class Simulation:
    """One simulated session: inputs, counts and the planted truth."""

    fit_mask: pd.Series
    stops: pd.DataFrame
    events: dict[str, pd.DataFrame]
    spline_basis: np.ndarray
    lags: np.ndarray
    lag_time: np.ndarray
    counts: np.ndarray
    folds: np.ndarray
    peak_lag_index: int


def simulate(
    seed: int,
    *,
    planted_mark: str = CURRENT_ARM_FRACTION,
    rewarded_rate_factor: float = 1.0,
    n_stops: int = 40,
    n_units: int = 40,
    stop_bins: int = 600,
    gap_bins: int = 100,
    rate_hz: float = 25.0,
) -> Simulation:
    """Simulate stops, events of both replay classes and Poisson counts.

    Events sit every 120 bins inside each stop, alternating class. Every unit
    shares the log-rate: ``log(rate_hz)`` plus ``log(rewarded_rate_factor)``
    on every bin of a rewarded stop, plus ``z_e * planted(t - onset_e)`` at
    each driven-class event, where ``z_e`` is the event's within-class
    standardized ``planted_mark``. Nothing else is event-locked.
    """
    rng = np.random.default_rng(seed)
    n_time = n_stops * (stop_bins + gap_bins)
    time = np.arange(n_time) / SAMPLING_FREQUENCY
    fitted = np.zeros(n_time, dtype=bool)
    rewarded = np.resize([1.0, 0.0], n_stops)
    starts, ends, onsets, stop_of_event = [], [], [], []
    for stop in range(n_stops):
        first = stop * (stop_bins + gap_bins)
        fitted[first : first + stop_bins] = True
        starts.append(time[first])
        ends.append(time[first + stop_bins - 1])
        for onset in first + 60 + np.arange(0, stop_bins - 120, 120):
            onsets.append(onset)
            stop_of_event.append(stop)
    onsets = np.asarray(onsets)
    stop_of_event = np.asarray(stop_of_event)
    stops = pd.DataFrame(
        {
            "stop_idx": np.arange(n_stops),
            "stop_start": starts,
            "stop_end": ends,
            "occupancy_well_id": 0.0,
            "is_rewarded_outcome": rewarded,
        }
    )
    fit_mask = pd.Series(fitted, index=time)
    lags = np.arange(-30, 31)
    basis, lag_time, _ = event_basis_functions(lags, SAMPLING_FREQUENCY)

    classes = np.resize(np.array(MARKED_EVENT_CLASSES), len(onsets))
    marks = {
        CURRENT_ARM_FRACTION: rng.uniform(0.0, 1.0, len(onsets)),
        "ripple_power_z": rng.normal(1.0, 1.0, len(onsets)),
        EVENT_DURATION_MARK: np.log(rng.uniform(0.02, 0.2, len(onsets))),
        REWARDED_CONTEXT: rewarded[stop_of_event],
    }
    events = {
        event_class: pd.DataFrame(
            {
                "event_id": np.flatnonzero(classes == event_class),
                "start_time": time[onsets[classes == event_class]],
                "stop_idx": stop_of_event[classes == event_class],
                **{
                    name: values[classes == event_class]
                    for name, values in marks.items()
                },
            }
        )
        for event_class in MARKED_EVENT_CLASSES
    }

    gaussian = np.exp(-0.5 * ((lag_time - 0.025) / 0.025) ** 2)
    planted = basis @ np.linalg.lstsq(basis, gaussian, rcond=None)[0]
    planted *= PLANTED_PEAK / planted.max()
    driven = classes == DRIVEN_CLASS
    values = marks[planted_mark][driven]
    standardized = (values - values.mean()) / values.std(ddof=0)
    log_rate = np.full(n_time, np.log(rate_hz / SAMPLING_FREQUENCY))
    for stop in np.flatnonzero(rewarded):
        first = stop * (stop_bins + gap_bins)
        log_rate[first : first + stop_bins] += np.log(rewarded_rate_factor)
    for onset, mark_z in zip(onsets[driven], standardized, strict=True):
        np.add.at(log_rate, np.clip(onset + lags, 0, n_time - 1), mark_z * planted)
    counts = rng.poisson(np.exp(log_rate)[:, None], size=(n_time, n_units))[fitted]
    folds = assign_blocked_folds(
        fit_mask, n_folds=5, seed=12, support_radius_bins=int(np.abs(lags).max())
    )
    return Simulation(
        fit_mask=fit_mask,
        stops=stops,
        events=events,
        spline_basis=basis,
        lags=lags,
        lag_time=lag_time,
        counts=counts.astype(float),
        folds=folds,
        peak_lag_index=int(np.argmax(planted)),
    )


def fit(simulation: Simulation, spec: MarkedFamilySpec, *, baseline: bool, **kwargs):
    """Build and fit one spec on a simulation, with or without the baseline."""
    columns = (
        build_baseline_columns(simulation.fit_mask, simulation.stops)[0]
        if baseline
        else None
    )
    ladder = build_marked_family_ladder(
        simulation.events,
        simulation.fit_mask,
        simulation.spline_basis,
        simulation.lags,
        spec,
        baseline=columns,
    )
    return fit_marked_family_ladder(
        ladder,
        simulation.counts,
        simulation.folds,
        penalty_sigma=3.0,
        design_batch_size=2,
        **kwargs,
    )


@pytest.mark.slow
def test_planted_kernel_is_recovered_with_a_context_offset_present():
    simulation = simulate(20261027, rewarded_rate_factor=1.5)
    spec = MarkedFamilySpec(
        "context_offset",
        (CURRENT_ARM_FRACTION, REWARDED_CONTEXT),
        (EVENT_DURATION_MARK,),
        "is a stationary reward offset kept out of the reward kernel?",
    )
    pre_onset = simulation.lag_time <= -0.03

    with_baseline = fit(simulation, spec, baseline=True)
    without_baseline = fit(simulation, spec, baseline=False)

    recovered = with_baseline.mark_curves[DRIVEN_CLASS][CURRENT_ARM_FRACTION]
    assert recovered[simulation.peak_lag_index].mean() == pytest.approx(
        PLANTED_PEAK, abs=0.35
    )
    # Averaged over 40 units the estimator's own spread on these pre-onset
    # means is about 0.02, so the tolerance is 0.05, not tighter.
    for event_class in MARKED_EVENT_CLASSES:
        reward = with_baseline.mark_curves[event_class][REWARDED_CONTEXT]
        assert abs(reward[pre_onset].mean()) < 0.05
        confounded = without_baseline.mark_curves[event_class][REWARDED_CONTEXT]
        assert confounded[pre_onset].mean() > 0.1
