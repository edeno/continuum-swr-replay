"""Planted-kernel recovery of the marked family with per-stop baseline terms.

One generator serves every test: stops of fitted bins at alternating rewarded
and unrewarded outcomes, replay events of both classes inside them, and a
log-rate that is a constant baseline, optionally raised by a stationary factor
on rewarded stops or on mixed stops (no well), plus a planted per-SD kernel on
one class's mark.
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
from continuum_swr_replay.analysis.replay.fit_provenance import fit_provenance_path

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
    mixed_rate_factor: float | None = None,
    onset_offset_s: float = 0.0,
    half_window_bins: int = 30,
    event_spacing_bins: int = 120,
) -> Simulation:
    """Simulate stops, events of both replay classes and Poisson counts.

    Events sit every ``event_spacing_bins`` bins inside each stop, alternating
    class; kernels span lags of ``half_window_bins`` either side. Every unit
    shares the log-rate: ``log(rate_hz)`` plus ``log(rewarded_rate_factor)``
    on every bin of a rewarded stop, plus ``z_e * planted(t - onset_e)`` at
    each driven-class event, where ``z_e`` is the event's within-class
    standardized ``planted_mark``. Nothing else is event-locked.

    With ``mixed_rate_factor``, every stop whose index is 2 or 3 modulo 4 is a
    mixed stop: no ``occupancy_well_id`` (NaN), ``log(mixed_rate_factor)``
    added on its bins, and every event slot used, while the other stops
    (well 0) keep only their first slot, so events are denser in mixed stops.
    ``onset_offset_s`` moves the first event's recorded ``start_time`` off
    the sample grid (inside its bin); the counts still use its bin.
    """
    rng = np.random.default_rng(seed)
    n_time = n_stops * (stop_bins + gap_bins)
    time = np.arange(n_time) / SAMPLING_FREQUENCY
    fitted = np.zeros(n_time, dtype=bool)
    rewarded = np.resize([1.0, 0.0], n_stops)
    mixed = (
        np.isin(np.arange(n_stops) % 4, (2, 3))
        if mixed_rate_factor is not None
        else np.zeros(n_stops, dtype=bool)
    )
    starts, ends, onsets, stop_of_event = [], [], [], []
    for stop in range(n_stops):
        first = stop * (stop_bins + gap_bins)
        fitted[first : first + stop_bins] = True
        starts.append(time[first])
        ends.append(time[first + stop_bins - 1])
        slots = first + 60 + np.arange(0, stop_bins - 120, event_spacing_bins)
        if mixed_rate_factor is not None and not mixed[stop]:
            slots = slots[:1]
        for onset in slots:
            onsets.append(onset)
            stop_of_event.append(stop)
    onsets = np.asarray(onsets)
    stop_of_event = np.asarray(stop_of_event)
    stops = pd.DataFrame(
        {
            "stop_idx": np.arange(n_stops),
            "stop_start": starts,
            "stop_end": ends,
            "occupancy_well_id": np.where(mixed, np.nan, 0.0),
            "is_rewarded_outcome": rewarded,
        }
    )
    fit_mask = pd.Series(fitted, index=time)
    lags = np.arange(-half_window_bins, half_window_bins + 1)
    basis, lag_time, _ = event_basis_functions(lags, SAMPLING_FREQUENCY)

    classes = np.resize(np.array(MARKED_EVENT_CLASSES), len(onsets))
    marks = {
        CURRENT_ARM_FRACTION: rng.uniform(0.0, 1.0, len(onsets)),
        "ripple_power_z": rng.normal(1.0, 1.0, len(onsets)),
        EVENT_DURATION_MARK: np.log(rng.uniform(0.02, 0.2, len(onsets))),
        REWARDED_CONTEXT: rewarded[stop_of_event],
    }
    start_time = time[onsets].copy()
    start_time[0] += onset_offset_s
    events = {
        event_class: pd.DataFrame(
            {
                "event_id": np.flatnonzero(classes == event_class),
                "start_time": start_time[classes == event_class],
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
    for stop in np.flatnonzero(mixed):
        first = stop * (stop_bins + gap_bins)
        log_rate[first : first + stop_bins] += np.log(mixed_rate_factor)
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


def fit(
    simulation: Simulation,
    spec: MarkedFamilySpec,
    *,
    baseline: bool,
    without_columns: tuple[str, ...] = (),
    **kwargs,
):
    """Build and fit one spec on a simulation, with or without the baseline.

    ``without_columns`` names baseline columns to leave out of the design.
    """
    columns = None
    if baseline:
        columns, names, _ = build_baseline_columns(
            simulation.fit_mask, simulation.stops
        )
        columns = columns[:, [name not in without_columns for name in names]]
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


@pytest.mark.slow
def test_mixed_stop_offset_moves_neither_kernel_nor_reference_level():
    """Mixed stops (no well) at twice the rate, with events denser in them.

    With ``well_unknown`` the offset goes to that column: the intercept is the
    reference well's level and the ordinary kernels stay at their true zero.
    Without it the mixed stops are coded as the reference well, so the
    intercept is raised and the kernels read the offset as event-locked.
    """
    simulation = simulate(20261030, mixed_rate_factor=2.0)
    spec = MarkedFamilySpec(
        "mixed_offset",
        (CURRENT_ARM_FRACTION,),
        (EVENT_DURATION_MARK,),
        "is a stationary mixed-stop offset kept out of the kernels?",
    )
    names = build_baseline_columns(simulation.fit_mask, simulation.stops)[1]
    assert names == ("well_unknown", "rewarded_outcome")
    true_level = np.log(25.0 / SAMPLING_FREQUENCY)
    options = {"baseline": True, "baseline_penalty_sigma": 3.0}

    with_column = fit(simulation, spec, **options)
    without_column = fit(simulation, spec, without_columns=("well_unknown",), **options)

    assert with_column.full_params[:, 0].mean() == pytest.approx(true_level, abs=0.03)
    assert with_column.full_params[:, 1].mean() == pytest.approx(np.log(2.0), abs=0.03)
    assert without_column.full_params[:, 0].mean() - true_level > 0.15
    for event_class in MARKED_EVENT_CLASSES:
        assert abs(np.nanmean(with_column.typical_curves[event_class])) < 0.03
        assert np.nanmean(without_column.typical_curves[event_class]) > 0.08
    recovered = with_column.mark_curves[DRIVEN_CLASS][CURRENT_ARM_FRACTION]
    curve = np.nanmean(recovered, axis=1)
    assert curve[simulation.peak_lag_index] == pytest.approx(PLANTED_PEAK, abs=0.35)
    assert abs(int(np.argmax(curve)) - simulation.peak_lag_index) <= 1


def _window_mean_and_band(result, event_class: str, mark: str):
    """Population window-mean of the slope kernel and its 95 % null band.

    The population curve is the mean over units; its mean over the whole
    window is compared with the same statistic of each shuffled refit.
    """
    observed = float(np.nanmean(result.mark_curves[event_class][mark]))
    null = result.slope_null[event_class][mark]  # (n_draws, n_lags, n_units)
    draws = np.nanmean(null, axis=(1, 2))
    return observed, np.percentile(draws, [2.5, 97.5])


def _window_z(result, event_class: str, mark: str) -> float:
    """Observed window mean in SDs of the shuffled refits' window means."""
    observed = float(np.nanmean(result.mark_curves[event_class][mark]))
    draws = np.nanmean(result.slope_null[event_class][mark], axis=(1, 2))
    return (observed - draws.mean()) / draws.std(ddof=1)


@pytest.mark.slow
def test_planted_context_kernel_is_separated_from_its_baseline():
    simulation = simulate(20261028, rewarded_rate_factor=1.5)
    spec = MarkedFamilySpec(
        "context_reference",
        (REWARDED_CONTEXT,),
        (EVENT_DURATION_MARK, CURRENT_ARM_FRACTION),
        "is a stationary reward offset read as an event-locked reward effect?",
    )
    options = {
        "slope_permutations": 20,
        "slope_permutation_seed": 3,
        "slope_permutation_targets": (REWARDED_CONTEXT,),
    }

    with_baseline = fit(simulation, spec, baseline=True, **options)
    without_baseline = fit(simulation, spec, baseline=False, **options)

    for event_class in MARKED_EVENT_CLASSES:
        # With the baseline the reward kernel sits with its stop-shuffled
        # reference. The bound is 3.5 reference SDs, not the 95 % band: on
        # simulated sessions with this stationary offset the observed window
        # mean scattered somewhat more widely than the shuffled refits (z RMS
        # about 1.2 over 16 class x seed fits, one |z| = 3.0), against about
        # 0.8 with no offset, so the band can be crossed by chance here. The
        # residualized reference does not change that (here the SWR class
        # sits at z -2.3, outside its band): the reward mark is unrelated to
        # the other marks by construction, so its extra spread does not come
        # from correlated marks.
        assert abs(_window_z(with_baseline, event_class, REWARDED_CONTEXT)) < 3.5
        # Without it the stationary offset is read as a reward kernel outside
        # the reference.
        observed, (_, high) = _window_mean_and_band(
            without_baseline, event_class, REWARDED_CONTEXT
        )
        assert observed > high


@pytest.mark.slow
def test_planted_strength_kernel_is_recovered():
    # One event's onset lies between two samples (0.4 of a bin after its
    # sample); it is binned with that sample, as the counts are.
    simulation = simulate(
        20261029,
        planted_mark="ripple_power_z",
        onset_offset_s=0.4 / SAMPLING_FREQUENCY,
    )
    spec = MarkedFamilySpec(
        "strength_reference",
        ("ripple_power_z", CURRENT_ARM_FRACTION),
        (EVENT_DURATION_MARK,),
        "is a planted ripple-power gain recovered and outside its reference?",
    )

    result = fit(
        simulation,
        spec,
        baseline=True,
        slope_permutations=20,
        slope_permutation_seed=5,
        slope_permutation_targets=("ripple_power_z", CURRENT_ARM_FRACTION),
    )

    peak = simulation.peak_lag_index
    curve = np.nanmean(result.mark_curves[DRIVEN_CLASS]["ripple_power_z"], axis=1)
    # Sign and peak lag of the planted kernel (within one 2 ms bin).
    assert curve[peak] == pytest.approx(PLANTED_PEAK, abs=0.35)
    assert abs(int(np.argmax(curve)) - peak) <= 1
    null = np.nanmean(result.slope_null[DRIVEN_CLASS]["ripple_power_z"], axis=2)
    assert abs(null[:, peak].mean()) < 0.15
    assert curve[peak] > np.percentile(null[:, peak], 97.5)
    # The unplanted target, and the planted mark on the other class, stay
    # inside their references.
    for event_class, mark in (
        (DRIVEN_CLASS, CURRENT_ARM_FRACTION),
        ("non_local_SWR", "ripple_power_z"),
    ):
        observed, (low, high) = _window_mean_and_band(result, event_class, mark)
        assert low <= observed <= high


def _correlated_marks_replicate(seed: int, n_draws: int):
    """One session where the target is correlated with an adjustment (r 0.75).

    The adjustment (``mua_rate_z``) carries a planted per-SD kernel and the
    target (``ripple_power_z``) none. Returns the target's observed
    population window mean and its two references' draws of the same
    statistic.
    """
    from continuum_swr_replay.analysis.pfc.marked_model_family import (
        MUA_RATE,
        RIPPLE_POWER,
        fit_marked_family_slope_null,
    )

    event_class = DRIVEN_CLASS
    rng = np.random.default_rng(seed)
    n_stops, stop_bins, gap_bins, n_units = 30, 600, 100, 20
    n_time = n_stops * (stop_bins + gap_bins)
    time = np.arange(n_time) / SAMPLING_FREQUENCY
    fitted = np.zeros(n_time, dtype=bool)
    onsets = []
    for stop in range(n_stops):
        first = stop * (stop_bins + gap_bins)
        fitted[first : first + stop_bins] = True
        onsets.extend(first + 60 + np.arange(0, stop_bins - 120, 60))
    onsets = np.asarray(onsets)
    fit_mask = pd.Series(fitted, index=time)
    lags = np.arange(-30, 31)
    basis, lag_time, _ = event_basis_functions(lags, SAMPLING_FREQUENCY)
    other = rng.normal(size=len(onsets))
    target = 0.75 * other + np.sqrt(1 - 0.75**2) * rng.normal(size=len(onsets))
    events = {
        event_class: pd.DataFrame(
            {
                "event_id": np.arange(len(onsets)),
                "start_time": time[onsets],
                RIPPLE_POWER: target,
                MUA_RATE: other,
            }
        )
    }
    kernel = (
        basis
        @ np.linalg.lstsq(
            basis, np.exp(-0.5 * ((lag_time - 0.02) / 0.03) ** 2), rcond=None
        )[0]
    )
    kernel *= 0.5 / kernel.max()
    log_rate = np.full(n_time, np.log(25.0 / SAMPLING_FREQUENCY))
    other_z = (other - other.mean()) / other.std()
    for onset, mark_z in zip(onsets, other_z, strict=True):
        np.add.at(
            log_rate, np.clip(onset + lags, 0, n_time - 1), (mark_z + 0.3) * kernel
        )
    counts = rng.poisson(np.exp(log_rate)[:, None], size=(n_time, n_units))[fitted]
    folds = assign_blocked_folds(fit_mask, n_folds=5, seed=1, support_radius_bins=30)
    spec = MarkedFamilySpec(
        "correlated_marks",
        (RIPPLE_POWER,),
        (MUA_RATE,),
        "is the reference as wide as the estimator?",
        event_classes=(event_class,),
    )
    ladder = build_marked_family_ladder(events, fit_mask, basis, lags, spec)
    observed = fit_marked_family_ladder(
        ladder, counts.astype(float), folds, penalty_sigma=3.0
    )
    statistic = float(np.nanmean(observed.mark_curves[event_class][RIPPLE_POWER]))
    references = {}
    for method in ("residualized_within_class", "raw_within_class"):
        null, diagnostics = fit_marked_family_slope_null(
            ladder,
            counts.astype(float),
            observed,
            penalty_sigma=3.0,
            n_draws=n_draws,
            seed=seed,
            method=method,
        )
        references[method] = np.nanmean(null[event_class][RIPPLE_POWER], axis=(1, 2))
        if method == "residualized_within_class":
            r_squared = diagnostics["slope_null_target_r_squared"][
                f"{event_class}:{RIPPLE_POWER}"
            ][event_class]
    return statistic, references, r_squared


@pytest.mark.slow
def test_residualized_reference_is_as_wide_as_the_estimator_under_correlated_marks():
    """The reference's spread matches the estimator's; the raw one's does not.

    Over 40 simulated sessions the across-session SD of the target's observed
    window mean is the estimator's sampling SD under no target effect. A raw
    permutation decorrelates the target from the correlated adjustment, so its
    draws are narrower by about sqrt(1 - R^2) (0.66 at r = 0.75); the
    residualized permutation keeps the correlation (measured at these seeds:
    estimator SD 0.0100, residualized reference 0.0103, raw 0.0071).
    """
    replicates = [_correlated_marks_replicate(seed, n_draws=40) for seed in range(40)]
    estimator_sd = np.std([observed for observed, _, _ in replicates], ddof=1)
    residualized_sd = np.mean(
        [
            references["residualized_within_class"].std(ddof=1)
            for _, references, _ in replicates
        ]
    )
    raw_sd = np.mean(
        [references["raw_within_class"].std(ddof=1) for _, references, _ in replicates]
    )
    r_squared = np.mean([r2 for _, _, r2 in replicates])

    assert r_squared == pytest.approx(0.5625, abs=0.1)
    assert residualized_sd / estimator_sd == pytest.approx(1.0, abs=0.2)
    assert raw_sd / estimator_sd < 0.85


def _load_runner():
    """Import the marked-family runner script as a module."""
    import importlib.util
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "pfc"
        / "run_pfc_marked_model_family.py"
    )
    spec = importlib.util.spec_from_file_location("run_pfc_marked_family_rt", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_runner_round_trip_keeps_the_planted_peak_and_drops_silent_units(
    tmp_path, monkeypatch
):
    """``run_analysis`` writes a bundle that reads back with the planted kernel.

    Only the Spyglass-backed loaders are replaced (they hand over the simulated
    session); eligibility, the design, the fit and the bundle writing are the
    runner's own. Two units have no spike in the fitted bins: under the
    baseline prior they are left out, and the table's ``neuron_index`` and the
    sidecar's ``kept_unit_indices`` name the recorded units that were kept.
    """
    import json

    from continuum_swr_replay.analysis.pfc.marked_glm import MarkedSessionContext
    from continuum_swr_replay.analysis.pfc.marked_model_family import (
        DECODE_ENTROPY,
        LEAF_CONTENT_FRACTION,
    )

    runner = _load_runner()
    model = f"marginal_{CURRENT_ARM_FRACTION}"
    simulation = simulate(
        20261031,
        n_units=12,
        stop_bins=1500,
        gap_bins=300,
        event_spacing_bins=300,
        half_window_bins=125,
        mixed_rate_factor=1.5,
    )
    silent = (3, 9)
    n_recorded = simulation.counts.shape[1] + len(silent)
    kept = np.setdiff1d(np.arange(n_recorded), silent)
    counts = np.zeros((simulation.counts.shape[0], n_recorded))
    counts[:, kept] = simulation.counts
    rng = np.random.default_rng(1)
    events = {
        event_class: frame.assign(
            **{
                LEAF_CONTENT_FRACTION: rng.uniform(0.2, 1.0, len(frame)),
                DECODE_ENTROPY: rng.uniform(1.0, 3.0, len(frame)),
            }
        )
        for event_class, frame in simulation.events.items()
    }
    baseline, names, report = build_baseline_columns(
        simulation.fit_mask, simulation.stops
    )
    hashes = {
        "canonical_event_table_sha256": "event",
        "data_over_time_sha256": "dot",
        "swr_interval_source": "kay",
        "swr_events_sha256": "swr",
    }
    context = MarkedSessionContext(
        fit_mask=simulation.fit_mask,
        counts=counts,
        events=events,
        spline_basis=simulation.spline_basis,
        lag_time=simulation.lag_time,
        lags=simulation.lags,
        cohort_info={"n_bins_fit": int(simulation.fit_mask.sum())},
        input_provenance=dict(hashes),
        baseline=baseline,
        baseline_names=names,
        baseline_report=report,
    )
    monkeypatch.setattr(
        runner,
        "record_inputs",
        lambda inputs: {key: {"sha256": key.split("_")[0]} for key in inputs},
    )
    monkeypatch.setattr(
        runner, "canonical_event_input_hashes", lambda *a, **k: dict(hashes)
    )
    monkeypatch.setattr(
        runner,
        "load_session_stop_cohort",
        lambda *a, **k: (simulation.stops, {"stop_table_sha256": "stop"}),
    )
    monkeypatch.setattr(
        runner, "prepare_marked_session_context", lambda *a, **k: context
    )
    monkeypatch.setattr(
        runner,
        "load_session_recommended_event_marks",
        lambda *a, **k: (
            events,
            {"event_arm_table_sha256": "event", "event_table_sha256": "event"},
        ),
    )
    monkeypatch.setattr(
        runner,
        "load_session_stop_patch_switch_marks",
        lambda *a, **k: (
            pd.DataFrame(
                {"trips_to_patch_switch": np.nan, "next_trip_switches_patch": np.nan},
                index=pd.Index(simulation.stops["stop_idx"], name="stop_idx"),
            ),
            "stop",
        ),
    )

    assert runner.run_analysis(
        "rat20200101_.nwb",
        "01_r1",
        model_names=(model,),
        output_dir=tmp_path,
        log_directory=tmp_path / "logs",
        event_arm_table_path=tmp_path / "event_arm_table.parquet",
        event_table_path=tmp_path / "event_table.parquet",
        baseline_penalty_sigma=3.0,
        n_folds=5,
        design_batch_size=4,
    )

    artifact = runner.family_artifact_path(tmp_path, "rat20200101_01_r1_mPFC", model)
    record = json.loads(fit_provenance_path(artifact).read_text())
    extra = record.get("extra", record)
    assert extra["bundle_status"] == "complete"
    assert extra["bundle_contract_version"] == 3
    assert extra["n_units_recorded"] == n_recorded
    assert extra["n_units_without_spikes"] == len(silent)
    assert extra["kept_unit_indices"] == kept.tolist()
    assert "well_unknown" in extra["baseline_column_names"]
    table = pd.read_pickle(artifact)
    assert table["neuron_index"].tolist() == kept.tolist()
    assert (table["fit_spike_count"] > 0).all()

    lag_time = np.load(artifact.with_suffix(".lag_time.npy"))
    slope = np.load(
        artifact.with_suffix(f".{DRIVEN_CLASS}.{CURRENT_ARM_FRACTION}.slope.npy")
    )
    assert slope.shape == (len(lag_time), len(kept))
    curve = np.nanmean(slope, axis=1)
    true_peak = simulation.lag_time[simulation.peak_lag_index]
    peak = int(np.argmin(np.abs(lag_time - true_peak)))
    assert abs(int(np.argmax(curve)) - peak) <= 1
    assert curve[peak] == pytest.approx(PLANTED_PEAK, abs=0.35)
