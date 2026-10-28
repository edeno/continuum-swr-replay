"""Stop-level context of the mPFC replay response (``analysis/pfc/stop_context``).

Simulated data first: a planted reward gain on the post-onset response must be
recovered on top of a stationary per-stop rate difference that the stop's own
baseline absorbs; no gain must give about zero; a stationary rate difference
alone must not produce a gain; clustered events (overlap) must be handled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.stop_context import (
    CONTEXT_TERMS,
    STOP_CONTEXT_CLASSES,
    assemble_stop_context_table,
    count_other_onsets,
    fit_stop_context_glm,
    model_frame,
    near_onset_mask,
    onset_bin_index,
    stop_context_design,
    summarize_stop_context,
)

ANIMALS = ("a1", "a2", "a3", "a4", "a5")


def simulate_frame(
    seed: int,
    *,
    reward_gain: float = 1.0,
    switch_gain: float = 1.0,
    overlap_gain: float = 1.0,
    rate_difference: float = 1.0,
    event_free_scale: float = 1.0,
    n_sessions: int = 20,
    units_per_session: int = 30,
    stops_per_session: int = 40,
) -> pd.DataFrame:
    """Model rows drawn from the stop-context model, at the count level.

    Rewarded stops are longer (more event-free time), carry more events and
    more clustered ones, and their units fire ``rate_difference`` times faster
    throughout (a stationary rate difference). Units are a log-normal mix of
    low- and high-rate cells, so many baselines are a few spikes;
    ``event_free_scale`` shortens every stop's event-free time.

    Returns
    -------
    pd.DataFrame
        Columns of :func:`~continuum_swr_replay.analysis.pfc.stop_context.model_frame`.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for session in range(n_sessions):
        rewarded = (rng.random(stops_per_session) < 0.6).astype(float)
        switch = (rng.random(stops_per_session) < 0.3).astype(float)
        event_free_s = event_free_scale * np.where(
            rewarded == 1,
            rng.uniform(3.0, 12.0, stops_per_session),
            rng.uniform(0.5, 3.0, stops_per_session),
        )
        n_events = rng.poisson(np.where(rewarded == 1, 4.0, 1.5))
        stop_of_event = np.repeat(np.arange(stops_per_session), n_events)
        overlap = rng.poisson(np.where(rewarded[stop_of_event] == 1, 0.6, 0.1))
        stop_noise = rng.normal(0.0, 0.3, stops_per_session)
        base_rate = np.exp(rng.normal(np.log(2.0), 1.0, units_per_session))
        unit_gain = np.exp(rng.normal(0.3, 0.3, units_per_session))
        stop_rate = (
            base_rate[:, None]
            * rate_difference ** rewarded[None, :]
            * np.exp(stop_noise)[None, :]
        )
        baseline = rng.poisson(stop_rate * event_free_s[None, :])
        mean_y = (
            stop_rate[:, stop_of_event]
            * 0.2
            * unit_gain[:, None]
            * reward_gain ** rewarded[stop_of_event]
            * switch_gain ** switch[stop_of_event]
            * overlap_gain**overlap
        )
        y = rng.poisson(mean_y)
        n_units, n_ev = y.shape
        unit = np.repeat(np.arange(n_units), n_ev)
        event = np.tile(np.arange(n_ev), n_units)
        stop = stop_of_event[event]
        frames.append(
            pd.DataFrame(
                {
                    "session": f"s{session}",
                    "animal": ANIMALS[session % len(ANIMALS)],
                    "unit": unit,
                    "event_id": event,
                    "stop_idx": stop,
                    "y": y.ravel(),
                    "post_exposure_s": 0.2,
                    "baseline_count": baseline[unit, stop],
                    "baseline_time_s": event_free_s[stop],
                    "rewarded": rewarded[stop],
                    "next_trip_switches_patch": switch[stop],
                    "overlap": overlap[event],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _fit(frame, estimator="conditional", terms=(*CONTEXT_TERMS, "overlap")):
    fit = fit_stop_context_glm(stop_context_design(frame, terms), estimator=estimator)
    assert fit.converged
    return dict(zip(fit.names, fit.coef, strict=True)), fit


# ---------------------------------------------------------------------------
# Simulation: the estimator
# ---------------------------------------------------------------------------


def test_planted_reward_gain_is_recovered_over_a_rate_difference():
    frame = simulate_frame(0, reward_gain=1.5, rate_difference=2.0)
    coef, fit = _fit(frame)
    assert np.exp(coef["rewarded"]) == pytest.approx(1.5, abs=0.06)
    assert np.exp(coef["next_trip_switches_patch"]) == pytest.approx(1.0, abs=0.05)
    assert np.exp(coef["overlap"]) == pytest.approx(1.0, abs=0.05)
    # The interval is honest about its own precision.
    se = fit.se_cluster[fit.names.index("rewarded")]
    assert 0 < se < 0.05


def test_no_gain_gives_about_zero():
    coef, _ = _fit(simulate_frame(1))
    for term in (*CONTEXT_TERMS, "overlap"):
        assert abs(coef[term]) < 0.05


def test_rate_difference_alone_produces_no_gain():
    coef, _ = _fit(simulate_frame(2, rate_difference=3.0))
    assert abs(coef["rewarded"]) < 0.05


def test_planted_switch_gain_is_recovered():
    coef, _ = _fit(simulate_frame(3, switch_gain=0.7, rate_difference=1.5))
    assert np.exp(coef["next_trip_switches_patch"]) == pytest.approx(0.7, abs=0.04)
    assert abs(coef["rewarded"]) < 0.05


def test_overlap_term_absorbs_clustered_events():
    # Clustered events are more frequent at rewarded stops; without the overlap
    # term their extra spikes read as a reward gain.
    frame = simulate_frame(4, overlap_gain=1.4, rate_difference=2.0)
    coef, _ = _fit(frame)
    assert abs(coef["rewarded"]) < 0.05
    assert np.exp(coef["overlap"]) == pytest.approx(1.4, abs=0.05)
    without, _ = _fit(frame, terms=CONTEXT_TERMS)
    assert np.exp(without["rewarded"]) > 1.08


def test_plugin_baseline_is_biased_where_conditional_is_not():
    # Low-count baselines: E[log B] < log E[B] most in the short (unrewarded)
    # stops, so the plugged-in offset reads a reward "gain" that is not there.
    frame = simulate_frame(5, event_free_scale=0.3)
    conditional, _ = _fit(frame, "conditional")
    plugin, _ = _fit(frame, "plugin")
    assert abs(conditional["rewarded"]) < 0.05
    assert plugin["rewarded"] > 0.08


def test_unidentified_term_is_nan_and_uninformative_units_are_counted():
    frame = simulate_frame(6, n_sessions=4)
    frame["rewarded"] = 1.0
    silent = frame["unit"].eq(0)
    frame.loc[silent, "y"] = 0
    coef, fit = _fit(frame)
    assert np.isnan(coef["rewarded"])
    assert np.isfinite(coef["next_trip_switches_patch"])
    assert fit.counts["n_units_no_post_spikes"] == 4


def test_separated_term_is_nan_not_a_number():
    frame = simulate_frame(8, n_sessions=2)
    frame.loc[frame["overlap"] > 0, "y"] = 0
    fit = fit_stop_context_glm(stop_context_design(frame))
    assert not fit.converged
    assert np.isnan(fit.coef).all()


def test_one_cluster_has_no_clustered_se():
    _, fit = _fit(simulate_frame(9, n_sessions=1))
    assert np.isnan(fit.se_cluster).all()
    assert np.isfinite(fit.se_model).all()


def test_summary_reports_animals_and_leave_one_out():
    frame = simulate_frame(7, reward_gain=1.3, rate_difference=2.0)
    summary = summarize_stop_context(frame).set_index("term")
    row = summary.loc["rewarded"]
    assert row["exp_coef"] == pytest.approx(1.3, abs=0.07)
    assert row["ci_low"] < row["exp_coef"] < row["ci_high"]
    assert row["n_animals_estimated"] == 5
    assert row["n_animals_pooled_sign"] == 5
    assert row["loao_exp_coef_min"] <= row["exp_coef"] <= row["loao_exp_coef_max"]
    assert {f"exp_coef_{animal}" for animal in ANIMALS} <= set(summary.columns)
    assert row["n_sessions"] == 20


# ---------------------------------------------------------------------------
# Session assembly helpers
# ---------------------------------------------------------------------------


def test_onset_bin_index_matches_spike_binning():
    time = np.arange(10) / 500.0
    onsets = np.array([0.0, 0.0019, 0.002, 0.0179])
    np.testing.assert_array_equal(onset_bin_index(onsets, time), [0, 0, 1, 8])
    with pytest.raises(ValueError):
        onset_bin_index(np.array([1.0]), time)


def test_near_onset_mask_is_inclusive():
    mask = near_onset_mask(np.array([5, 20]), 30, 2)
    np.testing.assert_array_equal(
        np.flatnonzero(mask), [3, 4, 5, 6, 7, 18, 19, 20, 21, 22]
    )


def test_count_other_onsets_counts_each_event_once():
    # Non-local A (bin 10), B (bin 15), C (bin 30). SWR at 12 is A's own, SWR
    # at 16 is B's own, SWR at 18 overlaps no replay. Dropped events at 19, 25.
    count = count_other_onsets(
        np.array([10, 15]),
        non_local_bins=np.array([10, 15, 30]),
        non_local_start=np.array([0.020, 0.030, 0.060]),
        non_local_end=np.array([0.028, 0.034, 0.070]),
        swr_bins=np.array([12, 16, 18]),
        swr_start=np.array([0.024, 0.032, 0.036]),
        swr_end=np.array([0.026, 0.033, 0.040]),
        dropped_bins=np.array([19, 25]),
        n_post_bins=10,
    )
    # A [10, 20): B once (its SWR is part of it), SWR 18, dropped 19 -> 3.
    # B [15, 25): SWR 18, dropped 19 -> 2 (its own onset and SWR not counted).
    np.testing.assert_array_equal(count, [3, 2])


def _toy_session():
    """A 10 s session at 500 Hz with two stops and hand-placed spikes/events."""
    fs = 500.0
    time = np.arange(5000) / fs
    fit = np.zeros(5000, dtype=bool)
    fit[0:2000] = True  # stop 0: 0-4 s
    fit[2500:4500] = True  # stop 1: 5-9 s
    stops = pd.DataFrame(
        {
            "stop_idx": [7, 9],
            "stop_start": [0.0, 5.0],
            "stop_end": [3.998, 8.998],
            "is_rewarded_outcome": [1.0, 0.0],
            "next_trip_switches_patch": [0.0, np.nan],
        }
    )
    events = {
        "non_local_SWR": pd.DataFrame(
            {"event_id": ["e1"], "start_time": [1.0], "end_time": [1.1]}
        ),
        "non_local_no_SWR": pd.DataFrame(
            {
                "event_id": ["e2", "e3"],
                "start_time": [6.0, 8.9],
                "end_time": [6.05, 8.95],
            }
        ),
    }
    all_non_local = pd.concat(events.values(), ignore_index=True)
    swr = pd.DataFrame({"start_time": [1.02, 3.0], "end_time": [1.08, 3.05]})
    dropped = pd.DataFrame({"start_time": np.empty(0), "end_time": np.empty(0)})
    # Unit 0: spikes in e1's post window (3), around the SWR at 3 s (inside its
    # +/-250 ms window, not baseline), baseline spikes (2 in stop 0, 1 in stop 1),
    # one in e3's post window and one just past stop 1's fitted end.
    unit0 = np.array([1.0, 1.05, 1.199, 1.2, 3.1, 0.5, 2.0, 7.0, 8.95, 9.05])
    unit1 = np.array([])
    return time, fit, stops, events, all_non_local, swr, dropped, [unit0, unit1]


def test_assembled_toy_session_counts_are_exact():
    time, fit, stops, events, all_nl, swr, dropped, spikes = _toy_session()
    table, report = assemble_stop_context_table(
        time=time,
        fit_mask=fit,
        spike_times=spikes,
        events=events,
        all_non_local=all_nl,
        swr_intervals=swr,
        dropped_events=dropped,
        stops=stops,
        sampling_frequency=500.0,
    )
    assert len(table) == 2 * 3
    unit0 = table.loc[table["unit"].eq(0)].set_index("event_id")
    # e1 post window [1.0, 1.2): 1.0, 1.05, 1.199 (1.2 is the next bin).
    assert unit0.loc["e1", "y"] == 3
    assert unit0.loc["e2", "y"] == 0
    # e3 at 8.9: window 8.9-9.1, fitted only to 8.998 -> 50 bins.
    assert unit0.loc["e3", "y"] == 1
    assert unit0.loc["e3", "post_exposure_s"] == pytest.approx(0.1)
    assert unit0.loc["e1", "post_exposure_s"] == pytest.approx(0.2)
    # Stop 0 event-free: 4 s less the two 251-bin windows around 1.0 and 1.02
    # (merged: 0.75-1.27 s, 261 bins) and the SWR at 3.0 (251 bins).
    assert unit0.loc["e1", "baseline_time_s"] == pytest.approx((2000 - 261 - 251) / 500)
    assert unit0.loc["e1", "baseline_count"] == 2
    assert unit0.loc["e1", "stop_fit_count"] == 7
    # Stop 1: windows around 6.0 and 8.9 (the latter clipped at the fitted end).
    assert unit0.loc["e2", "baseline_count"] == 1
    assert unit0.loc["e2", "stop_idx"] == 9
    assert np.isnan(unit0.loc["e2", "next_trip_switches_patch"])
    # e1's own SWR (1.02) is part of it; the SWR at 3.0 is outside its window.
    assert unit0.loc["e1", "overlap"] == 0
    assert set(table["event_class"]) == set(STOP_CONTEXT_CLASSES)
    assert table.loc[table["unit"].eq(0), "unit_fit_rate_hz"].iloc[0] == pytest.approx(
        9 / 8.0
    )
    assert report["n_events_partial_post_fit_time"] == 1


def test_model_frame_counts_every_drop():
    time, fit, stops, events, all_nl, swr, dropped, spikes = _toy_session()
    table, _ = assemble_stop_context_table(
        time=time,
        fit_mask=fit,
        spike_times=spikes,
        events=events,
        all_non_local=all_nl,
        swr_intervals=swr,
        dropped_events=dropped,
        stops=stops,
        sampling_frequency=500.0,
    )
    table = table.assign(nwb_file_name="x_.nwb", epoch="02_r1", animal="x")
    units = pd.DataFrame({"nwb_file_name": "x_.nwb", "epoch": "02_r1", "unit": [0, 1]})
    frame, drops = model_frame(table, event_class="non_local_SWR", units=units)
    # Unit 1 never fires: its cell is silent.
    assert drops["n_unit_stop_cells_silent"] == 1
    assert len(frame) == 1
    frame, drops = model_frame(table, event_class="non_local_no_SWR", units=units)
    assert drops["n_stops_unknown_switch"] == 1
    assert len(frame) == 0


def test_end_to_end_planted_gain_from_spike_trains():
    """Spike trains with a x1.5 post-onset gain at rewarded stops, rate x2 there."""
    rng = np.random.default_rng(11)
    fs = 500.0
    n_stops, stop_s, gap_s = 60, 6.0, 1.0
    n_units = 25
    time = np.arange(int(n_stops * (stop_s + gap_s) * fs)) / fs
    starts = np.arange(n_stops) * (stop_s + gap_s)
    rewarded = (rng.random(n_stops) < 0.5).astype(float)
    stops = pd.DataFrame(
        {
            "stop_idx": np.arange(n_stops),
            "stop_start": starts,
            "stop_end": starts + stop_s - 1 / fs,
            "is_rewarded_outcome": rewarded,
            "next_trip_switches_patch": (rng.random(n_stops) < 0.4).astype(float),
        }
    )
    fit = np.zeros(len(time), dtype=bool)
    for start in starts:
        fit[int(start * fs) : int((start + stop_s) * fs)] = True
    # Events every ~1 s within a stop, never within 0.5 s of another.
    onsets = np.concatenate(
        [
            start + np.sort(rng.choice(np.arange(0.5, stop_s - 0.5, 0.6), 4, False))
            for start in starts
        ]
    )
    is_swr = rng.random(len(onsets)) < 0.5
    frame = pd.DataFrame(
        {
            "event_id": np.arange(len(onsets)).astype(str),
            "start_time": onsets,
            "end_time": onsets + 0.08,
        }
    )
    events = {
        "non_local_SWR": frame.loc[is_swr].reset_index(drop=True),
        "non_local_no_SWR": frame.loc[~is_swr].reset_index(drop=True),
    }
    stop_of_bin = np.clip(np.searchsorted(starts, time, side="right") - 1, 0, None)
    in_post = np.zeros(len(time), dtype=bool)
    for onset_bin in onset_bin_index(onsets, time):
        in_post[onset_bin : onset_bin + 100] = True
    spikes = []
    for _ in range(n_units):
        base = np.exp(rng.normal(np.log(3.0), 0.7))
        gain = np.exp(rng.normal(0.4, 0.2))
        rate = base * 2.0 ** rewarded[stop_of_bin]
        rate = np.where(in_post, rate * gain * 1.5 ** rewarded[stop_of_bin], rate)
        n = rng.poisson(rate / fs)
        spikes.append(np.repeat(time, n) + rng.uniform(0, 1 / fs, n.sum()))
    table, _ = assemble_stop_context_table(
        time=time,
        fit_mask=fit,
        spike_times=spikes,
        events=events,
        all_non_local=frame,
        swr_intervals=pd.DataFrame({"start_time": [], "end_time": []}),
        dropped_events=pd.DataFrame({"start_time": [], "end_time": []}),
        stops=stops,
        sampling_frequency=fs,
    )
    table = table.assign(nwb_file_name="sim_.nwb", epoch="02_r1", animal="sim")
    units = pd.DataFrame(
        {"nwb_file_name": "sim_.nwb", "epoch": "02_r1", "unit": np.arange(n_units)}
    )
    pooled = pd.concat(
        [
            model_frame(table, event_class=event_class, units=units)[0]
            for event_class in STOP_CONTEXT_CLASSES
        ]
    )
    coef, fit_result = _fit(pooled)
    se = fit_result.se_model[fit_result.names.index("rewarded")]
    assert np.exp(coef["rewarded"]) == pytest.approx(1.5, abs=3 * se * 1.5)
    assert abs(coef["next_trip_switches_patch"]) < 3 * se


# ---------------------------------------------------------------------------
# Real data: the table is built on the canonical Figure 4 assembly
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.real_data
def test_real_session_matches_canonical_assembly():
    """Events equal the marked context's, and the units are the bundle's.

    The bundle join (unit count and ``immobility_avg_firing_rate`` recomputed
    from the counted spikes) is checked inside the assembly. Opt-in
    (``-m real_data``): needs the session inputs and a reachable Spyglass.
    """
    from continuum_swr_replay.analysis.pfc.cohort import load_session_stop_cohort
    from continuum_swr_replay.analysis.pfc.marked_glm import (
        prepare_marked_session_context,
    )
    from continuum_swr_replay.analysis.pfc.stop_context import (
        prepare_session_stop_context,
    )
    from continuum_swr_replay.paths import processed_data_dir

    nwb_file_name, epoch_name = "j1620210710_.nwb", "08_r4"
    table, report = prepare_session_stop_context(
        nwb_file_name,
        epoch_name,
        bundle_path=processed_data_dir() / "j1620210710_08_r4_mPFC_modulation.pkl",
    )
    assert report["bundle_rate_max_abs_difference_hz"] < 1e-9
    stops, _ = load_session_stop_cohort(nwb_file_name, epoch_name, cohort="at_well")
    context = prepare_marked_session_context(
        nwb_file_name, epoch_name, cohort_stops=stops
    )
    for event_class in STOP_CONTEXT_CLASSES:
        ours = table.loc[table["event_class"].eq(event_class)].drop_duplicates(
            "event_id"
        )
        assert set(ours["event_id"]) == set(context.events[event_class]["event_id"])
    # Post-window counts from the marked context's fitted-bin counts.
    time = context.fit_mask.index.to_numpy(dtype=float)
    full = np.zeros((len(time), context.counts.shape[1]))
    full[np.flatnonzero(context.fit_mask.to_numpy(dtype=bool))] = context.counts
    first = onset_bin_index(table["onset_s"].to_numpy(), time)
    cumulative = np.vstack([np.zeros((1, full.shape[1])), np.cumsum(full, axis=0)])
    unit = table["unit"].to_numpy()
    expected = cumulative[first + 100, unit] - cumulative[first, unit]
    np.testing.assert_array_equal(table["y"].to_numpy(), expected)
