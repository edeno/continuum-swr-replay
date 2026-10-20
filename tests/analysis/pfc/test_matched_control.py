"""The matched non-event control the PFC engagement test is read against.

A non-SWR replay regressor on its own cannot separate "responds to replay"
from "responds to the stretch of immobility replay happens in". The control
answers that by sampling the same fitted bins with the same per-event
durations and no real event in the window, so the replay question becomes a
nested one. These tests pin the four properties that make the comparison mean
that, at the level of the returned windows rather than of the code path:

- every control window lies inside the bins the models are fit on;
- no control window overlaps a real event of any class -- ``SWR`` included,
  whose windows the non-local classes do not cover;
- controls are at most as many as the source events, and each preserves its
  source's duration;
- the draw is a function of the seed alone.

and the one property that makes it usable: an event whose bout has no free
window of its duration is dropped and counted under the right one of the two
shortfall reasons, never placed somewhere else. Control *onsets* are distinct,
but their duration-matching windows may overlap one another. Duration only
constrains where an onset may be drawn; the GLM regressor contains the onsets,
not an event-in-progress box. Requiring the auxiliary windows to be disjoint
can make an exact count-matched null mathematically impossible in dense bouts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.glm import (
    _make_design_matrices,
    make_matched_control_times,
    matched_control_seed,
    process_neuron,
)

SAMPLING_FREQUENCY = 500.0


def _times(duration_s: float = 10.0) -> np.ndarray:
    """Sample times of a ``duration_s`` epoch at 500 Hz."""
    return np.arange(0.0, duration_s, 1.0 / SAMPLING_FREQUENCY)


def _windows(bounds: list[tuple[float, float]]) -> pd.DataFrame:
    """An event table from ``(start, end)`` pairs."""
    return pd.DataFrame(
        {
            "start_time": [float(start) for start, _ in bounds],
            "end_time": [float(end) for _, end in bounds],
        },
        dtype=float,
    )


def _event_sets(
    *,
    non_local_no_swr: list[tuple[float, float]],
    non_local_swr: list[tuple[float, float]],
    swr: list[tuple[float, float]],
) -> dict[str, pd.DataFrame]:
    """The four event sets keyed as ``build_session_event_sets`` returns them."""
    non_local = sorted(non_local_no_swr + non_local_swr)
    return {
        "non_local_no_SWR": _windows(non_local_no_swr),
        "non_local_SWR": _windows(non_local_swr),
        "non_local": _windows(non_local),
        "SWR": _windows(swr),
    }


@pytest.fixture
def fit_mask():
    """Two fitted bouts, 0.5-4 s and 6-9.5 s, with a gap the animal moves in."""
    times = _times()
    inside = ((times >= 0.5) & (times < 4.0)) | ((times >= 6.0) & (times < 9.5))
    return pd.Series(inside, index=times)


@pytest.fixture
def event_times():
    """Four events, two per bout, with one SWR that no non-local class holds."""
    return _event_sets(
        non_local_no_swr=[(1.0, 1.2), (6.5, 6.6)],
        non_local_swr=[(2.0, 2.3), (8.0, 8.1)],
        swr=[(2.0, 2.25), (8.0, 8.08), (3.0, 3.5)],
    )


def _overlaps(start, end, frame):
    """Whether ``[start, end)`` overlaps any half-open window of ``frame``."""
    return bool(
        np.any(
            (start < frame.end_time.to_numpy(dtype=float))
            & (frame.start_time.to_numpy(dtype=float) < end)
        )
    )


class TestMatchedControlInvariants:
    def test_every_control_window_lies_inside_the_fitted_bins(
        self, event_times, fit_mask
    ):
        # A control drawn outside the fit mask would be a regressor for bins no
        # model sees, so the engagement comparison would be against a baseline
        # measured somewhere else.
        controls = make_matched_control_times(
            event_times, fit_mask, np.random.default_rng(1)
        ).times

        assert len(controls) > 0, "nothing was placed; the invariant is vacuous"
        times = fit_mask.index.to_numpy(dtype=float)
        inside = fit_mask.to_numpy(dtype=bool)
        for start, end in zip(controls.start_time, controls.end_time, strict=True):
            covered = (times >= start) & (times <= end)
            assert covered.any(), f"control [{start}, {end}] covers no sample"
            assert inside[covered].all(), (
                f"control [{start}, {end}] covers a bin outside the fit mask"
            )

    def test_no_control_overlaps_a_real_event_of_any_class(self, event_times, fit_mask):
        # The SWR at 3.0-3.5 s belongs to no non-local class, so a control that
        # honoured only the non-local windows could land on it.
        controls = make_matched_control_times(
            event_times, fit_mask, np.random.default_rng(2)
        ).times

        assert len(controls) > 0
        for start, end in zip(controls.start_time, controls.end_time, strict=True):
            for name, frame in event_times.items():
                assert not _overlaps(start, end, frame), (
                    f"control [{start}, {end}] overlaps a {name} event"
                )

    def test_controls_are_at_most_as_many_as_events_and_keep_each_duration(
        self, event_times, fit_mask
    ):
        draw = make_matched_control_times(
            event_times, fit_mask, np.random.default_rng(3)
        )
        controls = draw.times

        source = event_times["non_local"]
        assert len(controls) <= len(source)
        # This configuration places all four, so row k pairs with source row k
        # and the durations must match one for one rather than in
        # distribution. The tolerance is the float64 rounding of a start plus a
        # duration, not room for a different duration.
        assert len(controls) == len(source)
        np.testing.assert_array_equal(draw.source_indices, np.arange(len(source)))
        np.testing.assert_allclose(
            (controls.end_time - controls.start_time).to_numpy(),
            (source.end_time - source.start_time).to_numpy(),
            rtol=0,
            atol=1e-9,
        )

    def test_the_draw_is_a_function_of_the_seed(self, event_times, fit_mask):
        first = make_matched_control_times(
            event_times, fit_mask, np.random.default_rng(4)
        ).times
        again = make_matched_control_times(
            event_times, fit_mask, np.random.default_rng(4)
        ).times
        other = make_matched_control_times(
            event_times, fit_mask, np.random.default_rng(5)
        ).times

        pd.testing.assert_frame_equal(first, again)
        assert not first.equals(other), (
            "two seeds produced the same windows; the draw is not random over "
            "the bout, so the control is one arbitrary placement rather than a "
            "sample of the non-event time"
        )

    def test_an_event_whose_bout_has_no_free_window_is_dropped(self):
        # One bout, 1-2 s. The first event is 0.4 s long and the rest of the
        # bout is filled by a SWR, so no 0.4 s window is free: the event must
        # be dropped, not placed on top of the SWR or outside the bout.
        times = _times(duration_s=4.0)
        mask = pd.Series((times >= 1.0) & (times < 2.0), index=times)
        crowded = _event_sets(
            non_local_no_swr=[(1.0, 1.4)],
            non_local_swr=[],
            swr=[(1.4, 2.0)],
        )

        draw = make_matched_control_times(crowded, mask, np.random.default_rng(6))

        assert len(draw.times) == 0
        assert draw.source_indices.tolist() == []
        assert list(draw.times.columns) == ["start_time", "end_time"]
        # A full bout, not an onset the fit mask never covered.
        assert (draw.n_onset_outside_fitted_bout, draw.n_bout_saturated) == (0, 1)
        assert draw.n_unplaced == 1

    def test_an_event_outside_every_fitted_bout_is_dropped(self, fit_mask):
        # An event whose onset is in the moving gap has no bout to be matched
        # in; it must be counted as unplaced rather than matched in a bout it
        # does not belong to.
        outside = _event_sets(
            non_local_no_swr=[(4.5, 4.6)],
            non_local_swr=[(1.0, 1.1)],
            swr=[],
        )

        draw = make_matched_control_times(outside, fit_mask, np.random.default_rng(7))

        assert len(draw.times) == 1
        assert draw.source_indices.tolist() == [0]
        assert 0.5 <= float(draw.times.start_time.iloc[0]) < 4.0
        # An onset the fit mask never covered, not a bout with no room.
        assert (draw.n_onset_outside_fitted_bout, draw.n_bout_saturated) == (1, 0)
        assert draw.n_unplaced == 1

    def test_dense_bout_keeps_exact_count_with_distinct_control_onsets(self):
        # Seven 0.1 s events leave only 0.3 s of non-event time. Seven
        # duration windows cannot be packed there without overlap, but the
        # onset-only GLM has hundreds of distinct legal onset bins. The null
        # must retain the event count instead of becoming a thinner regressor.
        times = _times(duration_s=3.0)
        mask = pd.Series((times >= 1.0) & (times < 2.0), index=times)
        packed = _event_sets(
            non_local_no_swr=[
                (1.0, 1.1),
                (1.1, 1.2),
                (1.2, 1.3),
                (1.3, 1.4),
                (1.4, 1.5),
                (1.5, 1.6),
                (1.6, 1.7),
            ],
            non_local_swr=[],
            swr=[],
        )

        draw = make_matched_control_times(packed, mask, np.random.default_rng(8))

        assert len(draw.times) == len(packed["non_local"])
        starts = draw.times.start_time.to_numpy(dtype=float)
        ends = draw.times.end_time.to_numpy(dtype=float)
        assert len(np.unique(starts)) == len(starts)
        order = np.argsort(starts)
        starts, ends = starts[order], ends[order]
        assert np.any(starts[1:] < ends[:-1]), (
            "the fixture requires overlapping duration windows; without one "
            "it does not distinguish exact onset matching from interval packing"
        )


class TestMatchedControlSeed:
    def test_the_seed_is_the_session_key(self):
        assert matched_control_seed(
            "wilbur20210329_.nwb", "02_r1"
        ) == matched_control_seed("wilbur20210329_.nwb", "02_r1")
        assert matched_control_seed(
            "wilbur20210329_.nwb", "02_r1"
        ) != matched_control_seed("wilbur20210329_.nwb", "04_r2")
        assert matched_control_seed(
            "wilbur20210329_.nwb", "02_r1"
        ) != matched_control_seed("j1620210710_.nwb", "02_r1")
        assert 0 <= matched_control_seed("wilbur20210329_.nwb", "02_r1") < 2**64


# ---------------------------------------------------------------------------
# The engagement test, end to end on a neuron that tracks non-SWR replay
# ---------------------------------------------------------------------------


@pytest.fixture
def engagement_session():
    """A 120 s session whose neuron fires in bursts at non-SWR replay onsets.

    Immobile throughout except for a moving window, so the control has room to
    be drawn. The neuron's rate is raised only in the 100 ms after each
    ``non_local_no_SWR`` onset -- nowhere else in the bout -- which is exactly
    the difference the engagement test must see and the matched control must
    not.
    """
    times = _times(duration_s=120.0)
    speed = np.full(times.shape, 2.0)
    speed[(times >= 60.0) & (times < 62.0)] = 10.0
    position_info = pd.DataFrame({"head_speed": speed}, index=times)
    is_immobile = position_info.head_speed < 4.0

    no_swr_onsets = np.arange(3.0, 58.0, 2.0)
    swr_onsets = np.arange(64.0, 118.0, 2.0)
    event_times = _event_sets(
        non_local_no_swr=[(t, t + 0.1) for t in no_swr_onsets],
        non_local_swr=[(t, t + 0.1) for t in swr_onsets],
        swr=[(t, t + 0.1) for t in swr_onsets],
    )

    rng = np.random.default_rng(11)
    dt = 1.0 / SAMPLING_FREQUENCY
    rate = np.full(times.shape, 4.0)
    for onset in no_swr_onsets:
        rate[(times >= onset) & (times < onset + 0.1)] = 90.0
    spike_times = times[rng.random(times.size) < rate * dt]

    theta_phase = pd.Series(
        np.mod(2 * np.pi * 8 * times, 2 * np.pi) - np.pi, index=times
    )
    return {
        "times": times,
        "position_info": position_info,
        "is_immobile": is_immobile,
        "event_times": event_times,
        "spike_times": spike_times,
        "theta_phase": theta_phase,
    }


def test_replay_beats_the_matched_control_for_a_replay_tracking_neuron(
    engagement_session,
):
    session = engagement_session
    event_times = dict(session["event_times"])
    event_times["matched_control"] = make_matched_control_times(
        event_times, session["is_immobile"], np.random.default_rng(12)
    ).times
    assert len(event_times["matched_control"]) == len(event_times["non_local"]), (
        "not every event got a control; the engagement comparison would be "
        "against a smaller baseline than the events it is read against"
    )

    design_matrices, spline_basis, lag_time = _make_design_matrices(
        event_times,
        session["position_info"],
        sampling_frequency=SAMPLING_FREQUENCY,
        lags=np.arange(-25, 26, 1),
    )
    stats, _, no_swr_mod, _, _, matched_control_mod = process_neuron(
        session["spike_times"],
        session["is_immobile"],
        session["theta_phase"],
        session["times"],
        design_matrices,
        spline_basis,
        lag_time,
    )

    # The engagement comparison, as a deviance fraction: the in-sample test
    # statistic is gone with the penalised estimator, and the held-out
    # difference is the runner's. What has to be true here is that non-SWR
    # replay explains deviance the matched baseline does not.
    engagement = stats[
        "matched_control + non_local_no_SWR vs. matched_control: frac_null_dev"
    ]
    assert engagement > 0.01, (
        "a neuron that fires only at non-SWR replay onsets explained almost no "
        "deviance beyond its duration-matched non-event baseline "
        f"(frac_null_dev = {engagement})"
    )
    # The baseline's own curve must be estimable and flat next to the replay
    # response; an all-NaN or replay-sized control curve would mean the
    # comparison is not measuring what it claims to.
    assert np.isfinite(matched_control_mod).all()
    assert np.abs(matched_control_mod).max() < np.abs(no_swr_mod).max()
    assert np.isfinite(stats["matched_control_max_mod"])
    assert np.isfinite(stats["matched_control vs. constant: frac_null_dev"])


# ---------------------------------------------------------------------------
# Real data: the runner writes the control's array and states its draw
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.real_data
def test_a_real_session_writes_the_control_array_and_records_its_draw(tmp_path):
    """The bundle a real session writes carries the control and its provenance.

    Needs session inputs on disk and a reachable Spyglass, so it is opt-in
    (``-m real_data``) and fails when either is missing.
    """
    import importlib.util
    import json
    from pathlib import Path

    from continuum_swr_replay.analysis.replay import prepare_data as pdm

    spec = importlib.util.spec_from_file_location(
        "run_pfc_analysis_bandit",
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "pfc"
        / "run_pfc_analysis_bandit.py",
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    sessions = [s for s in pdm.find_sessions(max_per_animal=1) if s["results_file"]]
    assert sessions, "no session with decode results on disk"

    # A session with no mPFC units writes arrays and a sidecar but never scores
    # anything, so it carries none of the inference the skip check asks about.
    # Take the first session that actually fits units.
    # The null is drawn at the shortened reporting floor rather than the
    # production count: the number of draws is not what the skip check reads,
    # and a production-sized null would make one test an hour long.
    n_control_draws = runner.MIN_USABLE_NULL_DRAWS
    for session in sessions:
        runner.run_analysis(
            session["nwb_file_name"],
            session["epoch_name"],
            log_directory=tmp_path / "logs",
            output_dir=tmp_path / "out",
            n_control_draws=n_control_draws,
            # The suite pins JAX to CPU so no test takes a card on a shared host.
            allow_cpu=True,
        )
        epoch_identifier = (
            f"{session['nwb_file_name'].replace('_.nwb', '')}_"
            f"{session['epoch_name']}_mPFC"
        )
        artifact = tmp_path / "out" / f"{epoch_identifier}_modulation.pkl"
        if artifact.exists() and len(pd.read_pickle(artifact)):
            break
    else:
        pytest.fail(
            "no listed session fit any mPFC unit (Spyglass, canonical inputs "
            f"or mPFC units unavailable); see {tmp_path / 'logs'}"
        )

    control_array = artifact.with_suffix(".matched_control_modulation.npy")
    assert control_array.exists(), "the bundle has no matched-control array"
    control_modulation = np.load(control_array)
    no_swr_modulation = np.load(
        artifact.with_suffix(".non_local_no_SWR_modulation.npy")
    )
    assert control_modulation.shape == no_swr_modulation.shape

    record = json.loads(artifact.with_suffix(".provenance.json").read_text())
    assert record["matched_control_seed"] == matched_control_seed(
        session["nwb_file_name"], session["epoch_name"]
    )
    assert record["n_matched_control"] >= 0
    assert (
        record["n_matched_control"] + record["n_matched_control_unplaced"]
        == record["cohort_counts"]["n_nonlocal_after"]
    )
    # A run that writes the array must also be skippable as complete -- by a
    # run asking for the same estimator, which is what the skip check compares:
    # the prior, the folds and the null's size, seed, p-value convention,
    # placement rule and per-unit usable-draw counts as well as the inputs.
    assert record["n_control_draws"] == n_control_draws
    assert record["min_usable_null_draws"] == runner.required_usable_null_draws(
        n_control_draws
    )
    assert set(record["null_usable_draws"]) >= set(runner.EVENT_CLASSES)
    assert runner.results_exist(
        tmp_path / "out",
        epoch_identifier,
        False,
        "at_well",
        record["stop_table_sha256"],
        record["canonical_event_table_sha256"],
        record["data_over_time_sha256"],
        record["fitter"],
        penalty_sigma=record["penalty_sigma_log_rate"],
        n_folds=record["n_folds"],
        fold_seed=record["fold_seed"],
        n_control_draws=record["n_control_draws"],
        control_draw_seed=record["control_draw_seed"],
        swr_interval_source=record["swr_interval_source"],
        swr_events_sha256=record["swr_events_sha256"],
    )


class TestEventsTheAnalysedTimeRuleDropped:
    """A control never lands on an event the analysed-time rule dropped (a
    non-local event or SWR in excluded time, or overlapping one dropped there),
    and the dropped events near fitted bins are counted, not hidden."""

    def test_a_control_avoids_a_dropped_event(self, event_times, fit_mask):
        # a dropped non-local event filling most of the second bout's free time
        dropped = _windows([(6.7, 7.9)])

        for seed in range(20):
            draw = make_matched_control_times(
                event_times, fit_mask, np.random.default_rng(seed), avoid=dropped
            )
            for start, end in draw.times.to_numpy():
                assert not _overlaps(start, end, dropped)

    def test_no_dropped_events_change_nothing(self, event_times, fit_mask):
        plain = make_matched_control_times(
            event_times, fit_mask, np.random.default_rng(3)
        )
        empty = make_matched_control_times(
            event_times, fit_mask, np.random.default_rng(3), avoid=_windows([])
        )

        pd.testing.assert_frame_equal(plain.times, empty.times)

    def test_the_dropped_events_are_the_non_local_events_and_swrs(self):
        from continuum_swr_replay.analysis.pfc.glm import artifact_dropped_events

        drops = pd.DataFrame(
            {
                "event_col": [
                    "SWR_event",
                    "SWR_event",
                    "MUA_event",
                    "non_local_event",
                    "non_local_event",
                    "non_local_event",
                ],
                "label": [1, 2, 1, 1, 2, 3],
                "start_time": [5.0, 1.0, 2.0, 3.0, 4.0, 0.5],
                "end_time": [5.1, 1.1, 2.1, 3.1, 4.1, 0.6],
                "speed_eligible": True,
                "drop_reason": [
                    "near_artifact",
                    "moving",
                    "artifact",
                    None,
                    "overlaps_dropped_event",
                    "artifact",
                ],
            }
        )

        dropped = artifact_dropped_events(drops)

        np.testing.assert_allclose(
            dropped.to_numpy(), [[0.5, 0.6], [4.0, 4.1], [5.0, 5.1]]
        )
