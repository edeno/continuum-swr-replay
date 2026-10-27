"""Simulated interval streams, and what each per-bin history design reads from them.

The simulator (``analysis/replay/duration_history_simulation.py``) draws two
streams of non-overlapping events with durations under known history
dependence. The recovery tests fit the shipped cross-history design and its
duration-aware twin with the real fitter and pin, per scenario, what each
reads. They use the scenarios at a raised base rate (so a few hundred stops
carry enough event pairs within the 150 ms history window); every seed is
fixed, so each assertion is deterministic, and the tolerances are stated
where they are asserted.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from continuum_swr_replay.analysis.replay.design_constants import N_HISTORY_LAGS
from continuum_swr_replay.analysis.replay.duration_history_simulation import (
    MIN_DURATION_S,
    SCENARIOS,
    STREAMS,
    HistoryKernel,
    fit_history_direction,
    simulate_interval_streams,
    summarize_direction,
    true_lag_kernel,
)


def _gaps(events):
    """Seconds from each event's end to the next onset in its stop."""
    ev = events.sort_values(["stop_id", "t_since_stop_onset"])
    same = ev["stop_id"].to_numpy()[1:] == ev["stop_id"].to_numpy()[:-1]
    gaps = (
        ev["t_since_stop_onset"].to_numpy()[1:]
        - ev["t_end_since_stop_onset"].to_numpy()[:-1]
    )
    return gaps[same]


# ── The simulator ──


def test_events_never_overlap_and_respect_the_minimum_duration():
    stops, events = simulate_interval_streams(
        SCENARIOS["b_fragmentation"], n_stops=300, seed=0
    )
    assert (_gaps(events) > 0).all()
    assert (events["duration_s"] >= MIN_DURATION_S).all()
    length = stops.set_index("stop_id")["immobility_duration_s"]
    assert (events["t_since_stop_onset"] < events["stop_id"].map(length)).all()
    assert set(events["stream"]) == set(STREAMS)


def test_simulation_is_a_function_of_the_seed():
    a = simulate_interval_streams(SCENARIOS["a_independent"], n_stops=50, seed=7)
    b = simulate_interval_streams(SCENARIOS["a_independent"], n_stops=50, seed=7)
    for x, y in zip(a, b, strict=True):
        assert x.equals(y)


def test_independent_onsets_have_the_base_rate_in_idle_time():
    """Scenario (a): onsets / idle seconds is the summed base rate (+-5 %),
    the stream split follows the base rates (+-3 points) and each stream's
    median duration is its configured median (+-10 %)."""
    sc = SCENARIOS["a_independent"]
    stops, events = simulate_interval_streams(sc, n_stops=2000, seed=1)
    length = events["stop_id"].map(stops.set_index("stop_id")["immobility_duration_s"])
    busy = (
        np.minimum(events["t_end_since_stop_onset"], length)
        - events["t_since_stop_onset"]
    ).sum()
    idle = stops["immobility_duration_s"].sum() - busy
    total_rate = sum(sc.base_rate_hz.values())
    assert len(events) / idle == pytest.approx(total_rate, rel=0.05)
    share = (events["stream"] == "swr").mean()
    assert share == pytest.approx(sc.base_rate_hz["swr"] / total_rate, abs=0.03)
    medians = events.groupby("stream")["duration_s"].median()
    for s in STREAMS:
        assert medians[s] == pytest.approx(sc.duration_median_s[s], rel=0.1)


def test_fragmentation_crowds_onsets_just_after_an_end():
    """Scenario (b) raises the hazard 20-60 ms after an END: far more gaps
    land there than without it (ratio > 3)."""
    _, frag = simulate_interval_streams(
        SCENARIOS["b_fragmentation"], n_stops=1000, seed=2
    )
    _, base = simulate_interval_streams(
        SCENARIOS["a_independent"], n_stops=1000, seed=2
    )

    def frac(ev):
        g = _gaps(ev)
        return np.mean((g > 0.02) & (g < 0.06))

    assert frac(frag) > 3 * frac(base)


def test_true_lag_kernel_is_the_kernel_averaged_over_the_bin_jitter():
    sc = SCENARIOS["d_cross_swr_to_nonswr"]
    truth = true_lag_kernel(sc, target="no_swr", source="swr", anchor="offset")
    assert truth.shape == (N_HISTORY_LAGS,)
    # peak at lag 3 (45 ms; the bump peaks at 50 ms), nothing past lag 6
    assert int(np.argmax(truth)) + 1 == 3
    assert np.all(truth[6:] < 1e-3)
    # the other direction and the onset anchor carry nothing
    assert not true_lag_kernel(sc, target="swr", source="no_swr", anchor="offset").any()
    assert not true_lag_kernel(sc, target="no_swr", source="swr", anchor="onset").any()


def test_kernel_rejects_support_past_the_window():
    with pytest.raises(ValueError, match="KERNEL_SUPPORT_S"):
        HistoryKernel("swr", "swr", "onset", 1.0, center_s=0.2, width_s=0.02)


# ── Recovery: what each design reads from each scenario ──
#
# Raised base rate (4x: 1.6 / 2.4 Hz) and 300 stops: about 1,100 SWR-coupled and
# 1,700 non-SWR events per scenario. Thresholds were set from seeds 0-4 with
# margin. Duration-aware design, where the truth has no such effect: self-block
# AIC gain <= 1.7 in (a), cross-block gain <= 9.0 in (a), (c) and the reverse
# direction of (d). Where it has one: every AIC gain or anchor contrast >= 41
# (self offset-vs-onset in (c): <= -41). Kernel peaks: (b) offset lag 2 in all
# ten fits, log RR 1.11-1.48; (c) SWR-coupled onset lag 6-7, 0.99-1.17;
# (d) cross offset lag 3, 0.98-1.31.

RATE_SCALE = 4.0
N_STOPS = 300
SEED = 0
#: A lag's fitted kernel is read only with this many expected onsets.
MIN_SUPPORT = 10.0


@pytest.fixture(scope="module")
def fits():
    """Lazily fit (scenario, response, design) once for the module."""
    cache: dict = {}
    sims: dict = {}

    def get(scenario: str, response: str, duration_aware: bool) -> dict:
        key = (scenario, response, duration_aware)
        if key not in cache:
            if scenario not in sims:
                sc = SCENARIOS[scenario]
                sc = dataclasses.replace(
                    sc,
                    base_rate_hz={
                        k: v * RATE_SCALE for k, v in sc.base_rate_hz.items()
                    },
                )
                sims[scenario] = simulate_interval_streams(
                    sc, n_stops=N_STOPS, seed=SEED
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = fit_history_direction(
                    *sims[scenario], response=response, duration_aware=duration_aware
                )
            cache[key] = (result, summarize_direction(result))
        return cache[key]

    return get


def _peak(result: dict, rung: str, block: str) -> tuple[int, float]:
    """(lag, log RR) of a block's largest supported kernel value."""
    kernel = np.where(
        result["support"][block] >= MIN_SUPPORT, result["kernels"][rung][block], np.nan
    )
    lag = int(np.nanargmax(kernel))
    return lag + 1, float(kernel[lag])


@pytest.mark.parametrize("response", STREAMS)
def test_a_current_design_reads_history_into_independent_events(fits, response):
    """(a) No dependence: the shipped design still finds self- and cross-history
    (AIC gains > 50) and a deep short-lag dip (log RR < -2): non-overlap read
    as refractoriness and as cross-stream inhibition."""
    result, summary = fits("a_independent", response, False)
    assert summary["delta_aic_self"] > 50
    assert summary["delta_aic_cross"] > 50
    for block in ("self_onset", "cross_onset"):
        assert result["kernels"]["self+cross"][block][0] < -2


@pytest.mark.parametrize("response", STREAMS)
def test_a_duration_aware_design_finds_no_history(fits, response):
    """(a) The at-risk design with onset+offset history finds nothing: the self
    block (both anchors) gains < 4 AIC and the cross block < 10, and every
    onset lies in an at-risk bin."""
    result, summary = fits("a_independent", response, True)
    assert summary["delta_aic_self"] < 4
    assert summary["delta_aic_cross"] < 10
    assert result["design_attrs"]["n_onsets_not_at_risk"] == 0


def test_b_current_design_reads_fragmentation_as_an_onset_locked_burst(fits):
    """(b) A hazard locked 20-60 ms after the previous END appears, in the
    shipped design, as a self-history burst peaking 90-150 ms after the
    previous ONSET (lag >= 6, rate ratio > 1.35), after a deep dip."""
    result, _ = fits("b_fragmentation", "no_swr", False)
    lag, log_rr = _peak(result, "self+cross", "self_onset")
    assert lag >= 6 and log_rr > np.log(1.35)
    assert result["kernels"]["self+cross"]["self_onset"][0] < -2


@pytest.mark.parametrize("response", STREAMS)
def test_b_duration_aware_design_finds_offset_locked_history(fits, response):
    """(b) Offset-only history beats onset-only by > 50 AIC for the self and
    the cross block, and the offset kernel peaks at lag 2-3 (30-45 ms; truth
    peaks at lag 3, log RR 1.39) with log RR within 0.4 of the truth's peak."""
    result, summary = fits("b_fragmentation", response, True)
    assert summary["self_offset_vs_onset"] > 50
    assert summary["cross_offset_vs_onset"] > 50
    assert summary["best_self"] in ("self_offset", "self_both")
    truth = true_lag_kernel(
        SCENARIOS["b_fragmentation"], target=response, source=response, anchor="offset"
    )
    lag, log_rr = _peak(result, result["best"], "self_offset")
    assert lag in (2, 3)
    assert log_rr == pytest.approx(truth.max(), abs=0.4)


@pytest.mark.parametrize("response", STREAMS)
def test_c_duration_aware_design_finds_onset_locked_rhythm(fits, response):
    """(c) Onset-only history beats offset-only by > 30 AIC, and adding the
    cross block gains < 10 (there is no coupling)."""
    _, summary = fits("c_onset_rhythm", response, True)
    assert summary["self_offset_vs_onset"] < -30
    assert summary["best_self"] in ("self_onset", "self_both")
    assert summary["delta_aic_cross"] < 10


def test_c_onset_rhythm_kernel_is_recovered(fits):
    """(c) The onset kernel peaks at lag 7 +-1 (truth: lag 7, 105 ms, log RR
    1.22), within 0.4 of the truth's peak."""
    result, _ = fits("c_onset_rhythm", "swr", True)
    truth = true_lag_kernel(
        SCENARIOS["c_onset_rhythm"], target="swr", source="swr", anchor="onset"
    )
    lag, log_rr = _peak(result, result["best"], "self_onset")
    assert abs(lag - 7) <= 1
    assert log_rr == pytest.approx(truth.max(), abs=0.4)


def test_d_current_design_reads_coupling_both_ways(fits):
    """(d) SWR-coupled -> non-SWR coupling only, yet the shipped design also
    gains > 50 AIC from cross-history in the reverse direction."""
    _, summary = fits("d_cross_swr_to_nonswr", "swr", False)
    assert summary["delta_aic_cross"] > 50


def test_d_duration_aware_design_finds_the_direction_and_lag(fits):
    """(d) Non-SWR given SWR-coupled: cross history gains > 50 AIC, offset
    beats onset by > 50, and the cross offset kernel peaks at lag 3 +-1
    (truth: lag 3, log RR 1.33) within 0.4; SWR-coupled given non-SWR: the
    cross block gains < 10."""
    result, summary = fits("d_cross_swr_to_nonswr", "no_swr", True)
    assert summary["delta_aic_cross"] > 50
    assert summary["cross_offset_vs_onset"] > 50
    truth = true_lag_kernel(
        SCENARIOS["d_cross_swr_to_nonswr"],
        target="no_swr",
        source="swr",
        anchor="offset",
    )
    lag, log_rr = _peak(result, result["best"], "cross_offset")
    assert abs(lag - 3) <= 1
    assert log_rr == pytest.approx(truth.max(), abs=0.4)
    _, reverse = fits("d_cross_swr_to_nonswr", "swr", True)
    assert reverse["delta_aic_cross"] < 10
