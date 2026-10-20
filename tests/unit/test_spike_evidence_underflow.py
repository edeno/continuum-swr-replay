"""The spike-evidence ("reality") engine leaves out events decoded where the CPU
rebuild is not the decode.

In chimi20200221_08_r4 (D1) the animal never entered patch 1, so the stored
occupancy at the far end of arm 1 is a float32 subnormal. The GPU decode kept
it; a CPU rebuild of the likelihood flushes it to zero, so an event whose MAP is
there is scored with numbers that differ from the decode's by up to 20 nats per
spike. Such events are left out and counted. Bins whose occupancy is exactly 0
are not: both devices read ``occupancy > 0`` as False there and agree. Every
input here is synthetic: a straight track of 2 cm bins and a stand-in encoding.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

import scripts.detection.analyze_event_spike_evidence as spike_evidence
from continuum_swr_replay.analysis.hippocampus.state_evidence import (
    UNDERFLOW_OCCUPANCY_FLOOR,
)

EPOCH_ID = "aa20200101_02_r1_HPC"
N_BINS = 40
CENTERS = 2.0 * np.arange(N_BINS)
ZERO_BIN = N_BINS - 4
"""A real bin with occupancy exactly 0, 72 cm: never reached, but not flagged."""
ZONE = np.arange(N_BINS - 3, N_BINS)
"""The last three real bins, 74-78 cm: subnormal and tiny occupancy, flagged."""
ANIMAL_CM = 8.0
N_TIME = 120


class StraightTrack:
    """Real bins every 2 cm on a straight track; all of them interior."""

    is_track_interior_ = np.ones(N_BINS, dtype=bool)

    def get_distances_to_interior_bins(self, positions):
        return np.abs(np.asarray(positions, dtype=float)[:, :1] - CENTERS[None, :])


def occupancy(underflowing: bool) -> np.ndarray:
    values = np.full(N_BINS, 3e-2, dtype=np.float32)
    if underflowing:
        values[ZERO_BIN] = 0.0
        values[ZONE] = [5.4e-43, 1.4e-24, 1.9e-22]
    return values


# four events of 5 bins; the MAP of the second is in the zone from its third bin
# on, and that of the fourth is in the bin whose occupancy is exactly 0
EVENTS = pd.DataFrame(
    {
        "s": [10, 40, 80, 100],
        "e": [14, 44, 84, 104],
        "n_bins": 5,
        "mean_abs_ab": 150.0,
        "is_swr": False,
        "group": "nonSWR-remote",
    }
)


def map_estimate() -> np.ndarray:
    mape = np.full(N_TIME, 30.0)
    mape[40:45] = [30.0, 30.0, 74.0, 76.0, 30.0]
    mape[100:105] = [30.0, 72.0, 72.0, 30.0, 30.0]
    return mape


def test_an_event_is_hit_when_a_scored_bin_s_map_is_in_a_left_out_bin():
    left_out = np.zeros(N_BINS, dtype=bool)
    left_out[ZONE] = True
    mape = map_estimate()
    lin = np.full(N_TIME, ANIMAL_CM)

    hit = spike_evidence.events_with_map_in_left_out_bins(
        EVENTS, mape, lin, StraightTrack(), left_out
    )

    # the fourth event's MAP is in the exact-zero bin, which is not left out
    np.testing.assert_array_equal(hit, [False, True, False, False])
    # a bin the metrics do not score (no finite animal position) does not hit
    lin_gap = lin.copy()
    lin_gap[42:44] = np.nan
    hit_gap = spike_evidence.events_with_map_in_left_out_bins(
        EVENTS, mape, lin_gap, StraightTrack(), left_out
    )
    np.testing.assert_array_equal(hit_gap, [False, False, False, False])


@pytest.fixture
def session(monkeypatch, tmp_path):
    """``analyze_session`` on stand-ins for everything it reads."""
    calls: list[tuple[int, int]] = []
    state = {"underflowing": True}
    time = 1000.0 + 0.002 * np.arange(N_TIME)
    dot = pd.DataFrame(
        {"map_estimate": map_estimate(), "linear_position": np.full(N_TIME, ANIMAL_CM)},
        index=pd.Index(time, name="time"),
    )

    def event_likelihood(s, e, t, lin, spike_times, waveform_features, encargs):
        calls.append((s, e))
        rng = np.random.default_rng(s)
        return rng.normal(-5.0, 2.0, (e - s + 1, N_BINS)), 6, 3

    def load_encoding(_epoch_id):
        return {
            "env": StraightTrack(),
            "encargs": {"occupancy": occupancy(state["underflowing"])},
            "pos": CENTERS,
            "gpi": np.ones(N_BINS),
        }

    patched = {
        "read_dot_with_swr_source": lambda _path, **_kwargs: dot,
        "validate_dot_event_contract": lambda _dot: None,
        "build_session_events": lambda _dot: EVENTS.copy(),
        "load_encoding": load_encoding,
        "load_marks": lambda *_a: ([], []),
        "event_likelihood": event_likelihood,
    }
    for name, value in patched.items():
        monkeypatch.setattr(spike_evidence, name, value)

    def analyze():
        return spike_evidence.analyze_session(EPOCH_ID, tmp_path, n_sample=10)

    return analyze, state, calls


def test_an_event_decoded_in_underflowed_bins_is_left_out_and_counted(session):
    analyze, _, calls = session

    metrics, context = analyze()

    # the event decoded in the exact-zero bin is scored: only the subnormal band
    # is where the CPU rebuild is not the decode
    assert sorted(metrics.start_time.round(3)) == sorted(
        (1000.0 + 0.002 * np.array([10, 80, 100])).round(3)
    )
    assert len(metrics) == 3
    # the left-out event's likelihood is not even built
    assert sorted(calls) == [(10, 14), (80, 84), (100, 104)]
    assert context["left_out"] == {
        "floor": UNDERFLOW_OCCUPANCY_FLOOR,
        "bins": ZONE.tolist(),
        "n_events": 4,
        "n_events_left_out": 1,
    }
    # the figures draw from the events that were scored
    assert context["events"].s.tolist() == [10, 80, 100]


def test_the_other_events_are_scored_as_when_nothing_underflows(session):
    analyze, state, calls = session
    state["underflowing"] = False
    all_events, context = analyze()

    assert len(all_events) == 4
    assert sorted(calls) == [(10, 14), (40, 44), (80, 84), (100, 104)]
    assert context["left_out"] == {
        "floor": UNDERFLOW_OCCUPANCY_FLOOR,
        "bins": [],
        "n_events": 4,
        "n_events_left_out": 0,
    }
    state["underflowing"] = True
    some_events, _ = analyze()

    # (the order of a group's sample depends on the group's size, so compare by event)
    def by_event(table: pd.DataFrame) -> pd.DataFrame:
        return table.sort_values("start_time").reset_index(drop=True)

    pd.testing.assert_frame_equal(
        by_event(some_events),
        by_event(all_events[all_events.start_time.isin(some_events.start_time)]),
    )


def test_the_run_records_how_many_events_each_epoch_left_out(
    session, monkeypatch, tmp_path
):
    analyze, state, _ = session
    metrics, context = analyze()
    monkeypatch.setattr(
        spike_evidence, "analyze_session", lambda *_a, **_k: (metrics, context)
    )
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_event_spike_evidence.py", "--epoch-identifier", EPOCH_ID,
         "--output-dir", str(out)],
    )  # fmt: skip

    spike_evidence.main()

    left_out = pd.read_csv(out / "event_spike_evidence_left_out.csv")
    assert left_out.to_dict("records") == [
        {
            "epoch_id": EPOCH_ID,
            "floor": UNDERFLOW_OCCUPANCY_FLOOR,
            "n_underflowed_bins": ZONE.size,
            "n_events": 4,
            "n_events_left_out": 1,
        }
    ]
