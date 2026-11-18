"""Clusterless recordings with known content, for the held-out tetrode score.

An animal runs laps of a linear track and rests at each end. Every tetrode but
one has cells with place fields spread along the whole track (as real tetrodes
do); the last tetrode's cells all sit in the middle of the track
(:data:`REGION_CM`), so it has no field at either end. Each cell fires at the
place it represents: the animal's own position, except in inserted events
during rest (:data:`EVENT_KINDS`), where it is

- ``"sweep"``: a trajectory far from the animal, at replay speed;
- ``"stationary"``: one remote place, held (also ``"boundary"``: the same, put
  across a time-block boundary of the held-out score);
- ``"fragmented"``: three remote places in turn;
- ``"close_20"``, ``"close_40"``, ``"close_80"``: one place that far from the
  animal, held;
- ``"short"``: a remote place for 16 ms;
- ``"local_burst"``: the animal's own place, at a higher rate;
- ``"nonspatial"``: each tetrode its own random place every bin, at a higher
  rate: a burst whose tetrodes share no spatial content, which a decoder may
  still call non-local by chance;
- ``"shared_random"``: one random place every bin for all tetrodes (fragmented
  content the population shares);
- ``"silent"``: no spikes at all.

The detector is the real decodes' (settings, transitions, Local kernel), fit by
EM on the running bins as a real decode is fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from non_local_detector import NonLocalClusterlessDetector
from non_local_detector.environment import Environment
from non_local_detector.simulate.simulate import simulate_multiunit_with_place_fields

from continuum_swr_replay.analysis.event_evidence.tetrode_holdout import time_blocks
from continuum_swr_replay.decoding.config import NonLocalDecodingConfig
from continuum_swr_replay.decoding.fit import algorithm_params
from continuum_swr_replay.decoding.transitions import nonlocal_detector_kwargs
from continuum_swr_replay.parameters import PLACE_BIN_SIZE_1D, SAMPLING_FREQUENCY

TRACK_CM = 175.0
RUN_CM_PER_S = 25.0
REST_S = 24.0
N_LAPS = 6
N_TETRODES = 8
"""Tetrodes with fields along the whole track; one more covers :data:`REGION_CM`."""
CELLS_PER_TETRODE = 10
PLACE_SD_CM = 8.0
"""Place-field width: with 80 cells, every place is covered by several tetrodes."""
REGION_CM = (80.0, 95.0)
EVENT_S = 0.12
SHORT_S = 0.016
REPLAY_CM_PER_S = 500.0
RUN_RATE_HZ = 30.0
REST_RATE_HZ = 5.0
EVENT_RATE_HZ = 60.0
EVENT_KINDS = (
    "sweep",
    "stationary",
    "fragmented",
    "close_20",
    "close_40",
    "close_80",
    "short",
    "local_burst",
    "silent",
)
CHANCE_KINDS = ("nonspatial",) * 6 + ("shared_random",) * 2 + ("local_burst",) * 2
"""The local-only recording's events: no remote content anywhere."""


@dataclass(frozen=True)
class Recording:
    time: np.ndarray
    position: np.ndarray  # (n_time,), cm
    is_running: np.ndarray
    spike_times: list[np.ndarray]
    marks: list[np.ndarray]
    events: pd.DataFrame  # kind, s, e (bins, inclusive)
    region_tetrode: int


def _layout(n_laps: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Position, running mask and each rest's (first, last) bin."""
    dt = 1 / SAMPLING_FREQUENCY
    position, running, rests = [], [], []
    start = 0
    for lap in range(n_laps):
        lap_bins = int(TRACK_CM / RUN_CM_PER_S / dt)
        run = np.linspace(0, TRACK_CM, lap_bins)
        if lap % 2:
            run = run[::-1]
        rest_bins = int(REST_S / dt)
        position += [run, np.full(rest_bins, run[-1])]
        running += [np.ones(lap_bins, bool), np.zeros(rest_bins, bool)]
        rests.append((start + lap_bins, start + lap_bins + rest_bins - 1))
        start += lap_bins + rest_bins
    return np.concatenate(position), np.concatenate(running), rests


def _content(kind: str, animal_cm: float, n: int, rng) -> tuple[np.ndarray, float]:
    """The represented place over an event's ``n`` bins, and its peak rate."""
    dt = 1 / SAMPLING_FREQUENCY
    far = TRACK_CM - animal_cm
    inward = 1.0 if animal_cm < TRACK_CM / 2 else -1.0  # toward the far end
    if kind == "sweep":
        start = far - inward * 90
        return start + inward * REPLAY_CM_PER_S * dt * np.arange(n), EVENT_RATE_HZ
    if kind in ("stationary", "boundary", "short"):
        return np.full(n, far - inward * 20), EVENT_RATE_HZ
    if kind == "fragmented":
        places = far - inward * np.array([20.0, 60.0, 40.0])
        return places[np.arange(n) * 3 // n], EVENT_RATE_HZ
    if kind.startswith("close_"):
        return np.full(n, animal_cm + inward * float(kind.split("_")[1])), EVENT_RATE_HZ
    if kind == "local_burst":
        return np.full(n, animal_cm), EVENT_RATE_HZ
    if kind in ("nonspatial", "shared_random"):
        return rng.uniform(0, TRACK_CM, n), EVENT_RATE_HZ
    if kind == "silent":
        return np.full(n, animal_cm), 0.0
    raise ValueError(kind)


def simulate_recording(
    kinds: tuple[str, ...] = EVENT_KINDS,
    n_laps: int = N_LAPS,
    boundary_event: bool = True,
    seed: int = 0,
) -> Recording:
    """A recording with ``kinds`` inserted in every rest, in order.

    With ``boundary_event``, a ``"boundary"`` event is also put across every
    boundary of the held-out score's time blocks that falls in a rest.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / SAMPLING_FREQUENCY
    position, running, rests = _layout(n_laps)
    time = np.arange(position.size) * dt
    n_all = N_TETRODES + 1
    represented = np.tile(position, (n_all, 1))  # (n_tetrodes, n_time)
    rate = np.where(running, RUN_RATE_HZ, REST_RATE_HZ)

    events = []
    for first, last in rests:
        gap = (last - first) // (len(kinds) + 1)
        for i, kind in enumerate(kinds):
            n = int((SHORT_S if kind == "short" else EVENT_S) / dt)
            s = first + (i + 1) * gap - n // 2
            events.append((kind, s, s + n - 1))
    if boundary_event:
        blocks = time_blocks(time)
        n = int(EVENT_S / dt)
        for edge in np.flatnonzero(np.diff(blocks)) + 1:
            s = edge - n // 2
            rest = any(first + 500 < s and s + n < last - 500 for first, last in rests)
            clear = all(abs(s - other) > n + 200 for _, other, _ in events)
            if rest and clear:
                events.append(("boundary", s, s + n - 1))
    for kind, s, e in events:
        if kind == "nonspatial":
            for j in range(n_all):
                represented[j, s : e + 1] = _content(kind, position[s], e - s + 1, rng)[
                    0
                ]
        else:
            represented[:, s : e + 1] = _content(kind, position[s], e - s + 1, rng)[0]
        rate[s : e + 1] = _content(kind, position[s], e - s + 1, rng)[1]

    means = np.linspace(5, TRACK_CM - 5, N_TETRODES * CELLS_PER_TETRODE)
    tetrode_means = [means[j::N_TETRODES] for j in range(N_TETRODES)]
    tetrode_means.append(np.linspace(*REGION_CM, CELLS_PER_TETRODE))
    top = rate.max()
    spike_times, marks = [], []
    for j, place_means in enumerate(tetrode_means):
        multiunit = simulate_multiunit_with_place_fields(
            place_means[:, None],
            represented[j][:, None],
            mark_spacing=10,
            n_mark_dims=4,
            place_variance=PLACE_SD_CM**2,
            max_rate=top,
            sampling_frequency=SAMPLING_FREQUENCY,
            rng=rng,
        )
        has = ~np.isnan(multiunit).any(axis=1)
        keep = has & (rng.random(time.size) < rate / top)
        spike_times.append(time[keep])
        marks.append(multiunit[keep])
    return Recording(
        time=time,
        position=position,
        is_running=running,
        spike_times=spike_times,
        marks=marks,
        events=pd.DataFrame(events, columns=["kind", "s", "e"]),
        region_tetrode=N_TETRODES,
    )


def fit_detector(recording: Recording) -> NonLocalClusterlessDetector:
    """The real decodes' detector, fit by EM on the running bins."""
    config = NonLocalDecodingConfig()
    detector = NonLocalClusterlessDetector(
        environments=Environment(place_bin_size=PLACE_BIN_SIZE_1D),
        **algorithm_params(config, "clusterless"),
        **nonlocal_detector_kwargs(config),
    )
    detector.estimate_parameters(
        position_time=recording.time,
        position=recording.position[:, None],
        spike_times=recording.spike_times,
        spike_waveform_features=recording.marks,
        time=recording.time,
        is_training=recording.is_running,
    )
    return detector


def detected_events(
    detector: NonLocalClusterlessDetector, recording: Recording, threshold: float = 0.8
) -> pd.DataFrame:
    """The detector's non-local events at rest: runs of at least 7 bins whose
    non-local probability is above ``threshold``, each labelled by the inserted
    event it overlaps (``"none"`` for a detection over plain rest)."""
    results = detector.predict(
        spike_times=recording.spike_times,
        spike_waveform_features=recording.marks,
        time=recording.time,
        position=recording.position[:, None],
        position_time=recording.time,
    )
    states = list(detector.state_names)
    probability = results["acausal_state_probabilities"].values
    nonlocal_prob = probability[
        :, [i for i, s in enumerate(states) if s.startswith("Non-Local")]
    ].sum(axis=1)
    above = (nonlocal_prob > threshold) & ~recording.is_running
    edges = np.flatnonzero(np.diff(np.r_[0, above.astype(int), 0]))
    starts, stops = edges[::2], edges[1::2] - 1
    keep = stops - starts + 1 >= 7
    rows = []
    for s, e in zip(starts[keep], stops[keep], strict=True):
        overlap = recording.events[
            (recording.events.s <= e) & (recording.events.e >= s)
        ]
        rows.append((overlap.kind.iloc[0] if len(overlap) else "none", s, e))
    return pd.DataFrame(rows, columns=["kind", "s", "e"])
