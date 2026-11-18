"""A clusterless recording with known content, for the held-out tetrode score.

An animal runs laps of a linear track and rests at each end. Every tetrode's
cells have place fields spread along the whole track (as real tetrodes do), and
each cell fires at the place it represents: the animal's own position, except
in inserted events during rest, where the represented place is

- ``"sweep"``: a trajectory far from the animal, at replay speed;
- ``"stationary"``: one remote place, held;
- ``"local_burst"``: the animal's own place, at a higher rate (local content, a
  negative case).

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

from continuum_swr_replay.decoding.config import NonLocalDecodingConfig
from continuum_swr_replay.decoding.fit import algorithm_params
from continuum_swr_replay.decoding.transitions import nonlocal_detector_kwargs
from continuum_swr_replay.parameters import PLACE_BIN_SIZE_1D, SAMPLING_FREQUENCY

TRACK_CM = 175.0
RUN_CM_PER_S = 25.0
REST_S = 20.0
N_LAPS = 6
N_TETRODES = 8
CELLS_PER_TETRODE = 5
EVENT_S = 0.12
REPLAY_CM_PER_S = 500.0
REST_RATE_HZ = 5.0
"""Each cell's peak rate at rest (local firing); running cells fire at 30 Hz."""
RUN_RATE_HZ = 30.0
EVENT_RATE_HZ = 60.0
EVENT_KINDS = ("sweep", "stationary", "local_burst")


@dataclass(frozen=True)
class Recording:
    time: np.ndarray
    position: np.ndarray  # (n_time,), cm
    is_running: np.ndarray
    spike_times: list[np.ndarray]
    marks: list[np.ndarray]
    events: pd.DataFrame  # kind, s, e (bins, inclusive)


def _spikes(place_means, represented, rate, rng, time):
    """Each tetrode's spikes for a stretch, given the represented place."""
    multiunit = simulate_multiunit_with_place_fields(
        place_means[:, None],
        represented[:, None],
        mark_spacing=10,
        n_mark_dims=4,
        max_rate=rate,
        sampling_frequency=SAMPLING_FREQUENCY,
        rng=rng,
    )
    has = ~np.isnan(multiunit).any(axis=1)
    return time[has], multiunit[has]


def simulate_recording(seed: int = 0) -> Recording:
    rng = np.random.default_rng(seed)
    dt = 1 / SAMPLING_FREQUENCY
    means = np.linspace(5, TRACK_CM - 5, N_TETRODES * CELLS_PER_TETRODE)
    # interleaved, so each tetrode covers the whole track
    tetrode_means = [means[j::N_TETRODES] for j in range(N_TETRODES)]

    pieces = []  # (position, represented, rate, running)
    events = []
    start = 0
    end_cm = 0.0
    for lap in range(N_LAPS):
        lap_bins = int(TRACK_CM / RUN_CM_PER_S / dt)
        position = np.linspace(0, TRACK_CM, lap_bins)
        if lap % 2:
            position = position[::-1]
        pieces.append((position, position, RUN_RATE_HZ, True))
        start += lap_bins
        end_cm = position[-1]
        rest_bins = int(REST_S / dt)
        rest = np.full(rest_bins, end_cm)
        represented = rest.copy()
        rate = np.full(rest_bins, REST_RATE_HZ)
        event_bins = int(EVENT_S / dt)
        for i, kind in enumerate(EVENT_KINDS * 2):
            s = int((i + 1) * rest_bins / 7)
            span = slice(s, s + event_bins)
            far = TRACK_CM - end_cm  # the other end
            if kind == "sweep":
                direction = 1 if far > end_cm else -1
                represented[span] = (
                    far - direction * 70 + direction * REPLAY_CM_PER_S * dt
                    * np.arange(event_bins)
                )  # fmt: skip
            elif kind == "stationary":
                represented[span] = far + (-20 if far > end_cm else 20)
            rate[span] = EVENT_RATE_HZ
            events.append((kind, start + s, start + s + event_bins - 1))
        pieces.append((rest, represented, rate, False))
        start += rest_bins

    position = np.concatenate([p[0] for p in pieces])
    represented = np.concatenate([p[1] for p in pieces])
    running = np.concatenate([np.full(p[0].size, p[3]) for p in pieces])
    rate = np.concatenate(
        [np.full(p[0].size, p[2]) if np.isscalar(p[2]) else p[2] for p in pieces]
    )
    time = np.arange(position.size) * dt
    spike_times, marks = [], []
    for means_j in tetrode_means:
        # rates vary in time: simulate at the top rate and thin
        st, mk = _spikes(means_j, represented, rate.max(), rng, time)
        keep = rng.random(st.size) < rate[np.round(st / dt).astype(int)] / rate.max()
        spike_times.append(st[keep])
        marks.append(mk[keep])
    return Recording(
        time=time,
        position=position,
        is_running=running,
        spike_times=spike_times,
        marks=marks,
        events=pd.DataFrame(events, columns=["kind", "s", "e"]),
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
