"""Synthetic epochs for the artifact detector's tests.

One epoch is 30 s at 1000 Hz: 20 CA1 channels and one reference electrode of
white noise (SD ``NOISE``), and 10 tetrodes with 20 Hz Poisson spikes whose
4-D marks sit in a tight cluster per tetrode. Each test adds one artifact or
one SWR-like event on top. Times start at a Unix-like ``BASE_TIME_S``, as
Spyglass stores them, so nothing can assume time 0. Offline: no database.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt

FS = 1000.0
BASE_TIME_S = 1.6e9
DURATION_S = 30.0
N_CHANNELS = 20
N_TETRODES = 10
NOISE = 50.0
"""White-noise SD of every LFP channel, in the LFP's raw units."""
SPIKE_RATE_HZ = 20.0
MARK_SD = 15.0
"""SD of a tetrode's in-distribution marks, in µV."""


@dataclass
class SyntheticEpoch:
    """One epoch's detector inputs, made from a seed."""

    seed: int = 0
    time: NDArray[np.float64] = field(init=False)
    ca1: NDArray[np.float64] = field(init=False)
    reference: NDArray[np.float64] = field(init=False)
    spike_times: list[NDArray[np.float64]] = field(init=False)
    spike_marks: list[NDArray[np.float64]] = field(init=False)
    mark_centers: NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        n_time = int(DURATION_S * FS)
        self.time = BASE_TIME_S + np.arange(n_time) / FS
        self.ca1 = rng.normal(0.0, NOISE, (n_time, N_CHANNELS))
        self.reference = rng.normal(0.0, NOISE, (n_time, 1))
        self.mark_centers = rng.uniform(80.0, 200.0, (N_TETRODES, 4))
        self.spike_times, self.spike_marks = [], []
        for tetrode in range(N_TETRODES):
            n_spikes = rng.poisson(SPIKE_RATE_HZ * DURATION_S)
            times = np.sort(rng.uniform(self.time[0], self.time[-1], n_spikes))
            marks = rng.normal(self.mark_centers[tetrode], MARK_SD, (n_spikes, 4))
            self.spike_times.append(times)
            self.spike_marks.append(marks)

    def sample(self, seconds: float) -> int:
        """The sample index ``seconds`` after the epoch's start."""
        return int(round(seconds * FS))

    def add_step(
        self,
        seconds: float,
        channels: list[int],
        amplitude: float,
        *,
        on_reference: bool = False,
        tau_s: float = 0.2,
    ) -> None:
        """A level step at ``seconds`` that recovers exponentially (tau_s)."""
        start = self.sample(seconds)
        decay = amplitude * np.exp(-np.arange(self.time.size - start) / (tau_s * FS))
        if on_reference:
            self.reference[start:, 0] += decay
        self.ca1[start:, channels] += decay[:, None]

    def add_ripple_burst(
        self, seconds: float, amplitude: float, *, reference_gain: float = 0.05
    ) -> None:
        """A 150 Hz burst on every CA1 channel under a 15-ms Gaussian envelope.

        Its cycles jump by many robust units within 5 ms, but its level
        returns at once, as at a giant SWR's onset. The reference sees it only
        weakly.
        """
        t = (np.arange(self.time.size) - self.sample(seconds)) / FS
        envelope = np.exp(-0.5 * (t / 0.015) ** 2)
        phases = np.linspace(0.0, np.pi / 2, N_CHANNELS)
        burst = (
            amplitude
            * envelope[:, None]
            * np.sin(2 * np.pi * 150.0 * t[:, None] + phases)
        )
        self.ca1 += burst
        self.reference[:, 0] += reference_gain * burst[:, 0]

    def add_band_noise(
        self,
        start_s: float,
        stop_s: float,
        channels: list[int],
        rms: float,
        seed: int = 1,
    ) -> None:
        """150-250 Hz noise of ``rms`` on ``channels`` over [start_s, stop_s)."""
        rng = np.random.default_rng(seed)
        sos = butter(4, [150.0, 250.0], btype="band", fs=FS, output="sos")
        noise = sosfiltfilt(
            sos, rng.normal(size=(self.time.size, len(channels))), axis=0
        )
        noise *= rms / noise.std()
        gate = np.zeros(self.time.size, dtype=bool)
        gate[self.sample(start_s) : self.sample(stop_s)] = True
        self.ca1[:, channels] += noise * gate[:, None]

    def add_spike_line(
        self, seconds: float, tetrodes: list[int], *, out_of_distribution: bool
    ) -> None:
        """One spike on each of ``tetrodes`` within 40 µs of ``seconds``.

        Out of distribution, the marks are 1500 µV on every channel, far from
        each tetrode's cluster; otherwise they sit at the cluster's center.
        """
        for offset, tetrode in enumerate(tetrodes):
            spike_time = self.time[0] + seconds + offset * 8e-6
            mark = (
                np.full(4, 1500.0)
                if out_of_distribution
                else self.mark_centers[tetrode].copy()
            )
            self._add_spike(tetrode, spike_time, mark)

    def add_spikes(
        self, tetrode: int, times_s: NDArray[np.float64], mark: NDArray[np.float64]
    ) -> None:
        """Spikes at ``times_s`` after the start, each with the same ``mark``."""
        for seconds in np.atleast_1d(times_s):
            self._add_spike(tetrode, self.time[0] + seconds, np.asarray(mark, float))

    def _add_spike(self, tetrode: int, spike_time: float, mark: NDArray) -> None:
        times = np.append(self.spike_times[tetrode], spike_time)
        marks = np.vstack([self.spike_marks[tetrode], mark])
        order = np.argsort(times, kind="stable")
        self.spike_times[tetrode] = times[order]
        self.spike_marks[tetrode] = marks[order]

    def at(self, seconds: float) -> float:
        """The time ``seconds`` after the epoch's start."""
        return float(self.time[0] + seconds)
