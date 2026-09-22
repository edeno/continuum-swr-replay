"""A small clusterless epoch for the memoryless detector's tests.

Fits a real clusterless KDE encoding model from ``non_local_detector`` on
simulated spikes over a Y-shaped track, so tests run against the library's own
conventions rather than a stand-in for them. Shared by the likelihood and
shuffle tests so both read one encoding model.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
)

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    ENCODING_KEY,
    interior_bin_distances_cm,
)

SAMPLING_FREQUENCY = 500.0
N_TIME = 600
N_FEATURES = 2
BOUNDARY_SAMPLES = (100, 140)
"""Samples that start a window, so a spike there tests the half-open support."""


@dataclass
class SimulatedEpoch:
    """Everything a decode needs, from a fitted encoding model."""

    environment: Environment
    encoding_model: dict
    time: np.ndarray
    linear_position: np.ndarray
    spike_times: list[np.ndarray]
    spike_waveform_features: list[np.ndarray]
    detector_path: str


class _StandInDetector:
    """Carries the one attribute the loader reads from a saved detector."""

    def __init__(self, encoding_model: dict) -> None:
        self.encoding_model_ = encoding_model


def _y_track() -> nx.Graph:
    """Three 20 cm arms meeting at a center node."""
    graph = nx.Graph()
    graph.add_edges_from([(0, 3), (1, 3), (2, 3)])
    nx.set_edge_attributes(
        graph, {(0, 3): 20.0, (1, 3): 20.0, (2, 3): 20.0}, "distance"
    )
    nx.set_edge_attributes(graph, {(0, 3): 0, (1, 3): 1, (2, 3): 2}, "edge_id")
    nx.set_node_attributes(
        graph,
        {0: (0.0, 0.0), 1: (40.0, 0.0), 2: (20.0, 20.0), 3: (20.0, 0.0)},
        "pos",
    )
    return graph


def _add_spikes_at(
    spike_times: np.ndarray, features: np.ndarray, at: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Insert spikes at exact times, keeping the arrays sorted and aligned."""
    times = np.concatenate([spike_times, at])
    features = np.concatenate([features, np.full((at.size, N_FEATURES), 100.0)])
    order = np.argsort(times, kind="stable")
    return times[order], features[order]


def _simulate_electrode(
    linear_position: np.ndarray,
    time: np.ndarray,
    field_center_cm: float,
    peak_rate_hz: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Place-tuned spikes with Gaussian waveform features."""
    rng = np.random.default_rng(seed)
    intensity = peak_rate_hz * np.exp(
        -0.5 * ((linear_position - field_center_cm) / 10.0) ** 2
    )
    counts = rng.poisson(intensity / SAMPLING_FREQUENCY)
    sample = np.repeat(np.arange(time.size), counts)
    spike_times = time[sample] + rng.uniform(0.0, 1.0 / SAMPLING_FREQUENCY, sample.size)
    features = rng.normal(100.0, 20.0, size=(sample.size, N_FEATURES))
    order = np.argsort(spike_times)
    return spike_times[order], features[order]


def build_simulated_epoch(directory: Path) -> SimulatedEpoch:
    """Fit a clusterless encoding model on simulated spikes over a Y track.

    Saves it as a stand-in detector in ``directory`` so the loader is
    exercised too.

    For arithmetic tests only: its decode is not a faithful place estimate.
    Position is passed 1-D with a track graph, and the library then fits the
    occupancy on the 1-D values but re-linearizes the positions at spike times
    through the graph (``get_position_at_time``), so the two sit in different
    coordinates; the simulated animal also runs past the last bin. The
    additivity and bin-assignment tests compare the library with itself on the
    same model and do not depend on either.
    """
    time = np.arange(N_TIME) / SAMPLING_FREQUENCY
    # the animal runs back and forth across the linearized track
    linear_position = np.clip(48.0 + 46.0 * np.sin(2 * np.pi * time / 0.6), 0.0, 95.0)

    environment = Environment(
        place_bin_size=4.0,
        track_graph=_y_track(),
        edge_order=[(0, 3), (3, 1), (3, 2)],
        edge_spacing=8.0,  # above zero, so junction placeholder bins exist
    ).fit_place_grid(linear_position[:, None], infer_track_interior=True)

    spike_times, spike_waveform_features = [], []
    for seed, field_center in enumerate([15.0, 50.0, 80.0]):
        times, features = _simulate_electrode(
            linear_position, time, field_center, 400.0, seed + 1
        )
        spike_times.append(times)
        spike_waveform_features.append(features)

    # random times never land on a sample exactly, so put the edge cases in the
    # data: two window boundaries and the epoch's final timestamp
    spike_times[0], spike_waveform_features[0] = _add_spikes_at(
        spike_times[0], spike_waveform_features[0], time[[*BOUNDARY_SAMPLES, -1]]
    )

    encoding_model = fit_clusterless_kde_encoding_model(
        time,
        linear_position[:, None],
        spike_times,
        spike_waveform_features,
        environment,
        sampling_frequency=int(SAMPLING_FREQUENCY),
        disable_progress_bar=True,
    )

    detector_path = Path(directory) / "epoch_detector.pkl"
    joblib.dump(_StandInDetector({ENCODING_KEY: encoding_model}), detector_path)

    return SimulatedEpoch(
        environment=environment,
        encoding_model=encoding_model,
        time=time,
        linear_position=linear_position,
        spike_times=spike_times,
        spike_waveform_features=spike_waveform_features,
        detector_path=str(detector_path),
    )


@dataclass
class TilingTetrode:
    time: np.ndarray
    linear_position: np.ndarray
    spike_times: list[np.ndarray]
    spike_waveform_features: list[np.ndarray]
    encoding_model: dict
    bin_distance_cm: np.ndarray


MOVING_WINDOWS = (115, 265)
"""Windows over which the animal runs from about 69 cm down to about 6 cm."""


def build_tiling_tetrode() -> TilingTetrode:
    """One electrode, five units, each its own waveform and a field of its own.

    The mark is the only thing that says where the animal is, so a shuffle of
    which mark goes with which place removes all of it. Decoded in-sample on a
    plain linear track, the posterior follows the animal to within a bin.
    """
    n_time = 2000
    time = np.arange(n_time) / SAMPLING_FREQUENCY
    linear_position = 38.0 + 34.0 * np.sin(2 * np.pi * time / 1.6)
    # a plain linear track: with a track graph the library would re-linearize
    # the 1-D positions at spike times but not the occupancy
    environment = Environment(place_bin_size=4.0).fit_place_grid(
        linear_position[:, None]
    )

    rng = np.random.default_rng(7)
    times, marks = [], []
    for center, amplitude in zip(
        [8.0, 24.0, 40.0, 56.0, 72.0], [50.0, 100.0, 150.0, 200.0, 250.0], strict=True
    ):
        intensity = 80.0 * np.exp(-0.5 * ((linear_position - center) / 8.0) ** 2)
        sample = np.repeat(
            np.arange(n_time), rng.poisson(intensity / SAMPLING_FREQUENCY)
        )
        times.append(
            time[sample] + rng.uniform(0.0, 1.0 / SAMPLING_FREQUENCY, sample.size)
        )
        marks.append(rng.normal(amplitude, 8.0, size=(sample.size, 2)))
    times, marks = np.concatenate(times), np.concatenate(marks)
    order = np.argsort(times)
    spike_times, features = [times[order]], [marks[order]]

    encoding_model = fit_clusterless_kde_encoding_model(
        time,
        linear_position[:, None],
        spike_times,
        features,
        environment,
        sampling_frequency=int(SAMPLING_FREQUENCY),
        disable_progress_bar=True,
    )
    return TilingTetrode(
        time=time,
        linear_position=linear_position,
        spike_times=spike_times,
        spike_waveform_features=features,
        encoding_model=encoding_model,
        bin_distance_cm=interior_bin_distances_cm(environment),
    )
