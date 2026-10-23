"""What the replay detection runners read from a saved HPC decode and from each other.

The memoryless, Carey and event-feature runners read the same things, by the
same rules:

- a decode's two saved files, its detector and its ``data_over_time``
  (:func:`input_paths`);
- its HPC spike marks, read as ``load_data`` read them for the decode
  (:data:`MARKS_RULE`), cached per decode file under :data:`MARKS_CACHE_DIR`
  (:func:`load_marks`, :func:`marks_cache_path`) and hashed in their records
  (:func:`marks_sha256`);
- a memoryless run's per-epoch files (:func:`memoryless_output_paths`).

It imports only foundation modules, so a runner reads these without importing
another runner.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray

from continuum_swr_replay.data_loaders.spikes import filter_spike_times, get_hpc_marks
from continuum_swr_replay.data_loaders.utils import EpochKey
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.paths import DETECTION_COMPARISON_DIR
from continuum_swr_replay.tables.io import write_atomically

__all__ = [
    "MARKS_CACHE_DIR",
    "MARKS_RULE",
    "MARKS_RULE_TAG",
    "input_paths",
    "load_marks",
    "marks_cache_path",
    "marks_sha256",
    "memoryless_output_paths",
]

MARKS_RULE = (
    "load_data: get_hpc_marks (tetrode sort groups; coincident spikes counted) "
    "then filter_spike_times over the decode's time span"
)
"""How the marks are read, for the manifest: as ``load_data`` read them."""

MARKS_RULE_TAG = "loader"
""":data:`MARKS_RULE`'s name in a marks cache file; another rule needs another tag."""

MARKS_CACHE_DIR = DETECTION_COMPARISON_DIR / "memoryless" / "marks_cache"
"""Where the runners cache the marks by default: the memoryless runner's
``--cache-dir`` default, which the Carey and event-feature runners share."""


def input_paths(epoch_id: str, root: Path) -> dict[str, Path]:
    """The two saved decode artifacts the detector reads (``decode_paths``)."""
    paths = decode_paths(epoch_id, root)
    return {"detector": paths.detector, "data_over_time": paths.data_over_time}


def marks_cache_path(key: EpochKey, cache_dir: Path, detector_sha256: str) -> Path:
    """Where one decode's marks are cached under :data:`MARKS_RULE`.

    The name carries the rule's tag and the first 16 hex digits of the decode's
    ``_detector.pkl`` sha256, so a cache made for another decode of the epoch,
    or under another rule, is never read.
    """
    return (
        cache_dir / f"{key.epoch_id}_{MARKS_RULE_TAG}_{detector_sha256[:16]}_marks.pkl"
    )


def load_marks(
    key: EpochKey,
    cache_dir: Path,
    span_s: tuple[float, float],
    detector_sha256: str,
) -> tuple[list[NDArray], list[NDArray]]:
    """A decode's HPC spike times and marks, read as ``load_data`` read them.

    ``get_hpc_marks``, then ``filter_spike_times`` over the decode's time span:
    ``load_data`` filters over the position grid, which is the decode's grid.
    Cached per decode file (:func:`marks_cache_path`), written atomically, so a
    worker stopped part way never leaves half a cache for a rerun to read.

    Parameters
    ----------
    key : EpochKey
        The decode.
    cache_dir : Path
        Where the marks are cached.
    span_s : tuple of float
        First and last time of the decode's grid, in seconds; spikes on either
        end are kept.
    detector_sha256 : str
        sha256 of the decode's ``_detector.pkl``, which keys the cache.

    Returns
    -------
    spike_times : list of ndarray, each shape (n_spikes,)
    spike_waveform_features : list of ndarray, each shape (n_spikes, n_features)
        One per tetrode with a spike in ``span_s``, in ``get_hpc_marks``'s order.
    """
    cache = marks_cache_path(key, cache_dir, detector_sha256)
    if cache.exists():
        cached: tuple[list[NDArray], list[NDArray]] = joblib.load(cache)
        return cached
    spike_times, spike_waveform_features = get_hpc_marks(key.nwb_file_name)
    in_span_times, in_span_features = filter_spike_times(
        {"HPC": spike_times},
        {"HPC": spike_waveform_features},
        np.asarray(span_s, dtype=np.float64),
    )
    marks = (in_span_times["HPC"], in_span_features["HPC"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_atomically(cache, lambda path: joblib.dump(marks, path))
    return marks


def marks_sha256(spike_times: list, spike_waveform_features: list) -> str:
    """One hash of every electrode's spike times and marks, in order."""
    digest = hashlib.sha256()
    for times, features in zip(spike_times, spike_waveform_features, strict=True):
        digest.update(np.ascontiguousarray(times, dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(features, dtype=np.float64).tobytes())
    return digest.hexdigest()


def memoryless_output_paths(epoch_id: str, output_dir: Path) -> dict[str, Path]:
    """Every file a memoryless run writes for one epoch, by output name."""
    return {
        "events": output_dir / f"{epoch_id}_memoryless_events.parquet",
        "shuffle_draws": output_dir / f"{epoch_id}_memoryless_shuffle_draws.parquet",
        "observed": output_dir / f"{epoch_id}_memoryless_observed.parquet",
        "candidates": output_dir / f"{epoch_id}_memoryless_candidates.pkl",
    }
