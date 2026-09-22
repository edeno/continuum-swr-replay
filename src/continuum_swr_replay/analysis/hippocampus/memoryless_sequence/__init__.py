"""Memoryless sliding-window sequence detection.

A port of the sequence detector of Widloski & Foster (2025), *Replay without
sharp wave ripples in a spatial memory task*, Nature Communications 16,
doi:10.1038/s41467-025-65181-5, onto this project's linearized track graph and
encoding models.

Each window is decoded independently under a uniform spatial prior, and
sequences are assembled afterwards from continuity criteria. The design,
including every departure from the original, is in
``docs/design/replay_detection_comparison/02-memoryless-sequence.md``.

The pipeline, in order: an ``EpochLikelihood`` from one of the
``*_epoch_log_likelihood`` functions; ``aggregate_windows`` and
``normalize_windows`` to get one posterior per window; ``window_midpoints_s``
to timestamp them; ``interior_bin_distances_cm`` for the track geometry; and
``detect_candidate_sequences`` for everything downstream of decoding.
"""

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.detect import (
    CandidateSequences,
    detect_candidate_sequences,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    EpochLikelihood,
    clusterless_epoch_log_likelihood,
    interior_bin_distances_cm,
    load_clusterless_encoding_model,
    sorted_spikes_epoch_log_likelihood,
    window_midpoints_s,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.statistics import (
    aggregate_windows,
    normalize_windows,
)

__all__ = [
    "CandidateSequences",
    "EpochLikelihood",
    "MemorylessSequenceConfig",
    "aggregate_windows",
    "clusterless_epoch_log_likelihood",
    "detect_candidate_sequences",
    "interior_bin_distances_cm",
    "load_clusterless_encoding_model",
    "normalize_windows",
    "sorted_spikes_epoch_log_likelihood",
    "window_midpoints_s",
]
