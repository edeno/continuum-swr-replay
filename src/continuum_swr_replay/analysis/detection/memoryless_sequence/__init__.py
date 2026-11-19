"""Memoryless sliding-window sequence detection.

A port of the sequence detector of Widloski & Foster (2025), *Replay without
sharp wave ripples in a spatial memory task*, Nature Communications 16,
doi:10.1038/s41467-025-65181-5, onto this project's linearized track graph and
encoding models.

Each window is decoded independently under a uniform spatial prior, and
sequences are assembled afterward from continuity criteria.

The pipeline, in order: an ``EpochLikelihood`` from one of the
``*_epoch_log_likelihood`` functions; ``aggregate_windows`` and
``normalize_windows`` to get one posterior per window; ``window_midpoints_s``
to timestamp them; ``decoding.encoding_model.interior_bin_distances_cm`` for
the track geometry; and
``detect_candidate_sequences`` for everything downstream of decoding. Then
``shuffle_eligible`` picks the candidates to test, ``shuffle_candidate`` tests
each against encoding-pair shuffles, and ``build_memoryless_event_table``
writes one row per candidate. ``run_epoch`` does all of it, in that order, for
one epoch.
"""

from continuum_swr_replay.analysis.detection.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.detect import (
    CandidateSequences,
    detect_candidate_sequences,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.epoch import (
    EpochDetection,
    run_epoch,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.event_table import (
    MEMORYLESS_EVENT_TABLE_DTYPES,
    SequenceIdentity,
    build_memoryless_event_table,
    build_shuffle_draw_table,
    shuffle_eligible,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.likelihood import (
    EpochLikelihood,
    clusterless_epoch_log_likelihood,
    sorted_spikes_epoch_log_likelihood,
    window_midpoints_s,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.shuffle import (
    ShuffleResult,
    ShuffleSeed,
    shuffle_candidate,
    shuffle_candidate_sorted,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.window_statistics import (
    aggregate_windows,
    normalize_windows,
)

__all__ = [
    "MEMORYLESS_EVENT_TABLE_DTYPES",
    "CandidateSequences",
    "EpochDetection",
    "EpochLikelihood",
    "MemorylessSequenceConfig",
    "SequenceIdentity",
    "ShuffleResult",
    "ShuffleSeed",
    "aggregate_windows",
    "build_memoryless_event_table",
    "build_shuffle_draw_table",
    "clusterless_epoch_log_likelihood",
    "detect_candidate_sequences",
    "normalize_windows",
    "run_epoch",
    "shuffle_candidate",
    "shuffle_candidate_sorted",
    "shuffle_eligible",
    "sorted_spikes_epoch_log_likelihood",
    "window_midpoints_s",
]
