"""Memoryless sliding-window sequence detection.

A port of the sequence detector of Widloski & Foster (2025), *Replay without
sharp wave ripples in a spatial memory task*, Nature Communications 16,
doi:10.1038/s41467-025-65181-5, onto this project's linearized track graph and
clusterless encoding model.

Each window is decoded independently under a uniform spatial prior, and
sequences are assembled afterwards from continuity criteria. The design,
including every departure from the original, is in
``docs/design/replay_detection_comparison/02-memoryless-sequence.md``.
"""

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)

__all__ = ["MemorylessSequenceConfig"]
