"""The artifact mask: the detector's cores, padded from one table.

The detector stores unpadded cores
(``hippocampus.artifacts.detect.detect_artifacts``). One table,
:data:`PADDING_S`, pads them, and :func:`artifact_intervals` gives the one mask
every part of the project uses: the SWR detectors' normalization and event
drops, MUA detection, the decoders (``is_missing`` and the encoding
exclusion), the Widloski runner and every analysis. Artifact time is
unobserved there.

Intervals are closed, ``[start_time, end_time]`` in seconds, the project's
convention: a time on an endpoint is masked. The generic closed-interval
functions (merging, membership, subtraction, padding, overlap and coverage)
live in :mod:`continuum_swr_replay.intervals`; ``intervals.pad_intervals``
widens the mask into excluded time (``eligibility.excluded_time``), and
excluded time by a peri-event window's half-width. Design note:
``docs/design/2026-10-09-artifact-detection.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.intervals import as_frame, merge_intervals

__all__ = [
    "PADDING_S",
    "artifact_intervals",
    "core_class",
]

LFP_SIGNATURES = frozenset({"ref", "ca1"})
"""T signatures read from the LFP: a step on the reference or on CA1."""
SPIKE_SIGNATURES = frozenset({"spk"})
"""T signatures read from spikes alone: a spike line."""
S_SIGNATURE = "S"

PADDING_S: Mapping[str, float] = MappingProxyType(
    {
        # Noise spikes on the affected tetrodes leak up to 1 s past some block
        # edges (detector_report.md §5.1 s_pad, validation/S_edge_spike_rates.csv).
        "S": 1.0,
        # The stored ripple FIR rings after a step: its half-length is 0.267 s,
        # and the ringing above 2 robust z around 425 cores has its 99th
        # percentile at 316 ms; 0.3 s covers 96.8-100 % of held-out cores
        # (followup/justification_report.md §1, c2_padding_heldout.csv). The
        # burst of out-of-distribution spikes after a step reaches +269 ms
        # (detector_report.md §2.3).
        "T_lfp": 0.3,
        # 98.8 % of the out-of-distribution marks after a spike line lie within
        # 100 ms (99th percentile 53 ms; justification_report.md §1).
        "T_spike": 0.1,
    }
)
"""Seconds of padding on each side of a core, by its row: ``S``, ``T_lfp`` (a T
core with an LFP signature, ``ref`` or ``ca1``) and ``T_spike`` (a T core with
the spike-line signature alone). No padding follows a step beyond 0.3 s: R8
reruns the headline results with 1 s masked after each step instead."""


def core_class(cores: pd.DataFrame) -> NDArray[np.str_]:
    """Each core's row of the padding table: ``S``, ``T_lfp`` or ``T_spike``.

    Parameters
    ----------
    cores : pd.DataFrame
        With a ``signatures`` column: ``"S"``, or T signatures joined by ``+``.

    Returns
    -------
    ndarray of str, shape (n_cores,)

    Raises
    ------
    ValueError
        If a signature is not one this table knows.
    """
    rows = []
    for signatures in cores["signatures"].astype(str):
        names = set(signatures.split("+"))
        if names == {S_SIGNATURE}:
            rows.append("S")
        elif names <= LFP_SIGNATURES | SPIKE_SIGNATURES and names & LFP_SIGNATURES:
            rows.append("T_lfp")
        elif names == set(SPIKE_SIGNATURES):
            rows.append("T_spike")
        else:
            raise ValueError(
                f"core signatures {signatures!r} have no row in the padding table"
            )
    return np.asarray(rows, dtype=str)


def artifact_intervals(cores: pd.DataFrame, span: tuple[float, float]) -> pd.DataFrame:
    """The artifact mask: the cores padded by :data:`PADDING_S`, merged and
    clipped.

    Parameters
    ----------
    cores : pd.DataFrame
        One epoch's cores, with ``signatures``, ``core_start`` and ``core_end``
        (seconds), as ``detect_artifacts`` returns them.
    span : tuple of float
        The epoch's first and last observed time (seconds); the intervals are
        clipped to it.

    Returns
    -------
    pd.DataFrame
        Columns ``start_time`` and ``end_time`` (float64, seconds), closed,
        disjoint and in time order; no rows when there are no cores.

    Raises
    ------
    ValueError
        If a core's signature has no row in :data:`PADDING_S`.
    """
    pad = np.array([PADDING_S[row] for row in core_class(cores)], dtype=np.float64)
    starts = np.maximum(cores["core_start"].to_numpy(dtype=np.float64) - pad, span[0])
    ends = np.minimum(cores["core_end"].to_numpy(dtype=np.float64) + pad, span[1])
    inside = starts <= ends
    return as_frame(merge_intervals(starts[inside], ends[inside]))
