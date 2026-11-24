"""Count each epoch's immobile time bins inside the well visits by non-local probability.

How the immobile time is spread over the decoder's non-local probability
(Continuous plus Fragmented), for the panel of Figure 2's supplement that shows
the time is almost all near 0 or near 1. The bins are the immobile ones outside
excluded time (``analyzed_time.immobile_analyzed_bins``) inside a well visit
(first poke-in to final poke-out, ``event_definition.well_visits.in_well_visit``,
the visits read from the session cache:
``event_tables.prepare_data.well_visit_times``), counted in 50 equal
probability bins (``analysis.decode_quality.nonlocal_probability.probability_counts``).
For the ``ok`` epochs of an event features run, from the same decodes (each
decode's sha256 must be the one the run recorded), written to
``Processed-Data/nonlocal_probability/counts.parquet`` (or ``--output-dir``)
with a ``manifest.json`` beside it that records each epoch's status and bins,
and the session-cache entries the visits were read from
(``prepare_data.session_cache_record``).

Usage::

    uv run python scripts/decoding/measure_nonlocal_probability.py \\
        --event-features-run e2331f976fd1 --n-workers 8
    uv run python scripts/decoding/measure_nonlocal_probability.py \\
        --event-features-run e2331f976fd1 --epochs chimi20200212_02_r1_HPC \\
        --output-dir /tmp/check
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.decode_quality.nonlocal_probability import (
    PROBABILITY_BIN_EDGES,
    probability_counts,
)
from continuum_swr_replay.paths import NONLOCAL_PROBABILITY_DIR
from continuum_swr_replay.runs.provenance import produced_by

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.event_properties._epoch_counts import run  # noqa: E402


def measure(dot: pd.DataFrame, in_visit: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """One epoch's counts, and its record: the bins counted."""
    counts = probability_counts(dot, in_visit)
    return counts, {"status": "ok", "n_bins": int(counts["n_bins"].sum())}


def main(argv: Sequence[str] | None = None) -> None:
    run(
        argv,
        description=__doc__.splitlines()[0],
        produced_by=produced_by(__file__),
        output_name=NONLOCAL_PROBABILITY_DIR,
        measure=measure,
        counted={
            "probability": "non-local probability (Continuous plus Fragmented)",
            "probability_bins": [float(edge) for edge in PROBABILITY_BIN_EDGES],
        },
    )


if __name__ == "__main__":
    main()
