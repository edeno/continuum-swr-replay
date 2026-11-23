"""Count each epoch's immobile time bins by ripple power, by MUA and by non-local probability.

Where the non-local time sits against the SWR threshold and against MUA: how
much of it has ripple power below the SWR threshold, and the share of bins that
are non-local at each power (Figure 2i), and the same for MUA (Figure 2's
supplement), over the immobile time inside the well visits (the epoch's port
experiences, read from the session cache:
``event_tables.prepare_data.well_visit_times``). One pass over each decode
frame counts, for each signal, the bins by signal bin and non-local probability
bin, 0.8 an edge (``analysis.event_properties.nonlocal_by_signal.rest_signal_counts``),
so every count the figures and the events summary read is a sum over one
table. The manifest records each epoch's immobile bins kept and left out as
outside every visit, and the session-cache entries the visits were read from
(``prepare_data.session_cache_record``). For the ``ok`` epochs of an event
features run, from the same decodes (each decode's sha256 must be the one the
run recorded), written to ``Processed-Data/nonlocal_by_signal/counts.parquet``
(or ``--output-dir``) with a ``manifest.json`` beside it.

Usage::

    uv run python scripts/event_properties/measure_nonlocal_by_signal.py \\
        --event-features-run e2331f976fd1 --n-workers 8
    uv run python scripts/event_properties/measure_nonlocal_by_signal.py \\
        --event-features-run e2331f976fd1 --epochs chimi20200212_02_r1_HPC \\
        --output-dir /tmp/check
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    immobile_analyzed_bins,
)
from continuum_swr_replay.analysis.event_properties.class_or_continuum import (
    KAY_ZSCORE_THRESHOLD,
)
from continuum_swr_replay.analysis.event_properties.nonlocal_by_signal import (
    MUA_LEVEL_Z,
    PROBABILITY_EDGES,
    SIGNALS,
    rest_signal_counts,
)
from continuum_swr_replay.parameters import DEFAULT_NON_LOCAL_THRESHOLD
from continuum_swr_replay.paths import NONLOCAL_BY_SIGNAL_DIR
from continuum_swr_replay.runs.provenance import produced_by

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.event_properties._epoch_counts import run  # noqa: E402


def measure(dot: pd.DataFrame, in_visit: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """One epoch's counts, and its immobile bins kept and left out by the visits."""
    immobile = immobile_analyzed_bins(dot)
    return rest_signal_counts(dot, in_visit), {
        "n_immobile_bins_in_visit": int((immobile & in_visit).sum()),
        "n_immobile_bins_outside_visit": int((immobile & ~in_visit).sum()),
    }


def main(argv: Sequence[str] | None = None) -> None:
    run(
        argv,
        description=__doc__.splitlines()[0],
        produced_by=produced_by(__file__),
        output_name=NONLOCAL_BY_SIGNAL_DIR,
        measure=measure,
        counted={
            "nonlocal": "non-local probability (Continuous plus Fragmented) above "
            f"{DEFAULT_NON_LOCAL_THRESHOLD:g}",
            "signals": {name: column for name, (column, _) in SIGNALS.items()},
            "signal_bins": "0.25 z wide from -2 to 10 with an open bin at each "
            f"end; the Kay threshold ({KAY_ZSCORE_THRESHOLD:g} z) and the MUA "
            f"level ({MUA_LEVEL_Z:g} z) are edges",
            "probability_bins": [float(edge) for edge in PROBABILITY_EDGES],
        },
    )


if __name__ == "__main__":
    main()
