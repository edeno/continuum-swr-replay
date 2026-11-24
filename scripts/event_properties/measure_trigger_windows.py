"""Store each epoch's signal windows around event starts and No-Spike starts.

Figure 2g's traces around the starts of the immobile non-local events (ripple
power, MUA, theta power and head speed, by SWR overlap), its supplement's
animal-by-animal traces of the same windows and its MUA around the start of
No-Spike periods are drawn from this output, as are the frames' event counts
by SWR overlap the canonical event table is checked against. One pass over each
decode frame stores every window
(``analysis.event_properties.triggered_analysis.epoch_trigger_windows``): one
row per window and signal, a column per lag, so a figure reduces the windows
by epoch, by animal or by both. Every immobile non-local event is stored, the
ones that start outside a well visit too (the figures select the events that
start inside one, by the canonical table); a No-Spike period is stored when it
starts inside a well visit (the epoch's port experiences, read from the session
cache: ``event_tables.prepare_data.well_visit_times``). The manifest records
each epoch's events by SWR overlap, its MUA level outside its No-Spike periods
and its periods, and the session-cache entries the visits were read from. For
the ``ok`` epochs of an event features run, from the same decodes (each
decode's sha256 must be the one the run recorded), written to
``Processed-Data/event_trigger_windows/windows.parquet`` (or ``--output-dir``)
with a ``manifest.json`` beside it.

Usage::

    uv run python scripts/event_properties/measure_trigger_windows.py \\
        --event-features-run e2331f976fd1 --n-workers 8
    uv run python scripts/event_properties/measure_trigger_windows.py \\
        --event-features-run e2331f976fd1 --epochs chimi20200212_02_r1_HPC \\
        --output-dir /tmp/check
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.detection.nonlocal_events import get_event_types
from continuum_swr_replay.analysis.event_properties.triggered_analysis import (
    NO_SPIKE_SIGNAL,
    TRIGGERED_SIGNALS,
    TRIGGERED_WINDOW_S,
    epoch_trigger_windows,
)
from continuum_swr_replay.parameters import SAMPLING_FREQUENCY
from continuum_swr_replay.paths import EVENT_TRIGGER_WINDOWS_DIR
from continuum_swr_replay.runs.provenance import produced_by

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.event_properties._epoch_counts import run  # noqa: E402


def measure(
    dot: pd.DataFrame,
    in_visit: np.ndarray,
    visits: tuple[np.ndarray, np.ndarray],
) -> tuple[pd.DataFrame, dict]:
    """One epoch's windows and its record (``epoch_trigger_windows``)."""
    windows, record = epoch_trigger_windows(dot, get_event_types(dot), *visits)
    return windows, {"status": "ok", **record}


def main(argv: Sequence[str] | None = None) -> None:
    run(
        argv,
        description=__doc__.splitlines()[0],
        produced_by=produced_by(__file__),
        output_name=EVENT_TRIGGER_WINDOWS_DIR,
        measure=measure,
        counted={
            "bins": "every immobile non-local event's start (those outside a well "
            "visit too) and every No-Spike period's start inside a well visit; a "
            "window that touches excluded time is left out",
            "windows": {
                "signals": list(TRIGGERED_SIGNALS),
                "no_spike_signal": NO_SPIKE_SIGNAL,
                "window_s": TRIGGERED_WINDOW_S,
                "lag_s": "the column's offset in samples over "
                f"{SAMPLING_FREQUENCY:g} Hz",
                "centered_on": "each event's and each period's start",
            },
        },
        output_file="windows.parquet",
        with_visits=True,
    )


if __name__ == "__main__":
    main()
