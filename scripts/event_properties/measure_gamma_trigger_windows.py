"""Store each epoch's slow- and fast-gamma windows around event starts.

The gamma traces of Figure 3's supplement (S3 i, j) are drawn from this output.
They are made as Figure 2g's ripple, MUA, theta and speed traces are
(``scripts/event_properties/measure_trigger_windows.py``): one pass over each
decode frame stores the window of each band around the start of every immobile
non-local event, by SWR overlap
(``analysis.event_properties.triggered_analysis.event_start_windows``), and a
window that touches excluded time is left out. A band's signal is its z-scored
power at every bin of the decode grid, read from the gamma run
(``Processed-Data/event_gamma/{epoch_id}_gamma_z.parquet``,
``scripts/event_properties/measure_event_gamma.py``), whose time grid must be
the frame's. Every immobile non-local event is stored, the ones that start
outside a well visit too (the figure selects the events that start inside one,
by the canonical table). For the ``ok`` epochs of an event features run, from
the same decodes (each decode's sha256 must be the one the run recorded),
written to ``Processed-Data/event_gamma_trigger_windows/windows.parquet`` (or
``--output-dir``) with a ``manifest.json`` beside it.

Usage::

    uv run python scripts/event_properties/measure_gamma_trigger_windows.py \\
        --event-features-run e2331f976fd1 --n-workers 8
    uv run python scripts/event_properties/measure_gamma_trigger_windows.py \\
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
from continuum_swr_replay.analysis.event_definition.swr_source import (
    swr_epoch_id_of_session,
)
from continuum_swr_replay.analysis.event_properties.gamma_power import GAMMA_BANDS
from continuum_swr_replay.analysis.event_properties.triggered_analysis import (
    TRIGGERED_WINDOW_S,
    event_start_windows,
    frame_event_counts,
)
from continuum_swr_replay.parameters import SAMPLING_FREQUENCY
from continuum_swr_replay.paths import (
    EVENT_GAMMA_DIR,
    EVENT_GAMMA_TRIGGER_WINDOWS_DIR,
    processed_data_dir,
)
from continuum_swr_replay.runs.io import sha256_file
from continuum_swr_replay.runs.provenance import produced_by

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.event_properties._epoch_counts import run  # noqa: E402

SIGNALS = tuple(f"{band}_z" for band in GAMMA_BANDS)
"""The gamma run's per-bin columns, and the stored windows' signals."""


def measure(dot: pd.DataFrame, in_visit: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """One epoch's gamma windows around its events' starts
    (``event_start_windows``) and its record: the frame's events by SWR
    overlap and the gamma trace read.

    Raises
    ------
    ValueError
        If the gamma run's time grid is not the decode frame's.
    """
    identifier = swr_epoch_id_of_session(
        *(dot[column].iloc[0] for column in ("animal", "day", "epoch"))
    )
    path = processed_data_dir() / EVENT_GAMMA_DIR / f"{identifier}_gamma_z.parquet"
    trace = pd.read_parquet(path)
    if not np.array_equal(trace["time"].to_numpy(), dot.index.to_numpy()):
        raise ValueError(f"{path}: the time grid is not the decode frame's")
    frame = dot.assign(**{signal: trace[signal].to_numpy() for signal in SIGNALS})
    event_types = get_event_types(dot)
    windows = event_start_windows(frame, event_types, SIGNALS)
    return windows, {
        "status": "ok",
        "n_events": frame_event_counts(event_types),
        "gamma_z_sha256": sha256_file(path),
    }


def main(argv: Sequence[str] | None = None) -> None:
    run(
        argv,
        description=__doc__.splitlines()[0],
        produced_by=produced_by(__file__),
        output_name=EVENT_GAMMA_TRIGGER_WINDOWS_DIR,
        measure=measure,
        counted={
            "bins": "every immobile non-local event's start (those outside a well "
            "visit too); a window that touches excluded time is left out",
            "windows": {
                "signals": list(SIGNALS),
                "gamma_run": EVENT_GAMMA_DIR,
                "window_s": TRIGGERED_WINDOW_S,
                "lag_s": "the column's offset in samples over "
                f"{SAMPLING_FREQUENCY:g} Hz",
                "centered_on": "each event's start",
            },
        },
        output_file="windows.parquet",
    )


if __name__ == "__main__":
    main()
