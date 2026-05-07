"""Regression test for ``compute_event_timing_offsets`` empty schema.

The pre-fix code returned ``pd.DataFrame(rows)`` directly when no
overlap pairs were found — a frame with no columns. Plot helpers
that index ``offsets["start_diff"]`` would then raise ``KeyError``
on ordinary no-overlap inputs (e.g. a session where SWRs and
non-local events never co-occur).

Fix returns a frame with the canonical empty schema instead.
"""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.hippocampus.event_analysis import (
    compute_event_timing_offsets,
)


class TestEmptyOverlapSchema:
    def test_no_overlap_returns_schema_complete_empty_frame(self):
        # Two events of each type but completely separated in time
        # → no overlap pairs.
        df = pd.DataFrame(
            {
                "animal": ["m"] * 6,
                "day": [1] * 6,
                "epoch": ["e"] * 6,
                "SWR_event": [1, 1, 0, 0, 0, 0],
                "non_local_event": [0, 0, 0, 0, 1, 1],
            },
            index=[0.0, 0.1, 1.0, 1.1, 5.0, 5.1],
        )
        out = compute_event_timing_offsets(df, "SWR_event", "non_local_event")
        assert out.empty
        # Indexing the documented schema must not raise.
        for col in (
            "animal", "day", "epoch",
            "SWR_event", "non_local_event",
            "start_diff", "end_diff",
        ):
            assert col in out.columns, (
                f"missing column {col!r}; downstream "
                "plot_event_timing_offsets indexes start_diff and "
                "would crash on the bare-empty frame the function "
                "used to return."
            )

    def test_boundary_touch_yields_no_overlap_row(self):
        """Regression: a SWR ending exactly at a non-local event's
        start (or vice versa) shares zero positive duration. The
        previous inclusive ``>=`` / ``<=`` overlap predicate emitted
        a phantom timing-offset row at start_diff/end_diff = 0 for
        such back-to-back intervals. Strict positive-duration overlap
        must not generate that row (matches the convention adopted
        by ``classify_swr_overlap`` and ``get_overlap_times``)."""
        # SWR labels [0.0, 1.0]; NL labels [1.0, 2.0]; share t=1.0.
        df = pd.DataFrame(
            {
                "animal": ["a"] * 5,
                "day": [1] * 5,
                "epoch": ["e1"] * 5,
                "SWR_event": [1, 1, 1, 0, 0],
                "non_local_event": [0, 0, 2, 2, 2],
            },
            index=[0.0, 0.5, 1.0, 1.5, 2.0],
        )
        out = compute_event_timing_offsets(df, "SWR_event", "non_local_event")
        assert len(out) == 0, (
            f"expected 0 rows on boundary-touch input, got {len(out)}: "
            f"{out.to_dict('records') if len(out) else ''}"
        )
