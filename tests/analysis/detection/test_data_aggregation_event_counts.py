"""Regression tests for ``compute_swr_nl_event_counts`` edge cases.

The helper used to crash when no event_col_a row was left
(``int(empty_series.sum())`` → TypeError). The frames here are shaped as the
read (``analyzed_time.with_swr_source``) hands them over: it holds every label to
the whole-event speed rule and the Non-Local probability contract, so the
helper neither filters nor validates them (``tests/analysis/event_definition/test_swr_source.py``).
"""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.detection.nonlocal_events import (
    compute_swr_nl_event_counts,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["Non-Local_Continuous_prob"] = (
        df["non_local_event"].gt(0).map({True: 0.9, False: 0.1})
    )
    df["Non-Local_Fragmented_prob"] = 0.0
    return df.set_index("time").rename_axis(None)


class TestEmptyA:
    def test_no_a_event_does_not_crash(self):
        # No non-local event is left (the read dropped the moving ones).
        # Used to raise TypeError.
        df = _frame(
            [
                {
                    "time": 0.0,
                    "animal": "m",
                    "day": 1,
                    "epoch": "e",
                    "head_speed": 50.0,
                    "non_local_event": 0,
                    "SWR_event": 0,
                },
                {
                    "time": 0.5,
                    "animal": "m",
                    "day": 1,
                    "epoch": "e",
                    "head_speed": 0.0,
                    "non_local_event": 0,
                    "SWR_event": 1,
                },
                {
                    "time": 1.0,
                    "animal": "m",
                    "day": 1,
                    "epoch": "e",
                    "head_speed": 0.0,
                    "non_local_event": 0,
                    "SWR_event": 1,
                },
            ]
        )
        out = compute_swr_nl_event_counts(df)
        assert out["n_a"] == 0
        assert out["n_a_with_b"] == 0
        assert out["n_b"] == 1


class TestCrossSessionTimeCollision:
    def test_shared_timestamp_across_sessions_does_not_leak_overlap(self):
        # Two sessions, both starting at t=0.0:
        #  - Session 1 has an NL+SWR overlap at t=0.0.
        #  - Session 2 has only an SWR at t=0.0 (no NL event).
        # The bare ``index.isin`` form would falsely count session
        # 2's SWR as overlapping the A-set because the time index 0.0
        # also belongs to session 1's NL event.
        df = _frame(
            [
                # Session 1 — NL and SWR overlap.
                {
                    "time": 0.0,
                    "animal": "s1",
                    "day": 1,
                    "epoch": "e",
                    "head_speed": 0.0,
                    "non_local_event": 1,
                    "SWR_event": 1,
                },
                {
                    "time": 0.5,
                    "animal": "s1",
                    "day": 1,
                    "epoch": "e",
                    "head_speed": 0.0,
                    "non_local_event": 1,
                    "SWR_event": 1,
                },
                # Session 2 — an SWR alone, at the same times.
                {
                    "time": 0.0,
                    "animal": "s2",
                    "day": 1,
                    "epoch": "e",
                    "head_speed": 0.0,
                    "non_local_event": 0,
                    "SWR_event": 1,
                },
                {
                    "time": 0.5,
                    "animal": "s2",
                    "day": 1,
                    "epoch": "e",
                    "head_speed": 0.0,
                    "non_local_event": 0,
                    "SWR_event": 1,
                },
            ]
        )
        out = compute_swr_nl_event_counts(df)
        # A is just session 1's NL event; B is both sessions' SWRs.
        assert out["n_a"] == 1
        assert out["n_b"] == 2
        assert out["n_b_with_a"] == 1, (
            f"n_b_with_a={out['n_b_with_a']} — bare time-index "
            "matching leaked session 2's SWR into session 1's overlap count."
        )


class TestBoundaryTouchOverlap:
    """Regression: label-based ``.any()`` checks counted boundary
    contact (a single shared timestamp from the inclusive
    ``get_labels`` writer) as overlap. The strict positive-duration
    semantics adopted by ``classify_swr_overlap`` and
    ``get_overlap_times`` should apply here too — events whose
    intervals only touch at a boundary share zero duration and must
    not be classified as overlapping.
    """

    def test_boundary_touch_yields_zero_overlap(self):
        # NL labels [0.0, 1.0]; SWR labels [1.0, 2.0]. They share only
        # the boundary timestamp at t=1.0.
        df = _frame(
            [
                {
                    "time": 0.0,
                    "animal": "a",
                    "day": 1,
                    "epoch": "e1",
                    "head_speed": 0.0,
                    "non_local_event": 1,
                    "SWR_event": 0,
                },
                {
                    "time": 0.5,
                    "animal": "a",
                    "day": 1,
                    "epoch": "e1",
                    "head_speed": 0.0,
                    "non_local_event": 1,
                    "SWR_event": 0,
                },
                {
                    "time": 1.0,
                    "animal": "a",
                    "day": 1,
                    "epoch": "e1",
                    "head_speed": 0.0,
                    "non_local_event": 1,
                    "SWR_event": 2,
                },
                {
                    "time": 1.5,
                    "animal": "a",
                    "day": 1,
                    "epoch": "e1",
                    "head_speed": 0.0,
                    "non_local_event": 0,
                    "SWR_event": 2,
                },
                {
                    "time": 2.0,
                    "animal": "a",
                    "day": 1,
                    "epoch": "e1",
                    "head_speed": 0.0,
                    "non_local_event": 0,
                    "SWR_event": 2,
                },
            ]
        )
        out = compute_swr_nl_event_counts(df)
        assert out["n_a"] == 1
        assert out["n_b"] == 1
        # Boundary touch only — strict semantics report zero overlap.
        assert out["n_a_with_b"] == 0, (
            f"n_a_with_b={out['n_a_with_b']} — NL [0.0, 1.0] touches "
            "SWR [1.0, 2.0] only at the boundary timestamp; strict "
            "positive-duration overlap must report zero."
        )
        assert out["n_b_with_a"] == 0

    def test_positive_duration_overlap_still_detected(self):
        """Sanity: any positive-duration overlap, however small,
        must still be counted. Differs from the boundary-touch test
        only in extending NL's labels by one bin into SWR's range."""
        df = _frame(
            [
                {
                    "time": 0.0,
                    "animal": "a",
                    "day": 1,
                    "epoch": "e1",
                    "head_speed": 0.0,
                    "non_local_event": 1,
                    "SWR_event": 0,
                },
                {
                    "time": 0.5,
                    "animal": "a",
                    "day": 1,
                    "epoch": "e1",
                    "head_speed": 0.0,
                    "non_local_event": 1,
                    "SWR_event": 0,
                },
                {
                    "time": 1.0,
                    "animal": "a",
                    "day": 1,
                    "epoch": "e1",
                    "head_speed": 0.0,
                    "non_local_event": 1,
                    "SWR_event": 2,
                },
                # NL extends to t=1.5 (overlap of [1.0, 1.5] with SWR).
                {
                    "time": 1.5,
                    "animal": "a",
                    "day": 1,
                    "epoch": "e1",
                    "head_speed": 0.0,
                    "non_local_event": 1,
                    "SWR_event": 2,
                },
                {
                    "time": 2.0,
                    "animal": "a",
                    "day": 1,
                    "epoch": "e1",
                    "head_speed": 0.0,
                    "non_local_event": 0,
                    "SWR_event": 2,
                },
            ]
        )
        out = compute_swr_nl_event_counts(df)
        assert out["n_a_with_b"] == 1
        assert out["n_b_with_a"] == 1
