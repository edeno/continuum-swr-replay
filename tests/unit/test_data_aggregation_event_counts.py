"""Regression tests for ``compute_swr_nl_event_counts`` edge cases.

The helper used to crash when immobility filtering removed every
event_col_a row (``int(empty_series.sum())`` → TypeError) and used
to mix filtered / unfiltered event populations. Both directions now use
the same globally qualified whole-event populations.
"""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
    compute_swr_nl_event_counts,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["Non-Local_Continuous_prob"] = df["non_local_event"].gt(0).map(
        {True: 0.9, False: 0.1}
    )
    df["Non-Local_Fragmented_prob"] = 0.0
    return df.set_index("time").rename_axis(None)


class TestEmptyA:
    def test_all_high_speed_a_does_not_crash(self):
        # Two non-local events, both at high speed → immobility
        # filter drops them all. Used to raise TypeError.
        df = _frame(
            [
                {
                    "time": 0.0, "animal": "m", "day": 1, "epoch": "e",
                    "head_speed": 50.0, "non_local_event": 1, "SWR_event": 1,
                },
                {
                    "time": 0.5, "animal": "m", "day": 1, "epoch": "e",
                    "head_speed": 50.0, "non_local_event": 1, "SWR_event": 1,
                },
                {
                    "time": 1.0, "animal": "m", "day": 1, "epoch": "e",
                    "head_speed": 50.0, "non_local_event": 2, "SWR_event": 2,
                },
            ]
        )
        out = compute_swr_nl_event_counts(df)
        assert out["n_a"] == 0
        assert out["n_a_with_b"] == 0


class TestCrossSessionTimeCollision:
    def test_shared_timestamp_across_sessions_does_not_leak_overlap(self):
        # Two sessions, both starting at t=0.0:
        #  - Session 1 has a low-speed NL+SWR overlap at t=0.0 (kept).
        #  - Session 2 has a HIGH-SPEED NL event at t=0.0 (excluded
        #    by the immobility filter) plus an SWR at t=0.0.
        # The bare ``index.isin`` form would falsely count session
        # 2's SWR as overlapping the surviving A-set because the
        # time index 0.0 also belongs to session 1's surviving row.
        df = _frame(
            [
                # Session 1 — low-speed, retained.
                {
                    "time": 0.0, "animal": "s1", "day": 1, "epoch": "e",
                    "head_speed": 0.0, "non_local_event": 1, "SWR_event": 1,
                },
                {
                    "time": 0.5, "animal": "s1", "day": 1, "epoch": "e",
                    "head_speed": 0.0, "non_local_event": 1, "SWR_event": 1,
                },
                # Session 2 — high-speed, excluded from both populations.
                {
                    "time": 0.0, "animal": "s2", "day": 1, "epoch": "e",
                    "head_speed": 50.0, "non_local_event": 1, "SWR_event": 1,
                },
                {
                    "time": 0.5, "animal": "s2", "day": 1, "epoch": "e",
                    "head_speed": 50.0, "non_local_event": 1, "SWR_event": 1,
                },
            ]
        )
        out = compute_swr_nl_event_counts(df)
        # A is just session 1's NL event.
        assert out["n_a"] == 1
        # SWRs now share the same whole-event speed cohort. Session 2's
        # high-speed SWR is excluded rather than retained as a raw B event.
        assert out["n_b"] == 1
        assert out["n_b_with_a"] == 1, (
            f"n_b_with_a={out['n_b_with_a']} — bare time-index "
            "matching leaked session 2's high-speed overlap into "
            "session 1's overlap count."
        )


class TestFilterConsistency:
    def test_high_speed_overlap_does_not_inflate_n_b_with_a(self):
        # Two non-local events: the first overlaps an SWR but is
        # high-speed (excluded by immobility filter); the second is
        # low-speed and does not overlap any SWR.
        df = _frame(
            [
                # NL=1 + SWR=1, but moving — should be excluded from A.
                {
                    "time": 0.0, "animal": "m", "day": 1, "epoch": "e",
                    "head_speed": 50.0, "non_local_event": 1, "SWR_event": 1,
                },
                {
                    "time": 0.5, "animal": "m", "day": 1, "epoch": "e",
                    "head_speed": 50.0, "non_local_event": 1, "SWR_event": 1,
                },
                # NL=2 with no SWR, low-speed — should be the only A.
                {
                    "time": 5.0, "animal": "m", "day": 1, "epoch": "e",
                    "head_speed": 0.0, "non_local_event": 2, "SWR_event": 0,
                },
            ]
        )
        out = compute_swr_nl_event_counts(df)
        # A is the immobility-filtered NL set: NL=2 only.
        assert out["n_a"] == 1
        assert out["n_a_with_b"] == 0
        # The co-located SWR is also a high-speed event and is excluded by the
        # same global criterion; it cannot remain as a secondary detector.
        assert out["n_b"] == 0
        # The SWR overlapped only the *excluded* high-speed NL event,
        # so it should NOT count as overlapping the A-set.
        assert out["n_b_with_a"] == 0, (
            f"n_b_with_a={out['n_b_with_a']} — high-speed NL events "
            "must not contribute to the SWR↔NL overlap count when "
            "non-local is immobility-filtered out of A."
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
                {"time": 0.0, "animal": "a", "day": 1, "epoch": "e1",
                 "head_speed": 0.0, "non_local_event": 1, "SWR_event": 0},
                {"time": 0.5, "animal": "a", "day": 1, "epoch": "e1",
                 "head_speed": 0.0, "non_local_event": 1, "SWR_event": 0},
                {"time": 1.0, "animal": "a", "day": 1, "epoch": "e1",
                 "head_speed": 0.0, "non_local_event": 1, "SWR_event": 2},
                {"time": 1.5, "animal": "a", "day": 1, "epoch": "e1",
                 "head_speed": 0.0, "non_local_event": 0, "SWR_event": 2},
                {"time": 2.0, "animal": "a", "day": 1, "epoch": "e1",
                 "head_speed": 0.0, "non_local_event": 0, "SWR_event": 2},
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
                {"time": 0.0, "animal": "a", "day": 1, "epoch": "e1",
                 "head_speed": 0.0, "non_local_event": 1, "SWR_event": 0},
                {"time": 0.5, "animal": "a", "day": 1, "epoch": "e1",
                 "head_speed": 0.0, "non_local_event": 1, "SWR_event": 0},
                {"time": 1.0, "animal": "a", "day": 1, "epoch": "e1",
                 "head_speed": 0.0, "non_local_event": 1, "SWR_event": 2},
                # NL extends to t=1.5 (overlap of [1.0, 1.5] with SWR).
                {"time": 1.5, "animal": "a", "day": 1, "epoch": "e1",
                 "head_speed": 0.0, "non_local_event": 1, "SWR_event": 2},
                {"time": 2.0, "animal": "a", "day": 1, "epoch": "e1",
                 "head_speed": 0.0, "non_local_event": 0, "SWR_event": 2},
            ]
        )
        out = compute_swr_nl_event_counts(df)
        assert out["n_a_with_b"] == 1
        assert out["n_b_with_a"] == 1
