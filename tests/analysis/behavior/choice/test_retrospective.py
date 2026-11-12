"""Tests for analysis/behavior/choice/retrospective.py.

The retrospective metrics ask (post hoc) whether replay content during a stop aligns
with the arm the animal will visit next. The metrics are:

- ``chosen_arm_fraction``: per-event ``arm_fraction`` at ``arm == next_arm``.
  Real distribution is compared against a within-session
  ``next_arm_circular_shift`` null.
- ``ends_at_next``: does the last in-event MAP arm equal ``next_arm``?
- ``contains_path_to_next``: does the event's MAP-arm trace cover the
  geometric path from ``current_arm`` to ``next_arm``?

The bandit-track topology is:
- 6 wells (arms 0–5), grouped into 3 patches: {0,1}=1, {2,3}=2, {4,5}=3.
- 3 trunk arms (6,7,8), one per patch, connecting the patch junction to
  the center node.

Geometric path semantics:
- same arm → {arm}
- same patch (no trunk traversal needed) → {current, next}
- different patch → {current, next, trunk(current_patch), trunk(next_patch)}
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.choice.retrospective import (
    arm_path_well_to_well,
    at_well_event_cohort,
    chosen_arm_fraction,
    chosen_arm_fraction_prev,
    chosen_patch_fraction,
    compute_retrospective_metrics,
    contains_path_from_prev,
    contains_path_to_next,
    ends_at_next,
    ends_at_next_patch,
    ends_at_prev,
    family_unit_counts,
    freeze_retrospective_cohort,
    retrospective_alignment_by_session,
    retrospective_alignment_with_shuffle,
    retrospective_status,
    select_event_subset,
)
from continuum_swr_replay.analysis.behavior.choice.shuffles import (
    next_arm_circular_shift,
)
from continuum_swr_replay.analysis.events.cohort_selection import select_event_cohort
from continuum_swr_replay.data_loaders.constants import WELL_PATCH_MAPPING

# ─────────────────────────────────────────────────────────────────────────────
# arm_path_well_to_well
# ─────────────────────────────────────────────────────────────────────────────


class TestArmPathWellToWell:
    def test_same_arm_returns_singleton(self):
        assert arm_path_well_to_well(0, 0) == {0}
        assert arm_path_well_to_well(3, 3) == {3}

    def test_same_patch_no_trunk(self):
        # Wells 0 and 1 share patch 1; path crosses no trunk.
        assert arm_path_well_to_well(0, 1) == {0, 1}
        # Wells 2, 3 share patch 2.
        assert arm_path_well_to_well(2, 3) == {2, 3}
        # Wells 4, 5 share patch 3.
        assert arm_path_well_to_well(4, 5) == {4, 5}

    def test_cross_patch_includes_both_trunks(self):
        # 0 (patch 1) → 2 (patch 2) traverses trunks 6 and 7.
        assert arm_path_well_to_well(0, 2) == {0, 2, 6, 7}
        # 1 (patch 1) → 4 (patch 3) traverses trunks 6 and 8.
        assert arm_path_well_to_well(1, 4) == {1, 4, 6, 8}

    def test_symmetric(self):
        # Path is undirected.
        assert arm_path_well_to_well(0, 5) == arm_path_well_to_well(5, 0)
        assert arm_path_well_to_well(2, 4) == arm_path_well_to_well(4, 2)

    def test_invalid_well_raises(self):
        # Trunk arms (6,7,8) and out-of-range arms are not valid wells.
        # Match the specific branch so a regression that swaps the two
        # validators (current_arm vs next_arm) is caught.
        with pytest.raises(ValueError, match="current_arm=6"):
            arm_path_well_to_well(6, 0)
        with pytest.raises(ValueError, match="next_arm=9"):
            arm_path_well_to_well(0, 9)


# ─────────────────────────────────────────────────────────────────────────────
# Shared minimal-tables fixture
# ─────────────────────────────────────────────────────────────────────────────


def _make_event_table(events: list[dict]) -> pd.DataFrame:
    """Build a minimal event_table with the columns the metrics need."""
    df = pd.DataFrame(events)
    # Fill columns retrospective.py expects but tests don't always set
    for col, default in {
        "animal": "test",
        "session_date": 1,
        "epoch": "01_r1",
        "event_midpoint_s": np.nan,
        "at_well": True,
        "n_nonlocal_bins": 4,
    }.items():
        if col not in df.columns:
            df[col] = default
    if "event_midpoint_s" in df.columns and df["event_midpoint_s"].isna().all():
        df["event_midpoint_s"] = np.arange(len(df), dtype=float)
    if "occupancy_state" not in df.columns:
        # Canonical tables carry both labels and ``table_contracts`` refuses a
        # table where they disagree, so derive one from the other rather than
        # letting a fixture state a combination production can never produce.
        is_at_well = (
            df["at_well"].eq(True).astype("boolean").fillna(False).to_numpy(dtype=bool)
        )
        df["occupancy_state"] = np.where(
            df["at_well"].isna().to_numpy(),
            "unknown",
            np.where(is_at_well, "at_well", "in_transit"),
        )
    if "stop_id" not in df.columns:
        df["stop_id"] = df["event_id"]
    if "completed_trial_id" not in df.columns:
        # One completed trip per event unless a test models a shared visit.
        df["completed_trial_id"] = np.arange(len(df), dtype=float)
    if "current_patch" not in df.columns:
        df["current_patch"] = pd.to_numeric(
            df["occupancy_well_arm"], errors="coerce"
        ).map(WELL_PATCH_MAPPING)
    return df


def _make_event_arm_table(
    per_event_arm_fractions: dict[int, list[float]],
) -> pd.DataFrame:
    """Build an event_arm_table from ``{event_id: [arm0_frac, ..., arm8_frac]}``."""
    rows = []
    for ev_id, fracs in per_event_arm_fractions.items():
        assert len(fracs) == 9, "expect 9 arms"
        for arm, f in enumerate(fracs):
            rows.append(
                {
                    "animal": "test",
                    "session_date": 1,
                    "epoch": "01_r1",
                    "event_id": ev_id,
                    "candidate_arm": arm,
                    "arm_fraction": f,
                }
            )
    return pd.DataFrame(rows)


def _make_event_sequence_table(per_event_arm_seq: dict[int, list[int]]) -> pd.DataFrame:
    """Build an event_sequence_table from ``{event_id: [arm_t0, arm_t1, ...]}``."""
    rows = []
    for ev_id, seq in per_event_arm_seq.items():
        for t, arm in enumerate(seq):
            rows.append(
                {
                    "animal": "test",
                    "session_date": 1,
                    "epoch": "01_r1",
                    "event_id": ev_id,
                    "t_in_event": t,
                    "decoded_arm_map": int(arm),
                    "nonlocal_prob": 1.0,
                }
            )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# chosen_arm_fraction
# ─────────────────────────────────────────────────────────────────────────────


class TestChosenArmFraction:
    def test_picks_arm_fraction_at_next_arm(self):
        ev = _make_event_table(
            [
                {"event_id": 1, "occupancy_well_arm": 0, "next_arm": 3},
                {"event_id": 2, "occupancy_well_arm": 3, "next_arm": 0},
            ]
        )
        ea = _make_event_arm_table(
            {
                # arm:                       0     1     2     3     4    5    6    7    8
                1: [0.05, 0.10, 0.15, 0.40, 0.05, 0.05, 0.05, 0.05, 0.10],
                2: [0.30, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.05, 0.10],
            }
        )
        out = chosen_arm_fraction(ev, ea)
        out = out.set_index("event_id")
        assert out.loc[1, "chosen_arm_fraction"] == pytest.approx(0.40)
        assert out.loc[2, "chosen_arm_fraction"] == pytest.approx(0.30)

    def test_drops_events_with_nan_next_arm(self):
        ev = _make_event_table(
            [
                {"event_id": 1, "occupancy_well_arm": 0, "next_arm": np.nan},
                {"event_id": 2, "occupancy_well_arm": 3, "next_arm": 0},
            ]
        )
        ea = _make_event_arm_table(
            {
                1: [0.10] * 9,
                2: [0.30, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.05, 0.10],
            }
        )
        out = chosen_arm_fraction(ev, ea)
        # Event 1 should be dropped because next_arm is missing.
        assert set(out["event_id"]) == {2}


# ─────────────────────────────────────────────────────────────────────────────
# ends_at_next
# ─────────────────────────────────────────────────────────────────────────────


class TestEndsAtNext:
    def test_last_bin_matches_next_arm(self):
        ev = _make_event_table(
            [
                {"event_id": 1, "occupancy_well_arm": 0, "next_arm": 3},
            ]
        )
        # last in-event bin is arm 3 → True
        es = _make_event_sequence_table({1: [0, 6, 7, 3]})
        out = ends_at_next(ev, es).set_index("event_id")
        assert bool(out.loc[1, "ends_at_next"]) is True

    def test_last_bin_mismatches_next_arm(self):
        ev = _make_event_table(
            [
                {"event_id": 1, "occupancy_well_arm": 0, "next_arm": 3},
            ]
        )
        # last in-event bin is arm 0 → False
        es = _make_event_sequence_table({1: [3, 6, 7, 0]})
        out = ends_at_next(ev, es).set_index("event_id")
        assert bool(out.loc[1, "ends_at_next"]) is False

    def test_singleton_event(self):
        ev = _make_event_table(
            [{"event_id": 1, "occupancy_well_arm": 0, "next_arm": 0}]
        )
        es = _make_event_sequence_table({1: [0]})
        out = ends_at_next(ev, es).set_index("event_id")
        assert bool(out.loc[1, "ends_at_next"]) is True

    def test_event_with_no_sequence_rows_recorded_as_false(self):
        """Regression (symmetric to ``ends_at_next_patch``): an event
        with valid ``next_arm`` but no decoded sequence row produces
        a NaN ``_last_arm`` after the left merge. Pre-fix the
        nullable ``Int64`` equality returned ``pd.NA``, which then
        survived as NaN through ``_per_session_metric_means``'s
        ``astype(float)`` → ``np.nanmean`` reduction and got
        *dropped* from the denominator — a session with one True
        endpoint and one missing sequence reported a mean of 1.0
        instead of the documented 0.5.
        """
        ev = _make_event_table(
            [
                {"event_id": 1, "occupancy_well_arm": 0, "next_arm": 3},
                {"event_id": 2, "occupancy_well_arm": 0, "next_arm": 3},
            ]
        )
        # Only event 1 has a sequence; event 2's _last_arm is NaN
        # after the left merge.
        es = _make_event_sequence_table({1: [0, 6, 7, 3]})
        out = ends_at_next(ev, es).set_index("event_id")
        assert bool(out.loc[1, "ends_at_next"]) is True
        assert bool(out.loc[2, "ends_at_next"]) is False, (
            f"event-with-no-sequence produced "
            f"{out.loc[2, 'ends_at_next']!r} instead of False"
        )
        # Plain ``bool`` dtype so downstream ``astype(float)`` →
        # ``np.nanmean`` never sees NaN.
        assert out["ends_at_next"].dtype == bool

    def test_missing_sequence_endpoint_does_not_inflate_session_mean(self):
        """End-to-end regression: 2-event table with one True and one
        missing-sequence endpoint must average to 0.5 (not 1.0)."""
        from continuum_swr_replay.analysis.behavior.choice.retrospective import (
            _per_session_metric_means,
        )

        ev = _make_event_table(
            [
                {"event_id": 1, "occupancy_well_arm": 0, "next_arm": 3},
                {"event_id": 2, "occupancy_well_arm": 0, "next_arm": 3},
            ]
        )
        es = _make_event_sequence_table({1: [0, 6, 7, 3]})
        metrics = ends_at_next(ev, es)
        means = _per_session_metric_means(metrics)
        assert means["ends_at_next"] == 0.5, (
            f"session mean = {means['ends_at_next']} — missing-sequence "
            "endpoint inflated the mean by being dropped from the "
            "np.nanmean denominator instead of being counted as False."
        )


# ─────────────────────────────────────────────────────────────────────────────
# contains_path_to_next
# ─────────────────────────────────────────────────────────────────────────────


class TestContainsPathToNext:
    def test_full_path_present(self):
        ev = _make_event_table(
            [
                {"event_id": 1, "occupancy_well_arm": 0, "next_arm": 2},
            ]
        )
        # 0→2 path = {0,2,6,7}; sequence covers all four arms (in any order).
        es = _make_event_sequence_table({1: [0, 6, 7, 2, 7]})
        out = contains_path_to_next(ev, es).set_index("event_id")
        assert bool(out.loc[1, "contains_path_to_next"]) is True

    def test_missing_one_path_arm(self):
        ev = _make_event_table(
            [
                {"event_id": 1, "occupancy_well_arm": 0, "next_arm": 2},
            ]
        )
        # Missing trunk 7 → False.
        es = _make_event_sequence_table({1: [0, 6, 2]})
        out = contains_path_to_next(ev, es).set_index("event_id")
        assert bool(out.loc[1, "contains_path_to_next"]) is False

    def test_set_membership_does_not_require_order_or_continuity(self):
        """Coverage is deliberately a set metric, not a trajectory metric."""
        ev = _make_event_table(
            [{"event_id": 1, "occupancy_well_arm": 0, "next_arm": 2}]
        )
        # Required set is {0, 2, 6, 7}.  It appears out of route order,
        # interleaved with unrelated arms, and with duplicates.  All of those
        # details are intentionally irrelevant to path-arm set coverage.
        es = _make_event_sequence_table({1: [7, 4, 0, 7, 8, 2, 6, 4]})

        out = contains_path_to_next(ev, es).set_index("event_id")

        assert bool(out.loc[1, "contains_path_to_next"]) is True

    def test_same_patch_path_short(self):
        ev = _make_event_table(
            [
                {"event_id": 1, "occupancy_well_arm": 0, "next_arm": 1},
            ]
        )
        # 0→1 path = {0,1}; sequence with both arms suffices.
        es = _make_event_sequence_table({1: [0, 6, 1]})
        out = contains_path_to_next(ev, es).set_index("event_id")
        assert bool(out.loc[1, "contains_path_to_next"]) is True

    def test_same_arm_requires_decoded_visit(self):
        # current_arm == next_arm == 3 → path = {3}. The MAP-arm trace
        # must include arm 3 at least once for the metric to be True.
        ev = _make_event_table(
            [
                {"event_id": 1, "occupancy_well_arm": 3, "next_arm": 3},
                {"event_id": 2, "occupancy_well_arm": 3, "next_arm": 3},
            ]
        )
        es = _make_event_sequence_table(
            {
                1: [3, 7, 9],  # visits arm 3 → True
                2: [7, 9, 0],  # never visits arm 3 → False
            }
        )
        out = contains_path_to_next(ev, es).set_index("event_id")
        assert bool(out.loc[1, "contains_path_to_next"]) is True
        assert bool(out.loc[2, "contains_path_to_next"]) is False


# ─────────────────────────────────────────────────────────────────────────────
# chosen_patch_fraction (sum of arm_fraction over arms in next_patch)
# ─────────────────────────────────────────────────────────────────────────────


class TestChosenPatchFraction:
    def test_sums_arms_in_target_patch(self):
        # next_patch == 2 → arms {2, 3, 7}; sum their arm_fractions.
        ev = _make_event_table(
            [
                {
                    "event_id": 1,
                    "occupancy_well_arm": 0,
                    "next_arm": 3,
                    "next_patch": 2,
                },
            ]
        )
        ea = _make_event_arm_table(
            {
                # arm:                       0     1     2     3     4    5    6    7    8
                1: [0.05, 0.05, 0.20, 0.30, 0.05, 0.05, 0.10, 0.10, 0.10],
            }
        )
        out = chosen_patch_fraction(ev, ea).set_index("event_id")
        # Patch 2 = arms {2, 3, 7}: 0.20 + 0.30 + 0.10 = 0.60
        assert out.loc[1, "chosen_patch_fraction"] == pytest.approx(0.60)

    def test_drops_events_with_nan_next_patch(self):
        ev = _make_event_table(
            [
                {
                    "event_id": 1,
                    "occupancy_well_arm": 0,
                    "next_arm": 0,
                    "next_patch": np.nan,
                },
                {
                    "event_id": 2,
                    "occupancy_well_arm": 3,
                    "next_arm": 3,
                    "next_patch": 2,
                },
            ]
        )
        ea = _make_event_arm_table(
            {
                1: [0.10] * 9,
                2: [0.05, 0.05, 0.20, 0.30, 0.05, 0.05, 0.10, 0.10, 0.10],
            }
        )
        out = chosen_patch_fraction(ev, ea)
        assert set(out["event_id"]) == {2}


# ─────────────────────────────────────────────────────────────────────────────
# ends_at_next_patch
# ─────────────────────────────────────────────────────────────────────────────


class TestEndsAtNextPatch:
    def test_last_bin_in_target_patch(self):
        # next_patch=2 (arms {2,3,7}); last bin arm=3 → ends in patch 2 → True
        ev = _make_event_table(
            [
                {
                    "event_id": 1,
                    "occupancy_well_arm": 0,
                    "next_arm": 3,
                    "next_patch": 2,
                },
            ]
        )
        es = _make_event_sequence_table({1: [0, 6, 7, 3]})
        out = ends_at_next_patch(ev, es).set_index("event_id")
        assert bool(out.loc[1, "ends_at_next_patch"]) is True

    def test_last_bin_in_different_patch(self):
        # next_patch=2; last bin arm=4 (patch 3) → False
        ev = _make_event_table(
            [
                {
                    "event_id": 1,
                    "occupancy_well_arm": 0,
                    "next_arm": 3,
                    "next_patch": 2,
                },
            ]
        )
        es = _make_event_sequence_table({1: [0, 6, 7, 4]})
        out = ends_at_next_patch(ev, es).set_index("event_id")
        assert bool(out.loc[1, "ends_at_next_patch"]) is False

    def test_last_bin_on_trunk_arm_of_target_patch(self):
        # last bin arm=7 (trunk for patch 2); ends_at_next_patch=2 → True
        ev = _make_event_table(
            [
                {
                    "event_id": 1,
                    "occupancy_well_arm": 0,
                    "next_arm": 3,
                    "next_patch": 2,
                },
            ]
        )
        es = _make_event_sequence_table({1: [0, 6, 9, 7]})
        out = ends_at_next_patch(ev, es).set_index("event_id")
        assert bool(out.loc[1, "ends_at_next_patch"]) is True

    def test_unmapped_last_arm_recorded_as_false(self):
        """Regression: docstring says "Bins whose MAP arm has no
        patch label (NaN, junction-bin artifact) yield NaN and are
        excluded from the equality test (the row is recorded as
        False)." Pre-fix the equality on nullable ``Int64`` returned
        ``pd.NA``, which then survived as NaN through
        ``_per_session_metric_means``'s ``astype(float)`` →
        ``np.nanmean`` reduction and got *dropped* from the
        denominator — so a session with a real True endpoint and one
        unmapped/junction endpoint reported a mean of 1.0 instead of
        the documented 0.5.
        """
        ev = _make_event_table(
            [
                {
                    "event_id": 1,
                    "occupancy_well_arm": 0,
                    "next_arm": 3,
                    "next_patch": 2,
                },
                {
                    "event_id": 2,
                    "occupancy_well_arm": 0,
                    "next_arm": 3,
                    "next_patch": 2,
                },
            ]
        )
        es = _make_event_sequence_table(
            {
                1: [0, 6, 7, 3],  # ends on arm 3 (patch 2) → True
                # Arm 9 is not a key in TRACK_SEGMENT_TO_PATCH, so
                # _last_patch is NaN. Pre-fix: ends_at_next_patch=<NA>;
                # post-fix: False.
                2: [0, 6, 7, 9],
            }
        )
        out = ends_at_next_patch(ev, es).set_index("event_id")
        assert bool(out.loc[1, "ends_at_next_patch"]) is True
        assert bool(out.loc[2, "ends_at_next_patch"]) is False, (
            f"unmapped endpoint produced "
            f"{out.loc[2, 'ends_at_next_patch']!r} instead of False"
        )
        # And the column dtype must be plain ``bool`` (not nullable
        # ``boolean``) so the downstream ``astype(float)`` →
        # ``np.nanmean`` reduction never sees NaN. Otherwise the
        # session-mean would silently drop these rows.
        assert out["ends_at_next_patch"].dtype == bool

    def test_unmapped_endpoint_does_not_inflate_session_mean(self):
        """End-to-end regression: feed a 2-event table with one True
        endpoint and one unmapped endpoint through
        ``_per_session_metric_means`` and assert the reported mean is
        0.5 (the documented value), not 1.0 (pre-fix)."""
        from continuum_swr_replay.analysis.behavior.choice.retrospective import (
            _per_session_metric_means,
        )

        ev = _make_event_table(
            [
                {
                    "event_id": 1,
                    "occupancy_well_arm": 0,
                    "next_arm": 3,
                    "next_patch": 2,
                },
                {
                    "event_id": 2,
                    "occupancy_well_arm": 0,
                    "next_arm": 3,
                    "next_patch": 2,
                },
            ]
        )
        es = _make_event_sequence_table({1: [0, 6, 7, 3], 2: [0, 6, 7, 9]})
        metrics = ends_at_next_patch(ev, es)
        means = _per_session_metric_means(metrics)
        assert means["ends_at_next_patch"] == 0.5, (
            f"session mean = {means['ends_at_next_patch']} — unmapped "
            "endpoint inflated the mean by being dropped from the "
            "np.nanmean denominator instead of being counted as False."
        )


# ─────────────────────────────────────────────────────────────────────────────
# compute_retrospective_metrics
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeRetrospectiveMetrics:
    def test_joins_three_metrics(self):
        ev = _make_event_table(
            [
                {"event_id": 1, "occupancy_well_arm": 0, "next_arm": 3},
                {"event_id": 2, "occupancy_well_arm": 3, "next_arm": 4},
            ]
        )
        ea = _make_event_arm_table(
            {
                1: [0.05, 0.05, 0.10, 0.50, 0.05, 0.05, 0.10, 0.05, 0.05],
                2: [0.10, 0.05, 0.10, 0.10, 0.40, 0.05, 0.05, 0.10, 0.05],
            }
        )
        es = _make_event_sequence_table(
            {
                1: [0, 6, 9, 7, 3],  # ends at 3 (chosen) ✓; path 0→3 = {0,3,6,7}
                2: [3, 7, 9, 8, 4],  # ends at 4 ✓; path 3→4 = {3,4,7,8}
            }
        )
        out = compute_retrospective_metrics(ev, ea, es).set_index("event_id")
        assert out.loc[1, "chosen_arm_fraction"] == pytest.approx(0.50)
        assert out.loc[2, "chosen_arm_fraction"] == pytest.approx(0.40)
        assert bool(out.loc[1, "ends_at_next"]) is True
        assert bool(out.loc[1, "contains_path_to_next"]) is True
        assert bool(out.loc[2, "ends_at_next"]) is True
        assert bool(out.loc[2, "contains_path_to_next"]) is True

    def test_empty_schema_complete_inputs_return_empty_metrics(self):
        ev = pd.DataFrame(
            columns=[
                "animal",
                "session_date",
                "epoch",
                "event_id",
                "occupancy_well_arm",
                "next_arm",
                "next_patch",
            ]
        )
        ea = pd.DataFrame(
            columns=[
                "animal",
                "session_date",
                "epoch",
                "event_id",
                "candidate_arm",
                "arm_fraction",
            ]
        )
        es = pd.DataFrame(
            columns=[
                "animal",
                "session_date",
                "epoch",
                "event_id",
                "t_in_event",
                "decoded_arm_map",
            ]
        )

        out = compute_retrospective_metrics(ev, ea, es)
        assert out.empty
        assert {"animal", "session_date", "epoch", "event_id"}.issubset(out.columns)


# ─────────────────────────────────────────────────────────────────────────────
# retrospective_alignment_with_shuffle
# ─────────────────────────────────────────────────────────────────────────────


class TestRetrospectiveAlignmentWithShuffle:
    def test_cohort_matches_the_shared_event_grain_selector(self):
        """The retrospective cohort is the shared rule plus an endpoint check.

        A ``mixed`` or ``unknown`` event is not at one well for its whole
        window, so neither the shared selector nor this one may keep it.  The
        two must therefore name the same events; any drift would silently give
        Stage 5B a different population from the content ladders.
        """
        ev = _make_event_table(
            [
                {
                    "event_id": 1,
                    "occupancy_well_arm": 0,
                    "next_arm": 1,
                    "next_patch": 1,
                    "at_well": True,
                    "occupancy_state": "at_well",
                },
                {
                    "event_id": 2,
                    "occupancy_well_arm": 3,
                    "next_arm": 0,
                    "next_patch": 1,
                    "at_well": False,
                    "occupancy_state": "mixed",
                },
                {
                    "event_id": 3,
                    "occupancy_well_arm": np.nan,
                    "next_arm": 2,
                    "next_patch": 2,
                    "at_well": pd.NA,
                    "occupancy_state": "unknown",
                },
                {
                    "event_id": 4,
                    "occupancy_well_arm": 2,
                    "next_arm": 4,
                    "next_patch": 3,
                    "at_well": True,
                    "occupancy_state": "at_well",
                },
            ]
        )
        ea = _make_event_arm_table(
            {
                1: [1.0] + [0.0] * 8,
                2: [0.0, 1.0] + [0.0] * 7,
                3: [0.0, 0.0, 1.0] + [0.0] * 6,
                4: [0.0, 0.0, 0.0, 1.0] + [0.0] * 5,
            }
        )
        es = _make_event_sequence_table({1: [0, 1], 2: [3], 3: [2], 4: [2, 4]})

        ev_at, ea_at, es_at = at_well_event_cohort(ev, ea, es)
        shared, report = select_event_cohort(ev, cohort="at_well")

        assert set(ev_at["event_id"]) == set(shared["event_id"]) == {1, 4}
        assert report["n_events_after"] == 2
        assert set(ea_at["event_id"]) == {1, 4}
        assert set(es_at["event_id"]) == {1, 4}

    def test_rejects_legacy_location_schema(self):
        ev = _make_event_table(
            [{"event_id": 1, "occupancy_well_arm": 0, "next_arm": 1}]
        ).drop(columns="at_well")
        ea = _make_event_arm_table({1: [1.0] + [0.0] * 8})
        es = _make_event_sequence_table({1: [0, 1]})
        with pytest.raises(KeyError, match="regenerate event tables"):
            at_well_event_cohort(ev, ea, es)

    def test_rejects_singleton_after_explicit_at_well_cohort(self):
        ev = _make_event_table(
            [
                {
                    "event_id": 1,
                    "occupancy_well_arm": 0,
                    "next_arm": 1,
                    "next_patch": 1,
                    "at_well": True,
                },
                {
                    "event_id": 2,
                    "occupancy_well_arm": np.nan,
                    "next_arm": 2,
                    "next_patch": 2,
                    "at_well": False,
                },
            ]
        )
        ea = _make_event_arm_table(
            {1: [0.1, 0.9, 0, 0, 0, 0, 0, 0, 0], 2: [0, 0, 1, 0, 0, 0, 0, 0, 0]}
        )
        es = _make_event_sequence_table({1: [0, 1], 2: [2]})

        ev_at, ea_at, es_at = at_well_event_cohort(ev, ea, es)
        assert ev_at["event_id"].tolist() == [1]
        assert set(ea_at["event_id"]) == {1}
        assert set(es_at["event_id"]) == {1}

        result = retrospective_alignment_with_shuffle(
            ev,
            ea,
            es,
            n_shuffles=1,
            rng=np.random.default_rng(0),
        )
        assert result["n_events_arm_family"] == 0
        assert result["n_events_patch_family"] == 0
        assert result["n_events_path_family"] == 0
        assert result["n_shuffle_units_excluded_singleton_arm_family"] == 1
        assert retrospective_status(result) == "no_movable_family"

    def test_missing_at_well_is_excluded_like_an_explicit_false(self):
        """``pd.NA`` must not survive the cohort filter.

        Before the first completed trip ``at_well`` is NA, where
        ``.eq(True)`` yields NA rather than False; without the explicit
        fill those location-unknown events would enter the cohort.
        """
        ev = _make_event_table(
            [
                {
                    "event_id": 1,
                    "occupancy_well_arm": 0,
                    "next_arm": 1,
                    "next_patch": 1,
                    "at_well": True,
                },
                {
                    "event_id": 2,
                    "occupancy_well_arm": np.nan,
                    "next_arm": 2,
                    "next_patch": 2,
                    "at_well": pd.NA,
                },
            ]
        )
        ea = _make_event_arm_table(
            {1: [0.1, 0.9, 0, 0, 0, 0, 0, 0, 0], 2: [0, 0, 1, 0, 0, 0, 0, 0, 0]}
        )
        es = _make_event_sequence_table({1: [0, 1], 2: [2]})

        ev_at, ea_at, es_at = at_well_event_cohort(ev, ea, es)

        assert ev_at["event_id"].tolist() == [1]
        assert set(ea_at["event_id"]) == {1}
        assert set(es_at["event_id"]) == {1}

    def test_missing_final_arm_is_endpoint_failure_not_cohort_exclusion(self):
        """A nonfinite final MAP bin stays in the endpoint denominator.

        The earlier valid bin makes both events observable for the path metric,
        while event 2's final junction bin is deliberately nonfinite.  The
        shared default cohort must retain both events: event 2 is a False
        endpoint, not an excluded endpoint.
        """
        ev = _make_event_table(
            [
                {
                    "event_id": 1,
                    "occupancy_well_arm": 0,
                    "next_arm": 2,
                    "next_patch": 2,
                },
                {
                    "event_id": 2,
                    "occupancy_well_arm": 0,
                    "next_arm": 4,
                    "next_patch": 3,
                },
            ]
        )
        ea = _make_event_arm_table(
            {
                1: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                2: [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        es = _make_event_sequence_table({1: [0, 6, 7, 2], 2: [0, 6, 8, 4, 1]})
        es.loc[
            es["event_id"].eq(2) & es["t_in_event"].eq(4),
            "decoded_arm_map",
        ] = np.nan

        result = retrospective_alignment_with_shuffle(
            ev,
            ea,
            es,
            n_shuffles=2,
            rng=np.random.default_rng(0),
        )

        assert "n_events" not in result
        assert result["n_events_arm_family"] == 2
        assert result["n_events_patch_family"] == 2
        assert result["n_events_path_family"] == 2
        assert result["n_events_ends_at_next"] == 2
        assert result["n_events_ends_at_next_patch"] == 2
        assert result["real_mean_ends_at_next"] == 0.5
        assert result["real_mean_ends_at_next_patch"] == 0.5

    def test_endpoint_only_counts_event_with_no_sequence_rows_as_failure(self):
        """Endpoint-only inference does not require a decoded sequence row."""
        ev = _make_event_table(
            [
                {"event_id": 1, "occupancy_well_arm": 0, "next_arm": 0},
                {"event_id": 2, "occupancy_well_arm": 0, "next_arm": 1},
            ]
        )
        ea = _make_event_arm_table(
            {
                1: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                2: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        es = _make_event_sequence_table({1: [0]})

        result = retrospective_alignment_with_shuffle(
            ev,
            ea,
            es,
            n_shuffles=2,
            metrics=("ends_at_next",),
            rng=np.random.default_rng(0),
        )

        assert result["n_events"] == 2
        assert result["n_events_ends_at_next"] == 2
        assert result["real_mean_ends_at_next"] == 0.5

    def test_rejects_missing_patch_target_before_inference(self):
        """A missing target cannot silently create a smaller denominator."""
        ev = _make_event_table(
            [
                {
                    "event_id": 1,
                    "occupancy_well_arm": 0,
                    "next_arm": 0,
                    "next_patch": 1,
                },
                {
                    "event_id": 2,
                    "occupancy_well_arm": 0,
                    "next_arm": 0,
                    "next_patch": np.nan,
                },
            ]
        )
        ea = _make_event_arm_table(
            {
                1: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                2: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        es = _make_event_sequence_table({1: [0], 2: [0]})

        with pytest.raises(ValueError, match="1 invalid next_patch"):
            retrospective_alignment_with_shuffle(
                ev,
                ea,
                es,
                n_shuffles=0,
                metrics=("chosen_patch_fraction",),
                rng=np.random.default_rng(0),
            )

    def test_metrics_selection_does_not_touch_unrequested_sequence_path(self):
        """Arm-only inference works with no sequence-table schema at all."""
        events = [
            {"event_id": event_id, "occupancy_well_arm": 0, "next_arm": next_arm}
            for event_id, next_arm in enumerate((1, 2, 3, 4))
        ]
        ev = _make_event_table(events)
        ea = _make_event_arm_table(
            {
                row["event_id"]: [float(arm == row["next_arm"]) for arm in range(9)]
                for row in events
            }
        )

        result = retrospective_alignment_with_shuffle(
            ev,
            ea,
            pd.DataFrame(),
            n_shuffles=2,
            metrics=("chosen_arm_fraction",),
            rng=np.random.default_rng(0),
        )

        assert result["n_events"] == 4
        assert result["n_events_chosen_arm_fraction"] == 4
        assert "real_mean_ends_at_next" not in result

    def test_current_patch_content_is_not_a_false_next_patch_null(self):
        """Deterministic within-patch routes are explicitly non-identifiable.

        A global label shift would exchange patch-1 and patch-2 labels and make
        replay of the animal's *current* patch look next-patch aligned. Once
        current patch is held fixed, next patch is deterministic in both
        strata, so the scientifically honest result is exclusion.
        """
        events = [
            {"event_id": 0, "occupancy_well_arm": 0, "next_arm": 1, "next_patch": 1},
            {"event_id": 1, "occupancy_well_arm": 0, "next_arm": 1, "next_patch": 1},
            {"event_id": 2, "occupancy_well_arm": 2, "next_arm": 3, "next_patch": 2},
            {"event_id": 3, "occupancy_well_arm": 2, "next_arm": 3, "next_patch": 2},
        ]
        ev = _make_event_table(events)
        ea = _make_event_arm_table(
            {
                row["event_id"]: [
                    float(arm == row["occupancy_well_arm"]) for arm in range(9)
                ]
                for row in events
            }
        )

        frozen_ev, _, _ = freeze_retrospective_cohort(
            ev,
            ea,
            pd.DataFrame(),
            metrics=("chosen_patch_fraction",),
        )
        assert frozen_ev.empty
        result = retrospective_alignment_with_shuffle(
            ev,
            ea,
            pd.DataFrame(),
            n_shuffles=1,
            metrics=("chosen_patch_fraction",),
            rng=np.random.default_rng(0),
        )
        assert result["n_events"] == 0
        assert result["n_shuffle_units_excluded_deterministic"] == 4
        assert np.isnan(result["real_mean_chosen_patch_fraction"])

    def test_long_bout_is_split_into_visit_context_units(self):
        """A bout may cross a boundary and later return to the first well."""
        # Bout 10 holds three visits: arm 0, then arm 2, then arm 0 again.
        # The completed-trip ID is what separates the two arm-0 visits --
        # ``current_arm`` alone reads them as one contiguous run, because the
        # intervening trip leaves no at-well event behind.
        events = [
            {
                "event_id": 0,
                "stop_id": 10,
                "occupancy_well_arm": 0,
                "next_arm": 1,
                "completed_trial_id": 5.0,
            },
            {
                "event_id": 1,
                "stop_id": 10,
                "occupancy_well_arm": 0,
                "next_arm": 1,
                "completed_trial_id": 5.0,
            },
            {
                "event_id": 2,
                "stop_id": 10,
                "occupancy_well_arm": 2,
                "next_arm": 3,
                "completed_trial_id": 6.0,
            },
            {
                "event_id": 3,
                "stop_id": 10,
                "occupancy_well_arm": 2,
                "next_arm": 3,
                "completed_trial_id": 6.0,
            },
            {
                "event_id": 4,
                "stop_id": 10,
                "occupancy_well_arm": 0,
                "next_arm": 4,
                "completed_trial_id": 7.0,
            },
            {
                "event_id": 5,
                "stop_id": 10,
                "occupancy_well_arm": 0,
                "next_arm": 4,
                "completed_trial_id": 7.0,
            },
            {
                "event_id": 6,
                "stop_id": 11,
                "occupancy_well_arm": 0,
                "next_arm": 2,
                "completed_trial_id": 8.0,
            },
            {
                "event_id": 7,
                "stop_id": 12,
                "occupancy_well_arm": 2,
                "next_arm": 4,
                "completed_trial_id": 9.0,
            },
        ]
        ev = _make_event_table(events)
        ea = _make_event_arm_table(
            {
                row["event_id"]: [float(arm == row["next_arm"]) for arm in range(9)]
                for row in events
            }
        )

        result = retrospective_alignment_with_shuffle(
            ev,
            ea,
            pd.DataFrame(),
            n_shuffles=2,
            metrics=("chosen_arm_fraction",),
            rng=np.random.default_rng(0),
        )

        assert ev["stop_id"].nunique() == 3
        assert result["n_shuffle_units_input"] == 5
        assert result["n_shuffle_units"] == 5
        assert result["n_events_chosen_arm_fraction"] == 8

    def test_visit_label_validation_precedes_observability_filter(self):
        """An unobservable event cannot hide conflicting labels in a visit."""
        ev = _make_event_table(
            [
                {
                    "event_id": 0,
                    "stop_id": 10,
                    "occupancy_well_arm": 0,
                    "next_arm": 1,
                    "completed_trial_id": 3.0,
                },
                {
                    "event_id": 1,
                    "stop_id": 10,
                    "occupancy_well_arm": 0,
                    "next_arm": 2,
                    "completed_trial_id": 3.0,
                },
            ]
        )
        # Event 1 is deliberately absent, so an observability-first pipeline
        # would silently discard the conflicting label.
        ea = _make_event_arm_table({0: [0.0, 1.0, *([0.0] * 7)]})

        with pytest.raises(ValueError, match="multiple 'next_arm' values"):
            retrospective_alignment_with_shuffle(
                ev,
                ea,
                pd.DataFrame(),
                n_shuffles=1,
                metrics=("chosen_arm_fraction",),
                rng=np.random.default_rng(0),
            )

    def test_default_families_have_distinct_calibrated_cohorts(self):
        """Arm metrics retain within-patch visits that path cannot permute."""
        events = [
            {
                "event_id": 0,
                "occupancy_well_arm": 0,
                "next_arm": 1,
                "next_patch": 1,
            },
            {
                "event_id": 1,
                "occupancy_well_arm": 0,
                "next_arm": 1,
                "next_patch": 1,
            },
            {
                "event_id": 2,
                "occupancy_well_arm": 0,
                "next_arm": 2,
                "next_patch": 2,
            },
            {
                "event_id": 3,
                "occupancy_well_arm": 0,
                "next_arm": 4,
                "next_patch": 3,
            },
        ]
        ev = _make_event_table(events)
        ea = _make_event_arm_table(
            {
                row["event_id"]: [float(arm == row["next_arm"]) for arm in range(9)]
                for row in events
            }
        )
        es = _make_event_sequence_table(
            {
                0: [0, 1],
                1: [0, 1],
                2: [0, 6, 7, 2],
                3: [0, 6, 8, 4],
            }
        )

        result = retrospective_alignment_with_shuffle(
            ev,
            ea,
            es,
            n_shuffles=2,
            rng=np.random.default_rng(0),
        )

        assert "n_events" not in result
        assert result["n_events_arm_family"] == 4
        assert result["n_events_chosen_arm_fraction"] == 4
        assert result["n_events_ends_at_next"] == 4
        assert result["n_events_patch_family"] == 4
        assert result["n_events_chosen_patch_fraction"] == 4
        assert result["n_events_ends_at_next_patch"] == 4
        assert result["n_events_path_family"] == 2
        assert result["n_events_contains_path_to_next"] == 2
        assert result["n_shuffle_units_excluded_deterministic_path_family"] == 2
        assert result["cohort_policy"] == "metric_family_specific"
        assert result["path_null_effectively_cross_patch"] is True
        assert retrospective_status(result) == "ok"

    def test_family_unit_counts_ignore_qualified_sibling_keys(self):
        """Only the retained-count key names a family.

        ``n_shuffle_units_excluded_singleton_arm_family`` shares the prefix
        and suffix but is a breakdown, and its legitimate zero would read as
        a family with nothing movable.
        """
        result = {
            "n_shuffle_units_arm_family": 4,
            "n_shuffle_units_input_arm_family": 9,
            "n_shuffle_units_excluded_singleton_arm_family": 0,
            "n_shuffle_units_excluded_deterministic_arm_family": 5,
            "n_shuffle_units_path_family": 2,
            "n_shuffle_units_excluded_singleton_path_family": 0,
        }

        assert family_unit_counts(result) == {"arm": 4, "path": 2}
        assert retrospective_status(result) == "ok"

    def test_status_separates_partial_from_no_movable_family(self):
        """A session can estimate one family and not another."""
        assert (
            retrospective_status(
                {"n_shuffle_units_arm_family": 4, "n_shuffle_units_path_family": 1}
            )
            == "partial_movable_families"
        )
        assert (
            retrospective_status(
                {"n_shuffle_units_arm_family": 1, "n_shuffle_units_path_family": 0}
            )
            == "no_movable_family"
        )
        assert retrospective_status({}) == "no_movable_family"

    def test_patch_family_exchanges_visits_across_arms_in_same_patch(self):
        """Patch calibration conditions on current patch, not current arm."""
        events = [
            {
                "event_id": 0,
                "occupancy_well_arm": 0,
                "next_arm": 2,
                "next_patch": 2,
            },
            {
                "event_id": 1,
                "occupancy_well_arm": 1,
                "next_arm": 4,
                "next_patch": 3,
            },
        ]
        ev = _make_event_table(events)
        ea = _make_event_arm_table(
            {
                row["event_id"]: [float(arm == row["next_arm"]) for arm in range(9)]
                for row in events
            }
        )

        result = retrospective_alignment_with_shuffle(
            ev,
            ea,
            pd.DataFrame(),
            n_shuffles=1,
            metrics=("chosen_patch_fraction",),
            rng=np.random.default_rng(0),
        )

        assert result["n_events"] == 2
        assert result["n_shuffle_units"] == 2
        assert result["n_shuffle_strata"] == 1

    def test_reports_singleton_and_deterministic_visit_exclusions(self):
        events = [
            {"event_id": 0, "occupancy_well_arm": 0, "next_arm": 2},
            {"event_id": 1, "occupancy_well_arm": 0, "next_arm": 4},
            {"event_id": 2, "occupancy_well_arm": 1, "next_arm": 3},
            {"event_id": 3, "occupancy_well_arm": 2, "next_arm": 3},
            {"event_id": 4, "occupancy_well_arm": 2, "next_arm": 3},
        ]
        ev = _make_event_table(events)
        ea = _make_event_arm_table(
            {
                row["event_id"]: [float(arm == row["next_arm"]) for arm in range(9)]
                for row in events
            }
        )

        result = retrospective_alignment_with_shuffle(
            ev,
            ea,
            pd.DataFrame(),
            n_shuffles=1,
            metrics=("chosen_arm_fraction",),
            rng=np.random.default_rng(0),
        )

        assert result["n_events"] == 2
        assert result["n_shuffle_units_input"] == 5
        assert result["n_shuffle_units"] == 2
        assert result["n_shuffle_units_excluded_singleton"] == 1
        assert result["n_shuffle_units_excluded_deterministic"] == 2
        assert result["n_shuffle_strata"] == 1
        assert result["n_shuffle_strata_excluded_singleton"] == 1
        assert result["n_shuffle_strata_excluded_deterministic"] == 1

    def test_stop_grain_shuffle_preserves_labels_and_length_strata(self):
        rows = []
        event_id = 0
        stop_id = 0
        strata = (
            (0, 0, (2, 3)),
            (0, 1, (4, 5)),
            (2, 0, (0, 1)),
        )
        for current_arm, length_bin, targets in strata:
            for target in targets:
                for within_stop in range(2):
                    rows.append(
                        {
                            "event_id": event_id,
                            "stop_id": stop_id,
                            "event_midpoint_s": float(event_id),
                            "occupancy_well_arm": current_arm,
                            "route_length": 4,
                            "length_bin": length_bin,
                            "next_arm": target,
                            "within_stop": within_stop,
                        }
                    )
                    event_id += 1
                stop_id += 1
        events = pd.DataFrame(rows)

        shuffled = next_arm_circular_shift(
            events,
            columns=("next_arm",),
            unit_col="stop_id",
            strata_cols=("occupancy_well_arm", "route_length", "length_bin"),
            variation_cols=("next_arm",),
            require_movement=True,
            rng=np.random.default_rng(0),
        )

        assert shuffled.groupby("stop_id")["next_arm"].nunique().eq(1).all()
        original_by_stop = events.groupby("stop_id")["next_arm"].first()
        shuffled_by_stop = shuffled.groupby("stop_id")["next_arm"].first()
        assert original_by_stop.ne(shuffled_by_stop).all()
        np.testing.assert_array_equal(
            shuffled["occupancy_well_arm"], events["occupancy_well_arm"]
        )
        np.testing.assert_array_equal(shuffled["route_length"], events["route_length"])
        np.testing.assert_array_equal(shuffled["length_bin"], events["length_bin"])
        for current_arm, length_bin, targets in strata:
            in_stratum = shuffled["occupancy_well_arm"].eq(current_arm) & shuffled[
                "length_bin"
            ].eq(length_bin)
            assert set(shuffled.loc[in_stratum, "next_arm"]) == set(targets)

    def test_shuffle_destroys_alignment_on_aligned_data(self):
        rng = np.random.default_rng(0)
        # Build a session where chosen_arm_fraction is reliably 0.6 on the
        # actual next_arm and ~0.05 elsewhere — strong alignment by design.
        n_events = 80
        events = []
        per_arm_fracs = {}
        for i in range(n_events):
            current_arm = int(rng.integers(0, 6))
            # Pick a next_arm distinct from current.
            choices = [a for a in range(6) if a != current_arm]
            next_arm = int(rng.choice(choices))
            events.append(
                {
                    "event_id": i,
                    "occupancy_well_arm": current_arm,
                    "next_arm": next_arm,
                    "event_midpoint_s": float(i),
                }
            )
            arm_fracs = [0.05] * 9
            arm_fracs[next_arm] = 0.6
            # Renormalise so it sums to 1.
            s = sum(arm_fracs)
            arm_fracs = [f / s for f in arm_fracs]
            per_arm_fracs[i] = arm_fracs
        ev = _make_event_table(events)
        ea = _make_event_arm_table(per_arm_fracs)
        # Sequence table not used for chosen_arm_fraction.
        es = _make_event_sequence_table(
            {i: [events[i]["occupancy_well_arm"]] for i in range(n_events)}
        )

        result = retrospective_alignment_with_shuffle(
            ev,
            ea,
            es,
            n_shuffles=20,
            rng=np.random.default_rng(0),
            metrics=("chosen_arm_fraction",),
        )
        # Real chosen_arm_fraction should be ~0.6/(0.6+8*0.05) ≈ 0.6
        # (after renormalisation), much greater than the ~1/9 chance.
        assert result["real_mean_chosen_arm_fraction"] > 0.5
        assert result["shuffled_mean_chosen_arm_fraction"] < 0.2
        assert result["real_minus_shuffled_chosen_arm_fraction"] > 0.3


# ─────────────────────────────────────────────────────────────────────
# prev_arm retrospective mirror
# ─────────────────────────────────────────────────────────────────────


class TestPrevArmMirror:
    """The prev_arm mirror reuses the same cores, targeting prev/current."""

    def test_chosen_arm_fraction_prev_picks_prev_arm(self):
        ev = _make_event_table(
            [
                {"event_id": 1, "occupancy_well_arm": 3, "prev_arm": 0, "next_arm": 3},
                {"event_id": 2, "occupancy_well_arm": 0, "prev_arm": 3, "next_arm": 0},
            ]
        )
        ea = _make_event_arm_table(
            {
                1: [0.40, 0, 0, 0.10, 0, 0, 0, 0, 0],  # arm 0 (prev) -> 0.40
                2: [0.10, 0, 0, 0.30, 0, 0, 0, 0, 0],  # arm 3 (prev) -> 0.30
            }
        )
        out = chosen_arm_fraction_prev(ev, ea).set_index("event_id")
        assert out.loc[1, "chosen_arm_fraction_prev"] == pytest.approx(0.40)
        assert out.loc[2, "chosen_arm_fraction_prev"] == pytest.approx(0.30)

    def test_ends_at_prev_last_bin_matches_prev_arm(self):
        ev = _make_event_table(
            [{"event_id": 1, "occupancy_well_arm": 3, "prev_arm": 0, "next_arm": 3}]
        )
        es = _make_event_sequence_table({1: [3, 7, 6, 0]})  # ends at arm 0 (prev)
        out = ends_at_prev(ev, es).set_index("event_id")
        assert bool(out.loc[1, "ends_at_prev"]) is True

        es2 = _make_event_sequence_table({1: [3, 7, 6, 1]})  # ends at arm 1
        out2 = ends_at_prev(ev, es2).set_index("event_id")
        assert bool(out2.loc[1, "ends_at_prev"]) is False

    def test_contains_path_from_prev_covers_prev_to_current(self):
        # prev=0, current=3 -> cross-patch path {0,3,6,7} (leaves + trunks)
        ev = _make_event_table(
            [{"event_id": 1, "occupancy_well_arm": 3, "prev_arm": 0, "next_arm": 3}]
        )
        path = sorted(arm_path_well_to_well(0, 3))
        es_full = _make_event_sequence_table({1: path})
        assert (
            bool(
                contains_path_from_prev(ev, es_full)
                .set_index("event_id")
                .loc[1, "contains_path_from_prev"]
            )
            is True
        )
        es_partial = _make_event_sequence_table({1: [0, 3]})  # missing trunks
        assert (
            bool(
                contains_path_from_prev(ev, es_partial)
                .set_index("event_id")
                .loc[1, "contains_path_from_prev"]
            )
            is False
        )

    def test_compute_metrics_includes_prev_columns(self):
        ev = _make_event_table(
            [
                {
                    "event_id": 1,
                    "occupancy_well_arm": 3,
                    "prev_arm": 0,
                    "next_arm": 3,
                    "next_patch": 2,
                }
            ]
        )
        ea = _make_event_arm_table({1: [0.4, 0, 0, 0.3, 0, 0, 0, 0, 0]})
        es = _make_event_sequence_table({1: [3, 7, 6, 0]})
        out = compute_retrospective_metrics(ev, ea, es)
        for col in (
            "chosen_arm_fraction_prev",
            "ends_at_prev",
            "contains_path_from_prev",
        ):
            assert col in out.columns
        # include_prev=False omits them
        out2 = compute_retrospective_metrics(ev, ea, es, include_prev=False)
        assert "ends_at_prev" not in out2.columns

    def test_ends_at_prev_missing_sequence_records_false_bool(self):
        ev = _make_event_table(
            [{"event_id": 1, "occupancy_well_arm": 3, "prev_arm": 0, "next_arm": 3}]
        )
        es = _make_event_sequence_table({2: [0, 1]})  # rows for event 2, not 1
        out = ends_at_prev(ev, es).set_index("event_id")
        assert out["ends_at_prev"].dtype == bool
        assert bool(out.loc[1, "ends_at_prev"]) is False


def test_the_shuffle_null_needs_the_callers_generator():
    """The null draws its shifts from the caller's generator, never an unseeded one."""
    empty = pd.DataFrame()
    with pytest.raises(TypeError, match="rng"):
        retrospective_alignment_with_shuffle(empty, empty, empty, n_shuffles=1)


def _tables_of(epoch: str, event_ids: list[int], is_swr: list[bool]):
    ev = _make_event_table(
        [
            {
                "event_id": event_id,
                "occupancy_well_arm": 0,
                "next_arm": 1,
                "next_patch": 1,
                "is_swr": swr,
            }
            for event_id, swr in zip(event_ids, is_swr, strict=True)
        ]
    ).assign(epoch=epoch)
    ea = _make_event_arm_table(
        {event_id: [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0] for event_id in event_ids}
    ).assign(epoch=epoch)
    ea["is_swr"] = ea["event_id"].map(dict(zip(event_ids, is_swr, strict=True)))
    es = _make_event_sequence_table(
        {event_id: [0, 1] for event_id in event_ids}
    ).assign(epoch=epoch)
    return ev, ea, es


def test_an_event_subset_keeps_the_same_events_in_all_three_tables():
    ev, ea, es = _tables_of("01_r1", [1, 2, 3], [True, False, True])

    swr = select_event_subset(ev, ea, es, "swr_only")
    non_swr = select_event_subset(ev, ea, es, "non_swr_only")
    everything = select_event_subset(ev, ea, es, "all")

    assert [set(t["event_id"]) for t in swr] == [{1, 3}] * 3
    assert [set(t["event_id"]) for t in non_swr] == [{2}] * 3
    assert all(a is b for a, b in zip(everything, (ev, ea, es), strict=True))
    with pytest.raises(ValueError, match="subset must be one of"):
        select_event_subset(ev, ea, es, "mua_only")


def test_the_sessions_are_run_in_order_each_with_its_own_seeded_draw():
    first = _tables_of("01_r1", [1, 2], [True, False])
    second = _tables_of("02_r1", [3, 4], [True, True])
    ev, ea, es = (
        pd.concat([a, b], ignore_index=True) for a, b in zip(first, second, strict=True)
    )

    def run(**kwargs):
        return retrospective_alignment_by_session(
            ev,
            ea,
            es,
            n_shuffles=2,
            seed=7,
            subset="all",
            stop_cohort="at_well",
            **kwargs,
        ).drop(columns="elapsed_s")

    out = run()
    assert list(out["epoch"]) == ["01_r1", "02_r1"]
    assert set(out["subset"]) == {"all"}
    assert set(out["stop_cohort"]) == {"at_well"}
    # Each session's events share one stop at one well: one visit context, so
    # no family can move a label.
    assert set(out["retrospective_status"]) == {"no_movable_family"}
    pd.testing.assert_frame_equal(out, run())
    # The first session's draw does not depend on the sessions after it.
    pd.testing.assert_frame_equal(run(max_sessions=1), out.iloc[:1])
