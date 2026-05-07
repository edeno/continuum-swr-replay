"""Tests for analysis/replay/retrospective.py.

Stage 5B asks (post hoc) whether replay content during a stop aligns
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

from continuum_swr_replay.analysis.replay.retrospective import (
    arm_path_well_to_well,
    chosen_arm_fraction,
    chosen_patch_fraction,
    compute_retrospective_metrics,
    contains_path_to_next,
    ends_at_next,
    ends_at_next_patch,
    retrospective_alignment_with_shuffle,
)

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
        with pytest.raises(ValueError):
            arm_path_well_to_well(6, 0)
        with pytest.raises(ValueError):
            arm_path_well_to_well(0, 9)


# ─────────────────────────────────────────────────────────────────────────────
# Shared minimal-tables fixture
# ─────────────────────────────────────────────────────────────────────────────


def _make_event_table(events: list[dict]) -> pd.DataFrame:
    """Build a minimal event_table with the columns Stage 5B needs."""
    df = pd.DataFrame(events)
    # Fill columns retrospective.py expects but tests don't always set
    for col, default in {
        "animal": "test",
        "day": 1,
        "epoch": "01_r1",
        "event_midpoint_time": np.nan,
    }.items():
        if col not in df.columns:
            df[col] = default
    if "event_midpoint_time" in df.columns and df["event_midpoint_time"].isna().all():
        df["event_midpoint_time"] = np.arange(len(df), dtype=float)
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
                    "day": 1,
                    "epoch": "01_r1",
                    "event_id": ev_id,
                    "arm": arm,
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
                    "day": 1,
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
                {"event_id": 1, "current_arm": 0, "next_arm": 3},
                {"event_id": 2, "current_arm": 3, "next_arm": 0},
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
                {"event_id": 1, "current_arm": 0, "next_arm": np.nan},
                {"event_id": 2, "current_arm": 3, "next_arm": 0},
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
                {"event_id": 1, "current_arm": 0, "next_arm": 3},
            ]
        )
        # last in-event bin is arm 3 → True
        es = _make_event_sequence_table({1: [0, 6, 7, 3]})
        out = ends_at_next(ev, es).set_index("event_id")
        assert bool(out.loc[1, "ends_at_next"]) is True

    def test_last_bin_mismatches_next_arm(self):
        ev = _make_event_table(
            [
                {"event_id": 1, "current_arm": 0, "next_arm": 3},
            ]
        )
        # last in-event bin is arm 0 → False
        es = _make_event_sequence_table({1: [3, 6, 7, 0]})
        out = ends_at_next(ev, es).set_index("event_id")
        assert bool(out.loc[1, "ends_at_next"]) is False

    def test_singleton_event(self):
        ev = _make_event_table([{"event_id": 1, "current_arm": 0, "next_arm": 0}])
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
                {"event_id": 1, "current_arm": 0, "next_arm": 3},
                {"event_id": 2, "current_arm": 0, "next_arm": 3},
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
        from continuum_swr_replay.analysis.replay.retrospective import (
            _per_session_metric_means,
        )

        ev = _make_event_table(
            [
                {"event_id": 1, "current_arm": 0, "next_arm": 3},
                {"event_id": 2, "current_arm": 0, "next_arm": 3},
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
                {"event_id": 1, "current_arm": 0, "next_arm": 2},
            ]
        )
        # 0→2 path = {0,2,6,7}; sequence covers all four arms (in any order).
        es = _make_event_sequence_table({1: [0, 6, 7, 2, 7]})
        out = contains_path_to_next(ev, es).set_index("event_id")
        assert bool(out.loc[1, "contains_path_to_next"]) is True

    def test_missing_one_path_arm(self):
        ev = _make_event_table(
            [
                {"event_id": 1, "current_arm": 0, "next_arm": 2},
            ]
        )
        # Missing trunk 7 → False.
        es = _make_event_sequence_table({1: [0, 6, 2]})
        out = contains_path_to_next(ev, es).set_index("event_id")
        assert bool(out.loc[1, "contains_path_to_next"]) is False

    def test_same_patch_path_short(self):
        ev = _make_event_table(
            [
                {"event_id": 1, "current_arm": 0, "next_arm": 1},
            ]
        )
        # 0→1 path = {0,1}; sequence with both arms suffices.
        es = _make_event_sequence_table({1: [0, 6, 1]})
        out = contains_path_to_next(ev, es).set_index("event_id")
        assert bool(out.loc[1, "contains_path_to_next"]) is True


# ─────────────────────────────────────────────────────────────────────────────
# chosen_patch_fraction (sum of arm_fraction over arms in next_patch)
# ─────────────────────────────────────────────────────────────────────────────


class TestChosenPatchFraction:
    def test_sums_arms_in_target_patch(self):
        # next_patch == 2 → arms {2, 3, 7}; sum their arm_fractions.
        ev = _make_event_table(
            [
                {"event_id": 1, "current_arm": 0, "next_arm": 3, "next_patch": 2},
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
                {"event_id": 1, "current_arm": 0, "next_arm": 0, "next_patch": np.nan},
                {"event_id": 2, "current_arm": 3, "next_arm": 3, "next_patch": 2},
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
                {"event_id": 1, "current_arm": 0, "next_arm": 3, "next_patch": 2},
            ]
        )
        es = _make_event_sequence_table({1: [0, 6, 7, 3]})
        out = ends_at_next_patch(ev, es).set_index("event_id")
        assert bool(out.loc[1, "ends_at_next_patch"]) is True

    def test_last_bin_in_different_patch(self):
        # next_patch=2; last bin arm=4 (patch 3) → False
        ev = _make_event_table(
            [
                {"event_id": 1, "current_arm": 0, "next_arm": 3, "next_patch": 2},
            ]
        )
        es = _make_event_sequence_table({1: [0, 6, 7, 4]})
        out = ends_at_next_patch(ev, es).set_index("event_id")
        assert bool(out.loc[1, "ends_at_next_patch"]) is False

    def test_last_bin_on_trunk_arm_of_target_patch(self):
        # last bin arm=7 (trunk for patch 2); ends_at_next_patch=2 → True
        ev = _make_event_table(
            [
                {"event_id": 1, "current_arm": 0, "next_arm": 3, "next_patch": 2},
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
                {"event_id": 1, "current_arm": 0, "next_arm": 3, "next_patch": 2},
                {"event_id": 2, "current_arm": 0, "next_arm": 3, "next_patch": 2},
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
        from continuum_swr_replay.analysis.replay.retrospective import (
            _per_session_metric_means,
        )

        ev = _make_event_table(
            [
                {"event_id": 1, "current_arm": 0, "next_arm": 3, "next_patch": 2},
                {"event_id": 2, "current_arm": 0, "next_arm": 3, "next_patch": 2},
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
                {"event_id": 1, "current_arm": 0, "next_arm": 3},
                {"event_id": 2, "current_arm": 3, "next_arm": 4},
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
                "animal", "day", "epoch", "event_id", "current_arm",
                "next_arm", "next_patch",
            ]
        )
        ea = pd.DataFrame(
            columns=["animal", "day", "epoch", "event_id", "arm", "arm_fraction"]
        )
        es = pd.DataFrame(
            columns=[
                "animal", "day", "epoch", "event_id", "t_in_event",
                "decoded_arm_map",
            ]
        )

        out = compute_retrospective_metrics(ev, ea, es)
        assert out.empty
        assert {"animal", "day", "epoch", "event_id"}.issubset(out.columns)


# ─────────────────────────────────────────────────────────────────────────────
# retrospective_alignment_with_shuffle
# ─────────────────────────────────────────────────────────────────────────────


class TestRetrospectiveAlignmentWithShuffle:
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
                    "current_arm": current_arm,
                    "next_arm": next_arm,
                    "event_midpoint_time": float(i),
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
            {i: [events[i]["current_arm"]] for i in range(n_events)}
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
