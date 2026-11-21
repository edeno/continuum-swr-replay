"""ContFrag output schemas carry event-source provenance."""

from __future__ import annotations

from continuum_swr_replay.analysis.contfrag.tables import (
    _CONTFRAG_STOP_TABLE_COLUMNS,
    _contfrag_event_arm_table_columns,
    _contfrag_event_table_columns,
)


def test_contfrag_table_schemas_include_source_and_area_provenance():
    event_cols = _contfrag_event_table_columns("SWR")
    stop_cols = _CONTFRAG_STOP_TABLE_COLUMNS
    event_arm_cols = _contfrag_event_arm_table_columns("MUA")

    assert event_cols[:5] == [
        "animal",
        "session_date",
        "epoch",
        "event_source",
        "brain_area",
    ]
    assert list(stop_cols[:5]) == [
        "animal",
        "session_date",
        "epoch",
        "event_source",
        "brain_area",
    ]
    assert event_arm_cols[:5] == [
        "animal",
        "session_date",
        "epoch",
        "event_source",
        "brain_area",
    ]


def test_contfrag_stop_table_schema_is_pinned_exactly():
    """The full ContFrag stop-table column list, pinned in order.

    The prefix assertion above only covers the five provenance columns, so a
    dropped or reordered column anywhere after them is invisible: deleting
    ``immobility_time`` — the exposure offset every rate model divides by —
    leaves the whole contfrag suite green. Pin the entire list in order so any
    schema edit has to be deliberate.
    """
    stop_cols = _CONTFRAG_STOP_TABLE_COLUMNS

    assert list(stop_cols) == [
        "animal",
        "session_date",
        "epoch",
        "event_source",
        "brain_area",
        "stop_idx",
        "stop_id",
        "stop_start",
        "stop_end",
        "immobility_duration_s",
        "reward_fraction_last10_known_trials",
        "is_swr_stop",
        "is_mua_stop",
        "has_replay",
        "n_selected_events",
        "did_last_completed_trip_switch_patch",
        "is_rewarded_outcome",
        "completed_trial_id",
        "completed_trial_arm",
        "patch_switch_fraction_last5_completed_trips",
        "leaky_q_value_diff",
        "is_stay",
        "occupancy_state",
        "at_well",
        "occupancy_well_id",
        "modal_observed_track_segment_id",
        "nearest_well_distance_cm_min",
        "nearest_well_distance_cm_mean",
        "nearest_well_distance_cm_max",
        "visit_context",
        "well_visit_id",
        "last_completed_well_visit_id",
        "time_since_last_poke_out_s",
        "time_until_next_poke_in_s",
        "overlapping_visit_id",
        "visit_first_poke_in_s",
        "visit_last_poke_out_s",
        "in_visit_immobile_s",
        "visit_stretch_index",
        "stop_end_kind",
        "occupancy_well_arm",
        "prev_arm",
        "next_arm",
        "last_rewarded_arm",
        "current_patch",
        "prev_patch",
        "next_patch",
        "reward_history_rcos_0",
        "reward_history_rcos_1",
        "reward_history_rcos_2",
        "reward_history_incomplete",
    ]
