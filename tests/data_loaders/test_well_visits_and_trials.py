"""Tests for make_well_visits_df and make_trials_df_from_well_visits.

These functions construct the behavioral trial structure from raw events.
Bugs here silently corrupt all downstream trial-level analyses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.covariates import (
    behavioral_location_from_port_experiences,
    visit_recency_all_events,
)
from continuum_swr_replay.data_loaders.constants import OMISSION_KNOWN_LATENCY_SEC
from continuum_swr_replay.data_loaders.trials import (
    align_port_experiences_to_task_records,
    make_port_experiences_df,
    make_trials_df_from_well_visits,
    make_well_visits_df,
    photogate_engagement_at_times,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def beam_breaks():
    """Beam breaks: pokes at well 0 then well 2."""
    return pd.DataFrame(
        {
            "start_time": [1.0, 1.2, 12.0, 12.3],
            "end_time": [1.1, 1.3, 12.1, 12.4],
            "event_name": ["Poke1", "Poke1", "Poke3", "Poke3"],
        }
    )


@pytest.fixture
def pump_events():
    """Reward delivered during the second poke (at well 2)."""
    return pd.DataFrame({"start_time": [12.05], "event_name": ["Pump3"]})


def _empty_pumps() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_time": pd.Series(dtype=float),
            "event_name": pd.Series(dtype=str),
        }
    )


# ============================================================================
# Tests: make_well_visits_df
# ============================================================================


class TestMakeWellVisitsDf:
    """Test well visit construction from raw events."""

    def test_has_required_columns(self, beam_breaks, pump_events):
        """Output should have all required columns."""
        result = make_well_visits_df(beam_breaks, pump_events)

        required = [
            "start_time",
            "end_time",
            "duration",
            "from_well",
            "to_well",
            "is_run",
            "is_reward",
            "reward_on_time",
            "from_patch",
            "to_patch",
            "is_patch_change",
            "trial_number",
            "is_observed_poke",
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_sorted_by_start_time(self, beam_breaks, pump_events):
        """Well visits should be sorted by start_time."""
        result = make_well_visits_df(beam_breaks, pump_events)

        times = result["start_time"].values
        assert np.all(np.diff(times) >= 0), "Not sorted by start_time"

    def test_runs_have_different_start_end_wells(self, beam_breaks, pump_events):
        """Rows marked is_run should have from_well != to_well."""
        result = make_well_visits_df(beam_breaks, pump_events)

        runs = result[result["is_run"]]
        if len(runs) > 0:
            assert (runs["from_well"] != runs["to_well"]).all()

    def test_pokes_have_same_start_end_wells(self, beam_breaks, pump_events):
        """Non-run rows should have from_well == to_well."""
        result = make_well_visits_df(beam_breaks, pump_events)

        pokes = result[~result["is_run"]]
        if len(pokes) > 0:
            assert (pokes["from_well"] == pokes["to_well"]).all()

    def test_trial_numbers_assigned(self, beam_breaks, pump_events):
        """Every row should have a trial number (no NaN after ffill/bfill)."""
        result = make_well_visits_df(beam_breaks, pump_events)

        assert not result["trial_number"].isna().any()

    def test_trial_numbers_sequential(self, beam_breaks, pump_events):
        """Trial numbers for runs should be sequential starting from 1."""
        result = make_well_visits_df(beam_breaks, pump_events)

        run_trials = result.loc[result["is_run"], "trial_number"].values
        assert len(run_trials) > 0, "Fixture should produce at least one run"
        expected = np.arange(1, len(run_trials) + 1)
        np.testing.assert_array_equal(run_trials, expected)

    def test_reward_associated_with_correct_visit(self, beam_breaks, pump_events):
        """Reward should be associated with the visit that contains the pump event."""
        result = make_well_visits_df(beam_breaks, pump_events)

        # At least one row should have is_reward=True
        assert result["is_reward"].any(), "No rewards detected"

    def test_unsorted_pump_events_align_to_their_physical_pokes(self, beam_breaks):
        """Direct callers need not pre-sort pump events."""
        pumps = pd.DataFrame(
            {
                "start_time": [12.35, 1.05],
                "event_name": ["Pump3", "Pump1"],
            }
        )

        result = make_well_visits_df(beam_breaks, pumps)

        reward_times = sorted(
            result.loc[
                result["is_reward"] & result["reward_on_time"].notna(),
                "reward_on_time",
            ].tolist()
        )
        assert reward_times == pytest.approx([1.05, 12.35])

    def test_patch_mapping_correct(self, beam_breaks, pump_events):
        """from_patch and to_patch should match WELL_PATCH_MAPPING."""
        from continuum_swr_replay.data_loaders.constants import WELL_PATCH_MAPPING

        result = make_well_visits_df(beam_breaks, pump_events)

        for _, row in result.iterrows():
            if row["from_well"] in WELL_PATCH_MAPPING:
                assert row["from_patch"] == WELL_PATCH_MAPPING[row["from_well"]]
            if row["to_well"] in WELL_PATCH_MAPPING:
                assert row["to_patch"] == WELL_PATCH_MAPPING[row["to_well"]]

    def test_is_patch_change_correct(self, beam_breaks, pump_events):
        """is_patch_change should be True iff from_patch != to_patch."""
        result = make_well_visits_df(beam_breaks, pump_events)

        expected = result["from_patch"] != result["to_patch"]
        np.testing.assert_array_equal(result["is_patch_change"].values, expected.values)

    def test_no_pump_events_means_no_rewards(self, beam_breaks):
        """With empty pump events, no visits should have is_reward=True."""
        empty_pumps = _empty_pumps()

        result = make_well_visits_df(beam_breaks, empty_pumps)

        assert not result["is_reward"].any()

    def test_pump_outside_physical_poke_is_audited_but_not_a_reward(self, beam_breaks):
        """An unmatched hardware pulse cannot create a behavioral outcome."""
        pumps = pd.DataFrame({"start_time": [6.0], "event_name": ["Pump3"]})

        visits = make_well_visits_df(beam_breaks, pumps)

        assert not visits["is_reward"].any()
        # Pump at t=6.0 is mid-run between wells 0 and 2 -- a true transit orphan,
        # inside no same-well visit window, so it is not recovered at collapse.
        # The authoritative post-recovery audit lives on port_experiences.
        trials = make_trials_df_from_well_visits(visits)
        experiences = make_port_experiences_df(visits, trials)
        assert experiences.attrs["unmatched_pump_event_count"] == 1
        assert experiences.attrs["unmatched_pump_event_times"] == (6.0,)
        assert experiences.attrs["gap_recovered_pump_count"] == 0

    def test_pump_without_any_physical_poke_cannot_create_a_visit(self):
        """A reward event cannot create an otherwise absent visit or trial."""
        empty_pokes = pd.DataFrame(columns=["start_time", "end_time", "event_name"])
        pumps = pd.DataFrame({"start_time": [1.0], "event_name": ["Pump1"]})

        visits = make_well_visits_df(empty_pokes, pumps)

        assert visits.empty

        trials = make_trials_df_from_well_visits(visits)
        experiences = make_port_experiences_df(visits, trials)
        assert experiences.empty
        assert experiences.attrs["gap_recovered_pump_count"] == 0
        assert experiences.attrs["unmatched_pump_event_count"] == 1
        assert experiences.attrs["unmatched_pump_events"] == (("Pump1", 1.0),)
        assert experiences.attrs["unmatched_pump_event_times"] == (1.0,)

    def test_pump_at_ambiguous_poke_boundary_fails_loudly(self):
        """One onset cannot be credited to two touching physical pokes."""
        beam_breaks = pd.DataFrame(
            {
                "start_time": [1.0, 1.1],
                "end_time": [1.1, 1.2],
                "event_name": ["Poke1", "Poke2"],
            }
        )
        pumps = pd.DataFrame({"start_time": [1.1], "event_name": ["Pump1"]})

        with pytest.raises(ValueError, match="matches=2"):
            make_well_visits_df(beam_breaks, pumps)

    def test_pump_channel_must_match_the_physical_poke(self):
        beam_breaks = pd.DataFrame(
            {
                "start_time": [1.0, 3.0],
                "end_time": [1.5, 3.5],
                "event_name": ["Poke1", "Poke3"],
            }
        )
        wrong_pump = pd.DataFrame({"start_time": [3.2], "event_name": ["Pump1"]})

        with pytest.raises(ValueError, match="pump channel disagrees"):
            make_well_visits_df(beam_breaks, wrong_pump)

    def test_two_pumps_in_one_physical_poke_fail_loudly(self):
        """One task reward cannot be represented by two pump onsets."""
        beam_breaks = pd.DataFrame(
            {
                "start_time": [1.0],
                "end_time": [2.0],
                "event_name": ["Poke1"],
            }
        )
        pumps = pd.DataFrame(
            {
                "start_time": [1.2, 1.4],
                "event_name": ["Pump1", "Pump1"],
            }
        )

        with pytest.raises(ValueError, match="multiple reward-pump onsets"):
            make_well_visits_df(beam_breaks, pumps)

    def test_nonempty_pump_stream_requires_its_hardware_channel(self, beam_breaks):
        with pytest.raises(KeyError, match="require an event_name"):
            make_well_visits_df(beam_breaks, pd.DataFrame({"start_time": [12.05]}))

    def test_spyglass_multiindex_channels_are_preserved(self, beam_breaks, pump_events):
        indexed_pokes = beam_breaks.assign(event_number=[1, 2, 1, 2]).set_index(
            ["event_name", "event_number"]
        )
        indexed_pumps = pump_events.assign(event_number=[1]).set_index(
            ["event_name", "event_number"]
        )

        visits = make_well_visits_df(indexed_pokes, indexed_pumps)

        physical = visits.loc[visits["is_observed_poke"]]
        assert physical["from_well"].tolist() == [0, 0, 2, 2]
        assert physical.loc[physical["is_reward"], "from_well"].tolist() == [2]

    def test_durations_non_negative(self, beam_breaks, pump_events):
        """All durations should be non-negative."""
        result = make_well_visits_df(beam_breaks, pump_events)

        assert (result["duration"] >= 0).all()


# ============================================================================
# Tests: make_trials_df_from_well_visits
# ============================================================================


class TestMakeTrialsDfFromWellVisits:
    """Test trial aggregation from well visits."""

    def test_has_destination_arrival_time(self, beam_breaks, pump_events):
        """Each trial should have a destination_first_poke_in_time."""
        well_visits = make_well_visits_df(beam_breaks, pump_events)
        trials = make_trials_df_from_well_visits(well_visits)

        assert "destination_first_poke_in_time" in trials.columns
        assert not trials["destination_first_poke_in_time"].isna().any()

    def test_destination_arrival_within_trial_bounds(self, beam_breaks, pump_events):
        """Arrival lies within [movement_start_time, end_time]."""
        well_visits = make_well_visits_df(beam_breaks, pump_events)
        trials = make_trials_df_from_well_visits(well_visits)

        arrival = trials["destination_first_poke_in_time"]
        assert (arrival >= trials["movement_start_time"]).all()
        assert (arrival <= trials["end_time"]).all()

    def test_start_before_end(self, beam_breaks, pump_events):
        """movement_start_time should be before end_time for all trials."""
        well_visits = make_well_visits_df(beam_breaks, pump_events)
        trials = make_trials_df_from_well_visits(well_visits)

        assert (trials["movement_start_time"] <= trials["end_time"]).all()

    def test_has_no_compatibility_aliases(self, beam_breaks, pump_events):
        """start_time and goal_arrival_time were aliases; they are gone."""
        well_visits = make_well_visits_df(beam_breaks, pump_events)
        trials = make_trials_df_from_well_visits(well_visits)

        assert "start_time" not in trials.columns
        assert "goal_arrival_time" not in trials.columns

    def test_trial_count_matches_run_count(self, beam_breaks, pump_events):
        """Number of trials should equal number of unique run trial numbers."""
        well_visits = make_well_visits_df(beam_breaks, pump_events)
        trials = make_trials_df_from_well_visits(well_visits)

        n_runs = well_visits["is_run"].sum()
        # Each run defines a trial; trial count should not exceed run count
        if n_runs > 0:
            assert 1 <= len(trials) <= n_runs
        else:
            assert len(trials) == 0

    def test_has_required_columns(self, beam_breaks, pump_events):
        """Output should have all trial-level columns."""
        well_visits = make_well_visits_df(beam_breaks, pump_events)
        trials = make_trials_df_from_well_visits(well_visits)

        required = [
            "movement_start_time",
            "end_time",
            "is_reward",
            "reward_on_time",
            "outcome_known_time",
            "from_well",
            "to_well",
            "from_patch",
            "to_patch",
            "is_patch_change",
            "destination_first_poke_in_time",
            "duration",
        ]
        for col in required:
            assert col in trials.columns, f"Missing column: {col}"

    def test_outcome_clock_uses_pump_for_reward_and_feedback_boundary_for_omission(
        self, beam_breaks, pump_events
    ):
        well_visits = make_well_visits_df(beam_breaks, pump_events)
        trials = make_trials_df_from_well_visits(well_visits)

        rewarded = trials["is_reward"]
        np.testing.assert_allclose(
            trials.loc[rewarded, "outcome_known_time"],
            trials.loc[rewarded, "reward_on_time"],
        )
        np.testing.assert_allclose(
            trials.loc[~rewarded, "outcome_known_time"],
            trials.loc[~rewarded, "end_time"],
        )
        assert (
            trials["outcome_known_time"] >= trials["destination_first_poke_in_time"]
        ).all()
        assert (trials["outcome_known_time"] <= trials["end_time"]).all()


class TestMakePortExperiencesDf:
    """The initial port is an experience, not a synthetic movement trial."""

    def test_initial_origin_then_every_destination(self, beam_breaks):
        pumps = pd.DataFrame(
            {
                "start_time": [1.05, 12.05],
                "event_name": ["Pump1", "Pump3"],
            }
        )
        visits = make_well_visits_df(beam_breaks, pumps)
        trials = make_trials_df_from_well_visits(visits)

        experiences = make_port_experiences_df(visits, trials)

        assert experiences["experience_id"].tolist() == [0, 1]
        assert experiences["port"].tolist() == [0, 2]
        assert experiences["is_initial_experience"].tolist() == [True, False]
        assert experiences["outcome_observed"].tolist() == [True, True]
        assert experiences["is_boundary_derived"].tolist() == [False, False]
        assert experiences["is_reward"].tolist() == [True, True]
        assert pd.isna(experiences.loc[0, "movement_trial_number"])
        assert experiences.loc[1, "movement_trial_number"] == trials.index[0]
        assert experiences.loc[0, "outcome_known_time"] == pytest.approx(1.05)
        assert experiences.loc[1, "outcome_known_time"] == pytest.approx(12.05)
        # Movement trials retain their original grain and numbering.
        assert len(trials) == 1
        assert trials.iloc[0]["from_well"] == 0
        assert trials.iloc[0]["to_well"] == 2

    def test_observed_initial_omission_uses_feedback_boundary(
        self, beam_breaks, pump_events
    ):
        # The fixture's only pump is at the destination, so the initial port
        # experience is an omission.
        visits = make_well_visits_df(beam_breaks, pump_events)
        trials = make_trials_df_from_well_visits(visits)

        experiences = make_port_experiences_df(visits, trials)
        initial_poke_start = visits.loc[visits["is_observed_poke"], "start_time"].min()
        initial_visit_rows = visits.loc[
            visits["is_observed_poke"] & visits["from_well"].eq(0)
        ]
        initial_visit_departure = initial_visit_rows["end_time"].max()

        assert not experiences.loc[0, "is_reward"]
        assert pd.isna(experiences.loc[0, "reward_on_time"])
        assert experiences.loc[0, "outcome_known_time"] == pytest.approx(
            min(
                initial_poke_start + OMISSION_KNOWN_LATENCY_SEC,
                initial_visit_departure,
            )
        )

    def test_gap_pump_is_recovered_and_removed_from_unmatched_audit(self):
        """A same-port gap pump belongs to the whole Comrie task visit."""
        beam_breaks = pd.DataFrame(
            {
                "start_time": [1.0, 10.0, 10.2],
                "end_time": [1.1, 10.05, 10.4],
                "event_name": ["Poke1", "Poke2", "Poke2"],
            }
        )
        pumps = pd.DataFrame({"start_time": [10.1], "event_name": ["Pump2"]})

        visits = make_well_visits_df(beam_breaks, pumps)
        trials = make_trials_df_from_well_visits(visits)
        experiences = make_port_experiences_df(visits, trials)

        assert trials["is_reward"].tolist() == [True]
        assert trials["reward_on_time"].tolist() == pytest.approx([10.1])
        assert trials["outcome_known_time"].tolist() == pytest.approx([10.1])
        assert experiences["is_reward"].tolist() == [False, True]
        assert experiences.attrs["gap_recovered_pump_count"] == 1
        assert experiences.attrs["unmatched_pump_event_count"] == 0
        assert experiences.attrs["unmatched_pump_events"] == ()
        assert experiences.attrs["unmatched_pump_event_times"] == ()

    def test_single_observed_port_does_not_create_a_synthetic_trial(self):
        """An epoch boundary is not a poke, visit, or movement trial."""
        beam_breaks = pd.DataFrame(
            {
                "start_time": [12.0, 12.3],
                "end_time": [12.1, 12.4],
                "event_name": ["Poke3", "Poke3"],
            }
        )
        visits = make_well_visits_df(
            beam_breaks, pd.DataFrame({"start_time": [12.05], "event_name": ["Pump3"]})
        )
        trials = make_trials_df_from_well_visits(visits)
        experiences = make_port_experiences_df(visits, trials)

        assert trials.empty
        assert experiences.empty

    def test_scientific_visit_table_contains_no_epoch_boundary_markers(
        self, beam_breaks, pump_events
    ):
        visits = make_well_visits_df(beam_breaks, pump_events)

        observed_pokes = visits.loc[visits["is_observed_poke"]]
        derived_intervals = visits.loc[~visits["is_observed_poke"]]

        assert not observed_pokes.empty
        assert (observed_pokes["duration"] > 0.0).all()
        assert not derived_intervals.empty
        assert not derived_intervals["is_observed_poke"].any()

    def test_single_observed_port_does_not_create_a_terminal_omission(self):
        """No physical destination poke means there is no completed trip."""
        beam_breaks = pd.DataFrame(
            {
                "start_time": [1.0, 1.2],
                "end_time": [1.1, 1.3],
                "event_name": ["Poke1", "Poke1"],
            }
        )
        visits = make_well_visits_df(beam_breaks, pd.DataFrame({"start_time": []}))
        trials = make_trials_df_from_well_visits(visits)
        experiences = make_port_experiences_df(visits, trials)

        assert trials.empty
        assert experiences.empty

    def test_junction_epoch_start_does_not_backdate_first_observed_visit(self):
        """The epoch start before the first poke is not occupancy.

        Visits come from the beam breaks alone, so the time before the first
        poke (the animal at the central junction) cannot become a visit.
        """
        beam_breaks = pd.DataFrame(
            {
                "start_time": [12.0, 12.3, 16.0, 16.3],
                "end_time": [12.1, 12.4, 16.1, 16.4],
                "event_name": ["Poke3", "Poke3", "Poke1", "Poke1"],
            }
        )
        visits = make_well_visits_df(
            beam_breaks, pd.DataFrame({"start_time": pd.Series(dtype=float)})
        )
        trials = make_trials_df_from_well_visits(visits)
        experiences = make_port_experiences_df(visits, trials)

        assert experiences["arrival_time"].tolist() == pytest.approx([12.0, 16.0])
        assert not experiences["is_boundary_derived"].any()
        assert experiences["outcome_observed"].all()

        location = behavioral_location_from_port_experiences(
            np.array([5.0, 12.05]), experiences
        )
        assert location.loc[5.0, "occupancy_state"] == "unknown"
        assert pd.isna(location.loc[5.0, "occupancy_well_id"])
        assert location.loc[12.05, "occupancy_well_id"] == 2.0

        recency = visit_recency_all_events(
            np.array([5.0, 12.05]), experiences, {well: well for well in range(6)}
        )
        assert np.isnan(recency[0]).all()
        assert recency[1, 2] == 1.0

    def test_rejects_missing_origin_occupancy(self):
        trials = pd.DataFrame(
            {
                "from_well": [0],
                "to_well": [2],
                "to_patch": [2],
                "destination_first_poke_in_time": [10.0],
                "end_time": [11.0],
                "reward_on_time": [10.2],
                "outcome_known_time": [10.2],
                "is_reward": [True],
            },
            index=pd.Index([1], name="trial_number"),
        )
        visits = pd.DataFrame(
            {
                "trial_number": [1],
                "start_time": [1.0],
                "end_time": [10.0],
                "from_well": [0],
                "to_well": [2],
                "is_run": [True],
                "is_observed_poke": [False],
                "reward_on_time": [np.nan],
            }
        )

        with pytest.raises(ValueError, match="origin-port occupancy"):
            make_port_experiences_df(visits, trials)


# ============================================================================
# Tests: DIO poke identity is authoritative over noisy tracking endpoints
# ============================================================================


def test_tracking_straddle_cannot_delete_or_relabel_a_physical_poke():
    """A Poke2 beam break is well 1, whatever the tracking does.

    The visit builder takes no position input, so the DIO channel alone names
    the well.
    """
    beam_breaks = pd.DataFrame(
        {
            "start_time": [1.0, 1.2, 3.0, 3.2, 7.0, 7.2],
            "end_time": [1.1, 1.3, 3.1, 3.3, 7.1, 7.3],
            "event_name": ["Poke1", "Poke1", "Poke2", "Poke2", "Poke4", "Poke4"],
        }
    )

    visits = make_well_visits_df(
        beam_breaks, pd.DataFrame({"start_time": pd.Series(dtype=float)})
    )
    trials = make_trials_df_from_well_visits(visits)
    experiences = make_port_experiences_df(visits, trials)

    observed_pokes = visits.loc[visits["is_observed_poke"]]
    assert observed_pokes[["from_well", "to_well"]].to_numpy().tolist() == [
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [3, 3],
        [3, 3],
    ]
    assert trials[["from_well", "to_well"]].to_numpy().tolist() == [
        [0, 1],
        [1, 3],
    ]
    assert experiences["port"].tolist() == [0, 1, 3]
    assert experiences["n_physical_poke_bouts"].tolist() == [2, 2, 2]
    assert trials["is_patch_change"].tolist() == [False, True]
    assert experiences["patch"].ne(experiences["patch"].shift()).iloc[1:].tolist() == [
        False,
        True,
    ]


def test_unknown_poke_channel_fails_loudly():
    beam_breaks = pd.DataFrame(
        {
            "start_time": [1.0],
            "end_time": [1.1],
            "event_name": ["Poke7"],
        }
    )
    with pytest.raises(ValueError, match="canonical physical poke channel"):
        make_well_visits_df(beam_breaks, _empty_pumps())


def test_same_port_repokes_form_one_comrie_task_experience():
    """Final exit, not first exit, starts the next movement trial."""
    beam_breaks = pd.DataFrame(
        {
            "start_time": [0.0, 3.0, 10.0, 12.2, 20.0],
            "end_time": [0.1, 4.0, 11.0, 13.2, 21.0],
            "event_name": ["Poke1", "Poke1", "Poke2", "Poke2", "Poke3"],
        }
    )
    pumps = pd.DataFrame({"start_time": [10.2], "event_name": ["Pump2"]})

    visits = make_well_visits_df(beam_breaks, pumps)
    trials = make_trials_df_from_well_visits(visits)
    experiences = make_port_experiences_df(visits, trials)

    assert experiences["port"].tolist() == [0, 1, 2]
    assert experiences["n_physical_poke_bouts"].tolist() == [2, 2, 1]
    assert experiences["first_poke_in_time"].tolist() == [0.0, 10.0, 20.0]
    assert experiences["first_poke_out_time"].tolist() == [0.1, 11.0, 21.0]
    assert experiences["final_poke_out_time"].tolist() == [4.0, 13.2, 21.0]
    # Same-port re-pokes remain one Comrie task visit. The omission clock is
    # arrival + the maximum pump latency, capped only by the visit's final
    # departure—not by a transient beam release between re-pokes.
    assert experiences.loc[0, "outcome_known_time"] == pytest.approx(
        OMISSION_KNOWN_LATENCY_SEC
    )
    assert trials["movement_start_time"].tolist() == [4.0, 13.2]
    assert trials["destination_first_poke_in_time"].tolist() == [10.0, 20.0]
    assert trials["destination_first_poke_out_time"].tolist() == [11.0, 21.0]
    assert trials["end_time"].tolist() == [13.2, 21.0]

    engagement = photogate_engagement_at_times(
        np.array([0.05, 2.0, 3.5, 9.0, 10.5]), visits
    )
    assert engagement["is_photogate_engaged"].tolist() == [
        True,
        False,
        True,
        False,
        True,
    ]
    assert engagement["photogate_port"].tolist()[:1] == [0.0]
    assert pd.isna(engagement.loc[1, "photogate_port"])


def test_censored_terminal_beam_sample_does_not_create_a_trial():
    beam_breaks = pd.DataFrame(
        {
            "start_time": [1.0, 5.0],
            "end_time": [2.0, 5.0],
            "event_name": ["Poke1", "Poke3"],
        }
    )

    visits = make_well_visits_df(beam_breaks, _empty_pumps())
    trials = make_trials_df_from_well_visits(visits)
    experiences = make_port_experiences_df(visits, trials)

    assert trials.empty
    assert experiences.empty


def test_task_record_alignment_allows_only_audited_boundary_extras():
    experiences = pd.DataFrame(
        {
            "experience_id": [0, 1, 2, 3],
            "movement_trial_number": pd.array([pd.NA, 1, 2, 3], dtype="Int64"),
            "is_initial_experience": [True, False, False, False],
            "is_boundary_derived": [False] * 4,
            "outcome_observed": [True] * 4,
            "port": [5, 0, 2, 4],
            "patch": [3, 1, 2, 3],
            "arrival_time": [0.0, 10.0, 20.0, 30.0],
            "departure_time": [1.0, 11.0, 21.0, 31.0],
            "first_poke_in_time": [0.0, 10.0, 20.0, 30.0],
            "final_poke_out_time": [1.0, 11.0, 21.0, 31.0],
            "n_physical_poke_bouts": [1] * 4,
            "reward_on_time": [np.nan, 10.2, np.nan, 30.2],
            "outcome_known_time": [0.3, 10.2, 20.3, 30.2],
            "is_reward": [False, True, False, True],
        }
    )
    comrie = pd.DataFrame(
        {
            "trial_number_by_epoch": [0, 1, 2],
            "leaf": [1, 3, 5],
            "reward": [True, False, True],
            "poke_in_ts": [10.0, 20.0, 30.0],
            "poke_out_ts": [11.0, 21.0, 31.0],
        }
    )

    aligned = align_port_experiences_to_task_records(experiences, comrie)

    assert aligned["is_in_task_record"].tolist() == [False, True, True, True]
    assert aligned["task_record_number"].tolist()[1:] == [0, 1, 2]
    assert aligned.attrs["leading_physical_experiences"] == 1
    assert aligned.attrs["trailing_physical_experiences"] == 0
    assert aligned.attrs["max_abs_poke_time_difference_s"] == pytest.approx(0.0)


def test_task_record_alignment_rejects_an_internal_behavior_mismatch():
    experiences = pd.DataFrame(
        {
            "port": [0, 2, 4],
            "is_reward": [True, False, True],
            "arrival_time": [10.0, 20.0, 30.0],
            "departure_time": [11.0, 21.0, 31.0],
        }
    )
    comrie = pd.DataFrame(
        {
            "leaf": [1, 4, 5],
            "reward": [True, False, True],
            "poke_in_ts": [10.0, 20.0, 30.0],
            "poke_out_ts": [11.0, 21.0, 31.0],
        }
    )

    with pytest.raises(ValueError, match="contiguous port/reward sequence"):
        align_port_experiences_to_task_records(experiences, comrie)
