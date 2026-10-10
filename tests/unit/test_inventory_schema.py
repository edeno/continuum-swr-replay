"""One schema for every detector's events."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.inventory_comparison.schema import (
    INVENTORY_DTYPES,
    MEMORYLESS_COHORTS,
    SessionIdentity,
    combine_inventories,
    empty_inventory,
    from_classified_nonlocal_table,
    from_memoryless_table,
    inventory_from_intervals,
    validate_inventory,
)

SESSION = SessionIdentity(animal="sim", day=20200101, epoch="02_r1", brain_area="HPC")
GRID = 100.0 + np.arange(2000) / 500.0


def still(n=GRID.size):
    return np.zeros(n)


def swr_inventory(starts, ends, speed=None):
    return inventory_from_intervals(
        np.asarray(starts, dtype=float),
        np.asarray(ends, dtype=float),
        native_keys=np.arange(1, len(starts) + 1),
        detector="Kay",
        session=SESSION,
        run_id="run",
        speed_time_s=GRID,
        speed_cm_per_s=still() if speed is None else speed,
    )


class TestInventoryFromIntervals:
    def test_every_column_with_its_declared_dtype(self):
        inventory = swr_inventory([GRID[10], GRID[100]], [GRID[30], GRID[140]])

        assert list(inventory.columns) == list(INVENTORY_DTYPES)
        for column, dtype in INVENTORY_DTYPES.items():
            assert inventory[column].dtype == dtype, column

    def test_the_span_is_end_minus_start_for_everyone(self):
        inventory = swr_inventory([GRID[10]], [GRID[30]])

        assert inventory.interval_span_s.iloc[0] == GRID[30] - GRID[10]
        assert inventory.event_midpoint_s.iloc[0] == (GRID[10] + GRID[30]) / 2

    def test_without_a_native_duration_the_span_stands_in_for_it(self):
        inventory = swr_inventory([GRID[10]], [GRID[30]])

        assert inventory.native_duration_s.iloc[0] == inventory.interval_span_s.iloc[0]

    def test_keys_are_namespaced_by_detector_and_indexed_in_time_order(self):
        inventory = swr_inventory([GRID[100], GRID[10]], [GRID[140], GRID[30]])

        assert inventory.event_key.tolist() == [
            "Kay:sim20200101_02_r1:2",
            "Kay:sim20200101_02_r1:1",
        ]
        assert inventory.event_start_s.is_monotonic_increasing
        assert inventory.event_index.tolist() == [0, 1]
        assert all(key.startswith("Kay:") for key in inventory.event_key)

    def test_speed_is_read_over_every_sample_of_the_event(self):
        speed = still()
        speed[20] = 9.0
        inventory = swr_inventory([GRID[10], GRID[100]], [GRID[30], GRID[140]], speed)

        assert inventory.whole_event_speed_eligible.tolist() == [False, True]
        assert inventory.animal_max_speed_cm_s.tolist() == [9.0, 0.0]

    def test_an_event_of_no_duration_is_refused(self):
        """It could never overlap anything under the strict rule; keeping it
        quietly would hide whatever made it."""
        with pytest.raises(ValueError, match="duration"):
            swr_inventory([GRID[10]], [GRID[10]])

    def test_duplicate_native_keys_are_refused(self):
        with pytest.raises(ValueError, match="key"):
            inventory_from_intervals(
                np.array([GRID[1], GRID[50]]),
                np.array([GRID[9], GRID[60]]),
                native_keys=np.array([7, 7]),
                detector="Kay",
                session=SESSION,
                run_id="run",
                speed_time_s=GRID,
                speed_cm_per_s=still(),
            )


class TestCombine:
    """Keys must be unique across every epoch compared, or joins mis-match."""

    def test_the_same_native_key_in_two_epochs_gives_two_keys(self):
        other = SessionIdentity(
            animal="sim", day=20200101, epoch="04_r2", brain_area="HPC"
        )
        first = swr_inventory([GRID[10]], [GRID[30]])
        second = inventory_from_intervals(
            np.array([GRID[10]]),
            np.array([GRID[30]]),
            native_keys=np.array([1]),
            detector="Kay",
            session=other,
            run_id="run",
            speed_time_s=GRID,
            speed_cm_per_s=still(),
        )

        eligible, _ = combine_inventories([first, second])

        assert eligible.event_key.tolist() == [
            "Kay:sim20200101_02_r1:1",
            "Kay:sim20200101_04_r2:1",
        ]

    def test_a_repeated_key_is_refused(self):
        inventory = swr_inventory([GRID[10]], [GRID[30]])

        with pytest.raises(ValueError, match="repeats"):
            combine_inventories([inventory, inventory])

    def test_events_during_movement_are_set_apart(self):
        """A movement-spanning event cannot mark an immobile one."""
        speed = still()
        speed[20] = 9.0
        inventory = swr_inventory([GRID[10], GRID[100]], [GRID[30], GRID[140]], speed)

        eligible, moving = combine_inventories([inventory, empty_inventory()])

        assert eligible.event_key.tolist() == ["Kay:sim20200101_02_r1:2"]
        assert moving.event_key.tolist() == ["Kay:sim20200101_02_r1:1"]
        for column, dtype in INVENTORY_DTYPES.items():
            assert eligible[column].dtype == dtype, column
            assert moving[column].dtype == dtype, column


class TestEmpty:
    def test_an_empty_inventory_keeps_every_dtype_through_concat(self):
        """The canonical builder's empty return is all-object; this is not."""
        full = swr_inventory([GRID[10]], [GRID[30]])

        combined = pd.concat([empty_inventory(), full], ignore_index=True)

        for column, dtype in INVENTORY_DTYPES.items():
            assert combined[column].dtype == dtype, column


class TestAdapters:
    def test_our_canonical_table_keeps_its_native_duration_apart(self):
        """Its duration adds one sample width to the span; the span does not."""
        classified = pd.DataFrame(
            {
                "animal": ["sim", "sim"],
                "day": [20200101, 20200101],
                "epoch": ["02_r1", "02_r1"],
                "event_id": [111, 222],
                "event_start": [GRID[10], GRID[100]],
                "event_end": [GRID[30], GRID[140]],
                "event_duration": [
                    GRID[30] - GRID[10] + 0.002,
                    GRID[140] - GRID[100] + 0.002,
                ],
                "animal_max_speed_cm_s": [0.0, 1.0],
                "animal_mean_speed_cm_s": [0.0, 0.5],
                "whole_event_speed_eligible": [True, True],
            }
        )

        inventory = from_classified_nonlocal_table(
            classified, brain_area="HPC", run_id="decode"
        )

        assert (inventory.detector == "state_space").all()
        assert inventory.event_key.tolist() == [
            "state_space:sim20200101_02_r1:111",
            "state_space:sim20200101_02_r1:222",
        ]
        np.testing.assert_allclose(
            inventory.native_duration_s - inventory.interval_span_s, 0.002
        )

    def test_the_memoryless_table_maps_onto_the_same_columns(self):
        table = pd.DataFrame(
            {
                "animal": ["sim"],
                "day": [20200101],
                "epoch": ["02_r1"],
                "brain_area": ["HPC"],
                "run_id": ["run"],
                "ms_event_id": [42],
                "event_start_s": [GRID[10]],
                "event_end_s": [GRID[60]],
                "event_duration_s": [GRID[60] - GRID[10]],
                "animal_max_speed_cm_s": [1.0],
                "animal_mean_speed_cm_s": [0.5],
                "whole_event_speed_eligible": [True],
                "is_replay": [True],
                "shuffle_pass": [True],
                "n_shuffles": [100],
            }
        )

        inventory = from_memoryless_table(table, cohort="published")

        assert inventory.event_key.tolist() == ["memoryless:sim20200101_02_r1:42"]
        assert inventory.run_id.tolist() == ["run"]
        assert list(inventory.columns) == list(INVENTORY_DTYPES)

    def test_the_memoryless_adapter_takes_only_the_named_cohort(self):
        """The memoryless table is one row per candidate, accepted or not;
        passing every row would count the detector's rejections as events."""
        table = pd.DataFrame(
            {
                "animal": ["sim"] * 3,
                "day": [20200101] * 3,
                "epoch": ["02_r1"] * 3,
                "brain_area": ["HPC"] * 3,
                "run_id": ["run"] * 3,
                "ms_event_id": [1, 2, 3],
                "event_start_s": [GRID[10], GRID[100], GRID[200]],
                "event_end_s": [GRID[60], GRID[160], GRID[260]],
                "event_duration_s": [0.1, 0.12, 0.12],
                "animal_max_speed_cm_s": [1.0] * 3,
                "animal_mean_speed_cm_s": [0.5] * 3,
                "whole_event_speed_eligible": [True] * 3,
                "is_replay": [True, True, False],
                "shuffle_pass": [True, False, False],
                "n_shuffles": [100, 100, 0],
            }
        )

        published = from_memoryless_table(table, cohort="published")
        criteria = from_memoryless_table(table, cohort="replay_criteria")

        assert published.event_key.tolist() == ["memoryless:sim20200101_02_r1:1"]
        assert criteria.event_key.tolist() == [
            "memoryless:sim20200101_02_r1:1",
            "memoryless:sim20200101_02_r1:2",
        ]
        with pytest.raises(ValueError, match="cohort"):
            from_memoryless_table(table, cohort="everything")
        with pytest.raises(ValueError, match="lacks"):
            from_memoryless_table(
                table.drop(columns="shuffle_pass"), cohort="published"
            )

    def test_a_candidate_the_artifact_mask_reaches_is_in_no_cohort(self):
        """Its row keeps its reason in the detector's table; no cohort reads it."""
        table = pd.DataFrame(
            {
                "animal": ["sim"] * 3,
                "day": [20200101] * 3,
                "epoch": ["02_r1"] * 3,
                "brain_area": ["HPC"] * 3,
                "run_id": ["run"] * 3,
                "ms_event_id": [1, 2, 3],
                "event_start_s": [GRID[10], GRID[100], GRID[200]],
                "event_end_s": [GRID[60], GRID[160], GRID[260]],
                "event_duration_s": [0.1, 0.12, 0.12],
                "animal_max_speed_cm_s": [1.0] * 3,
                "animal_mean_speed_cm_s": [1.0] * 3,
                "whole_event_speed_eligible": [True] * 3,
                "is_replay": [True] * 3,
                "shuffle_pass": [True] * 3,
                "n_shuffles": [100] * 3,
                "artifact_drop_reason": [None, "near_artifact", "artifact"],
            }
        )

        for cohort in MEMORYLESS_COHORTS:
            inventory = from_memoryless_table(table, cohort=cohort)
            assert inventory.event_key.tolist() == ["memoryless:sim20200101_02_r1:1"]

    def test_a_table_made_before_the_rule_is_refused_under_an_artifact_run(
        self, monkeypatch
    ):
        """It would read its candidates in excluded time as kept."""
        from continuum_swr_replay.analysis.hippocampus import swr_source
        from tests.unit.test_memoryless_event_table import build, make_recording

        table = build(make_recording()).drop(columns="artifact_drop_reason")
        assert len(from_memoryless_table(table, cohort="replay_criteria")) == 2
        monkeypatch.setattr(swr_source, "ARTIFACT_INTERVALS_RUN_ID", "a_run")

        with pytest.raises(ValueError, match="artifact_drop_reason"):
            from_memoryless_table(table, cohort="replay_criteria")

    def test_the_adapter_reads_what_the_memoryless_builder_writes(self):
        """Fed a real table, so a column rename in the builder fails here and
        not in the runner."""
        from tests.unit.test_memoryless_event_table import build, make_recording

        table = build(make_recording())

        inventory = from_memoryless_table(table, cohort="replay_criteria")

        assert len(inventory) == int(table.is_replay.sum()) == 2
        np.testing.assert_allclose(
            inventory.event_start_s, table[table.is_replay].event_start_s
        )
        assert inventory.whole_event_speed_eligible.tolist() == [True, False]

    def test_the_published_cohort_of_an_unshuffled_run_is_refused(self):
        """A run made without the shuffle passes nothing, and would read as a
        detector that found no replay."""
        from tests.unit.test_memoryless_event_table import build, make_recording

        recording = make_recording()
        unshuffled = build(recording, shuffle_results=None)

        assert len(from_memoryless_table(build(recording), cohort="published")) == 2
        assert len(from_memoryless_table(unshuffled, cohort="replay_criteria")) == 2
        with pytest.raises(ValueError, match="shuffle"):
            from_memoryless_table(unshuffled, cohort="published")


class TestInvariants:
    """What the schema refuses, so no count is silently wrong downstream."""

    def test_an_identity_whose_parts_are_malformed_is_refused(self):
        with pytest.raises(ValueError, match="brain_area"):
            SessionIdentity(animal="sim", day=20200101, epoch="02_r1", brain_area="hpc")
        with pytest.raises(TypeError, match="day"):
            SessionIdentity(
                animal="sim", day="20200101", epoch="02_r1", brain_area="HPC"
            )  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="day"):
            SessionIdentity(animal="sim", day=0, epoch="02_r1", brain_area="HPC")
        with pytest.raises(ValueError, match="animal"):
            SessionIdentity(animal=" ", day=20200101, epoch="02_r1", brain_area="HPC")

    def test_overlapping_events_of_one_detector_are_refused(self):
        """One detector's events must be disjoint for the recount's gap-zero
        identity and for each event to count once."""
        with pytest.raises(ValueError, match="overlap"):
            swr_inventory([GRID[10], GRID[20]], [GRID[30], GRID[40]])

    def test_a_missing_session_value_is_refused(self):
        """Every grouped count would drop the event without a word."""
        inventory = swr_inventory([GRID[10]], [GRID[30]])
        inventory["epoch"] = None

        with pytest.raises(ValueError, match="animal, day, epoch"):
            validate_inventory(inventory)

    def test_a_missing_eligibility_flag_is_refused(self):
        """astype(bool) would turn NaN into True and admit the event."""
        inventory = swr_inventory([GRID[10]], [GRID[30]]).astype(
            {"whole_event_speed_eligible": object}
        )
        inventory.loc[0, "whole_event_speed_eligible"] = np.nan

        with pytest.raises(ValueError, match="whole_event_speed_eligible"):
            validate_inventory(inventory)

    def test_a_key_that_does_not_name_its_own_detector_and_epoch_is_refused(self):
        inventory = swr_inventory([GRID[10]], [GRID[30]])
        inventory["detector"] = "other"

        with pytest.raises(ValueError, match="does not name"):
            validate_inventory(inventory)

    def test_a_detector_name_holding_the_key_separator_is_refused(self):
        with pytest.raises(ValueError, match="':'"):
            inventory_from_intervals(
                np.array([GRID[10]]),
                np.array([GRID[30]]),
                native_keys=np.array([0]),
                detector="swr:kay",
                session=SESSION,
                run_id="run",
                speed_time_s=GRID,
                speed_cm_per_s=np.zeros(GRID.size),
            )

    def test_two_brain_areas_in_one_epoch_are_refused(self):
        """The comparison keys on the epoch and would pair events across areas."""
        pfc = SessionIdentity(
            animal="sim", day=20200101, epoch="02_r1", brain_area="mPFC"
        )
        hpc = swr_inventory([GRID[10]], [GRID[30]])
        other = inventory_from_intervals(
            np.array([GRID[50]]),
            np.array([GRID[70]]),
            native_keys=np.array([0]),
            detector="mua",
            session=pfc,
            run_id="run",
            speed_time_s=GRID,
            speed_cm_per_s=np.zeros(GRID.size),
        )

        with pytest.raises(ValueError, match="brain area"):
            combine_inventories([hpc, other])

    def test_an_unsorted_speed_trace_is_refused(self):
        """searchsorted on it would read NaN speeds and mark every event moving."""
        with pytest.raises(ValueError, match="increasing"):
            inventory_from_intervals(
                np.array([GRID[10]]),
                np.array([GRID[30]]),
                native_keys=np.array([0]),
                detector="swr",
                session=SESSION,
                run_id="run",
                speed_time_s=GRID[::-1],
                speed_cm_per_s=np.zeros(GRID.size),
            )
