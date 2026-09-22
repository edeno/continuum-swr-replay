"""One schema for every detector's events (spec 03, "Common schema")."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.inventory_comparison.schema import (
    INVENTORY_DTYPES,
    SessionIdentity,
    empty_inventory,
    from_classified_nonlocal_table,
    from_memoryless_table,
    inventory_from_intervals,
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

        assert inventory.event_key.tolist() == ["Kay:2", "Kay:1"]
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
        assert inventory.event_key.tolist() == ["state_space:111", "state_space:222"]
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
            }
        )

        inventory = from_memoryless_table(table)

        assert inventory.event_key.tolist() == ["memoryless:42"]
        assert inventory.run_id.tolist() == ["run"]
        assert list(inventory.columns) == list(INVENTORY_DTYPES)
