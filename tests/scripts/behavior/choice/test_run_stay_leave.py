"""Stage 6 reads one stated population, and says which one on every artifact.

The stay/leave models used to pool every immobility stop in the epoch, while
the hippocampal rate and content ladders are read on the well-anchored stops
only. Three things keep the two from drifting apart again:

- the cohort is applied to the stop table *before* the per-stop summary, so an
  out-of-cohort stop cannot reach a fit;
- the canonical at-well bundle keeps the unsuffixed artifact names and the
  sensitivity bundle publishes ``_allstops`` ones, so they cannot be confused
  on disk;
- each fitted frame carries ``stop_cohort`` and each artifact a provenance
  sidecar naming it, so a bundle read months later still states its population.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.events.analysed_time import stamp_swr_source
from continuum_swr_replay.runs.fit_provenance import fit_provenance_path
from continuum_swr_replay.tables.covariate_registry import (
    stamp_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table, write_table
from scripts.behavior.choice import run_stay_leave as runner

SESSION = {"animal": "a", "session_date": 1, "epoch": "02_r1"}

#: 20 at-well + 10 mixed stops are well-anchored; the 10 in-transit stops are
#: the population the cohort exists to remove.
N_AT_WELL = 20
N_MIXED = 10
N_IN_TRANSIT = 10
N_STOPS = N_AT_WELL + N_MIXED + N_IN_TRANSIT


def _write_tables(input_dir):
    """Write a minimal canonical table set with in-transit stops in it."""
    stop_ids = np.arange(N_STOPS)
    states = (
        ["at_well"] * N_AT_WELL + ["mixed"] * N_MIXED + ["in_transit"] * N_IN_TRANSIT
    )
    stop = pd.DataFrame(
        {
            **SESSION,
            "stop_id": stop_ids,
            "occupancy_state": states,
            "occupancy_well_arm": 0.0,
            "prev_arm": 1.0,
            "last_rewarded_arm": 0.0,
            "is_rewarded_outcome": (stop_ids % 2).astype(float),
            "did_last_completed_trip_switch_patch": 0.0,
            "is_stay": (stop_ids % 3 == 0).astype(float),
            "is_swr_stop": True,
            "is_mua_stop": False,
            "n_selected_events": 1,
            "immobility_duration_s": 5.0,
        }
    )
    event = pd.DataFrame(
        {
            **SESSION,
            "event_id": stop_ids,
            "stop_id": stop_ids,
            "occupancy_well_arm": 0.0,
            "prev_arm": 1.0,
            "last_rewarded_arm": 0.0,
            "is_swr": True,
            "is_mua": False,
        }
    )
    # One event per stop, with the mass on the current arm varying across
    # stops so the content predictor is not constant within the session.
    arm = np.tile(np.arange(9), N_STOPS)
    event_id = np.repeat(stop_ids, 9)
    on_current = 0.1 + 0.08 * (event_id % 10)
    fraction = np.where(arm == 0, on_current, (1.0 - on_current) / 8.0)
    event_arm = pd.DataFrame(
        {
            **SESSION,
            "event_id": event_id,
            "stop_id": event_id,
            "candidate_arm": arm,
            "arm_fraction": fraction,
            "occupancy_well_arm": 0.0,
            "prev_arm": 1.0,
            "last_rewarded_arm": 0.0,
            "reward_fraction_candidate_arm_last10_visits": np.where(arm == 0, 1.0, 0.0),
            "is_swr": True,
            "is_mua": False,
        }
    )
    for frame, table in (
        (stop, "stop"),
        (event, "event"),
        (event_arm, "event_arm"),
    ):
        write_table(
            stamp_swr_source(stamp_covariate_schema(frame, table=table)),
            input_dir / f"{table}_table.parquet",
        )


@pytest.fixture
def stage6(replay_content_root):
    """A synthetic table set where the runner reads it, under a test data root."""
    _write_tables(replay_content_root)
    return replay_content_root


def _run(*cli: str) -> None:
    runner.main(["--min-stops", "5", "--rebuild-summary", *cli])


class TestStopCohort:
    def test_at_well_run_excludes_in_transit_stops(self, stage6):
        _run()

        summary = pd.read_pickle(stage6 / "stop_replay_summary.pkl")
        assert len(summary) == N_AT_WELL + N_MIXED
        # The in-transit stops are the last ten stop_ids.
        assert set(summary["stop_id"]) == set(range(N_AT_WELL + N_MIXED))
        assert summary.attrs["stop_cohort"] == "at_well"

    def test_all_stops_run_keeps_every_stop_under_its_own_name(self, stage6):
        _run("--stop-cohort", "all_stops")

        assert not (stage6 / "stop_replay_summary.pkl").exists()
        summary = pd.read_pickle(stage6 / "stop_replay_summary_allstops.pkl")
        assert len(summary) == N_STOPS
        assert (stage6 / "stage6_is_stay_logistic_allstops.pkl").exists()
        assert (stage6 / "stage6_is_stay_pooled_gee_allstops.pkl").exists()
        # The canonical names stay free for the canonical population.
        assert not (stage6 / "stage6_is_stay_logistic.pkl").exists()

    def test_both_cohorts_can_be_published_side_by_side(self, stage6):
        _run()
        _run("--stop-cohort", "all_stops")

        at_well = pd.read_pickle(stage6 / "stage6_is_stay_logistic.pkl")
        all_stops = pd.read_pickle(stage6 / "stage6_is_stay_logistic_allstops.pkl")
        assert set(at_well["stop_cohort"]) == {"at_well"}
        assert set(all_stops["stop_cohort"]) == {"all_stops"}
        assert at_well["n_stops"].max() < all_stops["n_stops"].max()

    def test_stratified_outputs_carry_both_qualifiers(self, stage6):
        _run("--content-stratum", "swr", "--stop-cohort", "all_stops")

        assert (stage6 / "stage6_is_stay_logistic_swr_allstops.pkl").exists()
        assert (stage6 / "stage6_is_stay_pooled_gee_swr_allstops.pkl").exists()

    def test_gee_rows_state_the_cohort(self, stage6):
        _run()

        gee = pd.read_pickle(stage6 / "stage6_is_stay_pooled_gee.pkl")
        assert set(gee["stop_cohort"]) == {"at_well"}


class TestProvenance:
    def test_sidecar_names_the_cohort_and_the_tables(self, stage6):
        _run()

        sidecar = fit_provenance_path(stage6 / "stage6_is_stay_logistic.pkl")
        record = json.loads(sidecar.read_text())
        assert record["stop_cohort"] == "at_well"
        assert record["stop_cohort_report"]["n_stops_after"] == N_AT_WELL + N_MIXED
        assert record["stop_cohort_report"]["n_stops_before"] == N_STOPS
        assert record["stop_table_sha256"] == record["inputs"]["stop_table"]["sha256"]
        assert record["n_stops"] == N_AT_WELL + N_MIXED

    def test_the_cached_summary_carries_its_own_sidecar(self, stage6):
        """The summary is reused across strata, so it states its own population."""
        _run()

        record = json.loads(
            fit_provenance_path(stage6 / "stop_replay_summary.pkl").read_text()
        )
        assert record["estimator"] == "per_stop_summary"
        assert record["stop_cohort"] == "at_well"
        assert record["stop_cohort_report"]["n_stops_after"] == N_AT_WELL + N_MIXED
        assert record["stop_table_sha256"] == record["inputs"]["stop_table"]["sha256"]

    def test_cached_summary_from_another_cohort_is_refused(self, stage6):
        """The canonical name is unsuffixed, so the cache must be checked.

        A summary built on every stop and one built on the well-anchored stops
        would otherwise be interchangeable on disk under that one name.
        """
        _run("--stop-cohort", "all_stops")
        # Stage the all-stops summary under the canonical at-well name, the
        # exact situation a pre-cohort artifact leaves behind.
        summary = pd.read_pickle(stage6 / "stop_replay_summary_allstops.pkl")
        summary.to_pickle(stage6 / "stop_replay_summary.pkl")

        with pytest.raises(ValueError, match="not 'at_well'"):
            runner.main(["--min-stops", "5"])

    def test_cached_summary_that_states_no_cohort_is_refused(self, stage6):
        """A pre-cohort artifact names no population, so it cannot be this one.

        It sits at exactly the canonical unsuffixed path and looks current in
        every other respect, which is the whole reason the attribute is checked
        rather than the file name.
        """
        _run()
        summary = pd.read_pickle(stage6 / "stop_replay_summary.pkl")
        del summary.attrs["stop_cohort"]
        summary.to_pickle(stage6 / "stop_replay_summary.pkl")

        with pytest.raises(ValueError, match="cohort None, not 'at_well'"):
            runner.main(["--min-stops", "5"])

    def test_cached_summary_cut_from_moved_tables_is_refused(self, stage6):
        """The cohort names a population; the hashes name the bytes.

        A regenerated stop table changes which stops the summary holds without
        changing its name or its cohort, so the cache is only usable while the
        tables it read are the tables on disk.
        """
        _run()
        stop_table = read_table(stage6 / "stop_table.parquet")
        regenerated = stop_table.copy()
        regenerated.attrs = dict(stop_table.attrs)
        regenerated["immobility_duration_s"] = 6.0
        write_table(regenerated, stage6 / "stop_table.parquet")

        with pytest.raises(ValueError, match="different generation of stop_table"):
            runner.main(["--min-stops", "5"])

    def test_cached_summary_is_reused_when_the_tables_have_not_moved(self, stage6):
        """The staleness check must not reject a cache that is simply current."""
        _run()
        stamp = (stage6 / "stop_replay_summary.pkl").stat().st_mtime_ns

        runner.main(["--min-stops", "5"])

        assert (stage6 / "stop_replay_summary.pkl").stat().st_mtime_ns == stamp
