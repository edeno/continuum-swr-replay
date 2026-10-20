"""The comparison runner end to end, on simulated epochs written to disk.

Every input is written to a temporary folder as its runner writes it: the
decode frame's pickle, the SWR run (Kay, Karlsson, Yu, Zugaro), a Carey run, a
memoryless run and an artifact run. The runner reads them through the adapters
and writes its tables, which the tests read back. Nothing here reaches
Spyglass or ``Processed-Data``.

Epoch ``02_r1`` is the worked epoch of ``test_inventory_comparison_end_to_end``
(0-60 s on a 500 Hz grid): the animal moves over 20-25 s, the posterior is
missing over 40-41 s, the stored band over 50-52 s, and the memoryless decoded
windows are [0.04, 39.962] and [41.04, 59.962]. The other five detectors
observe the whole epoch and find nothing, so the hand-worked exposures hold
for every comparison. Epoch ``04_r2`` repeats it with an artifact core at
30.05 s, which sets Kay's SWR at 30.0 s aside, and no memoryless record; its
Karlsson event at 29.5-29.7 s reaches excluded time and is set aside, and the
Kay SWR at 29.52-29.58 s that it overlaps is kept. Two more epochs fail: in
``06_r3`` the SWR run cut its band to a position span other than the decode
grid's, and ``08_r4`` has no decode.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.detection.run_inventory_comparison as runner
from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.analysis.events import analysed_time, swr_inventory
from continuum_swr_replay.analysis.inventory_comparison.availability import (
    coverage_from_observations,
    still_intervals,
    stored_coverage,
)
from continuum_swr_replay.analysis.inventory_comparison.cells import (
    membership_against,
)
from continuum_swr_replay.analysis.inventory_comparison.overlap import (
    PAIR_DTYPES,
    directional_agreement,
    event_matches,
)
from continuum_swr_replay.analysis.inventory_comparison.populations import (
    rebuild_matches,
)
from continuum_swr_replay.intervals import (
    intersect,
    total_duration,
)
from continuum_swr_replay.runs.io import sha256_file
from continuum_swr_replay.tables.io import read_table
from tests.unit._swr_inventory_fixtures import (
    kay_events,
    use_swr_source,
    write_artifact_run,
    write_swr_inventory,
)

pytestmark = [pytest.mark.slow, pytest.mark.integration]

ANIMAL, DAY = "sim", 20200101
EPOCHS = {"02_r1": "sim20200101_02_r1_HPC", "04_r2": "sim20200101_04_r2_HPC"}
WORKED, MASKED = EPOCHS["02_r1"], EPOCHS["04_r2"]
SHIFTED, NO_DECODE = "sim20200101_06_r3_HPC", "sim20200101_08_r4_HPC"
TIME = np.round(np.arange(30_001) * 0.002, 6)
SPEED = np.where((TIME >= 20.0) & (TIME < 25.0), 10.0, 1.0)

NON_LOCAL = [(5.0, 5.3), (10.0, 10.2), (41.1, 41.15), (51.0, 51.2)]
"""The state-space events, all still: the decoder labels only events that are
immobile as a whole, and the read (``analysed_time.with_swr_source``) refuses a
moving ``non_local_event`` label as a stale frame. The moving event is Kay's."""
KAY = [(5.05, 5.1), (21.0, 21.05), (30.0, 30.1)]
"""Kay's SWRs; the one at 21.0 s is during the movement bout, so it stays in
Kay's inventory flagged moving (``whole_event_speed_eligible`` False)."""
MASKED_KAY = [(5.05, 5.1), (29.52, 29.58), (30.0, 30.1)]
MASKED_KARLSSON = [(29.5, 29.7)]
STORED_BAND = [(0.0, 50.0), (52.0, 60.0)]
DECODED_WINDOWS = [(0.04, 39.962), (41.04, 59.962)]
MEMORYLESS = [(5.1, 5.4), (41.0, 41.2), (12.0, 12.1)]
"""Two published sequences (the second bridges a gap in the decoded windows)
and one meeting the replay criteria alone."""
CAREY_RUN, MEMORYLESS_RUN, RUN_ID = "careyrun01", "memorylessrun01", "comparison01"


def _labels(intervals) -> np.ndarray:
    labels = np.zeros(TIME.size, dtype=np.int64)
    for number, (start, end) in enumerate(intervals, start=1):
        labels[(TIME >= start) & (TIME <= end)] = number
    return labels


def write_decode(root: Path, epoch: str, identifier: str) -> None:
    """The decode's ``data_over_time`` pickle, as the decoder saves it."""
    continuous = np.full(TIME.size, 0.9)
    continuous[(TIME > 40.0) & (TIME < 41.0)] = np.nan
    pd.DataFrame(
        {
            "animal": ANIMAL,
            "day": DAY,
            "epoch": epoch,
            "head_speed": SPEED,
            "SWR_event": 0,
            "MUA_event": 0,
            "non_local_event": _labels(NON_LOCAL),
            "Non-Local_Continuous_prob": continuous,
            "Non-Local_Fragmented_prob": 0.05,
        },
        index=pd.Index(TIME, name="time"),
    ).to_pickle(root / f"{identifier}_80_data_over_time.pkl")


def add_swr_detector(
    run_dir: Path, identifier: str, detector: str, intervals=()
) -> None:
    """An SWR detector's events, in the run folder and its manifest."""
    path = run_dir / f"{identifier}_{detector}_events.parquet"
    kay_events(list(intervals)).to_parquet(path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["epochs"][identifier]["outputs"][detector] = {
        "path": str(path),
        "sha256": sha256_file(path),
    }
    manifest_path.write_text(json.dumps(manifest))


def write_carey_run(root: Path) -> None:
    """Carey ran on both epochs, observed all of each and found nothing."""
    directory = root / CAREY_RUN
    directory.mkdir(parents=True)
    records = {}
    for identifier in EPOCHS.values():
        events = directory / f"{identifier}_carey_events.parquet"
        kay_events([]).assign(n_active_units=np.array([], dtype=int)).to_parquet(events)
        observed = directory / f"{identifier}_carey_observed.parquet"
        pd.DataFrame(
            {"scope": ["carey"], "start_s": [0.0], "end_s": [60.0]}
        ).to_parquet(observed)
        records[identifier] = {
            "status": "ok",
            "outputs": {
                "events": {"path": str(events), "sha256": sha256_file(events)},
                "observed": {"path": str(observed), "sha256": sha256_file(observed)},
            },
        }
    (directory / "manifest.json").write_text(
        json.dumps({"run_id": CAREY_RUN, "epochs": records})
    )


def write_memoryless_run(root: Path) -> None:
    """The memoryless run, with a record for the worked epoch only."""
    directory = root / MEMORYLESS_RUN
    directory.mkdir(parents=True)
    starts = [start for start, _ in MEMORYLESS]
    ends = [end for _, end in MEMORYLESS]
    table = pd.DataFrame(
        {
            "animal": ANIMAL,
            "day": DAY,
            "epoch": "02_r1",
            "brain_area": "HPC",
            "run_id": MEMORYLESS_RUN,
            "ms_event_id": [0, 1, 2],
            "event_start_s": starts,
            "event_end_s": ends,
            "event_duration_s": np.subtract(ends, starts),
            "animal_max_speed_cm_s": 1.0,
            "animal_mean_speed_cm_s": 1.0,
            "whole_event_speed_eligible": True,
            "is_replay": True,
            "shuffle_pass": [True, True, False],
            "n_shuffles": 100,
            "swr_definition": swr_inventory.swr_source_name(),
            "artifact_drop_reason": [None, None, None],
        }
    )
    events = directory / f"{WORKED}_memoryless_events.parquet"
    table.to_parquet(events)
    observed = directory / f"{WORKED}_memoryless_observed.parquet"
    pd.DataFrame(
        {
            "kind": ["decoded_windows"] * 2 + ["swr_labels"],
            "start_s": [start for start, _ in DECODED_WINDOWS] + [0.0],
            "end_s": [end for _, end in DECODED_WINDOWS] + [60.0],
        }
    ).to_parquet(observed)
    record = {
        "status": "ok",
        "time_grid": {
            "start_s": float(TIME[0]),
            "end_s": float(TIME[-1]),
            "n_samples": int(TIME.size),
        },
        "n_candidates": len(table),
        "n_replay": int(table.is_replay.sum()),
        "n_shuffle_pass": int(table.shuffle_pass.sum()),
        "outputs": {"events": str(events), "observed": str(observed)},
    }
    (directory / "manifest.json").write_text(
        json.dumps({"run_id": MEMORYLESS_RUN, "epochs": {WORKED: record}})
    )


@pytest.fixture(scope="module")
def module_monkeypatch():
    with pytest.MonkeyPatch.context() as patch:
        yield patch


@pytest.fixture(scope="module")
def inputs(tmp_path_factory, module_monkeypatch):
    """Every input on disk, and the runner's roots pointed at them."""
    tmp_path = tmp_path_factory.mktemp("comparison")
    monkeypatch = module_monkeypatch
    swr_run = write_swr_inventory(
        tmp_path / "swr",
        {WORKED: KAY, MASKED: MASKED_KAY, SHIFTED: KAY},
        observed=dict.fromkeys([*EPOCHS.values(), SHIFTED], STORED_BAND),
    )
    # the span each epoch's band was cut to: the grid's, but for SHIFTED
    manifest = json.loads((swr_run / "manifest.json").read_text())
    for identifier, record in manifest["epochs"].items():
        record["position_span_s"] = [0.0, 59.0 if identifier == SHIFTED else 60.0]
    (swr_run / "manifest.json").write_text(json.dumps(manifest))
    for identifier in EPOCHS.values():
        for detector in ("karlsson", "yu", "zugaro"):
            intervals = (
                MASKED_KARLSSON
                if (identifier, detector) == (MASKED, "karlsson")
                else ()
            )
            add_swr_detector(swr_run, identifier, detector, intervals)
    artifact_run = write_artifact_run(
        tmp_path / "artifacts",
        {WORKED: [], MASKED: [("spk", 30.05, 30.05)], SHIFTED: []},
    )
    use_swr_source(monkeypatch, swr_run, artifact_run)
    decode_root = tmp_path / "decodes"
    decode_root.mkdir()
    for epoch, identifier in [*EPOCHS.items(), ("06_r3", SHIFTED)]:
        write_decode(decode_root, epoch, identifier)
    write_carey_run(tmp_path / "carey")
    write_memoryless_run(tmp_path / "memoryless")
    monkeypatch.setattr(runner, "DECODE_ROOT", decode_root)
    monkeypatch.setattr(runner, "CAREY_ROOT", tmp_path / "carey")
    monkeypatch.setattr(runner, "MEMORYLESS_ROOT", tmp_path / "memoryless")
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "inventory")
    monkeypatch.setattr(
        runner, "analysed_epoch_ids", lambda: [*EPOCHS.values(), SHIFTED, NO_DECODE]
    )
    return tmp_path


ARGS = [
    "--carey-run-id",
    CAREY_RUN,
    "--memoryless-run-id",
    MEMORYLESS_RUN,
    "--run-id",
    RUN_ID,
]


@pytest.fixture(scope="module")
def run(inputs):
    """The runner over the four epochs, two failing; its run folder."""
    with pytest.raises(SystemExit, match="2 epochs failed"):
        runner.main([*ARGS, "--animal", ANIMAL])
    return inputs / "inventory" / RUN_ID


def table(run_dir: Path, identifier: str, name: str) -> pd.DataFrame:
    return read_table(run_dir / f"{identifier}_{name}.parquet")


def at(frame: pd.DataFrame, start_column: str, start: float) -> pd.Series:
    """The one row of `frame` whose `start_column` is `start`."""
    rows = frame[np.isclose(frame[start_column], start)]
    assert len(rows) == 1
    return rows.iloc[0]


SHARED = (20.0 - 0.04) + (39.962 - 24.998) + (50.0 - 41.2) + (59.962 - 52.0)
"""The hand-worked shared exposure of state space, memoryless and Kay."""


class TestTheWorkedEpoch:
    def test_each_detectors_counts_and_own_exposure(self, run):
        counts = table(run, WORKED, "counts").set_index("detector")

        # No moving state-space event: the decoder never labels one, and the
        # read refuses a frame that has one (it was [5, 1, 4] when this fixture
        # carried a moving non-local label at 21.0 s; that event is now Kay's).
        assert counts.loc[
            "state_space", ["n_native", "n_moving", "n_eligible"]
        ].tolist() == [4, 0, 4]
        assert counts.loc["kay", ["n_native", "n_moving", "n_eligible"]].tolist() == [
            3,
            1,
            2,
        ]
        assert counts.loc[
            "memoryless", ["n_native", "n_set_aside", "n_eligible"]
        ].tolist() == [2, 1, 1]
        assert counts.loc["memoryless_criteria", "n_native"] == 3
        assert (
            counts.n_native.tolist()
            == (counts.n_eligible + counts.n_moving + counts.n_set_aside).tolist()
        )
        # still [0, 20.0] and [24.998, 60], each observed span within it
        exposures = counts.exposure_s.to_dict()
        assert exposures["state_space"] == pytest.approx(20.0 + 15.002 + 19.0)
        assert exposures["memoryless"] == pytest.approx(19.96 + 14.964 + 18.762)
        assert exposures["kay"] == pytest.approx(20.0 + 25.002 + 8.0)
        assert exposures["mua"] == pytest.approx(20.0 + 35.002)
        assert counts.loc["state_space", "rate_per_s"] == pytest.approx(
            4 / exposures["state_space"]
        )
        assert counts.loc["mua", "rate_per_s"] == 0.0
        assert (
            counts.loc["mua", "native_note"]
            and counts.loc["state_space", "native_note"]
        )
        assert counts.loc["kay", "native_note"] == ""

    def test_the_moving_event_stays_in_the_inventory_without_a_component(self, run):
        inventory = table(run, WORKED, "inventory")

        moving = at(inventory, "event_start_s", 21.0)
        assert moving.detector == "kay"
        assert not moving.whole_event_speed_eligible
        assert pd.isna(moving.component_id)
        # the components leave the criteria twin's two kept events out
        twin = inventory[inventory.detector == "memoryless_criteria"]
        assert len(twin) == 2 and twin.whole_event_speed_eligible.all()
        assert twin.component_id.isna().all()
        assert inventory.component_id.notna().sum() == len(inventory) - 1 - 2

    def test_the_bridging_sequence_is_set_aside(self, run):
        set_aside = table(run, WORKED, "set_aside")

        assert set(set_aside.detector) == {"memoryless", "memoryless_criteria"}
        assert set(set_aside.reason) == {"bridging"}
        assert np.allclose(set_aside.interval_start_s, 41.0)
        assert np.allclose(set_aside.interval_end_s, 41.2)

    def test_the_directional_denominators(self, run):
        directional = table(run, WORKED, "directional").set_index(
            ["source_detector", "target_detector"]
        )

        ss_to_ms = directional.loc[("state_space", "memoryless")]
        assert (ss_to_ms.n_source_events, ss_to_ms.n_eligible, ss_to_ms.n_matched) == (
            4,
            3,
            1,
        )
        ms_to_ss = directional.loc[("memoryless", "state_space")]
        assert (ms_to_ss.n_eligible, ms_to_ss.n_matched) == (1, 1)
        assert len(directional) == 9 * 8

    def test_the_three_cells(self, run):
        cells = table(run, WORKED, "cells")
        inventory = table(run, WORKED, "inventory").set_index("event_key")
        cells["start"] = inventory.event_start_s.loc[cells.event_key].to_numpy()

        def membership(cell):
            rows = cells[cells.cell == cell]
            return {
                (detector, round(start, 3)): value
                for detector, start, value in zip(
                    rows.detector, rows.start, rows.membership, strict=True
                )
            }

        assert membership("exactly_one_detector") == {
            ("state_space", 5.0): "matched",
            ("state_space", 10.0): "unmatched",
            ("state_space", 41.1): "unknown",
            ("state_space", 51.0): "unknown",
            ("memoryless", 5.1): "matched",
            ("kay", 5.05): "matched",
            ("kay", 30.0): "unmatched",
        }
        assert membership("swr_matched_by_no_decoder") == {
            ("kay", 5.05): "matched",
            ("kay", 30.0): "unmatched",
        }
        assert membership("sequence_matched_by_no_trigger") == {
            ("state_space", 5.0): "matched",
            ("state_space", 10.0): "unmatched",
            ("state_space", 41.1): "unmatched",
            ("state_space", 51.0): "unknown",
            # Kay's SWR at 5.05-5.1 s only touches it
            ("memoryless", 5.1): "unmatched",
            ("memoryless_criteria", 5.1): "unmatched",
            ("memoryless_criteria", 12.0): "unmatched",
        }

    def test_no_detector_time_is_the_hand_worked_shared_exposure(self, run):
        row = table(run, WORKED, "no_detector_time").iloc[0]

        assert row.shared_exposure_s == pytest.approx(SHARED)
        assert row.detected_s == pytest.approx(0.7)

    def test_excluded_time_by_reason(self, run):
        exclusion = table(run, WORKED, "exclusion_by_reason")

        totals = exclusion.groupby("comparison").seconds.sum()
        assert np.allclose(totals, 60.0)
        everyone = exclusion[exclusion.comparison == "all"].set_index("reason")
        # Karlsson misses the stored band's gap first, by name order
        assert everyone.seconds.to_dict() == pytest.approx(
            {
                "shared_exposure": SHARED,
                "karlsson: lfp_not_observed": 2.0,
                "memoryless: no_decoded_window": 0.04 + (41.04 - 39.962) + 0.038,
                "memoryless: bridging": 41.2 - 41.04,
                "moving": 24.998 - 20.0,
            }
        )
        pair = exclusion[exclusion.comparison == "kay+state_space"].set_index("reason")
        assert pair.seconds["state_space: no_posterior_or_in_artifact_mask"] == (
            pytest.approx(1.0)
        )
        n_pairs = 9 * 8 // 2
        assert exclusion.comparison.nunique() == n_pairs + 1 + 3

    def test_the_swr_matrix_and_its_paired_labels(self, run):
        matrix = table(run, WORKED, "swr_matrix").set_index(
            ["cohort", "swr_definition"]
        )
        paired = table(run, WORKED, "paired_labels")

        ss = matrix.loc[("state_space", "kay")]
        assert (ss.n_events, ss.n_eligible, ss.n_swr, ss.n_non_swr) == (4, 3, 1, 2)
        assert set(matrix.index.get_level_values("swr_definition")) == {
            "kay",
            "karlsson",
            "yu",
            "zugaro",
            "carey",
        }
        row = paired[
            (paired.detector_a == "memoryless")
            & (paired.detector_b == "state_space")
            & (paired.swr_definition == "kay")
        ].iloc[0]
        assert (row.label_a, row.label_b, row.agreement) == (
            "non-SWR",
            "SWR",
            "disagree",
        )

    def test_time_coverage_and_components(self, run):
        covered = table(run, WORKED, "time_coverage").set_index(
            ["detector_a", "detector_b"]
        )
        components = table(run, WORKED, "components")

        ss_k = covered.loc[("kay", "state_space")]
        assert ss_k.union_s == pytest.approx(0.3 + 0.2 + 0.05 + 0.1)
        assert ss_k.intersection_s == pytest.approx(0.05)
        joined = at(components, "span_start_s", 5.0)
        assert joined.detectors == "kay+memoryless+state_space"
        # Kay's SWR only touches the memoryless sequence: two pairs
        assert (joined.n_events, joined.n_pairs) == (3, 2)
        assert joined.span_end_s == 5.4
        # the criteria twin is in no component, and its event at 12.0 s,
        # overlapping nothing, is not one of its own
        assert not components.detectors.str.contains("memoryless_criteria").any()
        assert not np.isclose(components.span_start_s, 12.0).any()

    def test_the_recount_at_no_gap_is_the_native_count(self, run):
        recount = table(run, WORKED, "recount_directional")
        directional = table(run, WORKED, "directional")

        assert sorted(recount.gap_s.unique()) == [0.0, 0.01, 0.02, 0.05]
        native = recount[recount.gap_s == 0.0].drop(columns="gap_s")
        pd.testing.assert_frame_equal(
            native.reset_index(drop=True), directional.reset_index(drop=True)
        )


class TestAnEpochWithoutAMemorylessRecord:
    def test_memoryless_is_unavailable_in_every_table(self, run):
        observations = table(run, MASKED, "observations")
        directional = table(run, MASKED, "directional")
        matrix = table(run, MASKED, "swr_matrix")
        counts = table(run, MASKED, "counts").set_index("detector")

        for detector in ("memoryless", "memoryless_criteria"):
            rows = observations[observations.detector == detector]
            assert len(rows) == 1 and not rows.available.any()
            assert "not in" in rows.reason.iloc[0]
            touching = directional[
                (directional.source_detector == detector)
                | (directional.target_detector == detector)
            ]
            assert len(touching) == 2 * 8 and not touching.available.any()
            assert counts.loc[detector, "exposure_s"] == 0.0
            assert np.isnan(counts.loc[detector, "rate_per_s"])
        assert not matrix[matrix.cohort.str.startswith("memoryless")].available.any()

    def test_its_cells_are_unknown_not_unmatched(self, run):
        cells = table(run, MASKED, "cells")

        for cell in ("exactly_one_detector", "swr_matched_by_no_decoder"):
            assert "unmatched" not in set(cells[cells.cell == cell].membership)
        # the sequences' cell asks only the triggered inventories
        sequences = cells[cells.cell == "sequence_matched_by_no_trigger"]
        assert "unmatched" in set(sequences.membership)

    def test_the_swr_in_the_artifact_mask_is_set_aside(self, run):
        set_aside = table(run, MASKED, "set_aside")
        exclusion = table(run, MASKED, "exclusion_by_reason")

        kay = set_aside[set_aside.detector == "kay"]
        assert kay.reason.tolist() == ["artifact"]
        assert (kay.interval_start_s.iloc[0], kay.interval_end_s.iloc[0]) == (
            30.0,
            30.1,
        )
        assert np.allclose(exclusion.groupby("comparison").seconds.sum(), 60.0)
        assert "excluded_time" in set(exclusion.reason)

    def test_a_set_aside_events_span_is_unobserved_by_its_own_detector(self, run):
        """Karlsson's event at 29.5-29.7 s is set aside, so Kay's SWR inside its
        span leaves Kay's denominator against Karlsson: not observed by
        Karlsson, rather than missed."""
        set_aside = table(run, MASKED, "set_aside")
        directional = table(run, MASKED, "directional").set_index(
            ["source_detector", "target_detector"]
        )

        karlsson = set_aside[set_aside.detector == "karlsson"]
        assert karlsson.reason.tolist() == ["near_artifact"]
        kay = directional.loc[("kay", "karlsson")]
        assert (kay.n_source_events, kay.n_eligible, kay.n_matched) == (2, 1, 0)


@pytest.mark.parametrize("identifier", [WORKED, MASKED], scope="class")
class TestTheStoredTablesRebuildTheComparison:
    """The coverage and memberships rebuilt from the written tables are the
    ones ``compare_epoch`` built, exactly: the worked epoch holds a bridging
    sequence, the masked one events the rule dropped and no memoryless run."""

    @pytest.fixture(scope="class")
    def built(self, run, identifier):
        """``compare_epoch`` run again on the epoch, with what it passed to
        ``event_matches`` and what that returned; the epoch's still
        intervals, from the frame as the runner reads its speed."""
        seen = {}

        def spy(kept, pairs, coverage, **kwargs):
            seen.update(kept=kept, coverage=coverage)
            seen["matches"] = event_matches(kept, pairs, coverage, **kwargs)
            return seen["matches"]

        saved = pd.read_pickle(runner.decode_path(identifier))
        dot = analysed_time.with_swr_source(saved)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(runner, "event_matches", spy)
            runner.compare_epoch(
                dot,
                analysed_time.artifact_event_drops(saved),
                decode_run_id="decode",
                carey_dir=runner.CAREY_ROOT / CAREY_RUN,
                memoryless_dir=runner.MEMORYLESS_ROOT / MEMORYLESS_RUN,
            )
        time, speed = runner._speed_trace(dot)
        key = (ANIMAL, DAY, dot.epoch.iloc[0])
        still = {
            key: still_intervals(
                time, speed, DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC, runner.MAX_GAP_S
            )
        }
        return seen, still

    @staticmethod
    def stored(run, identifier):
        names = ("observations", "set_aside", "inventory", "pairs")
        return {
            name: table(run, identifier, name)
            for name in (*names, "directional", "cells")
        }

    def test_the_coverage_is_the_one_compare_epoch_built(self, run, identifier, built):
        seen, still = built
        tables = self.stored(run, identifier)

        rebuilt = stored_coverage(tables["observations"], tables["set_aside"], still)

        assert set(rebuilt) == set(seen["coverage"])
        for key, coverage in seen["coverage"].items():
            np.testing.assert_array_equal(rebuilt[key].observed, coverage.observed)
            np.testing.assert_array_equal(rebuilt[key].exposure, coverage.exposure)

    def test_the_memberships_reproduce_the_stored_directional_and_cells(
        self, run, identifier, built
    ):
        seen, still = built
        tables = self.stored(run, identifier)
        inventory = tables["inventory"]
        compared = inventory[inventory.whole_event_speed_eligible]
        coverage = stored_coverage(tables["observations"], tables["set_aside"], still)
        pairs = tables["pairs"][list(PAIR_DTYPES)]

        directional = directional_agreement(
            compared, pairs, coverage, detectors=runner.DETECTORS
        )
        _, matches = rebuild_matches(
            tables, still, detectors=runner.DETECTORS, cells=runner.CELLS
        )
        cells = pd.concat(
            [
                membership_against(
                    compared, matches, sources=sources, against=against
                ).assign(cell=name)
                for name, (sources, against) in runner.CELLS.items()
            ],
            ignore_index=True,
        )[list(runner.TABLE_DTYPES["cells"])]

        pd.testing.assert_frame_equal(directional, tables["directional"])
        order = ["cell", "event_key"]
        pd.testing.assert_frame_equal(
            cells.sort_values(order, ignore_index=True),
            tables["cells"].sort_values(order, ignore_index=True),
        )
        order = ["event_key", "other_detector"]
        pd.testing.assert_frame_equal(
            matches.sort_values(order, ignore_index=True),
            seen["matches"].sort_values(order, ignore_index=True),
        )

    def test_a_set_aside_span_is_unobserved_by_its_own_detector_only(
        self, run, identifier, built
    ):
        _, still = built
        tables = self.stored(run, identifier)
        set_aside = tables["set_aside"]

        rebuilt = stored_coverage(tables["observations"], set_aside, still)
        before = coverage_from_observations(tables["observations"], still)

        assert len(set_aside)
        for key, coverage in rebuilt.items():
            own = set_aside[set_aside.detector == key[3]]
            spans = own[["interval_start_s", "interval_end_s"]].to_numpy()
            if own.empty:
                np.testing.assert_array_equal(coverage.observed, before[key].observed)
                np.testing.assert_array_equal(coverage.exposure, before[key].exposure)
            else:
                assert total_duration(intersect(coverage.observed, spans)) == 0.0


def test_event_keys_are_unique_across_the_epochs(run):
    inventory = pd.concat(
        [table(run, identifier, "inventory") for identifier in EPOCHS.values()]
    )
    set_aside = pd.concat(
        [table(run, identifier, "set_aside") for identifier in EPOCHS.values()]
    )

    keys = pd.concat([inventory.event_key, set_aside.event_key])
    assert keys.is_unique
    # the same native events in both epochs, under keys naming each epoch
    assert set(inventory.epoch) == {"02_r1", "04_r2"}


def test_every_table_keeps_its_dtypes_through_a_write_and_read(run):
    for identifier in EPOCHS.values():
        for name, dtypes in runner.TABLE_DTYPES.items():
            read = table(run, identifier, name)
            assert list(read.columns) == list(dtypes), name
            assert read.dtypes.to_dict() == {
                column: pd.api.types.pandas_dtype(dtype)
                for column, dtype in dtypes.items()
            }, name


def test_the_manifest_records_the_inputs_and_each_epoch(run):
    manifest = json.loads((run / "manifest.json").read_text())

    settings = manifest["settings"]
    assert settings["input_runs"] == {
        "swr": swr_inventory.SWR_INVENTORY_RUN_ID,
        "carey": CAREY_RUN,
        "memoryless": MEMORYLESS_RUN,
        "artifacts": swr_inventory.ARTIFACT_INTERVALS_RUN_ID,
    }
    assert settings["swr_source"] == swr_inventory.swr_source_name()
    assert settings["roster"]["memoryless"] == "published"
    assert settings["roster"]["memoryless_criteria"] == "replay_criteria"
    assert set(settings["native_is_eligible"]) == {"mua", "state_space"}
    for identifier in EPOCHS.values():
        record = manifest["epochs"][identifier]
        assert record["status"] == "ok"
        decode = runner.DECODE_ROOT / f"{identifier}_80_data_over_time.pkl"
        assert record["decode"]["sha256"] == sha256_file(decode)
        assert record["epoch_bounds_s"] == [float(TIME[0]), float(TIME[-1])]
        assert set(record["outputs"]) == set(runner.TABLE_DTYPES)
        for output in record["outputs"].values():
            assert sha256_file(output["path"]) == output["sha256"]
    assert manifest["epochs"][WORKED]["counts"]["kay"]["n_native"] == len(KAY)
    assert manifest["epochs"][MASKED]["counts"]["kay"]["n_native"] == len(MASKED_KAY)
    for record in manifest["epochs"].values():
        assert record["wall_clock_s"] > 0 and record["process_peak_rss_gib"] > 0
    assert set(manifest["epochs"][MASKED]["unavailable"]) == {
        "memoryless",
        "memoryless_criteria",
    }


class TestAFailedEpoch:
    @pytest.mark.parametrize("identifier", [SHIFTED, NO_DECODE])
    def test_it_leaves_no_tables_and_a_failed_record(self, run, identifier):
        record = json.loads((run / "manifest.json").read_text())["epochs"][identifier]

        assert record["status"] == "failed"
        assert "outputs" not in record
        assert not list(run.glob(f"{identifier}_*"))

    def test_a_grid_other_than_the_position_span_fails_with_that_reason(self, run):
        record = json.loads((run / "manifest.json").read_text())["epochs"][SHIFTED]

        assert "position span" in record["reason"]
        assert record["epoch_bounds_s"] == [float(TIME[0]), float(TIME[-1])]

    def test_an_error_is_recorded_with_its_traceback(self, run):
        record = json.loads((run / "manifest.json").read_text())["epochs"][NO_DECODE]

        assert "FileNotFoundError" in record["error"]
        assert "Traceback" in record["traceback"]


def test_a_rerun_skips_ok_epochs(run, monkeypatch, capsys):
    before = {path.name: path.stat().st_mtime_ns for path in run.glob("*.parquet")}

    def refuse(*args, **kwargs):
        raise AssertionError("an ok epoch was run again")

    monkeypatch.setattr(runner, "compare_epoch", refuse)
    runner.main([*ARGS, "--epoch-id", WORKED])

    assert "already written" in capsys.readouterr().out
    after = {path.name: path.stat().st_mtime_ns for path in run.glob("*.parquet")}
    assert after == before


def test_a_rerun_under_other_settings_is_refused(run, inputs):
    other = inputs / "carey" / "careyrun02"
    other.mkdir()
    (other / "manifest.json").write_text(
        json.dumps({"run_id": "careyrun02", "epochs": {}})
    )
    changed = [*ARGS]
    changed[changed.index(CAREY_RUN)] = "careyrun02"

    with pytest.raises(SystemExit, match="other settings"):
        runner.main([*changed, "--epoch-id", WORKED])
