"""The event-features runner end to end, on simulated epochs written to disk.

The epochs are the comparison runner's simulated ones
(``test_run_inventory_comparison``): its decode frames, SWR run (Kay,
Karlsson, Yu, Zugaro), Carey run and artifact run, on its 60 s grid at 500 Hz.
Each decode frame gets the columns the features read (position, the MAP, the
state probabilities, the HPD, the ripple and MUA z-scores), and each epoch a
fitted clusterless encoding model and its marks, as the decode saves them and
as the marks cache holds them. The memoryless run is a real one, made by the
memoryless runner's ``run_one`` on the worked epoch alone. The comparison
runner is run on the three epochs first; the features runner then reads its
tables.

- ``02_r1``, the worked epoch: every detector, the memoryless run included.
- ``04_r2``: an artifact core at 30.05 s and no memoryless record.
- ``10_r5``: its decode is rewritten after the comparison ran, so its sha256
  is no longer the one the comparison recorded.
- ``12_r6``: no event of any inventory, no Carey record and no memoryless
  record, so no compared event.

The simulated animal sweeps 60 cm in 0.3 s once a second, firing only then,
while its saved speed says it is still (but over 20-25 s): the memoryless
detector finds sequences there, and the silent holds between sweeps break
them. The track is 280 cm, so every bin has bins over 100 cm away and the
evidence can be scored. Nothing here reaches Spyglass or ``Processed-Data``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import dask
import joblib
import numpy as np
import pandas as pd
import pytest
from dask.distributed import LocalCluster
from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
)

import scripts.detection.run_event_features as runner
import scripts.detection.run_inventory_comparison as comparison
from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.analysis.events import analysed_time, swr_source
from continuum_swr_replay.analysis.hippocampus.detection_inputs import (
    input_paths,
    marks_cache_path,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    ENCODING_KEY,
)
from continuum_swr_replay.analysis.inventory_comparison.availability import (
    still_intervals,
)
from continuum_swr_replay.analysis.inventory_comparison.cells import (
    membership_against,
)
from continuum_swr_replay.analysis.inventory_comparison.features import (
    DURATION_STRATUM_LABELS,
    _grid_rows,
)
from continuum_swr_replay.analysis.inventory_comparison.overlap import MATCH_DTYPES
from continuum_swr_replay.analysis.inventory_comparison.populations import (
    rebuild_matches,
)
from continuum_swr_replay.analysis.inventory_comparison.schema import (
    INVENTORY_DTYPES,
    typed_table,
)
from continuum_swr_replay.analysis.inventory_comparison.selection import (
    REJECTION_COHORTS,
    REJECTION_DTYPES,
)
from continuum_swr_replay.data_loaders.utils import parse_epoch_identifier
from continuum_swr_replay.runs.io import json_ready, sha256_file
from continuum_swr_replay.tables.io import read_table, write_atomically, write_table
from scripts.detection.run_memoryless_detection import run_one
from tests.integration.test_run_inventory_comparison import (
    ANIMAL,
    CAREY_RUN,
    DAY,
    KAY,
    MASKED_KARLSSON,
    MASKED_KAY,
    STORED_BAND,
    TIME,
    add_swr_detector,
    write_carey_run,
    write_decode,
)
from tests.unit._memoryless_clusterless_epoch import _StandInDetector
from tests.unit._swr_inventory_fixtures import (
    use_swr_source,
    write_artifact_run,
    write_swr_inventory,
)

pytestmark = [pytest.mark.slow, pytest.mark.integration]

EPOCHS = {
    "02_r1": "sim20200101_02_r1_HPC",
    "04_r2": "sim20200101_04_r2_HPC",
    "10_r5": "sim20200101_10_r5_HPC",
    "12_r6": "sim20200101_12_r6_HPC",
}
WORKED, MASKED, CHANGED, EMPTY = EPOCHS.values()
MEMORYLESS_RUN, INVENTORY_RUN = "memorylessrun01", "comparison01"
COMMIT = "featurestest0000000000000000000000000000"
RUN_ID = COMMIT[:12]
SAMPLING_FREQUENCY = 500.0
N_SHUFFLES = 19

TRACK_CM = 280.0
SWEEP_S, CYCLE_S, SWEEP_CM = 0.3, 1.0, 60.0


def sweeping_position() -> tuple[np.ndarray, np.ndarray]:
    """The animal's linear position on ``TIME``, and whether it is sweeping.

    Each second it sweeps 60 cm in 0.3 s from a start that steps along the
    track, then holds where it stopped.
    """
    cycle = np.floor(TIME / CYCLE_S).astype(int)
    phase = TIME - cycle * CYCLE_S
    start = 10.0 + (cycle * 37) % 200
    sweeping = phase < SWEEP_S
    position = start + SWEEP_CM * np.minimum(phase, SWEEP_S) / SWEEP_S
    return position, sweeping


def simulate_marks() -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Place-tuned marks on two tetrodes, each unit its own amplitude.

    Units fire during the sweeps only, so a hold is silent. Spikes fall
    inside the grid's rows, never on its final sample.
    """
    position, sweeping = sweeping_position()
    rng = np.random.default_rng(20261018)
    centers = np.arange(5.0, TRACK_CM, 10.0)
    spike_times, features = [], []
    for tetrode in range(2):
        times, marks = [], []
        for unit, center in enumerate(centers[tetrode::2]):
            rate = 80.0 * np.exp(-0.5 * ((position - center) / 8.0) ** 2) * sweeping
            counts = rng.poisson(rate[:-1] / SAMPLING_FREQUENCY)
            sample = np.repeat(np.arange(TIME.size - 1), counts)
            times.append(
                TIME[sample] + rng.uniform(0.0, 1.0 / SAMPLING_FREQUENCY, sample.size)
            )
            marks.append(rng.normal(40.0 + 15.0 * unit, 4.0, size=(sample.size, 2)))
        order = np.argsort(np.concatenate(times))
        spike_times.append(np.concatenate(times)[order])
        features.append(np.concatenate(marks)[order])
    return position, spike_times, features


def write_epoch(root: Path, epoch: str, identifier: str, simulated: dict) -> None:
    """The comparison test's decode frame with the columns the features read,
    and the decode's saved encoding model."""
    write_decode(root, epoch, identifier)
    path = comparison.decode_path(identifier)
    dot = pd.read_pickle(path)
    position = simulated["position"]
    continuous = dot["Non-Local_Continuous_prob"].to_numpy()
    dot = dot.assign(
        linear_position=position,
        map_estimate=np.clip(position + 40.0, 0.0, TRACK_CM),
        track_segment_id=0,
        patch_id=1,
        hpd=12.0,
        **{
            "Local_prob": 1.0 - continuous - 0.05,
            "No-Spike_prob": np.where(np.isnan(continuous), np.nan, 0.0),
        },
        SWR_zscore=np.sin(2 * np.pi * TIME / 7.0),
        MUA_zscore=np.cos(2 * np.pi * TIME / 5.0),
    )
    # the decode stores the state probabilities as float32
    states = [c for c in dot.columns if c.endswith("_prob")]
    dot[states] = dot[states].astype(np.float32)
    dot.to_pickle(path)
    joblib.dump(
        _StandInDetector({ENCODING_KEY: simulated["encoding_model"]}),
        input_paths(identifier, root)["detector"],
    )


def write_marks_cache(cache_dir: Path, root: Path, identifier: str, simulated: dict):
    """The marks as the memoryless runner caches them for this decode."""
    detector_sha256 = sha256_file(input_paths(identifier, root)["detector"])
    path = marks_cache_path(
        parse_epoch_identifier(identifier), cache_dir, detector_sha256
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((simulated["spike_times"], simulated["features"]), path)


def write_memoryless_run(root: Path, decode_root: Path, simulated: dict) -> None:
    """A real memoryless run on the worked epoch, with its manifest."""
    directory = root / MEMORYLESS_RUN
    record = run_one(
        WORKED,
        root=decode_root,
        output_dir=directory,
        cache_dir=root / "unused_cache",
        run_id=MEMORYLESS_RUN,
        run_seed=7,
        shuffle=True,
        n_shuffles=N_SHUFFLES,
        config=MemorylessSequenceConfig(sampling_frequency=SAMPLING_FREQUENCY),
        provenance={"git_commit": "test"},
        marks_loader=lambda key, cache, span, sha: (
            simulated["spike_times"],
            simulated["features"],
        ),
    )
    (directory / "manifest.json").write_text(
        json.dumps(json_ready({"run_id": MEMORYLESS_RUN, "epochs": {WORKED: record}}))
    )


def write_stop_table(path: Path) -> None:
    """Two stops per epoch: one at a well, one away."""
    rows = []
    for epoch in EPOCHS:
        rows.append(
            pd.DataFrame(
                {
                    "animal": ANIMAL,
                    "session_date": DAY,
                    "epoch": epoch,
                    "stop_id": np.array([11, 12], dtype=np.uint64),
                    "stop_start": [0.0, 25.0],
                    "stop_end": [20.0, 60.0],
                    "last_well_arrival_time_s": [0.0, 25.0],
                    "at_well": [True, False],
                    "occupancy_state": ["at_well", "away"],
                    "occupancy_well_id": [1.0, np.nan],
                    "current_patch": [1.0, 2.0],
                    "is_rewarded_outcome": [1.0, 0.0],
                }
            )
        )
    pd.concat(rows, ignore_index=True).to_parquet(path)


@pytest.fixture(scope="module")
def module_monkeypatch():
    with pytest.MonkeyPatch.context() as patch:
        yield patch


@pytest.fixture(scope="module")
def inputs(tmp_path_factory, module_monkeypatch):
    """Every input on disk, the comparison run on it, and both runners'
    roots pointed at them."""
    tmp_path = tmp_path_factory.mktemp("features")
    monkeypatch = module_monkeypatch
    position, spike_times, features = simulate_marks()
    environment = Environment(place_bin_size=4.0).fit_place_grid(position[:, None])
    simulated = {
        "position": position,
        "spike_times": spike_times,
        "features": features,
        "encoding_model": fit_clusterless_kde_encoding_model(
            TIME,
            position[:, None],
            spike_times,
            features,
            environment,
            sampling_frequency=int(SAMPLING_FREQUENCY),
            disable_progress_bar=True,
        ),
    }

    swr_run = write_swr_inventory(
        tmp_path / "swr",
        {WORKED: KAY, MASKED: MASKED_KAY, CHANGED: KAY, EMPTY: []},
        observed=dict.fromkeys(EPOCHS.values(), STORED_BAND),
    )
    manifest = json.loads((swr_run / "manifest.json").read_text())
    for record in manifest["epochs"].values():
        record["position_span_s"] = [0.0, 60.0]
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
        {WORKED: [], MASKED: [("spk", 30.05, 30.05)], CHANGED: [], EMPTY: []},
    )
    use_swr_source(monkeypatch, swr_run, artifact_run)

    decode_root = tmp_path / "decodes"
    decode_root.mkdir()
    monkeypatch.setattr(comparison, "DECODE_ROOT", decode_root)
    cache_dir = tmp_path / "marks_cache"
    for epoch, identifier in EPOCHS.items():
        write_epoch(decode_root, epoch, identifier, simulated)
        write_marks_cache(cache_dir, decode_root, identifier, simulated)
    empty = comparison.decode_path(EMPTY)
    pd.read_pickle(empty).assign(non_local_event=0).to_pickle(empty)
    write_carey_run(tmp_path / "carey")
    write_memoryless_run(tmp_path / "memoryless", decode_root, simulated)
    processed = tmp_path / "processed"
    stop_table = processed / "replay_content" / "stop_table.parquet"
    stop_table.parent.mkdir(parents=True)
    write_stop_table(stop_table)

    monkeypatch.setattr(comparison, "CAREY_ROOT", tmp_path / "carey")
    monkeypatch.setattr(comparison, "MEMORYLESS_ROOT", tmp_path / "memoryless")
    monkeypatch.setattr(comparison, "OUTPUT_ROOT", tmp_path / "inventory")
    monkeypatch.setattr(comparison, "analysed_epoch_ids", lambda: [*EPOCHS.values()])
    comparison.main(
        [
            "--carey-run-id",
            CAREY_RUN,
            "--memoryless-run-id",
            MEMORYLESS_RUN,
            "--run-id",
            INVENTORY_RUN,
            "--animal",
            ANIMAL,
        ]
    )
    # the decode changes after the comparison read it
    changed = comparison.decode_path(CHANGED)
    pd.read_pickle(changed).assign(head_speed=1.5).to_pickle(changed)

    monkeypatch.setattr(runner, "INVENTORY_ROOT", tmp_path / "inventory")
    monkeypatch.setattr(runner, "MEMORYLESS_ROOT", tmp_path / "memoryless")
    monkeypatch.setattr(runner, "MARKS_CACHE_DIR", cache_dir)
    monkeypatch.setattr(runner, "PROCESSED_DATA_DIR", processed)
    monkeypatch.setattr(runner, "STOP_TABLE", stop_table)
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "event_features")
    monkeypatch.setattr(
        runner, "provenance_record", lambda root: {"commit": COMMIT, "dirty": False}
    )
    monkeypatch.setattr(
        runner,
        "LocalCluster",
        lambda **kwargs: LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            processes=False,
            dashboard_address=":0",
            local_directory=str(tmp_path / "dask-scratch"),
        ),
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


ARGS = ["--inventory-run-id", INVENTORY_RUN, "--n-workers", "1"]


@pytest.fixture(scope="module")
def run(inputs):
    """The features runner over the three epochs, one failing; its folder."""
    with (
        dask.config.set({"temporary-directory": str(inputs / "dask-temporary")}),
        pytest.raises(SystemExit, match="1 epochs failed"),
    ):
        runner.main([*ARGS, "--animal", ANIMAL])
    return inputs / "event_features" / RUN_ID


def table(run_dir: Path, identifier: str, name: str) -> pd.DataFrame:
    return read_table(run_dir / f"{identifier}_{name}.parquet")


def stored(inputs: Path, identifier: str, name: str) -> pd.DataFrame:
    return read_table(
        comparison.output_path(identifier, inputs / "inventory" / INVENTORY_RUN, name)
    )


def compared_events(inputs: Path, identifier: str) -> pd.DataFrame:
    inventory = stored(inputs, identifier, "inventory")
    return inventory[inventory.whole_event_speed_eligible].reset_index(drop=True)


def manifest(run_dir: Path) -> dict:
    loaded: dict = json.loads((run_dir / "manifest.json").read_text())
    return loaded


def dtypes_of(mapping) -> dict:
    return {name: pd.api.types.pandas_dtype(dtype) for name, dtype in mapping.items()}


class TestTheWorkedEpoch:
    def test_it_writes_all_three_tables_and_an_ok_record(self, run):
        record = manifest(run)["epochs"][WORKED]

        assert record["status"] == "ok"
        assert set(record["outputs"]) == set(runner.TABLE_DTYPES)
        for output in record["outputs"].values():
            assert Path(output["path"]).exists()
            assert sha256_file(output["path"]) == output["sha256"]
        assert record["memoryless_available"] is True

    def test_the_memoryless_run_found_compared_events(self, inputs):
        compared = compared_events(inputs, WORKED)

        assert (compared.detector == "memoryless_criteria").any()

    def test_one_row_per_compared_event_and_no_baseline(self, run, inputs):
        features = table(run, WORKED, "features")
        compared = compared_events(inputs, WORKED)

        assert features.detector.value_counts().to_dict() == (
            compared.detector.value_counts().to_dict()
        )
        assert sorted(features.event_key) == sorted(compared.event_key)
        assert not [c for c in features.columns if c.startswith("baseline")]
        assert "matched_event_key" not in features
        assert "evidence_rank" not in features
        record = manifest(run)["epochs"][WORKED]
        assert record["counts"]["events"] == compared.detector.value_counts().to_dict()

    def test_the_evidence_is_computed_on_the_events_rows_alone(self, run):
        features = table(run, WORKED, "features")
        first, n_rows = _grid_rows(TIME, features)
        requested = np.zeros(TIME.size, dtype=bool)
        for start, n in zip(first, n_rows, strict=True):
            requested[start : start + n] = True

        record = manifest(run)["epochs"][WORKED]
        assert record["n_evidence_rows"] == int(requested.sum())

    def test_every_scored_row_was_computed(self, run):
        features = table(run, WORKED, "features")

        assert (features.evidence_unscorable == "").all()
        assert features.evidence_dll.notna().all()
        record = manifest(run)["epochs"][WORKED]
        assert record["counts"]["unscorable"] == {}
        assert 0 < record["likelihood_s"] < record["wall_clock_s"]
        assert 0 < record["likelihood_share"] < 1

    def test_the_posterior_features_are_read_in_both_namespaces(self, run):
        features = table(run, WORKED, "features")

        assert features.ss_local_prob_mean.notna().any()
        assert features.ms_n_windows.notna().all()
        assert (features.ms_n_windows > 0).any()
        ours = features[features.detector == "state_space"]
        assert ours.ss_disagreement.isna().all()
        assert ours.ss_nearest_event_gap_s.isna().all()

    def test_the_stop_context_and_the_strata(self, run):
        features = table(run, WORKED, "features")

        early = features[features.event_start_s < 20.0]
        assert (early.stop_id == 11).all() and early.at_well.all()
        assert features.duration_stratum.isin(DURATION_STRATUM_LABELS).all()

    def test_its_memberships_reproduce_the_stored_tables(self, run, inputs):
        membership = table(run, WORKED, "pair_membership")
        compared = compared_events(inputs, WORKED)
        names = ("observations", "set_aside", "inventory", "pairs", "directional")
        tables = {name: stored(inputs, WORKED, name) for name in (*names, "cells")}
        dot = analysed_time.with_swr_source(
            pd.read_pickle(comparison.decode_path(WORKED))
        )
        time, speed = comparison._speed_trace(dot)
        still = {
            (ANIMAL, DAY, "02_r1"): still_intervals(
                time,
                speed,
                DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
                comparison.MAX_GAP_S,
            )
        }

        _, matches = rebuild_matches(
            tables, still, detectors=comparison.DETECTORS, cells=comparison.CELLS
        )
        order = ["event_key", "other_detector"]
        pd.testing.assert_frame_equal(
            membership.sort_values(order, ignore_index=True),
            matches.sort_values(order, ignore_index=True)[list(MATCH_DTYPES)],
        )
        cells = pd.concat(
            [
                membership_against(
                    compared, membership, sources=sources, against=against
                ).assign(cell=name)
                for name, (sources, against) in comparison.CELLS.items()
            ],
            ignore_index=True,
        )[list(tables["cells"].columns)]
        order = ["cell", "event_key"]
        pd.testing.assert_frame_equal(
            cells.sort_values(order, ignore_index=True),
            tables["cells"].sort_values(order, ignore_index=True),
        )

    def test_in_cohort_exactly_when_matched(self, run, inputs):
        rejection = table(run, WORKED, "rejection")
        compared = compared_events(inputs, WORKED)

        ours = compared[compared.detector == "state_space"]
        assert len(rejection) == len(REJECTION_COHORTS) * len(ours)
        assert set(rejection.cohort) == set(REJECTION_COHORTS)
        assert (
            (rejection.stage == "in_cohort") == (rejection.membership == "matched")
        ).all()
        assert not (rejection.stage == "unknown").all()

    def test_each_published_event_reads_as_its_criteria_twin(self, run):
        """The same interval gets the same features; only its identity and its
        neighbours in its own inventory differ."""
        features = table(run, WORKED, "features")
        own = {
            *INVENTORY_DTYPES,
            "previous_gap_s",
            "next_gap_s",
        }
        read = [column for column in features.columns if column not in own]
        bounds = ["event_start_s", "event_end_s"]

        published = features[features.detector == "memoryless"]
        twins = published[bounds].merge(
            features[features.detector == "memoryless_criteria"], on=bounds
        )
        assert len(published) and len(twins) == len(published)
        pd.testing.assert_frame_equal(
            published.sort_values(bounds)[read].reset_index(drop=True),
            twins.sort_values(bounds)[read].reset_index(drop=True),
        )


class TestAnEpochWithoutAMemorylessRun:
    def test_its_ms_features_are_missing(self, run):
        features = table(run, MASKED, "features")
        record = manifest(run)["epochs"][MASKED]

        assert record["status"] == "ok"
        assert record["memoryless_available"] is False
        ms = [column for column in features.columns if column.startswith("ms_")]
        assert ms and features[ms].isna().all().all()
        assert features.ss_local_prob_mean.notna().any()

    def test_its_memberships_against_memoryless_are_unknown(self, run):
        membership = table(run, MASKED, "pair_membership")

        against = membership[
            membership.other_detector.isin(["memoryless", "memoryless_criteria"])
        ]
        assert len(against) and (against.membership == "unknown").all()

    def test_its_stages_are_unknown(self, run):
        rejection = table(run, MASKED, "rejection")

        assert len(rejection) and (rejection.stage == "unknown").all()
        assert (rejection.unknown_reason == "no_shared_exposure").all()
        assert rejection.n_windows.isna().all()


class TestAChangedDecode:
    def test_it_fails_the_epoch_and_writes_nothing(self, run):
        record = manifest(run)["epochs"][CHANGED]

        assert record["status"] == "failed"
        assert "sha256" in record["reason"]
        assert "outputs" not in record
        assert not list(run.glob(f"{CHANGED}_*"))


def test_memberships_that_do_not_reproduce_the_stored_tables_fail_the_epoch(
    run, inputs, tmp_path
):
    """A stored ``directional`` table one count off: the epoch fails, and its
    earlier outputs are removed."""
    inventory_dir = tmp_path / "inventory"
    shutil.copytree(inputs / "inventory" / INVENTORY_RUN, inventory_dir)
    path = comparison.output_path(WORKED, inventory_dir, "directional")
    directional = read_table(path)
    directional.loc[0, "n_source_events"] += 1
    write_atomically(path, lambda temporary: directional.to_parquet(temporary))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    for name in runner.TABLE_DTYPES:
        shutil.copy(run / f"{WORKED}_{name}.parquet", output_dir)

    key = parse_epoch_identifier(WORKED)
    record = runner.run_epoch(
        key.nwb_file_name,
        key.epoch_name,
        output_dir=output_dir,
        inventory_dir=inventory_dir,
        memoryless_dir=inputs / "memoryless" / MEMORYLESS_RUN,
        decode_sha256={WORKED: sha256_file(comparison.decode_path(WORKED))},
        stop_table=runner.STOP_TABLE,
        provenance={"commit": COMMIT, "dirty": False},
    )

    assert record["status"] == "failed"
    assert "directional" in record["error"]
    assert record["decode"]["sha256"] == sha256_file(comparison.decode_path(WORKED))
    assert not list(output_dir.glob("*.parquet"))


@pytest.mark.parametrize("name", ["marks", "detector", "data_over_time"])
def test_inputs_other_than_the_memoryless_runs_fail_the_epoch(
    run, inputs, tmp_path, name
):
    """The memoryless run recorded other marks, another detector file or
    another decode: the epoch is refused with that reason, and its earlier
    outputs are removed."""
    memoryless_dir = tmp_path / "memoryless"
    shutil.copytree(inputs / "memoryless" / MEMORYLESS_RUN, memoryless_dir)
    manifest_path = memoryless_dir / "manifest.json"
    recorded = json.loads(manifest_path.read_text())
    epoch = recorded["epochs"][WORKED]
    entry = epoch["marks"] if name == "marks" else epoch["inputs"][name]
    entry["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(recorded))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    for table_name in runner.TABLE_DTYPES:
        shutil.copy(run / f"{WORKED}_{table_name}.parquet", output_dir)

    key = parse_epoch_identifier(WORKED)
    record = runner.run_epoch(
        key.nwb_file_name,
        key.epoch_name,
        output_dir=output_dir,
        inventory_dir=inputs / "inventory" / INVENTORY_RUN,
        memoryless_dir=memoryless_dir,
        decode_sha256={WORKED: sha256_file(comparison.decode_path(WORKED))},
        stop_table=runner.STOP_TABLE,
        provenance={"commit": COMMIT, "dirty": False},
    )

    assert record["status"] == "failed"
    assert f"The {name} sha256 differ" in record["reason"]
    assert not list(output_dir.glob("*.parquet"))


def test_an_epoch_with_no_compared_event_writes_empty_typed_tables(run):
    record = manifest(run)["epochs"][EMPTY]

    assert record["status"] == "ok"
    assert record["n_evidence_rows"] == 0
    for name, dtypes in runner.TABLE_DTYPES.items():
        read = table(run, EMPTY, name)
        assert read.empty and read.dtypes.to_dict() == dtypes_of(dtypes), name


def test_a_comparison_under_another_swr_source_is_refused(run, monkeypatch):
    monkeypatch.setattr(swr_source, "swr_source_name", lambda: "yu@other+none")

    with pytest.raises(
        SystemExit,
        match=rf"Comparison run {INVENTORY_RUN} read SWR source \S+; the SWR "
        r"source is yu@other\+none\.",
    ):
        runner.main([*ARGS, "--epoch-id", WORKED])


def test_every_table_keeps_its_dtypes_through_a_write_and_read(run):
    for identifier in (WORKED, MASKED, EMPTY):
        for name, dtypes in runner.TABLE_DTYPES.items():
            read = table(run, identifier, name)
            assert list(read.columns) == list(dtypes), name
            assert read.dtypes.to_dict() == dtypes_of(dtypes), name
    assert runner.TABLE_DTYPES["rejection"] == REJECTION_DTYPES


def test_an_empty_epoch_keeps_the_declared_dtypes(tmp_path):
    for name, dtypes in runner.TABLE_DTYPES.items():
        path = tmp_path / f"{name}.parquet"
        write_table(typed_table([], dtypes), path)
        read = read_table(path)
        assert read.empty and read.dtypes.to_dict() == dtypes_of(dtypes), name


def test_every_interval_is_read_on_the_decode_grid(run):
    """Each interval's rows are the grid's samples within its bounds, the
    grid's last sample left out, and the evidence was computed on each of
    them."""
    features = table(run, WORKED, "features")
    first, n_rows = _grid_rows(TIME, features)

    np.testing.assert_array_equal(first, features.first_row)
    np.testing.assert_array_equal(n_rows, features.n_rows)
    assert (features.evidence_n_valid_rows == features.n_rows).all()


def test_the_manifest_records_the_inputs_and_the_rules(run):
    settings = manifest(run)["settings"]

    assert settings["input_runs"] == {
        "inventory": INVENTORY_RUN,
        "swr": swr_source.SWR_INVENTORY_RUN_ID,
        "carey": CAREY_RUN,
        "memoryless": MEMORYLESS_RUN,
        "artifacts": swr_source.ARTIFACT_INTERVALS_RUN_ID,
    }
    assert settings["evidence"]["remote_cm"] == runner.REMOTE_CM == 100.0
    assert settings["evidence"]["block_rows"] == runner.BLOCK_ROWS
    assert settings["evidence"]["library_block_size"] == 100
    assert "baselines" not in settings
    assert settings["grid"]["sample_interval_s"] == comparison.SAMPLE_INTERVAL_S
    assert "unavailable" in settings["tracking_confidence"]
    assert settings["stop_table"]["path"] == "replay_content/stop_table.parquet"
    assert len(settings["stop_table"]["sha256"]) == 64
    record = manifest(run)["epochs"][WORKED]
    assert record["decode"]["sha256"] == sha256_file(comparison.decode_path(WORKED))
    assert len(record["marks"]["sha256"]) == 64
    assert record["wall_clock_s"] > 0 and record["worker_peak_rss_gib"] > 0
    counts = record["counts"]
    assert set(counts) == {"events", "scorable", "unscorable", "rejection_stages"}


def test_the_settings_name_no_absolute_path(run, inputs, tmp_path, monkeypatch):
    """Two roots holding the same stop table, as two worktrees whose
    Processed-Data links to one folder, give the same settings."""
    inventory = json.loads(
        (inputs / "inventory" / INVENTORY_RUN / "manifest.json").read_text()
    )
    settings = []
    for root in (tmp_path / "one" / "Processed-Data", tmp_path / "two" / "elsewhere"):
        stop_table = root / "replay_content" / "stop_table.parquet"
        stop_table.parent.mkdir(parents=True)
        shutil.copy(runner.STOP_TABLE, stop_table)
        with monkeypatch.context() as patch:
            patch.setattr(runner, "PROCESSED_DATA_DIR", root)
            patch.setattr(runner, "STOP_TABLE", stop_table)
            settings.append(runner.run_settings(INVENTORY_RUN, inventory))

    assert settings[0] == settings[1] == manifest(run)["settings"]
    assert str(tmp_path) not in json.dumps(settings[0])


def test_a_rerun_skips_ok_epochs(run, monkeypatch, capsys):
    before = {path.name: path.stat().st_mtime_ns for path in run.glob("*.parquet")}

    def refuse(*args, **kwargs):
        raise AssertionError("an ok epoch was run again")

    monkeypatch.setattr(runner, "run_epoch", refuse)
    runner.main([*ARGS, "--epoch-id", WORKED])

    assert "already written" in capsys.readouterr().out
    after = {path.name: path.stat().st_mtime_ns for path in run.glob("*.parquet")}
    assert after == before


def test_a_rerun_under_other_settings_is_refused(run, monkeypatch):
    monkeypatch.setattr(runner, "REMOTE_CM", 50.0)

    with pytest.raises(SystemExit, match="other settings"):
        runner.main([*ARGS, "--epoch-id", WORKED])


def test_help_runs(capsys):
    with pytest.raises(SystemExit) as exit_:
        runner.main(["--help"])

    assert exit_.value.code == 0
    assert "--inventory-run-id" in capsys.readouterr().out
