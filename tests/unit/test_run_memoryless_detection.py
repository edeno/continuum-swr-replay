"""The per-epoch runner: its file handling, its manifest, and one real run.

Nothing here reaches the Spyglass database. The epoch is the simulated tetrode
the shuffle tests use, whose units tile the track, written to disk in the shapes
the decode saves, and its marks go into the runner's cache, where a second real
run would find them.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence import (
    MEMORYLESS_EVENT_TABLE_DTYPES,
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    ENCODING_KEY,
)
from continuum_swr_replay.table_io import read_table
from scripts.decoding.run_memoryless_detection import (
    DEFAULT_SWR_DEFINITION,
    find_epochs,
    input_paths,
    main,
    parse_epoch_identifier,
    run_one,
)
from tests.unit._memoryless_clusterless_epoch import (
    SAMPLING_FREQUENCY,
    _StandInDetector,
    build_tiling_tetrode,
)

EPOCH_ID = "sim20200101_02_r1_HPC"
CONFIG = MemorylessSequenceConfig(sampling_frequency=SAMPLING_FREQUENCY)


class TestParseEpochIdentifier:
    def test_a_default_decode_carries_kay_swrs(self):
        key = parse_epoch_identifier("chimi20200212_02_r1_HPC")

        assert key.nwb_file_name == "chimi20200212_.nwb"
        assert key.epoch_name == "02_r1"
        assert key.brain_area == "HPC"
        assert key.swr_definition == DEFAULT_SWR_DEFINITION == "Kay"

    def test_another_ripple_detector_is_read_from_the_suffix(self):
        key = parse_epoch_identifier("chimi20200212_02_r1_HPC_Karlsson")

        assert key.swr_definition == "Karlsson"
        assert key.brain_area == "HPC"

    @pytest.mark.parametrize("bad", ["chimi20200212_02", "a_b_c_d_e_f", "a__c_d"])
    def test_a_malformed_identifier_is_refused(self, bad):
        with pytest.raises(ValueError, match="brain_area"):
            parse_epoch_identifier(bad)


def touch(path):
    path.write_bytes(b"")


class TestFindEpochs:
    def test_only_hpc_epochs_with_both_artifacts(self, tmp_path):
        for epoch_id in ("a20200101_02_r1_HPC", "a20200101_04_r2_HPC"):
            for path in input_paths(epoch_id, tmp_path).values():
                touch(path)
        touch(input_paths("a20200101_06_r3_HPC", tmp_path)["detector"])  # no dot
        for path in input_paths("a20200101_02_r1_mPFC", tmp_path).values():
            touch(path)

        assert find_epochs(None, tmp_path) == [
            "a20200101_02_r1_HPC",
            "a20200101_04_r2_HPC",
        ]

    def test_an_animal_restricts_the_search(self, tmp_path):
        for epoch_id in ("a20200101_02_r1_HPC", "b20200101_02_r1_HPC"):
            for path in input_paths(epoch_id, tmp_path).values():
                touch(path)

        assert find_epochs("b", tmp_path) == ["b20200101_02_r1_HPC"]


@pytest.fixture(scope="module")
def decode_root(tmp_path_factory):
    """One simulated epoch saved as the decode saves it.

    The simulated animal runs, but the saved speed says it is still, so its
    own decoded runs are candidate sequences long enough to be replay and to
    be shuffled; one SWR is labeled early in the epoch.
    """
    root = tmp_path_factory.mktemp("processed")
    epoch = build_tiling_tetrode()
    paths = input_paths(EPOCH_ID, root)
    joblib.dump(
        _StandInDetector({ENCODING_KEY: epoch.encoding_model}), paths["detector"]
    )

    swr = np.zeros(epoch.time.size, dtype=int)
    swr[50:90] = 1
    pd.DataFrame(
        {
            "animal": "sim",
            "day": 20200101,
            "epoch": "02_r1",
            "linear_position": epoch.linear_position,
            "head_speed": np.zeros(epoch.time.size),
            "SWR_event": swr,
        },
        index=pd.Index(epoch.time, name="time"),
    ).to_pickle(paths["data_over_time"])
    return root, epoch


def run(decode_root, tmp_path, **overrides):
    root, epoch = decode_root
    arguments = {
        "root": root,
        "output_dir": tmp_path / "out",
        "cache_dir": tmp_path / "cache",
        "run_id": "test-run",
        "run_seed": 7,
        "shuffle": True,
        "n_shuffles": 3,
        "config": CONFIG,
        "provenance": {"git_commit": "test"},
        "marks_loader": lambda key, cache: (
            epoch.spike_times,
            epoch.spike_waveform_features,
        ),
    } | overrides
    return run_one(EPOCH_ID, **arguments)


class TestRunOne:
    def test_writes_the_three_outputs_and_reports_them(self, decode_root, tmp_path):
        record = run(decode_root, tmp_path)

        assert record["status"] == "ok"
        assert set(record["outputs"]) == {"events", "shuffle_draws", "candidates"}
        table = read_table(record["outputs"]["events"])
        assert list(table.columns) == list(MEMORYLESS_EVENT_TABLE_DTYPES)
        assert record["n_candidates"] == len(table)
        assert record["n_shuffled"] == int((table.n_shuffles > 0).sum()) > 0

    def test_every_draw_of_every_shuffled_candidate_is_written(
        self, decode_root, tmp_path
    ):
        record = run(decode_root, tmp_path)
        table = read_table(record["outputs"]["events"])
        draws = read_table(record["outputs"]["shuffle_draws"])
        shuffled = table[table.n_shuffles > 0]

        assert len(draws) == 3 * len(shuffled)
        assert set(draws.ms_event_id) == set(shuffled.ms_event_id)

    def test_no_shuffle_writes_an_untested_table(self, decode_root, tmp_path):
        record = run(decode_root, tmp_path, shuffle=False)

        assert record["n_shuffled"] == 0
        assert read_table(record["outputs"]["shuffle_draws"]).empty

    def test_the_table_is_labeled_from_the_saved_decode(self, decode_root, tmp_path):
        table = read_table(run(decode_root, tmp_path)["outputs"]["events"])

        assert (table.animal == "sim").all()
        assert (table.day == 20200101).all()
        assert (table.epoch == "02_r1").all()
        assert (table.brain_area == "HPC").all()
        assert (table.swr_definition == "Kay").all()
        # the saved labels are complete, so nothing is unknown
        assert not (table.overlap_class == "unknown").any()

    def test_every_input_is_hashed(self, decode_root, tmp_path):
        record = run(decode_root, tmp_path)

        assert set(record["inputs"]) == {"detector", "data_over_time"}
        assert all(len(entry["sha256"]) == 64 for entry in record["inputs"].values())

    def test_the_provenance_travels_with_the_table(self, decode_root, tmp_path):
        record = run(decode_root, tmp_path)

        attrs = read_table(record["outputs"]["events"]).attrs
        assert attrs["git_commit"] == "test"
        assert attrs["epoch_id"] == EPOCH_ID

    def test_a_missing_input_is_a_recorded_skip(self, decode_root, tmp_path):
        record = run_one(
            "sim20200101_04_r2_HPC",
            root=decode_root[0],
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            run_id="test-run",
            run_seed=7,
            shuffle=True,
            n_shuffles=3,
            config=CONFIG,
            provenance={},
        )

        assert record == {
            "status": "skipped",
            "reason": "missing detector, data_over_time",
        }


class TestMain:
    def test_a_run_writes_its_manifest_from_cached_marks(
        self, decode_root, tmp_path, capsys
    ):
        root, epoch = decode_root
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        key = parse_epoch_identifier(EPOCH_ID)
        joblib.dump(
            (epoch.spike_times, epoch.spike_waveform_features),
            cache_dir
            / f"{key.nwb_file_name}_{key.epoch_name}_{key.brain_area}_marks.pkl",
        )
        output_root = tmp_path / "memoryless"

        main(
            [
                "--epoch-id",
                EPOCH_ID,
                "--run-seed",
                "7",
                "--n-shuffles",
                "3",
                "--run-id",
                "cli-run",
                "--root",
                str(root),
                "--output-root",
                str(output_root),
                "--cache-dir",
                str(cache_dir),
            ]
        )

        manifest = json.loads((output_root / "cli-run" / "manifest.json").read_text())
        assert manifest["run_id"] == "cli-run"
        assert manifest["epochs"][EPOCH_ID]["status"] == "ok"
        assert manifest["shuffle"]["n_shuffles"] == 3
        assert manifest["shuffle"]["run_seed"] == 7
        assert manifest["config"]["dispersion_threshold_cm"] == 12.0
        assert "non_local_detector" in manifest["libraries"]
