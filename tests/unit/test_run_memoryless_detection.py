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
    marks_cache_path,
    parse_epoch_identifier,
    run_one,
)
from tests.unit._memoryless_clusterless_epoch import (
    SAMPLING_FREQUENCY,
    _StandInDetector,
    build_tiling_tetrode,
)

EPOCH_ID = "sim20200101_02_r1_HPC"
N_SHUFFLES = 19
"""The fewest draws at which p <= 0.05 is possible."""
CONFIG = MemorylessSequenceConfig(sampling_frequency=SAMPLING_FREQUENCY)


class TestParseEpochIdentifier:
    def test_the_canonical_decode_carries_kay_swrs(self):
        key = parse_epoch_identifier("chimi20200212_02_r1_HPC")

        assert key.nwb_file_name == "chimi20200212_.nwb"
        assert key.epoch_name == "02_r1"
        assert key.brain_area == "HPC"
        assert key.swr_definition == DEFAULT_SWR_DEFINITION == "Kay"
        assert key.is_canonical

    @pytest.mark.parametrize(
        ("suffix", "expected"),
        [
            ("_Karlsson", {"swr_definition": "Karlsson"}),
            ("_all", {"training_type": "all"}),
            ("_non_stationary", {"non_stationary": True}),
            ("_contfrag", {"contfrag": True}),
            (
                "_contfrag_Karlsson_non_stationary_all",
                {
                    "contfrag": True,
                    "swr_definition": "Karlsson",
                    "non_stationary": True,
                    "training_type": "all",
                },
            ),
        ],
    )
    def test_each_decode_suffix_is_read_in_the_order_it_is_appended(
        self, suffix, expected
    ):
        """As run_decoding_bandit.py and run_decoding_contfrag.py append them."""
        key = parse_epoch_identifier("chimi20200212_02_r1_HPC" + suffix)

        assert {name: getattr(key, name) for name in expected} == expected
        assert key.brain_area == "HPC"
        assert not key.is_canonical

    @pytest.mark.parametrize(
        "bad",
        [
            "chimi20200212_02",
            "a__c_d",
            "chimi20200212_02_r1_HPC_Kay",
            "x_02_r1_HPC_all_Karlsson",
        ],
    )
    def test_a_name_no_decode_writes_is_refused(self, bad):
        """Including suffixes out of order, and an explicit Kay, which no
        decode appends."""
        with pytest.raises(ValueError):
            parse_epoch_identifier(bad)


def touch(path):
    path.write_bytes(b"")


class TestFindEpochs:
    def write_decode(self, root, epoch_id, *, data_over_time=True):
        paths = input_paths(epoch_id, root)
        touch(paths["detector"])
        if data_over_time:
            touch(paths["data_over_time"])

    def test_canonical_hpc_decodes_with_both_artifacts(self, tmp_path):
        for epoch_id in ("a20200101_02_r1_HPC", "a20200101_04_r2_HPC"):
            self.write_decode(tmp_path, epoch_id)

        epochs, skipped = find_epochs(None, tmp_path)

        assert epochs == ["a20200101_02_r1_HPC", "a20200101_04_r2_HPC"]
        assert skipped == {}

    def test_what_is_left_out_is_recorded_with_its_reason(self, tmp_path):
        self.write_decode(tmp_path, "a20200101_02_r1_HPC")
        self.write_decode(tmp_path, "a20200101_04_r2_HPC", data_over_time=False)
        self.write_decode(tmp_path, "a20200101_06_r3_HPC_all")
        self.write_decode(tmp_path, "a20200101_08_r4_HPC_bogus")

        epochs, skipped = find_epochs(None, tmp_path)

        assert epochs == ["a20200101_02_r1_HPC"]
        assert skipped["a20200101_04_r2_HPC"] == "missing data_over_time"
        assert "variant" in skipped["a20200101_06_r3_HPC_all"]
        assert "no decode appends" in skipped["a20200101_08_r4_HPC_bogus"]

    def test_variants_can_be_asked_for(self, tmp_path):
        self.write_decode(tmp_path, "a20200101_02_r1_HPC")
        self.write_decode(tmp_path, "a20200101_02_r1_HPC_all")

        epochs, _ = find_epochs(None, tmp_path, include_variants=True)

        assert sorted(epochs) == ["a20200101_02_r1_HPC", "a20200101_02_r1_HPC_all"]

    def test_an_animal_restricts_the_search(self, tmp_path):
        for epoch_id in ("a20200101_02_r1_HPC", "b20200101_02_r1_HPC"):
            self.write_decode(tmp_path, epoch_id)

        epochs, _ = find_epochs("b", tmp_path)

        assert epochs == ["b20200101_02_r1_HPC"]


def write_epoch(root, epoch, epoch_id=EPOCH_ID, *, swr_zscore=None, corrupt=False):
    """Save one epoch as the decode saves it."""
    paths = input_paths(epoch_id, root)
    joblib.dump(
        _StandInDetector({ENCODING_KEY: epoch.encoding_model}), paths["detector"]
    )
    if corrupt:
        paths["data_over_time"].write_bytes(b"not a pickle")
        return
    swr = np.zeros(epoch.time.size, dtype=int)
    swr[50:90] = 1
    pd.DataFrame(
        {
            "animal": "sim",
            "day": 20200101,
            "epoch": epoch_id.split("_")[1] + "_" + epoch_id.split("_")[2],
            "linear_position": epoch.linear_position,
            "head_speed": np.zeros(epoch.time.size),
            "SWR_event": swr,
            "SWR_zscore": (
                np.zeros(epoch.time.size) if swr_zscore is None else swr_zscore
            ),
        },
        index=pd.Index(epoch.time, name="time"),
    ).to_pickle(paths["data_over_time"])


@pytest.fixture(scope="module")
def decode_root(tmp_path_factory):
    """One simulated epoch saved as the decode saves it.

    The simulated animal runs, but the saved speed says it is still, so its
    own decoded runs are candidate sequences long enough to be replay and to
    be shuffled; one SWR is labeled early in the epoch.
    """
    root = tmp_path_factory.mktemp("processed")
    epoch = build_tiling_tetrode()
    write_epoch(root, epoch)
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
        "n_shuffles": N_SHUFFLES,
        "config": CONFIG,
        "provenance": {"git_commit": "test"},
        "marks_loader": lambda key, cache, span: (
            epoch.spike_times,
            epoch.spike_waveform_features,
        ),
    } | overrides
    return run_one(EPOCH_ID, **arguments)


class TestRunOne:
    def test_writes_its_outputs_and_reports_them(self, decode_root, tmp_path):
        record = run(decode_root, tmp_path)

        assert record["status"] == "ok"
        assert set(record["outputs"]) == {
            "events",
            "shuffle_draws",
            "observed",
            "candidates",
        }
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

        assert len(draws) == N_SHUFFLES * len(shuffled)
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

    def test_a_missing_input_raises(self, decode_root, tmp_path):
        """Named explicitly, a missing decode is an error, not a quiet skip."""
        with pytest.raises(FileNotFoundError, match="missing detector"):
            run_one(
                "sim20200101_04_r2_HPC",
                root=decode_root[0],
                output_dir=tmp_path / "out",
                cache_dir=tmp_path / "cache",
                run_id="test-run",
                run_seed=7,
                shuffle=True,
                n_shuffles=N_SHUFFLES,
                config=CONFIG,
                provenance={"git_commit": "test"},
            )

    def test_marks_that_are_not_the_encoded_ones_are_refused(
        self, decode_root, tmp_path
    ):
        _, epoch = decode_root
        changed = [np.array(f) for f in epoch.spike_waveform_features]
        changed[0][10, 0] += 1.0

        with pytest.raises(ValueError, match="encoding marks"):
            run(
                decode_root,
                tmp_path,
                marks_loader=lambda key, cache, span: (epoch.spike_times, changed),
            )

    def test_what_was_observed_is_written_apart_from_what_was_found(
        self, decode_root, tmp_path
    ):
        _, epoch = decode_root
        observed = read_table(run(decode_root, tmp_path)["outputs"]["observed"])

        assert observed.kind.tolist() == ["decoded_windows", "swr_labels"]
        swr = observed[observed.kind == "swr_labels"].iloc[0]
        assert swr.start_s == epoch.time[0]
        assert swr.end_s == epoch.time[-1]
        windows = observed[observed.kind == "decoded_windows"].iloc[0]
        assert windows.start_s == epoch.time[CONFIG.window_n_samples // 2]

    def test_the_record_carries_the_grid_the_marks_and_the_encoding(
        self, decode_root, tmp_path
    ):
        _, epoch = decode_root
        record = run(decode_root, tmp_path)

        assert record["time_grid"]["n_samples"] == epoch.time.size
        assert record["time_grid"]["start_s"] == epoch.time[0]
        assert len(record["marks_sha256"]) == 64
        assert record["encoding"]["n_electrodes"] == len(epoch.spike_times)
        assert record["git_commit"] == "test"
        assert record["decode"]["swr_definition"] == "Kay"

    def test_the_record_counts_what_the_table_cannot_show(self, decode_root, tmp_path):
        """Unknown speed, unknown SWR coverage and untestable shuffles are
        counted per epoch, so a quiet epoch can be told from an untracked one."""
        record = run(decode_root, tmp_path)

        assert record["n_windows"] > 0
        assert record["n_windows_speed_unknown"] == 0
        assert record["n_swr_unknown"] == 0
        assert record["n_shuffle_untestable"] == 0

    def test_missing_lfp_reaches_the_table_as_unknown(self, tmp_path):
        """A NaN stretch of SWR_zscore is missing LFP: every candidate touching
        it is ``unknown``, never ``non_swr``, and the observed table says so."""
        epoch = build_tiling_tetrode()
        root = tmp_path / "processed"
        root.mkdir()
        zscore = np.zeros(epoch.time.size)
        zscore[: epoch.time.size // 2] = np.nan
        write_epoch(root, epoch, swr_zscore=zscore)

        record = run((root, epoch), tmp_path)
        table = read_table(record["outputs"]["events"])
        observed = read_table(record["outputs"]["observed"])

        assert record["n_swr_unknown"] == int((table.overlap_class == "unknown").sum())
        assert record["n_swr_unknown"] > 0
        assert (
            not (table.overlap_class == "non_swr").any()
            or (
                table[table.overlap_class == "non_swr"].event_start_s
                >= epoch.time[epoch.time.size // 2]
            ).all()
        )
        swr = observed[observed.kind == "swr_labels"]
        assert swr.start_s.min() >= epoch.time[epoch.time.size // 2]


class TestLoadMarks:
    def test_raw_marks_are_filtered_as_the_decode_was_and_cached(
        self, tmp_path, monkeypatch
    ):
        import scripts.decoding.run_memoryless_detection as runner

        raw_times = [
            np.array([1.0, 2.0, 99.0]),
            np.array([1.00001, 3.0]),
            np.array([1.00002, 4.0]),
            np.array([5.0]),
        ]
        raw_marks = [np.arange(t.size, dtype=float)[:, None] for t in raw_times]
        calls = []

        def fetch(nwb_file_name):
            calls.append(nwb_file_name)
            return raw_times, raw_marks

        monkeypatch.setattr(runner, "fetch_raw_hpc_unit_marks", fetch)
        key = parse_epoch_identifier(EPOCH_ID)

        times, marks = runner.load_marks(key, tmp_path, (0.0, 10.0))
        again = runner.load_marks(key, tmp_path, (0.0, 10.0))

        # the three-tetrode cluster at 1 s goes; 99 s is outside the decode
        assert [t.tolist() for t in times] == [[2.0], [3.0], [4.0], [5.0]]
        assert [m[:, 0].tolist() for m in marks] == [[1.0], [1.0], [1.0], [0.0]]
        assert calls == [key.nwb_file_name]
        assert [t.tolist() for t in again[0]] == [t.tolist() for t in times]
        assert marks_cache_path(key, tmp_path).exists()

    def test_the_record_names_the_marks_rule(self, decode_root, tmp_path):
        record = run(decode_root, tmp_path, shuffle=False)

        assert record["marks"]["decode_commit"] == "0820d39d"


class TestMain:
    def cli(self, decode_root, tmp_path, *extra):
        """The runner through its command line, its marks already cached."""
        root, epoch = decode_root
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(exist_ok=True)
        key = parse_epoch_identifier(EPOCH_ID)
        joblib.dump(
            (epoch.spike_times, epoch.spike_waveform_features),
            marks_cache_path(key, cache_dir),
        )
        output_root = tmp_path / "memoryless"
        arguments = {
            "--epoch-id": EPOCH_ID,
            "--run-seed": "7",
            "--n-shuffles": str(N_SHUFFLES),
            "--run-id": "cli-run",
            "--root": str(root),
            "--output-root": str(output_root),
            "--cache-dir": str(cache_dir),
        }
        for flag, value in zip(extra[::2], extra[1::2], strict=True):
            arguments[flag] = value
        main([item for pair in arguments.items() for item in pair if item != ""])
        return json.loads((output_root / "cli-run" / "manifest.json").read_text())

    def test_a_run_writes_its_manifest_from_cached_marks(self, decode_root, tmp_path):
        manifest = self.cli(decode_root, tmp_path)

        assert manifest["run_id"] == "cli-run"
        assert manifest["epochs"][EPOCH_ID]["status"] == "ok"
        assert manifest["settings"]["shuffle"]["n_shuffles"] == N_SHUFFLES
        assert manifest["settings"]["shuffle"]["run_seed"] == 7
        assert manifest["settings"]["config"]["dispersion_threshold_cm"] == 12.0
        assert manifest["settings"]["git_commit"]
        assert "non_local_detector" in manifest["settings"]["libraries"]

    def test_a_rerun_with_the_same_settings_resumes(
        self, decode_root, tmp_path, capsys
    ):
        first = self.cli(decode_root, tmp_path)
        capsys.readouterr()

        again = self.cli(decode_root, tmp_path)

        assert "already written" in capsys.readouterr().out
        assert again["epochs"][EPOCH_ID] == first["epochs"][EPOCH_ID]

    def test_overwrite_redoes_a_written_epoch(self, decode_root, tmp_path, capsys):
        self.cli(decode_root, tmp_path)
        capsys.readouterr()

        again = self.cli(decode_root, tmp_path, "--overwrite", "")

        out = capsys.readouterr().out
        assert "already written" not in out
        assert "candidates" in out
        assert again["epochs"][EPOCH_ID]["status"] == "ok"

    def test_a_rerun_with_other_settings_is_refused(self, decode_root, tmp_path):
        """The manifest would otherwise record settings the finished epochs
        were not made with."""
        self.cli(decode_root, tmp_path)

        with pytest.raises(SystemExit, match="other settings"):
            self.cli(decode_root, tmp_path, "--n-shuffles", str(N_SHUFFLES + 10))

    def test_a_rerun_under_another_library_version_is_refused(
        self, decode_root, tmp_path, monkeypatch
    ):
        """The finished epochs were made with the first versions; the rest
        must not be made with others under the same run."""
        self.cli(decode_root, tmp_path)
        monkeypatch.setattr(
            "scripts.decoding.run_memoryless_detection.version",
            lambda name: "0.0.0-upgraded",
        )

        with pytest.raises(SystemExit, match="other settings"):
            self.cli(decode_root, tmp_path)

    def test_too_few_shuffles_are_refused_before_anything_runs(
        self, decode_root, tmp_path
    ):
        with pytest.raises(SystemExit):
            self.cli(decode_root, tmp_path, "--n-shuffles", str(N_SHUFFLES - 1))

    def test_the_sensitivity_switch_is_recorded(self, decode_root, tmp_path):
        manifest = self.cli(decode_root, tmp_path, "--keep-single-window-runs", "")

        assert manifest["settings"]["config"]["drop_single_window_runs"] is False

    def test_one_failing_epoch_does_not_stop_the_others(self, tmp_path, capsys):
        """The second epoch's data_over_time is unreadable: it is recorded as
        failed with its error, the first is written, and the run exits non-zero."""
        epoch = build_tiling_tetrode()
        root = tmp_path / "processed"
        root.mkdir()
        write_epoch(root, epoch, "sim20200101_02_r1_HPC")
        write_epoch(root, epoch, "sim20200101_04_r2_HPC", corrupt=True)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        key = parse_epoch_identifier(EPOCH_ID)
        joblib.dump(
            (epoch.spike_times, epoch.spike_waveform_features),
            marks_cache_path(key, cache_dir),
        )
        output_root = tmp_path / "memoryless"

        with pytest.raises(SystemExit, match="1 epochs failed"):
            main(
                [
                    "--animal",
                    "sim",
                    "--run-seed",
                    "7",
                    "--n-shuffles",
                    str(N_SHUFFLES),
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
        assert manifest["epochs"]["sim20200101_02_r1_HPC"]["status"] == "ok"
        failed = manifest["epochs"]["sim20200101_04_r2_HPC"]
        assert failed["status"] == "failed"
        assert "Error" in failed["error"]
        assert "failed" in capsys.readouterr().out
        assert (output_root / "cli-run" / "run.log").exists()

    def test_a_skipped_epoch_is_told_on_the_console(self, tmp_path, capsys):
        epoch = build_tiling_tetrode()
        root = tmp_path / "processed"
        root.mkdir()
        write_epoch(root, epoch, "sim20200101_02_r1_HPC")
        input_paths("sim20200101_04_r2_HPC", root)["detector"].write_bytes(b"")
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        key = parse_epoch_identifier(EPOCH_ID)
        joblib.dump(
            (epoch.spike_times, epoch.spike_waveform_features),
            marks_cache_path(key, cache_dir),
        )

        main(
            [
                "--animal",
                "sim",
                "--run-seed",
                "7",
                "--n-shuffles",
                str(N_SHUFFLES),
                "--run-id",
                "cli-run",
                "--root",
                str(root),
                "--output-root",
                str(tmp_path / "memoryless"),
                "--cache-dir",
                str(cache_dir),
            ]
        )

        out = capsys.readouterr().out
        assert "left out sim20200101_04_r2_HPC: missing data_over_time" in out
