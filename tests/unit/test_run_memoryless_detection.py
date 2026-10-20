"""The per-epoch runner: its file handling, its manifest, and one real run.

Nothing here reaches the Spyglass database. The epoch is the simulated tetrode
the shuffle tests use, whose units tile the track, written to disk in the shapes
the decode saves, and its marks go into the runner's cache, where a second real
run would find them.

The parallel driver runs on an in-process ``LocalCluster`` of worker threads
standing in for ``LocalCUDACluster``; no test starts a CUDA cluster, and each
test that depends on ``HAS_CUDA`` sets it. Nothing is written outside the
test's ``tmp_path``: every path the runner takes is given under it, the tests
that run ``main`` work in it, so the per-epoch ``logs/`` land there, and dask's
scratch space is pointed into it.
"""

import hashlib
import json
import os
import threading
from pathlib import Path

import dask
import joblib
import numpy as np
import pandas as pd
import pytest
from dask.distributed import Client, LocalCluster

import scripts.decoding.run_memoryless_detection as runner
from continuum_swr_replay.analysis.events import swr_inventory
from continuum_swr_replay.analysis.events.eligibility import artifact_drop_reasons
from continuum_swr_replay.analysis.hippocampus import detection_inputs
from continuum_swr_replay.analysis.hippocampus.detection_inputs import (
    input_paths,
    marks_cache_path,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence import (
    MEMORYLESS_EVENT_TABLE_DTYPES,
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    ENCODING_KEY,
)
from continuum_swr_replay.data_loaders import spikes
from continuum_swr_replay.data_loaders.utils import parse_epoch_identifier
from continuum_swr_replay.parameters import ROOT_DIR
from continuum_swr_replay.runs.provenance import provenance_record
from continuum_swr_replay.tables.io import read_table
from scripts.decoding.run_memoryless_detection import find_epochs, main, run_one
from tests.unit._memoryless_clusterless_epoch import (
    SAMPLING_FREQUENCY,
    _StandInDetector,
    build_tiling_tetrode,
)
from tests.unit._swr_inventory_fixtures import (
    RUN_ID,
    write_artifact_run,
    write_swr_inventory,
)

EPOCH_ID = "sim20200101_02_r1_HPC"
N_SHUFFLES = 19
"""The fewest draws at which p <= 0.05 is possible."""
CONFIG = MemorylessSequenceConfig(sampling_frequency=SAMPLING_FREQUENCY)

DETERMINISTIC = "--xla_gpu_deterministic_ops=true"
TRITON_OFF = "--xla_gpu_enable_triton_gemm=false"
DUMP = "--xla_dump_hlo_as_text"

XLA_FLAGS_CASES = [
    (None, f"{DETERMINISTIC} {TRITON_OFF}"),
    ("", f"{DETERMINISTIC} {TRITON_OFF}"),
    # the user's other flags are kept as they are
    (DUMP, f"{DUMP} {DETERMINISTIC} {TRITON_OFF}"),
    (
        f"{DUMP} --xla_gpu_enable_triton_gemm=true --xla_other=1",
        f"{DUMP} --xla_gpu_enable_triton_gemm=true --xla_other=1 "
        f"{DETERMINISTIC} {TRITON_OFF}",
    ),
    # deterministic ops: the last value of the flag decides
    (DETERMINISTIC, f"{DETERMINISTIC} {TRITON_OFF}"),
    (f"{DETERMINISTIC} {DUMP}", f"{DETERMINISTIC} {DUMP} {TRITON_OFF}"),
    (
        "--xla_gpu_deterministic_ops=false",
        f"--xla_gpu_deterministic_ops=false {DETERMINISTIC} {TRITON_OFF}",
    ),
    (
        "--xla_gpu_deterministic_ops=true --xla_gpu_deterministic_ops=false",
        "--xla_gpu_deterministic_ops=true --xla_gpu_deterministic_ops=false "
        f"{DETERMINISTIC} {TRITON_OFF}",
    ),
    (
        "--xla_gpu_deterministic_ops=false --xla_gpu_deterministic_ops=true",
        f"--xla_gpu_deterministic_ops=false {DETERMINISTIC} {TRITON_OFF}",
    ),
    # Triton GEMM: the last value of the flag decides, and =false is not repeated
    (TRITON_OFF, f"{TRITON_OFF} {DETERMINISTIC}"),
    # a bare flag has no value of "false", so the required value is appended
    (
        "--xla_gpu_enable_triton_gemm",
        f"--xla_gpu_enable_triton_gemm {DETERMINISTIC} {TRITON_OFF}",
    ),
    (
        "--xla_gpu_enable_triton_gemm=true",
        f"--xla_gpu_enable_triton_gemm=true {DETERMINISTIC} {TRITON_OFF}",
    ),
    (
        f"{DETERMINISTIC} --xla_gpu_enable_triton_gemm=true",
        f"{DETERMINISTIC} --xla_gpu_enable_triton_gemm=true {TRITON_OFF}",
    ),
    (
        f"{TRITON_OFF} --xla_gpu_enable_triton_gemm=true",
        f"{TRITON_OFF} --xla_gpu_enable_triton_gemm=true {DETERMINISTIC} {TRITON_OFF}",
    ),
    (
        f"--xla_gpu_enable_triton_gemm=true {TRITON_OFF}",
        f"--xla_gpu_enable_triton_gemm=true {TRITON_OFF} {DETERMINISTIC}",
    ),
    # both already right: nothing is added
    (f"{DETERMINISTIC} {TRITON_OFF}", f"{DETERMINISTIC} {TRITON_OFF}"),
    (f"{TRITON_OFF} {DETERMINISTIC}", f"{TRITON_OFF} {DETERMINISTIC}"),
]
"""XLA_FLAGS as this process inherits them (None: unset), and as every process
that runs JAX gets them. XLA uses the last value of a repeated flag, so each
required flag (deterministic ops on, Triton GEMM off) is appended unless the last
occurrence of its name already has the required value; the user's flags are
kept."""

GIVEN_XLA_FLAGS = f"{DUMP} {DETERMINISTIC} {TRITON_OFF}"
"""XLA_FLAGS handed to ``cuda_cluster``, as ``main`` hands it the settings'."""


def set_xla_flags(monkeypatch, value: str | None) -> None:
    """This process's XLA_FLAGS for one test; None unsets them."""
    if value is None:
        monkeypatch.delenv("XLA_FLAGS", raising=False)
    else:
        monkeypatch.setenv("XLA_FLAGS", value)


class TestSelectEpochs:
    def test_date_and_epoch_filters_select_epochs(self):
        """The date is the NWB stem's last 8 characters, as prepare_data.py
        reads it, so j16's own digits are not read as part of it."""
        epoch_ids = [
            "chimi20990225_02_r1_HPC",
            "chimi20990225_04_r2_HPC",
            "chimi20990226_02_r1_HPC",
            "j1620990707_06_r3_HPC",
        ]

        assert runner.select_epochs(epoch_ids, date=20990225, epoch="02_r1") == [
            "chimi20990225_02_r1_HPC"
        ]
        assert runner.select_epochs(epoch_ids, date=20990707, epoch=None) == [
            "j1620990707_06_r3_HPC"
        ]
        assert runner.select_epochs(epoch_ids, date=None, epoch=None) == epoch_ids


def touch(path):
    path.write_bytes(b"")


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_marks(root, epoch, cache_dir, epoch_id=EPOCH_ID):
    """Cache an epoch's marks where the runner looks for its decode's."""
    cache_dir.mkdir(exist_ok=True)
    detector_sha256 = file_sha256(input_paths(epoch_id, root)["detector"])
    joblib.dump(
        (epoch.spike_times, epoch.spike_waveform_features),
        marks_cache_path(parse_epoch_identifier(epoch_id), cache_dir, detector_sha256),
    )


def analysed_keys(*epoch_ids: str) -> list[tuple[str, str]]:
    """The register's (nwb_file_name, epoch) keys of these canonical epochs."""
    return [
        (key.nwb_file_name, key.epoch_name)
        for key in map(parse_epoch_identifier, epoch_ids)
    ]


def analysed_frame(*epoch_ids: str) -> pd.DataFrame:
    """``epochs_to_run``'s frame (``get_epoch_info``'s shape) for these epochs."""
    keys = analysed_keys(*epoch_ids)
    return pd.DataFrame(
        {"epoch": [int(epoch.split("_")[0]) for _, epoch in keys]},
        index=pd.MultiIndex.from_tuples(
            keys, names=["nwb_file_name", "interval_list_name"]
        ),
    )


class TestFindEpochs:
    def write_decode(self, root, epoch_id, *, data_over_time=True):
        paths = input_paths(epoch_id, root)
        touch(paths["detector"])
        if data_over_time:
            touch(paths["data_over_time"])

    def test_the_analysed_epochs_canonical_decodes(self, tmp_path):
        for epoch_id in ("a20200101_02_r1_HPC", "a20200101_04_r2_HPC"):
            self.write_decode(tmp_path, epoch_id)

        epochs, skipped = find_epochs(
            None,
            tmp_path,
            analysed_keys("a20200101_02_r1_HPC", "a20200101_04_r2_HPC"),
        )

        assert epochs == ["a20200101_02_r1_HPC", "a20200101_04_r2_HPC"]
        assert skipped == {}

    def test_an_analysed_epoch_is_kept_without_its_decode(self, tmp_path):
        """Kept, so that ``run_one`` records it as failed, never left out."""
        self.write_decode(tmp_path, "a20200101_04_r2_HPC", data_over_time=False)

        epochs, skipped = find_epochs(
            None,
            tmp_path,
            analysed_keys(
                "a20200101_02_r1_HPC", "a20200101_04_r2_HPC", "a20200101_06_r3_HPC"
            ),
        )

        assert epochs == [
            "a20200101_02_r1_HPC",
            "a20200101_04_r2_HPC",
            "a20200101_06_r3_HPC",
        ]
        assert skipped == {}

    def test_what_is_left_out_is_recorded_with_its_reason(self, tmp_path):
        self.write_decode(tmp_path, "a20200101_02_r1_HPC")
        self.write_decode(tmp_path, "a20200101_04_r2_HPC")
        self.write_decode(tmp_path, "a20200101_02_r1_HPC_all")
        self.write_decode(tmp_path, "a20200101_08_r4_HPC_bogus")

        epochs, skipped = find_epochs(
            None, tmp_path, analysed_keys("a20200101_02_r1_HPC")
        )

        assert epochs == ["a20200101_02_r1_HPC"]
        assert skipped["a20200101_04_r2_HPC"] == (
            "not an analysed epoch in the epoch register"
        )
        assert "variant" in skipped["a20200101_02_r1_HPC_all"]
        assert "no decode appends" in skipped["a20200101_08_r4_HPC_bogus"]

    def test_variants_of_analysed_epochs_can_be_asked_for(self, tmp_path):
        self.write_decode(tmp_path, "a20200101_02_r1_HPC")
        self.write_decode(tmp_path, "a20200101_02_r1_HPC_all")
        self.write_decode(tmp_path, "a20200101_04_r2_HPC_all")
        self.write_decode(
            tmp_path, "a20200101_02_r1_HPC_non_stationary", data_over_time=False
        )

        epochs, skipped = find_epochs(
            None,
            tmp_path,
            analysed_keys("a20200101_02_r1_HPC"),
            include_variants=True,
        )

        assert epochs == ["a20200101_02_r1_HPC", "a20200101_02_r1_HPC_all"]
        assert skipped == {
            "a20200101_04_r2_HPC_all": "not an analysed epoch in the epoch register",
            "a20200101_02_r1_HPC_non_stationary": "missing data_over_time",
        }

    def test_a_pending_run_says_why_it_left_an_analysed_decode_out(self, tmp_path):
        self.write_decode(tmp_path, "a20200101_02_r1_HPC")  # analysed
        self.write_decode(tmp_path, "a20200101_04_r2_HPC")  # pending

        epochs, skipped = find_epochs(
            None,
            tmp_path,
            analysed_keys("a20200101_04_r2_HPC"),
            left_out=runner.NOT_PENDING,
        )

        assert epochs == ["a20200101_04_r2_HPC"]
        assert skipped == {
            "a20200101_02_r1_HPC": "not a pending epoch in the epoch register"
        }

    def test_an_animal_restricts_the_search(self, tmp_path):
        for epoch_id in ("a20200101_02_r1_HPC", "b20200101_02_r1_HPC"):
            self.write_decode(tmp_path, epoch_id)

        epochs, skipped = find_epochs(
            "b",
            tmp_path,
            analysed_keys("a20200101_02_r1_HPC", "b20200101_02_r1_HPC"),
        )

        assert epochs == ["b20200101_02_r1_HPC"]
        assert skipped == {}


SIM_TIME = np.arange(2000) / SAMPLING_FREQUENCY
"""``build_tiling_tetrode``'s grid."""
SIM_EPOCH_IDS = (
    "sim20200101_02_r1_HPC",
    "sim20200101_04_r2_HPC",
    "sim20200101_06_r3_HPC",
    "sim20200102_02_r1_HPC",
    "sim20200102_04_r2_HPC",
)
"""Every simulated epoch these tests run."""


def write_swr_source(root, *, swrs=None, observed=None):
    """The SWR source of the simulated epochs: by default one SWR early on, where
    ``write_epoch`` also labels one, observed over the whole grid."""
    swrs = [(SIM_TIME[50], SIM_TIME[89])] if swrs is None else swrs
    observed = (SIM_TIME[0], SIM_TIME[-1]) if observed is None else observed
    return write_swr_inventory(
        root,
        dict.fromkeys(SIM_EPOCH_IDS, swrs),
        observed={epoch_id: [observed] for epoch_id in SIM_EPOCH_IDS},
    )


def use_swr_source(monkeypatch, root):
    monkeypatch.setattr(swr_inventory, "SWR_INVENTORY_ROOT", root)
    monkeypatch.setattr(swr_inventory, "SWR_INVENTORY_RUN_ID", RUN_ID)


@pytest.fixture(scope="module")
def swr_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("swr")
    write_swr_source(root)
    return root


@pytest.fixture(autouse=True)
def the_swr_source(swr_root, monkeypatch):
    """Every run reads its SWRs from the synthetic source."""
    use_swr_source(monkeypatch, swr_root)


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
            "day": int(epoch_id.split("_")[0][-8:]),
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
        "marks_loader": lambda key, cache, span, detector_sha256: (
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
        assert (table.swr_definition == swr_inventory.swr_source_name()).all()
        # the saved labels are complete, so nothing is unknown
        assert not (table.overlap_class == "unknown").any()

    def test_the_swrs_are_the_sources_not_the_decodes(
        self, decode_root, tmp_path, monkeypatch
    ):
        """The source has one SWR over the whole epoch; the decode's stored
        labels mark only an early one. Every classified candidate is an SWR."""
        use_swr_source(
            monkeypatch,
            write_swr_source(
                tmp_path / "swr", swrs=[(SIM_TIME[0], SIM_TIME[-1])]
            ).parent,
        )

        record = run(decode_root, tmp_path)
        table = read_table(record["outputs"]["events"])

        known = table[table.overlap_class != "unknown"]
        assert len(known) > 0
        assert known.is_swr.astype(bool).all()
        assert (known.overlap_class == "swr_contained").all()
        assert record["swr_source"]["name"] == swr_inventory.swr_source_name()
        assert record["swr_source"]["name"].startswith(f"kay@{RUN_ID}+")
        assert len(record["swr_source"]["events"]["sha256"]) == 64
        assert table.attrs["swr_source"] == swr_inventory.swr_source_name()

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
                marks_loader=lambda key, cache, span, detector_sha256: (
                    epoch.spike_times,
                    changed,
                ),
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
        assert len(record["marks"]["sha256"]) == 64
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

    def test_missing_lfp_reaches_the_table_as_unknown(self, tmp_path, monkeypatch):
        """A stretch the SWR source did not observe is missing LFP: every
        candidate touching it is ``unknown``, never ``non_swr``, and the
        observed table says so."""
        epoch = build_tiling_tetrode()
        root = tmp_path / "processed"
        root.mkdir()
        write_epoch(root, epoch)
        use_swr_source(
            monkeypatch,
            write_swr_source(
                tmp_path / "swr", observed=(SIM_TIME[1000], SIM_TIME[-1])
            ).parent,
        )

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

    def test_the_artifact_mask_is_missing_on_the_decode_grid(
        self, decode_root, tmp_path, monkeypatch
    ):
        """The source's artifact mask, on the decode grid, is missing in the
        likelihood: no window holding it is decoded, no candidate touches it,
        the observed table leaves it out, and the record counts it."""
        _, epoch = decode_root
        step_s = float(epoch.time[1000])
        artifacts = write_artifact_run(
            tmp_path / "artifacts",
            {EPOCH_ID: [("ca1", step_s, step_s)]},
            span=(float(epoch.time[0]), float(epoch.time[-1])),
        )
        monkeypatch.setattr(swr_inventory, "ARTIFACT_INTERVALS_ROOT", artifacts.parent)
        monkeypatch.setattr(swr_inventory, "ARTIFACT_INTERVALS_RUN_ID", artifacts.name)
        missing = (epoch.time >= step_s - 0.3) & (epoch.time <= step_s + 0.3)

        record = run(decode_root, tmp_path)

        candidates = joblib.load(record["outputs"]["candidates"])
        decoded = candidates.decisions.gates.posterior_usable
        stride, width = CONFIG.stride_n_samples, CONFIG.window_n_samples
        touching = np.array(
            [
                missing[k * stride : k * stride + width].any()
                for k in range(decoded.size)
            ]
        )
        np.testing.assert_array_equal(decoded, ~touching)
        assert len(candidates.runs) >= 2
        for first, last in candidates.runs:
            assert not missing[first * stride : last * stride + width].any()
        table = read_table(record["outputs"]["events"])
        expected_reasons = artifact_drop_reasons(
            table.event_start_s.to_numpy(),
            table.event_end_s.to_numpy(),
            pd.DataFrame({"start_time": [step_s - 0.3], "end_time": [step_s + 0.3]}),
        )
        assert table.artifact_drop_reason.tolist() == expected_reasons.tolist()
        # the candidates on either side of the mask are within its reach
        assert (table.artifact_drop_reason == "near_artifact").sum() >= 1
        assert record["artifacts"] == {
            "n_intervals": 1,
            "masked_s": pytest.approx(0.6),
            "n_missing_bins": int(missing.sum()),
            "n_missing_windows": int(touching.sum()),
            "n_dropped": {
                reason: int((table.artifact_drop_reason == reason).sum())
                for reason in ("artifact", "near_artifact", "overlaps_dropped_event")
            },
        }
        observed = read_table(record["outputs"]["observed"])
        windows = observed[observed.kind == "decoded_windows"]
        midpoints = candidates.midpoint_s
        first, last = np.flatnonzero(touching)[[0, -1]]
        np.testing.assert_array_equal(
            windows[["start_s", "end_s"]].to_numpy(),
            [
                [midpoints[0], midpoints[first - 1]],
                [midpoints[last + 1], midpoints[-1]],
            ],
        )

    def test_without_an_artifact_run_nothing_is_missing(self, decode_root, tmp_path):
        record = run(decode_root, tmp_path)

        assert record["artifacts"] == {
            "n_intervals": 0,
            "masked_s": 0.0,
            "n_missing_bins": 0,
            "n_missing_windows": 0,
            "n_dropped": {
                "artifact": 0,
                "near_artifact": 0,
                "overlaps_dropped_event": 0,
            },
        }
        table = read_table(record["outputs"]["events"])
        assert table.artifact_drop_reason.isna().all()


KEY = parse_epoch_identifier(EPOCH_ID)
SPAN_S = (10.0, 20.0)
"""A decode's first and last time, in seconds."""


class TestLoadMarks:
    def test_marks_come_from_the_project_loader(self, tmp_path, monkeypatch):
        """``get_hpc_marks``, then ``filter_spike_times`` over the decode's
        span, both as ``load_data`` calls them: a spike outside the span goes,
        an electrode left with none is dropped, and each mark keeps its spike."""
        assert detection_inputs.get_hpc_marks is spikes.get_hpc_marks
        assert detection_inputs.filter_spike_times is spikes.filter_spike_times
        raw_times = [
            np.array([5.0, 12.0, 15.0]),  # one spike before the span
            np.array([1.0, 25.0]),  # every spike outside it
            np.array([10.0, 18.0, 20.0]),  # all inside, on both ends too
        ]
        # each mark row holds its spike's time and its electrode
        raw_marks = [
            np.column_stack([times, np.full(times.size, float(electrode))])
            for electrode, times in enumerate(raw_times)
        ]
        calls = []

        def get_hpc_marks(nwb_file_name):
            calls.append(nwb_file_name)
            return raw_times, raw_marks

        monkeypatch.setattr(detection_inputs, "get_hpc_marks", get_hpc_marks)

        times, marks = detection_inputs.load_marks(KEY, tmp_path, SPAN_S, "a" * 64)

        assert calls == [KEY.nwb_file_name]
        assert [t.tolist() for t in times] == [[12.0, 15.0], [10.0, 18.0, 20.0]]
        assert [m[:, 1].tolist() for m in marks] == [[0.0, 0.0], [2.0, 2.0, 2.0]]
        for electrode_times, electrode_marks in zip(times, marks, strict=True):
            np.testing.assert_array_equal(electrode_marks[:, 0], electrode_times)

    def test_a_cache_from_another_decode_is_not_read(self, tmp_path, monkeypatch):
        """Review Focus 2. Two decodes of one epoch have different detector
        files, so neither reads the marks cached for the other; a decode still
        reads its own cache rather than the database."""
        calls = []

        def get_hpc_marks(nwb_file_name):
            calls.append(nwb_file_name)
            # each load's marks say which load made them
            return [np.array([15.0])], [np.full((1, 4), float(len(calls)))]

        monkeypatch.setattr(detection_inputs, "get_hpc_marks", get_hpc_marks)

        first = detection_inputs.load_marks(KEY, tmp_path, SPAN_S, "a" * 64)
        second = detection_inputs.load_marks(KEY, tmp_path, SPAN_S, "b" * 64)
        first_again = detection_inputs.load_marks(KEY, tmp_path, SPAN_S, "a" * 64)

        assert len(calls) == 2
        assert first[1][0][0, 0] == 1.0
        assert second[1][0][0, 0] == 2.0
        assert first_again[1][0][0, 0] == 1.0

    def test_an_april_rule_cache_is_never_read(self, tmp_path, monkeypatch):
        """Review Focus 2. The pilot cached marks made under the April decodes'
        rule as ``{epoch_id}_decode_time_marks.pkl``. They keep more spikes than
        the loader, so the encoding-mark guard could pass them; they are never
        read."""
        joblib.dump(
            ([np.array([12.0, 15.0])], [np.full((2, 4), -1.0)]),
            tmp_path / f"{EPOCH_ID}_decode_time_marks.pkl",
        )
        calls = []

        def get_hpc_marks(nwb_file_name):
            calls.append(nwb_file_name)
            return [np.array([15.0])], [np.full((1, 4), 7.0)]

        monkeypatch.setattr(detection_inputs, "get_hpc_marks", get_hpc_marks)

        times, marks = detection_inputs.load_marks(KEY, tmp_path, SPAN_S, "a" * 64)

        assert calls == [KEY.nwb_file_name]
        assert [t.tolist() for t in times] == [[15.0]]
        np.testing.assert_array_equal(marks[0], np.full((1, 4), 7.0))

    def test_an_interrupted_cache_write_leaves_no_cache(self, tmp_path, monkeypatch):
        """A cache write cut short leaves nothing under the cache's name: half a
        cache would be read, and fail, on every rerun of the epoch. The cache
        is written through a temporary file, so a worker killed part way
        leaves at most that file."""
        monkeypatch.setattr(
            detection_inputs,
            "get_hpc_marks",
            lambda nwb_file_name: ([np.array([15.0])], [np.full((1, 4), 7.0)]),
        )

        def interrupted_dump(value, filename):
            Path(filename).write_bytes(b"half a pickle")
            raise OSError("simulated: the write was cut short")

        monkeypatch.setattr(detection_inputs.joblib, "dump", interrupted_dump)

        with pytest.raises(OSError, match="cut short"):
            detection_inputs.load_marks(KEY, tmp_path, SPAN_S, "a" * 64)

        assert list(tmp_path.iterdir()) == []

    def test_the_record_names_the_marks_rule(self, decode_root, tmp_path):
        """The loader is given the decode file's hash, and the record names the
        rule, that hash, the cache file and the marks' own hash."""
        root, epoch = decode_root
        detector_sha256 = file_sha256(input_paths(EPOCH_ID, root)["detector"])
        given = []

        def marks_loader(key, cache_dir, span_s, sha256):
            given.append(sha256)
            return epoch.spike_times, epoch.spike_waveform_features

        record = run(decode_root, tmp_path, shuffle=False, marks_loader=marks_loader)

        assert given == [detector_sha256]
        assert record["inputs"]["detector"]["sha256"] == detector_sha256
        assert record["marks"] == {
            "rule": detection_inputs.MARKS_RULE,
            "detector_sha256": detector_sha256,
            "cache": f"{EPOCH_ID}_loader_{detector_sha256[:16]}_marks.pkl",
            "sha256": detection_inputs.marks_sha256(
                epoch.spike_times, epoch.spike_waveform_features
            ),
        }


PARALLEL_EPOCH_IDS = (
    "sim20200101_02_r1_HPC",
    "sim20200101_04_r2_HPC",
    "sim20200101_06_r3_HPC",
)
"""More epochs than the parallel driver's two workers."""
FAILING_EPOCH_ID = PARALLEL_EPOCH_IDS[1]


def read_manifest(tmp_path):
    return json.loads(
        (tmp_path / "memoryless" / "cli-run" / "manifest.json").read_text()
    )


class TestMain:
    @pytest.fixture(autouse=True)
    def marks_only_from_the_cache(self, monkeypatch):
        """Every run here finds its marks cached; a miss fails the epoch
        instead of reaching the database."""

        def get_hpc_marks(nwb_file_name):
            raise AssertionError(f"the marks cache missed for {nwb_file_name}")

        monkeypatch.setattr(detection_inputs, "get_hpc_marks", get_hpc_marks)

    @pytest.fixture(autouse=True)
    def no_cuda_cluster(self, monkeypatch):
        """No test here starts a CUDA cluster, whether or not dask_cuda imports."""

        def refuse(**kwargs):
            raise AssertionError(f"a CUDA cluster was started with {kwargs}")

        monkeypatch.setattr(runner, "cuda_cluster_class", lambda: refuse)

    @pytest.fixture(autouse=True)
    def no_register(self, monkeypatch):
        """``--all`` and ``--animal`` read the epoch register, which reads the
        database; a test that runs them names its analysed epochs instead."""

        def epochs_to_run(pending_only=False):
            raise AssertionError("this test did not say which epochs are analysed")

        monkeypatch.setattr(runner, "epochs_to_run", epochs_to_run)

    @staticmethod
    def analyse(monkeypatch, *epoch_ids, pending=()):
        """The register analyses these epochs, holds `pending` pending, and no
        other: ``epochs_to_run`` returns one or the other."""

        def epochs_to_run(pending_only=False):
            return analysed_frame(*(pending if pending_only else epoch_ids))

        monkeypatch.setattr(runner, "epochs_to_run", epochs_to_run)

    @pytest.fixture(autouse=True)
    def in_tmp_path(self, tmp_path, monkeypatch):
        """What the runner writes relative to the working directory, such as
        the per-epoch logs, lands in the test's own folder."""
        monkeypatch.chdir(tmp_path)

    @pytest.fixture
    def parallel(self, tmp_path, monkeypatch):
        """Three saved decodes with their marks cached, and the cluster a device
        list asks for replaced by two worker threads in this process.

        Yields the arguments each ``LocalCUDACluster`` was made with.
        """
        epoch = build_tiling_tetrode()
        root = tmp_path / "processed"
        root.mkdir()
        for epoch_id in PARALLEL_EPOCH_IDS:
            write_epoch(root, epoch, epoch_id)
            cache_marks(root, epoch, tmp_path / "cache", epoch_id)
        self.analyse(monkeypatch, *PARALLEL_EPOCH_IDS)
        clusters = []

        def in_process_cluster(**kwargs):
            clusters.append(kwargs)
            return LocalCluster(
                n_workers=2,
                processes=False,
                dashboard_address=":0",
                local_directory=kwargs["local_directory"],
            )

        monkeypatch.setattr(runner, "cuda_cluster_class", lambda: in_process_cluster)
        # the scheduler keeps its scratch in dask's temporary directory,
        # whatever the workers are given
        with dask.config.set({"temporary-directory": str(tmp_path / "dask-temporary")}):
            yield clusters

    def parallel_cli(self, tmp_path):
        """The runner over the three decodes, given a device list."""
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
                str(tmp_path / "processed"),
                "--output-root",
                str(tmp_path / "memoryless"),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--cuda_visible_devices",
                "0,1",
                "--local_scratch_dir",
                str(tmp_path / "scratch"),
            ]
        )

    def cli(self, decode_root, tmp_path, *extra):
        """The runner through its command line, its marks already cached."""
        root, epoch = decode_root
        cache_dir = tmp_path / "cache"
        cache_marks(root, epoch, cache_dir)
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

    def test_one_failing_epoch_does_not_stop_the_others(
        self, tmp_path, capsys, monkeypatch
    ):
        """The second epoch's data_over_time is unreadable: it is recorded as
        failed with its error, the first is written, and the run exits non-zero."""
        self.analyse(monkeypatch, "sim20200101_02_r1_HPC", "sim20200101_04_r2_HPC")
        epoch = build_tiling_tetrode()
        root = tmp_path / "processed"
        root.mkdir()
        write_epoch(root, epoch, "sim20200101_02_r1_HPC")
        write_epoch(root, epoch, "sim20200101_04_r2_HPC", corrupt=True)
        cache_dir = tmp_path / "cache"
        cache_marks(root, epoch, cache_dir)
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

    def cli_over_sim(self, root, tmp_path, *extra):
        main(
            [
                *extra,
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
                str(tmp_path / "cache"),
            ]
        )

    def test_a_decode_the_register_does_not_analyse_is_left_out_and_told(
        self, tmp_path, capsys, monkeypatch
    ):
        self.analyse(monkeypatch, "sim20200101_02_r1_HPC")
        epoch = build_tiling_tetrode()
        root = tmp_path / "processed"
        root.mkdir()
        for epoch_id in ("sim20200101_02_r1_HPC", "sim20200101_04_r2_HPC"):
            write_epoch(root, epoch, epoch_id)
            cache_marks(root, epoch, tmp_path / "cache", epoch_id)

        self.cli_over_sim(root, tmp_path)

        out = capsys.readouterr().out
        assert (
            "left out sim20200101_04_r2_HPC: not an analysed epoch in the epoch "
            "register"
        ) in out
        epochs = read_manifest(tmp_path)["epochs"]
        assert epochs["sim20200101_02_r1_HPC"]["status"] == "ok"
        assert epochs["sim20200101_04_r2_HPC"] == {
            "status": "skipped",
            "reason": "not an analysed epoch in the epoch register",
        }

    def test_an_analysed_epoch_without_its_decode_fails_the_run(
        self, tmp_path, capsys, monkeypatch
    ):
        """It is recorded as failed, so a run never covers fewer epochs than the
        register analyses without saying so."""
        self.analyse(monkeypatch, "sim20200101_02_r1_HPC", "sim20200101_04_r2_HPC")
        epoch = build_tiling_tetrode()
        root = tmp_path / "processed"
        root.mkdir()
        write_epoch(root, epoch, "sim20200101_02_r1_HPC")
        cache_marks(root, epoch, tmp_path / "cache", "sim20200101_02_r1_HPC")
        input_paths("sim20200101_04_r2_HPC", root)["detector"].write_bytes(b"")

        with pytest.raises(SystemExit, match="1 epochs failed: sim20200101_04_r2_HPC"):
            self.cli_over_sim(root, tmp_path)

        epochs = read_manifest(tmp_path)["epochs"]
        assert epochs["sim20200101_02_r1_HPC"]["status"] == "ok"
        failed = epochs["sim20200101_04_r2_HPC"]
        assert failed["status"] == "failed"
        assert "missing data_over_time" in failed["error"]

    def test_pending_only_runs_the_pending_epochs_from_their_staged_decodes(
        self, tmp_path, capsys, monkeypatch
    ):
        """``--pending-only --root <staging>``: the staged pending decode runs; an
        analysed decode under the same root is left out, with why."""
        self.analyse(
            monkeypatch, "sim20200101_02_r1_HPC", pending=("sim20200101_04_r2_HPC",)
        )
        epoch = build_tiling_tetrode()
        root = tmp_path / "processed"
        root.mkdir()
        for epoch_id in ("sim20200101_02_r1_HPC", "sim20200101_04_r2_HPC"):
            write_epoch(root, epoch, epoch_id)
            cache_marks(root, epoch, tmp_path / "cache", epoch_id)

        self.cli_over_sim(root, tmp_path, "--pending-only")

        epochs = read_manifest(tmp_path)["epochs"]
        assert epochs["sim20200101_04_r2_HPC"]["status"] == "ok"
        assert epochs["sim20200101_02_r1_HPC"] == {
            "status": "skipped",
            "reason": "not a pending epoch in the epoch register",
        }

    def test_date_and_epoch_narrow_what_runs(self, tmp_path, monkeypatch):
        self.analyse(
            monkeypatch,
            "sim20200101_02_r1_HPC",
            "sim20200102_02_r1_HPC",
            "sim20200102_04_r2_HPC",
        )
        epoch = build_tiling_tetrode()
        root = tmp_path / "processed"
        root.mkdir()
        cache_dir = tmp_path / "cache"
        for epoch_id in (
            "sim20200101_02_r1_HPC",
            "sim20200102_02_r1_HPC",
            "sim20200102_04_r2_HPC",
        ):
            write_epoch(root, epoch, epoch_id)
            cache_marks(root, epoch, cache_dir, epoch_id)

        main(
            [
                "--animal",
                "sim",
                "--date",
                "20200102",
                "--epoch",
                "02_r1",
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

        epochs = read_manifest(tmp_path)["epochs"]
        assert list(epochs) == ["sim20200102_02_r1_HPC"]
        assert epochs["sim20200102_02_r1_HPC"]["status"] == "ok"

    def test_parallel_run_records_each_epoch_and_a_failure(
        self, parallel, tmp_path, monkeypatch
    ):
        """One task per epoch, on the cluster the device list asks for: the
        failing epoch is recorded with its error, the others are written, each
        logs to its own file, and the run exits non-zero."""
        run_one = runner.run_one

        def fails_for_one(epoch_id, **kwargs):
            if epoch_id == FAILING_EPOCH_ID:
                raise RuntimeError(f"simulated failure in {epoch_id}")
            return run_one(epoch_id, **kwargs)

        monkeypatch.setattr(runner, "run_one", fails_for_one)

        with pytest.raises(SystemExit, match=f"1 epochs failed: {FAILING_EPOCH_ID};"):
            self.parallel_cli(tmp_path)

        assert parallel == [
            {
                "CUDA_VISIBLE_DEVICES": "0,1",
                "local_directory": str(tmp_path / "scratch"),
            }
        ]
        epochs = read_manifest(tmp_path)["epochs"]
        assert set(epochs) == set(PARALLEL_EPOCH_IDS)
        failed = epochs[FAILING_EPOCH_ID]
        assert failed["status"] == "failed"
        assert failed["error"] == repr(
            RuntimeError(f"simulated failure in {FAILING_EPOCH_ID}")
        )
        for epoch_id in set(PARALLEL_EPOCH_IDS) - {FAILING_EPOCH_ID}:
            assert epochs[epoch_id]["status"] == "ok"
            outputs = epochs[epoch_id]["outputs"]
            assert outputs and all(Path(path).exists() for path in outputs.values())
        for epoch_id in PARALLEL_EPOCH_IDS:
            assert (tmp_path / "logs" / f"{epoch_id}_memoryless.log").exists()

    def test_only_the_main_thread_writes_the_manifest(
        self, parallel, tmp_path, monkeypatch
    ):
        """Once before the epochs run and once as each finishes, while the
        epochs themselves run in the workers' threads."""
        writers, runners = [], []
        write_manifest, run_one = runner.write_manifest, runner.run_one

        def recording_write_manifest(path, manifest):
            writers.append(threading.current_thread().name)
            write_manifest(path, manifest)

        def recording_run_one(epoch_id, **kwargs):
            runners.append(threading.current_thread().name)
            return run_one(epoch_id, **kwargs)

        monkeypatch.setattr(runner, "write_manifest", recording_write_manifest)
        monkeypatch.setattr(runner, "run_one", recording_run_one)

        self.parallel_cli(tmp_path)

        assert len(runners) == len(PARALLEL_EPOCH_IDS)
        assert "MainThread" not in runners
        assert writers == ["MainThread"] * (1 + len(PARALLEL_EPOCH_IDS))
        assert all(
            record["status"] == "ok"
            for record in read_manifest(tmp_path)["epochs"].values()
        )

    def test_a_parallel_rerun_runs_only_what_is_not_written(
        self, parallel, tmp_path, monkeypatch
    ):
        """As a serial rerun does: the failed epoch is retried and the written
        ones are not run again."""
        run_one = runner.run_one
        attempts = []

        def fails_the_first_time(epoch_id, **kwargs):
            attempts.append(epoch_id)
            if epoch_id == FAILING_EPOCH_ID and attempts.count(epoch_id) == 1:
                raise RuntimeError("simulated transient failure")
            return run_one(epoch_id, **kwargs)

        monkeypatch.setattr(runner, "run_one", fails_the_first_time)
        with pytest.raises(SystemExit, match="1 epochs failed"):
            self.parallel_cli(tmp_path)
        first = read_manifest(tmp_path)["epochs"]

        self.parallel_cli(tmp_path)

        assert sorted(attempts) == sorted([*PARALLEL_EPOCH_IDS, FAILING_EPOCH_ID])
        again = read_manifest(tmp_path)["epochs"]
        assert all(record["status"] == "ok" for record in again.values())
        for epoch_id in set(PARALLEL_EPOCH_IDS) - {FAILING_EPOCH_ID}:
            assert again[epoch_id] == first[epoch_id]

    def test_a_device_list_without_dask_cuda_is_refused(
        self, decode_root, tmp_path, monkeypatch
    ):
        """Before anything is written."""
        monkeypatch.setattr(runner, "cuda_cluster_class", lambda: None)

        with pytest.raises(SystemExit, match="dask_cuda"):
            self.cli(decode_root, tmp_path, "--cuda_visible_devices", "0")

        assert not (tmp_path / "memoryless" / "cli-run" / "manifest.json").exists()
        assert not (tmp_path / "memoryless").exists()

    @pytest.mark.parametrize(
        "field", ["uv_lock_sha256", "ripple_detection_commit", "cuda_plugin_version"]
    )
    def test_settings_record_the_environment(
        self, decode_root, tmp_path, monkeypatch, field
    ):
        """The decoders' provenance record without ``dirty``, which the run id
        already carries. A rerun under another environment is refused, as one
        under another library version is."""
        environment = {
            name: value
            for name, value in provenance_record(ROOT_DIR).items()
            if name != "dirty"
        }

        settings = runner.run_settings(7, True, N_SHUFFLES, CONFIG, "test")

        assert field in settings["environment"]
        assert settings["environment"] == environment

        self.cli(decode_root, tmp_path)
        monkeypatch.setattr(
            runner,
            "provenance_record",
            lambda repo: provenance_record(repo) | {field: "changed"},
        )

        with pytest.raises(SystemExit, match="other settings"):
            self.cli(decode_root, tmp_path)

    def test_settings_record_that_runs_are_bit_reproducible(self):
        """What a run states about itself: with deterministic XLA and Triton GEMM
        off two runs are bit-for-bit equal (R12 Triton spike). The earlier
        measurements are kept as history, each with the flags it used."""
        settings = runner.run_settings(7, True, N_SHUFFLES, CONFIG, "test")

        known = settings["known_limits"]["bit_reproducible"]
        assert known["value"] is True
        measured = known["measured"]
        # the spike's flags, in the order they were set
        assert measured.index(TRITON_OFF) < measured.index(DETERMINISTIC)
        assert "appended by the runner" in measured
        # the epoch R12 lost runs, twice, with equal outputs
        assert "chimi20200224_08_r4_HPC" in measured
        assert "233 candidates, 11 replay, 7 passing" in measured
        assert "bit-for-bit equal" in measured
        assert "only epoch run twice with Triton GEMM off" in measured
        # a passing epoch, against the R12 smoke test's output (Triton on)
        assert "On one passing epoch" in measured
        assert "chimi20200225_02_r1_HPC" in measured
        assert "237 candidates, 4 replay, 3 passing" in measured
        assert "38 spread and 73 jump" in measured
        assert "about 291,000 windows" in measured
        assert "4:56" in measured
        assert "4:20" in measured
        # the candidates' spread differs as much as the earlier no-flag run's did
        assert "spread_cm differs in all 291,301 windows" in measured
        assert "max relative 3.5e-3" in measured
        assert "jump_cm differs in 2,769 windows" in measured
        assert "1.7e-4" in measured
        assert "null_jump_cm in 7 values (max relative 1.8e-2)" in measured
        # and the Triton-off output is close to the earlier run with neither flag
        assert "2026-10-10 run with neither flag" in measured
        assert "0 gate flips" in measured
        assert "4.2e-4" in measured
        assert "the event bounds are the same" in measured
        assert "the same windows as the 111 flips" in measured

    def test_settings_keep_the_earlier_measurements_with_their_flags(self):
        history = runner.run_settings(7, True, N_SHUFFLES, CONFIG, "test")[
            "known_limits"
        ]["bit_reproducible"]["history"]

        # R12 smoke test: deterministic ops only, so Triton GEMM was on
        assert "2026-10-10" in history
        assert "Triton GEMM on" in history
        assert "111 of 295,369 windows" in history
        assert "5:10" in history
        # Task 17a: neither flag
        assert "2026-10-09" in history
        assert "neither flag" in history
        assert "1.2e-4" in history
        assert "j1620210706_02_r1" in history
        assert "every event-level output equal" in history

    def test_settings_say_that_r12_ran_with_triton_gemm_and_lost_epochs(self):
        before = runner.run_settings(7, True, N_SHUFFLES, CONFIG, "test")[
            "known_limits"
        ]["bit_reproducible"]["runs_before_this_change"]

        assert "c55cf318d199-seed20260922" in before
        assert "Triton GEMM on" in before
        assert "3 of 242 epochs" in before
        for epoch_id in (
            "chimi20200224_08_r4_HPC",
            "chimi20200227_04_r2_HPC",
            "chimi20200228_02_r1_HPC",
        ):
            assert epoch_id in before
        assert (
            "CANCELLED: Too small divisible part of the contracting dimension" in before
        )
        assert "clusterless_kde.py:89" in before
        # the claim is as wide as the evidence: one epoch of three, one compared
        assert (
            "Of these three, only chimi20200224_08_r4_HPC has been run with Triton GEMM off"
            in before
        )
        assert "chimi20200227_04_r2_HPC and chimi20200228_02_r1_HPC" in before
        assert "have not been run yet" in before
        assert "the one epoch compared" in before

    def test_settings_give_the_reason_for_triton_off_as_unconfirmed(self):
        """The mechanism is the working explanation, so the record says so; the
        hedge must not be lost when the text is edited."""
        before = runner.run_settings(7, True, N_SHUFFLES, CONFIG, "test")[
            "known_limits"
        ]["bit_reproducible"]["runs_before_this_change"]

        assert "the working explanation; not confirmed" in before
        assert "turn off XLA's autotuning" in before
        assert "the spike tested the fix, not the mechanism" in before

    @pytest.mark.parametrize(("inherited", "expected"), XLA_FLAGS_CASES)
    def test_settings_record_the_xla_flags(self, monkeypatch, inherited, expected):
        """The XLA_FLAGS every process that runs JAX gets: the user's, with the
        deterministic flag appended once."""
        set_xla_flags(monkeypatch, inherited)

        settings = runner.run_settings(7, True, N_SHUFFLES, CONFIG, "test")

        assert settings["xla_flags"] == expected

    def test_settings_record_both_flags_by_the_last_token_rule(self, monkeypatch):
        """Whatever the user gave, the last occurrence of each required flag in
        the record has the required value."""
        set_xla_flags(
            monkeypatch,
            "--xla_gpu_deterministic_ops=false --xla_gpu_enable_triton_gemm=true",
        )

        recorded = runner.run_settings(7, True, N_SHUFFLES, CONFIG, "test")[
            "xla_flags"
        ].split()

        assert [f for f in recorded if f.startswith("--xla_gpu_deterministic_ops=")][
            -1
        ] == DETERMINISTIC
        assert [f for f in recorded if f.startswith("--xla_gpu_enable_triton_gemm=")][
            -1
        ] == TRITON_OFF

    def test_a_rerun_with_other_xla_flags_is_refused(
        self, decode_root, tmp_path, monkeypatch
    ):
        """XLA flags change what the GPU computes, as a library version can."""
        monkeypatch.delenv("XLA_FLAGS", raising=False)
        self.cli(decode_root, tmp_path)
        monkeypatch.setenv("XLA_FLAGS", "--xla_dump_hlo_as_text")

        with pytest.raises(SystemExit, match="other settings"):
            self.cli(decode_root, tmp_path)

    def test_a_run_recorded_with_deterministic_ops_only_is_not_resumed(
        self, decode_root, tmp_path, monkeypatch
    ):
        """R12 ran with Triton GEMM on and recorded only the deterministic flag.
        Under the same run id, the runner now asks for both, so the settings
        differ and the resume is refused."""
        monkeypatch.delenv("XLA_FLAGS", raising=False)
        self.cli(decode_root, tmp_path)
        manifest_path = tmp_path / "memoryless" / "cli-run" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["settings"]["xla_flags"] == f"{DETERMINISTIC} {TRITON_OFF}"
        manifest["settings"]["xla_flags"] = DETERMINISTIC
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(SystemExit, match="other settings"):
            self.cli(decode_root, tmp_path)

    # --- open files: Spyglass's NWB cache and the open-file limit --------------
    COUNTS = dict.fromkeys(
        (
            "n_windows",
            "n_windows_speed_unknown",
            "n_candidates",
            "n_replay",
            "n_swr_unknown",
            "n_shuffled",
            "n_shuffle_untestable",
            "n_shuffle_pass",
        ),
        0,
    )

    @staticmethod
    def recording_release(monkeypatch) -> list[bool]:
        """Replace ``release_nwb_files``; return the list each call appends to."""
        closed: list[bool] = []

        def release_nwb_files() -> int:
            closed.append(True)
            return 14

        monkeypatch.setattr(runner, "release_nwb_files", release_nwb_files)
        return closed

    @pytest.mark.parametrize("outcome", ["ok", "raises"])
    def test_each_epoch_task_closes_the_nwb_files_spyglass_opened(
        self, tmp_path, monkeypatch, outcome
    ):
        """A GPU worker runs one epoch at a time, so when its epoch ends, pass or
        fail, it closes the NWB files Spyglass cached for it and logs how many."""
        closed = self.recording_release(monkeypatch)

        def run_one(epoch_id, **kwargs):
            if outcome == "raises":
                raise RuntimeError("simulated: the marks could not be read")
            return {"status": "ok", **self.COUNTS}

        monkeypatch.setattr(runner, "run_one", run_one)

        record = runner._run_one_logged(
            EPOCH_ID,
            {"provenance": {"git_commit": "test"}, "run_id": "test-run"},
            tmp_path / "logs",
        )

        assert record["status"] == ("ok" if outcome == "ok" else "failed")
        assert closed == [True]
        log = (tmp_path / "logs" / f"{EPOCH_ID}_memoryless.log").read_text()
        assert any(
            line.endswith("closed 14 NWB files Spyglass held open")
            for line in log.splitlines()
        )

    def test_each_epoch_run_in_this_process_closes_the_nwb_files(
        self, tmp_path, monkeypatch
    ):
        """Without a device list the epochs run one after another here, so each
        closes what Spyglass opened for it, whether it is written or fails."""
        closed = self.recording_release(monkeypatch)
        failing = "sim20200101_04_r2_HPC"

        def run_one(epoch_id, **kwargs):
            if epoch_id == failing:
                raise RuntimeError("simulated: the marks could not be read")
            return {"status": "ok", **self.COUNTS}

        monkeypatch.setattr(runner, "run_one", run_one)
        output_root = tmp_path / "memoryless"

        with pytest.raises(SystemExit, match=f"1 epochs failed: {failing};"):
            main(
                [
                    "--epoch-id",
                    EPOCH_ID,
                    "--epoch-id",
                    failing,
                    "--run-seed",
                    "7",
                    "--n-shuffles",
                    str(N_SHUFFLES),
                    "--run-id",
                    "cli-run",
                    "--root",
                    str(tmp_path / "processed"),
                    "--output-root",
                    str(output_root),
                    "--cache-dir",
                    str(tmp_path / "cache"),
                ]
            )

        assert closed == [True, True]
        log = (output_root / "cli-run" / "run.log").read_text().splitlines()
        assert sum(line.endswith("closed 14 NWB files Spyglass held open")
                   for line in log) == 2  # fmt: skip

    @pytest.mark.parametrize(("inherited", "expected"), XLA_FLAGS_CASES)
    def test_epochs_run_in_this_process_run_xla_deterministically(
        self, tmp_path, monkeypatch, inherited, expected
    ):
        """Without a device list this process runs JAX itself, so it holds the
        flag while each epoch runs (XLA reads it when JAX starts its backend, in
        the first epoch). The manifest records the same flags, and this
        process's environment is restored afterwards."""
        set_xla_flags(monkeypatch, inherited)
        self.recording_release(monkeypatch)
        seen = []

        def run_one(epoch_id, **kwargs):
            seen.append(os.environ.get("XLA_FLAGS"))
            return {"status": "ok", **self.COUNTS}

        monkeypatch.setattr(runner, "run_one", run_one)

        main(
            [
                "--epoch-id",
                EPOCH_ID,
                "--epoch-id",
                "sim20200101_04_r2_HPC",
                "--run-seed",
                "7",
                "--n-shuffles",
                str(N_SHUFFLES),
                "--run-id",
                "cli-run",
                "--root",
                str(tmp_path / "processed"),
                "--output-root",
                str(tmp_path / "memoryless"),
                "--cache-dir",
                str(tmp_path / "cache"),
            ]
        )

        assert seen == [expected, expected]
        assert read_manifest(tmp_path)["settings"]["xla_flags"] == expected
        assert os.environ.get("XLA_FLAGS") == inherited

    def test_main_raises_the_open_file_limit_before_its_cluster(
        self, parallel, tmp_path, monkeypatch
    ):
        """The workers inherit the limit only if it is raised before they start;
        the run log says what it was raised from and to."""
        self.recording_release(monkeypatch)
        clusters_at_raise = []

        def raise_open_file_limit():
            clusters_at_raise.append(len(parallel))
            return 1024, 65536

        monkeypatch.setattr(runner, "raise_open_file_limit", raise_open_file_limit)

        self.parallel_cli(tmp_path)

        assert clusters_at_raise == [0]
        assert len(parallel) == 1
        run_log = (tmp_path / "memoryless" / "cli-run" / "run.log").read_text()
        assert "open-file soft limit 1024 -> 65536" in run_log

    def test_the_cluster_is_given_the_xla_flags_the_settings_record(
        self, parallel, tmp_path, monkeypatch
    ):
        """The workers' XLA_FLAGS and the manifest's come from one value."""
        monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_deterministic_ops=false")
        self.recording_release(monkeypatch)
        monkeypatch.setattr(
            runner,
            "run_one",
            lambda epoch_id, **kwargs: {"status": "ok", **self.COUNTS},
        )
        given = []
        cuda_cluster = runner.cuda_cluster

        def recording_cuda_cluster(cuda_visible_devices, local_directory, xla_flags):
            given.append(xla_flags)
            return cuda_cluster(cuda_visible_devices, local_directory, xla_flags)

        monkeypatch.setattr(runner, "cuda_cluster", recording_cuda_cluster)

        self.parallel_cli(tmp_path)

        assert given == [read_manifest(tmp_path)["settings"]["xla_flags"]]
        assert given == [
            f"--xla_gpu_deterministic_ops=false {DETERMINISTIC} {TRITON_OFF}"
        ]


class TestRunEpochs:
    def test_an_error_its_worker_cannot_return_is_recorded(self, tmp_path, monkeypatch):
        """A worker whose process dies cannot return its epoch's failure; the
        error its task raises is recorded instead, and the rest still finish."""

        def task(epoch_id, run_one_kwargs, log_directory):
            if epoch_id == FAILING_EPOCH_ID:
                raise RuntimeError("simulated: the worker died")
            return {"status": "ok"}

        monkeypatch.setattr(runner, "_run_one_logged", task)
        monkeypatch.chdir(tmp_path)
        with dask.config.set({"temporary-directory": str(tmp_path / "dask-temporary")}):
            records = dict(
                runner.run_epochs(
                    list(PARALLEL_EPOCH_IDS),
                    {"provenance": {"git_commit": "test"}},
                    lambda: LocalCluster(
                        n_workers=2,
                        processes=False,
                        dashboard_address=":0",
                        local_directory=str(tmp_path / "scratch"),
                    ),
                )
            )

        assert set(records) == set(PARALLEL_EPOCH_IDS)
        failed = records.pop(FAILING_EPOCH_ID)
        assert failed["status"] == "failed"
        assert failed["error"] == repr(RuntimeError("simulated: the worker died"))
        assert failed["git_commit"] == "test"
        assert all(record == {"status": "ok"} for record in records.values())


def _thread_environ() -> dict[str, str | None]:
    """In a worker: its BLAS and OpenMP thread variables, and the trim setting
    dask's nanny also sets before spawning it."""
    import os

    names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MALLOC_TRIM_THRESHOLD_",
    )
    return {name: os.environ.get(name) for name in names}


class TestWorkerThreads:
    """dask's nanny sets OMP/MKL/OPENBLAS_NUM_THREADS to 1 in every worker it
    spawns, so the shuffles' matrix products ran on one core each (R4: 18 h
    against 5-6 h). The runner's GPU cluster gives each worker its share of
    the CPUs instead."""

    @pytest.mark.parametrize("inherited", [None, "1"])
    def test_each_worker_gets_its_share_of_the_cpus(
        self, tmp_path, monkeypatch, inherited
    ):
        """Whatever this process's environment says: dask's nanny lets an
        inherited variable override its configuration, and a nanny that has
        spawned a worker leaves its own (1) in its parent's environment."""
        for name in runner.THREAD_VARIABLES:
            if inherited is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, inherited)
        made = []

        def worker_processes(**kwargs):
            # LocalCUDACluster's stand-in: real worker processes, spawned by
            # nannies, so the variables are set as the GPU workers' are
            made.append(kwargs)
            return LocalCluster(
                n_workers=1,
                threads_per_worker=1,
                processes=True,
                dashboard_address=":0",
                local_directory=kwargs["local_directory"],
            )

        monkeypatch.setattr(runner, "cuda_cluster_class", lambda: worker_processes)
        monkeypatch.setattr(runner, "available_cpus", lambda: 13)
        with dask.config.set({"temporary-directory": str(tmp_path / "dask-temporary")}):
            with (
                runner.cuda_cluster(
                    "3,4,6", str(tmp_path / "scratch"), GIVEN_XLA_FLAGS
                ) as cluster,
                Client(cluster) as client,
            ):
                seen = client.run(_thread_environ)

        assert made == [
            {
                "CUDA_VISIBLE_DEVICES": "3,4,6",
                "local_directory": str(tmp_path / "scratch"),
            }
        ]
        assert list(seen.values()) == [
            {
                "OMP_NUM_THREADS": "4",  # 13 CPUs shared by 3 workers
                "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "MALLOC_TRIM_THRESHOLD_": "65536",
            }
        ]
        # this process's own environment is left as it was
        assert {name: os.environ.get(name) for name in runner.THREAD_VARIABLES} == (
            dict.fromkeys(runner.THREAD_VARIABLES, inherited)
        )

    def test_a_worker_never_gets_fewer_than_one_thread(self):
        assert runner.worker_thread_environ(n_workers=8, n_cpus=4) == {
            "OMP_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1,
            "OPENBLAS_NUM_THREADS": 1,
        }

    def test_the_cpus_are_the_ones_this_process_may_run_on(self):
        assert runner.available_cpus() == len(os.sched_getaffinity(0))


def _xla_flags() -> str | None:
    """In a worker: the XLA_FLAGS it was spawned with."""
    import os

    return os.environ.get("XLA_FLAGS")


class TestWorkerXlaFlags:
    """Two runs on the GPU differ unless XLA runs deterministically (R12 smoke
    test, 2026-10-10). XLA reads XLA_FLAGS when JAX starts its backend, so the
    flag must be in a worker's environment when its nanny spawns it."""

    @pytest.mark.parametrize("inherited", [None, "--xla_gpu_deterministic_ops=false"])
    def test_each_worker_gets_the_xla_flags_it_is_given(
        self, tmp_path, monkeypatch, inherited
    ):
        """Whatever this process's environment says: each worker gets the
        XLA_FLAGS ``cuda_cluster`` is given (``main`` gives it the ones the
        settings record), and this process's own are left as they were."""
        set_xla_flags(monkeypatch, inherited)

        def worker_processes(**kwargs):
            # LocalCUDACluster's stand-in: a real worker process, spawned by a
            # nanny, so its environment is set as the GPU workers' is
            return LocalCluster(
                n_workers=1,
                threads_per_worker=1,
                processes=True,
                dashboard_address=":0",
                local_directory=kwargs["local_directory"],
            )

        monkeypatch.setattr(runner, "cuda_cluster_class", lambda: worker_processes)
        with dask.config.set({"temporary-directory": str(tmp_path / "dask-temporary")}):
            with (
                runner.cuda_cluster(
                    "3,4,6", str(tmp_path / "scratch"), GIVEN_XLA_FLAGS
                ) as cluster,
                Client(cluster) as client,
            ):
                seen = client.run(_xla_flags)

        assert list(seen.values()) == [GIVEN_XLA_FLAGS]
        assert os.environ.get("XLA_FLAGS") == inherited

    @pytest.mark.parametrize(
        "inherited",
        [
            None,
            "--xla_gpu_deterministic_ops=false",
            "--xla_gpu_enable_triton_gemm=true",
            f"{DETERMINISTIC} --xla_gpu_enable_triton_gemm=true",
        ],
    )
    def test_each_worker_carries_both_flags(self, tmp_path, monkeypatch, inherited):
        """The workers get what ``xla_flags`` works out from this process's
        environment: deterministic ops on and Triton GEMM off, last."""
        set_xla_flags(monkeypatch, inherited)

        def worker_processes(**kwargs):
            return LocalCluster(
                n_workers=1,
                threads_per_worker=1,
                processes=True,
                dashboard_address=":0",
                local_directory=kwargs["local_directory"],
            )

        monkeypatch.setattr(runner, "cuda_cluster_class", lambda: worker_processes)
        with dask.config.set({"temporary-directory": str(tmp_path / "dask-temporary")}):
            with (
                runner.cuda_cluster(
                    "3,4,6", str(tmp_path / "scratch"), runner.xla_flags()
                ) as cluster,
                Client(cluster) as client,
            ):
                (seen,) = client.run(_xla_flags).values()

        flags = seen.split()
        assert [f for f in flags if f.startswith("--xla_gpu_deterministic_ops=")][
            -1
        ] == DETERMINISTIC
        assert [f for f in flags if f.startswith("--xla_gpu_enable_triton_gemm=")][
            -1
        ] == TRITON_OFF
        assert os.environ.get("XLA_FLAGS") == inherited
