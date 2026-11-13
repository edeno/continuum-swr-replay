"""The artifact run folder: one cores file per epoch and a manifest.

A reader gets an epoch's cores only from a run whose manifest names the
folder, holds the epoch as ``ok`` with the file's sha256, and records its span.
Offline: everything is written under ``tmp_path``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_definition.artifact_intervals import (
    CORE_COLUMNS,
    ArtifactRunError,
    cores_path,
    empty_cores,
    epoch_artifact_intervals,
    make_cores,
    read_epoch_cores,
    read_manifest,
    write_cores,
)
from continuum_swr_replay.analysis.event_definition.artifact_masks import (
    artifact_intervals,
)
from continuum_swr_replay.runs.io import sha256_file, write_json
from continuum_swr_replay.tables.io import read_attrs

EPOCH = "chimi20200225_02_r1_HPC"
SPAN = (1.58e9, 1.58e9 + 1200.0)


def _cores() -> pd.DataFrame:
    return make_cores(
        {
            "kind": ["S", "T", "T"],
            "signatures": ["S", "ca1+ref", "spk"],
            "core_start": [SPAN[0] + 10.0, SPAN[0] + 50.0, SPAN[0] + 60.0],
            "core_end": [SPAN[0] + 14.0, SPAN[0] + 50.004, SPAN[0] + 60.0001],
            "spec_max": [12.5, np.nan, np.nan],
            "n_windows": [4.0, np.nan, np.nan],
            "ref_j5_max": [np.nan, 31.9, np.nan],
            "ca1_frac_max": [np.nan, 0.25, np.nan],
            "spk_frac_max": [np.nan, np.nan, 0.5],
            "spk_ood_max": [np.nan, np.nan, 0.9],
        }
    )


def _run(tmp_path, cores: pd.DataFrame | None = None, **record) -> tuple:
    """A run folder holding one epoch; return (folder, manifest)."""
    directory = tmp_path / "artifacts" / "run0001"
    path = cores_path(directory, EPOCH)
    sha = write_cores(cores if cores is not None else _cores(), path, stamp={"x": 1})
    manifest = {
        "run_id": "run0001",
        "epochs": {
            EPOCH: {
                "status": "ok",
                "span_s": list(SPAN),
                "outputs": {"cores": {"path": str(path), "sha256": sha}},
                **record,
            }
        },
    }
    write_json(directory / "manifest.json", manifest, indent=2)
    return directory, manifest


class TestTheCoresFile:
    def test_a_written_file_reads_back_with_its_columns_and_stamp(self, tmp_path):
        path = cores_path(tmp_path, EPOCH)
        sha = write_cores(_cores(), path, stamp={"run_id": "run0001"})

        assert path.name == f"{EPOCH}_artifact_intervals.parquet"
        assert sha == sha256_file(path)
        assert read_attrs(path)["run_id"] == "run0001"
        back = pd.read_parquet(path)
        assert list(back.columns) == list(CORE_COLUMNS)
        pd.testing.assert_frame_equal(back, _cores(), check_dtype=False)

    def test_an_epoch_without_artifacts_writes_an_empty_file(self, tmp_path):
        path = cores_path(tmp_path, EPOCH)
        write_cores(empty_cores(), path, stamp={})

        back = pd.read_parquet(path)
        assert back.empty
        assert list(back.columns) == list(CORE_COLUMNS)

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            (lambda c: c.drop(columns="signatures"), "signatures"),
            (lambda c: c.assign(kind="X"), "kind"),
            (lambda c: c.assign(core_end=c["core_start"] - 1.0), "ends before"),
            (lambda c: c.assign(core_start=np.nan), "finite"),
        ],
        ids=["missing-column", "bad-kind", "reversed", "nan-bound"],
    )
    def test_malformed_cores_are_refused_before_writing(
        self, tmp_path, change, message
    ):
        path = cores_path(tmp_path, EPOCH)
        with pytest.raises(ArtifactRunError, match=message):
            write_cores(change(_cores()), path, stamp={})
        assert not path.exists()


class TestTheManifest:
    def test_a_folder_holding_another_runs_manifest_is_refused(self, tmp_path):
        directory, _ = _run(tmp_path)
        moved = directory.rename(directory.parent / "other")

        with pytest.raises(ArtifactRunError, match="run0001"):
            read_manifest(moved)

    def test_a_missing_manifest_is_refused(self, tmp_path):
        with pytest.raises(ArtifactRunError, match="manifest"):
            read_manifest(tmp_path)


class TestReadingAnEpoch:
    def test_cores_and_span_come_back(self, tmp_path):
        directory, _ = _run(tmp_path)

        epoch = read_epoch_cores(directory, EPOCH)

        pd.testing.assert_frame_equal(epoch.cores, _cores(), check_dtype=False)
        assert epoch.span == SPAN

    def test_an_epoch_without_artifacts_reads_as_no_cores(self, tmp_path):
        directory, _ = _run(tmp_path, cores=empty_cores())

        assert read_epoch_cores(directory, EPOCH).cores.empty

    def test_an_epoch_the_run_never_saw_is_refused(self, tmp_path):
        directory, _ = _run(tmp_path)

        with pytest.raises(ArtifactRunError, match="not in the artifact run"):
            read_epoch_cores(directory, "senor20201101_06_r3_HPC")

    def test_a_failed_epoch_is_refused(self, tmp_path):
        directory, manifest = _run(tmp_path)
        manifest["epochs"][EPOCH] = {"status": "failed", "error": "boom"}
        write_json(directory / "manifest.json", manifest, indent=2)

        with pytest.raises(ArtifactRunError, match="failed.*boom"):
            read_epoch_cores(directory, EPOCH)

    def test_a_missing_file_is_refused_not_read_as_no_artifacts(self, tmp_path):
        directory, _ = _run(tmp_path)
        cores_path(directory, EPOCH).unlink()

        with pytest.raises(ArtifactRunError, match="no artifact intervals"):
            read_epoch_cores(directory, EPOCH)

    def test_a_file_changed_after_the_run_is_refused(self, tmp_path):
        directory, _ = _run(tmp_path)
        write_cores(_cores().iloc[:1], cores_path(directory, EPOCH), stamp={})

        with pytest.raises(ArtifactRunError, match="sha256"):
            read_epoch_cores(directory, EPOCH)


def test_epoch_artifact_intervals_is_the_padded_mask_or_none(tmp_path):
    """The detection runners' mask: the run's cores padded by ``PADDING_S``
    and clipped to the span; no rows when no run is configured."""
    directory, _ = _run(tmp_path)

    mask = epoch_artifact_intervals(directory, EPOCH)

    expected = artifact_intervals(_cores(), SPAN)
    pd.testing.assert_frame_equal(mask, expected)
    assert len(mask) > 0
    empty = epoch_artifact_intervals(None, EPOCH)
    assert list(empty.columns) == ["start_time", "end_time"]
    assert empty.empty and (empty.dtypes == np.float64).all()
