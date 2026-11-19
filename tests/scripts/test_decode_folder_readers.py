"""Readers of a decode folder take the register's analyzed epochs, never a glob.

A pending epoch's decode in the folder, or any other epoch the register does
not analyze, must not reach a figure, a QA summary or an exploratory scan. Each
reader names the analyzed epochs' files through
``epoch_register.analyzed_decode_files``.

Offline: ``REGISTER_PATH`` points at a register written under ``tmp_path``,
analyzing ``a20200101 02_r1`` and ``b20200102 04_r2`` and holding
``a20200101 06_r3`` pending; the decode files are small stand-ins, and the SWR
source is a synthetic inventory with no SWRs (the frames are read through it).
The data root is ``tmp_path / "decodes"``: the inventory is written under it, and
the decode folder a reader scans by default is that root.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.checks.run_quality_metrics as quality_metrics
import scripts.event_evidence.analyze_event_spike_evidence as spike_evidence
import scripts.figures.generate_event_gallery as event_gallery
import scripts.replay_models.content.analyze_content_occupancy as content_occupancy
from continuum_swr_replay.analysis.contfrag.data_aggregation import (
    load_contfrag_data_over_time,
)
from continuum_swr_replay.analysis.detection.nonlocal_events import (
    load_data_over_time,
)
from continuum_swr_replay.data_loaders import epoch_register
from continuum_swr_replay.decoding.files import non_local_threshold_percent
from tests.support._swr_inventory_fixtures import use_swr_source, write_swr_inventory

ANALYZED = ("a20200101_02_r1_HPC", "b20200102_04_r2_HPC")
PENDING = "a20200101_06_r3_HPC"


@pytest.fixture(autouse=True)
def register(tmp_path, monkeypatch):
    path = tmp_path / "epoch_register.csv"
    pd.DataFrame(
        [
            ["a20200101_.nwb", "02_r1", "analyzed", "", "", "False"],
            ["b20200102_.nwb", "04_r2", "analyzed", "", "", "False"],
            [
                "a20200101_.nwb",
                "06_r3",
                "pending",
                "PENDING_POPULATE",
                "no PIM",
                "False",
            ],
        ],
        columns=list(epoch_register.COLUMNS),
    ).to_csv(path, index=False)
    monkeypatch.setattr(epoch_register, "REGISTER_PATH", path)


@pytest.fixture(autouse=True)
def source(tmp_path, monkeypatch):
    """The decode frames are read through the SWR source: an empty inventory,
    under the data root ``tmp_path / "decodes"``."""
    inventory = write_swr_inventory(
        tmp_path / "decodes", {epoch_id: [] for epoch_id in (*ANALYZED, PENDING)}
    )
    use_swr_source(monkeypatch, inventory)


def _frame(epoch_id: str) -> pd.DataFrame:
    """A two-row stand-in for a decode's ``data_over_time``."""
    stem, number, replicate, _ = epoch_id.split("_")
    return pd.DataFrame(
        {
            "epoch_id": epoch_id,
            "animal": stem[:-8],
            "day": int(stem[-8:]),
            "epoch": f"{number}_{replicate}",
            "head_speed": [0.0, 0.0],
            "Non-Local_Continuous_prob": [0.1, 0.2],
            "Non-Local_Fragmented_prob": [0.3, 0.4],
            "ahead_behind_distance": [-1.0, 2.0],
        },
        index=pd.Index([100.0, 100.002], name="time"),
    )


def _folder(tmp_path: Path, suffix: str, epoch_ids=(*ANALYZED, PENDING)) -> Path:
    folder = tmp_path / "decodes"
    folder.mkdir(parents=True, exist_ok=True)
    for epoch_id in epoch_ids:
        _frame(epoch_id).to_pickle(folder / f"{epoch_id}{suffix}")
    return folder


def test_the_nonlocal_loader_reads_only_the_analyzed_epochs(tmp_path):
    folder = _folder(tmp_path, "_80_data_over_time.pkl")

    data = load_data_over_time(folder)

    assert sorted(set(data["epoch_id"])) == list(ANALYZED)
    np.testing.assert_allclose(data["Non-Local_prob"], [0.4, 0.6, 0.4, 0.6])


def test_the_nonlocal_loader_refuses_a_missing_analyzed_epoch(tmp_path):
    folder = _folder(tmp_path, "_80_data_over_time.pkl", (ANALYZED[0], PENDING))

    with pytest.raises(FileNotFoundError, match=ANALYZED[1]):
        load_data_over_time(folder)


def test_the_contfrag_loader_reads_only_the_analyzed_epochs(tmp_path):
    folder = _folder(tmp_path, "_contfrag_data_over_time.pkl")

    data = load_contfrag_data_over_time(folder)

    assert sorted(set(data["epoch_id"])) == list(ANALYZED)
    with pytest.raises(FileNotFoundError, match=ANALYZED[0]):
        load_contfrag_data_over_time(
            _folder(tmp_path / "other", "_contfrag_data_over_time.pkl", (PENDING,))
        )


def test_quality_metrics_load_only_the_analyzed_epochs(tmp_path):
    _folder(tmp_path, "_80_data_over_time.pkl")  # the data root

    data = quality_metrics.load_all_sessions(threshold=80)
    one_animal = quality_metrics.load_all_sessions(animal="b", threshold=80)

    assert sorted(set(data["epoch_id"])) == list(ANALYZED)
    assert sorted(set(one_animal["epoch_id"])) == [ANALYZED[1]]
    with pytest.raises(ValueError, match="'c'"):
        quality_metrics.load_all_sessions(animal="c", threshold=80)


def test_the_content_scan_finds_only_analyzed_epochs(tmp_path, monkeypatch):
    folder = tmp_path / "decodes"
    for epoch_id in (*ANALYZED, PENDING):
        (folder / f"{epoch_id}_environment.pkl").touch()
        (folder / f"{epoch_id}_80_data_over_time.pkl").touch()
    (folder / f"{ANALYZED[1]}_80_data_over_time.pkl").unlink()

    assert content_occupancy.find_sessions(None) == [ANALYZED[0]]
    assert content_occupancy.find_sessions("b") == []


def test_the_spike_evidence_scan_finds_only_analyzed_epochs(tmp_path, monkeypatch):
    folder = tmp_path / "decodes"
    for epoch_id in (*ANALYZED, PENDING):
        for suffix in ("_detector.pkl", "_results.nc", "_80_data_over_time.pkl"):
            (folder / f"{epoch_id}{suffix}").touch()

    assert spike_evidence.find_sessions(None) == list(ANALYZED)
    assert spike_evidence.find_sessions("a") == [ANALYZED[0]]


def test_the_event_gallery_catalogs_only_analyzed_epochs(tmp_path):
    """The gallery read every ``*_HPC_80_data_over_time.pkl`` in the folder."""
    folder = _folder(tmp_path, "_80_data_over_time.pkl")
    _frame("z20990101_02_r1_HPC").to_pickle(
        folder / "z20990101_02_r1_HPC_80_data_over_time.pkl"
    )

    pkls = event_gallery.list_epoch_pkls(folder)

    assert pkls == [
        folder / f"{epoch_id}_80_data_over_time.pkl" for epoch_id in ANALYZED
    ]
    assert [event_gallery.epoch_id_from_pkl(pkl) for pkl in pkls] == [
        "a20200101_02_r1",
        "b20200102_04_r2",
    ]
    with pytest.raises(FileNotFoundError, match="No analyzed epoch"):
        event_gallery.list_epoch_pkls(
            _folder(tmp_path / "other", "_80_data_over_time.pkl", (PENDING,))
        )


def test_the_event_gallery_names_an_analyzed_epoch_it_leaves_out(tmp_path, caplog):
    folder = _folder(tmp_path, "_80_data_over_time.pkl", (ANALYZED[0], PENDING))

    with caplog.at_level("WARNING"):
        pkls = event_gallery.list_epoch_pkls(folder)

    assert pkls == [folder / f"{ANALYZED[0]}_80_data_over_time.pkl"]
    assert "1 of the 2 analyzed epochs" in caplog.text
    assert ANALYZED[1] in caplog.text
    assert PENDING not in caplog.text


def test_the_event_gallery_is_quiet_when_every_analyzed_epoch_is_there(
    tmp_path, caplog
):
    with caplog.at_level("WARNING"):
        event_gallery.list_epoch_pkls(_folder(tmp_path, "_80_data_over_time.pkl"))

    assert "left out" not in caplog.text


@pytest.mark.parametrize("script", [content_occupancy, spike_evidence])
def test_a_scan_names_the_file_its_threshold_validates(script, tmp_path):
    """The file a scan reads is the one the library's
    ``DEFAULT_NON_LOCAL_THRESHOLD`` names, the threshold the read then checks
    its labels at."""
    folder = tmp_path / "decodes"
    for suffix in (
        "_environment.pkl",
        "_detector.pkl",
        "_results.nc",
        "_50_data_over_time.pkl",
    ):
        (folder / f"{ANALYZED[0]}{suffix}").touch()

    assert script.find_sessions(None) == []
    percent = non_local_threshold_percent(script.DEFAULT_NON_LOCAL_THRESHOLD)
    (folder / f"{ANALYZED[0]}_{percent}_data_over_time.pkl").touch()
    assert script.find_sessions(None) == [ANALYZED[0]]


def test_the_event_gallery_renders_from_one_decode_folder(tmp_path, monkeypatch):
    """An event's frame, results and posterior all come from ``--data-dir``,
    never a frame from one decode and a posterior from another."""
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    pd.DataFrame({"epoch_id": ["a20200101_02_r1"], "category": ["non_swr"]}).to_parquet(
        gallery / "_catalog.parquet"
    )
    data_dir = tmp_path / "staging"
    read = {}

    class _Stop(Exception):
        pass

    def load_conditional_posterior(path):
        read["posterior"] = path
        raise _Stop

    monkeypatch.setattr(event_gallery, "load_data", lambda *a, **k: {})
    monkeypatch.setattr(event_gallery, "ripple_power_inputs", lambda epoch_id: None)
    monkeypatch.setattr(
        event_gallery,
        "load_results",
        lambda *a, root, **k: read.setdefault("results", root) and (None, None, None),
    )
    monkeypatch.setattr(
        event_gallery,
        "read_dot_with_swr_source",
        lambda path: read.setdefault("frame", path),
    )
    monkeypatch.setattr(event_gallery, "immobility_bout_intervals", lambda dot: None)
    monkeypatch.setattr(event_gallery, "swr_intervals", lambda dot: None)
    monkeypatch.setattr(
        event_gallery, "load_conditional_posterior", load_conditional_posterior
    )

    with pytest.raises(_Stop):
        event_gallery.render_epoch("a20200101_02_r1", data_dir, gallery, False)

    assert read == {
        "results": data_dir,
        "frame": data_dir / "a20200101_02_r1_HPC_80_data_over_time.pkl",
        "posterior": data_dir / "a20200101_02_r1_HPC_results.nc",
    }
