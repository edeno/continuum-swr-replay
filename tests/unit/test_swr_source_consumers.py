"""Every analysis that reads SWR labels or times reads them from the one source.

Each consumer's own entry point is run on a decode frame whose stored
``SWR_event`` (the decode's ``RippleTimesV1`` labels) marks a ripple the source
does not have, and the source's inventory marks three the frame does not. What
reaches the consumer must be the source's labels: the frame a builder gets,
the intervals the PFC path fits, the labels a loader returns. Where a consumer
reads the frame and then needs Spyglass or a decode the test cannot supply,
the next call it makes is replaced by a recorder that keeps the frame and stops
it there.

The static checks at the end keep it that way: no module outside the decode
writers and an explained list reads the stored ripples or a decode frame's
pickle directly.

Offline: the inventory and register are written under ``tmp_path``; scripts
that import ``load_data`` are imported with ``spyglass`` stubbed, as
``test_timebin_tables.py`` does.
"""

from __future__ import annotations

import importlib
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from continuum_swr_replay.analysis.hippocampus import swr_source
from continuum_swr_replay.analysis.hippocampus.event_table import (
    build_classified_nonlocal_event_table,
)
from continuum_swr_replay.analysis.hippocampus.swr_source import (
    SwrSourceError,
    swr_event_labels,
)
from continuum_swr_replay.data_loaders import epoch_register
from continuum_swr_replay.table_io import sha256_file
from tests.unit._swr_inventory_fixtures import RUN_ID, write_swr_inventory

REPO = Path(__file__).resolve().parents[2]
NWB = "j1620210706_.nwb"
EPOCH = "02_r1"
EPOCH_ID = "j1620210706_02_r1_HPC"
TIME = np.round(100.0 + np.arange(2000) / 500.0, 6)
"""A 500 Hz decode grid, 100.000-103.998 s."""
SOURCE_SWRS = [(100.50, 100.56), (101.00, 101.04), (102.00, 102.10)]
STORED_SWR = (103.00, 103.10)
"""The decode's stored ripple, which the source does not have."""


class _Stop(Exception):
    """Raised by a recorder once the consumer has handed it the frame."""


@pytest.fixture
def source(tmp_path, monkeypatch):
    """The source's default inventory is a synthetic one under ``tmp_path``."""
    root = tmp_path / "swr"
    write_swr_inventory(root, {EPOCH_ID: SOURCE_SWRS})
    monkeypatch.setattr(swr_source, "SWR_INVENTORY_ROOT", root)
    monkeypatch.setattr(swr_source, "SWR_INVENTORY_RUN_ID", RUN_ID)
    return root / RUN_ID


@pytest.fixture
def register(tmp_path, monkeypatch):
    path = tmp_path / "epoch_register.csv"
    pd.DataFrame(
        [[NWB, EPOCH, "analysed", "", ""]],
        columns=["nwb_file_name", "epoch", "status", "reason", "evidence"],
    ).to_csv(path, index=False)
    monkeypatch.setattr(epoch_register, "REGISTER_PATH", path)


def source_labels() -> np.ndarray:
    return swr_event_labels(TIME, EPOCH_ID, speed=np.zeros(TIME.size))


def stored_labels() -> np.ndarray:
    labels = np.zeros(TIME.size, dtype=int)
    labels[(TIME >= STORED_SWR[0]) & (TIME <= STORED_SWR[1])] = 1
    return labels


def decode_frame(**columns) -> pd.DataFrame:
    """A decode's ``data_over_time``, as ``get_data_over_time`` saves it."""
    return pd.DataFrame(
        {
            "non_local_event": 0,
            "SWR_event": stored_labels(),
            "MUA_event": 0,
            "Non-Local_Continuous_prob": 0.0,
            "Non-Local_Fragmented_prob": 0.0,
            "ahead_behind_distance": 0.0,
            "head_speed": 0.0,
            "n_SWR": 1,
            "animal": "j16",
            "day": 20210706,
            "epoch": EPOCH,
            "nwb_file_name": NWB,
            **columns,
        },
        index=pd.Index(TIME, name="time"),
    )


def assert_source_labels(frame: pd.DataFrame) -> None:
    np.testing.assert_array_equal(frame["SWR_event"].to_numpy(), source_labels())
    assert "n_SWR" not in frame.columns
    assert frame["swr_observed"].all()


def recorder(seen: list):
    def record(frame, *args, **kwargs):
        seen.append(frame)
        raise _Stop

    return record


@pytest.fixture
def offline(monkeypatch):
    """Import a script with ``spyglass`` stubbed; forget it afterwards."""
    common = types.ModuleType("spyglass.common")
    common.PositionIntervalMap = type("PositionIntervalMap", (), {})
    monkeypatch.setitem(sys.modules, "spyglass", types.ModuleType("spyglass"))
    monkeypatch.setitem(sys.modules, "spyglass.common", common)
    data_loaders = importlib.import_module("continuum_swr_replay.data_loaders")
    load_data_was_cached = "load_data" in vars(data_loaders)
    bound = ("continuum_swr_replay.data_loaders.bandit_task",)
    fresh_bound = [name for name in bound if name not in sys.modules]
    imported: list[str] = []

    def load(name: str):
        if name not in sys.modules:
            imported.append(name)
        return importlib.import_module(name)

    yield load

    for name in [*imported, *fresh_bound]:
        sys.modules.pop(name, None)
        parent, _, child = name.rpartition(".")
        if parent in sys.modules:
            vars(sys.modules[parent]).pop(child, None)
    if not load_data_was_cached:
        vars(data_loaders).pop("load_data", None)


# --------------------------------------------------------------------------
# The pooled loaders (Figures 2 and 3, the decoder-validation supplement)
# --------------------------------------------------------------------------


def test_the_nonlocal_loader_returns_the_sources_labels(tmp_path, source, register):
    from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
        load_data_over_time,
    )

    folder = tmp_path / "decodes"
    folder.mkdir()
    decode_frame().to_pickle(folder / f"{EPOCH_ID}_80_data_over_time.pkl")

    assert_source_labels(load_data_over_time(folder))


def test_the_contfrag_loader_returns_the_sources_labels(tmp_path, source, register):
    from continuum_swr_replay.analysis.contfrag.data_aggregation import (
        load_contfrag_data_over_time,
    )

    folder = tmp_path / "decodes"
    folder.mkdir()
    decode_frame().to_pickle(folder / f"{EPOCH_ID}_contfrag_data_over_time.pkl")

    assert_source_labels(load_contfrag_data_over_time(folder))


def test_the_quality_metrics_loader_returns_the_sources_labels(
    tmp_path, source, register, monkeypatch
):
    import scripts.qa.run_quality_metrics as quality_metrics

    folder = tmp_path / "decodes"
    folder.mkdir()
    decode_frame().to_pickle(folder / f"{EPOCH_ID}_80_data_over_time.pkl")
    monkeypatch.setattr(quality_metrics, "PROCESSED_DATA_DIR", folder)

    assert_source_labels(quality_metrics.load_all_sessions(threshold=80))


# --------------------------------------------------------------------------
# The canonical event table's is_swr
# --------------------------------------------------------------------------


def test_the_event_tables_is_swr_follows_the_source(source):
    """One non-local event inside a source SWR, one inside the stored ripple."""
    nonlocal_event = np.zeros(TIME.size, dtype=int)
    nonlocal_event[(TIME >= 101.01) & (TIME <= 101.03)] = 1
    nonlocal_event[(TIME >= 103.02) & (TIME <= 103.06)] = 2
    probability = np.where(nonlocal_event > 0, 0.95, 0.0)
    dot = decode_frame(
        non_local_event=nonlocal_event, **{"Non-Local_Continuous_prob": probability}
    )

    stored = build_classified_nonlocal_event_table(dot)
    new = build_classified_nonlocal_event_table(swr_source.with_swr_source(dot))

    assert stored["is_swr"].tolist() == [False, True]
    assert new["is_swr"].tolist() == [True, False]
    assert new["matched_swr_event"].tolist()[0] == 2


# --------------------------------------------------------------------------
# The table builders' per-session reads
# --------------------------------------------------------------------------


@pytest.fixture
def saved_decode(tmp_path):
    path = tmp_path / f"{EPOCH_ID}_80_data_over_time.pkl"
    decode_frame().to_pickle(path)
    results = tmp_path / f"{EPOCH_ID}_results.nc"
    results.write_bytes(b"")
    return {
        "nwb_file_name": NWB,
        "epoch_name": EPOCH,
        "session_id": EPOCH_ID,
        "brain_area": "HPC",
        "dot_file": path,
        "env_file": tmp_path / f"{EPOCH_ID}_environment.pkl",
        "results_file": results,
    }


def test_the_replay_tables_build_on_the_sources_labels(
    source, saved_decode, monkeypatch
):
    from continuum_swr_replay.analysis.replay import prepare_data

    seen: list = []
    monkeypatch.setattr(prepare_data, "validate_dot_event_contract", recorder(seen))

    with pytest.raises(_Stop):
        prepare_data._process_session(saved_decode)

    assert_source_labels(seen[0])


def test_the_stop_only_bootstrap_builds_on_the_sources_labels(
    source, saved_decode, monkeypatch, tmp_path
):
    from continuum_swr_replay.analysis.replay import prepare_data

    seen: list = []
    monkeypatch.setattr(prepare_data, "validate_dot_event_contract", recorder(seen))

    with pytest.raises(_Stop):
        prepare_data.build_stop_table_scope(
            [saved_decode], from_cache=True, output_dir=tmp_path / "out"
        )

    assert_source_labels(seen[0])


def test_the_contfrag_tables_build_on_the_sources_labels(
    source, saved_decode, monkeypatch, offline
):
    script = offline("scripts.decoding.prepare_contfrag_replay_content_data")
    seen: list = []
    monkeypatch.setattr(script, "validate_dot_event_contract", recorder(seen))

    with pytest.raises(_Stop):
        script._process_session(saved_decode, "SWR")

    assert_source_labels(seen[0])


def test_the_contfrag_covariate_refresh_builds_on_the_sources_labels(
    source, saved_decode, monkeypatch, offline
):
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "decoding"))
    offline("prepare_contfrag_replay_content_data")
    script = offline("refresh_contfrag_behavioral_covariates")
    seen: list = []
    monkeypatch.setattr(script, "validate_dot_event_contract", recorder(seen))

    with pytest.raises(_Stop):
        script._rebuild_session(
            saved_decode,
            event_source="SWR",
            contingency=pd.DataFrame(),
            old_event=pd.DataFrame(),
            old_arm=pd.DataFrame(),
            from_cache=True,
        )

    assert_source_labels(seen[0])


def test_the_timebin_tables_build_on_the_sources_labels(
    source, saved_decode, monkeypatch, offline
):
    script = offline("scripts.decoding.build_timebin_tables")
    seen: list = []
    monkeypatch.setattr(script, "load_data", lambda *args, **kwargs: {})
    monkeypatch.setattr(script, "build_event_onset_table", recorder(seen))

    with pytest.raises(_Stop):
        script.build_one(saved_decode)

    assert_source_labels(seen[0])


def test_the_stop_table_rebuild_builds_on_the_sources_labels(
    source, saved_decode, monkeypatch, offline
):
    script = offline("scripts.decoding.regenerate_stop_table")
    data_loaders = importlib.import_module("continuum_swr_replay.data_loaders")
    monkeypatch.setattr(
        data_loaders,
        "load_data",
        lambda *args, **kwargs: {
            "trials": pd.DataFrame(),
            "port_experiences": pd.DataFrame(),
            "track_graph": object(),
        },
        raising=False,
    )
    seen: list = []
    monkeypatch.setattr(script, "build_stop_table", recorder(seen))

    with pytest.raises(_Stop):
        script.build_one(saved_decode)

    assert_source_labels(seen[0])


# --------------------------------------------------------------------------
# The QA and content scans of decode frames
# --------------------------------------------------------------------------


def test_the_content_occupancy_scan_reads_the_sources_labels(
    source, monkeypatch, tmp_path
):
    import scripts.content.analyze_content_occupancy as occupancy

    decode_frame().to_pickle(tmp_path / f"{EPOCH_ID}_80_data_over_time.pkl")
    monkeypatch.setattr(occupancy, "PROCESSED_DATA_DIR", tmp_path)
    seen: list = []
    monkeypatch.setattr(occupancy, "validate_dot_event_contract", recorder(seen))

    with pytest.raises(_Stop):
        occupancy.session_content_distances(EPOCH_ID)

    assert_source_labels(seen[0])


def test_the_spike_evidence_scan_reads_the_sources_labels(
    source, monkeypatch, tmp_path
):
    import scripts.qa.analyze_event_spike_evidence as spike_evidence

    decode_frame().to_pickle(tmp_path / f"{EPOCH_ID}_80_data_over_time.pkl")
    monkeypatch.setattr(spike_evidence, "PROCESSED_DATA_DIR", tmp_path)
    seen: list = []
    monkeypatch.setattr(spike_evidence, "validate_dot_event_contract", recorder(seen))

    with pytest.raises(_Stop):
        spike_evidence.analyze_session(EPOCH_ID, tmp_path, n_sample=1)

    assert_source_labels(seen[0])


def test_the_overlap_geometry_count_classifies_against_the_source(
    source, monkeypatch, offline
):
    script = offline("scripts.content.quantify_swr_overlap_geometry")
    position = pd.DataFrame({"head_speed": np.zeros(TIME.size)}, index=TIME)
    states = ["Local", "No-Spike", "Non-Local Continuous", "Non-Local Fragmented"]
    results = xr.Dataset(
        {
            "acausal_state_probabilities": (
                ("time", "states"),
                np.zeros((TIME.size, len(states))),
            )
        },
        coords={"time": TIME, "states": states},
    )
    monkeypatch.setattr(
        script,
        "load_data",
        lambda *args, **kwargs: {
            "position_info": position,
            "ripple_times": pd.DataFrame(
                [STORED_SWR], columns=["start_time", "end_time"]
            ),
        },
    )
    monkeypatch.setattr(script, "load_results", lambda *a, **k: (results, None, None))
    seen: list = []

    def record(events, swrs):
        seen.append(swrs)
        raise _Stop

    monkeypatch.setattr(script, "classify_swr_overlap_geometry", record)

    with pytest.raises(_Stop):
        script.quantify_session(NWB, EPOCH)

    np.testing.assert_allclose(
        seen[0][["start_time", "end_time"]].to_numpy(), np.array(SOURCE_SWRS)
    )


# --------------------------------------------------------------------------
# The PFC path (Figure 4, Stage 5B, Stage 6, the PFC QA)
# --------------------------------------------------------------------------


def test_the_pfc_swr_intervals_are_the_sources(source, tmp_path):
    from continuum_swr_replay.analysis.pfc.canonical_events import (
        load_session_swr_intervals,
    )

    path = tmp_path / f"{EPOCH_ID}_80_data_over_time.pkl"
    decode_frame().to_pickle(path)

    intervals, sha = load_session_swr_intervals(NWB, EPOCH, dot_path=path)

    np.testing.assert_allclose(intervals.to_numpy(), np.array(SOURCE_SWRS))
    assert sha == sha256_file(path)


def test_the_pfc_provenance_names_the_source_and_hashes_its_events(
    source, tmp_path, monkeypatch
):
    import continuum_swr_replay.analysis.pfc.canonical_events as canonical

    monkeypatch.setattr(canonical, "PROCESSED_DATA_DIR", tmp_path)

    provenance = canonical.canonical_event_input_hashes(
        NWB, EPOCH, event_table_path=tmp_path / "absent.parquet"
    )

    assert provenance["swr_interval_source"] == f"kay@{RUN_ID}"
    assert provenance["swr_events_sha256"] == sha256_file(
        source / f"{EPOCH_ID}_kay_events.parquet"
    )


def test_a_pfc_epoch_the_source_cannot_serve_hashes_to_none(tmp_path, monkeypatch):
    import continuum_swr_replay.analysis.pfc.canonical_events as canonical

    root = tmp_path / "swr"
    write_swr_inventory(root, {}, records={EPOCH_ID: {"status": "failed"}})
    monkeypatch.setattr(swr_source, "SWR_INVENTORY_ROOT", root)
    monkeypatch.setattr(swr_source, "SWR_INVENTORY_RUN_ID", RUN_ID)
    monkeypatch.setattr(canonical, "PROCESSED_DATA_DIR", tmp_path)

    provenance = canonical.canonical_event_input_hashes(NWB, EPOCH)

    assert provenance["swr_events_sha256"] is None
    with pytest.raises(SwrSourceError):
        canonical.load_session_swr_intervals(
            NWB, EPOCH, dot_path=_saved(tmp_path, decode_frame())
        )


def _saved(folder: Path, frame: pd.DataFrame) -> Path:
    path = folder / "dot.pkl"
    frame.to_pickle(path)
    return path


# --------------------------------------------------------------------------
# Static: nothing else reads the stored ripples or a decode frame's pickle
# --------------------------------------------------------------------------

#: Files outside ``scripts/figures`` and ``scripts/exploratory`` that may name
#: the stored Spyglass ripples, and why each is not an analysis read.
STORED_RIPPLE_READERS = {
    "src/continuum_swr_replay/data_loaders/lfp.py": "the loader itself",
    "src/continuum_swr_replay/data_loaders/bandit_task.py": "load_data's decode input",
    "src/continuum_swr_replay/data_loaders/utils.py": (
        "the decoders' no_ripple encoding-training mask (unchanged, a separate "
        "decision)"
    ),
    "src/continuum_swr_replay/data_loaders/constants.py": "register evidence, prose",
    "src/continuum_swr_replay/analysis/hippocampus/swr_source.py": (
        "prose naming what it replaces"
    ),
    "src/continuum_swr_replay/types.py": "load_data's return types",
    "src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py": (
        "get_data_over_time: the decode's own SWR_event provenance"
    ),
    "src/continuum_swr_replay/analysis/contfrag/data_aggregation.py": (
        "get_contfrag_data_over_time: the decode's own SWR_event provenance"
    ),
    "src/continuum_swr_replay/analysis/hippocampus/event_detection.py": (
        "generic overlap classifiers; 'ripple_times' is a parameter name"
    ),
    "src/continuum_swr_replay/analysis/hippocampus/event_table.py": (
        "assert_drawn_class_matches_ripples, a parameter of Figure 2's check"
    ),
    "src/continuum_swr_replay/visualization/figurl.py": "the decode's figurl view",
    "src/continuum_swr_replay/visualization/plots.py": "decode-session plots",
    "src/continuum_swr_replay/visualization/session_tables.py": "prose",
    "src/continuum_swr_replay/visualization/animation.py": "docstring example",
    "src/continuum_swr_replay/visualization/__init__.py": "docstring example",
    "scripts/decoding/run_decoding_bandit.py": "the decode's figurl view",
    "scripts/decoding/run_swr_detection.py": "the Kay version check against it",
    "scripts/qa/check_data_availability.py": "which decode inputs load",
    "scripts/qa/diagnose_pfc_event_definition.py": (
        "reproduces the retired PFC event definition on purpose"
    ),
}

#: Files outside ``scripts/figures`` and ``scripts/exploratory`` that unpickle a
#: decode frame without the source, and why none of them reads its SWRs.
DECODE_FRAME_READERS = {
    "src/continuum_swr_replay/analysis/hippocampus/swr_source.py": "the reader",
    "scripts/qa/audit_refresh_consistency.py": "audits the decode's own outputs",
    "scripts/qa/run_state_evidence.py": "time axis, speed, non-local labels",
    "scripts/qa/run_half_tetrode_decodes.py": "time axis, non-local labels",
    "scripts/qa/diagnose_trial_detection.py": "position and trials",
    "scripts/qa/compare_pfc_cohorts.py": "PFC bundles; names data_over_time_sha256",
}


def _python_files() -> list[Path]:
    return [
        path
        for folder in ("src", "scripts")
        for path in (REPO / folder).rglob("*.py")
        if not {"figures", "exploratory"} & set(path.relative_to(REPO).parts)
    ]


def test_only_the_listed_modules_name_the_stored_ripples():
    pattern = re.compile(r"ripple_times|RippleTimesV1|get_ripple\(")
    found = {
        str(path.relative_to(REPO))
        for path in _python_files()
        if pattern.search(path.read_text())
    }

    assert found - set(STORED_RIPPLE_READERS) == set()


def test_only_the_listed_modules_unpickle_a_decode_frame():
    found = {
        str(path.relative_to(REPO))
        for path in _python_files()
        if "read_pickle(" in (text := path.read_text())
        and re.search(r"data_over_time|dot_file", text)
    }

    assert found - set(DECODE_FRAME_READERS) == set()
