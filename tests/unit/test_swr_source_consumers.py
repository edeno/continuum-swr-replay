"""Every analysis that reads SWR labels or times reads them from the one source.

Each consumer's own entry point is run on a decode frame whose stored
``SWR_event`` (the decode's ``RippleTimesV1`` labels) marks a ripple the source
does not have, and the source's inventory marks three the frame does not, one
of which the configured artifact run masks. What reaches the consumer must be
the source's two kept SWRs: the frame a builder gets, the intervals the PFC
path fits, the labels a loader returns. No consumer passes the artifacts; the
source resolves them, so every consumer gets the same ones. Where a consumer
reads the frame and then needs Spyglass or a decode the test cannot supply,
the next call it makes is replaced by a recorder that keeps the frame and stops
it there.

Every table built from these labels is published with the source's stamp, and
every reader that combines a shipped table's SWR columns with the source's live
labels refuses a table stamped with another source.

The static checks at the end keep it that way: no module outside the figure
archive, the decode writers and an explained list reads the stored ripples or
a decode frame's pickle directly.

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
from continuum_swr_replay.intervals import interval_mask
from continuum_swr_replay.tables.io import read_table, sha256_file, write_table
from tests.unit._swr_inventory_fixtures import (
    ARTIFACT_RUN_ID,
    RUN_ID,
    use_swr_source,
    write_artifact_run,
    write_swr_inventory,
)

REPO = Path(__file__).resolve().parents[2]
NWB = "j1620210706_.nwb"
EPOCH = "02_r1"
EPOCH_ID = "j1620210706_02_r1_HPC"
TIME = np.round(100.0 + np.arange(2000) / 500.0, 6)
"""A 500 Hz decode grid, 100.000-103.998 s."""
SOURCE_SWRS = [(100.50, 100.56), (101.00, 101.04), (102.00, 102.10)]
ARTIFACT = ("spk", 102.15, 102.15)
"""The configured artifact run's one core, a spike-only line: padded by 0.1 s
(``PADDING_S``) it covers [102.05, 102.25] and drops the third source
SWR."""
KEPT_SWRS = SOURCE_SWRS[:2]
"""The source's SWRs every consumer must see: the artifact-overlapping one is
dropped by the source itself, so no consumer passes artifacts."""
STORED_SWR = (103.00, 103.10)
"""The decode's stored ripple, which the source does not have."""


class _Stop(Exception):
    """Raised by a recorder once the consumer has handed it the frame."""


@pytest.fixture
def source(tmp_path, monkeypatch):
    """The source's configured inventory and artifact run are synthetic ones
    under ``tmp_path``."""
    inventory = write_swr_inventory(tmp_path / "swr", {EPOCH_ID: SOURCE_SWRS})
    artifacts = write_artifact_run(tmp_path / "artifacts", {EPOCH_ID: [ARTIFACT]})
    use_swr_source(monkeypatch, inventory, artifacts)
    return inventory


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
    labels = frame["SWR_event"].to_numpy()
    np.testing.assert_array_equal(labels, source_labels())
    assert sorted(set(labels) - {0}) == [1, 2]  # the artifact dropped SWR 3
    assert "n_SWR" not in frame.columns
    # excluded time (the mask widened by 0.3 s) is not analysed: neither SWR
    # nor non-SWR
    ((start, end),) = swr_source.excluded_intervals(EPOCH_ID).to_numpy()
    in_excluded = (TIME >= start) & (TIME <= end)
    assert in_excluded.any()
    np.testing.assert_array_equal(frame["swr_observed"].to_numpy(), ~in_excluded)


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
    source, saved_decode, monkeypatch, offline, tmp_path
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
        script.quantify_session(NWB, EPOCH, data_dir=tmp_path)

    np.testing.assert_allclose(
        seen[0][["start_time", "end_time"]].to_numpy(), np.array(KEPT_SWRS)
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

    np.testing.assert_allclose(intervals.to_numpy(), np.array(KEPT_SWRS))
    assert sha == sha256_file(path)


def test_the_pfc_provenance_names_the_source_and_hashes_its_events(
    source, tmp_path, monkeypatch
):
    import continuum_swr_replay.analysis.pfc.canonical_events as canonical

    monkeypatch.setattr(canonical, "PROCESSED_DATA_DIR", tmp_path)

    provenance = canonical.canonical_event_input_hashes(
        NWB, EPOCH, event_table_path=tmp_path / "absent.parquet"
    )

    assert provenance["swr_interval_source"] == (
        f"kay@{RUN_ID}+artifacts@{ARTIFACT_RUN_ID}"
    )
    assert provenance["swr_events_sha256"] == sha256_file(
        source / f"{EPOCH_ID}_kay_events.parquet"
    )


def test_a_pfc_epoch_the_source_cannot_serve_hashes_to_none(tmp_path, monkeypatch):
    import continuum_swr_replay.analysis.pfc.canonical_events as canonical

    inventory = write_swr_inventory(
        tmp_path / "swr", {}, records={EPOCH_ID: {"status": "failed"}}
    )
    use_swr_source(monkeypatch, inventory)
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
# Every table built from the source carries its stamp; mixers refuse others
# --------------------------------------------------------------------------

STALE = "kay@an_earlier_run+artifacts@none"


def _stop_frame(stamp: str | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "animal": ["j16"],
            "session_date": [20210706],
            "epoch": [EPOCH],
            "stop_id": [1],
            "n_selected_events": [0],
        }
    )
    if stamp is not None:
        frame.attrs["swr_source"] = stamp
    return frame


def test_the_replay_tables_are_published_with_the_sources_stamp(
    source, monkeypatch, tmp_path
):
    from continuum_swr_replay.analysis.replay import prepare_data

    published: dict = {}
    monkeypatch.setattr(prepare_data, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        prepare_data,
        "publish_canonical",
        lambda frame,
        path,
        *,
        table,
        basename,
        require_swr_source: published.setdefault(basename, dict(frame.attrs)),
    )

    prepare_data._publish_canonical_tables(
        (("stop_table", "stop", _stop_frame()), ("event_table", "event", _stop_frame()))
    )

    assert {name: attrs["swr_source"] for name, attrs in published.items()} == {
        "stop_table": swr_source.swr_source_name(),
        "event_table": swr_source.swr_source_name(),
    }


def test_the_stop_only_bootstrap_publishes_with_the_sources_stamp(
    source, saved_decode, monkeypatch, tmp_path
):
    from continuum_swr_replay.analysis.replay import prepare_data

    monkeypatch.setattr(
        prepare_data,
        "load_behavioral_context",
        lambda session, from_cache: (pd.DataFrame(), pd.DataFrame(), object(), True),
    )
    monkeypatch.setattr(
        prepare_data, "build_stop_table", lambda *args, **kwargs: _stop_frame()
    )

    prepare_data.build_stop_table_scope(
        [saved_decode], from_cache=True, output_dir=tmp_path / "out"
    )

    stop = read_table(tmp_path / "out" / "stop_table")
    assert stop.attrs["swr_source"] == swr_source.swr_source_name()


def test_the_timebin_tables_are_published_with_the_sources_stamp(
    source, monkeypatch, offline, tmp_path
):
    script = offline("scripts.decoding.build_timebin_tables")
    key = {"animal": "j16", "session_date": 20210706, "epoch": EPOCH}
    session = {"nwb_file_name": NWB, "epoch_name": EPOCH}
    monkeypatch.setattr(script, "find_sessions", lambda max_per_animal: [session])
    monkeypatch.setattr(
        script,
        "build_one",
        lambda s: (pd.DataFrame([key]), pd.DataFrame([{**key, "n_swr_events": 0}])),
    )
    monkeypatch.setattr(script, "validate", lambda onset, dense, in_excluded: None)
    stamps: dict = {}
    monkeypatch.setattr(
        script,
        "publish_canonical",
        lambda frame, path, *, table, basename, require_swr_source: stamps.setdefault(
            table, frame.attrs.get("swr_source")
        ),
    )
    monkeypatch.setattr(
        sys, "argv", ["build_timebin_tables.py", "--output-dir", str(tmp_path)]
    )

    script.main()

    assert stamps == {
        "event_onset": swr_source.swr_source_name(),
        "stop_timebin": swr_source.swr_source_name(),
    }


def test_the_stop_table_rebuild_refuses_an_event_table_from_another_source(
    source, monkeypatch, offline, tmp_path
):
    script = offline("scripts.decoding.regenerate_stop_table")
    event_table = _stop_frame(STALE)
    monkeypatch.setattr(
        script, "find_sessions", lambda max_per_animal: [{"nwb_file_name": NWB}]
    )
    monkeypatch.setattr(script, "build_one", lambda s: _stop_frame())
    monkeypatch.setattr(script, "read_table", lambda path: event_table)
    monkeypatch.setattr(script, "require_covariate_schema", lambda frame, table: None)
    monkeypatch.setattr(sys, "argv", ["regenerate_stop_table.py", "--no-save"])

    with pytest.raises(SwrSourceError, match="an_earlier_run"):
        script.main()


@pytest.mark.parametrize("stale_shard", [False, True], ids=["current", "stale"])
def test_the_stop_shard_assembly_needs_every_shard_from_the_source(
    source, monkeypatch, tmp_path, stale_shard
):
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "decoding"))
    from scripts.decoding.assemble_replay_content_shards import (
        assemble_replay_stop_shards,
    )

    shards = []
    for index, stamp in enumerate(
        (swr_source.swr_source_name(), STALE if stale_shard else None)
    ):
        shard = tmp_path / f"shard{index}"
        shard.mkdir()
        frame = _stop_frame(stamp or swr_source.swr_source_name())
        frame["epoch"] = f"0{2 * index + 2}_r{index + 1}"
        frame["stop_id"] = index + 1
        write_table(frame, shard / "stop_table.parquet")
        shards.append(shard)

    if stale_shard:
        with pytest.raises(SwrSourceError, match="shard1"):
            assemble_replay_stop_shards(
                shards, output_dir=tmp_path / "out", expected_sessions=2
            )
    else:
        assemble_replay_stop_shards(
            shards, output_dir=tmp_path / "out", expected_sessions=2
        )
        assembled = read_table(tmp_path / "out" / "stop_table")
        assert assembled.attrs["swr_source"] == swr_source.swr_source_name()


def test_the_content_occupancy_refuses_an_arm_table_from_another_source(
    source, monkeypatch, tmp_path
):
    import scripts.content.analyze_content_occupancy as occupancy

    folder = tmp_path / "replay_content"
    folder.mkdir()
    arm = pd.DataFrame(
        {"is_swr": [True], "candidate_arm": [0], "arm_posterior_mass": [1.0]}
    )
    arm.attrs["swr_source"] = STALE
    write_table(arm, folder / "event_arm_table.parquet")
    monkeypatch.setattr(occupancy, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        occupancy, "require_covariate_schema", lambda frame, table: None
    )

    with pytest.raises(SwrSourceError, match="an_earlier_run"):
        occupancy.arm_occupancy()


def test_the_widloski_runner_stamps_its_table(source):
    # The runner stamps its events table as the canonical tables are stamped;
    # its own run is tested in test_run_memoryless_detection.py.
    text = (REPO / "scripts/decoding/run_memoryless_detection.py").read_text()

    assert "stamp_swr_source(detection.table)" in text


# --------------------------------------------------------------------------
# The figures draw and select the source's intervals
# --------------------------------------------------------------------------


def test_the_event_gallery_catalogs_against_the_source(source, tmp_path, offline):
    gallery = offline("scripts.figures.generate_event_gallery")
    nonlocal_event = np.zeros(TIME.size, dtype=int)
    nonlocal_event[(TIME >= 101.01) & (TIME <= 101.03)] = 1  # inside source SWR 2
    nonlocal_event[(TIME >= 103.02) & (TIME <= 103.06)] = 2  # inside the stored one
    probability = np.where(nonlocal_event > 0, 0.95, 0.0)
    path = tmp_path / f"{EPOCH_ID}_80_data_over_time.pkl"
    decode_frame(
        non_local_event=nonlocal_event, **{"Non-Local_Continuous_prob": probability}
    ).to_pickle(path)

    catalog = gallery._catalog_one_epoch(path)

    by_category = catalog.groupby("category")["event_label"].apply(sorted).to_dict()
    # NL 1 is SWR-contained, NL 2 non-SWR; source SWR 1 overlaps no NL event,
    # source SWR 3 was dropped by the artifact, and the stored ripple is unseen.
    assert by_category == {
        "swr_contained": [1],
        "non_swr": [2],
        "swr_not_nonlocal": [1],
    }


FIGURE_SCRIPTS = (
    "scripts/figures/generate_figure02.py",
    "scripts/figures/generate_figure03.py",
    "scripts/figures/generate_event_gallery.py",
)


@pytest.mark.parametrize("script", FIGURE_SCRIPTS)
def test_the_figures_draw_the_sources_intervals(script):
    text = (REPO / script).read_text()

    # The plotted ``data["ripple_times"]`` is replaced by the session's source
    # intervals from the source-read frame, before anything draws or gates it ...
    assert re.search(r'data\["ripple_times"\] = swr_intervals\(', text)
    # ... and load_data's stored ripples are never re-gated or unpickled around it.
    assert not re.search(
        r'build_speed_eligible_event_cohort\(\s*data\["ripple_times"\]', text
    )
    assert "pd.read_pickle(" not in text


# --------------------------------------------------------------------------
# Static: nothing else reads the stored ripples or a decode frame's pickle
# --------------------------------------------------------------------------

#: Files outside ``scripts/figures/archive`` that may name
#: the stored Spyglass ripples, and why each is not an analysis read.
STORED_RIPPLE_READERS = {
    "src/continuum_swr_replay/data_loaders/lfp.py": "the loader itself",
    "src/continuum_swr_replay/data_loaders/bandit_task.py": "load_data's decode input",
    "src/continuum_swr_replay/data_loaders/utils.py": (
        "get_training_timepoints: the PFC decoders' no_ripple encoding mask (the "
        "HPC decoders' comes from the source, decoding_common.hpc_decode_masks)"
    ),
    "src/continuum_swr_replay/data_loaders/constants.py": "register evidence, prose",
    "src/continuum_swr_replay/analysis/hippocampus/swr_source.py": (
        "prose naming what it replaces"
    ),
    "src/continuum_swr_replay/types.py": "load_data's return types",
    "src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py": (
        "get_data_over_time: the stored labels, which an HPC decode replaces "
        "with the source's (organize_results)"
    ),
    "src/continuum_swr_replay/analysis/contfrag/data_aggregation.py": (
        "get_contfrag_data_over_time: the stored labels, which an HPC decode "
        "replaces with the source's (organize_results)"
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
    # The figures set ``data["ripple_times"]`` from the source before drawing
    # (test_the_figures_draw_the_sources_intervals pins that).
    "scripts/figures/generate_figure02.py": "replaced by swr_intervals",
    "scripts/figures/generate_figure03.py": "replaced by swr_intervals",
    "scripts/figures/generate_event_gallery.py": "replaced by swr_intervals",
}

#: Files outside ``scripts/figures/archive`` that unpickle a
#: decode frame without the source, and why none of them reads its SWRs.
DECODE_FRAME_READERS = {
    "src/continuum_swr_replay/analysis/hippocampus/swr_source.py": "the reader",
    "scripts/qa/run_state_evidence.py": (
        "reads it through with_swr_source, and keeps the decoder's own non-local "
        "labels so every decoded event stays out of the running-only encodings"
    ),
    "scripts/qa/diagnose_trial_detection.py": "position and trials",
    "scripts/qa/compare_pfc_cohorts.py": "PFC bundles; names data_over_time_sha256",
    "scripts/figures/generate_supp_pipeline_data_flow.py": (
        "reads its decode frame through read_dot_with_swr_source; read_pickle "
        "reads the M3 fit"
    ),
    "scripts/exploratory/_augment_data_over_time.py": "adds posterior columns",
    "scripts/exploratory/_trunk_zscore_extract.py": "SWR_zscore and MUA_zscore",
    "scripts/exploratory/_trunk_zscore_timebin.py": "SWR_zscore and MUA_zscore",
    "scripts/qa/run_inventory_comparison.py": (
        "reads it through with_swr_source; artifact_event_drops of the frame as "
        "saved, from the one read its sha256 names"
    ),
    "scripts/decoding/run_carey_detection.py": (
        "the grid's first and last time, the span the marks are loaded over"
    ),
    "scripts/figures/generate_inventory_examples.py": (
        "grid, speed, posterior and position; its events are the comparison's tables"
    ),
    "scripts/qa/run_event_features.py": (
        "reads it through with_swr_source; its events are the comparison's tables"
    ),
}


def _python_files() -> list[Path]:
    """Every module under src/ and scripts/ but the retired figure archive."""
    archive = REPO / "scripts" / "figures" / "archive"
    return [
        path
        for folder in ("src", "scripts")
        for path in (REPO / folder).rglob("*.py")
        if not path.is_relative_to(archive)
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


def test_the_overlap_geometry_count_drops_the_events_the_mask_reaches(
    source, saved_decode, monkeypatch, offline, tmp_path
):
    """The script re-detects non-local events from the posterior, so it applies
    the one analysed-time rule itself: an event in excluded time [101.75,
    102.55] (the configured mask widened by 0.3 s) is not counted."""
    script = offline("scripts.content.quantify_swr_overlap_geometry")
    position = pd.DataFrame({"head_speed": np.zeros(TIME.size)}, index=TIME)
    states = ["Local", "No-Spike", "Non-Local Continuous", "Non-Local Fragmented"]
    probabilities = np.zeros((TIME.size, len(states)))
    for start, end in [(101.50, 101.60), (101.80, 101.90), (102.10, 102.20)]:
        probabilities[(TIME >= start) & (TIME <= end), 2] = 0.95
    results = xr.Dataset(
        {"acausal_state_probabilities": (("time", "states"), probabilities)},
        coords={"time": TIME, "states": states},
    )
    monkeypatch.setattr(
        script, "load_data", lambda *args, **kwargs: {"position_info": position}
    )
    monkeypatch.setattr(script, "load_results", lambda *a, **k: (results, None, None))
    seen: list = []

    def record(events, swrs):
        seen.append(events)
        raise _Stop

    monkeypatch.setattr(script, "classify_swr_overlap_geometry", record)

    with pytest.raises(_Stop):
        script.quantify_session(NWB, EPOCH, data_dir=tmp_path)

    np.testing.assert_allclose(
        seen[0][["start_time", "end_time"]].to_numpy(), [[101.50, 101.60]]
    )


def test_the_spike_evidence_baseline_lies_outside_excluded_time(source):
    """The null's matched non-event windows are analysed time too: none may
    touch excluded time [101.75, 102.55] (the configured mask widened by
    0.3 s)."""
    from scripts.qa import analyze_event_spike_evidence as spike_evidence

    probability = np.full(TIME.size, 0.5)
    windows = [(100.20, 100.24), (101.72, 101.76), (102.20, 102.24), (103.00, 103.04)]
    for start, end in windows:  # short, clearly local stretches
        probability[(TIME >= start) & (TIME <= end)] = 0.0
    dot = swr_source.with_swr_source(
        decode_frame(**{"Non-Local_Continuous_prob": probability})
    )

    events = spike_evidence.build_session_events(dot)

    baseline = events[events.group == "baseline"]
    # the stretch at 101.72 is cut where excluded time begins; the one at
    # 102.20 lies in it and is gone
    np.testing.assert_allclose(TIME[baseline.s.to_numpy()], [100.20, 101.72, 103.00])
    excluded = swr_source.excluded_intervals(EPOCH_ID)
    for s, e in zip(baseline.s, baseline.e, strict=True):
        assert not interval_mask(TIME[s : e + 1], excluded).any()
