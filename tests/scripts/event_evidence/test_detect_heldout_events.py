"""The held-out event runner's wiring: what ``run_epoch`` reads, checks and
writes. The decode frame, the per-bin and the per-bin facts tables are made by
hand under a temporary processed-data root, with a synthetic SWR source; only
the event table's read is replaced."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence import nonlocal_restriction as nr
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.paths import (
    BIN_FACTS_NAME,
    CANONICAL_EVENTS_FOUND_NAME,
    EVENT_BIN_FACTS_DIR,
    EVENT_NONLOCAL_RESTRICTION_DIR,
    EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR,
    HELDOUT_EVENTS_NAME,
    HELDOUT_TIME_SUMS_NAME,
    NONLOCAL_RESTRICTION_BINS_NAME,
)
from continuum_swr_replay.runs.io import output_path
from scripts.event_evidence import _per_event_run
from scripts.event_evidence import detect_heldout_events as runner
from tests.support._restriction_tables import bin_facts_table, restriction_bins
from tests.support._swr_inventory_fixtures import use_swr_source, write_swr_inventory

EPOCH = "j1620210706_02_r1_HPC"
N = 1000
TIME = np.round(100.0 + np.arange(N) / 500.0, 6)
CANONICAL = {1: (100, 109), 2: (400, 409)}
"""The canonical events' labels and bins."""
HELDOUT = [(105, 114), (600, 609)]
"""The held-out events' bins: the first shares bins 105-109 with canonical
event 1 and overlaps the SWR; the second starts outside the visit."""
SWR = (TIME[110], TIME[120])
"""The one SWR: bins 110-120."""


def _trace(spans) -> np.ndarray:
    """0.9 over each span, 0.1 elsewhere."""
    prob = np.full(N, 0.1)
    for s, e in spans:
        prob[s : e + 1] = 0.9
    return prob


def _raw(labels: dict[int, tuple[int, int]] = CANONICAL) -> pd.DataFrame:
    """The decode frame as the decoder saved it: the canonical events' trace
    (two thirds continuous) and their labels."""
    prob = _trace(CANONICAL.values())
    label = np.zeros(N, dtype=np.int64)
    for value, (s, e) in labels.items():
        label[s : e + 1] = value
    return pd.DataFrame(
        {
            "animal": "j16",
            "day": 20210706,
            "epoch": "02_r1",
            "head_speed": np.zeros(N),
            "Non-Local_Continuous_prob": 2 * prob / 3,
            "Non-Local_Fragmented_prob": prob / 3,
            "non_local_event": label,
        },
        index=pd.Index(TIME, name="time"),
    )


def _bins() -> pd.DataFrame:
    """The fold-fit table: ``none`` 0, ``swr`` 1 and ``full`` 3 nats at every
    bin, the two-sided scores twice those, the SWR's bins in the category
    ``immobile_in_swr``, and the held-out trace (half continuous)."""
    one = np.ones(N, dtype=np.float32)
    in_swr = (np.arange(N) >= 110) & (np.arange(N) <= 120)
    prob = _trace(HELDOUT).astype(np.float32)
    states = dict.fromkeys(nr.STATE_PROBABILITY_COLUMNS.values(), 0.0 * one)
    states["full_prob_nonlocal_continuous"] = prob / 2
    states["full_prob_nonlocal_fragmented"] = prob / 2
    return restriction_bins(
        TIME,
        {
            "none": 0 * one,
            "swr": one,
            "full": 3 * one,
            "none_two_sided": 0 * one,
            "swr_two_sided": 2 * one,
            "full_two_sided": 6 * one,
        },
        in_swr,
        np.where(in_swr, "immobile_in_swr", "immobile_outside_swr"),
        **states,
    )


def _facts() -> pd.DataFrame:
    """One spike per bin; bins 0-499 inside a well visit."""
    return bin_facts_table(TIME, np.ones(N), np.arange(N) < 500)


TABLE = pd.DataFrame(
    {
        "event_id": np.array([11, 22], dtype="uint64"),
        "non_local_event": [1, 2],
        "is_swr": [False, False],
        "visit_context": ["in_visit", "spans_visit_end"],
        "animal": "j16",
        "session_date": 20210706,
        "epoch": "02_r1",
    }
)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    use_swr_source(monkeypatch, write_swr_inventory(tmp_path, {EPOCH: [SWR]}))
    _raw().to_pickle(decode_paths(EPOCH, tmp_path).data_over_time)
    for folder, name, table in (
        (EVENT_NONLOCAL_RESTRICTION_DIR, NONLOCAL_RESTRICTION_BINS_NAME, _bins()),
        (EVENT_BIN_FACTS_DIR, BIN_FACTS_NAME, _facts()),
    ):
        (tmp_path / folder).mkdir()
        table.to_parquet(output_path(EPOCH, tmp_path / folder, name))
    monkeypatch.setattr(runner, "read_swr_table", lambda path, columns: TABLE[columns])
    return tmp_path


def _run_epoch(data_dir) -> tuple[dict, dict[str, pd.DataFrame]]:
    output = data_dir / "out"
    output.mkdir()
    record = runner.run_epoch(EPOCH, data_dir / "event_table.parquet", data_dir, output)
    names = (HELDOUT_EVENTS_NAME, CANONICAL_EVENTS_FOUND_NAME, HELDOUT_TIME_SUMS_NAME)
    return record, {
        name: pd.read_parquet(output_path(EPOCH, output, name)) for name in names
    }


def test_run_epoch_writes_the_heldout_events_the_canonical_found_and_the_time(
    data_dir,
):
    record, tables = _run_epoch(data_dir)
    events = tables[HELDOUT_EVENTS_NAME]
    assert list(zip(events["s"], events["e"], strict=True)) == HELDOUT
    assert events["heldout_event_id"].tolist() == [f"{EPOCH}:1", f"{EPOCH}:2"]
    assert events["group"].tolist() == ["SWR", "no SWR"]
    assert events["starts_in_well_visit"].tolist() == [True, False]
    assert events["shares_bin_with_canonical"].tolist() == [True, False]
    assert set(events["kind"]) == {"event"} and set(events["animal"]) == {"j16"}
    # the window sums: 10 bins each, the first with 9 inside the SWR
    assert events["n_bins"].tolist() == [10, 10]
    assert events["n_bins_outside_swr"].tolist() == [5, 10]
    assert events["n_spikes"].tolist() == [10, 10]
    np.testing.assert_allclose(events["full"] - events["swr"], [20, 20])
    np.testing.assert_allclose(
        events["full_outside_swr"] - events["swr_outside_swr"], [10, 20]
    )
    assert events["n_bins_outside_swr_in_visit"].tolist() == [5, 0]
    # the canonical events: 1 shares bins with a held-out event, 2 does not
    found = tables[CANONICAL_EVENTS_FOUND_NAME]
    assert found["event_id"].tolist() == [11, 22]
    assert found["group"].tolist() == ["no SWR", "no SWR"]
    assert found["shares_bin_with_heldout"].tolist() == [True, False]
    # the time sums mark both held-out events' bins, wherever they start
    sums = tables[HELDOUT_TIME_SUMS_NAME]
    assert sums["n_bins"].sum() == N
    assert sums.loc[sums["in_event_window"], "n_bins"].sum() == 20
    assert set(sums["epoch_id"]) == {EPOCH}
    assert record["status"] == "ok"
    assert record["n_heldout_events"] == 2
    assert record["n_canonical_events_sharing_a_bin"] == 1
    assert record["bin_s"] == pytest.approx(0.002)
    json.dumps(record, allow_nan=False)


def test_a_facts_time_that_is_not_the_decodes_raises(data_dir):
    facts = _facts()
    facts["time"] += 1e-3
    facts.to_parquet(output_path(EPOCH, data_dir / EVENT_BIN_FACTS_DIR, "bin_facts"))
    with pytest.raises(ValueError, match="per-bin facts' time"):
        _run_epoch(data_dir)


def test_canonical_labels_that_are_not_the_event_tables_raise(data_dir, monkeypatch):
    monkeypatch.setattr(
        runner, "read_swr_table", lambda path, columns: TABLE.iloc[:1][columns]
    )
    with pytest.raises(ValueError, match="not the event table's"):
        _run_epoch(data_dir)


def _failed_record(data_dir, monkeypatch) -> dict:
    """Run the one epoch through the batch loop, which must fail it, and return
    its record."""
    monkeypatch.setattr(_per_event_run, "ProcessPoolExecutor", ThreadPoolExecutor)
    output = data_dir / "out"
    with pytest.raises(SystemExit, match="1 epochs failed"):
        _per_event_run.run(
            epochs=[EPOCH],
            task=runner.run_epoch,
            task_args=(data_dir / "e.parquet", data_dir, output),
            output=output,
            name=HELDOUT_EVENTS_NAME,
            header={
                "produced_by": "t",
                "settings": {},
                "inputs": {},
                "provenance": {},
            },
            overwrite=False,
            max_workers=1,
        )
    record = json.loads((output / "manifest.json").read_text())["epochs"][EPOCH]
    assert record["status"] == "failed"
    assert not output_path(EPOCH, output, HELDOUT_EVENTS_NAME).exists()
    return record


def test_canonical_labels_the_rule_does_not_give_fail_the_epoch(data_dir, monkeypatch):
    """The decode's canonical event 1 ends a bin before its trace does: the rule
    on its own probabilities does not give its labels, and the epoch fails."""
    _raw({1: (100, 108), 2: (400, 409)}).to_pickle(
        decode_paths(EPOCH, data_dir).data_over_time
    )
    record = _failed_record(data_dir, monkeypatch)
    assert "other labels than its canonical ones at 1 bins" in record["error"]


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        # event 1 (bins 100-109) ends before the SWR (110-120): no strict overlap
        ("is_swr", [True, False], "another SWR group than the event table's is_swr"),
        # event 2 (bins 400-409) starts inside the visit (bins 0-499)
        (
            "visit_context",
            ["in_visit", "track_leaf_arm"],
            "another start inside a well visit than select_visit_context",
        ),
    ],
)
def test_an_event_tables_group_or_visit_start_the_rule_does_not_give_fails_the_epoch(
    data_dir, monkeypatch, column, values, message
):
    monkeypatch.setattr(
        runner,
        "read_swr_table",
        lambda path, columns: TABLE.assign(**{column: values})[columns],
    )
    record = _failed_record(data_dir, monkeypatch)
    assert f"{message} for 1 of the epoch's events" in record["error"]


def test_main_runs_the_restriction_runs_ok_epochs_and_records_its_inputs(
    tmp_path, monkeypatch
):
    ran = {}
    manifest = {"epochs": {EPOCH: {"status": "ok"}, "x": {"status": "failed"}}}
    monkeypatch.setattr(runner, "event_table_path", lambda data_dir: tmp_path / "e")
    monkeypatch.setattr(runner, "file_input", lambda path: {"sha256": "a", "path": "e"})
    monkeypatch.setattr(runner, "run_input", lambda run_dir: {"sha256": run_dir.name})
    monkeypatch.setattr(runner, "provenance_record", lambda root: {})
    monkeypatch.setattr(runner, "read_manifest", lambda folder: manifest)
    monkeypatch.setattr(runner, "run", lambda **kwargs: ran.update(kwargs))
    runner.main(["--data-dir", str(tmp_path), "--restriction-dir", "fold_fit"])
    assert ran["epochs"] == [EPOCH]
    assert ran["output"] == tmp_path / EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR
    assert ran["header"]["inputs"] == {
        "event_table": {"sha256": "a", "path": "e"},
        "nonlocal_restriction": {"sha256": "fold_fit"},
        "bin_facts": {"sha256": EVENT_BIN_FACTS_DIR},
    }
