"""The HPC decoders' masks: ``is_missing`` and ``is_training`` from the SWR source.

``decode_masks.hpc_artifact_intervals`` and ``hpc_decode_masks`` build them
from the configured Kay inventory and artifact run, here synthetic run folders
written as the runners write them. Offline: nothing touches Spyglass or
``Processed-Data``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.events import swr_source
from continuum_swr_replay.analysis.hippocampus import decode_masks
from continuum_swr_replay.analysis.hippocampus.decode_masks import (
    hpc_artifact_intervals,
    hpc_decode_masks,
    require_hpc_decode_sources,
)
from tests.support._swr_inventory_fixtures import (
    ARTIFACT_RUN_ID,
    RUN_ID,
    use_swr_source,
    write_artifact_run,
    write_swr_inventory,
)

NWB_FILE_NAME = "j1620210706_.nwb"
EPOCH_NAME = "02_r1"
EPOCH = "j1620210706_02_r1_HPC"
TIME = np.round(100.0 + np.arange(5000) / 500.0, 6)
"""A 500 Hz decode grid (the position grid), 100.000-109.998 s."""

EVENTS = [
    (100.500, 100.560),  # immobile
    (102.400, 102.600),  # moving: still left out of encoding
    (104.000, 104.030),  # overlaps an artifact
]
CORES = [("spk", 104.11, 104.11), ("ca1", 107.0, 107.0)]
"""A spike line and a CA1 step: the one mask pads them to [104.01, 104.21]
and [106.7, 107.3]."""
MASK_INTERVALS = [(104.01, 104.21), (106.7, 107.3)]


@pytest.fixture
def sources(tmp_path, monkeypatch):
    """The SWR source configured with a synthetic inventory and artifact run."""
    inventory = write_swr_inventory(
        tmp_path, {EPOCH: EVENTS}, artifact_run_id=ARTIFACT_RUN_ID
    )
    artifacts = write_artifact_run(tmp_path, {EPOCH: CORES})
    use_swr_source(monkeypatch, inventory, artifacts)


def _position_info() -> pd.DataFrame:
    speed = np.zeros(TIME.size)
    speed[(TIME >= 102.0) & (TIME <= 102.8)] = 10.0
    return pd.DataFrame({"head_speed": speed}, index=pd.Index(TIME, name="time"))


def _inside(intervals) -> np.ndarray:
    inside = np.zeros(TIME.size, dtype=bool)
    for start, end in intervals:
        inside |= (TIME >= start) & (TIME <= end)
    return inside


def test_the_decoders_get_the_artifact_runs_one_mask(sources):
    np.testing.assert_allclose(
        hpc_artifact_intervals(NWB_FILE_NAME, EPOCH_NAME).to_numpy(), MASK_INTERVALS
    )


def test_is_missing_is_the_decoder_mask_on_the_decode_grid(sources):
    masks = hpc_decode_masks(
        NWB_FILE_NAME,
        EPOCH_NAME,
        _position_info(),
        hpc_artifact_intervals(NWB_FILE_NAME, EPOCH_NAME),
        "no_ripple",
    )

    assert masks.is_missing.dtype == bool and masks.is_missing.shape == TIME.shape
    np.testing.assert_array_equal(masks.is_missing, _inside(MASK_INTERVALS))


def test_encoding_leaves_out_every_inventory_event_and_the_mask(sources):
    """Every Kay event, moving or not, and every masked bin leave encoding."""
    masks = hpc_decode_masks(
        NWB_FILE_NAME,
        EPOCH_NAME,
        _position_info(),
        hpc_artifact_intervals(NWB_FILE_NAME, EPOCH_NAME),
        "no_ripple",
    )

    expected = ~_inside(EVENTS) & ~_inside(MASK_INTERVALS)
    np.testing.assert_array_equal(masks.is_training, expected)
    assert not masks.is_training[(TIME >= 102.4) & (TIME <= 102.6)].any()


def test_training_on_all_still_leaves_out_the_mask(sources):
    masks = hpc_decode_masks(
        NWB_FILE_NAME,
        EPOCH_NAME,
        _position_info(),
        hpc_artifact_intervals(NWB_FILE_NAME, EPOCH_NAME),
        "all",
    )

    np.testing.assert_array_equal(masks.is_training, ~_inside(MASK_INTERVALS))


def test_the_record_names_the_sources_and_the_seconds(sources):
    masks = hpc_decode_masks(
        NWB_FILE_NAME,
        EPOCH_NAME,
        _position_info(),
        hpc_artifact_intervals(NWB_FILE_NAME, EPOCH_NAME),
        "no_ripple",
    )

    record = masks.record
    assert record["swr_source"] == f"kay@{RUN_ID}+artifacts@{ARTIFACT_RUN_ID}"
    assert record["swr_inventory_run_id"] == RUN_ID
    assert record["artifact_run_id"] == ARTIFACT_RUN_ID
    assert record["padding_s"] == {"S": 1.0, "T_lfp": 0.3, "T_spike": 0.1}
    assert record["training_type"] == "no_ripple"
    assert record["n_bins"] == TIME.size
    assert record["bin_s"] == pytest.approx(0.002)
    assert record["n_missing_bins"] == int(_inside(MASK_INTERVALS).sum())
    assert record["missing_s"] == pytest.approx(record["n_missing_bins"] * 0.002)
    assert record["n_swr_events"] == len(EVENTS)
    removed = ~(~_inside(EVENTS) & ~_inside(MASK_INTERVALS))
    assert record["n_encoding_removed_bins"] == int(removed.sum())
    assert record["encoding_removed_s"] == pytest.approx(removed.sum() * 0.002)


@pytest.mark.parametrize("unset", ["SWR_INVENTORY_RUN_ID", "ARTIFACT_INTERVALS_RUN_ID"])
def test_an_unset_run_id_is_refused(sources, monkeypatch, unset):
    """Fail closed: an HPC decode needs both runs named."""
    monkeypatch.setattr(swr_source, unset, None)

    with pytest.raises(ValueError, match=unset):
        require_hpc_decode_sources()
    with pytest.raises(ValueError, match=unset):
        hpc_artifact_intervals(NWB_FILE_NAME, EPOCH_NAME)


def test_both_run_ids_unset_are_named(monkeypatch):
    monkeypatch.setattr(swr_source, "SWR_INVENTORY_RUN_ID", None)
    monkeypatch.setattr(swr_source, "ARTIFACT_INTERVALS_RUN_ID", None)

    with pytest.raises(ValueError) as refused:
        decode_masks.require_hpc_decode_sources()

    assert "SWR_INVENTORY_RUN_ID and swr_source.ARTIFACT_INTERVALS_RUN_ID" in str(
        refused.value
    )


@pytest.mark.parametrize("recorded", [None, "otherartifacts"])
def test_an_inventory_masked_by_another_artifact_run_is_refused(
    tmp_path, monkeypatch, recorded
):
    """The inventory's own manifest names the artifact run its events were
    masked with (none for a run made before the SWR runner masked artifacts,
    such as R3's); an HPC decode under another artifact run is refused, and
    the error names both run ids."""
    inventory = write_swr_inventory(tmp_path, {EPOCH: EVENTS}, artifact_run_id=recorded)
    artifacts = write_artifact_run(tmp_path, {EPOCH: CORES})
    use_swr_source(monkeypatch, inventory, artifacts)

    for refuse in (
        decode_masks.require_hpc_decode_sources,
        lambda: hpc_artifact_intervals(NWB_FILE_NAME, EPOCH_NAME),
    ):
        with pytest.raises(ValueError) as refused:
            refuse()
        message = str(refused.value)
        assert RUN_ID in message and ARTIFACT_RUN_ID in message
        assert (recorded or "no artifact run") in message


def test_an_inventory_masked_by_the_configured_artifact_run_is_accepted(sources):
    decode_masks.require_hpc_decode_sources()


def test_a_configured_run_missing_the_epoch_is_refused(tmp_path, monkeypatch):
    artifacts = write_artifact_run(
        tmp_path, {"chimi20200225_02_r1_HPC": CORES}, run_id="otherrun"
    )
    inventory = write_swr_inventory(
        tmp_path, {EPOCH: EVENTS}, artifact_run_id="otherrun"
    )
    use_swr_source(monkeypatch, inventory, artifacts)

    with pytest.raises(swr_source.SwrSourceError, match="not in the artifact run"):
        hpc_artifact_intervals(NWB_FILE_NAME, EPOCH_NAME)
