"""``paths``: each output location equals the inline spelling it replaced.

A migrated constant that resolved to a different ``Path`` would silently read a
stale artifact, so every constant is pinned to the literal it replaced, and
``table_path`` is checked over the whole canonical-table registry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from continuum_swr_replay import paths
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

RC = PROCESSED_DATA_DIR / "replay_content"


def test_roots_equal_the_literals_they_replaced():
    assert paths.REPLAY_CONTENT_DIR == RC
    assert paths.CONTFRAG_REPLAY_CONTENT_DIR == (
        PROCESSED_DATA_DIR / "contfrag_replay_content"
    )
    assert paths.DETECTION_COMPARISON_DIR == (
        PROCESSED_DATA_DIR / "replay_detection_comparison"
    )
    assert paths.SWR_INVENTORY_ROOT == (
        PROCESSED_DATA_DIR / "replay_detection_comparison" / "swr"
    )
    assert paths.ARTIFACT_INTERVALS_DIR == PROCESSED_DATA_DIR / "artifact_intervals"


def test_model_dirs_equal_the_literals_they_replaced():
    assert paths.MODEL_DIRS == {
        "content_arm": RC / "models",
        "content_patch": RC / "models_patch",
        "per_bin_rate": RC / "models_per_bin_rate",
        "per_stop_rate": RC / "models_per_stop_rate",
    }


def test_registry_is_the_one_parity_reads():
    from continuum_swr_replay.tables import parity

    assert parity.BASENAME_TO_TABLE is paths.BASENAME_TO_TABLE
    assert parity.CANONICAL_TABLE_BASENAMES is paths.CANONICAL_TABLE_BASENAMES
    assert paths.CANONICAL_TABLE_BASENAMES == (
        "event_table",
        "event_arm_table",
        "event_patch_table",
        "event_sequence_table",
        "stop_table",
        "qvals_table",
        "event_onset_table",
        "stop_timebin_table",
    )


@pytest.mark.parametrize("name", paths.CANONICAL_TABLE_BASENAMES)
@pytest.mark.parametrize("fmt", ["parquet", "pkl", ""])
def test_table_path_for_every_registry_name(name, fmt):
    directory = RC / "models_per_bin_rate" if name == "qvals_table" else RC
    file = f"{name}.{fmt}" if fmt else name
    assert paths.table_path(name, fmt=fmt) == directory / file
    root = Path("/staging/root")
    directory = root / "models_per_bin_rate" if name == "qvals_table" else root
    assert paths.table_path(name, root, fmt=fmt) == directory / file


def test_table_path_defaults_to_parquet_under_replay_content():
    assert paths.table_path("stop_table") == RC / "stop_table.parquet"
    assert paths.table_path("qvals_table") == (
        RC / "models_per_bin_rate" / "qvals_table.parquet"
    )


@pytest.mark.parametrize(
    "name", ["events", "stop_table.parquet", "event_arm_table_swr_HPC", ""]
)
def test_table_path_refuses_a_name_outside_the_registry(name):
    with pytest.raises(KeyError, match="not a canonical table"):
        paths.table_path(name)
