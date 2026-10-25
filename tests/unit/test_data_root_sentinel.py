"""No test reaches the live data root unless it asks to.

The root ``tests/conftest.py`` points ``$CSR_PROCESSED_DATA_DIR``, the one place
every data root is resolved from (``paths.processed_data_dir()``), at a sentinel
file before any test module is imported. These tests pin what a test that sets
no root gets, and that the two ways of asking for one work.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from continuum_swr_replay import parameters, paths
from tests import conftest


def test_the_conftest_module_is_the_one_pytest_loaded():
    """Importing it again would make a second sentinel and lose the live root."""
    assert (
        Path(os.environ[paths.PROCESSED_DATA_DIR_ENV])
        == conftest.NO_PROCESSED_DATA_ROOT
    )


def test_a_test_without_a_root_reads_and_writes_nothing():
    root = paths.processed_data_dir()

    assert root == conftest.NO_PROCESSED_DATA_ROOT
    assert root.is_file()
    assert parameters.PROCESSED_DATA_DIR == root
    assert not paths.table_path("stop_table").exists()
    assert list(root.glob("*")) == []
    with pytest.raises(NotADirectoryError):
        paths.table_path("stop_table").read_bytes()
    with pytest.raises(NotADirectoryError):
        paths.model_dir("per_bin_rate").mkdir(parents=True)
    with pytest.raises(NotADirectoryError):
        (paths.swr_inventory_root() / "manifest.json").write_text("{}")
    with pytest.raises(FileExistsError):
        root.mkdir(exist_ok=True)


def test_a_child_process_inherits_the_sentinel():
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from continuum_swr_replay.paths import processed_data_dir; "
            "print(processed_data_dir())",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert Path(out.stdout.strip()) == conftest.NO_PROCESSED_DATA_ROOT


def test_processed_data_root_is_the_tests_own(processed_data_root, tmp_path):
    assert processed_data_root == tmp_path / "Processed-Data"
    assert paths.processed_data_dir() == processed_data_root
    assert paths.replay_content_dir() == processed_data_root / "replay_content"


def test_replay_content_root_is_under_it(replay_content_root, processed_data_root):
    assert replay_content_root == paths.replay_content_dir()
    assert replay_content_root.is_dir()


@pytest.mark.real_data
def test_a_real_data_test_reads_the_live_root(live_processed_data_dir):
    assert paths.processed_data_dir() == live_processed_data_dir
    assert live_processed_data_dir.is_dir()
