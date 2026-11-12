"""``CSR_PROCESSED_DATA_DIR`` moves the data root; without it the root is the checkout's."""

import os
import subprocess
import sys
from pathlib import Path

import continuum_swr_replay.parameters as parameters


def _root_in_fresh_process(env: dict[str, str]) -> Path:
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from continuum_swr_replay.parameters import PROCESSED_DATA_DIR; "
            "print(PROCESSED_DATA_DIR)",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip())


def test_default_is_the_checkouts_processed_data():
    env = {k: v for k, v in os.environ.items() if k != "CSR_PROCESSED_DATA_DIR"}
    assert _root_in_fresh_process(env) == parameters.ROOT_DIR / "Processed-Data"


def test_the_environment_variable_sets_the_root(tmp_path):
    env = {**os.environ, "CSR_PROCESSED_DATA_DIR": str(tmp_path)}
    assert _root_in_fresh_process(env) == tmp_path
