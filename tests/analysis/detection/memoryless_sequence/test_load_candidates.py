"""``load_candidates`` reads a candidates file written under the former module path."""

from __future__ import annotations

import sys

import joblib
import pytest

from continuum_swr_replay.analysis.detection.memoryless_sequence import config, detect
from continuum_swr_replay.analysis.detection.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)

FORMER = f"{detect._FORMER_PACKAGE}.config"


@pytest.fixture
def former_path_pickle(tmp_path, monkeypatch):
    """A pickle of the config that names it under the former module path."""
    path = tmp_path / "e1_memoryless_candidates.pkl"
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, FORMER, config)
        patch.setattr(MemorylessSequenceConfig, "__module__", FORMER)
        joblib.dump(MemorylessSequenceConfig(), path)
    assert FORMER.encode() in path.read_bytes()
    return path


def test_a_plain_load_of_a_former_path_pickle_fails(former_path_pickle):
    with pytest.raises(ModuleNotFoundError):
        joblib.load(former_path_pickle)


def test_load_candidates_reads_it_and_leaves_no_alias(former_path_pickle):
    loaded = detect.load_candidates(former_path_pickle)
    assert loaded == MemorylessSequenceConfig()
    assert not [name for name in sys.modules if name.startswith(detect._FORMER_PACKAGE)]


def test_load_candidates_reads_a_current_pickle(tmp_path):
    path = tmp_path / "e1_memoryless_candidates.pkl"
    joblib.dump(MemorylessSequenceConfig(), path)
    assert detect.load_candidates(path) == MemorylessSequenceConfig()
