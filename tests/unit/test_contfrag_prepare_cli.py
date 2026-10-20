"""The ContFrag table build can join a staged qvals table and keep the canonical cache.

``prepare_contfrag_replay_content_data.py`` reads the qvals table from
``--replay-content-dir``, and by default its behavioral cache (trials and track
graph per session) from that folder's ``_session_cache``. ``--behavior-cache-dir``
overrides only the cache, so a build pointed at a staged replay-content folder
still reads the canonical cache (Stage 5, item 5, of
``docs/design/2026-10-03-full-pipeline-refresh-design.md``).

Offline: the script is imported with ``spyglass`` stubbed in ``sys.modules``, as
``test_decoding_save_dir.py`` does. Nothing is built or written: the test parses
arguments and applies ``configure_paths``, which resolves paths and sets
``SESSION_CACHE_DIR``.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

SCRIPT = "scripts.tables.prepare_contfrag_replay_content_data"


@pytest.fixture
def contfrag_prepare(monkeypatch):
    """The script, imported offline with ``spyglass`` stubbed.

    If the import is the first, the module is forgotten afterwards, so a later
    import does not get one bound to the stub.
    """
    common = types.ModuleType("spyglass.common")
    common.PositionIntervalMap = type("PositionIntervalMap", (), {})
    monkeypatch.setitem(sys.modules, "spyglass", types.ModuleType("spyglass"))
    monkeypatch.setitem(sys.modules, "spyglass.common", common)
    fresh = SCRIPT not in sys.modules

    yield importlib.import_module(SCRIPT)

    if fresh:
        sys.modules.pop(SCRIPT, None)
        parent, _, child = SCRIPT.rpartition(".")
        vars(sys.modules[parent]).pop(child, None)


def test_behavior_cache_dir_overrides_the_replay_content_cache(
    contfrag_prepare, monkeypatch
):
    """``--behavior-cache-dir`` replaces the cache, not the qvals root."""
    prepare_data = contfrag_prepare.replay_prepare_data
    # configure_paths sets this global; monkeypatch puts the original back.
    monkeypatch.setattr(
        prepare_data, "SESSION_CACHE_DIR", prepare_data.SESSION_CACHE_DIR
    )
    parser = contfrag_prepare.build_parser()

    _, replay_content_dir = contfrag_prepare.configure_paths(
        parser.parse_args(["--replay-content-dir", "/a", "--behavior-cache-dir", "/b"])
    )
    assert prepare_data.SESSION_CACHE_DIR == Path("/b")
    assert replay_content_dir == Path("/a")

    contfrag_prepare.configure_paths(parser.parse_args(["--replay-content-dir", "/a"]))
    assert prepare_data.SESSION_CACHE_DIR == Path("/a/_session_cache")
