"""Shared pytest fixtures for continuum-swr-replay tests.

Only fixtures that have downstream callers live here. Single-use
fixtures stay in the test module that uses them so renames / deletions
don't require cross-file coordination.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path

# The suite runs on a shared GPU host, and JAX preallocates most of every card
# it can see the first time it is used. A test must never do that to somebody
# else's job, so the suite is CPU-only unless the caller has deliberately said
# otherwise. Set before any test imports JAX -- the backend is chosen on first
# use, and this module is imported before every test module.
#
# ``JAX_PLATFORMS`` rather than ``CUDA_VISIBLE_DEVICES`` alone because it also
# stops the CUDA plugin being loaded at all: several tests fan out over
# processes, and paying for the CUDA runtime in every child makes the suite
# several times slower for a backend no test wants. Both are set; the second is
# what a child inherits if something clears the first.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# No test reads or writes the live data root unless it asks to. Every data root
# is resolved when it is used, from ``$CSR_PROCESSED_DATA_DIR``
# (``paths.processed_data_dir()``), so that variable is the one place a test
# moves them -- its child processes included. It is set here, before any test
# module (and so any package module) is imported, to a sentinel: a regular file,
# not a folder. Every path under it is unusable: a read raises
# ``NotADirectoryError``, a write (``mkdir``, ``open``) raises
# ``NotADirectoryError`` or ``FileExistsError``, ``exists()`` is False and a
# glob finds nothing. A test that needs a data root takes ``processed_data_root``
# (a folder under its ``tmp_path``); a ``real_data`` test gets the live root back
# (``real_data_reads_the_live_root``).
PROCESSED_DATA_DIR_ENV = "CSR_PROCESSED_DATA_DIR"
#: The root a ``real_data`` test reads: the caller's ``$CSR_PROCESSED_DATA_DIR``,
#: or None for the checkout's ``Processed-Data/``.
LIVE_PROCESSED_DATA_DIR = os.environ.get(PROCESSED_DATA_DIR_ENV)
NO_PROCESSED_DATA_ROOT = (
    Path(tempfile.mkdtemp(prefix="csr-tests-"))
    / "NO_PROCESSED_DATA_ROOT__use_the_processed_data_root_fixture"
)
NO_PROCESSED_DATA_ROOT.write_text(
    "Stand-in for the processed-data root during the test suite; see tests/conftest.py.\n"
)
os.environ[PROCESSED_DATA_DIR_ENV] = str(NO_PROCESSED_DATA_ROOT)

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest_plugins = ["pytester"]


def pytest_unconfigure(config):
    """Remove the sentinel root's folder, and whatever a test wrote beside it."""
    shutil.rmtree(NO_PROCESSED_DATA_ROOT.parent, ignore_errors=True)


# ============================================================================
# Real-data tests are opt-in
# ============================================================================

REAL_DATA_SKIP_REASON = "real_data: opt in with -m real_data"


def pytest_collection_modifyitems(config, items):
    """Skip ``real_data`` tests unless the ``-m`` expression names the marker.

    A real-data test reads ``Processed-Data``, a raw NWB file or the Spyglass
    database. Skipping them by default makes a plain ``pytest`` test the same
    code on every machine. Under ``-m real_data`` they run, and missing data
    fails them rather than skipping them. The marker must appear as a whole
    word, so a marker such as ``real_data_lite`` is not an opt-in.
    """
    if re.search(r"\breal_data\b", config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason=REAL_DATA_SKIP_REASON)
    for item in items:
        if item.get_closest_marker("real_data") is not None:
            item.add_marker(skip)


# ============================================================================
# The data root
# ============================================================================


@pytest.fixture
def processed_data_root(tmp_path, monkeypatch) -> Path:
    """An empty processed-data root under ``tmp_path``, the one every reader uses.

    Points ``$CSR_PROCESSED_DATA_DIR`` at ``tmp_path / "Processed-Data"`` for the
    test, so ``paths.processed_data_dir()`` and every root under it
    (``replay_content_dir()``, ``swr_inventory_root()``, ...) resolve there, in
    this process and in the processes it starts. Lay a test's inputs out as
    ``Processed-Data`` is laid out.
    """
    root = tmp_path / "Processed-Data"
    root.mkdir()
    monkeypatch.setenv(PROCESSED_DATA_DIR_ENV, str(root))
    return root


@pytest.fixture
def replay_content_root(processed_data_root) -> Path:
    """``paths.replay_content_dir()`` under ``processed_data_root``, created."""
    root = processed_data_root / "replay_content"
    root.mkdir()
    return root


@pytest.fixture(scope="session")
def live_processed_data_dir() -> Path:
    """The root ``real_data`` tests read (the caller's, or the checkout's).

    For a module- or session-scoped fixture of a ``real_data`` module, which
    ``real_data_reads_the_live_root`` does not reach: pass it as the root.
    """
    if LIVE_PROCESSED_DATA_DIR is not None:
        return Path(LIVE_PROCESSED_DATA_DIR)
    return Path(__file__).resolve().parents[1] / "Processed-Data"


@pytest.fixture(autouse=True)
def real_data_reads_the_live_root(request, monkeypatch):
    """A ``real_data`` test reads the live data root, as production does.

    Function-scoped, like ``real_data_sees_the_configured_artifact_run``, so a
    module- or session-scoped fixture still sees the sentinel root; one that
    reads real data passes ``live_processed_data_dir()`` as its root.
    """
    if request.node.get_closest_marker("real_data") is None:
        return
    if LIVE_PROCESSED_DATA_DIR is None:
        monkeypatch.delenv(PROCESSED_DATA_DIR_ENV)
    else:
        monkeypatch.setenv(PROCESSED_DATA_DIR_ENV, LIVE_PROCESSED_DATA_DIR)


# ============================================================================
# The configured artifact run
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def no_configured_artifact_run():
    """Every test starts with no artifact run configured.

    ``swr_source.ARTIFACT_INTERVALS_RUN_ID`` names a real run under
    Processed-Data. A unit test must not read it, so every test and every
    module-scoped fixture sees None unless a test sets a run of its own (most
    use a ``tmp_path`` run; ``monkeypatch`` restores None after it). Session
    scope, so the module-scoped fixtures that run a pipeline end to end see it
    too. The committed value is pinned by
    ``test_swr_source.TestTheConfiguredInventory``. Yields that committed value,
    which ``real_data`` tests get back.
    """
    from continuum_swr_replay.analysis.events import swr_source

    configured = swr_source.ARTIFACT_INTERVALS_RUN_ID
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(swr_source, "ARTIFACT_INTERVALS_RUN_ID", None)
        yield configured


@pytest.fixture(autouse=True)
def real_data_sees_the_configured_artifact_run(
    request, no_configured_artifact_run, monkeypatch
):
    """A ``real_data`` test reads the committed artifact run, as production does.

    The canonical tables record the SWR source they were built from, and the
    readers refuse a table whose source differs from the configured one. With
    the run forced to None, every canonical table would be refused.

    This fixture is function-scoped, so it reaches the test function only. A
    module- or session-scoped fixture in a ``real_data`` module still sees
    None; one that reads a table through ``analysed_time.with_swr_source`` must
    set the run itself.
    """
    if request.node.get_closest_marker("real_data") is None:
        return
    from continuum_swr_replay.analysis.events import swr_source

    monkeypatch.setattr(
        swr_source, "ARTIFACT_INTERVALS_RUN_ID", no_configured_artifact_run
    )


# ============================================================================
# Track Graph Fixtures
# ============================================================================


@pytest.fixture
def simple_linear_track() -> nx.Graph:
    """3-node linear track: 0 --10cm-- 1 --10cm-- 2.

    The most-used graph fixture in the suite (~100 callers).
    """
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2)])
    nx.set_edge_attributes(graph, {(0, 1): 10.0, (1, 2): 10.0}, "distance")
    return graph


@pytest.fixture
def simplified_star_track() -> nx.Graph:
    """7-node star graph: six wells (nodes 0-5) connect directly to a single
    center node (6) via six 20cm edges.

    NOT the real bandit topology — the production task is a 10-node, 9-edge
    spider with patch junctions between wells and center.
    Tests that depend on the real topology (multi-segment paths, trunk-arm
    semantics, patch junctions) should construct their own graph; this
    fixture is a minimal substrate for tests where only the well-to-center
    degree-1 property matters.
    """
    graph = nx.Graph()
    edges = [
        (6, 0),
        (6, 1),  # Patch 1
        (6, 2),
        (6, 3),  # Patch 2
        (6, 4),
        (6, 5),  # Patch 3
    ]
    graph.add_edges_from(edges)
    distances = dict.fromkeys(edges, 20.0)
    nx.set_edge_attributes(graph, distances, "distance")
    return graph


# ============================================================================
# Spike Data Fixtures
# ============================================================================


@pytest.fixture
def mock_spike_times() -> dict[str, list[np.ndarray]]:
    """Mock spike times for HPC (4 tetrodes) + mPFC (3 units) over 10 s."""
    rng = np.random.RandomState(42)
    return {
        "HPC": [
            np.sort(rng.uniform(0, 10, size=100)),
            np.sort(rng.uniform(0, 10, size=120)),
            np.sort(rng.uniform(0, 10, size=90)),
            np.sort(rng.uniform(0, 10, size=110)),
        ],
        "mPFC": [
            np.sort(rng.uniform(0, 10, size=80)),
            np.sort(rng.uniform(0, 10, size=60)),
            np.sort(rng.uniform(0, 10, size=70)),
        ],
    }
