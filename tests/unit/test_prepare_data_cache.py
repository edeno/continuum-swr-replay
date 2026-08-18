"""Phase-4 fast path: ``prepare_data --from-cache`` skips Spyglass via a session cache.

The only per-session Spyglass fetch (``trials`` + ``track_graph``) is pickled per
session on every full run and read back with ``--from-cache`` so a covariate-only
rebuild avoids ``load_data``. These tests pin the seam
(:func:`load_trials_track_graph`) hermetically — no Spyglass, no materialized tables —
by faking ``load_data`` and pointing the cache at ``tmp_path``.
"""

from __future__ import annotations

import pickle

import networkx as nx
import pandas as pd
import pytest

import continuum_swr_replay.data_loaders as dl
from continuum_swr_replay.analysis.replay import prepare_data as pdm


@pytest.fixture
def fake_session():
    return {
        "nwb_file_name": "fake20200101_.nwb",
        "epoch_name": "02_r1",
        "session_id": "fake20200101_02_r1_HPC",
    }


@pytest.fixture
def fake_payload():
    # The outcome-clock columns are what dates a cache entry: an entry without
    # them predates the clock and is rejected, so a payload standing in for a
    # current entry has to carry them.
    trials = pd.DataFrame(
        {
            "to_well": [0, 1, 2],
            "is_reward": [1, 0, 1],
            "reward_on_time": [10.0, float("nan"), 30.0],
            "outcome_known_time": [10.0, 20.3, 30.0],
        }
    )
    g = nx.Graph()
    g.add_edge(0, 6)
    return trials, g


@pytest.fixture
def patched(tmp_path, monkeypatch, fake_payload):
    """Redirect the cache to tmp_path and count/serve load_data calls."""
    monkeypatch.setattr(pdm, "SESSION_CACHE_DIR", tmp_path)
    calls = {"n": 0}
    trials, g = fake_payload

    def _fake_load_data(nwb, epoch, include=None):
        calls["n"] += 1
        return {"trials": trials, "track_graph": g}

    # `load_trials_track_graph` imports `load_data` lazily from
    # `data_loaders` (kept out of module scope so importing prepare_data
    # doesn't open DataJoint). Patch the name in the `data_loaders` __dict__
    # via setitem so we don't trigger its PEP-562 __getattr__ (which would
    # import spyglass) while setting up the fake.
    monkeypatch.setitem(dl.__dict__, "load_data", _fake_load_data)
    return calls, tmp_path


def test_cold_calls_spyglass_and_writes_cache(patched, fake_session, fake_payload):
    calls, cache_dir = patched
    trials, g, hit = pdm.load_trials_track_graph(fake_session, from_cache=False)
    assert calls["n"] == 1  # Spyglass hit
    assert hit is False
    pd.testing.assert_frame_equal(trials, fake_payload[0])
    assert (cache_dir / f"{fake_session['session_id']}.pkl").exists()


def test_warm_reads_cache_without_spyglass(patched, fake_session, fake_payload):
    calls, _ = patched
    # Prime the cache (one Spyglass call), then read it back.
    pdm.load_trials_track_graph(fake_session, from_cache=False)
    assert calls["n"] == 1
    trials, g, hit = pdm.load_trials_track_graph(fake_session, from_cache=True)
    assert calls["n"] == 1  # NO additional Spyglass call — served from cache
    assert hit is True
    pd.testing.assert_frame_equal(trials, fake_payload[0])
    assert set(g.edges) == set(fake_payload[1].edges)  # track_graph round-trips


def test_from_cache_falls_back_when_missing(patched, fake_session):
    """--from-cache on an uncached session still loads (and caches) it — never dropped."""
    calls, cache_dir = patched
    _, _, hit = pdm.load_trials_track_graph(fake_session, from_cache=True)
    assert calls["n"] == 1  # fell back to Spyglass
    assert hit is False
    assert (cache_dir / f"{fake_session['session_id']}.pkl").exists()  # now cached


def test_cache_entry_without_the_outcome_clock_is_not_trusted(
    patched, fake_session, fake_payload
):
    """A pre-outcome-clock entry must be re-fetched, not silently reused.

    Such an entry resolves every reward-history covariate at poke-out rather
    than pump onset, and nothing downstream reveals it -- the run simply
    reports one more session than a Spyglass pass would. This is exactly how
    three ContFrag sessions ended up on the retired clock.
    """
    calls, cache_dir = patched
    trials, graph = fake_payload
    stale = trials.drop(columns=["reward_on_time", "outcome_known_time"])
    with open(cache_dir / f"{fake_session['session_id']}.pkl", "wb") as f:
        pickle.dump({"trials": stale, "track_graph": graph}, f)

    got_trials, _, hit_cache = pdm.load_trials_track_graph(
        fake_session, from_cache=True
    )

    assert not hit_cache, "a pre-outcome-clock entry was served as a cache hit"
    assert calls["n"] == 1, "Spyglass was not consulted for the rejected entry"
    assert "outcome_known_time" in got_trials.columns


def test_cache_entry_with_the_outcome_clock_is_trusted(
    patched, fake_session, fake_payload
):
    """The rejection must key on the outcome clock, not reject every entry."""
    calls, cache_dir = patched
    trials, graph = fake_payload
    assert "outcome_known_time" in trials.columns  # the condition under test
    with open(cache_dir / f"{fake_session['session_id']}.pkl", "wb") as f:
        pickle.dump({"trials": trials, "track_graph": graph}, f)

    _, _, hit_cache = pdm.load_trials_track_graph(fake_session, from_cache=True)

    assert hit_cache
    assert calls["n"] == 0, "Spyglass was consulted for a current cache entry"
