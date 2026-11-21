"""Phase-4 fast path: ``prepare_data --from-cache`` skips Spyglass via a session cache.

The only per-session Spyglass fetch (movement trials, port experiences, and
``track_graph``) is pickled per
session on every full run and read back with ``--from-cache`` so a covariate-only
rebuild avoids ``load_data``. These tests pin the seam
(:func:`load_behavioral_context`) hermetically — no Spyglass or materialized tables —
by faking ``load_data`` and pointing the cache at ``tmp_path``.
"""

from __future__ import annotations

import pickle

import networkx as nx
import pandas as pd
import pytest

import continuum_swr_replay.data_loaders as dl
from continuum_swr_replay.analysis.event_tables import prepare_data as pdm
from scripts.tables import prepare_replay_content_data as prepare_cli


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
            "movement_start_time": [5.0, 15.0, 25.0],
            "destination_first_poke_in_time": [9.8, 20.0, 29.8],
            "destination_first_poke_out_time": [10.3, 20.3, 30.3],
        }
    )
    experiences = pd.DataFrame(
        {
            "experience_id": [0],
            "movement_trial_number": pd.array([pd.NA], dtype="Int64"),
            "is_initial_experience": [True],
            "is_boundary_derived": [False],
            "outcome_observed": [True],
            "port": [0],
            "patch": [1],
            "arrival_time": [0.0],
            "departure_time": [2.0],
            "first_poke_in_time": [0.0],
            "first_poke_out_time": [2.0],
            "final_poke_out_time": [2.0],
            "n_physical_poke_bouts": [1],
            "reward_on_time": [float("nan")],
            "outcome_known_time": [1.0],
            "is_reward": [False],
        }
    )
    experiences.attrs.update(
        {
            "gap_recovered_pump_count": 0,
            "unmatched_pump_event_count": 0,
            "unmatched_pump_events": (),
            "unmatched_pump_event_times": (),
        }
    )
    g = nx.Graph()
    g.add_edge(0, 6)
    return trials, experiences, g


@pytest.fixture
def patched(tmp_path, monkeypatch, fake_payload):
    """Redirect the cache to tmp_path and count/serve load_data calls."""
    monkeypatch.setattr(pdm, "SESSION_CACHE_DIR", tmp_path)
    calls = {"n": 0}
    trials, experiences, g = fake_payload

    def _fake_load_data(nwb, epoch, include=None):
        calls["n"] += 1
        return {
            "trials": trials,
            "port_experiences": experiences,
            "track_graph": g,
        }

    # `load_behavioral_context` imports `load_data` lazily from
    # `data_loaders` (kept out of module scope so importing prepare_data
    # doesn't open DataJoint). Patch the name in the `data_loaders` __dict__
    # via setitem so we don't trigger its PEP-562 __getattr__ (which would
    # import spyglass) while setting up the fake.
    monkeypatch.setitem(dl.__dict__, "load_data", _fake_load_data)
    return calls, tmp_path


def test_cold_calls_spyglass_and_writes_cache(patched, fake_session, fake_payload):
    calls, cache_dir = patched
    trials, experiences, g, hit = pdm.load_behavioral_context(
        fake_session, from_cache=False
    )
    assert calls["n"] == 1  # Spyglass hit
    assert hit is False
    pd.testing.assert_frame_equal(trials, fake_payload[0])
    pd.testing.assert_frame_equal(experiences, fake_payload[1])
    assert pdm.behavioral_context_cache_path(fake_session).exists()


def test_warm_reads_cache_without_spyglass(patched, fake_session, fake_payload):
    calls, _ = patched
    # Prime the cache (one Spyglass call), then read it back.
    pdm.load_behavioral_context(fake_session, from_cache=False)
    assert calls["n"] == 1
    trials, experiences, g, hit = pdm.load_behavioral_context(
        fake_session, from_cache=True
    )
    assert calls["n"] == 1  # NO additional Spyglass call — served from cache
    assert hit is True
    pd.testing.assert_frame_equal(trials, fake_payload[0])
    pd.testing.assert_frame_equal(experiences, fake_payload[1])
    assert set(g.edges) == set(fake_payload[2].edges)  # track_graph round-trips


def test_from_cache_falls_back_when_missing(patched, fake_session):
    """--from-cache on an uncached session still loads (and caches) it — never dropped."""
    calls, cache_dir = patched
    _, _, _, hit = pdm.load_behavioral_context(fake_session, from_cache=True)
    assert calls["n"] == 1  # fell back to Spyglass
    assert hit is False
    assert pdm.behavioral_context_cache_path(fake_session).exists()  # now cached


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
    trials, _, graph = fake_payload
    stale = trials.drop(columns=["reward_on_time", "outcome_known_time"])
    cache_path = pdm.behavioral_context_cache_path(fake_session)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "cache_schema_version": pdm.BEHAVIORAL_CONTEXT_CACHE_VERSION,
                "trials": stale,
                "track_graph": graph,
            },
            f,
        )

    got_trials, _, _, hit_cache = pdm.load_behavioral_context(
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
    trials, experiences, graph = fake_payload
    assert "outcome_known_time" in trials.columns  # the condition under test
    cache_path = pdm.behavioral_context_cache_path(fake_session)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "cache_schema_version": pdm.BEHAVIORAL_CONTEXT_CACHE_VERSION,
                "trials": trials,
                "port_experiences": experiences,
                "track_graph": graph,
            },
            f,
        )

    _, _, _, hit_cache = pdm.load_behavioral_context(fake_session, from_cache=True)

    assert hit_cache
    assert calls["n"] == 0, "Spyglass was consulted for a current cache entry"


def test_cache_entry_without_port_experiences_is_not_trusted(
    patched, fake_session, fake_payload
):
    """Destination-only history caches cannot masquerade as current inputs."""
    calls, cache_dir = patched
    trials, _, graph = fake_payload
    cache_path = pdm.behavioral_context_cache_path(fake_session)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "cache_schema_version": pdm.BEHAVIORAL_CONTEXT_CACHE_VERSION,
                "trials": trials,
                "track_graph": graph,
            },
            f,
        )

    got_trials, got_experiences, _, hit_cache = pdm.load_behavioral_context(
        fake_session, from_cache=True
    )

    assert not hit_cache
    assert calls["n"] == 1
    assert got_trials is not None
    assert got_experiences is not None
    assert got_experiences["is_initial_experience"].tolist()[0] is True


def test_legacy_cache_is_preserved_and_not_reused(patched, fake_session, fake_payload):
    """A prior cache remains available for comparison but cannot drive v4 tables."""
    calls, cache_dir = patched
    trials, experiences, graph = fake_payload
    legacy_path = cache_dir / f"{fake_session['session_id']}.pkl"
    with open(legacy_path, "wb") as f:
        pickle.dump(
            {
                "trials": trials,
                "port_experiences": experiences.drop(
                    columns=["is_boundary_derived", "outcome_observed"]
                ),
                "track_graph": graph,
            },
            f,
        )

    _, _, _, hit_cache = pdm.load_behavioral_context(fake_session, from_cache=True)

    assert not hit_cache
    assert calls["n"] == 1
    assert legacy_path.exists(), "the old comparison cache was overwritten"
    assert pdm.behavioral_context_cache_path(fake_session).exists()


def test_previous_versioned_cache_is_not_overwritten(
    patched, fake_session, fake_payload
):
    calls, cache_dir = patched
    trials, experiences, graph = fake_payload
    previous_path = (
        cache_dir / "physical_dio_poke_trials_v2" / f"{fake_session['session_id']}.pkl"
    )
    previous_path.parent.mkdir(parents=True)
    with open(previous_path, "wb") as f:
        pickle.dump(
            {
                "cache_schema_version": "physical_dio_poke_trials_v2",
                "trials": trials,
                "port_experiences": experiences,
                "track_graph": graph,
            },
            f,
        )

    pdm.load_behavioral_context(fake_session, from_cache=True)

    assert calls["n"] == 1
    assert previous_path.exists()
    assert pdm.behavioral_context_cache_path(fake_session).exists()


def test_v3_cache_is_preserved_when_v4_is_written(patched, fake_session, fake_payload):
    """The clock correction must create a sibling cache, not rewrite v3."""
    calls, cache_dir = patched
    trials, experiences, graph = fake_payload
    previous_path = (
        cache_dir / "comrie_photogate_trials_v3" / f"{fake_session['session_id']}.pkl"
    )
    previous_path.parent.mkdir(parents=True)
    with open(previous_path, "wb") as f:
        pickle.dump(
            {
                "cache_schema_version": "comrie_photogate_trials_v3",
                "trials": trials,
                "port_experiences": experiences,
                "track_graph": graph,
            },
            f,
        )

    pdm.load_behavioral_context(fake_session, from_cache=True)

    assert pdm.BEHAVIORAL_CONTEXT_CACHE_VERSION == "comrie_photogate_trials_v4"
    assert calls["n"] == 1
    assert previous_path.exists()
    assert pdm.behavioral_context_cache_path(fake_session).exists()
    assert pdm.behavioral_context_cache_path(fake_session) != previous_path


@pytest.mark.parametrize(
    "invalid_attrs",
    [
        {},
        {
            "gap_recovered_pump_count": 0,
            "unmatched_pump_event_count": 1,
            "unmatched_pump_events": (),
            "unmatched_pump_event_times": (),
        },
    ],
)
def test_cache_entry_without_valid_pump_audit_is_not_trusted(
    patched, fake_session, fake_payload, invalid_attrs
):
    calls, _ = patched
    trials, experiences, graph = fake_payload
    stale_experiences = experiences.copy()
    stale_experiences.attrs = invalid_attrs
    cache_path = pdm.behavioral_context_cache_path(fake_session)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "cache_schema_version": pdm.BEHAVIORAL_CONTEXT_CACHE_VERSION,
                "trials": trials,
                "port_experiences": stale_experiences,
                "track_graph": graph,
            },
            f,
        )

    _, got_experiences, _, hit_cache = pdm.load_behavioral_context(
        fake_session, from_cache=True
    )

    assert not hit_cache
    assert calls["n"] == 1
    assert got_experiences.attrs["unmatched_pump_event_count"] == 0


def test_canonical_build_aborts_instead_of_publishing_a_partial_session_set(
    monkeypatch, tmp_path
):
    session = {
        "nwb_file_name": "rat20200101_.nwb",
        "epoch_name": "02_r1",
        "session_id": "rat20200101_02_r1_HPC",
    }
    # run_prepare_data points the behavioral cache at the output folder.
    monkeypatch.setattr(pdm, "SESSION_CACHE_DIR", pdm.SESSION_CACHE_DIR)
    monkeypatch.setattr(
        prepare_cli, "find_sessions", lambda max_per_animal=None: [session]
    )
    monkeypatch.setattr(prepare_cli, "load_contingency", lambda path: None)

    def fail_session(*args, **kwargs):
        raise ValueError("physical-poke chain mismatch")

    monkeypatch.setattr(prepare_cli, "build_session_tables", fail_session)

    with pytest.raises(RuntimeError, match="No scientific tables were written"):
        prepare_cli.main(["--output-dir", str(tmp_path / "candidate")])
    assert not (tmp_path / "candidate").exists()


def test_cache_only_population_reuses_valid_entries_and_writes_no_tables(
    patched, fake_session, tmp_path
):
    calls, _ = patched
    pdm.load_behavioral_context(fake_session, from_cache=False)
    assert calls["n"] == 1

    prepare_cli.populate_behavioral_context_cache([fake_session], from_cache=True)

    assert calls["n"] == 1, "a valid versioned cache should not be re-fetched"
    assert not list(tmp_path.glob("*.parquet"))


def test_cache_only_population_fails_if_any_session_is_unavailable(monkeypatch):
    sessions = [
        {
            "nwb_file_name": "rat20200101_.nwb",
            "epoch_name": "02_r1",
            "session_id": "rat20200101_02_r1_HPC",
        },
        {
            "nwb_file_name": "rat20200102_.nwb",
            "epoch_name": "02_r1",
            "session_id": "rat20200102_02_r1_HPC",
        },
    ]

    def one_missing(session, *, from_cache):
        if "20200102" in session["nwb_file_name"]:
            return None, None, None, False
        return (
            pd.DataFrame({"trial": [0]}),
            pd.DataFrame({"visit": [0]}),
            object(),
            False,
        )

    monkeypatch.setattr(prepare_cli, "load_behavioral_context", one_missing)

    with pytest.raises(RuntimeError, match="1 of 2 session"):
        prepare_cli.populate_behavioral_context_cache(sessions, from_cache=True)


def test_well_visit_times_reads_the_visits_from_the_cache(patched, fake_session):
    calls, _ = patched
    pdm.load_behavioral_context(fake_session, from_cache=False)
    arrivals, departures = pdm.well_visit_times(fake_session["session_id"])
    assert calls["n"] == 1  # served from the cache
    assert arrivals.tolist() == [0.0]
    assert departures.tolist() == [2.0]


def test_well_visit_times_refuse_a_missing_entry_and_never_fetch(patched, fake_session):
    calls, cache_dir = patched
    with pytest.raises(FileNotFoundError, match="no session cache entry"):
        pdm.well_visit_times(fake_session["session_id"])
    assert calls["n"] == 0  # Spyglass was not asked
    assert not any(cache_dir.rglob("*.pkl"))  # and nothing was written


def test_well_visit_times_refuse_an_off_contract_entry_and_leave_it(
    patched, fake_session
):
    calls, _ = patched
    pdm.load_behavioral_context(fake_session, from_cache=False)
    path = pdm.behavioral_context_cache_path(fake_session)
    with path.open("rb") as f:
        entry = pickle.load(f)
    entry["cache_schema_version"] = "an older contract"
    with path.open("wb") as f:
        pickle.dump(entry, f)
    written = path.read_bytes()
    with pytest.raises(ValueError, match="off the current behavioral-context"):
        pdm.well_visit_times(fake_session["session_id"])
    assert calls["n"] == 1  # only the priming call
    assert path.read_bytes() == written  # the entry is not rewritten


def test_cached_port_experiences_read_the_entry_and_refuse_a_miss(
    patched, fake_session
):
    calls, cache_dir = patched
    with pytest.raises(FileNotFoundError, match="no session cache entry"):
        pdm.cached_port_experiences(fake_session["session_id"])
    assert calls["n"] == 0 and not any(cache_dir.rglob("*.pkl"))
    _, experiences, _, _ = pdm.load_behavioral_context(fake_session, from_cache=False)
    pd.testing.assert_frame_equal(
        pdm.cached_port_experiences(fake_session["session_id"]), experiences
    )
    assert calls["n"] == 1  # served from the cache
