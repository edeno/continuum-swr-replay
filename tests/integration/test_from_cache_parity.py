"""Phase-4 fast path: ``--from-cache`` rebuilds are byte-identical to a Spyglass run.

The unit tests in ``test_prepare_data_cache.py`` pin the cache *seam*
(:func:`load_behavioral_context`) with fake inputs. This slow test closes the
end-to-end gap the reviewer flagged: it runs the *whole* :func:`_process_session`
(all four table builders) twice for one real session — once populating the cache from
Spyglass (``from_cache=False``), once served entirely from the cache
(``from_cache=True``) — and asserts every output frame is identical. A future change to
``_process_session``'s control flow (e.g. reordering the cache write relative to a
builder) that broke cache/no-cache parity would fail here.

Skips when no materialized session inputs are on disk or Spyglass is unreachable (the
cold pass needs ``load_data``), so it is a no-op on hosts without data access.
"""

from __future__ import annotations

import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay import prepare_data as pdm
from continuum_swr_replay.analysis.replay.covariates import load_contingency

pytestmark = pytest.mark.slow

_FRAMES = ("event_table", "event_arm_table", "stop_table", "event_sequence_table")


def test_process_session_cache_parity(tmp_path, monkeypatch):
    sessions = pdm.find_sessions(max_per_animal=1)
    if not sessions:
        pytest.skip("no materialized session inputs on disk")
    session = sessions[0]

    # Isolate the cache so the test never touches the canonical _session_cache.
    monkeypatch.setattr(pdm, "SESSION_CACHE_DIR", tmp_path)
    try:
        cont = load_contingency(
            pdm.OUTPUT_DIR / "models_per_bin_rate" / "qvals_table.pkl"
        )
    except ValueError as exc:
        if "legacy or mismatched covariate schema" not in str(exc):
            raise
        pytest.skip(f"materialized qvals requires canonical regeneration: {exc}")

    try:
        cold = pdm._process_session(session, contingency=cont, from_cache=False)
    except Exception as e:  # noqa: BLE001 - Spyglass down / data gap -> skip, not fail
        pytest.skip(f"cold Spyglass load unavailable: {e}")
    if cold is None:
        pytest.skip("session could not be loaded (missing trials/track_graph)")

    # Cache must now exist; warm pass reads it (no Spyglass).
    assert pdm.behavioral_context_cache_path(session).exists()
    warm = pdm._process_session(session, contingency=cont, from_cache=True)
    assert warm is not None

    for key in _FRAMES:
        a, b = cold[key], warm[key]
        if a is None and b is None:
            continue
        pd.testing.assert_frame_equal(a, b, check_exact=True, obj=key)
