"""A session with no ``results.nc`` must not be built by default.

Without a conditional posterior ``_process_session`` does not merely lose a few
columns: ``build_event_sequence_table`` is skipped entirely, so the per-timebin
table is absent and with it every retrospective metric (``chosen_arm_fraction``,
``ends_at_next``, ``contains_path_to_next`` and the prev-arm mirrors, which are
merged only when that table exists). Such a session would ship with a different
column-completeness profile from every other one, and the corpus invariant
``sum(n_nonlocal_bins) == len(event_sequence_table)`` would break silently --
nothing downstream checks it. The refusal is unconditional: there is no opt-in.

0 of 218 sessions lack ``results.nc`` today, so the gate is a no-op on the
current corpus. These tests pin the behavior for the day one goes missing.
"""

from __future__ import annotations

import pytest

from continuum_swr_replay.analysis.behavior.tables import prepare_data


def _session_without_posterior() -> dict:
    """Minimal session dict; the gate fires before any file is opened."""
    return {
        "nwb_file_name": "wilbur20210329_.nwb",
        "epoch_name": "16_r8",
        "session_id": "wilbur20210329_16_r8_HPC",
        "dot_file": "/nonexistent/dot.pkl",
        "env_file": "/nonexistent/env.pkl",
        "results_file": None,
    }


def test_missing_posterior_is_refused_before_anything_is_loaded():
    """The gate fires, and it fires first.

    Fail fast: a ~1M-row dot pickle is not worth reading for a session about to
    be refused. The session dict points at paths that do not exist, so a gate
    placed after the loads would surface FileNotFoundError instead of the
    ValueError asserted here.
    """
    with pytest.raises(ValueError, match="no results.nc"):
        prepare_data._process_session(_session_without_posterior())
