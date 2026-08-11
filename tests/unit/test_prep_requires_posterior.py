"""A session with no ``results.nc`` must not be built by default.

Without a conditional posterior ``_process_session`` does not merely lose a few
columns: ``build_event_sequence_table`` is skipped entirely, so the per-timebin
table is absent and with it every retrospective metric (``chosen_arm_fraction``,
``ends_at_next``, ``contains_path_to_next`` and the prev-arm mirrors, which are
merged only when that table exists). Such a session would ship with a different
column-completeness profile from every other one, and the corpus invariant
``sum(n_nonlocal_bins) == len(event_sequence_table)`` would break silently --
nothing downstream checks it.

0 of 218 sessions lack ``results.nc`` today, so the gate is a no-op on the
current corpus. These tests pin the behavior for the day one goes missing.
"""

from __future__ import annotations

import inspect

import pytest

from continuum_swr_replay.analysis.replay import prepare_data


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


def test_missing_posterior_raises_by_default():
    with pytest.raises(ValueError, match="no results.nc"):
        prepare_data._process_session(_session_without_posterior())


def test_the_refusal_names_the_opt_in_flag():
    """An operator hitting this needs to know the escape hatch exists."""
    with pytest.raises(ValueError, match="--allow-missing-posterior"):
        prepare_data._process_session(_session_without_posterior())


def test_gate_runs_before_any_file_is_opened():
    """Fail fast: a ~1M-row dot pickle is not worth reading to then refuse.

    The session dict points at paths that do not exist, so if the gate ran
    after the loads this would surface as FileNotFoundError instead.
    """
    with pytest.raises(ValueError, match="no results.nc"):
        prepare_data._process_session(_session_without_posterior())


def test_opt_in_flag_lets_the_session_through_the_gate():
    """``--allow-missing-posterior`` must bypass the refusal, not the loading.

    With the flag set the gate passes and execution reaches the dot read, which
    fails on the nonexistent path -- proving the gate itself no longer fires.
    """
    with pytest.raises((FileNotFoundError, OSError)):
        prepare_data._process_session(
            _session_without_posterior(), allow_missing_posterior=True
        )


def test_default_is_fail_closed():
    """The opt-in must default to off, or the gate is decorative."""
    sig = inspect.signature(prepare_data._process_session)
    assert sig.parameters["allow_missing_posterior"].default is False


def test_runner_exposes_the_flag():
    """The gate is only usable if main() actually threads the flag through."""
    src = inspect.getsource(prepare_data.main)
    assert "--allow-missing-posterior" in src
    assert "allow_missing_posterior=args.allow_missing_posterior" in src
