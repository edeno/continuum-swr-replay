"""Integration: the content contrast runner emits a reconciling cohort audit.

Exercises the real ``_run`` → ``_fit_session`` → ``cohort_record`` → ``write_cohort_audit``
path on a synthetic multi-session arm frame (no re-decode). Asserts every emitted record
satisfies the reconciliation invariant ``n_in − Σdrops == n_modeled`` and that the
qvals-tail NaN lands in ``n_drop_qvals_gap`` (mechanism B), separate from own-predictor
missingness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay import contrast_aggregation
from continuum_swr_replay.analysis.replay.cohort_audit import (
    DROP_FIELDS,
    write_cohort_audit,
)
from scripts.content import fit_swr_contrast_content as mod

pytestmark = [pytest.mark.slow, pytest.mark.integration]

N_ARMS = 9


def _synthetic_arm_events(n_sessions=3, n_events=12, qvals_gap_events=2, seed=0):
    """Long arm frame: one chosen arm per event, is_current marks an arm, is_swr split,
    and the first ``qvals_gap_events`` events per session have NaN q_arm (the tail)."""
    rng = np.random.default_rng(seed)
    rows = []
    eid = 0
    for s in range(n_sessions):
        animal = ["a", "b", "c"][s % 3]
        for e in range(n_events):
            chosen = int(rng.integers(0, N_ARMS))
            current = int(rng.integers(0, N_ARMS))
            is_swr = bool(e % 2)
            qhmm_nan = e < qvals_gap_events
            for arm in range(N_ARMS):
                rows.append(
                    {
                        "animal": animal,
                        "session_date": s,
                        "epoch": "e",
                        "event_id": eid,
                        "candidate_arm": arm,
                        "arm_count_hard": float(arm == chosen),
                        "is_current_arm": float(arm == current),
                        "reward_belief_mean_candidate_arm_preoutcome": np.nan
                        if qhmm_nan
                        else float(rng.normal()),
                        "has_qvals_row": not qhmm_nan,
                        "is_swr": is_swr,
                        "time_since_stop_start_s": float(rng.uniform(0, 3)),
                    }
                )
            eid += 1
    return pd.DataFrame(rows)


def test_run_emits_reconciling_cohort_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(contrast_aggregation, "MIN_EVENTS", 3)
    ev = _synthetic_arm_events()
    ladder = {
        "C0_base": (["is_current_arm"], "split"),  # no value covariate
        "C4_value": (
            ["is_current_arm", "reward_belief_mean_candidate_arm_preoutcome"],
            "split",
        ),  # value rung -> qvals gap
    }
    results, audits = mod._run(
        ev, ladder, response_col="arm_count_hard", n_categories=N_ARMS
    )
    assert audits, "no cohort records emitted"

    # Every record reconciles (cohort_record already asserts, but re-check explicitly).
    for rec in audits:
        drops = sum(rec[f] for f in DROP_FIELDS)
        assert rec["n_in"] - drops == rec["n_modeled"], rec

    # The value rung's qvals-tail NaN must be attributed to the qvals gap, and the
    # base rung (no value covariate) must NOT have qvals-gap drops.
    by_key = {(r["model"], r["population"]): r for r in audits}
    c4_all = by_key[("C4_value", "all")]
    assert c4_all["n_drop_qvals_gap"] > 0
    assert by_key[("C0_base", "all")]["n_drop_qvals_gap"] == 0

    # The CSV writes and round-trips with the reconciliation intact.
    out = tmp_path / "run_cohort_audit.csv"
    write_cohort_audit(audits, out)
    reread = pd.read_csv(out)
    drops = reread[list(DROP_FIELDS)].sum(axis=1)
    assert ((reread["n_in"] - drops) == reread["n_modeled"]).all()
