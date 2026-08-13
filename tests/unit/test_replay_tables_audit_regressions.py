"""Regression tests from the 2026-08-08 multi-agent audit of the replay tables.

Two groups:

**Coverage tests** (expected to PASS) close gaps found by mutation testing —
the existing suite asserts only per-event *sums* over the arm axis and only the
*dtype* of ``is_swr``, so an arm-axis reversal or a global ``is_swr`` inversion
passed all 203 subsystem tests. These pin the values those mutations changed.

**Failing regressions** (expected to FAIL until the defect is fixed) encode
confirmed covariate-timing defects. Each is marked ``xfail(strict=True)`` so the
suite stays green while the defect stands, and turns red the moment a fix lands
without the marker being removed.

See ``.claude/docs/reviews/`` for the audit report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay._helpers import add_trial_context
from continuum_swr_replay.analysis.replay.covariates import visit_recency_all_events
from continuum_swr_replay.analysis.replay.tables import (
    build_event_arm_table,
    build_event_table,
)

from .test_replay_tables import (
    _bandit_graph,
    _EnvStub,
    _tiny_dot_with_two_events,
    _tiny_posterior,
)


@pytest.fixture
def audit_session():
    """The shared tiny two-event fixture, rebuilt locally.

    Reuses ``test_replay_tables``'s builders rather than importing its
    fixtures, so the two modules stay independent.
    """
    dot = _tiny_dot_with_two_events()
    trials = pd.DataFrame(
        columns=[
            "start_time",
            "end_time",
            "to_well",
            "from_well",
            "from_patch",
            "to_patch",
            "is_reward",
            "is_patch_change",
        ]
    )
    return dot, _EnvStub(), trials, _bandit_graph(), _tiny_posterior(dot)


# ── Coverage: pin what the mutation probes were able to change ──


def test_arm_count_hard_is_pinned_per_arm_not_only_per_event_sum(
    audit_session,
):
    """``arm_count_hard`` must land on the *correct* arm, not merely sum right.

    Every pre-existing arm-axis assertion is a per-event ``groupby.sum()``,
    which is invariant to any permutation of the nine arm columns. Reversing
    the arm axis (arm 0<->8, 1<->7, ...) therefore passed the whole suite while
    attributing every event's content to the wrong arm — which would invert the
    meaning of every fitted content coefficient.

    The fixture puts event 1's ten bins entirely on arm 1 and event 2's
    entirely on arm 3, so the full 9-vector is known exactly.
    """
    dot, env, trials, graph, posterior = audit_session
    et = build_event_table(dot, env, trials, graph, conditional_posterior=posterior)
    eat = build_event_arm_table(
        et, dot, env, trials, graph, conditional_posterior=posterior
    )

    et = et.sort_values("event_start")
    counts = (
        eat.set_index(["event_id", "arm"])["arm_count_hard"]
        .unstack("arm")
        .loc[et["event_id"].to_numpy()]
        .to_numpy()
    )

    expected = np.zeros((2, 9), dtype=float)
    expected[0, 1] = 10.0  # first event: all ten bins on arm 1
    expected[1, 3] = 10.0  # second event: all ten bins on arm 3
    np.testing.assert_array_equal(counts, expected)


def test_arm_fraction_is_pinned_per_arm(audit_session):
    """``arm_fraction`` must peak on the arm the event actually decoded to.

    Companion to the count test: the shipped assertion only checks that the
    nine values sum to 1.0, which any permutation or roll of the arm axis also
    satisfies.
    """
    dot, env, trials, graph, posterior = audit_session
    et = build_event_table(dot, env, trials, graph, conditional_posterior=posterior)
    eat = build_event_arm_table(
        et, dot, env, trials, graph, conditional_posterior=posterior
    )

    et = et.sort_values("event_start")
    frac = (
        eat.set_index(["event_id", "arm"])["arm_fraction"]
        .unstack("arm")
        .loc[et["event_id"].to_numpy()]
    )
    assert frac.to_numpy().argmax(axis=1).tolist() == [1, 3]
    np.testing.assert_allclose(frac.to_numpy()[0, 1], 1.0)
    np.testing.assert_allclose(frac.to_numpy()[1, 3], 1.0)


def test_is_swr_and_is_mua_values_are_pinned_not_only_dtype(
    audit_session,
):
    """The merged ``is_swr`` / ``is_mua`` *values* must be asserted.

    The only pre-existing assertion on these columns checks ``.dtype == bool``,
    so inverting ``is_swr`` on the classified frame before the merge — the
    signature of a wrong merge key or a suffix collision picking up
    ``is_swr_y`` — passed the entire suite. ``is_swr`` is the paper's primary
    contrast, and every ``swr_x_*`` interaction column is built from it.

    The fixture encodes the truth directly: event 1 has ``SWR_event = 1`` and no
    MUA; event 2 has ``MUA_event = 1`` and no SWR.
    """
    dot, env, trials, graph, posterior = audit_session
    et = build_event_table(
        dot, env, trials, graph, conditional_posterior=posterior
    ).sort_values("event_start")

    assert et["is_swr"].tolist() == [True, False]
    assert et["is_mua"].tolist() == [False, True]


def test_swr_interactions_follow_the_is_swr_flag(audit_session):
    """``swr_x_current`` must be ``is_swr * is_current``, arm by arm.

    Pins the interaction block against an ``is_swr`` inversion propagating
    silently into the design matrix.
    """
    dot, env, trials, graph, posterior = audit_session
    et = build_event_table(dot, env, trials, graph, conditional_posterior=posterior)
    eat = build_event_arm_table(
        et, dot, env, trials, graph, conditional_posterior=posterior
    )

    is_swr = eat["event_id"].map(et.set_index("event_id")["is_swr"]).astype(float)
    np.testing.assert_array_equal(
        eat["swr_x_current"].to_numpy(),
        (is_swr * eat["is_current"]).to_numpy(),
    )


# ── Failing regressions: confirmed covariate-timing defects ──


def test_visit_recency_does_not_depend_on_future_trials():
    """A covariate at time t must not change when later trials are appended.

    This is the defining property of a backward-looking covariate. The
    never-visited fill must use the number of trials completed at the event,
    not the eventual session length.
    """

    def _trials(n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "start_time": np.arange(n, dtype=float),
                "end_time": np.arange(1, n + 1, dtype=float),
                "goal_arrival_time": np.arange(1, n + 1, dtype=float),
                "to_well": [0, 1, 0, 1, 0][:n],
                "from_well": [1, 0, 1, 0, 1][:n],
                "is_reward": [1, 0, 1, 0, 1][:n],
                "is_patch_change": [0, 0, 0, 0, 0][:n],
            }
        )

    well_to_arm = {w: w for w in range(6)}
    event_times = np.array([2.5])  # after trial 2 completes, before trial 3

    short = visit_recency_all_events(event_times, _trials(3), well_to_arm)
    long = visit_recency_all_events(event_times, _trials(5), well_to_arm)

    # Arm 2 is never visited in either version, and nothing that happens
    # after t=2.5 can be known at t=2.5.
    np.testing.assert_array_equal(short[:, 2], long[:, 2])


def test_trial_context_is_not_taken_from_the_in_progress_trip():
    """Context during a run must not describe the run's own outcome.

    A trial spans ``[start_time, goal_arrival_time)`` — departure to arrival.
    At a timepoint inside that interval the animal is still running and the
    destination has not been reached, so neither the patch-change status nor
    the reward outcome of that trip is determined yet. With no earlier
    completed trial, the correct value is NaN.
    """
    trials = pd.DataFrame(
        {
            "start_time": [0.0],
            "end_time": [10.0],
            "goal_arrival_time": [10.0],
            "to_well": [2.0],  # patch 2
            "from_well": [0.0],  # patch 1 -> this trip IS a patch change
            "is_reward": [1.0],
            "is_patch_change": [1.0],
        }
    )
    # t = 5.0 is mid-run: departed at 0.0, will not arrive until 10.0.
    dot = pd.DataFrame({"x": [0.0]}, index=pd.Index([5.0], name="time"))

    out = add_trial_context(dot, trials)

    assert np.isnan(out["is_after_switch"].iloc[0])
    assert np.isnan(out["reward_outcome"].iloc[0])


@pytest.mark.xfail(
    strict=True,
    reason=(
        "AUDIT 2026-08-08: arm_uncertainty is standardized over all nine arms, "
        "but trunk arms 6/7/8 carry a constant 0.0 placeholder (they are not "
        "reward sites). That inflates the pooled SD by 1.62x, so the fitted "
        "coefficient is not 'per 1 SD' on the leaf support that identifies it, "
        "contradicting model_specs.py's 'well-scaled (SD 1)' claim."
    ),
)
def test_arm_uncertainty_z_is_unit_sd_on_the_support_that_identifies_it():
    """``<col>_z`` must have unit SD on the rows that carry information.

    Every ladder fitting ``arm_uncertainty_z`` also carries arm fixed effects,
    which absorb the constant trunk placeholder entirely — so the trunk rows
    contribute zero information to the slope while still setting its reporting
    unit. The coefficient is therefore quoted per 1.62 leaf-SDs, not per 1 SD.
    """
    tables = Path("Processed-Data/replay_content/event_arm_table.pkl")
    if not tables.exists():
        pytest.skip("materialized event_arm_table.pkl not available")

    eat = pd.read_pickle(tables)
    leaf = eat.loc[eat["arm"] < 6, "arm_uncertainty_z"].to_numpy()
    np.testing.assert_allclose(np.nanstd(leaf), 1.0, rtol=0.05)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "AUDIT 2026-08-08: per_bin_rate_model drops NaNs per rung, splitting "
        "the ladder into FOUR cohorts (7387816 / 7348182 / 6849272 / 6466795 "
        "bins), so cross-rung AIC is not comparable. A common-cohort refit "
        "preserves the M5-vs-M7r ranking but shrinks the margin ~10x."
    ),
)
def test_rate_ladder_rungs_share_a_cohort():
    """Rungs compared by AIC must be fit on the same rows.

    ``per_bin_rate_model`` drops NaNs against each rung's *own* predictor list,
    so reward-bearing rungs (which need ``reward_lag10``) lose the ~7% of stops
    occurring before a session's 10th trial. AIC scales with n, so the smaller
    cohort gains mechanically. ``freeze_common_cohort`` exists for exactly this
    and is called by ``scripts/content/`` but by nothing in ``scripts/rate/``.

    Guards the ladder definition rather than the fit: if a ladder mixes rungs
    with and without reward lags, the runner must freeze a common cohort.
    """
    fits = Path("Processed-Data/replay_content/models_per_bin_rate/all_models.pkl")
    if not fits.exists():
        pytest.skip("materialized per-bin rate fits not available")

    df = pd.read_pickle(fits)
    df = df[df["outcome"] == "non_local"] if "outcome" in df.columns else df
    n_by_model = df.groupby("model")["n_bins"].sum()
    assert n_by_model.nunique() == 1, (
        "per-bin rate rungs are fit on different numbers of bins, so their "
        f"AIC/BIC/deviance are not comparable:\n{n_by_model.sort_values()}"
    )
