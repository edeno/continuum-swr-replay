"""The ContFrag stop table carries every column the per-stop rate ladder reads.

R6 ran the event-source comparison on the ContFrag SWR and MUA stop tables.
Its ``rate_hierarchy`` block fit 0 of 242 sessions and the run still exited 0:
the tables lacked the reward-history basis that ``MODEL_PER_STOP_RATE``'s R1
reads, and each session was only recorded with status ``missing_columns``.

These tests pin the builder half of the fix. The ContFrag stop table carries
the ladder's predictors, computed by the same helper as the NonLocal stop
table, also when the session has no immobility bout. The comparison's hard
error for a missing column is tested in ``test_per_stop_rate_model.py``.

The session reuses the dot frame and graph of ``test_artifact_exposure.py``,
which builds both stop tables from one input. Four trips with mixed outcomes
replace its one trip, so the reward history differs from stop to stop. Each
outcome becomes known 0.5 s into the stop its arrival starts, so a builder that
read the history at the bout end, not the bout start, gets other values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.behavior.covariates import (
    reward_history_basis_features,
)
from continuum_swr_replay.analysis.behavior.design_constants import (
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_INCOMPLETE,
)
from continuum_swr_replay.analysis.contfrag.tables import (
    _CONTFRAG_STOP_TABLE_COLUMNS,
    build_contfrag_stop_table,
)
from continuum_swr_replay.analysis.event_tables.stop_table import build_stop_table
from continuum_swr_replay.analysis.replay_models.model_specs import MODEL_PER_STOP_RATE
from continuum_swr_replay.analysis.replay_models.rate.per_stop_rate_model import (
    fit_event_source_rate_hierarchy,
)
from tests.analysis.event_definition.test_artifact_exposure import TIME, _graph
from tests.analysis.event_definition.test_artifact_exposure import _dot as _exposure_dot
from tests.support._port_experience_fixtures import port_experiences_from_trials

REWARD_HISTORY = [*REWARD_HISTORY_COLUMNS, REWARD_HISTORY_INCOMPLETE]
ARRIVALS = np.array([3.0, 7.0, 11.0, 15.0])
"""Each trip leaves its well 1 s before it arrives and the next trip starts
3 s after it arrives."""


def _trials() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movement_start_time": ARRIVALS - 1.0,
            "destination_first_poke_in_time": ARRIVALS,
            "end_time": ARRIVALS + 3.0,
            "from_well": [1, 0, 1, 0],
            "to_well": [0, 1, 0, 1],
            "is_reward": [True, False, True, True],
            "is_patch_change": [False] * 4,
            "outcome_known_time": ARRIVALS + 0.5,
        }
    )


def _dot(trials: pd.DataFrame) -> pd.DataFrame:
    """The exposure fixture's frame, moving only between leaving and arriving.

    Stops: [0, 2], [3, 6], [7, 10], [11, 14], [15, 19.998].
    """
    moving = np.zeros(TIME.size, dtype=bool)
    for start, arrival in zip(
        trials["movement_start_time"],
        trials["destination_first_poke_in_time"],
        strict=True,
    ):
        moving |= (TIME > start) & (TIME < arrival)
    dot = _exposure_dot()
    dot["head_speed"] = np.where(moving, 10.0, 0.0)
    return dot


def _stop_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(ContFrag stop table, NonLocal stop table, port experiences)."""
    trials = _trials()
    experiences = port_experiences_from_trials(trials, prepend_initial=True)
    dot = _dot(trials)
    contfrag = build_contfrag_stop_table(dot, trials, experiences, _graph())
    nonlocal_ = build_stop_table(dot, trials, experiences, _graph())
    return contfrag, nonlocal_, experiences


def test_the_contfrag_stop_table_carries_every_per_stop_rate_predictor():
    contfrag, _, _ = _stop_tables()
    predictors = {
        predictor
        for spec in MODEL_PER_STOP_RATE.values()
        for predictor in spec.predictors
    }

    assert predictors, "the per-stop rate ladder reads no predictor"
    assert sorted(predictors - set(contfrag.columns)) == []


def test_the_comparison_rate_block_accepts_the_contfrag_stop_table():
    """The block's own required columns, not a copy of them, are checked."""
    contfrag, _, _ = _stop_tables()

    # The check is that this call does not raise. The fixture's one session has
    # fewer stops than the block's minimum, so it is not fit.
    _, provenance = fit_event_source_rate_hierarchy(contfrag, "SWR")

    assert len(provenance) == 1
    assert set(provenance["status"]) == {"too_few_units"}


def test_its_reward_history_is_the_nonlocal_stop_tables():
    contfrag, nonlocal_, experiences = _stop_tables()

    assert contfrag["stop_start"].tolist() == [0.0, 3.0, 7.0, 11.0, 15.0]
    pd.testing.assert_series_equal(contfrag["stop_start"], nonlocal_["stop_start"])
    pd.testing.assert_frame_equal(
        contfrag[REWARD_HISTORY], nonlocal_[REWARD_HISTORY], check_exact=True
    )

    # The fixture can tell the two clocks apart: stops with fewer than two
    # known outcomes are flagged, the others carry a graded history, and the
    # history at the bout end is not the history at the bout start.
    assert contfrag[REWARD_HISTORY_INCOMPLETE].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    at_start, _ = reward_history_basis_features(
        contfrag["stop_start"].to_numpy(), experiences
    )
    at_end, _ = reward_history_basis_features(
        contfrag["stop_end"].to_numpy(), experiences
    )
    np.testing.assert_array_equal(
        contfrag[list(REWARD_HISTORY_COLUMNS)].to_numpy(), at_start
    )
    assert not np.array_equal(at_start, at_end)
    assert len(np.unique(at_start[2:], axis=0)) == 3


def test_a_session_with_no_stop_keeps_the_full_column_set():
    trials = _trials()
    experiences = port_experiences_from_trials(trials, prepend_initial=True)
    dot = _exposure_dot()
    dot["head_speed"] = 10.0

    stops = build_contfrag_stop_table(dot, trials, experiences, _graph())

    assert stops.empty
    assert list(stops.columns) == list(_CONTFRAG_STOP_TABLE_COLUMNS)
    assert set(REWARD_HISTORY) <= set(stops.columns)
