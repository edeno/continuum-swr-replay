"""The PFC path's event counts equal the shipped event table's, on a real session.

The offline tests pin the loaders and the split against synthetic frames. This
one closes the end-to-end gap: for a session that has decode results on disk,
the event sets ``build_session_event_sets`` hands the design-matrix builder must
be exactly the shipped event table's rows for that session, restricted to the
well-anchored stop cohort and split by that table's own ``is_swr`` column. A
regression that re-detected events, re-split them, or silently dropped some at
the frame boundary would show up here as a count mismatch.

``build_session_event_sets`` is called rather than ``run_session`` because the
counts under test are the ones it computes and ``run_session`` only forwards;
calling it directly keeps this test to a data load rather than a full GLM fit
over every neuron in the session.

Needs materialized session inputs on disk and a reachable Spyglass, so it is
opt-in (``-m real_data``) and fails when either is missing.
"""

from __future__ import annotations

import numpy as np
import pytest

from continuum_swr_replay.analysis.events.cohort_selection import events_in_stops
from continuum_swr_replay.analysis.pfc.canonical_events import (
    load_session_canonical_events,
    load_session_swr_intervals,
)
from continuum_swr_replay.analysis.pfc.cohort import load_session_stop_cohort
from continuum_swr_replay.analysis.pfc.glm import build_session_event_sets
from continuum_swr_replay.analysis.replay import prepare_data as pdm

pytestmark = [pytest.mark.slow, pytest.mark.integration, pytest.mark.real_data]


def test_the_event_sets_are_the_shipped_tables_rows_for_that_session():
    from continuum_swr_replay.analysis.hippocampus import load_results

    sessions = [s for s in pdm.find_sessions(max_per_animal=1) if s["results_file"]]
    assert sessions, "no session with decode results on disk"
    session = sessions[0]
    nwb_file_name, epoch_name = session["nwb_file_name"], session["epoch_name"]

    canonical_events, _ = load_session_canonical_events(nwb_file_name, epoch_name)
    swr_intervals, _ = load_session_swr_intervals(nwb_file_name, epoch_name)
    cohort_stops, _ = load_session_stop_cohort(nwb_file_name, epoch_name)

    # importing the loader already needs Spyglass
    from continuum_swr_replay.data_loaders import load_data

    results, _, _ = load_results(
        nwb_file_name, epoch_name, decoding_brain_area="HPC", load_detector=False
    )
    data = load_data(nwb_file_name, epoch_name)

    event_times, counts = build_session_event_sets(
        data,
        results,
        canonical_events=canonical_events,
        swr_intervals=swr_intervals,
        cohort_stops=cohort_stops,
        stop_cohort="at_well",
    )

    # The expectation is computed from the shipped table alone, by the cohort
    # rule the hippocampal families use -- not from anything the PFC path did.
    kept = canonical_events.loc[
        events_in_stops(canonical_events.start_time.to_numpy(dtype=float), cohort_stops)
    ]
    expected_swr = int(kept.is_swr.to_numpy(dtype=bool).sum())

    assert counts["n_events_outside_frame"] == 0, (
        "a canonical event fell outside the aligned position frame; the decode "
        "and position axes disagree over more than an empty tail"
    )
    assert counts["n_nonlocal_before"] == len(canonical_events)
    assert counts["n_nonlocal_after"] == len(kept)
    assert counts["n_nonlocal_swr"] == expected_swr
    assert counts["n_nonlocal_no_swr"] == len(kept) - expected_swr
    assert len(event_times["non_local"]) == len(kept)
    np.testing.assert_array_equal(
        event_times["non_local"].event_id.to_numpy(), kept.event_id.to_numpy()
    )
    np.testing.assert_array_equal(
        event_times["non_local_SWR"].event_id.to_numpy(),
        kept.loc[kept.is_swr.to_numpy(dtype=bool)].event_id.to_numpy(),
    )
