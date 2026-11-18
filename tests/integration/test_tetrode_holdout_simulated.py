"""The held-out tetrode score end to end, on a clusterless recording with known
content (``tests.support.tetrode_holdout_simulation``).

Remote content (a sweep, a held remote place) must score clearly above local
coding on tetrodes that did not see it; local content (a burst at the animal's
place, quiet rest) must not.
"""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence import tetrode_holdout as th
from tests.support.tetrode_holdout_simulation import fit_detector, simulate_recording

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def scored():
    recording = simulate_recording(seed=0)
    detector = fit_detector(recording)
    events = recording.events
    rest = ~recording.is_running
    for s, e in zip(events.s, events.e, strict=True):
        rest[max(0, s - 250) : e + 250] = False
    length = int(events.e.iloc[0] - events.s.iloc[0] + 1)
    starts = np.flatnonzero(np.convolve(rest, np.ones(length, int), "valid") == length)
    controls = np.random.default_rng(1).choice(starts, 12, replace=False)
    windows = pd.concat(
        [
            events.assign(window_id=[f"event{i}" for i in range(len(events))]),
            pd.DataFrame(
                {
                    "kind": "rest",
                    "s": controls,
                    "e": controls + length - 1,
                    "window_id": [f"rest{i}" for i in range(controls.size)],
                }
            ),
        ],
        ignore_index=True,
    )
    scores, _ = th.holdout_scores(
        detector,
        recording.time,
        recording.position[:, None],
        recording.position,
        recording.spike_times,
        recording.marks,
        recording.is_running,
        np.zeros(recording.time.size, dtype=bool),
        windows,
        log=lambda message: None,
    )
    per_window = scores.groupby("window_id")[["gain", "n_spikes"]].sum()
    return per_window.join(windows.set_index("window_id")["kind"])


def test_remote_content_scores_above_local_coding_on_held_out_tetrodes(scored):
    remote = scored[scored["kind"].isin(["sweep", "stationary"])]
    assert len(remote) == 24
    assert (remote["gain"] / remote["n_spikes"] > 1.0).all()


def test_local_content_does_not_score_above_local_coding(scored):
    local = scored[scored["kind"].isin(["local_burst", "rest"])]
    assert len(local) == 24
    assert (local["gain"] < 0.1).all()
