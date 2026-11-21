"""Fixtures that the simulated decoder tests share.

``test_tetrode_holdout_simulated.py`` and ``test_nonlocal_restriction_simulated.py``
fit the real decoder to the same two simulated recordings
(``tests.support.tetrode_holdout_simulation``). Each fit takes about a minute,
so each is made once per test session rather than once per file. No test
changes the recordings or the fitted detectors: a test that needs a changed
detector copies it first.
"""

import pytest

from tests.support import tetrode_holdout_simulation as sim


@pytest.fixture(scope="session")
def simulated_recording():
    """The recording with every kind of event (seed 0) and its fitted detector."""
    recording = sim.simulate_recording(seed=0)
    return recording, sim.fit_detector(recording)


@pytest.fixture(scope="session")
def chance_recording():
    """The local-only recording (:data:`~tests.support.tetrode_holdout_simulation.CHANCE_KINDS`,
    seed 1) and its fitted detector."""
    recording = sim.simulate_recording(
        kinds=sim.CHANCE_KINDS, boundary_event=False, seed=1
    )
    return recording, sim.fit_detector(recording)
