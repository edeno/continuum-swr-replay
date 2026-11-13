"""The memoryless detector on an epoch with artifact time masked.

The simulated tetrode whose units tile the track, decoded in-sample, gives one
long candidate sequence that is shuffled. A stretch in its middle is marked
missing, as the runner marks the artifact mask's bins on the decode grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.detection.memoryless_sequence import (
    MemorylessSequenceConfig,
    SequenceIdentity,
    ShuffleSeed,
    run_epoch,
)
from tests.support._memoryless_clusterless_epoch import (
    SAMPLING_FREQUENCY,
    build_tiling_tetrode,
)

CONFIG = MemorylessSequenceConfig(sampling_frequency=SAMPLING_FREQUENCY)
IDENTITY = SequenceIdentity(
    animal="sim",
    day=20200101,
    epoch="02_r1",
    brain_area="HPC",
    epoch_id="sim20200101_02_r1_HPC",
    run_id="test",
)
SEED = ShuffleSeed(run=7, session=IDENTITY.epoch_id)
N_SHUFFLES = 19
MASK_S = (1.8, 2.1)
"""Seconds: the masked stretch, inside the one candidate of the unmasked epoch."""


@pytest.fixture(scope="module")
def epoch():
    return build_tiling_tetrode()


def _detect(epoch, *, is_missing=None, spike_times=None, features=None):
    return run_epoch(
        epoch.time,
        epoch.linear_position,
        np.zeros(epoch.time.size),
        epoch.spike_times if spike_times is None else spike_times,
        epoch.spike_waveform_features if features is None else features,
        epoch.encoding_model,
        IDENTITY,
        swr_intervals=None,
        swr_observed_intervals=None,
        swr_definition="none",
        seed=SEED,
        config=CONFIG,
        n_shuffles=N_SHUFFLES,
        is_missing=is_missing,
    )


def _missing(epoch) -> np.ndarray:
    return (epoch.time >= MASK_S[0]) & (epoch.time <= MASK_S[1])


def _windows_touching(missing: np.ndarray) -> np.ndarray:
    """Each window (start every stride, ``window_n_samples`` rows) holding a
    missing row; the final row, which no spike reaches, is not decoded."""
    rows = missing[:-1]
    n_windows = 1 + (rows.size - CONFIG.window_n_samples) // CONFIG.stride_n_samples
    starts = np.arange(n_windows) * CONFIG.stride_n_samples
    return np.array([rows[s : s + CONFIG.window_n_samples].any() for s in starts])


@pytest.fixture(scope="module")
def unmasked(epoch):
    return _detect(epoch)


@pytest.fixture(scope="module")
def masked(epoch):
    return _detect(epoch, is_missing=_missing(epoch))


def test_the_unmasked_epoch_has_one_candidate_across_the_mask(epoch, unmasked):
    runs = unmasked.candidates.runs
    assert len(runs) == 1
    assert _windows_touching(_missing(epoch))[runs[0, 0] : runs[0, 1] + 1].any()
    assert unmasked.table.n_shuffles.iloc[0] == N_SHUFFLES


def test_masked_bins_are_missing_in_the_likelihood(epoch, masked):
    """No window holding a masked bin has a posterior, and every other window
    is decoded."""
    touching = _windows_touching(_missing(epoch))
    usable = masked.candidates.decisions.gates.posterior_usable

    assert touching.any()
    assert not usable[touching].any()
    assert usable[~touching].all()


def test_no_candidate_window_touches_the_mask(epoch, masked):
    touching = _windows_touching(_missing(epoch))

    assert len(masked.candidates.runs) == 2, "one candidate on each side"
    for first, last in masked.candidates.runs:
        assert not touching[first : last + 1].any()
    # what the shuffle re-decodes: every row of every window of the candidate
    rows = np.zeros(epoch.time.size, dtype=bool)
    for first, last in masked.candidates.runs:
        rows[
            first * CONFIG.stride_n_samples : last * CONFIG.stride_n_samples
            + CONFIG.window_n_samples
        ] = True
    assert not (rows & _missing(epoch)).any()
    assert (masked.table.n_shuffles == N_SHUFFLES).all()


def test_masked_time_enters_no_null_and_no_normalization(epoch, masked):
    """Whatever spikes the masked bins hold, nothing changes: not one window's
    posterior, the candidates, nor any shuffle draw."""
    rng = np.random.default_rng(3)
    inside = rng.uniform(MASK_S[0], MASK_S[1], 400)
    spike_times = np.concatenate([epoch.spike_times[0], inside])
    features = np.concatenate(
        [epoch.spike_waveform_features[0], rng.normal(150.0, 80.0, (400, 2))]
    )
    order = np.argsort(spike_times, kind="stable")
    garbled = _detect(
        epoch,
        is_missing=_missing(epoch),
        spike_times=[spike_times[order]],
        features=[features[order]],
    )

    for name in ("center_index", "spread_cm", "jump_cm"):
        np.testing.assert_array_equal(
            getattr(garbled.candidates.decisions, name),
            getattr(masked.candidates.decisions, name),
            err_msg=name,
        )
    np.testing.assert_array_equal(garbled.candidates.runs, masked.candidates.runs)
    pd.testing.assert_frame_equal(garbled.table, masked.table)
    pd.testing.assert_frame_equal(garbled.shuffle_draws, masked.shuffle_draws)


def test_windows_before_the_mask_are_decoded_as_without_it(epoch, unmasked, masked):
    """A window before the mask sums the same rows from the same start; after
    it the running sum no longer carries the masked rows, so it agrees to
    rounding."""
    touching = _windows_touching(_missing(epoch))
    before = np.arange(touching.size) < np.flatnonzero(touching)[0]
    after = np.arange(touching.size) > np.flatnonzero(touching)[-1]
    unmasked_decisions = unmasked.candidates.decisions
    masked_decisions = masked.candidates.decisions

    np.testing.assert_array_equal(
        masked_decisions.spread_cm[before], unmasked_decisions.spread_cm[before]
    )
    np.testing.assert_array_equal(
        masked_decisions.center_index[~touching],
        unmasked_decisions.center_index[~touching],
    )
    np.testing.assert_allclose(
        masked_decisions.spread_cm[after],
        unmasked_decisions.spread_cm[after],
        rtol=1e-9,
    )


def test_no_masked_bin_is_the_epoch_without_a_mask(epoch, unmasked):
    nothing = _detect(epoch, is_missing=np.zeros(epoch.time.size, dtype=bool))

    pd.testing.assert_frame_equal(nothing.table, unmasked.table)
    pd.testing.assert_frame_equal(nothing.shuffle_draws, unmasked.shuffle_draws)


def test_the_mask_must_be_on_the_grid(epoch):
    with pytest.raises(ValueError, match="is_missing"):
        _detect(epoch, is_missing=np.zeros(epoch.time.size - 1, dtype=bool))
