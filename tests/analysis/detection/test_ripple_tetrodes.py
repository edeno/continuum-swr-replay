"""Rule R-T: keep the CA1 band tetrodes that ripple with the rest of CA1.

Synthetic bands at 1000 Hz: white noise on every tetrode, shared ripple bursts
on the tetrodes that ripple together, and, on a "rogue" tetrode, bursts of its
own at other times while it stays at baseline during the shared ones (the j16
pattern of ``followup_c1c_report.md``). Nothing here reads data.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

from continuum_swr_replay.analysis.detection.ripple_tetrodes import (
    COUPLING_CUT,
    LOO_THRESHOLD_Z,
    TetrodeSelectionError,
    check_one_channel_per_tetrode,
    select_ripple_tetrodes,
)

FS = 1000.0
DURATION_S = 120.0
SHARED_RIPPLE_TIMES_S = np.arange(2.0, DURATION_S - 1.0, 4.0)
"""Thirty shared ripples, 4 s apart: about 1 % of the time, as SWRs are."""
RIPPLE_AMPLITUDE = 8.0
"""Peak of a burst, against white noise of SD 1."""


def _burst(time: NDArray, at_s: float) -> NDArray:
    """A 160 Hz ripple under a Gaussian envelope of SD 15 ms, peak 1."""
    return np.exp(-0.5 * ((time - at_s) / 0.015) ** 2) * np.sin(
        2 * np.pi * 160.0 * (time - at_s)
    )


def _band(
    n_tetrodes: int, rogues: tuple[int, ...] = (), seed: int = 0
) -> tuple[NDArray, NDArray]:
    """A synthetic stored ripple band.

    Every tetrode not in ``rogues`` carries the shared ripples. Rogue tetrode
    ``k`` instead carries thirty bursts of its own, between the shared ones
    and apart from the other rogues', and nothing during the shared ripples.

    Returns
    -------
    time : NDArray, shape (n_time,)
    band : NDArray, shape (n_time, n_tetrodes)
    """
    rng = np.random.default_rng(seed)
    time = np.arange(round(DURATION_S * FS)) / FS
    band = rng.normal(0.0, 1.0, (time.size, n_tetrodes))
    shared = sum(_burst(time, at) for at in SHARED_RIPPLE_TIMES_S)
    for tetrode in range(n_tetrodes):
        if tetrode in rogues:
            offset = 1.0 + 0.4 * rogues.index(tetrode)
            own = sum(_burst(time, at + offset) for at in SHARED_RIPPLE_TIMES_S)
            band[:, tetrode] += RIPPLE_AMPLITUDE * own
        else:
            band[:, tetrode] += RIPPLE_AMPLITUDE * shared
    return time, band


def _still(time: NDArray) -> NDArray:
    return np.full(time.size, 2.0)


def _nothing_masked(time: NDArray) -> NDArray[np.bool_]:
    return np.zeros(time.size, dtype=bool)


def test_a_tetrode_rippling_on_its_own_is_dropped_and_the_others_kept():
    time, band = _band(6, rogues=(2,))

    selection = select_ripple_tetrodes(band, _still(time), _nothing_masked(time), FS)

    assert selection.keep.tolist() == [True, True, False, True, True, True]
    assert selection.coupling[2] < COUPLING_CUT
    assert (np.delete(selection.coupling, 2) > 3.0).all()
    # each tetrode's coupling is read where the other tetrodes ripple
    assert (selection.n_high > 0).all()


def test_tetrodes_that_ripple_together_are_all_kept():
    time, band = _band(6)

    selection = select_ripple_tetrodes(band, _still(time), _nothing_masked(time), FS)

    assert selection.keep.all()
    assert (selection.coupling >= COUPLING_CUT).all()


@pytest.mark.parametrize("left_out", ["masked", "moving", "not_finite"])
def test_samples_outside_the_reference_do_not_decide(left_out):
    """A loud broadband stretch on every tetrode, the rogue included, would
    make the rogue look coupled. Masked, moving or non-finite, it is not a
    reference sample, so the rogue is dropped."""
    time, band = _band(6, rogues=(0,))
    stretch = (time >= 60.0) & (time < 62.0)
    band[stretch] += np.random.default_rng(1).normal(0.0, 30.0, (stretch.sum(), 6))
    speed = _still(time)
    unobserved = _nothing_masked(time)

    as_reference = select_ripple_tetrodes(band, speed, unobserved, FS)
    assert as_reference.keep[0], "the stretch must be able to keep the rogue"

    if left_out == "masked":
        unobserved = stretch
    elif left_out == "moving":
        speed = np.where(stretch, 20.0, speed)
    else:
        band[stretch, 3] = np.nan
    selection = select_ripple_tetrodes(band, speed, unobserved, FS)

    assert selection.keep.tolist() == [False, True, True, True, True, True]
    assert selection.n_reference == as_reference.n_reference - stretch.sum()


def test_the_speed_rule_is_strict():
    """Reference samples are below 4 cm/s; one at exactly 4 is not."""
    time, band = _band(6)
    speed = np.where(time < 30.0, 4.0, 2.0)

    selection = select_ripple_tetrodes(band, speed, _nothing_masked(time), FS)

    assert selection.n_reference == np.count_nonzero(time >= 30.0)


@pytest.mark.parametrize(
    ("n_tetrodes", "n_rogues"),
    [(6, 2), (9, 3), (13, 4)],
    ids=["a third of 6", "a third of 9", "4 of wilbur's 13"],
)
def test_up_to_a_third_dropped_is_allowed(n_tetrodes, n_rogues):
    rogues = tuple(range(n_rogues))
    time, band = _band(n_tetrodes, rogues=rogues)

    selection = select_ripple_tetrodes(band, _still(time), _nothing_masked(time), FS)

    assert np.flatnonzero(~selection.keep).tolist() == list(rogues)


@pytest.mark.parametrize(
    ("n_tetrodes", "n_rogues"),
    [(6, 3), (9, 4), (13, 5)],
    ids=["3 of 6", "4 of 9", "5 of wilbur's 13"],
)
def test_more_than_a_third_dropped_fails_the_epoch(n_tetrodes, n_rogues):
    rogues = tuple(range(n_rogues))
    time, band = _band(n_tetrodes, rogues=rogues)

    with pytest.raises(TetrodeSelectionError, match="more than a third") as raised:
        select_ripple_tetrodes(band, _still(time), _nothing_masked(time), FS)

    # what was measured is kept for the record
    selection = raised.value.selection
    assert np.flatnonzero(~selection.keep).tolist() == list(rogues)
    assert selection.coupling.shape == (n_tetrodes,)


@pytest.mark.parametrize("how", ["all moving", "all masked", "no ripple reaches 5"])
def test_no_reference_sample_with_a_loud_consensus_fails_the_epoch(how):
    time, band = _band(6)
    speed = _still(time)
    unobserved = _nothing_masked(time)
    if how == "all moving":
        speed[:] = 10.0
    elif how == "all masked":
        unobserved[:] = True
    else:
        # rippling throughout, its amplitude swinging slowly: the consensus
        # is never 5 SD above its own mean
        swing = (1.0 + np.sin(2 * np.pi * 0.5 * time)) * np.sin(2 * np.pi * 160 * time)
        band = band + RIPPLE_AMPLITUDE * swing[:, None]

    with pytest.raises(TetrodeSelectionError, match="LOO") as raised:
        select_ripple_tetrodes(band, speed, unobserved, FS)

    assert (raised.value.selection.n_high == 0).any()


def test_a_flat_channel_fails_the_epoch():
    """A flat channel has MADn 0, so its robust z and coupling are undefined
    although the others ripple: the epoch fails closed, without a warning."""
    time, band = _band(6)
    band[:, 4] = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TetrodeSelectionError, match="MADn") as raised:
            select_ripple_tetrodes(band, _still(time), _nothing_masked(time), FS)

    selection = raised.value.selection
    assert np.isnan(selection.coupling[4])
    assert selection.n_high[4] > 0
    assert np.isfinite(np.delete(selection.coupling, 4)).all()


def test_one_band_channel_per_tetrode_is_required():
    check_one_channel_per_tetrode(np.array(["0", "1", "2", "10"]))

    with pytest.raises(
        TetrodeSelectionError, match="one band channel per tetrode"
    ) as raised:
        check_one_channel_per_tetrode(np.array(["0", "1", "1", "3", "3"]))

    assert "'1'" in str(raised.value) and "'3'" in str(raised.value)
    assert raised.value.selection is None


def _prototype_coupling(band: NDArray, speed: NDArray, unobserved: NDArray) -> NDArray:
    """``followup/scripts/c1c_consensus.py``'s rule, as the prototype wrote it."""
    finite = np.isfinite(band).all(1)
    rip = np.where(finite[:, None], band, 0.0)
    env = np.abs(hilbert(rip, axis=0))
    med = np.median(env[finite], 0)
    madn = 1.4826 * np.median(np.abs(env[finite] - med), 0)
    envz = (env - med) / madn
    refs = finite & np.isfinite(speed) & (speed < 4.0) & ~unobserved
    e2 = env**2
    loo = np.sqrt(
        np.maximum(
            gaussian_filter1d(e2.sum(1)[:, None] - e2, sigma=4.0, axis=0, truncate=8.0),
            0,
        )
    )
    coupling = np.full(band.shape[1], np.nan)
    for c in range(band.shape[1]):
        v = loo[refs, c]
        zl = (loo[:, c] - v.mean()) / v.std()
        hi = refs & (zl >= LOO_THRESHOLD_Z)
        coupling[c] = np.median(envz[hi, c])
    return coupling


def test_the_coupling_is_the_prototypes():
    """The port reproduces the C1c prototype's coupling exactly."""
    time, band = _band(7, rogues=(4,), seed=3)
    band[5000:5100] = np.nan
    speed = np.where((time > 20.0) & (time < 25.0), 12.0, 1.0)
    unobserved = (time > 70.0) & (time < 71.5)

    selection = select_ripple_tetrodes(band, speed, unobserved, FS)

    np.testing.assert_array_equal(
        selection.coupling, _prototype_coupling(band, speed, unobserved)
    )


def test_the_record_names_each_tetrode():
    time, band = _band(4, rogues=(1,))
    selection = select_ripple_tetrodes(band, _still(time), _nothing_masked(time), FS)

    record = selection.record(
        electrode_ids=np.array([0, 4, 8, 12]),
        electrode_group_names=np.array(["0", "1", "2", "3"]),
        reference_ids=np.array([16, 16, 81, -1]),
    )

    assert record == {
        "electrode_ids": [0, 4, 8, 12],
        "electrode_group_names": ["0", "1", "2", "3"],
        "reference_ids": [16, 16, 81, -1],
        "coupling": selection.coupling.tolist(),
        "n_high": selection.n_high.tolist(),
        "kept": [True, False, True, True],
        "n_reference": time.size,
        "dropped_electrode_groups": ["1"],
        "dropped_electrode_ids": [4],
    }
