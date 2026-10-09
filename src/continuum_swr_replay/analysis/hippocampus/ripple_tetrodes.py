"""Rule R-T: per epoch, keep the CA1 band tetrodes that ripple with the rest.

A tetrode whose ripple envelope stays low while the other CA1 tetrodes ripple
is not reporting CA1's ripples, whatever its own ripple-band power. Kay's
consensus sums raw squared envelopes, so such a tetrode, if loud, can carry
the consensus on its own. In j16, electrode groups 0, 10 and 22 did exactly
that on every day: they carried 5-7 times the ripple-band power of the other
tetrodes and most of j16's events without SWR features
(``.claude/research/refresh_artifact_detector/followup/followup_c1c_report.md``).

The rule, stated before its effect was measured (``followup/c1c_rule.json``)
and ported from ``followup/scripts/c1c_consensus.py``. For each CA1 band
tetrode ``c`` (the stored band holds one channel per tetrode in every epoch
of the cohort, so a column is its tetrode):

1. ``env_c = |hilbert(band_c)|``, with non-finite rows set to zero, and its
   robust z ``z_c = (env_c - median) / MADn`` over the epoch's finite rows.
2. ``LOO_c = sqrt(gaussian_smooth(sum over the other tetrodes of env**2,
   sigma 4 ms))``, z-scored by its mean and SD over the reference samples:
   finite rows, speed below 4 cm/s, outside the artifact mask.
3. ``coupling_c`` = the median of ``z_c`` over the reference samples where
   that z-score is at least 5.
4. Drop ``c`` when ``coupling_c < 1.0``.

It fails closed (:class:`TetrodeSelectionError`) when it would drop more than
a third of the tetrodes, or when a tetrode has no reference sample where the
others' consensus reaches 5 z, so its coupling is undefined. Every detector
of the SWR runner then reads the kept tetrodes. Design note:
``docs/design/2026-10-09-swr-tetrode-selection.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)

__all__ = [
    "COUPLING_CUT",
    "LOO_SMOOTHING_SIGMA_S",
    "LOO_THRESHOLD_Z",
    "MAXIMUM_DROPPED_FRACTION",
    "REFERENCE_SPEED_CM_PER_S",
    "TetrodeSelection",
    "TetrodeSelectionError",
    "select_ripple_tetrodes",
]

COUPLING_CUT = 1.0
"""Robust z. A mechanism constant: a rise of less than one robust SD while the
rest of CA1 ripples. Evidence: j16's groups 0, 10 and 22 score -0.12 to 0.81 in
all 58 epochs, and every other j16 tetrode scores at least 1.78
(``c1c_rule.json``, ``cut_basis``)."""

LOO_THRESHOLD_Z = 5.0
"""z of the other tetrodes' consensus: the moments the rest of CA1 ripples
strongly. Stated before any effect was measured. Evidence: at 5 the rule drops
all 174 culprit tetrode-epochs and 2 weakly rippling ones; 25 % lower (3.75)
drops 61 rippling ones, 25 % higher (6.25) none
(``followup/justification_report.md`` §3.3)."""

LOO_SMOOTHING_SIGMA_S = 0.004
"""Seconds: the consensus is smoothed as Kay smooths its own (``smoothing_sigma``
0.004 in the SWR inventory's settings). Evidence: 3 or 5 ms drops 1 or 4
weakly rippling tetrode-epochs and every culprit (§3.3)."""

LOO_SMOOTHING_TRUNCATE = 8.0
"""The Gaussian kernel reaches 8 SD on each side, as in the prototype."""

REFERENCE_SPEED_CM_PER_S = float(DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC)
"""cm/s: reference samples are strictly slower, the project's immobility rule."""

MAXIMUM_DROPPED_FRACTION = Fraction(1, 3)
"""The epoch fails when the rule would drop more than this share of its
tetrodes: CA1 itself would then be in question, not a few tetrodes. Exact, so
a third of 9 tetrodes (3) is allowed."""

MADN_SCALE = 1.4826
"""MAD to normal-consistent SD."""


@dataclass(frozen=True, eq=False)
class TetrodeSelection:
    """What rule R-T measured for one epoch, and which tetrodes it keeps.

    Attributes
    ----------
    coupling : NDArray, shape (n_tetrodes,)
        Each tetrode's median envelope robust z where the others' consensus
        reaches :data:`LOO_THRESHOLD_Z`; NaN where it never does.
    n_high : NDArray of int, shape (n_tetrodes,)
        The reference samples that median is taken over.
    n_reference : int
        Reference samples: finite, slower than the speed rule, unmasked.
    keep : NDArray of bool, shape (n_tetrodes,)
        ``coupling >= COUPLING_CUT``.
    """

    coupling: NDArray[np.float64]
    n_high: NDArray[np.int64]
    n_reference: int
    keep: NDArray[np.bool_]

    def record(
        self,
        electrode_ids: NDArray,
        electrode_group_names: NDArray,
        reference_ids: NDArray,
    ) -> dict[str, object]:
        """The manifest's record of the selection.

        Every list is in the band's column order, one entry per tetrode.

        Parameters
        ----------
        electrode_ids : NDArray, shape (n_tetrodes,)
            Each column's electrode.
        electrode_group_names : NDArray, shape (n_tetrodes,)
            Each column's electrode group (its tetrode).
        reference_ids : NDArray, shape (n_tetrodes,)
            Each column's reference electrode; -1 for none.
        """
        groups = np.asarray(electrode_group_names).astype(str)
        return {
            "electrode_ids": np.asarray(electrode_ids).tolist(),
            "electrode_group_names": groups.tolist(),
            "reference_ids": np.asarray(reference_ids).tolist(),
            "coupling": self.coupling.tolist(),
            "n_high": self.n_high.tolist(),
            "kept": self.keep.tolist(),
            "n_reference": self.n_reference,
            "dropped_electrode_groups": groups[~self.keep].tolist(),
            "dropped_electrode_ids": np.asarray(electrode_ids)[~self.keep].tolist(),
        }


class TetrodeSelectionError(ValueError):
    """Rule R-T failed closed for the epoch.

    Attributes
    ----------
    selection : TetrodeSelection
        What was measured, for the record.
    """

    def __init__(self, message: str, selection: TetrodeSelection) -> None:
        super().__init__(message)
        self.selection = selection


def select_ripple_tetrodes(
    ripple_band: NDArray,
    speed: NDArray,
    unobserved: NDArray[np.bool_],
    sampling_frequency: float,
    *,
    loo_threshold_z: float = LOO_THRESHOLD_Z,
) -> TetrodeSelection:
    """Apply rule R-T to one epoch's stored ripple band.

    Parameters
    ----------
    ripple_band : NDArray, shape (n_time, n_tetrodes)
        The stored 150-250 Hz band of the CA1 tetrodes, one channel each. A
        row with any non-finite value is not finite.
    speed : NDArray, shape (n_time,)
        Head speed in cm/s on the band's times; NaN is unknown and not
        immobile.
    unobserved : NDArray of bool, shape (n_time,)
        The artifact mask on the band's times.
    sampling_frequency : float
        The band's nominal rate in Hz.
    loo_threshold_z : float, optional
        :data:`LOO_THRESHOLD_Z`; another value only for a sensitivity check.

    Returns
    -------
    TetrodeSelection

    Raises
    ------
    TetrodeSelectionError
        If a tetrode has no reference sample where the others' consensus
        reaches ``loo_threshold_z`` (no reference sample at all included),
        or if more than a third of the tetrodes would be dropped.
    """
    band = np.asarray(ripple_band, dtype=np.float64)
    finite = np.isfinite(band).all(axis=1)
    speed = np.asarray(speed, dtype=np.float64)
    reference = (
        finite
        & np.isfinite(speed)
        & (speed < REFERENCE_SPEED_CM_PER_S)
        & ~np.asarray(unobserved, dtype=bool)
    )

    envelope = np.abs(hilbert(np.where(finite[:, None], band, 0.0), axis=0))
    power = envelope**2
    median = np.median(envelope[finite], axis=0)
    madn = MADN_SCALE * np.median(np.abs(envelope[finite] - median), axis=0)
    robust_z = envelope
    robust_z -= median
    robust_z /= madn

    others = power.sum(axis=1)[:, np.newaxis] - power
    del power
    consensus = gaussian_filter1d(
        others,
        sigma=LOO_SMOOTHING_SIGMA_S * sampling_frequency,
        axis=0,
        truncate=LOO_SMOOTHING_TRUNCATE,
    )
    del others
    np.maximum(consensus, 0.0, out=consensus)
    np.sqrt(consensus, out=consensus)

    n_tetrodes = band.shape[1]
    coupling = np.full(n_tetrodes, np.nan)
    n_high = np.zeros(n_tetrodes, dtype=np.int64)
    if reference.any():
        for tetrode in range(n_tetrodes):
            sample = consensus[reference, tetrode]
            z = (consensus[:, tetrode] - sample.mean()) / sample.std()
            high = reference & (z >= loo_threshold_z)
            n_high[tetrode] = np.count_nonzero(high)
            if n_high[tetrode]:
                coupling[tetrode] = np.median(robust_z[high, tetrode])

    selection = TetrodeSelection(
        coupling=coupling,
        n_high=n_high,
        n_reference=int(np.count_nonzero(reference)),
        keep=coupling >= COUPLING_CUT,
    )
    if (n_high == 0).any():
        raise TetrodeSelectionError(
            f"No reference sample has LOO >= {loo_threshold_z:g} z for tetrode "
            f"column(s) {np.flatnonzero(n_high == 0).tolist()} "
            f"({selection.n_reference} reference samples: finite, slower than "
            f"{REFERENCE_SPEED_CM_PER_S:g} cm/s and outside the artifact mask), "
            "so their coupling is undefined.",
            selection,
        )
    n_dropped = int(np.count_nonzero(~selection.keep))
    if n_dropped > MAXIMUM_DROPPED_FRACTION * n_tetrodes:
        raise TetrodeSelectionError(
            f"Rule R-T would drop {n_dropped} of {n_tetrodes} tetrodes, more than "
            "a third.",
            selection,
        )
    return selection
