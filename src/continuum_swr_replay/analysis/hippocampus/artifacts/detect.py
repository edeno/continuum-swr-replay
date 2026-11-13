"""The artifact detector: its constants and its four rules.

The detector masks a period only when it shows an artifact signature, never on
power alone (a large SWR also has high ripple power):

- **S**: ripple-band power raised on a fixed subset of CA1 channels for at
  least 2 s (:func:`s_cores`).
- **T-ref**: an abrupt level jump on the reference electrode, which sits in
  white matter outside CA1 (:func:`t_ref_events`).
- **T-ca1**: an abrupt and persistent level step on at least 3 CA1 channels
  at once (:func:`t_ca1_events`).
- **T-spk**: a spike line, at least 40 % of tetrodes in one 200-µs bin, with
  at least 20 % of those marks out of distribution (:func:`t_spk_events`).

T signatures at most 50 ms apart merge into one T core (:func:`t_cores`); the
merge chains, so a core can last longer than 50 ms. Each rule is its own
function, so a further signature is one more event source.
:func:`detect_artifacts` runs them on one epoch and returns unpadded cores
with their evidence; one table pads them into the mask (``events.artifact_masks.PADDING_S``).

By construction no rule can mask a ripple seen on one CA1 electrode alone: S
needs at least 2 channels for at least 2 s, T-ca1 needs at least 3 channels
with a lasting step, T-spk needs at least 40 % of tetrodes, and T-ref reads only
the reference electrodes. (T-ref takes the maximum over the reference
electrodes, so a jump on one reference electrode alone does fire it, as
intended.)

**Thresholds (rule P).** Each artifact threshold on a statistic SWRs also
reach is :data:`THRESHOLD_MARGIN` (1.5) times the largest value that statistic
reaches in the calibration SWR set: the reference SWR set, the model-reviewed
SWRs and the giant-day Kay∩Karlsson events, each measured over the event
widened by its padding. One animal's maximum over the other four animals'
maximum reaches 1.59 (T-ca1, chimi held out): 1.5 covers T-ref (1.27) and
R-rip (1.29), and K's 1.59 is why all five animals are in the calibration set.
The reference SWR set is
rule-defined (Kay∩Karlsson, 30-200 ms, speed < 4 cm/s over ± 250 ms, a sharp
wave, MUA >= 2x); model-reviewed labels come from one AI reviewer, with no
human and no second rater. Neither is ground truth. Window lengths, bins, run
lengths, persistence and padding are definitions, placed by a mechanism or a
measured tail.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay import parameters
from continuum_swr_replay.analysis.events.artifact_intervals import (
    empty_cores,
    make_cores,
    validate_cores,
)
from continuum_swr_replay.analysis.events.artifact_masks import artifact_intervals
from continuum_swr_replay.analysis.hippocampus.artifacts.features import (
    J5_REACH,
    L20_REACH,
    channel_specificity,
    level_jump,
    level_shift,
    nearest_mark_distance,
    persistent_step_count,
    reference_jump,
    robust_jump,
    spike_lines,
    window_band_power,
    window_centers,
    window_valid,
)
from continuum_swr_replay.intervals import contiguous_true_runs, interval_mask

__all__ = [
    "CONSTANT_NAMES",
    "EVENT_COLUMNS",
    "detect_artifacts",
    "detector_constants",
    "s_cores",
    "spike_ood",
    "t_ca1_events",
    "t_cores",
    "t_ref_events",
    "t_spk_events",
]

# --- rule P --------------------------------------------------------------------
THRESHOLD_MARGIN = 1.5
"""Rule P's margin over the calibration maximum. One animal's maximum over the
other four animals' maximum reaches 1.27 (T-ref) to 1.59 (T-ca1, chimi held
out): 1.5 covers T-ref and R-rip (1.29), and K's 1.59 is why the calibration
set holds all five animals."""
S_CORE_CALIBRATION_MAX = 2.8723976612091064
"""The largest window specificity over the calibration SWR set: a reference-set
SWR in j16. Measured with S padded by 0 s. Under the shipped 1-s S padding one
calibration event, a model-reviewed SWR 0.55 s before an S block, reaches 19.4;
the design note explains it."""
T_REF_CALIBRATION_MAX = 11.715611457824707
"""The largest reference |J5| (robust units) over the calibration SWR set: a
giant-day event (reference set 9.2, model-reviewed SWRs 7.2)."""
T_CA1_K_CALIBRATION_MAX = 16.682405471801758
"""The largest k-th largest persistent CA1 |J5| (robust units; k the channels
T-ca1 needs) over the calibration SWR set: chimi's giant day and a
model-reviewed SWR. Its cross-animal ratio, 1.59, is the largest."""

# --- S: sustained, channel-specific ripple-band noise --------------------------
S_WINDOW_S = 1.0
"""Window length (s): spec is computed per 1-s window."""
S_SEGMENT_S = 0.064
"""Welch segment (s): 15 Hann segments of 64 ms per window."""
S_BAND_HZ = (150.0, 250.0)
"""The ripple band of the window power, low inclusive, high exclusive."""
S_CORE = THRESHOLD_MARGIN * S_CORE_CALIBRATION_MAX
"""Core spec, rule P: 1.5 x 2.872 = 4.309 (the prototype's 4 had a 1.39x
margin)."""
S_CORE_RUN = 2
"""Consecutive core windows, a definition (>= 2 s): 1 adds 11 reference-set
SWRs and 121 s."""
S_EXT = 2.0
"""Extension spec: the run of windows >= 2 holding a core."""
S_GAP = 2
"""Blocks <= 2 windows apart merge: a step inside a block drops spec for one
window; without the fill senor20201101_14_r7 keeps SD / robust SD 1.61."""

# --- T-ref: a jump on the reference electrode -------------------------------
T_REF = THRESHOLD_MARGIN * T_REF_CALIBRATION_MAX
"""Reference |J5| (robust units), rule P: 1.5 x 11.716 = 17.573 (the
prototype's 15 had a 1.28x margin, and placed on four animals it failed on
chimi's giant days). Model-reviewed steps have a median of 31.9."""

# --- T-ca1: a persistent step on several CA1 channels -----------------------
T_CA1_K = THRESHOLD_MARGIN * T_CA1_K_CALIBRATION_MAX
"""CA1 |J5| (robust units) of a persistent step, rule P: 1.5 x 16.682 = 25.024
(the prototype's 20 had a 1.20x margin). Giant-SWR onsets fire at
K <= 12."""
T_CA1_RHO = 0.5
"""|L20| >= 0.5 |J5| with J5's sign: the step persists, a ripple cycle does not."""
T_CA1_FRAC = 0.15
"""Fraction of CA1 channels stepping at one sample: the calibration SWR set
reaches 1 channel, so k = 3-4 of 13-22 channels is a 3-4x margin."""
T_CA1_MIN = 3
"""At least 3 channels, however few the epoch has."""

# --- T-spk: an out-of-distribution spike line ------------------------------
T_SPK_BIN_S = 200e-6
"""Line bin (s), on two grids offset by half a bin: at 0.5 ms a clean-day giant
SWR is masked."""
T_SPK_FRAC = 0.40
"""Fraction of tetrodes in one bin. Off rule P: reference-set SWRs reach 0.364
in bins with OOD marks (a thin 1.10x margin), and the spike-only lines sit at
0.50-0.55, so no value has a 1.5x margin and still catches them. It rests on 3
model-reviewed lines; every spike-only core of the full 255-epoch run
(``swr_source.ARTIFACT_INTERVALS_RUN_ID``) was inspected when the run was
accepted."""
T_SPK_MIN_TETRODES = 3
"""A line needs at least 3 tetrodes: the prototype's line table kept bins with
>= max(3, 25 %) of tetrodes (artdet_detector.spike_lines); with 8 or more
tetrodes only the 40 % rule binds."""
T_SPK_OOD = 0.20
"""OOD fraction of the bin's spikes. Off rule P: the calibration SWR set's
maximum is 0, so it sits 1.35x below the smallest model-reviewed spike-only
line (0.27). It rests on those 3 lines; every spike-only core of the full
255-epoch run (``swr_source.ARTIFACT_INTERVALS_RUN_ID``) was inspected when the
run was accepted."""
T_SPK_OOD_NN = 4.0
"""A mark is OOD when its nearest reference mark is more than 4 x the decoder's
waveform_std away: there the decoder's mark kernel has no support."""
T_SPK_OOD_OWN_EVENT_S = 1e-3
"""A spike's own event: the same tetrode's marks at most 1 ms from it (itself
included), left out of its nearest-mark search. A neuron cannot fire twice
within its absolute refractory period (about 1 ms), so two detections that close
on one tetrode are two units, whose marks differ and are not each other's
nearest neighbour, or one waveform detected twice. Leaving them out stops an
artifact's twin detections from holding each other in distribution; on the 23
acceptance epochs it made 20 other spikes OOD and changed no other core. The
first full run, made before this rule (run ``a1083731940b``), missed the
chimi20200227_02_r1 line at 8.614 s this way (3 of 17 spikes OOD, 0.176; 9 of 17 with 1 ms left out). A definition,
off rule P. SpikeInterface 0.99.1's default refractory period is 1 ms
(``qualitymetrics`` ``rp_violation``, ``curation.get_potential_auto_merge``),
and it treats one unit's spikes within 0.3 ms as one spike detected twice
(``curation.remove_duplicated_spikes``, ``censored_period_ms``)."""
WAVEFORM_STD: float = parameters.WAVEFORM_STD
"""The decoder's mark-kernel bandwidth (µV), read from its settings
(``parameters.WAVEFORM_STD``, ``DecodingConfig``'s default)."""

# --- merging T signatures ------------------------------------------------------
T_MERGE_S = 0.05
"""T signatures at most 50 ms after the previous one join its core."""

CONSTANT_NAMES = (
    "THRESHOLD_MARGIN",
    "S_CORE_CALIBRATION_MAX",
    "T_REF_CALIBRATION_MAX",
    "T_CA1_K_CALIBRATION_MAX",
    "S_WINDOW_S",
    "S_SEGMENT_S",
    "S_BAND_HZ",
    "S_CORE",
    "S_CORE_RUN",
    "S_EXT",
    "S_GAP",
    "T_REF",
    "T_CA1_K",
    "T_CA1_RHO",
    "T_CA1_FRAC",
    "T_CA1_MIN",
    "T_SPK_BIN_S",
    "T_SPK_FRAC",
    "T_SPK_MIN_TETRODES",
    "T_SPK_OOD",
    "T_SPK_OOD_NN",
    "T_SPK_OOD_OWN_EVENT_S",
    "WAVEFORM_STD",
    "T_MERGE_S",
)
"""Every constant the rules read, in the order above."""


def detector_constants() -> dict[str, float | int | tuple[float, float]]:
    """Every constant's current value, by its lower-case name (``t_ref``, ...)."""
    values = globals()
    return {name.lower(): values[name] for name in CONSTANT_NAMES}


TIME_GAP_STEPS = 1.5
"""A timestamp step longer than 1.5 sample periods counts as a gap in
``attrs["counts"]["n_time_gaps"]``. Gaps are counted, not handled: windows
run over consecutive samples, as in the prototype."""

EVENT_COLUMNS = ["time", "signature", "ref_j5", "ca1_frac", "spk_frac", "spk_ood"]
"""A T event table: one row per sample (or spike-line bin) where a T signature
fires, with its evidence; NaN where the signature gives none."""

_EVENT_DTYPES = {
    "time": np.float64,
    "signature": object,
    "ref_j5": np.float64,
    "ca1_frac": np.float64,
    "spk_frac": np.float64,
    "spk_ood": np.float64,
}


def _events(
    signature: str, time: NDArray[np.float64], **evidence: NDArray
) -> pd.DataFrame:
    columns = {
        "time": time,
        "signature": np.full(time.size, signature, dtype=object),
        **{name: np.full(time.size, np.nan) for name in EVENT_COLUMNS[2:]},
        **evidence,
    }
    return pd.DataFrame(columns, columns=EVENT_COLUMNS).astype(_EVENT_DTYPES)


def s_cores(
    centers: NDArray[np.float64],
    specificity: NDArray[np.floating],
    *,
    half_window_s: float = S_WINDOW_S / 2,
) -> pd.DataFrame:
    """Rule S: blocks of 1-s windows with channel-specific ripple power.

    1. A core is a run of at least :data:`S_CORE_RUN` windows with
       ``spec >= S_CORE``.
    2. A block is the run of windows with ``spec >= S_EXT`` that holds a core.
    3. Blocks at most :data:`S_GAP` windows apart merge, the gap windows
       included.

    Parameters
    ----------
    centers : ndarray, shape (n_windows,)
        Each window's mean sample time (seconds).
    specificity : ndarray, shape (n_windows,)
        From ``features.channel_specificity``; NaN never fires.
    half_window_s : float
        A block runs from its first window's center less this to its last
        window's center plus this.

    Returns
    -------
    pd.DataFrame
        Cores in :data:`~continuum_swr_replay.analysis.events.artifact_intervals.CORE_COLUMNS`,
        kind and signatures ``"S"``, with ``spec_max`` (over the block, gap
        windows included) and ``n_windows``.
    """
    spec = np.asarray(specificity, dtype=np.float64)
    core_first, core_stop = contiguous_true_runs(spec >= S_CORE).T
    long_enough = (core_stop - core_first) >= S_CORE_RUN
    in_core = np.zeros(spec.size + 1, dtype=np.int64)
    np.add.at(in_core, core_first[long_enough], 1)
    np.add.at(in_core, core_stop[long_enough], -1)
    in_core = np.cumsum(in_core)[:-1] > 0
    ext_first, ext_stop = contiguous_true_runs(spec >= S_EXT).T
    cores_in = np.r_[0, np.cumsum(in_core)]
    holds_core = (cores_in[ext_stop] - cores_in[ext_first]) > 0
    first, stop = ext_first[holds_core], ext_stop[holds_core]
    if first.size == 0:
        return empty_cores()
    new_block = np.r_[True, (first[1:] - stop[:-1]) > S_GAP]
    starts = np.flatnonzero(new_block)
    block_first = first[starts]
    block_stop = np.maximum.reduceat(stop, starts)
    spec_max = [
        np.nanmax(spec[a:b]) for a, b in zip(block_first, block_stop, strict=True)
    ]
    return make_cores(
        {
            "kind": "S",
            "signatures": "S",
            "core_start": centers[block_first] - half_window_s,
            "core_end": centers[block_stop - 1] + half_window_s,
            "spec_max": spec_max,
            "n_windows": (block_stop - block_first).astype(np.float64),
        }
    )


def t_ref_events(
    time: NDArray[np.float64], reference_robust_jump: NDArray[np.floating]
) -> pd.DataFrame:
    """Rule T-ref: every sample where the reference |J5| >= :data:`T_REF`.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    reference_robust_jump : ndarray, shape (n_time,)
        From ``features.reference_jump`` (robust units).

    Returns
    -------
    pd.DataFrame
        :data:`EVENT_COLUMNS`, signature ``"ref"``, evidence ``ref_j5``.
    """
    fired = np.flatnonzero(reference_robust_jump >= T_REF)
    return _events(
        "ref", time[fired], ref_j5=reference_robust_jump[fired].astype(np.float64)
    )


def t_ca1_threshold(n_channels: int) -> int:
    """Channels needed for T-ca1: ``max(T_CA1_MIN, ceil(T_CA1_FRAC * n))``."""
    return max(T_CA1_MIN, int(np.ceil(T_CA1_FRAC * n_channels - 1e-9)))


def t_ca1_events(
    time: NDArray[np.float64], step_count: NDArray[np.integer], n_channels: int
) -> pd.DataFrame:
    """Rule T-ca1: every sample where enough CA1 channels step persistently.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    step_count : ndarray of int, shape (n_time,)
        From ``features.persistent_step_count`` at :data:`T_CA1_K` and
        :data:`T_CA1_RHO`.
    n_channels : int
        The epoch's CA1 channels.

    Returns
    -------
    pd.DataFrame
        :data:`EVENT_COLUMNS`, signature ``"ca1"``, evidence ``ca1_frac``.
    """
    fired = np.flatnonzero(step_count >= t_ca1_threshold(n_channels))
    return _events("ca1", time[fired], ca1_frac=step_count[fired] / n_channels)


def t_spk_events(lines: pd.DataFrame) -> pd.DataFrame:
    """Rule T-spk: every spike-line bin with enough tetrodes and OOD marks.

    Parameters
    ----------
    lines : pd.DataFrame
        From ``features.spike_lines`` at :data:`T_SPK_BIN_S`.

    Returns
    -------
    pd.DataFrame
        :data:`EVENT_COLUMNS`, signature ``"spk"``, evidence ``spk_frac`` and
        ``spk_ood``; the time is the bin's center.
    """
    fired = (
        (lines["n_tetrodes"] >= T_SPK_MIN_TETRODES)
        & (lines["tetrode_fraction"] >= T_SPK_FRAC)
        & (lines["ood_fraction"] >= T_SPK_OOD)
    ).to_numpy()
    return _events(
        "spk",
        lines["time"].to_numpy(dtype=np.float64)[fired],
        spk_frac=lines["tetrode_fraction"].to_numpy(dtype=np.float64)[fired],
        spk_ood=lines["ood_fraction"].to_numpy(dtype=np.float64)[fired],
    )


def t_cores(events: pd.DataFrame) -> pd.DataFrame:
    """Merge T events into T cores: an event at most :data:`T_MERGE_S` after
    the previous one joins its core.

    Parameters
    ----------
    events : pd.DataFrame
        :data:`EVENT_COLUMNS` rows from any T rule, in any order.

    Returns
    -------
    pd.DataFrame
        Cores in ``CORE_COLUMNS``, kind ``"T"``, in time order: from the first
        to the last event of each group, the signatures that fired (sorted,
        joined by ``+``) and the largest of each evidence.
    """
    if events.empty:
        return empty_cores()
    ordered = events.sort_values("time", kind="stable").reset_index(drop=True)
    time = ordered["time"].to_numpy()
    group = np.r_[0, np.cumsum(np.diff(time) > T_MERGE_S)]
    by_group = ordered.groupby(group, sort=True)
    signatures = by_group["signature"].agg(lambda names: "+".join(sorted(set(names))))
    evidence = by_group[["ref_j5", "ca1_frac", "spk_frac", "spk_ood"]].max()
    return make_cores(
        {
            "kind": "T",
            "signatures": signatures.to_numpy(),
            "core_start": by_group["time"].min().to_numpy(),
            "core_end": by_group["time"].max().to_numpy(),
            "ref_j5_max": evidence["ref_j5"].to_numpy(),
            "ca1_frac_max": evidence["ca1_frac"].to_numpy(),
            "spk_frac_max": evidence["spk_frac"].to_numpy(),
            "spk_ood_max": evidence["spk_ood"].to_numpy(),
        }
    )


def spike_ood(
    spike_times: Sequence[NDArray[np.float64]],
    spike_marks: Sequence[NDArray[np.floating]],
    span: tuple[float, float],
    excluded: pd.DataFrame,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.bool_], dict[str, int]]:
    """Each spike in the span, its tetrode and whether its mark is OOD.

    The reference set is decode-free: per tetrode, every mark of the epoch
    outside `excluded` (the LFP signatures' intervals). A spike's mark is OOD
    when its nearest reference mark outside its own event (the same tetrode's
    marks within :data:`T_SPK_OOD_OWN_EVENT_S` of it, itself included) is more
    than :data:`T_SPK_OOD_NN` x :data:`WAVEFORM_STD` away.

    Parameters
    ----------
    spike_times : sequence of ndarray, shape (n_spikes_k,)
        Per tetrode, seconds; spikes outside `span` are ignored.
    spike_marks : sequence of ndarray, shape (n_spikes_k, n_features)
        Per tetrode, the marks of `spike_times`.
    span : tuple of float
        The epoch's first and last LFP sample (seconds), closed.
    excluded : pd.DataFrame
        ``start_time`` / ``end_time`` intervals whose marks are not reference.

    Returns
    -------
    times : ndarray, shape (n_spikes,)
        In tetrode order, then time order within a tetrode.
    tetrode : ndarray of int, shape (n_spikes,)
        ``0 .. n_tetrodes - 1`` over the tetrodes with a spike in the span.
    is_ood : ndarray of bool, shape (n_spikes,)
    counts : dict
        ``n_tetrodes``, ``n_spikes``, ``n_reference_marks``, ``n_ood_spikes``.
    """
    times, tetrodes, ood = [], [], []
    n_reference = 0
    for spikes, marks in zip(spike_times, spike_marks, strict=True):
        spikes = np.asarray(spikes, dtype=np.float64)
        in_span = (spikes >= span[0]) & (spikes <= span[1])
        if not in_span.any():
            continue  # as filter_spike_times drops it for the decoder
        spikes = spikes[in_span]
        is_reference = ~interval_mask(spikes, excluded)
        distance = nearest_mark_distance(
            np.asarray(marks)[in_span],
            is_reference,
            spikes,
            own_event_s=T_SPK_OOD_OWN_EVENT_S,
        )
        times.append(spikes)
        tetrodes.append(np.full(spikes.size, len(times) - 1, dtype=np.int64))
        ood.append(distance / WAVEFORM_STD > T_SPK_OOD_NN)
        n_reference += int(is_reference.sum())
    if not times:
        empty = np.zeros(0)
        counts = {
            "n_tetrodes": 0,
            "n_spikes": 0,
            "n_reference_marks": 0,
            "n_ood_spikes": 0,
        }
        return empty, empty.astype(np.int64), empty.astype(bool), counts
    is_ood = np.concatenate(ood)
    counts = {
        "n_tetrodes": len(times),
        "n_spikes": int(is_ood.size),
        "n_reference_marks": n_reference,
        "n_ood_spikes": int(is_ood.sum()),
    }
    return np.concatenate(times), np.concatenate(tetrodes), is_ood, counts


def detect_artifacts(
    time: NDArray[np.float64],
    ca1_lfp: NDArray[np.floating],
    reference_lfp: NDArray[np.floating],
    spike_times: Sequence[NDArray[np.float64]],
    spike_marks: Sequence[NDArray[np.floating]],
    sampling_frequency: float,
    *,
    valid: NDArray[np.bool_] | None = None,
) -> pd.DataFrame:
    """One epoch's artifact cores: rules S, T-ref, T-ca1 and T-spk.

    1. S from the CA1 window spectra.
    2. T-ref and T-ca1 from the reference and CA1 steps.
    3. The OOD reference: per tetrode, the marks outside the S, T-ref and
       T-ca1 cores padded into the mask, so no decode is needed.
    4. T-spk from the spike lines; then all T signatures merge into T cores.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        LFP sample times (seconds), ascending: the span the detector reads.
    ca1_lfp : ndarray, shape (n_time, n_channels)
        The raw referenced CA1 LFP.
    reference_lfp : ndarray, shape (n_time, n_references)
        The raw, unreferenced reference electrodes; no columns when none.
    spike_times, spike_marks : sequence of ndarray
        Per tetrode, the HPC marks after the coincidence filter
        (``get_hpc_marks``); spikes outside ``[time[0], time[-1]]`` are ignored.
    sampling_frequency : float
        The LFP's nominal rate (Hz).
    valid : ndarray of bool, shape (n_time,), optional
        Samples to read; a sample with a non-finite CA1 or reference value is
        invalid too. No feature reads across an invalid sample: a sample whose
        J5 window (``t - 5 .. t + 5``) or L20 window (``t - 24 .. t + 24``)
        holds one has no step value, and a 1-s window holding one has no S
        value. The invalid samples themselves are not masked.

    Returns
    -------
    pd.DataFrame
        Columns ``CORE_COLUMNS``, one row per core, in time order, unpadded;
        no rows for a clean epoch. ``attrs["counts"]`` holds the inputs'
        counts (samples, invalid samples, timestamp gaps, windows, channels,
        references, tetrodes, spikes, reference marks, OOD spikes), whether
        T-ref was disabled for want of a reference electrode, and how many
        samples or bins fired each T signature.

    Raises
    ------
    ValueError
        If `ca1_lfp` or `reference_lfp` is not 2-D with one row per sample of
        `time`.
    """
    time = np.asarray(time, dtype=np.float64)
    n_time = time.size
    ca1 = np.array(ca1_lfp, dtype=np.float32)
    reference = np.array(reference_lfp, dtype=np.float32)
    for name, values in (("ca1_lfp", ca1), ("reference_lfp", reference)):
        if values.ndim != 2 or values.shape[0] != n_time:
            raise ValueError(
                f"{name} has shape {values.shape}; it needs one row per sample of "
                f"time ({n_time}), as (n_time, n_columns)"
            )
    usable = np.isfinite(ca1).all(axis=1) & np.isfinite(reference).all(axis=1)
    if valid is not None:
        usable &= np.asarray(valid, dtype=bool)
    # 0 keeps the filters finite; no feature reads these samples (below)
    ca1[~usable] = 0.0
    reference[~usable] = 0.0
    has_jump = window_valid(usable, J5_REACH)
    has_shift = window_valid(usable, L20_REACH)
    span = (float(time[0]), float(time[-1]))
    n_channels = ca1.shape[1]

    power = window_band_power(
        ca1,
        sampling_frequency,
        window_s=S_WINDOW_S,
        segment_s=S_SEGMENT_S,
        band_hz=S_BAND_HZ,
        valid=usable,
    )
    s = s_cores(
        window_centers(time, sampling_frequency, S_WINDOW_S),
        channel_specificity(power),
        half_window_s=S_WINDOW_S / 2,
    )

    jump = level_jump(ca1)
    robust = robust_jump(jump, has_jump)
    robust[~has_shift] = 0.0  # no L20 there, so no persistent step
    step_count = persistent_step_count(
        jump, robust, level_shift(ca1), threshold=T_CA1_K, persistence=T_CA1_RHO
    )
    del jump, robust
    lfp_events = pd.concat(
        [
            t_ref_events(time, reference_jump(reference, has_jump)),
            t_ca1_events(time, step_count, n_channels),
        ],
        ignore_index=True,
    )

    lfp_cores = pd.concat([s, t_cores(lfp_events)], ignore_index=True)
    excluded = artifact_intervals(lfp_cores, span)
    spikes, tetrode, is_ood, spike_counts = spike_ood(
        spike_times, spike_marks, span, excluded
    )
    lines = spike_lines(
        spikes,
        tetrode,
        is_ood,
        n_tetrodes=max(spike_counts["n_tetrodes"], 1),
        start_time=span[0],
        bin_s=T_SPK_BIN_S,
        min_tetrodes=T_SPK_MIN_TETRODES,
    )
    spk_events = t_spk_events(lines)

    cores = pd.concat(
        [s, t_cores(pd.concat([lfp_events, spk_events], ignore_index=True))],
        ignore_index=True,
    )
    cores = validate_cores(
        cores.sort_values(["core_start", "kind"], kind="stable"), "detected cores"
    )
    cores.attrs["counts"] = {
        "n_samples": int(n_time),
        "n_invalid_samples": int((~usable).sum()),
        "n_time_gaps": int((np.diff(time) > TIME_GAP_STEPS / sampling_frequency).sum()),
        "n_windows": int(power.shape[0]),
        "n_windows_without_power": int(np.isnan(power).any(axis=1).sum()),
        "n_channels": int(n_channels),
        "n_references": int(reference.shape[1]),
        "t_ref_disabled": bool(reference.shape[1] == 0),
        **spike_counts,
        "n_ref_samples": int((lfp_events["signature"] == "ref").sum()),
        "n_ca1_samples": int((lfp_events["signature"] == "ca1").sum()),
        "n_spike_line_bins": int(len(spk_events)),
    }
    return cores
