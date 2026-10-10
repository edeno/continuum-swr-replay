"""Decodes that see one tetrode half, so that the state labels are held out too.

The all-tetrode decode's labels are made with the scoring half C's spikes. Here
a decode sees only half S. Its encoding model is fit on S's tetrodes alone, at
the canonical decode's training bins, and it decodes S's spikes alone, with the
canonical decode's missing bins (its artifact mask, ``is_missing``) set
missing. Both masks are the canonical decode's own: its fitted detector keeps
the training bins, and the caller builds the missing bins as the decoder built
them (``decoding_common.hpc_decode_masks``). Two things come from the
canonical fit, held fixed, with no EM:

- the discrete transition matrix: 4 x 4 with rows summing to 1, so 12 free
  numbers. EM learned 9 of them; the No-Spike row is frozen at its prior.
- the initial conditions, which EM sets to the canonical smoother's posterior
  at the epoch's first bin, over every state bin.

Both were fit with every tetrode, C's included: an aggregate leak, and the only
information from C's spikes in the half decode's labels. The initial conditions
act only at the start of the epoch.

**The canonical encoding is unweighted.** ``estimate_parameters`` refits the
encoding model in every EM iteration, with the smoothed Local probability as
``weights`` (``non_local_detector/models/base.py``, ``_DetectorBase.
estimate_parameters``). The clusterless KDE fit
(``likelihoods/clusterless_kde.py``, ``fit_clusterless_kde_encoding_model``)
takes ``weights`` but never uses it, so every refit is the plain fit at the
training bins. The half encoding is therefore exactly the canonical one
restricted to S: the same per-electrode models, with the ground intensity
summed over S only.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from continuum_swr_replay.analysis.hippocampus.state_evidence.windows import (
    STATES,
    Seed,
    tetrode_halves,
)

HALF_PARTITIONS = 2
"""The first two of Task 12's tetrode partitions, each in both directions."""

DECODE_INPUT_KEYS = (
    "position_time",
    "position",
    "spike_times",
    "spike_waveform_features",
    "is_training",
)

NON_LOCAL_THRESHOLD = 0.8
"""The canonical decode's ``non_local_threshold``."""

PROBABILITY_DIFF_THRESHOLD = 1e-3
"""A bin's state probability "differs" from the reference's above this."""
_OVER = "share_bins_over_1e-3"

OCCUPANCY_MAX_REL = 1e-2
"""The half encoding's occupancy against the canonical one's, as a share of its
maximum. The two are the same KDE of the same training positions, but its
float32 sums over about 600,000 positions differ between the CPU and the
canonical decode's GPU: by 5.1e-4 (peanut) and 1.4e-4 (chimi) of the maximum on
the pilot. Training spikes, marks, mean rates and ground-intensity models must
match exactly, which catches any change of training bins."""

GROUND_MAX_REL = 1e-6
"""The half encoding's summed ground intensity against the sum of the canonical
per-electrode ground intensities over S, elementwise relative: the tolerance of
the unit tests. Both are evaluated on this platform, with the half encoding's
own occupancy; a CPU-fitted half encoding is within 1.5e-7 to 2.2e-7 of the
pilot's GPU-fitted canonical models (peanut, chimi)."""


@dataclass(frozen=True, eq=False)
class Split:
    """One half decode: half S decodes and selects, half C scores.

    Compared by identity (``eq=False``): its fields are arrays, which have no
    single truth value.

    Attributes
    ----------
    partition : int
        The index into :func:`~.windows.tetrode_halves`.
    direction : int
        0: the partition's first half decodes; 1: its second.
    selecting, scoring : ndarray of int
        S's and C's tetrode indices, sorted.
    """

    partition: int
    direction: int
    selecting: NDArray[np.int64]
    scoring: NDArray[np.int64]

    @property
    def name(self) -> str:
        """``p{partition}d{direction}``."""
        return f"p{self.partition}d{self.direction}"


def half_splits(
    n_tetrodes: int, *, seed: Seed, n_partitions: int = HALF_PARTITIONS
) -> list[Split]:
    """The first partitions of :func:`~.windows.tetrode_halves`, both directions.

    ``tetrode_halves`` draws its partitions in sequence, so its first two are
    the same whatever ``n_partitions`` it is asked for: these are the first two
    of Task 12's ten.

    Returns
    -------
    list of Split
        ``p0d0``, ``p0d1``, ``p1d0``, ...: per partition, the first half
        decoding, then the second.
    """
    splits = []
    for partition, (first, second) in enumerate(
        tetrode_halves(n_tetrodes, n_partitions=n_partitions, seed=seed)
    ):
        splits += [
            Split(partition, 0, first, second),
            Split(partition, 1, second, first),
        ]
    return splits


def decode_inputs(detector: Any) -> dict[str, Any]:
    """The canonical decode's own inputs, as its fitted detector keeps them.

    ``estimate_parameters`` keeps its inputs as ``_encoding_model_data`` for the
    encoding refits of EM, and the saved detector still holds them: at its end
    it deletes ``encoding_model_data_``, another name. They are the spikes,
    marks, positions and training mask the decode used, so a half decode trains
    at the canonical training bins without reloading anything. They do not hold
    the decode's missing bins.

    Returns
    -------
    dict
        ``position_time`` (s, shape (n_time,), also the decode's time axis),
        ``position`` (shape (n_time, n_dims)), ``spike_times`` and
        ``spike_waveform_features`` (one array per tetrode) and ``is_training``
        (bool, shape (n_time,)).

    Raises
    ------
    ValueError
        The detector does not hold them, they are inconsistent, or they carry
        encoding-group or environment labels (the canonical decode has neither).
    """
    data = getattr(detector, "_encoding_model_data", None)
    if not isinstance(data, dict) or any(
        data.get(key) is None for key in DECODE_INPUT_KEYS
    ):
        raise ValueError(
            "the detector does not hold its decode inputs (_encoding_model_data, "
            "kept by estimate_parameters); it cannot be decoded with half its "
            "tetrodes at its own training bins"
        )
    if (
        data.get("encoding_group_labels") is not None
        or data.get("environment_labels") is not None
    ):
        raise ValueError("encoding-group and environment labels are not supported")
    time = np.asarray(data["position_time"])
    position = np.asarray(data["position"])
    is_training = np.asarray(data["is_training"])
    if position.shape[0] != time.size or is_training.shape != time.shape:
        raise ValueError(
            "the decode inputs' position, time and mask disagree in length"
        )
    if is_training.dtype != bool:
        raise ValueError("the decode's training mask is not boolean")
    if len(data["spike_times"]) != len(data["spike_waveform_features"]):
        raise ValueError("the decode inputs' spike times and marks disagree")
    return {key: data[key] for key in DECODE_INPUT_KEYS}


def fixed_transition_detector(canonical: Any) -> Any:
    """An unfitted detector with the canonical's settings, its transitions fixed.

    Built from the canonical's constructor parameters (``get_params``,
    deep-copied, so fitting it touches nothing of the canonical's), with
    ``discrete_transition_type`` set to the canonical fit's matrix. sklearn's
    ``clone`` cannot build it: EM stores the first bin's float32 state
    probabilities as ``discrete_initial_conditions`` (summing to 1.000056 in a
    pilot epoch), which the constructor refuses, so they are normalized here.
    They only seed ``initial_conditions_``, which :func:`half_tetrode_decode`
    replaces with the canonical fit's after fitting.
    """
    transitions = np.array(canonical.discrete_state_transitions_, dtype=float)
    if transitions.ndim != 2:
        raise ValueError("only stationary discrete transitions can be fixed")
    from non_local_detector.discrete_state_transitions import DiscreteStationaryCustom

    params = copy.deepcopy(canonical.get_params(deep=False))
    params["discrete_transition_type"] = DiscreteStationaryCustom(values=transitions)
    initial = np.asarray(params["discrete_initial_conditions"], dtype=float)
    params["discrete_initial_conditions"] = initial / initial.sum()
    return type(canonical)(**params)


def _fix_initial_conditions(detector: Any, canonical: Any) -> None:
    """Give a fitted half detector the canonical fit's initial conditions.

    Raises
    ------
    ValueError
        Its state bins, continuous transitions or discrete transitions are not
        the canonical's, so the two decodes would not share a state space.
    """
    for name in ("state_ind_", "is_track_interior_state_bins_"):
        if not np.array_equal(getattr(detector, name), getattr(canonical, name)):
            raise ValueError(f"the half detector's {name} differs from the canonical's")
    if not np.allclose(
        detector.continuous_state_transitions_,
        canonical.continuous_state_transitions_,
        rtol=1e-6,
        atol=1e-12,
    ):
        raise ValueError(
            "the half detector's continuous transitions differ from the canonical's"
        )
    if not np.array_equal(
        detector.discrete_state_transitions_, canonical.discrete_state_transitions_
    ):
        raise ValueError("the half detector's discrete transitions are not fixed")
    detector.initial_conditions_ = np.array(canonical.initial_conditions_, copy=True)


def half_tetrode_decode(
    canonical: Any,
    tetrodes: NDArray,
    *,
    is_missing: NDArray[np.bool_],
    n_bins: int | None = None,
) -> tuple[xr.Dataset, Any]:
    """Decode with ``tetrodes`` only, the canonical fit's transitions held fixed.

    1. A detector with the canonical's settings and its fitted discrete
       transitions (:func:`fixed_transition_detector`).
    2. ``fit`` on the tetrodes' spikes and marks, at the canonical training
       bins (:func:`decode_inputs`): the encoding model sees no other tetrode.
    3. The canonical fit's initial conditions.
    4. ``predict`` with the tetrodes' spikes and ``is_missing``: one
       forward-backward pass, no EM.

    Parameters
    ----------
    canonical : NonLocalClusterlessDetector
        The canonical decode's fitted detector.
    tetrodes : ndarray of int
        Half S's tetrode indices.
    is_missing : ndarray of bool, shape (n_time,)
        The canonical decode's missing bins on its time axis
        (``decoding_common.hpc_decode_masks``' ``is_missing``), whose
        log-likelihood the decode sets to 0.
    n_bins : int, optional
        Decode only the first ``n_bins`` bins (a smoke test); default all.

    Returns
    -------
    results : xr.Dataset
        The library's ``predict`` output: ``acausal_posterior`` and
        ``acausal_state_probabilities`` on the decode's time axis.
    detector : NonLocalClusterlessDetector
        The fitted half detector.
    """
    inputs = decode_inputs(canonical)
    n_tetrodes = len(inputs["spike_times"])
    tetrodes = np.asarray(tetrodes, dtype=np.int64)
    if (
        tetrodes.ndim != 1
        or tetrodes.size == 0
        or np.unique(tetrodes).size != tetrodes.size
        or tetrodes.min() < 0
        or tetrodes.max() >= n_tetrodes
    ):
        raise ValueError(f"tetrodes must be distinct indices below {n_tetrodes}")
    spikes = [np.asarray(inputs["spike_times"][i]) for i in tetrodes]
    marks = [np.asarray(inputs["spike_waveform_features"][i]) for i in tetrodes]
    time = np.asarray(inputs["position_time"])
    position = np.asarray(inputs["position"])
    stop = time.size if n_bins is None else int(n_bins)
    if not 2 <= stop <= time.size:
        raise ValueError(f"n_bins must be between 2 and {time.size}")
    is_missing = np.asarray(is_missing)
    if is_missing.dtype != bool or is_missing.shape != time.shape:
        raise ValueError(
            f"is_missing must be boolean, one per bin of the decode ({time.size}); "
            f"got {is_missing.dtype} of shape {is_missing.shape}"
        )

    detector = fixed_transition_detector(canonical)
    detector.fit(time, position, spikes, marks, is_training=inputs["is_training"])
    _fix_initial_conditions(detector, canonical)
    results = detector.predict(
        spike_times=spikes,
        spike_waveform_features=marks,
        time=time[:stop],
        position=position[:stop],
        position_time=time[:stop],
        is_missing=is_missing[:stop],
    )
    return results, detector


def _ground_intensity(
    rate: float, model: Any, centers: NDArray, occupancy: NDArray
) -> NDArray:
    """One electrode's ground intensity per bin, as the clusterless KDE fit sums it."""
    from non_local_detector.likelihoods.common import EPS

    density = np.asarray(model.predict(centers), dtype=float)
    positive = occupancy > 0.0
    ratio = np.where(positive, density / np.where(positive, occupancy, 1.0), EPS)
    return np.clip(rate * ratio, EPS, None)


def _same_kde(first: Any, second: Any) -> bool:
    """Whether two fitted ``KDEModel`` hold the same samples, weights and settings."""
    if type(first) is not type(second) or set(vars(first)) != set(vars(second)):
        return False
    return all(
        np.array_equal(np.asarray(value), np.asarray(vars(second)[name]))
        for name, value in vars(first).items()
    )


def encoding_subset_check(
    half: dict, canonical: dict, tetrodes: NDArray
) -> dict[str, Any]:
    """Whether a half encoding is the canonical encoding restricted to ``tetrodes``.

    The canonical encoding is unweighted (see the module notes), so the half
    encoding fit at the same training bins must hold the canonical one's
    per-electrode models for ``tetrodes`` and nothing of the other tetrodes.

    Parameters
    ----------
    half, canonical : dict
        Clusterless KDE encoding models (``encoding_model_[ENCODING_KEY]``).
    tetrodes : ndarray of int
        Half S's tetrode indices into the canonical encoding.

    Returns
    -------
    dict
        - :func:`~.terms.encoding_comparison`'s: the training spikes' counts,
          positions and marks and the mean rates against the canonical ones of
          ``tetrodes``, which must match exactly;
        - ``same_keys``: the two encodings hold the same entries;
        - ``same_ground_models``: each electrode's ground-intensity KDE
          (``gpi_models``) is the canonical one's, samples and settings;
        - ``max_rel_occupancy`` (and ``max_abs_occupancy``): against the
          canonical occupancy, as a share of its maximum, at most
          :data:`OCCUPANCY_MAX_REL`;
        - ``max_rel_ground``: the summed ground intensity against the sum of
          the canonical per-electrode ground intensities over ``tetrodes``
          alone, elementwise, at most :data:`GROUND_MAX_REL`;
        - ``n_electrodes`` and ``same_as_canonical_subset``, all of the above.
    """
    from continuum_swr_replay.analysis.hippocampus.state_evidence.terms import (
        encoding_comparison,
    )

    tetrodes = np.asarray(tetrodes, dtype=np.int64)
    subset = {
        key: [canonical[key][i] for i in tetrodes]
        for key in (
            "encoding_positions",
            "encoding_spike_waveform_features",
            "mean_rates",
        )
    }
    check: dict[str, Any] = {
        "n_electrodes": int(tetrodes.size),
        **encoding_comparison(half, subset),
        "same_keys": set(half) == set(canonical),
    }
    check["same_ground_models"] = len(half["gpi_models"]) == tetrodes.size and all(
        _same_kde(model, canonical["gpi_models"][i])
        for model, i in zip(half["gpi_models"], tetrodes, strict=False)
    )
    occupancy = np.asarray(canonical["occupancy"], dtype=float)
    check["max_abs_occupancy"] = float(
        np.max(np.abs(np.asarray(half["occupancy"], dtype=float) - occupancy))
    )
    check["max_rel_occupancy"] = check["max_abs_occupancy"] / float(
        np.max(np.abs(occupancy))
    )
    environment = canonical["environment"]
    interior = np.asarray(environment.is_track_interior_).ravel()
    centers = np.asarray(environment.place_bin_centers_)[interior]
    own_occupancy = np.asarray(half["occupancy"], dtype=float)
    expected = np.sum(
        [
            _ground_intensity(
                canonical["mean_rates"][i],
                canonical["gpi_models"][i],
                centers,
                own_occupancy,
            )
            for i in tetrodes
        ],
        axis=0,
    )
    ground = np.asarray(half["summed_ground_process_intensity"], dtype=float)
    check["max_rel_ground"] = (
        float(np.max(np.abs(ground - expected) / np.abs(expected)))
        if ground.shape == expected.shape
        else float("inf")
    )
    check["same_as_canonical_subset"] = bool(
        check["same_n_encoding_spikes"]
        and check["max_abs_encoding_positions"] == 0.0
        and check["max_abs_encoding_marks"] == 0.0
        and check["max_abs_mean_rates"] == 0.0
        and check["same_keys"]
        and check["same_ground_models"]
        and check["max_rel_occupancy"] <= OCCUPANCY_MAX_REL
        and check["max_rel_ground"] <= GROUND_MAX_REL
    )
    return check


def non_local_event_labels(
    probabilities: NDArray,
    speed: NDArray,
    time: NDArray,
    *,
    threshold: float = NON_LOCAL_THRESHOLD,
) -> NDArray[np.int64]:
    """Non-local event labels, made as the canonical decode makes them.

    ``get_data_over_time``'s steps: runs of P(Non-Local Continuous) +
    P(Non-Local Fragmented) above ``threshold`` lasting at least 15 ms
    (``get_confident_times``), kept when the animal is immobile throughout
    (``build_speed_eligible_event_cohort``) and numbered from 1. The two
    probabilities are summed in their own dtype, as the pipeline does
    (float32 from the decoder).

    Parameters
    ----------
    probabilities : ndarray, shape (n_time, 4)
        State probabilities, in :data:`~.windows.STATES` order.
    speed : ndarray, shape (n_time,)
        Head speed, cm/s.
    time : ndarray, shape (n_time,)
        s.

    Returns
    -------
    ndarray of int, shape (n_time,)
        0 outside events, else the event's number.
    """
    from continuum_swr_replay.analysis.hippocampus.event_detection import (
        get_confident_times,
    )
    from continuum_swr_replay.analysis.hippocampus.event_immobility import (
        build_speed_eligible_event_cohort,
    )

    probabilities = np.asarray(probabilities)
    time = np.asarray(time, dtype=float)
    speed = np.asarray(speed, dtype=float)
    non_local = xr.DataArray(
        probabilities[:, 2] + probabilities[:, 3], dims="time", coords={"time": time}
    )
    detected = get_confident_times(non_local, speed, time, threshold)
    _, labels = build_speed_eligible_event_cohort(
        detected,
        speed=speed,
        time=pd.Index(time, name="time"),
        event_col="non_local_event",
        preserve_source_ids=False,
    )
    return np.asarray(labels, dtype=np.int64)


def _event_bounds(labels: NDArray) -> set[tuple[int, int]]:
    """Each event's first and last bin.

    Parameters
    ----------
    labels : ndarray of int, shape (n_time,)
        0 outside events, else the event's number (:func:`non_local_event_labels`).

    Returns
    -------
    set of (int, int)
        One (first bin, last bin) pair per event.
    """
    bins = np.flatnonzero(labels > 0)
    if bins.size == 0:
        return set()
    bounds = pd.Series(bins).groupby(labels[bins]).agg(["min", "max"])
    return set(zip(bounds["min"].tolist(), bounds["max"].tolist(), strict=True))


def label_agreement(
    reference_probabilities: NDArray,
    reference_labels: NDArray,
    probabilities: NDArray,
    labels: NDArray,
) -> dict[str, Any]:
    """How far one decode's states and events are from a reference decode's.

    Parameters
    ----------
    reference_probabilities, probabilities : ndarray, shape (n_time, 4)
        On the same bins, in :data:`~.windows.STATES` order.
    reference_labels, labels : ndarray of int, shape (n_time,)
        Their non-local event labels (:func:`non_local_event_labels`).

    Returns
    -------
    dict
        - ``n_bins``; ``max_abs_diff``, the largest |P - P_ref| over states;
        - ``state``: per state, ``max_abs_diff``, ``mean_abs_diff``,
          ``p999_abs_diff`` and ``share_bins_over_1e-3``;
        - events: ``events_reference`` and ``events_other``;
          ``events_identical`` (the same first and last bin);
          ``reference_events_shared`` and ``other_events_shared`` (overlapping
          an event of the other decode in at least one bin); and
          ``event_bin_jaccard`` (bins in both decodes' events over bins in
          either's; None when neither has any).
    """
    reference = np.asarray(reference_probabilities, dtype=float)
    other = np.asarray(probabilities, dtype=float)
    reference_labels = np.asarray(reference_labels)
    labels = np.asarray(labels)
    if other.shape != reference.shape or reference.shape[1:] != (len(STATES),):
        raise ValueError("the two decodes' probabilities must be on the same bins")
    if labels.shape != reference_labels.shape or labels.size != reference.shape[0]:
        raise ValueError("the event labels must be on the probabilities' bins")
    diff = np.abs(other - reference)
    both = (reference_labels > 0) & (labels > 0)
    either = int(((reference_labels > 0) | (labels > 0)).sum())
    return {
        "n_bins": int(reference.shape[0]),
        "max_abs_diff": float(diff.max()),
        "state": {
            state: {
                "max_abs_diff": float(diff[:, j].max()),
                "mean_abs_diff": float(diff[:, j].mean()),
                "p999_abs_diff": float(np.quantile(diff[:, j], 0.999)),
                _OVER: float((diff[:, j] > PROBABILITY_DIFF_THRESHOLD).mean()),
            }
            for j, state in enumerate(STATES)
        },
        "events_reference": int(np.unique(reference_labels[reference_labels > 0]).size),
        "events_other": int(np.unique(labels[labels > 0]).size),
        "events_identical": len(
            _event_bounds(reference_labels) & _event_bounds(labels)
        ),
        "reference_events_shared": int(np.unique(reference_labels[both]).size),
        "other_events_shared": int(np.unique(labels[both]).size),
        "event_bin_jaccard": float(both.sum() / either) if either else None,
    }
