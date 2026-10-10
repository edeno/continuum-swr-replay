"""Held-out state labels: decodes that see one tetrode half (Task 13).

A canonical decode is simulated with the project's own NonLocal detector and
transition settings: ``estimate_parameters`` with EM, on place-tuned spikes over
a Y track with two remote bursts and a silence, saved under a made-up epoch id as
the decode script saves it. The half decodes refit nothing but the half's
encoding and keep the canonical fit's transitions and initial conditions. The
database loader is replaced by the simulation's own inputs, so nothing here
touches Spyglass.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from non_local_detector import NonLocalClusterlessDetector
from non_local_detector.discrete_state_transitions import DiscreteStationaryCustom
from non_local_detector.environment import Environment
from non_local_detector.models import base as library_base
from track_linearization import get_linearized_position

from continuum_swr_replay.analysis.hippocampus.artifacts.masks import (
    artifact_intervals,
    interval_mask,
)
from continuum_swr_replay.analysis.hippocampus.state_evidence import (
    ENCODING_KEY,
    STATES,
    bout_statistics,
    decode_inputs,
    encoding_subset_check,
    half_splits,
    half_tetrode_decode,
    label_agreement,
    non_local_event_labels,
    paired_bouts,
    tetrode_halves,
    window_evidence,
)
from continuum_swr_replay.decoding_transitions import nonlocal_transition_settings
from continuum_swr_replay.parameters import LOCAL_POSITION_STD
from continuum_swr_replay.table_io import read_table
from scripts.qa import run_half_tetrode_decodes as runner
from scripts.qa import run_state_evidence
from tests.unit._memoryless_clusterless_epoch import _y_track
from tests.unit._swr_inventory_fixtures import (
    ARTIFACT_RUN_ID,
    RUN_ID,
    artifact_cores,
    use_swr_source,
    write_artifact_run,
    write_swr_inventory,
)
from tests.unit.test_state_evidence import (
    NO_SPIKE_RATE,
    linear_distance,
    overlapping_windows,
    random_terms,
)

pytestmark = [
    # JAX on the CPU cannot donate buffers to the library's filter; harmless
    pytest.mark.filterwarnings(
        "ignore:Some donated buffers were not usable:UserWarning"
    ),
    # in the full suite, analysis/replay/dirichlet_multinomial.py has turned on
    # JAX's x64 mode at import, so the library scatters float64 likelihoods
    # into its float32 array; a decode never imports that module
    pytest.mark.filterwarnings(
        "ignore:scatter inputs have incompatible types:FutureWarning"
    ),
]

EPOCH = "aa20200101_02_r1_HPC"
FS = 500.0
N_TIME = 6000
EDGE_ORDER = [(0, 3), (3, 1), (3, 2)]
FIELD_CENTERS_CM = [3.0, 12.0, 21.0, 30.0, 39.0, 46.0]
CLUSTERLESS_PARAMS = {
    "position_std": 3.5,
    "waveform_std": 24.0,
    "block_size": 100,
    "disable_progress_bar": True,
}
SELECTING = np.array([0, 2, 5])
BURSTS = [(300, 330, 3.0), (1100, 1130, 46.0), (3300, 3330, 3.0)]
"""Bins (first, last + 1) and replayed place (cm) of the remote bursts."""
ARTIFACT_CORES = [("spk", 1000.0 + 1200 / FS, 1000.0 + 1200 / FS)]
"""A spike-only line 140 ms after the second burst: the mask pads it by 0.1 s
(about bins 1150-1250), and excluded time, 0.3 s wider, holds the burst."""
PROVENANCE = {"commit": "c" * 40, "dirty": False}
"""A clean tree's provenance, so that decodes written in a test can be kept."""
REPO = Path(__file__).resolve().parents[2]


def environment() -> Environment:
    return Environment(
        place_bin_size=4.0,
        track_graph=_y_track(),
        edge_order=EDGE_ORDER,
        edge_spacing=8.0,
    )


@dataclass
class Canonical:
    """A simulated canonical decode: its fitted detector, results and inputs."""

    detector: Any
    results: xr.Dataset
    time: np.ndarray
    position: np.ndarray
    spike_times: list[np.ndarray]
    marks: list[np.ndarray]
    is_training: np.ndarray
    is_missing: np.ndarray
    speed: np.ndarray
    linear: np.ndarray


def simulate_canonical() -> Canonical:
    """Place-tuned spikes, two remote bursts and a silence, decoded with EM.

    The animal shuttles along the bottom of the Y and pauses at both ends. While
    it pauses at one end, a 60 ms burst replays the other end; the bursts are
    left out of the training bins, as the Kay inventory's events are. An
    artifact mask's bins are missing and out of the training bins, as
    ``hpc_decode_masks`` makes them for a canonical decode.
    """
    rng = np.random.default_rng(0)
    time = 1000.0 + np.arange(N_TIME) / FS
    phase = np.clip(1.6 * np.sin(2 * np.pi * np.arange(N_TIME) / FS / 3.0), -1, 1)
    position = np.c_[20.0 + 18.0 * phase, np.zeros(N_TIME)]
    speed = np.abs(np.gradient(position[:, 0], 1 / FS))
    linear = get_linearized_position(
        position, _y_track(), edge_order=EDGE_ORDER, edge_spacing=8.0
    ).linear_position.to_numpy()
    bursts = BURSTS
    silence = slice(5500, 5600)  # while paused
    spike_times, marks = [], []
    for k, center in enumerate(FIELD_CENTERS_CM):
        rate = 60.0 * np.exp(-0.5 * ((linear - center) / 5.0) ** 2) + 1.0
        for s, e, content in bursts:
            rate[s:e] = 300.0 * np.exp(-0.5 * ((content - center) / 5.0) ** 2) + 1.0
        rate[silence] = 0.0
        sample = np.repeat(np.arange(N_TIME), rng.poisson(rate / FS))
        spike_times.append(np.sort(time[sample] + rng.uniform(0, 1 / FS, sample.size)))
        marks.append(rng.normal(100.0 + 30.0 * k, 10.0, (sample.size, 2)))
    is_training = np.ones(N_TIME, dtype=bool)
    for s, e, _ in bursts:
        is_training[s - 10 : e + 10] = False
    is_missing = interval_mask(
        time, artifact_intervals(artifact_cores(ARTIFACT_CORES), (0.0, 1e6))
    )
    is_training &= ~is_missing

    detector = NonLocalClusterlessDetector(
        environments=environment(),
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params=dict(CLUSTERLESS_PARAMS),
        local_position_std=LOCAL_POSITION_STD,
        **nonlocal_transition_settings(FS),
    )
    results = detector.estimate_parameters(
        position_time=time,
        position=position,
        spike_times=spike_times,
        spike_waveform_features=marks,
        time=time,
        is_training=is_training,
        is_missing=is_missing,
        discrete_transition_covariate_data=None,
        return_outputs="log_likelihood",
        max_iter=2,
    )
    return Canonical(
        detector,
        results,
        time,
        position,
        spike_times,
        marks,
        is_training,
        is_missing,
        speed,
        linear,
    )


@pytest.fixture(scope="module")
def canonical() -> Canonical:
    return simulate_canonical()


def saved_decode_masks(canonical: Canonical, **changes) -> dict:
    """The ``decode_masks`` a decode made with the simulation's masks stamps."""
    return {
        "swr_source": f"kay@{RUN_ID}+artifacts@{ARTIFACT_RUN_ID}",
        "swr_inventory_run_id": RUN_ID,
        "artifact_run_id": ARTIFACT_RUN_ID,
        "training_type": "no_ripple",
        "n_bins": N_TIME,
        "n_missing_bins": int(canonical.is_missing.sum()),
        "n_encoding_removed_bins": int((~canonical.is_training).sum()),
        **changes,
    }


def write_canonical_decode(
    canonical: Canonical, directory: Path, decode_masks: dict | None = None
) -> None:
    """The decode's three files, written as ``run_decoding_bandit`` writes them,
    with its ``decode_masks`` (``saved_decode_masks`` by default)."""
    directory.mkdir(parents=True, exist_ok=True)
    canonical.detector.save_model(str(directory / f"{EPOCH}_detector.pkl"))
    saved = canonical.results.reset_index("state_bins")
    saved.attrs = {
        **saved.attrs,
        "decode_masks": json.dumps(
            saved_decode_masks(canonical) if decode_masks is None else decode_masks,
            sort_keys=True,
        ),
    }
    saved.to_netcdf(directory / f"{EPOCH}_results.nc", engine="h5netcdf")
    probabilities = canonical.results.acausal_state_probabilities.values
    dot = pd.DataFrame(
        {
            "animal": "aa",
            "day": 20200101,
            "epoch": "02_r1",
            "MUA_event": 0,
            "non_local_event": non_local_event_labels(
                probabilities, canonical.speed, canonical.time
            ),
            **{
                column: probabilities[:, j]
                for j, column in enumerate(run_state_evidence.PROBABILITY_COLUMNS)
            },
            "head_position_x": canonical.position[:, 0],
            "head_position_y": canonical.position[:, 1],
            "head_speed": canonical.speed,
            "linear_position": canonical.linear,
        },
        index=pd.Index(canonical.time, name="time"),
    )
    dot.to_pickle(directory / f"{EPOCH}_80_data_over_time.pkl")


def stand_in_loader(canonical: Canonical) -> dict:
    """What ``load_marks_and_training_mask`` gives for the simulated epoch."""
    return {
        "time": canonical.time,
        "spike_times": canonical.spike_times,
        "marks": canonical.marks,
        "is_training": canonical.is_training,
        "is_missing": canonical.is_missing,
    }


@pytest.fixture
def source(tmp_path, monkeypatch):
    """The SWR source the simulated decode was made under: the bursts' training
    gaps are its inventory's events, and ARTIFACT_CORES its artifact run's."""
    time = 1000.0 + np.arange(N_TIME) / FS
    events = [(time[s - 10], time[e + 9]) for s, e, _ in BURSTS]
    inventory = write_swr_inventory(
        tmp_path / "swr", {EPOCH: events}, artifact_run_id=ARTIFACT_RUN_ID
    )
    artifacts = write_artifact_run(tmp_path / "artifacts", {EPOCH: ARTIFACT_CORES})
    use_swr_source(monkeypatch, inventory, artifacts)


# --- the library's EM and the canonical encoding ------------------------------------------
def test_em_refits_the_clusterless_encoding_without_its_weights(canonical):
    """The canonical encoding is the plain fit at the training bins.

    Each EM iteration refits the encoding with the Local posterior as weights,
    but the clusterless KDE fit takes ``weights`` and never uses it, so EM ends
    where the unweighted fit began.
    """
    plain = NonLocalClusterlessDetector(
        environments=environment(),
        clusterless_algorithm_params=dict(CLUSTERLESS_PARAMS),
        local_position_std=LOCAL_POSITION_STD,
    )
    plain.fit(
        canonical.time,
        canonical.position,
        canonical.spike_times,
        canonical.marks,
        is_training=canonical.is_training,
    )
    after_em = canonical.detector.encoding_model_[ENCODING_KEY]
    fit = plain.encoding_model_[ENCODING_KEY]

    # EM ran and moved the transitions, so it did run its M-steps
    assert len(canonical.results.attrs["marginal_log_likelihoods"]) == 3
    for key in ("encoding_positions", "encoding_spike_waveform_features"):
        for refit, plain_fit in zip(after_em[key], fit[key], strict=True):
            np.testing.assert_array_equal(np.asarray(refit), np.asarray(plain_fit))
    for key in ("mean_rates", "occupancy", "summed_ground_process_intensity"):
        np.testing.assert_array_equal(np.asarray(after_em[key]), np.asarray(fit[key]))


# --- the splits and the decode's inputs --------------------------------------------------
def test_half_splits_are_the_first_two_partitions_in_both_directions():
    partitions = tetrode_halves(7, seed=20261003)  # Task 12's ten

    splits = half_splits(7, seed=20261003)

    assert [s.name for s in splits] == ["p0d0", "p0d1", "p1d0", "p1d1"]
    for split in splits:
        first, second = partitions[split.partition]
        selecting, scoring = (
            (first, second) if split.direction == 0 else (second, first)
        )
        np.testing.assert_array_equal(split.selecting, selecting)
        np.testing.assert_array_equal(split.scoring, scoring)
        assert sorted([*split.selecting, *split.scoring]) == list(range(7))


def test_decode_inputs_are_the_ones_the_canonical_fit_kept(canonical):
    inputs = decode_inputs(canonical.detector)

    np.testing.assert_array_equal(inputs["position_time"], canonical.time)
    np.testing.assert_array_equal(inputs["is_training"], canonical.is_training)
    assert len(inputs["spike_times"]) == len(FIELD_CENTERS_CM)
    stripped = copy.copy(canonical.detector)
    del stripped._encoding_model_data
    with pytest.raises(ValueError, match="does not hold its decode inputs"):
        decode_inputs(stripped)


# --- the half decode ----------------------------------------------------------------------
def test_the_half_encoding_holds_only_the_selecting_halfs_electrodes(canonical):
    _, half = half_tetrode_decode(
        canonical.detector, SELECTING, is_missing=canonical.is_missing
    )

    encoding = half.encoding_model_[ENCODING_KEY]
    stored = canonical.detector.encoding_model_[ENCODING_KEY]
    assert len(encoding["encoding_positions"]) == SELECTING.size
    assert len(encoding["mean_rates"]) == len(encoding["gpi_models"]) == SELECTING.size
    # each electrode's model is the canonical one's, fit on the same bins
    for j, i in enumerate(SELECTING):
        np.testing.assert_array_equal(
            encoding["encoding_positions"][j], stored["encoding_positions"][i]
        )
        np.testing.assert_array_equal(
            encoding["encoding_spike_waveform_features"][j],
            stored["encoding_spike_waveform_features"][i],
        )
        assert encoding["mean_rates"][j] == stored["mean_rates"][i]
    np.testing.assert_array_equal(encoding["occupancy"], stored["occupancy"])
    # and its ground intensity sums S's electrodes only
    interior = np.asarray(stored["environment"].is_track_interior_).ravel()
    centers = np.asarray(stored["environment"].place_bin_centers_)[interior]
    occupancy = np.asarray(stored["occupancy"])
    per_electrode = [
        np.clip(
            rate * np.asarray(model.predict(centers)) / occupancy,
            a_min=1e-15,
            a_max=None,
        )
        for rate, model in zip(stored["mean_rates"], stored["gpi_models"], strict=True)
    ]
    np.testing.assert_allclose(
        np.asarray(encoding["summed_ground_process_intensity"]),
        np.sum([per_electrode[i] for i in SELECTING], axis=0),
        rtol=1e-6,
    )
    # the runtime check says the same
    check = encoding_subset_check(encoding, stored, SELECTING)
    assert check["same_as_canonical_subset"]
    assert check["same_keys"] and check["same_ground_models"]
    assert check["max_rel_occupancy"] == 0.0 and check["max_rel_ground"] <= 1e-6


def test_the_half_decode_reads_nothing_of_the_scoring_half(canonical):
    """C's spikes, marks and encoding may change: the half decode does not."""
    expected, _ = half_tetrode_decode(
        canonical.detector, SELECTING, is_missing=canonical.is_missing
    )
    altered = copy.deepcopy(canonical.detector)
    scoring = np.setdiff1d(np.arange(len(FIELD_CENTERS_CM)), SELECTING)
    data = altered._encoding_model_data
    stored = altered.encoding_model_[ENCODING_KEY]
    for i in scoring:
        data["spike_times"][i] = data["spike_times"][i][::2] + 0.05
        data["spike_waveform_features"][i] = (
            data["spike_waveform_features"][i][::2] + 50
        )
        stored["encoding_positions"][i] = stored["encoding_positions"][i] + 10.0

    results, _ = half_tetrode_decode(
        altered, SELECTING, is_missing=canonical.is_missing
    )

    np.testing.assert_array_equal(
        results.acausal_state_probabilities.values,
        expected.acausal_state_probabilities.values,
    )


def test_no_em_iteration_runs(canonical, monkeypatch):
    calls: Counter = Counter()

    def refuse(*_args, **_kwargs):
        raise AssertionError("EM ran")

    def counted(name, method):
        def wrapper(self, *args, **kwargs):
            calls[name] += 1
            return method(self, *args, **kwargs)

        return wrapper

    monkeypatch.setattr(library_base._DetectorBase, "estimate_parameters", refuse)
    monkeypatch.setattr(library_base.ClusterlessDetector, "estimate_parameters", refuse)
    monkeypatch.setattr(library_base, "_estimate_discrete_transition", refuse)
    for cls, name in [
        (library_base.ClusterlessDetector, "fit_encoding_model"),
        (library_base._DetectorBase, "_predict"),
    ]:
        monkeypatch.setattr(cls, name, counted(name, getattr(cls, name)))

    results, half = half_tetrode_decode(
        canonical.detector, SELECTING, is_missing=canonical.is_missing
    )

    assert calls == {"fit_encoding_model": 1, "_predict": 1}
    np.testing.assert_array_equal(
        half.discrete_state_transitions_, canonical.detector.discrete_state_transitions_
    )
    np.testing.assert_array_equal(
        half.initial_conditions_, canonical.detector.initial_conditions_
    )
    assert np.ndim(results.attrs["marginal_log_likelihoods"]) == 0  # one pass


def test_the_fixed_transition_decode_equals_the_librarys_decode_with_the_same_transitions(
    canonical,
):
    fitted = canonical.detector.discrete_state_transitions_
    # EM moved the transitions, so its starting matrix would be another decode
    assert not np.allclose(fitted, canonical.detector.discrete_transition_type.values)
    results, _ = half_tetrode_decode(
        canonical.detector, SELECTING, is_missing=canonical.is_missing
    )

    spikes = [canonical.spike_times[i] for i in SELECTING]
    marks = [canonical.marks[i] for i in SELECTING]
    library = NonLocalClusterlessDetector(
        environments=environment(),
        clusterless_algorithm_params=dict(CLUSTERLESS_PARAMS),
        local_position_std=LOCAL_POSITION_STD,
        discrete_transition_type=DiscreteStationaryCustom(values=fitted),
    )
    library.fit(
        canonical.time, canonical.position, spikes, marks,
        is_training=canonical.is_training,
    )  # fmt: skip
    # the fit's initial conditions: the library takes them only as an attribute
    library.initial_conditions_ = canonical.detector.initial_conditions_
    expected = library.predict(
        spikes, marks, time=canonical.time, position=canonical.position,
        position_time=canonical.time, is_missing=canonical.is_missing,
    )  # fmt: skip

    for name in ("acausal_state_probabilities", "acausal_posterior"):
        np.testing.assert_allclose(
            results[name].values, expected[name].values, rtol=0, atol=1e-7
        )
    # the starting matrix decodes differently, so the comparison can tell them apart
    library.discrete_state_transitions_ = (
        canonical.detector.discrete_transition_type.values
    )
    start = library.predict(
        spikes, marks, time=canonical.time, position=canonical.position,
        position_time=canonical.time, is_missing=canonical.is_missing,
    )  # fmt: skip
    assert (
        np.abs(
            start.acausal_state_probabilities.values
            - results.acausal_state_probabilities.values
        ).max()
        > 1e-4
    )


def test_with_every_tetrode_the_half_decode_reproduces_the_canonical_decode(canonical):
    every = np.arange(len(FIELD_CENTERS_CM))

    results, _ = half_tetrode_decode(
        canonical.detector, every, is_missing=canonical.is_missing
    )

    np.testing.assert_allclose(
        results.acausal_state_probabilities.values,
        canonical.results.acausal_state_probabilities.values,
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        results.acausal_posterior.values,
        canonical.results.acausal_posterior.values,
        rtol=0,
        atol=1e-6,
    )


def test_the_missing_bins_must_be_on_the_decodes_axis(canonical):
    with pytest.raises(ValueError, match="is_missing"):
        half_tetrode_decode(
            canonical.detector, SELECTING, is_missing=canonical.is_missing[:-1]
        )
    with pytest.raises(ValueError, match="is_missing"):
        half_tetrode_decode(
            canonical.detector, SELECTING, is_missing=canonical.is_missing.astype(int)
        )


def test_a_decode_of_the_first_bins_only(canonical):
    full, _ = half_tetrode_decode(
        canonical.detector, SELECTING, is_missing=canonical.is_missing
    )

    first, _ = half_tetrode_decode(
        canonical.detector, SELECTING, is_missing=canonical.is_missing, n_bins=3000
    )

    assert first.time.size == 3000
    np.testing.assert_array_equal(first.time.values, canonical.time[:3000])
    # the causal pass is the same; the smoother differs only near the cut
    np.testing.assert_allclose(
        first.acausal_state_probabilities.values[:2000],
        full.acausal_state_probabilities.values[:2000],
        atol=1e-5,
    )


def test_the_simulated_decode_has_confident_bouts_of_several_states(canonical):
    """So the end-to-end test exercises more than the Local branch."""
    p = canonical.results.acausal_state_probabilities.values
    assert (p[:, 0] > 0.8).sum() > 1000  # Local
    assert (p[:, 1] > 0.8).sum() >= 20  # No-Spike, in the silence
    assert (p[:, 2:].sum(1) > 0.8).sum() >= 10  # non-local, in the bursts


# --- event labels and their agreement -------------------------------------------------------
def test_non_local_event_labels_keep_confident_immobile_runs():
    n = 400
    time = np.arange(n) / FS
    probabilities = np.tile([0.97, 0.01, 0.01, 0.01], (n, 1))
    probabilities[100:130] = [0.05, 0.05, 0.85, 0.05]  # 60 ms, immobile
    probabilities[200:204] = [0.05, 0.05, 0.05, 0.85]  # 8 ms: too short
    probabilities[300:330] = [0.05, 0.05, 0.45, 0.45]  # 60 ms, running
    speed = np.zeros(n)
    speed[290:340] = 30.0

    labels = non_local_event_labels(probabilities, speed, time)

    assert np.unique(labels).tolist() == [0, 1]
    assert np.flatnonzero(labels == 1).tolist() == list(range(100, 130))


def test_label_agreement_counts_shared_and_identical_events():
    reference = np.array([0, 1, 1, 1, 0, 0, 2, 2, 0, 3, 3, 0, 0])
    other = np.array([0, 1, 1, 1, 0, 2, 2, 0, 0, 0, 0, 0, 3])
    p_ref = np.tile([1.0, 0.0, 0.0, 0.0], (13, 1))
    p_other = p_ref.copy()
    p_other[5] = [0.5, 0.5, 0.0, 0.0]

    out = label_agreement(p_ref, reference, p_other, other)

    assert out["events_reference"] == 3 and out["events_other"] == 3
    assert out["events_identical"] == 1  # bins 1-3
    assert out["reference_events_shared"] == 2  # 1 and 2
    assert out["other_events_shared"] == 2
    assert out["event_bin_jaccard"] == pytest.approx(4 / 9)
    assert out["state"]["Local"]["max_abs_diff"] == pytest.approx(0.5)
    assert out["state"]["No-Spike"]["share_bins_over_1e-3"] == pytest.approx(1 / 13)


# --- Task 12's evidence with half-decode labels ----------------------------------------------
def test_one_direction_lets_only_the_first_half_select_and_the_second_score():
    n_time, n_pos, n_tet = 260, 25, 5
    terms = random_terms(n_time, n_pos, n_tet, rate=0.5, seed=11)
    windows = overlapping_windows(n_time, seed=12)
    animal = np.random.default_rng(13).integers(0, n_pos, n_time)
    distance = linear_distance(n_pos)
    (first, second), *_ = tetrode_halves(n_tet, n_partitions=1, seed=14)

    def evidence(halves, both_directions):
        return window_evidence(
            terms, windows, animal, distance, halves, n_partners=6,
            both_directions=both_directions, no_spike_rate=NO_SPIKE_RATE, seed=15,
        )  # fmt: skip

    forward = evidence([(first, second)], False)
    backward = evidence([(second, first)], False)
    both = evidence([(first, second)], True)

    for column in both.columns:
        if column.startswith("cond_"):  # summed over runs
            expected = forward[column] + backward[column]
        elif column.endswith("_sd_runs") or column.startswith("run0_"):
            continue
        else:
            with warnings.catch_warnings():  # windows NaN in both runs stay NaN
                warnings.simplefilter("ignore", RuntimeWarning)
                pair = np.c_[forward[column], backward[column]].astype(float)
                expected = np.nanmean(pair, axis=1)
        np.testing.assert_allclose(both[column], expected, rtol=1e-12, err_msg=column)
    in_window = [
        (terms.time_bin >= s) & (terms.time_bin <= e)
        for s, e in zip(windows.s, windows.e, strict=True)
    ]
    np.testing.assert_array_equal(
        forward.ns_count, [np.isin(terms.tetrode[m], second).sum() for m in in_window]
    )
    np.testing.assert_array_equal(
        forward.sel_count, [np.isin(terms.tetrode[m], first).sum() for m in in_window]
    )
    assert (forward.loc_mark_sd_runs.dropna() == 0).all()  # one run per window


def two_split_windows() -> pd.DataFrame:
    """One epoch's bouts and matched windows from two splits, at the same bins."""
    rows = []
    for split, value in [("p0d0", 1.0), ("p0d1", 3.0)]:
        for kind in ("bout", "baseline"):
            rows.append(
                {
                    "epoch_id": EPOCH, "animal": "aa", "split": split, "kind": kind,
                    "state": "Local", "regime": "immobile", "s": 10, "e": 19,
                    "n_bins": 10, "bout_s": 10.0 if kind == "bout" else np.nan,
                    "matched_bout_s": 10.0 if kind == "baseline" else np.nan,
                    "n_found": 4 if split == "p0d0" else 6,
                    "runcv_loc_mark": value if kind == "bout" else 0.0,
                }
            )  # fmt: skip
    return pd.DataFrame(rows)


def test_bouts_pair_with_their_matched_windows_within_each_split():
    windows = two_split_windows()

    paired = paired_bouts(windows)

    assert len(paired) == 2
    assert dict(zip(paired.split, paired.runcv_loc_mark, strict=True)) == {
        "p0d0": 1.0,
        "p0d1": 3.0,
    }
    assert (paired.runcv_loc_mark_base == 0.0).all()
    stats = bout_statistics(
        windows, metrics={"Local": ("runcv_loc_mark",)}, n_boot=10, n_flips=10
    )
    row = stats[stats.regime == "immobile"].iloc[0]
    assert row.n == 2 and row.n_found == 10 and row.n_epochs == 1
    with pytest.raises(pd.errors.MergeError):  # without the split, they collide
        paired_bouts(windows.drop(columns="split"))


# --- the runner ---------------------------------------------------------------------------------
@pytest.fixture
def device_environment(monkeypatch):
    """Whatever the runner sets is undone after the test, so JAX stays on the CPU."""
    for name in (
        "JAX_PLATFORMS",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    ):
        monkeypatch.setenv(name, "cuda" if name == "JAX_PLATFORMS" else "0")


def test_without_cuda_visible_devices_the_decodes_stay_on_the_cpu(device_environment):
    device = runner.configure_devices(None)

    assert device == "cpu"
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_a_named_gpu_must_be_one_idle_card(device_environment):
    usage = {3: (1206.0, 0.0), 4: (17300.0, 0.0), 5: (100.0, 7.0), 6: (0.0, np.nan)}

    def query():
        return usage

    assert runner.configure_devices("3", query=query) == "gpu:3"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert os.environ["JAX_PLATFORMS"] == "cuda"
    for devices, reason in [
        ("4", "17300 MiB in use"),
        ("5", "7 % busy"),
        ("6", "nan % busy"),
        ("9", "no GPU 9"),
        ("3,4", "exactly one GPU"),
    ]:
        with pytest.raises(SystemExit, match=reason):
            runner.configure_devices(devices, query=query)


def test_importing_the_runner_does_not_load_jax():
    """So ``configure_devices`` still chooses the device before JAX starts."""
    probe = (
        "import sys; import scripts.qa.run_half_tetrode_decodes; "
        "print('jax' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert out.stdout.strip() == "False"


def written_record(paths: dict, **record) -> None:
    """A decode file and its record, as an earlier run would leave them."""
    paths["decode"].parent.mkdir(parents=True, exist_ok=True)
    paths["decode"].touch()
    paths["record"].write_text(json.dumps(record))


def test_a_decode_is_kept_only_while_its_length_detector_and_code_match(tmp_path):
    paths = runner.decode_paths(EPOCH, "p0d0", tmp_path)
    stamp = {"name": f"{EPOCH}_detector.pkl", "size": 10, "mtime_ns": 5}
    code = {"commit": "a" * 40, "dirty": False}

    def reason(n_bins=None, *, now_stamp=stamp, now_code=code):
        return runner.reuse_reason(
            EPOCH, "p0d0", tmp_path, n_bins, stamp=now_stamp, code=now_code
        )

    assert reason() == "not written before"
    good = {
        "status": "ok",
        "n_bins": N_TIME,
        "complete": True,
        "canonical_detector": stamp,
        "code": code,
    }
    written_record(paths, **good)
    assert reason() is None
    assert reason(3000) == f"the earlier decode covers {N_TIME} bins, not 3000"
    # a smoke test's decode never stands in for a whole epoch's, nor the reverse
    written_record(paths, **{**good, "n_bins": 3000, "complete": False})
    assert reason(3000) is None
    assert reason() == "the earlier decode covers only the first 3000 bins"
    assert reason(2000) == "the earlier decode covers 3000 bins, not 2000"
    written_record(paths, **{**good, "status": "failed"})
    assert reason() == "the earlier decode failed"
    written_record(paths, **good)
    assert reason(now_stamp={**stamp, "mtime_ns": 6}) == (
        "the canonical detector file changed (name, size or mtime_ns)"
    )
    assert reason(now_code={"commit": "b" * 40, "dirty": False}) == (
        "code commit aaaaaaaaa -> bbbbbbbbb"
    )
    dirty = "uncommitted changes, so the commit does not identify the code"
    assert reason(now_code={**code, "dirty": True}) == dirty
    written_record(paths, **{**good, "code": {**code, "dirty": True}})
    assert reason() == dirty
    paths["decode"].unlink()
    assert reason() == "not written before"


def test_the_detector_stamp_survives_a_move_but_not_an_edit(tmp_path):
    staging, canonical_dir = tmp_path / "staging", tmp_path / "canonical"
    staging.mkdir()
    canonical_dir.mkdir()
    path = staging / f"{EPOCH}_detector.pkl"
    path.write_bytes(b"detector")
    stamp = runner.canonical_detector_stamp(EPOCH, staging)

    os.rename(path, canonical_dir / path.name)  # a move keeps size and mtime

    moved = canonical_dir / path.name
    assert runner.canonical_detector_stamp(EPOCH, canonical_dir) == stamp
    info = moved.stat()
    moved.write_bytes(b"detecto2")  # the same size, edited
    os.utime(moved, ns=(info.st_atime_ns, info.st_mtime_ns + 1))
    assert runner.canonical_detector_stamp(EPOCH, canonical_dir) != stamp
    moved.write_bytes(b"a longer detector")
    os.utime(moved, ns=(info.st_atime_ns, info.st_mtime_ns))  # the old mtime
    assert runner.canonical_detector_stamp(EPOCH, canonical_dir) != stamp


def test_one_split_is_decoded_checked_and_written(canonical, tmp_path, source):
    decode_dir = tmp_path / "decode"
    write_canonical_decode(canonical, decode_dir)
    loaded = runner.load_canonical(EPOCH, decode_dir)
    assert loaded.labels_reproduced and loaded.n_tetrodes == len(FIELD_CENTERS_CM)
    # the labels are read through the SWR source: the second burst's event lies
    # in excluded time, so of the decode's three events two are analysed
    raw = pd.read_pickle(decode_dir / f"{EPOCH}_80_data_over_time.pkl")
    assert np.unique(raw.non_local_event[raw.non_local_event > 0]).size == 3
    assert np.unique(loaded.labels[loaded.labels > 0]).size == 2
    assert not loaded.labels[1100:1128].any()
    job = runner.decode_jobs(loaded.n_tetrodes)[1]
    output_dir = tmp_path / "out"

    record = runner.decode_split(
        loaded, job, output_dir=output_dir, provenance=PROVENANCE
    )

    assert record["status"] == "ok", record.get("traceback")
    assert (record["split"], record["partition"], record["direction"]) == (
        "p0d1",
        0,
        1,
    )
    assert record["selecting_tetrodes"] == job.selecting.tolist()
    assert record["code"] == PROVENANCE and record["canonical_detector"] == loaded.stamp
    check = record["half_encoding"]
    assert check["same_as_canonical_subset"] and check["n_electrodes"] == 3
    assert check["max_rel_ground"] <= 1e-6 and check["same_ground_models"]
    assert record["n_bins"] == N_TIME and record["complete"]
    assert record["agreement"]["reference_labels_reproduced"]
    paths = runner.decode_paths(EPOCH, "p0d1", output_dir)
    assert json.loads(paths["record"].read_text()) == record
    expected, _ = half_tetrode_decode(
        canonical.detector, job.selecting, is_missing=canonical.is_missing
    )
    with xr.open_dataset(paths["decode"], engine="h5netcdf") as decoded:
        # stored as float32, which the decode already is unless another test
        # module has turned on JAX's x64 mode
        np.testing.assert_array_equal(
            decoded.acausal_state_probabilities.values,
            expected.acausal_state_probabilities.values.astype(np.float32),
        )
        assert json.loads(decoded.attrs["canonical_detector"]) == loaded.stamp
        assert decoded.attrs["complete"] == 1

    # a split that cannot be decoded is recorded, and leaves no decode file
    bad = job._replace(name="p9d9", selecting=np.array([99]))
    failed = runner.decode_split(
        loaded, bad, output_dir=output_dir, provenance=PROVENANCE
    )
    assert failed["status"] == "failed" and "distinct indices" in failed["error"]
    bad_paths = runner.decode_paths(EPOCH, "p9d9", output_dir)
    assert not bad_paths["decode"].exists() and bad_paths["record"].exists()


def test_the_half_decode_uses_the_canonical_decodes_masks(
    canonical, tmp_path, source, monkeypatch
):
    """Built by hpc_decode_masks as the decoder built them: the half encoding is
    fit at the canonical training bins and the decode sets the canonical
    missing bins missing; both agree with the decode's saved decode_masks."""
    decode_dir = tmp_path / "decode"
    write_canonical_decode(canonical, decode_dir)
    loaded = runner.load_canonical(EPOCH, decode_dir)
    seen: dict[str, np.ndarray] = {}
    fit = library_base.ClusterlessDetector.fit
    predict = library_base.ClusterlessDetector.predict

    def spy_fit(self, *args, **kwargs):
        seen["is_training"] = np.asarray(kwargs["is_training"])
        return fit(self, *args, **kwargs)

    def spy_predict(self, *args, **kwargs):
        seen["is_missing"] = np.asarray(kwargs["is_missing"])
        return predict(self, *args, **kwargs)

    monkeypatch.setattr(library_base.ClusterlessDetector, "fit", spy_fit)
    monkeypatch.setattr(library_base.ClusterlessDetector, "predict", spy_predict)

    record = runner.decode_split(
        loaded,
        runner.decode_jobs(loaded.n_tetrodes)[0],
        output_dir=tmp_path / "out",
        provenance=PROVENANCE,
    )

    assert record["status"] == "ok", record.get("traceback")
    assert canonical.is_missing.any()
    np.testing.assert_array_equal(loaded.is_missing, canonical.is_missing)
    np.testing.assert_array_equal(seen["is_training"], canonical.is_training)
    np.testing.assert_array_equal(seen["is_missing"], canonical.is_missing)
    saved = saved_decode_masks(canonical)
    assert int(seen["is_missing"].sum()) == saved["n_missing_bins"]
    assert int((~seen["is_training"]).sum()) == saved["n_encoding_removed_bins"]
    assert record["n_missing_bins"] == saved["n_missing_bins"]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"artifact_run_id": "another"}, "decoded with"),
        ({"n_missing_bins": 0}, "n_missing_bins"),
    ],
)
def test_a_canonical_decode_made_under_other_masks_is_refused(
    canonical, tmp_path, source, changes, message
):
    """Its encoding and missing bins would not be the ones built now."""
    decode_dir = tmp_path / "decode"
    write_canonical_decode(
        canonical, decode_dir, decode_masks=saved_decode_masks(canonical, **changes)
    )

    with pytest.raises(ValueError, match=message):
        runner.load_canonical(EPOCH, decode_dir)


def test_a_canonical_decode_trained_on_other_bins_is_refused(
    canonical, tmp_path, monkeypatch
):
    """The inventory's events moved by 5 bins: as many bins leave encoding, but
    not the ones the canonical decode left out."""
    time = 1000.0 + np.arange(N_TIME) / FS
    events = [(time[s - 5], time[e + 14]) for s, e, _ in BURSTS]
    inventory = write_swr_inventory(
        tmp_path / "swr", {EPOCH: events}, artifact_run_id=ARTIFACT_RUN_ID
    )
    artifacts = write_artifact_run(tmp_path / "artifacts", {EPOCH: ARTIFACT_CORES})
    use_swr_source(monkeypatch, inventory, artifacts)
    decode_dir = tmp_path / "decode"
    write_canonical_decode(canonical, decode_dir)

    with pytest.raises(ValueError, match="training bins"):
        runner.load_canonical(EPOCH, decode_dir)


@pytest.mark.parametrize("alteration", ["mean_rates", "ground_models", "keys"])
def test_a_canonical_encoding_that_is_not_the_plain_fit_fails_the_split(
    canonical, tmp_path, alteration, source
):
    """What a posterior-weighted canonical encoding, say, would do to the check."""
    decode_dir = tmp_path / "decode"
    write_canonical_decode(canonical, decode_dir)
    loaded = runner.load_canonical(EPOCH, decode_dir)
    job = runner.decode_jobs(loaded.n_tetrodes)[0]
    stored = loaded.detector.encoding_model_[ENCODING_KEY]
    if alteration == "mean_rates":
        stored["mean_rates"] = [rate * 1.01 for rate in stored["mean_rates"]]
    elif alteration == "ground_models":
        model = stored["gpi_models"][job.selecting[0]]
        model.std = model.std * 2
    else:
        stored["an_extra_entry"] = None

    record = runner.decode_split(
        loaded, job, output_dir=tmp_path / "out", provenance=PROVENANCE
    )

    assert record["status"] == "failed"
    assert "not the canonical encoding restricted" in record["error"]
    assert not runner.decode_paths(EPOCH, job.name, tmp_path / "out")["decode"].exists()


def test_half_labels_are_decoded_and_scored_end_to_end(
    canonical, tmp_path, monkeypatch, source
):
    decode_dir = tmp_path / "decode"
    write_canonical_decode(canonical, decode_dir)
    output_dir = tmp_path / "half_tetrode"
    monkeypatch.setattr(run_state_evidence, "CV_BLOCK_S", 2.0)  # a 12 s epoch
    monkeypatch.setattr(runner, "provenance_record", lambda _root: PROVENANCE)

    records = runner.decode_epoch(
        EPOCH, decode_dir=decode_dir, output_dir=output_dir, log=lambda _: None
    )
    check = runner.decode_epoch(
        EPOCH, decode_dir=decode_dir, output_dir=output_dir, all_tetrodes=True,
        log=lambda _: None,
    )  # fmt: skip

    assert [r["split"] for r in records] == ["p0d0", "p0d1", "p1d0", "p1d1"]
    for record in records:
        assert record["status"] == "ok", record.get("traceback")
        assert record["half_encoding"]["same_as_canonical_subset"]
        assert record["half_encoding"]["max_rel_occupancy"] == 0.0  # CPU and CPU
        assert record["device"] == "cpu"
        path = runner.decode_paths(EPOCH, record["split"], output_dir)
        with xr.open_dataset(path["decode"], engine="h5netcdf") as decoded:
            assert decoded.acausal_state_probabilities.shape == (N_TIME, 4)
            assert (
                json.loads(decoded.attrs["selecting_tetrodes"])
                == (record["selecting_tetrodes"])
            )
    # S = every tetrode reproduces the canonical decode, and its events are held
    # to the analysed-time rule as the canonical decode's are: two of three
    [agreement] = [r["agreement"] for r in check]
    assert check[0]["split"] == "all"
    assert agreement["max_abs_diff"] < 1e-5
    assert agreement["events_reference"] == agreement["events_other"] == 2
    assert agreement["events_identical"] == agreement["events_reference"]
    # a rerun keeps what is written
    again = runner.decode_epoch(
        EPOCH, decode_dir=decode_dir, output_dir=output_dir, log=lambda _: None
    )
    assert {r["status"] for r in again} == {"exists"}

    record = runner.score_epoch(
        EPOCH, decode_dir=decode_dir, output_dir=output_dir,
        load_inputs=lambda _epoch_id: stand_in_loader(canonical), log=lambda _: None,
    )  # fmt: skip

    assert record["status"] == "ok", record.get("traceback")
    label_sets = record["label_sets"]
    assert set(label_sets) == {
        f"{source}|{split}"
        for source in ("half", "all")
        for split in ("p0d0", "p0d1", "p1d0", "p1d1")
    }
    for label_set in label_sets.values():
        assert not label_set["both_directions"]
        assert label_set["windows"]["silent_ns_logratio_identity_max_abs"] < 1e-9
    table = read_table(runner.evidence_paths(EPOCH, output_dir)["windows"])
    assert set(table.label_source) == {"half", "all"}
    assert table.groupby(["label_source", "split"]).size().index.size == 8
    assert {"runcv_loc_mark", "run_nl_anim_mark", "can_ns_logratio"} <= set(table)
    # the matched comparison scores the canonical labels' windows, split by split
    every = table[table.label_source == "all"]
    first = every[every.split == "p0d0"][["kind", "state", "regime", "s", "e"]]
    for split in ("p0d1", "p1d0", "p1d1"):
        same = every[every.split == split][["kind", "state", "regime", "s", "e"]]
        pd.testing.assert_frame_equal(
            first.reset_index(drop=True), same.reset_index(drop=True)
        )
    # each half split's windows carry its own decode's probabilities
    half = table[(table.label_source == "half") & (table.split == "p0d0")]
    with xr.open_dataset(
        runner.decode_paths(EPOCH, "p0d0", output_dir)["decode"], engine="h5netcdf"
    ) as decoded:
        p = decoded.acausal_state_probabilities.values.astype(float)
    row = half.iloc[0]
    np.testing.assert_allclose(
        row[[f"p_{state}" for state in STATES]].to_numpy(dtype=float),
        p[int(row.s) : int(row.e) + 1].mean(0),
    )

    agreement = runner.summarize_label_sources(output_dir, [EPOCH])
    for source in ("half", "all"):
        for name in ("bout_stats.csv", "trend_stats.csv", "silence_stats.csv"):
            assert (output_dir / "evidence" / source / name).exists(), (source, name)
    side_by_side = pd.read_csv(output_dir / "evidence" / "half_vs_all_bouts.csv")
    assert {"animal_estimate_half", "animal_estimate_all"} <= set(side_by_side)
    assert side_by_side.primary.all()

    # how far the two sources' selections differ, per split and source
    written = pd.read_csv(output_dir / "evidence" / "half_vs_all_agreement.csv")
    assert len(written) == len(agreement) == 8
    by_split = {r["split"]: r["agreement"] for r in records}
    for row in agreement.itertuples(index=False):
        events = by_split[row.split]
        mine = "events_other" if row.label_source == "half" else "events_reference"
        assert row.events == events[mine]
        assert row.event_bin_jaccard == pytest.approx(events["event_bin_jaccard"])
        bouts = table[
            (table.kind == "bout")
            & (table.split == row.split)
            & (table.label_source == row.label_source)
            & (table.state == "Non-Local Continuous")
            & (table.regime == "immobile")
        ]
        assert row.bouts_nl_cont_immobile == (
            bouts.n_found.iloc[0] if len(bouts) else 0
        )
    assert agreement[agreement.label_source == "all"].bouts_nl_cont_immobile.min() > 0
    text = runner.readme({EPOCH: record}, PROVENANCE, decode_dir, agreement)
    assert "share their scoring runs" in text and "not their bouts" in text
    assert "counts bout x split" in text
    assert sum(line.startswith(f"| {EPOCH} | p") for line in text.splitlines()) == 4
    manifest = runner.score_manifest({EPOCH: record}, PROVENANCE, decode_dir, agreement)
    assert len(manifest["agreement"][EPOCH]) == 8
    json.dumps(manifest, allow_nan=False)

    # a canonical detector file moved to another folder still stands for its
    # decodes; an edited one does not
    moved = tmp_path / "canonical"
    decode_dir.rename(moved)
    assert (
        len(runner.half_label_sets(EPOCH, output_dir=output_dir, decode_dir=moved)) == 8
    )
    detector = moved / f"{EPOCH}_detector.pkl"
    info = detector.stat()
    os.utime(detector, ns=(info.st_atime_ns, info.st_mtime_ns + 10**9))
    stale = runner.score_epoch(
        EPOCH, decode_dir=moved, output_dir=output_dir,
        load_inputs=lambda _epoch_id: stand_in_loader(canonical), log=lambda _: None,
    )  # fmt: skip
    assert stale["status"] == "failed"
    assert "another canonical detector file" in stale["error"]


# --- the open-file limit ------------------------------------------------------------------
class _Started(Exception):
    """Raised where the run would start its work."""


def test_decode_raises_the_open_file_limit_first(tmp_path, monkeypatch):
    """Before the devices are chosen; the decode log says what it was raised to."""
    events = []

    def raise_open_file_limit():
        events.append("limit")
        return 1024, 65536

    def configure_devices(cuda_visible_devices):
        events.append("devices")
        return "cpu"

    monkeypatch.setattr(runner, "raise_open_file_limit", raise_open_file_limit)
    monkeypatch.setattr(runner, "configure_devices", configure_devices)
    monkeypatch.setattr(runner, "current_device", lambda: "cpu")
    monkeypatch.setattr(runner, "decode_epoch", lambda epoch_id, **kwargs: [])
    output_dir = tmp_path / "out"

    runner.main(["decode", "--epochs", EPOCH, "--decode-dir", str(tmp_path / "decode"),
                 "--output-dir", str(output_dir)])  # fmt: skip

    assert events == ["limit", "devices"]
    (log,) = (output_dir / "logs").glob("decode_*.log")
    assert "open-file soft limit 1024 -> 65536" in log.read_text()


def test_score_raises_the_open_file_limit_before_its_workers_start(
    tmp_path, monkeypatch, capsys
):
    """Its epochs run on Task 12's worker processes, which inherit the limit only
    if it is raised before they start."""
    events = []

    def raise_open_file_limit():
        events.append("limit")
        return 1024, 65536

    def run_epochs(epochs, kwargs, n_workers):
        events.append("workers")
        raise _Started

    monkeypatch.setattr(runner, "raise_open_file_limit", raise_open_file_limit)
    monkeypatch.setattr(runner, "provenance_record", lambda _root: PROVENANCE)
    monkeypatch.setattr(run_state_evidence, "run_epochs", run_epochs)

    with pytest.raises(_Started):
        runner.main(["score", "--epochs", EPOCH, "--decode-dir",
                     str(tmp_path / "decode"), "--output-dir", str(tmp_path / "out")])  # fmt: skip

    assert events == ["limit", "workers"]
    assert "open-file soft limit 1024 -> 65536" in capsys.readouterr().out
