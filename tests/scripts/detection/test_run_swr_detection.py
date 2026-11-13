"""The SWR runner: its outputs, its manifest record, and its parallel driver.

Nothing here reaches the Spyglass database. Every fetch is a fake: the epoch is
``ripple_detection``'s simulated session at 1000 Hz, written as the stored
ripple band and the referenced raw LFP behind it, and the band filter is a fake
that runs the package's 150-250 Hz band-pass where Spyglass would run its FIR.
Session names are made up, apart from the documented no-input epochs, which are
only looked up.

The parallel driver runs on an in-process ``LocalCluster`` of worker threads
standing in for the runner's worker processes. Nothing is written outside the
test's ``tmp_path``: every output folder is given under it, the tests that run
``main`` work in it, so the per-epoch ``logs/`` land there, and dask's scratch
space is pointed into it.
"""

from __future__ import annotations

import errno
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from functools import cache
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import dask
import numpy as np
import pandas as pd
import pytest
from dask.distributed import LocalCluster
from numpy.typing import NDArray
from ripple_detection import Kay_ripple_detector, filter_ripple_band
from ripple_detection.simulate import simulate_session, simulate_time

import scripts.detection.run_swr_detection as runner
from continuum_swr_replay.analysis.detection import swr_detectors
from continuum_swr_replay.analysis.detection.swr_detectors import (
    NoUsableChannelError,
    best_ripple_channel,
)
from continuum_swr_replay.analysis.events import swr_source
from continuum_swr_replay.decoding.identifiers import parse_nwb_filename
from continuum_swr_replay.runs import batch
from continuum_swr_replay.runs.io import sha256_file
from continuum_swr_replay.tables.io import read_attrs, read_table, write_table
from scripts.detection.run_swr_detection import main, run_epoch
from tests.support._swr_inventory_fixtures import (
    ARTIFACT_RUN_ID,
    use_artifact_run,
    write_artifact_run,
)

FS = 1000.0
BASE_TIME_S = 1.58e9
"""Unix-like timestamps, as Spyglass stores them, so nothing can assume time 0."""

NWB_FILE_NAME = "sim20200101_.nwb"
EPOCH_NAME = "02_r1"
DETECTORS = ("kay", "karlsson", "yu", "zugaro")

ELECTRODE_IDS = np.array([0, 4, 8, 12])
"""The band's electrodes, ascending, unlike their LFP columns."""

BAND_COLUMNS = np.array([5, 1, 2, 4])
"""Each band electrode's column in the six-column raw LFP."""

ELECTRODE_GROUP_NAMES = np.array(["0", "1", "2", "3"])
"""Each band electrode's group (tetrode): one band channel per tetrode, as rule
R-T requires and every recorded epoch has."""

LFP_CONVERSION = 1.95e-7
"""Volts per stored count, the LFP series' ``conversion``."""

SPYGLASS_KAY_PARAMETERS = {
    "speed_name": "head_speed",
    "ripple_detection_algorithm": "Kay_ripple_detector",
    "ripple_detection_params": dict(swr_detectors.DETECTOR_SETTINGS["kay"]),
}
"""Spyglass's stored ``"default"`` ripple parameters, equal to the direct Kay's."""

YU_SCALARS = {
    "threshold",
    "mode",
    "mode_index",
    "mean",
    "min",
    "flank_ratio",
    "out_of_grid_fraction",
    "n_values",
    "n_in_grid",
    "normalization_center",
    "normalization_scale",
    "threshold_zscore",
}
"""The Yu diagnostics the record keeps; the histogram arrays go to the sidecar."""

COUNTS_PER_SIMULATED_UNIT = 1_000.0
"""Scale from ``simulate_session``'s units to int16 counts. Unscaled, every
channel is below the dead-channel guard of 1 count (Task 8)."""

SPEED_CM_PER_S = 2.0
POSITION_RATE_HZ = 500.0

INJECTED_RIPPLE_TIMES_S = (5.0, 15.0, 25.0, 35.0, 45.0)
"""Seconds after the first sample: the five ripples of the 60 s epoch."""

MANIFEST_FIELDS = {
    "status",
    "detectors",
    "provenance",
    "electrode_ids",
    "reference_ids",
    "zugaro_channel",
    "band_check",
    "yu_diagnostics",
    "sampling",
    "lfp_filter_name",
    "lfp_units",
    "position_span_s",
    "n_events",
    "zugaro_dropped_over_150ms",
    "n_detecting_tetrodes",
    "meets_published_tetrode_criterion",
    "input_sha256",
    "artifacts",
    "tetrode_selection",
}
"""The fields of an epoch's manifest record the brief names."""


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _fake_fir_filter(
    timestamps, data, filter_coeff, valid_times, electrodes, decimation
):
    """``FirFilterParameters.filter_data``'s contract, by the package's band-pass.

    The ``electrodes`` columns of ``data`` within the valid times, filtered at
    150-250 Hz, decimated, and returned in ``data``'s dtype with their
    timestamps. ``filter_coeff`` is ignored: the package's band-pass stands in
    for the 535-tap ``ripple_150_250`` filter.
    """
    rows = np.zeros(len(timestamps), dtype=bool)
    for start, stop in valid_times:
        rows |= (timestamps >= start) & (timestamps <= stop)
    band = filter_ripple_band(
        data[rows][:, electrodes].astype(np.float64), sampling_frequency=FS
    )
    return band[::decimation].astype(data.dtype), timestamps[rows][::decimation]


@cache
def _simulated_epoch(
    duration_s: float, ripple_times_s: tuple[float, ...]
) -> SimpleNamespace:
    """A simulated session at 1000 Hz with 150-175 Hz ripples, as int16 counts.

    Cached, so the arrays are shared: copy before changing them.

    Returns
    -------
    SimpleNamespace
        ``time`` (n_time,), from ``BASE_TIME_S``; ``raw``, six LFP channels in
        int16 counts (n_time, 6); ``windows`` (n_ripples, 2), each injected
        ripple's start and end.
    """
    time = BASE_TIME_S + simulate_time(round(duration_s * FS), FS)
    session = simulate_session(
        time,
        BASE_TIME_S + np.asarray(ripple_times_s),
        n_channels=6,
        rng=0,
        ripple_frequency=(150, 175),
    )
    raw = np.round(COUNTS_PER_SIMULATED_UNIT * session.lfps).astype(np.int16)
    return SimpleNamespace(time=time, raw=raw, windows=session.ripple_windows)


def _band_inputs(epoch: SimpleNamespace) -> dict:
    """``RippleBandInputs`` whose stored band is ``_fake_fir_filter``'s output."""
    interval = SimpleNamespace(times=np.array([[epoch.time[0], epoch.time[-1]]]))
    band, time = _fake_fir_filter(
        epoch.time, epoch.raw, np.ones(3), interval.times, BAND_COLUMNS, 1
    )
    return {
        "time": time,
        "ripple_band": band.astype(np.float64),
        "ripple_band_dtype": str(band.dtype),
        "raw_referenced": epoch.raw[:, BAND_COLUMNS].astype(np.float64),
        "raw_referenced_native": epoch.raw,
        "raw_timestamps": epoch.time,
        "reference_overflow_count": 0,
        "electrode_ids": ELECTRODE_IDS,
        "reference_ids": np.array([16, 16, 81, 81]),
        "electrode_group_names": ELECTRODE_GROUP_NAMES,
        "lfp_band_elect_index": BAND_COLUMNS,
        "band_valid_interval": interval,
        "decimation": 1,
        "sampling_frequency": FS,
        "band_filter_coeff": np.ones(3),
        "band_filter_name": "ripple_150_250",
        "lfp_filter_name": "LFP 0-400 Hz",
        "lfp_units": "volts",
        "lfp_conversion": LFP_CONVERSION,
    }


def _fetchers(
    epoch: SimpleNamespace,
    *,
    position_span_s: tuple[float, float] | None = None,
    inputs: dict | None = None,
    calls: list | None = None,
    head_speed: Callable[[NDArray], NDArray] | None = None,
    spyglass_parameters: dict | None = None,
) -> dict:
    """``run_epoch``'s fetch arguments, every one a fake.

    Parameters
    ----------
    epoch : SimpleNamespace
        From ``_simulated_epoch``.
    position_span_s : tuple of float, optional
        Seconds after the first sample that the position covers; by default
        all of the LFP. Speed is 2 cm/s throughout, at 500 Hz.
    inputs : dict, optional
        The ripple-band inputs; by default ``_band_inputs(epoch)``.
    calls : list, optional
        Each fetch appends ``(name, nwb_file_name, epoch_name)`` to it.
    head_speed : Callable, optional
        Speed in cm/s as a function of seconds after the first sample; by
        default 2 cm/s throughout.
    spyglass_parameters : dict, optional
        Spyglass's stored ripple parameters; by default
        ``SPYGLASS_KAY_PARAMETERS``.
    """
    inputs = _band_inputs(epoch) if inputs is None else inputs
    time = inputs["time"]
    if position_span_s is None:
        start, end = time[0], time[-1]
    else:
        start, end = (BASE_TIME_S + bound for bound in position_span_s)
    position_time = np.linspace(start, end, round((end - start) * POSITION_RATE_HZ) + 1)
    speed = (
        np.full(position_time.size, SPEED_CM_PER_S)
        if head_speed is None
        else head_speed(position_time - time[0])
    )
    position_info = pd.DataFrame(
        {"head_position_x": 0.0, "head_position_y": 0.0, "head_speed": speed},
        index=pd.Index(position_time, name="time"),
    )
    canonical = pd.DataFrame(
        {"start_time": epoch.windows[:, 0], "end_time": epoch.windows[:, 1]},
        index=pd.Index(np.arange(len(epoch.windows)), name="id"),
    )
    lfp_time = pd.Index(time, name="time")
    spyglass_inputs = (
        pd.Series(SPEED_CM_PER_S, index=lfp_time),
        pd.DataFrame(inputs["ripple_band"], index=lfp_time),
        FS,
        SPYGLASS_KAY_PARAMETERS if spyglass_parameters is None else spyglass_parameters,
    )
    calls = [] if calls is None else calls

    def returning(name, value):
        def fetch(nwb_file_name, epoch_name):
            calls.append((name, nwb_file_name, epoch_name))
            return value

        return fetch

    return {
        "fetch_inputs": returning("inputs", inputs),
        "fetch_position": returning("position", {"position_info": position_info}),
        "fetch_canonical_kay": returning("canonical_kay", canonical),
        "fetch_spyglass_kay_inputs": returning("spyglass_kay_inputs", spyglass_inputs),
        "fir_filter": _fake_fir_filter,
    }


def _main_epoch() -> SimpleNamespace:
    return _simulated_epoch(60.0, INJECTED_RIPPLE_TIMES_S)


def _n_ripples_found(events: pd.DataFrame, windows: NDArray) -> int:
    """Count the injected ripples an event shares positive duration with."""
    start = events.start_time.to_numpy()[:, None]
    end = events.end_time.to_numpy()[:, None]
    shared = np.minimum(end, windows[:, 1]) - np.maximum(start, windows[:, 0])
    return int(np.count_nonzero((shared > 0).any(axis=0)))


def _outputs(output_dir, identifier) -> dict:
    """The output files the runner writes for an epoch, by name."""
    paths = {
        name: output_dir / f"{identifier}_{name}_events.parquet" for name in DETECTORS
    }
    paths["swr_observed"] = output_dir / f"{identifier}_swr_observed.parquet"
    paths["kay_version_check"] = output_dir / f"{identifier}_kay_version_check.parquet"
    paths["yu_diagnostics"] = output_dir / f"{identifier}_yu_diagnostics.json"
    return paths


def _epoch_info(*epochs: tuple[str, str]) -> pd.DataFrame:
    """``epochs_to_run``'s frame (``get_epoch_info``'s shape) for these
    (nwb_file_name, epoch_name) pairs."""
    animal_date = [parse_nwb_filename(nwb_file_name) for nwb_file_name, _ in epochs]
    return pd.DataFrame(
        {
            "animal": [animal for animal, _ in animal_date],
            "date": [date for _, date in animal_date],
            "epoch": [int(name.split("_")[0]) for _, name in epochs],
        },
        index=pd.MultiIndex.from_tuples(
            epochs, names=["nwb_file_name", "interval_list_name"]
        ),
    )


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------


def test_outputs_and_manifest_fields(tmp_path):
    """Every output is written, and the record carries every manifest field.

    Zugaro's channel is one of the band's electrodes, the band check passes
    with the filter that made the band, and each detector finds the injected
    ripples, so each read the input it should.
    """
    epoch = _main_epoch()
    calls: list = []
    fetchers = _fetchers(epoch, calls=calls)
    output_dir = tmp_path / "out"
    identifier = swr_source.swr_epoch_id(NWB_FILE_NAME, EPOCH_NAME)

    record = run_epoch(NWB_FILE_NAME, EPOCH_NAME, output_dir=output_dir, **fetchers)

    assert identifier == "sim20200101_02_r1_HPC"
    assert {(nwb, name) for _, nwb, name in calls} == {(NWB_FILE_NAME, EPOCH_NAME)}
    assert {fetch for fetch, _, _ in calls} == {
        "inputs",
        "position",
        "canonical_kay",
        "spyglass_kay_inputs",
    }
    json.dumps(record, allow_nan=False)
    assert MANIFEST_FIELDS <= set(record)
    assert record["status"] == "ok"
    assert record["detectors"] == dict.fromkeys(DETECTORS, "ok")

    paths = _outputs(output_dir, identifier)
    for path in paths.values():
        assert path.exists(), path
    for name in DETECTORS:
        events = read_table(paths[name])
        assert record["n_events"][name] == len(events)
        assert _n_ripples_found(events, epoch.windows) >= 4, name
        assert read_attrs(paths[name])["epoch_id"] == identifier
    observed = read_table(paths["swr_observed"])
    assert list(observed.columns) == ["scope", "start_s", "end_s"]
    assert set(observed.scope) == {"stored_band", "zugaro"}
    kay_check = read_table(paths["kay_version_check"])
    assert set(kay_check.comparison) == {"version_only", "version_and_inputs"}
    assert (kay_check.detector == "kay_spyglass").sum() == 2 * len(epoch.windows)
    by_comparison = kay_check.groupby("comparison").detector.unique().map(set)
    assert by_comparison.to_dict() == {
        "version_only": {"kay_spyglass_inputs", "kay_spyglass"},
        "version_and_inputs": {"kay", "kay_spyglass"},
    }
    # a key names one event: Kay on Spyglass's inputs and the direct Kay differ
    assert not kay_check.duplicated(["comparison", "event_key"]).any()
    keys = kay_check.groupby("detector").event_key.agg(set)
    assert not keys["kay"] & keys["kay_spyglass_inputs"]
    version_check = record["kay_version_check"]
    assert version_check["spyglass_parameters"] == SPYGLASS_KAY_PARAMETERS
    assert version_check["settings_mismatch"] is False
    assert version_check["differences"] == {
        "version_only": ["package version"],
        "version_and_inputs": ["package version", "inputs"],
    }
    for comparison, detectors in version_check["counts"].items():
        for detector, counts in detectors.items():
            rows = kay_check[
                (kay_check.comparison == comparison) & (kay_check.detector == detector)
            ]
            assert counts["all"] == rows.status.value_counts().to_dict()
            # at 2 cm/s throughout every event is speed-eligible
            assert counts["speed_eligible"] == counts["all"]

    yu = record["yu_diagnostics"]
    assert set(yu) == YU_SCALARS
    sidecar = json.loads(paths["yu_diagnostics"].read_text())
    assert sidecar["epoch_id"] == identifier
    assert sidecar["run_id"] == output_dir.name
    assert {name: sidecar["yu_diagnostics"][name] for name in YU_SCALARS} == yu
    for name in ("histogram_edges", "counts", "smoothed_counts", "mirrored_counts"):
        assert len(sidecar["yu_diagnostics"][name]) > 1000, name
    assert set(record["outputs"]) == set(paths)
    for name, path in paths.items():
        assert record["outputs"][name] == {
            "path": str(path),
            "sha256": sha256_file(path),
        }

    for field in ("commit", "dirty", "ripple_detection_commit", "uv_lock_sha256"):
        assert field in record["provenance"], field
    assert record["provenance"]["numpy_version"] == np.__version__
    assert "scipy_version" in record["provenance"]
    assert record["electrode_ids"] == ELECTRODE_IDS.tolist()
    assert record["reference_ids"] == [16, 16, 81, 81]

    channel = record["zugaro_channel"]
    assert set(channel) == {"electrode_id", "scores", "excluded"}
    assert channel["electrode_id"] in ELECTRODE_IDS.tolist()
    assert set(channel["scores"]) == {str(e) for e in ELECTRODE_IDS}
    assert channel["excluded"] == {}

    assert record["band_check"] == {
        "max_abs_difference": 0.0,
        "tolerance": 1.0,
        "band_dtype": "int16",
        "reference_overflow_count": 0,
    }
    yu_events = read_table(paths["yu"])
    assert record["yu_diagnostics"]["threshold_zscore"] == pytest.approx(
        yu_events.detection_threshold_zscore.iloc[0], abs=1e-9
    )
    assert record["sampling"]["nominal"] == FS
    assert record["sampling"]["measured"] == pytest.approx(FS, rel=1e-6)
    assert record["lfp_filter_name"] == "LFP 0-400 Hz"
    assert record["lfp_units"] == "volts"
    assert record["lfp_conversion"] == LFP_CONVERSION
    assert record["position_span_s"] == [epoch.time[0], epoch.time[-1]]
    assert (
        record["zugaro_dropped_over_150ms"]
        == read_attrs(paths["zugaro"])["n_dropped_over_maximum_duration"]
    )
    assert isinstance(record["zugaro_dropped_over_150ms"], int)
    assert record["n_detecting_tetrodes"] == 4  # four electrodes on four tetrodes
    assert record["worker_peak_rss_gib"] > 0
    assert record["meets_published_tetrode_criterion"] == "unknown"
    assert set(record["input_sha256"]) == {"time", "ripple_band", "raw_referenced"}
    assert (
        record["input_sha256"]["time"]
        == hashlib.sha256(epoch.time.astype(np.float64).tobytes()).hexdigest()
    )


def _raise(error):
    def detector(*args, **kwargs):
        raise error

    return detector


@pytest.mark.parametrize("failing", ["karlsson", "no_zugaro_channel"])
def test_one_detector_failing_makes_the_epoch_partial(tmp_path, monkeypatch, failing):
    """One detector's error is recorded; the other three still write theirs.

    A Karlsson that raises, and an epoch where no channel passes Zugaro's
    channel choice, both leave the epoch partial. A file the failed detector
    left from an earlier run is removed, so the folder matches the record.
    """
    output_dir = tmp_path / "out"
    identifier = swr_source.swr_epoch_id(NWB_FILE_NAME, EPOCH_NAME)
    paths = _outputs(output_dir, identifier)
    if failing == "karlsson":
        error = RuntimeError("simulated Karlsson failure")
        monkeypatch.setattr(swr_detectors, "Karlsson_ripple_detector", _raise(error))
        failed = "karlsson"
    else:
        error = NoUsableChannelError("simulated: every channel is below the guard")
        monkeypatch.setattr(runner, "best_ripple_channel", _raise(error))
        failed = "zugaro"
    write_table(pd.DataFrame({"start_time": [1.0]}), paths[failed])

    record = run_epoch(
        NWB_FILE_NAME, EPOCH_NAME, output_dir=output_dir, **_fetchers(_main_epoch())
    )

    json.dumps(record, allow_nan=False)
    assert record["status"] == "partial"
    assert record["detectors"][failed] == repr(error)
    assert all(
        record["detectors"][name] == "ok" for name in DETECTORS if name != failed
    )
    assert record["n_events"][failed] is None
    assert not paths[failed].exists()
    for name in DETECTORS:
        if name != failed:
            assert paths[name].exists(), name
    observed = read_table(paths["swr_observed"])
    if failed == "zugaro":
        assert record["zugaro_channel"]["electrode_id"] is None
        assert record["zugaro_dropped_over_150ms"] is None
        assert set(observed.scope) == {"stored_band"}
    else:
        assert set(observed.scope) == {"stored_band", "zugaro"}


def test_every_detector_failing_fails_the_epoch(tmp_path, monkeypatch):
    """With no detector left the epoch is failed, every error recorded, and
    nothing written."""
    errors = {
        "kay": RuntimeError("no Kay"),
        "karlsson": RuntimeError("no Karlsson"),
        "yu": ValueError("no Yu"),
        "zugaro": RuntimeError("no Zugaro"),
    }
    for name, error in errors.items():
        detector = f"{name.capitalize()}_ripple_detector"
        monkeypatch.setattr(swr_detectors, detector, _raise(error))

    record = run_epoch(
        NWB_FILE_NAME,
        EPOCH_NAME,
        output_dir=tmp_path / "out",
        **_fetchers(_main_epoch()),
    )

    json.dumps(record, allow_nan=False)
    assert record["status"] == "failed"
    assert record["detectors"] == {name: repr(e) for name, e in errors.items()}
    assert not list((tmp_path / "out").glob("*"))


def test_an_exception_fails_the_epoch_and_clears_its_outputs(tmp_path):
    """Any other exception is returned as the epoch's failure, not raised.

    The record carries the error's repr and its traceback, and the epoch's
    outputs from an earlier run are removed.
    """
    fetchers = _fetchers(_main_epoch())
    error = RuntimeError("simulated: the position could not be read")

    def fetch_position(nwb_file_name, epoch_name):
        raise error

    fetchers["fetch_position"] = fetch_position
    output_dir = tmp_path / "out"
    paths = _outputs(output_dir, swr_source.swr_epoch_id(NWB_FILE_NAME, EPOCH_NAME))
    write_table(pd.DataFrame({"start_time": [1.0]}), paths["kay"])
    paths["yu_diagnostics"].write_text("{}")

    record = run_epoch(NWB_FILE_NAME, EPOCH_NAME, output_dir=output_dir, **fetchers)

    json.dumps(record, allow_nan=False)
    assert record["status"] == "failed"
    assert record["error"] == repr(error)
    assert "fetch_position" in record["traceback"]
    assert not any(path.exists() for path in paths.values())


@pytest.mark.parametrize("fault", ["mismatch", "partial_coverage"])
def test_a_band_check_failure_stops_the_epoch(tmp_path, fault):
    """A re-filter that does not reproduce the stored band stops the epoch.

    Off by 10 counts everywhere, or covering only the first 100 of the
    band's timestamps: the epoch is failed with the reason and the band
    check's values, its position is never fetched, and nothing is written.
    """
    calls: list = []
    fetchers = _fetchers(_main_epoch(), calls=calls)

    def faulty_filter(*args):
        band, timestamps = _fake_fir_filter(*args)
        if fault == "mismatch":
            return band + np.int16(10), timestamps
        return band[:100], timestamps[:100]

    fetchers["fir_filter"] = faulty_filter

    record = run_epoch(
        NWB_FILE_NAME, EPOCH_NAME, output_dir=tmp_path / "out", **fetchers
    )

    json.dumps(record, allow_nan=False)
    assert record["status"] == "failed"
    assert record["band_check"]["band_dtype"] == "int16"
    assert record["band_check"]["reference_overflow_count"] == 0
    if fault == "mismatch":
        assert record["band_check"]["max_abs_difference"] == 10.0
        assert record["band_check"]["tolerance"] == 1.0
        assert "tolerance" in record["reason"]
    else:
        assert record["band_check"]["max_abs_difference"] is None
        assert record["reason"].startswith("The band check could not be completed:")
        assert "Only 100" in record["reason"]
    assert [name for name, _, _ in calls] == ["inputs"]
    assert not (tmp_path / "out").exists()


def test_unusable_candidates_are_excluded_and_written_as_null(tmp_path):
    """A candidate with no finite sample, and one with zero RMS, are excluded.

    The NaN column costs the other channels no samples: the winner is the
    one the other three would give alone, and Zugaro observes its whole
    stretch less the filter's edges. The zero-RMS channel's NaN score and the
    NaN column's missing score are written as null.
    """
    epoch = _main_epoch()
    inputs = _band_inputs(epoch)
    raw_referenced = inputs["raw_referenced"].copy()
    raw_referenced[:, 0] = np.nan
    raw_referenced[:, 1] = 0.0
    inputs["raw_referenced"] = raw_referenced
    output_dir = tmp_path / "out"

    record = run_epoch(
        NWB_FILE_NAME,
        EPOCH_NAME,
        output_dir=output_dir,
        **_fetchers(epoch, inputs=inputs),
    )

    json.dumps(record, allow_nan=False)
    assert record["status"] == "ok"
    channel = record["zugaro_channel"]
    expected = best_ripple_channel(
        raw_referenced[:, 2:], inputs["time"], FS, ELECTRODE_IDS[2:]
    )
    assert channel["electrode_id"] == expected.electrode_id
    assert channel["excluded"]["0"] == "no finite sample"
    assert channel["excluded"]["4"].startswith("mean RMS 0")
    assert channel["scores"]["0"] is None
    assert channel["scores"]["4"] is None
    observed = read_table(
        _outputs(output_dir, swr_source.swr_epoch_id(NWB_FILE_NAME, EPOCH_NAME))[
            "swr_observed"
        ]
    )
    zugaro = observed.loc[observed.scope == "zugaro", ["start_s", "end_s"]]
    edge = swr_detectors.settling_samples(
        swr_detectors.design_bandpass_sos(
            FS, swr_detectors.ZUGARO_BAND_HZ, swr_detectors.ZUGARO_FILTER_ORDER
        ),
        FS,
    )
    assert zugaro.to_numpy().tolist() == [
        [inputs["time"][edge], inputs["time"][-1 - edge]]
    ]


def test_lfp_outside_the_position_span_is_dropped(tmp_path):
    """Position over 2-8 s of a 10 s LFP: nothing outside [2, 8] s remains.

    Ripples at 1, 5 and 9 s: only the 5 s one is inside the span, so no
    event, no observed interval and no canonical Kay event lies outside it.
    """
    epoch = _simulated_epoch(10.0, (1.0, 5.0, 9.0))
    span = (BASE_TIME_S + 2.0, BASE_TIME_S + 8.0)
    output_dir = tmp_path / "out"
    identifier = swr_source.swr_epoch_id(NWB_FILE_NAME, EPOCH_NAME)
    fetchers = _fetchers(epoch, position_span_s=(2.0, 8.0))

    record = run_epoch(NWB_FILE_NAME, EPOCH_NAME, output_dir=output_dir, **fetchers)

    assert record["position_span_s"] == list(span)
    paths = _outputs(output_dir, identifier)
    written = [name for name in DETECTORS if paths[name].exists()]
    assert {"kay", "karlsson", "zugaro"} <= set(written)
    for name in written:
        events = read_table(paths[name])
        assert len(events) > 0, name
        assert (events.start_time >= span[0]).all(), name
        assert (events.end_time <= span[1]).all(), name
    observed = read_table(paths["swr_observed"])
    assert (observed.start_s >= span[0]).all()
    assert (observed.end_s <= span[1]).all()
    kay_check = read_table(paths["kay_version_check"])
    assert (kay_check.detector == "kay_spyglass").sum() == len(
        set(kay_check.comparison)
    )
    in_span = epoch.time[(epoch.time >= span[0]) & (epoch.time <= span[1])]
    assert (
        record["input_sha256"]["time"] == hashlib.sha256(in_span.tobytes()).hexdigest()
    )
    # Kay on Spyglass's inputs, which span the whole LFP, keeps only its events
    # inside the span
    speed, ripple_lfps, sampling_frequency, _ = fetchers["fetch_spyglass_kay_inputs"](
        NWB_FILE_NAME, EPOCH_NAME
    )
    on_spyglass_inputs = Kay_ripple_detector(
        np.asarray(ripple_lfps.index),
        np.asarray(ripple_lfps),
        np.asarray(speed),
        sampling_frequency,
        **swr_detectors.DETECTOR_SETTINGS["kay"],
    )
    inside = (on_spyglass_inputs.start_time >= span[0]) & (
        on_spyglass_inputs.end_time <= span[1]
    )
    assert 0 < inside.sum() < len(on_spyglass_inputs)
    version_only = (kay_check.detector == "kay_spyglass_inputs") & (
        kay_check.comparison == "version_only"
    )
    assert version_only.sum() == inside.sum()
    # 6 s of LFP is too little for Yu's threshold: no Yu events and no sidecar
    assert set(record["yu_diagnostics"]) == {"error"}
    assert not paths["yu"].exists()
    assert not paths["yu_diagnostics"].exists()


def test_zugaro_events_over_150_ms_are_counted(tmp_path, monkeypatch):
    """The events the 150 ms cap drops are counted in the record.

    The count travels in the capped table's ``attrs``; the runner reads it as
    soon as the detectors return.
    """
    durations = [0.100, 0.150, 0.151]

    def zugaro(time, signal, speed, sampling_frequency, **settings):
        start = time[1_000] + np.arange(len(durations))
        return pd.DataFrame(
            {"start_time": start, "end_time": start + durations, "duration": durations},
            index=pd.Index(np.arange(1, len(durations) + 1), name="event_number"),
        )

    monkeypatch.setattr(swr_detectors, "Zugaro_ripple_detector", zugaro)

    record = run_epoch(
        NWB_FILE_NAME,
        EPOCH_NAME,
        output_dir=tmp_path / "out",
        **_fetchers(_main_epoch()),
    )

    assert record["zugaro_dropped_over_150ms"] == 1
    assert record["n_events"]["zugaro"] == 2


def test_version_only_kay_reads_spyglass_inputs(tmp_path):
    """The version-only comparison runs Kay on Spyglass's inputs, not ours.

    Spyglass's speed here is 10 cm/s, past Kay's 4 cm/s rule, so Kay on
    Spyglass's inputs keeps no event, while the direct Kay, on the runner's
    2 cm/s, keeps its own.
    """
    fetchers = _fetchers(_main_epoch())
    speed, ripple_lfps, sampling_frequency, parameters = fetchers[
        "fetch_spyglass_kay_inputs"
    ](NWB_FILE_NAME, EPOCH_NAME)
    fetchers["fetch_spyglass_kay_inputs"] = lambda nwb_file_name, epoch_name: (
        5 * speed,
        ripple_lfps,
        sampling_frequency,
        parameters,
    )
    output_dir = tmp_path / "out"

    record = run_epoch(NWB_FILE_NAME, EPOCH_NAME, output_dir=output_dir, **fetchers)

    paths = _outputs(output_dir, swr_source.swr_epoch_id(NWB_FILE_NAME, EPOCH_NAME))
    kay_check = read_table(paths["kay_version_check"])
    assert not (kay_check.detector == "kay_spyglass_inputs").any()
    direct = kay_check.detector == "kay"
    assert (direct & (kay_check.comparison == "version_and_inputs")).sum() == (
        record["n_events"]["kay"]
    )
    assert record["n_events"]["kay"] > 0
    n_canonical = len(_main_epoch().windows)
    assert record["kay_version_check"]["counts"]["version_only"] == {
        "kay_spyglass": {
            "all": {"missed": n_canonical},
            "speed_eligible": {"missed": n_canonical},
        }
    }


def test_version_only_kay_runs_with_spyglass_stored_parameters(tmp_path):
    """Kay on Spyglass's inputs takes Spyglass's stored parameters.

    The stored z-score threshold here is 3, the direct Kay's 2. The
    version-only events are Kay's at 3 on Spyglass's inputs, and the record
    flags the mismatch, which the version-and-inputs comparison then also
    carries.
    """
    parameters = {
        **SPYGLASS_KAY_PARAMETERS,
        "ripple_detection_params": {
            **SPYGLASS_KAY_PARAMETERS["ripple_detection_params"],
            "zscore_threshold": 3.0,
        },
    }
    fetchers = _fetchers(_main_epoch(), spyglass_parameters=parameters)
    output_dir = tmp_path / "out"

    record = run_epoch(NWB_FILE_NAME, EPOCH_NAME, output_dir=output_dir, **fetchers)

    speed, ripple_lfps, sampling_frequency, _ = fetchers["fetch_spyglass_kay_inputs"](
        NWB_FILE_NAME, EPOCH_NAME
    )
    expected = {
        threshold: len(
            Kay_ripple_detector(
                np.asarray(ripple_lfps.index),
                np.asarray(ripple_lfps),
                np.asarray(speed),
                sampling_frequency,
                **{
                    **parameters["ripple_detection_params"],
                    "zscore_threshold": threshold,
                },
            )
        )
        for threshold in (2.0, 3.0)
    }
    assert expected[3.0] < expected[2.0]
    kay_check = read_table(
        _outputs(output_dir, swr_source.swr_epoch_id(NWB_FILE_NAME, EPOCH_NAME))[
            "kay_version_check"
        ]
    )
    assert (kay_check.detector == "kay_spyglass_inputs").sum() == expected[3.0]
    version_check = record["kay_version_check"]
    assert version_check["spyglass_parameters"] == parameters
    assert version_check["settings_mismatch"] is True
    assert version_check["differences"] == {
        "version_only": ["package version"],
        "version_and_inputs": ["package version", "inputs", "settings"],
    }


def test_spyglass_kay_inputs_that_cannot_be_read_leave_the_epoch_partial(tmp_path):
    """Spyglass's own Kay inputs, which only the version-only comparison reads,
    fail to load (two PositionOutput parents, as for j1620210710 02_r1 and
    04_r2): the four detectors' events and the version-and-inputs comparison
    are still written, the error is recorded under the Kay check, and the epoch
    is partial with that reason."""
    error = ValueError("Found 2 potential parents: [__interval_position_info, ...]")

    def unreadable(nwb_file_name, epoch_name):
        raise error

    fetchers = _fetchers(_main_epoch())
    fetchers["fetch_spyglass_kay_inputs"] = unreadable
    output_dir = tmp_path / "out"

    record = run_epoch(NWB_FILE_NAME, EPOCH_NAME, output_dir=output_dir, **fetchers)

    json.dumps(record, allow_nan=False)
    assert record["status"] == "partial"
    assert "version-only Kay check" in record["reason"]
    assert record["detectors"] == dict.fromkeys(DETECTORS, "ok")
    paths = _outputs(output_dir, swr_source.swr_epoch_id(NWB_FILE_NAME, EPOCH_NAME))
    for name in DETECTORS:
        assert paths[name].exists(), name
    version_check = record["kay_version_check"]
    assert version_check["version_only"] == {"error": repr(error)}
    assert version_check["spyglass_parameters"] is None
    # Spyglass's stored parameters could not be read, so whether the direct
    # Kay's settings match them is unknown, and said to be
    assert version_check["settings_mismatch"] == "unknown"
    assert version_check["differences"] == {
        "version_and_inputs": ["package version", "inputs", "settings unknown"]
    }
    assert set(version_check["counts"]) == {"version_and_inputs"}
    kay_check = read_table(paths["kay_version_check"])
    assert set(kay_check.comparison) == {"version_and_inputs"}
    assert "version-only Kay check" in runner.summary(record)


def test_with_no_kay_comparison_left_the_summary_claims_none(tmp_path, monkeypatch):
    """The direct Kay fails and Spyglass's inputs cannot be read: no comparison
    is left, no Kay-check table is written, and the summary prints no Kay check."""
    monkeypatch.setattr(
        swr_detectors,
        "Kay_ripple_detector",
        _raise(RuntimeError("simulated Kay failure")),
    )

    def unreadable(nwb_file_name, epoch_name):
        raise ValueError("Found 2 potential parents")

    fetchers = _fetchers(_main_epoch())
    fetchers["fetch_spyglass_kay_inputs"] = unreadable
    output_dir = tmp_path / "out"

    record = run_epoch(NWB_FILE_NAME, EPOCH_NAME, output_dir=output_dir, **fetchers)

    assert record["status"] == "partial"
    assert record["kay_version_check"]["counts"] == {}
    assert record["kay_version_check"]["differences"] == {}
    paths = _outputs(output_dir, swr_source.swr_epoch_id(NWB_FILE_NAME, EPOCH_NAME))
    assert not paths["kay_version_check"].exists()
    parts = runner.summary(record).split("; ")
    assert not any(part.startswith("Kay check") for part in parts)
    assert any("version-only Kay check did not run" in part for part in parts)


def test_a_version_check_of_another_algorithm_fails_the_epoch(tmp_path):
    """Canonical events that another algorithm made cannot check Kay."""
    parameters = {
        **SPYGLASS_KAY_PARAMETERS,
        "ripple_detection_algorithm": "Karlsson_ripple_detector",
    }

    record = run_epoch(
        NWB_FILE_NAME,
        EPOCH_NAME,
        output_dir=tmp_path / "out",
        **_fetchers(_main_epoch(), spyglass_parameters=parameters),
    )

    assert record["status"] == "failed"
    assert "Karlsson_ripple_detector" in record["error"]
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").glob("*"))


def test_a_failed_zugaro_preparation_leaves_the_epoch_partial(tmp_path, monkeypatch):
    """A Zugaro filter that raises is Zugaro's own error; the rest report.

    The channel was chosen, so the record keeps it beside the error.
    """
    error = RuntimeError("simulated: the Zugaro filter failed")
    monkeypatch.setattr(runner, "zero_phase_bandpass", _raise(error))
    output_dir = tmp_path / "out"

    record = run_epoch(
        NWB_FILE_NAME, EPOCH_NAME, output_dir=output_dir, **_fetchers(_main_epoch())
    )

    assert record["status"] == "partial"
    assert record["detectors"] == {
        "kay": "ok",
        "karlsson": "ok",
        "yu": "ok",
        "zugaro": repr(error),
    }
    assert record["zugaro_channel"]["electrode_id"] in ELECTRODE_IDS.tolist()
    assert record["zugaro_channel"]["error"] == repr(error)
    paths = _outputs(output_dir, swr_source.swr_epoch_id(NWB_FILE_NAME, EPOCH_NAME))
    assert not paths["zugaro"].exists()
    assert set(read_table(paths["swr_observed"]).scope) == {"stored_band"}


def test_zugaro_dropped_count_is_read_after_the_channel_error(tmp_path, monkeypatch):
    """With no usable channel Zugaro has no events, so no dropped count either,
    whatever Zugaro returns for the unobserved channel it is given."""
    error = NoUsableChannelError("simulated: every channel is below the guard")
    monkeypatch.setattr(runner, "best_ripple_channel", _raise(error))

    def zugaro(time, signal, speed, sampling_frequency, **settings):
        start = time[1_000] + np.arange(2)
        durations = [0.100, 0.200]
        return pd.DataFrame(
            {"start_time": start, "end_time": start + durations, "duration": durations}
        )

    monkeypatch.setattr(swr_detectors, "Zugaro_ripple_detector", zugaro)

    record = run_epoch(
        NWB_FILE_NAME,
        EPOCH_NAME,
        output_dir=tmp_path / "out",
        **_fetchers(_main_epoch()),
    )

    assert record["detectors"]["zugaro"] == repr(error)
    assert record["n_events"]["zugaro"] is None
    assert record["zugaro_dropped_over_150ms"] is None


def _varying_speed(seconds: NDArray) -> NDArray:
    """cm/s: 0.5 to 4.5, a 20 s cycle, at its fastest 5, 25 and 45 s in."""
    return 2.5 + 2.0 * np.sin(2 * np.pi * seconds / 20.0)


def test_head_speed_varying_over_time_is_interpolated(tmp_path):
    """Speed reaches the detectors linearly interpolated onto the LFP times.

    Each event's speed at its first and last sample is the position speed
    interpolated at that sample's time. The canonical ripples at 5, 25 and
    45 s fall where the speed tops 4 cm/s, so only the two at 15 and 35 s are
    speed-eligible, and the record's counts say so.
    """
    epoch = _main_epoch()
    fetchers = _fetchers(epoch, head_speed=_varying_speed)
    output_dir = tmp_path / "out"

    record = run_epoch(NWB_FILE_NAME, EPOCH_NAME, output_dir=output_dir, **fetchers)

    position_info = fetchers["fetch_position"](NWB_FILE_NAME, EPOCH_NAME)[
        "position_info"
    ]
    position_time = position_info.index.to_numpy()
    head_speed = position_info.head_speed.to_numpy()
    paths = _outputs(output_dir, swr_source.swr_epoch_id(NWB_FILE_NAME, EPOCH_NAME))
    for name in ("kay", "karlsson", "zugaro"):
        events = read_table(paths[name])
        assert len(events) > 0, name
        for bound in ("start", "end"):
            np.testing.assert_allclose(
                events[f"speed_at_{bound}"],
                np.interp(events[f"{bound}_time"], position_time, head_speed),
                rtol=0,
                atol=1e-12,
                err_msg=f"{name} {bound}",
            )
    counts = record["kay_version_check"]["counts"]["version_and_inputs"]
    canonical = counts["kay_spyglass"]
    assert sum(canonical["all"].values()) == len(epoch.windows)
    assert sum(canonical["speed_eligible"].values()) == 2
    kay_check = read_table(paths["kay_version_check"])
    for comparison, detectors in record["kay_version_check"]["counts"].items():
        for detector, by_scope in detectors.items():
            rows = kay_check[
                (kay_check.comparison == comparison)
                & (kay_check.detector == detector)
                & kay_check.whole_event_speed_eligible
            ]
            assert by_scope["speed_eligible"] == rows.status.value_counts().to_dict()


# ---------------------------------------------------------------------------
# Artifacts: the artifact mask
# ---------------------------------------------------------------------------

ARTIFACT_CORE_S = 15.0
"""Seconds after the first sample: an LFP step on the second injected ripple."""

ARTIFACT_MASK_S = (ARTIFACT_CORE_S - 0.3, ARTIFACT_CORE_S + 0.3)
"""Its interval in the artifact mask: an LFP step padded by 0.3 s on each side."""


def _artifact_run(tmp_path, identifier: str, cores=None) -> Path:
    """An artifact run whose cores for `identifier` are one ``ca1`` step at
    ``ARTIFACT_CORE_S``, unless `cores` says otherwise."""
    if cores is None:
        at = BASE_TIME_S + ARTIFACT_CORE_S
        cores = [("ca1", at, at)]
    return write_artifact_run(
        tmp_path,
        {identifier: cores},
        span=(BASE_TIME_S, BASE_TIME_S + 60.0),
    )


def _recording_detectors(monkeypatch) -> dict:
    """Wrap the four detectors so each call's ``normalization_mask`` is kept."""
    masks: dict = {}
    for name, attribute in {
        "kay": "Kay_ripple_detector",
        "karlsson": "Karlsson_ripple_detector",
        "yu": "Yu_ripple_detector",
        "zugaro": "Zugaro_ripple_detector",
    }.items():
        detector = getattr(swr_detectors, attribute)

        def recording(*args, _name=name, _detector=detector, **settings):
            masks[_name] = settings.get("normalization_mask")
            return _detector(*args, **settings)

        monkeypatch.setattr(swr_detectors, attribute, recording)
    return masks


def _overlapping(events: pd.DataFrame, lo: float, hi: float) -> pd.Series:
    """The events sharing any time with the closed interval ``[lo, hi]``."""
    return (events.start_time <= hi) & (events.end_time >= lo)


def test_one_artifact_mask_for_all_four_detectors(tmp_path, monkeypatch):
    """The ``swr`` mask leaves every detector's normalization the same way,
    every detector's events that overlap it are dropped and counted, and its
    time is not observed in either scope of ``swr_observed``."""
    epoch = _main_epoch()
    identifier = swr_source.swr_epoch_id(NWB_FILE_NAME, EPOCH_NAME)
    artifact_dir = _artifact_run(tmp_path, identifier)
    masks = _recording_detectors(monkeypatch)
    lo, hi = (BASE_TIME_S + bound for bound in ARTIFACT_MASK_S)
    unmasked_dir = tmp_path / "unmasked"
    unmasked = run_epoch(
        NWB_FILE_NAME, EPOCH_NAME, output_dir=unmasked_dir, **_fetchers(epoch)
    )
    output_dir = tmp_path / "out"

    record = run_epoch(
        NWB_FILE_NAME,
        EPOCH_NAME,
        output_dir=output_dir,
        artifact_dir=artifact_dir,
        **_fetchers(epoch),
    )

    assert record["status"] == "ok", record
    time = _band_inputs(epoch)["time"]
    in_mask = (time >= lo) & (time <= hi)
    assert set(masks) == set(DETECTORS)
    for name, mask in masks.items():
        # 2 cm/s throughout, so Yu's immobile sample is every sample too
        np.testing.assert_array_equal(mask, ~in_mask, err_msg=name)

    artifacts = record["artifacts"]
    assert artifacts["run_id"] == ARTIFACT_RUN_ID
    assert artifacts["n_intervals"] == 1
    # R-T's reference samples leave the mask out too
    assert record["tetrode_selection"]["n_reference"] == np.count_nonzero(~in_mask)
    # the masked, filtered Kay is compared with the unmasked canonical events
    assert record["kay_version_check"]["differences"] == {
        "version_only": ["package version"],
        "version_and_inputs": ["package version", "inputs", "artifact mask"],
    }
    assert unmasked["kay_version_check"]["differences"]["version_and_inputs"] == [
        "package version",
        "inputs",
    ]
    assert artifacts["masked_s"] == pytest.approx(0.6)
    paths = _outputs(output_dir, identifier)
    for name in DETECTORS:
        events = read_table(paths[name])
        assert not _overlapping(events, lo, hi).any(), name
        np.testing.assert_array_equal(events.index, np.arange(1, len(events) + 1))
        assert record["n_events"][name] == len(events), name
        # each detector finds the ripple under the step, and loses it
        before = read_table(_outputs(unmasked_dir, identifier)[name])
        assert _overlapping(before, lo, hi).any(), name
        assert artifacts["n_dropped"][name] >= 1, name
        assert unmasked["artifacts"]["n_dropped"][name] == 0, name

    observed = read_table(paths["swr_observed"])
    assert set(observed.scope) == {"stored_band", "zugaro"}
    for scope, rows in observed.groupby("scope"):
        assert not ((rows.start_s <= hi) & (rows.end_s >= lo)).any(), scope
    band = observed[observed.scope == "stored_band"]
    np.testing.assert_array_equal(
        band[["start_s", "end_s"]].to_numpy(),
        [[time[0], time[time < lo][-1]], [time[time > hi][0], time[-1]]],
    )


def test_without_an_artifact_run_nothing_is_masked(tmp_path, monkeypatch):
    """No artifact run: the detectors normalize over all their sample, nothing
    is dropped, and the record says no run was read."""
    masks = _recording_detectors(monkeypatch)

    record = run_epoch(
        NWB_FILE_NAME,
        EPOCH_NAME,
        output_dir=tmp_path / "out",
        **_fetchers(_main_epoch()),
    )

    assert record["status"] == "ok"
    for name, mask in masks.items():
        assert mask.all(), name
    assert record["artifacts"] == {
        "run_id": None,
        "n_intervals": 0,
        "masked_s": 0.0,
        "n_dropped": dict.fromkeys(DETECTORS, 0),
    }


def test_an_artifact_run_without_the_epoch_fails_it(tmp_path):
    """A configured run that never saw the epoch is an error, not an epoch
    without artifacts: the epoch fails and writes nothing."""
    artifact_dir = _artifact_run(tmp_path, "sim20200101_04_r2_HPC")
    output_dir = tmp_path / "out"

    record = run_epoch(
        NWB_FILE_NAME,
        EPOCH_NAME,
        output_dir=output_dir,
        artifact_dir=artifact_dir,
        **_fetchers(_main_epoch()),
    )

    assert record["status"] == "failed"
    assert "not in the artifact run" in record["error"]
    assert not any(
        path.exists()
        for path in _outputs(
            output_dir, swr_source.swr_epoch_id(NWB_FILE_NAME, EPOCH_NAME)
        ).values()
    )


ROGUE_RIPPLE_TIMES_S = tuple(np.arange(2.5, 58.0, 5.0))
"""Seconds after the first sample: a rogue tetrode's twelve own bursts, between
the shared ripples. More bursts than the others' five, so that Zugaro's
channel score (mean over median ripple-band RMS) prefers the rogue."""


def _epoch_with_rogues(rogue_columns: tuple[int, ...]) -> SimpleNamespace:
    """The main epoch with some band electrodes made rogue tetrodes.

    Each rogue's raw LFP is replaced by one channel of another simulated
    session, rippling at ``ROGUE_RIPPLE_TIMES_S`` (1 s later for a second
    rogue) and not at the shared ripples (the j16 pattern R-T drops). Its band
    is then the re-filtered raw, so the band check still holds.

    Parameters
    ----------
    rogue_columns : tuple of int
        Columns of the band (indices into ``ELECTRODE_IDS``).
    """
    epoch = _main_epoch()
    raw = epoch.raw.copy()
    for number, column in enumerate(rogue_columns):
        own = simulate_session(
            epoch.time,
            BASE_TIME_S + np.asarray(ROGUE_RIPPLE_TIMES_S) + number,
            n_channels=1,
            rng=10 + number,
            ripple_frequency=(150, 175),
        )
        raw[:, BAND_COLUMNS[column]] = np.round(
            COUNTS_PER_SIMULATED_UNIT * own.lfps[:, 0]
        ).astype(np.int16)
    return SimpleNamespace(time=epoch.time, raw=raw, windows=epoch.windows)


def _recording_signals(monkeypatch) -> dict:
    """Wrap Kay, Karlsson and Yu so each call's band is kept."""
    signals: dict = {}
    for name, attribute in {
        "kay": "Kay_ripple_detector",
        "karlsson": "Karlsson_ripple_detector",
        "yu": "Yu_ripple_detector",
    }.items():
        detector = getattr(swr_detectors, attribute)

        def recording(time, signal, *args, _name=name, _detector=detector, **kw):
            signals[_name] = np.array(signal)
            return _detector(time, signal, *args, **kw)

        monkeypatch.setattr(swr_detectors, attribute, recording)
    return signals


def test_rule_rt_chooses_the_tetrodes_every_detector_reads(tmp_path, monkeypatch):
    """R-T drops the rogue tetrode once, before any detector: Kay, Karlsson and
    Yu read the band of the kept tetrodes, Zugaro chooses its channel among
    them, and the record keeps every tetrode's coupling, the dropped group,
    and each kept electrode with its own reference."""
    rogue = 2
    epoch = _epoch_with_rogues((rogue,))
    inputs = _band_inputs(epoch)
    unselected = best_ripple_channel(
        inputs["raw_referenced"], inputs["time"], FS, ELECTRODE_IDS
    )
    assert unselected.electrode_id == ELECTRODE_IDS[rogue], (
        "the rogue must be the channel Zugaro would choose without R-T"
    )
    signals = _recording_signals(monkeypatch)

    record = run_epoch(
        NWB_FILE_NAME, EPOCH_NAME, output_dir=tmp_path / "out", **_fetchers(epoch)
    )

    assert record["status"] == "ok", record
    kept = np.arange(ELECTRODE_IDS.size) != rogue
    for name in ("kay", "karlsson", "yu"):
        np.testing.assert_array_equal(
            signals[name], inputs["ripple_band"][:, kept], err_msg=name
        )
    assert record["electrode_ids"] == ELECTRODE_IDS[kept].tolist()
    assert record["zugaro_channel"]["electrode_id"] in ELECTRODE_IDS[kept].tolist()
    assert set(record["zugaro_channel"]["scores"]) == {
        str(e) for e in ELECTRODE_IDS[kept]
    }
    # each kept electrode keeps its own reference; R-T drops no reference
    assert record["reference_ids"] == [16, 16, 81]
    selection = record["tetrode_selection"]
    assert selection["kept"] == kept.tolist()
    assert selection["dropped_electrode_groups"] == [ELECTRODE_GROUP_NAMES[rogue]]
    assert selection["dropped_electrode_ids"] == [int(ELECTRODE_IDS[rogue])]
    assert selection["coupling"][rogue] < 1.0
    assert min(np.delete(selection["coupling"], rogue)) >= 1.0
    # every band electrode, its group and its reference, in the band's order
    assert selection["electrode_ids"] == ELECTRODE_IDS.tolist()
    assert selection["electrode_group_names"] == ELECTRODE_GROUP_NAMES.tolist()
    assert selection["reference_ids"] == [16, 16, 81, 81]
    assert record["n_detecting_tetrodes"] == 3
    for name in DETECTORS:
        assert record["n_events"][name] > 0, name
    # the direct Kay reads fewer tetrodes than the canonical Spyglass events
    assert record["kay_version_check"]["differences"]["version_and_inputs"] == [
        "package version",
        "inputs",
        "tetrode selection (R-T)",
    ]


def test_rule_rt_failing_closed_fails_the_epoch(tmp_path):
    """Two rogues of four tetrodes is more than a third: the epoch fails with
    the reason and every tetrode's coupling, and writes nothing."""
    epoch = _epoch_with_rogues((0, 1))
    output_dir = tmp_path / "out"

    record = run_epoch(
        NWB_FILE_NAME, EPOCH_NAME, output_dir=output_dir, **_fetchers(epoch)
    )

    assert record["status"] == "failed"
    assert "more than a third" in record["reason"]
    selection = record["tetrode_selection"]
    assert {"0", "1"} <= set(selection["dropped_electrode_groups"])
    assert len(selection["coupling"]) == ELECTRODE_IDS.size
    assert not any(
        path.exists()
        for path in _outputs(
            output_dir, swr_source.swr_epoch_id(NWB_FILE_NAME, EPOCH_NAME)
        ).values()
    )


def test_two_band_channels_of_one_tetrode_fail_the_epoch(tmp_path):
    """R-T reads one band channel per tetrode: a sibling channel would count as
    "another tetrode" in the leave-one-out consensus. Such a band fails the
    epoch closed, with its reason, before R-T or any detector runs."""
    epoch = _main_epoch()
    inputs = _band_inputs(epoch)
    inputs["electrode_group_names"] = np.array(["0", "1", "2", "2"])
    output_dir = tmp_path / "out"

    record = run_epoch(
        NWB_FILE_NAME,
        EPOCH_NAME,
        output_dir=output_dir,
        **_fetchers(epoch, inputs=inputs),
    )

    assert record["status"] == "failed"
    assert "one band channel per tetrode" in record["reason"]
    assert "'2'" in record["reason"]
    assert "tetrode_selection" not in record
    assert not any(
        path.exists()
        for path in _outputs(
            output_dir, swr_source.swr_epoch_id(NWB_FILE_NAME, EPOCH_NAME)
        ).values()
    )


def test_rule_rt_keeps_every_tetrode_that_ripples_with_the_rest(tmp_path):
    record = run_epoch(
        NWB_FILE_NAME,
        EPOCH_NAME,
        output_dir=tmp_path / "out",
        **_fetchers(_main_epoch()),
    )

    selection = record["tetrode_selection"]
    assert selection["dropped_electrode_groups"] == []
    assert min(selection["coupling"]) >= 1.0
    assert record["electrode_ids"] == ELECTRODE_IDS.tolist()
    # 2 cm/s throughout and no mask: every sample is a reference sample
    assert selection["n_reference"] == _band_inputs(_main_epoch())["time"].size


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


@pytest.fixture
def in_tmp_path(tmp_path, monkeypatch):
    """What the runner writes relative to the working directory, such as the
    per-epoch logs, lands in the test's own folder."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def in_process_cluster(tmp_path, monkeypatch):
    """The runner's worker processes, replaced by two worker threads here.

    Yields the keyword arguments each ``LocalCluster`` was asked for.
    """
    asked = []

    def cluster(**kwargs):
        asked.append(kwargs)
        return LocalCluster(
            n_workers=2,
            threads_per_worker=1,
            processes=False,
            # A thread worker measures the whole test process, which can pass
            # dask's memory limit late in the suite and pause the worker for good.
            memory_limit=0,
            dashboard_address=":0",
            local_directory=str(tmp_path / "dask-scratch"),
        )

    monkeypatch.setattr(runner, "LocalCluster", cluster)
    # the scheduler keeps its scratch in dask's temporary directory
    with dask.config.set({"temporary-directory": str(tmp_path / "dask-temporary")}):
        yield asked


def _with_fakes(monkeypatch, failing_epoch: str | None = None) -> list:
    """Make the driver's ``run_epoch`` read fakes; return the epochs it runs.

    The epoch named ``failing_epoch`` fails to read its inputs.
    """
    real_run_epoch = runner.run_epoch
    ran = []

    def run_epoch_on_fakes(nwb_file_name, epoch_name, **kwargs):
        ran.append(epoch_name)
        fetchers = _fetchers(_main_epoch())
        if epoch_name == failing_epoch:

            def fetch_inputs(nwb_file_name, epoch_name):
                raise RuntimeError("simulated: the band could not be read")

            fetchers["fetch_inputs"] = fetch_inputs
        return real_run_epoch(nwb_file_name, epoch_name, **kwargs, **fetchers)

    monkeypatch.setattr(runner, "run_epoch", run_epoch_on_fakes)
    return ran


def _cli(tmp_path, *extra: str) -> dict:
    """``main`` over the sim animal; return the manifest."""
    output_root = tmp_path / "swr"
    main(
        [
            "--animal",
            "sim",
            "--run-id",
            "test-run",
            "--output-root",
            str(output_root),
            "--n-workers",
            "2",
            *extra,
        ]
    )
    return json.loads((output_root / "test-run" / "manifest.json").read_text())


def test_a_failed_epoch_is_recorded_and_the_rest_run(
    tmp_path, monkeypatch, in_tmp_path, in_process_cluster
):
    """Two epochs, one of which raises: one ok, one failed, a non-zero exit."""
    monkeypatch.setattr(
        runner,
        "epochs_to_run",
        lambda pending_only=False: _epoch_info(
            (NWB_FILE_NAME, "02_r1"), (NWB_FILE_NAME, "04_r2")
        ),
    )
    _with_fakes(monkeypatch, failing_epoch="04_r2")

    with pytest.raises(SystemExit) as stopped:
        _cli(tmp_path)

    assert stopped.value.code not in (0, None)
    assert in_process_cluster == [
        {"n_workers": 2, "processes": True, "threads_per_worker": 1}
    ]
    manifest = json.loads((tmp_path / "swr" / "test-run" / "manifest.json").read_text())
    epochs = manifest["epochs"]
    good, bad = (
        swr_source.swr_epoch_id(NWB_FILE_NAME, "02_r1"),
        swr_source.swr_epoch_id(NWB_FILE_NAME, "04_r2"),
    )
    assert set(epochs) == {good, bad}
    assert epochs[good]["status"] == "ok"
    assert epochs[bad]["status"] == "failed"
    assert epochs[bad]["error"] == repr(
        RuntimeError("simulated: the band could not be read")
    )
    assert manifest["settings"]["detectors"] == {
        name: dict(settings)
        for name, settings in swr_detectors.DETECTOR_SETTINGS.items()
    }
    assert manifest["settings"]["tetrode_selection"] == {
        "rule": "R-T",
        "coupling_cut": 1.0,
        "loo_threshold_z": 5.0,
        "loo_smoothing_sigma_s": 0.004,
        "reference_speed_cm_per_s": 4.0,
        "maximum_dropped_fraction": "1/3",
    }
    output_dir = tmp_path / "swr" / "test-run"
    assert all(path.exists() for path in _outputs(output_dir, good).values())
    assert not any(path.exists() for path in _outputs(output_dir, bad).values())
    for identifier in (good, bad):
        assert (tmp_path / "logs" / f"{identifier}_swr.log").exists()


def test_the_run_takes_its_epochs_from_the_epoch_register(
    tmp_path, monkeypatch, in_tmp_path, in_process_cluster
):
    """The epochs run are the register's analysed ones, narrowed by the flags:
    the runner keeps no list of epochs to skip, and no epoch the register does
    not analyse reaches it, so its manifest records none."""
    monkeypatch.setattr(
        runner,
        "epochs_to_run",
        lambda pending_only=False: _epoch_info(
            (NWB_FILE_NAME, "02_r1"), ("other20200101_.nwb", "02_r1")
        ),
    )
    ran = _with_fakes(monkeypatch)

    manifest = _cli(tmp_path)

    assert ran == ["02_r1"]
    assert set(manifest["epochs"]) == {swr_source.swr_epoch_id(NWB_FILE_NAME, "02_r1")}
    assert (
        manifest["epochs"][swr_source.swr_epoch_id(NWB_FILE_NAME, "02_r1")]["status"]
        == "ok"
    )
    assert not hasattr(runner, "KNOWN_NO_INPUT_EPOCHS")
    assert not hasattr(runner, "get_epoch_info")


@pytest.mark.parametrize(
    ("flags", "pending_only"), [([], False), (["--pending-only"], True)]
)
def test_pending_only_runs_the_registers_pending_epochs_alone(
    tmp_path, monkeypatch, in_tmp_path, in_process_cluster, flags, pending_only
):
    asked = []

    def epochs_to_run(pending_only=False):
        asked.append(pending_only)
        return _epoch_info((NWB_FILE_NAME, EPOCH_NAME))

    monkeypatch.setattr(runner, "epochs_to_run", epochs_to_run)
    _with_fakes(monkeypatch)

    manifest = _cli(tmp_path, *flags)

    assert asked == [pending_only]
    assert set(manifest["epochs"]) == {
        swr_source.swr_epoch_id(NWB_FILE_NAME, EPOCH_NAME)
    }


def test_rerun_without_overwrite_skips_done_epochs(
    tmp_path, monkeypatch, capsys, in_tmp_path, in_process_cluster
):
    """A finished epoch is not run again unless ``--overwrite`` is given."""
    monkeypatch.setattr(
        runner,
        "epochs_to_run",
        lambda pending_only=False: _epoch_info((NWB_FILE_NAME, EPOCH_NAME)),
    )
    ran = _with_fakes(monkeypatch)
    first = _cli(tmp_path)
    capsys.readouterr()

    again = _cli(tmp_path)

    assert "already written" in capsys.readouterr().out
    assert ran == [EPOCH_NAME]
    assert len(in_process_cluster) == 1
    assert again == first

    # an epoch about to be redone loses its record first, so a run stopped
    # part way cannot leave new files under the old record
    on_fakes, seen = runner.run_epoch, []

    def seeing_the_manifest(nwb_file_name, epoch_name, **kwargs):
        manifest = json.loads((kwargs["output_dir"] / "manifest.json").read_text())
        seen.append(set(manifest["epochs"]))
        return on_fakes(nwb_file_name, epoch_name, **kwargs)

    monkeypatch.setattr(runner, "run_epoch", seeing_the_manifest)

    overwritten = _cli(tmp_path, "--overwrite")

    assert ran == [EPOCH_NAME, EPOCH_NAME]
    assert seen == [set()]
    assert (
        overwritten["epochs"][swr_source.swr_epoch_id(NWB_FILE_NAME, EPOCH_NAME)][
            "status"
        ]
        == "ok"
    )


def test_a_rerun_with_other_settings_is_refused(
    tmp_path, monkeypatch, in_tmp_path, in_process_cluster
):
    """A run's epochs share one set of settings; a rerun under the same run
    id with other detector settings stops before running anything."""
    monkeypatch.setattr(
        runner,
        "epochs_to_run",
        lambda pending_only=False: _epoch_info((NWB_FILE_NAME, EPOCH_NAME)),
    )
    ran = _with_fakes(monkeypatch)
    _cli(tmp_path)
    changed = {name: dict(s) for name, s in swr_detectors.DETECTOR_SETTINGS.items()}
    changed["kay"]["zscore_threshold"] = 3.0
    monkeypatch.setattr(runner, "DETECTOR_SETTINGS", MappingProxyType(changed))

    with pytest.raises(SystemExit, match="other settings"):
        _cli(tmp_path, "--overwrite")

    assert ran == [EPOCH_NAME]


def test_the_run_reads_and_records_the_configured_artifact_run(
    tmp_path, monkeypatch, in_tmp_path, in_process_cluster
):
    """The artifact run is the SWR source's, read when the run starts: its id
    is in the run's settings, every epoch is masked by it, and a rerun under
    the same run id with another artifact run is refused."""
    identifier = swr_source.swr_epoch_id(NWB_FILE_NAME, EPOCH_NAME)
    artifact_dir = _artifact_run(tmp_path, identifier)
    use_artifact_run(monkeypatch, artifact_dir)
    monkeypatch.setattr(
        runner,
        "epochs_to_run",
        lambda pending_only=False: _epoch_info((NWB_FILE_NAME, EPOCH_NAME)),
    )
    ran = _with_fakes(monkeypatch)

    manifest = _cli(tmp_path)

    assert manifest["settings"]["artifacts"] == {
        "run_id": ARTIFACT_RUN_ID,
        "padding_s": {"S": 1.0, "T_lfp": 0.3, "T_spike": 0.1},
    }
    record = manifest["epochs"][identifier]
    assert record["status"] == "ok"
    assert record["artifacts"]["run_id"] == ARTIFACT_RUN_ID
    assert record["artifacts"]["n_intervals"] == 1

    monkeypatch.setattr(swr_source, "ARTIFACT_INTERVALS_RUN_ID", None)
    with pytest.raises(SystemExit, match="other settings"):
        _cli(tmp_path, "--overwrite")
    assert ran == [EPOCH_NAME]


_HOLD_LOCK = """
import fcntl, sys, time
handle = open(sys.argv[1], "a")
fcntl.lockf(handle, fcntl.LOCK_EX)
print("locked", flush=True)
time.sleep(120)
"""
"""Another process's run: it holds the run folder's lock until killed."""


def test_a_second_run_under_the_same_run_id_is_refused(
    tmp_path, monkeypatch, in_tmp_path, in_process_cluster
):
    """While another process holds the run's lock, a run under the same run
    id stops at once, before reading any epoch or writing the manifest; once
    the lock is free the run goes ahead and frees it again."""
    output_dir = tmp_path / "swr" / "test-run"
    output_dir.mkdir(parents=True)
    lock = output_dir / ".lock"

    def no_epochs_while_locked(pending_only=False):
        raise AssertionError("epochs were read without the run's lock")

    monkeypatch.setattr(runner, "epochs_to_run", no_epochs_while_locked)
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD_LOCK, str(lock)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(SystemExit) as stopped:
            _cli(tmp_path)
    finally:
        holder.kill()
        holder.wait()

    assert "Another run" in str(stopped.value.code)
    assert not (output_dir / "manifest.json").exists()

    monkeypatch.setattr(
        runner,
        "epochs_to_run",
        lambda pending_only=False: _epoch_info((NWB_FILE_NAME, EPOCH_NAME)),
    )
    _with_fakes(monkeypatch)
    manifest = _cli(tmp_path)

    assert (
        manifest["epochs"][swr_source.swr_epoch_id(NWB_FILE_NAME, EPOCH_NAME)]["status"]
        == "ok"
    )
    take_lock = "import fcntl, sys; fcntl.lockf(open(sys.argv[1], 'a'), fcntl.LOCK_EX | fcntl.LOCK_NB)"
    subprocess.run([sys.executable, "-c", take_lock, str(lock)], check=True)


def test_a_lock_that_cannot_be_taken_raises_as_itself(
    tmp_path, monkeypatch, in_tmp_path
):
    """Only a lock held by another run reads as one; a file system that cannot
    lock at all raises its own error."""

    def no_locks(handle, operation):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(batch.fcntl, "lockf", no_locks)

    with pytest.raises(OSError, match="No locks available"):
        _cli(tmp_path)


@pytest.mark.parametrize("run_id", ["../escape", ".", "..", "a/b", "with space", ""])
def test_run_id_must_be_a_plain_name(tmp_path, monkeypatch, in_tmp_path, run_id):
    """A run id names one folder under the output root, nothing else."""

    def must_not_read(*args):
        raise AssertionError("the run started")

    monkeypatch.setattr(runner, "epochs_to_run", must_not_read)

    with pytest.raises(SystemExit) as stopped:
        main(["--run-id", run_id, "--output-root", str(tmp_path / "swr")])

    assert stopped.value.code == 2
    assert not (tmp_path / "swr").exists()
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("outcome", ["ok", "raises"])
def test_each_epoch_task_closes_the_nwb_files_spyglass_opened(
    tmp_path, monkeypatch, outcome
):
    """A worker runs one epoch at a time, so when its epoch ends, pass or fail,
    it closes the NWB files Spyglass cached for it and logs that it did."""
    closed = []

    def release_nwb_files():
        closed.append(True)
        return True

    def run_epoch(nwb_file_name, epoch_name, **kwargs):
        if outcome == "raises":
            raise RuntimeError("simulated: the task itself failed")
        return {"status": "skipped", "reason": "simulated"}

    monkeypatch.setattr(runner, "run_epoch", run_epoch)
    monkeypatch.setattr(batch, "release_nwb_files", release_nwb_files)

    def task():
        return runner._run_epoch_logged(
            NWB_FILE_NAME,
            EPOCH_NAME,
            {"output_dir": tmp_path / "out"},
            tmp_path / "logs",
        )

    if outcome == "raises":
        with pytest.raises(RuntimeError, match="the task itself failed"):
            task()
    else:
        assert task()["status"] == "skipped"

    assert closed == [True]
    log = (
        tmp_path
        / "logs"
        / f"{swr_source.swr_epoch_id(NWB_FILE_NAME, EPOCH_NAME)}_swr.log"
    )
    assert any(
        line.endswith("NWB files Spyglass held open closed: True")
        for line in log.read_text().splitlines()
    )


def test_main_raises_the_open_file_limit_before_its_cluster(
    tmp_path, monkeypatch, capsys, in_tmp_path, in_process_cluster
):
    """The worker processes inherit the limit only if it is raised before they
    start; the run says what it was raised from and to."""
    clusters_at_raise = []

    def raise_open_file_limit():
        clusters_at_raise.append(len(in_process_cluster))
        return 1024, 65536

    monkeypatch.setattr(runner, "raise_open_file_limit", raise_open_file_limit)
    monkeypatch.setattr(
        runner,
        "epochs_to_run",
        lambda pending_only=False: _epoch_info((NWB_FILE_NAME, EPOCH_NAME)),
    )
    _with_fakes(monkeypatch)

    _cli(tmp_path)

    assert clusters_at_raise == [0]
    assert len(in_process_cluster) == 1
    assert "open-file soft limit 1024 -> 65536" in capsys.readouterr().out
