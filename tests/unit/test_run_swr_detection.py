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

import hashlib
import json
from functools import cache
from types import MappingProxyType, SimpleNamespace

import dask
import numpy as np
import pandas as pd
import pytest
from dask.distributed import LocalCluster
from numpy.typing import NDArray
from ripple_detection import Kay_ripple_detector, filter_ripple_band
from ripple_detection.simulate import simulate_session, simulate_time

import scripts.decoding.run_swr_detection as runner
from continuum_swr_replay.analysis.hippocampus import swr_inventories
from continuum_swr_replay.analysis.hippocampus.swr_inventories import (
    NoUsableChannelError,
    best_ripple_channel,
)
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename
from continuum_swr_replay.table_io import read_attrs, read_table, write_table
from scripts.decoding.run_swr_detection import (
    KNOWN_NO_INPUT_EPOCHS,
    epoch_id,
    main,
    run_epoch,
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
        "lfp_band_elect_index": BAND_COLUMNS,
        "band_valid_interval": interval,
        "decimation": 1,
        "sampling_frequency": FS,
        "band_filter_coeff": np.ones(3),
        "band_filter_name": "ripple_150_250",
        "lfp_filter_name": "LFP 0-400 Hz",
        "lfp_units": "volts",
    }


def _fetchers(
    epoch: SimpleNamespace,
    *,
    position_span_s: tuple[float, float] | None = None,
    inputs: dict | None = None,
    calls: list | None = None,
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
    """
    inputs = _band_inputs(epoch) if inputs is None else inputs
    time = inputs["time"]
    if position_span_s is None:
        start, end = time[0], time[-1]
    else:
        start, end = (BASE_TIME_S + bound for bound in position_span_s)
    position_time = np.linspace(start, end, round((end - start) * POSITION_RATE_HZ) + 1)
    position_info = pd.DataFrame(
        {"head_position_x": 0.0, "head_position_y": 0.0, "head_speed": SPEED_CM_PER_S},
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
    return paths


def _epoch_info(*epochs: tuple[str, str]) -> pd.DataFrame:
    """``get_epoch_info``'s frame for these (nwb_file_name, epoch_name) pairs."""
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
    identifier = epoch_id(NWB_FILE_NAME, EPOCH_NAME)

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
    assert record["position_span_s"] == [epoch.time[0], epoch.time[-1]]
    assert (
        record["zugaro_dropped_over_150ms"]
        == read_attrs(paths["zugaro"])["n_dropped_over_maximum_duration"]
    )
    assert isinstance(record["zugaro_dropped_over_150ms"], int)
    assert record["n_detecting_tetrodes"] == ELECTRODE_IDS.size
    assert record["meets_published_tetrode_criterion"] == "unknown"
    assert set(record["input_sha256"]) == {"time", "ripple_band", "raw_referenced"}
    assert (
        record["input_sha256"]["time"]
        == hashlib.sha256(epoch.time.astype(np.float64).tobytes()).hexdigest()
    )


def test_known_no_input_epochs_are_skipped_without_fetching(tmp_path):
    """A documented epoch with no inputs is skipped before any fetch."""

    def must_not_fetch(*args, **kwargs):
        raise AssertionError(f"fetched {args}")

    record = run_epoch(
        "senor20201105_.nwb",
        "09_r4",
        output_dir=tmp_path / "out",
        fetch_inputs=must_not_fetch,
        fetch_position=must_not_fetch,
        fetch_canonical_kay=must_not_fetch,
        fetch_spyglass_kay_inputs=must_not_fetch,
        fir_filter=must_not_fetch,
    )

    assert record == {
        "status": "skipped",
        "reason": KNOWN_NO_INPUT_EPOCHS[("senor20201105_.nwb", "09_r4")],
    }
    assert not (tmp_path / "out").exists()
    assert set(KNOWN_NO_INPUT_EPOCHS) == {
        ("senor20201104_.nwb", "06_r3"),
        ("senor20201104_.nwb", "08_r4"),
        ("senor20201105_.nwb", "09_r4"),
        *(
            ("wilbur20210404_.nwb", name)
            for name in ("02_r1", "04_r2", "06_r3", "08_r4", "10_r5")
        ),
    }


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
    identifier = epoch_id(NWB_FILE_NAME, EPOCH_NAME)
    paths = _outputs(output_dir, identifier)
    if failing == "karlsson":
        error = RuntimeError("simulated Karlsson failure")
        monkeypatch.setattr(swr_inventories, "Karlsson_ripple_detector", _raise(error))
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
        monkeypatch.setattr(swr_inventories, detector, _raise(error))

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
    stale = _outputs(output_dir, epoch_id(NWB_FILE_NAME, EPOCH_NAME))["kay"]
    write_table(pd.DataFrame({"start_time": [1.0]}), stale)

    record = run_epoch(NWB_FILE_NAME, EPOCH_NAME, output_dir=output_dir, **fetchers)

    json.dumps(record, allow_nan=False)
    assert record["status"] == "failed"
    assert record["error"] == repr(error)
    assert "fetch_position" in record["traceback"]
    assert not stale.exists()


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
        _outputs(output_dir, epoch_id(NWB_FILE_NAME, EPOCH_NAME))["swr_observed"]
    )
    zugaro = observed.loc[observed.scope == "zugaro", ["start_s", "end_s"]]
    edge = swr_inventories.settling_samples(
        swr_inventories.design_bandpass_sos(
            FS, swr_inventories.ZUGARO_BAND_HZ, swr_inventories.ZUGARO_FILTER_ORDER
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
    identifier = epoch_id(NWB_FILE_NAME, EPOCH_NAME)
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
    speed, ripple_lfps, sampling_frequency = fetchers["fetch_spyglass_kay_inputs"](
        NWB_FILE_NAME, EPOCH_NAME
    )
    on_spyglass_inputs = Kay_ripple_detector(
        np.asarray(ripple_lfps.index),
        np.asarray(ripple_lfps),
        np.asarray(speed),
        sampling_frequency,
        **swr_inventories.DETECTOR_SETTINGS["kay"],
    )
    inside = (on_spyglass_inputs.start_time >= span[0]) & (
        on_spyglass_inputs.end_time <= span[1]
    )
    assert 0 < inside.sum() < len(on_spyglass_inputs)
    version_only = (kay_check.detector == "kay") & (
        kay_check.comparison == "version_only"
    )
    assert version_only.sum() == inside.sum()


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

    monkeypatch.setattr(swr_inventories, "Zugaro_ripple_detector", zugaro)

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
    speed, ripple_lfps, sampling_frequency = fetchers["fetch_spyglass_kay_inputs"](
        NWB_FILE_NAME, EPOCH_NAME
    )
    fetchers["fetch_spyglass_kay_inputs"] = lambda nwb_file_name, epoch_name: (
        5 * speed,
        ripple_lfps,
        sampling_frequency,
    )
    output_dir = tmp_path / "out"

    record = run_epoch(NWB_FILE_NAME, EPOCH_NAME, output_dir=output_dir, **fetchers)

    paths = _outputs(output_dir, epoch_id(NWB_FILE_NAME, EPOCH_NAME))
    kay_check = read_table(paths["kay_version_check"])
    direct = kay_check.loc[kay_check.detector == "kay", "comparison"]
    assert (direct == "version_only").sum() == 0
    assert (direct == "version_and_inputs").sum() == record["n_events"]["kay"] > 0
    assert record["kay_version_check"]["version_only"] == {
        "kay_spyglass": {"missed": len(_main_epoch().windows)}
    }


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
        "get_epoch_info",
        lambda: _epoch_info((NWB_FILE_NAME, "02_r1"), (NWB_FILE_NAME, "04_r2")),
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
    good, bad = epoch_id(NWB_FILE_NAME, "02_r1"), epoch_id(NWB_FILE_NAME, "04_r2")
    assert set(epochs) == {good, bad}
    assert epochs[good]["status"] == "ok"
    assert epochs[bad]["status"] == "failed"
    assert epochs[bad]["error"] == repr(
        RuntimeError("simulated: the band could not be read")
    )
    assert manifest["settings"]["detectors"] == {
        name: dict(settings)
        for name, settings in swr_inventories.DETECTOR_SETTINGS.items()
    }
    output_dir = tmp_path / "swr" / "test-run"
    assert all(path.exists() for path in _outputs(output_dir, good).values())
    assert not any(path.exists() for path in _outputs(output_dir, bad).values())
    for identifier in (good, bad):
        assert (tmp_path / "logs" / f"{identifier}_swr.log").exists()


def test_rerun_without_overwrite_skips_done_epochs(
    tmp_path, monkeypatch, capsys, in_tmp_path, in_process_cluster
):
    """A finished epoch is not run again unless ``--overwrite`` is given."""
    monkeypatch.setattr(
        runner, "get_epoch_info", lambda: _epoch_info((NWB_FILE_NAME, EPOCH_NAME))
    )
    ran = _with_fakes(monkeypatch)
    first = _cli(tmp_path)
    capsys.readouterr()

    again = _cli(tmp_path)

    assert "already written" in capsys.readouterr().out
    assert ran == [EPOCH_NAME]
    assert len(in_process_cluster) == 1
    assert again == first

    _cli(tmp_path, "--overwrite")

    assert ran == [EPOCH_NAME, EPOCH_NAME]


def test_a_rerun_with_other_settings_is_refused(
    tmp_path, monkeypatch, in_tmp_path, in_process_cluster
):
    """A run's epochs share one set of settings; a rerun under the same run
    id with other detector settings stops before running anything."""
    monkeypatch.setattr(
        runner, "get_epoch_info", lambda: _epoch_info((NWB_FILE_NAME, EPOCH_NAME))
    )
    ran = _with_fakes(monkeypatch)
    _cli(tmp_path)
    changed = {name: dict(s) for name, s in swr_inventories.DETECTOR_SETTINGS.items()}
    changed["kay"]["zscore_threshold"] = 3.0
    monkeypatch.setattr(runner, "DETECTOR_SETTINGS", MappingProxyType(changed))

    with pytest.raises(SystemExit, match="other settings"):
        _cli(tmp_path, "--overwrite")

    assert ran == [EPOCH_NAME]


def test_an_error_its_worker_cannot_return_is_recorded(tmp_path, monkeypatch):
    """A task that dies instead of returning its record is recorded as failed,
    and the other epoch still finishes."""

    def task(nwb_file_name, epoch_name, run_epoch_kwargs, log_directory):
        if epoch_name == "04_r2":
            raise RuntimeError("simulated: the worker died")
        return {"status": "ok"}

    monkeypatch.setattr(runner, "_run_epoch_logged", task)
    monkeypatch.chdir(tmp_path)
    with dask.config.set({"temporary-directory": str(tmp_path / "dask-temporary")}):
        records = dict(
            runner.run_epochs(
                [(NWB_FILE_NAME, "02_r1"), (NWB_FILE_NAME, "04_r2")],
                {"output_dir": tmp_path / "out"},
                lambda: LocalCluster(
                    n_workers=2,
                    processes=False,
                    dashboard_address=":0",
                    local_directory=str(tmp_path / "dask-scratch"),
                ),
            )
        )

    failed = records.pop(epoch_id(NWB_FILE_NAME, "04_r2"))
    assert failed["status"] == "failed"
    assert failed["error"] == repr(RuntimeError("simulated: the worker died"))
    assert records == {epoch_id(NWB_FILE_NAME, "02_r1"): {"status": "ok"}}
