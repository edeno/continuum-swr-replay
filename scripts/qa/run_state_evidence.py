#!/usr/bin/env python
"""Held-out spike evidence for each decoder state, on every canonical HPC decode.

Does each state capture what it claims, judged by spikes that did not choose
the location? The method, its evidence columns and its caveats are in
``continuum_swr_replay.analysis.hippocampus.state_evidence``; this runner feeds
it one epoch at a time and summarizes the pooled result.

Per epoch (a dask CPU worker each):

1. **Decode.** ``{epoch}_80_data_over_time.pkl`` gives the time axis, head
   position and speed, linear position, the four state probabilities and the
   non-local event labels; ``{epoch}_detector.pkl`` the fitted encoding model
   and ``no_spike_rate``. The probabilities must be finite and sum to 1.
2. **Windows.** Each state's confident bouts and their matched windows, and the
   calibration windows (``build_windows``), seeded by ``(SEED, epoch_id)``.
3. **Marks** and the decode's ``no_ripple`` training mask, through the
   project's loader (``load_data``; read-only Spyglass). Its time axis must be
   the decode's, and refitting the encoding model on it with that mask must
   reproduce the stored one exactly, so the marks are the decode's.
4. **Encodings.** Primary: a running-only refit (head speed >= 4 cm/s); no
   event bin may be in it. Secondary: the decode's own.
5. **Per-tetrode terms** for the spikes in the scored bins, under both
   encodings. Checked twice: summed over all tetrodes they reproduce the stored
   ``results.nc`` log-likelihood on 2000 sampled bins (canonical), and the
   library's own likelihood (``event_likelihood``) on a 200-bin block (both).
6. **Evidence** under both encodings (``window_evidence``), 10 seeded tetrode
   half-partitions x 2 directions.

It writes ``per_epoch/{epoch}_windows.parquet`` (one row per window, columns
``run_*`` for the running-only encoding and ``can_*`` for the canonical one)
and ``per_epoch/{epoch}_validation.json``. An epoch whose decode files are
missing is recorded as skipped with that reason; one that fails a check, or
any step, as failed with its error, and the rest run. A rerun reuses an
epoch already written with the same settings unless ``--overwrite``.

Then, over every epoch written: ``bout_stats.csv``, ``calibration_stats.csv``,
``trend_stats.csv``, ``rate_calibration.csv`` and ``silence_stats.csv``
(``summarize_state_evidence``), three figures (bouts against matched windows,
calibration curves, coordinated silence), ``manifest.json`` (settings,
provenance and every epoch's record) and a ``README.md`` with the caveats.

It writes nothing outside ``--output-dir``. Per-epoch logs go to its ``logs/``.

Usage:
    python scripts/qa/run_state_evidence.py --n-workers 16
    python scripts/qa/run_state_evidence.py --epochs chimi20200225_02_r1_HPC \\
        --decode-dir Processed-Data/_staging_redecode_20261004 \\
        --output-dir /tmp/state_evidence --n-workers 1
"""

import os

# a CPU job: it must never take a GPU from the decodes
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time as clock  # noqa: E402
import traceback  # noqa: E402
from collections.abc import Callable, Iterator  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (  # noqa: E402
    interior_bin_distances_cm,
)
from continuum_swr_replay.analysis.hippocampus.state_evidence import (  # noqa: E402
    ENCODING_KEY,
    MIN_BIN_WINDOWS,
    PROBABILITY_BIN_LABELS,
    STATES,
    TetrodeTerms,
    build_windows,
    encoding_inputs,
    event_likelihood,
    interior_bin_mask,
    keyed_rng,
    load_detector,
    nearest_interior_bins,
    parse_epoch_identifier,
    per_tetrode_terms,
    regimes,
    summarize_state_evidence,
    tetrode_halves,
    window_evidence,
)
from continuum_swr_replay.data_loaders.utils import (  # noqa: E402
    get_training_timepoints,
    parse_nwb_filename,
)
from continuum_swr_replay.decoding_common import (  # noqa: E402
    provenance_record,
    setup_logger,
    write_atomically,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR, ROOT_DIR  # noqa: E402
from continuum_swr_replay.table_io import read_table, write_table  # noqa: E402

PRODUCED_BY = "scripts/qa/run_state_evidence.py"
SEED = 20261004
"""Names every window and partner draw, with the epoch id."""
HALVES_SEED = 20261003
"""The tetrode partitions' seed, the prototype's (its pilot splits are these)."""
RUNNING_ENCODING_MIN_SPEED_CM_S = 4.0
N_STORED_CHECK_BINS = 2000
LIBRARY_CHECK_BINS = 200
STORED_MAX_REL = 1e-3
"""The decode's float32 GPU log-likelihood against these CPU terms: 3.6e-5 at
most on the prototype's pilot; a mark the decode did not use moves it by far more."""
LIBRARY_MAX_REL = 1e-3
"""The library's likelihood reads the decode's stored summed ground intensity,
computed on its GPU; the terms recompute it on the CPU. On the pilot they
differ by up to 7.9e-5 (canonical encoding; 2e-7 for the CPU-fit running-only
one), so a tighter bound would fail an epoch on float error alone."""
PROBABILITY_SUM_TOLERANCE = 1e-3
PROBABILITY_COLUMNS = [f"{state.replace(' ', '_')}_prob" for state in STATES]
SHARED_COLUMNS = ["n_spikes", "n_tetrodes", "animal_center_bin", "has_loc_partner"]
ENCODINGS = ("run", "can")

CAVEATS = [
    "Only the remote location is held out: S alone chooses MAP_S. The bouts and "
    "probability bins come from the all-tetrode decode, so C's spikes helped make "
    "every state label.",
    "Run-averaged Local and No-Spike values (loc_*, ns_*) equal half the "
    "all-tetrode, in-sample values: no half chooses the animal's position, and "
    "each tetrode scores in exactly half of the 20 runs.",
    "One spike costs about 29 nats under No-Spike (log(1e-10 Hz x 2 ms)), so a "
    "confident No-Spike label forces both halves silent; only the size of E_C "
    "(ns_expected) is informative in No-Spike bouts. The coordinated-silence "
    "test over random windows is the only held-out No-Spike test.",
    "With few spikes, S's silence term pulls MAP_S toward low-rate bins, so "
    "MAP_S's distance from the animal (mapS_dist_cm) is not a Local test.",
    "Inference is at the epoch level, epochs nested in animals: hierarchical "
    "bootstrap CIs (animals, then epochs within each; Saravanan, Berman & Sober "
    "2020), sign-flip tests on per-epoch means (exact for 13 epochs or fewer) "
    "and k of N epochs and k of A animals. Columns ending _window_level treat "
    "windows as independent: they ignore the nesting, are not for inference, "
    "and are kept only to compare with the prototype.",
]


# --- inputs ----------------------------------------------------------------------------
def decode_files(epoch_id: str, decode_dir: Path) -> dict[str, Path]:
    """The canonical decode's three files for an epoch."""
    return {
        "data_over_time": decode_dir / f"{epoch_id}_80_data_over_time.pkl",
        "detector": decode_dir / f"{epoch_id}_detector.pkl",
        "results": decode_dir / f"{epoch_id}_results.nc",
    }


def canonical_epochs(decode_dir: Path) -> list[str]:
    """Every HPC epoch with any canonical decode file in ``decode_dir``.

    An epoch missing one of its files is still listed, so that the run records
    why it skips it.
    """
    stems = set()
    for suffix in ("_80_data_over_time.pkl", "_detector.pkl", "_results.nc"):
        for path in decode_dir.glob(f"*_HPC{suffix}"):
            stems.add(path.name.removesuffix(suffix))
    return sorted(stems)


def load_marks_and_training_mask(epoch_id: str) -> dict:
    """Marks, the position axis and the ``no_ripple`` mask, from ``load_data``.

    Returns
    -------
    dict
        ``time`` (the loader's position axis, s), ``spike_times`` and
        ``marks`` (per tetrode) and ``is_training`` (bool per time bin).
    """
    from continuum_swr_replay.data_loaders import load_data
    from continuum_swr_replay.types import SessionDataKeys as K

    nwb_file_name, epoch_name, brain_area = parse_epoch_identifier(epoch_id)
    data = load_data(
        nwb_file_name,
        epoch_name,
        include=frozenset({K.SPIKE_TIMES, K.SPIKE_WAVEFORM_FEATURES, K.RIPPLE_TIMES}),
    )
    return {
        "time": data["position_info"].index.to_numpy(dtype=float),
        "spike_times": [np.asarray(x) for x in data["spike_times"][brain_area]],
        "marks": [np.asarray(x) for x in data["spike_waveform_features"][brain_area]],
        "is_training": np.asarray(
            get_training_timepoints(data, training_type="no_ripple"), dtype=bool
        ),
    }


def stored_log_likelihood(
    path: Path, time_bins: NDArray, interior: NDArray[np.bool_]
) -> NDArray:
    """The decode's stored Non-Local Continuous log-likelihood on some bins.

    Returns
    -------
    ndarray, shape (n_bins, n_interior_bins)
        The prior-excluded clusterless log-likelihood the decode used.
    """
    import xarray as xr

    with xr.open_dataset(path, engine="h5netcdf") as results:  # as the decode wrote it
        state = results["state"].values
        idx = np.flatnonzero(state == "Non-Local Continuous")
        if idx.size != interior.size or not (np.diff(idx) == 1).all():
            raise ValueError(f"unexpected state_bins layout in {path}")
        stored = (
            results["log_likelihood"]
            .isel(time=np.asarray(time_bins), state_bins=slice(idx[0], idx[-1] + 1))
            .values
        )
    return np.asarray(stored, dtype=np.float64)[:, interior]


def encoding_comparison(refit: dict, stored: dict) -> dict:
    """How a refit encoding model differs from the stored one."""

    def max_abs(a: list, b: list) -> float:
        return max(
            (
                float(np.max(np.abs(np.asarray(x, float) - np.asarray(y, float))))
                for x, y in zip(a, b, strict=True)
                if np.size(x)
            ),
            default=0.0,
        )

    n_refit = [len(x) for x in refit["encoding_positions"]]
    n_stored = [len(x) for x in stored["encoding_positions"]]
    same = n_refit == n_stored
    return {
        "same_n_encoding_spikes": same,
        "n_encoding_spikes": n_stored,
        "max_abs_encoding_positions": (
            max_abs(refit["encoding_positions"], stored["encoding_positions"])
            if same
            else None
        ),
        "max_abs_encoding_marks": (
            max_abs(
                refit["encoding_spike_waveform_features"],
                stored["encoding_spike_waveform_features"],
            )
            if same
            else None
        ),
        "max_abs_mean_rates": float(
            np.max(
                np.abs(
                    np.asarray(refit["mean_rates"]) - np.asarray(stored["mean_rates"])
                )
            )
        ),
    }


def refit_encoding(
    detector: Any,
    time: NDArray,
    position: NDArray,
    spike_times: list[NDArray],
    marks: list[NDArray],
    is_training: NDArray,
) -> dict:
    """The decode's own encoding model, refit on other training bins.

    A copy of the detector is refit (``fit_encoding_model``: the same
    clusterless KDE, settings and environment); the detector is not changed.
    """
    refit = copy.deepcopy(detector)
    refit.clusterless_algorithm_params = {
        **(refit.clusterless_algorithm_params or {}),
        "disable_progress_bar": True,
    }
    refit.fit_encoding_model(
        time, position, spike_times, marks, is_training=is_training
    )
    encoding: dict = refit.encoding_model_[ENCODING_KEY]
    encoding["environment"]._is_fitted = True
    return encoding


def spikes_in_bins(
    spike_times: list[NDArray], marks: list[NDArray], time: NDArray, keep: NDArray
) -> tuple[list[NDArray], list[NDArray]]:
    """Only the spikes the decoder bins into a kept bin (``keep``, per bin)."""
    kept_times, kept_marks = [], []
    for st, mk in zip(spike_times, marks, strict=True):
        st = np.asarray(st)
        inside = np.flatnonzero((st >= time[0]) & (st <= time[-1]))
        mine = inside[keep[np.digitize(st[inside], time[1:-1])]]
        kept_times.append(st[mine])
        kept_marks.append(np.asarray(mk)[mine])
    return kept_times, kept_marks


def relative_error(mine: NDArray, reference: NDArray) -> dict:
    """Max absolute and relative (to max(1, |reference|)) difference."""
    diff = np.abs(mine - reference)
    return {
        "max_abs": float(diff.max()),
        "max_rel": float((diff / np.maximum(1.0, np.abs(reference))).max()),
    }


# --- one epoch -----------------------------------------------------------------------------
def output_paths(epoch_id: str, output_dir: Path) -> dict[str, Path]:
    return {
        "windows": output_dir / "per_epoch" / f"{epoch_id}_windows.parquet",
        "validation": output_dir / "per_epoch" / f"{epoch_id}_validation.json",
    }


def write_json(path: Path, content: object) -> None:
    text = json.dumps(content, indent=1, allow_nan=False, default=str)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, lambda temporary: temporary.write_text(text))


def _json_ready(value: Any) -> Any:
    """``value`` with numpy scalars as Python ones and NaN as None."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def epoch_table(
    epoch_id: str,
    decode_dir: Path,
    load_inputs: Callable[[str], dict],
    log: Callable[[str], None],
) -> tuple[pd.DataFrame, dict]:
    """One epoch's per-window table and its validation record.

    Raises
    ------
    ValueError
        A check failed; the message says which.
    """
    checks: dict[str, Any] = {}
    files = decode_files(epoch_id, decode_dir)
    dot = pd.read_pickle(files["data_over_time"])
    time = dot.index.to_numpy(dtype=float)
    speed = dot["head_speed"].to_numpy(dtype=float)
    linear = dot["linear_position"].to_numpy(dtype=float)
    position = dot[["head_position_x", "head_position_y"]].to_numpy(dtype=float)
    probabilities = dot[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    in_event = dot["non_local_event"].to_numpy() > 0
    del dot
    worst = float(np.max(np.abs(probabilities.sum(1) - 1.0)))
    if not np.isfinite(probabilities).all() or worst > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(f"state probabilities are invalid (max |sum - 1| {worst})")
    bin_s = float(np.median(np.diff(time)))
    checks["decode"] = {
        "decode_dir": str(decode_dir),
        "n_time": int(time.size),
        "bin_s": bin_s,
        "max_abs_probability_sum_error": worst,
    }

    # windows: a bin with no finite position is in no regime
    regime = regimes(speed)
    regime[~np.isfinite(linear)] = ""
    windows = build_windows(probabilities, regime, seed=(SEED, epoch_id))
    log(f"{len(windows)} windows")

    detector = load_detector(epoch_id, decode_dir)
    stored = detector.encoding_model_[ENCODING_KEY]
    environment = stored["environment"]
    interior = interior_bin_mask(environment)
    centers = np.asarray(environment.place_bin_centers_)[interior]
    distance = interior_bin_distances_cm(environment)
    no_spike_rate = float(detector.no_spike_rate)
    checks["decode"].update(
        no_spike_rate=no_spike_rate,
        sampling_frequency=float(detector.sampling_frequency),
        n_place_bins=int(interior.size),
        n_real_place_bins=int(interior.sum()),
    )

    # every bin a window scores, and its animal position
    s, e = windows.s.to_numpy(), windows.e.to_numpy()
    scored = np.zeros(time.size + 1, dtype=np.int64)
    np.add.at(scored, s, 1)
    np.add.at(scored, e + 1, -1)
    in_window = np.cumsum(scored[:-1]) > 0
    animal = np.full(time.size, -1, dtype=np.int64)
    animal[in_window] = nearest_interior_bins(environment, linear[in_window])

    # the decode's marks: same axis, and the stored encoding model refits exactly
    inputs = load_inputs(epoch_id)
    if not np.array_equal(inputs["time"], time):
        raise ValueError("the loader's time axis is not the decode's (a stale decode?)")
    spike_times, marks = inputs["spike_times"], inputs["marks"]
    same = encoding_comparison(
        refit_encoding(
            detector, time, position, spike_times, marks, inputs["is_training"]
        ),
        stored,
    )
    checks["canonical_refit"] = same
    if not same["same_n_encoding_spikes"] or any(
        same[k] != 0.0
        for k in (
            "max_abs_encoding_positions",
            "max_abs_encoding_marks",
            "max_abs_mean_rates",
        )
    ):
        raise ValueError(
            "refitting with the no_ripple mask does not reproduce the stored "
            f"encoding model, so the marks are not the decode's: {same}"
        )

    # the running-only encoding: no event bin may be a training bin
    running = speed >= RUNNING_ENCODING_MIN_SPEED_CM_S
    n_event_running = int((running & in_event).sum())
    if n_event_running:
        raise ValueError(f"{n_event_running} non-local event bins are running bins")
    running_only = refit_encoding(detector, time, position, spike_times, marks, running)
    checks["running_only"] = {
        "min_speed_cm_s": RUNNING_ENCODING_MIN_SPEED_CM_S,
        "training_bins": int(running.sum()),
        "event_bins_in_running": n_event_running,
        "n_encoding_spikes": [len(x) for x in running_only["encoding_positions"]],
    }

    # the spikes of every scored bin, of the stored-likelihood sample and of
    # the library check's block
    stream = keyed_rng((SEED, epoch_id), "checks")
    block_start = int(stream.integers(0, time.size - LIBRARY_CHECK_BINS))
    block = np.arange(block_start, block_start + LIBRARY_CHECK_BINS)
    needed = in_window.copy()
    needed[block] = True
    kept_times, kept_marks = spikes_in_bins(spike_times, marks, time, needed)
    terms: dict[str, TetrodeTerms] = {}
    for name, encoding in (("run", running_only), ("can", stored)):
        started = clock.perf_counter()
        terms[name] = per_tetrode_terms(encoding, kept_times, kept_marks, time, centers)
        log(f"{name} terms: {terms[name].time_bin.size} spikes, "
            f"{clock.perf_counter() - started:.0f} s")  # fmt: skip
    checks["terms"] = {
        "n_tetrodes": terms["can"].n_tetrodes,
        "n_spikes_in_epoch": [int(np.asarray(x).size) for x in spike_times],
        "n_spikes_scored": int(terms["can"].time_bin.size),
    }

    # check 1: the summed canonical terms are the decode's stored likelihood
    spiky = np.unique(terms["can"].time_bin[in_window[terms["can"].time_bin]])
    sample = np.sort(
        stream.choice(spiky, min(N_STORED_CHECK_BINS, spiky.size), replace=False)
    )
    stored_check = relative_error(
        terms["can"].log_likelihood(sample),
        stored_log_likelihood(files["results"], sample, interior),
    )
    checks["stored_log_likelihood"] = {"n_bins": int(sample.size), **stored_check}
    if stored_check["max_rel"] > STORED_MAX_REL:
        raise ValueError(
            f"the summed per-tetrode terms do not reproduce results.nc: {stored_check}"
        )
    # check 2: each encoding's terms are the library's likelihood (the final row
    # of a window's own decode holds no spike, so it is left out)
    checks["library_log_likelihood"] = {}
    for name, encoding in (("run", running_only), ("can", stored)):
        library, _, _ = event_likelihood(
            int(block[0]), int(block[-1]), time, linear, spike_times, marks,
            encoding_inputs(encoding)["encargs"],
        )  # fmt: skip
        error = relative_error(terms[name].log_likelihood(block)[:-1], library[:-1])
        checks["library_log_likelihood"][name] = {
            "n_bins": int(block.size - 1),
            **error,
        }
        if error["max_rel"] > LIBRARY_MAX_REL:
            raise ValueError(
                f"{name} terms differ from the library's likelihood: {error}"
            )

    # evidence under both encodings
    halves = tetrode_halves(terms["can"].n_tetrodes, seed=HALVES_SEED)
    columns, shared = [], None
    for name in ENCODINGS:
        started = clock.perf_counter()
        evidence = window_evidence(
            terms[name], windows, animal, distance, halves,
            no_spike_rate=no_spike_rate, seed=(SEED, epoch_id),
        )  # fmt: skip
        log(f"{name} evidence: {clock.perf_counter() - started:.0f} s")
        if shared is None:
            shared = evidence[SHARED_COLUMNS]
            columns.append(shared)
        elif not shared.equals(evidence[SHARED_COLUMNS]):
            raise ValueError("the encodings scored different spikes or partners")
        columns.append(evidence.drop(columns=SHARED_COLUMNS).add_prefix(f"{name}_"))
    table = pd.concat([windows, *columns], axis=1)
    cumulative_speed = np.r_[0.0, np.cumsum(np.where(np.isfinite(speed), speed, 0.0))]
    table.insert(0, "epoch_id", epoch_id)
    table.insert(1, "animal", parse_nwb_filename(f"{epoch_id.split('_')[0]}_.nwb")[0])
    table["start_time"] = time[s]
    table["mean_speed"] = (cumulative_speed[e + 1] - cumulative_speed[s]) / (e - s + 1)
    table["animal_center_cm"] = np.asarray(environment.place_bin_centers_).ravel()[
        interior
    ][table.animal_center_bin.to_numpy()]

    # what the windows are, and two identities the evidence must meet
    rho = no_spike_rate * bin_s
    silent = table.n_spikes == 0
    identity = table.run_ns_logratio - (
        table.run_ns_expected - terms["run"].n_tetrodes / 2 * rho * table.n_bins
    )
    no_spike_bouts = table[(table.kind == "bout") & (table.state == "No-Spike")]
    checks["windows"] = {
        "n": table.groupby(["kind", "state", "regime"]).size().to_dict(),
        "n_found": (
            table[table.kind == "bout"]
            .groupby(["state", "regime"])
            .n_found.first()
            .to_dict()
        ),
        "without_loc_partner": int((~table.has_loc_partner).sum()),
        "no_spike_bouts": len(no_spike_bouts),
        "no_spike_bouts_with_any_spike": int((no_spike_bouts.n_spikes > 0).sum()),
        "silent_ns_logratio_identity_max_abs": (
            float(identity[silent].abs().max()) if silent.any() else None
        ),
    }
    checks["windows"]["n"] = {"|".join(k): v for k, v in checks["windows"]["n"].items()}
    checks["windows"]["n_found"] = {
        "|".join(k): v for k, v in checks["windows"]["n_found"].items()
    }
    return table, checks


def run_settings() -> dict:
    """What every epoch of a run shares; a rerun reuses an epoch only on a match."""
    return {
        "seed": SEED,
        "halves_seed": HALVES_SEED,
        "n_partitions": 10,
        "regimes": "immobile head_speed < 4 cm/s, running > 10 cm/s",
        "bouts": "P(state) > 0.8, >= 5 bins, <= 1500 per state x regime, "
        "scored on a seeded <= 50-bin sub-window",
        "calibration": "2000 random 10-bin windows per regime",
        "partners": "20 per window from its pool; Local partners >= 20 cm away",
        "encodings": {
            "run": f"running-only refit, head_speed >= {RUNNING_ENCODING_MIN_SPEED_CM_S} cm/s",
            "can": "the decode's own (no_ripple training mask)",
        },
    }


def run_epoch(
    epoch_id: str,
    *,
    decode_dir: Path,
    output_dir: Path,
    load_inputs: Callable[[str], dict] = load_marks_and_training_mask,
    log: Callable[[str], None] = print,
) -> dict:
    """Compute and write one epoch; return its record (strict JSON types).

    ``status`` is ``"skipped"`` with ``reason`` when a decode file is missing,
    ``"failed"`` with ``error`` and ``traceback`` when any step or check fails
    (its outputs are then removed), else ``"ok"`` with the validation record.
    """
    paths = output_paths(epoch_id, output_dir)
    missing = [name for name, path in decode_files(epoch_id, decode_dir).items()
               if not path.exists()]  # fmt: skip
    if missing:
        for path in paths.values():
            path.unlink(missing_ok=True)
        return {
            "status": "skipped",
            "reason": f"no {', '.join(missing)} in {decode_dir}",
        }
    started = clock.perf_counter()
    before = resource.getrusage(resource.RUSAGE_SELF)
    try:
        table, checks = epoch_table(epoch_id, decode_dir, load_inputs, log)
        write_table(table, paths["windows"])
        record = {"status": "ok", "settings": run_settings(), **checks}
    except Exception as error:  # noqa: BLE001 - recorded; the run goes on
        for path in paths.values():
            path.unlink(missing_ok=True)
        record = {
            "status": "failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
    usage = resource.getrusage(resource.RUSAGE_SELF)
    record["wall_clock_s"] = clock.perf_counter() - started
    record["cpu_s"] = (usage.ru_utime - before.ru_utime) + (
        usage.ru_stime - before.ru_stime
    )
    # the worker's peak resident set over its life so far, so an epoch it ran
    # before counts too (ru_maxrss is in KiB on Linux)
    record["worker_peak_rss_gib"] = usage.ru_maxrss / 2**20
    ready: dict = _json_ready(record)
    if ready["status"] == "ok":
        write_json(paths["validation"], ready)
    return ready


def _run_epoch_logged(epoch_id: str, kwargs: dict, log_dir: Path) -> dict:
    """One dask task: ``run_epoch``, logged to ``logs/{epoch_id}.log``."""
    logger = setup_logger(
        name_logfile=f"{epoch_id}_state_evidence",
        path_logfile=log_dir / f"{epoch_id}.log",
    )
    logger.info("%s: start", epoch_id)
    record = run_epoch(epoch_id, log=logger.info, **kwargs)
    if record["status"] == "failed":
        logger.error("%s\n%s", record["error"], record["traceback"])
    else:
        logger.info("%s: %s", epoch_id, record["status"])
    return record


def run_epochs(
    epochs: list[str], kwargs: dict, n_workers: int
) -> Iterator[tuple[str, dict]]:
    """Each epoch on its own dask CPU worker; yield records as they finish."""
    from dask.distributed import Client, LocalCluster, as_completed

    if not epochs:
        return
    log_dir = kwargs["output_dir"] / "logs"
    with (
        LocalCluster(
            n_workers=n_workers,
            processes=True,
            threads_per_worker=1,
            dashboard_address=":0",  # any free port: 8787 is often taken here
        ) as cluster,
        Client(cluster) as client,
    ):
        futures = {
            client.submit(
                _run_epoch_logged, epoch_id, kwargs, log_dir, key=epoch_id
            ): epoch_id
            for epoch_id in epochs
        }
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001 - the worker could not return
                record = {
                    "status": "failed",
                    "error": repr(error),
                    "traceback": "".join(traceback.format_exception(error)),
                }
            yield futures[future], record


# --- summary --------------------------------------------------------------------------
OKABE_ITO = {
    "bout": "#0072B2",
    "baseline": "#999999",
    "run": "#000000",
    "can": "#D55E00",
}
BOUT_FIGURE = [
    ("Local", "run_loc_mark", "Local\nanimal vs mismatched position"),
    ("Non-Local Continuous", "run_nl_anim_mark", "NL-Continuous\nMAP_S vs animal"),
    (
        "Non-Local Continuous",
        "run_nl_null_mark",
        "NL-Continuous\nMAP_S vs mismatched MAP",
    ),
    ("Non-Local Fragmented", "run_nl_anim_mark", "NL-Fragmented\nMAP_S vs animal"),
    (
        "Non-Local Fragmented",
        "run_nl_null_mark",
        "NL-Fragmented\nMAP_S vs mismatched MAP",
    ),
    ("No-Spike", "run_ns_logratio", "No-Spike\nsilence vs local, running-only"),
    ("No-Spike", "can_ns_logratio", "No-Spike\nsilence vs local, canonical"),
]
CALIBRATION_FIGURE = [
    ("Local", "run_loc_mark", "Local\nanimal vs mismatched"),
    ("Local", "run_loc_vs_Sremote_mark", "Local\nanimal vs S's best remote bin"),
    ("Non-Local Continuous", "run_nl_anim_mark", "NL-Continuous\nMAP_S vs animal"),
    ("Non-Local Continuous", "run_nl_null_mark", "NL-Continuous\nMAP_S vs mismatched"),
    ("Non-Local Fragmented", "run_nl_anim_mark", "NL-Fragmented\nMAP_S vs animal"),
    ("Non-Local Fragmented", "run_nl_null_mark", "NL-Fragmented\nMAP_S vs mismatched"),
    ("No-Spike", "can_ns_logratio", "No-Spike\nsilence vs local, canonical"),
    ("No-Spike", "run_ns_silent_frac", "No-Spike\nheld-out half silent"),
]
BIN_CENTERS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])


def figure_bouts(windows: pd.DataFrame, path: Path) -> None:
    """Confident bouts against their matched windows, primary evidence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, len(BOUT_FIGURE), figsize=(2.4 * len(BOUT_FIGURE), 6.2), layout="constrained"
    )
    for i, regime in enumerate(("immobile", "running")):
        for j, (state, metric, title) in enumerate(BOUT_FIGURE):
            ax = axes[i, j]
            data, labels = [], []
            for kind in ("bout", "baseline"):
                v = windows.loc[
                    (windows.kind == kind)
                    & (windows.state == state)
                    & (windows.regime == regime),
                    metric,
                ].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                data.append(v if v.size else np.array([np.nan]))
                labels.append(f"{kind}\nn={v.size}")
            box = ax.boxplot(
                data, widths=0.6, whis=(5, 95), showfliers=False, patch_artist=True,
                showmeans=True, medianprops={"color": "k"},
                meanprops={"marker": "D", "markersize": 3, "markerfacecolor": "white",
                           "markeredgecolor": "k"},
            )  # fmt: skip
            for patch, kind in zip(box["boxes"], ("bout", "baseline"), strict=True):
                patch.set_facecolor(OKABE_ITO[kind])
            ax.axhline(0, color="k", lw=0.6, ls="--")
            if state == "No-Spike":
                ax.set_yscale("symlog", linthresh=1.0)
            ax.set_xticks([1, 2], labels, fontsize=7)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(title, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{regime}\nheld-out evidence (nats)", fontsize=8)
    fig.suptitle(
        "Held-out spike evidence: confident bouts (P > 0.8) against matched random "
        "windows (same regime, same length)\nRunning-only encoding, spike-mark part "
        "unless noted; box = IQR, whiskers 5-95 %, line = median, diamond = mean; "
        "20 tetrode splits averaged",
        fontsize=9,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _error_bars(estimate: NDArray, lo: NDArray, hi: NDArray) -> list[NDArray]:
    """Error-bar lengths, zero where a CI could not be computed."""
    return [np.nan_to_num(estimate - lo), np.nan_to_num(hi - estimate)]


def figure_calibration(bins: pd.DataFrame, path: Path) -> None:
    """Mean evidence over epochs by the state's probability, hierarchical 95 % CIs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(CALIBRATION_FIGURE)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.8), layout="constrained")
    for i, regime in enumerate(("immobile", "running")):
        for j, (state, metric, title) in enumerate(CALIBRATION_FIGURE):
            ax = axes[i, j]
            curve = (
                bins[
                    (bins.regime == regime)
                    & (bins.state == state)
                    & (bins.metric == metric)
                ]
                .set_index("pbin")
                .reindex(list(PROBABILITY_BIN_LABELS))
            )
            ok = (curve.n >= MIN_BIN_WINDOWS).to_numpy()
            mean = curve["epoch_mean"].to_numpy(dtype=float)[ok]
            lo = curve.epoch_ci_lo.to_numpy(dtype=float)[ok]
            hi = curve.epoch_ci_hi.to_numpy(dtype=float)[ok]
            ax.errorbar(
                BIN_CENTERS[ok], mean, yerr=_error_bars(mean, lo, hi),
                color=OKABE_ITO["bout"], marker="o", ms=3, lw=1.2, capsize=2,
            )  # fmt: skip
            ax.axhline(0, color="k", lw=0.6, ls="--")
            if metric.endswith("ns_logratio"):
                ax.set_yscale("symlog", linthresh=1.0)
            ax.set_xticks(BIN_CENTERS, PROBABILITY_BIN_LABELS, fontsize=6)
            ax.tick_params(labelsize=7)
            ax.set_xlabel(f"mean P({state})", fontsize=7)
            if i == 0:
                ax.set_title(title, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{regime}\nmean held-out evidence", fontsize=8)
    fig.suptitle(
        "Calibration: held-out evidence in random 20 ms windows against the state's "
        "mean posterior\nMean over epochs of each epoch's bin mean; 95 % hierarchical "
        f"bootstrap CI (animals, then epochs); bins with < {MIN_BIN_WINDOWS} windows "
        "omitted",
        fontsize=9,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_silence(silence: pd.DataFrame, path: Path) -> None:
    """P(held-out half silent | selecting half silent) against local coding."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), layout="constrained")
    for ax, regime in zip(axes, ("immobile", "running"), strict=True):
        by_bin = silence[
            (silence.regime == regime) & (silence.state == "P(No-Spike) bin")
        ]
        by_bin = by_bin.set_index("pbin").reindex(list(PROBABILITY_BIN_LABELS))
        ok = (by_bin.runs_S_silent >= 20).to_numpy()
        y = by_bin.epoch_obs_given_S.to_numpy(dtype=float)[ok]
        lo = by_bin.epoch_obs_given_S_ci_lo.to_numpy(dtype=float)[ok]
        hi = by_bin.epoch_obs_given_S_ci_hi.to_numpy(dtype=float)[ok]
        ax.errorbar(
            BIN_CENTERS[ok], y, yerr=_error_bars(y, lo, hi),
            color=OKABE_ITO["bout"], marker="o", ms=3.5, lw=1.2, capsize=2,
            label="observed",
        )  # fmt: skip
        for prefix, label in (("run", "running-only"), ("can", "canonical")):
            ax.plot(
                BIN_CENTERS[ok], by_bin[f"epoch_pred_given_S_{prefix}"].to_numpy()[ok],
                color=OKABE_ITO[prefix], ls=":", lw=1.2,
                label=f"local coding, {label}",
            )  # fmt: skip
        every = silence[(silence.regime == regime) & (silence.state == "all windows")]
        if len(every):
            ax.axhline(every.epoch_obs_given_S.iloc[0], color="k", lw=0.8,
                       ls="--", label="all windows, observed")  # fmt: skip
            ax.axhline(every.obs_C_silent.iloc[0], color="0.5", lw=0.8, ls="-.",
                       label="all windows, P(C silent)")  # fmt: skip
        ax.set_xticks(BIN_CENTERS, by_bin.index, fontsize=7)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("mean P(No-Spike) in the 20 ms window", fontsize=8)
        ax.set_title(regime, fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("P(held-out half C silent | S silent)", fontsize=8)
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.suptitle(
        "Held-out coordinated silence (random 20 ms windows): mean over epochs, "
        "95 % hierarchical bootstrap CI",
        fontsize=9,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_csv(table: pd.DataFrame, path: Path) -> None:
    """Write ``table`` as CSV through a temporary file."""

    def write(temporary: Path) -> None:
        table.to_csv(temporary, index=False)

    write_atomically(path, write)


def readme(records: dict[str, dict], settings: dict, provenance: dict) -> str:
    """The output folder's README: what was run, coverage and the caveats."""
    status = pd.Series({k: r["status"] for k, r in records.items()}, dtype=object)
    lines = [
        "# Held-out spike evidence per decoder state",
        "",
        f"Produced by `{PRODUCED_BY}` at commit {provenance['commit']}"
        f"{' (uncommitted changes)' if provenance['dirty'] else ''}. Method and "
        "column definitions: `continuum_swr_replay.analysis.hippocampus.state_evidence`.",
        "",
        f"Epochs: {len(records)}; " + ", ".join(
            f"{k} {v}" for k, v in status.value_counts().sort_index().items()
        ) + ".",
        "",
    ]  # fmt: skip
    not_ok = {k: r for k, r in records.items() if r["status"] != "ok"}
    if not_ok:
        lines += ["| epoch | status | reason |", "|---|---|---|"]
        lines += [
            f"| {k} | {r['status']} | {r.get('reason') or r.get('error')} |"
            for k, r in sorted(not_ok.items())
        ]
        lines.append("")
    lines += ["## Settings", "", "```json", json.dumps(settings, indent=1), "```", ""]
    lines += ["## Caveats (properties of the method, not defects)", ""]
    lines += [f"- {c}" for c in CAVEATS]
    lines += [
        "",
        "## Files",
        "",
        "- `per_epoch/{epoch}_windows.parquet`, `per_epoch/{epoch}_validation.json`",
        "- `bout_stats.csv`, `calibration_stats.csv`, `trend_stats.csv`, "
        "`rate_calibration.csv`, `silence_stats.csv`",
        "- `fig_bouts_vs_matched.png`, `fig_calibration.png`, "
        "`fig_coordinated_silence.png`",
        "- `manifest.json`: settings, provenance and every epoch's record",
        "- `logs/{epoch}.log`",
        "",
    ]
    return "\n".join(lines)


def summarize(output_dir: Path, epochs: list[str]) -> None:
    """Pool every written epoch: the summary CSVs and the three figures."""
    frames = [
        read_table(output_paths(epoch_id, output_dir)["windows"]) for epoch_id in epochs
    ]
    if not frames:
        print("no epoch written; no summary", flush=True)
        return
    windows = pd.concat(frames, ignore_index=True)
    started = clock.perf_counter()
    summary = summarize_state_evidence(windows)
    print(f"summary of {len(epochs)} epochs in {clock.perf_counter() - started:.0f} s")
    for name, file_name in [
        ("bouts", "bout_stats.csv"),
        ("calibration", "calibration_stats.csv"),
        ("trends", "trend_stats.csv"),
        ("rate_calibration", "rate_calibration.csv"),
        ("silence", "silence_stats.csv"),
    ]:
        write_csv(summary[name], output_dir / file_name)
    figure_bouts(windows, output_dir / "fig_bouts_vs_matched.png")
    figure_calibration(summary["calibration"], output_dir / "fig_calibration.png")
    figure_silence(summary["silence"], output_dir / "fig_coordinated_silence.png")
    primary = summary["bouts"][summary["bouts"].primary]
    shown = ["state", "regime", "metric", "n", "n_epochs", "median", "frac_pos",
             "epoch_mean", "epoch_ci_lo", "epoch_ci_hi", "p_epoch", "epochs_pos",
             "animals_pos", "diff_epoch_mean", "p_diff_epoch"]  # fmt: skip
    print(primary[shown].to_string(index=False), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        help="HPC epoch ids, such as chimi20200225_02_r1_HPC (default: every "
        "canonical HPC decode in --decode-dir).",
    )
    parser.add_argument(
        "--decode-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Where the decodes are read from (default: Processed-Data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR
        / f"_refresh_{date.today():%Y%m%d}"
        / "state_evidence",
        help="The only place written to (default: "
        "Processed-Data/_refresh_<today>/state_evidence).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=8,
        help="Dask CPU worker processes, one epoch each at a time (default 8).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute epochs already written."
    )
    args = parser.parse_args(argv)
    decode_dir, output_dir = args.decode_dir.resolve(), args.output_dir.resolve()
    if output_dir == decode_dir:
        parser.error("--output-dir must not be the decode folder itself")
    epochs = args.epochs or canonical_epochs(decode_dir)
    if not epochs:
        sys.exit(f"No HPC decodes in {decode_dir}.")

    provenance = provenance_record(ROOT_DIR)
    settings = run_settings()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    to_run = []
    for epoch_id in epochs:
        validation = output_paths(epoch_id, output_dir)["validation"]
        previous = json.loads(validation.read_text()) if validation.exists() else None
        reusable = (
            previous is not None
            and previous.get("settings") == settings
            and previous["decode"]["decode_dir"] == str(decode_dir)
            and output_paths(epoch_id, output_dir)["windows"].exists()
        )
        if reusable and not args.overwrite and previous is not None:
            records[epoch_id] = previous
            print(f"{epoch_id}: already written", flush=True)
        else:
            to_run.append(epoch_id)

    def write_manifest() -> None:
        write_json(
            output_dir / "manifest.json",
            {
                "produced_by": PRODUCED_BY,
                "provenance": provenance,
                "settings": settings,
                "decode_dir": str(decode_dir),
                "epochs": {k: records[k] for k in sorted(records)},
            },
        )

    print(
        f"{len(to_run)} of {len(epochs)} epochs to run on {args.n_workers} workers; "
        f"logs in {output_dir / 'logs'}",
        flush=True,
    )
    kwargs = {"decode_dir": decode_dir, "output_dir": output_dir}
    for number, (epoch_id, record) in enumerate(
        run_epochs(to_run, kwargs, args.n_workers), start=1
    ):
        records[epoch_id] = record
        detail = (
            record.get("reason")
            or record.get("error")
            or (
                f"{record['wall_clock_s']:.0f} s, peak RSS {record['worker_peak_rss_gib']:.1f} GiB"
            )
        )
        print(
            f"[{number}/{len(to_run)}] {epoch_id}: {record['status']} ({detail})",
            flush=True,
        )
        write_manifest()
    write_manifest()
    ok = [k for k in sorted(records) if records[k]["status"] == "ok"]
    summarize(output_dir, ok)
    (output_dir / "README.md").write_text(readme(records, settings, provenance))
    failed = [k for k, r in records.items() if r["status"] == "failed"]
    if failed:
        sys.exit(f"{len(failed)} epochs failed: {', '.join(sorted(failed))}")


if __name__ == "__main__":
    main()
