#!/usr/bin/env python
"""Held-out spike evidence for each decoder state, on every canonical HPC decode.

Does each state capture what it claims, judged by spikes that did not choose
the location? The method, its evidence columns and its caveats are in
``continuum_swr_replay.analysis.hippocampus.state_evidence``; this runner feeds
it one epoch at a time and summarizes the pooled result.

Per epoch (a dask CPU worker each):

1. **Decode.** ``{epoch}_results.nc`` must carry the ``decode_masks`` of the
   configured SWR source's runs (``saved_decode_masks``): a decode made before
   R11 is refused before anything is loaded. ``{epoch}_80_data_over_time.pkl``,
   read through the SWR source (``swr_source.with_swr_source``), gives the time
   axis, head position and speed, linear position and the four state
   probabilities; its own non-local event labels, as the decoder wrote them
   (before the analysed-time rule), give the event bins the running-only
   encodings leave out. ``{epoch}_detector.pkl`` gives the fitted encoding
   model and ``no_spike_rate``. The probabilities must be finite and sum to 1.
2. **Windows.** Each state's confident bouts and their matched windows, and the
   calibration windows (``build_windows``), seeded by ``(SEED, epoch_id)``. A
   bin in excluded time (``swr_source.excluded_intervals``: the artifact mask
   widened by 0.3 s) is in no regime, so every window lies wholly outside it:
   the decode set the mask's bins missing (their stored log-likelihood is 0),
   and their spikes are an artifact's.
3. **Marks** and the decode's masks, through the project's loader
   (``load_data``; read-only Spyglass) and ``decoding_common.hpc_decode_masks``,
   as the decoder built them: the training bins leave out the Kay inventory's
   events and the artifact mask, and the mask's bins are missing. Its time axis
   must be the decode's, and refitting the encoding model on it with those
   training bins must reproduce the stored one exactly, so the marks are the
   decode's.
4. **Encodings** (``running_only_masks``): running-only refits (head speed
   >= 4 cm/s), on every running bin and on each fold of alternating 30 s
   blocks (``time_folds``), leaving out running bins within two bins of a
   decoded non-local event bin (one the analysed-time rule drops included) and
   the decode's missing bins; no event bin may be in the library's padded
   training span of any of them (``training_spike_bins``).
   The canonical encoding is the decode's own.
5. **Per-tetrode terms** for the spikes in the scored bins, one encoding at a
   time. Checked twice: summed over all tetrodes the canonical terms reproduce
   the stored ``results.nc`` log-likelihood on 2000 sampled bins, and every
   encoding's terms reproduce the library's own likelihood
   (``event_likelihood``) on a 200-bin block. Real place bins whose stored
   occupancy is above 0 and below ``UNDERFLOW_OCCUPANCY_FLOOR`` (float32
   subnormals where the animal never went; a GPU decode keeps them, the CPU
   flushes them) are left out of both comparisons and of every encoding's MAP_S
   choice (``subnormal_occupancy_bin_mask``). A bin with an occupancy of exactly
   0 is not: both devices read ``occupancy > 0`` as False and agree there. The
   left-out bins are listed in the validation record (``underflowed_bins``: the
   floor, the count, the bins), and a window whose choice would have been one is
   scored on the remaining bins and marked (``*_mapS_left_out_runs``,
   ``*_mapSr_left_out_runs``) and counted (``windows.left_out_selection``). With
   none, the outputs are unchanged.
6. **Evidence** under every encoding (``window_evidence``), 10 seeded tetrode
   half-partitions x 2 directions. The primary, ``runcv_``, takes each window
   from the fold model without its centre bin (``held_out_evidence``), so a
   window's own spikes are not in the model that scores it.

``epoch_table`` and ``run_epoch`` also take other label sets (``LabelSet``):
labels from another decode, scored by chosen tetrode pairs, their rows stacked
(Task 13's half decodes used this; their runner is removed, see git history).

It writes ``per_epoch/{epoch}_windows.parquet`` (one row per window; columns
``runcv_*`` for the primary, time-held-out running-only encoding, ``run_*`` for
the running-only encoding fit on every running bin and ``can_*`` for the
canonical one; ``cv_fold`` and ``cv_insample_share``) and
``per_epoch/{epoch}_validation.json``. An epoch whose decode files are
missing is recorded as skipped with that reason; one that fails a check, or
any step, as failed with its error, and the rest run. Each epoch's record
holds the code's commit and dirty flag and every decode file's path, size and
``mtime_ns``. A rerun reuses an epoch only when all of them match, the run
settings match and neither tree has uncommitted changes (``reuse_reason``);
otherwise, or with ``--overwrite``, it recomputes it and logs why.

Without ``--epochs`` the epochs are every ``*_HPC`` epoch with a canonical
decode file in ``--decode-dir`` (``canonical_epochs``); coverage against the
canonical epoch list is the audit's job, not this script's.

Then, over every epoch written: ``bout_stats.csv``, ``calibration_stats.csv``,
``trend_stats.csv``, ``rate_calibration.csv`` and ``silence_stats.csv``
(``summarize_state_evidence``), three figures (bouts against matched windows,
calibration curves, coordinated silence), ``manifest.json`` (settings,
provenance and every epoch's record) and a ``README.md`` with the caveats.
``--window-level-stats`` adds the window-level columns (``*_window_level``),
which treat windows as independent and serve only to compare with the
prototype; they take most of a summary's time (about an hour for every epoch).

It writes nothing outside ``--output-dir``. Per-epoch logs go to its ``logs/``.

Usage:
    python scripts/qa/run_state_evidence.py --n-workers 16
    python scripts/qa/run_state_evidence.py --epochs chimi20200225_02_r1_HPC \\
        --decode-dir Processed-Data/_staging_redecode_20261004 \\
        --output-dir /tmp/state_evidence --n-workers 1 --window-level-stats
"""

import os

# a CPU job: it must never take a GPU from the decodes. Set here, before any
# JAX import, whatever the environment said; run_epoch checks it again
# (require_cpu), in every worker.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time as clock  # noqa: E402
import traceback  # noqa: E402
from collections.abc import Callable, Iterator, Mapping, Sequence  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from continuum_swr_replay.analysis.hippocampus import swr_source  # noqa: E402
from continuum_swr_replay.analysis.hippocampus.artifacts.masks import (  # noqa: E402
    interval_mask,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (  # noqa: E402
    interior_bin_distances_cm,
)
from continuum_swr_replay.analysis.hippocampus.state_evidence import (  # noqa: E402
    CV_BLOCK_S,
    ENCODING_HELD_OUT,
    ENCODING_KEY,
    ENCODING_LABELS,
    HALVES_SEED,
    MIN_BIN_WINDOWS,
    PRIMARY_ENCODING,
    PROBABILITY_BIN_LABELS,
    STATES,
    UNDERFLOW_OCCUPANCY_FLOOR,
    bout_animal_summaries,
    build_windows,
    code_version,
    encoding_comparison,
    encoding_inputs,
    event_likelihood,
    held_out_evidence,
    interior_bin_mask,
    json_ready,
    keyed_rng,
    load_detector,
    nearest_interior_bins,
    paired_bouts,
    per_tetrode_terms,
    regimes,
    saved_decode_masks,
    subnormal_occupancy_bin_mask,
    summarize_state_evidence,
    tetrode_halves,
    time_folds,
    training_spike_bins,
    window_evidence,
    window_folds,
    write_json,
)
from continuum_swr_replay.data_loaders.epoch_register import (  # noqa: E402
    analysed_epoch_ids,
)
from continuum_swr_replay.data_loaders.utils import (  # noqa: E402
    parse_epoch_identifier,
)
from continuum_swr_replay.decoding_common import (  # noqa: E402
    decode_paths,
    hpc_artifact_intervals,
    hpc_decode_masks,
    provenance_record,
    raise_open_file_limit,
    release_nwb_files,
    setup_logger,
    write_atomically,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR, ROOT_DIR  # noqa: E402
from continuum_swr_replay.table_io import read_table, write_table  # noqa: E402

PRODUCED_BY = "scripts/qa/run_state_evidence.py"
SEED = 20261004
"""Names every window and partner draw, with the epoch id."""
RUNNING_ENCODING_MIN_SPEED_CM_S = 4.0
EVENT_MARGIN_BINS = 2
"""Running bins this close to a non-local event bin are left out of every
running-only fit. The library pads each training run by one bin width, the
first two bins' spacing; at epoch timestamps (~1.6e9 s) a bin can be an ulp
narrower, so one bin of margin can leave a sliver of the event bin in the
span (one j16 pilot epoch did); two cannot."""
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
ENCODINGS = tuple(ENCODING_LABELS)
"""The evidence column prefixes, the primary (time-held-out) encoding first."""

CAVEATS = [
    "Encodings, by column prefix: "
    + "; ".join(f"{prefix}_ {text}" for prefix, text in ENCODING_HELD_OUT.items())
    + f". Running bins within {EVENT_MARGIN_BINS} bins of a non-local event bin "
    "are left out of every running-only fit, so the library's padding of each "
    "training run (one bin either side, with its float rounding) takes in no "
    "event spike.",
    "Only the remote location is held out by the tetrode split: S alone chooses "
    "MAP_S. The bouts and "
    "probability bins come from the all-tetrode decode, so C's spikes helped make "
    "every state label.",
    "Run-averaged Local and No-Spike values (loc_*, ns_*) equal half the "
    "all-tetrode, in-sample values: no half chooses the animal's position, and "
    "each tetrode scores in exactly half of the 20 runs.",
    "With few spikes, S's silence term pulls MAP_S toward low-rate bins, so "
    "MAP_S's distance from the animal (mapS_dist_cm) is not a Local test.",
    "Real place bins whose stored occupancy is above 0 and below "
    f"{UNDERFLOW_OCCUPANCY_FLOOR:g} (float32 subnormals where the animal never "
    "went; a GPU decode keeps them and a CPU rebuild flushes them to zero, D1) "
    "are left out of the stored-likelihood check and are no choice of MAP_S or "
    "of the best remote bin. A bin with an occupancy of exactly 0 is not: both "
    "devices agree there, and a silent window may still choose it. A window "
    "that would have chosen a left-out bin is scored on the remaining bins, and "
    "marked and counted in the epoch's validation record (underflowed_bins, "
    "windows.left_out_selection). D1's scan of the 242 analysed epochs found one "
    "with such bins, chimi20200221_08_r4.",
    "Each epoch enters as one value (epoch_summary): its median over windows "
    "for the log-likelihood ratios (loc_*, nl_*, nlr_*, ns_logratio), which are "
    "heavy-tailed (three spike bursts in one pilot chimi epoch score -1641, "
    "-1407 and -1016 nats), and its mean for expected counts, counts, shares "
    "and distances (ns_expected, ns_count, silent shares, MAP_S distances), "
    "which are bounded. The other_ columns repeat the inference over the other "
    "summary. These rules were set on the five pilot epochs, before the full "
    "run.",
    "Inference is at the animal level, each animal's summary being the mean of "
    "its epochs' values: animal_estimate (the mean over animals, each weighted "
    "equally) with its 95 % t-interval animal_ci_lo / animal_ci_hi and the "
    "two-sided t-test p_animal_t (A - 1 degrees of freedom, so 4 with five "
    "animals), with k of A animals and k of N epochs. Secondary: epoch_estimate "
    "(each epoch weighted equally) with its hierarchical bootstrap CI (animals, "
    "then epochs within each; Saravanan, Berman & Sober 2020), which under-covers "
    "with few animals (about 0.63 times the t-interval's width with the five "
    "pilot animals), and p_epoch_signflip, a sign-flip test over epochs that "
    "ignores the animals.",
    "A calibration verdict (rises, top_positive) reads the animal t-interval. "
    "It needs at least 10 windows in the top and in the bottom probability bins, "
    "pooled over epochs, else it reads 'too few windows' ('too few animals' when "
    "fewer than two animals give the interval); an epoch's slope needs two bins "
    "of 10 or more of its windows.",
    "Columns ending _window_level (only with --window-level-stats) treat "
    "windows as independent: they ignore the nesting, are not for inference, "
    "and are kept only to compare with the prototype.",
]


def caveats(records: dict[str, dict]) -> list[str]:
    """:data:`CAVEATS`, and the No-Spike one at the decoders' own rate.

    The rate and the bin come from the scored epochs' records
    (``decode.no_spike_rate``, ``decode.bin_s``), each distinct pair listed.
    """
    pairs = sorted(
        {
            (
                float(f"{r['decode']['no_spike_rate']:.6g}"),
                round(r["decode"]["bin_s"], 9),
            )
            for r in records.values()
            if r.get("status") == "ok"
        }
    )
    if not pairs:
        return [*CAVEATS, "No epoch was scored, so no No-Spike rate was read."]
    costs = "; ".join(
        f"{-np.log(rate * bin_s):.1f} nats (log({rate:g} Hz x {bin_s * 1e3:g} ms))"
        for rate, bin_s in pairs
    )
    no_spike = (
        f"One spike costs {costs} under No-Spike, at the decoders' own "
        "no_spike_rate, so a confident No-Spike label forces both halves silent; "
        "only the size of E_C (ns_expected) is informative in No-Spike bouts. The "
        "coordinated-silence test over random windows is the only held-out "
        "No-Spike test."
    )
    return [*CAVEATS, no_spike]


# --- inputs ----------------------------------------------------------------------------
def require_cpu() -> None:
    """Refuse to compute unless JAX is held to the CPU.

    Raises
    ------
    RuntimeError
        ``JAX_PLATFORMS`` is not "cpu" or ``CUDA_VISIBLE_DEVICES`` is not
        empty (the module sets both on import), or JAX's default backend is
        not the CPU.
    """
    if (
        os.environ.get("JAX_PLATFORMS") != "cpu"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise RuntimeError(
            "this runner computes on the CPU only: JAX_PLATFORMS must be 'cpu' and "
            "CUDA_VISIBLE_DEVICES empty"
        )
    import jax

    if jax.default_backend() != "cpu":
        raise RuntimeError(f"JAX's default backend is {jax.default_backend()!r}")


def decode_files(epoch_id: str, decode_dir: Path) -> dict[str, Path]:
    """The canonical decode's three files for an epoch (``decode_paths``)."""
    paths = decode_paths(epoch_id, decode_dir)
    return {
        "data_over_time": paths.data_over_time,
        "detector": paths.detector,
        "results": paths.results,
    }


def decode_file_stats(epoch_id: str, decode_dir: Path) -> dict[str, dict]:
    """Each present decode file's path, size and ``mtime_ns``, by file name."""
    stats = {}
    for name, path in decode_files(epoch_id, decode_dir).items():
        if path.exists():
            info = path.stat()
            stats[name] = {
                "path": str(path.resolve()),
                "size": info.st_size,
                "mtime_ns": info.st_mtime_ns,
            }
    return stats


def reuse_reason(
    previous: dict | None,
    *,
    settings: dict,
    code: dict,
    decode_stats: dict,
    windows_written: bool,
) -> str | None:
    """Why an epoch written before must be recomputed, or None to reuse it.

    It is reused only when its record is "ok" with its windows table present,
    and its run settings, code commit and decode files (path, size and
    ``mtime_ns``) all match this run's, with no uncommitted changes in either
    tree (then the commit does not identify the code).
    """
    if previous is None:
        return "not written before"
    if previous.get("status") != "ok" or not windows_written:
        return "no complete earlier output"
    if previous.get("settings") != settings:
        return "the run settings differ"
    earlier = previous.get("code") or {}
    if earlier.get("commit") != code["commit"]:
        return f"code commit {str(earlier.get('commit'))[:9]} -> {code['commit'][:9]}"
    if code["dirty"] or earlier.get("dirty", True):
        return "uncommitted changes, so the commit does not identify the code"
    if previous.get("decode_files") != decode_stats:
        return "a decode file's path, size or mtime_ns differs"
    return None


def canonical_epochs(decode_dir: Path) -> list[str]:
    """Every analysed HPC epoch with any canonical decode file in ``decode_dir``.

    The epoch register's analysed epochs (``analysed_epoch_ids``), never a
    glob, so a pending epoch's decode there is not read. An epoch missing one of
    its files is still listed, so that the run records why it skips it.
    """
    return [
        epoch_id
        for epoch_id in analysed_epoch_ids()
        if any(path.exists() for path in decode_files(epoch_id, decode_dir).values())
    ]


def load_marks_and_training_mask(epoch_id: str) -> dict:
    """Marks, the position axis and the decode's masks, from ``load_data``.

    The masks are ``decoding_common.hpc_decode_masks``' for a ``no_ripple``
    decode, from the configured SWR source, as the decoder built them.

    Returns
    -------
    dict
        ``time`` (the loader's position axis, s), ``spike_times`` and
        ``marks`` (per tetrode), ``is_training`` and ``is_missing`` (bool per
        time bin).
    """
    from continuum_swr_replay.data_loaders import load_data
    from continuum_swr_replay.types import SessionDataKeys as K

    key = parse_epoch_identifier(epoch_id)
    nwb_file_name, epoch_name, brain_area = (
        key.nwb_file_name,
        key.epoch_name,
        key.brain_area,
    )
    data = load_data(
        nwb_file_name,
        epoch_name,
        include=frozenset({K.SPIKE_TIMES, K.SPIKE_WAVEFORM_FEATURES}),
    )
    masks = hpc_decode_masks(
        nwb_file_name,
        epoch_name,
        data["position_info"],
        hpc_artifact_intervals(nwb_file_name, epoch_name),
        "no_ripple",
    )
    return {
        "time": data["position_info"].index.to_numpy(dtype=float),
        "spike_times": [np.asarray(x) for x in data["spike_times"][brain_area]],
        "marks": [np.asarray(x) for x in data["spike_waveform_features"][brain_area]],
        "is_training": masks.is_training,
        "is_missing": masks.is_missing,
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


@dataclass(frozen=True)
class LabelSet:
    """Where one set of windows' state labels comes from, and which runs score it.

    The default is Task 12's: the canonical decode's labels, scored by every
    seeded partition in both directions. Labels from a decode that saw one half
    S alone (Task 13's half decodes) are scored by its own
    half C only: ``halves=[(S, C)]`` and ``both_directions=False``.

    Attributes
    ----------
    name : str
        Keys the set's checks in the validation record.
    probabilities : ndarray, shape (n_time, 4), optional
        State probabilities in ``STATES`` order; None for the canonical
        decode's own.
    time : ndarray, shape (n_time,), optional
        Their time axis, which must be the decode's; required with
        ``probabilities``.
    halves : sequence of (ndarray, ndarray), optional
        The tetrode pairs that score; None for the seeded partitions,
        ``tetrode_halves(n_tetrodes, seed=HALVES_SEED)``.
    both_directions : bool
        ``window_evidence``'s: False when each pair's first half made the labels.
    columns : mapping
        Constant columns put on the set's rows, after ``animal``.
    """

    name: str = "all"
    probabilities: NDArray | None = None
    time: NDArray | None = None
    halves: Sequence[tuple[NDArray, NDArray]] | None = None
    both_directions: bool = True
    columns: Mapping[str, Any] = field(default_factory=dict)


def label_probabilities(label_set: LabelSet, time: NDArray) -> NDArray:
    """A label set's state probabilities, checked against the decode's time axis.

    Raises
    ------
    ValueError
        Its axis is not the decode's, or its probabilities are not finite rows
        of four summing to 1.
    """
    if label_set.time is None or not np.array_equal(
        np.asarray(label_set.time, dtype=float), time
    ):
        raise ValueError(f"{label_set.name}: its time axis is not the decode's")
    probabilities = np.asarray(label_set.probabilities, dtype=float)
    if probabilities.shape != (time.size, len(STATES)):
        raise ValueError(f"{label_set.name}: probabilities are {probabilities.shape}")
    worst = float(np.max(np.abs(probabilities.sum(1) - 1.0)))
    if not np.isfinite(probabilities).all() or worst > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(
            f"{label_set.name}: state probabilities are invalid (max |sum - 1| {worst})"
        )
    return probabilities


def checked_halves(
    halves: Sequence[tuple[NDArray, NDArray]], n_tetrodes: int
) -> list[tuple[NDArray, NDArray]]:
    """Tetrode pairs as int arrays: each two disjoint, non-empty sets of tetrodes."""
    out = []
    for first, second in halves:
        pair = (np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64))
        joined = np.concatenate(pair)
        if (
            not (pair[0].size and pair[1].size)
            or np.unique(joined).size != joined.size
            or joined.min() < 0
            or joined.max() >= n_tetrodes
        ):
            raise ValueError(
                f"a tetrode pair must be two disjoint, non-empty sets below {n_tetrodes}"
            )
        out.append(pair)
    return out


def time_held_out_record(
    windows: pd.DataFrame, cv: pd.DataFrame, in_span: dict[int, NDArray]
) -> dict:
    """How much of each window the time-held-out model that scores it was fit on."""
    s, e = windows.s.to_numpy(), windows.e.to_numpy()
    scoring = 1 - cv.cv_fold.to_numpy()
    touching = np.where(scoring == 0, in_span[0][e + 1] - in_span[0][s],
                        in_span[1][e + 1] - in_span[1][s]) > 0  # fmt: skip
    straddling = cv.cv_insample_share.to_numpy() > 0
    return {
        "block_s": CV_BLOCK_S,
        "windows": len(windows),
        "straddling_windows": int(straddling.sum()),
        "straddling_share": float(straddling.mean()),
        "windows_with_bins_in_the_scoring_models_spikes": int(touching.sum()),
    }


def epoch_table(
    epoch_id: str,
    decode_dir: Path,
    load_inputs: Callable[[str], dict],
    log: Callable[[str], None],
    label_sets: Sequence[LabelSet] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """One epoch's per-window table and its validation record.

    Each of ``label_sets`` (default: the canonical decode's labels,
    ``LabelSet()``) has its own windows and evidence, and the rows of every set
    are stacked. The encodings, their terms and their checks are shared. With
    the default the record holds ``time_held_out`` and ``windows`` at its top
    level; otherwise it holds them per set, under ``label_sets``.

    Raises
    ------
    ValueError
        A check failed; the message says which.
    """
    sets = [LabelSet()] if label_sets is None else list(label_sets)
    if not sets or len({label_set.name for label_set in sets}) != len(sets):
        raise ValueError("the label sets need distinct names")
    checks: dict[str, Any] = {}
    files = decode_files(epoch_id, decode_dir)
    # a decode made with the configured source's masks, before anything loads
    decode_masks = saved_decode_masks(files["results"])
    saved = pd.read_pickle(files["data_over_time"])
    # every event the decoder labelled, the ones the analysed-time rule drops
    # too: decoded non-local content stays out of the running-only encodings
    in_event = (
        swr_source.with_swr_source(saved, drop_masked_events=False)[
            "non_local_event"
        ].to_numpy()
        > 0
    )
    dot = swr_source.with_swr_source(saved)
    del saved
    excluded = swr_source.excluded_intervals_of(dot)
    time = dot.index.to_numpy(dtype=float)
    speed = dot["head_speed"].to_numpy(dtype=float)
    linear = dot["linear_position"].to_numpy(dtype=float)
    position = dot[["head_position_x", "head_position_y"]].to_numpy(dtype=float)
    probabilities = dot[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    del dot
    worst = float(np.max(np.abs(probabilities.sum(1) - 1.0)))
    if not np.isfinite(probabilities).all() or worst > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(f"state probabilities are invalid (max |sum - 1| {worst})")
    bin_s = float(np.median(np.diff(time)))
    checks["decode"] = {
        "decode_dir": str(decode_dir),
        "decode_masks": decode_masks,
        "n_time": int(time.size),
        "bin_s": bin_s,
        "max_abs_probability_sum_error": worst,
    }

    # windows: a bin with no finite position, or in excluded time, is in no regime
    regime = regimes(speed)
    regime[~np.isfinite(linear)] = ""
    in_excluded = interval_mask(time, excluded)
    regime[in_excluded] = ""
    checks["decode"]["excluded_bins"] = int(in_excluded.sum())
    windows: dict[str, pd.DataFrame] = {}
    for label_set in sets:
        labels = (
            probabilities
            if label_set.probabilities is None
            else label_probabilities(label_set, time)
        )
        windows[label_set.name] = build_windows(labels, regime, seed=(SEED, epoch_id))
        log(f"{label_set.name}: {len(windows[label_set.name])} windows")

    detector = load_detector(epoch_id, decode_dir)
    stored = detector.encoding_model_[ENCODING_KEY]
    environment = stored["environment"]
    interior = interior_bin_mask(environment)
    centers = np.asarray(environment.place_bin_centers_)[interior]
    distance = interior_bin_distances_cm(environment)
    no_spike_rate = float(detector.no_spike_rate)
    # real bins whose stored occupancy is a float32 subnormal or tiny: the CPU
    # rebuild is not the decode there (D1), so no check compares them and no
    # encoding picks them (an exact 0 is not one: both devices agree)
    left_out = subnormal_occupancy_bin_mask(
        stored["occupancy"], environment, floor=UNDERFLOW_OCCUPANCY_FLOOR
    )
    kept = ~left_out
    checks["underflowed_bins"] = {
        "floor": UNDERFLOW_OCCUPANCY_FLOOR,
        "n": int(left_out.sum()),
        "bins": np.flatnonzero(left_out).tolist(),
    }
    if left_out.any():
        log(f"{left_out.sum()} real place bins left out: stored occupancy above 0 "
            f"and below {UNDERFLOW_OCCUPANCY_FLOOR:g}")  # fmt: skip
    checks["decode"].update(
        no_spike_rate=no_spike_rate,
        sampling_frequency=float(detector.sampling_frequency),
        n_place_bins=int(interior.size),
        n_real_place_bins=int(interior.sum()),
    )

    # every bin a window of any set scores, and its animal position
    scored = np.zeros(time.size + 1, dtype=np.int64)
    for table in windows.values():
        np.add.at(scored, table.s.to_numpy(), 1)
        np.add.at(scored, table.e.to_numpy() + 1, -1)
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
            "refitting with the decode's training bins (hpc_decode_masks) does not "
            f"reproduce the stored encoding model, so the marks are not the "
            f"decode's, or it was decoded under other masks: {same}"
        )

    # the running-only encodings, checked against the library's padding
    folds = time_folds(time, CV_BLOCK_S)
    masks, checks["running_only"] = running_only_masks(
        time, speed, position, in_event, folds, is_missing=inputs["is_missing"]
    )
    in_span = {
        f: np.r_[0, np.cumsum(training_spike_bins(time, position, masks[f"runcv{f}"]))]
        for f in (0, 1)
    }
    cv = {name: window_folds(table, folds) for name, table in windows.items()}
    held_out = {
        name: time_held_out_record(windows[name], cv[name], in_span) for name in windows
    }
    if label_sets is None:
        checks["time_held_out"] = held_out[sets[0].name]

    # the spikes of every scored bin, of the stored-likelihood sample and of
    # the library check's block
    stream = keyed_rng((SEED, epoch_id), "checks")
    block_start = int(stream.integers(0, time.size - LIBRARY_CHECK_BINS))
    block = np.arange(block_start, block_start + LIBRARY_CHECK_BINS)
    needed = in_window.copy()
    needed[block] = True
    kept_times, kept_marks = spikes_in_bins(spike_times, marks, time, needed)
    checks["terms"] = {
        "n_tetrodes": len(spike_times),
        "n_spikes_in_epoch": [int(np.asarray(x).size) for x in spike_times],
    }

    # each encoding in turn: its terms, their checks and every set's evidence;
    # the canonical one first, as the stored likelihood checks the marks
    halves = {
        label_set.name: (
            tetrode_halves(len(spike_times), seed=HALVES_SEED)
            if label_set.halves is None
            else checked_halves(label_set.halves, len(spike_times))
        )
        for label_set in sets
    }
    evidence: dict[str, dict[str, pd.DataFrame]] = {name: {} for name in windows}
    checks["library_log_likelihood"] = {}
    for name in ("can", "run", "runcv0", "runcv1"):
        if name == "can":
            encoding = stored
        else:
            encoding = refit_encoding(
                detector, time, position, spike_times, marks, masks[name]
            )
            checks["running_only"][name]["n_encoding_spikes"] = [
                len(x) for x in encoding["encoding_positions"]
            ]
        started = clock.perf_counter()
        terms = per_tetrode_terms(encoding, kept_times, kept_marks, time, centers)
        log(f"{name} terms: {terms.time_bin.size} spikes, "
            f"{clock.perf_counter() - started:.0f} s")  # fmt: skip
        if name == "can":
            checks["terms"]["n_spikes_scored"] = int(terms.time_bin.size)
            # check 1: the summed canonical terms are the decode's stored likelihood
            spiky = np.unique(terms.time_bin[in_window[terms.time_bin]])
            sample = np.sort(
                stream.choice(
                    spiky, min(N_STORED_CHECK_BINS, spiky.size), replace=False
                )
            )
            stored_check = relative_error(
                terms.log_likelihood(sample)[:, kept],
                stored_log_likelihood(files["results"], sample, interior)[:, kept],
            )
            checks["stored_log_likelihood"] = {
                "n_bins": int(sample.size),
                **stored_check,
            }
            if stored_check["max_rel"] > STORED_MAX_REL:
                raise ValueError(
                    "the summed per-tetrode terms do not reproduce results.nc: "
                    f"{stored_check}"
                )
        # check 2: the terms are the library's likelihood under this encoding
        # (the final row of a window's own decode holds no spike, so it is left out)
        library, _, _ = event_likelihood(
            int(block[0]), int(block[-1]), time, linear, spike_times, marks,
            encoding_inputs(encoding)["encargs"],
        )  # fmt: skip
        error = relative_error(
            terms.log_likelihood(block)[:-1][:, kept], library[:-1][:, kept]
        )
        checks["library_log_likelihood"][name] = {
            "n_bins": int(block.size - 1),
            **error,
        }
        if error["max_rel"] > LIBRARY_MAX_REL:
            raise ValueError(
                f"{name} terms differ from the library's likelihood: {error}"
            )
        started = clock.perf_counter()
        for label_set in sets:
            evidence[label_set.name][name] = window_evidence(
                terms, windows[label_set.name], animal, distance,
                halves[label_set.name], both_directions=label_set.both_directions,
                left_out_bins=left_out, no_spike_rate=no_spike_rate,
                seed=(SEED, epoch_id),
            )  # fmt: skip
        log(f"{name} evidence: {clock.perf_counter() - started:.0f} s")
        del terms, encoding

    tables, per_set = [], {}
    for label_set in sets:
        table, record = label_set_table(
            epoch_id,
            label_set,
            windows[label_set.name],
            cv[label_set.name],
            evidence.pop(label_set.name),
            halves=halves[label_set.name],
            time=time,
            speed=speed,
            centers_cm=np.asarray(environment.place_bin_centers_).ravel()[interior],
            rho=no_spike_rate * bin_s,
        )
        tables.append(table)
        per_set[label_set.name] = record
    if label_sets is None:
        checks["windows"] = per_set[sets[0].name]
        return tables[0], checks
    checks["label_sets"] = {
        label_set.name: {
            "columns": dict(label_set.columns),
            "both_directions": label_set.both_directions,
            "halves": [
                [first.tolist(), second.tolist()]
                for first, second in halves[label_set.name]
            ],
            "time_held_out": held_out[label_set.name],
            "windows": per_set[label_set.name],
        }
        for label_set in sets
    }
    return pd.concat(tables, ignore_index=True), checks


def label_set_table(
    epoch_id: str,
    label_set: LabelSet,
    windows: pd.DataFrame,
    cv: pd.DataFrame,
    evidence: dict[str, pd.DataFrame],
    *,
    halves: Sequence[tuple[NDArray, NDArray]],
    time: NDArray,
    speed: NDArray,
    centers_cm: NDArray,
    rho: float,
) -> tuple[pd.DataFrame, dict]:
    """One label set's rows, and what its windows are.

    The primary ``runcv`` evidence takes each window from the fold model
    without its centre bin (``held_out_evidence``). Two identities the evidence
    must meet are recorded: every encoding scored the same spikes and partners,
    and a silent window's No-Spike log-ratio is E_C minus the scoring halves'
    silence rate (their mean size over the runs times rho times the bins).
    """
    evidence["runcv"] = held_out_evidence(
        [evidence.pop("runcv0"), evidence.pop("runcv1")], cv.cv_fold.to_numpy()
    )
    shared = evidence[PRIMARY_ENCODING][SHARED_COLUMNS]
    if any(not shared.equals(evidence[name][SHARED_COLUMNS]) for name in ENCODINGS):
        raise ValueError("the encodings scored different spikes or partners")
    columns = [
        evidence[name].drop(columns=SHARED_COLUMNS).add_prefix(f"{name}_")
        for name in ENCODINGS
    ]
    table = pd.concat([windows, cv, shared, *columns], axis=1)
    s, e = windows.s.to_numpy(), windows.e.to_numpy()
    cumulative_speed = np.r_[0.0, np.cumsum(np.where(np.isfinite(speed), speed, 0.0))]
    table.insert(0, "epoch_id", epoch_id)
    table.insert(1, "animal", parse_epoch_identifier(epoch_id).animal)
    for k, (column, value) in enumerate(label_set.columns.items()):
        table.insert(2 + k, column, value)
    table["start_time"] = time[s]
    table["mean_speed"] = (cumulative_speed[e + 1] - cumulative_speed[s]) / (e - s + 1)
    table["animal_center_cm"] = centers_cm[table.animal_center_bin.to_numpy()]

    # what the windows are, and the No-Spike identity
    scoring_sizes = [len(second) for _, second in halves]
    if label_set.both_directions:
        scoring_sizes += [len(first) for first, _ in halves]
    silent = table.n_spikes == 0
    identity = table[f"{PRIMARY_ENCODING}_ns_logratio"] - (
        table[f"{PRIMARY_ENCODING}_ns_expected"]
        - np.mean(scoring_sizes) * rho * table.n_bins
    )
    no_spike_bouts = table[(table.kind == "bout") & (table.state == "No-Spike")]
    record = {
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
    record["n"] = {"|".join(k): v for k, v in record["n"].items()}
    record["n_found"] = {"|".join(k): v for k, v in record["n_found"].items()}
    if f"{PRIMARY_ENCODING}_mapS_left_out_runs" in table:
        # windows whose selecting half would have chosen a left-out bin in some
        # run: scored on the remaining bins, marked here and in the table
        record["left_out_selection"] = {}
        for name in ENCODINGS:
            marked = (
                table[f"{name}_mapS_left_out_runs"]
                + table[f"{name}_mapSr_left_out_runs"]
            ) > 0
            record["left_out_selection"][name] = {
                "windows": int(marked.sum()),
                "bouts": int((marked & (table.kind == "bout")).sum()),
            }
    return table, record


def running_only_masks(
    time: NDArray,
    speed: NDArray,
    position: NDArray,
    in_event: NDArray,
    folds: NDArray,
    *,
    is_missing: NDArray,
) -> tuple[dict[str, NDArray], dict]:
    """The running-only encodings' training masks, and their record.

    ``run`` is every running bin (head speed >= 4 cm/s); ``runcv0`` and
    ``runcv1`` are the running bins of each fold (:func:`time_folds`). Running
    bins within :data:`EVENT_MARGIN_BINS` of a non-local event bin are left
    out: the library pads every training run by about one bin on each side and
    bridges bins without a position (:func:`training_spike_bins`), and no
    event spike may enter a running-only encoding. The decode's missing bins
    (its artifact mask) are left out too, as its own encoding leaves them out.

    Parameters
    ----------
    time, speed : ndarray, shape (n_time,)
    position : ndarray, shape (n_time, n_dims)
        What the encoding is fit with.
    in_event : ndarray of bool, shape (n_time,)
        Non-local event bins, every event the decoder labelled (the ones the
        analysed-time rule drops included).
    folds : ndarray of int, shape (n_time,)
    is_missing : ndarray of bool, shape (n_time,)
        The decode's missing bins (``hpc_decode_masks``).

    Returns
    -------
    masks : dict
        ``run``, ``runcv0`` and ``runcv1``: bool, shape (n_time,).
    record : dict
        The counts: event bins, those at running speed, the running bins left
        out near them, the missing bins at running speed (all left out), and
        per mask its training bins, the event and missing bins among them and
        the event bins in the library's padded span (0, or it raises). The
        missing bins are counted on the bin mask; the library's padding of
        each training run by about one bin is not counted (the canonical
        encoding is padded the same way).

    Raises
    ------
    ValueError
        An event bin is in a mask's library span, or a mask is empty.
    """
    at_speed = speed >= RUNNING_ENCODING_MIN_SPEED_CM_S
    near_event = in_event.copy()
    for shift in range(1, EVENT_MARGIN_BINS + 1):
        near_event[:-shift] |= in_event[shift:]
        near_event[shift:] |= in_event[:-shift]
    is_missing = np.asarray(is_missing, dtype=bool)
    running = at_speed & ~near_event & ~is_missing
    masks = {
        "run": running,
        "runcv0": running & (folds == 0),
        "runcv1": running & (folds == 1),
    }
    record: dict[str, Any] = {
        "min_speed_cm_s": RUNNING_ENCODING_MIN_SPEED_CM_S,
        "event_bins": int(in_event.sum()),
        "event_bins_at_running_speed": int((in_event & at_speed).sum()),
        "running_bins_left_out_near_events": int((near_event & at_speed).sum()),
        "missing_bins_at_running_speed": int((is_missing & at_speed).sum()),
    }
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"the {name} encoding has no training bin")
        in_span = int((in_event & training_spike_bins(time, position, mask)).sum())
        record[name] = {
            "training_bins": int(mask.sum()),
            "event_bins_in_training_bins": int((in_event & mask).sum()),
            "missing_bins_in_training_bins": int((is_missing & mask).sum()),
            "event_bins_in_library_span": in_span,
        }
        if in_span:
            raise ValueError(
                f"{in_span} non-local event bins are in the {name} encoding's "
                "training spikes (the library's padded span)"
            )
    return masks, record


def run_settings() -> dict:
    """What every epoch of a run shares; a rerun reuses an epoch only on a match."""
    return {
        "seed": SEED,
        "halves_seed": HALVES_SEED,
        "n_partitions": 10,
        "regimes": "immobile head_speed < 4 cm/s, running > 10 cm/s; none in "
        "excluded time (swr_source.excluded_intervals)",
        "bouts": "P(state) > 0.8, >= 5 bins, <= 1500 per state x regime, "
        "scored on a seeded <= 50-bin sub-window",
        "calibration": "2000 random 10-bin windows per regime",
        "partners": "20 per window from its pool; Local partners >= 20 cm away",
        "left_out_bins": "real place bins whose stored occupancy is above 0 and "
        f"below {UNDERFLOW_OCCUPANCY_FLOOR:g} (float32 subnormals: a CPU rebuild "
        "flushes them): no stored-likelihood or library comparison, and no MAP_S "
        "or remote-bin choice, of any encoding",
        "encodings": {
            "runcv": f"running-only (head_speed >= {RUNNING_ENCODING_MIN_SPEED_CM_S} "
            f"cm/s) refit on each fold of alternating {CV_BLOCK_S:g} s blocks; a "
            "window is scored by the fold without its centre bin",
            "run": "running-only refit on every running bin",
            "can": "the decode's own (hpc_decode_masks' no_ripple training bins)",
            "running_only_leaves_out": f"running bins within {EVENT_MARGIN_BINS} "
            "bins of a decoded non-local event bin (the decoder's own labels, "
            "before the analysed-time rule), and the decode's missing bins",
        },
    }


def run_epoch(
    epoch_id: str,
    *,
    decode_dir: Path,
    output_dir: Path,
    code: dict | None = None,
    load_inputs: Callable[[str], dict] = load_marks_and_training_mask,
    log: Callable[[str], None] = print,
    label_sets: Callable[[str], Sequence[LabelSet]] | None = None,
) -> dict:
    """Compute and write one epoch; return its record (strict JSON types).

    ``status`` is ``"skipped"`` with ``reason`` when a decode file is missing,
    ``"failed"`` with ``error`` and ``traceback`` when any step or check fails
    (its outputs are then removed), else ``"ok"`` with the validation record.
    Every record holds ``code`` (the commit and dirty flag; default this
    work tree's) and ``decode_files`` (``decode_file_stats``, taken before the
    files are read). ``label_sets`` gives an epoch's label sets
    (``epoch_table``'s); default the canonical decode's labels alone.
    """
    require_cpu()
    if code is None:
        code = code_version(provenance_record(ROOT_DIR))
    stamp = {"code": code, "decode_files": decode_file_stats(epoch_id, decode_dir)}
    paths = output_paths(epoch_id, output_dir)
    missing = [name for name, path in decode_files(epoch_id, decode_dir).items()
               if not path.exists()]  # fmt: skip
    if missing:
        for path in paths.values():
            path.unlink(missing_ok=True)
        return {
            "status": "skipped",
            "reason": f"no {', '.join(missing)} in {decode_dir}",
            **stamp,
        }
    started = clock.perf_counter()
    before = resource.getrusage(resource.RUSAGE_SELF)
    try:
        table, checks = epoch_table(
            epoch_id,
            decode_dir,
            load_inputs,
            log,
            None if label_sets is None else label_sets(epoch_id),
        )
        write_table(table, paths["windows"])
        record = {"status": "ok", "settings": run_settings(), **stamp, **checks}
    except Exception as error:  # noqa: BLE001 - recorded; the run goes on
        for path in paths.values():
            path.unlink(missing_ok=True)
        record = {
            "status": "failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
            **stamp,
        }
    usage = resource.getrusage(resource.RUSAGE_SELF)
    record["wall_clock_s"] = clock.perf_counter() - started
    record["cpu_s"] = (usage.ru_utime - before.ru_utime) + (
        usage.ru_stime - before.ru_stime
    )
    # the worker's peak resident set over its life so far, so an epoch it ran
    # before counts too (ru_maxrss is in KiB on Linux)
    record["worker_peak_rss_gib"] = usage.ru_maxrss / 2**20
    ready: dict = json_ready(record)
    if ready["status"] == "ok":
        write_json(paths["validation"], ready)
    return ready


def _run_epoch_logged(epoch_id: str, kwargs: dict, log_dir: Path) -> dict:
    """One dask task: ``run_epoch``, logged to ``logs/{epoch_id}.log``.

    When the epoch ends, pass or fail, the NWB files ``load_data`` opened through
    Spyglass are closed (``release_nwb_files``): ``run_epochs``'s workers run one
    task at a time (``threads_per_worker=1``), so no other epoch reads them.
    """
    logger = setup_logger(
        name_logfile=f"{epoch_id}_state_evidence",
        path_logfile=log_dir / f"{epoch_id}.log",
    )
    logger.info("%s: start", epoch_id)
    try:
        record = run_epoch(epoch_id, log=logger.info, **kwargs)
    finally:
        logger.info("closed %d NWB files Spyglass held open", release_nwb_files())
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
    "runcv": "#009E73",
    "run": "#000000",
    "can": "#D55E00",
}
BOUT_FIGURE = [
    ("Local", "runcv_loc_mark", "Local\nanimal vs mismatched position"),
    ("Non-Local Continuous", "runcv_nl_anim_mark", "NL-Continuous\nMAP_S vs animal"),
    (
        "Non-Local Continuous",
        "runcv_nl_null_mark",
        "NL-Continuous\nMAP_S vs mismatched MAP",
    ),
    ("Non-Local Fragmented", "runcv_nl_anim_mark", "NL-Fragmented\nMAP_S vs animal"),
    (
        "Non-Local Fragmented",
        "runcv_nl_null_mark",
        "NL-Fragmented\nMAP_S vs mismatched MAP",
    ),
    ("No-Spike", "runcv_ns_logratio", "No-Spike\nsilence vs local coding"),
]
CALIBRATION_FIGURE = [
    ("Local", "runcv_loc_mark", "Local\nanimal vs mismatched"),
    ("Local", "runcv_loc_vs_Sremote_mark", "Local\nanimal vs S's best remote bin"),
    ("Non-Local Continuous", "runcv_nl_anim_mark", "NL-Continuous\nMAP_S vs animal"),
    (
        "Non-Local Continuous",
        "runcv_nl_null_mark",
        "NL-Continuous\nMAP_S vs mismatched",
    ),
    ("Non-Local Fragmented", "runcv_nl_anim_mark", "NL-Fragmented\nMAP_S vs animal"),
    (
        "Non-Local Fragmented",
        "runcv_nl_null_mark",
        "NL-Fragmented\nMAP_S vs mismatched",
    ),
    ("No-Spike", "runcv_ns_logratio", "No-Spike\nsilence vs local coding"),
    ("No-Spike", "runcv_ns_silent_frac", "No-Spike\nheld-out half silent"),
]
BIN_CENTERS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])


def figure_bouts(windows: pd.DataFrame, bouts: pd.DataFrame, path: Path) -> None:
    """Each state's primary evidence in its confident bouts, animal by animal.

    Per state x regime, two columns: the bouts' values and their paired
    differences from their matched windows. A dot is one animal's summary
    (the mean of its epochs' medians or means, by metric); the diamond is the
    mean over animals with its 95 % t-interval, hollow with no bar when there
    is none; the faint bars are the windows' 5-95 % and 25-75 % ranges.

    Parameters
    ----------
    windows : pd.DataFrame
        The per-window table.
    bouts : pd.DataFrame
        :func:`bout_statistics` of it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paired = paired_bouts(windows)
    metrics: dict[str, list[str]] = {}
    for state, metric, _ in BOUT_FIGURE:
        metrics.setdefault(state, []).append(metric)
    animals = bout_animal_summaries(windows, metrics=metrics)
    fig, axes = plt.subplots(
        2, len(BOUT_FIGURE), figsize=(2.4 * len(BOUT_FIGURE), 6.2), layout="constrained"
    )
    for i, regime in enumerate(("immobile", "running")):
        for j, (state, metric, title) in enumerate(BOUT_FIGURE):
            ax = axes[i, j]
            cell = paired[(paired.state == state) & (paired.regime == regime)]
            value = cell[metric].to_numpy(dtype=float)
            for x, values in ((0, value), (1, value - cell[f"{metric}_base"])):
                finite = np.asarray(values, dtype=float)
                finite = finite[np.isfinite(finite)]
                if finite.size:
                    q05, q25, q75, q95 = np.percentile(finite, [5, 25, 75, 95])
                    ax.vlines(x, q05, q95, color="0.88", lw=1.5, zorder=0)
                    ax.vlines(x, q25, q75, color="0.78", lw=7, zorder=0)
            mine = animals[
                (animals.state == state)
                & (animals.regime == regime)
                & (animals.metric == metric)
            ]
            for x, column in ((0, "value"), (1, "diff")):
                ax.scatter(np.full(len(mine), x - 0.15), mine[column], s=12,
                           color="k", zorder=2)  # fmt: skip
            row = bouts[
                (bouts.state == state)
                & (bouts.regime == regime)
                & (bouts.metric == metric)
            ]
            if len(row) and row.n.iloc[0] > 0:
                r = row.iloc[0]
                _plot_estimates(
                    ax, np.array([0.15, 1.15]),
                    np.array([r.animal_estimate, r.diff_animal_estimate], float),
                    np.array([r.animal_ci_lo, r.diff_animal_ci_lo], float),
                    np.array([r.animal_ci_hi, r.diff_animal_ci_hi], float),
                    color=OKABE_ITO["bout"], line=False, marker="D", ms=5,
                )  # fmt: skip
            ax.axhline(0, color="k", lw=0.6, ls="--")
            if state == "No-Spike":
                ax.set_yscale("symlog", linthresh=1.0)
            ax.set_xticks([0, 1], ["bouts", "bout -\nmatched"], fontsize=7)
            ax.set_xlim(-0.6, 1.6)
            ax.tick_params(labelsize=7)
            ax.set_title(
                (f"{title}\n" if i == 0 else "")
                + f"{len(mine)} animals, {len(cell)} bouts",
                fontsize=7,
            )
            if j == 0:
                ax.set_ylabel(f"{regime}\nheld-out evidence (nats)", fontsize=8)
    fig.suptitle(
        "Held-out spike evidence in confident bouts (P > 0.8), against matched "
        "random windows (same regime, same length)\nTime-held-out running-only "
        "encoding, spike-mark part; dot = an animal, diamond = mean over animals "
        "with its 95 % t-interval; grey = the windows' 5-95 % and 25-75 %",
        fontsize=9,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _error_bars(
    estimate: NDArray, lo: NDArray, hi: NDArray
) -> tuple[NDArray[np.bool_], list[NDArray]]:
    """Which points have an interval, and the bar lengths of those that do.

    A point whose interval is NaN gets no bar, rather than a zero-length one
    that would read as certainty.
    """
    has = np.isfinite(lo) & np.isfinite(hi)
    return has, [(estimate - lo)[has], (hi - estimate)[has]]


def _plot_estimates(
    ax: Any,
    x: NDArray,
    estimate: NDArray,
    lo: NDArray,
    hi: NDArray,
    *,
    color: str,
    label: str | None = None,
    line: bool = True,
    marker: str = "o",
    ms: float = 3.0,
) -> None:
    """Estimates with their intervals; one without an interval is hollow, with no bar."""
    if line:
        ax.plot(x, estimate, color=color, lw=1.2)
    has, yerr = _error_bars(estimate, lo, hi)
    ax.errorbar(
        x[has], estimate[has], yerr=yerr, color=color, ls="none", marker=marker,
        ms=ms, capsize=2, label=label, zorder=3,
    )  # fmt: skip
    ax.plot(x[~has], estimate[~has], ls="none", marker=marker, ms=ms, mfc="none",
            mec=color, zorder=3)  # fmt: skip


def figure_calibration(bins: pd.DataFrame, path: Path) -> None:
    """Epoch-level evidence by the state's probability, hierarchical 95 % CIs."""
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
            estimate = curve.animal_estimate.to_numpy(dtype=float)[ok]
            lo = curve.animal_ci_lo.to_numpy(dtype=float)[ok]
            hi = curve.animal_ci_hi.to_numpy(dtype=float)[ok]
            _plot_estimates(
                ax, BIN_CENTERS[ok], estimate, lo, hi, color=OKABE_ITO["bout"]
            )
            ax.axhline(0, color="k", lw=0.6, ls="--")
            if metric.endswith("ns_logratio"):
                ax.set_yscale("symlog", linthresh=1.0)
            ax.set_xticks(BIN_CENTERS, PROBABILITY_BIN_LABELS, fontsize=6)
            ax.tick_params(labelsize=7)
            ax.set_xlabel(f"mean P({state})", fontsize=7)
            if i == 0:
                ax.set_title(title, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{regime}\nheld-out evidence", fontsize=8)
    fig.suptitle(
        "Calibration: held-out evidence in random 20 ms windows against the state's "
        "mean posterior\nMean over animals of each animal's mean epoch bin median "
        "(log-likelihood ratios) or mean (shares); 95 % t-interval over animals; "
        f"bins with < {MIN_BIN_WINDOWS} windows omitted",
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
        y = by_bin.obs_animal_estimate.to_numpy(dtype=float)[ok]
        lo = by_bin.obs_animal_ci_lo.to_numpy(dtype=float)[ok]
        hi = by_bin.obs_animal_ci_hi.to_numpy(dtype=float)[ok]
        _plot_estimates(
            ax, BIN_CENTERS[ok], y, lo, hi, color=OKABE_ITO["bout"],
            label="observed", ms=3.5,
        )  # fmt: skip
        for prefix, label in ENCODING_LABELS.items():
            ax.plot(
                BIN_CENTERS[ok], by_bin[f"pred_{prefix}_animal_estimate"].to_numpy()[ok],
                color=OKABE_ITO[prefix], ls=":", lw=1.2,
                label=f"local coding, {label}",
            )  # fmt: skip
        every = silence[(silence.regime == regime) & (silence.state == "all windows")]
        if len(every):
            ax.axhline(every.obs_animal_estimate.iloc[0], color="k", lw=0.8,
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
        "Held-out coordinated silence (random 20 ms windows): mean over animals, "
        "95 % t-interval",
        fontsize=9,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_csv(table: pd.DataFrame, path: Path) -> None:
    """Write ``table`` as CSV through a temporary file."""

    def write(temporary: Path) -> None:
        table.to_csv(temporary, index=False)

    write_atomically(path, write)


def readme(
    records: dict[str, dict],
    settings: dict,
    provenance: dict,
    discovery: dict,
    recomputed: dict[str, str],
) -> str:
    """The output folder's README: what was run, coverage and the caveats."""
    status = pd.Series({k: r["status"] for k, r in records.items()}, dtype=object)
    reused = sorted(set(records) - set(recomputed))
    lines = [
        "# Held-out spike evidence per decoder state",
        "",
        f"Produced by `{PRODUCED_BY}` at commit {provenance['commit']}"
        f"{' (uncommitted changes)' if provenance['dirty'] else ''}. Method and "
        "column definitions: `continuum_swr_replay.analysis.hippocampus.state_evidence`.",
        "",
        f"Epochs: {discovery['rule']}; {discovery['n_epochs']} listed"
        + (f", {discovery['n_with_all_files']} with all three decode files"
           if "n_with_all_files" in discovery else "")
        + f". {discovery['coverage']}",
        "",
        "Status: " + ", ".join(
            f"{k} {v}" for k, v in status.value_counts().sort_index().items()
        ) + f". Computed in this run: {len(recomputed)}; reused from an earlier "
        f"run with the same commit, settings and decode files: {len(reused)}"
        + (f" ({', '.join(reused)})" if reused else "") + ".",
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
    lines += [f"- {c}" for c in caveats(records)]
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


def summarize(
    output_dir: Path, epochs: list[str], *, window_level: bool = False
) -> None:
    """Pool every written epoch: the summary CSVs and the three figures.

    ``window_level`` adds the window-level columns to the CSVs.
    """
    frames = [
        read_table(output_paths(epoch_id, output_dir)["windows"]) for epoch_id in epochs
    ]
    if not frames:
        print("no epoch written; no summary", flush=True)
        return
    windows = pd.concat(frames, ignore_index=True)
    started = clock.perf_counter()
    summary = summarize_state_evidence(windows, window_level=window_level)
    print(f"summary of {len(epochs)} epochs in {clock.perf_counter() - started:.0f} s")
    for name, file_name in [
        ("bouts", "bout_stats.csv"),
        ("calibration", "calibration_stats.csv"),
        ("trends", "trend_stats.csv"),
        ("rate_calibration", "rate_calibration.csv"),
        ("silence", "silence_stats.csv"),
    ]:
        write_csv(summary[name], output_dir / file_name)
    figure_bouts(windows, summary["bouts"], output_dir / "fig_bouts_vs_matched.png")
    figure_calibration(summary["calibration"], output_dir / "fig_calibration.png")
    figure_silence(summary["silence"], output_dir / "fig_coordinated_silence.png")
    primary = summary["bouts"][summary["bouts"].primary]
    shown = ["state", "regime", "metric", "n", "n_epochs", "epoch_summary",
             "animal_estimate", "animal_ci_lo", "animal_ci_hi", "p_animal_t",
             "animals_pos", "epochs_pos", "diff_animal_estimate", "diff_animal_ci_lo",
             "diff_animal_ci_hi", "diff_p_animal_t", "diff_animals_pos"]  # fmt: skip
    print(primary[shown].to_string(index=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    """The command-line arguments."""
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
    parser.add_argument(
        "--window-level-stats",
        action="store_true",
        help="Also compute the window-level columns (*_window_level), which treat "
        "windows as independent; for comparison with the prototype only, not for "
        "inference. Adds about an hour to a summary of every epoch.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    decode_dir, output_dir = args.decode_dir.resolve(), args.output_dir.resolve()
    if output_dir == decode_dir:
        parser.error("--output-dir must not be the decode folder itself")
    # before the workers start: they inherit this limit
    old_soft, new_soft = raise_open_file_limit()
    print(f"open-file soft limit {old_soft} -> {new_soft}", flush=True)
    epochs = args.epochs or canonical_epochs(decode_dir)
    if not epochs:
        sys.exit(f"No analysed epoch's HPC decode in {decode_dir}.")
    discovery: dict[str, Any] = {
        "rule": (
            "the epochs given with --epochs"
            if args.epochs
            else "every epoch the epoch register analyses with a canonical HPC "
            f"decode file in {decode_dir}"
        ),
        "n_epochs": len(epochs),
        "coverage": "Coverage against the canonical epoch list is the audit's "
        "job; this script does not check it.",
    }
    if not args.epochs:
        discovery["n_with_all_files"] = sum(
            all(path.exists() for path in decode_files(e, decode_dir).values())
            for e in epochs
        )

    provenance = provenance_record(ROOT_DIR)
    code = code_version(provenance)
    settings = run_settings()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    recomputed: dict[str, str] = {}
    to_run = []
    for epoch_id in epochs:
        paths = output_paths(epoch_id, output_dir)
        previous = (
            json.loads(paths["validation"].read_text())
            if paths["validation"].exists()
            else None
        )
        reason = (
            "--overwrite"
            if args.overwrite
            else reuse_reason(
                previous,
                settings=settings,
                code=code,
                decode_stats=decode_file_stats(epoch_id, decode_dir),
                windows_written=paths["windows"].exists(),
            )
        )
        if reason is None and previous is not None:
            records[epoch_id] = previous
            print(f"{epoch_id}: reused (same commit, settings and decode files)")
        else:
            recomputed[epoch_id] = str(reason)
            to_run.append(epoch_id)
            print(f"{epoch_id}: to compute ({reason})", flush=True)

    def write_manifest() -> None:
        write_json(
            output_dir / "manifest.json",
            {
                "produced_by": PRODUCED_BY,
                "provenance": provenance,
                "settings": settings,
                "window_level_stats": args.window_level_stats,
                "decode_dir": str(decode_dir),
                "discovery": discovery,
                "reused": sorted(set(records) - set(recomputed)),
                "recomputed": dict(sorted(recomputed.items())),
                "epochs": {k: records[k] for k in sorted(records)},
            },
        )

    print(
        f"{len(to_run)} of {len(epochs)} epochs to run on {args.n_workers} workers; "
        f"logs in {output_dir / 'logs'}",
        flush=True,
    )
    kwargs = {"decode_dir": decode_dir, "output_dir": output_dir, "code": code}
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
    summarize(output_dir, ok, window_level=args.window_level_stats)
    (output_dir / "README.md").write_text(
        readme(records, settings, provenance, discovery, recomputed)
    )
    failed = [k for k, r in records.items() if r["status"] == "failed"]
    if failed:
        sys.exit(f"{len(failed)} epochs failed: {', '.join(sorted(failed))}")


if __name__ == "__main__":
    main()
