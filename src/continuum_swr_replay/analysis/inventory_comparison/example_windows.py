"""The memoryless windows of a drawn example, recomputed and checked.

The detection comparison's example figures
(``visualization.inventory_examples.draw_example``) draw the memoryless window
posteriors of the plotted span. They are not stored, so they are re-decoded
from the decode's marks and encoding model (:func:`recompute_windows`), from one
epoch's inputs (:func:`load_epoch`), and checked against the windows the
memoryless run stored: a figure is refused when they disagree beyond a device's
rounding (:data:`SPREAD_RTOL`, :data:`CENTER_TIE_RTOL`). Run them on the device
the memoryless run used (``runs.device.use_memoryless_device``).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from continuum_swr_replay.analysis.events import (
    analysed_time,
    swr_source,
)
from continuum_swr_replay.analysis.hippocampus.detection_inputs import decode_marks
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.detect import (
    CandidateSequences,
    WindowDecisions,
    sequence_dispersion_cm,
    window_decisions,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    check_marks_match_encoding,
    interior_bin_distances_cm,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.shuffle import (
    candidate_window_log_likelihood,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.window_statistics import (
    normalize_windows,
    windows_holding,
)
from continuum_swr_replay.analysis.inventory_comparison import inventory_run
from continuum_swr_replay.data_loaders.lfp import get_ripple_band_inputs
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.identifiers import nwb_file_name
from continuum_swr_replay.decoding.results import open_results
from continuum_swr_replay.intervals import interval_mask
from continuum_swr_replay.paths import processed_data_dir
from continuum_swr_replay.runs.io import output_path
from continuum_swr_replay.tables.io import read_table

__all__ = [
    "CENTER_TIE_RTOL",
    "SPREAD_RTOL",
    "EpochInputs",
    "load_epoch",
    "recompute_windows",
]

SPREAD_RTOL = 1e-2
"""How far a recomputed spread or jump may sit from the stored one, relative.

The stored statistics were computed on a GPU under the memoryless run's XLA
flags, and these are too (``runs.device.use_memoryless_device``); one GPU setting
against another moved spreads by up to 3.5e-3 relative (the memoryless run's
``known_limits``). On a CPU the pilot epoch's sat within 3.3e-3, and the 8
Phase 4 windows a CPU refused within 6e-3; on the GPU those 8 sat within
4.5e-4. Also how near its threshold a spread or jump must be for its gate to
flip."""
CENTER_TIE_RTOL = 1e-3
"""How near the minimum the stored center's spread, under the recomputed
posterior, must be for the center to move: a tie between two bins, which a
device's rounding breaks either way. On the pilot epoch 59 of 14,360 windows
moved, each to the next bin, within 3e-5. On a CPU, 8 of Phase 4's 160 example
figures (run 30cf1cc432e9) moved a center by up to 2.2e-3; on the GPU, under
the memoryless run's flags, their centers moved within 3e-5."""


@dataclass(frozen=True, eq=False)
class EpochInputs:
    """Everything one epoch's figures read.

    Attributes
    ----------
    identifier : str
    time : ndarray, shape (n_time,)
        The decode grid.
    dot : DataFrame
        The decode's ``data_over_time``, indexed by ``time``.
    results : xarray.Dataset
        The decode's ``results.nc``, opened lazily.
    spike_times, spike_waveform_features : list of ndarray
        The decode's marks, per tetrode, in the encoding model's order.
    encoding_model : dict
    bin_distance_cm : ndarray, shape (n_bins, n_bins)
    bin_centers_cm : ndarray, shape (n_bins,)
        The interior bins the memoryless posteriors are over.
    candidates : CandidateSequences
        The memoryless run's stored windows and candidates.
    missing_windows : ndarray of bool, shape (n_windows,)
        Windows holding a bin of the artifact mask, which the run left
        undecoded.
    band_time : ndarray, shape (n_band,)
    ripple_band : ndarray, shape (n_band, n_kept)
        The stored ripple band of the tetrodes the SWR source kept.
    excluded : DataFrame
        Excluded time, ``start_time`` and ``end_time``.
    inventory, set_aside : DataFrame
        The comparison's tables for the epoch.
    """

    identifier: str
    time: NDArray
    dot: pd.DataFrame
    results: xr.Dataset
    spike_times: list[NDArray]
    spike_waveform_features: list[NDArray]
    encoding_model: dict
    bin_distance_cm: NDArray
    bin_centers_cm: NDArray
    candidates: CandidateSequences
    missing_windows: NDArray
    band_time: NDArray
    ripple_band: NDArray
    excluded: pd.DataFrame
    inventory: pd.DataFrame
    set_aside: pd.DataFrame


def load_epoch(identifier: str, *, run_dir: Path, memoryless_dir: Path) -> EpochInputs:
    """One epoch's inputs: the decode, the memoryless run, the band, the tables.

    Raises
    ------
    ValueError
        The decode's marks are not the ones its encoding model was fitted on
        (``check_marks_match_encoding``), the stored windows are not the
        decode grid's, or the band does not hold every tetrode the SWR source
        kept.
    """
    dot = pd.read_pickle(inventory_run.decode_path(identifier))
    time = dot.index.to_numpy(dtype=np.float64)
    marks = decode_marks(identifier)
    encoding_model = marks["encoding_model"]
    check_marks_match_encoding(
        encoding_model, marks["spike_times"], marks["spike_waveform_features"]
    )
    environment = encoding_model["environment"]
    interior = np.asarray(environment.is_track_interior_).ravel()
    candidates: CandidateSequences = joblib.load(
        memoryless_dir / f"{identifier}_memoryless_candidates.pkl"
    )
    config = candidates.config
    n_windows = 1 + (time.size - 1 - config.window_n_samples) // config.stride_n_samples
    if candidates.midpoint_s.size != n_windows:
        raise ValueError(
            f"The stored candidates have {candidates.midpoint_s.size} windows; the "
            f"decode grid of {identifier} makes {n_windows}."
        )
    # the runner's is_missing: a bin inside a closed interval of the mask
    is_missing = interval_mask(time, swr_source.artifact_intervals(identifier))
    missing_windows = windows_holding(
        is_missing[:-1], config.window_n_samples, config.stride_n_samples
    )

    animal, day, epoch = (str(dot[c].iloc[0]) for c in ("animal", "day", "epoch"))
    band = get_ripple_band_inputs(nwb_file_name(animal, day), epoch)
    kept_ids = swr_source.ripple_power_inputs(identifier)["kept_electrode_ids"]
    kept = np.isin(band["electrode_ids"], kept_ids)
    if kept.sum() != len(kept_ids):
        raise ValueError(
            f"The band of {identifier} holds {band['electrode_ids'].tolist()}, not "
            f"every tetrode the SWR source kept ({list(kept_ids)})."
        )
    in_span = (band["time"] >= time[0]) & (band["time"] <= time[-1])

    def table(name: str) -> pd.DataFrame:
        return read_table(output_path(identifier, run_dir, name))

    return EpochInputs(
        identifier=identifier,
        time=time,
        dot=dot,
        # h5netcdf, as the decode wrote it (open_results): the netCDF4 library
        # fails on this file once Spyglass has loaded its own HDF5
        results=open_results(
            decode_paths(identifier, processed_data_dir()).results, cache=False
        ),
        spike_times=marks["spike_times"],
        spike_waveform_features=marks["spike_waveform_features"],
        encoding_model=encoding_model,
        bin_distance_cm=interior_bin_distances_cm(environment),
        bin_centers_cm=np.asarray(environment.place_bin_centers_).ravel()[interior],
        candidates=candidates,
        missing_windows=missing_windows,
        band_time=band["time"][in_span],
        ripple_band=band["ripple_band"][in_span][:, kept],
        excluded=analysed_time.excluded_intervals(identifier),
        inventory=table("inventory"),
        set_aside=table("set_aside"),
    )


def recompute_windows(
    inputs: EpochInputs, first: int, last: int
) -> tuple[NDArray, WindowDecisions, dict[str, Any]]:
    """The memoryless windows ``first`` to ``last``, recomputed and checked.

    The windows are re-decoded from the marks and the encoding model
    (``candidate_window_log_likelihood``), one window past ``last`` included
    so the last plotted jump has its successor, the windows the run left
    undecoded set to NaN, normalized and gated (``window_decisions``) at the
    speed the run read.

    Returns
    -------
    posterior : ndarray, shape (last - first + 1, n_bins)
    decisions : WindowDecisions
        Of the plotted windows.
    check : dict
        Against the stored windows: ``n_windows``; ``center_ties``, the
        centers that moved to a tied bin; the spread, jump and kept gates'
        ``_flips``; ``spread_max_rel_diff``, and ``jump_max_rel_diff`` over
        the jumps no tie moved, where both are finite; and for every stored
        candidate inside the windows, its ``dispersion_cm`` (recomputed from
        the stored center where a center moved to a tie) and
        ``mean_posterior_spread_cm``, stored and recomputed.

    Raises
    ------
    ValueError
        A spread, or a jump no tie moved, is further than :data:`SPREAD_RTOL`
        from the stored one; a center moved off a tie
        (:data:`CENTER_TIE_RTOL`); the usable or speed gate differs; a spread
        or jump gate flips away from its threshold and from any tie; a NaN
        falls elsewhere; or a candidate's dispersion or mean spread is further
        than :data:`SPREAD_RTOL`.
    """
    candidates = inputs.candidates
    config = candidates.config
    stop = min(last + 1, candidates.midpoint_s.size - 1)
    window_log_likelihood = candidate_window_log_likelihood(
        first,
        stop,
        inputs.time,
        inputs.spike_times,
        inputs.spike_waveform_features,
        inputs.encoding_model,
        config,
    )
    window_log_likelihood[inputs.missing_windows[first : stop + 1]] = np.nan
    posterior = normalize_windows(window_log_likelihood)
    speed = np.interp(
        candidates.midpoint_s[first : stop + 1],
        inputs.time,
        inputs.dot["head_speed"].to_numpy(dtype=np.float64),
    )
    full = window_decisions(posterior, inputs.bin_distance_cm, speed, config)
    n = last - first + 1
    plotted = slice(0, n)
    decisions = WindowDecisions(
        center_index=full.center_index[plotted],
        spread_cm=full.spread_cm[plotted],
        jump_cm=full.jump_cm[plotted],
        gates=type(full.gates)(
            **{
                name: getattr(full.gates, name)[plotted]
                for name in (
                    "posterior_usable",
                    "speed_eligible",
                    "spread_below_threshold",
                    "jump_below_threshold",
                )
            }
        ),
    )
    stored = candidates.decisions
    window = slice(first, last + 1)
    check: dict[str, Any] = {"first_window": first, "last_window": last, "n_windows": n}
    problems = []

    # a center may move only to a tie: the stored center, under the recomputed
    # posterior, spreads within CENTER_TIE_RTOL of the recomputed minimum
    old_center = stored.center_index[first : stop + 1]
    moved = full.center_index != old_center
    tied = np.zeros_like(moved)
    distance = inputs.bin_distance_cm
    squared = np.where(np.isfinite(distance), distance**2, np.inf)
    for k in np.flatnonzero(moved):
        if old_center[k] >= 0 and full.center_index[k] >= 0:
            spread_at_stored = np.sqrt(posterior[k] @ squared[:, old_center[k]])
            tied[k] = spread_at_stored <= full.spread_cm[k] * (1 + CENTER_TIE_RTOL)
    if (moved & ~tied).any():
        problems.append(f"{int((moved & ~tied).sum())} centers moved off a tie")
    check["center_ties"] = int(moved[:n].sum())
    # a jump reads its own center and its successor's; the last repeats its
    # predecessor's
    jump_moved = moved.copy()
    jump_moved[:-1] |= moved[1:]
    if moved.size > 1:
        jump_moved[-1] |= moved[-2]
    jump_moved = jump_moved[:n]

    for name, unaffected in (
        ("spread_cm", np.ones(n, dtype=bool)),
        ("jump_cm", ~jump_moved),
    ):
        new, old = getattr(decisions, name), getattr(stored, name)[window]
        if not np.array_equal(np.isnan(new), np.isnan(old)):
            problems.append(f"{name} is NaN in other windows than the stored")
        both = np.isfinite(new) & np.isfinite(old) & unaffected
        relative = np.abs(new[both] - old[both]) / np.maximum(np.abs(old[both]), 1.0)
        check[f"{name.removesuffix('_cm')}_max_rel_diff"] = (
            float(relative.max()) if relative.size else 0.0
        )
        if relative.size and relative.max() > SPREAD_RTOL:
            problems.append(f"{name} differs by {relative.max():.3g} relative")

    def flipped(name: str) -> NDArray:
        flips: NDArray = (
            getattr(decisions.gates, name) != getattr(stored.gates, name)[window]
        )
        return flips

    def near(values: NDArray, threshold: float) -> NDArray:
        close: NDArray = np.abs(values - threshold) <= SPREAD_RTOL * threshold
        return close

    for name in ("posterior_usable", "speed_eligible"):
        if flipped(name).any():
            problems.append(
                f"the {name} gate differs in {int(flipped(name).sum())} windows"
            )
    # a spread gate may flip only at its threshold; a jump gate there or where
    # a center tie moved the jump; kept only where one of those flipped
    spread_flips = flipped("spread_below_threshold")
    jump_flips = flipped("jump_below_threshold")
    explained = {
        "spread_below_threshold": near(decisions.spread_cm, config.spread_threshold_cm),
        "jump_below_threshold": jump_moved
        | near(decisions.jump_cm, config.jump_threshold_cm),
        "kept": spread_flips | jump_flips,
    }
    for name, allowed in explained.items():
        flips = flipped(name)
        check[f"{name}_flips"] = int(flips.sum())
        if (flips & ~allowed).any():
            problems.append(
                f"the {name} gate differs in {int((flips & ~allowed).sum())} windows "
                "away from its threshold and from any center tie"
            )

    inside = np.flatnonzero(
        (candidates.runs[:, 0] >= first) & (candidates.runs[:, 1] <= last)
    )
    # a dispersion reads the centers, so a center moved to a tie would move it
    # too; it reads the stored center there, as the jump check sets those aside
    recomputed_dispersion = sequence_dispersion_cm(
        candidates.runs[inside] - first,
        decisions.gates.kept,
        np.where(moved & tied, old_center, full.center_index)[:n],
        inputs.bin_distance_cm,
    )
    check["candidates"] = []
    for index, dispersion in zip(inside, recomputed_dispersion, strict=True):
        retained = np.concatenate(
            [
                np.arange(start, end + 1)
                for start, end in candidates.pre_merge_runs[candidates.pieces[index]]
            ]
        )
        record = {
            "first_window": int(candidates.runs[index, 0]),
            "dispersion_cm": [
                float(candidates.dispersion_cm[index]),
                float(dispersion),
            ],
            "mean_posterior_spread_cm": [
                float(stored.spread_cm[retained].mean()),
                float(decisions.spread_cm[retained - first].mean()),
            ],
        }
        check["candidates"].append(record)
        for name in ("dispersion_cm", "mean_posterior_spread_cm"):
            old, new = record[name]
            if abs(new - old) > SPREAD_RTOL * max(abs(old), 1.0):
                problems.append(
                    f"candidate at window {record['first_window']}: {name} "
                    f"{new:.6g}, stored {old:.6g}"
                )
    if problems:
        raise ValueError(
            f"{inputs.identifier} windows {first} to {last}, recomputed, are not "
            f"the stored ones: {'; '.join(problems)}."
        )
    return posterior[plotted], decisions, check
