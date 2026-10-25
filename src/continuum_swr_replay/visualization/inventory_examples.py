"""Sampled example figures of the detection comparison: the draw and the figure.

What the inventory comparison's examples (``scripts/figures/generate_inventory_examples.py``)
and the event features' examples (``scripts/figures/generate_event_feature_examples.py``)
share: the seeded draw of events from a table (:func:`draw_examples`), one
epoch's inputs (:func:`load_epoch`), the memoryless windows recomputed for the
plotted span and checked against the stored ones (:func:`recompute_windows`),
on the device the memoryless run used (:func:`use_memoryless_device`), and one
drawn event's figure (:func:`draw_example`). Design:
``docs/design/replay_detection_comparison/03-inventory-comparison.md``
(Reporting, item 5).
"""

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import jax
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
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
from continuum_swr_replay.analysis.inventory_comparison.overlap import (
    MATCHED,
    UNMATCHED,
)
from continuum_swr_replay.data_loaders.lfp import get_ripple_band_inputs
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.results import open_results
from continuum_swr_replay.figure_utilities import WONG
from continuum_swr_replay.intervals import interval_mask
from continuum_swr_replay.runs.device import xla_flags
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.visualization.palettes import COLORS
from continuum_swr_replay.visualization.plots import plot_linear_track_graph_strip

__all__ = [
    "CENTER_TIE_RTOL",
    "EXAMPLE_SEED",
    "N_PER_SIDE",
    "PAD_S",
    "PANELS_CAPTION",
    "PENDING",
    "SPREAD_RTOL",
    "EpochInputs",
    "detector_color",
    "detector_label",
    "draw_example",
    "draw_examples",
    "figure_name",
    "load_epoch",
    "recompute_device",
    "recompute_windows",
    "use_memoryless_device",
]

EXAMPLE_SEED = 20261017
"""The run seed every draw is seeded from, with its cell, detector and animal."""
N_PER_SIDE = 2
ROLES: Mapping[str, str] = MappingProxyType(
    {UNMATCHED: "example", MATCHED: "counterexample"}
)
"""What a drawn event of each membership is; ``unknown`` events are never drawn."""
UNIT = ("cell", "detector", "animal")
"""What each draw is made within."""
PAD_S = 0.5
"""Plotted time either side of the drawn event."""

PENDING = "pending the Phase 2 pilot checks"
PENDING_SENTENCE = (
    "Every figure shows the memoryless inventory and its window posteriors, so "
    f"every example is marked {PENDING}."
)

SPREAD_RTOL = 1e-2
"""How far a recomputed spread or jump may sit from the stored one, relative.

The stored statistics were computed on a GPU under the memoryless run's XLA
flags, and these are too (:func:`use_memoryless_device`); one GPU setting
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


def use_memoryless_device() -> None:
    """Recompute where the memoryless run computed: a GPU, under its XLA flags.

    Near-tied window centers are broken by the device's rounding, so a CPU
    moves more of them than the check allows. The GPU is set over any
    ``JAX_PLATFORMS`` the shell or a caller chose; with no GPU, JAX raises. Call
    before JAX starts its backend, which reads ``XLA_FLAGS`` once.
    """
    os.environ["XLA_FLAGS"] = xla_flags()
    os.environ["JAX_PLATFORMS"] = "cuda"
    jax.config.update("jax_platforms", "cuda")


def recompute_device() -> dict[str, str]:
    """The device the windows are recomputed on and its XLA flags, for the
    manifest."""
    return {
        "device": str(jax.devices()[0]),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }


GATE_ROWS: Mapping[str, str] = MappingProxyType(
    {
        "speed_eligible": "ML speed",
        "spread_below_threshold": "ML spread",
        "jump_below_threshold": "ML jump",
        "kept": "ML kept",
    }
)
"""The stored memoryless gates drawn, with their row labels."""
POSITION_STATES = ("Local", "Non-Local Continuous", "Non-Local Fragmented")
"""The state-space states with a position; ``No-Spike`` has none."""

ANIMAL_POSITION_COLOR = "magenta"
"""The animal's position over a posterior, as in Figure 1 and the event gallery."""
MEMORYLESS_COLOR = WONG[5]
"""The memoryless inventories: Wong blue, which no ``COLORS`` entry uses."""
WINDOW_CENTER_COLOR = WONG[2]
"""Each memoryless window's center (sky blue), as in the memoryless comparison."""
TRACK_MARGIN = 0.015
"""The y margin of each posterior and its track strip, as a fraction of the
track's length on each side, so a position at either end of the track is drawn
inside the axes."""


def detector_color(detector: str) -> str:
    """The paper's color for a detector's events: ``COLORS`` by family.

    Every SWR definition takes ``COLORS["SWR"]``, ``mua`` ``COLORS["MUA"]`` and
    ``state_space`` ``COLORS["Non-Local"]``; both memoryless inventories take
    :data:`MEMORYLESS_COLOR`. The bars' rows name the detector.
    """
    if detector in inventory_run.SWR_DEFINITIONS:
        return str(COLORS["SWR"])
    if detector == "mua":
        return str(COLORS["MUA"])
    if detector == "state_space":
        return str(COLORS["Non-Local"])
    return str(MEMORYLESS_COLOR)


def _track_key(ax: Any, environment: Any, max_position: float, label: str) -> None:
    """The linearized track beside a posterior, as in the event gallery.

    A strip flush to the posterior's left (``plot_linear_track_graph_strip``,
    the decode's own ``edge_order`` and ``edge_spacing``), each arm labeled by
    its ``edge_id`` at its middle. The strip takes the posterior's y label.
    Both share one y range, the track with :data:`TRACK_MARGIN` either side,
    and the strip keeps its 0 and maximum ticks.
    """
    strip = ax.inset_axes([-0.075, 0.0, 0.06, 1.0])
    graph = environment.track_graph
    plot_linear_track_graph_strip(
        graph,
        environment.edge_order,
        environment.edge_spacing,
        ax=strip,
        max_position=max_position,
        node_size=6,
    )
    n_edges = len(environment.edge_order)
    spacing = np.atleast_1d(np.asarray(environment.edge_spacing, dtype=float))
    # one gap after each edge but the last, as plot_graph_as_1D lays them
    gaps = np.resize(spacing, n_edges - 1) if spacing.size == 1 else spacing
    # plot_graph_as_1D's per-edge colors
    colors = getattr(plt.get_cmap("tab10"), "colors")  # noqa: B009
    position = 0.0
    for index, edge in enumerate(environment.edge_order):
        length = float(graph.edges[edge]["distance"])
        strip.text(
            -0.8,
            position + length / 2,
            f"a{graph.edges[edge]['edge_id']}",
            fontsize=4.5,
            color=colors[index % len(colors)],
            ha="center",
            va="center",
        )
        position += length + (gaps[index] if index < n_edges - 1 else 0.0)
    margin = TRACK_MARGIN * max_position
    strip.set_ylim(-margin, max_position + margin)
    strip.set_ylabel(label, fontsize=7)
    strip.tick_params(labelsize=6)
    ax.set_ylabel("")
    ax.set_ylim(-margin, max_position + margin)
    ax.tick_params(labelleft=False)


def _unit_rng(seed: int, unit: tuple[str, ...]) -> np.random.Generator:
    """A generator for one (cell, detector, animal), from the run seed.

    A keyed digest, not ``hash()``, which changes between interpreters.
    """
    digest = hashlib.blake2b("/".join(unit).encode("utf-8"), digest_size=8).digest()
    spawn_key = (
        int.from_bytes(digest[:4], "little"),
        int.from_bytes(digest[4:], "little"),
    )
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=spawn_key))


def draw_examples(
    cells: pd.DataFrame,
    *,
    seed: int = EXAMPLE_SEED,
    n: int = N_PER_SIDE,
    column: str = "membership",
    roles: Mapping[str, str] = ROLES,
) -> pd.DataFrame:
    """The events the examples' sampling rule draws from a run's ``cells`` table.

    Parameters
    ----------
    cells : DataFrame
        The runner's ``cells`` table, over any number of epochs, or any
        table with its ``cell``, ``detector``, ``animal``, ``day``, ``epoch``
        and ``event_key`` columns and ``column``.
    seed : int
        The run seed.
    n : int
        Events drawn of each value of ``column``, at most.
    column : str
        The column the events are drawn by.
    roles : mapping
        Each value of ``column`` drawn -> the role of its events, in the
        order drawn; events of any other value are never drawn.

    Returns
    -------
    DataFrame
        The drawn rows of ``cells`` with ``role`` (by default ``example`` for
        an ``unmatched`` event, ``counterexample`` for a ``matched`` one) and
        ``draw`` (1 to ``n``, in the order drawn), ordered by cell, detector
        and animal, then by ``roles``' order, each in draw order.
    """
    out = []
    for unit, events in cells.groupby(list(UNIT), sort=True):
        rng = _unit_rng(seed, tuple(str(value) for value in unit))
        for value, role in roles.items():
            pool = events[events[column] == value].sort_values(
                ["day", "epoch", "event_key"], kind="stable"
            )
            picks = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
            out.append(
                pool.iloc[picks].assign(role=role, draw=np.arange(1, len(picks) + 1))
            )
    columns = [*cells.columns, "role", "draw"]
    if not out:
        return pd.DataFrame(columns=columns)
    return pd.concat(out, ignore_index=True)[columns]


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
    band = get_ripple_band_inputs(f"{animal}{day}_.nwb", epoch)
    kept_ids = swr_source.ripple_power_inputs(identifier)["kept_electrode_ids"]
    kept = np.isin(band["electrode_ids"], kept_ids)
    if kept.sum() != len(kept_ids):
        raise ValueError(
            f"The band of {identifier} holds {band['electrode_ids'].tolist()}, not "
            f"every tetrode the SWR source kept ({list(kept_ids)})."
        )
    in_span = (band["time"] >= time[0]) & (band["time"] <= time[-1])

    def table(name: str) -> pd.DataFrame:
        return read_table(inventory_run.output_path(identifier, run_dir, name))

    return EpochInputs(
        identifier=identifier,
        time=time,
        dot=dot,
        # h5netcdf, as the decode wrote it (open_results): the netCDF4 library
        # fails on this file once Spyglass has loaded its own HDF5
        results=open_results(
            decode_paths(identifier, inventory_run.DECODE_ROOT).results, cache=False
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


def _spans(time: NDArray, mask: NDArray) -> list[tuple[float, float]]:
    """``(start, width)`` of each run of True samples, one step wide each."""
    if not mask.any():
        return []
    step = float(np.median(np.diff(time))) if time.size > 1 else 0.0
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return [
        (time[a], time[b] - time[a] + step) for a, b in zip(starts, stops, strict=True)
    ]


def detector_label(detector: str) -> str:
    """A detector's name as the figures print it: a memoryless one marked †
    (pending)."""
    return f"{detector}†" if detector.startswith("memoryless") else detector


def draw_example(
    inputs: EpochInputs,
    drawn: pd.Series,
    *,
    against: list[str],
    path: Path,
    run_id: str,
    seed: int = EXAMPLE_SEED,
    note: str = "",
) -> dict[str, Any]:
    """One drawn event's figure, written to ``path``; returns its record.

    ``run_id`` is the inventory run's and ``seed`` the run seed the event was
    drawn by, both named in the title; ``note``, if given, is the title's last
    line.

    Raises
    ------
    ValueError
        As :func:`recompute_windows`, or the event is not in the epoch's
        inventory.
    """
    event = inputs.inventory[inputs.inventory.event_key == drawn.event_key]
    if len(event) != 1:
        raise ValueError(f"{drawn.event_key} is not one event of the inventory.")
    start, end = float(event.event_start_s.iloc[0]), float(event.event_end_s.iloc[0])
    t0, t1 = start - PAD_S, end + PAD_S
    candidates = inputs.candidates
    midpoints = candidates.midpoint_s
    first = int(np.searchsorted(midpoints, t0, side="left"))
    last = int(np.searchsorted(midpoints, t1, side="right")) - 1
    posterior, decisions, check = recompute_windows(inputs, first, last)
    window_time = midpoints[first : last + 1]

    def rel(values: NDArray) -> NDArray:
        return np.asarray(values, dtype=np.float64) - start

    in_plot = (inputs.time >= t0) & (inputs.time <= t1)
    time = inputs.time[in_plot]
    dot = inputs.dot.loc[in_plot]
    speed = dot["head_speed"].to_numpy(dtype=np.float64)
    non_local = (
        dot["Non-Local_Continuous_prob"] + dot["Non-Local_Fragmented_prob"]
    ).to_numpy(dtype=np.float64)

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(7.2, 10.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.6, 1.6, 1.0, 0.7, 1.8, 1.5]},
        layout="constrained",
    )
    raster, state_space, memoryless, gates, speed_ax, band_ax, bars = axes

    spikes = [rel(times[(times >= t0) & (times <= t1)]) for times in inputs.spike_times]
    raster.eventplot(spikes, colors="k", linelengths=0.8, linewidths=0.4)
    raster.set_ylabel("tetrode", fontsize=7)
    raster.set_ylim(-0.5, len(spikes) - 0.5)

    posterior_ss = inputs.results.acausal_posterior.sel(time=slice(t0, t1))
    ss_time = posterior_ss.time.to_numpy()
    values = posterior_ss.to_numpy()
    states = inputs.results.state.to_numpy()
    position = inputs.results.position.to_numpy()
    by_state = [states == name for name in POSITION_STATES]
    bin_position = position[by_state[0]]
    if not all(np.array_equal(position[s], bin_position) for s in by_state):
        raise ValueError("The states with a position do not share one place grid.")
    summed = sum(values[:, s] for s in by_state)
    # the bins off the track interior hold NaN
    on_track = ~np.isnan(summed).all(axis=0)
    summed = summed[:, on_track]
    state_space.pcolormesh(
        rel(ss_time),
        bin_position[on_track],
        summed.T,
        shading="nearest",
        cmap="bone_r",
        vmin=0.0,
        vmax=max(float(np.quantile(summed, 0.99)), 1e-12),
        rasterized=True,
    )
    animal_position = dot["linear_position"].to_numpy(dtype=np.float64)
    environment = inputs.encoding_model["environment"]
    max_position = float(np.ceil(np.nanmax(environment.place_bin_edges_)))
    state_space.plot(
        rel(time), animal_position, color=ANIMAL_POSITION_COLOR, lw=0.8, alpha=0.9
    )
    _track_key(state_space, environment, max_position, "state-space\nposterior (cm)")

    usable = np.isfinite(posterior).all(axis=1)
    shown = np.where(usable[:, None], posterior, 0.0)
    memoryless.pcolormesh(
        rel(window_time),
        inputs.bin_centers_cm,
        shown.T,
        shading="nearest",
        cmap="bone_r",
        vmin=0.0,
        vmax=max(float(np.quantile(shown, 0.99)), 1e-12),
        rasterized=True,
    )
    has_center = decisions.center_index >= 0
    memoryless.scatter(
        rel(window_time[has_center]),
        inputs.bin_centers_cm[decisions.center_index[has_center]],
        s=1.5,
        color=WINDOW_CENTER_COLOR,
        linewidths=0,
    )
    memoryless.plot(
        rel(time), animal_position, color=ANIMAL_POSITION_COLOR, lw=0.8, alpha=0.9
    )
    _track_key(memoryless, environment, max_position, "memoryless†\nwindows (cm)")

    stored = candidates.decisions.gates
    rows = {
        f"p(non-local) > {DEFAULT_NON_LOCAL_THRESHOLD}": (
            time,
            non_local > DEFAULT_NON_LOCAL_THRESHOLD,
        ),
        **{
            label: (window_time, getattr(stored, name)[first : last + 1])
            for name, label in GATE_ROWS.items()
        },
        f"speed < {DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC} cm/s": (
            time,
            speed < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
        ),
    }
    row_colors = [COLORS["Non-Local"], *[MEMORYLESS_COLOR] * len(GATE_ROWS), "0.4"]
    for row, ((times, mask), color) in enumerate(
        zip(rows.values(), row_colors, strict=True)
    ):
        gates.broken_barh(
            [(s - start, w) for s, w in _spans(times, np.asarray(mask, bool))],
            (row - 0.4, 0.8),
            color=color,
            linewidth=0,
        )
    gates.set_yticks(range(len(rows)), list(rows), fontsize=6)
    gates.set_ylim(len(rows) - 0.5, -0.5)

    speed_ax.plot(rel(time), speed, color="k", lw=0.7)
    speed_ax.axhline(
        DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC, color="red", ls="--", lw=0.8
    )
    speed_ax.set_ylabel("speed\n(cm/s)", fontsize=7)

    in_band = (inputs.band_time >= t0) & (inputs.band_time <= t1)
    band = inputs.ripple_band[in_band]
    scale = float(np.nanquantile(np.abs(band), 0.99)) if band.size else 1.0
    offsets = -2.0 * scale * np.arange(band.shape[1])
    band_ax.plot(rel(inputs.band_time[in_band]), band + offsets, color="k", lw=0.3)
    band_ax.set_yticks([])
    band_ax.set_ylabel(f"ripple band\n({band.shape[1]} kept tetrodes)", fontsize=7)

    detectors = list(inventory_run.DETECTORS)
    for row, detector in enumerate(detectors):
        own = inputs.inventory[inputs.inventory.detector == detector]
        own = own[(own.event_end_s >= t0) & (own.event_start_s <= t1)]
        moving = ~own.whole_event_speed_eligible.to_numpy(dtype=bool)
        for subset, alpha in ((own[~moving], 0.9), (own[moving], 0.3)):
            bars.broken_barh(
                list(
                    zip(rel(subset.event_start_s), subset.interval_span_s, strict=True)
                ),
                (row - 0.4, 0.8),
                color=detector_color(detector),
                alpha=alpha,
                linewidth=0,
            )
        aside = inputs.set_aside[inputs.set_aside.detector == detector]
        aside = aside[(aside.interval_end_s >= t0) & (aside.interval_start_s <= t1)]
        bars.broken_barh(
            list(
                zip(
                    rel(aside.interval_start_s),
                    aside.interval_end_s - aside.interval_start_s,
                    strict=True,
                )
            ),
            (row - 0.4, 0.8),
            facecolor="none",
            edgecolor="0.3",
            linewidth=0.6,
        )
    bars.set_yticks(
        range(len(detectors)), [detector_label(d) for d in detectors], fontsize=6
    )
    bars.set_ylim(len(detectors) - 0.5, -0.5)
    bars.set_xlabel(
        f"time from the event's start (s); event lasts {end - start:.3f} s", fontsize=7
    )

    excluded = inputs.excluded[
        (inputs.excluded.end_time >= t0) & (inputs.excluded.start_time <= t1)
    ]
    for ax in axes:
        for _, interval in excluded.iterrows():
            ax.axvspan(
                interval.start_time - start,
                interval.end_time - start,
                color="0.6",
                alpha=0.35,
                linewidth=0,
            )
        for edge in (0.0, end - start):
            ax.axvline(edge, color="k", lw=0.6, ls=":")
        ax.tick_params(labelsize=6)
    raster.set_xlim(-PAD_S, end - start + PAD_S)
    fig.suptitle(
        f"{drawn.cell}: {detector_label(drawn.detector)} event, {drawn.role} "
        f"({drawn.membership} by {', '.join(detector_label(d) for d in against)})\n"
        f"{drawn.event_key} ({drawn.animal} {drawn.day} {drawn.epoch})\n"
        f"† {PENDING}; inventory run {run_id}; seed {seed}"
        + (f"\n{note}" if note else ""),
        fontsize=7,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {
        **drawn.to_dict(),
        "against": against,
        "event_start_s": start,
        "event_end_s": end,
        "path": str(path),
        "memoryless_check": check,
    }


def figure_name(drawn: pd.Series) -> str:
    return f"{drawn.cell}_{drawn.detector}_{drawn.animal}_{drawn.role}{drawn.draw}.png"


PANELS_CAPTION = (
    f" Each figure shows one event on one time axis, {PAD_S} s either side of "
    "it (dotted lines mark the event): the spikes of each tetrode (the decode's "
    "marks); the state-space posterior summed over the Local and both "
    "Non-Local states, with the animal's linear position (magenta); the "
    "memoryless window posteriors, recomputed for the plotted windows on the "
    "decode's encoding model, with each window's center (sky-blue dots) and the "
    "animal's position. Beside each posterior, the decode's linearized track, "
    "its arms labeled a0-a8. The recomputed windows' spreads, "
    "jumps and candidate statistics equal the stored ones within "
    f"{SPREAD_RTOL} relative, and their centers and gates equal them except "
    "where a center tied between two bins or a statistic sat at its threshold "
    "(manifest.json, memoryless_check). Then the gate masks "
    f"(non-local probability over {DEFAULT_NON_LOCAL_THRESHOLD}; the memoryless "
    "per-window speed, spread, jump and kept gates as stored; speed below "
    f"{DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC} cm/s); the speed; the stored ripple "
    "band of the tetrodes the SWR source kept; and every inventory's events "
    "(compared events solid, moving events light, events set aside outlined), "
    "colored by family as in the paper: SWR definitions orange, MUA purple, "
    "state-space green, memoryless blue. "
    "Excluded time is shaded grey in every panel. " + PENDING_SENTENCE
)
"""What every figure's panels show, after the sampling rule in a caption."""
