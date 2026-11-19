#!/usr/bin/env python
"""Run the port and the reference's own arithmetic on one input, and compare.

The port's detection and significance test run as the figure and the sweep
run them; ``scripts/checks/_memoryless_reference.py`` then recomputes every step
the way the reference's MATLAB does, from the same place fields, the same
window spike counts and the same thresholds, and with the same cell
permutations in the significance test. What differs is only what the port
adapts: continuous center of mass against a place-bin center, spread about
that center, and zero jumps set aside by value against pairs read from the
spikes. The port decodes sorted cells with the reference's ``+1e-3`` Hz floor
under every field, as the reference does; the reference also runs without it,
to show what the floor does.

Two inputs. The Figure 1 simulation of
``scripts/figures/generate_supp_memoryless_comparison.py`` is compared step by
step, so a disagreement can be placed: the posterior, each window's center,
spread and gates, the candidate sequences, their duration, dispersion and
replay verdict, and the p-values. Every cell of
``scripts/detection/sweep_memoryless_detection.py`` is then run through both, and
the candidates and verdicts are totaled, for the acceptance decisions on
thousands of candidates rather than a handful. Neither is scored against the
injected replay: the question is whether the port decides as the reference
would, not whether either is right.

Usage:
    python scripts/checks/compare_memoryless_to_reference.py
    python scripts/checks/compare_memoryless_to_reference.py --n-seeds 1
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from non_local_detector.likelihoods.common import get_spikecount_per_time_bin
from non_local_detector.simulate.sorted_spikes_simulation import (
    TRACK_HEIGHT,
)

from continuum_swr_replay.analysis.detection.memoryless_sequence import sweep
from continuum_swr_replay.analysis.detection.memoryless_sequence.comparison_simulation import (
    CONFIG,
    N_NEURONS,
    SHUFFLE_RUN_SEED,
    simulate_and_decode,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.detect import (
    detect_candidate_sequences,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.likelihood import (
    likelihood_grid,
    sorted_spikes_epoch_log_likelihood,
    window_midpoints_s,
    with_reference_rate_floor,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.shuffle import (
    DEFAULT_N_SHUFFLES,
    SIGNIFICANCE_LEVEL,
    ShuffleSeed,
    cell_identity_permutation,
    shuffle_candidate_sorted,
    sorted_candidate_set_aside_pairs,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.simulation import (
    draw_replay,
    place_field_means,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.sweep_replays import (
    cell_key,
    drawn_replay,
    planned_cells,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.window_statistics import (
    aggregate_windows,
    normalize_windows,
)
from continuum_swr_replay.decoding.results import ENCODING_KEY
from continuum_swr_replay.paths import detection_comparison_dir
from continuum_swr_replay.runs.device import default_jax_platforms
from continuum_swr_replay.runs.provenance import described_commit, produced_by
from continuum_swr_replay.tables.io import write_table

# Sibling-module import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.checks import _memoryless_reference as reference  # noqa: E402

SUMMARY_PATH = Path("paper/reported/inputs/detection/reference_check_summary.csv")
FIGURE_SEED = ShuffleSeed(run=SHUFFLE_RUN_SEED, session="figure-one-simulation")
"""The figure's seed, so both procedures see the same cell permutations."""

SWEEP_TOTALS = (
    "n_windows",
    "windows_posterior_differs_above_0.001",
    "spread_gate_disagreements",
    "jump_gate_disagreements",
    "kept_disagreements",
    "n_kept_port",
    "n_kept_reference",
    "n_candidates_port",
    "n_candidates_reference",
    "n_candidates_identical_bounds",
    "n_replay_port",
    "n_replay_reference",
    "n_published_port",
    "n_published_reference",
    "shuffle_verdicts_differ_on_port_windows",
    "zero_jump_pairs_observed",
    "zero_jump_disagreements_observed",
    "zero_jump_pairs_shuffled",
    "zero_jump_disagreements_shuffled",
)
"""Counts that add up across sweep cells; medians and quantiles do not."""


def window_counts(
    spike_times: list[np.ndarray],
    decode_time: np.ndarray,
    config: MemorylessSequenceConfig,
) -> np.ndarray:
    """Spikes of each cell in each window, binned as the library bins them.

    Spikes are binned on the grid the likelihood was computed on, the
    detector's (``likelihood_grid``) derived from the decode's, of which the
    likelihood keeps every row but the last.

    Parameters
    ----------
    spike_times : list of ndarray
        One per cell.
    decode_time : ndarray, shape (n_decode,)
        The decode grid, the full grid the likelihood was handed.
    config : MemorylessSequenceConfig

    Returns
    -------
    counts : ndarray, shape (n_windows, n_cells)
    """
    time = likelihood_grid(decode_time, config)
    rows = np.stack(
        [
            get_spikecount_per_time_bin(
                times[(times >= time[0]) & (times <= time[-1])], time
            )
            for times in spike_times
        ],
        axis=1,
    ).astype(np.float64)[:-1]
    cumulative = np.concatenate([np.zeros((1, rows.shape[1])), np.cumsum(rows, 0)])
    n_windows = (rows.shape[0] - config.window_n_samples) // config.stride_n_samples + 1
    start = config.stride_n_samples * np.arange(n_windows)
    return cumulative[start + config.window_n_samples] - cumulative[start]


def overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Which runs of ``b`` share a window with each run of ``a``."""
    return (a[:, None, 0] <= b[None, :, 1]) & (b[None, :, 0] <= a[:, None, 1])


VARIANTS = {
    "reference": reference.RATE_MAP_BASELINE_HZ,
    "reference_without_floor": 0.0,
}
"""The reference as published, which the port matches apart from the binned
center, and without its ``+1e-3`` floor, to show what the floor does."""


def shuffle_permutations(seed: ShuffleSeed, first: int, n_cells: int) -> list:
    """The shuffle's draws for one candidate, as the port draws them."""
    return [
        cell_identity_permutation(seed, candidate=first, draw=draw, n=n_cells)
        for draw in range(1, DEFAULT_N_SHUFFLES + 1)
    ]


def p_values(
    counts, rate_hz, config, first, last, baseline_hz, seed
) -> tuple[float, float]:
    permutations = shuffle_permutations(seed, first, rate_hz.shape[0])
    return reference.shuffle_p_values(
        counts[first : last + 1],
        rate_hz,
        config.window_duration_s,
        permutations,
        baseline_hz,
    )


def passes(p_spread: float, p_jump: float) -> bool:
    return bool(p_spread <= SIGNIFICANCE_LEVEL and p_jump <= SIGNIFICANCE_LEVEL)


def compare(
    decoded: dict,
    counts: np.ndarray,
    rate_hz: np.ndarray,
    centers_cm: np.ndarray,
    thresholds: reference.ReferenceThresholds,
    config: MemorylessSequenceConfig,
    baseline_hz: float,
    seed: ShuffleSeed,
    port_set_aside: dict,
) -> tuple[dict, pd.DataFrame]:
    """Every step of the port against one variant of the reference.

    ``port_set_aside`` maps each tested candidate to the pairs the port's
    shuffle set aside in each draw, observed first.
    """
    port = decoded["candidates"]
    bin_size_cm = thresholds.bin_size_cm
    posterior = reference.decode_windows(
        counts, rate_hz, config.window_duration_s, baseline_hz
    )
    ref = reference.detect(
        posterior, decoded["midpoint_s"], decoded["window_speed"], thresholds
    )

    decisions = port.decisions
    usable = ~np.isnan(ref.center_bins) & ~np.isnan(decisions.spread_cm)
    center_gap_cm = np.abs(
        centers_cm[decisions.center_index[usable]]
        - (centers_cm[0] + bin_size_cm * ref.center_bins[usable])
    )
    spread_gap_cm = (decisions.spread_cm - bin_size_cm * ref.spread_bins)[usable]
    speed_ok = decisions.gates.speed_eligible
    ref_spread_pass = bin_size_cm * ref.spread_bins < config.spread_threshold_cm
    ref_jump_pass = bin_size_cm * ref.jump_bins < config.jump_threshold_cm
    identical = (port.runs[:, None, :] == ref.runs[None, :, :]).all(axis=2)
    # the zero-jump rule, in the observed decode and every shuffled draw of
    # every tested candidate, merged gaps included
    zero_jump = {"observed": [0, 0], "shuffled": [0, 0]}
    for i, port_masks in port_set_aside.items():
        first, last = port.runs[i]
        for draw, permutation in enumerate(
            [None, *shuffle_permutations(seed, first, rate_hz.shape[0])]
        ):
            ref_mask = reference.set_aside_pairs(
                counts[first : last + 1],
                rate_hz if permutation is None else rate_hz[permutation],
                config.window_duration_s,
                baseline_hz,
            )
            tally = zero_jump["observed" if draw == 0 else "shuffled"]
            # the last entry only repeats the one before it
            tally[0] += ref_mask.size - 1
            tally[1] += int((ref_mask[:-1] != port_masks[draw][:-1]).sum())

    # the reference's test on each of its own replays, and on each of the
    # port's with the port's draws
    ref_p = {
        j: p_values(counts, rate_hz, config, *ref.runs[j], baseline_hz, seed)
        for j in np.flatnonzero(ref.is_replay)
    }
    same_windows_p = {
        i: p_values(counts, rate_hz, config, *port.runs[i], baseline_hz, seed)
        for i in decoded["shuffles"]
    }
    summary = {
        "n_windows": len(decisions.gates.kept),
        "posterior_max_abs_difference": float(
            np.abs(posterior - decoded["posterior"]).max(initial=0.0)
        ),
        "windows_posterior_differs_above_0.001": int(
            (np.abs(posterior - decoded["posterior"]).max(axis=1) > 1e-3).sum()
        ),
        "center_median_abs_difference_cm": float(np.median(center_gap_cm)),
        "center_p99_abs_difference_cm": float(np.quantile(center_gap_cm, 0.99)),
        "spread_median_difference_cm": float(np.median(spread_gap_cm)),
        "spread_p99_abs_difference_cm": float(np.quantile(np.abs(spread_gap_cm), 0.99)),
        "spread_gate_disagreements": int(
            (
                decisions.gates.spread_below_threshold[speed_ok]
                != ref_spread_pass[speed_ok]
            ).sum()
        ),
        "jump_gate_disagreements": int(
            (
                decisions.gates.jump_below_threshold[speed_ok]
                != ref_jump_pass[speed_ok]
            ).sum()
        ),
        "kept_disagreements": int((decisions.gates.kept != ref.kept).sum()),
        "n_kept_port": int(decisions.gates.kept.sum()),
        "n_kept_reference": int(ref.kept.sum()),
        "n_candidates_port": len(port.runs),
        "n_candidates_reference": len(ref.runs),
        "n_candidates_identical_bounds": int(identical.any(axis=1).sum()),
        "n_replay_port": int(port.is_replay.sum()),
        "n_replay_reference": int(ref.is_replay.sum()),
        "n_published_port": int(sum(r.passes for r in decoded["shuffles"].values())),
        "n_published_reference": int(sum(passes(*p) for p in ref_p.values())),
        "zero_jump_pairs_observed": zero_jump["observed"][0],
        "zero_jump_disagreements_observed": zero_jump["observed"][1],
        "zero_jump_pairs_shuffled": zero_jump["shuffled"][0],
        "zero_jump_disagreements_shuffled": zero_jump["shuffled"][1],
        "shuffle_verdicts_differ_on_port_windows": int(
            sum(
                passes(*same_windows_p[i]) != decoded["shuffles"][i].passes
                for i in decoded["shuffles"]
            )
        ),
    }

    # one row per port candidate, with the reference candidate overlapping it
    # when there is exactly one; then the reference candidates no port
    # candidate overlaps
    shared = overlap(port.runs, ref.runs)
    rows = []
    for i, (first, last) in enumerate(port.runs):
        matches = np.flatnonzero(shared[i])
        row = {
            "port_first": int(first),
            "port_last": int(last),
            "port_duration_s": float(port.duration_s[i]),
            "port_dispersion_cm": float(port.dispersion_cm[i]),
            "port_is_replay": bool(port.is_replay[i]),
            "n_reference_overlapping": len(matches),
        }
        if len(matches) == 1:
            row |= reference_row(ref, matches[0], bin_size_cm, ref_p)
        if i in decoded["shuffles"]:
            result = decoded["shuffles"][i]
            row |= {
                "port_p_spread": result.p_spread,
                "port_p_jump": result.p_jump,
                "port_passes": result.passes,
                "reference_p_spread_on_port_windows": same_windows_p[i][0],
                "reference_p_jump_on_port_windows": same_windows_p[i][1],
            }
        rows.append(row)
    for j in np.flatnonzero(~shared.any(axis=0)):
        rows.append(
            {"n_reference_overlapping": 1} | reference_row(ref, j, bin_size_cm, ref_p)
        )
    return summary, pd.DataFrame(rows)


def reference_row(ref, j: int, bin_size_cm: float, ref_p: dict) -> dict:
    p_spread, p_jump = ref_p.get(j, (np.nan, np.nan))
    return {
        "reference_first": int(ref.runs[j, 0]),
        "reference_last": int(ref.runs[j, 1]),
        "reference_duration_s": float(ref.duration_s[j]),
        "reference_dispersion_cm": float(bin_size_cm * ref.dispersion_bins[j]),
        "reference_is_replay": bool(ref.is_replay[j]),
        "reference_p_spread": p_spread,
        "reference_p_jump": p_jump,
    }


def fields_and_thresholds(
    encoding_model: dict, config: MemorylessSequenceConfig
) -> tuple[np.ndarray, np.ndarray, reference.ReferenceThresholds]:
    """The reference's rate maps in Hz, the bin centers, and its thresholds."""
    environment = encoding_model["environment"]
    interior = np.asarray(environment.is_track_interior_).ravel()
    centers_cm = np.asarray(environment.place_bin_centers_).ravel()[interior]
    bin_size_cm = float(np.diff(centers_cm).mean())
    if not np.allclose(np.diff(centers_cm), bin_size_cm):
        raise RuntimeError("The reference's bin units need evenly spaced bins.")
    # the library's fields are spikes per bin of the decode grid they were
    # fitted on; the reference's are Hz
    rate_hz = (
        np.asarray(encoding_model["place_fields"], dtype=np.float64)[:, interior]
        * config.decode_sampling_frequency
    )
    thresholds = reference.ReferenceThresholds(
        spread_cm=config.spread_threshold_cm,
        jump_cm=config.jump_threshold_cm,
        speed_cm_per_s=config.speed_threshold_cm_per_s,
        merge_gap_cm=config.merge_gap_cm,
        merge_gap_s=config.merge_gap_s,
        minimum_duration_s=config.minimum_duration_s,
        dispersion_cm=config.dispersion_threshold_cm,
        stride_s=config.stride_s,
        bin_size_cm=bin_size_cm,
    )
    return rate_hz, centers_cm, thresholds


def compare_variants(
    decoded: dict,
    encoding_model: dict,
    config: MemorylessSequenceConfig,
    seed: ShuffleSeed,
) -> tuple[dict, pd.DataFrame]:
    """``compare`` under each variant of the reference, keyed by variant."""
    rate_hz, centers_cm, thresholds = fields_and_thresholds(encoding_model, config)
    counts = window_counts(decoded["spike_times"], decoded["binning_time"], config)
    if counts.shape[0] != decoded["posterior"].shape[0]:
        raise RuntimeError(
            f"{counts.shape[0]} windows of counts for "
            f"{decoded['posterior'].shape[0]} decoded windows."
        )
    sliding_model = with_reference_rate_floor(encoding_model, config)
    port_set_aside = {
        i: sorted_candidate_set_aside_pairs(
            int(decoded["candidates"].runs[i, 0]),
            int(decoded["candidates"].runs[i, 1]),
            decoded["binning_time"],
            decoded["spike_times"],
            sliding_model,
            config,
            [
                None,
                *shuffle_permutations(
                    seed, int(decoded["candidates"].runs[i, 0]), rate_hz.shape[0]
                ),
            ],
        )
        for i in decoded["shuffles"]
    }
    summaries, tables = {}, []
    for variant, baseline_hz in VARIANTS.items():
        summary, table = compare(
            decoded,
            counts,
            rate_hz,
            centers_cm,
            thresholds,
            config,
            baseline_hz,
            seed,
            port_set_aside,
        )
        summaries[variant] = summary
        tables.append(table.assign(variant=variant))
    return summaries, pd.concat(tables, ignore_index=True)


def decode_sweep_cell(recording: dict, injected: list, window: tuple, key: str) -> dict:
    """The port on one sweep cell, as the sweep's ``memoryless_outcome`` runs it."""
    time, _, speed = window
    config = CONFIG
    result = sorted_spikes_epoch_log_likelihood(
        time, window[1], injected, recording["sliding_encoding_model"], config
    )
    posterior = normalize_windows(
        aggregate_windows(
            result.log_likelihood, config.window_n_samples, config.stride_n_samples
        )
    )
    midpoint_s = window_midpoints_s(result.time, config)
    window_speed = np.interp(midpoint_s, time, speed)
    candidates = detect_candidate_sequences(
        posterior, midpoint_s, recording["bin_distance_cm"], window_speed, config
    )
    seed = ShuffleSeed(run=sweep.SHUFFLE_RUN_SEED, session=key)
    shuffles = {
        int(i): shuffle_candidate_sorted(
            int(candidates.runs[i, 0]),
            int(candidates.runs[i, 1]),
            time,
            injected,
            recording["sliding_encoding_model"],
            recording["bin_distance_cm"],
            config,
            seed=seed,
            n_shuffles=DEFAULT_N_SHUFFLES,
        )
        for i in np.flatnonzero(candidates.is_replay)
    }
    return {
        "time": result.time,
        "binning_time": time,
        "spike_times": injected,
        "posterior": posterior,
        "midpoint_s": midpoint_s,
        "window_speed": window_speed,
        "candidates": candidates,
        "shuffles": shuffles,
        "seed": seed,
    }


def sweep_totals(n_seeds: int) -> pd.DataFrame:
    """Both procedures on every sweep cell, the counts totaled per variant.

    The cells, placements and injections are the sweep's own
    (``sweep_replays.planned_cells``), so each cell is the one the sweep scored.
    """
    recording = sweep.build_recording()
    means = place_field_means(N_NEURONS, TRACK_HEIGHT)
    cells = planned_cells(sweep.usable_intervals(recording), n_seeds)
    totals = {variant: dict.fromkeys(SWEEP_TOTALS, 0) for variant in VARIANTS}
    for variant in VARIANTS:
        totals[variant] |= {"n_cells": 0, "cells_with_replay_count_differing": 0}
        totals[variant] |= {"cells_with_published_count_differing": 0}
    for number, cell in enumerate(cells):
        injected, bounds = draw_replay(
            recording["spike_times"],
            means,
            drawn_replay(
                cell["path_cm"],
                cell["midpoint_cm"],
                cell["duration_s"],
                cell["direction"],
                cell["at_s"],
            ),
        )
        window = sweep.decode_window(recording, bounds)
        key = cell_key(
            cell["path_cm"],
            cell["midpoint_cm"],
            cell["duration_s"],
            cell["direction"],
            cell["seed"],
        )
        decoded = decode_sweep_cell(recording, injected, window, key)
        summaries, _ = compare_variants(
            decoded, recording["encoding_model"], CONFIG, decoded["seed"]
        )
        for variant, summary in summaries.items():
            total = totals[variant]
            for measure in SWEEP_TOTALS:
                total[measure] += summary[measure]
            total["n_cells"] += 1
            total["cells_with_replay_count_differing"] += int(
                summary["n_replay_port"] != summary["n_replay_reference"]
            )
            total["cells_with_published_count_differing"] += int(
                summary["n_published_port"] != summary["n_published_reference"]
            )
        if number % 400 == 0:
            print(f"  sweep {100 * (number + 1) / len(cells):5.1f}%", flush=True)
    return pd.DataFrame(totals).rename_axis("measure")


def main(argv: Sequence[str] | None = None) -> None:
    # the CPU unless JAX_PLATFORMS names another
    default_jax_platforms("cpu")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-neurons", type=int, default=N_NEURONS)
    parser.add_argument(
        "--n-seeds", type=int, default=8, help="Sweep seeds; 8 is the whole sweep."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=detection_comparison_dir() / "memoryless_reference_check.parquet",
    )
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args(argv)

    config = CONFIG
    decoded = simulate_and_decode(args.n_neurons, config)
    encoding_model = decoded["detector"].encoding_model_[ENCODING_KEY]
    summaries, table = compare_variants(decoded, encoding_model, config, FIGURE_SEED)
    figure_one = pd.DataFrame(summaries).rename_axis("measure")
    swept = sweep_totals(args.n_seeds)

    summary = pd.concat(
        [figure_one.assign(input="figure_one"), swept.assign(input="sweep")]
    ).reset_index()[["input", "measure", *VARIANTS]]
    provenance = {
        "produced_by": produced_by(__file__),
        "git_commit": described_commit(),
        "n_sweep_seeds": str(args.n_seeds),
    }
    for measure, note in provenance.items():
        summary.loc[len(summary)] = {
            "input": "provenance",
            "measure": measure,
        } | dict.fromkeys(VARIANTS, note)

    write_table(table, args.out)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.to_string())
        print()
        print(table.round(3).to_string())
    print(f"\n  Figure 1 per candidate -> {args.out}\n  summary -> {args.summary}")


if __name__ == "__main__":
    main()
