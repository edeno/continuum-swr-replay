"""Augment existing ``*_data_over_time.pkl`` with the new trajectory signals.

The saved decode results are locked to their original time axis, and a fresh
``load_data`` no longer returns an exactly length-matched ``position_info``
(it has drifted by a few samples), so rebuilding the table from scratch is not
faithful. Instead we *append* the new per-timebin columns to the existing
pickle, which is already on the correct axis and already carries ``head_speed``
and ``map_estimate``:

- ``step_distance``      recomputed from the stored ``map_estimate`` via the
                         track-graph node mapping and the canonical
                         ``step_distance_from_nodes`` (no multi-GB ``.nc`` load).
- ``run_speed_baseline`` p90 of the pickle's own ``head_speed`` (no load_data).
- ``ripple_power`` /     raw LFP/MUA signals from ``load_data``, nearest-
  ``mua_rate`` /         reindexed (4 ms tolerance) onto the pickle's index —
  ``theta_power``        the same alignment the stored z-scored columns use.

``load_data`` dominates per-epoch cost, so both scenarios of an epoch share a
single ``load_data`` call.

Flags
-----
--validate-step   additionally load the saved ``results.nc`` for each pkl,
                  recompute ``step_distance`` from the full posterior via the
                  canonical ``get_map_step_distance``, and assert it is
                  bit-identical to the ``map_estimate``-derived value. Use on a
                  smoke sample to confirm the cheap path before the full run.
--check-only      write a ``.pkl.new`` sidecar instead of overwriting in place.
--force           recompute even if the new columns are already present.
"""

from __future__ import annotations

import argparse
import os
import re
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from non_local_detector.environment import Environment

from continuum_swr_replay.analysis.contfrag.posterior_analysis import (
    get_contfrag_posterior,
)
from continuum_swr_replay.analysis.hippocampus.posterior_analysis import (
    get_conditional_non_local_posterior,
    get_map_step_distance,
    map_estimate_to_map_nodes,
    step_distance_from_nodes,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

# Raw signals are nearest-reindexed onto the saved decode axis; a handful of
# unmatched edge bins is acceptable, but a large fraction signals a real axis
# misalignment (e.g. a uniform offset or unit mismatch) that must fail loud.
_MAX_REINDEX_NAN_FRAC = 0.005

NEW_COLUMNS = (
    "step_distance",
    "ripple_power",
    "mua_rate",
    "theta_power",
    "run_speed_baseline",
)

_NONLOCAL_RE = re.compile(
    r"^(?P<animal>\w+?\d{8})_(?P<epoch>\d+_r\d+)_(?P<area>\w+?)_(?P<thr>\d+)_data_over_time\.pkl$"
)
_CONTFRAG_RE = re.compile(
    r"^(?P<animal>\w+?\d{8})_(?P<epoch>\d+_r\d+)_(?P<area>\w+?)_contfrag_data_over_time\.pkl$"
)


def _reindex(series: pd.Series, index: pd.Index, label: str) -> np.ndarray:
    """Nearest-reindex a raw signal onto the decode axis, asserting coverage.

    A small number of unmatched edge bins is tolerated; a large unmatched
    fraction means the signal's time axis does not line up with the saved
    decode axis (a uniform offset, a unit mismatch, or a coverage gap), which
    would otherwise be persisted as a silently-NaN column.
    """
    out = series.reindex(index, method="nearest", tolerance=0.004)
    nan_frac = float(out.isna().mean())
    if nan_frac > _MAX_REINDEX_NAN_FRAC:
        median_gap_ms = float(np.median(np.diff(series.index))) * 1e3
        raise ValueError(
            f"{label}: {nan_frac:.1%} of timebins had no sample within 4 ms "
            f"(median source gap {median_gap_ms:.2f} ms) — axis misalignment; "
            "refusing to write a silently-NaN column"
        )
    return out.to_numpy()


def _augment_pkl(
    pkl: Path, scenario: str, base: str, env: Environment, data: dict,
    brain_area: str, validate_step: bool, check_only: bool,
) -> None:
    dot = pd.read_pickle(pkl)

    node_ids, has_valid = map_estimate_to_map_nodes(
        dot["map_estimate"].to_numpy(), env.place_bin_centers_nodes_df_
    )
    step_distance = step_distance_from_nodes(
        node_ids, has_valid, env.track_graph_with_bin_centers_edges_
    )

    if validate_step:
        suffix = "" if scenario.startswith("nonlocal") else "_contfrag"
        with xr.open_dataset(
            PROCESSED_DATA_DIR / f"{base}{suffix}_results.nc", engine="h5netcdf"
        ) as results:
            results = results.set_index(state_bins=list(results["state_bins"].coords))
            if scenario.startswith("nonlocal"):
                posterior = get_conditional_non_local_posterior(results)
            else:
                posterior = get_contfrag_posterior(results, state="combined")
            canonical = get_map_step_distance(
                posterior.values,
                env.track_graph_with_bin_centers_edges_,
                np.asarray(env.place_bin_centers_nodes_df_.node_id),
            )
        np.testing.assert_array_equal(
            np.nan_to_num(step_distance, nan=-1.0),
            np.nan_to_num(canonical, nan=-1.0),
            err_msg=f"{pkl.name}: map_estimate step_distance != canonical",
        )
        print(f"[{pkl.name}] validate-step OK (matches canonical)")

    run_speed_baseline = float(np.nanpercentile(dot["head_speed"].to_numpy(), 90))
    if not np.isfinite(run_speed_baseline):
        raise ValueError(
            f"{pkl.name}: run_speed_baseline is not finite (head_speed all-NaN?)"
        )

    def _onto_decode_axis(series: pd.Series, column: str) -> np.ndarray:
        return _reindex(series, dot.index, f"{pkl.name}:{column}")

    dot["step_distance"] = step_distance
    dot["run_speed_baseline"] = run_speed_baseline
    dot["ripple_power"] = _onto_decode_axis(
        data["ripple_consensus_trace"], "ripple_power"
    )
    dot["mua_rate"] = _onto_decode_axis(
        data["multiunit_firing_rate"][brain_area].squeeze(), "mua_rate"
    )
    dot["theta_power"] = _onto_decode_axis(data["theta_power"].squeeze(), "theta_power")

    out = pkl.with_suffix(".pkl.new") if check_only else pkl
    # Atomic write: a kill mid-write must not corrupt the (unrecreatable)
    # original. Write a temp on the same filesystem, then rename.
    tmp = out.with_suffix(out.suffix + ".tmp")
    dot.to_pickle(tmp)
    os.replace(tmp, out)
    print(f"[{pkl.name}] wrote {out.name} ({len(dot.columns)} columns)")


def _augment_epoch(
    animal: str, epoch_name: str, brain_area: str,
    scenarios: dict[str, Path], validate_step: bool, check_only: bool, force: bool,
) -> None:
    def _needs_augment(p: Path) -> bool:
        if force:
            return True
        try:
            return not set(NEW_COLUMNS).issubset(pd.read_pickle(p).columns)
        except Exception as e:  # noqa: BLE001 - identify the unreadable file
            raise RuntimeError(f"cannot read {p.name}: {e!r}") from e

    pending = {s: p for s, p in scenarios.items() if _needs_augment(p)}
    if not pending:
        print(f"[{animal} {epoch_name} {brain_area}] already augmented; skip")
        return

    nwb_file_name = f"{animal}_.nwb"
    base = f"{animal}_{epoch_name}_{brain_area}"
    t0 = time.perf_counter()
    data = load_data(nwb_file_name, epoch_name)
    print(
        f"[{animal} {epoch_name} {brain_area}] load_data="
        f"{time.perf_counter() - t0:.1f}s; scenarios={list(pending)}"
    )

    for scenario, pkl in pending.items():
        env_suffix = "" if scenario.startswith("nonlocal") else "_contfrag"
        env = Environment.load_environment(
            str(PROCESSED_DATA_DIR / f"{base}{env_suffix}_environment.pkl")
        )
        _augment_pkl(
            pkl, scenario, base, env, data, brain_area, validate_step, check_only
        )


def _discover(files: list[Path]) -> dict[tuple[str, str, str], dict[str, Path]]:
    epochs: dict[tuple[str, str, str], dict[str, Path]] = defaultdict(dict)
    for pkl in files:
        m_nl = _NONLOCAL_RE.match(pkl.name)
        m_cf = _CONTFRAG_RE.match(pkl.name)
        if m_nl:
            key = (m_nl["animal"], m_nl["epoch"], m_nl["area"])
            epochs[key][f"nonlocal:{m_nl['thr']}"] = pkl
        elif m_cf:
            key = (m_cf["animal"], m_cf["epoch"], m_cf["area"])
            epochs[key]["contfrag"] = pkl
        else:
            print(f"[skip] unrecognized filename: {pkl.name}")
    return epochs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="*_data_over_time.pkl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validate-step", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    files = sorted(PROCESSED_DATA_DIR.glob(args.pattern))
    epochs = _discover(files)
    keys = sorted(epochs)
    if args.limit is not None:
        keys = keys[: args.limit]
    print(f"Augmenting {len(keys)} epoch(s) from {PROCESSED_DATA_DIR}")

    failures = []
    for animal, epoch_name, brain_area in keys:
        try:
            _augment_epoch(
                animal, epoch_name, brain_area,
                epochs[(animal, epoch_name, brain_area)],
                args.validate_step, args.check_only, args.force,
            )
        except MemoryError:
            # Don't keep launching more multi-GB loads after an OOM.
            raise
        except Exception as e:  # noqa: BLE001 - batch: record and continue
            print(f"[{animal} {epoch_name} {brain_area}] FAILED: {e!r}")
            traceback.print_exc()
            failures.append((animal, epoch_name, brain_area, repr(e)))

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}")
        raise SystemExit(1)
    print("\nAll done.")


if __name__ == "__main__":
    main()
