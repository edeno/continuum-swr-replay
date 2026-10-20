"""Augment existing ``*_data_over_time.pkl`` with step_distance + run_speed_baseline.

The decode results are locked to their original time axis, and a fresh
``load_data`` no longer reproduces it: ``get_position_info`` trims post-trial
padding (commit e5a0def) that post-dates the decodes, so reloaded position/LFP
ends earlier than the decode axis. We therefore append only the two new columns
that need no contemporaneous reload:

- ``step_distance``      from the saved ``results.nc`` posterior (decoder MAP
                         via ``get_map_step_distance``), exact.
- ``run_speed_baseline`` p90 of the pickle's own ``head_speed``.

The neural-signal rollups use the per-session z-scored columns already stored
in each pkl (``SWR_zscore`` / ``MUA_zscore`` / ``theta_power_zscore``), which
cover the full decode axis — so no signal reload is needed and the trimming
issue is sidestepped entirely.

Flags
-----
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
from non_local_detector.environment import Environment

from continuum_swr_replay.analysis.contfrag.posterior_analysis import (
    get_contfrag_posterior,
)
from continuum_swr_replay.analysis.hippocampus.posterior_analysis import (
    get_conditional_non_local_posterior,
    get_map_step_distance,
)
from continuum_swr_replay.decoding.results import open_results
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

NEW_COLUMNS = ("step_distance", "run_speed_baseline")

_NONLOCAL_RE = re.compile(
    r"^(?P<animal>\w+?\d{8})_(?P<epoch>\d+_r\d+)_(?P<area>\w+?)_(?P<thr>\d+)_data_over_time\.pkl$"
)
_CONTFRAG_RE = re.compile(
    r"^(?P<animal>\w+?\d{8})_(?P<epoch>\d+_r\d+)_(?P<area>\w+?)_contfrag_data_over_time\.pkl$"
)


def _step_distance_from_results(
    scenario: str, base: str, env: Environment, n_rows: int, pkl_name: str
) -> np.ndarray:
    """Exact per-timebin step distance from the saved posterior (argmax MAP)."""
    suffix = "" if scenario.startswith("nonlocal") else "_contfrag"
    with open_results(PROCESSED_DATA_DIR / f"{base}{suffix}_results.nc") as results:
        if scenario.startswith("nonlocal"):
            posterior = get_conditional_non_local_posterior(results)
        else:
            posterior = get_contfrag_posterior(results, state="combined")
        step_distance = get_map_step_distance(
            posterior.values,
            env.track_graph_with_bin_centers_edges_,
            np.asarray(env.place_bin_centers_nodes_df_.node_id),
        )
    if len(step_distance) != n_rows:
        raise ValueError(
            f"{pkl_name}: results.nc time ({len(step_distance)}) != pkl rows "
            f"({n_rows}); axes are not aligned"
        )
    return step_distance


# Raw-signal columns written by an earlier augmentation revision; the rollups
# now use the stored z-scores instead, so drop these stale columns on rewrite.
_STALE_COLUMNS = ["ripple_power", "mua_rate", "theta_power"]


def _augment_pkl(pkl: Path, scenario: str, base: str, env: Environment, check_only: bool) -> None:
    dot = pd.read_pickle(pkl)
    dot = dot.drop(columns=_STALE_COLUMNS, errors="ignore")

    step_distance = _step_distance_from_results(scenario, base, env, len(dot), pkl.name)

    run_speed_baseline = float(np.nanpercentile(dot["head_speed"].to_numpy(), 90))
    if not np.isfinite(run_speed_baseline):
        raise ValueError(f"{pkl.name}: run_speed_baseline is not finite (head_speed all-NaN?)")

    dot["step_distance"] = step_distance
    dot["run_speed_baseline"] = run_speed_baseline

    out = pkl.with_suffix(".pkl.new") if check_only else pkl
    # Atomic write: a kill mid-write must not corrupt the (unrecreatable)
    # original. Write a temp on the same filesystem, then rename.
    tmp = out.with_suffix(out.suffix + ".tmp")
    dot.to_pickle(tmp)
    os.replace(tmp, out)
    print(f"[{pkl.name}] wrote {out.name} ({len(dot.columns)} columns)")


def _augment_epoch(
    animal: str, epoch_name: str, brain_area: str,
    scenarios: dict[str, Path], check_only: bool, force: bool,
) -> None:
    def _needs_augment(p: Path) -> bool:
        if force:
            return True
        try:
            cols = set(pd.read_pickle(p).columns)
        except Exception as e:  # noqa: BLE001 - identify the unreadable file
            raise RuntimeError(f"cannot read {p.name}: {e!r}") from e
        # Re-process if the new columns are missing OR a stale raw-signal column
        # from a prior revision is still present (to drop it).
        missing_new = not set(NEW_COLUMNS).issubset(cols)
        has_stale = bool(set(_STALE_COLUMNS) & cols)
        return missing_new or has_stale

    pending = {s: p for s, p in scenarios.items() if _needs_augment(p)}
    if not pending:
        print(f"[{animal} {epoch_name} {brain_area}] already augmented; skip")
        return

    base = f"{animal}_{epoch_name}_{brain_area}"
    t0 = time.perf_counter()
    for scenario, pkl in pending.items():
        env_suffix = "" if scenario.startswith("nonlocal") else "_contfrag"
        env = Environment.load_environment(
            str(PROCESSED_DATA_DIR / f"{base}{env_suffix}_environment.pkl")
        )
        _augment_pkl(pkl, scenario, base, env, check_only)
    print(
        f"[{animal} {epoch_name} {brain_area}] done in "
        f"{time.perf_counter() - t0:.1f}s; scenarios={list(pending)}"
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
                args.check_only, args.force,
            )
        except MemoryError:
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
