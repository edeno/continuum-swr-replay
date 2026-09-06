"""Check data availability for all sessions in the spatial bandit task.

Systematically attempts to load each data component (position, spikes, LFP, events)
independently for every session/epoch, reporting which succeed and which fail.

Usage:
    uv run python scripts/qa/check_data_availability.py
    uv run python scripts/qa/check_data_availability.py --animal chimi
    uv run python scripts/qa/check_data_availability.py --output results.csv
"""

from __future__ import annotations

import argparse
import logging
import traceback
import warnings
from pathlib import Path

import pandas as pd
from spyglass.common import PositionIntervalMap

warnings.filterwarnings("ignore")
logging.getLogger("spyglass").setLevel(logging.CRITICAL)
logging.getLogger("datajoint").setLevel(logging.CRITICAL)
logging.getLogger("pynwb").setLevel(logging.CRITICAL)
logging.getLogger("hdmf").setLevel(logging.CRITICAL)

# Project imports are intentionally placed after warning/logger silencing
# so that import-time chatter from spyglass/datajoint/pynwb is suppressed.
from continuum_swr_replay.data_loaders.events import load_dios  # noqa: E402
from continuum_swr_replay.data_loaders.lfp import get_ripple, get_theta  # noqa: E402
from continuum_swr_replay.data_loaders.position import get_position_info  # noqa: E402
from continuum_swr_replay.data_loaders.spikes import get_spike_data  # noqa: E402
from continuum_swr_replay.data_loaders.utils import get_epoch_info  # noqa: E402


def check_position(nwb_file_name: str, epoch_name: str) -> dict:
    """Check if position data can be loaded."""
    try:
        position_interval_name = (
            PositionIntervalMap
            & {"nwb_file_name": nwb_file_name, "interval_list_name": epoch_name}
        ).fetch1("position_interval_name")
        position_data = get_position_info(nwb_file_name, epoch_name, position_interval_name)
        pos_info = position_data["position_info"].dropna(
            subset=["head_position_x", "head_position_y"]
        )
        n_timepoints = len(pos_info)
        return {
            "position_ok": True,
            "position_error": "",
            "n_timepoints": n_timepoints,
            "position_data": position_data,
            "position_info": pos_info,
        }
    except Exception as e:
        return {
            "position_ok": False,
            "position_error": _short_error(e),
            "n_timepoints": 0,
            "position_data": None,
            "position_info": None,
        }


def check_spikes(nwb_file_name: str) -> dict:
    """Check if spike data can be loaded."""
    try:
        spike_data = get_spike_data(nwb_file_name)
        brain_areas = sorted(spike_data["spike_times"].keys())
        n_units = {
            area: len(spike_data["spike_times"][area]) for area in brain_areas
        }
        return {
            "spikes_ok": True,
            "spikes_error": "",
            "brain_areas": ",".join(brain_areas) if brain_areas else "",
            "n_units_hpc": n_units.get("HPC", 0),
            "n_units_mpfc": n_units.get("mPFC", 0),
            "n_units_ofc": n_units.get("OFC", 0),
        }
    except Exception as e:
        return {
            "spikes_ok": False,
            "spikes_error": _short_error(e),
            "brain_areas": "",
            "n_units_hpc": 0,
            "n_units_mpfc": 0,
            "n_units_ofc": 0,
        }


def check_ripple(nwb_file_name: str, epoch_name: str) -> dict:
    """Check if ripple LFP data can be loaded."""
    try:
        ripple_data = get_ripple(nwb_file_name, epoch_name, "Kay")
        n_ripples = len(ripple_data["ripple_times"])
        return {
            "ripple_ok": True,
            "ripple_error": "",
            "n_ripples": n_ripples,
        }
    except Exception as e:
        return {
            "ripple_ok": False,
            "ripple_error": _short_error(e),
            "n_ripples": 0,
        }


def check_theta(nwb_file_name: str, epoch_name: str) -> dict:
    """Check if theta LFP data can be loaded."""
    try:
        get_theta(nwb_file_name, epoch_name)
        return {"theta_ok": True, "theta_error": ""}
    except Exception as e:
        return {"theta_ok": False, "theta_error": _short_error(e)}


EPOCH_EDGE_WINDOW_SEC = 1.0
"""How close to an epoch boundary a poke has to be to count as edge-clipped."""


def _n_edge_pokes(beam_breaks: pd.DataFrame, valid_times: tuple[float, float]) -> int:
    """Count beam-break bouts touching the first or last second of the epoch.

    A poke already in progress when the epoch starts (or still in progress
    when it ends) is clipped by the interval, so its poke-in is the epoch
    boundary rather than a real arrival. Those visits get an arrival time that
    is an artifact of where the recording was cut, which propagates into every
    arrival-anchored clock. This is a report, not a gate: it says how many
    visits per epoch are exposed.
    """
    if beam_breaks.empty:
        return 0
    start, end = float(valid_times[0]), float(valid_times[1])
    poke_in = beam_breaks["start_time"].to_numpy(dtype=float)
    poke_out = beam_breaks["end_time"].to_numpy(dtype=float)
    at_start = poke_in <= start + EPOCH_EDGE_WINDOW_SEC
    at_end = poke_out >= end - EPOCH_EDGE_WINDOW_SEC
    return int((at_start | at_end).sum())


def check_dios(nwb_file_name: str, valid_times: tuple[float, float]) -> dict:
    """Check if DIO events can be loaded."""
    try:
        dio_data = load_dios(nwb_file_name, valid_times)
        n_beam_breaks = len(dio_data["beam_breaks"])
        n_pump_events = len(dio_data["pump_events"])
        return {
            "dios_ok": True,
            "dios_error": "",
            "n_beam_breaks": n_beam_breaks,
            "n_pump_events": n_pump_events,
            "n_edge_pokes": _n_edge_pokes(dio_data["beam_breaks"], valid_times),
        }
    except Exception as e:
        return {
            "dios_ok": False,
            "dios_error": _short_error(e),
            "n_beam_breaks": 0,
            "n_pump_events": 0,
            "n_edge_pokes": 0,
        }


def _short_error(e: Exception) -> str:
    """Get a concise error description."""
    error_type = type(e).__name__
    msg = str(e)
    # Truncate long messages
    if len(msg) > 200:
        msg = msg[:200] + "..."
    return f"{error_type}: {msg}"


def check_session(nwb_file_name: str, epoch_name: str, animal: str, date: int) -> dict:
    """Check all data components for a single session/epoch."""
    result = {
        "nwb_file_name": nwb_file_name,
        "epoch_name": epoch_name,
        "animal": animal,
        "date": date,
    }

    # Position (must succeed for other checks that need time bounds)
    pos_result = check_position(nwb_file_name, epoch_name)
    result.update({k: v for k, v in pos_result.items() if k not in ("position_data", "position_info")})

    # Spikes (independent of position)
    spike_result = check_spikes(nwb_file_name)
    result.update(spike_result)

    # Ripple LFP (independent of position)
    ripple_result = check_ripple(nwb_file_name, epoch_name)
    result.update(ripple_result)

    # Theta LFP (independent of position)
    theta_result = check_theta(nwb_file_name, epoch_name)
    result.update(theta_result)

    # DIOs (need position time bounds)
    if pos_result["position_info"] is not None:
        valid_times = pos_result["position_info"].index.to_numpy()[[0, -1]]
        dio_result = check_dios(nwb_file_name, valid_times)
    else:
        dio_result = {
            "dios_ok": False,
            "dios_error": "Skipped: position failed",
            "n_beam_breaks": 0,
            "n_pump_events": 0,
            "n_edge_pokes": 0,
        }
    result.update(dio_result)

    # Summary
    result["all_ok"] = all([
        result["position_ok"],
        result["spikes_ok"],
        result["ripple_ok"],
        result["theta_ok"],
        result["dios_ok"],
    ])

    return result


def main():
    parser = argparse.ArgumentParser(description="Check data availability for all sessions")
    parser.add_argument("--animal", type=str, default=None, help="Filter to specific animal")
    parser.add_argument("--output", type=str, default=None, help="Save results to CSV")
    args = parser.parse_args()

    print("Loading epoch info...")
    epoch_info = get_epoch_info()

    if args.animal:
        epoch_info = epoch_info[epoch_info["animal"] == args.animal]

    n_epochs = len(epoch_info)
    print(f"Checking {n_epochs} epochs...\n")

    results = []
    for i, ((nwb_file_name, epoch_name), row) in enumerate(epoch_info.iterrows()):
        animal = row["animal"]
        date = row["date"]
        label = f"{animal}{date}_{epoch_name}"
        print(f"[{i + 1}/{n_epochs}] {label}...", end=" ", flush=True)

        try:
            result = check_session(nwb_file_name, epoch_name, animal, date)
        except Exception as e:
            print(f"FATAL: {e}")
            traceback.print_exc()
            result = {
                "nwb_file_name": nwb_file_name,
                "epoch_name": epoch_name,
                "animal": animal,
                "date": date,
                "position_ok": False,
                "position_error": f"FATAL: {_short_error(e)}",
                "spikes_ok": False,
                "spikes_error": "Skipped",
                "ripple_ok": False,
                "ripple_error": "Skipped",
                "theta_ok": False,
                "theta_error": "Skipped",
                "dios_ok": False,
                "dios_error": "Skipped",
                "all_ok": False,
            }

        results.append(result)

        # Print status line
        checks = []
        for component in ["position", "spikes", "ripple", "theta", "dios"]:
            ok = result.get(f"{component}_ok", False)
            checks.append(f"{component}:{'OK' if ok else 'FAIL'}")
        # Pokes clipped by the epoch boundary carry an artifactual arrival time.
        checks.append(f"edge_pokes:{int(result.get('n_edge_pokes', 0))}")
        print(" | ".join(checks))

    # Build results DataFrame
    df = pd.DataFrame(results)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nTotal epochs: {len(df)}")
    print(f"All OK: {df['all_ok'].sum()} ({100 * df['all_ok'].mean():.0f}%)")

    for component in ["position", "spikes", "ripple", "theta", "dios"]:
        col = f"{component}_ok"
        if col in df.columns:
            n_ok = df[col].sum()
            print(f"  {component:10s}: {n_ok}/{len(df)} OK ({100 * n_ok / len(df):.0f}%)")

    # Per-animal breakdown
    print("\nPer-animal breakdown:")
    for animal in sorted(df["animal"].unique()):
        adf = df[df["animal"] == animal]
        n = len(adf)
        n_ok = adf["all_ok"].sum()
        components = []
        for c in ["position", "spikes", "ripple", "theta", "dios"]:
            col = f"{c}_ok"
            if col in adf.columns:
                c_ok = adf[col].sum()
                if c_ok < n:
                    components.append(f"{c}:{c_ok}/{n}")
        status = f"  {animal:10s}: {n_ok}/{n} all OK"
        if components:
            status += f"  [{', '.join(components)}]"
        print(status)

    # List failures grouped by error
    print("\nFailure details:")
    for component in ["position", "spikes", "ripple", "theta", "dios"]:
        ok_col = f"{component}_ok"
        err_col = f"{component}_error"
        if ok_col not in df.columns or err_col not in df.columns:
            continue
        failures = df[~df[ok_col]]
        if failures.empty:
            continue
        print(f"\n  {component.upper()} failures ({len(failures)}):")
        for error, group in failures.groupby(err_col):
            sessions = [
                f"{r['animal']}{r['date']}_{r['epoch_name']}"
                for _, r in group.iterrows()
            ]
            print(f"    [{len(sessions)}] {error}")
            if len(sessions) <= 10:
                for s in sessions:
                    print(f"         {s}")
            else:
                for s in sessions[:5]:
                    print(f"         {s}")
                print(f"         ... and {len(sessions) - 5} more")

    # Save to CSV
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
    else:
        default_path = Path(__file__).with_name("data_availability.csv")
        df.to_csv(default_path, index=False)
        print(f"\nResults saved to {default_path}")


if __name__ == "__main__":
    main()
