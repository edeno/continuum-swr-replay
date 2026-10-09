"""Content occupancy of non-local events — where the decoded activity represents.

Characterizes *where* non-local (replay-like) events decode to, relative to the
animal and to the reward wells, separately for SWR-overlapping and non-SWR
events. **Both distances are track-graph distances** (per the project's
track-not-Euclidean convention).

Two views:

- **Arm occupancy** (from the canonical ``replay_content`` event tables):
  posterior mass per arm, SWR vs non-SWR — non-SWR content is well-dominated
  (leaf arms), SWR spreads onto the trunk.
- **Distance occupancy** (from ``data_over_time`` + the fitted environment, per
  session): for each non-local event timebin, the track distance from the
  decoded content to the animal (``ahead_behind_distance``) and to the nearest
  reward well (``Environment.get_distances_to_interior_bins``). Events are
  classified into four quadrants by two thresholds — far from the animal
  (> ``FAR_CM``) and at a well (< ``AT_WELL_CM``):

    remote reward well | own well (near animal) | remote path | near, off-well

The analysis reports the well/path fractions for the current canonical event
cohort. Previously rendered percentages must be regenerated after any event
definition change; they are intentionally not hard-coded here.

The analysis is deterministic (no sampling — all non-local timebins) and
re-decode-agnostic (reads whatever ``data_over_time``/``environment`` exist).

Usage
-----
    uv run python scripts/content/analyze_content_occupancy.py --all
    uv run python scripts/content/analyze_content_occupancy.py --animal chimi
    uv run python scripts/content/analyze_content_occupancy.py --epoch-identifier chimi20200213_02_r1_HPC
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

for _cat in (UserWarning, FutureWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=_cat)

from continuum_swr_replay.analysis.constants import (  # noqa: E402
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.analysis.hippocampus.data_aggregation import (  # noqa: E402
    validate_dot_event_contract,
)
from continuum_swr_replay.analysis.hippocampus.event_detection import (  # noqa: E402
    event_strict_overlap_mask,
)
from continuum_swr_replay.analysis.hippocampus.event_immobility import (  # noqa: E402
    EventDefinitionError,
    filter_event_labels_by_speed,
    validate_nonlocal_event_probability_contract,
)
from continuum_swr_replay.analysis.hippocampus.swr_source import (  # noqa: E402
    read_dot_with_swr_source,
    require_swr_source,
)
from continuum_swr_replay.analysis.replay.covariate_registry import (  # noqa: E402
    require_covariate_schema,
)
from continuum_swr_replay.data_loaders.epoch_register import (  # noqa: E402
    analysed_decode_files,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR  # noqa: E402
from continuum_swr_replay.table_io import read_table, table_exists  # noqa: E402

# --- Parameters ---------------------------------------------------------------
NON_LOCAL_THRESHOLD = DEFAULT_NON_LOCAL_THRESHOLD
IMMOBILITY_CM_PER_S = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
FAR_CM = 20.0  # content track-distance from animal above this = "far/remote"
AT_WELL_CM = 20.0  # content track-distance to nearest well below this = "at a well"
N_LEAF_ARMS = 6  # arms 0-5 terminate at reward wells; 6-8 are trunk
MIN_WELLS = 4  # require this many wells localizable in a session
MIN_IMMOBILE_BINS = 50  # per leaf arm, to estimate the well position

XE = np.arange(0, 401, 2)  # 2D histogram edges: content->animal (cm)
YE = np.arange(0, 201, 2)  # content->nearest well (cm)
DWE = np.arange(0, 201, 5)  # distance-to-well 1D histogram edges

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "paper" / "figures" / "content_occupancy"
GROUPS = ("nonSWR", "SWR")
GROUP_COLORS = {"nonSWR": "#0072B2", "SWR": "#009E73"}
QUADRANTS = ("remote_well", "own_well", "remote_path", "near_off")
QUADRANT_LABELS = ("remote\nreward well", "own well\n(near animal)", "remote\npath", "near,\noff-well")


# --- Session helpers ----------------------------------------------------------
def parse_epoch_identifier(epoch_id: str) -> tuple[str, str, str]:
    """Split ``{animal}{date}_{epoch}_{brain_area}`` into (nwb, epoch, area)."""
    parts = epoch_id.split("_")
    return f"{parts[0]}_.nwb", f"{parts[1]}_{parts[2]}", parts[3]


def animal_from_epoch_id(epoch_id: str) -> str:
    """``"chimi20200213_02_r1_HPC"`` -> ``"chimi"``."""
    return "".join(c for c in epoch_id.split("_")[0] if c.isalpha())


def find_sessions(animal: str | None) -> list[str]:
    """Analysed HPC epoch identifiers with data_over_time + environment present.

    The epoch register's analysed epochs (``analysed_decode_files``), never a
    glob, so a pending epoch's decode in the folder is not read.
    """
    thr_int = int(round(NON_LOCAL_THRESHOLD * 100))
    out = []
    for env in analysed_decode_files(PROCESSED_DATA_DIR, "_environment.pkl"):
        epoch_id = env.name.removesuffix("_environment.pkl")
        if animal is not None and not epoch_id.startswith(animal):
            continue
        if (PROCESSED_DATA_DIR / f"{epoch_id}_{thr_int}_data_over_time.pkl").exists():
            out.append(epoch_id)
    return out


def _well_positions(immobile: np.ndarray, segment: np.ndarray, linear: np.ndarray) -> list[float]:
    """Reward-well linear positions: the animal's median immobile position per leaf arm."""
    wells = []
    for arm in range(N_LEAF_ARMS):
        m = immobile & (segment == arm) & np.isfinite(linear)
        if m.sum() > MIN_IMMOBILE_BINS:
            wells.append(float(np.median(linear[m])))
    return wells


def session_content_distances(epoch_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Track distances for non-local event timebins in one session.

    Returns ``(d_animal, d_well, is_swr)`` over non-local immobile timebins, where
    ``d_animal`` is the track distance from the decoded content to the animal
    (``ahead_behind_distance``) and ``d_well`` is the track distance to the
    nearest reward well. Returns ``None`` if too few wells are localizable.
    """
    thr_int = int(round(NON_LOCAL_THRESHOLD * 100))
    dot = read_dot_with_swr_source(
        PROCESSED_DATA_DIR / f"{epoch_id}_{thr_int}_data_over_time.pkl"
    )
    # Validate the persisted cohort before any auxiliary load or early-return
    # path can turn corruption into an ordinary missing-data skip.
    validate_dot_event_contract(dot)
    validate_nonlocal_event_probability_contract(
        dot,
        probability_threshold=NON_LOCAL_THRESHOLD,
    )
    env = joblib.load(PROCESSED_DATA_DIR / f"{epoch_id}_environment.pkl")
    # The fit flag is not persisted, but the grid + track graph are intact.
    env._is_fitted = True
    interior = np.asarray(env.is_track_interior_).reshape(-1)
    pos = np.asarray(env.place_bin_centers_).reshape(-1)[interior]  # (n_interior,)

    immobile = dot["head_speed"].to_numpy() < IMMOBILITY_CM_PER_S
    segment = dot["track_segment_id"].to_numpy()
    linear = dot["linear_position"].to_numpy()
    well_lin = _well_positions(immobile, segment, linear)
    if len(well_lin) < MIN_WELLS:
        return None

    # Per-interior-bin track distance to the nearest reward well.
    dist_matrix = np.asarray(env.get_distances_to_interior_bins(pos[:, None]))  # (n_int, n_int)
    well_bins = [int(np.argmin(np.abs(pos - wl))) for wl in well_lin]
    nearest_well_dist = dist_matrix[:, well_bins].min(1)  # (n_interior,)

    # Retain validation at the business-logic boundary as defense in depth for
    # direct callers that may later refactor the entry path above.
    validate_nonlocal_event_probability_contract(
        dot,
        probability_threshold=NON_LOCAL_THRESHOLD,
    )
    eligible_dot = filter_event_labels_by_speed(
        dot, ["non_local_event", "SWR_event"]
    )
    event = eligible_dot["non_local_event"].to_numpy() > 0
    # Assign every bin of a Non-Local event to one group using the canonical
    # strict event-level overlap rule. A raw per-bin SWR mask would split a
    # partially overlapping event between both groups.
    is_swr = event_strict_overlap_mask(
        eligible_dot, "non_local_event", "SWR_event"
    ).to_numpy()
    map_estimate = dot["map_estimate"].to_numpy()
    d_animal = np.abs(dot["ahead_behind_distance"].to_numpy())
    content_bin = np.argmin(np.abs(pos[None, :] - map_estimate[:, None]), axis=1)
    d_well = nearest_well_dist[content_bin]

    # Guard against non-finite d_well (e.g. Inf from a disconnected graph component).
    valid = event & np.isfinite(map_estimate) & np.isfinite(d_animal) & np.isfinite(d_well)
    return d_animal[valid], d_well[valid], is_swr[valid]


def quadrant_fractions(d_animal: np.ndarray, d_well: np.ndarray) -> dict[str, float]:
    """Fraction of content *timebins* in each (far × at-well) quadrant.

    Timebin-weighted (where the decoded content spends its time), not
    event-weighted; longer events contribute more. Aggregation (`_wavg`) keeps
    this convention (weights per-session fractions by timebin count).
    """
    far = d_animal > FAR_CM
    at_well = d_well < AT_WELL_CM
    return {
        "remote_well": float((far & at_well).mean()),
        "own_well": float((~far & at_well).mean()),
        "remote_path": float((far & ~at_well).mean()),
        "near_off": float((~far & ~at_well).mean()),
    }


# --- Arm occupancy (from canonical event tables) ------------------------------
def arm_occupancy(animal: str | None = None) -> pd.DataFrame | None:
    """Per-arm posterior-mass occupancy, normalized within SWR group.

    Returns a (9-arm x {nonSWR, SWR}) fraction table, or None if the canonical
    ``replay_content/event_arm_table.pkl`` is unavailable.
    """
    f = PROCESSED_DATA_DIR / "replay_content" / "event_arm_table.pkl"
    if not table_exists(f):
        return None
    ea = read_table(f)
    require_covariate_schema(ea, table="event_arm")
    # Reported beside the live-source SWR split of session_content_distances.
    require_swr_source(ea, what=str(f))
    if animal is not None and "animal" in ea.columns:
        ea = ea[ea["animal"] == animal]
    ea = ea.assign(group=np.where(ea["is_swr"].astype(bool), "SWR", "nonSWR"))
    occ = ea.groupby(["group", "candidate_arm"])["arm_posterior_mass"].sum().unstack(0)
    return occ / occ.sum()


# --- Analysis -----------------------------------------------------------------
def analyze(sessions: list[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Process sessions; return per-session quadrant rows + accumulated 2D and 1D histograms."""
    hist2d = {k: np.zeros((len(XE) - 1, len(YE) - 1)) for k in GROUPS}
    dwell_hist = {k: np.zeros(len(DWE) - 1) for k in GROUPS}
    rows = []
    n_dropped = 0
    for epoch_id in sessions:
        try:
            result = session_content_distances(epoch_id)
        except EventDefinitionError:
            # Contract failures invalidate the requested cross-session
            # cohort; do not publish a plausible-looking partial aggregate.
            raise
        except Exception as exc:  # noqa: BLE001 - log and continue the sweep
            print(f"  {epoch_id}: FAILED ({type(exc).__name__}: {exc})")
            continue
        if result is None:
            # Fewer than MIN_WELLS leaf arms localizable; well distances would be
            # relative to an incomplete well set, so the session is dropped.
            n_dropped += 1
            continue
        d_animal, d_well, is_swr = result
        for group, mask in (("nonSWR", ~is_swr), ("SWR", is_swr)):
            da, dw = d_animal[mask], d_well[mask]
            if len(da) == 0:
                continue
            hist2d[group] += np.histogram2d(da, dw, bins=[XE, YE])[0]
            dwell_hist[group] += np.histogram(dw, bins=DWE)[0]
            row = {"epoch_id": epoch_id, "animal": animal_from_epoch_id(epoch_id), "group": group, "n": len(da)}
            row.update(quadrant_fractions(da, dw))
            rows.append(row)
    if n_dropped:
        print(f"  dropped {n_dropped} session(s): <{MIN_WELLS} reward wells localizable")
    return pd.DataFrame(rows), hist2d, dwell_hist


def _wavg(g: pd.DataFrame, col: str) -> float:
    """Timebin-count-weighted mean of a per-session fraction (NaN if no timebins)."""
    total = g["n"].sum()
    return float((g[col] * g["n"]).sum() / total) if total else float("nan")


# --- Figures ------------------------------------------------------------------
def render(metrics: pd.DataFrame, hist2d: dict, dwell_hist: dict, output_dir: Path) -> None:
    """Quadrant bar, 2D animal-vs-well, distance-to-well, and arm occupancy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. quadrant histogram (the headline)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(QUADRANTS))
    width = 0.38
    for i, group in enumerate(GROUPS):
        g = metrics[metrics.group == group]
        ax.bar(x + (i - 0.5) * width, [_wavg(g, q) for q in QUADRANTS], width, label=group, color=GROUP_COLORS[group])
    for j, q in enumerate(QUADRANTS):  # per-animal non-SWR dots
        pa = metrics[metrics.group == "nonSWR"].groupby("animal").apply(lambda gg, q=q: _wavg(gg, q))
        ax.scatter([x[j] - 0.5 * width] * len(pa), pa.to_numpy(), color="k", s=14, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(QUADRANT_LABELS)
    ax.set_ylabel("fraction of content mass")
    ax.legend()
    ax.set_title(f"Content quadrants — track distance (far>{FAR_CM:.0f} cm, at-well<{AT_WELL_CM:.0f} cm)")
    fig.tight_layout()
    fig.savefig(output_dir / "content_quadrants.png", dpi=140)
    plt.close(fig)

    # 2. 2D animal-vs-well (log counts)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True)
    for ax, group, cmap in zip(axs, GROUPS, ("Blues", "Greens"), strict=True):
        ax.pcolormesh(XE, YE, np.log1p(hist2d[group]).T, cmap=cmap)
        ax.axhline(AT_WELL_CM, color="r", ls="--", lw=1)
        ax.axvline(FAR_CM, color="k", ls=":", lw=1)
        ax.set_xlabel("content → animal (cm)")
        ax.set_title(f"{group} content")
    axs[0].set_ylabel("content → nearest reward well (cm)")
    fig.suptitle("Non-local content: distance from animal vs to reward well (track distance, log counts)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "content_animal_vs_well.png", dpi=140)
    plt.close(fig)

    # 3. distance-to-well 1D
    fig, ax = plt.subplots(figsize=(7, 4.5))
    centers = (DWE[:-1] + DWE[1:]) / 2
    for group in GROUPS:
        h = dwell_hist[group]
        ax.plot(centers, h / h.sum(), "-o", ms=3, color=GROUP_COLORS[group], label=group)
    ax.axvline(AT_WELL_CM, color="r", ls="--", lw=1)
    ax.set_xlabel("track distance to nearest reward well (cm)")
    ax.set_ylabel("density")
    ax.set_title("Content distance to nearest reward well")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "content_distance_to_well.png", dpi=140)
    plt.close(fig)

    # 4. arm occupancy (from event tables, if present)
    occ = arm_occupancy()
    if occ is not None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        arms = np.arange(9)
        for i, group in enumerate(GROUPS):
            ax.bar(arms + (i - 0.5) * width, occ[group].reindex(range(9)).fillna(0), width, label=group, color=GROUP_COLORS[group])
        for xx in range(N_LEAF_ARMS):
            ax.axvspan(xx - 0.5, xx + 0.5, color="gold", alpha=0.05)
        ax.set_xticks(arms)
        ax.set_xlabel("arm (0-5 reward wells, 6-8 trunk)")
        ax.set_ylabel("fraction of content mass")
        ax.set_title("Content occupancy by arm")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "content_arm_occupancy.png", dpi=140)
        plt.close(fig)


# --- CLI ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--epoch-identifier", help="single session, e.g. chimi20200213_02_r1_HPC")
    g.add_argument("--animal", help="all HPC sessions for this animal")
    g.add_argument("--all", action="store_true", help="every HPC session")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.epoch_identifier:
        sessions = [args.epoch_identifier]
    elif args.all:
        sessions = find_sessions(None)
    else:
        sessions = find_sessions(args.animal)
    print(f"Analyzing {len(sessions)} session(s)")

    metrics, hist2d, dwell_hist = analyze(sessions)
    if metrics.empty:
        print("No sessions analyzed.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "content_occupancy_per_session.csv", index=False)
    np.savez(args.output_dir / "content_occupancy_2dhist.npz", xe=XE, ye=YE,
             nonSWR=hist2d["nonSWR"], SWR=hist2d["SWR"],
             dwe=DWE, dwell_nonSWR=dwell_hist["nonSWR"], dwell_SWR=dwell_hist["SWR"])
    render(metrics, hist2d, dwell_hist, args.output_dir)

    print(f"\nWrote per-session metrics ({len(metrics)} rows) + 2D histogram + figures to {args.output_dir}")
    print("\nQuadrant fractions (n-weighted):")
    for group in GROUPS:
        g = metrics[metrics.group == group]
        print(f"  {group}: " + ", ".join(f"{q}={_wavg(g, q):.2f}" for q in QUADRANTS))
    print("\nPer-animal non-SWR (within-animal):")
    pa = metrics[metrics.group == "nonSWR"].groupby("animal").apply(
        lambda gg: pd.Series({q: _wavg(gg, q) for q in QUADRANTS})
    )
    print(pa.round(2).to_string())


if __name__ == "__main__":
    main()
