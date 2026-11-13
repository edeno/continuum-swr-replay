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
    uv run python scripts/behavior/content/analyze_content_occupancy.py --all
    uv run python scripts/behavior/content/analyze_content_occupancy.py --animal chimi
    uv run python scripts/behavior/content/analyze_content_occupancy.py --epoch-identifier chimi20200213_02_r1_HPC
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

for _cat in (UserWarning, FutureWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=_cat)

from continuum_swr_replay.analysis.constants import (  # noqa: E402
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.analysis.events.analysed_time import (  # noqa: E402
    read_dot_with_swr_source,
    require_swr_source,
)
from continuum_swr_replay.analysis.events.immobility import (  # noqa: E402
    EventDefinitionError,
)
from continuum_swr_replay.analysis.hippocampus import content_occupancy  # noqa: E402
from continuum_swr_replay.analysis.hippocampus.data_aggregation import (  # noqa: E402
    validate_dot_event_contract,
)
from continuum_swr_replay.analysis.hippocampus.event_statistics import (  # noqa: E402
    AT_WELL_CM,
    FAR_CM,
    OCCUPANCY_QUADRANTS,
    animal_occupancy_shares,
    arm_occupancy_shares,
    content_occupancy_shares,
    quadrant_fractions,
)
from continuum_swr_replay.data_loaders.constants import N_LEAF_ARMS  # noqa: E402
from continuum_swr_replay.data_loaders.epoch_register import (  # noqa: E402
    analysed_decode_files,
)
from continuum_swr_replay.decoding.files import (  # noqa: E402
    ENVIRONMENT_SUFFIX,
    decode_paths,
    epoch_identifier_of,
)
from continuum_swr_replay.decoding.identifiers import (  # noqa: E402
    parse_epoch_identifier,
)
from continuum_swr_replay.paths import (  # noqa: E402
    content_occupancy_dir,
    processed_data_dir,
    table_path,
)
from continuum_swr_replay.tables.covariate_registry import (  # noqa: E402
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table, table_exists  # noqa: E402

# --- Parameters ---------------------------------------------------------------

XE = np.arange(0, 401, 2)  # 2D histogram edges: content->animal (cm)
YE = np.arange(0, 201, 2)  # content->nearest well (cm)
DWE = np.arange(0, 201, 5)  # distance-to-well 1D histogram edges

GROUPS = ("nonSWR", "SWR")
GROUP_COLORS = {"nonSWR": "#0072B2", "SWR": "#009E73"}
QUADRANT_LABELS = (
    "remote\nreward well",
    "own well\n(near animal)",
    "remote\npath",
    "near,\noff-well",
)


# --- Session helpers ----------------------------------------------------------
def find_sessions(animal: str | None) -> list[str]:
    """Analysed HPC epoch identifiers with data_over_time + environment present.

    The epoch register's analysed epochs (``analysed_decode_files``), never a
    glob, so a pending epoch's decode in the folder is not read.
    """
    out = []
    for env in analysed_decode_files(processed_data_dir(), ENVIRONMENT_SUFFIX):
        epoch_id = epoch_identifier_of(env, ENVIRONMENT_SUFFIX)
        if animal is not None and parse_epoch_identifier(epoch_id).animal != animal:
            continue
        if decode_paths(epoch_id, processed_data_dir()).data_over_time.exists():
            out.append(epoch_id)
    return out


def session_content_distances(
    epoch_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Track distances for non-local event timebins in one session.

    Reads the session's decode frame and environment and returns
    :func:`content_occupancy.content_distances`: ``(d_animal, d_well, is_swr)``
    over the non-local event timebins, or ``None`` if fewer than
    ``content_occupancy.MIN_WELLS`` wells are localizable.
    """
    paths = decode_paths(epoch_id, processed_data_dir())
    # The read refuses labels that break the speed rule or the probability
    # contract.
    dot = read_dot_with_swr_source(
        paths.data_over_time, non_local_threshold=DEFAULT_NON_LOCAL_THRESHOLD
    )
    # Validate the persisted cohort before any auxiliary load or early-return
    # path can turn corruption into an ordinary missing-data skip.
    validate_dot_event_contract(dot)
    env = joblib.load(paths.environment)
    # The fit flag is not persisted, but the grid + track graph are intact.
    env._is_fitted = True
    return content_occupancy.content_distances(dot, env)


# --- Arm occupancy (from canonical event tables) ------------------------------
def arm_occupancy(animal: str | None = None) -> pd.DataFrame | None:
    """Per-arm posterior-mass occupancy, normalized within SWR group
    (:func:`arm_occupancy_shares`).

    Returns a (9-arm x {nonSWR, SWR}) fraction table, or None if the canonical
    ``replay_content/event_arm_table`` is unavailable.
    """
    f = table_path("event_arm_table")
    if not table_exists(f):
        return None
    ea = read_table(f)
    require_covariate_schema(ea, table="event_arm")
    # Reported beside the live-source SWR split of session_content_distances.
    require_swr_source(ea, what=str(f))
    if animal is not None and "animal" in ea.columns:
        ea = ea[ea["animal"] == animal]
    return arm_occupancy_shares(ea)


# --- Analysis -----------------------------------------------------------------
def analyze(
    sessions: list[str],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
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
            # Too few leaf arms localizable; well distances would be
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
            row = {
                "epoch_id": epoch_id,
                "animal": parse_epoch_identifier(epoch_id).animal,
                "group": group,
                "n": len(da),
            }
            row.update(quadrant_fractions(da, dw))
            rows.append(row)
    if n_dropped:
        print(
            f"  dropped {n_dropped} session(s): "
            f"<{content_occupancy.MIN_WELLS} reward wells localizable"
        )
    return pd.DataFrame(rows), hist2d, dwell_hist


def group_shares(metrics: pd.DataFrame) -> pd.DataFrame:
    """Each group's timebin-weighted share at each place
    (:func:`content_occupancy_shares`), a row of NaN for a group with no session.

    Returns
    -------
    shares : pd.DataFrame, shape (len(GROUPS), len(OCCUPANCY_PLACES))
    """
    return content_occupancy_shares(metrics).reindex(list(GROUPS))


# --- Figures ------------------------------------------------------------------
def render(
    metrics: pd.DataFrame, hist2d: dict, dwell_hist: dict, output_dir: Path
) -> None:
    """Quadrant bar, 2D animal-vs-well, distance-to-well, and arm occupancy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. quadrant histogram (the headline)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(OCCUPANCY_QUADRANTS))
    width = 0.38
    shares = group_shares(metrics)
    for i, group in enumerate(GROUPS):
        ax.bar(
            x + (i - 0.5) * width,
            shares.loc[group, list(OCCUPANCY_QUADRANTS)].tolist(),
            width,
            label=group,
            color=GROUP_COLORS[group],
        )
    pa = animal_occupancy_shares(metrics, "nonSWR")
    for j, q in enumerate(OCCUPANCY_QUADRANTS):  # per-animal non-SWR dots
        ax.scatter(
            [x[j] - 0.5 * width] * len(pa), pa[q].to_numpy(), color="k", s=14, zorder=3
        )
    ax.set_xticks(x)
    ax.set_xticklabels(QUADRANT_LABELS)
    ax.set_ylabel("fraction of content mass")
    ax.legend()
    ax.set_title(
        f"Content quadrants — track distance (far>{FAR_CM:.0f} cm, at-well<{AT_WELL_CM:.0f} cm)"
    )
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
    fig.suptitle(
        "Non-local content: distance from animal vs to reward well (track distance, log counts)"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "content_animal_vs_well.png", dpi=140)
    plt.close(fig)

    # 3. distance-to-well 1D
    fig, ax = plt.subplots(figsize=(7, 4.5))
    centers = (DWE[:-1] + DWE[1:]) / 2
    for group in GROUPS:
        h = dwell_hist[group]
        ax.plot(
            centers, h / h.sum(), "-o", ms=3, color=GROUP_COLORS[group], label=group
        )
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
            ax.bar(
                arms + (i - 0.5) * width,
                occ[group].reindex(range(9)).fillna(0),
                width,
                label=group,
                color=GROUP_COLORS[group],
            )
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
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--epoch-identifier", help="single session, e.g. chimi20200213_02_r1_HPC"
    )
    g.add_argument("--animal", help="all HPC sessions for this animal")
    g.add_argument("--all", action="store_true", help="every HPC session")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=content_occupancy_dir(),
        help="default: content_occupancy/ under the processed-data root",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

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
    np.savez(
        args.output_dir / "content_occupancy_2dhist.npz",
        xe=XE,
        ye=YE,
        nonSWR=hist2d["nonSWR"],
        SWR=hist2d["SWR"],
        dwe=DWE,
        dwell_nonSWR=dwell_hist["nonSWR"],
        dwell_SWR=dwell_hist["SWR"],
    )
    render(metrics, hist2d, dwell_hist, args.output_dir)

    print(
        f"\nWrote per-session metrics ({len(metrics)} rows) + 2D histogram + figures to {args.output_dir}"
    )
    print("\nQuadrant fractions (n-weighted):")
    shares = group_shares(metrics)
    for group in GROUPS:
        print(
            f"  {group}: "
            + ", ".join(f"{q}={shares.loc[group, q]:.2f}" for q in OCCUPANCY_QUADRANTS)
        )
    print("\nPer-animal non-SWR (mean over each animal's sessions):")
    pa = animal_occupancy_shares(metrics, "nonSWR")[list(OCCUPANCY_QUADRANTS)]
    print(pa.round(2).to_string())


if __name__ == "__main__":
    main()
