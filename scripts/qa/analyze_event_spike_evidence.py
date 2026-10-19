"""Spike-evidence reality analysis for non-SWR non-local events.

Tests whether non-local decoding events that do not overlap a sharp-wave
ripple (SWR) are *real* — driven by spikes that genuinely represent a remote
location — rather than decoding artifacts produced by the prior when spikes
are sparse or non-spatial.

The key idea: the decoder reports ``posterior = likelihood x prior``. Artifacts
produce a confident *posterior* via the *prior* while the *likelihood* (the
spike evidence) stays uninformative. So we recompute the **prior-excluded
clusterless spike likelihood** over track positions for each event, directly
from the saved encoding model and the event's marks (no re-decode), and ask
whether it supports the decoded position over the animal's own position,
above a matched non-event immobility baseline.

Per-event metrics (all bin lookups use native track-graph distance):

- ``dll_animal`` = mean_t [ logL_t(MAP_t) - logL_t(animal_t) ]: does the spike
  likelihood favor the decoded position over the animal's? Real > 0.
- ``dll_per_spike`` = sum_t(logL(MAP) - logL(animal)) / n_spikes: per-spike
  fidelity (real-but-small vs uninformative spikes).
- ``n_spikes`` / ``n_tetrodes``: population engagement (rules out a spike vacuum).
- ``dist_to_gpi_peak``: track distance from the event MAP to the
  ground-process-intensity peak, vs a random-position reference (content
  specificity).

The matched non-event immobility baseline is the null (its dll_animal and
dll_per_spike sit at ~0).

Events decoded where the CPU rebuild is not the decode are left out. A real
place bin whose stored occupancy is above 0 and below
``UNDERFLOW_OCCUPANCY_FLOOR`` (float32 subnormals where the animal never went; a
GPU decode keeps them, the CPU flushes them to zero;
``subnormal_occupancy_bin_mask``) has a CPU likelihood that is not the decode's,
up to 20 nats per spike off (D1). A bin with an occupancy of exactly 0 does not:
both devices read ``occupancy > 0`` as False and agree there. An event (or
baseline window) with a scored bin whose MAP is in a flagged bin is not scored;
``event_spike_evidence_left_out.csv`` counts them per epoch. With no flagged bin
(every analysed epoch but chimi20200221_08_r4) the outputs are unchanged.

Usage
-----
    # one session
    uv run python scripts/qa/analyze_event_spike_evidence.py \
        --epoch-identifier chimi20200213_02_r1_HPC --make-figures

    # all HPC sessions for an animal (metrics only, aggregated)
    uv run python scripts/qa/analyze_event_spike_evidence.py --animal chimi
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
from continuum_swr_replay.analysis.hippocampus.artifacts.masks import (  # noqa: E402
    interval_mask,
)
from continuum_swr_replay.analysis.hippocampus.data_aggregation import (  # noqa: E402
    validate_dot_event_contract,
)
from continuum_swr_replay.analysis.hippocampus.event_detection import (  # noqa: E402
    event_strict_overlap_mask,
)
from continuum_swr_replay.analysis.hippocampus.event_immobility import (  # noqa: E402
    EventDefinitionError,
)
from continuum_swr_replay.analysis.hippocampus.state_evidence import (  # noqa: E402
    UNDERFLOW_OCCUPANCY_FLOOR,
    event_likelihood,
    load_encoding,
    nearest_interior_bins,
    subnormal_occupancy_bin_mask,
)
from continuum_swr_replay.analysis.hippocampus.swr_source import (  # noqa: E402
    excluded_intervals_of,
    read_dot_with_swr_source,
)
from continuum_swr_replay.data_loaders.epoch_register import (  # noqa: E402
    analysed_decode_files,
)
from continuum_swr_replay.data_loaders.utils import parse_epoch_identifier  # noqa: E402
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR  # noqa: E402

# --- Parameters (provisional; see spec) ---------------------------------------
NON_LOCAL_THRESHOLD = DEFAULT_NON_LOCAL_THRESHOLD
NON_EVENT_THRESHOLD = 0.2  # non-local prob below this = clearly local (baseline)
IMMOBILITY_CM_PER_S = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
CLOSE_CM = 40.0  # mean |ahead-behind| below this = "close" subset
REMOTE_CM = 100.0  # above this = "remote" subset
GPI_REMOTE_CM = 100.0  # random-reference bins must be this far from the animal
MIN_EVENT_BINS = 3
MAX_BASELINE_BINS = 40
N_SAMPLE_PER_GROUP = 30

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = _REPO_ROOT / "logs" / "reality_marks_cache"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "paper" / "figures" / "reality"

GROUP_ORDER = ["nonSWR-remote", "nonSWR-close", "SWR-remote", "baseline"]
GROUP_COLORS = {
    "nonSWR-remote": "#0072B2",
    "nonSWR-close": "#E69F00",
    "SWR-remote": "#009E73",
    "baseline": "#999999",
}


# --- Session / epoch helpers --------------------------------------------------
def animal_from_epoch_id(epoch_id: str) -> str:
    """``"chimi20200213_02_r1_HPC"`` -> ``"chimi"``, ``"j1620210709_..."`` -> ``"j16"``.

    The NWB stem ends in an 8-digit date, so the animal id is everything before
    it (``j16`` keeps its digits), as :func:`parse_epoch_identifier` reads it.
    """
    return parse_epoch_identifier(epoch_id).animal


def find_sessions(animal: str | None) -> list[str]:
    """Return analysed HPC epoch identifiers with detector + results + data_over_time.

    The epoch register's analysed epochs (``analysed_decode_files``), never a
    glob, so a pending epoch's decode in the folder is not read.
    """
    thr_int = int(round(NON_LOCAL_THRESHOLD * 100))
    out = []
    for det in analysed_decode_files(PROCESSED_DATA_DIR, "_detector.pkl"):
        epoch_id = det.name.removesuffix("_detector.pkl")
        if animal is not None and not epoch_id.startswith(animal):
            continue
        has_results = (PROCESSED_DATA_DIR / f"{epoch_id}_results.nc").exists()
        has_dot = (
            PROCESSED_DATA_DIR / f"{epoch_id}_{thr_int}_data_over_time.pkl"
        ).exists()
        if has_results and has_dot:
            out.append(epoch_id)
    return out


def _contiguous_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/end index pairs (inclusive) of contiguous True runs in ``mask``."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    brk = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[brk + 1]]
    ends = np.r_[idx[brk], idx[-1]]
    return list(zip(starts, ends, strict=True))


def build_session_events(dot: pd.DataFrame) -> pd.DataFrame:
    """Classify non-local events and matched non-event baseline windows.

    Parameters
    ----------
    dot : pd.DataFrame
        Per-timebin ``data_over_time`` (decode time axis), with the state-prob,
        ``SWR_event``, ``head_speed`` and ``ahead_behind_distance`` columns.

    Returns
    -------
    pd.DataFrame
        One row per window: ``s, e`` (bin indices), ``mean_abs_ab``, ``is_swr``,
        ``n_bins``, ``group``.
    """
    # Business-logic boundary: direct callers get the same fail-closed input
    # contract as the file-loading entry point.
    validate_dot_event_contract(dot)
    nlp = (
        dot["Non-Local_Continuous_prob"] + dot["Non-Local_Fragmented_prob"]
    ).to_numpy()
    immobile = dot["head_speed"].to_numpy() < IMMOBILITY_CM_PER_S
    ab = dot["ahead_behind_distance"].to_numpy()
    # The read holds the labels to the speed rule and the probability contract.
    nonlocal_labels = dot["non_local_event"].to_numpy()
    is_swr_event = event_strict_overlap_mask(
        dot, "non_local_event", "SWR_event"
    ).to_numpy()

    rows = []
    for event_label in pd.unique(nonlocal_labels[nonlocal_labels > 0]):
        event_bins = np.flatnonzero(nonlocal_labels == event_label)
        n_bins = len(event_bins)
        if n_bins < MIN_EVENT_BINS:
            continue
        if n_bins > 1 and not (np.diff(event_bins) == 1).all():
            raise ValueError(f"non_local_event {event_label} is not one contiguous run")
        s, e = int(event_bins[0]), int(event_bins[-1])
        mab = float(np.nanmean(np.abs(ab[event_bins])))
        is_swr = bool(is_swr_event[event_bins[0]])
        if is_swr and mab > REMOTE_CM:
            group = "SWR-remote"
        elif not is_swr and mab > REMOTE_CM:
            group = "nonSWR-remote"
        elif not is_swr and mab < CLOSE_CM:
            group = "nonSWR-close"
        else:
            continue  # 40-100 cm middle band, or SWR-close: excluded for clarity
        rows.append(
            {
                "s": s,
                "e": e,
                "mean_abs_ab": mab,
                "is_swr": is_swr,
                "n_bins": n_bins,
                "group": group,
            }
        )

    # the null is analysed time too: no baseline window touches excluded time
    # (the artifact mask widened by 0.3 s; the low-non-local label is decoded)
    time = dot.index.to_numpy()
    observed = ~interval_mask(time, excluded_intervals_of(dot))
    for s, e in _contiguous_spans((nlp < NON_EVENT_THRESHOLD) & immobile & observed):
        n_bins = e - s + 1
        if n_bins < MIN_EVENT_BINS or n_bins > MAX_BASELINE_BINS:
            continue
        rows.append(
            {
                "s": s,
                "e": e,
                "mean_abs_ab": 0.0,
                "is_swr": False,
                "n_bins": n_bins,
                "group": "baseline",
            }
        )

    return pd.DataFrame(rows)


# --- Encoding model + marks ---------------------------------------------------
def load_marks(epoch_id: str, cache_dir: Path) -> tuple[list, list]:
    """Load (and cache) HPC clusterless marks: spike_times, waveform_features.

    Marks are absolute-time and align with the decode axis (position-tail drift
    does not affect spike times). ``load_data`` is slow, so results are cached.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{epoch_id}_marks.pkl"
    if cache.exists():
        return joblib.load(cache)
    from continuum_swr_replay.data_loaders import load_data

    key = parse_epoch_identifier(epoch_id)
    data = load_data(key.nwb_file_name, key.epoch_name)
    marks = (
        data["spike_times"][key.brain_area],
        data["spike_waveform_features"][key.brain_area],
    )
    joblib.dump(marks, cache)
    return marks


# --- Likelihood + metrics -----------------------------------------------------
def events_with_map_in_left_out_bins(
    events: pd.DataFrame,
    mape: np.ndarray,
    lin: np.ndarray,
    env,
    left_out: np.ndarray,
) -> np.ndarray:
    """Which events have a scored bin whose MAP is in a left-out place bin.

    The bins are those :func:`event_metrics` scores (finite MAP and animal
    position), and the MAP is binned as it bins it
    (``nearest_interior_bins``).

    Parameters
    ----------
    events : pd.DataFrame
        ``s`` and ``e``, inclusive bin indices.
    mape, lin : ndarray, shape (n_time,)
        Decoded MAP and animal linear position, cm.
    env : Environment
        Fitted.
    left_out : ndarray of bool, shape (n_real_bins,)
        :func:`subnormal_occupancy_bin_mask`.

    Returns
    -------
    ndarray of bool, shape (n_events,)
    """
    hit = np.zeros(len(events), dtype=bool)
    for k, (s, e) in enumerate(zip(events["s"], events["e"], strict=True)):
        mt, at = mape[s : e + 1], lin[s : e + 1]
        valid = np.isfinite(mt) & np.isfinite(at)
        if valid.any():
            hit[k] = left_out[nearest_interior_bins(env, mt[valid])].any()
    return hit


def event_metrics(
    s: int,
    e: int,
    L: np.ndarray,
    n_spikes: int,
    mape: np.ndarray,
    lin: np.ndarray,
    env,
    pos: np.ndarray,
    gpi_peak_pos: float,
) -> dict | None:
    """Compute the per-event reality metrics (track-distance bins)."""
    mt, at = mape[s : e + 1], lin[s : e + 1]
    valid = np.isfinite(mt) & np.isfinite(at)
    if valid.sum() < 2 or n_spikes == 0:
        return None
    Lv = L[valid]
    i_map = nearest_interior_bins(env, mt[valid])
    i_animal = nearest_interior_bins(env, at[valid])
    n = len(i_map)
    delta = Lv[np.arange(n), i_map] - Lv[np.arange(n), i_animal]
    map_mean = float(np.nanmean(mt))
    dist_gpi = float(
        np.asarray(env.get_distances_to_interior_bins(np.array([[map_mean]]))).ravel()[
            np.argmin(np.abs(pos - gpi_peak_pos))
        ]
    )
    return {
        "dll_animal": float(np.nanmean(delta)),
        "dll_per_spike": float(np.nansum(delta) / n_spikes),
        "n_spikes": int(n_spikes),
        "map_mean": map_mean,
        "dist_to_gpi_peak": dist_gpi,
    }


def analyze_session(
    epoch_id: str, cache_dir: Path, n_sample: int, seed: int = 0
) -> tuple[pd.DataFrame, dict]:
    """Compute per-event reality metrics for one session.

    Returns ``(metrics_df, context)`` where context holds arrays needed for
    figures (``pos``, ``gpi``, ``t``, ``mape``, ``lin``, etc.).
    """
    thr_int = int(round(NON_LOCAL_THRESHOLD * 100))
    # The read refuses labels that break the speed rule or the probability
    # contract.
    dot = read_dot_with_swr_source(
        PROCESSED_DATA_DIR / f"{epoch_id}_{thr_int}_data_over_time.pkl",
        non_local_threshold=NON_LOCAL_THRESHOLD,
    )
    # Validate before loading encoding data so a missing/corrupt auxiliary
    # artifact cannot mask malformed event labels as an ordinary session skip.
    validate_dot_event_contract(dot)
    enc = load_encoding(epoch_id)
    env, encargs, pos, gpi = enc["env"], enc["encargs"], enc["pos"], enc["gpi"]
    gpi_peak_pos = float(pos[np.argmax(gpi)])

    events = build_session_events(dot)
    spike_times, waveform_features = load_marks(epoch_id, cache_dir)

    t = dot.index.to_numpy()
    mape = dot["map_estimate"].to_numpy()
    lin = dot["linear_position"].to_numpy()

    # events decoded where the CPU rebuild is not the decode are not scored
    left_out = subnormal_occupancy_bin_mask(
        enc["encargs"]["occupancy"], env, floor=UNDERFLOW_OCCUPANCY_FLOOR
    )
    n_events = len(events)
    n_events_left_out = 0
    if left_out.any():
        hit = events_with_map_in_left_out_bins(events, mape, lin, env, left_out)
        events = events[~hit]
        n_events_left_out = int(hit.sum())

    records = []
    animal = animal_from_epoch_id(epoch_id)
    for group, g in events.groupby("group"):
        if len(g) > n_sample:
            print(f"    {epoch_id} [{group}]: capping {len(g)} -> {n_sample} events")
        sample = g.sample(min(n_sample, len(g)), random_state=seed)
        for _, r in sample.iterrows():
            s, ev = int(r.s), int(r.e)
            L, n_spk, n_tet = event_likelihood(
                s, ev, t, lin, spike_times, waveform_features, encargs
            )
            m = event_metrics(s, ev, L, n_spk, mape, lin, env, pos, gpi_peak_pos)
            if m is not None:
                m.update(
                    group=group,
                    n_tetrodes=n_tet,
                    n_bins=int(r.n_bins),
                    start_time=float(t[s]),
                    end_time=float(t[ev]),
                    epoch_id=epoch_id,
                    animal=animal,
                )
                records.append(m)

    metrics_df = pd.DataFrame(records)
    context = {
        "pos": pos,
        "gpi": gpi,
        "t": t,
        "mape": mape,
        "lin": lin,
        "env": env,
        "encargs": encargs,
        "events": events,
        "spike_times": spike_times,
        "waveform_features": waveform_features,
        "gpi_peak_pos": gpi_peak_pos,
        "epoch_id": epoch_id,
        "left_out": {
            "floor": UNDERFLOW_OCCUPANCY_FLOOR,
            "bins": np.flatnonzero(left_out).tolist(),
            "n_events": n_events,
            "n_events_left_out": n_events_left_out,
        },
    }
    return metrics_df, context


# --- Figures ------------------------------------------------------------------
def render_session_figures(
    metrics_df: pd.DataFrame, context: dict, output_dir: Path
) -> None:
    """Heatmaps, metric distributions (with baseline) and the GPI-position panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    pos, gpi = context["pos"], context["gpi"]
    t, mape, lin = context["t"], context["mape"], context["lin"]
    events, epoch_id = context["events"], context["epoch_id"]
    order = np.argsort(pos)
    pos_sorted = pos[order]
    groups = [g for g in GROUP_ORDER if g in set(metrics_df["group"])]

    # 1. example likelihood heatmaps (per-timebin normalized)
    ex_groups = [
        g
        for g in ["nonSWR-remote", "nonSWR-close", "SWR-remote"]
        if g in set(events["group"])
    ]
    fig, axes = plt.subplots(
        1, len(ex_groups), figsize=(5.3 * len(ex_groups), 4.2), squeeze=False
    )
    for ax, name in zip(axes[0], ex_groups, strict=True):
        row = events[events.group == name].nlargest(1, "n_bins").iloc[0]
        s, e = int(row.s), int(row.e)
        L, n_spk, n_tet = event_likelihood(
            s,
            e,
            t,
            lin,
            context["spike_times"],
            context["waveform_features"],
            context["encargs"],
        )
        ln = np.exp(L - L.max(1, keepdims=True))
        ln = ln / ln.sum(1, keepdims=True)
        tr = (t[s : e + 1] - t[s]) * 1000
        ax.pcolormesh(tr, pos_sorted, ln[:, order].T, cmap="magma", shading="auto")
        ax.plot(tr, mape[s : e + 1], color="cyan", lw=1.5, label="MAP")
        ax.plot(tr, lin[s : e + 1], color="lime", lw=1.3, label="animal")
        ax.set_title(f"{name}  nspk={n_spk} ntet={n_tet}", fontsize=9)
        ax.set_xlabel("ms")
    axes[0][0].legend(fontsize=8, loc="upper right")
    axes[0][0].set_ylabel("linear position (cm)")
    fig.suptitle(f"{epoch_id}: per-timebin-normalized spike likelihood", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_dir / f"{epoch_id}_01_likelihood_heatmaps.png", dpi=140)
    plt.close(fig)

    # 2. metric distributions (with baseline)
    rng = np.random.RandomState(1)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (metric, lab) in zip(
        axs,
        [("dll_animal", "dLL: MAP vs animal"), ("dll_per_spike", "dLL per spike")],
        strict=True,
    ):
        for i, name in enumerate(groups):
            v = metrics_df.loc[metrics_df.group == name, metric].dropna().to_numpy()
            ax.scatter(
                np.full(len(v), i) + rng.uniform(-0.12, 0.12, len(v)),
                v,
                s=13,
                alpha=0.6,
                color=GROUP_COLORS[name],
            )
            ax.hlines(np.median(v), i - 0.22, i + 0.22, color="k", lw=2)
        ax.axhline(0, color="r", ls="--", lw=0.8)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, fontsize=7, rotation=15)
        ax.set_title(lab, fontsize=10)
    fig.suptitle(
        f"{epoch_id}: spike-evidence reality metrics (baseline = null)", fontsize=11
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / f"{epoch_id}_02_reality_metrics.png", dpi=140)
    plt.close(fig)

    # 3. GPI position panel
    gpi_peak, gpi_min = pos[np.argmax(gpi)], pos[np.argmin(gpi)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
    a1.plot(pos_sorted, gpi[order], color="purple")
    a1.axvline(gpi_peak, color="k", ls="--", label=f"GPI peak {gpi_peak:.0f}")
    a1.axvline(gpi_min, color="gray", ls=":", label=f"GPI min {gpi_min:.0f}")
    a1.set_xlabel("linear position (cm)")
    a1.set_ylabel("ground process intensity")
    a1.legend(fontsize=8)
    a1.set_title("GPI(x)")
    for name in ex_groups:
        v = metrics_df.loc[metrics_df.group == name, "map_mean"].dropna()
        a2.hist(
            v,
            bins=25,
            histtype="step",
            color=GROUP_COLORS[name],
            label=name,
            density=True,
        )
    a2.axvline(gpi_peak, color="k", ls="--")
    a2.axvline(gpi_min, color="gray", ls=":")
    a2.set_xlabel("event MAP position (cm)")
    a2.set_ylabel("density")
    a2.legend(fontsize=8)
    a2.set_title("Do event MAPs pile up at the GPI peak?")
    fig.tight_layout()
    fig.savefig(output_dir / f"{epoch_id}_03_gpi_position.png", dpi=140)
    plt.close(fig)


def summarize(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Per-session, per-group medians of the reality metrics."""
    return (
        metrics_df.groupby(["epoch_id", "group"])
        .agg(
            dll_animal=("dll_animal", "median"),
            dll_per_spike=("dll_per_spike", "median"),
            n_spikes=("n_spikes", "median"),
            n_tetrodes=("n_tetrodes", "median"),
            dist_to_gpi_peak=("dist_to_gpi_peak", "median"),
            n_events=("dll_animal", "size"),
        )
        .reset_index()
    )


def per_animal_medians(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Per-animal x group median of ``dll_per_spike`` (within-animal summary).

    This is the primary inferential grain (avoid event-level pseudoreplication):
    each animal contributes one value per group, and the headline check is how
    many animals show each event group exceeding their own non-event baseline.
    """
    return (
        metrics_df.groupby(["animal", "group"])["dll_per_spike"]
        .median()
        .unstack("group")
    )


# --- CLI ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--epoch-identifier", help="single session, e.g. chimi20200213_02_r1_HPC"
    )
    g.add_argument("--animal", help="run all HPC sessions for this animal")
    g.add_argument("--epoch-identifiers", help="comma-separated list of sessions")
    g.add_argument("--all", action="store_true", help="run every HPC session")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-sample", type=int, default=N_SAMPLE_PER_GROUP)
    parser.add_argument(
        "--make-figures", action="store_true", help="render per-session figures"
    )
    args = parser.parse_args()

    if args.epoch_identifier:
        sessions = [args.epoch_identifier]
    elif args.epoch_identifiers:
        sessions = [s.strip() for s in args.epoch_identifiers.split(",")]
    elif args.all:
        sessions = find_sessions(None)
    else:
        sessions = find_sessions(args.animal)
    print(f"Analyzing {len(sessions)} session(s)")

    all_metrics = []
    left_out_rows = []
    for epoch_id in sessions:
        try:
            metrics_df, context = analyze_session(
                epoch_id, args.cache_dir, args.n_sample
            )
        except EventDefinitionError:
            # A malformed saved event cohort invalidates the multi-session
            # comparison. Abort instead of writing a selectively incomplete
            # QA result.
            raise
        except Exception as exc:  # noqa: BLE001 - log and continue the sweep
            print(f"  {epoch_id}: FAILED ({type(exc).__name__}: {exc})")
            continue
        all_metrics.append(metrics_df)
        left_out = context["left_out"]
        left_out_rows.append(
            {
                "epoch_id": epoch_id,
                "floor": left_out["floor"],
                "n_underflowed_bins": len(left_out["bins"]),
                "n_events": left_out["n_events"],
                "n_events_left_out": left_out["n_events_left_out"],
            }
        )
        if left_out["n_events_left_out"]:
            print(
                f"    {epoch_id}: {left_out['n_events_left_out']} of "
                f"{left_out['n_events']} events left out (MAP in the "
                f"{len(left_out['bins'])} real bins with 0 < occupancy < "
                f"{left_out['floor']:g})"
            )
        med = metrics_df.groupby("group")["dll_per_spike"].median()
        print(
            f"  {epoch_id}: per-spike dLL by group -> "
            + ", ".join(f"{k}={v:.2f}" for k, v in med.items())
        )
        if args.make_figures:
            render_session_figures(metrics_df, context, args.output_dir)

    if not all_metrics:
        print("No sessions analyzed.")
        return
    metrics = pd.concat(all_metrics, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(args.output_dir / "event_spike_evidence_metrics.parquet")
    summary = summarize(metrics)
    summary.to_csv(args.output_dir / "event_spike_evidence_summary.csv", index=False)
    pd.DataFrame(left_out_rows).to_csv(
        args.output_dir / "event_spike_evidence_left_out.csv", index=False
    )
    print(f"\nWrote metrics ({len(metrics)} events) and summary to {args.output_dir}")
    print(
        "\nPooled per-group medians (NOT the inferential statistic — see per-animal below):"
    )
    print(
        metrics.groupby("group")[
            ["dll_animal", "dll_per_spike", "n_spikes", "dist_to_gpi_peak"]
        ]
        .median()
        .to_string()
    )

    pam = per_animal_medians(metrics)
    pam.to_csv(args.output_dir / "event_spike_evidence_per_animal.csv")
    print("\nPer-animal median dLL/spike (within-animal summary):")
    print(pam.to_string())
    if "baseline" in pam.columns:
        print(
            "\nWithin-animal direction test: event group median > that animal's baseline"
        )
        for grp in [g for g in GROUP_ORDER if g != "baseline" and g in pam.columns]:
            valid = pam[grp].notna() & pam["baseline"].notna()
            n_exceed = int((pam.loc[valid, grp] > pam.loc[valid, "baseline"]).sum())
            print(f"  {grp}: {n_exceed}/{int(valid.sum())} animals")


if __name__ == "__main__":
    main()
