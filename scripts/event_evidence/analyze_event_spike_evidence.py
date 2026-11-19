"""Spike-evidence reality analysis for non-SWR non-local events.

Tests whether non-local decoding events that do not overlap a sharp-wave
ripple (SWR) are *real* — driven by spikes that genuinely represent a remote
location — rather than decoding artifacts produced by the prior when spikes
are sparse or non-spatial.

The key idea: the decoder reports ``posterior = likelihood x prior``. Artifacts
produce a confident *posterior* via the *prior* while the *likelihood* (the
spike evidence) stays uninformative. So we read the **prior-excluded
clusterless spike likelihood** over track positions for each event and ask
whether it supports the decoded position over the animal's own position,
above a matched non-event immobility baseline.

The likelihood is the decode's own: the ``log_likelihood`` the HPC decode
stores in ``results.nc`` beside the posterior (``--likelihood stored``, the
default). Its Non-Local Continuous columns at the interior bins are the
non-local likelihood (the Fragmented columns are the same). Row ``k`` holds the
spikes in ``[t_k, t_{k+1})``, so an event of bins ``s..e`` spans
``[t_s, t_{e+1})`` and its spikes are counted over that span.
``--likelihood rebuild`` instead recomputes it on the CPU from the saved
encoding model and the event's marks (``state_evidence.event_likelihood``);
that rebuild bins spikes on the window's own axis, so its last row holds no
spike, and it is kept as the reference the stored path was checked against
(rows ``s..e`` of the stored likelihood equal the rebuild over ``s..e+1`` to
float32 rounding).

Per-event metrics, computed by ``analysis.event_evidence.spike_evidence`` (which
also builds the event groups and quiet windows; all bin lookups use native
track-graph distance):

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

A real place bin whose stored occupancy is above 0 and below
``UNDERFLOW_OCCUPANCY_FLOOR`` (float32 subnormals where the animal never went; a
GPU decode keeps them, the CPU flushes them to zero;
``subnormal_occupancy_bin_mask``) has a CPU likelihood that is not the decode's,
up to 20 nats per spike off (D1). The stored likelihood is the decode's, so it
scores every event; ``event_spike_evidence_left_out.csv`` counts per epoch the
events (and baseline windows) with a scored bin whose MAP is in a flagged bin
(``n_events_map_in_flagged_bins``). Only the rebuild leaves them out
(``n_events_left_out``).

Usage
-----
    # one session
    uv run python scripts/event_evidence/analyze_event_spike_evidence.py \
        --epoch-identifier chimi20200213_02_r1_HPC --make-figures

    # all HPC sessions for an animal (metrics only, aggregated)
    uv run python scripts/event_evidence/analyze_event_spike_evidence.py --animal chimi
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

from continuum_swr_replay.analysis.detection.nonlocal_events import (  # noqa: E402
    validate_dot_event_contract,
)
from continuum_swr_replay.analysis.event_definition.analyzed_time import (  # noqa: E402
    read_dot_with_swr_source,
)
from continuum_swr_replay.analysis.event_definition.immobility import (  # noqa: E402
    EventDefinitionError,
)
from continuum_swr_replay.analysis.event_evidence.spike_evidence import (  # noqa: E402
    SPIKE_EVIDENCE_METRICS,
    build_session_events,
    event_metrics,
    stored_event_likelihood,
    stored_likelihood_columns,
    window_spike_counts,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence import (  # noqa: E402
    UNDERFLOW_OCCUPANCY_FLOOR,
    event_likelihood,
    load_encoding,
    nearest_interior_bins,
    subnormal_occupancy_bin_mask,
)
from continuum_swr_replay.analysis.event_properties import (  # noqa: E402
    event_statistics,
)
from continuum_swr_replay.analysis.shared_statistics import (  # noqa: E402
    animal_medians,
)
from continuum_swr_replay.data_loaders.epoch_register import (  # noqa: E402
    analyzed_decode_files,
)
from continuum_swr_replay.decoding.files import (  # noqa: E402
    DETECTOR_SUFFIX,
    decode_paths,
    epoch_identifier_of,
)
from continuum_swr_replay.decoding.identifiers import (  # noqa: E402
    parse_epoch_identifier,
)
from continuum_swr_replay.decoding.results import open_results  # noqa: E402
from continuum_swr_replay.intervals import sample_step  # noqa: E402
from continuum_swr_replay.parameters import DEFAULT_NON_LOCAL_THRESHOLD  # noqa: E402
from continuum_swr_replay.paths import processed_data_dir  # noqa: E402

# --- Parameters ---------------------------------------------------------------
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
def find_sessions(animal: str | None) -> list[str]:
    """Return analyzed HPC epoch identifiers with detector + results + data_over_time.

    The epoch register's analyzed epochs (``analyzed_decode_files``), never a
    glob, so a pending epoch's decode in the folder is not read.
    """
    out = []
    for det in analyzed_decode_files(processed_data_dir(), DETECTOR_SUFFIX):
        epoch_id = epoch_identifier_of(det, DETECTOR_SUFFIX)
        if animal is not None and parse_epoch_identifier(epoch_id).animal != animal:
            continue
        paths = decode_paths(epoch_id, processed_data_dir())
        if paths.results.exists() and paths.data_over_time.exists():
            out.append(epoch_id)
    return out


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


def analyze_session(
    epoch_id: str,
    cache_dir: Path,
    n_sample: int,
    seed: int = 0,
    likelihood: str = "stored",
) -> tuple[pd.DataFrame, dict]:
    """Compute per-event reality metrics for one session.

    ``likelihood`` is ``"stored"`` (the decode's ``log_likelihood``) or
    ``"rebuild"`` (recomputed on the CPU; see the module docstring).

    Returns ``(metrics_df, context)`` where context holds arrays needed for
    figures (``pos``, ``gpi``, ``t``, ``mape``, ``lin``, etc.).
    """
    if likelihood not in ("stored", "rebuild"):
        raise ValueError(
            f"likelihood must be 'stored' or 'rebuild', not {likelihood!r}"
        )
    # The read refuses labels that break the speed rule or the probability
    # contract.
    paths = decode_paths(epoch_id, processed_data_dir())
    dot = read_dot_with_swr_source(
        paths.data_over_time, non_local_threshold=DEFAULT_NON_LOCAL_THRESHOLD
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

    # where the CPU rebuild is not the decode: the rebuild leaves those events
    # out; the stored likelihood is the decode's and scores them
    left_out = subnormal_occupancy_bin_mask(
        enc["encargs"]["occupancy"], env, floor=UNDERFLOW_OCCUPANCY_FLOOR
    )
    n_events = len(events)
    hit = np.zeros(n_events, dtype=bool)
    if left_out.any():
        hit = events_with_map_in_left_out_bins(events, mape, lin, env, left_out)
    n_events_map_in_flagged_bins = int(hit.sum())
    n_events_left_out = 0
    if likelihood == "rebuild":
        events = events[~hit]
        n_events_left_out = n_events_map_in_flagged_bins

    if likelihood == "stored":
        results = open_results(paths.results, cache=False)
        if not np.array_equal(results["time"].values, t):
            raise ValueError(f"{epoch_id}: results.nc and data_over_time times differ")
        log_likelihood = results["log_likelihood"]
        cols = stored_likelihood_columns(results, env)
        bin_s = sample_step(t)

    records = []
    animal = parse_epoch_identifier(epoch_id).animal
    for group, g in events.groupby("group"):
        if len(g) > n_sample:
            print(f"    {epoch_id} [{group}]: capping {len(g)} -> {n_sample} events")
        sample = g.sample(min(n_sample, len(g)), random_state=seed)
        for _, r in sample.iterrows():
            s, ev = int(r.s), int(r.e)
            if likelihood == "stored":
                L = stored_event_likelihood(log_likelihood, cols, s, ev)
                stop = t[ev + 1] if ev + 1 < len(t) else t[ev] + bin_s
                n_spk, n_tet = window_spike_counts(spike_times, t[s], stop)
            else:
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
    if likelihood == "stored":
        results.close()

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
            "n_events_map_in_flagged_bins": n_events_map_in_flagged_bins,
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
    many animals show each event group exceeding their own non-event baseline
    (``n_animals_above_baseline`` of
    ``event_statistics.spike_evidence_statistics``, which compares these
    medians).
    """
    return pd.DataFrame(
        {
            group: animal_medians(rows["dll_per_spike"], rows["animal"])
            for group, rows in metrics_df.groupby("group")
        }
    ).rename_axis(columns="group")


# --- CLI ----------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> None:
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
        "--likelihood",
        choices=("stored", "rebuild"),
        default="stored",
        help="the decode's stored log_likelihood (default) or a CPU rebuild",
    )
    parser.add_argument(
        "--make-figures", action="store_true", help="render per-session figures"
    )
    args = parser.parse_args(argv)

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
                epoch_id, args.cache_dir, args.n_sample, likelihood=args.likelihood
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
                "n_events_map_in_flagged_bins": left_out[
                    "n_events_map_in_flagged_bins"
                ],
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
    metrics.to_parquet(args.output_dir / SPIKE_EVIDENCE_METRICS)
    summary = summarize(metrics)
    summary.to_csv(args.output_dir / "event_spike_evidence_summary.csv", index=False)
    pd.DataFrame(left_out_rows).to_csv(
        args.output_dir / "event_spike_evidence_left_out.csv", index=False
    )
    print(f"\nWrote metrics ({len(metrics)} events) and summary to {args.output_dir}")

    pam = per_animal_medians(metrics)
    pam.to_csv(args.output_dir / "event_spike_evidence_per_animal.csv")
    print("\nPer-animal median dLL/spike (within-animal summary):")
    print(pam.to_string())
    if "baseline" not in set(metrics["group"]):
        return
    # the statistics the events summary reports from these metrics; they refuse
    # a run in which an epoch has no quiet window to compare with
    statistics = event_statistics.spike_evidence_statistics(metrics, "baseline")
    print(
        "\nEach group against the matched quiet windows: pooled medians, and the "
        "epochs and animals whose median dLL/spike is above their own baseline's "
        "(event_statistics.spike_evidence_statistics):"
    )
    print(
        statistics.assign(
            n_animals_scored=metrics.groupby("group")["animal"].nunique()
        ).to_string()
    )


if __name__ == "__main__":
    main()
