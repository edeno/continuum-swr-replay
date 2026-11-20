"""One class or two: do SWR-coupled and non-SWR non-local events differ by kind or by degree of ripple power?

Two tests over the immobile non-local events of the canonical event table
(``analysis.event_properties.class_or_continuum`` holds the method):

1. **One mode or two.** Per animal and pooled, the distribution of event ripple
   power (the event's peak and mean of the z-scored Kay consensus, and the Kay
   detector's own criterion around the event), with Hartigan's dip test,
   Sarle's coefficient (descriptive) and the KDE's modes, against the Kay
   threshold. Calibrated first on simulated unimodal samples matched to each
   animal's mean, SD and skew, and on two-class mixtures at the observed
   separation of the labels.
2. **A step at the threshold.** Per animal and property: ``y ~ s(power)``
   against ``y ~ s(power) + label``, with and without a smooth of log
   duration, by held-out squared error (5 folds grouped by epoch) and by the
   label term with an epoch-clustered 95 % interval. With the event's peak or
   mean power (label ``is_swr``, rows in the common support), and as a sharp
   discontinuity design on the detector's criterion (label ``criterion >= 2``,
   rows within ``--window`` z of the threshold). Calibrated on simulated data:
   a planted step or none, and one continuum labeled by the Kay rule on a
   simulated ripple trace.

Stage ``measure`` reads each epoch's decode frame through the SWR source (its
sha256 must be the one the event features run recorded) and writes, per event,
the event's peak and mean ripple power and the Kay criterion, and whether its
rows hold an SWR label (``criterion/<epoch>.parquet``). Stage ``analyze``
writes the tables, figures and ``manifest.json`` to
``Processed-Data/ripple_class_structure/``.

Usage::

    uv run python scripts/event_properties/measure_ripple_class_structure.py measure \\
        --n-workers 8 [--epochs chimi20200212_02_r1_HPC]
    uv run python scripts/event_properties/measure_ripple_class_structure.py analyze
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from continuum_swr_replay.analysis.constants import REMOTE_CM  # noqa: E402
from continuum_swr_replay.analysis.detector_comparison import (  # noqa: E402
    inventory_run,
)
from continuum_swr_replay.analysis.detector_comparison.feature_run import (  # noqa: E402
    STATE_SPACE_FEATURES_RUN,
)
from continuum_swr_replay.analysis.detector_comparison.features import (  # noqa: E402
    _grid_rows,
)
from continuum_swr_replay.analysis.detector_comparison.schema import (  # noqa: E402
    native_event_keys,
)
from continuum_swr_replay.analysis.event_definition import analyzed_time  # noqa: E402
from continuum_swr_replay.analysis.event_definition.analyzed_time import read_swr_table
from continuum_swr_replay.analysis.event_definition.swr_source import (  # noqa: E402
    swr_epoch_id_of_session,
)
from continuum_swr_replay.analysis.event_evidence.spike_evidence import (  # noqa: E402
    SPIKE_EVIDENCE_METRICS,
)
from continuum_swr_replay.analysis.event_properties import (
    class_or_continuum as cc,  # noqa: E402
)
from continuum_swr_replay.paths import (
    RIPPLE_CLASS_STRUCTURE_DIR,
    # noqa: E402
    event_features_root,
    event_spike_evidence_dir,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.io import (  # noqa: E402
    files_sha256,
    ok_table_paths,
    read_manifest,
    sha256_file,
    write_json,
)
from continuum_swr_replay.runs.provenance import (  # noqa: E402
    described_commit,
    produced_by,
)
from continuum_swr_replay.visualization.palettes import COLORS  # noqa: E402

SEED = 20261027


def out_dir() -> Path:
    return processed_data_dir() / RIPPLE_CLASS_STRUCTURE_DIR


# --- stage measure -------------------------------------------------------------


def measure_epoch(job: tuple[str, str, pd.DataFrame]) -> pd.DataFrame:
    """One epoch's events: peak, mean and criterion of the ripple power, and
    whether their rows hold an SWR label of the frame."""
    identifier, expected_sha256, events = job
    dot = analyzed_time.read_verified_dot_with_swr_source(
        inventory_run.decode_path(identifier), expected_sha256
    )
    first, n_rows = _grid_rows(dot.index.to_numpy(dtype=np.float64), events)
    try:
        power = cc.event_ripple_power(
            dot["SWR_zscore"].to_numpy(dtype=np.float64),
            dot["SWR_event"].to_numpy() > 0,
            first,
            n_rows,
        )
    except ValueError as error:
        raise ValueError(f"{identifier}: {error}") from error
    return pd.DataFrame(
        {
            "event_id": events.event_id.to_numpy(),
            "epoch_id": identifier,
            "decode_sha256": expected_sha256,
            "first_row": first,
            "n_rows": n_rows,
            **power,
        }
    )


def epoch_identifier(row: pd.Series) -> str:
    return swr_epoch_id_of_session(row.animal, row.session_date, row.epoch)


def run_measure(args: argparse.Namespace) -> None:
    manifest = read_manifest(event_features_root() / STATE_SPACE_FEATURES_RUN)
    events = read_swr_table(table_path("event_table", replay_content_dir()))
    events = events.assign(epoch_id=events.apply(epoch_identifier, axis=1))
    by_epoch = dict(tuple(events.groupby("epoch_id")))
    identifiers = sorted(by_epoch)
    if args.epochs:
        identifiers = [i for i in identifiers if i in set(args.epochs)]
    jobs = []
    for identifier in identifiers:
        record = manifest["epochs"][identifier]
        if record["status"] != "ok":
            raise ValueError(f"{identifier}: event features status {record['status']}")
        jobs.append(
            (
                identifier,
                record["decode"]["sha256"],
                by_epoch[identifier][["event_id", "event_start_s", "event_end_s"]],
            )
        )
    folder = out_dir() / "criterion"
    folder.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        for table in pool.map(measure_epoch, jobs):
            identifier = table.epoch_id.iloc[0]
            tmp = folder / f"{identifier}.parquet.tmp"
            table.to_parquet(tmp, index=False)
            tmp.replace(folder / f"{identifier}.parquet")
            print(f"{identifier}: {len(table)} events", flush=True)


# --- stage analyze: inputs -------------------------------------------------------


def load_events() -> tuple[pd.DataFrame, dict[str, Any]]:
    """The canonical events with their ripple power, spike counts and spike evidence.

    The spike counts are read from the features tables of the epochs the event
    features run's manifest marks ``ok`` (``runs.io.ok_table_paths``)."""
    event_table = table_path("event_table", replay_content_dir())
    events = read_swr_table(event_table)
    n_events = len(events)
    if not events.whole_event_speed_eligible.all():
        raise ValueError(
            "the event table holds events that are not whole-event immobile"
        )

    criterion_paths = sorted((out_dir() / "criterion").glob("*.parquet"))
    criterion = pd.concat(map(pd.read_parquet, criterion_paths), ignore_index=True)
    run = event_features_root() / STATE_SPACE_FEATURES_RUN
    feature_paths = ok_table_paths(run, "features")
    features = pd.concat(
        (
            pd.read_parquet(
                p,
                columns=["event_key", "ripple_z_max", "ripple_z_mean", "n_spikes"],
                filters=[("detector", "==", "state_space")],
            )
            for p in feature_paths
        ),
        ignore_index=True,
    )
    features["event_id"] = native_event_keys(features.event_key).astype("uint64")
    events = events.merge(
        features[["event_id", "ripple_z_max", "ripple_z_mean", "n_spikes"]].rename(
            columns={
                "ripple_z_max": "features_z_max",
                "ripple_z_mean": "features_z_mean",
            }
        ),
        on="event_id",
        how="left",
        validate="1:1",
    ).merge(criterion, on="event_id", how="left", validate="1:1")
    if events[["features_z_max", "ripple_z_criterion"]].isna().any().any():
        raise ValueError("an event has no features-run or criterion row")
    checks = {
        "n_events": n_events,
        "peak_differs_from_features_run": int(
            (np.abs(events.ripple_z_max - events.features_z_max) > 1e-9).sum()
        ),
        "mean_differs_from_features_run": int(
            (np.abs(events.ripple_z_mean - events.features_z_mean) > 1e-9).sum()
        ),
        "mean_differs_from_event_table": int(
            (np.abs(events.ripple_z_mean - events.ripple_power_z) > 1e-9).sum()
        ),
        "rows_hold_swr_disagrees_with_is_swr": int(
            (events.rows_hold_swr != events.is_swr).sum()
        ),
    }

    metric_paths = sorted(
        event_spike_evidence_dir().glob(f"*/{SPIKE_EVIDENCE_METRICS}")
    )
    metrics = pd.concat(map(pd.read_parquet, metric_paths), ignore_index=True)
    metrics = metrics[metrics.group != "baseline"]
    events = events.merge(
        metrics[
            ["epoch_id", "start_time", "end_time", "dll_per_spike", "group"]
        ].rename(columns={"start_time": "event_start_s", "end_time": "event_end_s"}),
        on=["epoch_id", "event_start_s", "event_end_s"],
        how="left",
        validate="1:1",
    )
    scored = events.group.notna()
    if (
        events.loc[scored, "is_swr"]
        != events.loc[scored, "group"].str.startswith("SWR")
    ).any():
        raise ValueError("a spike-evidence group disagrees with is_swr")
    checks["events_with_spike_evidence"] = int(scored.sum())
    events["remote"] = events.mean_abs_ahead_behind_cm > REMOTE_CM
    checks["remote_events_without_spike_evidence"] = int(
        (events.remote & events.dll_per_spike.isna()).sum()
    )
    inputs = {
        "event_table": {"path": str(event_table), "sha256": sha256_file(event_table)},
        "event_features_run": {
            "run": STATE_SPACE_FEATURES_RUN,
            "n_files": len(feature_paths),
            "sha256": files_sha256(feature_paths),
        },
        "criterion": {
            "n_files": len(criterion_paths),
            "sha256": files_sha256(criterion_paths),
            "decode_sha256": dict(
                sorted(criterion.groupby("epoch_id").decode_sha256.first().items())
            ),
        },
        "spike_evidence": {
            "path": str(event_spike_evidence_dir()),
            "n_files": len(metric_paths),
            "sha256": files_sha256(metric_paths),
        },
    }
    return events, {"checks": checks, "inputs": inputs}


# --- test 1 -------------------------------------------------------------------------


SEPARATION_FACTORS = (1.0, 2.0, 3.0, 4.0)
"""Two-class mixtures with the labels' separation times these factors."""


def distribution_simulation(distribution: pd.DataFrame, n_reps: int) -> pd.DataFrame:
    """Dip and Sarle on samples whose answer is known, matched to each row of
    ``distribution``: unimodal (gamma, lognormal) at its mean, SD and skew, and
    two Gaussian classes at the labels' medians (the SWR class moved out to
    :data:`SEPARATION_FACTORS` times the observed separation), robust SDs and
    shares."""
    rng = np.random.default_rng(SEED)
    scenarios = [("unimodal_gamma", np.nan), ("unimodal_lognormal", np.nan)]
    scenarios += [("two_classes", k) for k in SEPARATION_FACTORS]
    rows = []
    for row in distribution.itertuples():
        for scenario, factor in scenarios:
            for rep in range(n_reps):
                x = _matched_sample(rng, scenario, row, factor)
                dip, p_value = cc.dip_test(x)
                rows.append(
                    {
                        "animal": row.animal,
                        "power": row.power,
                        "scenario": scenario,
                        "separation_factor": factor,
                        "rep": rep,
                        "dip": dip,
                        "dip_p": p_value,
                        "sarle": cc.sarle_bimodality(x),
                    }
                )
    return pd.DataFrame(rows)


def _matched_sample(
    rng: np.random.Generator, scenario: str, row: Any, factor: float
) -> np.ndarray:
    n = int(row.n_events)
    if scenario == "unimodal_gamma":
        return cc.matched_unimodal_sample(rng, n, row.mean, row.sd, row.skew)
    if scenario == "unimodal_lognormal":
        return cc.matched_lognormal_sample(rng, n, row.mean, row.sd, row.skew)
    separation = row.median_swr - row.median_non_swr
    return cc.two_class_mixture_sample(
        rng,
        n,
        (row.median_non_swr, row.median_non_swr + factor * separation),
        (row.robust_sd_non_swr, row.robust_sd_swr),
        row.swr_share,
    )


def plot_distributions(
    events: pd.DataFrame, distribution: pd.DataFrame, path: Path
) -> None:
    animals = sorted(events.animal.unique())
    fig, axes = plt.subplots(
        len(cc.RIPPLE_POWERS),
        len(animals),
        figsize=(3.0 * len(animals), 2.4 * len(cc.RIPPLE_POWERS)),
    )
    for i, (word, column) in enumerate(cc.RIPPLE_POWERS.items()):
        for j, animal in enumerate(animals):
            ax = axes[i, j]
            frame = events[(events.animal == animal) & np.isfinite(events[column])]
            x = frame[column]
            hi = np.quantile(x, 0.99)
            lo = min(np.quantile(x, 0.001), -1.0)
            bins = np.linspace(lo, hi, 80)
            ax.hist(
                [x[~frame.is_swr], x[frame.is_swr]],
                bins=bins,
                stacked=True,
                color=[COLORS["No SWR"], COLORS["SWR"]],
                label=["non-SWR", "SWR"],
            )
            ax.axvline(cc.KAY_ZSCORE_THRESHOLD, color="k", lw=1, ls="--")
            row = distribution[
                (distribution.animal == animal) & (distribution.power == word)
            ].iloc[0]
            ax.set_title(
                f"{animal}: dip p={row.dip_p:.2g}, modes={row.n_modes}", fontsize=8
            )
            ax.set_xlabel(f"ripple {word} (z)", fontsize=8)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel("events", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Event ripple power; dashed: Kay threshold (2 z)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- test 2 -------------------------------------------------------------------------


def step_simulation(n_reps: int, window: float, n_knots: int) -> pd.DataFrame:
    """The step test on data whose answer is known."""
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for planted in (0.0, 0.3):
        for rep in range(n_reps):
            d = cc.simulate_step_data(rng, step=planted)
            low, high = cc.common_support(d.power, d.is_swr)
            s = d[(d.power >= low) & (d.power <= high)]
            fit = cc.fit_step(s.y, s.power, s.is_swr, s.group, n_knots=n_knots)
            rows.append(_sim_row("smooth_label", planted, rep, "power", "none", fit))
    for planted in (0.0, 0.3):
        for rep in range(n_reps):
            d = pd.concat(
                [
                    cc.simulate_ripple_events(
                        rng, duration_s=300.0, step=planted
                    ).assign(group=g)
                    for g in range(12)
                ],
                ignore_index=True,
            )
            for power, column in (
                ("peak", "power_peak"),
                ("mean", "power_mean"),
                ("criterion", "power_criterion"),
            ):
                x = d[column].to_numpy()
                if power == "criterion":
                    keep = np.abs(x - cc.KAY_ZSCORE_THRESHOLD) <= window
                    label = x[keep] >= cc.KAY_ZSCORE_THRESHOLD
                else:
                    low, high = cc.common_support(x, d.is_swr.to_numpy())
                    keep = (x >= low) & (x <= high)
                    label = d.is_swr.to_numpy()[keep]
                s = d[keep]
                for covariate in ("none", "log_duration"):
                    fit = cc.fit_step(
                        s.y,
                        s[column],
                        label,
                        s.group,
                        covariate=None if covariate == "none" else s.log_duration,
                        n_knots=n_knots,
                    )
                    rows.append(
                        _sim_row("ripple_trace", planted, rep, power, covariate, fit)
                    )
    return pd.DataFrame(rows)


def _sim_row(scenario, planted, rep, power, covariate, fit) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "planted_step": planted,
        "rep": rep,
        "power": power,
        "covariate": covariate,
        **fit.__dict__,
        "has_step": fit.has_step,
        "covers_planted": fit.step_ci_low <= planted <= fit.step_ci_high,
    }


def plot_property(events: pd.DataFrame, word: str, window: float, path: Path) -> None:
    """Binned means of the property against ripple power, by label, per animal."""
    animals = sorted(events.animal.unique())
    fig, axes = plt.subplots(3, len(animals), figsize=(3.0 * len(animals), 7.2))
    for i, power in enumerate(cc.RIPPLE_POWERS):
        for j, animal in enumerate(animals):
            ax = axes[i, j]
            frame = cc.property_frame(events[events.animal == animal], word)
            keep, label, low, high = cc.design_rows(frame, power, window)
            x = frame[cc.RIPPLE_POWERS[power]]
            span = (
                (low - 1.0, high + 1.0)
                if power != "criterion"
                else (low - 0.5, high + 0.5)
            )
            in_view = (x >= span[0]) & (x <= span[1])
            edges = np.quantile(x[in_view], np.linspace(0, 1, 21))
            for is_swr, name in ((False, "Non-Local"), (True, "SWR")):
                rows = frame[in_view & (frame.is_swr == is_swr)]
                bins = pd.cut(
                    rows[cc.RIPPLE_POWERS[power]], np.unique(edges), include_lowest=True
                )
                stats = rows.groupby(bins, observed=True).y.agg(["mean", "sem", "size"])
                stats = stats[stats["size"] >= 20]
                centers = [b.mid for b in stats.index]
                ax.errorbar(
                    centers,
                    stats["mean"],
                    yerr=stats["sem"],
                    color=COLORS[name],
                    marker="o",
                    ms=3,
                    lw=1,
                    label="SWR" if is_swr else "non-SWR",
                )
            ax.axvline(cc.KAY_ZSCORE_THRESHOLD, color="k", lw=1, ls="--")
            ax.axvspan(low, high, color="0.9", zorder=-1)
            ax.set_title(animal, fontsize=8)
            ax.set_xlabel(f"ripple {power} (z)", fontsize=8)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(cc.STEP_PROPERTIES[word][0], fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(
        f"{cc.STEP_PROPERTIES[word][0]} by ripple power and label; gray: rows compared; "
        "dashed: Kay threshold",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- spot checks ----------------------------------------------------------------------


def spot_check_events(events: pd.DataFrame) -> pd.DataFrame:
    """Per animal, typical and extreme events: the label's two odd corners
    (non-SWR at or above threshold in the event, SWR below it), events just
    either side of the threshold on the criterion, and the longest and fastest."""
    rng = np.random.default_rng(SEED + 2)
    picks = []
    t = cc.KAY_ZSCORE_THRESHOLD
    for _, frame in events.groupby("animal"):
        crit = frame.ripple_z_criterion
        cases = {
            "non_swr_peak_above_threshold": frame[
                ~frame.is_swr & (frame.ripple_z_max >= t)
            ],
            "swr_peak_below_threshold": frame[frame.is_swr & (frame.ripple_z_max < t)],
            "criterion_just_below": frame[(crit >= t - 0.1) & (crit < t)],
            "criterion_just_above": frame[(crit >= t) & (crit < t + 0.1)],
        }
        for case, rows in cases.items():
            if len(rows):
                picks.append(
                    rows.sample(
                        min(3, len(rows)), random_state=rng.integers(1 << 31)
                    ).assign(case=case)
                )
        picks.append(frame.nlargest(2, "event_duration_s").assign(case="longest"))
        picks.append(
            frame.nlargest(2, "replay_speed_median_cm_s").assign(case="fastest")
        )
    columns = [
        "case",
        "animal",
        "epoch_id",
        "event_id",
        "event_start_s",
        "event_end_s",
        "event_duration_s",
        "is_swr",
        "overlap_class",
        "matched_swr_start",
        "matched_swr_end",
        "ripple_z_max",
        "ripple_z_mean",
        "ripple_z_criterion",
        "spatial_extent_cm",
        "replay_speed_median_cm_s",
        "posterior_peak_mean",
        "n_spikes",
        "n_unique_arms",
    ]
    return pd.concat(picks, ignore_index=True)[columns]


def plot_spot_checks(spots: pd.DataFrame, path: Path) -> None:
    """The ripple trace around one spot-check event of each case, in one epoch
    per animal (the epoch holding most picks)."""
    cases = [
        "non_swr_peak_above_threshold",
        "swr_peak_below_threshold",
        "criterion_just_below",
        "criterion_just_above",
    ]
    animals = sorted(spots.animal.unique())
    fig, axes = plt.subplots(len(animals), len(cases), figsize=(12, 2.0 * len(animals)))
    for i, animal in enumerate(animals):
        mine = spots[(spots.animal == animal) & spots.case.isin(cases)]
        for j, case in enumerate(cases):
            ax = axes[i, j]
            rows = mine[mine.case == case]
            if rows.empty:
                ax.axis("off")
                continue
            event = rows.iloc[0]
            dot = analyzed_time.read_dot_with_swr_source(
                inventory_run.decode_path(event.epoch_id)
            )
            window = dot.loc[event.event_start_s - 0.15 : event.event_end_s + 0.15]
            t0 = event.event_start_s
            ax.plot(window.index - t0, window.SWR_zscore, color="k", lw=0.8)
            ax.axhline(cc.KAY_ZSCORE_THRESHOLD, color="k", ls="--", lw=0.6)
            ax.axhline(0, color="0.6", lw=0.5)
            ax.axvspan(0, event.event_end_s - t0, color=COLORS["Non-Local"], alpha=0.3)
            labels = window.SWR_event.to_numpy() > 0
            if labels.any():
                tt = window.index.to_numpy() - t0
                ax.fill_between(
                    tt, -1, 0, where=labels, color=COLORS["SWR"], alpha=0.6, step="mid"
                )
            ax.set_title(
                f"{animal} {case}\npeak {event.ripple_z_max:.1f}, criterion "
                f"{event.ripple_z_criterion:.1f}, is_swr {bool(event.is_swr)}",
                fontsize=7,
            )
            ax.tick_params(labelsize=6)
    fig.suptitle(
        "Ripple z around spot-check events (green: event; orange bar: SWR rows; "
        "dashed: 2 z); time from event start (s)",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- stage analyze --------------------------------------------------------------------


def run_analyze(args: argparse.Namespace) -> None:
    folder = out_dir()
    figures = folder / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    events, record = load_events()
    print(json.dumps(record["checks"], indent=1), flush=True)

    t = cc.KAY_ZSCORE_THRESHOLD
    above = events.ripple_z_criterion >= t
    labels = (
        events.assign(criterion_above=above, peak_above=events.ripple_z_max >= t)
        .groupby("animal")
        .apply(
            lambda g: pd.Series(
                {
                    "n_events": len(g),
                    "n_swr": int(g.is_swr.sum()),
                    "criterion_agrees": float((g.criterion_above == g.is_swr).mean()),
                    "swr_below_criterion": int((g.is_swr & ~g.criterion_above).sum()),
                    "non_swr_above_criterion": int(
                        (~g.is_swr & g.criterion_above).sum()
                    ),
                    "non_swr_peak_above_fraction": float(
                        g.peak_above[~g.is_swr].mean()
                    ),
                    "swr_peak_below_fraction": float((~g.peak_above[g.is_swr]).mean()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    pooled = {
        "animal": "pooled",
        "n_events": len(events),
        "n_swr": int(events.is_swr.sum()),
        "criterion_agrees": float((above == events.is_swr).mean()),
        "swr_below_criterion": int((events.is_swr & ~above).sum()),
        "non_swr_above_criterion": int((~events.is_swr & above).sum()),
        "non_swr_peak_above_fraction": float(
            (events.ripple_z_max >= t)[~events.is_swr].mean()
        ),
        "swr_peak_below_fraction": float(
            (events.ripple_z_max < t)[events.is_swr].mean()
        ),
    }
    labels = pd.concat([labels, pd.DataFrame([pooled])], ignore_index=True)
    labels.to_csv(folder / "labels.csv", index=False)
    print(labels.to_string(), flush=True)

    distribution = cc.distribution_table(events)
    distribution.to_csv(folder / "distribution.csv", index=False)
    print(distribution.to_string(), flush=True)
    plot_distributions(events, distribution, figures / "ripple_power_distributions.png")
    dist_sim = distribution_simulation(distribution, args.n_dip_reps)
    dist_sim.to_csv(folder / "distribution_simulation.csv", index=False)
    print(
        dist_sim.assign(rejects=dist_sim.dip_p < 0.05)
        .groupby(["scenario", "separation_factor", "power"], dropna=False)
        .rejects.mean()
        .unstack()
        .to_string(),
        flush=True,
    )

    windows = (args.window, *args.sensitivity_windows)
    steps = cc.step_table(events, windows, (args.n_knots, *args.sensitivity_knots))
    steps.to_csv(folder / "step_fits.csv", index=False)
    main = steps[
        (steps.n_knots == args.n_knots)
        & ((steps.power != "criterion") | (steps.window_z == args.window))
    ]
    print(
        main.groupby(["property", "power", "covariate"])
        .agg(
            animals_step=("has_step", "sum"),
            share_median=("step_share", "median"),
            gain_median=("heldout_gain", "median"),
            rows=("rows_fraction", "median"),
        )
        .to_string(),
        flush=True,
    )
    for word in cc.STEP_PROPERTIES:
        plot_property(events, word, args.window, figures / f"step_{word}.png")

    step_sim = step_simulation(args.n_step_reps, args.window, args.n_knots)
    step_sim.to_csv(folder / "step_simulation.csv", index=False)
    print(
        step_sim.groupby(["scenario", "planted_step", "power", "covariate"])
        .agg(
            has_step=("has_step", "mean"),
            covers=("covers_planted", "mean"),
            step=("step", "median"),
            share=("step_share", "median"),
        )
        .to_string(),
        flush=True,
    )

    spots = spot_check_events(events)
    spots.to_csv(folder / "spot_checks.csv", index=False)
    plot_spot_checks(spots, figures / "spot_checks.png")

    manifest = {
        "produced_by": produced_by(__file__),
        "commit": described_commit(),
        "inputs": record["inputs"],
        "checks": record["checks"],
        "settings": {
            "kay_threshold_z": t,
            "grid_minimum_samples": cc.GRID_MINIMUM_SAMPLES,
            "common_support_tail": 0.025,
            "criterion_window_z": args.window,
            "sensitivity_windows_z": list(args.sensitivity_windows),
            "n_knots": args.n_knots,
            "sensitivity_knots": list(args.sensitivity_knots),
            "covariate_knots": 6,
            "n_folds": 5,
            "fold_groups": "epoch",
            "ci": "95 %, clustered by epoch",
            "has_step": "the 95 % interval of the label term excludes 0 and the "
            "label lowers the held-out squared error",
            "remote_cm": REMOTE_CM,
            "n_dip_reps": args.n_dip_reps,
            "n_step_reps": args.n_step_reps,
            "seed": SEED,
            "properties": {
                w: p[0] + f" ({p[2]} events)" for w, p in cc.STEP_PROPERTIES.items()
            },
        },
    }
    write_json(folder / "manifest.json", manifest, indent=1, final_newline=True)
    print(f"wrote {folder}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    measure = sub.add_parser("measure")
    measure.add_argument("--n-workers", type=int, default=1)
    measure.add_argument("--epochs", nargs="*")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--window", type=float, default=1.5)
    analyze.add_argument(
        "--sensitivity-windows", type=float, nargs="*", default=[1.0, 2.5]
    )
    analyze.add_argument("--n-knots", type=int, default=8)
    analyze.add_argument("--sensitivity-knots", type=int, nargs="*", default=[5, 12])
    analyze.add_argument("--n-dip-reps", type=int, default=200)
    analyze.add_argument("--n-step-reps", type=int, default=100)
    args = parser.parse_args(argv)
    if args.stage == "measure":
        run_measure(args)
    else:
        run_analyze(args)


if __name__ == "__main__":
    main()
