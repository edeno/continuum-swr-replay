"""The events summary's spike-evidence values (questions 2c.3, 2d.9): the
spike-evidence run's scores of the events against matched quiet windows, and the
held-out tetrode test split by SWR."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_evidence.spike_evidence import (
    BASELINE_GROUP,
    SPIKE_EVIDENCE_METRICS,
    in_analyzed_subset,
    with_event_ids,
)
from continuum_swr_replay.analysis.event_properties import event_statistics
from continuum_swr_replay.analysis.event_tables.prepare_data import time_mask_staleness
from continuum_swr_replay.paths import NL_BOUTS_BY_SWR_STEM, ROOT_DIR
from continuum_swr_replay.reported.events.common import (
    BOUT_SUBSET,
    SPIKE_EVIDENCE_SUBSET,
    TAGS,
    SummaryReads,
)
from continuum_swr_replay.reported.values import (
    Value,
    file_record,
    files_record,
    over,
    pend,
    rel,
    value_factory,
)

_SPIKE_EVIDENCE_GROUPS = {
    "swr_remote": "SWR-remote",
    "non_swr_remote": "nonSWR-remote",
    "non_swr_close": "nonSWR-close",
}


def spike_evidence_metrics(per_epoch_dir: Path, n_epochs: int) -> pd.DataFrame:
    """The spike-evidence run's per-event metrics, one table per epoch, concatenated.

    Parameters
    ----------
    per_epoch_dir : Path
        The run's folder, with one folder per epoch.
    n_epochs : int
        The epochs the run must hold: it scores every analyzed epoch.

    Raises
    ------
    ValueError
        The run does not hold ``n_epochs`` epochs, or an epoch's table names
        another epoch.
    """
    paths = sorted(per_epoch_dir.glob(f"*/{SPIKE_EVIDENCE_METRICS}"))
    if len(paths) != n_epochs:
        raise ValueError(
            f"{per_epoch_dir} holds {len(paths)} epochs' metrics, not {n_epochs}"
        )
    tables = []
    for path in paths:
        table = pd.read_parquet(path)
        if set(table["epoch_id"]) != {path.parent.name}:
            raise ValueError(f"{path} holds epochs {sorted(set(table['epoch_id']))}")
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def spike_evidence_values(metrics: pd.DataFrame, source: str) -> list[Value]:
    """Questions 2c.3, 2c.4 and 2d.9: each event group's spike evidence against matched
    quiet windows (``baseline``), the negative control. Each median is event ->
    epoch -> animal (each epoch's median, each animal's median over its epochs,
    the mean over animals); the run scores every event and every quiet window with at least one
    spike (no cap; the evidence is per spike, so a quiet window without spikes has
    none)."""
    make = value_factory(spike_evidence_values, source, TAGS["spike"])
    values = [
        make(
            "evidence_epochs",
            metrics["epoch_id"].nunique(),
            "count",
            "count",
            "epochs in the spike-evidence run",
        ),
    ]
    statistics = event_statistics.spike_evidence_statistics(metrics, "baseline")
    groups = {"baseline": "baseline", **_SPIKE_EVIDENCE_GROUPS}
    for word, group in groups.items():
        stats = statistics.loc[group]
        label = "matched quiet windows" if word == "baseline" else f"{group} events"
        values += [
            make(
                f"{word}_scored",
                stats["n_scored"],
                "count",
                "count",
                f"{label} scored",
                tags=TAGS["spike"],
            ),
            make(
                f"{word}_spikes_median",
                stats["spikes_median"],
                "quantity",
                "int",
                f"median spike count of {label}: each epoch's median, each animal's "
                "median over its epochs, the mean over animals",
                "spikes",
                tags=TAGS["spike_count"],
            ),
            make(
                f"{word}_tetrodes_median",
                stats["tetrodes_median"],
                "quantity",
                "int",
                f"median number of tetrodes with a spike in {label}: each epoch's "
                "median, each animal's median over its epochs, the mean over animals",
                "tetrodes",
                tags=TAGS["spike_count"],
            ),
            make(
                f"{word}_dll_per_spike_median",
                stats["dll_per_spike_median"],
                "quantity",
                "dec2",
                f"median log-likelihood ratio per spike, decoded location against the "
                f"animal's, of {label}: each epoch's median, each animal's median over "
                "its epochs, the mean over animals",
                "nats per spike",
                tags=TAGS["spike_evidence"],
            ),
            make(
                f"{word}_gpi_distance_median_cm",
                stats["gpi_distance_median_cm"],
                "quantity",
                "int",
                f"median track distance from the MAP of {label} to the ground-process "
                "intensity peak: each epoch's median, each animal's median over its "
                "epochs, the mean over animals",
                "cm",
                tags=TAGS["spike_gpi"],
            ),
        ]
        if word == "baseline":
            continue
        values += [
            make(
                f"{word}_epochs_above_baseline",
                stats["n_epochs_above_baseline"],
                "count",
                "count",
                f"epochs whose median {group} event scores above the epoch's median quiet "
                "window, per spike",
                tags=TAGS["spike_evidence"],
            ),
            make(
                f"{word}_epochs_scored",
                stats["n_epochs_scored"],
                "count",
                "count",
                f"epochs with a scored {group} event",
                tags=TAGS["spike_evidence"],
            ),
            make(
                f"{word}_animals_above_baseline",
                stats["n_animals_above_baseline"],
                "count",
                "count",
                f"animals whose median {group} event scores above their median quiet "
                "window, per spike, each the median over the animal's epochs of the "
                "epoch's median",
                tags=TAGS["spike_evidence"],
            ),
        ]
    values.append(
        make(
            "evidence_animals",
            metrics["animal"].nunique(),
            "count",
            "count",
            "animals in the spike-evidence run",
        )
    )
    return values


def spike_evidence_visit_values(
    metrics: pd.DataFrame, kept: np.ndarray, source: str
) -> list[Value]:
    """Questions 2c.3 and 2d.9: the scored events and quiet windows of each group
    left out because they start outside every well visit
    (``spike_evidence.in_analyzed_subset``)."""
    make = value_factory(spike_evidence_visit_values, source, TAGS["spike"])
    left_out = metrics.loc[~kept, "group"].value_counts()
    groups = {"baseline": BASELINE_GROUP, **_SPIKE_EVIDENCE_GROUPS}
    return [
        make(
            f"{word}_left_out",
            int(left_out.get(group, 0)),
            "count",
            "count",
            (
                "matched quiet windows scored whose first sample is outside every "
                "well visit, left out"
                if word == "baseline"
                else f"{group} events scored that start outside every well visit, "
                "left out"
            ),
        )
        for word, group in groups.items()
    ]


def heldout_by_swr_values(stats: pd.DataFrame, source: str) -> list[Value]:
    """Question 2c.3: the held-out tetrode test of immobile Non-Local Continuous bouts,
    split by SWR, per held-out spike against the animal's position."""
    make = value_factory(heldout_by_swr_values, source, TAGS["spike_evidence"])
    rows = stats[
        (stats["regime"] == "immobile") & (stats["metric"] == "runcv_nl_anim_mark_ps")
    ]
    values = []
    for word, state in (
        ("swr", "Non-Local Continuous | SWR"),
        ("non_swr", "Non-Local Continuous | no SWR"),
    ):
        (row,) = rows[rows["state"] == state].itertuples()
        positive, total = (int(n) for n in row.animals_pos.split("/"))
        label = "with an SWR" if word == "swr" else "without an SWR"
        values += [
            make(
                f"heldout_{word}_per_spike",
                row.animal_estimate,
                "quantity",
                "dec2",
                f"held-out evidence per held-out spike for immobile Non-Local Continuous "
                f"bouts {label}, against the animal's position: each epoch's median "
                "over its bouts, each animal's median over its epochs, the mean over "
                "animals",
                "nats per spike",
            ),
            make(
                f"heldout_{word}_per_spike_ci_low",
                row.animal_ci_lo,
                "quantity",
                "dec2",
                f"lower bound of its 95 % interval (nested bootstrap of the animals "
                f"and their epochs, small-sample t correction over the {total} "
                "animals)",
                "nats per spike",
            ),
            make(
                f"heldout_{word}_per_spike_ci_high",
                row.animal_ci_hi,
                "quantity",
                "dec2",
                "upper bound of that interval",
                "nats per spike",
            ),
            make(
                f"heldout_{word}_animals_positive",
                positive,
                "count",
                "count",
                "animals with a positive mean",
            ),
            make(
                f"heldout_{word}_animals",
                total,
                "count",
                "count",
                "animals in that test",
            ),
        ]
    return values


def heldout_bout_values(bouts: pd.DataFrame, source: str) -> list[Value]:
    """Question 2c.3: the immobile Non-Local Continuous bouts of the held-out test
    split by SWR that start inside a well visit and are read, those left out, and
    the read ones whose matched window starts outside every visit (the matched
    windows were drawn from all immobile time). ``bouts`` is the split's count
    table (``state_evidence.bout_visit_counts``, recorded in its provenance)."""
    make = value_factory(heldout_bout_values, source, TAGS["spike_evidence"])
    rows = bouts[bouts["regime"] == "immobile"].set_index("state")
    values = []
    for word, state in (
        ("swr", "Non-Local Continuous | SWR"),
        ("non_swr", "Non-Local Continuous | no SWR"),
    ):
        row = rows.loc[state]
        label = "with an SWR" if word == "swr" else "without an SWR"
        values += [
            make(
                f"heldout_{word}_bouts_in_subset",
                row["n_kept"],
                "count",
                "count",
                f"immobile Non-Local Continuous bouts {label} whose first sample is "
                "inside a well visit, read",
            ),
            make(
                f"heldout_{word}_bouts_left_out",
                row["n_left_out"],
                "count",
                "count",
                f"immobile Non-Local Continuous bouts {label} whose first sample is "
                "outside every well visit, left out",
            ),
            make(
                f"heldout_{word}_bouts_matched_outside_visit",
                row["n_kept_matched_outside_visit"],
                "count",
                "count",
                f"of the read bouts {label}, those whose matched window starts "
                "outside every well visit",
            ),
        ]
    return values


def add_spike_evidence(reads: SummaryReads) -> dict[str, object]:
    """Add the spike-evidence values and the held-out test split by SWR; return
    the spike-evidence run's record, which the class-step values check."""
    summary = reads.summary
    inputs = reads.inputs
    table = reads.table
    events = reads.events
    visits = reads.visits
    n_epochs = reads.n_epochs
    # the run scores every event: each scored event takes its event_id, and the
    # events the selection keeps and the quiet windows in a visit are read
    metrics = with_event_ids(
        spike_evidence_metrics(inputs.spike_evidence_dir, n_epochs), table
    )
    in_subset = in_analyzed_subset(metrics, events["event_id"], visits)
    evidence_source = (
        f"{rel(inputs.spike_evidence_dir)}/*/{SPIKE_EVIDENCE_METRICS} (every event, "
        "no cap), each scored event matched to the event table by epoch and start, "
        "with each epoch's well visits"
    )
    summary.add(
        *over(
            spike_evidence_values(metrics[in_subset], evidence_source),
            SPIKE_EVIDENCE_SUBSET,
        )
    )
    summary.add(*spike_evidence_visit_values(metrics, in_subset, evidence_source))
    spike_evidence_record = files_record(
        SPIKE_EVIDENCE_METRICS,
        inputs.spike_evidence_dir.glob(f"*/{SPIKE_EVIDENCE_METRICS}"),
        ROOT_DIR,
    )
    summary.inputs.append(spike_evidence_record)

    stats = pd.read_csv(inputs.heldout_by_swr)
    heldout_provenance_path = inputs.heldout_by_swr.with_name(
        f"{NL_BOUTS_BY_SWR_STEM}.provenance.json"
    )
    heldout_provenance = json.loads(heldout_provenance_path.read_text())
    heldout_off_subset = time_mask_staleness(
        heldout_provenance, heldout_provenance["epochs"], rel(heldout_provenance_path)
    )
    summary.add(
        *over(
            pend(
                heldout_by_swr_values(stats, rel(inputs.heldout_by_swr)),
                heldout_off_subset,
            ),
            BOUT_SUBSET,
        )
    )
    if heldout_off_subset is None:
        summary.add(
            *heldout_bout_values(
                pd.DataFrame(heldout_provenance["bouts"]), rel(heldout_provenance_path)
            )
        )
    summary.inputs += [
        file_record(inputs.heldout_by_swr, ROOT_DIR),
        file_record(heldout_provenance_path, ROOT_DIR),
    ]
    return spike_evidence_record
