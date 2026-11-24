"""What the parts of the events summary share: the tags of its values, the
subsets their descriptions name, the reads every part takes
(:class:`SummaryReads`), and the helpers several parts' builders call."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    subset_staleness,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    SWR_LABELS,
)
from continuum_swr_replay.analysis.event_tables.prepare_data import (
    session_cache_staleness,
    time_mask_staleness,
)
from continuum_swr_replay.data_loaders.epoch_register import require_analyzed_epochs
from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.inputs import EventsInputs, event_table_staleness
from continuum_swr_replay.reported.values import (
    Summary,
    Tags,
    Value,
    file_record,
    over,
    pend,
    rel,
)
from continuum_swr_replay.runs.staleness import read_output_manifest

TAGS = {
    "decoding_error": Tags(("2a.1",)),
    "calibration": Tags(("2a.1",), ("S2d",)),
    "nonlocal_probability": Tags(("2a.6",), ("S2c",)),
    "state_fractions": Tags(("2a.6",), ("decoder_validation",)),
    "state_fractions_immobile": Tags(("2a.6",), ("2a",)),
    "cohort": Tags(("2b.2",), ("2d",)),
    "cohort_pooled": Tags(("2b.2",)),
    "visit_context": Tags(("2b.2",)),
    "left_out": Tags(("2b.2",)),
    "other_detectors_main": Tags(("2b.3",), ("2",)),
    "other_detectors": Tags(("2b.3",)),
    "nonlocal_found": Tags(("2b.3",), ("2e",)),
    "swr_content": Tags(("2b.3",), ("2c",)),
    "boundaries": Tags(("2b.4",), ("2f",)),
    "boundary_timing": Tags(("2b.4",), ("S2f",)),
    "boundary_timing_both": Tags(("2b.4",)),
    "boundary_timing_pooled": Tags(("2b.4",)),
    "ripple": Tags(("2d.8", "2b.5"), ("2g",)),
    "mua": Tags(("2d.8",), ("2g",)),
    "theta": Tags(("2d.8",), ("2g",)),
    "gamma_z": Tags(("2d.8",), ("S3",)),
    "brain_state": Tags(("2d.8",)),
    "class_distribution": Tags(("2b.5",)),
    "class_step": Tags(("2b.5", "2d.2", "2d.3", "2d.5", "2d.6")),
    "class_step_evidence": Tags(("2b.5",)),
    "nonlocal_power": Tags(("2b.5",), ("2i",)),
    "nonlocal_mua": Tags(("2b.5",), ("S2g",)),
    "state_by_signal": Tags(("2b.5",)),
    "rule_sweep": Tags(("2a.5",), ("S2e",)),
    "rule_sweep_pooled": Tags(("2a.5",)),
    "state_calibration": Tags(("2a.7",), ("S2a",)),
    "participation": Tags(("2d.9",), ("2h",)),
    "spike_rate": Tags(("2d.9",)),
    "state_composition": Tags(("2d.10",), ("3b",)),
    "likelihood_advantage": Tags(("2c.3",)),
    "tetrode_holdout": Tags(("2c.3",), ("2j", "S2h")),
    "tetrode_holdout_split": Tags(("2c.3",), ("S2h",)),
    "tetrode_holdout_text": Tags(("2c.3",)),
    "tetrode_holdout_extra_figure": Tags(("2c.3",), ("2j",)),
    "nonlocal_restriction": Tags(("2c.3",)),
    "nonlocal_restriction_figure": Tags(("2c.3",), ("2k",)),
    "shared_content": Tags(("2c.3",)),
    "shared_content_raw": Tags(("2c.3",)),
    "shared_content_text": Tags(("2c.3",)),
    "nonlocal_restriction_distance": Tags(("2c.3",), ("S2i",)),
    "nonlocal_restriction_swr_allowed": Tags(("2c.3",), ("S2l",)),
    "spike": Tags(("2d.9", "2c.3")),
    "spike_count": Tags(("2d.9",), ("2",)),
    "spike_evidence": Tags(("2c.3",), ("2",)),
    "spike_gpi": Tags(("2c.4",)),
    "content_occupancy": Tags(("2d.1",), ("3f",)),
    "content_by_animal": Tags(("2d.1",)),
    "content_well_side": Tags(("2d.1",), ("S3c",)),
    "duration": Tags(("2d.6",), ("3a",)),
    "extent": Tags(("2d.3",), ("3c",)),
    "arms": Tags(("2d.3",)),
    "speed": Tags(("2d.2",), ("3d",)),
    "transport_speed": Tags(("2d.2",), ("S3b",)),
    "certainty": Tags(("2d.5",)),
    "hpd": Tags(("2d.5",), ("S2k",)),
    "content_distance": Tags(("2d.1",), ("3g",)),
    "content_distance_pooled": Tags(("2d.1",)),
    "stillness": Tags(("2d.4",)),
    "location": Tags(("2d.1",)),
}
"""Each group of values: the questions it answers and the figures that show it."""


EVENTS_SUBSET = (
    "the immobile non-local events that start inside a well visit (from the first "
    "poke-in at a well to the final poke-out; visit_context in_visit or "
    "spans_visit_end)"
)
"""What the values of the events are over, as their descriptions say."""


HELDOUT_EVENTS_SUBSET = (
    "the immobile non-local events detected from the held-out decode (each fold's "
    "bins under the decoder fit on the other folds, by the canonical event rule; "
    "a small share span two folds, each bin still under a fit that did not see "
    "it) "
    "that start inside a well visit (their first bin is inside one, from the first "
    "poke-in at a well to the final poke-out)"
)
"""What the values of the events detected from the held-out decode are over."""


CONTROL_SUBSET = (
    f"{EVENTS_SUBSET}, against Local-rest control windows of each event's length "
    "lying wholly inside one well visit"
)
"""What the participation values are over: their controls are Local rest inside
one well visit (``event_evidence.participation.CONTROL_RULE``)."""


CANONICAL_WINDOWS_SUBSET = (
    "the canonical events, chosen from the decode fit on the whole epoch (the "
    "scores are held out; the choice of events is not; Figure 2k's values use "
    f"the events detected from the held-out decode instead): {CONTROL_SUBSET}"
)
"""What the restriction values over the canonical windows are over
(:func:`restriction_window_values`)."""


TIME_SUBSET = (
    "time inside a well visit (from the first poke-in at a well to the final poke-out)"
)
"""What the values of time (rest, immobile bins) are over."""


DETECTOR_SUBSET = (
    "the other detectors' events whose first sample is inside a well visit and "
    "the non-local events that start inside one, each selected by its own start; "
    "whether an event overlaps an event of the other side is read against all of "
    "that side's events, so its partner may start outside a visit"
)
"""What the values of the detector comparison are over."""


LEFT_OUT_SUBSET = (
    f"{EVENTS_SUBSET}, and the non-local detector's exposure inside a well visit; "
    "an event is left out when no event of the other inventory overlaps it, "
    "wherever that event starts"
)
"""What the values of what another inventory leaves out are over."""


SPIKE_EVIDENCE_SUBSET = (
    f"{EVENTS_SUBSET}, against the matched quiet windows whose first sample is "
    "inside a well visit"
)
"""What the values of the spike-evidence run are over."""


BOUT_SUBSET = (
    "the decoder's immobile Non-Local Continuous bouts whose first sample is inside "
    "a well visit (from the first poke-in at a well to the final poke-out); the "
    "bouts' matched windows are drawn from all of the epoch's time in the regime, "
    "and the values reported here do not use them"
)
"""What the values of the held-out test split by SWR are over."""


#: The event groups of the per-event runs, by the word their keys use.
_RUN_GROUPS = {"non_swr": SWR_LABELS[False], "swr": SWR_LABELS[True]}


def _per_spike_interval(
    make: Callable[..., Value],
    key: str,
    row: pd.Series,
    description: str,
    style: str = "dec2",
    unit: str = "nats per spike",
) -> list[Value]:
    """A per-animal estimate, per spike unless ``unit`` says otherwise, and the
    bounds of its 95 % interval (``likelihood_advantage_statistics``'
    ``estimate``, ``ci_low``, ``ci_high``)."""
    return [
        make(key, row["estimate"], "quantity", style, description, unit),
        make(
            f"{key}_ci_low",
            row["ci_low"],
            "quantity",
            style,
            "lower bound of its 95 % interval (nested bootstrap of the animals "
            "and their epochs, small-sample t correction over the "
            f"{row['n_animals']} animals)",
            unit,
        ),
        make(
            f"{key}_ci_high",
            row["ci_high"],
            "quantity",
            style,
            "upper bound of that interval",
            unit,
        ),
    ]


def _interval_and_animals_positive(
    make: Callable[..., Value],
    key: str,
    row: pd.Series,
    description: str,
    unit: str = "nats per spike",
) -> list[Value]:
    """:func:`_per_spike_interval` to two significant digits, and the animals
    whose value is above zero (``n_animals_positive``)."""
    return [
        *_per_spike_interval(make, key, row, description, style="sig2", unit=unit),
        make(
            f"{key}_animals_positive",
            row["n_animals_positive"],
            "count",
            "count",
            "animals whose value is above zero",
        ),
    ]


def _animal_range_values(
    make: Callable[..., Value],
    key: str,
    shares: tuple[float, float, float],
    descriptions: tuple[str, str, str],
    lowest_style: str = "pctnum",
    *,
    animal_mean: tuple[float, str] | None = None,
    pooled_tags: Tags | None = None,
    style: str = "pct",
    kind: str = "fraction",
    unit: str = "",
) -> list[Value]:
    """A share, then the lowest and the highest animal's (``{key}_animal_min``,
    ``{key}_animal_max``): ``shares`` and ``descriptions`` in that order, each
    printed in ``style`` (a percent; the lowest in ``lowest_style``), of
    ``kind`` and in ``unit``.

    With ``animal_mean`` (the mean over animals of the animals' shares and its
    description), ``{key}_animal_mean`` follows the share; ``pooled_tags``
    then replaces the share's tags (a headline share that no panel draws)."""
    values = [
        make(key_, share, kind, style_, description, unit)
        for key_, share, style_, description in zip(
            (key, f"{key}_animal_min", f"{key}_animal_max"),
            shares,
            (style, lowest_style, style),
            descriptions,
            strict=True,
        )
    ]
    if animal_mean is None:
        return values
    values[0] = make(
        key, shares[0], "fraction", "pct", descriptions[0], tags=pooled_tags
    )
    mean, description = animal_mean
    return [
        values[0],
        make(f"{key}_animal_mean", mean, "fraction", "pct", description),
        *values[1:],
    ]


def _share_interval(
    make: Callable[..., Value], key: str, row: pd.Series, description: str
) -> list[Value]:
    """A share and the bounds of its 95 % interval (``estimate``, ``ci_low``,
    ``ci_high``)."""
    return [
        make(key, row["estimate"], "fraction", "pct", description),
        make(
            f"{key}_ci_low",
            row["ci_low"],
            "fraction",
            "pct",
            "lower bound of its 95 % interval",
        ),
        make(
            f"{key}_ci_high",
            row["ci_high"],
            "fraction",
            "pct",
            "upper bound of that interval",
        ),
    ]


@dataclass(frozen=True)
class SummaryReads:
    """What every part of the events summary reads, and the summary it adds to.

    :func:`~continuum_swr_replay.reported.events.summarize` reads these once;
    each part adds its values and its inputs to ``summary``.
    """

    summary: Summary
    inputs: EventsInputs
    table: pd.DataFrame
    """The canonical event table, every event."""
    events: pd.DataFrame
    """The events that start inside a well visit, the one selection."""
    run: Path
    """The event features run."""
    features_manifest: dict
    """The event features run's manifest."""
    run_epochs: list[str]
    """The epochs that manifest marks ok."""
    visits: dict
    """Each of those epochs' well visits."""
    event_table_record: dict
    """The event table's file record."""
    current_sha256: str
    """The event table's SHA-256."""
    n_epochs: int
    """How many epochs the register analyzes: every per-epoch input holds them."""
    source: str
    """The values' source when they are read from the event table."""

    def stale(self, folder: Path) -> str | None:
        """Why an output was not made from the current event table, or None."""
        return event_table_staleness(
            read_output_manifest(folder), self.current_sha256, rel(folder)
        )

    def off_subset(self, folder: Path) -> str | None:
        """Why an output is not over the subset, or None."""
        return subset_staleness(read_output_manifest(folder), rel(folder))

    def off_visits(self, folder: Path, epoch_ids: Iterable[str]) -> str | None:
        """Why a time-masked output was not made from the current visits."""
        return session_cache_staleness(
            read_output_manifest(folder), epoch_ids, rel(folder)
        )

    def off_time_mask(self, folder: Path, epoch_ids: Iterable[str]) -> str | None:
        """Why a time-masked output is not over the subset and the current visits."""
        return time_mask_staleness(read_output_manifest(folder), epoch_ids, rel(folder))

    def add_counts(
        self,
        folder: Path,
        made_by: str,
        builders: Sequence[Callable[[pd.DataFrame, str], list[Value]]],
        subset: str,
        *notes: str | None,
    ) -> None:
        """The values of a per-epoch counts run (``counts.parquet`` in
        ``folder``, of every analyzed epoch): each builder's, pending on
        ``notes`` and on the run not being over the subset and the current
        visits, over ``subset``; and the run's files as inputs."""
        counts = pd.read_parquet(folder / "counts.parquet")
        require_analyzed_epochs(counts["epoch_id"], str(folder / "counts.parquet"))
        for builder_values in builders:
            self.summary.add(
                *over(
                    pend(
                        builder_values(counts, f"{rel(folder)} ({made_by})"),
                        *notes,
                        self.off_time_mask(folder, counts["epoch_id"].unique()),
                    ),
                    subset,
                )
            )
        self.summary.inputs += [
            file_record(folder / name, ROOT_DIR)
            for name in ("counts.parquet", "manifest.json")
        ]
