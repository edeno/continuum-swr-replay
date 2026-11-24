"""The events summary's signal values (question 2d.8): theta, ripple power, MUA
and gamma during events against quiet rest. The statistics are computed in
``analysis.event_properties.event_statistics``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_properties import (
    event_statistics,
    gamma_power,
    triggered_analysis,
)
from continuum_swr_replay.parameters import SAMPLING_FREQUENCY
from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.events.common import (
    EVENTS_SUBSET,
    TAGS,
    TIME_SUBSET,
    SummaryReads,
)
from continuum_swr_replay.reported.inputs import EventsInputs
from continuum_swr_replay.reported.values import (
    Value,
    file_record,
    files_record,
    over,
    pend,
    rel,
    value_factory,
)
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

_SIGNAL_EVENT_COLUMNS = {
    "theta": "theta_z",
    "ripple": "ripple_power_z",
    "mua": "mua_rate_z",
    "slow_gamma": "slow_gamma_z",
    "fast_gamma": "fast_gamma_z",
}


_SIGNAL_LABELS = {
    "theta": "theta power",
    "ripple": "ripple power",
    "mua": "MUA rate",
    "slow_gamma": "slow-gamma power (25-55 Hz)",
    "fast_gamma": "fast-gamma power (65-100 Hz)",
}


def brain_state_values(
    events: pd.DataFrame, reference: pd.DataFrame, source: str
) -> list[Value]:
    """Questions 2d.8 and 2b.5: theta, ripple power and MUA during events, against
    running and quiet rest.

    Each event's value is its mean over its bins of the epoch's z-scored signal
    (the event table's ``theta_z``, ``ripple_power_z``, ``mua_rate_z``); the
    references are each epoch's median during running (>10 cm/s) and quiet rest
    (<4 cm/s, in no event), on the same scale (``brain_state_reference``). An
    event is theta-like when its theta reaches its epoch's running median.
    """
    makes = {
        word: value_factory(
            brain_state_values,
            source,
            {"theta": TAGS["theta"], "ripple": TAGS["ripple"], "mua": TAGS["mua"]}[
                word
            ],
        )
        for word in ("theta", "ripple", "mua")
    }
    return _signal_values(makes, events, reference)


def gamma_z_values(
    events: pd.DataFrame, reference: pd.DataFrame, source: str
) -> list[Value]:
    """Question 2d.8: slow and fast gamma during events, against quiet-rest and
    running windows as long as the events.

    On the gamma pass's ``slow_gamma_z`` and ``fast_gamma_z``, joined on
    ``event_id``. An event's value is the mean over its bins, and band power is
    skewed, so it is compared with the level of windows of the same length
    (``gamma_power.matched_level``), each class with its own: not with the
    median of single bins, which a window's mean exceeds when nothing differs.
    """
    make = value_factory(gamma_z_values, source, TAGS["gamma_z"])
    words = ["slow_gamma", "fast_gamma"]
    levels = event_statistics.matched_reference_medians(reference, words)
    statistics = event_statistics.event_signal_statistics(
        events,
        reference,
        {word: _SIGNAL_EVENT_COLUMNS[word] for word in words},
        matched=True,
    )
    values: list[Value] = []
    for word in words:
        label = _SIGNAL_LABELS[word]
        for group, is_swr in (("non_swr", False), ("swr", True)):
            stats = statistics.loc[(word, is_swr)]
            name = "non-SWR" if group == "non_swr" else "SWR-overlapping"
            animal_value = (
                f"animals whose median {name} event {label} (the median over their "
                "epochs of each epoch's median)"
            )
            values.append(
                make(
                    f"{word}_median_{group}",
                    stats["median"],
                    "quantity",
                    "dec2",
                    f"median z-scored {label} of immobile {name} non-local events (each "
                    "event's mean over its bins; each epoch's median, each animal's "
                    "median over its epochs, the mean over animals)",
                    "z",
                )
            )
            for regime, words_of_regime in (
                ("rest", "quiet-rest"),
                ("running", "running"),
            ):
                values.append(
                    make(
                        f"{word}_{regime}_matched_reference_{group}",
                        levels.loc[(word, is_swr), regime],
                        "quantity",
                        "dec2",
                        "mean over animals of each animal's median over its epochs "
                        f"(those with such events) of the epoch's z-scored {label} in "
                        f"{words_of_regime} windows as long as its {name} events (the "
                        "median over the events of one window's mean each, averaged "
                        "over the draws)",
                        "z",
                    )
                )
            values += [
                make(
                    f"{word}_animals_below_running_{group}",
                    stats["n_animals_below_running"],
                    "count",
                    "count",
                    f"{animal_value} is below the median over the same epochs of the "
                    "epoch's level in running windows as long as those events",
                ),
                make(
                    f"{word}_animals_above_rest_{group}",
                    stats["n_animals_above_rest"],
                    "count",
                    "count",
                    f"{animal_value} is above the median over the same epochs of the "
                    "epoch's level in quiet-rest windows as long as those events",
                ),
            ]
    return values


def _signal_values(
    makes: dict[str, Callable[..., Value]],
    events: pd.DataFrame,
    reference: pd.DataFrame,
) -> list[Value]:
    """The values of each signal in ``makes``, made by its maker: its running and
    rest references, and each class's median and animal counts (and, for theta,
    the theta-like share)."""
    words = list(makes)
    levels = event_statistics.reference_medians(reference, words)
    statistics = event_statistics.event_signal_statistics(
        events, reference, {word: _SIGNAL_EVENT_COLUMNS[word] for word in words}
    )
    values: list[Value] = []
    for word, make in makes.items():
        label = _SIGNAL_LABELS[word]
        values += [
            make(
                f"{word}_running_reference",
                levels.loc[word, "running"],
                "quantity",
                "dec2",
                f"mean over animals of each animal's median over its epochs of the "
                f"epoch's median z-scored {label} while running",
                "z",
            ),
            make(
                f"{word}_rest_reference",
                levels.loc[word, "rest"],
                "quantity",
                "dec2",
                f"mean over animals of each animal's median over its epochs of the "
                f"epoch's median z-scored {label} in quiet rest",
                "z",
            ),
        ]
        for group, is_swr in (("non_swr", False), ("swr", True)):
            stats = statistics.loc[(word, is_swr)]
            name = "non-SWR" if group == "non_swr" else "SWR-overlapping"
            values += [
                make(
                    f"{word}_median_{group}",
                    stats["median"],
                    "quantity",
                    "dec2",
                    f"median z-scored {label} of immobile {name} non-local events (each "
                    "event's mean over its bins; each epoch's median, each animal's "
                    "median over its epochs, the mean over animals)",
                    "z",
                ),
                make(
                    f"{word}_animals_below_running_{group}",
                    stats["n_animals_below_running"],
                    "count",
                    "count",
                    f"animals whose median {name} event {label} (the median over their "
                    "epochs of each epoch's median) is below the median over the same "
                    "epochs of the epoch's median running level",
                ),
                make(
                    f"{word}_animals_above_rest_{group}",
                    stats["n_animals_above_rest"],
                    "count",
                    "count",
                    f"animals whose median {name} event {label} (the median over their "
                    "epochs of each epoch's median) is above the median over the same "
                    "epochs of the epoch's median quiet-rest level",
                ),
            ]
            if word == "theta":
                values.append(
                    make(
                        f"theta_like_fraction_{group}",
                        stats["reaches_running_fraction"],
                        "fraction",
                        "pct",
                        f"share of immobile {name} non-local events whose theta power "
                        "reaches their epoch's median running level: each epoch's "
                        "share, each animal's mean over its epochs, the mean over "
                        "animals",
                    )
                )
    return values


def gamma_db_values(reference: pd.DataFrame, source: str) -> list[Value]:
    """Question 2d.8: slow and fast gamma during events, as a change in band power
    against the same epoch's quiet rest, in dB.

    For each epoch and class: ``10 log10(mean band power over the class's event
    samples / mean band power over quiet rest)`` (below 4 cm/s, in no non-local,
    SWR or MUA event, outside excluded time), on the same band power the z
    values are made from (``scripts/event_properties/measure_event_gamma.py``). Each
    animal's median over its epochs, the mean over animals, and the animals
    whose median over their epochs is above 0 dB. The z values (``gamma_z_values``) stay as they are.
    """
    make = value_factory(gamma_db_values, source, TAGS["brain_state"])
    bands = ["slow_gamma", "fast_gamma"]
    statistics = event_statistics.power_change_statistics(reference, bands)
    values: list[Value] = []
    for word in bands:
        label = _SIGNAL_LABELS[word]
        for group, is_swr in (("non_swr", False), ("swr", True)):
            name = "non-SWR" if group == "non_swr" else "SWR-overlapping"
            stats = statistics.loc[(word, is_swr)]
            values += [
                make(
                    f"{word}_db_median_{group}",
                    stats["median"],
                    "quantity",
                    "sig2s",
                    f"change in {label} during {name} events against the epoch's quiet "
                    "rest (10 log10 of the ratio of mean band powers): each animal's "
                    "median over its epochs, then the mean over animals",
                    "dB",
                ),
                make(
                    f"{word}_db_animals_above_rest_{group}",
                    stats["n_animals_above_rest"],
                    "count",
                    "count",
                    f"animals whose median over epochs of that change ({name} events) "
                    "is above 0 dB",
                ),
            ]
    return values


def gamma_triggered_curves(
    events: pd.DataFrame, folder: Path, n_epochs: int
) -> tuple[dict[str, dict], list[Path]]:
    """Slow and fast gamma around the starts of the events of ``events``, by
    kind (``triggered_analysis.kind_triggered_curves``, keyed by band), from
    the gamma pass's per-bin z-scored power
    (``<epoch>_gamma_z.parquet``, on the decode's time grid), over
    ``triggered_analysis.TRIGGERED_WINDOW_S``; and the files read.

    Raises
    ------
    ValueError
        If ``folder`` does not hold ``n_epochs`` epochs' traces, or an event's
        epoch has none.
    """
    paths = sorted(folder.glob("*_gamma_z.parquet"))
    if len(paths) != n_epochs:
        raise ValueError(f"{folder} holds {len(paths)} epochs' gamma, not {n_epochs}")
    n_each_side = int(
        round(triggered_analysis.TRIGGERED_WINDOW_S / 2 * SAMPLING_FREQUENCY)
    )
    # the gamma pass's own table names each epoch's file
    keys = ["animal", "session_date", "epoch"]
    named = pd.read_parquet(folder / "reference.parquet", columns=["epoch_id", *keys])
    epoch_id = events[keys].merge(named, on=keys, how="left", validate="m:1")[
        "epoch_id"
    ]
    if epoch_id.isna().any():
        raise ValueError(f"{folder} has no gamma for an event's epoch")
    by_path = {path.name.removesuffix("_gamma_z.parquet"): path for path in paths}
    if not set(epoch_id).issubset(by_path):
        raise ValueError(f"{folder} has no gamma for an event's epoch")
    windows: dict[str, list] = {band: [] for band in gamma_power.GAMMA_BANDS}
    kept = []
    for epoch, mine in events.groupby(epoch_id.to_numpy()):
        trace = pd.read_parquet(by_path[epoch])
        for band in windows:
            windows[band].append(
                triggered_analysis.windows_at(
                    trace["time"].to_numpy(),
                    trace[f"{band}_z"].to_numpy(),
                    mine["event_start_s"].to_numpy(),
                    n_each_side,
                )
            )
        kept.append(mine)
    kept = pd.concat(kept)
    labels = triggered_analysis.epoch_labels(kept["session_date"], kept["epoch"])
    return {
        band: triggered_analysis.kind_triggered_curves(
            np.concatenate(windows[band]),
            kept["is_swr"].to_numpy(dtype=bool),
            kept["animal"].to_numpy(),
            labels,
        )
        for band in windows
    }, paths


def with_brain_state(
    events: pd.DataFrame, inputs: EventsInputs, n_epochs: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``events`` with their slow and fast gamma, and the brain-state reference
    with the gamma reference's columns joined; also the gamma reference itself.

    Raises
    ------
    ValueError
        If a reference does not hold ``n_epochs`` epochs, or an event has no
        gamma value.
    """
    reference = pd.read_parquet(inputs.brain_state_reference)
    gamma_reference = pd.read_parquet(inputs.event_gamma / "reference.parquet")
    if len(reference) != n_epochs or len(gamma_reference) != n_epochs:
        raise ValueError(
            f"the brain-state reference holds {len(reference)} epochs and the gamma "
            f"reference {len(gamma_reference)}, not {n_epochs}"
        )
    gamma_columns = [c for c in gamma_reference.columns if "gamma" in c]
    reference = reference.merge(
        gamma_reference[[*SESSION_COLUMNS, *gamma_columns]],
        on=list(SESSION_COLUMNS),
        how="inner",
        validate="1:1",
    )
    gamma_events = pd.concat(
        pd.read_parquet(path, columns=["event_id", "slow_gamma_z", "fast_gamma_z"])
        for path in sorted(inputs.event_gamma.glob("*_events.parquet"))
    )
    joined = events.merge(gamma_events, on="event_id", how="left", validate="1:1")
    if joined[["slow_gamma_z", "fast_gamma_z"]].isna().all(axis=1).any():
        raise ValueError("an event has no gamma value")
    return joined, reference, gamma_reference


def add_signals(reads: SummaryReads) -> None:
    """Add the brain-state levels during events and the gamma values."""
    summary = reads.summary
    inputs = reads.inputs
    events = reads.events
    n_epochs = reads.n_epochs
    stale = reads.stale
    off_subset = reads.off_subset
    off_visits = reads.off_visits
    off_time_mask = reads.off_time_mask
    events_with_gamma, reference, gamma_reference = with_brain_state(
        events, inputs, n_epochs
    )
    signal_source = (
        f"{rel(inputs.event_table)} (event theta_z, ripple_power_z, mua_rate_z) "
        f"and {rel(inputs.brain_state_reference)} (scripts/event_properties/"
        "measure_brain_state_reference.py) and the gamma pass "
        f"{rel(inputs.event_gamma)} (scripts/event_properties/measure_event_gamma.py)"
    )
    against_rest = f"{EVENTS_SUBSET}, against quiet rest in {TIME_SUBSET}"
    summary.add(
        *over(
            pend(
                brain_state_values(events_with_gamma, reference, signal_source),
                off_time_mask(
                    inputs.brain_state_reference.parent, reference["epoch_id"]
                ),
            ),
            against_rest,
        )
    )
    summary.add(
        *over(
            pend(
                gamma_z_values(events_with_gamma, reference, signal_source),
                stale(inputs.event_gamma),
                off_subset(inputs.brain_state_reference.parent),
                off_subset(inputs.event_gamma),
                off_visits(inputs.brain_state_reference.parent, reference["epoch_id"]),
                off_visits(inputs.event_gamma, gamma_reference["epoch_id"]),
            ),
            against_rest,
        )
    )
    summary.inputs += [
        file_record(inputs.brain_state_reference, ROOT_DIR),
        file_record(inputs.event_gamma / "manifest.json", ROOT_DIR),
    ]
    summary.add(
        *over(
            pend(
                gamma_db_values(
                    gamma_reference,
                    f"{rel(inputs.event_gamma)}/reference.parquet "
                    "(scripts/event_properties/measure_event_gamma.py)",
                ),
                stale(inputs.event_gamma),
                off_time_mask(inputs.event_gamma, gamma_reference["epoch_id"]),
            ),
            against_rest,
        )
    )
    summary.inputs.append(
        file_record(inputs.event_gamma / "reference.parquet", ROOT_DIR)
    )
    summary.inputs.append(
        files_record(
            "*_events.parquet", inputs.event_gamma.glob("*_events.parquet"), ROOT_DIR
        )
    )
