"""How the count of non-local events and the share with no SWR depend on the
event rule's settings: a sensitivity check, not a change to the events.

Three settings are swept, one at a time, the others at the rule's values:

- the non-local probability an event stays strictly above
  (:data:`PROBABILITY_THRESHOLDS`, the rule's 0.8 among them);
- the minimum run length in decode samples (:data:`MINIMUM_SAMPLES`, the
  rule's 7 among them: its 15 ms rounds to 7 samples on the 2 ms grid);
- the SWR detector's threshold (:data:`SWR_THRESHOLDS_Z`, the Kay detector's 2 z among
  them).

For the first two, :func:`epoch_sweep_counts` detects the events again from a
decode frame's stored non-local probability with the rule itself
(``nonlocal_event_rule.detect_nonlocal_events``), holds them to the one
analyzed-time rule and labels their SWR overlap as the canonical table does
(``analyzed_time.with_swr_source``, then
``classified_table.build_classified_nonlocal_event_table``: strict positive
overlap with an SWR of the source), and counts those that start inside a well
visit. At the rule's own values these are the canonical events.

The SWRs are not detected again at another threshold: the detector ran on the
LFP and is not stored to re-run. Instead, for the events at the rule's own
values, each event's Kay criterion on the decode grid
(``class_or_continuum.kay_criterion`` of ``SWR_zscore``, the highest level the
ripple power sustains for the detector's minimum duration in the event or in
any run of positive power it overlaps) says whether it would overlap a Kay-like
SWR at a threshold: exactly when the criterion is at or above it. The
criterion's label at 2 z is the canonical one for about 99 % of the events (the
one-class-or-two analysis's ``criterion_agrees``); :func:`epoch_sweep_counts`
counts how often it agrees, so the stand-in's faithfulness is reported with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    with_swr_source,
)
from continuum_swr_replay.analysis.event_definition.classified_table import (
    build_classified_nonlocal_event_table,
)
from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    detect_nonlocal_events,
    nonlocal_probability,
)
from continuum_swr_replay.analysis.event_definition.well_visits import in_well_visit
from continuum_swr_replay.analysis.event_properties.class_or_continuum import (
    KAY_ZSCORE_THRESHOLD,
    kay_criterion,
)
from continuum_swr_replay.analysis.shared_statistics import animal_session_values
from continuum_swr_replay.parameters import (
    DECODE_SAMPLE_S,
    DEFAULT_NON_LOCAL_THRESHOLD,
)

PROBABILITY_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
"""The non-local probabilities an event stays strictly above."""
MINIMUM_SAMPLES = (7, 10, 15, 25, 38, 50)
"""The minimum run lengths, in 2 ms decode samples: 7 is the rule's 15 ms, whose
shortest event lasts 14 ms (7 samples of 2 ms); 25 and 50 are 50 and 100 ms."""
SWR_THRESHOLDS_Z = tuple(float(z) for z in np.arange(0.0, 6.01, 0.5))
"""The SWR thresholds, z of the Kay consensus ripple power: 0 to 6 z in steps
of 0.5, so that the series spans the events' peak ripple power it is drawn
beside."""
RULE_MINIMUM_SAMPLES = 7
"""The rule's minimum, ``NON_LOCAL_MINIMUM_DURATION_S`` (15 ms) in samples."""
RULE = {
    "probability_threshold": DEFAULT_NON_LOCAL_THRESHOLD,
    "minimum_samples": RULE_MINIMUM_SAMPLES,
    "swr_threshold_z": KAY_ZSCORE_THRESHOLD,
}
"""Each swept setting's value in the rule."""
SETTINGS = {
    "probability_threshold": PROBABILITY_THRESHOLDS,
    "minimum_samples": MINIMUM_SAMPLES,
    "swr_threshold_z": SWR_THRESHOLDS_Z,
}
"""Each swept setting and its values."""

assert all(RULE[name] in values for name, values in SETTINGS.items())


def detect_classified_events(
    raw: pd.DataFrame, probability_threshold: float, minimum_samples: int
) -> pd.DataFrame:
    """One epoch's non-local events at other rule settings, labeled by SWR
    overlap as the canonical table labels them.

    Parameters
    ----------
    raw : pd.DataFrame
        The epoch's decode frame as stored (not yet read through the SWR
        source), with ``head_speed`` and the two non-local state probabilities.
    probability_threshold : float
    minimum_samples : int
        2 ms decode samples.

    Returns
    -------
    pd.DataFrame
        ``build_classified_nonlocal_event_table``'s rows of the events kept by
        the analyzed-time rule.
    """
    time = raw.index.to_numpy(dtype=float)
    _, labels = detect_nonlocal_events(
        xr.DataArray(
            nonlocal_probability(raw).to_numpy(), dims="time", coords={"time": time}
        ),
        raw["head_speed"].to_numpy(),
        raw.index,
        probability_threshold,
        minimum_samples * DECODE_SAMPLE_S,
    )
    # a shallow copy: the new column leaves ``raw``, whose other columns it shares,
    # as it was
    labeled = raw.copy(deep=False)
    labeled["non_local_event"] = labels.to_numpy()
    frame = with_swr_source(labeled, non_local_threshold=probability_threshold)
    return build_classified_nonlocal_event_table(
        frame, nonlocal_threshold=probability_threshold
    )


def epoch_sweep_counts(
    raw: pd.DataFrame, arrival: np.ndarray, departure: np.ndarray
) -> pd.DataFrame:
    """One epoch's events that start inside a well visit, and those with no
    SWR, at each swept setting.

    Parameters
    ----------
    raw : pd.DataFrame
        The epoch's decode frame as stored.
    arrival, departure : ndarray
        The epoch's well visits (first poke-in, final poke-out).

    Returns
    -------
    pd.DataFrame
        One row per setting and value: ``setting``, ``value``, ``n_events``,
        ``n_no_swr``, and for the SWR thresholds ``n_criterion_agrees`` (the
        events whose criterion label at 2 z is their canonical label; the same
        on each of those rows, NaN on the others).
    """

    def events_in_visits(
        probability_threshold: float, minimum_samples: int
    ) -> pd.DataFrame:
        events = detect_classified_events(raw, probability_threshold, minimum_samples)
        return events[
            in_well_visit(events["event_start"].to_numpy(), arrival, departure)
        ]

    rule_events = events_in_visits(
        RULE["probability_threshold"], RULE["minimum_samples"]
    )
    rows = []
    for setting in ("probability_threshold", "minimum_samples"):
        for value in SETTINGS[setting]:
            settings = {**RULE, setting: value}
            events = (
                rule_events
                if settings == RULE
                else events_in_visits(
                    settings["probability_threshold"], settings["minimum_samples"]
                )
            )
            rows.append(
                {
                    "setting": setting,
                    "value": float(value),
                    "n_events": len(events),
                    "n_no_swr": int((~events["is_swr"].astype(bool)).sum()),
                }
            )
    time = raw.index.to_numpy()
    first = np.searchsorted(time, rule_events["event_start"].to_numpy(), side="left")
    last = np.searchsorted(time, rule_events["event_end"].to_numpy(), side="right") - 1
    criterion = kay_criterion(raw["SWR_zscore"].to_numpy(), first, last)
    agrees = int(
        (
            (criterion >= KAY_ZSCORE_THRESHOLD) == rule_events["is_swr"].astype(bool)
        ).sum()
    )
    for value in SWR_THRESHOLDS_Z:
        rows.append(
            {
                "setting": "swr_threshold_z",
                "value": float(value),
                "n_events": len(rule_events),
                "n_no_swr": int((criterion < value).sum()),
                "n_criterion_agrees": agrees,
            }
        )
    return pd.DataFrame(rows)


def rule_rows(counts: pd.DataFrame, setting: str) -> pd.DataFrame:
    """The rows of :func:`epoch_sweep_counts`' counts at the rule's value of
    ``setting`` (a key of :data:`RULE`): every epoch's events at the rule's
    settings."""
    return counts[
        (counts["setting"] == setting) & np.isclose(counts["value"], RULE[setting])
    ]


def sweep_statistics(counts: pd.DataFrame) -> pd.DataFrame:
    """The share of events with no SWR, and the events, at each setting.

    Parameters
    ----------
    counts : pd.DataFrame
        :func:`epoch_sweep_counts`' rows of every epoch, with ``epoch_id``,
        ``animal`` and the session columns ``session_date`` and ``epoch``.

    Returns
    -------
    pd.DataFrame
        One row per setting and value, in :data:`SETTINGS`' order: ``setting``,
        ``value``, ``is_rule``, ``n_events`` (summed over epochs),
        ``events_vs_rule`` (over the rule's events), ``no_swr_fraction`` (the
        same number as ``no_swr_fraction_animal_mean``, as the events summary's
        ``non_swr_fraction``), ``animals`` and ``animal_names`` (each animal's
        share, the mean over
        its epochs with an event of each epoch's share, as
        ``event_statistics.overlap_class_animal_shares`` takes it),
        ``no_swr_fraction_animal_mean`` (the mean over animals of those
        shares, the panel's bold line), ``animal_min`` and ``animal_max``.

    Raises
    ------
    ValueError
        If a value of :data:`SETTINGS` has no row (counts made with other
        settings).
    """
    missing = [
        f"{setting} {value:g}"
        for setting, values in SETTINGS.items()
        for value in values
        if not (
            (counts["setting"] == setting) & np.isclose(counts["value"], value)
        ).any()
    ]
    if missing:
        raise ValueError(
            f"the sweep's counts have no rows at {missing}: they were made with "
            "other settings. Rerun scripts/event_properties/sweep_event_rule.py"
        )
    rows = []
    rule_events = int(rule_rows(counts, "probability_threshold")["n_events"].sum())
    for setting, values in SETTINGS.items():
        for value in values:
            mine = counts[
                (counts["setting"] == setting) & np.isclose(counts["value"], value)
            ]
            with_events = mine[mine["n_events"] > 0]
            animals = animal_session_values(
                with_events["n_no_swr"] / with_events["n_events"],
                with_events["animal"],
                with_events[["session_date", "epoch"]],
                "mean",
            )
            n_events = int(mine["n_events"].sum())
            rows.append(
                {
                    "setting": setting,
                    "value": float(value),
                    "is_rule": bool(np.isclose(value, RULE[setting])),
                    "n_events": n_events,
                    "events_vs_rule": n_events / rule_events,
                    "no_swr_fraction": float(animals.mean()),
                    "animals": animals.to_numpy(),
                    "animal_names": animals.index.to_numpy(),
                    "no_swr_fraction_animal_mean": float(animals.mean()),
                    "animal_min": float(animals.min()),
                    "animal_max": float(animals.max()),
                }
            )
    return pd.DataFrame(rows)


def criterion_agreement(counts: pd.DataFrame) -> float:
    """The share of the rule's events whose Kay criterion label at 2 z is their
    canonical SWR label (:func:`epoch_sweep_counts`' ``n_criterion_agrees``):
    each epoch's share, each animal's mean over its epochs, the mean over
    animals. An epoch with no event is not one of its animal's epochs."""
    rule = rule_rows(counts, "swr_threshold_z")
    rule = rule[rule["n_events"] > 0]
    animals = animal_session_values(
        rule["n_criterion_agrees"] / rule["n_events"],
        rule["animal"],
        rule[["session_date", "epoch"]],
        "mean",
    )
    return float(animals.mean())
