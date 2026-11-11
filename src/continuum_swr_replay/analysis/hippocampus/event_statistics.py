"""Summary statistics of the immobile non-local events.

Each function takes a table an analysis wrote (the canonical event table, a
reference table, a run's per-event metrics) and returns named numbers: a dict,
or a table with one row per group. They are the statistics the events summary
reports (``continuum_swr_replay.reported.events`` names and describes them) and
a figure that shows one of them calls the same function.

SWR status is the event table's ``is_swr``: strict positive-duration overlap
with an SWR. "Animals" statistics count the animals whose own value meets the
stated condition. Times are in seconds.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

# --- how the events sit against SWRs ----------------------------------------------


def compute_swr_overlap_class_counts(
    classified_events: pd.DataFrame,
) -> dict[str, int]:
    """Count immobile non-local events by SWR-overlap class.

    Operates on the output of
    :func:`continuum_swr_replay.analysis.events.classified_table.build_classified_nonlocal_event_table`,
    restricted to immobile events (``whole_event_speed_eligible``), and
    tallies the ``overlap_class`` column. The single overlap convention
    (strict positive-duration overlap, "contained" = some single SWR spans
    the event) lives in that builder, so this is a thin tally with no
    classification logic of its own.

    Parameters
    ----------
    classified_events : pd.DataFrame
        Per-non-local-event table with ``overlap_class`` and
        ``whole_event_speed_eligible`` columns. May be the builder's
        zero-row empty frame.

    Returns
    -------
    dict[str, int]
        Mapping ``{"swr_contained": n, "swr_partial": n,
        "non_swr": n, "total": n}``.

    """
    zero = {"swr_contained": 0, "swr_partial": 0, "non_swr": 0, "total": 0}
    if classified_events.empty:
        return zero
    # ``.loc`` + ``astype(bool)`` so the builder's zero-row empty frame
    # (where ``whole_event_speed_eligible`` is object-dtype) still indexes
    # cleanly rather than raising on a non-boolean mask.
    immobile = classified_events.loc[
        classified_events["whole_event_speed_eligible"].astype(bool)
    ]
    if immobile.empty:
        return zero
    counts = immobile["overlap_class"].value_counts().to_dict()
    return {
        "swr_contained": int(counts.get("swr_contained", 0)),
        "swr_partial": int(counts.get("swr_partial", 0)),
        "non_swr": int(counts.get("non_swr", 0)),
        "total": int(len(immobile)),
    }


def overlap_class_statistics(events: pd.DataFrame) -> dict[str, float]:
    """How the immobile non-local events divide among the SWR-overlap classes.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        The canonical event table, with ``overlap_class`` (``non_swr``,
        ``swr_contained`` or ``swr_partial``), ``whole_event_speed_eligible``
        and the session columns
        (:data:`~continuum_swr_replay.tables.session_keys.SESSION_COLUMNS`).

    Returns
    -------
    statistics : dict
        - ``n_events``: the events.
        - ``n_non_swr``, ``n_swr_contained``, ``n_swr_partial``: the events of
          each class.
        - ``non_swr_fraction``: the share overlapping no SWR, pooled over animals.
        - ``non_swr_fraction_animal_min``, ``non_swr_fraction_animal_max``: the
          lowest and highest of the animals' own shares; an animal with no
          event of a class has a share of 0 in it.
        - ``n_animals``, ``n_epochs``: animals and epochs with an event.
        - ``n_animals_non_swr_plurality``: animals in which non-SWR is the
          largest of the three classes.
        - ``n_animals_non_swr_majority``: animals in which more than half the
          events are non-SWR.

    Raises
    ------
    ValueError
        If an event is not immobile as a whole, or its class is none of the
        three.
    """
    if not events["whole_event_speed_eligible"].all():
        raise ValueError(
            "the event table holds events that are not whole-event immobile"
        )
    counts = compute_swr_overlap_class_counts(events)
    n_classed = counts["non_swr"] + counts["swr_contained"] + counts["swr_partial"]
    if n_classed != counts["total"]:
        raise ValueError(
            f"{counts['total'] - n_classed} events have an overlap_class that is "
            "none of non_swr, swr_contained and swr_partial"
        )
    shares = (
        events.groupby("animal")["overlap_class"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )
    non_swr = shares["non_swr"]
    return {
        "n_events": counts["total"],
        "n_non_swr": counts["non_swr"],
        "n_swr_contained": counts["swr_contained"],
        "n_swr_partial": counts["swr_partial"],
        "non_swr_fraction": counts["non_swr"] / counts["total"],
        "non_swr_fraction_animal_min": non_swr.min(),
        "non_swr_fraction_animal_max": non_swr.max(),
        "n_animals": len(shares),
        "n_epochs": events.groupby(list(SESSION_COLUMNS)).ngroups,
        "n_animals_non_swr_plurality": (
            non_swr > shares.drop(columns="non_swr").max(axis=1)
        ).sum(),
        "n_animals_non_swr_majority": (non_swr > 0.5).sum(),
    }


def boundary_offsets(events: pd.DataFrame) -> pd.DataFrame:
    """Each SWR-overlapping event's bounds against its matched SWR's.

    The matched SWR is the one containing the event when there is one, else the
    earliest one overlapping it (the event table's ``matched_swr_start`` and
    ``matched_swr_end``).

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        The canonical event table, with ``overlap_class``, ``event_start_s``,
        ``event_end_s``, ``matched_swr_start`` and ``matched_swr_end``.

    Returns
    -------
    offsets : pd.DataFrame, shape (n_overlapping_events, 2)
        ``start_offset_s`` (event start minus SWR start) and ``end_offset_s``
        (event end minus SWR end), indexed as ``events``.

    Raises
    ------
    ValueError
        If an SWR-overlapping event has no matched SWR bounds.
    """
    overlapping = events[events["overlap_class"] != "non_swr"]
    offsets = pd.DataFrame(
        {
            "start_offset_s": overlapping["event_start_s"]
            - overlapping["matched_swr_start"],
            "end_offset_s": overlapping["event_end_s"] - overlapping["matched_swr_end"],
        }
    )
    if offsets.isna().any(axis=None):
        raise ValueError("an SWR-overlapping event has no matched SWR bounds")
    return offsets


def boundary_offset_statistics(events: pd.DataFrame) -> dict[str, float]:
    """The typical offset of an SWR-overlapping event's bounds from its SWR's.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        The canonical event table (:func:`boundary_offsets`).

    Returns
    -------
    statistics : dict
        ``n_overlapping``, the events overlapping an SWR, and
        ``start_offset_median_s`` and ``end_offset_median_s``, the medians of
        their :func:`boundary_offsets`.
    """
    offsets = boundary_offsets(events)
    return {
        "n_overlapping": len(offsets),
        "start_offset_median_s": offsets["start_offset_s"].median(),
        "end_offset_median_s": offsets["end_offset_s"].median(),
    }


# --- theta, ripple, MUA and gamma during events -----------------------------------


def reference_medians(reference: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    """The typical running and quiet-rest level of each signal.

    Parameters
    ----------
    reference : pd.DataFrame, shape (n_epochs, n_columns)
        The brain-state reference: for each signal, each epoch's median while
        running (``<signal>_running_median``) and in quiet rest
        (``<signal>_rest_median``), z-scored within the epoch.
    signals : list of str
        The signals (``"theta"``, ``"ripple"``, ...).

    Returns
    -------
    medians : pd.DataFrame, shape (n_signals, 2)
        ``running`` and ``rest``: the median over epochs of each, by signal.
    """
    return pd.DataFrame(
        {
            "running": {s: reference[f"{s}_running_median"].median() for s in signals},
            "rest": {s: reference[f"{s}_rest_median"].median() for s in signals},
        }
    )


def event_signal_statistics(
    events: pd.DataFrame, reference: pd.DataFrame, signals: Mapping[str, str]
) -> pd.DataFrame:
    """Each signal during events, against the same epochs' running and rest levels.

    An event's value of a signal is its mean over the event's bins of the
    epoch's z-scored signal. Each event is compared with the reference of the
    epoch it occurred in: an animal's running (quiet-rest) level is the median,
    over its events of the class, of each event's own epoch's level, so an
    epoch counts once for each event in it, and an epoch with no such event
    does not count.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        One row per event, with ``is_swr``, the session columns
        (:data:`~continuum_swr_replay.tables.session_keys.SESSION_COLUMNS`) and each
        signal's column.
    reference : pd.DataFrame, shape (n_epochs, n_columns)
        The brain-state reference (:func:`reference_medians`), one row per epoch.
    signals : mapping of str to str
        Each signal's column in ``events``, by signal.

    Returns
    -------
    statistics : pd.DataFrame, shape (2 * n_signals, 4)
        Indexed by signal and ``is_swr``:

        - ``median``: the median event value, z, over the class's events.
        - ``n_animals_below_running``: animals, of those with an event of the
          class, whose median event value is below the median over the same
          events of each event's own epoch's running level.
        - ``n_animals_above_rest``: animals, of those with an event of the
          class, whose median event value is above the median over the same
          events of each event's own epoch's quiet-rest level.
        - ``reaches_running_fraction``: the share of the class's events whose
          value reaches their own epoch's median running level.

    Raises
    ------
    ValueError
        If an event has no value of a signal, or its epoch no running or
        quiet-rest level of one.
    """
    joined = events.merge(
        reference, on=list(SESSION_COLUMNS), how="left", validate="m:1"
    )
    rows = {}
    for signal, column in signals.items():
        missing = (
            joined[[column, f"{signal}_running_median", f"{signal}_rest_median"]]
            .isna()
            .sum()
        )
        if missing.any():
            raise ValueError(
                f"events with no {signal} value, or whose epoch has no running or "
                f"quiet-rest level of it:\n{missing[missing > 0]}"
            )
        for is_swr in (False, True):
            group = joined[joined["is_swr"] == is_swr]
            per_animal = group.groupby("animal").agg(
                event=(column, "median"),
                running=(f"{signal}_running_median", "median"),
                rest=(f"{signal}_rest_median", "median"),
            )
            rows[(signal, is_swr)] = {
                "median": group[column].median(),
                "n_animals_below_running": (
                    per_animal["event"] < per_animal["running"]
                ).sum(),
                "n_animals_above_rest": (
                    per_animal["event"] > per_animal["rest"]
                ).sum(),
                "reaches_running_fraction": (
                    group[column] >= group[f"{signal}_running_median"]
                ).mean(),
            }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(
        ["signal", "is_swr"]
    )


def power_change_statistics(reference: pd.DataFrame, bands: list[str]) -> pd.DataFrame:
    """Each band's power during events against the same epoch's quiet rest, in dB.

    Parameters
    ----------
    reference : pd.DataFrame, shape (n_epochs, n_columns)
        One row per epoch, with ``animal`` and, for each band and class,
        ``<band>_db_non_swr`` and ``<band>_db_swr``: ``10 log10`` of the mean
        band power over the class's event samples over that of quiet rest.
    bands : list of str
        The bands (``"slow_gamma"``, ``"fast_gamma"``).

    Returns
    -------
    statistics : pd.DataFrame, shape (2 * n_bands, 2)
        Indexed by band and ``is_swr``: ``median`` over epochs, and
        ``n_animals_above_rest``, the animals whose median over their epochs is
        above 0 dB.

    Raises
    ------
    ValueError
        If an epoch has no value for a band and class.
    """
    rows = {}
    for band in bands:
        for is_swr, group in ((False, "non_swr"), (True, "swr")):
            column = f"{band}_db_{group}"
            if reference[column].isna().any():
                raise ValueError(
                    f"{int(reference[column].isna().sum())} epochs have no {column}"
                )
            per_animal = reference.groupby("animal")[column].median()
            rows[(band, is_swr)] = {
                "median": reference[column].median(),
                "n_animals_above_rest": (per_animal > 0).sum(),
            }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["band", "is_swr"])


def rest_matched_theta_statistics(events: pd.DataFrame) -> pd.DataFrame:
    """The theta-like share of duration-matched quiet rest, by SWR status.

    An event, or a quiet-rest window, is theta-like when its mean theta reaches
    its epoch's median running theta. Each event has windows of quiet rest with
    as many bins as it has; the reference share of a class is the mean over its
    events of their windows' theta-like share, so it carries the class's
    durations.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        One row per event, with ``is_swr``, ``animal``, ``theta_z``,
        ``theta_running_median``, ``rest_windows`` (how many windows it has)
        and ``rest_theta_like_share`` (the share of them that are theta-like).

    Returns
    -------
    statistics : pd.DataFrame, shape (2, 2)
        Indexed by ``is_swr``: ``rest_theta_like_fraction``, the mean over the
        class's events of ``rest_theta_like_share``, and
        ``n_animals_above_rest``, the animals, of those with an event of the
        class, whose share of theta-like events is above the mean of their
        events' ``rest_theta_like_share``.

    Raises
    ------
    ValueError
        If an event has no window (no rest stretch of its epoch is as long), or
        lacks its theta, its epoch's running theta, its number of windows or
        their theta-like share.
    """
    missing = (
        events[
            ["theta_z", "theta_running_median", "rest_windows", "rest_theta_like_share"]
        ]
        .isna()
        .sum()
    )
    if missing.any():
        raise ValueError(f"events lack a value:\n{missing[missing > 0]}")
    if (events.rest_windows <= 0).any():
        raise ValueError(
            f"{int((events.rest_windows <= 0).sum())} events have no duration-matched "
            "quiet-rest window"
        )
    usable = events.assign(theta_like=events.theta_z >= events.theta_running_median)
    rows = {}
    for is_swr in (False, True):
        group = usable[usable.is_swr == is_swr]
        per_animal = group.groupby("animal").agg(
            events=("theta_like", "mean"), rest=("rest_theta_like_share", "mean")
        )
        rows[is_swr] = {
            "rest_theta_like_fraction": group.rest_theta_like_share.mean(),
            "n_animals_above_rest": (per_animal.events > per_animal.rest).sum(),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("is_swr")


# --- spike support ----------------------------------------------------------------


def spike_evidence_statistics(metrics: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Each group's spike evidence, against matched quiet windows.

    Event-level medians, pooled over epochs.

    Parameters
    ----------
    metrics : pd.DataFrame, shape (n_events_and_windows, n_columns)
        The spike-evidence run's metrics: one row per scored event or quiet
        window, with ``group``, ``animal``, ``epoch_id``, ``n_spikes``,
        ``n_tetrodes``, ``dll_per_spike`` (the log-likelihood ratio per spike of
        the decoded location against the animal's) and ``dist_to_gpi_peak``
        (track distance from the MAP to the ground-process intensity peak, cm).
    baseline : str
        The ``group`` of the matched quiet windows.

    Returns
    -------
    statistics : pd.DataFrame, shape (n_groups, 8)
        Indexed by group:

        - ``n_scored``: rows scored.
        - ``spikes_median`` (spikes), ``tetrodes_median`` (tetrodes),
          ``dll_per_spike_median`` (nats per spike), ``gpi_distance_median_cm``
          (cm): medians over the group's rows.
        - ``n_epochs_scored``: epochs with a row of the group.
        - ``n_epochs_above_baseline``, ``n_animals_above_baseline``: of the
          epochs (animals) with a row of the group, those whose median
          ``dll_per_spike`` is above that of their own quiet windows. NaN for
          the baseline itself.

    Raises
    ------
    ValueError
        If an epoch with a scored row has no ``dll_per_spike`` of a quiet
        window to be compared with.
    """
    base = metrics[metrics["group"] == baseline]
    epoch_base = base.groupby("epoch_id")["dll_per_spike"].median()
    animal_base = base.groupby("animal")["dll_per_spike"].median()
    without = epoch_base.reindex(metrics["epoch_id"].unique()).isna()
    if without.any():
        raise ValueError(
            f"{int(without.sum())} epochs have no {baseline!r} dll_per_spike to "
            f"compare with: {sorted(without.index[without])}"
        )
    rows = {}
    for group, group_rows in metrics.groupby("group"):
        epoch_median = group_rows.groupby("epoch_id")["dll_per_spike"].median()
        animal_median = group_rows.groupby("animal")["dll_per_spike"].median()
        is_baseline = group == baseline
        rows[group] = {
            "n_scored": len(group_rows),
            "spikes_median": group_rows["n_spikes"].median(),
            "tetrodes_median": group_rows["n_tetrodes"].median(),
            "dll_per_spike_median": group_rows["dll_per_spike"].median(),
            "gpi_distance_median_cm": group_rows["dist_to_gpi_peak"].median(),
            "n_epochs_scored": len(epoch_median),
            "n_epochs_above_baseline": np.nan
            if is_baseline
            else (epoch_median > epoch_base.reindex(epoch_median.index)).sum(),
            "n_animals_above_baseline": np.nan
            if is_baseline
            else (animal_median > animal_base.reindex(animal_median.index)).sum(),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("group")


# --- one class or two -------------------------------------------------------------


def peak_ripple_mode_statistics(
    distribution: pd.DataFrame, n_animals: int, alpha: float = 0.05
) -> dict[str, float]:
    """Whether the events' peak ripple power has one mode, and where it is.

    Parameters
    ----------
    distribution : pd.DataFrame, shape (n_rows, n_columns)
        The one-class-or-two analysis's per-animal distribution tests, with
        ``animal`` (an animal, or ``"pooled"``), ``power`` (the power measure),
        ``dip_p`` (Hartigan's dip test) and ``main_mode_z`` (the KDE's main
        mode, z).
    n_animals : int
        The animals the peak power must have a test for.
    alpha : float
        The level below which the dip test rejects one mode.

    Returns
    -------
    statistics : dict
        - ``n_animals_unimodal``: animals, of ``n_animals``, in which the dip
          test does not reject one mode of the peak power (``dip_p`` at or
          above ``alpha``).
        - ``mode_animal_min``, ``mode_animal_max``: the lowest and highest of
          the animals' main modes, z.

    Raises
    ------
    ValueError
        If the peak power does not have one test for each of ``n_animals``
        animals, or a test has no ``dip_p`` or no ``main_mode_z``.
    """
    peak = distribution[
        (distribution.animal != "pooled") & (distribution.power == "peak")
    ]
    if len(peak) != n_animals or peak.animal.nunique() != n_animals:
        raise ValueError(
            f"the peak-power tests do not hold one row for each of {n_animals} "
            f"animals: {sorted(peak.animal)}"
        )
    for column in ("dip_p", "main_mode_z"):
        if peak[column].isna().any():
            raise ValueError(
                f"{int(peak[column].isna().sum())} animals' peak-power tests have "
                f"no {column}"
            )
    return {
        "n_animals_unimodal": (peak.dip_p >= alpha).sum(),
        "mode_animal_min": peak.main_mode_z.min(),
        "mode_animal_max": peak.main_mode_z.max(),
    }


def threshold_step_statistics(
    steps: pd.DataFrame, n_knots: int, window_z: float, n_animals: int
) -> pd.DataFrame:
    """For each property, the animals with a step at the SWR threshold.

    Parameters
    ----------
    steps : pd.DataFrame, shape (n_fits, n_columns)
        The one-class-or-two analysis's step fits: one row per animal, property
        and setting, with ``n_knots``, ``power``, ``window_z``, ``covariate``,
        ``property``, ``has_step`` (the step term's interval excludes 0 and it
        lowers the held-out error) and ``step_share_overall`` (the term over
        the animal's SWR minus non-SWR difference).
    n_knots : int
        The spline's knots in the main setting.
    window_z : float
        The half-width of the main setting's window around the threshold, z.
    n_animals : int
        The animals every property must have one fit for.

    Returns
    -------
    statistics : pd.DataFrame, shape (n_properties, 3)
        Indexed by property, for the main setting (the Kay criterion, no
        covariate), each of ``n_animals`` animals: ``n_animals_with_step``;
        ``n_animals_same_sign``, the animals whose step points the way of their
        SWR minus non-SWR difference (a positive share); and
        ``step_share_median``, the median over animals of the step as a share
        of that difference.

    Raises
    ------
    ValueError
        If a property does not have one fit for each of ``n_animals`` animals
        in the main setting.
    """
    main = steps[
        (steps.n_knots == n_knots)
        & (steps.power == "criterion")
        & np.isclose(steps.window_z, window_z)
        & (steps.covariate == "none")
    ]
    by_property = main.groupby("property").agg(
        n_animals_with_step=("has_step", "sum"),
        n_fits=("has_step", "size"),
        n_animals_fitted=("animal", "nunique"),
        n_animals_same_sign=("step_share_overall", lambda share: (share > 0).sum()),
        step_share_median=("step_share_overall", "median"),
    )
    fits = by_property[["n_fits", "n_animals_fitted"]]
    if not (fits == n_animals).all(axis=None):
        raise ValueError(
            f"not every property has one fit for each of {n_animals} animals:\n{fits}"
        )
    return by_property.drop(columns=fits.columns)


def most_steps(
    statistics: pd.DataFrame, properties: list[str]
) -> tuple[int, list[str]]:
    """The most animals with a step in any of ``properties``
    (:func:`threshold_step_statistics`), and the properties that attain it."""
    with_step = statistics.loc[properties, "n_animals_with_step"]
    return int(with_step.max()), list(with_step.index[with_step == with_step.max()])


# --- properties -------------------------------------------------------------------


def class_medians(
    events: pd.DataFrame, column: str, direction: str
) -> dict[str, float]:
    """A per-event property's median in non-SWR and in SWR events.

    Plain medians of all events with a value; not duration-controlled.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        One row per event, with ``is_swr``, ``animal`` and ``column``.
    column : str
        The property.
    direction : {"lower", "higher"}
        The direction of the non-SWR median against the SWR median to count
        animals for.

    Returns
    -------
    statistics : dict
        ``median_non_swr``, ``median_swr``, and ``n_animals_in_direction``: the
        animals whose own non-SWR median is ``direction`` than their SWR median.

    Raises
    ------
    ValueError
        If ``direction`` is neither ``"lower"`` nor ``"higher"``.
    """
    if direction not in ("lower", "higher"):
        raise ValueError(f"direction {direction!r} is neither 'lower' nor 'higher'")
    known = events[[column, "is_swr", "animal"]].dropna(subset=[column])
    median = known.groupby("is_swr")[column].median()
    by_animal = known.groupby(["animal", "is_swr"])[column].median().unstack()
    compare = pd.Series.lt if direction == "lower" else pd.Series.gt
    return {
        "median_non_swr": median[False],
        "median_swr": median[True],
        "n_animals_in_direction": compare(by_animal[False], by_animal[True]).sum(),
    }


# --- where content sits -----------------------------------------------------------

OCCUPANCY_PLACES = ("remote_well", "own_well", "at_well", "remote_path", "near_off")
"""Where content can sit: at a remote well, at the animal's own well, at either,
on a remote path, or near the animal off the wells."""


def content_occupancy_shares(occupancy: pd.DataFrame) -> pd.DataFrame:
    """Where content sits against the animal and the reward wells.

    Parameters
    ----------
    occupancy : pd.DataFrame, shape (n_sessions * n_groups, n_columns)
        The content-occupancy table: per session and ``group`` (``"SWR"``,
        ``"nonSWR"``), ``n`` non-local time bins and the share of them at each
        place (``remote_well``, ``own_well``, ``remote_path``, ``near_off``).

    Returns
    -------
    shares : pd.DataFrame, shape (n_groups, 5)
        For each group, the share of its time bins at each of
        :data:`OCCUPANCY_PLACES`, weighted by time bins over sessions.

    Raises
    ------
    ValueError
        If a session with time bins has no share at a place.
    """
    return _weighted_shares(occupancy, ["group"])


def n_animals_non_swr_more_at_well(occupancy: pd.DataFrame) -> int:
    """The animals whose non-SWR content is at a well more often than their SWR
    content: each animal's time-bin-weighted share at a well
    (:func:`content_occupancy_shares`, with ``animal``), by group."""
    at_well = _weighted_shares(occupancy, ["animal", "group"])["at_well"].unstack()
    return int((at_well["nonSWR"] > at_well["SWR"]).sum())


def _weighted_shares(occupancy: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """The share of time bins at each place, weighted by time bins, per ``by``."""
    places = ["remote_well", "own_well", "remote_path", "near_off"]
    without = occupancy.loc[occupancy["n"] > 0, places].isna().any(axis=1)
    if without.any():
        raise ValueError(
            f"{int(without.sum())} sessions with time bins have no share at a place"
        )
    occupancy = occupancy.assign(
        at_well=occupancy["remote_well"] + occupancy["own_well"]
    )

    def weighted(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                place: float((frame[place] * frame["n"]).sum() / frame["n"].sum())
                for place in OCCUPANCY_PLACES
            }
        )

    return occupancy.groupby(by).apply(weighted, include_groups=False)
