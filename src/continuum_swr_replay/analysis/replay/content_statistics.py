"""Summary statistics of what the non-local events represent.

Each function takes a canonical table (the event-arm table, one row per event
and candidate arm; the event table, one row per event) or fitted contrasts and
returns named numbers: how much of an event's content sits on the trunk arms,
how long SWR and non-SWR events are, and what the Dirichlet concentration
contrast between them amounts to. SWR status is the tables' ``is_swr``. A share
is a fraction of one event's content; "animals" statistics count the animals
whose own value meets the stated condition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.hippocampus.event_statistics import class_medians
from continuum_swr_replay.data_loaders.constants import (
    TRACK_SEGMENT_TO_PATCH,
    WELL_PATCH_MAPPING,
)

PATCH_TRUNK = {
    patch: arm
    for arm, patch in TRACK_SEGMENT_TO_PATCH.items()
    if arm not in WELL_PATCH_MAPPING
}
"""The trunk arm of each patch (``TRACK_SEGMENT_TO_PATCH``): patch -> trunk arm."""

TRUNK_COLUMNS = ["animal", "event_id", "is_swr", "candidate_arm", "is_current_arm"]
"""The event-arm columns :func:`trunk_shares` reads, besides the response."""

TRUNK_RESPONSE = "arm_fraction"
"""The event-arm column that holds an event's content share on each arm."""

TRUNK_PARTS = ("trunk", "current_patch_trunk", "other_trunks")
"""The parts of the trunk content :func:`trunk_shares` separates."""


def trunk_shares(arm: pd.DataFrame) -> pd.DataFrame:
    """Each event's content share on the trunks, split by the current patch.

    Parameters
    ----------
    arm : pd.DataFrame
        The event-arm table on the content contrast's cohort (at-well events):
        one row per event x candidate arm, with :data:`TRUNK_COLUMNS` and
        ``arm_fraction``.

    Returns
    -------
    pd.DataFrame
        One row per event with exactly one current arm: ``animal``, ``is_swr``,
        ``trunk`` (share on arms 6-8), ``current_patch_trunk`` (the trunk of the
        patch holding the current arm) and ``other_trunks`` (the other two).
    """
    arm = arm[arm["is_swr"].notna()]
    current = arm.loc[arm["is_current_arm"] == 1, ["event_id", "candidate_arm"]]
    current = current[~current["event_id"].duplicated(keep=False)]
    trunk_of = current.set_index("event_id")["candidate_arm"].map(
        lambda a: PATCH_TRUNK[TRACK_SEGMENT_TO_PATCH[int(a)]]
    )
    arm = arm[arm["event_id"].isin(trunk_of.index)]
    on_trunk = arm["candidate_arm"].isin(PATCH_TRUNK.values())
    on_current = (
        arm["candidate_arm"].to_numpy() == arm["event_id"].map(trunk_of).to_numpy()
    )
    share = arm[TRUNK_RESPONSE].astype(float)
    frame = pd.DataFrame(
        {
            "event_id": arm["event_id"],
            "trunk": share.where(on_trunk, 0.0),
            "current_patch_trunk": share.where(on_trunk & on_current, 0.0),
        }
    )
    events = frame.groupby("event_id", sort=False).sum()
    events["other_trunks"] = events["trunk"] - events["current_patch_trunk"]
    meta = arm.drop_duplicates("event_id").set_index("event_id")[["animal", "is_swr"]]
    events = events.join(meta)
    events["is_swr"] = events["is_swr"].astype(bool)
    return events.reset_index()


def _animal_mean_shares(events: pd.DataFrame) -> pd.DataFrame:
    """Each animal's mean share on each trunk part over its SWR and its non-SWR
    events: indexed by ``animal`` and ``is_swr``, one column per part."""
    return events.groupby(["animal", "is_swr"])[list(TRUNK_PARTS)].mean()


def trunk_share_statistics(events: pd.DataFrame) -> pd.DataFrame:
    """How much of an event's content sits on the trunks, in SWR and non-SWR events.

    An animal's share is the mean over its events.

    Parameters
    ----------
    events : pd.DataFrame
        One row per event (:func:`trunk_shares`).

    Returns
    -------
    statistics : pd.DataFrame
        One row per part of the trunk content (:data:`TRUNK_PARTS`):

        - ``swr_median``, ``non_swr_median``: the median over animals of the
          animal's share in SWR and in non-SWR events.
        - ``n_animals``: the animals.
        - ``n_animals_swr_higher``: animals whose SWR share is above their
          non-SWR share.
    """
    per_animal = _animal_mean_shares(events)
    rows = {}
    for part in TRUNK_PARTS:
        table = per_animal[part].unstack("is_swr")
        rows[part] = {
            "swr_median": table[True].median(),
            "non_swr_median": table[False].median(),
            "n_animals": len(table),
            "n_animals_swr_higher": int((table[True] > table[False]).sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def current_patch_trunk_fraction(events: pd.DataFrame) -> dict[str, float]:
    """The current patch's trunk as a share of all trunk content.

    One third if the three trunks shared the trunk content evenly.

    Parameters
    ----------
    events : pd.DataFrame
        One row per event (:func:`trunk_shares`).

    Returns
    -------
    fractions : dict
        ``swr`` and ``non_swr``: the median over animals of the animal's mean
        current-patch trunk share divided by its mean share on all trunks, in
        SWR and in non-SWR events.
    """
    per_animal = _animal_mean_shares(events)
    fractions = {}
    for name, is_swr in (("swr", True), ("non_swr", False)):
        means = per_animal.xs(is_swr, level="is_swr")
        fractions[name] = (means["current_patch_trunk"] / means["trunk"]).median()
    return fractions


def event_length_statistics(events: pd.DataFrame) -> dict[str, float]:
    """How many non-local time bins SWR and non-SWR events hold.

    Parameters
    ----------
    events : pd.DataFrame
        The event table, with ``animal``, ``is_swr`` and ``n_nonlocal_bins``.

    Returns
    -------
    statistics : dict
        ``median_bins``, the median over all events; ``median_bins_swr`` and
        ``median_bins_non_swr``, the medians of the SWR and non-SWR events; and
        ``swr_to_non_swr_ratio``, the SWR median over the non-SWR median.
    """
    medians = class_medians(events, "n_nonlocal_bins", "lower")
    return {
        "median_bins": events["n_nonlocal_bins"].median(),
        "median_bins_swr": medians["median_swr"],
        "median_bins_non_swr": medians["median_non_swr"],
        "swr_to_non_swr_ratio": medians["median_swr"] / medians["median_non_swr"],
    }


def concentration_contrast_statistics(
    total: float, length_adjusted: float
) -> dict[str, float]:
    """What a log concentration contrast between SWR and non-SWR events amounts to.

    Parameters
    ----------
    total : float
        The ``is_swr`` contrast of the Dirichlet-multinomial log concentration.
    length_adjusted : float
        The same contrast in the model whose concentration also depends on the
        event's length.

    Returns
    -------
    statistics : dict
        ``ratio`` and ``length_adjusted_ratio``, ``exp`` of each contrast (an
        SWR event's concentration over a non-SWR event's), and
        ``length_share``, the share of the contrast the length term absorbs:
        ``1 - length_adjusted / total``.
    """
    return {
        "ratio": float(np.exp(total)),
        "length_adjusted_ratio": float(np.exp(length_adjusted)),
        "length_share": 1 - length_adjusted / total,
    }
