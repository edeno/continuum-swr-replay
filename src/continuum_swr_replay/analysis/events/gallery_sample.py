"""The event gallery's sample: each event's category and ambiguity, and the draw.

The QC gallery (``scripts/figures/generate_event_gallery.py``) shows a sample
of events per SWR-overlap category and animal. :func:`event_catalog` lists one
epoch's events with their category and an ambiguity feature (how near the
event is to another category); :func:`select_events` draws about ``n`` events
per category and animal, spread over the ambiguity's quantile bins
(:func:`stratified_sample`), each group's generator seeded from its strings
(:func:`group_seed`), so the sample is the same in every run.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.events.classified_table import (
    build_classified_nonlocal_event_table,
)
from continuum_swr_replay.analysis.events.immobility import intervals_from_labels
from continuum_swr_replay.analysis.events.nonlocal_detection import (
    strict_overlap_matrix,
)
from continuum_swr_replay.intervals import min_gap_to_intervals
from continuum_swr_replay.tables.session_keys import one_session

__all__ = [
    "CATEGORIES",
    "SAMPLE_SEED",
    "event_catalog",
    "group_seed",
    "select_events",
    "stratified_sample",
]

CATEGORIES = ("swr_contained", "swr_partial", "non_swr", "swr_not_nonlocal")
"""The gallery's categories: the three SWR-overlap classes of the immobile
non-local events, and the SWRs overlapping no non-local event."""
SAMPLE_SEED = 20260617
"""The base of every group's sample seed (:func:`group_seed`)."""


def event_catalog(dot: pd.DataFrame, epoch_id: str, epoch_name: str) -> pd.DataFrame:
    """One epoch's gallery catalog: each event's category and ambiguity.

    Parameters
    ----------
    dot : pd.DataFrame
        The epoch's decode frame read through the SWR source
        (``analysed_time.read_dot_with_swr_source``). A ``Non-Local_prob``
        column (Continuous plus Fragmented) is added to it.
    epoch_id : str
        ``{animal}{day}_{epoch}``, written on each row.
    epoch_name : str
        The epoch, written on each row.

    Returns
    -------
    pd.DataFrame
        One row per immobile non-local event (categories ``swr_contained``,
        ``swr_partial``, ``non_swr``, from the classified table's
        ``overlap_class``) and one per SWR interval overlapping no non-local
        event (``swr_not_nonlocal``), each with ``ambiguity``: for a partial
        event the share of it inside its matched SWR; for a contained one its
        duration over its SWR's; for a non-SWR one the gap to the nearest SWR
        (:func:`~continuum_swr_replay.intervals.min_gap_to_intervals`); for an
        SWR alone the peak non-local probability inside it.
    """
    dot["Non-Local_prob"] = (
        dot["Non-Local_Continuous_prob"] + dot["Non-Local_Fragmented_prob"]
    )
    ct = build_classified_nonlocal_event_table(dot)
    rows: list[dict] = []

    if len(ct):
        animal, day, _ = one_session(ct, "The gallery catalog")
        day = int(day)
        # Immobile only (matches the regression analysis population).
        ct = ct[ct["whole_event_speed_eligible"]].copy()
        ev_dur = (ct["event_end"] - ct["event_start"]).to_numpy()

        swr_intervals = intervals_from_labels(dot, "SWR_event")
        swr_starts = swr_intervals["start_time"].to_numpy()
        swr_ends = swr_intervals["end_time"].to_numpy()

        # Ambiguity feature per category (see module docstring).
        #  - swr_partial: fraction of the EVENT inside its matched SWR
        #    (low -> barely overlapping -> near non_swr; high -> near contained)
        #  - swr_contained: event/SWR duration ratio (->1 means event nearly
        #    fills the SWR, i.e. near partial; small means snugly contained)
        #  - non_swr: temporal gap to nearest SWR (small -> near partial)
        for i, (_, ev) in enumerate(ct.iterrows()):
            cls = ev["overlap_class"]
            common = {
                "epoch_id": epoch_id,
                "animal": animal,
                "day": day,
                "epoch_name": epoch_name,
                "category": cls,
                "event_label": int(ev["non_local_event"]),
                "event_id": int(ev["event_id"]),
                "plot_start": float(ev["event_start"]),
                "plot_end": float(ev["event_end"]),
                "duration_ms": float(1000 * ev_dur[i]),
                "peak_nonlocal_prob": float(ev["peak_nonlocal_prob"]),
            }
            if cls == "swr_partial":
                ov_start = max(ev["event_start"], ev["matched_swr_start"])
                ov_end = min(ev["event_end"], ev["matched_swr_end"])
                ov = max(0.0, ov_end - ov_start)
                feat = ov / ev_dur[i] if ev_dur[i] > 0 else 0.0
            elif cls == "swr_contained":
                swr_dur = ev["matched_swr_end"] - ev["matched_swr_start"]
                feat = ev_dur[i] / swr_dur if swr_dur > 0 else 1.0
            else:  # non_swr
                feat = float(
                    min_gap_to_intervals(
                        np.array([ev["event_start"]]),
                        np.array([ev["event_end"]]),
                        swr_starts,
                        swr_ends,
                    )[0]
                )
            common["ambiguity"] = float(feat)
            rows.append(common)

        # swr_not_nonlocal: SWR intervals overlapping no non-local event.
        nl_intervals = intervals_from_labels(dot, "non_local_event")
        nl_starts = nl_intervals["start_time"].to_numpy()
        nl_ends = nl_intervals["end_time"].to_numpy()
        if len(swr_intervals):
            overlaps_nl = strict_overlap_matrix(
                swr_starts, swr_ends, nl_starts, nl_ends
            ).any(axis=1)
            only = swr_intervals.loc[~overlaps_nl]
            for _, sw in only.iterrows():
                s, e = float(sw["start_time"]), float(sw["end_time"])
                win = dot.loc[s:e, "Non-Local_prob"]
                # ambiguity = peak non-local prob inside the SWR window
                # (high but no labeled event = borderline "almost an event")
                peak = float(win.max()) if len(win) else 0.0
                rows.append(
                    {
                        "epoch_id": epoch_id,
                        "animal": animal,
                        "day": day,
                        "epoch_name": epoch_name,
                        "category": "swr_not_nonlocal",
                        "event_label": int(sw["SWR_event"]),
                        "event_id": -1,
                        "plot_start": s,
                        "plot_end": e,
                        "duration_ms": float(1000 * (e - s)),
                        "peak_nonlocal_prob": peak,
                        "ambiguity": peak,
                    }
                )

    return pd.DataFrame(rows)


def stratified_sample(
    df: pd.DataFrame, n: int, n_bins: int = 5, seed: int = 0
) -> pd.DataFrame:
    """Sample ~n rows spread across quantile bins of ``ambiguity``.

    Quantile-binning then sampling evenly per bin guarantees the sample spans
    the full ambiguity range — both clear and boundary cases — rather than
    concentrating on the typical event.
    """
    if len(df) <= n:
        return df
    rng = np.random.default_rng(seed)
    ranks = df["ambiguity"].rank(method="first")
    bins = pd.qcut(ranks, q=min(n_bins, len(df)), labels=False, duplicates="drop")
    df = df.assign(_bin=bins)
    n_bin = df["_bin"].nunique()
    per_bin = max(1, n // n_bin)
    picks = []
    for _, g in df.groupby("_bin"):
        take = min(len(g), per_bin)
        picks.append(g.iloc[rng.permutation(len(g))[:take]])
    res = pd.concat(picks)
    if len(res) < n:  # top up from the remainder
        extra = df.loc[~df.index.isin(res.index)]
        k = min(len(extra), n - len(res))
        if k:
            res = pd.concat([res, extra.iloc[rng.permutation(len(extra))[:k]]])
    return res.drop(columns="_bin")


def group_seed(animal: str, category: str) -> int:
    """The sample seed of one (animal, category) group, the same in every process.

    ``SAMPLE_SEED`` plus a CRC-32 of the two strings. Python's ``hash()`` of a
    string is salted per process (``PYTHONHASHSEED``), so a seed built from it
    selected different events in each run.
    """
    return SAMPLE_SEED + (zlib.crc32(f"{animal}|{category}".encode()) & 0xFFFF)


def select_events(catalog: pd.DataFrame, n_per_category: int) -> pd.DataFrame:
    """Stratified sample of ~n events per category per animal, deterministic per group.

    Parameters
    ----------
    catalog : pd.DataFrame
        :func:`event_catalog` rows, with ``animal``, ``category`` and
        ``ambiguity``.
    n_per_category : int

    Returns
    -------
    pd.DataFrame
        The selected rows.
    """
    selected = []
    for (animal, cls), g in catalog.groupby(["animal", "category"]):
        selected.append(
            stratified_sample(g, n_per_category, seed=group_seed(animal, cls))
        )
    return pd.concat(selected, ignore_index=True)
