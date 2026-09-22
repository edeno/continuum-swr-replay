"""Connected components of the overlap graph, as a diagnostic.

Spec 03, "Overlap is not transitive". A can overlap B and B overlap C while A
and C share no time, so membership in "found by all" is not defined by the
pairs alone. Each component is labelled by which detectors contribute to it,
with its size and span, so a long chain fusing events that do not individually
overlap is visible. A component is never an event: it carries no bounds that
could be mistaken for a replacement interval, only a span.
"""

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pandas as pd
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

from continuum_swr_replay.session_keys import SESSION_COLUMNS_DAY

__all__ = ["COMPONENT_DTYPES", "overlap_components"]

COMPONENT_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "component_id": np.int64,
        "n_events": np.int64,
        "n_pairs": np.int64,
        "n_detectors": np.int64,
        "detectors": object,
        "span_start_s": np.float64,
        "span_end_s": np.float64,
        "span_s": np.float64,
    }
)
"""One row per component. ``detectors`` joins the contributing detectors'
names in name order with ``+``; ``component_id`` counts from 0 by onset
within each epoch."""


def overlap_components(
    inventory: pd.DataFrame, pairs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label each event with its component, and summarize the components.

    Parameters
    ----------
    inventory : DataFrame
        Events in the common schema, bridging events already removed.
    pairs : DataFrame
        ``overlap_pairs(inventory)``.

    Returns
    -------
    events : DataFrame
        ``inventory`` with a ``component_id`` column; bounds unchanged. An
        event overlapping nothing is a component of its own.
    components : DataFrame
        Columns and dtypes as ``COMPONENT_DTYPES``.

    """
    events = inventory.sort_values(
        [*SESSION_COLUMNS_DAY, "event_start_s", "event_key"]
    ).reset_index(drop=True)
    position = pd.Series(np.arange(len(events)), index=events.event_key)
    graph = coo_array(
        (
            np.ones(len(pairs)),
            (
                position.loc[pairs.event_key_a].to_numpy(),
                position.loc[pairs.event_key_b].to_numpy(),
            ),
        ),
        shape=(len(events), len(events)),
    )
    _, label = connected_components(graph, directed=False)
    # events are in onset order within contiguous epochs, and pairs never
    # cross epochs: numbering labels by first appearance numbers each epoch's
    # components by onset, in one contiguous block per epoch
    first_seen = np.empty(len(events), dtype=np.int64)
    first_seen[pd.unique(label)] = np.arange(len(pd.unique(label)))
    ordinal = pd.Series(first_seen[label])
    events["component_id"] = ordinal - ordinal.groupby(
        [events[c] for c in SESSION_COLUMNS_DAY]
    ).transform("min")

    pair_component = events.set_index("event_key").component_id.loc[pairs.event_key_a]
    n_pairs = (
        pd.DataFrame(
            {c: pairs[c].to_numpy() for c in SESSION_COLUMNS_DAY}
            | {"component_id": pair_component.to_numpy()}
        )
        .groupby([*SESSION_COLUMNS_DAY, "component_id"])
        .size()
    )
    grouped = events.groupby([*SESSION_COLUMNS_DAY, "component_id"])
    components = pd.DataFrame(
        {
            "n_events": grouped.size(),
            "n_pairs": n_pairs,
            "n_detectors": grouped.detector.nunique(),
            "detectors": grouped.detector.agg(lambda d: "+".join(sorted(set(d)))),
            "span_start_s": grouped.event_start_s.min(),
            "span_end_s": grouped.event_end_s.max(),
        }
    )
    components["n_pairs"] = components.n_pairs.fillna(0)
    components["span_s"] = components.span_end_s - components.span_start_s
    components = components.reset_index()
    if components.empty:
        components = pd.DataFrame(columns=list(COMPONENT_DTYPES))
    return events, components[list(COMPONENT_DTYPES)].astype(dict(COMPONENT_DTYPES))
