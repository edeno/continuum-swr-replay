"""Theta, ripple and MUA levels of running and of quiet rest, the references for an event's.

Each event's ``theta_z``, ``ripple_power_z`` and ``mua_rate_z`` in the event
table are means, over its bins, of the decode frame's per-epoch z-scores
(``theta_power_zscore``, ``SWR_zscore``, ``MUA_zscore``). On that same scale,
this gives one epoch's levels during running and during quiet rest, so an event
can be read against them: an event whose theta reaches the running level is
theta-like.

- **Running**: bins above 10 cm/s (``state_evidence.windows.regimes``).
- **Quiet rest**: bins below 4 cm/s that are in no non-local, SWR or MUA event.

Bins in excluded time (``in_excluded_time``) are left out of both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.hippocampus.state_evidence.windows import regimes

SIGNALS = {
    "theta": "theta_power_zscore",
    "ripple": "SWR_zscore",
    "mua": "MUA_zscore",
}
"""Each signal's word and its decode-frame column."""

EVENT_COLUMNS = ("non_local_event", "SWR_event", "MUA_event")
"""The frame's event labels; 0 is no event."""


def brain_state_reference(dot: pd.DataFrame) -> dict[str, float]:
    """One epoch's signal levels during running and during quiet rest.

    Parameters
    ----------
    dot : pd.DataFrame
        A decode frame read through the SWR source
        (``analysed_time.with_swr_source``), with ``head_speed``,
        ``in_excluded_time``, :data:`EVENT_COLUMNS` and the columns of
        :data:`SIGNALS`.

    Returns
    -------
    reference : dict
        ``n_running_bins``, ``n_rest_bins``, and for each signal and regime
        ``{signal}_{regime}_median`` and ``{signal}_{regime}_mean`` (NaN when
        the regime has no bin).
    """
    usable = ~dot["in_excluded_time"].to_numpy(dtype=bool)
    regime = regimes(dot["head_speed"].to_numpy(dtype=np.float64))
    in_event = np.zeros(len(dot), dtype=bool)
    for column in EVENT_COLUMNS:
        in_event |= dot[column].to_numpy() > 0
    masks = {
        "running": usable & (regime == "running"),
        "rest": usable & (regime == "immobile") & ~in_event,
    }
    reference: dict[str, float] = {
        f"n_{name}_bins": int(mask.sum()) for name, mask in masks.items()
    }
    for word, column in SIGNALS.items():
        values = dot[column].to_numpy(dtype=np.float64)
        for name, mask in masks.items():
            chosen = values[mask]
            chosen = chosen[np.isfinite(chosen)]
            reference[f"{word}_{name}_median"] = (
                float(np.median(chosen)) if chosen.size else np.nan
            )
            reference[f"{word}_{name}_mean"] = (
                float(chosen.mean()) if chosen.size else np.nan
            )
    return reference
