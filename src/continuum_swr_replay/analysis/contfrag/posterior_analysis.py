"""Posterior analysis for the Continuous-Fragmented classifier.

The ContFrag model has 2 states: Continuous (RandomWalk) and Fragmented
(Uniform). Both states decode meaningful spatial content — unlike the
NonLocal detector where Local and No-Spike states carry no spatial info.
The combined posterior is simply the full posterior (always normalized).
"""

from typing import Literal

import numpy as np
import xarray as xr


def get_contfrag_posterior(
    results: xr.Dataset,
    state: Literal["combined", "Continuous", "Fragmented"] = "combined",
) -> xr.DataArray:
    """Extract posterior from ContFrag results, optionally conditioned on one state.

    Parameters
    ----------
    results : xr.Dataset
        ContFrag decoding results containing ``acausal_posterior`` with
        states=["Continuous", "Fragmented"].
    state : {"combined", "Continuous", "Fragmented"}
        Which posterior to extract:
        - ``"combined"``: sum of both states (= full posterior, always normalized)
        - ``"Continuous"``: posterior conditioned on Continuous state
        - ``"Fragmented"``: posterior conditioned on Fragmented state

    Returns
    -------
    xr.DataArray
        Shape ``(time, position)``. For per-state conditioning, time points
        with negligible state probability are forward/backward filled.
    """
    if state == "combined":
        posterior = (
            results.acausal_posterior.sel(state="Continuous")
            + results.acausal_posterior.sel(state="Fragmented")
        )
        # Combined posterior is already normalized (sums to 1 per time bin)
        return posterior

    # Per-state conditional posterior
    acausal_posterior = results.acausal_posterior.sel(state=state)
    state_prob = acausal_posterior.sum("position")

    conditional_posterior = xr.where(
        state_prob > 1e-10,
        acausal_posterior / state_prob,
        np.nan,
    )
    conditional_posterior = conditional_posterior.ffill("time").bfill("time")

    return conditional_posterior  # type: ignore[no-any-return]
