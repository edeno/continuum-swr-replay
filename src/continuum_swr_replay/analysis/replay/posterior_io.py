"""Load the conditional non-local posterior from a decoder results file.

The decoder writes the full acausal posterior to
``{epoch_identifier}_results.nc`` at the end of
``scripts/run_decoding_bandit.py``. For the arm-level replay content
analysis we need the conditional non-local posterior — the posterior
over place bins conditioned on the non-local state — which is derived
from the acausal posterior by
``analysis.hippocampus.posterior_analysis.get_conditional_non_local_posterior``.

This module wraps that derivation plus the netCDF load, so
``tables.py`` does not need to know about ``xarray.open_dataset`` or the
detector-results internals.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr


def load_conditional_posterior(results_path: str | Path) -> xr.DataArray:
    """Load decoder results and return the conditional non-local posterior.

    Parameters
    ----------
    results_path : str | Path
        Path to ``{epoch_identifier}_results.nc`` saved by the decoder.

    Returns
    -------
    xr.DataArray
        Shape ``(time, position)`` — a proper probability distribution
        over place bins per timebin, conditioned on the non-local state.
        Place bins that include junction (``edge_id == -1``) placeholders
        are still present; downstream code excludes them via
        ``summaries.posterior_to_arm_marginal``.

    Notes
    -----
    Uses the ``h5netcdf`` engine to match how the file was written. The
    returned DataArray is lazy-loaded if the file is chunked; call
    ``.load()`` on the caller side if eager materialization is desired.
    """
    # Import lazily to avoid a circular dependency at module import time:
    # hippocampus.posterior_analysis → types → data_loaders → heavy deps.
    from continuum_swr_replay.analysis.hippocampus.posterior_analysis import (
        get_conditional_non_local_posterior,
    )

    results = xr.open_dataset(Path(results_path), engine="h5netcdf")
    return get_conditional_non_local_posterior(results)
