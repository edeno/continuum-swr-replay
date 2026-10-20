"""The one reader of a decode's ``{epoch_identifier}_results.nc``.

The decode runners write the fitted decode's ``results.nc``
(``save_results_and_model`` in ``scripts/decoding/_decode_runner.py``) after
``reset_index("state_bins")``: netCDF cannot store a multi-index, so its
``state`` and ``position`` come back as **non-index** coordinates on
``state_bins``. :func:`open_results` reads the file and rebuilds that
multi-index, so ``results.acausal_posterior.sel(state="Non-Local Continuous")``
resolves. Every reader of the file opens it here.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr


def open_results(path: str | Path, *, cache: bool = True) -> xr.Dataset:
    """Open a decode's ``results.nc``, with the ``state_bins`` multi-index rebuilt.

    Parameters
    ----------
    path : str or Path
        ``{epoch_identifier}_results.nc``, as ``decode_paths(...).results``
        names it.
    cache : bool, optional
        Passed to ``xarray.open_dataset``: False for a reader that reads
        many windows of a large file and keeps none of them in memory.

    Returns
    -------
    xr.Dataset
        Opened lazily with the ``h5netcdf`` engine, as the file was written;
        ``state_bins`` is the (``state``, ``position``) multi-index. Its
        ``close`` (or a ``with`` block) closes the file.
    """
    results = xr.open_dataset(Path(path), engine="h5netcdf", cache=cache)
    indexed = results.set_index(state_bins=list(results["state_bins"].coords))
    # set_index returns a new Dataset that no longer closes the file
    indexed.set_close(results.close)
    return indexed
