"""The one reader of a decode's ``{epoch_identifier}_results.nc``, and of the
clusterless detector it saved beside it (:func:`load_saved_detector`).

The decode runners write the fitted decode's ``results.nc``
(``save_results_and_model`` in ``scripts/decoding/_decode_runner.py``) after
``reset_index("state_bins")``: netCDF cannot store a multi-index, so its
``state`` and ``position`` come back as **non-index** coordinates on
``state_bins``. :func:`open_results` reads the file and rebuilds that
multi-index, so ``results.acausal_posterior.sel(state="Non-Local Continuous")``
resolves. Every reader of the file opens it here.

The decode frame (``data_over_time``) carries each state's probability as a
column; :func:`nonlocal_probability` sums the two non-local states'.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import xarray as xr

ENCODING_KEY = ("", 0)
"""The encoding group a single-environment decode is saved under."""


def nonlocal_probability(frame: pd.DataFrame) -> pd.Series:
    """Each row's non-local probability: Non-Local Continuous plus Fragmented.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows of a decode frame, with ``Non-Local_Continuous_prob`` and
        ``Non-Local_Fragmented_prob``.

    Returns
    -------
    pd.Series, shape (n_rows,)
        On ``frame``'s index, in the columns' dtype; NaN where either is NaN.
    """
    return frame["Non-Local_Continuous_prob"] + frame["Non-Local_Fragmented_prob"]


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


def load_saved_detector(path: str | Path) -> Any:
    """A decode's saved clusterless detector, its encoding environment marked fitted.

    The fit flag is not persisted through a save (and a detector saved by an
    older library version carries none), though the grid is intact; the
    track-distance methods refuse to run without it. Only a grid that is
    actually there is vouched for.

    Parameters
    ----------
    path : str or Path
        The decode's ``{epoch_identifier}_detector.pkl``
        (``decode_paths(...).detector``).

    Returns
    -------
    detector
        As saved, with ``encoding_model_[ENCODING_KEY]["environment"]`` fitted.

    Raises
    ------
    FileNotFoundError
        There is no detector at `path`. Nothing is substituted for it.
    KeyError
        The detector holds no :data:`ENCODING_KEY` encoding group.
    ValueError
        The encoding environment has no place grid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No detector at {path}.")
    detector = joblib.load(path)
    if ENCODING_KEY not in detector.encoding_model_:
        raise KeyError(
            f"The detector at {path} has no {ENCODING_KEY!r} encoding group; it "
            f"has {list(detector.encoding_model_)}."
        )
    environment = detector.encoding_model_[ENCODING_KEY]["environment"]
    if (
        getattr(environment, "place_bin_centers_", None) is None
        or getattr(environment, "is_track_interior_", None) is None
    ):
        raise ValueError(f"The environment in {path} has no fitted place grid.")
    environment._is_fitted = True
    return detector
