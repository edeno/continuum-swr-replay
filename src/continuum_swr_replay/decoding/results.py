"""The one reader of a decode's ``{epoch_identifier}_results.nc``, and of the
clusterless detector it saved beside it (:func:`load_saved_detector`);
:func:`load_results` reads an epoch's saved decode back whole: its results,
fitted detector and environment.

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
from typing import TYPE_CHECKING, Any

import joblib
import pandas as pd
import xarray as xr

from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.identifiers import (
    DEFAULT_SWR_DEFINITION,
    DEFAULT_TRAINING_TYPE,
    decode_identifier,
    format_epoch_identifier,
)
from continuum_swr_replay.paths import processed_data_dir

if TYPE_CHECKING:
    from non_local_detector import NonLocalClusterlessDetector
    from non_local_detector.environment import Environment

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


def load_results(
    nwb_file_name: str,
    epoch_name: str,
    decoding_brain_area: str = "HPC",
    load_detector: bool = True,
    non_stationary: bool = False,
    ripple_detector_name: str = DEFAULT_SWR_DEFINITION,
    training_type: str = DEFAULT_TRAINING_TYPE,
    root: Path | None = None,
) -> tuple[xr.Dataset, NonLocalClusterlessDetector | None, Environment | None]:
    """Load saved decoding results and detector model.

    Parameters
    ----------
    nwb_file_name : str
        Name of the NWB file (e.g., "chimi20200212_.nwb")
    epoch_name : str
        Name of the epoch (e.g., "02_r1")
    decoding_brain_area : str, optional
        Brain area used for decoding, by default "HPC"
    load_detector : bool, optional
        Whether to load the detector model, by default True
    non_stationary : bool, optional
        Whether the saved run used a non-stationary transition model. Must
        match the value passed to ``run_decoding_bandit.py`` so that the
        ``_non_stationary`` filename suffix is reconstructed correctly.
    ripple_detector_name : str, optional
        Ripple detector used for the saved run. Non-Kay detectors append
        ``_{ripple_detector_name}`` to the filename.
    training_type : str, optional
        Training regime used for the saved run. Must match the value passed to
        ``run_decoding_bandit.py`` so the ``_{training_type}`` suffix is
        reconstructed correctly. The default ``"no_ripple"`` appends no suffix.
    root : Path, optional
        Folder every file is read from; ``paths.processed_data_dir()`` when
        None. A decode resuming from a staging folder passes that folder.

    Returns
    -------
    results : xr.Dataset
        Decoding results as xarray Dataset
    detector : NonLocalClusterlessDetector | None
        Fitted detector model (None if load_detector=False)
    env : Environment | None
        Environment configuration (None if load_detector=False)

    """
    return _load_saved_decode(
        format_epoch_identifier(nwb_file_name, epoch_name, decoding_brain_area),
        load_detector=load_detector,
        non_stationary=non_stationary,
        ripple_detector_name=ripple_detector_name,
        training_type=training_type,
        root=root,
    )


def _load_saved_decode(
    epoch_stem: str,
    *,
    load_detector: bool,
    non_stationary: bool,
    ripple_detector_name: str,
    training_type: str,
    root: Path | None,
) -> tuple[xr.Dataset, Any | None, Environment | None]:
    """Read one saved decode's results and, optionally, its fitted model.

    Shared by :func:`load_results` and
    :func:`~continuum_swr_replay.analysis.contfrag.result_loading.load_contfrag_results`,
    which differ only in ``epoch_stem``. The detector is read by
    :func:`~continuum_swr_replay.decoding.results.load_saved_detector`.

    Parameters
    ----------
    epoch_stem : str
        Epoch identifier before the run suffixes, such as
        ``"chimi20200212_02_r1_HPC"``.
    load_detector, non_stationary, ripple_detector_name, training_type, root
        As in :func:`load_results`.

    Returns
    -------
    results : xr.Dataset
    detector : NonLocalClusterlessDetector, ContFragClusterlessClassifier or None
    env : Environment or None
    """
    epoch_identifier = decode_identifier(
        epoch_stem,
        ripple_detector_name=ripple_detector_name,
        non_stationary=non_stationary,
        training_type=training_type,
    )

    paths = decode_paths(
        epoch_identifier, processed_data_dir() if root is None else root
    )
    results = open_results(paths.results)

    if load_detector:
        detector = load_saved_detector(paths.detector)
        env = detector.environments[0]
    else:
        detector = None
        env = None

    return results, detector, env
