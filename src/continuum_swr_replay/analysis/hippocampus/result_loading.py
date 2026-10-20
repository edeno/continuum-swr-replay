"""Loading and managing saved decoding results.

This module provides functions for loading previously saved decoding results
and detector models from disk.

Examples
--------
Load decoding results with detector model:

>>> from continuum_swr_replay.analysis.hippocampus import load_results
>>> results, detector, env = load_results(
...     nwb_file_name="chimi20200212_.nwb",
...     epoch_name="02_r1",
...     decoding_brain_area="HPC"
... )
>>> print(f"Loaded results with {len(results.time)} time points")
>>> print(f"Track has {env.place_bin_size} cm bins")

Load results only (without detector):

>>> results, detector, env = load_results(
...     "chimi20200212_.nwb",
...     "02_r1",
...     load_detector=False
... )
>>> # detector and env will be None
>>> print(f"States: {list(results.states.values)}")

"""

from pathlib import Path
from typing import Any

import xarray as xr
from non_local_detector import NonLocalClusterlessDetector
from non_local_detector.environment import Environment

from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.results import open_results
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR


def load_results(
    nwb_file_name: str,
    epoch_name: str,
    decoding_brain_area: str = "HPC",
    load_detector: bool = True,
    non_stationary: bool = False,
    ripple_detector_name: str = "Kay",
    training_type: str = "no_ripple",
    root: Path = PROCESSED_DATA_DIR,
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
        Folder every file is read from, by default ``PROCESSED_DATA_DIR``. A
        decode resuming from a staging folder passes that folder.

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
        f"{nwb_file_name.replace('_.nwb', '')}_{epoch_name}_{decoding_brain_area}",
        NonLocalClusterlessDetector,
        load_detector=load_detector,
        non_stationary=non_stationary,
        ripple_detector_name=ripple_detector_name,
        training_type=training_type,
        root=root,
    )


def _load_saved_decode(
    epoch_stem: str,
    model_class: Any,
    *,
    load_detector: bool,
    non_stationary: bool,
    ripple_detector_name: str,
    training_type: str,
    root: Path,
) -> tuple[xr.Dataset, Any | None, Environment | None]:
    """Read one saved decode's results and, optionally, its fitted model.

    Shared by :func:`load_results` and
    :func:`~continuum_swr_replay.analysis.contfrag.result_loading.load_contfrag_results`,
    which differ only in ``epoch_stem`` and ``model_class``.

    Parameters
    ----------
    epoch_stem : str
        Epoch identifier before the run suffixes, such as
        ``"chimi20200212_02_r1_HPC"``.
    model_class : type
        Detector class whose ``load_model`` reads the ``_detector.pkl``.
    load_detector, non_stationary, ripple_detector_name, training_type, root
        As in :func:`load_results`.

    Returns
    -------
    results : xr.Dataset
    detector : model_class instance or None
    env : Environment or None
    """
    epoch_identifier = epoch_stem
    if ripple_detector_name != "Kay":
        epoch_identifier += f"_{ripple_detector_name}"
    if non_stationary:
        epoch_identifier += "_non_stationary"
    if training_type != "no_ripple":
        epoch_identifier += f"_{training_type}"

    paths = decode_paths(epoch_identifier, root)
    results = open_results(paths.results)

    if load_detector:
        detector = model_class.load_model(paths.detector)
        env = detector.environments[0]
    else:
        detector = None
        env = None

    return results, detector, env
