"""Loading saved ContFrag decoding results."""

from pathlib import Path

import xarray as xr
from non_local_detector import ContFragClusterlessClassifier
from non_local_detector.environment import Environment

from continuum_swr_replay.analysis.hippocampus.result_loading import _load_saved_decode
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR


def load_contfrag_results(
    nwb_file_name: str,
    epoch_name: str,
    decoding_brain_area: str = "HPC",
    load_detector: bool = True,
    non_stationary: bool = False,
    ripple_detector_name: str = "Kay",
    training_type: str = "no_ripple",
    root: Path = PROCESSED_DATA_DIR,
) -> tuple[xr.Dataset, ContFragClusterlessClassifier | None, Environment | None]:
    """Load saved ContFrag decoding results and classifier model.

    Parameters
    ----------
    nwb_file_name : str
        Name of the NWB file (e.g., "chimi20200212_.nwb").
    epoch_name : str
        Name of the epoch (e.g., "02_r1").
    decoding_brain_area : str, optional
        Brain area used for decoding, by default "HPC".
    load_detector : bool, optional
        Whether to load the classifier model, by default True.
    non_stationary : bool, optional
        Whether the saved run used a non-stationary transition model. Must
        match the value passed to ``run_decoding_contfrag.py`` so that the
        ``_non_stationary`` filename suffix is reconstructed correctly.
    ripple_detector_name : str, optional
        Ripple detector used for the saved run. Non-Kay detectors append
        ``_{ripple_detector_name}`` to the filename.
    training_type : str, optional
        Training regime used for the saved run. Must match the value passed to
        ``run_decoding_contfrag.py`` so the ``_{training_type}`` suffix is
        reconstructed correctly. The default ``"no_ripple"`` appends no suffix.
    root : Path, optional
        Folder every file is read from, by default ``PROCESSED_DATA_DIR``. A
        decode resuming from a staging folder passes that folder.

    Returns
    -------
    results : xr.Dataset
        ContFrag decoding results.
    detector : ContFragClusterlessClassifier | None
        Fitted classifier (None if load_detector=False).
    env : Environment | None
        Environment configuration (None if load_detector=False).
    """
    return _load_saved_decode(
        f"{nwb_file_name.replace('_.nwb', '')}_{epoch_name}"
        f"_{decoding_brain_area}_contfrag",
        ContFragClusterlessClassifier,
        load_detector=load_detector,
        non_stationary=non_stationary,
        ripple_detector_name=ripple_detector_name,
        training_type=training_type,
        root=root,
    )
