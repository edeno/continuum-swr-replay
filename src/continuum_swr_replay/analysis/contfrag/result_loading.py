"""Loading saved ContFrag decoding results."""

from pathlib import Path

import xarray as xr
from non_local_detector import ContFragClusterlessClassifier
from non_local_detector.environment import Environment

from continuum_swr_replay.decoding.identifiers import (
    CONTFRAG_SUFFIX,
    DEFAULT_SWR_DEFINITION,
    DEFAULT_TRAINING_TYPE,
    format_epoch_identifier,
)
from continuum_swr_replay.decoding.result_loading import _load_saved_decode


def load_contfrag_results(
    nwb_file_name: str,
    epoch_name: str,
    decoding_brain_area: str = "HPC",
    load_detector: bool = True,
    non_stationary: bool = False,
    ripple_detector_name: str = DEFAULT_SWR_DEFINITION,
    training_type: str = DEFAULT_TRAINING_TYPE,
    root: Path | None = None,
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
        Folder every file is read from; ``paths.processed_data_dir()`` when
        None. A decode resuming from a staging folder passes that folder.

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
        format_epoch_identifier(
            nwb_file_name, epoch_name, decoding_brain_area, CONTFRAG_SUFFIX
        ),
        load_detector=load_detector,
        non_stationary=non_stationary,
        ripple_detector_name=ripple_detector_name,
        training_type=training_type,
        root=root,
    )
