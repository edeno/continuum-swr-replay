"""Loading saved ContFrag decoding results."""

import xarray as xr
from non_local_detector import ContFragClusterlessClassifier
from non_local_detector.environment import Environment

from continuum_swr_replay.parameters import PROCESSED_DATA_DIR


def load_contfrag_results(
    nwb_file_name: str,
    epoch_name: str,
    decoding_brain_area: str = "HPC",
    load_detector: bool = True,
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

    Returns
    -------
    results : xr.Dataset
        ContFrag decoding results.
    detector : ContFragClusterlessClassifier | None
        Fitted classifier (None if load_detector=False).
    env : Environment | None
        Environment configuration (None if load_detector=False).
    """
    epoch_identifier = (
        f"{nwb_file_name.replace('_.nwb', '')}_{epoch_name}"
        f"_{decoding_brain_area}_contfrag"
    )

    results = xr.open_dataset(
        PROCESSED_DATA_DIR / f"{epoch_identifier}_results.nc",
        engine="h5netcdf",
    )
    state_bin_coord_names = list(results["state_bins"].coords)
    results = results.set_index(state_bins=state_bin_coord_names)

    if load_detector:
        detector = ContFragClusterlessClassifier.load_model(
            PROCESSED_DATA_DIR / f"{epoch_identifier}_detector.pkl"
        )
        env = detector.environments[0]
    else:
        detector = None
        env = None

    return results, detector, env
