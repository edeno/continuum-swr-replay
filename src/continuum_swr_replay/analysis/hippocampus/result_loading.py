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

import xarray as xr
from non_local_detector import NonLocalClusterlessDetector
from non_local_detector.environment import Environment

from continuum_swr_replay.parameters import PROCESSED_DATA_DIR


def load_results(
    nwb_file_name: str,
    epoch_name: str,
    decoding_brain_area: str = "HPC",
    load_detector: bool = True,
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

    Returns
    -------
    results : xr.Dataset
        Decoding results as xarray Dataset
    detector : NonLocalClusterlessDetector | None
        Fitted detector model (None if load_detector=False)
    env : Environment | None
        Environment configuration (None if load_detector=False)
    """
    epoch_identifier = f"{nwb_file_name.replace('_.nwb', '')}_{epoch_name}"

    results = xr.open_dataset(
        PROCESSED_DATA_DIR / f"{epoch_identifier}_{decoding_brain_area}_results.nc"
    )
    state_bin_coord_names = list(results["state_bins"].coords)
    results = results.set_index(state_bins=state_bin_coord_names)

    if load_detector:
        detector = NonLocalClusterlessDetector.load_model(
            PROCESSED_DATA_DIR
            / f"{epoch_identifier}_{decoding_brain_area}_detector.pkl"
        )
        env = detector.environments[0]
    else:
        detector = None
        env = None

    return results, detector, env
