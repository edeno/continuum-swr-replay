"""Shared data loading for PFC population figure panels.

Loads and filters the batch PFC GLM results for use by figure04 panel scripts.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import false_discovery_control


def load_population_data(
    data_dir: Path,
    min_firing_rate: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """Load and filter PFC population data.

    Parameters
    ----------
    data_dir : Path
        Directory containing *_mPFC_modulation.pkl and .npy files.
    min_firing_rate : float
        Minimum immobility firing rate (Hz) to include a neuron.

    Returns
    -------
    df : pd.DataFrame
        Per-neuron statistics, filtered and with FDR-corrected p-values.
    modulations : dict[str, np.ndarray]
        Modulation time courses keyed by "SWR", "no_SWR", "nl_SWR", "nl_no_SWR".
        Each shape (n_lag_times, n_good_neurons).
    lag_time : np.ndarray
        Lag time axis in seconds.
    """
    lag_time = np.arange(-100, 101) / 500.0

    pkls = sorted(data_dir.glob("*_mPFC_modulation.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No *_mPFC_modulation.pkl files in {data_dir}")

    dfs = []
    swr_list, no_swr_list, nl_swr_list, nl_no_swr_list = [], [], [], []

    for p in pkls:
        df_sess = pd.read_pickle(p)
        dfs.append(df_sess)

        base = str(p).replace("_modulation.pkl", "_modulation")
        swr_list.append(np.load(f"{base}.SWR_modulation.npy"))
        no_swr_list.append(np.load(f"{base}.no_SWR_modulation.npy"))
        nl_swr_list.append(np.load(f"{base}.non_local_SWR_modulation.npy"))
        nl_no_swr_list.append(np.load(f"{base}.non_local_no_SWR_modulation.npy"))

    df = pd.concat(dfs, ignore_index=True)
    df["animal"] = df["nwb_file_name"].str.split("2").str[0]

    swr_all = np.hstack(swr_list)
    no_swr_all = np.hstack(no_swr_list)
    nl_swr_all = np.hstack(nl_swr_list)
    nl_no_swr_all = np.hstack(nl_no_swr_list)

    # Filter: GLM succeeded, firing rate threshold, no NaN modulation
    has_fit = df["SWR vs. constant: pval"].notna().values
    has_fr = (df["immobility_avg_firing_rate"] >= min_firing_rate).values
    no_nan_mod = (
        ~np.any(np.isnan(swr_all), axis=0)
        & ~np.any(np.isnan(no_swr_all), axis=0)
    )
    good = has_fit & has_fr & no_nan_mod

    df = df.loc[good].reset_index(drop=True)
    swr_all = swr_all[:, good]
    no_swr_all = no_swr_all[:, good]
    nl_swr_all = nl_swr_all[:, good]
    nl_no_swr_all = nl_no_swr_all[:, good]

    # FDR correction on key tests
    fdr_tests = [
        "SWR vs. constant",
        "non_local vs. constant",
        "non_local_no_SWR vs. constant",
        "SWR + non_local_no_SWR vs. SWR",
        "SWR + non_local_no_SWR vs. non_local_no_SWR",
        "non_local_SWR + non_local_no_SWR vs. non_local",
    ]
    for test in fdr_tests:
        pval_col = f"{test}: pval"
        fdr_col = f"{test}: fdr_q"
        pvals = np.clip(df[pval_col].values, 1e-300, 1.0)
        df[fdr_col] = false_discovery_control(pvals, method="bh")

    modulations = {
        "SWR": swr_all,
        "no_SWR": no_swr_all,
        "nl_SWR": nl_swr_all,
        "nl_no_SWR": nl_no_swr_all,
    }

    return df, modulations, lag_time
