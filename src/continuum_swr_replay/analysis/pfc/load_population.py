"""Shared data loading for PFC population figure panels.

Loads and filters the batch PFC GLM results for use by figure04 panel scripts.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from continuum_swr_replay.data_loaders.utils import parse_nwb_filename

logger = logging.getLogger(__name__)


def _extract_animal(nwb_file_name: str) -> str:
    """Animal name from an NWB filename.

    Wraps ``parse_nwb_filename`` so that j16 sessions resolve to ``"j16"``
    rather than ``"j1"`` (which is what the historical
    ``str.split("2")[0]`` heuristic produced — the date YYYYMMDD often
    contains a ``"2"`` early on).
    """
    return parse_nwb_filename(nwb_file_name)[0]


def load_population_data(
    data_dir: Path,
    min_firing_rate: float = 1.0,
    artifact_suffix: str = "",
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """Load and filter PFC population data.

    Parameters
    ----------
    data_dir : Path
        Directory containing *_mPFC_modulation.pkl and .npy files.
    min_firing_rate : float
        Minimum immobility firing rate (Hz) to include a neuron. The screened
        column, ``immobility_avg_firing_rate``, is the mean rate over the bins
        the bundle was fit on -- with a stop cohort, that cohort's immobile
        bins, not every immobile bin -- so the gate screens the sample the
        estimates come from. Which cohort that was is named in the bundle's
        provenance sidecar; ``immobility_avg_firing_rate_all_immobile`` carries
        the unrestricted rate alongside it.
    artifact_suffix : str
        Suffix naming the stop cohort the bundle was fit on ("" for the
        canonical at-well bundle, "_allstops" for the sensitivity one). Only
        bundles carrying this suffix are read, so two cohorts can never be
        pooled into one population.

    Returns
    -------
    df : pd.DataFrame
        Per-neuron statistics, filtered and with FDR-corrected p-values.
    modulations : dict[str, np.ndarray]
        Modulation time courses keyed by "swr", "no_swr", "non_local_swr", "non_local_no_swr".
        Each shape (n_lag_times, n_good_neurons).
    lag_time : np.ndarray
        Lag time axis in seconds.

    """
    lag_time = np.arange(-100, 101) / 500.0

    stem = f"_mPFC_modulation{artifact_suffix}"
    pkls = sorted(data_dir.glob(f"*{stem}.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No *{stem}.pkl files in {data_dir}")

    dfs = []
    swr_list, no_swr_list, nl_swr_list, nl_no_swr_list = [], [], [], []

    n_lag_times = len(lag_time)

    for p in pkls:
        base = str(p).removesuffix(".pkl")
        sess_swr = np.load(f"{base}.SWR_modulation.npy")
        sess_no_swr = np.load(f"{base}.no_SWR_modulation.npy")
        sess_nl_swr = np.load(f"{base}.non_local_SWR_modulation.npy")
        sess_nl_no_swr = np.load(f"{base}.non_local_no_SWR_modulation.npy")

        # Sessions with no surviving neurons in this brain area save
        # placeholder arrays of shape ``(0, 0)`` (see ``run_session``
        # in analysis/pfc/glm.py). hstack would silently broadcast a
        # ``(0, 0)`` array against the expected ``(n_lag_times, k)``,
        # producing a 0-row population matrix and breaking every
        # downstream consumer. Skip such sessions explicitly.
        mod_arrays = (sess_swr, sess_no_swr, sess_nl_swr, sess_nl_no_swr)
        if any(arr.ndim != 2 or arr.shape[0] != n_lag_times for arr in mod_arrays):
            # Distinguish the legitimate no-neuron placeholder from an
            # unexpected shape (corrupted file or a stale lag window from
            # an older run): the latter is silently dropped from the
            # population otherwise, biasing summaries.
            if all(arr.shape == (0, 0) for arr in mod_arrays):
                logger.debug(
                    "load_population: skipping %s (no surviving neurons)", p.name
                )
            else:
                logger.warning(
                    "load_population: skipping %s — modulation array shapes %s do "
                    "not match expected (%d, n); corrupted file or stale lag "
                    "window?",
                    p.name,
                    [a.shape for a in mod_arrays],
                    n_lag_times,
                )
            continue

        # Cross-session alignment guard. The pickles and the
        # modulation arrays are loaded independently, then
        # ``pd.concat`` and ``np.hstack`` are used to assemble the
        # population. If a single session has a different number of
        # pickle rows than modulation columns (or its four arrays
        # disagree on column count), every later session's neurons
        # silently shift along the modulation axis: the second pickle
        # row of session A ends up paired with session B's first
        # modulation column. Validate per session before appending so
        # the row↔column correspondence is always 1-to-1.
        df_session = pd.read_pickle(p)
        n_rows = len(df_session)
        col_counts = {
            "SWR_modulation": sess_swr.shape[1],
            "no_SWR_modulation": sess_no_swr.shape[1],
            "non_local_SWR_modulation": sess_nl_swr.shape[1],
            "non_local_no_SWR_modulation": sess_nl_no_swr.shape[1],
        }
        if any(c != n_rows for c in col_counts.values()):
            raise ValueError(
                f"Per-session shape mismatch in {p.name}: pickle has "
                f"{n_rows} rows but modulation column counts are "
                f"{col_counts}. All four modulation arrays must have "
                f"exactly {n_rows} columns; otherwise concatenation "
                "across sessions silently misaligns neuron metadata "
                "with modulation traces."
            )

        dfs.append(df_session)
        swr_list.append(sess_swr)
        no_swr_list.append(sess_no_swr)
        nl_swr_list.append(sess_nl_swr)
        nl_no_swr_list.append(sess_nl_no_swr)

    if not dfs:
        raise FileNotFoundError(
            f"No usable mPFC modulation files in {data_dir} "
            "(every session had empty / wrong-shape arrays)"
        )

    df = pd.concat(dfs, ignore_index=True)
    df["animal"] = df["nwb_file_name"].map(_extract_animal)

    swr_all = np.hstack(swr_list)
    no_swr_all = np.hstack(no_swr_list)
    nl_swr_all = np.hstack(nl_swr_list)
    nl_no_swr_all = np.hstack(nl_no_swr_list)

    # Filter: GLM succeeded, firing rate threshold, no NaN modulation
    # in *any* of the four returned modulation arrays. The previous
    # form only checked ``swr_all`` and ``no_swr_all``, so neurons
    # with NaN entries in ``nl_swr_all`` / ``nl_no_swr_all`` slipped
    # through and downstream consumers saw NaN-laced non-local
    # modulation traces.
    #
    # If every neuron in a saved session had its GLM fail (the
    # ``fit_neuron`` fallback writes only firing-rate columns), the
    # primary p-value column may not even exist in that session's
    # pickle. After ``pd.concat`` it'll exist with NaN for those
    # rows IFF *some* other session succeeded; if EVERY session
    # failed, the column is absent entirely and ``df["..."]`` raises
    # KeyError. Default to all-False ``has_fit`` in that case so
    # the failed neurons are dropped cleanly.
    primary_pval_col = "SWR vs. constant: pval"
    if primary_pval_col in df.columns:
        has_fit = df[primary_pval_col].notna().values
    else:
        has_fit = np.zeros(len(df), dtype=bool)
    has_fr = (df["immobility_avg_firing_rate"] >= min_firing_rate).values
    no_nan_mod = (
        ~np.any(np.isnan(swr_all), axis=0)
        & ~np.any(np.isnan(no_swr_all), axis=0)
        & ~np.any(np.isnan(nl_swr_all), axis=0)
        & ~np.any(np.isnan(nl_no_swr_all), axis=0)
    )
    good = has_fit & has_fr & no_nan_mod

    df = df.loc[good].reset_index(drop=True)
    swr_all = swr_all[:, good]
    no_swr_all = no_swr_all[:, good]
    nl_swr_all = nl_swr_all[:, good]
    nl_no_swr_all = nl_no_swr_all[:, good]

    # FDR correction on key tests. ``false_discovery_control`` raises
    # ``ValueError: ps must include only numbers between 0 and 1`` if
    # any input is NaN — and the upstream filter only checks the
    # *primary* ``"SWR vs. constant: pval"`` column, so a neuron with
    # NaN in any *secondary* test (e.g. ``non_local_no_SWR vs. constant``)
    # passes ``good`` and then crashes the loader. Apply BH per-test
    # to the finite p-values only and leave the NaN positions as NaN
    # in the corresponding ``fdr_q`` column.
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
        if pval_col not in df.columns:
            # Column absent because every session that ran this loader
            # call had only failed-GLM neurons (the fallback path
            # writes only firing-rate columns). Surface NaN rather
            # than KeyError.
            df[fdr_col] = np.full(len(df), np.nan)
            continue
        raw = df[pval_col].to_numpy(dtype=float)
        ok = np.isfinite(raw)
        fdr_q = np.full_like(raw, np.nan)
        if ok.any():
            pvals = np.clip(raw[ok], 1e-300, 1.0)
            fdr_q[ok] = false_discovery_control(pvals, method="bh")
        df[fdr_col] = fdr_q

    modulations = {
        "swr": swr_all,
        "no_swr": no_swr_all,
        "non_local_swr": nl_swr_all,
        "non_local_no_swr": nl_no_swr_all,
    }

    return df, modulations, lag_time
