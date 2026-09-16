"""Shared data loading for PFC population figure panels.

Loads and filters the batch PFC GLM results for use by figure04 panel scripts.

The bundles are fit under a Gaussian prior on the event-basis coefficients and
tested out of fold: a unit's evidence for an event class is the improvement in
held-out Poisson deviance that class's regressor buys, read against repeated
draws of the matched non-event control. So the population's "significant
fraction" is a fraction of units whose *empirical* p-value survives
Benjamini-Hochberg -- not a fraction of asymptotic likelihood-ratio tests. A
bundle that still carries the likelihood-ratio columns was produced by the
earlier estimator and is refused rather than pooled with these.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from continuum_swr_replay.data_loaders.utils import parse_nwb_filename

logger = logging.getLogger(__name__)

#: Column suffixes only the maximum-likelihood ladder produced. A penalised
#: fit has no likelihood-ratio reference distribution and no AIC at a stated
#: number of parameters, so their presence identifies a bundle from the
#: earlier estimator.
RETIRED_SUFFIXES = (": lr", ": pval", ": aic")

#: Per-unit empirical p-values the population fractions are read from: one per
#: event class, plus the engagement comparison against the matched non-event
#: baseline. Each is the fraction of control draws whose held-out improvement
#: matched or beat the observed one.
HELDOUT_P_COLUMNS = {
    "SWR": "SWR: p_heldout",
    "non_local": "non_local: p_heldout",
    "non_local_no_SWR": "non_local_no_SWR: p_heldout",
    "non_local_SWR": "non_local_SWR: p_heldout",
    "engagement": (
        "matched_control + non_local_no_SWR vs. matched_control: p_heldout"
    ),
}

#: Column a unit is judged to have a usable fit by. A unit the fitter could not
#: place -- no spikes on the fitted bins, or a design it could not estimate --
#: carries NaN here.
HELDOUT_FIT_COLUMN = "SWR: heldout_deviance"

#: Columns a current bundle must carry. A bundle without them predates the
#: held-out inference, and its statistics are not these.
REQUIRED_COLUMNS = (
    HELDOUT_FIT_COLUMN,
    "min_heldout_model",
    *HELDOUT_P_COLUMNS.values(),
)


#: Tests the superseded bundles were FDR-corrected on. Kept so an audit can
#: read those bundles as they were read then; nothing current produces them.
LEGACY_FDR_TESTS = (
    "SWR vs. constant",
    "non_local vs. constant",
    "non_local_no_SWR vs. constant",
    "SWR + non_local_no_SWR vs. SWR",
    "SWR + non_local_no_SWR vs. non_local_no_SWR",
    "non_local_SWR + non_local_no_SWR vs. non_local",
)


def heldout_q_column(name: str) -> str:
    """BH-adjusted column name for one of :data:`HELDOUT_P_COLUMNS`."""
    return f"{HELDOUT_P_COLUMNS[name]}: heldout_q"


def _refuse_stale_bundle(df: pd.DataFrame, inference: str, data_dir: Path) -> None:
    """Refuse a population whose columns are not the requested inference.

    The canonical bundle name is unsuffixed, so a directory holding the
    maximum-likelihood bundles and one holding the ridge bundles look
    identical from the outside. Reading either as the other would report
    asymptotic p-values as held-out ones, or silently return a population with
    no significance columns at all.
    """
    retired = sorted(c for c in df.columns if c.endswith(RETIRED_SUFFIXES))
    missing = sorted(c for c in REQUIRED_COLUMNS if c not in df.columns)
    if inference == "heldout":
        if retired or missing:
            raise ValueError(
                f"{data_dir} holds bundles from the earlier estimator: "
                f"{len(retired)} likelihood-ratio/AIC column(s) "
                f"{retired[:4]}{' ...' if len(retired) > 4 else ''} and "
                f"{len(missing)} missing held-out column(s) {missing}. Refit "
                "them, or read them with inference='legacy'."
            )
    elif inference == "legacy":
        if not missing:
            raise ValueError(
                f"{data_dir} holds bundles scored out of fold, which carry no "
                "likelihood-ratio statistics; read them with "
                "inference='heldout'."
            )
    else:
        raise ValueError(
            f"inference={inference!r} is not one of 'heldout', 'legacy'"
        )


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
    inference: str = "heldout",
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
    inference : {"heldout", "legacy"}
        Which generation of bundle is being read. ``"heldout"`` (the shipped
        one) requires the out-of-fold columns and refuses a directory still
        carrying likelihood-ratio statistics; ``"legacy"`` is the reverse, for
        an audit that reads the superseded bundles on purpose. Neither reads
        the other: the canonical bundle name is unsuffixed, so the two trees
        are indistinguishable from the outside.

    Returns
    -------
    df : pd.DataFrame
        Per-neuron statistics, filtered, with BH-adjusted ``heldout_q`` columns
        beside each empirical p-value (or the legacy ``fdr_q`` columns when
        reading legacy bundles).
    modulations : dict[str, np.ndarray]
        Modulation time courses keyed by "swr", "no_swr", "non_local_swr",
        "non_local_no_swr" and "matched_control". Each shape
        (n_lag_times, n_good_neurons). The first four are NaN-free by
        construction -- a unit with a NaN in any of them is dropped -- while
        the matched-control curve may carry NaN for a unit whose engagement
        model was not estimable, so reduce it with ``np.nanmean``. It is not
        part of the drop rule because it is a baseline the shipped curves are
        read against, not one of them: gating on it would change which units
        the population holds.
    lag_time : np.ndarray
        Lag time axis in seconds.

    Raises
    ------
    ValueError
        If the bundles in ``data_dir`` were produced by a different estimator
        than ``inference`` names.

    """
    lag_time = np.arange(-100, 101) / 500.0

    stem = f"_mPFC_modulation{artifact_suffix}"
    pkls = sorted(data_dir.glob(f"*{stem}.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No *{stem}.pkl files in {data_dir}")

    dfs = []
    swr_list, no_swr_list, nl_swr_list, nl_no_swr_list = [], [], [], []
    matched_control_list = []

    n_lag_times = len(lag_time)

    for p in pkls:
        base = str(p).removesuffix(".pkl")
        sess_swr = np.load(f"{base}.SWR_modulation.npy")
        sess_no_swr = np.load(f"{base}.no_SWR_modulation.npy")
        sess_nl_swr = np.load(f"{base}.non_local_SWR_modulation.npy")
        sess_nl_no_swr = np.load(f"{base}.non_local_no_SWR_modulation.npy")
        sess_matched = np.load(f"{base}.matched_control_modulation.npy")

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
            "matched_control_modulation": sess_matched.shape[1],
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
        matched_control_list.append(sess_matched)

    if not dfs:
        raise FileNotFoundError(
            f"No usable mPFC modulation files in {data_dir} "
            "(every session had empty / wrong-shape arrays)"
        )

    df = pd.concat(dfs, ignore_index=True)
    _refuse_stale_bundle(df, inference, data_dir)
    df["animal"] = df["nwb_file_name"].map(_extract_animal)

    swr_all = np.hstack(swr_list)
    no_swr_all = np.hstack(no_swr_list)
    nl_swr_all = np.hstack(nl_swr_list)
    nl_no_swr_all = np.hstack(nl_no_swr_list)
    matched_control_all = np.hstack(matched_control_list)

    # Filter: GLM succeeded, firing rate threshold, no NaN modulation
    # in *any* of the four returned modulation arrays. The previous
    # form only checked ``swr_all`` and ``no_swr_all``, so neurons
    # with NaN entries in ``nl_swr_all`` / ``nl_no_swr_all`` slipped
    # through and downstream consumers saw NaN-laced non-local
    # modulation traces.
    #
    # If every neuron in a saved session had its GLM fail (the fallback path
    # writes only firing-rate columns), the column that says a neuron was fit
    # at all may not even exist in that session's pickle. After ``pd.concat``
    # it'll exist with NaN for those rows IFF *some* other session succeeded;
    # if EVERY session failed, the column is absent entirely and ``df["..."]``
    # raises KeyError. Default to all-False ``has_fit`` in that case so the
    # failed neurons are dropped cleanly.
    primary_fit_col = (
        HELDOUT_FIT_COLUMN if inference == "heldout" else "SWR vs. constant: pval"
    )
    if primary_fit_col in df.columns:
        has_fit = df[primary_fit_col].notna().values
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
    matched_control_all = matched_control_all[:, good]

    # Multiple comparisons across units, per test. The correction is the same
    # Benjamini-Hochberg it always was; what it is applied to has changed. An
    # empirical p-value from R control draws lives on a grid of 1/R and can be
    # exactly zero (no draw matched the observed improvement), which is a
    # statement bounded by the number of draws, not a vanishing tail -- so the
    # q-values are bounded the same way, and a fraction read off them cannot
    # resolve finer than the null's size.
    #
    # ``false_discovery_control`` raises on NaN input, and the gate above only
    # screens one column, so a unit whose *secondary* test is NaN (its session
    # had no events of the class that test needs) passes the gate and would
    # crash the correction. Apply BH per test to the finite values only and
    # leave the NaN positions NaN.
    if inference == "heldout":
        adjusted = {
            HELDOUT_P_COLUMNS[name]: heldout_q_column(name)
            for name in HELDOUT_P_COLUMNS
        }
    else:
        adjusted = {
            f"{test}: pval": f"{test}: fdr_q"
            for test in LEGACY_FDR_TESTS
        }
    for source_col, q_col in adjusted.items():
        if source_col not in df.columns:
            # Column absent because every session that ran this loader call had
            # only failed-GLM neurons (the fallback path writes only
            # firing-rate columns). Surface NaN rather than KeyError.
            df[q_col] = np.full(len(df), np.nan)
            continue
        raw = df[source_col].to_numpy(dtype=float)
        ok = np.isfinite(raw)
        q = np.full_like(raw, np.nan)
        if ok.any():
            q[ok] = false_discovery_control(
                np.clip(raw[ok], 1e-300, 1.0), method="bh"
            )
        df[q_col] = q

    modulations = {
        "swr": swr_all,
        "no_swr": no_swr_all,
        "non_local_swr": nl_swr_all,
        "non_local_no_swr": nl_no_swr_all,
        "matched_control": matched_control_all,
    }

    return df, modulations, lag_time


def significant_fraction(
    df: pd.DataFrame, alpha: float = 0.05
) -> dict[str, tuple[int, int]]:
    """Units whose held-out improvement survives BH, per class and engagement.

    Parameters
    ----------
    df : pandas.DataFrame
        A population from :func:`load_population_data` (``inference="heldout"``).
    alpha : float
        BH level.

    Returns
    -------
    dict[str, tuple[int, int]]
        ``(n_significant, n_tested)`` per key of :data:`HELDOUT_P_COLUMNS`.
        ``n_tested`` counts the units with a finite q: a unit is untested for a
        class its session had no events of, and must not be counted as a
        negative.
    """
    fractions: dict[str, tuple[int, int]] = {}
    for name in HELDOUT_P_COLUMNS:
        q = df[heldout_q_column(name)].to_numpy(dtype=float)
        tested = np.isfinite(q)
        fractions[name] = (int((q[tested] < alpha).sum()), int(tested.sum()))
    return fractions
