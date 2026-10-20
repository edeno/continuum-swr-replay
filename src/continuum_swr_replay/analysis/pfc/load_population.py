"""Shared data loading for PFC population figure panels.

Loads and filters the batch PFC GLM results for use by figure04 panel scripts.

The bundles are fit under a Gaussian prior on the event-basis coefficients and
tested out of fold: a unit's evidence for an event class is the improvement in
held-out Poisson deviance its fixed, matchable event subset buys, read against
repeated draws of the matched non-event control for that identical subset. The
all-event difference remains a separate descriptive result. So the population's "significant
fraction" here is the explicitly descriptive fraction of units with
raw *empirical* ``p < .05`` -- not a population-level inferential result and
not a fraction of asymptotic likelihood-ratio tests. Population inference is
performed only after aggregation from neuron to session to animal. A bundle
that still carries likelihood-ratio columns, or does not record the corrected
Monte Carlo method, is refused rather than pooled with these.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from continuum_swr_replay.analysis.pfc.batched_glm import DEFAULT_PENALTY_SIGMA
from continuum_swr_replay.analysis.pfc.heldout import (
    CONTROL_NULL_EVENT_POLICY,
    DEFAULT_CONTROL_DRAWS,
    EMPIRICAL_PVALUE_METHOD,
    required_usable_null_draws,
)
from continuum_swr_replay.analysis.pfc.matched_control import (
    MATCHED_CONTROL_PLACEMENT_METHOD,
)
from continuum_swr_replay.analysis.replay.fit_provenance import fit_provenance_path
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename

logger = logging.getLogger(__name__)

#: Column suffixes only the maximum-likelihood ladder produced. A penalised
#: fit has no likelihood-ratio reference distribution and no AIC at a stated
#: number of parameters, so their presence identifies a bundle from the
#: earlier estimator.
RETIRED_SUFFIXES = (": lr", ": pval", ": aic")

#: Per-unit empirical p-values the population fractions are read from: one per
#: event class, plus the engagement comparison against the matched non-event
#: baseline. Each uses the finite-sample plus-one Monte Carlo correction.
HELDOUT_P_COLUMNS = {
    "SWR": "SWR: p_heldout",
    "non_local": "non_local: p_heldout",
    "non_local_no_SWR": "non_local_no_SWR: p_heldout",
    "non_local_SWR": "non_local_SWR: p_heldout",
    "engagement": ("matched_control + non_local_no_SWR vs. matched_control: p_heldout"),
}

#: Per-unit denominator of each empirical p-value. A finite p-value in the
#: canonical R=1,000 bundle must carry exactly 1,000 usable draws; retaining
#: the counts also makes failed fits auditable instead of silently changing
#: the p-value grid.
HELDOUT_USABLE_DRAW_COLUMNS = {
    "SWR": "SWR: n_usable_null_draws",
    "non_local": "non_local: n_usable_null_draws",
    "non_local_no_SWR": "non_local_no_SWR: n_usable_null_draws",
    "non_local_SWR": "non_local_SWR: n_usable_null_draws",
    "engagement": (
        "matched_control + non_local_no_SWR vs. matched_control: n_usable_null_draws"
    ),
}

#: Observed statistics those empirical p-values compare against. These are
#: deliberately separate from the all-event held-out differences because a
#: real event with no legal within-bout control cannot enter the matched test.
MATCHED_NULL_OBSERVED_COLUMNS = {
    "SWR": "SWR vs. constant: matched_null_observed_delta_dev",
    "non_local": "non_local vs. constant: matched_null_observed_delta_dev",
    "non_local_no_SWR": (
        "non_local_no_SWR vs. constant: matched_null_observed_delta_dev"
    ),
    "non_local_SWR": "non_local_SWR vs. constant: matched_null_observed_delta_dev",
    "engagement": (
        "matched_control + non_local_no_SWR vs. matched_control: "
        "matched_null_observed_delta_dev"
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
    *HELDOUT_USABLE_DRAW_COLUMNS.values(),
    *MATCHED_NULL_OBSERVED_COLUMNS.values(),
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
        absent = sorted(
            f"{test}: pval"
            for test in LEGACY_FDR_TESTS
            if f"{test}: pval" not in df.columns
        )
        if absent:
            raise ValueError(
                f"{data_dir} was read as a maximum-likelihood tree but carries "
                f"none of {absent}: it is neither generation's bundle. Check "
                "the directory."
            )
    else:
        raise ValueError(f"inference={inference!r} is not one of 'heldout', 'legacy'")


def _extract_animal(nwb_file_name: str) -> str:
    """Animal name from an NWB filename.

    Wraps ``parse_nwb_filename`` so that j16 sessions resolve to ``"j16"``
    rather than ``"j1"`` (which is what the historical
    ``str.split("2")[0]`` heuristic produced — the date YYYYMMDD often
    contains a ``"2"`` early on).
    """
    return parse_nwb_filename(nwb_file_name)[0]


def _validate_heldout_provenance(
    artifacts: list[Path],
    frames: list[pd.DataFrame],
    *,
    expected_n_control_draws: int,
    expected_n_folds: int,
    expected_penalty_sigma: float,
) -> None:
    """Refuse bundles with incompatible or unauditable inference settings."""
    required_draws = required_usable_null_draws(expected_n_control_draws)
    expected = {
        "estimator": "ridge_poisson_glm",
        "inference": "blocked_heldout_deviance",
        "penalty_sigma_log_rate": expected_penalty_sigma,
        "n_folds": expected_n_folds,
        "n_control_draws": expected_n_control_draws,
        "empirical_pvalue_method": EMPIRICAL_PVALUE_METHOD,
        "control_null_event_policy": CONTROL_NULL_EVENT_POLICY,
        "min_usable_null_draws": required_draws,
        "matched_control_placement_method": MATCHED_CONTROL_PLACEMENT_METHOD,
    }
    for artifact, frame in zip(artifacts, frames, strict=True):
        sidecar = fit_provenance_path(artifact)
        try:
            record = json.loads(sidecar.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"{artifact.name} has no readable provenance sidecar at {sidecar}; "
                "its held-out p-value semantics cannot be verified. Refit the bundle."
            ) from error
        if not isinstance(record, dict):
            raise ValueError(
                f"{artifact.name} has a non-object provenance record at {sidecar}; "
                "its held-out p-value semantics cannot be verified. Refit the bundle."
            )
        mismatches = {
            name: {"expected": value, "observed": record.get(name)}
            for name, value in expected.items()
            if record.get(name) != value
        }
        if mismatches:
            raise ValueError(
                f"{artifact.name} has incompatible held-out provenance {mismatches}; "
                "refit it before population loading."
            )
        recorded_counts = record.get("null_usable_draws")
        if not isinstance(recorded_counts, dict):
            raise ValueError(
                f"{artifact.name} does not record per-unit usable null draws; "
                "refit it before population loading."
            )
        for name, column in HELDOUT_USABLE_DRAW_COLUMNS.items():
            raw = recorded_counts.get(name)
            if not isinstance(raw, list) or len(raw) != len(frame):
                raise ValueError(
                    f"{artifact.name} has invalid usable null draws for {name!r}: "
                    f"expected {len(frame)} per-unit values, got {raw!r}."
                )
            sidecar_values = np.asarray(raw)
            table_values = frame[column].to_numpy()
            if (
                not np.issubdtype(sidecar_values.dtype, np.integer)
                or np.any(sidecar_values < 0)
                or np.any(sidecar_values > expected_n_control_draws)
                or not np.array_equal(sidecar_values, table_values)
            ):
                raise ValueError(
                    f"{artifact.name} has inconsistent usable null draws for "
                    f"{name!r} between its sidecar and unit table."
                )
            finite_p = np.isfinite(frame[HELDOUT_P_COLUMNS[name]].to_numpy(dtype=float))
            if np.any(finite_p & (table_values != required_draws)):
                raise ValueError(
                    f"{artifact.name} reports a finite {name!r} p-value without "
                    f"all {required_draws} usable null draws."
                )


def load_population_data(
    data_dir: Path,
    min_firing_rate: float = 1.0,
    artifact_suffix: str = "",
    inference: str = "heldout",
    *,
    expected_n_control_draws: int = DEFAULT_CONTROL_DRAWS,
    expected_n_folds: int = 5,
    expected_penalty_sigma: float = DEFAULT_PENALTY_SIGMA,
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
    expected_n_control_draws, expected_n_folds, expected_penalty_sigma : number
        Canonical held-out inference settings. Every loaded sidecar must match
        all three, preventing partial reruns from pooling generations with
        different p-value grids, folds, or estimators. Ignored for legacy
        reads.

    Returns
    -------
    df : pd.DataFrame
        Per-neuron statistics, filtered. Held-out empirical p-values remain raw
        for the descriptive preview; legacy reads gain their historical
        BH-adjusted ``fdr_q`` columns.
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
    loaded_artifacts: list[Path] = []
    swr_list, no_swr_list, nl_swr_list, nl_no_swr_list = [], [], [], []
    matched_control_list = []

    n_lag_times = len(lag_time)

    for p in pkls:
        base = str(p).removesuffix(".pkl")
        sess_swr = np.load(f"{base}.SWR_modulation.npy")
        sess_no_swr = np.load(f"{base}.no_SWR_modulation.npy")
        sess_nl_swr = np.load(f"{base}.non_local_SWR_modulation.npy")
        sess_nl_no_swr = np.load(f"{base}.non_local_no_SWR_modulation.npy")
        # The matched non-event control postdates the first bundles, so a
        # tree written before it has no such array. Reading one is still
        # legitimate -- an audit does it on purpose -- and the curve is then
        # absent rather than zero: NaN, which the nanmean reduction the
        # docstring asks for already handles. Only on the legacy path: a
        # current bundle always writes the array, and a missing one there is a
        # truncated write, not a vintage.
        if inference == "legacy":
            try:
                sess_matched = np.load(f"{base}.matched_control_modulation.npy")
            except FileNotFoundError:
                sess_matched = np.full_like(sess_swr, np.nan)
        else:
            sess_matched = np.load(f"{base}.matched_control_modulation.npy")

        # Sessions with no surviving neurons in this brain area save
        # placeholder arrays of shape ``(0, 0)`` (see ``run_session``
        # in analysis/pfc/session.py). hstack would silently broadcast a
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
                f"{col_counts}. All five modulation arrays must have "
                f"exactly {n_rows} columns; otherwise concatenation "
                "across sessions silently misaligns neuron metadata "
                "with modulation traces."
            )

        dfs.append(df_session)
        loaded_artifacts.append(p)
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
    if inference == "heldout":
        _validate_heldout_provenance(
            loaded_artifacts,
            dfs,
            expected_n_control_draws=expected_n_control_draws,
            expected_n_folds=expected_n_folds,
            expected_penalty_sigma=expected_penalty_sigma,
        )
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

    # The legacy audit keeps its historical BH correction. Current held-out
    # p-values use a finite R=1,000 Monte Carlo grid; its 1/1001 minimum still
    # cannot support BH across thousands of neurons, so they remain a raw-p
    # descriptive preview rather than a population-level inferential result.
    # The population inference itself belongs to the session -> animal layer.
    adjusted = (
        {f"{test}: pval": f"{test}: fdr_q" for test in LEGACY_FDR_TESTS}
        if inference == "legacy"
        else {}
    )
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
            q[ok] = false_discovery_control(np.clip(raw[ok], 1e-300, 1.0), method="bh")
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
    """Descriptive raw-p fraction per class and engagement.

    Parameters
    ----------
    df : pandas.DataFrame
        A population from :func:`load_population_data` (``inference="heldout"``).
    alpha : float
        Raw empirical-p threshold. This is a preview, not population
        inference; neurons are not independent replicates.

    Returns
    -------
    dict[str, tuple[int, int]]
        ``(n_below_alpha, n_tested)`` per key of :data:`HELDOUT_P_COLUMNS`.
        ``n_tested`` counts the units with a finite p: a unit is untested for a
        class its session had no events of, and must not be counted as a
        negative.
    """
    fractions: dict[str, tuple[int, int]] = {}
    for name, column in HELDOUT_P_COLUMNS.items():
        pvalue = df[column].to_numpy(dtype=float)
        tested = np.isfinite(pvalue)
        fractions[name] = (
            int((pvalue[tested] < alpha).sum()),
            int(tested.sum()),
        )
    return fractions
