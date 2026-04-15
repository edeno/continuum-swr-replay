"""Raw-data quality statistics for hippocampal clusterless decoding.

These statistics characterize the raw inputs to the clusterless decoder —
per-tetrode spike counts, overall spike yield, and waveform feature
amplitude distributions — so that QC failures observed in the decoded
output can be traced back to recording-level causes (low spike yield,
noise/artifact contamination on a tetrode, etc.).

Unlike the metrics in :mod:`quality_metrics`, which operate on the cheap
pre-computed ``data_over_time`` pickle files, these functions require
loading raw spike times and waveform features via
:func:`~continuum_swr_replay.data_loaders.bandit_task.load_data`. That is
significantly more expensive, so the per-session stats are computed once
by :mod:`scripts.compute_hpc_raw_stats` and saved to a standalone CSV that
:mod:`scripts.run_quality_metrics` merges into ``quality_summary.csv``.

Waveform features are signed, so all amplitude characterization uses
absolute values.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

RAW_STAT_COLUMNS: tuple[str, ...] = (
    "n_tetrodes",
    "n_spikes_total",
    "epoch_duration_s",
    "rate_hz_total",
    "rate_hz_per_tetrode",
    "median_spikes_per_tetrode",
    "min_spikes_per_tetrode",
    "wf_abs_median",
    "wf_abs_iqr",
    "wf_abs_p99",
    "wf_abs_max",
    "wf_per_tet_abs_median_max",
    "wf_per_tet_abs_median_min",
)

# Canonical ordering of discrete states in the non-local detector. Must match
# ``NonLocalClusterlessDetector.state_names`` order produced by the fitting
# pipeline. Used only for naming transition-matrix columns in CSV output.
DISCRETE_STATE_ORDER: tuple[str, ...] = (
    "Local",
    "No-Spike",
    "Non-Local Continuous",
    "Non-Local Fragmented",
)

_SHORT_STATE_NAMES: dict[str, str] = {
    "Local": "local",
    "No-Spike": "nospike",
    "Non-Local Continuous": "nlc",
    "Non-Local Fragmented": "nlf",
}

TRANSITION_MATRIX_COLUMNS: tuple[str, ...] = tuple(
    f"T_{_SHORT_STATE_NAMES[src]}_{_SHORT_STATE_NAMES[dst]}"
    for src in DISCRETE_STATE_ORDER
    for dst in DISCRETE_STATE_ORDER
)


def extract_discrete_transition_matrix(detector) -> dict[str, float]:
    """Flatten a fitted detector's discrete transition matrix into named columns.

    Parameters
    ----------
    detector : NonLocalClusterlessDetector or NonLocalSortedSpikesDetector
        A fitted detector loaded via ``load_model``. Must expose
        ``discrete_state_transitions_`` and ``state_names``.

    Returns
    -------
    dict
        Keys are ``TRANSITION_MATRIX_COLUMNS`` (16 entries). Row ordering
        follows ``DISCRETE_STATE_ORDER`` — the detector's ``state_names``
        are reordered to match, and rows/cols for unexpected states are
        filled with NaN. Non-stationary transition matrices (3D arrays
        averaged over time or covariate) have their mean taken so the
        output is always a 4x4 stationary-equivalent.

    """
    mat = getattr(detector, "discrete_state_transitions_", None)
    state_names = list(getattr(detector, "state_names", []))
    if mat is None or not state_names:
        return dict.fromkeys(TRANSITION_MATRIX_COLUMNS, np.nan)

    mat = np.asarray(mat, dtype=float)
    if mat.ndim == 3:
        # Non-stationary: (n_time, n_states, n_states). Use the mean as a
        # stationary-equivalent summary; non-stationary sessions can be
        # re-inspected separately if needed.
        mat = mat.mean(axis=0)

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return dict.fromkeys(TRANSITION_MATRIX_COLUMNS, np.nan)

    # Reorder rows/cols to DISCRETE_STATE_ORDER.
    name_to_idx = {name: i for i, name in enumerate(state_names)}
    ordered = np.full((4, 4), np.nan, dtype=float)
    for i, src in enumerate(DISCRETE_STATE_ORDER):
        si = name_to_idx.get(src)
        if si is None:
            continue
        for j, dst in enumerate(DISCRETE_STATE_ORDER):
            dj = name_to_idx.get(dst)
            if dj is None:
                continue
            ordered[i, j] = float(mat[si, dj])

    out: dict[str, float] = {}
    for i, src in enumerate(DISCRETE_STATE_ORDER):
        for j, dst in enumerate(DISCRETE_STATE_ORDER):
            col = f"T_{_SHORT_STATE_NAMES[src]}_{_SHORT_STATE_NAMES[dst]}"
            out[col] = ordered[i, j]
    return out


def _nan_stats() -> dict[str, float]:
    return dict.fromkeys(RAW_STAT_COLUMNS, np.nan)


def compute_hpc_raw_stats_single(
    spike_times: Sequence[np.ndarray],
    waveform_features: Sequence[np.ndarray],
    epoch_duration_s: float,
) -> dict[str, float]:
    """Compute per-session raw spike/waveform QC stats for one HPC epoch.

    Parameters
    ----------
    spike_times : sequence of ndarray
        Per-tetrode spike times in seconds, as returned by
        ``data['spike_times']['HPC']`` after coincident-spike filtering.
        One array per tetrode.
    waveform_features : sequence of ndarray
        Per-tetrode waveform feature arrays with shape ``(n_spikes, n_channels)``,
        as returned by ``data['spike_waveform_features']['HPC']``. Values are
        signed; absolute values are used for all amplitude statistics.
    epoch_duration_s : float
        Total epoch duration in seconds. Used to normalize spike rates.

    Returns
    -------
    dict
        Keys are ``RAW_STAT_COLUMNS``. Entries are NaN when the session has
        zero tetrodes, zero spikes, or non-positive duration.

    Notes
    -----
    The waveform statistics are computed on absolute feature values pooled
    across all tetrodes and channels:

    - ``wf_abs_median`` — typical spike amplitude magnitude
    - ``wf_abs_iqr`` — spread of amplitudes (Q75 − Q25)
    - ``wf_abs_p99`` — 99th percentile, sensitive to noise/artifact bursts
    - ``wf_abs_max`` — hardest extreme, useful for catching blown-out tetrodes
    - ``wf_per_tet_abs_median_max`` — max across tetrodes of per-tetrode
      median absolute feature. A large value with a normal ``wf_abs_median``
      points to a single contaminated tetrode.

    """
    if epoch_duration_s <= 0 or len(spike_times) == 0:
        return _nan_stats()

    spikes_per_tet = np.array([int(len(st)) for st in spike_times])
    n_tet = int(len(spikes_per_tet))
    n_total = int(spikes_per_tet.sum())

    if n_total == 0:
        stats = _nan_stats()
        stats.update({
            "n_tetrodes": n_tet,
            "n_spikes_total": 0,
            "epoch_duration_s": float(epoch_duration_s),
            "rate_hz_total": 0.0,
            "rate_hz_per_tetrode": 0.0,
            "median_spikes_per_tetrode": 0.0,
            "min_spikes_per_tetrode": 0,
        })
        return stats

    per_tet_abs_medians: list[float] = []
    all_abs_chunks: list[np.ndarray] = []
    for f in waveform_features:
        f = np.asarray(f)
        if f.size == 0:
            continue
        abs_f = np.abs(f)
        per_tet_abs_medians.append(float(np.median(abs_f)))
        all_abs_chunks.append(abs_f.ravel())

    if all_abs_chunks:
        all_abs = np.concatenate(all_abs_chunks)
        q25, q75, q99 = np.percentile(all_abs, [25, 75, 99])
        wf_abs_median = float(np.median(all_abs))
        wf_abs_iqr = float(q75 - q25)
        wf_abs_p99 = float(q99)
        wf_abs_max = float(all_abs.max())
    else:
        wf_abs_median = wf_abs_iqr = wf_abs_p99 = wf_abs_max = np.nan

    per_tet_abs_arr = np.asarray(per_tet_abs_medians)
    wf_per_tet_max = float(per_tet_abs_arr.max()) if per_tet_abs_arr.size else np.nan
    wf_per_tet_min = float(per_tet_abs_arr.min()) if per_tet_abs_arr.size else np.nan

    return {
        "n_tetrodes": n_tet,
        "n_spikes_total": n_total,
        "epoch_duration_s": float(epoch_duration_s),
        "rate_hz_total": n_total / float(epoch_duration_s),
        "rate_hz_per_tetrode": n_total / float(epoch_duration_s) / n_tet,
        "median_spikes_per_tetrode": float(np.median(spikes_per_tet)),
        "min_spikes_per_tetrode": int(spikes_per_tet.min()),
        "wf_abs_median": wf_abs_median,
        "wf_abs_iqr": wf_abs_iqr,
        "wf_abs_p99": wf_abs_p99,
        "wf_abs_max": wf_abs_max,
        "wf_per_tet_abs_median_max": wf_per_tet_max,
        "wf_per_tet_abs_median_min": wf_per_tet_min,
    }


def check_transition_matrix_flags(summary: pd.DataFrame) -> pd.DataFrame:
    """Apply per-state transition-matrix flagging to a quality summary.

    Encodes biophysically motivated thresholds on the 4×4 discrete state
    transition matrix. Thresholds are expressed in terms of expected dwell
    times (``dwell = 1 / (1 - p_self)`` timebins at 2 ms).

    Diagonal (self-transition) error thresholds
    --------------------------------------------
    ==================  ===========  ==========================
    State               Lower bound  Upper bound
    ==================  ===========  ==========================
    Local               < 0.933      > 0.9999 (dwell > 20 s)
                        (dwell < 30 ms — too short for attractor)
    No-Spike            < 0.80       > 0.9999 (dwell > 20 s)
    NL-Continuous       < 0.80       > 0.998 (dwell > 1 s)
    NL-Fragmented       (not flagged — noisiest state, typical
                        range 0.88–0.96; only flagged via the
                        cross-type check below)
    ==================  ===========  ==========================

    Off-diagonal error thresholds
    -----------------------------
    - ``Local → NL-Continuous`` or ``Local → NL-Fragmented`` > 0.02: would
      imply non-local events every ~50 bins (100 ms).
    - ``NL-Continuous ↔ NL-Fragmented`` cross-type > self-transition:
      fragmented/continuous switching exceeds persistence, making the
      distinction meaningless.

    Structural checks
    -----------------
    - Any entry < 0 or > 1.
    - Row sum ≠ 1.0 (tolerance 1e-6).

    Adds to the returned DataFrame:

    - ``flag_transition_error`` (bool) — any error-level violation above.
    - ``transition_warning_count`` (int) — count of warning-level violations.
      Lower-bound warnings: Local < 0.95, No-Spike < 0.90, NL-Continuous < 0.90.
      Upper-bound warnings: Local > 0.9999 (absorbing), No-Spike > 0.999
      (dwell > 2 s, near-absorbing), NL-Continuous > 0.987 (dwell > 150 ms,
      longer than typical SWR replay). NL-Fragmented is excluded from both
      warnings and errors because its typical range (0.88–0.96) would
      dominate false positives; it is still checked via the cross-type
      off-diagonal rule.

    Both columns are False / 0 for sessions lacking T_* columns.

    """
    out = summary.copy()
    has_t = any(c.startswith("T_") for c in out.columns)

    if not has_t:
        out["flag_transition_error"] = False
        out["transition_warning_count"] = 0
        return out

    def _col(name: str) -> pd.Series:
        return out[name] if name in out.columns else pd.Series(np.nan, index=out.index)

    errors = pd.Series(False, index=out.index)

    # --- Diagonal lower-bound errors ---
    # NL-Fragmented is excluded: it is the noisiest state (typical range
    # 0.88–0.96) and would dominate false positives. It is still checked
    # via the cross-type rule below.
    diag_error_lo = {
        "T_local_local": 0.933,
        "T_nospike_nospike": 0.80,
        "T_nlc_nlc": 0.80,
    }
    for col, threshold in diag_error_lo.items():
        errors |= (_col(col) < threshold).fillna(False)

    # --- Diagonal upper-bound errors (absorbing / stuck states) ---
    diag_error_hi = {
        "T_local_local": 0.9999,     # dwell > 20 s — absorbing
        "T_nospike_nospike": 0.9999,  # dwell > 20 s — absorbing
        "T_nlc_nlc": 0.998,           # dwell > 1 s — stuck in non-local
    }
    for col, threshold in diag_error_hi.items():
        errors |= (_col(col) > threshold).fillna(False)

    # --- Off-diagonal: Local → NL too frequent ---
    for col in ["T_local_nlc", "T_local_nlf"]:
        errors |= (_col(col) > 0.02).fillna(False)

    # --- Off-diagonal: NL cross-type > self-transition ---
    nlc_cross_gt_self = _col("T_nlc_nlf") > _col("T_nlc_nlc")
    errors |= nlc_cross_gt_self.fillna(False)
    nlf_cross_gt_self = _col("T_nlf_nlc") > _col("T_nlf_nlf")
    errors |= nlf_cross_gt_self.fillna(False)

    # --- Structural: entries in [0, 1] ---
    t_cols = [c for c in out.columns if c.startswith("T_")]
    for col in t_cols:
        v = _col(col)
        errors |= ((v < 0) | (v > 1)).fillna(False)

    # --- Structural: row sums = 1.0 ---
    state_shorts = ["local", "nospike", "nlc", "nlf"]
    for src in state_shorts:
        row_cols = [f"T_{src}_{dst}" for dst in state_shorts]
        present = [c for c in row_cols if c in out.columns]
        if len(present) == 4:
            row_sum = out[present].sum(axis=1)
            errors |= ((row_sum - 1.0).abs() > 1e-6).fillna(False)

    out["flag_transition_error"] = errors

    # --- Warning count (sub-threshold but not error) ---
    warnings = pd.Series(0, index=out.index, dtype=int)

    # NL-Fragmented excluded from warnings (same rationale as errors).
    diag_warn_lo = {
        "T_local_local": 0.95,
        "T_nospike_nospike": 0.90,
        "T_nlc_nlc": 0.90,
    }
    for col, threshold in diag_warn_lo.items():
        warnings += (_col(col) < threshold).fillna(False).astype(int)

    diag_warn_hi = {
        "T_local_local": 0.9999,      # absorbing Local
        "T_nospike_nospike": 0.999,    # dwell > 2 s — near-absorbing No-Spike
        "T_nlc_nlc": 0.9867,           # dwell > 150 ms — non-local events longer
    }                                   # than typical SWR replay
    for col, threshold in diag_warn_hi.items():
        warnings += (_col(col) > threshold).fillna(False).astype(int)

    out["transition_warning_count"] = warnings

    return out


def apply_raw_stats_flags(
    summary: pd.DataFrame,
    rate_hz_per_tetrode_flag_threshold: float = 10.0,
    waveform_outlier_ratio_threshold: float = 3.0,
) -> pd.DataFrame:
    """Add raw-data flag columns to a quality summary that includes raw stats.

    Call this after merging a ``hpc_raw_stats.csv`` into the summary produced
    by :func:`~continuum_swr_replay.analysis.hippocampus.quality_metrics.compute_session_quality_summary`.

    Adds:

    - ``flag_low_spike_yield`` — ``rate_hz_per_tetrode`` below threshold.
      Default 10 Hz/tet catches early-recording-day sessions where tetrodes
      have not yet stabilized (observed in j16 20210706–07).
    - ``flag_waveform_extreme`` — ``wf_per_tet_abs_median_max`` greater than
      ``waveform_outlier_ratio_threshold`` times the dataset median of
      ``wf_abs_median`` (i.e., the worst tetrode in the session is more than
      ~3× the typical spike amplitude). Catches single-tetrode noise/artifact
      contamination (observed in senor 20201103 10_r5).
    - ``flag_transition_error`` — any error-level violation of the discrete
      transition matrix (see :func:`check_transition_matrix_flags`).
    - ``transition_warning_count`` — number of warning-level violations.

    All flags are False for sessions lacking the relevant raw-stat columns.

    The returned DataFrame has ``n_flags`` recomputed over the expanded set
    of ``flag_*`` columns.

    """
    out = summary.copy()

    def _flag(col: str, condition) -> pd.Series:
        if col not in out.columns:
            return pd.Series(False, index=out.index)
        return condition(out[col]).fillna(False)

    out["flag_low_spike_yield"] = _flag(
        "rate_hz_per_tetrode",
        lambda x: x < rate_hz_per_tetrode_flag_threshold,
    )

    if "wf_per_tet_abs_median_max" in out.columns and "wf_abs_median" in out.columns:
        baseline = float(out["wf_abs_median"].median(skipna=True))
        if np.isfinite(baseline) and baseline > 0:
            extreme = (
                out["wf_per_tet_abs_median_max"]
                > waveform_outlier_ratio_threshold * baseline
            ).fillna(False)
            out["flag_waveform_extreme"] = extreme
        else:
            out["flag_waveform_extreme"] = pd.Series(False, index=out.index)
    else:
        out["flag_waveform_extreme"] = pd.Series(False, index=out.index)

    out = check_transition_matrix_flags(out)

    flag_cols = [c for c in out.columns if c.startswith("flag_")]
    out["n_flags"] = out[flag_cols].sum(axis=1).astype(int)
    return out
