"""Bases and design matrices for the PFC GLMs.

The theta design the running-bin test is fit on, the B-spline event
basis and its convolution with event onsets, and the per-model design
matrices the immobility GLMs are fit against.
"""

import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from patsy import dmatrix

from continuum_swr_replay.analysis.constants import DEFAULT_SAMPLING_FREQUENCY_HZ
from continuum_swr_replay.analysis.pfc.event_sets import SWR_NO_NON_LOCAL
from continuum_swr_replay.analysis.pfc.modulation import THREE_CLASS_MODEL
from continuum_swr_replay.intervals import get_spikecount_per_time_bin


def reindex_by(
    df: pd.Series | pd.DataFrame,
    index: pd.Index,
    tolerance: float | None = None,
) -> pd.Series | pd.DataFrame:
    """Reindex pandas object to new index using nearest neighbor.

    Uses pd.merge_asof for efficient O(n log n) nearest-neighbor matching.
    The original pd.reindex(method='nearest') can use O(n*m) memory when
    source and target indices have different sampling rates.

    Parameters
    ----------
    df : pd.Series | pd.DataFrame
        Data to reindex. Index must be sorted.
    index : pd.Index
        Target index to align to. Must be sorted.
    tolerance : float | None, optional
        Maximum distance for nearest neighbor match (in index units, typically
        seconds for time indices). If None, uses unlimited matching which may
        match to distant points. Recommended to set this to the sampling period
        (e.g., 0.002 seconds for 500 Hz data) to prevent spurious matches.

    Returns
    -------
    pd.Series | pd.DataFrame
        Reindexed data with values from nearest time point within tolerance.
        Values outside tolerance will be NaN.

    """
    if tolerance is None:
        warnings.warn(
            "reindex_by called with tolerance=None; nearest neighbor matching will "
            "match to arbitrarily distant points. Consider setting tolerance to the "
            "sampling period (e.g., 0.002 for 500 Hz data) to prevent spurious matches.",
            UserWarning,
            stacklevel=2,
        )

    target_df = pd.DataFrame(index=index)
    source_df = df.copy() if isinstance(df, pd.DataFrame) else df.to_frame()

    # Ensure index dtypes match for merge_asof (empty Series may have int index)
    if len(source_df) == 0:
        source_df.index = source_df.index.astype(target_df.index.dtype)

    merged = pd.merge_asof(
        target_df,
        source_df,
        left_index=True,
        right_index=True,
        direction="nearest",
        tolerance=tolerance,
    )

    if isinstance(df, pd.Series):
        return merged.iloc[:, 0].rename(df.name)
    return merged


#: The three-column theta design, as a patsy formula.
THETA_FORMULA = "1 + np.cos(theta_phase) + np.sin(theta_phase)"


def build_theta_design(
    theta_phase: pd.Series,
    is_immobile: pd.Series,
    sampling_frequency: float = DEFAULT_SAMPLING_FREQUENCY_HZ,
) -> tuple[NDArray, NDArray]:
    """The running-bin mask and theta design the theta test is fit on.

    Split out of :func:`test_theta_modulation` so a session can build the one
    design every unit shares once, and fit every unit against it in a single
    batched call.

    Parameters
    ----------
    theta_phase : pd.Series, shape (n_lfp_samples,)
        Theta phase timeseries (radians), on its own clock.
    is_immobile : pd.Series, shape (n_time,)
        Boolean series indicating immobility at each time bin.
    sampling_frequency : float
        Sampling frequency in Hz, used to set the reindex tolerance.

    Returns
    -------
    joint_mask : NDArray of bool, shape (n_time,)
        Running AND theta phase available.
    design_matrix : NDArray, shape (n_joint, 3)
        Intercept, cosine and sine of theta phase on those bins.
    """
    reindexed_theta = reindex_by(
        theta_phase,
        is_immobile.index,
        tolerance=1.0 / sampling_frequency,
    )
    # Joint mask: running AND theta phase available
    valid = reindexed_theta.notna().to_numpy().ravel()
    joint_mask = (~is_immobile).to_numpy() & valid

    design_matrix = dmatrix(
        THETA_FORMULA,
        {"theta_phase": reindexed_theta[joint_mask]},
        return_type="dataframe",
    )
    return joint_mask, design_matrix.values


#: Spacing of the B-spline's internal knots, in seconds. 50 ms resolves the
#: sharp peri-SWR responses (~50 ms peaks) without giving the kernel more
#: freedom than the events can support.
KNOT_SPACING_S = 0.050


def symmetric_lags(half_window_s: float, sampling_frequency: float) -> NDArray:
    """Integer sample lags covering ``-half_window_s`` to ``+half_window_s``.

    Parameters
    ----------
    half_window_s : float
        Half-width of the window in seconds, e.g.
        :data:`~continuum_swr_replay.analysis.pfc.event_sets.DEFAULT_BASIS_HALF_WINDOW_S`
        (the fit window). The reported lags are a subset of the fit window's,
        chosen by
        :func:`~continuum_swr_replay.analysis.pfc.modulation.in_report_window`.
    sampling_frequency : float
        Sampling frequency in Hz.

    Returns
    -------
    NDArray of int, shape (2 * n_half + 1,)
        ``-n_half .. n_half`` with ``n_half = round(half_window_s *
        sampling_frequency)``, ascending by one sample.
    """
    n_half = int(round(half_window_s * sampling_frequency))
    return np.arange(-n_half, n_half + 1, 1)


def event_basis_functions(
    lags: NDArray, sampling_frequency: float
) -> tuple[NDArray, NDArray, NDArray]:
    """Cubic B-spline basis over the lag window, zero at both ends.

    The kernel a design column carries is ``sum_b beta_b B_b(lag)``. A clamped
    cubic B-spline has exactly two functions that are non-zero at the window's
    edges -- the first, which is 1 at the left edge, and the last, which is 1
    at the right -- and dropping *both* is what makes the kernel return to
    baseline at both ends of the fit window rather than being pinned at one end
    and free at the other. What is left sums to one between the first and last
    internal knot, so a flat modulation over that span is representable by
    equal coefficients, and each coefficient reads as the log-rate modulation
    near its knot.

    Parameters
    ----------
    lags : NDArray, shape (n_lags,)
        Lag indices in samples, positive meaning after onset.
    sampling_frequency : float
        Sampling frequency in Hz.

    Returns
    -------
    spline_basis : NDArray, shape (n_lags, n_basis)
        ``n_basis = len(knots) + 2`` for a cubic spline.
    lag_time : NDArray, shape (n_lags,)
        Lag times in seconds.
    knots : NDArray, shape (n_knots,)
        Internal knot positions in seconds.
    """
    lag_time = np.asarray(lags, dtype=float) / sampling_frequency
    knots = np.arange(lag_time[0], lag_time[-1], KNOT_SPACING_S)[1:]
    # ``include_intercept=True`` asks patsy for the whole basis (patsy drops
    # the first function otherwise); both edge functions are dropped here.
    spline_basis = np.asarray(
        dmatrix(
            "bs(lag_time, knots=knots, include_intercept=True) - 1",
            {"lag_time": lag_time, "knots": knots},
        )
    )[:, 1:-1]
    return spline_basis, lag_time, knots


def convolve_event_basis(
    onset_counts: NDArray, spline_basis: NDArray, lags: NDArray
) -> NDArray:
    """Convolve an onset indicator with each basis function.

    Column ``b`` is ``sum_L indicator[t - L] B_b(L)`` over the lags ``L``, so a
    positive lag is time *after* an onset. Event onsets are sparse, so each
    non-zero bin adds the small basis block at its affected rows. This avoids
    both the old ``(n_time, n_lags)`` matrix of shifted copies (about 1.4 GB
    per event class on the longest sessions) and a dense full-session
    convolution for every basis function. It is exact: all three forms sum the
    same terms.

    Parameters
    ----------
    onset_counts : NDArray, shape (n_time,)
        Event onsets per time bin.
    spline_basis : NDArray, shape (n_lags, n_basis)
        Basis functions evaluated at the lags, from
        :func:`event_basis_functions`.
    lags : NDArray, shape (n_lags,)
        Contiguous lag indices in samples, ascending by one sample.

    Returns
    -------
    NDArray, shape (n_time, n_basis)
    """
    signal = np.asarray(onset_counts, dtype=float)
    basis = np.asarray(spline_basis, dtype=float)
    lag_indices = np.asarray(lags)
    if signal.ndim != 1:
        raise ValueError(f"onset_counts must be 1-D, got shape {signal.shape}")
    if basis.ndim != 2:
        raise ValueError(f"spline_basis must be 2-D, got shape {basis.shape}")
    if lag_indices.ndim != 1 or len(lag_indices) != basis.shape[0]:
        raise ValueError(
            "lags must be 1-D with one value per spline_basis row; got "
            f"{lag_indices.shape} and {basis.shape}"
        )
    if not np.issubdtype(lag_indices.dtype, np.integer) or (
        len(lag_indices) > 1 and not np.all(np.diff(lag_indices) == 1)
    ):
        raise ValueError("lags must be contiguous ascending integer sample offsets")

    n_time = signal.shape[0]
    convolved = np.zeros((n_time, basis.shape[1]), dtype=float)
    for onset in np.flatnonzero(signal):
        positions = onset + lag_indices
        valid = (positions >= 0) & (positions < n_time)
        convolved[positions[valid]] += signal[onset] * basis[valid]
    return convolved


def event_onset_counts(times: pd.DataFrame, time: NDArray) -> NDArray:
    """Event onsets per time bin, from a table of event windows.

    Parameters
    ----------
    times : pandas.DataFrame
        Event windows with a ``start_time`` column, in seconds.
    time : NDArray, shape (n_time,)
        Time bin starts in seconds.

    Returns
    -------
    NDArray, shape (n_time,)
    """
    return get_spikecount_per_time_bin(times.start_time, time)


def _make_conv_spline_basis(
    times: dict[str, pd.DataFrame],
    position_info: pd.DataFrame,
    sampling_frequency: float = 500,
    lags: NDArray | None = None,
) -> tuple[dict[str, NDArray], NDArray, NDArray]:
    # Every production caller (``run_session`` and Figure 4) passes ``lags``
    # explicitly -- the +/-250 ms fit window, ``symmetric_lags(
    # DEFAULT_BASIS_HALF_WINDOW_S, ...)``. This +/-200 ms default is the
    # older window, kept because the unit tests build designs without lags.
    if lags is None:
        lags = np.arange(-100, 101, 1)
    if not times:
        raise ValueError("times dict must be non-empty")

    # The basis depends only on the lags, not on the per-class onsets.
    spline_basis, lag_time, _ = event_basis_functions(lags, sampling_frequency)

    time = position_info.index.to_numpy()
    conv_spline_basis: dict[str, NDArray] = {
        times_name: convolve_event_basis(
            event_onset_counts(times_df, time), spline_basis, lags
        )
        for times_name, times_df in times.items()
    }

    return conv_spline_basis, spline_basis, lag_time


def _make_design_matrices(
    times: dict[str, pd.DataFrame],
    position_info: pd.DataFrame,
    sampling_frequency: float = 500,
    lags: NDArray | None = None,
) -> tuple[dict[str, NDArray], NDArray, NDArray]:
    """Build GLM design matrices for all model variants.

    Parameters
    ----------
    times : dict[str, pd.DataFrame]
        Event times keyed by name. Each DataFrame must have ``start_time``.
        The five keys the models are built from are ``non_local_no_SWR``,
        ``non_local_SWR``, ``non_local``, ``SWR`` and ``matched_control`` --
        the last from :func:`make_matched_control_times`, so a caller that has
        not drawn the control fails here rather than silently fitting a ladder
        without its non-event baseline.
    position_info : pd.DataFrame
        Position data (index defines time bins).
    sampling_frequency : float
        Sampling frequency in Hz.
    lags : NDArray, shape (n_lags,), optional
        Lag indices in samples.

    Returns
    -------
    design_matrices : dict[str, NDArray]
        Design matrices keyed by model name. Each shape
        (n_time, 1 + n_basis * n_event_types_in_model).
    spline_basis : NDArray, shape (n_lags, n_basis)
        B-spline basis functions.
    lag_time : NDArray, shape (n_lags,)
        Lag times in seconds.

    """
    constant_design_matrix = np.ones((len(position_info), 1))

    conv_spline_basis, spline_basis, lag_time = _make_conv_spline_basis(
        times, position_info, sampling_frequency=sampling_frequency, lags=lags
    )

    non_local_no_swr_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["non_local_no_SWR"],
        ],
        axis=1,
    )

    non_local_swr_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["non_local_SWR"],
        ],
        axis=1,
    )

    swr_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["SWR"],
        ],
        axis=1,
    )

    non_local_swr_and_non_local_no_swr_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["non_local_no_SWR"],
            conv_spline_basis["non_local_SWR"],
        ],
        axis=1,
    )

    swr_and_non_local_no_swr_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["SWR"],
            conv_spline_basis["non_local_no_SWR"],
        ],
        axis=1,
    )

    non_local_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["non_local"],
        ],
        axis=1,
    )

    # The matched non-event control, built exactly as ``non_local_no_SWR`` is,
    # and jointly with it: the pair is what makes the replay question nested
    # ("does replay add beyond a duration-matched non-event baseline?") rather
    # than a comparison of two single-regressor models the LR test cannot
    # license.
    matched_control_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["matched_control"],
        ],
        axis=1,
    )

    matched_control_and_non_local_no_swr_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["matched_control"],
            conv_spline_basis["non_local_no_SWR"],
        ],
        axis=1,
    )

    design_matrices = {
        "constant": constant_design_matrix,
        "non_local_no_SWR": non_local_no_swr_design_matrix,
        "non_local_SWR": non_local_swr_design_matrix,
        "SWR": swr_design_matrix,
        "non_local_SWR + non_local_no_SWR": non_local_swr_and_non_local_no_swr_design_matrix,
        "SWR + non_local_no_SWR": swr_and_non_local_no_swr_design_matrix,
        "non_local": non_local_design_matrix,
        "matched_control": matched_control_design_matrix,
        "matched_control + non_local_no_SWR": (
            matched_control_and_non_local_no_swr_design_matrix
        ),
    }

    # The extended event set: SWRs without replay alone, and beside the replay
    # split. Present only when the caller drew that class, so a canonical run
    # builds exactly the ladder above.
    if SWR_NO_NON_LOCAL in conv_spline_basis:
        design_matrices[SWR_NO_NON_LOCAL] = np.concatenate(
            [constant_design_matrix, conv_spline_basis[SWR_NO_NON_LOCAL]], axis=1
        )
        design_matrices[THREE_CLASS_MODEL] = np.concatenate(
            [
                constant_design_matrix,
                conv_spline_basis["non_local_no_SWR"],
                conv_spline_basis["non_local_SWR"],
                conv_spline_basis[SWR_NO_NON_LOCAL],
            ],
            axis=1,
        )

    return design_matrices, spline_basis, lag_time
