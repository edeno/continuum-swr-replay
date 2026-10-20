"""Session-tables inspection figure.

``plot_session_tables`` stacks a time-aligned multi-panel view of one
session so the user can trace the transformation from raw decoder
output through the four canonical tables. It is the canonical-tables
analog of :func:`plot_detector`: same layout conventions, a superset
of the core panels, plus overlays and optional rows that expose each
step of the derivation chain.

Panels fall into four groups, dispatched by name through the ``panels``
argument. :data:`DEFAULT_PANELS` is a deliberately small
``plot_detector``-style overview; **every covariate panel is opt-in**, so
a bare call shows none of them:

* **Responses and decoder state** — ``arm_count_hard`` and the
  non-local-probability-weighted MAP count ``arm_count_soft``
  / ``arm_fraction`` / ``arm_posterior_mass``, posterior heatmaps, and the
  derivation-debug rows (``arm_markers``, ``mismatch_pointers``).
* **Arm-grain fitted covariates** (:data:`ARM_COVARIATE_PANELS`) — the
  arm-grain predictors, spanning the M ladder plus terms fit only by the
  content-contrast (``candidate_arm_distance_rank``) and choice (``cumulative_reward``)
  models. It is *not* a faithful M-ladder design matrix in either
  direction: those two are in no M rung, and the ``arm_fe_*`` dummies
  carried by every rung from M0a have no panel (they are arm identity,
  which is already the y-axis). :data:`BEHAVIORAL_MODEL_PANELS` is the
  narrower belief group — candidate-arm reward-belief mean and variance from
  the beta-bernoulli behavioral model, alongside the programmed candidate-arm
  reward probability they are read against.
* **Patch-grain fitted covariates** (:data:`PATCH_COVARIATE_PANELS`) — the
  P-ladder predictors, plus ``is_prev_patch`` (patch-switch ladder only)
  and the canonical ``*_patch`` aggregates (``Pv`` only);
  ``visit_recency_trials_patch`` has no panel. These require
  ``event_patch_table``.
* **Stop-grain fitted covariates** (:data:`RATE_COVARIATE_PANELS`) — the
  Family-A rate design matrix, keyed on stops rather than events. The
  per-bin reward-history basis (``reward_history_rcos_*``,
  ``reward_history_incomplete``) has no panel of its own: it is a fixed
  transform of the known-outcome lag vector that the reward-lag kernel
  panel draws.

Note that the task-reward-probability patch aggregate is the experimenter's
*programmed* reward probability, while the reward-belief-mean patch aggregate
is the *fitted behavioral model's* belief. They are
different quantities on the same scale; plotting them adjacent is the
point.

Three families of fitted predictors are deliberately **not** given their
own panels, because a panel would carry no information the reader cannot
already read off an existing one:

* ``swr_x_*`` interactions are the elementwise product of ``is_swr`` with
  a base covariate. Read them as the base panel masked by the ``is_swr``
  row of ``context_flags``.
* ``*_z`` columns (for example,
  ``reward_belief_variance_candidate_arm_preoutcome_z``) are affine rescalings of
  the raw column that is plotted; the spatial/temporal pattern is
  identical and only the color-bar units would differ.
* ``replay_fraction`` / ``replay_fraction_swr`` /
  ``replay_fraction_non_swr`` are
  choice-grain columns built by ``build_choice_dataset``, not by the
  canonical table builders, so they are not reachable from this
  function's inputs. ``replay_frac_by_role`` covers the same quantity
  summarized by arm role.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator

from continuum_swr_replay.analysis.replay.model_specs import RECENCY_SPLINE_COLUMNS

from ..palettes import patch_grouped_arm_palette
from ._common import (
    _ANNOTATION_FONTSIZE,
    _MISSING_CELL_COLOR,
    _PROBABILITY_CMAP,
    _REWARDED_COLOR,
    _STRUCTURAL_NA_COLOR,
    _ZERO_CELL_COLOR,
    _PanelContext,
    _require_columns,
)
from .arm_covariates import (
    _make_recency_spline_panel,
    _plot_arm_dist_rank,
    _plot_arm_never_visited,
    _plot_cumulative_reward,
    _plot_distance_to_candidate_arm,
    _plot_is_current,
    _plot_is_last_rewarded_elsewhere,
    _plot_is_prev,
    _plot_recent_reward_rate,
    _plot_reward_belief_mean_candidate_arm,
    _plot_reward_belief_variance_candidate_arm,
    _plot_task_reward_probability_candidate_arm,
    _plot_visit_recency_trials,
    _plot_visit_recency_trials_event_leaf_centered,
)
from .patch_covariates import (
    _plot_is_current_patch,
    _plot_is_last_rewarded_noncurrent_patch,
    _plot_is_prev_patch,
    _plot_reward_belief_mean_candidate_arm_preoutcome_patch,
    _plot_reward_count_candidate_arm_last10_known_trials_patch,
    _plot_reward_fraction_candidate_arm_last10_visits_patch,
    _plot_task_reward_probability_candidate_arm_patch,
)
from .responses import (
    _compute_mismatch_table,
    _plot_arm_count_hard,
    _plot_arm_count_soft,
    _plot_arm_fraction,
    _plot_arm_markers,
    _plot_arm_mass_stackplot,
    _plot_arm_posterior_heatmap,
    _plot_arm_posterior_mass,
    _plot_conditional_posterior,
    _plot_context_flags,
    _plot_event_bars,
    _plot_hpd_widths,
    _plot_mismatch_pointers,
    _plot_mua,
    _plot_n_nonlocal_bins,
    _plot_posterior_entropy_mean,
    _plot_posterior_peak_mean,
    _plot_replay_frac_by_role,
    _plot_speed,
    _plot_spike_times,
    _plot_state_probs,
    _plot_stop_activity,
    _plot_swr_lfp,
    _plot_time_since_stop,
    _plot_trial_rail,
)
from .stop_covariates import (
    _plot_last_completed_trip_switch,
    _plot_reward_lag_kernel,
    _plot_reward_rate_trial_k10,
)

# Core panels (always available). Later chunks register additional
# panel names in this registry.
DEFAULT_PANELS: tuple[str, ...] = (
    "swr_lfp",
    "mua",
    "spike_times",
    "state_probs",
    "conditional_posterior",
    "speed",
    "trial_rail",
)


# ── Panel registry ──


_PANEL_REGISTRY: dict[str, Callable[[Axes, _PanelContext], None]] = {
    "swr_lfp": _plot_swr_lfp,
    "mua": _plot_mua,
    "spike_times": _plot_spike_times,
    "state_probs": _plot_state_probs,
    "conditional_posterior": _plot_conditional_posterior,
    "speed": _plot_speed,
    "trial_rail": _plot_trial_rail,
    "arm_markers": _plot_arm_markers,
    "mismatch_pointers": _plot_mismatch_pointers,
    "arm_posterior_heatmap": _plot_arm_posterior_heatmap,
    "arm_count_hard": _plot_arm_count_hard,
    "arm_count_soft": _plot_arm_count_soft,
    "arm_posterior_mass": _plot_arm_posterior_mass,
    "arm_fraction": _plot_arm_fraction,
    "posterior_peak_mean": _plot_posterior_peak_mean,
    "posterior_entropy_mean": _plot_posterior_entropy_mean,
    "hpd_widths": _plot_hpd_widths,
    "n_nonlocal_bins": _plot_n_nonlocal_bins,
    "time_since_stop_start_s": _plot_time_since_stop,
    "visit_recency_trials": _plot_visit_recency_trials,
    "reward_count_candidate_arm_last10_known_trials": _plot_cumulative_reward,
    "reward_fraction_candidate_arm_last10_visits": _plot_recent_reward_rate,
    "task_reward_probability_candidate_arm": _plot_task_reward_probability_candidate_arm,
    "distance_to_candidate_arm_cm": _plot_distance_to_candidate_arm,
    "candidate_arm_distance_rank": _plot_arm_dist_rank,
    "arm_mass_stackplot": _plot_arm_mass_stackplot,
    "replay_frac_by_role": _plot_replay_frac_by_role,
    "event_bars": _plot_event_bars,
    "stop_activity": _plot_stop_activity,
    "context_flags": _plot_context_flags,
    # Behavioral-model (beta-bernoulli) beliefs, arm grain.
    "reward_belief_mean_candidate_arm_preoutcome": _plot_reward_belief_mean_candidate_arm,
    "reward_belief_variance_candidate_arm_preoutcome": _plot_reward_belief_variance_candidate_arm,
    # Remaining fitted arm-grain predictors.
    "visit_recency_trials_event_leaf_centered": _plot_visit_recency_trials_event_leaf_centered,
    **{
        _c: _make_recency_spline_panel(_c, _i + 1)
        for _i, _c in enumerate(RECENCY_SPLINE_COLUMNS)
    },
    "is_candidate_arm_never_visited": _plot_arm_never_visited,
    "is_current_arm": _plot_is_current,
    "is_prev_arm": _plot_is_prev,
    "is_last_rewarded_noncurrent_arm": _plot_is_last_rewarded_elsewhere,
    # Patch ladder (P0–P4) design matrix.
    "reward_belief_mean_candidate_arm_preoutcome_patch": _plot_reward_belief_mean_candidate_arm_preoutcome_patch,
    "task_reward_probability_candidate_arm_patch": _plot_task_reward_probability_candidate_arm_patch,
    "reward_fraction_candidate_arm_last10_visits_patch": _plot_reward_fraction_candidate_arm_last10_visits_patch,
    "reward_count_candidate_arm_last10_known_trials_patch": _plot_reward_count_candidate_arm_last10_known_trials_patch,
    "is_current_patch": _plot_is_current_patch,
    "is_last_rewarded_noncurrent_patch": _plot_is_last_rewarded_noncurrent_patch,
    "is_prev_patch": _plot_is_prev_patch,
    # Family-A per-stop rate design matrix.
    "reward_lag_kernel": _plot_reward_lag_kernel,
    "reward_fraction_last10_known_trials": _plot_reward_rate_trial_k10,
    "did_last_completed_trip_switch_patch": _plot_last_completed_trip_switch,
}

# Panel groups for convenient dispatch — the covariate panels are opt-in
# because ``DEFAULT_PANELS`` stays a detector-style overview.
BEHAVIORAL_MODEL_PANELS: tuple[str, ...] = (
    "task_reward_probability_candidate_arm",
    "reward_belief_mean_candidate_arm_preoutcome",
    "reward_belief_variance_candidate_arm_preoutcome",
)
ARM_COVARIATE_PANELS: tuple[str, ...] = (
    "is_current_arm",
    "is_prev_arm",
    "is_last_rewarded_noncurrent_arm",
    "visit_recency_trials_event_leaf_centered",
    *RECENCY_SPLINE_COLUMNS,
    "is_candidate_arm_never_visited",
    "reward_fraction_candidate_arm_last10_visits",
    "reward_count_candidate_arm_last10_known_trials",
    "candidate_arm_distance_rank",
    # Programmed reward probability and reward-belief mean share a colorbar
    # group, so they must stay adjacent: a shared bar spans the block of axes
    # it serves, and an unrelated panel between them would be drawn over.
    "task_reward_probability_candidate_arm",
    "reward_belief_mean_candidate_arm_preoutcome",
    "reward_belief_variance_candidate_arm_preoutcome",
)
PATCH_COVARIATE_PANELS: tuple[str, ...] = (
    "is_current_patch",
    "is_prev_patch",
    "is_last_rewarded_noncurrent_patch",
    "reward_fraction_candidate_arm_last10_visits_patch",
    "reward_count_candidate_arm_last10_known_trials_patch",
    "task_reward_probability_candidate_arm_patch",
    "reward_belief_mean_candidate_arm_preoutcome_patch",
)
RATE_COVARIATE_PANELS: tuple[str, ...] = (
    "reward_lag_kernel",
    "reward_fraction_last10_known_trials",
    "did_last_completed_trip_switch_patch",
)

_EVENT_INTERVAL_CELL_PANELS = frozenset(
    {
        "arm_count_hard",
        "arm_count_soft",
        "arm_posterior_mass",
        "arm_fraction",
        "visit_recency_trials",
        "reward_count_candidate_arm_last10_known_trials",
        "reward_fraction_candidate_arm_last10_visits",
        "task_reward_probability_candidate_arm",
        "distance_to_candidate_arm_cm",
        "candidate_arm_distance_rank",
        "visit_recency_trials_event_leaf_centered",
        *RECENCY_SPLINE_COLUMNS,
        "is_candidate_arm_never_visited",
        "reward_belief_mean_candidate_arm_preoutcome",
        "reward_belief_variance_candidate_arm_preoutcome",
        "is_current_arm",
        "is_prev_arm",
        "is_last_rewarded_noncurrent_arm",
        *PATCH_COVARIATE_PANELS,
    }
)
_STRUCTURAL_NA_PANELS = frozenset(
    {
        "visit_recency_trials",
        "reward_count_candidate_arm_last10_known_trials",
        "reward_fraction_candidate_arm_last10_visits",
        "task_reward_probability_candidate_arm",
        "visit_recency_trials_event_leaf_centered",
        *RECENCY_SPLINE_COLUMNS,
        "reward_belief_mean_candidate_arm_preoutcome",
        "reward_belief_variance_candidate_arm_preoutcome",
    }
)


def _add_figure_semantic_legend(fig: Figure, panel_names: tuple[str, ...]) -> None:
    """Add one compact key for state encodings used across audit panels."""
    selected = set(panel_names)
    has_event_cells = bool(selected & _EVENT_INTERVAL_CELL_PANELS)
    has_discrete_cells = has_event_cells or bool(
        selected
        & {
            "reward_lag_kernel",
            "did_last_completed_trip_switch_patch",
            # Both draw hatched missing-input cells and so need the key; without
            # them a hatched band renders with nothing explaining it.
            "reward_fraction_last10_known_trials",
            "context_flags",
        }
    )
    if not (has_discrete_cells or "trial_rail" in selected):
        return

    handles: list[Any] = []
    if has_event_cells:
        handles.append(
            Patch(
                facecolor=plt.get_cmap(_PROBABILITY_CMAP)(0.65),
                edgecolor="none",
                label="event-sampled value",
            )
        )
    if has_discrete_cells:
        handles.extend(
            [
                Patch(
                    facecolor=_ZERO_CELL_COLOR,
                    edgecolor="#bdbdbd",
                    label="observed 0 / omission",
                ),
                Patch(
                    facecolor=_MISSING_CELL_COLOR,
                    edgecolor="#666666",
                    hatch="///",
                    label="missing input",
                ),
            ]
        )
    if selected & _STRUCTURAL_NA_PANELS:
        handles.append(
            Patch(
                facecolor=_STRUCTURAL_NA_COLOR,
                edgecolor="#bdbdbd",
                hatch="..",
                # Masked for reading, NOT dropped from the fit: production
                # writes 0.0 on trunk rows precisely so the conditional-Poisson
                # fitter does not discard the whole nine-arm event.
                label="trunk placeholder (fit as 0.0)",
            )
        )
    if "reward_lag_kernel" in selected:
        handles.append(
            Patch(
                facecolor=_REWARDED_COLOR,
                edgecolor="#bdbdbd",
                label="rewarded history",
            )
        )
    if "trial_rail" in selected:
        handles.extend(
            [
                Line2D(
                    [],
                    [],
                    marker="*",
                    linestyle="none",
                    markerfacecolor="gold",
                    markeredgecolor="black",
                    label="reward delivery",
                ),
                Line2D(
                    [],
                    [],
                    color="#9467bd",
                    linewidth=1.2,
                    label="patch-switch trial start",
                ),
            ]
        )

    # "outside upper center" is laid out by constrained_layout, which reserves
    # space for the key. An explicit bbox_to_anchor above the axes is not
    # managed, so the key was drawn over the top panel and pushed the suptitle
    # off the canvas.
    legend = fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=min(4, len(handles)),
        frameon=False,
        fontsize=_ANNOTATION_FONTSIZE,
        title=(
            "Blank time in event×arm rows = no event"
            if has_event_cells
            else "Cell-state key"
        ),
    )
    legend.get_title().set_fontsize(_ANNOTATION_FONTSIZE)


# Relative height weight per panel — inherits the ``plot_detector``
# template (posterior and raster get weight 3).
_PANEL_HEIGHTS: dict[str, float] = {
    "swr_lfp": 1.0,
    "mua": 1.0,
    "spike_times": 3.0,
    "state_probs": 1.0,
    "conditional_posterior": 3.0,
    "speed": 1.0,
    "trial_rail": 1.0,
    "arm_markers": 1.4,
    "mismatch_pointers": 1.4,
    "arm_posterior_heatmap": 2.0,
    "arm_count_hard": 1.4,
    "arm_count_soft": 1.4,
    "arm_posterior_mass": 1.4,
    "arm_fraction": 1.4,
    "posterior_peak_mean": 1.0,
    "posterior_entropy_mean": 1.0,
    "hpd_widths": 1.0,
    "n_nonlocal_bins": 1.0,
    "time_since_stop_start_s": 1.0,
    "visit_recency_trials": 1.4,
    "reward_count_candidate_arm_last10_known_trials": 1.4,
    "reward_fraction_candidate_arm_last10_visits": 1.4,
    "task_reward_probability_candidate_arm": 1.4,
    "distance_to_candidate_arm_cm": 1.4,
    "candidate_arm_distance_rank": 1.4,
    "arm_mass_stackplot": 1.4,
    "replay_frac_by_role": 1.2,
    "event_bars": 1.2,
    "stop_activity": 1.2,
    "context_flags": 1.4,
    "reward_belief_mean_candidate_arm_preoutcome": 1.4,
    "reward_belief_variance_candidate_arm_preoutcome": 1.4,
    "visit_recency_trials_event_leaf_centered": 1.4,
    **dict.fromkeys(RECENCY_SPLINE_COLUMNS, 1.4),
    "is_candidate_arm_never_visited": 1.4,
    "is_current_arm": 1.4,
    "is_prev_arm": 1.4,
    "is_last_rewarded_noncurrent_arm": 1.4,
    "reward_belief_mean_candidate_arm_preoutcome_patch": 1.0,
    "task_reward_probability_candidate_arm_patch": 1.0,
    "reward_fraction_candidate_arm_last10_visits_patch": 1.0,
    "reward_count_candidate_arm_last10_known_trials_patch": 1.0,
    "is_current_patch": 1.0,
    "is_last_rewarded_noncurrent_patch": 1.0,
    "is_prev_patch": 1.0,
    "reward_lag_kernel": 1.6,
    "reward_fraction_last10_known_trials": 1.0,
    "did_last_completed_trip_switch_patch": 0.5,
}


def register_panel(
    name: str,
    plotter: Callable[[Axes, _PanelContext], None],
    *,
    height: float = 1.0,
) -> None:
    """Register a new panel plotter under ``name``.

    Later viz chunks (arm markers, mismatch pointers, content
    heatmaps, scalar rows) add entries here so ``plot_session_tables``
    can dispatch them by name in the ``panels`` kwarg.
    """
    if name in _PANEL_REGISTRY:
        raise ValueError(f"panel {name!r} already registered")
    _PANEL_REGISTRY[name] = plotter
    _PANEL_HEIGHTS[name] = height


# ── Public entry point ──


def _validated_result_time(results: xr.Dataset | xr.DataArray) -> np.ndarray:
    """Return a finite, strictly increasing one-dimensional time coordinate."""
    if not isinstance(results, (xr.Dataset, xr.DataArray)):
        raise TypeError("results must be an xarray Dataset or DataArray")
    if "time" not in results.coords:
        raise ValueError("results must have a 'time' coordinate")
    time = np.asarray(results.coords["time"].values, dtype=float)
    if time.ndim != 1 or time.size < 2:
        raise ValueError("results.time must be one-dimensional with at least 2 samples")
    if not np.isfinite(time).all():
        raise ValueError("results.time must contain only finite values")
    if np.any(np.diff(time) <= 0):
        raise ValueError("results.time must be strictly increasing")
    return time


def _validated_panels(panels: Sequence[str]) -> tuple[str, ...]:
    """Materialize and validate the requested panel sequence."""
    if isinstance(panels, (str, bytes)):
        raise TypeError("panels must be a non-empty sequence of panel names")
    try:
        panel_names = tuple(panels)
    except TypeError as exc:
        raise TypeError("panels must be a non-empty sequence of panel names") from exc
    if not panel_names:
        raise ValueError("panels must contain at least one panel name")
    if not all(isinstance(panel, str) for panel in panel_names):
        raise TypeError("every panels entry must be a string")
    unknown = [panel for panel in panel_names if panel not in _PANEL_REGISTRY]
    if unknown:
        raise KeyError(
            f"unknown panel name(s): {unknown}. known: {sorted(_PANEL_REGISTRY)}"
        )
    return panel_names


def _validated_time_window(
    result_time: np.ndarray,
    time_window: tuple[float, float] | None,
) -> tuple[float, float]:
    """Resolve a requested window and reject empty or reversed intersections."""
    result_start = float(result_time[0])
    result_end = float(result_time[-1])
    if time_window is None:
        return result_start, result_end
    try:
        requested_start, requested_end = (float(value) for value in time_window)
    except (TypeError, ValueError) as exc:
        raise ValueError("time_window must be a finite (start, end) pair") from exc
    if not np.isfinite([requested_start, requested_end]).all():
        raise ValueError("time_window bounds must be finite")
    if requested_start >= requested_end:
        raise ValueError("time_window start must be less than its end")
    start = max(result_start, requested_start)
    end = min(result_end, requested_end)
    if start >= end:
        raise ValueError("time_window does not overlap results.time")
    return start, end


def _relative_time_ticks(
    start: float,
    end: float,
    reference: float,
    *,
    nbins: int = 8,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return readable ticks anchored to a relative-time reference.

    Matplotlib's default locator operates on the large absolute epoch-time
    coordinates.  Formatting those ticks after subtracting ``reference`` can
    therefore produce arbitrary offsets such as ``-0.0836782`` and omit the
    scientifically important zero.  Locate ticks in relative coordinates
    first, force zero when it lies in the window, and only then translate the
    positions back to the absolute coordinate system used by the data.
    """
    relative_start = float(start - reference)
    relative_end = float(end - reference)
    locator = MaxNLocator(nbins=nbins, steps=(1.0, 2.0, 2.5, 5.0, 10.0))
    offsets = np.asarray(locator.tick_values(relative_start, relative_end), dtype=float)
    tolerance = max(relative_end - relative_start, 1.0) * 1e-10
    offsets = offsets[
        (offsets >= relative_start - tolerance) & (offsets <= relative_end + tolerance)
    ]
    if (
        relative_start <= 0.0 <= relative_end
        and not np.isclose(offsets, 0.0, atol=tolerance, rtol=0.0).any()
    ):
        offsets = np.append(offsets, 0.0)
    offsets = np.unique(np.where(np.abs(offsets) <= tolerance, 0.0, offsets))
    labels = tuple(f"{offset:g}" for offset in offsets)
    return reference + offsets, labels


def _validate_event_intervals(event_table: pd.DataFrame) -> None:
    """Validate interval columns used by every public figure."""
    _require_columns(
        event_table,
        "event_table",
        (
            "event_id",
            "event_start_s",
            "event_end_s",
            "event_midpoint_s",
            "next_arm",
        ),
    )
    if (
        event_table["event_id"].isna().any()
        or event_table["event_id"].duplicated().any()
    ):
        raise ValueError("event_table.event_id must be unique and non-missing")
    starts = event_table["event_start_s"].to_numpy(dtype=float)
    ends = event_table["event_end_s"].to_numpy(dtype=float)
    midpoints = event_table["event_midpoint_s"].to_numpy(dtype=float)
    if not (
        np.isfinite(starts).all()
        and np.isfinite(ends).all()
        and np.isfinite(midpoints).all()
    ):
        raise ValueError("event_table event intervals and midpoints must be finite")
    if np.any(ends < starts):
        raise ValueError("event_table.event_end_s must be at or after event_start_s")
    if np.any((midpoints < starts) | (midpoints > ends)):
        raise ValueError("event_table.event_midpoint_s must lie inside each event")


def plot_session_tables(
    data: Mapping[str, Any],
    detector: Any,
    results: xr.Dataset | xr.DataArray,
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
    stop_table: pd.DataFrame,
    *,
    event_patch_table: pd.DataFrame | None = None,
    trials: pd.DataFrame | None = None,
    port_experiences: pd.DataFrame | None = None,
    track_graph: Any = None,
    time_window: tuple[float, float] | None = None,
    panels: Sequence[str] = DEFAULT_PANELS,
    figsize: tuple[float, float] = (12.0, 9.0),
    data_type: str = "clusterless",
    time_reference: float | None = None,
    xlabel: str | None = None,
    title: str | None = None,
    panel_group_labels: Mapping[str, str] | None = None,
    show_semantic_legend: bool = True,
    save_path: Path | None = None,
    swr_intervals: pd.DataFrame | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """Multi-panel time-aligned view of one session's canonical tables.

    Parameters
    ----------
    data : mapping
        ``load_data``-style dict. Must contain ``position_info``,
        ``hse_times``, ``multiunit_rate_zscore``,
        ``zscored_ripple_consensus_trace`` — the keys :func:`plot_detector`
        consumes except ``ripple_times``, which the ``swr_lfp`` panel takes
        from `swr_intervals` instead. ``trials`` and ``track_graph`` may be
        here or passed separately. Canonical behavioral-context panels also
        require ``port_experiences``.
    detector, results
        Fitted detector and decoding results — same roles as in
        :func:`plot_detector`. ``results`` may be a Dataset or DataArray as
        long as it has a finite, increasing ``time`` coordinate. Covariate-
        only figures may pass ``detector=None`` and provide
        ``data['environment']``; decoder-dependent panels still require their
        normal fitted-detector inputs.
    event_table, event_arm_table, event_sequence_table, stop_table
        Canonical tables produced by the per-grain builders in
        ``continuum_swr_replay.analysis.replay.tables_event`` /
        ``tables_stop`` / ``tables_arm_patch``.
    event_patch_table : pd.DataFrame, optional
        Patch-collapsed table from
        :func:`~continuum_swr_replay.analysis.replay.tables_arm_patch.build_event_patch_table`.
        Required only by the patch-grain panels in
        :data:`PATCH_COVARIATE_PANELS`; every other panel ignores it.
    trials : pd.DataFrame, optional
        Trial DataFrame; falls back to ``data['trials']`` when absent.
    port_experiences : pd.DataFrame, optional
        Canonical starting-port-plus-destinations experience stream; falls
        back to ``data['port_experiences']``. This is deliberately distinct
        from movement trials and is required to reproduce history covariates.
    time_window : (float, float), optional
        ``(start_time, end_time)`` in seconds. Default: full session.
    panels : sequence of str
        Panel names, top-to-bottom order. See :data:`DEFAULT_PANELS`.
        Unknown names raise :class:`KeyError`.
    time_reference : float, optional
        Display x ticks relative to this absolute timestamp. Scientific table
        lookups remain in absolute seconds; only tick formatting changes.
    xlabel, title : str, optional
        Explicit bottom-axis label and figure title. When ``time_reference`` is
        supplied, the default x label is ``"Time from reference [s]"``.
    panel_group_labels : mapping, optional
        ``{first_panel_name: group_heading}`` labels used to structure long
        audit figures without adding a separate axes row.
    show_semantic_legend : bool
        Add one figure-level key for event support, observed zero, missing
        input, structural trunk N/A, and any reward markers in the panel set.
    figsize, data_type, save_path : passthrough.
    swr_intervals : pd.DataFrame, optional
        The session's SWR intervals (``start_time``, ``end_time``) that the
        ``swr_lfp`` panel shades: the SWR source's, such as
        ``analysed_time.swr_intervals`` on the session's source-read
        ``data_over_time``, so the shading shows the SWRs the tables were
        built from. Required when ``swr_lfp`` is among `panels`; a
        ``data['ripple_times']`` (Spyglass's stored ripples) is never drawn.

    Returns
    -------
    fig : matplotlib.figure.Figure
    mismatch_table : pd.DataFrame
        Comparison between stored canonical ``next_arm`` and trial context
        recomputed at ``event_start_s`` through the production context and
        well-to-arm helpers. ``event_midpoint_s`` is retained as the
        plotting coordinate.
    """
    panel_names = _validated_panels(panels)
    if "swr_lfp" in panel_names:
        if swr_intervals is None:
            raise ValueError(
                "the swr_lfp panel shades the SWR source's intervals: pass "
                "swr_intervals= (analysed_time.swr_intervals on the session's "
                "source-read data_over_time)"
            )
        data = {**data, "ripple_times": swr_intervals}
    result_time = _validated_result_time(results)
    t_start, t_end = _validated_time_window(result_time, time_window)
    _validate_event_intervals(event_table)
    if time_reference is not None:
        try:
            time_reference = float(time_reference)
        except (TypeError, ValueError) as exc:
            raise ValueError("time_reference must be a finite timestamp") from exc
        if not np.isfinite(time_reference):
            raise ValueError("time_reference must be a finite timestamp")

    if trials is None:
        trials = data.get("trials")
        if trials is None:
            raise KeyError(
                "trials not provided and not present in ``data``; "
                "pass ``trials=`` explicitly."
            )
    if port_experiences is None:
        port_experiences = data.get("port_experiences")
        if port_experiences is None:
            raise KeyError(
                "port_experiences not provided and not present in ``data``; "
                "regenerate behavioral inputs or pass ``port_experiences=`` "
                "explicitly."
            )
    if track_graph is None:
        track_graph = data.get("track_graph")
        if track_graph is None:
            raise KeyError(
                "track_graph not provided and not present in ``data``; "
                "pass ``track_graph=`` explicitly."
            )

    time_slice = slice(t_start, t_end)

    env = None
    environments = getattr(detector, "environments", None)
    if environments is not None and len(environments) > 0:
        env = environments[0]
    if env is None:
        env = data.get("environment")
    ctx = _PanelContext(
        data=data,
        detector=detector,
        results=results,
        event_table=event_table,
        event_arm_table=event_arm_table,
        event_sequence_table=event_sequence_table,
        stop_table=stop_table,
        trials=trials,
        port_experiences=port_experiences,
        track_graph=track_graph,
        time_slice=time_slice,
        arm_palette=patch_grouped_arm_palette(),
        env=env,
        data_type=data_type,
        event_patch_table=event_patch_table,
        defer_colorbars=True,
    )

    heights = [_PANEL_HEIGHTS.get(panel, 1.0) for panel in panel_names]
    fig, axes = plt.subplots(
        len(panel_names),
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": heights},
        dpi=100,
    )
    # ``plt.subplots`` returns a single Axes when n=1; normalize to
    # an array so ``zip(panels, axes)`` works uniformly.
    axes = np.atleast_1d(axes)

    for panel_name, ax in zip(panel_names, axes, strict=True):
        _PANEL_REGISTRY[panel_name](ax, ctx)

    axes_order = {id(ax): index for index, ax in enumerate(axes)}
    for request in ctx._colorbar_requests.values():
        # A shared colorbar is anchored across the full span of the axes it
        # serves, so if those axes are not adjacent it is drawn over whatever
        # sits between them — including that panel's own colorbar, which then
        # reads against the wrong scale. Fall back to per-axes bars instead of
        # rendering a misleading one.
        indices = sorted(axes_order[id(ax)] for ax in request.axes)
        contiguous = indices == list(range(indices[0], indices[0] + len(indices)))
        targets = [request.axes] if contiguous else [[ax] for ax in request.axes]
        for target in targets:
            fig.colorbar(
                request.mappable,
                ax=target,
                shrink=0.7,
                pad=0.01,
                aspect=10,
                label=request.label,
                ticks=request.ticks,
                extend=request.extend,
            )

    if panel_group_labels is not None:
        for panel_name, group_label in panel_group_labels.items():
            if panel_name not in panel_names:
                raise ValueError(
                    f"panel_group_labels key {panel_name!r} is not in panels"
                )
            group_ax = axes[panel_names.index(panel_name)]
            group_ax.set_title(
                str(group_label),
                loc="left",
                fontsize=9,
                fontweight="bold",
                pad=6,
            )

    if time_reference is None:
        axes[-1].ticklabel_format(useOffset=False, style="plain", axis="x")
        default_xlabel = "Time [s]"
    else:
        reference = time_reference
        tick_positions, tick_labels = _relative_time_ticks(t_start, t_end, reference)
        axes[-1].xaxis.set_major_locator(FixedLocator(tick_positions))
        axes[-1].xaxis.set_major_formatter(FixedFormatter(tick_labels))
        if t_start <= reference <= t_end:
            for ax in axes:
                ax.axvline(
                    reference,
                    color="0.25",
                    linestyle=":",
                    linewidth=0.8,
                    alpha=0.8,
                    zorder=4,
                    label="t = 0 reference",
                )
        default_xlabel = "Time from reference [s]"
    axes[-1].set_xlabel(default_xlabel if xlabel is None else xlabel)
    axes[-1].set_xlim(t_start, t_end)
    if title is not None:
        # No explicit y: constrained_layout places the suptitle inside the
        # canvas and above the semantic key. A hard y=1.035 put it off-figure,
        # so the title was clipped in any inline render.
        fig.suptitle(title, fontsize=11, fontweight="bold")
    if show_semantic_legend:
        _add_figure_semantic_legend(fig, panel_names)

    # Vertical event-span guides align continuous/context and stop-grain rows.
    # Event-cell panels already encode support with their own cell widths and
    # deliberately skip the redundant gray overlay.
    et_windowed = event_table[
        (event_table["event_start_s"] <= t_end)
        & (event_table["event_end_s"] >= t_start)
    ]
    if len(et_windowed) > 0:
        event_starts = np.maximum(
            et_windowed["event_start_s"].to_numpy(dtype=float), t_start
        )
        event_ends = np.minimum(et_windowed["event_end_s"].to_numpy(dtype=float), t_end)
        # One axes-coordinate collection replaces one ``axvspan`` artist per
        # event and panel. This keeps full-session covariate groups tractable
        # while preserving the same event-span geometry and alpha.
        span_vertices = [
            ((start, 0.0), (end, 0.0), (end, 1.0), (start, 1.0))
            for start, end in zip(event_starts, event_ends, strict=True)
        ]
        for panel_name, ax in zip(panel_names, axes, strict=True):
            if panel_name in _EVENT_INTERVAL_CELL_PANELS:
                # These rows already encode event support with their own cell
                # widths; another gray span only darkens the same geometry.
                continue
            guides = PolyCollection(
                span_vertices,
                facecolors="black",
                edgecolors="none",
                alpha=0.04,
                transform=ax.get_xaxis_transform(),
                zorder=0,
                rasterized=True,
            )
            ax.add_collection(guides, autolim=False)

    sns.despine(offset=5)

    # If the mismatch-pointers panel ran it stashed the full table on
    # ctx. Otherwise compute once here so callers get the same shape
    # regardless of panel selection (useful for headless CI / audit).
    mismatch_table = ctx._mismatch_table
    if mismatch_table is None:
        mismatch_table = _compute_mismatch_table(
            event_table, trials, port_experiences, track_graph
        )

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, mismatch_table
