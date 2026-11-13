"""Stop-grain fitted-covariate panels (the Family-A rate design matrix)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from continuum_swr_replay.analysis.behavior.model_specs import N_REWARD_LAGS
from continuum_swr_replay.tables.columns import require_columns

from ._common import (
    _MISSING_CELL_COLOR,
    _REWARDED_COLOR,
    _ZERO_CELL_COLOR,
    _add_interval_cells,
    _PanelContext,
    _validate_binary,
)

# ── Stop-grain panels (the Family-A rate design matrix) ──
#
# These vary per STOP, not per (event, arm), so they are keyed on
# ``stop_start`` / ``stop_end`` rather than event windows. Rendering them
# on the same time axis is what makes it possible to see that a rate
# covariate is constant across every event inside one stop.


def _stop_view(ctx: _PanelContext, cols: Sequence[str]) -> pd.DataFrame:
    """Validated, chronologically sorted stop rows overlapping the window."""
    st = ctx.stop_table
    require_columns(st, ("stop_start", "stop_end", *cols), "stop_table")
    starts = st["stop_start"].to_numpy(dtype=float)
    ends = st["stop_end"].to_numpy(dtype=float)
    if not (np.isfinite(starts).all() and np.isfinite(ends).all()):
        raise ValueError("stop_table intervals must be finite")
    if np.any(ends < starts):
        raise ValueError("stop_table.stop_end must be at or after stop_start")
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    view = st[(st["stop_start"] <= hi) & (st["stop_end"] >= lo)]
    return view.sort_values("stop_start", kind="stable")


def _plot_reward_lag_kernel(ax: Axes, ctx: _PanelContext) -> None:
    """The persisted known-outcome lag vector: lags 1–10, per stop.

    One row per lag, one column block per stop. Blue = rewarded, pale =
    observed omission, and dark-gray hatch = insufficient trial history.
    This is the stop metadata the reward-history basis is built from; the
    missing-history cells are the ones the basis normalizes over (and, when a
    bump's whole support is missing, flags with ``reward_history_incomplete``).
    A plotting diagnostic only: it does not alter any fitting cohort.
    """
    lag_cols = [f"known_reward_outcome_lag_{j}" for j in range(1, N_REWARD_LAGS + 1)]
    view = _stop_view(ctx, lag_cols)

    ax.set_ylabel("reward lag\n(trials back)")
    ax.set_ylim(0.5, N_REWARD_LAGS + 0.5)
    ax.set_yticks([1, 5, 10])
    ax.invert_yaxis()

    if len(view) == 0:
        ax.text(
            0.5,
            0.5,
            "no stops in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="grey",
        )
        return

    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    starts = view["stop_start"].to_numpy()
    ends = view["stop_end"].to_numpy()
    min_width = max((hi - lo) * 0.002, 1e-6)
    for lag_idx, col in enumerate(lag_cols, start=1):
        vals = view[col].to_numpy(dtype=float)
        _validate_binary(vals, f"stop_table.{col}")
        finite = np.isfinite(vals)
        rows = np.full(len(vals), lag_idx, dtype=float)
        colors = [
            _REWARDED_COLOR if value == 1.0 else _ZERO_CELL_COLOR
            for value in vals[finite]
        ]
        _add_interval_cells(
            ax,
            starts[finite],
            ends[finite],
            rows[finite],
            colors,
            min_width=min_width,
            edgecolors="#bdbdbd",
            alpha=0.9,
        )
        missing = ~finite
        _add_interval_cells(
            ax,
            starts[missing],
            ends[missing],
            rows[missing],
            [_MISSING_CELL_COLOR] * int(missing.sum()),
            min_width=min_width,
            edgecolors="#666666",
            hatch="///",
        )


def _plot_reward_rate_trial_k10(ax: Axes, ctx: _PanelContext) -> None:
    """Rewarded fraction among up to ten completed trials, per stop.

    The value averages the available most-recent trials (maximum ten) at
    stop start and is missing only before the first completed trial. It is
    persisted stop metadata, not a ladder predictor: the rate models encode
    reward history with the raised-cosine basis instead, which spans the same
    ten known outcomes without collapsing them to one unweighted mean.
    """
    view = _stop_view(ctx, ["reward_fraction_last10_known_trials"])
    ax.set_ylabel("reward fraction\nprior ≤10 trials")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.0, color="grey", linewidth=0.3, zorder=0)

    if len(view) == 0:
        ax.text(
            0.5,
            0.5,
            "no stops in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="grey",
        )
        return

    starts = view["stop_start"].to_numpy(dtype=float)
    ends = view["stop_end"].to_numpy(dtype=float)
    vals = view["reward_fraction_last10_known_trials"].to_numpy(dtype=float)
    finite = np.isfinite(vals)
    missing = ~finite
    if missing.any():
        lo, hi = ctx.time_slice.start, ctx.time_slice.stop
        _add_interval_cells(
            ax,
            np.maximum(starts[missing], lo),
            np.minimum(ends[missing], hi),
            np.full(int(missing.sum()), 0.5),
            [_MISSING_CELL_COLOR] * int(missing.sum()),
            min_width=max((hi - lo) * 0.002, 1e-6),
            edgecolors="#666666",
            alpha=0.35,
            hatch="///",
        )
    if finite.any():
        lo, hi = ctx.time_slice.start, ctx.time_slice.stop
        clipped_starts = np.maximum(starts[finite], lo)
        clipped_ends = np.minimum(ends[finite], hi)
        nonempty = clipped_ends >= clipped_starts
        segments = [
            ((start, value), (end, value))
            for start, end, value in zip(
                clipped_starts[nonempty],
                clipped_ends[nonempty],
                vals[finite][nonempty],
                strict=True,
            )
        ]
        ax.add_collection(
            LineCollection(
                segments,
                colors="#8c564b",
                linewidths=1.5,
                zorder=2,
            )
        )
        mids = 0.5 * (clipped_starts[nonempty] + clipped_ends[nonempty])
        ax.scatter(
            mids,
            vals[finite][nonempty],
            s=10,
            color="#8c564b",
            zorder=3,
        )


def _plot_last_completed_trip_switch(ax: Axes, ctx: _PanelContext) -> None:
    """Per-stop patch-switch context with explicit zero/missing states."""
    view = _stop_view(ctx, ["did_last_completed_trip_switch_patch"])
    ax.set_ylabel("last trip\nswitched patch")
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])

    if len(view) == 0:
        ax.text(
            0.5,
            0.5,
            "no stops in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="grey",
        )
        return

    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    starts = view["stop_start"].to_numpy(dtype=float)
    ends = view["stop_end"].to_numpy(dtype=float)
    vals = view["did_last_completed_trip_switch_patch"].to_numpy(dtype=float)
    _validate_binary(vals, "stop_table.did_last_completed_trip_switch_patch")
    finite = np.isfinite(vals)
    _add_interval_cells(
        ax,
        starts[finite],
        ends[finite],
        np.zeros(int(finite.sum())),
        ["#9467bd" if value == 1.0 else _ZERO_CELL_COLOR for value in vals[finite]],
        min_width=max((hi - lo) * 0.002, 1e-6),
        edgecolors="#bdbdbd",
        alpha=0.9,
    )
    missing = ~finite
    _add_interval_cells(
        ax,
        starts[missing],
        ends[missing],
        np.zeros(int(missing.sum())),
        [_MISSING_CELL_COLOR] * int(missing.sum()),
        min_width=max((hi - lo) * 0.002, 1e-6),
        edgecolors="#666666",
        hatch="///",
    )
