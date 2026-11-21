#!/usr/bin/env python
"""Compare the shared fitter without a penalty with the current fitters on real sessions.

For one session each of chimi, j16 and wilbur (the first in session order),
read-only from the canonical tables:

- the per-bin rate model ``M7r_time_histcos_rew`` on the at-well stop cohort's
  at-risk design for the SWR outcome, as the per-bin ladder builds and freezes
  it, fit by the current per-bin fitter (``fit_per_bin_glm``, statsmodels'
  Poisson GLM) and by the shared fitter's Poisson likelihood with the same
  design matrix and offset;
- the content model ``M0a`` (arm fixed effects) on the at-well event cohort's
  arm fractions, fit by the current content fitter (``fit_conditional_poisson``)
  and by the shared fitter's grouped softmax on the same events.

Neither shared fit has a penalty. The script prints, for each session and
model, the number of coefficients and the largest relative difference
``|ours - current| / |current|``, and the design columns that are zero on
every row (not identified; the current fitter gives them zero, the shared
fitter leaves them out).

Usage:
    python scripts/checks/compare_regression_to_current_fitters.py
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np
import pandas as pd
import patsy

from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    select_event_cohort,
    select_stop_cohort,
)
from continuum_swr_replay.analysis.regression.likelihoods import (
    GroupedSoftmax,
    Poisson,
)
from continuum_swr_replay.analysis.regression.newton import penalized_newton
from continuum_swr_replay.analysis.replay_models.fitting import fit_conditional_poisson
from continuum_swr_replay.analysis.replay_models.model_specs import (
    MODEL_HIERARCHY,
    MODEL_PER_BIN_RATE,
    PerBinRateModelSpec,
    add_arm_fixed_effects,
)
from continuum_swr_replay.analysis.replay_models.per_bin_design import (
    attach_is_swr_to_onsets,
)
from continuum_swr_replay.analysis.replay_models.rate.duration_aware_history import (
    attach_event_ends,
)
from continuum_swr_replay.analysis.replay_models.rate.per_bin_rate_ladder import (
    ladder_design,
)
from continuum_swr_replay.analysis.replay_models.rate.per_bin_rate_model import (
    fit_per_bin_rate_model,
    freeze_per_bin_cohort,
)
from continuum_swr_replay.paths import replay_content_dir, table_path
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

ANIMALS = ("chimi", "j16", "wilbur")
PER_BIN_MODEL = "M7r_time_histcos_rew"
PER_BIN_OUTCOME = "swr"
CONTENT_MODEL = "M0a"


def per_bin_coefficients(
    design: pd.DataFrame, spec: PerBinRateModelSpec
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """A per-bin rung's coefficients by the current fitter and the shared one.

    A design column that is zero on every row is not identified. The current
    fitter (statsmodels, a pseudo-inverse) gives it the minimum-norm
    coefficient, zero, and the other coefficients of the fit without it; the
    shared fitter refuses an unidentified design, so it is fit without such
    columns. On the at-well cohort ``reward_history_incomplete`` is one: every
    stop there has a complete reward history.

    Parameters
    ----------
    design : pd.DataFrame, shape (n_bins, n_columns)
        A frozen per-bin design (``freeze_per_bin_cohort``).
    spec : PerBinRateModelSpec

    Returns
    -------
    ours, current : pd.Series
        The coefficients of the columns that are not zero on every row,
        indexed by the design's column names (patsy's, as statsmodels names
        them).
    left_out : pd.Series
        The current fitter's coefficients of the columns that are.

    Raises
    ------
    ValueError
        If the current fitter skips the design or either fit does not converge.
    """
    fit = fit_per_bin_rate_model(design, spec)
    if fit is None or not fit["converged"]:
        raise ValueError(f"{spec.name}: the current per-bin fitter did not fit")
    X = patsy.dmatrix(spec.formula().split("~", 1)[1], design, return_type="dataframe")
    current = pd.Series({name: fit[f"coef_{name}"] for name in X.columns})
    zero = (X == 0).all(axis=0)
    X = X.loc[:, ~zero]
    y = design["count"].to_numpy(dtype=float)
    exposure = design["exposure"].to_numpy(dtype=float)
    start = np.zeros(X.shape[1])
    start[list(X.columns).index("Intercept")] = np.log(y.sum() / exposure.sum())
    ours = penalized_newton(
        Poisson(),
        X.to_numpy(dtype=float),
        y,
        np.log(exposure),
        np.zeros((X.shape[1], X.shape[1])),
        start,
    )
    if not ours.converged:
        raise ValueError(
            f"{spec.name}: the shared fitter did not converge ({ours.status})"
        )
    return (
        pd.Series(ours.beta, index=X.columns),
        current[X.columns],
        current[zero.to_numpy()],
    )


def content_coefficients(
    long_df: pd.DataFrame,
    predictors: Sequence[str],
    response_col: str = "arm_fraction",
    event_col: str = "event_id",
) -> tuple[pd.Series, pd.Series]:
    """A content rung's coefficients by the current fitter and the shared one.

    The shared fit uses the events the current fitter's likelihood uses: those
    with a positive total response that is not the same on every row.

    Parameters
    ----------
    long_df : pd.DataFrame, shape (n_events * n_arms, n_columns)
        One row per (event, arm), complete in the response and the predictors.
    predictors : sequence of str

    Returns
    -------
    ours, current : pd.Series
        Indexed by ``predictors``.

    Raises
    ------
    ValueError
        If the current fitter does not fit, leaves a term unidentified or does
        not converge, or the shared fit does not converge.
    """
    predictors = list(predictors)
    fit = fit_conditional_poisson(
        long_df, predictors, response_col=response_col, event_col=event_col
    )
    if fit is None or not fit["converged"] or fit["unidentified_terms"]:
        raise ValueError(f"the current content fitter did not fit {predictors}")
    frame = long_df.sort_values(event_col, kind="stable")
    by_event = frame.groupby(event_col)[response_col]
    used = (by_event.transform("sum") > 0) & (by_event.transform("std") > 0)
    frame = frame[used]
    ours = penalized_newton(
        GroupedSoftmax(),
        frame[predictors].to_numpy(dtype=float),
        frame[response_col].to_numpy(dtype=float),
        frame[event_col].to_numpy(),
        np.zeros((len(predictors), len(predictors))),
        np.zeros(len(predictors)),
    )
    if not ours.converged:
        raise ValueError(f"the shared grouped softmax did not converge ({ours.status})")
    current = pd.Series({name: fit[f"coef_{name}"] for name in predictors})
    return pd.Series(ours.beta, index=predictors), current


def largest_relative_difference(ours: pd.Series, current: pd.Series) -> float:
    """``max |ours - current| / |current|`` over the coefficients."""
    return float(np.max(np.abs(ours - current) / np.abs(current)))


def _first_session(table: pd.DataFrame, animal: str) -> tuple:
    keys = table.loc[table["animal"] == animal, list(SESSION_COLUMNS)]
    return tuple(keys.drop_duplicates().sort_values(list(SESSION_COLUMNS)).iloc[0])


def _rows_of(table: pd.DataFrame, keys: tuple) -> pd.DataFrame:
    mask = np.logical_and.reduce(
        [table[c] == k for c, k in zip(SESSION_COLUMNS, keys, strict=True)]
    )
    return table[mask]


def main() -> int:
    rc = replay_content_dir()
    stop, _ = select_stop_cohort(read_table(table_path("stop_table", rc)))
    event_table = read_table(table_path("event_table", rc))
    onsets = attach_event_ends(
        attach_is_swr_to_onsets(
            read_table(table_path("event_onset_table", rc)), event_table
        ),
        event_table,
    )
    event_arm, _ = select_event_cohort(read_table(table_path("event_arm_table", rc)))
    event_arm = add_arm_fixed_effects(event_arm.copy())
    spec = MODEL_PER_BIN_RATE[PER_BIN_MODEL]
    predictors = list(MODEL_HIERARCHY[CONTENT_MODEL].predictors)

    worst = 0.0
    for animal in ANIMALS:
        keys = _first_session(stop, animal)
        design = ladder_design(
            _rows_of(stop, keys), _rows_of(onsets, keys), PER_BIN_OUTCOME
        )
        frozen, _ = freeze_per_bin_cohort(design, MODEL_PER_BIN_RATE.values())
        ours, current, left_out = per_bin_coefficients(frozen, spec)
        rate = largest_relative_difference(ours, current)
        print(
            f"{'_'.join(map(str, keys))} {PER_BIN_MODEL} ({PER_BIN_OUTCOME}): "
            f"{len(frozen)} bins, {int(frozen['count'].sum())} events, "
            f"{len(ours)} coefficients, largest relative difference {rate:.1e}"
        )
        for name, value in left_out.items():
            print(
                f"  {name} is zero on every bin; current fitter's coefficient {value:.1e}"
            )
        content_keys = _first_session(event_arm, animal)
        rows = _rows_of(event_arm, content_keys)
        ours, current = content_coefficients(rows, predictors)
        content = largest_relative_difference(ours, current)
        print(
            f"{'_'.join(map(str, content_keys))} {CONTENT_MODEL}: "
            f"{rows['event_id'].nunique()} events, {len(ours)} coefficients, "
            f"largest relative difference {content:.1e}"
        )
        worst = max(worst, rate, content)
    print(f"largest relative difference over all fits: {worst:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
