"""The audit's arrival-clock and outcome-boundary comparisons.

Two quantities are read off a rate refit that no nested test can supply. The
first is the shape of the fitted time course under each clock: the rungs are fit
to different row sets, so only a scale-free comparison (peak position, and the
KS distance between the two normalized curves on relative time) is licensed. The
second is the reward-history block before and after the per-bin features are
recomputed at the outcome boundary: same rung, same basis, so the coefficients
are paired term for term.

Every fixture here plants a curve whose peak is known analytically and encodes
it as spline coefficients by least squares, so the assertions are on recovered
values rather than on the table merely being non-empty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from patsy import dmatrix

from continuum_swr_replay.analysis.replay.model_specs import (
    ARRIVAL_CLOCK_COLUMN,
    ARRIVAL_SPLINE_KNOTS,
    REWARD_HISTORY_COLUMNS,
    SPLINE_KNOTS,
    TMAX,
    TMAX_ARRIVAL,
)
from scripts.qa.compare_fits import (
    ARRIVAL_TIME_RUNG,
    BOUNDARY_REWARD_RUNG,
    STOP_TIME_RUNG,
    arrival_clock_cv_table,
    arrival_clock_shape_table,
    reward_boundary_shift_table,
)

SESSIONS = [("chimi", "20200212", "02_r1"), ("j16", "20210706", "02_r1")]


def _basis(clock: str, knots: tuple[float, ...], tmax: float, x: np.ndarray):
    """The ladder's spline basis at ``x``, with its patsy column names."""
    design = dmatrix(
        f"bs({clock}, knots={knots}, lower_bound=0.0, upper_bound={tmax}) - 1",
        {clock: x},
        return_type="dataframe",
    )
    return np.asarray(design), list(design.columns)


def _encode_bump(
    clock: str, knots: tuple[float, ...], tmax: float, peak: float, width: float
):
    """Spline coefficients whose fitted curve is a Gaussian bump at ``peak``.

    Returns the intercept, the coefficients in basis order, and the basis column
    names, so a fixture can write exactly the columns a real fit would carry.
    """
    grid = np.linspace(0.0, tmax, 2001)
    target = np.log(0.5 + 4.0 * np.exp(-0.5 * ((grid - peak) / width) ** 2))
    basis, names = _basis(clock, knots, tmax, grid)
    augmented = np.column_stack([np.ones_like(grid), basis])
    solution, *_ = np.linalg.lstsq(augmented, target, rcond=None)
    return float(solution[0]), solution[1:], names


def _time_rung_rows(
    model: str,
    clock: str,
    knots: tuple[float, ...],
    tmax: float,
    peak: float,
    width: float,
    cv_bits: list[float],
) -> pd.DataFrame:
    intercept, coefficients, names = _encode_bump(clock, knots, tmax, peak, width)
    rows = []
    for (animal, date, epoch), bits in zip(SESSIONS, cv_bits, strict=True):
        row = {
            "animal": animal,
            "session_date": date,
            "epoch": epoch,
            "model": model,
            "outcome": "non_local",
            "coef_Intercept": intercept,
            "cv_bits_per_event": bits,
        }
        row.update(
            {
                f"coef_{name}": value
                for name, value in zip(names, coefficients, strict=True)
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _two_clock_fits(
    stop_peak: float,
    stop_width: float,
    arrival_peak: float,
    arrival_width: float,
    stop_cv: list[float] | None = None,
    arrival_cv: list[float] | None = None,
) -> pd.DataFrame:
    stop = _time_rung_rows(
        STOP_TIME_RUNG,
        "tc",
        SPLINE_KNOTS,
        TMAX,
        stop_peak,
        stop_width,
        stop_cv if stop_cv is not None else [0.10, 0.12],
    )
    arrival = _time_rung_rows(
        ARRIVAL_TIME_RUNG,
        ARRIVAL_CLOCK_COLUMN,
        ARRIVAL_SPLINE_KNOTS,
        TMAX_ARRIVAL,
        arrival_peak,
        arrival_width,
        arrival_cv if arrival_cv is not None else [0.20, 0.24],
    )
    return pd.concat([stop, arrival], ignore_index=True)


def test_shape_table_recovers_each_clocks_planted_peak():
    """A bump planted at 1.0 s on tc and 2.0 s after arrival is read back."""
    fits = _two_clock_fits(
        stop_peak=1.0, stop_width=0.5, arrival_peak=2.0, arrival_width=1.0
    )
    table = arrival_clock_shape_table(fits).set_index("outcome")
    row = table.loc["non_local"]
    assert row["n_sessions_stop"] == 2
    assert row["n_sessions_arrival"] == 2
    assert row["peak_s_stop"] == pytest.approx(1.0, abs=0.1)
    assert row["peak_s_arrival"] == pytest.approx(2.0, abs=0.1)
    assert row["peak_frac_window_stop"] == pytest.approx(0.25, abs=0.03)
    assert row["peak_frac_window_arrival"] == pytest.approx(0.25, abs=0.03)


def test_matched_relative_shape_gives_a_near_zero_ks_distance():
    """Same bump at the same fraction of each window: the shapes agree."""
    fits = _two_clock_fits(
        stop_peak=1.0, stop_width=0.5, arrival_peak=2.0, arrival_width=1.0
    )
    ks = arrival_clock_shape_table(fits).set_index("outcome").loc["non_local"]
    assert ks["ks_distance"] < 0.05


def test_ks_distance_separates_clocks_that_disagree():
    """An early stop-clock bump against a late arrival bump is far apart."""
    fits = _two_clock_fits(
        stop_peak=0.5, stop_width=0.4, arrival_peak=6.0, arrival_width=0.8
    )
    row = arrival_clock_shape_table(fits).set_index("outcome").loc["non_local"]
    assert row["peak_s_arrival"] == pytest.approx(6.0, abs=0.15)
    assert row["ks_distance"] > 0.5


def test_shape_table_raises_when_the_saved_basis_is_a_different_size():
    """A fit made with other knots is not silently reconstructed."""
    fits = _two_clock_fits(
        stop_peak=1.0, stop_width=0.5, arrival_peak=2.0, arrival_width=1.0
    )
    dropped = [c for c in fits.columns if c.startswith("coef_bs(tc,")][0]
    with pytest.raises(ValueError, match="different knots"):
        arrival_clock_shape_table(fits.drop(columns=[dropped]))


def test_cv_table_counts_finite_fits_and_pairs_within_session():
    """One arrival fit loses its CV, so it counts in neither median nor pair."""
    fits = _two_clock_fits(
        stop_peak=1.0,
        stop_width=0.5,
        arrival_peak=2.0,
        arrival_width=1.0,
        stop_cv=[0.10, 0.20],
        arrival_cv=[0.30, np.nan],
    )
    row = arrival_clock_cv_table(fits).set_index("outcome").loc["non_local"]
    assert row["n_fits_stop"] == 2
    assert row["n_fits_arrival"] == 2
    assert row["pct_cv_finite_stop"] == pytest.approx(100.0)
    assert row["pct_cv_finite_arrival"] == pytest.approx(50.0)
    assert row["median_cv_bits_stop"] == pytest.approx(0.15)
    assert row["median_cv_bits_arrival"] == pytest.approx(0.30)
    assert row["n_paired_cv_finite"] == 1
    assert row["median_delta_bits_arrival_minus_stop"] == pytest.approx(0.20)


def _reward_rung(coefficients: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for index, (animal, date, epoch) in enumerate(SESSIONS):
        row = {
            "animal": animal,
            "session_date": date,
            "epoch": epoch,
            "model": BOUNDARY_REWARD_RUNG,
            "outcome": "non_local",
        }
        row.update(
            {f"coef_{term}": values[index] for term, values in coefficients.items()}
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_reward_boundary_table_pairs_coefficients_within_session():
    """One session's immediate component moves, the other's does not."""
    terms = REWARD_HISTORY_COLUMNS
    before = _reward_rung(
        {terms[0]: [0.40, 0.60], terms[1]: [0.10, 0.20], terms[2]: [0.0, 0.0]}
    )
    after = _reward_rung(
        {terms[0]: [0.50, 0.60], terms[1]: [0.10, 0.20], terms[2]: [0.0, 0.0]}
    )
    table = reward_boundary_shift_table(before, after).set_index("term")
    immediate = table.loc[terms[0]]
    assert immediate["n_paired"] == 2
    assert immediate["median_before"] == pytest.approx(0.50)
    assert immediate["median_after"] == pytest.approx(0.55)
    assert immediate["median_delta"] == pytest.approx(0.05)
    assert immediate["pct_sessions_moved"] == pytest.approx(50.0)
    unchanged = table.loc[terms[1]]
    assert unchanged["median_delta"] == pytest.approx(0.0)
    assert unchanged["pct_sessions_moved"] == pytest.approx(0.0)
    assert unchanged["signrank_p"] == "—"


def test_reward_boundary_table_reports_a_side_that_lacks_the_rung():
    """A backup without the rung is an empty side, not a missing section."""
    terms = REWARD_HISTORY_COLUMNS
    after = _reward_rung(
        {terms[0]: [0.5, 0.6], terms[1]: [0.1, 0.2], terms[2]: [0.0, 0.0]}
    )
    before = after.assign(model="M7_time_hist_rew")
    table = reward_boundary_shift_table(before, after).set_index("term")
    immediate = table.loc[terms[0]]
    assert immediate["n_sessions_before"] == 0
    assert immediate["n_sessions_after"] == 2
    assert immediate["n_paired"] == 0
    assert np.isnan(immediate["median_delta"])
