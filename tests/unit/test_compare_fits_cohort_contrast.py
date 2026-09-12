"""The audit that places one rate ladder's three stop cohorts side by side.

The cohort-contrast mode exists to answer one question: does a rate number move
because the effect moved, or because the population did? So every assertion here
is on a planted value -- a known reward coefficient, a known AIC gap, a known
spline peak, a known cohort size -- recovered per cohort, rather than on a table
merely being produced. The fixtures are the smallest frames carrying the columns
a real per-bin fit carries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from patsy import dmatrix

from continuum_swr_replay.analysis.replay.model_specs import (
    REWARD_HISTORY_COLUMNS,
    SPLINE_KNOTS,
    TMAX,
)
from scripts.qa.compare_fits import (
    COHORT_ORDER,
    FULL_MODEL,
    IMMEDIATE_COEF,
    STOP_TIME_RUNG,
    _animal_balanced,
    build_cohort_report,
    build_parser,
    cohort_burst_peak_table,
    cohort_contrast_rung_table,
    cohort_delta_aic_table,
    cohort_paired_swr_table,
    cohort_reward_rr_table,
    cohort_size_table,
    collect_cohort_inputs,
    run_cohort_contrast_mode,
)

# Two animals x two sessions, so an animal-balanced median is not the same
# number as a session median and the test can tell them apart.
SESSIONS = [
    ("chimi", "20200212", "02_r1"),
    ("chimi", "20200213", "02_r1"),
    ("j16", "20210706", "02_r1"),
    ("j16", "20210707", "02_r1"),
]
POPULATIONS = ("non_local", "non_local_swr", "non_local_no_swr")


def _spline_bump(peak: float, width: float = 0.4) -> tuple[float, np.ndarray, list]:
    """Intercept + stop-clock spline coefficients for a bump peaking at ``peak``."""
    grid = np.linspace(0.0, TMAX, 2001)
    design = dmatrix(
        f"bs(tc, knots={tuple(SPLINE_KNOTS)}, lower_bound=0.0, upper_bound={TMAX}) - 1",
        {"tc": grid},
        return_type="dataframe",
    )
    basis = np.asarray(design)
    target = np.log(0.5 + 4.0 * np.exp(-0.5 * ((grid - peak) / width) ** 2))
    solution, *_ = np.linalg.lstsq(
        np.column_stack([np.ones_like(grid), basis]), target, rcond=None
    )
    return float(solution[0]), solution[1:], list(design.columns)


def _per_bin_fits(
    *,
    immediate_by_population: dict[str, float],
    aic: dict[tuple[str, str], float],
    n_stops: int,
    n_bins: int,
    n_events: int,
    peak: float,
) -> pd.DataFrame:
    """One cohort's per-bin ladder: the null, the two reward steps and the time rung.

    ``immediate_by_population`` plants ``coef_reward_history_rcos_0`` on the full
    rung; ``aic`` plants one AIC per (rung, population); ``peak`` plants the
    within-stop time course of ``M5_time``.
    """
    intercept, coefficients, names = _spline_bump(peak)
    rungs = ("M0_const", "M1_rew", "M6r_time_histcos", FULL_MODEL, STOP_TIME_RUNG)
    rows = []
    for animal, date, epoch in SESSIONS:
        for population in POPULATIONS:
            for rung in rungs:
                row = {
                    "animal": animal,
                    "session_date": date,
                    "epoch": epoch,
                    "outcome": population,
                    "model": rung,
                    "stop_cohort": "planted",
                    "n_stops": n_stops,
                    "n_bins": n_bins,
                    "n_events": n_events,
                    "aic": aic.get((rung, population), 100.0),
                    "separated": False,
                }
                if rung == FULL_MODEL:
                    row[IMMEDIATE_COEF] = immediate_by_population[population]
                    # The paired test reconstructs the whole three-bump kernel,
                    # so the later components have to exist even when only the
                    # immediate one carries a planted value.
                    for column in REWARD_HISTORY_COLUMNS[1:]:
                        row[f"coef_{column}"] = 0.0
                if rung == STOP_TIME_RUNG:
                    row["coef_Intercept"] = intercept
                    row.update(
                        {
                            f"coef_{name}": value
                            for name, value in zip(names, coefficients, strict=True)
                        }
                    )
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def cohort_fits() -> dict[str, pd.DataFrame]:
    """Three cohorts differing only in the values this audit is supposed to read."""
    return {
        "at_well": _per_bin_fits(
            immediate_by_population={
                "non_local": np.log(2.0),
                "non_local_swr": np.log(3.0),
                "non_local_no_swr": np.log(1.0),
            },
            aic={(FULL_MODEL, p): 90.0 for p in POPULATIONS}  # 10 better than M6r's 100
            | {("M1_rew", p): 95.0 for p in POPULATIONS},
            n_stops=100,
            n_bins=1000,
            n_events=50,
            peak=1.5,
        ),
        "all_stops": _per_bin_fits(
            immediate_by_population={
                "non_local": np.log(1.5),
                "non_local_swr": np.log(2.0),
                "non_local_no_swr": np.log(1.2),
            },
            aic={(FULL_MODEL, p): 104.0 for p in POPULATIONS},
            n_stops=130,
            n_bins=1400,
            n_events=70,
            peak=2.5,
        ),
        "in_transit": _per_bin_fits(
            immediate_by_population={
                "non_local": np.log(0.8),
                "non_local_swr": np.log(0.9),
                "non_local_no_swr": np.log(0.7),
            },
            aic={(FULL_MODEL, p): 99.0 for p in POPULATIONS},
            n_stops=30,
            n_bins=400,
            n_events=20,
            peak=2.0,
        ),
    }


def _row(table: pd.DataFrame, **keys) -> pd.Series:
    mask = np.ones(len(table), dtype=bool)
    for column, value in keys.items():
        mask &= (table[column] == value).to_numpy()
    matched = table[mask]
    assert len(matched) == 1, f"{keys} matched {len(matched)} rows"
    return matched.iloc[0]


def test_animal_balanced_medians_within_animal_first():
    """One animal's extra sessions must not outvote the other animal."""
    frame = pd.DataFrame(
        {
            "animal": ["a", "a", "a", "b"],
            "coef_x": [10.0, 10.0, 10.0, -2.0],
        }
    )
    median, n_positive, n_animals = _animal_balanced(frame, "coef_x")
    assert (n_positive, n_animals) == (1, 2)
    assert median == pytest.approx(4.0)  # median of [10, -2], not of the 4 sessions


def test_animal_balanced_reports_nothing_when_the_term_is_absent():
    median, n_positive, n_animals = _animal_balanced(
        pd.DataFrame({"animal": ["a"]}), "coef_missing"
    )
    assert np.isnan(median)
    assert (n_positive, n_animals) == (0, 0)


def test_cohort_size_table_totals_the_frozen_cohort(cohort_fits):
    table = cohort_size_table(cohort_fits)
    row = _row(table, outcome="non_local", stop_cohort="at_well")
    assert row["n_sessions"] == len(SESSIONS)
    assert row["n_stops"] == 100 * len(SESSIONS)
    assert row["n_bins"] == 1000 * len(SESSIONS)
    assert row["n_event_onsets"] == 50 * len(SESSIONS)
    assert _row(table, outcome="non_local", stop_cohort="in_transit")["n_stops"] == (
        30 * len(SESSIONS)
    )


def test_reward_rr_recovers_the_planted_rate_ratio(cohort_fits):
    table = cohort_reward_rr_table(cohort_fits)
    assert _row(table, outcome="non_local_swr", stop_cohort="at_well")[
        "rr_immediate"
    ] == pytest.approx(3.0)
    assert _row(table, outcome="non_local", stop_cohort="all_stops")[
        "rr_immediate"
    ] == pytest.approx(1.5)
    below_one = _row(table, outcome="non_local", stop_cohort="in_transit")
    assert below_one["rr_immediate"] == pytest.approx(0.8)
    assert below_one["animals_positive"] == "0/2"
    assert (
        _row(table, outcome="non_local", stop_cohort="at_well")["animals_positive"]
        == "2/2"
    )


def test_paired_swr_test_reads_the_planted_direction(cohort_fits):
    table = cohort_paired_swr_table(cohort_fits)
    assert list(table["stop_cohort"]) == ["at_well", "all_stops", "in_transit"]
    at_well = _row(table, stop_cohort="at_well")
    # SWR-coupled 3.0 vs non-SWR 1.0 in every session.
    assert at_well["pct_sessions_positive"] == pytest.approx(100.0)
    assert at_well["animals_positive"] == "2/2"
    assert at_well["n_paired_sessions"] == len(SESSIONS)


def test_delta_aic_steps_are_taken_within_each_cohort(cohort_fits):
    table = cohort_delta_aic_table(cohort_fits)
    full_step = f"{FULL_MODEL} − M6r_time_histcos"
    at_well = _row(table, step=full_step, outcome="non_local", stop_cohort="at_well")
    assert at_well["median_delta_aic"] == pytest.approx(-10.0)
    assert at_well["pct_sessions_child_better"] == pytest.approx(100.0)
    all_stops = _row(
        table, step=full_step, outcome="non_local", stop_cohort="all_stops"
    )
    assert all_stops["median_delta_aic"] == pytest.approx(4.0)
    assert all_stops["pct_sessions_child_better"] == pytest.approx(0.0)
    base_step = "M1_rew − M0_const"
    assert _row(table, step=base_step, outcome="non_local", stop_cohort="at_well")[
        "median_delta_aic"
    ] == pytest.approx(-5.0)


def test_burst_peak_recovers_each_cohort_time_course(cohort_fits):
    """Each cohort's planted peak is recovered, well inside the 0.5 s spacing.

    The tolerance is a spline-approximation budget, not a measurement one: a
    6-column cubic basis cannot reproduce a narrow Gaussian exactly, so its
    argmax lands a fraction of a knot span off the planted peak. It is far
    tighter than the 0.5 s gap between the three planted peaks, which is what
    the table has to be able to tell apart.
    """
    table = cohort_burst_peak_table(cohort_fits)
    assert _row(table, outcome="non_local", stop_cohort="at_well")[
        "burst_peak_s"
    ] == pytest.approx(1.5, abs=0.15)
    assert _row(table, outcome="non_local", stop_cohort="all_stops")[
        "burst_peak_s"
    ] == pytest.approx(2.5, abs=0.15)
    assert _row(table, outcome="non_local", stop_cohort="in_transit")[
        "burst_peak_s"
    ] == pytest.approx(2.0, abs=0.15)


def test_contrast_rung_table_reads_only_the_primary_fitting_cohort():
    def bundle(coefficient: float) -> pd.DataFrame:
        rows = []
        for animal, date, epoch in SESSIONS:
            for cohort, value in (
                ("primary", coefficient),
                ("post_outcome", coefficient + 5.0),
            ):
                rows.append(
                    {
                        "animal": animal,
                        "session_date": date,
                        "epoch": epoch,
                        "model": "G0_base",
                        "population": "non_local",
                        "cohort": cohort,
                        "stop_cohort": "planted",
                        "n_stops": 10,
                        "n_bins": 100,
                        IMMEDIATE_COEF: value,
                    }
                )
        return pd.DataFrame(rows)

    table = cohort_contrast_rung_table(
        {"at_well": bundle(np.log(2.0)), "all_stops": bundle(np.log(1.25))}
    )
    assert _row(table, model="G0_base", population="non_local", stop_cohort="at_well")[
        "rr_immediate"
    ] == pytest.approx(2.0)
    assert _row(
        table, model="G0_base", population="non_local", stop_cohort="all_stops"
    )["rr_immediate"] == pytest.approx(1.25)
    assert set(table["stop_cohort"]) == {"at_well", "all_stops"}


def test_report_names_every_cohort_and_its_command(cohort_fits, tmp_path):
    artifact = tmp_path / "all_models.pkl"
    cohort_fits["at_well"].to_pickle(artifact)
    report = build_cohort_report(
        cohort_fits, {}, {"per-bin (at_well)": artifact}, command="uv run x"
    )
    for cohort in COHORT_ORDER:
        assert cohort in report
    assert "Command: `uv run x`" in report
    assert "rows say ['planted']" in report
    # The likelihood-scale caveat is the one claim the mode must not lose.
    assert "WITHIN one cohort" in report


def test_report_inputs_section_hashes_a_real_file(cohort_fits, tmp_path):
    artifact = tmp_path / "all_models.pkl"
    cohort_fits["at_well"].to_pickle(artifact)
    report = build_cohort_report(
        cohort_fits, {}, {"per-bin (at_well)": artifact}, command=None
    )
    assert str(artifact) in report


def test_missing_sensitivity_bundles_are_skipped_but_at_well_is_required(
    cohort_fits, tmp_path
):
    at_well = tmp_path / "all_models.pkl"
    cohort_fits["at_well"].to_pickle(at_well)
    fits, contrast, inputs = collect_cohort_inputs(
        {
            "at_well": at_well,
            "all_stops": tmp_path / "absent_allstops.pkl",
            "in_transit": tmp_path / "absent_intransit.pkl",
        },
        {"at_well": tmp_path / "absent_contrast.pkl"},
    )
    assert set(fits) == {"at_well"}
    assert contrast == {}
    assert set(inputs) == {"per-bin (at_well)"}

    with pytest.raises(SystemExit, match="canonical at-well per-bin fit is required"):
        collect_cohort_inputs({"at_well": tmp_path / "nope.pkl"}, {})


def test_cohort_contrast_mode_writes_a_report(cohort_fits, tmp_path):
    at_well = tmp_path / "all_models.pkl"
    cohort_fits["at_well"].to_pickle(at_well)
    in_transit = tmp_path / "all_models_intransit.pkl"
    cohort_fits["in_transit"].to_pickle(in_transit)
    out = tmp_path / "audits" / "cohort.md"
    args = build_parser().parse_args(
        [
            "--cohort-contrast",
            "--at-well-per-bin",
            str(at_well),
            "--all-stops-per-bin",
            str(tmp_path / "absent.pkl"),
            "--in-transit-per-bin",
            str(in_transit),
            "--at-well-contrast",
            str(tmp_path / "absent_contrast.pkl"),
            "--all-stops-contrast",
            str(tmp_path / "absent_contrast2.pkl"),
            "--out",
            str(out),
        ]
    )
    run_cohort_contrast_mode(args)
    text = out.read_text()
    assert "Rate ladders across the three stop cohorts" in text
    # The cohort key names all three and says which bundle was supplied, so an
    # absent sensitivity fit reads as absent rather than as an empty result.
    assert "| all_stops | no |" in text
    assert "| in_transit | yes |" in text
    assert IMMEDIATE_COEF.removeprefix("coef_") == REWARD_HISTORY_COLUMNS[0]
