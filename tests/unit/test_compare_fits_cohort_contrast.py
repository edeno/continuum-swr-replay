"""The audit that places one rate ladder's three stop cohorts side by side.

The cohort-contrast mode exists to answer one question: does a rate number move
because the effect moved, or because the population did? So every assertion here
is on a planted value -- a known reward coefficient, a known AIC gap, a known
spline peak, a known cohort size -- recovered per cohort, rather than on a table
merely being produced. The fixtures are the smallest frames carrying the columns
a real per-bin fit carries.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from patsy import dmatrix

from continuum_swr_replay.analysis.replay.model_specs import (
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_BASIS,
    REWARD_HISTORY_N_LAGS,
    SPLINE_KNOTS,
    TMAX,
)
from continuum_swr_replay.analysis.replay.per_bin_design import reward_kernel_weights
from scripts.qa.compare_fits import (
    COHORT_ORDER,
    FULL_MODEL,
    IMMEDIATE_COEF,
    REWARD_X_TIME_RUNGS,
    STOP_TIME_RUNG,
    SWR_IMMEDIATE_COEF,
    _animal_balanced,
    _flag_support_edge,
    build_cohort_report,
    build_parser,
    cohort_burst_peak_table,
    cohort_contrast_rung_table,
    cohort_delta_aic_table,
    cohort_paired_swr_table,
    cohort_provenance_table,
    cohort_reward_rr_table,
    cohort_size_table,
    collect_cohort_inputs,
    in_transit_power_note,
    rate_rung_sort_key,
    reward_kernel_lag1,
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
    later_bumps: tuple[float, float] = (0.0, 0.0),
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
                    # The paired test and the lag-1 kernel both read the whole
                    # three-bump block, so the later components must exist.
                    for column, value in zip(
                        REWARD_HISTORY_COLUMNS[1:], later_bumps, strict=True
                    ):
                        row[f"coef_{column}"] = value
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
    assert below_one["animals_pos_immediate"] == "0/2"
    assert (
        _row(table, outcome="non_local", stop_cohort="at_well")["animals_pos_immediate"]
        == "2/2"
    )


def test_lag1_is_the_kernel_blend_and_not_the_immediate_coefficient(cohort_fits):
    """`rr_lag1` must be the published mixture, never a restatement of bump 0.

    With bumps 1 and 2 planted at zero, the lag-1 kernel is exactly
    ``W[0, 0] * beta_0``, a strict shrink of the immediate contrast towards 1 --
    so a table that quietly reported the same number twice would fail here.
    """
    weights = reward_kernel_weights(REWARD_HISTORY_N_LAGS, REWARD_HISTORY_N_BASIS)
    blend = weights[:, 0]
    assert blend == pytest.approx([0.325, 0.066, 0.0], abs=5e-4)

    table = cohort_reward_rr_table(cohort_fits)
    for outcome, planted in (
        ("non_local_swr", 3.0),
        ("non_local", 2.0),
        ("non_local_no_swr", 1.0),
    ):
        row = _row(table, outcome=outcome, stop_cohort="at_well")
        expected = float(np.exp(blend[0] * np.log(planted)))
        assert row["rr_lag1_session"] == pytest.approx(expected)
        assert row["rr_lag1_animal"] == pytest.approx(expected)
        assert row["rr_immediate"] == pytest.approx(planted)
        if planted != 1.0:
            assert row["rr_lag1_session"] != pytest.approx(row["rr_immediate"])


def test_lag1_mixes_the_later_bumps_in(cohort_fits):
    """A non-zero second bump has to move lag 1, by its own 0.066 weight."""
    weights = reward_kernel_weights(REWARD_HISTORY_N_LAGS, REWARD_HISTORY_N_BASIS)
    fits = _per_bin_fits(
        immediate_by_population=dict.fromkeys(POPULATIONS, np.log(2.0)),
        later_bumps=(np.log(5.0), np.log(7.0)),
        aic={},
        n_stops=1,
        n_bins=1,
        n_events=1,
        peak=1.5,
    )
    lag1 = reward_kernel_lag1(fits[fits.model == FULL_MODEL])
    expected = (
        weights[0, 0] * np.log(2.0)
        + weights[1, 0] * np.log(5.0)
        + weights[2, 0] * np.log(7.0)
    )
    assert len(lag1) == len(SESSIONS) * len(POPULATIONS)
    assert lag1.to_numpy() == pytest.approx(expected)


def test_lag1_is_empty_when_the_reward_block_is_absent():
    assert reward_kernel_lag1(pd.DataFrame({"animal": ["a"]})).empty


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


def _contrast_bundle(
    coefficient: float,
    *,
    models: tuple[str, ...] = ("G0_base",),
    population: str = "non_local",
    swr_coefficient: float | None = None,
) -> pd.DataFrame:
    """A per-session is_swr-contrast bundle, both fitting cohorts populated."""
    rows = []
    for model in models:
        for animal, date, epoch in SESSIONS:
            for cohort, value in (
                ("primary", coefficient),
                ("post_outcome", coefficient + 5.0),
            ):
                row = {
                    "animal": animal,
                    "session_date": date,
                    "epoch": epoch,
                    "model": model,
                    "population": population,
                    "cohort": cohort,
                    "stop_cohort": "planted",
                    "n_stops": 10,
                    "n_bins": 100,
                    IMMEDIATE_COEF: value,
                }
                if swr_coefficient is not None:
                    row[SWR_IMMEDIATE_COEF] = swr_coefficient
                rows.append(row)
    return pd.DataFrame(rows)


def test_contrast_rung_table_reads_only_the_primary_fitting_cohort():
    table = cohort_contrast_rung_table(
        {
            "at_well": _contrast_bundle(np.log(2.0)),
            "all_stops": _contrast_bundle(np.log(1.25)),
        }
    )
    assert _row(table, model="G0_base", population="non_local", stop_cohort="at_well")[
        "rr_immediate"
    ] == pytest.approx(2.0)
    assert _row(
        table, model="G0_base", population="non_local", stop_cohort="all_stops"
    )["rr_immediate"] == pytest.approx(1.25)
    assert set(table["stop_cohort"]) == {"at_well", "all_stops"}


def test_reward_x_time_rungs_publish_no_immediate_rate_ratio():
    """The main effect there is a boundary extrapolation, so it must stay blank.

    Printing it produced a 39.8x cell: on a reward-x-time rung the reward main
    effect is the slope where every spline column is zero, which is off the
    fitted window.
    """
    boundary = REWARD_X_TIME_RUNGS[0]
    table = cohort_contrast_rung_table(
        {"at_well": _contrast_bundle(np.log(40.0), models=("G0_base", boundary))}
    )
    plain = _row(table, model="G0_base", population="non_local", stop_cohort="at_well")
    blanked = _row(table, model=boundary, population="non_local", stop_cohort="at_well")
    assert plain["rr_immediate"] == pytest.approx(40.0)
    assert np.isnan(blanked["rr_immediate"])
    assert np.isnan(blanked["rr_swr_x_immediate"])
    assert blanked["animals_positive"] == "—"
    # The row is still reported, with its cohort size: only the estimand is gone.
    assert blanked["n_sessions"] == len(SESSIONS)


def test_s_rungs_report_the_swr_multiplier_beside_the_reference_slope():
    """An S rung's main effect is the non-SWR level; the interaction is the rest."""
    table = cohort_contrast_rung_table(
        {
            "at_well": _contrast_bundle(
                np.log(2.0),
                models=("S1_rew",),
                population="contrast",
                swr_coefficient=np.log(3.0),
            )
        }
    )
    row = _row(table, model="S1_rew", population="contrast", stop_cohort="at_well")
    assert row["rr_immediate"] == pytest.approx(2.0)
    assert row["rr_swr_x_immediate"] == pytest.approx(3.0)


def test_a_g_rung_without_the_interaction_reports_no_multiplier():
    table = cohort_contrast_rung_table({"at_well": _contrast_bundle(np.log(2.0))})
    row = _row(table, model="G0_base", population="non_local", stop_cohort="at_well")
    assert np.isnan(row["rr_swr_x_immediate"])


def test_rate_rungs_sort_by_number_not_by_text():
    ladder = [
        "S10_rewxtime",
        "S1_rew",
        "S0_rate",
        "S9_update",
        "G8_rewxtime",
        "G0_base",
    ]
    assert sorted(ladder, key=rate_rung_sort_key) == [
        "G0_base",
        "G8_rewxtime",
        "S0_rate",
        "S1_rew",
        "S9_update",
        "S10_rewxtime",
    ]


def test_contrast_rung_table_rows_come_out_in_ladder_order():
    table = cohort_contrast_rung_table(
        {"at_well": _contrast_bundle(0.0, models=("S10_rewxtime", "S2_base", "S1_rew"))}
    )
    assert list(table["model"]) == ["S1_rew", "S2_base", "S10_rewxtime"]


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


def test_in_transit_note_counts_the_sessions_that_reach_the_ladder(cohort_fits):
    """The caveat must be derived from the bundle, not asserted in prose."""
    # Both chimi sessions clear every population; one j16 session clears only
    # the pooled non-local one, and the fourth session clears nothing -- the
    # shape a sparse cohort actually has.
    thin = cohort_fits["in_transit"]
    keep = thin["animal"].eq("chimi") | (
        thin["session_date"].eq("20210706") & thin["outcome"].eq("non_local")
    )
    cohort_fits["in_transit"] = thin[keep].reset_index(drop=True)

    note = in_transit_power_note(cohort_fits)

    assert "**3 of 4** sessions" in note
    assert "non_local 3" in note
    assert "non_local_swr 2" in note
    assert "against 4 on the canonical cohort" in note
    assert "absence of evidence" in note


def test_in_transit_note_is_empty_without_an_in_transit_bundle(cohort_fits):
    del cohort_fits["in_transit"]
    assert in_transit_power_note(cohort_fits) == ""


def test_burst_peak_at_the_support_edge_is_starred(cohort_fits):
    table = cohort_burst_peak_table(cohort_fits)
    table.loc[0, "peak_frac_support"] = 1.0
    table.loc[1, "peak_frac_support"] = 0.5
    shown = _flag_support_edge(table)
    assert shown.loc[0, "burst_peak_s"].endswith("*")
    assert not shown.loc[1, "burst_peak_s"].endswith("*")
    # The source table keeps a number; only the rendered copy is starred.
    assert isinstance(table.loc[0, "burst_peak_s"], float)


def test_provenance_table_reads_the_sidecars(cohort_fits, tmp_path):
    artifact = tmp_path / "all_models.pkl"
    cohort_fits["at_well"].to_pickle(artifact)
    (tmp_path / "all_models.provenance.json").write_text(
        json.dumps(
            {
                "git_hash": "deadbeef1234",
                "stop_cohort": "at_well",
                "written_at": "2026-09-12T00:00:00+00:00",
            }
        )
    )
    table = cohort_provenance_table(
        {"per-bin (at_well)": artifact, "absent": tmp_path / "nope.pkl"}
    )
    assert len(table) == 1
    assert table.loc[0, "git_hash"] == "deadbeef1234"
    assert table.loc[0, "stop_cohort"] == "at_well"


def test_report_defines_both_reward_quantities_and_carries_the_caveats(
    cohort_fits, tmp_path
):
    artifact = tmp_path / "all_models.pkl"
    cohort_fits["at_well"].to_pickle(artifact)
    report = build_cohort_report(
        cohort_fits, {}, {"per-bin (at_well)": artifact}, command="uv run x"
    )
    assert "rr_lag1" in report and "rr_immediate" in report
    assert "[0.325, 0.066, 0]" in report
    assert "absence of evidence" in report
    assert "peak_frac_support" in report
