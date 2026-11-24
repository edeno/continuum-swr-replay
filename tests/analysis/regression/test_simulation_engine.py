"""The simulation engine on tiny cohorts: the outcome drawn, the real fit.

Two synthetic models of the engine's own, on rows from ``conftest``:

- a rate model (Poisson): an intercept, the last outcome (0/1) with the weak
  penalty, a smooth of the time since the visit began and the exposure's
  offset; 5 animals of 2 sessions, each on its own day (``rate_rows``);
- a content model (grouped softmax): the role category with the weak penalty;
  5 animals of 2 sessions (``content_rows``, with each event's number of
  decode bins), its shares drawn by each of the softmax's noise models.

``python -m tests.analysis.regression.test_simulation_engine`` writes the
fixed-seed fixture ``data/numpy_reference/simulation_engine.json``.
"""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from continuum_swr_replay.analysis.regression import simulation as engine
from continuum_swr_replay.analysis.regression.contrasts import (
    coefficient_contrast,
    range_mean_contrast,
)
from continuum_swr_replay.analysis.regression.likelihoods import (
    Gaussian,
    GroupedSoftmax,
    Poisson,
)
from continuum_swr_replay.analysis.regression.simulation import (
    CAUSES,
    SUMMARY_COLUMNS,
    InsideScenario,
    Model,
    OutsideScenario,
    Simulation,
    bootstrap_seeds,
    data_seed,
    draw_data_set,
)
from continuum_swr_replay.analysis.regression.terms import (
    Category,
    Offset,
    Slope,
    Smooth,
    build_design,
    log_penalty_from_prior_sd,
)
from continuum_swr_replay.analysis.shared_statistics import (
    estimand_summary,
    signed_rank_p,
)

from .conftest import ROLES, content_rows, rate_rows

#: The weak penalty of the tests: a prior SD of 2 on each simple coefficient.
WEAK = log_penalty_from_prior_sd(2.0)
CLOCK = (0.0, 10.0)
REWARD = "last_rewarded"
FIXTURE = Path(__file__).parent / "data" / "numpy_reference" / "simulation_engine.json"


def _cohort(make, n_animals: int = 5, n_sessions: int = 2) -> dict:
    """``n_animals`` animals of ``n_sessions`` sessions, each on its own day,
    their rows from ``make(seed)``."""
    out = {}
    for a in range(n_animals):
        for e in range(n_sessions):
            key = (f"animal{a}", f"2026{e + 1:04d}", "02_r1")
            rows = make(n_sessions * a + e)
            out[key] = rows.assign(animal=key[0], session_date=key[1], epoch=key[2])
    return out


def rate_model(sessions: dict) -> Model:
    """The rate model and its two estimands: the last outcome's coefficient
    (``reward``) and the clock's mean over 1 to 2 s less its mean over the
    first 0.5 s (``rise``)."""
    pooled = pd.concat(sessions.values(), ignore_index=True)
    design = build_design(
        [
            Slope(REWARD, WEAK),
            Smooth("time_since_s", *CLOCK, 12, 3.0),
            Offset("exposure"),
        ],
        pooled,
        intercept=True,
    )
    columns = design.term_columns["s(time_since_s)"]

    def curve_rows(x):
        X = design.build(
            pd.DataFrame({"time_since_s": x, REWARD: 0.0, "exposure": 1.0})
        ).X
        rows = np.zeros_like(X)
        rows[:, columns] = X[:, columns]
        return rows

    return Model(
        "rate",
        design,
        Poisson(),
        "count",
        "well_visit_id",
        {
            "reward": coefficient_contrast(design.coef_names, [REWARD]),
            "rise": range_mean_contrast(
                curve_rows, (1.0, 2.0), (0.0, 0.5), term_range=CLOCK
            ),
        },
    )


def content_model(sessions: dict) -> Model:
    """The content model and its estimand, the current role's coefficient
    against another leaf arm (``current``)."""
    pooled = pd.concat(sessions.values(), ignore_index=True)
    role = Category("role", ROLES, reference="other", penalty=WEAK)
    design = build_design([role], pooled, intercept=False)
    return Model(
        "content",
        design,
        GroupedSoftmax(),
        "share",
        "event_id",
        {"current": coefficient_contrast(design.coef_names, ["role[current]"])},
    )


#: The smooth is planted flat: a curve that bends is shrunk by its penalty, so
#: its contrast is not recovered without bias (the declared-penalty check
#: measures that bias).
RATE_INSIDE = InsideScenario("planted", {"(Intercept)": np.log(0.8), REWARD: 0.5})
CONTENT_INSIDE = InsideScenario(
    "planted",
    {"role[current]": 1.0, "role[previous]": 0.3, "role[trunk]": -0.5},
    n_trials=5,
)


def expected_rate(sessions, rng, *, zero_sessions=(), no_event_when_unrewarded=()):
    """A rise of the log rate by log 2 over the first second, a step of 0.5
    after a reward, and an animal effect drawn anew in each data set (SD
    0.3): a curve the smooth holds only approximately. ``zero_sessions`` have
    no events; in ``no_event_when_unrewarded`` the rows after an omission
    have none."""
    animals = sorted({key[0] for key in sessions})
    effect = dict(zip(animals, 0.3 * rng.normal(size=len(animals)), strict=True))
    out = {}
    for key, rows in sessions.items():
        t, reward = rows["time_since_s"].to_numpy(), rows[REWARD].to_numpy()
        log_rate = (
            np.log(0.8)
            + effect[key[0]]
            + np.log(2.0) * (1.0 - np.exp(-t / 0.3))
            + 0.5 * reward
        )
        mean = rows["exposure"].to_numpy() * np.exp(log_rate)
        if key in zero_sessions:
            mean = np.zeros_like(mean)
        if key in no_event_when_unrewarded:
            mean = np.where(reward == 0, 0.0, mean)
        out[key] = mean
    return out


@pytest.fixture(scope="module")
def rate_sessions_cohort() -> dict:
    return _cohort(rate_rows)


def _content_rows(seed: int) -> pd.DataFrame:
    """``content_rows`` with ``n_bins``, each event's number of decode bins
    (3 to 30, the same on its rows)."""
    rows = content_rows(seed)
    n_bins = np.random.default_rng(seed).integers(3, 31, rows["event_id"].max() + 1)
    return rows.assign(n_bins=n_bins[rows["event_id"]])


@pytest.fixture(scope="module")
def content_sessions_cohort() -> dict:
    return _cohort(_content_rows)


#: The share noise of the content model: one categorical choice per event;
#: a multinomial of the event's decode bins; the same, overdispersed.
CONTENT_NOISE = {
    "categorical": {},
    "multinomial": {"n_trials": "n_bins"},
    "dirichlet_multinomial": {"n_trials": "n_bins", "concentration": 2.0},
}


@pytest.mark.parametrize("noise", list(CONTENT_NOISE))
def test_content_share_noise(noise: str, content_sessions_cohort) -> None:
    """Each share noise model, chosen by the scenario, draws shares of the
    kind it names (0 or 1; counts over the event's ``n_bins``), and the
    planted current-role coefficient is recovered (bias within 2 SEs and the
    weak penalty's shrinkage)."""
    sessions = content_sessions_cohort
    model = content_model(sessions)
    scenario = InsideScenario(
        noise, CONTENT_INSIDE.coefficients, **CONTENT_NOISE[noise]
    )
    key = next(iter(sessions))
    drawn = draw_data_set(model, sessions, scenario, np.random.default_rng(1))[key]
    share = drawn["share"].to_numpy()
    np.testing.assert_allclose(drawn.groupby("event_id")["share"].sum(), 1.0)
    if noise == "categorical":
        assert set(np.unique(share)) == {0.0, 1.0}
    else:
        counts = share * drawn["n_bins"].to_numpy()
        np.testing.assert_allclose(counts, np.round(counts), atol=1e-9)
        assert len(np.unique(share)) > 2
    result = engine.run(_simulation(model, sessions, [scenario], n_data_sets=6))
    row = result.summary.iloc[0]
    assert row["planted"] == 1.0
    # The weak penalty (prior SD 2) shrinks the coefficient by about 0.01 at
    # this size: with the prior SD at 100, 30 data sets of the multinomial
    # gave a bias of 0.006 (SE 0.005), and at SD 2, 6 data sets -0.025 (SE
    # 0.006). The bound allows that shrinkage beside 2 SEs.
    assert abs(row["bias"]) < 2 * row["bias_se"] + 0.02


def _simulation(model, sessions, scenarios, n_data_sets=2, seed=5) -> Simulation:
    return Simulation([model], sessions, scenarios, n_data_sets, 200, seed)


def _capture(monkeypatch) -> list[dict]:
    """The data sets given to ``fit_sessions``, recorded as the engine runs."""
    seen = []
    real = engine.fit_sessions

    def recording(design, sessions, *args, **kwargs):
        seen.append({key: rows.copy() for key, rows in sessions.items()})
        return real(design, sessions, *args, **kwargs)

    monkeypatch.setattr(engine, "fit_sessions", recording)
    return seen


def test_same_seed_same_data(monkeypatch, rate_sessions_cohort) -> None:
    """Two runs with one seed give the same outcome columns, bit for bit, and
    the same tables; a second data set and a second seed give other
    outcomes. ``draw_data_set`` at the data set's seed gives the same data."""
    sessions = rate_sessions_cohort
    model = rate_model(sessions)
    simulation = _simulation(model, sessions, [RATE_INSIDE])
    seen = _capture(monkeypatch)
    first = engine.run(simulation)
    second = engine.run(simulation)
    assert len(seen) == 4
    for a, b in zip(seen[:2], seen[2:], strict=True):
        for key in sessions:
            assert a[key]["count"].to_numpy().tobytes() == (
                b[key]["count"].to_numpy().tobytes()
            )
    for x, y in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(x, y)
    key = next(iter(sessions))
    assert not np.array_equal(seen[0][key]["count"], seen[1][key]["count"])
    other = draw_data_set(
        model, sessions, RATE_INSIDE, np.random.default_rng(data_seed(6, 0, 0))
    )
    assert not np.array_equal(other[key]["count"], seen[0][key]["count"])
    again = draw_data_set(
        model, sessions, RATE_INSIDE, np.random.default_rng(data_seed(5, 0, 1))
    )
    for k in sessions:
        np.testing.assert_array_equal(again[k]["count"], seen[1][k]["count"])


def test_only_the_outcome_changes(monkeypatch, rate_sessions_cohort) -> None:
    """The rows given to ``fit_sessions`` are the real rows' columns the run
    reads (the design's, the unit; :func:`.simulation.run_columns`), equal
    to them with their index, and the outcome, drawn anew."""
    sessions = rate_sessions_cohort
    seen = _capture(monkeypatch)
    simulation = _simulation(rate_model(sessions), sessions, [RATE_INSIDE])
    kept = engine.run_columns(simulation)
    assert set(kept) == {"time_since_s", REWARD, "exposure", "well_visit_id"}
    engine.run(simulation)
    for data in seen:
        assert list(data) == list(sessions)
        for key, rows in sessions.items():
            columns = [c for c in rows.columns if c in kept]
            assert list(data[key].columns) == [*columns, "count"]
            pd.testing.assert_frame_equal(
                data[key].drop(columns="count"), rows[columns]
            )
            assert not np.array_equal(data[key]["count"], rows["count"])


def test_run_columns_name_what_a_scenario_reads(content_sessions_cohort) -> None:
    """A run keeps a column a scenario names for the softmax's noise and
    one an outside scenario's expected outcome reads; a session without a
    kept column is refused."""
    sessions = content_sessions_cohort
    model = content_model(sessions)
    noise = InsideScenario("noise", CONTENT_INSIDE.coefficients, n_trials="n_bins")
    outside = OutsideScenario("outside", _shares, columns=("clock_s",))
    simulation = _simulation(model, sessions, [noise, outside])
    assert engine.run_columns(simulation) == ("role", "event_id", "n_bins", "clock_s")
    key = next(iter(sessions))
    short = {**sessions, key: sessions[key].drop(columns="n_bins")}
    with pytest.raises(ValueError, match=r"no columns \['n_bins'\]"):
        engine.run(_simulation(model, short, [noise]))


def test_prepared_draws_give_the_same_data_set(content_sessions_cohort) -> None:
    """Prepared once and drawn twice, the data sets are those of the
    one-call form at the same seeds; draws prepared for another scenario are
    refused."""
    sessions = content_sessions_cohort
    model = content_model(sessions)
    scenario = InsideScenario("noise", CONTENT_INSIDE.coefficients, n_trials="n_bins")
    prepared = engine.prepare_draws(model, sessions, scenario)
    for seed in (3, 4):
        once = draw_data_set(model, sessions, scenario, np.random.default_rng(seed))
        again = draw_data_set(
            model, sessions, scenario, np.random.default_rng(seed), prepared
        )
        assert list(again) == list(once)
        for key in once:
            pd.testing.assert_frame_equal(again[key], once[key])
    with pytest.raises(ValueError, match="another model, scenario"):
        draw_data_set(
            model, sessions, CONTENT_INSIDE, np.random.default_rng(0), prepared
        )


def test_planted_coefficients_recovered(rate_sessions_cohort) -> None:
    """Inside the model, on 5 animals of 2 sessions: the planted last-outcome
    coefficient and the flat curve's contrast (``c' beta`` at the planted
    coefficients, 0) are recovered, the bias within 2 simulation SEs, every
    session has a value and every fit converges. The table has its columns
    and one row per estimand."""
    sessions = rate_sessions_cohort
    model = rate_model(sessions)
    result = engine.run(_simulation(model, sessions, [RATE_INSIDE], n_data_sets=20))
    table = result.summary.set_index("estimand")
    assert list(result.summary.columns) == list(SUMMARY_COLUMNS)
    assert table.loc["reward", "planted"] == 0.5
    assert table.loc["rise", "planted"] == 0.0
    for name in ("reward", "rise"):
        row = table.loc[name]
        assert abs(row["bias"]) < 2 * row["bias_se"], name
        assert row["share_without_value"] == 0.0
        assert row["share_not_converged"] == 0.0
        assert row["n_sessions_used_min"] == 10 and row["n_data_sets"] == 20
    assert table.loc["reward", "excludes_zero"] == 1.0
    assert len(result.data_sets) == 40 and len(result.sessions) == 400


def test_planted_content_coefficient_recovered(content_sessions_cohort) -> None:
    """Under the grouped softmax, with shares of five choices per event: the
    planted current-role coefficient is recovered (bias within 2 SEs)."""
    sessions = content_sessions_cohort
    model = content_model(sessions)
    result = engine.run(_simulation(model, sessions, [CONTENT_INSIDE], n_data_sets=10))
    row = result.summary.iloc[0]
    assert row["planted"] == 1.0
    assert abs(row["bias"]) < 2 * row["bias_se"]
    assert row["share_without_value"] == 0.0


def test_outside_scenario(rate_sessions_cohort) -> None:
    """An expected outcome given as a function of the rows (a curve outside
    the smooth's span, an animal effect drawn in each data set) is drawn and
    fitted. The scenario names the planted value of ``reward`` and not of
    ``rise``: ``rise`` has the planted value NaN and no bias or coverage,
    and its estimate is reported."""
    sessions = rate_sessions_cohort
    scenario = OutsideScenario("outside", expected_rate, planted={"reward": 0.5})
    result = engine.run(
        _simulation(rate_model(sessions), sessions, [scenario], n_data_sets=10)
    )
    table = result.summary.set_index("estimand")
    reward, rise = table.loc["reward"], table.loc["rise"]
    assert reward["planted"] == 0.5
    assert abs(reward["bias"]) < 2 * reward["bias_se"]
    assert 0.0 <= reward["coverage"] <= 1.0
    assert np.isnan(rise["planted"]) and np.isnan(rise["bias"])
    assert np.isnan(rise["coverage"]) and np.isnan(rise["coverage_se"])
    # The planted rise's contrast is about 0.33; the smooth's penalty shrinks
    # this sharp a rise, which is why the scenario states no value for it.
    assert 0.1 < rise["mean_estimate"] < 0.33
    assert rise["n_data_sets_with_estimate"] == 10


def test_a_planted_value_may_be_a_function_of_the_model(rate_sessions_cohort) -> None:
    """A planted value given as a function receives the model."""
    sessions = rate_sessions_cohort
    model = rate_model(sessions)
    scenario = OutsideScenario(
        "outside",
        expected_rate,
        planted={"rise": lambda m: float(m.estimands["rise"].sum())},
    )
    assert engine.planted_value(scenario, model, "rise") == pytest.approx(
        model.estimands["rise"].sum()
    )
    assert np.isnan(engine.planted_value(scenario, model, "reward"))


def test_sessions_without_a_value_counted(rate_sessions_cohort) -> None:
    """Of 10 sessions: one has no events (no fit, ``no_events``, for both
    estimands); in two the last outcome is always a reward, so the support
    rule leaves the coefficient out (``support``, for ``reward`` only); in two
    the rows after an omission have no events, and the penalized
    coefficient still has a value there. The shares are those counts over
    the sessions, and no fit tried failed to converge."""
    keys = list(rate_sessions_cohort)
    sessions = dict(rate_sessions_cohort)
    for key in keys[1:3]:
        sessions[key] = sessions[key].assign(**{REWARD: 1.0})
    scenario = OutsideScenario(
        "gaps",
        partial(
            expected_rate,
            zero_sessions=(keys[0],),
            no_event_when_unrewarded=tuple(keys[3:5]),
        ),
    )
    result = engine.run(_simulation(rate_model(sessions), sessions, [scenario]))
    table = result.summary.set_index("estimand")
    assert set(CAUSES) >= {"no_events", "support"}
    reward, rise = table.loc["reward"], table.loc["rise"]
    assert reward["without_value_no_events"] == pytest.approx(0.1)
    assert reward["without_value_support"] == pytest.approx(0.2)
    assert reward["share_without_value"] == pytest.approx(0.3)
    assert rise["without_value_no_events"] == pytest.approx(0.1)
    assert rise["share_without_value"] == pytest.approx(0.1)
    assert rise["without_value_support"] == 0.0
    assert reward["share_not_converged"] == 0.0
    assert reward["n_sessions_used_min"] == 7 and rise["n_sessions_used_min"] == 9
    sessions_table = result.sessions.query("estimand == 'reward'")
    by_key = sessions_table.set_index(["animal", "session_date", "epoch"]).sort_index()
    assert set(by_key.loc[keys[0], "without_value"]) == {"no_events"}
    assert set(by_key.loc[keys[3], "without_value"]) == {""}


def test_no_interval_is_neither_covered_nor_a_rejection(rate_sessions_cohort) -> None:
    """Only animal0 has events: each data set's estimate is finite but its
    interval is NaN (one animal has no t quantile). Its ``covered`` and
    ``excludes_zero`` are NaN, not 0 and 1, and the summary's rates leave it
    out: with no data set with an interval both are NaN, and it counts the
    data sets with an estimate, an interval and a day-level p."""
    keys = list(rate_sessions_cohort)
    zero = tuple(key for key in keys if key[0] != "animal0")
    scenario = OutsideScenario(
        "one_animal",
        partial(expected_rate, zero_sessions=zero),
        planted={"reward": 0.5},
    )
    model = rate_model(rate_sessions_cohort)
    result = engine.run(_simulation(model, rate_sessions_cohort, [scenario]))
    reward = result.data_sets.query("estimand == 'reward'")
    assert np.isfinite(reward["estimate"]).all()
    assert reward["low"].isna().all() and reward["high"].isna().all()
    assert reward["covered"].isna().all()
    assert reward["excludes_zero"].isna().all()
    row = result.summary.set_index("estimand").loc["reward"]
    assert row["n_data_sets_with_estimate"] == 2
    assert row["n_data_sets_with_interval"] == 0
    assert row["n_data_sets_with_day_p"] == 0
    assert np.isnan(row["coverage"]) and np.isnan(row["coverage_se"])
    assert np.isnan(row["excludes_zero"]) and np.isnan(row["excludes_zero_se"])
    # A NaN day-level p is no rejection (the documented rule).
    assert row["reject_day"] == 0.0


def _hand_tables() -> tuple[pd.DataFrame, pd.DataFrame, Simulation]:
    """A ``data_sets`` and a ``sessions`` table of one (scenario, model,
    estimand) with answers known by hand, and a run of four sessions.

    Five data sets, planted 1: three with an interval (two covering it, two
    excluding zero), one with an estimate but no interval (and no day-level
    p), one without an estimate. Ten session rows: two ``no_events`` (not
    fitted), one ``support``, and one fit tried that did not converge."""
    nan = np.nan
    data_sets = pd.DataFrame(
        {
            "scenario": "s",
            "model": "m",
            "data_set": range(5),
            "estimand": "e",
            "planted": 1.0,
            "estimate": [1.0, 2.0, 0.4, 1.2, nan],
            "low": [0.5, 1.8, -0.1, nan, nan],
            "high": [1.5, 2.2, 0.9, nan, nan],
            "day_p": [0.01, 0.2, nan, nan, nan],
            "n_sessions_used": [4, 4, 3, 2, 0],
            "covered": [1.0, 0.0, 1.0, nan, nan],
            "excludes_zero": [1.0, 1.0, 0.0, nan, nan],
        }
    )
    causes = ["", "", "", "", "", "", "no_events", "no_events", "support", ""]
    status = ["converged"] * 6 + ["not fitted"] * 2 + ["converged", "max_iter"]
    sessions = pd.DataFrame(
        {
            "scenario": "s",
            "model": "m",
            "estimand": "e",
            "without_value": causes,
            "status": status,
        }
    )
    keys = [(f"animal{a}", "20260001", "02_r1") for a in range(4)]
    simulation = Simulation([], dict.fromkeys(keys), [], 5, 100, 7)
    return data_sets, sessions, simulation


def test_summary_arithmetic_by_hand() -> None:
    """The summary of :func:`_hand_tables`, every value worked by hand."""
    row = engine.summarize(*_hand_tables()).iloc[0]
    estimates = np.array([1.0, 2.0, 0.4, 1.2])  # the four with an estimate
    sd = np.sqrt(((estimates - 1.15) ** 2).sum() / 3)
    expected = {
        "planted": 1.0,
        "mean_estimate": 1.15,
        "bias": 0.15,
        "bias_share": 0.15,
        "sd": sd,
        "bias_se": sd / 2,
        # Over the three data sets with an interval.
        "coverage": 2 / 3,
        "coverage_se": np.sqrt(2 / 9 / 3),
        "excludes_zero": 2 / 3,
        "excludes_zero_se": np.sqrt(2 / 9 / 3),
        # Over the four with an estimate, a NaN p no rejection.
        "reject_day": 0.25,
        "reject_day_se": np.sqrt(0.25 * 0.75 / 4),
        "share_without_value": 0.3,
        "without_value_no_events": 0.2,
        "without_value_support": 0.1,
        "share_not_converged": 1 / 8,
        "n_sessions": 4,
        "n_sessions_used_min": 0,
        "n_sessions_used_mean": 2.6,
        "n_data_sets": 5,
        "n_data_sets_with_estimate": 4,
        "n_data_sets_with_interval": 3,
        "n_data_sets_with_day_p": 2,
        "n_draws": 100,
        "seed": 7,
    }
    for column, value in expected.items():
        assert row[column] == pytest.approx(value), column
    others = [c for c in CAUSES if c not in ("no_events", "support")]
    assert all(row[f"without_value_{c}"] == 0.0 for c in others)


def _shares(sessions, rng, *, scale_first=1.0):
    """Equal shares within each event; the first event of the first session
    has its shares times ``scale_first``."""
    out = {}
    for k, (key, rows) in enumerate(sessions.items()):
        size = rows.groupby("event_id")["event_id"].transform("size").to_numpy()
        share = 1.0 / size
        if k == 0:
            share = np.where(
                rows["event_id"] == rows["event_id"].iloc[0], share * scale_first, share
            )
        out[key] = share
    return out


def test_expected_shares_must_sum_to_one(content_sessions_cohort) -> None:
    """Under the grouped softmax an event whose expected shares are all zero
    is refused (the draw would give its first row a share of one), as is one
    whose shares do not sum to one within :data:`SHARE_TOLERANCE`; shares
    off by less are drawn."""
    sessions = content_sessions_cohort
    model = content_model(sessions)
    rng = np.random.default_rng(0)
    for scale, match in ((0.0, "sum to zero"), (0.8, "sum to 0.8, not one")):
        scenario = OutsideScenario("bad", partial(_shares, scale_first=scale))
        with pytest.raises(ValueError, match=match):
            draw_data_set(model, sessions, scenario, rng)
    near = 1.0 + engine.SHARE_TOLERANCE / 10
    scenario = OutsideScenario("near", partial(_shares, scale_first=near))
    drawn = draw_data_set(model, sessions, scenario, rng)
    assert all(np.isfinite(rows["share"]).all() for rows in drawn.values())


def _fixed_rate(sessions, rng):
    """Each row's expected count: its exposure times a rate that rises over
    the visit and is 0.5 higher (log) after a reward. ``rng`` is not read."""
    out = {}
    for key, rows in sessions.items():
        t = rows["time_since_s"].to_numpy()
        log_rate = (
            np.log(0.8)
            + np.log(2.0) * (1.0 - np.exp(-t / 0.3))
            + 0.5 * rows[REWARD].to_numpy()
        )
        out[key] = rows["exposure"].to_numpy() * np.exp(log_rate)
    return out


def test_outside_draws_have_the_expected_mean(rate_sessions_cohort) -> None:
    """The mean count of 400 data sets drawn by ``draw_data_set`` under an
    outside scenario is its expected count, the exposure (0.01 to 0.2 s) in
    the model's offset counted once: summed over each session's rows in each
    third of exposure, within 4 Poisson SEs. An exposure counted twice or not
    at all would put the sums off by a factor of about 10."""
    rng = np.random.default_rng(0)
    sessions = {
        key: rows.assign(exposure=rng.uniform(0.01, 0.2, len(rows)))
        for key, rows in list(rate_sessions_cohort.items())[:4]
    }
    model = rate_model(sessions)
    assert model.design.offset.column == "exposure"
    scenario = OutsideScenario("fixed", _fixed_rate)
    expected = _fixed_rate(sessions, None)
    prepared = engine.prepare_draws(model, sessions, scenario)
    n_data_sets = 400
    total = {key: np.zeros(len(rows)) for key, rows in sessions.items()}
    for _ in range(n_data_sets):
        drawn = draw_data_set(model, sessions, scenario, rng, prepared)
        for key in sessions:
            total[key] += drawn[key]["count"].to_numpy()
    for key, rows in sessions.items():
        assert np.isfinite(total[key]).all()
        third = np.digitize(
            rows["exposure"], np.quantile(rows["exposure"], [1 / 3, 2 / 3])
        )
        for k in range(3):
            want = n_data_sets * expected[key][third == k].sum()
            got = total[key][third == k].sum()
            assert abs(got - want) < 4 * np.sqrt(want), (key, k, got, want)


def test_day_level_test_and_interval_on_nine_days() -> None:
    """3 animals of 3 sessions, each on its own day (9 recording days): every
    data set has a day-level p, and its estimate, interval and p are
    ``estimand_summary``'s on the sessions table's values at the data set's
    bootstrap seed. The interval's half width is ``t(0.975, 2) sqrt(3 / 2)``
    times the draws' SD, the p the signed-rank p over the 9 day values (one
    session each), and the summary's ``reject_day`` the share of p below
    0.05 over the data sets."""
    sessions = _cohort(rate_rows, n_animals=3, n_sessions=3)
    model = rate_model(sessions)
    result = engine.run(_simulation(model, sessions, [RATE_INSIDE], n_data_sets=3))
    data_sets = result.data_sets.query("estimand == 'reward'")
    fits = result.sessions.query("estimand == 'reward'")
    t = scipy.stats.t.ppf(0.975, 2) * np.sqrt(3 / 2)
    for row in data_sets.itertuples():
        mine = fits[fits["data_set"] == row.data_set]
        index = pd.MultiIndex.from_frame(mine[["animal", "session_date", "epoch"]])
        values = pd.Series(mine["estimate"].to_numpy(), index=index)
        labels = [pd.Series(index.get_level_values(k), index=index) for k in range(3)]
        direct = estimand_summary(values, *labels, n_draws=200, seed=row.bootstrap_seed)
        assert (row.estimate, row.low, row.high, row.day_p) == (
            direct["estimate"],
            direct["low"],
            direct["high"],
            direct["day_p"],
        )
        assert direct["n_days"] == 9 and np.isfinite(row.day_p)
        assert (row.high - row.low) / 2 == pytest.approx(t * row.interval_sd)
        assert row.day_p == pytest.approx(signed_rank_p(values.to_numpy()))
    summary = result.summary.set_index("estimand").loc["reward"]
    assert summary["n_data_sets_with_day_p"] == 3
    assert summary["n_data_sets_with_interval"] == 3
    assert summary["reject_day"] == pytest.approx(np.mean(data_sets["day_p"] < 0.05))


def test_bootstrap_seeds_differ() -> None:
    """Each (model, estimand) of each data set has its own bootstrap seed,
    and the data sets of a scenario their own data seed."""
    seeds = [bootstrap_seeds(5, s, i, 3) for s in range(2) for i in range(2)]
    assert len({seed for row in seeds for seed in row}) == 12


def test_workers_give_the_same_tables(rate_sessions_cohort) -> None:
    """A spawn pool of two workers gives the tables of one process."""
    sessions = rate_sessions_cohort
    scenario = OutsideScenario("outside", expected_rate, planted={"reward": 0.5})
    simulation = _simulation(rate_model(sessions), sessions, [RATE_INSIDE, scenario])
    for x, y in zip(
        engine.run(simulation), engine.run(simulation, workers=2), strict=True
    ):
        pd.testing.assert_frame_equal(x, y)


def test_refusals(rate_sessions_cohort) -> None:
    """A tested coefficient without a penalty, an unknown planted
    coefficient, models that cannot share data sets, repeated names and a
    Gaussian scenario without a scale are refused."""
    sessions = rate_sessions_cohort
    model = rate_model(sessions)
    with pytest.raises(ValueError, match="without a fixed penalty"):
        Model(
            "bad",
            model.design,
            Poisson(),
            "count",
            "well_visit_id",
            {"base": coefficient_contrast(model.design.coef_names, ["(Intercept)"])},
        )
    with pytest.raises(ValueError, match="not coefficients of the design"):
        engine.run(
            _simulation(model, sessions, [InsideScenario("x", {"nothing": 1.0})])
        )
    with pytest.raises(ValueError, match="names repeat"):
        engine.run(_simulation(model, sessions, [RATE_INSIDE, RATE_INSIDE]))
    other = Model(
        "other",
        model.design,
        GroupedSoftmax(),
        "count",
        "well_visit_id",
        model.estimands,
    )
    with pytest.raises(ValueError, match="cannot share data sets"):
        engine.run(Simulation([model, other], sessions, [RATE_INSIDE], 2, 200, 5))
    gaussian = Model(
        "gaussian", model.design, Gaussian(), "count", "well_visit_id", model.estimands
    )
    with pytest.raises(ValueError, match="needs a scale"):
        draw_data_set(
            gaussian, sessions, InsideScenario("g", {}), np.random.default_rng(0)
        )


# --- A model's own fit ------------------------------------------------------------


def own_session_fit(data, *, design, estimands) -> pd.DataFrame:
    """An own fit that does what the engine does for a plain model: each
    session's fit by ``fit_sessions`` and each estimand's contrast, one row
    per session and estimand."""
    records = engine.fit_sessions(
        design,
        data,
        Poisson(),
        model="own",
        kind="own",
        response="count",
        unit="well_visit_id",
        table_sha256={},
    )
    rows = []
    for record in records:
        coef, vcov = np.array(record.coef), np.array(record.vcov)
        for name, c in estimands.items():
            used = [design.coef_names[j] for j in np.flatnonzero(c)]
            left = [record.left_out[n] for n in used if n in record.left_out]
            cause = record.no_fit or (left[0] if left else "")
            value = None if cause else engine.evaluate_contrast(c, coef, vcov)
            rows.append(
                {
                    "estimand": name,
                    "animal": record.animal,
                    "session_date": record.session_date,
                    "epoch": record.epoch,
                    "status": record.status,
                    "no_fit": record.no_fit,
                    "without_value": cause,
                    "estimate": np.nan if value is None else value.estimate,
                    "se": np.nan if value is None else value.se,
                    "edf": record.edf,
                    "n_sessions": 1,
                }
            )
    return pd.DataFrame(rows)


def own_animal_fit(data, *, without=()) -> pd.DataFrame:
    """An own fit of one value per animal: the log of the animal's events per
    second, over its sessions; the animals in ``without`` have no value."""
    totals: dict[str, list[float]] = {}
    for (animal, _, _), rows in data.items():
        count, exposure, n = totals.setdefault(animal, [0.0, 0.0, 0])
        totals[animal] = [
            count + rows["count"].sum(),
            exposure + rows["exposure"].sum(),
            n + 1,
        ]
    return pd.DataFrame(
        [
            {
                "estimand": "log_rate",
                "animal": animal,
                "session_date": None,
                "epoch": None,
                "status": "converged",
                "no_fit": "no_events" if animal in without else "",
                "without_value": "no_events" if animal in without else "",
                "estimate": np.nan if animal in without else np.log(count / exposure),
                "se": np.nan,
                "edf": {},
                "n_sessions": n,
            }
            for animal, (count, exposure, n) in totals.items()
        ]
    )


def test_an_own_fit_at_the_session_level_gives_the_engine_tables(
    rate_sessions_cohort,
) -> None:
    """A model whose own fit does what the engine does gives the plain
    model's ``data_sets`` and ``sessions`` rows, at the same seed."""
    sessions = rate_sessions_cohort
    plain = rate_model(sessions)
    own = Model(
        "rate",
        plain.design,
        Poisson(),
        "count",
        "well_visit_id",
        {},
        fit=partial(own_session_fit, design=plain.design, estimands=plain.estimands),
        fit_estimands=tuple(plain.estimands),
    )
    scenario = OutsideScenario("outside", expected_rate, planted={"reward": 0.5})
    a = engine.run(_simulation(plain, sessions, [scenario]))
    b = engine.run(_simulation(own, sessions, [scenario]))
    pd.testing.assert_frame_equal(a.data_sets, b.data_sets)
    columns = list(a.sessions.columns)
    pd.testing.assert_frame_equal(a.sessions, b.sessions[columns])
    pd.testing.assert_frame_equal(a.summary, b.summary)


def test_an_own_fit_at_the_animal_level_is_reduced_over_the_animals(
    rate_sessions_cohort,
) -> None:
    """One value per animal (no session labels): a data set's estimate is
    the mean over the animals with a value, its interval the t interval over
    them, its sessions used those of those animals; the share without a value
    is over the animals. The model's coefficient names need not be the
    plain model's."""
    sessions = rate_sessions_cohort
    plain = rate_model(sessions)
    pooled = pd.concat(sessions.values(), ignore_index=True)
    design = build_design([Offset("exposure")], pooled, intercept=True)
    animal = Model(
        "animal",
        design,
        Poisson(),
        "count",
        "well_visit_id",
        {},
        fit=partial(own_animal_fit, without=("animal0",)),
        fit_estimands=("log_rate",),
        fit_columns=("time_since_s",),
    )
    scenario = OutsideScenario(
        "outside", expected_rate, planted={"reward": 0.5, "log_rate": 0.0}
    )
    simulation = Simulation([plain, animal], sessions, [scenario], 3, 200, 5)
    alone = Simulation([animal], sessions, [scenario], 3, 200, 5)
    assert engine.run_columns(alone) == ("exposure", "well_visit_id", "time_since_s")
    result = engine.run(simulation)
    units = result.sessions.query("model == 'animal'")
    assert len(units) == 3 * 5 and units["session_date"].isna().all()
    for i, part in result.data_sets.query("model == 'animal'").groupby("data_set"):
        (row,) = part.to_dict("records")
        values = units[units["data_set"] == i].set_index("animal")["estimate"]
        finite = values.dropna()
        assert len(finite) == 4
        se = finite.std(ddof=1) / 2.0
        half = scipy.stats.t.ppf(0.975, 3) * se
        assert row["estimate"] == pytest.approx(finite.mean())
        assert row["low"] == pytest.approx(finite.mean() - half)
        assert row["high"] == pytest.approx(finite.mean() + half)
        assert row["interval_sd"] == pytest.approx(se)
        assert np.isnan(row["day_p"])
        assert row["n_sessions_used"] == 8
        assert row["covered"] == float(row["low"] <= 0.0 <= row["high"])
    summary = result.summary.set_index("model").loc["animal"]
    assert summary["share_without_value"] == pytest.approx(0.2)
    assert summary["without_value_no_events"] == pytest.approx(0.2)
    assert summary["n_sessions_used_min"] == 8
    assert summary["n_data_sets_with_day_p"] == 0
    assert np.isnan(summary["reject_day"]) and np.isnan(summary["reject_day_se"])


def test_own_fit_refusals(rate_sessions_cohort) -> None:
    """Values of an own fit without one, an own fit that gives other
    estimands, and an estimand with units of both levels are refused."""
    sessions = rate_sessions_cohort
    plain = rate_model(sessions)
    with pytest.raises(ValueError, match="own fit"):
        Model(
            "x",
            plain.design,
            Poisson(),
            "count",
            "well_visit_id",
            {},
            fit_estimands=("log_rate",),
        )
    with pytest.raises(ValueError, match="no estimand"):
        Model(
            "x",
            plain.design,
            Poisson(),
            "count",
            "well_visit_id",
            {},
            fit=own_animal_fit,
        )
    scenario = OutsideScenario("outside", expected_rate)
    wrong = Model(
        "x",
        plain.design,
        Poisson(),
        "count",
        "well_visit_id",
        {},
        fit=own_animal_fit,
        fit_estimands=("other",),
    )
    with pytest.raises(ValueError, match="estimands"):
        engine.run(_simulation(wrong, sessions, [scenario]))

    def mixed(data):
        table = own_animal_fit(data)
        table.loc[0, ["session_date", "epoch"]] = ["20260001", "02_r1"]
        return table

    both = Model(
        "x",
        plain.design,
        Poisson(),
        "count",
        "well_visit_id",
        {},
        fit=mixed,
        fit_estimands=("log_rate",),
    )
    with pytest.raises(ValueError, match="both"):
        engine.run(_simulation(both, sessions, [scenario]))


def test_an_own_fit_that_gives_a_session_twice_is_refused(rate_sessions_cohort) -> None:
    """An own fit whose table holds one session's value of an estimand twice
    is refused, as one with an animal twice is."""
    sessions = rate_sessions_cohort
    plain = rate_model(sessions)

    def twice(data):
        table = own_session_fit(data, design=plain.design, estimands=plain.estimands)
        return pd.concat([table, table.iloc[:1]], ignore_index=True)

    def animal_twice(data):
        table = own_animal_fit(data)
        return pd.concat([table, table.iloc[:1]], ignore_index=True)

    scenario = OutsideScenario("outside", expected_rate)
    for fit, names in ((twice, tuple(plain.estimands)), (animal_twice, ("log_rate",))):
        model = Model(
            "x",
            plain.design,
            Poisson(),
            "count",
            "well_visit_id",
            {},
            fit=fit,
            fit_estimands=names,
        )
        with pytest.raises(ValueError, match="more than once"):
            engine.run(_simulation(model, sessions, [scenario]))


def test_an_own_fit_may_be_the_first_model(rate_sessions_cohort) -> None:
    """The draws use the first model's design (its offset, missing rows and,
    under an inside scenario, its coefficients), whether or not that model
    has its own fit: an own fit that does what the engine does, first, gives
    the plain model's estimates on the same data sets."""
    sessions = rate_sessions_cohort
    plain = rate_model(sessions)
    own = Model(
        "own",
        plain.design,
        Poisson(),
        "count",
        "well_visit_id",
        {},
        fit=partial(own_session_fit, design=plain.design, estimands=plain.estimands),
        fit_estimands=tuple(plain.estimands),
    )
    result = engine.run(Simulation([own, plain], sessions, [RATE_INSIDE], 2, 200, 5))
    sets = result.data_sets.set_index(["data_set", "estimand", "model"])["estimate"]
    for _, part in sets.groupby(level=[0, 1]):
        values = part.droplevel([0, 1])
        assert values["own"] == values["rate"]


def test_an_inside_coefficient_must_be_in_the_first_design(
    rate_sessions_cohort,
) -> None:
    """A planted coefficient that the first model's design lacks (a typo) is
    refused before any draw, with an own-fit model in the run or not."""
    sessions = rate_sessions_cohort
    plain = rate_model(sessions)
    own = Model(
        "own",
        plain.design,
        Poisson(),
        "count",
        "well_visit_id",
        {},
        fit=partial(own_session_fit, design=plain.design, estimands=plain.estimands),
        fit_estimands=tuple(plain.estimands),
    )
    typo = InsideScenario("typo", {"(Intercept)": np.log(0.8), "rewrd": 0.5})
    for models in ([plain, own], [own, plain]):
        with pytest.raises(ValueError, match="rewrd.*first model"):
            engine.run(Simulation(models, sessions, [typo], 2, 200, 5))


def reference_tables() -> dict[str, list[dict]]:
    """The summary tables of the fixture: the rate model under the inside and
    the outside scenario, and the content model inside, on fixed seeds."""
    rate_sessions = _cohort(rate_rows)
    content_sessions = _cohort(_content_rows)
    rate = engine.run(
        _simulation(
            rate_model(rate_sessions),
            rate_sessions,
            [
                RATE_INSIDE,
                OutsideScenario("outside", expected_rate, planted={"reward": 0.5}),
            ],
            n_data_sets=4,
        )
    )
    content = engine.run(
        _simulation(
            content_model(content_sessions),
            content_sessions,
            [CONTENT_INSIDE],
            n_data_sets=4,
        )
    )
    return {
        "rate": json.loads(rate.summary.to_json(orient="records")),
        "content": json.loads(content.summary.to_json(orient="records")),
    }


#: The summary's columns that the fixture predates.
ADDED_COLUMNS = {"n_data_sets_with_interval", "n_data_sets_with_day_p"}


@pytest.mark.slow
def test_matches_the_fixed_seed_fixture() -> None:
    """The fixture's tables, to the tolerance of the solver's port
    (``1e-11 + 1e-9 |b|``), with the counts exactly equal. The summary's two
    counts added after the fixture was written (:data:`ADDED_COLUMNS`) are
    not stored: every fixture data set with an estimate has an interval."""
    stored = json.loads(FIXTURE.read_text())["tables"]
    for name, rows in reference_tables().items():
        assert len(rows) == len(stored[name])
        for got, want in zip(rows, stored[name], strict=True):
            assert got.keys() - want.keys() == ADDED_COLUMNS
            assert want.keys() <= got.keys()
            n = got["n_data_sets_with_estimate"]
            assert got["n_data_sets_with_interval"] == n
            assert 0 <= got["n_data_sets_with_day_p"] <= n
            for column, value in want.items():
                if isinstance(value, float):
                    assert got[column] == pytest.approx(
                        value, rel=1e-9, abs=1e-11, nan_ok=True
                    ), (name, column)
                else:
                    assert got[column] == value, (name, column)


if __name__ == "__main__":
    from continuum_swr_replay.runs.provenance import described_commit

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(
        json.dumps(
            {
                "commit": described_commit(),
                "command": "python -m tests.analysis.regression.test_simulation_engine",
                "tables": reference_tables(),
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {FIXTURE}", file=sys.stderr)
