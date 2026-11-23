"""The simulation engine: a planted truth drawn on a model's own rows, fitted
by the real code.

A simulation replaces only the outcome column of a model's rows. The rows, the
design, the per-session fit (:func:`.sessions.fit_sessions`), the estimands
(:func:`.contrasts.evaluate_contrast`) and the reduction session -> recording
day -> animal with the project's interval
(:func:`continuum_swr_replay.analysis.shared_statistics.estimand_summary`) are
the real functions. Nothing here builds a model: the caller gives the
:class:`Model` as the model's own code builds it, and the engine only draws
outcomes (:func:`.likelihoods.draw`) and calls those functions.

A scenario is one of two kinds:

- :class:`InsideScenario`: planted coefficients on the model's own design.
  The linear predictor of a row is its design row times the planted
  coefficients (the Poisson offset is added by the draw), and the planted
  value of each estimand is ``c' beta`` at them.
- :class:`OutsideScenario`: any expected outcome as a function of the rows (a
  curve the model cannot hold exactly, an effect that differs between
  sessions and animals). It has no planted coefficients, so it states the
  planted value of each estimand itself, as a number or as a function of the
  model; an estimand it does not name has the planted value NaN, and its
  table gives the estimate without a bias or a coverage.

The draws use a NumPy generator only, never the fitter, so the same seeds give
the same data whatever the solver. Data set ``i`` of the scenario at place
``s`` in the scenario list is drawn from ``default_rng(data_seed(seed, s,
i))``, the sessions in the order given; the seed therefore follows the
scenario's place in the list. Every model of a run is fitted to the same data
sets. The bootstrap of each (model, estimand) of a data set has its own seed,
the child at place ``m * n_estimands + e`` of ``data_seed(seed, s,
i).spawn(...)`` (:func:`bootstrap_seeds`).

A row with a missing value the design reads gets a missing outcome: the fit
removes it in any case. Under the grouped softmax a group's outcome is drawn
over its rows that are not missing.

What depends only on the rows is computed once per run: each session's
offset, its rows with a missing value, and its linear predictor under each
inside scenario. Each data set is fitted by
``fit_sessions``, which builds a session's design again from the rows; the
fitter's interface has no way to pass a built design in.

The tables of a run (:class:`SimulationResult`):

- ``sessions``: one row per (scenario, model, data set, estimand, session):
  the fit's ``status`` and ``no_fit``, the estimate and its standard error,
  ``without_value`` (empty when the session has a value; otherwise the
  record's ``no_fit`` code, or the ``left_out`` code of the first
  coefficient of the estimand that the session left out) and the record's
  ``edf``;
- ``data_sets``: one row per (scenario, model, data set, estimand): the
  planted value, ``estimand_summary``'s estimate, interval, day-level p and
  sessions used, whether the interval holds the planted value
  (``covered``) and whether it excludes zero (``excludes_zero``);
- ``summary``: one row per (scenario, model, estimand), :data:`SUMMARY_COLUMNS`
  (described in :func:`summarize`).
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.shared_statistics import estimand_summary

from .contrasts import evaluate_contrast
from .likelihoods import GroupedSoftmax, Likelihood, Poisson, draw
from .newton import CONVERGED
from .record import FitRecord, NoFit
from .sessions import NOT_FITTED, SessionKey, fit_sessions
from .support import Removed, require_tested_penalized
from .terms import Design

__all__ = [
    "CAUSES",
    "SUMMARY_COLUMNS",
    "InsideScenario",
    "Model",
    "OutsideScenario",
    "Scenario",
    "Simulation",
    "SimulationResult",
    "bootstrap_seeds",
    "data_seed",
    "draw_data_set",
    "planted_value",
    "run",
    "summarize",
]

#: Every cause of a session without a value of an estimand: a record's
#: ``no_fit`` code, then a ``left_out`` code.
CAUSES = (*(code.value for code in NoFit), *(code.value for code in Removed))

#: A data set rejects by the day-level test when its p is below this.
REJECTION_LEVEL = 0.05


@dataclass(frozen=True)
class Model:
    """A model as its own code builds it, and the estimands a simulation reads.

    Attributes
    ----------
    name : str
        Names the model in the tables and in its fit records.
    design : Design
        Built on the pooled rows of every session (:func:`.terms.build_design`).
    likelihood : Likelihood
    response, unit : str
        The outcome column the simulation replaces, and the unit column, as
        :func:`.sessions.fit_sessions` takes them.
    estimands : mapping of str to np.ndarray
        Estimand name -> its ``c`` over ``design.coef_names``, shape
        ``(n_coef,)`` (:mod:`.contrasts`). Every coefficient an estimand uses
        must carry a fixed penalty (:func:`.support.require_tested_penalized`).
    """

    name: str
    design: Design
    likelihood: Likelihood
    response: str
    unit: str
    estimands: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        n_coef = len(self.design.coef_names)
        if not self.estimands:
            raise ValueError(f"model {self.name!r} has no estimand")
        estimands = {}
        for name, c in self.estimands.items():
            c = np.asarray(c, dtype=float)
            if c.shape != (n_coef,) or not np.all(np.isfinite(c)) or not c.any():
                raise ValueError(
                    f"model {self.name!r}, estimand {name!r}: c must be finite, not "
                    f"all zero and of shape ({n_coef},), not {c.shape}"
                )
            estimands[name] = c
        object.__setattr__(self, "estimands", estimands)
        require_tested_penalized(
            self.design,
            dict.fromkeys(name for c in estimands.values() for name in self.used(c)),
        )

    def used(self, c: np.ndarray) -> list[str]:
        """The coefficient names on which ``c`` is not zero, in column order."""
        return [self.design.coef_names[j] for j in np.flatnonzero(c)]


@dataclass(frozen=True)
class InsideScenario:
    """Planted coefficients on the model's own design.

    Attributes
    ----------
    name : str
    coefficients : mapping of str to float
        Coefficient name (of ``design.coef_names``) -> planted value; a
        coefficient not named is zero.
    scale : float or None
        The Gaussian's variance; required under the Gaussian, not read
        otherwise.
    n_trials : int
        The choices per group of a share draw under the grouped softmax
        (:func:`.likelihoods.draw`).
    """

    name: str
    coefficients: Mapping[str, float]
    scale: float | None = None
    n_trials: int = 1


#: Each session's rows -> its expected outcome per row, shape ``(n_rows,)``:
#: the mean count (Poisson, the exposure included), the probability
#: (Bernoulli), the mean (Gaussian) or the share (grouped softmax, summing to
#: one within each group). It may read the generator, for effects drawn anew
#: in each data set; it must be a module-level function so that a process
#: pool can send it.
ExpectedOutcome = Callable[
    [Mapping[SessionKey, pd.DataFrame], np.random.Generator],
    Mapping[SessionKey, np.ndarray],
]


@dataclass(frozen=True)
class OutsideScenario:
    """Any expected outcome as a function of the rows.

    Attributes
    ----------
    name : str
    expected : callable
        :data:`ExpectedOutcome`.
    planted : mapping of str to float or callable
        Estimand name -> its planted value, a number or a function of the
        :class:`Model` that returns one (a module-level function, for a run
        with workers); an estimand not named has the planted value NaN.
    scale, n_trials
        As :class:`InsideScenario`.
    """

    name: str
    expected: ExpectedOutcome
    planted: Mapping[str, float | Callable[[Model], float]] = field(
        default_factory=dict
    )
    scale: float | None = None
    n_trials: int = 1


Scenario = InsideScenario | OutsideScenario


@dataclass(frozen=True)
class Simulation:
    """One run: the models, their rows, the scenarios, the run's size and seed.

    A scenario list beside a model builds one (the runner,
    ``scripts/checks/run_simulations.py``, calls it).

    Attributes
    ----------
    models : sequence of Model
        Fitted to the same data sets: the same coefficient names, likelihood,
        response and unit (they may differ in their penalties or estimands).
        An :class:`InsideScenario` is drawn on the first model's design.
    sessions : mapping of (animal, session_date, epoch) to pd.DataFrame
        The model's real rows of each session; a missing response column is
        added.
    scenarios : sequence of Scenario
    n_data_sets : int
        Data sets per scenario.
    n_draws : int
        Bootstrap replicates of each interval.
    seed : int
        The run's seed (module docstring).
    attrs : mapping
        Facts about how the models were made (a calibration), written with
        the table.
    """

    models: Sequence[Model]
    sessions: Mapping[SessionKey, pd.DataFrame]
    scenarios: Sequence[Scenario]
    n_data_sets: int
    n_draws: int
    seed: int
    attrs: Mapping[str, object] = field(default_factory=dict)


class SimulationResult(NamedTuple):
    """The three tables of a run (module docstring)."""

    summary: pd.DataFrame
    data_sets: pd.DataFrame
    sessions: pd.DataFrame


def data_seed(seed: int, scenario: int, data_set: int) -> np.random.SeedSequence:
    """The seed of data set ``data_set`` of the scenario at place ``scenario``."""
    return np.random.SeedSequence([seed, scenario, data_set])


def bootstrap_seeds(seed: int, scenario: int, data_set: int, n: int) -> list[int]:
    """The bootstrap seeds of a data set's ``n`` (model, estimand) pairs,
    spawned from its :func:`data_seed`."""
    return [
        int(child.generate_state(1)[0])
        for child in data_seed(seed, scenario, data_set).spawn(n)
    ]


def planted_value(scenario: Scenario, model: Model, estimand: str) -> float:
    """The planted value of ``estimand`` of ``model`` under ``scenario``:
    ``c' beta`` inside the model, the scenario's own value outside it (NaN
    when it names none)."""
    if isinstance(scenario, InsideScenario):
        return float(model.estimands[estimand] @ _planted_beta(scenario, model.design))
    value = scenario.planted.get(estimand, np.nan)
    return float(value(model) if callable(value) else value)


def _planted_beta(scenario: InsideScenario, design: Design) -> np.ndarray:
    """The planted coefficients over ``design.coef_names``, shape ``(n_coef,)``."""
    names = list(design.coef_names)
    unknown = [name for name in scenario.coefficients if name not in names]
    if unknown:
        raise ValueError(
            f"scenario {scenario.name!r}: {unknown} not coefficients of the design"
        )
    beta = np.zeros(len(names))
    for name, value in scenario.coefficients.items():
        beta[names.index(name)] = value
    return beta


@dataclass(frozen=True)
class _Prepared:
    """What a session's draws need from its rows, computed once: the offset,
    the rows with a missing value, the unit code of each row, and the linear
    predictor of each inside scenario (by its place in the list)."""

    offset: np.ndarray | None  # (n_rows,)
    missing: np.ndarray  # (n_rows,) bool
    unit: np.ndarray  # (n_rows,) int
    inside_eta: dict[int, np.ndarray]  # place -> (n_rows,)


def _prepare(
    model: Model,
    sessions: Mapping[SessionKey, pd.DataFrame],
    scenarios: Sequence[Scenario],
) -> dict[SessionKey, _Prepared]:
    """Each session's :class:`_Prepared`. The design matrix itself is not
    kept: only its product with each inside scenario's coefficients."""
    betas = {
        s: _planted_beta(scenario, model.design)
        for s, scenario in enumerate(scenarios)
        if isinstance(scenario, InsideScenario)
    }
    out = {}
    for key, rows in sessions.items():
        built = model.design.build(rows)
        unit = pd.factorize(rows[model.unit].to_numpy(), use_na_sentinel=False)[0]
        etas = {s: built.X @ beta for s, beta in betas.items()}
        out[key] = _Prepared(built.offset, built.missing, unit, etas)
    return out


def _linear_predictor(
    lik: Likelihood, mean: np.ndarray, offset: np.ndarray | None
) -> np.ndarray:
    """The ``eta`` at which :func:`.likelihoods.draw` has mean ``mean``: the
    inverse link, less the offset under the Poisson; under the softmax the
    log share (a constant within a group does not matter)."""
    with np.errstate(divide="ignore"):
        if lik.name == "poisson":
            return np.log(mean) - (0.0 if offset is None else offset)
        if lik.name == "bernoulli":
            return np.log(mean) - np.log1p(-mean)
        if lik.name == "grouped_softmax":
            return np.log(mean)
    return mean


def _check_expected(lik: Likelihood, mean: np.ndarray, where: str) -> None:
    """Refuse an expected outcome outside what the family takes."""
    if not np.all(np.isfinite(mean)):
        raise ValueError(f"{where}: an expected outcome is not finite")
    if lik.name in ("poisson", "grouped_softmax") and np.any(mean < 0):
        raise ValueError(f"{where}: an expected outcome is below zero")
    if lik.name == "bernoulli" and np.any((mean < 0) | (mean > 1)):
        raise ValueError(f"{where}: an expected probability is outside [0, 1]")


def _draw_session(
    model: Model,
    scenario: Scenario,
    prepared: _Prepared,
    eta: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """One session's outcomes at ``eta``, NaN on its rows with a missing
    value, shape ``(n_rows,)``."""
    lik = model.likelihood
    y = np.full(eta.size, np.nan)
    rows = np.flatnonzero(~prepared.missing)
    if isinstance(lik, GroupedSoftmax):
        rows = rows[np.argsort(prepared.unit[rows], kind="stable")]
        aux = prepared.unit[rows]
    elif isinstance(lik, Poisson):
        aux = np.zeros(rows.size) if prepared.offset is None else prepared.offset[rows]
    elif lik.name == "gaussian":
        if scenario.scale is None:
            raise ValueError(f"scenario {scenario.name!r}: the Gaussian needs a scale")
        aux = scenario.scale
    else:
        aux = None
    y[rows] = draw(lik, eta[rows], aux, rng, n_trials=scenario.n_trials)
    return y


def _draw(
    model: Model,
    sessions: Mapping[SessionKey, pd.DataFrame],
    prepared: Mapping[SessionKey, _Prepared],
    scenario: Scenario,
    place: int,
    rng: np.random.Generator,
) -> dict[SessionKey, pd.DataFrame]:
    """One data set of the scenario at ``place`` (module docstring)."""
    if isinstance(scenario, InsideScenario):
        etas = {key: p.inside_eta[place] for key, p in prepared.items()}
    else:
        means = scenario.expected(sessions, rng)
        if set(means) != set(sessions):
            raise ValueError(
                f"scenario {scenario.name!r}: the expected outcome is not given for "
                "exactly the sessions given"
            )
        etas = {}
        for key, p in prepared.items():
            mean = np.asarray(means[key], dtype=float)
            where = f"scenario {scenario.name!r}, session {'/'.join(key)}"
            if mean.shape != p.missing.shape:
                raise ValueError(f"{where}: {mean.shape} expected outcomes")
            _check_expected(model.likelihood, mean[~p.missing], where)
            etas[key] = _linear_predictor(model.likelihood, mean, p.offset)
    return {
        key: rows.assign(
            **{
                model.response: _draw_session(
                    model, scenario, prepared[key], etas[key], rng
                )
            }
        )
        for key, rows in sessions.items()
    }


def draw_data_set(
    model: Model,
    sessions: Mapping[SessionKey, pd.DataFrame],
    scenario: Scenario,
    rng: np.random.Generator,
) -> dict[SessionKey, pd.DataFrame]:
    """One data set: each session's rows with the outcome drawn under
    ``scenario`` (module docstring).

    Parameters
    ----------
    model : Model
    sessions : mapping of (animal, session_date, epoch) to pd.DataFrame
        The model's rows of each session.
    scenario : Scenario
    rng : np.random.Generator
        Read session by session, in the order given.

    Returns
    -------
    dict of (animal, session_date, epoch) to pd.DataFrame
        The rows, with ``model.response`` replaced (or added).
    """
    prepared = _prepare(model, sessions, [scenario])
    return _draw(model, sessions, prepared, scenario, 0, rng)


def _session_rows(
    model: Model, records: Sequence[FitRecord], base: dict
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """The ``sessions`` table's rows of one model and data set, and each
    estimand's session estimates, shape ``(n_sessions,)``."""
    rows, estimates = [], {name: [] for name in model.estimands}
    for record in records:
        coef, vcov = np.asarray(record.coef), np.asarray(record.vcov)
        for name, c in model.estimands.items():
            left = [record.left_out[n] for n in model.used(c) if n in record.left_out]
            cause = record.no_fit or (left[0] if left else "")
            estimate, se = np.nan, np.nan
            if not cause:
                contrast = evaluate_contrast(c, coef, vcov)
                estimate, se = contrast.estimate, contrast.se
            estimates[name].append(estimate)
            rows.append(
                {
                    **base,
                    "estimand": name,
                    "animal": record.animal,
                    "session_date": record.session_date,
                    "epoch": record.epoch,
                    "status": record.status,
                    "no_fit": record.no_fit,
                    "without_value": cause,
                    "estimate": estimate,
                    "se": se,
                    "edf": record.edf,
                }
            )
    return rows, {name: np.array(v) for name, v in estimates.items()}


def _data_set_row(
    values: np.ndarray,
    keys: list[SessionKey],
    planted: float,
    n_draws: int,
    seed: int,
) -> dict:
    """One estimand's reduction over a data set's sessions (module docstring);
    NaN when no session has a value."""
    index = pd.MultiIndex.from_tuples(keys)
    if not np.isfinite(values).any():
        return {
            "planted": planted,
            "estimate": np.nan,
            "low": np.nan,
            "high": np.nan,
            "interval_sd": np.nan,
            "day_p": np.nan,
            "n_sessions_used": 0,
            "n_animals_same_sign": 0,
            "covered": np.nan,
            "excludes_zero": np.nan,
            "bootstrap_seed": seed,
        }
    labels = [pd.Series([key[k] for key in keys], index=index) for k in range(3)]
    summary = estimand_summary(
        pd.Series(values, index=index), *labels, n_draws=n_draws, seed=seed
    )
    low, high = summary["low"], summary["high"]
    return {
        "planted": planted,
        "estimate": summary["estimate"],
        "low": low,
        "high": high,
        "interval_sd": summary["sd"],
        "day_p": float(summary["day_p"]),
        "n_sessions_used": int(summary["n_sessions"]),
        "n_animals_same_sign": int(summary["n_animals_same_sign"]),
        "covered": float(low <= planted <= high) if np.isfinite(planted) else np.nan,
        "excludes_zero": float(not low <= 0.0 <= high),
        "bootstrap_seed": seed,
    }


def _run_task(
    simulation: Simulation,
    prepared: Mapping[SessionKey, _Prepared],
    task: tuple[int, int],
) -> tuple[list[dict], list[dict]]:
    """One data set of one scenario, fitted by every model: its ``data_sets``
    and ``sessions`` rows."""
    s, i = task
    scenario = simulation.scenarios[s]
    models = simulation.models
    rng = np.random.default_rng(data_seed(simulation.seed, s, i))
    data = _draw(models[0], simulation.sessions, prepared, scenario, s, rng)
    keys = list(data)
    n_estimands = [len(model.estimands) for model in models]
    seeds = iter(bootstrap_seeds(simulation.seed, s, i, sum(n_estimands)))
    data_set_rows, session_rows = [], []
    for model in models:
        records = fit_sessions(
            model.design,
            data,
            model.likelihood,
            model=model.name,
            kind=scenario.name,
            response=model.response,
            unit=model.unit,
            table_sha256={},
        )
        base = {"scenario": scenario.name, "model": model.name, "data_set": i}
        rows, estimates = _session_rows(model, records, base)
        session_rows += rows
        for name, values in estimates.items():
            planted = planted_value(scenario, model, name)
            row = _data_set_row(values, keys, planted, simulation.n_draws, next(seeds))
            data_set_rows.append({**base, "estimand": name, **row})
    return data_set_rows, session_rows


#: A worker process's run, set once by the pool's initializer.
_WORKER: dict = {}


def _init_worker(simulation: Simulation) -> None:
    _WORKER["simulation"] = simulation
    _WORKER["prepared"] = _prepare(
        simulation.models[0], simulation.sessions, simulation.scenarios
    )


def _worker_task(task: tuple[int, int]) -> tuple[list[dict], list[dict]]:
    return _run_task(_WORKER["simulation"], _WORKER["prepared"], task)


def _check(simulation: Simulation) -> None:
    """Refuse a run whose models cannot share data sets or whose names repeat."""
    models = simulation.models
    if not models or not simulation.scenarios:
        raise ValueError("a simulation needs a model and a scenario")
    first = models[0]
    for model in models[1:]:
        if (
            model.design.coef_names != first.design.coef_names
            or model.likelihood.name != first.likelihood.name
            or (model.response, model.unit) != (first.response, first.unit)
        ):
            raise ValueError(
                f"model {model.name!r} cannot share data sets with {first.name!r}: "
                "the coefficient names, likelihood, response and unit must agree"
            )
    for what, names in (
        ("model", [m.name for m in models]),
        ("scenario", [s.name for s in simulation.scenarios]),
    ):
        if len(set(names)) != len(names):
            raise ValueError(f"{what} names repeat: {names}")
    if simulation.n_data_sets < 2:
        raise ValueError("a simulation needs at least two data sets per scenario")


def run(
    simulation: Simulation,
    workers: int = 1,
    scenarios: Sequence[str] | None = None,
) -> SimulationResult:
    """Every data set of every scenario, fitted by every model, and their
    tables (module docstring).

    Parameters
    ----------
    simulation : Simulation
    workers : int
        Processes; above one, a ``spawn`` pool whose workers each receive the
        run once. The tables do not depend on it.
    scenarios : sequence of str or None
        Run only the scenarios of these names; each keeps its place in the
        list, so its data sets are those of a run of every scenario.

    Returns
    -------
    SimulationResult

    Raises
    ------
    ValueError
        If the models cannot share data sets, a name repeats or is not a
        scenario, a scenario names a coefficient the design does not have, an
        expected outcome is outside what the family takes, or a session's
        first fit fails (:func:`.sessions.fit_sessions`).
    """
    _check(simulation)
    names = [scenario.name for scenario in simulation.scenarios]
    unknown = sorted(set(scenarios or ()) - set(names))
    if unknown:
        raise ValueError(f"{unknown} are not scenarios of the run: {names}")
    tasks = [
        (s, i)
        for s, name in enumerate(names)
        if scenarios is None or name in scenarios
        for i in range(simulation.n_data_sets)
    ]
    if workers == 1:
        prepared = _prepare(
            simulation.models[0], simulation.sessions, simulation.scenarios
        )
        parts = [_run_task(simulation, prepared, task) for task in tasks]
    else:
        context = multiprocessing.get_context("spawn")
        with context.Pool(workers, _init_worker, (simulation,)) as pool:
            parts = pool.map(_worker_task, tasks, chunksize=1)
    data_sets = pd.DataFrame([row for part in parts for row in part[0]])
    sessions = pd.DataFrame([row for part in parts for row in part[1]])
    return SimulationResult(
        summarize(data_sets, sessions, simulation), data_sets, sessions
    )


#: The columns of the summary, in order (:func:`summarize`).
SUMMARY_COLUMNS = (
    "scenario",
    "model",
    "estimand",
    "planted",
    "mean_estimate",
    "bias",
    "bias_share",
    "sd",
    "bias_se",
    "coverage",
    "coverage_se",
    "excludes_zero",
    "excludes_zero_se",
    "reject_day",
    "reject_day_se",
    "share_without_value",
    *(f"without_value_{cause}" for cause in CAUSES),
    "share_not_converged",
    "n_sessions",
    "n_sessions_used_min",
    "n_sessions_used_mean",
    "n_data_sets",
    "n_data_sets_with_estimate",
    "n_draws",
    "seed",
)


def _share(hits: np.ndarray, name: str) -> dict[str, float]:
    """The share of ``hits`` (0/1) and its binomial standard error; NaN
    without any."""
    if hits.size == 0:
        return {name: np.nan, f"{name}_se": np.nan}
    p = float(hits.mean())
    return {name: p, f"{name}_se": float(np.sqrt(p * (1 - p) / hits.size))}


def summarize(
    data_sets: pd.DataFrame, sessions: pd.DataFrame, simulation: Simulation
) -> pd.DataFrame:
    """One row per (scenario, model, estimand), in the order of the run.

    The data sets are those whose estimate is finite (with a session with a
    value): ``mean_estimate`` and ``sd`` over them (``sd`` with ``ddof=1``),
    ``bias = mean_estimate - planted``, ``bias_share = bias / planted`` (NaN
    for a planted value of zero or NaN), ``bias_se = sd / sqrt(n)``;
    ``coverage``, the share whose 95 % interval holds the planted value (NaN
    when it is NaN); ``excludes_zero``, the share whose interval excludes
    zero; ``reject_day``, the share whose day-level p is below
    :data:`REJECTION_LEVEL`, a NaN p (fewer than six non-zero day values)
    counting as no rejection. Each share has its binomial standard error
    (``*_se``). Under a planted value of zero ``excludes_zero`` and
    ``reject_day`` are false-positive rates, otherwise power.

    Over every session of every data set: ``share_without_value`` and, for
    each code of :data:`CAUSES`, ``without_value_<code>``, the share of
    sessions without a value for that cause; ``share_not_converged``, the
    share of the session fits tried (status not ``not fitted``) that did not
    converge. ``n_sessions`` is the cohort's, ``n_sessions_used_min`` and
    ``n_sessions_used_mean`` the sessions in the estimates; ``n_data_sets``
    and ``n_data_sets_with_estimate``; ``n_draws`` and ``seed`` the run's.
    """
    out = []
    keys = ["scenario", "model", "estimand"]
    by_session = sessions.groupby(keys, sort=False)
    for (scenario, model, estimand), group in data_sets.groupby(keys, sort=False):
        fits = by_session.get_group((scenario, model, estimand))
        planted = float(group["planted"].iloc[0])
        found = group[np.isfinite(group["estimate"])]
        estimate = found["estimate"].to_numpy()
        n = estimate.size
        mean = float(estimate.mean()) if n else np.nan
        sd = float(np.std(estimate, ddof=1)) if n > 1 else np.nan
        bias = mean - planted
        cause = fits["without_value"].to_numpy()
        tried = fits["status"].to_numpy() != NOT_FITTED
        out.append(
            {
                "scenario": scenario,
                "model": model,
                "estimand": estimand,
                "planted": planted,
                "mean_estimate": mean,
                "bias": bias,
                "bias_share": bias / planted if planted != 0 else np.nan,
                "sd": sd,
                "bias_se": sd / np.sqrt(n) if n else np.nan,
                **(
                    _share(found["covered"].to_numpy(), "coverage")
                    if np.isfinite(planted)
                    else {"coverage": np.nan, "coverage_se": np.nan}
                ),
                **_share(found["excludes_zero"].to_numpy(), "excludes_zero"),
                **_share(
                    (found["day_p"] < REJECTION_LEVEL).to_numpy(dtype=float),
                    "reject_day",
                ),
                "share_without_value": float(np.mean(cause != "")),
                **{
                    f"without_value_{code}": float(np.mean(cause == code))
                    for code in CAUSES
                },
                "share_not_converged": (
                    float(np.mean(fits["status"].to_numpy()[tried] != CONVERGED))
                    if tried.any()
                    else np.nan
                ),
                "n_sessions": len(simulation.sessions),
                "n_sessions_used_min": int(group["n_sessions_used"].min()),
                "n_sessions_used_mean": float(group["n_sessions_used"].mean()),
                "n_data_sets": len(group),
                "n_data_sets_with_estimate": n,
                "n_draws": simulation.n_draws,
                "seed": simulation.seed,
            }
        )
    return pd.DataFrame(out, columns=list(SUMMARY_COLUMNS))
