"""The simulation engine: a planted truth drawn on a model's own rows, fitted
by the real code.

A simulation replaces only the outcome column of a model's rows. The rows, the
design, the fit of each unit (:func:`.sessions.fit_sessions` at the model's
``fit_by``: a session, a recording day or an animal), the estimands
(:func:`.contrasts.evaluate_contrast`) and the reduction over the units with
the project's interval
(:func:`continuum_swr_replay.analysis.shared_statistics.unit_summary`) are the
real functions. Nothing here builds a model: the caller gives the
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
sets, so models that differ only in their ``fit_by`` fit the same draws: one
run fits each data set at several units. The bootstrap of each (model, estimand) of a data set has its own seed,
the child at place ``m * n_estimands + e`` of ``data_seed(seed, s,
i).spawn(...)`` (:func:`bootstrap_seeds`).

A row with a missing value the design reads gets a missing outcome: the fit
removes it in any case. Under the grouped softmax a group's outcome is drawn
over its rows that are not missing, by the noise model the scenario chooses
(``n_trials`` and ``concentration``: one categorical choice per event, a
multinomial, or a Dirichlet-multinomial; :func:`.likelihoods.draw`).

What depends only on the rows is computed once per run: each session's
offset, its rows with a missing value, and its linear predictor under each
inside scenario (:func:`prepare_draws` does the same for a caller of
:func:`draw_data_set`). A run keeps of each session's rows only the columns
it reads (:func:`run_columns`), and a data set's rows with the drawn outcome
are made one session at a time, as the fit reads them. Each data set is
fitted by ``fit_sessions``, which builds a session's design again from the
rows; the fitter's interface has no way to pass a built design in.

Two planted values (``docs/design/2026-11-24-unit-of-fit.md``, section 6):

- ``planted``, the present target: the planted value of the estimand, the
  session -> day -> animal mean of the sessions' values (``c' beta`` inside
  the model; the scenario's own value outside it);
- ``planted_unit``, the unit's own target: the model fitted by the real fitter
  at its ``fit_by`` to the expected outcome itself (no noise), with the weak
  penalty off (every simple term's fixed penalty removed; a smooth keeps its
  own), reduced over the units as the estimates are. It is computed once per
  worker process for an inside scenario and once per data set for an outside
  one (whose expected outcome may read the generator). The Bernoulli takes
  only 0/1 outcomes, so under it ``planted_unit`` is NaN. The noise-free
  target never drops a unit for lack of events, and a drawn data set can, so
  ``bias_unit`` can compare slightly different sets of units.

The tables of a run (:class:`SimulationResult`):

- ``sessions``: one row per (scenario, model, data set, estimand, unit of the
  fit): the unit's key (``""`` for the parts a unit does not have), ``fit_by``,
  its numbers of sessions and events, the fit's ``status`` and ``no_fit``, the
  estimate and its standard error, ``without_value`` (empty when the unit has
  a value; otherwise the record's ``no_fit`` code, or the ``left_out`` code
  of the first coefficient of the estimand that the fit left out) and the
  record's ``edf``;
- ``data_sets``: one row per (scenario, model, data set, estimand): the two
  planted values, ``unit_summary``'s estimate, interval and day-level p, the
  units and sessions used, whether the interval holds each planted value
  (``covered``, ``covered_unit``) and whether it excludes zero
  (``excludes_zero``), each NaN without a finite interval;
- ``summary``: one row per (scenario, model, estimand), :data:`SUMMARY_COLUMNS`
  (described in :func:`summarize`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.special import expit

from continuum_swr_replay.analysis.shared_statistics import unit_summary
from continuum_swr_replay.runs.threads import pool_context
from continuum_swr_replay.tables.session_keys import FIT_BY_COLUMNS, FitBy

from .contrasts import evaluate_contrast
from .likelihoods import (
    Bernoulli,
    GroupedSoftmax,
    Likelihood,
    Poisson,
    draw,
    group_starts,
)
from .newton import CONVERGED
from .record import FitRecord, NoFit
from .sessions import NOT_FITTED, SessionKey, fit_sessions
from .support import Removed, require_tested_penalized
from .terms import Design

__all__ = [
    "CAUSES",
    "SHARE_TOLERANCE",
    "SUMMARY_COLUMNS",
    "InsideScenario",
    "Model",
    "OutsideScenario",
    "PreparedDraws",
    "Scenario",
    "Simulation",
    "SimulationResult",
    "bootstrap_seeds",
    "data_seed",
    "draw_data_set",
    "planted_value",
    "prepare_draws",
    "run",
    "run_columns",
    "summarize",
]

#: Every cause of a session without a value of an estimand: a record's
#: ``no_fit`` code, then a ``left_out`` code.
CAUSES = (*(code.value for code in NoFit), *(code.value for code in Removed))

#: Under the grouped softmax, the most an event's expected shares may differ
#: from summing to one.
SHARE_TOLERANCE = 1e-9

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
        must carry a fixed penalty (:func:`.support.require_tested_penalized`)
        and none may be of a baseline term (copied per session in a fit of
        several sessions; a fit's ``c`` is ``c`` mapped onto its coefficient
        names by name).
    fit_by : {"session", "day", "animal"}
        The unit of the fit.
    """

    name: str
    design: Design
    likelihood: Likelihood
    response: str
    unit: str
    estimands: Mapping[str, np.ndarray]
    fit_by: FitBy = "session"

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
        if self.fit_by not in FIT_BY_COLUMNS:
            raise ValueError(f"model {self.name!r}: fit_by {self.fit_by!r}")
        baseline = [
            n
            for c in estimands.values()
            for n in self.used(c)
            if self.design.column_terms[self.design.coef_names.index(n)].term
            in self.design.baseline
        ]
        if baseline:
            raise ValueError(
                f"model {self.name!r}: an estimand reads the baseline {baseline}"
            )
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
    n_trials : int or str
        Under the grouped softmax, the choices per event of the share draw
        (:func:`.likelihoods.draw`): a number, or the name of a column of the
        rows that holds it for each event (read on the event's first row;
        for example the event's number of decode bins). 1 with no
        concentration is one categorical draw per event; more is a
        multinomial. Not read otherwise.
    concentration : float, str or None
        Under the grouped softmax, the Dirichlet concentration of an
        overdispersed (Dirichlet-multinomial) share draw: a number, or the
        name of a column that holds it for each event; None for none. Not
        read otherwise.
    """

    name: str
    coefficients: Mapping[str, float]
    scale: float | None = None
    n_trials: int | str = 1
    concentration: float | str | None = None


#: Each session's rows -> its expected outcome per row, shape ``(n_rows,)``:
#: the mean count (Poisson, the exposure included), the probability
#: (Bernoulli), the mean (Gaussian) or the share (grouped softmax, summing to
#: one within each group, to :data:`SHARE_TOLERANCE`). It may read the
#: generator, for effects drawn anew in each data set; it must be a
#: module-level function so that a process pool can send it.
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
    scale, n_trials, concentration
        As :class:`InsideScenario`.
    columns : tuple of str
        The columns ``expected`` reads that the models do not (a run gives
        it only the columns it keeps, :func:`run_columns`).
    """

    name: str
    expected: ExpectedOutcome
    planted: Mapping[str, float | Callable[[Model], float]] = field(
        default_factory=dict
    )
    scale: float | None = None
    n_trials: int | str = 1
    concentration: float | str | None = None
    columns: tuple[str, ...] = ()


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


def _check_expected(
    lik: Likelihood, mean: np.ndarray, units: np.ndarray, where: str
) -> None:
    """Refuse an expected outcome outside what the family takes: under the
    grouped softmax also an event (a unit of ``units``, shape ``(n_rows,)``)
    whose shares sum to zero or differ from one by more than
    :data:`SHARE_TOLERANCE`."""
    if not np.all(np.isfinite(mean)):
        raise ValueError(f"{where}: an expected outcome is not finite")
    if lik.name in ("poisson", "grouped_softmax") and np.any(mean < 0):
        raise ValueError(f"{where}: an expected outcome is below zero")
    if lik.name == "bernoulli" and np.any((mean < 0) | (mean > 1)):
        raise ValueError(f"{where}: an expected probability is outside [0, 1]")
    if lik.name == "grouped_softmax" and mean.size:
        codes, events = pd.factorize(units, use_na_sentinel=False)
        sums = np.bincount(codes, weights=mean)
        zero = np.flatnonzero(sums == 0)
        if zero.size:
            raise ValueError(
                f"{where}: the expected shares of {zero.size} events sum to zero "
                f"(the first, {events[zero[0]]!r}): an event needs a share"
            )
        off = np.flatnonzero(np.abs(sums - 1.0) > SHARE_TOLERANCE)
        if off.size:
            raise ValueError(
                f"{where}: the expected shares of {off.size} events do not sum to "
                f"one within {SHARE_TOLERANCE:g}; those of the first, "
                f"{events[off[0]]!r}, sum to {sums[off[0]]:.6g}, not one"
            )


def _per_event(value: object, frame: pd.DataFrame, first: np.ndarray) -> object:
    """A noise parameter of the softmax draw: a number as it is; a column's
    value on each event's first row, shape ``(n_events,)``."""
    if isinstance(value, str):
        return frame[value].to_numpy(dtype=float)[first]
    return value


def _draw_session(
    model: Model,
    scenario: Scenario,
    frame: pd.DataFrame,
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
        first = rows[group_starts(aux)]
        y[rows] = draw(
            lik,
            eta[rows],
            aux,
            rng,
            n_trials=_per_event(scenario.n_trials, frame, first),
            concentration=_per_event(scenario.concentration, frame, first),
        )
        return y
    elif isinstance(lik, Poisson):
        aux = np.zeros(rows.size) if prepared.offset is None else prepared.offset[rows]
    elif lik.name == "gaussian":
        if scenario.scale is None:
            raise ValueError(f"scenario {scenario.name!r}: the Gaussian needs a scale")
        aux = scenario.scale
    else:
        aux = None
    y[rows] = draw(lik, eta[rows], aux, rng)
    return y


def _inside_mean(lik: Likelihood, eta: np.ndarray, prepared: _Prepared) -> np.ndarray:
    """The expected outcome at ``eta`` (what :func:`.likelihoods.draw` has as
    its mean), NaN on the rows with a missing value, shape ``(n_rows,)``:
    under the grouped softmax each event's shares over its rows that are not
    missing."""
    mean = np.full(eta.size, np.nan)
    rows = ~prepared.missing
    if lik.name == "poisson":
        offset = 0.0 if prepared.offset is None else prepared.offset[rows]
        mean[rows] = np.exp(eta[rows] + offset)
    elif lik.name == "bernoulli":
        mean[rows] = expit(eta[rows])
    elif lik.name == "grouped_softmax":
        codes = pd.factorize(prepared.unit[rows])[0]
        top = np.full(codes.max() + 1 if codes.size else 0, -np.inf)
        np.maximum.at(top, codes, eta[rows])
        weight = np.exp(eta[rows] - top[codes])
        mean[rows] = weight / np.bincount(codes, weights=weight)[codes]
    else:
        mean[rows] = eta[rows]
    return mean


def _draw(
    model: Model,
    sessions: Mapping[SessionKey, pd.DataFrame],
    prepared: Mapping[SessionKey, _Prepared],
    scenario: Scenario,
    place: int,
    rng: np.random.Generator,
) -> tuple[dict[SessionKey, np.ndarray], dict[SessionKey, np.ndarray] | None]:
    """The outcomes of one data set of the scenario at ``place`` (module
    docstring), each session's, shape ``(n_rows,)``; and, for an outside
    scenario, each session's expected outcome (None inside the model, whose
    expected outcome does not change between data sets)."""
    means = None
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
            units = sessions[key][model.unit].to_numpy()[~p.missing]
            _check_expected(model.likelihood, mean[~p.missing], units, where)
            etas[key] = _linear_predictor(model.likelihood, mean, p.offset)
    outcomes = {
        key: _draw_session(model, scenario, rows, prepared[key], etas[key], rng)
        for key, rows in sessions.items()
    }
    return outcomes, means


class _DataSet(Mapping):
    """A data set: each session's rows with its drawn outcome, made only when
    the session is read, so that the rows of every session are not copied at
    once."""

    def __init__(
        self,
        sessions: Mapping[SessionKey, pd.DataFrame],
        response: str,
        outcomes: Mapping[SessionKey, np.ndarray],
    ) -> None:
        self._sessions, self._response, self._outcomes = sessions, response, outcomes

    def __getitem__(self, key: SessionKey) -> pd.DataFrame:
        return self._sessions[key].assign(**{self._response: self._outcomes[key]})

    def __iter__(self):
        return iter(self._sessions)

    def __len__(self) -> int:
        return len(self._sessions)


@dataclass(frozen=True)
class PreparedDraws:
    """What the draws of one model under one scenario need from each
    session's rows, computed once (:func:`prepare_draws`), so that many data
    sets are drawn without building each session's design again."""

    model: Model
    scenario: Scenario
    sessions: Mapping[SessionKey, _Prepared]


def prepare_draws(
    model: Model, sessions: Mapping[SessionKey, pd.DataFrame], scenario: Scenario
) -> PreparedDraws:
    """The :class:`PreparedDraws` of ``model`` and ``scenario`` on
    ``sessions``, for :func:`draw_data_set`."""
    return PreparedDraws(model, scenario, _prepare(model, sessions, [scenario]))


def draw_data_set(
    model: Model,
    sessions: Mapping[SessionKey, pd.DataFrame],
    scenario: Scenario,
    rng: np.random.Generator,
    prepared: PreparedDraws | None = None,
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
    prepared : PreparedDraws or None
        :func:`prepare_draws` of this model, scenario and sessions, to draw
        many data sets with one preparation; None prepares them here. The
        draw is the same either way.

    Returns
    -------
    dict of (animal, session_date, epoch) to pd.DataFrame
        The rows, with ``model.response`` replaced (or added).

    Raises
    ------
    ValueError
        If ``prepared`` is of another model, scenario or set of sessions.
    """
    if prepared is None:
        prepared = prepare_draws(model, sessions, scenario)
    elif (
        prepared.model is not model
        or prepared.scenario is not scenario
        or list(prepared.sessions) != list(sessions)
    ):
        raise ValueError(
            "the prepared draws are of another model, scenario or set of sessions"
        )
    outcomes, _ = _draw(model, sessions, prepared.sessions, scenario, 0, rng)
    return dict(_DataSet(sessions, model.response, outcomes))


def _unit_c(c: np.ndarray, model: Model, record: FitRecord) -> np.ndarray:
    """``c`` (over the model's coefficient names) over the fit's coefficient
    names, by name, shape ``(n_coef_of_the_fit,)``: an estimand reads no
    baseline, so every name it uses is a coefficient of every fit."""
    index = {name: j for j, name in enumerate(record.coef_names)}
    out = np.zeros(len(record.coef_names))
    for j in np.flatnonzero(c):
        out[index[model.design.coef_names[j]]] = c[j]
    return out


def _unit_key(record: FitRecord) -> tuple:
    """The key of the unit a record is of."""
    full = (record.animal, record.session_date, record.epoch)
    return full[: len(FIT_BY_COLUMNS[record.fit_by])]


def _session_rows(
    model: Model, records: Sequence[FitRecord], base: dict
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """The ``sessions`` table's rows of one model and data set, one per unit
    of the fit, and each estimand's unit estimates, shape ``(n_units,)``."""
    rows, estimates = [], {name: [] for name in model.estimands}
    for record in records:
        coef, vcov = np.asarray(record.coef), np.asarray(record.vcov)
        for name, c in model.estimands.items():
            left = [record.left_out[n] for n in model.used(c) if n in record.left_out]
            cause = record.no_fit or (left[0] if left else "")
            estimate, se = np.nan, np.nan
            if not cause:
                contrast = evaluate_contrast(_unit_c(c, model, record), coef, vcov)
                estimate, se = contrast.estimate, contrast.se
            estimates[name].append(estimate)
            rows.append(
                {
                    **base,
                    "estimand": name,
                    "animal": record.animal,
                    "session_date": record.session_date,
                    "epoch": record.epoch,
                    "fit_by": record.fit_by,
                    "n_sessions": sum(e["n_rows"] > 0 for e in record.sessions),
                    "n_events": record.n_events,
                    "status": record.status,
                    "no_fit": record.no_fit,
                    "without_value": cause,
                    "estimate": estimate,
                    "se": se,
                    "edf": record.edf,
                }
            )
    return rows, {name: np.array(v) for name, v in estimates.items()}


def _unit_values(values: np.ndarray, records: Sequence[FitRecord]) -> pd.Series:
    """One estimate per unit, indexed by the unit's key."""
    keys = [_unit_key(r) for r in records]
    index = pd.Index([k[0] for k in keys]) if len(keys[0]) == 1 else None
    if index is None:
        index = pd.MultiIndex.from_tuples(keys)
    return pd.Series(values, index=index)


def _reduced_estimate(values: np.ndarray, records: Sequence[FitRecord]) -> float:
    """The estimate of :func:`unit_summary` over the units (it does not read
    the bootstrap's draws); NaN when no unit has a value."""
    if not np.isfinite(values).any():
        return np.nan
    fit_by = records[0].fit_by
    return float(
        unit_summary(_unit_values(values, records), fit_by, n_draws=2, seed=0)[
            "estimate"
        ]
    )


def _data_set_row(
    values: np.ndarray,
    records: Sequence[FitRecord],
    planted: float,
    planted_unit: float,
    n_draws: int,
    seed: int,
) -> dict:
    """One estimand's reduction over a data set's units (module docstring);
    NaN when no unit has a value. ``covered``, ``covered_unit`` and
    ``excludes_zero`` are NaN without a finite interval (none with fewer than
    two animals), so such a data set counts in none of those rates."""
    fit_by = records[0].fit_by
    if not np.isfinite(values).any():
        return {
            "fit_by": fit_by,
            "planted": planted,
            "planted_unit": planted_unit,
            "estimate": np.nan,
            "low": np.nan,
            "high": np.nan,
            "interval_sd": np.nan,
            "day_p": np.nan,
            "n_sessions_used": 0,
            "n_units_used": 0,
            "n_animals_same_sign": 0,
            "covered": np.nan,
            "covered_unit": np.nan,
            "excludes_zero": np.nan,
            "bootstrap_seed": seed,
        }
    summary = unit_summary(
        _unit_values(values, records), fit_by, n_draws=n_draws, seed=seed
    )
    low, high = summary["low"], summary["high"]
    interval = bool(np.isfinite(low) and np.isfinite(high))

    def covers(value: float) -> float:
        return (
            float(low <= value <= high) if interval and np.isfinite(value) else np.nan
        )

    return {
        "fit_by": fit_by,
        "planted": planted,
        "planted_unit": planted_unit,
        "estimate": summary["estimate"],
        "low": low,
        "high": high,
        "interval_sd": summary["sd"],
        "day_p": float(summary["day_p"]),
        "n_sessions_used": int(
            sum(
                e["n_rows"] > 0
                for r, v in zip(records, values, strict=True)
                if np.isfinite(v)
                for e in r.sessions
            )
        ),
        "n_units_used": int(summary["n_units"]),
        "n_animals_same_sign": int(summary["n_animals_same_sign"]),
        "covered": covers(planted),
        "covered_unit": covers(planted_unit),
        "excludes_zero": float(not low <= 0.0 <= high) if interval else np.nan,
        "bootstrap_seed": seed,
    }


def _without_weak_penalty(design: Design) -> Design:
    """``design`` with the fixed penalty of every simple term (a level, a 0/1
    or a continuous slope) removed; a smooth or a lag functional keeps its
    own."""
    simple = ("level", "binary", "slope")
    terms, blocks, rho = [], [], []
    for term in design.terms:
        block = None
        if term.block is not None and term.kind not in simple:
            block = len(blocks)
            blocks.append(design.blocks[term.block])
            rho.append(design.log_penalties[term.block])
        terms.append(replace(term, block=block))
    return replace(
        design,
        terms=tuple(terms),
        blocks=tuple(blocks),
        log_penalties=np.array(rho, dtype=float),
    )


def _unit_targets(
    model: Model,
    sessions: Mapping[SessionKey, pd.DataFrame],
    means: Mapping[SessionKey, np.ndarray],
) -> dict[str, float]:
    """Each estimand's unit target (module docstring): the model at its
    ``fit_by`` fitted to the expected outcome ``means`` with the weak penalty
    off, reduced over the units; NaN under the Bernoulli."""
    if isinstance(model.likelihood, Bernoulli):
        return dict.fromkeys(model.estimands, np.nan)
    records = fit_sessions(
        _without_weak_penalty(model.design),
        _DataSet(sessions, model.response, means),
        model.likelihood,
        model=model.name,
        kind="target",
        response=model.response,
        unit=model.unit,
        table_sha256={},
        fit_by=model.fit_by,
    )
    _, estimates = _session_rows(model, records, {})
    return {
        name: _reduced_estimate(values, records) for name, values in estimates.items()
    }


def _inside_means(
    model: Model, prepared: Mapping[SessionKey, _Prepared], place: int
) -> dict[SessionKey, np.ndarray]:
    """Each session's expected outcome under the inside scenario at
    ``place``."""
    return {
        key: _inside_mean(model.likelihood, p.inside_eta[place], p)
        for key, p in prepared.items()
    }


def _run_task(
    simulation: Simulation,
    prepared: Mapping[SessionKey, _Prepared],
    task: tuple[int, int],
    targets: dict,
) -> tuple[list[dict], list[dict]]:
    """One data set of one scenario, fitted by every model: its ``data_sets``
    and ``sessions`` rows. ``targets`` keeps each (scenario, model)'s unit
    targets of an inside scenario, computed at its first data set."""
    s, i = task
    scenario = simulation.scenarios[s]
    models = simulation.models
    rng = np.random.default_rng(data_seed(simulation.seed, s, i))
    outcomes, means = _draw(models[0], simulation.sessions, prepared, scenario, s, rng)
    data = _DataSet(simulation.sessions, models[0].response, outcomes)
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
            fit_by=model.fit_by,
        )
        if means is not None:
            target = _unit_targets(model, simulation.sessions, means)
        else:
            if (s, model.name) not in targets:
                targets[s, model.name] = _unit_targets(
                    model, simulation.sessions, _inside_means(model, prepared, s)
                )
            target = targets[s, model.name]
        base = {"scenario": scenario.name, "model": model.name, "data_set": i}
        rows, estimates = _session_rows(model, records, base)
        session_rows += rows
        for name, values in estimates.items():
            planted = planted_value(scenario, model, name)
            row = _data_set_row(
                values,
                records,
                planted,
                target[name],
                simulation.n_draws,
                next(seeds),
            )
            data_set_rows.append({**base, "estimand": name, **row})
    return data_set_rows, session_rows


#: A worker process's run, set once by the pool's initializer.
_WORKER: dict = {}


def _init_worker(simulation: Simulation) -> None:
    _WORKER["simulation"] = simulation
    _WORKER["prepared"] = _prepare(
        simulation.models[0], simulation.sessions, simulation.scenarios
    )
    _WORKER["targets"] = {}


def _worker_task(task: tuple[int, int]) -> tuple[list[dict], list[dict]]:
    return _run_task(
        _WORKER["simulation"], _WORKER["prepared"], task, _WORKER["targets"]
    )


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


def run_columns(simulation: Simulation) -> tuple[str, ...]:
    """The columns of the rows a run keeps, each once: every model's design
    columns (:attr:`.terms.Design.columns`) and unit, the columns a scenario
    names for the softmax's noise, and an outside scenario's ``columns``. The
    response is not kept: it is drawn."""
    out = [c for model in simulation.models for c in model.design.columns]
    out += [model.unit for model in simulation.models]
    for scenario in simulation.scenarios:
        out += [
            value
            for value in (scenario.n_trials, scenario.concentration)
            if isinstance(value, str)
        ]
        out += list(getattr(scenario, "columns", ()))
    response = simulation.models[0].response
    return tuple(c for c in dict.fromkeys(out) if c != response)


def _kept_rows(simulation: Simulation) -> dict[SessionKey, pd.DataFrame]:
    """Each session's rows reduced to :func:`run_columns`, in the rows' own
    column order."""
    kept = set(run_columns(simulation))
    out = {}
    for key, rows in simulation.sessions.items():
        absent = sorted(kept - set(rows.columns))
        if absent:
            raise ValueError(f"session {'/'.join(key)}: no columns {absent}")
        out[key] = rows[[c for c in rows.columns if c in kept]]
    return out


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
        run once, with each session's rows reduced to :func:`run_columns`.
        The tables do not depend on it.
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
    simulation = replace(simulation, sessions=_kept_rows(simulation))
    if workers == 1:
        prepared = _prepare(
            simulation.models[0], simulation.sessions, simulation.scenarios
        )
        targets: dict = {}
        parts = [_run_task(simulation, prepared, task, targets) for task in tasks]
    else:
        with pool_context().Pool(workers, _init_worker, (simulation,)) as pool:
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
    "n_data_sets_with_interval",
    "n_data_sets_with_day_p",
    "n_draws",
    "seed",
    "fit_by",
    "planted_unit",
    "bias_unit",
    "coverage_unit",
    "coverage_unit_se",
    "n_units_used_mean",
)


def _share(hits: np.ndarray, name: str) -> dict[str, float]:
    """The share of ``hits`` (0/1, NaN for a data set that has none) and its
    binomial standard error, over the data sets that have one; NaN without
    any."""
    hits = hits[np.isfinite(hits)]
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
    zero. Both are over the data sets with a finite interval only
    (``n_data_sets_with_interval``; none with fewer than two animals with a
    value), and NaN without any. ``reject_day`` is the share whose day-level
    p is below :data:`REJECTION_LEVEL`, over every data set with an
    estimate, a NaN p (fewer than six non-zero day values) counting as no
    rejection; ``n_data_sets_with_day_p`` says how many had a p, so that a
    rate of zero without any is not read as a test that never rejects.
    Each share has its binomial standard error (``*_se``). Under a planted
    value of zero ``excludes_zero`` and ``reject_day`` are false-positive
    rates, otherwise power.

    Against the unit's own target (module docstring), over the same data
    sets: ``planted_unit``, its mean; ``bias_unit``, the mean of each data
    set's estimate less its ``planted_unit``; ``coverage_unit``, the share of
    intervals that hold it. ``fit_by`` is the model's unit, and
    ``n_units_used_mean`` the units in the estimates. At the animal unit no
    data set has a day-level p (``n_data_sets_with_day_p`` is 0).

    Over every session of every data set: ``share_without_value`` and, for
    each code of :data:`CAUSES`, ``without_value_<code>``, the share of
    sessions without a value for that cause; ``share_not_converged``, the
    share of the session fits tried (status not ``not fitted``) that did not
    converge. ``n_sessions`` is the cohort's, ``n_sessions_used_min`` and
    ``n_sessions_used_mean`` the sessions in the estimates; ``n_data_sets``,
    ``n_data_sets_with_estimate``, ``n_data_sets_with_interval`` and
    ``n_data_sets_with_day_p``; ``n_draws`` and ``seed`` the run's.
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
        unit_target = found["planted_unit"].to_numpy(dtype=float)
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
                "n_data_sets_with_interval": int(
                    (np.isfinite(found["low"]) & np.isfinite(found["high"])).sum()
                ),
                "n_data_sets_with_day_p": int(np.isfinite(found["day_p"]).sum()),
                "n_draws": simulation.n_draws,
                "seed": simulation.seed,
                "fit_by": group["fit_by"].iloc[0],
                "planted_unit": (float(np.mean(unit_target)) if n else np.nan),
                "bias_unit": (float(np.mean(estimate - unit_target)) if n else np.nan),
                **_share(found["covered_unit"].to_numpy(), "coverage_unit"),
                "n_units_used_mean": float(group["n_units_used"].mean()),
            }
        )
    return pd.DataFrame(out, columns=list(SUMMARY_COLUMNS))
