"""The simulation engine: a planted truth drawn on a model's own rows, fitted
by the real code.

A simulation replaces only the outcome column of a model's rows. The rows, the
design, the fit of each unit (:func:`.sessions.fit_sessions` at the model's
``fit_by``: a session, a recording day or an animal), the estimands
(:func:`.contrasts.evaluate_contrast`) and the reduction over the units with
the project's interval
(:func:`.units.unit_summary`) are the
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
i).spawn(...)`` (:func:`bootstrap_seeds`); each (difference, estimand)'s
follows them, so that a difference changes no model's seed.

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
  run, before the data sets, for an inside scenario and once per data set for
  an outside one (whose expected outcome may read the generator). The Bernoulli takes
  only 0/1 outcomes, and the Gaussian's scale cannot be profiled on a
  noise-free outcome (its residuals are zero when the design holds the mean),
  so under either ``planted_unit`` is NaN. The noise-free target never drops
  a unit for lack of events, and a drawn data set can, so ``bias_unit`` can
  compare slightly different sets of units.

Two models' difference. A model may read some of each session's rows only
(``Model.rows_where``: the rows whose column equals a value, for example one
kind of event when each session's rows hold both kinds). The values form
groups of rows, each drawn on the design of its first model: inside the
model at that design's linear predictor; outside it from the scenario's
``expected``, called once with every row of every session and the data
set's generator. The groups are drawn one after the other, in the models'
order, each session by session, from that one generator. The groups'
outcomes are therefore drawn independently given each group's expected
outcome; a dependence between them exists only when a scenario plants it
through the means (an outside scenario's ``expected`` sees every row and the
generator). That independence is the coordinator's call (2026-11-27), not
the lead's (``docs/design/regression_fitter.md``,
"Simulation"). Each model's fit, support hook and unit target see only its
group's rows, and each keeps its own penalties. A run without
``rows_where`` is one group of every row, drawn as before.

A :class:`Difference` names two models of two groups at one unit, the first
less the second. Its rows in the tables carry its name as their ``model``:

- ``sessions``, one row per unit: the two models' estimates subtracted (NaN
  unless both have one); ``se``, the two standard errors' ``hypot`` (a
  description only: the interval is the bootstrap's); ``without_value``, the
  first's cause, else the second's, so that a unit without a value in either
  model has no difference and is counted with that cause; ``no_fit`` the
  same; ``status``, that of a fit tried that did not converge in either
  model (the first's if both), else ``not fitted`` if either was not
  fitted, else converged;
  ``n_events`` the sum; ``n_sessions``, the unit's sessions with rows in both
  fits; ``edf`` empty;
- ``data_sets``: ``unit_summary`` of the units' differences (the interval is
  the bootstrap of the differences, not a combination of the two models'
  intervals), over the sessions with rows in both fits. Its two targets:
  ``planted``, the first's less the second's (NaN if either is NaN);
  ``planted_unit``, the reduction of the units' differences of the two unit
  targets over the units where both are finite (not the difference of the
  two reductions, which can be over different units);
- ``summary``: as a model's, after the models' rows of each scenario.

A difference is reported for every estimand both models declare, or those it
names.

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

from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np
import pandas as pd

from continuum_swr_replay.runs.threads import pool_context
from continuum_swr_replay.tables.session_keys import FitBy, fit_by_columns

from .contrasts import evaluate_contrast
from .likelihoods import (
    Bernoulli,
    Gaussian,
    GroupedSoftmax,
    Likelihood,
    Poisson,
    draw,
    group_starts,
    mean,
)
from .newton import CONVERGED
from .record import FitRecord, NoFit
from .sessions import NOT_FITTED, SessionKey, fit_sessions
from .support import Removed, require_tested_penalized
from .terms import Design
from .units import unit_estimate, unit_summary, unit_values

__all__ = [
    "CAUSES",
    "SHARE_TOLERANCE",
    "SUMMARY_COLUMNS",
    "Difference",
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
        several sessions; a fit's ``c`` is :meth:`.terms.Design.contrast_on`).
    fit_by : {"session", "day", "animal"}
        The unit of the fit.
    support : callable or None
        The model's own support of its estimands in a unit, beyond the
        fitter's: ``support(record, sessions)``, the unit's
        :class:`.record.FitRecord` and its sessions' rows (those with rows in
        the fit, keyed by session) -> estimand name -> whether it has support
        there; an estimand it does not name has support. A unit without
        support has no value for that estimand, for the cause ``support``
        (``Removed.SUPPORT``), as the model's own statistics report it. It
        reads no outcome, so the unit target leaves out the same units. A
        module-level function (or a ``functools.partial`` of one), so that a
        process pool can send it. None: only the fitter's.
    rows_where : tuple of (str, hashable) or None
        ``(column, value)``: the model's rows are those whose ``column``
        equals ``value``; None: every row. A run whose models set it draws
        each value's rows on its own (module docstring, "Two models'
        difference"); the model's fit, support hook and unit target see only
        its rows.
    """

    name: str
    design: Design
    likelihood: Likelihood
    response: str
    unit: str
    estimands: Mapping[str, np.ndarray]
    fit_by: FitBy = "session"
    support: (
        Callable[[FitRecord, Mapping[SessionKey, pd.DataFrame]], Mapping[str, bool]]
        | None
    ) = None
    rows_where: tuple[str, Hashable] | None = None

    def __post_init__(self) -> None:
        where = self.rows_where
        if where is not None and not (
            isinstance(where, tuple) and len(where) == 2 and isinstance(where[0], str)
        ):
            raise ValueError(
                f"model {self.name!r}: rows_where must be None or (column, value), "
                f"not {where!r}"
            )
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
        fit_by_columns(self.fit_by)
        baseline = [n for c in estimands.values() for n in self.design.baseline_read(c)]
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
class Difference:
    """The difference of two models' unit values, the first's less the
    second's, unit by unit (module docstring, "Two models' difference").

    Attributes
    ----------
    name : str
        Names its rows in the tables (their ``model``); not a model's name.
    first, second : str
        The two models' names: models of the run with ``rows_where`` values
        that differ (their rows are disjoint) and the same ``fit_by``.
    estimands : tuple of str
        The estimands of the difference, each declared by both models; empty:
        every estimand both declare, in the first's order.
    """

    name: str
    first: str
    second: str
    estimands: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "estimands", tuple(self.estimands))


@dataclass(frozen=True)
class Simulation:
    """One run: the models, their rows, the scenarios, the run's size and seed.

    A scenario list beside a model builds one (the runner,
    ``scripts/checks/run_simulations.py``, calls it).

    Attributes
    ----------
    models : sequence of Model
        Fitted to the same data sets: the same likelihood, response and unit,
        and within each group of rows (one ``rows_where`` value) the same
        coefficient names (they may differ in their penalties or estimands).
        An :class:`InsideScenario` is drawn on each group's first model's
        design.
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
    differences : sequence of Difference
        The differences of two models' unit values reported beside the
        models (module docstring, "Two models' difference").
    """

    models: Sequence[Model]
    sessions: Mapping[SessionKey, pd.DataFrame]
    scenarios: Sequence[Scenario]
    n_data_sets: int
    n_draws: int
    seed: int
    attrs: Mapping[str, object] = field(default_factory=dict)
    differences: Sequence[Difference] = ()


class SimulationResult(NamedTuple):
    """The three tables of a run (module docstring)."""

    summary: pd.DataFrame
    data_sets: pd.DataFrame
    sessions: pd.DataFrame


def data_seed(seed: int, scenario: int, data_set: int) -> np.random.SeedSequence:
    """The seed of data set ``data_set`` of the scenario at place ``scenario``."""
    return np.random.SeedSequence([seed, scenario, data_set])


def bootstrap_seeds(seed: int, scenario: int, data_set: int, n: int) -> list[int]:
    """The bootstrap seeds of a data set's ``n`` (model, estimand) pairs, then
    (difference, estimand) pairs, spawned from its :func:`data_seed`."""
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
    value, shape ``(n_rows,)``; all NaN when no row is kept (the fit records
    the session as ``no_rows``)."""
    lik = model.likelihood
    y = np.full(eta.size, np.nan)
    if prepared.missing.all():
        return y
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
    """The expected outcome at ``eta`` (:func:`.likelihoods.mean`), NaN on the
    rows with a missing value, shape ``(n_rows,)``: under the grouped softmax
    each event's shares over its rows that are not missing. All NaN when no
    row is kept."""
    out = np.full(eta.size, np.nan)
    if prepared.missing.all():
        return out
    rows = np.flatnonzero(~prepared.missing)
    if isinstance(lik, GroupedSoftmax):
        rows = rows[np.argsort(prepared.unit[rows], kind="stable")]
        aux = prepared.unit[rows]
    else:
        aux = 0.0 if prepared.offset is None else prepared.offset[rows]
    out[rows] = mean(lik, eta[rows], aux)
    return out


def _expected(
    scenario: OutsideScenario,
    sessions: Mapping[SessionKey, pd.DataFrame],
    rng: np.random.Generator,
) -> dict[SessionKey, np.ndarray]:
    """An outside scenario's expected outcome of each session's rows, shape
    ``(n_rows,)``, read from ``expected`` (with every row of every session)."""
    means = scenario.expected(sessions, rng)
    if set(means) != set(sessions):
        raise ValueError(
            f"scenario {scenario.name!r}: the expected outcome is not given for "
            "exactly the sessions given"
        )
    out = {}
    for key, rows in sessions.items():
        mean = np.asarray(means[key], dtype=float)
        if mean.shape != (len(rows),):
            where = f"scenario {scenario.name!r}, session {'/'.join(key)}"
            raise ValueError(f"{where}: {mean.shape} expected outcomes")
        out[key] = mean
    return out


def _draw(
    model: Model,
    sessions: Mapping[SessionKey, pd.DataFrame],
    prepared: Mapping[SessionKey, _Prepared],
    scenario: Scenario,
    place: int,
    rng: np.random.Generator,
    means: Mapping[SessionKey, np.ndarray] | None,
) -> dict[SessionKey, np.ndarray]:
    """The outcomes of one data set of the scenario at ``place`` (module
    docstring), each session's, shape ``(n_rows,)``: inside the model at its
    prepared linear predictor, outside it at ``means``, the scenario's
    expected outcome of these rows (:func:`_expected`)."""
    if isinstance(scenario, InsideScenario):
        etas = {key: p.inside_eta[place] for key, p in prepared.items()}
    else:
        etas = {}
        for key, p in prepared.items():
            mean = means[key]
            where = f"scenario {scenario.name!r}, session {'/'.join(key)}"
            units = sessions[key][model.unit].to_numpy()[~p.missing]
            _check_expected(model.likelihood, mean[~p.missing], units, where)
            etas[key] = _linear_predictor(model.likelihood, mean, p.offset)
    return {
        key: _draw_session(model, scenario, rows, prepared[key], etas[key], rng)
        for key, rows in sessions.items()
    }


class _Where(Mapping):
    """Each session's rows whose ``column`` equals ``value``
    (``Model.rows_where``), made when the session is read."""

    def __init__(
        self, sessions: Mapping[SessionKey, pd.DataFrame], where: tuple[str, Hashable]
    ) -> None:
        self._sessions, self._where = sessions, where

    def mask(self, key: SessionKey) -> np.ndarray:
        """The session's rows of the value, shape ``(n_rows,)`` bool."""
        column, value = self._where
        return self._sessions[key][column].to_numpy() == value

    def __getitem__(self, key: SessionKey) -> pd.DataFrame:
        return self._sessions[key][self.mask(key)]

    def __iter__(self):
        return iter(self._sessions)

    def __len__(self) -> int:
        return len(self._sessions)


def _groups(
    simulation: Simulation,
) -> dict[tuple[str, Hashable] | None, tuple[Model, Mapping[SessionKey, pd.DataFrame]]]:
    """Each group of rows (a ``rows_where`` value, in the models' order) ->
    its first model, which draws it, and its rows; without ``rows_where``,
    one group of every row."""
    out = {}
    for model in simulation.models:
        where = model.rows_where
        if where not in out:
            rows = (
                simulation.sessions
                if where is None
                else _Where(simulation.sessions, where)
            )
            out[where] = (model, rows)
    return out


def _of_group(
    means: Mapping[SessionKey, np.ndarray], rows: Mapping[SessionKey, pd.DataFrame]
) -> Mapping[SessionKey, np.ndarray]:
    """The expected outcome of a group's rows, from that of every row."""
    if not isinstance(rows, _Where):
        return means
    return {key: mean[rows.mask(key)] for key, mean in means.items()}


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
    ``sessions``, for :func:`draw_data_set`; a model with ``rows_where`` is
    refused (its groups are drawn by :func:`run` only)."""
    _refuse_rows_where(model)
    return PreparedDraws(model, scenario, _prepare(model, sessions, [scenario]))


def _refuse_rows_where(model: Model) -> None:
    if model.rows_where is not None:
        raise ValueError(
            f"model {model.name!r} reads some rows only (rows_where): its groups "
            "are drawn by run(), not draw_data_set()"
        )


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
        If ``prepared`` is of another model, scenario or set of sessions, or
        the model has ``rows_where``.
    """
    _refuse_rows_where(model)
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
    means = (
        None
        if isinstance(scenario, InsideScenario)
        else _expected(scenario, sessions, rng)
    )
    outcomes = _draw(model, sessions, prepared.sessions, scenario, 0, rng, means)
    return dict(_DataSet(sessions, model.response, outcomes))


def _n_sessions(record: FitRecord) -> int:
    """The sessions of a fit with rows in it (a session dropped from a unit
    has none)."""
    return sum(entry["n_rows"] > 0 for entry in record.sessions)


def _with_rows(record: FitRecord) -> list[SessionKey]:
    """The keys of the sessions of a fit with rows in it."""
    return [
        (e["animal"], e["session_date"], e["epoch"])
        for e in record.sessions
        if e["n_rows"] > 0
    ]


def _model_support(
    model: Model, record: FitRecord, sessions: Mapping[SessionKey, pd.DataFrame]
) -> Mapping[str, bool]:
    """The model's own support of its estimands in one unit (``Model.support``);
    every estimand has it without a hook."""
    if model.support is None:
        return {}
    return model.support(record, {key: sessions[key] for key in _with_rows(record)})


def _session_rows(
    model: Model,
    records: Sequence[FitRecord],
    base: dict,
    sessions: Mapping[SessionKey, pd.DataFrame],
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """The ``sessions`` table's rows of one model and data set, one per unit
    of the fit, and each estimand's unit estimates, shape ``(n_units,)``.

    A unit has an estimate when its record has no cause; such a record's fit
    is always converged, as the real statistics require: ``no_fit`` is empty
    only when the refit ended ``CONVERGED`` (``sessions._Session.apply``),
    which ``reml.refit_session`` sets only when its last fit converged, and
    the record's status is that fit's (``sessions._record``)."""
    rows, estimates = [], {name: [] for name in model.estimands}
    for record in records:
        coef, vcov = np.asarray(record.coef), np.asarray(record.vcov)
        supported = _model_support(model, record, sessions)
        for name, c in model.estimands.items():
            left = [record.left_out[n] for n in model.used(c) if n in record.left_out]
            cause = record.no_fit or (left[0] if left else "")
            if not cause and not supported.get(name, True):
                cause = Removed.SUPPORT.value
            estimate, se = np.nan, np.nan
            if not cause:
                contrast = evaluate_contrast(
                    model.design.contrast_on(c, record.coef_names), coef, vcov
                )
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
                    "n_sessions": _n_sessions(record),
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


def _data_set_row(
    values: pd.Series,
    n_sessions: Sequence[int],
    fit_by: FitBy,
    planted: float,
    planted_unit: float,
    n_draws: int,
    seed: int,
) -> dict:
    """One estimand's reduction over a data set's units (module docstring):
    ``values``, one per unit (:func:`.units.unit_values`), and the sessions
    behind each; NaN when no unit has a value. ``covered``, ``covered_unit``
    and ``excludes_zero`` are NaN without a finite interval (none with fewer
    than two animals), so such a data set counts in none of those rates."""
    summary = (
        unit_summary(values, fit_by, n_draws=n_draws, seed=seed)
        if np.isfinite(values).any()
        else dict.fromkeys(("estimate", "low", "high", "sd", "day_p"), np.nan)
        | {"n_units": 0, "n_animals_same_sign": 0}
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
                n
                for n, v in zip(n_sessions, values.to_numpy(), strict=True)
                if np.isfinite(v)
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
) -> dict[str, pd.Series | None]:
    """Each estimand's unit targets (module docstring), one per unit
    (:func:`.units.unit_values`): the model at its ``fit_by`` fitted to the
    expected outcome ``means`` of its rows with the weak penalty off; None
    under the Bernoulli and the Gaussian. :func:`_reduced` reduces them."""
    if isinstance(model.likelihood, (Bernoulli, Gaussian)):
        return dict.fromkeys(model.estimands)
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
    _, estimates = _session_rows(model, records, {}, sessions)
    return {name: unit_values(values, records) for name, values in estimates.items()}


def _reduced(targets: pd.Series | None, fit_by: FitBy) -> float:
    """The unit targets reduced over the units as the estimates are
    (``planted_unit``); NaN for None."""
    return np.nan if targets is None else unit_estimate(targets, fit_by)


#: A group of rows -> each session's :class:`_Prepared` of its rows.
_PreparedGroups = Mapping[tuple[str, Hashable] | None, Mapping[SessionKey, _Prepared]]
#: (scenario place, model name) -> its unit targets (:func:`_unit_targets`).
_Targets = Mapping[tuple[int, str], dict[str, pd.Series | None]]


def _inside_targets(
    simulation: Simulation, prepared: _PreparedGroups, places: Sequence[int]
) -> _Targets:
    """(scenario place, model name) -> its unit targets, for each inside
    scenario at ``places``: its expected outcome does not change between
    data sets, so the target is fitted once per run."""
    groups = _groups(simulation)
    out = {}
    for s in places:
        if not isinstance(simulation.scenarios[s], InsideScenario):
            continue
        for model in simulation.models:
            means = {
                key: _inside_mean(model.likelihood, p.inside_eta[s], p)
                for key, p in prepared[model.rows_where].items()
            }
            rows = groups[model.rows_where][1]
            out[s, model.name] = _unit_targets(model, rows, means)
    return out


def _run_task(
    simulation: Simulation,
    prepared: _PreparedGroups,
    task: tuple[int, int],
    targets: _Targets,
) -> tuple[list[dict], list[dict]]:
    """One data set of one scenario, fitted by every model, and each
    difference: its ``data_sets`` and ``sessions`` rows. ``targets`` holds
    each inside scenario's unit targets (:func:`_inside_targets`); an outside
    scenario's are fitted here, on the data set's expected outcome."""
    s, i = task
    scenario = simulation.scenarios[s]
    models = simulation.models
    rng = np.random.default_rng(data_seed(simulation.seed, s, i))
    means = (
        None
        if isinstance(scenario, InsideScenario)
        else _expected(scenario, simulation.sessions, rng)
    )
    groups = _groups(simulation)
    data, group_means = {}, {}
    for where, (drawer, rows) in groups.items():
        group_means[where] = None if means is None else _of_group(means, rows)
        outcomes = _draw(
            drawer, rows, prepared[where], scenario, s, rng, group_means[where]
        )
        data[where] = _DataSet(rows, drawer.response, outcomes)
    n_pairs = sum(len(model.estimands) for model in models) + sum(
        len(_difference_estimands(d, simulation)) for d in simulation.differences
    )
    seeds = iter(bootstrap_seeds(simulation.seed, s, i, n_pairs))
    data_set_rows, session_rows, fitted = [], [], {}
    for model in models:
        where = model.rows_where
        rows = groups[where][1]
        records = fit_sessions(
            model.design,
            data[where],
            model.likelihood,
            model=model.name,
            kind=scenario.name,
            response=model.response,
            unit=model.unit,
            table_sha256={},
            fit_by=model.fit_by,
        )
        target = (
            targets[s, model.name]
            if means is None
            else _unit_targets(model, rows, group_means[where])
        )
        base = {"scenario": scenario.name, "model": model.name, "data_set": i}
        unit_rows, estimates = _session_rows(model, records, base, rows)
        session_rows += unit_rows
        n_sessions = [_n_sessions(r) for r in records]
        for name, values in estimates.items():
            row = _data_set_row(
                unit_values(values, records),
                n_sessions,
                model.fit_by,
                planted_value(scenario, model, name),
                _reduced(target[name], model.fit_by),
                simulation.n_draws,
                next(seeds),
            )
            data_set_rows.append({**base, "estimand": name, **row})
        fitted[model.name] = (records, unit_rows, target)
    for difference in simulation.differences:
        rows_d, units_d = _difference_rows(
            simulation, difference, scenario, i, fitted, seeds
        )
        data_set_rows += rows_d
        session_rows += units_d
    return data_set_rows, session_rows


def _difference_estimands(
    difference: Difference, simulation: Simulation
) -> tuple[str, ...]:
    """The estimands of a difference: those it names, else every estimand
    both its models declare, in the first's order."""
    if difference.estimands:
        return difference.estimands
    models = {m.name: m for m in simulation.models}
    second = models[difference.second].estimands
    return tuple(e for e in models[difference.first].estimands if e in second)


def _difference_rows(
    simulation: Simulation,
    difference: Difference,
    scenario: Scenario,
    i: int,
    fitted: Mapping[str, tuple[list[FitRecord], list[dict], dict]],
    seeds: Iterator[int],
) -> tuple[list[dict], list[dict]]:
    """A difference's ``data_sets`` and ``sessions`` rows of data set ``i``
    (module docstring, "Two models' difference"), from its two models' unit
    rows and unit targets (``fitted``, by model name); ``seeds`` gives the
    next bootstrap seeds."""
    models = {m.name: m for m in simulation.models}
    first, second = models[difference.first], models[difference.second]
    records_a, units_a, target_a = fitted[first.name]
    records_b, units_b, target_b = fitted[second.name]
    if [r.unit_key for r in records_a] != [r.unit_key for r in records_b]:
        raise ValueError(
            f"difference {difference.name!r}: the units of {first.name!r} and "
            f"{second.name!r} differ"
        )
    n_both = [
        len(set(_with_rows(a)) & set(_with_rows(b)))
        for a, b in zip(records_a, records_b, strict=True)
    ]
    base = {"scenario": scenario.name, "model": difference.name, "data_set": i}
    data_set_rows, session_rows = [], []
    for name in _difference_estimands(difference, simulation):
        units = [
            _difference_unit(base, a, b, n)
            for a, b, n in zip(
                [u for u in units_a if u["estimand"] == name],
                [u for u in units_b if u["estimand"] == name],
                n_both,
                strict=True,
            )
        ]
        session_rows += units
        a, b = target_a[name], target_b[name]
        planted = planted_value(scenario, first, name) - planted_value(
            scenario, second, name
        )
        row = _data_set_row(
            unit_values(np.array([u["estimate"] for u in units]), records_a),
            n_both,
            first.fit_by,
            planted,
            np.nan if a is None or b is None else _reduced(a - b, first.fit_by),
            simulation.n_draws,
            next(seeds),
        )
        data_set_rows.append({**base, "estimand": name, **row})
    return data_set_rows, session_rows


def _difference_status(a: str, b: str) -> str:
    """A difference unit's fit status from its two models': a fit tried that
    did not converge, the first's if both (so that ``share_not_converged``
    counts it); else ``NOT_FITTED`` if either was not fitted; else
    converged."""
    failed = [s for s in (a, b) if s not in (CONVERGED, NOT_FITTED)]
    if failed:
        return failed[0]
    return NOT_FITTED if NOT_FITTED in (a, b) else CONVERGED


def _difference_unit(base: dict, a: dict, b: dict, n_both: int) -> dict:
    """One unit's ``sessions`` row of a difference, from its two models' rows
    of the unit, ``a`` less ``b``; ``n_both``, the unit's sessions with rows
    in both fits."""
    return {
        **base,
        **{k: a[k] for k in ("estimand", "animal", "session_date", "epoch", "fit_by")},
        "n_sessions": n_both,
        "n_events": a["n_events"] + b["n_events"],
        "status": _difference_status(a["status"], b["status"]),
        "no_fit": a["no_fit"] or b["no_fit"],
        "without_value": a["without_value"] or b["without_value"],
        "estimate": a["estimate"] - b["estimate"],
        "se": float(np.hypot(a["se"], b["se"])),
        "edf": {},
    }


#: A worker process's run, set once by the pool's initializer.
_WORKER: dict = {}


def _init_worker(
    simulation: Simulation, prepared: _PreparedGroups, targets: _Targets
) -> None:
    _WORKER.update(simulation=simulation, prepared=prepared, targets=targets)


def _worker_task(task: tuple[int, int]) -> tuple[list[dict], list[dict]]:
    return _run_task(
        _WORKER["simulation"], _WORKER["prepared"], task, _WORKER["targets"]
    )


def _check(simulation: Simulation) -> None:
    """Refuse a run whose models cannot share data sets, whose row filters
    or differences are not well formed, or whose names repeat."""
    models = simulation.models
    if not models or not simulation.scenarios:
        raise ValueError("a simulation needs a model and a scenario")
    first = models[0]
    wheres = [m.rows_where for m in models]
    if any(w is not None for w in wheres):
        if None in wheres:
            raise ValueError("rows_where must be set on every model or on none")
        columns = sorted({w[0] for w in wheres})
        if len(columns) > 1:
            raise ValueError(
                f"the models' rows_where must name one column, not {columns}"
            )
        if columns[0] == first.response:
            raise ValueError(f"rows_where cannot read the response {first.response!r}")
    leads = {}
    for model in models:
        lead = leads.setdefault(model.rows_where, model)
        if (
            model.design.coef_names != lead.design.coef_names
            or model.likelihood.name != first.likelihood.name
            or (model.response, model.unit) != (first.response, first.unit)
        ):
            raise ValueError(
                f"model {model.name!r} cannot share data sets with {lead.name!r}: "
                "the coefficient names (within a group of rows), likelihood, "
                "response and unit must agree"
            )
    for what, names in (
        ("model", [m.name for m in models]),
        ("scenario", [s.name for s in simulation.scenarios]),
        ("difference", [d.name for d in simulation.differences]),
    ):
        if len(set(names)) != len(names):
            raise ValueError(f"{what} names repeat: {names}")
    by_name = {m.name: m for m in models}
    for difference in simulation.differences:
        _check_difference(difference, by_name, simulation)
    if simulation.n_data_sets < 2:
        raise ValueError("a simulation needs at least two data sets per scenario")


def _check_difference(
    difference: Difference, models: Mapping[str, Model], simulation: Simulation
) -> None:
    """Refuse a difference that is not of two models of the run on disjoint
    rows, at one unit, with estimands both declare."""
    where = f"difference {difference.name!r}"
    if difference.name in models:
        raise ValueError(f"{where} has a model's name")
    missing = [n for n in (difference.first, difference.second) if n not in models]
    if missing:
        raise ValueError(f"{where}: {missing} is not a model of the run")
    if difference.first == difference.second:
        raise ValueError(f"{where}: a model less itself")
    a, b = models[difference.first], models[difference.second]
    if a.rows_where is None:
        raise ValueError(f"{where}: its models need rows_where (two groups of rows)")
    if a.rows_where == b.rows_where:
        raise ValueError(
            f"{where}: its two models must read rows of different values of "
            f"{a.rows_where[0]!r}, not both {a.rows_where[1]!r}"
        )
    if a.fit_by != b.fit_by:
        raise ValueError(f"{where}: its models' fit_by differ ({a.fit_by}, {b.fit_by})")
    for model in (a, b):
        absent = [e for e in difference.estimands if e not in model.estimands]
        if absent:
            raise ValueError(f"{where}: {model.name!r} does not declare {absent}")
    if not _difference_estimands(difference, simulation):
        raise ValueError(f"{where}: no estimand both models declare")


def run_columns(simulation: Simulation) -> tuple[str, ...]:
    """The columns of the rows a run keeps, each once: every model's design
    columns (:attr:`.terms.Design.columns`) and unit, the columns a scenario
    names for the softmax's noise, the models' ``rows_where`` column and an
    outside scenario's ``columns``. The response is not kept: it is drawn."""
    out = [c for model in simulation.models for c in model.design.columns]
    out += [model.unit for model in simulation.models]
    out += [m.rows_where[0] for m in simulation.models if m.rows_where is not None]
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
        scenario, a row filter or a difference is not well formed (or a
        session has rows of no model's ``rows_where`` value), a scenario names
        a coefficient the design does not have, an expected outcome is outside
        what the family takes, or a session's first fit fails
        (:func:`.sessions.fit_sessions`).
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
    _check_groups(simulation)
    prepared = {
        where: _prepare(model, rows, simulation.scenarios)
        for where, (model, rows) in _groups(simulation).items()
    }
    targets = _inside_targets(simulation, prepared, sorted({s for s, _ in tasks}))
    if workers == 1:
        parts = [_run_task(simulation, prepared, task, targets) for task in tasks]
    else:
        context = (simulation, prepared, targets)
        with pool_context().Pool(workers, _init_worker, context) as pool:
            parts = pool.map(_worker_task, tasks, chunksize=1)
    data_sets = pd.DataFrame([row for part in parts for row in part[0]])
    sessions = pd.DataFrame([row for part in parts for row in part[1]])
    return SimulationResult(
        summarize(data_sets, sessions, simulation), data_sets, sessions
    )


def _check_groups(simulation: Simulation) -> None:
    """Refuse a session with rows of no model's ``rows_where`` value."""
    wheres = [w for w in _groups(simulation) if w is not None]
    if not wheres:
        return
    column, values = wheres[0][0], [w[1] for w in wheres]
    for key, rows in simulation.sessions.items():
        n = int((~rows[column].isin(values)).sum())
        if n:
            raise ValueError(
                f"session {'/'.join(key)}: {n} rows of {column} match no model's "
                f"rows_where value ({values})"
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
    "share_sessions_unused",
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

    Over every unit of the fit of every data set (a session, a day or an
    animal, at the model's ``fit_by``): ``share_without_value`` and, for each
    code of :data:`CAUSES`, ``without_value_<code>``, the share of units
    without a value for that cause; ``share_not_converged``, the share of the
    unit fits tried (status not ``not fitted``) that did not converge. A
    session dropped from a day or an animal that keeps a value is in none of
    them. Over the sessions: ``n_sessions`` is the cohort's,
    ``n_sessions_used_min`` and ``n_sessions_used_mean`` the sessions in the
    estimates, and ``share_sessions_unused``, ``1 - n_sessions_used_mean /
    n_sessions``, the mean share of the cohort's sessions an estimate leaves
    out; ``n_data_sets``,
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
                "share_sessions_unused": 1.0
                - float(group["n_sessions_used"].mean()) / len(simulation.sessions),
            }
        )
    return pd.DataFrame(out, columns=list(SUMMARY_COLUMNS))
