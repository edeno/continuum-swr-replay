"""The role content models in simulation: their scenario list for the engine.

Every number a run of this module writes is a simulation number. The real
response (the event-arm table's ``arm_fraction``) is never fitted: the cohort
rows are kept for their covariates, roles, events and decode bins, and the
engine (``regression.simulation``) draws the shares each scenario expects and
fits the real model (``role_content.role_design``, the grouped softmax,
``regression.sessions.fit_sessions``), reads its estimands
(``role_content.estimand_contrasts``) and reduces them over sessions, days
and animals (``shared_statistics.estimand_summary``). The one place the real
shares are read is the truth from the data (``role_content_scenarios
.truth_from_data``), whose band means are its planted curves; that truth is
then drawn and fitted like every other.

A run (:func:`build_simulation`) is one model, one variant and one kind
(``SWR`` or ``No SWR``): the engine fits each session of one kind, and the
two models have different coefficients. Its models are the same design at
each weak-penalty width (:data:`PRIOR_SDS`), fitted to the same data sets.
The noise of the shares (:data:`NOISES`) is the run's: one categorical
choice per event; a multinomial over the event's decode bins
(``n_nonlocal_bins``); or a Dirichlet-multinomial over them at a stated
concentration ``a``, whose share variance is that of the multinomial times
``(n + a) / (1 + a)`` for an event of ``n`` bins.

The smooths' declared log penalties are :data:`CALIBRATED_LOG_PENALTIES`,
found by :func:`calibrate_smooth_penalties` (``regression.calibration
.log_penalty_for_edf``) on the real rows at the targets
:data:`TARGET_EDF`, with the scales pooled over both kinds.

Every fit reads Q5a at both low points, 20 against 3 trials (``Q5a``, the
declared one) and 20 against 5 (``Q5a_low5``).

The scenarios (:func:`scenario_list`) are those of
``role_content_scenarios.role_content_scenarios`` and
``uncertainty_scenarios.uncertainty_scenarios`` for the model and variant,
then the truth from the data (``from_data``), effects that vary over
animals and sessions (``spread``) and, for the plain model, a current-arm
level that grows with the event's length fitted without the length terms
(``length_current_unadjusted``, no planted value). Thin support is a run on thinned rows
(``thin_events``): :data:`THIN_EVENTS` events per session.

``simulation(quick, workers)`` is the runner's entry
(``scripts/checks/run_simulations.py --model role_content``): the plain model
on the No SWR events at every width with the multinomial noise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.regression import simulation as engine
from continuum_swr_replay.analysis.regression.calibration import log_penalty_for_edf
from continuum_swr_replay.analysis.regression.likelihoods import GroupedSoftmax
from continuum_swr_replay.analysis.regression.sessions import SessionKey

from . import role_content as rc
from . import role_content_scenarios as sc
from . import uncertainty_scenarios as uc

__all__ = [
    "CALIBRATED_LOG_PENALTIES",
    "CENTRAL_NOISE",
    "CENTRAL_PRIOR_SD",
    "CONCENTRATIONS",
    "N_BINS",
    "NOISES",
    "PRIOR_SDS",
    "Q5A_LOW5",
    "SEED",
    "TABLE_COLUMNS",
    "TARGET_EDF",
    "THIN_EVENTS",
    "SimulationCohort",
    "build_simulation",
    "calibrate_smooth_penalties",
    "cohort_from_table",
    "engine_model",
    "load_table",
    "model_settings",
    "scenario_list",
    "simulation",
]

SEED = 20261123
N_BINS = "n_nonlocal_bins"
"""The event-arm table's number of non-local decode bins of an event."""

CONCENTRATIONS: tuple[float, ...] = (300.0, 10.0, 1.0)
"""The Dirichlet-multinomial concentrations, from nearly independent bins to
strongly clustered ones. At the median event of 32 bins the share variance is
that of independent bins times 1.10, 3.8 and 16.5 (32 for one categorical
choice)."""

#: Noise name -> (n_trials, concentration), as the engine's scenarios take them.
NOISES: Mapping[str, tuple[int | str, float | None]] = {
    "categorical": (1, None),
    "multinomial": (N_BINS, None),
    **{f"dirichlet_{a:g}": (N_BINS, a) for a in CONCENTRATIONS},
}
CENTRAL_NOISE = "multinomial"

PRIOR_SDS: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0)
CENTRAL_PRIOR_SD = 8.0

TARGET_EDF: Mapping[str, float] = {
    rc.RECENCY: 5.0,
    rc.REWARD_FRACTION: 3.0,
    rc.LEAVING: 6.0,
}
"""The target median edf of each smooth: recency 5 (a flexible curve, the
belief-variance test's condition); the reward fraction 3 (a curve with a
bend, on a covariate of 11 values); time until leaving 6 (a rise within the
last half second of a 3 s range)."""

CALIBRATED_LOG_PENALTIES: Mapping[str, float] = {
    rc.REWARD_FRACTION: -4.554901123046875,
    rc.RECENCY: -4.409332275390625,
    rc.LEAVING: 1.013336181640625,
}
"""The declared log penalties :func:`calibrate_smooth_penalties` found for
:data:`TARGET_EDF` on the 242 real sessions' No SWR rows (2026-11-23; median
edf 3.000, 5.000 and 6.000 over 242 sessions, each term read on its role:
the reward fraction and time until leaving on the current role, recency on
the other-leaf role). The reward fraction's value is used for its smooth on
every leaf role."""

Q5A_LOW5 = "Q5a_low5"
"""Q5a read at 20 against 5 trials."""

THIN_EVENTS = 30

TABLE_COLUMNS: tuple[str, ...] = (
    "animal",
    "session_date",
    "epoch",
    "event_id",
    "candidate_arm",
    "arm_fraction",
    "is_swr",
    "visit_context",
    "occupancy_well_arm",
    "prev_arm",
    "reward_fraction_candidate_arm_last10_visits",
    "reward_belief_mean_candidate_arm_preoutcome",
    "visit_recency_trials",
    "is_candidate_arm_never_visited",
    "time_until_visit_end_s",
    "event_duration_s",
    "reward_belief_variance_candidate_arm_preoutcome",
    "log1p_n_known_visits_candidate_arm",
    "n_visits_arrived",
    N_BINS,
)
"""The event-arm table's columns a run reads."""


def load_table() -> pd.DataFrame:
    """The real event-arm table's :data:`TABLE_COLUMNS` (read only)."""
    from continuum_swr_replay.paths import table_path
    from continuum_swr_replay.tables.io import read_table

    return read_table(table_path("event_arm_table"), columns=list(TABLE_COLUMNS))


@dataclass(frozen=True)
class SimulationCohort:
    """The cohort rows of a simulation and the truths taken from the data.

    Attributes
    ----------
    rows : pd.DataFrame
        ``role_content.COHORT_COLUMNS`` and :data:`N_BINS`, both kinds, with
        the response removed.
    reference_log_duration : float
        ``L0``.
    data_truths : dict
        Kind -> model -> animal -> ``Truth`` (``truth_from_data`` of that
        kind's rows kept for the model).
    """

    rows: pd.DataFrame
    reference_log_duration: float
    data_truths: Mapping[str, Mapping[str, Mapping[str, sc.Truth]]]

    def model_rows(self, model: str) -> pd.DataFrame:
        """The rows of the events kept for ``model``."""
        return self.rows.loc[self.rows[rc._in_model(model)]].reset_index(drop=True)


def cohort_from_table(
    table: pd.DataFrame, sessions_per_animal: int | None = None
) -> SimulationCohort:
    """The cohort of a simulation from the event-arm table.

    Parameters
    ----------
    table : pd.DataFrame
        The event-arm table (:data:`TABLE_COLUMNS`).
    sessions_per_animal : int or None
        Keep this many sessions of each animal, evenly spaced over its
        sessions in order of day and epoch; None keeps all.
    """
    table = table.assign(
        **{c: table[c].astype(str) for c in rc.SESSION_COLUMNS}
    ).reset_index(drop=True)
    if sessions_per_animal is not None:
        keys = table[list(rc.SESSION_COLUMNS)].drop_duplicates()
        chosen = []
        for _, part in keys.sort_values(list(rc.SESSION_COLUMNS)).groupby("animal"):
            index = np.linspace(0, len(part) - 1, min(sessions_per_animal, len(part)))
            chosen.append(part.iloc[np.unique(np.round(index).astype(int))])
        table = table.merge(pd.concat(chosen), on=list(rc.SESSION_COLUMNS))
    cohort = rc.role_content_cohort(table)
    bins = table.groupby(rc.UNIT)[N_BINS].first()
    rows = cohort.rows.assign(**{N_BINS: cohort.rows[rc.UNIT].map(bins).to_numpy()})
    truths = {}
    for kind in ("SWR", "No SWR"):
        truths[kind] = {}
        for model in rc.MODELS:
            kept = cohort.model_rows(model)
            truths[kind][model] = sc.truth_from_data(kept[kept[rc.KIND] == kind])
    return SimulationCohort(
        rows.drop(columns=[rc.RESPONSE]), cohort.reference_log_duration, truths
    )


def model_settings(
    prior_sd: float, smooth_log_penalties: Mapping[str, float] | None = None
) -> rc.RoleContentSettings:
    """The settings of a run's model at one width; the smooths at their
    calibrated penalties unless given."""
    penalties = dict(smooth_log_penalties or CALIBRATED_LOG_PENALTIES)
    return rc.RoleContentSettings(prior_sd=prior_sd, smooth_log_penalties=penalties)


def engine_model(
    name: str,
    model: str,
    variant: str,
    settings: rc.RoleContentSettings,
    scale_rows: pd.DataFrame,
    reference_log_duration: float,
) -> engine.Model:
    """The role model as its own code builds it, for the engine: its design on
    the scale rows, its estimands and, for the uncertainty model,
    :data:`Q5A_LOW5`."""
    reference = rc.ContrastReference.from_rows(
        scale_rows, reference_log_duration, model
    )
    design = rc.role_design(scale_rows, model, variant, settings, reference=reference)
    estimands = rc.estimand_contrasts(design, model, variant, reference, settings)
    if "Q5a" in estimands:
        low5 = replace(settings, q5a_trials=(settings.q5a_trials[0], 5.0))
        estimands[Q5A_LOW5] = rc.estimand_contrasts(
            design, model, variant, reference, low5
        )["Q5a"]
    return engine.Model(name, design, GroupedSoftmax(), rc.RESPONSE, rc.UNIT, estimands)


def _fixed_shares(
    sessions: Mapping[SessionKey, pd.DataFrame],
    rng: np.random.Generator,
    *,
    shares: Mapping[SessionKey, np.ndarray],
) -> dict[SessionKey, np.ndarray]:
    """The expected shares computed once (they do not change between data
    sets); ``rng`` is not read."""
    return {key: shares[key] for key in sessions}


def _not_computed(
    sessions: Mapping[SessionKey, pd.DataFrame], rng: np.random.Generator, *, name: str
) -> dict:
    """A scenario whose shares the run did not compute."""
    raise ValueError(f"scenario {name!r} was not computed in this run")


def _planted(
    truths: Mapping[SessionKey, sc.Truth],
    reference: sc.ScenarioReference,
    variant: str,
    settings: rc.RoleContentSettings,
) -> dict[str, float]:
    """Each estimand's planted value: of the truth when it is the same in
    every session, else the reduction's target of the sessions' values."""
    low5 = replace(settings, q5a_trials=(settings.q5a_trials[0], 5.0))
    rows = []
    for key, truth in truths.items():
        values = sc.planted_values(truth, reference, variant, settings)
        values[Q5A_LOW5] = sc.planted_values(truth, reference, variant, low5)["Q5a"]
        rows.append(dict(zip(rc.SESSION_COLUMNS, key, strict=True)) | values)
    frame = pd.DataFrame(rows)
    undefined = [
        c for c in frame if c not in rc.SESSION_COLUMNS and frame[c].isna().all()
    ]
    out = sc.planted_target(frame.drop(columns=undefined)).to_dict()
    return out | dict.fromkeys(undefined, np.nan)


def scenario_list(
    cohort: SimulationCohort, model: str, variant: str, kind: str
) -> list[tuple[str, dict[SessionKey, sc.Truth] | sc.Truth]]:
    """(name, truth) of each scenario of a run: one truth, or one per
    session (module docstring)."""
    noise = sc.ShareNoise("categorical")  # not read: the run sets the noise
    listed = [
        s
        for s in [
            *sc.role_content_scenarios(noise=noise),
            *uc.uncertainty_scenarios(noise=noise),
        ]
        if s.model == model and s.variant == variant
    ]
    out: list = [(s.name, s.truth_for(None)) for s in listed]
    if variant != "unadjusted":
        return out
    keys = sorted(rc.session_rows(cohort.model_rows(model)))
    by_animal = cohort.data_truths[kind][model]
    out.append(("from_data", {key: by_animal[key[0]] for key in keys}))
    arms = sc.arm_effects()
    if model == "role_content":
        mean = sc.Truth(
            arm=arms,
            role={"current": 0.5, "previous": -0.3},
            slopes={(rc.VALUE, rc.REFERENCE_ROLE): 0.2},
        )
    else:
        mean = sc.Truth(
            arm=arms,
            curves={(rc.RECENCY, rc.REFERENCE_ROLE): sc.recency_curve(0.6)},
            slopes={
                (rc.BELIEF_VARIANCE, rc.REFERENCE_ROLE): 0.2,
                (rc.SAMPLE_COUNT, rc.REFERENCE_ROLE): 0.0,
            },
        )
    out.append(
        (
            "spread",
            sc.spread_truths(keys, mean, animal_sd=0.2, session_sd=0.2, seed=SEED),
        )
    )
    if model == "role_content":
        # A current-arm level that grows with the event's length, fitted by
        # the unadjusted variant: it has no planted value (the variant
        # averages over lengths), and shows which events an event weight
        # favors.
        out.append(
            (
                "length_current_unadjusted",
                sc.Truth(arm=arms, role={"current": 0.5}, length={"current": 0.6}),
            )
        )
    return out


def build_simulation(
    cohort: SimulationCohort,
    *,
    model: str,
    kind: str,
    variant: str = "unadjusted",
    noise: str = CENTRAL_NOISE,
    prior_sds: Sequence[float] = PRIOR_SDS,
    smooth_log_penalties: Mapping[str, float] | None = None,
    scenarios: Sequence[str] | None = None,
    thin: int | None = None,
    n_data_sets: int = 20,
    n_draws: int = 1000,
    seed: int = SEED,
) -> engine.Simulation:
    """One run (module docstring).

    Parameters
    ----------
    cohort : SimulationCohort
    model, kind, variant : str
    noise : str
        One of :data:`NOISES`.
    prior_sds : sequence of float
        One model per width, named ``sd<width>``.
    smooth_log_penalties : mapping or None
        The smooths' declared log penalties; :data:`CALIBRATED_LOG_PENALTIES`
        when None.
    scenarios : sequence of str or None
        The scenarios whose shares are computed; None computes all. Every
        scenario stays in the list at its place, so that its data sets (whose
        seeds follow the place) are those of a run of every scenario; run the
        others and the engine refuses (pass the same names to
        ``regression.simulation.run``).
    thin : int or None
        Keep this many events of each session (``thin_events``); the design
        stays built on every row.
    n_data_sets, n_draws, seed : int
    """
    n_trials, concentration = NOISES[noise]
    scale = cohort.model_rows(model)
    l0 = cohort.reference_log_duration
    models = [
        engine_model(
            f"sd{sd:g}",
            model,
            variant,
            model_settings(sd, smooth_log_penalties),
            scale,
            l0,
        )
        for sd in prior_sds
    ]
    reference = rc.ContrastReference.from_rows(scale, l0, model)
    sessions = {}
    for i, (key, rows) in enumerate(
        rc.session_rows(scale[scale[rc.KIND] == kind]).items()
    ):
        if thin is not None:
            rows = sc.thin_events(rows, thin, seed + i)
        sessions[key] = rc.design_rows(rows, model, variant, reference)
    truth_reference = sc.ScenarioReference(scale, l0)
    settings = model_settings(prior_sds[0], smooth_log_penalties)
    out = []
    for name, truth in scenario_list(cohort, model, variant, kind):
        if scenarios is not None and name not in scenarios:
            out.append(engine.OutsideScenario(name, partial(_not_computed, name=name)))
            continue
        truths = truth if isinstance(truth, Mapping) else dict.fromkeys(sessions, truth)
        shares = {
            key: sc.expected_shares(rows, truths[key], truth_reference)
            for key, rows in sessions.items()
        }
        planted = _planted(
            {k: truths[k] for k in sessions}, truth_reference, variant, settings
        )
        out.append(
            engine.OutsideScenario(
                name,
                partial(_fixed_shares, shares=shares),
                planted={e: planted[e] for e in models[0].estimands},
                n_trials=n_trials,
                concentration=concentration,
            )
        )
    return engine.Simulation(
        models=models,
        sessions=sessions,
        scenarios=out,
        n_data_sets=n_data_sets,
        n_draws=n_draws,
        seed=seed,
        attrs={
            "content_model": model,
            "variant": variant,
            "kind": kind,
            "noise": noise,
            "thin": thin,
            "smooth_log_penalties": dict(
                smooth_log_penalties or CALIBRATED_LOG_PENALTIES
            ),
            "reference_log_duration": l0,
        },
    )


def calibrate_smooth_penalties(
    cohort: SimulationCohort,
    *,
    kind: str = "No SWR",
    targets: Mapping[str, float] = TARGET_EDF,
    seed: int = SEED,
) -> dict[str, dict[str, float]]:
    """Each smooth's log penalty at its target median edf over the sessions.

    The response is one data set of arm effects only, the multinomial over
    decode bins (the edf reads the fitted shares, not the real ones). The
    recency smooth is calibrated in the uncertainty model, the two others in
    the plain model, every other penalty at :data:`CALIBRATED_LOG_PENALTIES`
    and the weak penalty at :data:`CENTRAL_PRIOR_SD`.

    Returns
    -------
    dict
        Covariate -> ``log_penalty``, ``median_edf``, ``n_sessions_used``.
    """
    out = {}
    for covariate, target in targets.items():
        model = (
            "role_content_uncertainty" if covariate == rc.RECENCY else "role_content"
        )
        run = build_simulation(
            cohort,
            model=model,
            kind=kind,
            prior_sds=(CENTRAL_PRIOR_SD,),
            scenarios=[f"null_{model}"],
            n_data_sets=2,
            seed=seed,
        )
        place = [s.name for s in run.scenarios].index(f"null_{model}")
        data = engine.draw_data_set(
            run.models[0],
            run.sessions,
            run.scenarios[place],
            np.random.default_rng(engine.data_seed(seed, place, 0)),
        )
        role = "current" if covariate != rc.RECENCY else rc.REFERENCE_ROLE
        term = f"s({covariate}):{rc.ROLE}={role}"
        found = log_penalty_for_edf(
            run.models[0].design,
            data,
            GroupedSoftmax(),
            term=term,
            target_edf=target,
            response=rc.RESPONSE,
            unit=rc.UNIT,
        )
        out[covariate] = {
            "term": term,
            "target_edf": target,
            "log_penalty": float(found.log_penalty),
            "median_edf": float(found.median_edf),
            "n_sessions_used": int(found.n_sessions_used),
        }
    return out


#: The quick configuration: sessions per animal, scenarios, data sets, draws.
_QUICK = {
    "sessions_per_animal": 1,
    "scenarios": ("null_role_content", "role_current_large"),
    "n_data_sets": 2,
    "n_draws": 100,
}


def simulation(quick: bool, workers: int) -> engine.Simulation:
    """The runner's entry: the plain model on the No SWR events, every width,
    the multinomial noise; ``quick`` a small cohort and two scenarios.
    ``workers`` is not read (the cohort is built in one process)."""
    del workers
    cohort = cohort_from_table(
        load_table(), _QUICK["sessions_per_animal"] if quick else None
    )
    if not quick:
        return build_simulation(cohort, model="role_content", kind="No SWR")
    run = build_simulation(
        cohort,
        model="role_content",
        kind="No SWR",
        prior_sds=(CENTRAL_PRIOR_SD,),
        scenarios=_QUICK["scenarios"],
        n_data_sets=_QUICK["n_data_sets"],
        n_draws=_QUICK["n_draws"],
    )
    # The quick list holds its two scenarios only (their places change).
    kept = [s for s in run.scenarios if s.name in _QUICK["scenarios"]]
    return replace(run, scenarios=kept)
