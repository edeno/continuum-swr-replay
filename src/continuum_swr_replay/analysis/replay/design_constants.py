"""Design constants shared by the covariate builders and the model ladders.

Both sides of the pipeline need the same numbers: the builders that write the
persisted columns (how many reward lags, which spline knots, what the basis
columns are called) and the ladders in :mod:`model_specs` that name those
columns in formulas. Keeping the constants here lets a builder read them
without importing the fitted specification, so the persisted table stays
independent of which rungs happen to be fit (enforced by
``tests/unit/test_import_layering.py``).

Constants only: no functions, and no imports from the rest of the package, so
this module can sit at the bottom of the import graph. ``model_specs``
re-exports every name below, so existing ``from ... import model_specs`` call
sites keep working.
"""

from __future__ import annotations

__all__ = [
    "ARM_FE_PREDICTORS",
    "ARRIVAL_CLOCK_COLUMN",
    "ARRIVAL_SPLINE_KNOTS",
    "BIN_S",
    "N_HISTORY_BASIS",
    "N_HISTORY_LAGS",
    "N_REWARD_LAGS",
    "RECENCY_SPLINE_BOUNDS",
    "RECENCY_SPLINE_COLUMNS",
    "RECENCY_SPLINE_KNOTS",
    "REWARD_HISTORY_COLUMNS",
    "REWARD_HISTORY_INCOMPLETE",
    "REWARD_HISTORY_N_BASIS",
    "REWARD_HISTORY_N_LAGS",
    "SPLINE_KNOTS",
    "TMAX",
    "TMAX_ARRIVAL",
]


# ── Arm fixed effects ──
#
# Eight indicator columns ``is_candidate_arm_0`` … ``is_candidate_arm_7`` (``arm_fe_k`` =
# 1 iff ``arm == k``). Arm 8 is the dropped reference category — with
# all nine dummies the set is collinear with the conditional
# likelihood's per-event normalization. These are generated at fit
# time from the canonical ``arm`` column by
# ``_helpers.add_arm_fixed_effects`` (they are not stored in the
# canonical event-arm table).
ARM_FE_PREDICTORS: tuple[str, ...] = tuple(f"is_candidate_arm_{a}" for a in range(8))


# ── Visit-recency spline ──
#
# Natural-cubic-spline basis for visit recency, on the raw trial-count scale,
# masked to the arms where recency is not already determined by
# ``is_current_arm`` / ``is_prev_arm``, with one basis column dropped as
# reference. Why masked, why fixed global knots, and the raw-vs-log1p
# measurement: ``docs/design/model_history.md`` § "Visit-recency spline".
#
# Interior knot positions (in trial counts) and clamp bounds for the natural
# cubic spline on visit recency. The two knots (12, 40 trials) place flexibility
# across the bulk of the recency distribution; the bounds (3, 200) clamp
# extreme/right-censored recencies before the basis is evaluated. Values are
# empirical (chosen for the observed recency distribution), not derived — retune
# them together if the recency scale changes.
RECENCY_SPLINE_KNOTS: tuple[float, float] = (12.0, 40.0)
RECENCY_SPLINE_BOUNDS: tuple[float, float] = (3.0, 200.0)
RECENCY_SPLINE_COLUMNS: tuple[str, ...] = (
    "visit_recency_trials_event_leaf_centered_spline_1",
    "visit_recency_trials_event_leaf_centered_spline_2",
    "visit_recency_trials_event_leaf_centered_spline_3",
)


# ── Reward-history raised-cosine basis ──
#
# The smooth replacement for the ten per-lag dummies: the same known-outcome lag
# vector projected onto log-spaced raised-cosine bumps, each normalized over the
# lags actually known at the query time. Features are weighted reward fractions
# in [0, 1] (coefficients per unit rate, not per SD); feature 0 is the immediate
# component. ``REWARD_HISTORY_INCOMPLETE`` flags stops whose slowest bump has no
# known outcome in its support (fewer than two known outcomes) and whose features
# were therefore zero-filled -- a structural-missingness nuisance main effect, not
# a behavioral quantity to report.
REWARD_HISTORY_N_LAGS = 10
REWARD_HISTORY_N_BASIS = 3
REWARD_HISTORY_COLUMNS = (
    "reward_history_rcos_0",
    "reward_history_rcos_1",
    "reward_history_rcos_2",
)
REWARD_HISTORY_INCOMPLETE = "reward_history_incomplete"

# Span of the reward-history basis: the number of persisted known-outcome lag
# columns the bumps are built over (``reward_basis_from_lags``).
N_REWARD_LAGS = REWARD_HISTORY_N_LAGS


# ── Per-bin conditional-intensity design ──
#
# ALL per-bin rate models share one bin width ``BIN_S`` so the ladder is nested
# and AIC / CV / deviance / GOF are directly comparable (fine bins are required
# anyway for the self-history term and time-rescaling GOF).
BIN_S = 0.015  # 15 ms — ladder-wide (refractory <60 ms, bursty ~100 ms)
TMAX = 4.0  # model only the first 4 s of each stop (see overview Non-Goals)
# Fixed interior knots (seconds) for the within-stop time-course spline, with
# boundary knots pinned to [0, TMAX]. Fixed (not data-driven quantile) knots make
# the spline basis a deterministic function of tc: the fitted curve is
# reconstructable from the saved coefficients (rebuild bs with these knots), the
# spline coefficients are comparable across sessions (same basis everywhere), and
# held-out CV never extrapolates beyond the training knots. Cubic bs over [0, 4]
# with 3 interior knots -> 6 spline columns.
SPLINE_KNOTS = (1.0, 2.0, 3.0)
# Arrival-anchored counterpart of the within-stop clock. ``tc`` starts at the
# bout (the moment the animal became immobile); a stop can begin well after the
# animal reached the well, so a rate that is locked to the ARRIVAL is smeared
# across ``tc`` by the arrival-to-bout offset. The arrival rungs re-anchor the
# same spline block on ``time_since_last_well_arrival_s`` (bin centre minus the
# arrival of the well occupied at bout start) to ask which clock the rate
# follows. Window and knots come from the at-well event distribution: median
# 3.7 s after arrival, IQR 2.7-5.1 s, 87.6 % of events within 8 s, balanced
# across the SWR classes.
TMAX_ARRIVAL = 8.0
ARRIVAL_SPLINE_KNOTS = (1.0, 2.0, 4.0, 6.0)
ARRIVAL_CLOCK_COLUMN = "time_since_last_well_arrival_s"
N_HISTORY_LAGS = 10  # onset_lag1..10 = 150 ms self-history window
N_HISTORY_BASIS = 4  # raised-cosine bumps over the history window (*_histcos models)
