"""The shared fitter's numerical rules, in one place.

Every family's fit reads these; change a value only here.
"""

from __future__ import annotations

# Penalized Newton. The fit is at its optimum once the Newton decrement
# ``g' A^-1 g / 2`` (the gain the next step predicts, in log-likelihood units)
# is below NEWTON_DECREMENT_TOL; the value is the content fitter's
# (``analysis/replay_models/fitting.py``). A step is first shortened so that no
# coefficient moves more than MAX_STEP_LENGTH (from a start whose
# probabilities saturate the information is tiny and the raw step enormous),
# then halved at most MAX_STEP_HALVINGS times until the objective does not
# rise: 2**-30 of a step is below the objective's floating-point resolution.
NEWTON_DECREMENT_TOL = 1e-8
NEWTON_MAX_STEPS = 100
MAX_STEP_HALVINGS = 30
MAX_STEP_LENGTH = 10.0

# Bounds of every log penalty strength in the REML search.
LOG_PENALTY_BOUNDS = (-10.0, 20.0)

# A term the other terms explain at R^2 >= COLLINEARITY_R2 (VIF >= 5) is read
# only jointly with its partner.
COLLINEARITY_R2 = 0.8

# Support rule: a declared unpenalized term needs at least SUPPORT_MIN_UNITS
# units on each side in a session, and leaves the per-session model when it
# fails in more than MAX_FAILING_SHARE of the sessions with a fit (the value of
# ``analysis/replay_models/contrast_aggregation.py:MAX_UNIDENTIFIED_SHARE``).
SUPPORT_MIN_UNITS = 5
MAX_FAILING_SHARE = 0.20

# Folds: whole trials balanced by destination well, a stay longer than
# MAX_STAY_PIECE_S cut into equal pieces.
N_FOLDS = 5
MAX_STAY_PIECE_S = 30.0
