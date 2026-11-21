"""The shared fitter's numerical rules, in one place.

Every family's fit reads these; change a value only here.
"""

from __future__ import annotations

# Penalized Newton. The fit is at its optimum once the Newton decrement
# ``g' A^-1 g / 2`` (the gain the next step predicts, in log-likelihood units)
# is below NEWTON_DECREMENT_TOL; the value is the content fitter's
# (``analysis/replay_models/fitting.py``). A step is first shortened so that no
# coefficient moves more than MAX_STEP_LENGTH (from a start whose
# probabilities saturate the information is tiny and the raw step enormous;
# a quadratic likelihood, the Gaussian, is not capped: its full step is the
# optimum), then halved at most MAX_STEP_HALVINGS times until the objective does not
# rise: 2**-30 of a step is below the objective's floating-point resolution.
NEWTON_DECREMENT_TOL = 1e-8
NEWTON_MAX_STEPS = 100
MAX_STEP_HALVINGS = 30
MAX_STEP_LENGTH = 10.0

# A decrement below the tolerance is a finite maximum only if its Newton step
# moves the linear predictor by at most NO_MAXIMUM_ETA_STEP on every row
# (``max |X @ step|``, on the link scale, whatever the columns' units). Where
# the likelihood has no finite maximum along a direction (a 0/1 column whose
# rows all have y = 1, a Poisson cell with no events), the decrement goes to
# zero while each step still moves eta by about one. Not applied to a quadratic
# likelihood (the Gaussian), whose maximum is finite when A is positive definite
# and whose eta is on the outcome's scale.
NO_MAXIMUM_ETA_STEP = 0.1

# The design identifies the coefficients the penalty leaves free (unpenalized
# columns, penalties' null spaces) when the Gram matrix of those columns of
# ``X``, each scaled to unit length, has its smallest eigenvalue above
# IDENTIFIABILITY_TOL (1 - R^2 of a column on the others is at least that): an
# exactly collinear design gives rounding, about 1e-16.
IDENTIFIABILITY_TOL = 1e-12

# A penalty matrix is symmetric positive semidefinite to rounding when its
# asymmetry and its most negative eigenvalue are at most PENALTY_ROUNDING_TOL
# times its largest absolute entry and eigenvalue.
PENALTY_ROUNDING_TOL = 1e-10

# Bounds of every log penalty strength in the REML search.
LOG_PENALTY_BOUNDS = (-10.0, 20.0)

# REML penalty selection. The search starts from the best point of a grid over
# the log penalties (the score can have two optima): every integer in the
# bounds for one term; the product of REML_GRID_POINTS for two or three; for
# more, REML_GRID_PASSES passes of one-term integer grids from all terms at
# REML_GRID_CENTER. Newton on the log penalties then floors the Hessian's
# eigenvalues at REML_EIGEN_FLOOR times max(1, its largest |eigenvalue|) and
# stops when its Newton decrement (g' H^-1 g / 2 on the free terms with the
# floored Hessian: the gain in the score, in log-likelihood units, the next step
# predicts) is below REML_DECREMENT_TOL; after REML_MAX_STEPS steps it raises. A term is at the boundary
# when its log penalty is within AT_BOUNDARY_MARGIN of a bound. A term is shrunk
# to its penalty's null space (for a P-spline, a straight line) when its edf is
# within NULL_SPACE_EDF_TOL of the null space's dimension: the search can stop on
# the plateau of the score well before the bound, where at_boundary is false.
REML_GRID_POINTS = (-2.0, 2.0, 6.0, 10.0, 14.0)
REML_GRID_CENTER = 6.0
REML_GRID_PASSES = 2
REML_EIGEN_FLOOR = 1e-6
REML_DECREMENT_TOL = 1e-8
REML_MAX_STEPS = 50
AT_BOUNDARY_MARGIN = 0.5
NULL_SPACE_EDF_TOL = 0.01

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
