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
# (``max |X @ step|``, on the link scale, whatever the columns' units; for the
# grouped softmax, ``X @ step`` less each group's mean, the movement its
# softmax sees). Where
# the likelihood has no finite maximum along a direction (a 0/1 column whose
# rows all have y = 1, a Poisson cell with no events), the decrement goes to
# zero while each step still moves eta by about one. Not applied to a quadratic
# likelihood (the Gaussian), whose maximum is finite when A is positive definite
# and whose eta is on the outcome's scale.
NO_MAXIMUM_ETA_STEP = 0.1
# The columns running away are listed by their part of the step, the largest
# first. The columns of one direction (an arm column with a role column) have
# parts that differ in the last digit only, and which comes out larger depends
# on the machine's linear-algebra kernel. Parts whose base-10 logarithms agree
# to this many decimals (a relative difference below about 2e-9) are a tie, in
# column order.
RUNNING_AWAY_TIE_DECIMALS = 9
# A fit that reaches NEWTON_MAX_STEPS has no finite maximum, too, when each of
# its last NO_MAXIMUM_CAPPED_STEPS steps was cut to MAX_STEP_LENGTH and lowered
# the objective while the full Newton step's movement of eta did not shrink: a
# separation along a direction whose progress the cap keeps linear (a 0/1
# outcome that is a step in x, separated by a smooth's unpenalized line).
NO_MAXIMUM_CAPPED_STEPS = 10

# The design identifies the coefficients the penalty leaves free (unpenalized
# columns, penalties' null spaces) when the Gram matrix of those columns of
# ``X``, each scaled to unit length, has its smallest eigenvalue above
# IDENTIFIABILITY_TOL (1 - R^2 of a column on the others is at least that): an
# exactly collinear design gives rounding, about 1e-16.
IDENTIFIABILITY_TOL = 1e-12
# An error names the coordinates that take part in what the design does not
# identify: those with more than NAMED_SHARE of their unit length in the space
# of the combinations it does not identify.
NAMED_SHARE = 0.1

# A penalty matrix is symmetric positive semidefinite to rounding when its
# asymmetry and its most negative eigenvalue are at most PENALTY_ROUNDING_TOL
# times its largest absolute entry and eigenvalue.
PENALTY_ROUNDING_TOL = 1e-10

# A Gaussian fit reproduces its outcome exactly, and has no scale to profile,
# when its penalized residual sum of squares ``D_p`` is at most
# EXACT_FIT_RESIDUAL**2 times the outcome's sum of squares about its mean
# (``n var(y)``): the outcome fitted to ten significant digits. Rounding leaves
# about 1e-32 of it, exactly zero on some machines' linear-algebra kernels and
# not on others, so the test cannot be ``D_p > 0``.
EXACT_FIT_RESIDUAL = 1e-10

# Bounds of every log penalty strength in the REML search.
LOG_PENALTY_BOUNDS = (-10.0, 20.0)

# The declared log penalty of a per-session smooth (``calibration``) is found by
# bisection inside LOG_PENALTY_BOUNDS until the bracket is narrower than
# CALIBRATION_LOG_PENALTY_TOL: 15 halvings of the bounds' 30 units.
CALIBRATION_LOG_PENALTY_TOL = 1e-3

# The median edf at the returned log penalty must be the target to within
# CALIBRATION_EDF_TOL, or the calibration raises. With a fixed Hessian a
# term's edf is sum_i 1 / (1 + lambda d_i), whose slope in the log penalty is
# at most 1/4 per penalized column; the returned point is within half the
# bracket, 5e-4, of the crossing, so a median over a fixed set of sessions
# is within 1.25e-4 per penalized column of the target: under 0.01 for up to
# 80 columns. A larger gap means the median jumped, a session entering or
# leaving it as the penalty moved.
CALIBRATION_EDF_TOL = 0.01

# REML penalty selection. The search starts from the best point of a grid over
# the log penalties (the score can have two optima): every integer in the
# bounds for one term; the product of REML_GRID_POINTS for two or three; for
# more, REML_GRID_PASSES passes of one-term integer grids from all terms at
# REML_GRID_CENTER. Newton on the log penalties then scales the Hessian by each
# term's sqrt(max(1, |H_jj|)), floors its eigenvalues there at REML_EIGEN_FLOOR
# times max(1, its largest |eigenvalue|), and stops when its Newton decrement
# (g' H^-1 g / 2 on the free terms with the floored Hessian: the gain in the
# score, in log-likelihood units, the next step predicts) is below
# REML_DECREMENT_TOL and the Hessian is positive semidefinite to that floor;
# after REML_MAX_STEPS steps it raises. A term is at the boundary
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

# The most rounds of columns made inactive in the refit of a session set aside
# by the penalty search (the rule is in ``reml.select_penalties``).
REFIT_MAX_ROUNDS = 5

# A penalty search raises when more than MAX_FAILING_SHARE of the sessions end
# without a fit (the value of
# ``analysis/replay_models/contrast_aggregation.py:MAX_UNIDENTIFIED_SHARE``).
MAX_FAILING_SHARE = 0.20
