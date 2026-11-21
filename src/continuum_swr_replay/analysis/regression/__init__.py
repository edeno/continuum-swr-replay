"""The shared regression fitter: arrays in, fits out.

It knows nothing of the task. It imports only the foundation,
``analysis.constants`` and ``analysis.shared_statistics``; any ``analysis``
package may import it.

- ``constants``: the fitter's numerical rules (convergence, step control,
  penalty bounds, the penalty search, the share of sessions without a fit).
- ``likelihoods``: Poisson with an offset, Bernoulli, Gaussian with a scale and
  the grouped softmax, each with its derivatives on the linear-predictor scale.
- ``newton``: the penalized Newton fit at fixed penalty strengths.
- ``reml``: penalty strengths by REML with exact derivatives, one per smooth
  term and shared by the sessions (pooled), from a grid start.
- ``pspline``: the cubic B-spline basis on fixed, equally spaced knots, its
  second-order difference penalty and the sum-to-zero constraint.
- ``contrasts``: linear functionals of a fit (range-mean curve contrasts and
  coefficient differences) with their exact variance.
- ``folds``: the one fold rule, five folds of whole trials balanced by
  destination well, long stays cut into pieces.
- ``bands``: simultaneous 95 % bands for a curve, one fit's and the
  animal-balanced curve's across sessions.
- ``declared_tests``: the declared family of tests, each estimand with its
  question, model and contrast, tested for SWR, No SWR and their difference.

Submodules are imported by name; nothing is re-exported here.
"""
