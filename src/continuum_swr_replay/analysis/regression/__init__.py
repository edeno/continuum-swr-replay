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
- ``terms``: the model terms, the design they build on the pooled rows, the
  names of terms and coefficients, the standardization and the design check.
- ``collinearity``: how well the other terms explain each term, on the
  information-weighted design (VIF, mgcv's concurvity).
- ``support``: the support rule of a simple term and the guard that a
  tested coefficient carries a penalty.
- ``sessions``: the fitting loop, one fit for each unit of the fit (a
  session, a recording day or an animal, ``fit_by``), with the row rule for
  an inactive level of a category and the handling of a fit without a
  finite maximum.
- ``calibration``: the declared log penalty of a per-session smooth, the one
  at which its median edf over the sessions is a target.
- ``record``: the fit-row record, one per unit of the fit with its
  sessions, and the causes of a fit not kept.
- ``contrasts``: linear functionals of a fit (range-mean curve contrasts and
  coefficient differences) with their exact variance.
- ``folds``: the one fold rule, five folds of whole trials balanced by
  destination well, long stays cut into pieces.
- ``bands``: simultaneous 95 % bands for a curve, one fit's and the
  animal-balanced curve's across sessions.
- ``runtime``: how the fitter's JAX runs: float64 for each call, the device
  guard, the float64 kernel guard and the log of traces. It imports JAX; the
  other modules do not.
- ``padding``: rows and groups padded to bucket sizes, with the weight that
  marks the real rows.
- ``simulation``: the simulation engine: an outcome drawn from a planted
  truth on a model's own rows, fitted and reduced by the real code.

Submodules are imported by name; nothing is re-exported here.
"""
