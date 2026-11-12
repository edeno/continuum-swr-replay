"""Prefrontal cortex (mPFC/OFC) neural activity analysis.

Poisson GLMs relate each sorted mPFC/OFC unit's spiking during immobility to
the peri-event kernels of hippocampal events: all SWRs, all non-local (replay)
events, and the replay events split by SWR overlap, plus a duration-matched
non-event control. Each kernel is a B-spline over +/-250 ms of the event onset
(reported over +/-200 ms), fit under a Gaussian prior on the basis
coefficients, and the models are compared by blocked held-out Poisson deviance
read against redraws of the matched control.

Entry points
------------
run_session
    One session end to end: load and align the data, build the fit mask and
    the canonical event sets, draw the matched control, build the design
    matrices, bin every unit's spikes, fit every unit against every design at
    once (``fitter="jax"``, penalised by ``penalty_sigma``), and return the
    per-unit statistics and peri-event curves. With ``scoring_out`` it also
    hands back the fit material the held-out scoring needs. The batch runner
    is ``scripts/pfc/run_pfc_analysis_bandit.py``.
process_neuron
    One unit's statistics and curves from its fits (prefitted by
    ``run_session``, or fit here).
build_session_fit_mask, build_session_event_sets
    The bins a session is fit on (immobility inside the stop cohort) and the
    event classes its designs are keyed on (``event_sets``).
make_matched_control_times, matched_control_seed
    The duration-matched non-event control windows (``matched_control``).
make_design_matrices, fit_immobility_models
    The model ladder's designs (``design``) and the per-unit fits (``fit``).

Held-out scoring (``heldout``)
------------------------------
``assign_blocked_folds``, ``control_draw_null`` and ``empirical_pvalue`` are
the blocked folds, the matched-control null and its plus-one Monte Carlo
p-value; ``ENGAGEMENT_PAIR`` and ``BEST_MODEL_CANDIDATES`` name the engagement
comparison and the best-model candidate sets. ``scoring.score_session``
assembles them into a session's held-out statistics, and
``scoring.maximum_likelihood_comparison`` reads the prior's effect on each
curve's peak against the same curve fit unpenalised.

Population (``load_population``)
--------------------------------
``load_population_data`` reads the per-session bundles back, refuses bundles
from another estimator or inference setting, applies the firing-rate gate, and
returns the unit table and the stacked curves Figure 4 draws;
``significant_fraction`` is the descriptive raw-p fraction; and
``penalty_sigma_suffix`` names a bundle fit at a non-canonical prior.

Population statistics (``population_statistics``, ``timing``)
-------------------------------------------------------------
The summary statistics of the unit table and of the curves' timing: each
function takes tables or arrays and returns named numbers. The mpfc summary
(``reported/mpfc.py``) reports them.

See Also
--------
continuum_swr_replay.analysis.pfc.session : Per-neuron and per-session GLM fits
continuum_swr_replay.analysis.pfc.design : Event bases and design matrices
continuum_swr_replay.analysis.pfc.fit : Fitters, penalties and fit diagnostics
continuum_swr_replay.analysis.pfc.batched_glm : The batched penalised IRLS
continuum_swr_replay.analysis.pfc.heldout : Held-out deviance and its null
continuum_swr_replay.analysis.pfc.scoring : A session's held-out statistics
continuum_swr_replay.analysis.pfc.modulation : Deviance and modulation statistics
continuum_swr_replay.analysis.pfc.event_sets : Fit masks and event sets
continuum_swr_replay.analysis.pfc.matched_control : Matched non-event controls
continuum_swr_replay.analysis.pfc.load_population : Population bundles
continuum_swr_replay.analysis.pfc.population_statistics : Statistics of the unit population
continuum_swr_replay.analysis.pfc.timing : When a response peaks and starts
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Public names are exposed through PEP 562 ``__getattr__`` so that importing a
# leaf submodule (e.g. ``continuum_swr_replay.analysis.pfc.cohort``) does not
# run ``session``, which imports dask and (through ``fit``) statsmodels at
# module top level.

#: Exported name -> (submodule of this package, attribute it is bound to).
_LAZY_EXPORTS = {
    "MatchedControlDraw": ("matched_control", "MatchedControlDraw"),
    "build_session_event_sets": ("event_sets", "build_session_event_sets"),
    "build_session_fit_mask": ("event_sets", "build_session_fit_mask"),
    "fit_immobility_models": ("fit", "_fit_immobility_models"),
    "make_design_matrices": ("design", "_make_design_matrices"),
    "make_matched_control_times": ("matched_control", "make_matched_control_times"),
    "matched_control_seed": ("matched_control", "matched_control_seed"),
    "process_neuron": ("session", "process_neuron"),
    "run_session": ("session", "run_session"),
}


def __getattr__(name: str):
    """Import an exported name from its submodule on first attribute access."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    module = importlib.import_module(f"{__name__}.{module_name}")
    attr = getattr(module, attribute)
    globals()[name] = attr
    return attr


__all__ = sorted(_LAZY_EXPORTS)


if TYPE_CHECKING:  # pragma: no cover - types only
    from .design import _make_design_matrices as make_design_matrices  # noqa: F401
    from .event_sets import (  # noqa: F401
        build_session_event_sets,
        build_session_fit_mask,
    )
    from .fit import _fit_immobility_models as fit_immobility_models  # noqa: F401
    from .matched_control import (  # noqa: F401
        MatchedControlDraw,
        make_matched_control_times,
        matched_control_seed,
    )
    from .session import process_neuron, run_session  # noqa: F401
