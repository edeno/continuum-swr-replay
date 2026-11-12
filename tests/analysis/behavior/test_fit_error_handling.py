"""Expected failures are skipped and logged; coding errors propagate.

Every fitter that turns a failed fit into ``None`` (or a NaN diagnostic) catches
exactly ``_fit_diagnostics.EXPECTED_FIT_ERRORS``. A ``ValueError`` (statsmodels'
"first guess on the deviance function returned a nan", the one the real
in-transit stops raise) is skipped with its traceback logged; a ``TypeError``,
standing in for a coding bug, reaches the caller. The loaders' expected
absences follow the same rule: ``get_spike_data`` treats only
``AreaNotRecordedError`` as "area not recorded".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.discrete.conditional_models as conditional_models

from continuum_swr_replay.analysis.behavior import _fit_diagnostics, fitting
from continuum_swr_replay.analysis.behavior.choice import choice_prediction
from continuum_swr_replay.analysis.behavior.content import model_checking
from continuum_swr_replay.analysis.behavior.rate import (
    per_bin_rate_model,
    per_stop_rate_model,
)
from continuum_swr_replay.data_loaders import position, spikes


def _raiser(exc: Exception):
    def raise_(*args, **kwargs):
        raise exc

    return raise_


def _expected() -> Exception:
    return ValueError("simulated: deviance first guess is nan")


def _bug() -> Exception:
    return TypeError("simulated coding bug")


def _stop_table(n_stops: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "n_selected_events": rng.poisson(2.0, size=n_stops),
            "immobility_duration_s": rng.uniform(5.0, 30.0, size=n_stops),
            "x": rng.integers(0, 2, size=n_stops).astype(float),
        }
    )


def _long_df(n_events: int = 24, n_arms: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for event in range(n_events):
        x = rng.normal(size=n_arms)
        counts = rng.multinomial(5, np.exp(x) / np.exp(x).sum())
        for arm in range(n_arms):
            rows.append(
                {
                    "event_id": event,
                    "candidate_arm": arm,
                    "arm_count_hard": float(counts[arm]),
                    "pred": float(x[arm]),
                }
            )
    return pd.DataFrame(rows)


def _choice_df(n_stops: int = 20, n_arms: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for stop in range(n_stops):
        x = rng.normal(size=n_arms)
        chosen = int(np.argmax(x + rng.normal(size=n_arms)))
        for arm in range(n_arms):
            rows.append(
                {
                    "stop_id": stop,
                    "chosen": int(arm == chosen),
                    "replay_fraction": float(x[arm]),
                }
            )
    return pd.DataFrame(rows)


def _logged_with_traceback(caplog) -> bool:
    return any(r.exc_info is not None for r in caplog.records)


def test_per_bin_fitter_uses_the_shared_tuple():
    assert (
        per_bin_rate_model.EXPECTED_FIT_ERRORS is _fit_diagnostics.EXPECTED_FIT_ERRORS
    )


# ── Per-stop rate GLM and its CV ──


def test_per_stop_fit_skips_expected_failure_and_logs_it(monkeypatch, caplog):
    monkeypatch.setattr(sm, "GLM", _raiser(_expected()))
    with caplog.at_level(logging.WARNING, logger=per_stop_rate_model.logger.name):
        assert per_stop_rate_model.fit_per_stop_rate_model(_stop_table(), ["x"]) is None
    assert _logged_with_traceback(caplog)


def test_per_stop_fit_propagates_coding_error(monkeypatch):
    monkeypatch.setattr(sm, "GLM", _raiser(_bug()))
    with pytest.raises(TypeError, match="simulated coding bug"):
        per_stop_rate_model.fit_per_stop_rate_model(_stop_table(), ["x"])


def test_per_stop_cv_skips_expected_failure_and_logs_it(monkeypatch, caplog):
    monkeypatch.setattr(sm, "GLM", _raiser(_expected()))
    with caplog.at_level(logging.DEBUG, logger=per_stop_rate_model.logger.name):
        assert (
            per_stop_rate_model.cross_validate_per_stop_rate_model(_stop_table(), ["x"])
            is None
        )
    assert _logged_with_traceback(caplog)


def test_per_stop_cv_propagates_coding_error(monkeypatch):
    monkeypatch.setattr(sm, "GLM", _raiser(_bug()))
    with pytest.raises(TypeError, match="simulated coding bug"):
        per_stop_rate_model.cross_validate_per_stop_rate_model(_stop_table(), ["x"])


# ── Conditional logit (content robustness and choice) ──


def test_conditional_logit_skips_expected_failure(monkeypatch, caplog):
    monkeypatch.setattr(fitting, "ConditionalLogit", _raiser(_expected()))
    with caplog.at_level(logging.WARNING, logger=fitting.logger.name):
        assert fitting.fit_conditional_logit(_long_df(), predictors=["pred"]) is None
    assert _logged_with_traceback(caplog)
    result, reason = fitting.conditional_logit_or_reason(
        _long_df(), predictors=["pred"]
    )
    assert result is None
    assert reason == "ValueError: simulated: deviance first guess is nan"


def test_conditional_logit_propagates_coding_error(monkeypatch):
    monkeypatch.setattr(fitting, "ConditionalLogit", _raiser(_bug()))
    with pytest.raises(TypeError, match="simulated coding bug"):
        fitting.fit_conditional_logit(_long_df(), predictors=["pred"])


def test_choice_model_skips_expected_failure(monkeypatch, caplog):
    monkeypatch.setattr(conditional_models, "ConditionalLogit", _raiser(_expected()))
    with caplog.at_level(logging.WARNING, logger=choice_prediction.logger.name):
        assert (
            choice_prediction.fit_choice_model(_choice_df(), ["replay_fraction"])
            is None
        )
    assert _logged_with_traceback(caplog)


def test_choice_model_propagates_coding_error(monkeypatch):
    monkeypatch.setattr(conditional_models, "ConditionalLogit", _raiser(_bug()))
    with pytest.raises(TypeError, match="simulated coding bug"):
        choice_prediction.fit_choice_model(_choice_df(), ["replay_fraction"])


# ── Conditional Poisson CV ──


def test_conditional_poisson_cv_skips_expected_failure(monkeypatch, caplog):
    monkeypatch.setattr(model_checking, "ConditionalPoisson", _raiser(_expected()))
    with caplog.at_level(logging.DEBUG, logger=model_checking.logger.name):
        out = model_checking.cross_validate_conditional_poisson(
            _long_df(), predictors=["pred"], n_folds=3, n_categories=3
        )
    assert out is None
    assert _logged_with_traceback(caplog)


def test_conditional_poisson_cv_propagates_coding_error(monkeypatch):
    monkeypatch.setattr(model_checking, "ConditionalPoisson", _raiser(_bug()))
    with pytest.raises(TypeError, match="simulated coding bug"):
        model_checking.cross_validate_conditional_poisson(
            _long_df(), predictors=["pred"], n_folds=3, n_categories=3
        )


# ── Conditional Poisson diagnostics (condition number, VIF) ──


def test_diagnostics_failure_keeps_the_fit(monkeypatch, caplog):
    monkeypatch.setattr(
        fitting, "_compute_conditional_diagnostics", _raiser(np.linalg.LinAlgError())
    )
    with caplog.at_level(logging.WARNING, logger=fitting.logger.name):
        out = fitting.fit_conditional_poisson(
            _long_df(), predictors=["pred"], n_categories=3
        )
    assert out is not None
    assert "cond_number" not in out
    assert _logged_with_traceback(caplog)


def test_diagnostics_coding_error_propagates(monkeypatch):
    monkeypatch.setattr(fitting, "_compute_conditional_diagnostics", _raiser(_bug()))
    with pytest.raises(TypeError, match="simulated coding bug"):
        fitting.fit_conditional_poisson(_long_df(), predictors=["pred"], n_categories=3)


def test_vif_failure_is_nan(monkeypatch):
    monkeypatch.setattr(
        fitting, "variance_inflation_factor", _raiser(np.linalg.LinAlgError())
    )
    X = np.random.default_rng(0).normal(size=(30, 2))
    out = fitting._compute_conditional_diagnostics(X, ["a", "b"])
    assert np.isnan(out["vif_a"]) and np.isnan(out["vif_b"])
    assert np.isfinite(out["cond_number"])


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_vif_with_a_constant_column_keeps_each_name():
    """A nonzero-constant column must not stop the intercept being added.

    Left unaugmented, every VIF index shifts by one and the last raises
    IndexError, which is not an expected fit error and so discards the fit.
    """
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=30), np.full(30, 2.0), rng.normal(size=30)])
    out = fitting._compute_conditional_diagnostics(X, ["a", "b", "c"])
    assert {"vif_a", "vif_b", "vif_c"} <= out.keys()
    reference = sm.add_constant(X[:, [0, 2]])
    np.testing.assert_allclose(
        [out["vif_a"], out["vif_c"]],
        [
            fitting.variance_inflation_factor(reference, 1),
            fitting.variance_inflation_factor(reference, 2),
        ],
    )


def test_vif_coding_error_propagates(monkeypatch):
    monkeypatch.setattr(fitting, "variance_inflation_factor", _raiser(_bug()))
    X = np.random.default_rng(0).normal(size=(30, 2))
    with pytest.raises(TypeError, match="simulated coding bug"):
        fitting._compute_conditional_diagnostics(X, ["a", "b"])


# ── Spike loader: only AreaNotRecordedError means "not recorded" ──


def test_spike_data_skips_areas_not_recorded(monkeypatch):
    hpc_times = [np.array([0.1, 0.2])]
    hpc_features = [np.ones((2, 4))]
    monkeypatch.setattr(spikes, "get_hpc_marks", lambda nwb: (hpc_times, hpc_features))

    def pfc(nwb, area):
        if area == "OFC":
            raise spikes.AreaNotRecordedError("no OFC sort groups")
        return [np.array([1.0])]

    monkeypatch.setattr(spikes, "get_pfc_spike_times", pfc)
    out = spikes.get_spike_data("x_.nwb")
    assert sorted(out["spike_times"]) == ["HPC", "mPFC"]
    assert sorted(out["spike_waveform_features"]) == ["HPC"]


def test_spike_data_without_hpc(monkeypatch):
    monkeypatch.setattr(
        spikes, "get_hpc_marks", _raiser(spikes.AreaNotRecordedError("no tetrodes"))
    )
    monkeypatch.setattr(
        spikes,
        "get_pfc_spike_times",
        _raiser(spikes.AreaNotRecordedError("no sorting")),
    )
    out = spikes.get_spike_data("x_.nwb")
    assert out == {"spike_times": {}, "spike_waveform_features": {}}


@pytest.mark.parametrize("loader", ["get_hpc_marks", "get_pfc_spike_times"])
def test_spike_data_propagates_other_value_errors(monkeypatch, loader):
    monkeypatch.setattr(spikes, "get_hpc_marks", lambda nwb: ([], []))
    monkeypatch.setattr(spikes, "get_pfc_spike_times", lambda nwb, area: [])
    monkeypatch.setattr(
        spikes, loader, _raiser(ValueError("zip() argument 2 is shorter"))
    )
    with pytest.raises(ValueError, match="zip"):
        spikes.get_spike_data("x_.nwb")


def test_area_not_recorded_is_a_value_error():
    """Callers that caught ``ValueError`` from the loaders still catch it."""
    assert issubclass(spikes.AreaNotRecordedError, ValueError)


# ── Position loader: which linearized track graph ──


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        ([], "senor20201030"),  # not linearized: the session-named graph
        (["spatial_bandit"], "spatial_bandit"),  # one linearization: its graph
        (
            ["6 arm", "senor20201030"],
            "senor20201030",
        ),  # several, session-named among them
    ],
)
def test_track_graph_name(names, expected):
    assert position._track_graph_name("senor20201030_.nwb", names) == expected


def test_track_graph_name_ambiguous_raises():
    with pytest.raises(ValueError, match="several track graphs"):
        position._track_graph_name("senor20201030_.nwb", ["6 arm", "other"])
