"""The decoders' discrete-transition settings are explicit, sourced values.

The decoders used to leave these settings to ``non_local_detector``'s defaults,
which changed after April (+32 % non-local events on identical inputs).
``decoding_transitions`` now holds every value: setting P30-B for the NonLocal
detector (decision R10, ``docs/design/2026-10-04-decoder-transition-prior.md``)
and, for the ContFrag classifier, the values C0 constructed. C0 is commit
``de15650``, the decode code just before this pin.

Offline: both decoder scripts import ``bandit_task``, which imports Spyglass at
import time, so they are imported with ``spyglass`` stubbed, as
``test_decoding_save_dir.py`` does. No detector is fitted (``estimate_parameters``
is replaced), nothing reads ``Processed-Data`` and nothing is written.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest
from non_local_detector import (
    ContFragClusterlessClassifier,
    NonLocalClusterlessDetector,
)
from non_local_detector.discrete_state_transitions import (
    DiscreteStationaryCustom,
    DiscreteStationaryDiagonal,
    get_transition_prior,
)

from continuum_swr_replay import decoding_transitions
from continuum_swr_replay.decoding_common import DecodingConfig
from continuum_swr_replay.decoding_transitions import (
    contfrag_transition_settings,
    nonlocal_transition_settings,
)

BIN_S = 0.002
"""The decode's 2 ms bin, written out so a wrong module constant cannot hide."""
LEARNED_ROWS = [0, 2, 3]
"""Local, Non-Local Continuous, Non-Local Fragmented. Row 1, No-Spike, is frozen."""
PINNED_KEYS = {
    "discrete_transition_type",
    "discrete_transition_concentration",
    "discrete_transition_stickiness",
    "discrete_transition_prior_weight",
    "frozen_discrete_transition_rows",
}
"""Every discrete-transition argument the stationary EM fit reads."""

_DECODER_MODULES = {
    "bandit": "scripts.decoding.run_decoding_bandit",
    "contfrag": "scripts.decoding.run_decoding_contfrag",
}
#: The detector class each decoder's ``perform_decoding`` builds for HPC.
_FITTED_CLASSES = {
    "bandit": NonLocalClusterlessDetector,
    "contfrag": ContFragClusterlessClassifier,
}
_SHARED_MODULE = "scripts.decoding._decode_runner"
"""The flow both decoder scripts run, each with its ``RUNNER`` record."""
#: Modules that bind the Spyglass stub when imported under it.
_STUB_BOUND_MODULES = (
    "continuum_swr_replay.data_loaders.bandit_task",
    _SHARED_MODULE,
    *_DECODER_MODULES.values(),
)


@pytest.fixture
def decoders(monkeypatch) -> Iterator[dict[str, types.ModuleType]]:
    """Both decoder scripts, imported offline with ``spyglass`` stubbed.

    The modules imported here that bind the stub are forgotten afterwards, so a
    later import gets the real ones.
    """
    common = types.ModuleType("spyglass.common")
    common.PositionIntervalMap = type("PositionIntervalMap", (), {})
    monkeypatch.setitem(sys.modules, "spyglass", types.ModuleType("spyglass"))
    monkeypatch.setitem(sys.modules, "spyglass.common", common)

    data_loaders = importlib.import_module("continuum_swr_replay.data_loaders")
    fresh = [name for name in _STUB_BOUND_MODULES if name not in sys.modules]
    load_data_was_cached = "load_data" in vars(data_loaders)

    yield {key: importlib.import_module(name) for key, name in _DECODER_MODULES.items()}

    for name in fresh:
        sys.modules.pop(name, None)
        parent, _, child = name.rpartition(".")
        vars(sys.modules[parent]).pop(child, None)
    if not load_data_was_cached:
        vars(data_loaders).pop("load_data", None)


def _capture_fit(monkeypatch, detector_class: type) -> dict[str, Any]:
    """Replace ``estimate_parameters`` so a decode builds its detector but fits nothing."""
    captured: dict[str, Any] = {}

    def estimate_parameters(self, **kwargs: Any) -> str:
        captured["detector"] = self
        captured["kwargs"] = kwargs
        return "results"

    monkeypatch.setattr(detector_class, "estimate_parameters", estimate_parameters)
    return captured


def _perform_decoding(module: types.ModuleType, **kwargs: Any):
    """Run the shared ``perform_decoding`` with a script's ``RUNNER`` on two bins
    and a dummy (track-less) environment."""
    time = np.array([0.0, BIN_S])
    position_info = pd.DataFrame(
        {
            "head_position_x": [1.0, 1.0],
            "head_position_y": [2.0, 2.0],
            "head_speed": [0.0, 0.0],
        },
        index=pd.Index(time, name="time"),
    )
    return sys.modules[_SHARED_MODULE].perform_decoding(
        module.RUNNER,
        logging.getLogger(__name__),
        [np.array([0.001])],
        [np.zeros((1, 4))],
        position_info,
        None,
        None,
        None,
        np.ones(time.size, dtype=bool),
        brain_area="HPC",
        **kwargs,
    )


def _no_batch(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("run_batch was reached; --non_stationary was not refused")


# --- NonLocal: setting P30-B ------------------------------------------------


def test_rows_sum_to_one_and_expected_stays_are_the_dwell_times() -> None:
    matrix = nonlocal_transition_settings()["discrete_transition_type"].values

    assert decoding_transitions.BIN_S == BIN_S
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    expected_stay_s = BIN_S / (1 - np.diag(matrix))
    np.testing.assert_allclose(expected_stay_s, [3.0, 0.1, 0.1, 0.1], rtol=1e-9)


def test_frozen_no_spike_row_exits_by_base_rates() -> None:
    settings = nonlocal_transition_settings()

    assert settings["frozen_discrete_transition_rows"] == (1,)
    np.testing.assert_allclose(
        settings["discrete_transition_type"].values[1],
        [0.0192, 0.98, 0.0004, 0.0004],
        rtol=0,
        atol=1e-12,
    )


def test_learned_rows_prior_mode_is_the_matrix_and_worth_30_visits() -> None:
    """Each learned row's prior mode is its starting row.

    The mode is the M-step's zero-data estimate, the normalized ``alpha - 1``.
    The M-step adds 30 visits x the expected stay in bins as pseudo-counts:
    45,000 for Local (3 s) and 1,500 for each non-local state (100 ms). The
    Dirichlet's effective sample size, ``sum(alpha)``, is 4 more. ``alpha`` comes
    from the library, so this also checks the library reads the values as meant.
    """
    settings = nonlocal_transition_settings()
    concentration = settings["discrete_transition_concentration"]
    stickiness = settings["discrete_transition_stickiness"]
    matrix = settings["discrete_transition_type"].values

    alpha = get_transition_prior(concentration, stickiness, 4)
    np.testing.assert_allclose(
        alpha, concentration + np.diag(stickiness), rtol=1e-15, atol=0
    )
    pseudo_counts = (alpha - 1)[LEARNED_ROWS]
    total = pseudo_counts.sum(axis=1)
    np.testing.assert_allclose(
        pseudo_counts / total[:, np.newaxis], matrix[LEARNED_ROWS], rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(total, [45_000.0, 1_500.0, 1_500.0], rtol=1e-12)
    np.testing.assert_allclose(
        alpha[LEARNED_ROWS].sum(axis=1), [45_004.0, 1_504.0, 1_504.0], rtol=1e-12
    )


def test_dwell_times_and_visits_hold_at_another_sampling_frequency() -> None:
    """At 1 kHz the bins are 1 ms, but the dwell times and prior strength hold.

    The expected stays are still 3 s and 100 ms. The learned rows are still
    worth 30 visits, now 90,000 and 3,000 pseudo-counts (30 x dwell x fs). The
    No-Spike row is still the frozen row: a 100 ms stay, with its exits split
    0.96 / 0.02 / 0.02.
    """
    settings = nonlocal_transition_settings(sampling_frequency=1000.0)
    matrix = settings["discrete_transition_type"].values
    alpha = get_transition_prior(
        settings["discrete_transition_concentration"],
        settings["discrete_transition_stickiness"],
        4,
    )

    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        0.001 / (1 - np.diag(matrix)), [3.0, 0.1, 0.1, 0.1], rtol=1e-9
    )
    assert settings["discrete_transition_concentration"] == 11.0
    np.testing.assert_allclose(
        (alpha - 1)[LEARNED_ROWS].sum(axis=1), [90_000.0, 3_000.0, 3_000.0], rtol=1e-9
    )
    assert settings["frozen_discrete_transition_rows"] == (1,)
    no_spike = matrix[1]
    np.testing.assert_allclose(
        no_spike[[0, 2, 3]] / (1 - no_spike[1]), [0.96, 0.02, 0.02], rtol=1e-9
    )


def test_prior_strength_and_weight_are_pinned_literally() -> None:
    settings = nonlocal_transition_settings()

    assert set(settings) == PINNED_KEYS
    assert type(settings["discrete_transition_type"]) is DiscreteStationaryCustom
    assert settings["discrete_transition_concentration"] == 11.0
    np.testing.assert_allclose(
        settings["discrete_transition_stickiness"],
        [44960.0, 1460.0, 1460.0, 1460.0],
        rtol=1e-9,
    )
    assert settings["discrete_transition_prior_weight"] == 0.0


def test_each_call_returns_fresh_arrays() -> None:
    """A detector that mutates its matrix or stickiness must not change the next one's."""
    first, second = nonlocal_transition_settings(), nonlocal_transition_settings()

    assert (
        first["discrete_transition_type"].values
        is not second["discrete_transition_type"].values
    )
    assert (
        first["discrete_transition_stickiness"]
        is not second["discrete_transition_stickiness"]
    )


def test_hpc_detector_built_by_the_decode_carries_the_pinned_settings(
    decoders, monkeypatch
) -> None:
    captured = _capture_fit(monkeypatch, NonLocalClusterlessDetector)

    _, detector, _ = _perform_decoding(decoders["bandit"])

    assert type(detector) is NonLocalClusterlessDetector
    assert captured["detector"] is detector
    assert captured["kwargs"]["discrete_transition_covariate_data"] is None
    expected = nonlocal_transition_settings()
    assert type(detector.discrete_transition_type) is DiscreteStationaryCustom
    np.testing.assert_array_equal(
        detector.discrete_transition_type.values,
        expected["discrete_transition_type"].values,
    )
    assert detector.discrete_transition_concentration == 11.0
    np.testing.assert_array_equal(
        detector.discrete_transition_stickiness,
        expected["discrete_transition_stickiness"],
    )
    assert detector.discrete_transition_prior_weight == 0.0
    assert detector.frozen_discrete_transition_rows == (1,)
    assert detector.local_position_std == 2.5


@pytest.mark.parametrize("decoder", sorted(_DECODER_MODULES))
def test_non_stationary_flag_is_refused_before_any_work(
    decoders, decoder, monkeypatch, capsys
) -> None:
    """``--non_stationary`` is refused at the command line, naming #50 and #51.

    The library's non-stationary transitions cannot be saved (#50) and add their
    transition prior at every bin (#51). Both decoders refuse the flag.
    """
    module = decoders[decoder]
    monkeypatch.setattr(module, "run_batch", _no_batch)
    monkeypatch.setattr(sys, "argv", [f"{module.__name__}.py", "--non_stationary"])

    with pytest.raises(SystemExit) as refused:
        module.main()

    assert refused.value.code == 2
    message = capsys.readouterr().err
    assert "#50" in message
    assert "#51" in message


@pytest.mark.parametrize("decoder", sorted(_DECODER_MODULES))
def test_perform_decoding_refuses_non_stationary(
    decoders, decoder, monkeypatch
) -> None:
    """A direct caller cannot build a non-stationary detector either."""
    _capture_fit(monkeypatch, _FITTED_CLASSES[decoder])

    with pytest.raises(ValueError) as refused:
        _perform_decoding(decoders[decoder], non_stationary=True)

    assert "#50" in str(refused.value)
    assert "#51" in str(refused.value)


# --- ContFrag: the values C0 constructed ------------------------------------


def test_contfrag_classifier_keeps_the_values_c0_constructed(
    decoders, monkeypatch
) -> None:
    """The pinned values, literally, and equal to C0's classifier.

    C0 passed ``discrete_transition_type=None`` and left every other transition
    argument to ``non_local_detector`` (the version locked in ``uv.lock``). If
    the second half fails after a library upgrade, the library's defaults have
    moved; the decode keeps the pinned values of the first half.
    """
    captured = _capture_fit(monkeypatch, ContFragClusterlessClassifier)

    _, classifier, env = _perform_decoding(decoders["contfrag"])

    assert type(classifier) is ContFragClusterlessClassifier
    assert captured["detector"] is classifier
    assert captured["kwargs"]["discrete_transition_covariate_data"] is None
    assert type(classifier.discrete_transition_type) is DiscreteStationaryDiagonal
    np.testing.assert_array_equal(
        classifier.discrete_transition_type.diagonal_values, [0.98, 0.98]
    )
    assert classifier.discrete_transition_concentration == 1.1
    np.testing.assert_array_equal(classifier.discrete_transition_stickiness, [0.0, 0.0])
    assert classifier.discrete_transition_prior_weight == 0.0
    assert classifier.frozen_discrete_transition_rows is None

    c0 = ContFragClusterlessClassifier(
        environments=env,
        clusterless_algorithm_params=DecodingConfig().to_clusterless_params(),
        discrete_transition_type=None,
    )
    assert type(c0.discrete_transition_type) is type(
        classifier.discrete_transition_type
    )
    np.testing.assert_array_equal(
        c0.discrete_transition_type.diagonal_values,
        classifier.discrete_transition_type.diagonal_values,
    )
    assert (
        c0.discrete_transition_concentration
        == classifier.discrete_transition_concentration
    )
    np.testing.assert_array_equal(
        c0.discrete_transition_stickiness, classifier.discrete_transition_stickiness
    )
    assert (
        c0.discrete_transition_prior_weight
        == classifier.discrete_transition_prior_weight
    )
    assert (
        c0.frozen_discrete_transition_rows == classifier.frozen_discrete_transition_rows
    )


def test_contfrag_settings_are_documented_values() -> None:
    settings = contfrag_transition_settings()

    assert set(settings) == PINNED_KEYS
    stay = settings["discrete_transition_type"].diagonal_values
    np.testing.assert_allclose(BIN_S / (1 - stay), [0.1, 0.1], rtol=1e-9)
