"""``prepare_session`` is the one preparation of a session that ``run_session`` fits.

The three steps are replaced by fakes, so no session is loaded: what is checked
is that ``prepare_session`` hands their outputs on unchanged and that
``run_session`` and the PFC check scripts prepare a session through it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc import session
from continuum_swr_replay.analysis.pfc.design import symmetric_lags
from continuum_swr_replay.analysis.pfc.event_sets import (
    DEFAULT_BASIS_HALF_WINDOW_S,
    EVENT_CLASSES,
)
from continuum_swr_replay.analysis.pfc.matched_control import MatchedControlDraw
from continuum_swr_replay.parameters import SAMPLING_FREQUENCY

_REPO = Path(__file__).resolve().parents[3]

#: A session of 2,000 bins at 500 Hz, every bin fitted.
_TIME = np.arange(2000) / SAMPLING_FREQUENCY


def _events(starts: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_time": _TIME[starts],
            "end_time": _TIME[np.asarray(starts, dtype=int) + 20],
        }
    )


def _event_times() -> dict[str, pd.DataFrame]:
    no_swr, with_swr = _events([200, 900]), _events([500, 1400])
    return {
        "non_local_no_SWR": no_swr,
        "non_local_SWR": with_swr,
        "non_local": pd.concat([no_swr, with_swr], ignore_index=True),
        "SWR": _events([495, 1395, 1700]),
        "matched_control": _events([300, 1000, 1100, 1500]),
    }


@pytest.fixture
def fake_steps(monkeypatch):
    """Replace the three preparation steps; record what each was given."""
    calls: dict[str, dict] = {}
    position_info = pd.DataFrame({"head_speed": np.zeros(len(_TIME))}, index=_TIME)
    fit_mask = pd.Series(True, index=_TIME)
    draw = MatchedControlDraw(
        times=_event_times()["matched_control"],
        source_indices=np.arange(4),
        n_onset_outside_fitted_bout=0,
        n_bout_saturated=0,
    )

    def load(nwb_file_name, epoch_name, logger):
        calls["load"] = {"nwb_file_name": nwb_file_name, "epoch_name": epoch_name}
        data = {"position_info": position_info, "theta_phase": pd.Series(0.0)}
        return "results", data, position_info, {"dropped_at": "none"}

    def population(data, nwb_file_name, epoch_name, cohort_stops, stop_cohort, logger):
        calls["population"] = {"stop_cohort": stop_cohort, "cohort_stops": cohort_stops}
        excluded = pd.DataFrame({"start_time": [], "end_time": []})
        return {**data, "blanked": True}, excluded, fit_mask, fit_mask, {"n": 1}

    def events(data, results, nwb_file_name, epoch_name, **kwargs):
        calls["events"] = {"data": data, "results": results, **kwargs}
        return _event_times(), {"n_events": 2}, {"sha": "x"}, _events([]), 7, draw

    monkeypatch.setattr(session, "_load_aligned_session", load)
    monkeypatch.setattr(session, "_session_fit_population", population)
    monkeypatch.setattr(session, "_session_events", events)
    return calls


def test_it_hands_each_steps_output_on(fake_steps):
    prepared = session.prepare_session(
        "s_.nwb", "02_r1", stop_cohort="all_stops", event_set="canonical"
    )

    assert prepared.results == "results"
    # The events are drawn on the data the fit population returned (theta
    # blanked), on the aligned time axis and the fit lags.
    assert fake_steps["events"]["data"]["blanked"]
    assert prepared.data["blanked"]
    np.testing.assert_array_equal(prepared.time, _TIME)
    np.testing.assert_array_equal(fake_steps["events"]["time"], _TIME)
    np.testing.assert_array_equal(
        prepared.fit_lags,
        symmetric_lags(DEFAULT_BASIS_HALF_WINDOW_S, SAMPLING_FREQUENCY),
    )
    np.testing.assert_array_equal(fake_steps["events"]["fit_lags"], prepared.fit_lags)
    assert fake_steps["population"]["stop_cohort"] == "all_stops"
    assert fake_steps["events"]["stop_cohort"] == "all_stops"
    assert fake_steps["events"]["event_set"] == "canonical"
    assert prepared.control_seed == 7
    assert sorted(prepared.event_times) == sorted(_event_times())


class _Stop(Exception):
    pass


def test_run_session_fits_what_prepare_session_returns(monkeypatch):
    seen = {}

    def prepare(nwb_file_name, epoch_name, **kwargs):
        seen.update(kwargs, nwb_file_name=nwb_file_name, epoch_name=epoch_name)
        raise _Stop

    monkeypatch.setattr(session, "prepare_session", prepare)
    with pytest.raises(_Stop):
        session.run_session(
            "s_.nwb", "02_r1", stop_cohort="all_stops", event_set="canonical"
        )
    assert seen["nwb_file_name"] == "s_.nwb"
    assert seen["stop_cohort"] == "all_stops"
    assert seen["event_set"] == "canonical"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, _REPO / "scripts" / "pfc" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_null_geometry_report_reads_the_prepared_session(fake_steps, monkeypatch):
    geometry = _load_script("measure_pfc_null_geometry")
    sidecar = {"stop_cohort": "all_stops", "control_draw_seed": 3, "n_control_draws": 5}

    real, drawn = geometry.session_window_sets("s_.nwb", "02_r1", sidecar, n_draws=2)

    assert fake_steps["load"] == {"nwb_file_name": "s_.nwb", "epoch_name": "02_r1"}
    # The real windows, the shipped control included, are the prepared ones.
    for name, times in _event_times().items():
        pd.testing.assert_frame_equal(real[name], times)
    assert sorted(drawn) == sorted(EVENT_CLASSES)
    assert all(len(draws) == 2 for draws in drawn.values())
