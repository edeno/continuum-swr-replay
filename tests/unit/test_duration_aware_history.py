"""The duration-aware per-bin history design: at-risk time and offset history.

Non-local events are intervals, and one decoder's events never overlap. The
shipped per-bin design counts every bin of a stop as at risk and builds history
from past ONSETS. The opt-in options of ``aggregate_to_timebins``
(``at_risk_events``, ``offset_history``) and the ``*_history_anchors`` of
``PerBinRateModelSpec`` (``MODEL_CROSS_HISTORY_DURATION_RATE``) let a bin inside
an ongoing event drop out and a dependence locked to an event's END be fit as
such. These tests pin the design columns on hand-checked fixtures; the
simulation-recovery tests are in ``test_duration_history_simulation.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay import model_specs
from continuum_swr_replay.analysis.replay.per_bin_design import (
    NO_EXCLUDED_TIME,
    aggregate_to_timebins,
    raised_cosine_history_basis,
)

BIN = 0.1


def _stops(duration_s: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "animal": ["x"],
            "session_date": [1],
            "epoch": ["02_r1"],
            "stop_id": [1],
            "immobility_duration_s": [duration_s],
        }
    )


def _events(starts, ends) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stop_id": 1,
            "t_since_stop_onset": np.asarray(starts, dtype=float),
            "t_end_since_stop_onset": np.asarray(ends, dtype=float),
        }
    )


# Response event [0.15, 0.32) and cross event [0.45, 0.62) in a 1 s stop, 0.1 s
# bins: bins 2 and 5 are wholly inside an event; bins 1, 3, 4 and 6 partly.
RESP = _events([0.15], [0.32])
CROSS = _events([0.45], [0.62])


def _design(**kwargs) -> pd.DataFrame:
    return (
        aggregate_to_timebins(
            _stops(),
            RESP,
            bin_s=BIN,
            tmax=1.0,
            self_history_lags=3,
            cross_onset_times=CROSS,
            cross_history_lags=3,
            excluded_intervals=NO_EXCLUDED_TIME,
            arrival_clocks=False,
            **kwargs,
        )
        .sort_values("binj")
        .set_index("binj")
    )


def test_default_keeps_every_bin_at_full_exposure():
    """Without the options, the shipped design: every bin at risk, no offsets."""
    df = _design()
    assert list(df.index) == list(range(10))
    np.testing.assert_allclose(df["exposure"], BIN)
    assert not any(c.startswith(("offset_lag", "cross_offset_lag")) for c in df)
    assert "n_bins_not_at_risk" not in df.attrs


def test_at_risk_exposure_is_the_time_no_event_covers():
    events = pd.concat([RESP, CROSS], ignore_index=True)
    df = _design(at_risk_events=events)
    # bins 2 and 5 lie inside an event: dropped
    assert list(df.index) == [0, 1, 3, 4, 6, 7, 8, 9]
    expected = {0: 0.1, 1: 0.05, 3: 0.08, 4: 0.05, 6: 0.08, 7: 0.1, 8: 0.1, 9: 0.1}
    np.testing.assert_allclose(df["exposure"], [expected[b] for b in df.index])
    assert df.attrs["n_bins_not_at_risk"] == 2
    assert df.attrs["n_onsets_not_at_risk"] == 0
    # the onset keeps its bin, and the lags count it although bin 2 is gone
    assert df.loc[1, "count"] == 1
    assert df.loc[3, "onset_lag2"] == 1
    assert df.loc[6, "cross_lag2"] == 1


def test_at_risk_coverage_is_clipped_at_the_stop_end():
    """An event running past the stop's end covers only the stop's last bin."""
    events = _events([0.33], [0.9])  # 0.35 s stop: bins 0..3, last bin 0.05 s
    df = aggregate_to_timebins(
        _stops(0.35),
        events,
        bin_s=BIN,
        tmax=1.0,
        excluded_intervals=NO_EXCLUDED_TIME,
        arrival_clocks=False,
        at_risk_events=events,
    ).set_index("binj")
    assert list(df.index) == [0, 1, 2, 3]
    np.testing.assert_allclose(df["exposure"], [0.1, 0.1, 0.1, 0.03])


def test_offset_history_counts_event_ends():
    df = _design(offset_history=True)
    # the response event ends in bin 3, the cross event in bin 6
    assert df["offset_lag1"].to_dict() == {b: float(b == 4) for b in range(10)}
    assert df["offset_lag3"].to_dict() == {b: float(b == 6) for b in range(10)}
    assert df["cross_offset_lag1"].to_dict() == {b: float(b == 7) for b in range(10)}
    # onset lags are untouched: the response onset is in bin 1
    assert df["onset_lag1"].to_dict() == {b: float(b == 2) for b in range(10)}


def test_offset_history_basis_is_the_projected_lags():
    df = aggregate_to_timebins(
        _stops(),
        RESP,
        bin_s=BIN,
        tmax=1.0,
        self_history_lags=3,
        self_history_basis=2,
        cross_onset_times=CROSS,
        cross_history_lags=3,
        cross_history_basis=2,
        excluded_intervals=NO_EXCLUDED_TIME,
        arrival_clocks=False,
        offset_history=True,
    )
    basis = raised_cosine_history_basis(3, 2)
    for lag, rcos in (
        ("offset_lag", "offhist_rcos_"),
        ("cross_offset_lag", "cross_offhist_rcos_"),
    ):
        lags = df[[f"{lag}{j}" for j in (1, 2, 3)]].to_numpy()
        np.testing.assert_allclose(
            df[[f"{rcos}{k}" for k in (0, 1)]].to_numpy(), lags @ basis.T
        )


def test_overlapping_at_risk_events_raise():
    with pytest.raises(ValueError, match="overlap"):
        _design(at_risk_events=_events([0.1, 0.2], [0.3, 0.4]))


def test_offset_history_needs_event_ends_and_self_lags():
    no_end = RESP.drop(columns="t_end_since_stop_onset")
    with pytest.raises(KeyError, match="t_end_since_stop_onset"):
        aggregate_to_timebins(
            _stops(),
            no_end,
            bin_s=BIN,
            tmax=1.0,
            self_history_lags=3,
            excluded_intervals=NO_EXCLUDED_TIME,
            arrival_clocks=False,
            offset_history=True,
        )
    with pytest.raises(ValueError, match="self_history_lags"):
        aggregate_to_timebins(
            _stops(),
            RESP,
            bin_s=BIN,
            tmax=1.0,
            excluded_intervals=NO_EXCLUDED_TIME,
            arrival_clocks=False,
            offset_history=True,
        )


# ── The specs ──


def test_shipped_cross_history_formulas_are_unchanged():
    """The anchors field leaves the shipped rungs' formulas as they were."""
    m_self, m_cross = model_specs.MODEL_CROSS_HISTORY_RATE.values()
    base = (
        "count ~ bs(tc, knots=(1.0, 2.0, 3.0), lower_bound=0.0, upper_bound=4.0)"
        " + reward_history_rcos_0 + reward_history_rcos_1 + reward_history_rcos_2"
        " + reward_history_incomplete"
        " + hist_rcos_0 + hist_rcos_1 + hist_rcos_2 + hist_rcos_3"
    )
    cross = (
        " + cross_hist_rcos_0 + cross_hist_rcos_1 + cross_hist_rcos_2"
        " + cross_hist_rcos_3"
    )
    assert m_self.formula() == base
    assert m_cross.formula() == base + cross


def test_duration_aware_rungs_add_the_offset_blocks():
    ladder = model_specs.MODEL_CROSS_HISTORY_DURATION_RATE
    dur_self, dur_cross = ladder["Mself_dur"], ladder["Mselfcross_dur"]
    m_self, m_cross = model_specs.MODEL_CROSS_HISTORY_RATE.values()
    nb = model_specs.N_HISTORY_BASIS
    on = [f"hist_rcos_{k}" for k in range(nb)]
    off = [f"offhist_rcos_{k}" for k in range(nb)]
    cross_on = [f"cross_hist_rcos_{k}" for k in range(nb)]
    cross_off = [f"cross_offhist_rcos_{k}" for k in range(nb)]
    base = [c for c in m_self.design_columns() if c not in on]
    assert dur_self.design_columns() == m_self.design_columns() + off
    # the anchor-contrast rungs: each block from onsets, ends, or neither
    assert ladder["Mnohist_dur"].design_columns() == base
    assert ladder["Mself_onset_dur"].design_columns() == base + on
    assert ladder["Mself_offset_dur"].design_columns() == base + off
    assert ladder["Mselfcross_onset_dur"].design_columns() == (
        base + on + off + cross_on
    )
    assert ladder["Mselfcross_offset_dur"].design_columns() == (
        base + on + off + cross_off
    )
    assert dur_cross.design_columns() == (
        m_self.design_columns()
        + off
        + [c for c in m_cross.design_columns() if c.startswith("cross_")]
        + cross_off
    )
    # the offset blocks look back as far as the onset blocks
    assert dur_cross.history_lags() == m_cross.history_lags()


def test_offset_only_rung_drops_the_onset_block():
    import dataclasses

    spec = dataclasses.replace(
        model_specs.MODEL_CROSS_HISTORY_RATE["Mself"], self_history_anchors=("offset",)
    )
    cols = spec.design_columns()
    assert "offhist_rcos_0" in cols and "hist_rcos_0" not in cols


@pytest.mark.parametrize("anchors", [(), ("onset", "onset"), ("end",)])
def test_invalid_anchors_raise(anchors):
    import dataclasses

    with pytest.raises(ValueError, match="history_anchors"):
        dataclasses.replace(
            model_specs.MODEL_CROSS_HISTORY_RATE["Mself"], self_history_anchors=anchors
        )


# ── The real-data helpers (analysis/replay/duration_aware_history.py) ──


def test_attach_event_ends_gives_non_local_onsets_their_event_end():
    from continuum_swr_replay.analysis.replay.duration_aware_history import (
        END_COLUMN,
        attach_event_ends,
    )

    keys = {"animal": "x", "session_date": 1, "epoch": "02_r1", "stop_id": 1}
    onsets = pd.DataFrame(
        [
            {
                **keys,
                "event_type": "non_local",
                "onset_time": 10.0,
                "t_since_stop_onset": 0.5,
            },
            {
                **keys,
                "event_type": "MUA",
                "onset_time": 10.2,
                "t_since_stop_onset": 0.7,
            },
        ]
    )
    events = pd.DataFrame([{**keys, "event_start_s": 10.0, "event_end_s": 10.08}])
    out = attach_event_ends(onsets, events)
    assert out.loc[0, END_COLUMN] == pytest.approx(0.58)
    assert np.isnan(out.loc[1, END_COLUMN])
    with pytest.raises(ValueError, match="no event_table event"):
        attach_event_ends(onsets, events.assign(event_start_s=10.002))


def _simulated_design(n_stops: int = 300) -> pd.DataFrame:
    """Scenario (b) at 4x the base rate: the non-SWR stream's at-risk design."""
    import dataclasses

    from continuum_swr_replay.analysis.replay.duration_history_simulation import (
        SCENARIOS,
        simulate_interval_streams,
    )

    sc = SCENARIOS["b_fragmentation"]
    sc = dataclasses.replace(
        sc, base_rate_hz={k: 4 * v for k, v in sc.base_rate_hz.items()}
    )
    stops, events = simulate_interval_streams(sc, n_stops=n_stops, seed=0)
    return aggregate_to_timebins(
        stops,
        events[events["stream"] == "no_swr"],
        bin_s=model_specs.BIN_S,
        tmax=model_specs.TMAX,
        reward_basis=True,
        self_history_lags=model_specs.N_HISTORY_LAGS,
        self_history_basis=model_specs.N_HISTORY_BASIS,
        excluded_intervals=NO_EXCLUDED_TIME,
        arrival_clocks=False,
        outcome_shift=False,
        at_risk_events=events,
        offset_history=True,
    )


def test_pooled_fit_of_one_session_matches_the_per_session_fitter():
    """With one session the fixed effect is the intercept, so the pooled fit is
    the shipped fitter's (coefficients and AIC equal to 1e-6)."""
    from continuum_swr_replay.analysis.replay.duration_aware_history import (
        fit_pooled_history,
    )
    from continuum_swr_replay.analysis.replay.per_bin_rate_model import (
        fit_per_bin_rate_model,
        freeze_per_bin_cohort,
    )

    spec = model_specs.MODEL_CROSS_HISTORY_DURATION_RATE["Mself_offset_dur"]
    design, _ = freeze_per_bin_cohort(_simulated_design(), [spec])
    design["session"] = "only"
    pooled = fit_pooled_history(design, spec)
    single = fit_per_bin_rate_model(design, spec)
    assert pooled["ok"]
    coef = [single[f"coef_offhist_rcos_{k}"] for k in range(4)]
    np.testing.assert_allclose(
        pooled["kernels"]["self_offset"]["coef"], coef, atol=1e-6
    )
    assert pooled["aic"] == pytest.approx(single["aic"], abs=1e-6)


def test_pooled_fit_recovers_the_end_locked_kernel_with_a_session_effect():
    """The stops split into two sessions, each with its own baseline term: the
    end-only kernel still peaks at lag 2-3 (30-45 ms; the truth peaks at
    45 ms), with finite positive SEs."""
    from continuum_swr_replay.analysis.replay.duration_aware_history import (
        fit_pooled_history,
    )
    from continuum_swr_replay.analysis.replay.per_bin_rate_model import (
        freeze_per_bin_cohort,
    )

    spec = model_specs.MODEL_CROSS_HISTORY_DURATION_RATE["Mself_offset_dur"]
    design, _ = freeze_per_bin_cohort(_simulated_design(), [spec])
    design["session"] = np.where(design["stop_id"] % 2 == 0, "a", "b")
    fit = fit_pooled_history(design, spec)
    kernel = np.array(fit["kernels"]["self_offset"]["log_rr"])
    se = np.array(fit["kernels"]["self_offset"]["se"])
    assert fit["ok"] and fit["converged"]
    assert int(np.argmax(kernel)) + 1 in (2, 3)
    assert np.all(np.isfinite(se)) and np.all(se > 0)
