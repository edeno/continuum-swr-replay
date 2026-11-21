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

from continuum_swr_replay.analysis.behavior import design_constants
from continuum_swr_replay.analysis.behavior.history_basis import (
    raised_cosine_history_basis,
)
from continuum_swr_replay.analysis.replay_models import model_specs
from continuum_swr_replay.analysis.replay_models.per_bin_design import (
    NO_EXCLUDED_TIME,
    aggregate_to_timebins,
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
    with pytest.raises(ValueError, match="t_end_since_stop_onset"):
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
    nb = design_constants.N_HISTORY_BASIS
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


# ── The real-data helpers (analysis/replay_models/rate/duration_aware_history.py) ──


def test_attach_event_ends_gives_non_local_onsets_their_event_end():
    from continuum_swr_replay.analysis.replay_models.rate.duration_aware_history import (
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

    from continuum_swr_replay.analysis.replay_models.rate.duration_history_simulation import (
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
        bin_s=design_constants.RATE_BIN_S,
        tmax=design_constants.TMAX,
        reward_basis=True,
        self_history_lags=design_constants.N_HISTORY_LAGS,
        self_history_basis=design_constants.N_HISTORY_BASIS,
        excluded_intervals=NO_EXCLUDED_TIME,
        arrival_clocks=False,
        outcome_shift=False,
        at_risk_events=events,
        offset_history=True,
    )


def test_pooled_fit_of_one_session_matches_the_per_session_fitter():
    """With one session the fixed effect is the intercept, so the pooled fit is
    the shipped fitter's (coefficients and AIC equal to 1e-6)."""
    from continuum_swr_replay.analysis.replay_models.rate.duration_aware_history import (
        fit_pooled_history,
    )
    from continuum_swr_replay.analysis.replay_models.rate.per_bin_rate_model import (
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
    from continuum_swr_replay.analysis.replay_models.rate.duration_aware_history import (
        fit_pooled_history,
    )
    from continuum_swr_replay.analysis.replay_models.rate.per_bin_rate_model import (
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


# ── Readouts of a design or a fit, on hand-built inputs ──


def test_history_kernels_weight_the_bumps_by_each_block_coefficients():
    from continuum_swr_replay.analysis.replay_models.rate.duration_aware_history import (
        history_kernels,
    )

    n_lags, n_basis = design_constants.N_HISTORY_LAGS, design_constants.N_HISTORY_BASIS
    basis = raised_cosine_history_basis(n_lags, n_basis)  # (n_basis, n_lags)
    fit = {f"coef_hist_rcos_{k}": float(k == 1) for k in range(n_basis)}
    fit |= {f"coef_offhist_rcos_{k}": 2.0 for k in range(n_basis)}
    fit["coef_cross_hist_rcos_0"] = 1.0  # an incomplete block is not read
    kernels = history_kernels(fit)
    assert set(kernels) == {"self_onset", "self_offset"}
    np.testing.assert_allclose(kernels["self_onset"], basis[1])
    np.testing.assert_allclose(kernels["self_offset"], 2.0 * basis.sum(axis=0))
    assert kernels["self_onset"].shape == (n_lags,)


def _lag_design() -> pd.DataFrame:
    """Four bins, two onsets in 0.06 s; onset-lag columns marked by hand."""
    n_lags = design_constants.N_HISTORY_LAGS
    design = pd.DataFrame(
        {"count": [1, 0, 1, 0], "exposure": [0.015, 0.015, 0.01, 0.02]}
    )
    for j in range(1, n_lags + 1):
        design[f"onset_lag{j}"] = 0
        design[f"offset_lag{j}"] = 0
    design.loc[[1, 3], "onset_lag1"] = 1  # exposure 0.035 has a past onset at lag 1
    design.loc[2, "onset_lag2"] = 2  # a count, not a flag: still one bin's exposure
    design.loc[[0, 1, 2, 3], "offset_lag3"] = 1
    return design


def test_lag_support_is_the_mean_rate_times_the_exposure_at_each_lag():
    from continuum_swr_replay.analysis.replay_models.rate.duration_aware_history import (
        lag_support,
    )

    support = lag_support(_lag_design())
    assert set(support) == {"self_onset", "self_offset"}
    rate = 2 / 0.06
    expected = np.zeros(design_constants.N_HISTORY_LAGS)
    expected[0], expected[1] = rate * 0.035, rate * 0.01
    np.testing.assert_allclose(support["self_onset"], expected)
    assert support["self_offset"][2] == pytest.approx(2.0)  # rate x all exposure
    assert support["self_offset"][[0, 1, 3]].sum() == 0


def test_onset_offset_collinearity_is_one_for_shifted_copies_and_lower_otherwise():
    from continuum_swr_replay.analysis.replay_models.rate.duration_aware_history import (
        onset_offset_collinearity,
    )

    rng = np.random.default_rng(0)
    n_basis = design_constants.N_HISTORY_BASIS
    onset = rng.normal(size=(200, n_basis))
    onset[:20] = 0.0  # bins with no history in either block are left out
    same = pd.DataFrame(
        {
            **{f"hist_rcos_{k}": onset[:, k] for k in range(n_basis)},
            # the offset block spans the same space (a mixing of the onset bumps)
            **{
                f"offhist_rcos_{k}": onset @ rng.normal(size=n_basis)
                for k in range(n_basis)
            },
        }
    )
    assert onset_offset_collinearity(same) == {
        "canonical_corr_self": pytest.approx(1.0)
    }
    other = rng.normal(size=(200, n_basis))
    independent = same.assign(
        **{f"offhist_rcos_{k}": other[:, k] for k in range(n_basis)},
        **{f"cross_hist_rcos_{k}": other[:, k] for k in range(n_basis)},
        **{f"cross_offhist_rcos_{k}": other[:, k] for k in range(n_basis)},
    )
    result = onset_offset_collinearity(independent)
    assert 0 < result["canonical_corr_self"] < 0.5
    assert result["canonical_corr_cross"] == pytest.approx(1.0)
    assert onset_offset_collinearity(same.drop(columns="offhist_rcos_0")) == {}


def _swr_onsets(times, t_since) -> pd.DataFrame:
    keys = {"animal": "x", "session_date": 1, "epoch": "02_r1", "stop_id": 1}
    return pd.DataFrame(
        [
            {**keys, "event_type": "SWR", "onset_time": t, "t_since_stop_onset": s}
            for t, s in zip(times, t_since, strict=True)
        ]
    )


@pytest.fixture
def kay_inventory(monkeypatch):
    """A Kay inventory of three events, 1 ms apart at the middle pair."""
    from continuum_swr_replay.analysis.replay_models.rate import (
        duration_aware_history as dah,
    )

    inventory = pd.DataFrame(
        {"start_time": [19.9, 20.0, 20.003], "end_time": [19.95, 20.05, 20.103]}
    )
    monkeypatch.setattr(dah, "swr_epoch_id_of_session", lambda *keys: "epoch")
    monkeypatch.setattr(dah, "read_swr_inventory", lambda epoch_id: inventory)
    return dah


def _no_non_local_events() -> pd.DataFrame:
    """An event table with no rows, typed as the real one."""
    return pd.DataFrame(
        {
            "animal": pd.Series(dtype=object),
            "session_date": pd.Series(dtype=int),
            "epoch": pd.Series(dtype=object),
            "stop_id": pd.Series(dtype=int),
            "event_start_s": pd.Series(dtype=float),
            "event_end_s": pd.Series(dtype=float),
        }
    )


def test_attach_event_ends_gives_an_swr_onset_its_nearest_kay_event(kay_inventory):
    dah = kay_inventory
    # 20.001 is 1 ms from 20.0 and 2 ms from 20.003: the nearer one's end;
    # 20.0024 is nearer 20.003; 19.8995 matches the first event within 0.5 ms.
    out = dah.attach_event_ends(
        _swr_onsets([20.001, 20.0024, 19.8995], [1.0, 2.0, 3.0]),
        _no_non_local_events(),
    )
    np.testing.assert_allclose(out[dah.END_COLUMN], [1.05, 2.1, 3.05])


def test_attach_event_ends_raises_when_no_kay_event_is_within_tolerance(
    kay_inventory,
):
    dah = kay_inventory
    # 2.6 ms after the last start: beyond SWR_START_TOLERANCE_S (2.5 ms)
    with pytest.raises(ValueError, match="1 SWR onsets have no Kay event"):
        dah.attach_event_ends(
            _swr_onsets([20.0056, 20.0], [1.0, 2.0]), _no_non_local_events()
        )


def test_pooled_fit_records_its_convergence_warnings(monkeypatch):
    import warnings

    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    from continuum_swr_replay.analysis.replay_models.rate.duration_aware_history import (
        fit_pooled_history,
    )
    from continuum_swr_replay.analysis.replay_models.rate.per_bin_rate_model import (
        freeze_per_bin_cohort,
    )

    spec = model_specs.MODEL_CROSS_HISTORY_DURATION_RATE["Mself_offset_dur"]
    design, _ = freeze_per_bin_cohort(_simulated_design(n_stops=100), [spec])
    design["session"] = "only"
    fit = fit_pooled_history(design, spec)
    assert fit["n_convergence_warnings"] == 0 and fit["warnings"] == []

    original = GLM.fit

    def warning_fit(self, *args, **kwargs):
        warnings.warn("IRLS hit the iteration limit", ConvergenceWarning, stacklevel=2)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GLM, "fit", warning_fit)
    fit = fit_pooled_history(design, spec)
    assert fit["ok"]
    assert fit["n_convergence_warnings"] == 1
    assert fit["warnings"] == ["IRLS hit the iteration limit"]


def test_stack_sessions_labels_each_session_and_leaves_out_the_sparse_ones(
    monkeypatch,
):
    """Sessions with 3, 0 and 2 onsets: a gate of 2 keeps the first and last,
    in order, each labeled by its keys; the default keeps all three."""
    from continuum_swr_replay.analysis.replay_models.rate import (
        duration_aware_history as dah,
    )

    counts = {("a", 1, "02_r1"): [1, 2], ("a", 2, "02_r1"): [0], ("a", 3, "04_r2"): [2]}

    def _design(keys, stops_g, onsets_g, outcome, *, duration_aware, ladder):
        return pd.DataFrame({"count": counts[keys]}), ("Mself_dur",)

    monkeypatch.setattr(dah, "build_design", _design)
    sessions = [(keys, None, None) for keys in counts]

    design, rungs, dropped = dah.stack_sessions(sessions, "non_local", min_events=2)

    assert design["session"].tolist() == ["a_1_02_r1", "a_1_02_r1", "a_3_04_r2"]
    assert design["count"].tolist() == [1, 2, 2]
    assert design.index.tolist() == [0, 1, 2]
    assert rungs == ("Mself_dur",)
    assert dropped == ["a_2_02_r1"]

    every, _, none_dropped = dah.stack_sessions(sessions, "non_local")
    assert every["session"].nunique() == 3
    assert none_dropped == []
