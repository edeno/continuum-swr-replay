"""Tests for the per-bin rate-model orchestrator's tidy output.

Runs the orchestrator's in-memory entry point on a tiny synthetic two-session
table (no disk I/O) and checks the tidy schema: one row per
(outcome, model, animal, session_date, epoch), with the coefficient / CV / GOF columns,
and ``outcome`` / ``model`` values drawn from the canonical key sets.
"""

from __future__ import annotations

import functools
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from continuum_swr_replay.analysis.events.analysed_time import stamp_swr_source
from continuum_swr_replay.analysis.replay import (
    MODEL_PER_BIN_RATE,
    POPULATIONS,
    TMAX_ARRIVAL,
)
from tests.unit._executor_fixtures import InlineExecutor, completion_order

_SPEC = importlib.util.spec_from_file_location(
    "fit_per_bin_rate_models",
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "rate"
    / "fit_per_bin_rate_models.py",
)
fprm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fprm)


def _toy_tables(
    n_stops_per_session: int = 24,
    seed: int = 0,
    arrival_lead: float = 0.5,
    n_stops_without_arrival: int = 0,
    n_stops_long_lead: int = 0,
):
    """Two sessions of stop + onset rows with a planted per-stop reward effect.

    Deliberately small (short stops -> few 15 ms bins) so the full ladder + CV
    runs in seconds while still exercising every schema column.

    Every stop carries the absolute clocks the per-bin design derives its two
    clocks from: the animal arrived ``arrival_lead`` seconds before the bout and
    the outcome became observable 0.1 s into it. The first
    ``n_stops_without_arrival`` stops of each session instead have no arrival
    (as a stop that begins before the session's first well arrival does), so
    their bins have no arrival clock and leave the ladder's common cohort, and
    the ``n_stops_long_lead`` after those arrived long enough before the bout
    that the stop runs off the end of the arrival window. The table carries
    no outcome lags, so the fits opt out of the outcome-boundary reward shift
    (:func:`without_outcome_shift`) and the planted per-stop reward effect
    stays exactly what the stop table says it is.
    """
    rng = np.random.default_rng(seed)
    stop_rows, onset_rows = [], []
    sid = 0
    for animal, session_date, epoch in [("a", 1, "02_r1"), ("b", 1, "02_r1")]:
        for stop_in_session in range(n_stops_per_session):
            immob = float(rng.uniform(0.3, 0.7))
            stop_start = 100.0 + 10.0 * stop_in_session
            lead = (
                TMAX_ARRIVAL - 0.1
                if stop_in_session < n_stops_without_arrival + n_stops_long_lead
                else arrival_lead
            )
            arrival = (
                np.nan
                if stop_in_session < n_stops_without_arrival
                else stop_start - lead
            )
            # Every twelfth stop is structurally incomplete: features zero-filled,
            # indicator 1, baseline rate (the persisted stop-table contract).
            incomplete = float(sid % 12 == 0)
            reward = 0.0 if incomplete else float(sid % 2)
            stop_rows.append(
                {
                    "animal": animal,
                    "session_date": session_date,
                    "epoch": epoch,
                    "stop_id": sid,
                    "immobility_duration_s": immob,
                    "reward_history_rcos_0": reward,
                    "reward_history_rcos_1": 0.0
                    if incomplete
                    else float(rng.uniform(0, 1)),
                    "reward_history_rcos_2": 0.0
                    if incomplete
                    else float(rng.uniform(0, 1)),
                    "reward_history_incomplete": incomplete,
                    "did_last_completed_trip_switch_patch": float(rng.integers(0, 2)),
                    "stop_start": stop_start,
                    "last_well_arrival_time_s": arrival,
                    "outcome_known_time_s": stop_start + 0.1,
                }
            )
            # onset times within the stop; rewarded stops get more events.
            rate = 12.0 if reward else 6.0
            n_ev = rng.poisson(rate * min(immob, 4.0))
            for et in ["SWR", "non_local"]:
                for _ in range(n_ev):
                    t = float(rng.uniform(0, min(immob, 4.0)))
                    is_swr = bool(rng.integers(0, 2)) if et == "non_local" else np.nan
                    onset_rows.append(
                        {
                            "animal": animal,
                            "session_date": session_date,
                            "epoch": epoch,
                            "stop_id": sid,
                            "event_type": et,
                            "t_since_stop_onset": t,
                            "onset_time": t,
                            "is_swr": is_swr,
                        }
                    )
            sid += 1
    return pd.DataFrame(stop_rows), pd.DataFrame(onset_rows)


@pytest.fixture
def without_outcome_shift(monkeypatch):
    """Build the fitter's designs with ``outcome_shift=False``.

    The toy stop table plants its reward-history features directly and has no
    outcome lags to re-derive them from, so its designs opt out of the shift by
    name; every other argument is the fitter's own.
    """
    monkeypatch.setattr(
        fprm,
        "aggregate_to_timebins",
        functools.partial(fprm.aggregate_to_timebins, outcome_shift=False),
    )


@pytest.mark.slow
def test_tidy_schema(without_outcome_shift):
    stop, onsets = _toy_tables()
    # onsets already carry is_swr (as attach_is_swr_to_onsets would set it).
    # Pin BLAS: tiny-matrix GLM fits are faster single-threaded.
    with threadpool_limits(limits=1):
        out = fprm.fit_per_bin_rate_models(stop, onsets, run_cv=True, run_gof=True)

    assert len(out) > 0
    expected_cols = {
        "outcome",
        "model",
        "animal",
        "session_date",
        "epoch",
        "rr_reward_history_rcos_0",
        "n_bins",
        "n_events",
        "aic",
        "deviance",
        "dispersion",
        "ks_statistic",
        "ks_within_band",
        "cv_ll_per_bin_mean",
        "cv_bits_per_event",
        "cv_n_folds",
    }
    assert expected_cols <= set(out.columns)

    # All four populations must produce rows (not just a subset) so a silently
    # dropped outcome can't make the schema check vacuous.
    assert set(out["outcome"]) == set(POPULATIONS)
    assert set(out["model"]) <= set(MODEL_PER_BIN_RATE)
    # The core (always-identifiable) models must be present for every outcome.
    for outcome in POPULATIONS:
        sub_models = set(out[out.outcome == outcome]["model"])
        assert {"M0_const", "M1_rew", "M5_time"} <= sub_models

    # The planted immediate-reward effect (12 vs 6 Hz) is recovered by the
    # reward-history rung as a rate ratio near 2 on the immediate component.
    m1 = out[(out.model == "M1_rew") & (out.outcome == "non_local")]
    rr = m1["rr_reward_history_rcos_0"].dropna()
    assert len(rr) == 2  # both sessions
    assert rr.between(1.5, 2.7).all()

    # 'predicted' (the per-bin Series) must not leak into the tidy table.
    assert "predicted" not in out.columns

    # One row per (outcome, model, session) — no duplicates.
    keys = ["outcome", "model", "animal", "session_date", "epoch"]
    assert not out.duplicated(keys).any()


def test_baseline_present_across_outcomes(without_outcome_shift):
    stop, onsets = _toy_tables()
    with threadpool_limits(limits=1):
        out = fprm.fit_per_bin_rate_models(stop, onsets, run_cv=False, run_gof=False)
    # M0_const must fit for every outcome that produced any rows.
    for outcome in out["outcome"].unique():
        sub = out[out.outcome == outcome]
        assert "M0_const" in set(sub["model"])


def test_no_stop_dropped_by_freeze(without_outcome_shift):
    """Freezing the shipped ladder keeps every stop, incomplete prefix included.

    The first stops of an epoch have fewer than two known outcomes, so their
    reward-history features are zero-filled and ``reward_history_incomplete``
    is 1 — zero-filled, never NaN. The cohort freeze must therefore remove no
    stop at all, and every rung must report the same bins and stops.
    """
    stop, onsets = _toy_tables()
    # An unbroken prefix of each epoch is structurally incomplete.
    prefix = stop.groupby(["animal", "session_date", "epoch"], sort=False).head(3)
    stop["reward_history_incomplete"] = 0.0
    stop.loc[prefix.index, "reward_history_incomplete"] = 1.0
    for col in (
        "reward_history_rcos_0",
        "reward_history_rcos_1",
        "reward_history_rcos_2",
    ):
        stop.loc[prefix.index, col] = 0.0

    with threadpool_limits(limits=1):
        out = fprm.fit_per_bin_rate_models(stop, onsets, run_cv=False, run_gof=False)

    assert out.attrs["cohort_stop_drops"] == dict.fromkeys(POPULATIONS, 0)

    n_stops_by_session = (
        stop[stop.immobility_duration_s > 0]
        .groupby(["animal", "session_date", "epoch"])["stop_id"]
        .nunique()
    )
    for keys, cell in out.groupby(
        ["outcome", "animal", "session_date", "epoch", "time_clock"]
    ):
        # One cohort per (session, outcome, clock): every rung on the same rows.
        assert cell["n_bins"].nunique() == 1, keys
        assert cell["n_stops"].nunique() == 1, keys
        # ... and that cohort is every stop of the session, prefix included
        # (every stop of this fixture arrives inside the arrival window, so the
        # window carves nothing out of it).
        assert cell["n_stops"].iloc[0] == n_stops_by_session.loc[keys[1:-1]]

    # The cohort audit attributes nothing to either drop bucket.
    records = fprm._cohort_records(out, stop, out.attrs["cohort_stop_drops"])
    assert {r["n_drop_own_predictor_missing"] for r in records} == {0}
    assert {r["n_drop_common_cohort_other"] for r in records} == {0}


ARRIVAL_MODELS = {
    name for name, spec in MODEL_PER_BIN_RATE.items() if spec.time_clock == "arrival"
}


def test_arrival_rungs_carry_their_clock_and_window(without_outcome_shift):
    """Arrival rungs are fit on the arrival window and say so in every row.

    Stops with no arrival leave the common cohort of EVERY rung (the freeze is
    the ladder's, not the rung's) and the run reports how many; stops that
    started long after the arrival then leave the arrival rungs' window only.
    """
    stop, onsets = _toy_tables(n_stops_without_arrival=2, n_stops_long_lead=6)
    with threadpool_limits(limits=1):
        out = fprm.fit_per_bin_rate_models(stop, onsets, run_cv=False, run_gof=False)

    assert set(out.loc[out.model.isin(ARRIVAL_MODELS), "time_clock"]) == {"arrival"}
    assert set(out.loc[~out.model.isin(ARRIVAL_MODELS), "time_clock"]) == {"stop"}

    # Two arrival-less stops per session x two sessions, gone from every rung.
    assert out.attrs["cohort_stop_drops"] == dict.fromkeys(POPULATIONS, 4)
    for outcome, counts in out.attrs["arrival_clock_cohort"].items():
        assert counts["n_stops_dropped_arrival_clock"] == 4, outcome
        assert counts["n_bins_dropped_arrival_clock"] > 0, outcome
        assert counts["n_bins_arrival_window"] < counts["n_bins_frozen"], outcome

    window_cols = [
        "n_bins_arrival_window",
        "n_stops_arrival_window",
        "retained_share_arrival_bins",
        "retained_share_arrival_events",
    ]
    arrival_rows = out[out.model.isin(ARRIVAL_MODELS)]
    stop_rows = out[~out.model.isin(ARRIVAL_MODELS)]
    assert arrival_rows[window_cols].notna().all().all()
    assert stop_rows[window_cols].isna().all().all()
    assert (arrival_rows["n_bins"] == arrival_rows["n_bins_arrival_window"]).all()
    assert arrival_rows["retained_share_arrival_bins"].between(0.0, 1.0).all()
    assert (arrival_rows["retained_share_arrival_bins"] < 1.0).any()

    for keys, cell in out.groupby(
        ["outcome", "animal", "session_date", "epoch", "time_clock"]
    ):
        assert cell["n_bins"].nunique() == 1, keys
        assert cell["n_stops"].nunique() == 1, keys
    # The window is carved out of the frozen cohort, never added to it.
    for keys, cell in out.groupby(["outcome", "animal", "session_date", "epoch"]):
        by_clock = cell.groupby("time_clock")["n_bins"].first()
        if {"stop", "arrival"} <= set(by_clock.index):
            assert by_clock["arrival"] < by_clock["stop"], keys


def test_cohort_audit_attributes_the_missing_arrival_clock(
    without_outcome_shift,
):
    """A stop with no arrival clock is a common-cohort drop, not an unexplained one."""
    stop, onsets = _toy_tables(n_stops_without_arrival=2)
    with threadpool_limits(limits=1):
        out = fprm.fit_per_bin_rate_models(stop, onsets, run_cv=False, run_gof=False)

    records = fprm._cohort_records(out, stop, out.attrs["cohort_stop_drops"])
    by_model = {(r["model"], r["population"]): r for r in records}
    for (name, _outcome), record in by_model.items():
        # The arrival clock is missing at the stop level, so the stop-clock
        # rungs carry it as somebody else's column and the arrival rungs as
        # their own; either way all four stops are accounted for.
        attributed = (
            record["n_drop_own_predictor_missing"]
            + record["n_drop_common_cohort_other"]
        )
        assert attributed == 4, name
        if name in ARRIVAL_MODELS:
            assert record["n_drop_own_predictor_missing"] == 4, name
        else:
            assert record["n_drop_common_cohort_other"] == 4, name


def _cohort_stop_table() -> pd.DataFrame:
    """Six stops of one session spanning the occupancy states."""
    return pd.DataFrame(
        {
            "animal": ["a"] * 6,
            "session_date": [1] * 6,
            "epoch": ["02_r1"] * 6,
            "stop_id": [f"s{i}" for i in range(6)],
            "occupancy_state": [
                "at_well",
                "at_well",
                "in_transit",
                "in_transit",
                "mixed",
                "at_well",
            ],
        }
    )


@pytest.fixture()
def stubbed_run(monkeypatch, tmp_path):
    """Run ``main`` without fitting: record what the fitter was handed."""
    seen: dict = {}

    stop = _cohort_stop_table()

    # stamped by the configured SWR source, as the shipped tables are
    stamped_stop = stamp_swr_source(stop.copy())
    stamped_other = stamp_swr_source(pd.DataFrame())

    def _read_table(path):
        return stamped_stop if "stop_table" in str(path) else stamped_other

    def _fit(stop_table, onsets, **kwargs):
        seen["stop_ids"] = list(stop_table["stop_id"])
        out = pd.DataFrame(
            {
                "animal": ["a"],
                "session_date": [1],
                "epoch": ["02_r1"],
                "outcome": ["non_local"],
                "model": ["M0_const"],
                "n_stops": [3],
            }
        )
        out.attrs["cohort_stop_drops"] = {}
        return out

    monkeypatch.setattr(fprm, "RC", tmp_path)
    monkeypatch.setattr(fprm, "read_table", _read_table)
    monkeypatch.setattr(fprm, "require_covariate_schema", lambda *a, **k: None)
    monkeypatch.setattr(fprm, "record_inputs", lambda inputs: {})
    monkeypatch.setattr(fprm, "attach_is_swr_to_onsets", lambda o, e: pd.DataFrame())
    monkeypatch.setattr(fprm, "fit_per_bin_rate_models", _fit)
    monkeypatch.setattr(fprm, "_cohort_records", lambda *a, **k: [])
    monkeypatch.setattr(
        fprm, "write_cohort_audit", lambda records, path: seen.setdefault("audit", path)
    )
    return seen, tmp_path


def test_default_cohort_is_at_well_and_publishes_the_canonical_names(
    monkeypatch, stubbed_run
):
    seen, tmp_path = stubbed_run
    monkeypatch.setattr(fprm.sys, "argv", ["fit_per_bin_rate_models.py"])

    fprm.main()

    # Only the well-anchored stops reached the fitter -- the 'mixed' stop s4 is
    # one of them (an at-well bout whose first samples read as in-transit).
    assert seen["stop_ids"] == ["s0", "s1", "s4", "s5"]
    out = tmp_path / "models_per_bin_rate" / "all_models.pkl"
    assert out.exists()
    assert seen["audit"].name == "per_bin_rate_cohort_audit.csv"
    models = pd.read_pickle(out)
    assert list(models["stop_cohort"]) == ["at_well"]
    provenance = json.loads((out.parent / "all_models.provenance.json").read_text())
    assert provenance["stop_cohort"] == "at_well"


def test_all_stops_cohort_publishes_the_suffixed_bundle(monkeypatch, stubbed_run):
    seen, tmp_path = stubbed_run
    monkeypatch.setattr(
        fprm.sys,
        "argv",
        [
            "fit_per_bin_rate_models.py",
            "--stop-cohort",
            "all_stops",
            "--max-per-animal",
            "1",
        ],
    )

    fprm.main()

    # No occupancy restriction: every stop is fit.
    assert seen["stop_ids"] == [f"s{i}" for i in range(6)]
    out = tmp_path / "models_per_bin_rate" / "all_models_allstops_smoke.pkl"
    assert out.exists()
    # The canonical name is never written by a sensitivity run.
    assert not (tmp_path / "models_per_bin_rate" / "all_models.pkl").exists()
    assert seen["audit"].name == "per_bin_rate_cohort_audit_allstops_smoke.csv"
    assert list(pd.read_pickle(out)["stop_cohort"]) == ["all_stops"]


def test_in_transit_cohort_publishes_its_own_bundle(monkeypatch, stubbed_run):
    seen, tmp_path = stubbed_run
    monkeypatch.setattr(
        fprm.sys,
        "argv",
        ["fit_per_bin_rate_models.py", "--stop-cohort", "in_transit"],
    )

    fprm.main()

    assert seen["stop_ids"] == ["s2", "s3"]
    assert (tmp_path / "models_per_bin_rate" / "all_models_intransit.pkl").exists()


def test_unknown_cohort_is_refused_by_the_parser(monkeypatch, capsys):
    monkeypatch.setattr(
        fprm.sys,
        "argv",
        ["fit_per_bin_rate_models.py", "--stop-cohort", "at_the_well"],
    )

    with pytest.raises(SystemExit) as excinfo:
        fprm.main()

    assert excinfo.value.code == 2
    assert "at_the_well" in capsys.readouterr().err


def test_tables_from_another_swr_source_are_refused(monkeypatch, stubbed_run):
    """The bins are dropped by the configured source's excluded time, so tables
    made under another source (another artifact run, say) would mix two masks."""
    from continuum_swr_replay.analysis.events import swr_source

    monkeypatch.setattr(fprm.sys, "argv", ["fit_per_bin_rate_models.py"])
    monkeypatch.setattr(swr_source, "ARTIFACT_INTERVALS_RUN_ID", "another_run")

    with pytest.raises(swr_source.SwrSourceError, match="another_run"):
        fprm.main()


# ── Row order of the saved frame (task 27) ──

#: Shuffles each session's rows differently on every call, as a fan-out split
#: below the session would deliver them.
_ROW_ORDER = np.random.default_rng(0)


def _fake_fit_one_session(keys, stops_g, onsets_g, run_cv, run_gof):
    """One row per (outcome, model), carrying values that identify the session."""
    rows = [
        {
            "outcome": outcome,
            "model": model,
            **dict(zip(fprm.SESSION_COLUMNS, keys, strict=True)),
            "n_stops": len(stops_g),
            "n_events": len(onsets_g),
        }
        for outcome in POPULATIONS
        for model in MODEL_PER_BIN_RATE
    ]
    rows = [rows[i] for i in _ROW_ORDER.permutation(len(rows))]
    drops = dict.fromkeys(POPULATIONS, 0)
    return keys, rows, {}, {}, drops, {}, 0.0


def _sessions_frame(n_sessions: int = 5):
    """Stops and onsets of several sessions, one session per animal-day."""
    keys = [(f"r{i % 2}", 20200100 + i, "02_r1") for i in range(n_sessions)]
    stops = pd.DataFrame(
        [
            dict(zip(fprm.SESSION_COLUMNS, k, strict=True))
            for k in keys
            for _ in range(3)
        ]
    )
    onsets = stops.iloc[: len(stops) - 1].copy()
    return stops, onsets


@pytest.mark.parametrize("finish_order", ["reversed", "shuffled"])
def test_saved_frame_does_not_depend_on_the_order_sessions_finish(
    monkeypatch, finish_order
):
    """Workers finish in any order, and a session's rows may arrive in any
    order; the frame main saves must follow neither."""
    stop, onsets = _sessions_frame()
    monkeypatch.setattr(fprm, "_fit_one_session", _fake_fit_one_session)
    serial = fprm.fit_per_bin_rate_models(stop, onsets, workers=1)

    monkeypatch.setattr(fprm, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(fprm, "as_completed", completion_order(finish_order))
    parallel = fprm.fit_per_bin_rate_models(stop, onsets, workers=3)

    pd.testing.assert_frame_equal(parallel, serial)
    assert parallel.attrs == serial.attrs
    key = [*fprm.SESSION_COLUMNS, "outcome", "model"]
    assert not parallel.duplicated(key).any()
    pd.testing.assert_frame_equal(
        parallel, parallel.sort_values(key, ignore_index=True)
    )


def test_stop_sessions_without_onsets_are_named_not_dropped_silently(
    monkeypatch, caplog
):
    """A stop-session with no onset rows is skipped, and the skip is logged."""
    stop, onsets = _sessions_frame()
    silent = tuple(stop.iloc[0][list(fprm.SESSION_COLUMNS)])
    onsets = onsets[
        ~(
            onsets[list(fprm.SESSION_COLUMNS)]
            == pd.Series(silent, index=fprm.SESSION_COLUMNS)
        ).all(axis=1)
    ]
    monkeypatch.setattr(fprm, "_fit_one_session", _fake_fit_one_session)
    with caplog.at_level("WARNING", logger=fprm.logger.name):
        out = fprm.fit_per_bin_rate_models(stop, onsets, workers=1)

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warnings == [
        "Skipping 1 stop-session(s) with no event onsets: " + "_".join(map(str, silent))
    ]
    fitted = set(map(tuple, out[list(fprm.SESSION_COLUMNS)].drop_duplicates().values))
    assert silent not in fitted
    assert len(fitted) == stop.groupby(list(fprm.SESSION_COLUMNS)).ngroups - 1
