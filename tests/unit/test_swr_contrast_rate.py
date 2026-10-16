"""Cohort gating and separation flagging in the is_swr-contrast rate runner.

Two guards:

* ``_mask_post_outcome_covariates`` -- a stop that starts *before* the trial's
  outcome is known cannot carry a post-outcome covariate, so those values are
  NaN'd (and later removed by the ladder's cohort freeze) rather than being fit
  as if the animal already knew the outcome.
* ``_fit`` -- the separation flag is judged on the effect scale, so a
  small-SD design column with an ordinary effect is not mistaken for a
  quasi-separated fit just because its per-unit coefficient is large.
* ``_fit_one_session`` -- the gated stops cost only the rungs that need a
  post-outcome covariate: those are fit on a second cohort together with their
  ladder's null, while the nested ladder keeps every stop.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus.swr_source import stamp_swr_source
from continuum_swr_replay.analysis.replay import (
    BIN_S,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    POPULATIONS,
    TMAX,
    aggregate_to_timebins,
)
from scripts.rate import fit_swr_contrast_rate as mod


@pytest.fixture
def stop_frame() -> pd.DataFrame:
    """Three stops: one starts before its outcome is known, two do not."""
    return pd.DataFrame(
        {
            "stop_id": [1, 2, 3],
            "stop_start": [10.0, 20.0, 30.0],
            # stop 2 starts 1 s before the outcome is known; stop 3 starts
            # exactly at it (not before -> keeps its post-outcome values).
            "outcome_known_time": [9.5, 21.0, 30.0],
            "occupancy_state": ["at_well", "at_well", "in_transit"],
            "rpe_postoutcome": [0.4, -0.6, 0.2],
            "reward_belief_mean_update_l1_all_ports_postoutcome": [0.1, 0.2, 0.3],
            "relative_value_switch_minus_stay_preoutcome": [1.0, 2.0, 3.0],
        }
    )


def test_post_outcome_gate_nans_only_pre_outcome_stops(stop_frame):
    gated, counts = mod._mask_post_outcome_covariates(stop_frame)

    assert gated["rpe_postoutcome"].tolist()[0] == 0.4
    assert np.isnan(gated["rpe_postoutcome"].iloc[1])
    assert gated["rpe_postoutcome"].iloc[2] == 0.2
    update = gated["reward_belief_mean_update_l1_all_ports_postoutcome"]
    assert update.iloc[0] == 0.1
    assert np.isnan(update.iloc[1])
    assert update.iloc[2] == 0.3
    # Pre-outcome covariates are untouched on the same rows.
    assert gated["relative_value_switch_minus_stay_preoutcome"].tolist() == [
        1.0,
        2.0,
        3.0,
    ]
    assert counts["at_well"] == 1
    assert counts.get("in_transit", 0) == 0


def test_post_outcome_gate_does_not_mutate_the_input(stop_frame):
    mod._mask_post_outcome_covariates(stop_frame)
    assert stop_frame["rpe_postoutcome"].tolist() == [0.4, -0.6, 0.2]


def test_post_outcome_gate_requires_the_outcome_clock(stop_frame):
    with pytest.raises(ValueError, match="outcome_known_time"):
        mod._mask_post_outcome_covariates(stop_frame.drop(columns="outcome_known_time"))


def test_small_sd_predictor_with_ordinary_effect_is_not_flagged_separated():
    """|coef| = 12 on a column with SD 0.03 is a 0.36-log-unit effect, not separation.

    The legacy ``|coef| >= 10`` rule flagged exactly this design; the effect-scale
    rule (``bse * SD``) does not.
    """
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(0.0, 0.03, n)
    exposure = np.full(n, 0.015)
    rate = np.exp(np.log(2.0) + 12.0 * x)
    design = pd.DataFrame(
        {
            "stop_id": np.repeat(np.arange(n // 10), 10),
            "exposure": exposure,
            "x": x,
            "count": rng.poisson(rate * exposure),
        }
    ).astype({"count": float})

    row, reason = mod._fit(design, "x", ["x"])

    assert reason is None
    assert row["coef_x"] == pytest.approx(12.0, abs=3.0)
    # The design column's SD is small, so the per-unit coefficient is large ...
    assert abs(row["coef_x"]) >= 10.0
    # ... but the effect-scale standard error is far below the separation bound.
    assert row["bse_x"] * x.std() < 1.0
    assert row["separated"] is False
    assert row["converged"] is True


# ── Two cohorts per ladder ──


def _gated_session(session_date: int, n_stops: int = 24, seed: int = 0):
    """One synthetic session where every third stop is gated as pre-outcome.

    Returns ``(stop_frame, onset_frame, gated_stop_ids)``. The gate is applied
    through :func:`_mask_post_outcome_covariates`, exactly as the runner does.
    """
    rng = np.random.default_rng(seed)
    stop_rows, onset_rows = [], []
    for i in range(n_stops):
        stop_id = session_date * 1000 + i
        immobility = 1.2
        pre_outcome = i % 3 == 0  # a third of the stops start before the outcome
        stop_rows.append(
            {
                "animal": "a",
                "session_date": session_date,
                "epoch": "02_r1",
                "stop_id": stop_id,
                "immobility_duration_s": immobility,
                "stop_start": 100.0 * i,
                "outcome_known_time": 100.0 * i + (1.0 if pre_outcome else -1.0),
                "occupancy_state": "at_well",
                "reward_history_rcos_0": float(i % 2),
                "reward_history_rcos_1": float(rng.uniform(0, 1)),
                "reward_history_rcos_2": float(rng.uniform(0, 1)),
                "reward_history_incomplete": 0.0,
                **{c: float(rng.normal()) for c in mod.STOP_COVARS},
            }
        )
        for _ in range(rng.poisson(14 * immobility)):
            onset_rows.append(
                {
                    "animal": "a",
                    "session_date": session_date,
                    "epoch": "02_r1",
                    "stop_id": stop_id,
                    "event_type": "non_local",
                    "t_since_stop_onset": float(rng.uniform(0, immobility)),
                    "is_swr": bool(rng.integers(0, 2)),
                }
            )
    stops = pd.DataFrame(stop_rows)
    gated = [r["stop_id"] for i, r in enumerate(stop_rows) if i % 3 == 0]
    return stops, pd.DataFrame(onset_rows), gated


@pytest.fixture(scope="module")
def two_session_fits():
    """Both ladders fit on two synthetic sessions with gated post-outcome stops."""
    from threadpoolctl import threadpool_limits

    frames, rows, drops, gated_ids = [], [], {}, []
    for session_date in (1, 2):
        stops, onsets, gated = _gated_session(session_date, seed=session_date)
        stops, _ = mod._mask_post_outcome_covariates(stops)
        frames.append(stops)
        gated_ids.extend(gated)
        with threadpool_limits(limits=1):
            _, session_rows, skips, session_drops = mod._fit_one_session(
                ("a", session_date, "02_r1"), stops, onsets
            )
        assert not skips, skips
        rows.extend(session_rows)
        for key, value in session_drops.items():
            drops[key] = drops.get(key, 0) + value
    return pd.DataFrame(rows), pd.concat(frames, ignore_index=True), drops, gated_ids


def test_primary_rungs_share_one_cohort_unaffected_by_the_gate(two_session_fits):
    """The gate costs the nested ladder nothing: every primary rung, all stops."""
    fits, stops, _drops, _gated = two_session_fits
    primary = fits[fits.cohort == "primary"]
    assert set(primary.model) == {
        m
        for specs in (mod.G_SPECS, mod.S_SPECS)
        for m in specs
        if m not in mod.POST_OUTCOME_RATE_RUNGS
    }
    for (session_date, population), cell in primary.groupby(
        ["session_date", "population"]
    ):
        assert cell["n_bins"].nunique() == 1, (session_date, population)
        assert cell["n_stops"].nunique() == 1, (session_date, population)
        assert cell["n_stops"].iloc[0] == 24  # every stop of the session


def test_post_outcome_rung_and_its_null_share_the_gated_cohort(two_session_fits):
    """G5_rpe and the G0_base refit beside it lose exactly the gated stops' bins."""
    fits, stops, _drops, gated = two_session_fits
    design = aggregate_to_timebins(
        stops[stops.session_date == 1],
        POPULATIONS["non_local"](_gated_session(1, seed=1)[1].assign(is_swr=False)),
        bin_s=BIN_S,
        tmax=TMAX,
        reward_basis=True,
        self_history_lags=N_HISTORY_LAGS,
        self_history_basis=N_HISTORY_BASIS,
        stop_covariates=mod.STOP_COVARS,
    )
    gated_bins = int(design["stop_id"].isin(gated).sum())
    assert gated_bins > 0

    cell = fits[(fits.session_date == 1) & (fits.population == "non_local")]
    primary_bins = cell.loc[cell.cohort == "primary", "n_bins"].iloc[0]
    post = cell[cell.cohort == "post_outcome"]
    assert set(post.model) == {"G5_rpe", "G7_update", "G0_base"}
    assert post["n_bins"].nunique() == 1
    assert post["n_stops"].iloc[0] == 16  # 24 stops - 8 gated
    assert int(primary_bins) - int(post["n_bins"].iloc[0]) == gated_bins


def test_contrast_ladder_keeps_its_pairing_in_both_cohorts(two_session_fits):
    """The stacked design carries two rows per bin in the post-outcome cohort too."""
    fits, _stops, _drops, _gated = two_session_fits
    cell = fits[fits.session_date == 1]
    contrast = cell[cell.population == "contrast"]
    stratum = cell[cell.population == "non_local_swr"]
    for cohort in ("primary", "post_outcome"):
        assert (
            contrast.loc[contrast.cohort == cohort, "n_bins"].iloc[0]
            == 2 * stratum.loc[stratum.cohort == cohort, "n_bins"].iloc[0]
        )
    assert set(contrast.loc[contrast.cohort == "post_outcome", "model"]) == {
        "S7_rpe",
        "S9_update",
        "S0_rate",
    }


def test_fitted_rows_are_unique_per_session_population_model_cohort(two_session_fits):
    fits, _stops, _drops, _gated = two_session_fits
    keys = ["animal", "session_date", "epoch", "population", "model", "cohort"]
    assert not fits.duplicated(subset=keys).any()
    # The ladder null appears once per cohort, not twice in one.
    nulls = fits[(fits.model == "G0_base") & (fits.population == "non_local")]
    assert sorted(nulls.cohort.unique()) == ["post_outcome", "primary"]
    assert len(nulls) == 4  # 2 sessions x 2 cohorts


def test_cohort_audit_reconciles_across_both_cohorts(two_session_fits):
    fits, stops, drops, gated = two_session_fits
    records = mod._cohort_records(fits, stops, 0, drops)
    for record in records:
        recon = record["n_in"] - sum(
            record[f]
            for f in (
                "n_drop_qvals_gap",
                "n_drop_own_predictor_missing",
                "n_drop_nonfinite",
                "n_drop_zero_response",
                "n_drop_common_cohort_other",
                "n_excluded_session",
            )
        )
        assert recon == record["n_modeled"], record

    by = {(r["model"], r["population"], r["cohort_mode"]): r for r in records}
    n_gated = len(gated)
    assert n_gated == 16  # 8 per session
    # The rung that needs the gated covariate books them as its own missing.
    g5 = by[("G5_rpe", "all", "common_post_outcome")]
    assert g5["n_drop_own_predictor_missing"] == n_gated
    assert g5["n_drop_common_cohort_other"] == 0
    assert g5["n_modeled"] == 32  # 2 sessions x 16 stops
    # The null refit beside it loses the same stops, but to another rung's column.
    null_post = by[("G0_base", "all", "common_post_outcome")]
    assert null_post["n_drop_own_predictor_missing"] == 0
    assert null_post["n_drop_common_cohort_other"] == n_gated
    assert null_post["n_modeled"] == 32
    # On the primary cohort the ladder null keeps every stop.
    null_primary = by[("G0_base", "all", "common")]
    assert null_primary["n_drop_own_predictor_missing"] == 0
    assert null_primary["n_drop_common_cohort_other"] == 0
    assert null_primary["n_modeled"] == 48  # 2 sessions x 24 stops


def test_aggregate_only_refuses_fits_without_per_term_flags(tmp_path, monkeypatch):
    """A pre-term-level fit cannot be re-aggregated under the per-term drop rule.

    Without ``separated_terms`` every session would read as unseparated for every
    term, silently turning the identifiability filter off instead of applying it.
    """
    out_dir = tmp_path / "models_per_bin_rate"
    out_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "population": ["contrast"],
            "model": ["S1_rew"],
            "cohort": [mod.PRIMARY_COHORT],
            "animal": ["a"],
            "coef_is_swr:reward_history_rcos_0": [0.3],
            "separated": [False],
        }
    ).to_pickle(out_dir / "swr_contrast_rate.pkl")
    monkeypatch.setattr(mod, "RC", tmp_path)

    with pytest.raises(SystemExit, match="separated_terms"):
        mod._aggregate_only(smoke=False, cohort="at_well")


def test_aggregate_only_requires_the_per_session_fits(tmp_path, monkeypatch):
    (tmp_path / "models_per_bin_rate").mkdir(parents=True)
    monkeypatch.setattr(mod, "RC", tmp_path)

    with pytest.raises(SystemExit, match="no per-session fits"):
        mod._aggregate_only(smoke=False, cohort="at_well")


def test_aggregate_only_reads_the_cohort_bundle(monkeypatch, tmp_path):
    """``--aggregate-only`` must look for the cohort's artifact, not the canonical one.

    Re-aggregating an ``_allstops`` run against the unsuffixed pickle would
    silently rewrite the canonical audited table from a sensitivity fit.
    """
    monkeypatch.setattr(mod, "RC", tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        mod._aggregate_only(smoke=False, cohort="all_stops")

    assert "swr_contrast_rate_allstops.pkl" in str(excinfo.value)

    with pytest.raises(SystemExit) as canonical:
        mod._aggregate_only(smoke=False, cohort="at_well")

    assert "swr_contrast_rate.pkl" in str(canonical.value)


# ── The audited bundle names its own stop population ──


def _aggregation_inputs(monkeypatch, tmp_path, stops):
    """Point the aggregation at ``tmp_path`` and at a synthetic covariate source.

    ``_write_aggregation`` re-reads the qvals and stop tables only to recover the
    per-SD factors for its ``std_*`` columns, which is not what these tests are
    about; the synthetic stop frame carries the same covariate columns, so the
    aggregation runs end to end without the canonical tables.
    """
    out_dir = tmp_path / "models_per_bin_rate"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mod, "RC", tmp_path)
    monkeypatch.setattr(mod, "read_table", lambda _path: stamp_swr_source(stops))
    monkeypatch.setattr(mod, "require_covariate_schema", lambda *a, **k: None)
    monkeypatch.setattr(mod, "add_uncertainty_covariates", lambda frame: frame)
    return out_dir


def test_audited_bundle_carries_the_cohort_its_fits_were_made_on(
    two_session_fits, monkeypatch, tmp_path
):
    """A sensitivity bundle must not be readable as the canonical one."""
    fits, stops, _drops, _gated = two_session_fits
    out_dir = _aggregation_inputs(monkeypatch, tmp_path, stops)
    # The aggregation refuses to publish a term fit in fewer than five
    # sessions, so the two synthetic sessions are replicated across three
    # animals. The coefficients repeat; what is under test is the label the
    # aggregate carries, not its value.
    labelled = pd.concat(
        [
            fits.assign(
                stop_cohort="all_stops",
                animal=animal,
                session_date=fits["session_date"] + 10 * replica,
            )
            for replica, animal in enumerate("abc")
        ],
        ignore_index=True,
    )

    n_rows = mod._write_aggregation(
        labelled, out_dir, "_allstops", {}, ["fit_swr_contrast_rate.py"]
    )

    assert n_rows > 0
    audited = pd.read_pickle(out_dir / "swr_contrast_rate_audited_allstops.pkl")
    assert set(audited["stop_cohort"]) == {"all_stops"}
    sidecar = json.loads(
        (out_dir / "swr_contrast_rate_audited_allstops.provenance.json").read_text()
    )
    assert sidecar["stop_cohort"] == "all_stops"


def test_aggregation_refuses_fits_that_mix_stop_cohorts(
    two_session_fits, monkeypatch, tmp_path
):
    """Two populations concatenated into one bundle is not one audited ladder."""
    fits, stops, _drops, _gated = two_session_fits
    out_dir = _aggregation_inputs(monkeypatch, tmp_path, stops)
    mixed = fits.assign(
        stop_cohort=np.where(np.arange(len(fits)) % 2, "at_well", "all_stops")
    )

    with pytest.raises(ValueError, match="one stop cohort"):
        mod._write_aggregation(mixed, out_dir, "", {}, ["fit_swr_contrast_rate.py"])


def test_aggregate_only_records_the_bundle_it_read_by_name(monkeypatch, tmp_path):
    """The sidecar names the sensitivity pickle, not the canonical one.

    The freshness sweep resolves a recorded input name to a file beside the
    sidecar, so a bare ``swr_contrast_rate`` would point an ``_allstops``
    aggregate at the canonical at-well fit and report it as permanently stale.
    """
    out_dir = tmp_path / "models_per_bin_rate"
    out_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "population": ["contrast"],
            "model": ["S1_rew"],
            "cohort": [mod.PRIMARY_COHORT],
            "animal": ["a"],
            "separated": [False],
            "separated_terms": [""],
            "stop_cohort": ["all_stops"],
        }
    ).to_pickle(out_dir / "swr_contrast_rate_allstops.pkl")
    monkeypatch.setattr(mod, "RC", tmp_path)
    recorded: dict[str, object] = {}
    monkeypatch.setattr(
        mod,
        "_write_aggregation",
        lambda models, out, suffix, inputs, cli: recorded.update(inputs) or 0,
    )

    mod._aggregate_only(smoke=False, cohort="all_stops")

    assert list(recorded) == ["swr_contrast_rate_allstops"]


# ── The per-SD reference is scoped to the bundle's own stop cohort ──


def _sd_source_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A stop/qvals pair whose SDs change when the at-well cohort is applied.

    Four well-anchored stops (``at_well``/``mixed``) carry a narrow spread and
    four ``in_transit`` stops a wide one, on one stop-table covariate and one
    qvals covariate, so an SD taken over the full tables cannot be mistaken for
    an SD taken over the well-anchored cohort.
    """
    stops = pd.DataFrame(
        {
            "stop_id": list(range(8)),
            "occupancy_state": [
                "at_well",
                "mixed",
                "at_well",
                "mixed",
                "in_transit",
                "in_transit",
                "in_transit",
                "in_transit",
            ],
            "patch_switch_fraction_last5_completed_trips": [
                1.0,
                1.0,
                2.0,
                2.0,
                10.0,
                20.0,
                30.0,
                40.0,
            ],
        }
    )
    qvals = pd.DataFrame(
        {
            "stop_id": list(range(8)),
            "periswitch_index_trials": [
                0.0,
                0.0,
                1.0,
                1.0,
                100.0,
                200.0,
                300.0,
                400.0,
            ],
        }
    )
    return stops, qvals


def _sd_probe_fits(stop_cohort: str) -> pd.DataFrame:
    """Six sessions over three animals fitting the two probe rungs."""
    rows = []
    for i in range(6):
        for model, term in (
            ("G3_switchrate", "patch_switch_fraction_last5_completed_trips"),
            ("G1_periswitch", "periswitch_index_trials"),
        ):
            rows.append(
                {
                    "animal": "abc"[i % 3],
                    "session_date": i,
                    "epoch": "02_r1",
                    "population": "non_local",
                    "model": model,
                    "cohort": mod.PRIMARY_COHORT,
                    "stop_cohort": stop_cohort,
                    "converged": True,
                    "separated": False,
                    "separated_terms": "",
                    f"coef_{term}": 0.2 + 0.01 * i,
                    f"bse_{term}": 0.05,
                    f"pval_{term}": 0.01,
                    f"sd_{term}": 1.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("stop_cohort", "suffix"), [("at_well", ""), ("all_stops", "_allstops")]
)
def test_std_columns_use_the_bundles_own_cohort_sds(
    monkeypatch, tmp_path, stop_cohort, suffix
):
    """``std_*`` rescales by the SD of the population the fits were made on.

    Standardizing an at-well coefficient by an all-stops SD would report a
    per-SD effect for stops the rung was never fit on.  ``all_stops`` keeps
    every stop, so the same code path leaves that bundle's SDs at the
    full-table values.
    """
    stops, qvals = _sd_source_tables()
    out_dir = tmp_path / "models_per_bin_rate"
    out_dir.mkdir(parents=True)
    monkeypatch.setattr(mod, "RC", tmp_path)
    monkeypatch.setattr(
        mod,
        "read_table",
        lambda path: qvals if "qvals" in str(path) else stamp_swr_source(stops),
    )
    monkeypatch.setattr(mod, "require_covariate_schema", lambda *a, **k: None)
    monkeypatch.setattr(mod, "add_uncertainty_covariates", lambda frame: frame)

    n_rows = mod._write_aggregation(
        _sd_probe_fits(stop_cohort),
        out_dir,
        suffix,
        {},
        ["fit_swr_contrast_rate.py"],
    )
    assert n_rows > 0

    well_anchored = stops["occupancy_state"] != "in_transit"
    expected = {
        "at_well": {
            "patch_switch_fraction_last5_completed_trips": float(
                np.nanstd(
                    stops.loc[
                        well_anchored, "patch_switch_fraction_last5_completed_trips"
                    ]
                )
            ),
            "periswitch_index_trials": float(
                np.nanstd(
                    qvals.loc[well_anchored.to_numpy(), "periswitch_index_trials"]
                )
            ),
        },
        "all_stops": {
            "patch_switch_fraction_last5_completed_trips": float(
                np.nanstd(stops["patch_switch_fraction_last5_completed_trips"])
            ),
            "periswitch_index_trials": float(
                np.nanstd(qvals["periswitch_index_trials"])
            ),
        },
    }[stop_cohort]

    meta = json.loads((out_dir / f"swr_contrast_rate_meta{suffix}.json").read_text())
    assert stop_cohort in meta["sd_scope"]
    for covariate, sd in expected.items():
        assert meta["fit_time_sd_reference"][covariate] == pytest.approx(sd)

    audited = pd.read_pickle(out_dir / f"swr_contrast_rate_audited{suffix}.pkl")
    factors = audited.set_index("term")["std_factor"]
    for covariate, sd in expected.items():
        assert factors[covariate] == pytest.approx(sd)
        assert audited.set_index("term").loc[covariate, "std_coef_animal_median"] == (
            pytest.approx(
                audited.set_index("term").loc[covariate, "coef_animal_median"] * sd
            )
        )

    reported = meta["max_relative_sd_difference_vs_full_tables"]
    if stop_cohort == "all_stops":
        # The restriction is a no-op, so this bundle's SDs are the full ones.
        assert reported["relative_difference"] == pytest.approx(0.0)
    else:
        assert reported["relative_difference"] < -0.9
        assert reported["covariate"] in expected


# ── `_load` restricts the fitted stops to one cohort ──


def _load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A stop/qvals pair spanning the three occupancy states, joinable on stop_id."""
    n = 6
    stops = pd.DataFrame(
        {
            "animal": ["a"] * n,
            "session_date": [1] * n,
            "epoch": ["02_r1"] * n,
            "stop_id": [f"s{i}" for i in range(n)],
            "stop_start": [100.0 * i for i in range(n)],
            "occupancy_state": [
                "at_well",
                "at_well",
                "in_transit",
                "in_transit",
                "mixed",
                "at_well",
            ],
            # only the stop-table-sourced covariate lives here; the rest arrive
            # on the qvals join, exactly as they do on the canonical tables
            "patch_switch_fraction_last5_completed_trips": [0.0] * n,
        }
    )
    qvals = pd.DataFrame(
        {
            "stop_id": [f"s{i}" for i in range(n)],
            # every stop starts after its outcome is known, so the post-outcome
            # gate is not what decides which stops come back
            "outcome_known_time": [100.0 * i - 1.0 for i in range(n)],
            "periswitch_index_trials": [0.0] * n,
            "is_periswitch_post": [0.0] * n,
            "trials_since_block_change": [0.0] * n,
            "relative_value_switch_minus_stay_preoutcome": [0.0] * n,
            "rpe_postoutcome": [0.0] * n,
            "reward_belief_differential_entropy_leaf_mean_preoutcome": [0.0] * n,
            "reward_belief_mean_update_l1_all_ports_postoutcome": [0.0] * n,
        }
    )
    return stops, qvals


@pytest.fixture()
def stubbed_load(monkeypatch):
    """Route ``_load`` at synthetic tables, leaving its cohort step untouched."""
    stops, qvals = _load_tables()
    # the stop and onset tables are stamped by the configured SWR source
    monkeypatch.setattr(
        mod,
        "read_table",
        lambda path: qvals if "qvals" in str(path) else stamp_swr_source(stops.copy()),
    )
    monkeypatch.setattr(mod, "require_covariate_schema", lambda *a, **k: None)
    monkeypatch.setattr(mod, "add_uncertainty_covariates", lambda frame: frame)
    monkeypatch.setattr(mod, "assert_qvals_aligned", lambda *a, **k: None)
    monkeypatch.setattr(
        mod, "attach_is_swr_to_onsets", lambda onsets, events: pd.DataFrame()
    )
    return stops


def test_load_defaults_to_the_well_anchored_stops(stubbed_load):
    stop, _onsets, n_qvals_dropped = mod._load("at_well")

    # The 'mixed' stop s4 is well-anchored -- an at-well bout whose first
    # settling samples read as in-transit -- so it is kept.
    assert list(stop["stop_id"]) == ["s0", "s1", "s4", "s5"]
    assert n_qvals_dropped == 0


def test_load_all_stops_keeps_every_occupancy_state(stubbed_load):
    stop, _onsets, _dropped = mod._load("all_stops")

    assert list(stop["stop_id"]) == [f"s{i}" for i in range(6)]


def test_load_in_transit_keeps_only_the_transit_stops(stubbed_load):
    stop, _onsets, _dropped = mod._load("in_transit")

    assert list(stop["stop_id"]) == ["s2", "s3"]


# ── Row order of the saved per-session fits (task 27) ──


class _InlineExecutor:
    """``ProcessPoolExecutor`` stand-in: runs each task as it is submitted."""

    def __init__(self, max_workers):
        del max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, function, *args):
        from concurrent.futures import Future

        future = Future()
        future.set_result(function(*args))
        return future


def _fake_fit_one_session(keys, stops_g, onsets_g):
    """Rows per (population, model, cohort), with values that identify the session."""
    base = dict(zip(mod.SESSION_COLUMNS, keys, strict=True))
    rows = [
        {
            "population": population,
            "model": model,
            "cohort": cohort,
            **base,
            "n_stops": len(stops_g),
            "n_bins": 10 * len(stops_g),
            "n_events": len(onsets_g),
        }
        for population in (*mod.STRATA, "contrast")
        for model in ("R0_null", "R1_rew")
        for cohort in (mod.PRIMARY_COHORT, mod.POST_OUTCOME_COHORT)
    ]
    return keys, rows, {}, {"contrast": 0}


@pytest.fixture()
def stubbed_fan_out(monkeypatch, tmp_path):
    """Run ``main`` on fake per-session fits; return what it saved per run."""
    keys = [(f"r{i % 2}", 20200100 + i, "02_r1") for i in range(5)]
    stops = pd.DataFrame(
        [
            {**dict(zip(mod.SESSION_COLUMNS, k, strict=True)), "stop_id": f"{j}{n}"}
            for j, k in enumerate(keys)
            for n in range(j + 2)
        ]
    )
    onsets = stops.iloc[:-1].copy()
    aggregated: list[pd.DataFrame] = []
    monkeypatch.setattr(mod, "RC", tmp_path)
    monkeypatch.setattr(mod, "_load", lambda cohort: (stops, onsets, 0))
    monkeypatch.setattr(mod, "record_inputs", lambda inputs: {})
    monkeypatch.setattr(mod, "_fit_one_session", _fake_fit_one_session)
    monkeypatch.setattr(mod, "_cohort_records", lambda *a, **k: [])
    monkeypatch.setattr(mod, "write_cohort_audit", lambda records, path: None)
    monkeypatch.setattr(
        mod, "_write_aggregation", lambda models, *a, **k: aggregated.append(models)
    )
    monkeypatch.setattr(mod, "ProcessPoolExecutor", _InlineExecutor)
    saved = tmp_path / "models_per_bin_rate" / "swr_contrast_rate.pkl"

    def run(workers: int, finish=lambda futures: list(futures)):
        monkeypatch.setattr(mod, "as_completed", finish)
        monkeypatch.setattr(
            mod.sys, "argv", ["fit_swr_contrast_rate.py", "--workers", str(workers)]
        )
        mod.main()
        return saved.read_bytes(), pd.read_pickle(saved), aggregated[-1]

    return run


@pytest.mark.parametrize("finish_order", ["reversed", "shuffled"])
def test_saved_fits_do_not_depend_on_the_order_sessions_finish(
    stubbed_fan_out, finish_order
):
    """Workers finish in any order; the saved pickle must be the same bytes."""

    def finish(futures):
        futures = list(futures)
        if finish_order == "reversed":
            return futures[::-1]
        return [futures[i] for i in np.random.default_rng(27).permutation(len(futures))]

    serial_bytes, serial, _ = stubbed_fan_out(workers=1)
    parallel_bytes, parallel, aggregated = stubbed_fan_out(workers=3, finish=finish)

    pd.testing.assert_frame_equal(parallel, serial)
    assert parallel_bytes == serial_bytes
    # the audited aggregation is computed from the same ordered frame
    pd.testing.assert_frame_equal(aggregated, parallel)
    sessions = parallel[list(mod.SESSION_COLUMNS)].drop_duplicates()
    assert sessions.equals(sessions.sort_values(list(mod.SESSION_COLUMNS)))
