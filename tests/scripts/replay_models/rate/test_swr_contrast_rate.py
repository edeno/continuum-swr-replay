"""Cohort gating and separation flagging in the is_swr-contrast rate runner.

The ladders' own tests are in
``tests/analysis/replay_models/rate/test_swr_contrast_rate.py``; these cover the
runner's loading, aggregation and writing.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_definition.analysed_time import (
    stamp_swr_source,
)
from continuum_swr_replay.analysis.replay_models.rate import (
    swr_contrast_rate as contrast,
)
from scripts.replay_models.rate import fit_swr_contrast_rate as mod
from tests.analysis.replay_models.rate.test_swr_contrast_rate import fit_two_sessions
from tests.support._executor_fixtures import InlineExecutor, completion_order


def test_aggregate_only_refuses_fits_without_per_term_flags(
    replay_content_root, monkeypatch
):
    """A pre-term-level fit cannot be re-aggregated under the per-term drop rule.

    Without ``separated_terms`` every session would read as unseparated for every
    term, silently turning the identifiability filter off instead of applying it.
    """
    out_dir = replay_content_root / "models_per_bin_rate"
    out_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "population": ["contrast"],
            "model": ["S1_rew"],
            "cohort": [contrast.PRIMARY_COHORT],
            "animal": ["a"],
            "coef_is_swr:reward_history_rcos_0": [0.3],
            "separated": [False],
        }
    ).to_pickle(out_dir / "swr_contrast_rate.pkl")

    with pytest.raises(SystemExit, match="separated_terms"):
        mod._aggregate_only(
            smoke=False, cohort="at_well", cli=["fit_swr_contrast_rate.py"]
        )


def test_aggregate_only_requires_the_per_session_fits(replay_content_root, monkeypatch):
    (replay_content_root / "models_per_bin_rate").mkdir(parents=True)

    with pytest.raises(SystemExit, match="no per-session fits"):
        mod._aggregate_only(
            smoke=False, cohort="at_well", cli=["fit_swr_contrast_rate.py"]
        )


def test_aggregate_only_reads_the_cohort_bundle(monkeypatch, replay_content_root):
    """``--aggregate-only`` must look for the cohort's artifact, not the canonical one.

    Re-aggregating an ``_allstops`` run against the unsuffixed pickle would
    silently rewrite the canonical audited table from a sensitivity fit.
    """

    with pytest.raises(SystemExit) as excinfo:
        mod._aggregate_only(
            smoke=False, cohort="all_stops", cli=["fit_swr_contrast_rate.py"]
        )

    assert "swr_contrast_rate_allstops.pkl" in str(excinfo.value)

    with pytest.raises(SystemExit) as canonical:
        mod._aggregate_only(
            smoke=False, cohort="at_well", cli=["fit_swr_contrast_rate.py"]
        )

    assert "swr_contrast_rate.pkl" in str(canonical.value)


# ── The audited bundle names its own stop population ──


@pytest.fixture(scope="module")
def two_session_fits():
    """Both ladders fit on two synthetic sessions with gated post-outcome stops."""
    return fit_two_sessions()


def _aggregation_inputs(monkeypatch, replay_content_root, stops):
    """Point the aggregation at `replay_content_root` (the fixture's
    ``paths.replay_content_dir()``) and at a synthetic covariate source.

    ``_write_aggregation`` re-reads the qvals and stop tables only to recover the
    per-SD factors for its ``std_*`` columns, which is not what these tests are
    about; the synthetic stop frame carries the same covariate columns, so the
    aggregation runs end to end without the canonical tables.
    """
    out_dir = replay_content_root / "models_per_bin_rate"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mod, "read_table", lambda _path: stamp_swr_source(stops))
    monkeypatch.setattr(mod, "require_covariate_schema", lambda *a, **k: None)
    monkeypatch.setattr(contrast, "add_uncertainty_covariates", lambda frame: frame)
    return out_dir


def test_audited_bundle_carries_the_cohort_its_fits_were_made_on(
    two_session_fits, monkeypatch, replay_content_root
):
    """A sensitivity bundle must not be readable as the canonical one."""
    fits, stops, _drops, _gated = two_session_fits
    out_dir = _aggregation_inputs(monkeypatch, replay_content_root, stops)
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
    two_session_fits, monkeypatch, replay_content_root
):
    """Two populations concatenated into one bundle is not one audited ladder."""
    fits, stops, _drops, _gated = two_session_fits
    out_dir = _aggregation_inputs(monkeypatch, replay_content_root, stops)
    mixed = fits.assign(
        stop_cohort=np.where(np.arange(len(fits)) % 2, "at_well", "all_stops")
    )

    with pytest.raises(ValueError, match="one stop cohort"):
        mod._write_aggregation(mixed, out_dir, "", {}, ["fit_swr_contrast_rate.py"])


def test_aggregate_only_records_the_bundle_it_read_by_name(
    monkeypatch, replay_content_root
):
    """The sidecar names the sensitivity pickle, not the canonical one.

    The freshness sweep resolves a recorded input name to a file beside the
    sidecar, so a bare ``swr_contrast_rate`` would point an ``_allstops``
    aggregate at the canonical at-well fit and report it as permanently stale.
    """
    out_dir = replay_content_root / "models_per_bin_rate"
    out_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "population": ["contrast"],
            "model": ["S1_rew"],
            "cohort": [contrast.PRIMARY_COHORT],
            "animal": ["a"],
            "separated": [False],
            "separated_terms": [""],
            "stop_cohort": ["all_stops"],
        }
    ).to_pickle(out_dir / "swr_contrast_rate_allstops.pkl")
    recorded: dict[str, object] = {}
    monkeypatch.setattr(
        mod,
        "_write_aggregation",
        lambda models, out, suffix, inputs, cli: recorded.update(inputs) or 0,
    )

    mod._aggregate_only(
        smoke=False, cohort="all_stops", cli=["fit_swr_contrast_rate.py"]
    )

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
                    "cohort": contrast.PRIMARY_COHORT,
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
    monkeypatch, replay_content_root, stop_cohort, suffix
):
    """``std_*`` rescales by the SD of the population the fits were made on.

    Standardizing an at-well coefficient by an all-stops SD would report a
    per-SD effect for stops the rung was never fit on.  ``all_stops`` keeps
    every stop, so the same code path leaves that bundle's SDs at the
    full-table values.
    """
    stops, qvals = _sd_source_tables()
    out_dir = replay_content_root / "models_per_bin_rate"
    out_dir.mkdir(parents=True)
    monkeypatch.setattr(
        mod,
        "read_table",
        lambda path: qvals if "qvals" in str(path) else stamp_swr_source(stops),
    )
    monkeypatch.setattr(mod, "require_covariate_schema", lambda *a, **k: None)
    monkeypatch.setattr(contrast, "add_uncertainty_covariates", lambda frame: frame)

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
    monkeypatch.setattr(mod, "attach_event_ends", lambda onsets, events: onsets)
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


# ── Row order of the saved per-session fits ──

#: Shuffles each session's rows differently on every call, as a fan-out split
#: below the session would deliver them.
_ROW_ORDER = np.random.default_rng(0)


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
        for population in (*contrast.STRATA, "contrast")
        for model in ("R0_null", "R1_rew")
        for cohort in (contrast.PRIMARY_COHORT, contrast.POST_OUTCOME_COHORT)
    ]
    rows = [rows[i] for i in _ROW_ORDER.permutation(len(rows))]
    return keys, rows, {}, {"contrast": 0}


@pytest.fixture()
def stubbed_fan_out(monkeypatch, replay_content_root):
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
    monkeypatch.setattr(mod, "_load", lambda cohort: (stops, onsets, 0))
    monkeypatch.setattr(mod, "record_inputs", lambda inputs: {})
    monkeypatch.setattr(contrast, "fit_contrast_session", _fake_fit_one_session)
    monkeypatch.setattr(mod, "contrast_cohort_records", lambda *a, **k: [])
    monkeypatch.setattr(mod, "write_cohort_audit", lambda records, path: None)
    monkeypatch.setattr(
        mod, "_write_aggregation", lambda models, *a, **k: aggregated.append(models)
    )
    monkeypatch.setattr(contrast, "ProcessPoolExecutor", InlineExecutor)
    saved = replay_content_root / "models_per_bin_rate" / "swr_contrast_rate.pkl"

    def run(workers: int, finish=lambda futures: list(futures)):
        monkeypatch.setattr(contrast, "as_completed", finish)
        mod.main(["--workers", str(workers)])
        return saved.read_bytes(), pd.read_pickle(saved), aggregated[-1]

    return run


@pytest.mark.parametrize("finish_order", ["reversed", "shuffled"])
def test_saved_fits_do_not_depend_on_the_order_sessions_finish(
    stubbed_fan_out, finish_order
):
    """Workers finish in any order, and a session's rows may arrive in any
    order; the saved pickle must be the same bytes."""
    serial_bytes, serial, _ = stubbed_fan_out(workers=1)
    parallel_bytes, parallel, aggregated = stubbed_fan_out(
        workers=3, finish=completion_order(finish_order)
    )

    pd.testing.assert_frame_equal(parallel, serial)
    assert parallel_bytes == serial_bytes
    # the audited aggregation is computed from the same ordered frame
    pd.testing.assert_frame_equal(aggregated, parallel)
    key = [*mod.SESSION_COLUMNS, "population", "model", "cohort"]
    assert not parallel.duplicated(key).any()
    pd.testing.assert_frame_equal(
        parallel, parallel.sort_values(key, ignore_index=True)
    )


def test_main_records_the_arguments_it_was_given(monkeypatch, replay_content_root):
    """An in-process ``main([...])`` records its own arguments after the program,
    not the command line of the process that called it."""
    out_dir = replay_content_root / "models_per_bin_rate"
    out_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "population": ["contrast"],
            "model": ["S1_rew"],
            "cohort": [contrast.PRIMARY_COHORT],
            "animal": ["a"],
            "separated": [False],
            "separated_terms": [""],
            "stop_cohort": ["all_stops"],
        }
    ).to_pickle(out_dir / "swr_contrast_rate_allstops.pkl")
    recorded: list[list[str]] = []
    monkeypatch.setattr(
        mod,
        "_write_aggregation",
        lambda models, out, suffix, inputs, cli: recorded.append(cli) or 0,
    )
    argv = ["--aggregate-only", "--stop-cohort", "all_stops"]

    mod.main(argv)

    assert recorded == [[sys.argv[0], *argv]]
