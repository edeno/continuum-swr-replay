"""Output-directory keying for partial content-ladder runs.

A run that fits only some rungs must not publish into the directory that holds
the full ladder: the canonical ``models/`` bundle is a single cohort-matched
set, and a partial run silently replacing part of it leaves rungs that were fit
on different event sets side by side.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay import fit_outputs, model_specs
from continuum_swr_replay.analysis.replay.cohort_audit import event_drop_buckets
from continuum_swr_replay.analysis.replay.fit_provenance import (
    check_fit_provenance,
    record_inputs,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    ModelSpec,
    default_content_model_names,
)
from continuum_swr_replay.analysis.replay.simulation import (
    simulate_dirichlet_multinomial_counts,
    simulate_with_event_covariates,
)
from continuum_swr_replay.data_loaders.constants import (
    ALL_ARMS,
    TRACK_SEGMENT_TO_PATCH,
    WELL_ARMS,
)
from continuum_swr_replay.tables.io import write_table
from scripts.content import fit_replay_content_models as mod

# the runner resolves its directories, and /tmp is itself a symlink on macOS
CONTENT_IN = Path("/tmp/content-in").resolve()
STAGING = Path("/tmp/staging").resolve()


def _args(**overrides) -> argparse.Namespace:
    defaults = {
        "estimator": "conditional_poisson",
        "logit": False,
        "n_jobs": 1,
        "min_events_session_phi": 40,
        "cv": False,
        "event_source": "non_local",
        "hierarchy": "arm",
        "brain_area": "HPC",
        "swr_subset": "all",
        "grain": "timebin",
        "response": "fraction",
        "leaf_only": False,
        "input_dir": CONTENT_IN,
        "output_dir": None,
        "models": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_default_run_writes_the_canonical_directory() -> None:
    config = mod._resolve_run_config(_args())
    assert config.output_dir == CONTENT_IN / "models"
    assert list(config.model_names) == list(
        default_content_model_names(config.hierarchy)
    )


def test_explicit_full_ladder_still_writes_the_canonical_directory() -> None:
    """Naming every default rung is the canonical run, not a subset."""
    names = default_content_model_names(mod.MODEL_HIERARCHY)
    config = mod._resolve_run_config(_args(models=",".join(names)))
    assert config.output_dir == CONTENT_IN / "models"


def test_subset_run_is_keyed_by_its_rungs() -> None:
    config = mod._resolve_run_config(_args(models="M4,Mv,Mu,Mvu"))
    assert config.output_dir == CONTENT_IN / "models_M4-Mv-Mu-Mvu"
    assert list(config.model_names) == ["M4", "Mv", "Mu", "Mvu"]


def test_patch_subset_run_is_keyed_off_the_patch_directory() -> None:
    config = mod._resolve_run_config(_args(hierarchy="patch", models="P3,Pv"))
    assert config.output_dir == CONTENT_IN / "models_patch_P3-Pv"


def test_subset_models_do_not_write_canonical_dir() -> None:
    """An explicit --output-dir cannot aim a subset at the canonical bundle."""
    with pytest.raises(ValueError, match="models"):
        mod._resolve_run_config(_args(models="M4,Mv", output_dir=CONTENT_IN / "models"))
    with pytest.raises(ValueError, match="models_patch"):
        mod._resolve_run_config(
            _args(
                hierarchy="patch",
                models="P3,Pv",
                output_dir=CONTENT_IN / "models_patch",
            )
        )


def test_subset_run_may_be_sent_to_an_explicit_non_canonical_directory() -> None:
    config = mod._resolve_run_config(
        _args(models="M4,Mv", output_dir=STAGING / "models_belief")
    )
    assert config.output_dir == STAGING / "models_belief"


def test_subset_may_not_target_another_configurations_canonical_bundle() -> None:
    """The guard covers every canonical bundle, not just this invocation's.

    A patch run pointed at ``models/``, or an SWR-subset run pointed at the
    unrestricted ``models/``, would replace part of a bundle the run does not
    even resolve to.
    """
    with pytest.raises(ValueError, match="canonical models/ bundle"):
        mod._resolve_run_config(
            _args(
                hierarchy="patch",
                models="P3,Pv",
                output_dir=CONTENT_IN / "models",
            )
        )
    with pytest.raises(ValueError, match="canonical models/ bundle"):
        mod._resolve_run_config(
            _args(
                swr_subset="T",
                models="M0,M1",
                output_dir=CONTENT_IN / "models",
            )
        )


def test_canonical_bundle_names_cover_every_suffix_variant() -> None:
    names = fit_outputs.canonical_output_subdirs()
    assert {"models", "models_patch"} <= names
    assert {"models_swr_T", "models_patch_swr_F"} <= names
    assert {"models_event_grain", "models_swr_T_event_grain"} <= names
    assert {"models_swr_HPC", "models_patch_mua_OFC"} <= names
    # Every name a full run of some configuration resolves to is in the set.
    for hierarchy in ("arm", "patch"):
        for swr_subset in ("all", "T"):
            for grain in ("timebin", "event"):
                config = mod._resolve_run_config(
                    _args(hierarchy=hierarchy, swr_subset=swr_subset, grain=grain)
                )
                assert config.output_dir.name in names


def test_swr_subset_run_keeps_its_own_suffixed_bundle() -> None:
    config = mod._resolve_run_config(_args(swr_subset="T", grain="event"))
    assert config.output_dir == CONTENT_IN / "models_swr_T_event_grain"


# ── Response selection ──


def _canonical_arm_table(
    *,
    n_events: int = 4,
    fraction_scale: float = 1.0,
    undecoded_event: str | None = None,
    partial_event: str | None = None,
) -> pd.DataFrame:
    """A minimal table satisfying the canonical arm contract.

    ``fraction_scale`` rescales every event's fractions (1.0 is the contract).
    ``undecoded_event`` leaves one event's fraction missing on every candidate,
    as an event with no real-arm bins; ``partial_event`` leaves it missing on
    one candidate only, which the contract forbids.
    """
    rows = []
    for index in range(n_events):
        event = f"e{index}"
        current = index % len(WELL_ARMS)
        other = (current + 1) % len(WELL_ARMS)
        counts = np.zeros(len(ALL_ARMS))
        counts[current] = 6.0
        counts[other] = 2.0
        fractions = counts / counts.sum() * fraction_scale
        for arm in ALL_ARMS:
            fraction = fractions[arm]
            if event == undecoded_event or (
                event == partial_event and arm == len(ALL_ARMS) - 1
            ):
                fraction = np.nan
            rows.append(
                {
                    "animal": "chimi",
                    "session_date": "20200212",
                    "epoch": "02_r1",
                    "event_id": event,
                    "candidate_arm": arm,
                    "arm_count_hard": counts[arm],
                    "arm_fraction": fraction,
                    "occupancy_state": "at_well",
                    "at_well": True,
                    "occupancy_well_id": current,
                    "occupancy_well_arm": current,
                    "is_current_arm": float(arm == current),
                    "visit_recency_trials_event_leaf_centered": float(arm),
                    "is_candidate_arm_never_visited": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _canonical_patch_table(*, n_events: int = 4) -> pd.DataFrame:
    """A minimal table satisfying the canonical patch contract."""
    patches = sorted(set(TRACK_SEGMENT_TO_PATCH.values()))
    rows = []
    for index in range(n_events):
        current_arm = index % len(WELL_ARMS)
        current_patch = TRACK_SEGMENT_TO_PATCH[current_arm]
        counts = dict.fromkeys(patches, 0.0)
        counts[current_patch] = 5.0
        counts[patches[(patches.index(current_patch) + 1) % len(patches)]] = 3.0
        total = sum(counts.values())
        for patch in patches:
            rows.append(
                {
                    "animal": "chimi",
                    "session_date": "20200212",
                    "epoch": "02_r1",
                    "event_id": f"e{index}",
                    "patch": patch,
                    "patch_count_hard": counts[patch],
                    "patch_fraction": counts[patch] / total,
                    "occupancy_state": "at_well",
                    "at_well": True,
                    "occupancy_well_id": current_arm,
                    "occupancy_well_arm": current_arm,
                    "is_current_patch": float(patch == current_patch),
                    "visit_recency_trials_patch": float(patch),
                    "is_patch_never_visited": 0.0,
                }
            )
    return pd.DataFrame(rows)


_ARM_LADDER = {"X": ModelSpec("X", ("is_current_arm",), "current-arm bias", None)}
_PATCH_LADDER = {"X": ModelSpec("X", ("is_current_patch",), "current patch", None)}


def _prepare_arm(table: pd.DataFrame, **kwargs):
    return mod.prepare_canonical_fit_request(
        table,
        hierarchy=_ARM_LADDER,
        model_names=["X"],
        hierarchy_kind="arm",
        table_label="synthetic-arm-table",
        **kwargs,
    )


def test_default_response_is_the_per_event_arm_fraction() -> None:
    _prepared, _models, _preds, response_col, event_col = _prepare_arm(
        _canonical_arm_table()
    )
    assert response_col == "arm_fraction"
    assert event_col == "event_id"


def test_hard_count_response_resolves_to_the_arm_counts() -> None:
    _prepared, _models, _preds, response_col, _event = _prepare_arm(
        _canonical_arm_table(), response="hard_count"
    )
    assert response_col == "arm_count_hard"


def test_patch_hierarchy_resolves_to_the_patch_fraction_and_counts() -> None:
    for response, expected in (
        ("fraction", "patch_fraction"),
        ("hard_count", "patch_count_hard"),
    ):
        _prepared, _models, _preds, response_col, _event = (
            mod.prepare_canonical_fit_request(
                _canonical_patch_table(),
                hierarchy=_PATCH_LADDER,
                model_names=["X"],
                hierarchy_kind="patch",
                table_label="synthetic-patch-table",
                response=response,
            )
        )
        assert response_col == expected


def test_fraction_that_does_not_sum_to_one_is_refused() -> None:
    table = _canonical_arm_table(fraction_scale=0.9)
    with pytest.raises(ValueError, match="does not sum to 1.0"):
        _prepare_arm(table)
    # The same table is fine on the counts, which are untouched.
    assert _prepare_arm(table, response="hard_count")[3] == "arm_count_hard"


def test_fraction_off_by_more_than_the_tolerance_is_refused() -> None:
    table = _canonical_arm_table()
    table.loc[0, "arm_fraction"] += 1e-8
    with pytest.raises(ValueError, match="does not sum to 1.0"):
        _prepare_arm(table)


def test_fraction_within_the_tolerance_is_accepted() -> None:
    table = _canonical_arm_table()
    table.loc[0, "arm_fraction"] += 1e-12
    assert _prepare_arm(table)[3] == "arm_fraction"


def test_missing_fraction_column_names_the_column_and_the_way_out() -> None:
    table = _canonical_arm_table().drop(columns=["arm_fraction"])
    with pytest.raises(ValueError, match="arm_fraction.*--response hard_count"):
        _prepare_arm(table)


def test_event_with_no_real_arm_bins_is_allowed_as_all_missing() -> None:
    """An all-junction event carries no fraction anywhere; the cohort drops it."""
    assert _prepare_arm(_canonical_arm_table(undecoded_event="e2"))[3] == "arm_fraction"


def test_fraction_defined_on_only_some_candidates_is_refused() -> None:
    with pytest.raises(ValueError, match="only some candidates"):
        _prepare_arm(_canonical_arm_table(partial_event="e1"))


def test_undecoded_events_are_reported_as_zero_response_drops() -> None:
    """The audit must name the mechanism: no response, not a missing predictor."""
    table = _canonical_arm_table(n_events=5, undecoded_event="e3")
    buckets = event_drop_buckets(
        table,
        event_col="event_id",
        response_col="arm_fraction",
        own_predictors=["is_current_arm"],
        union_predictors=["is_current_arm"],
    )
    assert buckets["n_drop_zero_response"] == 1
    assert buckets["n_drop_own_predictor_missing"] == 0
    assert buckets["dropped_event_ids"] == {"e3"}


def test_hard_count_runs_publish_to_their_own_bundle() -> None:
    config = mod._resolve_run_config(_args(response="hard_count"))
    assert config.output_dir == CONTENT_IN / "models_hard_count"
    assert "models_hard_count" in fit_outputs.canonical_output_subdirs()
    assert "models_patch_hard_count" in fit_outputs.canonical_output_subdirs()


def test_fraction_runs_keep_the_unsuffixed_canonical_bundle() -> None:
    assert (
        fit_outputs.canonical_output_subdir("non_local", "HPC", "arm", "all", "timebin")
        == "models"
    )
    assert (
        fit_outputs.canonical_output_subdir(
            "non_local", "HPC", "arm", "all", "timebin", "fraction"
        )
        == "models"
    )


def test_parser_leaves_the_response_to_the_estimator_by_default() -> None:
    parser = mod._build_arg_parser()
    assert parser.parse_args([]).response is None
    assert parser.parse_args(["--response", "hard_count"]).response == "hard_count"
    assert mod._resolve_run_config(_args(response=None)).response == "fraction"


# ── Estimator and response recorded on every fitted row ──


def _simulated_session(response_col: str, seed: int = 5) -> pd.DataFrame:
    df = simulate_with_event_covariates(
        {"is_current_arm": 0.7}, n_events=60, L_e=20, seed=seed
    )
    df["arm_fraction"] = df["arm_count_hard"] / df.groupby("event_id")[
        "arm_count_hard"
    ].transform("sum")
    df["animal"] = "chimi"
    df["session_date"] = "20200212"
    df["epoch"] = "02_r1"
    assert response_col in df.columns
    return df


@pytest.mark.parametrize(
    "response_col,estimator",
    [
        ("arm_fraction", "conditional_poisson_fraction"),
        ("arm_count_hard", "conditional_poisson_count"),
    ],
)
def test_fitted_rows_record_the_estimator_and_the_response(
    response_col, estimator
) -> None:
    result = mod.fit_model_per_session(
        _simulated_session(response_col),
        "X",
        ["is_current_arm"],
        response_col=response_col,
    )
    assert len(result) == 1
    assert result["estimator"].tolist() == [estimator]
    assert result["response_col"].tolist() == [response_col]


def test_fitted_rows_carry_the_optimizer_diagnostics() -> None:
    result = mod.fit_model_per_session(
        _simulated_session("arm_fraction"),
        "X",
        ["is_current_arm"],
        response_col="arm_fraction",
    )
    row = result.iloc[0]
    assert row["llf"] < 0
    assert row["newton_decrement"] < 1e-8
    assert row["max_abs_grad"] < 1e-3
    assert row["n_newton_steps"] >= 0
    assert row["unidentified_terms"] == ""


def test_the_printed_se_skips_a_session_that_cannot_inform_the_term(capsys) -> None:
    """The fitter reports such a term with a NaN coefficient and an infinite SE.

    The session drops out of that term's summary, SE included, rather than
    pulling the median SE toward infinity.
    """
    mdf = pd.DataFrame(
        {
            "animal": ["a", "a", "b", "b", "c"],
            "coef_is_candidate_arm_0": [0.5, 0.6, np.nan, np.nan, 0.7],
            "bse_is_candidate_arm_0": [0.1, 0.2, np.inf, np.inf, 0.3],
            "unidentified_terms": [
                "",
                "",
                "is_candidate_arm_0:separated_to_minus_inf",
                "is_candidate_arm_0:no_within_event_variation",
                "",
            ],
        }
    )
    hierarchy = {"X": argparse.Namespace(description="test rung")}

    mod.print_model_results("X", mdf, ["is_candidate_arm_0"], hierarchy=hierarchy)

    line = next(
        text
        for text in capsys.readouterr().out.splitlines()
        if text.strip().startswith("is_candidate_arm_0")
    )
    # Columns: name, animal median, agree, session median, SE, p, unidentified.
    assert float(line.split()[4]) == pytest.approx(0.2)
    # The sessions that cannot inform the term are reported, not just skipped.
    assert line.split()[6] == "2"


# ── Event-scale reporting ──


def _comparison_row(estimator: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "parent": "M0",
                "child": "M1",
                "estimator": estimator,
                "n_sessions": 200,
                "delta_aic_median": 12.5,
                "delta_aic_animal_median": 11.0,
                "n_animals": 5,
                "n_animals_child_better": 5,
                "child_better_aic": 180,
                "lrt_significant": 175,
                "wilcoxon_p": 1e-9,
            }
        ]
    )


def test_count_run_headline_omits_the_per_session_win_counts() -> None:
    lines = mod.format_comparison_table(_comparison_row("conditional_poisson_count"))
    text = "\n".join(lines)

    assert "timebin-scale" in text
    assert "Child wins" not in text
    assert "LRT sig" not in text
    assert "180" not in text and "175" not in text
    assert "12.5" in text
    assert "model_comparison.pkl" in text


def test_fraction_run_headline_labels_the_win_counts_event_scale() -> None:
    lines = mod.format_comparison_table(_comparison_row("conditional_poisson_fraction"))
    text = "\n".join(lines)

    assert "event-scale (quasi)" in text
    assert "Child wins" in text and "LRT sig" in text
    assert "180/200" in text and "175/200" in text


def test_logit_run_headline_is_also_event_scale() -> None:
    text = "\n".join(
        mod.format_comparison_table(_comparison_row("conditional_logit_dominant"))
    )
    assert "event-scale (quasi)" in text
    assert "Child wins" in text


def test_scale_label_refuses_to_claim_a_scale_it_does_not_know() -> None:
    assert fit_outputs.scale_label(None) == "unlabelled scale"
    assert fit_outputs.scale_label("some_new_estimator") == "unlabelled scale"


# ── Provenance ──


def test_fit_inputs_name_the_table_the_ladder_reads() -> None:
    inputs = mod.content_fit_inputs(CONTENT_IN, "event_arm_table")
    assert inputs == {"event_arm_table": CONTENT_IN / "event_arm_table.parquet"}


def test_bundle_writes_a_provenance_sidecar_beside_every_pickle(tmp_path) -> None:
    table = tmp_path / "event_arm_table.parquet"
    write_table(_canonical_arm_table(), table)
    fit_inputs = record_inputs({"event_arm_table": table})
    frame = pd.DataFrame({"model": ["X"], "coef_is_current_arm": [0.5]})
    output_dir = tmp_path / "models"

    mod._write_complete_output_bundle(
        output_dir=output_dir,
        model_results={"X": frame},
        combined=frame,
        cohort_audits=[],
        comparison=frame,
        logit_results={"X": frame},
        fit_inputs=fit_inputs,
        estimator="conditional_poisson_fraction",
        cli=["fit_replay_content_models.py", "--response", "fraction"],
    )

    for artifact in (
        output_dir / "X.pkl",
        output_dir / "all_models.pkl",
        output_dir / "model_comparison.pkl",
        output_dir / "logit" / "X_logit.pkl",
    ):
        assert check_fit_provenance(artifact, {"event_arm_table": table}) == []
    record = json.loads((output_dir / "all_models.provenance.json").read_text())
    assert record["estimator"] == "conditional_poisson_fraction"
    assert record["cli"][-1] == "fraction"


def test_zero_response_events_cover_undefined_and_empty_allocations() -> None:
    """Both mechanisms are the same drop: the event has no allocation to fit."""
    frame = pd.DataFrame(
        {
            "event_id": ["a", "a", "b", "b", "c", "c"],
            "arm_fraction": [0.5, 0.5, np.nan, np.nan, 0.0, 0.0],
        }
    )

    assert mod._zero_response_event_ids(
        frame, event_col="event_id", response_col="arm_fraction"
    ) == {"b", "c"}


# ── Leaf-only candidate set ──


def test_leaf_only_run_publishes_to_its_own_bundle() -> None:
    """A six-arm fit is a different likelihood, so it gets a different bundle."""
    config = mod._resolve_run_config(_args(leaf_only=True))
    assert config.output_dir == CONTENT_IN / "models_leaf"
    assert "models_leaf" in fit_outputs.canonical_output_subdirs()
    assert "models_swr_T_leaf" in fit_outputs.canonical_output_subdirs()
    assert "models_hard_count_leaf" in fit_outputs.canonical_output_subdirs()


def test_leaf_only_fits_six_candidates_not_nine() -> None:
    assert mod.fit_n_categories("arm", leaf_only=True) == 6
    assert mod.fit_n_categories("arm", leaf_only=False) == 9
    assert mod.fit_n_categories("patch", leaf_only=False) == 3


def test_leaf_only_drops_the_trunk_arm_fixed_effects_from_every_rung() -> None:
    """Arms 6 and 7 are identically zero on leaves; the ladder must not carry them."""
    for name, spec in mod.MODEL_HIERARCHY.items():
        predictors = model_specs.rung_predictors(spec, leaf_only=True)
        assert "is_candidate_arm_5" not in predictors, name
        assert "is_candidate_arm_6" not in predictors, name
        assert "is_candidate_arm_7" not in predictors, name
        # Everything that is not a dropped arm dummy survives, in order.
        assert predictors == [
            p
            for p in spec.predictors
            if p
            not in {
                "is_candidate_arm_5",
                "is_candidate_arm_6",
                "is_candidate_arm_7",
            }
        ], name
        assert model_specs.rung_predictors(spec, leaf_only=False) == list(
            spec.predictors
        )

    arm_fe = model_specs.rung_predictors(mod.MODEL_HIERARCHY["M0a"], leaf_only=True)
    assert arm_fe == [f"is_candidate_arm_{i}" for i in range(5)]


def test_leaf_only_cohort_keeps_the_six_leaf_rows_of_every_event() -> None:
    table = _canonical_arm_table(n_events=3)
    leaves = model_specs.restrict_to_leaf_arms(table)

    assert sorted(leaves["candidate_arm"].unique()) == [0, 1, 2, 3, 4, 5]
    assert leaves.groupby("event_id").size().eq(6).all()
    assert len(leaves) == 18
    # The trunk mass is dropped, not redistributed: every retained row keeps its
    # nine-arm fraction.
    merged = leaves.merge(
        table, on=["event_id", "candidate_arm"], suffixes=("_leaf", "_all")
    )
    assert (merged["arm_fraction_leaf"] == merged["arm_fraction_all"]).all()


def test_leaf_only_cohort_needs_the_candidate_arm_column() -> None:
    with pytest.raises(ValueError, match="candidate_arm"):
        model_specs.restrict_to_leaf_arms(
            _canonical_arm_table().drop(columns=["candidate_arm"])
        )


def test_leaf_only_is_refused_on_the_patch_hierarchy() -> None:
    """Patches are not arms; there is no leaf subset of the three patches."""
    with pytest.raises(ValueError, match="--leaf-only"):
        mod._resolve_run_config(_args(hierarchy="patch", leaf_only=True))
    with pytest.raises(ValueError, match="leaf"):
        fit_outputs.canonical_output_subdir(
            "non_local", "HPC", "patch", "all", "timebin", "fraction", leaf_only=True
        )


def test_leaf_only_subset_may_not_target_a_canonical_bundle() -> None:
    with pytest.raises(ValueError, match="canonical models_leaf/ bundle"):
        mod._resolve_run_config(
            _args(
                leaf_only=True,
                models="M4,Mv",
                output_dir=CONTENT_IN / "models_leaf",
            )
        )


def test_leaf_only_subset_run_is_keyed_off_the_leaf_directory() -> None:
    config = mod._resolve_run_config(
        _args(leaf_only=True, models="M0,M0a,M1,M2,M3,M4,Mv,Mu,Mvu")
    )
    assert config.output_dir == CONTENT_IN / "models_leaf_M0-M0a-M1-M2-M3-M4-Mv-Mu-Mvu"


def test_parser_defaults_to_the_nine_arm_candidate_set() -> None:
    parser = mod._build_arg_parser()
    assert parser.parse_args([]).leaf_only is False
    assert parser.parse_args(["--leaf-only"]).leaf_only is True


# ── Leaf-only response renormalization ──


def _table_with_trunk_mass(trunk_only_event: str | None = None) -> pd.DataFrame:
    """Three events; each puts a third of its mass on the trunk arms.

    ``trunk_only_event`` puts *all* of one event's mass on the trunks, leaving
    it nothing on the leaves.
    """
    trunks = [arm for arm in ALL_ARMS if arm >= len(WELL_ARMS)]
    rows = []
    for index in range(3):
        event = f"e{index}"
        current = index % len(WELL_ARMS)
        counts = np.zeros(len(ALL_ARMS))
        if event == trunk_only_event:
            counts[trunks] = 4.0
        else:
            counts[current] = 6.0
            counts[(current + 1) % len(WELL_ARMS)] = 2.0
            counts[trunks[0]] = 4.0
        fractions = counts / counts.sum()
        for arm in ALL_ARMS:
            rows.append(
                {
                    "animal": "chimi",
                    "session_date": "20200212",
                    "epoch": "02_r1",
                    "event_id": event,
                    "candidate_arm": arm,
                    "arm_count_hard": counts[arm],
                    "arm_fraction": fractions[arm],
                    "is_current_arm": float(arm == current),
                }
            )
    return pd.DataFrame(rows)


def test_leaf_fractions_are_rescaled_to_unit_mass_per_event() -> None:
    """Without this an event's weight is its leaf share, not one observation."""
    table = _table_with_trunk_mass()
    leaves, renormalized = model_specs.leaf_only_cohort(
        table, response_col="arm_fraction", response="fraction"
    )

    assert renormalized is True
    sums = leaves.groupby("event_id")["arm_fraction"].sum()
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-12)
    # e0 held 6/12 and 2/12 on two leaves and 4/12 on a trunk; over the leaves
    # that is 0.75 and 0.25.
    e0 = leaves[leaves["event_id"].eq("e0")].set_index("candidate_arm")["arm_fraction"]
    assert e0[0] == pytest.approx(0.75)
    assert e0[1] == pytest.approx(0.25)
    assert e0[2] == pytest.approx(0.0)


def test_leaf_only_count_response_keeps_its_timebin_counts() -> None:
    table = _table_with_trunk_mass()
    leaves, renormalized = model_specs.leaf_only_cohort(
        table, response_col="arm_count_hard", response="hard_count"
    )

    assert renormalized is False
    expected = table[table["candidate_arm"] < 6]["arm_count_hard"].to_numpy()
    assert np.array_equal(leaves["arm_count_hard"].to_numpy(), expected)
    assert leaves.groupby("event_id")["arm_count_hard"].sum().eq(8.0).all()


def test_event_with_only_trunk_mass_becomes_a_zero_response_drop() -> None:
    """It has no leaf allocation to fit, and the audit must say so."""
    leaves, _renormalized = model_specs.leaf_only_cohort(
        _table_with_trunk_mass(trunk_only_event="e1"),
        response_col="arm_fraction",
        response="fraction",
    )

    e1 = leaves[leaves["event_id"].eq("e1")]["arm_fraction"]
    assert e1.isna().all()
    assert mod._zero_response_event_ids(
        leaves, event_col="event_id", response_col="arm_fraction"
    ) == {"e1"}

    buckets = event_drop_buckets(
        leaves,
        event_col="event_id",
        response_col="arm_fraction",
        own_predictors=["is_current_arm"],
        union_predictors=["is_current_arm"],
    )
    assert buckets["n_drop_zero_response"] == 1
    assert buckets["dropped_event_ids"] == {"e1"}
    # The other two events keep unit mass.
    kept = leaves[~leaves["event_id"].eq("e1")]
    assert np.allclose(kept.groupby("event_id")["arm_fraction"].sum(), 1.0, atol=1e-12)


def test_partially_defined_event_has_no_total_to_rescale_by() -> None:
    table = _table_with_trunk_mass()
    table.loc[
        table["event_id"].eq("e0") & table["candidate_arm"].eq(2), "arm_fraction"
    ] = np.nan

    with pytest.raises(ValueError, match="not summing to 1.0"):
        model_specs.leaf_only_cohort(
            table, response_col="arm_fraction", response="fraction"
        )


def test_renormalized_leaf_bundles_say_so_in_their_provenance(tmp_path) -> None:
    table = tmp_path / "event_arm_table.parquet"
    write_table(_canonical_arm_table(), table)
    frame = pd.DataFrame({"model": ["X"], "coef_is_current_arm": [0.5]})
    output_dir = tmp_path / "models_leaf"

    mod._write_complete_output_bundle(
        output_dir=output_dir,
        model_results={"X": frame},
        combined=frame,
        cohort_audits=[],
        comparison=pd.DataFrame(),
        logit_results={},
        fit_inputs=record_inputs({"event_arm_table": table}),
        estimator="conditional_poisson_fraction",
        cli=["fit_replay_content_models.py", "--leaf-only"],
        response_renormalized=True,
    )

    record = json.loads((output_dir / "all_models.provenance.json").read_text())
    assert record["response_renormalized"] is True


def test_leaf_only_ladder_fits_six_categories_on_the_rescaled_response() -> None:
    """End to end on simulated data: six-category null, and the rows say so."""
    sim = _simulated_session("arm_fraction").rename(columns={"arm_id": "candidate_arm"})
    long_df, renormalized = model_specs.leaf_only_cohort(
        sim, response_col="arm_fraction", response="fraction"
    )
    complete = long_df.groupby("event_id")["arm_fraction"].transform(
        lambda values: values.notna().all()
    )
    long_df = long_df[complete].reset_index(drop=True)

    config = mod._RunConfig(
        hierarchy=_ARM_LADDER,
        input_dir=CONTENT_IN,
        output_dir=CONTENT_IN / "models_leaf",
        table_name="event_arm_table",
        model_names=["X"],
    )
    cohort = mod._ContentCohort(
        long_df=long_df,
        long_df_prefilter=sim,
        valid_model_names=["X"],
        union_predictors=["is_current_arm"],
        response_col="arm_fraction",
        event_col="event_id",
        n_events_before_complete_case=int(sim["event_id"].nunique()),
        fit_inputs={},
        response_renormalized=renormalized,
    )

    results, _skips = mod._fit_content_ladder(
        _args(leaf_only=True, cv=False), config, cohort
    )

    fitted = results["X"]
    assert fitted["response_renormalized"].tolist() == [True]
    n_events = int(fitted["n_events"].iloc[0])
    # Unit mass per event against a six-category uniform null.
    assert fitted["llf_null"].iloc[0] == pytest.approx(-n_events * np.log(6))


# ── Dirichlet-multinomial estimator ──


def test_dm_bundle_is_suffixed_last_and_named_in_the_canonical_set() -> None:
    """The overdispersed ladder never publishes over the conditional bundle."""
    assert (
        fit_outputs.canonical_output_subdir(
            "non_local",
            "HPC",
            "arm",
            "all",
            "timebin",
            "hard_count",
            False,
            "dirichlet_multinomial",
        )
        == "models_dm"
    )
    # ``_dm`` is appended after every other suffix, so stripping it names the
    # conditional bundle this fit is the counterpart of.
    assert (
        fit_outputs.canonical_output_subdir(
            "non_local",
            "HPC",
            "arm",
            "T",
            "timebin",
            "hard_count",
            True,
            "dirichlet_multinomial",
        )
        == "models_swr_T_leaf_dm"
    )
    subdirs = fit_outputs.canonical_output_subdirs()
    assert {"models", "models_hard_count", "models_dm", "models_leaf_dm"} <= subdirs
    assert "models_patch_dm" in subdirs
    # The count response is the estimator's own, so it is not named twice.
    assert "models_hard_count_dm" not in subdirs


def test_dm_run_resolves_the_count_response_and_its_own_directory() -> None:
    config = mod._resolve_run_config(
        _args(estimator="dirichlet_multinomial", response=None)
    )
    assert config.response == "hard_count"
    assert config.output_dir == CONTENT_IN / "models_dm"


def test_dm_refuses_the_fraction_response_naming_both_flags() -> None:
    with pytest.raises(ValueError) as excinfo:
        mod._resolve_run_config(
            _args(estimator="dirichlet_multinomial", response="fraction")
        )
    message = str(excinfo.value)
    assert "--response fraction" in message
    assert "--estimator dirichlet_multinomial" in message


def test_dm_parser_defaults() -> None:
    parser = mod._build_arg_parser()
    args = parser.parse_args([])
    assert args.estimator == "conditional_poisson"
    assert args.n_jobs == mod.DEFAULT_N_JOBS
    assert args.min_events_session_phi == 40
    assert (
        parser.parse_args(["--estimator", "dirichlet_multinomial"]).estimator
        == "dirichlet_multinomial"
    )


def test_dm_numbers_are_labelled_a_full_event_likelihood() -> None:
    assert (
        fit_outputs.scale_label("dirichlet_multinomial")
        == "event-scale (full likelihood)"
    )


def _dm_session(
    n_events: int, *, seed: int, animal: str = "chimi", epoch: str = "02_r1"
) -> pd.DataFrame:
    df = simulate_dirichlet_multinomial_counts(
        {"is_current_arm": 0.8},
        log_phi_intercept=1.2,
        n_events=n_events,
        L_e=20,
        seed=seed,
    )
    df["animal"] = animal
    df["session_date"] = "20200212"
    df["epoch"] = epoch
    # Event ids are globally unique; the simulator numbers each session from 0.
    df["event_id"] = [f"{epoch}-{index}" for index in df["event_id"]]
    # The simulator leaves its drawn allocations in attrs, which pandas compares
    # element-wise (and so raises on) when concatenating two sessions.
    df.attrs.clear()
    return df


def test_short_dm_session_borrows_its_animals_concentration() -> None:
    """A 20-event session cannot pin down its own concentration; a 60-event one can."""
    cohort = pd.concat(
        [
            _dm_session(60, seed=0, epoch="02_r1"),
            _dm_session(20, seed=1, epoch="04_r2"),
        ],
        ignore_index=True,
    )

    result = mod.fit_model_per_session(
        cohort,
        "X",
        ["is_current_arm"],
        response_col="arm_count_hard",
        estimator="dirichlet_multinomial",
        n_jobs=2,
        min_events_session_phi=40,
    )

    assert result["estimator"].tolist() == ["dirichlet_multinomial"] * 2
    assert result["response_col"].tolist() == ["arm_count_hard"] * 2
    rows = result.set_index("epoch")
    assert rows.loc["02_r1", "n_events"] == 60
    assert rows.loc["04_r2", "n_events"] == 20
    assert rows.loc["02_r1", "phi_source"] == "session"
    assert rows.loc["04_r2", "phi_source"] == "animal_pooled"
    assert rows.loc["04_r2", "coef_log_phi_intercept"] == pytest.approx(
        rows.loc["02_r1", "coef_log_phi_intercept"]
    )
    # The concentration modifier is estimated because the cohort carries both
    # kinds of event.
    assert "coef_log_phi_is_swr" in result.columns
    assert result["coef_log_phi_is_swr"].notna().all()
    assert bool(result["llf_includes_constants"].all())


def test_dm_concentration_modifier_is_dropped_within_an_swr_subset() -> None:
    """Inside a subset ``is_swr`` is constant, so there is no contrast to fit."""
    cohort = _dm_session(60, seed=0)
    assert mod.concentration_cols_for_cohort(cohort) == ("is_swr",)
    subset = cohort[cohort["is_swr"] == 1.0]
    assert mod.concentration_cols_for_cohort(subset) == ()
    assert mod.concentration_cols_for_cohort(cohort.drop(columns="is_swr")) == ()


def test_dm_session_with_a_refused_design_is_an_audited_skip() -> None:
    """A predictor constant within every event raises; the ladder books it instead."""
    good = _dm_session(60, seed=0, epoch="02_r1")
    bad = _dm_session(60, seed=2, epoch="04_r2").assign(is_prev_arm=0.0)
    cohort = pd.concat([good, bad], ignore_index=True)

    result = mod.fit_model_per_session(
        cohort,
        "X",
        ["is_current_arm", "is_prev_arm"],
        response_col="arm_count_hard",
        estimator="dirichlet_multinomial",
        n_jobs=2,
        min_events_session_phi=40,
    )

    assert result["epoch"].tolist() == ["02_r1"]
    skips = result.attrs["estimator_skips"]
    assert [skip["session"] for skip in skips] == ["chimi/20200212/04_r2"]
    assert skips[0]["reason"].startswith("dm_design:")


def test_a_refused_dm_session_is_named_in_the_published_cohort_audit(tmp_path) -> None:
    """The skip channel's consumer: the audit says which session, and why.

    A session the estimator refuses is not a dropped event -- its events are
    already counted under ``n_excluded_session`` -- so the only place a reader
    can learn that a session is missing from the fitted cohort, and for what
    reason, is the audit's ``drop_notes``.
    """
    good = _dm_session(60, seed=0, epoch="02_r1")
    bad = _dm_session(60, seed=2, epoch="04_r2").assign(is_prev_arm=0.0)
    cohort_df = pd.concat([good, bad], ignore_index=True)
    ladder = {"X": ModelSpec("X", ("is_current_arm", "is_prev_arm"), "two arms", None)}
    args = _args(estimator="dirichlet_multinomial", response="hard_count", n_jobs=2)
    args.logit = False
    config = mod._RunConfig(
        hierarchy=ladder,
        input_dir=tmp_path,
        output_dir=tmp_path / "models_dm",
        table_name="event_arm_table",
        model_names=["X"],
        response="hard_count",
    )
    cohort = mod._ContentCohort(
        long_df=cohort_df,
        long_df_prefilter=cohort_df,
        valid_model_names=["X"],
        union_predictors=["is_current_arm", "is_prev_arm"],
        response_col="arm_count_hard",
        event_col="event_id",
        n_events_before_complete_case=int(cohort_df["event_id"].nunique()),
        fit_inputs={},
        response_renormalized=False,
    )

    model_results, estimator_skips = mod._fit_content_ladder(args, config, cohort)
    mod._audit_compare_and_publish(
        args, config, cohort, model_results, estimator_skips, cli=["fit.py"]
    )

    # The refused session contributes no row to the fit frame.
    assert model_results["X"]["epoch"].tolist() == ["02_r1"]

    audit = pd.read_csv(config.output_dir / "all_models_cohort_audit.csv")
    assert len(audit) == 1
    notes = str(audit["drop_notes"].iloc[0])
    assert "estimator skipped chimi/20200212/04_r2" in notes
    assert "dm_design:" in notes
    # Its events are excluded at the session level, not booked as event drops.
    assert int(audit["n_drop_own_predictor_missing"].iloc[0]) == 0
    assert int(audit["n_excluded_session"].iloc[0]) == 60


# ── Conditional-logit robustness: a refused session is skipped, not fatal ──


def _logit_session(epoch: str, seed: int) -> pd.DataFrame:
    df = simulate_with_event_covariates(
        {"is_current_arm": 0.7, "is_prev_arm": -0.3}, n_events=60, L_e=20, seed=seed
    )
    df["arm_fraction"] = df["arm_count_hard"] / df.groupby("event_id")[
        "arm_count_hard"
    ].transform("sum")
    df["event_id"] = epoch + "_" + df["event_id"].astype(str)
    df["animal"] = "chimi"
    df["session_date"] = "20200212"
    df["epoch"] = epoch
    return df


_LOGIT_LADDER = {
    "X": ModelSpec("X", ("is_current_arm",), "current", None),
    "Y": ModelSpec("Y", ("is_current_arm", "is_prev_arm"), "current + prev", "X"),
}


def _logit_fixture(tmp_path):
    cohort_df = pd.concat(
        [_logit_session("02_r1", seed=0), _logit_session("04_r2", seed=1)],
        ignore_index=True,
    )
    args = _args(logit=True)
    config = mod._RunConfig(
        hierarchy=_LOGIT_LADDER,
        input_dir=tmp_path,
        output_dir=tmp_path / "models",
        table_name="event_arm_table",
        model_names=["X", "Y"],
    )
    cohort = mod._ContentCohort(
        long_df=cohort_df,
        long_df_prefilter=cohort_df,
        valid_model_names=["X", "Y"],
        union_predictors=["is_current_arm", "is_prev_arm"],
        response_col="arm_fraction",
        event_col="event_id",
        n_events_before_complete_case=int(cohort_df["event_id"].nunique()),
        fit_inputs={},
        response_renormalized=False,
    )
    return args, config, cohort


def _fail_logit_for(monkeypatch, epoch: str, n_predictors: int, error: Exception):
    """Make ConditionalLogit raise ``error`` for one session at one rung."""
    from continuum_swr_replay.analysis.replay import fitting

    real = fitting.ConditionalLogit

    def picky(y, X, groups):
        if X.shape[1] == n_predictors and str(groups[0]).startswith(epoch):
            raise error
        return real(y, X, groups=groups)

    monkeypatch.setattr(fitting, "ConditionalLogit", picky)


def test_a_session_the_logit_fit_refuses_is_skipped_and_recorded(
    tmp_path, monkeypatch
) -> None:
    args, config, cohort = _logit_fixture(tmp_path)
    model_results, _ = mod._fit_content_ladder(args, config, cohort)
    _fail_logit_for(
        monkeypatch,
        "04_r2",
        2,
        ValueError("need covariance of parameters for computing"),
    )

    logit, skips = mod._fit_conditional_logit_robustness(
        args, config, cohort, model_results
    )

    assert skips.to_dict("records") == [
        {
            "model": "Y",
            "animal": "chimi",
            "session_date": "20200212",
            "epoch": "04_r2",
            "reason": "ValueError: need covariance of parameters for computing",
        }
    ]
    # X fitted 04_r2, but the session leaves every logit rung so the rungs
    # stay on one common set -- a subset of the primary ladder's sessions.
    assert set(logit) == {"X", "Y"}
    for frame in logit.values():
        assert frame["epoch"].tolist() == ["02_r1"]
    assert set(model_results["X"]["epoch"]) == {"02_r1", "04_r2"}


def test_an_unexpected_logit_error_still_propagates(tmp_path, monkeypatch) -> None:
    args, config, cohort = _logit_fixture(tmp_path)
    model_results, _ = mod._fit_content_ladder(args, config, cohort)
    _fail_logit_for(monkeypatch, "04_r2", 2, KeyError("a coding bug"))

    with pytest.raises(KeyError, match="a coding bug"):
        mod._fit_conditional_logit_robustness(args, config, cohort, model_results)


def test_logit_skips_are_published_beside_the_logit_fits(tmp_path, monkeypatch) -> None:
    args, config, cohort = _logit_fixture(tmp_path)
    model_results, skips = mod._fit_content_ladder(args, config, cohort)
    _fail_logit_for(monkeypatch, "04_r2", 2, np.linalg.LinAlgError("singular"))

    mod._audit_compare_and_publish(
        args, config, cohort, model_results, skips, cli=["fit.py"]
    )

    logit_dir = config.output_dir / "logit"
    published = pd.read_csv(logit_dir / "logit_skipped_sessions.csv", dtype=str)
    assert published.to_dict("records") == [
        {
            "model": "Y",
            "animal": "chimi",
            "session_date": "20200212",
            "epoch": "04_r2",
            "reason": "LinAlgError: singular",
        }
    ]
    assert pd.read_pickle(logit_dir / "Y_logit.pkl")["epoch"].tolist() == ["02_r1"]
    # The primary ladder keeps both sessions.
    assert pd.read_pickle(config.output_dir / "X.pkl")["epoch"].nunique() == 2


def test_a_logit_run_with_no_skips_publishes_an_empty_record(tmp_path) -> None:
    args, config, cohort = _logit_fixture(tmp_path)
    model_results, skips = mod._fit_content_ladder(args, config, cohort)

    mod._audit_compare_and_publish(
        args, config, cohort, model_results, skips, cli=["fit.py"]
    )

    published = pd.read_csv(config.output_dir / "logit" / "logit_skipped_sessions.csv")
    assert published.empty
    assert tuple(published.columns) == mod.LOGIT_SKIP_COLUMNS
