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
    simulate_with_event_covariates,
)
from continuum_swr_replay.data_loaders.constants import (
    ALL_ARMS,
    TRACK_SEGMENT_TO_PATCH,
    WELL_ARMS,
)
from continuum_swr_replay.table_io import write_table
from scripts.content import fit_replay_content_models as mod


def _args(**overrides) -> argparse.Namespace:
    defaults = {
        "event_source": "non_local",
        "hierarchy": "arm",
        "brain_area": "HPC",
        "swr_subset": "all",
        "grain": "timebin",
        "response": "fraction",
        "leaf_only": False,
        "input_dir": Path("/tmp/content-in"),
        "output_dir": None,
        "models": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_default_run_writes_the_canonical_directory() -> None:
    config = mod._resolve_run_config(_args())
    assert config.output_dir == Path("/tmp/content-in/models")
    assert list(config.model_names) == list(
        default_content_model_names(config.hierarchy)
    )


def test_explicit_full_ladder_still_writes_the_canonical_directory() -> None:
    """Naming every default rung is the canonical run, not a subset."""
    names = default_content_model_names(mod.MODEL_HIERARCHY)
    config = mod._resolve_run_config(_args(models=",".join(names)))
    assert config.output_dir == Path("/tmp/content-in/models")


def test_subset_run_is_keyed_by_its_rungs() -> None:
    config = mod._resolve_run_config(_args(models="M4,Mv,Mu,Mvu"))
    assert config.output_dir == Path("/tmp/content-in/models_M4-Mv-Mu-Mvu")
    assert list(config.model_names) == ["M4", "Mv", "Mu", "Mvu"]


def test_patch_subset_run_is_keyed_off_the_patch_directory() -> None:
    config = mod._resolve_run_config(_args(hierarchy="patch", models="P3,Pv"))
    assert config.output_dir == Path("/tmp/content-in/models_patch_P3-Pv")


def test_subset_models_do_not_write_canonical_dir() -> None:
    """An explicit --output-dir cannot aim a subset at the canonical bundle."""
    with pytest.raises(ValueError, match="models"):
        mod._resolve_run_config(
            _args(models="M4,Mv", output_dir=Path("/tmp/content-in/models"))
        )
    with pytest.raises(ValueError, match="models_patch"):
        mod._resolve_run_config(
            _args(
                hierarchy="patch",
                models="P3,Pv",
                output_dir=Path("/tmp/content-in/models_patch"),
            )
        )


def test_subset_run_may_be_sent_to_an_explicit_non_canonical_directory() -> None:
    config = mod._resolve_run_config(
        _args(models="M4,Mv", output_dir=Path("/tmp/staging/models_belief"))
    )
    assert config.output_dir == Path("/tmp/staging/models_belief")


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
                output_dir=Path("/tmp/content-in/models"),
            )
        )
    with pytest.raises(ValueError, match="canonical models/ bundle"):
        mod._resolve_run_config(
            _args(
                swr_subset="T",
                models="M0,M1",
                output_dir=Path("/tmp/content-in/models"),
            )
        )


def test_canonical_bundle_names_cover_every_suffix_variant() -> None:
    names = mod.canonical_output_subdirs()
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
    assert config.output_dir == Path("/tmp/content-in/models_swr_T_event_grain")


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
    assert config.output_dir == Path("/tmp/content-in/models_hard_count")
    assert "models_hard_count" in mod.canonical_output_subdirs()
    assert "models_patch_hard_count" in mod.canonical_output_subdirs()


def test_fraction_runs_keep_the_unsuffixed_canonical_bundle() -> None:
    assert (
        mod.canonical_output_subdir("non_local", "HPC", "arm", "all", "timebin")
        == "models"
    )
    assert (
        mod.canonical_output_subdir(
            "non_local", "HPC", "arm", "all", "timebin", "fraction"
        )
        == "models"
    )


def test_parser_defaults_to_the_fraction_response() -> None:
    parser = mod._build_arg_parser()
    assert parser.parse_args([]).response == "fraction"
    assert parser.parse_args(["--response", "hard_count"]).response == "hard_count"


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
    assert mod.scale_label(None) == "unlabelled scale"
    assert mod.scale_label("some_new_estimator") == "unlabelled scale"


# ── Provenance ──


def test_fit_inputs_name_the_table_the_ladder_reads() -> None:
    inputs = mod.content_fit_inputs(Path("/tmp/content-in"), "event_arm_table")
    assert inputs == {"event_arm_table": Path("/tmp/content-in/event_arm_table.pkl")}


def test_bundle_writes_a_provenance_sidecar_beside_every_pickle(tmp_path) -> None:
    table = tmp_path / "event_arm_table.pkl"
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
    assert config.output_dir == Path("/tmp/content-in/models_leaf")
    assert "models_leaf" in mod.canonical_output_subdirs()
    assert "models_swr_T_leaf" in mod.canonical_output_subdirs()
    assert "models_hard_count_leaf" in mod.canonical_output_subdirs()


def test_leaf_only_fits_six_candidates_not_nine() -> None:
    assert mod.fit_n_categories("arm", leaf_only=True) == 6
    assert mod.fit_n_categories("arm", leaf_only=False) == 9
    assert mod.fit_n_categories("patch", leaf_only=False) == 3


def test_leaf_only_drops_the_trunk_arm_fixed_effects_from_every_rung() -> None:
    """Arms 6 and 7 are identically zero on leaves; the ladder must not carry them."""
    for name, spec in mod.MODEL_HIERARCHY.items():
        predictors = mod.rung_predictors(spec, leaf_only=True)
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
        assert mod.rung_predictors(spec, leaf_only=False) == list(spec.predictors)

    arm_fe = mod.rung_predictors(mod.MODEL_HIERARCHY["M0a"], leaf_only=True)
    assert arm_fe == [f"is_candidate_arm_{i}" for i in range(5)]


def test_leaf_only_cohort_keeps_the_six_leaf_rows_of_every_event() -> None:
    table = _canonical_arm_table(n_events=3)
    leaves = mod.restrict_to_leaf_arms(table)

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
        mod.restrict_to_leaf_arms(
            _canonical_arm_table().drop(columns=["candidate_arm"])
        )


def test_leaf_only_is_refused_on_the_patch_hierarchy() -> None:
    """Patches are not arms; there is no leaf subset of the three patches."""
    with pytest.raises(ValueError, match="--leaf-only"):
        mod._resolve_run_config(_args(hierarchy="patch", leaf_only=True))
    with pytest.raises(ValueError, match="leaf"):
        mod.canonical_output_subdir(
            "non_local", "HPC", "patch", "all", "timebin", "fraction", leaf_only=True
        )


def test_leaf_only_subset_may_not_target_a_canonical_bundle() -> None:
    with pytest.raises(ValueError, match="canonical models_leaf/ bundle"):
        mod._resolve_run_config(
            _args(
                leaf_only=True,
                models="M4,Mv",
                output_dir=Path("/tmp/content-in/models_leaf"),
            )
        )


def test_leaf_only_subset_run_is_keyed_off_the_leaf_directory() -> None:
    config = mod._resolve_run_config(
        _args(leaf_only=True, models="M0,M0a,M1,M2,M3,M4,Mv,Mu,Mvu")
    )
    assert config.output_dir == Path(
        "/tmp/content-in/models_leaf_M0-M0a-M1-M2-M3-M4-Mv-Mu-Mvu"
    )


def test_parser_defaults_to_the_nine_arm_candidate_set() -> None:
    parser = mod._build_arg_parser()
    assert parser.parse_args([]).leaf_only is False
    assert parser.parse_args(["--leaf-only"]).leaf_only is True
