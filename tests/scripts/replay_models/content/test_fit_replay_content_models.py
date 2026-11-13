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

from continuum_swr_replay.analysis.replay_models import model_specs
from continuum_swr_replay.analysis.replay_models.cohort_audit import event_drop_buckets
from continuum_swr_replay.analysis.replay_models.content import fit_outputs
from continuum_swr_replay.analysis.replay_models.content.cohort import (
    zero_response_event_ids,
)
from continuum_swr_replay.analysis.replay_models.content.session_fits import (
    LOGIT_SKIP_COLUMNS,
)
from continuum_swr_replay.analysis.replay_models.model_specs import (
    ModelSpec,
    default_content_model_names,
)
from continuum_swr_replay.data_loaders.constants import ALL_ARMS, WELL_ARMS
from continuum_swr_replay.runs.fit_provenance import (
    check_fit_provenance,
    record_inputs,
)
from continuum_swr_replay.tables.io import write_table
from scripts.replay_models.content import fit_replay_content_models as mod
from tests.support._content_tables import (
    canonical_arm_table,
    dm_session,
    fail_logit_for,
    logit_session,
)

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


_ARM_LADDER = {"X": ModelSpec("X", ("is_current_arm",), "current-arm bias", None)}


def test_undecoded_events_are_reported_as_zero_response_drops() -> None:
    """The audit must name the mechanism: no response, not a missing predictor."""
    table = canonical_arm_table(n_events=5, undecoded_event="e3")
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


def test_the_printed_se_skips_a_session_that_cannot_inform_the_term(capsys) -> None:
    """The fitter reports such a term with a NaN coefficient and an infinite SE.

    The session drops out of that term's summary, SE included, rather than
    pulling the median SE toward infinity.
    """
    mdf = pd.DataFrame(
        {
            "animal": ["a", "a", "b", "b", "c"],
            "session_date": list(range(5)),
            "epoch": "02_r1",
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
    # Columns: name, animal median, animals positive, session median, SE, p,
    # unidentified.
    assert float(line.split()[4]) == pytest.approx(0.2)
    # The sessions that cannot inform the term are reported, not just skipped.
    assert line.split()[6] == "2"


def test_the_animal_column_counts_animals_with_a_positive_median(capsys) -> None:
    """Animal a has five negative sessions, b and c one positive each.

    The session median is negative (5 of 7 sessions), the animal median is
    positive (2 of 3 animals): the column prints 2/3 animals positive, not
    the 1/3 that share the session median's sign.
    """
    mdf = pd.DataFrame(
        {
            "animal": ["a"] * 5 + ["b", "c"],
            "session_date": list(range(7)),
            "epoch": "02_r1",
            "coef_is_candidate_arm_0": [-1.0, -1.1, -1.2, -1.3, -1.4, 0.5, 0.7],
            "bse_is_candidate_arm_0": [0.1] * 7,
        }
    )
    hierarchy = {"X": argparse.Namespace(description="test rung")}

    mod.print_model_results("X", mdf, ["is_candidate_arm_0"], hierarchy=hierarchy)

    lines = capsys.readouterr().out.splitlines()
    header = next(text for text in lines if text.strip().startswith("Predictor"))
    line = next(t for t in lines if t.strip().startswith("is_candidate_arm_0"))
    assert header.split()[2] == "anim>0"
    # Columns: name, animal median, animals positive, session median, ...
    assert float(line.split()[1]) == pytest.approx(0.5)
    assert float(line.split()[3]) == pytest.approx(-1.1)
    assert line.split()[2] == "2/3"


# ── Event-scale reporting ──


def _comparison_row(estimator: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "parent": "M0",
                "child": "M1",
                "estimator": estimator,
                "n_sessions": 200,
                # a child better in most sessions: AIC(child) - AIC(parent) < 0
                "delta_aic_median": -12.5,
                "delta_aic_animal_median": -11.0,
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
    assert "-12.5" in text and "-11.0" in text
    assert "ΔAIC = child − parent, negative = child better" in text


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
    write_table(canonical_arm_table(), table)
    fit_inputs = record_inputs({"event_arm_table": table})
    frame = pd.DataFrame({"model": ["X"], "coef_is_current_arm": [0.5]})
    output_dir = tmp_path / "models"

    mod._write_complete_output_bundle(
        output_dir=output_dir,
        model_results={"X": frame},
        combined=frame,
        cohort_audits=[],
        logit_results={"X": frame},
        fit_inputs=fit_inputs,
        estimator="conditional_poisson_fraction",
        cli=["fit_replay_content_models.py", "--response", "fraction"],
    )

    for artifact in (
        output_dir / "X.pkl",
        output_dir / "all_models.pkl",
        output_dir / "logit" / "X_logit.pkl",
    ):
        assert check_fit_provenance(artifact, {"event_arm_table": table}) == []
    record = json.loads((output_dir / "all_models.provenance.json").read_text())
    assert record["estimator"] == "conditional_poisson_fraction"
    assert record["cli"][-1] == "fraction"


# ── Leaf-only candidate set ──


def test_leaf_only_run_publishes_to_its_own_bundle() -> None:
    """A six-arm fit is a different likelihood, so it gets a different bundle."""
    config = mod._resolve_run_config(_args(leaf_only=True))
    assert config.output_dir == CONTENT_IN / "models_leaf"
    assert "models_leaf" in fit_outputs.canonical_output_subdirs()
    assert "models_swr_T_leaf" in fit_outputs.canonical_output_subdirs()
    assert "models_hard_count_leaf" in fit_outputs.canonical_output_subdirs()


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
    table = canonical_arm_table(n_events=3)
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
            canonical_arm_table().drop(columns=["candidate_arm"])
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
    assert zero_response_event_ids(
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
    write_table(canonical_arm_table(), table)
    frame = pd.DataFrame({"model": ["X"], "coef_is_current_arm": [0.5]})
    output_dir = tmp_path / "models_leaf"

    mod._write_complete_output_bundle(
        output_dir=output_dir,
        model_results={"X": frame},
        combined=frame,
        cohort_audits=[],
        logit_results={},
        fit_inputs=record_inputs({"event_arm_table": table}),
        estimator="conditional_poisson_fraction",
        cli=["fit_replay_content_models.py", "--leaf-only"],
        response_renormalized=True,
    )

    record = json.loads((output_dir / "all_models.provenance.json").read_text())
    assert record["response_renormalized"] is True


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


def test_a_refused_dm_session_is_named_in_the_published_cohort_audit(tmp_path) -> None:
    """The skip channel's consumer: the audit says which session, and why.

    A session the estimator refuses is not a dropped event -- its events are
    already counted under ``n_excluded_session`` -- so the only place a reader
    can learn that a session is missing from the fitted cohort, and for what
    reason, is the audit's ``drop_notes``.
    """
    good = dm_session(60, seed=0, epoch="02_r1")
    bad = dm_session(60, seed=2, epoch="04_r2").assign(is_prev_arm=0.0)
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


_LOGIT_LADDER = {
    "X": ModelSpec("X", ("is_current_arm",), "current", None),
    "Y": ModelSpec("Y", ("is_current_arm", "is_prev_arm"), "current + prev", "X"),
}


def _logit_fixture(tmp_path):
    cohort_df = pd.concat(
        [logit_session("02_r1", seed=0), logit_session("04_r2", seed=1)],
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


def test_logit_skips_are_published_beside_the_logit_fits(tmp_path, monkeypatch) -> None:
    args, config, cohort = _logit_fixture(tmp_path)
    model_results, skips = mod._fit_content_ladder(args, config, cohort)
    fail_logit_for(monkeypatch, "04_r2", 2, np.linalg.LinAlgError("singular"))

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
    assert tuple(published.columns) == LOGIT_SKIP_COLUMNS
