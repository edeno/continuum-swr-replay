"""Tests for the covariate naming registry (the old->new migration source)."""

from __future__ import annotations

import keyword

import pytest

from continuum_swr_replay.analysis.replay import covariate_registry as reg


def test_registry_validates() -> None:
    reg.validate()


def test_rename_map_is_a_function_with_unique_targets() -> None:
    """Every name maps uniquely within its declared table/grain."""
    all_currents = [(e.table, n) for e in reg.REGISTRY for n in e.current]
    assert len(all_currents) == len(set(all_currents))
    canon_to_entries: dict[tuple[str, str], int] = {}
    for e in reg.REGISTRY:
        if e.canonical:
            key = (e.table, e.canonical)
            canon_to_entries[key] = canon_to_entries.get(key, 0) + 1
    assert all(c == 1 for c in canon_to_entries.values())
    for table in {e.table for e in reg.REGISTRY}:
        assert set(reg.rename_map(table=table).values()) == reg.canonical_names(
            table=table
        )


def test_actions_partition_columns() -> None:
    """A current column has one and only one action."""
    buckets = [
        {
            (e.table, n)
            for e in reg.REGISTRY
            if e.action in {"rename", "alias", "canonicalize"}
            for n in e.current
        },
        {(e.table, n) for e in reg.REGISTRY if e.action == "retire" for n in e.current},
        {(e.table, n) for e in reg.REGISTRY if e.action == "drop" for n in e.current},
        {(e.table, n) for e in reg.derived_columns() for n in e.current},
        {(e.table, n) for e in reg.audit_columns() for n in e.current},
    ]
    flat = [n for b in buckets for n in b]
    assert len(flat) == len(set(flat)), "a column falls under two actions"
    # And together they account for every registry column.
    assert set(flat) == {(e.table, n) for e in reg.REGISTRY for n in e.current}


def test_canonical_names_are_lowercase_snake_case() -> None:
    for name in reg.canonical_names():
        assert name == name.lower(), f"{name!r} is not lowercase"
        assert reg._is_snake_identifier(name)
        assert not keyword.iskeyword(name)
        assert " " not in name and "-" not in name


def test_no_canonical_collides_with_a_legacy_name() -> None:
    """A new name must not shadow a legacy column in the same table."""
    for table in {e.table for e in reg.REGISTRY}:
        assert reg.canonical_names(table=table).isdisjoint(
            reg.all_current_columns(table=table)
        )


def test_digit_boundary_rule() -> None:
    assert "reward_belief_mean_port_0" in reg.canonical_names()
    assert "decision_value_patch_1" in reg.canonical_names()
    assert not reg._digit_boundary_ok("reward_belief_mean_port0")
    assert reg._digit_boundary_ok("reward_fraction_candidate_arm_last10_visits")
    assert reg._digit_boundary_ok("posterior_hpd_arm_count_mean_coverage_50pct")


def test_indexed_families_are_complete_and_correctly_based() -> None:
    qvals = reg.rename_map(table="qvals")
    stop = reg.rename_map(table="stop")
    fit_time = reg.rename_map(table="fit_time")
    for i in range(6):
        assert qvals[f"q_port{i}"] == f"reward_belief_mean_port_{i}"
        assert qvals[f"betadist_mu{i + 1}"] == f"reward_belief_mean_port_{i}"
        assert qvals[f"betadist_var{i + 1}"] == f"reward_belief_variance_port_{i}"
        assert qvals[f"p_reward_port{i}"] == f"task_reward_probability_port_{i}"
    for i in range(1, 4):
        assert qvals[f"Qstem{i}"] == f"decision_value_patch_{i}"
    for j in range(1, 11):
        assert stop[f"reward_lag{j}"] == f"known_reward_outcome_lag_{j}"
    for k in range(8):
        assert fit_time[f"arm_fe_{k}"] == f"is_candidate_arm_{k}"
    assert "arm_fe_8" not in fit_time  # arm 8 is the dropped reference


def test_exact_alias_families_collapse_with_survivors_present() -> None:
    """Each byte-identical alias group resolves to its canonical name."""
    stop = reg.rename_map(table="stop")
    event = reg.rename_map(table="event")
    assert stop["current_arm"] == stop["location_arm"] == "occupancy_well_arm"
    assert stop["reward_outcome"] == stop["is_after_reward"] == "is_rewarded_outcome"
    patch = reg.rename_map(table="event_patch")
    assert patch["current_arm"] == patch["location_arm"] == "occupancy_well_arm"
    assert (
        event["posterior_mass_total"]
        == event["nonlocal_weight_sum"]
        == "nonlocal_probability_sum"
    )
    assert (
        event["time_since_port_entry"]
        == event["time_since_last_port_entry"]
        == "time_since_last_well_arrival_s"
    )
    for stem in ("start", "end", "duration"):
        assert (
            event[f"label_event_{stem}"] == event[f"event_{stem}"] == f"event_{stem}_s"
        )
    assert (
        event["label_event_midpoint_time"]
        == event["event_midpoint_time"]
        == "event_midpoint_s"
    )


def test_alias_vs_canonicalize_use_correct_actions() -> None:
    """Byte-identical pairs are 'alias' (bit-for-bit); numerical-path pairs are
    'canonicalize' (declared tolerance)."""
    by_current = {n: e for e in reg.REGISTRY for n in e.current}
    assert by_current["reward_outcome"].action == "alias"
    assert by_current["time_since_port_entry"].action == "alias"
    assert by_current["current_arm"].action == "alias"
    assert by_current["q_port0"].action == "canonicalize"
    assert by_current["prev_global_variance"].action == "canonicalize"


def test_hpd_and_geometry_named_exceptions_present() -> None:
    event = reg.rename_map(table="event")
    for c in (50, 80, 95):
        assert (
            event[f"posterior_hpd_width_{c}"]
            == f"posterior_hpd_arm_count_mean_coverage_{c}pct"
        )
    assert event["day"] == "session_date"
    assert event["n_junction_bins"] == "n_undecoded_bins"


def test_switch_interaction_is_retired_not_renamed() -> None:
    """switch_x_current is disendorsed with its base predictor; it must not
    survive under a new name."""
    assert "switch_x_current" in reg.retired_columns()
    assert "switch_x_current" not in reg.rename_map(table="event_arm")
    # The base switch predictor is still renamed (it stays a stored primitive;
    # removing it from the per-stop rate ladder is a separate
    # model-definition change).
    assert (
        reg.rename_map(table="stop")["is_after_switch"]
        == "did_last_completed_trip_switch_patch"
    )


def test_belief_and_contingency_are_disjoint_namespaces() -> None:
    canon = reg.canonical_names()
    beliefs = {n for n in canon if n.startswith("reward_belief_")}
    contingencies = {n for n in canon if n.startswith("task_reward_probability_")}
    assert beliefs and contingencies
    assert beliefs.isdisjoint(contingencies)
    assert not any(n.startswith("posterior_") for n in beliefs), (
        "reward_belief_ avoids the spatial-decoder 'posterior_' namespace"
    )


def test_complements_reference_a_stored_primitive() -> None:
    stored = reg.canonical_names()
    for e in reg.derived_columns():
        assert e.primitive in stored


def test_depletion_drop_family_is_complete() -> None:
    dropped = reg.dropped_columns()
    for i in range(1, 7):
        assert f"depletion_leaf_{i}" in dropped
    for i in range(1, 4):
        assert f"depletion_mean_stem_{i}" in dropped
    assert len(dropped) == 9


@pytest.mark.parametrize(
    "bad,reason",
    [
        ("port0", "glued index"),
        ("patch3", "glued index"),
        ("Value", "not snake_case"),
        ("class", "python keyword"),
    ],
)
def test_digit_boundary_rejects_glued_or_nonsnake(bad: str, reason: str) -> None:
    assert not (reg._is_snake_identifier(bad) and reg._digit_boundary_ok(bad)), reason


def test_event_geometry_units_and_subjects() -> None:
    m = reg.rename_map(table="event")
    # Animal head speed: subject prefix + cm/s.
    for s in ("max", "mean", "start", "end"):
        assert m[f"{s}_speed"] == f"animal_{s}_speed_cm_s"
    # Replay MAP speed: already replay_-prefixed, append cm/s.
    assert m["replay_speed_median"] == "replay_speed_median_cm_s"
    assert m["replay_speed_p90"] == "replay_speed_p90_cm_s"
    assert m["ab_slope"] == "ab_slope_cm_s"
    # Replay position collides with the animal's -> subject prefix + cm.
    assert m["start_position"] == "replay_start_position_cm"
    assert m["end_position"] == "replay_end_position_cm"
    # Unambiguous replay geometry: append cm, no prefix needed.
    for c in (
        "net_displacement",
        "path_length",
        "spatial_extent",
        "step_distance_median",
        "step_distance_p90",
        "step_distance_max",
        "max_ahead",
        "max_behind",
        "max_abs_ahead_behind",
        "mean_abs_ahead_behind",
    ):
        assert m[c] == f"{c}_cm"
    # Fraction: only path_continuity lacked the word.
    assert m["path_continuity"] == "path_continuity_fraction"


def test_geometry_renames_target_real_event_table_columns() -> None:
    """Every geometry rename source must be a live event_table column, and no
    canonical may shadow one."""
    pytest.importorskip("pandas")
    from pathlib import Path

    from continuum_swr_replay.table_io import read_table, table_exists

    path = Path("Processed-Data/replay_content/event_table")
    if not table_exists(path):
        pytest.skip("event_table not materialized")
    cols = set(read_table(path).columns)
    # Post-migration the materialized event_table carries CANONICAL names, so the
    # invariant is that every geometry rename TARGET is a live column (and the
    # legacy sources have been renamed away). The pre-migration form of this test
    # checked the sources instead, which was correct only while the table still
    # held legacy spellings.
    geometry = [
        e
        for e in reg.REGISTRY
        if e.table == "event"
        and (e.canonical or "").endswith(("_cm", "_cm_s", "_fraction"))
    ]
    missing_targets = {e.canonical for e in geometry} - cols
    assert not missing_targets, (
        f"geometry rename targets absent from event_table: {missing_targets}"
    )
    surviving_sources = {n for e in geometry for n in e.current} & cols
    assert not surviving_sources, (
        f"legacy geometry sources still present (should be renamed): {surviving_sources}"
    )


def test_plan_rename_on_a_synthetic_frame() -> None:
    """Actions are applied only within the requested table/grain."""
    event_arm = reg.plan_rename(
        ["q_arm", "switch_x_current", "stop_id"], table="event_arm"
    )
    assert event_arm.rename["q_arm"] == "reward_belief_mean_candidate_arm_preoutcome"
    assert "switch_x_current" in event_arm.drop
    assert "stop_id" not in event_arm.drop and "stop_id" not in event_arm.rename

    stop = reg.plan_rename(
        ["current_arm", "location_arm", "is_after_stay", "stop_id"], table="stop"
    )
    survivors = [c for c in ("current_arm", "location_arm") if c in stop.rename]
    assert len(survivors) == 1
    assert stop.rename[survivors[0]] == "occupancy_well_arm"
    other = ({"current_arm", "location_arm"} - {survivors[0]}).pop()
    assert other in stop.drop
    assert "is_after_stay" in stop.drop

    qvals = reg.plan_rename(["depletion_leaf_1", "stop_id"], table="qvals")
    assert "depletion_leaf_1" in qvals.drop


def test_same_legacy_spelling_has_table_specific_semantics() -> None:
    pd = pytest.importorskip("pandas")

    frame = pd.DataFrame({"arm": [2], "trial_reward": [1]})
    event_arm = reg.apply_rename(frame, table="event_arm")
    qvals = reg.apply_rename(frame, table="qvals")
    assert list(event_arm) == ["candidate_arm", "trial_reward"]
    assert list(qvals) == ["chosen_arm", "trial_reward"]


def test_plan_rename_detects_a_collision() -> None:
    """If a canonical name would land on an existing keep column, fail loud."""
    # occupancy_well_arm is q_arm's neighbour in name only; construct a direct
    # collision: a frame already containing a canonical target plus its source.
    cols = ["q_arm", "reward_belief_mean_candidate_arm_preoutcome"]
    with pytest.raises(reg.RegistryError):
        reg.plan_rename(cols, table="event_arm")


def test_materialized_tables_are_canonical_after_regeneration() -> None:
    """A regenerated artifact must satisfy the canonical schema contract.

    Legacy materialized tables are expected on a source-only rename checkout;
    they are not evidence about the new producer schema and must be regenerated.
    """
    pytest.importorskip("pandas")
    from pathlib import Path

    from continuum_swr_replay.table_io import read_table, table_exists

    rc = Path("Processed-Data/replay_content")
    tables = {
        "event_arm": rc / "event_arm_table",
        "event_patch": rc / "event_patch_table",
        "stop": rc / "stop_table",
        "qvals": rc / "models_per_bin_rate" / "qvals_table",
        "event": rc / "event_table",
    }
    if not all(table_exists(p) for p in tables.values()):
        pytest.skip("canonical tables not materialized")
    frames = {table: read_table(path) for table, path in tables.items()}
    legacy = [
        table
        for table, frame in frames.items()
        if (
            frame.attrs.get("covariate_schema_version") != reg.COVARIATE_SCHEMA_VERSION
            or frame.attrs.get("covariate_schema_scope") != reg.COVARIATE_SCHEMA_SCOPE
        )
    ]
    if legacy:
        pytest.skip(f"materialized tables require canonical regeneration: {legacy}")
    for table, frame in frames.items():
        reg.require_covariate_schema(frame, table=table)


def test_fitted_model_specs_do_not_reference_registry_legacy_names() -> None:
    """Executable predictor/formula registries use canonical vocabulary only."""
    import re

    from continuum_swr_replay.analysis.replay import model_specs as specs

    model_groups = (
        specs.MODEL_HIERARCHY,
        specs.MODEL_HIERARCHY_SWR,
        specs.MODEL_HIERARCHY_MUA,
        specs.MODEL_PATCH,
        specs.MODEL_PATCH_SWR,
        specs.MODEL_PATCH_MUA,
        specs.MODEL_PATCH_SWITCH,
        specs.MODEL_CHOICE,
        specs.MODEL_PER_STOP_RATE,
    )
    executable = [
        predictor
        for group in model_groups
        for model in group.values()
        for predictor in model.predictors
    ]
    executable.extend(model.formula() for model in specs.MODEL_PER_BIN_RATE.values())
    executable.extend(
        formula for formula, _ in specs.SWR_CONTRAST_RATE_G_SPECS.values()
    )
    executable.extend(
        formula for formula, _ in specs.SWR_CONTRAST_RATE_S_SPECS.values()
    )
    text = "\n".join(executable)
    surviving = sorted(
        name
        for name in reg.all_current_columns()
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text)
    )
    assert not surviving, f"model specs still reference legacy names: {surviving}"


def _event_arm_frame():
    pd = pytest.importorskip("pandas")
    import numpy as np

    # A synthetic event-arm frame: a 1:1 rename and a keep column.
    v = np.array([0.1, 0.4, np.nan, 0.9])
    return pd.DataFrame(
        {
            "q_arm": v,  # rename
            "stop_id": [10, 11, 12, 13],  # keep
        }
    )


def test_apply_rename_is_value_preserving_and_collapses_aliases() -> None:
    pytest.importorskip("pandas")
    import numpy as np

    df = _event_arm_frame()
    out = reg.apply_rename(df, table="event_arm")
    # 1:1 rename, values identical.
    assert "reward_belief_mean_candidate_arm_preoutcome" in out.columns
    np.testing.assert_array_equal(
        out["reward_belief_mean_candidate_arm_preoutcome"].to_numpy(),
        df["q_arm"].to_numpy(),
    )
    # Keep column untouched; no duplicate output columns.
    assert "stop_id" in out.columns
    assert not out.columns.duplicated().any()


def test_apply_rename_rejects_alias_twins_that_differ() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "current_arm": [0.0, 1.0],
            "location_arm": [0.0, 1.0],
        }
    )
    df.loc[0, "location_arm"] = 5.0  # break the byte-identical invariant
    with pytest.raises(reg.RegistryError, match="not bit-for-bit"):
        reg.apply_rename(df, table="stop")


def test_apply_rename_collapses_alias_twins_within_table() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "current_arm": [0.0, 1.0],
            "location_arm": [0.0, 1.0],
            "stop_id": [1, 2],
        }
    )
    out = reg.apply_rename(df, table="stop")
    assert list(out) == ["occupancy_well_arm", "stop_id"]


def test_event_patch_location_alias_migrates_at_its_own_table_boundary() -> None:
    """The patch reducer used ``location_arm`` even though the original
    registry declared the alias only for arm-grain tables."""
    pd = pytest.importorskip("pandas")

    legacy = pd.DataFrame(
        {
            "animal": ["a"],
            "day": [20200101],
            "epoch": [2],
            "event_id": [10],
            "patch": [1],
            "location_arm": [0.0],
        }
    )
    out = reg.finalize_covariate_schema(legacy, table="event_patch")
    assert out["occupancy_well_arm"].tolist() == [0.0]
    assert not {"current_arm", "location_arm", "day"} & set(out.columns)
    reg.require_covariate_schema(out, table="event_patch")


def test_qvals_and_count_migrations_are_table_scoped_and_explicit() -> None:
    qvals = reg.rename_map(table="qvals")
    stop = reg.rename_map(table="stop")
    stop_timebin = reg.rename_map(table="stop_timebin")

    assert qvals["p_reward_current"] == "task_reward_probability_chosen_arm"
    assert qvals["p_reward_best"] == "task_reward_probability_best_arm"
    assert qvals["chosen_leaf_var_preoutcome_sess"] == (
        "reward_belief_variance_chosen_arm_preoutcome"
    )
    assert stop["n_events"] == stop["n_non_local"] == "n_selected_events"
    assert stop["n_SWR"] == "n_swr_events"
    assert stop_timebin["n_non_local"] == "n_nonlocal_events"


def test_new_qvals_and_count_renames_preserve_values_exactly() -> None:
    pd = pytest.importorskip("pandas")
    import numpy as np

    qvals = pd.DataFrame(
        {
            "stop_id": [1, 2],
            "p_reward_current": [0.2, 0.8],
            "chosen_leaf_mu_update_abs_postoutcome_sess": [np.nan, 0.125],
        }
    )
    qvals_out = reg.apply_rename(qvals, table="qvals")
    np.testing.assert_array_equal(
        qvals_out["task_reward_probability_chosen_arm"].to_numpy(),
        qvals["p_reward_current"].to_numpy(),
    )
    np.testing.assert_array_equal(
        qvals_out["reward_belief_mean_update_chosen_arm_postoutcome_abs"].to_numpy(),
        qvals["chosen_leaf_mu_update_abs_postoutcome_sess"].to_numpy(),
    )

    stop = pd.DataFrame(
        {
            "n_events": [0, 3],
            "n_non_local": [0, 3],
            "n_SWR": [1, 2],
        }
    )
    stop_out = reg.apply_rename(stop, table="stop")
    np.testing.assert_array_equal(stop_out["n_selected_events"], stop["n_events"])
    np.testing.assert_array_equal(stop_out["n_swr_events"], stop["n_SWR"])


def test_schema_vocabulary_and_scientific_row_identity_are_distinct_layers() -> None:
    pd = pytest.importorskip("pandas")

    duplicated = pd.DataFrame(
        {
            "stop_id": [1, 1],
            "task_reward_probability_chosen_arm": [0.2, 0.2],
        }
    )
    # Names are canonical; row identity is not.
    reg.assert_canonical_vocabulary(duplicated, table="qvals")
    with pytest.raises(reg.RegistryError, match="duplicated scientific keys"):
        reg.assert_unique_scientific_rows(duplicated, table="qvals")
    with pytest.raises(reg.RegistryError, match="duplicated scientific keys"):
        reg.stamp_covariate_schema(duplicated, table="qvals")


def test_schema_stamp_declares_and_requires_its_exact_scope() -> None:
    pd = pytest.importorskip("pandas")

    frame = pd.DataFrame(
        {
            "stop_id": [1, 2],
            "task_reward_probability_chosen_arm": [0.2, 0.8],
        }
    )
    stamped = reg.stamp_covariate_schema(frame, table="qvals")
    assert stamped.attrs["covariate_schema_scope"] == reg.COVARIATE_SCHEMA_SCOPE
    reg.require_covariate_schema(stamped, table="qvals")

    prior_definition = stamped.copy()
    prior_definition.attrs["covariate_schema_version"] = "covariate_names_v1"
    with pytest.raises(ValueError, match="legacy or mismatched"):
        reg.require_covariate_schema(prior_definition, table="qvals")

    mislabeled = stamped.copy()
    mislabeled.attrs["covariate_schema_scope"] = "names_only"
    with pytest.raises(ValueError, match="legacy or mismatched"):
        reg.require_covariate_schema(mislabeled, table="qvals")


def test_apply_rename_leaves_canonicalize_and_prune_columns_alone() -> None:
    """The mechanical rename must not touch numerical-path or dropped columns."""
    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(
        {
            "q_port0": [0.5, 0.6],  # canonicalize -> left for the other step
            "betadist_mu1": [0.5, 0.6],
            "switch_x_current": [0.0, 1.0],  # retire -> left (pruning step)
            "depletion_leaf_1": [1.0, 1.0],  # drop -> left (pruning step)
        }
    )
    out = reg.apply_rename(df, table="qvals")
    assert set(out.columns) == set(df.columns), "apply_rename touched non-rename cols"


def test_canonicalize_numerical_paths_within_tolerance() -> None:
    pd = pytest.importorskip("pandas")
    import numpy as np

    df = pd.DataFrame(
        {
            "Q1": [0.5, 0.6, np.nan],
            "q_port0": [0.5, 0.6, np.nan],
            "betadist_mu1": [0.5 + 1e-16, 0.6 - 1e-16, np.nan],  # fp-identical
            "stop_id": [1, 2, 3],
        }
    )
    out = reg.canonicalize_numerical_paths(df, table="qvals")
    assert "reward_belief_mean_port_0" in out.columns
    assert not {"Q1", "q_port0", "betadist_mu1"} & set(out.columns)
    assert "stop_id" in out.columns


def test_canonicalize_rejects_paths_beyond_tolerance() -> None:
    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(
        {
            "Q1": [0.5, 0.6],
            "q_port0": [0.5, 0.6],
            "betadist_mu1": [0.5, 0.61],
        }
    )
    with pytest.raises(reg.RegistryError, match="exceed tolerance"):
        reg.canonicalize_numerical_paths(df, table="qvals")


def test_canonicalize_rejects_all_nan_family_as_unverifiable() -> None:
    """An all-NaN family cannot be verified equal -> refuse, don't collapse."""
    pd = pytest.importorskip("pandas")
    import numpy as np

    df = pd.DataFrame(
        {
            "Q1": [np.nan, np.nan],
            "q_port0": [np.nan, np.nan],
            "betadist_mu1": [np.nan, np.nan],
        }
    )
    with pytest.raises(reg.RegistryError, match="unverifiable"):
        reg.canonicalize_numerical_paths(df, table="qvals")


def test_canonicalize_rejects_mismatched_missingness() -> None:
    pd = pytest.importorskip("pandas")
    import numpy as np

    df = pd.DataFrame(
        {
            "Q1": [0.5, np.nan],
            "q_port0": [0.5, np.nan],
            "betadist_mu1": [0.5, 0.6],
        }
    )
    with pytest.raises(reg.RegistryError, match="missingness"):
        reg.canonicalize_numerical_paths(df, table="qvals")


def test_export_json_roundtrips() -> None:
    import json

    payload = json.loads(reg.export_json())
    assert isinstance(payload, list) and len(payload) == len(reg.REGISTRY)
