"""Tests for the covariate naming registry (the old->new migration source)."""

from __future__ import annotations

import keyword

import pytest

from continuum_swr_replay.analysis.replay import covariate_registry as reg


def test_registry_validates() -> None:
    reg.validate()


def test_rename_map_is_a_function_with_unique_targets() -> None:
    """Every current name maps to one canonical, and distinct quantities never
    collide on a canonical name."""
    m = reg.rename_map()
    all_currents = [n for e in reg.REGISTRY for n in e.current]
    assert len(all_currents) == len(set(all_currents))
    canon_to_entries: dict[str, int] = {}
    for e in reg.REGISTRY:
        if e.canonical:
            canon_to_entries[e.canonical] = canon_to_entries.get(e.canonical, 0) + 1
    assert all(c == 1 for c in canon_to_entries.values())
    assert set(m.values()) == reg.canonical_names()


def test_actions_partition_columns() -> None:
    """A current column has one and only one action."""
    buckets = [
        set(reg.rename_map()),  # rename + alias + canonicalize
        reg.retired_columns(),
        reg.dropped_columns(),
        {n for e in reg.derived_columns() for n in e.current},
        {n for e in reg.audit_columns() for n in e.current},
    ]
    flat = [n for b in buckets for n in b]
    assert len(flat) == len(set(flat)), "a column falls under two actions"
    # And together they account for every registry column.
    assert set(flat) == set(reg.all_current_columns())


def test_canonical_names_are_lowercase_snake_case() -> None:
    for name in reg.canonical_names():
        assert name == name.lower(), f"{name!r} is not lowercase"
        assert reg._is_snake_identifier(name)
        assert not keyword.iskeyword(name)
        assert " " not in name and "-" not in name


def test_no_canonical_collides_with_a_legacy_name() -> None:
    """A new name must never equal a column still present under its old name."""
    assert reg.canonical_names().isdisjoint(reg.all_current_columns())


def test_digit_boundary_rule() -> None:
    assert "reward_belief_mean_port_0" in reg.canonical_names()
    assert "decision_value_patch_1" in reg.canonical_names()
    assert not reg._digit_boundary_ok("reward_belief_mean_port0")
    assert reg._digit_boundary_ok("reward_fraction_candidate_arm_last10_visits")
    assert reg._digit_boundary_ok("posterior_hpd_arm_count_mean_coverage_50pct")


def test_indexed_families_are_complete_and_correctly_based() -> None:
    m = reg.rename_map()
    for i in range(6):
        assert m[f"q_port{i}"] == f"reward_belief_mean_port_{i}"
        assert m[f"betadist_mu{i + 1}"] == f"reward_belief_mean_port_{i}"
        assert m[f"betadist_var{i + 1}"] == f"reward_belief_variance_port_{i}"
        assert m[f"p_reward_port{i}"] == f"task_reward_probability_port_{i}"
    for i in range(1, 4):
        assert m[f"Qstem{i}"] == f"decision_value_patch_{i}"
    for j in range(1, 11):
        assert m[f"reward_lag{j}"] == f"known_reward_outcome_lag_{j}"
    for k in range(8):
        assert m[f"arm_fe_{k}"] == f"is_candidate_arm_{k}"
    assert "arm_fe_8" not in m  # arm 8 is the dropped reference


def test_exact_alias_families_collapse_with_survivors_present() -> None:
    """Each byte-identical alias group resolves to its canonical name."""
    m = reg.rename_map()
    assert m["current_arm"] == m["location_arm"] == "occupancy_well_arm"
    assert m["reward_outcome"] == m["is_after_reward"] == "is_rewarded_outcome"
    assert m["posterior_mass_total"] == m["nonlocal_weight_sum"] == "nonlocal_probability_sum"
    assert m["time_since_port_entry"] == m["time_since_last_port_entry"] == "time_since_last_well_arrival_s"
    for stem in ("start", "end", "duration"):
        assert m[f"label_event_{stem}"] == m[f"event_{stem}"] == f"event_{stem}_s"
    assert m["label_event_midpoint_time"] == m["event_midpoint_time"] == "event_midpoint_s"


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
    m = reg.rename_map()
    for c in (50, 80, 95):
        assert m[f"posterior_hpd_width_{c}"] == f"posterior_hpd_arm_count_mean_coverage_{c}pct"
    assert m["day"] == "session_date"
    assert m["n_junction_bins"] == "n_undecoded_bins"


def test_switch_interaction_is_retired_not_renamed() -> None:
    """switch_x_current is disendorsed with its base predictor; it must not
    survive under a new name."""
    assert "switch_x_current" in reg.retired_columns()
    assert "switch_x_current" not in reg.rename_map()
    # The base switch predictor is still renamed (it stays a stored primitive;
    # removing it from R1/R2/Rlag is a separate model-definition change).
    assert reg.rename_map()["is_after_switch"] == "did_last_completed_trip_switch_patch"


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
    m = reg.rename_map()
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
    for c in ("net_displacement", "path_length", "spatial_extent",
              "step_distance_median", "step_distance_p90", "step_distance_max",
              "max_ahead", "max_behind", "max_abs_ahead_behind",
              "mean_abs_ahead_behind"):
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
    geometry_sources = {
        c for c, e in ((n, e) for e in reg.REGISTRY for n in e.current)
        if e.table == "event" and (
            e.canonical or ""
        ).endswith(("_cm", "_cm_s", "_fraction"))
    }
    missing = geometry_sources - cols
    assert not missing, f"geometry rename sources absent from event_table: {missing}"


def test_plan_rename_on_a_synthetic_frame() -> None:
    """A simple rename, an alias collapse, a retire, a drop, a derive, and an
    untouched keep column each resolve correctly."""
    cols = [
        "q_arm",  # rename
        "current_arm", "location_arm",  # alias collapse -> one survivor
        "switch_x_current",  # retire
        "depletion_leaf_1",  # drop (dead)
        "is_after_stay",  # derive (complement, dropped from storage)
        "stop_id",  # keep (structural, not in registry)
        "whole_event_speed_eligible",  # audit (left in place here)
    ]
    plan = reg.plan_rename(cols)
    assert plan.rename["q_arm"] == "reward_belief_mean_candidate_arm_preoutcome"
    # Exactly one of the alias twins survives (renamed); the other is dropped.
    survivors = [c for c in ("current_arm", "location_arm") if c in plan.rename]
    assert len(survivors) == 1
    assert plan.rename[survivors[0]] == "occupancy_well_arm"
    other = ({"current_arm", "location_arm"} - {survivors[0]}).pop()
    assert other in plan.drop
    assert "switch_x_current" in plan.drop
    assert "depletion_leaf_1" in plan.drop
    assert "is_after_stay" in plan.drop
    assert "stop_id" not in plan.drop and "stop_id" not in plan.rename
    assert plan.audit_present == ("whole_event_speed_eligible",)


def test_plan_rename_detects_a_collision() -> None:
    """If a canonical name would land on an existing keep column, fail loud."""
    # occupancy_well_arm is q_arm's neighbour in name only; construct a direct
    # collision: a frame already containing a canonical target plus its source.
    cols = ["q_arm", "reward_belief_mean_candidate_arm_preoutcome"]
    with pytest.raises(reg.RegistryError):
        reg.plan_rename(cols)


def test_every_registry_source_is_a_live_column() -> None:
    """No registry entry may reference a column that does not exist — a drift
    guard against typos and stale names across the whole registry."""
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
    live: set[str] = set()
    for p in tables.values():
        live |= set(read_table(p).columns)
    # arm_fe_* are fit-time constructions, not stored table columns.
    exempt = {f"arm_fe_{k}" for k in range(8)}
    sources = reg.all_current_columns() - exempt
    missing = sorted(sources - live)
    assert not missing, f"registry references non-existent columns: {missing}"


def _frame():
    pd = pytest.importorskip("pandas")
    import numpy as np

    # A synthetic frame mimicking real columns: a 1:1 rename, a byte-identical
    # alias pair, and a keep column.
    v = np.array([0.1, 0.4, np.nan, 0.9])
    return pd.DataFrame(
        {
            "q_arm": v,  # rename
            "current_arm": [0.0, 1.0, np.nan, 2.0],  # alias survivor/twin
            "location_arm": [0.0, 1.0, np.nan, 2.0],  # byte-identical twin
            "stop_id": [10, 11, 12, 13],  # keep
        }
    )


def test_apply_rename_is_value_preserving_and_collapses_aliases() -> None:
    pytest.importorskip("pandas")
    import numpy as np

    df = _frame()
    out = reg.apply_rename(df)
    # 1:1 rename, values identical.
    assert "reward_belief_mean_candidate_arm_preoutcome" in out.columns
    np.testing.assert_array_equal(
        out["reward_belief_mean_candidate_arm_preoutcome"].to_numpy(),
        df["q_arm"].to_numpy(),
    )
    # Alias pair collapsed to one canonical column; the twin is gone.
    assert "occupancy_well_arm" in out.columns
    assert "current_arm" not in out.columns and "location_arm" not in out.columns
    assert list(out.columns).count("occupancy_well_arm") == 1
    # Keep column untouched; no duplicate output columns.
    assert "stop_id" in out.columns
    assert not out.columns.duplicated().any()


def test_apply_rename_rejects_alias_twins_that_differ() -> None:
    pytest.importorskip("pandas")
    df = _frame()
    df.loc[0, "location_arm"] = 5.0  # break the byte-identical invariant
    with pytest.raises(reg.RegistryError, match="not bit-for-bit"):
        reg.apply_rename(df)


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
    out = reg.apply_rename(df)
    assert set(out.columns) == set(df.columns), "apply_rename touched non-rename cols"


def test_canonicalize_numerical_paths_within_tolerance() -> None:
    pd = pytest.importorskip("pandas")
    import numpy as np

    df = pd.DataFrame(
        {
            "q_port0": [0.5, 0.6, np.nan],
            "betadist_mu1": [0.5 + 1e-16, 0.6 - 1e-16, np.nan],  # fp-identical
            "stop_id": [1, 2, 3],
        }
    )
    out = reg.canonicalize_numerical_paths(df)
    assert "reward_belief_mean_port_0" in out.columns
    assert "q_port0" not in out.columns and "betadist_mu1" not in out.columns
    assert "stop_id" in out.columns


def test_canonicalize_rejects_paths_beyond_tolerance() -> None:
    pd = pytest.importorskip("pandas")

    df = pd.DataFrame({"q_port0": [0.5, 0.6], "betadist_mu1": [0.5, 0.61]})
    with pytest.raises(reg.RegistryError, match="exceed tolerance"):
        reg.canonicalize_numerical_paths(df)


def test_canonicalize_rejects_all_nan_family_as_unverifiable() -> None:
    """An all-NaN family cannot be verified equal -> refuse, don't collapse."""
    pd = pytest.importorskip("pandas")
    import numpy as np

    df = pd.DataFrame({"q_port0": [np.nan, np.nan], "betadist_mu1": [np.nan, np.nan]})
    with pytest.raises(reg.RegistryError, match="unverifiable"):
        reg.canonicalize_numerical_paths(df)


def test_canonicalize_rejects_mismatched_missingness() -> None:
    pd = pytest.importorskip("pandas")
    import numpy as np

    df = pd.DataFrame({"q_port0": [0.5, np.nan], "betadist_mu1": [0.5, 0.6]})
    with pytest.raises(reg.RegistryError, match="missingness"):
        reg.canonicalize_numerical_paths(df)


def test_export_json_roundtrips() -> None:
    import json

    payload = json.loads(reg.export_json())
    assert isinstance(payload, list) and len(payload) == len(reg.REGISTRY)
