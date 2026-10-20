"""Tests for the covariate naming contract.

The full old->new ledger now lives in the appendix of
``docs/design/2026-08-21-covariate-naming-scheme.md``. What stays executable in
``covariate_registry`` is the runtime half: the frozen deny-list, the row-key
contract and the stamp built on them, plus the ``qvals`` ledger rows that the
vendored behavioral-model ingestion still translates on every build. These
tests cover that half, and pin the frozen deny-list back to the published
ledger it was extracted from.
"""

from __future__ import annotations

import keyword
import re
from pathlib import Path

import pytest

from continuum_swr_replay.analysis.replay import covariate_registry as reg

REPO = Path(__file__).resolve().parents[2]
NAMING_SCHEME = REPO / "docs/design/2026-08-21-covariate-naming-scheme.md"
APPENDIX_HEADING = "## Appendix: the rename ledger"
# Actions whose third column names a canonical replacement; ``derive``,
# ``retire``, ``drop`` and ``audit`` rows carry no canonical by construction
# (``validate`` rejects one).
CANONICAL_ACTIONS = frozenset({"rename", "alias", "canonicalize"})


def _qvals_rename_map() -> dict[str, str]:
    """Legacy qvals spelling -> canonical, from the surviving ledger rows."""
    return {
        name: entry.canonical
        for entry in reg.REGISTRY
        if entry.canonical is not None
        for name in entry.current
    }


# ---------------------------------------------------------------------------
# The frozen deny-list
# ---------------------------------------------------------------------------
def test_deny_list_rejects_a_legacy_column() -> None:
    """A retired spelling cannot reach a stamped frame."""
    pd = pytest.importorskip("pandas")

    frame = pd.DataFrame({"stop_id": [1, 2], "q_arm": [0.2, 0.8]})
    with pytest.raises(reg.RegistryError, match="legacy/retired covariate columns"):
        reg.assert_canonical_vocabulary(frame, table="qvals")
    with pytest.raises(reg.RegistryError, match="legacy/retired covariate columns"):
        reg.stamp_covariate_schema(frame, table="qvals")


def test_deny_list_rejects_a_decorated_legacy_column() -> None:
    """A legacy predictor also cannot survive as a standardized or fitted name."""
    pd = pytest.importorskip("pandas")

    for decorated in ("q_arm_z", "coef_q_arm", "bse_arm_uncertainty_z"):
        frame = pd.DataFrame({"stop_id": [1, 2], decorated: [0.2, 0.8]})
        with pytest.raises(reg.RegistryError, match="legacy derived result columns"):
            reg.assert_canonical_vocabulary(frame, table="qvals")


def test_deny_list_leaves_a_canonical_decorated_column_alone() -> None:
    pd = pytest.importorskip("pandas")

    frame = pd.DataFrame(
        {
            "stop_id": [1, 2],
            "reward_belief_mean_port_0_z": [0.2, 0.8],
            "coef_reward_belief_mean_port_0": [0.2, 0.8],
        }
    )
    reg.assert_canonical_vocabulary(frame, table="qvals")


def _appendix_ledger_rows() -> list[tuple[str, str, str, str, str]]:
    """Parse the published ledger's tables into five-column rows.

    Reads only the tables under ``## Appendix: the rename ledger`` in the
    naming-scheme design doc: header and separator lines are skipped, and a row
    is ``| legacy spelling(s) | action | canonical / target | role | notes |``.
    """
    appendix = NAMING_SCHEME.read_text().split(APPENDIX_HEADING, 1)
    assert len(appendix) == 2, f"{NAMING_SCHEME} has no {APPENDIX_HEADING!r}"

    rows: list[tuple[str, str, str, str, str]] = []
    for line in appendix[1].splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert len(cells) == 5, f"ledger row is not 5 columns: {line}"
        if cells[0] in {"legacy spelling(s)", "---"}:
            continue
        rows.append((cells[0], cells[1], cells[2], cells[3], cells[4]))
    return rows


def test_deny_list_matches_the_published_ledger() -> None:
    """``LEGACY_COLUMN_NAMES``/``_LEGACY_RENAMED`` are the appendix's own sets.

    The frozen sets were extracted from the ledger when it moved out of the
    module; this recomputes them from the published table so neither side can
    drift. Removing a name from the appendix fails this test.
    """
    rows = _appendix_ledger_rows()
    assert rows, "the appendix ledger parsed to zero rows"

    non_audit: set[str] = set()
    with_canonical: set[str] = set()
    for legacy, action, target, _role, _notes in rows:
        # A cell may collapse several legacy spellings: "`a`, `b`".
        names = re.findall(r"`([^`]+)`", legacy)
        assert names, f"ledger row names no legacy column: {legacy!r}"
        if action != "audit":
            non_audit |= set(names)
        if action in CANONICAL_ACTIONS:
            assert re.fullmatch(r"`[a-z][a-z0-9_]*`", target), (
                f"{action} row {names} has no single canonical target: {target!r}"
            )
            with_canonical |= set(names)

    assert non_audit == set(reg.LEGACY_COLUMN_NAMES)
    assert with_canonical == set(reg._LEGACY_RENAMED)
    assert len(reg.LEGACY_COLUMN_NAMES) == 219


def test_fitted_model_specs_do_not_reference_legacy_names() -> None:
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
        for name in reg.LEGACY_COLUMN_NAMES
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text)
    )
    assert not surviving, f"model specs still reference legacy names: {surviving}"


# ---------------------------------------------------------------------------
# The surviving qvals ledger rows
# ---------------------------------------------------------------------------
def test_registry_validates() -> None:
    reg.validate()


def test_only_the_qvals_grain_stays_executable() -> None:
    assert {e.table for e in reg.REGISTRY} == {"qvals"}
    assert reg._entries_for("stop") == ()
    with pytest.raises(reg.RegistryError, match="unknown covariate-registry table"):
        reg._entries_for("not_a_table")


def test_canonical_names_are_lowercase_snake_case() -> None:
    for name in reg.canonical_names():
        assert name == name.lower(), f"{name!r} is not lowercase"
        assert reg._is_snake_identifier(name)
        assert not keyword.iskeyword(name)
        assert " " not in name and "-" not in name


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


def test_digit_boundary_rule() -> None:
    assert "reward_belief_mean_port_0" in reg.canonical_names()
    assert "decision_value_patch_1" in reg.canonical_names()
    assert not reg._digit_boundary_ok("reward_belief_mean_port0")
    assert reg._digit_boundary_ok("reward_fraction_candidate_arm_last10_visits")
    assert reg._digit_boundary_ok("posterior_hpd_arm_count_mean_coverage_50pct")


def test_indexed_qvals_families_are_complete_and_correctly_based() -> None:
    qvals = _qvals_rename_map()
    for i in range(6):
        assert qvals[f"q_port{i}"] == f"reward_belief_mean_port_{i}"
        assert qvals[f"betadist_mu{i + 1}"] == f"reward_belief_mean_port_{i}"
        assert qvals[f"betadist_var{i + 1}"] == f"reward_belief_variance_port_{i}"
        assert qvals[f"p_reward_port{i}"] == f"task_reward_probability_port_{i}"
    for i in range(1, 4):
        assert qvals[f"Qstem{i}"] == f"decision_value_patch_{i}"


def test_alias_vs_canonicalize_use_correct_actions() -> None:
    """Numerical-path pairs are 'canonicalize' (declared tolerance)."""
    by_current = {n: e for e in reg.REGISTRY for n in e.current}
    assert by_current["q_port0"].action == "canonicalize"
    assert by_current["prev_global_variance"].action == "canonicalize"


def test_belief_and_contingency_are_disjoint_namespaces() -> None:
    canon = reg.canonical_names()
    beliefs = {n for n in canon if n.startswith("reward_belief_")}
    contingencies = {n for n in canon if n.startswith("task_reward_probability_")}
    assert beliefs and contingencies
    assert beliefs.isdisjoint(contingencies)
    assert not any(n.startswith("posterior_") for n in beliefs), (
        "reward_belief_ avoids the spatial-decoder 'posterior_' namespace"
    )


def test_depletion_drop_family_is_complete() -> None:
    dropped = {n for e in reg.REGISTRY if e.action == "drop" for n in e.current}
    for i in range(1, 7):
        assert f"depletion_leaf_{i}" in dropped
    for i in range(1, 4):
        assert f"depletion_mean_stem_{i}" in dropped
    assert len(dropped) == 9


# ---------------------------------------------------------------------------
# qvals ingestion
# ---------------------------------------------------------------------------
def test_plan_rename_on_a_synthetic_qvals_frame() -> None:
    plan = reg.plan_rename(["arm", "depletion_leaf_1", "stop_id"], table="qvals")
    assert plan.rename["arm"] == "chosen_arm"
    assert "depletion_leaf_1" in plan.drop
    assert "stop_id" not in plan.drop and "stop_id" not in plan.rename


def test_renames_are_scoped_to_the_grain_that_declares_them() -> None:
    """``arm`` is the realized destination in qvals and nothing anywhere else."""
    pd = pytest.importorskip("pandas")

    frame = pd.DataFrame({"arm": [2], "trial_reward": [1]})
    assert list(reg.apply_rename(frame, table="qvals")) == [
        "chosen_arm",
        "trial_reward",
    ]
    assert list(reg.apply_rename(frame, table="event_arm")) == ["arm", "trial_reward"]


def test_plan_rename_detects_a_collision() -> None:
    """If a canonical name would land on an existing keep column, fail loud."""
    with pytest.raises(reg.RegistryError):
        reg.plan_rename(["arm", "chosen_arm"], table="qvals")


def test_native_qvals_vocabulary_canonicalizes_end_to_end() -> None:
    """The vendored model's native spellings become the canonical schema."""
    pd = pytest.importorskip("pandas")
    import numpy as np

    native = pd.DataFrame(
        {
            "stop_id": [1, 2],
            # numerical-path family: three spellings of one quantity
            "Q1": [0.5, 0.6],
            "q_port0": [0.5, 0.6],
            "betadist_mu1": [0.5 + 1e-16, 0.6 - 1e-16],
            # plain renames
            "betadist_var1": [0.01, 0.02],
            "Qstem1": [0.4, 0.7],
            "rpe": [-0.1, 0.3],
            "arm": [2, 5],
            # a dead output that the pruning step removes
            "depletion_leaf_1": [1.0, 1.0],
        }
    )

    out = reg.canonicalize_numerical_paths(native, table="qvals")
    out = reg.apply_rename(out, table="qvals")
    out = reg.rename_dynamic_columns(out, table="qvals")
    out = out.drop(columns=list(reg.plan_rename(out.columns, table="qvals").drop))
    stamped = reg.stamp_covariate_schema(out, table="qvals")

    assert set(stamped.columns) == {
        "stop_id",
        "reward_belief_mean_port_0",
        "reward_belief_variance_port_0",
        "decision_value_patch_1",
        "rpe_postoutcome",
        "chosen_arm",
    }
    np.testing.assert_array_equal(
        stamped["reward_belief_mean_port_0"].to_numpy(), native["Q1"].to_numpy()
    )
    np.testing.assert_array_equal(
        stamped["rpe_postoutcome"].to_numpy(), native["rpe"].to_numpy()
    )
    reg.require_covariate_schema(stamped, table="qvals")


def test_qvals_renames_preserve_values_exactly() -> None:
    pd = pytest.importorskip("pandas")
    import numpy as np

    qvals = pd.DataFrame(
        {
            "stop_id": [1, 2],
            "p_reward_current": [0.2, 0.8],
            "chosen_leaf_mu_update_abs_postoutcome_sess": [np.nan, 0.125],
        }
    )
    out = reg.apply_rename(qvals, table="qvals")
    np.testing.assert_array_equal(
        out["task_reward_probability_chosen_arm"].to_numpy(),
        qvals["p_reward_current"].to_numpy(),
    )
    np.testing.assert_array_equal(
        out["reward_belief_mean_update_chosen_arm_postoutcome_abs"].to_numpy(),
        qvals["chosen_leaf_mu_update_abs_postoutcome_sess"].to_numpy(),
    )


def test_apply_rename_leaves_canonicalize_and_prune_columns_alone() -> None:
    """The mechanical rename must not touch numerical-path or dropped columns."""
    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(
        {
            "q_port0": [0.5, 0.6],  # canonicalize -> left for the other step
            "betadist_mu1": [0.5, 0.6],
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


# ---------------------------------------------------------------------------
# Row identity and the stamp
# ---------------------------------------------------------------------------
def test_row_keys_are_registered_for_every_persisted_grain() -> None:
    assert reg.row_keys(table="qvals") == ("stop_id",)
    assert reg.row_keys(table="stop") == ("animal", "session_date", "epoch", "stop_id")
    with pytest.raises(reg.RegistryError):
        reg.row_keys(table="not_a_table")


def test_missing_row_key_column_is_rejected() -> None:
    pd = pytest.importorskip("pandas")

    frame = pd.DataFrame({"task_reward_probability_chosen_arm": [0.2]})
    with pytest.raises(reg.RegistryError, match="missing required row keys"):
        reg.assert_canonical_vocabulary(frame, table="qvals")


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
    with pytest.raises(ValueError, match="covariate schema version"):
        reg.require_covariate_schema(prior_definition, table="qvals")

    unstamped = stamped.copy()
    del unstamped.attrs["covariate_schema_version"]
    with pytest.raises(ValueError, match="legacy or mismatched"):
        reg.require_covariate_schema(unstamped, table="qvals")

    mislabeled = stamped.copy()
    mislabeled.attrs["covariate_schema_scope"] = "names_only"
    with pytest.raises(ValueError, match="legacy or mismatched"):
        reg.require_covariate_schema(mislabeled, table="qvals")


@pytest.mark.real_data
def test_materialized_tables_are_canonical_after_regeneration() -> None:
    """A regenerated artifact must satisfy the canonical schema contract.

    Legacy materialized tables are expected on a source-only rename checkout;
    they are not evidence about the new producer schema and must be regenerated,
    so under ``-m real_data`` they fail this test.
    """
    pytest.importorskip("pandas")

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
        pytest.fail("canonical tables not materialized")
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
        pytest.fail(f"materialized tables require canonical regeneration: {legacy}")
    for table, frame in frames.items():
        reg.require_covariate_schema(frame, table=table)
