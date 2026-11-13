"""``paths``: each output location equals the inline spelling it replaced.

A migrated location that resolved to a different ``Path`` would silently read a
stale artifact, so every root is pinned to the literal it replaced, and
``table_path`` is checked over the whole canonical-table registry. Every root is
resolved when called, from ``$CSR_PROCESSED_DATA_DIR``: the suite points it at a
sentinel (``tests/conftest.py``), and these tests at a root of their own.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

from continuum_swr_replay import parameters, paths
from continuum_swr_replay.parameters import FIGURE_DIR, ROOT_DIR

#: The data root these tests point ``$CSR_PROCESSED_DATA_DIR`` at.
DATA_ROOT = Path("/a/data/root")
RC = DATA_ROOT / "replay_content"

ROOT_LITERALS = {
    "replay_content_dir": "replay_content",
    "contfrag_replay_content_dir": "contfrag_replay_content",
    "detection_comparison_dir": "replay_detection_comparison",
    "swr_inventory_root": "replay_detection_comparison/swr",
    "memoryless_root": "replay_detection_comparison/memoryless",
    "carey_root": "replay_detection_comparison/carey",
    "inventory_comparison_root": "replay_detection_comparison/inventory",
    "event_features_root": "replay_detection_comparison/event_features",
    "artifact_intervals_root": "artifact_intervals",
    "event_spike_evidence_dir": "event_spike_evidence",
    "content_occupancy_dir": "content_occupancy",
}
"""Each root's accessor -> its spelling under the data root."""


@pytest.fixture(autouse=True)
def data_root(monkeypatch):
    monkeypatch.setenv(paths.PROCESSED_DATA_DIR_ENV, str(DATA_ROOT))


def test_roots_equal_the_literals_they_replaced():
    assert paths.processed_data_dir() == DATA_ROOT
    for accessor, literal in ROOT_LITERALS.items():
        assert getattr(paths, accessor)() == DATA_ROOT / literal, accessor


def test_every_root_follows_the_environment_when_called(monkeypatch, tmp_path):
    """No root is bound at import: moving the variable moves every root."""
    monkeypatch.setenv(paths.PROCESSED_DATA_DIR_ENV, str(tmp_path))

    assert paths.processed_data_dir() == tmp_path
    for accessor, literal in ROOT_LITERALS.items():
        assert getattr(paths, accessor)() == tmp_path / literal, accessor
    assert (
        paths.table_path("stop_table") == tmp_path / "replay_content/stop_table.parquet"
    )
    assert (
        paths.model_dir("per_bin_rate")
        == tmp_path / "replay_content/models_per_bin_rate"
    )


def test_without_the_variable_the_root_is_the_checkouts(monkeypatch):
    monkeypatch.delenv(paths.PROCESSED_DATA_DIR_ENV)

    assert paths.processed_data_dir() == ROOT_DIR / "Processed-Data"


def test_the_upper_case_names_resolve_when_read(monkeypatch, tmp_path):
    """Kept for notebooks: ``paths.REPLAY_CONTENT_DIR`` and the rest, and
    ``parameters.PROCESSED_DATA_DIR``, are the accessors' values at the read."""
    names = {
        "REPLAY_CONTENT_DIR": "replay_content_dir",
        "CONTFRAG_REPLAY_CONTENT_DIR": "contfrag_replay_content_dir",
        "DETECTION_COMPARISON_DIR": "detection_comparison_dir",
        "SWR_INVENTORY_ROOT": "swr_inventory_root",
        "MEMORYLESS_ROOT": "memoryless_root",
        "CAREY_ROOT": "carey_root",
        "INVENTORY_COMPARISON_ROOT": "inventory_comparison_root",
        "EVENT_FEATURES_ROOT": "event_features_root",
        "ARTIFACT_INTERVALS_DIR": "artifact_intervals_root",
    }
    for root in (DATA_ROOT, tmp_path):
        monkeypatch.setenv(paths.PROCESSED_DATA_DIR_ENV, str(root))
        assert parameters.PROCESSED_DATA_DIR == root
        for name, accessor in names.items():
            assert getattr(paths, name) == getattr(paths, accessor)(), name
    from continuum_swr_replay.analysis.event_definition import swr_source

    for root in (DATA_ROOT, tmp_path):
        monkeypatch.setenv(paths.PROCESSED_DATA_DIR_ENV, str(root))
        assert (
            swr_source.SWR_INVENTORY_ROOT == root / ROOT_LITERALS["swr_inventory_root"]
        )
        assert swr_source.ARTIFACT_INTERVALS_DIR == (
            root / ROOT_LITERALS["artifact_intervals_root"]
        )
    with pytest.raises(AttributeError):
        swr_source.NOT_A_ROOT  # noqa: B018
    with pytest.raises(AttributeError):
        paths.NOT_A_ROOT  # noqa: B018
    with pytest.raises(AttributeError):
        parameters.NOT_A_PARAMETER  # noqa: B018


# The modules that may name the upper-case roots (``paths``', ``parameters``' and
# ``swr_source``'s ``SWR_INVENTORY_ROOT`` / ``ARTIFACT_INTERVALS_DIR``): their definitions, and the
# unsupported one-off scripts (``scripts/exploratory``), which notebooks resemble.
_ROOT_NAME = re.compile(
    r"\b(PROCESSED_DATA_DIR|REPLAY_CONTENT_DIR|CONTFRAG_REPLAY_CONTENT_DIR|"
    r"DETECTION_COMPARISON_DIR|SWR_INVENTORY_ROOT|MEMORYLESS_ROOT|CAREY_ROOT|"
    r"INVENTORY_COMPARISON_ROOT|EVENT_FEATURES_ROOT|ARTIFACT_INTERVALS_DIR)\b"
)


def test_no_supported_module_binds_a_root_by_its_upper_case_name():
    """A module that imported a root by name would hold it from its import on,
    and a test would have to patch that module to move it."""
    allowed = {
        ROOT_DIR / "src/continuum_swr_replay/paths.py",
        ROOT_DIR / "src/continuum_swr_replay/parameters.py",
        ROOT_DIR / "src/continuum_swr_replay/analysis/event_definition/swr_source.py",
    }
    offenders = []
    for folder in ("src", "scripts"):
        for path in sorted((ROOT_DIR / folder).rglob("*.py")):
            if path in allowed or "exploratory" in path.parts:
                continue
            tree = ast.parse(path.read_text())
            names = (
                {
                    alias.name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom)
                    for alias in node.names
                }
                | {
                    node.attr
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Attribute)
                }
                | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            )
            hits = sorted(name for name in names if _ROOT_NAME.fullmatch(name))
            if hits:
                offenders.append(f"{path.relative_to(ROOT_DIR)}: {hits}")

    assert offenders == []


MODEL_LITERALS = {
    "content_arm": "models",
    "content_patch": "models_patch",
    "per_bin_rate": "models_per_bin_rate",
    "per_event_swr_power": "models_per_event_swr_power",
    "per_stop_rate": "models_per_stop_rate",
    "swr_contrast": "models_swr_contrast",
}


def test_model_subdirs_equal_the_literals_they_replaced():
    assert paths.MODEL_SUBDIRS == MODEL_LITERALS


@pytest.mark.parametrize("key", sorted(MODEL_LITERALS))
def test_model_dir_under_the_default_and_a_staging_root(key):
    assert paths.model_dir(key) == RC / MODEL_LITERALS[key]
    root = Path("/staging/root")
    assert paths.model_dir(key, root) == root / MODEL_LITERALS[key]


def test_model_dir_keeps_a_nested_subdir_whole(monkeypatch):
    # The old ``root / MODEL_DIRS[k].name`` idiom kept only the last component.
    monkeypatch.setitem(paths.MODEL_SUBDIRS, "per_bin_rate", "models/per_bin_rate")
    root = Path("/staging/root")
    assert paths.model_dir("per_bin_rate", root) == root / "models" / "per_bin_rate"


@pytest.mark.parametrize("key", ["models", "per_bin", "content", ""])
def test_model_dir_refuses_an_unknown_key(key):
    with pytest.raises(KeyError, match="not a fitted-model directory"):
        paths.model_dir(key)


def test_registry_is_the_one_parity_reads():
    from continuum_swr_replay.tables import parity

    assert parity.BASENAME_TO_TABLE is paths.BASENAME_TO_TABLE
    assert parity.CANONICAL_TABLE_BASENAMES is paths.CANONICAL_TABLE_BASENAMES
    assert paths.CANONICAL_TABLE_BASENAMES == (
        "event_table",
        "event_arm_table",
        "event_patch_table",
        "event_sequence_table",
        "stop_table",
        "qvals_table",
        "event_onset_table",
        "stop_timebin_table",
    )


def test_the_one_table_list_names_the_registry_row_keys_and_swr_tables():
    from continuum_swr_replay.tables import covariate_registry
    from continuum_swr_replay.tables.io import SWR_SOURCE_TABLES

    assert set(covariate_registry._ROW_KEYS) == set(paths.CANONICAL_TABLES)
    assert SWR_SOURCE_TABLES == {
        "event",
        "event_arm",
        "event_patch",
        "event_sequence",
        "stop",
        "event_onset",
        "stop_timebin",
    }


@pytest.mark.parametrize("name", paths.CANONICAL_TABLE_BASENAMES)
@pytest.mark.parametrize("fmt", ["parquet", "pkl"])
def test_table_path_for_every_registry_name(name, fmt):
    # Every canonical table, qvals_table included, sits at the root it names.
    file = f"{name}.{fmt}"
    assert paths.table_path(name, fmt=fmt) == RC / file
    root = Path("/staging/root")
    assert paths.table_path(name, root, fmt=fmt) == root / file


def test_table_path_defaults_to_parquet_under_replay_content():
    assert paths.table_path("stop_table") == RC / "stop_table.parquet"
    assert paths.table_path("qvals_table") == RC / "qvals_table.parquet"


def test_every_swr_power_output_resolves_under_its_own_model_dir(tmp_path):
    """The per-event ripple-power model is not a per-bin rate fit."""
    import pandas as pd

    from scripts.replay_models.coupling import fit_swr_power_model as mod

    out_dir = paths.model_dir("per_event_swr_power", tmp_path)
    mod._write_outputs(
        out_dir,
        coefs=pd.DataFrame({"a": [1.0]}),
        z_reference={},
        gate_audit=pd.DataFrame({"animal": ["chimi"]}),
        cohort_records=[
            {
                "model": "swr_power",
                "n_in": 1,
                "n_events_excluded_pre_outcome": 2,
                "n_events_excluded_outcome_imbalance": 3,
            }
        ],
        smoke=False,
    )
    # The cohort CSV carries the power model's two split exclusion counts.
    cohort = pd.read_csv(out_dir / "swr_power_cohort_audit.csv")
    assert cohort[list(mod.COHORT_EXTRA_COLUMNS)].to_dict("records") == [
        {
            "n_events_excluded_pre_outcome": 2,
            "n_events_excluded_outcome_imbalance": 3,
        }
    ]
    written = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")}
    assert written == {
        "models_per_event_swr_power",
        "models_per_event_swr_power/swr_power_model.pkl",
        "models_per_event_swr_power/swr_power_model.z_reference_by_session.json",
        "models_per_event_swr_power/swr_power_post_outcome_gate_audit.csv",
        "models_per_event_swr_power/swr_power_cohort_audit.csv",
    }


@pytest.mark.parametrize(
    "name", ["events", "stop_table.parquet", "event_arm_table_swr_HPC", ""]
)
def test_table_path_refuses_a_name_outside_the_registry(name):
    with pytest.raises(KeyError, match="not a canonical table"):
        paths.table_path(name)


def test_the_swr_power_command_line_writes_under_its_own_model_dir():
    from scripts.replay_models.coupling import fit_swr_power_model as mod

    assert mod.build_parser().parse_args([]).output_dir == (
        RC / "models_per_event_swr_power"
    )


def test_the_qvals_builder_writes_beside_the_other_canonical_tables(monkeypatch):
    from continuum_swr_replay.analysis.behavior.covariates import qvals_table as builder

    written = {}
    monkeypatch.setattr(
        builder,
        "publish_canonical",
        lambda frame, path, *, table, basename: written.setdefault(
            "table", Path(path) / f"{basename}.parquet"
        ),
    )
    monkeypatch.setattr(
        builder, "save_reference", lambda ref, path: written.setdefault("ref", path)
    )
    root = Path("/staging/root")

    builder.write_qvals_outputs(None, {}, root)

    assert written == {
        "table": paths.table_path("qvals_table", root),
        "ref": root / "qvals_standardization_reference.json",
    }


@pytest.mark.parametrize("topic", ["rate", "replay_content", "qc_diagnosis"])
def test_figure_dir_on_the_default_root_is_the_paper_figure_dir(topic, monkeypatch):
    monkeypatch.delenv(paths.PROCESSED_DATA_DIR_ENV)

    assert paths.figure_dir(topic) == FIGURE_DIR / topic


def test_figure_dir_on_a_staging_root_sits_beside_it():
    root = Path("/staging/Processed-Data")
    assert paths.figure_dir("rate", root) == Path("/staging/paper/figures/rate")


def test_content_occupancy_writes_its_table_under_the_data_root(tmp_path, monkeypatch):
    # The events summary reads the table from the data root, so a default run
    # must write it there, on whichever root the run is given.
    from scripts.replay_models.content import analyze_content_occupancy

    monkeypatch.setenv(paths.PROCESSED_DATA_DIR_ENV, str(tmp_path))

    output_dir = analyze_content_occupancy._parse_args(["--all"]).output_dir

    assert output_dir == tmp_path / "content_occupancy"
    assert output_dir == paths.content_occupancy_dir()


# Each run_refits figure script (and compare_content_estimators, a fit that
# also draws), and where it writes: an expression on the imported script, or,
# for one that calls ``figure_dir`` where it saves, None (its source is checked).
FIGURE_SCRIPT_DEFAULTS = {
    "scripts/replay_models/rate/plot_per_bin_rate_models.py": (None, "rate"),
    "scripts/replay_models/rate/plot_per_bin_rate_summary.py": (None, "rate"),
    "scripts/replay_models/rate/plot_model_checking.py": (None, "rate"),
    "scripts/replay_models/rate/plot_uncertainty_rate.py": (None, "rate"),
    "scripts/figures/generate_uncertainty_content_figure.py": (
        "_parse_args([]).output_dir",
        "uncertainty_content",
    ),
    "scripts/replay_models/content/generate_replay_content_figures.py": (
        "output_dir()",
        "replay_content",
    ),
    "scripts/figures/generate_supp_pipeline_data_flow.py": (
        "build_parser().parse_args([]).output_dir",
        "qc_diagnosis",
    ),
    "scripts/figures/plot_swr_contrast_coefs.py": (None, "swr_contrast"),
    "scripts/replay_models/content/compare_content_estimators.py": (
        "_parse_args([]).figures_dir",
        "content_estimators",
    ),
}


@pytest.mark.parametrize("rel", sorted(FIGURE_SCRIPT_DEFAULTS))
def test_figure_scripts_write_beside_a_staging_data_root(rel, tmp_path, monkeypatch):
    # A run on another root (run_refits on a staging root, cwd the repo) must
    # not write into the paper's figures. The root is read when the figure
    # directory is resolved, so the script is imported here, root already set.
    import importlib.util

    expression, topic = FIGURE_SCRIPT_DEFAULTS[rel]
    path = ROOT_DIR / rel
    if expression is None:
        source = path.read_text()
        assert f'figure_dir("{topic}")' in source
        assert "FIGURE_DIR" not in source
        return
    monkeypatch.setenv(paths.PROCESSED_DATA_DIR_ENV, str(tmp_path / "Processed-Data"))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)  # dataclasses look it up
    spec.loader.exec_module(module)

    assert eval("module." + expression) == tmp_path / "paper" / "figures" / topic
