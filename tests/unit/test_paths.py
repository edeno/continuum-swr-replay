"""``paths``: each output location equals the inline spelling it replaced.

A migrated constant that resolved to a different ``Path`` would silently read a
stale artifact, so every constant is pinned to the literal it replaced, and
``table_path`` is checked over the whole canonical-table registry.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from continuum_swr_replay import paths
from continuum_swr_replay.parameters import FIGURE_DIR, PROCESSED_DATA_DIR, ROOT_DIR

RC = PROCESSED_DATA_DIR / "replay_content"


def test_roots_equal_the_literals_they_replaced():
    assert paths.REPLAY_CONTENT_DIR == RC
    assert paths.CONTFRAG_REPLAY_CONTENT_DIR == (
        PROCESSED_DATA_DIR / "contfrag_replay_content"
    )
    assert paths.DETECTION_COMPARISON_DIR == (
        PROCESSED_DATA_DIR / "replay_detection_comparison"
    )
    assert paths.SWR_INVENTORY_ROOT == (
        PROCESSED_DATA_DIR / "replay_detection_comparison" / "swr"
    )
    assert paths.ARTIFACT_INTERVALS_DIR == PROCESSED_DATA_DIR / "artifact_intervals"


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

    from scripts.coupling import fit_swr_power_model as mod

    assert mod.OUTPUT_DIR == RC / "models_per_event_swr_power"
    assert mod.OUTPUT_DIR == paths.model_dir("per_event_swr_power")
    out_dir = paths.model_dir("per_event_swr_power", tmp_path)
    mod._write_outputs(
        out_dir,
        coefs=pd.DataFrame({"a": [1.0]}),
        z_reference={},
        gate_audit=pd.DataFrame({"animal": ["chimi"]}),
        cohort_records=[{"model": "swr_power", "n_in": 1}],
        smoke=False,
    )
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
    from scripts.coupling import fit_swr_power_model as mod

    assert mod.build_parser().parse_args([]).output_dir == (
        RC / "models_per_event_swr_power"
    )


def test_the_qvals_builder_writes_beside_the_other_canonical_tables(monkeypatch):
    from continuum_swr_replay.analysis.behavior import qvals_table as builder

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
def test_figure_dir_on_the_default_root_is_the_paper_figure_dir(topic):
    assert paths.figure_dir(topic) == FIGURE_DIR / topic


def test_figure_dir_on_a_staging_root_sits_beside_it():
    root = Path("/staging/Processed-Data")
    assert paths.figure_dir("rate", root) == Path("/staging/paper/figures/rate")


# Each run_refits figure script, and the module-level default it writes to.
# ``plot_uncertainty_rate`` builds its directory inside ``main``, so it is
# not listed; it calls ``figure_dir("rate")`` directly.
FIGURE_SCRIPT_DEFAULTS = {
    "scripts/rate/plot_per_bin_rate_models.py": ("FIGS", "rate"),
    "scripts/rate/plot_per_bin_rate_summary.py": ("FIGS", "rate"),
    "scripts/rate/plot_model_checking.py": ("FIGS", "rate"),
    "scripts/figures/generate_uncertainty_content_figure.py": (
        "DEFAULT_OUTPUT_DIR",
        "uncertainty_content",
    ),
    "scripts/content/generate_replay_content_figures.py": (
        "OUTPUT_DIR",
        "replay_content",
    ),
    "scripts/figures/generate_supp_pipeline_data_flow.py": (
        "build_parser().parse_args([]).output_dir",
        "qc_diagnosis",
    ),
}

_PRINT_DEFAULTS = """
import importlib.util, sys
from pathlib import Path
for rel, attr in zip(sys.argv[1::2], sys.argv[2::2]):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(eval("mod." + attr))
"""


def test_figure_scripts_write_beside_a_staging_data_root(tmp_path):
    # A run on another root (run_refits on a staging root, cwd the repo) must
    # not write into the paper's figures. The env var is read at import, so
    # the scripts are imported in a fresh interpreter.
    root = tmp_path / "Processed-Data"
    argv = [a for rel, (attr, _) in FIGURE_SCRIPT_DEFAULTS.items() for a in (rel, attr)]
    env = {**os.environ, "CSR_PROCESSED_DATA_DIR": str(root), "MPLBACKEND": "Agg"}
    out = subprocess.run(
        [sys.executable, "-c", _PRINT_DEFAULTS, *argv],
        cwd=ROOT_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    expected = [
        str(tmp_path / "paper" / "figures" / topic)
        for _, topic in FIGURE_SCRIPT_DEFAULTS.values()
    ]
    assert out == expected
