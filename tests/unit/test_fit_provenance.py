"""Value-level tests for the fitted-artifact provenance sidecars.

Every assertion here is against bytes on disk: the sha256 the writer records is
compared to ``hashlib.sha256`` of the file, and staleness is provoked by
changing exactly one byte of an input.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.fit_provenance import (
    check_fit_provenance,
    fit_provenance_path,
    git_hash,
    require_fresh_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.table_io import write_table

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flip_one_byte(path: Path) -> None:
    """Change a single byte in place, leaving the file's size untouched."""
    with path.open("r+b") as handle:
        handle.seek(0, 2)
        handle.seek(handle.tell() - 1)
        last = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([last[0] ^ 0xFF]))


@pytest.fixture()
def fit_dir(tmp_path: Path) -> Path:
    """A directory holding two input tables and one fitted artifact."""
    for name, version in (("stop_table", "v9"), ("event_table", "v9")):
        frame = pd.DataFrame({"stop_id": [0, 1, 2], "count": [1, 0, 4]})
        frame.attrs["covariate_schema_version"] = version
        write_table(frame, tmp_path / f"{name}.pkl")
    (tmp_path / "all_models.pkl").write_bytes(b"fitted-model-bytes")
    return tmp_path


@pytest.fixture()
def inputs(fit_dir: Path) -> dict[str, Path]:
    return {
        "stop_table": fit_dir / "stop_table.pkl",
        "event_table": fit_dir / "event_table.pkl",
    }


def test_sidecar_records_the_sha256_of_every_input_file(fit_dir, inputs):
    sidecar = write_fit_provenance(
        fit_dir / "all_models.pkl", inputs, "poisson_glm", ["fit.py", "--workers", "4"]
    )

    assert sidecar == fit_dir / "all_models.provenance.json"
    record = json.loads(sidecar.read_text())
    assert record["artifact"] == "all_models.pkl"
    assert record["estimator"] == "poisson_glm"
    assert record["cli"] == ["fit.py", "--workers", "4"]
    assert set(record["inputs"]) == {"stop_table", "event_table"}
    for name, path in inputs.items():
        parquet = path.with_suffix(".parquet")
        assert record["inputs"][name]["sha256"] == _sha256(parquet)
        assert record["inputs"][name]["mtime"] == parquet.stat().st_mtime
        assert record["inputs"][name]["covariate_schema_version"] == "v9"


def test_sidecar_hashes_the_parquet_a_pkl_path_resolves_to(fit_dir, inputs):
    """``read_table`` reads the parquet sibling, so the parquet is what is hashed."""
    (fit_dir / "stop_table.pkl").write_bytes(b"a stale pickle nobody reads")

    write_fit_provenance(fit_dir / "all_models.pkl", inputs, "poisson_glm", ["fit.py"])

    recorded = json.loads(fit_provenance_path(fit_dir / "all_models.pkl").read_text())
    assert recorded["inputs"]["stop_table"]["sha256"] == _sha256(
        fit_dir / "stop_table.parquet"
    )
    assert recorded["inputs"]["stop_table"]["sha256"] != _sha256(
        fit_dir / "stop_table.pkl"
    )


def test_sidecar_records_the_current_commit(fit_dir, inputs):
    write_fit_provenance(fit_dir / "all_models.pkl", inputs, "poisson_glm", ["fit.py"])

    recorded = json.loads(fit_provenance_path(fit_dir / "all_models.pkl").read_text())
    assert recorded["git_hash"] == git_hash()
    assert re.fullmatch(r"[0-9a-f]{40}(-dirty)?", recorded["git_hash"])


def test_checker_returns_no_stale_inputs_when_nothing_changed(fit_dir, inputs):
    write_fit_provenance(fit_dir / "all_models.pkl", inputs, "poisson_glm", ["fit.py"])

    assert check_fit_provenance(fit_dir / "all_models.pkl", inputs) == []


def test_checker_names_the_input_whose_bytes_changed(fit_dir, inputs):
    write_fit_provenance(fit_dir / "all_models.pkl", inputs, "poisson_glm", ["fit.py"])
    before = _sha256(fit_dir / "stop_table.parquet")

    _flip_one_byte(fit_dir / "stop_table.parquet")

    assert _sha256(fit_dir / "stop_table.parquet") != before
    assert check_fit_provenance(fit_dir / "all_models.pkl", inputs) == ["stop_table"]


def test_checker_names_an_input_the_sidecar_never_recorded(fit_dir, inputs):
    write_fit_provenance(
        fit_dir / "all_models.pkl",
        {"stop_table": inputs["stop_table"]},
        "poisson_glm",
        ["fit.py"],
    )

    assert check_fit_provenance(fit_dir / "all_models.pkl", inputs) == ["event_table"]


def test_checker_raises_when_the_sidecar_is_missing(fit_dir, inputs):
    with pytest.raises(FileNotFoundError, match="stale by definition"):
        check_fit_provenance(fit_dir / "all_models.pkl", inputs)


def test_reader_gate_raises_and_names_the_stale_input(fit_dir, inputs):
    write_fit_provenance(fit_dir / "all_models.pkl", inputs, "poisson_glm", ["fit.py"])
    _flip_one_byte(fit_dir / "event_table.parquet")

    with pytest.raises(RuntimeError, match="event_table changed"):
        require_fresh_inputs(fit_dir / "all_models.pkl", inputs, allow_stale=False)


def test_reader_gate_passes_with_allow_stale(fit_dir, inputs):
    write_fit_provenance(fit_dir / "all_models.pkl", inputs, "poisson_glm", ["fit.py"])
    _flip_one_byte(fit_dir / "event_table.parquet")

    assert (
        require_fresh_inputs(fit_dir / "all_models.pkl", inputs, allow_stale=True)
        is None
    )


def test_reader_gate_passes_with_allow_stale_when_the_sidecar_is_missing(
    fit_dir, inputs
):
    assert (
        require_fresh_inputs(fit_dir / "all_models.pkl", inputs, allow_stale=True)
        is None
    )


def test_reader_gate_is_silent_on_a_current_fit(fit_dir, inputs):
    write_fit_provenance(fit_dir / "all_models.pkl", inputs, "poisson_glm", ["fit.py"])

    assert (
        require_fresh_inputs(fit_dir / "all_models.pkl", inputs, allow_stale=False)
        is None
    )


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / "rate" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "reader", ["summarize_per_bin_rate_models", "plot_per_bin_rate_models"]
)
def test_readers_check_the_same_inputs_the_fitter_recorded(reader):
    """A reader checking a different table set than the fitter recorded is blind."""
    fitter = _load_script("fit_per_bin_rate_models")
    module = _load_script(reader)

    assert module.ALL_MODELS_INPUTS == fitter.FIT_INPUTS
    assert module.ALL_MODELS == fitter.RC / "models_per_bin_rate" / "all_models.pkl"


@pytest.mark.parametrize(
    "reader", ["summarize_per_bin_rate_models", "plot_per_bin_rate_models"]
)
def test_readers_default_to_refusing_a_stale_fit(reader):
    module = _load_script(reader)

    assert module.build_parser().parse_args([]).allow_stale is False
    assert module.build_parser().parse_args(["--allow-stale"]).allow_stale is True
