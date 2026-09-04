"""Value-level tests for the fitted-artifact provenance sidecars.

Every assertion here is against bytes on disk: the sha256 the writer records is
compared to ``hashlib.sha256`` of the file, and staleness is provoked by
changing exactly one byte of an input.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.fit_provenance import (
    check_fit_provenance,
    fit_provenance_path,
    git_hash,
    record_inputs,
    require_fresh_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.table_io import write_table

REPO_ROOT = Path(__file__).resolve().parents[2]
RATE = REPO_ROOT / "scripts" / "rate"
FIGURES = REPO_ROOT / "scripts" / "figures"


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


def test_recorded_hashes_survive_a_table_regenerated_during_the_fit(fit_dir, inputs):
    """The sidecar describes the bytes the fit read, not the bytes it finished on.

    A ladder over 218 sessions runs long enough for a table to be regenerated
    underneath it; hashing at write time would certify a table the fit never saw.
    """
    fit_inputs = record_inputs(inputs)
    before = _sha256(fit_dir / "stop_table.parquet")

    _flip_one_byte(fit_dir / "stop_table.parquet")
    write_fit_provenance(
        fit_dir / "all_models.pkl", fit_inputs, "poisson_glm", ["fit.py"]
    )

    recorded = json.loads(fit_provenance_path(fit_dir / "all_models.pkl").read_text())
    assert recorded["inputs"]["stop_table"]["sha256"] == before
    assert recorded["inputs"]["stop_table"]["sha256"] != _sha256(
        fit_dir / "stop_table.parquet"
    )
    # ... and the fit is then correctly reported stale against the table on disk.
    assert check_fit_provenance(fit_dir / "all_models.pkl", inputs) == ["stop_table"]


def test_each_input_carries_the_time_it_was_hashed(fit_dir, inputs):
    fit_inputs = record_inputs(inputs)
    write_fit_provenance(
        fit_dir / "all_models.pkl", fit_inputs, "poisson_glm", ["fit.py"]
    )

    recorded = json.loads(fit_provenance_path(fit_dir / "all_models.pkl").read_text())
    hashed_at = recorded["inputs"]["stop_table"]["hashed_at"]
    assert hashed_at == fit_inputs["stop_table"]["hashed_at"]
    assert hashed_at <= recorded["written_at"]


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


_MODULES: dict[str, ModuleType] = {}


def _load_script(path: Path) -> ModuleType:
    """Import a script as a module (module level only -- never its ``main``)."""
    if path.name not in _MODULES:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULES[path.name] = module
    return _MODULES[path.name]


def _module_constants(path: Path) -> dict[str, object]:
    """Evaluate a script's module-level constants from source, without importing.

    For scripts that do their work at import time (the swr-contrast coefficient
    figure has no ``main``), so importing them would draw the figures. Only
    assignments are evaluated, in order, against a namespace seeded with the
    data-directory constants; anything that needs a name the namespace does not
    carry (a dataframe read, a plot handle) is skipped.
    """
    namespace: dict[str, object] = {
        "PROCESSED_DATA_DIR": PROCESSED_DATA_DIR,
        "P": PROCESSED_DATA_DIR,
        "Path": Path,
    }
    for node in ast.parse(path.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                value = eval(ast.unparse(node.value), namespace)  # noqa: S307
            except Exception:
                continue
            namespace[node.targets[0].id] = value
    return namespace


def _read_table_paths(path: Path, constants: dict[str, object]) -> set[Path]:
    """Every table path the script hands to ``read_table``, from its source."""
    paths = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "read_table"
        ):
            continue
        expression = ast.unparse(node.args[0])
        try:
            paths.add(Path(eval(expression, dict(constants))))  # noqa: S307
        except Exception as error:  # pragma: no cover - drift makes this fire
            raise AssertionError(
                f"{path.name}: cannot resolve read_table({expression}) against the "
                "module constants, so the provenance mapping cannot be checked "
                "against it -- read the table through a module-level constant."
            ) from error
    return paths


# Each fitter and the readers of the artifacts it writes, with the module-level
# name each one declares its input mapping under.
FITTERS = [
    "fit_per_bin_rate_models",
    "fit_per_stop_rate_models",
    "fit_swr_contrast_rate",
    "fit_uncertainty_rate",
]
# (reader, name of its input mapping, the fitter that writes what it reads).
# ``plot_swr_contrast_coefs`` draws its figures at import, so its constants are
# read from source rather than imported.
READERS = [
    (
        RATE / "summarize_per_bin_rate_models.py",
        "ALL_MODELS_INPUTS",
        "fit_per_bin_rate_models",
    ),
    (
        RATE / "plot_per_bin_rate_models.py",
        "ALL_MODELS_INPUTS",
        "fit_per_bin_rate_models",
    ),
    (RATE / "plot_model_checking.py", "ALL_MODELS_INPUTS", "fit_per_bin_rate_models"),
    (
        RATE / "plot_per_bin_rate_summary.py",
        "ALL_MODELS_INPUTS",
        "fit_per_bin_rate_models",
    ),
    (
        RATE / "plot_uncertainty_rate.py",
        "UNCERTAINTY_RATE_INPUTS",
        "fit_uncertainty_rate",
    ),
    (
        FIGURES / "plot_swr_contrast_coefs.py",
        "SWR_CONTRAST_RATE_INPUTS",
        "fit_swr_contrast_rate",
    ),
]
RUNS_AT_IMPORT = {"plot_swr_contrast_coefs.py"}


@pytest.mark.parametrize(
    "reader,mapping,fitter", READERS, ids=lambda v: getattr(v, "stem", v)
)
def test_readers_check_the_same_inputs_the_fitter_recorded(reader, mapping, fitter):
    """A reader checking a different table set than the fitter recorded is blind."""
    expected = _load_script(RATE / f"{fitter}.py").FIT_INPUTS
    declared = (
        _module_constants(reader)
        if reader.name in RUNS_AT_IMPORT
        else vars(_load_script(reader))
    )

    assert declared[mapping] == expected


@pytest.mark.parametrize(
    "reader",
    [
        "summarize_per_bin_rate_models",
        "plot_per_bin_rate_models",
        "plot_model_checking",
        "plot_per_bin_rate_summary",
        "plot_uncertainty_rate",
    ],
)
def test_readers_default_to_refusing_a_stale_fit(reader):
    module = _load_script(RATE / f"{reader}.py")

    assert module.build_parser().parse_args([]).allow_stale is False
    assert module.build_parser().parse_args(["--allow-stale"]).allow_stale is True


@pytest.mark.parametrize(
    "reader",
    [
        "summarize_per_bin_rate_models",
        "plot_per_bin_rate_models",
        "plot_model_checking",
        "plot_per_bin_rate_summary",
    ],
)
def test_per_bin_readers_gate_the_ladder_artifact(reader):
    module = _load_script(RATE / f"{reader}.py")

    assert module.ALL_MODELS == module.RC / "models_per_bin_rate" / "all_models.pkl"


def test_contrast_figure_gates_the_rate_artifact():
    constants = _module_constants(FIGURES / "plot_swr_contrast_coefs.py")
    fitter = _load_script(RATE / "fit_swr_contrast_rate.py")

    assert constants["SWR_CONTRAST_RATE"] == (
        fitter.RC / "models_per_bin_rate" / "swr_contrast_rate.pkl"
    )


@pytest.mark.parametrize("fitter", FITTERS)
def test_fitters_record_every_table_they_read(fitter):
    """The declared mapping is the whole set of tables the runner loads.

    A new ``read_table`` that never reaches ``FIT_INPUTS`` would leave an input
    out of the sidecar, so the fit would pass the reader gate after that table
    was regenerated.
    """
    script = RATE / f"{fitter}.py"
    constants = _module_constants(script)

    assert _read_table_paths(script, constants) == set(constants["FIT_INPUTS"].values())
