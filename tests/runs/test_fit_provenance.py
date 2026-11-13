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
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay_models.content import fit_outputs
from continuum_swr_replay.analysis.replay_models.contrast_aggregation import (
    contrast_fit_inputs,
)
from continuum_swr_replay.paths import model_dir, table_path
from continuum_swr_replay.runs import fit_provenance
from continuum_swr_replay.runs.fit_provenance import (
    check_fit_provenance,
    fit_provenance_path,
    git_hash,
    load_fit_time_sd_reference,
    load_per_sd_scale,
    matching_sidecar,
    record_inputs,
    recorded_input_paths,
    remove_bundle,
    require_fresh_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.tables.io import write_table

REPO_ROOT = Path(__file__).resolve().parents[2]
RATE = REPO_ROOT / "scripts" / "replay_models" / "rate"
CONTENT = REPO_ROOT / "scripts" / "replay_models" / "content"
FIGURES = REPO_ROOT / "scripts" / "figures"
CONTRAST_AGGREGATION = (
    REPO_ROOT
    / "src"
    / "continuum_swr_replay"
    / "analysis"
    / "replay_models"
    / "contrast_aggregation.py"
)


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
        write_table(frame, tmp_path / f"{name}.parquet")
    (tmp_path / "all_models.pkl").write_bytes(b"fitted-model-bytes")
    return tmp_path


@pytest.fixture()
def inputs(fit_dir: Path) -> dict[str, Path]:
    return {
        "stop_table": fit_dir / "stop_table.parquet",
        "event_table": fit_dir / "event_table.parquet",
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


def test_sidecar_records_extra_facts_about_how_the_fit_was_produced(fit_dir, inputs):
    """A runner can state a fit-shaping fact the standard fields cannot carry."""
    sidecar = write_fit_provenance(
        fit_dir / "all_models.pkl",
        inputs,
        "conditional_poisson_fraction",
        ["fit.py", "--leaf-only"],
        extra={"response_renormalized": True},
    )

    record = json.loads(sidecar.read_text())
    assert record["response_renormalized"] is True
    assert record["estimator"] == "conditional_poisson_fraction"


def test_sidecar_refuses_extras_that_would_overwrite_its_own_fields(fit_dir, inputs):
    """Silently replacing the estimator or the input hashes would falsify it."""
    with pytest.raises(ValueError, match="estimator"):
        write_fit_provenance(
            fit_dir / "all_models.pkl",
            inputs,
            "poisson_glm",
            ["fit.py"],
            extra={"estimator": "something_else", "response_renormalized": True},
        )
    assert not fit_provenance_path(fit_dir / "all_models.pkl").exists()


def test_a_sidecar_is_strict_json_and_a_refused_one_leaves_the_old_one(fit_dir, inputs):
    """Written sorted, indented and newline-ended; NaN is refused, and the
    sidecar already there is not replaced by half a file."""
    artifact = fit_dir / "all_models.pkl"
    sidecar = write_fit_provenance(artifact, inputs, "poisson_glm", ["fit.py"])
    text = sidecar.read_text()
    assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"

    with pytest.raises(ValueError):
        write_fit_provenance(
            artifact, inputs, "poisson_glm", ["fit.py"], extra={"x": float("nan")}
        )
    assert sidecar.read_text() == text


def test_a_pkl_input_is_the_fitted_bundle_hashed_as_it_is(fit_dir):
    """An aggregate-only run reads a fitted bundle, which stays pickled."""
    bundle = fit_dir / "swr_contrast_rate.pkl"
    pd.DataFrame({"coef": [0.1]}).to_pickle(bundle)

    write_fit_provenance(
        fit_dir / "all_models.pkl",
        {"swr_contrast_rate": bundle},
        "poisson_glm",
        ["fit.py"],
    )

    recorded = json.loads(fit_provenance_path(fit_dir / "all_models.pkl").read_text())
    assert recorded["inputs"]["swr_contrast_rate"]["sha256"] == _sha256(bundle)
    assert recorded["inputs"]["swr_contrast_rate"]["covariate_schema_version"] is None


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


_GIT_CONFIG = [
    *("-c", "user.name=test", "-c", "user.email=test@example.com"),
    *("-c", "commit.gpgsign=false"),
]


def _git_repo(path: Path) -> Callable[..., str]:
    """Commit one file in a new git repo at `path`; return a git runner for it."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=path, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q")
    (path / "tracked.txt").write_text("tracked")
    git("add", "tracked.txt")
    git(*_GIT_CONFIG, "commit", "-q", "-m", "tracked")
    return git


def _load_copy_in(repo: Path) -> ModuleType:
    """Import a copy of ``fit_provenance`` placed inside `repo`, as a runner would.

    The copy sits at the real module's depth below the repo root, so its
    ``_REPO_ROOT`` is `repo`: what it reads of the checkout when it is imported
    is `repo` at that moment. The copy is untracked, so it does not make the
    tree dirty.
    """
    target = repo / "src" / "continuum_swr_replay" / "runs"
    target.mkdir(parents=True)
    target = target / "fit_provenance.py"
    target.write_bytes(Path(fit_provenance.__file__).read_bytes())
    spec = importlib.util.spec_from_file_location("fit_provenance_copy", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recorded_commit(module: ModuleType, tmp_path: Path) -> str:
    """The commit a sidecar written now by `module` records."""
    sidecar = module.write_fit_provenance(
        tmp_path / "all_models.pkl", {}, "poisson_glm", ["fit.py"]
    )
    return json.loads(sidecar.read_text())["git_hash"]


def test_sidecar_records_the_commit_loaded_not_a_later_head(tmp_path):
    """A long run in a checkout that gets a new commit still ran the old code."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git = _git_repo(repo)
    loaded = git("rev-parse", "HEAD")
    module = _load_copy_in(repo)

    (repo / "tracked.txt").write_text("next commit")
    git(*_GIT_CONFIG, "commit", "-q", "-am", "next")
    assert git("rev-parse", "HEAD") != loaded

    assert _recorded_commit(module, tmp_path) == loaded


def test_sidecar_ignores_a_tracked_file_edited_after_import(tmp_path):
    """An edit made after the code was loaded did not change how it ran."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git = _git_repo(repo)
    loaded = git("rev-parse", "HEAD")
    module = _load_copy_in(repo)

    (repo / "tracked.txt").write_text("edited after import")
    assert git("status", "--porcelain", "--untracked-files=no") != ""

    assert _recorded_commit(module, tmp_path) == loaded


def test_sidecar_records_a_tree_dirty_at_import_as_dirty(tmp_path):
    """The run loaded uncommitted code, even if the edit is undone before it ends."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git = _git_repo(repo)
    loaded = git("rev-parse", "HEAD")
    (repo / "tracked.txt").write_text("edited before import")
    module = _load_copy_in(repo)

    git("checkout", "--", "tracked.txt")
    assert git("status", "--porcelain", "--untracked-files=no") == ""

    assert _recorded_commit(module, tmp_path) == f"{loaded}-dirty"
    assert module.git_hash() == f"{loaded}-dirty"


def test_sidecar_records_a_commit_passed_in(fit_dir, inputs):
    """A worker that imported the code after its parent records the parent's."""
    parent = "0123456789abcdef0123456789abcdef01234567-dirty"
    sidecar = write_fit_provenance(
        fit_dir / "all_models.pkl", inputs, "poisson_glm", ["fit.py"], commit=parent
    )

    assert json.loads(sidecar.read_text())["git_hash"] == parent


@pytest.mark.parametrize(
    "no_git, cause",
    [("not_a_repo", subprocess.CalledProcessError), ("no_git_binary", OSError)],
)
def test_the_module_imports_outside_a_git_checkout(
    tmp_path, monkeypatch, no_git, cause
):
    """Readers import fit_provenance without git; only a sidecar needs a commit."""
    # git must not find a repository above tmp_path either
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    if no_git == "no_git_binary":
        monkeypatch.setenv("PATH", "")

    module = _load_copy_in(tmp_path)

    with pytest.raises(RuntimeError, match="could not read the git checkout") as error:
        module.git_hash()
    assert isinstance(error.value.__cause__, cause)
    with pytest.raises(RuntimeError, match="could not read the git checkout"):
        module.write_fit_provenance(tmp_path / "fit.pkl", {}, "poisson_glm", ["fit.py"])
    sidecar = module.write_fit_provenance(
        tmp_path / "fit.pkl", {}, "poisson_glm", ["fit.py"], commit="parent-commit"
    )
    assert json.loads(sidecar.read_text())["git_hash"] == "parent-commit"


def test_git_hash_never_writes_the_git_index(tmp_path):
    """By default ``git status`` refreshes the index and writes it back under a
    lock. The Widloski runner records its commit in a checkout that other work
    commits in, so, like ``runs.provenance.provenance_record``, it must not write.
    """
    git = _git_repo(tmp_path)
    tracked = tmp_path / "tracked.txt"
    stat = tracked.stat()
    # A new mtime with the same content: the index's cached stat is now stale.
    os.utime(tracked, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    index = tmp_path / ".git" / "index"
    before = _sha256(index)

    assert _load_copy_in(tmp_path).git_hash() == git("rev-parse", "HEAD")
    assert _sha256(index) == before

    # The control: a plain status does refresh and rewrite this index.
    git("status", "--porcelain")
    assert _sha256(index) != before


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


def _read_table_keys(path: Path, mappings: set[str]) -> set[str]:
    """The key of every ``read_table(<mapping>["key"])`` call in ``path``.

    Every ``read_table`` must read through one of the named `mappings`, so a
    table the script reads is a table its provenance mapping names.
    """
    keys = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "read_table"
        ):
            continue
        argument = node.args[0]
        if not (
            isinstance(argument, ast.Subscript)
            and isinstance(argument.value, ast.Name)
            and argument.value.id in mappings
            and isinstance(argument.slice, ast.Constant)
        ):  # pragma: no cover - drift makes this fire
            raise AssertionError(
                f"{path.name}: read_table({ast.unparse(argument)}) does not read "
                f"through the provenance mapping ({sorted(mappings)}), so the "
                "mapping cannot be checked against it."
            )
        keys.add(argument.slice.value)
    return keys


def _names_bound_to(path: Path, function: str) -> set[str]:
    """The names ``path`` assigns from a call to ``function()``."""
    return {
        target.id
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == function
        for target in node.targets
        if isinstance(target, ast.Name)
    }


# Each fitter and the readers of the artifacts it writes. A fitter declares its
# input mapping as ``fit_input_paths()``. The content contrast runner is not
# listed: its tables are read in ``contrast_aggregation``'s assemblers, and
# ``test_content_contrast_fits_record_every_table_they_read`` covers it.
FITTERS = [
    RATE / "fit_per_bin_rate_models.py",
    RATE / "fit_per_stop_rate_models.py",
    RATE / "fit_swr_contrast_rate.py",
    RATE / "fit_uncertainty_rate.py",
]
# (reader, the function that returns its input mapping, the fitter that writes
# what it reads: a script, whose ``fit_input_paths()`` it records, or the
# function that returns the recorded mapping).
READERS = [
    (
        RATE / "plot_per_bin_rate_models.py",
        "all_models_input_paths",
        RATE / "fit_per_bin_rate_models.py",
    ),
    (
        RATE / "plot_model_checking.py",
        "all_models_input_paths",
        RATE / "fit_per_bin_rate_models.py",
    ),
    (
        RATE / "plot_per_bin_rate_summary.py",
        "all_models_input_paths",
        RATE / "fit_per_bin_rate_models.py",
    ),
    (
        RATE / "plot_uncertainty_rate.py",
        "_fit.fit_input_paths",
        RATE / "fit_uncertainty_rate.py",
    ),
    (
        FIGURES / "plot_swr_contrast_coefs.py",
        "rate_input_paths",
        RATE / "fit_swr_contrast_rate.py",
    ),
    (
        FIGURES / "plot_swr_contrast_coefs.py",
        "content_input_paths",
        contrast_fit_inputs,
    ),
]


@pytest.mark.parametrize(
    "reader,mapping,fitter",
    READERS,
    ids=lambda v: getattr(v, "stem", getattr(v, "__name__", v)),
)
def test_readers_check_the_same_inputs_the_fitter_recorded(
    reader, mapping, fitter, processed_data_root
):
    """A reader checking a different table set than the fitter recorded is blind."""
    expected = (
        _load_script(fitter).fit_input_paths() if isinstance(fitter, Path) else fitter()
    )
    declared = eval(f"module.{mapping}()", {"module": _load_script(reader)})

    assert declared == expected
    assert all(path.is_relative_to(processed_data_root) for path in declared.values())


@pytest.mark.parametrize(
    "reader",
    [
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
        "plot_per_bin_rate_models",
        "plot_model_checking",
        "plot_per_bin_rate_summary",
    ],
)
def test_per_bin_readers_gate_the_ladder_artifact(reader, replay_content_root):
    module = _load_script(RATE / f"{reader}.py")

    assert module.all_models_path() == (
        replay_content_root / "models_per_bin_rate" / "all_models.pkl"
    )


def test_contrast_figure_gates_the_rate_artifact(replay_content_root):
    figure = _load_script(FIGURES / "plot_swr_contrast_coefs.py")
    fitter = (RATE / "fit_swr_contrast_rate.py").read_text()

    assert figure.fit_artifact_paths()["rate"] == (
        replay_content_root / "models_per_bin_rate" / "swr_contrast_rate.pkl"
    )
    # the fitter writes the per-session fits there
    assert 'out_dir = model_dir("per_bin_rate")' in fitter
    assert 'out_dir / f"swr_contrast_rate{csfx}.pkl"' in fitter


def test_contrast_figure_gates_both_content_artifacts(replay_content_root):
    """The content half draws from two pickles; both are gated, not just one."""
    artifacts = _load_script(
        FIGURES / "plot_swr_contrast_coefs.py"
    ).fit_artifact_paths()
    fitter = (CONTENT / "fit_swr_contrast_content.py").read_text()

    assert artifacts["content_arm"] == (
        replay_content_root / "models_swr_contrast" / "swr_contrast_content_arm.pkl"
    )
    assert artifacts["content_patch"] == (
        replay_content_root / "models_swr_contrast" / "swr_contrast_content_patch.pkl"
    )
    assert 'model_dir("swr_contrast") / f"swr_contrast_content_arm{sfx}.pkl"' in fitter
    assert (
        'model_dir("swr_contrast") / f"swr_contrast_content_patch{sfx}.pkl"' in fitter
    )


def test_content_contrast_fits_live_outside_the_arm_ladder_bundle():
    """The arm ladder publishes ``content_arm`` by replacing the directory whole,
    so a contrast fit written there was deleted by the next ladder refit."""
    source = (CONTENT / "fit_swr_contrast_content.py").read_text()
    contrast, arm = model_dir("swr_contrast"), model_dir("content_arm")

    assert 'model_dir("content_arm"' not in source
    assert 'model_dir("swr_contrast")' in source
    assert not contrast.is_relative_to(arm) and not arm.is_relative_to(contrast)


def test_content_ladder_readers_check_the_table_the_ladder_records(
    replay_content_root,
):
    """The ladder resolves its input from the CLI, so readers rebuild the map.

    ``fit_replay_content_models.py`` reads one candidate table whose path
    depends on the event source and hierarchy, so it has no ``fit_input_paths``.
    Its readers declare the canonical arm mapping instead, and that declaration
    is checked against the helper the fitter itself uses.
    """
    fitter = _load_script(CONTENT / "fit_replay_content_models.py")
    expected = fitter.content_fit_inputs(replay_content_root, "event_arm_table")
    figures = _load_script(CONTENT / "generate_replay_content_figures.py")
    uncertainty = _load_script(CONTENT / "analyze_uncertainty_content.py")
    data_dir = uncertainty.build_parser().parse_args([]).data_dir

    assert figures.all_models_input_paths() == expected
    assert uncertainty.mu_inputs(data_dir) == expected


def test_content_ladder_readers_gate_the_canonical_bundle(replay_content_root):
    """The gated artifacts live in the directory a full fraction run publishes."""
    figures = _load_script(CONTENT / "generate_replay_content_figures.py")
    bundle = replay_content_root / (
        fit_outputs.canonical_output_subdir("non_local", "HPC", "arm", "all", "timebin")
    )

    assert figures.bundle_dir() == bundle
    assert (
        'all_models_path = models_dir / "all_models.pkl"'
        in (CONTENT / "generate_replay_content_figures.py").read_text()
    )


def test_uncertainty_reader_defaults_to_the_bundle_the_belief_chain_writes():
    """``Mu`` lives in the leaf-only partial-ladder bundle, not ``models/``.

    The expected directory is resolved by running the fitter's own CLI
    resolution for the documented invocation, so the reader's default tracks
    wherever the runner publishes the chain.
    """
    fitter = _load_script(CONTENT / "fit_replay_content_models.py")
    uncertainty = _load_script(CONTENT / "analyze_uncertainty_content.py")
    runner_args = fitter._build_arg_parser().parse_args(
        ["--leaf-only", "--models", ",".join(fit_outputs.BELIEF_CHAIN_MODEL_NAMES)]
    )
    expected = fitter._resolve_run_config(runner_args).output_dir
    data_dir = uncertainty.build_parser().parse_args([]).data_dir

    resolved = uncertainty.resolve_models_dir(data_dir)

    assert resolved == expected
    assert resolved.name != "models"
    assert uncertainty.resolve_models_dir(data_dir, Path("/tmp/mine")) == Path(
        "/tmp/mine"
    )


@pytest.mark.parametrize(
    "reader",
    [
        CONTENT / "generate_replay_content_figures.py",
        CONTENT / "analyze_uncertainty_content.py",
    ],
    ids=lambda path: path.stem,
)
def test_content_readers_default_to_refusing_a_stale_fit(reader):
    module = _load_script(reader)

    assert module.build_parser().parse_args([]).allow_stale is False
    assert module.build_parser().parse_args(["--allow-stale"]).allow_stale is True


@pytest.mark.parametrize("fitter", FITTERS, ids=lambda path: path.stem)
def test_fitters_record_every_table_they_read(fitter):
    """The declared mapping is the whole set of tables the runner loads.

    A new ``read_table`` that never reaches ``fit_input_paths()`` would leave an
    input out of the sidecar, so the fit would pass the reader gate after that
    table was regenerated.
    """
    mappings = _names_bound_to(fitter, "fit_input_paths")

    assert mappings
    assert _read_table_keys(fitter, mappings) == set(
        _load_script(fitter).fit_input_paths()
    )


def _mapping_arguments(path: Path) -> dict[str, list[str]]:
    """The mapping each ``record_inputs`` / ``assemble_*`` call in ``path`` is handed.

    An assembler called without one reads ``contrast_fit_inputs()`` (pinned by
    ``test_the_assemblers_default_to_the_contrast_fit_inputs``); it is listed
    as that call.
    """
    position = {"record_inputs": 0, "assemble_arm": 0, "assemble_patch": 1}
    found: dict[str, list[str]] = {name: [] for name in position}
    for node in ast.walk(ast.parse(path.read_text())):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in position
        ):
            index = position[node.func.id]
            keyword = [k.value for k in node.keywords if k.arg == "fit_inputs"]
            argument = (
                node.args[index]
                if len(node.args) > index
                else (keyword[0] if keyword else ast.parse("contrast_fit_inputs()"))
            )
            found[node.func.id].append(ast.unparse(argument))
    return found


@pytest.mark.parametrize("assembler", ["assemble_arm", "assemble_patch"])
def test_the_assemblers_default_to_the_contrast_fit_inputs(
    assembler, replay_content_root, monkeypatch
):
    from continuum_swr_replay.analysis.replay_models import contrast_aggregation

    read = []

    def stop(path):
        read.append(path)
        raise _StopReading

    monkeypatch.setattr(contrast_aggregation, "read_table", stop)
    with pytest.raises(_StopReading):
        if assembler == "assemble_arm":
            contrast_aggregation.assemble_arm()
        else:
            contrast_aggregation.assemble_patch(pd.DataFrame())

    table = "event_arm_table" if assembler == "assemble_arm" else "event_patch_table"
    assert read == [contrast_fit_inputs()[table]]
    assert read[0].parent == replay_content_root


class _StopReading(Exception):
    """Raised by a ``read_table`` stand-in at the first read."""


@pytest.mark.parametrize(
    ("script", "assemblers"),
    [
        (CONTENT / "fit_swr_contrast_content.py", ("assemble_arm", "assemble_patch")),
        # compares the arm cohort alone, so it assembles no patch cohort
        (CONTENT / "compare_content_estimators.py", ("assemble_arm",)),
    ],
    ids=lambda value: value.stem if isinstance(value, Path) else None,
)
def test_content_contrast_fits_record_every_table_they_read(script, assemblers):
    """The content contrast tables are read by ``contrast_aggregation``'s
    assemblers, through the mapping they are handed. Each script that fits on
    those cohorts hands them one ``contrast_fit_inputs()``, records that same
    mapping, and reads no table of its own, so a table cannot reach the fit
    without reaching the sidecar."""
    assert _read_table_keys(CONTRAST_AGGREGATION, {"fit_inputs"}) == set(
        contrast_fit_inputs()
    )
    assert _read_table_keys(script, set()) == set()
    (mapping,) = _names_bound_to(script, "contrast_fit_inputs")
    assert _mapping_arguments(script) == {
        "record_inputs": [mapping],
        "assemble_arm": [mapping],
        "assemble_patch": [mapping] if "assemble_patch" in assemblers else [],
    }


def test_data_flow_figure_gates_the_arm_model_it_draws(replay_content_root):
    """The pipeline figure plots fitted M3 coefficients, so it is gated too."""
    figure = _load_script(FIGURES / "generate_supp_pipeline_data_flow.py")
    fitter = _load_script(CONTENT / "fit_replay_content_models.py")
    expected = fitter.content_fit_inputs(replay_content_root, "event_arm_table")

    assert figure.all_models_input_paths() == expected
    assert figure.m3_path() == replay_content_root / "models" / "M3.pkl"
    assert figure.build_parser().parse_args([]).allow_stale is False
    assert figure.build_parser().parse_args(["--allow-stale"]).allow_stale is True


# ── The sweep script ──

QA = REPO_ROOT / "scripts" / "checks"


@pytest.fixture()
def sweep_tree(tmp_path: Path) -> Path:
    """A models tree holding one current fit and one whose input has moved."""
    root = tmp_path / "replay_content"
    models = root / "models_toy"
    models.mkdir(parents=True)
    frame = pd.DataFrame({"stop_id": [0, 1, 2], "count": [1, 0, 4]})
    frame.attrs["covariate_schema_version"] = "v9"
    write_table(frame, root / "stop_table.parquet")
    inputs = {"stop_table": root / "stop_table.parquet"}

    pd.DataFrame({"coef": [0.1]}).to_pickle(models / "stale.pkl")
    write_fit_provenance(models / "stale.pkl", inputs, "poisson_glm", ["fit.py"])
    # Regenerate the table between the two fits, as a rebuild would.
    frame["count"] = [2, 0, 4]
    write_table(frame, root / "stop_table.parquet")
    pd.DataFrame({"coef": [0.2]}).to_pickle(models / "fresh.pkl")
    write_fit_provenance(models / "fresh.pkl", inputs, "poisson_glm", ["fit.py"])
    return root


def _run_sweep(argv: list[str], capsys) -> tuple[int, str]:
    module = _load_script(QA / "check_fit_provenance.py")
    code = module.main(argv)
    return code, capsys.readouterr().out


def test_sweep_separates_the_current_fit_from_the_stale_one(sweep_tree, capsys):
    code, out = _run_sweep(["--root", str(sweep_tree)], capsys)

    assert code == 1
    assert "fresh          models_toy/fresh.pkl" in out
    assert "STALE          models_toy/stale.pkl  (changed: stop_table)" in out
    assert "2 artifact(s): 1 fresh, 1 stale" in out


def test_sweep_exits_zero_when_every_fit_is_current(sweep_tree, capsys):
    (sweep_tree / "models_toy" / "stale.pkl").unlink()
    fit_provenance_path(sweep_tree / "models_toy" / "stale.pkl").unlink()

    code, out = _run_sweep(["--root", str(sweep_tree)], capsys)

    assert code == 0
    assert "1 artifact(s): 1 fresh" in out


def test_sweep_reports_a_fit_with_no_sidecar(sweep_tree, capsys):
    orphan = sweep_tree / "models_toy" / "orphan.pkl"
    orphan.write_bytes(b"a fit nobody recorded")

    code, out = _run_sweep([str(orphan)], capsys)

    assert code == 1
    assert "no sidecar" in out


def test_sweep_json_output_carries_the_stale_input_names(sweep_tree, capsys):
    code, out = _run_sweep(["--root", str(sweep_tree), "--json"], capsys)

    assert code == 1
    by_name = {Path(r["artifact"]).name: r for r in json.loads(out)}
    assert by_name["fresh.pkl"]["status"] == "fresh"
    assert by_name["stale.pkl"]["status"] == "stale"
    assert by_name["stale.pkl"]["stale_inputs"] == ["stop_table"]


def test_sweep_carries_staleness_forward_through_a_derived_fit(sweep_tree, capsys):
    """An aggregate of a stale fit is stale, even though its own bytes match."""
    models = sweep_tree / "models_toy"
    pd.DataFrame({"coef": [0.3]}).to_pickle(models / "audited.pkl")
    write_fit_provenance(
        models / "audited.pkl",
        {"stale": models / "stale.pkl"},
        "aggregate",
        ["fit.py", "--aggregate-only"],
    )

    code, out = _run_sweep(["--root", str(sweep_tree)], capsys)

    assert code == 1
    assert "STALE          models_toy/audited.pkl  (inherited: stale)" in out


def test_sweep_reports_a_sidecar_whose_artifact_is_gone(sweep_tree, capsys):
    """A sidecar outliving its fit certifies a file that is not there."""
    (sweep_tree / "models_toy" / "fresh.pkl").unlink()

    code, out = _run_sweep(["--root", str(sweep_tree)], capsys)

    assert code == 1
    assert "no artifact    models_toy/fresh.pkl" in out
    assert "1 fresh" not in out


def test_sweep_skips_backup_directories(sweep_tree, capsys):
    """Frozen copies have moved on by design; reporting them buries the rest."""
    backup = sweep_tree / "_pre_phase3_backup_20260906" / "models_toy"
    backup.mkdir(parents=True)
    (backup / "stale.pkl").write_bytes(b"frozen")
    write_fit_provenance(
        backup / "stale.pkl",
        {"stop_table": sweep_tree / "stop_table.parquet"},
        "poisson_glm",
        ["fit.py"],
    )

    _, out = _run_sweep(["--root", str(sweep_tree)], capsys)

    assert "_pre_phase3_backup" not in out


def test_a_canonical_input_resolves_at_the_root_not_beside_the_sidecar(tmp_path):
    """A copy or symlink at a table's old place does not shadow the canonical one."""
    module = _load_script(QA / "check_fit_provenance.py")
    root = tmp_path / "replay_content"
    per_bin = root / "models_per_bin_rate"
    per_bin.mkdir(parents=True)
    (root / "qvals_table.parquet").write_bytes(b"canonical")
    (per_bin / "qvals_table.parquet").symlink_to(root / "qvals_table.parquet")
    (per_bin / "swr_contrast_rate.pkl").write_bytes(b"fit")
    sidecar = per_bin / "swr_contrast_rate_audited.provenance.json"

    assert module.resolve_input("qvals_table", sidecar, root) == (
        root / "qvals_table.parquet"
    )
    # A non-canonical input (an aggregate-only run's fit) is the file beside it.
    assert module.resolve_input("swr_contrast_rate", sidecar, root) == (
        per_bin / "swr_contrast_rate.pkl"
    )
    # A non-canonical table (a ContFrag table) is its parquet at the root.
    (root / "event_arm_table_swr_HPC.parquet").write_bytes(b"table")
    assert module.resolve_input("event_arm_table_swr_HPC", sidecar, root) == (
        root / "event_arm_table_swr_HPC.parquet"
    )


def test_matching_sidecar_needs_a_mapping_that_records_the_contract(tmp_path):
    artifact = tmp_path / "bundle.pkl"
    sidecar = fit_provenance_path(artifact)
    assert matching_sidecar(artifact, {}) is None  # no sidecar
    sidecar.write_text("")
    assert matching_sidecar(artifact, {}) is None  # empty
    sidecar.write_text("{not json")
    assert matching_sidecar(artifact, {}) is None
    sidecar.write_text("[1, 2]")
    assert matching_sidecar(artifact, {}) is None  # not a mapping
    sidecar.write_text(json.dumps({"version": 3, "status": "complete"}))
    assert matching_sidecar(artifact, {"version": 3}) == {
        "version": 3,
        "status": "complete",
    }
    assert matching_sidecar(artifact, {"version": 2}) is None
    # an unrecorded key reads as None
    assert matching_sidecar(artifact, {"version": 3, "absent": None}) is not None
    assert matching_sidecar(artifact, {"absent": 0}) is None


def test_remove_bundle_unlinks_sidecar_table_and_companions(tmp_path):
    artifact = tmp_path / "bundle.pkl"
    companions = [tmp_path / "bundle.a.npy", tmp_path / "bundle.b.npy"]
    for path in (artifact, fit_provenance_path(artifact), companions[0]):
        path.write_text("x")
    (tmp_path / "other.pkl").write_text("x")

    remove_bundle(artifact, companions)  # one companion was never written

    assert sorted(path.name for path in tmp_path.iterdir()) == ["other.pkl"]


def test_recorded_input_paths_reads_tables_and_upstream_fits(replay_content_root):
    """A canonical table is read at its replay-content path; any other recorded
    name is an upstream fit beside the artifact."""
    frame = pd.DataFrame({"stop_id": [0, 1], "count": [1, 0]})
    write_table(frame, table_path("stop_table", replay_content_root))
    models = replay_content_root / "models"
    models.mkdir()
    frame.to_pickle(models / "parent_fit.pkl")
    artifact = models / "child_fit.pkl"
    artifact.write_bytes(b"child")
    write_fit_provenance(
        artifact,
        {
            "stop_table": table_path("stop_table", replay_content_root),
            "parent_fit": models / "parent_fit.pkl",
        },
        "poisson_glm",
        ["fit.py"],
    )
    assert recorded_input_paths(artifact) == {
        "parent_fit": models / "parent_fit.pkl",
        "stop_table": table_path("stop_table", replay_content_root),
    }


def test_recorded_input_paths_without_a_sidecar_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        recorded_input_paths(tmp_path / "all_models.pkl")


# ── The per-SD unit a contrast fit records ──


def _per_sd_record(tmp_path: Path, record: dict) -> Path:
    path = tmp_path / "swr_contrast_content_provenance.json"
    path.write_text(json.dumps(record))
    return path


def test_a_record_without_the_frozen_units_is_refused(tmp_path: Path) -> None:
    """A fit in an unknown unit must fail, not fall back to a recomputed SD."""
    with pytest.raises(KeyError, match="per_sd_scale"):
        load_per_sd_scale(_per_sd_record(tmp_path, {"per_sd_scale": {}}), "arm_cohort")
    with pytest.raises(KeyError, match="per_sd_scale"):
        load_per_sd_scale(_per_sd_record(tmp_path, {}), "arm_cohort")
    with pytest.raises(FileNotFoundError):
        load_per_sd_scale(tmp_path / "absent.json", "arm_cohort")


@pytest.mark.parametrize("record", [{}, {"fit_time_sd_reference": {}}])
def test_a_rate_meta_without_the_frozen_units_is_refused(
    record, tmp_path: Path
) -> None:
    path = tmp_path / "swr_contrast_rate_meta.json"
    path.write_text(json.dumps(record))

    with pytest.raises(KeyError, match="fit_time_sd_reference"):
        load_fit_time_sd_reference(path)


def test_an_absent_rate_meta_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_fit_time_sd_reference(tmp_path / "absent.json")
