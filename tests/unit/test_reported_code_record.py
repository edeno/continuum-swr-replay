"""The record of the code that built a summary, and the release check that reads it.

A summary records the files its build depended on. The record must say when that
code has really changed since (not when only a docstring or comment did), say when
the build ran from uncommitted or untracked code and keep a diff that applies,
survive git failing, and round-trip through the summary file. These run in a
throwaway git repository.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from continuum_swr_replay.reported import values
from continuum_swr_replay.reported.code_record import (
    build_files,
    code_changes,
    code_digest,
    code_record,
    try_code_record,
)
from continuum_swr_replay.reported.release import release_problems, staleness

ROOT = Path(__file__).resolve().parents[2]

SOURCE = '''"""A module."""

FACTOR = 2
"""How much f multiplies by."""


def f(x):
    """Multiply x."""
    y = FACTOR * x  # scaled
    a = 1
    b = 2

    return y
'''


def test_the_code_digest_ignores_docstrings_comments_and_layout():
    same = (
        SOURCE.replace('"""Multiply x."""', '"""Scale x, reworded."""')
        .replace('"""How much f multiplies by."""', '"""The factor."""')
        .replace("# scaled", "# times the factor")
        .replace("y = FACTOR * x", "y = (FACTOR *\n         x)")
    )
    assert code_digest(same) == code_digest(SOURCE)
    assert code_digest(SOURCE.replace("FACTOR = 2", "FACTOR = 3")) != code_digest(
        SOURCE
    )


def test_a_build_records_its_modules_and_the_epoch_register_but_not_the_question_list():
    names = {str(p.relative_to(ROOT)) for p in build_files(ROOT)}
    package = "src/continuum_swr_replay"
    assert f"{package}/reported/values.py" in names
    assert f"{package}/data_loaders/epoch_register.csv" in names
    assert f"{package}/reported/questions.py" not in names
    assert f"{package}/reported/code_record.py" not in names


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(SOURCE)
    (tmp_path / "uv.lock").write_text("lock 1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "start")
    return tmp_path


def test_a_record_of_committed_code_is_current_until_the_code_changes(repo):
    module = repo / "pkg" / "mod.py"
    record = code_record(repo, [module])
    assert record["head"] == _git(repo, "rev-parse", "HEAD").strip()
    assert record["uncommitted"] is False
    assert record["diff"] == "" and record["untracked"] == []
    assert list(record["files"]) == ["pkg/mod.py"]
    assert code_changes(record, repo) == []

    module.write_text(SOURCE.replace('"""Multiply x."""', '"""Reworded."""'))
    assert code_changes(record, repo) == [], "a docstring edit is not a change"

    module.write_text(SOURCE.replace("FACTOR = 2", "FACTOR = 3"))
    assert code_changes(record, repo) == ["pkg/mod.py"]


def test_a_changed_lock_or_a_removed_file_is_a_change(repo):
    record = code_record(repo, [repo / "pkg" / "mod.py"])
    (repo / "uv.lock").write_text("lock 2\n")
    (repo / "pkg" / "mod.py").unlink()
    assert code_changes(record, repo) == ["pkg/mod.py", "uv.lock"]


def test_the_diff_of_uncommitted_code_applies_to_the_recorded_commit(repo):
    module = repo / "pkg" / "mod.py"
    edited = SOURCE.replace("y = FACTOR * x", "y = FACTOR * x + 1")
    module.write_text(edited)
    record = code_record(repo, [module])
    assert record["uncommitted"] is True
    assert record["diff"].endswith("\n")
    _git(repo, "checkout", "-q", "--", "pkg/mod.py")
    patch = repo / "record.patch"
    patch.write_text(record["diff"])
    _git(repo, "apply", str(patch))
    assert module.read_text() == edited


def test_an_untracked_module_is_uncommitted_and_named(repo):
    new = repo / "pkg" / "new.py"
    new.write_text("Z = 1\n")
    record = code_record(repo, [repo / "pkg" / "mod.py", new])
    assert record["uncommitted"] is True
    assert record["untracked"] == ["pkg/new.py"]
    assert record["diff"] == ""
    assert staleness(_summary(code=record), repo) == [
        "built from files git does not track: pkg/new.py"
    ]


def test_a_file_modified_after_the_build_started_is_named(repo):
    module = repo / "pkg" / "mod.py"
    started = module.stat().st_mtime - 1
    record = code_record(repo, [module], started=started)
    assert record["modified_during_build"] == ["pkg/mod.py"]
    assert code_record(repo, [module])["modified_during_build"] == []


def test_under_another_python_only_identical_bytes_count_as_unchanged(repo):
    module = repo / "pkg" / "mod.py"
    record = code_record(repo, [module])
    record["python"] = "2.7"
    assert code_changes(record, repo) == []
    module.write_text(SOURCE.replace('"""Multiply x."""', '"""Reworded."""'))
    assert code_changes(record, repo) == ["pkg/mod.py"]


def test_when_git_cannot_say_the_record_says_why_and_the_build_is_kept(tmp_path):
    (tmp_path / "uv.lock").write_text("lock\n")
    (tmp_path / "mod.py").write_text("X = 1\n")
    record = try_code_record(tmp_path, [tmp_path / "mod.py"])
    assert set(record) == {"error"}
    assert staleness(_summary(code=record), tmp_path)[0].startswith("no code record: ")


def _summary(code=None, claims=("15",)) -> values.Summary:
    summary = values.Summary(part=2, produced_by="by", commit="c", code=code)
    with values._tags_unchecked():
        summary.add(
            values.Value(
                key="share",
                value=0.5,
                kind="fraction",
                style="pct",
                unit="",
                description="d",
                claims=claims,
                used_in=("paper",),
                source="s",
            )
        )
    return summary


def test_the_code_record_round_trips_and_an_old_summary_has_none(repo, tmp_path):
    record = code_record(repo, [repo / "pkg" / "mod.py"])
    path = tmp_path / "part2.json"
    _summary(code=record).write(path)
    assert values.read_summary(path).code == record

    old = json.loads(path.read_text())
    old["schema_version"] = 1
    del old["code"]
    path.write_text(json.dumps(old))
    assert values.read_summary(path).code is None


def test_staleness_names_each_reason_and_the_release_check_adds_bad_tags(repo):
    module = repo / "pkg" / "mod.py"
    assert staleness(_summary(), repo) == ["no code record"]
    current = _summary(code=code_record(repo, [module]))
    assert staleness(current, repo) == []
    module.write_text(SOURCE.replace("FACTOR = 2", "FACTOR = 3"))
    dirty = _summary(code=code_record(repo, [module]))
    assert staleness(dirty, repo) == [
        "built from uncommitted code (the diff is in the record)"
    ]
    _git(repo, "commit", "-q", "-am", "triple")
    module.write_text(SOURCE)
    assert staleness(dirty, repo) == [
        "built from uncommitted code (the diff is in the record)",
        "code changed since the build: pkg/mod.py",
    ]
    problems = release_problems(
        {2: current, 3: _summary(code=current.code, claims=("12",))}, repo
    )
    assert problems == {3: ["1 value with tags the question list does not know"]}
