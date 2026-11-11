"""The record of the code that built a summary, and the release check that reads it.

A summary records the modules its builder loaded; the record must say when that
code has really changed since (not when only a docstring or comment did), say when
the build ran from uncommitted code and keep the diff, and round-trip through the
summary file. These run in a throwaway git repository.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from continuum_swr_replay.reported import values
from continuum_swr_replay.reported.code_record import (
    code_changes,
    code_digest,
    code_record,
    loaded_modules,
)
from continuum_swr_replay.reported.release import release_problems, staleness

SOURCE = '''"""A module."""


def f(x):
    """Double x."""
    return 2 * x  # twice
'''


def test_the_code_digest_ignores_docstrings_comments_and_layout():
    same = SOURCE.replace('"""Double x."""', '"""Twice x, reworded."""').replace(
        "# twice", "# times two"
    )
    same = same.replace("return 2 * x", "return (2 *\n        x)")
    assert code_digest(same) == code_digest(SOURCE)
    assert code_digest(SOURCE.replace("2 * x", "3 * x")) != code_digest(SOURCE)


def test_the_loaded_modules_leave_out_the_question_list_and_the_record_itself():
    root = Path(__file__).resolve().parents[2]
    names = {str(p.relative_to(root)) for p in loaded_modules(root)}
    reported = "src/continuum_swr_replay/reported"
    assert f"{reported}/values.py" in names
    assert f"{reported}/questions.py" not in names
    assert f"{reported}/code_record.py" not in names


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


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
    assert record["uncommitted"] is False
    assert record["diff"] == ""
    assert list(record["modules"]) == ["pkg/mod.py"]
    assert code_changes(record, repo) == []

    module.write_text(SOURCE.replace('"""Double x."""', '"""Reworded."""'))
    assert code_changes(record, repo) == [], "a docstring edit is not a change"

    module.write_text(SOURCE.replace("2 * x", "3 * x"))
    assert code_changes(record, repo) == ["pkg/mod.py"]


def test_a_changed_lock_or_a_removed_module_is_a_change(repo):
    record = code_record(repo, [repo / "pkg" / "mod.py"])
    (repo / "uv.lock").write_text("lock 2\n")
    (repo / "pkg" / "mod.py").unlink()
    assert code_changes(record, repo) == ["pkg/mod.py", "uv.lock"]


def test_a_record_of_uncommitted_code_keeps_the_diff(repo):
    module = repo / "pkg" / "mod.py"
    module.write_text(SOURCE.replace("2 * x", "3 * x"))
    record = code_record(repo, [module])
    assert record["uncommitted"] is True
    assert "+    return 3 * x" in record["diff"]
    assert code_changes(record, repo) == []


def test_under_another_python_only_identical_bytes_count_as_unchanged(repo):
    module = repo / "pkg" / "mod.py"
    record = code_record(repo, [module])
    record["python"] = "2.7"
    assert code_changes(record, repo) == []
    module.write_text(SOURCE.replace('"""Double x."""', '"""Reworded."""'))
    assert code_changes(record, repo) == ["pkg/mod.py"]


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
    assert staleness(_summary(), repo) == [
        "no code record: built before code records were kept"
    ]
    current = _summary(code=code_record(repo, [module]))
    assert staleness(current, repo) == []
    module.write_text(SOURCE.replace("2 * x", "3 * x"))
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
    assert problems == {
        3: ["1 value with tags the question list does not know"],
    }
