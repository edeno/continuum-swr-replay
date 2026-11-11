"""The record of the code that built a summary, and the release check that reads it.

A summary records the files its build depended on. The record must say when that
code has really changed since (not when only a docstring or comment did), say when
the build ran from uncommitted or untracked code and keep a diff that applies,
survive git failing, and round-trip through the summary file. These run in a
throwaway git repository.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
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
from continuum_swr_replay.reported.values import file_record

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


def test_when_git_cannot_say_the_record_gives_git_s_message_and_the_build_is_kept(
    tmp_path,
):
    (tmp_path / "uv.lock").write_text("lock\n")
    (tmp_path / "mod.py").write_text("X = 1\n")
    record = try_code_record(tmp_path, [tmp_path / "mod.py"])
    assert set(record) == {"error"}
    assert "not a git repository" in record["error"]
    assert staleness(_summary(code=record), tmp_path) == [
        f"no code record: {record['error']}"
    ]
    with pytest.raises(ValueError, match="not a code record"):
        code_changes(record, tmp_path)


def test_a_user_s_external_diff_tool_does_not_hide_uncommitted_code(repo):
    module = repo / "pkg" / "mod.py"
    _git(repo, "config", "diff.external", "true")
    _git(repo, "config", "color.ui", "always")
    edited = SOURCE.replace("FACTOR = 2", "FACTOR = 3")
    module.write_text(edited)
    record = code_record(repo, [module])
    assert record["uncommitted"] is True
    _git(repo, "checkout", "-q", "--", "pkg/mod.py")
    (repo / "record.patch").write_text(record["diff"])
    _git(repo, "apply", "record.patch")
    assert module.read_text() == edited


def test_an_uncommitted_lock_is_uncommitted_and_in_the_diff(repo):
    (repo / "uv.lock").write_text("lock 2\n")
    record = code_record(repo, [repo / "pkg" / "mod.py"])
    assert record["uncommitted"] is True
    assert "+lock 2" in record["diff"]


def test_untracked_files_and_a_diff_are_both_named(repo):
    module = repo / "pkg" / "mod.py"
    module.write_text(SOURCE.replace("FACTOR = 2", "FACTOR = 3"))
    new = repo / "pkg" / "new.py"
    new.write_text("Z = 1\n")
    record = code_record(repo, [module, new])
    assert staleness(_summary(code=record), repo) == [
        "built from files git does not track: pkg/new.py",
        "built from uncommitted code (the diff is in the record)",
    ]


def test_a_module_that_no_longer_parses_counts_as_changed(repo):
    module = repo / "pkg" / "mod.py"
    record = code_record(repo, [module])
    module.write_text("X = (\n")
    assert code_changes(record, repo) == ["pkg/mod.py"]


def test_a_changed_data_file_is_a_change(repo):
    cohort = repo / "pkg" / "cohort.csv"
    cohort.write_text("epoch,kept\na,1\n")
    _git(repo, "add", "pkg/cohort.csv")
    _git(repo, "commit", "-q", "-m", "cohort")
    record = code_record(repo, [cohort])
    assert (
        record["files"]["pkg/cohort.csv"]["code_sha256"]
        == (record["files"]["pkg/cohort.csv"]["sha256"])
    )
    cohort.write_text("epoch,kept\na,0\n")
    assert code_changes(record, repo) == ["pkg/cohort.csv"]


def test_a_build_started_after_the_last_edit_lists_nothing_modified(repo):
    module = repo / "pkg" / "mod.py"
    later = code_record(repo, [module], started=module.stat().st_mtime + 1)
    assert later["modified_during_build"] == []
    earlier = code_record(repo, [module], started=module.stat().st_mtime - 1)
    assert staleness(_summary(code=earlier), repo) == [
        "files modified while it was built: pkg/mod.py"
    ]


def test_a_record_of_no_files_is_not_current(repo):
    assert staleness(_summary(code=code_record(repo, [])), repo) == [
        "the code record lists no files"
    ]


def _summary(
    code=None, questions=("2d.2",), name="events", inputs=(), migrated=None
) -> values.Summary:
    summary = values.Summary(
        name=name,
        produced_by="by",
        commit="c",
        code=code,
        inputs=list(inputs),
        migrated=migrated,
    )
    summary.add(
        values.Value(
            key="share",
            value=0.5,
            kind="fraction",
            style="pct",
            unit="",
            description="d",
            questions=questions,
            used_in=("paper",),
            source="s",
        )
    )
    return summary


def test_the_code_record_and_a_migration_round_trip(repo, tmp_path):
    record = code_record(repo, [repo / "pkg" / "mod.py"])
    path = tmp_path / "events.json"
    _summary(code=record).write(path)
    assert values.read_summary(path).code == record
    migrated = {"from": "paper/reported/part2.json", "changes": ["renamed"]}
    _summary(migrated=migrated).write(path)
    assert values.read_summary(path).migrated == migrated


@pytest.mark.parametrize("schema", [4, 6])
def test_another_schema_is_refused(tmp_path, schema):
    path = tmp_path / "events.json"
    _summary().write(path)
    payload = json.loads(path.read_text())
    payload["schema_version"] = schema
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="expected 5"):
        values.read_summary(path)


@pytest.mark.parametrize("field", ["code", "migrated"])
def test_a_summary_file_must_say_whether_it_has_a_code_record_and_a_migration(
    tmp_path, field
):
    path = tmp_path / "events.json"
    _summary().write(path)
    payload = json.loads(path.read_text())
    del payload[field]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=f"no {field}"):
        values.read_summary(path)


def test_a_summary_has_a_known_name():
    with pytest.raises(ValueError, match="no summary 'part2'"):
        _summary(name="part2")


def _write_with_an_old_tag(path: Path) -> None:
    """A behavior summary as one built before the question list changed: a value
    tagged with a question id the list no longer has."""
    _summary(name="behavior").write(path)
    path.write_text(path.read_text().replace('"2d.2"', '"12"'))


def _write_all(directory: Path, code=None) -> None:
    for name in values.SUMMARIES:
        _summary(code=code, name=name).write(directory / f"{name}.json")


def test_the_summaries_are_exactly_one_file_each(tmp_path):
    _write_all(tmp_path)
    assert list(values.read_summaries(tmp_path)) == list(values.SUMMARIES)
    (tmp_path / "events_old.json").write_text((tmp_path / "events.json").read_text())
    with pytest.raises(ValueError, match="unexpected \\['events_old.json'\\]"):
        values.read_summaries(tmp_path)
    (tmp_path / "events_old.json").unlink()
    (tmp_path / "mpfc.json").unlink()
    with pytest.raises(ValueError, match="missing \\['mpfc.json'\\]"):
        values.read_summaries(tmp_path)
    _summary(name="behavior").write(tmp_path / "mpfc.json")
    with pytest.raises(ValueError, match="holds summary behavior"):
        values.read_summaries(tmp_path)


def test_staleness_names_each_reason_and_the_release_check_adds_bad_tags(repo):
    module = repo / "pkg" / "mod.py"
    assert staleness(_summary(), repo) == ["no code record"]
    migrated = _summary(migrated={"from": "paper/reported/part2.json"})
    assert staleness(migrated, repo) == [
        "migrated from paper/reported/part2.json, not rebuilt: no code record"
    ]
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
    problems, _ = release_problems(
        {
            "events": current,
            "behavior": _summary(code=current.code, questions=("12",), name="behavior"),
        },
        repo,
    )
    assert problems == {"behavior": ["1 value with tag problems: share"]}


def test_the_release_check_compares_single_file_inputs_and_notes_file_sets(repo):
    table = repo / "table.csv"
    table.write_text("a\n1\n")
    inputs = [
        file_record(table, repo),
        {"path": "gone.csv", "sha256": "0"},
        {"path": "runs/*.parquet", "n_files": 3, "sha256": "0"},
    ]
    summary = _summary(code=code_record(repo, [repo / "pkg" / "mod.py"]), inputs=inputs)
    problems, notes = release_problems({"events": summary}, repo)
    assert problems == {"events": ["input not on this machine: gone.csv"]}
    assert notes == {"events": ["input file set not re-checked: runs/*.parquet"]}
    table.write_text("a\n2\n")
    problems, _ = release_problems({"events": summary}, repo)
    assert "input changed since the build: table.csv" in problems["events"]
    problems, notes = release_problems({"events": summary}, repo, check_inputs=False)
    assert problems == {}
    assert notes == {"events": ["3 inputs not checked (--code-only)"]}


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_script_{name}", ROOT / "scripts" / "reported" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_release_exits_0_only_when_every_summary_is_current(
    repo, monkeypatch, capsys
):
    script = _load_script("check_release")
    summaries = repo / "reported"
    summaries.mkdir()
    monkeypatch.setattr(script, "SUMMARY_DIR", summaries)
    monkeypatch.setattr(script, "ROOT_DIR", repo)
    _write_all(summaries, code=code_record(repo, [repo / "pkg" / "mod.py"]))
    assert script.main([]) == 0
    assert "all 5 summaries are current" in capsys.readouterr().out

    _write_with_an_old_tag(summaries / "behavior.json")
    assert script.main([]) == 1
    out = capsys.readouterr().out
    assert "behavior.json:" in out and "no code record" in out
    assert "1 value with tag problems: share" in out

    (summaries / "mpfc.json").unlink()
    assert script.main([]) == 1
    assert "missing ['mpfc.json']" in capsys.readouterr().out


def test_the_emitter_names_old_summaries_and_writes_nothing_before_it_can_fail(
    repo, monkeypatch, capsys
):
    script = _load_script("emit_reported_values")
    reported = repo / "reported"
    reported.mkdir()
    for name, path in {
        "SUMMARY_DIR": reported,
        "TEX_PATH": repo / "values.tex",
        "CHANGES_PATH": reported / "changes.md",
        "QUESTIONS_PATH": reported / "questions.md",
        "DOC_ROOTS": (repo / "docs",),
        "ROOT_DIR": repo,
    }.items():
        monkeypatch.setattr(script, name, path)
    (repo / "docs").mkdir()
    doc = repo / "docs" / "note.md"
    doc.write_text("Share <!--v:events.share-->99 %<!--/v-->.\n")
    _write_all(reported, code=code_record(repo, [repo / "pkg" / "mod.py"]))
    _write_with_an_old_tag(reported / "behavior.json")
    script.main([])
    out = capsys.readouterr().out
    assert "behavior.json: 1 values with tag problems" in out
    assert "behavior.json: no code record" in out
    assert (reported / "questions.md").is_file()
    assert "99 %" not in doc.read_text()
    assert "99 %" in (reported / "changes.md").read_text()

    doc.write_text(
        "<!--v:events.share-->50 %<!--/v--> <!--v:events.gone-->1<!--/v-->\n"
    )
    before = (repo / "values.tex").read_text()
    with pytest.raises(KeyError, match="events.gone"):
        script.main([])
    assert "50 %" in doc.read_text()
    assert (repo / "values.tex").read_text() == before


def test_summarize_records_only_what_the_survey_s_build_used(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-P",
            str(ROOT / "scripts" / "reported" / "summarize.py"),
            "--summary",
            "survey",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = values.read_summary(tmp_path / "survey.json")
    files = set(summary.code["files"])
    package = "src/continuum_swr_replay"
    assert "scripts/reported/summarize.py" in files
    assert f"{package}/reported/survey.py" in files
    assert f"{package}/data_loaders/epoch_register.csv" in files
    others = {name for name in values.SUMMARIES if name != "survey"}
    assert not {f"{package}/reported/{name}.py" for name in others} & files
    assert (
        not {
            f"{package}/reported/questions.py",
            f"{package}/reported/code_record.py",
        }
        & files
    )
    assert code_changes(summary.code, ROOT) == []
