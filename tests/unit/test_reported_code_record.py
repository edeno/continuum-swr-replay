"""The record of the code that built a summary, and the release check that reads it.

A summary records the files its build depended on. The record must say when that
code has really changed since (not when only a docstring or comment did), name the
files that were uncommitted or untracked when it was built, and round-trip through
the summary file; and no summary is written among the canonical ones from
uncommitted code. These run in a throwaway git repository.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from continuum_swr_replay.reported import emit, values
from continuum_swr_replay.reported.code_record import (
    build_files,
    code_changes,
    code_digest,
    code_record,
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
def repo(tmp_path, monkeypatch):
    """A git repository with one module and a lock file, committed. Git reads no
    user or system configuration, so a user's signing or hook settings cannot
    break it."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
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
    assert record["uncommitted"] == []
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


def test_an_edited_module_an_edited_lock_and_an_untracked_module_are_uncommitted(repo):
    module = repo / "pkg" / "mod.py"
    new = repo / "pkg" / "new.py"
    new.write_text("Z = 1\n")
    assert code_record(repo, [module, new])["uncommitted"] == ["pkg/new.py"]
    module.write_text(SOURCE.replace("y = FACTOR * x", "y = FACTOR * x + 1"))
    assert code_record(repo, [module])["uncommitted"] == ["pkg/mod.py"]
    (repo / "uv.lock").write_text("lock 2\n")
    record = code_record(repo, [module, new])
    assert record["uncommitted"] == ["pkg/mod.py", "pkg/new.py", "uv.lock"]
    assert staleness(_summary(code=record), repo) == [
        "built from uncommitted code: pkg/mod.py, pkg/new.py, uv.lock"
    ]


def test_a_file_the_build_did_not_use_may_be_uncommitted(repo):
    (repo / "notes.md").write_text("draft\n")
    (repo / "pkg" / "other.py").write_text("Z = 1\n")
    assert code_record(repo, [repo / "pkg" / "mod.py"])["uncommitted"] == []


def test_under_another_python_only_identical_bytes_count_as_unchanged(repo):
    module = repo / "pkg" / "mod.py"
    record = code_record(repo, [module])
    record["python"] = "2.7"
    assert code_changes(record, repo) == []
    module.write_text(SOURCE.replace('"""Multiply x."""', '"""Reworded."""'))
    assert code_changes(record, repo) == ["pkg/mod.py"]


def test_when_git_cannot_say_no_record_is_made(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    (tmp_path / "uv.lock").write_text("lock\n")
    (tmp_path / "mod.py").write_text("X = 1\n")
    with pytest.raises(subprocess.CalledProcessError) as failed:
        code_record(tmp_path, [tmp_path / "mod.py"])
    assert "not a git repository" in failed.value.stderr


def test_a_user_s_external_diff_tool_does_not_hide_uncommitted_code(repo):
    module = repo / "pkg" / "mod.py"
    _git(repo, "config", "diff.external", "true")
    _git(repo, "config", "color.ui", "always")
    module.write_text(SOURCE.replace("FACTOR = 2", "FACTOR = 3"))
    assert code_record(repo, [module])["uncommitted"] == ["pkg/mod.py"]


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


def test_a_record_of_no_files_is_not_current(repo):
    assert staleness(_summary(code=code_record(repo, [])), repo) == [
        "the code record lists no files"
    ]


def _summary(
    code=None,
    questions=("2d.2",),
    name="events",
    inputs=(),
    pending=None,
) -> values.Summary:
    summary = values.Summary(
        name=name,
        produced_by="by",
        commit="c",
        code=code,
        inputs=list(inputs),
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
            figures=(),
            source="s",
            function="f",
            pending=pending,
        )
    )
    return summary


def test_the_code_record_round_trips(repo, tmp_path):
    record = code_record(repo, [repo / "pkg" / "mod.py"])
    path = tmp_path / "events.json"
    _summary(code=record).write(path)
    assert values.read_summary(path).code == record


@pytest.mark.parametrize("schema", [5, 7])
def test_another_schema_is_refused(tmp_path, schema):
    path = tmp_path / "events.json"
    _summary().write(path)
    payload = json.loads(path.read_text())
    payload["schema_version"] = schema
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="expected 6"):
        values.read_summary(path)


def test_a_summary_file_must_say_whether_it_has_a_code_record(tmp_path):
    path = tmp_path / "events.json"
    _summary().write(path)
    payload = json.loads(path.read_text())
    del payload["code"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="no code"):
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
    current = _summary(code=code_record(repo, [module]))
    assert staleness(current, repo) == []
    module.write_text(SOURCE.replace("FACTOR = 2", "FACTOR = 3"))
    dirty = _summary(code=code_record(repo, [module]))
    assert staleness(dirty, repo) == ["built from uncommitted code: pkg/mod.py"]
    _git(repo, "commit", "-q", "-am", "triple")
    module.write_text(SOURCE)
    assert staleness(dirty, repo) == [
        "built from uncommitted code: pkg/mod.py",
        "code changed since the build: pkg/mod.py",
    ]
    problems, _ = release_problems(
        {
            "events": current,
            "behavior": _summary(code=current.code, questions=("12",), name="behavior"),
        },
        repo,
        "",
    )
    assert problems == {"behavior": ["1 value with tag problems: share"]}


def test_the_release_check_counts_cited_values_and_refuses_a_cited_pending_one(repo):
    record = code_record(repo, [repo / "pkg" / "mod.py"])
    summaries = {
        "events": _summary(code=record),
        "behavior": _summary(code=record, name="behavior", pending="rerun it"),
    }
    problems, notes = release_problems(
        summaries, repo, "Events \\EventsShare{}.\n% not \\BehaviorShare\n"
    )
    assert problems == {}
    assert notes == {
        "events": ["the manuscript cites 1 of 1 values"],
        "behavior": ["the manuscript cites 0 of 1 values"],
    }
    problems, _ = release_problems(summaries, repo, "\\EventsShare \\BehaviorShare")
    assert problems == {"behavior": ["the manuscript cites pending values: share"]}


def test_the_release_check_compares_single_file_inputs_and_notes_file_sets(repo):
    table = repo / "table.csv"
    table.write_text("a\n1\n")
    inputs = [
        file_record(table, repo),
        {"path": "gone.csv", "sha256": "0"},
        {"path": "runs/*.parquet", "n_files": 3, "sha256": "0"},
    ]
    summary = _summary(code=code_record(repo, [repo / "pkg" / "mod.py"]), inputs=inputs)
    cited = "the manuscript cites 0 of 1 values"
    problems, notes = release_problems({"events": summary}, repo, "")
    assert problems == {"events": ["input not on this machine: gone.csv"]}
    assert notes == {"events": [cited, "input file set not re-checked: runs/*.parquet"]}
    table.write_text("a\n2\n")
    problems, _ = release_problems({"events": summary}, repo, "")
    assert "input changed since the build: table.csv" in problems["events"]
    problems, notes = release_problems(
        {"events": summary}, repo, "", check_inputs=False
    )
    assert problems == {}
    assert notes == {"events": [cited, "3 inputs not checked (--code-only)"]}


def _uncommitted(repo: Path) -> dict:
    module = repo / "pkg" / "mod.py"
    module.write_text(SOURCE.replace("FACTOR = 2", "FACTOR = 3"))
    return code_record(repo, [module])


@pytest.mark.parametrize(
    "make_code, cell",
    [
        (lambda repo: None, "none"),
        (lambda repo: code_record(repo, [repo / "pkg" / "mod.py"]), "yes"),
        (_uncommitted, "yes, uncommitted"),
    ],
    ids=["none", "committed", "uncommitted"],
)
def test_the_view_s_summary_table_says_what_each_code_record_is(repo, make_code, cell):
    summary = _summary(code=make_code(repo))
    view = emit.questions_view({"events": summary})
    assert f"| `events.json` | `c` | {cell} | 1 | 0 |" in view


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
    monkeypatch.setattr(script, "MANUSCRIPT", _manuscript(repo))
    _write_all(summaries, code=code_record(repo, [repo / "pkg" / "mod.py"]))
    assert script.main([]) == 0
    out = capsys.readouterr().out
    assert "all 5 summaries are current" in out
    assert "the manuscript cites 1 of 1 values" in out

    _write_with_an_old_tag(summaries / "behavior.json")
    assert script.main([]) == 1
    out = capsys.readouterr().out
    assert "behavior.json:" in out and "no code record" in out
    assert "1 value with tag problems: share" in out

    (summaries / "mpfc.json").unlink()
    assert script.main([]) == 1
    assert "missing ['mpfc.json']" in capsys.readouterr().out


def test_check_release_checks_the_inputs_unless_told_code_only(
    repo, monkeypatch, capsys
):
    script = _load_script("check_release")
    summaries = repo / "reported"
    summaries.mkdir()
    monkeypatch.setattr(script, "SUMMARY_DIR", summaries)
    monkeypatch.setattr(script, "ROOT_DIR", repo)
    monkeypatch.setattr(script, "MANUSCRIPT", _manuscript(repo))
    record = code_record(repo, [repo / "pkg" / "mod.py"])
    _write_all(summaries, code=record)
    _summary(code=record, inputs=[{"path": "gone.csv", "sha256": "0"}]).write(
        summaries / "events.json"
    )
    assert script.main([]) == 1
    assert "input not on this machine: gone.csv" in capsys.readouterr().out
    assert script.main(["--code-only"]) == 0
    out = capsys.readouterr().out
    assert "1 inputs not checked (--code-only)" in out
    assert "code, tags and citations checked" in out


def _manuscript(repo: Path) -> Path:
    """A manuscript in ``repo`` that cites the events summary's one value."""
    path = repo / "main.tex"
    path.write_text("Share \\EventsShare{}.\n")
    return path


def _emitter(repo: Path, monkeypatch):
    """The emitter script, pointed at ``repo``: summaries in ``repo/reported``,
    docs in ``repo/docs``, the manuscript ``repo/main.tex``."""
    script = _load_script("emit_reported_values")
    reported = repo / "reported"
    reported.mkdir()
    (repo / "docs").mkdir()
    for name, path in {
        "SUMMARY_DIR": reported,
        "TEX_PATH": repo / "values.tex",
        "MANUSCRIPT": _manuscript(repo),
        "CHANGES_PATH": reported / "changes.md",
        "QUESTIONS_PATH": reported / "questions.md",
        "DOC_ROOTS": (repo / "docs",),
        "ROOT_DIR": repo,
    }.items():
        monkeypatch.setattr(script, name, path)
    return script


def test_the_emitter_names_old_summaries_and_records_changed_numbers(
    repo, monkeypatch, capsys
):
    script = _emitter(repo, monkeypatch)
    reported = repo / "reported"
    doc = repo / "docs" / "note.md"
    doc.write_text("Share <!--v:events.share-->99 %<!--/v-->.\n")
    _write_all(reported, code=code_record(repo, [repo / "pkg" / "mod.py"]))
    _write_with_an_old_tag(reported / "behavior.json")
    script.main([])
    out = capsys.readouterr().out
    assert "behavior.json: 1 values with tag problems" in out
    assert "behavior.json: no code record" in out
    assert "main.tex cites 1 of 5 values" in out
    assert (reported / "questions.md").is_file()
    assert "99 %" not in doc.read_text()
    assert "99 %" in (reported / "changes.md").read_text()


def _break_marker(repo: Path) -> None:
    (repo / "docs" / "b.md").write_text("<!--v:events.gone-->1<!--/v-->\n")


def _break_code_record(repo: Path) -> None:
    path = repo / "reported" / "detection.json"
    payload = json.loads(path.read_text())
    del payload["code"]["files"]
    path.write_text(json.dumps(payload))


@pytest.mark.parametrize(
    "break_it, error",
    [(_break_marker, "events.gone"), (_break_code_record, "files")],
    ids=["a marker naming no key", "a code record with no files"],
)
def test_the_emitter_writes_nothing_when_it_fails(repo, monkeypatch, break_it, error):
    """Every output would change (a new value for the macro file, the view, the
    change log and a doc filled before the failing one), and none is written."""
    script = _emitter(repo, monkeypatch)
    reported = repo / "reported"
    doc = repo / "docs" / "a.md"
    doc.write_text("<!--v:events.share-->99 %<!--/v-->\n")
    _write_all(reported, code=code_record(repo, [repo / "pkg" / "mod.py"]))
    script.main([])
    events = json.loads((reported / "events.json").read_text())
    events["values"]["share"]["value"] = 0.25
    (reported / "events.json").write_text(json.dumps(events))
    doc.write_text("<!--v:events.share-->50 %<!--/v-->\n")
    break_it(repo)
    outputs = [
        repo / "values.tex",
        reported / "questions.md",
        reported / "changes.md",
        doc,
    ]
    before = {path: path.read_text() for path in outputs}
    with pytest.raises(KeyError, match=error):
        script.main([])
    assert {path: path.read_text() for path in outputs} == before


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


def test_summarize_writes_no_canonical_summary_from_uncommitted_code(
    repo, monkeypatch, capsys
):
    """``repo`` stands for the checkout: its ``uv.lock`` is part of every build."""
    script = _load_script("summarize")
    canonical = repo / "paper" / "reported"
    monkeypatch.setattr(script, "ROOT_DIR", repo)
    monkeypatch.setattr(script, "SUMMARY_DIR", canonical)
    (repo / "uv.lock").write_text("lock 2\n")
    with pytest.raises(SystemExit) as refused:
        script.main(["--summary", "survey"])
    assert refused.value.code == 1
    assert "differ from the commit: uv.lock" in capsys.readouterr().err
    assert not canonical.exists()

    trial = repo / "trial"
    script.main(["--summary", "survey", "--output-dir", str(trial)])
    built = values.read_summary(trial / "survey.json")
    head = _git(repo, "rev-parse", "HEAD").strip()
    assert built.code["uncommitted"] == ["uv.lock"]
    assert built.commit == head[:12] + "-dirty"

    _git(repo, "commit", "-q", "-am", "lock 2")
    (repo / "notes.md").write_text("an uncommitted file no build uses\n")
    script.main(["--summary", "survey"])
    built = values.read_summary(canonical / "survey.json")
    assert built.code["uncommitted"] == []
    assert built.commit == _git(repo, "rev-parse", "HEAD").strip()[:12]
