"""The question list, and every reported value's tags against it.

Each value names the questions it answers (``questions``), the figures that show
it (``figures``) and the builder function that made it (``function``); the
question list (``reported/questions.py``) is the one place questions and figures
are defined. These tests check the list itself, the tags and functions the
builders write (read from their source, so no lab data is needed), the tags of
the committed summaries, and the generated view.
"""

from __future__ import annotations

import ast
import importlib
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.reported import (
    behavior,
    emit,
    events,
    values,
)
from continuum_swr_replay.reported.questions import (
    ANSWERED,
    FIGURES,
    QUESTIONS,
    SECTIONS,
    STATUSES,
    parse_figure,
    tag_problems,
)
from continuum_swr_replay.reported.values import Tags, value_factory

ROOT = Path(__file__).resolve().parents[2]
REPORTED = ROOT / "paper" / "reported"
BUILDERS = ROOT / "src" / "continuum_swr_replay" / "reported"

# --- the list itself ---------------------------------------------------------------


def test_question_ids_are_unique():
    ids = [q.id for q in QUESTIONS]
    assert len(ids) == len(set(ids))


def test_a_question_s_id_is_a_section_code_and_a_number():
    codes = {s.code for s in SECTIONS}
    assert len(codes) == len(SECTIONS)
    for q in QUESTIONS:
        code, number = q.id.rsplit(".", 1)
        assert code in codes and number.isdigit() and int(number) > 0, q.id


def test_a_status_other_than_computed_says_why():
    for q in QUESTIONS:
        assert q.status in STATUSES, q.id
        assert bool(q.reason) == (q.status != "computed"), q.id
    assert set(ANSWERED) < set(STATUSES)


def test_a_question_is_written_as_a_question():
    for q in QUESTIONS:
        assert re.sub(r"\s*\([^()]*\)$", "", q.text).endswith("?"), q.id


def test_every_figure_a_question_lists_exists():
    for q in QUESTIONS:
        for figure in q.figures:
            assert parse_figure(figure) is not None, (q.id, figure)


def test_no_figure_id_reads_as_another_figure_and_a_panel():
    panels = {name + p for name, figure in FIGURES.items() for p in figure.panels}
    assert not panels & FIGURES.keys()


def test_every_figure_script_exists_and_the_figure_pipeline_names_it():
    pipeline = (ROOT / "docs" / "figure-pipeline.md").read_text()
    for name, figure in FIGURES.items():
        assert (ROOT / figure.script).is_file(), name
        assert figure.script.removeprefix("scripts/") in pipeline, (
            f"docs/figure-pipeline.md does not name {figure.script} (figure {name!r})"
        )


# --- tag checking ------------------------------------------------------------------


def test_a_figure_name_is_a_figure_or_a_lettered_panel():
    assert parse_figure("2b") == ("2", "b")
    assert parse_figure("2") == ("2", "")
    assert parse_figure("memoryless_comparison") == ("memoryless_comparison", "")
    assert parse_figure("2z") is None
    assert parse_figure("spike support") is None
    assert parse_figure("") is None


def test_tag_problems_names_each_mismatch():
    assert tag_problems(("2d.2",), ("3c",)) == []
    assert tag_problems(("2d.2",), ()) == []
    assert tag_problems(("nope",), ()) == ["no question 'nope'"]
    assert tag_problems(("2d.2",), ("supplement",)) == ["no figure 'supplement'"]
    assert tag_problems(("2d.2",), ("2b",)) == [
        "question 2d.2 does not list figure '2b'"
    ]
    assert tag_problems(("2c.6",), ()) == [
        "question 2c.6 is not computed, so has no values"
    ]


def _value(**fields) -> values.Value:
    base = {
        "key": "share",
        "value": 0.5,
        "kind": "fraction",
        "style": "pct",
        "unit": "",
        "description": "d",
        "questions": ("2d.2",),
        "figures": (),
        "source": "s",
        "function": "f",
    }
    return values.Value(**{**base, **fields})


def test_a_summary_with_a_bad_tag_is_not_written(tmp_path):
    summary = values.Summary(name="events", produced_by="by")
    summary.add(_value(questions=("12",)))
    summary.add(_value(key="other", figures=("main, left out",)))
    path = tmp_path / "events.json"
    with pytest.raises(ValueError) as refused:
        summary.write(path)
    assert "share: no question '12'" in str(refused.value)
    assert "other: no figure 'main, left out'" in str(refused.value)
    assert not path.exists()


def test_a_summary_written_before_the_list_changed_is_read_and_its_problems_listed(
    tmp_path,
):
    summary = values.Summary(name="detection", produced_by="by")
    summary.add(_value())
    path = tmp_path / "detection.json"
    summary.write(path)
    path.write_text(path.read_text().replace('"2d.2"', '"12"'))
    old = values.read_summary(path)
    assert values.summary_tag_problems(old) == {"share": ["no question '12'"]}


# --- the tags the builders write ---------------------------------------------------

_ID = re.compile(r"[0-9][a-d]?\.[0-9]+")


def _builder(name: str):
    return importlib.import_module(f"continuum_swr_replay.reported.{name}")


def _tags_lookups(tree: ast.AST) -> tuple[set[str], int]:
    """The keys a module looks up in ``TAGS`` by a literal, and how many lookups
    it makes by any other expression."""
    keys, other = set(), 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "TAGS"
        ):
            if isinstance(node.slice, ast.Constant):
                keys.add(node.slice.value)
            else:
                other += 1
    return keys, other


@pytest.mark.parametrize("name", list(values.SUMMARIES))
def test_every_tag_group_a_builder_declares_is_known(name):
    tags = _builder(name).TAGS
    assert tags and all(isinstance(t, Tags) for t in tags.values())
    problems = {k: tag_problems(t.questions, t.figures) for k, t in tags.items()}
    assert not {k: found for k, found in problems.items() if found}


@pytest.mark.parametrize("name", list(values.SUMMARIES))
def test_a_builder_uses_every_tag_group_it_declares_and_no_other(name):
    tree = ast.parse((BUILDERS / f"{name}.py").read_text())
    keys, other = _tags_lookups(tree)
    assert other == 0
    assert keys == set(_builder(name).TAGS)


@pytest.mark.parametrize("name", list(values.SUMMARIES))
def test_a_builder_declares_its_tags_only_in_its_table(name):
    """Every ``Tags`` and every question id in a builder is in its ``TAGS`` table,
    so a value's tags are declared once, as data."""
    tree = ast.parse((BUILDERS / f"{name}.py").read_text())
    table = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "TAGS"
    )
    stray = [
        (node.lineno, ast.unparse(node))
        for node in ast.walk(tree)
        if (
            (isinstance(node, ast.Constant) and isinstance(node.value, str))
            and _ID.fullmatch(node.value)
            or (isinstance(node, ast.Call) and ast.unparse(node.func) == "Tags")
        )
        and not table.lineno <= node.lineno <= table.end_lineno
    ]
    assert not stray


def _factory_calls(tree: ast.Module) -> list[tuple[int, str, set[str]]]:
    """Each value-factory call of a builder: its line, the function it names, and
    the functions it may name: the module-level function the call is in, and any
    function it is passed to."""
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    calls = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "value_factory"
        ):
            continue
        allowed, at = set(), node
        while at in parents:
            parent = parents[at]
            if isinstance(parent, ast.Call) and at in parent.args:
                allowed.add(ast.unparse(parent.func))
            if isinstance(parent, ast.FunctionDef) and parents[parent] is tree:
                allowed.add(parent.name)
            at = parent
        calls.append((node.lineno, ast.unparse(node.args[0]), allowed))
    return calls


@pytest.mark.parametrize("name", list(values.SUMMARIES))
def test_a_value_factory_names_the_function_its_values_are_made_in(name):
    """A value's ``function`` is where a reader looks for its computation, so each
    factory names the function it is made in, or the one it is handed to."""
    tree = ast.parse((BUILDERS / f"{name}.py").read_text())
    calls = _factory_calls(tree)
    assert calls
    wrong = [(line, named) for line, named, allowed in calls if named not in allowed]
    assert not wrong


def test_a_factory_records_the_function_and_fills_in_what_is_shared():
    def some_values():
        pass

    shared = Tags(("2d.2",), ("3c",))
    make = value_factory(some_values, "a table", shared, pending="rerun it")
    value = make("n_events", np.int64(3), "count", "count", "events")
    assert (value.function, value.source) == ("some_values", "a table")
    assert (value.questions, value.figures) == (("2d.2",), ("3c",))
    assert value.value == 3 and isinstance(value.value, int)
    assert value.pending == "rerun it"
    other = make(
        "share",
        np.float32(0.5),
        "fraction",
        "pct",
        "d",
        tags=Tags(("2d.3",)),
        pending="p",
    )
    assert (other.questions, other.figures, other.pending) == (("2d.3",), (), "p")
    assert isinstance(other.value, float)
    with pytest.raises(ValueError, match="a count must be whole"):
        make("n_half", 2.5, "count", "count", "d")


def test_the_content_values_find_their_tags_by_key_word():
    assert (
        behavior._content_tags("swr_armreward_total")
        == behavior.TAGS["content_reward_value"]
    )
    assert behavior._content_tags("no_such_word") == behavior.TAGS["anchoring"]


def test_the_remote_groups_spike_evidence_also_answers_question_2c_7():
    rng = np.random.default_rng(0)
    groups = ["baseline", "SWR-remote", "nonSWR-remote", "nonSWR-close"]
    rows = list(itertools.product(["a", "b"], ["e1", "e2"], groups, range(3)))
    metrics = pd.DataFrame(rows, columns=["animal", "epoch", "group", "i"]).assign(
        epoch_id=lambda d: d["animal"] + d["epoch"],
        n_spikes=lambda d: rng.integers(1, 50, len(d)),
        n_tetrodes=lambda d: rng.integers(1, 10, len(d)),
        dll_per_spike=lambda d: rng.normal(size=len(d)),
        dist_to_gpi_peak=lambda d: rng.uniform(0, 100, len(d)),
    )
    by_key = {v.key: v for v in events.spike_evidence_values(metrics, "s")}
    close = [k for k in by_key if k.startswith("non_swr_close_")]
    remote = [k for k in by_key if k.startswith(("swr_remote_", "non_swr_remote_"))]
    assert close and len(remote) == 2 * len(close)
    for key, value in by_key.items():
        assert ("2c.7" in value.questions) == (key in remote), key


# --- the committed summaries -------------------------------------------------------


@pytest.fixture(scope="module")
def committed():
    return values.read_summaries(REPORTED)


def test_every_committed_summary_matches_the_question_list(committed):
    problems = {
        name: values.summary_tag_problems(summary)
        for name, summary in committed.items()
    }
    assert not {name: found for name, found in problems.items() if found}


def test_a_question_has_values_exactly_when_its_status_says_so(committed):
    answered = {
        c for s in committed.values() for v in s.values.values() for c in v.questions
    }
    wrong = {
        f"{q.id} ({q.status})"
        for q in QUESTIONS
        if (q.id in answered) != (q.status in ANSWERED)
    }
    assert not wrong


# --- the generated view -------------------------------------------------------------


def test_the_view_lists_a_value_under_each_question_it_answers_and_flags_old_tags():
    summary = values.Summary(name="events", produced_by="by")
    summary.add(_value(key="shared", questions=("2d.3", "2d.4"), figures=("3",)))
    summary.add(_value(key="old", questions=("12",)))
    view = emit.questions_view({"events": summary})
    extent = view.split("### 2d.3. ")[1].split("### ")[0]
    stillness = view.split("### 2d.4. ")[1].split("### ")[0]
    assert "`\\EventsShared`" in extent and "`events.f`" in extent
    assert "`\\EventsShared`" in stillness
    flagged = view.split("## Values with tag problems")[1].split("## Values by")[0]
    assert "`\\EventsOld`" in flagged
    assert "no question '12'" in flagged
    by_figure = view.split("### Figure 3: ")[1].split("### Figure")[0]
    assert "Unlettered: `\\EventsShared`" in by_figure


def test_a_question_with_no_figure_shows_no_figures_line():
    view = emit.questions_view({})
    question_2d_5 = view.split("### 2d.5. ")[1].split("\n\n")[1]
    assert (
        question_2d_5.startswith("Status: computed.") and "Figures" not in question_2d_5
    )
