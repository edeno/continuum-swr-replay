"""The question list, and every reported value's tags against it.

Each value names the questions it answers (``questions``) and the figures that show
it (``used_in``); the question list (``reported/questions.py``) is the one place
those are defined. These tests check the list itself, the tags the part builders
write (read from their source, so no lab data is needed), the tags of the
committed summaries, and the generated view.
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
    detection,
    emit,
    events,
    survey,
    values,
)
from continuum_swr_replay.reported.questions import (
    ANSWERED,
    FIGURES,
    QUESTIONS,
    SECTION_CODES,
    SECTIONS,
    STATUSES,
    parse_figure_tag,
    tag_problems,
)

ROOT = Path(__file__).resolve().parents[2]
REPORTED = ROOT / "paper" / "reported"
BUILDERS = ROOT / "src" / "continuum_swr_replay" / "reported"

# --- the list itself ---------------------------------------------------------------


def test_question_ids_are_unique():
    ids = [q.id for q in QUESTIONS]
    assert len(ids) == len(set(ids))


def test_every_question_sits_in_the_outline():
    for q in QUESTIONS:
        assert (q.part, q.section) in SECTIONS, q.id


def test_a_question_s_id_is_its_section_code_and_number():
    assert set(SECTION_CODES) == set(SECTIONS)
    for q in QUESTIONS:
        code, number = q.id.rsplit(".", 1)
        assert code == SECTION_CODES[(q.part, q.section)] and number.isdigit(), q.id


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
            assert parse_figure_tag(f"figure: {figure}") is not None, (q.id, figure)


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


def test_figure_tags_name_a_figure_or_a_lettered_panel():
    assert parse_figure_tag("figure: 2b") == ("2", "b")
    assert parse_figure_tag("figure: 2") == ("2", "")
    assert parse_figure_tag("figure: memoryless comparison") == (
        "memoryless comparison",
        "",
    )
    assert parse_figure_tag("figure: 2z") is None
    assert parse_figure_tag("figure: spike support") is None
    assert parse_figure_tag("supplement") is None


def test_tag_problems_names_each_mismatch():
    assert tag_problems(("2d.2",), ("paper", "figure: 3c")) == []
    assert tag_problems(("nope",), ("paper",)) == ["no question 'nope'"]
    assert tag_problems(("2d.2",), ("docs: somewhere",)) == [
        "unknown use 'docs: somewhere'"
    ]
    assert tag_problems(("2d.2",), ("figure: 2b",)) == [
        "question 2d.2 does not list figure '2b'"
    ]
    assert tag_problems(("2c.6",), ("supplement",)) == [
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
        "used_in": ("paper",),
        "source": "s",
    }
    return values.Value(**{**base, **fields})


def test_a_value_with_a_bad_tag_is_refused_unless_told_not_to_check():
    with pytest.raises(ValueError, match="no question '12'"):
        _value(questions=("12",))
    with pytest.raises(ValueError, match="unknown use 'figure: main, left out'"):
        _value(used_in=("figure: main, left out",))
    assert _value(questions=("12",), check_tags=False).questions == ("12",)
    with pytest.raises(ValueError, match="no question '12'"):
        _value(questions=("12",))


def test_replacing_a_field_checks_the_tags_again():
    from dataclasses import replace

    with pytest.raises(ValueError, match="no question 'x'"):
        replace(_value(), questions=("x",))


def test_an_old_summary_is_read_unchecked_and_its_problems_listed(tmp_path):
    summary = values.Summary(name="detection", produced_by="by", commit="c")
    summary.add(_value())
    path = tmp_path / "detection.json"
    path.write_text(summary.to_json().replace('"2d.2"', '"12"'))
    with pytest.raises(ValueError, match="no question '12'"):
        values.read_summary(path)
    old = values.read_summary(path, check_tags=False)
    assert values.summary_tag_problems(old) == {"share": ["no question '12'"]}


# --- the tags the builders write ---------------------------------------------------


def _branches(node: ast.expr) -> list[ast.expr]:
    """Both arms of a conditional expression, recursively; else the node."""
    if isinstance(node, ast.IfExp):
        return _branches(node.body) + _branches(node.orelse)
    return [node]


def _options(
    node: ast.expr | None, module, local: dict[str, tuple[str, ...]]
) -> list[tuple[str, ...]] | None:
    """The tag tuples an argument can be: each arm of a conditional, a literal, a
    name the function assigned a literal to, or a module constant; None when it is
    only known at run time."""
    if node is None:
        return None
    options = []
    for branch in _branches(node):
        if isinstance(branch, ast.Name) and branch.id in local:
            options.append(local[branch.id])
            continue
        if isinstance(branch, ast.Name) and isinstance(
            getattr(module, branch.id, None), tuple
        ):
            options.append(getattr(module, branch.id))
            continue
        try:
            options.append(tuple(ast.literal_eval(branch)))
        except ValueError:
            return None
    return options


def _tag_sites(name: str, source: str | None = None):
    """Every (questions, used_in) a builder can give a value, as far as its source
    says, and the halves it cannot resolve.

    In each function: every ``_value_factory(source, questions, used_in)`` call; every
    call with a ``questions_`` / ``used_in_`` override, paired with the factory's
    other half when the function assigned that factory; every ``Value(...)``
    call. Returns ``(pairs, unresolved)``: pairs are ``(line, function, questions
    options or None, used_in options or None)``; unresolved are ``(function,
    which, expression)`` for each half known only at run time."""
    module = importlib.import_module(f"continuum_swr_replay.reported.{name}")
    text = (BUILDERS / f"{name}.py").read_text() if source is None else source
    pairs, unresolved = [], set()
    for function in ast.walk(ast.parse(text)):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local = {}
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                try:
                    value = ast.literal_eval(node.value)
                except ValueError:
                    continue
                if isinstance(value, tuple):
                    local[node.targets[0].id] = value

        def resolve(which, node, function=function, local=local):
            found = _options(node, module, local)
            if found is None:
                expression = "?" if node is None else ast.unparse(node)
                unresolved.add((function.name, which, expression))
            return found

        factories = {}
        assigned = {
            id(node.value): node.targets[0].id
            for node in ast.walk(function)
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        }
        for call in ast.walk(function):
            if not (
                isinstance(call, ast.Call)
                and getattr(call.func, "id", None) == "_value_factory"
            ):
                continue
            questions = resolve("questions", call.args[1])
            used_in = resolve("used_in", call.args[2])
            pairs.append((call.lineno, function.name, questions, used_in))
            if id(call) in assigned:
                factories[assigned[id(call)]] = (questions, used_in)
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            keywords = {k.arg: k.value for k in node.keywords}
            name = getattr(node.func, "id", None)
            if name == "Value":
                questions = resolve("questions", keywords.get("questions"))
                used_in = resolve("used_in", keywords.get("used_in"))
                pairs.append((node.lineno, function.name, questions, used_in))
            elif {"questions_", "used_in_"} & set(keywords):
                questions, used_in = factories.get(name, (None, None))
                for which, half in (
                    ("questions", "questions_"),
                    ("used_in", "used_in_"),
                ):
                    if half in keywords:
                        found = resolve(half, keywords[half])
                    elif name in factories:
                        continue
                    else:
                        found = resolve(which, None)
                    if which == "questions":
                        questions = found
                    else:
                        used_in = found
                pairs.append((node.lineno, function.name, questions, used_in))
    return pairs, unresolved


_PASS_THROUGH = {
    (fn, which, expression)
    for fn in ("_value_factory", "make")
    for which, expression in (
        ("questions", "questions"),
        ("used_in", "used_in"),
        ("questions", "questions_ or questions"),
        ("used_in", "used_in_ or used_in"),
    )
}
"""Each part's factory, which passes its caller's tags on: checked at its callers."""

UNRESOLVED = {
    "survey": set(),
    "detection": {s for s in _PASS_THROUGH if s[2] in ("questions", "used_in")},
    "events": {s for s in _PASS_THROUGH if "or" in s[2]}
    | {
        (fn, which, which)
        for fn in ("property_interval_values", "property_values")
        for which in ("questions", "used_in")
    },
    "behavior": {s for s in _PASS_THROUGH if "or" in s[2]}
    | {("content_contrast_values", "questions", "questions")},
    "mpfc": {
        ("_value_factory", "questions", "questions"),
        ("_value_factory", "used_in", "used_in_ or used_in"),
        ("make", "questions", "questions"),
        ("make", "used_in", "used_in_ or used_in"),
        ("called_fraction_values", "questions", "?"),
        ("cohort_values", "questions", "?"),
        ("followup_values", "questions", "questions"),
    },
}
"""The tag halves the scan cannot resolve, each covered elsewhere: the factories'
pass-through; the events builder's property loops (``_PROPERTIES``) and the behavior builder's content
questions (``_CONTENT_QUESTIONS``), checked by :func:`test_every_named_tag_a_builder_writes_is_known`;
the mpfc builder's functions that take their factory or questions as an
argument, whose questions are :data:`MPFC_QUESTIONS`. A new unresolved half fails until it is covered here."""

MPFC_QUESTIONS = ("4.1",)
"""Every questions tuple the mpfc builder writes; its unresolved questions halves are checked as
these."""


@pytest.mark.parametrize("name", list(values.SUMMARIES))
def test_every_tag_pair_a_builder_writes_is_known(name):
    pairs, _ = _tag_sites(name)
    assert pairs, f"no tags found in {name}.py"
    problems = []
    for line, _, questions, used_in in pairs:
        if questions is None and name == "mpfc":
            questions = [MPFC_QUESTIONS]
        for c in questions or [()]:
            for u in used_in or [()]:
                problems += [f"{name}.py:{line}: {p}" for p in tag_problems(c, u)]
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("name", list(values.SUMMARIES))
def test_the_scan_resolves_every_tag_but_the_listed_ones(name):
    _, unresolved = _tag_sites(name)
    assert unresolved == UNRESOLVED[name]


def test_the_mpfc_builder_writes_only_its_one_question():
    pairs, _ = _tag_sites("mpfc")
    questions = {c for _, _, options, _ in pairs if options for c in options}
    assert questions == {MPFC_QUESTIONS}


SYNTHETIC = """
CONSTANT = ("paper", "figure: 2")


def f(source, word, unknown):
    local = ("2b.6",)
    make = _value_factory(source, ("2d.2",), ("paper", "figure: 2"))
    make("a", 1, "count", "count", "d", questions_=local)
    make("b", 1, "count", "count", "d", used_in_=CONSTANT if word else ("paper",))
    Value(questions=("2d.5",), used_in=("figure: 3",))
    make("c", 1, "count", "count", "d", questions_=unknown)
"""


def test_the_scan_follows_overrides_conditionals_and_constants(monkeypatch):
    module = importlib.import_module("continuum_swr_replay.reported.survey")
    monkeypatch.setattr(module, "CONSTANT", ("paper", "figure: 2"), raising=False)
    pairs, unresolved = _tag_sites("survey", source=SYNTHETIC)
    assert sorted((line, c, u) for line, _, c, u in pairs) == [
        (7, [("2d.2",)], [("paper", "figure: 2")]),
        (8, [("2b.6",)], [("paper", "figure: 2")]),
        (9, [("2d.2",)], [("paper", "figure: 2"), ("paper",)]),
        (10, [("2d.5",)], [("figure: 3",)]),
        (11, None, [("paper", "figure: 2")]),
    ]
    assert unresolved == {("f", "questions_", "unknown")}
    problems = sorted(
        {
            p
            for _, _, c, u in pairs
            for ci in c or [()]
            for ui in u
            for p in tag_problems(ci, ui)
        }
    )
    assert problems == [
        "question 2b.6 does not list figure '2'",
        "question 2d.2 does not list figure '2'",
        "question 2d.5 does not list figure '3'",
    ]


def _named_tags() -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    """Tags the builders hold in module constants: (where, questions, used_in), an
    empty tuple where the constant gives only one of the two."""
    found = [("survey.CLAIMS/USED_IN", survey.QUESTIONS_ANSWERED, survey.USED_IN)]
    found.append(("detection.MEMORYLESS_FIGURE", (), detection.MEMORYLESS_FIGURE))
    for row in events._PROPERTIES:
        found.append((f"events._PROPERTIES {row[0]}", row[6], row[7]))
    for lead, questions in behavior._CONTENT_QUESTIONS:
        found.append((f"behavior._CONTENT_CLAIMS {lead}", questions, ("supplement",)))
    found.append(("behavior._content_questions fallback", ("3.1",), ("supplement",)))
    return found


def test_every_named_tag_a_builder_writes_is_known():
    problems = [
        f"{where}: {p}"
        for where, questions, used_in in _named_tags()
        for p in tag_problems(questions, used_in)
    ]
    assert not problems, "\n".join(problems)
    assert behavior._content_questions("no_such_word") == ("3.1",)


def test_the_remote_groups_spike_evidence_also_answers_question_9():
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
    return values.read_summaries(REPORTED, check_tags=False)


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
    summary = values.Summary(name="events", produced_by="by", commit="c")
    summary.add(
        _value(key="shared", questions=("2b.2", "2b.3"), used_in=("figure: 2b",))
    )
    summary.add(_value(key="old", questions=("12",), check_tags=False))
    view = emit.questions_view({"events": summary})
    section_2 = view.split("### 2b.2. ")[1].split("### ")[0]
    section_21 = view.split("### 2b.3. ")[1].split("### ")[0]
    assert "`\\EventsShared`" in section_2
    assert "`\\EventsShared`" in section_21
    flagged = view.split("## Values with tag problems")[1].split("## Values by")[0]
    assert "`\\EventsOld`" in flagged
    assert "no question '12'" in flagged
    by_figure = view.split("### Figure 2: ")[1].split("### Figure")[0]
    assert "Panel b: `\\EventsShared`" in by_figure


def test_a_question_with_no_figure_shows_no_figures_line():
    view = emit.questions_view({})
    question_17 = view.split("### 2d.5. ")[1].split("\n\n")[1]
    assert question_17.startswith("Status: computed.") and "Figures" not in question_17
