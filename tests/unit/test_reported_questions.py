"""The question list, and every reported value's tags against it.

Each value names the questions it answers (``claims``) and the figures that show
it (``used_in``); the question list (``reported/questions.py``) is the one place
those are defined. These tests check the list itself, the tags the part builders
write (read from their source, so no lab data is needed), and the tags of the
committed summaries.
"""

from __future__ import annotations

import ast
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.reported import part0, part1, part2, values
from continuum_swr_replay.reported.questions import (
    FIGURES,
    QUESTION_IDS,
    QUESTIONS,
    SECTIONS,
    STATUSES,
    parse_figure_tag,
    tag_problems,
)

ROOT = Path(__file__).resolve().parents[2]
REPORTED = ROOT / "paper" / "reported"
BUILDERS = ROOT / "src" / "continuum_swr_replay" / "reported"

AWAITING_REBUILD = {1, 2, 3}
"""The parts whose committed summary was written before the question list
changed their tags (question 12 split into 12a and 12b; the Part 2 figure tags; the
Part 3 values that only a document used). Remove a part once its summary is
rebuilt; the strict xfail below fails until then."""


# --- the list itself ---------------------------------------------------------------


def test_question_ids_are_unique():
    ids = [q.id for q in QUESTIONS]
    assert len(ids) == len(set(ids))


def test_every_question_sits_in_the_outline():
    for q in QUESTIONS:
        assert (q.part, q.section) in SECTIONS, q.id


def test_a_status_other_than_computed_says_why():
    for q in QUESTIONS:
        assert q.status in STATUSES, q.id
        assert bool(q.reason) == (q.status != "computed"), q.id


def test_a_question_is_written_as_a_question():
    for q in QUESTIONS:
        assert re.sub(r"\s*\([^()]*\)$", "", q.text).endswith("?"), q.id


def test_every_figure_a_question_lists_exists():
    for q in QUESTIONS:
        for figure in q.figures:
            assert parse_figure_tag(f"figure: {figure}") is not None, (q.id, figure)


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
    assert tag_problems(("15",), ("paper", "figure: 3c")) == []
    assert tag_problems(("nope",), ("paper",)) == ["no question 'nope'"]
    assert tag_problems(("15",), ("docs: somewhere",)) == [
        "unknown use 'docs: somewhere'"
    ]
    assert tag_problems(("15",), ("figure: 2b",)) == [
        "question 15 does not list figure '2b'"
    ]
    assert tag_problems(("8",), ("supplement",)) == [
        "question 8 is not computed, so has no values"
    ]


def _value(**fields) -> values.Value:
    base = {
        "key": "share",
        "value": 0.5,
        "kind": "fraction",
        "style": "pct",
        "unit": "",
        "description": "d",
        "claims": ("15",),
        "used_in": ("paper",),
        "source": "s",
    }
    return values.Value(**{**base, **fields})


def test_a_value_with_a_bad_tag_is_refused():
    with pytest.raises(ValueError, match="no question '12'"):
        _value(claims=("12",))
    with pytest.raises(ValueError, match="unknown use 'figure: main, left out'"):
        _value(used_in=("figure: main, left out",))


def test_an_old_summary_is_read_unchecked_and_its_problems_listed(tmp_path):
    summary = values.Summary(part=1, produced_by="by", commit="c")
    summary.add(_value())
    path = tmp_path / "part1.json"
    path.write_text(summary.to_json().replace('"15"', '"12"'))
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


def _literal_tags(path: Path) -> list[tuple[int, str, tuple[str, ...]]]:
    """Every literal claims or used_in tuple a builder passes to ``Value``, its
    ``_value_factory`` or a factory's ``make`` (as ``claims_`` / ``used_in_``):
    (line, which, tags). Arguments built at run time are skipped."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        arguments = {}
        if name == "_value_factory" and len(node.args) >= 3:
            arguments = {"claims": node.args[1], "used_in": node.args[2]}
        for keyword in node.keywords:
            if keyword.arg in ("claims", "used_in", "claims_", "used_in_"):
                arguments[keyword.arg.rstrip("_")] = keyword.value
        for which, argument in arguments.items():
            for branch in _branches(argument):
                try:
                    tags = ast.literal_eval(branch)
                except ValueError:
                    continue
                found.append((node.lineno, which, tuple(tags)))
    return found


def _named_tags() -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    """Tags the builders hold in module constants: (where, claims, used_in), an
    empty tuple where the constant gives only one of the two."""
    found = [("part0.CLAIMS/USED_IN", part0.CLAIMS, part0.USED_IN)]
    found.append(("part1.MEMORYLESS_FIGURE", (), part1.MEMORYLESS_FIGURE))
    for row in part2._PROPERTIES:
        found.append((f"part2._PROPERTIES {row[0]}", row[6], row[7]))
    return found


@pytest.mark.parametrize("part", range(5))
def test_every_literal_tag_a_builder_writes_is_known(part):
    path = BUILDERS / f"part{part}.py"
    problems = []
    for line, which, tags in _literal_tags(path):
        claims, used_in = (tags, ()) if which == "claims" else ((), tags)
        found = tag_problems(claims, used_in)
        problems += [f"part{part}.py:{line}: {p}" for p in found]
    assert not problems, "\n".join(problems)


def test_every_named_tag_a_builder_writes_is_known():
    problems = [
        f"{where}: {p}"
        for where, claims, used_in in _named_tags()
        for p in tag_problems(claims, used_in)
    ]
    assert not problems, "\n".join(problems)


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
    by_key = {v.key: v for v in part2.spike_evidence_values(metrics, "s")}
    remote = [k for k in by_key if k.startswith(("swr_remote_", "non_swr_remote_"))]
    assert len(remote) == 16
    for key, value in by_key.items():
        assert ("9" in value.claims) == (key in remote), key


# --- the committed summaries -------------------------------------------------------


def _read(part: int) -> values.Summary:
    return values.read_summary(REPORTED / f"part{part}.json", check_tags=False)


@pytest.mark.parametrize(
    "part",
    [
        pytest.param(
            p,
            marks=pytest.mark.xfail(
                strict=True, reason="summary built before its tags changed"
            ),
        )
        if p in AWAITING_REBUILD
        else p
        for p in range(5)
    ],
)
def test_a_committed_summary_has_only_known_tags(part):
    problems = values.summary_tag_problems(_read(part))
    assert not problems, problems


@pytest.mark.xfail(
    bool(AWAITING_REBUILD), strict=True, reason="summaries awaiting a rebuild"
)
def test_a_question_has_values_exactly_when_its_status_says_so():
    answered = {c for p in range(5) for v in _read(p).values.values() for c in v.claims}
    wrong = [
        f"{q.id} ({q.status})"
        for q in QUESTIONS
        if (q.id in answered) != (q.status in ("computed", "partly computed"))
    ]
    assert not wrong, wrong


def test_every_question_id_a_committed_value_names_is_a_question_or_awaits_a_rebuild():
    unknown = {
        (p, c)
        for p in range(5)
        for v in _read(p).values.values()
        for c in v.claims
        if c not in QUESTION_IDS
    }
    assert {p for p, _ in unknown} <= AWAITING_REBUILD, unknown


# --- the generated view -------------------------------------------------------------


def test_the_view_lists_a_value_under_each_question_it_answers_and_flags_old_tags():
    from continuum_swr_replay.reported import emit

    summary = values.Summary(part=2, produced_by="by", commit="c")
    summary.add(_value(key="shared", claims=("2", "21"), used_in=("figure: 2b",)))
    with values._tags_unchecked():
        summary.add(_value(key="old", claims=("12",)))
    view = emit.questions_view({2: summary})
    section_2 = view.split("### 2. ")[1].split("### ")[0]
    section_21 = view.split("### 21. ")[1].split("### ")[0]
    assert "`\\PartTwoShared`" in section_2
    assert "`\\PartTwoShared`" in section_21
    unknown = view.split("## Values whose tags the question list does not know")[1]
    assert "`\\PartTwoOld`" in unknown.split("## Values by figure")[0]
    assert "no question '12'" in unknown
    by_figure = view.split("### Figure 2: ")[1].split("### Figure")[0]
    assert "Panel b: `\\PartTwoShared`" in by_figure
