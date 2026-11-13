"""Every code path a tracked Markdown doc cites must exist.

Docs name scripts and modules by path, and a module move that leaves a doc
behind sends the reader to a file that is not there. This test reads every
tracked ``*.md`` (``git ls-files``, so untracked drafts are ignored) and checks
four kinds of reference:

- a file path ``scripts/...py``, resolved against the repo root;
- a file path ``src/continuum_swr_replay/...py``, resolved the same way;
- a bare file path ``analysis/...py`` that starts with a package directory
  (:data:`PACKAGE_DIRS`), resolved against ``src/continuum_swr_replay/`` --
  the form CLAUDE.md uses for modules -- or against ``scripts/``, whose
  ``tables/`` and ``decoding/`` share those names;
- a dotted name ``continuum_swr_replay.a.b`` inside backticks (an inline code
  span or a fenced code block). It must name a module or package, or a name
  that module defines. Names are found statically with ``ast`` -- a top-level
  ``def``/``class``/assignment/import, a key of a package's ``_LAZY_EXPORTS``
  table, or, one level down, a member of a class. Nothing is imported.

A path written with a placeholder (``scripts/qa/<name>.py``) or a glob
(``scripts/figures/*.py``) is not matched: ``<`` and ``*`` end the match before
``.py``. A reference to a file that is gone on purpose is written with
``(removed in <commit>)`` right after it, and is not checked.

Historical records are not checked: implementation plans
(``.claude/docs/plans/``), review and audit reports (``.claude/docs/reviews/``),
the known-issues log (``docs/known_issues/``), the dated design specs
(``docs/design/YYYY-MM-DD-*.md``) and the record of moved code
(``docs/history/``) describe the tree as it was when they were written. Nor
are the vendored agent skills (``.claude/skills/``), which name files relative
to the skill's own directory.
"""

from __future__ import annotations

import ast
import functools
import re
import subprocess
from pathlib import Path

from tests.support._imports import defined_names, parse

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
PACKAGE = "continuum_swr_replay"
assert (SRC / PACKAGE / "__init__.py").is_file(), f"package not found under {SRC}"

#: Tracked docs under these prefixes are historical and not checked.
HISTORICAL_PREFIXES = (
    ".claude/docs/plans/",
    ".claude/docs/reviews/",
    "docs/known_issues/",
    ".claude/skills/",
    "docs/history/",
)

#: A dated design spec, a historical record like those above.
_DATED_DESIGN = re.compile(r"docs/design/\d{4}-\d{2}-\d{2}-[^/]*\.md")

#: Package subdirectories a doc may cite by bare path
#: (``analysis/events/classified_table.py``).
PACKAGE_DIRS = tuple(
    sorted(p.parent.name for p in (SRC / PACKAGE).glob("*/__init__.py"))
)

#: A repo-relative file path. The lookbehind stops a match inside a longer path
#: (``tests/scripts/x.py``); a leading ``./`` or ``../`` is allowed and dropped.
_FILE_PATH = re.compile(
    r"(?<![\w/.-])(?:\.\.?/)*"
    r"((?:scripts|src/continuum_swr_replay)/[\w./-]*\.py)(?![\w])"
)
#: A package-relative file path. The same lookbehind keeps it from matching the
#: tail of ``src/continuum_swr_replay/analysis/...`` or ``tests/analysis/...``.
_PACKAGE_PATH = re.compile(
    rf"(?<![\w/.-])((?:{'|'.join(PACKAGE_DIRS)})/[\w./-]*\.py)(?![\w])"
)
_DOTTED = re.compile(r"(?<![\w.])continuum_swr_replay(?:\.\w+)+")
_INLINE_CODE = re.compile(r"`([^`]+)`")
_FENCE = re.compile(r"^\s*(```|~~~)")
#: Text right after a reference that marks its file as deliberately gone.
_REMOVED = re.compile(r"^`*\)?\s*\(removed\b")


def _tracked_docs() -> list[str]:
    """Tracked ``*.md`` paths, repo-relative, minus the historical ones."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md"],
        cwd=REPO,
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    docs = [p for p in out.split("\0") if p]
    assert docs, "git ls-files found no Markdown; the scan would be vacuous"
    return [
        p
        for p in docs
        if not p.startswith(HISTORICAL_PREFIXES) and not _DATED_DESIGN.fullmatch(p)
    ]


def _code_spans(text: str) -> list[tuple[int, str]]:
    """``(line number, text)`` for every backticked span in ``text``.

    A fenced block contributes each of its lines; outside fences, each inline
    code span on a line is one entry.
    """
    spans: list[tuple[int, str]] = []
    in_fence = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            spans.append((lineno, line))
        else:
            spans.extend((lineno, m.group(1)) for m in _INLINE_CODE.finditer(line))
    return spans


@functools.cache
def _module_names(path: Path) -> dict[str, ast.stmt]:
    return defined_names(parse(path).body)


def _resolves(dotted: str) -> bool:
    """Whether ``dotted`` names a module, or a name a module defines."""
    parts = dotted.split(".")
    here = SRC / parts[0]
    module_file = here / "__init__.py"
    rest = parts[1:]
    while rest:
        nxt = rest[0]
        if module_file.name == "__init__.py" and (here / nxt / "__init__.py").is_file():
            here = here / nxt
            module_file = here / "__init__.py"
        elif module_file.name == "__init__.py" and (here / f"{nxt}.py").is_file():
            module_file = here / f"{nxt}.py"
        else:
            break
        rest = rest[1:]
    if not rest:
        return True
    node = _module_names(module_file).get(rest[0])
    if node is None:
        return False
    if len(rest) > 1 and isinstance(node, ast.ClassDef):
        return rest[1] in defined_names(node.body)
    # A deeper attribute of a function, a constant or an imported name is not
    # followed; the first component is the part a module move breaks.
    return True


def _references(doc: str) -> list[tuple[int, str]]:
    """``(line, reference)`` for every reference in ``doc`` that does not resolve."""
    text = (REPO / doc).read_text()
    bad: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for m in _FILE_PATH.finditer(line):
            if _REMOVED.match(line[m.end() :]):
                continue
            if not (REPO / m.group(1)).is_file():
                bad.append((lineno, m.group(1)))
        for m in _PACKAGE_PATH.finditer(line):
            if _REMOVED.match(line[m.end() :]):
                continue
            if not any(
                (base / m.group(1)).is_file()
                for base in (SRC / PACKAGE, REPO / "scripts")
            ):
                bad.append((lineno, m.group(1)))
    for lineno, span in _code_spans(text):
        for m in _DOTTED.finditer(span):
            if not _resolves(m.group(0)):
                bad.append((lineno, m.group(0)))
    return bad


def test_every_cited_path_resolves() -> None:
    unresolved = [
        f"{doc}:{lineno}: {ref}"
        for doc in _tracked_docs()
        for lineno, ref in _references(doc)
    ]
    assert not unresolved, (
        f"{len(unresolved)} doc reference(s) name no existing file, module or "
        "attribute. Point each at the current path:\n  " + "\n  ".join(unresolved)
    )


def test_resolver_on_known_names() -> None:
    """The resolver accepts real modules and names and rejects invented ones."""
    assert _resolves("continuum_swr_replay.parameters")
    assert _resolves("continuum_swr_replay.paths.processed_data_dir")
    assert _resolves("continuum_swr_replay.analysis.behavior")
    assert _resolves(
        "continuum_swr_replay.analysis.replay_models.fitting.fit_conditional_poisson"
    )
    # A lazily exported name, found in the package's _LAZY_EXPORTS table.
    assert _resolves("continuum_swr_replay.data_loaders.load_data")
    assert not _resolves("continuum_swr_replay.parameters.NO_SUCH_NAME")
    assert not _resolves("continuum_swr_replay.no_such_module")
    assert not _resolves("continuum_swr_replay.analysis.behavior.no_such_module.x")


def test_path_pattern_boundaries() -> None:
    """A path inside a longer path, or with a placeholder, is not matched."""

    def found(s: str) -> list[str]:
        return [m.group(1) for m in _FILE_PATH.finditer(s)]

    assert found("see ../scripts/qa/x.py.") == ["scripts/qa/x.py"]
    assert found("[a](src/continuum_swr_replay/data_types.py)") == [
        "src/continuum_swr_replay/data_types.py"
    ]
    assert found("tests/scripts/x.py") == []
    assert found("scripts/qa/<name>.py and scripts/figures/*.py") == []

    def package_found(s: str) -> list[str]:
        return [m.group(1) for m in _PACKAGE_PATH.finditer(s)]

    assert package_found(
        "`analysis/events/classified_table.py:build_classified_nonlocal_event_table`"
    ) == ["analysis/events/classified_table.py"]
    assert package_found("in data_loaders/constants.py.") == [
        "data_loaders/constants.py"
    ]
    assert package_found("src/continuum_swr_replay/analysis/x.py") == []
    assert package_found("tests/analysis/pfc/test_x.py") == []
    assert package_found("scripts/analysis/x.py") == []

    line = "`scripts/qa/gone.py` (removed in abc1234)"
    m = next(_FILE_PATH.finditer(line))
    assert _REMOVED.match(line[m.end() :])
