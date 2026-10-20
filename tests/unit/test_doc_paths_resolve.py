"""Every code path a tracked Markdown doc cites must exist.

Docs name scripts and modules by path, and a module move that leaves a doc
behind sends the reader to a file that is not there. This test reads every
tracked ``*.md`` (``git ls-files``, so untracked drafts are ignored) and checks
three kinds of reference:

- a file path ``scripts/...py``, resolved against the repo root;
- a file path ``src/continuum_swr_replay/...py``, resolved the same way;
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
(``.claude/docs/plans/``, and ``plans/``, where the April 2026 plans were kept
before that home existed), review and audit reports (``.claude/docs/reviews/``)
and the known-issues log (``docs/known_issues/``) describe the tree as it was
when they were written.

A reference that cannot be fixed goes in :data:`ALLOWLIST` with the reason.
The test also fails on an allowlist entry that no longer occurs, so the list
can only shrink.
"""

from __future__ import annotations

import ast
import functools
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
PACKAGE = "continuum_swr_replay"
assert (SRC / PACKAGE / "__init__.py").is_file(), f"package not found under {SRC}"

#: Tracked docs under these prefixes are historical and not checked.
HISTORICAL_PREFIXES = (
    ".claude/docs/plans/",
    # Implementation plans from April 2026, kept here before .claude/docs/plans/
    # existed; the same doc type, and as historical.
    "plans/",
    ".claude/docs/reviews/",
    "docs/known_issues/",
)

#: ``(doc path, reference) -> why it stays``. Keyed without line numbers, so
#: editing a doc elsewhere does not invalidate an entry.
ALLOWLIST: dict[tuple[str, str], str] = {
    # The vendored agent skills name files relative to the skill's own
    # directory, not the repo root.
    (
        ".claude/skills/scientific-figures-paper/SKILL.md",
        "scripts/figure_utilities.py",
    ): "skill-relative: .claude/skills/scientific-figures-paper/scripts/",
    (
        ".claude/skills/scientific-figures-paper/SKILL.md",
        "scripts/figsizing.py",
    ): "skill-relative, and the vendored skill does not ship it",
    (
        ".claude/skills/scientific-figures-paper/SKILL.md",
        "scripts/label_panels.py",
    ): "skill-relative, and the vendored skill does not ship it",
    (
        ".claude/skills/scientific-figures-paper/SKILL.md",
        "scripts/color_palettes.py",
    ): "skill-relative, and the vendored skill does not ship it",
    (
        ".claude/skills/scientific-presentations/SKILL.md",
        "scripts/validate_presentation.py",
    ): "skill-relative: .claude/skills/scientific-presentations/scripts/",
    (
        ".claude/skills/scientific-presentations/references/design_checklist.md",
        "scripts/validate_presentation.py",
    ): "skill-relative: .claude/skills/scientific-presentations/scripts/",
}

#: A repo-relative file path. The lookbehind stops a match inside a longer path
#: (``tests/scripts/x.py``); a leading ``./`` or ``../`` is allowed and dropped.
_FILE_PATH = re.compile(
    r"(?<![\w/.-])(?:\.\.?/)*"
    r"((?:scripts|src/continuum_swr_replay)/[\w./-]*\.py)(?![\w])"
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
    return [p for p in docs if not p.startswith(HISTORICAL_PREFIXES)]


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


def _top_level_statements(body: list[ast.stmt]) -> list[ast.stmt]:
    """Module-level statements, looking inside ``if``/``try`` blocks."""
    out: list[ast.stmt] = []
    for node in body:
        if isinstance(node, ast.If):
            out.extend(_top_level_statements(node.body + node.orelse))
        elif isinstance(node, ast.Try):
            handlers = [s for h in node.handlers for s in h.body]
            out.extend(
                _top_level_statements(
                    node.body + handlers + node.orelse + node.finalbody
                )
            )
        else:
            out.append(node)
    return out


def _defined_names(body: list[ast.stmt]) -> dict[str, ast.stmt]:
    """Names bound by ``body``, each mapped to the statement that binds it."""
    names: dict[str, ast.stmt] = {}
    for node in _top_level_statements(body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        names[sub.id] = node
            value = node.value
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "_LAZY_EXPORTS"
                    for t in node.targets
                )
                and isinstance(value, ast.Dict)
            ):
                for key in value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        names[key.value] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                names[bound] = node
    return names


@functools.cache
def _module_names(path: Path) -> dict[str, ast.stmt]:
    return _defined_names(ast.parse(path.read_text(), filename=str(path)).body)


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
        return rest[1] in _defined_names(node.body)
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
    for lineno, span in _code_spans(text):
        for m in _DOTTED.finditer(span):
            if not _resolves(m.group(0)):
                bad.append((lineno, m.group(0)))
    return bad


def test_every_cited_path_resolves() -> None:
    unresolved: list[str] = []
    seen: set[tuple[str, str]] = set()
    for doc in _tracked_docs():
        for lineno, ref in _references(doc):
            seen.add((doc, ref))
            if (doc, ref) not in ALLOWLIST:
                unresolved.append(f"{doc}:{lineno}: {ref}")
    stale = sorted(f"{doc}: {ref}" for doc, ref in set(ALLOWLIST) - seen)
    assert not unresolved, (
        f"{len(unresolved)} doc reference(s) name no existing file, module or "
        "attribute. Point each at the current path, or add it to ALLOWLIST with "
        "the reason:\n  " + "\n  ".join(unresolved)
    )
    assert not stale, (
        "ALLOWLIST entries that no longer occur; remove them:\n  " + "\n  ".join(stale)
    )


def test_resolver_on_known_names() -> None:
    """The resolver accepts real modules and names and rejects invented ones."""
    assert _resolves("continuum_swr_replay.parameters")
    assert _resolves("continuum_swr_replay.parameters.PROCESSED_DATA_DIR")
    assert _resolves("continuum_swr_replay.analysis.replay")
    # A lazily exported name, found in the package's _LAZY_EXPORTS table.
    assert _resolves("continuum_swr_replay.analysis.replay.fit_conditional_poisson")
    assert _resolves("continuum_swr_replay.data_loaders.load_data")
    assert not _resolves("continuum_swr_replay.parameters.NO_SUCH_NAME")
    assert not _resolves("continuum_swr_replay.no_such_module")
    assert not _resolves("continuum_swr_replay.analysis.replay.no_such_module.x")


def test_path_pattern_boundaries() -> None:
    """A path inside a longer path, or with a placeholder, is not matched."""

    def found(s: str) -> list[str]:
        return [m.group(1) for m in _FILE_PATH.finditer(s)]

    assert found("see ../scripts/qa/x.py.") == ["scripts/qa/x.py"]
    assert found("[a](src/continuum_swr_replay/types.py)") == [
        "src/continuum_swr_replay/types.py"
    ]
    assert found("tests/scripts/x.py") == []
    assert found("scripts/qa/<name>.py and scripts/figures/*.py") == []
    line = "`scripts/qa/gone.py` (removed in abc1234)"
    m = next(_FILE_PATH.finditer(line))
    assert _REMOVED.match(line[m.end() :])
