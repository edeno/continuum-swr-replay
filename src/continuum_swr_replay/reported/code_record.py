"""Which code built a summary, whether that code has changed since, and whether
the summary's inputs still match.

A summary records, when it is built, the files the build depended on: every
package module loaded during the build (the summary's builder and the building
script's own imports), the script itself, and every non-Python file under the
package directory (today only ``data_loaders/epoch_register.csv``, the analysed
cohort), each with two digests. It also records the commit checked out, the digest
of ``uv.lock``, and which of those files or ``uv.lock`` differ from that commit.
The rule is that a summary in ``paper/reported/`` is built from committed code:
``scripts/reported/summarize.py`` refuses to write one there otherwise (a trial
build goes to another folder), so the commit and the digests are the whole
record. :func:`code_changes` says later which recorded files no longer match;
that never refuses anything, and only ``scripts/reported/check_release.py``
fails on a stale summary, at milestones.

Each file has two digests: ``sha256``, of its bytes, and ``code_sha256``. For a
module the second is of its syntax tree with every bare string statement (module,
class, function and attribute docstrings) removed; comments and layout are not in
the tree either, so editing them does not make a summary stale. For a data file,
or a module that does not parse, the two are the same. A syntax-tree dump can
differ between Python versions, so the second digest is compared only under the
Python version that recorded it; otherwise a changed file counts as changed.
"""

from __future__ import annotations

import ast
import hashlib
import platform
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any, TypedDict

PACKAGE = "continuum_swr_replay"

NOT_RECORDED = (
    "continuum_swr_replay.reported.questions",
    "continuum_swr_replay.reported.code_record",
)
"""Modules a build loads that cannot change a value: the question list (its tags
are checked when each value is built, and reported at every emit) and this
record. Left out so that rewording a question does not make every summary stale."""

LOCK = "uv.lock"


class FileDigests(TypedDict):
    sha256: str
    code_sha256: str


class CodeRecord(TypedDict):
    """What :func:`code_record` returns; its keys are described there."""

    head: str
    python: str
    uv_lock_sha256: str
    uncommitted: list[str]
    files: dict[str, FileDigests]


_STATEMENT_LISTS = ("body", "orelse", "finalbody")


def _is_bare_string(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """``tree`` without bare string statements, which do nothing when run."""
    for node in ast.walk(tree):
        for field in _STATEMENT_LISTS:
            statements = getattr(node, field, None)
            if isinstance(statements, list) and statements:
                kept = [s for s in statements if not _is_bare_string(s)]
                if len(kept) != len(statements):
                    setattr(node, field, kept or [ast.Pass()])
    return tree


def code_digest(source: str) -> str:
    """The sha256 of ``source``'s syntax tree with its docstrings removed.

    Two sources that differ only in comments, docstrings (attribute docstrings
    included) or layout have the same digest; a change to what the code does
    changes it.

    Raises
    ------
    SyntaxError
        If ``source`` does not parse.
    """
    tree = _strip_docstrings(ast.parse(source))
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def _digests(file: Path) -> FileDigests:
    raw = hashlib.sha256(file.read_bytes()).hexdigest()
    code = raw
    if file.suffix == ".py":
        try:
            code = code_digest(file.read_text())
        except SyntaxError:
            pass
    return {"sha256": raw, "code_sha256": code}


def _python() -> str:
    return ".".join(platform.python_version_tuple()[:2])


def build_files(repo: Path) -> list[Path]:
    """The files a build depends on, sorted: the package's modules loaded in this
    process (other than :data:`NOT_RECORDED`), the running script, and every
    non-Python file under the package directory. Only files under ``repo``."""
    files = set()
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if (name == PACKAGE or name.startswith(PACKAGE + ".")) and (
            file and name not in NOT_RECORDED
        ):
            files.add(Path(file).resolve())
    main = sys.modules.get("__main__")
    main_file = (
        getattr(main, "__file__", None) if isinstance(main, ModuleType) else None
    )
    if main_file:
        files.add(Path(main_file).resolve())
    package_file = getattr(sys.modules.get(PACKAGE), "__file__", None)
    if package_file:
        files |= {
            path.resolve()
            for path in Path(package_file).parent.rglob("*")
            if path.is_file()
            and path.suffix not in (".py", ".pyc")
            and "__pycache__" not in path.parts
        }
    root = repo.resolve()
    return sorted(f for f in files if f.is_relative_to(root))


def _git(repo: Path, *args: str) -> str:
    """The output of ``git <args>`` in ``repo``; a failure raises, with git's own
    message."""
    result = subprocess.run(
        ["git", "--no-optional-locks", *args],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, result.args, result.stdout, result.stderr
        )
    return result.stdout


def code_record(repo: Path, files: Iterable[Path]) -> CodeRecord:
    """What a summary records about the code that built it.

    Parameters
    ----------
    repo : Path
        The git work tree; ``uv.lock`` is read from its root.
    files : iterable of Path
        The files the build depended on (:func:`build_files`).

    Returns
    -------
    CodeRecord
        - ``head``: the commit checked out (``git rev-parse HEAD``).
        - ``python``: the major.minor version that computed the ``code_sha256``s.
        - ``uv_lock_sha256``: the sha256 of ``uv.lock``.
        - ``uncommitted``: those of ``files`` and ``uv.lock`` that differ from
          ``head`` or that git does not track, sorted; empty when the build is
          the code of ``head``.
        - ``files``: each file, relative to ``repo``, with its ``sha256`` and
          ``code_sha256``.

    Raises
    ------
    OSError, subprocess.CalledProcessError
        If a file or ``uv.lock`` cannot be read, or git fails.
    """
    root = repo.resolve()
    relative = [str(Path(f).resolve().relative_to(root)) for f in files]
    head = _git(repo, "rev-parse", "HEAD").strip()
    paths = {*relative, LOCK}
    tracked = set(_git(repo, "ls-files", "--", *sorted(paths)).splitlines())
    changed = (
        _git(repo, "diff", "--name-only", head, "--", *sorted(tracked)).splitlines()
        if tracked
        else []
    )
    return {
        "head": head,
        "python": _python(),
        "uv_lock_sha256": hashlib.sha256((repo / LOCK).read_bytes()).hexdigest(),
        "uncommitted": sorted({*changed, *(paths - tracked)}),
        "files": {path: _digests(root / path) for path in relative},
    }


def code_changes(record: CodeRecord, repo: Path) -> list[str]:
    """The recorded files that differ from the working tree now, and ``uv.lock``
    if it changed; empty when the summary is current.

    A file is unchanged when its bytes match, or, under the recording Python
    version, when its ``code_sha256`` does. A removed file, or a module that no
    longer parses, counts as changed.

    """
    root = repo.resolve()
    same_python = record["python"] == _python()
    changed = []
    for path, digests in sorted(record["files"].items()):
        file = root / path
        if not file.is_file():
            changed.append(path)
            continue
        now = _digests(file)
        if now["sha256"] == digests["sha256"]:
            continue
        parses = now["code_sha256"] != now["sha256"]
        if same_python and parses and now["code_sha256"] == digests["code_sha256"]:
            continue
        changed.append(path)
    lock = root / LOCK
    if not lock.is_file() or (
        hashlib.sha256(lock.read_bytes()).hexdigest() != record["uv_lock_sha256"]
    ):
        changed.append(LOCK)
    return changed


def input_changes(inputs: Iterable[dict[str, Any]], root: Path) -> dict[str, list[str]]:
    """Which of a summary's recorded inputs no longer match, by what can be said.

    Returns
    -------
    dict
        - ``changed``: single files whose sha256 differs now.
        - ``missing``: single files not on this machine.
        - ``unchecked``: sets of files (records with ``n_files``); their records
          name a label, not the selection the builder made, so they cannot be
          re-checked without rebuilding.
    """
    found: dict[str, list[str]] = {"changed": [], "missing": [], "unchecked": []}
    for record in inputs:
        path = record["path"]
        if "n_files" in record:
            found["unchecked"].append(path)
            continue
        file = Path(path) if Path(path).is_absolute() else root / path
        if not file.is_file():
            found["missing"].append(path)
        elif hashlib.sha256(file.read_bytes()).hexdigest() != record["sha256"]:
            found["changed"].append(path)
    return found
