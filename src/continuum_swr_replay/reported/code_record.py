"""Which code built a summary, and whether that code has changed since.

A summary records, when it is built, every module of the package its builder
loaded and the package's data files (with two digests each), the commit checked
out, the digest of ``uv.lock``, and, when any of those files differs from that
commit, the diff and the files git does not track. Nothing refuses a build: the
record makes any build traceable, and :func:`code_changes` says later which
recorded files no longer match. Only ``scripts/reported/check_release.py`` fails
on a stale or uncommitted summary, and it is run at milestones, not on every
commit.

Each file has two digests: ``sha256``, of its bytes, and ``code_sha256``. For a
module the second is of its syntax tree with every bare string statement (module,
class, function and attribute docstrings) removed; comments and layout are not in
the tree either, so editing them does not make a summary stale. For a data file
the two are the same. A syntax-tree dump can differ between Python versions, so
the second digest is compared only under the Python version that recorded it;
otherwise a changed file counts as changed.
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
from typing import Any

PACKAGE = "continuum_swr_replay"

NOT_RECORDED = (
    "continuum_swr_replay.reported.questions",
    "continuum_swr_replay.reported.code_record",
)
"""Modules a build loads that cannot change a value: the question list (its tags
are checked on their own, at every emit) and this record. Left out so that
rewording a question does not make every summary stale."""

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
    """
    tree = _strip_docstrings(ast.parse(source))
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def _digests(file: Path) -> dict[str, str]:
    raw = hashlib.sha256(file.read_bytes()).hexdigest()
    code = code_digest(file.read_text()) if file.suffix == ".py" else raw
    return {"sha256": raw, "code_sha256": code}


def _python() -> str:
    return ".".join(platform.python_version_tuple()[:2])


def build_files(repo: Path) -> list[Path]:
    """The files a build depends on, sorted: the package's modules loaded in this
    process (other than :data:`NOT_RECORDED`), the running script, and the
    package's data files (``data_loaders/epoch_register.csv``, the analysed
    cohort). Only files under ``repo``."""
    files = {
        Path(module.__file__).resolve()
        for name, module in list(sys.modules.items())
        if (name == PACKAGE or name.startswith(PACKAGE + "."))
        and name not in NOT_RECORDED
        and getattr(module, "__file__", None)
    }
    main = sys.modules.get("__main__")
    if isinstance(main, ModuleType) and getattr(main, "__file__", None):
        files.add(Path(main.__file__).resolve())
    package = sys.modules.get(PACKAGE)
    if package is not None and getattr(package, "__file__", None):
        files |= {
            path.resolve()
            for path in Path(package.__file__).parent.rglob("*")
            if path.is_file()
            and path.suffix not in (".py", ".pyc")
            and "__pycache__" not in path.parts
        }
    root = repo.resolve()
    return sorted(f for f in files if f.is_relative_to(root))


def _git(repo: Path, *args: str) -> str:
    """``git <args>`` in ``repo``: its stdout exactly as printed, so a diff keeps
    its last newline and can be applied."""
    return subprocess.run(
        ["git", "--no-optional-locks", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def code_record(
    repo: Path, files: Iterable[Path], started: float | None = None
) -> dict[str, Any]:
    """What a summary records about the code that built it.

    Parameters
    ----------
    repo : Path
        The git work tree; ``uv.lock`` is read from its root.
    files : iterable of Path
        The files the build depended on (:func:`build_files`).
    started : float, optional
        When the build started (``time.time()``). A file modified after it is
        listed in ``modified_during_build``: the record may not be the code that
        ran.

    Returns
    -------
    dict
        - ``head``: the commit checked out (``git rev-parse HEAD``).
        - ``python``: the major.minor version that computed the ``code_sha256``s.
        - ``uv_lock_sha256``: the sha256 of ``uv.lock``.
        - ``uncommitted``: whether any of ``files`` differs from ``head`` or is
          not tracked.
        - ``diff``: ``git diff <head>`` of the tracked files; empty when they
          match it. It applies with ``git apply`` on ``head``.
        - ``untracked``: the files git does not track; their content is in no
          commit and not in ``diff``, only their digests are kept.
        - ``modified_during_build``: the files modified after ``started``.
        - ``files``: each file, relative to ``repo``, with its ``sha256`` and
          ``code_sha256``.
    """
    root = repo.resolve()
    relative = [str(Path(f).resolve().relative_to(root)) for f in files]
    head = _git(repo, "rev-parse", "HEAD").strip()
    tracked = (
        set(_git(repo, "ls-files", "--", *relative).splitlines()) if relative else set()
    )
    untracked = sorted(set(relative) - tracked)
    diff = _git(repo, "diff", head, "--", *sorted(tracked)) if tracked else ""
    modified = (
        sorted(p for p in relative if (root / p).stat().st_mtime > started)
        if started is not None
        else []
    )
    return {
        "head": head,
        "python": _python(),
        "uv_lock_sha256": hashlib.sha256((repo / "uv.lock").read_bytes()).hexdigest(),
        "uncommitted": bool(diff) or bool(untracked),
        "diff": diff,
        "untracked": untracked,
        "modified_during_build": modified,
        "files": {path: _digests(root / path) for path in relative},
    }


def try_code_record(
    repo: Path, files: Iterable[Path], started: float | None = None
) -> dict[str, Any]:
    """:func:`code_record`, or ``{"error": ...}`` when git cannot say, so that a
    finished build is still written."""
    try:
        return code_record(repo, files, started)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"warning: no code record ({error})", file=sys.stderr)
        return {"error": str(error)}


def code_changes(record: dict[str, Any], repo: Path) -> list[str]:
    """The recorded files that differ from the working tree now, and ``uv.lock``
    if it changed; empty when the summary is current.

    A file is unchanged when its bytes match, or, under the recording Python
    version, when its ``code_sha256`` does. A removed file counts as changed.
    """
    root = repo.resolve()
    same_python = record["python"] == _python()
    changed = []
    for path, digests in sorted(record["files"].items()):
        file = root / path
        if not file.is_file():
            changed.append(path)
            continue
        now_digests = _digests(file)
        if now_digests["sha256"] == digests["sha256"]:
            continue
        if same_python and now_digests["code_sha256"] == digests["code_sha256"]:
            continue
        changed.append(path)
    lock = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    if lock != record["uv_lock_sha256"]:
        changed.append("uv.lock")
    return changed
