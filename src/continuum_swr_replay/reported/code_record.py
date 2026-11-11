"""Which code built a summary, and whether that code has changed since.

A summary records, when it is built, every module of the package its builder
loaded (with two digests each), the digest of ``uv.lock``, and, when the working
tree had uncommitted changes, the diff of those modules. Nothing refuses a build:
the record makes any build traceable, and :func:`code_changes` says later which
recorded modules no longer match. Only ``scripts/reported/check_release.py``
fails on a stale or uncommitted summary, and it is run at milestones, not on every
commit.

Each module has two digests: ``sha256``, of the file's bytes, and ``code_sha256``,
of its syntax tree with docstrings removed. Comments, docstrings and layout are not
in the second, so editing them does not make a summary stale. A syntax-tree dump
can differ between Python versions, so the second digest is compared only under
the Python version that recorded it; otherwise a changed file counts as changed.
"""

from __future__ import annotations

import ast
import hashlib
import platform
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

from continuum_swr_replay.runs.provenance import git

PACKAGE = "continuum_swr_replay"


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return tree


def code_digest(source: str) -> str:
    """The sha256 of ``source``'s syntax tree with docstrings removed.

    Two sources that differ only in comments, docstrings or layout have the same
    digest; a change to what the code does changes it.
    """
    tree = _strip_docstrings(ast.parse(source))
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def _python() -> str:
    return ".".join(platform.python_version_tuple()[:2])


def loaded_modules(repo: Path) -> list[Path]:
    """The files under ``repo`` of the package's modules loaded in this process,
    plus the running script, sorted."""
    files = {
        Path(module.__file__).resolve()
        for name, module in list(sys.modules.items())
        if (name == PACKAGE or name.startswith(PACKAGE + "."))
        and getattr(module, "__file__", None)
    }
    main = sys.modules.get("__main__")
    if isinstance(main, ModuleType) and getattr(main, "__file__", None):
        files.add(Path(main.__file__).resolve())
    root = repo.resolve()
    return sorted(f for f in files if f.is_relative_to(root))


def code_record(repo: Path, files: Iterable[Path]) -> dict[str, Any]:
    """What a summary records about the code that built it.

    Parameters
    ----------
    repo : Path
        The git work tree; ``uv.lock`` is read from its root.
    files : iterable of Path
        The source files the build loaded (:func:`loaded_modules`).

    Returns
    -------
    dict
        - ``python``: the major.minor version that computed the ``code_sha256``s.
        - ``uv_lock_sha256``: the sha256 of ``uv.lock``.
        - ``uncommitted``: whether any of ``files`` differs from ``HEAD``.
        - ``diff``: ``git diff HEAD`` of those files; empty when ``uncommitted``
          is False.
        - ``modules``: each file, relative to ``repo``, with its ``sha256`` and
          ``code_sha256``.
    """
    root = repo.resolve()
    relative = [str(Path(f).resolve().relative_to(root)) for f in files]
    diff = git(repo, "diff", "HEAD", "--", *relative) if relative else ""
    return {
        "python": _python(),
        "uv_lock_sha256": hashlib.sha256((repo / "uv.lock").read_bytes()).hexdigest(),
        "uncommitted": bool(diff),
        "diff": diff,
        "modules": {
            path: {
                "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest(),
                "code_sha256": code_digest((root / path).read_text()),
            }
            for path in relative
        },
    }


def code_changes(record: dict[str, Any], repo: Path) -> list[str]:
    """The recorded files whose code differs from the working tree now, and
    ``uv.lock`` if it changed; empty when the summary is current.

    A file is unchanged when its bytes match, or, under the recording Python
    version, when its docstring-stripped syntax tree does. A removed file counts
    as changed.
    """
    root = repo.resolve()
    same_python = record["python"] == _python()
    changed = []
    for path, digests in sorted(record["modules"].items()):
        file = root / path
        if not file.is_file():
            changed.append(path)
            continue
        if hashlib.sha256(file.read_bytes()).hexdigest() == digests["sha256"]:
            continue
        if same_python and code_digest(file.read_text()) == digests["code_sha256"]:
            continue
        changed.append(path)
    lock = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    if lock != record["uv_lock_sha256"]:
        changed.append("uv.lock")
    return changed
