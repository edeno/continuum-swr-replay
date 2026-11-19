"""Which files of the checkout built a summary, and whether they have changed since.

A summary records, when it is built, the files of the checkout the build
depended on: every package module loaded during the build (the summary's builder
and the building script's own imports), the script itself, every non-Python file
under the package directory that git does not ignore (today only
``data_loaders/epoch_register.csv``, the analyzed cohort), and the input files
the repository tracks (the literature survey table, the committed sweep
summaries), each with two digests. It also records the commit checked out, the
digest of ``uv.lock``, and which of those files or ``uv.lock`` differ from that
commit. The rule is that a summary in ``paper/reported/`` is built from committed
files: ``scripts/reported/summarize.py`` refuses to write one there otherwise (a
trial build goes to another folder), so the commit and the digests are the whole
record. A file differs from the commit when its content does, whatever git's
index says of it. :func:`code_changes` says later which recorded files no longer
match; that never refuses anything, and only
``scripts/reported/check_release.py`` fails on a stale summary, at milestones.

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
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import TypedDict

from continuum_swr_replay.runs.io import sha256_file
from continuum_swr_replay.runs.provenance import commit_label as runs_commit_label
from continuum_swr_replay.runs.provenance import git, git_commit

PACKAGE = "continuum_swr_replay"

NOT_RECORDED = (
    "continuum_swr_replay.reported.questions",
    "continuum_swr_replay.reported.code_record",
)
"""Modules a build loads that cannot change a value: the question list (its tags
are checked when a summary is written, and reported at every emit) and this
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
    data = file.read_bytes()
    raw = hashlib.sha256(data).hexdigest()
    code = raw
    if file.suffix == ".py":
        try:
            code = code_digest(data.decode())
        except SyntaxError:
            pass
    return {"sha256": raw, "code_sha256": code}


def _python() -> str:
    return ".".join(platform.python_version_tuple()[:2])


def build_files(repo: Path) -> list[Path]:
    """The code a build depends on, sorted: the package's modules loaded in this
    process (other than :data:`NOT_RECORDED`), the running script, and every
    non-Python file under the package directory that git does not ignore.

    Raises
    ------
    ValueError
        If a loaded module of the package, or the running script, is outside
        ``repo``: the record could not vouch for it.
    """
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
    root = repo.resolve()
    outside = sorted(str(f) for f in files if not f.is_relative_to(root))
    if outside:
        raise ValueError(f"the build ran code from outside {root}: {outside}")
    package_file = getattr(sys.modules.get(PACKAGE), "__file__", None)
    if package_file:
        listed = git(
            repo,
            *("ls-files", "-z", "--cached", "--others", "--exclude-standard", "--"),
            str(Path(package_file).resolve().parent),
        )
        files |= {
            root / path
            for path in listed.split("\0")
            if path
            and Path(path).suffix not in (".py", ".pyc")
            and (root / path).is_file()
        }
    return sorted(files)


def _committed_blobs(repo: Path, paths: Iterable[str]) -> dict[str, str]:
    """The git blob id of each of ``paths`` (relative to ``repo``) in the commit
    checked out; a path the commit does not hold is absent."""
    listed = git(repo, "ls-tree", "-r", "-z", "HEAD", "--", *sorted(paths))
    blobs = {}
    for entry in listed.split("\0"):
        if entry:
            mode_type_blob, path = entry.split("\t", 1)
            blobs[path] = mode_type_blob.split()[2]
    return blobs


def tracked(repo: Path, paths: Iterable[str]) -> list[str]:
    """Those of ``paths`` (relative to ``repo``) that the commit checked out holds."""
    paths = list(paths)
    committed = _committed_blobs(repo, paths) if paths else {}
    return [path for path in paths if path in committed]


def _relative(repo: Path, files: Iterable[Path]) -> list[str]:
    root = repo.resolve()
    return [str(Path(f).resolve().relative_to(root)) for f in files]


def uncommitted(repo: Path, files: Iterable[Path]) -> list[str]:
    """Those of ``files`` and ``uv.lock`` whose content is not what the commit
    checked out holds for them (changed, or not in the commit), relative to
    ``repo`` and sorted: :func:`code_record`'s ``uncommitted``, without the
    digests. The working file's content is compared, so a file hidden from
    ``git status`` (``assume-unchanged``, ``skip-worktree``) is still found."""
    paths = sorted({*_relative(repo, files), LOCK})
    committed = _committed_blobs(repo, paths)
    working = dict(
        zip(paths, git(repo, "hash-object", "--", *paths).split(), strict=True)
    )
    return [p for p in paths if committed.get(p) != working[p]]


def code_record(repo: Path, files: Iterable[Path]) -> CodeRecord:
    """What a summary records about the files of the checkout that built it.

    Parameters
    ----------
    repo : Path
        The git work tree; ``uv.lock`` is read from its root.
    files : iterable of Path
        The files the build depended on: its code (:func:`build_files`) and the
        inputs the repository tracks.

    Returns
    -------
    CodeRecord
        - ``head``: the commit checked out (``git rev-parse HEAD``).
        - ``python``: the major.minor version that computed the ``code_sha256``s.
        - ``uv_lock_sha256``: the sha256 of ``uv.lock``.
        - ``uncommitted``: those of ``files`` and ``uv.lock`` whose content is
          not what ``head`` holds for them (:func:`uncommitted`); empty when
          the build is the files of ``head``.
        - ``files``: each file, relative to ``repo``, with its ``sha256`` and
          ``code_sha256``.

    Raises
    ------
    OSError, subprocess.CalledProcessError, ValueError
        If a file or ``uv.lock`` cannot be read, git fails, or a file is outside
        ``repo``.
    """
    files = list(files)
    root = repo.resolve()
    return {
        "head": git_commit(repo),
        "python": _python(),
        "uv_lock_sha256": sha256_file(repo / LOCK),
        "uncommitted": uncommitted(repo, files),
        "files": {path: _digests(root / path) for path in _relative(repo, files)},
    }


def commit_label(record: CodeRecord) -> str:
    """The commit a summary names as its build: the first 12 characters of
    ``head``, with ``-dirty`` when a file the build used was uncommitted. Other
    uncommitted files in the checkout (a summary just written, a draft) do not
    make it dirty."""
    return runs_commit_label(record["head"], bool(record["uncommitted"]))


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
        if sha256_file(file) == digests["sha256"]:
            continue
        if same_python and _digests(file)["code_sha256"] == digests["code_sha256"]:
            continue
        changed.append(path)
    lock = root / LOCK
    if not lock.is_file() or sha256_file(lock) != record["uv_lock_sha256"]:
        changed.append(LOCK)
    return changed
