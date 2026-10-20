"""What code a run ran: the git checkout and the environment it ran from.

Every runner that records its commit reads it through :func:`git`, which runs
``git --no-optional-locks``: by default ``git status`` and ``git describe
--dirty`` refresh the index and write it back under a lock, and the runners
record their commit in a checkout that other work commits in. A decoder or a
detection runner records the fuller :func:`provenance_record`.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any

__all__ = ["git", "git_commit", "git_dirty", "provenance_record"]


def git(repo: Path, *args: str, check: bool = True) -> str:
    """Run ``git --no-optional-locks <args>`` in `repo`; return its stdout, stripped.

    Parameters
    ----------
    repo : Path
        The work tree (any folder inside it).
    *args : str
        The git command, such as ``"rev-parse", "HEAD"``.
    check : bool, optional
        Raise on a non-zero exit (the default). With False a failure returns
        whatever git printed, usually ``""``.

    Raises
    ------
    subprocess.CalledProcessError
        If git exits non-zero and `check` is True.
    OSError
        If there is no git to run.
    """
    return subprocess.run(
        ["git", "--no-optional-locks", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=check,
    ).stdout.strip()


def git_commit(repo: Path) -> str:
    """``git rev-parse HEAD`` in `repo`: the full commit checked out."""
    return git(repo, "rev-parse", "HEAD")


def git_dirty(repo: Path) -> bool:
    """Whether a tracked file in `repo` differs from its commit.

    True when ``git status --porcelain --untracked-files=no`` lists anything.
    Untracked files are not counted: the work tree holds untracked files from
    unrelated work, and they cannot change how the code ran.
    """
    return bool(git(repo, "status", "--porcelain", "--untracked-files=no"))


def provenance_record(repo: Path) -> dict[str, Any]:
    """Record the code and environment a decode or a detection run runs from.

    Each decoder logs this once per epoch, after START, as ``PROVENANCE `` plus
    ``json.dumps(record, sort_keys=True)``, and a fresh save stores the same JSON
    in ``results.nc`` as ``attrs["provenance"]``, where it outlives a resume.

    Parameters
    ----------
    repo : Path
        The git work tree the decode runs from. ``uv.lock`` is read from its root.

    Returns
    -------
    dict
        - ``commit``: ``git rev-parse HEAD`` (:func:`git_commit`).
        - ``dirty``: :func:`git_dirty`; untracked files are not counted.
        - ``uv_lock_sha256``: the sha256 of ``repo / "uv.lock"``.
        - ``ripple_detection_commit``: the git commit ``ripple_detection`` was
          installed from, read from its ``direct_url.json``, or None when that
          names no commit. Its version string comes from the last tag, so it
          does not identify the commit.
        - ``non_local_detector_version``: the installed version.
        - ``cuda_plugin_version``: the installed ``jax-cuda12-plugin`` version,
          or None when it is not installed.
    """
    direct_url = importlib.metadata.distribution("ripple-detection").read_text(
        "direct_url.json"
    )
    vcs_info = json.loads(direct_url).get("vcs_info", {}) if direct_url else {}
    try:
        cuda_plugin_version = importlib.metadata.version("jax-cuda12-plugin")
    except importlib.metadata.PackageNotFoundError:
        cuda_plugin_version = None

    return {
        "commit": git_commit(repo),
        "dirty": git_dirty(repo),
        "uv_lock_sha256": hashlib.sha256((repo / "uv.lock").read_bytes()).hexdigest(),
        "ripple_detection_commit": vcs_info.get("commit_id"),
        "non_local_detector_version": importlib.metadata.version("non-local-detector"),
        "cuda_plugin_version": cuda_plugin_version,
    }
