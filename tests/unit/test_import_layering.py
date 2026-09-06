"""The package's import layering, checked statically.

Three directed dependencies are forbidden by design, and each one is cheap to
reintroduce by adding a convenient import. The rules are checked with ``ast``
over the source text -- nothing is imported, so a violation is reported as a
plain edge rather than as an ``ImportError`` from a cycle:

1. Covariate builders must not import the model ladders. A builder writes
   columns; a ladder names them. Letting a builder read ``model_specs`` couples
   the persisted table to the fitted specification, and the shared design
   constants (``design_constants``) exist precisely so both sides can read the
   same numbers without that edge.
2. ``data_loaders`` must not import ``analysis``. Loading is upstream of every
   analysis; the reverse edge makes the loaders unusable on their own and
   invites an import cycle.
3. ``analysis/hippocampus`` must not import ``analysis/replay``, except for the
   two leaf modules the event tables genuinely share.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "continuum_swr_replay"
assert SRC.is_dir(), f"package root not found at {SRC}; the scan would be vacuous"

#: The only ``analysis.replay`` modules the hippocampus package may import.
HIPPOCAMPUS_ALLOWED_REPLAY_MODULES = frozenset({"bouts", "event_ids"})


def _imported_modules(path: Path, root: Path) -> list[str]:
    """Absolute dotted module names imported by ``path``.

    Relative imports are resolved against the module's CONTAINING package, which
    for both ``mod.py`` and ``pkg/__init__.py`` is the directory the file sits
    in -- ``level`` 1 is that package, 2 its parent, and so on. Anchoring on the
    directory rather than on the module's own dotted name is what keeps a
    package's ``__init__.py`` honest: resolving ``from ..model_specs import X``
    against ``…covariates.__init__`` would strip one segment too many and let
    the edge escape the rule.
    """
    package_parts = [root.name, *path.relative_to(root).parts[:-1]]
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                base_parts = package_parts[: len(package_parts) - (node.level - 1)]
                base = ".".join([*base_parts, *([node.module] if node.module else [])])
            modules.append(base)
            modules.extend(f"{base}.{alias.name}" for alias in node.names)
    return modules


def _modules_under(root: Path, *relative_parts: str) -> list[Path]:
    """Every module under ``root/relative_parts``; never an empty scan."""
    directory = root.joinpath(*relative_parts)
    paths = sorted(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)
    assert paths, f"no modules under {directory}; the rule would pass vacuously"
    return paths


def _violations(
    paths: list[Path], predicate: Callable[[str], bool], *, root: Path
) -> list[str]:
    """Every ``file -> imported module`` edge ``predicate`` rejects."""
    found: list[str] = []
    for path in paths:
        for module in _imported_modules(path, root):
            if predicate(module):
                found.append(f"{path.relative_to(root)} -> {module}")
    return sorted(set(found))


def _model_specs_edges(root: Path) -> list[str]:
    """Rule 1's violations under an arbitrary package root."""
    target = f"{root.name}.analysis.replay.model_specs"
    return _violations(
        _modules_under(root, "analysis", "replay", "covariates"),
        lambda m: m == target or m.startswith(f"{target}."),
        root=root,
    )


def test_covariate_builders_do_not_import_model_specs() -> None:
    """Rule 1: no ``covariates/`` module reads the fitted model ladders."""
    violations = _model_specs_edges(SRC)
    assert not violations, (
        "covariate builders must read design_constants, not model_specs:\n"
        + "\n".join(violations)
    )


def test_relative_import_in_a_package_init_is_not_missed(tmp_path: Path) -> None:
    """A ``covariates/__init__.py`` must not escape rule 1 on its dot count.

    ``from ..model_specs import BIN_S`` inside ``covariates/__init__.py`` names
    the same module as the absolute spelling. Resolving it against the module's
    own dotted name instead of its package would yield
    ``analysis.model_specs`` -- a module that does not exist -- and the edge
    would silently pass.
    """
    root = tmp_path / "continuum_swr_replay"
    covariates = root / "analysis" / "replay" / "covariates"
    covariates.mkdir(parents=True)
    (covariates / "__init__.py").write_text("from ..model_specs import BIN_S\n")

    assert _model_specs_edges(root) == [
        "analysis/replay/covariates/__init__.py -> "
        "continuum_swr_replay.analysis.replay.model_specs",
        "analysis/replay/covariates/__init__.py -> "
        "continuum_swr_replay.analysis.replay.model_specs.BIN_S",
    ]


def test_data_loaders_do_not_import_analysis() -> None:
    """Rule 2: loading is upstream of analysis."""
    target = "continuum_swr_replay.analysis"
    violations = _violations(
        _modules_under(SRC, "data_loaders"),
        lambda m: m == target or m.startswith(f"{target}."),
        root=SRC,
    )
    assert not violations, "data_loaders must not import analysis:\n" + "\n".join(
        violations
    )


@pytest.mark.xfail(
    strict=True,
    reason="moves land with the `_helpers` split",
)
def test_hippocampus_imports_only_shared_replay_leaves() -> None:
    """Rule 3: the hippocampus package shares only ``bouts`` and ``event_ids``."""
    target = "continuum_swr_replay.analysis.replay"
    allowed = {f"{target}.{name}" for name in HIPPOCAMPUS_ALLOWED_REPLAY_MODULES}

    def forbidden(module: str) -> bool:
        if module != target and not module.startswith(f"{target}."):
            return False
        return not any(
            module == name or module.startswith(f"{name}.") for name in allowed
        )

    violations = _violations(
        _modules_under(SRC, "analysis", "hippocampus"), forbidden, root=SRC
    )
    assert not violations, (
        "analysis/hippocampus may import only "
        f"{sorted(HIPPOCAMPUS_ALLOWED_REPLAY_MODULES)} from analysis/replay:\n"
        + "\n".join(violations)
    )
