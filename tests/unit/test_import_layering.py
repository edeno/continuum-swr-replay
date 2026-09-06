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

#: The only ``analysis.replay`` modules the hippocampus package may import.
HIPPOCAMPUS_ALLOWED_REPLAY_MODULES = frozenset({"bouts", "event_ids"})


def _imported_modules(path: Path) -> list[str]:
    """Absolute dotted module names imported by ``path``.

    Relative imports are resolved against the file's own package so that
    ``from ..model_specs import X`` and
    ``from continuum_swr_replay.analysis.replay.model_specs import X`` are the
    same edge.
    """
    package_parts = ["continuum_swr_replay", *path.relative_to(SRC).parts[:-1]]
    if path.name != "__init__.py":
        package_parts.append(path.stem)
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                # ``level`` counts steps up from the module's own name.
                base_parts = package_parts[: -node.level]
                base = ".".join([*base_parts, *([node.module] if node.module else [])])
            modules.append(base)
            modules.extend(f"{base}.{alias.name}" for alias in node.names)
    return modules


def _modules_under(*relative_parts: str) -> list[Path]:
    root = SRC.joinpath(*relative_parts)
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _violations(paths: list[Path], predicate: Callable[[str], bool]) -> list[str]:
    """Every ``file -> imported module`` edge ``predicate`` rejects."""
    found: list[str] = []
    for path in paths:
        for module in _imported_modules(path):
            if predicate(module):
                found.append(f"{path.relative_to(SRC)} -> {module}")
    return sorted(set(found))


def test_covariate_builders_do_not_import_model_specs() -> None:
    """Rule 1: no ``covariates/`` module reads the fitted model ladders."""
    target = "continuum_swr_replay.analysis.replay.model_specs"
    violations = _violations(
        _modules_under("analysis", "replay", "covariates"),
        lambda m: m == target or m.startswith(f"{target}."),
    )
    assert not violations, (
        "covariate builders must read design_constants, not model_specs:\n"
        + "\n".join(violations)
    )


def test_data_loaders_do_not_import_analysis() -> None:
    """Rule 2: loading is upstream of analysis."""
    target = "continuum_swr_replay.analysis"
    violations = _violations(
        _modules_under("data_loaders"),
        lambda m: m == target or m.startswith(f"{target}."),
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

    violations = _violations(_modules_under("analysis", "hippocampus"), forbidden)
    assert not violations, (
        "analysis/hippocampus may import only "
        f"{sorted(HIPPOCAMPUS_ALLOWED_REPLAY_MODULES)} from analysis/replay:\n"
        + "\n".join(violations)
    )
