"""The one static import scanner the guard tests share.

Each file is parsed once per test session (:func:`parse`), and every import is
resolved by one rule (:func:`imported_modules`), so the layering, script-import
and lazy-import guards cannot disagree about what a module imports. Nothing is
imported: a violation is reported as a plain edge, not as an ``ImportError``.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "continuum_swr_replay"
assert SRC.is_dir(), f"package root not found at {SRC}; the scan would be vacuous"


@functools.cache
def parse(path: Path) -> ast.Module:
    """``path``'s syntax tree, parsed once per session."""
    return ast.parse(path.read_text(), filename=str(path))


@functools.cache
def module_index(root: Path) -> dict[str, Path]:
    """Every module under ``root`` by dotted name relative to ``root``.

    A package's ``__init__.py`` is named after the package; the root package
    itself is ``""``.
    """
    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        index[".".join(parts)] = path
    assert index, f"no modules under {root}; the scan would be vacuous"
    return index


def _relative(name: str, root: Path) -> str | None:
    """``name`` relative to ``root``, or None outside it."""
    if name == root.name:
        return ""
    prefix = f"{root.name}."
    return name[len(prefix) :] if name.startswith(prefix) else None


@functools.cache
def imported_modules(path: Path, root: Path) -> frozenset[str]:
    """Absolute dotted names of the modules ``path``, a file under ``root``, imports.

    An import is resolved to the most specific module it names:
    ``from pkg.mod import name`` names ``pkg.mod``, and ``from pkg import mod``
    names ``pkg.mod`` when ``mod`` is a module under ``root``. Imports from
    outside ``root`` keep the name they are written with. Imports inside
    functions and ``TYPE_CHECKING`` blocks count.

    Relative imports are resolved against the module's CONTAINING package,
    which for both ``mod.py`` and ``pkg/__init__.py`` is the directory the file
    sits in -- ``level`` 1 is that package, 2 its parent, and so on. Anchoring
    on the module's own dotted name instead would strip one segment too many
    from a package ``__init__``'s ``from ..x import y`` and let the edge escape.
    """
    index = module_index(root)
    package_parts = [root.name, *path.relative_to(root).parts[:-1]]
    targets: list[str] = []
    for node in ast.walk(parse(path)):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                kept = package_parts[: len(package_parts) - (node.level - 1)]
                base = ".".join([*kept, *([node.module] if node.module else [])])
            for alias in node.names:
                child = f"{base}.{alias.name}"
                targets.append(child if _relative(child, root) in index else base)
    modules: set[str] = set()
    for target in targets:
        relative = _relative(target, root)
        if relative is None:
            modules.add(target)
            continue
        # A name that is not itself a module (a typo'd or removed path) falls
        # back to its longest prefix that is one.
        while relative not in index and "." in relative:
            relative = relative.rsplit(".", 1)[0]
        modules.add(f"{root.name}.{relative}" if relative else root.name)
    return frozenset(modules)


@functools.cache
def module_edges(root: Path = SRC) -> frozenset[tuple[str, str]]:
    """Every ``(importer, imported)`` module edge under ``root``.

    Both ends are relative to ``root`` (see :func:`imported_modules`), except
    that an import from outside ``root`` keeps its absolute name, so a
    ``scripts.x`` import from the package surfaces as ``("…", "scripts.x")``.
    """
    edges: set[tuple[str, str]] = set()
    for importer, path in module_index(root).items():
        for module in imported_modules(path, root):
            relative = _relative(module, root)
            imported = module if relative is None else relative
            if imported != importer:
                edges.add((importer, imported))
    return frozenset(edges)
