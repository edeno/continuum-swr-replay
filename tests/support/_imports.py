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

#: Package nodes (see :func:`package_of`) that form the foundation layer: within
#: the package they import only each other, and they import nothing from
#: ``scripts`` (``test_import_layering``).
FOUNDATION = frozenset(
    {
        "columns",
        "data_loaders",
        "decoding",
        "intervals",
        "parameters",
        "paths",
        "runs",
        "data_types",
        "tables",
    }
)


def package_of(module: str) -> str:
    """The package node of a module name relative to the package root.

    The first dotted component, except that ``analysis.<pkg>`` is its own node
    (``analysis.event_definition.analysed_time`` -> ``analysis.event_definition``,
    ``analysis.constants`` -> ``analysis.constants``). The root package is
    ``""``.
    """
    parts = module.split(".")
    if parts[0] == "analysis" and len(parts) > 1:
        return ".".join(parts[:2])
    return parts[0]


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


def relative_name(name: str, root: Path) -> str | None:
    """``name`` relative to ``root``, or None outside it."""
    if name == root.name:
        return ""
    prefix = f"{root.name}."
    return name[len(prefix) :] if name.startswith(prefix) else None


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


def defined_names(body: list[ast.stmt]) -> dict[str, ast.stmt]:
    """Names bound by ``body``, each mapped to the statement that binds it.

    A top-level ``def``/``class``/assignment/import binds a name, and so does
    each key of a package's ``_LAZY_EXPORTS`` table.
    """
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


def lazy_exports(init: Path, root: Path) -> dict[str, tuple[str, str]]:
    """``init``'s ``_LAZY_EXPORTS`` table as ``name -> (module, attribute)``.

    A target is a submodule relative to the package or an absolute module
    name, either plain (the attribute is the exported name) or as a
    ``(module, attribute)`` pair; modules come back absolute. Empty when
    ``init`` has no ``_LAZY_EXPORTS``.
    """
    package = ".".join([root.name, *init.relative_to(root).parts[:-1]])
    for node in parse(init).body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(t, ast.Name) and t.id == "_LAZY_EXPORTS" for t in targets):
            table = ast.literal_eval(node.value)
            break
    else:
        return {}
    exports: dict[str, tuple[str, str]] = {}
    for name, target in table.items():
        module, attribute = target if isinstance(target, tuple) else (target, name)
        if relative_name(module, root) is None:
            module = f"{package}.{module}"
        exports[name] = (module, attribute)
    return exports


@functools.cache
def imported_modules(path: Path, root: Path) -> frozenset[str]:
    """Absolute dotted names of the modules ``path``, a file under ``root``, imports.

    An import is resolved to the most specific module it names:
    ``from pkg.mod import name`` names ``pkg.mod``, and ``from pkg import mod``
    names ``pkg.mod`` when ``mod`` is a module under ``root``. Imports from
    outside ``root`` keep the name they are written with. Imports inside
    functions and ``TYPE_CHECKING`` blocks count, and so does each target of a
    package's ``_LAZY_EXPORTS`` table (:func:`lazy_exports`), an import
    deferred to first attribute access.

    Relative imports are resolved against the module's CONTAINING package,
    which for both ``mod.py`` and ``pkg/__init__.py`` is the directory the file
    sits in -- ``level`` 1 is that package, 2 its parent, and so on. Anchoring
    on the module's own dotted name instead would strip one segment too many
    from a package ``__init__``'s ``from ..x import y`` and let the edge escape.
    """
    index = module_index(root)
    package_parts = [root.name, *path.relative_to(root).parts[:-1]]
    # (module, name imported from it, or None for ``import module``)
    imports: list[tuple[str, str | None]] = list(lazy_exports(path, root).values())
    for node in ast.walk(parse(path)):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, None) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                kept = package_parts[: len(package_parts) - (node.level - 1)]
                base = ".".join([*kept, *([node.module] if node.module else [])])
            imports.extend((base, alias.name) for alias in node.names)
    modules: set[str] = set()
    for base, name in imports:
        target = base if name is None else f"{base}.{name}"
        if relative_name(target, root) not in index:
            target = base
        relative = relative_name(target, root)
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
            relative = relative_name(module, root)
            imported = module if relative is None else relative
            if imported != importer:
                edges.add((importer, imported))
    return frozenset(edges)
