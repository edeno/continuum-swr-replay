"""Which ``analysis`` functions each builder function calls, read from the builder's source.

A value records the builder function that made it (``Value.function``); the
statistic is computed by the ``analysis`` functions that builder function calls.
This module finds those from the builder's source, without importing it: each
call whose name resolves, through the file's imports, to
``continuum_swr_replay.analysis``, made in the function or in a helper of the
same builder that makes no value of its own. A call to another builder function
that does make values is listed apart (:attr:`BuilderCalls.through`), since its
own calls are listed under it. ``paper/reported/questions.md`` shows the result
(:func:`~continuum_swr_replay.reported.emit.questions_view`).

Only calls are found: an ``analysis`` constant a builder reads, or a function it
passes along without calling, is not listed. A call made through a variable
(``fit = coefficient_statistics; fit.median(...)``) is not resolved.
"""

from __future__ import annotations

import ast
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

PACKAGE = "continuum_swr_replay"
ANALYSIS = f"{PACKAGE}.analysis."
BUILDERS = Path(__file__).resolve().parent
"""The folder of the summary builders: ``reported/<summary>.py``, or a package
``reported/<summary>/`` of modules."""


@dataclass(frozen=True)
class BuilderCalls:
    """What one builder function calls.

    Attributes
    ----------
    file : str
        The file that defines it, relative to the package (``reported/events.py``).
    analysis : tuple of str
        The ``analysis`` functions it calls, directly or through a helper that
        makes no value, as dotted names without the package
        (``analysis.event_properties.event_statistics.event_signal_statistics``),
        sorted.
    through : tuple of str
        The other value-making builder functions it calls, sorted.
    """

    file: str
    analysis: tuple[str, ...]
    through: tuple[str, ...]


def builder_files(name: str, root: Path = BUILDERS) -> list[Path]:
    """The source files of summary ``name``'s builder: ``<root>/<name>.py``, or
    every module of the package ``<root>/<name>/``, sorted.

    Raises
    ------
    FileNotFoundError
        If neither exists.
    """
    module = root / f"{name}.py"
    if module.is_file():
        return [module]
    files = sorted((root / name).glob("*.py"))
    if not files:
        raise FileNotFoundError(f"no builder {name!r} under {root}")
    return files


def _dotted(node: ast.expr) -> str | None:
    """``a.b.c`` of a name or an attribute chain on a name, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join([node.id, *reversed(parts)])


def _imports(tree: ast.Module, module: str) -> dict[str, str]:
    """Each name a module's top-level imports bind, with what it stands for."""
    names = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    names[alias.asname] = alias.name
                else:
                    first = alias.name.split(".")[0]
                    names[first] = first
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = module.rsplit(".", node.level)[0]
                base = f"{parent}.{base}" if base else parent
            for alias in node.names:
                names[alias.asname or alias.name] = f"{base}.{alias.name}"
    return names


class _Definitions:
    """Where a dotted name of the package is defined, read from the source under
    ``package_dir`` (the folder of ``continuum_swr_replay``)."""

    def __init__(self, package_dir: Path) -> None:
        self.package_dir = package_dir
        self._modules: dict[str, tuple[set[str], dict[str, str]] | None] = {}

    def _module(self, module: str) -> tuple[set[str], dict[str, str]] | None:
        """The functions and classes a module defines, and the names it imports;
        None if it is no module of the package."""
        if module not in self._modules:
            parts = module.split(".")
            found = None
            if parts[0] == PACKAGE:
                base = self.package_dir.joinpath(*parts[1:])
                for path in (base.with_suffix(".py"), base / "__init__.py"):
                    if path.is_file():
                        tree = ast.parse(path.read_text())
                        defined = {
                            node.name
                            for node in tree.body
                            if isinstance(
                                node,
                                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                            )
                        }
                        found = (defined, _imports(tree, module))
                        break
            self._modules[module] = found
        return self._modules[module]

    def definition(self, dotted: str, hops: int = 5) -> str | None:
        """The function or class ``dotted`` names, as ``<module>.<name>`` of the
        module that defines it (a name a package imports is followed to its
        definition), or None when it names no function or class (a constant, a
        method of one)."""
        parts = dotted.split(".")
        for split in range(len(parts) - 1, 0, -1):
            module = ".".join(parts[:split])
            found = self._module(module)
            if found is None:
                continue
            rest = parts[split:]
            if len(rest) != 1:
                return None
            defined, imports = found
            if rest[0] in defined:
                return dotted
            if rest[0] in imports and hops:
                return self.definition(imports[rest[0]], hops - 1)
            return None
        return None


def builder_calls(
    files: Iterable[Path],
    builder: str,
    value_functions: Collection[str],
    root: Path = BUILDERS,
    package_dir: Path | None = None,
) -> dict[str, BuilderCalls]:
    """What each top-level function of a builder calls.

    Parameters
    ----------
    files : iterable of Path
        The builder's source files (:func:`builder_files`), under ``root``.
    builder : str
        The builder's dotted module or package name
        (``continuum_swr_replay.reported.behavior``); an import from it, or from
        one of its modules, names a builder function.
    value_functions : collection of str
        The builder functions values name (``Value.function``). A helper in
        none of them is followed into; a call to one of them is listed in
        ``through`` instead.
    root : Path
        The folder ``files`` are under, so that ``file`` reads
        ``reported/<file>``.
    package_dir : Path, optional
        The folder of ``continuum_swr_replay`` whose source says what an
        ``analysis`` name is: a call is listed only when it names a function or
        class an ``analysis`` module defines (a method of an ``analysis``
        constant is not). By default ``root``'s parent.

    Returns
    -------
    calls : dict of str to BuilderCalls
        By function name, every top-level function of ``files``.

    Raises
    ------
    ValueError
        If two files of the builder define a function of the same name: a
        value's ``function`` would not say which.
    """
    definitions = _Definitions(package_dir or root.parent)
    defined: dict[str, tuple[str, ast.FunctionDef, dict[str, str]]] = {}
    for path in files:
        relative = path.relative_to(root)
        module = ".".join(
            (f"{PACKAGE}.reported", *relative.with_suffix("").parts)
        ).removesuffix(".__init__")
        tree = ast.parse(path.read_text())
        imports = _imports(tree, module)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                if node.name in defined:
                    raise ValueError(
                        f"{builder}: {node.name} is defined in {defined[node.name][0]} "
                        f"and {relative}"
                    )
                defined[node.name] = (f"reported/{relative}", node, imports)

    def resolve(name: str, imports: Mapping[str, str]) -> str:
        """``name`` as the dotted name it stands for in its file."""
        head, _, rest = name.partition(".")
        if head in defined and head not in imports:
            return f"{builder}.{name}"
        full = imports.get(head, head)
        return f"{full}.{rest}" if rest else full

    def local(full: str) -> str | None:
        """The builder function ``full`` names, or None."""
        if full.startswith(f"{builder}."):
            last = full.rsplit(".", 1)[-1]
            return last if last in defined else None
        return None

    direct: dict[str, tuple[set[str], set[str]]] = {}
    for name, (_, node, imports) in defined.items():
        found, helpers = set(), set()
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            dotted = _dotted(call.func)
            if dotted is None:
                continue
            full = resolve(dotted, imports)
            if full.startswith(ANALYSIS):
                target = definitions.definition(full)
                if target is not None and target.startswith(ANALYSIS):
                    found.add(target.removeprefix(f"{PACKAGE}."))
            elif (other := local(full)) is not None and other != name:
                helpers.add(other)
        direct[name] = (found, helpers)

    calls = {}
    for name, (file, _, _) in defined.items():
        found, through, seen = set(), set(), {name}
        stack = [name]
        while stack:
            current = stack.pop()
            analysis, helpers = direct[current]
            found |= analysis
            for helper in helpers - seen:
                seen.add(helper)
                if helper in value_functions:
                    through.add(helper)
                else:
                    stack.append(helper)
        calls[name] = BuilderCalls(file, tuple(sorted(found)), tuple(sorted(through)))
    return calls
