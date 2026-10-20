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
3. ``analysis/hippocampus`` must not import ``analysis/replay``. The one leaf
   module the event tables shared (``event_ids``) now lives in the foundation
   ``tables`` package.

Seven more rules are checked over the module import graph
(:func:`module_edges`) and the package graph built from it
(:func:`package_graph`). Each holds today only up to a known-violations
allowlist (:data:`KNOWN_VIOLATIONS`) that the code-organization plan works
down to empty. A rule fails on a violation the allowlist does not name AND on
an allowlist entry that no longer occurs, so the allowlist can only shrink as
edges are removed, never go stale:

- ``analysis_imports_no_visualization``: analysis code does not import
  plotting code.
- ``src_imports_no_scripts``: the package does not import the ``scripts/``
  tree.
- ``analysis_imports_no_decoding_common``: analysis code does not import
  the decode-run driver (``decoding_common``).
- ``foundation_imports_only_foundation``: a foundation module
  (:data:`FOUNDATION`) imports, from within the package, only other
  foundation modules, and nothing from ``scripts``.
- ``hippocampus_imports_no_pfc_or_inventory_comparison``: the hippocampus
  package does not reach sideways into those two analysis packages.
- ``events_imports_only_foundation``: the analysed-time and event substrate
  (``analysis.events``) imports, from within the package, only foundation
  modules, ``analysis.constants`` and itself, and nothing from ``scripts``.
- ``no_package_cycles``: the package graph is acyclic. Its entries are
  package-level edges (not module-level ones) that lie on a cycle, so a new
  module edge along a package pair it already lists is invisible to it. The
  back edges that close today's cycles are pinned at module level instead,
  by the other rules: hippocampus -> replay by rule 3, hippocampus ->
  inventory_comparison (and -> pfc) by the hippocampus rule, hippocampus ->
  decoding_common by the decoding_common rule, and a foundation or events
  module's edges by the foundation and events rules.

Imports inside functions count: a function-local import is how a cycle is
usually hidden, and it is still a dependency.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "continuum_swr_replay"
assert SRC.is_dir(), f"package root not found at {SRC}; the scan would be vacuous"

#: The only ``analysis.replay`` modules the hippocampus package may import:
#: none, since ``event_ids`` moved to ``tables``.
HIPPOCAMPUS_ALLOWED_REPLAY_MODULES: frozenset[str] = frozenset()


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


def test_hippocampus_imports_only_shared_replay_leaves() -> None:
    """Rule 3: the hippocampus package imports nothing from ``analysis/replay``."""
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


# ---------------------------------------------------------------------------
# The package-level graph and the rules checked over it.
# ---------------------------------------------------------------------------

#: Package nodes (see :func:`package_of`) that form the foundation layer: within
#: the package they import only each other, and they import nothing from
#: ``scripts``. Later tasks of the code-organization plan add the new
#: foundation homes here.
FOUNDATION = frozenset(
    {
        "data_loaders",
        "decode_files",
        "decoding_transitions",
        "intervals",
        "parameters",
        "tables",
        "types",
    }
)

#: Package nodes, besides the foundation, that ``analysis.events`` may import.
EVENTS_MAY_IMPORT = frozenset({"analysis.events", "analysis.constants"})

#: Edges each rule tolerates today, as ``(importer, imported)`` pairs of
#: module names relative to the package root (``no_package_cycles`` holds
#: package-node pairs instead). Entries are only ever deleted: each one names
#: an edge a later task removes.
KNOWN_VIOLATIONS: dict[str, set[tuple[str, str]]] = {
    "analysis_imports_no_visualization": set(),
    "src_imports_no_scripts": set(),
    "analysis_imports_no_decoding_common": {
        ("analysis.hippocampus.artifacts.detect", "decoding_common"),
        ("analysis.hippocampus.state_evidence.io", "decoding_common"),
    },
    "foundation_imports_only_foundation": {
        ("decode_files", "analysis.constants"),
    },
    "hippocampus_imports_no_pfc_or_inventory_comparison": set(),
    "events_imports_only_foundation": set(),
    # No package cycles (code-organization review, F2). hippocampus <-> pfc
    # left with the spike binning, Task 4; hippocampus -> replay and table_io
    # -> replay with the tables package, Task 5; the replay and
    # decoding_common cycles through hippocampus with the event substrate's
    # move to analysis.events, and hippocampus <-> inventory_comparison with
    # kay_version_check's move to inventory_comparison, Task 6.
    "no_package_cycles": set(),
}


def _module_index(root: Path) -> dict[str, Path]:
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
    assert index, f"no modules under {root}; the graph would be vacuous"
    return index


def _relative(name: str, root: Path) -> str | None:
    """``name`` relative to the package root, or None outside the package."""
    if name == root.name:
        return ""
    prefix = f"{root.name}."
    return name[len(prefix) :] if name.startswith(prefix) else None


def module_edges(root: Path = SRC) -> set[tuple[str, str]]:
    """Every ``(importer, imported)`` module edge, both relative to ``root``.

    An import is resolved to the most specific module it names:
    ``from pkg.mod import name`` is an edge to ``pkg.mod``, and
    ``from pkg import mod`` is an edge to ``pkg.mod`` when ``mod`` is a module.
    Imports from outside the package are kept under their absolute name, so a
    ``scripts.x`` import surfaces as ``("…", "scripts.x")``. Imports inside
    functions and ``TYPE_CHECKING`` blocks count.
    """
    index = _module_index(root)
    edges: set[tuple[str, str]] = set()
    for importer, path in index.items():
        package_parts = [root.name, *path.relative_to(root).parts[:-1]]
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    base = node.module or ""
                else:
                    kept = package_parts[: len(package_parts) - (node.level - 1)]
                    base = ".".join([*kept, *([node.module] if node.module else [])])
                for alias in node.names:
                    child = _relative(f"{base}.{alias.name}", root)
                    targets.append(f"{base}.{alias.name}" if child in index else base)
            for target in targets:
                relative = _relative(target, root)
                if relative is None:
                    edges.add((importer, target))
                    continue
                # A name that is not itself a module (a typo'd or removed path)
                # falls back to its longest prefix that is one.
                while relative not in index and "." in relative:
                    relative = relative.rsplit(".", 1)[0]
                if relative != importer:
                    edges.add((importer, relative))
    return edges


def package_of(module: str) -> str:
    """The package node of a module name relative to the package root.

    The first dotted component, except that ``analysis.<pkg>`` is its own node
    (``analysis.events.swr_source`` -> ``analysis.events``,
    ``analysis.constants`` -> ``analysis.constants``). The root package is
    ``""``.
    """
    parts = module.split(".")
    if parts[0] == "analysis" and len(parts) > 1:
        return ".".join(parts[:2])
    return parts[0]


def package_graph(root: Path = SRC) -> dict[str, set[str]]:
    """Package node -> the other package nodes it imports, within ``root``."""
    index = _module_index(root)
    graph: dict[str, set[str]] = {package_of(m): set() for m in index}
    for importer, imported in module_edges(root):
        if imported not in index:
            continue
        a, b = package_of(importer), package_of(imported)
        if a != b:
            graph[a].add(b)
    return graph


def _strongly_connected_components(graph: dict[str, set[str]]) -> list[set[str]]:
    """Tarjan's algorithm, iteratively; every node appears in one component."""
    index_of: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    components: list[set[str]] = []
    counter = 0
    for start in sorted(graph):
        if start in index_of:
            continue
        work = [(start, iter(sorted(graph[start])))]
        index_of[start] = lowlink[start] = counter
        counter += 1
        stack.append(start)
        on_stack.add(start)
        while work:
            node, children = work[-1]
            for child in children:
                if child not in index_of:
                    index_of[child] = lowlink[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(graph[child]))))
                    break
                if child in on_stack:
                    lowlink[node] = min(lowlink[node], index_of[child])
            else:
                work.pop()
                if work:
                    parent = work[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[node])
                if lowlink[node] == index_of[node]:
                    component: set[str] = set()
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        component.add(member)
                        if member == node:
                            break
                    components.append(component)
    return components


def _cycle_edges(graph: dict[str, set[str]]) -> set[tuple[str, str]]:
    """Every package edge whose two ends lie on a common cycle."""
    edges: set[tuple[str, str]] = set()
    for component in _strongly_connected_components(graph):
        if len(component) < 2:
            continue
        edges |= {(a, b) for a in component for b in graph[a] if b in component}
    return edges


def _starts_with(module: str, *prefixes: str) -> bool:
    return any(module == p or module.startswith(f"{p}.") for p in prefixes)


def rule_violations(root: Path = SRC) -> dict[str, set[tuple[str, str]]]:
    """Each graph rule's current violations, keyed like :data:`KNOWN_VIOLATIONS`."""
    edges = module_edges(root)
    index = _module_index(root)
    return {
        "analysis_imports_no_visualization": {
            (a, b)
            for a, b in edges
            if _starts_with(a, "analysis") and _starts_with(b, "visualization")
        },
        "src_imports_no_scripts": {
            (a, b) for a, b in edges if _starts_with(b, "scripts")
        },
        "analysis_imports_no_decoding_common": {
            (a, b)
            for a, b in edges
            if _starts_with(a, "analysis") and _starts_with(b, "decoding_common")
        },
        "foundation_imports_only_foundation": {
            (a, b)
            for a, b in edges
            if package_of(a) in FOUNDATION
            and (
                (b in index and package_of(b) not in FOUNDATION)
                or _starts_with(b, "scripts")
            )
        },
        "hippocampus_imports_no_pfc_or_inventory_comparison": {
            (a, b)
            for a, b in edges
            if _starts_with(a, "analysis.hippocampus")
            and _starts_with(b, "analysis.pfc", "analysis.inventory_comparison")
        },
        "events_imports_only_foundation": {
            (a, b)
            for a, b in edges
            if package_of(a) == "analysis.events"
            and (
                (
                    b in index
                    and package_of(b) not in FOUNDATION
                    and package_of(b) not in EVENTS_MAY_IMPORT
                )
                or _starts_with(b, "scripts")
            )
        },
        "no_package_cycles": _cycle_edges(package_graph(root)),
    }


def test_the_graph_is_not_vacuous() -> None:
    """The scan sees the packages the rules are about, and their known edges."""
    graph = package_graph()
    for node in (
        "analysis.events",
        "analysis.hippocampus",
        "analysis.replay",
        "analysis.pfc",
        "data_loaders",
    ):
        assert node in graph, f"{node} missing from the package graph"
    assert "analysis.hippocampus" in graph["analysis.replay"]
    assert set(FOUNDATION) <= set(graph), sorted(set(FOUNDATION) - set(graph))


@pytest.mark.parametrize("rule", sorted(KNOWN_VIOLATIONS))
def test_layering_rule(rule: str) -> None:
    """A rule's violations are exactly its known, allowlisted ones."""
    violations = rule_violations()[rule]
    new = violations - KNOWN_VIOLATIONS[rule]
    stale = KNOWN_VIOLATIONS[rule] - violations
    assert not new, f"{rule}: new violations:\n" + "\n".join(map(str, sorted(new)))
    assert not stale, (
        f"{rule}: allowlisted edges that no longer occur; delete them from "
        "KNOWN_VIOLATIONS:\n" + "\n".join(map(str, sorted(stale)))
    )


def test_graph_rules_see_function_local_imports_and_cycles(tmp_path: Path) -> None:
    """A cycle closed by a lazy, function-local import is still a cycle.

    The same synthetic tree checks that a relative import resolves to the
    module it names, that an import of ``scripts`` is seen, that analysis
    reaching ``decoding_common`` is seen, and that a foundation module
    importing any non-foundation module of the package -- not only
    ``analysis`` -- is seen, while one importing another foundation module
    is not.
    """
    root = tmp_path / "continuum_swr_replay"
    for package in ("analysis/alpha", "analysis/beta", "visualization"):
        (root / package).mkdir(parents=True)
        (root / package / "__init__.py").write_text("")
    (root / "__init__.py").write_text("")
    (root / "analysis" / "__init__.py").write_text("")
    (root / "analysis" / "alpha" / "a.py").write_text(
        "from continuum_swr_replay.analysis.beta import b\n"
    )
    (root / "analysis" / "beta" / "b.py").write_text(
        "import scripts.qa.tool\n"
        "def f():\n    from ..alpha.a import x\n    from ...visualization import v\n"
    )
    (root / "visualization" / "v.py").write_text("")
    (root / "decoding_common.py").write_text("")
    (root / "analysis" / "alpha" / "d.py").write_text(
        "def g():\n    from ... import decoding_common\n"
    )
    (root / "types.py").write_text("")
    (root / "parameters.py").write_text(
        "from . import types\nfrom .decoding_common import run\n"
    )

    found = rule_violations(root)
    assert found["no_package_cycles"] == {
        ("analysis.alpha", "analysis.beta"),
        ("analysis.beta", "analysis.alpha"),
    }
    assert found["analysis_imports_no_visualization"] == {
        ("analysis.beta.b", "visualization.v")
    }
    assert found["src_imports_no_scripts"] == {("analysis.beta.b", "scripts.qa.tool")}
    assert found["analysis_imports_no_decoding_common"] == {
        ("analysis.alpha.d", "decoding_common")
    }
    assert found["foundation_imports_only_foundation"] == {
        ("parameters", "decoding_common")
    }


def test_events_rule_allows_only_foundation_constants_and_itself(
    tmp_path: Path,
) -> None:
    """An ``analysis.events`` module may import the foundation,
    ``analysis.constants`` and its own package; a function-local import of
    an analysis package, or of ``scripts``, is a violation."""
    root = tmp_path / "continuum_swr_replay"
    for package in ("analysis/events", "analysis/hippocampus"):
        (root / package).mkdir(parents=True)
        (root / package / "__init__.py").write_text("")
    (root / "__init__.py").write_text("")
    (root / "analysis" / "__init__.py").write_text("")
    (root / "analysis" / "constants.py").write_text("")
    (root / "intervals.py").write_text("")
    (root / "analysis" / "hippocampus" / "h.py").write_text("")
    (root / "analysis" / "events" / "other.py").write_text("")
    (root / "analysis" / "events" / "e.py").write_text(
        "from continuum_swr_replay import intervals\n"
        "from ..constants import X\n"
        "from .other import Y\n"
        "import numpy\n"
        "def f():\n"
        "    from ..hippocampus import h\n"
        "    import scripts.qa.tool\n"
    )

    assert rule_violations(root)["events_imports_only_foundation"] == {
        ("analysis.events.e", "analysis.hippocampus.h"),
        ("analysis.events.e", "scripts.qa.tool"),
    }
