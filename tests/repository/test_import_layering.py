"""The package's import layering, checked statically.

Each rule names edges that are forbidden by design and cheap to reintroduce by
adding a convenient import. The rules are predicates over the module import
graph (:func:`tests.support._imports.module_edges`) and the package graph built
from it (:func:`package_graph`). Nothing is imported, so a violation is
reported as a plain edge rather than as an ``ImportError`` from a cycle:

- ``covariates_import_no_model_specs``: covariate builders must not import the
  model ladders. A builder writes columns; a ladder names them. Letting a
  builder read ``model_specs`` couples the persisted table to the fitted
  specification, and the shared design constants (``design_constants``) exist
  precisely so both sides can read the same numbers without that edge.
- ``data_loaders_import_no_analysis``: loading is upstream of every analysis;
  the reverse edge makes the loaders unusable on their own and invites an
  import cycle.
- ``analysis_imports_no_visualization``: analysis code does not import
  plotting code.
- ``src_imports_no_scripts``: the package does not import the ``scripts/``
  tree.
- ``foundation_imports_only_foundation``: a foundation module
  (:data:`FOUNDATION`) imports, from within the package, only other
  foundation modules, and nothing from ``scripts``.
- ``events_imports_only_foundation``: the analysed-time and event substrate
  (``analysis.events``) imports, from within the package, only foundation
  modules, ``analysis.constants``, ``analysis.shared_statistics`` and itself, and
  nothing from ``scripts``.
- ``hippocampus_imports_no_behavior`` and
  ``hippocampus_imports_no_pfc_or_inventory_comparison``: the hippocampus
  package does not reach into those analysis packages. Today each of them
  imports hippocampus, so a back edge is also a cycle; these rules hold
  hippocampus below them even if one stops importing it.
- ``no_package_cycles``: the package graph is acyclic.

Imports inside functions count: a function-local import is how a cycle is
usually hidden, and it is still a dependency.
"""

from __future__ import annotations

import graphlib
from pathlib import Path

import pytest

from tests.support._imports import (
    FOUNDATION,
    SRC,
    module_edges,
    module_index,
    package_of,
)

#: Package nodes, besides the foundation, that ``analysis.events`` may import.
EVENTS_MAY_IMPORT = frozenset(
    {"analysis.events", "analysis.constants", "analysis.shared_statistics"}
)


def package_graph(root: Path = SRC) -> dict[str, set[str]]:
    """Package node -> the other package nodes it imports, within ``root``."""
    index = module_index(root)
    graph: dict[str, set[str]] = {package_of(m): set() for m in index}
    for importer, imported in module_edges(root):
        if imported not in index:
            continue
        a, b = package_of(importer), package_of(imported)
        if a != b:
            graph[a].add(b)
    return graph


def _cycle_edges(graph: dict[str, set[str]]) -> set[tuple[str, str]]:
    """The ``(importer, imported)`` package edges of one cycle, or none."""
    try:
        graphlib.TopologicalSorter(graph).prepare()
    except graphlib.CycleError as error:
        # Each node of the cycle is a predecessor -- an import -- of the next.
        cycle = error.args[1]
        return set(zip(cycle[1:], cycle[:-1], strict=True))
    return set()


def _starts_with(module: str, *prefixes: str) -> bool:
    return any(module == p or module.startswith(f"{p}.") for p in prefixes)


def rule_violations(root: Path = SRC) -> dict[str, set[tuple[str, str]]]:
    """Each rule's ``(importer, imported)`` violations, relative to ``root``."""
    edges = module_edges(root)
    index = module_index(root)

    def outside(b: str, allowed: frozenset[str]) -> bool:
        return (b in index and package_of(b) not in allowed) or _starts_with(
            b, "scripts"
        )

    return {
        "covariates_import_no_model_specs": {
            (a, b)
            for a, b in edges
            if _starts_with(a, "analysis.behavior.covariates")
            and _starts_with(b, "analysis.behavior.model_specs")
        },
        "data_loaders_import_no_analysis": {
            (a, b)
            for a, b in edges
            if _starts_with(a, "data_loaders") and _starts_with(b, "analysis")
        },
        "analysis_imports_no_visualization": {
            (a, b)
            for a, b in edges
            if _starts_with(a, "analysis") and _starts_with(b, "visualization")
        },
        "src_imports_no_scripts": {
            (a, b) for a, b in edges if _starts_with(b, "scripts")
        },
        "foundation_imports_only_foundation": {
            (a, b)
            for a, b in edges
            if package_of(a) in FOUNDATION and outside(b, FOUNDATION)
        },
        "events_imports_only_foundation": {
            (a, b)
            for a, b in edges
            if package_of(a) == "analysis.events"
            and outside(b, FOUNDATION | EVENTS_MAY_IMPORT)
        },
        "hippocampus_imports_no_behavior": {
            (a, b)
            for a, b in edges
            if _starts_with(a, "analysis.hippocampus")
            and _starts_with(b, "analysis.behavior")
        },
        "hippocampus_imports_no_pfc_or_inventory_comparison": {
            (a, b)
            for a, b in edges
            if _starts_with(a, "analysis.hippocampus")
            and _starts_with(b, "analysis.pfc", "analysis.inventory_comparison")
        },
        "no_package_cycles": _cycle_edges(package_graph(root)),
    }


RULES = sorted(rule_violations())


def test_the_graph_is_not_vacuous() -> None:
    """The scan sees the packages the rules are about, and their known edges."""
    graph = package_graph()
    for node in (
        "analysis.events",
        "analysis.hippocampus",
        "analysis.behavior",
        "analysis.pfc",
        "data_loaders",
    ):
        assert node in graph, f"{node} missing from the package graph"
    assert "analysis.behavior.covariates" in module_index(SRC)
    assert "analysis.hippocampus" in graph["analysis.behavior"]
    assert set(FOUNDATION) <= set(graph), sorted(set(FOUNDATION) - set(graph))


@pytest.mark.parametrize("rule", RULES)
def test_layering_rule(rule: str) -> None:
    violations = rule_violations()[rule]
    assert not violations, f"{rule}:\n" + "\n".join(map(str, sorted(violations)))


def test_relative_import_in_a_package_init_is_not_missed(tmp_path: Path) -> None:
    """A ``covariates/__init__.py`` must not escape the rule on its dot count.

    ``from ..model_specs import BIN_S`` inside ``covariates/__init__.py`` names
    the same module as the absolute spelling. Resolving it against the module's
    own dotted name instead of its package would yield
    ``analysis.model_specs`` -- a module that does not exist -- and the edge
    would silently pass.
    """
    root = tmp_path / "continuum_swr_replay"
    covariates = root / "analysis" / "behavior" / "covariates"
    covariates.mkdir(parents=True)
    (covariates / "__init__.py").write_text("from ..model_specs import BIN_S\n")
    (root / "analysis" / "behavior" / "model_specs.py").write_text("BIN_S = 0.02\n")

    assert rule_violations(root)["covariates_import_no_model_specs"] == {
        ("analysis.behavior.covariates", "analysis.behavior.model_specs")
    }


def test_graph_rules_see_function_local_imports_and_cycles(tmp_path: Path) -> None:
    """A cycle closed by a lazy, function-local import is still a cycle.

    The same synthetic tree checks that a relative import resolves to the
    module it names, that an import of ``scripts`` is seen, and that a
    foundation module
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
    (root / "driver.py").write_text("")
    (root / "data_types.py").write_text("")
    (root / "parameters.py").write_text(
        "from . import data_types\nfrom .driver import run\n"
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
    assert found["foundation_imports_only_foundation"] == {("parameters", "driver")}


def test_a_longer_cycle_is_reported_in_import_order(tmp_path: Path) -> None:
    """Each reported edge points from the importer to the module it imports."""
    root = tmp_path / "continuum_swr_replay"
    (root / "analysis").mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "analysis" / "__init__.py").write_text("")
    for name, imports in (("a", "b"), ("b", "c"), ("c", "a")):
        (root / "analysis" / f"{name}.py").write_text(f"from . import {imports}\n")

    assert rule_violations(root)["no_package_cycles"] == {
        ("analysis.a", "analysis.b"),
        ("analysis.b", "analysis.c"),
        ("analysis.c", "analysis.a"),
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
