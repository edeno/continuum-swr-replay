"""No code asks for a ``fork`` pool, and no JAX code makes a pool that forks.

A process forked from one in which JAX's threads run can deadlock, so a pool
that may compute with JAX starts its processes by spawn
(``runs.threads.pool_context``). Two scans of ``src/`` and ``scripts/`` check
this.

The first finds an explicit request for fork: a call of ``get_context("fork")``
or ``set_start_method("fork")``.

The second finds a pool made with no start method, which forks on Linux: a
``Pool(...)`` or ``ProcessPoolExecutor(...)`` call with no ``mp_context=`` or
``context=`` keyword (nor that argument by position), and not called on
``get_context(...)`` or ``pool_context()``. A ``**options`` argument does not
count as a context. It reports such a call only in a module that imports JAX:
one that imports ``jax``, the regression fitter
(``continuum_swr_replay.analysis.regression``) or ``non_local_detector`` (which
imports ``jax``), directly or through the modules of this repository it
imports, with their parent packages. An import inside a function counts. The
present offenders are listed in :data:`ALLOWED_FORKING_POOLS` with the reason
each is allowed for now. The test fails when a new one appears and when a
listed one is gone, so the list cannot go stale.

A pool made on a context held in a variable (``ctx.Pool(...)``) is reported;
call ``pool_context().Pool(...)`` or pass ``mp_context=`` instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.support._imports import REPO, SRC, imported_modules, module_index

SCRIPTS = REPO / "scripts"

FORK_CALLS = frozenset({"get_context", "set_start_method"})

POOL_CALLS = {"Pool": 5, "ProcessPoolExecutor": 2}
"""Each pool constructor, with the position of its context argument (1-based):
``multiprocessing.Pool(processes, initializer, initargs, maxtasksperchild,
context)`` and ``ProcessPoolExecutor(max_workers, mp_context, ...)``."""

CONTEXT_KEYWORDS = frozenset({"context", "mp_context"})
CONTEXT_RECEIVERS = frozenset({"get_context", "pool_context"})

JAX_IMPORTS = ("jax", "continuum_swr_replay.analysis.regression", "non_local_detector")
"""A module that imports one of these (or a submodule) imports JAX."""

_PARENT_ONLY = (
    "The parent only reads manifests and tables before it starts the pool; "
    "JAX reaches it through imports, not a computation."
)
_RATE_RUNNER = (
    "A statsmodels rate fit; it reaches JAX only through the import of "
    "non_local_detector's time-rescaling check in rate/per_bin_rate_model.py, "
    "inside a function."
)

ALLOWED_FORKING_POOLS: dict[str, str] = {
    "src/continuum_swr_replay/analysis/replay_models/rate/per_bin_rate_ladder.py": (
        f"{_RATE_RUNNER} Removed by Phase 5a (legacy inventory 5a-1)."
    ),
    "src/continuum_swr_replay/analysis/replay_models/rate/swr_contrast_rate.py": (
        f"{_RATE_RUNNER} Removed by Phase 5b (legacy inventory 5b-2)."
    ),
    "scripts/replay_models/rate/fit_duration_aware_history_rate.py": (
        f"{_RATE_RUNNER} Removed by Phase 11 (legacy inventory 11-5)."
    ),
    "scripts/exploratory/_covariate_ranking.py": (
        f"{_RATE_RUNNER} An unsupported exploratory one-off."
    ),
    "scripts/event_evidence/_per_event_run.py": (
        "`run` passes `**pool_options`; both of its makers, `cpu_pool` and "
        "`gpu_pool`, give a spawn `mp_context`."
    ),
    "scripts/checks/check_stored_folds_reproduce.py": _PARENT_ONLY,
    "scripts/decoding/measure_decoder_quality.py": _PARENT_ONLY,
    "scripts/detector_comparison/add_state_shares.py": _PARENT_ONLY,
    "scripts/detector_comparison/measure_exposure_in_well_visits.py": _PARENT_ONLY,
    "scripts/event_properties/measure_brain_state_reference.py": _PARENT_ONLY,
    "scripts/event_properties/measure_content_along_arms.py": _PARENT_ONLY,
    "scripts/event_properties/measure_event_gamma.py": _PARENT_ONLY,
    "scripts/event_properties/measure_event_start_and_rest_theta.py": _PARENT_ONLY,
    "scripts/event_properties/measure_nonlocal_by_signal.py": _PARENT_ONLY,
    "scripts/event_properties/measure_ripple_class_structure.py": _PARENT_ONLY,
    "scripts/event_properties/sweep_event_rule.py": _PARENT_ONLY,
    "scripts/figures/generate_supp_figure02.py": _PARENT_ONLY,
}
"""The modules that import JAX and make a pool with no start method, allowed
for now, each with its reason."""


def _call_name(node: ast.Call) -> str | None:
    return getattr(node.func, "attr", getattr(node.func, "id", None))


def fork_calls(path: Path) -> list[int]:
    """Lines of `path` that call one of :data:`FORK_CALLS` with ``"fork"``."""
    lines = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
        if _call_name(node) in FORK_CALLS and any(
            isinstance(a, ast.Constant) and a.value == "fork" for a in arguments
        ):
            lines.append(node.lineno)
    return sorted(lines)


def default_pool_calls(path: Path) -> list[int]:
    """Lines of `path` that make a pool with no start method or context."""
    lines = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call) or _call_name(node) not in POOL_CALLS:
            continue
        receiver = getattr(node.func, "value", None)
        explicit = (
            any(keyword.arg in CONTEXT_KEYWORDS for keyword in node.keywords)
            or len(node.args) >= POOL_CALLS[_call_name(node)]
            or (
                isinstance(receiver, ast.Call)
                and _call_name(receiver) in CONTEXT_RECEIVERS
            )
        )
        if not explicit:
            lines.append(node.lineno)
    return sorted(lines)


def _module_graph() -> dict[str, tuple[Path, frozenset[str]]]:
    """Every module of ``src/`` and ``scripts/`` by absolute name, with its
    path and the modules it imports."""
    graph = {}
    for root in (SRC, SCRIPTS):
        for name, path in module_index(root).items():
            full = f"{root.name}.{name}" if name else root.name
            graph[full] = (path, imported_modules(path, root))
    return graph


def imports_jax(module: str, graph: dict[str, tuple[Path, frozenset[str]]]) -> bool:
    """Whether `module` imports one of :data:`JAX_IMPORTS`, directly or through
    the modules of `graph` it imports and their parent packages."""
    seen: set[str] = set()
    stack = [module]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        if any(name == j or name.startswith(f"{j}.") for j in JAX_IMPORTS):
            return True
        if name in graph:
            stack.extend(graph[name][1])
        if "." in name:
            stack.append(name.rsplit(".", 1)[0])
    return False


def forking_jax_pools() -> dict[str, list[int]]:
    """The modules that import JAX and make a pool with no start method, by
    path relative to the repository, with the lines of those calls."""
    graph = _module_graph()
    return {
        str(path.relative_to(REPO)): lines
        for module, (path, _) in graph.items()
        if (lines := default_pool_calls(path)) and imports_jax(module, graph)
    }


def test_no_code_requests_a_fork_pool():
    files = [*SRC.rglob("*.py"), *SCRIPTS.rglob("*.py")]
    assert len(files) > 100
    found = {
        str(path.relative_to(REPO)): lines
        for path in files
        if (lines := fork_calls(path))
    }
    assert found == {}


def test_the_scan_sees_a_fork_call(tmp_path):
    path = tmp_path / "pool.py"
    path.write_text(
        "import multiprocessing\n"
        "multiprocessing.get_context('spawn')\n"
        "multiprocessing.get_context('fork').Pool(2)\n"
        "multiprocessing.set_start_method(method='fork')\n"
    )
    assert fork_calls(path) == [3, 4]


def test_no_new_jax_module_makes_a_pool_that_forks():
    found = forking_jax_pools()
    new = sorted(set(found) - set(ALLOWED_FORKING_POOLS))
    gone = sorted(set(ALLOWED_FORKING_POOLS) - set(found))
    assert not new, (
        "a module that imports JAX makes a pool with no start method: "
        f"{ {path: found[path] for path in new} }; pass "
        "mp_context=runs.threads.pool_context()"
    )
    assert not gone, (
        f"no longer makes such a pool; remove from ALLOWED_FORKING_POOLS: {gone}"
    )


def test_the_scan_sees_a_pool_with_no_start_method(tmp_path):
    path = tmp_path / "pool.py"
    path.write_text(
        "import multiprocessing\n"
        "from concurrent.futures import ProcessPoolExecutor\n"
        "from continuum_swr_replay.runs.threads import pool_context\n"
        "multiprocessing.Pool(2)\n"
        "ProcessPoolExecutor(max_workers=2)\n"
        "ProcessPoolExecutor(2, **options)\n"
        "multiprocessing.get_context('spawn').Pool(2)\n"
        "pool_context().Pool(2)\n"
        "ProcessPoolExecutor(2, mp_context=pool_context())\n"
        "ProcessPoolExecutor(2, pool_context())\n"
        "multiprocessing.Pool(2, context=pool_context())\n"
        "ThreadPoolExecutor(2)\n"
    )
    assert default_pool_calls(path) == [4, 5, 6]


def test_the_import_scan_follows_imports_and_parents():
    graph = {
        "a": (Path("a"), frozenset()),
        "a.b": (Path("b"), frozenset({"c"})),
        "c": (Path("c"), frozenset({"jax.numpy"})),
        "d": (Path("d"), frozenset({"numpy"})),
        "e.f": (Path("f"), frozenset()),
        "e": (Path("e"), frozenset({"non_local_detector"})),
    }
    assert imports_jax("a.b", graph)
    assert imports_jax("e.f", graph)
    assert not imports_jax("a", graph)
    assert not imports_jax("d", graph)
