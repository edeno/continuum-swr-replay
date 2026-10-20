"""Regression tests for the lazy-import contract.

The package's two top-level ``__init__.py`` files re-export functions
from submodules that pull in ``spyglass.common`` at module top level
(opening a DataJoint connection at import time). Those re-exports are
lazy via PEP 562 ``__getattr__`` so that pure-analysis leaf imports
(e.g. ``continuum_swr_replay.analysis.replay.fitting``) finish without
touching Spyglass — keeping the offline test slice fast.

A regression here would re-introduce the hang that
``timeout 8 uv run python -c "from continuum_swr_replay.analysis.replay.fitting import fit_conditional_poisson"``
used to hit on no-VPN machines.

All listed imports run in a single subprocess (rather than one per
import) because subprocess startup dominates the actual work —
~13 s across 13 separate tests vs ~2 s for one combined run.
Sequential ordering matters: once ``spyglass.*`` lands in
``sys.modules`` it stays, so the first leak surfaces immediately
with the offending import statement.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Every import statement here must complete in a fresh interpreter
# without dragging ``spyglass.*`` into ``sys.modules``. Comments after
# each block name the surface the lazy contract protects — update
# them if you move or rename a function.
_LAZY_IMPORTS: list[str] = [
    # Package roots.
    "import continuum_swr_replay",
    "from continuum_swr_replay.types import DataLoadingPreset",
    # ``data_loaders/*`` pure helpers. The Spyglass-touching loaders
    # in the same modules (``get_hpc_marks``, ``get_theta``,
    # ``get_position_info``, ``load_dios``, …) must keep their
    # spyglass imports function-local so the helpers below stay
    # offline-importable.
    "from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH",
    "from continuum_swr_replay.data_loaders import NWB_FILES",
    "from continuum_swr_replay.data_loaders.spikes import detect_coincident_spikes, filter_spike_times",
    "from continuum_swr_replay.data_loaders.lfp import get_multiunit, get_multiunit_hse, filter_ripple_lfp_time",
    "from continuum_swr_replay.data_loaders.events import convert_dio_events_to_start_stop_times",
    "from continuum_swr_replay.data_loaders.position import get_interpolated_position_info",
    # The one epoch-identifier parser, read by offline decode-file readers.
    "from continuum_swr_replay.data_loaders.utils import parse_epoch_identifier",
    # The one decode-file naming rule, read by the same offline readers.
    "from continuum_swr_replay.decode_files import decode_paths",
    # ``analysis/*`` leaf modules. The hippocampus leaves used to drag
    # ``data_aggregation`` (which imports ``get_labels`` → spyglass);
    # the PFC leaves used to be poisoned by the eager re-export from
    # ``analysis/pfc/__init__.py``.
    "from continuum_swr_replay.analysis.replay.fitting import fit_conditional_poisson",
    "from continuum_swr_replay.analysis.hippocampus.event_analysis import get_event_start_end_times",
    "from continuum_swr_replay.analysis.hippocampus.event_detection import get_confident_times",
    "from continuum_swr_replay.analysis.pfc.load_population import _extract_animal",
    "from continuum_swr_replay.analysis.pfc.glm import process_neuron",
    # The foundation interval algebra and spike binning.
    "from continuum_swr_replay.intervals import get_spikecount_per_time_bin",
]


def test_no_lazy_import_loads_spyglass():
    """Run every contracted lazy import in one fresh interpreter; fail
    on the first one that puts ``spyglass.*`` into ``sys.modules``."""
    code = textwrap.dedent(
        """
        import sys
        imports = {imports!r}
        for stmt in imports:
            exec(stmt, {{}})
            leaked = sorted(
                m for m in sys.modules
                if m == "spyglass" or m.startswith("spyglass.")
            )
            if leaked:
                print(f"LEAKED after {{stmt!r}}: {{','.join(leaked)}}")
                sys.exit(1)
        print("OK")
        """
    ).format(imports=_LAZY_IMPORTS)

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        # Guards against a hang, not against slowness. The import chain cost
        # ~46 s on this machine when the replay package __init__ was eager,
        # dominated by non_local_detector (~22 s) and statsmodels.api (~8 s),
        # measured with `python -X importtime`; the fitting and glm leaves
        # still pull both in. The
        # old 30 s budget failed whenever the filesystem cache was cold, which
        # reads as a broken lazy-import contract when the contract is fine
        # (the subprocess exits 0 and prints OK).
        timeout=180,
    )
    assert result.returncode == 0, (
        f"lazy-import contract broken.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout


# Readers of a decode folder that do not load the decode runners' scheduler.
# Each one was free of ``dask.distributed`` and ``dask_cuda`` at 736e18b
# (``state_evidence.terms`` was not: its package imports ``decoding_common``,
# so it is not listed). They read the decode files' names from the leaf
# ``decode_files``, never from ``decoding_common``.
_NO_SCHEDULER_IMPORTS: list[str] = [
    "import continuum_swr_replay.decode_files",
    "import continuum_swr_replay.analysis.pfc.canonical_events",
    "import continuum_swr_replay.analysis.hippocampus.result_loading",
    "import continuum_swr_replay.analysis.hippocampus.data_aggregation",
    "import continuum_swr_replay.analysis.contfrag.data_aggregation",
    "import continuum_swr_replay.analysis.replay.prepare_data",
]


def test_decode_file_readers_do_not_load_the_dask_scheduler():
    """Run each reader import in one fresh interpreter; fail on the first one
    that puts ``dask.distributed`` or ``dask_cuda`` into ``sys.modules``."""
    code = textwrap.dedent(
        """
        import sys
        imports = {imports!r}
        for stmt in imports:
            exec(stmt, {{}})
            leaked = sorted(
                m for m in sys.modules
                if m in ("dask.distributed", "dask_cuda")
                or m.startswith("dask_cuda.")
            )
            if leaked:
                print(f"LEAKED after {{stmt!r}}: {{','.join(leaked)}}")
                sys.exit(1)
        print("OK")
        """
    ).format(imports=_NO_SCHEDULER_IMPORTS)

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,  # a hang guard, as above
    )
    assert result.returncode == 0, (
        f"a decode-file reader loads the dask scheduler.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout


# Leaf modules other packages import for a row key, a cohort rule, a column
# registry or a model ladder. Each one is free of JAX, ``non_local_detector``
# and ``statsmodels`` now that the replay and PFC package ``__init__``s are
# lazy; before, every one of them cost ~4.5 s and loaded all three (the
# code-organization review, F1). The two package roots are listed first: a
# re-export made eager again would load the heavy chain for every leaf.
JAX_FREE: list[str] = [
    "continuum_swr_replay.analysis.replay",
    "continuum_swr_replay.analysis.pfc",
    "continuum_swr_replay.analysis.replay.event_ids",
    "continuum_swr_replay.analysis.replay.cohort_selection",
    "continuum_swr_replay.analysis.replay.covariate_registry",
    "continuum_swr_replay.analysis.replay.versions",
    "continuum_swr_replay.analysis.replay.standardization",
    "continuum_swr_replay.analysis.replay.design_constants",
    "continuum_swr_replay.table_parity",
    "continuum_swr_replay.analysis.pfc.cohort",
    "continuum_swr_replay.analysis.replay.model_specs",
    "continuum_swr_replay.intervals",
]

#: Leaf modules that should be JAX-free but are not yet, each with the import
#: chain that loads JAX. Empty: every leaf above is clean.
JAX_FREE_PENDING: dict[str, str] = {}

_HEAVY_MODULES = ("jax", "non_local_detector", "statsmodels")


def test_leaf_modules_do_not_load_jax():
    """Import every ``JAX_FREE`` module in one fresh interpreter; fail on the
    first one that puts JAX, ``non_local_detector`` or ``statsmodels`` into
    ``sys.modules``."""
    code = textwrap.dedent(
        """
        import importlib
        import sys
        heavy = {heavy!r}
        for module in {modules!r}:
            importlib.import_module(module)
            leaked = sorted(
                m for m in sys.modules
                if any(m == h or m.startswith(h + ".") for h in heavy)
            )
            if leaked:
                print(f"LEAKED after importing {{module}}: {{','.join(leaked[:10])}}")
                sys.exit(1)
        print("OK")
        """
    ).format(heavy=_HEAVY_MODULES, modules=JAX_FREE)

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,  # a hang guard, as above
    )
    assert result.returncode == 0, (
        f"a JAX-free leaf module loads a heavy dependency.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout


def test_intervals_imports_only_numpy_and_pandas():
    """The foundation interval module loads no dask (phase-A review M12), no
    JAX chain, and no other module of this package."""
    code = textwrap.dedent(
        """
        import sys
        import continuum_swr_replay.intervals
        heavy = ("dask", "jax", "non_local_detector", "statsmodels", "spyglass")
        leaked = sorted(
            m for m in sys.modules
            if any(m == h or m.startswith(h + ".") for h in heavy)
        )
        ours = sorted(
            m for m in sys.modules
            if m.startswith("continuum_swr_replay")
            and m not in ("continuum_swr_replay", "continuum_swr_replay.intervals")
        )
        print("LEAKED" if leaked or ours else "OK", leaked[:10], ours[:10])
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,  # a hang guard, as above
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("OK"), result.stdout


def test_jax_free_lists_are_disjoint():
    assert not set(JAX_FREE) & set(JAX_FREE_PENDING)


@pytest.mark.parametrize(
    "package",
    ["continuum_swr_replay.analysis.replay", "continuum_swr_replay.analysis.pfc"],
)
def test_every_lazy_export_imports_by_name(package):
    """``from <package> import <name>`` works for every name in ``__all__``,
    and binds the object its submodule defines."""
    module = importlib.import_module(package)
    assert set(module.__all__) == set(module._LAZY_EXPORTS)
    for name in module.__all__:
        namespace: dict[str, object] = {}
        exec(f"from {package} import {name}", namespace)  # noqa: S102
        target = module._LAZY_EXPORTS[name]
        submodule, attribute = target if isinstance(target, tuple) else (target, name)
        defining = importlib.import_module(f"{package}.{submodule}")
        assert namespace[name] is getattr(defining, attribute), name


def test_an_unknown_name_is_an_attribute_error():
    module = importlib.import_module("continuum_swr_replay.analysis.replay")
    with pytest.raises(AttributeError, match="no_such_name"):
        module.no_such_name  # noqa: B018


_SRC = Path(__file__).resolve().parents[2] / "src" / "continuum_swr_replay"

#: The package ``__init__``s that re-export lazily, relative to ``_SRC``.
_LAZY_INITS = (
    "__init__.py",
    "data_loaders/__init__.py",
    "analysis/hippocampus/__init__.py",
    "analysis/pfc/__init__.py",
    "analysis/replay/__init__.py",
)


def _lazy_exports_and_type_checking_imports(
    init: Path,
) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]]]:
    """``(module, attribute, bound name)`` triples, read statically from ``init``.

    The first set is what ``_LAZY_EXPORTS`` promises; the second is what the
    ``if TYPE_CHECKING:`` block imports. Modules are absolute dotted names.
    """
    package_parts = [_SRC.name, *init.parent.relative_to(_SRC).parts]
    package = ".".join(package_parts)
    tree = ast.parse(init.read_text(), filename=str(init))

    exports: set[tuple[str, str, str]] | None = None
    imports: set[tuple[str, str, str]] = set()
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_LAZY_EXPORTS"
                for t in node.targets
            )
        ) or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_LAZY_EXPORTS"
        ):
            exports = set()
            for name, target in ast.literal_eval(node.value).items():
                module, attribute = (
                    target if isinstance(target, tuple) else (target, name)
                )
                if not module.startswith(f"{_SRC.name}."):
                    module = f"{package}.{module}"
                exports.add((module, attribute, name))
        elif (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            for statement in ast.walk(node):
                if not isinstance(statement, ast.ImportFrom):
                    continue
                if statement.level == 0:
                    module = statement.module or ""
                else:
                    kept = package_parts[: len(package_parts) - (statement.level - 1)]
                    module = ".".join(
                        [*kept, *([statement.module] if statement.module else [])]
                    )
                for alias in statement.names:
                    imports.add((module, alias.name, alias.asname or alias.name))
    assert exports, f"{init}: no _LAZY_EXPORTS found; the check would be vacuous"
    return exports, imports


@pytest.mark.parametrize("init", _LAZY_INITS)
def test_every_lazy_export_has_a_type_checking_mirror(init):
    """Each ``_LAZY_EXPORTS`` target is imported under ``TYPE_CHECKING`` too.

    The import-layering scan (``test_import_layering.module_edges``) sees a
    lazy export's dependency only through that mirror: ``_LAZY_EXPORTS`` holds
    strings, which ``ast`` does not read as imports. A missing mirror hides
    the edge from every layering rule.
    """
    exports, imports = _lazy_exports_and_type_checking_imports(_SRC / init)
    missing = sorted(exports - imports)
    assert not missing, (
        f"{init}: lazy exports with no TYPE_CHECKING import "
        "(module, attribute, bound name):\n" + "\n".join(map(str, missing))
    )


def test_the_lazy_init_list_is_complete():
    """Every ``__init__`` that defines ``_LAZY_EXPORTS`` is in the mirror check."""
    found = {
        str(p.relative_to(_SRC))
        for p in _SRC.rglob("__init__.py")
        if "_LAZY_EXPORTS" in p.read_text()
    }
    assert found == set(_LAZY_INITS)


def test_a_missing_type_checking_mirror_is_caught(tmp_path, monkeypatch):
    """A lazy export whose ``TYPE_CHECKING`` import was dropped is reported."""
    package = tmp_path / "continuum_swr_replay" / "pkg"
    package.mkdir(parents=True)
    init = package / "__init__.py"
    init.write_text(
        '_LAZY_EXPORTS = {"f": "mod", "g": ("mod", "_g"), '
        '"h": "continuum_swr_replay.other"}\n'
        "if TYPE_CHECKING:\n"
        "    from .mod import f\n"
        "    from .mod import g\n"  # the alias is missing: binds _g as g
    )
    monkeypatch.setattr(
        sys.modules[__name__], "_SRC", tmp_path / "continuum_swr_replay"
    )
    exports, imports = _lazy_exports_and_type_checking_imports(init)
    assert sorted(exports - imports) == [
        ("continuum_swr_replay.other", "h", "h"),
        ("continuum_swr_replay.pkg.mod", "_g", "g"),
    ]
