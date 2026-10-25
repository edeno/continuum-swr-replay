"""Regression tests for the lazy-import contract.

The package's two top-level ``__init__.py`` files re-export loader
functions lazily via PEP 562 ``__getattr__``, and the loaders import
``spyglass`` (which opens a DataJoint connection at import time) only
inside the functions that read through it, so that pure-analysis leaf
imports (e.g. ``continuum_swr_replay.analysis.replay.fitting``) finish
without touching Spyglass — keeping the offline test slice fast.

A regression here would re-introduce the hang that
``timeout 8 uv run python -c "from continuum_swr_replay.analysis.replay.fitting import fit_conditional_poisson"``
used to hit on no-VPN machines.

Each heavy-import check runs all of its imports in a single subprocess
(rather than one per import) because subprocess startup dominates the
actual work — ~13 s across 13 separate tests vs ~2 s for one combined run.
Sequential ordering matters: once a forbidden module lands in
``sys.modules`` it stays, so the first leak surfaces immediately
with the offending import statement.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.unit._imports import (
    FOUNDATION,
    SRC,
    defined_names,
    lazy_exports,
    module_edges,
    module_index,
    package_of,
    parse,
    relative_name,
)

_LEAK_CHECK = textwrap.dedent(
    """
    import sys
    for statements, forbidden in {steps!r}:
        for stmt in statements:
            exec(stmt, {{}})
            leaked = sorted(
                m for m in sys.modules
                if any(m == f or m.startswith(f + ".") for f in forbidden)
            )
            if leaked:
                print(f"LEAKED after {{stmt!r}}: {{','.join(leaked[:10])}}")
                sys.exit(1)
    print("OK")
    """
)


def _assert_imports_avoid(*steps: tuple[list[str], tuple[str, ...]]) -> None:
    """Run each ``(import statements, forbidden modules)`` step, in order, in
    one fresh interpreter; fail on the first statement after which a forbidden
    module (or a submodule of one) is in ``sys.modules``."""
    result = subprocess.run(
        [sys.executable, "-c", _LEAK_CHECK.format(steps=list(steps))],
        capture_output=True,
        text=True,
        # Guards against a hang, not against slowness. The import chain cost
        # ~46 s on this machine when the replay package __init__ was eager,
        # dominated by non_local_detector (~22 s) and statsmodels.api (~8 s),
        # measured with `python -X importtime`; the fitting and pfc.session leaves
        # still pull both in. The
        # old 30 s budget failed whenever the filesystem cache was cold, which
        # reads as a broken lazy-import contract when the contract is fine
        # (the subprocess exits 0 and prints OK).
        timeout=180,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "OK" in result.stdout


# Every import statement here must complete in a fresh interpreter
# without dragging ``spyglass.*`` into ``sys.modules``. Comments after
# each block name the surface the lazy contract protects — update
# them if you move or rename a function.
_LAZY_IMPORTS: list[str] = [
    # Package roots.
    "import continuum_swr_replay",
    "from continuum_swr_replay.data_types import DataLoadingPreset",
    # ``data_loaders/*`` pure helpers. The Spyglass-touching loaders
    # in the same modules (``get_hpc_marks``, ``get_theta``,
    # ``get_position_info``, ``load_dios``, …) must keep their
    # spyglass imports function-local so the helpers below stay
    # offline-importable.
    "from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH",
    "from continuum_swr_replay.data_loaders import NWB_FILES",
    # The loaders import Spyglass only inside the functions that read through it.
    "from continuum_swr_replay.data_loaders import load_data",
    "from continuum_swr_replay.data_loaders.spikes import detect_coincident_spikes, filter_spike_times",
    "from continuum_swr_replay.data_loaders.lfp import get_multiunit, get_multiunit_hse, filter_ripple_lfp_time",
    "from continuum_swr_replay.data_loaders.events import convert_dio_events_to_start_stop_times",
    "from continuum_swr_replay.data_loaders.position import get_interpolated_position_info",
    # The one epoch-identifier parser, read by offline decode-file readers.
    "from continuum_swr_replay.decoding.identifiers import parse_epoch_identifier",
    # The one decode-file naming rule, read by the same offline readers.
    "from continuum_swr_replay.decoding.files import decode_paths",
    # ``analysis/*`` leaf modules. A hippocampus leaf loads the package
    # ``__init__``, which imports every exported submodule eagerly (none
    # loads spyglass); the PFC leaves used to be poisoned by the eager
    # re-export from ``analysis/pfc/__init__.py``.
    "from continuum_swr_replay.analysis.replay.fitting import fit_conditional_poisson",
    "from continuum_swr_replay.analysis.hippocampus.event_analysis import get_event_start_end_times",
    "from continuum_swr_replay.analysis.events.nonlocal_detection import get_confident_times",
    "from continuum_swr_replay.analysis.pfc.load_population import _extract_animal",
    "from continuum_swr_replay.analysis.pfc.session import process_neuron",
    # The foundation interval algebra and spike binning.
    "from continuum_swr_replay.intervals import get_spikecount_per_time_bin",
]


def test_no_lazy_import_loads_spyglass():
    """No contracted lazy import puts ``spyglass.*`` into ``sys.modules``."""
    _assert_imports_avoid((_LAZY_IMPORTS, ("spyglass",)))


# Readers of a decode folder that do not load the decode runners' scheduler.
# Each one was free of ``dask.distributed`` and ``dask_cuda`` at 736e18b, but
# ``state_evidence.terms``, whose package imported the old decode driver
# module, which imported dask until the run infrastructure moved to ``runs`` (whose
# ``batch`` imports dask only inside the functions that start a cluster). They
# read the decode files' names from the leaf ``decoding.files``, and open
# ``results.nc`` through ``decoding.results``.
_NO_SCHEDULER_IMPORTS: list[str] = [
    "import continuum_swr_replay.decoding.files",
    "import continuum_swr_replay.decoding.results",
    "import continuum_swr_replay.analysis.pfc.canonical_events",
    "import continuum_swr_replay.analysis.hippocampus.result_loading",
    "import continuum_swr_replay.analysis.hippocampus.data_aggregation",
    "import continuum_swr_replay.analysis.hippocampus.state_evidence.terms",
    "import continuum_swr_replay.analysis.contfrag.data_aggregation",
    "import continuum_swr_replay.analysis.replay.prepare_data",
]


def test_decode_file_readers_do_not_load_the_dask_scheduler():
    """No decode-file reader puts ``dask.distributed`` or ``dask_cuda`` into
    ``sys.modules``."""
    _assert_imports_avoid((_NO_SCHEDULER_IMPORTS, ("dask.distributed", "dask_cuda")))


#: Every foundation module, relative to the package root.
_FOUNDATION_MODULES = sorted(
    m for m in module_index(SRC) if package_of(m) in FOUNDATION
)

#: Foundation modules the JAX-free check does not import, and why.
NOT_JAX_FREE = {
    "decoding.transitions": "imports non_local_detector",
}

# Leaf modules other packages import for a row key, a cohort rule, a column
# registry or a model ladder, and every foundation module but those in
# ``NOT_JAX_FREE``. Each one is free of JAX, ``non_local_detector`` and
# ``statsmodels`` now that the replay and PFC package ``__init__``s are lazy;
# before, every analysis leaf cost ~4.5 s and loaded all three (the
# code-organization review, F1). The two package roots are listed first: a
# re-export made eager again would load the heavy chain for every leaf.
JAX_FREE: list[str] = [
    "continuum_swr_replay.analysis.replay",
    "continuum_swr_replay.analysis.pfc",
    "continuum_swr_replay.analysis.events.cohort_selection",
    "continuum_swr_replay.analysis.replay.design_constants",
    "continuum_swr_replay.analysis.pfc.cohort",
    "continuum_swr_replay.analysis.replay.model_specs",
    *(
        f"continuum_swr_replay.{m}"
        for m in _FOUNDATION_MODULES
        if m not in NOT_JAX_FREE
    ),
]


def test_not_jax_free_names_foundation_modules():
    assert set(NOT_JAX_FREE) <= set(_FOUNDATION_MODULES)
    assert "continuum_swr_replay.tables.io" in JAX_FREE


def test_leaf_modules_do_not_load_jax():
    """No ``JAX_FREE`` module puts JAX, ``non_local_detector`` or
    ``statsmodels`` into ``sys.modules``."""
    _assert_imports_avoid(
        (
            [f"import {m}" for m in JAX_FREE],
            ("jax", "non_local_detector", "statsmodels"),
        )
    )


def test_intervals_and_runs_load_no_dask():
    """The foundation interval module loads no dask (phase-A review M12), no
    JAX chain, no Spyglass and no other module of this package. Then the run
    infrastructure loads no dask at import either (``batch`` imports it inside
    the functions that start a cluster), no JAX chain and no Spyglass.

    One interpreter checks both; ``intervals`` goes first, so the package
    modules ``runs`` loads cannot hide behind it.
    """
    heavy = ("dask", "distributed", "dask_cuda", "jax", "non_local_detector")
    heavy += ("statsmodels", "spyglass")
    other_packages = tuple(
        f"continuum_swr_replay.{top}"
        for top in sorted(
            {m.split(".")[0] for m in module_index(SRC)} - {"", "intervals"}
        )
    )
    _assert_imports_avoid(
        (["import continuum_swr_replay.intervals"], heavy + other_packages),
        (
            [
                f"import continuum_swr_replay.{m}"
                for m in _FOUNDATION_MODULES
                if package_of(m) == "runs"
            ],
            heavy,
        ),
    )


@pytest.mark.parametrize(
    "package",
    ["continuum_swr_replay.analysis.replay", "continuum_swr_replay.analysis.pfc"],
)
def test_every_lazy_export_imports_by_name(package):
    """``from <package> import <name>`` works for every name in ``__all__``,
    and binds the object its submodule defines."""
    module = importlib.import_module(package)
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


#: Every package ``__init__`` that re-exports lazily.
_LAZY_INITS = sorted(p for p in SRC.rglob("__init__.py") if lazy_exports(p, SRC))


def _unresolved_exports(init: Path, root: Path) -> list[tuple[str, str, str]]:
    """``(name, module, attribute)`` for each ``_LAZY_EXPORTS`` entry of ``init``
    whose module is not one under ``root`` or does not bind the attribute."""
    index = module_index(root)
    unresolved = []
    for name, (module, attribute) in lazy_exports(init, root).items():
        path = index.get(relative_name(module, root))
        if path is None or attribute not in defined_names(parse(path).body):
            unresolved.append((name, module, attribute))
    return unresolved


def test_the_lazy_inits_are_found():
    assert SRC / "analysis" / "replay" / "__init__.py" in _LAZY_INITS
    assert SRC / "data_loaders" / "__init__.py" in _LAZY_INITS


@pytest.mark.parametrize("init", _LAZY_INITS, ids=lambda p: str(p.relative_to(SRC)))
def test_every_lazy_export_names_a_defined_attribute(init):
    """Each ``_LAZY_EXPORTS`` target binds its attribute, checked statically.

    Nothing is imported, so the exports whose modules need Spyglass are
    covered offline too.
    """
    assert not _unresolved_exports(init, SRC)


def test_a_lazy_export_is_an_import_edge_and_a_bad_one_is_caught(tmp_path):
    """The layering scan sees each ``_LAZY_EXPORTS`` target as an import, with
    no ``TYPE_CHECKING`` mirror, and a target that binds nothing is reported."""
    root = tmp_path / "continuum_swr_replay"
    package = root / "pkg"
    package.mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "other.py").write_text("h = 1\n")
    (package / "mod.py").write_text("def f():\n    pass\n\n\ndef _g():\n    pass\n")
    (package / "__init__.py").write_text(
        '_LAZY_EXPORTS = {"f": "mod", "g": ("mod", "_g"), '
        '"h": "continuum_swr_replay.other", "k": "mod", "m": "gone"}\n'
    )

    assert module_edges(root) == {("pkg", "pkg.mod"), ("pkg", "other")}
    assert _unresolved_exports(package / "__init__.py", root) == [
        ("k", "continuum_swr_replay.pkg.mod", "k"),
        ("m", "continuum_swr_replay.pkg.gone", "m"),
    ]
