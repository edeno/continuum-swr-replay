"""Scripts are not libraries: only ``_``-prefixed script modules are importable.

Shared code belongs in the package (``src/continuum_swr_replay``) or in a
``_``-prefixed module beside the scripts that share it, such as
``scripts/qa/_report.py``. This test fails on any ``from scripts.<path> import``
or ``import scripts.<path>`` in ``scripts/`` or ``src/`` whose target module has
no ``_``-prefixed component, unless the (importing file, target module) pair is
in :data:`ALLOWED`. Tests may import scripts.

:data:`ALLOWED` is the current exception list and only shrinks: an entry whose
import is gone fails the test, so moving shared code out removes its entry.
``scripts/README.md`` states the rule and points here.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOTS = (REPO / "scripts", REPO / "src")

_INVENTORY = (
    "the detection-comparison family: the report, the feature runner and the "
    "example figures read the inventory runner's run layout (roots, output "
    "paths, table dtypes), detector roster and decode helpers"
)
_MEMORYLESS = (
    "the memoryless sweep family: the control, held-out and reference checks "
    "reuse the base sweep's simulated recording and its detector runs"
)

#: (importing file relative to the repo, target module) -> why it is allowed.
ALLOWED: dict[tuple[str, str], str] = {
    (
        "scripts/exploratory/_content_cohort_control.py",
        "scripts.content.fit_swr_contrast_content",
    ): "an unsupported one-off that drives the runner's main with the cohort "
    "rule bypassed",
    (
        "scripts/figures/generate_event_feature_examples.py",
        "scripts.figures.generate_inventory_examples",
    ): _INVENTORY,
    (
        "scripts/figures/generate_event_feature_examples.py",
        "scripts.qa.report_event_features",
    ): _INVENTORY,
    (
        "scripts/figures/generate_event_feature_examples.py",
        "scripts.qa.run_event_features",
    ): _INVENTORY,
    (
        "scripts/figures/generate_event_feature_examples.py",
        "scripts.qa.run_inventory_comparison",
    ): _INVENTORY,
    (
        "scripts/figures/generate_inventory_examples.py",
        "scripts.decoding.run_carey_detection",
    ): _INVENTORY,
    (
        "scripts/figures/generate_inventory_examples.py",
        "scripts.decoding.run_memoryless_detection",
    ): _INVENTORY,
    (
        "scripts/figures/generate_inventory_examples.py",
        "scripts.qa.run_inventory_comparison",
    ): _INVENTORY,
    (
        "scripts/qa/report_event_features.py",
        "scripts.figures.generate_inventory_examples",
    ): _INVENTORY,
    (
        "scripts/qa/report_event_features.py",
        "scripts.qa.run_event_features",
    ): _INVENTORY,
    (
        "scripts/qa/report_event_features.py",
        "scripts.qa.run_inventory_comparison",
    ): _INVENTORY,
    (
        "scripts/qa/report_inventory_comparison.py",
        "scripts.qa.run_inventory_comparison",
    ): _INVENTORY,
    (
        "scripts/qa/run_event_features.py",
        "scripts.decoding.run_carey_detection",
    ): _INVENTORY,
    (
        "scripts/qa/run_event_features.py",
        "scripts.qa.analyze_event_spike_evidence",
    ): _INVENTORY,
    (
        "scripts/qa/run_event_features.py",
        "scripts.qa.run_inventory_comparison",
    ): _INVENTORY,
    (
        "scripts/qa/compare_memoryless_to_reference.py",
        "scripts.figures.generate_memoryless_comparison",
    ): _MEMORYLESS,
    (
        "scripts/qa/compare_memoryless_to_reference.py",
        "scripts.qa.memoryless_reference",
    ): _MEMORYLESS,
    (
        "scripts/qa/compare_memoryless_to_reference.py",
        "scripts.qa.sweep_memoryless_detection",
    ): _MEMORYLESS,
    (
        "scripts/qa/sweep_memoryless_controls.py",
        "scripts.qa.sweep_memoryless_detection",
    ): _MEMORYLESS,
    (
        "scripts/qa/sweep_memoryless_heldout.py",
        "scripts.qa.sweep_memoryless_controls",
    ): _MEMORYLESS,
    (
        "scripts/qa/sweep_memoryless_heldout.py",
        "scripts.qa.sweep_memoryless_detection",
    ): _MEMORYLESS,
}


def _target(module: str, name: str | None) -> str:
    """The module an import binds: ``module``, or ``module.name`` when ``name``
    is itself a module (``from scripts.qa import run_event_features``)."""
    if name is None:
        return module
    package = REPO.joinpath(*module.split("."))
    if (package / f"{name}.py").is_file() or (package / name).is_dir():
        return f"{module}.{name}"
    return module


def _script_imports(path: Path) -> set[str]:
    """Every ``scripts.*`` module ``path`` imports, at any depth."""
    targets = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            targets |= {
                alias.name
                for alias in node.names
                if alias.name.split(".")[0] == "scripts"
            }
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
            and node.module.split(".")[0] == "scripts"
        ):
            targets |= {_target(node.module, alias.name) for alias in node.names}
    return targets


def _is_importable(target: str) -> bool:
    """A script module may be imported only through a ``_``-prefixed component."""
    return any(part.startswith("_") for part in target.split(".")[1:])


def _forbidden_imports() -> set[tuple[str, str]]:
    return {
        (str(path.relative_to(REPO)), target)
        for root in ROOTS
        for path in sorted(root.rglob("*.py"))
        for target in _script_imports(path)
        if not _is_importable(target)
    }


def test_no_script_imports_another_script_outside_the_allowlist():
    new = sorted(_forbidden_imports() - set(ALLOWED))

    assert new == [], (
        "These import a script that is not a `_`-prefixed module. Move the shared "
        "code into src/continuum_swr_replay or into a `_`-prefixed module beside "
        f"the scripts that share it (see scripts/README.md): {new}"
    )


def test_the_allowlist_only_names_imports_that_exist():
    """An allowlist entry whose import is gone must be removed, so the list shrinks."""
    stale = sorted(set(ALLOWED) - _forbidden_imports())

    assert stale == []


def test_the_rule_reads_the_module_not_the_imported_name(tmp_path):
    """``_report`` is importable; a private name of a public script is not."""
    assert _is_importable("scripts.qa._report")
    assert _is_importable("scripts.decoding._decode_runner")
    assert _is_importable("scripts.exploratory._statescript_contingency")
    assert not _is_importable("scripts.qa.report_inventory_comparison")
    assert (
        _target("scripts.qa", "run_event_features") == "scripts.qa.run_event_features"
    )
    assert (
        _target("scripts.qa.report_inventory_comparison", "_caption")
        == "scripts.qa.report_inventory_comparison"
    )

    probe = tmp_path / "probe.py"
    probe.write_text(
        "from scripts.qa import _report\n"
        "from scripts.qa.report_inventory_comparison import _private\n"
        "def run():\n"
        "    import scripts.content.fit_replay_content_models as fitter\n"
    )
    assert _script_imports(probe) == {
        "scripts.qa._report",
        "scripts.qa.report_inventory_comparison",
        "scripts.content.fit_replay_content_models",
    }
