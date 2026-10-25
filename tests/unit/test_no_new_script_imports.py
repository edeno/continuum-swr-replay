"""Scripts are not libraries: only ``_``-prefixed script modules are importable.

Shared code belongs in the package (``src/continuum_swr_replay``) or in a
``_``-prefixed module beside the scripts that share it, such as
``scripts/qa/_report.py``. This test fails on any ``from scripts.<path> import``
or ``import scripts.<path>`` in ``scripts/`` whose target module has no
``_``-prefixed component, unless the (importing file, target module) pair is in
:data:`ALLOWED`. Tests may import scripts. The package may not import
``scripts`` at all (``test_import_layering``'s ``src_imports_no_scripts``).

:data:`ALLOWED` is the current exception list and only shrinks: an entry whose
import is gone fails the test, so moving shared code out removes its entry.
``scripts/README.md`` states the rule and points here.
"""

from __future__ import annotations

import functools
from pathlib import Path

from tests.unit._imports import REPO, imported_modules, module_index

SCRIPTS = REPO / "scripts"

#: (importing file relative to the repo, target module) -> why it is allowed.
ALLOWED: dict[tuple[str, str], str] = {
    (
        "scripts/exploratory/_content_cohort_control.py",
        "scripts.content.fit_swr_contrast_content",
    ): "an unsupported one-off that drives the runner's main with the cohort "
    "rule bypassed",
}


def _script_imports(path: Path, root: Path = SCRIPTS) -> set[str]:
    """Every ``scripts.*`` module ``path`` (a file under ``root``) imports."""
    return {m for m in imported_modules(path, root) if m.split(".")[0] == "scripts"}


def _is_importable(target: str) -> bool:
    """A script module may be imported only through a ``_``-prefixed component."""
    return any(part.startswith("_") for part in target.split(".")[1:])


@functools.cache
def _forbidden_imports(root: Path = SCRIPTS) -> frozenset[tuple[str, str]]:
    return frozenset(
        (str(path.relative_to(root.parent)), target)
        for path in module_index(root).values()
        for target in _script_imports(path, root)
        if not _is_importable(target)
    )


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
    assert not _is_importable("scripts.detection.report_inventory_comparison")

    root = tmp_path / "scripts"
    for module in (
        "qa/_report.py",
        "detection/report_inventory_comparison.py",
        "detection/run_event_features.py",
        "content/fit_replay_content_models.py",
    ):
        (root / module).parent.mkdir(parents=True, exist_ok=True)
        (root / module).write_text("")
    probe = root / "qa" / "probe.py"
    probe.write_text(
        "from scripts.qa import _report\n"
        "from scripts.detection import run_event_features\n"
        "from scripts.detection.report_inventory_comparison import _private\n"
        "def run():\n"
        "    import scripts.content.fit_replay_content_models as fitter\n"
    )
    assert _script_imports(probe, root) == {
        "scripts.qa._report",
        "scripts.detection.run_event_features",
        "scripts.detection.report_inventory_comparison",
        "scripts.content.fit_replay_content_models",
    }
    assert _forbidden_imports(root) == {
        ("scripts/qa/probe.py", "scripts.detection.run_event_features"),
        ("scripts/qa/probe.py", "scripts.detection.report_inventory_comparison"),
        ("scripts/qa/probe.py", "scripts.content.fit_replay_content_models"),
    }
