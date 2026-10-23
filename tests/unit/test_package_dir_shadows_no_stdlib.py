"""No package directory shadows a standard-library module.

A tool run with its working directory inside a package directory puts that
directory first on ``sys.path``. A module there named like a standard-library
one (the old ``types.py``) is then imported in its place: ``python -c "import
ast"`` failed with a partially-initialized-module ``ImportError``. The module is
``data_types.py`` now.

Every package directory under ``src/continuum_swr_replay/`` is checked. The
names in :data:`ALLOWLIST` are kept on purpose; a new shadowing module fails.
"""

import subprocess
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[2] / "src" / "continuum_swr_replay"

ALLOWLIST: dict[str, str] = {
    "io": (
        "preloaded at interpreter startup, so runs/io.py, tables/io.py and "
        "state_evidence/io.py cannot shadow it"
    ),
}


def _package_dirs() -> list[Path]:
    return sorted(p.parent for p in PACKAGE_DIR.rglob("__init__.py"))


def test_import_ast_succeeds_from_inside_the_package_directory():
    # ``-S`` skips ``site``, which in this venv already imports the real
    # ``types`` at startup and so would hide the shadowing.
    result = subprocess.run(
        [sys.executable, "-S", "-c", "import ast"],
        cwd=PACKAGE_DIR,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_every_package_directory_is_walked():
    dirs = _package_dirs()
    assert PACKAGE_DIR in dirs
    assert PACKAGE_DIR / "analysis" / "hippocampus" / "state_evidence" in dirs


def test_no_module_in_any_package_directory_has_a_stdlib_name():
    shadowing = {
        (str(directory.relative_to(PACKAGE_DIR)), name)
        for directory in _package_dirs()
        for name in {p.stem for p in directory.glob("*.py")}
        | {p.name for p in directory.iterdir() if (p / "__init__.py").is_file()}
        if name in sys.stdlib_module_names and name not in ALLOWLIST
    }

    assert shadowing == set()


def test_every_allowlisted_name_still_occurs():
    names = {p.stem for d in _package_dirs() for p in d.glob("*.py")}

    assert set(ALLOWLIST) <= names
