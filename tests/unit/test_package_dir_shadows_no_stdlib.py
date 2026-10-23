"""The package directory shadows no standard-library module.

A tool run with its working directory inside ``src/continuum_swr_replay/`` puts
that directory first on ``sys.path``. A module there named like a
standard-library one (the old ``types.py``) is then imported in its place:
``python -c "import ast"`` failed with a partially-initialized-module
``ImportError``. The module is ``schemas.py`` now.
"""

import subprocess
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[2] / "src" / "continuum_swr_replay"


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


def test_no_module_in_the_package_directory_has_a_stdlib_name():
    names = {p.stem for p in PACKAGE_DIR.glob("*.py")} | {
        p.name for p in PACKAGE_DIR.iterdir() if (p / "__init__.py").is_file()
    }

    assert names & sys.stdlib_module_names == set()
