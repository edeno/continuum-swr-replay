"""Regression test: ``_plot_calibration_curves`` must not crash on
empty calibration output.

When ``compute_calibration_curve`` legitimately returns an empty
frame (all-NaN ``Local_prob``, no decoder bins survive a filter,
etc.), the script used to compute ``n_animals = 0`` and call
``plt.subplots(1, 0, ...)``, raising
``ValueError: Number of columns must be a positive integer, not 0``.

The fix logs a "no calibration data" message and returns early.
The helper is loaded by ``ast`` so the test stays offline regardless
of the script's surrounding imports.
"""

from __future__ import annotations

import ast
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_script_helper(filename: str, attr_name: str, namespace: dict):
    """Compile + exec one named function from ``scripts/`` into namespace."""
    src = (SCRIPTS_DIR / filename).read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == attr_name:
            wrapper = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(wrapper)
            exec(compile(wrapper, filename, "exec"), namespace)  # noqa: S102
            return namespace[attr_name]
    raise AttributeError(f"{attr_name!r} not found in {filename}")


def test_plot_calibration_curves_handles_empty_calib_frame(tmp_path):
    # Stub ``compute_calibration_curve`` to return an empty frame
    # so the helper hits the new short-circuit instead of raising
    # ``Number of columns must be a positive integer, not 0``.
    fake_module = types.SimpleNamespace(
        compute_calibration_curve=lambda data: pd.DataFrame(),
    )
    namespace = {
        "compute_calibration_curve": fake_module.compute_calibration_curve,
        "plt": plt,
        "pd": pd,
    }
    fn = _load_script_helper(
        "run_quality_metrics.py", "_plot_calibration_curves", namespace
    )
    # Must not raise.
    fn(pd.DataFrame(), tmp_path)
