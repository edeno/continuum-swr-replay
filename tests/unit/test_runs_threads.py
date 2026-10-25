"""``runs.threads.pin_blas_threads``: set before numpy, never overriding."""

from __future__ import annotations

import ast
import os
import subprocess
import sys

import pytest

from continuum_swr_replay.runs.threads import BLAS_THREAD_VARIABLES, pin_blas_threads
from tests.unit._imports import REPO

#: The runners that pin BLAS threads.
ENTRY_POINTS = (
    "scripts/rate/fit_per_bin_rate_models.py",
    "scripts/rate/fit_swr_contrast_rate.py",
    "scripts/rate/fit_uncertainty_rate.py",
    "scripts/detection/run_event_features.py",
    "scripts/pfc/run_pfc_analysis_bandit.py",
)


def test_pin_sets_unset_variables_and_keeps_set_ones(monkeypatch):
    for name in (*BLAS_THREAD_VARIABLES, "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "8")

    pin_blas_threads("NUMEXPR_NUM_THREADS")

    assert {name: os.environ[name] for name in BLAS_THREAD_VARIABLES} == {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "1",
    }
    assert os.environ["NUMEXPR_NUM_THREADS"] == "1"


def test_importing_the_helper_does_not_import_numpy():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, continuum_swr_replay.runs.threads; "
            "print('numpy' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


def _is_pin_call(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "pin_blas_threads"
    )


@pytest.mark.parametrize("path", ENTRY_POINTS)
def test_entry_point_pins_before_any_third_party_import(path):
    body = ast.parse((REPO / path).read_text()).body
    pin = next(i for i, node in enumerate(body) if _is_pin_call(node))
    before = []
    for node in body[:pin]:
        if isinstance(node, ast.Import):
            before += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            before.append(node.module or "")
    allowed = set(sys.stdlib_module_names) | {"__future__"}
    assert [
        module
        for module in before
        if module.split(".")[0] not in allowed
        and module != "continuum_swr_replay.runs.threads"
    ] == []
