"""``runs.threads``: BLAS threads set before numpy, never overriding; a process
pinned to a CPU set; pools started by spawn."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys

import pytest

from continuum_swr_replay.runs.threads import (
    BLAS_THREAD_VARIABLES,
    pin_blas_threads,
    pin_to_cpus,
    pool_context,
)
from tests.support._imports import REPO

#: The runners that pin BLAS threads.
ENTRY_POINTS = (
    "scripts/replay_models/rate/fit_per_bin_rate_models.py",
    "scripts/replay_models/rate/fit_swr_contrast_rate.py",
    "scripts/replay_models/rate/fit_uncertainty_rate.py",
    "scripts/detector_comparison/run_event_features.py",
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


linux_only = pytest.mark.skipif(
    not hasattr(os, "sched_setaffinity"), reason="CPU affinity is Linux-only"
)


def _jitted_in_a_pinned_worker() -> dict:
    """Run in a spawned worker: a jitted float64 function, then where every
    thread of the process may run."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    with jax.enable_x64(True):
        value = float(jax.jit(lambda x: jnp.sum(x - 1.0))(np.full(10, 1.0 + 1e-12)))
    threads = {
        frozenset(os.sched_getaffinity(int(t))) for t in os.listdir("/proc/self/task")
    }
    return {
        "value": value,
        "threads": threads,
        "x64_after": jax.config.x64_enabled,
        "blas": {name: os.environ[name] for name in BLAS_THREAD_VARIABLES},
    }


@linux_only
def test_a_spawned_pinned_worker_runs_a_jitted_function():
    from concurrent.futures import ProcessPoolExecutor

    cpu = min(os.sched_getaffinity(0))
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=pool_context(),
        initializer=pin_to_cpus,
        initargs=([cpu],),
    ) as pool:
        result = pool.submit(_jitted_in_a_pinned_worker).result(timeout=300)

    assert result["value"] == pytest.approx(10 * ((1.0 + 1e-12) - 1.0), rel=1e-12)
    assert result["threads"] == {frozenset({cpu})}
    assert result["x64_after"] is False
    assert result["blas"] == dict.fromkeys(BLAS_THREAD_VARIABLES, "1")


def test_pools_start_by_spawn():
    assert pool_context().get_start_method() == "spawn"


@linux_only
def test_pinning_refuses_a_thread_count_numpy_already_read(monkeypatch):
    for name in BLAS_THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    before = os.sched_getaffinity(0)
    cpus = sorted(before)[:2]
    if len(cpus) < 2:
        pytest.skip("needs two CPUs")

    with pytest.raises(RuntimeError, match="numpy was imported"):
        pin_to_cpus(cpus)
    assert os.sched_getaffinity(0) == before
    with pytest.raises(ValueError, match="no CPUs"):
        pin_to_cpus([])


_PIN_OUTSIDE_THE_ALLOWED_SET = """
import json, os, sys
from continuum_swr_replay.runs.threads import BLAS_THREAD_VARIABLES, pin_to_cpus

for name in BLAS_THREAD_VARIABLES:
    os.environ[name] = "7"
before = os.sched_getaffinity(0)
cpus = [min(before), max(before) + 1]
message = None
try:
    pin_to_cpus(cpus)
except ValueError as error:
    message = str(error)
threads = [sorted(os.sched_getaffinity(int(t))) for t in os.listdir("/proc/self/task")]
print(json.dumps({
    "numpy": "numpy" in sys.modules,
    "message": message,
    "blas": [os.environ[name] for name in BLAS_THREAD_VARIABLES],
    "unchanged": all(set(t) == before for t in threads),
}))
"""


@linux_only
def test_pinning_refuses_cpus_outside_the_allowed_set_before_changing_anything():
    # In a fresh process without numpy, so the thread-count check passes and
    # only the CPU check can refuse.
    result = subprocess.run(
        [sys.executable, "-c", _PIN_OUTSIDE_THE_ALLOWED_SET],
        capture_output=True,
        text=True,
        check=True,
    )
    out = json.loads(result.stdout)
    assert out["numpy"] is False
    assert "not in this process's allowed CPUs" in out["message"]
    assert out["blas"] == ["7"] * len(BLAS_THREAD_VARIABLES)
    assert out["unchanged"] is True
