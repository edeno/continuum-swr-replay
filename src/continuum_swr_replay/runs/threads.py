"""Pin the BLAS and OpenMP thread pools, and a process to a set of CPUs.

numpy reads these variables once, when it is first imported, so a runner calls
:func:`pin_blas_threads` above its numpy import. This module imports only the
standard library, so importing it does not load numpy.

JAX's CPU thread pool has no size setting: its threads are made when JAX starts
its backend (its first computation), and their number follows the process's CPU
set. ``--xla_cpu_multi_thread_eigen=false`` in ``XLA_FLAGS`` does not change it
(JAX 0.8.0, measured). So a pool of processes that each compute with JAX pins
each process to its own CPUs (:func:`pin_to_cpus`) before that first
computation, and starts its processes by ``spawn`` (:data:`POOL_START_METHOD`):
a process forked from one in which JAX's threads are running can deadlock.
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.context
import os
import sys
from collections.abc import Collection

BLAS_THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
"""The OpenMP, MKL and OpenBLAS thread-count variables."""


def pin_blas_threads(*extra_variables: str) -> None:
    """Set each of ``BLAS_THREAD_VARIABLES`` (and ``extra_variables``) to 1.

    A variable already set is left as it is, so a caller can still choose a
    count from the shell. Call it before numpy is imported: with one session per
    process, multi-threaded BLAS would oversubscribe the cores.

    Parameters
    ----------
    *extra_variables : str
        Further thread-count variables to set, e.g. ``"NUMEXPR_NUM_THREADS"``.
    """
    for name in (*BLAS_THREAD_VARIABLES, *extra_variables):
        os.environ.setdefault(name, "1")


POOL_START_METHOD = "spawn"
"""How a pool that may compute with JAX starts its processes. ``fork`` copies a
process whose JAX threads may hold locks; ``spawn`` starts a fresh interpreter,
which imports what it needs."""


def pool_context() -> multiprocessing.context.BaseContext:
    """The ``multiprocessing`` context of a pool that may compute with JAX.

    Returns
    -------
    multiprocessing.context.BaseContext
        The :data:`POOL_START_METHOD` context, for ``mp_context=`` of a
        ``ProcessPoolExecutor`` or ``.Pool`` of ``multiprocessing``.
    """
    return multiprocessing.get_context(POOL_START_METHOD)


def pin_to_cpus(cpus: Collection[int]) -> None:
    """Run this process on `cpus` only, with one BLAS thread per CPU of the set.

    Every thread of the process is moved to `cpus` (Linux), and a thread made
    later inherits the set. JAX sizes its CPU thread pool from the set when it
    starts its backend, so call this before JAX's first computation; a pool's
    worker calls it in its initializer. Each of ``BLAS_THREAD_VARIABLES`` is set
    to the number of CPUs, for libraries loaded later and the processes this one
    starts.

    Parameters
    ----------
    cpus : collection of int
        The CPUs, as ``os.sched_getaffinity`` names them.

    Raises
    ------
    ValueError
        When `cpus` is empty.
    RuntimeError
        When numpy is already imported and one of the variables says another
        number of threads: numpy's BLAS read it at import and keeps it, so the
        process would run more (or fewer) BLAS threads than it has CPUs. Pin
        the parent with :func:`pin_blas_threads` and give each worker as many
        CPUs as that.
    """
    cpus = sorted({int(cpu) for cpu in cpus})
    if not cpus:
        raise ValueError("no CPUs to pin to")
    count = str(len(cpus))
    if "numpy" in sys.modules:
        read = {name: os.environ.get(name) for name in BLAS_THREAD_VARIABLES}
        if any(value != count for value in read.values()):
            raise RuntimeError(
                f"numpy was imported with {read}, which its BLAS keeps, so "
                f"pinning to {len(cpus)} CPU(s) would not match. Set the "
                f"variables to {count} before numpy is imported "
                "(runs.threads.pin_blas_threads) or pin to that many CPUs."
            )
    for name in BLAS_THREAD_VARIABLES:
        os.environ[name] = count
    for thread in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(thread), cpus)
        except ProcessLookupError:  # the thread ended after it was listed
            continue
