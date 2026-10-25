"""Pin the BLAS and OpenMP thread pools before numpy is imported.

numpy reads these variables once, when it is first imported, so a runner calls
:func:`pin_blas_threads` above its numpy import. This module imports only the
standard library, so importing it does not load numpy.
"""

from __future__ import annotations

import os

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
