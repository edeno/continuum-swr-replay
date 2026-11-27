"""How the fitter's JAX code runs: float64 for each call, one device, float64
kernels, and a log of the programs traced.

- :func:`in_float64` goes on each public entry of the fitter that makes a JAX
  array or runs a kernel. It checks the device
  (:func:`continuum_swr_replay.runs.device.require_one_jax_device`) and
  turns JAX's 64-bit types on for that call only, in the calling thread. The
  decoder in the same process computes in float32, so the flag is never set for
  the whole process.
- :func:`float64_kernel` jits a kernel. When it is traced, it refuses any
  floating input that is not float64, and any call made outside
  :func:`in_float64`, and it records the trace in :data:`COMPILE_LOG`.
- :data:`COMPILE_LOG` counts traces by kernel and input signature. JAX traces a
  jitted function once for each new signature (shapes, dtypes, static
  arguments, the 64-bit flag) and compiles each trace; a program read from the
  persistent compilation cache is still traced. So the count is the number of
  programs this process needed, whatever the cache holds.
- :data:`LIVE_PROGRAM_LIMIT` bounds the programs kept compiled. Each compiled
  XLA CPU program holds 6 to 111 memory mappings, and Linux stops a
  process at ``vm.max_map_count`` (65,530 by default): a stream of fits over
  many shapes died at about 735 programs. :func:`in_float64` counts the
  programs traced since JAX's caches were last cleared and, at the entry of a
  call when no call of the fitter is running in any thread, clears them with
  ``jax.clear_caches()`` once the count has reached the limit. That clears
  every jitted function of the process, another library's too; a cleared
  program is compiled again on its next use.

This module imports JAX, and so does every module of the fitter that imports
``likelihoods``, whose kernels it runs.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import jax
import jax.numpy as jnp

from continuum_swr_replay.runs.device import require_one_jax_device

__all__ = [
    "COMPILE_LOG",
    "LIVE_PROGRAM_LIMIT",
    "CompileLog",
    "float64_kernel",
    "in_float64",
    "live_programs",
]

LIVE_PROGRAM_LIMIT = 256
"""Programs traced since the last clear at which the next outermost call of the
fitter clears JAX's caches. An iteration kernel took 111 mappings, an
evaluation kernel 24 and an information kernel 6 (CPU, 2026-11-23): 256 of the
largest are 28,400 mappings, under half the default ``vm.max_map_count`` with
the process's own (about 700) and one call's new programs added. Read at each
call."""

_P = ParamSpec("_P")
_R = TypeVar("_R")


def in_float64(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Run `function` with JAX's 64-bit types on, for that call only.

    The device is checked first: the fitter's JAX refuses to start unless it
    can reach at most one card
    (:func:`continuum_swr_replay.runs.device.require_one_jax_device`, which
    raises ``UnconfiguredDeviceError`` with what to set). A fresh
    ``jax.enable_x64(True)`` is entered on each call. The flag is thread-local:
    other threads, and this one after the call (also when it raises), keep
    the value they had.

    Before the call, when no call so decorated is running in any thread (so
    never inside a fit, nor while a kernel runs) and the fitter has traced
    ``LIVE_PROGRAM_LIMIT`` programs or more since the last clear, JAX's caches
    are cleared (``jax.clear_caches()``): every jitted function of the
    process, another library's too, compiles again on its next use.

    Parameters
    ----------
    function : callable
        A public entry of the fitter.

    Returns
    -------
    callable
        `function`, run in float64.
    """

    @wraps(function)
    def run(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        require_one_jax_device("The regression fitter")
        _PROGRAMS.enter()
        try:
            with jax.enable_x64(True):
                return function(*args, **kwargs)
        finally:
            _PROGRAMS.leave()

    return run


class _LivePrograms:
    """The programs traced since JAX's caches were last cleared, and the calls of
    the fitter running, in every thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.live = 0
        self.running = 0

    def enter(self) -> None:
        """Start a call; first clear the caches when no call is running and the
        live programs have reached ``LIVE_PROGRAM_LIMIT``."""
        with self._lock:
            if self.running == 0 and self.live >= LIVE_PROGRAM_LIMIT:
                jax.clear_caches()
                self.live = 0
            self.running += 1

    def leave(self) -> None:
        with self._lock:
            self.running -= 1

    def traced(self) -> None:
        with self._lock:
            self.live += 1


_PROGRAMS = _LivePrograms()


def live_programs() -> int:
    """The fitter's programs traced since JAX's caches were last cleared."""
    return _PROGRAMS.live


class CompileLog:
    """The traces of the fitter's kernels, by kernel name and input signature.

    Thread-safe. A test clears it, runs a set of fits and reads :meth:`count`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._traces: Counter[tuple[str, tuple]] = Counter()

    def record(self, name: str, signature: tuple) -> None:
        """Count one trace of kernel `name` with input `signature`."""
        with self._lock:
            self._traces[(name, signature)] += 1

    def count(self, name: str | None = None) -> int:
        """Traces of kernel `name`, or of every kernel when `name` is None."""
        with self._lock:
            return sum(
                n for (kernel, _), n in self._traces.items() if name in (None, kernel)
            )

    def signatures(self, name: str) -> list[tuple]:
        """The input signatures kernel `name` was traced with, in no order."""
        with self._lock:
            return [sig for (kernel, sig) in self._traces if kernel == name]

    def clear(self) -> None:
        """Forget every trace."""
        with self._lock:
            self._traces.clear()


COMPILE_LOG = CompileLog()
"""The process's log of kernel traces."""


def _signature(leaves: Sequence[Any]) -> tuple:
    """Shape and dtype of each array input, and the value of each other one."""
    return tuple(
        (tuple(leaf.shape), str(leaf.dtype)) if hasattr(leaf, "dtype") else leaf
        for leaf in leaves
    )


def _require_float64(name: str, leaves: Sequence[Any]) -> None:
    """Raise unless 64-bit types are on and every floating input is float64."""
    if not jax.config.x64_enabled:
        raise TypeError(
            f"kernel {name!r} was called with JAX's 64-bit types off: its inputs "
            "would be float32. Call it from a function decorated with "
            "regression.runtime.in_float64."
        )
    wrong = [
        str(leaf.dtype)
        for leaf in leaves
        if hasattr(leaf, "dtype")
        and jnp.issubdtype(leaf.dtype, jnp.floating)
        and leaf.dtype != jnp.float64
    ]
    if wrong:
        raise TypeError(
            f"kernel {name!r} takes float64 inputs only; it was given {wrong}"
        )


def float64_kernel(
    function: Callable[..., Any] | None = None,
    *,
    static_argnames: Sequence[str] = (),
) -> Any:
    """``jax.jit`` a kernel that refuses, when traced, any input but float64.

    Each trace is recorded in :data:`COMPILE_LOG` under the function's name.
    The checks run when JAX traces the kernel, not on each call, so they cost
    nothing once a signature is compiled. A rule constant is passed as an
    argument (static or traced), never read from a module when traced: a
    compiled kernel would keep the value it was traced with.

    Parameters
    ----------
    function : callable, optional
        The kernel. Without it, returns a decorator.
    static_argnames : sequence of str, optional
        Passed to ``jax.jit``.

    Returns
    -------
    callable
        The jitted kernel.

    Raises
    ------
    TypeError
        When traced with 64-bit types off, or with a floating input that is not
        float64.
    """
    if function is None:
        return lambda f: float64_kernel(f, static_argnames=static_argnames)
    name = function.__qualname__

    @wraps(function)
    def traced(*args: Any, **kwargs: Any) -> Any:
        leaves = jax.tree_util.tree_leaves((args, kwargs))
        _require_float64(name, leaves)
        COMPILE_LOG.record(name, _signature(leaves))
        _PROGRAMS.traced()
        return function(*args, **kwargs)

    return jax.jit(traced, static_argnames=tuple(static_argnames))
