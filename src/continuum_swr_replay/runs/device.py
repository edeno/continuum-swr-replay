"""Where a runner's JAX computes, and the XLA flags of a deterministic GPU run.

JAX reads ``JAX_PLATFORMS`` once, when it is imported, and starts its backend
at its first computation, which reads ``XLA_FLAGS`` once. The package imports
JAX, so a runner chooses its device in ``main``, after its imports and before
any computation, with :func:`default_jax_platforms` (which updates JAX's
config, not only the environment). Nothing here imports JAX until it is called.
"""

from __future__ import annotations

import os

__all__ = [
    "DETERMINISTIC_XLA_FLAG",
    "REQUIRED_XLA_FLAGS",
    "TRITON_GEMM_OFF_XLA_FLAG",
    "default_jax_platforms",
    "recompute_device",
    "use_memoryless_device",
    "xla_flags",
]

DETERMINISTIC_XLA_FLAG = "--xla_gpu_deterministic_ops=true"
"""Makes XLA's GPU operations deterministic, so two memoryless runs at identical
settings are bit-for-bit equal (``run_memoryless_detection.KNOWN_LIMITS``). XLA
reads ``XLA_FLAGS`` once in a process, when JAX starts its backend; importing
this module does not start it."""

TRITON_GEMM_OFF_XLA_FLAG = "--xla_gpu_enable_triton_gemm=false"
"""Turns off XLA's Triton GEMM for matrix products; XLA then uses another GEMM path
(presumably cuBLAS; not checked). Deterministic ops turn off XLA's autotuning, and the default Triton GEMM setup then fails on some
shapes (the working explanation; not confirmed: the spike tested the fix, not the
mechanism). With deterministic ops alone, R12 lost 3 of 242 epochs in the shuffle to
``JaxRuntimeError('CANCELLED: Too small divisible part of the contracting
dimension.')``, raised at ``non_local_detector/likelihoods/clusterless_kde.py:89``
(``spike_waveform_feature_distance.T @ position_distance``). With this flag as well,
one of them, chimi20200224_08_r4_HPC, ran in the spike
(``run_memoryless_detection.KNOWN_LIMITS``)."""

REQUIRED_XLA_FLAGS = (DETERMINISTIC_XLA_FLAG, TRITON_GEMM_OFF_XLA_FLAG)
"""The flags every process that runs the memoryless detector's JAX gets, in the
order :func:`xla_flags` appends them."""


def xla_flags() -> str:
    """The ``XLA_FLAGS`` every process that runs JAX gets: this process's, with
    each flag in :data:`REQUIRED_XLA_FLAGS` added at the end unless the last
    occurrence of that flag's name in them already has the required value.

    XLA uses the last value of a repeated flag, so an earlier
    ``--xla_gpu_deterministic_ops=true`` is overridden by a later ``=false``, an
    earlier ``--xla_gpu_enable_triton_gemm=true`` by a later ``=false``, and so
    on; the appended flag overrides them. A flag that is already right is not
    added again. The user's other flags are kept as they are. Triton GEMM is
    off because deterministic ops turn off autotuning and the default Triton GEMM
    setup then fails on some shapes (the working explanation; not confirmed;
    :data:`TRITON_GEMM_OFF_XLA_FLAG`). The memoryless runner's settings record
    the value (``run_settings``), and its ``main`` hands that record to the
    processes: the dask workers (``cuda_cluster``), or this process when it runs
    the epochs itself.
    """
    flags = os.environ.get("XLA_FLAGS", "").strip()
    for required in REQUIRED_XLA_FLAGS:
        name = required.split("=")[0]
        given = [flag for flag in flags.split() if flag.split("=")[0] == name]
        if not given or given[-1] != required:
            flags = f"{flags} {required}".lstrip()
    return flags


def default_jax_platforms(platforms: str = "cpu") -> str:
    """Run JAX on ``JAX_PLATFORMS``, or on `platforms` when it is unset.

    ``JAX_PLATFORMS`` is set to `platforms` when unset, so the processes this
    one starts inherit the choice, and JAX's ``jax_platforms`` config is set to
    the result: JAX read the variable when it was imported, so setting the
    variable alone would change nothing in this process. Call it before JAX's
    first computation, which starts its backend; afterwards the backend stays
    where it started.

    Parameters
    ----------
    platforms : str, optional
        The platforms when ``JAX_PLATFORMS`` is unset, as JAX spells them
        (``"cpu"`` by default).

    Returns
    -------
    str
        The platforms JAX runs on: ``JAX_PLATFORMS`` as it now stands.
    """
    import jax

    chosen = os.environ.setdefault("JAX_PLATFORMS", platforms)
    jax.config.update("jax_platforms", chosen)
    return chosen


def use_memoryless_device() -> None:
    """Recompute where the memoryless run computed: a GPU, under its XLA flags.

    Near-tied window centers are broken by the device's rounding, so a CPU
    moves more of them than the check allows. The GPU is set over any
    ``JAX_PLATFORMS`` the shell or a caller chose; with no GPU, JAX raises. Call
    before JAX starts its backend, which reads ``XLA_FLAGS`` once.
    """
    import jax

    os.environ["XLA_FLAGS"] = xla_flags()
    os.environ["JAX_PLATFORMS"] = "cuda"
    jax.config.update("jax_platforms", "cuda")


def recompute_device() -> dict[str, str]:
    """The device the windows are recomputed on and its XLA flags, for the
    manifest."""
    import jax

    return {
        "device": str(jax.devices()[0]),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }
