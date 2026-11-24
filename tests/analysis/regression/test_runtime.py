"""``regression.runtime``: float64 for each call, the float64 kernel guard, the
device guard at the entry, and the log of traces."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from continuum_swr_replay.analysis.regression import runtime
from continuum_swr_replay.analysis.regression.runtime import (
    COMPILE_LOG,
    float64_kernel,
    in_float64,
    live_programs,
)
from continuum_swr_replay.runs import device


@float64_kernel
def _tiny_sum(x):
    """``sum(x - 1)``: about 1e-12 per entry of `VALUES` in float64, 0 in
    float32."""
    return jnp.sum(x - 1.0)


@float64_kernel(static_argnames=("power",))
def _powered(x, power):
    return x**power


@in_float64
def _public_entry(values: np.ndarray) -> float:
    """A public entry: NumPy in, a kernel run, NumPy out."""
    return float(_tiny_sum(jnp.asarray(values, dtype=jnp.float64)))


@in_float64
def _state() -> tuple[bool, str]:
    """The flag and the dtype of an array made from float64 NumPy, inside."""
    return jax.config.x64_enabled, str(jnp.asarray(np.ones(2)).dtype)


VALUES = np.full(1000, 1.0 + 1e-12)


def test_the_flag_is_on_inside_and_as_it_was_after():
    assert not jax.config.x64_enabled
    assert _state() == (True, "float64")
    assert not jax.config.x64_enabled
    with jax.enable_x64(True):
        assert _state() == (True, "float64")
        assert jax.config.x64_enabled


def test_the_flag_is_as_it_was_after_a_call_that_raises():
    @in_float64
    def fails():
        assert jax.config.x64_enabled
        raise RuntimeError("inside")

    with pytest.raises(RuntimeError, match="inside"):
        fails()
    assert not jax.config.x64_enabled


def test_a_call_in_a_thread_leaves_every_thread_as_it_was():
    with ThreadPoolExecutor(max_workers=1) as pool:
        before = pool.submit(lambda: jax.config.x64_enabled).result()
        inside = pool.submit(_state).result()
        after = pool.submit(lambda: jax.config.x64_enabled).result()
    assert (before, inside, after) == (False, (True, "float64"), False)
    assert not jax.config.x64_enabled


def test_the_callers_arrays_stay_float32_after_a_call():
    _public_entry(VALUES)
    assert jnp.asarray(np.ones(2)).dtype == jnp.float32


def test_an_array_made_inside_stays_float64():
    made = in_float64(lambda: jnp.asarray(np.ones(2)))()
    assert made.dtype == jnp.float64


def test_a_kernel_called_with_the_flag_off_raises():
    with pytest.raises(TypeError, match="64-bit types off"):
        _tiny_sum(VALUES)


def test_a_kernel_given_float32_raises_inside_the_scope():
    with jax.enable_x64(True), pytest.raises(TypeError, match="float32"):
        _tiny_sum(np.ones(3, dtype=np.float32))


def test_integer_inputs_pass_the_guard():
    with jax.enable_x64(True):
        assert int(_powered(np.arange(3), power=2)[-1]) == 4


def test_the_result_is_the_same_with_the_callers_flag_on_or_off():
    off = _public_entry(VALUES)
    with jax.enable_x64(True):
        on = _public_entry(VALUES)
    assert off == on
    np.testing.assert_allclose(off, np.sum(VALUES - 1.0), rtol=1e-12)
    assert off != 0.0  # float32 would round each entry to 1


def test_the_log_counts_one_trace_per_signature():
    COMPILE_LOG.clear()
    with jax.enable_x64(True):
        for n in (100, 101, 100, 128, 101):
            _tiny_sum(np.ones(n))
        _powered(np.ones(4), power=2)
        _powered(np.ones(4), power=3)
        _powered(np.ones(4), power=2)
    assert COMPILE_LOG.count("_tiny_sum") == 3
    assert sorted(COMPILE_LOG.signatures("_tiny_sum")) == [
        (((n,), "float64"),) for n in (100, 101, 128)
    ]
    assert COMPILE_LOG.count("_powered") == 2
    assert COMPILE_LOG.count() == 5
    COMPILE_LOG.clear()
    assert COMPILE_LOG.count() == 0


def test_the_entry_refuses_before_running_when_no_device_was_chosen(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    monkeypatch.setattr(device, "_CONFIGURED", None)
    ran = []

    with pytest.raises(device.UnconfiguredDeviceError, match="JAX_PLATFORMS=cpu"):
        in_float64(lambda: ran.append(True))()
    assert ran == []


def test_the_fitters_public_calls_leave_the_flag_as_it_was():
    """A fit, a REML score, a likelihood method and the weighted design, in
    this thread and in another: the process's flag is off before and after."""
    from continuum_swr_replay.analysis.regression.collinearity import weighted_design
    from continuum_swr_replay.analysis.regression.likelihoods import Gaussian, Poisson
    from continuum_swr_replay.analysis.regression.newton import penalized_newton
    from continuum_swr_replay.analysis.regression.reml import (
        PenaltyBlock,
        reml_and_derivatives,
    )

    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(50), rng.normal(size=(50, 2))])
    y = rng.poisson(1.0, size=50).astype(float)
    offset = np.zeros(50)
    blocks = [PenaltyBlock(np.array([1, 2]), np.eye(2))]

    def calls():
        fit = penalized_newton(Poisson(), X, y, offset, np.zeros((3, 3)), np.zeros(3))
        reml_and_derivatives(Poisson(), X, y, offset, blocks, [0.0])
        reml_and_derivatives(Gaussian(), X, y, None, blocks, [0.0])
        Poisson().information(X, X @ fit.beta, y, offset)
        weighted_design(Poisson(), X, y, offset, X @ fit.beta)
        return jax.config.x64_enabled

    assert not jax.config.x64_enabled
    calls()
    assert not jax.config.x64_enabled
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert not pool.submit(calls).result()
    assert jnp.asarray(np.ones(2)).dtype == jnp.float32


def _mappings() -> int | None:
    """The process's memory mappings (Linux), else None."""
    if not os.path.exists("/proc/self/maps"):
        return None
    with open("/proc/self/maps") as maps:
        return sum(1 for _ in maps)


def _stream_of_fits():
    """Poisson fits of 2 to 9 columns and two row buckets: 16 designs, each
    tracing new programs."""
    from continuum_swr_replay.analysis.regression.likelihoods import Poisson
    from continuum_swr_replay.analysis.regression.newton import penalized_newton

    fits = []
    for p in range(2, 10):
        for n in (300, 700):
            rng = np.random.default_rng(10 * p + n)
            X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
            y = rng.poisson(np.exp(0.2 + 0.1 * X[:, 1])).astype(float)
            fit = penalized_newton(Poisson(), X, y, np.zeros(n), np.eye(p), np.zeros(p))
            fits.append((fit, live_programs()))
    return fits


def test_the_live_programs_stay_bounded_and_the_fits_do_not_change(monkeypatch):
    """With the limit at 5, a stream of fits over 16 designs clears JAX's
    caches several times, never lets the live programs pass the limit by
    more than one fit's programs, and fits every design as without the bound;
    on Linux each clear releases memory mappings."""
    monkeypatch.setattr(runtime, "LIVE_PROGRAM_LIMIT", 10**9)
    unbounded = [fit for fit, _ in _stream_of_fits()]
    clears = []
    clear = jax.clear_caches

    def counted():
        before = _mappings()
        clear()
        clears.append((before, _mappings()))

    monkeypatch.setattr(jax, "clear_caches", counted)
    monkeypatch.setattr(runtime, "LIVE_PROGRAM_LIMIT", 5)
    jax.clear_caches()
    clears.clear()
    runtime._PROGRAMS.live = 0
    bounded = _stream_of_fits()
    assert len(clears) >= 3
    # A fit traces at most two programs (evaluation and iteration).
    assert max(live for _, live in bounded) <= 5 - 1 + 2
    for (got, _), want in zip(bounded, unbounded, strict=True):
        assert (got.status, got.n_iter) == (want.status, want.n_iter)
        np.testing.assert_array_equal(got.beta, want.beta)
        np.testing.assert_array_equal(got.A, want.A)
    if _mappings() is not None:
        assert all(after < before for before, after in clears)


def test_the_caches_are_cleared_only_between_outermost_calls(monkeypatch):
    """With the limit at 0, a REML score (which runs a fit and the
    likelihood's public methods inside it) clears the caches once, at its
    entry, with no call of the fitter running."""
    from continuum_swr_replay.analysis.regression.likelihoods import Poisson
    from continuum_swr_replay.analysis.regression.reml import (
        PenaltyBlock,
        reml_and_derivatives,
    )

    running = []
    clear = jax.clear_caches

    def watched():
        running.append(runtime._PROGRAMS.running)
        clear()

    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(50), rng.normal(size=(50, 2))])
    y = rng.poisson(1.0, size=50).astype(float)
    blocks = [PenaltyBlock(np.array([1, 2]), np.eye(2))]
    monkeypatch.setattr(jax, "clear_caches", watched)
    monkeypatch.setattr(runtime, "LIVE_PROGRAM_LIMIT", 0)
    score = reml_and_derivatives(Poisson(), X, y, np.zeros(50), blocks, [0.0])
    assert score.fit.converged
    assert running == [0]
    assert runtime._PROGRAMS.running == 0
