"""``regression.runtime``: float64 for each call, the float64 kernel guard, the
device guard at the entry, and the log of traces."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from continuum_swr_replay.analysis.regression.runtime import (
    COMPILE_LOG,
    float64_kernel,
    in_float64,
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
