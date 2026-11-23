"""``runs.device.default_jax_platforms``: the environment's platforms, or the default.

JAX reads ``JAX_PLATFORMS`` when imported, so the choice must reach its config,
not only the environment; the environment is set too, for the processes a runner
starts.
"""

from __future__ import annotations

import os

import jax
import pytest

from continuum_swr_replay.runs import device
from continuum_swr_replay.runs.device import default_jax_platforms


@pytest.fixture
def updates(monkeypatch):
    """Every ``jax.config.update`` call, recorded instead of made."""
    calls: list[tuple] = []
    monkeypatch.setattr(jax.config, "update", lambda *args: calls.append(args))
    return calls


def test_unset_platforms_take_the_default_in_the_config_and_the_environment(
    monkeypatch, updates
):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)

    assert default_jax_platforms("cpu") == "cpu"

    assert updates == [("jax_platforms", "cpu")]
    assert os.environ["JAX_PLATFORMS"] == "cpu"


def test_platforms_the_environment_names_are_kept(monkeypatch, updates):
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")

    assert default_jax_platforms("cpu") == "cuda"

    assert updates == [("jax_platforms", "cuda")]
    assert os.environ["JAX_PLATFORMS"] == "cuda"


def test_the_recompute_runs_on_the_gpu_even_when_the_cpu_was_pinned(
    monkeypatch,
) -> None:
    """A shell or a caller may pin JAX_PLATFORMS to the CPU; the recompute still
    asks JAX for the GPU, under the memoryless run's XLA flags."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("XLA_FLAGS", "")
    updates = []
    monkeypatch.setattr(jax.config, "update", lambda *args: updates.append(args))

    device.use_memoryless_device()

    assert updates == [("jax_platforms", "cuda")]
    assert os.environ["JAX_PLATFORMS"] == "cuda"
    for flag in device.REQUIRED_XLA_FLAGS:
        assert flag in os.environ["XLA_FLAGS"].split()


@pytest.mark.parametrize(
    ("platforms", "visible", "allowed"),
    [
        ("cpu", None, True),
        ("cpu", "GPU-a,GPU-b", True),  # JAX_PLATFORMS keeps every card away
        (None, "GPU-a", True),  # one card
        (None, "", True),  # no card
        ("cuda", "GPU-a", True),
        (None, None, False),  # every card on the host
        (None, "GPU-a,GPU-b", False),
        ("cuda", None, False),
        ("cuda,cpu", "0,1", False),
    ],
)
def test_jax_may_compute_only_where_it_reaches_at_most_one_card(
    monkeypatch, platforms, visible, allowed
):
    monkeypatch.setattr(device, "_CONFIGURED", None)
    for name, value in (
        ("JAX_PLATFORMS", platforms),
        ("CUDA_VISIBLE_DEVICES", visible),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    if allowed:
        device.require_one_jax_device("The fitter")
    else:
        with pytest.raises(device.UnconfiguredDeviceError) as raised:
            device.require_one_jax_device("The fitter")
        message = str(raised.value)
        assert message.startswith("The fitter computes with JAX")
        assert "JAX_PLATFORMS=cpu" in message
        assert "configure_jax_gpu()" in message


def test_a_configured_device_lets_jax_compute(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(device, "_CONFIGURED", {"jax_backend": "gpu"})

    device.require_one_jax_device("The fitter")
