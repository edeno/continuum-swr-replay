"""Choosing a GPU on a machine other people are using.

The rules the tests pin are the ones that keep a batch off somebody else's
job: a card in use is not free however much memory it has left, a claimed card
is not free to a sibling worker, a process that already has a card keeps it
rather than taking a second, "no free card" is an error carrying the occupancy
table rather than a silent fall back to CPU, and a process that ended up on
more than the one card it configured says so instead of running.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from continuum_swr_replay.analysis.pfc import gpu as gpu_module
from continuum_swr_replay.analysis.pfc.gpu import (
    NoFreeGPUError,
    WrongJAXDeviceError,
    configure_jax_gpu,
    occupancy_table,
    parse_gpu_query,
    select_free_gpu,
    verify_jax_device,
)


@dataclass(frozen=True)
class FakeDevice:
    """A stand-in for a ``jax.Device``, carrying only what the check reads."""

    platform: str = "gpu"
    local_hardware_id: int = 0
    device_kind: str = "NVIDIA A100 80GB PCIe"


#: The host this was written on: six idle cards carrying other users' contexts,
#: four carrying their jobs.
OCCUPANCY = """\
0, 13097, 81920, 0, GPU-aaa, NVIDIA A100 80GB PCIe
1, 7366, 81920, 0, GPU-bbb, NVIDIA A100 80GB PCIe
2, 443, 81920, 0, GPU-ccc, NVIDIA A100 80GB PCIe
3, 443, 81920, 55, GPU-ddd, NVIDIA A100 80GB PCIe
4, 300, 81920, 0, GPU-eee, NVIDIA A100 80GB PCIe
5, 79390, 81920, 0, GPU-fff, NVIDIA A100 80GB PCIe
"""


@pytest.fixture
def claims(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_module, "CLAIM_DIRECTORY", tmp_path / "claims")
    # Each test is its own process as far as the selector is concerned, and
    # the device check is exercised on its own below rather than by
    # initialising a real JAX backend on a real card from the suite.
    monkeypatch.setattr(gpu_module, "_CONFIGURED", None)
    monkeypatch.setattr(gpu_module, "verify_jax_device", lambda record: None)
    # The suite pins JAX to CPU so no test can take a card. The selection
    # tests are about the selection rules, so they run as if unpinned; the
    # pin has its own test.
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    return tmp_path / "claims"


@pytest.fixture
def occupied(monkeypatch):
    monkeypatch.setattr(gpu_module, "query_gpus", lambda: parse_gpu_query(OCCUPANCY))


class TestParsing:
    def test_every_field_survives_the_csv(self):
        cards = parse_gpu_query(OCCUPANCY)

        assert [card.index for card in cards] == [0, 1, 2, 3, 4, 5]
        assert cards[0].used_mib == 13097
        assert cards[0].total_mib == 81920
        assert cards[3].utilization_pct == 55
        assert cards[2].uuid == "GPU-ccc"
        assert cards[2].name == "NVIDIA A100 80GB PCIe"

    def test_blank_and_short_lines_are_skipped(self):
        assert parse_gpu_query("\n\nnot, enough, fields\n") == []


class TestSelection:
    def test_the_emptiest_idle_card_is_chosen(self, occupied, claims):
        chosen = select_free_gpu()

        # 4 is emptier than 2; 0, 1 and 5 carry other people's jobs and 3 is
        # busy despite being nearly empty.
        assert chosen.index == 4
        assert chosen.uuid == "GPU-eee"

    def test_a_busy_card_is_not_free_however_empty(self, occupied, claims):
        assert select_free_gpu().index != 3

    def test_a_card_a_sibling_worker_holds_goes_to_the_next_one(self, occupied, claims):
        # A live PID that is not this one, so the claim reads as a sibling's
        # rather than as this process re-taking its own.
        claims.mkdir(parents=True, exist_ok=True)
        (claims / "gpu4.claim").write_text(str(os.getppid()))

        assert select_free_gpu().index == 2

    def test_a_stale_claim_is_taken_over(self, occupied, claims):
        select_free_gpu()
        # A claim whose process is gone must not park a card forever.
        (claims / "gpu4.claim").write_text("999999999")

        assert select_free_gpu().index == 4

    def test_no_free_card_raises_with_the_occupancy(self, monkeypatch, claims):
        busy = "\n".join(
            f"{index}, 50000, 81920, 0, GPU-{index}, A100" for index in range(3)
        )
        monkeypatch.setattr(gpu_module, "query_gpus", lambda: parse_gpu_query(busy))

        with pytest.raises(NoFreeGPUError, match="50000/81920 MiB used"):
            select_free_gpu()

    def test_no_driver_at_all_raises_too(self, monkeypatch, claims):
        monkeypatch.setattr(gpu_module, "query_gpus", list)

        with pytest.raises(NoFreeGPUError, match="no NVIDIA GPUs are visible"):
            select_free_gpu()


class TestConfiguration:
    def test_the_chosen_card_is_the_only_one_jax_can_see(
        self, occupied, claims, monkeypatch
    ):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        record = configure_jax_gpu()

        assert os.environ["CUDA_VISIBLE_DEVICES"] == "4"
        assert record == {
            "jax_backend": "gpu",
            "gpu_index": 4,
            "gpu_uuid": "GPU-eee",
            "gpu_name": "NVIDIA A100 80GB PCIe",
        }

    def test_the_card_is_not_preallocated_whole(self, occupied, claims, monkeypatch):
        monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
        monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)

        configure_jax_gpu(mem_fraction=0.5)

        assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
        assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.5"

    def test_a_forced_card_is_taken_whatever_is_on_it(
        self, occupied, claims, monkeypatch
    ):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        record = configure_jax_gpu(gpu=0)

        assert record["gpu_index"] == 0
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"

    def test_cpu_is_refused_unless_asked_for(self, monkeypatch, claims):
        monkeypatch.setattr(gpu_module, "query_gpus", list)

        with pytest.raises(NoFreeGPUError):
            configure_jax_gpu()

    def test_cpu_is_allowed_when_asked_for(self, monkeypatch, claims):
        monkeypatch.setattr(gpu_module, "query_gpus", list)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        record = configure_jax_gpu(allow_cpu=True)

        assert record["jax_backend"] == "cpu"
        assert record["gpu_index"] is None
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""

    def test_a_cpu_pinned_environment_never_claims_a_card(
        self, occupied, claims, monkeypatch
    ):
        # ``JAX_PLATFORMS=cpu`` makes a card unreachable however free it is, so
        # claiming one would park it for a process that cannot use it -- and
        # the run would then fail verification anyway.
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        with pytest.raises(NoFreeGPUError, match="JAX_PLATFORMS"):
            configure_jax_gpu()

        record = configure_jax_gpu(allow_cpu=True)
        assert record["jax_backend"] == "cpu"
        assert not claims.exists() or not list(claims.iterdir())

    def test_a_second_session_in_the_same_worker_keeps_its_card(
        self, occupied, claims, monkeypatch
    ):
        # A Dask worker handed a second session must not select again. JAX is
        # bound to a device by then, so a second CUDA_VISIBLE_DEVICES changes
        # nothing except what the second session's sidecar claims -- and it
        # would consume a free card that nothing releases until the process
        # exits, which is how a four-worker batch runs out of cards.
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        first = configure_jax_gpu()
        second = configure_jax_gpu()

        assert first == second
        assert os.environ["CUDA_VISIBLE_DEVICES"] == str(first["gpu_index"])
        assert sorted(path.name for path in claims.iterdir()) == [
            f"gpu{first['gpu_index']}.claim"
        ]

    def test_reclaiming_the_card_this_process_holds_succeeds(self, claims):
        # The claim file is not a mutex against its own holder; treating it as
        # one is what sent a second session to a different card.
        assert gpu_module._claim(7)
        assert gpu_module._claim(7)
        assert (claims / "gpu7.claim").read_text().strip() == str(os.getpid())


class TestDeviceVerification:
    def test_one_configured_card_and_one_visible_device_passes(self):
        record = {
            "jax_backend": "gpu",
            "gpu_index": 4,
            "gpu_uuid": "GPU-eee",
            "gpu_name": "NVIDIA A100 80GB PCIe",
        }
        verify_jax_device(record, devices=[FakeDevice()])

    def test_seeing_every_card_is_an_error_not_a_silent_grab(self):
        # The failure this exists for: the backend was initialised before
        # CUDA_VISIBLE_DEVICES was set, so the process is on all ten cards --
        # other people's included -- and nothing says so until their jobs die.
        record = {
            "jax_backend": "gpu",
            "gpu_index": 4,
            "gpu_uuid": "GPU-eee",
            "gpu_name": "NVIDIA A100 80GB PCIe",
        }
        with pytest.raises(WrongJAXDeviceError, match="sees 10 devices"):
            verify_jax_device(record, devices=[FakeDevice()] * 10)

    def test_a_cpu_device_where_a_gpu_was_configured_is_an_error(self):
        record = {
            "jax_backend": "gpu",
            "gpu_index": 4,
            "gpu_uuid": "GPU-eee",
            "gpu_name": "NVIDIA A100 80GB PCIe",
        }
        with pytest.raises(WrongJAXDeviceError, match="'cpu' device"):
            verify_jax_device(record, devices=[FakeDevice(platform="cpu")])

    def test_a_nonzero_local_ordinal_means_the_mapping_never_took_effect(self):
        record = {
            "jax_backend": "gpu",
            "gpu_index": 4,
            "gpu_uuid": "GPU-eee",
            "gpu_name": "NVIDIA A100 80GB PCIe",
        }
        with pytest.raises(WrongJAXDeviceError, match="local ordinal 4"):
            verify_jax_device(record, devices=[FakeDevice(local_hardware_id=4)])

    def test_a_different_card_model_is_an_error(self):
        record = {
            "jax_backend": "gpu",
            "gpu_index": 4,
            "gpu_uuid": "GPU-eee",
            "gpu_name": "NVIDIA A100 80GB PCIe",
        }
        with pytest.raises(WrongJAXDeviceError):
            verify_jax_device(
                record, devices=[FakeDevice(device_kind="NVIDIA H100 80GB HBM3")]
            )

    def test_a_cpu_run_wants_a_cpu_device(self):
        record = {
            "jax_backend": "cpu",
            "gpu_index": None,
            "gpu_uuid": None,
            "gpu_name": None,
        }
        verify_jax_device(record, devices=[FakeDevice(platform="cpu")])
        with pytest.raises(WrongJAXDeviceError, match="'gpu' device"):
            verify_jax_device(record, devices=[FakeDevice(platform="gpu")])


def test_the_occupancy_table_names_every_card():
    table = occupancy_table(parse_gpu_query(OCCUPANCY))

    assert table.count("GPU ") == 6
    assert "13097/81920 MiB used, 0% utilisation" in table
