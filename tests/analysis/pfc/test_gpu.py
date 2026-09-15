"""Choosing a GPU on a machine other people are using.

The rules the tests pin are the ones that keep a batch off somebody else's
job: a card in use is not free however much memory it has left, a claimed card
is not free to a sibling worker, and "no free card" is an error carrying the
occupancy table rather than a silent fall back to CPU.
"""

from __future__ import annotations

import os

import pytest

from continuum_swr_replay.analysis.pfc import gpu as gpu_module
from continuum_swr_replay.analysis.pfc.gpu import (
    NoFreeGPUError,
    configure_jax_gpu,
    occupancy_table,
    parse_gpu_query,
    select_free_gpu,
)

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
    return tmp_path / "claims"


@pytest.fixture
def occupied(monkeypatch):
    monkeypatch.setattr(
        gpu_module, "query_gpus", lambda: parse_gpu_query(OCCUPANCY)
    )


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

    def test_a_claimed_card_goes_to_the_next_one(self, occupied, claims):
        first = select_free_gpu()
        second = select_free_gpu()

        assert first.index == 4
        assert second.index == 2

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


def test_the_occupancy_table_names_every_card():
    table = occupancy_table(parse_gpu_query(OCCUPANCY))

    assert table.count("GPU ") == 6
    assert "13097/81920 MiB used, 0% utilisation" in table
