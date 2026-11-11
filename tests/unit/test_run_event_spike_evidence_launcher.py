"""The spike-evidence launcher reports each epoch truthfully and never leaves a
half-written epoch where the readers look.

The launcher is run for real, in a copy of the checkout's layout whose ``python``
is a stand-in: it answers the launcher's two questions (the output folder, the
epochs) and plays the analysis script, writing the metrics file or not as the
epoch's name says.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

LAUNCHER = (
    Path(__file__).parents[2] / "scripts" / "detection" / "run_event_spike_evidence.sh"
)
METRICS = "event_spike_evidence_metrics.parquet"

STAND_IN = r"""#!/usr/bin/env bash
if [ "$1" = "-c" ]; then
  case "$2" in
    *event_spike_evidence_dir*) [ -n "${STAND_IN_NO_ROOT:-}" ] && exit 1; echo "$STAND_IN_OUT" ;;
    *find_sessions*) for epoch in $STAND_IN_EPOCHS; do echo "$epoch"; done ;;
  esac
  exit 0
fi
echo "$*" >> "$STAND_IN_CALLS"
while [ $# -gt 0 ]; do
  case "$1" in
    --epoch-identifiers) epoch="$2"; shift ;;
    --output-dir) out="$2"; shift ;;
  esac
  shift
done
case "$epoch" in
  raises*) echo "  $epoch: FAILED (ValueError)"; echo "No sessions analyzed."; exit 0 ;;
  crashes*) exit 1 ;;
  dies_writing*) mkdir -p "$out"; echo half > "$out/event_spike_evidence_metrics.parquet"; exit 137 ;;
  *) mkdir -p "$out"; echo metrics > "$out/event_spike_evidence_metrics.parquet" ;;
esac
"""


@pytest.fixture
def checkout(tmp_path):
    """A checkout's layout with the launcher and a stand-in ``python``."""
    repo = tmp_path / "repo"
    (repo / "scripts" / "detection").mkdir(parents=True)
    shutil.copy(LAUNCHER, repo / "scripts" / "detection" / LAUNCHER.name)
    stand_in_dir = tmp_path / "bin"
    stand_in_dir.mkdir()
    python = stand_in_dir / "python"
    python.write_text(STAND_IN)
    python.chmod(0o755)
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "activate").write_text(
        f'export PATH="{stand_in_dir}:$PATH"\n'
    )
    return repo


def launch(checkout: Path, epochs: str, *arguments: str, no_root: bool = False):
    out = checkout.parent / "data" / "event_spike_evidence"
    environment = {
        "PATH": "/usr/bin:/bin",
        "STAND_IN_OUT": str(out),
        "STAND_IN_EPOCHS": epochs,
        "STAND_IN_CALLS": str(checkout.parent / "calls.txt"),
    }
    if no_root:
        environment["STAND_IN_NO_ROOT"] = "1"
    run = subprocess.run(
        ["bash", str(checkout / "scripts/detection" / LAUNCHER.name), *arguments],
        capture_output=True,
        text=True,
        env=environment,
        cwd=checkout.parent,
        check=False,
    )
    return run, out


def test_every_epoch_scored_is_reported_ok_and_the_run_done(checkout):
    run, out = launch(checkout, "a_HPC b_HPC", "2", "--cache-dir", "/a cache")

    assert sorted(line.split()[:2] for line in run.stdout.splitlines()[:2]) == [
        ["a_HPC", "ok"],
        ["b_HPC", "ok"],
    ]
    assert run.stdout.splitlines()[-1] == "EVENT_SPIKE_EVIDENCE_DONE"
    assert run.returncode == 0
    assert (out / "a_HPC" / METRICS).read_text() == "metrics\n"
    assert (out / "b_HPC" / METRICS).exists()
    calls = (checkout.parent / "calls.txt").read_text()
    assert calls.count("--n-sample 1000000 --likelihood stored") == 2
    assert calls.count("--cache-dir /a cache") == 2


def test_an_epoch_already_scored_is_skipped(checkout):
    launch(checkout, "a_HPC", "1")
    (checkout.parent / "calls.txt").unlink()

    run, _ = launch(checkout, "a_HPC b_HPC", "1")

    assert "a_HPC skip" in run.stdout.splitlines()
    assert (checkout.parent / "calls.txt").read_text().count("--epoch-identifiers") == 1
    assert run.returncode == 0


@pytest.mark.parametrize("epoch", ["raises_HPC", "crashes_HPC", "dies_writing_HPC"])
def test_an_epoch_without_its_metrics_is_reported_failed(checkout, epoch):
    # The analysis script exits 0 when an epoch's analysis raises; a killed run
    # can leave half a metrics file. Neither is "ok", and neither may be skipped
    # on the next run or read by the summary.
    run, out = launch(checkout, f"a_HPC {epoch}", "1")

    lines = run.stdout.splitlines()
    assert any(line.startswith(f"{epoch} FAILED") for line in lines)
    assert not any(line.startswith(f"{epoch} ok") for line in lines)
    assert any(line.startswith("a_HPC ok") for line in lines)
    assert "EVENT_SPIKE_EVIDENCE_DONE" not in lines
    assert run.returncode != 0
    assert not (out / epoch).exists()
    assert [path.parent.name for path in out.glob(f"*/{METRICS}")] == ["a_HPC"]


def test_no_epoch_is_run_when_the_data_root_cannot_be_read(checkout):
    run, out = launch(checkout, "a_HPC", "1", no_root=True)

    assert run.returncode != 0
    assert "EVENT_SPIKE_EVIDENCE_DONE" not in run.stdout
    assert not (checkout.parent / "calls.txt").exists()
    assert not out.exists()


def test_no_epoch_to_score_is_an_error(checkout):
    run, _ = launch(checkout, "", "1")

    assert run.returncode != 0
    assert "EVENT_SPIKE_EVIDENCE_DONE" not in run.stdout
