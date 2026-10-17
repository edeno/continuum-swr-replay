"""The refit driver: its run list, what a selection adds, and how runs are scheduled."""

import json
import sys

import pytest

from scripts.qa import run_refits as rr


def test_every_dependency_is_a_run_listed_before_it():
    seen = set()
    for run in rr.RUNS:
        assert set(run.deps) <= seen, run.id
        seen.add(run.id)


def test_every_script_exists():
    for run in rr.RUNS:
        assert (rr.REPO / run.argv[0]).is_file(), run.argv[0]


def test_selecting_c01_adds_the_contrasts_and_the_figures_that_read_them():
    ids = [r.id for r in rr.select(["C01"])]
    assert ids == ["C01", "K01", "K02", "K03", "K04", "K05", "K06", "K07", "G06", "G08"]


def test_no_dependents_selects_only_what_is_named():
    assert [r.id for r in rr.select(["C01"], with_dependents=False)] == ["C01"]


def test_a_group_selects_its_runs():
    ids = {r.id for r in rr.select(["pfc"], with_dependents=False)}
    assert ids == {"P01", "P02", "P03", *(f"S0{i}" for i in range(1, 9))}


def test_an_unknown_name_is_refused():
    with pytest.raises(SystemExit):
        rr.select(["C99"])


def test_the_environment_caps_threads_and_keeps_jax_on_cpu():
    env = rr.run_env(rr.Run("T", "x", ("t.py",), cores=4, threads=3))
    assert {env[v] for v in rr.THREAD_VARS} == {"3"}
    assert env["JAX_PLATFORMS"] == "cpu" and env["CUDA_VISIBLE_DEVICES"] == ""
    assert "intra_op_parallelism_threads=3" in env["XLA_FLAGS"]


def _py(code):
    return (sys.executable, "-c", code)


def test_a_failed_run_blocks_its_dependents_and_nothing_else(tmp_path):
    out = tmp_path / "threads.txt"
    runs = [
        rr.Run(
            "A",
            "x",
            _py(
                f"import os; open({str(out)!r}, 'w').write(os.environ['OMP_NUM_THREADS'])"
            ),
            1,
            threads=2,
        ),
        rr.Run("B", "x", _py("raise SystemExit(3)"), 1),
        rr.Run("C", "x", _py("pass"), 1, deps=("B",)),
        rr.Run(
            "D", "x", _py("pass"), 1, deps=("A", "Z")
        ),  # Z is not selected: taken as done
        rr.Run("E", "x", _py("pass"), 1, deps=("C",)),
    ]
    codes = rr.schedule(
        runs, budget=2, log_dir=tmp_path / "logs", prefix=[], poll_s=0.05
    )
    assert codes == {"A": 0, "B": 3, "C": None, "D": 0, "E": None}
    assert out.read_text() == "2"
    record = json.loads((tmp_path / "logs" / "B.json").read_text())
    assert record["exit_code"] == 3 and record["argv"][-1] == "raise SystemExit(3)"
    assert not (tmp_path / "logs" / "C.json").exists()


def test_runs_wait_for_cores_and_a_run_larger_than_the_budget_still_runs(tmp_path):
    stamp = "import time, sys; print(time.time()); time.sleep(0.3); print(time.time())"
    runs = [rr.Run(i, "x", _py(stamp), 2) for i in ("A", "B")] + [
        rr.Run("C", "x", _py("pass"), 9)
    ]
    logs = tmp_path / "logs"
    codes = rr.schedule(runs, budget=3, log_dir=logs, prefix=[], poll_s=0.02)
    assert codes == {"A": 0, "B": 0, "C": 0}
    a_end = float((logs / "A.log").read_text().split()[1])
    b_start = float((logs / "B.log").read_text().split()[0])
    assert b_start >= a_end
