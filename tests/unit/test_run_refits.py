"""The refit driver: its run list, what a selection adds, and how runs are scheduled."""

import json
import sys

import pytest

from scripts.qa import run_refits as rr


def test_every_dependency_is_a_run_listed_before_it():
    seen = set()
    for run in rr.RUNS:
        assert set(run.deps) <= seen, run.name
        seen.add(run.name)


def test_every_script_exists_and_names_are_unique():
    for run in rr.RUNS:
        assert (rr.REPO / run.argv[0]).is_file(), run.argv[0]
    assert len(rr.BY_NAME) == len(rr.RUNS)


def test_every_run_answers_a_listed_question():
    assert {r.question for r in rr.RUNS} <= set(rr.QUESTIONS)


def test_selecting_the_arm_ladder_adds_the_contrasts_and_the_figures_that_read_them():
    names = [r.name for r in rr.select(["content.arm_ladder.all_arms"])]
    assert names == [
        "content.arm_ladder.all_arms",
        "content.swr_contrast.all",
        "content.swr_contrast.rewarded",
        "content.swr_contrast.omitted",
        "content.breadth_contrast.all",
        "content.breadth_contrast.all_length_adjusted",
        "content.breadth_contrast.rewarded",
        "content.breadth_contrast.omitted",
        "content.fig.swr_contrast_coefs",
        "content.fig.ladder",
    ]


def test_no_dependents_selects_only_what_is_named():
    names = [r.name for r in rr.select(["content.arm_ladder.all_arms"], False)]
    assert names == ["content.arm_ladder.all_arms"]


def test_a_prefix_selects_every_run_under_it():
    names = {r.name for r in rr.select(["behavior.retrospective"], False)}
    assert names == {f"behavior.retrospective.{s}" for s in ("all", "swr", "non_swr")}
    assert {r.question for r in rr.select(["rate"], False)} == {"rate"}


def test_a_prefix_must_end_at_a_dot():
    with pytest.raises(SystemExit):
        rr.select(["rate.per"])


def test_the_environment_caps_threads_and_keeps_jax_on_cpu():
    env = rr.run_env(rr.Run("rate.t", ("t.py",), cores=4, threads=3))
    assert {env[v] for v in rr.THREAD_VARS} == {"3"}
    assert env["JAX_PLATFORMS"] == "cpu" and env["CUDA_VISIBLE_DEVICES"] == ""
    assert "intra_op_parallelism_threads=3" in env["XLA_FLAGS"]


def _py(code):
    return (sys.executable, "-c", code)


def test_a_failed_run_blocks_its_dependents_and_nothing_else(tmp_path):
    out = tmp_path / "threads.txt"
    write_threads = (
        f"import os; open({str(out)!r}, 'w').write(os.environ['OMP_NUM_THREADS'])"
    )
    runs = [
        rr.Run("rate.a", _py(write_threads), 1, threads=2),
        rr.Run("rate.b", _py("raise SystemExit(3)"), 1),
        rr.Run("rate.c", _py("pass"), 1, deps=("rate.b",)),
        # rate.z is not selected, so it is taken as done.
        rr.Run("rate.d", _py("pass"), 1, deps=("rate.a", "rate.z")),
        rr.Run("rate.e", _py("pass"), 1, deps=("rate.c",)),
    ]
    logs = tmp_path / "logs"
    codes = rr.schedule(runs, budget=2, log_dir=logs, prefix=[], poll_s=0.05)
    assert codes == {
        "rate.a": 0,
        "rate.b": 3,
        "rate.c": None,
        "rate.d": 0,
        "rate.e": None,
    }
    assert out.read_text() == "2"
    record = json.loads((logs / "rate.b.json").read_text())
    assert record["exit_code"] == 3 and record["argv"][-1] == "raise SystemExit(3)"
    assert record["question"] == rr.QUESTIONS["rate"]
    assert not (logs / "rate.c.json").exists()


def test_runs_wait_for_cores_and_a_run_larger_than_the_budget_still_runs(tmp_path):
    stamp = "import time; print(time.time()); time.sleep(0.3); print(time.time())"
    runs = [rr.Run(f"rate.{i}", _py(stamp), 2) for i in ("a", "b")]
    runs.append(rr.Run("rate.c", _py("pass"), 9))
    logs = tmp_path / "logs"
    codes = rr.schedule(runs, budget=3, log_dir=logs, prefix=[], poll_s=0.02)
    assert codes == {"rate.a": 0, "rate.b": 0, "rate.c": 0}
    a_end = float((logs / "rate.a.log").read_text().split()[1])
    b_start = float((logs / "rate.b.log").read_text().split()[0])
    assert b_start >= a_end
