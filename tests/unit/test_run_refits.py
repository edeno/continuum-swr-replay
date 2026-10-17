"""The refit driver: its run list, what a selection adds, and how runs are scheduled."""

import json
import os
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
        "content.fig.pipeline_data_flow",
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


def _py(code):
    return (sys.executable, "-c", code)


def _cpus(n):
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < n:
        pytest.skip(f"needs {n} CPUs")
    return cpus[:n]


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
    codes = rr.schedule(runs, _cpus(2), log_dir=logs, prefix=[], poll_s=0.05)
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
    assert record["env"]["OMP_NUM_THREADS"] == "1"
    blocked = {n: json.loads((logs / f"rate.{n}.json").read_text()) for n in "ce"}
    assert blocked["c"]["status"] == "blocked" and blocked["c"]["blocked_by"] == [
        "rate.b"
    ]
    assert blocked["e"]["blocked_by"] == ["rate.c"]


def test_each_run_is_pinned_to_its_own_cpus(tmp_path):
    cpus = _cpus(3)
    show = "import os, time; print(sorted(os.sched_getaffinity(0))); time.sleep(0.3)"
    runs = [rr.Run("rate.a", _py(show), 2), rr.Run("rate.b", _py(show), 1)]
    logs = tmp_path / "logs"
    rr.schedule(runs, cpus, log_dir=logs, prefix=[], poll_s=0.02)
    a = json.loads((logs / "rate.a.log").read_text())
    b = json.loads((logs / "rate.b.log").read_text())
    assert len(a) == 2 and len(b) == 1 and not set(a) & set(b)
    assert set(a) | set(b) == set(cpus)


def test_runs_wait_for_cpus_and_a_run_larger_than_the_list_still_runs(tmp_path):
    stamp = "import time; print(time.time()); time.sleep(0.3); print(time.time())"
    runs = [rr.Run(f"rate.{i}", _py(stamp), 2) for i in ("a", "b")]
    runs.append(
        rr.Run("rate.c", _py("import os; print(len(os.sched_getaffinity(0)))"), 9)
    )
    logs = tmp_path / "logs"
    codes = rr.schedule(runs, _cpus(3), log_dir=logs, prefix=[], poll_s=0.02)
    assert codes == {"rate.a": 0, "rate.b": 0, "rate.c": 0}
    a_end = float((logs / "rate.a.log").read_text().split()[1])
    b_start = float((logs / "rate.b.log").read_text().split()[0])
    assert b_start >= a_end
    assert (logs / "rate.c.log").read_text().strip() == "3"


def test_only_artifacts_whose_sidecar_changed_are_checked(tmp_path):
    def fit(path, ext=".pkl"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(ext).write_bytes(b"")
        path.with_suffix(".provenance.json").write_text("{}")

    fit(tmp_path / "models_dm" / "M0")  # untouched by the runs
    fit(tmp_path / "models" / "C1")  # rewritten by the runs
    before = rr.sidecar_times(tmp_path)
    os.utime(tmp_path / "models" / "C1.provenance.json", (1, 1))
    fit(tmp_path / "_backup_x" / "C1")  # backups are skipped
    fit(tmp_path / "stage6", ".parquet")  # new, at the root
    # A z-reference file sits beside its fit and has no sidecar of its own.
    fit(tmp_path / "models" / "K1")
    (tmp_path / "models" / "K1.z_reference.json").write_text("{}")
    found = rr.written_artifacts(tmp_path, before)
    assert sorted(p.relative_to(tmp_path).as_posix() for p in found) == [
        "models/C1.pkl",
        "models/K1.pkl",
        "stage6.parquet",
    ]


def test_optional_runs_are_left_out_unless_asked_for():
    optional = {r.name for r in rr.RUNS if r.optional}
    assert optional == {
        "rate.per_bin.in_transit",
        "behavior.stay_leave.all_stops",
        "behavior.stay_leave.all_stops_swr",
        "behavior.stay_leave.all_stops_non_swr",
        "behavior.stay_leave.all_stops_mua",
    }
    assert not optional & {r.name for r in rr.select(None)}
    assert not optional & {r.name for r in rr.select(["behavior.stay_leave"])}
    assert optional <= {r.name for r in rr.select(None, include_optional=True)}
    # Named in full, an optional run comes with the optional runs that need it.
    named = {r.name for r in rr.select(["behavior.stay_leave.all_stops"])}
    assert named == optional - {"rate.per_bin.in_transit"}
