"""The duration-aware history runner keeps its results in job order."""

from __future__ import annotations

import time

from scripts.behavior.rate import fit_duration_aware_history_rate as mod


def _job(index: int, delay_s: float) -> tuple[tuple, int, float]:
    time.sleep(delay_s)
    return (index,), 10 * index, delay_s


def test_parallel_results_come_back_in_job_order_not_completion_order():
    """The first job finishes last; its result still comes first."""
    jobs = [(0, 0.5), (1, 0.0), (2, 0.0)]

    assert mod._run(_job, jobs, workers=3) == [0, 10, 20]
    assert mod._run(_job, jobs, workers=1) == [0, 10, 20]
