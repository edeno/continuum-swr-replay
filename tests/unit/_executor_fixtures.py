"""Stand-ins for a process pool, so a runner's fan-out runs in the test process."""

from __future__ import annotations

from concurrent.futures import Future

import numpy as np


class InlineExecutor:
    """``ProcessPoolExecutor`` stand-in: runs each task as it is submitted."""

    def __init__(self, max_workers):
        del max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, function, *args):
        future = Future()
        future.set_result(function(*args))
        return future


def completion_order(order: str):
    """An ``as_completed`` stand-in that yields the futures in another order.

    Parameters
    ----------
    order : {"reversed", "shuffled"}
        Reverse the submission order, or permute it with a fixed seed.
    """

    def as_completed(futures):
        futures = list(futures)
        if order == "reversed":
            return futures[::-1]
        permutation = np.random.default_rng(27).permutation(len(futures))
        return [futures[i] for i in permutation]

    return as_completed
