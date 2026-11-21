#!/usr/bin/env python
"""Coverage of the animal-balanced interval under three ways to resample.

The estimate the convention reports is the mean over animals of each animal's
mean over its recording days of each day's mean over its session values, with
the t-corrected 95 % interval of ``shared_statistics.bootstrap_interval``. This
simulation measures how often that interval covers the truth when its draws
come from

- ``animals_sessions``: animals, then each drawn animal's sessions, the
  estimate recomputed session -> day -> animal on the sessions drawn (the
  convention's rule, ``shared_statistics.animal_balanced_bootstrap``);
- ``animals_days_sessions``: animals, then each drawn animal's days, then each
  drawn day's sessions (``nested_bootstrap_draws`` with the days as its
  sessions; the rejected third level, computed here);
- ``animals``: animals only, each drawn animal keeping its own value.

The cohort's shape: 5 animals of 6, 8, 9, 10 and 12 recording days, 4 to 7
sessions a day (drawn once, seed 0: 45 days, 244 sessions). A session's value
is ``animal effect + day effect + session noise`` (normal, truth 0) at the
standard deviations of :data:`SETTINGS`, one setting with 15 % of the session
values missing at random, and two paired kind differences: two kinds measured
in the same sessions whose animal and day effects correlate 0.5 (session noise
independent), the value being their difference.

Writes a CSV, one row per setting and scheme: ``coverage``, ``mean_width``,
``coverage_se`` (the simulation's binomial standard error of the coverage),
``n_sets`` and ``n_draws``::

    python scripts/checks/simulate_interval_coverage.py --out coverage.csv
    python scripts/checks/simulate_interval_coverage.py --out coverage.csv --quick

The full run (2,000 data sets of 1,000 draws per setting) takes about 16 min on
one core; ``--quick`` runs 40 data sets of 200 draws.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.shared_statistics import (
    animal_balanced_bootstrap,
    bootstrap_interval,
    nested_bootstrap_draws,
    nested_codes,
)

DAYS_PER_ANIMAL = (6, 8, 9, 10, 12)
SESSIONS_PER_DAY = (4, 7)  # inclusive range
STRUCTURE_SEED = 0
SCHEMES = ("animals_sessions", "animals_days_sessions", "animals")


@dataclass(frozen=True)
class Setting:
    """One simulated setting: the effects' SDs (animal, day, session)."""

    sds: tuple[float, float, float]
    seed: int
    missing: float = 0.0
    paired: bool = False

    @property
    def name(self) -> str:
        label = "sd {:g}/{:g}/{:g}".format(*self.sds)
        if self.paired:
            label += ", paired difference"
        if self.missing:
            label += f", {self.missing:.0%} missing"
        return label


SETTINGS = (
    Setting((1.0, 0.0, 1.0), 101),
    Setting((1.0, 0.5, 1.0), 102),
    Setting((1.0, 1.0, 1.0), 103),
    Setting((0.3, 1.0, 1.0), 104),
    Setting((0.0, 1.0, 1.0), 105),
    Setting((0.0, 0.0, 1.0), 106),
    Setting((1.0, 1.0, 0.3), 107),
    Setting((1.0, 1.0, 1.0), 108, missing=0.15),
    Setting((1.0, 0.5, 1.0), 201, paired=True),
    Setting((0.0, 1.0, 1.0), 202, paired=True),
)


def cohort() -> tuple[np.ndarray, np.ndarray]:
    """Each session's animal and (cohort-wide) day code, shape (n_sessions,)."""
    rng = np.random.default_rng(STRUCTURE_SEED)
    animal, day, first_day = [], [], 0
    for a, n_days in enumerate(DAYS_PER_ANIMAL):
        for d in range(n_days):
            n = int(rng.integers(SESSIONS_PER_DAY[0], SESSIONS_PER_DAY[1] + 1))
            animal += [a] * n
            day += [first_day + d] * n
        first_day += n_days
    return np.array(animal), np.array(day)


def _animal_day_session_weights(
    animal: np.ndarray, day: np.ndarray, rng: np.random.Generator, n_draws: int
) -> np.ndarray:
    """Weights of the mean of means when animals, days and sessions are all
    resampled, shape (n_draws, n_sessions)."""
    draws = nested_bootstrap_draws(animal, day, rng, n_draws)
    n_animals, n_days = draws.animals.shape[1], draws.sessions.shape[1]
    day_animal = np.zeros(n_days, dtype=np.int64)
    day_animal[day] = animal
    day_total = np.stack([np.bincount(day, r, n_days) for r in draws.rows])
    animal_total = np.stack(
        [np.bincount(day_animal, d, n_animals) for d in draws.sessions]
    )
    within_day = np.divide(
        draws.rows,
        day_total[:, day],
        out=np.zeros(draws.rows.shape),
        where=draws.rows > 0,
    )
    within_animal = np.divide(
        draws.sessions,
        animal_total[:, day_animal],
        out=np.zeros(draws.sessions.shape),
        where=draws.sessions > 0,
    )
    return draws.animals[:, animal] / n_animals * within_animal[:, day] * within_day


def _animal_values(
    values: np.ndarray, animal: np.ndarray, day: np.ndarray
) -> np.ndarray:
    """Each animal's mean over its days of each day's mean, shape (n_animals,)."""
    day_mean = np.bincount(day, values) / np.bincount(day)
    day_animal = np.zeros(day_mean.size, dtype=np.int64)
    day_animal[day] = animal
    return np.bincount(day_animal, day_mean) / np.bincount(day_animal)


def intervals(
    values: np.ndarray,
    animal: np.ndarray,
    day: np.ndarray,
    rng: np.random.Generator,
    n_draws: int,
) -> dict[str, tuple[float, float]]:
    """The 95 % interval of each scheme for one data set."""
    keep = np.isfinite(values)
    values, animal, day = values[keep], animal[keep], day[keep]
    animal, day = nested_codes(animal, np.unique(day, return_inverse=True)[1])
    n_animals = int(animal.max()) + 1
    convention = animal_balanced_bootstrap(
        values, animal, day, n_draws=n_draws, seed=int(rng.integers(2**31))
    )
    estimate = float(convention.estimate)
    third_level = _animal_day_session_weights(animal, day, rng, n_draws) @ values
    animal_draws = rng.multinomial(
        n_animals, np.full(n_animals, 1 / n_animals), n_draws
    )
    by_animal = animal_draws @ _animal_values(values, animal, day) / n_animals
    out = {}
    for scheme, draws in zip(
        SCHEMES, (convention.draws, third_level, by_animal), strict=True
    ):
        interval = bootstrap_interval(estimate, draws, n_animals)
        out[scheme] = (interval.low, interval.high)
    return out


def simulate(setting: Setting, n_sets: int, n_draws: int) -> pd.DataFrame:
    """Coverage and mean width of each scheme over ``n_sets`` data sets."""
    animal, day = cohort()
    n_animals, n_days = animal.max() + 1, day.max() + 1
    sd_animal, sd_day, sd_session = setting.sds
    rng = np.random.default_rng(setting.seed)
    covered = {s: [] for s in SCHEMES}
    width = {s: [] for s in SCHEMES}
    shared = np.array([[1.0, 0.5], [0.5, 1.0]])
    for _ in range(n_sets):
        if setting.paired:
            kinds = (
                rng.multivariate_normal([0, 0], shared, n_animals)[animal] * sd_animal
                + rng.multivariate_normal([0, 0], shared, n_days)[day] * sd_day
                + rng.normal(0.0, sd_session, (animal.size, 2))
            )
            values = kinds[:, 0] - kinds[:, 1]
        else:
            values = (
                rng.normal(0.0, sd_animal, n_animals)[animal]
                + rng.normal(0.0, sd_day, n_days)[day]
                + rng.normal(0.0, sd_session, animal.size)
            )
        if setting.missing:
            values[rng.random(animal.size) < setting.missing] = np.nan
        for scheme, (low, high) in intervals(values, animal, day, rng, n_draws).items():
            covered[scheme].append(low <= 0.0 <= high)
            width[scheme].append(high - low)
    rows = []
    for scheme in SCHEMES:
        coverage = float(np.mean(covered[scheme]))
        rows.append(
            {
                "setting": setting.name,
                "scheme": scheme,
                "coverage": coverage,
                "mean_width": float(np.mean(width[scheme])),
                "coverage_se": float(np.sqrt(coverage * (1 - coverage) / n_sets)),
                "n_sets": n_sets,
                "n_draws": n_draws,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="the CSV to write")
    parser.add_argument(
        "--quick", action="store_true", help="40 data sets of 200 draws per setting"
    )
    args = parser.parse_args(argv)
    n_sets, n_draws = (40, 200) if args.quick else (2000, 1000)
    table = pd.concat(
        [simulate(setting, n_sets, n_draws) for setting in SETTINGS], ignore_index=True
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
