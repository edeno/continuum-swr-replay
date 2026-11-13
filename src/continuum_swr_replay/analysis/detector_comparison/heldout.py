"""Held-out neural evidence for a detected event: does its content reproduce?

Events are detected from discovery cells; their content is then assessed with
evaluation cells that took no part in detection, decoding or tuning. The rule
is one and the same whichever method found the event: over the event's own
interval, on the decode grid, take a content posterior estimated from the
discovery cells and ask how well it predicts the evaluation cells' spikes,

    score = sum_t sum_x p_t(x) sum_c [k_tc log(f_c(x)) - f_c(x)],

with ``f`` each evaluation cell's place field per sample of the grid and ``k``
its spike count. The first term is the spikes' evidence, the second the
silence's: an evaluation cell silent where the content says it should fire
counts against the content. The two are kept apart, so a verdict driven by
silence alone can be seen.

The score is read against two nulls. Against the animal's own position over
the same bins, it asks whether the event beats local coding; against the
evaluation fields permuted among the evaluation cells, whether it beats chance.
An event has held-out support when both hold; it is uninformative, rather than
unsupported, when the evaluation cells are expected to fire too little along
its content to say either way.

Held-out support tests spatial information, not sequence structure: activity
that represents places without a sequence also has it. Its absence does not
make an event false when evaluation coverage is poor, and its presence says
nothing about the events no method detected.

:func:`heldout_summary` gives the support and error rates of the held-out check
on simulated data (``scripts/detector_comparison/sweep_memoryless_heldout.py``).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "COVERAGE_FLOOR",
    "COVERAGE_SENSITIVITY",
    "SUPPORT_LEVEL",
    "HeldoutScore",
    "event_classes",
    "expected_log_likelihood",
    "heldout_score",
    "heldout_summary",
    "interleaved_folds",
    "nearest_window_content",
    "poor_coverage_fold",
]

SUPPORT_LEVEL = 0.05
"""Largest shuffle p-value at which an event has held-out support."""

COVERAGE_FLOOR = 5.0
"""Fewest expected evaluation spikes along an event's content for its score to
count; 2 and 10 are the sensitivity."""

COVERAGE_SENSITIVITY = (2.0, COVERAGE_FLOOR, 10.0)
"""The coverage floors the held-out summary is given at: the rule's own
(:data:`COVERAGE_FLOOR`, the unsuffixed measures) and two others (the
``_floor<f>`` measures)."""


def interleaved_folds(n_cells: int) -> dict[str, tuple[NDArray, NDArray]]:
    """Two folds, alternate cells by field position, the roles swapped.

    Cells are assumed ordered by field position, so each half tiles the track.

    Returns
    -------
    folds : dict
        ``{"A": (discovery, evaluation), "B": (discovery, evaluation)}``.
    """
    even, odd = np.arange(0, n_cells, 2), np.arange(1, n_cells, 2)
    return {"A": (odd, even), "B": (even, odd)}


def poor_coverage_fold(n_cells: int) -> dict[str, tuple[NDArray, NDArray]]:
    """One fold whose evaluation cells cover only the lower half of the track.

    The evaluation cells are the even cells in the lower half; discovery keeps
    every other cell, so it still covers the whole track.
    """
    even = np.arange(0, n_cells, 2)
    evaluation = even[even < n_cells // 2]
    return {"C": (np.setdiff1d(np.arange(n_cells), evaluation), evaluation)}


def expected_log_likelihood(
    content: NDArray, counts: NDArray, fields: NDArray
) -> tuple[float, float]:
    """The evaluation spikes' and silence's expected log-likelihood under a content.

    Parameters
    ----------
    content : ndarray, shape (n_time, n_bins)
        Each row a posterior over place, summing to one.
    counts : ndarray, shape (n_time, n_cells)
        Evaluation spikes per bin of the grid.
    fields : ndarray, shape (n_cells, n_bins)
        Each evaluation cell's rate per sample of the grid; positive.

    Returns
    -------
    spike_term, silence_term : float

    """
    content = np.asarray(content, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    fields = np.asarray(fields, dtype=np.float64)
    spike_term = float(np.sum(content * (counts @ np.log(fields))))
    silence_term = -float(np.sum(content @ fields.sum(axis=0)))
    return spike_term, silence_term


@dataclass(frozen=True)
class HeldoutScore:
    """How well an event's discovery content predicts the evaluation cells.

    Attributes
    ----------
    spike_term, silence_term : float
        The two parts of the score under the content.
    gain_vs_animal : float
        Score minus its value under the animal's own position, per evaluation
        spike (per one when there is none).
    p_shuffle : float
        ``(1 + number of permutations scoring at least the observed) /
        (1 + number of permutations)``.
    coverage : float
        Evaluation spikes the content expects over the event.
    n_spikes : int
        Evaluation spikes observed over the event.
    """

    spike_term: float
    silence_term: float
    gain_vs_animal: float
    p_shuffle: float
    coverage: float
    n_spikes: int

    @property
    def supported(self) -> bool:
        """Beats both the animal's own position and the permuted fields."""
        return bool(self.p_shuffle <= SUPPORT_LEVEL and self.gain_vs_animal > 0)

    def informative(self, coverage_floor: float = COVERAGE_FLOOR) -> bool:
        """The evaluation cells are expected to fire enough to say either way."""
        return bool(self.coverage >= coverage_floor)


def heldout_score(
    content: NDArray,
    counts: NDArray,
    fields: NDArray,
    animal_bin: NDArray,
    permutations: list[NDArray],
) -> HeldoutScore:
    """Score one event's content on the evaluation cells, against both nulls.

    Parameters
    ----------
    content : ndarray, shape (n_time, n_bins)
    counts : ndarray, shape (n_time, n_cells)
    fields : ndarray, shape (n_cells, n_bins)
    animal_bin : ndarray of int, shape (n_time,)
        The bin of the animal's own position at each time.
    permutations : list of ndarray of int, each shape (n_cells,)
        Cell ``i`` is scored with the field of cell ``permutation[i]``.

    Returns
    -------
    score : HeldoutScore

    """
    content = np.asarray(content, dtype=np.float64)
    fields = np.asarray(fields, dtype=np.float64)
    spike_term, silence_term = expected_log_likelihood(content, counts, fields)
    observed = spike_term + silence_term

    local = np.zeros_like(content)
    local[np.arange(content.shape[0]), np.asarray(animal_bin, dtype=int)] = 1.0
    animal = sum(expected_log_likelihood(local, counts, fields))

    null = np.array(
        [
            sum(expected_log_likelihood(content, counts, fields[permutation]))
            for permutation in permutations
        ]
    )
    n_spikes = int(np.asarray(counts).sum())
    return HeldoutScore(
        spike_term=spike_term,
        silence_term=silence_term,
        gain_vs_animal=(observed - animal) / max(n_spikes, 1),
        p_shuffle=float((1 + np.sum(null >= observed)) / (1 + len(permutations))),
        coverage=-silence_term,
        n_spikes=n_spikes,
    )


def nearest_window_content(
    window_posterior: NDArray, midpoint_s: NDArray, time: NDArray
) -> NDArray:
    """A window-by-window posterior on the decode grid: each bin takes its nearest window.

    Parameters
    ----------
    window_posterior : ndarray, shape (n_windows, n_bins)
    midpoint_s : ndarray, shape (n_windows,)
        Increasing.
    time : ndarray, shape (n_time,)

    Returns
    -------
    content : ndarray, shape (n_time, n_bins)

    """
    midpoint_s = np.asarray(midpoint_s, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    after = np.clip(np.searchsorted(midpoint_s, time), 0, midpoint_s.size - 1)
    before = np.clip(after - 1, 0, midpoint_s.size - 1)
    nearest = np.where(
        np.abs(midpoint_s[before] - time) <= np.abs(midpoint_s[after] - time),
        before,
        after,
    )
    return np.asarray(window_posterior)[nearest]


def event_classes(
    ours: list[tuple[float, float]], theirs: list[tuple[float, float]]
) -> tuple[list[str], list[str]]:
    """Which of two methods' events the other also found, by strictly positive overlap.

    Returns
    -------
    ours_classes, theirs_classes : list of str
        ``"shared"`` for an event overlapping some event of the other method
        for a positive time, else ``"ours_only"`` or ``"comparator_only"``.
    """

    def overlapping(event, others) -> bool:
        return any(
            min(event[1], other[1]) - max(event[0], other[0]) > 0 for other in others
        )

    return (
        ["shared" if overlapping(event, theirs) else "ours_only" for event in ours],
        [
            "shared" if overlapping(event, ours) else "comparator_only"
            for event in theirs
        ],
    )


def heldout_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Held-out support rates by fold, estimator, class and truth, and the error rates.

    Parameters
    ----------
    table : pd.DataFrame, shape (n_rows, n_columns)
        One row per reported event and content estimator: ``fold``,
        ``trial``, ``detector``, ``event``, ``event_class``, ``truth``,
        ``estimator``, ``coverage``, ``gain_vs_animal`` and ``supported``.

    Returns
    -------
    summary : pd.DataFrame, shape (n_values, 9)
        ``summary`` (``by_class``, ``error_rates`` or ``agreement``),
        ``measure``, ``fold``, ``estimator``, ``detector``, ``event_class``,
        ``truth``, ``value`` and an empty ``note``. An event is informative
        when its coverage reaches the floor (each of
        :data:`COVERAGE_SENSITIVITY`); a rate over no informative event is
        NaN. The estimators' agreement is over the events informative under
        both.
    """
    rows = []

    def add(summary, measure, fold, estimator, detector, event_class, truth, value):
        rows.append(
            {
                "summary": summary,
                "measure": measure,
                "fold": fold,
                "estimator": estimator,
                "detector": detector,
                "event_class": event_class,
                "truth": truth,
                "value": float(value),
            }
        )

    keys = ["fold", "estimator", "detector"]
    for (fold, estimator, detector, event_class, truth), group in table.groupby(
        [*keys, "event_class", "truth"]
    ):
        where = (fold, estimator, detector, event_class, truth)
        add("by_class", "n_events_all", *where, len(group))
        add(
            "by_class",
            "informative_share",
            *where,
            (group.coverage >= COVERAGE_FLOOR).mean(),
        )
        add("by_class", "median_gain", *where, group.gain_vs_animal.median())
        add("by_class", "median_coverage", *where, group.coverage.median())
        for floor in COVERAGE_SENSITIVITY:
            suffix = "" if floor == COVERAGE_FLOOR else f"_floor{floor:g}"
            informative = group[group.coverage >= floor]
            add("by_class", f"n_events{suffix}", *where, len(informative))
            add(
                "by_class",
                f"support_rate{suffix}",
                *where,
                informative.supported.mean() if len(informative) else np.nan,
            )

    kinds = {
        "false_support_rate": ["local", "no_content", "silent_gap"],
        "support_rate_with_content": ["continuous_replay", "non_local"],
        "support_rate_mixed": ["mixed"],
    }
    for (fold, estimator, detector), group in table.groupby(keys):
        for floor in COVERAGE_SENSITIVITY:
            suffix = "" if floor == COVERAGE_FLOOR else f"_floor{floor:g}"
            informative = group[group.coverage >= floor]
            for measure, truths in kinds.items():
                chosen = informative[informative.truth.isin(truths)]
                where = (fold, estimator, detector, "all", "")
                add("error_rates", f"n_{measure}{suffix}", *where, len(chosen))
                add(
                    "error_rates",
                    f"{measure}{suffix}",
                    *where,
                    chosen.supported.mean() if len(chosen) else np.nan,
                )
            with_content = informative[
                informative.truth.isin(kinds["support_rate_with_content"])
            ]
            add(
                "error_rates",
                f"under_crediting{suffix}",
                fold,
                estimator,
                detector,
                "all",
                "",
                1 - with_content.supported.mean() if len(with_content) else np.nan,
            )

    # the two estimators' verdicts on the same event, among events informative
    # under both
    verdicts = (
        table[table.coverage >= COVERAGE_FLOOR]
        .pivot_table(
            index=["fold", "trial", "detector", "event"],
            columns="estimator",
            values="supported",
            aggfunc="first",
        )
        .reindex(columns=["state_space", "sliding_window"])
        .dropna()
    )
    for fold, group in verdicts.groupby(level="fold"):
        add(
            "agreement",
            "estimator_agreement",
            fold,
            "both",
            "both",
            "all",
            "",
            (group["state_space"] == group["sliding_window"]).mean(),
        )
    summary = pd.DataFrame(rows)
    summary["note"] = ""
    return summary
