"""What every part of the behavior summary shares: the tags of its values,
the note a value read through a refusing per-term gate carries, and the
checks that refuse a build."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from continuum_swr_replay.analysis.replay_models.contrast_aggregation import (
    MAX_UNIDENTIFIED_SHARE,
    MIN_SESSIONS_PER_TERM,
)
from continuum_swr_replay.data_loaders.epoch_register import analyzed_animals
from continuum_swr_replay.reported.formatting import format_number
from continuum_swr_replay.reported.values import (
    Tags,
)

TAGS = {
    "rate": Tags(("3.3",)),
    "anchoring": Tags(("3.1",)),
    "estimators": Tags(("3.1", "3.6", "3.7"), ("content_estimators",)),
    "estimators_anchoring": Tags(("3.1",), ("content_estimators",)),
    "leaf_ladder": Tags(("3.7",), ("uncertainty_content",)),
    "choice": Tags(("3.4",)),
    "stay_leave": Tags(("3.5",)),
    "stay_leave_rate": Tags(("3.5", "3.3")),
    "ripple_power": Tags(("3.8",)),
    "content_reward": Tags(("3.2",)),
    "content_value": Tags(("3.6",)),
    "content_uncertainty": Tags(("3.7",)),
    "content_reward_value": Tags(("3.2", "3.6")),
    "content_reward_uncertainty": Tags(("3.2", "3.7")),
}
"""Each group of values: the questions it answers and the figures that show it."""


def _gate_note(statistics: Mapping[str, Any], term: str) -> str | None:
    """Why the values read from a gated statistic are pending, or None.

    The per-term session gate refuses a term when it keeps too few sessions or
    leaves out too large a share of them
    (:attr:`~continuum_swr_replay.analysis.replay_models.contrast_aggregation.TermSessionGate.refusal`),
    and the audited aggregates publish no row for such a term. The values read
    here over the kept sessions are then kept and marked pending.
    """
    if not statistics["gate_refuses"]:
        return None
    return (
        f"the per-term session gate refuses {term}: it keeps fewer than "
        f"{MIN_SESSIONS_PER_TERM} sessions, or leaves out more than "
        f"{format_number(MAX_UNIDENTIFIED_SHARE, 'pct')} of the sessions with a fit"
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_every_animal(n_animals: float, what: str) -> None:
    """Refuse a statistic that does not span every analyzed animal."""
    expected = len(analyzed_animals())
    _require(
        n_animals == expected,
        f"{what}: {n_animals} animals, not the {expected} analyzed",
    )
