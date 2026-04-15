"""Task-agnostic replay content analysis package.

Houses the conditional Poisson fitter, model hierarchy, spatial mapping,
simulation utilities, and model comparison/checking helpers used across
the replay content analysis pipeline. Independent of brain area —
designed to be reusable for any decoder + track graph combination that
produces non-local events on a graph of arms.
"""

from .spatial import arm_dist_rank, dist_to_arm_from_well
from .tables import build_event_table

__all__ = [
    "arm_dist_rank",
    "build_event_table",
    "dist_to_arm_from_well",
]
