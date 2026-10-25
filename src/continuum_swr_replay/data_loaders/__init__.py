"""Data loading module for the spatial bandit task.

Loader functions (``load_data``, ``get_epoch_info``,
``get_training_timepoints``, ``get_labels``) are exposed here via PEP 562
``__getattr__``, so callers who only need the leaf-level constants
(``TRACK_SEGMENT_TO_PATCH`` etc., consumed by analysis modules) import
this package without importing the loader modules. The loaders import
Spyglass, which opens a DataJoint connection, only inside the functions
that read through it, so importing a loader module needs no database.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .constants import (
    ALL_ARMS,
    NWB_FILES,
    TRACK_SEGMENT_TO_PATCH,
    WELL_PATCH_MAPPING,
)

_LAZY_EXPORTS = {
    "load_data": "continuum_swr_replay.data_loaders.bandit_task",
    "get_epoch_info": "continuum_swr_replay.data_loaders.utils",
    "get_training_timepoints": "continuum_swr_replay.data_loaders.utils",
    "get_labels": "continuum_swr_replay.data_loaders.utils",
}

__all__ = [
    "ALL_ARMS",
    "NWB_FILES",
    "TRACK_SEGMENT_TO_PATCH",
    "WELL_PATCH_MAPPING",
    *sorted(_LAZY_EXPORTS),
]


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target)
    attr = getattr(module, name)
    globals()[name] = attr
    return attr


if TYPE_CHECKING:  # pragma: no cover - types only
    from .bandit_task import load_data  # noqa: F401
    from .utils import (  # noqa: F401
        get_epoch_info,
        get_labels,
        get_training_timepoints,
    )
