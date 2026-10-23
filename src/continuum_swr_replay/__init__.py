"""Continuum SWR Replay Analysis Package.

This package provides tools for analyzing neural replay during sharp-wave ripples
in a spatial bandit task. It includes data loading, decoding algorithms, and
visualization utilities.

The public re-exports below are loaded **lazily** via PEP 562
``__getattr__``. Importing a leaf module
(e.g. ``continuum_swr_replay.analysis.replay.fitting``) used to drag in
``data_loaders.bandit_task`` here, which itself does
``from spyglass.common import PositionIntervalMap`` at module top level
and opens a DataJoint connection at import time. That made every
"offline" unit test fragile against a missing VPN. Lazy loading keeps
``from continuum_swr_replay import load_data`` working while letting
``import continuum_swr_replay.analysis.replay.fitting`` finish without
touching Spyglass.
"""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING

# This runs on a shared GPU host. JAX's default is to preallocate ~75% of every
# card it can see the first time it is used, which on a box carrying other
# people's jobs is not a default anyone can afford; a run that wants a card
# should take what it needs and no more. Set here rather than in each entry
# point because it has to be in place before JAX chooses a backend, and it is
# only a default -- a caller that means to preallocate still can.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "load_data": "continuum_swr_replay.data_loaders",
    "get_epoch_info": "continuum_swr_replay.data_loaders",
    "get_training_timepoints": "continuum_swr_replay.data_loaders",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target)
    attr = getattr(module, name)
    globals()[name] = attr
    return attr


if TYPE_CHECKING:  # pragma: no cover - types only
    from .data_loaders import (  # noqa: F401
        get_epoch_info,
        get_training_timepoints,
        load_data,
    )
