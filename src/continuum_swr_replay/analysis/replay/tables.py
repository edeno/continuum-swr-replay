"""Canonical replay-content table builders, split by grain.

This module is a thin compatibility shim: the builders now live in per-grain
modules so each answers one question, and every public name is re-exported here
so existing ``from ...tables import build_*`` imports keep working unchanged.

- event-grain (event, event-sequence): ``tables_event``
- stop-grain (stop, SWR split): ``tables_stop``
- arm- and patch-grain: ``tables_arm_patch``
- prep helpers, standardized-column list, and column orders shared across
  grains: ``_tables_common``
"""

from __future__ import annotations

from ._tables_common import (
    _EVENT_ARM_TABLE_BUILD_COLUMNS,
    _EVENT_ARM_TABLE_COLUMNS,
    _EVENT_SEQUENCE_TABLE_BUILD_COLUMNS,
    _EVENT_SEQUENCE_TABLE_COLUMNS,
    _EVENT_TABLE_BUILD_COLUMNS,
    _EVENT_TABLE_COLUMNS,
    _STOP_TABLE_BUILD_COLUMNS,
    _STOP_TABLE_COLUMNS,
    EVENT_ARM_STANDARDIZED_COLS,
)
from .tables_arm_patch import (
    build_event_arm_table,
    build_event_patch_table,
)
from .tables_event import (
    build_event_sequence_table,
    build_event_table,
)
from .tables_stop import (
    add_nonlocal_swr_split,
    build_stop_table,
)

__all__ = [
    "EVENT_ARM_STANDARDIZED_COLS",
    "build_event_table",
    "build_event_sequence_table",
    "build_stop_table",
    "add_nonlocal_swr_split",
    "build_event_arm_table",
    "build_event_patch_table",
    # Column-order constants re-exported for empty-session short-circuits and
    # the parity/provenance tests that pin table shapes.
    "_EVENT_TABLE_BUILD_COLUMNS",
    "_EVENT_TABLE_COLUMNS",
    "_STOP_TABLE_BUILD_COLUMNS",
    "_STOP_TABLE_COLUMNS",
    "_EVENT_ARM_TABLE_BUILD_COLUMNS",
    "_EVENT_ARM_TABLE_COLUMNS",
    "_EVENT_SEQUENCE_TABLE_BUILD_COLUMNS",
    "_EVENT_SEQUENCE_TABLE_COLUMNS",
]
