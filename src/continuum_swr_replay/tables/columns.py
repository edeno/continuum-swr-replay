"""The required-column check, for the code that reads the canonical tables.

The check itself lives in :mod:`continuum_swr_replay.columns`, a module with no
package imports, so that ``data_loaders`` (which ``tables`` imports) can call it
too.
"""

from continuum_swr_replay.columns import require_columns

__all__ = ["require_columns"]
