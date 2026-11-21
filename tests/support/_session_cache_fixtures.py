"""Stand-in session-cache entries, for tests of a producer that records the entries
it read the well visits from (``prepare_data.session_cache_record``)."""

from __future__ import annotations

from pathlib import Path

from continuum_swr_replay.analysis.event_tables.prepare_data import (
    behavioral_context_cache_path,
)


def write_session_cache_entries(
    identifiers: list[str], content: bytes = b"the epoch's visits"
) -> list[Path]:
    """One entry per epoch, holding ``content``, where the cache is read; the
    paths. The visits themselves are stubbed by the test (the entry is not a
    pickle)."""
    paths = []
    for identifier in identifiers:
        path = behavioral_context_cache_path({"session_id": identifier})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content + identifier.encode())
        paths.append(path)
    return paths
