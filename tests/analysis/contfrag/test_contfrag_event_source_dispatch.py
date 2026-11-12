"""Regression test for strict ``event_source`` dispatch in contfrag tables.

The previous form ``"SWR_event" if event_source == "SWR" else "MUA_event"``
silently routed any non-``"SWR"`` value (including typos like
``"swr"`` / ``"Swr"`` / ``"ripple"``) to the MUA column, so direct
calls into the public table builders could quietly produce a table
keyed against the wrong event source. The ``_resolve_event_col``
helper raises ``ValueError`` for anything outside ``{"SWR", "MUA"}``.
"""

from __future__ import annotations

import pytest

from continuum_swr_replay.analysis.contfrag.tables import _resolve_event_col


class TestResolveEventCol:
    def test_swr_resolves_to_swr_event_column(self):
        assert _resolve_event_col("SWR") == "SWR_event"

    def test_mua_resolves_to_mua_event_column(self):
        assert _resolve_event_col("MUA") == "MUA_event"

    @pytest.mark.parametrize("bad", ["swr", "Swr", "mua", "ripple", "", "SWR "])
    def test_typos_raise_value_error(self, bad):
        with pytest.raises(ValueError, match="event_source must be"):
            _resolve_event_col(bad)
