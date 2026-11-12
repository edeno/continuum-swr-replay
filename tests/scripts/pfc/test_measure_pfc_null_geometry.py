"""The null-geometry report's sections that read only the bundles' sidecars."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "measure_pfc_null_geometry",
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "pfc"
    / "measure_pfc_null_geometry.py",
)
geometry = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(geometry)


def _window(total: int, excluded: int) -> dict[str, int]:
    return {
        "n_events_total": total,
        "n_events_tested": total - excluded,
        "n_events_excluded_unmatchable": excluded,
    }


#: Per session: the engagement null's median and largest overlap with the
#: shipped control, and each class's events and those with no legal control.
_SESSIONS = {
    "s1": ((0.1, 0.5), {"SWR": _window(10, 1), "non_local": _window(0, 0)}),
    "s2": ((0.4, 0.45), {"SWR": _window(30, 3), "non_local": _window(20, 10)}),
    "s3": ((0.2, 0.9), {"non_local": _window(4, 3)}),
}


@pytest.fixture
def data_dir(tmp_path):
    for session, ((median, largest), windows) in _SESSIONS.items():
        bundle = tmp_path / f"{session}_mPFC_modulation.pkl"
        bundle.touch()
        bundle.with_suffix(".provenance.json").write_text(
            json.dumps(
                {
                    "engagement_null_overlap": {"median": median, "max": largest},
                    "null_windows": windows,
                }
            )
        )
    return tmp_path


def test_overlap_section_summarises_the_sessions_medians(data_dir):
    text, table = geometry.overlap_section(data_dir)
    assert list(table["session"]) == ["s1", "s2", "s3"]
    assert list(table["median_overlap_fraction"]) == [0.1, 0.4, 0.2]
    # Over sessions: the median of the medians is 0.2 and they range from 0.1
    # to 0.4; the worst session is the one with the largest median, s2, not the
    # one with the largest single draw (s3, 0.9).
    assert "| 3 | 0.2 | 0.1 | 0.4 | s2 | 0.9 |" in text


def test_excluded_section_pools_classes_and_names_the_worst_session(data_dir):
    text, table = geometry.excluded_section(data_dir)
    assert len(table) == 5
    # SWRs: 4 of 40 events over two sessions, at most 10 % in one; all
    # non-local events: 13 of 24 over three sessions, at most 3 of 4 in one
    assert "| SWR | 2 | 40 | 4 | 0.1 | 0.1 |" in text
    assert "| non_local | 3 | 24 | 13 | 0.75 | 0.5417 |" in text
    # the largest share, not the largest count (s2's 10 of 20); the session
    # with no event of the class has no share
    assert "s3 / non_local, which excludes 3 of 4 events (75.0 %)" in text
