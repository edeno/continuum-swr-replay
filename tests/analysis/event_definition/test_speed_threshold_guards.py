"""The speed threshold cannot split the speed cohorts.

The read (``analyzed_time.with_swr_source``) gates the SWR and MUA labels at the
canonical 4 cm/s, once for every consumer. A table builder that gated its own
events at another value would classify them against SWRs and MUA events gated
at 4 cm/s, so no builder takes a speed threshold: each uses the canonical
constant. The two filters that re-gate the read's SWRs themselves honor a
stricter value and refuse only a looser one, which could not re-admit the SWRs
the read dropped.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.contfrag.tables import (
    build_contfrag_event_arm_table,
    build_contfrag_event_table,
    build_contfrag_stop_table,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.event_table import (
    canonical_swr_intervals,
)
from continuum_swr_replay.analysis.event_definition.classified_table import (
    build_classified_nonlocal_event_table,
)
from continuum_swr_replay.analysis.event_definition.immobility import (
    filter_event_labels_by_speed,
    intervals_from_labels,
    require_canonical_speed_threshold,
)
from continuum_swr_replay.analysis.event_tables.arm_patch_tables import (
    build_event_arm_table,
)
from continuum_swr_replay.analysis.event_tables.event_table import (
    build_event_sequence_table,
    build_event_table,
)
from continuum_swr_replay.analysis.event_tables.stop_table import build_stop_table
from continuum_swr_replay.analysis.pfc.canonical_events import (
    load_session_swr_intervals,
)
from continuum_swr_replay.analysis.pfc.session import run_session
from continuum_swr_replay.parameters import DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC

CANONICAL = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC


@pytest.mark.parametrize(
    "builder",
    (
        build_classified_nonlocal_event_table,
        build_event_table,
        build_event_sequence_table,
        build_event_arm_table,
        build_stop_table,
        build_contfrag_event_table,
        build_contfrag_event_arm_table,
        build_contfrag_stop_table,
        run_session,
    ),
    ids=lambda builder: builder.__name__,
)
def test_a_builder_takes_no_speed_threshold(builder):
    parameters = inspect.signature(builder).parameters
    assert "speed_threshold" not in parameters
    assert "immobility_threshold" not in parameters


def _swr_frame(n: int = 60) -> pd.DataFrame:
    labels = np.zeros(n, dtype=int)
    labels[10:20], labels[30:40] = 1, 2
    speed = np.zeros(n)
    speed[35] = 3.0  # immobile at 4 cm/s, moving at 2 cm/s
    return pd.DataFrame(
        {
            "animal": "sim",
            "day": 1,
            "epoch": "02_r1",
            "head_speed": speed,
            "SWR_event": labels,
        },
        index=pd.Index(100.0 + np.arange(n) / 500.0, name="time"),
    )


class TestCanonicalSwrIntervals:
    def test_a_looser_threshold_is_refused(self):
        with pytest.raises(
            ValueError, match=r"canonical_swr_intervals: .*4 cm/s or a stricter"
        ):
            canonical_swr_intervals(_swr_frame(), speed_threshold=5.0)

    def test_a_stricter_threshold_is_honored(self):
        frame = _swr_frame()

        stricter = canonical_swr_intervals(frame, speed_threshold=2.0)

        assert stricter.SWR_event.tolist() == [1]
        pd.testing.assert_frame_equal(
            stricter,
            intervals_from_labels(
                filter_event_labels_by_speed(frame, ["SWR_event"], speed_threshold=2.0),
                "SWR_event",
            ),
        )
        assert canonical_swr_intervals(
            frame, speed_threshold=4.0
        ).SWR_event.tolist() == [
            1,
            2,
        ]


class TestLoadSessionSwrIntervals:
    def test_a_looser_threshold_is_refused_before_the_read(self, tmp_path):
        with pytest.raises(
            ValueError, match=r"load_session_swr_intervals: .*4 cm/s or a stricter"
        ):
            load_session_swr_intervals(
                "sim20200101_.nwb",
                "02_r1",
                dot_path=tmp_path / "missing.pkl",
                speed_threshold=5.0,
            )

    @pytest.mark.parametrize("speed_threshold", (2.0, CANONICAL))
    def test_the_canonical_or_a_stricter_threshold_reaches_the_read(
        self, tmp_path, speed_threshold
    ):
        # Past the guard, the missing frame is the next refusal.
        with pytest.raises(FileNotFoundError):
            load_session_swr_intervals(
                "sim20200101_.nwb",
                "02_r1",
                dot_path=tmp_path / "missing.pkl",
                speed_threshold=speed_threshold,
            )


def test_the_guard_names_the_canonical_value():
    with pytest.raises(ValueError, match="the canonical 4 cm/s"):
        require_canonical_speed_threshold(4.5, "caller")
    require_canonical_speed_threshold(CANONICAL, "caller")
    require_canonical_speed_threshold(1.0, "caller", allow_stricter=True)
    with pytest.raises(ValueError, match="4 cm/s or a stricter"):
        require_canonical_speed_threshold(float("nan"), "caller", allow_stricter=True)
