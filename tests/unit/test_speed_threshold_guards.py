"""A builder's ``speed_threshold`` cannot split the speed cohorts.

The read (``swr_source.with_swr_source``) gates the SWR and MUA labels at the
canonical 4 cm/s, once for every consumer. A table builder that gated its own
events at another value would classify them against SWRs and MUA events gated
at 4 cm/s, so every builder refuses a non-default value. The two filters that
re-gate the read's SWRs themselves honour a stricter value and refuse only a
looser one, which could not re-admit the SWRs the read dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.analysis.hippocampus.event_immobility import (
    filter_event_labels_by_speed,
    intervals_from_labels,
    require_canonical_speed_threshold,
)
from continuum_swr_replay.analysis.hippocampus.event_table import (
    build_classified_nonlocal_event_table,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.event_table import (
    canonical_swr_intervals,
)
from continuum_swr_replay.analysis.pfc.canonical_events import (
    load_session_swr_intervals,
)
from tests.unit import test_contfrag_straddling_event_n_events as contfrag
from tests.unit import test_replay_tables as replay

CANONICAL = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
NON_DEFAULT = (2.0, 5.0, float("nan"))


def _session():
    dot = replay._tiny_dot_with_two_events()
    return dot, replay._EnvStub(), contfrag._empty_trials(), replay._bandit_graph()


def _contfrag_dot() -> pd.DataFrame:
    return contfrag._two_event_dot()


def _call_builder(name: str, speed_threshold: float):
    dot, env, trials, graph = _session()
    cdot = _contfrag_dot()
    cf_env, cf_graph = contfrag._EnvStub(), contfrag._bandit_graph()
    builders = {
        "build_classified_nonlocal_event_table": lambda: (
            build_classified_nonlocal_event_table(dot, speed_threshold=speed_threshold)
        ),
        "build_event_table": lambda: replay.build_event_table(
            dot, env, trials, graph, speed_threshold=speed_threshold
        ),
        "build_event_sequence_table": lambda: replay.build_event_sequence_table(
            dot, env, replay._tiny_posterior(dot), speed_threshold=speed_threshold
        ),
        "build_stop_table": lambda: replay.build_stop_table(
            dot, trials, graph, speed_threshold=speed_threshold
        ),
        "build_contfrag_event_table": lambda: contfrag.build_contfrag_event_table(
            cdot, cf_env, trials, cf_graph, speed_threshold=speed_threshold
        ),
        "build_contfrag_event_arm_table": lambda: (
            contfrag.build_contfrag_event_arm_table(
                contfrag.build_contfrag_event_table(cdot, cf_env, trials, cf_graph),
                cdot,
                cf_env,
                trials,
                cf_graph,
                speed_threshold=speed_threshold,
            )
        ),
        "build_contfrag_stop_table": lambda: contfrag.build_contfrag_stop_table(
            cdot, trials, cf_graph, speed_threshold=speed_threshold
        ),
    }
    return builders[name]()


BUILDERS = (
    "build_classified_nonlocal_event_table",
    "build_event_table",
    "build_event_sequence_table",
    "build_stop_table",
    "build_contfrag_event_table",
    "build_contfrag_event_arm_table",
    "build_contfrag_stop_table",
)


@pytest.mark.parametrize("name", BUILDERS)
@pytest.mark.parametrize("speed_threshold", NON_DEFAULT)
def test_a_builder_refuses_a_non_default_speed_threshold(name, speed_threshold):
    with pytest.raises(ValueError, match=rf"{name}: .*the canonical 4 cm/s"):
        _call_builder(name, speed_threshold)


@pytest.mark.parametrize("name", BUILDERS)
def test_a_builder_accepts_the_canonical_speed_threshold(name):
    _call_builder(name, CANONICAL)


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

    def test_a_stricter_threshold_is_honoured(self):
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
