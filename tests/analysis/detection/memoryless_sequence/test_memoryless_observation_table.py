"""What one epoch of the memoryless detector observed: decoded windows and SWR labels."""

import numpy as np

from continuum_swr_replay.analysis.detection.memoryless_sequence import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.epoch import (
    observation_table,
)

CONFIG = MemorylessSequenceConfig(sampling_frequency=500.0)
TIME = np.round(np.arange(50) * 0.002, 3)
"""50 samples at 500 Hz. Without the final row, 49 samples hold five 40-sample
windows two samples apart, timestamped at samples 20, 22, 24, 26 and 28:
0.040, 0.044, 0.048, 0.052 and 0.056 s."""


def test_decoded_stretches_skip_the_windows_holding_a_missing_bin():
    table = observation_table(
        TIME,
        np.array([[0.0, 0.05]]),
        CONFIG,
        np.array([False, False, True, False, False]),
    )

    assert list(table.kind) == ["decoded_windows", "decoded_windows", "swr_labels"]
    np.testing.assert_allclose(table.start_s, [0.040, 0.052, 0.0])
    np.testing.assert_allclose(table.end_s, [0.044, 0.056, 0.05])
    assert dict(table.dtypes.astype(str)) == {
        "kind": "object",
        "start_s": "float64",
        "end_s": "float64",
    }


def test_every_window_missing_leaves_only_the_swr_stretches():
    table = observation_table(TIME, np.empty((0, 2)), CONFIG, np.ones(5, dtype=bool))

    assert table.empty
    assert list(table.columns) == ["kind", "start_s", "end_s"]
