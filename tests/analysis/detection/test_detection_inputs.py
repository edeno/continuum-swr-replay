"""The LFP samples the SWR and Carey runners keep, and the speed at them."""

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.detection.detection_inputs import (
    position_span_samples,
)


def test_position_span_samples_keep_both_ends_and_interpolate_speed():
    """Position runs 1.0 to 2.0 s; samples at exactly 1.0 and 2.0 are kept, the
    ones outside are not, and the speed is linear between position samples."""
    time = np.array([0.5, 1.0, 1.25, 1.5, 2.0, 2.5])
    position_info = pd.DataFrame(
        {"head_speed": [0.0, 10.0, 2.0]}, index=pd.Index([1.0, 1.5, 2.0])
    )

    span_s, samples, speed = position_span_samples(time, position_info)

    assert span_s == (1.0, 2.0)
    assert samples == slice(1, 5)
    np.testing.assert_array_equal(time[samples], [1.0, 1.25, 1.5, 2.0])
    np.testing.assert_allclose(speed, [0.0, 5.0, 10.0, 2.0])


def test_position_span_samples_leave_a_missing_speed_missing():
    """A NaN head speed stays NaN at the samples beside it."""
    time = np.array([0.0, 0.5, 1.0])
    position_info = pd.DataFrame(
        {"head_speed": [1.0, np.nan]}, index=pd.Index([0.0, 1.0])
    )

    _, samples, speed = position_span_samples(time, position_info)

    assert samples == slice(0, 3)
    assert speed[0] == 1.0
    assert np.isnan(speed[1:]).all()
