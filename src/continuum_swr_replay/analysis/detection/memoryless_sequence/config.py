"""Every threshold the memoryless sequence detector uses, in one place.

The equivalent of the original's ``Analysis_Information.mat``. Most settings
are retained from Widloski & Foster (2025); the stride, the speed threshold
and the spread threshold are adapted to this project's sampling grid,
immobility rule and track geometry, as each field's description says.
"""

import math
from dataclasses import dataclass, fields

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.parameters import SAMPLING_FREQUENCY

__all__ = ["MemorylessSequenceConfig"]

_WHOLE_NUMBER_TOLERANCE = 1e-9
"""How far a derived count may sit from an integer before it is not one."""


@dataclass(frozen=True)
class MemorylessSequenceConfig:
    """Thresholds for one detection run.

    Counts of windows are derived from `stride_s`, never carried over from the
    original as literal constants. At the reference's 5 ms stride the 100 ms
    duration filter keeps runs of index length 20, meaning 21 windows; at this
    project's 4 ms stride the same 100 ms keeps index length 25, meaning 26
    windows. A ``minimum_duration_s`` that is not a whole number of strides
    is rounded *up* to the next stride, never down: the reference's 1D branch
    uses 50 ms, which at a 4 ms stride keeps runs of 13 strides (52 ms), the
    shortest run that is at least 50 ms.

    Attributes
    ----------
    window_duration_s : float
        Decoding window, 0.080 s. Retained from the original.
    stride_s : float
        Step between window starts, 0.004 s. The original uses 0.005 s, which is
        2.5 samples on this project's 500 Hz grid.
    sampling_frequency : float
        Samples per second of the likelihood's time grid.
    place_bin_size_cm : float
        Spatial bin width, 2.0 cm. Retained. The detector reads its geometry
        from the bin-distance matrix it is handed, so whoever builds the
        environment must build it at this size.
    spread_threshold_cm : float
        A window is kept when its posterior spread is below this. The original's
        statistic is an area-like product moment in bin units; this port uses a
        graph-distance RMS spread in cm, so 10 cm is an adapted threshold rather
        than the same criterion.
    jump_threshold_cm : float
        A window is kept when the track-graph distance to the next window's
        center is below this, 20.0 cm. Retained, measured on the graph.
    speed_threshold_cm_per_s : float
        A window is kept when the speed at its midpoint is below this. This
        project's canonical immobility rule, 4.0 cm/s, shared with every other
        event cohort through
        :data:`~continuum_swr_replay.analysis.constants.DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC`.
        The original uses 5.0.
    merge_gap_cm, merge_gap_s : float
        Neighboring runs merge when separated by less than both, 20.0 cm and
        0.050 s. Retained. Two distinct runs are always at least two strides
        apart, so a ``merge_gap_s`` of ``2 * stride_s`` or less turns merging
        off.
    minimum_duration_s : float
        0.100 s between a sequence's first and last window midpoint. Retained.
        A candidate survives assembly when it spans at least this much
        (``>=``, matching the reference's ``lengths < thr`` removal) and is
        replay only when it spans strictly more (``>``), so a candidate of
        exactly 0.100 s is kept with ``is_replay`` False.
    dispersion_threshold_cm : float
        Smallest spatial dispersion of a replay event, 12.0 cm. Retained.
    drop_single_window_runs : bool
        Drop runs of one kept window before merging, True. Whether the
        reference does depends on the length convention of ``seqle``, which is
        vendored nowhere; False is the sensitivity setting, in which a lone
        window can link two runs into one merge.

    """

    window_duration_s: float = 0.080
    stride_s: float = 0.004
    sampling_frequency: float = float(SAMPLING_FREQUENCY)
    place_bin_size_cm: float = 2.0
    spread_threshold_cm: float = 10.0
    jump_threshold_cm: float = 20.0
    speed_threshold_cm_per_s: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
    merge_gap_cm: float = 20.0
    merge_gap_s: float = 0.050
    minimum_duration_s: float = 0.100
    dispersion_threshold_cm: float = 12.0
    drop_single_window_runs: bool = True

    def __post_init__(self) -> None:
        # every setting is a duration, a rate or a threshold, so every one
        # must be a positive number
        for setting in fields(self):
            value = getattr(self, setting.name)
            if setting.name == "drop_single_window_runs":
                if not isinstance(value, bool):
                    raise TypeError(f"{setting.name} must be True or False.")
                continue
            if isinstance(value, bool) or not (math.isfinite(value) and value > 0):
                raise ValueError(
                    f"{setting.name} must be a positive finite number, got {value!r}."
                )
        if self.stride_s > self.window_duration_s:
            raise ValueError(
                f"stride_s ({self.stride_s}) is longer than window_duration_s "
                f"({self.window_duration_s}); the windows would leave gaps."
            )
        for field in ("window_duration_s", "stride_s"):
            seconds = getattr(self, field)
            samples = seconds * self.sampling_frequency
            if abs(samples - round(samples)) > _WHOLE_NUMBER_TOLERANCE:
                raise ValueError(
                    f"{field} ({seconds} s) is {samples} samples at "
                    f"{self.sampling_frequency} Hz, which is not a whole number."
                )
        if round(self.window_duration_s * self.sampling_frequency) % 2:
            raise ValueError(
                f"window_duration_s ({self.window_duration_s} s) must be an even "
                f"number of samples, so its midpoint is a sample; event bounds are "
                f"samples."
            )
        if self.minimum_duration_s < self.stride_s:
            raise ValueError(
                f"minimum_duration_s ({self.minimum_duration_s} s) is shorter than "
                f"one stride ({self.stride_s} s); every run of two windows would pass."
            )

    @property
    def window_n_samples(self) -> int:
        """Rows of the likelihood that one window spans."""
        return round(self.window_duration_s * self.sampling_frequency)

    @property
    def stride_n_samples(self) -> int:
        """Rows between the starts of consecutive windows."""
        return round(self.stride_s * self.sampling_frequency)

    @property
    def minimum_run_index_length(self) -> int:
        """Shortest run that the duration filter keeps, as an index difference.

        A run of index length ``n`` holds ``n + 1`` windows and spans
        ``n * stride_s`` between its first and last midpoint. The smallest
        ``n`` whose span is at least ``minimum_duration_s``; a ratio within
        float tolerance of a whole number is that whole number.
        """
        strides = self.minimum_duration_s / self.stride_s
        if abs(strides - round(strides)) <= _WHOLE_NUMBER_TOLERANCE:
            return round(strides)
        return math.ceil(strides)
