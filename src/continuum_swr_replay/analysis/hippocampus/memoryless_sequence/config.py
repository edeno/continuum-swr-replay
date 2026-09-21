"""Every threshold the memoryless sequence detector uses, in one place.

The equivalent of the original's ``Analysis_Information.mat``. Spec 02 records
which settings are retained from Widloski & Foster (2025) and which are adapted
to this project's track geometry and sampling grid.
"""

from dataclasses import dataclass

__all__ = ["MemorylessSequenceConfig"]


@dataclass(frozen=True)
class MemorylessSequenceConfig:
    """Thresholds for one detection run.

    Counts of windows are derived from `stride_s`, never carried over from the
    original as literal constants (spec 02, V10). At the reference's 5 ms stride
    the 100 ms duration filter keeps runs of index length 20, meaning 21
    windows; at this project's 4 ms stride the same 100 ms keeps index length
    25, meaning 26 windows.

    Attributes
    ----------
    window_duration_s : float
        Decoding window, 0.080 s. Retained from the original.
    stride_s : float
        Step between window starts, 0.004 s. The original uses 0.005 s, which is
        2.5 samples on this project's 500 Hz grid (V10).
    sampling_frequency : float
        Samples per second of the likelihood's time grid.
    place_bin_size_cm : float
        Spatial bin width, 2.0 cm. Retained.
    spread_threshold_cm : float
        A window is kept when its posterior spread is below this. The original's
        statistic is an area-like product moment in bin units; this port uses a
        graph-distance RMS spread in cm, so 10 cm is an adapted threshold rather
        than the same criterion (V9).
    jump_threshold_cm : float
        A window is kept when the track-graph distance to the next window's
        center is below this, 20.0 cm. Retained, measured on the graph (V6).
    speed_threshold_cm_per_s : float
        A window is kept when the speed at its midpoint is below this. This
        project uses 4.0 cm/s, its canonical immobility rule, where the original
        uses 5.0 (D3, V12).
    merge_gap_cm, merge_gap_s : float
        Neighboring runs merge when separated by less than both, 20.0 cm and
        0.050 s. Retained.
    minimum_duration_s : float
        Shortest sequence, 0.100 s between the first and last window midpoint.
        Retained.
    dispersion_threshold_cm : float
        Smallest spatial dispersion of a replay event, 12.0 cm. Retained.

    """

    window_duration_s: float = 0.080
    stride_s: float = 0.004
    sampling_frequency: float = 500.0
    place_bin_size_cm: float = 2.0
    spread_threshold_cm: float = 10.0
    jump_threshold_cm: float = 20.0
    speed_threshold_cm_per_s: float = 4.0
    merge_gap_cm: float = 20.0
    merge_gap_s: float = 0.050
    minimum_duration_s: float = 0.100
    dispersion_threshold_cm: float = 12.0

    def __post_init__(self) -> None:
        for field in (
            "window_duration_s",
            "stride_s",
            "sampling_frequency",
            "place_bin_size_cm",
            "spread_threshold_cm",
            "jump_threshold_cm",
            "speed_threshold_cm_per_s",
            "merge_gap_cm",
            "merge_gap_s",
            "minimum_duration_s",
            "dispersion_threshold_cm",
        ):
            value = getattr(self, field)
            if not value > 0:
                raise ValueError(f"{field} must be positive, got {value}.")
        if self.stride_s > self.window_duration_s:
            raise ValueError(
                f"stride_s ({self.stride_s}) is longer than window_duration_s "
                f"({self.window_duration_s}); the windows would not overlap."
            )
        for field in ("window_duration_s", "stride_s"):
            samples = getattr(self, field) * self.sampling_frequency
            if abs(samples - round(samples)) > 1e-9:
                raise ValueError(
                    f"{field} ({getattr(self, field)} s) is {samples} samples at "
                    f"{self.sampling_frequency} Hz, which is not a whole number."
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
        ``n * stride_s`` between its first and last midpoint.
        """
        return round(self.minimum_duration_s / self.stride_s)

    @property
    def merge_gap_index_length(self) -> int:
        """Largest gap between runs that still merges, as an index difference."""
        return round(self.merge_gap_s / self.stride_s)
