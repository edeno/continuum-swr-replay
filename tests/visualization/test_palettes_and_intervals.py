"""Tests for the promoted visualization helpers.

Covers :mod:`continuum_swr_replay.visualization.palettes` and the
generalized :func:`intervals_to_timebins` in
:mod:`continuum_swr_replay.visualization.plots`.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.data_loaders.constants import (
    TRACK_SEGMENT_TO_PATCH,
    WELL_PATCH_MAPPING,
)
from continuum_swr_replay.visualization.palettes import (
    ALL_ARMS,
    build_arm_color_map,
    patch_grouped_arm_palette,
)
from continuum_swr_replay.visualization.plots import (
    _intervals_to_mask,
    intervals_to_timebins,
)

# ── patch_grouped_arm_palette ──


class TestPatchGroupedPalette:
    def test_covers_every_arm(self):
        palette = patch_grouped_arm_palette()
        assert set(palette.keys()) == set(ALL_ARMS)

    def test_rgba_tuple_shape(self):
        palette = patch_grouped_arm_palette()
        for arm, rgba in palette.items():
            assert len(rgba) == 4, f"arm {arm} has non-RGBA color: {rgba}"
            assert all(0.0 <= v <= 1.0 for v in rgba[:3])

    def test_same_patch_shares_hue(self):
        """Arms in the same patch should have similar red/green/blue
        dominance so the palette reads as "patch by color"."""
        palette = patch_grouped_arm_palette()
        for patch_id in (1, 2, 3):
            arms_in_patch = [
                a for a, p in TRACK_SEGMENT_TO_PATCH.items() if p == patch_id
            ]
            colors = np.array([palette[a][:3] for a in arms_in_patch])
            # All three arms in the patch should agree on which channel
            # is brightest (B for Blues, G for Greens, R for Reds).
            dominant = np.argmax(colors, axis=1)
            assert len(np.unique(dominant)) == 1, (
                f"patch {patch_id} arms {arms_in_patch} disagree on dominant "
                f"channel: {dominant}"
            )

    def test_trunk_arm_darker_than_leaf(self):
        """Trunk arms use the darkest sample of their patch colormap,
        so their luminance should be lower than either leaf arm."""
        palette = patch_grouped_arm_palette()
        for patch_id in (1, 2, 3):
            arms = [a for a, p in TRACK_SEGMENT_TO_PATCH.items() if p == patch_id]
            leaves = [a for a in arms if a in WELL_PATCH_MAPPING]
            trunks = [a for a in arms if a not in WELL_PATCH_MAPPING]
            trunk_lum = sum(palette[trunks[0]][:3])
            for leaf in leaves:
                leaf_lum = sum(palette[leaf][:3])
                assert trunk_lum <= leaf_lum, (
                    f"trunk arm {trunks[0]} brighter than leaf arm {leaf} "
                    f"in patch {patch_id}"
                )


# ── build_arm_color_map (legacy) ──


class TestLegacyArmColorMap:
    def test_covers_every_arm_even_when_edge_order_partial(self):
        g = nx.Graph()
        for u, v, eid in [
            (0, 6, 0),
            (1, 6, 1),
            (2, 7, 2),
            (3, 7, 3),
            (4, 8, 4),
            (5, 8, 5),
            (6, 9, 6),
            (7, 9, 7),
            (8, 9, 8),
        ]:
            g.add_edge(u, v, edge_id=eid)
        # edge_order only visits two of the nine edges
        partial_order = np.array([(0, 6), (1, 6)])
        palette = build_arm_color_map(g, partial_order)
        assert set(palette.keys()) >= set(ALL_ARMS)


# ── intervals_to_timebins ──


class TestIntervalsToTimebins:
    def test_empty_intervals_boolean(self):
        intervals = pd.DataFrame(columns=["start_time", "end_time"])
        out = intervals_to_timebins(intervals, np.arange(5.0))
        assert out.dtype == bool
        assert not out.any()

    def test_empty_intervals_values(self):
        intervals = pd.DataFrame(columns=["start_time", "end_time", "val"])
        out = intervals_to_timebins(
            intervals, np.arange(5.0), value_col="val", fill_value=-1
        )
        assert (out == -1).all()

    def test_single_interval_mask(self):
        intervals = pd.DataFrame({"start_time": [1.0], "end_time": [3.0]})
        time_idx = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        out = intervals_to_timebins(intervals, time_idx)
        np.testing.assert_array_equal(out, [False, True, True, True, False])

    def test_value_carrying(self):
        intervals = pd.DataFrame(
            {
                "start_time": [0.0, 5.0, 10.0],
                "end_time": [2.0, 7.0, 12.0],
                "label": [10, 20, 30],
            }
        )
        time_idx = np.array([0.0, 1.0, 3.0, 6.0, 8.0, 11.0, 13.0])
        out = intervals_to_timebins(
            intervals, time_idx, value_col="label", fill_value=-1
        )
        np.testing.assert_array_equal(out, [10, 10, -1, 20, -1, 30, -1])

    def test_unsorted_intervals_handled(self):
        """Regression: intervals DataFrame need not be pre-sorted."""
        intervals = pd.DataFrame(
            {
                "start_time": [10.0, 0.0, 5.0],  # deliberately out of order
                "end_time": [12.0, 2.0, 7.0],
                "label": [30, 10, 20],
            }
        )
        time_idx = np.array([1.0, 6.0, 11.0])
        out = intervals_to_timebins(
            intervals, time_idx, value_col="label", fill_value=-1
        )
        np.testing.assert_array_equal(out, [10, 20, 30])

    def test_boundary_inclusive(self):
        """Both start and end are included in the interval."""
        intervals = pd.DataFrame({"start_time": [1.0], "end_time": [3.0]})
        time_idx = np.array([1.0, 3.0])
        out = intervals_to_timebins(intervals, time_idx)
        assert out.all()

    def test_custom_column_names(self):
        intervals = pd.DataFrame({"event_start": [1.0], "event_end": [3.0]})
        out = intervals_to_timebins(
            intervals,
            np.array([0.0, 2.0, 4.0]),
            start_col="event_start",
            end_col="event_end",
        )
        np.testing.assert_array_equal(out, [False, True, False])

    def test_backward_compat_wrapper(self):
        """``_intervals_to_mask`` must keep producing the same boolean
        mask as before the generalization."""
        intervals = pd.DataFrame({"start_time": [2.0, 5.0], "end_time": [3.0, 7.0]})
        time_idx = np.array([1.0, 2.5, 4.0, 6.0, 8.0])
        out = _intervals_to_mask(intervals, time_idx)
        np.testing.assert_array_equal(out, [False, True, False, True, False])

    def test_overlapping_intervals_union(self):
        """Overlapping intervals form a union — every bin covered by *any*
        interval must be True."""
        intervals = pd.DataFrame({"start_time": [1.0, 1.5], "end_time": [2.0, 2.5]})
        time_idx = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        out = intervals_to_timebins(intervals, time_idx)
        np.testing.assert_array_equal(out, [False, True, True, True, True, False])

    def test_zero_width_interval_includes_exact_match(self):
        """A degenerate ``[t, t]`` interval still includes the bin at
        exactly ``t`` (boundaries are inclusive on both sides)."""
        intervals = pd.DataFrame({"start_time": [1.0], "end_time": [1.0]})
        time_idx = np.array([0.999, 1.0, 1.001])
        out = intervals_to_timebins(intervals, time_idx)
        np.testing.assert_array_equal(out, [False, True, False])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ── plot_spike_times ──


def test_spike_raster_ticks_sit_on_the_first_and_last_rows():
    """eventplot draws row k at y = k; the ticks are on rows 0 and n - 1,
    labeled 1 and n."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xarray as xr

    from continuum_swr_replay.visualization.plots import plot_spike_times

    times = np.arange(0.0, 1.0, 0.1)
    states = ["Local", "No-Spike", "Non-Local Continuous", "Non-Local Fragmented"]
    results = xr.Dataset(
        {
            "acausal_state_probabilities": (
                ("time", "states"),
                np.tile([0.7, 0.1, 0.1, 0.1], (times.size, 1)),
            )
        },
        coords={"time": times, "states": states},
    )

    class NoPlaceFields:
        encoding_model_: dict = {}

    spikes = [np.array([0.2]), np.array([0.3, 0.4]), np.array([0.5])]
    fig, ax = plt.subplots()
    plot_spike_times(
        {"spike_times": {"HPC": spikes}}, slice(0.0, 1.0), NoPlaceFields(), None,
        results, ax,
    )  # fmt: skip
    rows = sorted({float(c.get_lineoffset()) for c in ax.collections[:3]})
    assert rows == [0.0, 1.0, 2.0]
    assert list(ax.get_yticks()) == [rows[0], rows[-1]]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["1", "3"]
    plt.close(fig)


@pytest.mark.parametrize(
    ("color", "expected"),
    [
        ("#add8e6", "black"),
        ("#b0b0b0", "black"),
        ("#1b9e77", "white"),
        ("#D55E00", "white"),
    ],
)
def test_text_on_a_light_fill_is_black_and_on_a_dark_fill_white(color, expected):
    from continuum_swr_replay.visualization.palettes import text_color_on

    assert text_color_on(color) == expected


def test_a_light_color_is_darkened_for_text_and_a_dark_one_kept():
    from colorsys import rgb_to_hls

    from matplotlib.colors import to_rgb

    from continuum_swr_replay.visualization.palettes import readable_text_color

    assert readable_text_color("#1b9e77") == "#1b9e77"
    darkened = readable_text_color("#add8e6")
    hue, lightness, _ = rgb_to_hls(*to_rgb(darkened))
    assert lightness == pytest.approx(0.4, abs=0.01)
    assert hue == pytest.approx(rgb_to_hls(*to_rgb("#add8e6"))[0], abs=0.01)


def test_the_event_kinds_and_classes_have_one_name_and_color_each():
    """The no-SWR events are a wine, not the Non-Local state's green, which
    every non-local event is drawn in, nor the SWR color; the class with no
    SWR is the no-SWR kind."""
    from continuum_swr_replay.visualization.palettes import (
        COLORS,
        EVENT_KINDS,
        OVERLAP_CLASSES,
    )

    assert list(EVENT_KINDS) == [False, True]  # the figures' order
    no_swr, swr = EVENT_KINDS[False][1], EVENT_KINDS[True][1]
    assert no_swr not in (COLORS["Non-Local"], swr)
    assert OVERLAP_CLASSES["non_swr"] == EVENT_KINDS[False]
    assert len({color for _, color in OVERLAP_CLASSES.values()}) == 3
