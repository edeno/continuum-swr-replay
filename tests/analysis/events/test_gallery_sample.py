"""The event gallery's sample: spread over the ambiguity's quantile bins, and the
same in every process."""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.events.gallery_sample import (
    SAMPLE_SEED,
    group_seed,
    stratified_sample,
)


def test_a_small_group_is_kept_whole():
    group = pd.DataFrame({"ambiguity": [0.3, 0.1, 0.2]})

    pd.testing.assert_frame_equal(stratified_sample(group, n=3), group)


def test_the_sample_takes_the_same_number_from_each_ambiguity_bin():
    # 20 events, ambiguity 0..19: five quantile bins of four (0-3, 4-7, ...).
    # n=10 takes two from each bin.
    group = pd.DataFrame({"ambiguity": np.arange(20.0)[::-1]})

    sample = stratified_sample(group, n=10, n_bins=5, seed=3)

    assert len(sample) == 10
    bins = (sample["ambiguity"] // 4).value_counts().sort_index()
    assert bins.tolist() == [2, 2, 2, 2, 2]
    assert not sample.index.duplicated().any()


def test_a_short_bin_is_topped_up_from_the_remainder():
    # 7 events in five bins (two of two, three of one): n=7 takes one per bin
    # (n // 5 = 1), then two more from the rest.
    group = pd.DataFrame({"ambiguity": np.arange(7.0)})

    sample = stratified_sample(group, n=6, n_bins=5, seed=0)

    assert len(sample) == 6
    assert set(sample.index) <= set(group.index)


def test_the_group_seed_is_built_from_the_strings():
    assert group_seed("chimi", "non_swr") == group_seed("chimi", "non_swr")
    assert group_seed("chimi", "non_swr") != group_seed("j16", "non_swr")
    assert 0 <= group_seed("chimi", "non_swr") - SAMPLE_SEED <= 0xFFFF
