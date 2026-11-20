"""Tests for the one-class-or-two helpers (``analysis.event_properties.class_or_continuum``)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from continuum_swr_replay.analysis.event_properties import class_or_continuum as cc


def test_kay_settings_are_the_detectors():
    assert cc.KAY_ZSCORE_THRESHOLD == 2.0
    assert cc.KAY_MINIMUM_DURATION_S == 0.015


# --- test 1 ----------------------------------------------------------------------


def test_sarle_of_uniform_and_normal():
    rng = np.random.default_rng(0)
    assert cc.sarle_bimodality(rng.uniform(size=200_000)) == pytest.approx(
        cc.SARLE_CUTOFF, abs=0.01
    )
    assert cc.sarle_bimodality(rng.normal(size=200_000)) == pytest.approx(
        1 / 3, abs=0.01
    )


def test_sarle_calls_a_skewed_unimodal_law_bimodal():
    # why it is descriptive only: an exponential sits at the cutoff, a
    # more skewed gamma above it
    rng = np.random.default_rng(1)
    assert cc.sarle_bimodality(rng.gamma(0.5, size=200_000)) > cc.SARLE_CUTOFF


def test_dip_test_separates_skewed_unimodal_from_two_classes():
    rng = np.random.default_rng(2)
    _, p_unimodal = cc.dip_test(rng.lognormal(0.0, 1.0, 20_000))
    _, p_mixture = cc.dip_test(
        cc.two_class_mixture_sample(rng, 20_000, (0.0, 5.0), (1.0, 1.0), 0.5)
    )
    assert p_unimodal > 0.5
    assert p_mixture < 1e-3


def test_kde_modes():
    rng = np.random.default_rng(3)
    two = cc.kde_modes(
        cc.two_class_mixture_sample(rng, 20_000, (0.0, 6.0), (1.0, 1.0), 0.4)
    )
    assert two.size == 2
    assert two[0] == pytest.approx(0.0, abs=0.3)  # the larger class first
    assert two[1] == pytest.approx(6.0, abs=0.3)
    one = cc.kde_modes(rng.normal(1.0, 1.0, 20_000))
    assert one.size == 1
    assert one[0] == pytest.approx(1.0, abs=0.2)
    # a long tail's ripples are not modes
    assert cc.kde_modes(rng.lognormal(0.0, 1.0, 20_000)).size == 1


@pytest.mark.parametrize(
    "sampler", [cc.matched_unimodal_sample, cc.matched_lognormal_sample]
)
def test_matched_samples_have_the_moments(sampler):
    rng = np.random.default_rng(4)
    x = sampler(rng, 400_000, 1.5, 2.0, 2.3)
    assert x.mean() == pytest.approx(1.5, abs=0.02)
    assert x.std() == pytest.approx(2.0, rel=0.02)
    assert stats.skew(x) == pytest.approx(2.3, rel=0.1)
    with pytest.raises(ValueError, match="positive"):
        sampler(rng, 10, 0.0, 1.0, -0.1)


# --- test 2 ----------------------------------------------------------------------


def test_common_support():
    power = np.array([0.0, 1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 4.0])
    is_swr = np.array([False, False, False, False, True, True, True, True])
    low, high = cc.common_support(power, is_swr, tail=0.0)
    assert (low, high) == (1.5, 3.0)
    low, high = cc.common_support(np.array([0.0, 1.0, 2.0, 3.0]), is_swr[2:6], tail=0)
    assert low >= high  # disjoint labels: no common support


def test_fit_step_null_and_planted():
    rng = np.random.default_rng(5)
    for step, expect_step in ((0.0, False), (0.4, True)):
        d = cc.simulate_step_data(rng, step=step)
        low, high = cc.common_support(d.power, d.is_swr)
        s = d[(d.power >= low) & (d.power <= high)]
        fit = cc.fit_step(s.y, s.power, s.is_swr, s.group)
        assert fit.step_ci_low < step < fit.step_ci_high
        assert fit.has_step is expect_step
        assert fit.n_folds == 5
        assert fit.n_events == len(s)
        assert fit.n_labeled == int(s.is_swr.sum())


def test_fit_step_with_a_covariate():
    rng = np.random.default_rng(6)
    d = cc.simulate_step_data(rng, step=0.0)
    covariate = d.power + rng.normal(0, 1, len(d))
    d = d.assign(y=d.y + np.sin(covariate))
    fit = cc.fit_step(d.y, d.power, d.is_swr, d.group, covariate=covariate)
    assert fit.step_ci_low < 0 < fit.step_ci_high
    assert fit.label_vif > 1


def _fit(step: float, ci_low: float, ci_high: float) -> cc.StepFit:
    """A step test with only its term set; the share reads nothing else."""
    return cc.StepFit(
        n_events=0,
        n_groups=0,
        n_labeled=0,
        step=step,
        step_ci_low=ci_low,
        step_ci_high=ci_high,
        raw_difference=np.nan,
        step_share=np.nan,
        label_vif=np.nan,
        label_power_correlation=np.nan,
        mse_smooth=np.nan,
        mse_step=np.nan,
        heldout_gain=np.nan,
        folds_step_better=0,
        n_folds=0,
    )


def test_overall_step_share_is_over_the_mean_difference_of_every_event():
    # non-SWR 1, 2 and 6 (mean 3, median 2); SWR 5 and 7 (mean 6)
    y = np.array([1.0, 2.0, 5.0, 6.0, 7.0])
    is_swr = np.array([False, False, True, False, True])
    share = cc.overall_step_share(_fit(1.5, -0.3, 3.0), y, is_swr)
    assert share == pytest.approx(
        {
            "overall_difference": 3.0,
            "step_share_overall": 0.5,
            "share_overall_ci_low": -0.1,
            "share_overall_ci_high": 1.0,
        }
    )


def test_overall_step_share_keeps_the_lower_bound_first_below_zero():
    # the SWR events lower: a difference of -3 swaps the bounds
    y = np.array([4.0, 5.0, 1.0, 2.0])
    is_swr = np.array([False, False, True, True])
    share = cc.overall_step_share(_fit(1.5, -0.3, 3.0), y, is_swr)
    assert share["step_share_overall"] == pytest.approx(-0.5)
    assert share["share_overall_ci_low"] == pytest.approx(-1.0)
    assert share["share_overall_ci_high"] == pytest.approx(0.1)


def test_sustained_level():
    z = np.array([0.0, 3.0, 3.0, 2.0, np.nan, 5.0])
    level = cc.sustained_level(z, 2)
    np.testing.assert_array_equal(level, [0.0, 3.0, 2.0, -np.inf, -np.inf, -np.inf])


def _trace(*values):
    return np.array(values, dtype=float)


def test_kay_like_intervals():
    # a run of 3 samples at or above 2 inside a positive run: one SWR, extended
    # to the positive run; a run of 2 is too short
    z = _trace(-1, 0.5, 2, 2.5, 2, 1, -1, 0.5, 3, 3, 0.2, -1)
    np.testing.assert_array_equal(cc.kay_like_intervals(z, minimum_samples=3), [[1, 5]])
    # a NaN splits the run above threshold
    z[3] = np.nan
    assert cc.kay_like_intervals(z, minimum_samples=3).shape == (0, 2)


def test_kay_criterion_reads_the_ripple_the_event_touches():
    z = _trace(-1, 0.5, 2, 2.5, 2, 1, -1, 0.5, 3, 3, 0.2, -1)
    # an event on the SWR's flank only (rows 4-5) still sees its sustained 2.0
    # an event on the brief burst (rows 8-9; its run is rows 7-10) sees 0.5,
    # the best 3-sample minimum of the run; an event in the trough sees <= 0
    first = np.array([4, 8, 6])
    last = np.array([5, 9, 6])
    criterion = cc.kay_criterion(z, first, last, minimum_samples=3)
    np.testing.assert_allclose(criterion, [2.0, 0.5, -1.0])


def test_criterion_and_label_agree_on_simulated_events():
    rng = np.random.default_rng(7)
    events = cc.simulate_ripple_events(rng, duration_s=120.0)
    assert events.is_swr.any() and (~events.is_swr).any()
    np.testing.assert_array_equal(
        events.power_criterion >= cc.KAY_ZSCORE_THRESHOLD, events.is_swr
    )
    # the event window's own peak misses part of the ripple
    assert (events.power_peak <= events.power_criterion + 1e-12).mean() < 1.0


def _simulated_sessions(rng, step, n=12):
    frames = []
    for g in range(n):
        e = cc.simulate_ripple_events(rng, duration_s=300.0, step=step)
        frames.append(e.assign(group=g))
    return pd.concat(frames, ignore_index=True)


def test_discontinuity_design_on_one_continuum_and_a_planted_step():
    # over repeated simulations: one continuum mostly gives an interval
    # covering 0, a planted step mostly an interval above 0 that covers it
    rng = np.random.default_rng(8)
    for step in (0.0, 1.0):
        covers, positive = 0, 0
        for _ in range(6):
            d = _simulated_sessions(rng, step, n=12)
            window = d[(d.power_criterion >= 0.5) & (d.power_criterion <= 3.5)]
            above = window.power_criterion >= cc.KAY_ZSCORE_THRESHOLD
            fit = cc.fit_step(
                window.y, window.power_criterion, above, window.group, n_knots=5
            )
            covers += fit.step_ci_low < step < fit.step_ci_high
            positive += fit.step_ci_low > 0
        # a 95 % interval misses now and then (the script measures the rate
        # over many more repeats); six repeats allow one or two misses
        assert covers >= 4
        assert positive >= 5 if step > 0 else positive <= 2


# --- the events' tests -------------------------------------------------------------


def test_event_ripple_power_reads_each_event_s_rows():
    z = _trace(-1, 0.5, 2, 2.5, 2, 1, -1, 0.5, 3, 3, 0.2, -1)
    swr = np.zeros(z.size, dtype=bool)
    swr[2:5] = True
    first = np.array([1, 5, 8])
    n_rows = np.array([2, 3, 1])

    power = cc.event_ripple_power(z, swr, first, n_rows)

    np.testing.assert_allclose(power["ripple_z_max"], [2.0, 1.0, 3.0])
    np.testing.assert_allclose(power["ripple_z_mean"], [1.25, 0.5 / 3, 3.0])
    np.testing.assert_array_equal(power["rows_hold_swr"], [True, False, False])
    np.testing.assert_allclose(
        power["ripple_z_criterion"], cc.kay_criterion(z, first, first + n_rows - 1)
    )
    with pytest.raises(ValueError, match="holds no grid row"):
        cc.event_ripple_power(z, swr, first, np.array([2, 0, 1]))


def test_robust_sd_is_the_scaled_median_absolute_deviation():
    # median 2; deviations 1, 0, 0, 2, 7 -> median 1
    assert cc.robust_sd(np.array([1.0, 2.0, 2.0, 4.0, 9.0])) == pytest.approx(1.4826)


def _power_events() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    frames = []
    for animal, n in (("a", 120), ("b", 80)):
        is_swr = np.arange(n) % 4 == 0
        peak = rng.normal(1.0, 1.0, n) + 2.0 * is_swr
        frames.append(
            pd.DataFrame(
                {
                    "animal": animal,
                    "is_swr": is_swr,
                    "ripple_z_max": peak,
                    "ripple_z_mean": peak - 1.0,
                    "ripple_z_criterion": peak + 0.5,
                }
            )
        )
    events = pd.concat(frames, ignore_index=True)
    events.loc[3, "ripple_z_mean"] = np.nan
    return events


def test_the_distribution_table_has_a_row_per_animal_pooled_and_power():
    events = _power_events()
    table = cc.distribution_table(events)

    assert list(zip(table.animal, table.power, strict=True)) == [
        (a, p) for a in ("a", "b", "pooled") for p in ("peak", "mean", "criterion")
    ]
    row = table[(table.animal == "a") & (table.power == "mean")].iloc[0]
    # the event without a mean is left out of that row only
    x = events.loc[(events.animal == "a") & events.ripple_z_mean.notna()]
    assert row.n_events == 119
    assert row["mean"] == pytest.approx(x.ripple_z_mean.mean())
    assert row.sd == pytest.approx(x.ripple_z_mean.std(ddof=0))
    assert row.median_swr == pytest.approx(x.ripple_z_mean[x.is_swr].median())
    assert row.robust_sd_non_swr == pytest.approx(
        cc.robust_sd(x.ripple_z_mean[~x.is_swr].to_numpy())
    )
    assert row.swr_share == pytest.approx(x.is_swr.mean())
    assert row.share_at_or_above_threshold == pytest.approx(
        (x.ripple_z_mean >= cc.KAY_ZSCORE_THRESHOLD).mean()
    )
    pooled = table[(table.animal == "pooled") & (table.power == "peak")].iloc[0]
    assert pooled.n_events == 200
    assert pooled.dip_p == pytest.approx(cc.dip_test(events.ripple_z_max.to_numpy())[1])


def test_a_property_frame_reads_its_property_on_its_events():
    events = pd.DataFrame(
        {
            "event_duration_s": [0.05, 0.1, 0.2],
            "n_spikes": [0, 9, 3],
            "dll_per_spike": [0.5, np.nan, -0.2],
            "remote": [True, True, False],
        }
    )
    spikes = cc.property_frame(events, "spikes")
    np.testing.assert_allclose(spikes.y, np.log1p([0, 9, 3]))
    np.testing.assert_allclose(spikes.log_duration, np.log([0.05, 0.1, 0.2]))
    # spike evidence is defined on the remote events, and a NaN is left out
    evidence = cc.property_frame(events, "evidence")
    assert evidence.index.tolist() == [0]
    assert evidence.y.tolist() == [0.5]


def test_design_rows_on_the_criterion_window_and_the_common_support():
    t = cc.KAY_ZSCORE_THRESHOLD
    frame = pd.DataFrame(
        {
            "ripple_z_criterion": [t - 1.5, t - 1.6, t, t + 1.5, t + 1.0],
            "ripple_z_max": [0.0, 1.0, 2.0, 3.0, 4.0],
            "is_swr": [False, False, True, True, False],
        }
    )
    keep, label, low, high = cc.design_rows(frame, "criterion", 1.5)
    # the window's bounds are in it
    assert keep.tolist() == [True, False, True, True, True]
    assert label.tolist() == [False, True, True, True]
    assert (low, high) == (t - 1.5, t + 1.5)

    keep, label, low, high = cc.design_rows(frame, "peak", np.nan)
    assert (low, high) == cc.common_support(frame.ripple_z_max, frame.is_swr)
    assert keep.tolist() == list(
        (frame.ripple_z_max >= low) & (frame.ripple_z_max <= high)
    )
    assert label.tolist() == frame.is_swr[keep].tolist()


def test_the_step_table_fits_every_setting_of_each_animal():
    rng = np.random.default_rng(5)
    frames = []
    for animal in ("a", "b"):
        d = _simulated_sessions(rng, 0.0, n=4).rename(
            columns={
                "power_peak": "ripple_z_max",
                "power_mean": "ripple_z_mean",
                "power_criterion": "ripple_z_criterion",
            }
        )
        n = len(d)
        frames.append(
            d.assign(
                animal=animal,
                epoch_id=d.group,
                event_duration_s=np.exp(d.log_duration),
                spatial_extent_cm=rng.uniform(0, 100, n),
                replay_speed_median_cm_s=rng.uniform(0, 500, n),
                posterior_peak_mean=rng.uniform(0, 1, n),
                hpd_coverage_cm=rng.uniform(5, 100, n),
                n_spikes=rng.integers(1, 30, n),
                dll_per_spike=rng.normal(0, 1, n),
                n_unique_arms=rng.integers(1, 4, n),
                remote=np.arange(n) % 2 == 0,
            )
        )
    events = pd.concat(frames, ignore_index=True)

    steps = cc.step_table(events, windows=(1.5,), knots=(5,))

    # per animal: 3 powers x 2 covariates for 7 properties, 3 for duration
    assert len(steps) == 2 * (7 * 6 + 3)
    assert set(steps.covariate[steps.property == "duration"]) == {"none"}
    assert steps.window_z[steps.power != "criterion"].isna().all()
    n_animal = (events.animal == "a").sum()
    evidence = steps[(steps.property == "evidence") & (steps.animal == "a")]
    assert (
        evidence.n_property_events == (events.remote & (events.animal == "a")).sum()
    ).all()
    duration = steps[(steps.property == "duration") & (steps.animal == "a")]
    assert (duration.n_property_events == n_animal).all()
    assert steps.rows_fraction.between(0, 1).all()


def test_nonlocal_share_by_power_counts_each_group_and_bin():
    from continuum_swr_replay.analysis.event_properties.class_or_continuum import (
        nonlocal_share_by_power,
    )

    power = np.array([0.5, 0.5, 0.5, 2.5, 2.5, 0.5, 2.5, 2.5])
    probability = np.array([0.9, 0.1, 0.1, 0.9, 0.95, 0.85, 0.2, 0.9])
    group = np.array(["a", "a", "a", "a", "a", "b", "b", "b"])
    shares = nonlocal_share_by_power(
        power, probability, group, edges=np.array([0.0, 1.0, 2.0, 3.0]), threshold=0.8
    )
    a = shares[shares["group"] == "a"].set_index("power_low")
    assert a.loc[0.0, "n_bins"] == 3
    assert a.loc[0.0, "share"] == pytest.approx(1 / 3)
    assert a.loc[1.0, "n_bins"] == 0
    assert np.isnan(a.loc[1.0, "share"])
    assert a.loc[2.0, "share"] == pytest.approx(1.0)
    b = shares[shares["group"] == "b"].set_index("power_low")
    assert b.loc[0.0, "share"] == pytest.approx(1.0)
    assert b.loc[2.0, "share"] == pytest.approx(0.5)


def test_nonlocal_share_by_power_leaves_out_bins_outside_the_edges_and_not_finite():
    from continuum_swr_replay.analysis.event_properties.class_or_continuum import (
        nonlocal_share_by_power,
    )

    power = np.array([-5.0, 0.5, np.nan, 0.5])
    probability = np.array([0.9, 0.9, 0.9, np.nan])
    shares = nonlocal_share_by_power(
        power,
        probability,
        np.array(["a"] * 4),
        edges=np.array([0.0, 1.0]),
        threshold=0.8,
    )
    assert shares["n_bins"].tolist() == [1]
    assert shares["share"].tolist() == [1.0]


def test_nonlocal_share_by_power_counts_the_nonlocal_bins():
    power = np.array([0.5, 0.5, 0.5, 2.5, 2.5, 0.5, 2.5, 2.5])
    probability = np.array([0.9, 0.1, 0.1, 0.9, 0.95, 0.85, 0.2, 0.9])
    group = np.array(["a", "a", "a", "a", "a", "b", "b", "b"])
    shares = cc.nonlocal_share_by_power(
        power, probability, group, edges=np.array([0.0, 1.0, 2.0, 3.0]), threshold=0.8
    )
    counts = shares.set_index(["group", "power_low"])["n_nonlocal"]
    assert counts.to_dict() == {
        ("a", 0.0): 1,
        ("a", 1.0): 0,
        ("a", 2.0): 2,
        ("b", 0.0): 1,
        ("b", 1.0): 0,
        ("b", 2.0): 1,
    }


def test_ripple_power_edges_count_every_power_and_split_at_the_kay_threshold():
    edges = cc.RIPPLE_POWER_EDGES
    assert edges[0] == -np.inf and edges[-1] == np.inf
    assert (np.diff(edges) > 0).all()
    assert cc.KAY_ZSCORE_THRESHOLD in edges
    # a power below the lowest finite edge, at the threshold and above the highest
    power = np.array([-7.0, cc.KAY_ZSCORE_THRESHOLD, 40.0])
    shares = cc.nonlocal_share_by_power(
        power, np.full(3, 0.9), np.zeros(3, dtype=int), edges=edges, threshold=0.8
    )
    assert shares["n_bins"].sum() == 3
    occupied = shares[shares["n_bins"] > 0]
    assert occupied["power_low"].tolist() == [-np.inf, cc.KAY_ZSCORE_THRESHOLD, 10.0]


def test_rest_power_counts_count_the_immobile_analyzed_bins_only():
    dot = pd.DataFrame(
        {
            # immobile, moving, immobile in excluded time, immobile, immobile
            "head_speed": [1.0, 20.0, 1.0, 1.0, 1.0],
            "in_excluded_time": [False, False, True, False, False],
            "SWR_zscore": [0.1, 0.1, 0.1, 0.1, 3.1],
            "Non-Local_Continuous_prob": [0.5, 0.9, 0.9, 0.1, 0.9],
            "Non-Local_Fragmented_prob": [0.4, 0.0, 0.0, 0.1, 0.0],
        }
    )
    counts = cc.rest_power_counts(dot).set_index("power_low")
    assert len(counts) == len(cc.RIPPLE_POWER_EDGES) - 1
    assert counts["n_bins"].sum() == 3
    assert counts.loc[0.0, ["n_bins", "n_nonlocal"]].tolist() == [2, 1]
    assert counts.loc[3.0, ["n_bins", "n_nonlocal"]].tolist() == [1, 1]
