"""Aggregation rules of the marked-family population statistics."""

import json

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc import marked_family_statistics as stats


def _units() -> pd.DataFrame:
    """Two animals; animal a has two sessions, animal b one, with a blow-up."""
    rows = []
    for animal, session, deltas in (
        ("a", "a1", [1.0, -1.0, 2.0, np.nan]),  # 2/3 positive over finite
        ("a", "a2", [-1.0, -2.0, 3.0, -4.0]),  # 1/4 positive
        ("b", "b1", [5.0, 5000.0, -1.0, 1.0]),  # blow-up excluded -> 2/3
    ):
        for delta in deltas:
            rows.append(
                {
                    "animal": animal,
                    "session": session,
                    "model": "m",
                    "event_class": "non_local_no_SWR",
                    "target": "t",
                    "delta": delta,
                    "curve_finite": True,
                }
            )
    return pd.DataFrame(rows)


def test_fraction_positive_is_the_median_of_animal_medians_not_a_mean():
    """Session shares: a 0, 1/4, 1 (median 1/4, mean 5/12); b 1/2; c 1, 0
    (a zero delta is not positive). Animal medians 1/4, 1/2, 1/2: their median
    is 1/2, their mean 5/12, and the median over the six sessions 3/8."""
    rows = [
        {"animal": animal, "session": session, "delta": delta}
        for animal, session, deltas in (
            ("a", "a1", [-1.0, -1.0]),
            ("a", "a2", [1.0, -1.0, -1.0, -1.0]),
            ("a", "a3", [1.0]),
            ("b", "b1", [1.0, -1.0]),
            ("c", "c1", [1.0]),
            ("c", "c2", [-1.0, 0.0]),
        )
        for delta in deltas
    ]
    units = pd.DataFrame(rows).assign(model="m", event_class="pooled", target="t")

    row = stats.animal_balanced_fraction_positive(units, blowup=1000.0).iloc[0]

    assert (row.n_sessions, row.n_animals) == (6, 3)
    assert row.frac_pos == pytest.approx(0.5)
    assert (row.frac_pos_min, row.frac_pos_max) == (0.25, 0.5)


def test_fraction_positive_is_animal_balanced_and_excludes_nan_and_blowups():
    out = stats.animal_balanced_fraction_positive(_units(), blowup=1000.0)
    row = out.iloc[0]
    assert row.n_units == 12
    assert row.n_usable == 10
    assert row.n_excluded_blowup == 1
    assert row.n_sessions == 3 and row.n_animals == 2
    # animal a: median(2/3, 1/4) = 11/24; animal b: 2/3; animal-balanced median
    expected_a = np.median([2 / 3, 1 / 4])
    assert row.frac_pos == np.median([expected_a, 2 / 3])
    assert row.frac_pos_min == expected_a and row.frac_pos_max == 2 / 3


def test_population_curve_peak_and_sign_consistency():
    lag = np.linspace(-0.2, 0.2, 5)
    bump = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    curves = {
        ("m", "non_local_no_SWR", "t"): {
            "a1": 2 * bump,
            "a2": 0.5 * bump,
            "b1": -0.5 * bump,
        }
    }
    out = stats.population_curve_summary(curves, lag, {"a1": "a", "a2": "a", "b1": "b"})
    row = out.iloc[0]
    assert row.peak_lag_ms == 0.0
    # unit -> session -> animal: animal a's sessions average first.
    assert row.peak_value == np.mean([np.mean([2, 0.5]), -0.5])
    assert row.sem_at_peak == pytest.approx(np.std([1.25, -0.5], ddof=1) / np.sqrt(2))
    assert row.sessions_same_sign == 2 / 3
    assert row.animals_same_sign == 1 and row.curve_n_animals == 2
    # Lags 0.0 and 0.1 s fall in the 0 to +150 ms window: animal a's mean there
    # is +0.625, b's -0.25, the population's +0.1875.
    assert row.post_onset_animals_same_sign == 1


def test_post_onset_sign_count_reads_the_fixed_window_not_the_peak():
    """Animals agree at the peak lag but not over 0 to +150 ms: the post-onset
    count must not inherit the peak's agreement."""
    lag = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    curves = {
        ("m", "non_local_no_SWR", "t"): {
            # Both animals peak at +0.2 s (outside the window) with one sign...
            "a1": np.array([0.0, 0.0, 0.3, 0.3, 2.0]),
            # ...but b is negative inside the window.
            "b1": np.array([0.0, 0.0, -0.2, -0.2, 1.0]),
        }
    }
    row = stats.population_curve_summary(curves, lag, {"a1": "a", "b1": "b"}).iloc[0]
    assert row.peak_lag_ms == pytest.approx(200.0)
    assert row.animals_same_sign == 2
    assert row.post_onset_mean > 0
    assert row.post_onset_animals_same_sign == 1


def test_summary_band_and_peak_call():
    bump = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    key = ("m", "non_local_no_SWR", "t")
    # Draws r = 0..99 put r/100 - 0.495 at every lag: the band is known.
    draws = (np.arange(100) / 100 - 0.495)[:, None] * np.ones(5)
    curves = {key: {"a1": 0.9 * bump, "a2": 0.7 * bump, "b1": 0.8 * bump}}
    nulls = {key: {"a1": draws, "a2": draws, "b1": draws}}
    animal_of = {"a1": "a", "a2": "a", "b1": "b"}

    out = stats.null_band_summary(curves, nulls, animal_of).iloc[0]

    expected_low, expected_high = np.percentile(draws[:, 0], [2.5, 97.5])
    assert out.band_low_at_peak == pytest.approx(expected_low)
    assert out.band_high_at_peak == pytest.approx(expected_high)
    assert out.peak_outside_band
    assert out.animals_peak_outside_band == 2 and out.n_null_animals == 2
    # animal a: mean(0.9, 0.7) = 0.8; animal b: 0.8 -> population 0.8 at the
    # peak lag, 0.16 over the window.
    assert out.window_mean == pytest.approx(0.16)
    # Max statistic: every draw's largest |curve| is |r/100 - 0.495|, at most
    # 0.505, so the observed 0.8 exceeds all 100 draws.
    assert out.max_abs_value == pytest.approx(0.8)
    assert out.max_abs_null_p95 == pytest.approx(np.percentile(np.abs(draws[:, 0]), 95))
    assert out.max_outside_reference
    assert out.max_p_value == pytest.approx(1 / 101)
    assert not out.window_outside_band
    assert out.n_draws == 100

    inside = stats.null_band_summary(
        {key: dict.fromkeys(animal_of, 0.3 * bump)}, nulls, animal_of
    ).iloc[0]
    assert not inside.peak_outside_band


def test_observed_curve_and_reference_share_the_animal_weighting():
    """Two sessions of animal a and one of b: every quantity weights animals equally."""
    lag = np.linspace(-0.2, 0.2, 5)
    bump = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    key = ("m", "non_local_no_SWR", "t")
    animal_of = {"a1": "a", "a2": "a", "b1": "b"}
    curves = {key: {"a1": 0.6 * bump, "a2": 0.6 * bump, "b1": 0.0 * bump}}
    # Each session's draws are constant; the population draw is 0.3 when the
    # animals are weighted equally and 0.4 when the sessions are.
    nulls = {
        key: {
            "a1": np.full((20, 5), 0.6),
            "a2": np.full((20, 5), 0.6),
            "b1": np.zeros((20, 5)),
        }
    }

    peaks = stats.population_curve_summary(curves, lag, animal_of).iloc[0]
    band = stats.null_band_summary(curves, nulls, animal_of).iloc[0]

    assert peaks.peak_value == pytest.approx(0.3)
    assert band.band_low_at_peak == pytest.approx(0.3)
    assert band.band_high_at_peak == pytest.approx(0.3)
    assert band.max_abs_value == pytest.approx(peaks.peak_value)
    assert band.max_abs_null_p95 == pytest.approx(0.3)
    marginal = {("marginal_t", "non_local_no_SWR", "t"): curves[key]}
    marginal_peaks = stats.population_curve_summary(marginal, lag, animal_of)
    assert marginal_peaks.iloc[0].post_onset_mean == pytest.approx(
        peaks.post_onset_mean
    )


def test_animal_curve_sem_is_over_animals_at_each_lag():
    # Animal means at lag 0: 1, 2, 4 (mean 7/3, sample SD sqrt(7/3)); at lag 1
    # all equal.
    per_animal = {
        "a": np.array([1.0, 5.0]),
        "b": np.array([2.0, 5.0]),
        "c": np.array([4.0, 5.0]),
    }

    sem = stats.animal_curve_sem(per_animal)

    np.testing.assert_allclose(sem, [np.sqrt(7 / 3) / np.sqrt(3), 0.0])
    assert np.isnan(stats.animal_curve_sem({"a": np.array([1.0, 5.0])})).all()


def test_animal_curve_sem_at_the_peak_is_the_summary_sem():
    lag = np.linspace(-0.2, 0.2, 5)
    key = ("m", "non_local_no_SWR", "t")
    bump = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    animal_of = {"a1": "a", "a2": "a", "b1": "b"}
    by_session = {"a1": 0.9 * bump, "a2": 0.5 * bump, "b1": 0.2 * bump}

    peaks = stats.population_curve_summary({key: by_session}, lag, animal_of)
    _, per_animal = stats.animal_balanced_curve(by_session, animal_of)

    # animal a: 0.7, animal b: 0.2 -> SEM 0.25 at the peak lag
    assert peaks.iloc[0].sem_at_peak == pytest.approx(0.25)
    assert stats.animal_curve_sem(per_animal)[2] == pytest.approx(0.25)


def test_shuffled_band_keeps_shared_sessions_and_the_fewest_draws():
    animal_of = {"a1": "a", "a2": "a", "b1": "b"}
    observed = {"a1": np.array([1.0, 0.0]), "b1": np.array([3.0, 0.0])}
    # a1 holds 3 draws, b1 2; a2 has draws but no observed curve.
    nulls = {
        "a1": np.array([[0.0, 0.0], [2.0, 0.0], [100.0, 0.0]]),
        "a2": np.full((2, 2), 50.0),
        "b1": np.array([[0.0, 0.0], [4.0, 0.0]]),
    }

    band = stats.shuffled_band(observed, nulls, animal_of)

    assert band.sessions == ["a1", "b1"]
    assert band.n_draws == 2
    np.testing.assert_allclose(band.observed, [2.0, 0.0])
    # draw 0: (0 + 0) / 2; draw 1: (2 + 4) / 2; a1's third draw is dropped
    np.testing.assert_allclose(band.draws[:, 0], [0.0, 3.0])
    np.testing.assert_allclose(band.low, [0.075, 0.0])
    np.testing.assert_allclose(band.high, [2.925, 0.0])
    assert stats.shuffled_band(observed, {"a2": nulls["a2"]}, animal_of) is None


def test_max_statistic_call_is_the_plus_one_p_not_the_interpolated_percentile():
    key = ("m", "non_local_no_SWR", "t")
    animal_of = {"a1": "a"}
    # 20 draws whose largest |curve| is 0.05, 0.10, ..., 1.00.
    draws = (np.arange(1, 21) / 20)[:, None] * np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    nulls = {key: {"a1": draws}}

    def call(observed_peak):
        curves = {key: {"a1": observed_peak * np.array([0.0, 0.0, 1.0, 0.0, 0.0])}}
        return stats.null_band_summary(curves, nulls, animal_of).iloc[0]

    # 0.97 is above the interpolated 95th percentile (0.9525) but one draw
    # (1.00) is at least as large: p = 2/21, inside.
    near = call(0.97)
    assert near.max_abs_value > near.max_abs_null_p95
    assert near.max_p_value == pytest.approx(2 / 21)
    assert not near.max_outside_reference
    # Above every draw: p = 1/21 <= 0.05, outside.
    above = call(1.2)
    assert above.max_p_value == pytest.approx(1 / 21)
    assert above.max_outside_reference


def test_mark_correlations_are_animal_balanced(tmp_path):
    """Two chimi sessions at r = 0.2 and 0.4 and one wilbur session at 0.9
    give the mean of the animal means, 0.6, not the session mean."""

    def geometry(r):
        return {
            "non_local_no_SWR": {
                "marks": ["a", "b"],
                "correlation": [[1.0, r], [r, 1.0]],
            }
        }

    for session, r in (
        ("chimi20200212_02_r1", 0.2),
        ("chimi20200212_04_r2", 0.4),
        ("wilbur20210329_02_r1", 0.9),
    ):
        nwb, epoch = session.split("_", 1)
        record = {
            "bundle_status": "complete",
            "model_name": "m",
            "nwb_file_name": f"{nwb}_.nwb",
            "epoch_name": epoch,
            "marked_family_diagnostics": {"event_mark_geometry_by_class": geometry(r)},
        }
        path = tmp_path / f"{session}_mPFC_m_marked_family.provenance.json"
        path.write_text(json.dumps(record))

    table = stats.mark_correlations(tmp_path)

    assert table["target"].tolist() == ["a & b"]
    assert table["statistic"].tolist() == ["mark_correlation"]
    assert table["value"].iloc[0] == pytest.approx(0.6)
    assert table["n_sessions"].iloc[0] == 3


def test_complete_bundle_records_keep_only_complete_bundles_in_name_order(tmp_path):
    """A sidecar with another status, or a non-sidecar file, is not a bundle; a
    record with an ``extra`` block is read from it, one without from itself."""
    sidecars = {
        "c_marked_family.provenance.json": {
            "extra": {"bundle_status": "complete", "model_name": "c"}
        },
        "a_marked_family.provenance.json": {
            "bundle_status": "complete",
            "model_name": "a",
        },
        "b_marked_family.provenance.json": {"extra": {"bundle_status": "no_units"}},
        "d_marked_family.provenance.json": {"bundle_status": "ineligible"},
        "e_other.provenance.json": {"bundle_status": "complete"},
    }
    for name, record in sidecars.items():
        (tmp_path / name).write_text(json.dumps(record))

    records = stats.complete_bundle_records(tmp_path)

    assert [path.name for path, _ in records] == [
        "a_marked_family.provenance.json",
        "c_marked_family.provenance.json",
    ]
    assert [extra["model_name"] for _, extra in records] == ["a", "c"]
