"""Per-stop baseline columns of the marked designs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.batched_glm import make_ridge_penalty
from continuum_swr_replay.analysis.pfc.design import event_basis_functions
from continuum_swr_replay.analysis.pfc.marked_glm import (
    EVENT_DURATION_MARK,
    MARKED_EVENT_CLASSES,
    EventKernelBlock,
    KernelDesign,
    build_baseline_columns,
    build_kernel_design,
    build_marked_model_ladder,
    fit_marked_kernel_ladder,
)
from continuum_swr_replay.analysis.pfc.marked_model_family import (
    CURRENT_ARM_FRACTION,
    MarkedFamilySpec,
    build_marked_family_ladder,
    family_ridge_penalty,
    fit_marked_family_ladder,
)

SAMPLING_FREQUENCY = 500.0
STOP_BINS = 150
GAP_BINS = 50


def _stops(wells, outcomes, **context) -> tuple[pd.Series, pd.DataFrame]:
    """Stops of 150 fitted bins separated by 50 unfitted bins.

    Returns the fit mask (fitted only inside a stop) and the cohort stops, with
    any extra per-stop ``context`` columns.
    """
    n_stops = len(wells)
    n_time = n_stops * (STOP_BINS + GAP_BINS)
    time = np.arange(n_time) / SAMPLING_FREQUENCY
    fitted = np.zeros(n_time, dtype=bool)
    starts, ends = [], []
    for index in range(n_stops):
        first = index * (STOP_BINS + GAP_BINS)
        fitted[first : first + STOP_BINS] = True
        starts.append(time[first])
        ends.append(time[first + STOP_BINS - 1])
    stops = pd.DataFrame(
        {
            "stop_start": starts,
            "stop_end": ends,
            "occupancy_well_id": np.asarray(wells, dtype=float),
            "is_rewarded_outcome": np.asarray(outcomes, dtype=float),
            **{
                name: np.asarray(values, dtype=float)
                for name, values in context.items()
            },
        }
    )
    return pd.Series(fitted, index=time), stops


def _stop_of_fitted_bin(fit_mask: pd.Series) -> np.ndarray:
    return np.repeat(np.arange(int(fit_mask.sum()) // STOP_BINS), STOP_BINS)


def test_baseline_columns_map_fitted_bins_to_their_stop():
    fit_mask, stops = _stops(wells=[0, 2, 2], outcomes=[1, 0, np.nan])

    baseline, names, report = build_baseline_columns(fit_mask, stops)

    assert names == ("well_2", "rewarded_outcome", "outcome_unknown")
    assert baseline.shape == (3 * STOP_BINS, 3)
    stop = _stop_of_fitted_bin(fit_mask)
    np.testing.assert_array_equal(baseline[:, 0], np.isin(stop, [1, 2]))
    np.testing.assert_array_equal(baseline[:, 1], stop == 0)
    np.testing.assert_array_equal(baseline[:, 2], stop == 2)
    assert report["baseline_reference_well"] == 0
    assert report["baseline_dropped_constant"] == []
    assert report["n_bins_without_well"] == 0


def test_baseline_reference_falls_back_when_well_0_is_absent():
    fit_mask, stops = _stops(wells=[3, 4, 3], outcomes=[1, 0, np.nan])

    _, names, report = build_baseline_columns(fit_mask, stops)

    assert names == ("well_4", "rewarded_outcome", "outcome_unknown")
    assert report["baseline_reference_well"] == 3


def test_baseline_drops_constant_columns_and_reports_them():
    fit_mask, stops = _stops(wells=[0, 1, 0], outcomes=[1, 1, 1])

    baseline, names, report = build_baseline_columns(fit_mask, stops)

    assert names == ("well_1",)
    assert baseline.shape == (3 * STOP_BINS, 1)
    assert report["baseline_dropped_constant"] == [
        "rewarded_outcome",
        "outcome_unknown",
    ]


def test_baseline_gives_bins_in_a_stop_without_a_well_their_own_column():
    fit_mask, stops = _stops(wells=[0, np.nan, 2], outcomes=[1, 0, 1])

    baseline, names, report = build_baseline_columns(fit_mask, stops)

    assert report["n_bins_without_well"] == STOP_BINS
    # ``well_unknown`` sits right after the well indicators, so the well-less
    # (mixed) stop is not coded like the reference well.
    assert names == ("well_2", "well_unknown", "rewarded_outcome")
    stop = _stop_of_fitted_bin(fit_mask)
    np.testing.assert_array_equal(baseline[:, names.index("well_2")], stop == 2)
    np.testing.assert_array_equal(baseline[:, names.index("well_unknown")], stop == 1)
    assert report["baseline_dropped_constant"] == ["outcome_unknown"]


def test_baseline_adds_no_well_unknown_column_when_every_well_is_known():
    fit_mask, stops = _stops(wells=[0, 1, 2], outcomes=[1, 0, 1])

    _, names, report = build_baseline_columns(fit_mask, stops)

    assert "well_unknown" not in names
    assert "well_unknown" not in report["baseline_dropped_constant"]


def test_baseline_refuses_a_fitted_bin_outside_every_stop():
    fit_mask, stops = _stops(wells=[0, 2, 2], outcomes=[1, 0, np.nan])
    fit_mask.iloc[-1] = True  # past the last stop_end

    with pytest.raises(ValueError, match="1 fitted bins lie outside every cohort stop"):
        build_baseline_columns(fit_mask, stops)


def test_context_baseline_column_is_constant_within_stop():
    trips = [2.0, 1.0, 0.0, np.nan]
    fit_mask, stops = _stops(
        wells=[0, 0, 1, 1], outcomes=[1, 0, 1, 0], trips_to_patch_switch=trips
    )

    baseline, names, report = build_baseline_columns(
        fit_mask, stops, context_columns=("trips_to_patch_switch",)
    )

    assert names[-2:] == ("trips_to_patch_switch", "trips_to_patch_switch_unknown")
    stop = _stop_of_fitted_bin(fit_mask)
    for column in (-2, -1):
        for index in range(4):
            assert np.unique(baseline[stop == index, column]).size == 1
    # z-scored over the fitted bins where it is defined; zero where it is not
    defined = np.repeat([2.0, 1.0, 0.0], STOP_BINS)
    expected = (defined - defined.mean()) / defined.std()
    np.testing.assert_allclose(baseline[stop < 3, -2], expected)
    np.testing.assert_array_equal(baseline[stop == 3, -2], 0.0)
    np.testing.assert_array_equal(baseline[:, -1], stop == 3)
    assert report["baseline_context_scales"]["trips_to_patch_switch"] == {
        "mean": pytest.approx(1.0),
        "std": pytest.approx(np.std([2.0, 1.0, 0.0])),
    }

    # ... and unpenalized in the design that carries it.
    lags = np.arange(-4, 5)
    basis, _, _ = event_basis_functions(lags, SAMPLING_FREQUENCY)
    events = {"a": pd.DataFrame({"start_time": fit_mask.index[[10, 400, 610]]})}
    design = build_kernel_design(
        events,
        fit_mask,
        basis,
        lags,
        (EventKernelBlock("a:event", "a"),),
        baseline=baseline,
    )
    penalty = family_ridge_penalty(design, 3.0, 0.5)
    np.testing.assert_array_equal(penalty[: 1 + len(names)], 0.0)


def _design_inputs():
    fit_mask, stops = _stops(wells=[0, 2, 2, 4], outcomes=[1, 0, np.nan, 1])
    lags = np.arange(-4, 5)
    basis, _, _ = event_basis_functions(lags, SAMPLING_FREQUENCY)
    time = fit_mask.index.to_numpy()
    events = {
        event_class: pd.DataFrame(
            {
                "event_id": 10 * class_index + np.arange(12),
                "start_time": time[20 + 60 * np.arange(12) + class_index],
                EVENT_DURATION_MARK: np.linspace(-3.0, -1.0, 12),
                CURRENT_ARM_FRACTION: np.linspace(0.1, 0.9, 12) ** (class_index + 1),
            }
        )
        for class_index, event_class in enumerate(MARKED_EVENT_CLASSES)
    }
    baseline, names, _ = build_baseline_columns(fit_mask, stops)
    return fit_mask, basis, lags, events, baseline, names


def test_kernel_design_places_first_block_after_the_baseline():
    fit_mask, basis, lags, events, baseline, names = _design_inputs()
    assert baseline.shape[1] == 3 + 1  # well_2, well_4, rewarded, unknown
    blocks = tuple(
        EventKernelBlock(f"{event_class}:event", event_class)
        for event_class in MARKED_EVENT_CLASSES
    )

    design = build_kernel_design(
        events, fit_mask, basis, lags, blocks, baseline=baseline
    )

    n_basis = basis.shape[1]
    assert design.n_baseline == 4
    assert design.block_slices[blocks[0].name] == slice(5, 5 + n_basis)
    np.testing.assert_array_equal(design.matrix[:, 1:5], baseline)
    with pytest.raises(
        ValueError, match="expected a non-empty span starting at column 5"
    ):
        KernelDesign(
            matrix=design.matrix,
            block_slices={
                blocks[0].name: slice(1, 1 + n_basis),
                blocks[1].name: slice(1 + n_basis, 1 + 2 * n_basis),
            },
            blocks=blocks,
            n_baseline=4,
        )
    with pytest.raises(ValueError, match="one row per fitted bin"):
        build_kernel_design(
            events, fit_mask, basis, lags, blocks, baseline=baseline[1:]
        )


def test_without_block_keeps_the_baseline_columns():
    fit_mask, basis, lags, events, baseline, _ = _design_inputs()
    spec = MarkedFamilySpec(
        "baseline_probe", (CURRENT_ARM_FRACTION,), (EVENT_DURATION_MARK,), "probe"
    )
    ladder = build_marked_family_ladder(
        events, fit_mask, basis, lags, spec, baseline=baseline
    )

    for reduced in ladder.reduced.values():
        np.testing.assert_array_equal(reduced.matrix[:, :5], ladder.full.matrix[:, :5])
        assert reduced.n_baseline == 4
        assert min(span.start for span in reduced.block_slices.values()) == 5


def test_penalty_leaves_baseline_columns_free():
    fit_mask, basis, lags, events, baseline, _ = _design_inputs()
    spec = MarkedFamilySpec(
        "baseline_probe", (CURRENT_ARM_FRACTION,), (EVENT_DURATION_MARK,), "probe"
    )
    design = build_marked_family_ladder(
        events, fit_mask, basis, lags, spec, baseline=baseline
    ).full

    penalty = family_ridge_penalty(design, 3.0, 3.0)

    np.testing.assert_array_equal(penalty[:5], 0.0)
    np.testing.assert_allclose(penalty[5:], 1 / 9)
    # The single-mark estimator frees the same columns.
    np.testing.assert_array_equal(
        make_ridge_penalty(design.matrix.shape[1], 3.0, n_unpenalized=5), penalty
    )


def test_empty_baseline_reproduces_the_contract_one_fit():
    fit_mask, basis, lags, events, baseline, _ = _design_inputs()
    empty = np.empty((baseline.shape[0], 0))
    spec = MarkedFamilySpec(
        "baseline_probe", (CURRENT_ARM_FRACTION,), (EVENT_DURATION_MARK,), "probe"
    )
    without = build_marked_family_ladder(events, fit_mask, basis, lags, spec)
    with_empty = build_marked_family_ladder(
        events, fit_mask, basis, lags, spec, baseline=empty
    )
    np.testing.assert_array_equal(without.full.matrix, with_empty.full.matrix)
    single = build_marked_model_ladder(
        events,
        fit_mask,
        basis,
        lags,
        scientific_mark=CURRENT_ARM_FRACTION,
        nuisance_marks=(EVENT_DURATION_MARK,),
        baseline=empty,
    )
    np.testing.assert_array_equal(single.full.matrix, without.full.matrix)

    from continuum_swr_replay.analysis.pfc.heldout import assign_blocked_folds

    rng = np.random.default_rng(5)
    counts = rng.poisson(0.2, size=(without.full.matrix.shape[0], 2)).astype(float)
    folds = assign_blocked_folds(fit_mask, n_folds=2, seed=1, support_radius_bins=4)
    family = fit_marked_family_ladder(
        with_empty, counts, folds, penalty_sigma=3.0, mark_penalty_sigma=None
    )
    reference = fit_marked_kernel_ladder(single, counts, folds, penalty_sigma=3.0)
    np.testing.assert_allclose(
        family.statistics["family_full_heldout_deviance"],
        reference.statistics["marked_full_heldout_deviance"],
        rtol=1e-10,
        atol=1e-8,
    )


def test_baseline_prior_penalizes_only_the_baseline_columns():
    fit_mask, basis, lags, events, baseline, _ = _design_inputs()
    spec = MarkedFamilySpec(
        "baseline_probe", (CURRENT_ARM_FRACTION,), (EVENT_DURATION_MARK,), "probe"
    )
    design = build_marked_family_ladder(
        events, fit_mask, basis, lags, spec, baseline=baseline
    ).full

    penalty = family_ridge_penalty(design, 3.0, 0.5, baseline_penalty_sigma=3.0)

    assert penalty[0] == 0.0
    np.testing.assert_allclose(penalty[1:5], 1 / 9)
    np.testing.assert_array_equal(
        penalty[5:], family_ridge_penalty(design, 3.0, 0.5)[5:]
    )
    with pytest.raises(ValueError):
        family_ridge_penalty(design, 3.0, 0.5, baseline_penalty_sigma=0.0)


def test_baseline_prior_makes_a_unit_silent_at_one_well_converge():
    """No spikes in one well's stops: the free baseline coefficient runs off and
    the fit never converges; under the prior it converges and a unit with
    spikes everywhere barely moves."""
    from continuum_swr_replay.analysis.pfc.batched_glm import fit_poisson_glms

    fit_mask, basis, lags, events, baseline, _ = _design_inputs()
    spec = MarkedFamilySpec(
        "baseline_probe", (CURRENT_ARM_FRACTION,), (EVENT_DURATION_MARK,), "probe"
    )
    design = build_marked_family_ladder(
        events, fit_mask, basis, lags, spec, baseline=baseline
    ).full
    rng = np.random.default_rng(0)
    n_rows = design.matrix.shape[0]
    active = rng.poisson(0.05, size=n_rows).astype(float)
    silent_at_one_well = rng.poisson(0.05, size=n_rows).astype(float)
    # Column 1 is the first non-reference well's indicator.
    silent_at_one_well[design.matrix[:, 1] > 0] = 0.0
    counts = np.column_stack([active, silent_at_one_well])

    free = fit_poisson_glms(
        design.matrix,
        counts,
        penalty=family_ridge_penalty(design, 3.0, 0.5),
        compute_covariance=False,
    )
    prior = fit_poisson_glms(
        design.matrix,
        counts,
        penalty=family_ridge_penalty(design, 3.0, 0.5, baseline_penalty_sigma=3.0),
        compute_covariance=False,
    )

    assert not bool(np.asarray(free.converged)[1])
    assert np.asarray(prior.converged).all()
    assert np.abs(np.asarray(prior.params)[1]).max() < 20
    np.testing.assert_allclose(
        np.asarray(prior.params)[0], np.asarray(free.params)[0], atol=0.05
    )
