"""Value-level tests for mark-modulated mPFC temporal kernels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.design import (
    convolve_event_basis,
    event_basis_functions,
    event_onset_counts,
)
from continuum_swr_replay.analysis.pfc.heldout import assign_blocked_folds
from continuum_swr_replay.analysis.pfc.marked_glm import (
    EVENT_DURATION_MARK,
    MARKED_EVENT_CLASSES,
    MIN_MARKED_EVENTS_PER_CLASS,
    EventKernelBlock,
    KernelDesign,
    MarkedModelLadder,
    _mark_block_geometry,
    attach_current_arm_marks,
    build_kernel_design,
    build_marked_model_ladder,
    event_basis_block,
    fit_marked_kernel_ladder,
    fit_rate_columns,
    permuted_mark_blocks,
    standardize_event_marks,
    weighted_event_onset_counts,
    within_available_mark_permutations,
)

SAMPLING_FREQUENCY = 500.0


def _fit_mask(run_lengths: list[int], gap: int = 80) -> pd.Series:
    pieces = []
    for length in run_lengths:
        pieces.extend([np.ones(length, dtype=bool), np.zeros(gap, dtype=bool)])
    values = np.concatenate(pieces)
    return pd.Series(values, index=np.arange(len(values)) / SAMPLING_FREQUENCY)


def _shuffled_full_design(
    ladder: MarkedModelLadder,
    event_class: str,
    permutation: np.ndarray,
) -> np.ndarray:
    """Scalar reference: one full-session convolution per permuted draw.

    This is the obvious implementation :func:`permuted_mark_blocks` replaces.
    It lives here, not in the module, because production never calls it: it
    exists only to pin the batched sparse-scatter version to an independent
    construction of the same matrix.
    """
    events = ladder.events[event_class]
    permutation = np.asarray(permutation)
    if permutation.shape != (len(events),):
        raise ValueError("permutation must contain one index per event")
    shuffled = events.copy()
    shuffled[ladder.scientific_mark_z] = events[ladder.scientific_mark_z].to_numpy(
        dtype=float
    )[permutation]
    time = ladder.fit_mask.index.to_numpy(dtype=float)
    rows = ladder.fit_mask.to_numpy(dtype=bool)
    shuffled_block = event_basis_block(
        shuffled,
        time,
        ladder.spline_basis,
        ladder.lags,
        mark_column=ladder.scientific_mark_z,
    )[rows]
    design = ladder.full.matrix.copy()
    design[:, ladder.full.block_slices[f"{event_class}:{ladder.scientific_mark}"]] = (
        shuffled_block
    )
    return design


def _small_ladder(onset_bins: dict[str, list[int]] | None = None) -> MarkedModelLadder:
    """A two-class ladder over two short immobility runs separated by a gap."""
    fit_mask = _fit_mask([100, 100], gap=20)
    time = fit_mask.index.to_numpy()
    lags = np.arange(-5, 6)
    basis, _, _ = event_basis_functions(lags, SAMPLING_FREQUENCY)
    if onset_bins is None:
        onset_bins = {
            "non_local_no_SWR": [30, 60, 90, 130, 160, 190],
            "non_local_SWR": [40, 70, 125, 150, 175, 195],
        }
    rng = np.random.default_rng(7)
    events = {}
    for index, event_class in enumerate(MARKED_EVENT_CLASSES):
        bins = onset_bins[event_class]
        events[event_class] = pd.DataFrame(
            {
                "event_id": 100 * index + np.arange(len(bins)),
                "start_time": time[bins],
                "content": rng.uniform(0.0, 1.0, len(bins)),
                "log_duration": rng.normal(size=len(bins)),
            }
        )
    return build_marked_model_ladder(
        events,
        fit_mask,
        basis,
        lags,
        scientific_mark="content",
        nuisance_marks=("log_duration",),
        min_marked_events=2,
    )


@pytest.fixture
def small_ladder() -> MarkedModelLadder:
    return _small_ladder()


def _canonical_events_and_arm_rows():
    classes = {
        "non_local_no_SWR": pd.DataFrame(
            {
                "event_id": [12, 10],
                "start_time": [3.0, 1.0],
                "end_time": [3.08, 1.04],
                "is_swr": [False, False],
            }
        ),
        "non_local_SWR": pd.DataFrame(
            {
                "event_id": [11],
                "start_time": [2.0],
                "end_time": [2.06],
                "is_swr": [True],
            }
        ),
    }
    rows = []
    values = {10: (0.2, 0.3), 11: (0.6, 0.55), 12: (0.9, 0.8)}
    timing = {10: (1.0, 1.04, False), 11: (2.0, 2.06, True), 12: (3.0, 3.08, False)}
    for event_id in (10, 11, 12):
        start, end, is_swr = timing[event_id]
        duration = end - start + 1 / SAMPLING_FREQUENCY
        hard, posterior = values[event_id]
        rows.extend(
            [
                {
                    "event_id": event_id,
                    "is_current_arm": True,
                    "is_swr": is_swr,
                    "event_start_s": start,
                    "event_end_s": end,
                    "event_duration_s": duration,
                    "occupancy_state": "at_well",
                    "at_well": True,
                    "occupancy_well_arm": 2,
                    "arm_fraction": hard,
                    "arm_fraction_posterior": posterior,
                },
                {
                    "event_id": event_id,
                    "is_current_arm": False,
                    "is_swr": is_swr,
                    "event_start_s": start,
                    "event_end_s": end,
                    "event_duration_s": duration,
                    "occupancy_state": "at_well",
                    "at_well": True,
                    "occupancy_well_arm": 2,
                    "arm_fraction": 1.0 - hard,
                    "arm_fraction_posterior": 1.0 - posterior,
                },
            ]
        )
    return classes, pd.DataFrame(rows)


def test_current_arm_marks_join_one_to_one_without_changing_event_order():
    classes, arm_rows = _canonical_events_and_arm_rows()

    marked = attach_current_arm_marks(classes, arm_rows)

    assert marked["non_local_no_SWR"].event_id.tolist() == [12, 10]
    assert marked["non_local_no_SWR"].current_arm_fraction.tolist() == [0.9, 0.2]
    assert marked["non_local_SWR"].current_arm_fraction_posterior.tolist() == [0.55]
    assert marked["non_local_no_SWR"][EVENT_DURATION_MARK].tolist() == pytest.approx(
        np.log([0.082, 0.042])
    )
    assert "current_arm_fraction" not in classes["non_local_no_SWR"]


def test_current_arm_marks_fail_on_duplicate_missing_or_misaligned_rows():
    classes, arm_rows = _canonical_events_and_arm_rows()
    duplicate = pd.concat(
        [arm_rows, arm_rows[(arm_rows.event_id == 10) & arm_rows.is_current_arm]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="more than one current-arm"):
        attach_current_arm_marks(classes, duplicate)

    missing = arm_rows[arm_rows.event_id != 11]
    with pytest.raises(ValueError, match="no event-arm rows"):
        attach_current_arm_marks(classes, missing)

    wrong_class = arm_rows.copy()
    wrong_class.loc[wrong_class.event_id == 11, "is_swr"] = False
    with pytest.raises(ValueError, match="labels disagree"):
        attach_current_arm_marks(classes, wrong_class)

    wrong_time = arm_rows.copy()
    wrong_time.loc[wrong_time.event_id == 12, "event_start_s"] += 0.01
    with pytest.raises(ValueError, match="timing column"):
        attach_current_arm_marks(classes, wrong_time)


def test_current_arm_marks_fail_on_partially_available_content_columns():
    classes, arm_rows = _canonical_events_and_arm_rows()
    partial = arm_rows.copy()
    partial.loc[
        (partial.event_id == 11) & partial.is_current_arm, "arm_fraction_posterior"
    ] = np.nan

    with pytest.raises(ValueError, match="only partially available"):
        attach_current_arm_marks(classes, partial)


def test_current_arm_marks_fail_when_an_occupied_arm_has_no_content_row():
    classes, arm_rows = _canonical_events_and_arm_rows()
    # The occupancy arm stays defined, so missing content is a table defect,
    # not a genuinely undefined predictor.
    dropped = arm_rows[~((arm_rows.event_id == 11) & arm_rows.is_current_arm)]

    with pytest.raises(ValueError, match="defined occupancy well arm"):
        attach_current_arm_marks(classes, dropped)


def test_current_arm_marks_fail_on_non_positive_event_duration():
    classes, arm_rows = _canonical_events_and_arm_rows()
    collapsed = {name: frame.copy() for name, frame in classes.items()}
    collapsed["non_local_no_SWR"].loc[
        collapsed["non_local_no_SWR"].event_id == 12, "end_time"
    ] = 3.0
    zero_duration = arm_rows.copy()
    zero_duration.loc[zero_duration.event_id == 12, "event_end_s"] = 3.0
    zero_duration.loc[zero_duration.event_id == 12, "event_duration_s"] = 0.0

    with pytest.raises(ValueError, match="non-positive durations"):
        attach_current_arm_marks(collapsed, zero_duration)


def test_current_arm_marks_fail_on_a_missing_is_swr_label():
    classes, arm_rows = _canonical_events_and_arm_rows()
    na_swr = arm_rows.copy()
    na_swr["is_swr"] = na_swr["is_swr"].astype(object)
    na_swr.loc[na_swr.event_id == 11, "is_swr"] = np.nan

    with pytest.raises(ValueError, match="missing event-arm is_swr"):
        attach_current_arm_marks(classes, na_swr)


def test_event_without_an_occupancy_arm_stays_in_base_population():
    classes, arm_rows = _canonical_events_and_arm_rows()
    undefined = arm_rows[~((arm_rows.event_id == 11) & arm_rows.is_current_arm)].copy()
    undefined["is_current_arm"] = undefined["is_current_arm"].astype(object)
    undefined.loc[undefined.event_id == 11, "is_current_arm"] = np.nan
    undefined.loc[undefined.event_id == 11, "occupancy_state"] = "mixed"
    undefined.loc[undefined.event_id == 11, "at_well"] = False
    undefined.loc[undefined.event_id == 11, "occupancy_well_arm"] = np.nan

    marked = attach_current_arm_marks(classes, undefined)
    swr_event = marked["non_local_SWR"].iloc[0]

    assert np.isnan(swr_event.current_arm_fraction)
    assert "current_arm_mark_available" not in marked["non_local_SWR"]
    assert "current_arm_mark_missing_reason" not in marked["non_local_SWR"]

    # The availability flag has exactly one source: standardization.
    standardized, _ = standardize_event_marks(
        pd.DataFrame(
            {
                "event_id": [11, 13, 14],
                "current_arm_fraction": [swr_event.current_arm_fraction, 0.4, 0.8],
            }
        ),
        ("current_arm_fraction",),
        allow_missing=("current_arm_fraction",),
        min_marked_events=2,
    )

    assert not standardized.current_arm_fraction_available.iloc[0]
    assert standardized.current_arm_fraction_z.iloc[0] == 0.0


def test_explicitly_constant_nuisance_standardizes_to_a_zero_block():
    events = pd.DataFrame({"event_id": [1, 2, 3], "context": [1.0, 1.0, 1.0]})

    standardized, scales = standardize_event_marks(
        events,
        ("context",),
        allow_constant=("context",),
        min_marked_events=2,
    )

    np.testing.assert_array_equal(standardized.context_z, 0.0)
    assert scales["context"].std == 0.0
    with pytest.raises(ValueError, match="no within-class variation"):
        standardize_event_marks(events, ("context",), min_marked_events=2)


def test_mark_permutations_never_move_undefined_events():
    available = np.array([True, False, True, True, False])

    permutations = within_available_mark_permutations(
        np.random.default_rng(31), available, n_draws=20
    )

    np.testing.assert_array_equal(permutations[:, ~available], [[1, 4]] * 20)
    for permutation in permutations:
        np.testing.assert_array_equal(
            np.sort(permutation[available]), np.flatnonzero(available)
        )


def test_mark_permutation_stream_is_batch_and_prefix_invariant():
    available = np.array([True, False, True, True, False, True])
    whole = within_available_mark_permutations(
        np.random.default_rng(91), available, n_draws=11
    )
    batched_rng = np.random.default_rng(91)
    batched = np.vstack(
        [
            within_available_mark_permutations(batched_rng, available, n_draws=4),
            within_available_mark_permutations(batched_rng, available, n_draws=7),
        ]
    )
    prefix = within_available_mark_permutations(
        np.random.default_rng(91), available, n_draws=6
    )

    np.testing.assert_array_equal(batched, whole)
    np.testing.assert_array_equal(prefix, whole[:6])


def test_unit_weights_reproduce_the_ordinary_onset_train_exactly():
    time = np.arange(20, dtype=float) / SAMPLING_FREQUENCY
    events = pd.DataFrame({"start_time": [time[2], time[2], time[10], time[-1]]})

    ordinary = event_onset_counts(events, time)
    weighted = weighted_event_onset_counts(events, time, np.ones(len(events)))

    np.testing.assert_array_equal(weighted, ordinary)


def test_marked_convolution_matches_a_slow_event_by_event_reference():
    time = np.arange(40, dtype=float) / SAMPLING_FREQUENCY
    lags = np.arange(-4, 5)
    basis, _, _ = event_basis_functions(lags, SAMPLING_FREQUENCY)
    events = pd.DataFrame(
        {
            "start_time": time[[1, 8, 8, 38]],
            "mark_z": [-1.25, 0.5, 2.0, -0.75],
        }
    )

    actual = event_basis_block(events, time, basis, lags, mark_column="mark_z")
    expected = np.zeros_like(actual)
    onset_bins = np.digitize(events.start_time.to_numpy(), time[1:])
    for onset, weight in zip(onset_bins, events.mark_z, strict=True):
        positions = onset + lags
        valid = (positions >= 0) & (positions < len(time))
        expected[positions[valid]] += weight * basis[valid]

    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_standardization_is_within_class_and_fails_on_unidentified_marks():
    events = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4],
            "start_time": [0.1, 0.2, 0.3, 0.4],
            "content": [0.0, 0.25, 0.75, 1.0],
        }
    )

    standardized, reference = standardize_event_marks(
        events, ("content",), min_marked_events=2
    )

    assert standardized.content_z.mean() == pytest.approx(0.0, abs=1e-15)
    assert standardized.content_z.std(ddof=0) == pytest.approx(1.0)
    assert reference["content"].mean == pytest.approx(0.5)
    assert reference["content"].n_events == 4
    assert reference["content"].n_missing == 0
    assert "content_z" not in events

    with pytest.raises(ValueError, match="no within-class variation"):
        standardize_event_marks(
            events.assign(content=1.0), ("content",), min_marked_events=2
        )
    with pytest.raises(ValueError, match="must be finite"):
        standardize_event_marks(
            events.assign(content=[0.0, 0.5, np.nan, 1.0]),
            ("content",),
            min_marked_events=2,
        )
    with pytest.raises(ValueError, match="must be finite"):
        standardize_event_marks(
            events.assign(content=[0.0, 0.5, "bad", 1.0]),
            ("content",),
            allow_missing=("content",),
            min_marked_events=2,
        )

    missing, missing_reference = standardize_event_marks(
        events.assign(content=[0.0, 0.5, np.nan, 1.0]),
        ("content",),
        allow_missing=("content",),
        min_marked_events=2,
    )
    assert missing.content_z.iloc[2] == 0.0
    assert not missing.content_available.iloc[2]
    assert missing_reference["content"].n_events == 3
    assert missing_reference["content"].n_missing == 1


def test_too_few_marked_events_are_refused_by_default_and_named_by_class():
    events = pd.DataFrame(
        {
            "event_id": np.arange(4),
            "start_time": np.linspace(0.1, 0.4, 4),
            "content": [0.0, 0.25, 0.75, 1.0],
        }
    )

    assert MIN_MARKED_EVENTS_PER_CLASS == 10
    with pytest.raises(ValueError, match=r"only 4 finite events; at least 10"):
        standardize_event_marks(events, ("content",))
    with pytest.raises(ValueError, match="min_marked_events must be at least two"):
        standardize_event_marks(events, ("content",), min_marked_events=1)

    fit_mask = _fit_mask([100, 100], gap=20)
    time = fit_mask.index.to_numpy()
    lags = np.arange(-5, 6)
    basis, _, _ = event_basis_functions(lags, SAMPLING_FREQUENCY)
    classes = {
        name: pd.DataFrame(
            {
                "event_id": 100 * index + np.arange(4),
                "start_time": time[[30, 60, 130, 160]],
                "content": [0.0, 0.25, 0.75, 1.0],
            }
        )
        for index, name in enumerate(MARKED_EVENT_CLASSES)
    }

    with pytest.raises(
        ValueError,
        match=r"'non_local_no_SWR': event mark 'content' has only 4 finite events",
    ):
        build_marked_model_ladder(
            classes, fit_mask, basis, lags, scientific_mark="content"
        )


def test_kernel_design_refuses_block_spans_that_do_not_tile_the_columns():
    blocks = (EventKernelBlock("a", "class"), EventKernelBlock("b", "class"))
    matrix = np.ones((4, 5))

    with pytest.raises(ValueError, match="name the blocks in column order"):
        KernelDesign(matrix, {"b": slice(1, 3), "a": slice(3, 5)}, blocks)
    with pytest.raises(ValueError, match="expected a non-empty span starting at"):
        # Column 0 is the unpenalised intercept and belongs to no block.
        KernelDesign(matrix, {"a": slice(0, 2), "b": slice(2, 5)}, blocks)
    with pytest.raises(ValueError, match="expected a non-empty span starting at"):
        KernelDesign(matrix, {"a": slice(1, 3), "b": slice(4, 5)}, blocks)
    with pytest.raises(ValueError, match="but the design has 5 columns"):
        KernelDesign(matrix, {"a": slice(1, 2), "b": slice(2, 4)}, blocks)


def test_ladder_has_exact_nested_column_order_and_separate_class_scales():
    fit_mask = _fit_mask([100, 100], gap=20)
    time = fit_mask.index.to_numpy()
    lags = np.arange(-5, 6)
    basis, _, _ = event_basis_functions(lags, SAMPLING_FREQUENCY)
    events = {
        "non_local_no_SWR": pd.DataFrame(
            {
                "event_id": [1, 2, 3],
                "start_time": time[[30, 150, 230]],
                "content": [0.0, 0.2, 1.0],
                "log_duration": [0.0, 0.4, 0.7],
            }
        ),
        "non_local_SWR": pd.DataFrame(
            {
                "event_id": [4, 5, 6],
                "start_time": time[[50, 170, 225]],
                "content": [0.4, 0.6, 0.8],
                "log_duration": [0.2, 0.9, 1.1],
            }
        ),
    }

    ladder = build_marked_model_ladder(
        events,
        fit_mask,
        basis,
        lags,
        scientific_mark="content",
        nuisance_marks=("log_duration",),
        min_marked_events=2,
    )

    n_basis = basis.shape[1]
    assert ladder.classes == MARKED_EVENT_CLASSES
    assert tuple(ladder.reduced) == ladder.classes
    assert ladder.scientific_mark_z == "content_z"
    assert ladder.full.matrix.shape == (int(fit_mask.sum()), 1 + 6 * n_basis)
    for event_class in MARKED_EVENT_CLASSES:
        reduced = ladder.reduced[event_class]
        assert reduced.matrix.shape[1] + n_basis == ladder.full.matrix.shape[1]
        dropped = ladder.full.block_slices[f"{event_class}:content"]
        np.testing.assert_array_equal(
            reduced.matrix,
            np.delete(ladder.full.matrix, np.s_[dropped], axis=1),
        )
    assert ladder.scales["non_local_no_SWR"]["content"].mean == pytest.approx(0.4)
    assert ladder.scales["non_local_SWR"]["content"].mean == pytest.approx(0.6)


def test_batched_permuted_blocks_match_the_scalar_reference_at_the_boundaries():
    # Onsets 2 and 98 straddle the axis start and the first fit-mask gap: their
    # +/- 5-bin windows spill outside the fitted rows on either side.
    ladder = _small_ladder(
        onset_bins={
            "non_local_no_SWR": [2, 98, 60, 130, 160, 190],
            "non_local_SWR": [40, 70, 125, 150, 175, 195],
        }
    )
    geometry_class = "non_local_no_SWR"
    permutations = np.array(
        [[2, 0, 1, 5, 3, 4], [1, 2, 0, 4, 5, 3], [5, 4, 3, 2, 1, 0]]
    )

    batched = permuted_mark_blocks(ladder, geometry_class, permutations)
    mark_slice = ladder.full.block_slices[f"{geometry_class}:content"]
    scalar = np.stack(
        [
            _shuffled_full_design(ladder, geometry_class, permutation)[:, mark_slice]
            for permutation in permutations
        ]
    )

    assert np.count_nonzero(batched) > 0
    np.testing.assert_allclose(batched, scalar, atol=1e-12, rtol=1e-12)


def test_permuted_block_rebuild_is_ladder_agnostic():
    from continuum_swr_replay.analysis.pfc.marked_glm import (
        mark_block_geometry,
        mark_blocks_from_weights,
    )

    ladder = _small_ladder(
        onset_bins={
            "non_local_no_SWR": [2, 98, 60, 130, 160, 190],
            "non_local_SWR": [40, 70, 125, 150, 175, 195],
        }
    )
    event_class = "non_local_no_SWR"
    permutations = np.array([[2, 0, 1, 5, 3, 4], [5, 4, 3, 2, 1, 0]])
    events = ladder.events[event_class]
    geometry = mark_block_geometry(events, ladder.fit_mask, ladder.lags, event_class)

    explicit = mark_blocks_from_weights(
        events["content_z"].to_numpy()[permutations], geometry, ladder.spline_basis
    )

    mark_slice = ladder.full.block_slices[f"{event_class}:content"]
    scalar = np.stack(
        [
            _shuffled_full_design(ladder, event_class, permutation)[:, mark_slice]
            for permutation in permutations
        ]
    )
    np.testing.assert_allclose(explicit, scalar, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(
        explicit, permuted_mark_blocks(ladder, event_class, permutations)
    )


def test_permuted_blocks_refuse_a_geometry_built_for_another_class(small_ladder):
    other = _mark_block_geometry(small_ladder, "non_local_SWR")

    with pytest.raises(ValueError, match="built for class 'non_local_SWR'"):
        permuted_mark_blocks(
            small_ladder,
            "non_local_no_SWR",
            np.arange(6)[None, :],
            geometry=other,
        )


def test_block_builder_refuses_duplicate_names_and_missing_classes():
    fit_mask = _fit_mask([100, 100], gap=20)
    basis, _, _ = event_basis_functions(np.arange(-5, 6), SAMPLING_FREQUENCY)
    events = {
        "a": pd.DataFrame({"start_time": [0.1], "mark": [1.0]}),
    }

    duplicate = (
        EventKernelBlock("same", "a"),
        EventKernelBlock("same", "a", "mark"),
    )
    with pytest.raises(ValueError, match="unique"):
        build_kernel_design(events, fit_mask, basis, np.arange(-5, 6), duplicate)
    with pytest.raises(KeyError, match="missing"):
        build_kernel_design(
            events,
            fit_mask,
            basis,
            np.arange(-5, 6),
            (EventKernelBlock("other", "absent"),),
        )


def _small_fit(ladder: MarkedModelLadder, **kwargs):
    rng = np.random.default_rng(5)
    counts = rng.poisson(0.3, size=(ladder.full.matrix.shape[0], 2)).astype(float)
    folds = assign_blocked_folds(
        ladder.fit_mask,
        n_folds=2,
        seed=3,
        support_radius_bins=int(np.abs(ladder.lags).max()),
    )
    return fit_marked_kernel_ladder(ladder, counts, folds, penalty_sigma=3.0, **kwargs)


def test_shuffle_reference_is_seeded_and_batch_size_invariant(small_ladder):
    reference = _small_fit(
        small_ladder, n_shuffles=4, shuffle_seed=11, design_batch_size=2
    )
    repeated = _small_fit(
        small_ladder, n_shuffles=4, shuffle_seed=11, design_batch_size=2
    )
    rebatched = _small_fit(
        small_ladder, n_shuffles=4, shuffle_seed=11, design_batch_size=3
    )
    reseeded = _small_fit(
        small_ladder, n_shuffles=4, shuffle_seed=12, design_batch_size=2
    )

    for event_class in MARKED_EVENT_CLASSES:
        np.testing.assert_array_equal(
            reference.randomization_reference[event_class],
            repeated.randomization_reference[event_class],
        )
        np.testing.assert_allclose(
            reference.randomization_reference[event_class],
            rebatched.randomization_reference[event_class],
            atol=1e-8,
            rtol=0.0,
        )
    assert not np.allclose(
        reference.randomization_reference["non_local_no_SWR"],
        reseeded.randomization_reference["non_local_no_SWR"],
    )


def test_diagnostics_expose_nan_channels_and_one_shared_full_deviance(small_ladder):
    result = _small_fit(small_ladder, n_shuffles=3, shuffle_seed=4, design_batch_size=2)
    no_shuffles = _small_fit(small_ladder, n_shuffles=0, shuffle_seed=4)

    diagnostics = result.diagnostics
    assert isinstance(diagnostics["n_units_full_fit_not_converged"], int)
    for key in (
        "n_units_nan_heldout_delta_by_class",
        "n_units_nan_tail_fraction_by_class",
        "n_marked_events_by_class",
        "n_distinct_permutations_by_class",
    ):
        assert set(diagnostics[key]) == set(MARKED_EVENT_CLASSES)
        assert all(isinstance(value, int) for value in diagnostics[key].values())
    assert diagnostics["n_marked_events_by_class"] == dict.fromkeys(
        MARKED_EVENT_CLASSES, 6
    )
    assert diagnostics["n_distinct_permutations_by_class"] == dict.fromkeys(
        MARKED_EVENT_CLASSES, 720
    )
    assert no_shuffles.diagnostics[
        "n_units_nan_tail_fraction_by_class"
    ] == dict.fromkeys(MARKED_EVENT_CLASSES, 0)

    assert "marked_full_heldout_deviance" in result.statistics
    for event_class in MARKED_EVENT_CLASSES:
        assert f"{event_class}: marked_full_heldout_deviance" not in result.statistics
        assert f"{event_class}: marked_reduced_heldout_deviance" in result.statistics
        assert f"{event_class}: marked_heldout_delta_dev" in result.statistics


@pytest.mark.slow
def test_synthetic_mark_kernel_is_recovered_and_improves_heldout_prediction():
    rng = np.random.default_rng(20260918)
    fit_mask = _fit_mask([1200] * 10, gap=120)
    time = fit_mask.index.to_numpy()
    fit_rows = fit_mask.to_numpy()
    lags = np.arange(-30, 31)
    basis, lag_time, _ = event_basis_functions(lags, SAMPLING_FREQUENCY)

    usable_bins = np.flatnonzero(fit_rows)
    onset_bins = usable_bins[200:-200:180]
    classes = np.resize(np.array(MARKED_EVENT_CLASSES), len(onset_bins))
    marks = rng.uniform(0.0, 1.0, len(onset_bins))
    durations = rng.uniform(0.02, 0.2, len(onset_bins))
    events = {}
    for event_class in MARKED_EVENT_CLASSES:
        keep = classes == event_class
        events[event_class] = pd.DataFrame(
            {
                "event_id": np.flatnonzero(keep),
                "start_time": time[onset_bins[keep]],
                "content": marks[keep],
                "log_duration": np.log(durations[keep]),
            }
        )

    # Ground truth, built without any of the module's design machinery: a
    # constant baseline plus, at every driven-class onset, that event's own
    # standardized mark times a fixed peri-event curve. The curve is the basis
    # projection of a Gaussian only so that the truth is representable at all;
    # nothing here consults the fitted design.
    driven_class = "non_local_no_SWR"
    gaussian = np.exp(-0.5 * ((lag_time - 0.025) / 0.025) ** 2)
    planted_curve = basis @ np.linalg.lstsq(basis, gaussian, rcond=None)[0]
    planted_curve *= 0.9 / planted_curve.max()
    peak_lag_index = int(np.argmax(planted_curve))

    driven_marks = events[driven_class]["content"].to_numpy(dtype=float)
    finite = np.isfinite(driven_marks)
    standardized = (driven_marks - driven_marks[finite].mean()) / driven_marks[
        finite
    ].std(ddof=0)
    log_rate = np.full(len(time), np.log(25.0 / SAMPLING_FREQUENCY))
    for onset_bin, mark_z in zip(
        onset_bins[classes == driven_class], standardized, strict=True
    ):
        positions = np.clip(onset_bin + lags, 0, len(time) - 1)
        np.add.at(log_rate, positions, mark_z * planted_curve)

    driven = rng.poisson(np.exp(log_rate))[fit_rows]
    flat = rng.poisson(25.0 / SAMPLING_FREQUENCY, size=len(driven))
    counts = np.column_stack([driven, flat, np.zeros(len(driven))]).astype(float)
    folds = assign_blocked_folds(
        fit_mask,
        n_folds=5,
        seed=12,
        support_radius_bins=int(np.abs(lags).max()),
    )

    ladder = build_marked_model_ladder(
        events,
        fit_mask,
        basis,
        lags,
        scientific_mark="content",
        nuisance_marks=("log_duration",),
    )
    result = fit_marked_kernel_ladder(
        ladder,
        counts,
        folds,
        penalty_sigma=3.0,
        n_shuffles=4,
        shuffle_seed=91,
        design_batch_size=2,
    )

    delta = result.statistics[f"{driven_class}: marked_heldout_delta_dev"].to_numpy()
    other_delta = result.statistics[
        "non_local_SWR: marked_heldout_delta_dev"
    ].to_numpy()
    recovered = result.mark_curves[driven_class]

    assert delta[0] > 0
    assert recovered[peak_lag_index, 0] == pytest.approx(0.9, abs=0.35)
    # Negative controls: an unmodulated unit buys essentially nothing from the
    # driven class's content and carries a near-flat content curve, and the
    # driven unit buys nothing from the class whose content was never planted.
    # Measured here: the flat unit's gain is 0.34 deviance units against the
    # driven unit's 24.0, and its content curve peaks at 0.52 against a planted
    # 0.9; the driven unit's unplanted-class gain is -1.8.
    assert delta[1] < 1.0
    assert np.nanmax(np.abs(recovered[:, 1])) < 0.6
    assert other_delta[0] <= 0

    reference = result.randomization_reference[driven_class]
    tail = result.statistics[
        f"{driven_class}: marked_label_shuffle_tail_fraction"
    ].to_numpy()
    expected_tail = (1 + (reference >= delta[None, :]).sum(axis=0)) / (
        1 + np.isfinite(reference).sum(axis=0)
    )
    assert reference.shape == (4, 3)
    np.testing.assert_allclose(tail[:2], expected_tail[:2])
    # The all-zero unit has no finite shuffled fit, so it falls below the
    # usable-draw floor rather than being given a plus-one denominator of one.
    assert result.statistics[f"{driven_class}: n_usable_mark_shuffles"].iloc[2] == 0
    assert np.isnan(tail[2])
    assert result.diagnostics["n_shuffles"] == 4
    assert (
        result.diagnostics["shuffled_design_rank_by_class"][driven_class][
            "n_draws_different_from_observed_full"
        ]
        == 0
    )


def test_fit_rate_columns_are_spikes_and_mean_rate_on_the_fitted_bins():
    """Four fitted bins at 500 Hz: 3 spikes are 375 Hz, none is 0 Hz."""
    counts = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0], [1.0, 0.0]])

    columns = fit_rate_columns(counts, 500.0)

    np.testing.assert_array_equal(columns["fit_spike_count"], [3, 0])
    assert columns["fit_spike_count"].dtype.kind == "i"
    np.testing.assert_allclose(columns["fit_firing_rate_hz"], [375.0, 0.0])
    assert list(columns) == ["fit_spike_count", "fit_firing_rate_hz"]


def test_ordinary_and_unit_weighted_basis_blocks_are_identical():
    time = np.arange(300) / SAMPLING_FREQUENCY
    events = pd.DataFrame({"start_time": time[[20, 120, 220]], "one": np.ones(3)})
    lags = np.arange(-10, 11)
    basis, _, _ = event_basis_functions(lags, SAMPLING_FREQUENCY)

    ordinary = convolve_event_basis(event_onset_counts(events, time), basis, lags)
    weighted = event_basis_block(events, time, basis, lags, mark_column="one")

    np.testing.assert_array_equal(weighted, ordinary)


# ---------------------------------------------------------------------------
# Real data: the marked context is the canonical session assembly
# ---------------------------------------------------------------------------

#: The mPFC session with the fewest units among the canonical bundles (74 units).
REAL_SESSION = ("j1620210710_.nwb", "08_r4")


class _StopBeforeFit(Exception):
    """Raised in place of the unit fits, once ``run_session`` has handed over."""


@pytest.mark.slow
@pytest.mark.real_data
def test_marked_context_matches_canonical_assembly(monkeypatch):
    """The marked fit mask and replay onsets are the canonical ones, exactly.

    ``run_session`` is stopped where the unit fits would start, after it has
    filled ``scoring_out`` with the fit mask, counts and event sets the
    canonical fit uses; no model is fit. Needs session inputs on disk and a
    reachable Spyglass, so it is opt-in (``-m real_data``).
    """
    from continuum_swr_replay.analysis.pfc import session
    from continuum_swr_replay.analysis.pfc.cohort import load_session_stop_cohort
    from continuum_swr_replay.analysis.pfc.marked_glm import (
        prepare_marked_session_context,
    )

    nwb_file_name, epoch_name = REAL_SESSION
    cohort_stops, _ = load_session_stop_cohort(
        nwb_file_name, epoch_name, cohort="at_well"
    )
    context = prepare_marked_session_context(
        nwb_file_name,
        epoch_name,
        sampling_frequency=SAMPLING_FREQUENCY,
        cohort_stops=cohort_stops,
    )

    def stop_before_fit(*args, **kwargs):
        raise _StopBeforeFit

    monkeypatch.setattr(session, "_fit_session_neurons", stop_before_fit)
    canonical: dict = {}
    with pytest.raises(_StopBeforeFit):
        session.run_session(
            nwb_file_name,
            epoch_name,
            sampling_frequency=SAMPLING_FREQUENCY,
            run_parallel=False,
            stop_cohort="at_well",
            cohort_stops=cohort_stops,
            scoring_out=canonical,
        )

    pd.testing.assert_series_equal(context.fit_mask, canonical["fit_mask"])
    fit = np.asarray(canonical["fit_mask"], dtype=bool)
    np.testing.assert_array_equal(context.counts, canonical["counts"][fit])
    np.testing.assert_array_equal(context.lags, canonical["lags"])
    for event_class in MARKED_EVENT_CLASSES:
        np.testing.assert_array_equal(
            context.events[event_class]["start_time"].to_numpy(),
            canonical["event_times"][event_class]["start_time"].to_numpy(),
        )
    assert context.cohort_info["n_bins_fit"] == int(fit.sum())
    assert "n_artifact_dropped_events" in context.cohort_info
    assert context.input_provenance["swr_events_sha256"] is not None


@pytest.mark.real_data
def test_event_ripple_power_rule_reproduces_the_event_table():
    """The SWRs-without-replay ripple power uses the event table's own rule.

    ``event_mean_ripple_power`` on the cached decode frame, given the
    non-local events' sampled bounds, must return the event table's
    ``ripple_power_z`` for every event of the session. CPU only; reads the
    event table and one decode frame (opt-in, ``-m real_data``).
    """
    from continuum_swr_replay.analysis.pfc.canonical_events import (
        load_session_event_ripple_power,
    )
    from continuum_swr_replay.decoding.identifiers import parse_nwb_filename
    from continuum_swr_replay.paths import table_path
    from continuum_swr_replay.tables.io import read_table

    nwb_file_name, epoch_name = REAL_SESSION
    animal, session_date = parse_nwb_filename(nwb_file_name)
    table = read_table(table_path("event_table"))
    session = table.loc[
        table.animal.eq(animal)
        & table.session_date.eq(session_date)
        & table.epoch.eq(epoch_name)
    ]
    assert len(session) > 0
    events = pd.DataFrame(
        {
            "start_time": session["event_start_s"].to_numpy(dtype=float),
            "end_time": session["event_end_s"].to_numpy(dtype=float),
        }
    )

    ripple_power, _ = load_session_event_ripple_power(nwb_file_name, epoch_name, events)

    np.testing.assert_allclose(
        ripple_power,
        session["ripple_power_z"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.slow
@pytest.mark.real_data
def test_extended_marked_context_carries_swrs_without_replay():
    """The extended context's SWRs without replay carry duration and power."""
    from continuum_swr_replay.analysis.pfc.canonical_events import (
        load_session_event_ripple_power,
    )
    from continuum_swr_replay.analysis.pfc.cohort import load_session_stop_cohort
    from continuum_swr_replay.analysis.pfc.marked_glm import (
        EXTENDED_MARKED_EVENT_CLASSES,
        RIPPLE_POWER_MARK,
        prepare_marked_session_context,
    )

    nwb_file_name, epoch_name = REAL_SESSION
    cohort_stops, _ = load_session_stop_cohort(
        nwb_file_name, epoch_name, cohort="at_well"
    )
    context = prepare_marked_session_context(
        nwb_file_name,
        epoch_name,
        sampling_frequency=SAMPLING_FREQUENCY,
        cohort_stops=cohort_stops,
        event_set="with_swr_without_replay",
    )

    swrs = context.events[EXTENDED_MARKED_EVENT_CLASSES[-1]]
    assert set(EXTENDED_MARKED_EVENT_CLASSES) <= set(context.events)
    assert len(swrs) == context.cohort_info["n_swr_no_non_local"] > 0
    assert np.isfinite(swrs[EVENT_DURATION_MARK]).all()
    expected, _ = load_session_event_ripple_power(nwb_file_name, epoch_name, swrs)
    np.testing.assert_array_equal(swrs[RIPPLE_POWER_MARK].to_numpy(), expected)
    assert np.isfinite(expected).all()


def test_event_mean_ripple_power_averages_each_events_own_bins():
    from continuum_swr_replay.analysis.pfc.canonical_events import (
        event_mean_ripple_power,
    )

    time = np.arange(20) / SAMPLING_FREQUENCY
    power = np.arange(20, dtype=float)
    power[6] = np.nan  # a gap bin is skipped, not zero-filled
    frame = pd.DataFrame({"SWR_zscore": power}, index=time)

    # Events out of time order; bounds are sampled bins, both included.
    means = event_mean_ripple_power(
        frame,
        start_time=time[[10, 2, 15]],
        end_time=time[[12, 7, 15]],
    )

    np.testing.assert_allclose(means, [11.0, np.mean([2, 3, 4, 5, 7]), 15.0])
    no_bins = event_mean_ripple_power(frame, [1.0], [1.1])
    assert np.isnan(no_bins).all()
    with pytest.raises(ValueError, match="overlap"):
        event_mean_ripple_power(frame, time[[2, 4]], time[[5, 8]])
