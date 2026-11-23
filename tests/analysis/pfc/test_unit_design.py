"""The shared design of a session's unit models, in the convention's terms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.design import (
    _make_design_matrices,
    symmetric_lags,
)
from continuum_swr_replay.analysis.pfc.marked_glm import (
    EventKernelBlock,
    build_kernel_design,
)
from continuum_swr_replay.analysis.pfc.unit_design import (
    Kernel,
    KernelPenalty,
    build_unit_design,
    unit_term_support,
    without_terms,
)
from continuum_swr_replay.analysis.regression.pspline import difference_penalty
from continuum_swr_replay.analysis.regression.terms import (
    Category,
    Slope,
    build_design,
    log_penalty_from_prior_sd,
)

FS = 500.0
LAGS = symmetric_lags(0.25, FS)
DECLARED = KernelPenalty("declared", log_penalty=2.0)


@pytest.fixture
def session():
    """60 s at 500 Hz; ten visits of 4 s, one every 6 s; SWRs in eight visits,
    replay without SWR in all ten, a mark on the replay events."""
    rng = np.random.default_rng(0)
    time = np.arange(30_000) / FS
    visit = np.full(time.size, -1)
    for v in range(10):
        visit[(time >= 6 * v + 1) & (time < 6 * v + 5)] = v
    fit_mask = pd.Series(visit >= 0, index=pd.Index(time, name="time"))
    swr = 6 * np.arange(8) + 1.5 + rng.uniform(0, 3, 8)
    replay = 6 * np.arange(10) + 1.5 + rng.uniform(0, 3, 10)
    control = 6 * np.arange(10) + 1.2 + rng.uniform(0, 3, 10)
    events = {
        "SWR": pd.DataFrame({"start_time": swr}),
        "non_local_no_SWR": pd.DataFrame(
            {"start_time": replay, "mark_z": np.linspace(-1.5, 1.5, 10)}
        ),
        "non_local_SWR": pd.DataFrame({"start_time": swr + 0.01}),
        "non_local": pd.DataFrame({"start_time": np.sort(np.r_[swr + 0.01, replay])}),
        "matched_control": pd.DataFrame({"start_time": control}),
    }
    return fit_mask, events, visit[visit >= 0]


def test_kernel_columns_equal_the_shipped_design(session):
    fit_mask, events, _ = session
    shipped, _, _ = _make_design_matrices(
        events, fit_mask.to_frame(), sampling_frequency=FS, lags=LAGS
    )
    design = build_unit_design(
        events,
        fit_mask,
        kernels=[Kernel("SWR"), Kernel("non_local_no_SWR")],
        lags=LAGS,
        sampling_frequency=FS,
        penalty=DECLARED,
    )
    np.testing.assert_array_equal(
        design.X, shipped["SWR + non_local_no_SWR"][fit_mask.to_numpy()]
    )
    assert design.coef_names[0] == "(Intercept)"
    assert design.coef_names[1] == "k(SWR).0"


def test_marked_kernel_columns_equal_the_marked_design(session):
    fit_mask, events, _ = session
    marked = build_kernel_design(
        events,
        fit_mask,
        build_unit_design(
            events,
            fit_mask,
            kernels=[Kernel("SWR")],
            lags=LAGS,
            sampling_frequency=FS,
            penalty=DECLARED,
        ).spline_basis,
        LAGS,
        (
            EventKernelBlock("a", "non_local_no_SWR"),
            EventKernelBlock("b", "non_local_no_SWR", "mark_z"),
        ),
    )
    design = build_unit_design(
        events,
        fit_mask,
        kernels=[Kernel("non_local_no_SWR"), Kernel("non_local_no_SWR", "mark_z")],
        lags=LAGS,
        sampling_frequency=FS,
        penalty=DECLARED,
    )
    np.testing.assert_array_equal(design.X, marked.matrix)


def test_each_kernel_carries_the_difference_penalty_and_the_intercept_none(session):
    fit_mask, events, _ = session
    design = build_unit_design(
        events,
        fit_mask,
        kernels=[Kernel("SWR"), Kernel("non_local_no_SWR")],
        lags=LAGS,
        sampling_frequency=FS,
        penalty=DECLARED,
    )
    n_basis = design.spline_basis.shape[1]
    assert len(design.blocks) == 2
    for block, name in zip(
        design.blocks, ("k(SWR)", "k(non_local_no_SWR)"), strict=True
    ):
        np.testing.assert_array_equal(
            block.columns, np.arange(design.X.shape[1])[design.kernel_columns[name]]
        )
        np.testing.assert_array_equal(block.S, difference_penalty(n_basis, 2))
        assert block.rank == n_basis - 2
    S = design.penalty_matrix()
    assert not S[0].any() and not S[:, 0].any()
    np.testing.assert_allclose(
        S[1 : 1 + n_basis, 1 : 1 + n_basis] / np.exp(2.0),
        difference_penalty(n_basis, 2),
    )
    # The penalty leaves a kernel's plateau (equal coefficients) free; the
    # shipped ridge shrank it.
    plateau = np.zeros(design.X.shape[1])
    plateau[design.kernel_columns["k(SWR)"]] = 1.0
    np.testing.assert_allclose(S @ plateau, 0.0, atol=1e-12)


def test_the_two_settings_of_the_kernel_penalty(session):
    fit_mask, events, _ = session
    kwargs = {
        "kernels": [Kernel("SWR"), Kernel("non_local_no_SWR")],
        "lags": LAGS,
        "sampling_frequency": FS,
    }
    declared = build_unit_design(events, fit_mask, penalty=DECLARED, **kwargs)
    reml = build_unit_design(
        events, fit_mask, penalty=KernelPenalty("reml_over_units"), **kwargs
    )
    np.testing.assert_array_equal(declared.X, reml.X)
    np.testing.assert_array_equal(declared.log_penalties, [2.0, 2.0])
    assert np.isnan(reml.log_penalties).all()
    with pytest.raises(ValueError, match="REML"):
        reml.penalty_matrix()
    np.testing.assert_array_equal(
        reml.penalty_matrix(reml.with_kernel_log_penalties([2.0, 2.0])),
        declared.penalty_matrix(),
    )
    assert reml.penalty_record()["k(SWR)"] == {
        "set": "reml_over_units",
        "log_penalty": None,
    }
    assert declared.penalty_record()["k(SWR)"] == {"set": "fixed", "log_penalty": 2.0}
    with pytest.raises(ValueError):
        KernelPenalty("declared")
    with pytest.raises(ValueError):
        KernelPenalty("reml_over_units", log_penalty=1.0)


@pytest.fixture
def baseline_rows(session):
    fit_mask, _, visit = session
    return pd.DataFrame(
        {"well": visit % 3, "rewarded_outcome": (visit % 2).astype(float)}
    )


def test_a_tested_covariate_carries_the_weak_penalty(session, baseline_rows):
    fit_mask, events, _ = session
    weak = log_penalty_from_prior_sd(2.0)
    baseline = build_design(
        [
            Category("well", (0, 1, 2), 0, "none"),
            Slope("rewarded_outcome", weak),
        ],
        baseline_rows,
        intercept=True,
    )
    design = build_unit_design(
        events,
        fit_mask,
        kernels=[Kernel("SWR")],
        lags=LAGS,
        sampling_frequency=FS,
        penalty=DECLARED,
        baseline=baseline,
        baseline_rows=baseline_rows,
        tested=["rewarded_outcome"],
    )
    assert design.coef_names[:4] == (
        "(Intercept)",
        "well[1]",
        "well[2]",
        "rewarded_outcome",
    )
    np.testing.assert_array_equal(design.log_penalties, [weak, 2.0])
    record = design.penalty_record()
    assert record["well"] == {"set": "none", "log_penalty": None}
    assert record["rewarded_outcome"] == {"set": "fixed", "log_penalty": weak}

    unpenalized = build_design(
        [Slope("rewarded_outcome", "none")], baseline_rows, intercept=True
    )
    with pytest.raises(ValueError, match="without a fixed penalty"):
        build_unit_design(
            events,
            fit_mask,
            kernels=[Kernel("SWR")],
            lags=LAGS,
            sampling_frequency=FS,
            penalty=DECLARED,
            baseline=unpenalized,
            baseline_rows=baseline_rows,
            tested=["rewarded_outcome"],
        )


def test_support_counts_the_visits_on_each_side(session, baseline_rows):
    fit_mask, events, visit = session
    few = {**events, "SWR": events["SWR"].iloc[:4]}  # SWRs in four visits only
    baseline = build_design(
        [Category("well", (0, 1, 2), 0, "none")], baseline_rows, intercept=True
    )
    design = build_unit_design(
        few,
        fit_mask,
        kernels=[
            Kernel("SWR"),
            Kernel("non_local_no_SWR"),
            Kernel("non_local_no_SWR", "mark_z"),
        ],
        lags=LAGS,
        sampling_frequency=FS,
        penalty=DECLARED,
        baseline=baseline,
        baseline_rows=baseline_rows,
    )
    support = unit_term_support(design, visit)
    assert (support["k(SWR)"].n_high, support["k(SWR)"].passes) == (4, False)
    assert support["k(SWR)"].n_low == 10
    assert support["k(non_local_no_SWR)"].n_high == 10
    assert support["k(non_local_no_SWR)"].passes
    # five marks below zero and five above
    marked = support["k(non_local_no_SWR:mark_z)"]
    assert (marked.n_low, marked.n_high, marked.passes) == (5, 5, True)
    assert support["well[1]"].n_high == 3 and not support["well[1]"].passes

    reduced = without_terms(design, ["k(SWR)"])
    assert [k.name for k in reduced.kernels] == [
        "k(non_local_no_SWR)",
        "k(non_local_no_SWR:mark_z)",
    ]
    n_basis = design.spline_basis.shape[1]
    assert reduced.X.shape[1] == design.X.shape[1] - n_basis
    np.testing.assert_array_equal(
        reduced.X[:, reduced.kernel_columns["k(non_local_no_SWR)"]],
        design.X[:, design.kernel_columns["k(non_local_no_SWR)"]],
    )
    assert len(reduced.blocks) == 2
    assert all("k(SWR)" not in name for name in reduced.coef_names)
