"""The mPFC cohort audit must read both sides off the artifacts, not restate them.

The report is the only place the bin-level and event-level restriction is
written down with numbers, so these tests pin the three things a reader relies
on, on a synthetic pair of bundle trees: the per-session before/after counts
come from the sidecar's own fields (and total correctly); a session present in
only one tree is called out rather than silently making the population totals
incomparable; and the firing-rate gate is counted on both the fit-bin rate the
new bundles screen and the all-immobile rate the old ones carried under that
name, so the gate's effect is separable from the cohort's.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.qa.compare_pfc_cohorts import (
    FDR_TESTS,
    MATCHED_CONTROL_TESTS,
    RERUN_PARAM_TOL,
    _md,
    _short_hash,
    alignment_prose,
    build_report,
    bundle_sessions,
    coverage_summary,
    event_count_agreement,
    fitter_table,
    gate_counts,
    matched_control_arrays,
    matched_control_significance,
    provenance_table,
    read_sidecars,
    same_events_precondition,
    session_cohort_table,
    shared_statistic_differences,
    significance_table,
    try_read_sidecars,
    worker_time_table,
)

N_LAG_TIMES = 201


def _write_bundle(
    data_dir,
    session: str,
    *,
    n_neurons: int,
    fit_rate: float,
    all_immobile_rate: float | None,
    pvals: float,
    sidecar: dict | None,
    lr: float = 3.0,
    modulation: float = 0.0,
    matched_control_pvals: dict[str, float] | None = None,
    matched_control_array: bool = False,
) -> None:
    """A minimal mPFC bundle: the stats pickle, the modulation arrays, sidecar.

    ``matched_control_pvals`` adds the two engagement-test p-value columns and
    ``matched_control_array`` the array beside them, so a bundle can be written
    with the control, without it, or with only half of it.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "nwb_file_name": [f"{session.split('_')[0]}_.nwb"] * n_neurons,
            "epoch_name": ["02_r1"] * n_neurons,
            "immobility_avg_firing_rate": [fit_rate] * n_neurons,
        }
    )
    if all_immobile_rate is not None:
        frame["immobility_avg_firing_rate_all_immobile"] = [
            all_immobile_rate
        ] * n_neurons
    for test in FDR_TESTS:
        frame[f"{test}: pval"] = pvals
        frame[f"{test}: lr"] = lr
    frame["min_aic_model"] = "SWR"
    for test, value in (matched_control_pvals or {}).items():
        frame[f"{test}: pval"] = value

    base = data_dir / f"{session}_mPFC_modulation"
    frame.to_pickle(base.with_suffix(".pkl"))
    names = [
        "SWR_modulation",
        "no_SWR_modulation",
        "non_local_SWR_modulation",
        "non_local_no_SWR_modulation",
    ]
    if matched_control_array:
        names.append("matched_control_modulation")
    for name in names:
        np.save(
            f"{base}.{name}.npy", np.full((N_LAG_TIMES, n_neurons), float(modulation))
        )
    if sidecar is not None:
        (data_dir / f"{session}_mPFC_modulation.provenance.json").write_text(
            json.dumps(sidecar)
        )


def _sidecar(
    session: str,
    *,
    bins_immobile: int,
    bins_fit: int,
    nonlocal_before: int,
    nonlocal_after: int,
    ripple_before: int,
    ripple_after: int,
    overlapping: int = 0,
    alignment: dict | None = None,
    git_hash: str = "abc1234def",
    written_at: str = "2026-09-14T00:00:00+00:00",
    stop_table_sha256: str = "0" * 64,
    extra: dict | None = None,
) -> dict:
    counts = {
        "stop_cohort": "at_well",
        "n_bins_total": bins_immobile * 2,
        "n_bins_immobile": bins_immobile,
        "n_bins_fit": bins_fit,
        "n_nonlocal_before": nonlocal_before,
        "n_nonlocal_after": nonlocal_after,
        "n_nonlocal_swr": nonlocal_after // 2,
        "n_nonlocal_no_swr": nonlocal_after - nonlocal_after // 2,
        "n_ripple_before": ripple_before,
        "n_ripple_after": ripple_after,
        "n_nonlocal_dropped_overlapping_fit": overlapping,
        "n_nonlocal_dropped_overlapping_fit_frac_of_kept": (
            overlapping / nonlocal_after if nonlocal_after else None
        ),
        "n_ripple_dropped_overlapping_fit": 0,
        "n_ripple_dropped_overlapping_fit_frac_of_kept": 0.0,
    }
    if alignment is not None:
        counts["alignment"] = alignment
    return {
        "artifact": f"{session}_mPFC_modulation.pkl",
        "brain_area": "mPFC",
        "nwb_file_name": f"{session.split('_')[0]}_.nwb",
        "epoch_name": "02_r1",
        "git_hash": git_hash,
        "written_at": written_at,
        "stop_cohort": "at_well",
        "stop_table_sha256": stop_table_sha256,
        "cohort_counts": counts,
        **(extra or {}),
    }


@pytest.fixture
def bundle_trees(tmp_path):
    """A before tree with three sessions and an after tree with two of them."""
    before = tmp_path / "before"
    after = tmp_path / "after"

    # Same two sessions on both sides; the third only exists before (its refit
    # has not landed), which is the situation the report must announce.
    _write_bundle(
        before,
        "chimi20200223_02_r1",
        n_neurons=3,
        fit_rate=5.0,
        all_immobile_rate=None,
        pvals=0.001,
        sidecar=None,
    )
    _write_bundle(
        before,
        "j1620210706_02_r1",
        n_neurons=2,
        fit_rate=5.0,
        all_immobile_rate=None,
        pvals=0.5,
        sidecar=None,
    )
    _write_bundle(
        before,
        "j1620210706_10_r5",
        n_neurons=4,
        fit_rate=5.0,
        all_immobile_rate=None,
        pvals=0.5,
        sidecar=None,
    )

    # After: the fit-bin rate falls below the gate for the chimi neurons while
    # the all-immobile rate stays above it, so the two gate columns disagree.
    _write_bundle(
        after,
        "chimi20200223_02_r1",
        n_neurons=3,
        fit_rate=0.5,
        all_immobile_rate=5.0,
        pvals=0.001,
        sidecar=_sidecar(
            "chimi20200223_02_r1",
            bins_immobile=1000,
            bins_fit=900,
            nonlocal_before=100,
            nonlocal_after=90,
            ripple_before=50,
            ripple_after=48,
            overlapping=2,
            alignment={
                "dropped_at": "tail",
                "n_results_bins_dropped": 8,
                "dropped_seconds_results": 0.016,
                "n_position_rows_dropped": 0,
                "dropped_seconds_position": 0.0,
            },
        ),
    )
    _write_bundle(
        after,
        "j1620210706_02_r1",
        n_neurons=2,
        fit_rate=5.0,
        all_immobile_rate=6.0,
        pvals=0.5,
        sidecar=_sidecar(
            "j1620210706_02_r1",
            bins_immobile=2000,
            bins_fit=1000,
            nonlocal_before=200,
            nonlocal_after=150,
            ripple_before=80,
            ripple_after=72,
            git_hash="def5678abc",
            written_at="2026-09-14T02:00:00+00:00",
        ),
    )
    return before, after


class TestMarkdown:
    def test_nullable_integer_counts_render_as_counts(self):
        """A neuron count must not render as ``10986.0`` because one row is NA."""
        frame = pd.DataFrame(
            [{"side": "before", "n": None}, {"side": "after", "n": 10986}]
        )
        frame["n"] = frame["n"].astype("Int64")
        rendered = _md(frame)
        assert "| after | 10986 |" in rendered
        assert "| before |  |" in rendered
        assert "10986.0" not in rendered


class TestProvenance:
    def test_a_dirty_hash_keeps_its_marker_when_abbreviated(self):
        """Truncating to seven characters must not launder a dirty tree."""
        assert _short_hash("1ba32acbec635f7dc36d01423d82e05789279536") == "1ba32ac"
        assert (
            _short_hash("1ba32acbec635f7dc36d01423d82e05789279536-dirty")
            == "1ba32ac-dirty"
        )

    def test_runner_code_comes_from_the_sidecar_not_the_recorded_hash(
        self, bundle_trees
    ):
        """The recorded HEAD drifts ahead of the imported code; two facts do not.

        A session with no alignment block predates that step; a session trimmed
        past the single budget could only have been accepted by the per-axis
        form. Both bundles here record the same hash family, and they must still
        land in different regimes.
        """
        _, after = bundle_trees
        _write_bundle(
            after,
            "senor20201101_02_r1",
            n_neurons=1,
            fit_rate=5.0,
            all_immobile_rate=5.0,
            pvals=0.5,
            sidecar=_sidecar(
                "senor20201101_02_r1",
                bins_immobile=1000,
                bins_fit=900,
                nonlocal_before=10,
                nonlocal_after=9,
                ripple_before=5,
                ripple_after=5,
                alignment={
                    "dropped_at": "tail",
                    "n_results_bins_dropped": 2232,
                    "dropped_seconds_results": 4.464,
                    "n_position_rows_dropped": 0,
                    "dropped_seconds_position": 0.0,
                },
            ),
        )
        table = provenance_table(read_sidecars(after))
        regimes = dict(zip(table.runner_code, table.n_sessions, strict=True))
        assert regimes["50bf4b7 (no alignment step)"] == 1
        assert regimes["7362347 (alignment, 0.1 s budget)"] == 1
        assert regimes["6c162ce (alignment, per-axis budgets)"] == 1
        # The over-budget session is the one that names the per-axis form.
        per_axis = table.loc[
            table.runner_code == "6c162ce (alignment, per-axis budgets)"
        ]
        assert per_axis.alignment_block.item() == "yes"


class TestSidecarTables:
    def test_session_table_carries_both_sides_and_totals(self, bundle_trees):
        _, after = bundle_trees
        table = session_cohort_table(read_sidecars(after))

        chimi = table.loc[table.session == "chimi20200223_02_r1"].iloc[0]
        assert chimi.bins_immobile_before == 1000
        assert chimi.bins_fit_after == 900
        assert chimi.bins_kept_pct == pytest.approx(90.0)
        assert chimi.nonlocal_before == 100
        assert chimi.nonlocal_after == 90
        assert chimi.ripple_before == 50
        assert chimi.ripple_after == 48
        assert chimi.align_results_bins_dropped == 8

        total = table.iloc[-1]
        assert total.session.startswith("ALL (2 sessions)")
        assert total.bins_immobile_before == 3000
        assert total.bins_fit_after == 1900
        # Weighted by bins, not a mean of the per-session percentages.
        assert total.bins_kept_pct == pytest.approx(100 * 1900 / 3000)
        assert total.nonlocal_before == 300
        assert total.nonlocal_after == 240

    def test_missing_alignment_block_is_blank_not_zero(self, bundle_trees):
        """A sidecar written before the alignment step must not read as "0 dropped"."""
        _, after = bundle_trees
        table = session_cohort_table(read_sidecars(after))
        j16 = table.loc[table.session == "j1620210706_02_r1"].iloc[0]
        assert pd.isna(j16.align_results_bins_dropped)
        assert pd.isna(j16.align_results_s_dropped)

    def test_coverage_summary_names_the_affected_sessions(self, bundle_trees):
        _, after = bundle_trees
        totals, affected = coverage_summary(read_sidecars(after))
        assert totals["n_nonlocal_dropped_overlapping_fit"] == 2
        assert totals["n_sessions_nonlocal"] == 1
        assert totals["n_ripple_dropped_overlapping_fit"] == 0
        assert list(affected.session) == ["chimi20200223_02_r1"]

    def test_alignment_prose_separates_no_record_from_no_trim(self, bundle_trees):
        """A sidecar with no alignment block is not a session that needed none."""
        _, after = bundle_trees
        prose = alignment_prose(read_sidecars(after))
        assert "1 of 2 sidecars record" in prose
        assert "8 decode bins" in prose
        assert "0 position rows" in prose
        assert "1 session(s) were fit before the step existed" in prose

    def test_alignment_prose_counts_a_recorded_no_op_as_a_no_op(self, tmp_path):
        """A recorded ``dropped_at: none`` must not be reported as a trim.

        The step records itself on every session once it exists, so "recorded"
        and "trimmed" come apart as soon as a session whose axes already agreed
        is fit by the current code.
        """
        tree = tmp_path / "after"
        _write_bundle(
            tree,
            "chimi20200223_02_r1",
            n_neurons=1,
            fit_rate=5.0,
            all_immobile_rate=5.0,
            pvals=0.5,
            sidecar=_sidecar(
                "chimi20200223_02_r1",
                bins_immobile=1000,
                bins_fit=900,
                nonlocal_before=10,
                nonlocal_after=9,
                ripple_before=5,
                ripple_after=5,
                alignment={
                    "dropped_at": "none",
                    "n_results_bins_dropped": 0,
                    "dropped_seconds_results": 0.0,
                    "n_position_rows_dropped": 0,
                    "dropped_seconds_position": 0.0,
                },
            ),
        )
        _write_bundle(
            tree,
            "chimi20200223_04_r2",
            n_neurons=1,
            fit_rate=5.0,
            all_immobile_rate=5.0,
            pvals=0.5,
            sidecar=_sidecar(
                "chimi20200223_04_r2",
                bins_immobile=1000,
                bins_fit=900,
                nonlocal_before=10,
                nonlocal_after=9,
                ripple_before=5,
                ripple_after=5,
                alignment={
                    "dropped_at": "tail",
                    "n_results_bins_dropped": 4,
                    "dropped_seconds_results": 0.008,
                    "n_position_rows_dropped": 0,
                    "dropped_seconds_position": 0.0,
                },
            ),
        )
        prose = alignment_prose(read_sidecars(tree))
        assert "2 of 2 sidecars record" in prose
        assert "1 of those needed a trim" in prose
        assert "1 record the step as a no-op" in prose
        assert "every one of those" not in prose
        assert "`chimi20200223_04_r2`" in prose

    def test_a_bundle_without_a_sidecar_is_refused(self, bundle_trees):
        """A bundle with no sidecar would be counted by the population only."""
        _, after = bundle_trees
        _write_bundle(
            after,
            "peanut20201128_06_r3",
            n_neurons=1,
            fit_rate=5.0,
            all_immobile_rate=5.0,
            pvals=0.5,
            sidecar=None,
        )
        with pytest.raises(ValueError, match="no provenance sidecar"):
            read_sidecars(after)

    def test_animal_is_not_split_on_a_digit(self, bundle_trees):
        """``j16`` sessions must not collapse to ``j1``."""
        _, after = bundle_trees
        assert set(read_sidecars(after).animal) == {"chimi", "j16"}


class TestPopulation:
    def test_gate_counts_separate_the_two_rate_columns(self, bundle_trees):
        before, after = bundle_trees

        before_gate = gate_counts(before, min_firing_rate=1.0)
        assert before_gate["n_neurons_with_fit"] == 9
        assert before_gate["n_passing_fit_bin_rate"] == 9
        # The pre-cohort bundles do not carry the unrestricted twin.
        assert before_gate["n_passing_all_immobile_rate"] is None

        after_gate = gate_counts(after, min_firing_rate=1.0)
        assert after_gate["n_neurons_with_fit"] == 5
        # The three chimi neurons fall below the gate on the fit-bin rate only.
        assert after_gate["n_passing_fit_bin_rate"] == 2
        assert after_gate["n_passing_all_immobile_rate"] == 5

    def test_significance_fraction_is_per_animal_and_per_test(self, bundle_trees):
        before, _ = bundle_trees
        from continuum_swr_replay.analysis.pfc.load_population import (
            load_population_data,
        )

        frame, _, _ = load_population_data(before, min_firing_rate=1.0)
        table = significance_table(frame, alpha=0.05)

        chimi = table.loc[
            (table.animal == "chimi") & (table.test == "SWR vs. constant")
        ].iloc[0]
        assert chimi.n_tested == 3
        assert chimi.n_sig == 3
        assert chimi.frac_sig == pytest.approx(1.0)

        j16 = table.loc[
            (table.animal == "j16") & (table.test == "SWR vs. constant")
        ].iloc[0]
        assert j16.n_tested == 6
        assert j16.n_sig == 0

        assert set(table.animal) == {"all", "chimi", "j16"}
        assert set(table.test) == set(FDR_TESTS)


class TestReport:
    def test_report_announces_a_session_only_one_tree_has(self, bundle_trees):
        before, after = bundle_trees
        assert bundle_sessions(before) - bundle_sessions(after) == {"j1620210706_10_r5"}

        from continuum_swr_replay.analysis.pfc.load_population import (
            load_population_data,
        )

        before_frame, _, _ = load_population_data(before, min_firing_rate=1.0)
        after_frame, _, _ = load_population_data(after, min_firing_rate=1.0)
        report = build_report(
            sidecars=read_sidecars(after),
            before=before_frame,
            after=after_frame,
            before_gate=gate_counts(before, 1.0),
            after_gate=gate_counts(after, 1.0),
            before_dir=before,
            after_dir=after,
            min_firing_rate=1.0,
            alpha=0.05,
            notes=["A note the caller supplied."],
            command="uv run python scripts/qa/compare_pfc_cohorts.py",
        )

        assert "do not hold the same sessions" in report
        assert "j1620210706_10_r5" in report
        assert "A note the caller supplied." in report
        assert "Command: `uv run python scripts/qa/compare_pfc_cohorts.py`" in report
        # The headline restriction, computed from the sidecars.
        assert "1,900 of 3,000" in report
        assert "240 of 300" in report


def test_provenance_marks_no_alignment_block_when_no_sidecar_has_one():
    """A corpus fit entirely before the alignment step has no alignment column."""
    sidecars = pd.DataFrame(
        {
            "session": ["a", "b"],
            "git_hash": ["50bf4b7", "50bf4b7"],
            "written_at": ["2026-09-13T18:00:00Z", "2026-09-13T19:00:00Z"],
        }
    )
    from scripts.qa.compare_pfc_cohorts import provenance_table

    table = provenance_table(sidecars)
    assert set(table["alignment_block"]) == {"no"}


# ---------------------------------------------------------------------------
# The rerun audit: same events, new estimator, plus the matched control
# ---------------------------------------------------------------------------


def _rerun_sidecar(session: str, **kwargs) -> dict:
    """A sidecar with the counts the rerun tables read."""
    defaults = {
        "bins_immobile": 1000,
        "bins_fit": 900,
        "nonlocal_before": 100,
        "nonlocal_after": 90,
        "ripple_before": 50,
        "ripple_after": 48,
    }
    defaults.update(kwargs)
    return _sidecar(session, **defaults)


def _after_extra(**overrides) -> dict:
    """The top-level keys a canonical-events, batched-fitter sidecar carries."""
    extra = {
        "fitter": "jax",
        "fitter_tol": 1e-8,
        "jax_backend": "gpu",
        "jax_version": "0.8.0",
        "gpu_index": 3,
        "gpu_name": "NVIDIA A100-SXM4-80GB",
        "gpu_uuid": "GPU-abc",
        "n_neurons_reference_fit": 1,
        "n_model_fits": 27,
        "n_nonconverged_model_fits": 2,
        "n_neurons_with_nonconverged_model": 1,
        "canonical_event_table_sha256": "c" * 64,
        "data_over_time_sha256": "d" * 64,
        "is_swr_definition": "canonical_event_table",
        "n_matched_control": 88,
        "n_matched_control_unplaced": 2,
        "n_matched_control_onset_outside_fitted_bout": 1,
        "n_matched_control_bout_saturated": 1,
    }
    extra.update(overrides)
    return extra


@pytest.fixture
def rerun_trees(tmp_path):
    """Two trees holding the same two sessions, the after one with the control.

    Before: sidecars with no ``canonical_event_table_sha256`` and no fitter.
    After: the same events and stop table, fit by the batched estimator, with
    the matched-control columns and array.
    """
    before, after = tmp_path / "before", tmp_path / "after"
    for session, n_neurons in (("chimi20200223_02_r1", 3), ("j1620210706_02_r1", 2)):
        _write_bundle(
            before,
            session,
            n_neurons=n_neurons,
            fit_rate=5.0,
            all_immobile_rate=5.0,
            pvals=0.001,
            lr=3.0,
            modulation=0.25,
            sidecar=_rerun_sidecar(session),
        )
        _write_bundle(
            after,
            session,
            n_neurons=n_neurons,
            fit_rate=5.0,
            all_immobile_rate=5.0,
            pvals=0.001,
            lr=3.0,
            modulation=0.25,
            sidecar=_rerun_sidecar(session, extra=_after_extra()),
            matched_control_pvals=dict.fromkeys(MATCHED_CONTROL_TESTS, 0.001),
            matched_control_array=True,
        )
    return before, after


class TestSameEventsPrecondition:
    def test_the_two_trees_are_accepted_when_only_the_estimator_changed(
        self, rerun_trees
    ):
        before, after = rerun_trees
        ok, why = same_events_precondition(read_sidecars(before), read_sidecars(after))
        assert ok, why
        assert "2 shared session(s)" in why

    def test_a_tree_with_no_sidecars_is_refused_rather_than_assumed(self, tmp_path):
        # The pre-cohort tree has no sidecars at all. Nothing can be said about
        # what it was fit on, so no coefficient comparison may be claimed.
        after = tmp_path / "after"
        _write_bundle(
            after,
            "chimi20200223_02_r1",
            n_neurons=1,
            fit_rate=5.0,
            all_immobile_rate=5.0,
            pvals=0.001,
            sidecar=_rerun_sidecar("chimi20200223_02_r1", extra=_after_extra()),
        )
        assert try_read_sidecars(tmp_path / "missing") is None
        ok, why = same_events_precondition(None, read_sidecars(after))
        assert not ok
        assert "no provenance sidecars" in why

    def test_a_different_stop_table_is_refused(self, rerun_trees):
        before, after = rerun_trees
        # Repoint one before-side session at another stop table: the cohort is
        # then not the same, so nothing downstream is a check of the estimator.
        _write_bundle(
            before,
            "chimi20200223_02_r1",
            n_neurons=3,
            fit_rate=5.0,
            all_immobile_rate=5.0,
            pvals=0.001,
            sidecar=_rerun_sidecar("chimi20200223_02_r1", stop_table_sha256="9" * 64),
        )
        ok, why = same_events_precondition(read_sidecars(before), read_sidecars(after))
        assert not ok
        assert "different stop table" in why

    def test_a_before_tree_that_already_read_the_canonical_events_is_refused(
        self, rerun_trees
    ):
        before, after = rerun_trees
        _write_bundle(
            before,
            "chimi20200223_02_r1",
            n_neurons=3,
            fit_rate=5.0,
            all_immobile_rate=5.0,
            pvals=0.001,
            sidecar=_rerun_sidecar(
                "chimi20200223_02_r1",
                extra={"canonical_event_table_sha256": "c" * 64},
            ),
        )
        ok, why = same_events_precondition(read_sidecars(before), read_sidecars(after))
        assert not ok
        assert "already records" in why

    def test_event_counts_are_compared_beside_the_marker(self, rerun_trees):
        # The marker says only that the before tree recorded no event hash. The
        # counts are what decide whether the two sides really saw the same
        # regressors, so a session whose counts moved must be named.
        before, after = rerun_trees
        _write_bundle(
            before,
            "j1620210706_02_r1",
            n_neurons=2,
            fit_rate=5.0,
            all_immobile_rate=5.0,
            pvals=0.001,
            sidecar=_rerun_sidecar("j1620210706_02_r1", nonlocal_after=80),
        )
        table, n_disagree = event_count_agreement(
            read_sidecars(before), read_sidecars(after)
        )
        assert n_disagree == 1
        disagreeing = table.loc[~table["same_events"], "session"].tolist()
        assert disagreeing == ["j1620210706_02_r1"]


class TestSharedStatisticDifferences:
    def test_identical_trees_report_zero_outside_tolerance(self, rerun_trees):
        before, after = rerun_trees
        differences, per_session, unpairable = shared_statistic_differences(
            before,
            after,
            ["chimi20200223_02_r1", "j1620210706_02_r1"],
        )
        assert unpairable.empty
        assert int(differences["n_outside_tolerance"].sum()) == 0
        assert differences.attrs["n_neurons_paired"] == 5
        # "Expected zero" is a per-session claim, so it is made per session.
        assert list(per_session["session"]) == [
            "chimi20200223_02_r1",
            "j1620210706_02_r1",
        ]
        assert int(per_session["n_cells_outside_tolerance"].sum()) == 0
        assert int(per_session["n_best_aic_labels_on_one_side_only"].sum()) == 0
        compared = differences.loc[differences["n_cells"] > 0, "max_abs_diff"]
        assert (compared.fillna(0) == 0).all()

    def test_a_perturbed_unit_is_counted_and_its_size_reported(self, rerun_trees):
        before, after = rerun_trees
        # One neuron, one p-value, moved by 1e-3: far outside the 1e-6 gate.
        path = after / "chimi20200223_02_r1_mPFC_modulation.pkl"
        frame = pd.read_pickle(path)
        column = f"{FDR_TESTS[0]}: pval"
        frame.loc[0, column] = frame.loc[0, column] + 1e-3
        frame.to_pickle(path)

        differences, per_session, _ = shared_statistic_differences(
            before,
            after,
            ["chimi20200223_02_r1", "j1620210706_02_r1"],
        )
        row = differences.loc[differences["quantity"] == "LR p-value (`: pval`)"].iloc[
            0
        ]
        assert row["n_outside_tolerance"] == 1
        assert row["max_abs_diff"] == pytest.approx(1e-3)
        # ...and the session it is in is named, not just the corpus total.
        moved = per_session.set_index("session")["n_cells_outside_tolerance"]
        assert moved["chimi20200223_02_r1"] == 1
        assert moved["j1620210706_02_r1"] == 0
        # Nothing else moved.
        others = differences.loc[
            differences["quantity"] != "LR p-value (`: pval`)", "n_outside_tolerance"
        ]
        assert int(others.sum()) == 0

    def test_a_moved_modulation_trace_is_caught(self, rerun_trees):
        before, after = rerun_trees
        path = after / "chimi20200223_02_r1_mPFC_modulation.SWR_modulation.npy"
        array = np.load(path)
        array[0, 0] += 10 * RERUN_PARAM_TOL
        np.save(path, array)

        differences, per_session, _ = shared_statistic_differences(
            before, after, ["chimi20200223_02_r1"]
        )
        row = differences.loc[
            differences["quantity"] == "modulation `SWR_modulation`"
        ].iloc[0]
        assert row["n_outside_tolerance"] == 1
        assert row["max_abs_diff"] == pytest.approx(10 * RERUN_PARAM_TOL)

    def test_a_cell_finite_on_one_side_only_is_counted_separately(self, rerun_trees):
        # A model that became estimable (or stopped being) is a change no
        # maximum over finite differences can see.
        before, after = rerun_trees
        path = after / "chimi20200223_02_r1_mPFC_modulation.pkl"
        frame = pd.read_pickle(path)
        frame.loc[0, f"{FDR_TESTS[0]}: pval"] = np.nan
        frame.to_pickle(path)

        differences, per_session, _ = shared_statistic_differences(
            before, after, ["chimi20200223_02_r1"]
        )
        row = differences.loc[differences["quantity"] == "LR p-value (`: pval`)"].iloc[
            0
        ]
        assert row["n_finite_on_one_side_only"] == 1

    def test_a_session_with_different_neuron_counts_is_not_aligned(self, rerun_trees):
        before, after = rerun_trees
        _write_bundle(
            after,
            "chimi20200223_02_r1",
            n_neurons=2,
            fit_rate=5.0,
            all_immobile_rate=5.0,
            pvals=0.001,
            modulation=0.25,
            sidecar=_rerun_sidecar("chimi20200223_02_r1", extra=_after_extra()),
            matched_control_array=True,
        )
        differences, per_session, unpairable = shared_statistic_differences(
            before, after, ["chimi20200223_02_r1", "j1620210706_02_r1"]
        )
        assert list(unpairable.session) == ["chimi20200223_02_r1"]
        assert unpairable.iloc[0].n_neurons_before == 3
        assert unpairable.iloc[0].n_neurons_after == 2
        # The other session is still compared.
        assert differences.attrs["n_sessions_paired"] == 1


class TestBestAICComparison:
    def test_a_neuron_with_no_fit_on_either_side_is_not_a_disagreement(
        self, rerun_trees
    ):
        # A neuron statsmodels refused carries no best-AIC label in either
        # tree. Comparing the two labels with != would call NaN != NaN a
        # disagreement and report the corpus's unfittable neurons as an
        # estimator difference.
        before, after = rerun_trees
        for tree in (before, after):
            path = tree / "chimi20200223_02_r1_mPFC_modulation.pkl"
            frame = pd.read_pickle(path)
            frame.loc[0, "min_aic_model"] = np.nan
            frame.to_pickle(path)

        differences, per_session, _ = shared_statistic_differences(
            before, after, ["chimi20200223_02_r1"]
        )

        row = differences.loc[
            differences["quantity"] == "best-AIC model (identity)"
        ].iloc[0]
        assert row["n_outside_tolerance"] == 0
        assert row["n_finite_on_one_side_only"] == 0
        assert int(per_session["n_best_aic_labels_on_one_side_only"].sum()) == 0

    def test_a_neuron_that_lost_its_fit_is_counted_as_one_sided(self, rerun_trees):
        before, after = rerun_trees
        path = after / "chimi20200223_02_r1_mPFC_modulation.pkl"
        frame = pd.read_pickle(path)
        frame.loc[0, "min_aic_model"] = np.nan
        frame.to_pickle(path)

        differences, per_session, _ = shared_statistic_differences(
            before, after, ["chimi20200223_02_r1"]
        )

        row = differences.loc[
            differences["quantity"] == "best-AIC model (identity)"
        ].iloc[0]
        assert row["n_finite_on_one_side_only"] == 1
        assert row["n_outside_tolerance"] == 0
        assert (
            per_session.set_index("session").loc[
                "chimi20200223_02_r1", "n_best_aic_labels_on_one_side_only"
            ]
            == 1
        )


class TestWorkerTimes:
    def test_a_workers_wall_clock_and_session_count_are_read(self, tmp_path):
        (tmp_path / "worker0.time").write_text(
            '\tCommand being timed: "uv run python x.py"\n'
            "\tPercent of CPU this job got: 338%\n"
            "\tElapsed (wall clock) time (h:mm:ss or m:ss): 12:19:27\n"
            "\tMaximum resident set size (kbytes): 225591152\n"
            "\tExit status: 0\n"
        )
        (tmp_path / "worker0.sessions").write_text(
            "# a comment\nchimi20200223_.nwb 02_r1\n\nj1620210706_.nwb 02_r1\n"
        )

        table = worker_time_table([tmp_path / "worker0.time"])

        assert len(table) == 1
        row = table.iloc[0]
        assert row["worker"] == "worker0"
        assert row["wall_clock"] == "12:19:27"
        assert row["cpu_pct"] == "338%"
        assert row["peak_rss_gb"] == pytest.approx(225.6)
        assert row["exit_status"] == "0"
        assert row["n_sessions"] == 2

    def test_a_worker_without_a_partition_file_still_reports(self, tmp_path):
        (tmp_path / "worker1.time").write_text(
            "\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:00\n\tExit status: 2\n"
        )

        row = worker_time_table([tmp_path / "worker1.time"]).iloc[0]

        assert row["n_sessions"] is None
        assert row["exit_status"] == "2"


class TestMatchedControl:
    def test_every_bundle_is_checked_for_the_control_array(self, rerun_trees):
        _, after = rerun_trees
        table = matched_control_arrays(after)
        assert table["matched_control_array"].all()
        assert table["matches_non_local_no_SWR_shape"].all()

    def test_a_bundle_without_the_control_array_is_named(self, rerun_trees):
        _, after = rerun_trees
        (
            after / "j1620210706_02_r1_mPFC_modulation.matched_control_modulation.npy"
        ).unlink()
        table = matched_control_arrays(after)
        missing = table.loc[~table["matched_control_array"], "session"].tolist()
        assert missing == ["j1620210706_02_r1"]

    def test_the_two_engagement_tests_get_fdr_fractions_pooled_and_per_animal(
        self, rerun_trees
    ):
        _, after = rerun_trees
        frame = pd.DataFrame(
            {
                "animal": ["chimi"] * 3 + ["j16"] * 2,
                f"{MATCHED_CONTROL_TESTS[0]}: pval": [1e-6, 1e-6, 0.9, 1e-6, 0.9],
                f"{MATCHED_CONTROL_TESTS[1]}: pval": [0.9, 0.9, 0.9, 0.9, 0.9],
            }
        )
        table = matched_control_significance(frame, alpha=0.05)
        assert set(table["animal"]) == {"all", "chimi", "j16"}
        assert set(table["test"]) == set(MATCHED_CONTROL_TESTS)

        pooled = table.loc[
            (table.animal == "all") & (table.test == MATCHED_CONTROL_TESTS[0])
        ].iloc[0]
        assert pooled.n_tested == 5
        assert pooled.n_sig == 3
        assert pooled.frac_sig == pytest.approx(3 / 5)
        # The baseline test is significant nowhere.
        baseline = table.loc[
            (table.animal == "all") & (table.test == MATCHED_CONTROL_TESTS[1])
        ].iloc[0]
        assert baseline.n_sig == 0
        chimi = table.loc[
            (table.animal == "chimi") & (table.test == MATCHED_CONTROL_TESTS[0])
        ].iloc[0]
        assert (chimi.n_tested, chimi.n_sig) == (3, 2)

    def test_a_test_absent_from_the_frame_reports_nothing_tested(self):
        # A population fit before the control existed has neither column; the
        # fraction must be an empty denominator, not a crash or a zero.
        frame = pd.DataFrame({"animal": ["chimi", "j16"]})
        table = matched_control_significance(frame, alpha=0.05)
        assert (table["n_tested"] == 0).all()
        assert table["frac_sig"].isna().all()


class TestFitterTable:
    def test_one_batch_collapses_to_one_row_with_its_hand_back_total(self, rerun_trees):
        _, after = rerun_trees
        table = fitter_table(read_sidecars(after))
        assert len(table) == 1
        row = table.iloc[0]
        assert row.fitter == "jax"
        assert row.jax_backend == "gpu"
        assert row.gpu_index == 3
        assert row.n_sessions == 2
        assert row.n_neurons_reference_fit == 2
        # The count of fits that ran to maxiter is the only trace those fits
        # leave: their LR tests and AIC are reported like any other.
        assert row.n_model_fits == 54
        assert row.n_nonconverged_model_fits == 4
        assert row.n_neurons_with_nonconverged_model == 2
        # Hashes are abbreviated, not dropped.
        assert row.canonical_event_table_sha256 == "c" * 12

    def test_a_session_fit_on_another_card_gets_its_own_row(self, rerun_trees):
        _, after = rerun_trees
        _write_bundle(
            after,
            "j1620210706_02_r1",
            n_neurons=2,
            fit_rate=5.0,
            all_immobile_rate=5.0,
            pvals=0.001,
            modulation=0.25,
            sidecar=_rerun_sidecar(
                "j1620210706_02_r1", extra=_after_extra(gpu_index=4)
            ),
            matched_control_array=True,
        )
        table = fitter_table(read_sidecars(after))
        assert len(table) == 2
        assert sorted(table.gpu_index) == [3, 4]


class TestRerunReport:
    def test_the_report_states_the_tolerance_result_and_the_control(self, rerun_trees):
        before, after = rerun_trees
        before_sidecars = read_sidecars(before)
        after_sidecars = read_sidecars(after)
        population_before, _, _ = __import__(
            "continuum_swr_replay.analysis.pfc.load_population",
            fromlist=["load_population_data"],
        ).load_population_data(before, min_firing_rate=1.0)
        population_after, _, _ = __import__(
            "continuum_swr_replay.analysis.pfc.load_population",
            fromlist=["load_population_data"],
        ).load_population_data(after, min_firing_rate=1.0)

        report = build_report(
            sidecars=after_sidecars,
            before_sidecars=before_sidecars,
            before=population_before,
            after=population_after,
            before_gate=gate_counts(before, 1.0),
            after_gate=gate_counts(after, 1.0),
            before_dir=before,
            after_dir=after,
            min_firing_rate=1.0,
            alpha=0.05,
            notes=[],
            command="",
        )
        assert "## Same events, new estimator" in report
        assert "No cell falls outside tolerance." in report
        assert "## The matched non-event control" in report
        assert "2 of 2 bundle(s) carry a `matched_control_modulation` array" in report
        assert "## Estimator, device and inputs" in report
        assert "4 of 54 shipped model fits did not converge" in report

    def test_a_tree_without_the_estimator_fields_gets_no_rerun_sections(
        self, bundle_trees
    ):
        # The cohort audit's own trees: no fitter, no event hash. The rerun
        # sections must not appear at all rather than appear empty.
        before, after = bundle_trees
        population_before, _, _ = __import__(
            "continuum_swr_replay.analysis.pfc.load_population",
            fromlist=["load_population_data"],
        ).load_population_data(before, min_firing_rate=1.0)
        population_after, _, _ = __import__(
            "continuum_swr_replay.analysis.pfc.load_population",
            fromlist=["load_population_data"],
        ).load_population_data(after, min_firing_rate=0.0)
        report = build_report(
            sidecars=read_sidecars(after),
            before_sidecars=try_read_sidecars(before),
            before=population_before,
            after=population_after,
            before_gate=gate_counts(before, 1.0),
            after_gate=gate_counts(after, 1.0),
            before_dir=before,
            after_dir=after,
            min_firing_rate=1.0,
            alpha=0.05,
            notes=[],
            command="",
        )
        assert "## Same events, new estimator" not in report
        assert "## Estimator, device and inputs" not in report
