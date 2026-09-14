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
    _md,
    alignment_prose,
    build_report,
    bundle_sessions,
    coverage_summary,
    gate_counts,
    read_sidecars,
    session_cohort_table,
    significance_table,
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
) -> None:
    """A minimal mPFC bundle: the stats pickle, four modulation arrays, sidecar."""
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

    base = data_dir / f"{session}_mPFC_modulation"
    frame.to_pickle(base.with_suffix(".pkl"))
    for name in (
        "SWR_modulation",
        "no_SWR_modulation",
        "non_local_SWR_modulation",
        "non_local_no_SWR_modulation",
    ):
        np.save(f"{base}.{name}.npy", np.zeros((N_LAG_TIMES, n_neurons)))
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
        "stop_table_sha256": "0" * 64,
        "cohort_counts": counts,
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
