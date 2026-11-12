"""The content contrast ladder's audited table: its specs, its concentration
family and the rungs it names as fit on a rescaled response."""

from __future__ import annotations

import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.content.session_fits import (
    fit_contrast_ladder,
)
from continuum_swr_replay.analysis.behavior.contrast_aggregation import (
    aggregate_contrast,
    audited_content_contrast,
    concentration_specs,
    renormalized_rungs,
    specs_from_ladder,
)
from tests.support._content_tables import dm_contrast_session, with_event_length


def test_concentration_specs_cover_every_rung_and_population():
    specs = concentration_specs({"C0_base": ([], "pooled"), "C1_split": ([], "split")})
    # Both concentration terms on every (rung, population): the length term is
    # only fitted under --concentration-length, and on a run without it the
    # coefficient column is absent so aggregate_contrast emits no row for it.
    assert specs == [
        ("C0_base", "all", "log_phi_is_swr"),
        ("C0_base", "all", "log_phi_log_n_nonlocal_bins_z"),
        ("C1_split", "all", "log_phi_is_swr"),
        ("C1_split", "all", "log_phi_log_n_nonlocal_bins_z"),
        ("C1_split", "SWR", "log_phi_is_swr"),
        ("C1_split", "SWR", "log_phi_log_n_nonlocal_bins_z"),
        ("C1_split", "nonSWR", "log_phi_is_swr"),
        ("C1_split", "nonSWR", "log_phi_log_n_nonlocal_bins_z"),
    ]


def test_only_the_rescaled_rungs_that_produced_rows_are_named():
    """A rung nothing was fitted for cannot be claimed as rescaled."""
    fitted = pd.DataFrame(
        {
            "model": ["C4_armvalue", "C4_armvalue", "C1_nine_arm"],
            "response_renormalized": [True, True, False],
        }
    )
    empty = pd.DataFrame(columns=["model", "response_renormalized"])

    assert renormalized_rungs(fitted, empty) == ["C4_armvalue"]
    assert renormalized_rungs(empty) == []
    # A count run rescales nothing, so nothing is named.
    assert renormalized_rungs(fitted.assign(response_renormalized=False)) == []
    # A frame without the column claims nothing either.
    assert renormalized_rungs(fitted.drop(columns="response_renormalized")) == []


def _fits(model: str, term: str, coefs: list[float]) -> pd.DataFrame:
    n = len(coefs)
    return pd.DataFrame(
        {
            "model": model,
            "population": "all",
            "animal": [f"a{i % 3}" for i in range(n)],
            "session_date": [f"d{i}" for i in range(n)],
            "epoch": "02_r1",
            f"coef_{term}": coefs,
            f"bse_{term}": 0.1,
            f"sd_{term}": 1.0,
            "coef_log_phi_is_swr": [-0.5 - 0.1 * i for i in range(n)],
            "bse_log_phi_is_swr": 0.2,
            "separated_terms": "",
            "converged": True,
            "separated": False,
        }
    )


def test_the_audited_table_is_the_key_terms_then_the_concentration_family():
    arm_ladder = {"C1": (["is_current_arm"], "pooled")}
    patch_ladder = {"CP1": (["is_current_patch"], "pooled")}
    arm_key = {"C1": "is_current_arm"}
    patch_key = {"CP1": "is_current_patch"}
    arm = _fits("C1", "is_current_arm", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    patch = _fits("CP1", "is_current_patch", [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6])
    scale = {"arm_cohort": {}, "patch_cohort": {}}

    audited = audited_content_contrast(
        arm,
        patch,
        arm_ladder=arm_ladder,
        arm_key_term=arm_key,
        patch_ladder=patch_ladder,
        patch_key_term=patch_key,
        per_sd_scale=scale,
    )

    expected = pd.concat(
        [
            aggregate_contrast(
                arm, specs_from_ladder(arm_ladder, arm_key), "candidate_arm", {}
            ),
            aggregate_contrast(
                patch, specs_from_ladder(patch_ladder, patch_key), "patch", {}
            ),
            aggregate_contrast(arm, concentration_specs(arm_ladder), "candidate_arm"),
            aggregate_contrast(patch, concentration_specs(patch_ladder), "patch"),
        ],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(audited, expected)
    assert list(zip(audited["level"], audited["term"], strict=True)) == [
        ("candidate_arm", "is_current_arm"),
        ("patch", "is_current_patch"),
        ("candidate_arm", "log_phi_is_swr"),
        ("patch", "log_phi_is_swr"),
    ]
    # The key terms' medians: 0.35 at the arm grain, -0.35 at the patch grain.
    assert audited.loc[0, "coef_session_median"] == pytest.approx(0.35)
    assert audited.loc[1, "coef_session_median"] == pytest.approx(-0.35)


class TestLengthAdjustedConcentration:
    """``--concentration-length`` fits the length-adjusted twin of log_phi_is_swr."""

    def test_the_aggregation_reports_the_length_term_next_to_the_contrast(self):
        cohort = pd.concat(
            [
                with_event_length(
                    dm_contrast_session(epoch=f"{index:02d}_r1", seed=index), seed=index
                )
                for index in range(6)
            ],
            ignore_index=True,
        )
        ladder = {"C0_base": (["is_current_arm"], "pooled")}
        res, _ = fit_contrast_ladder(
            cohort,
            ladder,
            response_col="arm_count_hard",
            response="hard_count",
            n_categories=9,
            estimator="dirichlet_multinomial",
            n_jobs=2,
            min_events_session_phi=40,
            concentration_length=True,
        )
        agg = aggregate_contrast(res, concentration_specs(ladder), "candidate_arm")

        assert set(agg["term"]) == {
            "log_phi_is_swr",
            "log_phi_log_n_nonlocal_bins_z",
        }
        # A z-scored column is reported per SD, not per category.
        length = agg[agg["term"] == "log_phi_log_n_nonlocal_bins_z"].iloc[0]
        assert length["std_unit"] == "per_SD"
