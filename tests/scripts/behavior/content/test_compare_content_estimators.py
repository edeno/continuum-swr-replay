"""The three-estimator comparison's runner: its figures, caption and partial runs.

The comparison itself (``estimator_comparison.run_comparison``) is tested in
``tests/analysis/behavior/content/test_run_comparison.py``; this file checks
what the runner draws and writes from its result.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.content import estimator_comparison as ec
from continuum_swr_replay.analysis.behavior.model_specs import (
    ComparisonRung,
    resolve_comparison_rungs,
)
from scripts.behavior.content import compare_content_estimators as mod
from tests.support._content_tables import estimator_comparison_cohort

BASE_RUNG = ComparisonRung(
    name="B_spatial",
    predictors=("is_current_arm", "is_prev_arm"),
    parent=None,
    leaf_only=False,
    key_term="is_current_arm",
)
REWARD_RUNG = ComparisonRung(
    name="B_reward",
    predictors=(
        "is_current_arm",
        "is_prev_arm",
        "reward_fraction_candidate_arm_last10_visits",
    ),
    parent="B_spatial",
    leaf_only=False,
    key_term="reward_fraction_candidate_arm_last10_visits",
)


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    return estimator_comparison_cohort()


@pytest.fixture(scope="module")
def one_rung(cohort) -> ec.ComparisonResult:
    return ec.run_comparison(
        cohort, (BASE_RUNG,), n_jobs=2, n_bootstrap=200, seed=ec.BOOTSTRAP_SEED
    )


@pytest.fixture(scope="module")
def two_rungs(cohort) -> ec.ComparisonResult:
    return ec.run_comparison(
        cohort,
        (BASE_RUNG, REWARD_RUNG),
        n_jobs=2,
        n_bootstrap=200,
        seed=ec.BOOTSTRAP_SEED,
    )


@pytest.mark.slow
def test_figures_and_caption_are_written(two_rungs, tmp_path):
    agreement = mod.plot_coefficient_agreement(
        two_rungs.coefficients,
        (BASE_RUNG, REWARD_RUNG),
        tmp_path / "coef_agreement.png",
    )
    log_phi = mod.plot_log_phi_by_session(
        two_rungs.coefficients,
        two_rungs.bootstrap,
        tmp_path / "log_phi_by_session.png",
        rung=BASE_RUNG.name,
    )
    caption = mod.write_caption(tmp_path / "content_estimators_caption.md")
    assert agreement.stat().st_size > 0
    assert log_phi.stat().st_size > 0
    assert "seed 0" in caption.read_text()


def test_the_concentration_figure_refuses_a_rung_it_has_no_fits_for(one_rung, tmp_path):
    with pytest.raises(ValueError, match="no identified Dirichlet-multinomial fits"):
        mod.plot_log_phi_by_session(
            one_rung.coefficients,
            one_rung.bootstrap,
            tmp_path / "log_phi_by_session.png",
            rung="not_a_fitted_rung",
        )


def test_the_concentration_figure_is_optional_but_the_provenance_is_not(
    one_rung, cohort, tmp_path, monkeypatch
):
    """A rung selection without the log-phi rung must still finish the run.

    The figure is pinned to one rung; raising there aborted AFTER the tables
    were written and BEFORE the provenance sidecar, leaving artifacts no reader
    could date or trace back to their inputs.
    """
    monkeypatch.setattr(mod, "assemble_arm", lambda *args: (cohort.copy(), {}))
    monkeypatch.setattr(
        mod, "select_outcome_stratum", lambda ev, stratum: (ev, {"stratum": "all"})
    )
    monkeypatch.setattr(mod, "record_inputs", lambda inputs: {})
    monkeypatch.setattr(mod, "run_comparison", lambda *args, **kwargs: one_rung)
    figures_dir = tmp_path / "figures"

    # M3 pulls in its parent M2; neither is the concentration figure's rung.
    mod.main(
        [
            "--rungs",
            "M3",
            "--output-dir",
            str(tmp_path / "out"),
            "--figures-dir",
            str(figures_dir),
            "--n-bootstrap",
            "10",
        ]
    )

    assert not (figures_dir / "log_phi_by_session.png").exists()
    provenance = tmp_path / "out" / f"{mod.COEFFICIENTS_BASENAME}.provenance.json"
    assert provenance.exists()
    assert json.loads(provenance.read_text())["rungs"] == ["M2", "M3"]


@pytest.mark.parametrize(
    ("names", "expected"),
    [(["C2_base"], mod.DEFAULT_LOG_PHI_RUNG), (["M3"], None)],
)
def test_the_figure_rung_is_reported_only_when_it_was_selected(names, expected):
    assert mod.log_phi_figure_rung(resolve_comparison_rungs(names)) == expected
