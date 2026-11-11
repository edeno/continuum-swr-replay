"""Synthetic probes for the accompanying analysis-validity review.

Run from the repository root with the project environment. These are diagnostic
counterexamples, not recording-data results and not changes to the analyses.
Each probe prints the current behavior beside an independently specified check.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.coefficient_statistics import (
    aic_difference_statistics,
    coefficient_statistics,
)
from continuum_swr_replay.analysis.replay.fitting import fit_conditional_poisson
from continuum_swr_replay.analysis.replay.learning import fit_is_stay_pooled_gee
from continuum_swr_replay.analysis.replay.model_specs import (
    MODEL_PER_BIN_RATE,
    REWARD_HISTORY_COLUMNS,
)
from continuum_swr_replay.analysis.replay.per_bin_design import (
    reward_kernel_rate_ratios,
)
from continuum_swr_replay.analysis.replay.rate_fit_statistics import (
    reward_kernel_statistics,
)

ROOT = Path(__file__).resolve().parents[3]


def allocation_frame(probability: float, start: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": np.repeat(np.arange(start, start + 20), 3),
            "x": np.tile([0.0, 0.0, 1.0], 20),
            "patch_fraction": np.tile(
                [(1 - probability) / 2, (1 - probability) / 2, probability], 20
            ),
        }
    )


def fit(frame: pd.DataFrame, predictors: list[str]) -> dict:
    result = fit_conditional_poisson(
        frame, predictors, response_col="patch_fraction", n_categories=3
    )
    if result is None:
        raise RuntimeError("The synthetic fit was refused")
    return result


def uniform_allocations() -> dict:
    base = allocation_frame(0.8)
    uniform = allocation_frame(1 / 3, 20)
    # Exactly equal fractions, without roundoff from (1 - 1/3) / 2.
    uniform["patch_fraction"] = 1 / 3
    combined = pd.concat([base, uniform], ignore_index=True)
    original, added = fit(base, ["x"]), fit(combined, ["x"])
    beta = added["coef_x"]
    return {
        "fitted_probability_with_uniform_events": float(
            np.exp(beta) / (2 + np.exp(beta))
        ),
        "analytic_optimum_probability": float(
            combined.loc[combined.x == 1, "patch_fraction"].mean()
        ),
        "original_llf": original["llf"],
        "llf_with_uniform_events": added["llf"],
        "original_n_events": original["n_events"],
        "n_events_with_uniform_events": added["n_events"],
        "original_null_llf": original["llf_null"],
        "null_llf_with_uniform_events": added["llf_null"],
    }


def canceling_predictor() -> dict:
    frame = allocation_frame(0.8)
    frame["z"] = np.repeat(np.arange(20), 3)
    original, added = fit(frame, ["x"]), fit(frame, ["x", "z"])
    return {
        "llf_change": added["llf"] - original["llf"],
        "aic_change": added["aic"] - original["aic"],
        "expected_aic_change_for_identical_model": 0.0,
        "reported_n_params": added["n_params"],
        "unidentified_terms": added["unidentified_terms"],
    }


class RecordingAxes:
    """Capture the actual plot function's bars without rendering a figure."""

    def bar(self, x, height, **kwargs):
        self.heights = list(height)

    def twinx(self):
        return self

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def paired_aic_plot() -> dict:
    path = ROOT / "scripts/rate/plot_per_bin_rate_summary.py"
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_panel_ladder"
    )
    ladder = [
        name for name, spec in MODEL_PER_BIN_RATE.items() if spec.time_clock == "stop"
    ]
    namespace = {"np": np, "LADDER": ladder}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    null = pd.DataFrame(
        {
            "animal": ["a"] * 3,
            "session_date": [1, 2, 3],
            "epoch": ["02_r1"] * 3,
            "aic": [0.0, 100.0, 101.0],
        }
    )
    child = null.assign(aic=[90.0, 91.0, 201.0])
    frames = [
        (child if name == ladder[1] else null).assign(
            model=name, outcome="non_local", cv_bits_per_event=0.0
        )
        for name in ladder
    ]
    axes = RecordingAxes()
    namespace["_panel_ladder"](axes, pd.concat(frames, ignore_index=True))
    return {
        "child_model": ladder[1],
        "plotted_delta": axes.heights[1],
        "paired_median_delta": aic_difference_statistics(child, null)["median"],
    }


def kernel_cohort() -> dict:
    frame = pd.DataFrame(
        {
            "animal": ["a", "a", "b", "b", "c", "c"],
            "session_date": np.arange(6),
            "epoch": ["02_r1"] * 6,
            "converged": True,
            "separated_terms": [""] * 5 + [REWARD_HISTORY_COLUMNS[0]],
        }
    )
    for index, term in enumerate(REWARD_HISTORY_COLUMNS):
        frame[f"coef_{term}"] = [0.1, 0.2, 0.3, 0.4, 0.5, 2.0] if index == 0 else 0.0
    summary = reward_kernel_statistics(frame)
    return {
        "plot_lag_one": float(reward_kernel_rate_ratios(frame)[0]),
        "summary_lag_one": summary["rate_ratio_lag_1"],
        "summary_n_sessions": summary["n_sessions"],
        "summary_n_dropped": summary["n_dropped"],
    }


def gee_convergence() -> dict:
    rng = np.random.default_rng(8)
    frame = pd.DataFrame(
        {
            "animal": np.repeat(["a", "b", "c", "d", "e"], 20),
            "session_date": np.repeat(np.arange(5), 20),
            "epoch": "02_r1",
            "is_stay": np.tile([0, 0, 1, 1], 25),
            "is_omitted_outcome": np.tile([0, 1], 50),
            "content": rng.uniform(size=100),
        }
    )
    # Exercise how the real wrapper handles a backend result that explicitly
    # reports failure to converge. No claim about the lab fits is made here.
    failed_result = SimpleNamespace(
        converged=False, params=np.zeros(4), bse=np.ones(4), pvalues=np.full(4, 0.01)
    )
    with patch(
        "statsmodels.genmod.generalized_estimating_equations.GEE.fit",
        return_value=failed_result,
    ):
        result = fit_is_stay_pooled_gee(frame, content_col="content")
    return {
        "backend_converged": False,
        "reported_status": result["status"],
        "reported_interaction_p": result["p_inter"],
    }


def repeated_sessions() -> dict:
    frame = pd.DataFrame(
        {
            "animal": ["a", "b", "c", "d", "e"],
            "session_date": 1,
            "epoch": "02_r1",
            "coef_x": [0.1, 0.2, 0.4, 0.6, 0.8],
        }
    )
    repeated = pd.concat(
        [frame.assign(epoch=f"{i + 2:02d}_r1") for i in range(10)], ignore_index=True
    )
    original = coefficient_statistics(frame, "coef_x")
    copied = coefficient_statistics(repeated, "coef_x")
    return {
        "original_session_p": original["signed_rank_p"],
        "p_after_copying_each_animals_value_into_ten_epochs": copied["signed_rank_p"],
        "original_animals": original["n_animals"],
        "animals_after_copying": copied["n_animals"],
    }


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "uniform_allocations": uniform_allocations(),
                "canceling_predictor": canceling_predictor(),
                "paired_aic_plot": paired_aic_plot(),
                "kernel_cohort": kernel_cohort(),
                "gee_convergence": gee_convergence(),
                "repeated_sessions": repeated_sessions(),
            },
            indent=2,
        )
    )
