#!/usr/bin/env python
"""Generate all figures for the replay content analysis.

Loads pre-computed results from fit_replay_content_models.py and
run_behavior_prediction.py, generates publication-quality figures.

Usage:
    uv run python scripts/generate_replay_content_figures.py
"""

import logging
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.replay.model_specs import (
    MODEL_HIERARCHY,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.visualization.replay_content_plots import (
    plot_behavior_prediction_results,
    plot_coefficient_forest,
    plot_model_hierarchy_comparison,
    plot_pseudo_r2_progression,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)

figure_utilities.set_figure_defaults(context="paper")

MODEL_DIR = PROCESSED_DATA_DIR / "replay_content" / "models"
OUTPUT_DIR = Path("paper/figures/replay_content")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load results ──
    all_models_path = MODEL_DIR / "all_models.pkl"
    comparison_path = MODEL_DIR / "model_comparison.pkl"
    behavior_path = PROCESSED_DATA_DIR / "replay_content" / "behavior_prediction.pkl"

    if not all_models_path.exists():
        logging.error(
            "Model results not found. Run fit_replay_content_models.py first."
        )
        return

    all_results = pd.read_pickle(all_models_path)
    logging.info("Loaded %d model fits", len(all_results))

    # ── Figure 1: Model hierarchy comparison ──
    if comparison_path.exists():
        comparison = pd.read_pickle(comparison_path)
        fig = plot_model_hierarchy_comparison(comparison)
        if fig is not None:
            path = OUTPUT_DIR / "model_hierarchy_comparison.png"
            fig.savefig(path, dpi=300)
            plt.close(fig)
            logging.info("Saved %s", path)

    # ── Figure 2: Pseudo-R² progression ──
    fig = plot_pseudo_r2_progression(all_results)
    if fig is not None:
        path = OUTPUT_DIR / "pseudo_r2_progression.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        logging.info("Saved %s", path)

    # ── Figure 3: Coefficient forest plots for key models ──
    key_models = ["M2", "M4", "M6", "M7"]
    for model_name in key_models:
        if model_name not in MODEL_HIERARCHY:
            continue
        mdf = all_results[all_results["model"] == model_name]
        if len(mdf) == 0:
            continue
        spec = MODEL_HIERARCHY[model_name]
        fig = plot_coefficient_forest(mdf, spec.predictors, model_name)
        if fig is not None:
            path = OUTPUT_DIR / f"coefficients_{model_name}.png"
            fig.savefig(path, dpi=300)
            plt.close(fig)
            logging.info("Saved %s", path)

    # ── Figure 4: Behavior prediction ──
    if behavior_path.exists():
        cv_results = pd.read_pickle(behavior_path)
        # Determine chance level from data
        bout_path = PROCESSED_DATA_DIR / "replay_content" / "bout_data.pkl"
        chance = 1 / 6  # default
        if bout_path.exists():
            bout_data = pd.read_pickle(bout_path)
            n_wells = bout_data["next_well"].nunique()
            chance = 1 / n_wells

        fig = plot_behavior_prediction_results(cv_results, chance_level=chance)
        if fig is not None:
            path = OUTPUT_DIR / "behavior_prediction.png"
            fig.savefig(path, dpi=300)
            plt.close(fig)
            logging.info("Saved %s", path)

    # ── Figure 5: Mixed-effects results (if available) ──
    mixed_dir = MODEL_DIR / "mixed_effects"
    if mixed_dir.exists():
        for pkl_file in sorted(mixed_dir.glob("*_mixed.pkl")):
            model_name = pkl_file.stem.replace("_mixed", "")
            with open(pkl_file, "rb") as f:
                me = pickle.load(f)

            fig, ax = plt.subplots(
                figsize=(5, 0.4 * len(me["fe_names"]) + 1), constrained_layout=True
            )
            y_pos = range(len(me["fe_names"]))
            means = me["fe_mean"]
            sds = me["fe_sd"]
            labels = [n.replace("is_", "").replace("_x_", "×") for n in me["fe_names"]]

            colors = [
                "#1b9e77" if abs(m / s) > 1.96 else "#cccccc"
                for m, s in zip(means, sds, strict=True)
            ]

            ax.barh(
                y_pos,
                means,
                color=colors,
                alpha=0.7,
                edgecolor="none",
                height=0.6,
            )
            ax.errorbar(
                means,
                y_pos,
                xerr=[1.96 * s for s in sds],
                fmt="none",
                color="black",
                capsize=2,
                linewidth=0.5,
            )
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Fixed Effect (mean ± 95% CI)")
            ax.set_title(f"Mixed-Effects {model_name}")
            ax.invert_yaxis()
            sns.despine(ax=ax)

            path = OUTPUT_DIR / f"mixed_effects_{model_name}.png"
            fig.savefig(path, dpi=300)
            plt.close(fig)
            logging.info("Saved %s", path)

    logging.info("Done! All figures saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
