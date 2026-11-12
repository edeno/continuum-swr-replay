#!/usr/bin/env python
"""Generate all figures for the replay content analysis.

Loads pre-computed results from fit_replay_content_models.py and generates
publication-quality figures.

Figures 4 and 5 read inputs that nothing in this repository produces any more:
``behavior_prediction.pkl`` and ``bout_data.pkl`` under the replay-content root
(written by a ``run_behavior_prediction.py`` that no longer exists) and the
``mixed_effects/`` directory under the arm bundle. Each is drawn only when its
input is already on disk, so on a current checkout those figures are skipped.

Usage:
    uv run python scripts/behavior/content/generate_replay_content_figures.py
    uv run python scripts/behavior/content/generate_replay_content_figures.py --allow-stale
"""

import argparse
import logging
import pickle
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import continuum_swr_replay.visualization.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.behavior.content.fit_outputs import (
    frame_estimator,
    scale_label,
)
from continuum_swr_replay.analysis.behavior.content.model_comparison import (
    run_full_hierarchy_comparison,
)
from continuum_swr_replay.analysis.behavior.fit_provenance import (
    require_fresh_inputs,
)
from continuum_swr_replay.analysis.behavior.model_specs import (
    MODEL_HIERARCHY,
)
from continuum_swr_replay.paths import (
    figure_dir,
    model_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.cli import fit_figure_parser
from continuum_swr_replay.visualization.palettes import COLORS
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


def bundle_dir() -> Path:
    """The canonical arm bundle the figures are drawn from."""
    return model_dir("content_arm")


def all_models_input_paths() -> dict[str, Path]:
    """The table fit_replay_content_models.py fit the canonical arm bundle on.

    Checked against each artifact's provenance sidecar before plotting, so a
    figure cannot be rendered from coefficients that predate the table on disk.
    """
    return {"event_arm_table": table_path("event_arm_table")}


def output_dir() -> Path:
    """Where the figures go: beside the data root (``paths.figure_dir``)."""
    return figure_dir("replay_content")


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser (also the provenance gate's opt-out)."""
    return fit_figure_parser(__doc__)


def main(argv: Sequence[str] | None = None):
    args = build_parser().parse_args(argv)
    rc, models_dir, out_dir = replay_content_dir(), bundle_dir(), output_dir()
    all_models_inputs = all_models_input_paths()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load results ──
    all_models_path = models_dir / "all_models.pkl"
    behavior_path = rc / "behavior_prediction.pkl"

    if not all_models_path.exists():
        logging.error(
            "Model results not found. Run fit_replay_content_models.py first."
        )
        return

    require_fresh_inputs(
        all_models_path, all_models_inputs, allow_stale=args.allow_stale
    )
    all_results = pd.read_pickle(all_models_path)
    logging.info("Loaded %d model fits", len(all_results))

    # ── Figure 1: Model hierarchy comparison ──
    # Taken from the fits rather than from the bundle's model_comparison.pkl,
    # so the figure has the comparison's current sign whatever code wrote it.
    comparison = run_full_hierarchy_comparison(all_results, hierarchy=MODEL_HIERARCHY)
    if not comparison.empty:
        fig = plot_model_hierarchy_comparison(
            comparison, scale_label=scale_label(frame_estimator(comparison))
        )
        if fig is not None:
            path = out_dir / "model_hierarchy_comparison.png"
            figure_utilities.save_figure(
                fig,
                path.stem,
                base_dir=path.parent,
                dpi=300,
                formats=["png"],
                transparent=False,
            )
            plt.close(fig)
            logging.info("Saved %s", path)

    # ── Figure 2: Pseudo-R² progression ──
    fig = plot_pseudo_r2_progression(all_results)
    if fig is not None:
        path = out_dir / "pseudo_r2_progression.png"
        figure_utilities.save_figure(
            fig,
            path.stem,
            base_dir=path.parent,
            dpi=300,
            formats=["png"],
            transparent=False,
        )
        plt.close(fig)
        logging.info("Saved %s", path)

    # ── Figure 3: Coefficient forest plots for key models ──
    key_models = ["M2", "M4", "M6"]
    for model_name in key_models:
        mdf = all_results[all_results["model"] == model_name]
        if len(mdf) == 0:
            continue
        spec = MODEL_HIERARCHY[model_name]
        fig = plot_coefficient_forest(mdf, spec.predictors, model_name)
        if fig is not None:
            path = out_dir / f"coefficients_{model_name}.png"
            figure_utilities.save_figure(
                fig,
                path.stem,
                base_dir=path.parent,
                dpi=300,
                formats=["png"],
                transparent=False,
            )
            plt.close(fig)
            logging.info("Saved %s", path)

    # ── Figure 4: Behavior prediction ──
    if behavior_path.exists():
        cv_results = pd.read_pickle(behavior_path)
        # Determine chance level from data
        bout_path = rc / "bout_data.pkl"
        chance = 1 / 6  # default
        if bout_path.exists():
            bout_data = pd.read_pickle(bout_path)
            n_wells = bout_data["next_well"].nunique()
            chance = 1 / n_wells

        fig = plot_behavior_prediction_results(cv_results, chance_level=chance)
        if fig is not None:
            path = out_dir / "behavior_prediction.png"
            figure_utilities.save_figure(
                fig,
                path.stem,
                base_dir=path.parent,
                dpi=300,
                formats=["png"],
                transparent=False,
            )
            plt.close(fig)
            logging.info("Saved %s", path)

    # ── Figure 5: Mixed-effects results (if available) ──
    mixed_dir = models_dir / "mixed_effects"
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
                COLORS["Non-Local"] if abs(m / s) > 1.96 else "#cccccc"
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

            path = out_dir / f"mixed_effects_{model_name}.png"
            figure_utilities.save_figure(
                fig,
                path.stem,
                base_dir=path.parent,
                dpi=300,
                formats=["png"],
                transparent=False,
            )
            plt.close(fig)
            logging.info("Saved %s", path)

    logging.info("Done! All figures saved to %s", out_dir)


if __name__ == "__main__":
    main()
