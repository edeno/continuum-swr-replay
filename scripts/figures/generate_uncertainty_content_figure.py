#!/usr/bin/env python
"""Render the ground-up reward-belief uncertainty/replay-content figures.

Requires the derived tables from::

    uv run python scripts/behavior/content/analyze_uncertainty_content.py

Usage
-----
uv run python scripts/figures/generate_uncertainty_content_figure.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from continuum_swr_replay.data_loaders.epoch_register import analysed_animals
from continuum_swr_replay.paths import (
    UNCERTAINTY_CONTENT_DIR,
    figure_dir,
    replay_content_dir,
)
from continuum_swr_replay.visualization.figure_utilities import (
    WONG,
    label_panels,
    save_figure,
    set_figure_defaults,
)

ANIMAL_ORDER = list(analysed_animals())
ANIMAL_COLORS = dict(zip(ANIMAL_ORDER, [WONG[i] for i in [1, 2, 3, 5, 7]], strict=True))
ANIMAL_MARKERS = dict(zip(ANIMAL_ORDER, ["o", "s", "^", "D", "P"], strict=True))
SCOPE_COLORS = {"all": "#666666", "unchosen": WONG[5]}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=replay_content_dir() / UNCERTAINTY_CONTENT_DIR,
        help="Directory written by analyze_uncertainty_content.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=figure_dir("uncertainty_content"),
        help="Figure output directory.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Keep figures open after saving.",
    )
    return parser.parse_args(argv)


def _load_results(results_dir: Path) -> dict[str, pd.DataFrame | dict]:
    paths = {
        "trial": results_dir / "trial_leaf_table.pkl",
        "profiles": results_dir / "quartile_profiles.csv",
        "contrasts": results_dir / "quartile_contrasts.csv",
        "mu_sessions": results_dir / "mu_session_coefficients.csv",
        "mu_animals": results_dir / "mu_animal_coefficients.csv",
        "mu_overall": results_dir / "mu_overall_coefficients.csv",
        "horserace_summary": results_dir / "horserace_summary.csv",
        "audit": results_dir / "cohort_audit.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing uncertainty-content result artifacts. Run the analysis first: "
            + ", ".join(missing)
        )
    audit = json.loads(paths["audit"].read_text())
    if audit.get("arm_standardization_support") != "leaf_arms":
        raise ValueError(
            "Uncertainty-content results predate leaf-support standardization. "
            "Regenerate the canonical event-arm table, Mu fit, and analysis bundle."
        )
    contrasts = pd.read_csv(paths["contrasts"])
    if "n_animals_positive" not in contrasts.columns:
        raise ValueError(
            f"{paths['contrasts']} predates the animals-positive count the figure "
            "annotates; rerun scripts/behavior/content/analyze_uncertainty_content.py."
        )
    return {
        "trial": pd.read_pickle(paths["trial"]),
        "profiles": pd.read_csv(paths["profiles"]),
        "contrasts": contrasts,
        "mu_sessions": pd.read_csv(paths["mu_sessions"]),
        "mu_animals": pd.read_csv(paths["mu_animals"]),
        "mu_overall": pd.read_csv(paths["mu_overall"]),
        "horserace_summary": pd.read_csv(paths["horserace_summary"]),
        "audit": audit,
    }


def _profile(
    profiles: pd.DataFrame,
    *,
    metric: str,
    response: str,
    scope: str,
    level: str,
) -> pd.DataFrame:
    return profiles[
        profiles["metric"].eq(metric)
        & profiles["response"].eq(response)
        & profiles["scope"].eq(scope)
        & profiles["level"].eq(level)
    ].sort_values(["animal", "uncertainty_quartile"], na_position="last")


def _example_matrices(
    trial: pd.DataFrame, audit: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = trial[
        trial["animal"].astype(str).eq(str(audit["example_animal"]))
        & trial["session_date"].astype(str).eq(str(audit["example_session_date"]))
        & trial["epoch"].astype(str).eq(str(audit["example_epoch"]))
    ].copy()
    if selected.empty:
        raise ValueError("representative session is absent from trial_leaf_table")
    uncertainty = selected.pivot(
        index="candidate_arm",
        columns="trial_number_by_epoch",
        values="reward_belief_variance_candidate_arm_preoutcome",
    ).sort_index()
    content = selected.pivot(
        index="candidate_arm",
        columns="trial_number_by_epoch",
        values="posterior_event_share",
    ).reindex(index=uncertainty.index, columns=uncertainty.columns)
    context = selected.drop_duplicates("trial_number_by_epoch").set_index(
        "trial_number_by_epoch"
    )
    context = context.reindex(uncertainty.columns)
    return uncertainty, content, context


def _plot_timeline(
    ax_uncertainty: plt.Axes,
    ax_content: plt.Axes,
    trial: pd.DataFrame,
    audit: dict,
) -> tuple[matplotlib.image.AxesImage, matplotlib.image.AxesImage]:
    uncertainty, content, context = _example_matrices(trial, audit)
    trial_numbers = uncertainty.columns.to_numpy()
    x = np.arange(len(trial_numbers))

    uncertainty_image = ax_uncertainty.imshow(
        uncertainty.to_numpy(),
        aspect="auto",
        interpolation="none",
        cmap="viridis",
        vmin=0,
        vmax=1 / 12,
        rasterized=True,
    )
    chosen = context["chosen_arm"].to_numpy(dtype=int)
    rewarded = context["trial_reward"].astype(bool).to_numpy()
    ax_uncertainty.scatter(
        x,
        chosen,
        s=7,
        marker="s",
        facecolors="none",
        edgecolors="white",
        linewidths=0.4,
        rasterized=True,
    )
    ax_uncertainty.scatter(
        x[rewarded],
        chosen[rewarded],
        s=7,
        marker="o",
        facecolors=WONG[1],
        edgecolors="black",
        linewidths=0.25,
        rasterized=True,
    )

    content_cmap = plt.colormaps["magma"].copy()
    content_cmap.set_bad("#eeeeee")
    content_image = ax_content.imshow(
        np.ma.masked_invalid(content.to_numpy()),
        aspect="auto",
        interpolation="none",
        cmap=content_cmap,
        vmin=0,
        vmax=1,
        rasterized=True,
    )

    n_ticks = min(7, len(trial_numbers))
    tick_positions = np.unique(
        np.linspace(0, len(trial_numbers) - 1, n_ticks).astype(int)
    )
    for ax in [ax_uncertainty, ax_content]:
        ax.set_yticks(range(6), [f"well {arm + 1}" for arm in range(6)])
        ax.set_xticks(tick_positions, trial_numbers[tick_positions])
        ax.tick_params(length=2, pad=1)
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax_uncertainty.tick_params(labelbottom=False)
    ax_content.set_xlabel("behavioral trial", labelpad=6)
    animal = str(audit["example_animal"])
    session_date = str(audit["example_session_date"])
    epoch = str(audit["example_epoch"])
    ax_uncertainty.set_title(
        f"Pre-outcome reward-belief uncertainty — {animal}, {session_date}, {epoch}",
        loc="left",
        pad=4,
    )
    ax_content.set_title(
        "Replay allocation among reward wells (gray = no replay)",
        loc="left",
        pad=4,
    )
    return uncertainty_image, content_image


def _plot_animal_profile(
    ax: plt.Axes,
    profiles: pd.DataFrame,
    *,
    metric: str,
    scale: float,
    ylabel: str,
    title: str,
    annotation: str,
) -> None:
    animals = _profile(
        profiles,
        metric=metric,
        response="posterior",
        scope="unchosen",
        level="animal",
    )
    overall = _profile(
        profiles,
        metric=metric,
        response="posterior",
        scope="unchosen",
        level="overall",
    )
    for animal in ANIMAL_ORDER:
        subset = animals[animals["animal"].eq(animal)]
        ax.plot(
            subset["uncertainty_quartile"],
            scale * subset["value"],
            color=ANIMAL_COLORS[animal],
            marker=ANIMAL_MARKERS[animal],
            markerfacecolor="white",
            markeredgewidth=0.8,
            linewidth=0.9,
            zorder=2,
        )
    ax.plot(
        overall["uncertainty_quartile"],
        scale * overall["value"],
        color="black",
        marker="o",
        linewidth=1.8,
        markersize=3.5,
        zorder=3,
    )
    ax.set_xticks([1, 2, 3, 4], ["Q1", "Q2", "Q3", "Q4"])
    ax.set_xlabel("same-well uncertainty quartile", labelpad=7)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_title(title, loc="left", pad=4)
    ax.text(
        0.03,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    ax.spines[["top", "right"]].set_visible(False)


def _contrast(
    contrasts: pd.DataFrame, *, metric: str, scope: str, level: str
) -> pd.DataFrame:
    return contrasts[
        contrasts["metric"].eq(metric)
        & contrasts["scope"].eq(scope)
        & contrasts["level"].eq(level)
    ]


def _positive_animals_note(contrasts: pd.DataFrame, metric: str) -> str:
    """The animals whose Q4 - Q1 contrast is above zero, as the analysis counts
    them (``paired_quartile_contrasts``)."""
    overall = _contrast(contrasts, metric=metric, scope="unchosen", level="overall")
    rows = overall[overall["response"].eq("posterior")]
    if rows.empty:
        raise ValueError(
            f"quartile_contrasts.csv has no overall posterior row for {metric} at "
            "unchosen wells; rerun scripts/behavior/content/analyze_uncertainty_content.py."
        )
    row = rows.iloc[0]
    return (
        f"Q4 − Q1 > 0 in {int(row['n_animals_positive'])}/{int(row['n_animals'])} "
        "animals"
    )


def _plot_response_robustness(ax: plt.Axes, contrasts: pd.DataFrame) -> None:
    data = _contrast(
        contrasts, metric="conditional_event_share", scope="unchosen", level="animal"
    )
    responses = ["hard", "soft", "posterior"]
    x = np.arange(len(responses))
    for animal in ANIMAL_ORDER:
        subset = (
            data[data["animal"].eq(animal)].set_index("response").reindex(responses)
        )
        ax.plot(
            x,
            100 * subset["contrast"],
            color=ANIMAL_COLORS[animal],
            marker=ANIMAL_MARKERS[animal],
            linewidth=0.9,
            markerfacecolor="white",
            markeredgewidth=0.8,
        )
    overall = (
        _contrast(
            contrasts,
            metric="conditional_event_share",
            scope="unchosen",
            level="overall",
        )
        .set_index("response")["contrast"]
        .reindex(responses)
    )
    ax.plot(x, 100 * overall, color="black", marker="o", linewidth=1.8, markersize=3.5)
    ax.axhline(0, color="#888888", linewidth=0.7, linestyle="--")
    ax.set_xticks(x, ["hard MAP", "soft MAP", "posterior"])
    ax.set_ylabel("Q4 − Q1 allocation\n(percentage points)", labelpad=8)
    ax.set_title("Unchosen-well response robustness", loc="left", pad=4)
    ax.spines[["top", "right"]].set_visible(False)


def _plot_mu_coefficients(
    ax: plt.Axes,
    mu_sessions: pd.DataFrame,
    mu_animals: pd.DataFrame,
) -> None:
    y_positions = dict(zip(ANIMAL_ORDER, range(len(ANIMAL_ORDER)), strict=True))
    for animal in ANIMAL_ORDER:
        subset = mu_sessions[mu_sessions["animal"].eq(animal)].sort_values("session_id")
        base = y_positions[animal]
        jitter = np.linspace(-0.18, 0.18, len(subset))
        clean = subset["fit_quality"].eq("clean").to_numpy()
        ax.scatter(
            subset.loc[
                clean,
                "coef_reward_belief_variance_candidate_arm_preoutcome_z",
            ],
            base + jitter[clean],
            s=5,
            color="#a9a9a9",
            alpha=0.65,
            edgecolors="none",
            rasterized=True,
        )
        ax.scatter(
            subset.loc[
                ~clean,
                "coef_reward_belief_variance_candidate_arm_preoutcome_z",
            ],
            base + jitter[~clean],
            s=7,
            color="#cfcfcf",
            marker="x",
            linewidths=0.45,
            rasterized=True,
        )
        clean_median = mu_animals[
            mu_animals["animal"].eq(animal) & mu_animals["cohort"].eq("clean")
        ]["coefficient"].item()
        ax.scatter(
            clean_median,
            base,
            s=24,
            color=ANIMAL_COLORS[animal],
            marker=ANIMAL_MARKERS[animal],
            edgecolors="black",
            linewidths=0.45,
            zorder=4,
        )
    ax.axvline(0, color="#666666", linewidth=0.7, linestyle="--")
    ax.set_yticks(range(len(ANIMAL_ORDER)), ANIMAL_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("uncertainty coefficient (log relative weight)", labelpad=7)
    ax.set_title("Adjusted event-conditioned model", loc="left", pad=4)
    ax.spines[["top", "right"]].set_visible(False)


def _animal_curve_handles(*, include_mean: bool = True) -> list[Line2D]:
    """Return the shared animal/mean curve key used across profile panels."""
    handles = [
        Line2D(
            [0],
            [0],
            color=ANIMAL_COLORS[animal],
            marker=ANIMAL_MARKERS[animal],
            markerfacecolor="white",
            markeredgewidth=0.8,
            linewidth=0.9,
            label=animal,
        )
        for animal in ANIMAL_ORDER
    ]
    if include_mean:
        handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                label="equal-animal mean",
            )
        )
    return handles


def _add_behavior_legend(ax: plt.Axes) -> None:
    """Define the behavioral markers used in the example uncertainty panel."""
    ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markeredgecolor="#333333",
            markerfacecolor="none",
            markersize=5,
            label="chosen well",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markeredgecolor="black",
            markerfacecolor=WONG[1],
            markersize=5,
            label="rewarded choice",
        ),
    ]
    ax.legend(
        handles=handles,
        title="Panel a: behavior",
        loc="upper left",
        borderaxespad=0,
        handlelength=1.4,
        labelspacing=0.65,
        fontsize=7,
        title_fontsize=7,
        frameon=False,
    )


def _add_replay_availability_legend(ax: plt.Axes) -> None:
    """Define the explicit no-replay encoding in the example content panel."""
    ax.axis("off")
    ax.legend(
        handles=[
            Patch(
                facecolor="#eeeeee",
                edgecolor="#777777",
                linewidth=0.5,
                label="no replay in trial",
            )
        ],
        title="Panel b: availability",
        loc="upper left",
        borderaxespad=0,
        handlelength=1.4,
        labelspacing=0.65,
        fontsize=7,
        title_fontsize=7,
        frameon=False,
    )
    ax.text(
        0,
        0.52,
        "Colorbar: fraction of all\nreplay content assigned\nto each reward well",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        linespacing=1.25,
    )


def _add_curve_legend(ax: plt.Axes) -> None:
    """Define animal curves and their equal-animal mean for panels C--E."""
    ax.axis("off")
    ax.legend(
        handles=_animal_curve_handles(),
        title="Panels c–e: curves",
        loc="center left",
        borderaxespad=0,
        handlelength=1.5,
        labelspacing=0.55,
        fontsize=7,
        title_fontsize=7,
        frameon=False,
    )


def _add_model_legend(ax: plt.Axes, mu_overall: pd.DataFrame) -> None:
    """Define all model symbols and report the fit-quality-clean summary."""
    ax.axis("off")
    clean_overall = mu_overall[mu_overall["cohort"].eq("clean")].iloc[0]
    ax.text(
        0,
        0.98,
        "Panel f: adjusted model\n"
        f"animal median β = {clean_overall['coefficient']:.2f}\n"
        f"exp(β) = {clean_overall['relative_weight']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        linespacing=1.35,
    )
    symbol_handles = [
        Line2D(
            [0],
            [0],
            marker=".",
            color="#a9a9a9",
            linestyle="none",
            markersize=6,
            label="clean session",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#bdbdbd",
            linestyle="none",
            markersize=5,
            label="fit-quality flag",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markeredgecolor="black",
            markerfacecolor=WONG[5],
            markersize=5,
            label="clean animal median",
        ),
        Line2D(
            [0],
            [0],
            color="#666666",
            linestyle="--",
            linewidth=0.7,
            label="zero effect",
        ),
    ]
    ax.legend(
        handles=symbol_handles,
        loc="lower left",
        borderaxespad=0,
        handlelength=1.5,
        labelspacing=0.55,
        fontsize=7,
        frameon=False,
    )


def _validate_and_save(fig: plt.Figure, name: str, output_dir: Path) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.canvas.draw()
    if caught:
        messages = "; ".join(str(item.message) for item in caught)
        raise RuntimeError(f"figure layout emitted warnings: {messages}")
    save_figure(fig, name, base_dir=output_dir)


def make_main_figure(results: dict, output_dir: Path) -> plt.Figure:
    layout = [
        ["A", "A", "A", "A", "LA"],
        ["B", "B", "B", "B", "LB"],
        ["C", "C", "D", "D", "LC"],
        ["E", "E", "F", "F", "LF"],
    ]
    fig, axes = plt.subplot_mosaic(
        layout,
        figsize=(7.0, 6.3),
        width_ratios=[1, 1, 1, 1, 0.88],
        height_ratios=[0.63, 0.63, 1.0, 1.0],
        dpi=450,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, wspace=0.04, hspace=0.05)
    uncertainty_image, content_image = _plot_timeline(
        axes["A"], axes["B"], results["trial"], results["audit"]
    )
    uncertainty_bar = fig.colorbar(
        uncertainty_image, ax=axes["A"], fraction=0.018, pad=0.012, aspect=18
    )
    uncertainty_bar.set_label("Beta variance", fontsize=7, labelpad=5)
    uncertainty_bar.ax.tick_params(labelsize=5, length=2)
    content_bar = fig.colorbar(
        content_image, ax=axes["B"], fraction=0.018, pad=0.012, aspect=18
    )
    content_bar.set_label("event share", fontsize=7, labelpad=5)
    content_bar.ax.tick_params(labelsize=5, length=2)

    _plot_animal_profile(
        axes["C"],
        results["profiles"],
        metric="conditional_event_share",
        scale=100,
        ylabel="allocation to well\n(% of all replay content)",
        title="Conditional allocation to unchosen wells",
        annotation=_positive_animals_note(
            results["contrasts"], "conditional_event_share"
        ),
    )
    _plot_animal_profile(
        axes["D"],
        results["profiles"],
        metric="absolute_rate",
        scale=1,
        ylabel="posterior-weighted\ndecoder bins s$^{-1}$",
        title="Absolute content rate at unchosen wells",
        annotation=_positive_animals_note(results["contrasts"], "absolute_rate"),
    )
    _plot_response_robustness(axes["E"], results["contrasts"])
    _plot_mu_coefficients(
        axes["F"],
        results["mu_sessions"],
        results["mu_animals"],
    )
    _add_behavior_legend(axes["LA"])
    _add_replay_availability_legend(axes["LB"])
    _add_curve_legend(axes["LC"])
    _add_model_legend(axes["LF"], results["mu_overall"])
    label_panels(
        [axes[label] for label in ["A", "B", "C", "D", "E", "F"]],
        xpos=-0.07,
        ypos=1.06,
        fontsize=8,
    )
    _validate_and_save(fig, "uncertainty_content_story", output_dir)
    return fig


def _plot_scope_comparison(
    ax: plt.Axes,
    profiles: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    scale: float = 1,
) -> None:
    for scope in ["all", "unchosen"]:
        subset = _profile(
            profiles,
            metric=metric,
            response="posterior",
            scope=scope,
            level="overall",
        )
        ax.plot(
            subset["uncertainty_quartile"],
            scale * subset["value"],
            color=SCOPE_COLORS[scope],
            marker="o" if scope == "all" else "s",
            linewidth=1.4,
            label=scope,
        )
    ax.set_xticks([1, 2, 3, 4], ["Q1", "Q2", "Q3", "Q4"])
    ax.set_xlabel("same-well uncertainty quartile", labelpad=7)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_title(title, loc="left", pad=4)
    ax.spines[["top", "right"]].set_visible(False)


def _add_scope_legend(ax: plt.Axes) -> None:
    """Define the scope encodings used in diagnostic panels A and B."""
    ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            color=SCOPE_COLORS[scope],
            marker="o" if scope == "all" else "s",
            linewidth=1.4,
            label="all wells" if scope == "all" else "unchosen wells",
        )
        for scope in ["all", "unchosen"]
    ]
    ax.legend(
        handles=handles,
        title="Panels a–b: scope",
        loc="upper left",
        borderaxespad=0,
        handlelength=1.5,
        labelspacing=0.65,
        fontsize=7,
        title_fontsize=7,
        frameon=False,
    )
    ax.text(
        0,
        0.56,
        "Each curve is an\nequal-animal mean.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        linespacing=1.25,
    )


def _add_diagnostic_curve_legend(ax: plt.Axes) -> None:
    """Define animal, mean, and null-reference encodings in panels C and D."""
    ax.axis("off")
    handles = _animal_curve_handles(include_mean=False)
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="black",
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                label="equal-animal mean (c)",
            ),
            Line2D(
                [0],
                [0],
                color="#777777",
                linestyle="--",
                linewidth=0.7,
                label="zero effect (d)",
            ),
        ]
    )
    ax.legend(
        handles=handles,
        title="Panels c–d: curves",
        loc="center left",
        borderaxespad=0,
        handlelength=1.5,
        labelspacing=0.5,
        fontsize=7,
        title_fontsize=7,
        frameon=False,
    )


def _plot_horserace(ax: plt.Axes, horserace_summary: pd.DataFrame) -> None:
    """Within-arm horse-race: off-current content is driven by belief uncertainty,
    not by its parallels (distance, recency) or by value.

    Per-predictor across-session median coefficient (per 1 SD) for the pooled
    population, arm fixed-effects, non-current leaf arms. Uncertainty dominates and
    is positive; distance flips negative, value/recency are ~null — the confound
    defense for the Mu uncertainty effect.
    """
    order = [
        "reward_belief_variance_candidate_arm_preoutcome_z",
        "reward_belief_mean_candidate_arm_preoutcome_z",
        "distance_to_candidate_arm_cm_z",
        "visit_recency_trials_z",
    ]
    labels = {
        "reward_belief_variance_candidate_arm_preoutcome_z": "belief\nuncertainty",
        "reward_belief_mean_candidate_arm_preoutcome_z": "beta value",
        "distance_to_candidate_arm_cm_z": "distance",
        "visit_recency_trials_z": "visit recency",
    }
    table = horserace_summary[horserace_summary["population"] == "all"].set_index(
        "predictor"
    )
    y = np.arange(len(order))[::-1]
    label_transform = ax.get_yaxis_transform()  # x in axes fraction, y in data
    for yi, predictor in zip(y, order, strict=True):
        if predictor not in table.index:
            continue
        row = table.loc[predictor]
        color = (
            "#CC6677"
            if predictor == "reward_belief_variance_candidate_arm_preoutcome_z"
            else "#999999"
        )
        ax.barh(yi, row["median"], color=color, height=0.62, zorder=2)
        stars = (
            "***"
            if row["wilcoxon_p"] < 1e-3
            else "*"
            if row["wilcoxon_p"] < 0.05
            else "n.s."
        )
        # significance / animal-consistency at the right margin (avoids bar/label overlap)
        ax.text(
            0.99,
            yi,
            f"{stars} {int(row['n_animals_consistent'])}/{int(row['n_animals'])}",
            transform=label_transform,
            va="center",
            ha="right",
            fontsize=6,
        )
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks(y, [labels[p] for p in order])
    ax.set_xlabel("within-arm content coefficient (per 1 SD)", labelpad=6)
    ax.set_title(
        "Horse-race: uncertainty, not value / distance / recency", loc="left", pad=4
    )
    ax.spines[["top", "right"]].set_visible(False)


def make_diagnostic_figure(results: dict, output_dir: Path) -> plt.Figure:
    layout = [
        ["A", "A", "B", "B", "LS"],
        ["C", "C", "D", "D", "LC"],
        ["E", "E", "E", "E", "."],
    ]
    fig, axes = plt.subplot_mosaic(
        layout,
        figsize=(7.0, 6.0),
        width_ratios=[1, 1, 1, 1, 0.88],
        height_ratios=[1, 1, 0.85],
        dpi=450,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, wspace=0.04, hspace=0.06)
    _plot_scope_comparison(
        axes["A"],
        results["profiles"],
        metric="conditional_event_share",
        scale=100,
        ylabel="allocation to well\n(% of all replay content)",
        title="Conditional allocation: all vs unchosen",
    )
    _plot_scope_comparison(
        axes["B"],
        results["profiles"],
        metric="absolute_rate",
        ylabel="posterior-weighted\ndecoder bins s$^{-1}$",
        title="Absolute rate: all vs unchosen",
    )

    chosen_animals = _profile(
        results["profiles"],
        metric="chosen_prevalence",
        response="binary",
        scope="all",
        level="animal",
    )
    chosen_overall = _profile(
        results["profiles"],
        metric="chosen_prevalence",
        response="binary",
        scope="all",
        level="overall",
    )
    for animal in ANIMAL_ORDER:
        subset = chosen_animals[chosen_animals["animal"].eq(animal)]
        axes["C"].plot(
            subset["uncertainty_quartile"],
            100 * subset["value"],
            color=ANIMAL_COLORS[animal],
            marker=ANIMAL_MARKERS[animal],
            markerfacecolor="white",
            markeredgewidth=0.8,
            linewidth=0.9,
        )
    axes["C"].plot(
        chosen_overall["uncertainty_quartile"],
        100 * chosen_overall["value"],
        color="black",
        marker="o",
        linewidth=1.8,
        markersize=3.5,
    )
    axes["C"].set_xticks([1, 2, 3, 4], ["Q1", "Q2", "Q3", "Q4"])
    axes["C"].set_xlabel("same-well uncertainty quartile", labelpad=7)
    axes["C"].set_ylabel("trials choosing that well (%)", labelpad=8)
    axes["C"].set_title("Chosen-well bias runs opposite the effect", loc="left", pad=4)
    axes["C"].spines[["top", "right"]].set_visible(False)

    mu_animals = results["mu_animals"]
    for animal in ANIMAL_ORDER:
        subset = mu_animals[mu_animals["animal"].eq(animal)].set_index("cohort")
        axes["D"].plot(
            [0, 1],
            subset.loc[["all", "clean"], "coefficient"],
            color=ANIMAL_COLORS[animal],
            marker=ANIMAL_MARKERS[animal],
            markerfacecolor="white",
            markeredgewidth=0.8,
            linewidth=0.9,
        )
    axes["D"].axhline(0, color="#777777", linewidth=0.7, linestyle="--")
    axes["D"].set_xticks([0, 1], ["all sessions", "fit-quality clean"])
    axes["D"].set_ylabel("Mu uncertainty coefficient", labelpad=8)
    axes["D"].set_title("Adjusted effect survives fit exclusions", loc="left", pad=4)
    axes["D"].spines[["top", "right"]].set_visible(False)

    _plot_horserace(axes["E"], results["horserace_summary"])

    _add_scope_legend(axes["LS"])
    _add_diagnostic_curve_legend(axes["LC"])
    label_panels(
        [axes[label] for label in ["A", "B", "C", "D", "E"]],
        xpos=-0.08,
        ypos=1.06,
        fontsize=8,
    )
    _validate_and_save(fig, "uncertainty_content_diagnostics", output_dir)
    return fig


def _write_caption(output_dir: Path, audit: dict) -> None:
    caption = f"""# Reward-belief uncertainty and replay content

**A–B, representative session.** Pre-outcome Beta-posterior variance and the
posterior allocation of decoded replay content among six reward wells. Hollow
squares mark the chosen well and orange circles mark rewarded choices; gray
cells indicate trials without replay. The
session ({audit["example_animal"]}, {audit["example_session_date"]},
{audit["example_epoch"]}) was selected by an association-blind cohort-medoid
rule. **C, conditional allocation.** Within each session and physical well,
trials were divided into uncertainty quartiles using all mapped trial
opportunities. Among replay-bearing trials, unchosen wells received a larger
fraction of all replay content in higher uncertainty quartiles. Colors and open
symbols identify animals; black gives their equal-weight mean. **D, absolute content rate.**
Posterior-weighted decoder bins per immobility second, retaining trials without
replay as zeros. **E, response robustness.** Paired high-minus-low uncertainty
contrasts using hard MAP counts, soft MAP counts, or full posterior mass.
**F, adjusted model.** Existing per-session Mu conditional-Poisson coefficients
after arm identity, current/previous arm, reward history, visit recency, and
last-rewarded-location controls. Gray points are clean-session coefficients,
crosses mark fit-quality flags, colored filled symbols are clean-session animal
medians, and the dashed vertical line marks zero effect.

Uncertainty denotes uncertainty about reward probability at a well, not animal
position or decoder confidence. Panels C and F estimate relative content
allocation. Panel D is an exploratory absolute-rate extension. All associations
are correlational.
"""
    (output_dir / "uncertainty_content_caption.md").write_text(caption)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    set_figure_defaults(context="paper")
    results = _load_results(args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_figure = make_main_figure(results, args.output_dir)
    diagnostic_figure = make_diagnostic_figure(results, args.output_dir)
    _write_caption(args.output_dir, results["audit"])
    if args.show:
        plt.show()
    else:
        plt.close(main_figure)
        plt.close(diagnostic_figure)


if __name__ == "__main__":
    main()
