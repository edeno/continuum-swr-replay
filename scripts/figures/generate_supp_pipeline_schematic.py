#!/usr/bin/env python
"""Generate a supplemental schematic figure of the replay-content pipeline.

Visualises every transformation from raw decoder output through the
fitted conditional Poisson, with data shapes and the load-bearing
assumption attached to each step.

Output:
- ``paper/figures/qc_diagnosis/pipeline_schematic.{png,pdf}``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from continuum_swr_replay.figure_utilities import save_figure

STEPS = [
    {
        "title": "0. Decoder output",
        "shape": "T × (map_estimate, Non-Local_prob, head_speed, SWR_event)",
        "assumption": "Place-bin discretization is fine; emission/transition priors are correct.",
    },
    {
        "title": "1. MAP → arm  (map_estimate_to_arm)",
        "shape": "T × replay_arm ∈ {0..8} ∪ NaN",
        "assumption": "Junction bins (edge_id = −1) are interpolation artifacts, not real positions.",
    },
    {
        "title": "2. Add trial context  (add_trial_context)",
        "shape": "T × {prev_well, next_well, last_rewarded_well, switch/omission flags}",
        "assumption": "In-progress trip semantics: latest start_time ≤ t.  Trials are non-overlapping.",
    },
    {
        "title": "3. Filter to non-local immobile",
        "shape": "T_nl rows  (accepted labels: every bin speed < 4 and prob > 0.8)",
        "assumption": "Thresholds cleanly separate non-local from local; immobility is required.",
    },
    {
        "title": "4. Collapse to events  (collapse_to_events)",
        "shape": "E events  (modal arm/well + first-row onset flags)",
        "assumption": "Modal arm represents the event;  onset flags use the first timebin.\nReal data: 0.12 % of events straddle a trial boundary  →  modal-context is unambiguous.",
    },
    {
        "title": "5. Compute arm counts  (compute_arm_counts)",
        "shape": "E × (arm_count_0 ... arm_count_8, total_count L_e)",
        "assumption": "Each timebin contributes 1 count to 1 arm  (MAP-collapse, hard assignment).\nReal data: median fragmentation = 0  →  events sit on a single arm.",
    },
    {
        "title": "6. Long format  (make_long_format_arm)",
        "shape": "E·9 rows × {is_current, is_prev, is_next, is_last_rewarded, ...}",
        "assumption": "9 arms are the candidate set for every event. Trunk arms (6,7,8) are 'neutral' to well-derived indicators.",
    },
    {
        "title": "7. Add per-arm + interaction predictors",
        "shape": "+ visit_recency, cumulative_reward, dist_to_arm, swr_×_*, time_×_*, switch_×_*",
        "assumption": "Event-level scalars cancel in conditioning; identifiable only as interactions.",
    },
    {
        "title": "8. Pool sessions / animals  (concat + add_event_uid)",
        "shape": "Σ_sessions  events × 9 arms  with globally unique event_uid",
        "assumption": "Events are exchangeable across animals (no per-animal random effect).\n→ Tested: pooled SE under-covers if per-animal β truly differs.",
    },
    {
        "title": "9. Fit ConditionalPoisson  (statsmodels)",
        "shape": "β̂ , SE, p, AIC, BIC, VIF  per predictor",
        "assumption": "Within-event timebins independent  (multinomial).  Events independent.\n→ Tested: nominal SE under-covers under within-event autocorrelation (ρ≈0.94 on real data).",
    },
]


def draw_schematic(output_dir: Path) -> None:
    n = len(STEPS)
    fig_h = 1.55 * n + 0.6
    fig, ax = plt.subplots(figsize=(11.5, fig_h), constrained_layout=False)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n)
    ax.invert_yaxis()
    ax.axis("off")

    box_color = "#e8eef5"
    edge_color = "#2c3e50"
    title_color = "#1a2433"
    shape_color = "#2c3e50"
    note_color = "#7a3b3b"

    arrow_kwargs = {
        "arrowstyle": "-|>",
        "mutation_scale": 18,
        "linewidth": 1.5,
        "color": edge_color,
        "shrinkA": 0,
        "shrinkB": 0,
    }

    box_w = 9.0
    box_x = 0.5
    box_h = 0.78
    for i, step in enumerate(STEPS):
        cx = box_x
        cy = i + 0.05
        rect = FancyBboxPatch(
            (cx, cy),
            box_w,
            box_h,
            boxstyle="round,pad=0.04,rounding_size=0.06",
            linewidth=1.4,
            edgecolor=edge_color,
            facecolor=box_color,
        )
        ax.add_patch(rect)
        # Title (left column)
        ax.text(
            cx + 0.18,
            cy + 0.20,
            step["title"],
            fontsize=11,
            fontweight="bold",
            color=title_color,
            va="center",
        )
        # Shape (right of title)
        ax.text(
            cx + 0.18,
            cy + 0.43,
            f"shape:  {step['shape']}",
            fontsize=8.5,
            family="monospace",
            color=shape_color,
            va="center",
        )
        # Assumption (smaller, italics, in red-brown)
        ax.text(
            cx + 0.18,
            cy + 0.66,
            step["assumption"],
            fontsize=8.5,
            style="italic",
            color=note_color,
            va="center",
        )

        # Arrow to next step
        if i < n - 1:
            arrow = FancyArrowPatch(
                (cx + box_w / 2, cy + box_h),
                (cx + box_w / 2, (i + 1) + 0.05),
                **arrow_kwargs,
            )
            ax.add_patch(arrow)

    # Title and legend
    fig.suptitle(
        "Replay-content GLM pipeline — transformation steps and load-bearing assumptions",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    # Bottom legend explaining the colour code for italic notes
    legend_handles = [
        mpatches.Patch(color=box_color, label="Transformation step"),
        mpatches.Patch(color="white", label=" "),
    ]
    note_proxy = plt.Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor=note_color,
        markersize=10,
        label="Assumption (red text); Real-data check noted where applicable.",
    )
    legend_handles.append(note_proxy)
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )

    # The PNG at 200 dpi, the PDF at the figure's own dpi.
    save_figure(
        fig,
        "pipeline_schematic",
        base_dir=output_dir,
        dpi=200,
        formats=["png"],
        transparent=False,
    )
    save_figure(
        fig,
        "pipeline_schematic",
        base_dir=output_dir,
        formats=["pdf"],
        transparent=False,
    )
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures/qc_diagnosis"),
    )
    args = parser.parse_args(argv)
    draw_schematic(args.output_dir)
    print(f"Saved schematic to {args.output_dir}/pipeline_schematic.{{png,pdf}}")


if __name__ == "__main__":
    main()
