#!/usr/bin/env python
"""Build trial-level reward-belief uncertainty/replay-content results.

This is the supported, results-of-record version of the ground-up uncertainty
analysis.  It preserves zero-replay trials for absolute content-rate analyses,
aggregates repeated stops/events to the behavioral trial grain, and balances
physical wells, sessions, and animals hierarchically.

Usage
-----
uv run python scripts/behavior/content/analyze_uncertainty_content.py
uv run python scripts/behavior/content/analyze_uncertainty_content.py --cohort all
uv run python scripts/behavior/content/analyze_uncertainty_content.py --allow-stale
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.behavior.content.fit_outputs import (
    belief_chain_output_subdir,
)
from continuum_swr_replay.analysis.behavior.content.uncertainty_content import (
    ANALYSIS_COHORTS,
    HORSERACE_SUMMARY,
    analysis_cohort_counts,
    build_trial_leaf_table,
    quartile_profiles_and_contrasts,
    restrict_to_analysis_cohort,
    select_example_session,
    summarize_horserace,
    summarize_mu_coefficients,
    within_arm_horserace,
)
from continuum_swr_replay.analysis.behavior.model_specs import LEAF_ONLY_N_CATEGORIES
from continuum_swr_replay.analysis.events.analysed_time import require_swr_source
from continuum_swr_replay.paths import (
    STANDARDIZATION_REFERENCE,
    UNCERTAINTY_CONTENT_DIR,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.fit_provenance import (
    record_inputs,
    require_fresh_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.tables.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table, table_exists
from continuum_swr_replay.tables.standardization import (
    assert_standardized_matches_reference,
    load_reference,
)

LOGGER = logging.getLogger(__name__)


def mu_inputs(data_dir: Path) -> dict[str, Path]:
    """The table fit_replay_content_models.py fit ``models/Mu.pkl`` on.

    Checked against the fit's provenance sidecar before the coefficients are
    summarized, so an uncertainty result cannot be built from a rung that
    predates the event-arm table it is being described against.
    """
    return {"event_arm_table": table_path("event_arm_table", data_dir)}


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser (also the provenance gate's opt-out)."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=replay_content_dir(),
        help="Replay-content artifact directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=replay_content_dir() / UNCERTAINTY_CONTENT_DIR,
        help="Directory for derived uncertainty-content tables.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help=(
            "Bundle directory holding Mu.pkl. Defaults to the directory the "
            "leaf-only belief chain publishes into, derived from the fitting "
            "runner itself (currently "
            f"{belief_chain_output_subdir()}/, relative to --data-dir)."
        ),
    )
    parser.add_argument(
        "--cohort",
        choices=list(ANALYSIS_COHORTS),
        default="mu",
        help="Use sessions represented in Mu.pkl (default) or all mapped sessions.",
    )
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help=(
            "Summarize the Mu fit even if its input table has changed since it "
            "was fit (or it carries no provenance sidecar at all)."
        ),
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_models_dir(data_dir: Path, models_dir: Path | None = None) -> Path:
    """Bundle directory holding the ``Mu.pkl`` this reader summarizes.

    ``Mu`` is fit only by the leaf-only belief chain, which publishes into its
    own partial-ladder bundle rather than ``models/``. The name is taken from
    the fitting runner's own helper so the reader cannot be left pointing at a
    directory the fitter no longer writes.

    Parameters
    ----------
    data_dir : Path
        Replay-content artifact directory.
    models_dir : Path or None, optional
        Explicit override from ``--models-dir``.

    Returns
    -------
    Path
        Directory expected to contain ``Mu.pkl``.
    """
    if models_dir is not None:
        return models_dir
    return data_dir / belief_chain_output_subdir()


def _format_profile(values: pd.DataFrame) -> str:
    ordered = values.sort_values("uncertainty_quartile")["value"].to_numpy()
    return ", ".join(f"{value:.4f}" for value in ordered)


def _write_summary(
    output_path: Path,
    profiles: pd.DataFrame,
    contrasts: pd.DataFrame,
    mu_summary: dict[str, pd.DataFrame],
    audit: dict[str, float | int | str],
) -> None:
    def select_profile(metric: str, response: str, scope: str) -> pd.DataFrame:
        return profiles[
            profiles["metric"].eq(metric)
            & profiles["response"].eq(response)
            & profiles["scope"].eq(scope)
            & profiles["level"].eq("overall")
        ]

    def select_overall_contrast(metric: str, response: str, scope: str) -> pd.Series:
        """The ``overall`` row of :func:`paired_quartile_contrasts`."""
        rows = contrasts[
            contrasts["metric"].eq(metric)
            & contrasts["response"].eq(response)
            & contrasts["scope"].eq(scope)
            & contrasts["level"].eq("overall")
        ]
        if rows.empty:
            raise ValueError(
                f"paired_quartile_contrasts gave no overall row for {metric} "
                f"({response} response, {scope} wells)"
            )
        return rows.iloc[0]

    def positive_animals(contrast: pd.Series) -> str:
        return f"{int(contrast['n_animals_positive'])}/{int(contrast['n_animals'])}"

    conditional = select_profile("conditional_event_share", "posterior", "unchosen")
    leaf_conditional = select_profile("conditional_leaf_share", "posterior", "unchosen")
    absolute = select_profile("absolute_rate", "posterior", "unchosen")
    conditional_contrast = select_overall_contrast(
        "conditional_event_share", "posterior", "unchosen"
    )
    leaf_conditional_contrast = select_overall_contrast(
        "conditional_leaf_share", "posterior", "unchosen"
    )
    absolute_contrast = select_overall_contrast(
        "absolute_rate", "posterior", "unchosen"
    )
    mu_clean = mu_summary["overall"].query("cohort == 'clean'").iloc[0]

    lines = [
        "# Ground-up reward-belief uncertainty/content analysis",
        "",
        "## Cohort",
        "",
        f"- {audit['n_analysis_sessions']} sessions, {audit['n_analysis_trials']} mapped trials, "
        f"{audit['n_analysis_replay_trials']} replay-bearing trials.",
        f"- Stop-to-belief coverage: {100 * float(audit['stop_qvals_coverage']):.2f}%.",
        f"- Event-to-belief coverage: {100 * float(audit['event_qvals_coverage']):.2f}%.",
        "- Each trial contributes six reward-well candidate rows; path/trunk content "
        "remains in the full-event denominator.",
        "",
        "## Model-light results",
        "",
        "- Conditional share of all replay content allocated to the well across "
        "within-well uncertainty quartiles "
        f"(unchosen wells): {_format_profile(conditional)}.",
        "- Conditional Q4-Q1 was positive in "
        f"{positive_animals(conditional_contrast)} animals; equal-animal mean "
        f"{conditional_contrast['contrast']:.4f}.",
        "- Leaf-normalized sensitivity across the same quartiles: "
        f"{_format_profile(leaf_conditional)}; Q4-Q1 positive in "
        f"{positive_animals(leaf_conditional_contrast)} animals.",
        "- Absolute posterior-weighted decoder bins/s across the same quartiles, with "
        f"zero-replay trials retained: {_format_profile(absolute)}.",
        "- Absolute-rate Q4-Q1 was positive in "
        f"{positive_animals(absolute_contrast)} animals; equal-animal mean "
        f"{absolute_contrast['contrast']:.4f}.",
        "",
        "## Existing adjusted model",
        "",
        "- Fit-quality-clean Mu animal-median coefficient: "
        f"{mu_clean['coefficient']:.3f} log relative-weight units; "
        f"exp(beta)={mu_clean['relative_weight']:.2f}.",
        f"- Mu is a {LEAF_ONLY_N_CATEGORIES}-leaf conditional-Poisson model "
        "conditioned on an observed replay event, with each event's response "
        "renormalized to unit mass over the leaves; it does not estimate event "
        "generation.",
        "",
        "## Claim boundary",
        "",
        "These results support enrichment of replay content at more uncertain reward "
        "wells. The absolute-rate panel is an exploratory extension and remains "
        "correlational; uncertainty is strongly coupled to visit recency by the "
        "behavioral model's forgetting rule.",
        "",
    ]
    output_path.write_text("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    event_arm_path = table_path("event_arm_table", args.data_dir)
    stop_path = table_path("stop_table", args.data_dir)
    qvals_path = table_path("qvals_table", args.data_dir)
    standardization_path = args.data_dir / STANDARDIZATION_REFERENCE
    models_dir = resolve_models_dir(args.data_dir, args.models_dir)
    mu_path = models_dir / "Mu.pkl"
    for path in [event_arm_path, stop_path, qvals_path]:
        if not table_exists(path):
            raise FileNotFoundError(path)
    if not mu_path.exists():
        raise FileNotFoundError(mu_path)
    if not standardization_path.exists():
        raise FileNotFoundError(
            f"{standardization_path} is required to validate the uncertainty unit"
        )

    LOGGER.info("Loading canonical tables")
    event_arm = read_table(event_arm_path)
    stops = read_table(stop_path)
    qvals = read_table(qvals_path)
    require_covariate_schema(event_arm, table="event_arm")
    require_covariate_schema(stops, table="stop")
    require_covariate_schema(qvals, table="qvals")
    require_swr_source(event_arm, what=str(event_arm_path))
    require_swr_source(stops, what=str(stop_path))
    require_fresh_inputs(
        mu_path, mu_inputs(args.data_dir), allow_stale=args.allow_stale
    )
    # Hash the inputs now, while they are being read. The tables are the
    # sidecar's inputs; the Mu fit (it only picks the session cohort) is
    # recorded beside them, since it lives in another directory.
    fit_inputs = record_inputs(
        {
            "event_arm_table": event_arm_path,
            "stop_table": stop_path,
            "qvals_table": qvals_path,
        }
    )
    mu_record = record_inputs({"Mu": mu_path})["Mu"]
    mu = pd.read_pickle(mu_path)
    try:
        assert_standardized_matches_reference(
            event_arm,
            ("reward_belief_variance_candidate_arm_preoutcome",),
            load_reference(standardization_path),
            support_label="leaf_arms",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "The uncertainty-content inputs use a stale or invalid "
            "arm_uncertainty_z unit. Regenerate the canonical event-arm table, "
            f"leaf-support reference, and Mu fit before analysis: {exc}"
        ) from exc

    LOGGER.info("Building trial x reward-well table")
    trial_leaf, audit = build_trial_leaf_table(event_arm, stops, qvals)
    mu_summary = summarize_mu_coefficients(mu)

    LOGGER.info("Within-arm value x uncertainty horse-race (exploit vs explore)")
    horserace = within_arm_horserace(event_arm)

    trial_leaf, horserace = restrict_to_analysis_cohort(
        trial_leaf, horserace, mu_summary, args.cohort
    )
    horserace_summary = summarize_horserace(horserace)

    audit.update(
        {
            "analysis_cohort": args.cohort,
            "arm_standardization_support": "leaf_arms",
            # The Mu fit summarized here is the leaf-only belief chain: its
            # conditional likelihood conditions on the six reward-site arms, and
            # each event's response is renormalized to unit mass over them.
            "mu_bundle": str(models_dir),
            "mu_n_categories": LEAF_ONLY_N_CATEGORIES,
            "mu_response_renormalized": True,
            **analysis_cohort_counts(trial_leaf),
        }
    )
    example = select_example_session(trial_leaf)
    audit.update({f"example_{key}": value for key, value in example.items()})

    LOGGER.info("Computing hierarchical quartile profiles")
    profiles, contrasts = quartile_profiles_and_contrasts(trial_leaf)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trial_leaf.to_pickle(args.output_dir / "trial_leaf_table.pkl")
    profiles.to_csv(args.output_dir / "quartile_profiles.csv", index=False)
    contrasts.to_csv(args.output_dir / "quartile_contrasts.csv", index=False)
    mu_summary["session"].to_csv(
        args.output_dir / "mu_session_coefficients.csv", index=False
    )
    mu_summary["animal"].to_csv(
        args.output_dir / "mu_animal_coefficients.csv", index=False
    )
    mu_summary["overall"].to_csv(
        args.output_dir / "mu_overall_coefficients.csv", index=False
    )
    horserace.to_csv(
        args.output_dir / "horserace_session_coefficients.csv", index=False
    )
    horserace_summary.to_csv(args.output_dir / HORSERACE_SUMMARY, index=False)
    for artifact in (HORSERACE_SUMMARY, "horserace_session_coefficients.csv"):
        write_fit_provenance(
            args.output_dir / artifact,
            fit_inputs,
            "ols_within_arm",
            [__file__, *(argv if argv is not None else sys.argv[1:])],
            extra={
                "cohort": args.cohort,
                "mu_fit": {"path": str(mu_path), **mu_record},
            },
        )
    (args.output_dir / "cohort_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True)
    )
    _write_summary(
        args.output_dir / "analysis_summary.md",
        profiles,
        contrasts,
        mu_summary,
        audit,
    )

    LOGGER.info(
        "Wrote %d trial x well rows and result summaries to %s",
        len(trial_leaf),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
