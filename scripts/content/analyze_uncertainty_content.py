#!/usr/bin/env python
"""Build trial-level reward-belief uncertainty/replay-content results.

This is the supported, results-of-record version of the ground-up uncertainty
analysis.  It preserves zero-replay trials for absolute content-rate analyses,
aggregates repeated stops/events to the behavioral trial grain, and balances
physical wells, sessions, and animals hierarchically.

Usage
-----
uv run python scripts/content/analyze_uncertainty_content.py
uv run python scripts/content/analyze_uncertainty_content.py --cohort all
uv run python scripts/content/analyze_uncertainty_content.py --allow-stale
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.analysis.replay.fit_provenance import (
    require_fresh_inputs,
)
from continuum_swr_replay.analysis.replay.model_specs import LEAF_ONLY_N_CATEGORIES
from continuum_swr_replay.analysis.replay.standardization import (
    assert_standardized_matches_reference,
    load_reference,
)
from continuum_swr_replay.analysis.replay.uncertainty_content import (
    build_trial_leaf_table,
    hierarchical_quartile_profile,
    paired_quartile_contrasts,
    summarize_horserace,
    summarize_mu_coefficients,
    within_arm_horserace,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.table_io import read_table, table_exists

# Sibling-script import: the Mu bundle name is derived from the fitting runner's
# own helpers, so this reader cannot drift from where the belief chain publishes
# Mu.pkl. Run as a script, only this file's directory is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.content.fit_replay_content_models import (  # noqa: E402
    belief_chain_output_subdir,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_DATA_DIR = PROCESSED_DATA_DIR / "replay_content"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "uncertainty_content"
DEFAULT_EXAMPLE_SESSION = ("wilbur", "20210329", "08_r4")


def mu_inputs(data_dir: Path) -> dict[str, Path]:
    """The table fit_replay_content_models.py fit ``models/Mu.pkl`` on.

    Checked against the fit's provenance sidecar before the coefficients are
    summarized, so an uncertainty result cannot be built from a rung that
    predates the event-arm table it is being described against.
    """
    return {"event_arm_table": data_dir / "event_arm_table.pkl"}


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser (also the provenance gate's opt-out)."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Replay-content artifact directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
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
        choices=["mu", "all"],
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


def _parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


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


def _add_result_metadata(
    frame: pd.DataFrame,
    *,
    metric: str,
    response: str,
    scope: str,
    level: str,
) -> pd.DataFrame:
    return frame.assign(
        metric=metric,
        response=response,
        scope=scope,
        level=level,
    )


def _select_example_session(trial_leaf: pd.DataFrame) -> dict[str, str]:
    animal, session_date, epoch = DEFAULT_EXAMPLE_SESSION
    present = (
        trial_leaf["animal"].astype(str).eq(animal)
        & trial_leaf["session_date"].astype(str).eq(session_date)
        & trial_leaf["epoch"].astype(str).eq(epoch)
    )
    if present.any():
        return {
            "animal": animal,
            "session_date": session_date,
            "epoch": epoch,
            "selection": "association-blind cohort medoid selected before plotting",
        }

    # Portable fallback for alternate input cohorts: select the robust medoid on
    # trial count, event count, and within-well uncertainty range.  No
    # uncertainty-content association enters this rule.
    trial = trial_leaf.drop_duplicates(["session_id", "trial_number_by_epoch"])
    counts = trial.groupby(
        ["animal", "session_date", "epoch", "session_id"], as_index=False
    ).agg(
        n_trials=("trial_number_by_epoch", "nunique"),
        n_events=("n_events", "sum"),
    )
    uncertainty_range = (
        trial_leaf.groupby(["session_id", "candidate_arm"], observed=True)["reward_belief_variance_candidate_arm_preoutcome"]
        .agg(lambda x: x.max() - x.min())
        .groupby("session_id")
        .median()
        .rename("uncertainty_range")
    )
    counts = counts.merge(uncertainty_range, on="session_id", validate="one_to_one")
    features = ["n_trials", "n_events", "uncertainty_range"]
    distance = np.zeros(len(counts), dtype=float)
    for feature in features:
        values = counts[feature].to_numpy(dtype=float)
        median = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - median))
        scale = 1.4826 * mad if mad > 0 else np.nanstd(values)
        if not np.isfinite(scale) or scale == 0:
            continue
        distance += ((values - median) / scale) ** 2
    counts["distance"] = np.sqrt(distance)
    selected = counts.sort_values(["distance", "session_id"]).iloc[0]
    return {
        "animal": str(selected["animal"]),
        "session_date": str(selected["session_date"]),
        "epoch": str(selected["epoch"]),
        "selection": "fallback robust medoid; association not used",
    }


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

    def select_animal_contrast(metric: str, response: str, scope: str) -> pd.DataFrame:
        return contrasts[
            contrasts["metric"].eq(metric)
            & contrasts["response"].eq(response)
            & contrasts["scope"].eq(scope)
            & contrasts["level"].eq("animal")
        ]

    conditional = select_profile("conditional_event_share", "posterior", "unchosen")
    leaf_conditional = select_profile("conditional_leaf_share", "posterior", "unchosen")
    absolute = select_profile("absolute_rate", "posterior", "unchosen")
    conditional_animals = select_animal_contrast(
        "conditional_event_share", "posterior", "unchosen"
    )
    leaf_conditional_animals = select_animal_contrast(
        "conditional_leaf_share", "posterior", "unchosen"
    )
    absolute_animals = select_animal_contrast("absolute_rate", "posterior", "unchosen")
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
        f"{int((conditional_animals['contrast'] > 0).sum())}/{len(conditional_animals)} "
        "animals; equal-animal mean "
        f"{conditional_animals['contrast'].mean():.4f}.",
        "- Leaf-normalized sensitivity across the same quartiles: "
        f"{_format_profile(leaf_conditional)}; Q4-Q1 positive in "
        f"{int((leaf_conditional_animals['contrast'] > 0).sum())}/"
        f"{len(leaf_conditional_animals)} animals.",
        "- Absolute posterior-weighted decoder bins/s across the same quartiles, with "
        f"zero-replay trials retained: {_format_profile(absolute)}.",
        "- Absolute-rate Q4-Q1 was positive in "
        f"{int((absolute_animals['contrast'] > 0).sum())}/{len(absolute_animals)} "
        f"animals; equal-animal mean {absolute_animals['contrast'].mean():.4f}.",
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


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    event_arm_path = args.data_dir / "event_arm_table.pkl"
    stop_path = args.data_dir / "stop_table.pkl"
    qvals_path = args.data_dir / "models_per_bin_rate" / "qvals_table.pkl"
    standardization_path = args.data_dir / "standardization_reference.json"
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
    require_fresh_inputs(
        mu_path, mu_inputs(args.data_dir), allow_stale=args.allow_stale
    )
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

    if args.cohort == "mu":
        valid_sessions = set(mu_summary["session"]["session_id"])
        trial_leaf = trial_leaf[trial_leaf["session_id"].isin(valid_sessions)].copy()
        horserace = horserace[horserace["session_id"].isin(valid_sessions)].copy()
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
            "n_analysis_sessions": int(trial_leaf["session_id"].nunique()),
            "n_analysis_trials": int(
                trial_leaf[["session_id", "trial_number_by_epoch"]]
                .drop_duplicates()
                .shape[0]
            ),
            "n_analysis_replay_trials": int(
                trial_leaf.drop_duplicates(["session_id", "trial_number_by_epoch"])[
                    "has_replay"
                ].sum()
            ),
            "n_analysis_trial_leaf_rows": int(len(trial_leaf)),
        }
    )
    example = _select_example_session(trial_leaf)
    audit.update({f"example_{key}": value for key, value in example.items()})

    configurations = []
    for metric, response_columns in [
        (
            "conditional_event_share",
            [
                ("hard", "hard_event_share"),
                ("soft", "soft_event_share"),
                ("posterior", "posterior_event_share"),
            ],
        ),
        (
            "conditional_leaf_share",
            [
                ("hard", "hard_leaf_share"),
                ("soft", "soft_leaf_share"),
                ("posterior", "posterior_leaf_share"),
            ],
        ),
    ]:
        for response, share in response_columns:
            for scope in ["all", "unchosen"]:
                configurations.append(
                    {
                        "metric": metric,
                        "response": response,
                        "scope": scope,
                        "numerator": share,
                        "denominator": None,
                    }
                )
    for response, mass in [
        ("hard", "hard_count"),
        ("soft", "soft_count"),
        ("posterior", "posterior_mass"),
    ]:
        for scope in ["all", "unchosen"]:
            configurations.append(
                {
                    "metric": "absolute_rate",
                    "response": response,
                    "scope": scope,
                    "numerator": mass,
                    "denominator": "immobility_duration_s",
                }
            )

    profile_frames = []
    contrast_frames = []
    LOGGER.info("Computing hierarchical quartile profiles")
    for config in configurations:
        profile = hierarchical_quartile_profile(
            trial_leaf,
            numerator_col=config["numerator"],
            denominator_col=config["denominator"],
            scope=config["scope"],
        )
        for level, frame in profile.items():
            if frame.empty:
                continue
            profile_frames.append(
                _add_result_metadata(
                    frame,
                    metric=config["metric"],
                    response=config["response"],
                    scope=config["scope"],
                    level=level,
                )
            )
        contrasts = paired_quartile_contrasts(profile["candidate_arm"])
        for level, frame in contrasts.items():
            if frame.empty:
                continue
            contrast_frames.append(
                _add_result_metadata(
                    frame,
                    metric=config["metric"],
                    response=config["response"],
                    scope=config["scope"],
                    level=level,
                )
            )

    profiles = pd.concat(profile_frames, ignore_index=True, sort=False)
    contrasts = pd.concat(contrast_frames, ignore_index=True, sort=False)

    chosen_profile = hierarchical_quartile_profile(
        trial_leaf.assign(is_chosen_value=trial_leaf["is_chosen"].astype(float)),
        numerator_col="is_chosen_value",
        scope="all",
    )
    for level, frame in chosen_profile.items():
        if frame.empty:
            continue
        profiles = pd.concat(
            [
                profiles,
                _add_result_metadata(
                    frame,
                    metric="chosen_prevalence",
                    response="binary",
                    scope="all",
                    level=level,
                ),
            ],
            ignore_index=True,
            sort=False,
        )

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
    horserace_summary.to_csv(
        args.output_dir / "horserace_summary.csv", index=False
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
