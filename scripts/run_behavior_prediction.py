#!/usr/bin/env python
"""Test whether replay content predicts subsequent arm choice.

Following Gillespie et al. 2021: aggregates replay events per trial,
then uses leave-one-session-out cross-validation to compare a baseline
model (current arm only) with replay-informed models.

Usage:
    uv run python scripts/run_behavior_prediction.py
"""

import argparse
import logging
import time

import pandas as pd
from scipy.stats import wilcoxon

from continuum_swr_replay.analysis.hippocampus.replay_content.behavior_prediction import (
    aggregate_replay_per_trial,
    cross_validate_choice_prediction,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.data_loaders.constants import NWB_FILES
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.types import SessionDataKeys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)

INPUT_DIR = PROCESSED_DATA_DIR / "replay_content"
OUTPUT_DIR = PROCESSED_DATA_DIR / "replay_content"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.parse_args()

    # Load events
    events_path = INPUT_DIR / "events.pkl"
    if not events_path.exists():
        logging.error("Events not found. Run prepare_replay_content_data.py first.")
        return

    logging.info("Loading events...")
    events = pd.read_pickle(events_path)
    logging.info(
        "Loaded %d events from %d sessions",
        len(events),
        events.groupby(["animal", "day", "epoch"]).ngroups,
    )

    # We need track_graph for well_to_arm_set mapping.
    # Load from the first session's Spyglass data.

    sample_event = events.iloc[0]
    animal_date = f"{sample_event['animal']}{sample_event['day']}"
    nwb = [n for n in NWB_FILES if animal_date in n][0]
    epoch = sample_event["epoch"]

    logging.info("Loading track_graph from %s %s...", nwb, epoch)
    data = load_data(nwb, epoch, include=frozenset({SessionDataKeys.TRIALS}))
    track_graph = data["track_graph"]

    # Aggregate replay events per trial
    logging.info("Aggregating replay events per trial...")
    t0 = time.time()
    bout_data = aggregate_replay_per_trial(events, track_graph)
    logging.info(
        "Aggregated into %d trials in %.1fs",
        len(bout_data),
        time.time() - t0,
    )

    # Filter: require at least 1 replay event per trial
    bout_data = bout_data[bout_data["n_events"] >= 1].copy()
    logging.info("After filtering (n_events >= 1): %d trials", len(bout_data))

    # Summary
    n_sessions = bout_data.groupby(["animal", "day", "epoch"]).ngroups
    logging.info(
        "Trials: %d, Sessions: %d, Events/trial: median=%.0f",
        len(bout_data),
        n_sessions,
        bout_data["n_events"].median(),
    )

    # Cross-validate
    logging.info("Running leave-one-session-out CV...")
    t0 = time.time()
    cv_results = cross_validate_choice_prediction(bout_data)
    elapsed = time.time() - t0
    logging.info("CV completed in %.1fs (%d sessions)", elapsed, len(cv_results))

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "behavior_prediction.pkl"
    cv_results.to_pickle(results_path)
    logging.info("Saved to %s", results_path)

    bout_path = OUTPUT_DIR / "bout_data.pkl"
    bout_data.to_pickle(bout_path)
    logging.info("Saved bout data to %s", bout_path)

    # Print results
    print(f"\n{'=' * 60}")
    print("BEHAVIOR PREDICTION: Does replay predict subsequent choice?")
    print(f"{'=' * 60}")
    n_wells = bout_data["next_well"].nunique()
    print(f"\nTrials: {len(bout_data)}, Sessions: {n_sessions}")
    print(f"Chance accuracy: {1 / n_wells:.3f} (1 of {n_wells} wells)")

    models = ["baseline", "replay_all", "replay_swr", "replay_noswr"]
    print(f"\n{'Model':<20} {'Accuracy':>10} {'Log-loss':>10} {'vs baseline p':>15}")
    print("-" * 58)

    for model_name in models:
        acc_col = f"{model_name}_accuracy"
        ll_col = f"{model_name}_logloss"

        if acc_col not in cv_results.columns:
            continue

        acc = cv_results[acc_col].dropna()
        ll = cv_results[ll_col].dropna()

        p_str = "—"
        if model_name != "baseline" and len(acc) >= 5:
            # Paired comparison with baseline
            paired = cv_results[["baseline_accuracy", acc_col]].dropna()
            if len(paired) >= 5:
                diff = paired[acc_col] - paired["baseline_accuracy"]
                try:
                    _, p = wilcoxon(diff)
                    p_str = f"{p:.2e}"
                except ValueError:
                    pass

        print(f"{model_name:<20} {acc.median():10.3f} {ll.median():10.3f} {p_str:>15}")

    # Per-model detail
    chance = 1 / n_wells
    print(f"\nAbove chance (1/{n_wells} = {chance:.3f})?")
    for model_name in models:
        acc_col = f"{model_name}_accuracy"
        acc = cv_results[acc_col].dropna()
        if len(acc) >= 5:
            try:
                _, p = wilcoxon(acc - chance)
                direction = "ABOVE" if acc.median() > chance else "below"
                print(
                    f"  {model_name:<20}: {acc.median():.3f} ({direction}), p={p:.2e}"
                )
            except ValueError:
                pass


if __name__ == "__main__":
    main()
