"""The mpfc summary's follow-up values: the marked-family models (event
content, strength and ripple power) and the stop-context fits, each pending
when its inputs are stale (:func:`marked_family_staleness`,
:func:`stop_context_staleness`). The marked-family statistics are computed in
``analysis.pfc.marked_family_statistics``, through the summary
``scripts/pfc/summarize_pfc_marked_family.py`` writes."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from continuum_swr_replay.paths import (
    ROOT_DIR,
)
from continuum_swr_replay.reported.mpfc.common import (
    TAGS,
    _finite_adder,
)
from continuum_swr_replay.reported.mpfc.inputs import (
    MpfcInputs,
    _stale_note,
)
from continuum_swr_replay.reported.values import (
    Value,
    file_record,
    rel,
    value_factory,
)
from continuum_swr_replay.runs.io import sha256_file

#: Joint follow-up models and targets the mpfc summary quotes, by key word.
MARKED_MODELS = {
    "content": "replay_content",
    "strength": "event_strength",
    "ripple": "ripple_power",
    "ripplepooled": "ripple_power_pooled",
}
MARKED_TARGETS = {
    "current_arm_fraction": "current_arm",
    "content_weighted_recent_reward": "recent_reward",
    "content_weighted_belief_uncertainty": "uncertainty",
    "log_event_duration_s": "duration",
    "ripple_power_z": "ripple_power",
    "mua_rate_z": "mua_rate",
}
MARKED_CLASSES = {
    "non_local_SWR": "swr_replay",
    "non_local_no_SWR": "non_swr_replay",
    "SWR_no_non_local": "swr_no_replay",
    "pooled": "pooled",
}
#: The (model, target) pairs whose held-out-gain share is quoted, per replay class
#: (the lead, 2026-10-29): replay content's current-arm fraction and recent
#: reward, event strength's multiunit rate and ripple power.
MARKED_HELDOUT_SHARES = {
    ("replay_content", "current_arm_fraction"),
    ("replay_content", "content_weighted_recent_reward"),
    ("event_strength", "mua_rate_z"),
    ("event_strength", "ripple_power_z"),
}
MARKED_HELDOUT_SHARE_CLASSES = ("non_local_SWR", "non_local_no_SWR")
"""The replay classes whose held-out-gain share is quoted for the pairs of
:data:`MARKED_HELDOUT_SHARES` (not the pooled fit, nor SWRs without replay)."""


def stop_context_staleness(
    results: Path, bundle_dir: Path, current: dict[str, str], n_cohort: int
) -> str | None:
    """Why the stop-context fit is stale against the current inputs, or None.

    The fit's ``results.provenance.json`` records the SHA-256 of every canonical
    bundle and cached session table it read; each session table's provenance
    records the event and stop tables it was assembled from. Any of them that
    differs from the current file (matched by file name) makes the fit stale.

    Raises
    ------
    ValueError
        If the fit does not cover the ``n_cohort`` PFC epochs.
    """
    provenance = json.loads(results.with_name("results.provenance.json").read_text())
    if len(provenance["bundles"]) != n_cohort:
        raise ValueError(
            f"{results}: fit on {len(provenance['bundles'])} sessions, not the "
            f"{n_cohort} of the PFC cohort"
        )
    sessions_dir = results.parent / "sessions"
    tables = {
        Path(path).name: sha for path, sha in provenance["session_tables"].items()
    }
    records = [
        json.loads((sessions_dir / name).with_suffix(".provenance.json").read_text())
        for name in tables
    ]

    def differs(path: Path, sha: str) -> bool:
        return not path.exists() or sha256_file(path) != sha

    stale = {
        "bundle_sha256": sum(
            differs(bundle_dir / Path(path).name, sha)
            for path, sha in provenance["bundles"].items()
        ),
        "session_table_sha256": sum(
            differs(sessions_dir / name, sha) for name, sha in tables.items()
        ),
        **{
            name: sum(r["input_provenance"].get(name) != sha for r in records)
            for name, sha in current.items()
        },
    }
    return _stale_note("the stop-context fits", stale, n_cohort)


def marked_family_staleness(
    summary_path: Path, current: dict[str, str], n_cohort: int
) -> str | None:
    """Why the marked-family summary is stale, or None.

    Reads every bundle sidecar (``*_marked_family.provenance.json``) beside the
    summary, wherever it lives: each records the event and stop tables its fit
    read. A sidecar recording other tables, or a bundle written after the
    summary, makes the summary stale.

    Raises
    ------
    ValueError
        If the sidecars do not cover the ``n_cohort`` PFC epochs.
    """
    folder = summary_path.parent
    sidecars = [
        json.loads(path.read_text())
        for path in sorted(folder.glob("*_marked_family.provenance.json"))
    ]
    sessions = {(s["nwb_file_name"], s["epoch_name"]) for s in sidecars}
    if len(sessions) != n_cohort:
        raise ValueError(
            f"{folder}: marked-family bundles of {len(sessions)} sessions, not the "
            f"{n_cohort} of the PFC cohort"
        )
    written = summary_path.stat().st_mtime
    stale = {
        **{
            name: sum(s.get(name) != sha for s in sidecars)
            for name, sha in current.items()
        },
        "bundle written after the summary": sum(
            path.stat().st_mtime > written
            for path in folder.glob("*_marked_family.pkl")
        ),
    }
    return _stale_note("the marked-family bundles", stale, len(sidecars))


# --- marked family: content, strength, ripple power --------------------------------


def marked_family_values(
    summary: pd.DataFrame, make: Callable[..., Value]
) -> list[Value]:
    """How the mPFC response changes with what is replayed and how strong the event is.

    Read from the marked-family summary of the follow-up joint models (fit with
    a weak prior on the per-stop baseline columns, every unit with a spike
    kept, no rate gate). For each model, target mark and event class: the
    population slope at its peak (log-rate change of the peri-event response
    per within-session SD of the mark; unit -> session -> animal), the curve's
    mean over 0 to +150 ms, and how many animals' own 0 to +150 ms means share
    its sign. For the marks a sentence quotes (``MARKED_HELDOUT_SHARES``), per
    replay class, the share of units whose held-out deviance improves when the
    mark's kernel is in the model (animal-balanced). No shuffle reference was
    run, so the share expected by chance is unknown and the share is not a test.
    Beside them, the within-session correlation of
    ripple power and multiunit rate in each replay class: under the shared
    ridge prior a small slope on either can carry part of the other's effect,
    with the same sign in every animal (possible leakage). Full tables are in
    the summary's CSV and parquet.
    """
    values: list[Value] = []

    add = _finite_adder(values, make)

    wide = summary[summary["statistic"] != "mark_correlation"].pivot_table(
        index=["model", "event_class", "target"], columns="statistic", values="value"
    )
    for model_word, model in MARKED_MODELS.items():
        for (row_model, event_class, target), row in wide.iterrows():
            if row_model != model or target not in MARKED_TARGETS:
                continue
            stem = (
                f"marked_{model_word}_{MARKED_TARGETS[target]}_"
                f"{MARKED_CLASSES[event_class]}"
            )
            what = (
                f"{target} in model {model}, "
                f"{'all classes pooled' if event_class == 'pooled' else event_class}"
            )
            add(
                f"{stem}_peak",
                row["peak_value"],
                "quantity",
                "sig2",
                f"population slope at its peak: change of the peri-event log rate per "
                f"within-session SD of {what} (unit -> session -> animal)",
                "log rate",
            )
            add(
                f"{stem}_peak_lag_ms",
                row["peak_lag_ms"],
                "quantity",
                "int",
                f"lag of that peak, {what}",
                "ms",
            )
            add(
                f"{stem}_post_onset_mean",
                row["post_onset_mean"],
                "quantity",
                "sig2",
                f"mean of the population slope curve from 0 to +150 ms after onset, {what}",
                "log rate",
            )
            add(
                f"{stem}_post_onset_animals_same_sign",
                row["post_onset_animals_same_sign"],
                "count",
                "count",
                f"animals whose own mean slope from 0 to +150 ms after onset has the "
                f"sign of the population's, {what} (a window fixed in advance; the "
                "count at the peak lag is not reported, since the lag is chosen where "
                "the population curve is largest and the animals agree there by "
                "selection)",
            )
            if (model, target) in MARKED_HELDOUT_SHARES and (
                event_class in MARKED_HELDOUT_SHARE_CLASSES
            ):
                add(
                    f"{stem}_heldout_gain_fraction",
                    row["frac_pos"],
                    "fraction",
                    "pct",
                    f"share of units whose held-out deviance improves when the mark's "
                    f"kernel is in the model, {what} (per session, mean over sessions "
                    "within animal, mean over animals; over "
                    f"{row['n_usable']:.0f} usable of {row['n_units']:.0f} unit fits, "
                    f"{row['n_excluded_blowup']:.0f} left out as blow-ups, in "
                    f"{row['n_sessions']:.0f} sessions of {row['n_animals']:.0f} "
                    "animals). No chance reference was computed, so the share expected "
                    "by chance is unknown: this is not a test",
                )
    correlations = summary[
        (summary["statistic"] == "mark_correlation")
        & (summary["model"] == "event_strength")
        & (summary["target"] == "ripple_power_z & mua_rate_z")
    ]
    for _, row in correlations.iterrows():
        add(
            f"marked_ripple_mua_correlation_{MARKED_CLASSES[row['event_class']]}",
            row["value"],
            "quantity",
            "dec2",
            "within-session correlation of ripple power and hippocampal multiunit rate "
            f"over {row['event_class']} events (mean over animals of session means): "
            "their slopes may carry each other's effect (ridge leakage)",
        )
    animals = wide["n_animals"].max() if "n_animals" in wide else np.nan
    add(
        "marked_animals",
        animals,
        "count",
        "count",
        "animals in the marked-family summary",
    )
    return values


# --- stop context ---------------------------------------------------------------------


def stop_context_values(
    results: pd.DataFrame, make: Callable[..., Value]
) -> list[Value]:
    """Does the response to replay differ with the stop's reward outcome or a coming
    patch switch?

    Read from the stop-level context fits (``analysis/pfc/stop_context.py``): a
    Poisson model, pooled over sessions, of each unit's spike count in the
    ``[0, +200) ms`` window after each replay onset against its rate in the
    same stop's event-free fitted time (the stop baseline estimated
    conditionally), with free unit terms, the stop's known reward outcome, the
    binary next-trip-switches-patch mark and the count of overlapping events;
    fit for SWR-coupled and non-SWR replay, for units whose canonical peak is an
    increase or a decrease. Main numbers: units called for the class
    (conditional estimator); all units are a check (reward, increases only).
    ``exp(beta)`` is the multiplicative change of the post-onset response
    relative to the stop baseline at rewarded stops (or before a patch-leaving
    trip); intervals are 95 % intervals clustered by recording day (mPFC units
    recur in every epoch of a day; CR1 factor, t with ``G - 1`` degrees of
    freedom); animals counted with the pooled sign out of those estimated;
    leave-one-animal-out range, which needs every leave-out to have an estimate.
    """
    values: list[Value] = []

    add = _finite_adder(values, make)

    rows = results[results["estimator"] == "conditional"]
    classes = {"non_local_SWR": "swr_replay", "non_local_no_SWR": "non_swr_replay"}
    terms = {"rewarded": "reward", "next_trip_switches_patch": "switch"}
    for (event_class, sign, unit_set, term), row in rows.set_index(
        ["event_class", "sign", "unit_set", "term"]
    ).iterrows():
        if event_class not in classes or term not in terms:
            continue
        if unit_set == "all" and not (term == "rewarded" and sign == "increase"):
            continue
        stem = f"stop_context_{terms[term]}_gain_{classes[event_class]}_{sign}"
        if unit_set == "all":
            stem += "_all_units"
        what = (
            f"{'rewarded stops' if term == 'rewarded' else 'stops before a patch-leaving trip'}"
            f", {event_class}, units whose peak is {'an increase' if sign == 'increase' else 'a decrease'}"
            f", {'called' if unit_set == 'called' else 'all'} units"
        )
        add(
            stem,
            row["exp_coef"],
            "quantity",
            "dec2",
            f"multiplicative change of the post-onset response relative to the stop "
            f"baseline at {what}",
            "x",
        )
        add(
            f"{stem}_ci_low",
            row["ci_low"],
            "quantity",
            "dec2",
            f"lower end of the recording-day-clustered 95 % interval of that (t with "
            f"G - 1 degrees of freedom), {what}",
            "x",
        )
        add(
            f"{stem}_ci_high",
            row["ci_high"],
            "quantity",
            "dec2",
            f"upper end of that interval, {what}",
            "x",
        )
        if unit_set == "called" and sign == "increase":
            if row["n_loao_estimated"] != row["n_animals"]:
                raise ValueError(
                    f"{stem}: {row['n_loao_estimated']:.0f} of {row['n_animals']:.0f} "
                    "leave-one-animal-out fits have an estimate"
                )
            add(
                f"{stem}_animals_pooled_sign",
                row["n_animals_pooled_sign"],
                "count",
                "count",
                f"animals whose own estimate has the pooled sign, {what}",
            )
            add(
                f"{stem}_animals_estimated",
                row["n_animals_estimated"],
                "count",
                "count",
                f"animals with an estimate, {what}",
            )
            add(
                f"{stem}_loao_min",
                row["loao_exp_coef_min"],
                "quantity",
                "dec2",
                f"smallest leave-one-animal-out estimate, {what}",
                "x",
            )
            add(
                f"{stem}_loao_max",
                row["loao_exp_coef_max"],
                "quantity",
                "dec2",
                f"largest leave-one-animal-out estimate, {what}",
                "x",
            )
    return values


def followup_values(
    inputs: MpfcInputs,
    current: dict[str, str],
    n_cohort: int,
) -> tuple[list[Value], list[dict[str, Any]]]:
    """The marked-family and stop-context values, each pending when its inputs
    are stale (:func:`marked_family_staleness`, :func:`stop_context_staleness`),
    and the records of the files read. A missing output gives no values."""
    values: list[Value] = []
    records: list[dict[str, Any]] = []
    if inputs.marked_family_summary.exists():
        values += marked_family_values(
            pd.read_parquet(inputs.marked_family_summary),
            value_factory(
                marked_family_values,
                f"{rel(inputs.marked_family_summary)} "
                "(scripts/pfc/summarize_pfc_marked_family.py over "
                "scripts/pfc/run_pfc_marked_model_family.py --model-set followup "
                "--baseline-penalty-sigma 3 --slope-permutations 0 bundles)",
                TAGS["modulation"],
                marked_family_staleness(
                    inputs.marked_family_summary, current, n_cohort
                ),
            ),
        )
        records.append(file_record(inputs.marked_family_summary, ROOT_DIR))
    if inputs.stop_context_results.exists():
        results = pd.read_csv(inputs.stop_context_results)
        if not results["n_sessions"].eq(n_cohort).all():
            raise ValueError(
                f"{inputs.stop_context_results}: a fit covers "
                f"{sorted(results['n_sessions'].unique())} sessions, not the "
                f"{n_cohort} of the PFC cohort"
            )
        values += stop_context_values(
            results,
            value_factory(
                stop_context_values,
                f"{rel(inputs.stop_context_results)} "
                "(scripts/pfc/run_pfc_stop_context.py fit; conditional estimator)",
                TAGS["modulation"],
                stop_context_staleness(
                    inputs.stop_context_results, inputs.bundle_dir, current, n_cohort
                ),
            ),
        )
        records.append(file_record(inputs.stop_context_results, ROOT_DIR))
    return values, records
