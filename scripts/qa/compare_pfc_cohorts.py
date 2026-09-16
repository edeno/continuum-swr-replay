#!/usr/bin/env python
"""Write the before/after audit of the mPFC unit-GLM stop cohort.

The mPFC GLM used to fit every immobile 2 ms bin in the epoch and to key its
design matrices on every speed-eligible event. It now fits the bins inside
well-anchored stops and the events whose onset lies inside one, through the
same selector the hippocampal rate and content ladders read
(``analysis.replay.cohort_selection``). This script states what that did, from
the artifacts themselves:

- the per-session bin and event counts, before and after, read from each new
  bundle's provenance sidecar (which records both sides of the restriction);
- the decode/position alignment trim each session needed, from the same
  sidecars;
- the leading-edge coverage diagnostic (dropped events whose regressor basis
  would still have covered fitted bins);
- the population that reaches Figure 4, loaded from both bundle trees with
  ``load_population_data`` -- neurons passing the firing-rate gate and the
  FDR-significant fraction per test, overall and per animal.

Usage::

    uv run python scripts/qa/compare_pfc_cohorts.py \\
        --out docs/audits/2026-09-14-pfc-atwell-cohort.md

When the earlier tree also carries provenance sidecars, three further sections
are written: whether the two trees describe the same events, the unit-by-unit
before/after differences on everything both carry (against the tolerances the
batched estimator was accepted at), what the matched non-event control says,
and which estimator and device fit each session::

    uv run python scripts/qa/compare_pfc_cohorts.py \\
        --before Processed-Data/_pre_canonical_events_backup_20260914 \\
        --after Processed-Data \\
        --out docs/audits/2026-09-14-pfc-canonical-rerun.md

When the after tree's sidecars name the penalised estimator, a different
report is written instead: what the prior did to each unit's peak
modulation, how many units carried a basis column with no spikes under it,
what the two inferences call, how steady the held-out difference is between
folds, and the sensitivity of the curves to the prior::

    uv run python scripts/qa/compare_pfc_cohorts.py \\
        --before Processed-Data/_pre_ridge_backup \\
        --after Processed-Data \\
        --out docs/audits/<date>-pfc-ridge-heldout.md
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import false_discovery_control

from continuum_swr_replay.analysis.pfc.load_population import (
    load_population_data,
    significant_fraction,
)
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

REPO = Path(__file__).resolve().parents[2]

#: The firing-rate column the shipped gate screens, and the unrestricted twin
#: the pre-cohort bundles carried under the same name.
FIT_RATE_COLUMN = "immobility_avg_firing_rate"
ALL_IMMOBILE_RATE_COLUMN = "immobility_avg_firing_rate_all_immobile"

#: Tests ``load_population_data`` FDR-corrects. Reported here in full: the four
#: event classes the figure draws are covered by the ``vs. constant`` rows, and
#: the model-comparison rows are what panel f reads.
FDR_TESTS = [
    "SWR vs. constant",
    "non_local vs. constant",
    "non_local_no_SWR vs. constant",
    "SWR + non_local_no_SWR vs. SWR",
    "SWR + non_local_no_SWR vs. non_local_no_SWR",
    "non_local_SWR + non_local_no_SWR vs. non_local",
]


def _invocation() -> str:
    """This run's command line, relative to the repo root and shell-quoted."""
    script = Path(sys.argv[0])
    try:
        script = script.resolve().relative_to(REPO)
    except ValueError:
        pass
    return shlex.join(["uv", "run", "python", str(script), *sys.argv[1:]])


#: Top-level sidecar keys carried into the per-session frame beside the
#: ``cohort_counts`` block: which estimator and device fit the session, and
#: which event inputs it was fit on. A sidecar written before one of these
#: existed simply yields ``None``, which is what the rerun precondition reads.
SIDECAR_TOP_LEVEL_KEYS = [
    "estimator",
    "inference",
    "penalty_sigma_log_rate",
    "n_folds",
    "fold_seed",
    "n_control_draws",
    "control_draw_seed",
    "n_basis_per_class",
    "fitter",
    "fitter_tol",
    "jax_backend",
    "jax_version",
    "gpu_index",
    "gpu_uuid",
    "gpu_name",
    "n_model_fits",
    "n_nonconverged_model_fits",
    "n_neurons_with_nonconverged_model",
    "canonical_event_table_sha256",
    "canonical_event_table_path",
    "data_over_time_sha256",
    "is_swr_definition",
    "n_events_total",
    "n_events_swr",
    "n_events_non_swr",
    "n_matched_control",
    "n_matched_control_unplaced",
    "n_matched_control_onset_outside_fitted_bout",
    "n_matched_control_bout_saturated",
]


#: Suffix ``fit_provenance`` appends when the working tree carried uncommitted
#: changes. It is part of the provenance, so it survives truncation.
DIRTY_SUFFIX = "-dirty"


def _short_hash(git_hash: str) -> str:
    """Abbreviate a recorded hash, keeping any ``-dirty`` marker.

    Truncating to seven characters would turn a dirty tree into a clean commit,
    which is the one thing the marker exists to prevent.
    """
    if git_hash.endswith(DIRTY_SUFFIX):
        return git_hash[: -len(DIRTY_SUFFIX)][:7] + DIRTY_SUFFIX
    return git_hash[:7]


def _commit_subject(git_hash: str) -> str:
    """One-line subject of ``git_hash``, or an empty string if unavailable.

    A ``-dirty`` hash names a real commit plus uncommitted changes; strip the
    marker so the commit is still found.
    """
    git_hash = git_hash.removesuffix(DIRTY_SUFFIX)
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "show", "-s", "--format=%s", git_hash],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return ""
    return out.stdout.strip()


def read_sidecars(data_dir: Path, artifact_suffix: str = "") -> pd.DataFrame:
    """One row per mPFC bundle, from its provenance sidecar.

    Parameters
    ----------
    data_dir : Path
        Directory holding ``*_mPFC_modulation{suffix}.provenance.json``.
    artifact_suffix : str
        Cohort suffix on the bundle names ("" for the canonical at-well one).

    Returns
    -------
    pandas.DataFrame
        Columns: ``session``, ``animal``, ``stop_cohort``,
        ``stop_table_sha256``, ``git_hash``, ``written_at``, the
        ``cohort_counts`` fields, and the ``alignment`` fields flattened with
        an ``align_`` prefix (NaN for sidecars written before the alignment
        step existed).
    """
    stem = f"_mPFC_modulation{artifact_suffix}"
    paths = sorted(data_dir.glob(f"*{stem}.provenance.json"))
    rows = []
    for path in paths:
        sidecar = json.loads(path.read_text())
        counts = dict(sidecar.get("cohort_counts", {}))
        alignment = counts.pop("alignment", None) or {}
        row = {
            "session": path.name.replace(f"{stem}.provenance.json", ""),
            "nwb_file_name": sidecar.get("nwb_file_name", ""),
            "stop_cohort": sidecar.get("stop_cohort"),
            "stop_table_sha256": sidecar.get("stop_table_sha256"),
            "git_hash": _short_hash(sidecar.get("git_hash", "")),
            "written_at": sidecar.get("written_at", "")[:19],
            **counts,
            **{f"align_{k}": v for k, v in alignment.items()},
            # Top-level keys the estimator and event-definition tables read.
            # Absent from sidecars written before those fields existed, which
            # is itself the signal the rerun audit's precondition tests.
            **{key: sidecar.get(key) for key in SIDECAR_TOP_LEVEL_KEYS},
        }
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No *{stem}.provenance.json sidecars in {data_dir}")
    frame = pd.DataFrame(rows).sort_values("session").reset_index(drop=True)
    # Every bundle must have a sidecar. A bundle without one is invisible to the
    # bins/events tables while ``load_population_data`` still counts its
    # neurons, so the two halves of the report would describe different
    # session sets without saying so.
    described = set(frame["session"])
    present = bundle_sessions(data_dir, artifact_suffix)
    missing = sorted(present - described)
    orphaned = sorted(described - present)
    if missing or orphaned:
        raise ValueError(
            f"bundle/sidecar mismatch in {data_dir}: "
            f"{len(missing)} bundle(s) with no provenance sidecar "
            f"({', '.join(missing) or 'none'}) and {len(orphaned)} sidecar(s) "
            f"with no bundle ({', '.join(orphaned) or 'none'}). Refit the "
            "session(s) so the runner writes a sidecar, or point --after-dir at "
            "a complete tree; a bundle with no sidecar would be counted in the "
            "population summary but absent from the per-session tables."
        )
    # Through the shared parser, so ``j16`` sessions resolve to "j16" and not
    # to "j1" (the leading alphabetic run) or "j" (a split on the first digit).
    frame.insert(
        1,
        "animal",
        frame["nwb_file_name"].map(lambda name: parse_nwb_filename(name)[0]),
    )
    return frame


#: Budget (seconds) the single-budget form of ``align_results_to_position``
#: allowed on either axis. A session trimmed by more than this could only have
#: been fit by the later per-axis form, so the trim size identifies the code.
SINGLE_BUDGET_S = 0.1

#: How a sidecar's own fields identify the runner code the worker had imported.
#: The recorded ``git_hash`` is the repository HEAD when the sidecar was
#: WRITTEN, which drifts ahead of the running process; these two facts do not.
RUNNER_CODE_PRE_ALIGNMENT = "50bf4b7 (no alignment step)"
RUNNER_CODE_SINGLE_BUDGET = f"7362347 (alignment, {SINGLE_BUDGET_S} s budget)"
RUNNER_CODE_PER_AXIS = "6c162ce (alignment, per-axis budgets)"


def runner_code(sidecars: pd.DataFrame) -> pd.Series:
    """Which runner code fit each session, from the sidecar's own fields.

    Three regimes, separated by two observable facts rather than by the
    recorded hash:

    - no ``alignment`` block at all: the code predates the alignment step;
    - an alignment block with a trim inside the single budget: the step exists,
      and either budget form would have accepted it;
    - a trim larger than that budget: only the per-axis form would have
      accepted it, so that is the code that ran.
    """
    bins = sidecars.get("align_n_results_bins_dropped")
    if bins is None:
        return pd.Series(RUNNER_CODE_PRE_ALIGNMENT, index=sidecars.index)
    zeros = pd.Series(0.0, index=sidecars.index)
    seconds = sidecars.get("align_dropped_seconds_results", zeros).fillna(0)
    code = pd.Series(RUNNER_CODE_SINGLE_BUDGET, index=sidecars.index)
    code[bins.isna()] = RUNNER_CODE_PRE_ALIGNMENT
    code[bins.notna() & (seconds > SINGLE_BUDGET_S)] = RUNNER_CODE_PER_AXIS
    return code


def provenance_table(sidecars: pd.DataFrame) -> pd.DataFrame:
    """Sessions grouped by the code that fit them and the hash they recorded."""
    frame = sidecars.copy()
    frame["runner_code"] = runner_code(frame)
    dropped = frame.get("align_n_results_bins_dropped")
    frame["alignment_block"] = (
        "no" if dropped is None else np.where(dropped.notna(), "yes", "no")
    )
    grouped = (
        frame.groupby(["runner_code", "alignment_block", "git_hash"])
        .agg(
            n_sessions=("session", "size"),
            first_written=("written_at", "min"),
            last_written=("written_at", "max"),
        )
        .reset_index()
        .sort_values("first_written")
    )
    grouped["recorded_hash_subject"] = grouped["git_hash"].map(_commit_subject)
    return grouped


def bundle_sessions(data_dir: Path, artifact_suffix: str = "") -> set[str]:
    """Session keys of the mPFC bundles in ``data_dir``.

    Read from the pickle names rather than the sidecars, so a tree that
    predates provenance sidecars still reports its sessions.
    """
    stem = f"_mPFC_modulation{artifact_suffix}.pkl"
    return {path.name.removesuffix(stem) for path in data_dir.glob(f"*{stem}")}


def session_cohort_table(sidecars: pd.DataFrame) -> pd.DataFrame:
    """Per-session bins and events before vs after the cohort restriction.

    ``n_bins_immobile`` is the "before" bin count -- every immobile bin was fit
    before -- and ``n_bins_fit`` the "after"; the event columns carry their own
    before/after pair.
    """
    table = pd.DataFrame(
        {
            "session": sidecars["session"],
            "bins_immobile_before": sidecars["n_bins_immobile"],
            "bins_fit_after": sidecars["n_bins_fit"],
            "bins_kept_pct": 100.0
            * sidecars["n_bins_fit"]
            / sidecars["n_bins_immobile"],
            "nonlocal_before": sidecars["n_nonlocal_before"],
            "nonlocal_after": sidecars["n_nonlocal_after"],
            "nonlocal_kept_pct": 100.0
            * sidecars["n_nonlocal_after"]
            / sidecars["n_nonlocal_before"],
            "ripple_before": sidecars["n_ripple_before"],
            "ripple_after": sidecars["n_ripple_after"],
            "nonlocal_swr_after": sidecars["n_nonlocal_swr"],
            "nonlocal_no_swr_after": sidecars["n_nonlocal_no_swr"],
            "align_results_bins_dropped": sidecars.get(
                "align_n_results_bins_dropped", np.nan
            ),
            "align_results_s_dropped": sidecars.get(
                "align_dropped_seconds_results", np.nan
            ),
        }
    )
    total = {
        "session": f"ALL ({len(table)} sessions)",
        "bins_immobile_before": table["bins_immobile_before"].sum(),
        "bins_fit_after": table["bins_fit_after"].sum(),
        "bins_kept_pct": 100.0
        * table["bins_fit_after"].sum()
        / table["bins_immobile_before"].sum(),
        "nonlocal_before": table["nonlocal_before"].sum(),
        "nonlocal_after": table["nonlocal_after"].sum(),
        "nonlocal_kept_pct": 100.0
        * table["nonlocal_after"].sum()
        / table["nonlocal_before"].sum(),
        "ripple_before": table["ripple_before"].sum(),
        "ripple_after": table["ripple_after"].sum(),
        "nonlocal_swr_after": table["nonlocal_swr_after"].sum(),
        "nonlocal_no_swr_after": table["nonlocal_no_swr_after"].sum(),
        "align_results_bins_dropped": table["align_results_bins_dropped"].sum(),
        "align_results_s_dropped": table["align_results_s_dropped"].sum(),
    }
    table = pd.concat([table, pd.DataFrame([total])], ignore_index=True)
    # Keep the dropped-bin count an integer (nullable, because sidecars written
    # before the alignment step carry no value) so it does not render as a
    # float beside the true floats.
    table["align_results_bins_dropped"] = table["align_results_bins_dropped"].astype(
        "Int64"
    )
    return table


def coverage_summary(sidecars: pd.DataFrame) -> tuple[dict[str, int], pd.DataFrame]:
    """The leading-edge coverage diagnostic, totalled and per affected session.

    Returns
    -------
    totals : dict
        Summed dropped-but-covering counts for both event classes.
    affected : pandas.DataFrame
        The sessions with a non-zero count, named so the diagnostic is not a
        bare total.
    """
    nl = sidecars["n_nonlocal_dropped_overlapping_fit"].fillna(0).astype(int)
    rip = sidecars["n_ripple_dropped_overlapping_fit"].fillna(0).astype(int)
    totals = {
        "n_nonlocal_dropped_overlapping_fit": int(nl.sum()),
        "n_ripple_dropped_overlapping_fit": int(rip.sum()),
        "n_sessions_nonlocal": int((nl > 0).sum()),
        "n_sessions_ripple": int((rip > 0).sum()),
    }
    affected = sidecars.loc[
        (nl > 0) | (rip > 0),
        [
            "session",
            "n_nonlocal_dropped_overlapping_fit",
            "n_nonlocal_after",
            "n_nonlocal_dropped_overlapping_fit_frac_of_kept",
            "n_ripple_dropped_overlapping_fit",
            "n_ripple_after",
        ],
    ].reset_index(drop=True)
    return totals, affected


def alignment_prose(sidecars: pd.DataFrame) -> str:
    """What the decode/position trim did, across the corpus.

    The PFC path is the only consumer that reloads ``load_data`` beside a saved
    decode, so it is the only one that meets the length mismatch between a
    decode written before ``get_position_info`` was trimmed and the position
    frame returned today. Sidecars written before that step existed carry no
    ``alignment`` block at all, which is distinct from a recorded no-op.
    """
    bins = sidecars.get("align_n_results_bins_dropped")
    if bins is None:
        return (
            "No sidecar in this tree records an alignment block, so nothing "
            "can be said here about the decode/position trim.\n"
        )
    zeros = pd.Series(0.0, index=sidecars.index)
    seconds = sidecars.get("align_dropped_seconds_results", zeros).fillna(0)
    position_rows = sidecars.get("align_n_position_rows_dropped", zeros).fillna(0)
    recorded = bins.notna()
    trimmed = bins.fillna(0) > 0
    n_recorded, n_trimmed = int(recorded.sum()), int(trimmed.sum())
    if n_trimmed == n_recorded:
        how_many = f"and every one of those {n_trimmed} session(s) needed a trim"
    else:
        how_many = (
            f"and {n_trimmed} of those needed a trim (the other "
            f"{n_recorded - n_trimmed} record the step as a no-op)"
        )
    largest = ""
    if n_trimmed:
        worst = sidecars.loc[seconds.idxmax()]
        largest = (
            f" The largest single trim is {seconds.max():.1f} s in "
            f"`{worst['session']}`."
        )
    return (
        f"{n_recorded} of {len(sidecars)} sidecars record the "
        f"decode/position alignment step, {how_many}: "
        f"{int(bins.fillna(0).sum()):,} decode bins "
        f"({seconds.sum():.1f} s) dropped in total, against "
        f"{int(position_rows.sum()):,} position rows -- the decode always "
        f"outlives the position frame, never the reverse.{largest} The remaining "
        f"{int((~recorded).sum())} session(s) were fit before the step existed, "
        "by code whose strict 1-to-1 validator refused any mismatch at all, so "
        "their axes already agreed exactly and the step would be the identity "
        "on them.\n"
    )


def gate_counts(
    data_dir: Path,
    min_firing_rate: float,
    suffix: str = "",
    inference: str = "legacy",
) -> dict:
    """How many neurons survive the loader, and the firing-rate gate.

    The gate is applied to whichever rate column the bundle carries under
    ``immobility_avg_firing_rate``. For a cohort-restricted bundle that is the
    mean rate over the bins fit; the pre-cohort bundles carry the same name for
    the rate over every immobile bin. Both are counted here when the bundle has
    both, so the two sides can be compared on one quantity as well as on the
    shipped one.
    """
    frame, _, _ = load_population_data(
        data_dir,
        min_firing_rate=0.0,
        artifact_suffix=suffix,
        inference=inference,
    )
    counts = {
        "n_neurons_with_fit": int(len(frame)),
        "n_passing_fit_bin_rate": int(
            (frame[FIT_RATE_COLUMN] >= min_firing_rate).sum()
        ),
    }
    if ALL_IMMOBILE_RATE_COLUMN in frame.columns:
        counts["n_passing_all_immobile_rate"] = int(
            (frame[ALL_IMMOBILE_RATE_COLUMN] >= min_firing_rate).sum()
        )
    else:
        counts["n_passing_all_immobile_rate"] = None
    return counts


def significance_table(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """FDR-significant fraction per test, overall and per animal."""
    rows = []
    groups = [("all", frame)] + [
        (animal, sub) for animal, sub in frame.groupby("animal", sort=True)
    ]
    for animal, sub in groups:
        for test in FDR_TESTS:
            column = f"{test}: fdr_q"
            if column not in sub.columns:
                rows.append(
                    {
                        "animal": animal,
                        "test": test,
                        "n_tested": 0,
                        "n_sig": 0,
                        "frac_sig": np.nan,
                    }
                )
                continue
            q = sub[column].to_numpy(dtype=float)
            tested = np.isfinite(q)
            n_tested = int(tested.sum())
            n_sig = int((q[tested] < alpha).sum())
            rows.append(
                {
                    "animal": animal,
                    "test": test,
                    "n_tested": n_tested,
                    "n_sig": n_sig,
                    "frac_sig": n_sig / n_tested if n_tested else np.nan,
                }
            )
    return pd.DataFrame(rows)


def population_comparison(
    before: pd.DataFrame, after: pd.DataFrame, alpha: float
) -> pd.DataFrame:
    """Join the before/after significance tables on animal and test."""
    return significance_table(before, alpha).merge(
        significance_table(after, alpha),
        on=["animal", "test"],
        suffixes=("_before", "_after"),
        how="outer",
    )


def _fdr_movement_prose(comparison: pd.DataFrame) -> str:
    """How far the pooled significant fractions moved, and which one fell.

    The interesting quantity is the size of the move, not its direction on any
    one test: a cohort that changed which neurons are significant would show it
    here.
    """
    pooled = comparison.loc[comparison["animal"] == "all"].copy()
    if pooled.empty:
        return ""
    pooled["delta_pp"] = 100.0 * (pooled["frac_sig_after"] - pooled["frac_sig_before"])
    biggest = pooled.loc[pooled["delta_pp"].abs().idxmax()]
    fell = pooled.loc[pooled["delta_pp"] < 0]
    if fell.empty:
        which_fell = "No test's fraction fell."
    else:
        named = "; ".join(
            f"`{row.test}` ({100 * row.frac_sig_before:.1f}% to "
            f"{100 * row.frac_sig_after:.1f}%)"
            for row in fell.itertuples()
        )
        which_fell = (
            f"{len(fell)} of {len(pooled)} fell: {named}"
            if len(fell) > 1
            else f"Exactly one fell: {named}."
        )
    return (
        "**In the pooled population the significant fractions barely move**: "
        f"the largest change across the {len(pooled)} tests is "
        f"{biggest['delta_pp']:+.1f} percentage points (`{biggest['test']}`), "
        f"and every test moves by at most {pooled['delta_pp'].abs().max():.1f} "
        f"points. {which_fell}\n"
    )


# ---------------------------------------------------------------------------
# The canonical-events rerun audit: before/after on the same events
# ---------------------------------------------------------------------------

#: Tolerances the batched fitter was accepted at against statsmodels on the
#: three gate sessions. Reused here because the rerun is the same swap on the
#: whole corpus: when the two sides describe the same events on the same
#: cohort, every statistic a pre-existing model produces must land inside them.
RERUN_PARAM_TOL = 1e-5
RERUN_PVALUE_TOL = 1e-6

#: The two LR tests the matched non-event control adds. Not in ``FDR_TESTS``:
#: ``load_population_data`` does not FDR-correct them, so this script does.
MATCHED_CONTROL_TESTS = [
    "matched_control + non_local_no_SWR vs. matched_control",
    "matched_control vs. constant",
]

#: The modulation array the control adds, and the four both sides carry.
MATCHED_CONTROL_ARRAY = "matched_control_modulation"
SHARED_MODULATION_ARRAYS = [
    "SWR_modulation",
    "no_SWR_modulation",
    "non_local_SWR_modulation",
    "non_local_no_SWR_modulation",
]


def _numeric(values: pd.Series) -> pd.Series:
    """A sidecar column as floats with missing entries at zero.

    Columns assembled from sidecars of different vintages come back as object
    dtype when some rows filled the key and others carry ``None``; summing
    those directly either raises or silently downcasts.
    """
    return pd.to_numeric(values, errors="coerce").fillna(0.0)


def try_read_sidecars(data_dir: Path, artifact_suffix: str = "") -> pd.DataFrame | None:
    """``read_sidecars``, or ``None`` when the tree carries no sidecars at all.

    The before side of the cohort audit predates provenance sidecars entirely;
    the before side of the rerun audit has them. Distinguishing the two is what
    decides whether the rerun sections can be written.
    """
    try:
        return read_sidecars(data_dir, artifact_suffix)
    except FileNotFoundError:
        return None


def same_events_precondition(
    before: pd.DataFrame | None, after: pd.DataFrame
) -> tuple[bool, str]:
    """Whether the two trees were fit on the same events, and what says so.

    The unit-by-unit tolerance check below is only a check of the *fitter* when
    nothing else about the fit changed. Three facts have to hold across every
    session both trees hold:

    - the before side has provenance sidecars at all (otherwise nothing can be
      said about what it was fit on);
    - both sides name the same stop table, so the cohort is the same;
    - the before side records no ``canonical_event_table_sha256``, i.e. it is
      the tree from before the canonical events were read, and the after side
      does record one.

    The third is a *marker*, not a proof: a tree with no recorded event hash
    cannot be shown to describe the same events as one that has it. The
    per-session event counts are compared alongside, and it is those that
    decide whether the comparison reads as a fitter check -- see
    :func:`event_count_agreement`.
    """
    if before is None:
        return False, (
            "the before tree carries no provenance sidecars, so what it was "
            "fit on cannot be read from it"
        )
    if (
        "canonical_event_table_sha256" not in after.columns
        or not after["canonical_event_table_sha256"].notna().any()
    ):
        return False, (
            "the after tree records no `canonical_event_table_sha256`, so it "
            "is not a canonical-events tree"
        )
    shared = sorted(set(before["session"]) & set(after["session"]))
    if not shared:
        return False, "the two trees hold no session in common"
    before_rows = before.set_index("session").loc[shared]
    after_rows = after.set_index("session").loc[shared]
    mismatched = [
        session
        for session in shared
        if before_rows.at[session, "stop_table_sha256"]
        != after_rows.at[session, "stop_table_sha256"]
    ]
    if mismatched:
        return False, (
            f"{len(mismatched)} shared session(s) name a different stop table "
            f"on the two sides ({', '.join(mismatched[:5])}"
            f"{', ...' if len(mismatched) > 5 else ''}), so the cohort is not "
            "the same"
        )
    if "canonical_event_table_sha256" in before_rows.columns and (
        before_rows["canonical_event_table_sha256"].notna().any()
    ):
        return False, (
            "the before tree already records a `canonical_event_table_sha256`, "
            "so it is not the pre-canonical tree this comparison is for"
        )
    return True, (
        f"{len(shared)} shared session(s), same stop table on both sides, and "
        "no recorded event-table hash on the before side"
    )


def event_count_agreement(
    before: pd.DataFrame, after: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Per-session event counts on the two sides, and how many disagree.

    The coefficient comparison is a check of the fitter only where these agree.
    Where they do not, the two sides were fit on different regressors and any
    difference in the estimates is the event definition, not the solver.
    """
    columns = ["n_nonlocal_after", "n_nonlocal_swr", "n_nonlocal_no_swr"]
    shared = sorted(set(before["session"]) & set(after["session"]))
    table = (
        before.set_index("session")
        .loc[shared, columns]
        .join(
            after.set_index("session").loc[shared, columns],
            lsuffix="_before",
            rsuffix="_after",
        )
        .reset_index()
    )
    disagree = np.zeros(len(table), dtype=bool)
    for column in columns:
        disagree |= (
            table[f"{column}_before"].to_numpy() != table[f"{column}_after"].to_numpy()
        )
    table.insert(1, "same_events", ~disagree)
    return table, int(disagree.sum())


def _bundle_statistics(data_dir: Path, session: str, suffix: str) -> pd.DataFrame:
    """One session's statistics table, unfiltered."""
    return pd.read_pickle(data_dir / f"{session}_mPFC_modulation{suffix}.pkl")


def _bundle_modulation(data_dir: Path, session: str, suffix: str, name: str) -> NDArray:
    """One session's modulation array, shape ``(n_lag_times, n_neurons)``."""
    return np.load(data_dir / f"{session}_mPFC_modulation{suffix}.{name}.npy")


def _max_abs_difference(before: NDArray, after: NDArray) -> tuple[float, int]:
    """Largest finite ``|after - before|``, and how many entries are unpaired.

    A NaN on exactly one side is a change the maximum cannot see -- a model
    that became estimable, or stopped being -- so it is counted separately
    rather than skipped.
    """
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    both = np.isfinite(before) & np.isfinite(after)
    neither = ~np.isfinite(before) & ~np.isfinite(after)
    unpaired = int((~both & ~neither).sum())
    if not both.any():
        return float("nan"), unpaired
    return float(np.abs(after[both] - before[both]).max()), unpaired


def shared_statistic_differences(
    before_dir: Path,
    after_dir: Path,
    sessions: list[str],
    *,
    artifact_suffix: str = "",
    param_tol: float = RERUN_PARAM_TOL,
    pvalue_tol: float = RERUN_PVALUE_TOL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unit-by-unit before/after differences on everything both trees carry.

    Neurons are paired by their row position within a session, which is the
    order ``run_session`` writes them in; a session whose two bundles hold
    different numbers of neurons cannot be paired at all and is reported as
    such instead of being aligned by truncation.

    Returns
    -------
    per_quantity : pandas.DataFrame
        One row per compared quantity family (the LR statistics, the LR
        p-values, the fractional-deviance effect sizes, the peak modulations
        and their times, the theta columns, the firing rates, and each shared
        modulation time course), with the largest absolute difference over
        every neuron and session, the tolerance it is read against, the number
        of neuron x column cells outside it, and the cells that are finite on
        one side only.
    per_session : pandas.DataFrame
        One row per paired session: its neuron count, the cells outside
        tolerance, the cells finite on one side only, and how many neurons
        have a best-AIC model on one side and none on the other -- a neuron
        that lost (or gained) its whole row. A corpus total hides which
        sessions carry it, and "expected zero" is a per-session claim.
    unpairable : pandas.DataFrame
        Sessions whose neuron counts differ between the trees.
    """
    families = {
        "LR statistic (`: lr`)": (lambda c: c.endswith(": lr"), param_tol),
        "LR p-value (`: pval`)": (lambda c: c.endswith(": pval"), pvalue_tol),
        "fractional deviance": (
            lambda c: c.endswith(": frac_null_dev")
            or c.endswith(": frac_explained_dev"),
            param_tol,
        ),
        "peak modulation": (lambda c: c.endswith("_max_mod"), param_tol),
        "peak modulation time": (lambda c: c.endswith("_max_mod_time"), param_tol),
        "theta": (lambda c: c.startswith("theta"), param_tol),
        "firing rate": (lambda c: c.endswith("_avg_firing_rate"), param_tol),
    }
    worst = dict.fromkeys(families, 0.0)
    outside = dict.fromkeys(families, 0)
    unpaired = dict.fromkeys(families, 0)
    n_cells = dict.fromkeys(families, 0)
    array_worst = dict.fromkeys(SHARED_MODULATION_ARRAYS, 0.0)
    array_outside = dict.fromkeys(SHARED_MODULATION_ARRAYS, 0)
    array_unpaired = dict.fromkeys(SHARED_MODULATION_ARRAYS, 0)
    array_cells = dict.fromkeys(SHARED_MODULATION_ARRAYS, 0)
    model_disagreements = 0
    model_unpaired = 0
    n_models_compared = 0
    n_neurons = 0
    unpairable_rows = []
    per_session_rows = []

    for session in sessions:
        session_outside = 0
        session_unpaired = 0
        session_model_unpaired = 0
        before = _bundle_statistics(before_dir, session, "")
        after = _bundle_statistics(after_dir, session, artifact_suffix)
        if len(before) != len(after):
            unpairable_rows.append(
                {
                    "session": session,
                    "n_neurons_before": len(before),
                    "n_neurons_after": len(after),
                }
            )
            continue
        n_neurons += len(before)
        shared_columns = [c for c in before.columns if c in after.columns]
        for family, (matches, tol) in families.items():
            columns = [c for c in shared_columns if matches(c)]
            for column in columns:
                if before[column].dtype == object:
                    continue
                difference, n_unpaired = _max_abs_difference(
                    before[column].to_numpy(), after[column].to_numpy()
                )
                n_cells[family] += len(before)
                unpaired[family] += n_unpaired
                session_unpaired += n_unpaired
                if np.isfinite(difference):
                    worst[family] = max(worst[family], difference)
                    n_column_outside = int(
                        (
                            np.abs(
                                after[column].to_numpy(dtype=float)
                                - before[column].to_numpy(dtype=float)
                            )
                            > tol
                        ).sum()
                    )
                    outside[family] += n_column_outside
                    session_outside += n_column_outside
        # The selected model is a label, so it is compared for identity --
        # and a neuron with no fit at all carries no label on either side,
        # which is agreement, not a disagreement between two NaNs.
        for column in ("min_aic_model", "min_aic_model_replay_split"):
            if column in shared_columns:
                n_models_compared += len(before)
                named_before = before[column].notna().to_numpy()
                named_after = after[column].notna().to_numpy()
                both_named = named_before & named_after
                n_column_unpaired = int((named_before != named_after).sum())
                model_unpaired += n_column_unpaired
                session_model_unpaired += n_column_unpaired
                model_disagreements += int(
                    (
                        before[column].to_numpy()[both_named]
                        != after[column].to_numpy()[both_named]
                    ).sum()
                )
        for name in SHARED_MODULATION_ARRAYS:
            before_array = _bundle_modulation(before_dir, session, "", name)
            after_array = _bundle_modulation(after_dir, session, artifact_suffix, name)
            if before_array.shape != after_array.shape:
                continue
            difference, n_unpaired = _max_abs_difference(before_array, after_array)
            array_cells[name] += before_array.size
            array_unpaired[name] += n_unpaired
            session_unpaired += n_unpaired
            if np.isfinite(difference):
                array_worst[name] = max(array_worst[name], difference)
                finite = np.isfinite(before_array) & np.isfinite(after_array)
                n_array_outside = int(
                    (
                        np.abs(after_array[finite] - before_array[finite]) > param_tol
                    ).sum()
                )
                array_outside[name] += n_array_outside
                session_outside += n_array_outside
        per_session_rows.append(
            {
                "session": session,
                "n_neurons": len(before),
                "n_cells_outside_tolerance": session_outside,
                "n_cells_finite_on_one_side_only": session_unpaired,
                "n_best_aic_labels_on_one_side_only": session_model_unpaired,
            }
        )

    rows = [
        {
            "quantity": family,
            "n_cells": n_cells[family],
            "max_abs_diff": worst[family] if n_cells[family] else np.nan,
            "tolerance": families[family][1],
            "n_outside_tolerance": outside[family],
            "n_finite_on_one_side_only": unpaired[family],
        }
        for family in families
    ]
    rows += [
        {
            "quantity": f"modulation `{name}`",
            "n_cells": array_cells[name],
            "max_abs_diff": array_worst[name] if array_cells[name] else np.nan,
            "tolerance": param_tol,
            "n_outside_tolerance": array_outside[name],
            "n_finite_on_one_side_only": array_unpaired[name],
        }
        for name in SHARED_MODULATION_ARRAYS
    ]
    rows.append(
        {
            "quantity": "best-AIC model (identity)",
            "n_cells": n_models_compared,
            "max_abs_diff": np.nan,
            "tolerance": np.nan,
            "n_outside_tolerance": model_disagreements,
            "n_finite_on_one_side_only": model_unpaired,
        }
    )
    per_quantity = pd.DataFrame(rows)
    per_quantity.attrs["n_neurons_paired"] = n_neurons
    per_quantity.attrs["n_sessions_paired"] = len(sessions) - len(unpairable_rows)
    return per_quantity, pd.DataFrame(per_session_rows), pd.DataFrame(unpairable_rows)


def matched_control_arrays(data_dir: Path, artifact_suffix: str = "") -> pd.DataFrame:
    """Whether every bundle carries the matched-control modulation, and its shape.

    A bundle without the array was fit by code that had no control, so its
    engagement test does not exist; the audit must say so per session rather
    than report a pooled fraction over an unknown denominator.
    """
    rows = []
    for session in sorted(bundle_sessions(data_dir, artifact_suffix)):
        base = data_dir / f"{session}_mPFC_modulation{artifact_suffix}"
        array_path = Path(f"{base}.{MATCHED_CONTROL_ARRAY}.npy")
        reference = Path(f"{base}.non_local_no_SWR_modulation.npy")
        present = array_path.exists()
        shape = np.load(array_path).shape if present else None
        rows.append(
            {
                "session": session,
                "matched_control_array": present,
                "shape": str(shape) if present else "",
                "matches_non_local_no_SWR_shape": (
                    present and reference.exists() and shape == np.load(reference).shape
                ),
            }
        )
    return pd.DataFrame(rows)


def _fdr_q(frame: pd.DataFrame, test: str) -> NDArray:
    """BH-corrected q-values for ``test``, from the frame's own p-values.

    ``load_population_data`` corrects only the six tests it ships, so the two
    matched-control tests are corrected here, over the same population and by
    the same method, and only over the finite p-values.
    """
    column = f"{test}: pval"
    if column not in frame.columns:
        return np.full(len(frame), np.nan)
    raw = frame[column].to_numpy(dtype=float)
    q = np.full_like(raw, np.nan)
    ok = np.isfinite(raw)
    if ok.any():
        q[ok] = false_discovery_control(np.clip(raw[ok], 1e-300, 1.0), method="bh")
    return q


def matched_control_significance(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """FDR-significant fraction of the two matched-control tests, per animal.

    BH is applied within the pooled population, which is where the two tests
    are read; the per-animal rows are that same correction sliced by animal,
    so they are comparable with the pooled row rather than separate families.
    """
    frame = frame.copy()
    for test in MATCHED_CONTROL_TESTS:
        frame[f"{test}: fdr_q"] = _fdr_q(frame, test)
    rows = []
    groups = [("all", frame)] + [
        (animal, sub) for animal, sub in frame.groupby("animal", sort=True)
    ]
    for animal, sub in groups:
        for test in MATCHED_CONTROL_TESTS:
            q = sub[f"{test}: fdr_q"].to_numpy(dtype=float)
            tested = np.isfinite(q)
            n_tested = int(tested.sum())
            n_sig = int((q[tested] < alpha).sum())
            rows.append(
                {
                    "animal": animal,
                    "test": test,
                    "n_tested": n_tested,
                    "n_sig": n_sig,
                    "frac_sig": n_sig / n_tested if n_tested else np.nan,
                }
            )
    return pd.DataFrame(rows)


def fitter_table(sidecars: pd.DataFrame) -> pd.DataFrame:
    """Which estimator, device and inputs each session was fit with.

    One row per distinct combination rather than per session: a batch run as
    one job should collapse to a single row per card, and any session that did
    not is then visible without reading 71 rows. The cached decode frame's
    hash is deliberately not part of the combination -- it is a property of the
    session, not of the run, so grouping on it would put every session in a row
    of its own; the number of distinct values is reported instead, and it
    should equal the number of sessions.
    """
    columns = [
        "fitter",
        "fitter_tol",
        "jax_backend",
        "jax_version",
        "gpu_index",
        "gpu_name",
        "stop_table_sha256",
        "canonical_event_table_sha256",
        "git_hash",
    ]
    present = [c for c in columns if c in sidecars.columns]
    frame = sidecars.copy()
    for column in present:
        if column.endswith("sha256"):
            frame[column] = frame[column].map(
                lambda value: value[:12] if isinstance(value, str) else value
            )
    grouped = frame.groupby(present, dropna=False)
    table = grouped.agg(
        n_sessions=("session", "size"),
        n_distinct_data_over_time=("data_over_time_sha256", "nunique"),
    ).reset_index()
    for column in (
            "n_model_fits",
        "n_nonconverged_model_fits",
        "n_neurons_with_nonconverged_model",
    ):
        values = frame.get(column)
        if values is None:
            continue
        totals = (
            frame.assign(_total=_numeric(values))
            .groupby(present, dropna=False)["_total"]
            .sum()
            .reset_index(name=column)
        )
        table = table.merge(totals, on=present, how="left")
        table[column] = table[column].astype("Int64")
    return table


def session_fit_counts_table(sidecars: pd.DataFrame) -> pd.DataFrame:
    """Per session: the card it was fit on, and what its fits did.

    The collapsed estimator table says the batch was one job; this says what
    each session inside it cost in fits that never settled, which is the
    quantity a reader has to weigh a session's claims against.
    """
    columns = [
        "session",
        "gpu_index",
            "n_model_fits",
        "n_nonconverged_model_fits",
        "n_neurons_with_nonconverged_model",
    ]
    present = [c for c in columns if c in sidecars.columns]
    table = sidecars[present].copy().sort_values("session")
    for column in present[1:]:
        table[column] = _numeric(table[column]).astype("Int64")
    if {"n_nonconverged_model_fits", "n_model_fits"} <= set(present):
        table["nonconverged_frac"] = (
            _numeric(table["n_nonconverged_model_fits"])
            / _numeric(table["n_model_fits"])
        ).round(4)
    return table.reset_index(drop=True)


#: The ``/usr/bin/time -v`` fields a batch's wall clock is read from.
_TIME_FIELDS = {
    "Elapsed (wall clock) time (h:mm:ss or m:ss)": "wall_clock",
    "Percent of CPU this job got": "cpu_pct",
    "Maximum resident set size (kbytes)": "max_rss_kb",
    "Exit status": "exit_status",
}


def worker_time_table(paths: list[Path]) -> pd.DataFrame:
    """What each worker of the batch cost, from its ``/usr/bin/time -v`` file.

    Parameters
    ----------
    paths : list of Path
        ``worker*.time`` files. A sibling ``worker*.sessions`` partition file,
        when present, gives the number of sessions that worker was handed.

    Returns
    -------
    pandas.DataFrame
        One row per worker: wall clock, CPU percentage, peak resident set in
        GB, exit status and session count.
    """
    rows = []
    for path in sorted(paths):
        fields: dict[str, str] = {}
        for line in Path(path).read_text().splitlines():
            key, _, value = line.strip().partition(": ")
            if key in _TIME_FIELDS:
                fields[_TIME_FIELDS[key]] = value.strip()
        sessions = Path(path).with_suffix(".sessions")
        n_sessions = (
            len(
                [
                    line
                    for line in sessions.read_text().splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
            )
            if sessions.exists()
            else None
        )
        rows.append(
            {
                "worker": Path(path).stem,
                "n_sessions": n_sessions,
                "wall_clock": fields.get("wall_clock"),
                "cpu_pct": fields.get("cpu_pct"),
                "peak_rss_gb": (
                    round(int(fields["max_rss_kb"]) / 1e6, 1)
                    if "max_rss_kb" in fields
                    else None
                ),
                "exit_status": fields.get("exit_status"),
            }
        )
    return pd.DataFrame(rows)


def _cell(value: object, float_format: str) -> str:
    """One markdown cell: missing renders blank, a count renders as a count."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return float_format.format(value)
    return str(value)


def _md(frame: pd.DataFrame, float_format: str = "{:.4g}") -> str:
    """A markdown table, with missing values rendered as an empty cell.

    Cells are rendered by iterating the columns rather than with
    ``Series.map``: mapping over a nullable-integer column coerces its values
    to float, which turns a neuron count into ``10986.0``.
    """
    if frame.empty:
        return "_(no rows)_\n"
    columns = {
        name: [_cell(value, float_format) for value in frame[name]]
        for name in frame.columns
    }
    header = "| " + " | ".join(str(name) for name in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(cells[i] for cells in columns.values()) + " |"
        for i in range(len(frame))
    ]
    return "\n".join([header, rule, *body]) + "\n"


# ---------------------------------------------------------------------------
# The estimator change: penalised fits, scored out of fold
# ---------------------------------------------------------------------------

#: Sidecar keys that state the estimator and the inference a bundle was
#: produced by. A bundle from the earlier maximum-likelihood batch carries none
#: of them, which is how the two trees are told apart.
ESTIMATOR_SIDECAR_KEYS = [
    "estimator",
    "inference",
    "penalty_sigma_log_rate",
    "n_folds",
    "fold_seed",
    "n_control_draws",
    "control_draw_seed",
    "n_basis_per_class",
]

#: The nested ``full vs. reduced`` comparisons the bundles carry a held-out
#: difference for, as the statistics table spells them. Mirrors the runner's
#: list: the audit reads the columns by name, so it has to know them.
NESTED_MODEL_PAIRS = (
    "SWR vs. constant",
    "non_local vs. constant",
    "non_local_no_SWR vs. constant",
    "non_local_SWR vs. constant",
    "SWR + non_local_no_SWR vs. constant",
    "SWR + non_local_no_SWR vs. SWR",
    "SWR + non_local_no_SWR vs. non_local_no_SWR",
    "non_local_SWR + non_local_no_SWR vs. constant",
    "non_local_SWR + non_local_no_SWR vs. non_local_SWR",
    "non_local_SWR + non_local_no_SWR vs. non_local_no_SWR",
    "non_local_SWR + non_local_no_SWR vs. non_local",
    "matched_control + non_local_no_SWR vs. matched_control",
    "matched_control vs. constant",
)

#: Peri-event curve each event class is read from, and the model it comes from.
#: The names are the bundle's array names; ``non_local_no_SWR_split`` is the
#: same class seen through the replay-split model rather than the SWR-
#: controlled one.
MODULATION_CURVES = {
    "SWR": "SWR_modulation",
    "non_local_no_SWR": "no_SWR_modulation",
    "non_local_SWR": "non_local_SWR_modulation",
    "non_local_no_SWR_split": "non_local_no_SWR_modulation",
    "matched_control": "matched_control_modulation",
}

#: Which model's ``separated_columns`` count says whether a class's fit had a
#: basis column with no spikes under it -- the configuration whose maximum
#: likelihood estimate does not exist.
SEPARATION_MODEL = {
    "SWR": "SWR + non_local_no_SWR",
    "non_local_no_SWR": "SWR + non_local_no_SWR",
    "non_local_SWR": "non_local_SWR + non_local_no_SWR",
    "non_local_no_SWR_split": "non_local_SWR + non_local_no_SWR",
    "matched_control": "matched_control + non_local_no_SWR",
}

#: Event count in the sidecar that drives each class's peak, for the
#: approximate estimation error below.
CLASS_EVENT_COUNT = {
    "SWR": "n_ripple_after",
    "non_local_no_SWR": "n_events_non_swr",
    "non_local_SWR": "n_events_swr",
    "non_local_no_SWR_split": "n_events_non_swr",
    "matched_control": "n_matched_control",
}

#: Integral of a squared cubic B-spline basis function of this basis, in
#: seconds: 50 ms knots, so a basis function spans four knot intervals and
#: ``int b(t)^2 dt`` is about 0.6 of one interval. Used only in the
#: approximate estimation error below, where a factor of two does not change
#: which side of the threshold a unit falls on.
BASIS_SECOND_MOMENT_S = 0.030


def curve_peaks(curves: NDArray) -> NDArray:
    """Peak log-modulation per unit: the curve's value where ``|curve|`` is largest.

    The shipped curve is ``spline_basis @ params``, so this is the same
    quantity a curve-level peak helper returns from the coefficients -- read
    off the artifact rather than refit.

    Parameters
    ----------
    curves : NDArray, shape (n_lag_times, n_units)

    Returns
    -------
    NDArray, shape (n_units,)
        ``nan`` for a unit whose curve is not finite (its model was not
        estimable).
    """
    curves = np.asarray(curves, dtype=float)
    peaks = np.full(curves.shape[1], np.nan)
    usable = np.isfinite(curves).all(axis=0)
    if usable.any():
        block = curves[:, usable]
        peaks[usable] = block[np.abs(block).argmax(axis=0), np.arange(block.shape[1])]
    return peaks


def approximate_peak_error(rate_hz: NDArray, n_events: NDArray) -> NDArray:
    """Scale of the estimation error on a peak log-rate modulation.

    The peak of a partition-of-unity B-spline kernel is essentially one
    coefficient, and the Poisson information for that coefficient is
    ``sum_t mu_t x_tk**2``, which for ``n`` events at baseline rate
    ``lambda`` is about ``lambda * n * int b_k(t)**2 dt``. So the standard
    error is about ``1 / sqrt(lambda * n * m)`` with ``m`` the basis
    function's second moment in seconds.

    This is a scale, not the fit's own standard error: the bundles ship the
    fitted curves and not their covariance, so a unit-by-unit posterior
    standard error is not available from the artifacts. It is used only to ask
    whether a coefficient moved by more than the data could pin it down, where
    an approximation good to a factor of order one is enough to separate "the
    prior did the work" from "the spikes did".

    Parameters
    ----------
    rate_hz : NDArray, shape (n_units,)
        Each unit's mean firing rate over the fitted bins.
    n_events : NDArray, shape (n_units,)
        Events of the class the peak belongs to, in that unit's session.

    Returns
    -------
    NDArray, shape (n_units,)
    """
    information = (
        np.asarray(rate_hz, dtype=float)
        * np.asarray(n_events, dtype=float)
        * BASIS_SECOND_MOMENT_S
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(information > 0, 1.0 / np.sqrt(information), np.inf)


def peak_shrinkage(
    before_dir: Path,
    after_dir: Path,
    artifact_suffix: str = "",
    sidecars: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per unit and class, the peak log-modulation under each estimator.

    The two trees hold the same sessions fit the same way except for the
    prior, and each session's units are in the order its spike-time list gave,
    so the two tables pair row for row. A session whose row counts disagree is
    reported rather than paired.

    Returns
    -------
    pandas.DataFrame
        One row per (session, unit, class): ``peak_mle``, ``peak_penalised``,
        ``delta``, ``separated_columns`` (from the after bundle), the unit's
        fitted-bin firing rate and its class's event count.
    """
    events = (
        sidecars.set_index("session") if sidecars is not None else pd.DataFrame()
    )
    rows = []
    shared = sorted(
        bundle_sessions(before_dir) & bundle_sessions(after_dir, artifact_suffix)
    )
    for session in shared:
        before_stats = _bundle_statistics(before_dir, session, "")
        after_stats = _bundle_statistics(after_dir, session, artifact_suffix)
        if len(before_stats) != len(after_stats) or not len(after_stats):
            continue
        rate = _numeric(
            after_stats.get(FIT_RATE_COLUMN, pd.Series(np.nan, index=after_stats.index))
        ).to_numpy()
        for name, array in MODULATION_CURVES.items():
            try:
                before_curves = _bundle_modulation(before_dir, session, "", array)
                after_curves = _bundle_modulation(
                    after_dir, session, artifact_suffix, array
                )
            except FileNotFoundError:
                # The matched non-event control postdates the first bundles,
                # so a tree written before it has no curve for that class.
                # There is nothing to compare, which is not the same as a
                # comparison that came out at zero.
                continue
            before_peak = curve_peaks(before_curves)
            after_peak = curve_peaks(after_curves)
            separated_col = f"{SEPARATION_MODEL[name]}: separated_columns"
            separated = (
                _numeric(after_stats[separated_col]).to_numpy()
                if separated_col in after_stats
                else np.full(len(after_stats), np.nan)
            )
            n_events = np.nan
            if session in events.index:
                n_events = pd.to_numeric(
                    events.loc[session].get(CLASS_EVENT_COUNT[name], np.nan),
                    errors="coerce",
                )
            rows.append(
                pd.DataFrame(
                    {
                        "session": session,
                        "unit": np.arange(len(after_stats)),
                        "event_class": name,
                        "peak_mle": before_peak,
                        "peak_penalised": after_peak,
                        "separated_columns": separated,
                        "fit_rate_hz": rate,
                        "n_events": float(n_events),
                    }
                )
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "session",
                "unit",
                "event_class",
                "peak_mle",
                "peak_penalised",
                "separated_columns",
                "fit_rate_hz",
                "n_events",
                "delta",
                "peak_error",
                "prior_binds",
            ]
        )
    frame = pd.concat(rows, ignore_index=True)
    frame["delta"] = frame["peak_penalised"] - frame["peak_mle"]
    frame["peak_error"] = approximate_peak_error(
        frame["fit_rate_hz"].to_numpy(), frame["n_events"].to_numpy()
    )
    frame["prior_binds"] = frame["delta"].abs() > frame["peak_error"]
    return frame


def shrinkage_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per class and separation status: how far the prior moved the peak."""
    if frame.empty:
        return frame
    paired = frame.dropna(subset=["peak_mle", "peak_penalised"]).copy()
    paired["units"] = np.where(
        paired["separated_columns"].fillna(0) > 0, "separated", "well posed"
    )
    grouped = paired.groupby(["event_class", "units"], dropna=False)
    table = grouped.agg(
        n_units=("delta", "size"),
        median_abs_peak_mle=("peak_mle", lambda s: float(np.median(np.abs(s)))),
        median_abs_peak_penalised=(
            "peak_penalised",
            lambda s: float(np.median(np.abs(s))),
        ),
        median_abs_delta=("delta", lambda s: float(np.median(np.abs(s)))),
        max_abs_delta=("delta", lambda s: float(np.max(np.abs(s)))),
        n_prior_binds=("prior_binds", "sum"),
    ).reset_index()
    table["prior_binds_pct"] = (
        100.0 * table["n_prior_binds"] / table["n_units"]
    ).round(1)
    return table


def heldout_fraction_table(
    after: pd.DataFrame, alpha: float
) -> pd.DataFrame:
    """Units beating the control-draw null, per class, overall and per animal."""
    rows = []
    for scope, frame in [("all", after), *sorted(after.groupby("animal"))]:
        row: dict[str, object] = {"animal": scope, "n_units": len(frame)}
        for name, (n_significant, n_tested) in significant_fraction(
            frame, alpha=alpha
        ).items():
            row[f"{name}_n_tested"] = n_tested
            row[f"{name}_pct"] = (
                round(100.0 * n_significant / n_tested, 1) if n_tested else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def legacy_fdr_table(before: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """The same fractions the superseded bundles reported, per test and animal."""
    rows = []
    for scope, frame in [("all", before), *sorted(before.groupby("animal"))]:
        row: dict[str, object] = {"animal": scope, "n_units": len(frame)}
        for test in FDR_TESTS:
            column = f"{test}: fdr_q"
            if column not in frame:
                row[f"{test}_pct"] = np.nan
                continue
            q = frame[column].to_numpy(dtype=float)
            tested = np.isfinite(q)
            row[f"{test}_n_tested"] = int(tested.sum())
            row[f"{test}_pct"] = (
                round(100.0 * (q[tested] < alpha).sum() / tested.sum(), 1)
                if tested.any()
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def separated_column_table(after: pd.DataFrame) -> pd.DataFrame:
    """Per model, how many units had a basis column with no spikes under it."""
    rows = []
    for column in sorted(
        c for c in after.columns if c.endswith(": separated_columns")
    ):
        values = _numeric(after[column]).to_numpy()
        finite = np.isfinite(values)
        rows.append(
            {
                "model": column.removesuffix(": separated_columns"),
                "n_units": int(finite.sum()),
                "n_units_with_a_separated_column": int((values[finite] > 0).sum()),
                "pct_units": round(
                    100.0 * (values[finite] > 0).sum() / finite.sum(), 1
                )
                if finite.any()
                else np.nan,
                "max_separated_columns": (
                    int(values[finite].max()) if finite.any() else 0
                ),
            }
        )
    return pd.DataFrame(rows)


def fold_spread_table(after: pd.DataFrame) -> pd.DataFrame:
    """Per nested pair, how much the held-out difference moves between folds.

    The statistic is each unit's fold-to-fold standard deviation of the
    difference divided by the absolute value of its mean; the table reports
    the median over units. A pair whose fold-to-fold spread dwarfs its mean is
    a pair whose total is one fold's accident, which is what the number of
    folds has to be chosen against.
    """
    rows = []
    for column in sorted(
        c for c in after.columns if c.endswith(": heldout_delta_dev_folds")
    ):
        stacked = np.array(
            [np.asarray(value, dtype=float) for value in after[column]]
        )
        if stacked.ndim != 2 or not stacked.size:
            continue
        mean = stacked.mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = stacked.std(axis=1, ddof=1) / np.abs(mean)
        usable = np.isfinite(relative)
        rows.append(
            {
                "pair": column.removesuffix(": heldout_delta_dev_folds"),
                "n_folds": stacked.shape[1],
                "n_units": int(usable.sum()),
                "median_fold_sd_over_abs_mean": (
                    round(float(np.median(relative[usable])), 3)
                    if usable.any()
                    else np.nan
                ),
                "median_abs_mean": (
                    round(float(np.median(np.abs(mean[np.isfinite(mean)]))), 4)
                    if np.isfinite(mean).any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def sigma_sensitivity_table(
    after_dir: Path, artifact_suffix: str = ""
) -> pd.DataFrame:
    """Peak modulation at the canonical prior against each sensitivity bundle.

    Reads whatever ``*_modulation<suffix>_sigma*`` bundles are on disk beside
    the canonical ones and pairs them session by session; empty when none were
    written.
    """
    rows = []
    stem = f"_mPFC_modulation{artifact_suffix}_sigma"
    suffixes = sorted(
        {
            path.name.split(stem)[1].removesuffix(".pkl")
            for path in after_dir.glob(f"*{stem}*.pkl")
        }
    )
    for sigma in suffixes:
        variant = f"{artifact_suffix}_sigma{sigma}"
        shared = sorted(
            bundle_sessions(after_dir, artifact_suffix)
            & bundle_sessions(after_dir, variant)
        )
        for name, array in MODULATION_CURVES.items():
            deltas = []
            for session in shared:
                canonical = curve_peaks(
                    _bundle_modulation(after_dir, session, artifact_suffix, array)
                )
                other = curve_peaks(
                    _bundle_modulation(after_dir, session, variant, array)
                )
                if canonical.shape != other.shape:
                    continue
                deltas.append(other - canonical)
            if not deltas:
                continue
            stacked = np.concatenate(deltas)
            finite = np.isfinite(stacked)
            rows.append(
                {
                    "sigma": sigma.replace("p", "."),
                    "event_class": name,
                    "n_units": int(finite.sum()),
                    "median_abs_delta_peak": round(
                        float(np.median(np.abs(stacked[finite]))), 4
                    )
                    if finite.any()
                    else np.nan,
                    "max_abs_delta_peak": round(
                        float(np.max(np.abs(stacked[finite]))), 4
                    )
                    if finite.any()
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


#: Sidecar keys that are per session by construction: they are derived from
#: the session name, so grouping on them would put every session in a row of
#: its own. Their distinctness is reported instead of their value.
PER_SESSION_SEED_KEYS = ("fold_seed", "control_draw_seed")


def estimator_provenance_table(sidecars: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct estimator/inference combination across the bundles.

    A batch run as one job collapses to a single row, and any session that was
    produced differently is then visible without reading every sidecar. The
    two draw seeds are session-derived, so the table counts how many distinct
    values each took: anything other than one per session would mean two
    sessions shared a split or a null.
    """
    present = [
        c
        for c in ESTIMATOR_SIDECAR_KEYS
        if c in sidecars.columns and c not in PER_SESSION_SEED_KEYS
    ]
    if not present:
        return pd.DataFrame()
    grouped = sidecars.groupby(present, dropna=False)
    aggregations = {"n_sessions": ("session", "size")}
    for key in PER_SESSION_SEED_KEYS:
        if key in sidecars.columns:
            aggregations[f"n_distinct_{key}"] = (key, "nunique")
    return grouped.agg(**aggregations).reset_index()


def rerun_sections(
    *,
    before_sidecars: pd.DataFrame | None,
    after_sidecars: pd.DataFrame,
    after: pd.DataFrame,
    before_dir: Path,
    after_dir: Path,
    alpha: float,
    artifact_suffix: str,
    worker_times: list[Path],
) -> list[str]:
    """The rerun audit: same events, new estimator, plus the new control.

    Three things a reader of the rerun needs, in order. Whether the two trees
    really do describe the same events (without which the next section is not a
    check of the estimator at all); the unit-by-unit differences on everything
    both trees carry, against the tolerances the estimator was accepted at; and
    what the matched non-event control -- which exists only on the after side
    -- says. The estimator/device provenance closes it.

    Returns an empty list when the after tree records no estimator and no
    matched control, i.e. it is not a rerun tree and none of this applies.
    """
    lines: list[str] = []
    add = lines.append
    # Every sidecar row carries these keys, ``None`` where the sidecar predates
    # them, so a tree is a rerun tree only if some row actually filled one in.
    is_rerun_tree = any(
        column in after_sidecars.columns and after_sidecars[column].notna().any()
        for column in ("fitter", "canonical_event_table_sha256")
    )
    if not is_rerun_tree:
        return lines

    add("## Same events, new estimator\n")
    same_events, why = same_events_precondition(before_sidecars, after_sidecars)
    if not same_events:
        add(
            "**The before/after coefficient check is not run**: "
            f"{why}. Without it a difference between the two trees cannot be "
            "attributed to the estimator rather than to what it was fit on.\n"
        )
    else:
        counts, n_disagree = event_count_agreement(before_sidecars, after_sidecars)
        add(
            f"Precondition: {why}. Whether that marker is the whole story is "
            "settled by the event counts themselves -- "
            + (
                f"**{n_disagree} of {len(counts)} shared session(s) have "
                "different event counts on the two sides**, so for those the "
                "comparison below is not a check of the estimator: they were "
                "fit on different regressors.\n"
                if n_disagree
                else f"all {len(counts)} shared session(s) carry identical "
                "non-local, SWR-coupled and non-SWR counts, so the two trees "
                "differ only in how the same models were fit.\n"
            )
        )
        if n_disagree:
            add(_md(counts.loc[~counts["same_events"]].head(20)))
        shared = sorted(
            set(before_sidecars["session"]) & set(after_sidecars["session"])
        )
        differences, per_session, unpairable = shared_statistic_differences(
            before_dir, after_dir, shared, artifact_suffix=artifact_suffix
        )
        add(
            "Every neuron of every shared session, paired by its row position "
            "within the session, on every column and modulation array both "
            "bundles carry. The tolerances are the ones the estimator was "
            f"accepted at on the gate sessions: {RERUN_PARAM_TOL:g} on "
            f"parameters and anything derived from them, {RERUN_PVALUE_TOL:g} "
            "on LR p-values. `n_finite_on_one_side_only` counts cells that are "
            "NaN on exactly one side -- a model that became estimable or "
            "stopped being, which no maximum over finite differences can "
            f"see. {differences.attrs['n_neurons_paired']:,} neurons over "
            f"{differences.attrs['n_sessions_paired']} session(s).\n"
        )
        add(_md(differences, "{:.3g}"))
        n_outside = int(differences["n_outside_tolerance"].sum())
        add(
            f"**{n_outside:,} cell(s) fall outside tolerance.**\n"
            if n_outside
            else "**No cell falls outside tolerance.**\n"
        )
        if not per_session.empty:
            moved = per_session.loc[
                (per_session["n_cells_outside_tolerance"] > 0)
                | (per_session["n_cells_finite_on_one_side_only"] > 0)
                | (per_session["n_best_aic_labels_on_one_side_only"] > 0)
            ]
            add(
                'Per session, because "expected zero" is a claim about every '
                "session and not about a corpus total: "
                + (
                    f"**all {len(per_session)} paired session(s) have zero "
                    "cells outside tolerance, zero cells finite on one side "
                    "only and zero neurons that carry a best-AIC label on one "
                    "side and none on the other.**\n"
                    if moved.empty
                    else f"{len(per_session) - len(moved)} of "
                    f"{len(per_session)} paired session(s) are clean on all "
                    f"three counts; the {len(moved)} that are not are listed "
                    "below.\n"
                )
            )
            if not moved.empty:
                add(_md(moved))
                lost = int(moved["n_best_aic_labels_on_one_side_only"].sum())
                add(
                    f"{lost} best-AIC label(s) exist on one side only, over "
                    "the two label columns each neuron carries.\n"
                )
        if not unpairable.empty:
            add(
                f"{len(unpairable)} session(s) could not be paired at all: the "
                "two trees hold different numbers of neurons for them, so no "
                "row correspondence exists.\n"
            )
            add(_md(unpairable))

    add("## The matched non-event control\n")
    arrays = matched_control_arrays(after_dir, artifact_suffix)
    missing = arrays.loc[~arrays["matched_control_array"]]
    mismatched = arrays.loc[
        arrays["matched_control_array"] & ~arrays["matches_non_local_no_SWR_shape"]
    ]
    add(
        f"{len(arrays) - len(missing)} of {len(arrays)} bundle(s) carry a "
        f"`{MATCHED_CONTROL_ARRAY}` array"
        + (
            ", every one the same shape as that bundle's "
            "`non_local_no_SWR_modulation`.\n"
            if missing.empty and mismatched.empty
            else f"; {len(missing)} have none and {len(mismatched)} have one "
            "whose shape does not match that bundle's "
            "`non_local_no_SWR_modulation`, so those sessions have no usable "
            "engagement test.\n"
        )
    )
    if not missing.empty or not mismatched.empty:
        add(_md(pd.concat([missing, mismatched]).head(20)))
    placed = after_sidecars.get("n_matched_control")
    if placed is not None and placed.notna().any():
        unplaced = _numeric(
            after_sidecars.get(
                "n_matched_control_unplaced",
                pd.Series(0, index=after_sidecars.index),
            )
        )
        outside_bout = after_sidecars.get(
            "n_matched_control_onset_outside_fitted_bout",
            pd.Series(np.nan, index=after_sidecars.index),
        )
        saturated = after_sidecars.get(
            "n_matched_control_bout_saturated",
            pd.Series(np.nan, index=after_sidecars.index),
        )
        source = after_sidecars["n_nonlocal_after"]
        n_placed = _numeric(placed).sum()
        add(
            f"{int(n_placed):,} control windows were placed against "
            f"{int(source.sum()):,} non-local events "
            f"({100.0 * n_placed / source.sum():.1f}%); "
            f"{int(unplaced.sum()):,} events got none"
            + (
                f" ({int(_numeric(outside_bout).sum()):,} with an onset in no "
                f"fitted bout, {int(_numeric(saturated).sum()):,} in a bout "
                "with no free window of their duration).\n"
                if outside_bout.notna().any()
                else ".\n"
            )
        )
        add(
            "The control is matched one per non-local event, SWR-coupled ones "
            "included, so it is more numerous than the `non_local_no_SWR` "
            "class the engagement test compares against; a source event's "
            "duration fixes its control's duration and constrains where the "
            "window fits, and nothing else about the event is matched.\n"
        )
    add(f"### FDR-significant fraction, matched-control tests (q < {alpha})\n")
    add(
        "`load_population_data` does not correct these two, so BH is applied "
        "here over the same pooled population, per test, on the finite "
        "p-values only; the per-animal rows slice that one correction. "
        "`matched_control + non_local_no_SWR vs. matched_control` is the "
        "engagement test -- does non-SWR replay add beyond a duration-matched "
        "non-event baseline from the same bouts. `matched_control vs. "
        "constant` asks whether that baseline modulates firing on its own; "
        "where it does, the peri-event window is not exchangeable with the "
        "rest of the bout and the engagement test is the comparison to read.\n"
    )
    add(_md(matched_control_significance(after, alpha)))

    add("## Estimator, device and inputs\n")
    add(
        "One row per distinct combination, so a batch run as one job per card "
        "collapses to one row per card. `n_nonconverged_model_fits` counts the "
        "shipped neuron x model fits that ran to `maxiter` without settling, "
        "over `n_neurons_with_nonconverged_model` neurons. Those fits are "
        "non-estimable for the neuron that owns them -- every statistic that "
        "reads one is NaN -- so these counts are the only record of how much "
        "was dropped and from where.\n"
    )
    add(_md(fitter_table(after_sidecars)))
    add(
        "And per session, so a claim can be weighed against the fits it rests "
        "on: `nonconverged_frac` is `n_nonconverged_model_fits / "
        "n_model_fits`.\n"
    )
    add(_md(session_fit_counts_table(after_sidecars)))
    nonconverged = after_sidecars.get("n_nonconverged_model_fits")
    if nonconverged is not None and nonconverged.notna().any():
        neurons = _numeric(
            after_sidecars.get(
                "n_neurons_with_nonconverged_model",
                pd.Series(np.nan, index=after_sidecars.index),
            )
        )
        worst = after_sidecars.loc[_numeric(nonconverged).idxmax()]
        add(
            f"Across the corpus {int(_numeric(nonconverged).sum()):,} of "
            f"{int(_numeric(after_sidecars['n_model_fits']).sum()):,} shipped "
            f"model fits did not converge, over {int(neurons.sum()):,} "
            f"neurons; the worst single session is `{worst['session']}` with "
            f"{int(worst['n_nonconverged_model_fits'])}.\n"
        )
    if worker_times:
        add("## Wall clock\n")
        add(
            "One row per worker process of the batch, from the "
            "`/usr/bin/time -v` file each was run under. Sessions were split "
            "across the workers in advance, one card per worker, so the batch "
            "is as long as its slowest worker rather than the sum.\n"
        )
        add(_md(worker_time_table(worker_times)))
    return lines


def build_report(
    *,
    sidecars: pd.DataFrame,
    before: pd.DataFrame,
    after: pd.DataFrame,
    before_gate: dict,
    after_gate: dict,
    before_dir: Path,
    after_dir: Path,
    min_firing_rate: float,
    alpha: float,
    notes: list[str],
    command: str,
    artifact_suffix: str = "",
    before_sidecars: pd.DataFrame | None = None,
    worker_times: list[Path] | None = None,
) -> str:
    """Assemble the audit markdown."""
    worker_times = worker_times or []
    lines: list[str] = []
    add = lines.append

    add("# The mPFC unit GLM on the shared well-anchored stop cohort\n")
    add(
        "Generated by `scripts/qa/compare_pfc_cohorts.py`. The mPFC GLM used to "
        "fit every immobile 2 ms bin in the epoch and to key its design "
        "matrices on every speed-eligible non-local event and SWR. It now fits "
        "the immobile bins that fall inside a well-anchored stop, and only the "
        "events whose onset lies inside one -- the population the hippocampal "
        "rate and content ladders are read on, selected through the same "
        "`analysis.replay.cohort_selection` functions. The regressor basis "
        "still spans +/-250 ms around each kept event; bins outside the cohort "
        "are not fit.\n"
    )
    if command:
        add(f"Command: `{command}`\n")
    for note in notes:
        add(f"{note}\n")

    add("## Inputs\n")
    cohorts = sorted(sidecars["stop_cohort"].dropna().unique())
    hashes = sidecars["stop_table_sha256"].dropna().unique()
    before_sidecar_clause = (
        f"{len(before_sidecars)} provenance sidecars read.\n"
        if before_sidecars is not None and len(before_sidecars)
        else "no cohort restriction, no provenance sidecars (these bundles predate them).\n"
    )
    add(
        f"- before: `{before_dir}` -- {before_gate['n_neurons_with_fit']} neurons "
        f"with a usable fit, {before_sidecar_clause}"
        f"- after: `{after_dir}` -- {len(sidecars)} bundles, each with a "
        "provenance sidecar (a bundle without one is refused, so the tables "
        "below and the population summary describe one session set), every one "
        f"`stop_cohort` {cohorts}, every one on stop table "
        f"`{hashes[0][:12] if len(hashes) else '?'}`"
        f"{' (SHA-256 agrees across all bundles)' if len(hashes) == 1 else ''}.\n"
    )
    before_sessions = bundle_sessions(before_dir)
    after_sessions = bundle_sessions(after_dir, artifact_suffix)
    before_only = sorted(before_sessions - after_sessions)
    after_only = sorted(after_sessions - before_sessions)
    if before_only or after_only:
        add(
            "**The two trees do not hold the same sessions**, so the population "
            "totals below are not a like-for-like difference: "
            f"{len(before_only)} session(s) only in the before tree "
            f"({', '.join(before_only) or 'none'}) and {len(after_only)} only in "
            f"the after tree ({', '.join(after_only) or 'none'}).\n"
        )
    provenance = provenance_table(sidecars)
    add(
        "The sessions were fit in several passes, and each sidecar records the "
        "repository HEAD **at the moment it was written** -- which drifts ahead "
        "of the code the worker had already imported, and can even carry a "
        "`-dirty` marker for changes that are not in the imported code at all. "
        "`runner_code` is therefore derived from the sidecar's own fields "
        "instead: a session with no `alignment` block was fit before that step "
        f"existed, and a session trimmed by more than the {SINGLE_BUDGET_S} s "
        "single budget could only have been accepted by the later per-axis "
        "form.\n"
    )
    add(_md(provenance))
    dirty = sorted(
        sidecars.loc[sidecars["git_hash"].str.endswith(DIRTY_SUFFIX), "session"]
    )
    if dirty:
        add(
            f"{len(dirty)} sidecar(s) carry a `{DIRTY_SUFFIX}` marker "
            f"({', '.join(dirty)}): the working tree held uncommitted changes "
            "when the sidecar was written. The marker describes the tree at "
            "that moment, not the code the worker had imported hours earlier, "
            "so read it with `runner_code` beside it.\n"
        )
    add(
        "Between the earliest and the latest of those three, "
        "`git diff 50bf4b7..6c162ce -- src scripts/pfc` touches four files: "
        "`analysis/pfc/glm.py`, where the whole change is swapping the strict "
        "1-to-1 validator for `align_results_to_position` (once in "
        "`run_session`, once in `build_session_event_sets`), logging what it "
        "trimmed, and adding the `alignment` block to `cohort_info`; "
        "`analysis/hippocampus/data_aggregation.py`, which is the new function "
        "and its budget constants; and `scripts/pfc/run_stage6_learning.py` "
        "plus `analysis/replay/retrospective.py`, which are the Stage 6 and "
        "Stage 5B paths and are not imported by the unit runner. No estimation "
        "code moved. On matched axes the alignment step is the identity, which "
        "is what `tests/analysis/pfc/test_alignment.py::"
        "test_matched_axes_are_returned_untouched` pins, so the three regimes "
        "differ only in which sessions they accept -- not in what they fit.\n"
    )

    add("## Bins and events, before and after\n")
    table = session_cohort_table(sidecars)
    total = table.iloc[-1]
    add(
        f"Across {len(sidecars)} sessions the cohort keeps "
        f"**{total['bins_kept_pct']:.1f}%** of the immobile bins "
        f"({int(total['bins_fit_after']):,} of "
        f"{int(total['bins_immobile_before']):,}), "
        f"**{total['nonlocal_kept_pct']:.1f}%** of the non-local events "
        f"({int(total['nonlocal_after']):,} of "
        f"{int(total['nonlocal_before']):,}) and "
        f"{100.0 * total['ripple_after'] / total['ripple_before']:.1f}% of the "
        f"SWRs ({int(total['ripple_after']):,} of "
        f"{int(total['ripple_before']):,}). The kept non-local events split "
        f"{int(total['nonlocal_swr_after']):,} SWR-coupled / "
        f"{int(total['nonlocal_no_swr_after']):,} non-SWR.\n"
    )
    add(
        "`bins_immobile_before` is the before column -- every immobile bin was "
        "fit before -- and `bins_fit_after` the after. "
        "`align_results_bins_dropped` / `align_results_s_dropped` are the "
        "decode-axis trim this session needed to line the decode up with the "
        "position frame; blank where the sidecar predates that step.\n"
    )
    add(_md(table, "{:.2f}"))

    add("## Decode/position alignment\n")
    add(alignment_prose(sidecars))

    totals, affected = coverage_summary(sidecars)
    add("## Leading-edge coverage\n")
    add(
        "Events are dropped by onset, but the basis spans +/-250 ms, so an "
        "event just outside a stop could still have covered fitted bins. Those "
        "bins are fit with that event's contribution set to zero. Across the "
        f"corpus that is **{totals['n_nonlocal_dropped_overlapping_fit']} "
        f"non-local events** in {totals['n_sessions_nonlocal']} sessions and "
        f"**{totals['n_ripple_dropped_overlapping_fit']} SWRs** in "
        f"{totals['n_sessions_ripple']} sessions, out of "
        f"{int(total['nonlocal_after']):,} and {int(total['ripple_after']):,} "
        "kept events respectively.\n"
    )
    add(_md(affected))

    add("## The population that reaches Figure 4\n")
    add(
        f"Loaded with `load_population_data(min_firing_rate={min_firing_rate})` "
        "from each tree. A neuron enters when its GLM produced a finite primary "
        "p-value, its four modulation traces are NaN-free, and its "
        f"`{FIT_RATE_COLUMN}` clears the gate.\n"
    )
    gate = pd.DataFrame(
        [
            {"side": "before", **before_gate},
            {"side": "after", **after_gate},
        ]
    )
    for column in gate.columns.drop("side"):
        gate[column] = gate[column].astype("Int64")
    add(_md(gate))
    matched = after_gate.get("n_passing_all_immobile_rate")
    if matched is not None:
        shipped_before = before_gate["n_passing_fit_bin_rate"]
        shipped_after = after_gate["n_passing_fit_bin_rate"]
        add(
            "**Decomposed, the population moves in two independent steps.** "
            f"{shipped_before:,} neurons passed the gate before. Holding the "
            f"gated quantity fixed at the old all-immobile rate, the new tree "
            f"has {matched:,} ({matched - shipped_before:+,}) -- that step is "
            "per-neuron GLM convergence on the same screen, not the cohort. "
            f"Switching the screen to the fit-bin denominator then takes it to "
            f"{shipped_after:,} ({shipped_after - matched:+,}): the neurons "
            "whose rate over the cohort's immobile bins falls below the "
            f"threshold their whole-epoch rate cleared. Net "
            f"{shipped_after - shipped_before:+,}.\n"
        )
    add(
        f"The gated column means different things on the two sides. "
        f"`{FIT_RATE_COLUMN}` is the mean rate over the bins the bundle was fit "
        "on, so before the change it was the rate over every immobile bin and "
        "after it is the rate over the cohort's immobile bins. The new bundles "
        f"also carry `{ALL_IMMOBILE_RATE_COLUMN}`, the unrestricted rate, and "
        "`n_passing_all_immobile_rate` counts the neurons the *old* quantity "
        "would have admitted -- so the gate's contribution to any change in n "
        "is separable from the cohort's.\n"
    )

    add("### Neurons per animal\n")
    per_animal = (
        before.groupby("animal")
        .size()
        .rename("n_before")
        .to_frame()
        .join(after.groupby("animal").size().rename("n_after"), how="outer")
    ).reset_index()
    per_animal["delta"] = per_animal["n_after"] - per_animal["n_before"]
    add(_md(per_animal))

    add(f"### FDR-significant fraction per test (q < {alpha})\n")
    add(
        "`n_tested` is the neurons with a finite q for that test; a test is "
        "NaN for a neuron whose session had no events of the class it needs. "
        "BH is applied within each tree, so the q-values are not paired across "
        "the two sides -- read the fractions, not a per-neuron difference.\n"
    )
    comparison = population_comparison(before, after, alpha)
    add(_fdr_movement_prose(comparison))
    add(_md(comparison))

    lines.extend(
        rerun_sections(
            before_sidecars=before_sidecars,
            after_sidecars=sidecars,
            after=after,
            before_dir=before_dir,
            after_dir=after_dir,
            alpha=alpha,
            artifact_suffix=artifact_suffix,
            worker_times=worker_times,
        )
    )

    return "\n".join(lines)


def build_estimator_report(
    *,
    sidecars: pd.DataFrame,
    before: pd.DataFrame,
    after: pd.DataFrame,
    before_gate: dict,
    after_gate: dict,
    before_dir: Path,
    after_dir: Path,
    min_firing_rate: float,
    alpha: float,
    notes: list[str],
    command: str,
    artifact_suffix: str = "",
    worker_times: list[Path] | None = None,
) -> str:
    """Assemble the audit of the estimator and inference change.

    The before tree holds maximum-likelihood fits tested by a likelihood ratio
    and selected by AIC; the after tree holds penalised fits scored out of
    fold against a matched-control null. The two are not two runs of one
    estimator, so nothing here is a paired difference of a statistic: what is
    compared is the peak modulation each estimator reports for the same unit,
    and the fraction of units each inference calls.
    """
    worker_times = worker_times or []
    lines: list[str] = []
    add = lines.append

    add("# The mPFC unit GLM under a prior, scored out of fold\n")
    add(
        "Generated by `scripts/qa/compare_pfc_cohorts.py`. The unit GLMs were "
        "fit by maximum likelihood and tested by a likelihood ratio against an "
        "asymptotic chi-square, with AIC selecting between models. Neither "
        "survives the fits themselves: a basis column with no spikes under its "
        "support drives its coefficient to minus infinity, and neighbouring "
        "2 ms bins are not independent, so an in-sample deviance drop is worth "
        "much less than its nominal degrees of freedom. The models now carry a "
        "Gaussian prior on the event-basis coefficients -- loose enough that "
        "the spikes shape any modulation they can support -- and are compared "
        "by how well they predict spikes they were not fit on, in folds of "
        "whole immobility runs, against a null built by redrawing the "
        "duration-matched non-event control.\n"
    )
    if command:
        add(f"Command: `{command}`\n")
    for note in notes:
        add(f"{note}\n")

    add("## Inputs\n")
    add(
        f"- before: `{before_dir}` -- {before_gate['n_neurons_with_fit']} units "
        "with a usable fit, maximum likelihood, likelihood-ratio tests, AIC "
        "selection.\n"
        f"- after: `{after_dir}` -- {len(sidecars)} bundles, "
        f"{after_gate['n_neurons_with_fit']} units with a usable fit, "
        "penalised fits scored out of fold.\n"
    )
    before_sessions = bundle_sessions(before_dir)
    after_sessions = bundle_sessions(after_dir, artifact_suffix)
    before_only = sorted(before_sessions - after_sessions)
    after_only = sorted(after_sessions - before_sessions)
    if before_only or after_only:
        add(
            "**The two trees do not hold the same sessions**, so the totals "
            f"below are not a like-for-like difference: {len(before_only)} "
            f"session(s) only in the before tree ({', '.join(before_only) or 'none'}) "
            f"and {len(after_only)} only in the after tree "
            f"({', '.join(after_only) or 'none'}).\n"
        )

    add("## Estimator and inference, as the bundles state them\n")
    add(
        "One row per distinct combination. `penalty_sigma_log_rate` is the "
        "prior standard deviation of an event-basis coefficient on the "
        "log-rate scale; `n_folds` the blocked split the models were scored "
        "out of; `n_control_draws` the size of the null each held-out "
        "improvement is read against, so the finest p-value it can resolve is "
        "one over that. The two seeds are derived from the session name, so a "
        "session refit on its own reproduces the batch's split and null; the "
        "table counts their distinct values rather than showing them.\n"
    )
    add(_md(estimator_provenance_table(sidecars)))

    add("## What the prior did to the fits\n")
    add(
        "Per unit and event class, the peak of the peri-event log-modulation "
        "curve under each estimator -- the curve the bundle ships, so this is "
        "the coefficient-level peak read off the artifact rather than refit. "
        "Units are split by whether that class's model had a basis column with "
        "no spikes under its support in the fitted bins (`separated_columns`): "
        "those are the fits whose maximum likelihood estimate does not exist, "
        "and are where a prior is supposed to do the work.\n"
    )
    add(
        "`prior_binds` counts units whose peak moved by more than the "
        "estimation error the spikes afford it. The bundles ship fitted curves "
        "and not their covariance, so that error is the analytic scale "
        "`1 / sqrt(rate * n_events * m)` with `m` the basis function's second "
        "moment in seconds -- a scale good to a factor of order one, which is "
        "enough to separate 'the prior did the work' from 'the spikes did', "
        "and not a substitute for the fit's own posterior standard error.\n"
    )
    shrinkage = peak_shrinkage(before_dir, after_dir, artifact_suffix, sidecars)
    add(_md(shrinkage_table(shrinkage)))

    add("## The separating columns the prior was introduced for\n")
    add(
        "A basis column with no spikes anywhere under its support is the "
        "configuration that has no maximum likelihood estimate. The count is "
        "reported, not acted on: under the prior those coefficients are finite "
        "and small, and the unit is fit like any other.\n"
    )
    add(_md(separated_column_table(after)))

    add("## What the two inferences call\n")
    add(
        f"Before, BH-corrected likelihood-ratio p-values at q < {alpha}; after, "
        "BH-corrected empirical p-values from the control-draw null at the same "
        "level. The two are not comparable test by test -- they test different "
        "things against different references -- so read them as two accounts of "
        "the same population, not as a movement. `n_tested` is the units with a "
        "finite value for that test: a unit whose session had no events of a "
        "class is untested for it, not a negative.\n"
    )
    add("### Before: FDR-significant fraction per likelihood-ratio test\n")
    add(_md(legacy_fdr_table(before, alpha)))
    add("### After: fraction beating the control-draw null, per class\n")
    add(_md(heldout_fraction_table(after, alpha)))

    add("## How steady the held-out difference is between folds\n")
    add(
        "Each unit's fold-to-fold standard deviation of a nested pair's "
        "held-out difference, divided by the absolute value of its mean over "
        "folds; the median over units. It is the quantity the number of folds "
        "is chosen against: a pair whose spread dwarfs its mean has a total "
        "that is one fold's accident.\n"
    )
    add(_md(fold_spread_table(after)))

    add("## Sensitivity to the prior\n")
    sigma_table = sigma_sensitivity_table(after_dir, artifact_suffix)
    if sigma_table.empty:
        add(
            "No sensitivity bundle at another prior is on disk beside the "
            "canonical ones, so this section is empty. Write one with "
            "`run_pfc_analysis_bandit.py --penalty-sigma <value>`.\n"
        )
    else:
        add(
            "Peak log-modulation at each sensitivity prior against the "
            "canonical one, unit by unit. A curve the spikes determine barely "
            "moves; one the prior determines moves with it.\n"
        )
        add(_md(sigma_table))

    add("## The population that reaches Figure 4\n")
    add(
        f"Loaded with `load_population_data(min_firing_rate={min_firing_rate})` "
        "from each tree. A unit enters when its fit exists, its four shipped "
        f"modulation traces are NaN-free, and its `{FIT_RATE_COLUMN}` clears "
        "the gate.\n"
    )
    gate = pd.DataFrame(
        [{"side": "before", **before_gate}, {"side": "after", **after_gate}]
    )
    for column in gate.columns.drop("side"):
        gate[column] = gate[column].astype("Int64")
    add(_md(gate))

    add("### Units per animal\n")
    per_animal = (
        before.groupby("animal")
        .size()
        .rename("n_before")
        .to_frame()
        .join(after.groupby("animal").size().rename("n_after"), how="outer")
    ).reset_index()
    per_animal["delta"] = per_animal["n_after"] - per_animal["n_before"]
    add(_md(per_animal))

    add("## Fits that never settled\n")
    add(
        "A unit x model that ran to `maxiter` without settling is "
        "non-estimable for the unit that owns it -- every statistic that reads "
        "it is NaN -- so these counts are the only record of how much was "
        "dropped and from where.\n"
    )
    add(_md(session_fit_counts_table(sidecars)))

    if worker_times:
        add("## Wall clock\n")
        add(
            "One row per worker process of the batch, from the "
            "`/usr/bin/time -v` file each was run under.\n"
        )
        add(_md(worker_time_table(worker_times)))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before-dir",
        "--before",
        dest="before_dir",
        type=Path,
        default=PROCESSED_DATA_DIR / "_pre_atwell_pfc_backup_20260913",
        help=(
            "Directory holding the earlier mPFC bundles. For the cohort audit "
            "that is the pre-cohort tree; for the rerun audit it is the "
            "backup taken before the rerun."
        ),
    )
    parser.add_argument(
        "--after-dir",
        "--after",
        dest="after_dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory holding the current mPFC bundles.",
    )
    parser.add_argument(
        "--artifact-suffix",
        type=str,
        default="",
        help="Cohort suffix on the after-side bundle names ('' = canonical).",
    )
    parser.add_argument(
        "--comparison",
        type=str,
        default="auto",
        choices=["auto", "cohort", "estimator"],
        help=(
            "Which audit to write. 'estimator' compares the maximum-likelihood "
            "bundles against the penalised ones scored out of fold; 'cohort' "
            "is the earlier before/after of the stop restriction, for "
            "reproducing the audits that were written with it. 'auto' (the "
            "default) reads the after tree's sidecars and picks the one they "
            "describe."
        ),
    )
    parser.add_argument("--min-firing-rate", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="Extra paragraph to place under the Command line. Repeatable.",
    )
    parser.add_argument(
        "--worker-time",
        dest="worker_times",
        type=Path,
        nargs="*",
        default=[],
        help=(
            "``/usr/bin/time -v`` output files, one per worker process of the "
            "batch (e.g. logs/<batch>/worker*.time), to report the wall clock "
            "the bundles cost. A sibling worker*.sessions file, where one "
            "exists, gives that worker's session count."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the report here instead of stdout.",
    )
    args = parser.parse_args()

    sidecars = read_sidecars(args.after_dir, args.artifact_suffix)
    comparison = args.comparison
    if comparison == "auto":
        # The bundles state which estimator produced them, so the audit does
        # not have to be told: an after tree that names the penalised
        # estimator is an estimator change, not another cut of the cohort.
        stated = set(sidecars.get("estimator", pd.Series(dtype=object)).dropna())
        comparison = "estimator" if stated == {"ridge_poisson_glm"} else "cohort"

    # Each tree is read as the estimator that produced it. The two generations
    # carry different statistics under the same file names, so reading one as
    # the other would report a likelihood-ratio p-value as a held-out one.
    before, _, _ = load_population_data(
        args.before_dir, min_firing_rate=args.min_firing_rate, inference="legacy"
    )
    after, _, _ = load_population_data(
        args.after_dir,
        min_firing_rate=args.min_firing_rate,
        artifact_suffix=args.artifact_suffix,
        inference="heldout" if comparison == "estimator" else "legacy",
    )
    common = {
        "sidecars": sidecars,
        "before": before,
        "after": after,
        "before_gate": gate_counts(args.before_dir, args.min_firing_rate),
        "after_gate": gate_counts(
            args.after_dir,
            args.min_firing_rate,
            args.artifact_suffix,
            inference="heldout" if comparison == "estimator" else "legacy",
        ),
        "before_dir": args.before_dir,
        "after_dir": args.after_dir,
        "min_firing_rate": args.min_firing_rate,
        "alpha": args.alpha,
        "notes": args.note,
        "command": _invocation(),
        "artifact_suffix": args.artifact_suffix,
        "worker_times": args.worker_times,
    }
    report = (
        build_estimator_report(**common)
        if comparison == "estimator"
        else build_report(
            before_sidecars=try_read_sidecars(args.before_dir), **common
        )
    )
    if args.out is None:
        print(report)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
