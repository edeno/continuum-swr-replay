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

from continuum_swr_replay.analysis.pfc.load_population import (
    load_population_data,
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


def _commit_subject(git_hash: str) -> str:
    """One-line subject of ``git_hash``, or an empty string if unavailable."""
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
            "git_hash": sidecar.get("git_hash", "")[:7],
            "written_at": sidecar.get("written_at", "")[:19],
            **counts,
            **{f"align_{k}": v for k, v in alignment.items()},
        }
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No *{stem}.provenance.json sidecars in {data_dir}")
    frame = pd.DataFrame(rows).sort_values("session").reset_index(drop=True)
    # Through the shared parser, so ``j16`` sessions resolve to "j16" and not
    # to "j1" (the leading alphabetic run) or "j" (a split on the first digit).
    frame.insert(
        1,
        "animal",
        frame["nwb_file_name"].map(lambda name: parse_nwb_filename(name)[0]),
    )
    return frame


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
    recorded = bins.notna()
    trimmed = bins.fillna(0) > 0
    seconds = sidecars["align_dropped_seconds_results"].fillna(0)
    position_rows = sidecars["align_n_position_rows_dropped"].fillna(0)
    worst = sidecars.loc[seconds.idxmax()]
    return (
        f"{int(recorded.sum())} of {len(sidecars)} sidecars record the "
        f"decode/position alignment step, and every one of those "
        f"{int(trimmed.sum())} session(s) needed a trim: "
        f"{int(bins.fillna(0).sum()):,} decode bins "
        f"({seconds.sum():.1f} s) dropped in total, against "
        f"{int(position_rows.sum()):,} position rows -- the decode always "
        "outlives the position frame, never the reverse. The largest single "
        f"trim is {seconds.max():.1f} s in `{worst['session']}`. The remaining "
        f"{int((~recorded).sum())} session(s) were fit before the step existed, "
        "by code whose strict 1-to-1 validator refused any mismatch at all, so "
        "their axes already agreed exactly and the step would be the identity "
        "on them.\n"
    )


def gate_counts(data_dir: Path, min_firing_rate: float, suffix: str = "") -> dict:
    """How many neurons survive the loader, and the firing-rate gate.

    The gate is applied to whichever rate column the bundle carries under
    ``immobility_avg_firing_rate``. For a cohort-restricted bundle that is the
    mean rate over the bins fit; the pre-cohort bundles carry the same name for
    the rate over every immobile bin. Both are counted here when the bundle has
    both, so the two sides can be compared on one quantity as well as on the
    shipped one.
    """
    frame, _, _ = load_population_data(
        data_dir, min_firing_rate=0.0, artifact_suffix=suffix
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
) -> str:
    """Assemble the audit markdown."""
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
    add(
        f"- before: `{before_dir}` -- {before_gate['n_neurons_with_fit']} neurons "
        "with a usable fit, no cohort restriction, no provenance sidecars "
        "(these bundles predate them).\n"
        f"- after: `{after_dir}` -- {len(sidecars)} bundles, every one "
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
    provenance = (
        sidecars.groupby("git_hash")
        .agg(
            n_sessions=("session", "size"),
            first_written=("written_at", "min"),
            last_written=("written_at", "max"),
        )
        .reset_index()
        .sort_values("first_written")
    )
    provenance["subject"] = provenance["git_hash"].map(_commit_subject)
    add(
        "The sessions were fit in several passes and each sidecar records the "
        "repository HEAD **at the moment it was written**, so a bundle's "
        "`git_hash` can be a commit later than the code the worker had already "
        "imported:\n"
    )
    add(_md(provenance))

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
    add(_md(population_comparison(before, after, alpha)))

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before-dir",
        type=Path,
        default=PROCESSED_DATA_DIR / "_pre_atwell_pfc_backup_20260913",
        help="Directory holding the pre-cohort mPFC bundles.",
    )
    parser.add_argument(
        "--after-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory holding the cohort-restricted mPFC bundles.",
    )
    parser.add_argument(
        "--artifact-suffix",
        type=str,
        default="",
        help="Cohort suffix on the after-side bundle names ('' = canonical).",
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
        "--out",
        type=Path,
        default=None,
        help="Write the report here instead of stdout.",
    )
    args = parser.parse_args()

    sidecars = read_sidecars(args.after_dir, args.artifact_suffix)
    before, _, _ = load_population_data(
        args.before_dir, min_firing_rate=args.min_firing_rate
    )
    after, _, _ = load_population_data(
        args.after_dir,
        min_firing_rate=args.min_firing_rate,
        artifact_suffix=args.artifact_suffix,
    )
    report = build_report(
        sidecars=sidecars,
        before=before,
        after=after,
        before_gate=gate_counts(args.before_dir, args.min_firing_rate),
        after_gate=gate_counts(
            args.after_dir, args.min_firing_rate, args.artifact_suffix
        ),
        before_dir=args.before_dir,
        after_dir=args.after_dir,
        min_firing_rate=args.min_firing_rate,
        alpha=args.alpha,
        notes=args.note,
        command=_invocation(),
        artifact_suffix=args.artifact_suffix,
    )
    if args.out is None:
        print(report)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
