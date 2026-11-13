#!/usr/bin/env python
"""Concatenate and flag every ``*_cohort_audit.csv`` under the outputs dir.

Answers "did complete-case filtering change the estimand?" at a glance: it stacks all
per-runner cohort-audit tables (content base/contrast, rate contrast/per-stop/per-bin)
and flags rows where the retained fraction ``n_modeled / n_in`` falls below a threshold
(default 0.9) or where the ``drop_notes`` profile indicates a biased drop (early-session
= mechanism A, or low-value / late = mechanism B).

Reporting only — reads the emitted CSVs, never re-fits.

Usage:
    uv run python scripts/checks/summarize_cohort_audits.py
    uv run python scripts/checks/summarize_cohort_audits.py --root Processed-Data --min-retained 0.85
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.replay_models.cohort_audit import (
    COHORT_COLUMNS,
    DROP_FIELDS,
)
from continuum_swr_replay.paths import processed_data_dir


def load_all(root: Path) -> pd.DataFrame:
    """Stack every ``*_cohort_audit.csv`` under ``root`` (recursively), tagged by source."""
    frames = []
    for path in sorted(root.rglob("*_cohort_audit.csv")):
        df = pd.read_csv(path)
        df.insert(0, "source", str(path.relative_to(root)))
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=("source", *COHORT_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def flag(df: pd.DataFrame, *, min_retained: float) -> pd.DataFrame:
    """Add ``retained_frac`` and a ``flag`` reason for low-retention / biased rows."""
    if df.empty:
        return df.assign(retained_frac=[], flag=[])
    out = df.copy()
    out["retained_frac"] = out["n_modeled"] / out["n_in"].where(out["n_in"] > 0)
    notes = out["drop_notes"].fillna("").astype(str)
    reasons = []
    for frac, note, qvals in zip(
        out["retained_frac"], notes, out["n_drop_qvals_gap"], strict=True
    ):
        r = []
        if pd.notna(frac) and frac < min_retained:
            r.append(f"low_retention({frac:.2f})")
        if int(qvals) > 0:
            r.append("qvals_gap(mechanismB)")
        if "median_tss=" in note:
            # a very small median within-stop time among dropped events = early-drop bias
            try:
                tss = float(note.split("median_tss=")[1].split(";")[0])
                if tss < 0.5:
                    r.append(f"early_drop(tss={tss:.2f})")
            except (ValueError, IndexError):
                pass
        reasons.append("; ".join(r))
    out["flag"] = reasons
    return out


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=processed_data_dir(),
        help="directory to scan recursively for *_cohort_audit.csv",
    )
    ap.add_argument(
        "--min-retained",
        type=float,
        default=0.9,
        help="flag rows whose n_modeled/n_in falls below this (default 0.9)",
    )
    ap.add_argument("--out", type=Path, default=None, help="optional combined-CSV path")
    args = ap.parse_args(argv)

    df = load_all(args.root)
    if df.empty:
        print(f"No *_cohort_audit.csv found under {args.root}", file=sys.stderr)
        return

    flagged = flag(df, min_retained=args.min_retained)
    total_drops = flagged[list(DROP_FIELDS)].sum(axis=1)
    # Reconciliation self-check across every emitted row.
    bad = flagged[flagged["n_in"] - total_drops != flagged["n_modeled"]]
    if len(bad):
        print(
            f"WARNING: {len(bad)} row(s) do not reconcile (n_in - Σdrops != n_modeled):",
            file=sys.stderr,
        )
        print(
            bad[["source", "model", "population"]].to_string(index=False),
            file=sys.stderr,
        )

    cols = [
        "source",
        "model",
        "population",
        "grain",
        "n_in",
        "n_modeled",
        "retained_frac",
        *DROP_FIELDS,
        "flag",
    ]
    with pd.option_context("display.max_rows", None, "display.width", 240):
        print(flagged[cols].round(3).to_string(index=False))

    flagged_rows = flagged[flagged["flag"] != ""]
    print(
        f"\n{len(flagged_rows)}/{len(flagged)} rows flagged "
        f"(retained < {args.min_retained} or biased/mechanism-B drop)."
    )
    if args.out is not None:
        flagged.to_csv(args.out, index=False)
        print(f"Wrote combined table -> {args.out}")


if __name__ == "__main__":
    main()
