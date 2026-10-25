#!/usr/bin/env python
"""Print Stage 6's per-role cross-session FDR table for a saved bundle.

``run_stay_leave.py`` logs this table for the run it just did. An audit
needs the same table for a bundle it did *not* just fit -- a backed-up bundle
from before a cohort change, or the ``_allstops`` sensitivity bundle next to the
canonical one -- so that a before/after comparison is two readings of one
statistic rather than two implementations of it. This script is that reading:
it calls the function the runner logs,
:func:`~continuum_swr_replay.analysis.replay.learning.cross_session_role_summary`.

Usage::

    uv run python scripts/qa/summarize_stage6_fdr.py \\
        Processed-Data/replay_content/stage6_is_stay_logistic.pkl
    uv run python scripts/qa/summarize_stage6_fdr.py --csv out.csv bundle.pkl
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

# The statistic is read with the runner's own function, never a second copy.
from continuum_swr_replay.analysis.replay.learning import cross_session_role_summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bundles",
        type=Path,
        nargs="+",
        help="stage6_is_stay_logistic*.pkl file(s) to summarize.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Also write the summarized rows (with a 'bundle' column) here.",
    )
    args = parser.parse_args(argv)

    frames = []
    for bundle in args.bundles:
        coefs = pd.read_pickle(bundle)
        summary = cross_session_role_summary(coefs)
        summary.insert(0, "bundle", bundle.name)
        stratum = sorted(coefs["content_stratum"].dropna().unique())
        cohort = (
            sorted(coefs["stop_cohort"].dropna().unique())
            if "stop_cohort" in coefs
            else ["unstated"]
        )
        print(f"\n== {bundle}")
        print(f"   content_stratum={stratum}  stop_cohort={cohort}")
        print(f"   rows={len(coefs)}  converged={int((coefs['status'] == 'ok').sum())}")
        print(summary.to_string(index=False, float_format=lambda v: f"{v:+.4g}"))
        frames.append(summary)

    if args.csv is not None:
        pd.concat(frames, ignore_index=True).to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
