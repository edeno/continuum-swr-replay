#!/usr/bin/env python
"""Rebuild ContFrag tables while reusing immutable posterior aggregates.

Behavioral clocks, locations, histories, event boundaries, hard counts, and
q-value attachments are recomputed from their canonical inputs.  Only values
that require the expensive conditional posterior are carried from a preserved
source bundle, after exact event/candidate-key validation.  This makes the
dependency boundary explicit and avoids re-reading every session's large
posterior file when the decoder results themselves have not changed.

Command-line front end of
:func:`continuum_swr_replay.analysis.contfrag.prepare_data.refresh_contfrag_behavioral_covariates`.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from continuum_swr_replay.analysis.contfrag.prepare_data import (
    refresh_contfrag_behavioral_covariates,
)
from continuum_swr_replay.data_loaders.epoch_register import analysed_epoch_ids


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replay-content-dir", required=True, type=Path)
    parser.add_argument("--event-source", required=True, choices=("SWR", "MUA"))
    parser.add_argument("--brain-area", required=True, choices=("HPC", "mPFC", "OFC"))
    parser.add_argument(
        "--expected-sessions",
        type=int,
        default=len(analysed_epoch_ids()),
        help="Epochs (sessions) the tables must hold; default: the epoch register's "
        "analysed epochs (HPC cohort).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    refresh_contfrag_behavioral_covariates(
        source_dir=args.source_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        replay_content_dir=args.replay_content_dir.resolve(),
        event_source=args.event_source,
        brain_area=args.brain_area,
        expected_sessions=args.expected_sessions,
    )


if __name__ == "__main__":
    main()
