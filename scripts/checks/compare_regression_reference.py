#!/usr/bin/env python
"""Fit a real slice with the shared regression fitter, and compare it with a
stored fit of the same slice.

The slice: three sessions, the first analyzed session in the stop table of chimi
(20200212, 02_r1), j16 (20210706, 02_r1) and wilbur (20210326, 03_r1), SWR
events only, each fitted alone by ``sessions.fit_sessions`` at fixed log
penalties with two models:

- rate (Poisson): the in-visit 15 ms bins of ``visit_design.visit_timebins``
  over the canonical stop and event tables, with the last outcome known; a
  P-spline of the time since arrival (43 basis functions over 0 to 20 s, log
  penalty 1.77) and of the time until departure (33 over 0 to 3 s, 3.17), the
  last outcome as an unpenalized 0/1 slope, the exposure's offset and an
  intercept. These are the terms and penalties of the timing prototype of
  2026-11-23 (log penalties at about 6 edf per curve), standing in until the
  rate model's module declares its own.
- content (grouped softmax): the canonical event x arm table's events that
  start in a well visit, each event's nine candidate arms, the arm's posterior
  fraction as the outcome; the arm category (unpenalized), the current-arm and
  previous-arm indicators (each with the weak penalty, a Normal prior of SD 2)
  and a P-spline of the arm's reward fraction over its last 10 visits (6 basis
  functions over 0 to 1, log penalty 2). A stand-in until the content model's
  module declares its design.

The tables are read only, from ``$CSR_PROCESSED_DATA_DIR``; nothing is written
there. ``--write DIR`` writes the records (every field, as plain values) to
``DIR/real_slice.json`` with the commit; a ``DIR`` inside the repository or
under the processed-data root is refused before anything is read; ``--against DIR`` fits the slice again
and compares it with that file by the rule of
``_regression_reference.differences`` (a float within ``1e-11 + 1e-9 |b|``, the
rest equal), exiting 1 on a difference. Real data are not committed: the stored
file lives in a scratch folder.

Usage (from the repository root):

    CSR_PROCESSED_DATA_DIR=... python scripts/checks/compare_regression_reference.py \\
        --write /tmp/slice_before
    CSR_PROCESSED_DATA_DIR=... python scripts/checks/compare_regression_reference.py \\
        --against /tmp/slice_before
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from continuum_swr_replay.analysis.event_definition.analyzed_time import (  # noqa: E402
    excluded_intervals_by_session,
    read_swr_table,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (  # noqa: E402
    select_visit_context,
)
from continuum_swr_replay.analysis.event_tables import prepare_data  # noqa: E402
from continuum_swr_replay.analysis.regression.likelihoods import (  # noqa: E402
    GroupedSoftmax,
    Poisson,
)
from continuum_swr_replay.analysis.regression.sessions import fit_sessions  # noqa: E402
from continuum_swr_replay.analysis.regression.terms import (  # noqa: E402
    Category,
    Offset,
    Slope,
    Smooth,
    build_design,
    log_penalty_from_prior_sd,
)
from continuum_swr_replay.analysis.replay_models.visit_design import (  # noqa: E402
    visit_timebins,
)
from continuum_swr_replay.decoding.identifiers import (  # noqa: E402
    format_epoch_identifier,
    nwb_file_name,
)
from continuum_swr_replay.paths import processed_data_dir, table_path  # noqa: E402
from continuum_swr_replay.runs.device import default_jax_platforms  # noqa: E402
from continuum_swr_replay.runs.provenance import git_commit, git_dirty  # noqa: E402
from scripts.checks import _regression_reference as reference  # noqa: E402

SESSIONS = (
    ("chimi", 20200212, "02_r1"),
    ("j16", 20210706, "02_r1"),
    ("wilbur", 20210326, "03_r1"),
)
KEYS = ["animal", "session_date", "epoch"]
LAG1 = "known_reward_outcome_lag_1"
RATE_LOG_PENALTIES = (1.77, 3.17)
WEAK = log_penalty_from_prior_sd(2.0)
FILE_NAME = "real_slice.json"


def _of_slice(frame: pd.DataFrame) -> pd.DataFrame:
    keys = pd.MultiIndex.from_tuples(SESSIONS, names=KEYS)
    return frame[pd.MultiIndex.from_frame(frame[KEYS]).isin(keys)]


def rate_sessions() -> dict[tuple, pd.DataFrame]:
    """Each session's in-visit bins of SWR starts with the last outcome known."""
    stops = _of_slice(read_swr_table(table_path("stop_table")))
    events = _of_slice(read_swr_table(table_path("event_table")))
    excluded = excluded_intervals_by_session(stops)
    out = {}
    for key in SESSIONS:
        session_stops = stops[(stops[KEYS] == key).all(axis=1)]
        experiences = prepare_data.cached_port_experiences(
            format_epoch_identifier(nwb_file_name(key[0], key[1]), key[2], "HPC")
        )
        bins = select_visit_context(
            visit_timebins(
                session_stops,
                events[(events[KEYS] == key).all(axis=1)],
                experiences,
                kind="SWR",
                excluded_intervals=excluded[(excluded[KEYS] == key).all(axis=1)],
            )
        )
        bins = bins[bins[LAG1].notna()].reset_index(drop=True)
        bins["rewarded"] = bins[LAG1].astype(float)
        bins["well_visit_id"] = bins["well_visit_id"].astype("int64")
        out[tuple(str(k) for k in key)] = bins
    return out


def content_sessions() -> dict[tuple, pd.DataFrame]:
    """Each session's candidate rows of SWR events that start in a well visit."""
    rows = select_visit_context(
        _of_slice(read_swr_table(table_path("event_arm_table")))
    )
    rows = rows[rows["is_swr"]]
    return {
        tuple(str(k) for k in key): rows[(rows[KEYS] == key).all(axis=1)].reset_index(
            drop=True
        )
        for key in SESSIONS
    }


def rate_records() -> list[dict]:
    sessions = rate_sessions()
    pooled = pd.concat(sessions.values(), ignore_index=True)
    since, until = RATE_LOG_PENALTIES
    design = build_design(
        [
            Smooth("time_since_visit_start_s", 0.0, 20.0, 43, since),
            Smooth("time_until_visit_end_s", 0.0, 3.0, 33, until),
            Slope("rewarded", "none"),
            Offset("exposure"),
        ],
        pooled,
        intercept=True,
    )
    return _records(design, sessions, Poisson(), "count", "well_visit_id", "rate")


def content_records() -> list[dict]:
    sessions = content_sessions()
    pooled = pd.concat(sessions.values(), ignore_index=True)
    design = build_design(
        [
            Category("candidate_arm", tuple(range(9)), reference=0, penalty="none"),
            Slope("is_current_arm", WEAK),
            Slope("is_prev_arm", WEAK),
            Smooth("reward_fraction_candidate_arm_last10_visits", 0.0, 1.0, 6, 2.0),
        ],
        pooled,
        intercept=False,
    )
    return _records(
        design, sessions, GroupedSoftmax(), "arm_fraction", "event_id", "content"
    )


def _records(design, sessions, lik, response, unit, model) -> list[dict]:
    records = fit_sessions(
        design,
        sessions,
        lik,
        model=model,
        kind="SWR",
        response=response,
        unit=unit,
        table_sha256={},
    )
    return [asdict(r) for r in records]


def real_slice() -> dict:
    """The slice's records, as plain values."""
    return reference.plain({"rate": rate_records(), "content": content_records()})


def why_not_a_write_folder(folder: Path, repo: Path = REPO) -> str | None:
    """Why ``folder`` may not hold the slice's fits, or None when it may.

    The fits are of real sessions: they are not committed, so they may not be
    written inside the repository, nor among the canonical tables, under the
    processed-data root. Both are compared after resolving links.
    """
    folder = folder.resolve()
    for root, what in (
        (repo.resolve(), "the repository"),
        (processed_data_dir().resolve(), "the processed-data root"),
    ):
        if folder.is_relative_to(root):
            return (
                f"--write {folder} is inside {what} ({root}): the fits of real "
                "sessions go to a scratch folder outside both"
            )
    return None


def main(argv: Sequence[str] | None = None) -> int:
    default_jax_platforms("cpu")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", type=Path, help="Write the slice's fits here.")
    action.add_argument("--against", type=Path, help="Compare with the fits here.")
    args = parser.parse_args(argv)
    if args.write and (refused := why_not_a_write_folder(args.write)):
        parser.error(refused)
    fits = real_slice()
    for model, records in fits.items():
        statuses = pd.Series([r["status"] for r in records]).value_counts().to_dict()
        rows = [r["n_rows"] for r in records]
        print(f"{model}: rows {rows}, statuses {statuses}")
    if args.write:
        args.write.mkdir(parents=True, exist_ok=True)
        commit = git_commit(REPO) + ("-dirty" if git_dirty(REPO) else "")
        text = json.dumps({"commit": commit, "fits": fits}, allow_nan=False)
        (args.write / FILE_NAME).write_text(text + "\n")
        print(f"wrote {args.write / FILE_NAME} ({len(text):,} bytes)")
        return 0
    stored = json.loads((args.against / FILE_NAME).read_text())
    mismatches = reference.differences(fits, stored["fits"], "real_slice")
    print(f"against {stored['commit']}: {len(mismatches)} differences")
    print("\n".join(mismatches))
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
