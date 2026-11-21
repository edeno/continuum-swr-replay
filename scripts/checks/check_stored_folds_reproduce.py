#!/usr/bin/env python
"""Check that the held-out runs' stored folds are the fold rule's folds.

Two results, from one command:

(a) For every epoch of the non-local restriction run, the folds that
    ``tetrode_holdout.epoch_folds`` makes from the session cache's trials and
    the stored per-bin ``time`` column equal the stored ``block`` column, and
    their counts of segments and groups equal the epoch's record in the run's
    manifest. So the stored folds are those of ``analysis/regression/folds.py``
    applied to the current trials, and the records describe them.
(b) For every epoch, the held-out tetrode run's manifest records the same
    ``n_segments``, ``n_groups``, ``trial_bin_share`` and ``margin_bins`` as
    the restriction run's: the two runs held out the same folds.

It only reads: the ``time`` and ``block`` columns of each epoch's
``<epoch>_nonlocal_restriction_bins.parquet``, the two runs' ``manifest.json``
and the trials of each epoch's session-cache entry, all under
``paths.processed_data_dir()``. Nothing is read from the lab's database and
nothing is written. It prints one summary line per result and exits non-zero
when an epoch differs or cannot be read::

    python scripts/checks/check_stored_folds_reproduce.py --workers 8
    python scripts/checks/check_stored_folds_reproduce.py --epochs <epoch_id> ...
"""

from __future__ import annotations

import argparse
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_evidence import tetrode_holdout
from continuum_swr_replay.analysis.event_tables.prepare_data import (
    behavioral_context_cache_path,
)
from continuum_swr_replay.paths import (
    EVENT_NONLOCAL_RESTRICTION_DIR,
    EVENT_TETRODE_HOLDOUT_DIR,
    NONLOCAL_RESTRICTION_BINS_NAME,
    processed_data_dir,
)
from continuum_swr_replay.runs.io import output_path, read_manifest

RECORD_KEYS = ("n_segments", "n_groups", "trial_bin_share", "margin_bins")
"""The fold records the two runs must share."""


def epoch_folds_check(epoch_id: str) -> tuple[str, str | None, dict]:
    """The folds of one epoch from the cache's trials, beside the stored ones.

    Returns
    -------
    epoch_id : str
    error : str or None
        Why the epoch could not be checked.
    result : dict
        ``n_differ`` (bins whose fold differs), ``n_segments``, ``n_groups``.
    """
    try:
        bins = pd.read_parquet(
            output_path(
                epoch_id,
                processed_data_dir() / EVENT_NONLOCAL_RESTRICTION_DIR,
                NONLOCAL_RESTRICTION_BINS_NAME,
            ),
            columns=["time", "block"],
        )
        with open(behavioral_context_cache_path({"session_id": epoch_id}), "rb") as f:
            trials = pickle.load(f)["trials"]
        folds, _, counts = tetrode_holdout.epoch_folds(
            bins["time"].to_numpy(dtype=float), trials
        )
    except Exception as error:  # reported per epoch, not raised
        return epoch_id, repr(error)[:200], {}
    stored = bins["block"].to_numpy()
    n_differ = (
        int(np.sum(np.asarray(folds) != stored)) if folds.shape == stored.shape else -1
    )
    return epoch_id, None, {"n_differ": n_differ, **counts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=1, help="processes")
    parser.add_argument("--epochs", nargs="+", help="check only these epoch ids")
    args = parser.parse_args(argv)

    root = processed_data_dir()
    restriction = read_manifest(root / EVENT_NONLOCAL_RESTRICTION_DIR)["epochs"]
    holdout = read_manifest(root / EVENT_TETRODE_HOLDOUT_DIR)["epochs"]
    epochs = sorted(args.epochs if args.epochs else restriction)

    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(epoch_folds_check, epochs))
    else:
        results = [epoch_folds_check(e) for e in epochs]
    errors = [(e, error) for e, error, _ in results if error]
    folds_differ = [(e, r) for e, error, r in results if not error and r["n_differ"]]
    counts_differ = [
        (e, k, r[k], restriction[e].get(k))
        for e, error, r in results
        if not error and e in restriction
        for k in ("n_segments", "n_groups")
        if r[k] != restriction[e].get(k)
    ]
    errors += [
        (e, "not in the restriction run's manifest")
        for e in epochs
        if e not in restriction
    ]
    n_equal = len(epochs) - len(
        {e for e, _ in errors}
        | {e for e, _ in folds_differ}
        | {d[0] for d in counts_differ}
    )
    print(
        f"(a) epochs: {len(epochs)} | stored folds and their counts reproduced "
        f"from the session cache's trials: {n_equal} | folds differ: "
        f"{len(folds_differ)} | counts differ: {len({d[0] for d in counts_differ})} "
        f"| errors: {len(errors)}"
    )

    records_differ = [
        (e, k, holdout.get(e, {}).get(k), restriction.get(e, {}).get(k))
        for e in epochs
        for k in RECORD_KEYS
        if e not in holdout
        or e not in restriction
        or holdout[e].get(k) != restriction[e].get(k)
    ]
    n_differ = len({d[0] for d in records_differ})
    print(
        f"(b) epochs: {len(epochs)} | the held-out tetrode run's fold records "
        f"equal the restriction run's: {len(epochs) - n_differ} | differ: {n_differ}"
    )
    for e, error in errors[:10]:
        print("ERROR", e, error)
    for e, r in folds_differ[:10]:
        print("FOLDS DIFFER", e, f"{r['n_differ']} bins")
    for d in (counts_differ + records_differ)[:10]:
        print("RECORD DIFFERS", *d)
    return 1 if errors or folds_differ or counts_differ or records_differ else 0


if __name__ == "__main__":
    sys.exit(main())
