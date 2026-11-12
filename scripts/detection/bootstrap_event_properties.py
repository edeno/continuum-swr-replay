"""Bootstrap the SWR against non-SWR property contrasts: class medians, their difference, and the duration-matched difference.

For each Part 2 event property (``reported.events`` ``property_values`` and the
two added ones, start distance and arms visited), over the immobile non-local
events of the canonical event table: the median of each class, the SWR minus
non-SWR difference, and the duration-matched difference (within ten quantile
strata of the pooled event durations in whole decode samples, averaged by
stratum size; ``analysis.hippocampus.property_contrast``), each with a 95 %
interval from a nested bootstrap (animals, then epochs within animal, then
events within epoch; fixed seed): the estimate plus or minus a small-sample t
correction over the animals times the replicates' standard deviation. Values are on the scale Part 2 reports
(duration in ms).

Writes ``Processed-Data/event_property_bootstrap/``: ``replicates.parquet``
(every replicate), ``intervals.csv`` (estimates and intervals, with each
column's count of replicates that have no estimate and the count of replicates
whose matched difference left out a stratum; more than 1 % without an estimate
raises) and ``manifest.json``.

Usage::

    uv run python scripts/detection/bootstrap_event_properties.py \\
        [--n-replicates 1000] [--properties duration] [--out-name ...]
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.hippocampus.property_contrast import (
    DECODE_SAMPLE_S,
    MAX_NAN_REPLICATE_SHARE,
    bootstrap_contrasts,
    bootstrap_intervals,
)
from continuum_swr_replay.paths import (
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.io import files_sha256, sha256_file
from continuum_swr_replay.runs.provenance import described_commit, produced_by

SEED = 20261028
N_STRATA = 10
MIN_PER_CLASS = 20
START_DIR = "event_start_rest_theta"

#: property word (as ``reported.events`` names it) -> (column, scale)
PROPERTIES = {
    "duration": ("event_duration_s", 1000.0),
    "extent": ("spatial_extent_cm", 1.0),
    "speed": ("replay_speed_median_cm_s", 1.0),
    "peak": ("posterior_peak_mean", 1.0),
    "hpd": ("hpd_coverage_cm", 1.0),
    "pause": ("pause_fraction", 1.0),
    "max_step": ("step_distance_max_cm", 1.0),
    "reversals": ("n_reversals", 1.0),
    "multimodal": ("multimodal_fraction", 1.0),
    "start_distance": ("start_distance_cm", 1.0),
    "arms": ("n_unique_arms", 1.0),
}


def load_events() -> tuple[pd.DataFrame, list[Path]]:
    events = pd.read_parquet(table_path("event_table", replay_content_dir()))
    paths = sorted((processed_data_dir() / START_DIR).glob("*.parquet"))
    start = pd.concat(
        (pd.read_parquet(p, columns=["event_id", "start_distance_cm"]) for p in paths),
        ignore_index=True,
    )
    events = events.merge(start, on="event_id", how="left", validate="1:1")
    if events.start_distance_cm.isna().all():
        raise ValueError(f"no start distance in {processed_data_dir() / START_DIR}")
    events["session"] = (
        events.animal + events.session_date.astype(str) + "_" + events.epoch
    )
    return events, paths


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-replicates", type=int, default=1000)
    parser.add_argument("--properties", nargs="*", default=list(PROPERTIES))
    parser.add_argument("--out-name", default="event_property_bootstrap")
    args = parser.parse_args(argv)
    events, start_paths = load_events()
    values = {
        word: events[PROPERTIES[word][0]].to_numpy(dtype=float) * PROPERTIES[word][1]
        for word in args.properties
    }
    began = time.process_time()
    replicates = bootstrap_contrasts(
        values,
        events.is_swr.to_numpy(dtype=bool),
        events.animal.to_numpy(),
        events.session.to_numpy(),
        events.event_duration_s.to_numpy(dtype=float),
        n_replicates=args.n_replicates,
        seed=SEED,
        n_strata=N_STRATA,
        min_per_class=MIN_PER_CLASS,
    )
    cpu_s = time.process_time() - began
    intervals = bootstrap_intervals(replicates, events.animal.nunique())
    folder = processed_data_dir() / args.out_name
    folder.mkdir(parents=True, exist_ok=True)
    replicates.to_parquet(folder / "replicates.parquet", index=False)
    intervals.to_csv(folder / "intervals.csv")
    event_table = table_path("event_table", replay_content_dir())
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "produced_by": produced_by(__file__),
                "commit": described_commit(),
                "event_table": {
                    "path": str(event_table),
                    "sha256": sha256_file(event_table),
                },
                "start_distance": {
                    "folder": START_DIR,
                    "n_files": len(start_paths),
                    "sha256": files_sha256(start_paths),
                },
                "n_events": len(events),
                "n_replicates": args.n_replicates,
                "seed": SEED,
                "levels": "animals, then epochs within animal, then events within "
                "epoch, each with replacement",
                "duration_strata": f"{N_STRATA} quantiles of the pooled event durations "
                f"in whole {DECODE_SAMPLE_S * 1e3:g} ms decode samples; a stratum with "
                f"fewer than {MIN_PER_CLASS} events of a class is left out, and a "
                "replicate drawing no event of a class in a stratum leaves it out "
                "(counted in intervals.csv)",
                "max_nan_replicate_share": MAX_NAN_REPLICATE_SHARE,
                "interval": "95 %: the estimate +/- t(0.975, n_animals - 1) "
                "sqrt(n_animals / (n_animals - 1)) SD(replicates)",
                "n_animals": int(events.animal.nunique()),
                "properties": {w: list(PROPERTIES[w]) for w in args.properties},
                "cpu_seconds": cpu_s,
            },
            indent=1,
        )
        + "\n"
    )
    print(intervals.to_string())
    print(
        f"{args.n_replicates} replicates x {len(values)} properties: {cpu_s:.1f} CPU s"
    )


if __name__ == "__main__":
    main()
