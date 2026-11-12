#!/usr/bin/env python
"""Sampled examples from each cell of one inventory comparison run.

Reads the ``cells`` and ``inventory`` tables that
``scripts/detection/run_inventory_comparison.py`` wrote for the run's ``ok`` epochs
and draws events from each cell by :data:`SAMPLING_RULE`. Design:
``docs/design/replay_detection_comparison/03-inventory-comparison.md``
(Reporting, item 5).

The cells are the runner's (``CELLS``): each detector's exactly-one events,
SWRs matched by neither decoder, and sequences matched by no triggered
inventory. Each figure is one drawn event, its panels on one time axis
(:data:`PAD_S` either side of the event):

- the spikes of each tetrode, from the decode's marks;
- the state-space posterior, from the decode's ``results.nc``, summed over the
  states that have a position, with the animal's linear position;
- the memoryless window posteriors, recomputed for the plotted windows with
  ``memoryless_sequence``'s functions on the decode's encoding model. Their
  per-window centers, spreads, jumps and gates, and the dispersion and mean
  spread of every stored candidate inside the plotted windows, are checked
  against the run's ``{epoch_id}_memoryless_candidates.pkl``
  (``recompute_windows``); a figure whose recomputation disagrees is not
  written, and its record in the manifest says why;
- the gate masks: non-local probability over 0.8, the memoryless per-window
  gates as stored in the candidates file, and speed below 4 cm/s; excluded time
  is shaded in every panel;
- the speed;
- the stored ripple band of the tetrodes the SWR source kept;
- every inventory's events as bars: compared events solid, moving events
  light, events set aside outlined.

The draw is ``analysis.inventory_comparison.selection.draw_examples``; an
epoch's inputs and the recomputation are
``analysis.inventory_comparison.example_windows``'s, run on a GPU under the
memoryless run's XLA flags (``runs.device.use_memoryless_device``); the figure
is ``visualization.inventory_examples``'s. ``generate_event_feature_examples.py``
shares all four.

Figures, ``captions.md`` and ``manifest.json`` (the rule, the seed, every draw
and its memoryless check) go to
``paper/figures/inventory_comparison/<run id>/examples/``. Every caption is
marked "pending the Phase 2 pilot checks": every figure shows a memoryless
inventory.

Usage:
    python scripts/figures/generate_inventory_examples.py --run-id <inventory run id>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from continuum_swr_replay.analysis.inventory_comparison.example_windows import (  # noqa: E402
    CENTER_TIE_RTOL,
    SPREAD_RTOL,
    EpochInputs,
    load_epoch,
)
from continuum_swr_replay.analysis.inventory_comparison.selection import (  # noqa: E402
    EXAMPLE_SEED,
    N_PER_SIDE,
    draw_examples,
)
from continuum_swr_replay.parameters import FIGURE_DIR  # noqa: E402
from continuum_swr_replay.paths import (  # noqa: E402
    inventory_comparison_root,
    memoryless_root,
)
from continuum_swr_replay.runs.cli import parse_run_id  # noqa: E402
from continuum_swr_replay.runs.device import (  # noqa: E402
    recompute_device,
    use_memoryless_device,
)
from continuum_swr_replay.runs.io import (  # noqa: E402
    json_ready,
    ok_epochs,
    output_path,
    read_manifest,
)
from continuum_swr_replay.runs.provenance import produced_by  # noqa: E402
from continuum_swr_replay.tables.io import read_table  # noqa: E402
from continuum_swr_replay.visualization.inventory_examples import (  # noqa: E402
    PAD_S,
    PANELS_CAPTION,
    PENDING,
    draw_example,
    figure_name,
)

PRODUCED_BY = produced_by(__file__)
FIGURE_ROOT = FIGURE_DIR / "inventory_comparison"
EXAMPLES_DIR = "examples"
"""The examples' folder inside the run's figure folder."""

SAMPLING_RULE = (
    f"For each cell, each of its source detectors and each animal, {N_PER_SIDE} "
    "events with membership unmatched are drawn and, as counterexamples in the "
    f"other direction, {N_PER_SIDE} events of the same cell, source detector and "
    "animal with membership matched by the same set; fewer when the cell has "
    "fewer. Events with membership unknown are never drawn. The draws are made "
    "without replacement from the events sorted by session and event_key, by "
    "numpy.random.default_rng seeded from the run seed "
    f"{EXAMPLE_SEED} and the (cell, detector, animal) triple."
)
CAPTION = (
    "Sampled examples of each cell of the inventory comparison. "
    + SAMPLING_RULE
    + PANELS_CAPTION
)


def write_examples(run_dir: Path, figure_dir: Path) -> list[dict[str, Any]]:
    """Draw every example of the run and write them, captions and manifest.

    A draw whose figure cannot be made (its recomputed windows are not the
    stored ones, or its epoch's inputs cannot be read) is recorded with
    ``status`` ``not_written`` and its ``reason``, and the run goes on.

    Returns
    -------
    list of dict
        Each draw's record, as the manifest holds it: ``status`` ``written``
        or ``not_written``.
    """
    manifest = read_manifest(run_dir)
    settings = manifest["settings"]
    ok = ok_epochs(manifest)
    cells = pd.concat(
        [read_table(output_path(e, run_dir, "cells")).assign(epoch_id=e) for e in ok],
        ignore_index=True,
    )
    drawn = draw_examples(cells)
    memoryless_dir = memoryless_root() / settings["input_runs"]["memoryless"]
    run_id = run_dir.name
    records = []
    for identifier, group in drawn.groupby("epoch_id", sort=True):
        try:
            inputs: EpochInputs | None = load_epoch(
                identifier, run_dir=run_dir, memoryless_dir=memoryless_dir
            )
            epoch_error: str | None = None
        except ValueError as error:
            # the epoch's draws fail, not the run
            inputs, epoch_error = None, str(error)
        for _, row in group.iterrows():
            against = sorted(
                set(settings["cells"][row.cell]["against"]) - {row.detector}
            )
            path = figure_dir / figure_name(row)
            reason = epoch_error
            if inputs is not None:
                try:
                    record = draw_example(
                        inputs, row, against=against, path=path, run_id=run_id
                    )
                except ValueError as error:
                    reason = str(error)
            if reason is not None:
                # no figure of an earlier run stands beside this record
                path.unlink(missing_ok=True)
                record = {
                    **row.to_dict(),
                    "against": against,
                    "path": str(path),
                    "reason": reason,
                }
            status = "written" if reason is None else "not_written"
            records.append({"status": status, **record})
            print(f"{identifier}: {figure_name(row)} {status}", flush=True)
        if inputs is not None:
            inputs.results.close()
    figure_dir.mkdir(parents=True, exist_ok=True)
    (figure_dir / "manifest.json").write_text(
        json.dumps(
            json_ready(
                {
                    "produced_by": PRODUCED_BY,
                    "inventory_run_id": run_id,
                    "input_runs": settings["input_runs"],
                    "seed": EXAMPLE_SEED,
                    "n_per_side": N_PER_SIDE,
                    "sampling_rule": SAMPLING_RULE,
                    "pad_s": PAD_S,
                    "memoryless_check": {
                        "spread_rtol": SPREAD_RTOL,
                        "center_tie_rtol": CENTER_TIE_RTOL,
                        **recompute_device(),
                    },
                    "pending": PENDING,
                    "figures": records,
                }
            ),
            indent=2,
            allow_nan=False,
        )
    )
    lines = [f"# Inventory comparison examples: run {run_id}", "", CAPTION, ""]
    lines += [
        f"- `{Path(r['path']).name}`: {r['event_key']}"
        + ("" if r["status"] == "written" else f" (not written: {r['reason']})")
        for r in records
    ]
    (figure_dir / "captions.md").write_text("\n".join(lines) + "\n")
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-id",
        type=parse_run_id,
        required=True,
        help="The inventory run's folder name, under "
        "replay_detection_comparison/inventory/.",
    )
    args = parser.parse_args(argv)
    use_memoryless_device()
    run_dir = inventory_comparison_root() / args.run_id
    if not (run_dir / "manifest.json").exists():
        sys.exit(f"No inventory run manifest at {run_dir}.")
    figure_dir = FIGURE_ROOT / args.run_id / EXAMPLES_DIR
    records = write_examples(run_dir, figure_dir)
    written = sum(r["status"] == "written" for r in records)
    print(f"{written} of {len(records)} examples written -> {figure_dir}", flush=True)


if __name__ == "__main__":
    main()
