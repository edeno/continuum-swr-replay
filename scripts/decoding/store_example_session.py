"""Store the recording of one epoch that a figure draws beside its decode.

Figure 2's panels a and b draw an example epoch's ripple trace and its HPC
tetrodes' spikes beside its decode; the decode frame, the decode's results and
its environment (the track graph, its edge order and spacing) are stored
outputs already, so only these two come from the lab database. This reads them
once, with ``load_data`` (the ripple trace of the SWR source: the Kay consensus
over the tetrodes rule R-T keeps), and writes them whole, the epoch's full
time, to ``Processed-Data/example_session/<epoch_id>/``: ``ripple.parquet``
(``time``, ``zscore``) and ``spikes.parquet`` (``tetrode``, ``time``, the
tetrodes in ``load_data``'s order), with a ``manifest.json`` that records the
commit, the epoch's decode frame's sha256 and the SWR source (the SWR inventory
run and the artifact run). The lab database is only read.

Usage::

    uv run python scripts/decoding/store_example_session.py
    uv run python scripts/decoding/store_example_session.py \\
        --nwb-file-name wilbur20210329_.nwb --epoch 02_r1 --output-dir /tmp/check
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.detector_comparison import inventory_run
from continuum_swr_replay.analysis.event_definition.swr_source import (
    ripple_power_inputs,
    swr_epoch_id,
    swr_source_name,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.decoding.identifiers import format_epoch_identifier
from continuum_swr_replay.parameters import EXAMPLE_EPOCH, EXAMPLE_NWB_FILE
from continuum_swr_replay.paths import EXAMPLE_SESSION_DIR, processed_data_dir
from continuum_swr_replay.runs.io import sha256_file, write_json
from continuum_swr_replay.runs.provenance import described_commit, produced_by
from continuum_swr_replay.tables.io import write_atomically


def session_tables(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The ripple trace and the HPC spikes of ``load_data``'s result, as stored.

    Returns
    -------
    ripple : pd.DataFrame
        ``time`` (s) and ``zscore``: ``zscored_ripple_consensus_trace``.
    spikes : pd.DataFrame
        ``tetrode`` (its place in ``spike_times["HPC"]``) and ``time`` (s).
    """
    trace = data["zscored_ripple_consensus_trace"]
    trace = trace.iloc[:, 0] if trace.ndim > 1 else trace
    ripple = pd.DataFrame(
        {"time": trace.index.to_numpy(dtype=float), "zscore": trace.to_numpy()}
    )
    tetrodes = data["spike_times"]["HPC"]
    spikes = pd.DataFrame(
        {
            "tetrode": np.concatenate(
                [
                    np.full(len(times), k, dtype=np.int64)
                    for k, times in enumerate(tetrodes)
                ]
            ),
            "time": np.concatenate(
                [np.asarray(times, dtype=float) for times in tetrodes]
            ),
        }
    )
    return ripple, spikes


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nwb-file-name", default=EXAMPLE_NWB_FILE)
    parser.add_argument("--epoch", default=EXAMPLE_EPOCH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=f"default: Processed-Data/{EXAMPLE_SESSION_DIR}/<epoch_id>",
    )
    args = parser.parse_args(argv)
    identifier = format_epoch_identifier(args.nwb_file_name, args.epoch, "HPC")
    data = load_data(
        args.nwb_file_name,
        args.epoch,
        ripple_power=ripple_power_inputs(swr_epoch_id(args.nwb_file_name, args.epoch)),
    )
    ripple, spikes = session_tables(data)
    out_dir = args.output_dir or processed_data_dir() / EXAMPLE_SESSION_DIR / identifier
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (("ripple.parquet", ripple), ("spikes.parquet", spikes)):
        write_atomically(
            out_dir / name, lambda path, t=table: t.to_parquet(path, index=False)
        )
    write_json(
        out_dir / "manifest.json",
        {
            "produced_by": produced_by(__file__),
            "commit": described_commit(),
            "nwb_file_name": args.nwb_file_name,
            "epoch": args.epoch,
            "epoch_id": identifier,
            "decode_sha256": sha256_file(inventory_run.decode_path(identifier)),
            "swr_source": swr_source_name(),
            "n_ripple_samples": len(ripple),
            "n_tetrodes": len(data["spike_times"]["HPC"]),
            "n_spikes": len(spikes),
        },
        indent=1,
        final_newline=True,
    )
    print(f"wrote {identifier} to {out_dir}")


if __name__ == "__main__":
    main()
