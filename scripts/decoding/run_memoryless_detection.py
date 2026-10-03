#!/usr/bin/env python
"""Run the memoryless sequence detector over recorded HPC epochs.

The Widloski & Foster sliding-window detector, ported in
``continuum_swr_replay.analysis.hippocampus.memoryless_sequence``, run on each
epoch's saved clusterless decode: the encoding model from
``{epoch_id}_detector.pkl``, the decode grid, position, speed and SWR labels
from ``{epoch_id}_80_data_over_time.pkl``, and the spike marks read as
``load_data`` read them for the decode (``MARKS_RULE``): ``get_hpc_marks``,
then ``filter_spike_times`` over the decode's time span.
Both procedures therefore read one encoding model.
``check_marks_match_encoding`` refuses them unless every mark the encoding
model was fitted on is among them. Those are the training span's, so it cannot
see a spike the reload adds, or one dropped only during SWRs, which
``no_ripple`` training leaves out; the refresh design leaves that gap to its
Stage 8 one-commit check (``docs/design/2026-10-03-full-pipeline-refresh-design.md``).

Each epoch writes, under ``Processed-Data/replay_detection_comparison/memoryless/<run_id>/``:

- ``{epoch_id}_memoryless_events.parquet``, one row per candidate sequence;
- ``{epoch_id}_memoryless_shuffle_draws.parquet``, every shuffled draw;
- ``{epoch_id}_memoryless_observed.parquet``, what was observed: the span the
  windows cover, and the stretches where SWR labels were observed;
- ``{epoch_id}_memoryless_candidates.pkl``, the per-window statistics, gates
  and pre-merge runs that explain each rejection.

``manifest.json`` records the run's settings and, per epoch, the commit, the
time grid, a hash of every input, the marks' rule and hash, the encoding model's
settings, and the outcome. It is rewritten atomically after each epoch, so a
run that stops part way still says what it did, and a rerun under the same
run id refuses to continue with different settings, code or library
versions. An epoch that fails is recorded as ``failed`` with its error and the
run goes on to the next; the run then exits non-zero, and a rerun under the
same run id retries the failed epochs. That holds for a transient failure; after a code fix the
commit, and so the default run id, has changed, and the finished epochs are
recomputed under the new one, which is what the settings gate is for.
``run.log`` beside the manifest keeps everything printed.

The marks come from the Spyglass database unless ``--cache-dir`` already holds
this decode's from an earlier run. A cache file is named for the epoch, the
rule (``MARKS_RULE_TAG``) and the first 16 hex digits of the decode's
``_detector.pkl`` sha256, so it serves only marks read under this rule for this
decode file: one made for another decode of the epoch is never read, and
neither is the pilot's ``{epoch_id}_decode_time_marks.pkl``, made under the
April decodes' rule.

Epoch identifiers name the decode: ``chimi20200212_02_r1_HPC`` is the
canonical one (Kay's SWRs, trained on ``no_ripple``, stationary). A decode
made another way carries suffixes in the order the decode appends them:
``_contfrag``, a ripple detector (``_Karlsson``), ``_non_stationary``, and a
training type (``_all``). ``--animal`` and ``--all`` run canonical decodes
only unless ``--include-variants`` is given; ``--epoch-id`` runs whatever it
names.

Usage:
    python scripts/decoding/run_memoryless_detection.py --run-seed 20260922 \\
        --epoch-id chimi20200212_02_r1_HPC
    python scripts/decoding/run_memoryless_detection.py --run-seed 20260922 \\
        --animal chimi
    python scripts/decoding/run_memoryless_detection.py --run-seed 20260922 \\
        --animal chimi --no-shuffle      # the detector alone, much faster
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.constants import DEFAULT_NON_LOCAL_THRESHOLD
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence import (
    MemorylessSequenceConfig,
    SequenceIdentity,
    ShuffleSeed,
    load_clusterless_encoding_model,
    run_epoch,
    window_midpoints_s,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.event_table import (
    canonical_swr_intervals,
    observed_intervals,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    check_marks_match_encoding,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.shuffle import (
    DEFAULT_N_SHUFFLES,
    MINIMUM_N_SHUFFLES,
    SIGNIFICANCE_LEVEL,
)
from continuum_swr_replay.analysis.replay.fit_provenance import git_hash
from continuum_swr_replay.data_loaders.spikes import filter_spike_times, get_hpc_marks
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.table_io import sha256_file, write_table

OUTPUT_ROOT = PROCESSED_DATA_DIR / "replay_detection_comparison" / "memoryless"
DESIGN_SPEC = "docs/design/replay_detection_comparison/02-memoryless-sequence.md"
NON_LOCAL_THRESHOLD_PERCENT = round(100 * DEFAULT_NON_LOCAL_THRESHOLD)
"""The ``data_over_time`` file is named for the decode's non-local threshold."""

DEFAULT_SWR_DEFINITION = "Kay"
"""A decode run with the default ripple detector adds no suffix to its identifier."""

OTHER_RIPPLE_DETECTORS = ("Karlsson", "Yu")
"""The ripple detectors whose name a decode would append; Kay's it does not.

No decode carries ``_Yu`` yet; it is listed so the day one does, the runner
reads it as a ripple detector and not as a name no decode writes."""

DEFAULT_TRAINING_TYPE = "no_ripple"
"""The training type a decode does not append to its identifier."""

MARKS_RULE = (
    "load_data: get_hpc_marks (tetrode sort groups; coincident spikes counted) "
    "then filter_spike_times over the decode's time span"
)
"""How the marks are read, for the manifest: as ``load_data`` read them."""

MARKS_RULE_TAG = "loader"
""":data:`MARKS_RULE`'s name in a marks cache file; another rule needs another tag."""


@dataclass(frozen=True)
class EpochKey:
    """What an epoch identifier says about the decode it names."""

    epoch_id: str
    nwb_file_name: str
    epoch_name: str
    brain_area: str
    swr_definition: str
    contfrag: bool
    non_stationary: bool
    training_type: str

    @property
    def is_canonical(self) -> bool:
        """The decode the comparison uses: Kay's SWRs, no_ripple, stationary."""
        return (
            not self.contfrag
            and self.swr_definition == DEFAULT_SWR_DEFINITION
            and not self.non_stationary
            and self.training_type == DEFAULT_TRAINING_TYPE
        )


def parse_epoch_identifier(epoch_id: str) -> EpochKey:
    """Split ``{animal}{date}_{epoch}_{brain_area}`` and any decode suffixes.

    Suffixes come in the order ``run_decoding_bandit.py`` and
    ``run_decoding_contfrag.py`` append them: ``_contfrag``, a ripple detector
    other than Kay, ``_non_stationary``, and a training type other than
    ``no_ripple``. Anything else is refused rather than read as, say, a
    ripple detector called ``all``.
    """
    parts = epoch_id.split("_")
    if len(parts) < 4 or not all(parts):
        raise ValueError(
            f"{epoch_id!r} is not {{animal}}{{date}}_{{epoch}}_{{brain_area}} with "
            f"optional decode suffixes."
        )
    stem, number, replicate, brain_area, *rest = parts
    contfrag = rest[:1] == ["contfrag"]
    rest = rest[1:] if contfrag else rest
    swr_definition = DEFAULT_SWR_DEFINITION
    if rest and rest[0] in OTHER_RIPPLE_DETECTORS:
        swr_definition, rest = rest[0], rest[1:]
    non_stationary = rest[:2] == ["non", "stationary"]
    rest = rest[2:] if non_stationary else rest
    training_type = DEFAULT_TRAINING_TYPE
    if rest[:1] == ["all"]:
        training_type, rest = "all", rest[1:]
    if rest:
        raise ValueError(
            f"{epoch_id!r} ends in {'_'.join(rest)!r}, which no decode appends."
        )
    return EpochKey(
        epoch_id=epoch_id,
        nwb_file_name=f"{stem}_.nwb",
        epoch_name=f"{number}_{replicate}",
        brain_area=brain_area,
        swr_definition=swr_definition,
        contfrag=contfrag,
        non_stationary=non_stationary,
        training_type=training_type,
    )


def input_paths(epoch_id: str, root: Path) -> dict[str, Path]:
    """The two saved decode artifacts the detector reads."""
    return {
        "detector": root / f"{epoch_id}_detector.pkl",
        "data_over_time": root
        / f"{epoch_id}_{NON_LOCAL_THRESHOLD_PERCENT}_data_over_time.pkl",
    }


def find_epochs(
    animal: str | None, root: Path, *, include_variants: bool = False
) -> tuple[list[str], dict[str, str]]:
    """HPC decodes with both artifacts saved, and why every other one was left out.

    HPC only: this port reads a clusterless encoding model. A name no decode
    writes, or a missing artifact, is recorded rather than raised, so one
    stray file cannot stop a run over many epochs.

    Returns
    -------
    epochs : list of str
    skipped : dict of str to str
        Epoch identifier to the reason it was left out.
    """
    pattern = f"{animal}*_HPC*_detector.pkl" if animal else "*_HPC*_detector.pkl"
    epochs, skipped = [], {}
    for detector in sorted(root.glob(pattern)):
        epoch_id = detector.name.removesuffix("_detector.pkl")
        missing = [
            name
            for name, path in input_paths(epoch_id, root).items()
            if not path.exists()
        ]
        if missing:
            skipped[epoch_id] = f"missing {', '.join(missing)}"
            continue
        try:
            key = parse_epoch_identifier(epoch_id)
        except ValueError as error:
            skipped[epoch_id] = str(error)
            continue
        if key.brain_area != "HPC":
            skipped[epoch_id] = f"brain area {key.brain_area}, not HPC"
        elif not key.is_canonical and not include_variants:
            skipped[epoch_id] = "a decode variant; pass --include-variants to run it"
        else:
            epochs.append(epoch_id)
    return epochs, skipped


def marks_cache_path(key: EpochKey, cache_dir: Path, detector_sha256: str) -> Path:
    """Where one decode's marks are cached under :data:`MARKS_RULE`.

    The name carries the rule's tag and the first 16 hex digits of the decode's
    ``_detector.pkl`` sha256, so a cache made for another decode of the epoch,
    or under another rule, is never read.
    """
    return (
        cache_dir / f"{key.epoch_id}_{MARKS_RULE_TAG}_{detector_sha256[:16]}_marks.pkl"
    )


def load_marks(
    key: EpochKey,
    cache_dir: Path,
    span_s: tuple[float, float],
    detector_sha256: str,
) -> tuple[list[NDArray], list[NDArray]]:
    """A decode's HPC spike times and marks, read as ``load_data`` read them.

    ``get_hpc_marks``, then ``filter_spike_times`` over the decode's time span:
    ``load_data`` filters over the position grid, which is the decode's grid.
    Cached per decode file (:func:`marks_cache_path`).

    Parameters
    ----------
    key : EpochKey
        The decode.
    cache_dir : Path
        Where the marks are cached.
    span_s : tuple of float
        First and last time of the decode's grid, in seconds; spikes on either
        end are kept.
    detector_sha256 : str
        sha256 of the decode's ``_detector.pkl``, which keys the cache.

    Returns
    -------
    spike_times : list of ndarray, each shape (n_spikes,)
    spike_waveform_features : list of ndarray, each shape (n_spikes, n_features)
        One per tetrode with a spike in ``span_s``, in ``get_hpc_marks``'s order.
    """
    cache = marks_cache_path(key, cache_dir, detector_sha256)
    if cache.exists():
        cached: tuple[list[NDArray], list[NDArray]] = joblib.load(cache)
        return cached
    spike_times, spike_waveform_features = get_hpc_marks(key.nwb_file_name)
    in_span_times, in_span_features = filter_spike_times(
        {"HPC": spike_times},
        {"HPC": spike_waveform_features},
        np.asarray(span_s, dtype=np.float64),
    )
    marks = (in_span_times["HPC"], in_span_features["HPC"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(marks, cache)
    return marks


def marks_sha256(spike_times: list, spike_waveform_features: list) -> str:
    """One hash of every electrode's spike times and marks, in order."""
    digest = hashlib.sha256()
    for times, features in zip(spike_times, spike_waveform_features, strict=True):
        digest.update(np.ascontiguousarray(times, dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(features, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _one_value(frame: pd.DataFrame, column: str) -> Any:
    values = pd.unique(frame[column])
    if len(values) != 1:
        raise ValueError(
            f"data_over_time holds {len(values)} values of {column!r}, not one."
        )
    return values[0]


def output_paths(epoch_id: str, output_dir: Path) -> dict[str, Path]:
    return {
        "events": output_dir / f"{epoch_id}_memoryless_events.parquet",
        "shuffle_draws": output_dir / f"{epoch_id}_memoryless_shuffle_draws.parquet",
        "observed": output_dir / f"{epoch_id}_memoryless_observed.parquet",
        "candidates": output_dir / f"{epoch_id}_memoryless_candidates.pkl",
    }


def observation_table(
    time: np.ndarray, swr_observed: np.ndarray, config: MemorylessSequenceConfig
) -> pd.DataFrame:
    """The stretches this epoch observed, independent of what it detected.

    ``decoded_windows`` spans the first and last window midpoint the grid
    supports, the final row, which no spike can reach, excluded;
    ``swr_labels`` are the stretches where SWR labels were observed.
    """
    midpoints = window_midpoints_s(time[:-1], config)
    rows = []
    if midpoints.size:
        rows.append(("decoded_windows", midpoints[0], midpoints[-1]))
    rows += [("swr_labels", start, end) for start, end in swr_observed]
    return pd.DataFrame(rows, columns=["kind", "start_s", "end_s"]).astype(
        {"kind": object, "start_s": np.float64, "end_s": np.float64}
    )


def run_one(
    epoch_id: str,
    *,
    root: Path,
    output_dir: Path,
    cache_dir: Path,
    run_id: str,
    run_seed: int,
    shuffle: bool,
    n_shuffles: int,
    config: MemorylessSequenceConfig,
    provenance: dict,
    marks_loader: Callable[
        [EpochKey, Path, tuple[float, float], str],
        tuple[list[NDArray], list[NDArray]],
    ] = load_marks,
) -> dict:
    """Detect, test and save one epoch; return what the manifest records for it.

    Raises
    ------
    FileNotFoundError
        An input is missing. A run over many epochs never asks for one:
        :func:`find_epochs` leaves such epochs out and records why.
    ValueError
        The marks are not the ones the encoding model was fitted on, or the
        grid, the identifier or ``data_over_time`` is not what the detector
        needs.
    """
    key = parse_epoch_identifier(epoch_id)
    inputs = input_paths(epoch_id, root)
    missing = [name for name, path in inputs.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{epoch_id}: missing {', '.join(missing)}.")
    input_sha256 = {name: sha256_file(path) for name, path in inputs.items()}

    encoding_model = load_clusterless_encoding_model(inputs["detector"])
    data_over_time = pd.read_pickle(inputs["data_over_time"])
    if "time" in data_over_time.columns:
        data_over_time = data_over_time.set_index("time")
    identity = SequenceIdentity(
        animal=str(_one_value(data_over_time, "animal")),
        day=int(_one_value(data_over_time, "day")),
        epoch=str(_one_value(data_over_time, "epoch")),
        brain_area=key.brain_area,
        epoch_id=epoch_id,
        run_id=run_id,
    )
    time = data_over_time.index.to_numpy(dtype=np.float64)
    spike_times, spike_waveform_features = marks_loader(
        key, cache_dir, (float(time[0]), float(time[-1])), input_sha256["detector"]
    )
    check_marks_match_encoding(encoding_model, spike_times, spike_waveform_features)

    speed = data_over_time["head_speed"].to_numpy(dtype=np.float64)
    midpoint_s = window_midpoints_s(time[:-1], config)
    # SWR_event is an integer and never NaN; SWR_zscore is NaN where the LFP is missing
    swr_observed = observed_intervals(data_over_time, "SWR_zscore")
    detection = run_epoch(
        time,
        data_over_time["linear_position"].to_numpy(dtype=np.float64),
        speed,
        spike_times,
        spike_waveform_features,
        encoding_model,
        identity,
        swr_intervals=canonical_swr_intervals(
            data_over_time, speed_threshold=config.speed_threshold_cm_per_s
        ),
        swr_observed_intervals=swr_observed,
        swr_definition=key.swr_definition,
        seed=ShuffleSeed(run=run_seed, session=epoch_id) if shuffle else None,
        config=config,
        n_shuffles=n_shuffles,
    )

    outputs = output_paths(epoch_id, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observed = observation_table(time, swr_observed, config)
    for frame in (detection.table, detection.shuffle_draws, observed):
        frame.attrs = provenance | {"epoch_id": epoch_id}
    write_table(detection.table, outputs["events"])
    write_table(detection.shuffle_draws, outputs["shuffle_draws"])
    write_table(observed, outputs["observed"])
    joblib.dump(detection.candidates, outputs["candidates"])

    table = detection.table
    gates = detection.candidates.decisions.gates
    return {
        "status": "ok",
        "git_commit": provenance["git_commit"],
        "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "decode": {
            "swr_definition": key.swr_definition,
            "contfrag": key.contfrag,
            "non_stationary": key.non_stationary,
            "training_type": key.training_type,
        },
        "time_grid": {
            "start_s": float(time[0]),
            "end_s": float(time[-1]),
            "n_samples": int(time.size),
            "sampling_frequency": config.sampling_frequency,
        },
        "encoding": {
            "n_electrodes": len(encoding_model["encoding_positions"]),
            "n_encoding_spikes": [
                int(np.asarray(p).shape[0])
                for p in encoding_model["encoding_positions"]
            ],
            "position_std": np.asarray(encoding_model["position_std"]).tolist(),
            "waveform_std": np.asarray(encoding_model["waveform_std"]).tolist(),
            "block_size": int(encoding_model["block_size"]),
        },
        "n_windows": int(gates.speed_eligible.size),
        # a window whose speed was NaN is rejected as not still; how many
        # says whether a quiet epoch was quiet or untracked
        "n_windows_speed_unknown": int(
            np.count_nonzero(~np.isfinite(np.interp(midpoint_s, time, speed)))
        ),
        "n_candidates": int(len(table)),
        "n_replay": int(table.is_replay.sum()),
        "n_swr_unknown": int((table.overlap_class == "unknown").sum()),
        "n_shuffled": int((table.n_shuffles > 0).sum()),
        "n_shuffle_untestable": int(
            ((table.n_shuffles > 0) & ~table.shuffle_testable).sum()
        ),
        "n_shuffle_pass": int(table.shuffle_pass.sum()),
        "inputs": {
            name: {"path": str(path), "sha256": input_sha256[name]}
            for name, path in inputs.items()
        },
        "marks": {
            "rule": MARKS_RULE,
            "detector_sha256": input_sha256["detector"],
            "cache": marks_cache_path(key, cache_dir, input_sha256["detector"]).name,
            "sha256": marks_sha256(spike_times, spike_waveform_features),
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }


def run_settings(
    run_seed: int,
    shuffle: bool,
    n_shuffles: int,
    config: MemorylessSequenceConfig,
    git_commit: str,
) -> dict:
    """What every epoch of one run must share; a resume must match it."""
    return {
        "git_commit": git_commit,
        "libraries": {
            name: version(name) for name in ("non_local_detector", "numpy", "pandas")
        },
        "config": asdict(config),
        "shuffle": {
            "run": shuffle,
            "definition": "encoding-pair: within each electrode, encoding "
            "positions permuted against their waveform marks",
            "n_shuffles": n_shuffles if shuffle else 0,
            "run_seed": run_seed,
            "seeded_by": "run seed, epoch_id, candidate's first window, draw, electrode",
            "tested": "every candidate meeting the replay criteria, as the reference",
            "statistics": "mean posterior spread and mean forward jump over every "
            "window; the jump of a pair no spike entered or left set aside, which "
            "is the reference's jumps(jumps==0)=nan on a continuous center; NaN "
            "skipped",
            "p_value": "(1 + count(null < observed)) / (n_shuffles + 1), as the reference",
            "passes": f"both p-values at or below {SIGNIFICANCE_LEVEL}, as the reference",
        },
    }


def write_manifest(path: Path, manifest: dict) -> None:
    """Write through a temporary file, so a stopped run never leaves half a manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--epoch-id", action="append", help="One epoch; repeatable.")
    which.add_argument("--animal", help="Every canonical HPC decode of this animal.")
    which.add_argument("--all", action="store_true", help="Every canonical HPC decode.")
    parser.add_argument(
        "--run-seed", type=int, required=True, help="Recorded in the manifest."
    )
    parser.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument(
        "--include-variants",
        action="store_true",
        help="Also run decodes made another way (_all, _non_stationary, ...).",
    )
    parser.add_argument("--run-id", help="Defaults to the commit and the seed.")
    parser.add_argument("--root", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=OUTPUT_ROOT / "marks_cache")
    parser.add_argument(
        "--overwrite", action="store_true", help="Redo epochs already written."
    )
    parser.add_argument(
        "--keep-single-window-runs",
        action="store_true",
        help="The sensitivity setting: do not drop runs of one kept window "
        "before merging. Recorded in the manifest's config.",
    )
    args = parser.parse_args(argv)

    shuffle = not args.no_shuffle
    if shuffle and args.n_shuffles < MINIMUM_N_SHUFFLES:
        parser.error(
            f"--n-shuffles must be at least {MINIMUM_N_SHUFFLES}, so that a "
            f"p-value at or below {SIGNIFICANCE_LEVEL} is possible."
        )
    commit = git_hash()
    run_id = args.run_id or (
        f"{commit[:12]}{'-dirty' if commit.endswith('-dirty') else ''}"
        f"-seed{args.run_seed}{'' if shuffle else '-noshuffle'}"
        f"{'-singles' if args.keep_single_window_runs else ''}"
    )
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log = _logger(output_dir / "run.log")
    if commit.endswith("-dirty"):
        log.warning(
            "uncommitted changes (%s); this run cannot be tied to a commit", commit
        )
    config = MemorylessSequenceConfig(
        drop_single_window_runs=not args.keep_single_window_runs
    )
    settings = run_settings(args.run_seed, shuffle, args.n_shuffles, config, commit)
    provenance = {
        "produced_by": "scripts/decoding/run_memoryless_detection.py",
        "git_commit": commit,
        "run_id": run_id,
    }

    skipped: dict[str, str] = {}
    if args.epoch_id:
        epoch_ids = list(args.epoch_id)
    else:
        epoch_ids, skipped = find_epochs(
            None if args.all else args.animal,
            args.root,
            include_variants=args.include_variants,
        )
    for epoch_id, why in sorted(skipped.items()):
        log.info("left out %s: %s", epoch_id, why)
    if skipped:
        log.info("%d epochs left out; %d to run", len(skipped), len(epoch_ids))
    if not epoch_ids:
        sys.exit(f"No HPC decodes to run under {args.root}.")

    manifest_path = output_dir / "manifest.json"
    epochs: dict[str, dict] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("settings") != settings:
            sys.exit(
                f"{manifest_path} was written with other settings, code or "
                "library versions; a run must not mix them. Start a new --run-id."
            )
        epochs = previous["epochs"]
    for epoch_id, why in skipped.items():
        # an epoch already written under this run stays written; a skip only
        # says why it was not run this time
        if epochs.get(epoch_id, {}).get("status") != "ok":
            epochs[epoch_id] = {"status": "skipped", "reason": why}

    def manifest() -> dict:
        return {
            "produced_by": provenance["produced_by"],
            "run_id": run_id,
            "design_spec": DESIGN_SPEC,
            "settings": settings,
            "epochs": epochs,
        }

    write_manifest(manifest_path, manifest())
    failed = []
    for number, epoch_id in enumerate(epoch_ids, start=1):
        done = epochs.get(epoch_id, {}).get("status") == "ok"
        if done and not args.overwrite:
            log.info("[%d/%d] %s: already written", number, len(epoch_ids), epoch_id)
            continue
        log.info("[%d/%d] %s", number, len(epoch_ids), epoch_id)
        try:
            epochs[epoch_id] = run_one(
                epoch_id,
                root=args.root,
                output_dir=output_dir,
                cache_dir=args.cache_dir,
                run_id=run_id,
                run_seed=args.run_seed,
                shuffle=shuffle,
                n_shuffles=args.n_shuffles,
                config=config,
                provenance=provenance,
            )
        except Exception as error:  # noqa: BLE001 - one epoch must not stop the rest
            # recorded, not swallowed: the manifest says which epoch failed and
            # why, the run exits non-zero, and a rerun retries it
            failed.append(epoch_id)
            epochs[epoch_id] = {
                "status": "failed",
                "git_commit": commit,
                "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            log.error("    failed: %s: %s", type(error).__name__, error)
        else:
            record = epochs[epoch_id]
            log.info(
                "    %d windows (%d with unknown speed), %d candidates, %d replay, "
                "%d unknown SWR coverage, %d shuffled (%d untestable), "
                "%d passing the shuffle",
                record["n_windows"],
                record["n_windows_speed_unknown"],
                record["n_candidates"],
                record["n_replay"],
                record["n_swr_unknown"],
                record["n_shuffled"],
                record["n_shuffle_untestable"],
                record["n_shuffle_pass"],
            )
        write_manifest(manifest_path, manifest())
    log.info("manifest -> %s", manifest_path)
    if failed:
        sys.exit(
            f"{len(failed)} epochs failed: {', '.join(failed)}; see {manifest_path}"
        )


def _logger(path: Path) -> logging.Logger:
    """Everything the run says, on the console and in ``run.log`` beside the manifest."""
    log = logging.getLogger("run_memoryless_detection")
    log.setLevel(logging.INFO)
    log.propagate = False
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path)):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log


if __name__ == "__main__":
    main()
