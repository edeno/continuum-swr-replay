"""Run Continuous-Fragmented (ContFrag) decoding on spatial bandit task data.

This script fits a 2-state ContFrag classifier to neural activity, classifying
all time as either Continuous (smooth spatial trajectory, RandomWalk transitions)
or Fragmented (disjointed spatial representation, Uniform transitions).

Unlike the NonLocal 4-state detector (run_decoding_bandit.py), the ContFrag
model has no Local or No-Spike states — it decodes spatial position everywhere.
Events for downstream analysis are defined externally via SWR and high-MUA times,
not by model-detected probability thresholding.

Usage
-----
# Decode all sessions for an animal
python run_decoding_contfrag.py --animal chimi --brain_area HPC

# Decode specific session
python run_decoding_contfrag.py --animal chimi --date 20200212 --epoch 02_r1

# Write to (and resume from) a staging folder instead of Processed-Data
python run_decoding_contfrag.py --animal chimi --save_dir Processed-Data/_staging_redecode_20261003

Output Files
------------
Saved to ``--save_dir`` (default ``paths.processed_data_dir()``) with
``_contfrag`` suffix:

{epoch_identifier}_contfrag_results.nc
{epoch_identifier}_contfrag_detector.pkl
{epoch_identifier}_contfrag_environment.pkl
{epoch_identifier}_contfrag_data_over_time.pkl
"""

import sys
import warnings
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from continuum_swr_replay.analysis.contfrag import (
    get_contfrag_data_over_time,
    load_contfrag_results,
)
from continuum_swr_replay.decoding.config import (
    DecodingConfig,
    contfrag_classifier_classes,
)
from continuum_swr_replay.decoding.identifiers import (
    CONTFRAG_SUFFIX,
    DEFAULT_SWR_DEFINITION,
    DEFAULT_TRAINING_TYPE,
)
from continuum_swr_replay.decoding.transitions import contfrag_transition_settings
from continuum_swr_replay.runs.batch import run_batch
from continuum_swr_replay.runs.cli import add_common_args

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.decoding import _decode_runner  # noqa: E402

warnings.filterwarnings("ignore")


def contfrag_detector_kwargs(config: DecodingConfig) -> dict[str, Any]:
    """The ContFrag classifier's arguments beyond its environment and params.

    The discrete transitions come from ``contfrag_transition_settings``: the
    values the decoder took from the library before the pin (commit de15650),
    made explicit, so the library's current defaults never reach the decode.
    """
    return contfrag_transition_settings()


RUNNER = _decode_runner.DecodeRunner(
    detector_classes=contfrag_classifier_classes(),
    config_class=DecodingConfig,
    detector_kwargs=contfrag_detector_kwargs,
    aggregate=get_contfrag_data_over_time,
    load_saved=load_contfrag_results,
    file_suffix=CONTFRAG_SUFFIX,
    # Unlike the NonLocal runner, the ContFrag decode does not keep its per-bin
    # log-likelihood in results.nc.
    hpc_log_likelihood=False,
    figurl=None,
    start_banner=" START (ContFrag) ",
    decoding_label="ContFrag decoding",
    model_line="model: ContFrag (2-state: Continuous, Fragmented)",
)
"""The ContFrag 2-state classifier's decode (``_decode_runner.run_decode``)."""


def run_decode(
    nwb_file_name: str,
    epoch_name: str,
    log_directory: str = "",
    overwrite: bool = False,
    training_type: str = DEFAULT_TRAINING_TYPE,
    non_stationary: bool = False,
    ripple_detector_name: str = DEFAULT_SWR_DEFINITION,
    brain_area: str = "HPC",
    save_dir: Path | None = None,
    release_nwb: bool = False,
) -> str | None:
    """Run ContFrag decoding for a single session (``_decode_runner.run_decode``).

    Writes ``{epoch_identifier}_results.nc``, ``_detector.pkl``,
    ``_environment.pkl`` and ``_data_over_time.pkl``, the identifier ending in
    ``_contfrag``, to ``save_dir``, or on a resume ``data_over_time`` alone.

    Returns
    -------
    str | None
        ``None`` on success or graceful skip; the ``epoch_identifier``
        on caught exception so ``run_batch`` can aggregate failures.
    """
    return _decode_runner.run_decode(
        RUNNER,
        nwb_file_name,
        epoch_name,
        log_directory=log_directory,
        overwrite=overwrite,
        training_type=training_type,
        non_stationary=non_stationary,
        ripple_detector_name=ripple_detector_name,
        brain_area=brain_area,
        save_dir=save_dir,
        release_nwb=release_nwb,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Main entry point for batch ContFrag decoding."""
    parser = add_common_args(
        ArgumentParser(description="Run ContFrag decoding on spatial bandit data")
    )
    args = _decode_runner.parse_decode_args(parser, argv)
    run_batch(run_decode, args)


if __name__ == "__main__":
    main()
