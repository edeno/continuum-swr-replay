#!/usr/bin/env python
"""Generate all manuscript figures.

This script orchestrates the generation of all composite figures for the
manuscript by running each figure generation script in sequence.

Figures generated:
- Figure 1: Switching state-space model (simulated example)
- Figure 2: Model works on real data (example + distributions + triggered avg)
- Figure 3: Events outside SWRs (non-SWR example + SWR z-score by state +
  replay speed + replayed position, SWR vs non-SWR)
- Figure 4: PFC responds regardless (GLM + population heatmaps + significance)

Usage:
    python scripts/figures/generate_all_figures.py
    python scripts/figures/generate_all_figures.py --skip figure01 figure04
    python scripts/figures/generate_all_figures.py --only figure02 figure03
"""

import argparse
import logging
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


# Map of figure scripts and their output directories
FIGURE_SCRIPTS = {
    "figure01": {
        "script": "scripts/figures/generate_figure01.py",
        "output_dir": "paper/figures/figure01",
    },
    "figure02": {
        "script": "scripts/figures/generate_figure02.py",
        "output_dir": "paper/figures/figure02",
    },
    "figure03": {
        "script": "scripts/figures/generate_figure03.py",
        "output_dir": "paper/figures/figure03",
    },
    "figure04": {
        "script": "scripts/figures/generate_figure04.py",
        "output_dir": "paper/figures/figure04",
    },
}


def run_figure_script(
    script_name: str,
    script_config: dict,
    data_dir: Path | None,
) -> tuple[bool, float]:
    """Run a single figure generation script.

    Parameters
    ----------
    script_name : str
        Name of the figure (e.g., "figure01")
    script_config : dict
        Script configuration with 'script' and 'output_dir' keys
    data_dir : Path | None
        Data directory override (if applicable)

    Returns
    -------
    tuple[bool, float]
        (success, elapsed_time) - whether script succeeded and time taken
    """
    logging.info(f"Generating {script_name}...")

    cmd = ["uv", "run", "python", script_config["script"]]
    cmd.extend(["--output-dir", script_config["output_dir"]])

    if data_dir and script_name != "figure01":
        cmd.extend(["--data-dir", str(data_dir)])

    start_time = time.time()
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - start_time
        logging.info(f"  {script_name} completed in {elapsed:.1f}s")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logging.error(f"  {script_name} failed after {elapsed:.1f}s")
        logging.error(f"  Error: {e.stderr[-500:]}")
        return False, elapsed


def main(argv: Sequence[str] | None = None) -> None:
    """Main script execution."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory override (default: PROCESSED_DATA_DIR)",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Figure(s) to skip (e.g., --skip figure01 figure04)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Only generate specific figure(s) (e.g., --only figure02 figure03)",
    )

    args = parser.parse_args(argv)

    # Determine which figures to generate
    if args.only:
        figures_to_generate = {
            k: v for k, v in FIGURE_SCRIPTS.items() if k in args.only
        }
    else:
        figures_to_generate = {
            k: v for k, v in FIGURE_SCRIPTS.items() if k not in args.skip
        }

    # Validate figure names
    invalid_names = set(args.skip + args.only) - set(FIGURE_SCRIPTS.keys())
    if invalid_names:
        logging.error(f"Invalid figure name(s): {', '.join(invalid_names)}")
        logging.error(f"Valid names: {', '.join(FIGURE_SCRIPTS.keys())}")
        sys.exit(1)

    # Print summary
    logging.info("=" * 60)
    logging.info("Generating Manuscript Figures")
    logging.info("=" * 60)
    logging.info(f"Figures: {', '.join(figures_to_generate.keys())}")
    if args.skip:
        logging.info(f"Skipping: {', '.join(args.skip)}")
    logging.info("=" * 60)

    # Run each figure script
    results = {}
    total_start = time.time()

    for name, config in figures_to_generate.items():
        success, elapsed = run_figure_script(name, config, args.data_dir)
        results[name] = {"success": success, "time": elapsed}

    total_elapsed = time.time() - total_start

    # Print summary
    logging.info("=" * 60)
    logging.info("Summary")
    logging.info("=" * 60)

    successful = [name for name, res in results.items() if res["success"]]
    failed = [name for name, res in results.items() if not res["success"]]

    for name in successful:
        logging.info(f"  {name:12s} {results[name]['time']:6.1f}s")

    for name in failed:
        logging.error(f"  {name:12s} {results[name]['time']:6.1f}s")

    logging.info("-" * 60)
    logging.info(
        f"Total: {len(successful)}/{len(results)} succeeded in {total_elapsed:.1f}s"
    )

    if failed:
        logging.error(f"Failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        logging.info("All figures generated successfully!")


if __name__ == "__main__":
    main()
