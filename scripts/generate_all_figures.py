#!/usr/bin/env python
"""Generate all manuscript figures.

This script orchestrates the generation of all figures for the manuscript by
running each figure generation script in sequence. It provides progress reporting
and error handling.

Figures generated:
- Figure 1: Simulated non-local detection example
- Figure 2: Non-local event detection and characterization (4 panels)
- Figure 3: Event characteristics and physiological correlates
- Figure 4: GLM analysis of mPFC modulation

Usage:
    python scripts/generate_all_figures.py
    python scripts/generate_all_figures.py --output-dir paper/figures --format pdf
    python scripts/generate_all_figures.py --skip figure01 figure04
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


# Map of figure scripts
FIGURE_SCRIPTS = {
    "figure01": "scripts/generate_figure01.py",
    "figure02a": "scripts/generate_figure02a_distributions.py",
    "figure02b": "scripts/generate_figure02b_triggered_averages.py",
    "figure02c": "scripts/generate_figure02c_overlaps.py",
    "figure02d": "scripts/generate_figure02d_timing.py",
    "figure03": "scripts/generate_figure03.py",
    "figure04": "scripts/generate_figure04.py",
}


def run_figure_script(
    script_name: str,
    script_path: str,
    output_dir: Path,
    format: str,
    dpi: int,
    data_dir: Path | None,
) -> tuple[bool, float]:
    """Run a single figure generation script.

    Parameters
    ----------
    script_name : str
        Name of the figure (e.g., "figure01")
    script_path : str
        Path to the script file
    output_dir : Path
        Output directory for figures
    format : str
        Output format (pdf, png, svg)
    dpi : int
        DPI for raster formats
    data_dir : Path | None
        Data directory (if applicable)

    Returns
    -------
    tuple[bool, float]
        (success, elapsed_time) - whether script succeeded and time taken
    """
    logging.info(f"Generating {script_name}...")

    # Build command
    cmd = ["uv", "run", "python", script_path]
    cmd.extend(["--output-dir", str(output_dir)])
    cmd.extend(["--format", format])
    cmd.extend(["--dpi", str(dpi)])

    # Add data-dir for applicable scripts
    if data_dir and script_name not in ["figure01", "figure04"]:
        cmd.extend(["--data-dir", str(data_dir)])

    # Run script
    start_time = time.time()
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - start_time
        logging.info(f" {script_name} completed in {elapsed:.1f}s")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logging.error(f" {script_name} failed after {elapsed:.1f}s")
        logging.error(f"  Error: {e.stderr[:500]}")  # Show first 500 chars
        return False, elapsed


def main() -> None:
    """Main script execution."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Output directory for all figures (default: paper/figures)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pdf", "png", "svg"],
        default="pdf",
        help="Output format for all figures (default: pdf)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster formats (default: 300)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory for figure02/03 scripts (default: PROCESSED_DATA_DIR)",
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
        help="Only generate specific figure(s) (e.g., --only figure02a figure03)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Format: {args.format} (DPI: {args.dpi})")
    if args.data_dir:
        logging.info(f"Data directory: {args.data_dir}")
    logging.info(f"Figures to generate: {', '.join(figures_to_generate.keys())}")
    if args.skip:
        logging.info(f"Skipping: {', '.join(args.skip)}")
    logging.info("=" * 60)

    # Run each figure script
    results = {}
    total_start = time.time()

    for name, script in figures_to_generate.items():
        success, elapsed = run_figure_script(
            name, script, args.output_dir, args.format, args.dpi, args.data_dir
        )
        results[name] = {"success": success, "time": elapsed}

    total_elapsed = time.time() - total_start

    # Print summary
    logging.info("=" * 60)
    logging.info("Summary")
    logging.info("=" * 60)

    successful = [name for name, res in results.items() if res["success"]]
    failed = [name for name, res in results.items() if not res["success"]]

    for name in successful:
        logging.info(f" {name:12s} {results[name]['time']:6.1f}s")

    for name in failed:
        logging.error(f" {name:12s} {results[name]['time']:6.1f}s")

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
