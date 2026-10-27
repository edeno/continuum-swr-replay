"""Does an extended-event-set mPFC bundle reproduce the canonical bundle where they overlap?

Adding the class ``SWR_no_non_local`` (``run_pfc_analysis_bandit.py
--event-set with_swr_without_replay``) changes no canonical class, design,
fold, seed or matched control, so every canonical statistic and curve of the
extended bundle should equal the canonical bundle's, up to the fitter's
run-to-run tolerance. This compares the two per session and prints, for every
canonical column and curve, the largest absolute difference, and the new
class's counts from the extended sidecar.

Usage
-----
uv run python scripts/pfc/compare_pfc_event_set_bundles.py \
    --extended Processed-Data/pfc_response_followups/swr_without_replay_smoke
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

CURVES = (
    "SWR",
    "no_SWR",
    "non_local_SWR",
    "non_local_no_SWR",
    "matched_control",
)
STEM = "_mPFC_modulation"


def compare_session(canonical: Path, extended: Path) -> dict[str, object]:
    """Largest absolute differences between two bundles of one session.

    Parameters
    ----------
    canonical, extended : Path
        The two ``*_mPFC_modulation.pkl`` files.

    Returns
    -------
    dict
        ``max_abs_column_difference`` (over every numeric canonical column,
        NaN-aware: a NaN in one bundle and not the other counts as infinite),
        ``worst_column``, ``max_abs_curve_difference`` per curve, and
        ``n_units``.
    """
    left = pd.read_pickle(canonical)
    right = pd.read_pickle(extended)
    if len(left) != len(right):
        raise ValueError(f"{canonical.name}: {len(left)} units against {len(right)}")
    worst, worst_column = 0.0, None
    for column in left.columns:
        if column not in right.columns:
            raise KeyError(f"{extended.name} lacks canonical column {column!r}")
        a = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float)
        if np.all(np.isnan(a)) and np.all(np.isnan(b)):
            if not left[column].astype(str).equals(right[column].astype(str)):
                difference = np.inf
            else:
                continue
        else:
            mismatch = np.isnan(a) != np.isnan(b)
            difference = (
                np.inf
                if mismatch.any()
                else float(np.nanmax(np.abs(a - b), initial=0.0))
            )
        if difference > worst:
            worst, worst_column = difference, column
    curves = {}
    for curve in CURVES:
        a = np.load(canonical.with_suffix(f".{curve}_modulation.npy"))
        b = np.load(extended.with_suffix(f".{curve}_modulation.npy"))
        curves[curve] = float(np.nanmax(np.abs(a - b), initial=0.0))
    return {
        "n_units": len(left),
        "max_abs_column_difference": worst,
        "worst_column": worst_column,
        "max_abs_curve_difference": curves,
    }


def main(argv: Sequence[str] | None = None) -> None:
    from continuum_swr_replay.paths import processed_data_dir

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extended", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, default=None)
    args = parser.parse_args(argv)
    canonical_dir = args.canonical or processed_data_dir()
    for extended in sorted(args.extended.glob(f"*{STEM}.pkl")):
        canonical = canonical_dir / extended.name
        sidecar = json.loads(extended.with_suffix(".provenance.json").read_text())
        result = compare_session(canonical, extended)
        print(
            f"{extended.name.removesuffix(STEM + '.pkl')}: units {result['n_units']}, "
            f"max |column diff| {result['max_abs_column_difference']:.3g} "
            f"({result['worst_column']}), max |curve diff| "
            f"{max(result['max_abs_curve_difference'].values()):.3g}; "
            f"SWRs without replay {sidecar.get('n_swr_no_non_local')} "
            f"(with replay {sidecar.get('n_swr_with_non_local')}, unknown "
            f"{sidecar.get('n_swr_no_non_local_unknown')}), event_set "
            f"{sidecar.get('event_set')}"
        )


if __name__ == "__main__":
    main()
