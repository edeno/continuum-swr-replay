"""How much do two marked-family bundle trees differ for the same sessions and models?

Written to check that a weak prior on the per-stop baseline columns
(``run_pfc_marked_model_family.py --baseline-penalty-sigma``) barely moves the
units that converged without it. For every (session, model) present in both
trees it pairs units by their recorded index (the second tree may leave out
units with no spike; its sidecar lists ``kept_unit_indices``), keeps units at
or above ``--min-rate`` Hz in the canonical Figure 4 bundle of the session
(``immobility_avg_firing_rate``; the comparison only), and reports per model,
class and target the largest absolute difference of the slope curves (log-rate
per SD), of the typical curves, and of the held-out deltas, plus the median
absolute difference of the slope curves.

Usage
-----
uv run python scripts/pfc/compare_pfc_marked_family_trees.py \
    --reference Processed-Data/pfc_response_followups/marked_family \
    --candidate Processed-Data/pfc_response_followups/marked_family_baseline_prior \
    --out comparison.csv
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

SUFFIX = "_marked_family"


def _extra(sidecar: Path) -> dict:
    record = json.loads(sidecar.read_text())
    return record.get("extra", record)


def compare_pair(
    reference: Path, candidate: Path, rates: np.ndarray, min_rate: float
) -> list[dict]:
    """Rows of differences for one (session, model) bundle pair.

    Parameters
    ----------
    reference, candidate : Path
        The two bundles' ``.pkl`` paths.
    rates : NDArray, shape (n_units_recorded,)
        The session's units' rates over the fitted bins, in recorded order.
    min_rate : float
        Units below this rate (Hz) are left out of the comparison.
    """
    ref_extra = _extra(reference.with_suffix(".provenance.json"))
    cand_extra = _extra(candidate.with_suffix(".provenance.json"))
    n_recorded = len(rates)
    ref_index = np.asarray(
        ref_extra.get("kept_unit_indices", list(range(n_recorded))), dtype=int
    )
    cand_index = np.asarray(
        cand_extra.get("kept_unit_indices", list(range(n_recorded))), dtype=int
    )
    common = np.intersect1d(ref_index, cand_index)
    common = common[rates[common] >= min_rate]
    ref_rows = np.searchsorted(ref_index, common)
    cand_rows = np.searchsorted(cand_index, common)
    ref_stats = pd.read_pickle(reference)
    cand_stats = pd.read_pickle(candidate)
    targets = ref_extra["marked_family_diagnostics"]["scientific_marks"]

    # A class either session dropped (too few events) has no curves in that
    # tree, so only classes both trees fitted are compared.
    def fitted(extra: dict) -> set[str]:
        return set(
            extra.get("fitted_event_classes", extra.get("target_classes", []))
        ) | {"pooled"}

    classes = [
        c
        for c in ref_extra.get("target_classes", [])
        if c in set(cand_extra.get("target_classes", []))
        and c in fitted(ref_extra)
        and c in fitted(cand_extra)
    ]
    rows = []
    for event_class in classes:
        for target in targets:
            name = f".{event_class}.{target}.slope.npy"
            ref_slope = np.load(reference.with_suffix(name))[:, ref_rows]
            cand_slope = np.load(candidate.with_suffix(name))[:, cand_rows]
            both = np.isfinite(ref_slope).all(0) & np.isfinite(cand_slope).all(0)
            diff = np.abs(ref_slope[:, both] - cand_slope[:, both])
            column = f"{event_class}:{target}: heldout_delta_deviance"
            delta = np.abs(
                ref_stats[column].to_numpy(float)[ref_rows]
                - cand_stats[column].to_numpy(float)[cand_rows]
            )
            rows.append(
                {
                    "session": f"{ref_extra['nwb_file_name'].replace('_.nwb', '')}_"
                    f"{ref_extra['epoch_name']}",
                    "model": ref_extra["model_name"],
                    "event_class": event_class,
                    "target": target,
                    "n_units_compared": int(both.sum()),
                    "slope_max_abs_diff": float(diff.max()) if diff.size else np.nan,
                    "slope_median_abs_diff": float(np.median(diff))
                    if diff.size
                    else np.nan,
                    "slope_max_abs_reference": float(np.abs(ref_slope[:, both]).max())
                    if both.any()
                    else np.nan,
                    "heldout_delta_max_abs_diff": float(np.nanmax(delta))
                    if np.isfinite(delta).any()
                    else np.nan,
                }
            )
    for event_class in [c for c in classes if c != "pooled"]:
        name = f".{event_class}.typical.npy"
        if (
            reference.with_suffix(name).is_file()
            and candidate.with_suffix(name).is_file()
        ):
            ref_typical = np.load(reference.with_suffix(name))[:, ref_rows]
            cand_typical = np.load(candidate.with_suffix(name))[:, cand_rows]
            both = np.isfinite(ref_typical).all(0) & np.isfinite(cand_typical).all(0)
            diff = np.abs(ref_typical[:, both] - cand_typical[:, both])
            rows.append(
                {
                    "session": rows[0]["session"] if rows else "",
                    "model": ref_extra["model_name"],
                    "event_class": event_class,
                    "target": "typical",
                    "n_units_compared": int(both.sum()),
                    "slope_max_abs_diff": float(diff.max()) if diff.size else np.nan,
                    "slope_median_abs_diff": float(np.median(diff))
                    if diff.size
                    else np.nan,
                    "slope_max_abs_reference": float(np.abs(ref_typical[:, both]).max())
                    if both.any()
                    else np.nan,
                    "heldout_delta_max_abs_diff": np.nan,
                }
            )
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    from continuum_swr_replay.paths import processed_data_dir

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--min-rate", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows: list[dict] = []
    for candidate in sorted(args.candidate.glob(f"*{SUFFIX}.pkl")):
        reference = args.reference / candidate.name
        if not reference.is_file():
            continue
        session = candidate.name.split("_mPFC_")[0]
        canonical = processed_data_dir() / f"{session}_mPFC_modulation.pkl"
        rates = pd.read_pickle(canonical)["immobility_avg_firing_rate"].to_numpy(float)
        rows += compare_pair(reference, candidate, rates, args.min_rate)
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200):
        print(
            table.sort_values("slope_max_abs_diff", ascending=False)
            .head(15)
            .to_string(index=False)
        )
    print(f"wrote {args.out} ({len(table)} rows)")


if __name__ == "__main__":
    main()
