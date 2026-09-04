#!/usr/bin/env python
"""Compare a backed-up rate fit against the current one and write a markdown audit.

The reward block of both rate families changed from per-lag indicators to a
three-bump raised-cosine basis, so individual lag coefficients have no
counterpart in the new fits and are deliberately **not** compared. What is
comparable is the *joint* evidence the reward block buys, the goodness-of-fit
of the reward-free time model, and the size of the fitted cohort:

* Family B (per-bin conditional intensity) reward step, twice: the reward rung
  against its reward-free parent, as a per-session AIC difference
  (``AIC(reward rung) - AIC(parent)``; negative = the reward block helps).
  Old ``M2_rewhist`` / new ``M1_rew`` share the parent ``M0_const``; old and
  new ``M7_time_hist_rew`` share the parent ``M6_time_hist``.
* Family B time-only rung ``M5_time``: median KS statistic and the share of
  sessions inside the KS band — a reward-free rung, so a shift here means the
  cohort or the binning moved, not the reward respecification.
* ``n_bins`` per (session, outcome): the frozen-cohort size.
* Family A (per-stop Poisson) reward step: the new ``R1`` against ``R0``,
  read beside the old ``R2`` and ``Rlag`` against ``R0``.

Usage:
    uv run python scripts/qa/compare_fits.py
    uv run python scripts/qa/compare_fits.py --out docs/audits/my-comparison.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from continuum_swr_replay.analysis.replay._helpers import reconstruct_reward_kernels
from continuum_swr_replay.analysis.replay.model_specs import (
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_BASIS,
    REWARD_HISTORY_N_LAGS,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.session_keys import SESSION_COLUMNS

RC = PROCESSED_DATA_DIR / "replay_content"
REPO = Path(__file__).resolve().parents[2]
SC = list(SESSION_COLUMNS)
KEY = SC + ["outcome"]

DEFAULT_OLD_PER_BIN = (
    RC / "_pre_fix_backup_20260904" / "models_per_bin_rate" / "all_models.pkl"
)
DEFAULT_NEW_PER_BIN = RC / "models_per_bin_rate" / "all_models.pkl"
DEFAULT_OLD_PER_STOP = (
    RC
    / "_pre_v5_regression_refit_backup_20260902"
    / "models_per_stop_rate"
    / "all_models.pkl"
)
DEFAULT_NEW_PER_STOP = RC / "models_per_stop_rate" / "all_models.pkl"

# The reward step of each family: (rung, its reward-free parent).
OLD_PER_BIN_STEPS = {
    "base": ("M2_rewhist", "M0_const"),
    "full": ("M7_time_hist_rew", "M6_time_hist"),
}
NEW_PER_BIN_STEPS = {
    "base": ("M1_rew", "M0_const"),
    "full": ("M7_time_hist_rew", "M6_time_hist"),
}
# Family A: the new single reward rung, and the two old rungs it replaced.
OLD_PER_STOP_STEPS = {"R2 vs R0": ("R2", "R0"), "Rlag vs R0": ("Rlag", "R0")}
NEW_PER_STOP_STEPS = {"R1 vs R0": ("R1", "R0")}

OUTCOME_ORDER = ["swr", "non_local", "non_local_swr", "non_local_no_swr"]
# The rung both reward quantities below are read from.
FULL_MODEL = "M7_time_hist_rew"
IMMEDIATE_COEF = f"coef_{REWARD_HISTORY_COLUMNS[0]}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def delta_aic(
    models: pd.DataFrame, rung: str, parent: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Per (session, outcome) ``AIC(rung) - AIC(parent)``, and what pairing cost.

    A session contributes a difference only when both rungs were fit there and
    both carry a finite AIC. Sessions that reach only one side of the pair are
    silently unusable for this comparison, so they are counted rather than
    dropped without trace.

    Parameters
    ----------
    models : pd.DataFrame
        Tidy per-session fits with ``model``, ``outcome``, the session keys and
        ``aic``.
    rung, parent : str
        The reward rung and its reward-free parent.

    Returns
    -------
    paired : pd.DataFrame
        One row per (session, outcome) with ``delta_aic``.
    unpaired : dict of str to int
        Per outcome, sessions carrying either rung that did not survive the
        pairing (present on one side only, or with a non-finite AIC).
    """
    have = set(models["model"].unique())
    if rung not in have or parent not in have:
        return pd.DataFrame(columns=KEY + ["delta_aic"]), {}
    a = models.loc[models.model == rung, KEY + ["aic"]]
    b = models.loc[models.model == parent, KEY + ["aic"]]
    merged = a.merge(b, on=KEY, suffixes=("_rung", "_parent"))
    merged["delta_aic"] = merged["aic_rung"] - merged["aic_parent"]
    paired = merged[KEY + ["delta_aic"]].dropna(subset=["delta_aic"])
    either = pd.concat([a[KEY], b[KEY]]).drop_duplicates()
    n_either = either.groupby("outcome", observed=True).size()
    n_paired = paired.groupby("outcome", observed=True).size()
    unpaired = n_either.subtract(n_paired, fill_value=0).astype(int).to_dict()
    return paired, unpaired


def _median_table(
    old: pd.DataFrame,
    new: pd.DataFrame,
    value: str,
    old_unpaired: dict[str, int] | None = None,
    new_unpaired: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Per-outcome median of ``value`` in each fit, with session counts.

    ``old_unpaired`` / ``new_unpaired`` add the sessions each side lost to the
    rung-against-parent pairing, so a shrinking median cannot be read as a
    shifted effect when it is really a shrinking denominator.
    """
    rows = []
    for outcome in OUTCOME_ORDER:
        o = (
            old.loc[old.outcome == outcome, value]
            if len(old)
            else pd.Series(dtype=float)
        )
        n = (
            new.loc[new.outcome == outcome, value]
            if len(new)
            else pd.Series(dtype=float)
        )
        if o.empty and n.empty:
            continue
        row = {
            "outcome": outcome,
            "n_sessions_before": int(o.notna().sum()),
            "median_before": o.median(),
            "n_sessions_after": int(n.notna().sum()),
            "median_after": n.median(),
        }
        if old_unpaired is not None or new_unpaired is not None:
            row["n_unpaired_before"] = int((old_unpaired or {}).get(outcome, 0))
            row["n_unpaired_after"] = int((new_unpaired or {}).get(outcome, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def _per_animal_median_table(
    old: pd.DataFrame, new: pd.DataFrame, value: str
) -> pd.DataFrame:
    """Per (outcome, animal) median of ``value`` in each fit."""
    rows = []
    animals = sorted(
        set(old.get("animal", pd.Series(dtype=object)))
        | set(new.get("animal", pd.Series(dtype=object)))
    )
    for outcome in OUTCOME_ORDER:
        for animal in animals:
            o = (
                old[(old.outcome == outcome) & (old.animal == animal)][value]
                if len(old)
                else pd.Series(dtype=float)
            )
            n = (
                new[(new.outcome == outcome) & (new.animal == animal)][value]
                if len(new)
                else pd.Series(dtype=float)
            )
            if o.empty and n.empty:
                continue
            rows.append(
                {
                    "outcome": outcome,
                    "animal": animal,
                    "n_before": int(o.notna().sum()),
                    "median_before": o.median(),
                    "n_after": int(n.notna().sum()),
                    "median_after": n.median(),
                }
            )
    return pd.DataFrame(rows)


def n_bins_table(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Per-outcome cohort size before/after, from any rung (all rungs share it)."""
    rows = []
    for outcome in OUTCOME_ORDER:
        o = old[(old.outcome == outcome) & (old.model == "M0_const")]
        n = new[(new.outcome == outcome) & (new.model == "M0_const")]
        if o.empty and n.empty:
            continue
        merged = o[SC + ["n_bins"]].merge(
            n[SC + ["n_bins"]], on=SC, suffixes=("_before", "_after")
        )
        diff = merged["n_bins_after"] - merged["n_bins_before"]
        rows.append(
            {
                "outcome": outcome,
                "n_sessions_before": len(o),
                "n_sessions_after": len(n),
                "n_sessions_paired": len(merged),
                "median_n_bins_before": o["n_bins"].median(),
                "median_n_bins_after": n["n_bins"].median(),
                "median_delta": diff.median() if len(merged) else np.nan,
                "n_sessions_changed": int((diff != 0).sum()) if len(merged) else 0,
            }
        )
    return pd.DataFrame(rows)


def reward_quantities(new_bin: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two reward quantities the refit is usually summarized by.

    They are computed from different things and must not be quoted as one
    clause. The first is the **reconstructed kernel**: each session's three
    basis coefficients are mapped through the row-normalized basis and the
    per-lag rate ratios are summarized by their median across sessions. The
    second is a **paired test on one coefficient**, the immediate basis
    component, comparing the SWR-coupled and non-SWR non-local populations
    within session.

    Parameters
    ----------
    new_bin : pd.DataFrame
        Tidy per-session per-bin fits.

    Returns
    -------
    kernel : pd.DataFrame
        Median reconstructed rate ratio at lags 1-3 per outcome.
    paired : pd.DataFrame
        One row: the within-session paired comparison of the immediate
        component between the two non-local populations.
    """
    full = new_bin[new_bin.model == FULL_MODEL]
    columns = [f"coef_{c}" for c in REWARD_HISTORY_COLUMNS]
    rows = []
    for outcome in OUTCOME_ORDER:
        fitted = full.loc[full.outcome == outcome, columns].dropna()
        if fitted.empty:
            continue
        kernels = np.exp(
            reconstruct_reward_kernels(
                fitted.to_numpy(dtype=float),
                REWARD_HISTORY_N_LAGS,
                REWARD_HISTORY_N_BASIS,
            )
        )
        median = np.median(kernels, axis=0)
        rows.append(
            {
                "outcome": outcome,
                "n_sessions": len(fitted),
                "rr_lag1": median[0],
                "rr_lag2": median[1],
                "rr_lag3": median[2],
            }
        )
    kernel = pd.DataFrame(rows)

    a = full.loc[full.outcome == "non_local_swr", SC + [IMMEDIATE_COEF]]
    b = full.loc[full.outcome == "non_local_no_swr", SC + [IMMEDIATE_COEF]]
    merged = a.merge(b, on=SC, suffixes=("_swr", "_no_swr")).dropna()
    if len(merged) < 2:
        return kernel, pd.DataFrame()
    difference = merged[f"{IMMEDIATE_COEF}_swr"] - merged[f"{IMMEDIATE_COEF}_no_swr"]
    per_animal = merged.assign(d=difference).groupby("animal")["d"].median()
    paired = pd.DataFrame(
        [
            {
                "comparison": "non_local_swr − non_local_no_swr",
                "n_paired_sessions": len(merged),
                "pct_sessions_positive": 100 * float((difference > 0).mean()),
                "animals_positive": f"{int((per_animal > 0).sum())}/{len(per_animal)}",
                # Pre-formatted: a p-value of this magnitude is unreadable
                # under the table's shared fixed-point float format.
                "signrank_p": "{:.1e}".format(
                    wilcoxon(
                        merged[f"{IMMEDIATE_COEF}_swr"],
                        merged[f"{IMMEDIATE_COEF}_no_swr"],
                    ).pvalue
                ),
            }
        ]
    )
    return kernel, paired


def rung_session_counts(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Sessions fitted per (rung, outcome) in each fit; rungs absent show as 0."""

    def _counts(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["model", "outcome", f"n_sessions{suffix}"])
        out = (
            df.groupby(["model", "outcome"], sort=False)
            .size()
            .rename(f"n_sessions{suffix}")
        )
        return out.reset_index()

    merged = _counts(old, "_before").merge(
        _counts(new, "_after"), on=["model", "outcome"], how="outer"
    )
    merged[["n_sessions_before", "n_sessions_after"]] = (
        merged[["n_sessions_before", "n_sessions_after"]].fillna(0).astype(int)
    )
    merged["outcome"] = pd.Categorical(merged["outcome"], OUTCOME_ORDER, ordered=True)
    return merged.sort_values(["model", "outcome"]).reset_index(drop=True)


def _md(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    """Markdown table; NaN renders as an em dash."""
    if df.empty:
        return "_(no rows)_\n"
    shown = df.copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(
                lambda v: "—" if pd.isna(v) else floatfmt.format(v)
            )
    header = "| " + " | ".join(shown.columns) + " |"
    rule = "| " + " | ".join("---" for _ in shown.columns) + " |"
    body = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in shown.itertuples(index=False)
    )
    return f"{header}\n{rule}\n{body}\n"


def build_report(
    old_bin: pd.DataFrame,
    new_bin: pd.DataFrame,
    old_stop: pd.DataFrame,
    new_stop: pd.DataFrame,
    inputs: dict[str, Path],
    per_stop_note: str,
) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Rate refit comparison: pre-fix fits vs the reward-basis refit\n")
    add(
        "Generated by `scripts/qa/compare_fits.py`. The reward block changed from "
        "per-lag reward indicators to a three-bump raised-cosine basis, so lag "
        "coefficients are **not** compared term for term. Comparable quantities "
        "are the joint reward step (ΔAIC of a reward rung against its reward-free "
        "parent, negative = reward block helps), the reward-free `M5_time` "
        "goodness-of-fit, the frozen-cohort size, and the per-rung session counts.\n"
    )
    add("## Inputs\n")
    rows = []
    for name, path in inputs.items():
        rows.append(
            {
                "artifact": name,
                "path": str(path.relative_to(REPO))
                if path.is_relative_to(REPO)
                else str(path),
                "sha256[:12]": _sha256(path)[:12],
                "modified": dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(
                    timespec="seconds"
                ),
            }
        )
    add(_md(pd.DataFrame(rows)))
    add(f"\n{per_stop_note}\n")

    kernel, paired = reward_quantities(new_bin)
    add("\n## Reward effect after the refit — two distinct quantities\n")
    add(
        "These two are routinely quoted together and are **not** the same "
        f"measurement. Both are read from `{FULL_MODEL}` in the new fit.\n"
    )
    add(
        f"\n**(a) Reconstructed kernel.** Each session's "
        f"{REWARD_HISTORY_N_BASIS} basis coefficients are mapped through the "
        "row-normalized basis to a rate ratio per reward lag; the table is the "
        "median of those per-session kernels. It describes the size of the "
        "reward effect at a given lag.\n"
    )
    add(_md(kernel, floatfmt="{:.2f}"))
    add(
        "\n**(b) Paired coefficient test.** A within-session comparison of the "
        f"single immediate basis coefficient (`{IMMEDIATE_COEF}`) between the "
        "two non-local populations. It describes how *consistently* the two "
        "populations differ, not how large the kernel is. The session share, "
        "the animal count and the p-value all come from this test and from no "
        "other quantity.\n"
    )
    add(_md(paired, floatfmt="{:.1f}"))

    add("\n## Family B — joint reward step (per-bin conditional intensity)\n")
    for label, title in [
        (
            "base",
            "Reward rung vs constant baseline (old `M2_rewhist` − `M0_const`; new `M1_rew` − `M0_const`)",
        ),
        (
            "full",
            "Full model vs time+history (old and new `M7_time_hist_rew` − `M6_time_hist`)",
        ),
    ]:
        old_d, old_unpaired = delta_aic(old_bin, *OLD_PER_BIN_STEPS[label])
        new_d, new_unpaired = delta_aic(new_bin, *NEW_PER_BIN_STEPS[label])
        add(f"\n### {title}\n")
        add(
            "Per-session ΔAIC, median across sessions. `n_unpaired_*` counts "
            "sessions that carried one of the two rungs but did not survive the "
            "pairing, and so contribute to neither median:\n"
        )
        add(_md(_median_table(old_d, new_d, "delta_aic", old_unpaired, new_unpaired)))
        add("\nPer-animal medians:\n")
        add(_md(_per_animal_median_table(old_d, new_d, "delta_aic")))

    add("\n## Family B — `M5_time` goodness of fit (reward-free rung)\n")
    old_m5 = old_bin[old_bin.model == "M5_time"]
    new_m5 = new_bin[new_bin.model == "M5_time"]
    add("KS statistic (median across sessions):\n")
    add(_md(_median_table(old_m5, new_m5, "ks_statistic"), floatfmt="{:.4f}"))
    add("\nShare of sessions inside the KS band (`ks_within_band` mean):\n")
    band = _median_table(
        old_m5.assign(ks_within_band=old_m5["ks_within_band"].astype(float)),
        new_m5.assign(ks_within_band=new_m5["ks_within_band"].astype(float)),
        "ks_within_band",
    )
    for df, src in [(old_m5, "median_before"), (new_m5, "median_after")]:
        band[src.replace("median", "share")] = [
            df.loc[df.outcome == oc, "ks_within_band"].astype(float).mean()
            for oc in band["outcome"]
        ]
    add(_md(band.drop(columns=["median_before", "median_after"]), floatfmt="{:.3f}"))

    add("\n## Family B — frozen cohort size (`n_bins`, `M0_const`)\n")
    add(_md(n_bins_table(old_bin, new_bin), floatfmt="{:.1f}"))

    add("\n## Family A — per-stop reward step\n")
    add(
        "The new ladder carries one reward rung (`R1`: three basis features plus "
        "the structural-missingness indicator, 5 parameters); the old ladder "
        "carried `R1` (the lag-1 indicator alone, 2 parameters), `R2` (lag-1 + "
        "the ten-trial reward fraction, 3) and `Rlag` (ten lag indicators, 11). "
        "The rung name `R1` therefore denotes **different specifications** "
        "before and after — read the new step beside all three old steps, not "
        "as a term-for-term replacement.\n"
    )
    for label, (rung, parent) in {**OLD_PER_STOP_STEPS, **NEW_PER_STOP_STEPS}.items():
        old_d, old_unpaired = delta_aic(old_stop, rung, parent)
        new_d, new_unpaired = delta_aic(new_stop, rung, parent)
        add(f"\n### {label}\n")
        add(_md(_median_table(old_d, new_d, "delta_aic", old_unpaired, new_unpaired)))

    add("\n## Sessions fitted per rung\n")
    add("### Family B\n")
    add(_md(rung_session_counts(old_bin, new_bin)))
    add("\n### Family A\n")
    add(_md(rung_session_counts(old_stop, new_stop)))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-per-bin", type=Path, default=DEFAULT_OLD_PER_BIN)
    parser.add_argument("--new-per-bin", type=Path, default=DEFAULT_NEW_PER_BIN)
    parser.add_argument("--old-per-stop", type=Path, default=DEFAULT_OLD_PER_STOP)
    parser.add_argument("--new-per-stop", type=Path, default=DEFAULT_NEW_PER_STOP)
    parser.add_argument(
        "--per-stop-note",
        default=(
            "**Family A caveat.** The per-stop fit that preceded this refit was "
            'overwritten in place before it could be archived, so the "before" '
            "artifact here is one refit older than the lost 2026-09-02 fit; same "
            "spec family (R0/R1/R2/Rlag)."
        ),
        help="Provenance caveat printed under the input table.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO
        / "docs"
        / "audits"
        / f"{dt.date.today().isoformat()}-rate-refit-comparison.md",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    inputs = {
        "per-bin (before)": args.old_per_bin,
        "per-bin (after)": args.new_per_bin,
        "per-stop (before)": args.old_per_stop,
        "per-stop (after)": args.new_per_stop,
    }
    missing = [str(p) for p in inputs.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing fit artifact(s): {missing}")
    report = build_report(
        pd.read_pickle(args.old_per_bin),
        pd.read_pickle(args.new_per_bin),
        pd.read_pickle(args.old_per_stop),
        pd.read_pickle(args.new_per_stop),
        inputs,
        args.per_stop_note,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
