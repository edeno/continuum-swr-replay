#!/usr/bin/env python
"""Decompose Figure 2b (non-SWR non-local plurality) by animal and session.

Question: is the population-level dominance of non-SWR non-local events driven
by a minority of animals/days, or is it broadly distributed?

Reproduces the panel-b classification exactly (immobile non-local events,
3-way overlap_class) then breaks the counts down per-animal and per-session,
runs leave-one-animal-out, and quantifies concentration. Writes a CSV per
level plus a text summary to .claude/research/fig2b_outputs/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.decode_quality.data_aggregation import (
    load_data_over_time,
)
from continuum_swr_replay.analysis.event_definition.classified_table import (
    build_classified_nonlocal_event_table,
)
from continuum_swr_replay.analysis.event_properties.event_statistics import (
    compute_swr_overlap_class_counts,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)

OUT = Path(__file__).resolve().parents[2] / ".claude/research/fig2b_outputs"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["swr_contained", "swr_partial", "non_swr"]


def class_breakdown(df: pd.DataFrame) -> dict:
    """3-way counts + non_swr fraction + plurality flag for an immobile subset."""
    vc = df["overlap_class"].value_counts()
    c = {k: int(vc.get(k, 0)) for k in CLASSES}
    total = int(len(df))
    c["total"] = total
    c["f_nonswr"] = c["non_swr"] / total if total else np.nan
    c["f_swr_any"] = (
        (c["swr_contained"] + c["swr_partial"]) / total if total else np.nan
    )
    # D1 plurality: non_swr is strictly the largest class
    c["non_swr_is_plurality"] = total > 0 and c["non_swr"] > max(
        c["swr_contained"], c["swr_partial"]
    )
    # D2 majority
    c["non_swr_is_majority"] = total > 0 and c["non_swr"] > total / 2
    return c


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main() -> None:
    logging.info("Loading data_over_time from %s ...", PROCESSED_DATA_DIR)
    dot = load_data_over_time(PROCESSED_DATA_DIR)
    logging.info("Building classified non-local event table ...")
    tbl = build_classified_nonlocal_event_table(dot)

    # event_id uniqueness sanity (no cross-session collision inflating a class)
    n_dup = int(tbl["event_id"].duplicated().sum())
    logging.info("event_id duplicates: %d (expect 0)", n_dup)

    immobile = tbl.loc[tbl["whole_event_speed_eligible"].astype(bool)].copy()

    # ── 0. Reproduce pooled panel-b counts ──
    pooled = compute_swr_overlap_class_counts(tbl)
    pooled_bd = class_breakdown(immobile)

    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit("=" * 72)
    emit("FIG 2b ROBUSTNESS DECOMPOSITION")
    emit("=" * 72)
    emit(
        f"Sessions: {immobile.groupby(['animal', 'day', 'epoch']).ngroups}  "
        f"Animals: {immobile['animal'].nunique()}"
    )
    emit(f"Total non-local events (all): {len(tbl):,}  immobile: {len(immobile):,}")
    emit("")
    emit("[0] POOLED (event-weighted) — should match panel b:")
    emit(
        f"    contained={pooled['swr_contained']:,}  "
        f"partial={pooled['swr_partial']:,}  non_swr={pooled['non_swr']:,}  "
        f"total={pooled['total']:,}"
    )
    emit(
        f"    f_nonswr (pooled) = {pooled_bd['f_nonswr']:.3f}   "
        f"non_swr plurality={pooled_bd['non_swr_is_plurality']}  "
        f"majority={pooled_bd['non_swr_is_majority']}"
    )
    assert pooled["non_swr"] == pooled_bd["non_swr"], "reproduction mismatch"

    # ── 1. Per-animal ──
    emit("")
    emit("[1] PER-ANIMAL breakdown:")
    rows = []
    for animal, g in immobile.groupby("animal"):
        bd = class_breakdown(g)
        bd["animal"] = animal
        bd["n_sessions"] = g.groupby(["day", "epoch"]).ngroups
        rows.append(bd)
        emit(
            f"    {animal:8s} n={bd['total']:6,d} sess={bd['n_sessions']:3d}  "
            f"contained={bd['swr_contained']:5d} partial={bd['swr_partial']:5d} "
            f"non_swr={bd['non_swr']:6d}  f_nonswr={bd['f_nonswr']:.3f}  "
            f"plurality={str(bd['non_swr_is_plurality']):5s} "
            f"majority={bd['non_swr_is_majority']}"
        )
    per_animal = pd.DataFrame(rows).set_index("animal")
    per_animal.to_csv(OUT / "per_animal.csv")

    n_animals_plurality = int(per_animal["non_swr_is_plurality"].sum())
    emit(
        f"    --> non_swr is plurality in {n_animals_plurality}/"
        f"{len(per_animal)} animals"
    )
    emit(
        f"    --> f_nonswr range across animals: "
        f"[{per_animal['f_nonswr'].min():.3f}, {per_animal['f_nonswr'].max():.3f}]"
    )

    # ── 2. Per-session ──
    emit("")
    emit("[2] PER-SESSION distribution:")
    srows = []
    for (animal, day, epoch), g in immobile.groupby(["animal", "day", "epoch"]):
        bd = class_breakdown(g)
        bd.update(animal=animal, day=day, epoch=epoch)
        srows.append(bd)
    per_session = pd.DataFrame(srows)
    per_session.to_csv(OUT / "per_session.csv", index=False)

    # require a minimum number of events for a session-level fraction to count
    MIN_N = 10
    sized = per_session[per_session["total"] >= MIN_N]
    emit(
        f"    {len(per_session)} sessions; {len(sized)} with >= {MIN_N} immobile events"
    )
    emit(
        f"    session event-count: median={per_session['total'].median():.0f} "
        f"min={per_session['total'].min():.0f} max={per_session['total'].max():.0f}"
    )
    emit(
        f"    f_nonswr per session (>= {MIN_N} ev): "
        f"median={sized['f_nonswr'].median():.3f}  "
        f"mean(unweighted)={sized['f_nonswr'].mean():.3f}  "
        f"IQR=[{sized['f_nonswr'].quantile(0.25):.3f}, "
        f"{sized['f_nonswr'].quantile(0.75):.3f}]"
    )
    frac_plur = sized["non_swr_is_plurality"].mean()
    frac_maj = sized["non_swr_is_majority"].mean()
    emit(
        f"    sessions where non_swr is PLURALITY: {frac_plur:.1%} "
        f"({int(sized['non_swr_is_plurality'].sum())}/{len(sized)})"
    )
    emit(f"    sessions where non_swr is MAJORITY:  {frac_maj:.1%}")
    emit(
        f"    --> event-weighted pooled f_nonswr={pooled_bd['f_nonswr']:.3f} vs "
        f"unweighted session-mean f_nonswr={sized['f_nonswr'].mean():.3f} "
        f"(H3 Simpson check: large gap => count-imbalance)"
    )

    # ── 3. Concentration ──
    emit("")
    emit("[3] CONCENTRATION of non_swr events:")
    tot_nonswr = per_session["non_swr"].sum()
    top_animal = per_animal["non_swr"].sort_values(ascending=False)
    emit(
        f"    top animal by non_swr count: {top_animal.index[0]} "
        f"({top_animal.iloc[0]:,} = {top_animal.iloc[0] / tot_nonswr:.1%} of all non_swr)"
    )
    sess_sorted = per_session.sort_values("non_swr", ascending=False)
    for k in (1, 5, 10):
        share = sess_sorted["non_swr"].head(k).sum() / tot_nonswr
        emit(f"    top {k:2d} sessions hold {share:.1%} of all non_swr events")
    emit(
        f"    Gini of non_swr across sessions: "
        f"{gini(per_session['non_swr'].values):.3f}  "
        f"(events/session Gini: {gini(per_session['total'].values):.3f})"
    )

    # ── 4. Leave-one-animal-out ──
    emit("")
    emit("[4] LEAVE-ONE-ANIMAL-OUT (does pooled plurality survive?):")
    for animal in sorted(immobile["animal"].unique()):
        sub = immobile[immobile["animal"] != animal]
        bd = class_breakdown(sub)
        emit(
            f"    drop {animal:8s}: non_swr={bd['non_swr']:6d} "
            f"partial={bd['swr_partial']:5d} contained={bd['swr_contained']:5d}  "
            f"f_nonswr={bd['f_nonswr']:.3f}  plurality={bd['non_swr_is_plurality']}"
        )

    # ── 5. H3: f_nonswr vs session size ──
    emit("")
    emit("[5] H3 count-imbalance: corr(session f_nonswr, log event count):")
    s = sized.copy()
    r = np.corrcoef(np.log(s["total"]), s["f_nonswr"])[0, 1]
    emit(f"    Pearson r = {r:+.3f} over {len(s)} sized sessions")

    (OUT / "summary.txt").write_text("\n".join(lines))
    emit("")
    emit(f"Wrote: {OUT}/per_animal.csv, per_session.csv, summary.txt")


if __name__ == "__main__":
    main()
