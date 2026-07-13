"""Within-arm temporal test: does the SAME arm attract more replay content when
its belief uncertainty is transiently higher?

This is the GLM-free, confound-killing twin of the ``Mu`` content result
(``arm_uncertainty_z`` +0.83). The conditional-Poisson hierarchy identifies the
uncertainty effect from *within-leaf-arm* variation (arm fixed effects soak up
arm identity/geometry); a skeptic worried about the VIF-18 / CS6 artifacts can
see the same effect here as a plain within-arm slope.

Design
------
For each session (and each SWR population) we regress content on uncertainty
*with arm fixed effects*, i.e. we within-(session, arm) demean both the response
``arm_fraction`` (hard-count content weight, sums to 1 over the 9 arms per event)
and the predictor ``arm_uncertainty_z``, then OLS the pooled demeaned pairs. The
slope is "extra content fraction per +1 SD of that arm's own uncertainty,
holding arm identity fixed." Leaf arms only (arms 0-5 carry uncertainty; trunk
arms 6-8 are 0).

The headline version **excludes rows where the arm is the current arm**: content
is current-locked and the current arm is freshly visited (low uncertainty), a
within-arm confound that runs opposite to the effect. Removing it isolates the
uncertainty preference among the non-current arms. A recency-controlled partial
slope (adds within-arm-demeaned ``visit_recency``) addresses the built-in
uncertainty<->recency coupling.

Inference follows the project convention: fit per session, then test the
per-session slopes across sessions (Wilcoxon signed-rank; session = exchangeable
unit) with per-animal sign consistency (n/5).

Everything is correlational.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

TABLE = Path("Processed-Data/replay_content/event_arm_table.pkl")
OUTDIR = Path(".claude/research/within_arm_outputs")
ANIMAL_ORDER = ["chimi", "j16", "peanut", "senor", "wilbur"]


def _fe_slope(df: pd.DataFrame, controls: tuple[str, ...] = ()) -> float:
    """Within-(arm) fixed-effects OLS slope of arm_fraction on arm_uncertainty_z.

    Demeans response and predictors within each arm (the FE transform), pools the
    demeaned rows, and returns the uncertainty coefficient. ``controls`` adds
    further within-arm-demeaned covariates (e.g. visit_recency) to the design.

    Parameters
    ----------
    df : pd.DataFrame
        Rows are (event, arm) pairs for one session/population; must contain
        ``arm``, ``arm_fraction``, ``arm_uncertainty_z`` and any ``controls``.
    controls : tuple of str
        Extra columns to residualize alongside uncertainty.

    Returns
    -------
    float
        Within-arm slope (extra content fraction per +1 SD uncertainty), or NaN
        if under-determined / singular.
    """
    cols = ["arm_fraction", "arm_uncertainty_z", *controls]
    d = df[["arm", *cols]].dropna()
    if d["arm"].nunique() < 2 or len(d) < 30:
        return np.nan
    dm = d[cols] - d.groupby(d["arm"])[cols].transform("mean")  # within-arm demean
    x = dm[["arm_uncertainty_z", *controls]].to_numpy()
    y = dm["arm_fraction"].to_numpy()
    if not np.isfinite(x).all() or np.linalg.matrix_rank(x) < x.shape[1]:
        return np.nan
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(beta[0])


def per_session_slopes(
    leaf: pd.DataFrame, exclude_current: bool, controls: tuple[str, ...] = ()
) -> pd.DataFrame:
    """One within-arm slope per (session, population)."""
    rows = []
    src = leaf[leaf["is_current"] == 0] if exclude_current else leaf
    for (session, animal), g in src.groupby(["session", "animal"], sort=False):
        for pop, mask in [
            ("all", slice(None)),
            ("non_SWR", ~g["is_swr"].astype(bool)),
            ("SWR", g["is_swr"].astype(bool)),
        ]:
            sub = g if isinstance(mask, slice) else g[mask]
            rows.append(
                dict(
                    session=session,
                    animal=animal,
                    population=pop,
                    slope=_fe_slope(sub, controls),
                )
            )
    return pd.DataFrame(rows)


def summarize(slopes: pd.DataFrame, label: str) -> None:
    print(f"\n### {label}")
    print("   (within-arm slope: extra content fraction per +1 SD of that arm's uncertainty)")
    for pop in ["all", "non_SWR", "SWR"]:
        s = slopes[slopes.population == pop].dropna(subset=["slope"])
        med = s["slope"].median()
        try:
            p = wilcoxon(s["slope"]).pvalue
        except ValueError:
            p = np.nan
        per_animal = s.groupby("animal")["slope"].median()
        n_sign = int((np.sign(per_animal) == np.sign(med)).sum())
        tag = "  ".join(
            f"{a[:4]}:{per_animal.get(a, np.nan):+.4f}" for a in ANIMAL_ORDER
        )
        print(
            f"   {pop:8s} median={med:+.5f}  p={p:.1e}  animals_sign={n_sign}/5  "
            f"(n_sess={len(s)})"
        )
        print(f"            per-animal: {tag}")


def binned_profile(leaf: pd.DataFrame) -> pd.DataFrame:
    """Within-arm-demeaned content by uncertainty tertile (for the figure).

    Pools non-current leaf rows, within-arm demeans both quantities, then bins by
    demeaned-uncertainty tertile and reports mean demeaned content per SWR pop.
    """
    src = leaf[leaf["is_current"] == 0].copy()
    cols = ["arm_fraction", "arm_uncertainty_z"]
    key = src["session"].astype(str) + "|" + src["arm"].astype(str)
    dm = src[cols] - src.groupby(key)[cols].transform("mean")
    src["unc_dm"] = dm["arm_uncertainty_z"]
    src["frac_dm"] = dm["arm_fraction"]
    src = src.dropna(subset=["unc_dm", "frac_dm"])
    out = []
    for pop, mask in [
        ("non_SWR", ~src["is_swr"].astype(bool)),
        ("SWR", src["is_swr"].astype(bool)),
    ]:
        sub = src[mask].copy()
        sub["tert"] = pd.qcut(sub["unc_dm"], 3, labels=["low", "mid", "high"])
        m = sub.groupby("tert", observed=True)["frac_dm"].mean()
        out.append(dict(population=pop, **m.to_dict()))
    return pd.DataFrame(out).set_index("population")


def raw_binscatter(leaf: pd.DataFrame, n_bins: int = 10) -> dict[str, pd.DataFrame]:
    """Raw relationship: mean content share (%) vs uncertainty, no demeaning.

    Non-current leaf (event, arm) rows are binned into global deciles of
    ``arm_uncertainty_z``; within each SWR population we average ``arm_fraction``
    per session per bin, then report the across-session mean and SEM (session is
    the inferential unit). This is the rawest population view — it still mixes arm
    identity (that is what the within-arm panel controls).
    """
    src = leaf[leaf["is_current"] == 0].copy()
    src["ubin"] = pd.qcut(src["arm_uncertainty_z"], n_bins, labels=False, duplicates="drop")
    src["ucenter"] = src.groupby("ubin")["arm_uncertainty_z"].transform("mean")
    out = {}
    for pop, mask in [("non_SWR", ~src["is_swr"].astype(bool)), ("SWR", src["is_swr"].astype(bool))]:
        sub = src[mask]
        per_sess = sub.groupby(["session", "ubin"], observed=True).agg(
            frac=("arm_fraction", "mean"), ucenter=("ucenter", "first")
        )
        agg = per_sess.groupby("ubin").agg(
            pct=("frac", lambda x: 100 * x.mean()),
            sem=("frac", lambda x: 100 * x.std() / np.sqrt(len(x))),
            ucenter=("ucenter", "mean"),
        )
        out[pop] = agg
    return out


def within_arm_tertile_raw(leaf: pd.DataFrame) -> pd.DataFrame:
    """Content share (%) by an arm's OWN uncertainty tertile — same arms, raw %.

    Uncertainty is ranked *within* each (session, arm) so every arm contributes
    equally to its own low/mid/high thirds (arm identity is balanced out). Content
    is the raw ``arm_fraction``; we average per session per tertile then across
    sessions (mean, SEM). Answers: "when an arm is in its high-uncertainty third,
    what % of the replay does it get, vs its low third?"
    """
    src = leaf[leaf["is_current"] == 0].copy()
    key = src["session"].astype(str) + "|" + src["arm"].astype(str)
    src["urank"] = src.groupby(key)["arm_uncertainty_z"].rank(pct=True)
    src["tert"] = pd.cut(src["urank"], [0, 1 / 3, 2 / 3, 1.0],
                         labels=["low", "mid", "high"], include_lowest=True)
    rows = []
    for pop, mask in [("non_SWR", ~src["is_swr"].astype(bool)), ("SWR", src["is_swr"].astype(bool))]:
        sub = src[mask]
        per_sess = sub.groupby(["session", "tert"], observed=True)["arm_fraction"].mean()
        agg = per_sess.groupby("tert", observed=True).agg(
            pct=lambda x: 100 * x.mean(), sem=lambda x: 100 * x.std() / np.sqrt(len(x))
        )
        rows.append(dict(population=pop, **{f"{t}_{c}": agg.loc[t, c]
                                            for t in ["low", "mid", "high"] for c in ["pct", "sem"]}))
    return pd.DataFrame(rows).set_index("population")


def make_raw_figure(binsc: dict[str, pd.DataFrame], tert: pd.DataFrame) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    colors = {"non_SWR": "#4477AA", "SWR": "#CC6677"}
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Panel A: raw binscatter — content share vs uncertainty (no demeaning).
    for pop, agg in binsc.items():
        ax0.errorbar(agg["ucenter"], agg["pct"], yerr=agg["sem"], marker="o", ms=4,
                     lw=1.5, capsize=2, color=colors[pop], label=pop.replace("_", "-"))
    ax0.set_xlabel("arm belief uncertainty (z)")
    ax0.set_ylabel("mean replay content share (%)")
    ax0.set_title("A  Raw: more content for more-uncertain arms\n(non-current leaf arms, binned; error = SEM across sessions)")
    ax0.legend(frameon=False)

    # Panel B: same arm, own-uncertainty tertile, raw %.
    x = np.arange(3)
    w = 0.38
    for i, pop in enumerate(["non_SWR", "SWR"]):
        pcts = [tert.loc[pop, f"{t}_pct"] for t in ["low", "mid", "high"]]
        sems = [tert.loc[pop, f"{t}_sem"] for t in ["low", "mid", "high"]]
        ax1.bar(x + (i - 0.5) * w, pcts, yerr=sems, width=w, capsize=3,
                color=colors[pop], label=pop.replace("_", "-"))
    ax1.set_xticks(x)
    ax1.set_xticklabels(["low", "mid", "high"])
    ax1.set_xlabel("the arm's own uncertainty tertile")
    ax1.set_ylabel("that arm's replay content share (%)")
    ax1.set_title("B  Same arms, more content in their\nmore-uncertain moments")
    ax1.legend(frameon=False)

    fig.tight_layout()
    path = OUTDIR / "within_arm_uncertainty_raw.png"
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def horse_race(oc: pd.DataFrame) -> pd.DataFrame:
    """Per-session within-arm multiple regression: uncertainty vs its parallels vs value.

    Non-current leaf rows; response ``arm_fraction`` and all predictors are
    within-(session, arm) demeaned (arm fixed effects), predictors z-scored so
    coefficients are directly comparable. One row per session.
    """
    preds = ["arm_uncertainty_z", "dist_to_arm_z", "visit_recency_z", "q_arm_z"]
    d = oc.copy()
    for c in ["visit_recency", "dist_to_arm", "q_arm"]:
        d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()
    rows = []
    for (s, a), g in d.groupby(["session", "animal"], sort=False):
        sub = g[["arm", "arm_fraction", *preds]].dropna()
        if sub["arm"].nunique() < 2 or len(sub) < 60:
            continue
        dm = sub[["arm_fraction", *preds]] - sub.groupby(sub["arm"])[
            ["arm_fraction", *preds]
        ].transform("mean")
        X = dm[preds].to_numpy()
        if np.linalg.matrix_rank(X) < X.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(X, dm["arm_fraction"].to_numpy(), rcond=None)
        rows.append(dict(session=s, animal=a, **dict(zip(preds, beta))))
    return pd.DataFrame(rows)


def value_uncertainty_map(oc: pd.DataFrame) -> dict[str, np.ndarray]:
    """3x3 mean content share (%) over within-arm uncertainty x HMM-value tertiles.

    Both axes ranked *within* (session, arm) so arm identity is balanced and we
    read the joint modulation over time. Rows = value tertile (low->high, bottom
    to top), cols = uncertainty tertile (low->high). Because the two axes are
    ~orthogonal, content should climb left->right (uncertainty) and stay ~flat
    bottom->top (value).
    """
    d = oc[["session", "arm", "is_swr", "arm_fraction", "arm_uncertainty_z", "q_arm"]].dropna().copy()
    key = d["session"].astype(str) + "|" + d["arm"].astype(str)
    for src, name in [("arm_uncertainty_z", "u_t"), ("q_arm", "v_t")]:
        rank = d.groupby(key)[src].rank(pct=True)
        d[name] = pd.cut(rank, [0, 1 / 3, 2 / 3, 1.0], labels=[0, 1, 2], include_lowest=True).astype(float)
    out = {}
    for pop, mask in [("non_SWR", ~d["is_swr"].astype(bool)), ("SWR", d["is_swr"].astype(bool))]:
        sub = d[mask]
        grid = np.full((3, 3), np.nan)
        cell = sub.groupby(["v_t", "u_t"])["arm_fraction"].mean() * 100
        for (v, u), val in cell.items():
            grid[int(v), int(u)] = val
        out[pop] = grid
    return out


def make_horse_race_figure(hr: pd.DataFrame) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    preds = ["arm_uncertainty_z", "dist_to_arm_z", "visit_recency_z", "q_arm_z"]
    labels = ["belief\nuncertainty", "distance\nto arm", "visit\nrecency", "beta\nvalue"]
    colors = ["#CC6677", "#999999", "#999999", "#4477AA"]
    fig, ax = plt.subplots(figsize=(6, 4.4))
    for i, p in enumerate(preds):
        v = hr[p].dropna()
        med = v.median()
        try:
            pv = wilcoxon(v).pvalue
        except ValueError:
            pv = np.nan
        lo, hi = np.percentile(v, [25, 75])
        ax.bar(i, med, width=0.62, color=colors[i], zorder=2)
        ax.errorbar(i, med, yerr=[[med - lo], [hi - med]], color="k", capsize=4, lw=1.2, zorder=3)
        star = "***" if pv < 1e-3 else ("*" if pv < 0.05 else "n.s.")
        ax.text(i, hi + 0.004, star, ha="center", fontsize=10)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(preds)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("within-arm content coefficient\n(z-scored predictors, comparable)")
    ax.set_title("Horse-race: off-current content is driven by UNCERTAINTY\n"
                 "(non-current leaf arms; bar=median session, whisker=IQR)")
    fig.tight_layout()
    path = OUTDIR / "off_arm_horse_race.png"
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def make_map_figure(grids: dict[str, np.ndarray]) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
    for ax, pop in zip(axes, ["non_SWR", "SWR"]):
        g = grids[pop]
        im = ax.imshow(g, origin="lower", cmap="viridis", aspect="equal")
        for v in range(3):
            for u in range(3):
                ax.text(u, v, f"{g[v, u]:.1f}", ha="center", va="center",
                        color="w" if g[v, u] < np.nanmean(g) else "k", fontsize=10)
        ax.set_xticks(range(3)); ax.set_xticklabels(["low", "mid", "high"])
        ax.set_yticks(range(3)); ax.set_yticklabels(["low", "mid", "high"])
        ax.set_xlabel("belief uncertainty tertile →")
        ax.set_ylabel("beta value tertile ↑")
        ax.set_title(f"{pop.replace('_', '-')}   content share (%)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Content climbs with UNCERTAINTY (→), flat across VALUE (↑): beta value ⟂ uncertainty",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUTDIR / "uncertainty_value_map.png"
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def make_figure(slopes_nc: pd.DataFrame, profile: pd.DataFrame) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Panel A: per-session within-arm slopes, non-current, SWR vs non-SWR.
    pops = ["non_SWR", "SWR"]
    colors = {"non_SWR": "#4477AA", "SWR": "#CC6677"}
    for i, pop in enumerate(pops):
        v = slopes_nc[slopes_nc.population == pop]["slope"].dropna()
        x = np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v))
        ax0.scatter(x, v, s=6, alpha=0.35, color=colors[pop], edgecolors="none")
        ax0.hlines(v.median(), i - 0.28, i + 0.28, color="k", lw=2, zorder=3)
    ax0.axhline(0, color="grey", lw=0.8, ls="--")
    ax0.set_xticks(range(len(pops)))
    ax0.set_xticklabels(["non-SWR", "SWR"])
    ax0.set_ylabel("within-arm content slope\n(Δ fraction / +1 SD uncertainty)")
    ax0.set_title("A  Same arm, higher uncertainty → more content\n(non-current leaf arms, one dot = session)")

    # Panel B: within-arm-demeaned content by uncertainty tertile.
    x = np.arange(3)
    w = 0.38
    for i, pop in enumerate(pops):
        ax1.bar(
            x + (i - 0.5) * w,
            profile.loc[pop, ["low", "mid", "high"]].to_numpy(),
            width=w,
            color=colors[pop],
            label=pop.replace("_", "-"),
        )
    ax1.axhline(0, color="grey", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["low", "mid", "high"])
    ax1.set_xlabel("within-arm uncertainty tertile")
    ax1.set_ylabel("within-arm content (demeaned fraction)")
    ax1.set_title("B  Monotonic in uncertainty")
    ax1.legend(frameon=False)

    fig.tight_layout()
    path = OUTDIR / "within_arm_uncertainty_content.png"
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def main() -> None:
    t = pickle.load(open(TABLE, "rb"))
    t["session"] = (
        t["animal"].astype(str) + "_" + t["day"].astype(str) + "_" + t["epoch"].astype(str)
    )
    leaf = t[t["arm"] <= 5].copy()  # leaf arms carry uncertainty; trunk = 0
    print(f"loaded {TABLE}: {t['event_id'].nunique()} events, "
          f"{t['session'].nunique()} sessions, {len(leaf)} leaf-arm rows")

    slopes_all = per_session_slopes(leaf, exclude_current=False)
    slopes_nc = per_session_slopes(leaf, exclude_current=True)
    slopes_nc_rec = per_session_slopes(
        leaf, exclude_current=True, controls=("visit_recency",)
    )

    summarize(slopes_all, "A. ALL leaf arms (includes current — expect confounded/weak)")
    summarize(slopes_nc, "B. NON-CURRENT leaf arms (headline: isolates uncertainty)")
    summarize(
        slopes_nc_rec,
        "C. NON-CURRENT + within-arm recency control (uncertainty net of recency)",
    )

    profile = binned_profile(leaf)
    print("\nWithin-arm-demeaned content by uncertainty tertile:")
    print(profile.round(5).to_string())

    fig_path = make_figure(slopes_nc, profile)
    print(f"\nfigure -> {fig_path}")

    binsc = raw_binscatter(leaf)
    tert_raw = within_arm_tertile_raw(leaf)
    print("\nRaw within-arm content share (%) by the arm's own uncertainty tertile:")
    print(tert_raw[[f"{t}_pct" for t in ["low", "mid", "high"]]].round(3).to_string())
    raw_path = make_raw_figure(binsc, tert_raw)
    print(f"raw figure -> {raw_path}")

    oc = leaf[leaf["is_current"] == 0].copy()
    hr = horse_race(oc)
    hr_path = make_horse_race_figure(hr)
    print(f"horse-race figure -> {hr_path}  (n_sess={len(hr)})")

    grids = value_uncertainty_map(oc)
    map_path = make_map_figure(grids)
    print(f"uncertainty x value map -> {map_path}")
    for pop, g in grids.items():
        print(f"  {pop} grid (rows=value low->high, cols=uncertainty low->high):\n{np.round(g,2)}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    slopes_nc.to_csv(OUTDIR / "within_arm_slopes_noncurrent.csv", index=False)


if __name__ == "__main__":
    main()
