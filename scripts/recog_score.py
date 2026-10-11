"""H4 scoring: target-vs-swapped-decoy AUROC per recogniser x LF x angle, layout bootstrap CIs, catchments, KT4 gates.

Definitions are the pre-registered ones (docs/LOG.md "[WS-B] H4 protocol (pre-data)").
  uv run python scripts/recog_score.py              # results/recog_scores.parquet -> recog_auroc.json, recog_gate.json, fig_recog_*.png
  uv run python scripts/recog_score.py --selfcheck  # synthetic data with a known answer
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import stamp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
RECS = ["pixels", "colour", "mb", "dino_cls", "dino_patch", "mb_edge", "pixels_crop", "mb_crop"]
PREREG = RECS[:5]                     # mb_edge pre-registered as exploratory; *_crop added post hoc (LOG), exploratory
THR, THR_SENS, N_BOOT = 0.75, 0.8, 2000


def u_matrix(pos, neg):
    """pos/neg: lists over layouts of score arrays -> U[i, j] = sum over pos_i x neg_j of [a > b] + 0.5 [a == b]."""
    L = len(pos)
    U = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            d = pos[i][:, None] - neg[j][None]
            U[i, j] = (d > 0).sum() + 0.5 * (d == 0).sum()
    return U, np.array([len(p) for p in pos]), np.array([len(n) for n in neg])


def auroc_boot(U, p, n, C):
    """U [T, L, L]; C [B, L] layout multiplicities -> AUROC [B, T] (pooled over the resampled layouts)."""
    return np.einsum("bi,tij,bj->bt", C, U, C) / ((C @ p) * (C @ n))[:, None]


def catchment(a, thr):
    """a [..., T] AUROC by angle bin -> number of consecutive bins from 0 deg with AUROC >= thr."""
    ok = a >= thr
    return np.where(ok.all(-1), ok.shape[-1], np.argmin(ok, -1))


def analyse(df, n_boot=N_BOOT, seed=0):
    layouts = sorted(df.layout.unique())
    thetas = sorted(df.theta.unique())
    L = len(layouts)
    C = np.random.default_rng(seed).multinomial(L, np.ones(L) / L, size=n_boot).astype(float)
    C = np.vstack([np.ones(L), C])                                    # row 0 = point estimate
    g = {k: v.score.to_numpy() for k, v in df.groupby(["rec", "lf", "theta", "layout", df.obj == "target"])}
    sub = {k: v.score.to_numpy() for k, v in df.groupby(["rec", "lf", "theta", "layout", "obj"])}
    sub_d = {k: v.score.to_numpy() for k, v in df.groupby(["rec", "lf", "theta", "layout", "dist", df.obj == "target"])}
    decoys = sorted(o for o in df.obj.unique() if o != "target")
    dists = sorted(df.dist.unique())
    recs = [r for r in RECS if r in set(df.rec)]
    A, out = {}, {"thetas": [int(t) for t in thetas], "layouts": [int(x) for x in layouts], "recognisers": {}}
    for r in recs:
        for lf in (0, 1):
            Us = [u_matrix([g[(r, lf, t, l, True)] for l in layouts], [g[(r, lf, t, l, False)] for l in layouts]) for t in thetas]
            U = np.stack([u[0] for u in Us]); p, n = Us[0][1], Us[0][2]
            a = auroc_boot(U, p, n, C)                                  # [B+1, T]
            A[(r, lf)] = a
            ci = np.percentile(a[1:], [2.5, 97.5], axis=0)
            cat = catchment(a, THR)
            per_decoy = {}
            for dc in decoys:
                Ud = np.stack([u_matrix([g[(r, lf, t, l, True)] for l in layouts], [sub[(r, lf, t, l, dc)] for l in layouts])[0] for t in thetas])
                ad = auroc_boot(Ud, p, p.copy(), C)
                per_decoy[dc] = {"auroc": ad[0].round(4).tolist(), "ci95": np.percentile(ad[1:], [2.5, 97.5], 0).round(4).tolist()}
            per_dist = {}
            for d in dists:
                Uu = [u_matrix([sub_d[(r, lf, t, l, d, True)] for l in layouts], [sub_d[(r, lf, t, l, d, False)] for l in layouts]) for t in thetas]
                ad = auroc_boot(np.stack([u[0] for u in Uu]), Uu[0][1], Uu[0][2], C[:1])
                per_dist[f"{d:g}m"] = ad[0].round(4).tolist()
            out["recognisers"].setdefault(r, {})["lf_on" if lf else "lf_off"] = {
                "auroc": a[0].round(4).tolist(), "ci95_lo": ci[0].round(4).tolist(), "ci95_hi": ci[1].round(4).tolist(),
                "n_pos_per_angle": int(p.sum()), "n_neg_per_angle": int(n.sum()),
                "catchment_bins": int(cat[0]), "catchment_bins_ci95": np.percentile(cat[1:], [2.5, 97.5]).tolist(),
                "catchment_bins_at_0.8": int(catchment(a[0], THR_SENS)),
                "mean_auroc_out_of_arc": float(a[0][np.array(thetas) > 45].mean()), "mean_auroc_all": float(a[0].mean()),
                "per_decoy": per_decoy, "per_distance_point": per_dist}
    # paired differences (same bootstrap rows)
    out_arc = np.array(thetas) > 45
    diff = {}
    for r in recs:
        lfgain = A[(r, 1)][:, out_arc].mean(1) - A[(r, 0)][:, out_arc].mean(1)
        on, off = catchment(A[(r, 1)], THR), catchment(A[(r, 0)], THR)
        ratio = np.where(off > 0, on / np.maximum(off, 1), np.where(on > 0, np.inf, 0.0))   # 0/0 = no widening
        diff[r] = {"lf_gain_out_of_arc_mean_auroc": float(lfgain[0]), "ci95": np.percentile(lfgain[1:], [2.5, 97.5]).round(4).tolist(),
                   "widening_ratio": float(ratio[0]), "catchment_bins_off_on": [int(off[0]), int(on[0])],
                   "widening_ratio_ci95_finite": np.percentile(ratio[1:][np.isfinite(ratio[1:])], [2.5, 97.5]).tolist() if np.isfinite(ratio[1:]).any() else None,
                   "boot_frac_off_catchment_zero": float((off[1:] == 0).mean())}
    for lf in (0, 1):
        d = A[("mb", lf)] - A[("pixels", lf)]
        diff[f"mb_minus_pixels_lf{lf}"] = {"by_angle": d[0].round(4).tolist(), "ci95_lo": np.percentile(d[1:], 2.5, 0).round(4).tolist(),
                                           "ci95_hi": np.percentile(d[1:], 97.5, 0).round(4).tolist()}
    out["paired"] = diff
    # EXPLORATORY post hoc: paired win rate, target vs its own swapped decoy under the identical camera + sun (ties 0.5)
    key = ["layout", "sun", "theta", "dist", "rep", "rec", "lf"]
    t = df[df.obj == "target"].set_index(key).score
    win = {}
    for dc in decoys:
        w = (t - df[df.obj == dc].set_index(key).score).dropna()
        w = (w > 0) + 0.5 * (w == 0)
        win[dc] = {f"{r}_lf{lf}": round(float(v), 4) for (r, lf), v in w.groupby(level=["rec", "lf"]).mean().items()}
    out["paired_win_rate_exploratory"] = win
    return out, A


def gates(out):
    R = out["recognisers"]
    mb_off, mb_on, px_on, px_off = (R[r][k]["auroc"] for r, k in [("mb", "lf_off"), ("mb", "lf_on"), ("pixels", "lf_on"), ("pixels", "lf_off")])
    ratio = out["paired"]["mb"]["widening_ratio"]
    g = {
        "a_mb_below_0.75_at_0deg": {"fires": mb_off[0] < THR, "value": mb_off[0], "rule": "MB (LF off) AUROC at 0 deg < 0.75"},
        "b_mb_not_above_pixels": {"fires": bool(all(m - p <= 0.03 for m, p in zip(mb_on, px_on))),
                                  "max_mb_minus_pixels_lf_on": max(m - p for m, p in zip(mb_on, px_on)),
                                  "max_mb_minus_pixels_lf_off": max(m - p for m, p in zip(mb_off, px_off)),
                                  "fires_lf_off_version": bool(all(m - p <= 0.03 for m, p in zip(mb_off, px_off))),
                                  "rule": "MB+LF minus pixels+LF <= 0.03 at every angle"},
        "c_lf_widening_below_1.5": {"fires": not (ratio >= 1.5), "value": ratio,
                                    "catchment_off": R["mb"]["lf_off"]["catchment_bins"], "catchment_on": R["mb"]["lf_on"]["catchment_bins"],
                                    "rule": "MB catchment(LF on)/catchment(LF off) < 1.5"},
        "d_colour_ge_mb": {"fires": R["colour"]["lf_on"]["mean_auroc_all"] >= R["mb"]["lf_on"]["mean_auroc_all"],
                           "colour_lf_on": R["colour"]["lf_on"]["mean_auroc_all"], "mb_lf_on": R["mb"]["lf_on"]["mean_auroc_all"],
                           "colour_lf_off": R["colour"]["lf_off"]["mean_auroc_all"], "mb_lf_off": R["mb"]["lf_off"]["mean_auroc_all"],
                           "rule": "colour+LF mean AUROC over 13 angles >= MB+LF"}}
    for v in g.values():
        v["fires"] = bool(v["fires"])
    killed = any(v["fires"] for v in g.values())
    return {"gates": g, "learning_flight_claim": "KILLED" if killed else "SURVIVES",
            "h4_mb_ge_0.75_at_0deg": bool(mb_off[0] >= THR), "h4_widening_ge_1.5": bool(ratio >= 1.5)}


COL = {"pixels": "#2a78d6", "colour": "#eb6834", "mb": "#1baf7a", "dino_cls": "#eda100", "dino_patch": "#e87ba4", "mb_edge": "#008300",
       "pixels_crop": "#6250d6", "mb_crop": "#e34948"}
LABEL = {"pixels": "Raw pixels", "colour": "Colour histogram", "mb": "Mushroom body (pixels)", "dino_cls": "DINOv2-S CLS",
         "dino_patch": "DINOv2-S patch-max", "mb_edge": "MB on edges (exploratory)",
         "pixels_crop": "Pixels, oracle object crop (post hoc)", "mb_crop": "MB, oracle object crop (post hoc)"}


def human_bytes(b):
    return f"{b / 1e6:.0f} MB" if b >= 1e6 else (f"{b / 1e3:.1f} kB" if b >= 1e3 else f"{b} B")


def figures(out, cost):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    th = out["thetas"]
    _auroc_fig(plt, out, cost, th, PREREG + ["mb_edge"], "fig_recog_auroc.png",
               "H4: recognising the tent against look-alikes on the same spot (10 forest layouts, 95% CI over layouts)")
    _auroc_fig(plt, out, cost, th, ["pixels", "mb", "pixels_crop", "mb_crop", "dino_patch"], "fig_recog_exploratory.png",
               "EXPLORATORY (post hoc): does an oracle crop around the object rescue pixels / MB?")
    _per_decoy(plt, out, th)


def _auroc_fig(plt, out, cost, th, recs, fname, suptitle):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, key, title in [(axes[0], "lf_off", "Goal view only (one 0° view at 6 m)"), (axes[1], "lf_on", "Goal view + 10 s learning flight (±45°, 12 views)")]:
        if key == "lf_on":
            ax.axvspan(-2, 45, color="#888", alpha=0.10, lw=0)
            ax.text(22, 0.985, "flown arc", ha="center", fontsize=8, color="#555")
        ax.axhline(0.5, color="#bbb", lw=1, ls=":")
        ax.axhline(THR, color="#888", lw=1, ls="--")
        for r in recs:
            v = out["recognisers"][r][key]
            sb = cost["stored_goal_bytes"][key][r] if cost else None
            lab = LABEL[r] + (f" ({human_bytes(sb)}" + (f" + {human_bytes(cost['model_bytes'][r])} model)" if cost["model_bytes"][r] else ")") if cost else "")
            ls = "--" if r in ("mb_edge", "pixels_crop", "mb_crop") else "-"
            ax.fill_between(th, v["ci95_lo"], v["ci95_hi"], color=COL[r], alpha=0.12, lw=0)
            ax.plot(th, v["auroc"], ls, color=COL[r], lw=2, marker="o", ms=4, label=lab)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("viewpoint change from the goal view (azimuth, °)")
        ax.set_xticks(range(0, 181, 30)); ax.set_xlim(-2, 182); ax.set_ylim(0.3, 1.01)
        ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("AUROC, target vs swapped decoy")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=3, frameon=False, title="stored memory with the learning flight", title_fontsize=8)
    fig.suptitle(suptitle, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(os.path.join(RES, fname), dpi=150)
    plt.close(fig)


def _per_decoy(plt, out, th):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    names = {"d1_tex": "Decoy 1: same tent, different pattern", "d2_mesh": "Decoy 2: dome tent, same fabric", "d3_twin": "Decoy 3: near-twin (door open)"}
    for ax, (dc, nm) in zip(axes, names.items()):
        ax.axhline(THR, color="#888", lw=1, ls="--")
        for r in ["pixels", "colour", "mb", "dino_cls", "dino_patch"]:
            v = out["recognisers"][r]["lf_on"]["per_decoy"][dc]
            ax.plot(th, v["auroc"], color=COL[r], lw=2, marker="o", ms=3, label=LABEL[r])
        ax.set_title(nm, fontsize=9); ax.set_xticks(range(0, 181, 45)); ax.set_ylim(0.2, 1.01); ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlabel("azimuth (°)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("AUROC (learning flight on)")
    axes[2].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_recog_per_decoy.png"), dpi=150)
    plt.close(fig)


def main():
    df = pd.read_parquet(os.path.join(RES, "recog_scores.parquet"))
    import pyarrow.parquet as pq
    prov = json.loads(pq.read_schema(os.path.join(RES, "recog_scores.parquet")).metadata[b"provenance"])
    out, _ = analyse(df)
    cost = json.load(open(os.path.join(RES, "recog_cost.json")))
    st = stamp(seed=out["layouts"], param_source="docs/LOG.md [WS-B] H4 protocol (pre-data); results/recog_scores.parquet")
    json.dump({**st, "scores_provenance": prov, "definition": "Mann-Whitney AUROC, target probes vs swapped-decoy probes per azimuth bin, pooled over "
               "distance (3,6,12 m), 2 random suns, 2 jitter reps and 10 layouts; 95% CI = percentile bootstrap over layouts (2000). "
               "Catchment = consecutive 15-deg bins from 0 with AUROC >= 0.75.", **out,
               "bytes": {k: cost[k] for k in ("stored_goal_bytes", "model_bytes", "notes")},
               "ms_per_query_median": cost["ms_per_query_median"], "ms_note": cost["ms_note"]},
              open(os.path.join(RES, "recog_auroc.json"), "w"), indent=1)
    gt = gates(out)
    json.dump({**st, **gt, "source": "results/recog_auroc.json"}, open(os.path.join(RES, "recog_gate.json"), "w"), indent=1)
    figures(out, cost)
    R = out["recognisers"]
    print(f"{'rec':12s} {'LF':3s} " + " ".join(f"{t:>6d}" for t in (0, 45, 90, 180)) + "  catch  outarc")
    for r in R:
        for k in ("lf_off", "lf_on"):
            v = R[r][k]; a = v["auroc"]
            print(f"{r:12s} {k[3:]:3s} " + " ".join(f"{a[out['thetas'].index(t)]:6.3f}" for t in (0, 45, 90, 180)) + f"  {v['catchment_bins']:5d}  {v['mean_auroc_out_of_arc']:.3f}")
    print(json.dumps(gt, indent=1))


def _selfcheck():
    rs = np.random.default_rng(1)
    rows = []
    for l in range(6):
        for t in range(0, 181, 15):
            for o in ["target", "d1_tex", "d2_mesh", "d3_twin"]:
                for k in range(8):
                    for lf in (0, 1):
                        sep = 3.0 if (t <= 30 or (lf and t <= 60)) else 0.0   # known: catchment 3 bins off, 5 on
                        rows.append(dict(rec="mb", lf=lf, theta=t, layout=l, obj=o, dist=6.0, sun=0, rep=k, score=rs.normal() + (sep if o == "target" else 0)))
                        rows.append(dict(rec="pixels", lf=lf, theta=t, layout=l, obj=o, dist=6.0, sun=0, rep=k, score=rs.normal()))
                        rows.append(dict(rec="colour", lf=lf, theta=t, layout=l, obj=o, dist=6.0, sun=0, rep=k, score=rs.normal()))
    out, A = analyse(pd.DataFrame(rows), n_boot=200)
    R = out["recognisers"]
    assert R["mb"]["lf_off"]["catchment_bins"] == 3 and R["mb"]["lf_on"]["catchment_bins"] == 5, (R["mb"]["lf_off"]["catchment_bins"], R["mb"]["lf_on"]["catchment_bins"])
    assert R["mb"]["lf_off"]["auroc"][0] > 0.95 and abs(R["pixels"]["lf_off"]["auroc"][5] - 0.5) < 0.1
    lo, hi = np.array(R["pixels"]["lf_on"]["ci95_lo"]), np.array(R["pixels"]["lf_on"]["ci95_hi"])
    assert ((lo < 0.5) & (0.5 < hi)).mean() > 0.75 and (hi - lo).min() > 0.02    # null recogniser: CIs mostly cover 0.5
    # pooled AUROC from U matrices == direct Mann-Whitney
    from scipy.stats import mannwhitneyu
    d = pd.DataFrame(rows); s = d[(d.rec == "mb") & (d.lf == 0) & (d.theta == 45)]
    u = mannwhitneyu(s[s.obj == "target"].score, s[s.obj != "target"].score).statistic / ((s.obj == "target").sum() * (s.obj != "target").sum())
    assert abs(u - R["mb"]["lf_off"]["auroc"][3]) < 1e-4
    g = gates(out)
    assert g["gates"]["c_lf_widening_below_1.5"]["fires"] is False and abs(g["gates"]["c_lf_widening_below_1.5"]["value"] - 5 / 3) < 1e-9
    assert g["gates"]["a_mb_below_0.75_at_0deg"]["fires"] is False and g["gates"]["b_mb_not_above_pixels"]["fires"] is False
    out["paired"]["mb"]["widening_ratio"] = 0.0                                 # both catchments 0 -> no widening -> gate (c) fires
    assert gates(out)["gates"]["c_lf_widening_below_1.5"]["fires"] is True
    assert catchment(np.array([0.8, 0.9, 0.7, 0.9]), 0.75) == 2 and catchment(np.array([0.7, 0.9]), 0.75) == 0 and catchment(np.array([0.8, 0.8]), 0.75) == 2
    print("recog_score self-check ok")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
