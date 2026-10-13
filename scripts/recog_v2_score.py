"""H8 scoring: AUROC per arm x memory set x angle (v1 H4 definition), layout bootstrap, per-decoy AUROC, H8 gate,
and the offline inspection-rule analysis.

  uv run python scripts/recog_v2_score.py              # results/recog_v2_scores_test.parquet -> recog_v2_auroc.json, recog_v2_gate.json, figures
  uv run python scripts/recog_v2_score.py --train      # same analysis on the training layouts (design reference only)
  uv run python scripts/recog_v2_score.py --selfcheck
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recog_score import u_matrix, auroc_boot, catchment  # noqa: E402
from brain.mb_front import inspect_stat  # noqa: E402
from provenance import stamp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
THR, N_BOOT, GATE_MARGIN = 0.75, 2000, 0.05
PRIMARY = ("mb_front", "ms")
COMPARISONS = {   # name -> (a, b) or (a, [b1, b2] = max of the b's, re-taken in every bootstrap row)
    "gate_mb_front_ms_minus_best_pixels_ms": (PRIMARY, [("pixels", "ms"), ("pixels_crop", "ms")]),
    "mb_front_ms_minus_pixels_lf13_v1": (PRIMARY, [("pixels", "lf13")]),
    "mb_front_ms_minus_nn_front_ms": (PRIMARY, [("nn_front", "ms")]),
    "mb_front_ms_minus_mb_front_goal": (PRIMARY, [("mb_front", "goal")]),
    "mb_front_ms_minus_mb_front_lf13": (PRIMARY, [("mb_front", "lf13")]),
    "mb_front_ms_minus_mb_front_ms1": (PRIMARY, [("mb_front", "ms1")]),
    "mb_front_ms_minus_mb_v1_lf13": (PRIMARY, [("mb_v1", "lf13")]),
    "mb_front_ms_minus_mb_crop_ms": (PRIMARY, [("mb_crop", "ms")]),
    "mb_front_ms_minus_dino_patch_ms": (PRIMARY, [("dino_patch", "ms")]),
}


def pooled_auroc(g):
    """Mann-Whitney AUROC of target vs all swapped-decoy scores (ties 0.5)."""
    t = g.obj == "target"
    return float(mannwhitneyu(g.score[t], g.score[~t]).statistic / (t.sum() * (~t).sum()))


def analyse(df, n_boot=N_BOOT, seed=0):
    layouts, thetas = sorted(df.layout.unique()), sorted(df.theta.unique())
    L = len(layouts)
    C = np.vstack([np.ones(L), np.random.default_rng(seed).multinomial(L, np.ones(L) / L, size=n_boot)]).astype(float)
    g = {k: v.score.to_numpy() for k, v in df.groupby(["rec", "memset", "theta", "layout", df.obj == "target"])}
    sub = {k: v.score.to_numpy() for k, v in df.groupby(["rec", "memset", "theta", "layout", "obj"])}
    decoys = sorted(o for o in df.obj.unique() if o != "target")
    out, A = {"thetas": [int(t) for t in thetas], "layouts": [int(x) for x in layouts], "n_boot": n_boot, "arms": {}}, {}
    for (r, m) in sorted({(r, m) for r, m in df[["rec", "memset"]].drop_duplicates().itertuples(index=False)}):
        Us = [u_matrix([g[(r, m, t, l, True)] for l in layouts], [g[(r, m, t, l, False)] for l in layouts]) for t in thetas]
        p, n = Us[0][1], Us[0][2]
        a = auroc_boot(np.stack([u[0] for u in Us]), p, n, C)
        A[(r, m)] = a
        pdec = {}
        for dc in decoys:
            Ud = np.stack([u_matrix([g[(r, m, t, l, True)] for l in layouts], [sub[(r, m, t, l, dc)] for l in layouts])[0] for t in thetas])
            ad = auroc_boot(Ud, p, p.copy(), C)
            pdec[dc] = {"auroc": ad[0].round(6).tolist(), "ci95_lo": np.percentile(ad[1:], 2.5, 0).round(6).tolist(),
                        "ci95_hi": np.percentile(ad[1:], 97.5, 0).round(6).tolist()}
        out["arms"].setdefault(r, {})[m] = {
            "auroc": a[0].round(6).tolist(), "ci95_lo": np.percentile(a[1:], 2.5, 0).round(6).tolist(),
            "ci95_hi": np.percentile(a[1:], 97.5, 0).round(6).tolist(), "n_pos_per_angle": int(p.sum()), "n_neg_per_angle": int(n.sum()),
            "catchment_bins": int(catchment(a[0], THR)), "mean_auroc_out_of_arc": float(a[0][np.array(thetas) > 45].mean()),
            "mean_auroc_all": float(a[0].mean()), "per_decoy": pdec}
    comp = {}
    for name, (x, ys) in COMPARISONS.items():
        if x in A and all(y in A for y in ys):
            d = A[x] - np.max(np.stack([A[y] for y in ys]), 0)
            comp[name] = {"a": "@".join(x), "b": " max ".join("@".join(y) for y in ys), "by_theta": d[0].round(6).tolist(),
                          "ci95_lo": np.percentile(d[1:], 2.5, 0).round(6).tolist(), "ci95_hi": np.percentile(d[1:], 97.5, 0).round(6).tolist()}
    out["paired_differences"] = comp
    return out, A


def gate(out):
    a0 = out["arms"][PRIMARY[0]][PRIMARY[1]]["auroc"][0]
    d = out["paired_differences"]["gate_mb_front_ms_minus_best_pixels_ms"]
    ok1, ok2 = a0 >= THR, d["by_theta"][0] >= GATE_MARGIN
    return {"rule": f"H8 passes iff MB(front)+multi-scale LF AUROC at 0 deg >= {THR} AND it beats max(raw pixels, pixel crop) "
                    f"with the same memory by >= {GATE_MARGIN} at 0 deg (point estimates; CIs reported)",
            "auroc_0deg": a0, "auroc_0deg_ci95": [out["arms"][PRIMARY[0]][PRIMARY[1]]["ci95_lo"][0], out["arms"][PRIMARY[0]][PRIMARY[1]]["ci95_hi"][0]],
            "margin_over_pixels_0deg": d["by_theta"][0], "margin_ci95": [d["ci95_lo"][0], d["ci95_hi"][0]],
            "part1_auroc_ge_0.75": bool(ok1), "part2_margin_ge_0.05": bool(ok2), "H8_recognition": "PASS" if ok1 and ok2 else "FAIL"}


# ---------------- inspection rule (offline approach sequences, training layouts) ----------------
def seq_stats(df, sep_deg=20.0, last=3):
    """One row per (layout, obj, app): 'single' = mean familiarity of the last `last` frames (mission-like stop rule);
    'inspect' = inspect_stat over the inspection-arc frames."""
    rows = []
    for (l, o, a), s in df.sort_values("fr").groupby(["layout", "obj", "app"]):
        rows.append(dict(layout=l, obj=o, app=a, single=float(s.fam.to_numpy()[-last:].mean()),
                         inspect=float(inspect_stat(np.radians(s.th.to_numpy()), s.fam.to_numpy(), s.insp.to_numpy(), sep_deg))))
    return pd.DataFrame(rows)


def inspection_analysis(df, target_rate=0.9, calib=(0, 1, 2, 3, 4), n_boot=N_BOOT):
    res = {}
    for sep in (10.0, 20.0, 30.0):
        st = seq_stats(df, sep)
        cal, ev = st[st.layout.isin(calib)], st[~st.layout.isin(calib)]
        r = {}
        for rule in ("single", "inspect"):
            tv = np.sort(cal[cal.obj == "target"][rule].to_numpy())
            thr = float(tv[int(np.floor((1 - target_rate) * len(tv)))])          # >= target_rate of calibration targets accepted
            acc = lambda d: float((d[rule] >= thr).mean())  # noqa: E731
            r[rule] = {"thr": thr, "calib_target_accept": acc(cal[cal.obj == "target"]), "eval_target_accept": acc(ev[ev.obj == "target"]),
                       "eval_decoy_accept": acc(ev[ev.obj != "target"]),
                       "eval_decoy_accept_per_decoy": {dc: acc(ev[ev.obj == dc]) for dc in sorted(set(ev.obj) - {"target"})},
                       "auroc_all_layouts": pooled_auroc(st.assign(score=st[rule]))}
        # paired layout bootstrap of the AUROC difference (inspect - single), all 10 layouts
        layouts = sorted(st.layout.unique()); rs = np.random.default_rng(0)
        diffs = []
        for _ in range(n_boot):
            b = pd.concat([st[st.layout == l] for l in rs.choice(layouts, len(layouts))])
            diffs.append(pooled_auroc(b.assign(score=b.inspect)) - pooled_auroc(b.assign(score=b.single)))
        r["auroc_diff_inspect_minus_single"] = {"point": r["inspect"]["auroc_all_layouts"] - r["single"]["auroc_all_layouts"],
                                               "ci95": np.percentile(diffs, [2.5, 97.5]).round(6).tolist()}
        res[f"sep_{sep:g}deg"] = r
    s20 = res["sep_20deg"]
    return {"definition": f"Thresholds set on layouts {list(calib)} so that >= {target_rate:.0%} of target approaches are accepted; "
                          "acceptance rates measured on the other training layouts. Sequences: 8 approaches x 4 objects (target + 3 swapped decoys, "
                          "same camera path) per layout; familiarity = MB(front) with the multi-scale memory.",
            "summary": {"sep_deg": 20, "single_eval_target_accept": s20["single"]["eval_target_accept"], "single_eval_decoy_accept": s20["single"]["eval_decoy_accept"],
                        "inspect_eval_target_accept": s20["inspect"]["eval_target_accept"], "inspect_eval_decoy_accept": s20["inspect"]["eval_decoy_accept"],
                        "auroc_single": s20["single"]["auroc_all_layouts"], "auroc_inspect": s20["inspect"]["auroc_all_layouts"],
                        "auroc_diff": s20["auroc_diff_inspect_minus_single"]},
            "by_separation": res}


# ---------------- figures ----------------
COL = {"pixels": "#2a78d6", "pixels_crop": "#6250d6", "colour": "#eb6834", "mb_v1": "#7f7f7f", "mb_crop": "#a0522d",
       "mb_front": "#1baf7a", "nn_front": "#008300", "dino_patch": "#e87ba4"}
LABEL = {"pixels": "Raw pixels", "pixels_crop": "Pixels, object crop", "colour": "Colour histogram", "mb_v1": "v1 MB (pixels)",
         "mb_crop": "MB on pixel crop", "mb_front": "MB on insect front end", "nn_front": "Front end, nearest neighbour (no MB)",
         "dino_patch": "DINOv2-S patch-max"}


def figures(out, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    th = out["thetas"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
    for ax, ms, title in [(axes[0], "lf13", "v1 learning flight (13 views, 6 m, one sun)"), (axes[1], "ms", "Multi-scale learning flight (31 views, 3-12 m, 3 suns)")]:
        ax.axvspan(-2, 45, color="#888", alpha=0.10, lw=0)
        ax.axhline(0.5, color="#bbb", lw=1, ls=":"); ax.axhline(THR, color="#888", lw=1, ls="--")
        for r in ["pixels", "pixels_crop", "colour", "mb_v1", "mb_front", "nn_front", "dino_patch"]:
            if r not in out["arms"]:
                continue
            v = out["arms"][r][ms]
            ax.fill_between(th, v["ci95_lo"], v["ci95_hi"], color=COL[r], alpha=0.12, lw=0)
            ax.plot(th, v["auroc"], color=COL[r], lw=2.4 if r == "mb_front" else 1.6, marker="o", ms=3.5, label=LABEL[r])
        ax.set_title(title, fontsize=10); ax.set_xlabel("viewpoint change from the goal view (azimuth, °)")
        ax.set_xticks(range(0, 181, 30)); ax.set_xlim(-2, 182); ax.set_ylim(0.3, 1.01); ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("AUROC, target vs swapped decoy")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"H8: insect front end + multi-scale learning flight ({tag} layouts {out['layouts'][0]}-{out['layouts'][-1]}, 95% CI over layouts)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(RES, f"fig_recog_v2_auroc{'' if tag == 'test' else '_' + tag}.png"), dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    names = {"d1_tex": "Decoy 1: same tent, check pattern", "d2_mesh": "Decoy 2: dome tent, same fabric", "d3_twin": "Decoy 3: near-twin (door open)"}
    for ax, (dc, nm) in zip(axes, names.items()):
        ax.axhline(THR, color="#888", lw=1, ls="--"); ax.axhline(0.5, color="#bbb", lw=1, ls=":")
        for r in ["pixels", "pixels_crop", "colour", "mb_front", "nn_front", "dino_patch"]:
            if r in out["arms"]:
                ax.plot(th, out["arms"][r]["ms"]["per_decoy"][dc]["auroc"], color=COL[r], lw=2, marker="o", ms=3, label=LABEL[r])
        ax.set_title(nm, fontsize=9); ax.set_xticks(range(0, 181, 45)); ax.set_ylim(0.0, 1.01); ax.grid(alpha=0.25, lw=0.5); ax.set_xlabel("azimuth (°)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("AUROC (multi-scale memory)"); axes[2].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(RES, f"fig_recog_v2_per_decoy{'' if tag == 'test' else '_' + tag}.png"), dpi=150); plt.close(fig)


def inspect_figure(path=os.path.join(RES, "recog_v2_inspect_frames.parquet")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = seq_stats(pd.read_parquet(path))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for rule, c in (("single", "#2a78d6"), ("inspect", "#1baf7a")):
        t, d = st[st.obj == "target"][rule].to_numpy(), st[st.obj != "target"][rule].to_numpy()
        thr = np.unique(np.r_[t, d])[::-1]
        ax.plot([(d >= x).mean() for x in thr], [(t >= x).mean() for x in thr], color=c, lw=2,
                label={"single": "last 3 frames (mission-like)", "inspect": "inspection: 2 views ≥ 20° apart"}[rule])
    ax.plot([0, 1], [0, 1], ":", color="#bbb"); ax.axhline(0.9, color="#888", lw=1, ls="--")
    ax.set_xlabel("look-alike accepted (fraction of decoy approaches)"); ax.set_ylabel("target accepted")
    ax.set_title("Stop rules on approach sequences (training layouts 0-9)", fontsize=9.5); ax.legend(fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_recog_v2_inspect.png"), dpi=150); plt.close(fig)


def main(split):
    import pyarrow.parquet as pq
    path = os.path.join(RES, f"recog_v2_scores_{split}.parquet")
    df = pd.read_parquet(path)
    out, _ = analyse(df)
    st = stamp(seed=out["layouts"], param_source=f"H8 protocol (pre-registered); {os.path.basename(path)}")
    tagged = lambda n: n if split == "test" else n.replace(".json", f"_{split}.json")  # noqa: E731
    cost = json.load(open(os.path.join(RES, "recog_v2_cost.json"))) if os.path.exists(os.path.join(RES, "recog_v2_cost.json")) else None
    json.dump({**st, "scores_provenance": json.loads(pq.read_schema(path).metadata[b"provenance"]),
               "definition": "v1 H4 definition: Mann-Whitney AUROC, target probes vs swapped-decoy probes per azimuth bin, pooled over "
                             "distance (3, 6, 12 m), 2 random suns, 2 jitter reps and 10 layouts; 95% CI = percentile bootstrap over layouts "
                             f"({N_BOOT}); catchment = consecutive 15-deg bins from 0 with AUROC >= 0.75.", **out,
               "cost": cost}, open(os.path.join(RES, tagged("recog_v2_auroc.json")), "w"), indent=1)
    gt = gate(out)
    if split == "test":
        json.dump({**st, **gt, "source": "results/recog_v2_auroc.json"}, open(os.path.join(RES, "recog_v2_gate.json"), "w"), indent=1)
    figures(out, split)
    if os.path.exists(os.path.join(RES, "recog_v2_inspect_frames.parquet")):
        inspect_figure()
    print(f"{'arm@memset':22s} " + " ".join(f"{t:>6d}" for t in (0, 45, 90, 180)) + "  catch  outarc")
    for r, d in out["arms"].items():
        for m, v in d.items():
            a = v["auroc"]
            print(f"{r + '@' + m:22s} " + " ".join(f"{a[out['thetas'].index(t)]:6.3f}" for t in (0, 45, 90, 180)) + f"  {v['catchment_bins']:5d}  {v['mean_auroc_out_of_arc']:.3f}")
    for k, v in out["paired_differences"].items():
        print(f"{k:45s} 0deg {v['by_theta'][0]:+.3f} [{v['ci95_lo'][0]:+.3f}, {v['ci95_hi'][0]:+.3f}]")
    print(json.dumps(gt, indent=1))


def _selfcheck():
    rs = np.random.default_rng(1)
    rows = []
    for l in range(5):
        for t in range(0, 181, 15):
            for o in ["target", "d1_tex", "d2_mesh", "d3_twin"]:
                for k in range(6):
                    for rec, sep in (("mb_front", 3.0 if t <= 30 else 0.0), ("pixels", 0.0), ("pixels_crop", 0.0), ("nn_front", 0.0)):
                        for m in ("ms", "lf13", "goal", "ms1"):
                            rows.append(dict(rec=rec, memset=m, theta=t, layout=l, obj=o, score=rs.normal() + (sep if o == "target" else 0)))
    out, _ = analyse(pd.DataFrame(rows), n_boot=200)
    a = out["arms"]["mb_front"]["ms"]
    assert a["catchment_bins"] == 3 and a["auroc"][0] > 0.95 and abs(out["arms"]["pixels"]["ms"]["auroc"][6] - 0.5) < 0.1
    g = gate(out)
    assert g["H8_recognition"] == "PASS" and g["margin_over_pixels_0deg"] > 0.3
    d = pd.DataFrame(rows); s = d[(d.rec == "mb_front") & (d.memset == "ms") & (d.theta == 15)]
    assert abs(pooled_auroc(s) - a["auroc"][1]) < 1e-4
    # inspection analysis on synthetic sequences: decoys familiar from ONE viewpoint only -> inspection rejects them
    fr = []
    for l in range(10):
        for o in ["target", "d1_tex"]:
            for a_ in range(8):
                for i in range(16):
                    th = 0 if i < 7 else (i - 7) / 8 * 30
                    f = 0.8 + 0.05 * rs.normal() if o == "target" else (0.8 if i >= 13 else 0.3) + 0.05 * rs.normal()
                    fr.append(dict(layout=l, obj=o, app=a_, fr=i, th=th, insp=i >= 7, fam=f))
    r = inspection_analysis(pd.DataFrame(fr), n_boot=50)
    s = r["summary"]
    assert s["inspect_eval_decoy_accept"] < 0.1 < s["single_eval_decoy_accept"] and s["inspect_eval_target_accept"] >= 0.8, s
    print("recog_v2_score self-check ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main("train" if "--train" in sys.argv else "test")
