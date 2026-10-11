"""3D mission experiments: equal-budget tuning, H1 factorial on test seeds, H3 stubbornness sweep in 3D.

Stages (each shardable; units are independent and written to results/3d/<stage>/<unit>.parquet):
  r1    64 random configs per arm (same draws as KT2) x train seeds 0-4 x all conditions
  r2    top 8 per arm from r1 x train seeds 0-39 x all conditions          -> results/tuned3d_params.json (frozen)
  test  6 arms (frozen params) x test seeds 1000-1299 x all conditions    -> results/mission_h1.json
  h3    KT3 sweeps (ring gain, SM timeout, accumulator leak, 3 single arms) x test seeds x {occlusion 10 s,
        decoy similarity 0.8}                                             -> results/mission_h3.json
Usage:  run_3d.py run <stage> --shard i --nshards n [--procs p]   |   run_3d.py reduce <stage>   |   run_3d.py --selfcheck
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brain.arbiters import ARBITERS, CLASSES, sample_params
from kt2_five_arms import N_CONFIGS, SEARCH_SEED
from kt3_stubbornness import SWEEPS, pareto_interp, dominance_frontier, verdict
from mission import run_batch
from provenance import stamp
from scenario import sample_scenario
from stats import boot_idx, holm, paired_bootstrap, paired_diff

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/3d"
WORLD = {"landscape": "forest", "obstacle_density": 0.3}
FAULT_SEV = {"occlusion": [0, 1 / 3, 2 / 3, 1], "decoy": [0, 1 / 3, 2 / 3, 1], "compass_half": [1.0],
             "eye_columns": [0, 1 / 3, 2 / 3, 1], "false_sun": [0, 1 / 3, 2 / 3, 1],
             "speed_scale": [0.25, 0.5, 0.75, 1.0], "gust": [0.1, 0.5, 0.9]}
CONDS = [("none", None)] + [(f, s) for f, ss in FAULT_SEV.items() for s in ss]      # 25 conditions
R1_SEEDS, R2_SEEDS, R2_TOP = range(0, 5), range(0, 40), 8
TEST_SEEDS = range(1000, 1300)
H3_PROTOCOLS = {"occlusion": ("occlusion", 1.0), "decoy": ("decoy", 2 / 3)}
CHUNK = 250                                                   # max rows per run_batch call


def configs(arm):
    """The same 64 random configs per arm as KT2 (numpy default_rng(SEARCH_SEED))."""
    names = list(CLASSES[arm].PARAM_SPACE)
    U = np.random.default_rng(SEARCH_SEED).random((N_CONFIGS, 16))[:, :len(names)]
    return [sample_params(arm, u) for u in U]


def scen(arm, params, seed, split, fault, sev):
    p = dict(params)
    g = float(p.pop("ring_gain", 0.0)) if arm == "ring" else 0.0
    s = sample_scenario(seed, split=split, fault=fault, severity=sev, arbiter=arm, ring_gain=g, params=p, **WORLD)
    s["_meta"]["split"] = split
    return s


def units(stage):
    """-> list of (unit_id, [scenarios]) for a stage. Deterministic order."""
    out = []
    if stage == "r1":
        for arm in ARBITERS:
            for i, p in enumerate(configs(arm)):
                out.append((f"{arm}__c{i:02d}", [scen(arm, p, s, "train", f, v) for s in R1_SEEDS for f, v in CONDS]))
    elif stage == "r2":
        top = json.loads((ROOT / "results/tune3d_r1.json").read_text())["top"]
        for arm in ARBITERS:
            cf = configs(arm)
            for i in top[arm]:
                for k in range(0, len(R2_SEEDS), 10):
                    seeds = list(R2_SEEDS)[k:k + 10]
                    out.append((f"{arm}__c{i:02d}__s{k:02d}", [scen(arm, cf[i], s, "train", f, v) for s in seeds for f, v in CONDS]))
    elif stage == "test":
        tuned = json.loads((ROOT / "results/tuned3d_params.json").read_text())["arms"]
        for arm in ARBITERS:
            for ci, (f, v) in enumerate(CONDS):
                for k in (0, 150):
                    seeds = list(TEST_SEEDS)[k:k + 150]
                    out.append((f"{arm}__k{ci:02d}__s{k:03d}", [scen(arm, tuned[arm]["params"], s, "test", f, v) for s in seeds]))
    elif stage == "h3":
        tuned = json.loads((ROOT / "results/tuned3d_params.json").read_text())["arms"]
        for arm, (knob, vals) in SWEEPS.items():
            for j, val in enumerate(vals):
                p = dict(tuned[arm]["params"]) | ({knob: float(val)} if knob else {})
                for proto, (f, v) in H3_PROTOCOLS.items():
                    for k in (0, 150):
                        seeds = list(TEST_SEEDS)[k:k + 150]
                        out.append((f"{arm}__p{j:02d}__{proto}__s{k:03d}", [scen(arm, p, s, "test", f, v) for s in seeds]))
    else:
        raise SystemExit(f"unknown stage {stage}")
    return out


def run_unit(args):
    stage, uid, scs = args
    path = OUT / stage / f"{uid}.parquet"
    if path.exists():
        return uid, 0.0
    t0 = time.time()
    df = pd.concat([run_batch(scs[i:i + CHUNK]) for i in range(0, len(scs), CHUNK)], ignore_index=True)
    df["unit"] = uid
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path.with_suffix(".tmp"))
    os.replace(path.with_suffix(".tmp"), path)
    return uid, time.time() - t0


def run(stage, shard, nshards, procs):
    us = units(stage)[shard::nshards]
    print(f"{stage} shard {shard}/{nshards}: {len(us)} units, {sum(len(s) for _, s in us)} missions", flush=True)
    with ProcessPoolExecutor(max_workers=procs) as ex:
        for uid, dt in ex.map(run_unit, [(stage, u, s) for u, s in us]):
            print(f"  {uid} {dt:.0f}s", flush=True)


def load(stage):
    fs = sorted((OUT / stage).glob("*.parquet"))
    assert fs, f"no results for {stage}"
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def success(df):
    return (df.success_target & df.success_home & (df.collisions == 0)).astype(float)


def auc_table(df):
    """df rows of one arm/config -> per-seed equal-weight AUC over the 7 faults (KT2 definition: trapezoid over
    severity levels with 'none' at x = 0, x scaled to [0, 1])."""
    df = df.assign(ok=success(df))
    piv = {}
    for (f, v), g in df.groupby(["fault", df.severity.fillna(-1.0)]):
        piv[(f, None if v < 0 else v)] = g.set_index("seed").ok
    base = piv[("none", None)]
    per = {}
    for f, ss in FAULT_SEV.items():
        ys = [base] + [piv[(f, v)].reindex(base.index) for v in ss]
        x = np.linspace(0, 1, len(ys))
        per[f] = sum((ys[i] + ys[i + 1]) / 2 * (x[i + 1] - x[i]) for i in range(len(ys) - 1))
    return pd.DataFrame(per)                                   # index seed, columns faults


def cfg_index(uid):
    return int(uid.split("__")[1][1:])


def reduce(stage):
    df = load(stage)
    st = stamp(seed=sorted(df.seed.unique().tolist())[:3] + ["..."], param_source=f"results/3d/{stage}")
    if stage in ("r1", "r2"):
        df["cfg"] = df.unit.map(cfg_index)
        scores = {arm: {int(c): float(auc_table(g).mean(1).mean()) for c, g in df[df.arbiter == arm].groupby("cfg")} for arm in ARBITERS}
        if stage == "r1":
            top = {a: [c for c, _ in sorted(s.items(), key=lambda kv: (-kv[1], kv[0]))[:R2_TOP]] for a, s in scores.items()}
            res = {"stage": "r1", "seeds": "train 0-4", "conds": len(CONDS), "scores": scores, "top": top, "provenance": st}
            (ROOT / "results/tune3d_r1.json").write_text(json.dumps(res, indent=1))
        else:
            arms = {}
            for a, s in scores.items():
                c = sorted(s.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                arms[a] = {"params": configs(a)[c], "config_index": c, "train_auc": s[c], "rung2": s, "n_params": len(CLASSES[a].PARAM_SPACE)}
            res = {"stage": "r2 (frozen before any test seed)", "budget": f"{N_CONFIGS} configs (default_rng({SEARCH_SEED})); rung 1 train seeds 0-4; "
                   f"rung 2 top {R2_TOP} on train seeds 0-39; {len(CONDS)} conditions; same for every arm",
                   "objective": "mean over 7 faults of mean-over-seeds success-vs-severity AUC", "world": WORLD, "arms": arms, "provenance": st}
            (ROOT / "results/tuned3d_params.json").write_text(json.dumps(res, indent=1))
        for a in ARBITERS:
            print(a, sorted(scores[a].values())[-3:])
    elif stage == "test":
        df["split"] = "test"
        (ROOT / "results/runs").mkdir(exist_ok=True)
        df.drop(columns=["unit"]).to_parquet(ROOT / "results/runs/mission_test.parquet")
        auc = {a: auc_table(df[df.arbiter == a]) for a in ARBITERS}
        agg = {a: auc[a].mean(1).sort_index().to_numpy() for a in ARBITERS}
        others = [a for a in ARBITERS if a != "ring"]
        best = paired_bootstrap(lambda r, *o: r.mean() - max(x.mean() for x in o), agg["ring"], *[agg[a] for a in others])
        pair = {a: paired_diff(agg["ring"], agg[a]) for a in others}
        adj = holm([pair[a]["p"] for a in others])
        for a, pa in zip(others, adj):
            pair[a]["p_holm"] = pa
        ok = df.assign(ok=success(df))
        curves = {a: {f: [{"severity": 0.0 if v is None else float(v), "success": float(ok[(ok.arbiter == a) & ((ok.fault == "none") if v is None else ((ok.fault == f) & np.isclose(ok.severity.fillna(-1), v)))].ok.mean())}
                          for v in [None] + FAULT_SEV[f]] for f in FAULT_SEV} for a in ARBITERS}
        fault_auc = {a: {f: float(auc[a][f].mean()) for f in FAULT_SEV} for a in ARBITERS}
        res = {"test": "H1 mission factorial (3D)", "seeds": "test 1000-1299", "n_seeds": len(agg["ring"]), "conds": len(CONDS), "world": WORLD,
               "auc": {a: {"mean": float(agg[a].mean()), "ci": [float(x) for x in np.quantile(
                   [agg[a][i].mean() for i in boot_idx(len(agg[a]))], [0.025, 0.975])]} for a in ARBITERS},
               "ring_minus_best_other": best, "ring_vs": pair, "fault_auc": fault_auc,
               "worst_fault": {a: min(fault_auc[a], key=fault_auc[a].get) for a in ARBITERS},
               "success_curves": curves,
               "rates": {a: {k: float(df[df.arbiter == a][k].mean()) for k in ("success_target", "success_home", "wrong_object_stop")} |
                         {"collision_rate": float((df[df.arbiter == a].collisions > 0).mean()), "success": float(success(df[df.arbiter == a]).mean()),
                          "median_time_s": float(df[df.arbiter == a].time_s.median())} for a in ARBITERS},
               "h1_rule": "H1 holds if the 95% CI of ring minus best other excludes 0 (two-sided)",
               "provenance": st | {"tuned_params": "results/tuned3d_params.json"}}
        (ROOT / "results/mission_h1.json").write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps({a: res["auc"][a] for a in ARBITERS}), "\nring - best other", best)
    elif stage == "h3":
        df["arm_pt"] = df.unit.str.split("__").str[1].str[1:].astype(int)
        df["proto"] = df.unit.str.split("__").str[2]
        seeds = sorted(df.seed.unique())
        H, R, pts = {}, {}, {}
        for arm, (knob, vals) in SWEEPS.items():
            hs, rs = [], []
            for j, _v in enumerate(vals):
                d = df[(df.arbiter == arm) & (df.arm_pt == j)]
                hs.append(d[d.proto == "occlusion"].set_index("seed").t_hold_occlusion_s.reindex(seeds).to_numpy())
                rs.append(d[d.proto == "decoy"].set_index("seed").t_release_decoy_s.reindex(seeds).to_numpy())
            H[arm], R[arm] = np.array(hs), np.array(rs)
        mean = lambda D, idx=None: {a: np.nanmean(v if idx is None else v[:, idx], -1) for a, v in D.items()}

        def deltas(mh, mr, frontier):
            return {rv: frontier(mh[rv], mr[rv])(mh["ring"]) - mr["ring"] for rv in ("state_machine", "accumulator")}
        out = {}
        for name, fr, nb in (("preregistered", pareto_interp, 10_000), ("posthoc_dominance", dominance_frontier, 2_000)):
            est = deltas(mean(H), mean(R), fr)
            bs = {rv: [] for rv in est}
            for idx in boot_idx(len(seeds), n_boot=nb, seed=0 if nb == 10_000 else 1):
                d = deltas(mean(H, idx), mean(R, idx), fr)
                for rv in d:
                    bs[rv].append(d[rv])
            ci = {rv: np.nanquantile(np.array(b), [0.025, 0.975], axis=0) for rv, b in bs.items()}
            gate, dead = verdict(est, {rv: np.where(np.isnan(est[rv]), np.nan, ci[rv][0]) for rv in est})
            gains = SWEEPS["ring"][1]
            out[name] = {"gate": gate, "dead_because": dead, "delta": {rv: [
                {"ring_gain": float(g), "delta_s": None if np.isnan(e) else float(e), "lo": None if np.isnan(e) else float(l),
                 "hi": None if np.isnan(e) else float(h)} for g, e, l, h in zip(gains, est[rv], ci[rv][0], ci[rv][1])] for rv in est}}
        from scipy.stats import spearmanr
        mh, mr = mean(H), mean(R)
        res = {"test": "H3 stubbornness frontier in the 3D mission", "seeds": "test 1000-1299", "world": WORLD,
               "protocols": {"hold": "occlusion 10 s (severity 1)", "release": "swapped decoy, similarity 0.8 (severity 2/3)"},
               "points": {a: [{"knob": SWEEPS[a][0], "value": None if v is None else float(v), "hold_mean_s": float(mh[a][j]),
                               "release_mean_s": float(mr[a][j]), "n_hold": int(np.isfinite(H[a][j]).sum()), "n_release": int(np.isfinite(R[a][j]).sum()),
                               "hold_ci": [float(x) for x in np.nanquantile([np.nanmean(H[a][j][i]) for i in boot_idx(len(seeds), 2000)], [.025, .975])],
                               "release_ci": [float(x) for x in np.nanquantile([np.nanmean(R[a][j][i]) for i in boot_idx(len(seeds), 2000)], [.025, .975])]}
                              for j, v in enumerate(SWEEPS[a][1])] for a in SWEEPS},
               "spearman_vs_gain": {"hold": float(spearmanr(SWEEPS["ring"][1], mh["ring"])[0]),
                                    "release": float(spearmanr(SWEEPS["ring"][1], mr["ring"])[0])},
               "ring_vs_rival": out, "rule": "same as KT3 (ALIVE needs >= 2 adjacent ring gains with lower CI of both deltas > 0)",
               "provenance": st | {"tuned_params": "results/tuned3d_params.json"}}
        (ROOT / "results/mission_h3.json").write_text(json.dumps(res, indent=1))
        print("H3 3D:", out["preregistered"]["gate"], out["preregistered"]["dead_because"], "| posthoc:", out["posthoc_dominance"]["gate"])


def selfcheck():
    assert len(CONDS) == 25 and len(units("r1")) == 6 * N_CONFIGS
    assert configs("vote")[0] == configs("vote")[0] and len(configs("ring")[0]) == len(CLASSES["ring"].PARAM_SPACE)
    s = scen("ring", configs("ring")[3], 0, "train", "decoy", 1 / 3)
    assert s["brain"]["ring_gain"] == configs("ring")[3]["ring_gain"] and "ring_gain" not in s["brain"]["params"]
    assert s["world"]["landscape"] == "forest" and s["decoys"][0]["swap"]
    # AUC on a synthetic table: all success -> 1; success only without fault -> mean of the first trapezoid
    rows = [{"seed": sd, "fault": f, "severity": v, "success_target": True, "success_home": f == "none", "collisions": 0}
            for sd in (0, 1) for f, v in CONDS]
    a = auc_table(pd.DataFrame(rows))
    assert np.allclose(a["compass_half"], 0.5) and np.allclose(a["gust"], 0.5 / 3)
    # one real mini unit, timed
    t0 = time.time()
    df = run_batch([scen("vote", configs("vote")[0], sd, "train", "none", None) for sd in (0, 1)])
    assert len(df) == 2 and set(df.columns) >= {"success_target", "t_hold_occlusion_s"}
    print(f"run_3d self-check OK ({time.time() - t0:.0f} s for 2 missions)")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck(); sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "reduce", "count"])
    ap.add_argument("stage")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.stage, a.shard, a.nshards, a.procs)
    elif a.cmd == "count":
        u = units(a.stage); print(len(u), "units", sum(len(s) for _, s in u), "missions")
    else:
        reduce(a.stage)
