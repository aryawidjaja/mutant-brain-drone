"""v2 mission experiments (H9 design): equal-budget tuning on the v2 sensory stack, test factorial, reduction.

World and conditions as the 3D study (run_3d.py): forest, obstacle density 0.3, 100 x 100 m, target 25-40 m, max 300 s,
the same 25 conditions (none + 7 faults at the same severities). v2 seeds: train 100-199, test 2000-2299.
Success (every arm, both brain versions) = success_full = target stop & landed on the pad (horizontal error <= 0.5 m at
touchdown, or at the hover end for brains without the landing module) & no collision. Score per seed and fault =
success-vs-severity trapezoid AUC with "none" at x = 0 (KT2 definition, run_3d.auc_table); aggregate = equal-weight mean
over the 7 faults.
Default v2 stack (DEFAULT_MODULES): heading (HeadingRing, results/kt5_tuned_params.json), home, landing, front,
gated_arbiter, and inspect only if results/v2_inspect_decision.json says so (train-seed decision).
Stages (units are independent, written to results/v2/<stage>/<unit>.parquet; shard with --shard/--nshards; run them
in this order, each followed by `reduce <stage>`):
  calib_alt  search altitude 2.5 vs 3.5 m: default stack, 3D-tuned vote, train seeds 100-119 x {none, decoy 0.8}
             -> results/v2_alt_decision.json; then set ALT_SEARCH in mission.py and mission.js (selfcheck asserts it)
  sanity     v2 brain (all default modules, ring, pre-tuning params) on train seeds 100-109, forest, no fault
             -> results/v2_sanity.json
  r1    6 arbiters on the v2 stack (ring = RingGated, rivals visibility-gated, state machine frozen) x 64 random configs
        (numpy default_rng(2026), same uniform matrix as KT2/3D, first n_params columns, arbiters_v2.sample_params)
        x train seeds 100-104 x 25 conditions                         -> results/tune_v2_r1.json (top 4 per arm)
  r2    top 4 per arm x train seeds 100-119 x 25 conditions (Amendment 3 successive halving)
                                                                       -> results/tuned_v2_params.json (frozen)
  test  test seeds 2000-2299 x 25 conditions x arms:
          v2_<arm>            the six arbiters on the v2 stack, frozen v2 params (v2_ring = the v2 brain)
          v2_ring_kf          v2 brain with KalmanHeading (labelled comparison arm)
          v1                  the v1 brain (version 1, ring with results/tuned3d_params.json)
          v2_ring_minus_<m>   ablation: the v2 brain without module m, one per default module (without
                              gated_arbiter the ring is the v1 Ring with the tuned ring params it shares)
          v2_ring_plus_inspect  only when inspect is not in the default set
          v2_ring_lf0         v2 brain with a single target photo (learning flight 0 s; main arms use 10 s)
          v2_ring_no_tip      sensitivity: no operator tip (brain.tip = null; spiral search from home, 12 m pitch)
          v2_ring_poor_tip    sensitivity: poor tip (bearing sd 30 deg, distance sd 0.3)
                                                                       -> results/mission_v2.json
H9 criteria (pre-registered, PLAN_V2; operational definitions fixed here before any test data):
  (i)   no-fault full success of v2_ring >= 0.90 (point estimate over 300 seeds; Wilson 95 % CI reported)
  (ii)  aggregate AUC(v2_ring) - AUC(v1) >= 0.15 (point estimate; paired bootstrap 95 % CI reported); secondary: the v1
        brain scored with its own v1 success definition (home within 2 m)
  (iii) every fault's success at its middle severity >= 0.50, middle = the level closest to severity 0.5, the more
        severe one when two tie (occlusion 7 s, decoy 0.80, eye columns 30 %, false sun 90 deg, speed 0.15, gust
        moderate, compass_half its single level)
  H9 met iff (i) and (ii) and (iii). Also reported: v2 ring vs best other v2 arm (Amendment 4 permutation test + Holm
  pairwise), per-module ablation deltas (paired), learning flight 10 s vs 0 s (paired), H6 landing metrics
  (landed_on_pad rate, landing error, touchdown speed) v2_ring vs v2_ring_minus_home per fault.
Usage:  run_v2.py run <stage> --shard i --nshards n [--procs p]  |  run_v2.py reduce <stage>  |  run_v2.py count <stage>
        run_v2.py --selfcheck  |  run_v2.py --v1check  |  run_v2.py calib  |  run_v2.py decide_inspect  |  run_v2.py pilot
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent / "brain")]
from mission import Mission, run_batch, T_HLF  # noqa: E402
from provenance import stamp  # noqa: E402
from scenario import MODULES, sample_scenario  # noqa: E402
from run_3d import FAULT_SEV, CONDS, CHUNK, scen, auc_table, param_value  # noqa: E402
from kt2_five_arms import N_CONFIGS, SEARCH_SEED  # noqa: E402
from stats import arm_permutation_test, boot_idx, holm, paired_bootstrap, paired_diff  # noqa: E402
import arbiters_v2  # noqa: E402
import eye  # noqa: E402
from mission import V2_ARB  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORLD = {"landscape": "forest", "obstacle_density": 0.3}
OUT = ROOT / "results/v2"
ARMS = ("ring", "ring_static", "vote", "sum", "accumulator", "state_machine")
R1_SEEDS, R2_SEEDS, R2_TOP = range(100, 105), range(100, 120), 4
TEST_SEEDS = range(2000, 2300)
MIDDLE = {"occlusion": 2 / 3, "decoy": 2 / 3, "compass_half": 1.0, "eye_columns": 2 / 3, "false_sun": 2 / 3,
          "speed_scale": 0.5, "gust": 0.5}


def default_modules():
    f = ROOT / "results/v2_inspect_decision.json"
    on = json.loads(f.read_text())["inspect_default_on"] if f.exists() else True
    return [m for m in MODULES if m != "inspect" or on]


def heading_params(est="ring"):
    return json.loads((ROOT / "results/kt5_tuned_params.json").read_text())["estimators"][est]["params"]


def configs(arm):
    """64 random configs of the v2 class for an arm (same uniform matrix as KT2 and the 3D study)."""
    name = V2_ARB[arm]
    names = list(arbiters_v2.CLASSES[name].PARAM_SPACE)
    U = np.random.default_rng(SEARCH_SEED).random((N_CONFIGS, 16))[:, :len(names)]
    return [arbiters_v2.sample_params(name, u) for u in U]


def scen2(arm, params, seed, split, fault, sev, modules=None, est="ring", lf=10.0, version=2, tip="v1"):
    """One v2 scenario (v1 brain when version = 1). ring_gain travels in brain.ring_gain as in the 3D study."""
    if version == 1:
        return scen(arm, params, seed, split, fault, sev)
    p = dict(params)
    g = float(p.pop("ring_gain", 0.0)) if arm == "ring" else 0.0
    mods = default_modules() if modules is None else modules
    if "gated_arbiter" not in mods:          # the v1 class: keep only the parameters it has
        from arbiters import CLASSES as V1
        p = {k: v for k, v in p.items() if k in V1[arm].DEFAULTS}
    mp = {"heading": heading_params(est)} | ({"heading_estimator": est} if est != "ring" else {}) if "heading" in mods else {}
    s = sample_scenario(seed, split=split, fault=fault, severity=sev, arbiter=arm, ring_gain=g, params=p,
                        learning_flight_s=lf, version=2, modules=mods, module_params=mp or None, **WORLD)
    if tip != "v1":                          # operator tip: None = no tip, {bearing_sd_deg, dist_sd} = its quality
        s["brain"]["tip"] = tip
    s["_meta"]["split"] = split
    return s


def test_arms():
    """-> {arm label: kwargs for scen2 except seed/split/fault/sev, plus the arbiter}."""
    tuned = json.loads((ROOT / "results/tuned_v2_params.json").read_text())["arms"]
    t3 = json.loads((ROOT / "results/tuned3d_params.json").read_text())["arms"]
    dm = default_modules()
    ring = tuned["ring"]["params"]
    out = {f"v2_{a}": dict(arm=a, params=tuned[a]["params"]) for a in ARMS}
    out["v2_ring_kf"] = dict(arm="ring", params=ring, est="kf")
    out["v1"] = dict(arm="ring", params=t3["ring"]["params"], version=1)
    for m in dm:
        out[f"v2_ring_minus_{m}"] = dict(arm="ring", params=ring, modules=[x for x in dm if x != m])
    if "inspect" not in dm:
        out["v2_ring_plus_inspect"] = dict(arm="ring", params=ring, modules=dm + ["inspect"])
    out["v2_ring_lf0"] = dict(arm="ring", params=ring, lf=0.0)
    out["v2_ring_no_tip"] = dict(arm="ring", params=ring, tip=None)                       # sensitivity arms (owner request)
    out["v2_ring_poor_tip"] = dict(arm="ring", params=ring, tip={"bearing_sd_deg": 30.0, "dist_sd": 0.3})
    return out


ALTS = (2.5, 3.5)
ALT_SEEDS = range(100, 120)


def units(stage):
    out = []
    if stage == "calib_alt":          # v2 search altitude (protocol fixed before running): default stack, 3D-tuned vote params
        t3 = json.loads((ROOT / "results/tuned3d_params.json").read_text())["arms"]
        for a in ALTS:
            for k in range(0, len(ALT_SEEDS), 10):
                out.append((f"a{a}__s{k:02d}", [scen2("vote", t3["vote"]["params"], s, "train", f, v) for s in list(ALT_SEEDS)[k:k + 10]
                                                for f, v in (("none", None), ("decoy", 2 / 3))]))
    elif stage == "sanity":           # task sanity: v2 brain, all default modules, ring (dev params), no fault, train seeds
        out.append(("ring_dev__s100", [scen2("ring", dev_ring_params(), s, "train", "none", None) for s in range(100, 110)]))
    elif stage == "r1":
        for arm in ARMS:
            for i, p in enumerate(configs(arm)):
                out.append((f"{arm}__c{i:02d}", [scen2(arm, p, s, "train", f, v) for s in R1_SEEDS for f, v in CONDS]))
    elif stage == "r2":
        top = json.loads((ROOT / "results/tune_v2_r1.json").read_text())["top"]
        for arm in ARMS:
            cf = configs(arm)
            for i in top[arm]:
                for k in range(0, len(R2_SEEDS), 10):
                    seeds = list(R2_SEEDS)[k:k + 10]
                    out.append((f"{arm}__c{i:02d}__s{k:02d}", [scen2(arm, cf[i], s, "train", f, v) for s in seeds for f, v in CONDS]))
    elif stage == "test":
        for label, kw in test_arms().items():
            kw = dict(kw)
            arm, params = kw.pop("arm"), kw.pop("params")
            for ci, (f, v) in enumerate(CONDS):
                out.append((f"{label}__k{ci:02d}", [scen2(arm, params, s, "test", f, v, **kw) for s in TEST_SEEDS]))
    else:
        raise SystemExit(f"unknown stage {stage}")
    return out


def run_unit(args):
    stage, uid, scs = args
    path = OUT / stage / f"{uid}.parquet"
    if path.exists():
        return uid, 0.0
    if stage == "calib_alt":                       # this unit's search altitude (module constant, set in the worker)
        import mission
        mission.ALT_SEARCH = float(uid.split("__")[0][1:])
    t0 = time.time()
    df = pd.concat([run_batch(scs[i:i + CHUNK]) for i in range(0, len(scs), CHUNK)], ignore_index=True)
    df["unit"] = uid
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path.with_suffix(".tmp"))
    os.replace(path.with_suffix(".tmp"), path)
    return uid, time.time() - t0


def run(stage, shard, nshards, procs):
    from concurrent.futures import ProcessPoolExecutor
    us = units(stage)[shard::nshards]
    print(f"{stage} shard {shard}/{nshards}: {len(us)} units, {sum(len(s) for _, s in us)} missions", flush=True)
    with ProcessPoolExecutor(max_workers=procs) as ex:
        for uid, dt in ex.map(run_unit, [(stage, u, s) for u, s in us]):
            print(f"  {uid} {dt:.0f}s", flush=True)


def load_stage(stage):
    fs = sorted((OUT / stage).glob("*.parquet"))
    assert fs, f"no results for {stage}"
    want = {u for u, _ in units(stage)} if stage != "sanity" else set()
    missing = want - {f.stem for f in fs}
    assert not missing, f"{stage}: {len(missing)} of {len(want)} units missing (e.g. {sorted(missing)[:3]}); refusing to reduce"
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df.loc[df.fault == "none", "severity"] = np.nan
    df["label"] = df.unit.str.split("__").str[0]
    return df


def full(df):
    """auc_table scores success = success_target & success_home & no collision; v2 uses success_full."""
    return df.assign(success_home=df.success_full.astype(bool))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def reduce(stage):
    df = load_stage(stage)
    st = stamp(seed=sorted(df.seed.unique().tolist())[:3] + ["..."], param_source=f"results/v2/{stage}")
    if stage == "calib_alt":
        df["alt"] = df.unit.str.split("__").str[0].str[1:].astype(float)
        res = {str(a): {f: {"success_full": float(g.success_full.mean()), "collisions": float((g.collisions > 0).mean()),
                            "wrong_object_stop": float(g.wrong_object_stop.mean()), "n": int(len(g))} for f, g in d.groupby("fault")}
               | {"all": float(d.success_full.mean())} for a, d in df.groupby("alt")}
        best = max(ALTS, key=lambda a: (res[str(a)]["all"], a))
        out = {"rule": "higher full success over none + decoy 0.8 (train 100-119); ties -> 3.5 m", "result": res,
               "ALT_SEARCH": best, "provenance": st, "next": "set ALT_SEARCH in scripts/mission.py and app/sim/mission.js"}
        (ROOT / "results/v2_alt_decision.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(res, indent=1), "\nALT_SEARCH ->", best)
        return
    if stage == "sanity":
        cols = ["seed", "success_target", "landed_on_pad", "success_full", "landing_error_m", "touchdown_speed_mps", "time_s", "collisions"]
        out = {"seeds": "train 100-109", "modules": default_modules(), "params": "dev_ring_params()", "world": WORLD, "fault": "none",
               "full_success": float(df.success_full.mean()), "rows": df.sort_values("seed")[cols].to_dict("records"), "provenance": st}
        (ROOT / "results/v2_sanity.json").write_text(json.dumps(out, indent=1, default=float))
        print(df.sort_values("seed")[cols].to_string(), "\nfull success", out["full_success"])
        return
    if stage in ("r1", "r2"):
        df["cfg"] = df.unit.str.split("__").str[1].str[1:].astype(int)
        scores = {arm: {int(c): float(auc_table(full(g)).mean(1).mean()) for c, g in df[df.label == arm].groupby("cfg")} for arm in ARMS}
        rank = lambda sc: sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))  # noqa: E731
        if stage == "r1":
            res = {"stage": "r1", "seeds": "train 100-104", "conds": len(CONDS), "scores": scores,
                   "top": {a: [c for c, _ in rank(sc)[:R2_TOP]] for a, sc in scores.items()}, "provenance": st}
            (ROOT / "results/tune_v2_r1.json").write_text(json.dumps(res, indent=1))
        else:
            arms = {}
            for a, sc in scores.items():
                c = rank(sc)[0][0]
                arms[a] = {"class": V2_ARB[a], "params": configs(a)[c], "config_index": c, "train_auc": sc[c], "rung2": sc,
                           "n_params": len(arbiters_v2.CLASSES[V2_ARB[a]].PARAM_SPACE)}
            res = {"stage": "r2 (frozen before any v2 test seed)", "modules": default_modules(),
                   "budget": f"{N_CONFIGS} configs (default_rng({SEARCH_SEED})); rung 1 train seeds 100-104; rung 2 top {R2_TOP} "
                             f"on train seeds 100-119; {len(CONDS)} conditions; same for every arm",
                   "objective": "mean over 7 faults of mean-over-seeds success_full-vs-severity AUC", "world": WORLD,
                   "arms": arms, "provenance": st}
            (ROOT / "results/tuned_v2_params.json").write_text(json.dumps(res, indent=1))
        for a in ARMS:
            print(a, sorted(scores[a].values())[-3:])
        return
    assert stage == "test"
    labels = sorted(df.label.unique())
    auc = {l: auc_table(full(df[df.label == l])) for l in labels}
    agg = {l: auc[l].mean(1).sort_index().to_numpy() for l in labels}
    ok = df.assign(ok=df.success_full.astype(float))
    sbc = {l: {f"{f}:{param_value(f, v) if f != 'none' else None}": float(g.ok.mean())
               for (f, v), g in ok[ok.label == l].groupby(["fault", ok.severity.fillna(-1.0)])} for l in labels}
    sub = lambda l: ok[ok.label == l]  # noqa: E731
    v2r = sub("v2_ring")
    n0 = v2r[v2r.fault == "none"]
    crit_i = {"success": float(n0.ok.mean()), "wilson95": wilson(n0.ok.sum(), len(n0)), "n": int(len(n0))}
    crit_i["pass"] = crit_i["success"] >= 0.90
    d = paired_diff(agg["v2_ring"], agg["v1"])
    v1own = auc_table(df[df.label == "v1"]).mean(1).sort_index().to_numpy()
    crit_ii = {"auc_v2_ring": float(agg["v2_ring"].mean()), "auc_v1": float(agg["v1"].mean()), "diff": d,
               "pass": d["est"] >= 0.15,
               "secondary_v1_own_definition": {"auc_v1_own": float(v1own.mean()), "diff": paired_diff(agg["v2_ring"], v1own)}}
    mid = {}
    for f, sv in MIDDLE.items():
        g = v2r[(v2r.fault == f) & np.isclose(v2r.severity.fillna(-1.0), sv)]
        mid[f] = {"severity": sv, "level": param_value(f, sv), "success": float(g.ok.mean()), "n": int(len(g))}
    crit_iii = {"per_fault": mid, "pass": all(m["success"] >= 0.5 for m in mid.values())}
    others = [f"v2_{a}" for a in ARMS if a != "ring"]
    perm = arm_permutation_test(agg["v2_ring"], [agg[o] for o in others])
    pair = {o: paired_diff(agg["v2_ring"], agg[o]) for o in others}
    for o, pa in zip(others, holm([pair[o]["p"] for o in others])):
        pair[o]["p_holm"] = pa
    abl = {l: paired_diff(agg["v2_ring"], agg[l]) for l in labels if l.startswith("v2_ring_minus") or l.startswith("v2_ring_plus")}
    lf = paired_diff(agg["v2_ring"], agg["v2_ring_lf0"])
    land = {}
    for l in ("v2_ring", "v2_ring_minus_home", "v2_ring_minus_landing", "v1"):
        if l in labels:
            g = sub(l)
            land[l] = {f: {"landed_on_pad": float(x.landed_on_pad.mean()), "landing_error_median_m": float(x.landing_error_m.median()),
                           "touchdown_speed_median_mps": float(x.touchdown_speed_mps.median()) if x.touchdown_speed_mps.notna().any() else None}
                       for f, x in g.groupby("fault")}
    res = {"test": "H9 v2 mission factorial", "seeds": "test 2000-2299", "n_seeds": len(agg["v2_ring"]), "conds": len(CONDS),
           "world": WORLD, "modules_default": default_modules(), "arms": labels,
           "aggregate_auc": {l: dict(zip(("mean", "lo", "hi"), [float(agg[l].mean())] + [float(x) for x in np.quantile(
               [agg[l][i].mean() for i in boot_idx(len(agg[l]))], [0.025, 0.975])])) for l in labels},
           "per_fault_auc": {l: {f: float(auc[l][f].mean()) for f in FAULT_SEV} for l in labels},
           "success_by_condition": sbc, "H9_i": crit_i, "H9_ii": crit_ii, "H9_iii": crit_iii,
           "H9": bool(crit_i["pass"] and crit_ii["pass"] and crit_iii["pass"]),
           "v2_ring_vs_rivals": {"permutation_test": perm, "pairwise": pair}, "ablations_v2_ring_minus_arm": abl,
           "learning_flight_10_vs_0": lf, "landing": land,
           "wrong_object_stop_rate": {l: float(sub(l).wrong_object_stop.mean()) for l in labels},
           "collision_rate": {l: float((sub(l).collisions > 0).mean()) for l in labels},
           "provenance": st | {"tuned_params": "results/tuned_v2_params.json"}}
    (ROOT / "results/mission_v2.json").write_text(json.dumps(res, indent=1, default=float))
    df.drop(columns=["unit"]).to_parquet(ROOT / "results/runs/mission_v2_test.parquet")
    print("H9:", res["H9"], "| (i)", crit_i, "| (ii)", crit_ii["diff"], "| (iii)", {f: m["success"] for f, m in mid.items()})


def v1check(n_print=True):
    """v1 bit-identity guard: 24 fixed rows of results/runs/mission_test.parquet (6 arms x 4 conditions) re-run with the
    current code. Discrete outcomes and time_s must be identical; continuous sums within 1e-9 (the parquet rows ran in
    other batch compositions, which moves the last bits of BLAS-ordered sums by <= ~1e-12)."""
    tuned = json.loads((ROOT / "results/tuned3d_params.json").read_text())["arms"]
    df = pd.read_parquet(ROOT / "results/runs/mission_test.parquet")
    rs = np.random.default_rng(7)
    conds = ["none", "decoy", "occlusion", "false_sun", "speed_scale", "gust", "compass_half", "eye_columns"]
    pick = []
    for i, arm in enumerate(tuned):
        d = df[(df.arbiter == arm) & (df.time_s < 150)]
        for f in (conds * 2)[i % 8:i % 8 + 4]:
            g = d[d.fault == f]
            pick.append(g.iloc[rs.integers(len(g))])
    P = pd.DataFrame(pick).reset_index(drop=True)
    scs = [scen(r.arbiter, tuned[r.arbiter]["params"], int(r.seed), "test", r.fault,
                None if r.fault == "none" else FAULT_SEV[r.fault][int(r.level) - 1]) for r in P.itertuples()]
    out = run_batch(scs)
    exact = ["scenario_hash", "seed", "arbiter", "success_target", "success_home", "collisions", "wrong_object_stop", "time_s",
             "t_hold_occlusion_s", "t_release_decoy_s"]
    close = ["ring_gain", "path_len_m", "home_error_m", "max_bank_deg", "energy_proxy"]
    for c in exact:
        a, b = out[c].to_numpy(), P[c].to_numpy()
        ok = [(x == y) or (isinstance(x, float) and math.isnan(x) and math.isnan(y)) for x, y in zip(a, b)]
        assert all(ok), (c, a[~np.array(ok)], b[~np.array(ok)])
    dmax = max(float(np.nanmax(np.abs(out[c].to_numpy(float) - P[c].to_numpy(float)))) for c in close)
    assert dmax <= 1e-9, dmax
    if n_print:
        print(f"v1check OK: {len(P)} rows of mission_test.parquet ({P.arbiter.nunique()} arms, {P.fault.nunique()} faults): "
              f"discrete + time_s identical, continuous max |diff| {dmax:.1e}")
    return P, out


def _looks_at(m, scene, rows, k, P, rs):
    g, _ = eye.looks(eye.sub_scene(scene, rows, np.full(len(rows), k)), P, 100.0, **m.look_kw)
    return g[:, 0] + eye.NOISE * rs.standard_normal(g[:, 0].shape)


def calib(front, seeds=range(100, 140), n_pose=20):
    """Calibration on train seeds (procedure fixed in the protocol before running). Home memory: fly the take-off
    learning flight, then score pad looks from the final-approach envelope (1-8 m, 2.5 m up) and target/decoy looks
    from 1.5-5 m; F_HOME = smallest 0.01 step with <= 2 % of object looks above it. With the front end also the
    familiarity -> evidence map (v1 rule): F0 = median target+decoy familiarity at 15-20 m, F1 = median target
    familiarity at 1.5-4 m (target memory = 10 s learning flight)."""
    tuned = json.loads((ROOT / "results/tuned3d_params.json").read_text())["arms"]
    mods = ["home", "landing"] + (["front"] if front else [])
    scs = [sample_scenario(s, fault="decoy", severity=2 / 3, arbiter="vote", params=tuned["vote"]["params"], version=2,
                           modules=mods, **WORLD) for s in seeds]
    m = Mission(scs)
    while m.t < T_HLF - 1e-9:
        m.step()
    B, rows = m.B, np.arange(m.B)
    rs = np.random.default_rng(0)
    ring = lambda c, lo, hi: np.stack([c[:, 0] + (r := rs.uniform(lo, hi, B)) * np.cos(a := rs.uniform(0, 2 * np.pi, B)),  # noqa: E731
                                       c[:, 1] + r * np.sin(a), np.full(B, 2.5)], 1)
    tfam = lambda x: 1.0 - (m.code(x) & m.mbw).sum(-1) / m.code(x).sum(-1)  # noqa: E731
    fam = {"pad": [], "object": [], "t_near": [], "obj_far": []}
    for j in range(n_pose):
        fam["pad"].append(m.hm.familiarity(rows, _looks_at(m, m.PS, rows, 0, ring(m.home, 1, 8), rs)))
        for k in range(m.K):
            c = m.S.obj[:, k]
            fam["object"].append(m.hm.familiarity(rows, _looks_at(m, m.S, rows, k, ring(c, 1.5, 5), rs)))
            if front:
                near = _looks_at(m, m.S, rows, k, ring(c, 1.5, 4), rs)
                fam["t_near"].append(tfam(near)[m.S.role[:, k] == 1])
                fam["obj_far"].append(tfam(_looks_at(m, m.S, rows, k, ring(c, 15, 20), rs)))
    pad, obj = np.concatenate(fam["pad"]), np.concatenate(fam["object"])
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    f_home = float(next(g for g in grid if (obj >= g).mean() <= 0.02))
    q = lambda x: [float(v) for v in np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])]  # noqa: E731
    res = {"features": "front" if front else "grey", "seeds": f"train {seeds.start}-{seeds.stop - 1}",
           "views_per_seed": float(m.hm.n_views.mean()), "pad_quantiles_5_25_50_75_95": q(pad),
           "object_quantiles_5_25_50_75_95": q(obj), "F_HOME": f_home, "F_TRACK": f_home / 2,
           "pad_recognised_frac": float((pad >= f_home).mean()), "object_false_lock_frac": float((obj >= f_home).mean())}
    if front:
        tn, of = np.concatenate(fam["t_near"]), np.concatenate(fam["obj_far"])
        res |= {"F0": float(np.median(of)), "F1": float(np.median(tn)), "target_near_quantiles": q(tn), "objects_far_quantiles": q(of)}
    path = ROOT / "results/v2_calib.json"
    allr = json.loads(path.read_text()) if path.exists() else {}
    allr[res["features"]] = res | stamp(seed=res["seeds"], param_source="scripts/run_v2.py calib")
    path.write_text(json.dumps(allr, indent=1))
    print(json.dumps(res, indent=1))
    return res


def _batch(args):
    scs, = args
    return run_batch(scs)


def decide_inspect(seeds=range(100, 140)):
    """Default-module decision for the inspection rule (protocol fixed before running): v2 stack with and without
    'inspect', train seeds 100-139 x {no fault, swapped decoy 0.8}, 3D-tuned vote params."""
    from concurrent.futures import ProcessPoolExecutor
    tuned = json.loads((ROOT / "results/tuned3d_params.json").read_text())["arms"]
    sets = {"with": MODULES, "without": [m for m in MODULES if m != "inspect"]}
    jobs = [[sample_scenario(s, fault=f, severity=v, arbiter="vote", params=tuned["vote"]["params"], version=2, modules=mods, **WORLD)
             for s in seeds for f, v in (("none", None), ("decoy", 2 / 3))] for mods in sets.values()]
    with ProcessPoolExecutor(2) as ex:
        dfs = list(ex.map(_batch, [(j,) for j in jobs]))
    res = {}
    for (k, _), df in zip(sets.items(), dfs):
        res[k] = {f: {"success_full": float(g.success_full.mean()), "wrong_object_stop": float(g.wrong_object_stop.mean()),
                      "success_target": float(g.success_target.mean())} for f, g in df.groupby("fault")}
    keep = (res["with"]["decoy"]["success_full"] >= res["without"]["decoy"]["success_full"]
            and res["with"]["decoy"]["wrong_object_stop"] < res["without"]["decoy"]["wrong_object_stop"])
    out = {"rule": "keep inspect in the default set iff decoy-0.8 full success does not drop and wrong stops fall",
           "result": res, "inspect_default_on": bool(keep), **stamp(seed="train 100-139", param_source="scripts/run_v2.py decide_inspect")}
    (ROOT / "results/v2_inspect_decision.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(res, indent=1), "\ninspect default on:", keep)
    return out


def dev_ring_params():
    """Pre-tuning v2 ring params for the sanity run and the demo replay: RingGated defaults with the 3D-tuned values of
    the parameters it shares with the v1 ring (no v2 tuning result is used or needed)."""
    t3 = json.loads((ROOT / "results/tuned3d_params.json").read_text())["arms"]["ring"]["params"]
    return {**arbiters_v2.RingGated.DEFAULTS, **t3}


def demo(seed):
    """Short replay for the app (training-seed illustration, export_runs.py format): app/runs/v2_demo_<seed>.json."""
    import base64
    from export_runs import r4, FRAME_EVERY, EYE_EVERY
    sc = scen2("ring", dev_ring_params(), seed, "train", "none", None)
    m = Mission([sc], log_rows=[0])
    n = 0
    while m.t < sc["mission"]["max_time_s"] - 1e-9 and m.phase[0] < 4:
        m.step()
        if len(m.frames[0]) > n:
            m.frames[0][-1]["pi"]["heading_est_deg"] = math.degrees(float(m.heading_est[0]))
            n = len(m.frames[0])
    out = m.outcome_row(0)
    fr = m.frames[0]
    keep = [f for i, f in enumerate(fr) if i % FRAME_EVERY == 0 or i == len(fr) - 1]
    for i, f in enumerate(keep):
        e = f.pop("eye")
        if i % (EYE_EVERY // FRAME_EVERY) == 0:
            f["eye"] = {"w": e["w"], "h": e["h"], "b64": base64.b64encode(bytes(e["rgb"])).decode()}
    keep[-1]["outcome"] = out
    st = stamp(param_source="scripts/run_v2.py demo (dev ring params, training seed)")
    run = {"meta": {"scenario": {k: v for k, v in sc.items() if not k.startswith("_")}, "arbiter": "ring", "seed": seed,
                    "git": st["git"], "sim_version": st["sim_version"], "created": st["created"], "outcome": out,
                    "label": f"v2 brain (all modules), no fault, forest: training-seed illustration (seed {seed}), not a result",
                    "params_source": "pre-tuning v2 ring params (RingGated defaults + 3D-tuned shared values)",
                    "note": "Illustrative replay recorded by scripts/run_v2.py demo with the Python reference on a TRAINING seed; not a result."},
           "world": m.worlds[0], "frames": [r4(f) for f in keep]}
    path = ROOT / f"app/runs/v2_demo_{seed}.json"
    path.write_text(json.dumps(run, separators=(",", ":")))
    assert run["frames"][-1]["outcome"] == out and all(b["t"] > a["t"] for a, b in zip(run["frames"], run["frames"][1:]))
    assert "home_pad" in run["world"] and all("hud" in f and "home" in f for f in run["frames"])
    print(f"{path.name}: {len(keep)} frames, {path.stat().st_size / 1e6:.2f} MB, outcome {out}")
    return run


def pilot():
    """<= 2-min timing smoke on TRAIN seeds: v2 ring (config 0), 2 seeds x 3 conditions, 40 s of mission each. Reports
    CPU-s per simulated mission-second per row and extrapolates every stage with the mean v1 3D mission duration
    (results/runs/mission_test.parquet, ring arm) + the 6 s longer take-off; small batches overestimate the per-row cost."""
    scs = [scen2("ring", configs("ring")[0], s, "train", f, v) for s in (100, 101) for f, v in (("none", None), ("decoy", 2 / 3), ("false_sun", 2 / 3))]
    m = Mission(scs)
    c0, t0 = time.process_time(), time.time()
    while m.t < 40.0 - 1e-9:
        m.step()
    cpu, wall = time.process_time() - c0, time.time() - t0
    per_s = cpu / (len(scs) * 40.0)
    dur = float(pd.read_parquet(ROOT / "results/runs/mission_test.parquet", columns=["arbiter", "time_s"]).query("arbiter == 'ring'").time_s.mean()) + 6.0
    ext = {}
    for st in ("calib_alt", "sanity", "r1", "r2", "test"):
        try:
            us = units(st)
        except FileNotFoundError:
            n_units = {"r2": 6 * R2_TOP * 2, "test": None}[st]
            n_m = {"r2": 6 * R2_TOP * len(R2_SEEDS) * len(CONDS), "test": None}[st]
            if st == "test":
                n_arms = 6 + 1 + 1 + len(default_modules()) + (0 if "inspect" in default_modules() else 1) + 1 + 2
                n_units, n_m = n_arms * len(CONDS), n_arms * len(CONDS) * len(TEST_SEEDS)
        else:
            n_units, n_m = len(us), sum(len(x) for _, x in us)
        ext[st] = {"units": n_units, "missions": n_m, "cpu_hours": n_m * dur * per_s / 3600, "cpu_min_per_unit": n_m / n_units * dur * per_s / 60}
    res = {"rows": len(scs), "sim_s": 40.0, "cpu_s": cpu, "wall_s": wall, "cpu_s_per_mission_second": per_s,
           "assumed_mean_mission_s": dur, "extrapolation": ext}
    print(json.dumps(res, indent=1))
    return res


def selfcheck():
    assert len(CONDS) == 25 and len(units("r1")) == 6 * N_CONFIGS and len(configs("ring")[0]) == 10
    s = scen2("ring", configs("ring")[3], 100, "train", "decoy", 1 / 3)
    assert s["brain"]["version"] == 2 and s["brain"]["ring_gain"] == configs("ring")[3]["ring_gain"]
    assert "ring_gain" not in s["brain"]["params"] and s["brain"]["module_params"]["heading"] == heading_params()
    s = scen2("ring", configs("ring")[3], 100, "train", "none", None, modules=["home", "landing"])
    assert "module_params" not in s["brain"] and set(s["brain"]["params"]) <= {"kappa", "g_obj", "g_home", "g_search", "tau_s"}
    assert scen2("ring", {"g_home": 3.0}, 2000, "test", "none", None, version=1)["brain"].get("version", 1) == 1
    try:
        scen2("ring", {}, 1500, "test", "none", None); raise AssertionError("seed outside both test ranges accepted")
    except AssertionError as e:
        assert "not in test" in str(e)
    assert all(np.isclose(MIDDLE[f], min(FAULT_SEV[f], key=lambda v: (abs(v - 0.5), -v))) for f in FAULT_SEV)
    # success_full drives the AUC: a synthetic table where only the no-fault rows landed on the pad
    rows = [{"seed": sd, "fault": f, "severity": v, "success_target": True, "success_home": True, "success_full": f == "none",
             "collisions": 0} for sd in (0, 1) for f, v in CONDS]
    a = auc_table(full(pd.DataFrame(rows)))
    assert np.allclose(a["compass_half"], 0.5) and np.allclose(a["gust"], 0.5 / 3)
    assert abs(wilson(270, 300)[0] - 0.861) < 0.002
    import mission, re
    f = ROOT / "results/v2_alt_decision.json"
    js_alt = float(re.search(r"ALT_SEARCH = ([0-9.]+)", (ROOT / "app/sim/mission.js").read_text()).group(1))
    assert js_alt == mission.ALT_SEARCH, (js_alt, mission.ALT_SEARCH)
    if f.exists():
        assert json.loads(f.read_text())["ALT_SEARCH"] == mission.ALT_SEARCH, "set ALT_SEARCH from results/v2_alt_decision.json"
    t0 = time.time()
    scs = [scen2("vote", configs("vote")[0], sd, "train", "none", None) for sd in (100, 101)]
    scs[1]["brain"]["tip"] = None
    for x in scs:
        x["mission"]["max_time_s"] = 20.0
    df = run_batch(scs)
    assert len(df) == 2 and {"success_full", "landed_on_pad", "landing_error_m", "touchdown_speed_mps"} <= set(df.columns)
    assert (df.version == 2).all() and (df.time_s <= 20.0 + 1e-9).all()
    print(f"run_v2 self-check OK ({time.time() - t0:.0f} s for 2 x 20 s v2 missions); stage sizes: "
          + ", ".join(f"{st} {len(units(st))} units" for st in ("calib_alt", "sanity", "r1")))


if __name__ == "__main__":
    if "--v1check" in sys.argv:
        v1check(); sys.exit(0)
    if "--selfcheck" in sys.argv:
        selfcheck(); sys.exit(0)
    if "demo" in sys.argv:
        demo(int(sys.argv[sys.argv.index("demo") + 1])); sys.exit(0)
    if "pilot" in sys.argv:
        pilot(); sys.exit(0)
    if "decide_inspect" in sys.argv:
        decide_inspect(); sys.exit(0)
    if "calib" in sys.argv:
        calib(front=False); calib(front=True); sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "reduce", "count"])
    ap.add_argument("stage")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    if a.stage == "test" and a.cmd == "run":
        assert (ROOT / "results/tuned_v2_params.json").exists(), "freeze the v2 params (r2) first"
    if a.cmd == "run":
        run(a.stage, a.shard, a.nshards, a.procs)
    elif a.cmd == "count":
        u = units(a.stage); print(len(u), "units", sum(len(x) for _, x in u), "missions")
    else:
        reduce(a.stage)
