"""KT7 (H7): an un-stubborn ring. Evidence-gated ring vs visibility-gated rivals on a three-trial battery + the KT2 world.

Pre-registered (H7 protocol, before any KT7 data). Trials: KT3 hold (occ) and release (dec), imported unchanged from
kt3_stubbornness, plus a premature-switch trial (sw, below). Equal tuning: 64 random configs per arm on 2D training
seeds 0-99, two rungs (as KT2 amendment A1), objective J = 0.5 AUC_KT2 + 0.5 mean(hold/10, 1 - release/20,
1 - time_off/21). Test on 2D test seeds 1300-1599: the KT3 frontier rule (pareto_interp primary, dominance frontier as
sensitivity), unclamped release, premature-switch metrics, and the KT2-world six-arm success with the within-seed
arm-permutation test.

sw trial: target (0.8) and a look-alike (0.3), both 80 m away and visible throughout, search drive on the other side.
Five transient events with onsets jittered by U[0, 1] s: target dips to 0.25 (0.6 s at 7 s, 1.0 s at 13 s), look-alike
spikes to 0.95 (0.8 s at 10 s, 1.2 s at 17 s), both at once (1.0 s at 21 s). Over [5, 26) s: time off target
(|cmd - target| > 20 deg) and wrongful switch (any tick with |cmd - look-alike| < 20 deg).

Usage (KT_PROCS caps worker processes, default 2):
  uv run python scripts/kt7_stubbornness_v2.py --pilot       # timings only
  uv run python scripts/kt7_stubbornness_v2.py --tune        # -> results/kt7_tuned_params.json, kt7_envelope.json, fig_kt7_envelope.png
  uv run python scripts/kt7_stubbornness_v2.py --test        # -> results/kt7.json, fig_kt7.png
  uv run python scripts/kt7_stubbornness_v2.py --smoke       # whole --test code path on 6 training seeds, temp dir
  uv run python scripts/kt7_stubbornness_v2.py --selfcheck
  (append --procs N to any mode to set the worker count)
"""
import hashlib, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import kt2_five_arms as kt2
import kt3_stubbornness as kt3
from brain import arbiters_v2 as v2
from rng import Rng
from provenance import stamp
from stats import arm_permutation_test, boot_idx, holm, paired_diff

kt2.make = kt3.make = v2.make            # the unchanged v1 trial code builds v2 arms by name
ROOT = Path(__file__).resolve().parents[1]
PROCS = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else int(os.environ.get("KT_PROCS", 2))
DT, TOL = kt2.DT, kt3.TOL
TRAIN_SEEDS, TEST_SEEDS = np.arange(0, 100), np.arange(1300, 1600)
RUNG1_KT2_SEEDS, RUNG2_TOP, N_CONFIGS, SEARCH_SEED = np.arange(0, 20), 8, 64, 2026
ARMS = v2.ARBITERS_V2                    # ring_gated, ring_vg, ring_static_vg, vote_vg, sum_vg, accumulator_vg, state_machine
SIX = ("ring_gated", "ring_static_vg", "vote_vg", "sum_vg", "accumulator_vg", "state_machine")
RIVALS = ("state_machine", "accumulator_vg")
SW_T, SW_W0 = 26.0, 5.0
SW_EVENTS = ((7.0, 0.6, "t"), (10.0, 0.8, "l"), (13.0, 1.0, "t"), (17.0, 1.2, "l"), (21.0, 1.0, "tl"))
OFF_MAX = SW_T - SW_W0
NONINF_MARGIN_S = 0.5
GAINS = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
SWEEPS = {"ring_gated": ("ring_gain", GAINS), "ring_vg": ("ring_gain", GAINS),
          "state_machine": ("timeout_s", list(np.geomspace(0.2, 10, 15))),
          "accumulator_vg": ("tau_leak_s", list(np.geomspace(0.05, 20, 15))),
          "ring_static_vg": (None, [None]), "vote_vg": (None, [None]), "sum_vg": (None, [None])}


def configs(arm):
    names = list(v2.CLASSES[arm].PARAM_SPACE)
    U = np.random.default_rng(SEARCH_SEED).random((N_CONFIGS, 16))[:, :len(names)]
    return [v2.sample_params(arm, u) for u in U]


def trial_sw(arm, params, seeds):
    """Premature-switch trial -> time_off [B] (s), wrongful_switch [B] bool, on target at 5 s [B] bool."""
    B = len(seeds)
    r, rj = Rng(seeds, 26), Rng(seeds, 25)
    rc, rv, rg = Rng(seeds, 22), Rng(seeds, 23), Rng(seeds, 24)
    phi = 2 * np.pi * r.uniform()
    sgn = np.sign(r.uniform() - 0.5)
    d1, d2 = np.deg2rad(60 + 90 * r.uniform()), np.deg2rad(60 + 90 * r.uniform())
    h = phi + np.deg2rad(60) * (r.uniform() - 0.5)
    on = [b + rj.uniform() for b, _, _ in SW_EVENTS]
    ang = np.stack([phi, phi + sgn * d1], 1)                     # slot 0 target, slot 1 look-alike
    obj = 80.0 * np.stack([np.cos(ang), np.sin(ang)], -1)
    search_b = wrap(phi - sgn * d2)
    arb = v2.make(arm, params, B)
    pos = np.zeros((B, 2))
    off, wrong, on5 = np.zeros(B), np.zeros(B, bool), np.zeros(B, bool)
    for k in range(int(round(SW_T / DT))):
        t = k * DT
        hhat = h + kt2.SIG_COMPASS * rc.normal()
        rel = obj - pos[:, None]
        bear = wrap(np.arctan2(rel[..., 1], rel[..., 0]) - h[:, None] + hhat[:, None])
        ev_t, ev_l = np.full(B, 0.8), np.full(B, 0.3)
        for o, (_, dur, who) in zip(on, SW_EVENTS):
            act = (t >= o) & (t < o + dur)
            if "t" in who:
                ev_t = np.where(act, 0.25, ev_t)
            if "l" in who:
                ev_l = np.where(act, 0.95, ev_l)
        noise = np.stack([rv.normal(), rv.normal()], 1)
        strength = np.clip(np.stack([ev_t, ev_l], 1) + kt2.SIG_VIS * noise, 0, 1)
        obs = {"objects": {"bearing": bear, "strength": strength, "visible": np.ones((B, 2), bool)},
               "home": {"bearing": np.zeros(B), "strength": np.zeros(B)},
               "search": {"bearing": search_b, "strength": np.ones(B)}, "heading": wrap(hhat)}
        cmd = arb.step(obs, DT)["heading_cmd"]
        e_t, e_l = np.abs(wrap(cmd - bear[:, 0])), np.abs(wrap(cmd - bear[:, 1]))
        if abs(t - SW_W0) < DT / 2:
            on5 = e_t <= TOL
        if t >= SW_W0 - DT / 2:
            off += DT * (e_t > TOL)
            wrong |= e_l < TOL
        turn = np.clip(wrap(cmd - hhat), -kt2.OMEGA * DT, kt2.OMEGA * DT)
        h = h + turn + np.deg2rad(kt2.GUST_DEG["none"]) * np.sqrt(DT) * rg.normal()
        pos = pos + kt2.V * DT * np.stack([np.cos(h), np.sin(h)], 1)
    return off, wrong, on5


wrap = v2.wrap


def battery(arm, params, seeds):
    """The three trials on `seeds` -> dict of per-seed arrays."""
    hold, on_t = kt3.trial(arm, params, seeds, "occ")
    rel, on_d, det = kt3.trial(arm, params, seeds, "dec", detail=True)
    off, wrong, on5 = trial_sw(arm, params, seeds)
    return {"hold": hold, "rel": rel, "rel_signed": det["release_signed"], "ratio": det["ratio_at_switch"], "on_t4": on_t,
            "on_d5": on_d, "off": off, "wrong": wrong, "on5": on5}


def battery_score(b):
    return float(np.mean([b["hold"].mean() / kt3.HOLD_CAP, 1 - b["rel"].mean() / kt3.REL_CAP, 1 - b["off"].mean() / OFF_MAX]))


def kt2_auc(arm, params, seeds):
    """KT2-world per-seed equal-weight aggregate AUC [n_seeds] (unchanged world, 28 conditions)."""
    a = kt2.auc_per_seed(kt2.simulate(arm, params, seeds), len(seeds))
    return np.mean([v for v in a.values()], 0)


def _job(a):
    kind, arm, i, params, seeds = a
    t0 = time.time()
    out = battery(arm, params, seeds) if kind == "battery" else kt2_auc(arm, params, seeds)
    return kind, arm, i, out, time.time() - t0


def _pool(jobs):
    with ProcessPoolExecutor(max_workers=PROCS) as ex:
        return list(ex.map(_job, jobs, chunksize=1))


# ------------------------------------------------------------------------------------------------ pilot / tuning
def pilot():
    t = {}
    for arm in ARMS:
        p = configs(arm)[0]
        t0 = time.time(); battery(arm, p, TRAIN_SEEDS); tb = time.time() - t0
        t0 = time.time(); kt2_auc(arm, p, RUNG1_KT2_SEEDS); tk = time.time() - t0
        t[arm] = (tb, tk)
        print(f"{arm:15s} battery(100 seeds) {tb:5.1f}s   kt2(20 seeds) {tk:5.1f}s", flush=True)
    r1 = sum(64 * (b + k) for b, k in t.values())
    r2 = sum(RUNG2_TOP * 5 * k for _, k in t.values())
    print(f"extrapolated tuning CPU: rung1 {r1 / 60:.1f} min + rung2 {r2 / 60:.1f} min = {(r1 + r2) / 60:.1f} min; "
          f"wall on {PROCS} procs ~{(r1 + r2) / 60 / PROCS:.0f} min")


def tune():
    t0 = time.time()
    cf = {a: configs(a) for a in ARMS}
    jobs = [(k, a, i, p, s) for a in ARMS for i, p in enumerate(cf[a])
            for k, s in (("battery", TRAIN_SEEDS), ("kt2", RUNG1_KT2_SEEDS))]
    jobs.sort(key=lambda j: j[1] in ("ring_gated", "ring_vg"), reverse=True)   # slowest first
    r1 = _pool(jobs)
    bat = {(a, i): o for k, a, i, o, _ in r1 if k == "battery"}
    auc1 = {(a, i): float(o.mean()) for k, a, i, o, _ in r1 if k == "kt2"}
    J1 = {key: 0.5 * auc1[key] + 0.5 * battery_score(bat[key]) for key in bat}
    top = {a: sorted(range(N_CONFIGS), key=lambda i: -J1[(a, i)])[:RUNG2_TOP] for a in ARMS}
    r2 = _pool([("kt2", a, i, cf[a][i], TRAIN_SEEDS) for a in ARMS for i in top[a]])
    auc2 = {(a, i): float(o.mean()) for _, a, i, o, _ in r2}
    J2 = {key: 0.5 * auc2[key] + 0.5 * battery_score(bat[key]) for key in auc2}
    arms = {}
    for a in ARMS:
        best = max(top[a], key=lambda i: J2[(a, i)])
        b = bat[(a, best)]
        arms[a] = {"params": cf[a][best], "config_index": best, "n_params": len(v2.CLASSES[a].PARAM_SPACE),
                   "train_J": J2[(a, best)], "train_auc_kt2": auc2[(a, best)], "train_battery_score": battery_score(b),
                   "train_hold_s": float(b["hold"].mean()), "train_release_s": float(b["rel"].mean()),
                   "train_time_off_s": float(b["off"].mean()),
                   "rung1_J": [J1[(a, i)] for i in range(N_CONFIGS)], "rung1_auc_kt2_seeds0_19": [auc1[(a, i)] for i in range(N_CONFIGS)],
                   "rung2_J": {int(i): J2[(a, i)] for i in top[a]}}
    cpu = sum(x[4] for x in r1 + r2)
    out = {"test": "KT7 tuning (H7)", "seeds": "train 0-99",
           "budget": f"{N_CONFIGS} random configs per arm (numpy default_rng({SEARCH_SEED}).random((64,16)), first n_params columns); "
                     f"rung 1: battery on train 0-99 + KT2 world on train 0-19; rung 2: top {RUNG2_TOP} on KT2 world train 0-99",
           "objective": "J = 0.5 AUC_KT2 + 0.5 mean(hold/10, 1 - release/20, 1 - time_off/21)",
           "arms": arms, "wall_s": time.time() - t0, "cpu_s": cpu,
           "provenance": stamp(seed="train 0-99", param_source="random search (H7 protocol)")}
    path = ROOT / "results/kt7_tuned_params.json"
    path.write_text(json.dumps(out, indent=1))
    print("kt7_tuned_params.json sha256", hashlib.sha256(path.read_bytes()).hexdigest())
    for a, v in arms.items():
        print(f"{a:15s} J {v['train_J']:.3f} auc {v['train_auc_kt2']:.3f} hold {v['train_hold_s']:.2f} rel {v['train_release_s']:.2f} "
              f"off {v['train_time_off_s']:.2f}  {v['params']}")
    print(f"tuning wall {out['wall_s']:.0f}s cpu {cpu / 60:.1f} min")
    envelope(cf, bat)


def efficient(h, r):
    return np.array([not np.any((h >= h[i]) & (r <= r[i]) & ((h > h[i]) | (r < r[i]))) for i in range(len(h))])


FRONTIERS = {"pareto_interp": kt3.pareto_interp, "dominance_frontier": kt3.dominance_frontier}


def envelope(cf, bat, n_boot=2000):
    """POST HOC-style sensitivity on training seeds: all 64 configs per arm; best ring_gated (and ring_vg) config vs each
    gate rival's 64-config envelope. Selection on the same seeds (optimistic for the ring), CI with the selection fixed."""
    H = {a: np.array([bat[(a, i)]["hold"] for i in range(N_CONFIGS)]) for a in ARMS}
    R = {a: np.array([bat[(a, i)]["rel"] for i in range(N_CONFIGS)]) for a in ARMS}
    OFF = {a: np.array([bat[(a, i)]["off"] for i in range(N_CONFIGS)]) for a in ARMS}
    mh, mr, mo = ({a: X[a].mean(1) for a in ARMS} for X in (H, R, OFF))
    idx = boot_idx(H["ring_gated"].shape[1], n_boot=n_boot, seed=0)
    out = {"test": "KT7 64-config envelope (sensitivity)", "seeds": "train 0-99",
           "label": "Sensitivity analysis on TRAINING seeds 0-99 (rung-1 battery data); best ring config selected on the same seeds",
           "configs": {a: [{"index": i, "params": cf[a][i], "hold_mean_s": float(mh[a][i]), "release_mean_s": float(mr[a][i]),
                            "time_off_mean_s": float(mo[a][i]), "wrongful_switch_rate": float(bat[(a, i)]["wrong"].mean())}
                           for i in range(N_CONFIGS)] for a in ARMS},
           "envelope": {a: [int(i) for i in np.flatnonzero(efficient(mh[a], mr[a]))] for a in ARMS}, "best_ring_vs_rival": {}}
    for ring in ("ring_gated", "ring_vg"):
        for rv in RIVALS:
            for fn, f in FRONTIERS.items():
                d = f(mh[rv], mr[rv])(mh[ring]) - mr[ring]
                key = f"{ring}|{rv}|{fn}"
                if np.isnan(d).all():
                    out["best_ring_vs_rival"][key] = {"n_ring_configs_defined": 0}
                    continue
                b = int(np.nanargmax(d))
                bs = np.array([f(H[rv][:, i].mean(1), R[rv][:, i].mean(1))(H[ring][b, i].mean()) - R[ring][b, i].mean() for i in idx]).ravel()
                do = OFF[ring][b] - OFF[rv]                                     # [64 rival configs, seeds]
                out["best_ring_vs_rival"][key] = {
                    "n_ring_configs_defined": int((~np.isnan(d)).sum()), "n_ring_configs_delta_gt_0": int((d > 0).sum()),
                    "best_ring_config_index": b, "best_ring_params": cf[ring][b], "best_ring_hold_s": float(mh[ring][b]),
                    "best_ring_release_s": float(mr[ring][b]), "best_ring_time_off_s": float(mo[ring][b]),
                    "delta_s": float(d[b]), "lo": float(np.nanquantile(bs, 0.025)), "hi": float(np.nanquantile(bs, 0.975)),
                    "frac_resamples_defined": float((~np.isnan(bs)).mean()),
                    "frac_rival_configs_with_less_time_off": float((do.mean(1) > 0).mean())}
    out["provenance"] = stamp(seed="train 0-99", param_source="kt7 configs (64 per arm)")
    (ROOT / "results/kt7_envelope.json").write_text(json.dumps(out, indent=1))
    figure_envelope(out)
    for k, v in out["best_ring_vs_rival"].items():
        print(k, {x: v[x] for x in ("delta_s", "lo", "hi", "best_ring_hold_s", "best_ring_release_s", "best_ring_time_off_s",
                                    "n_ring_configs_delta_gt_0") if x in v})


# ------------------------------------------------------------------------------------------------ test
def dominated3(x, P):
    """x [3] = (hold, release, time_off); P [n, 3] rival points. Higher hold, lower release and time_off are better."""
    better = (P[:, 0] >= x[0]) & (P[:, 1] <= x[1]) & (P[:, 2] <= x[2])
    strict = (P[:, 0] > x[0]) | (P[:, 1] < x[1]) | (P[:, 2] < x[2])
    return bool(np.any(better & strict))


def frontier_deltas(mh, mr, f):
    return {rv: f(mh[rv], mr[rv])(mh["ring_gated"]) - mr["ring_gated"] for rv in RIVALS}


def gate(est, lo, off_ok, est_dom):
    ok = (lo["state_machine"] > 0) & (lo["accumulator_vg"] > 0) & off_ok
    alive = bool(np.any(ok[1:] & ok[:-1]))
    dead = []
    for rv in RIVALS:
        e = est[rv] if not np.isnan(est[rv]).all() else est_dom[rv]
        e = e[~np.isnan(e)]
        if len(e) and np.all(e <= 0):
            dead.append(rv)
    return ("ALIVE" if alive else "DEAD" if dead else "INCONCLUSIVE"), dead


def test(seeds=TEST_SEEDS, out=ROOT / "results", label="2D v2 test 1300-1599"):
    """Pre-registered H7 test. Writes out/kt7.json and out/fig_kt7.png. The smoke mode passes training seeds and a scratch dir."""
    tuned = json.loads((ROOT / "results/kt7_tuned_params.json").read_text())
    tp = {a: tuned["arms"][a]["params"] for a in ARMS}
    keys = [(a, v) for a, (knob, vals) in SWEEPS.items() for v in vals]
    pj = [("battery", a, i, tp[a] | ({SWEEPS[a][0]: float(v)} if SWEEPS[a][0] else {}), seeds) for i, (a, v) in enumerate(keys)]
    pj += [("battery", rv, "tuned_" + rv, tp[rv], seeds) for rv in RIVALS]
    v1_ring = json.loads((ROOT / "results/kt2_tuned_params.json").read_text())["arms"]["ring"]["params"]
    kj = [("kt2", a, -1, tp[a], seeds) for a in ARMS] + [("kt2", "ring", -1, v1_ring, seeds)]
    t0 = time.time()
    res_all = _pool(sorted(pj + kj, key=lambda j: (j[0] != "kt2", j[1] not in ("ring_gated", "ring_vg", "ring"))))
    bat = {i: o for k, a, i, o, _ in res_all if k == "battery"}
    auc = {a: o for k, a, i, o, _ in res_all if k == "kt2"}
    per = {a: [bat[i] for i, (aa, _) in enumerate(keys) if aa == a] for a in SWEEPS}
    Hs = {a: np.array([b["hold"] for b in per[a]]) for a in SWEEPS}
    Rs = {a: np.array([b["rel"] for b in per[a]]) for a in SWEEPS}
    Os = {a: np.array([b["off"] for b in per[a]]) for a in SWEEPS}
    m = lambda D, ix=None: {a: (v if ix is None else v[:, ix]).mean(-1) for a, v in D.items()}
    est = frontier_deltas(m(Hs), m(Rs), kt3.pareto_interp)
    boots = {rv: [] for rv in RIVALS}
    idx = boot_idx(len(seeds))
    for ix in idx:
        d = frontier_deltas(m(Hs, ix), m(Rs, ix), kt3.pareto_interp)
        for rv in RIVALS:
            boots[rv].append(d[rv])
    ci = {rv: np.nanquantile(np.array(b), [0.025, 0.975], axis=0) for rv, b in boots.items()}
    lo = {rv: np.where(np.isnan(est[rv]), np.nan, ci[rv][0]) for rv in RIVALS}
    est_dom = frontier_deltas(m(Hs), m(Rs), kt3.dominance_frontier)
    bd = {rv: [] for rv in RIVALS}
    for ix in boot_idx(len(seeds), n_boot=2000, seed=1):
        d = frontier_deltas(m(Hs, ix), m(Rs, ix), kt3.dominance_frontier)
        for rv in RIVALS:
            bd[rv].append(d[rv])
    ci_dom = {rv: np.nanquantile(np.array(b), [0.025, 0.975], axis=0) for rv, b in bd.items()}
    lo_dom = {rv: np.where(np.isnan(est_dom[rv]), np.nan, ci_dom[rv][0]) for rv in RIVALS}
    # premature-switch non-inferiority vs the tuned rivals (paired over seeds, ring at each gain)
    tuned_off = {rv: bat["tuned_" + rv]["off"] for rv in RIVALS}
    noninf = {rv: [paired_diff(Os["ring_gated"][j], tuned_off[rv], n_boot=10000) for j in range(len(GAINS))] for rv in RIVALS}
    off_ok = np.array([all(noninf[rv][j]["hi"] < NONINF_MARGIN_S for rv in RIVALS) for j in range(len(GAINS))])
    verdict, dead = gate(est, lo, off_ok, est_dom)
    verdict_dom, dead_dom = gate(est_dom, lo_dom, off_ok, est_dom)
    # amendment A7.1 (secondary, descriptive): three-way dominance on (hold, release, time off target), means over seeds;
    # rival points = its sweep + its tuned config
    rp = {rv: np.array([[b["hold"].mean(), b["rel"].mean(), b["off"].mean()] for b in per[rv] + [bat["tuned_" + rv]]]) for rv in RIVALS}
    dom3 = {rv: [dominated3(np.array([Hs["ring_gated"][j].mean(), Rs["ring_gated"][j].mean(), Os["ring_gated"][j].mean()]), rp[rv])
                 for j in range(len(GAINS))] for rv in RIVALS}
    # KT2 world, six arms
    others = [a for a in SIX if a != "ring_gated"]
    perm = arm_permutation_test(auc["ring_gated"], [auc[a] for a in others], n_perm=5000, seed=0)
    pair = {a: paired_diff(auc["ring_gated"], auc[a]) for a in others}
    for a, pa in zip(others, holm([pair[a]["p"] for a in others])):
        pair[a]["p_holm"] = float(pa)

    def ptdict(a, j, v):
        b = per[a][j]
        q = b["ratio"]
        return {"knob": SWEEPS[a][0], "value": None if v is None else float(v),
                "hold_mean_s": float(b["hold"].mean()), "hold_sem": float(b["hold"].std() / np.sqrt(len(b["hold"]))),
                "release_mean_s": float(b["rel"].mean()), "release_sem": float(b["rel"].std() / np.sqrt(len(b["rel"]))),
                "release_signed_mean_s": float(b["rel_signed"].mean()),
                "release_signed_q05_q50_q95_s": [float(x) for x in np.quantile(b["rel_signed"], [.05, .5, .95])],
                "frac_switched_before_crossing": float((b["rel_signed"] < 0).mean()),
                "ratio_at_switch_median": None if np.isnan(q).all() else float(np.nanmedian(q)),
                "time_off_target_mean_s": float(b["off"].mean()), "time_off_target_sem": float(b["off"].std() / np.sqrt(len(b["off"]))),
                "wrongful_switch_rate": float(b["wrong"].mean()), "on_target_at_4s": float(b["on_t4"].mean()),
                "on_decoy_at_5s": float(b["on_d5"].mean()), "sw_on_target_at_5s": float(b["on5"].mean())}
    dl = lambda e, c: [{"ring_gain": g, "delta_s": None if np.isnan(x) else float(x), "lo": None if np.isnan(x) else float(l),
                        "hi": None if np.isnan(x) else float(h)} for g, x, l, h in zip(GAINS, e, c[0], c[1])]
    res = {"test": "KT7 un-stubborn ring (H7)", "seeds": label, "n_seeds": len(seeds), "t_cross_s": kt3.T_CROSS,
           "tuned_params": tp,
           "points": {a: [ptdict(a, j, v) for j, v in enumerate(SWEEPS[a][1])] for a in SWEEPS},
           "ring_vs_rival": {rv: dl(est[rv], ci[rv]) for rv in RIVALS},
           "premature_switch_noninferiority": {"margin_s": NONINF_MARGIN_S, "rival_tuned_time_off_s": {rv: float(tuned_off[rv].mean()) for rv in RIVALS},
                                               "ring_minus_rival": {rv: [{"ring_gain": g, **noninf[rv][j]} for j, g in enumerate(GAINS)] for rv in RIVALS},
                                               "ok_per_gain": off_ok.tolist()},
           "gate": verdict, "dead_because": dead,
           "dominance_frontier_sensitivity": {"gate": verdict_dom, "dead_because": dead_dom, "delta": {rv: dl(est_dom[rv], ci_dom[rv]) for rv in RIVALS}},
           "rule": "ALIVE iff >= 2 adjacent ring gains with lower 95% CI of Delta_SM and Delta_acc_vg > 0 (pareto_interp) and the ring's sw "
                   f"time off target non-inferior to both tuned rivals (upper 95% CI of ring - rival < {NONINF_MARGIN_S} s); DEAD iff SM or "
                   "accumulator_vg Delta <= 0 wherever defined (dominance frontier if pareto_interp undefined at every gain); else INCONCLUSIVE",
           "secondary_3d_dominance": {"label": "Amendment A7.1, fixed before any test seed after seeing training-seed tuning data: "
                                               "descriptive, not a gate. A ring point is dominated by a rival if some rival point "
                                               "(sweep or tuned) has hold >=, release <= and time off target <= with one strict",
                                      "dominated_by": {rv: [{"ring_gain": g, "dominated": d} for g, d in zip(GAINS, dom3[rv])] for rv in RIVALS},
                                      "n_ring_gains_not_dominated_by_either": int(sum(not (a or b) for a, b in zip(*dom3.values())))},
           "kt2_world": {"arms": list(SIX), "aggregate_auc": {a: {"mean": float(auc[a].mean()), "sem": float(auc[a].std() / np.sqrt(len(auc[a])))} for a in auc},
                         "ring_gated_minus_best_other_permutation": perm, "ring_gated_minus_each": pair,
                         "ablation_ring_gated_minus_ring_vg": paired_diff(auc["ring_gated"], auc["ring_vg"]),
                         "reference_ring_gated_minus_v1_ring_kt2_tuned": paired_diff(auc["ring_gated"], auc["ring"]),
                         "note": "ring_vg (ablation) and ring (v1, KT2-tuned) are reported but not part of the six-arm test"},
           "wall_s": time.time() - t0,
           "provenance": stamp(seed=label, param_source="results/kt7_tuned_params.json + swept knob; results/kt2_tuned_params.json (v1 ring reference)")}
    res["kt2_world"]["time_limit_rate"] = latch_rates(tp | {"ring": v1_ring}, seeds)
    out.mkdir(parents=True, exist_ok=True)
    (out / "kt7.json").write_text(json.dumps(res, indent=1))
    figure(res, out / "fig_kt7.png")
    report(res)


def _latch(a):
    arm, p, seeds = a
    r = kt2.simulate(arm, p, seeds, conds=[kt2.CONDS[0]])
    return arm, float((~r["success"] & ~r["wrong_object_stop"] & (r["time_s"] >= kt2.T_MAX - 1e-9)).mean())


def latch_rates(tp, seeds):
    """No-fault KT2 world: fraction of test seeds that reach the 100 s limit without finishing (the v1 latch failure)."""
    with ProcessPoolExecutor(max_workers=PROCS) as ex:
        return dict(ex.map(_latch, [(a, p, seeds) for a, p in tp.items()]))


def report(res):
    for a, pts in res["points"].items():
        print(a, " ".join(f"({p['hold_mean_s']:.2f},{p['release_mean_s']:.2f},{p['time_off_target_mean_s']:.2f})" for p in pts))
    for rv in RIVALS:
        print(f"Delta_{rv}:", " ".join("nan" if d["delta_s"] is None else f"{d['delta_s']:+.2f}[{d['lo']:+.2f}]" for d in res["ring_vs_rival"][rv]))
    print("noninferior per gain:", res["premature_switch_noninferiority"]["ok_per_gain"])
    print("3D dominance, ring gains not dominated by either rival:", res["secondary_3d_dominance"]["n_ring_gains_not_dominated_by_either"])
    print("H7 gate:", res["gate"], res["dead_because"], "| dominance sensitivity:", res["dominance_frontier_sensitivity"]["gate"])
    k = res["kt2_world"]
    print("KT2 AUC:", {a: round(v["mean"], 3) for a, v in k["aggregate_auc"].items()})
    p = k["ring_gated_minus_best_other_permutation"]
    print(f"ring_gated - best other {p['est']:+.4f}, null [{p['null_lo']:+.4f}, {p['null_hi']:+.4f}], p {p['p_two_sided']:.3f}")
    print("time-limit rate (no fault):", k["time_limit_rate"])


# ------------------------------------------------------------------------------------------------ figures
STY = {"ring_gated": ("C1", "-o"), "ring_vg": ("C5", ":o"), "state_machine": ("C0", "-s"), "accumulator_vg": ("C2", "-^"),
       "vote_vg": ("0.4", "D"), "sum_vg": ("0.6", "P"), "ring_static_vg": ("0.2", "X")}


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for a, pts in res["points"].items():
        c, mk = STY[a]
        lab = a + (f" ({pts[0]['knob']} swept)" if pts[0]["knob"] else "")
        ax[0, 0].plot([p["hold_mean_s"] for p in pts], [p["release_mean_s"] for p in pts], mk, color=c, ms=4, lw=1, label=lab)
        ax[0, 1].plot([p["hold_mean_s"] for p in pts], [p["time_off_target_mean_s"] for p in pts], mk, color=c, ms=4, lw=1, label=lab)
    ax[0, 0].set(xlabel="hold a hidden target (s, cap 10)", ylabel="release a decoy (s, cap 20)", title="hold vs release (lower right is better)")
    ax[0, 1].set(xlabel="hold a hidden target (s, cap 10)", ylabel="time off target, premature-switch trial (s)",
                 title="hold vs premature switching (lower right is better)")
    ax[0, 0].legend(fontsize=7)
    for rv, c in (("state_machine", "C0"), ("accumulator_vg", "C2")):
        d = [x for x in res["ring_vs_rival"][rv] if x["delta_s"] is not None]
        if d:
            ax[1, 0].errorbar([x["ring_gain"] for x in d], [x["delta_s"] for x in d],
                              yerr=[[x["delta_s"] - x["lo"] for x in d], [x["hi"] - x["delta_s"] for x in d]], fmt="-o", color=c, ms=3, label=f"vs {rv}")
        dd = [x for x in res["dominance_frontier_sensitivity"]["delta"][rv] if x["delta_s"] is not None]
        ax[1, 0].plot([x["ring_gain"] for x in dd], [x["delta_s"] for x in dd], ":", color=c, lw=1, label=f"vs {rv} (dominance)")
    ax[1, 0].axhline(0, color="k", lw=0.8)
    ax[1, 0].set(xlabel="ring_gated ring_gain", ylabel="rival release - ring release at matched hold (s)", title=f"> 0: ring above rival. H7 {res['gate']}")
    ax[1, 0].legend(fontsize=7)
    k = res["kt2_world"]["aggregate_auc"]
    names = list(SIX) + ["ring_vg", "ring"]
    ax[1, 1].bar(range(len(names)), [k[a]["mean"] for a in names], yerr=[1.96 * k[a]["sem"] for a in names],
                 color=["C1"] + ["0.6"] * 5 + ["C5", "0.85"])
    ax[1, 1].set_xticks(range(len(names)), [n if n != "ring" else "ring (v1)" for n in names], rotation=35, fontsize=7)
    p = res["kt2_world"]["ring_gated_minus_best_other_permutation"]
    ax[1, 1].set(ylim=(0, 1), ylabel="aggregate success-vs-severity AUC", title=f"KT2 world, test seeds: ring_gated - best other {p['est']:+.3f} (p {p['p_two_sided']:.2f})")
    fig.suptitle("KT7 (H7), 2D test seeds 1300-1599")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def figure_envelope(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for a, cfs in out["configs"].items():
        c, mk = STY[a]
        if a not in ("ring_gated", "ring_vg", "state_machine", "accumulator_vg"):
            continue
        ax[0].scatter([x["hold_mean_s"] for x in cfs], [x["release_mean_s"] for x in cfs], s=12, color=c, marker=mk[-1], alpha=0.45, label=f"{a}: 64 configs")
        env = sorted((cfs[i] for i in out["envelope"][a]), key=lambda x: x["hold_mean_s"])
        ax[0].step([x["hold_mean_s"] for x in env], [x["release_mean_s"] for x in env], where="pre", color=c, lw=1.5)
        ax[1].scatter([x["hold_mean_s"] for x in cfs], [x["time_off_mean_s"] for x in cfs], s=12, color=c, marker=mk[-1], alpha=0.45, label=a)
    ax[0].set(xlabel="hold (s, cap 10)", ylabel="release (s, cap 20)", title="training seeds 0-99: every random config (lower right is better)")
    ax[1].set(xlabel="hold (s, cap 10)", ylabel="time off target, premature-switch trial (s)", title="hold vs premature switching")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "results/fig_kt7_envelope.png", dpi=120)


def smoke():
    """<= 2 min run of the whole --test code path on 6 TRAINING seeds, written to a temp dir (never results/)."""
    import tempfile
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        test(seeds=TRAIN_SEEDS[:6], out=Path(d), label="SMOKE train 0-5")
        r = json.loads((Path(d) / "kt7.json").read_text())
        assert r["gate"] in ("ALIVE", "DEAD", "INCONCLUSIVE") and (Path(d) / "fig_kt7.png").stat().st_size > 0
        assert len(r["points"]["ring_gated"]) == len(GAINS) and set(r["kt2_world"]["aggregate_auc"]) == set(ARMS) | {"ring"}
        assert set(r["kt2_world"]["time_limit_rate"]) == set(ARMS) | {"ring"}
    print(f"kt7 smoke OK ({time.time() - t0:.0f} s)")


# ------------------------------------------------------------------------------------------------ self-check
def selfcheck():
    s = np.arange(4)
    # unchanged KT3 trials run v2 arms; the vis-gated accumulator holds a hidden target that the v1-style vote loses
    h, _ = kt3.trial("vote_vg", {"tau_hold_s": 0.1}, s, "occ")
    assert (h < 0.5).all(), h
    h, _ = kt3.trial("accumulator_vg", {"tau_hold_s": 20.0, "tau_leak_s": 1.0, "g_search": 0.1}, s, "occ")
    assert (h >= kt3.HOLD_CAP).all(), h
    # sw trial: a memoryless vote follows the long look-alike spike; a slow accumulator rides the dips out
    off, wrong, on5 = trial_sw("vote_vg", {}, s)
    assert on5.all() and wrong.all() and (off > 1.0).all(), (off, wrong)
    off2, wrong2, on52 = trial_sw("accumulator_vg", {"tau_leak_s": 5.0}, s)
    assert on52.all() and not wrong2.any() and (off2 < off).all(), (off2, wrong2)
    assert 0 <= battery_score(battery("ring_gated", {}, s)) <= 1
    # same seed -> same sw trial, whatever the batch composition
    a, _, _ = trial_sw("ring_gated", {}, np.arange(6))
    b, _, _ = trial_sw("ring_gated", {}, np.arange(3, 6))
    assert np.array_equal(a[3:], b)
    # KT2 world: dropping finished rows changes nothing for the v2 arms; RingGated at high gain does not latch where v1 does
    sub = [kt2.CONDS[0], kt2.CONDS[4], kt2.CONDS[8], kt2.CONDS[20]]
    for arm in ("ring_gated", "accumulator_vg", "vote_vg"):
        x1 = kt2.simulate(arm, {}, np.arange(6), conds=sub, compact_every=50)
        x2 = kt2.simulate(arm, {}, np.arange(6), conds=sub, compact_every=10 ** 9)
        assert all(np.array_equal(x1[k], x2[k], equal_nan=True) for k in ("success", "time_s", "t_hold_occlusion_s")), arm
    hi = {"ring_gain": 2.5, "g_home": 1.0, "g_obj": 1.0, "tau_s": 0.2}
    rg = kt2.simulate("ring_gated", {**hi, "lo_frac": 0.3, "tau_m_s": 0.3, "theta_c": 0.1}, np.arange(10), conds=[kt2.CONDS[0]])
    r1 = kt2.simulate("ring", hi, np.arange(10), conds=[kt2.CONDS[0]])
    print("clean KT2 success at gain 2.5: ring_gated", rg["success"].mean(), "v1 ring", r1["success"].mean())
    assert rg["success"].mean() >= r1["success"].mean() + 0.3
    # gate logic
    e = {"state_machine": np.array([1., 1, -1]), "accumulator_vg": np.array([1., 1, 1])}
    assert gate(e, e, np.array([True, True, True]), e)[0] == "ALIVE"
    assert gate(e, e, np.array([True, False, True]), e)[0] == "INCONCLUSIVE"
    n = {"state_machine": np.array([np.nan] * 3), "accumulator_vg": np.array([1., 1, 1])}
    assert gate(n, n, np.ones(3, bool), {"state_machine": np.array([-1., -1, -1]), "accumulator_vg": e["accumulator_vg"]}) == ("DEAD", ["state_machine"])
    P = np.array([[10.0, 0.0, 2.0], [7.0, 1.2, 0.0]])
    assert dominated3(np.array([10.0, 0.5, 2.0]), P) and not dominated3(np.array([10.0, 0.5, 1.0]), P)
    assert not dominated3(np.array([10.0, 0.0, 2.0]), P)
    assert len(configs("ring_gated")) == 64 and configs("vote_vg")[5]["kappa"] == kt2.configs("vote")[5]["kappa"]
    print("kt7 self-check OK")


if __name__ == "__main__":
    {"--pilot": pilot, "--tune": tune, "--test": test, "--selfcheck": selfcheck, "--smoke": smoke}[sys.argv[1]]()
