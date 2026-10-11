"""KT2 (H1 pilot): six arbiters in a 2D point-mass find-and-return mission, equal tuning, paired test.

Pre-registered in docs/LOG.md ([WS-A] KT2). The 2D world (no obstacles, no rendering):
  agent: constant 2 m/s, turn rate <= 120 deg/s, 50 Hz; heading from a compass (noise + faults);
  path integrator (PI) integrates odometry speed along the compass heading (Stone 2017 in spirit,
  vector form, not the neural model); home drive = PI home vector, or the true home direction once
  home is within 8 m (a visual home landmark); search drive = a noisy prior direction plus a +-45 deg
  zigzag, out/back legs on PI distance; objects (target, optional look-alike decoy) give
  strength = quality * (1 - 0.4 d/20) + N(0, 0.08), clipped to [0, 1], visible within 20 m.
  Shared stop rule: within 2 m of an object whose current strength > 0.6.
  Gust is represented as heading-disturbance noise (deg/sqrt(s)), not wind drift.

Usage:
  uv run python scripts/kt2_five_arms.py --pilot      # time one config per arm, log extrapolation
  uv run python scripts/kt2_five_arms.py --tune       # 64 configs/arm on training seeds 0-99 -> results/kt2_tuned_params.json
  uv run python scripts/kt2_five_arms.py --test       # frozen params on test seeds 1000-1299 -> results/kt2.json, parquet, fig
  uv run python scripts/kt2_five_arms.py --selfcheck
"""
import hashlib, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per process (we parallelise over processes)
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from brain.arbiters import ARBITERS, CLASSES, make, sample_params, wrap
from rng import Rng
from provenance import stamp

ROOT = Path(__file__).resolve().parents[1]
DT, V, OMEGA = 0.02, 2.0, np.deg2rad(120.0)
T_MAX = 100.0
R_VIS, R_HOME_VIS, STOP_R, STOP_THR = 20.0, 8.0, 2.0, 0.6
SIG_VIS, SIG_COMPASS, SIG_ODO = 0.08, np.deg2rad(2.0), 0.05
N_COLS = 36
GUST_DEG = {"none": 5.0, "light": 20.0, "moderate": 45.0, "severe": 90.0}  # heading noise, deg/sqrt(s)
FAULTS = {"occlusion": [1.0, 4.0, 7.0, 10.0], "decoy": [0.5, 0.65, 0.8, 0.95], "compass_half": [0.25, 0.5, 0.75, 1.0],
          "eye_columns": [0.1, 0.2, 0.3, 0.4], "false_sun": [30.0, 60.0, 90.0, 120.0],
          "speed_scale": [0.075, 0.15, 0.225, 0.3], "gust": ["light", "moderate", "severe"]}
CONDS = [("none", 0, None)] + [(f, i + 1, v) for f, vs in FAULTS.items() for i, v in enumerate(vs)]
TRAIN_SEEDS, TEST_SEEDS = np.arange(0, 100), np.arange(1000, 1300)
N_CONFIGS, SEARCH_SEED = 64, 2026


def decoy_quality(d, s):
    """Look-alike: quality s from >= 8 m, falling linearly to s^2 at 2 m (evidence turns against it)."""
    return np.where(d >= 8, s, s - (s - s * s) * np.clip((8 - d) / 6, 0, 1))


def _compact(obj, keep, B):
    """Drop finished rows from any object holding per-row numpy state (arbiter or Rng)."""
    for k, v in vars(obj).items():
        if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == B and k != "W":
            setattr(obj, k, v[keep])
    if hasattr(obj, "B"):
        obj.B = int(keep.sum())


OUT_KEYS = ("success_target", "success_home", "wrong_object_stop", "t_end", "path", "turn_sum", "t_hold", "t_rel",
            "t_cross", "t_first", "occ_on", "pos")


def simulate(arm, params, seeds, conds=CONDS, track=False, compact_every=250):
    """Run every (cond, seed) row in one batch. Returns dict of per-row arrays.
    Finished rows are dropped from the batch every `compact_every` ticks (results unchanged)."""
    cond_idx = np.repeat(np.arange(len(conds)), len(seeds))
    seed = np.tile(np.asarray(seeds), len(conds))
    B = len(seed)
    val = lambda f, default: np.array([conds[c][2] if conds[c][0] == f else default for c in cond_idx], dtype=object)
    S = {"occ_dur": val("occlusion", 0.0).astype(float), "dec_s": val("decoy", np.nan).astype(float),
         "comp_a": val("compass_half", 0.0).astype(float),
         "n_dead": np.round(val("eye_columns", 0.0).astype(float) * N_COLS).astype(int),
         "sun": np.deg2rad(val("false_sun", 0.0).astype(float)), "spd_m": val("speed_scale", 0.0).astype(float),
         "gust": np.deg2rad(np.array([GUST_DEG[g] for g in val("gust", "none")]))}
    rngs = {k: Rng(seed, s) for k, s in (("r0", 0), ("r5", 5), ("r6", 6), ("rg", 1), ("rc", 2), ("ro", 3), ("rv", 4))}
    r0, r5, r6 = rngs["r0"], rngs["r5"], rngs["r6"]
    d_t, phi_t = 30 + 10 * r0.uniform(), 2 * np.pi * r0.uniform()
    S["h"] = 2 * np.pi * r0.uniform()
    swap = r5.uniform() < 0.5
    S["occ_delay"] = 1 + 2 * r5.uniform()
    dec_ang = np.deg2rad(20 + 30 * r5.uniform()) * np.sign(r5.uniform() - 0.5)
    comp_phi = 2 * np.pi * r5.uniform()
    S["col0"] = np.floor(N_COLS * r5.uniform()).astype(int)
    S["sun_sign"], S["sun_on"] = np.sign(r5.uniform() - 0.5), 5 + 35 * r5.uniform()
    S["spd"] = 1 + S["spd_m"] * np.sign(r5.uniform() - 0.5)
    S["prior"] = phi_t + np.deg2rad(20) * r6.normal()
    S["zz"] = 2 * np.pi * r6.uniform()
    tgt = d_t[:, None] * np.stack([np.cos(phi_t), np.sin(phi_t)], 1)
    dec = tgt + 15 * np.stack([np.cos(phi_t + np.pi + dec_ang), np.sin(phi_t + np.pi + dec_ang)], 1)
    S["has_dec"] = ~np.isnan(S["dec_s"])
    S["obj"] = np.where(swap[:, None, None], np.stack([dec, tgt], 1), np.stack([tgt, dec], 1))  # [B, 2, 2]
    S["is_t"] = np.stack([~swap, swap], 1)
    S["s_dec"] = np.where(S["has_dec"], S["dec_s"], 0.0)[:, None]
    cell = np.arange(8) * np.pi / 4
    S["comp_gain"] = 1 - S["comp_a"][:, None] * (wrap(cell[None] - comp_phi[:, None]) % (2 * np.pi) < np.pi)
    S |= {"pos": np.zeros((B, 2)), "phat": np.zeros((B, 2)), "phase": np.zeros(B, int), "leg": np.zeros(B, int),
          "last_b": np.zeros((B, 2)), "t_first": np.full(B, np.nan), "success_target": np.zeros(B, bool),
          "success_home": np.zeros(B, bool), "wrong_object_stop": np.zeros(B, bool), "t_end": np.full(B, T_MAX),
          "path": np.zeros(B), "turn_sum": np.zeros(B), "hold_alive": np.ones(B, bool), "t_hold": np.full(B, np.nan),
          "dec_seen": np.zeros(B, bool), "t_cross": np.full(B, np.nan), "t_rel": np.full(B, np.nan), "occ_on": np.full(B, np.nan)}
    full = {k: S[k].copy() for k in OUT_KEYS}
    idx = np.arange(B)
    arb = make(arm, params, B)
    trace = []
    for k in range(int(round(T_MAX / DT))):
        t = k * DT
        nb = len(idx)
        if k and k % compact_every == 0 and not track:
            keep = S["phase"] < 2
            if not keep.any():
                break
            if keep.mean() < 0.9:
                for key in OUT_KEYS:
                    full[key][idx] = S[key]
                S = {key: v[keep] for key, v in S.items()}
                for o in (arb, *rngs.values()):
                    _compact(o, keep, nb)
                idx, nb = idx[keep], int(keep.sum())
        g = S
        h, pos, phat, phase = g["h"], g["pos"], g["phat"], g["phase"]
        act = phase < 2
        if not act.any():
            break
        is_t = g["is_t"]
        # compass estimate: noise, half-silenced TB1-like population, false sun window
        z = (np.cos(h[:, None] - cell[None]) * g["comp_gain"] * np.exp(1j * cell[None])).sum(1)
        bias = wrap(np.angle(z) - h) + np.where((t >= g["sun_on"]) & (t < g["sun_on"] + 20), g["sun_sign"] * g["sun"], 0.0)
        hhat = h + SIG_COMPASS * rngs["rc"].normal() + bias
        # objects
        rel = g["obj"] - pos[:, None]
        dist = np.hypot(rel[..., 0], rel[..., 1])
        wb = np.arctan2(rel[..., 1], rel[..., 0])
        col = np.floor((wb - h[:, None]) % (2 * np.pi) / (2 * np.pi / N_COLS)).astype(int)
        dead = (col - g["col0"][:, None]) % N_COLS < g["n_dead"][:, None]
        geo = (dist < R_VIS) & ~dead & (phase == 0)[:, None] & (is_t | g["has_dec"][:, None])
        tv = (geo & is_t).any(1)
        g["t_first"] = np.where(np.isnan(g["t_first"]) & tv, t, g["t_first"])
        g["occ_on"] = occ_on = g["t_first"] + g["occ_delay"]
        occ = (t >= occ_on) & (t < occ_on + g["occ_dur"])
        vis = geo & ~(is_t & occ[:, None])
        quality = np.where(is_t, 1.0, decoy_quality(dist, g["s_dec"]))
        ev = quality * (1 - 0.4 * dist / R_VIS)
        noise = np.stack([rngs["rv"].normal(), rngs["rv"].normal()], 1)
        strength = np.clip(ev + SIG_VIS * noise, 0, 1) * vis
        bear = wrap(wb - h[:, None] + hhat[:, None])
        g["last_b"] = np.where(vis, bear, g["last_b"])
        # home and search drives
        hd = np.hypot(pos[:, 0], pos[:, 1])
        hb_true = np.arctan2(-pos[:, 1], -pos[:, 0])
        hcol = np.floor((hb_true - h) % (2 * np.pi) / (2 * np.pi / N_COLS)).astype(int)
        hvis = (phase == 1) & (hd < R_HOME_VIS) & ~((hcol - g["col0"]) % N_COLS < g["n_dead"])
        home_b = np.where(hvis, wrap(hb_true - h + hhat), np.arctan2(-phat[:, 1], -phat[:, 0]))
        pid = np.hypot(phat[:, 0], phat[:, 1])
        leg = g["leg"] = np.where((g["leg"] == 0) & (pid > 50), 1, np.where((g["leg"] == 1) & (pid < 10), 0, g["leg"]))
        sb = wrap(g["prior"] + np.pi * leg + np.deg2rad(45) * np.sin(2 * np.pi * t / 16 + g["zz"]))
        obs = {"objects": {"bearing": g["last_b"].copy(), "strength": strength, "visible": vis},
               "home": {"bearing": home_b, "strength": (phase == 1).astype(float)},
               "search": {"bearing": sb, "strength": (phase == 0).astype(float)},
               "heading": wrap(hhat)}
        cmd = arb.step(obs, DT)["heading_cmd"]
        # occlusion hold / decoy release metrics (true target bearing in the compass frame)
        on_t = np.abs(wrap(cmd - bear[np.arange(nb), is_t.argmax(1)])) <= np.deg2rad(20)
        lose = occ & g["hold_alive"] & ~on_t & act
        g["t_hold"] = np.where(lose, t - occ_on, g["t_hold"])
        g["hold_alive"] &= ~lose
        evg = np.where(geo, ev, 0.0)
        ev_t, ev_d = (evg * is_t).sum(1), (evg * ~is_t).sum(1)
        g["dec_seen"] |= ev_d > 0
        newc = np.isnan(g["t_cross"]) & g["dec_seen"] & (ev_t > 1.5 * ev_d) & (phase == 0)
        g["t_cross"] = np.where(newc, t, g["t_cross"])
        rel_now = ~np.isnan(g["t_cross"]) & np.isnan(g["t_rel"]) & on_t & act
        g["t_rel"] = np.where(rel_now, t - g["t_cross"], g["t_rel"])
        # shared stop rule
        cand = vis & (dist < STOP_R) & (strength > STOP_THR) & (phase == 0)[:, None]
        st_t = (cand & is_t).any(1)
        st_d = (cand & ~is_t).any(1) & ~st_t
        g["success_target"] |= st_t
        g["wrong_object_stop"] |= st_d
        g["t_end"] = np.where(st_d, t, g["t_end"])
        phase = np.where(st_t, 1, np.where(st_d, 2, phase))
        home_ok = (phase == 1) & (hd < STOP_R)
        g["success_home"] |= home_ok
        g["t_end"] = np.where(home_ok, t, g["t_end"])
        g["phase"] = phase = np.where(home_ok, 2, phase)
        if track:
            trace.append((pos.copy(), h.copy(), cmd.copy(), phase.copy(), strength.copy()))
        # kinematics and odometry (only rows still running)
        m = phase < 2
        turn = np.clip(wrap(cmd - hhat), -OMEGA * DT, OMEGA * DT)
        g["h"] = h = np.where(m, h + turn + g["gust"] * np.sqrt(DT) * rngs["rg"].normal(), h)
        g["pos"] = np.where(m[:, None], pos + V * DT * np.stack([np.cos(h), np.sin(h)], 1), pos)
        v_odo = V * g["spd"] * (1 + SIG_ODO * rngs["ro"].normal())
        g["phat"] = np.where(m[:, None], phat + v_odo[:, None] * DT * np.stack([np.cos(hhat), np.sin(hhat)], 1), phat)
        g["path"] = g["path"] + m * V * DT
        g["turn_sum"] = g["turn_sum"] + m * np.abs(turn)
    for key in OUT_KEYS:
        full[key][idx] = S[key]
    occ_dur, has_dec = val("occlusion", 0.0).astype(float), ~np.isnan(val("decoy", np.nan).astype(float))
    t_end, t_hold, t_rel, t_cross, occ_on = full["t_end"], full["t_hold"], full["t_rel"], full["t_cross"], full["occ_on"]
    started = ~np.isnan(occ_on) & (occ_dur > 0)
    t_hold = np.where(started & np.isnan(t_hold) & (occ_on < t_end), np.minimum(occ_dur, t_end - occ_on), t_hold)
    t_rel = np.where(~np.isnan(t_cross) & np.isnan(t_rel), np.minimum(20.0, t_end - t_cross), np.minimum(t_rel, 20.0))
    res = {"cond": cond_idx, "seed": seed, "fault": np.array([conds[c][0] for c in cond_idx]),
           **{k: full[k] for k in ("success_target", "success_home", "wrong_object_stop")},
           "success": full["success_target"] & full["success_home"], "time_s": t_end, "path_len_m": full["path"],
           "home_error_m": np.hypot(full["pos"][:, 0], full["pos"][:, 1]), "energy_proxy": full["turn_sum"],
           "t_hold_occlusion_s": np.where(occ_dur > 0, t_hold, np.nan),
           "t_release_decoy_s": np.where(has_dec, t_rel, np.nan)}
    if track:
        res["trace"] = trace
    return res


def auc_per_seed(res, n_seeds):
    """Success-vs-severity AUC per (fault, seed): trapezoid over levels (0 = no fault), x scaled to [0, 1]."""
    succ = res["success"].reshape(len(CONDS), n_seeds).astype(float)
    out = {}
    for f, vs in FAULTS.items():
        ys = np.vstack([succ[0]] + [succ[i] for i, c in enumerate(CONDS) if c[0] == f])
        x = np.linspace(0, 1, len(ys))
        out[f] = ((ys[1:] + ys[:-1]) / 2 * np.diff(x)[:, None]).sum(0)
    return out


def objective(arm, params, seeds):
    a = auc_per_seed(simulate(arm, params, seeds), len(seeds))
    return float(np.mean([v.mean() for v in a.values()]))


def configs(arm):
    names = list(CLASSES[arm].PARAM_SPACE)
    U = np.random.default_rng(SEARCH_SEED).random((N_CONFIGS, 16))[:, :len(names)]
    return [sample_params(arm, u) for u in U]


def _job(a):
    arm, i, p, seeds = a
    t0 = time.time()
    return arm, i, p, objective(arm, p, seeds), time.time() - t0


def pilot():
    t0 = time.time()
    times = {}
    for arm in ARBITERS:
        t1 = time.time()
        objective(arm, configs(arm)[0], TRAIN_SEEDS)
        times[arm] = time.time() - t1
    total = sum(times.values())
    print(f"pilot (1 config x 100 train seeds x {len(CONDS)} conds per arm): " + ", ".join(f"{k} {v:.1f}s" for k, v in times.items()))
    print(f"extrapolated tuning CPU: {total * N_CONFIGS / 60:.1f} min; test CPU ~{total * 3 / 60:.1f} min")
    return times


RUNG1_SEEDS, RUNG2_TOP = np.arange(0, 20), 8


def tune():
    """Amendment A1: rung 1 = 64 configs x train seeds 0-19; rung 2 = top 8 x train seeds 0-99."""
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        r1 = list(ex.map(_job, [(arm, i, p, RUNG1_SEEDS) for arm in ARBITERS for i, p in enumerate(configs(arm))]))
        top = [x for arm in ARBITERS for x in sorted([x for x in r1 if x[0] == arm], key=lambda x: -x[3])[:RUNG2_TOP]]
        r2 = list(ex.map(_job, [(arm, i, p, TRAIN_SEEDS) for arm, i, p, _, _ in top]))
    arms = {}
    for arm in ARBITERS:
        r = sorted([x for x in r2 if x[0] == arm], key=lambda x: -x[3])
        arms[arm] = {"params": r[0][2], "train_auc": r[0][3], "config_index": r[0][1], "n_params": len(CLASSES[arm].PARAM_SPACE),
                     "rung1_auc": [x[3] for x in sorted([x for x in r1 if x[0] == arm], key=lambda x: x[1])],
                     "rung2": {int(x[1]): x[3] for x in r}}
    out = {"arms": arms, "budget": f"{N_CONFIGS} random configs (numpy default_rng({SEARCH_SEED})); rung 1 train seeds 0-19, "
                                   f"rung 2 top {RUNG2_TOP} on train seeds 0-99; {len(CONDS)} conds",
           "objective": "mean over 7 faults of mean-over-seeds success-vs-severity AUC", "wall_s": time.time() - t0,
           "cpu_s": sum(x[4] for x in r1 + r2), "provenance": stamp(seed="train 0-99", param_source="random search, amendment A1")}
    (ROOT / "results/kt2_tuned_params.json").write_text(json.dumps(out, indent=1))
    for a, v in arms.items():
        print(f"{a:14s} train AUC {v['train_auc']:.3f}  {v['params']}")
    print(f"tuning wall {out['wall_s']:.0f}s cpu {out['cpu_s'] / 60:.1f} min")


def _test_job(arm):
    p = json.loads((ROOT / "results/kt2_tuned_params.json").read_text())["arms"][arm]["params"]
    return arm, simulate(arm, p, TEST_SEEDS)


def test():
    from stats import paired_bootstrap, paired_diff, holm
    import pandas as pd
    tuned = json.loads((ROOT / "results/kt2_tuned_params.json").read_text())
    with ProcessPoolExecutor(max_workers=6) as ex:
        runs = dict(ex.map(_test_job, ARBITERS))
    n = len(TEST_SEEDS)
    auc = {a: auc_per_seed(r, n) for a, r in runs.items()}
    agg = {a: np.mean([v for v in auc[a].values()], 0) for a in ARBITERS}  # [n] per-seed equal-weight AUC
    others = [a for a in ARBITERS if a != "ring"]
    best_other = paired_bootstrap(lambda r, *o: r.mean() - max(x.mean() for x in o), agg["ring"], *[agg[a] for a in others])
    pair = {a: paired_diff(agg["ring"], agg[a]) for a in others}
    adj = holm([pair[a]["p"] for a in others])
    for a, pa in zip(others, adj):
        pair[a]["p_holm"] = float(pa)
    per_fault = {a: {f: float(v.mean()) for f, v in auc[a].items()} for a in ARBITERS}
    succ_lvl = {a: {f"{c[0]}:{c[2]}": float(runs[a]["success"][runs[a]["cond"] == i].mean()) for i, c in enumerate(CONDS)} for a in ARBITERS}
    worst = {a: {"fault": min(per_fault[a], key=per_fault[a].get), "auc": min(per_fault[a].values()),
                 "worst_condition": min(succ_lvl[a], key=succ_lvl[a].get), "worst_condition_success": min(succ_lvl[a].values())} for a in ARBITERS}
    gate = "H1 likely null" if best_other["lo"] <= 0 <= best_other["hi"] else ("ring better than best other" if best_other["lo"] > 0 else "ring worse than best other")
    res = {"test": "KT2 six arbiters, 2D point-mass mission (H1 pilot)", "seeds": "test 1000-1299", "n_seeds": n,
           "aggregate_auc": {a: {"mean": float(agg[a].mean()), **{k: v for k, v in paired_bootstrap(np.mean, agg[a]).items() if k in ("lo", "hi")}} for a in ARBITERS},
           "ring_minus_best_other": best_other, "best_other_point": max(others, key=lambda a: agg[a].mean()),
           "ring_minus_each": pair, "per_fault_auc": per_fault, "success_by_condition": succ_lvl, "worst": worst,
           "wrong_object_stop_rate": {a: float(runs[a]["wrong_object_stop"].mean()) for a in ARBITERS},
           "t_hold_occlusion_mean_s": {a: float(np.nanmean(runs[a]["t_hold_occlusion_s"])) for a in ARBITERS},
           "t_release_decoy_mean_s": {a: float(np.nanmean(runs[a]["t_release_decoy_s"])) for a in ARBITERS},
           "gate": gate, "tuned_params": {a: tuned["arms"][a]["params"] for a in ARBITERS},
           "provenance": stamp(seed="test 1000-1299", param_source="results/kt2_tuned_params.json")}
    (ROOT / "results/kt2.json").write_text(json.dumps(res, indent=1))
    prov = res["provenance"]
    rows = []
    for a, r in runs.items():
        for i in range(len(r["seed"])):
            c = CONDS[r["cond"][i]]
            rows.append({"scenario_hash": hashlib.sha1(f"kt2|{c[0]}|{c[2]}|{r['seed'][i]}".encode()).hexdigest()[:12],
                         "seed": int(r["seed"][i]), "arbiter": a, "fault": c[0], "severity": str(c[2]), "level": c[1],
                         "ring_gain": tuned["arms"][a]["params"].get("ring_gain", np.nan) if a == "ring" else (0.0 if a == "ring_static" else np.nan),
                         "split": "test", "success_target": bool(r["success_target"][i]), "success_home": bool(r["success_home"][i]),
                         "time_s": float(r["time_s"][i]), "path_len_m": float(r["path_len_m"][i]), "collisions": 0,
                         "home_error_m": float(r["home_error_m"][i]), "wrong_object_stop": bool(r["wrong_object_stop"][i]),
                         "max_bank_deg": np.nan, "energy_proxy": float(r["energy_proxy"][i]),
                         "t_hold_occlusion_s": float(r["t_hold_occlusion_s"][i]), "t_release_decoy_s": float(r["t_release_decoy_s"][i]),
                         "git": prov["git"], "sim_version": prov["sim_version"], "created": prov["created"]})
    (ROOT / "results/runs").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(ROOT / "results/runs/kt2_test.parquet", index=False)
    figure(res, runs)
    for a in ARBITERS:
        print(f"{a:14s} AUC {agg[a].mean():.3f}  worst {worst[a]['fault']} ({worst[a]['auc']:.2f})  wrong stops {res['wrong_object_stop_rate'][a]:.3f}")
    b = best_other
    print(f"ring - best other ({res['best_other_point']}): {b['est']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}] -> {gate}")


COLORS = {"ring": "C1", "ring_static": "0.25", "vote": "C0", "sum": "C2", "accumulator": "C3", "state_machine": "C4"}


def figure(res, runs=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 4, figsize=(15, 7))
    for ax, f in zip(axs.ravel(), FAULTS):
        for a in ARBITERS:
            ys = [res["success_by_condition"][a]["none:None"]] + [res["success_by_condition"][a][f"{f}:{v}"] for v in FAULTS[f]]
            ax.plot(range(len(ys)), ys, "-o", ms=3, label=a, color=COLORS[a], lw=2.2 if a == "ring" else 1)
        ax.set_xticks(range(len(ys)), ["0"] + [str(v) for v in FAULTS[f]], fontsize=7)
        ax.set(title=f, ylim=(0, 1.02), ylabel="success")
    ax = axs.ravel()[-1]
    m = [res["aggregate_auc"][a]["mean"] for a in ARBITERS]
    lo = [res["aggregate_auc"][a]["mean"] - res["aggregate_auc"][a]["lo"] for a in ARBITERS]
    hi = [res["aggregate_auc"][a]["hi"] - res["aggregate_auc"][a]["mean"] for a in ARBITERS]
    ax.bar(range(6), m, yerr=[lo, hi], color=["C1"] + ["0.6"] * 5)
    ax.set_xticks(range(6), ARBITERS, rotation=40, fontsize=7)
    ax.set(title="aggregate AUC (95% CI)", ylim=(0, 1))
    axs.ravel()[0].legend(fontsize=6)
    fig.suptitle(f"KT2 test seeds 1000-1299: ring - best other = {res['ring_minus_best_other']['est']:+.3f} "
                 f"[{res['ring_minus_best_other']['lo']:+.3f}, {res['ring_minus_best_other']['hi']:+.3f}]")
    fig.tight_layout()
    fig.savefig(ROOT / "results/fig_kt2.png", dpi=120)


def selfcheck():
    seeds = np.arange(3)
    r = simulate("vote", {}, seeds, conds=[("none", 0, None)])
    assert r["success"].shape == (3,)
    # determinism and pairing: same seed -> same world draws regardless of arm
    r2 = simulate("vote", {}, seeds, conds=[("none", 0, None)])
    assert np.array_equal(r["time_s"], r2["time_s"])
    # decoy quality turns against the decoy up close and never exceeds s
    q = decoy_quality(np.array([20, 8, 5, 2, 0.5]), 0.8)
    assert np.allclose(q[[0, 1, 3, 4]], [0.8, 0.8, 0.64, 0.64]) and 0.64 < q[2] < 0.8
    # half-silenced compass with zero attenuation is unbiased; full gust/noise conds run
    r = simulate("sum", {}, seeds, conds=[("compass_half", 1, 0.0), ("gust", 3, "severe"), ("occlusion", 4, 10.0), ("decoy", 4, 0.95)])
    assert np.isfinite(r["time_s"]).all() and (r["path_len_m"] > 0).all()
    assert np.isnan(r["t_hold_occlusion_s"][:6]).all() and np.isnan(r["t_release_decoy_s"][:9]).all()
    # dropping finished rows from the batch changes nothing (every arm, with faults)
    sub = [CONDS[0], CONDS[4], CONDS[8], CONDS[20], CONDS[27]]
    for arm in ARBITERS:
        a1 = simulate(arm, {}, np.arange(6), conds=sub, compact_every=50)
        a2 = simulate(arm, {}, np.arange(6), conds=sub, compact_every=10 ** 9)
        for key in ("success", "time_s", "path_len_m", "home_error_m", "t_hold_occlusion_s", "t_release_decoy_s"):
            assert np.array_equal(a1[key], a2[key], equal_nan=True), (arm, key)
    # the clean mission is solvable: a well-set vote finds and returns on most seeds
    r = simulate("vote", {}, np.arange(20), conds=[("none", 0, None)])
    assert r["success"].mean() >= 0.7, r["success"].mean()
    a = auc_per_seed(simulate("vote", {}, seeds), 3)
    assert set(a) == set(FAULTS) and all(((v >= 0) & (v <= 1)).all() for v in a.values())
    print("kt2 self-check OK; clean vote success", r["success"].mean())


if __name__ == "__main__":
    {"--pilot": pilot, "--tune": tune, "--test": test, "--selfcheck": selfcheck}[sys.argv[1]]()
