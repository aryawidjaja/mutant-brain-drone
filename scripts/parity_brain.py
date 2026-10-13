"""Brain + mission parity: Python reference vs the JS twin (Node), on fixed seeds. Writes results/parity_brain.json.

Module outputs are compared on recorded inputs (exact or <= 1e-9); full closed-loop missions are compared on the
trajectory with a stated tolerance (closed loops amplify floating-point order differences between numpy and V8):
  scenarios, hashes, worlds ............ exact
  eye panorama pixels .................. <= 0.5 % of pixels differ, by <= 1 level
  object looks (grey) .................. max |diff| <= 1e-9
  MB familiarity on Python's looks ..... exact;  on each side's own looks + same noise: max |diff| <= 0.02
  path integrator (recorded inputs) .... heading <= 1e-9 rad, home vector <= 1e-6 m
  six arbiters (recorded obs) .......... heading <= 1e-9 rad, commit <= 1e-9
  avoidance reflex ..................... <= 1e-12;  mind-reader strings: exact
  missions (Python batch vs JS batch 1)  position <= 0.01 m for the first 20 s; same success/stop flags
    v1: all six arbiters closed loop; v2 (brain.version 2, all six modules, RingGated and visibility-gated rivals,
    HeadingRing and KalmanHeading): same rule, run to 90 s so the take-off learning flight, the pad lock and the
    touchdown are inside the compared window; landing flags (landed_on_pad) must agree too; post-study adaptive_search
    (Study 2 modules + adaptive_search, frozen v2 ring): a tent and a backpack mission, same rule, size estimate and
    search altitude equal to 1e-9
  eye RGB crops (v2 front end input) .... max |diff| <= 1e-9
Run: uv run python scripts/parity_brain.py
"""
import json, math, os, subprocess, sys, tempfile, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "brain")]
from provenance import stamp  # noqa: E402
from scenario import sample_scenario, scenario_hash, strip, load, MODULES, OPTIONAL_MODULES  # noqa: E402
from world import build_world  # noqa: E402
import eye  # noqa: E402
import mission as M  # noqa: E402
from pi import PathIntegrator  # noqa: E402
from avoid import reflex  # noqa: E402
from mind import say  # noqa: E402
from arbiters import make, ARBITERS  # noqa: E402

ROOT = os.path.dirname(HERE)


def wrapd(a):
    return np.abs((np.asarray(a) + np.pi) % (2 * np.pi) - np.pi)


def deep_eq(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out.append(f"{path}: keys {sorted(set(a) ^ set(b))}")
        for k in set(a) & set(b):
            deep_eq(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: len {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            deep_eq(x, y, f"{path}[{i}]", out)
    elif a != b and not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and float(a) == float(b)):
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def main():
    t0 = time.time()
    rs = np.random.default_rng(12345)
    spec, py = {}, {}
    # 1. scenarios
    sc_cases = [{"seed": 0, "kw": {"fault": "none"}}, {"seed": 7, "kw": {"fault": "decoy", "severity": 0.4}},
                {"seed": 42, "kw": {}}, {"seed": 99, "kw": {"fault": "eye_columns", "landscape": "forest", "obstacle_density": 0.6}},
                {"seed": 1000, "kw": {"split": "test", "arbiter": "state_machine"}}, {"seed": 1234, "kw": {"split": "test", "fault": "gust", "severity": 0.9}}]
    spec["scenarios"] = sc_cases
    py["scenarios"] = []
    for c in sc_cases:
        s = sample_scenario(c["seed"], **c["kw"])
        py["scenarios"].append({"scenario": s, "hash": scenario_hash(strip(s))})
    # 2. worlds
    wsc = [strip(sample_scenario(s, fault=f, landscape=l, obstacle_density=d))
           for s, f, l, d in [(3, "decoy", "meadow", 0.3), (11, "none", "forest", 0.8), (23, "decoy", "urban", 0.7), (60, "none", "meadow", 0.0)]]
    wsc[0]["decoys"].append({"kind": "pad", "pos": [5.0, 5.0, 0.0], "swap": False, "similarity": 0.6, "t_s": 10.0})
    spec["worlds"] = wsc
    py["worlds"] = [build_world(load(s)) for s in wsc]
    # 3. panoramas (fog, low light, false sun, dead columns)
    fog = strip(sample_scenario(11, fault="none", landscape="forest", obstacle_density=0.5)); fog["world"]["fog_m"] = 15.0; fog["world"]["light"] = 0.6
    dead = [bool(i in range(10, 30)) for i in range(eye.NC)]
    pans = [{"scenario": wsc[0], "pos": [0.0, 0.0, 2.5], "yaw": 0.3, "t": 0.0, "dead": None, "false_sun": None},
            {"scenario": fog, "pos": [3.0, -2.0, 2.5], "yaw": -1.2, "t": 5.0, "dead": dead, "false_sun": None},
            {"scenario": wsc[2], "pos": [1.0, 1.0, 2.5], "yaw": 2.0, "t": 0.0, "dead": None, "false_sun": 1.0}]
    spec["panoramas"] = pans
    py["panoramas"] = []
    for p in pans:
        S = eye.Scene([build_world(load(p["scenario"]))])
        img = eye.panorama(S, np.array([p["pos"]]), np.array([p["yaw"]]), p["t"],
                           dead=None if p["dead"] is None else np.array([p["dead"]]),
                           false_sun=None if p["false_sun"] is None else np.array([p["false_sun"]]))
        py["panoramas"].append(img[0].reshape(-1))
    # 4-5. looks and MB
    spec["mb"] = {"n_in": eye.LH * eye.LW}
    spec["looks"], py["looks"], py["mb"] = [], [], []
    for sc_, lf in [(wsc[0], 10.0), (wsc[2], 0.0)]:
        w = build_world(load(sc_))
        S = eye.Scene([w])
        tgt = [o for o in w["objects"] if o["role"] == "target"][0]
        dec = [o for o in w["objects"] if o["role"] == "decoy"][0]
        poses = [[tgt["x"] + d * math.cos(a), tgt["y"] + d * math.sin(a), 2.5] for d, a in [(3, 0.3), (8, 2.0), (15, -1.0), (2, 4.0)]]
        for slot in (tgt["slot"], dec["slot"]):
            sub = eye.sub_scene(S, np.zeros(len(poses), int), np.full(len(poses), slot))
            g, _ = eye.looks(sub, np.array(poses), 0.0)
            g = g[:, 0]
            L = eye.learning_looks(S, lf)[0]
            noise = rs.standard_normal(g.shape)
            probes = g + eye.NOISE * noise
            wv = ~M.kc_active(M.ranks(L)).any(0)
            act = M.kc_active(M.ranks(probes))
            fam = 1 - (act & wv).sum(-1) / act.sum(-1)
            spec["looks"].append({"scenario": sc_, "slot": slot, "poses": poses, "t": 0.0, "learning_flight_s": lf,
                                  "noise": noise.tolist(), "py_learn": L.tolist(), "py_probes": probes.tolist()})
            py["looks"].append({"looks": g, "learn": L})
            py["mb"].append(fam)
    # 5b. v2 RGB crops (front-end input; object windows and the home pad)
    spec["rgb_looks"], py["rgb_looks"] = [], []
    for sc_ in (strip(sample_scenario(105, fault="decoy", severity=0.5, landscape="forest", version=2)),):
        w = build_world(load(sc_))
        for slot, scene, pos in [(0, eye.Scene([w]), [w["objects"][0]["x"] + 3, w["objects"][0]["y"] + 1, 2.5]),
                                 (1, eye.Scene([w]), [w["objects"][1]["x"] - 6, w["objects"][1]["y"] + 2, 2.5]),
                                 (0, eye.Scene([{**w, "objects": [w["home_pad"]]}]), [2.0, 1.0, 2.5])]:
            g, _ = eye.looks(eye.sub_scene(scene, np.zeros(1, int), np.full(1, slot)), np.array([pos]), 0.0, half_m=1.1, rgb=True)
            spec["rgb_looks"].append({"scenario": sc_, "slot": slot, "pad": scene.K == 1 and slot == 0 and pos == [2.0, 1.0, 2.5], "pos": pos})
            py["rgb_looks"].append(g[0, 0].reshape(-1))
    # 6. PI on a recorded input sequence (wandering flight, compass_half switched on halfway)
    n = 3000
    th = np.cumsum(rs.normal(0, 0.05, n)) + 0.4
    v = np.stack([2.0 + 0.3 * np.sin(np.arange(n) * 0.01), 0.4 * np.sin(np.arange(n) * 0.03)], 1)
    mask = np.ones((n, 16)); mask[n // 2:, [0, 1, 2, 3, 8, 9, 10, 11]] = 0
    spec["pi"] = {"theta": th.tolist(), "v": v.tolist(), "mask": mask.tolist(), "dt": 0.02}
    p = PathIntegrator(1)
    rec = []
    for i in range(n):
        p.col_mask[:] = mask[i]
        h = p.step(th[i:i + 1], v[i:i + 1], 0.02)
        if i % 50 == 49:
            rec.append([h[0], *p.home_vector()[0]])
    py["pi"] = np.array(rec)
    # 7. arbiters on a recorded obs sequence (K = 3; visibility flicker; home switched on at the end)
    T, K = 600, 3
    bear = np.cumsum(rs.normal(0, 0.03, (T, K)), 0) + np.array([0.5, -1.0, 2.5])
    stren = np.clip(np.array([0.7, 0.4, 0.2]) + 0.15 * rs.standard_normal((T, K)), 0, 1)
    vis = rs.random((T, K)) > 0.2
    vis[200:260, 0] = False
    obs_seq = [{"objects": {"bearing": bear[t].tolist(), "strength": stren[t].tolist(), "visible": vis[t].tolist()},
                "home": {"bearing": 2.8, "strength": 1.0 if t > 450 else 0.0},
                "search": {"bearing": float(np.sin(t / 40)), "strength": 0.5 if t <= 450 else 0.0}, "heading": 0.1} for t in range(T)]
    params = {"ring": {"ring_gain": 1.3, "g_home": 3.0}, "ring_static": {}, "vote": {"kappa": 3.0}, "sum": {"p_obj": 2.0},
              "accumulator": {"tau_leak_s": 2.0}, "state_machine": {}}
    spec["arbiters"] = {"params": params, "obs": obs_seq}
    py["arbiters"] = {}
    for name in ARBITERS:
        a = make(name, params[name], 1)
        outs = []
        for o in obs_seq:
            ob = {"objects": {k: np.array([v_]) for k, v_ in o["objects"].items()},
                  "home": {k: np.array([v_]) for k, v_ in o["home"].items()},
                  "search": {k: np.array([v_]) for k, v_ in o["search"].items()}, "heading": np.array([o["heading"]])}
            r = a.step(ob, M.DT_B)
            outs.append([float(r["heading_cmd"][0]), float(r["commit"][0])])
        py["arbiters"][name] = np.array(outs)
    # 8. avoidance
    av = []
    for i in range(20):
        dep = np.minimum(rs.uniform(0.8, 30, eye.NC), eye.DEPTH_MAX)
        av.append({"depth": dep.tolist(), "v": [float(rs.uniform(0, 2.5)), float(rs.uniform(-0.5, 0.5))],
                   "noise": rs.standard_normal(eye.NC).tolist(), "dead": None if i % 2 else [bool(j < 20) for j in range(eye.NC)]})
    spec["avoid"] = av
    py["avoid"] = np.array([np.array(reflex(np.array([c["depth"]]), np.array([c["v"]]), np.array([c["noise"]]),
                                            None if c["dead"] is None else np.array([c["dead"]])))[:, 0] for c in av])
    # 9. mind
    mind_cases = [["search", 0.2, [-25.0, 3.0], 0.62, 2, 0.1, [-27.0, 5.0], ["decoy", "target", "decoy"], [10, 5, 6.2], ["false_sun"]],
                  ["home", -2.0, [0.3, 0.2], 0.125, -1, -2.1, [0.5, 0.1], ["target"], [3.0], []],
                  ["search", 1.0, [10.0, -10.0], 0.0, -1, 1.0, [9.5, -10.2], ["target", "decoy"], [30.0, 12.5], ["occlusion", "speed_scale"]]]
    spec["mind"] = mind_cases
    py["mind"] = [list(say(*c)) for c in mind_cases]
    # 10. missions: one Python batch of three ring scenarios + one scenario per other arm
    ring_batch = [sample_scenario(s, fault=f, arbiter="ring", params={"g_home": 3.0}, obstacle_density=0.3)
                  for s, f in [(5, "none"), (8, "decoy"), (13, "gust")]]
    others = [sample_scenario(21, fault="occlusion", arbiter="state_machine", landscape="forest", obstacle_density=0.3),
              sample_scenario(34, fault="none", arbiter="vote", landscape="urban", obstacle_density=0.4),
              sample_scenario(55, fault="false_sun", arbiter="accumulator"),
              sample_scenario(62, fault="decoy", severity=0.6, arbiter="ring_static", landscape="forest", obstacle_density=0.3),
              sample_scenario(77, fault="speed_scale", severity=0.8, arbiter="sum", landscape="meadow", obstacle_density=0.3)]
    hp = json.load(open(os.path.join(ROOT, "results", "kt5_tuned_params.json")))["estimators"]
    t3 = json.load(open(os.path.join(ROOT, "results", "tuned3d_params.json")))["arms"]
    v2kw = dict(landscape="forest", obstacle_density=0.3, version=2)
    v2_batch = [sample_scenario(s, fault=f, severity=v, arbiter="ring", ring_gain=0.5, params={"g_home": 3.0},
                                module_params={"heading": hp["ring"]["params"]}, **v2kw)
                for s, f, v in [(101, "none", None), (104, "compass_half", 1.0), (108, "speed_scale", 1.0)]]
    v2_others = [sample_scenario(103, fault="false_sun", severity=2 / 3, arbiter="state_machine", params=t3["state_machine"]["params"],
                                 module_params={"heading": hp["ring"]["params"]}, **v2kw),
                 sample_scenario(106, fault="decoy", severity=2 / 3, arbiter="vote", params=t3["vote"]["params"],
                                 module_params={"heading_estimator": "kf", "heading": hp["kf"]["params"]}, **v2kw),
                 sample_scenario(109, fault="none", arbiter="accumulator", params=t3["accumulator"]["params"],
                                 modules=["home", "landing"], **v2kw),
                 sample_scenario(110, fault="occlusion", severity=1.0, arbiter="sum", params=t3["sum"]["params"],
                                 module_params={"heading": hp["ring"]["params"]}, **v2kw),
                 sample_scenario(111, fault="eye_columns", severity=0.5, arbiter="ring_static", params=t3["ring_static"]["params"],
                                 module_params={"heading": hp["ring"]["params"]}, **v2kw)]
    v2_others[3]["brain"]["tip"] = None                                  # no operator tip: spiral from home
    v2_others[4]["brain"]["tip"] = {"bearing_sd_deg": 30.0, "dist_sd": 0.3}
    t2 = json.load(open(os.path.join(ROOT, "results", "tuned_v2_params.json")))["arms"]["ring"]["params"]
    v2_others += [sample_scenario(s, fault=f, severity=v, arbiter="ring", ring_gain=t2["ring_gain"],
                                  params={k: x for k, x in t2.items() if k != "ring_gain"}, modules=MODULES + OPTIONAL_MODULES,
                                  module_params={"heading": hp["ring"]["params"]}, target_kind=k, **v2kw)
                  for s, f, v, k in [(112, "none", None, "tent"), (113, "decoy", 2 / 3, "backpack")]]
    TM, TM2 = 60.0, 90.0
    v1_all = ring_batch + others
    spec["missions"] = [{"scenario": strip(s), "T": TM if i < len(v1_all) else TM2, "eye": i in (0, len(v1_all))}
                        for i, s in enumerate(v1_all + v2_batch + v2_others)]
    py_frames, py_out = [], []
    groups = [(ring_batch, TM)] + [([s], TM) for s in others] + [(v2_batch, TM2)] + [([s], TM2) for s in v2_others]
    for g, T_ in groups:
        m = M.Mission(g, log_rows=range(len(g)))
        o = m.run(max_time=T_)
        for b in range(len(g)):
            py_frames.append(m.frames[b])
            py_out.append({k: (bool(v_[b]) if v_.dtype == bool else float(v_[b])) for k, v_ in o.items()})
    # ---- run JS
    with tempfile.TemporaryDirectory() as td:
        sp, op = os.path.join(td, "spec.json"), os.path.join(td, "out.json")
        json.dump(spec, open(sp, "w"))
        tj = time.time()
        subprocess.run(["node", os.path.join(HERE, "parity_brain_runner.mjs"), sp, op], check=True)
        js_time = time.time() - tj
        js = json.load(open(op))
    # ---- compare
    R = {}
    diffs = []
    for a, b in zip(py["scenarios"], js["scenarios"]):
        diffs += deep_eq(a["scenario"], b["scenario"], "scenario")
        if a["hash"] != b["hash"]:
            diffs.append(f"hash {a['hash']} vs {b['hash']}")
    R["scenarios"] = {"n": len(sc_cases), "diffs": diffs[:5], "pass": not diffs}
    wd = []
    for a, b in zip(py["worlds"], js["worlds"]):
        wd += deep_eq(a, b, "world")
    R["worlds"] = {"n": len(wsc), "diffs": wd[:5], "pass": not wd}
    pd_ = [np.abs(np.asarray(a, int) - np.asarray(b, int)) for a, b in zip(py["panoramas"], js["panoramas"])]
    R["eye_panorama"] = {"n_images": len(pd_), "pixels_per_image": eye.NR * eye.NC * 3,
                         "max_level_diff": int(max(d.max() for d in pd_)), "frac_values_differing": float(np.mean([(d > 0).mean() for d in pd_])),
                         "pass": all(d.max() <= 1 and (d > 0).mean() <= 0.005 for d in pd_)}
    ld = max(max(np.abs(np.asarray(a["looks"]) - np.asarray(b["looks"])).max(), np.abs(np.asarray(a["learn"]) - np.asarray(b["learn"])).max())
             for a, b in zip(py["looks"], js["looks"]))
    R["eye_looks"] = {"max_abs_diff": float(ld), "pass": ld <= 1e-9}
    rd = float(np.abs(np.asarray(py["rgb_looks"]) - np.asarray(js["rgb_looks"])).max())
    R["eye_looks_rgb"] = {"n": len(spec["rgb_looks"]), "max_abs_diff": rd, "pass": rd <= 1e-9}
    R["heading_defaults"] = {"js_HEADING_TUNED_equals_kt5_file": js["heading_tuned"] == hp["ring"]["params"],
                             "pass": js["heading_tuned"] == hp["ring"]["params"]}
    md = max(np.abs(np.asarray(a) - np.asarray(b["recorded"])).max() for a, b in zip(py["mb"], js["mb"]))
    mo = max(np.abs(np.asarray(a) - np.asarray(b["own"])).max() for a, b in zip(py["mb"], js["mb"]))
    idx_ok = js["mb_first_idx"] == M.wiring().reshape(-1)[:25].tolist()
    R["mb_familiarity"] = {"recorded_inputs_max_diff": float(md), "own_looks_max_diff": float(mo), "wiring_first25_equal": idx_ok,
                           "familiarities_py": [list(map(float, f)) for f in py["mb"]], "pass": md == 0 and mo <= 0.02 and idx_ok}
    pj = np.array(js["pi"])
    R["path_integrator"] = {"steps": n, "heading_max_diff": float(wrapd(pj[:, 0] - py["pi"][:, 0]).max()),
                            "home_max_diff_m": float(np.abs(pj[:, 1:] - py["pi"][:, 1:]).max()),
                            "final_home_vector_py": py["pi"][-1, 1:].tolist()}
    R["path_integrator"]["pass"] = R["path_integrator"]["heading_max_diff"] <= 1e-9 and R["path_integrator"]["home_max_diff_m"] <= 1e-6
    R["arbiters"] = {}
    for name in ARBITERS:
        a, b = py["arbiters"][name], np.array(js["arbiters"][name])
        hd, cd = float(wrapd(a[:, 0] - b[:, 0]).max()), float(np.abs(a[:, 1] - b[:, 1]).max())
        R["arbiters"][name] = {"steps": T, "heading_max_diff": hd, "commit_max_diff": cd, "pass": hd <= 1e-9 and cd <= 1e-9}
    ad = float(np.abs(py["avoid"] - np.array(js["avoid"])).max())
    R["avoid"] = {"cases": len(av), "max_diff": ad, "pass": ad <= 1e-12}
    R["mind"] = {"cases": len(mind_cases), "py": py["mind"], "js": js["mind"], "pass": py["mind"] == js["mind"]}
    R["missions"] = []
    for i, (pf, jm, po) in enumerate(zip(py_frames, js["missions"], py_out)):
        jf = jm["frames"]
        nf = min(len(pf), len(jf))
        P = np.array([f["quad"]["pos"] for f in pf[:nf]]); J = np.array([f["pos"] for f in jf[:nf]])
        tt = np.array([f["t"] for f in pf[:nf]])
        err = np.linalg.norm(P - J, axis=1)
        div = tt[np.argmax(err > 0.01)] if (err > 0.01).any() else None
        e20 = float(err[tt <= 20.0].max())
        flags = {k: (po[k], jm["outcome"][k]) for k in ("success_target", "success_home", "wrong_object_stop", "landed_on_pad")}
        same_flags = all(a == b for a, b in flags.values())
        if "size_est_m" in po:                                           # adaptive_search: size estimate and altitude
            same_flags &= all(abs(po[k] - jm["outcome"][k]) <= 1e-9 for k in ("size_est_m", "search_alt_m"))
        eye_d = None
        if spec["missions"][i]["eye"]:
            eye_d = float(np.mean([np.mean(np.asarray(a["eye"]["rgb"]) != np.asarray(b["eye"])) for a, b in zip(pf[:nf], jf[:nf])]))
        b_ = spec["missions"][i]["scenario"]["brain"]
        rec = {"seed": spec["missions"][i]["scenario"]["seed"], "arbiter": b_["arbiter"], "version": b_.get("version", 1),
               "tip": b_.get("tip", "v1"), "target_kind": spec["missions"][i]["scenario"]["target"]["kind"],
               "modules": b_.get("modules"), "heading_estimator": b_.get("module_params", {}).get("heading_estimator"),
               "python_batch": "ring x3" if i < 3 or len(v1_all) <= i < len(v1_all) + 3 else "single",
               "landing_error_m_py_js": [po["landing_error_m"], jm["outcome"]["landing_error_m"]],
               **({"size_est_m_py_js": [po["size_est_m"], jm["outcome"]["size_est_m"]],
                   "search_alt_m_py_js": [po["search_alt_m"], jm["outcome"]["search_alt_m"]]} if "size_est_m" in po else {}), "frames": nf, "frames_py": len(pf), "frames_js": len(jf),
               "max_pos_err_first_20s_m": e20, "max_pos_err_all_m": float(err.max()), "first_divergence_1cm_s": div,
               "phase_agreement": float(np.mean([a["phase"] == b["phase"] for a, b in zip(pf[:nf], jf[:nf])])),
               "outcome_flags_py_js": flags, "time_s_py_js": [po["time_s"], jm["outcome"]["time_s"]],
               "eye_values_differing_frac": eye_d, "pass": e20 <= 0.01 and same_flags}
        R["missions"].append(rec)
    ok = all(v["pass"] for k, v in R.items() if isinstance(v, dict) and "pass" in v) and all(v["pass"] for v in R["arbiters"].values()) \
        and all(m["pass"] for m in R["missions"])
    out = {"pass": bool(ok), "tolerances": __doc__.split("\n\n")[1], "results": R, "js_seconds": js_time,
           "wall_seconds": time.time() - t0, **stamp(seed="fixed set (see results)", param_source="scripts/parity_brain.py")}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "parity_brain.json"), "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    out = main()
    R = out["results"]
    for k, v in R.items():
        if k == "arbiters":
            for a, r in v.items():
                print(f"  arbiter {a:14s} pass={r['pass']} heading {r['heading_max_diff']:.2e} commit {r['commit_max_diff']:.2e}")
        elif k == "missions":
            for m in v:
                print(f"  mission seed {m['seed']} {m['arbiter']:14s} pass={m['pass']} err20 {m['max_pos_err_first_20s_m']:.2e} m, "
                      f"all {m['max_pos_err_all_m']:.2e} m, diverge {m['first_divergence_1cm_s']}, flags {m['outcome_flags_py_js']}")
        else:
            print(f"  {k:16s} pass={v['pass']} " + ", ".join(f"{a}={b}" for a, b in v.items() if a not in ("pass", "py", "js", "familiarities_py", "diffs")))
    print("PARITY", "PASS" if out["pass"] else "FAIL", f"({out['wall_seconds']:.0f} s)")
    assert out["pass"], "brain/mission parity failed: see results/parity_brain.json"
