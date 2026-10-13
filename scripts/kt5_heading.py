"""KT5 (H5): does a ring-attractor heading integrator survive sun-compass faults better than a complementary filter,
a Kalman filter and the Study 1 direct sun compass? Pre-registered (H5; the paper's Study 2 protocol and amendments
section).

Traces. scripts/mission.py (Study 1 brain: ring arbiter with results/tuned3d_params.json, forest density 0.3, no wind, no
fault) is flown on seeds 100-199 (tuning) and 1300-1599 (test); the true yaw and true horizontal body velocity are
recorded at 50 Hz until the mission ends (data/raw/kt5_traces.npz, regenerable with --record).
Sensors (per seed, scripts/rng.py streams): gyro (20) = true yaw rate + bias N(0, 0.5 deg/s) + N(0, 0.1 deg/s) per
sample; sun compass = the Study 1 TL->CL1->TB1 compass of brain/pi.py fed with true yaw + N(0, 2 deg) (2) + false-sun
offset, with the v1 CL1 mask under compass_half; fault onset 5 + 20 u (21); gust yaw disturbance (22); odometry
noise for the path integrator (3). The vehicle model has no aerodynamic yaw moment, so the gust condition adds a yaw
disturbance to the recorded trace (yaw-rate Gauss-Markov 30 deg/s, 0.2 s, pulled back by the yaw loop with 1 s;
an assumption). sun_valid is true throughout.
Metric: per seed RMS heading error (deg) over the recorded mission. Home vector: each heading source is fed as the
compass input of the Study 1 PathIntegrator, the trace is replayed in a loop to 300 m of path, error at 300 m.
Tuning: 64 configs per estimator (default_rng(2026)), seeds 100-199, J = mean RMS over the 6 fault conditions + RMS
with no fault. Frozen to results/kt5_tuned_params.json before the test seeds are touched.
Usage: uv run python scripts/kt5_heading.py --record | --tune | --test | (no flag: self-check)
"""
import hashlib, json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "brain")]
from rng import Rng  # noqa: E402
from provenance import stamp  # noqa: E402
from stats import arm_permutation_test, paired_bootstrap, paired_diff  # noqa: E402
from heading import CLASSES, sample_params, js_parity, wrap  # noqa: E402
from pi import PathIntegrator, SIGMA_COMPASS, SIGMA_SPEED  # noqa: E402

ROOT = HERE.parent
TRACES = ROOT / "data/raw/kt5_traces.npz"
TUNED = ROOT / "results/kt5_tuned_params.json"
OUT = ROOT / "results/kt5.json"
FIG = ROOT / "results/fig_kt5.png"
DT = 0.02
TRAIN_SEEDS, TEST_SEEDS = np.arange(100, 200), np.arange(1300, 1600)
CONDS = ["none", "false_sun_30", "false_sun_60", "false_sun_90", "false_sun_120", "compass_half", "gust"]
FAULT_SET = ["false_sun_30", "false_sun_60", "false_sun_90", "false_sun_120", "compass_half"]
EST = ["ring", "cf", "kf"]
SOURCES = EST + ["v1"]                 # v1 = the direct sun compass (TB1 decode), no tuning
N_CONFIGS, SEARCH_SEED = 64, 2026
GYRO_BIAS_SD, GYRO_NOISE_SD = math.radians(0.5), math.radians(0.1)
GUST_RATE_SD, GUST_TAU, YAW_LOOP_TAU = math.radians(30.0), 0.2, 1.0
HOME_PATH_M = 300.0
KILL_DEG = 2.0
PROCS = min(2, int(os.environ.get("KT_PROCS", 2)))


# ------------------------------------------------------------------ traces from the 3D mission
def _scenario(seed, params, ring_gain):
    """scenario.sample_scenario(..., 'none') for a seed outside the v1 splits (that function asserts them)."""
    from scenario import sample_scenario
    s = sample_scenario(0, "train", "none", None, "ring", ring_gain, params=params, landscape="forest", obstacle_density=0.3)
    r = Rng([seed], 5)
    u = [float(r.uniform()[0]) for _ in range(3)]          # same draws as sample_scenario: distance, bearing, yaw
    d, b = 25.0 + 15.0 * u[0], 2 * math.pi * u[1]
    s["seed"] = int(seed)
    s["target"]["pos"] = [d * math.cos(b), d * math.sin(b), 0.0]
    s["target"]["yaw_deg"] = 360 * u[2]
    return s


def _record_batch(seeds):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from mission import Mission
    tp = json.loads((ROOT / "results/tuned3d_params.json").read_text())["arms"]["ring"]["params"]
    p = dict(tp)
    g = p.pop("ring_gain")
    m = Mission([_scenario(s, p, g) for s in seeds])
    B = len(seeds)
    yaw, vb, n = [], [], np.zeros(B, int)
    while m.t < 300.0 - 1e-9 and (m.phase < 4).any():
        y = m.quad.euler()[:, 2]
        c, s_ = np.cos(y), np.sin(y)
        v = m.quad.vel
        yaw.append(y.copy())
        vb.append(np.stack([c * v[:, 0] + s_ * v[:, 1], -s_ * v[:, 0] + c * v[:, 1]], 1))
        n += m.phase < 4
        m.step()
    yaw.append(m.quad.euler()[:, 2].copy())                 # state after the last tick
    yaw, vb = np.array(yaw), np.array(vb)                   # [T + 1, B], [T, B, 2]
    assert len(yaw) >= n.max() + 1
    return [(int(seeds[b]), yaw[:n[b] + 1, b], vb[:n[b], b].astype(np.float32)) for b in range(B)]


def record():
    seeds = np.concatenate([TRAIN_SEEDS, TEST_SEEDS])
    chunks = [seeds[i:i + 50] for i in range(0, len(seeds), 50)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=PROCS) as ex:
        out = [r for rs in ex.map(_record_batch, chunks) for r in rs]
    out.sort()
    lens = np.array([len(o[2]) for o in out])
    TRACES.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TRACES, seeds=np.array([o[0] for o in out]), lens=lens,
                        yaw=np.concatenate([o[1] for o in out]), vb=np.concatenate([o[2] for o in out]))
    print(f"recorded {len(out)} traces, ticks mean {lens.mean():.0f} max {lens.max()}, {time.time() - t0:.0f} s")


def load_traces(seeds):
    z = np.load(TRACES)
    idx = {int(s): i for i, s in enumerate(z["seeds"])}
    lens = z["lens"]
    off_y = np.concatenate([[0], np.cumsum(lens + 1)])
    off_v = np.concatenate([[0], np.cumsum(lens)])
    out = []
    for s in seeds:
        i = idx[int(s)]
        y = z["yaw"][off_y[i]:off_y[i + 1]]
        assert len(y) == lens[i] + 1
        out.append((wrap(np.diff(y)) / DT, z["vb"][off_v[i]:off_v[i + 1]].astype(float), float(y[0])))
    return out


def traces_sha():
    return hashlib.sha256(TRACES.read_bytes()).hexdigest()


# ------------------------------------------------------------------ sensor inputs
def build(seeds, cond, traces, loop=False):
    """-> dict of [T, R] arrays: truth, gyro, sun (v1 compass), live (first pass), path (cumulative m), vb [T, R, 2].
    Tick k: gyro = mean true yaw rate over the last tick (0 at k = 0), sun from the true yaw at k."""
    R = len(seeds)
    n1 = np.array([len(t[0]) for t in traces]) + 1                       # ticks in one pass (yaw samples)
    if loop:
        per = np.array([np.hypot(*t[1].T).sum() * DT for t in traces])
        reps = np.ceil(HOME_PATH_M / np.maximum(per, 1e-3)).astype(int) + 1
        reps = np.minimum(reps, 60)
    else:
        reps = np.ones(R, int)
    T = int((n1 * reps).max())
    rate = np.zeros((T, R)); vb = np.zeros((T, R, 2)); live = np.zeros((T, R), bool)
    for j, (r, v, _) in enumerate(traces):
        rr = np.tile(np.r_[0.0, r], reps[j])                             # tick 0 of each pass: no rotation
        vv = np.tile(np.r_[v, v[-1:]], (reps[j], 1))
        rate[:len(rr), j], vb[:len(vv), j] = rr, vv
        live[:n1[j], j] = True
    yaw0 = np.array([t[2] for t in traces])
    t = np.arange(T) * DT
    r_g, r_c, r_f, r_w = Rng(seeds, 20), Rng(seeds, 2), Rng(seeds, 21), Rng(seeds, 22)
    bias = GYRO_BIAS_SD * r_g.normal()
    t_on = 5.0 + 20.0 * r_f.uniform()
    if cond == "gust":
        a = math.exp(-DT / GUST_TAU)
        rg, dpsi = np.zeros(R), np.zeros(R)
        for k in range(1, T):
            rg = a * rg + GUST_RATE_SD * math.sqrt(1 - a * a) * r_w.normal()
            new = dpsi + DT * (rg - dpsi / YAW_LOOP_TAU)
            rate[k] += (new - dpsi) / DT
            dpsi = new
    truth = yaw0[None, :] + np.cumsum(rate, 0) * DT
    gyro = np.zeros((T, R))
    for k in range(T):
        gyro[k] = rate[k] + bias + GYRO_NOISE_SD * r_g.normal()
    gyro[0] = 0.0
    off = math.radians(float(cond.split("_")[-1])) if cond.startswith("false_sun") else 0.0
    comp = PathIntegrator(R)
    sun = np.zeros((T, R))
    zero = np.zeros((R, 2))
    for k in range(T):
        on = t[k] >= t_on
        comp.col_mask[:] = 1.0
        if cond == "compass_half":
            comp.col_mask[on] = np.tile(np.r_[np.zeros(4), np.ones(4)], 2)
        theta = truth[k] + SIGMA_COMPASS * r_c.normal() + np.where(on, off, 0.0)
        sun[k] = comp.step(theta, zero, DT)
    path = np.cumsum(np.hypot(vb[..., 0], vb[..., 1]), 0) * DT
    return {"truth": wrap(truth), "gyro": gyro, "sun": sun, "live": live, "vb": vb, "path": path, "t_on": t_on,
            "bias": bias, "seeds": np.asarray(seeds)}


def run_est(name, params, inp, n_cfg=1, keep=False):
    """Run an estimator on inputs [T, R] for n_cfg configs at once (params arrays of length n_cfg, broadcast over R).
    -> RMS [n_cfg, R] (deg, first pass), and the headings [T, n_cfg*R] if keep."""
    T, R = inp["gyro"].shape
    P = {k: np.repeat(np.asarray(v, float).reshape(-1), R) for k, v in params.items()}
    est = CLASSES[name](n_cfg * R, P)
    valid = np.ones(n_cfg * R, bool)
    ss = np.zeros(n_cfg * R)
    live = np.tile(inp["live"], (1, n_cfg))
    H = np.zeros((T, n_cfg * R)) if keep else None
    for k in range(T):
        h = est.step(np.tile(inp["gyro"][k], n_cfg), np.tile(inp["sun"][k], n_cfg), valid, DT)
        e = wrap(h - np.tile(inp["truth"][k], n_cfg))
        ss += live[k] * e * e
        if keep:
            H[k] = h
    rms = np.degrees(np.sqrt(ss / live.sum(0))).reshape(n_cfg, R)
    return rms, H


def rms_of(h, inp):
    e = wrap(h - inp["truth"])
    return np.degrees(np.sqrt((inp["live"] * e * e).sum(0) / inp["live"].sum(0)))


# ------------------------------------------------------------------ tuning (train seeds only)
def configs(name):
    n = len(CLASSES[name].PARAM_SPACE)
    U = np.random.default_rng(SEARCH_SEED).random((N_CONFIGS, 16))[:, :n]
    return sample_params(name, U)                                      # {param: [64]}


def _tune_job(cond):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    inp = build(TRAIN_SEEDS, cond, load_traces(TRAIN_SEEDS))
    return cond, {n: run_est(n, configs(n), inp, n_cfg=N_CONFIGS)[0].mean(1) for n in EST}


def tune():
    assert not TUNED.exists(), f"{TUNED} exists: tuning is frozen"
    t0 = time.time()
    M = {n: np.zeros((len(CONDS), N_CONFIGS)) for n in EST}
    with ProcessPoolExecutor(max_workers=PROCS) as ex:
        for cond, r in ex.map(_tune_job, CONDS):
            for n in EST:
                M[n][CONDS.index(cond)] = r[n]
    out = {"stage": "frozen before any test seed (1300-1599)", "train_seeds": "100-199",
           "budget": f"{N_CONFIGS} configs per estimator, numpy default_rng({SEARCH_SEED}).random((64, 16))[:, :n_params]",
           "objective": "J = mean over the 6 fault conditions of mean-over-seeds RMS heading error (deg) + RMS with no fault",
           "conditions": CONDS, "estimators": {}}
    for n in EST:
        J = M[n][1:].mean(0) + M[n][0]
        i = int(np.argmin(J))
        cf = configs(n)
        out["estimators"][n] = {"params": {k: float(v[i]) for k, v in cf.items()}, "config_index": i, "J": float(J[i]),
                                "train_rms_deg": dict(zip(CONDS, M[n][:, i].tolist())), "n_params": len(cf),
                                "J_all": J.tolist(), "J_median": float(np.median(J))}
    out["provenance"] = stamp(seed="train 100-199", param_source="random search", traces_sha256=traces_sha(),
                              wall_s=round(time.time() - t0, 1))
    TUNED.write_text(json.dumps(out, indent=1))
    for n in EST:
        e = out["estimators"][n]
        print(n, e["config_index"], round(e["J"], 3), {k: round(v, 2) for k, v in e["train_rms_deg"].items()})


# ------------------------------------------------------------------ test (seeds 1300-1599)
def _test_job(cond):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    tuned = json.loads(TUNED.read_text())["estimators"]
    inp = build(TEST_SEEDS, cond, load_traces(TEST_SEEDS), loop=True)
    R = len(TEST_SEEDS)
    heads, rms = {}, {}
    for n in EST:
        r, H = run_est(n, tuned[n]["params"], inp, keep=True)
        rms[n], heads[n] = r[0], H
    heads["v1"], rms["v1"] = inp["sun"], rms_of(inp["sun"], inp)
    heads["truth"] = inp["truth"]
    # path integrator on each heading source; error at the first tick with 300 m of path
    names = SOURCES + ["truth"]
    pi_ = PathIntegrator(len(names) * R)
    r_s = Rng(np.tile(TEST_SEEDS, len(names)), 3)
    T = len(inp["gyro"])
    k300 = np.argmax(inp["path"] >= HOME_PATH_M, 0)
    ok = inp["path"][-1] >= HOME_PATH_M
    pos = np.zeros((R, 2))
    err = np.full((len(names), R), np.nan)
    for k in range(T):
        th = np.concatenate([heads[n][k] for n in names])
        vb = np.tile(inp["vb"][k], (len(names), 1))
        n1, n2 = r_s.normal(), r_s.normal()
        pi_.step(th, vb + SIGMA_SPEED * np.stack([n1, n2], 1), DT)
        c, s = np.cos(inp["truth"][k]), np.sin(inp["truth"][k])
        v = inp["vb"][k]
        pos += DT * np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], 1)
        hit = (k == k300) & ok
        if hit.any():
            hv = pi_.home_vector().reshape(len(names), R, 2)
            e = np.linalg.norm(hv - (-pos)[None], axis=-1)
            err[:, hit] = e[:, hit]
    # small JS parity sample on this condition's inputs (seed 1300, first 3000 ticks)
    par = {n: js_parity(n, tuned[n]["params"], inp["gyro"][:3000, 0], inp["sun"][:3000, 0], np.ones(3000, bool), DT)
           for n in EST}
    return cond, {n: rms[n].tolist() for n in SOURCES}, {n: err[i].tolist() for i, n in enumerate(names)}, par, \
        {"t_on": inp["t_on"].tolist(), "bias_deg_s": np.degrees(inp["bias"]).tolist(),
         "first_pass_s": (inp["live"].sum(0) * DT).tolist()}


def ci_mean(x):
    return paired_bootstrap(lambda a: np.mean(a), np.asarray(x))


def test():
    assert TUNED.exists(), "tune first"
    tuned_txt = TUNED.read_text()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=PROCS) as ex:
        res = list(ex.map(_test_job, CONDS))
    RMS = {c: {n: np.array(v) for n, v in r.items()} for c, r, _, _, _ in res}
    HOME = {c: {n: np.array(v) for n, v in h.items()} for c, _, h, _, _ in res}
    PAR = {c: p for c, _, _, p, _ in res}
    meta = res[0][4]
    per = {}
    for c in CONDS:
        d = {"rms_deg": {n: ci_mean(RMS[c][n]) for n in SOURCES},
             "delta_vs_ring_deg": {n: paired_diff(RMS[c][n], RMS[c]["ring"]) for n in SOURCES if n != "ring"},
             "perm_ring_vs_best_other": arm_permutation_test(-RMS[c]["ring"], [-RMS[c][n] for n in SOURCES if n != "ring"]),
             "perm_ring_vs_best_filter": arm_permutation_test(-RMS[c]["ring"], [-RMS[c]["cf"], -RMS[c]["kf"]])}
        ok = np.isfinite(HOME[c]["truth"])
        d["home_error_300m_m"] = {n: ci_mean(HOME[c][n][ok]) for n in SOURCES + ["truth"]}
        d["home_error_300m_m"]["n_seeds"] = int(ok.sum())
        per[c] = d
    agg = {n: np.mean([RMS[c][n] for c in FAULT_SET], 0) for n in SOURCES}
    aggregate = {"fault_set": FAULT_SET, "rms_deg": {n: ci_mean(agg[n]) for n in SOURCES},
                 "delta_vs_ring_deg": {n: paired_diff(agg[n], agg["ring"]) for n in SOURCES if n != "ring"},
                 "perm_ring_vs_best_other": arm_permutation_test(-agg["ring"], [-agg[n] for n in SOURCES if n != "ring"])}
    # gate (pre-registered)
    kill = [f for f in ("cf", "kf") if all(abs(per[c]["delta_vs_ring_deg"][f]["est"]) < KILL_DEG for c in CONDS)]
    lo_ok = {n: aggregate["delta_vs_ring_deg"][n]["lo"] > 0 for n in ("v1", "cf", "kf")}
    p = aggregate["perm_ring_vs_best_other"]
    perm_ok = p["est"] > 0 and p["p_two_sided"] < 0.05
    if kill:
        gate = f"KILL: ring = filter ({', '.join(kill)} within {KILL_DEG} deg RMS of the ring in every condition)"
    elif all(lo_ok.values()) and perm_ok:
        gate = "H5 SUPPORTED: ring heading error lower than every rival on the fault set"
    else:
        better = [n for n in ("v1", "cf", "kf") if aggregate["delta_vs_ring_deg"][n]["hi"] < 0]
        gate = "H5 NOT SUPPORTED" + (f": {', '.join(better)} lower than the ring on the fault set (CI excludes 0)" if better else "")
    parity = max(max(v) for p_ in PAR.values() for v in p_.values())
    out = {"hypothesis": "H5", "test_seeds": "1300-1599", "conditions": CONDS, "fault_set": FAULT_SET,
           "gate": gate, "gate_detail": {"kill_filters": kill, "ci_lo_above_0": lo_ok, "perm_ok": perm_ok,
                                         "kill_deg": KILL_DEG},
           "per_condition": per, "aggregate_fault_set": aggregate,
           "tuned_params": {n: json.loads(tuned_txt)["estimators"][n]["params"] for n in EST},
           "per_seed_rms_deg": {c: {n: RMS[c][n].tolist() for n in SOURCES} for c in CONDS},
           "per_seed_home_error_300m_m": {c: {n: HOME[c][n].tolist() for n in HOME[c]} for c in CONDS},
           "trace_meta": {"mean_first_pass_s": float(np.mean(meta["first_pass_s"])),
                          "min_first_pass_s": float(np.min(meta["first_pass_s"])),
                          "gyro_bias_deg_s_sd": float(np.std(meta["bias_deg_s"]))},
           "js_parity_max_abs": parity, "js_parity_per_condition": PAR,
           "sensors": {"gyro_bias_sd_deg_s": 0.5, "gyro_noise_sd_deg_s": 0.1, "compass_noise_sd_deg": 2.0,
                       "gust": "yaw-rate Gauss-Markov 30 deg/s, tau 0.2 s, yaw loop 1 s (assumption)",
                       "fault_onset_s": "5 + 20 u, Rng stream 21"},
           "provenance": stamp(seed="test 1300-1599", param_source="results/kt5_tuned_params.json",
                               tuned_sha256=hashlib.sha256(tuned_txt.encode()).hexdigest(), traces_sha256=traces_sha(),
                               wall_s=round(time.time() - t0, 1))}
    OUT.write_text(json.dumps(out, indent=1))
    figure(out)
    print(gate)
    for c in CONDS:
        print(f"{c:15s}", "  ".join(f"{n} {per[c]['rms_deg'][n]['est']:6.2f}" for n in SOURCES),
              " | home@300m", "  ".join(f"{n} {per[c]['home_error_300m_m'][n]['est']:6.1f}" for n in SOURCES + ["truth"]))
    print("JS parity max", parity)


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"ring": "#1b6ca8", "cf": "#d08c2a", "kf": "#3a9a5b", "v1": "#8a8a8a", "truth": "#222222"}
    lab = {"ring": "ring attractor", "cf": "complementary filter", "kf": "Kalman filter", "v1": "v1 sun compass",
           "truth": "true heading"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(CONDS))
    for ax, key, srcs, ylab in ((axs[0], "rms_deg", SOURCES, "RMS heading error (deg)"),
                                (axs[1], "home_error_300m_m", SOURCES + ["truth"], "home-vector error at 300 m (m)")):
        w = 0.8 / len(srcs)
        for i, n in enumerate(srcs):
            v = [out["per_condition"][c][key][n] for c in CONDS]
            est = np.array([d["est"] for d in v])
            err = np.array([[d["est"] - d["lo"] for d in v], [d["hi"] - d["est"] for d in v]])
            ax.bar(x + (i - (len(srcs) - 1) / 2) * w, est, w, yerr=err, color=col[n], label=lab[n], capsize=1.5)
        ax.set_xticks(x, [c.replace("false_sun_", "false sun ").replace("_", " ") for c in CONDS], rotation=30, ha="right")
        ax.set_ylabel(ylab)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
    axs[1].legend(fontsize=8, frameon=False)
    axs[0].set_title("KT5 heading error, test seeds 1300-1599 (95% CI)")
    axs[1].set_title("v1 path integrator fed with each heading")
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)


# ------------------------------------------------------------------ self-check
def selfcheck():
    # 1) inputs: with no bias/noise the truth is the integral of the gyro; v1 compass tracks the truth without faults
    tr = [(np.radians(40) * np.sin(np.arange(800) * 0.02), np.tile([2.0, 0.0], (800, 1)), 0.3) for _ in range(3)]
    inp = build(np.array([100, 101, 102]), "none", tr)
    exp = wrap(0.3 + np.cumsum(np.r_[0.0, tr[0][0]]) * DT)
    assert np.abs(wrap(inp["truth"][:, 0] - exp)).max() < 1e-9
    assert np.all(np.abs((inp["gyro"][1:] - np.array(tr[0][0])[:, None]).mean(0) - inp["bias"]) < np.radians(0.02))
    assert rms_of(inp["sun"], inp).max() < 3.0
    fs = build(np.array([100, 101, 102]), "false_sun_90", tr)
    late = fs["live"] & (np.arange(len(fs["sun"]))[:, None] * DT > 26)
    assert np.all(np.abs(np.degrees(wrap(fs["sun"] - fs["truth"]))[late] - 90) < 10)
    ch = build(np.array([100, 101, 102]), "compass_half", tr)
    assert np.degrees(np.abs(wrap(ch["sun"] - ch["truth"]))).max() > 20
    # 2) loop to 300 m: path reaches 300 m and the true-heading PI home error stays small
    lp = build(np.array([100]), "none", tr[:1], loop=True)
    assert lp["path"][-1, 0] >= HOME_PATH_M and lp["live"].sum() == 801
    # 3) batched tuning equals per-config runs
    P = {k: v[:3] for k, v in configs("ring").items()}
    r3 = run_est("ring", P, inp, n_cfg=3)[0]
    r1 = run_est("ring", {k: v[1:2] for k, v in P.items()}, inp)[0]
    assert np.allclose(r3[1], r1[0], atol=1e-12)
    # 4) frozen-before-test: if both files exist, the test used the tuned file as it is now and ran after it
    if OUT.exists() and TUNED.exists():
        o, t = json.loads(OUT.read_text()), json.loads(TUNED.read_text())
        assert o["provenance"]["tuned_sha256"] == hashlib.sha256(TUNED.read_bytes()).hexdigest()
        assert t["provenance"]["created"] <= o["provenance"]["created"]
        assert o["js_parity_max_abs"] <= 1e-9
        for c in CONDS:                                                    # stored summaries reproduce from per-seed rows
            for n in SOURCES:
                assert abs(np.mean(o["per_seed_rms_deg"][c][n]) - o["per_condition"][c]["rms_deg"][n]["est"]) < 1e-9
        print("kt5 result file consistent; gate:", o["gate"])
    print("kt5 self-check OK")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "--selfcheck"
    {"--record": record, "--tune": tune, "--test": test, "--selfcheck": selfcheck}[arg]()
