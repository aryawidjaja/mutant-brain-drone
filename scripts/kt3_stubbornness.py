"""KT3 (H3): the stubbornness frontier. Hold a hidden target vs release a decoy, as each arm's memory knob is swept.

Pre-registered in docs/LOG.md ([WS-A] KT3). Uses KT2-tuned params for everything but the swept knob.
Usage: uv run python scripts/kt3_stubbornness.py [--selfcheck]
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from brain.arbiters import make, wrap
from kt2_five_arms import DT, V, OMEGA, SIG_VIS, SIG_COMPASS, GUST_DEG, TEST_SEEDS
from rng import Rng
from provenance import stamp
from stats import boot_idx

ROOT = Path(__file__).resolve().parents[1]
TOL = np.deg2rad(20)
T_OCC_ON, HOLD_CAP = 4.0, 10.0
T_RAMP0, T_RAMP1, REL_CAP = 5.0, 9.0, 20.0
F_CROSS = 1.05 / 1.65                       # 0.3 + 0.6 f = 1.5 (0.9 - 0.7 f)
T_CROSS = T_RAMP0 + (T_RAMP1 - T_RAMP0) * F_CROSS
SWEEPS = {"ring": ("ring_gain", [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]),
          "state_machine": ("timeout_s", list(np.geomspace(0.2, 10, 15))),
          "accumulator": ("tau_leak_s", list(np.geomspace(0.05, 20, 15))),
          "vote": (None, [None]), "sum": (None, [None]), "ring_static": (None, [None])}


def trial(arm, params, seeds, kind):
    """kind 'occ' -> t_hold [B] and on-target at 4 s; kind 'dec' -> t_release [B] and on-decoy at 5 s."""
    B = len(seeds)
    r = Rng(seeds, 20 if kind == "occ" else 21)
    rc, rv, rg = Rng(seeds, 22), Rng(seeds, 23), Rng(seeds, 24)
    phi = 2 * np.pi * r.uniform()
    sgn = np.sign(r.uniform() - 0.5)
    d1, d2 = np.deg2rad(60 + 90 * r.uniform()), np.deg2rad(60 + 90 * r.uniform())
    h = phi + np.deg2rad(60) * (r.uniform() - 0.5)
    dist = 45.0 if kind == "occ" else 80.0
    ang = np.stack([phi, phi + sgn * d1], 1)                     # slot 0 target, slot 1 decoy
    obj = dist * np.stack([np.cos(ang), np.sin(ang)], -1)
    search_b = phi + sgn * d1 if kind == "occ" else phi - sgn * d2
    K = 1 if kind == "occ" else 2
    arb = make(arm, params, B)
    pos = np.zeros((B, 2))
    out = np.full(B, np.nan)
    flag = np.zeros(B, bool)
    T = T_OCC_ON + HOLD_CAP if kind == "occ" else T_CROSS + REL_CAP
    for k in range(int(round(T / DT)) + 1):
        t = k * DT
        hhat = h + SIG_COMPASS * rc.normal()
        rel = obj[:, :K] - pos[:, None]
        wb = np.arctan2(rel[..., 1], rel[..., 0])
        bear = wrap(wb - h[:, None] + hhat[:, None])
        if kind == "occ":
            vis = np.full((B, 1), t < T_OCC_ON)
            ev = np.full((B, 1), 0.8)
        else:
            f = np.clip((t - T_RAMP0) / (T_RAMP1 - T_RAMP0), 0, 1)
            vis = np.ones((B, 2), bool)
            ev = np.tile([0.3 + 0.6 * f, 0.9 - 0.7 * f], (B, 1))
        noise = np.stack([rv.normal() for _ in range(K)], 1)
        strength = np.clip(ev + SIG_VIS * noise, 0, 1) * vis
        obs = {"objects": {"bearing": bear, "strength": strength, "visible": vis},
               "home": {"bearing": np.zeros(B), "strength": np.zeros(B)},
               "search": {"bearing": wrap(search_b), "strength": np.ones(B)}, "heading": wrap(hhat)}
        cmd = arb.step(obs, DT)["heading_cmd"]
        err_t = np.abs(wrap(cmd - bear[:, 0]))
        if kind == "occ":
            if abs(t - (T_OCC_ON - DT)) < DT / 2:  # last visible tick
                flag = err_t <= TOL
            if t >= T_OCC_ON:
                out = np.where(np.isnan(out) & (err_t > TOL), t - T_OCC_ON, out)
        else:
            if abs(t - T_RAMP0) < DT / 2:
                flag = np.abs(wrap(cmd - bear[:, 1])) <= TOL
            if t >= T_CROSS - DT / 2:
                out = np.where(np.isnan(out) & (err_t < TOL), t - T_CROSS, out)
        turn = np.clip(wrap(cmd - hhat), -OMEGA * DT, OMEGA * DT)
        h = h + turn + np.deg2rad(GUST_DEG["none"]) * np.sqrt(DT) * rg.normal()
        pos = pos + V * DT * np.stack([np.cos(h), np.sin(h)], 1)
    cap = HOLD_CAP if kind == "occ" else REL_CAP
    return np.clip(np.where(np.isnan(out), cap, out), 0.0, cap), flag


def _point(a):
    arm, params, seeds = a
    hold, on_t = trial(arm, params, seeds, "occ")
    rel, on_d = trial(arm, params, seeds, "dec")
    return hold, rel, on_t.mean(), on_d.mean()


def pareto_interp(H, R):
    """Rival frontier through Pareto-efficient (max hold, min release) points; returns f(h) (NaN outside)."""
    H, R = np.asarray(H, float), np.asarray(R, float)
    eff = np.array([not np.any((H >= H[i]) & (R <= R[i]) & ((H > H[i]) | (R < R[i]))) for i in range(len(H))])
    h, r = H[eff], R[eff]
    o = np.argsort(h)
    h, r = h[o], r[o]
    def f(x):
        x = np.asarray(x, float)
        if len(h) == 1:
            return np.where(np.isclose(x, h[0]), r[0], np.nan)
        return np.where((x >= h[0]) & (x <= h[-1]), np.interp(x, h, r), np.nan)
    return f


def dominance_frontier(H, R):
    """Post-hoc sensitivity: best rival release among its points that hold at least h (NaN above max hold)."""
    H, R = np.asarray(H, float), np.asarray(R, float)
    return lambda x: np.array([R[H >= v].min() if np.any(H >= v) else np.nan for v in np.atleast_1d(x)])


def deltas(hold, rel, frontier=None):
    """hold/rel: dict arm -> [n_points] means. Returns Delta_rival over ring gains for SM and accumulator."""
    frontier = frontier or pareto_interp
    return {rv: frontier(hold[rv], rel[rv])(hold["ring"]) - rel["ring"] for rv in ("state_machine", "accumulator")}


def verdict(d_est, d_lo):
    both = (d_lo["state_machine"] > 0) & (d_lo["accumulator"] > 0)
    alive = bool(np.any(both[1:] & both[:-1]))
    dead = []
    for rv in ("state_machine", "accumulator"):
        e = d_est[rv][~np.isnan(d_est[rv])]
        if len(e) and np.all(e <= 0):
            dead.append(rv)
    return "ALIVE" if alive else ("DEAD" if dead else "INCONCLUSIVE"), dead


def main():
    tuned = json.loads((ROOT / "results/kt2_tuned_params.json").read_text())["arms"]
    jobs, keys = [], []
    for arm, (knob, vals) in SWEEPS.items():
        for v in vals:
            p = dict(tuned[arm]["params"]) | ({knob: float(v)} if knob else {})
            jobs.append((arm, p, TEST_SEEDS))
            keys.append((arm, v))
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        pts = list(ex.map(_point, jobs))
    per = {arm: [(v, *pts[i]) for i, (a, v) in enumerate(keys) if a == arm] for arm in SWEEPS}
    Hs = {a: np.array([x[1] for x in per[a]]) for a in SWEEPS}     # [n_points, n_seeds]
    Rs = {a: np.array([x[2] for x in per[a]]) for a in SWEEPS}
    mean = lambda D, idx=None: {a: (v if idx is None else v[:, idx]).mean(-1) for a, v in D.items()}
    est = deltas(mean(Hs), mean(Rs))
    boots = {rv: [] for rv in est}
    for idx in boot_idx(len(TEST_SEEDS)):
        d = deltas(mean(Hs, idx), mean(Rs, idx))
        for rv in d:
            boots[rv].append(d[rv])
    ci = {rv: np.nanquantile(np.array(b), [0.025, 0.975], axis=0) for rv, b in boots.items()}
    lo = {rv: np.where(np.isnan(est[rv]), np.nan, ci[rv][0]) for rv in est}
    gate, dead = verdict(est, lo)
    est_dom = deltas(mean(Hs), mean(Rs), dominance_frontier)
    bd = {rv: [] for rv in est_dom}
    for idx in boot_idx(len(TEST_SEEDS), n_boot=2000, seed=1):
        d = deltas(mean(Hs, idx), mean(Rs, idx), dominance_frontier)
        for rv in d:
            bd[rv].append(d[rv])
    ci_dom = {rv: np.nanquantile(np.array(b), [0.025, 0.975], axis=0) for rv, b in bd.items()}
    gate_dom, dead_dom = verdict(est_dom, {rv: np.where(np.isnan(est_dom[rv]), np.nan, ci_dom[rv][0]) for rv in est_dom})
    from scipy.stats import spearmanr
    gains = np.array(SWEEPS["ring"][1], float)
    mh, mr = mean(Hs), mean(Rs)
    res = {"test": "KT3 stubbornness frontier (H3)", "seeds": "test 1000-1299", "t_cross_s": T_CROSS,
           "points": {a: [{"knob": SWEEPS[a][0], "value": (None if v is None else float(v)), "hold_mean_s": float(Hs[a][i].mean()),
                           "release_mean_s": float(Rs[a][i].mean()), "hold_sem": float(Hs[a][i].std() / np.sqrt(Hs[a].shape[1])),
                           "release_sem": float(Rs[a][i].std() / np.sqrt(Rs[a].shape[1])),
                           "on_target_at_4s": float(per[a][i][3]), "on_decoy_at_5s": float(per[a][i][4])}
                          for i, (v, *_r) in enumerate(per[a])] for a in SWEEPS},
           "ring_vs_rival": {rv: [{"ring_gain": float(g), "delta_s": None if np.isnan(e) else float(e),
                                   "lo": None if np.isnan(e) else float(l), "hi": None if np.isnan(e) else float(h)}
                                  for g, e, l, h in zip(gains, est[rv], ci[rv][0], ci[rv][1])] for rv in est},
           "spearman_vs_gain": {"hold": float(spearmanr(gains, mh["ring"])[0]), "release": float(spearmanr(gains, mr["ring"])[0])},
           "gate": gate, "dead_because": dead,
           "posthoc_dominance_frontier": {"note": "post hoc, labelled: R(h) = min rival release over points with hold >= h; 2,000 resamples",
                                          "gate": gate_dom, "dead_because": dead_dom,
                                          "delta": {rv: [{"ring_gain": float(g), "delta_s": None if np.isnan(e) else float(e),
                                                          "lo": None if np.isnan(e) else float(l), "hi": None if np.isnan(e) else float(h)}
                                                         for g, e, l, h in zip(gains, est_dom[rv], ci_dom[rv][0], ci_dom[rv][1])] for rv in est_dom}},
           "rule": "ALIVE if >=2 adjacent ring gains with lower CI of Delta_SM and Delta_acc > 0; DEAD if SM or acc has Delta <= 0 wherever defined",
           "provenance": stamp(seed="test 1000-1299", param_source="results/kt2_tuned_params.json + swept knob")}
    (ROOT / "results/kt3.json").write_text(json.dumps(res, indent=1))
    figure(res)
    for a in SWEEPS:
        print(a, " ".join(f"({p['hold_mean_s']:.2f},{p['release_mean_s']:.2f})" for p in res["points"][a]))
    for rv in est:
        print(f"Delta_{rv}:", " ".join("nan" if np.isnan(e) else f"{e:+.2f}[{l:+.2f},{h:+.2f}]" for e, l, h in zip(est[rv], ci[rv][0], ci[rv][1])))
    print("spearman", res["spearman_vs_gain"], "-> eureka", gate, dead, "| post-hoc dominance frontier:", gate_dom, dead_dom)
    for rv in est_dom:
        print(f"  posthoc Delta_{rv}:", " ".join("nan" if np.isnan(e) else f"{e:+.2f}" for e in est_dom[rv]))


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    sty = {"ring": ("C1", "-o"), "state_machine": ("C0", "-s"), "accumulator": ("C2", "-^"),
           "vote": ("0.4", "D"), "sum": ("0.6", "P"), "ring_static": ("0.2", "X")}
    for a, pts in res["points"].items():
        c, m = sty[a]
        ax[0].errorbar([p["hold_mean_s"] for p in pts], [p["release_mean_s"] for p in pts],
                       xerr=[1.96 * p["hold_sem"] for p in pts], yerr=[1.96 * p["release_sem"] for p in pts],
                       fmt=m, color=c, ms=4, lw=1, label=a + (f" ({pts[0]['knob']} swept)" if pts[0]["knob"] else ""))
    ax[0].set(xlabel="hold a hidden target (s, cap 10)", ylabel="release a decoy (s, cap 20)",
              title="stubbornness frontier (lower-right is better)")
    ax[0].legend(fontsize=7)
    for rv, c in (("state_machine", "C0"), ("accumulator", "C2")):
        d = res["ring_vs_rival"][rv]
        g = [x["ring_gain"] for x in d if x["delta_s"] is not None]
        ax[1].errorbar(g, [x["delta_s"] for x in d if x["delta_s"] is not None],
                       yerr=[[x["delta_s"] - x["lo"] for x in d if x["delta_s"] is not None],
                             [x["hi"] - x["delta_s"] for x in d if x["delta_s"] is not None]], fmt="-o", color=c, ms=3, label=f"vs {rv}")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(xlabel="ring_gain", ylabel="rival release - ring release at matched hold (s)",
              title=f"> 0: ring above rival. Eureka {res['gate']}")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "results/fig_kt3.png", dpi=130)


def selfcheck():
    s = np.arange(4)
    # vote has no memory: loses a hidden target at once; an accumulator with a long leak holds it
    hv, on = trial("vote", {}, s, "occ")
    assert on.all() and (hv < 0.5).all(), hv
    ha, _ = trial("accumulator", {"tau_leak_s": 20.0, "g_search": 0.1}, s, "occ")
    assert (ha >= HOLD_CAP).all(), ha
    hs, _ = trial("state_machine", {"timeout_s": 3.0}, s, "occ")
    assert np.all(np.abs(hs - 3.0) < 0.5), hs
    # decoy trial: everyone starts on the decoy; a memoryless vote releases quickly after the crossing
    rv, ond = trial("vote", {}, s, "dec")
    assert ond.all() and (rv < 2.0).all(), rv
    # Pareto interpolation: dominated points are ignored; outside range is NaN
    f = pareto_interp([1, 2, 3, 2.5], [1, 2, 3, 5])
    assert np.isclose(f(1.5), 1.5) and np.isnan(f(0.5)) and np.isnan(f(3.5)) and np.isclose(f(2.5), 2.5)
    g = dominance_frontier([1, 2, 3], [0, 0, 0])
    assert np.allclose(g([0.5, 2.5, 3]), 0) and np.isnan(g(3.5))[0]
    print("kt3 self-check OK")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main()
