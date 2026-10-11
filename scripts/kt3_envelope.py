"""POST HOC sensitivity analysis for KT3 (H3), training seeds 0-99 only. Not pre-registered.

Prompted by an independent review: the pre-registered KT3 sweeps the ring's gain while its other parameters stay at the
KT2 optimum (tuned at gain ~0.02). Here every one of the 64 random-search configs of the ring, the accumulator and the
state machine (the same configs as KT2, kt2_five_arms.configs) is run through the unchanged KT3 trials (imported from
kt3_stubbornness) on training seeds. Per arm: each config's mean (hold, release) and the Pareto envelope on the
hold/release plane. The ring configs are compared with each rival's envelope using the KT3 frontier functions
(pareto_interp, the pre-registered form; dominance_frontier, the post hoc form): Delta = rival release at the ring's
hold - ring release (> 0: ring better). As a second rival frontier, the rival's 64 configs are pooled with its KT3
knob sweep (KT2-tuned params) on the same training seeds. The best ring config is selected on the same seeds it is
scored on, so its Delta is optimistic for the ring; its bootstrap CI keeps the selection fixed.

Usage: uv run python scripts/kt3_envelope.py [--selfcheck]      (KT_PROCS=n caps worker processes)
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from kt2_five_arms import TRAIN_SEEDS, configs
from kt3_stubbornness import PROCS, REL_CAP, SWEEPS, dominance_frontier, pareto_interp, trial
from provenance import stamp
from stats import boot_idx

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("ring", "accumulator", "state_machine")
RIVALS = ("state_machine", "accumulator")
FRONTIERS = {"pareto_interp": pareto_interp, "dominance_frontier": dominance_frontier}


def _job(a):
    arm, params, seeds = a
    return trial(arm, params, seeds, "occ")[0], trial(arm, params, seeds, "dec")[0]


def run(seeds, n_cfg=64, sweep=True):
    """-> points {arm: [{"source", "index", "params"}]}, H, R {arm: [n_points, n_seeds]}."""
    tuned = json.loads((ROOT / "results/kt2_tuned_params.json").read_text())["arms"]
    pts = {a: [{"source": "random_config", "index": i, "params": p} for i, p in enumerate(configs(a)[:n_cfg])] for a in ARMS}
    if sweep:
        for a in RIVALS:
            knob, vals = SWEEPS[a]
            pts[a] += [{"source": "kt3_sweep", "index": j, "params": dict(tuned[a]["params"]) | {knob: float(v)}} for j, v in enumerate(vals)]
    jobs = [(a, p["params"], seeds) for a in ARMS for p in pts[a]]
    with ProcessPoolExecutor(max_workers=PROCS) as ex:
        out = list(ex.map(_job, jobs))
    H, R, k = {}, {}, 0
    for a in ARMS:
        n = len(pts[a])
        H[a], R[a] = np.array([o[0] for o in out[k:k + n]]), np.array([o[1] for o in out[k:k + n]])
        k += n
    return pts, H, R


def efficient(h, r):
    """Pareto-efficient points (max hold, min release); the same rule as kt3_stubbornness.pareto_interp."""
    return np.array([not np.any((h >= h[i]) & (r <= r[i]) & ((h > h[i]) | (r < r[i]))) for i in range(len(h))])


def deltas(mh, mr, ring_sel, rival_sel):
    """Delta [n_ring_points] per (rival, frontier); ring_sel / rival_sel: boolean masks over each arm's points."""
    return {rv: {fn: f(mh[rv][rival_sel[rv]], mr[rv][rival_sel[rv]])(mh["ring"][ring_sel]) - mr["ring"][ring_sel]
                 for fn, f in FRONTIERS.items()} for rv in RIVALS}


def analyse(pts, H, R, n_boot=2000):
    mh, mr = {a: H[a].mean(1) for a in ARMS}, {a: R[a].mean(1) for a in ARMS}
    rnd = {a: np.array([p["source"] == "random_config" for p in pts[a]]) for a in ARMS}
    sets = {"random64": rnd, "random64_plus_kt3_sweep": {a: np.ones(len(pts[a]), bool) for a in ARMS}}
    gains = np.array([p["params"]["ring_gain"] for p in pts["ring"]])
    out = {"configs": {a: [{"source": p["source"], "index": p["index"], "params": p["params"], "hold_mean_s": float(mh[a][i]),
                            "release_mean_s": float(mr[a][i])} for i, p in enumerate(pts[a])] for a in ARMS},
           "envelope": {a: [int(i) for i in np.flatnonzero(rnd[a])[efficient(mh[a][rnd[a]], mr[a][rnd[a]])]] for a in ARMS},
           "ring_high_gain": {"n_configs_gain_ge_0.7": int((gains >= 0.7).sum()),
                              "frac_gain_ge_0.7_releasing_before_cap": float((mr["ring"][gains >= 0.7] < REL_CAP - 1e-9).mean())},
           "best_ring_vs_rival": {}}
    boot = [({a: H[a][:, i].mean(1) for a in ARMS}, {a: R[a][:, i].mean(1) for a in ARMS})
            for i in boot_idx(H["ring"].shape[1], n_boot=n_boot, seed=0)]
    for sname, sel in sets.items():
        d = deltas(mh, mr, rnd["ring"], sel)
        dboot = [deltas(bh, br, rnd["ring"], sel) for bh, br in boot]
        res = {}
        for rv in RIVALS:
            for fn in FRONTIERS:
                v = d[rv][fn]
                if np.isnan(v).all():
                    res[f"{rv}|{fn}"] = {"n_ring_configs_defined": 0}
                    continue
                b = int(np.nanargmax(v))
                bs = np.array([x[rv][fn][b] for x in dboot])
                res[f"{rv}|{fn}"] = {"n_ring_configs_defined": int((~np.isnan(v)).sum()), "n_ring_configs_delta_gt_0": int((v > 0).sum()),
                                     "best_ring_config_index": b, "best_ring_params": pts["ring"][b]["params"],
                                     "best_ring_hold_s": float(mh["ring"][b]), "best_ring_release_s": float(mr["ring"][b]),
                                     "delta_s": float(v[b]), "lo": float(np.nanquantile(bs, 0.025)), "hi": float(np.nanquantile(bs, 0.975)),
                                     "frac_resamples_defined": float((~np.isnan(bs)).mean())}
        out["best_ring_vs_rival"][sname] = res
    return out


def figure(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for a, c, m in (("ring", "C1", "o"), ("state_machine", "C0", "s"), ("accumulator", "C2", "^")):
        cf = out["configs"][a]
        rnd = [x for x in cf if x["source"] == "random_config"]
        ax.scatter([x["hold_mean_s"] for x in rnd], [x["release_mean_s"] for x in rnd], s=14, color=c, marker=m, alpha=0.45,
                   label=f"{a}: 64 random configs")
        env = sorted((cf[i] for i in out["envelope"][a]), key=lambda x: x["hold_mean_s"])
        ax.step([x["hold_mean_s"] for x in env], [x["release_mean_s"] for x in env], where="pre", color=c, lw=1.6)   # = dominance_frontier
        sw = [x for x in cf if x["source"] == "kt3_sweep"]
        if sw:
            ax.plot([x["hold_mean_s"] for x in sw], [x["release_mean_s"] for x in sw], ":", color=c, lw=1, label=f"{a}: KT3 sweep")
    b = out["best_ring_vs_rival"]["random64"]["accumulator|pareto_interp"]
    if "best_ring_config_index" in b:
        ax.plot(b["best_ring_hold_s"], b["best_ring_release_s"], "*", ms=14, color="C1", mec="k",
                label=f"best ring config vs accumulator envelope (Delta {b['delta_s']:+.2f} s)")
    ax.set(xlabel="hold a hidden target (s, cap 10)", ylabel="release a decoy (s, cap 20)",
           title="POST HOC, training seeds 0-99: all random-search configs (lower-right is better)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    pts, H, R = run(TRAIN_SEEDS)
    out = analyse(pts, H, R)
    res = {"test": "KT3 envelope sensitivity (H3)",
           "label": "POST HOC sensitivity analysis, not pre-registered; TRAINING seeds 0-99 only (no test seed used)",
           "seeds": "train 0-99",
           "design": "KT3 trials unchanged (kt3_stubbornness.trial); all 64 KT2 random-search configs of ring, accumulator and "
                     "state_machine (kt2_five_arms.configs, numpy default_rng(2026)); envelope = Pareto-efficient random configs "
                     "(max hold, min release); Delta = rival frontier release at the ring config's hold - ring release (> 0: ring "
                     "better), frontier functions from kt3_stubbornness; rival sets 'random64' and 'random64_plus_kt3_sweep' (the "
                     "rival's KT3 knob sweep at KT2-tuned params added); best ring config = max Delta, selected on the same seeds "
                     "(optimistic for the ring); CI = 2,000 paired bootstrap resamples over seeds with the selection fixed",
           **out, "provenance": stamp(seed="train 0-99", param_source="kt2_five_arms.configs (64 per arm); results/kt2_tuned_params.json (KT3 sweeps)")}
    (ROOT / "results/kt3_envelope.json").write_text(json.dumps(res, indent=1))
    figure(res, ROOT / "results/fig_kt3_envelope.png")
    print(res["ring_high_gain"])
    for s, d in res["best_ring_vs_rival"].items():
        for k, v in d.items():
            if "delta_s" in v:
                print(f"{s:24s} {k:34s} best ring #{v['best_ring_config_index']:2d} gain {v['best_ring_params']['ring_gain']:.2f} "
                      f"(hold {v['best_ring_hold_s']:.2f}, release {v['best_ring_release_s']:.2f}) Delta {v['delta_s']:+.2f} "
                      f"[{v['lo']:+.2f}, {v['hi']:+.2f}]; defined {v['n_ring_configs_defined']}, > 0: {v['n_ring_configs_delta_gt_0']}")
            else:
                print(f"{s:24s} {k:34s} undefined for every ring config")


def selfcheck():
    seeds = TRAIN_SEEDS[:4]
    pts, H, R = run(seeds, n_cfg=3, sweep=False)
    assert all(H[a].shape == R[a].shape == (3, 4) for a in ARMS)
    h, _ = trial("ring", configs("ring")[1], seeds, "occ")
    assert np.array_equal(h, H["ring"][1])                       # same trial code, same numbers
    out = analyse(pts, H, R, n_boot=20)
    for a in ARMS:                                               # envelope points are mutually non-dominated
        e = out["envelope"][a]
        hh, rr = H[a].mean(1)[e], R[a].mean(1)[e]
        assert len(e) >= 1 and efficient(hh, rr).all()
    f = pareto_interp([1, 2, 3, 2.5], [1, 2, 3, 5])
    assert efficient(np.array([1, 2, 3, 2.5]), np.array([1, 2, 3, 5])).tolist() == [True, True, True, False] and np.isclose(f(2), 2)
    for v in out["best_ring_vs_rival"]["random64"].values():
        if "delta_s" in v:
            mh = {a: H[a].mean(1) for a in ARMS}
            assert np.isclose(v["best_ring_hold_s"], mh["ring"][v["best_ring_config_index"]])
    print("kt3_envelope self-check OK")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main()
