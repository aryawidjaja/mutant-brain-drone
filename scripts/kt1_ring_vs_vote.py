"""KT1 (H2): does the ring's steady steering differ from a static population vote by > 5 deg anywhere?

Pre-registered in docs/LOG.md ([WS-A] KT1). Writes results/kt1.json and results/fig_kt1.png.
Usage: uv run python scripts/kt1_ring_vs_vote.py [--selfcheck]
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from brain.arbiters import Ring, make, wrap, _obs
from provenance import stamp

ROOT = Path(__file__).resolve().parents[1]
DT = 0.02
DELTAS = np.arange(60, 181, 15)
RATIOS = np.array([0.25, 1 / 3, 0.5, 2 / 3, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0])
KAPPA_GRID = np.geomspace(0.25, 16, 25)


def grid():
    D, R = np.meshgrid(DELTAS, RATIOS, indexing="ij")
    D, R = D.ravel().astype(float), R.ravel()
    w1 = np.where(R >= 1, 0.8 / R, 0.8)
    w2 = np.where(R >= 1, 0.8, 0.8 * R)
    valid = ~((D == 180) & (R == 1.0))
    return D, R, w1, w2, valid


def steady(name, params, D, w1, w2, T=10.0):
    B = len(D)
    arb = make(name, params, B)
    o = _obs(B, 2, np.stack([np.zeros(B), np.deg2rad(D)], 1), np.stack([w1, w2], 1), True)
    for _ in range(int(round(T / DT))):
        out = arb.step(o, DT)
    return np.rad2deg(out["heading_cmd"])


def vecsum(D, w1, w2):
    return np.rad2deg(np.angle(w1 + w2 * np.exp(1j * np.deg2rad(D))))


def dev(a, b):
    return np.abs(np.rad2deg(wrap(np.deg2rad(a - b))))


def compare(ring_params):
    D, R, w1, w2, valid = grid()
    ring = steady("ring", ring_params, D, w1, w2)
    ring9 = steady("ring", ring_params, D, w1, w2, T=9.0)
    vp = lambda k: {k2: ring_params[k2] for k2 in ("g_obj", "g_home", "g_search")} | {"kappa": float(k)}
    kap = np.unique(np.append(KAPPA_GRID, ring_params["kappa"]))
    maxdev = np.array([dev(ring, steady("vote", vp(k), D, w1, w2, T=DT))[valid].max() for k in kap])
    kbest = kap[maxdev.argmin()]
    noties = valid & (R != 1.0)  # post-hoc diagnostic: equal-certainty cells are tie-breaks, not computation
    maxdev_nt = np.array([dev(ring, steady("vote", vp(k), D, w1, w2, T=DT))[noties].max() for k in kap])
    vote_best = steady("vote", vp(kbest), D, w1, w2, T=DT)
    vote_same = steady("vote", vp(ring_params["kappa"]), D, w1, w2, T=DT)
    vs = vecsum(D, w1, w2)
    return {"ring_params": ring_params,
            "D_star_deg": float(maxdev.min()), "kappa_v_best": float(kbest),
            "maxdev_by_kappa": dict(zip(map(float, kap), map(float, maxdev))),
            "posthoc_D_star_excl_ratio1_deg": float(maxdev_nt.min()), "posthoc_kappa_v_best_excl_ratio1": float(kap[maxdev_nt.argmin()]),
            "same_kappa_maxdev_deg": float(dev(ring, vote_same)[valid].max()),
            "vecsum_maxdev_deg": float(dev(ring, vs)[valid].max()),
            "ring_converged_maxchange_deg": float(dev(ring, ring9)[valid].max()),
            "cells": [{"delta_deg": float(d), "ratio": float(r), "ring_deg": float(a), "vote_best_deg": float(b),
                       "vote_same_deg": float(c), "vecsum_deg": float(s), "valid": bool(v)}
                      for d, r, a, b, c, s, v in zip(D, R, ring, vote_best, vote_same, vs, valid)]}


def hysteresis(name, params, delta_deg, up=(0.5, 0.75, 1, 1.25, 1.5, 2, 2.5), w1=0.4, t_step=3.0):
    """Commit to cue 1 for 5 s, then step cue 2 up then down. Returns headings per step and switch ratios."""
    arb = make(name, params, 1)
    o1 = _obs(1, 2, [[0.0, np.deg2rad(delta_deg)]], [[w1, 0.0]], [[True, False]])
    for _ in range(int(5 / DT)):
        arb.step(o1, DT)
    seq = list(up) + list(up[::-1][1:])
    heads = []
    for r in seq:
        o = _obs(1, 2, [[0.0, np.deg2rad(delta_deg)]], [[w1, w1 * r]], True)
        for _ in range(int(t_step / DT)):
            out = arb.step(o, DT)
        heads.append(float(np.rad2deg(out["heading_cmd"][0])))
    on2 = [dev(h, delta_deg) < 20 for h in heads]
    n = len(up)
    s_up = next((seq[i] for i in range(n) if on2[i]), None)
    s_down = None if s_up is None else next((seq[i] for i in range(n - 1, len(seq)) if not on2[i]), None)
    return {"ratios": seq, "heading_deg": heads, "switch_up_ratio": s_up, "switch_back_ratio": s_down}


def sensitivity():
    out = []
    for g in (0.0, 0.5, 1.0, 2.0, 3.0):
        for k in (1.0, 2.0, 4.0):
            r = compare({**Ring.DEFAULTS, "ring_gain": g, "kappa": k})
            out.append({"ring_gain": g, "kappa": k, "D_star_deg": r["D_star_deg"], "kappa_v_best": r["kappa_v_best"],
                        "same_kappa_maxdev_deg": r["same_kappa_maxdev_deg"]})
    return out


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = res["primary"]["cells"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for r, col in ((1.0, "C1"), (0.5, "C0"), (2.0, "C2")):
        sel = [x for x in c if abs(x["ratio"] - r) < 1e-9 and x["valid"]]
        d = [x["delta_deg"] for x in sel]
        ax[0].plot(d, [x["ring_deg"] for x in sel], "-o", color=col, label=f"ring, w2/w1={r:g}")
        ax[0].plot(d, [x["vote_best_deg"] for x in sel], "--", color=col, label=f"vote (best kappa), {r:g}")
        ax[0].plot(d, [x["vecsum_deg"] for x in sel], ":", color=col, alpha=0.6)
    ax[0].set(xlabel="cue separation (deg)", ylabel="steady heading (deg); cue 1 at 0", title="ring vs static vote (dotted: vector sum)")
    ax[0].legend(fontsize=6)
    k = res["primary"]["maxdev_by_kappa"]
    ax[1].semilogx(list(map(float, k)), list(k.values()), "-o", color="0.4")
    ax[1].axhline(5, color="r", lw=0.8)
    ax[1].set(xlabel="vote input width kappa_v", ylabel="max |ring - vote| over cells (deg)", title=f"D* = {res['primary']['D_star_deg']:.1f} deg")
    for d, h in res["hysteresis"].items():
        ax[2].plot(h["ring"]["heading_deg"], "-o", label=f"ring, cue2 at {d}")
        ax[2].plot(h["vote"]["heading_deg"], "--", label=f"vote, cue2 at {d}")
    ax[2].set_xticks(range(len(h["ring"]["ratios"])), [f"{x:g}" for x in h["ring"]["ratios"]], fontsize=7)
    ax[2].set(xlabel="cue2/cue1 ratio (up then down)", ylabel="heading (deg)", title="hysteresis")
    ax[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    primary = compare(dict(Ring.DEFAULTS))
    hyst = {}
    for d in (90, 120, 170):
        kv = {"kappa": primary["kappa_v_best"], **{k: Ring.DEFAULTS[k] for k in ("g_obj", "g_home", "g_search")}}
        hyst[str(d)] = {"ring": hysteresis("ring", dict(Ring.DEFAULTS), d), "vote": hysteresis("vote", kv, d),
                        "vote_same_kappa": hysteresis("vote", {k: Ring.DEFAULTS[k] for k in ("kappa", "g_obj", "g_home", "g_search")}, d)}
    res = {"test": "KT1 ring vs static vote (H2)", "rule": "KILL H2 if D* <= 5 deg (vote kappa chosen to minimise max deviation)",
           "primary": primary, "gate": "KILL" if primary["D_star_deg"] <= 5 else "PASS",
           "sensitivity": sensitivity(), "hysteresis": hyst}
    tuned = ROOT / "results/kt2_tuned_params.json"
    if tuned.exists():
        tp = json.loads(tuned.read_text())["arms"]["ring"]["params"]
        s = compare(tp)
        res["secondary_tuned_ring"] = {k: s[k] for k in ("ring_params", "D_star_deg", "kappa_v_best", "same_kappa_maxdev_deg", "vecsum_maxdev_deg", "posthoc_D_star_excl_ratio1_deg", "ring_converged_maxchange_deg")}
    res["provenance"] = stamp(seed=None, param_source="arbiters.Ring.DEFAULTS (primary); results/kt2_tuned_params.json (secondary)")
    (ROOT / "results/kt1.json").write_text(json.dumps(res, indent=1))
    figure(res, ROOT / "results/fig_kt1.png")
    p = primary
    print(f"KT1: D*={p['D_star_deg']:.2f} deg at kappa_v={p['kappa_v_best']:.2f}; same-kappa {p['same_kappa_maxdev_deg']:.1f}; "
          f"vecsum {p['vecsum_maxdev_deg']:.1f}; post-hoc excl. ratio 1: {p['posthoc_D_star_excl_ratio1_deg']:.1f}; converged within {p['ring_converged_maxchange_deg']:.3f} deg -> {res['gate']}")
    for d, h in hyst.items():
        print(f"  hysteresis cue2 at {d}: ring up {h['ring']['switch_up_ratio']} back {h['ring']['switch_back_ratio']} | "
              f"vote up {h['vote']['switch_up_ratio']} back {h['vote']['switch_back_ratio']}")
    if "secondary_tuned_ring" in res:
        print(f"  secondary (tuned ring): D*={res['secondary_tuned_ring']['D_star_deg']:.2f}")
    return res


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        D = np.array([60.0, 120.0]); w = np.array([0.8, 0.8])
        assert np.allclose(vecsum(D, w, w), [30, 60])
        # vote with very wide input ~ vector sum; one cue -> heading on it
        v = steady("vote", {"kappa": 0.05}, D, w, w, T=DT)
        assert np.all(dev(v, [30, 60]) < 3), v
        h = hysteresis("vote", {"kappa": 4.0}, 120)
        assert h["switch_up_ratio"] == 1.25 and h["switch_back_ratio"] == 1.0, h  # no memory: on cue 2 only while ratio > 1
        print("kt1 self-check OK")
    else:
        main()
