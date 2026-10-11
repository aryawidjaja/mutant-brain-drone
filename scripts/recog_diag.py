"""EXPLORATORY post-hoc diagnostic for H4 (declared in docs/LOG.md before running): does the pipeline carry
object signal when the pose matches the goal view, and how much pose jitter / sun change breaks each recogniser?

Training seeds 0-4 only. Probes at theta 0, 6 m; jitter scale s in {0, 0.25, 0.5, 1} times the pre-registered
jitter (height U[1.5,3] about 2.0 m, azimuth +-4 deg, distance +-5%, gaze +-0.2 m); memory sun vs a new sun.
  uv run python scripts/recog_diag.py   -> results/recog_diag.json (self-check asserts at the end)
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recog_scene import Scene, OBJECTS  # noqa: E402
from recog_run import score_layout, Dino, LF_THETAS  # noqa: E402
from recog_score import u_matrix  # noqa: E402
from brain.mb import MushroomBody  # noqa: E402
from provenance import stamp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALES, N_REP, SEEDS = [0.0, 0.25, 0.5, 1.0], 8, [0, 1, 2, 3, 4]


def main():
    dino, mb, mbe = Dino(), MushroomBody(1024, seed=0), MushroomBody(1024, seed=0)
    rec_scores = {}                                                       # (rec, sun, s) -> {"pos": [...], "neg": [...]} per seed
    for seed in SEEDS:
        sc = Scene(seed)
        sc.set_sun(0); sc.show("target")
        mem = np.stack([sc.render(0, 6.0, 2.0)] + [sc.render(t, 6.0, 2.0) for t in LF_THETAS])
        rs = np.random.default_rng([seed, 13])
        imgs, meta = [], []
        for sun in (0, 1):
            sc.set_sun(sun)
            for s in SCALES:
                for k in range(1 if s == 0 else N_REP):
                    j = dict(th=s * rs.uniform(-4, 4), d=6.0 * (1 + s * rs.uniform(-0.05, 0.05)), h=2.0 + s * (rs.uniform(1.5, 3.0) - 2.0),
                             g=(s * rs.uniform(-0.2, 0.2), s * rs.uniform(-0.2, 0.2)))
                    for o in OBJECTS:
                        sc.show(o)
                        imgs.append(sc.render(j["th"], j["d"], j["h"], j["g"])); meta.append((sun, s, o, j["d"]))
        sc.close()
        out, _ = score_layout(mem, np.stack(imgs), dino, mb, mbe, [m[3] for m in meta])
        for rec, d in out.items():
            for (sun, s, o, _), v in zip(meta, d[0]):                       # LF off: goal view only
                rec_scores.setdefault((rec, sun, s), {}).setdefault(seed, {"pos": [], "neg": []})["pos" if o == "target" else "neg"].append(v)
        print("seed", seed, "done", flush=True)
    res = {}
    for (rec, sun, s), by_seed in rec_scores.items():
        U, p, n = u_matrix([np.array(by_seed[x]["pos"]) for x in SEEDS], [np.array(by_seed[x]["neg"]) for x in SEEDS])
        res.setdefault(rec, {}).setdefault("memory_sun" if sun == 0 else "new_sun", {})[f"{s:g}"] = {
            "auroc": round(float(U.sum() / (p.sum() * n.sum())), 4),
            "target_score_mean": float(np.mean([v for x in SEEDS for v in by_seed[x]["pos"]])),
            "decoy_score_mean": float(np.mean([v for x in SEEDS for v in by_seed[x]["neg"]]))}
    json.dump({**stamp(seed=SEEDS, param_source="scripts/recog_diag.py (EXPLORATORY, post-hoc; training seeds)"),
               "note": "AUROC target vs swapped decoys at theta 0, 6 m, LF off, pooled over 5 training layouts; keys = jitter scale "
                       "(x pre-registered jitter). s=0 has 1 probe per object per layout (n_pos=5, n_neg=15), else 8 reps.",
               "results": res}, open(os.path.join(ROOT, "results", "recog_diag.json"), "w"), indent=1)
    for rec in res:
        print(f"{rec:12s}", " | ".join(f"{k}: " + " ".join(f"{v['auroc']:.2f}" for v in d.values()) for k, d in res[rec].items()))
    # self-check: at zero jitter under the memory sun the target IS the goal image -> MB familiarity 1, pixels exact
    z = res["mb"]["memory_sun"]["0"]
    assert z["target_score_mean"] == 1.0 and z["decoy_score_mean"] < 1.0 and z["auroc"] >= 0.85, z
    assert res["pixels"]["memory_sun"]["0"]["auroc"] == 1.0
    print("recog_diag self-check ok")


if __name__ == "__main__":
    main()
