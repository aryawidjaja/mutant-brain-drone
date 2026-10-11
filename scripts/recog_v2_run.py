"""H8 recognition v2: render (MuJoCo, v1 scene and probe protocol), object-centred crops, every arm x memory set.

  uv run python scripts/recog_v2_run.py render --split train      # layouts 0-9    -> results/recog_renders/v2_cache/train_<seed>.npz
  uv run python scripts/recog_v2_run.py tune                      # front-end design search on train (--round 1|2) -> results/recog_v2_tune.json
  uv run python scripts/recog_v2_run.py render --split test       # layouts 1010-1019 (+ DINOv2 scores) -> v2_cache/test_<seed>.npz
  uv run python scripts/recog_v2_run.py score --split test        # -> results/recog_v2_scores_test.parquet, results/recog_v2_cost.json
  uv run python scripts/recog_v2_run.py inspect                   # approach sequences, train layouts -> results/recog_v2_inspect.json
  uv run python scripts/recog_v2_run.py --selfcheck

Memory sets (stored views of the target): goal = 1 view (0 deg, 6 m, 2 m, memory sun); lf13 = v1 learning flight
(goal + 12 views on a +-45 deg arc at 6 m); ms = goal + 30 multi-scale views (5 azimuths x 3 distances x 2 heights,
cycling over 3 memory suns); ms1 = the same 30 poses under the memory sun only (secondary).
"""
import argparse, json, os, sys, time
import multiprocessing as mp
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain.mb import MushroomBody, preprocess  # noqa: E402
from brain import mb_front  # noqa: E402
from recog_run import probe_plan, colour_hist, Dino, central_descriptor, LF_THETAS  # noqa: E402
from provenance import stamp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE0 = os.path.join(ROOT, "results", "recog_renders", "v2_cache")
SPLITS = {"train": list(range(0, 10)), "test": list(range(1010, 1020))}
F_PX = 112 / np.tan(np.radians(30))           # 224 px, 60 deg vertical FOV
OBJ_Z, CROP = 0.6, 48
HALF_M = mb_front.CFG["half_m"]          # crop half-width (m); --half_m overrides for the design search


def cache_dir():
    return os.path.join(CACHE0, f"h{HALF_M:g}")
MS_AZ, MS_D, MS_H = [-45, -22.5, 0, 22.5, 45], [3.0, 6.0, 12.0], [1.5, 3.0]
MEMSETS = {"goal": [0], "lf13": list(range(13)), "ms": [0] + list(range(13, 43)), "ms1": [0] + list(range(43, 73))}
ARMS = ["pixels", "pixels_crop", "colour", "mb_v1", "mb_crop", "mb_front", "nn_front", "dino_patch"]
PROCS = 3


def mem_poses():
    """[(theta, dist, height, sun_slot)] for the 73 memory renders; sun_slot 0 = layout memory sun, 1/2 = extra suns."""
    p = [(0.0, 6.0, 2.0, 0)] + [(float(t), 6.0, 2.0, 0) for t in LF_THETAS]
    ms = [(float(a), d, h) for a in MS_AZ for d in MS_D for h in MS_H]
    return p + [(a, d, h, i % 3) for i, (a, d, h) in enumerate(ms)] + [(a, d, h, 0) for a, d, h in ms]


def extra_suns(seed):
    rs = np.random.default_rng([seed, 21])
    return [dict(az=float(rs.uniform(0, 360)), el=float(rs.uniform(20, 70)), diffuse=float(rs.uniform(0.5, 1.0)),
                 ambient=float(rs.uniform(0.15, 0.35))) for _ in range(2)]


def crop(img, rng3d):
    """Object-centred square crop (half-width F_PX * HALF_M / range, about the image centre) -> CROP x CROP RGB uint8."""
    h = int(round(min(112, float(mb_front.crop_box(rng3d, F_PX, HALF_M)))))
    return np.asarray(Image.fromarray(img[112 - h:112 + h, 112 - h:112 + h]).resize((CROP, CROP), Image.BOX))


def rng3d(d, h):
    return float(np.hypot(d, h - OBJ_Z))


def _set_sun(sc, s):
    az, el = np.radians(s["az"]), np.radians(s["el"])
    sc.m.light_dir[0] = -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    sc.m.light_diffuse[0] = s["diffuse"]; sc.m.light_ambient[0] = s["ambient"]


def render_layout(seed):
    """-> dict of images: mem [73,224,224,3], probes [N,...], meta (probe dicts incl. obj)."""
    from recog_scene import Scene, OBJECTS
    sc = Scene(seed)
    suns = [sc.lay["suns"][0]] + extra_suns(seed)
    sc.show("target")
    mem = []
    for th, d, h, k in mem_poses():
        _set_sun(sc, suns[k]); mem.append(sc.render(th, d, h))
    plan = probe_plan(seed)
    imgs, meta = [], []
    for s in range(2):
        sc.set_sun(1 + s)
        for o in OBJECTS:
            sc.show(o)
            for p in plan:
                if p["sun"] == s:
                    imgs.append(sc.render(p["th_j"], p["d_j"], p["h"], (p["gx"], p["gy"])))
                    meta.append({**p, "obj": o, "layout": seed})
    sc.close()
    return seed, np.stack(mem), np.stack(imgs), meta


def summarise(seed, mem, probes, meta, dino=None):
    """Compact per-layout cache: crops, grey32, colour histograms (+ DINOv2 patch-max scores per memory set)."""
    mp_ = mem_poses()
    out = dict(seed=seed, meta=json.dumps(meta),
               crop=np.stack([crop(im, rng3d(m["d_j"], m["h"])) for im, m in zip(probes, meta)]),
               mem_crop=np.stack([crop(im, rng3d(d, h)) for im, (_, d, h, _) in zip(mem, mp_)]),
               grey=preprocess(probes).astype(np.float32), mem_grey=preprocess(mem).astype(np.float32),
               hist=colour_hist(probes), mem_hist=colour_hist(mem))
    out["mem_crop_ms"] = out["mem_crop"][13:43]
    if dino is not None:
        _, pm = dino(mem)
        dm = central_descriptor(pm)                                              # [73, 384]
        sc = []
        for i in range(0, len(probes), 128):
            _, pp = dino(probes[i:i + 128])
            sc.append(np.einsum("npc,mc->npm", pp, dm).max(1))                    # [n, 73]
        out["dino_pmax"] = np.concatenate(sc).astype(np.float32)
    return out


def render(split):
    os.makedirs(cache_dir(), exist_ok=True)
    seeds = [s for s in SPLITS[split] if not os.path.exists(os.path.join(cache_dir(), f"{split}_{s}.npz"))]
    dino = Dino() if split == "test" else None
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(PROCS, max(1, len(seeds)))) as pool:
        for seed, mem, probes, meta in pool.imap(render_layout, seeds):
            np.savez_compressed(os.path.join(cache_dir(), f"{split}_{seed}.npz"), **summarise(seed, mem, probes, meta, dino))
            if seed == SPLITS[split][0]:
                idx = [i for i, m in enumerate(meta) if m["sun"] == 0 and m["rep"] == 0 and m["theta"] in (0, 90) and m["dist"] in (3.0, 12.0)]
                Image.fromarray(np.concatenate([np.concatenate([crop(probes[i], rng3d(meta[i]["d_j"], meta[i]["h"])) for i in idx[j::4]][:4], 1)
                                                for j in range(4)], 0)).resize((384, 384), Image.NEAREST).save(
                    os.path.join(ROOT, "results", "recog_renders", f"v2_crops_{split}_{seed}.png"))
            print(f"{split} layout {seed}: {len(probes)} probes, {time.time() - t0:.0f}s", flush=True)


def load(split, half_m=None):
    cd = os.path.join(CACHE0, f"h{half_m:g}") if half_m else cache_dir()
    out = []
    for s in SPLITS[split]:
        z = np.load(os.path.join(cd, f"{split}_{s}.npz"))
        out.append({k: z[k] for k in z.files})
    return out


def arm_scores(L, cfg=None, arms=ARMS, memsets=MEMSETS):
    """L = one layout cache -> {(arm, memset): scores [N]} (higher = more target-like)."""
    cfg = cfg or mb_front.CFG
    res, pc, mc, xf, mf = {}, None, None, None, None
    need = lambda a: a in arms  # noqa: E731
    if need("pixels"):
        rms = np.sqrt(((L["grey"][:, None] - L["mem_grey"][None]) ** 2).mean(-1))
        res.update({("pixels", m): -rms[:, v].min(1) for m, v in memsets.items()})
    if need("pixels_crop") or need("mb_crop"):
        pc, mc = preprocess(L["crop"], (24, 24)), preprocess(L["mem_crop"], (24, 24))
        rms = np.sqrt(((pc[:, None] - mc[None]) ** 2).mean(-1))
        res.update({("pixels_crop", m): -rms[:, v].min(1) for m, v in memsets.items()} if need("pixels_crop") else {})
    if need("colour"):
        inter = np.minimum(L["hist"][:, None], L["mem_hist"][None]).sum(-1)
        res.update({("colour", m): inter[:, v].max(1) for m, v in memsets.items()})
    if need("nn_front") or need("mb_front"):
        xf, mf = mb_front.front(L["crop"], **cfg), mb_front.front(L["mem_crop"], **cfg)
        if need("nn_front"):
            rms = np.sqrt(((xf[:, None] - mf[None]) ** 2).mean(-1))
            res.update({("nn_front", m): -rms[:, v].min(1) for m, v in memsets.items()})
    for arm, x, xm in [("mb_v1", L["grey"], L["mem_grey"]), ("mb_crop", pc, mc), ("mb_front", xf, mf)]:
        if need(arm):
            mb = MushroomBody(x.shape[-1], seed=0)
            for m, v in memsets.items():
                mb.reset(); mb.learn(xm[v]); res[(arm, m)] = mb.familiarity(x)
    if need("dino_patch") and "dino_pmax" in L:
        res.update({("dino_patch", m): L["dino_pmax"][:, v].max(1) for m, v in memsets.items()})
    return res


def score(split):
    rows = []
    for L in load(split):
        meta = pd.DataFrame(json.loads(str(L["meta"])))
        for (arm, m), s in arm_scores(L).items():
            rows.append(meta.assign(rec=arm, memset=m, score=s.astype(float)))
    df = pd.concat(rows, ignore_index=True)
    st = stamp(seed=SPLITS[split], param_source=f"H8 protocol (pre-registered); front end {mb_front.CFG}; scripts/recog_v2_run.py",
               dino="facebookresearch/dinov2 dinov2_vits14 (torch hub, Apache-2.0), patch-max as v1")
    import pyarrow as pa, pyarrow.parquet as pq
    tbl = pa.Table.from_pandas(df)
    path = os.path.join(ROOT, "results", f"recog_v2_scores_{split}.parquet")
    pq.write_table(tbl.replace_schema_metadata({**(tbl.schema.metadata or {}), b"provenance": json.dumps(st).encode()}), path)
    print(f"{len(df)} rows -> {path}")
    if split == "test":
        cost(load(split)[0])
    return df


def cost(L):
    """Bytes of stored memory per arm x memory set, and median ms per query (one probe vs the ms memory)."""
    nv = {m: len(v) for m, v in MEMSETS.items()}
    n_front = mb_front.n_in()
    per_view = {"pixels": 1024, "pixels_crop": 576, "colour": 1024, "nn_front": 2 * n_front, "dino_patch": 768}
    stored = {m: {**{a: n * per_view[a] for a in per_view}, "mb_v1": 2500, "mb_crop": 2500, "mb_front": 2500} for m, n in nv.items()}
    q, qc = L["crop"][:1], L["crop"][0]
    mb = MushroomBody(n_front, seed=0); mb.learn(mb_front.front(L["mem_crop"][MEMSETS["ms"]]))
    mf = mb_front.front(L["mem_crop"][MEMSETS["ms"]])
    fns = {"front_only": lambda: mb_front.front(qc), "mb_front": lambda: mb.familiarity(mb_front.front(qc)),
           "nn_front": lambda: np.sqrt(((mb_front.front(qc)[None] - mf) ** 2).mean(-1)).min()}
    ms = {}
    for k, f in fns.items():
        f(); ts = []
        for _ in range(100):
            t0 = time.perf_counter(); f(); ts.append((time.perf_counter() - t0) * 1e3)
        ms[k] = float(np.median(ts))
    v1 = json.load(open(os.path.join(ROOT, "results", "recog_cost.json")))["ms_per_query_median"]
    json.dump({**stamp(seed=int(L["seed"]), param_source="scripts/recog_v2_run.py cost"),
               "stored_memory_bytes": stored, "views_per_memset": nv,
               "notes": f"MB: 20,000 1-bit KC->MBON weights whatever the number of views (PN->KC wiring regenerated from seed). "
                        f"Front end: fixed filters, 0 learned bytes; n_in {n_front}. nn_front stores uint16 ranks per view. "
                        "pixels uint8 32x32/view, pixels_crop uint8 24x24/view, colour uint16 512 bins/view, DINO fp16 384-d/view "
                        "(+ 22.06 M fp32 model weights, 88 MB, not counted here).",
               "ms_per_query_median": ms, "ms_note": "one 48x48 crop vs the 31-view ms memory, M4, numpy CPU, excl. rendering/cropping",
               "v1_ms_per_query_median_reference": v1}, open(os.path.join(ROOT, "results", "recog_v2_cost.json"), "w"), indent=1)
    print("ms/query", ms)


# ---------------- design search (training layouts only) ----------------
GRID_SEARCH = {1: [dict(grid=g, r=4, colour_edges=ce, per_channel=pc, half_m=1.6) for g in (4, 6, 8) for ce in (False, True) for pc in (False, True)],
               2: [dict(grid=g, r=r, colour_edges=True, per_channel=True, half_m=h) for h in (1.1, 1.6) for r in (2, 4) for g in (4, 6)]}


def tune(rnd):
    """Design search round `rnd` on the training layouts; results of all rounds accumulate in recog_v2_tune.json."""
    from recog_v2_score import pooled_auroc
    path = os.path.join(ROOT, "results", "recog_v2_tune.json")
    prev = json.load(open(path))["results"] if rnd > 1 else []
    res = []
    for cfg in GRID_SEARCH[rnd]:
        Ls = load("train", cfg["half_m"])
        metas = [pd.DataFrame(json.loads(str(L["meta"]))) for L in Ls]
        sc = [arm_scores(L, cfg, arms=["mb_front", "nn_front"], memsets={"ms": MEMSETS["ms"]}) for L in Ls]
        r = {"cfg": cfg}
        for arm in ("mb_front", "nn_front"):
            df = pd.concat([m.assign(score=s[(arm, "ms")]) for m, s in zip(metas, sc)])
            a = {t: pooled_auroc(g) for t, g in df.groupby("theta")}
            r[arm] = {"auroc_by_theta": {int(t): round(v, 4) for t, v in a.items()}, "objective_0_45": float(np.mean([a[t] for t in (0, 15, 30, 45)]))}
        res.append({**r, "round": rnd})
        print(cfg, {k: round(r[k]["objective_0_45"], 3) for k in ("mb_front", "nn_front")}, flush=True)
    res = prev + res
    best = max(res, key=lambda r: r["mb_front"]["objective_0_45"])
    json.dump({**stamp(seed=SPLITS["train"], param_source="scripts/recog_v2_run.py tune (training layouts 0-9 only)"),
               "objective": "MB(front) + ms memory: pooled target-vs-swapped-decoy AUROC averaged over 0/15/30/45 deg",
               "budget": len(res), "rounds": {k: len(v) for k, v in GRID_SEARCH.items() if k <= rnd}, "results": res, "best_cfg": best["cfg"]},
              open(path, "w"), indent=1)
    print("best", best["cfg"], best["mb_front"]["objective_0_45"])


# ---------------- inspection rule, offline (training layouts only) ----------------
N_APP, N_FR, ARC = 8, 16, 30.0


def approach_plan(seed):
    """8 approaches per layout: 8 frames straight in from 12 to 6 m at azimuth th0, then 8 frames on a 30-deg arc
    (random side) while closing 6 -> 3 m (inspection range = the arc). Height 2.5 -> 2.0 m, gaze jitter +-0.2 m."""
    rs = np.random.default_rng([seed, 22])
    out = []
    for a in range(N_APP):
        th0, side = rs.uniform(-45, 45), rs.choice([-1.0, 1.0])
        for i in range(N_FR):
            s = i / (N_FR - 1)
            arc = max(0.0, (i - (N_FR // 2 - 1)) / (N_FR // 2)) * ARC * side
            out.append(dict(app=a, fr=i, sun=a % 2, th=float(th0 + arc), d=float(12 * 0.25 ** s), h=2.5 - 0.5 * s,
                            gx=float(rs.uniform(-0.2, 0.2)), gy=float(rs.uniform(-0.2, 0.2)), insp=i >= N_FR // 2 - 1))
    return out


def render_approach(seed):
    from recog_scene import Scene, OBJECTS
    sc = Scene(seed)
    plan = approach_plan(seed)
    crops, meta = [], []
    for o in OBJECTS:
        sc.show(o)
        for p in plan:
            sc.set_sun(1 + p["sun"])
            crops.append(crop(sc.render(p["th"], p["d"], p["h"], (p["gx"], p["gy"])), rng3d(p["d"], p["h"])))
            meta.append({**p, "obj": o, "layout": seed})
    sc.close()
    return seed, np.stack(crops), meta


def inspect():
    from recog_v2_score import inspection_analysis
    Ls = {int(L["seed"]): L for L in load("train")}
    rows = []
    with mp.get_context("spawn").Pool(PROCS) as pool:
        for seed, crops, meta in pool.imap(render_approach, SPLITS["train"]):
            mb = MushroomBody(mb_front.n_in(), seed=0)
            mb.learn(mb_front.front(Ls[seed]["mem_crop"][MEMSETS["ms"]]))
            f = mb.familiarity(mb_front.front(crops))
            rows += [{**m, "fam": float(v)} for m, v in zip(meta, f)]
            print("approach layout", seed, flush=True)
    df = pd.DataFrame(rows)
    out = inspection_analysis(df)
    json.dump({**stamp(seed=SPLITS["train"], param_source="scripts/recog_v2_run.py inspect; mb_front.inspect_stat; training layouts only"),
               "protocol": approach_plan.__doc__.strip(), **out}, open(os.path.join(ROOT, "results", "recog_v2_inspect.json"), "w"), indent=1)
    df.to_parquet(os.path.join(ROOT, "results", "recog_v2_inspect_frames.parquet"))
    print(json.dumps(out["summary"], indent=1))


def selfcheck():
    assert len(mem_poses()) == 73 and mem_poses()[13] == (-45.0, 3.0, 1.5, 0) and mem_poses()[14][3] == 1
    assert all(mem_poses()[i][:3] == mem_poses()[i + 30][:3] for i in range(13, 43))
    assert extra_suns(5) == extra_suns(5) and extra_suns(5) != extra_suns(6)
    seed, mem, probes, meta = render_layout(0)
    keep = [i for i, m in enumerate(meta) if m["theta"] in (0, 90) and m["dist"] == 6.0 and m["rep"] == 0]
    L = summarise(seed, mem, probes[keep], [meta[i] for i in keep])
    assert L["crop"].shape == (len(keep), CROP, CROP, 3) and L["mem_crop"].shape == (73, CROP, CROP, 3)
    sc = arm_scores(L)
    assert {a for a, _ in sc} == set(ARMS) - {"dino_patch"} and all(np.isfinite(v).all() for v in sc.values())
    mb = MushroomBody(mb_front.n_in(), seed=0); mb.learn(mb_front.front(L["mem_crop"][:1]))
    assert mb.familiarity(mb_front.front(L["mem_crop"][:1]))[0] == 1.0           # the goal view is fully familiar
    for a in ("mb_front", "mb_v1"):
        assert (sc[(a, "lf13")] >= sc[(a, "goal")] - 1e-12).all() and (sc[(a, "ms")] >= sc[(a, "goal")] - 1e-12).all()
    p = approach_plan(0)
    assert len(p) == N_APP * N_FR and abs(p[-1]["d"] - 3.0) < 1e-9 and sum(q["insp"] for q in p[:N_FR]) == N_FR // 2 + 1
    arc = [q["th"] for q in p[:N_FR] if q["insp"]]
    assert abs(abs(arc[-1] - arc[0]) - ARC) < 1e-9
    print("recog_v2_run self-check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["render", "tune", "score", "inspect"])
    ap.add_argument("--split", default="train", choices=list(SPLITS))
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--half_m", type=float, default=None)
    ap.add_argument("--round", type=int, default=1)
    a = ap.parse_args()
    if a.half_m:
        HALF_M = a.half_m
    if a.selfcheck:
        selfcheck()
    elif a.cmd == "render":
        render(a.split)
    elif a.cmd == "score":
        score(a.split)
    else:
        tune(a.round) if a.cmd == "tune" else inspect()
