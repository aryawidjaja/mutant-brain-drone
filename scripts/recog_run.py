"""H4 render + recogniser pass: every probe of every test layout -> one score per recogniser x LF on/off.

Protocol: the paper's Protocol and amendments section. Renders with recog_scene.Scene (worker processes),
features and scores in the main process (DINOv2 on MPS if available).

  uv run python scripts/recog_run.py                 # test layouts 1000-1009 -> results/recog_scores.parquet
  uv run python scripts/recog_run.py --seeds 0 1 --out /tmp/x.parquet   # training-seed smoke run
  uv run python scripts/recog_run.py --selfcheck     # tiny run on training seed 0 with asserts
Outputs: results/recog_scores.parquet, results/recog_renders/grey32.npz, results/recog_cost.json,
         results/recog_renders/contact_<seed>.png (first test layout).
"""
import argparse, json, os, sys, time
import multiprocessing as mp
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain.mb import MushroomBody, preprocess, equalise  # noqa: E402
from provenance import stamp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THETAS = list(range(0, 181, 15))
DISTS = [3.0, 6.0, 12.0]
N_SUN, N_REP = 2, 2
LF_THETAS = np.linspace(-45, 45, 12)
RECS = ["pixels", "colour", "mb", "dino_cls", "dino_patch", "mb_edge", "pixels_crop", "mb_crop"]  # last 3 exploratory
DINO_DIR = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")


def probe_plan(seed, thetas=THETAS, dists=DISTS, n_sun=N_SUN, n_rep=N_REP):
    """Jitter is drawn once per (sun, theta, dist, rep) and shared by the target and its swapped decoys."""
    rs = np.random.default_rng([seed, 12])
    plan = []
    for s in range(n_sun):
        for th in thetas:
            for d in dists:
                for k in range(n_rep):
                    plan.append(dict(sun=s, theta=th, dist=d, rep=k, th_j=float(th + rs.uniform(-4, 4)), d_j=float(d * rs.uniform(0.95, 1.05)),
                                     h=float(rs.uniform(1.5, 3.0)), gx=float(rs.uniform(-0.2, 0.2)), gy=float(rs.uniform(-0.2, 0.2))))
    return plan


def render_layout(args):
    seed, plan = args
    from recog_scene import Scene, OBJECTS
    sc = Scene(seed)
    sc.set_sun(0); sc.show("target")                                       # memory session: sun index 0 of layout
    mem = np.stack([sc.render(0, 6.0, 2.0)] + [sc.render(t, 6.0, 2.0) for t in LF_THETAS])
    imgs, meta = [], []
    for s in range(N_SUN):
        sc.set_sun(1 + s)
        for o in OBJECTS:
            sc.show(o)
            for p in plan:
                if p["sun"] != s:
                    continue
                imgs.append(sc.render(p["th_j"], p["d_j"], p["h"], (p["gx"], p["gy"])))
                meta.append({**p, "obj": o, "layout": seed, "yaw": sc.yaw})
    sc.close()
    return seed, mem, np.stack(imgs), meta


# ---------- recogniser front ends ----------
def colour_hist(imgs):
    b = (imgs.astype(np.int32) // 32)
    code = (b[..., 0] * 64 + b[..., 1] * 8 + b[..., 2]).reshape(len(imgs), -1)
    h = np.stack([np.bincount(c, minlength=512) for c in code]).astype(np.float32)
    return h / h.sum(1, keepdims=True)


def edge_input(imgs):
    """EXPLORATORY Roper-2017-like front end: 4 orientation maps of a 64x64 grey image, 4x4 pooled -> 1024."""
    g = np.stack([np.asarray(Image.fromarray(im).convert("L").resize((64, 64), Image.BOX), np.float32) / 255 for im in imgs])
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, :, 1:-1] = (g[:, :, 2:] - g[:, :, :-2]) / 2
    gy[:, 1:-1] = (g[:, 2:] - g[:, :-2]) / 2
    maps = [np.abs(np.cos(a) * gx + np.sin(a) * gy) for a in np.radians([0, 45, 90, 135])]
    m = np.stack(maps, 1).reshape(len(g), 4, 16, 4, 16, 4).mean((3, 5)).reshape(len(g), -1)
    return equalise(m)


def crop(imgs, dists, half_m=1.6):
    """EXPLORATORY post-hoc oracle object crop: square of half-width f*half_m/d px about the image centre -> 32x32 -> preprocess."""
    f = 112 / np.tan(np.radians(30))
    out = []
    for im, d in zip(imgs, dists):
        h = int(round(min(112, f * half_m / d)))
        out.append(np.asarray(Image.fromarray(im[112 - h:112 + h, 112 - h:112 + h]).resize((32, 32), Image.BOX)))
    return preprocess(np.stack(out))


class Dino:
    def __init__(self):
        import torch
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.m = torch.hub.load(DINO_DIR, "dinov2_vits14", source="local", verbose=False).eval().to(self.dev)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.n_params = sum(p.numel() for p in self.m.parameters())

    def __call__(self, imgs, bs=64):
        """-> cls [N, 384], patches [N, 256, 384], both L2-normalised (float32 numpy)."""
        t = self.torch
        cls, pat = [], []
        with t.no_grad():
            for i in range(0, len(imgs), bs):
                x = (t.from_numpy(imgs[i:i + bs]).permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std
                f = self.m.forward_features(x.to(self.dev))
                cls.append(t.nn.functional.normalize(f["x_norm_clstoken"], dim=-1).cpu().numpy())
                pat.append(t.nn.functional.normalize(f["x_norm_patchtokens"], dim=-1).cpu().numpy())
        return np.concatenate(cls), np.concatenate(pat)


def central_descriptor(pat):
    """Mean of the central 4x4 of the 16x16 patch grid, L2-normalised -> [N, 384]."""
    d = pat.reshape(len(pat), 16, 16, -1)[:, 6:10, 6:10].reshape(len(pat), 16, -1).mean(1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def score_layout(mem, probes, dino, mb, mbe, dists):
    """Return {rec: {lf: scores [N]}} for one layout. Memory row 0 = goal, rows 1..12 = learning flight."""
    out = {}
    views = {0: slice(0, 1), 1: slice(0, 13)}
    xm, xp = preprocess(mem), preprocess(probes)
    rms = np.sqrt(((xp[:, None] - xm[None]) ** 2).mean(-1))                # [N, 13]
    out["pixels"] = {lf: -rms[:, v].min(1) for lf, v in views.items()}
    hm, hp = colour_hist(mem), colour_hist(probes)
    inter = np.minimum(hp[:, None], hm[None]).sum(-1)
    out["colour"] = {lf: inter[:, v].max(1) for lf, v in views.items()}
    em, ep = edge_input(mem), edge_input(probes)
    cm_, cp_ = crop(mem, [6.0] * len(mem)), crop(probes, dists)
    rmsc = np.sqrt(((cp_[:, None] - cm_[None]) ** 2).mean(-1))
    out["pixels_crop"] = {lf: -rmsc[:, v].min(1) for lf, v in views.items()}
    for name, net, a, b in [("mb", mb, xm, xp), ("mb_edge", mbe, em, ep), ("mb_crop", mb, cm_, cp_)]:
        out[name] = {}
        for lf, v in views.items():
            net.reset(); net.learn(a[v])
            out[name][lf] = net.familiarity(b)
    cm, pm = dino(mem)
    cp, pp = dino(probes)
    cos = cp @ cm.T
    out["dino_cls"] = {lf: cos[:, v].max(1) for lf, v in views.items()}
    dm = central_descriptor(pm)                                             # [13, 384]
    pmax = np.einsum("npc,mc->npm", pp, dm).max(1)                          # [N, 13]
    out["dino_patch"] = {lf: pmax[:, v].max(1) for lf, v in views.items()}
    return out, xp


def bytes_table(n_params):
    rows = {}
    for lf, nv in [(0, 1), (1, 13)]:
        rows[lf] = {"pixels": nv * 1024, "colour": nv * 512 * 2, "mb": 2500, "mb_edge": 2500, "pixels_crop": nv * 1024, "mb_crop": 2500,
                    "dino_cls": nv * 384 * 2, "dino_patch": nv * 384 * 2}
    return {"stored_goal_bytes": {"lf_off": rows[0], "lf_on": rows[1]},
            "model_bytes": {"pixels": 0, "colour": 0, "mb": 0, "mb_edge": 0, "pixels_crop": 0, "mb_crop": 0, "dino_cls": n_params * 4, "dino_patch": n_params * 4},
            "notes": "MB: 20,000 1-bit KC->MBON weights; PN->KC wiring regenerated from seed (400,000 B if stored as uint16). "
                     "pixels uint8 32x32 per view; colour uint16 512 bins per view; DINO fp16 384-d per view, weights fp32."}


def time_queries(mem, probe, dino, mb, mbe, reps=50):
    """Median ms for one query against the 13-view (LF on) memory, excluding rendering."""
    import torch
    xm, hm, em = preprocess(mem), colour_hist(mem), edge_input(mem)
    mb.reset(); mb.learn(xm); mbe.reset(); mbe.learn(em)
    cm, pm = dino(mem); dm = central_descriptor(pm)
    q = probe[None]

    def sync():
        if dino.dev == "mps":
            torch.mps.synchronize()
    fns = {"pixels": lambda: np.sqrt(((preprocess(q)[:, None] - xm[None]) ** 2).mean(-1)).min(),
           "colour": lambda: np.minimum(colour_hist(q)[:, None], hm[None]).sum(-1).max(),
           "mb": lambda: mb.familiarity(preprocess(q)),
           "mb_edge": lambda: mbe.familiarity(edge_input(q)),
           "dino_cls": lambda: (dino(q)[0] @ cm.T).max(),
           "dino_patch": lambda: np.einsum("npc,mc->npm", dino(q)[1], dm).max()}
    out = {}
    for k, f in fns.items():
        f(); sync()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); f(); sync(); ts.append((time.perf_counter() - t0) * 1e3)
        out[k] = float(np.median(ts))
    return out


def run(seeds, out_path, grey_path=None, cost_path=None, contact=True, plan_kw=None, procs=4):
    plan_kw = plan_kw or {}
    t0 = time.time()
    dino = Dino()
    mb, mbe = MushroomBody(1024, seed=0), MushroomBody(1024, seed=0)
    rows, greys, timing = [], {}, {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(procs, len(seeds))) as pool:
        for seed, mem, probes, meta in pool.imap(render_layout, [(s, probe_plan(s, **plan_kw)) for s in seeds]):
            t1 = time.time()
            sc, xp = score_layout(mem, probes, dino, mb, mbe, [m["d_j"] for m in meta])
            for rec, d in sc.items():
                for lf, s in d.items():
                    for m, v in zip(meta, s):
                        rows.append({**m, "rec": rec, "lf": lf, "score": float(v)})
            greys[f"L{seed}_probes"] = xp.astype(np.float16); greys[f"L{seed}_mem"] = preprocess(mem).astype(np.float16)
            if contact and seed == seeds[0]:
                idx = [i for i, m in enumerate(meta) if m["sun"] == 0 and m["rep"] == 0 and m["dist"] == 6.0 and m["theta"] in (0, 45, 90, 180)]
                grid = probes[idx].reshape(4, 4, 224, 224, 3)                # [obj, theta]
                sheet = np.concatenate([np.concatenate(list(r), 1) for r in grid], 0)
                Image.fromarray(np.concatenate([np.concatenate([mem[0], mem[1], mem[6], mem[12]], 1), sheet], 0)).resize((448, 640)).save(
                    os.path.join(ROOT, "results", "recog_renders", f"contact_{seed}.png"))
            timing[seed] = time.time() - t1
            print(f"layout {seed}: {len(probes)} probes, score {timing[seed]:.1f}s, elapsed {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    st = stamp(seed=list(map(int, seeds)), param_source="H4 protocol (pre-registered; paper, Protocol and amendments); scripts/recog_scene.py; scripts/brain/mb.py",
               dino=f"facebookresearch/dinov2 dinov2_vits14 (torch hub, Apache-2.0), {dino.n_params} params, device {dino.dev}")
    df.attrs["provenance"] = json.dumps(st)
    import pyarrow as pa, pyarrow.parquet as pq
    tbl = pa.Table.from_pandas(df)
    pq.write_table(tbl.replace_schema_metadata({**(tbl.schema.metadata or {}), b"provenance": json.dumps(st).encode()}), out_path)
    if grey_path:
        np.savez_compressed(grey_path, **greys)
    if cost_path:
        ms = time_queries(mem, probes[0], dino, mb, mbe)
        json.dump({**stamp(seed=int(seeds[-1]), param_source="scripts/recog_run.py time_queries"), "ms_per_query_median": ms,
                   "ms_note": f"one query vs the 13-view memory, M4, numpy CPU for pixels/colour/MB, DINOv2 on {dino.dev}, excl. rendering",
                   **bytes_table(dino.n_params), "wall_s_total": time.time() - t0}, open(cost_path, "w"), indent=1)
    print(f"done: {len(df)} rows in {time.time() - t0:.0f}s -> {out_path}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1000, 1010)))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "recog_scores.parquet"))
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        p = os.path.join(ROOT, "results", "recog_renders", "selfcheck_train.parquet")
        df = run([0], p, contact=False, plan_kw=dict(thetas=[0, 90], dists=[6.0], n_rep=1))
        assert len(df) == 2 * 1 * 2 * 4 * len(RECS) * 2                      # theta x dist x sun x obj x rec x lf
        assert set(df.rec) == set(RECS) and df.score.notna().all()
        assert not df.duplicated(["obj", "theta", "sun", "rec", "lf"]).any()
        on, off = (df[(df.rec == "mb") & (df.lf == k)].set_index(["obj", "theta", "sun"]).score for k in (1, 0))
        assert (on >= off - 1e-9).all()                                    # learning more views never lowers MB familiarity
        os.remove(p)
        print("recog_run self-check ok")
    else:
        run(a.seeds, a.out, os.path.join(ROOT, "results", "recog_renders", "grey32.npz"), os.path.join(ROOT, "results", "recog_cost.json"))
