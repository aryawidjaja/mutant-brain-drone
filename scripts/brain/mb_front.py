"""Insect-style visual front end for the mushroom body (H8), plus the inspection stop rule.

front(img) turns an object-centred RGB crop into a vector for the unchanged MushroomBody (scripts/brain/mb.py):

  1. Receptors: log(channel + EPS) per RGB channel (photoreceptor-like compression; makes a global light scale
     an additive offset). RGB has no UV channel, so the bee's UV/blue/green triplet is APPROXIMATED:
     luminance = mean of the three log channels; opponent o_bg = log B - log G ("blue-green"),
     o_rg = log R - log G (a long-wave stand-in for the missing UV channel). Log ratios are unchanged by any
     global scaling of the light.
  2. Local contrast normalisation (lamina-like lateral inhibition): c = lum - box(lum), n = c / sqrt(box(c^2) + LCN_EPS).
     box = (2r+1)^2 mean with edge-clamped borders.
  3. Oriented edges (lobula-like orientation-sensitive units, Roper et al. 2017): |cos(a) gx + sin(a) gy| of n at
     a = 0, 45, 90, 135 deg (polarity-free, so a sun that flips which face is bright keeps the edge).
  4. Coarse retinotopic pooling: mean of every channel (4 edge + 2 opponent [+ 2 opponent-edge]) over a
     grid x grid tiling of the crop (cell edges floor(i*H/grid)).
  5. Rank equalisation (stable, ties by index) -> rank / n_in in (0, 1], over the whole vector or per channel.

Fixed filters, no learned parameters (0 bytes). Twin: app/sim/brain/mb_front.js (asserted <= 1e-9 on recorded crops).

inspect_ok(view_az, fam, valid, thr, sep_deg) is the inspection stop rule (pure function, batched).
"""
import numpy as np

EPS = 0.02          # receptor offset (grey units 0..1)
LCN_EPS = 1e-4
ANGLES = np.radians([0.0, 45.0, 90.0, 135.0])
CFG = dict(grid=6, r=2, colour_edges=True, per_channel=True, half_m=1.1)   # frozen by the training-layout design search


def box(a, r):
    """(2r+1)^2 box mean over the last two axes, edge-clamped. Cumulative sums (same order in the JS twin)."""
    H, W = a.shape[-2:]
    out = a
    for ax, n in ((-1, W), (-2, H)):
        p = np.concatenate([np.repeat(np.take(out, [0], ax), r + 1, ax), out, np.repeat(np.take(out, [n - 1], ax), r, ax)], ax)
        cs = np.cumsum(p, ax)
        hi = np.take(cs, np.arange(2 * r + 1, 2 * r + 1 + n), ax)
        lo = np.take(cs, np.arange(0, n), ax)
        out = (hi - lo) / (2 * r + 1)
    return out


def grad(a):
    """Central differences with edge-clamped neighbours -> gx, gy (per pixel)."""
    xp = np.concatenate([a[..., :, 1:], a[..., :, -1:]], -1); xm = np.concatenate([a[..., :, :1], a[..., :, :-1]], -1)
    yp = np.concatenate([a[..., 1:, :], a[..., -1:, :]], -2); ym = np.concatenate([a[..., :1, :], a[..., :-1, :]], -2)
    return (xp - xm) / 2, (yp - ym) / 2


def features(img, grid=6, r=4, colour_edges=False, **_):
    """img [..., H, W, 3] (uint8 0..255 or float 0..1) -> pooled raw features [..., C*grid*grid] (channel-major)."""
    a = np.asarray(img)
    a = a / 255.0 if a.dtype == np.uint8 else a.astype(np.float64)
    lg = np.log(a + EPS)
    lum = (lg[..., 0] + lg[..., 1] + lg[..., 2]) / 3
    o_bg, o_rg = lg[..., 2] - lg[..., 1], lg[..., 0] - lg[..., 1]
    c = lum - box(lum, r)
    n = c / np.sqrt(box(c * c, r) + LCN_EPS)
    gx, gy = grad(n)
    ch = [np.abs(np.cos(t) * gx + np.sin(t) * gy) for t in ANGLES] + [o_bg, o_rg]
    if colour_edges:
        for o in (o_bg, o_rg):
            ox, oy = grad(o)
            ch.append(np.sqrt(ox * ox + oy * oy))
    ch = np.stack(ch, -3)                                                  # [..., C, H, W]
    H, W = ch.shape[-2:]
    ey, ex = [H * i // grid for i in range(grid + 1)], [W * i // grid for i in range(grid + 1)]
    cells = [ch[..., ey[i]:ey[i + 1], ex[j]:ex[j + 1]].mean((-2, -1)) for i in range(grid) for j in range(grid)]
    return np.stack(cells, -1).reshape(*ch.shape[:-3], -1)                # [..., C * grid^2]


def rank01(f):
    """Stable rank along the last axis -> rank / n in (0, 1]."""
    o = np.argsort(f, axis=-1, kind="stable")
    r = np.empty_like(o)
    np.put_along_axis(r, o, np.broadcast_to(np.arange(1, f.shape[-1] + 1), o.shape), -1)
    return r / f.shape[-1]


def front(img, **cfg):
    """img [..., H, W, 3] object-centred crop -> x [..., n_in] in (0, 1] for MushroomBody (n_in = C * grid^2)."""
    cfg = {**CFG, **cfg}
    f = features(img, **cfg)
    if not cfg["per_channel"]:
        return rank01(f)
    g2 = cfg["grid"] ** 2
    return rank01(f.reshape(*f.shape[:-1], -1, g2)).reshape(f.shape)


def n_in(**cfg):
    cfg = {**CFG, **cfg}
    return (8 if cfg["colour_edges"] else 6) * cfg["grid"] ** 2


def crop_box(dist_m, f_px, half_m=1.6):
    """Half-width (px) of the object-centred window for an object at range dist_m (scale-normalised attention)."""
    return f_px * half_m / np.asarray(dist_m, float)


def inspect_ok(view_az, fam, valid, thr, sep_deg=20.0):
    """Inspection stop rule: accept only if at least two valid samples with familiarity >= thr were taken from
    viewpoints at least sep_deg apart around the object.

    view_az [..., T] rad: direction from the object to the eye (world or compass frame; only differences matter).
    fam [..., T] familiarity or evidence; valid [..., T] bool (sample exists, object visible, in inspection range).
    -> bool [...]. inspect_stat() gives the threshold-free statistic (accept iff stat >= thr).
    """
    return inspect_stat(view_az, fam, valid, sep_deg) >= thr


def inspect_stat(view_az, fam, valid, sep_deg=20.0):
    """max over sample pairs (i, j) with circular separation >= sep_deg of min(fam_i, fam_j); -inf if no such pair."""
    az, f, v = np.asarray(view_az, float), np.asarray(fam, float), np.asarray(valid, bool)
    d = np.abs((az[..., :, None] - az[..., None, :] + np.pi) % (2 * np.pi) - np.pi)
    ok = (d >= np.radians(sep_deg) - 1e-12) & v[..., :, None] & v[..., None, :]
    m = np.minimum(f[..., :, None], f[..., None, :])
    return np.where(ok, m, -np.inf).max((-2, -1))


if __name__ == "__main__":
    import json, os, subprocess, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mb import MushroomBody
    rs = np.random.default_rng(0)
    # 1. light invariance: a global light scale (in the receptor-linear domain, away from EPS) barely moves the vector
    img = rs.uniform(0.55, 0.6, (48, 48, 3))
    yy, xx = np.mgrid[0:48, 0:48]
    img[(xx // 6) % 2 == 0] *= [1.0, 0.5, 0.3]                          # vertical stripes, orange-ish
    x1, x2 = front(img), front(img * 0.6)
    assert x1.shape == (n_in(),) and x1.min() > 0 and x1.max() == 1.0
    assert np.corrcoef(x1, x2)[0, 1] > 0.9, np.corrcoef(x1, x2)[0, 1]
    # 2. orientation selectivity: vertical stripes drive the 0-deg (x-gradient) channel, not the 90-deg one
    f = features(img).reshape(6, 36)
    assert f[0].mean() > 2 * f[2].mean(), (f[0].mean(), f[2].mean())
    # 3. per-channel ranks are a permutation of 1..36 inside each channel
    xp = front(img, colour_edges=False, per_channel=True).reshape(6, 36) * 36
    assert all(sorted(np.rint(r).astype(int)) == list(range(1, 37)) for r in xp)
    # 4. batch == single, MB familiarity: learned view 1, rotated stripes lower
    B = np.stack([img, img.transpose(1, 0, 2), rs.uniform(0, 1, (48, 48, 3))])
    X = front(B)
    assert np.array_equal(X[0], x1)
    mb = MushroomBody(n_in(), seed=0); mb.learn(X[0])
    fam = mb.familiarity(X)
    assert fam[0] == 1.0 and fam[1] < 0.8 and fam[2] < 0.8, fam
    # 5. box filter equals a direct window mean with clamped indices
    a = rs.normal(size=(7, 9))
    ref = np.array([[a[np.clip(np.arange(i - 2, i + 3), 0, 6)][:, np.clip(np.arange(j - 2, j + 3), 0, 8)].mean() for j in range(9)] for i in range(7)])
    assert np.allclose(box(a, 2), ref, atol=1e-12)
    # 6. inspection rule
    az = np.radians([[0, 5, 10, 25], [0, 5, 10, 25], [0, 5, 10, 25]])
    fm = np.array([[0.9, 0.9, 0.9, 0.2], [0.9, 0.2, 0.2, 0.8], [0.9, 0.9, 0.9, 0.9]])
    vd = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0]], bool)
    assert inspect_ok(az, fm, vd, 0.7).tolist() == [False, True, False]   # needs two familiar views >= 20 deg apart
    assert np.allclose(inspect_stat(az, fm, vd), [0.2, 0.8, -np.inf])
    assert inspect_ok(np.radians([[350, 15]]), np.array([[0.8, 0.8]]), np.ones((1, 2), bool), 0.7, 20)[0]   # wraps
    # 7. JS twin agreement (synthetic + recorded crops if the render cache exists) -> results/recog_v2_parity.json
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    crops = [np.clip(rs.uniform(0, 1, (48, 48, 3)) * 255, 0, 255).astype(np.uint8), (img * 255).astype(np.uint8)]
    src = "synthetic"
    cache = os.path.join(ROOT, "results", "recog_renders", "v2_cache", f"h{CFG['half_m']:g}", "train_0.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        crops += list(z["crop"][::157][:8]) + list(z["mem_crop_ms"][:2])
        src = "synthetic x2 + recorded MuJoCo crops (training layout 0, probes every 157th, 2 memory views)"
    cfgs = [CFG, {**CFG, "colour_edges": False, "per_channel": False, "grid": 4, "r": 4}, {**CFG, "grid": 8, "r": 3}]
    js = ("import { front, features, inspectStat } from '%s';\nimport { readFileSync } from 'fs';\n"
          "const inp = JSON.parse(readFileSync(0, 'utf8'));\nconst out = {feat: [], x: [], stat: []};\n"
          "for (const c of inp.cfgs) for (const im of inp.crops) { const u = Uint8Array.from(im);\n"
          "  out.feat.push(Array.from(features(u, 48, 48, c))); out.x.push(Array.from(front(u, 48, 48, c))); }\n"
          "for (const s of inp.ins) out.stat.push(inspectStat(s.az, s.f, s.v, 20));\n"
          "process.stdout.write(JSON.stringify(out));\n") % os.path.join(ROOT, "app", "sim", "brain", "mb_front.js")
    ins = [dict(az=az[i].tolist(), f=fm[i].tolist(), v=vd[i].tolist()) for i in range(3)]
    res = subprocess.run(["node", "--input-type=module", "-e", js], input=json.dumps(
        {"crops": [c.reshape(-1).tolist() for c in crops], "cfgs": [{k: v for k, v in c.items()} for c in cfgs], "ins": ins}),
        capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    o = json.loads(res.stdout)
    pf = [features(c, **cf) for cf in cfgs for c in crops]
    px = [front(c, **cf) for cf in cfgs for c in crops]
    df = max(float(np.abs(np.array(a) - b).max()) for a, b in zip(o["feat"], pf))
    dx = max(float(np.abs(np.array(a) - b).max()) for a, b in zip(o["x"], px))
    st = np.array([s if s is not None else -np.inf for s in o["stat"]])
    assert df <= 1e-9 and dx <= 1e-9 and np.array_equal(st, inspect_stat(az, fm, vd)), (df, dx, st)
    if src != "synthetic":
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from provenance import stamp
        json.dump({**stamp(seed=0, param_source="scripts/brain/mb_front.py self-check; app/sim/brain/mb_front.js"),
                   "crops": src, "n_crops": len(crops), "configs": cfgs, "max_abs_diff_features": df, "max_abs_diff_x": dx,
                   "gate": 1e-9, "pass": True}, open(os.path.join(ROOT, "results", "recog_v2_parity.json"), "w"), indent=1)
    print(f"mb_front self-check ok; n_in {n_in()}; JS twin max |diff| features {df:.2e}, x {dx:.2e} on {len(crops)} crops x {len(cfgs)} configs ({src})")
