"""The drone's eye: an analytic ray-cast panorama plus per-object "looks" for the mushroom body (LOG Amendment 1).

Identical equations in app/sim/brain/eye.js. Angles: world frame, radians, counter-clockwise from +x (east).

Panorama (for the UI eye panel and replays): 72 x 12 px, 5 deg/px, 360 deg azimuth, elevation +20..-40 deg,
  one ray per pixel. Column c looks at body-relative azimuth 180 - (c + 0.5) * 5 deg (image left = drone's left),
  row r at elevation 20 - (r + 0.5) * 5 deg. Renders sky (gradient + sun disc; a false sun when that fault is on),
  ground, obstacles (vertical cylinders, axis-aligned boxes) and objects (yawed boxes with their texture),
  Lambert shading from the sun, light level and exponential fog.
Object look (MB input): for each object a 32 x 24 sub-pixel window centred on the object, angular size
  2 atan(0.8 D / range) (D = object diagonal), so the object fills the window at any range (attention window);
  blurred by the eye's acuity (Gaussian, sigma = ACUITY_DEG) so far objects lose fine pattern detail; area
  down-sampled to 16 x 12, grey, pixel noise sigma NOISE / light (Rng stream 4), then mb.equalise (rank + L2).
  Objects and ground/sky only: occlusion by obstacles is handled by the visibility test, not drawn in the look.
Visibility of object k: present (t >= t_s), within detection range (angular size >= DETECT_DEG and fog contrast
  >= 0.2), at least 2 of 3 sight lines (centre, both ends) clear of obstacles, not in a dead eye column, and not
  hidden by the occlusion fault.
Depth ring (for optic flow): 72 horizontal rays at the drone's altitude against obstacles, range <= 20 m.
  Geometric optic flow is computed from these ranges plus noise (avoid.py): "geometric optic flow from ray-casts
  plus noise", not a model of motion detectors.
"""
import math
import os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rng import Rng  # noqa: E402

NC, NR, PX_DEG, EL_TOP = 72, 12, 5.0, 20.0
LW, LH, SS = 16, 12, 2                 # look size (px) and supersampling per axis
ACUITY_DEG = 2.0                       # chosen eye acuity (blur sigma); a model parameter, not a measured value
DETECT_DEG = 4.0                       # minimum angular size to register an object
NOISE = 0.03                           # pixel noise sigma at light 1 (grey 0..1 units)
DEPTH_MAX = 20.0
FAR = 1e9
PATTERNS = {"stripes": 0, "chevron": 1, "dots": 2}
DOOR_RGB = (0.12, 0.10, 0.08)
FOG_RGB = (0.80, 0.82, 0.85)


def col_az():
    return np.radians(180.0 - (np.arange(NC) + 0.5) * PX_DEG)


def row_el():
    return np.radians(EL_TOP - (np.arange(NR) + 0.5) * PX_DEG)


def uniforms(rng, n):
    """n sequential draws per batch row from an Rng (mulberry32 state is a counter), advancing it. -> [B, n]."""
    k = np.arange(n, dtype=np.uint64)
    a0 = rng.a.copy()
    t = Rng([0], 0)
    t.a = ((a0[:, None] + k[None, :] * np.uint64(0x6D2B79F5)) & np.uint64(0xFFFFFFFF)).reshape(-1)
    u = t.uniform().reshape(len(a0), n)
    rng.a = (a0 + np.uint64(n) * np.uint64(0x6D2B79F5)) & np.uint64(0xFFFFFFFF)
    return u


def normals(rng, n):
    """n sequential Rng.normal() draws per row (each consumes two uniforms, as rng.normal does). -> [B, n]."""
    u = uniforms(rng, 2 * n)
    return np.sqrt(-2.0 * np.log(1.0 - u[:, 0::2])) * np.cos(2.0 * np.pi * u[:, 1::2])


class Scene:
    """Batched world arrays (padded). worlds: list of C5 dicts (one per batch row)."""

    def __init__(self, worlds):
        B = len(worlds)
        cyl = [[o for o in w["obstacles"] if o["type"] != "building"] for w in worlds]
        box = [[o for o in w["obstacles"] if o["type"] == "building"] for w in worlds]
        Mc, Mb, K = max(1, max(map(len, cyl))), max(1, max(map(len, box))), max(1, max(len(w["objects"]) for w in worlds))
        self.B, self.K = B, K
        self.cyl = np.zeros((B, Mc, 7)); self.cyl[:, :, 0] = FAR   # x y r h r g b
        self.box = np.zeros((B, Mb, 8)); self.box[:, :, 0] = FAR   # x y hw hd h r g b
        self.obj = np.zeros((B, K, 17)); self.obj[:, :, 0] = FAR; self.obj[:, :, 4:7] = 1.0; self.obj[:, :, 15] = 1.0   # x y z yaw hw hd h base3 accent3 is_target pattern period t_s
        self.kvalid = np.zeros((B, K), bool)
        self.role = np.zeros((B, K), int)                           # 1 target, 0 decoy
        for b, w in enumerate(worlds):
            for i, o in enumerate(cyl[b]):
                self.cyl[b, i] = [o["x"], o["y"], o["r"], o["h"], *o["rgb"]]
            for i, o in enumerate(box[b]):
                self.box[b, i] = [o["x"], o["y"], o["w"] / 2, o["d"] / 2, o["h"], *o["rgb"]]
            for o in w["objects"]:
                k, a = o["slot"], o["appearance"]
                self.obj[b, k] = [o["x"], o["y"], o["z"], math.radians(o["yaw_deg"]), o["dims"][0] / 2, o["dims"][1] / 2,
                                  o["dims"][2], *a["base"], *a["accent"], float(o["role"] == "target"),
                                  PATTERNS[a["pattern"]], a["period_m"], o["t_s"]]
                self.kvalid[b, k] = True
                self.role[b, k] = o["role"] == "target"
        self.ground = np.array([w["ground"]["rgb"] for w in worlds])
        az = np.radians([w["sun"]["azimuth_deg"] for w in worlds]); el = np.radians([w["sun"]["elevation_deg"] for w in worlds])
        self.sun_az, self.sun_el = az, el
        self.sun = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)
        self.light = np.array([w["light"] for w in worlds], float)
        self.fog = np.array([w["fog_m"] if w["fog_m"] else 0.0 for w in worlds], float)   # 0 = no fog


# ---------------------------------------------------------------- ray primitives (o [B,R,3] or broadcastable, d [B,R,3])
def hit_cyl(o, d, cyl):
    """-> t [B,R], idx [B,R], normal [B,R,3]. Vertical capped cylinders standing on z = 0."""
    ox, oy, oz = (o[..., i][..., None] for i in range(3))
    dx, dy, dz = (d[..., i][..., None] for i in range(3))
    cx, cy, r, h = (cyl[:, None, :, i] for i in range(4))
    px, py = ox - cx, oy - cy
    a = dx * dx + dy * dy
    bq = px * dx + py * dy
    c = px * px + py * py - r * r
    disc = bq * bq - a * c
    sq = np.sqrt(np.maximum(disc, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ts = (-bq - sq) / a
        zs = oz + ts * dz
        side = (disc > 0) & (a > 1e-12) & (ts > 1e-6) & (zs >= 0) & (zs <= h)
        tt = (h - oz) / dz
        qx, qy = px + tt * dx, py + tt * dy
        top = (dz < 0) & (oz > h) & (tt > 1e-6) & (qx * qx + qy * qy <= r * r)
    t = np.where(side, ts, np.where(top, tt, FAR))
    i = np.argmin(t, -1)
    tb = np.take_along_axis(t, i[..., None], -1)[..., 0]
    istop = np.take_along_axis(top & ~side, i[..., None], -1)[..., 0]
    hx = o[..., 0] + tb * d[..., 0] - np.take_along_axis(np.broadcast_to(cx, t.shape), i[..., None], -1)[..., 0]
    hy = o[..., 1] + tb * d[..., 1] - np.take_along_axis(np.broadcast_to(cy, t.shape), i[..., None], -1)[..., 0]
    nr = np.sqrt(hx * hx + hy * hy) + 1e-12
    n = np.where(istop[..., None], np.array([0.0, 0.0, 1.0]), np.stack([hx / nr, hy / nr, np.zeros_like(hx)], -1))
    return tb, i, n


def _slab(o, d, lo, hi):
    """AABB slab test, broadcast. -> t_entry, entry axis (0,1,2), valid."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
        t1, t2 = (lo - o) * inv, (hi - o) * inv
    tn, tf = np.minimum(t1, t2), np.maximum(t1, t2)
    tnear, tfar = tn.max(-1), tf.min(-1)
    axis = tn.argmax(-1)
    ok = (tnear <= tfar) & (tnear > 1e-6)
    return tnear, axis, ok


def hit_box(o, d, box):
    """Axis-aligned boxes on z = 0. -> t [B,R], idx, normal."""
    O, D = o[..., None, :], d[..., None, :]
    c = box[:, None, :, :]
    lo = np.stack([c[..., 0] - c[..., 2], c[..., 1] - c[..., 3], np.zeros_like(c[..., 0])], -1)
    hi = np.stack([c[..., 0] + c[..., 2], c[..., 1] + c[..., 3], c[..., 4]], -1)
    tn, ax, ok = _slab(O, D, lo, hi)
    t = np.where(ok, tn, FAR)
    i = np.argmin(t, -1)
    tb = np.take_along_axis(t, i[..., None], -1)[..., 0]
    a = np.take_along_axis(ax, i[..., None], -1)[..., 0]
    sgn = np.broadcast_to(-np.sign(d), tb.shape + (3,))
    n = np.zeros(tb.shape + (3,))
    for k in range(3):
        n[..., k] = np.where(a == k, sgn[..., k], 0.0)
    return tb, i, n


def hit_obj(o, d, obj, t_now):
    """Yawed textured boxes. obj [B,K,17]. -> t [B,R], idx, normal (world), rgb [B,R,3]."""
    O, D = o[..., None, :], d[..., None, :]                    # [B,R,1,3]
    c = obj[:, None, :, :]                                     # [B,1,K,17]
    cs, sn = np.cos(c[..., 3]), np.sin(c[..., 3])
    rx, ry = O[..., 0] - c[..., 0], O[..., 1] - c[..., 1]
    lo_ = np.stack([cs * rx + sn * ry, -sn * rx + cs * ry, O[..., 2] - c[..., 2]], -1)      # ray origin in object frame
    ld = np.stack([cs * D[..., 0] + sn * D[..., 1], -sn * D[..., 0] + cs * D[..., 1], D[..., 2] + 0.0 * cs], -1)
    lo_b = np.stack([-c[..., 4], -c[..., 5], np.zeros_like(c[..., 4])], -1)
    hi_b = np.stack([c[..., 4], c[..., 5], c[..., 6]], -1)
    tn, ax, ok = _slab(lo_, ld, lo_b, hi_b)
    present = np.broadcast_to((c[..., 16] <= t_now) & (c[..., 0] < FAR / 2), tn.shape)
    t = np.where(ok & present, tn, FAR)
    i = np.argmin(t, -1)
    g = lambda x: np.take_along_axis(np.broadcast_to(x, t.shape), i[..., None], -1)[..., 0]  # noqa: E731
    tb, a = g(t), g(ax)
    P = [g(lo_[..., j]) + tb * g(ld[..., j]) for j in range(3)]  # local hit point
    dl = [g(ld[..., j]) for j in range(3)]
    hw, hd, hh = g(c[..., 4]), g(c[..., 5]), g(c[..., 6])
    u = np.where(a == 0, P[1], P[0])
    v = np.where(a == 2, P[1], P[2])
    per = g(c[..., 15])
    pat = g(c[..., 14])
    fu, fv = u / per - np.floor(u / per), v / per - np.floor(v / per)
    fc = (np.abs(u) + v) / per
    on = np.where(pat == 0, fu < 0.5, np.where(pat == 1, fc - np.floor(fc) < 0.5, (fu - 0.5) ** 2 + (fv - 0.5) ** 2 < 0.09))
    base = np.stack([g(c[..., 7 + j]) for j in range(3)], -1)
    acc = np.stack([g(c[..., 10 + j]) for j in range(3)], -1)
    rgb = np.where(on[..., None], acc, base)
    front = (a == 0) & (dl[0] < 0) & (np.abs(P[1]) < 0.2 * hd * 2) & (P[2] < 0.75 * hh)   # door on the +x face
    rgb = np.where(front[..., None], np.array(DOOR_RGB), rgb)
    sgn = -np.sign(np.stack(dl, -1))
    nl = np.stack([np.where(a == k, sgn[..., k], 0.0) for k in range(3)], -1)
    yaw = g(c[..., 3])
    cy, sy = np.cos(yaw), np.sin(yaw)
    n = np.stack([cy * nl[..., 0] - sy * nl[..., 1], sy * nl[..., 0] + cy * nl[..., 1], nl[..., 2]], -1)
    return tb, i, n, rgb


def shade(sc, o, d, t_now, objects=True, obstacles=True, false_sun=None):
    """Full colour ray-cast. o [B,1,3] or [B,R,3], d [B,R,3] unit. -> rgb [B,R,3] (0..1), dist [B,R]."""
    B, R = d.shape[:2]
    lam = lambda n: 0.45 + 0.55 * np.maximum(0.0, (n * sc.sun[:, None, :]).sum(-1))  # noqa: E731
    dz = d[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.where(dz < -1e-9, -o[..., 2] / dz, FAR)
    tg = np.broadcast_to(tg, (B, R))
    best = tg.copy()
    rgb = np.broadcast_to((sc.ground * (0.45 + 0.55 * np.sin(sc.sun_el))[:, None])[:, None, :], (B, R, 3)).copy()
    if obstacles:
        tc, ic, nc = hit_cyl(o, d, sc.cyl)
        colc = np.take_along_axis(sc.cyl[:, :, 4:7], ic.reshape(B, -1, 1).repeat(3, -1), 1).reshape(B, R, 3)
        m = tc < best
        rgb = np.where(m[..., None], colc * lam(nc)[..., None], rgb); best = np.where(m, tc, best)
        tb, ib, nb = hit_box(o, d, sc.box)
        colb = np.take_along_axis(sc.box[:, :, 5:8], ib.reshape(B, -1, 1).repeat(3, -1), 1).reshape(B, R, 3)
        m = tb < best
        rgb = np.where(m[..., None], colb * lam(nb)[..., None], rgb); best = np.where(m, tb, best)
    if objects:
        to, io, no, colo = hit_obj(o, d, sc.obj, t_now)
        m = to < best
        rgb = np.where(m[..., None], colo * lam(no)[..., None], rgb); best = np.where(m, to, best)
    # sky
    el = np.arcsin(np.clip(dz, -1, 1))
    hz, zen = np.array([0.75, 0.85, 0.95]), np.array([0.35, 0.55, 0.90])
    sky = hz + (zen - hz) * np.clip(el / (np.pi / 2), 0, 1)[..., None]
    cosang = (d * sc.sun[:, None, :]).sum(-1)
    sky = sky + np.where(cosang > np.cos(np.radians(6.0)), 1.0, 0.0)[..., None] * (np.array([1.0, 1.0, 0.9]) - sky)
    if false_sun is not None:   # [B] azimuth (rad) of a false sun at the true sun's elevation, nan = none
        fs = np.stack([np.cos(sc.sun_el) * np.cos(false_sun), np.cos(sc.sun_el) * np.sin(false_sun), np.sin(sc.sun_el)], -1)
        cf = (d * fs[:, None, :]).sum(-1)
        sky = sky + np.where(np.isfinite(false_sun)[:, None] & (cf > np.cos(np.radians(6.0))), 1.0, 0.0)[..., None] * (np.array([1.0, 1.0, 0.9]) - sky)
    hit = best < FAR / 2
    rgb = np.where(hit[..., None], rgb, sky)
    fogf = np.where((sc.fog[:, None] > 0) & hit, np.exp(-best / np.where(sc.fog > 0, sc.fog, 1.0)[:, None]), 1.0)
    rgb = rgb * fogf[..., None] + np.array(FOG_RGB) * (1 - fogf[..., None])
    return np.clip(rgb * sc.light[:, None, None], 0, 1), np.where(hit, best, FAR)


def panorama(sc, pos, yaw, t_now, dead=None, false_sun=None):
    """pos [B,3], yaw [B] -> uint8 [B, NR, NC, 3]. Dead columns (bool [B, NC]) render black."""
    az = yaw[:, None, None] + col_az()[None, None, :]
    el = np.broadcast_to(row_el()[None, :, None], (len(yaw), NR, NC))
    d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1).reshape(len(yaw), NR * NC, 3)
    rgb, _ = shade(sc, pos[:, None, :], d, t_now, false_sun=false_sun)
    img = np.floor(rgb.reshape(len(yaw), NR, NC, 3) * 255 + 0.5).astype(np.uint8)
    if dead is not None:
        img[np.broadcast_to(dead[:, None, :], img.shape[:3])] = 0
    return img


def sub_scene(sc, bi, ki):
    """One-object scene per (row bi[p], slot ki[p]) pair: rows = pairs, K = 1 (for rendering only the looks needed)."""
    sub = Scene.__new__(Scene)
    sub.B, sub.K = len(bi), 1
    sub.obj = sc.obj[bi, ki][:, None, :]
    sub.kvalid, sub.role = sc.kvalid[bi, ki][:, None], sc.role[bi, ki][:, None]
    sub.cyl, sub.box = sc.cyl[bi], sc.box[bi]
    for k in ("ground", "sun_az", "sun_el", "sun", "light", "fog"):
        setattr(sub, k, getattr(sc, k)[bi])
    return sub


def obj_centre(sc):
    return np.stack([sc.obj[..., 0], sc.obj[..., 1], sc.obj[..., 2] + sc.obj[..., 6] / 2], -1)   # [B,K,3]


def obj_diag(sc):
    return 2 * np.sqrt(sc.obj[..., 4] ** 2 + sc.obj[..., 5] ** 2 + (sc.obj[..., 6] / 2) ** 2)


def _blur_mats(sigma, n):
    """[..., n, n] row-normalised Gaussian blur matrices (sigma in sub-pixels)."""
    x = np.arange(n, dtype=float)
    s = np.maximum(sigma, 1e-3)[..., None, None]
    g = np.exp(-0.5 * ((x[:, None] - x[None, :]) / s) ** 2)
    return g / g.sum(-1, keepdims=True)


def looks(sc, eye_pos, t_now, k_mask=None, half_m=None, rgb=False):
    """Object looks from eye_pos [B,3] (or [B,K,3]): -> grey [B,K,LH*LW] (pre-noise, 0..1 times light), range [B,K].
    v2 options: half_m sizes the window to +/- half_m metres at the object's range (mb_front.crop_box) instead of
    0.8 x the object diagonal; rgb=True returns the colour window at the sub-pixel size, acuity-blurred per channel,
    [B,K,LH*SS,LW*SS,3], instead of the grey look."""
    rgb_out = rgb
    B, K = sc.B, sc.K
    c = obj_centre(sc)
    e = np.broadcast_to(eye_pos if eye_pos.ndim == 3 else eye_pos[:, None, :], (B, K, 3))
    f = c - e
    rng_ = np.linalg.norm(f, axis=-1)
    f = f / np.maximum(rng_, 1e-9)[..., None]
    rt = np.stack([f[..., 1], -f[..., 0], np.zeros_like(f[..., 0])], -1)    # f x z
    rt = rt / np.maximum(np.linalg.norm(rt, axis=-1), 1e-9)[..., None]
    up = np.cross(rt, f)
    win = 2 * np.arctan((0.8 * obj_diag(sc) if half_m is None else half_m) / np.maximum(rng_, 1e-3))     # window width (rad)
    W, H = LW * SS, LH * SS
    pitch = win / W
    ix = (np.arange(W) - (W - 1) / 2)[None, None, None, :] * pitch[..., None, None]
    iy = ((H - 1) / 2 - np.arange(H))[None, None, :, None] * pitch[..., None, None]
    d = f[:, :, None, None, :] + np.tan(ix)[..., None] * rt[:, :, None, None, :] + np.tan(iy)[..., None] * up[:, :, None, None, :]
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)                    # [B,K,H,W,3]
    # render each object's window against that object only (plus ground/sky): fold K into the ray axis
    rgb = np.zeros((B, K, H, W, 3))
    for k in range(K):
        sub = Scene.__new__(Scene)
        sub.__dict__.update(sc.__dict__)
        sub.obj = sc.obj[:, k:k + 1]
        rgb[:, k] = shade(sub, e[:, k][:, None, :], d[:, k].reshape(B, H * W, 3), t_now, obstacles=False)[0].reshape(B, H, W, 3)
    sig = np.radians(ACUITY_DEG) / pitch                                    # [B,K] sub-pixels
    gx, gy = _blur_mats(sig, W), _blur_mats(sig, H)
    if rgb_out:
        return np.stack([gy @ rgb[..., c] @ np.swapaxes(gx, -1, -2) for c in range(3)], -1), rng_
    grey = rgb.mean(-1)
    grey = gy @ grey @ np.swapaxes(gx, -1, -2)
    grey = grey.reshape(B, K, LH, SS, LW, SS).mean((3, 5)).reshape(B, K, LH * LW)
    return grey, rng_


def visibility(sc, pos, yaw, t_now, dead=None, occluded=None):
    """-> visible [B,K] bool, bearing [B,K] (world, rad), range_h [B,K] (horizontal m)."""
    B, K = sc.B, sc.K
    c = obj_centre(sc)
    rel = c - pos[:, None, :]
    bearing = np.arctan2(rel[..., 1], rel[..., 0])
    rh = np.hypot(rel[..., 0], rel[..., 1])
    dist = np.linalg.norm(rel, axis=-1)
    size_ok = obj_diag(sc) / np.maximum(dist, 1e-6) >= np.tan(np.radians(DETECT_DEG))
    fog_ok = np.where(sc.fog[:, None] > 0, np.exp(-dist / np.where(sc.fog > 0, sc.fog, 1.0)[:, None]) >= 0.2, True)
    present = sc.kvalid & (sc.obj[..., 16] <= t_now)
    # three sight lines: centre and both ends along the object's local x at mid height
    ends = []
    for s in (0.0, 0.8, -0.8):
        px = c[..., 0] + s * sc.obj[..., 4] * np.cos(sc.obj[..., 3])
        py = c[..., 1] + s * sc.obj[..., 4] * np.sin(sc.obj[..., 3])
        ends.append(np.stack([px, py, c[..., 2]], -1))
    P = np.stack(ends, 2).reshape(B, K * 3, 3)
    v = P - pos[:, None, :]
    L = np.linalg.norm(v, axis=-1)
    d = v / np.maximum(L, 1e-9)[..., None]
    o = np.broadcast_to(pos[:, None, :], d.shape)
    tc = hit_cyl(o, d, sc.cyl)[0]
    tb = hit_box(o, d, sc.box)[0]
    clear = (np.minimum(tc, tb) >= L).reshape(B, K, 3).sum(-1) >= 2
    vis = present & size_ok & fog_ok & clear
    if dead is not None:
        rel_az = (bearing - yaw[:, None] + np.pi) % (2 * np.pi) - np.pi
        col = np.clip(np.floor((np.pi - rel_az) / np.radians(PX_DEG)).astype(int), 0, NC - 1)
        vis &= ~np.take_along_axis(dead, col, 1)
    if occluded is not None:
        vis &= ~occluded
    return vis, bearing, rh


def depth_ring(sc, pos, yaw):
    """Horizontal ranges to obstacles at the drone's altitude, [B, NC] (m, capped at DEPTH_MAX)."""
    az = yaw[:, None] + col_az()[None, :]
    d = np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], -1)
    o = np.broadcast_to(pos[:, None, :], d.shape)
    t = np.minimum(hit_cyl(o, d, sc.cyl)[0], hit_box(o, d, sc.box)[0])
    return np.minimum(t, DEPTH_MAX)


def learning_poses(sc, learning_flight_s):
    """Eye poses of the learning flight around each row's target. -> [B, V, 3] (V = max(1, round(5 Hz * s))).

    0 s: one 'photo' from 4 m in front of the target at 2.5 m height. Otherwise a bee-style learning flight:
    arcs of growing amplitude (40 -> 120 deg) and radius (2 -> 7 m) while climbing (1 -> 3.5 m), facing the target.
    """
    V = max(1, int(math.floor(5 * learning_flight_s + 0.5)))
    tk = np.argmax(sc.role, 1)
    T = sc.obj[np.arange(sc.B), tk]                                         # [B,17]
    if V == 1:
        p = np.zeros(1); az = np.zeros(1); rad = np.full(1, 4.0); alt = np.full(1, 2.5)
    else:
        p = np.arange(V) / (V - 1)
        az = np.radians(40 + 80 * p) * np.sin(3 * np.pi * p)
        rad, alt = 2 + 5 * p, 1 + 2.5 * p
    a = T[:, 3][:, None] + az[None, :]
    return np.stack([T[:, 0][:, None] + rad * np.cos(a), T[:, 1][:, None] + rad * np.sin(a),
                     np.broadcast_to(alt, (sc.B, V))], -1)


def learning_looks(sc, learning_flight_s, **kw):
    """Noise-free looks of the target from the learning poses -> grey [B, V, LH*LW] (kw: looks() v2 options)."""
    P = learning_poses(sc, learning_flight_s)
    tk = np.argmax(sc.role, 1)
    out = []
    for v in range(P.shape[1]):
        g, _ = looks(sc, np.repeat(P[:, v][:, None, :], sc.K, 1), 1e9, **kw)
        out.append(g[np.arange(sc.B), tk])
    return np.stack(out, 1)


if __name__ == "__main__":
    from scenario import sample_scenario, strip
    from world import build_world
    sc_ = strip(sample_scenario(4, fault="decoy", severity=0.3, landscape="forest", obstacle_density=0.5))
    w = build_world(sc_)
    S = Scene([w, w])
    pos = np.array([[0.0, 0.0, 2.5], [1.0, 0.0, 2.5]])
    yaw = np.array([0.0, 1.0])
    img = panorama(S, pos, yaw, 0.0)
    assert img.shape == (2, NR, NC, 3) and img.dtype == np.uint8
    assert img[:, 0].mean() > img[:, -1].mean() * 0.8  # sky row brighter-ish than ground row
    # a ray straight down hits the ground at exactly z/|dz|
    _, dist = shade(S, pos[:, None, :], np.array([[[0, 0, -1.0]]] * 2), 0.0)
    assert np.allclose(dist[:, 0], 2.5)
    # cylinder hit distance analytic check
    cyl = np.array([[[5.0, 0.0, 1.0, 3.0, 0, 0, 0]]])
    t, _, n = hit_cyl(np.array([[[0.0, 0.0, 1.0]]]), np.array([[[1.0, 0.0, 0.0]]]), cyl)
    assert abs(t[0, 0] - 4.0) < 1e-12 and np.allclose(n[0, 0], [-1, 0, 0])
    t, _, n = hit_cyl(np.array([[[5.0, 0.0, 5.0]]]), np.array([[[0.0, 0.0, -1.0]]]), cyl)
    assert abs(t[0, 0] - 2.0) < 1e-12 and np.allclose(n[0, 0], [0, 0, 1])
    box = np.array([[[5.0, 0.0, 1.0, 1.0, 3.0, 0, 0, 0]]])
    t, _, n = hit_box(np.array([[[0.0, 0.0, 1.0]]]), np.array([[[1.0, 0.0, 0.0]]]), box)
    assert abs(t[0, 0] - 4.0) < 1e-12 and np.allclose(n[0, 0], [-1, 0, 0])
    # visibility: objects close and unoccluded are visible; dead columns hide them
    tgt = [o for o in w["objects"] if o["role"] == "target"][0]
    near = np.array([[tgt["x"] - 4.0, tgt["y"], 2.5]] * 2)
    vis, bear, rh = visibility(S, near, np.zeros(2), 0.0)
    k = tgt["slot"]
    assert vis[0, k] and abs(bear[0, k]) < 0.05 and abs(rh[0, k] - 4.0) < 1e-9
    dead = np.zeros((2, NC), bool); dead[:, 30:42] = True
    vis2, _, _ = visibility(S, near, np.zeros(2), 0.0, dead=dead)
    assert not vis2[0, k]
    # looks: shape, and target look differs from decoy look at close range more than far away
    g, r = looks(S, near, 0.0)
    assert g.shape == (2, w and S.K, LH * LW) and np.all(g >= 0) and np.all(g <= 1)
    L = learning_looks(S, 10.0)
    assert L.shape == (2, 50, LH * LW)
    L1 = learning_looks(S, 0.0)
    assert L1.shape == (2, 1, LH * LW)
    # depth ring sees an obstacle placed straight ahead
    S2 = Scene([{**w, "obstacles": [{"type": "tree", "x": 6.0, "y": 0.0, "r": 1.0, "h": 6.0, "rgb": [0, 1, 0]}]}])
    dr = depth_ring(S2, np.array([[0.0, 0.0, 2.5]]), np.zeros(1))
    assert abs(dr[0, 35] - 5.029) < 0.01 and dr[0, 35] == dr[0, 36] and dr[0, 0] == DEPTH_MAX, dr[0, 33:39]
    # uniforms()/normals() equal sequential Rng draws
    r1, r2 = Rng([3, 9], 4), Rng([3, 9], 4)
    u = uniforms(r1, 5)
    seq = np.stack([r2.uniform() for _ in range(5)], 1)
    assert np.array_equal(u, seq) and np.array_equal(r1.a, r2.a)
    z = normals(r1, 3); zs = np.stack([r2.normal() for _ in range(3)], 1)
    assert np.allclose(z, zs, rtol=0, atol=1e-15)
    print("eye self-check OK; panorama", img.shape, "look", g.shape)
