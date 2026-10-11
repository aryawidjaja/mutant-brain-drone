"""H4 recognition scene: a textured MuJoCo forest with a ridge tent (target) and three swap decoys.

All objects are UV-textured procedural meshes (textures are generated PNGs, written once to
results/recog_renders/textures/). Every object is a mocap body; the one being shown sits at the target's
exact pose, the others 100 m under ground (swap control: same spot, same background, same camera, same sun).

  OBJECTS   target  ridge tent, fabric T1 (stripes) + door panel, guy-lines
            d1_tex  same mesh, texture T2 (same two colours, checks)
            d2_mesh dome tent, texture T1
            d3_twin same mesh + T1, door open (dark opening + rolled flap), no guy-lines
            pad, backpack: app gallery only

  uv run python scripts/recog_scene.py            # self-check (seed 0, training) + contact sheet
  uv run python scripts/recog_scene.py --gallery  # app/assets/targets/*.png + manifest.json
"""
import json, os, sys
import numpy as np
from PIL import Image

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
import mujoco  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "results", "recog_renders", "textures")
OBJECTS = ["target", "d1_tex", "d2_mesh", "d3_twin"]
GALLERY = ["pad", "backpack"]
ORANGE, GREY, YELLOW = (0.93, 0.45, 0.12), (0.33, 0.35, 0.38), (0.95, 0.85, 0.15)
APPEARANCE = {  # for the app manifest (C5 appearance fields)
    "target": {"kind": "tent", "base": ORANGE, "accent": GREY, "pattern": "stripes"},
    "d1_tex": {"kind": "tent", "base": ORANGE, "accent": GREY, "pattern": "checks"},
    "d2_mesh": {"kind": "tent", "base": ORANGE, "accent": GREY, "pattern": "stripes", "shape": "dome"},
    "d3_twin": {"kind": "tent", "base": ORANGE, "accent": GREY, "pattern": "stripes", "detail": "door open, no guy-lines"},
    "pad": {"kind": "pad", "base": ORANGE, "accent": (1, 1, 1), "pattern": "H"},
    "backpack": {"kind": "backpack", "base": (0.2, 0.32, 0.65), "accent": ORANGE, "pattern": "straps"},
}
OBJ_Z = 0.6          # gaze height (object centre)
FOVY = 60.0
TENT_L, TENT_W, TENT_H = 2.2, 1.6, 1.2


# ---------- textures ----------
def _save(name, a):
    Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8)).save(os.path.join(TEX, name + ".png"))


def make_textures():
    os.makedirs(TEX, exist_ok=True)
    rs = np.random.default_rng(7)
    o, g, y = map(np.array, (ORANGE, GREY, YELLOW))
    weave = 1 + 0.06 * rs.standard_normal((256, 512, 1))                       # fabric grain
    yy, xx = np.mgrid[0:256, 0:512]

    def fabric(pattern):
        if pattern == "stripes":
            m = ((xx // 32) % 2 == 0)
        else:  # checks: same colours, different arrangement
            m = (((xx // 32) + (yy // 32)) % 2 == 0)
        a = np.where(m[..., None], o, g) * weave
        # u in [0.5, 0.75): front end panel with a door (darker trapezoid + zipper); [0.75, 1): back end plain
        a[:, 256:] = o * weave[:, 256:]
        u, v = (xx - 256) / 128.0, yy / 256.0                                   # local coords of front panel
        door = (xx >= 256) & (xx < 384) & (v > 0.35) & (np.abs(u - 0.5) < 0.32 * (v - 0.2))
        a[door] = (0.55 * o)[None] * weave[door]
        a[(xx >= 318) & (xx < 322) & (v > 0.4)] = y
        return a

    _save("fabric_t1", fabric("stripes"))
    _save("fabric_t2", fabric("checks"))
    _save("interior", np.full((64, 64, 3), 0.06) + 0.02 * rs.standard_normal((64, 64, 3)))
    n = rs.standard_normal((256, 256))
    blur = lambda a, k: sum(np.roll(np.roll(a, i, 0), j, 1) for i in range(-k, k + 1) for j in range(-k, k + 1)) / (2 * k + 1) ** 2
    gr = blur(n, 2)
    grass = np.stack([0.30 + 0.10 * gr, 0.42 + 0.14 * gr + 0.05 * n, 0.16 + 0.05 * gr], -1)
    patches = blur(rs.standard_normal((256, 256)), 8) > 0.15                     # dry leaf-litter patches
    grass[patches] = np.array([0.45, 0.36, 0.22]) + 0.06 * n[patches][:, None]
    _save("grass", grass)
    bark_n = np.repeat(rs.standard_normal((1, 256)), 256, 0) + 0.4 * rs.standard_normal((256, 256))
    _save("bark", np.stack([0.30 + 0.07 * bark_n, 0.22 + 0.05 * bark_n, 0.15 + 0.03 * bark_n], -1))
    lv = blur(rs.standard_normal((256, 256)), 1)
    _save("leaves", np.stack([0.14 + 0.08 * lv, 0.32 + 0.12 * lv, 0.10 + 0.05 * lv], -1))
    yy, xx = np.mgrid[0:256, 0:256]
    pad = np.ones((256, 256, 3)) * o * weave[:, :256]
    r = np.hypot(xx - 128, yy - 128)
    pad[(r > 92) & (r < 104)] = 1
    hmask = ((np.abs(xx - 98) < 9) | (np.abs(xx - 158) < 9)) & (np.abs(yy - 128) < 50) | (np.abs(yy - 128) < 8) & (np.abs(xx - 128) < 30)
    pad[hmask] = 1
    _save("pad", pad)
    bp = np.ones((256, 256, 3)) * np.array(APPEARANCE["backpack"]["base"]) * weave[:, :256]
    bp[(np.abs(xx - 80) < 10) | (np.abs(xx - 176) < 10)] = o
    bp[(yy > 150) & (yy < 220) & (np.abs(xx - 128) < 60)] *= 0.7
    _save("backpack", bp)


# ---------- meshes (triangle soup: unshared vertices -> flat facets, per-face UVs) ----------
class Soup:
    def __init__(self):
        self.v, self.t = [], []

    def tri(self, p, uv):
        self.v += list(p); self.t += list(uv)

    def quad(self, p, uv):
        self.tri([p[0], p[1], p[2]], [uv[0], uv[1], uv[2]]); self.tri([p[0], p[2], p[3]], [uv[0], uv[2], uv[3]])

    def xml(self, name, tex_v_flip=True):
        v = np.asarray(self.v, float); t = np.asarray(self.t, float)
        if tex_v_flip:
            t[:, 1] = 1 - t[:, 1]
        f = np.arange(len(v))
        s = lambda a: " ".join(f"{x:.5g}" for x in a.ravel())
        return f'<mesh name="{name}" inertia="shell" vertex="{s(v)}" texcoord="{s(t)}" face="{s(f)}"/>'


def ridge_tent():
    L, W, H = TENT_L / 2, TENT_W / 2, TENT_H
    s = Soup()
    # roof panels -> stripes region u in [0, 0.5]
    s.quad([(-L, W, 0), (-L, 0, H), (L, 0, H), (L, W, 0)], [(0, 0), (0, 1), (0.5, 1), (0.5, 0)])
    s.quad([(L, -W, 0), (L, 0, H), (-L, 0, H), (-L, -W, 0)], [(0, 0), (0, 1), (0.5, 1), (0.5, 0)])
    s.tri([(L, -W, 0), (L, W, 0), (L, 0, H)], [(0.5, 0), (0.75, 0), (0.625, 1)])   # front, with door
    s.tri([(-L, W, 0), (-L, -W, 0), (-L, 0, H)], [(0.75, 0), (1, 0), (0.875, 1)])  # back, plain
    s.quad([(-L, -W, 0.01), (L, -W, 0.01), (L, W, 0.01), (-L, W, 0.01)], [(0.8, 0)] * 4)
    return s


def dome_tent(rx=1.2, ry=0.95, rz=1.2, n_el=6):
    s = Soup()
    az = np.radians(np.r_[np.linspace(-30, 30, 4), np.linspace(30, 330, 16)[1:]])
    az = np.r_[az, az[0] + 2 * np.pi]
    el = np.linspace(0, np.pi / 2, n_el + 1)

    def P(a, e):
        return (rx * np.cos(e) * np.cos(a), ry * np.cos(e) * np.sin(a), rz * np.sin(e))

    def U(a, e):
        d = np.degrees(a)
        u = 0.5 + (d + 30) / 60 * 0.25 if d <= 30.001 else (d - 30) / 300 * 0.5
        return (u, e / (np.pi / 2))
    for i in range(len(az) - 1):
        for j in range(n_el):
            a0, a1, e0, e1 = az[i], az[i + 1], el[j], el[j + 1]
            s.quad([P(a0, e0), P(a1, e0), P(a1, e1), P(a0, e1)], [U(a0, e0), U(a1, e0), U(a1, e1), U(a0, e1)])
    return s


def door_opening():
    L, s = TENT_L / 2 + 0.01, Soup()
    s.tri([(L, -0.3, 0.02), (L, 0.3, 0.02), (L, 0, 0.95)], [(0, 0), (1, 0), (0.5, 1)])
    s.tri([(L, 0.3, 0.02), (L, -0.3, 0.02), (L, 0, 0.95)], [(0, 0), (1, 0), (0.5, 1)])  # two-sided
    return s


def box(sx, sy, sz):
    s, (x, y, z) = Soup(), (sx / 2, sy / 2, sz / 2)
    uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for p in ([(-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)], [(-x, y, -z), (x, y, -z), (x, -y, -z), (-x, -y, -z)],
              [(x, -y, -z), (x, y, -z), (x, y, z), (x, -y, z)], [(-x, y, -z), (-x, -y, -z), (-x, -y, z), (-x, y, z)],
              [(-x, -y, -z), (x, -y, -z), (x, -y, z), (-x, -y, z)], [(x, y, -z), (-x, y, -z), (-x, y, z), (x, y, z)]):
        s.quad(p, uv)
    return s


def lathe(prof, n=12, vrep=1.0):
    """Surface of revolution about z; prof = [(r, z), ...] bottom to top. u = azimuth, v = z * vrep."""
    s = Soup()
    a = np.linspace(0, 2 * np.pi, n + 1)
    for i in range(n):
        for (r0, z0), (r1, z1) in zip(prof[:-1], prof[1:]):
            c0, s0, c1, s1 = np.cos(a[i]), np.sin(a[i]), np.cos(a[i + 1]), np.sin(a[i + 1])
            s.quad([(r0 * c0, r0 * s0, z0), (r0 * c1, r0 * s1, z0), (r1 * c1, r1 * s1, z1), (r1 * c0, r1 * s0, z1)],
                   [(i / n, z0 * vrep), ((i + 1) / n, z0 * vrep), ((i + 1) / n, z1 * vrep), (i / n, z1 * vrep)])
    return s


def trunk(r, h):
    return lathe([(r * 1.3, 0), (r, 0.6), (r * 0.7, h)], n=10, vrep=0.5)


def blob(rx, rz, n=10):
    e = np.linspace(-np.pi / 2, np.pi / 2, 7)
    return lathe([(rx * np.cos(x) + 1e-3, rz * (1 + np.sin(x))) for x in e], n=n, vrep=1 / max(rz, 0.3))


# ---------- layout ----------
def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1), t * np.linalg.norm(ab)


def layout(seed):
    """Deterministic layout for one forest seed: object yaw, trees, bushes, suns."""
    rs = np.random.default_rng([seed, 11])
    yaw = rs.uniform(0, 360)
    sight = [(th, 12.7) for th in range(0, 181, 15)] + [(th, 6.4) for th in np.linspace(-45, 45, 12)]

    def clear(xy, rad):
        ok = np.hypot(*xy.T) > 3.0 + rad
        for th, d in sight:
            ang = np.radians(yaw + th)
            dist, t = _seg_dist(xy, np.zeros(2), d * np.array([np.cos(ang), np.sin(ang)]))
            ok &= dist > rad + 0.6 + 0.07 * t                                      # 0.07 ~ sin(4 deg) azimuth jitter
        return ok
    xy = rs.uniform(-28, 28, (260, 2)); tr = rs.uniform(0.12, 0.3, 260); th_ = rs.uniform(6, 11, 260)
    keep = clear(xy, tr)
    trees = [dict(x=float(x), y=float(y), r=float(r), h=float(h), v=int(rs.integers(0, 6)), yaw=float(rs.uniform(0, 360)))
             for (x, y), r, h, k in zip(xy, tr, th_, keep) if k]
    bxy = rs.uniform(-25, 25, (90, 2)); br = rs.uniform(0.3, 0.7, 90)
    bk = clear(bxy, br)
    bushes = [dict(x=float(x), y=float(y), r=float(r), v=int(rs.integers(0, 4))) for (x, y), r, k in zip(bxy, br, bk) if k]
    suns = [dict(az=float(rs.uniform(0, 360)), el=float(rs.uniform(20, 70)), diffuse=float(rs.uniform(0.5, 1.0)),
                 ambient=float(rs.uniform(0.15, 0.35))) for _ in range(3)]               # [memory, probe0, probe1]
    return dict(seed=seed, yaw=yaw, trees=trees, bushes=bushes, suns=suns)


def build_xml(lay, size=224):
    meshes = [ridge_tent().xml("tent_ridge"), dome_tent().xml("tent_dome"), door_opening().xml("door_open"),
              box(1.6, 1.6, 0.05).xml("pad_m"), box(0.35, 0.25, 0.5).xml("pack_m")]
    tv = [(0.15 + 0.03 * i, 7 + i) for i in range(6)]
    meshes += [trunk(r, h).xml(f"trunk{i}") for i, (r, h) in enumerate(tv)]
    meshes += [blob(1.6 + 0.3 * i, 1.4 + 0.4 * i).xml(f"canopy{i}") for i in range(3)]
    meshes += [blob(1.0, 0.8).xml(f"bush{i}") for i in range(1)]
    tex = "".join(f'<texture name="{n}" type="2d" file="{os.path.join(TEX, n + ".png")}"/>'
                  for n in ["fabric_t1", "fabric_t2", "interior", "grass", "bark", "leaves", "pad", "backpack"])
    mats = "".join(f'<material name="m_{n}" texture="{n}" specular="0.05" shininess="0.1"/>'
                   for n in ["fabric_t1", "fabric_t2", "interior", "bark", "leaves", "pad", "backpack"])
    mats += '<material name="m_grass" texture="grass" texrepeat="40 40" specular="0"/>'
    world = []
    for i, t in enumerate(lay["trees"]):
        r0, h0 = tv[t["v"]]
        sc = t["r"] / r0
        world.append(f'<geom type="mesh" mesh="trunk{t["v"]}" material="m_bark" pos="{t["x"]:.3f} {t["y"]:.3f} 0" euler="0 0 {t["yaw"]:.1f}"/>')
        world.append(f'<geom type="mesh" mesh="canopy{t["v"] % 3}" material="m_leaves" pos="{t["x"]:.3f} {t["y"]:.3f} {0.55 * h0 + 1.5 * sc:.2f}" euler="0 0 {t["yaw"]:.1f}"/>')
    for b in lay["bushes"]:
        world.append(f'<geom type="mesh" mesh="bush0" material="m_leaves" pos="{b["x"]:.3f} {b["y"]:.3f} -0.15" euler="0 0 {b["x"] * 37 % 360:.0f}"/>')
    L, W, H = TENT_L / 2, TENT_W / 2, TENT_H
    guy = "".join(f'<geom type="capsule" size="0.015" fromto="{a[0]} {a[1]} {a[2]} {b[0]} {b[1]} {b[2]}" rgba="0.95 0.9 0.2 1"/>'
                  for a, b in [((L, 0, H), (L + 0.9, 0, 0)), ((-L, 0, H), (-L - 0.9, 0, 0)),
                               ((L * 0.6, W * 0.5, H * 0.5), (L * 0.6 + 0.3, W + 0.7, 0)), ((L * 0.6, -W * 0.5, H * 0.5), (L * 0.6 + 0.3, -W - 0.7, 0)),
                               ((-L * 0.6, W * 0.5, H * 0.5), (-L * 0.6 - 0.3, W + 0.7, 0)), ((-L * 0.6, -W * 0.5, H * 0.5), (-L * 0.6 - 0.3, -W - 0.7, 0))])
    dguy = "".join(f'<geom type="capsule" size="0.015" fromto="{a[0]:.3f} {a[1]:.3f} 0.85 {b[0]:.3f} {b[1]:.3f} 0" rgba="0.95 0.9 0.2 1"/>'
                   for a, b in [((0.85 * np.cos(q), 0.67 * np.sin(q)), (1.9 * np.cos(q), 1.6 * np.sin(q))) for q in np.radians([45, 135, 225, 315])])
    body = lambda n, inner: f'<body name="{n}" mocap="true" pos="0 0 -100">{inner}</body>'
    objs = [body("target", '<geom type="mesh" mesh="tent_ridge" material="m_fabric_t1"/>' + guy),
            body("d1_tex", '<geom type="mesh" mesh="tent_ridge" material="m_fabric_t2"/>' + guy),
            body("d2_mesh", '<geom type="mesh" mesh="tent_dome" material="m_fabric_t1"/>' + dguy),
            body("d3_twin", '<geom type="mesh" mesh="tent_ridge" material="m_fabric_t1"/><geom type="mesh" mesh="door_open" material="m_interior"/>'
                 f'<geom type="capsule" size="0.06" fromto="{L + 0.06} -0.3 0.98 {L + 0.06} 0.3 0.98" rgba="{ORANGE[0] * .8} {ORANGE[1] * .8} {ORANGE[2] * .8} 1"/>'),
            body("pad", '<geom type="mesh" mesh="pad_m" material="m_pad" pos="0 0 0.025"/>'),
            body("backpack", '<geom type="mesh" mesh="pack_m" material="m_backpack" pos="0 0 0.25"/>')]
    return f"""<mujoco model="recog_forest">
  <compiler inertiafromgeom="false"/>
  <visual><global fovy="{FOVY}" offwidth="{size}" offheight="{size}"/><quality shadowsize="4096"/>
    <headlight ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/><map znear="0.01" zfar="30"/></visual>
  <statistic extent="10" center="0 0 1"/>
  <default><geom contype="0" conaffinity="0"/></default>
  <asset>{tex}{mats}{''.join(meshes)}
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.62 0.75 0.9" rgb2="0.9 0.93 0.97" width="256" height="256"/></asset>
  <worldbody>
    <light name="sun" type="directional" castshadow="true" dir="0 0 -1" diffuse="0.8 0.8 0.8" ambient="0.25 0.25 0.25" specular="0.1 0.1 0.1"/>
    <geom type="plane" size="60 60 0.1" material="m_grass"/>
    {''.join(world)}
    {''.join(objs)}
  </worldbody>
</mujoco>"""


class Scene:
    """One forest layout with an offscreen renderer. show(obj), set_sun(i), render(theta, dist, height, ...)."""

    def __init__(self, seed, size=224):
        make_textures() if not os.path.exists(os.path.join(TEX, "backpack.png")) else None
        self.lay = layout(seed)
        self.m = mujoco.MjModel.from_xml_string(build_xml(self.lay, size))
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, size, size)
        self.cam = mujoco.MjvCamera()
        self.names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self.m.nbody)]
        self.yaw = self.lay["yaw"]

    def show(self, obj):
        q = np.radians(self.yaw) / 2
        for i, n in enumerate(self.names):
            mid = self.m.body_mocapid[i]
            if mid >= 0:
                self.d.mocap_pos[mid] = (0, 0, 0) if n == obj else (0, 0, -100)
                self.d.mocap_quat[mid] = (np.cos(q), 0, 0, np.sin(q))

    def set_sun(self, k):
        s = self.lay["suns"][k]
        az, el = np.radians(s["az"]), np.radians(s["el"])
        self.m.light_dir[0] = -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        self.m.light_diffuse[0] = s["diffuse"]
        self.m.light_ambient[0] = s["ambient"]

    def render(self, theta, dist, height=2.0, gaze=(0.0, 0.0)):
        """Camera on a circle `dist` m from the object, `theta` deg from the door direction, at `height` m,
        looking at the object centre (+ gaze offset in world x, y)."""
        mujoco.mj_forward(self.m, self.d)
        psi = self.yaw + theta
        dz = height - OBJ_Z
        self.cam.lookat[:] = (gaze[0], gaze[1], OBJ_Z)
        self.cam.distance = float(np.hypot(dist, dz))
        self.cam.azimuth = psi + 180.0                     # MuJoCo: camera sits at lookat - distance * forward(az, el)
        self.cam.elevation = -np.degrees(np.arctan2(dz, dist))
        self.r.update_scene(self.d, self.cam)
        return self.r.render().copy()

    def close(self):
        self.r.close()


def gallery():
    out = os.path.join(ROOT, "app", "assets", "targets")
    os.makedirs(out, exist_ok=True)
    sc = Scene(0, size=256)
    sc.set_sun(0)
    framing = {"target": (3.6, 1.8), "d1_tex": (3.6, 1.8), "d2_mesh": (3.6, 1.8), "d3_twin": (3.6, 1.8), "pad": (3.0, 3.0), "backpack": (1.4, 1.3)}
    labels = {"target": "Target: ridge tent (orange/grey stripes)", "d1_tex": "Decoy 1: same tent, check pattern",
              "d2_mesh": "Decoy 2: dome tent, same fabric", "d3_twin": "Decoy 3: near-twin, door open, no guy-lines",
              "pad": "Supply drop pad", "backpack": "Backpack"}
    items = []
    for n, (d, h) in framing.items():
        sc.show(n)
        Image.fromarray(sc.render(15, d, h)).save(os.path.join(out, f"{n}.png"))
        items.append({"id": n, "file": f"{n}.png", "label": labels[n], "role": "target" if n in ("target", "pad", "backpack") else "decoy",
                      **{k: (list(v) if isinstance(v, tuple) else v) for k, v in APPEARANCE[n].items()},
                      "view": {"theta_deg": 15, "dist_m": d, "height_m": h}})
    sc.close()
    json.dump({"source": "scripts/recog_scene.py --gallery (MuJoCo 3.15 offscreen render, forest seed 0, procedural textures, CC0-equivalent: generated here)",
               "size_px": 256, "items": items}, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print("gallery ->", out)


if __name__ == "__main__":
    if "--gallery" in sys.argv:
        gallery(); sys.exit()
    make_textures()
    sc = Scene(0)                                          # training seed: debug only
    sc.set_sun(1)
    imgs = {}
    for o in OBJECTS + [None]:
        sc.show(o)
        imgs[o] = sc.render(0, 6, 2.0).astype(int)
    lay = sc.lay
    assert len(lay["trees"]) > 80, len(lay["trees"])
    empty = imgs[None]
    for o in OBJECTS:
        diff = np.abs(imgs[o] - empty).sum(-1) > 6
        assert diff.mean() > 0.01, (o, diff.mean())       # the object is visible
        if o != "target":
            dt = np.abs(imgs[o] - imgs["target"]).sum(-1) > 6
            assert dt.mean() > 0.003, (o, dt.mean())      # decoy differs from target
            edge = dt.copy(); edge[40:184, 30:194] = False
            assert edge.mean() < 0.01, (o, edge.mean())   # swap control: differences stay near the object (+ shadow)
    # determinism
    sc2 = Scene(0); sc2.set_sun(1); sc2.show("target")
    assert (sc2.render(0, 6, 2.0).astype(int) == imgs["target"]).all()
    sc2.close()
    # contact sheet for eyeballing (training seed 0)
    rows = []
    for o in OBJECTS:
        sc.show(o)
        rows.append(np.concatenate([sc.render(th, d, 2.0) for th, d in [(0, 3), (0, 6), (0, 12), (45, 6), (90, 6), (180, 6)]], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(ROOT, "results", "recog_renders", "contact_seed0_train.png"))
    sc.close()
    print("recog_scene self-check ok; trees", len(lay["trees"]), "bushes", len(lay["bushes"]))
