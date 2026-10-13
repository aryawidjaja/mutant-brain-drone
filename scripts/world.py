"""World builder: procedural meadow / forest / urban from the scenario seed (Rng stream 0).

Identical in app/sim/world.js (parity_brain.py asserts equality). Only +, *, floor on the uniform draws (no trig),
so Python and JS agree bit-for-bit.

Geometry: world rectangle centred on (0, 0); obstacles are vertical cylinders (tree crowns as one cylinder, rocks)
or axis-aligned boxes (buildings). Objects (target, decoys) are yawed boxes with a textured appearance:
  appearance = {base, accent, pattern: stripes|chevron|dots, period_m, similarity}
A decoy of similarity s blends its colours toward the target's (lerp from an off-palette colour by s) and keeps
the target's pattern type, but its pattern period differs by a factor 1 + 0.8 (1 - s): far away (blurred) it
looks like the target; up close the finer detail differs. s = 1 is a perfect copy.
Run: uv run python scripts/world.py
"""
import math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rng import Rng  # noqa: E402

N_MAX = {"meadow": 40, "forest": 250, "urban": 60}            # obstacle count at density 1 over a 100 x 100 m world
GROUND = {"meadow": [0.36, 0.52, 0.24], "forest": [0.28, 0.33, 0.18], "urban": [0.42, 0.42, 0.44]}
DIMS = {"tent": [2.2, 1.8, 1.3], "pad": [2.0, 2.0, 0.12], "backpack": [0.6, 0.4, 0.7]}   # w (local x), d (local y), h  [m]
# target picture gallery: (base, accent, pattern, period_m) per kind; the seed picks one
GALLERY = {
    "tent": [([0.90, 0.45, 0.10], [0.95, 0.90, 0.80], "stripes", 0.50), ([0.15, 0.45, 0.80], [0.95, 0.85, 0.20], "chevron", 0.60),
             ([0.80, 0.15, 0.15], [0.95, 0.95, 0.95], "dots", 0.45), ([0.20, 0.60, 0.30], [0.95, 0.75, 0.10], "stripes", 0.40)],
    "pad": [([0.95, 0.85, 0.10], [0.10, 0.10, 0.10], "chevron", 0.50), ([0.90, 0.30, 0.10], [0.95, 0.95, 0.95], "stripes", 0.40),
            ([0.20, 0.30, 0.85], [0.95, 0.95, 0.95], "dots", 0.40), ([0.95, 0.95, 0.95], [0.85, 0.10, 0.10], "chevron", 0.60)],
    "backpack": [([0.85, 0.20, 0.10], [0.15, 0.15, 0.15], "stripes", 0.15), ([0.10, 0.35, 0.75], [0.95, 0.80, 0.10], "dots", 0.15),
                 ([0.95, 0.60, 0.05], [0.20, 0.20, 0.20], "chevron", 0.20), ([0.30, 0.55, 0.20], [0.90, 0.90, 0.85], "stripes", 0.12)],
}
CLEAR_M = 3.0
# v2 only: the home landing pad, a flat 1 m x 1 m mat at home with a pattern no target or decoy uses (magenta and white
# stripes, 0.25 m period). It is not an object slot (never in the arbiter's obs or the stop rule); the home memory sees it.
HOME_PAD = {"role": "home", "kind": "home_pad", "yaw_deg": 0.0, "dims": [1.0, 1.0, 0.06], "t_s": 0.0, "slot": 0,
            "appearance": {"base": [0.75, 0.10, 0.60], "accent": [0.97, 0.97, 0.97], "pattern": "stripes", "period_m": 0.25,
                           "similarity": 1.0}}


def _lerp(a, b, s):
    return [x + (y - x) * s for x, y in zip(a, b)]


def _box_dist(px, py, o):
    dx = max(abs(px - o["x"]) - o["w"] / 2, 0.0)
    dy = max(abs(py - o["y"]) - o["d"] / 2, 0.0)
    return (dx * dx + dy * dy) ** 0.5


def build_world(sc):
    """scenario (validated, defaults filled) -> world dict. Deterministic from sc['seed'] via Rng stream 0."""
    r = Rng([sc["seed"]], 0)
    U = lambda: float(r.uniform()[0])  # noqa: E731
    W, H = sc["world"]["size_m"]
    land = sc["world"]["landscape"]
    sun = {"azimuth_deg": 360 * U(), "elevation_deg": 25 + 35 * U()}
    t = sc["target"]
    u = U()                                              # always drawn, so a chosen picture leaves the layout unchanged
    base, accent, pattern, period = GALLERY[t["kind"]][t["picture"] if "picture" in t else min(3, int(u * 4))]
    objects = [{"role": "target", "kind": t["kind"], "x": t["pos"][0], "y": t["pos"][1], "z": t["pos"][2],
                "yaw_deg": t["yaw_deg"], "dims": DIMS[t["kind"]], "t_s": 0.0,
                "appearance": {"base": base, "accent": accent, "pattern": pattern, "period_m": period, "similarity": 1.0}}]
    for dc in sc["decoys"]:
        s = dc["similarity"]
        off_b, off_a = [U(), U(), U()], [U(), U(), U()]        # off-palette colours the decoy blends away from
        same = dc["kind"] == t["kind"]
        db, da, dp, dper = (base, accent, pattern, period) if same else GALLERY[dc["kind"]][0]
        objects.append({"role": "decoy", "kind": dc["kind"], "x": dc["pos"][0], "y": dc["pos"][1], "z": dc["pos"][2],
                        "yaw_deg": t["yaw_deg"] + 360 * U(), "dims": DIMS[dc["kind"]], "t_s": dc["t_s"],
                        "appearance": {"base": _lerp(off_b, db, s), "accent": _lerp(off_a, da, s), "pattern": dp,
                                       "period_m": dper * (1 + 0.8 * (1 - s)), "similarity": s}})
    # slot order is random per seed (slot 0 is not "the target"): Fisher-Yates on stream 0
    order = list(range(len(objects)))
    for i in range(len(order) - 1, 0, -1):
        j = min(i, int(U() * (i + 1)))
        order[i], order[j] = order[j], order[i]
    objects = [objects[k] for k in order]
    for k, o in enumerate(objects):
        o["slot"] = k
    clear = [(sc["home"][0], sc["home"][1])] + [(o["x"], o["y"]) for o in objects]
    n = int(math.floor(sc["world"]["obstacle_density"] * N_MAX[land] + 0.5))   # half-up, same as JS Math.round
    obstacles = []
    for _ in range(n):
        for _try in range(30):
            x, y = (U() - 0.5) * W, (U() - 0.5) * H
            kind_u, a, b, c, g = U(), U(), U(), U(), U()
            if land == "urban":
                o = {"type": "building", "x": x, "y": y, "w": 4 + 8 * a, "d": 4 + 8 * b, "h": 5 + 10 * c,
                     "rgb": [0.55 + 0.25 * g, 0.53 + 0.22 * g, 0.50 + 0.2 * g]}
                ok = all(_box_dist(px, py, o) > CLEAR_M for px, py in clear)
            else:
                tree = kind_u < (0.9 if land == "forest" else 0.35)
                o = ({"type": "tree", "x": x, "y": y, "r": 0.8 + 0.8 * a, "h": 4 + 5 * b,
                      "rgb": [0.10 + 0.10 * g, 0.30 + 0.20 * c, 0.08 + 0.08 * g]} if tree else
                     {"type": "rock", "x": x, "y": y, "r": 0.3 + 0.5 * a, "h": 0.4 + 0.8 * b,
                      "rgb": [0.45 + 0.2 * g, 0.45 + 0.2 * g, 0.43 + 0.2 * g]})
                ok = all(((px - x) ** 2 + (py - y) ** 2) ** 0.5 > CLEAR_M + o["r"] for px, py in clear)
            if ok:
                obstacles.append(o)
                break
    w = {"size": [W, H], "ground": {"rgb": GROUND[land]}, "obstacles": obstacles, "objects": objects,
         "home": list(sc["home"]), "sun": sun, "light": sc["world"]["light"], "fog_m": sc["world"]["fog_m"]}
    if sc["brain"].get("version", 1) == 2:             # no draws: the v1 layout is unchanged
        w["home_pad"] = {**HOME_PAD, "x": sc["home"][0], "y": sc["home"][1], "z": sc["home"][2],
                         "dims": list(HOME_PAD["dims"]), "appearance": {k: (list(v) if isinstance(v, list) else v) for k, v in HOME_PAD["appearance"].items()}}
    return w


if __name__ == "__main__":
    from scenario import sample_scenario, strip
    for land, dens in [("meadow", 0.3), ("forest", 0.6), ("urban", 0.5)]:
        sc = strip(sample_scenario(7, fault="decoy", severity=0.8, landscape=land, obstacle_density=dens))
        w = build_world(sc)
        assert w == build_world(sc), "not deterministic"
        assert len(w["obstacles"]) > 0.8 * round(dens * N_MAX[land]), (land, len(w["obstacles"]))
        pts = [w["home"][:2]] + [[o["x"], o["y"]] for o in w["objects"]]
        for o in w["obstacles"]:
            for px, py in pts:
                d = _box_dist(px, py, o) if o["type"] == "building" else ((px - o["x"]) ** 2 + (py - o["y"]) ** 2) ** 0.5 - o["r"]
                assert d > CLEAR_M, (land, o, px, py)
        roles = sorted(o["role"] for o in w["objects"])
        assert roles == ["decoy", "target"] and sorted(o["slot"] for o in w["objects"]) == [0, 1]
        dec = [o for o in w["objects"] if o["role"] == "decoy"][0]
        assert abs(dec["appearance"]["similarity"] - (0.5 + 0.45 * 0.8)) < 1e-12
    # slot order is not always target-first
    firsts = {build_world(strip(sample_scenario(s, fault="decoy")))["objects"][0]["role"] for s in range(20)}
    assert firsts == {"target", "decoy"}, firsts
    v2 = strip(sample_scenario(7, fault="decoy", severity=0.8, version=2))
    w2 = build_world(v2)
    assert {k: v for k, v in w2.items() if k != "home_pad"} == build_world(strip(sample_scenario(7, fault="decoy", severity=0.8)))
    assert w2["home_pad"]["x"] == v2["home"][0] and "home_pad" not in build_world(strip(sample_scenario(7)))
    print("world self-check OK;", len(w["obstacles"]), "urban obstacles; sun", w["sun"])
