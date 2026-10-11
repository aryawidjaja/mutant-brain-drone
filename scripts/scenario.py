"""Scenario: validate against schema/scenario.schema.json, fill defaults, hash, sample.

The validator is a ~40-line walker of the JSON schema itself (type, enum, required, additionalProperties,
minimum/maximum/exclusiveMinimum, minItems/maxItems, items, $ref, default), so it follows the schema file.
app/sim/scenario.js embeds a copy of the schema; the self-check asserts the copy is identical.

sample_scenario() draws the PLAN neutral fault list with Rng stream 5 (same draws in scenario.js).
Run: uv run python scripts/scenario.py
"""
import copy, hashlib, json, math, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rng import Rng  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = json.load(open(os.path.join(ROOT, "schema", "scenario.schema.json")))
FAULTS = ["occlusion", "decoy", "compass_half", "eye_columns", "false_sun", "speed_scale", "gust"]  # PLAN neutral list
ARBITERS = ["ring", "ring_static", "vote", "sum", "accumulator", "state_machine"]


def _check(v, s, path, root):
    if "$ref" in s:
        s = root["$defs"][s["$ref"].split("/")[-1]]
    t = s.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        ok = {"object": isinstance(v, dict), "array": isinstance(v, list), "string": isinstance(v, str),
              "boolean": isinstance(v, bool), "null": v is None,
              "integer": isinstance(v, int) and not isinstance(v, bool),
              "number": isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}
        if not any(ok[x] for x in types):
            raise ValueError(f"{path}: expected {t}, got {v!r}")
    if "enum" in s and v not in s["enum"]:
        raise ValueError(f"{path}: {v!r} not in {s['enum']}")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if "minimum" in s and v < s["minimum"]: raise ValueError(f"{path}: {v} < {s['minimum']}")
        if "maximum" in s and v > s["maximum"]: raise ValueError(f"{path}: {v} > {s['maximum']}")
        if "exclusiveMinimum" in s and v <= s["exclusiveMinimum"]: raise ValueError(f"{path}: {v} <= {s['exclusiveMinimum']}")
    if isinstance(v, list):
        if len(v) < s.get("minItems", 0) or len(v) > s.get("maxItems", 1 << 30):
            raise ValueError(f"{path}: length {len(v)} outside [{s.get('minItems')}, {s.get('maxItems')}]")
        if "items" in s:
            return [_check(x, s["items"], f"{path}[{i}]", root) for i, x in enumerate(v)]
        return list(v)
    if isinstance(v, dict):
        props = s.get("properties", {})
        for k in s.get("required", []):
            if k not in v: raise ValueError(f"{path}: missing required '{k}'")
        if s.get("additionalProperties") is False:
            extra = set(v) - set(props)
            if extra: raise ValueError(f"{path}: unknown keys {sorted(extra)}")
        out = {}
        for k, ps in props.items():
            if k in v:
                out[k] = _check(v[k], ps, f"{path}.{k}", root)
            elif "default" in ps:
                out[k] = copy.deepcopy(ps["default"])
        for k in v:
            if k not in out: out[k] = copy.deepcopy(v[k])
        return out
    return v


def load(s):
    """dict or JSON string or path -> validated scenario with defaults filled (raises ValueError)."""
    if isinstance(s, str):
        s = json.load(open(s)) if os.path.exists(s) else json.loads(s)
    return _check(s, SCHEMA, "scenario", SCHEMA)


def _canon(v):
    """Canonical text identical in Python and JS: numbers as round(x*1e6) integers, keys sorted."""
    if isinstance(v, bool): return "true" if v else "false"
    if v is None: return "null"
    if isinstance(v, (int, float)): return str(int(math.floor(v * 1e6 + 0.5)))
    if isinstance(v, str): return json.dumps(v)
    if isinstance(v, list): return "[" + ",".join(_canon(x) for x in v) + "]"
    return "{" + ",".join(json.dumps(k) + ":" + _canon(v[k]) for k in sorted(v)) + "}"


def scenario_hash(s):
    """Condition hash: everything except the seed (paired seeds share it). sha256, 12 hex chars."""
    s = load(s)
    return hashlib.sha256(_canon({k: v for k, v in s.items() if k != "seed"}).encode()).hexdigest()[:12]


def fault_params(fault, sev):
    """PLAN neutral fault list; severity in [0,1] maps linearly onto the pre-registered range."""
    return {"occlusion": {"duration_s": 1 + 9 * sev},           # 1-10 s
            "decoy": {"similarity": 0.5 + 0.45 * sev},          # 0.5-0.95 (swapped decoy)
            "compass_half": {},                                 # half the compass columns silenced
            "eye_columns": {"frac": 0.1 + 0.3 * sev},           # 10-40 % dead
            "false_sun": {"offset_deg": 30 + 90 * sev},         # 30-120 deg
            "speed_scale": {"dev": 0.3 * sev},                  # |scale-1| in 0-0.3 (sign drawn)
            "gust": {"level": ["light", "moderate", "severe"][min(2, int(sev * 3))]}}[fault]


def sample_scenario(seed, split="train", fault=None, severity=None, arbiter="ring", ring_gain=1.0,
                    learning_flight_s=10.0, landscape="meadow", obstacle_density=0.3, params=None,
                    max_time_s=300.0, dist_m=(25.0, 40.0)):
    """One scenario per (seed, condition). Draw order on Rng(seed, 5) is fixed (mirrored in scenario.js):
    u0 target distance, u1 target bearing, u2 target yaw, u3 fault pick (used only if fault is None),
    u4 severity (only if severity is None), u5 fault onset, u6 sign / swap offset angle, u7 swap offset distance,
    u8 wind direction, u9 dead-column start. fault='none' gives the no-fault baseline."""
    assert split in ("train", "test")
    assert (0 <= seed <= 99) if split == "train" else (1000 <= seed <= 1299), f"seed {seed} not in {split} split"
    r = Rng([seed], 5)
    u = [float(r.uniform()[0]) for _ in range(10)]
    d = dist_m[0] + (dist_m[1] - dist_m[0]) * u[0]
    b = 2 * math.pi * u[1]
    tpos = [d * math.cos(b), d * math.sin(b), 0.0]
    fault = FAULTS[min(6, int(u[3] * 7))] if fault is None else fault
    sev = u[4] if severity is None else float(severity)
    s = {"seed": int(seed),
         "world": {"size_m": [100.0, 100.0], "landscape": landscape, "obstacle_density": float(obstacle_density),
                   "light": 1.0, "fog_m": None},
         "home": [0.0, 0.0, 0.0], "target": {"kind": "tent", "pos": tpos, "yaw_deg": 360 * u[2]},
         "decoys": [], "wind": {"mean_mps": [0.0, 0.0, 0.0], "turbulence": "none", "shared_fraction": 1.0},
         "faults": [],
         "brain": {"arbiter": arbiter, "ring_gain": float(ring_gain), "learning_flight_s": float(learning_flight_s),
                   "params": params or {}},
         "mission": {"max_time_s": float(max_time_s), "dt_s": 0.002}}
    t_on = 5 + 20 * u[5]                                          # onset of PI/eye faults: 5-25 s
    if fault != "none":
        p = fault_params(fault, sev)
        if fault == "decoy":
            # swap control: the look-alike stands where the target was reported; the target moved 8-14 m
            a, off = 2 * math.pi * u[6], 8 + 6 * u[7]
            s["decoys"] = [{"kind": "tent", "pos": list(tpos), "swap": True, "similarity": p["similarity"], "t_s": 0.0}]
            s["target"]["pos"] = [tpos[0] + off * math.cos(a), tpos[1] + off * math.sin(a), 0.0]
        elif fault == "gust":
            a = 2 * math.pi * u[8]
            s["wind"] = {"mean_mps": [1.5 * math.cos(a), 1.5 * math.sin(a), 0.0], "turbulence": p["level"],
                         "shared_fraction": 1.0}
        elif fault == "occlusion":
            s["faults"] = [{"kind": "occlusion", "t_s": 0.0, "params": p}]   # starts at the first close sighting
        elif fault == "speed_scale":
            s["faults"] = [{"kind": "speed_scale", "t_s": t_on, "params": {"scale": 1 + (p["dev"] if u[6] < 0.5 else -p["dev"])}}]
        elif fault == "eye_columns":
            s["faults"] = [{"kind": "eye_columns", "t_s": t_on, "params": {**p, "start_col": int(u[9] * 72)}}]
        else:
            s["faults"] = [{"kind": fault, "t_s": t_on, "params": p}]
    s = load(s)
    s["_meta"] = {"split": split, "fault": fault, "severity": sev}       # not part of the schema; stripped by strip()
    return s


def strip(s):
    return {k: v for k, v in s.items() if not k.startswith("_")}


if __name__ == "__main__":
    s = sample_scenario(3, fault="decoy", severity=0.5)
    assert s["decoys"][0]["swap"] and abs(s["decoys"][0]["similarity"] - 0.725) < 1e-12
    dt = math.dist(s["decoys"][0]["pos"], s["target"]["pos"])
    assert 8 <= dt <= 14, dt
    assert load(strip(s)) == strip(s)                                     # idempotent
    h = scenario_hash(strip(s))
    s3 = copy.deepcopy(strip(s)); s3["seed"] = 99
    assert scenario_hash(s3) == h                                         # seed excluded from the condition hash
    # defaults filled
    m = load({"seed": 0, "world": {"size_m": [80, 80], "landscape": "meadow", "obstacle_density": 0},
              "home": [0, 0, 0], "target": {"kind": "tent", "pos": [40, 25, 0]},
              "wind": {"mean_mps": [0, 0, 0], "turbulence": "none"}, "brain": {"arbiter": "ring"},
              "mission": {"max_time_s": 300, "dt_s": 0.002}})
    assert m["world"]["light"] == 1.0 and m["world"]["fog_m"] is None and m["decoys"] == [] and m["faults"] == []
    assert m["brain"]["ring_gain"] == 1.0 and m["brain"]["learning_flight_s"] == 10 and m["target"]["yaw_deg"] == 0
    for bad in [{**m, "extra": 1}, {**m, "seed": -1}, {**m, "world": {**m["world"], "landscape": "moon"}},
                {**m, "home": [0, 0]}, {**m, "brain": {"arbiter": "llm"}}, {k: v for k, v in m.items() if k != "home"}]:
        try:
            load(bad); raise AssertionError(f"accepted bad scenario {bad}")
        except ValueError:
            pass
    # every fault samples and validates; split guards
    for f in FAULTS + ["none"]:
        for sd in (0, 50, 99):
            x = sample_scenario(sd, fault=f); assert x["_meta"]["fault"] == f
    picks = [sample_scenario(sd)["_meta"]["fault"] for sd in range(100)]
    assert set(picks) == set(FAULTS), set(picks)
    try:
        sample_scenario(5, split="test"); raise AssertionError("train seed accepted as test")
    except AssertionError as e:
        assert "not in test" in str(e)
    # the JS copy of the schema is identical to the schema file
    js = open(os.path.join(ROOT, "app", "sim", "scenario.js")).read()
    emb = re.search(r"/\*SCHEMA\*/(.*?)/\*END\*/", js, re.S).group(1)
    assert json.loads(emb) == SCHEMA, "app/sim/scenario.js schema copy is stale"
    print("scenario self-check OK; hash", h, "sample seed 3 decoy:", s["target"]["pos"], s["decoys"][0]["pos"])
