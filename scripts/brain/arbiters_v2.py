"""Study 2 arbiters: the evidence-gated ring (RingGated) and visibility-gated versions of the Study 1 arms.

Same arbiter interface as brain/arbiters.py (obs dict with a leading batch axis B; step -> heading_cmd, commit, viz).
The Study 1 file and the frozen state machine are imported, never edited.

Visibility-gated input (every "_vg" arm and RingGated; the frozen state machine already freezes hidden evidence).
Per object slot k the arm keeps a held strength s_k and a last-seen bearing b_k:
    visible:      s_k = strength_k,                b_k = bearing_k
    not visible:  s_k <- s_k * exp(-dt / tau_hold_s), b_k unchanged
and the arm sees (b_k, s_k, visible = s_k > 0) in place of the raw slot. Occlusion (slot reported not visible) and
fading evidence (visible, strength falling) therefore travel on different channels: a hidden object's evidence decays
slowly at the tuned tau_hold_s, a visible object's evidence is used as it is.

RingGated (model hypothesis, not a measured insect circuit). The Study 1 ring (Sun, Yue & Mangan 2020-style; N = 32,
Gaussian excitation sigma 30 deg, folded uniform inhibition, clip [0, 1], population-vector readout) with its
recurrent gain set every tick by a fast conflict signal instead of a constant:
    I            = object + home + search drive on the ring (shared von Mises input stage, held object strengths)
    theta_b      = population-vector angle of the ring activity r (no bump -> no conflict)
    k_i          = exp(K_C (cos(theta_i - theta_b) - 1)),  K_C = ln 2 / (1 - cos 45 deg)  (k = 0.5 at 45 deg)
    S_c          = max_i I_i k_i                 input support for the current bump direction
    S_alt        = max_i I_i (1 - k_i)           strongest input support away from it
    x            = (S_alt - S_c) / (S_alt + S_c)                          in [-1, 1]
    c            = clip((x - theta_c) / X_W, 0, 1),  X_W = 0.1             instantaneous conflict
    m           <- m + (1 - exp(-dt / tau_m_s)) (c - m)                    low-passed conflict ("neuromodulator")
    g_hi         = ring_gain,  g_lo = min(ring_gain, lo_frac * G_CRIT),  G_CRIT = 1 / lambda_max(W_RING) = 0.2288
    g            = g_hi + (g_lo - g_hi) m
    tau_s dr/dt  = -r + clip(g W_RING r + I, 0, 1)
Bounded gain: g never exceeds g_hi, and under sustained conflict it falls to g_lo < G_CRIT. Below G_CRIT the loop
gain of every active subset of neurons is < 1 (Cauchy interlacing on the symmetric W_RING), so the ring holds no
self-sustained bump and its state is set by the input: the bump cannot latch against contrary evidence.
ring_gain = 0 gives g = 0 at all times, i.e. exactly ring_static_vg (as for the Study 1 ring; asserted below).

Biological motivation (cited in the paper as motivation only): the fly heading bump persists without input and shows
"both abrupt jumps and gradual transitions ... when triggered by strong visual input" (Seelig & Jayaraman 2015,
Nature 521:186); local excitation plus global inhibition (Kim et al. 2017, Science 356:849); central-complex visual
responses gated by behavioural state (Weir, Schnell & Dickinson 2014, J Neurophysiol 111:62). Gating the recurrent
gain by an input-conflict signal is this model's hypothesis, not a finding of those papers.
"""
import numpy as np

try:
    from brain import arbiters as v1
except ImportError:  # run as a file from scripts/brain
    import arbiters as v1

wrap, TH, N_RING, C_INH, W_RING, drive_inputs, _a = v1.wrap, v1.TH, v1.N_RING, v1.C_INH, v1.W_RING, v1.drive_inputs, v1._a
G_CRIT = float(1.0 / np.linalg.eigvalsh(W_RING)[-1])
K_C = float(np.log(2.0) / (1.0 - np.cos(np.pi / 4)))
X_W = 0.1
TAU_HOLD = {"tau_hold_s": (0.1, 20.0, "log")}
ARBITERS_V2 = ("ring_gated", "ring_vg", "ring_static_vg", "vote_vg", "sum_vg", "accumulator_vg", "state_machine")


class _Hold:
    """Mixin: visibility-gated input. Per-row state hs, hb [B, K] (compactable by kt2_five_arms._compact)."""
    hs = hb = None

    def _held(self, obs, dt):
        o = obs["objects"]
        vis = np.asarray(o["visible"], bool)
        s, b = np.asarray(o["strength"], float), np.asarray(o["bearing"], float)
        if self.hs is None:
            self.hs, self.hb = np.zeros(s.shape), b.copy()
        self.hs = np.where(vis, s, self.hs * np.exp(-dt / self.p["tau_hold_s"]))
        self.hb = np.where(vis, b, self.hb)
        return {**obs, "objects": {"bearing": self.hb.copy(), "strength": self.hs.copy(), "visible": self.hs > 0}}


def _vg(base):
    class G(_Hold, base):
        name = base.name + "_vg"
        PARAM_SPACE = {**base.PARAM_SPACE, **TAU_HOLD}
        DEFAULTS = {**base.DEFAULTS, "tau_hold_s": 2.0}

        def step(self, obs, dt):
            return base.step(self, self._held(obs, dt), dt)
    G.__name__ = base.__name__ + "VG"
    return G


RingVG, RingStaticVG, VoteVG, SumVG, AccumulatorVG = map(_vg, (v1.Ring, v1.RingStatic, v1.Vote, v1.Sum, v1.Accumulator))


class RingGated(_Hold):
    name = "ring_gated"
    PARAM_SPACE = {**v1.Ring.PARAM_SPACE, "lo_frac": (0.0, 1.0, "lin"), "tau_m_s": (0.05, 5.0, "log"),
                   "theta_c": (0.0, 0.6, "lin"), **TAU_HOLD}
    DEFAULTS = {**v1.Ring.DEFAULTS, "lo_frac": 0.5, "tau_m_s": 0.5, "theta_c": 0.2, "tau_hold_s": 2.0}

    def __init__(self, params, batch, rng=None):
        self.p = {**self.DEFAULTS, **params}
        assert set(self.p) == set(self.DEFAULTS), set(self.p) ^ set(self.DEFAULTS)
        self.B = batch
        self.r = np.zeros((batch, N_RING))
        self.m = np.zeros(batch)
        self.g_hi = float(self.p["ring_gain"])
        self.g_lo = min(self.g_hi, self.p["lo_frac"] * G_CRIT)

    def step(self, obs, dt):
        p = self.p
        d = drive_inputs(self._held(obs, dt), p["kappa"], p["g_obj"], p["g_home"], p["g_search"])
        I = d["objects"] + d["home"] + d["search"]
        z = (self.r * np.exp(1j * TH)).sum(-1)
        k = np.exp(K_C * (np.cos(TH[None, :] - np.angle(z)[:, None]) - 1.0))
        s_c, s_alt = (I * k).max(-1), (I * (1.0 - k)).max(-1)
        x = (s_alt - s_c) / np.maximum(s_alt + s_c, 1e-9)
        c = np.where(np.abs(z) > 1e-6, np.clip((x - p["theta_c"]) / X_W, 0.0, 1.0), 0.0)
        self.m += (1.0 - np.exp(-dt / p["tau_m_s"])) * (c - self.m)
        g = (self.g_hi + (self.g_lo - self.g_hi) * self.m)[:, None]
        n = max(1, int(np.ceil(dt / p["tau_s"] * (1.0 + self.g_hi * C_INH))))
        h = dt / p["tau_s"] / n
        for _ in range(n):
            self.r += h * (np.clip((self.r @ W_RING) * g + I, 0.0, 1.0) - self.r)
        heading = _a(obs["heading"], self.B)
        return {"heading_cmd": wrap(v1._pv(self.r, heading)), "commit": self.r.max(-1),
                "viz": {"ring": self.r.copy(), "drives": d, "gain": g[:, 0].copy(), "conflict": self.m.copy()}}


CLASSES = {"ring_gated": RingGated, "ring_vg": RingVG, "ring_static_vg": RingStaticVG, "vote_vg": VoteVG, "sum_vg": SumVG,
           "accumulator_vg": AccumulatorVG, **v1.CLASSES}


def make(name, params, batch, rng=None):
    return CLASSES[name](dict(params), batch, rng)


def sample_params(name, u):
    """Map uniforms u to a config from PARAM_SPACE (same rule as v1.sample_params)."""
    out = {}
    for x, (k, (lo, hi, sc)) in zip(u, CLASSES[name].PARAM_SPACE.items()):
        out[k] = float(np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo)))) if sc == "log" else float(lo + x * (hi - lo))
    return out


def record_obs(T=400, K=3, seed=0):
    """A fixed obs sequence (batch 1) for the Python/JS parity check: two cues that swap evidence, a third that
    flickers, a slot occluded for 2 s, a search drive, and the home drive switched on for the last 2 s."""
    rng = np.random.default_rng(seed)
    seq = []
    for t in range(T):
        f = min(max((t - 120) / 80, 0.0), 1.0)
        s = np.clip(np.array([0.9 - 0.7 * f, 0.3 + 0.6 * f, 0.5 * (t % 37 < 9)]) + rng.normal(0, 0.08, K), 0, 1)
        vis = np.array([True, not 230 <= t < 330, t % 37 < 9])
        seq.append({"objects": {"bearing": [float(x) for x in wrap(np.array([1.0, -1.2, 2.5]) + 0.002 * t)],
                                "strength": [float(x) for x in s * vis], "visible": [bool(v) for v in vis]},
                    "home": {"bearing": 2.9, "strength": float(t >= T - 100)},
                    "search": {"bearing": -2.0, "strength": float(t < T - 100)}, "heading": float(wrap(0.01 * t))})
    return seq


def batch1(o):
    """Recorded batch-1 obs (lists/scalars) -> numpy obs with B = 1."""
    return {"objects": {k: np.array([v]) for k, v in o["objects"].items()},
            "home": {k: np.array([v]) for k, v in o["home"].items()},
            "search": {k: np.array([v]) for k, v in o["search"].items()}, "heading": np.array([o["heading"]])}


PARITY_PARAMS = {"ring_gated": {"ring_gain": 2.0, "kappa": 3.0, "tau_s": 0.15, "lo_frac": 0.4, "tau_m_s": 0.3, "theta_c": 0.1,
                                "tau_hold_s": 1.5},
                 "ring_vg": {"ring_gain": 1.2, "tau_hold_s": 0.8}, "ring_static_vg": {"tau_hold_s": 3.0},
                 "vote_vg": {"tau_hold_s": 1.0}, "sum_vg": {"tau_hold_s": 1.0, "p_obj": 2.0}, "accumulator_vg": {"tau_hold_s": 5.0}}


def parity_js(tol=1e-9):
    """Run the JS twin (app/sim/brain/arbiters_v2.js, batch 1) on a recorded obs sequence; max |Python - JS|."""
    import json, subprocess, tempfile
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    seq = record_obs()
    py = {}
    for name, prm in PARITY_PARAMS.items():
        a = make(name, prm, 1)
        py[name] = []
        for o in seq:
            out = a.step(batch1(o), 0.02)
            py[name].append([float(out["heading_cmd"][0]), float(out["commit"][0])] +
                            ([float(out["viz"]["gain"][0]), float(out["viz"]["conflict"][0])] if name == "ring_gated" else []))
    with tempfile.TemporaryDirectory() as td:
        spec, res = Path(td) / "spec.json", Path(td) / "out.json"
        spec.write_text(json.dumps({"params": PARITY_PARAMS, "obs": seq}))
        js = ("import { readFileSync, writeFileSync } from 'node:fs';"
              f"import {{ make }} from '{(root / 'app/sim/brain/arbiters_v2.js').as_uri()}';"
              "const s = JSON.parse(readFileSync(process.argv[1], 'utf8')); const out = {};"
              "for (const [n, p] of Object.entries(s.params)) { const a = make(n, p);"
              " out[n] = s.obs.map((o) => { const r = a.step(o, 0.02);"
              " return [r.heading_cmd, r.commit].concat(n === 'ring_gated' ? [r.viz.gain, r.viz.conflict] : []); }); }"
              "writeFileSync(process.argv[2], JSON.stringify(out));")
        subprocess.run(["node", "--input-type=module", "-e", js, str(spec), str(res)], check=True)
        jsout = json.loads(res.read_text())
    diff = {n: float(np.max(np.abs(wrap(np.array(py[n]) - np.array(jsout[n]))))) for n in py}
    return diff, all(v <= tol for v in diff.values())


if __name__ == "__main__":
    dt, B = 0.02, 3
    ob = v1._obs
    ang = lambda a, b: np.abs(wrap(np.asarray(a) - b))
    assert abs(G_CRIT - 0.22877692) < 1e-6 and abs(np.exp(K_C * (np.cos(np.pi / 4) - 1)) - 0.5) < 1e-12
    for name, c in CLASSES.items():
        p = sample_params(name, np.full(len(c.PARAM_SPACE), 0.5))
        assert all(lo <= p[k] <= hi for k, (lo, hi, _) in c.PARAM_SPACE.items()) and set(getattr(c, "DEFAULTS", c.PARAM_SPACE)) == set(c.PARAM_SPACE), name
    assert len(RingGated.PARAM_SPACE) == len(v1.CLASSES["state_machine"].PARAM_SPACE) == 10
    # first parameters of every _vg arm are the v1 arm's, in order (the shared random-search matrix maps the same way)
    for n in ("ring", "ring_static", "vote", "sum", "accumulator"):
        assert list(CLASSES[n + "_vg"].PARAM_SPACE)[:-1] == list(v1.CLASSES[n].PARAM_SPACE)

    # 1) ring_gain = 0: RingGated == ring_static_vg exactly, on a random stimulus sequence with occlusions
    rng = np.random.default_rng(0)
    pr = {"kappa": 3.0, "g_obj": 1.2, "g_home": 0.8, "g_search": 0.4, "tau_s": 0.15, "tau_hold_s": 1.0}
    a, b = make("ring_gated", {**pr, "ring_gain": 0.0, "lo_frac": 0.7, "tau_m_s": 0.2, "theta_c": 0.1}, B), make("ring_static_vg", pr, B)
    for t in range(200):
        o = ob(B, 2, rng.uniform(-np.pi, np.pi, (B, 2)), rng.uniform(0, 1, (B, 2)), rng.uniform(size=(B, 2)) > 0.3,
               rng.uniform(0, 1, B), rng.uniform(-3, 3, B), 1.0, rng.uniform(-3, 3, B), rng.uniform(-3, 3, B))
        oa, obb = a.step(o, dt), b.step(o, dt)
        assert np.array_equal(oa["heading_cmd"], obb["heading_cmd"]) and np.array_equal(a.r, b.r)
    # 2) with a huge tau_hold the _vg wrapper is the v1 arm whenever everything is visible
    for n in ("ring", "vote", "sum", "accumulator"):
        x, y = make(n, {}, B), make(n + "_vg", {"tau_hold_s": 20.0}, B)
        for t in range(50):
            o = ob(B, 2, rng.uniform(-3, 3, (B, 2)), rng.uniform(0, 1, (B, 2)), True, search_s=1.0, search_b=0.3)
            assert np.array_equal(x.step(o, dt)["heading_cmd"], y.step(o, dt)["heading_cmd"]), n

    # 3) visibility gate: a hidden object keeps pulling (decaying at tau_hold); a visible object at strength 0 does not
    def run(name, prm, phases):
        arb = make(name, prm, 1)
        for n, kw in phases:
            for _ in range(n):
                out = arb.step(ob(1, 1, 1.0, **kw, search_s=1.0, search_b=-1.5), dt)
        return out
    occ = [(150, {"stren": 0.8, "vis": True}), (100, {"stren": 0.0, "vis": False})]
    fade = [(150, {"stren": 0.8, "vis": True}), (100, {"stren": 0.0, "vis": True})]
    for n in ("vote_vg", "accumulator_vg"):
        assert ang(run(n, {"tau_hold_s": 10.0, "g_search": 0.3}, occ)["heading_cmd"], 1.0)[0] < np.deg2rad(5), n
        assert ang(run(n, {"tau_hold_s": 10.0, "g_search": 0.3, **({"tau_leak_s": 0.1} if "acc" in n else {})},
                       fade)["heading_cmd"], -1.5)[0] < np.deg2rad(5), n
    assert ang(run("vote", {"g_search": 0.3}, occ)["heading_cmd"], -1.5)[0] < np.deg2rad(5)  # v1: forgets at once
    s_occ = ang(run("sum_vg", {"tau_hold_s": 10.0, "g_search": 0.3}, occ)["heading_cmd"], 1.0)[0]
    assert s_occ < np.deg2rad(30) and ang(run("sum_vg", {"tau_hold_s": 10.0, "g_search": 0.3}, fade)["heading_cmd"], -1.5)[0] < 1e-9

    # 4) RingGated holds a hidden target (high gain, no conflict) ...
    hi = {"ring_gain": 2.5, "kappa": 4.0, "g_obj": 1.5, "g_search": 0.3, "tau_s": 0.2, "lo_frac": 0.5, "tau_m_s": 0.3,
          "theta_c": 0.2, "tau_hold_s": 20.0}
    out = run("ring_gated", hi, occ + [(400, {"stren": 0.0, "vis": False})])
    assert ang(out["heading_cmd"], 1.0)[0] < np.deg2rad(5) and out["viz"]["gain"][0] > 2.0
    # ... but releases when the evidence turns (a visible rival cue), where the v1 ring at the same gain latches
    def swap(name, prm):
        arb = make(name, prm, 1)
        for t in range(600):
            s = [0.9, 0.1] if t < 150 else [0.1, 0.9]
            out = arb.step(ob(1, 2, [[1.0, -1.5]], [s], True), dt)
        return out
    g_out = swap("ring_gated", hi)
    v_out = swap("ring", {k: hi[k] for k in v1.Ring.DEFAULTS if k in hi})
    assert ang(g_out["heading_cmd"], -1.5)[0] < np.deg2rad(5), np.rad2deg(g_out["heading_cmd"])
    assert ang(v_out["heading_cmd"], 1.0)[0] < np.deg2rad(5), "v1 ring at gain 2.5 should latch here"
    assert g_out["viz"]["conflict"][0] < 0.5 and g_out["viz"]["gain"][0] > 1.0   # re-committed after the reset
    # 5) bounded gain: at g_lo (< G_CRIT) a bump with no input dies out; at g_hi it persists
    for g, alive in ((0.999 * G_CRIT, False), (2.0, True)):
        r = np.zeros((1, N_RING)); r[0, :4] = 1.0
        for _ in range(5000):
            r += 0.1 * (np.clip(g * r @ W_RING, 0, 1) - r)
        assert (r.max() > 0.5) == alive, (g, r.max())
    # 6) no drive -> keep current heading
    out = make("ring_gated", {}, 1).step(ob(1, 1, 0.0, 0.0, False, heading=0.7), dt)
    assert ang(out["heading_cmd"], 0.7)[0] < 1e-9
    # 7) JS twin agrees on a recorded obs sequence
    diff, ok = parity_js()
    print("JS parity max |diff|:", {k: f"{v:.1e}" for k, v in diff.items()})
    assert ok, diff
    print(f"arbiters_v2 self-check OK (G_CRIT {G_CRIT:.4f}, K_C {K_C:.3f})")
