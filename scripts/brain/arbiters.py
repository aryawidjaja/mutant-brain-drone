"""The six heading arbiters. Numpy, vectorised over a leading batch axis B.

Every arm maps the same `obs` (objects, home, search drives; world-frame bearings from the
drone's own compass) to a heading command. Params are scalars shared by the batch.

ring         Sun, Yue & Mangan 2020 (eLife 9:e54026, Eqs 26-30; MIT code
             github.com/XuelongSun/InsectNavigationToolkitModelling, RingAttractorModel)-style
             rate ring: threshold-linear integration neurons with Gaussian recurrent
             excitation and a uniform inhibitory neuron, cues injected as bumps scaled by
             certainty, population-vector readout. Reimplemented, not copied. Differences
             from Sun (stated in the paper):
               * N = 32 neurons over world heading (Sun: 8 per hemisphere), excitation width
                 sigma = 30 deg (Sun's code: 130 deg on 8 neurons) so that a localised bump
                 can form on 32 neurons;
               * the inhibitory neuron is taken at quasi-steady state (Sun: tau_I = tau_E/20),
                 which makes its loop a linear -J_I*mean(r) term; ring_gain multiplies the
                 whole recurrent loop (E->E and E->I->E), so ring_gain = 0 is no recurrence;
               * activation clipped to [0, 1] (Sun: max(0, rho + c), unbounded) and no offsets;
               * state carries over between brain ticks (Sun resets the ring every call), which
                 is what gives it memory; tau_s is tunable (Sun: 5 ms).
             Constants J_E = 24, uniform-mode inhibition c = 8 were set on a noise-free
             calibration (persistence onset near ring_gain ~ 0.8-1) before any mission data.
ring_static  the same input stage, neurons, tau and readout with ring_gain fixed at 0.
vote         static population vote (DAMN/VFH-like): same von Mises input stage, no dynamics,
             argmax with 3-point circular parabolic interpolation. No memory.
sum          linear vector sum of drive vectors (strength^p for objects). No memory.
accumulator  leaky evidence accumulator: one leaky average per option (each object slot,
             home, search), steer to the argmax option at its last-seen bearing.
state_machine  blind-authored rival in brain/state_machine.py (imported, never edited here).

Bearings of hidden objects: arbiters that need them keep their own last-seen bearing; none
relies on obs bearings of invisible slots.
"""
import numpy as np

try:
    from brain.state_machine import StateMachine
except ImportError:  # run as a file from scripts/brain
    from state_machine import StateMachine

ARBITERS = ("ring", "ring_static", "vote", "sum", "accumulator", "state_machine")
N_RING = 32
SIGMA_DEG = 30.0
J_E = 24.0
C_INH = 8.0
TH = np.arange(N_RING) * 2 * np.pi / N_RING


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _kernel():
    d = np.abs(wrap(TH[:, None] - TH[None, :]))
    G = np.exp(-np.rad2deg(d) ** 2 / (2 * SIGMA_DEG ** 2))
    J_I = G[0].mean() * J_E + C_INH          # uniform mode of the loop = -C_INH per unit gain
    return (J_E * G - J_I) / N_RING          # W_EE - W_EI*W_IE folded (quasi-steady inhibition)


W_RING = _kernel()


def _a(x, B):
    return np.broadcast_to(np.asarray(x, float), (B,) if np.ndim(x) <= 1 else np.shape(x))


COS_TH, SIN_TH = np.cos(TH), np.sin(TH)


def _bump(mu, s, kappa):
    """s * exp(kappa(cos(th - mu) - 1)), shape s.shape + (N,); computed only where s > 0."""
    mu, s = np.broadcast_arrays(np.asarray(mu, float), np.asarray(s, float))
    out = np.zeros(s.shape + (N_RING,))
    m = s > 0
    if m.any():
        mm = mu[m][:, None]
        out[m] = s[m][:, None] * np.exp(kappa * (np.cos(mm) * COS_TH + np.sin(mm) * SIN_TH - 1.0))
    return out


def drive_inputs(obs, kappa, g_obj, g_home, g_search):
    """Shared input stage: each drive is a von Mises bump exp(kappa(cos(th-mu)-1)) on the ring,
    scaled by gain * strength. Returns dict of [B, N] arrays (objects, home, search)."""
    o = obs["objects"]
    s = np.asarray(o["strength"], float) * np.asarray(o["visible"], bool)
    return {"objects": g_obj * _bump(o["bearing"], s, kappa).sum(-2),
            "home": g_home * _bump(obs["home"]["bearing"], obs["home"]["strength"], kappa),
            "search": g_search * _bump(obs["search"]["bearing"], obs["search"]["strength"], kappa)}


def _pv(r, heading):
    z = (r * np.exp(1j * TH)).sum(-1)
    return np.where(np.abs(z) > 1e-6, np.angle(z), heading)


class Ring:
    name = "ring"
    PARAM_SPACE = {"kappa": (0.5, 8.0, "log"), "g_obj": (0.2, 3.0, "log"), "g_home": (0.2, 3.0, "log"),
                   "g_search": (0.05, 1.5, "log"), "tau_s": (0.05, 2.0, "log"), "ring_gain": (0.0, 3.0, "lin")}
    DEFAULTS = {"kappa": 2.0, "g_obj": 1.5, "g_home": 1.5, "g_search": 0.5, "tau_s": 0.2, "ring_gain": 1.0}

    def __init__(self, params, batch, rng=None):
        self.p = {**self.DEFAULTS, **params}
        assert set(self.p) == set(self.DEFAULTS), set(self.p) ^ set(self.DEFAULTS)
        self.B = batch
        self.r = np.zeros((batch, N_RING))
        self.W = self.p.get("ring_gain", 0.0) * W_RING

    def step(self, obs, dt):
        p = self.p
        d = drive_inputs(obs, p["kappa"], p["g_obj"], p["g_home"], p["g_search"])
        I = d["objects"] + d["home"] + d["search"]
        g = p.get("ring_gain", 0.0)
        n = max(1, int(np.ceil(dt / p["tau_s"] * (1.0 + g * C_INH))))  # h*(1+|lambda_min|) <= 1: no overshoot
        h = dt / p["tau_s"] / n
        for _ in range(n):
            self.r += h * (np.clip(self.r @ self.W + I, 0.0, 1.0) - self.r)
        heading = _a(obs["heading"], self.B)
        return {"heading_cmd": wrap(_pv(self.r, heading)), "commit": self.r.max(-1),
                "viz": {"ring": self.r.copy(), "drives": d}}


class RingStatic(Ring):
    name = "ring_static"
    PARAM_SPACE = {k: v for k, v in Ring.PARAM_SPACE.items() if k != "ring_gain"}
    DEFAULTS = {k: v for k, v in Ring.DEFAULTS.items() if k != "ring_gain"}


class Vote:
    name = "vote"
    PARAM_SPACE = {"kappa": (0.5, 8.0, "log"), "g_obj": (0.2, 3.0, "log"), "g_home": (0.2, 3.0, "log"),
                   "g_search": (0.05, 1.5, "log")}
    DEFAULTS = {"kappa": 2.0, "g_obj": 1.5, "g_home": 1.5, "g_search": 0.5}

    def __init__(self, params, batch, rng=None):
        self.p = {**self.DEFAULTS, **params}
        assert set(self.p) == set(self.DEFAULTS)
        self.B = batch

    def step(self, obs, dt):
        p = self.p
        d = drive_inputs(obs, p["kappa"], p["g_obj"], p["g_home"], p["g_search"])
        I = d["objects"] + d["home"] + d["search"]
        rows = np.arange(self.B)
        i = I.argmax(-1)
        ym, y0, yp = I[rows, (i - 1) % N_RING], I[rows, i], I[rows, (i + 1) % N_RING]
        den = ym - 2 * y0 + yp
        off = np.where(np.abs(den) > 1e-12, 0.5 * (ym - yp) / np.where(den == 0, 1, den), 0.0)
        cmd = TH[i] + np.clip(off, -0.5, 0.5) * 2 * np.pi / N_RING
        heading = _a(obs["heading"], self.B)
        return {"heading_cmd": wrap(np.where(y0 > 1e-9, cmd, heading)), "commit": np.clip(y0, 0, 1),
                "viz": {"ring": I, "drives": d}}


class Sum:
    name = "sum"
    PARAM_SPACE = {"g_obj": (0.2, 3.0, "log"), "g_home": (0.2, 3.0, "log"), "g_search": (0.05, 1.5, "log"),
                   "p_obj": (0.25, 4.0, "log")}
    DEFAULTS = {"g_obj": 1.5, "g_home": 1.5, "g_search": 0.5, "p_obj": 1.0}

    def __init__(self, params, batch, rng=None):
        self.p = {**self.DEFAULTS, **params}
        assert set(self.p) == set(self.DEFAULTS)
        self.B = batch

    def step(self, obs, dt):
        p, o = self.p, obs["objects"]
        s = np.asarray(o["strength"], float) * np.asarray(o["visible"], bool)
        vec = {"objects": p["g_obj"] * (s ** p["p_obj"] * np.exp(1j * np.asarray(o["bearing"], float))).sum(-1),
               "home": p["g_home"] * np.asarray(obs["home"]["strength"], float) * np.exp(1j * np.asarray(obs["home"]["bearing"], float)),
               "search": p["g_search"] * np.asarray(obs["search"]["strength"], float) * np.exp(1j * np.asarray(obs["search"]["bearing"], float))}
        z = vec["objects"] + vec["home"] + vec["search"]
        heading = _a(obs["heading"], self.B)
        return {"heading_cmd": wrap(np.where(np.abs(z) > 1e-9, np.angle(z), heading)), "commit": np.clip(np.abs(z), 0, 1),
                "viz": {"vectors": {k: np.stack([v.real, v.imag], -1) for k, v in vec.items()}}}


class Accumulator:
    name = "accumulator"
    PARAM_SPACE = {"g_obj": (0.2, 3.0, "log"), "g_home": (0.2, 3.0, "log"), "g_search": (0.05, 1.5, "log"),
                   "tau_leak_s": (0.05, 20.0, "log")}
    DEFAULTS = {"g_obj": 1.5, "g_home": 1.5, "g_search": 0.5, "tau_leak_s": 1.0}

    def __init__(self, params, batch, rng=None):
        self.p = {**self.DEFAULTS, **params}
        assert set(self.p) == set(self.DEFAULTS)
        self.B = batch
        self.a = None

    def step(self, obs, dt):
        p, o = self.p, obs["objects"]
        vis = np.asarray(o["visible"], bool)
        bear = np.asarray(o["bearing"], float)
        if self.a is None:
            self.a = np.zeros((self.B, bear.shape[1] + 2))
            self.last = bear.copy()
        self.last = np.where(vis, bear, self.last)
        e = np.concatenate([p["g_obj"] * np.asarray(o["strength"], float) * vis,
                            p["g_home"] * _a(obs["home"]["strength"], self.B)[:, None],
                            p["g_search"] * _a(obs["search"]["strength"], self.B)[:, None]], -1)
        self.a += (1.0 - np.exp(-dt / p["tau_leak_s"])) * (e - self.a)
        bearings = np.concatenate([self.last, _a(obs["home"]["bearing"], self.B)[:, None],
                                   _a(obs["search"]["bearing"], self.B)[:, None]], -1)
        k = self.a.argmax(-1)
        amax = self.a[np.arange(self.B), k]
        heading = _a(obs["heading"], self.B)
        return {"heading_cmd": wrap(np.where(amax > 1e-9, bearings[np.arange(self.B), k], heading)),
                "commit": amax / np.maximum(self.a.sum(-1), 1e-9), "viz": {"evidence": self.a.copy(), "choice": k}}


CLASSES = {"ring": Ring, "ring_static": RingStatic, "vote": Vote, "sum": Sum, "accumulator": Accumulator,
           "state_machine": StateMachine}
assert set(CLASSES) == set(ARBITERS)


def make(name, params, batch, rng=None):
    return CLASSES[name](dict(params), batch, rng)


def sample_params(name, u):
    """Map a vector of uniforms u (len = #params, in [0,1)) to a config from PARAM_SPACE."""
    out = {}
    for x, (k, (lo, hi, sc)) in zip(u, CLASSES[name].PARAM_SPACE.items()):
        out[k] = float(np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo)))) if sc == "log" else float(lo + x * (hi - lo))
    return out


def _obs(B, K, bear, stren, vis, home_s=0.0, home_b=0.0, search_s=0.0, search_b=0.0, heading=0.0):
    f = lambda x, sh: np.broadcast_to(np.asarray(x, float), sh).copy()
    return {"objects": {"bearing": f(bear, (B, K)), "strength": f(stren, (B, K)),
                        "visible": np.broadcast_to(np.asarray(vis, bool), (B, K)).copy()},
            "home": {"bearing": f(home_b, (B,)), "strength": f(home_s, (B,))},
            "search": {"bearing": f(search_b, (B,)), "strength": f(search_s, (B,))},
            "heading": f(heading, (B,))}


if __name__ == "__main__":
    dt, B = 0.02, 3
    ang = lambda a, b: np.abs(wrap(np.asarray(a) - b))
    counts = {k: len(c.PARAM_SPACE) for k, c in CLASSES.items()}
    print("param counts", counts)
    for name, c in CLASSES.items():
        assert c.name == name
        p = sample_params(name, np.full(len(c.PARAM_SPACE), 0.5))
        for k, (lo, hi, _) in c.PARAM_SPACE.items():
            assert lo <= p[k] <= hi

    # 1) ring_gain = 0 is exactly ring_static, on a random stimulus sequence
    rng = np.random.default_rng(0)
    pr = {"kappa": 3.0, "g_obj": 1.2, "g_home": 0.8, "g_search": 0.4, "tau_s": 0.15}
    a, b = make("ring", {**pr, "ring_gain": 0.0}, B), make("ring_static", pr, B)
    for t in range(200):
        o = _obs(B, 2, rng.uniform(-np.pi, np.pi, (B, 2)), rng.uniform(0, 1, (B, 2)), rng.uniform(size=(B, 2)) > 0.3,
                 rng.uniform(0, 1, B), rng.uniform(-3, 3, B), 1.0, rng.uniform(-3, 3, B), rng.uniform(-3, 3, B))
        oa, ob = a.step(o, dt), b.step(o, dt)
        assert np.array_equal(oa["heading_cmd"], ob["heading_cmd"]) and np.array_equal(a.r, b.r)

    # 2) single cue: every arm converges on its bearing (batch rows with different bearings)
    mu = np.array([0.4, -2.0, 2.9])
    for name in ARBITERS:
        arb = make(name, {}, B)
        for _ in range(150):
            out = arb.step(_obs(B, 1, mu[:, None], 0.8, True), dt)
        assert out["heading_cmd"].shape == (B,) and np.all((out["commit"] >= 0) & (out["commit"] <= 1 + 1e-9))
        assert ang(out["heading_cmd"], mu).max() < np.deg2rad(2), (name, np.rad2deg(out["heading_cmd"]))

    # 3) persistence: high gain holds the bump 5 s after input is removed; gain 0 and vote forget
    def hold(name, params):
        arb = make(name, params, 1)
        for _ in range(150):
            arb.step(_obs(1, 1, 1.0, 0.8, True, search_s=1.0, search_b=-1.5), dt)
        for _ in range(250):
            out = arb.step(_obs(1, 1, 1.0, 0.0, False, search_s=1.0, search_b=-1.5), dt)
        return ang(out["heading_cmd"], 1.0)[0]
    assert hold("ring", {"ring_gain": 2.5, "kappa": 4.0, "g_search": 0.3}) < np.deg2rad(5)
    assert hold("ring", {"ring_gain": 0.0, "kappa": 4.0, "g_search": 0.3}) > np.deg2rad(60)
    assert hold("vote", {"g_search": 0.3}) > np.deg2rad(60)
    assert hold("accumulator", {"tau_leak_s": 20.0, "g_search": 0.3}) < np.deg2rad(1)
    assert hold("accumulator", {"tau_leak_s": 0.1, "g_search": 0.3}) > np.deg2rad(60)
    # bump with no input at all persists at high gain, decays at low gain
    arb = make("ring", {"ring_gain": 2.0, "kappa": 4.0}, 1)
    for _ in range(150):
        arb.step(_obs(1, 1, 1.0, 0.8, True), dt)
    for _ in range(500):
        arb.step(_obs(1, 1, 1.0, 0.0, False), dt)
    assert arb.r.max() > 0.5
    # 4) vote interpolation is sub-bin accurate; sum of two equal cues 90 deg apart points between them
    v = make("vote", {"kappa": 4.0}, 1).step(_obs(1, 1, 0.123, 0.8, True), dt)
    assert ang(v["heading_cmd"], 0.123)[0] < np.deg2rad(1.0)
    s = make("sum", {}, 1).step(_obs(1, 2, [[0.0, np.pi / 2]], 0.8, True), dt)
    assert ang(s["heading_cmd"], np.pi / 4)[0] < 1e-9
    # 5) no drive at all -> every arm keeps the current heading
    for name in ("ring", "ring_static", "vote", "sum", "accumulator"):
        out = make(name, {}, 1).step(_obs(1, 1, 0.0, 0.0, False, heading=0.7), dt)
        assert ang(out["heading_cmd"], 0.7)[0] < 1e-9, name
    print("arbiters self-check OK")
