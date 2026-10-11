"""Heading estimators that fuse a yaw-rate gyro with the sun compass (batched; app/sim/brain/heading.js is the
batch-1 twin). One interface for all three:
    est = Cls(B, params);  heading[B] = est.step(gyro_z[B] rad/s, sun_theta[B] rad, sun_valid[B] bool, dt);  est.conf[B]
Angles CCW from +x (east). Every estimator snaps to the first valid sun reading. conf = 1 s low-pass of "the sun
cue was accepted this tick" (same definition for all three). Params may be scalars or [B] arrays (batched tuning).

HeadingRing: EPG-like ring attractor (fly head-direction circuit; Seelig & Jayaraman 2015, Kim et al. 2017,
  Turner-Evans et al. 2017, Green et al. 2017).
  - N_RING rate neurons with preferred directions phi_i; the bump is sinusoidal, e_i = 1 + cos(phi_i - heading).
  - PEN-like shift: P-ENs carry the bump with a one-neuron offset and are driven by angular velocity on one side.
    Here the bump is mixed with its copy rotated by one neuron toward the turn, weight s = sin x / (sin x + sin(D - x)),
    x = |gyro| dt, D = 2 pi / N, which rotates the bump's first harmonic by exactly x (a calibrated PEN gain).
  - Attractor: recurrent cosine weights W_ij = (2/N) cos(phi_i - phi_j) and divisive normalisation (global
    inhibition) restore the bump shape and amplitude every tick.
  - Sun cue (ring-neuron input): cosine profile at (sun - o) with weight w_sun dt. Cue-conflict gate: the cue is used
    only while |wrap(sun - o - heading)| <= gate_deg, so an abrupt false sun is ignored while slow gyro drift (small
    conflict) is still corrected.
  - Cue-map plasticity (Fisher et al. 2019): when the conflict has lasted longer than remap_delay_s, the cue-to-ring
    offset o relaxes toward it with time constant remap_tau_s (cue ignored meanwhile) until the conflict is below
    REMAP_END * gate_deg, then o takes the residual too; the cue then corrects gyro drift again in the new frame.
ComplementaryFilter: gyro integration + k_sun * clip(innovation, +/- clamp_deg); innovations beyond reject_deg ignored.
KalmanHeading: states (heading, gyro bias); random-walk process noise q_psi (gyro), q_bias (bias); sun noise r_sun_deg;
  measurement ignored when y^2 / S > chi2. Prior P0 = diag((5 deg)^2, (0.5 deg/s)^2).
"""
import numpy as np

N_RING = 32
PHI = 2 * np.pi * np.arange(N_RING) / N_RING
D_RING = 2 * np.pi / N_RING
W_REC = (2.0 / N_RING) * np.cos(PHI[:, None] - PHI[None, :])
COS_PHI, SIN_PHI = np.cos(PHI), np.sin(PHI)
TAU_CONF = 1.0
DEG = np.pi / 180
P0_PSI, P0_BIAS = (5 * DEG) ** 2, (0.5 * DEG) ** 2
REMAP_END = 0.2


def wrap(a):
    """Angle to [-pi, pi); same operations in heading.js."""
    return a - 2 * np.pi * np.floor((a + np.pi) / (2 * np.pi))


def _bump(th):
    """[B] -> [B, N_RING] sinusoidal bump 1 + cos(phi_i - th), written without per-neuron trig."""
    return 1.0 + COS_PHI[None, :] * np.cos(th)[:, None] + SIN_PHI[None, :] * np.sin(th)[:, None]


def _p(params, k):
    return np.asarray(params[k], float)


class _Base:
    PARAM_SPACE = {}

    def __init__(self, B, params):
        self.B = B
        self.p = {k: _p(params, k) for k in self.PARAM_SPACE}
        self.init = np.zeros(B, bool)
        self.conf = np.zeros(B)

    def _conf(self, acc, dt):
        self.conf = self.conf + dt / TAU_CONF * (acc - self.conf)


class HeadingRing(_Base):
    name = "ring"
    PARAM_SPACE = {"w_sun": (0.05, 5.0, "log"), "gate_deg": (5.0, 120.0, "lin"),
                   "remap_delay_s": (0.5, 30.0, "log"), "remap_tau_s": (1.0, 1000.0, "log")}

    def __init__(self, B, params):
        super().__init__(B, params)
        self.e = np.ones((B, N_RING)) + COS_PHI
        self.o = np.zeros(B)            # cue-to-ring offset (plastic)
        self.timer = np.zeros(B)        # how long the cue has been in conflict
        self.remap = np.zeros(B, bool)  # remapping in progress (cue ignored until the conflict is < REMAP_END * gate)

    def decode(self):
        return np.arctan2(self.e @ SIN_PHI, self.e @ COS_PHI)

    def step(self, gyro_z, sun_theta, sun_valid, dt):
        e = self.e
        x = np.minimum(np.abs(gyro_z) * dt, D_RING)
        s = np.sin(x) / (np.sin(x) + np.sin(D_RING - x))
        rot = np.where((gyro_z >= 0)[:, None], np.roll(e, 1, axis=1), np.roll(e, -1, axis=1))
        e = (1 - s)[:, None] * e + s[:, None] * rot                       # PEN shift
        self.e = e
        new = sun_valid & ~self.init
        self.e = np.where(new[:, None], _bump(sun_theta), self.e)
        self.o = np.where(new, 0.0, self.o)
        self.remap &= ~new
        self.init |= sun_valid
        psi = self.decode()
        c = np.abs(wrap(sun_theta - self.o - psi))
        gate = self.p["gate_deg"] * DEG
        agree = sun_valid & (c <= gate)
        self.timer = np.where(agree, 0.0, np.where(sun_valid, self.timer + dt, self.timer))
        done = self.remap & sun_valid & (c <= REMAP_END * gate)
        self.o = np.where(done, wrap(sun_theta - psi), self.o)            # remap complete: absorb the residual
        self.remap = (self.remap | (sun_valid & (self.timer > self.p["remap_delay_s"]))) & ~done
        acc = agree & ~self.remap
        self.o = np.where(self.remap & sun_valid,
                          wrap(self.o + dt / self.p["remap_tau_s"] * wrap(sun_theta - psi - self.o)), self.o)
        cue = _bump(sun_theta - self.o)
        u = self.e + (acc * self.p["w_sun"] * dt)[:, None] * cue
        a = u @ W_REC                                                      # recurrent cosine projection
        amp = np.sqrt(2.0 * (a * a).mean(1))
        self.e = 1.0 + a / amp[:, None]                                    # global inhibition: fixed amplitude
        self._conf(acc, dt)
        return self.decode()


class ComplementaryFilter(_Base):
    name = "cf"
    PARAM_SPACE = {"k_sun": (0.05, 5.0, "log"), "clamp_deg": (1.0, 90.0, "log"), "reject_deg": (5.0, 180.0, "lin")}

    def __init__(self, B, params):
        super().__init__(B, params)
        self.psi = np.zeros(B)

    def step(self, gyro_z, sun_theta, sun_valid, dt):
        psi = wrap(self.psi + gyro_z * dt)
        new = sun_valid & ~self.init
        psi = np.where(new, sun_theta, psi)
        self.init |= sun_valid
        y = wrap(sun_theta - psi)
        acc = sun_valid & (np.abs(y) <= self.p["reject_deg"] * DEG)
        cl = self.p["clamp_deg"] * DEG
        self.psi = np.where(acc, wrap(psi + self.p["k_sun"] * dt * np.clip(y, -cl, cl)), psi)
        self._conf(acc, dt)
        return self.psi


class KalmanHeading(_Base):
    name = "kf"
    PARAM_SPACE = {"q_psi": (0.01, 2.0, "log"),       # gyro noise, deg / sqrt(s)
                   "q_bias": (1e-4, 0.1, "log"),      # bias random walk, deg/s / sqrt(s)
                   "r_sun_deg": (0.5, 20.0, "log"),   # sun compass noise, deg
                   "chi2": (1.0, 100.0, "log")}       # innovation gate on y^2 / S

    def __init__(self, B, params):
        super().__init__(B, params)
        self.psi, self.b = np.zeros(B), np.zeros(B)
        self.p00, self.p01, self.p11 = np.full(B, P0_PSI), np.zeros(B), np.full(B, P0_BIAS)

    def step(self, gyro_z, sun_theta, sun_valid, dt):
        qp, qb = (self.p["q_psi"] * DEG) ** 2, (self.p["q_bias"] * DEG) ** 2
        psi = wrap(self.psi + (gyro_z - self.b) * dt)
        p00 = self.p00 - 2 * dt * self.p01 + dt * dt * self.p11 + qp * dt
        p01 = self.p01 - dt * self.p11
        p11 = self.p11 + qb * dt
        new = sun_valid & ~self.init
        psi = np.where(new, sun_theta, psi)
        b = np.where(new, 0.0, self.b)
        p00, p01, p11 = np.where(new, P0_PSI, p00), np.where(new, 0.0, p01), np.where(new, P0_BIAS, p11)
        self.init |= sun_valid
        y = wrap(sun_theta - psi)
        S = p00 + (self.p["r_sun_deg"] * DEG) ** 2
        acc = sun_valid & (y * y / S <= self.p["chi2"])
        k0, k1 = p00 / S, p01 / S
        self.psi = np.where(acc, wrap(psi + k0 * y), psi)
        self.b = np.where(acc, b + k1 * y, b)
        self.p00 = np.where(acc, p00 - k0 * p00, p00)
        self.p11 = np.where(acc, p11 - k1 * p01, p11)
        self.p01 = np.where(acc, p01 - k0 * p01, p01)
        self._conf(acc, dt)
        return self.psi


CLASSES = {c.name: c for c in (HeadingRing, ComplementaryFilter, KalmanHeading)}


def sample_params(name, u):
    """Uniforms u [len(PARAM_SPACE)] (or [n, len]) -> params dict (values scalar or [n])."""
    u = np.asarray(u, float)
    out = {}
    for j, (k, (lo, hi, sc)) in enumerate(CLASSES[name].PARAM_SPACE.items()):
        x = u[..., j]
        out[k] = np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo))) if sc == "log" else lo + x * (hi - lo)
    return out


def js_parity(name, params, gyro, sun, valid, dt):
    """Run the JS twin in Node on one input sequence ([T] arrays); return max |heading| and |conf| difference."""
    import json, os, subprocess, tempfile
    est = CLASSES[name](1, {k: float(v) for k, v in params.items()})
    hp, cp = [], []
    for g, s, v in zip(gyro, sun, valid):
        hp.append(float(est.step(np.array([g]), np.array([s]), np.array([bool(v)]), dt)[0]))
        cp.append(float(est.conf[0]))
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    runner = os.path.join(root, "scripts", "parity_heading.mjs")
    with tempfile.TemporaryDirectory() as d:
        fi, fo = os.path.join(d, "in.json"), os.path.join(d, "out.json")
        json.dump({"name": name, "params": {k: float(v) for k, v in params.items()}, "dt": dt,
                   "gyro": list(map(float, gyro)), "sun": list(map(float, sun)), "valid": list(map(bool, valid))}, open(fi, "w"))
        subprocess.run(["node", runner, fi, fo], check=True)
        o = json.load(open(fo))
    dh = np.abs(wrap(np.array(o["heading"]) - np.array(hp))).max()
    dc = np.abs(np.array(o["conf"]) - np.array(cp)).max()
    return float(dh), float(dc)


if __name__ == "__main__":
    dt, T = 0.02, 6000                                    # 120 s
    rs = np.random.default_rng(0)
    t = np.arange(T) * dt
    rate = np.radians(60) * np.sin(2 * np.pi * t / 17) + np.radians(30) * np.sign(np.sin(2 * np.pi * t / 41))
    psi = np.cumsum(rate) * dt + 0.7
    bias = np.radians(0.5)
    gyro = rate + bias + np.radians(0.1) * rs.normal(size=T)
    valid = np.ones(T, bool)
    clean = wrap(psi + np.radians(2) * rs.normal(size=T))
    fs = clean + np.where(t >= 40, np.radians(90), 0.0)                      # false sun at 40 s
    mid = {n: sample_params(n, np.full(len(c.PARAM_SPACE), 0.5)) for n, c in CLASSES.items()}
    good = {"ring": {"w_sun": 0.5, "gate_deg": 25.0, "remap_delay_s": 3.0, "remap_tau_s": 5.0},
            "cf": {"k_sun": 0.5, "clamp_deg": 10.0, "reject_deg": 25.0},
            "kf": {"q_psi": 0.3, "q_bias": 0.003, "r_sun_deg": 2.0, "chi2": 16.0}}

    def run(name, params, sun, g=gyro, v=valid, B=1):
        est = CLASSES[name](B, params)
        out = np.array([est.step(np.full(B, g[k]), np.full(B, sun[k]), np.full(B, v[k]), dt) for k in range(T)])
        return out[:, 0], est

    def rms(h, a=0, b=T):
        return np.degrees(np.sqrt(np.mean(wrap(h - psi)[a:b] ** 2)))

    # 1) ring integration: pure gyro (no sun after the snap) tracks the rotation within 0.05 deg over 120 s
    v0 = np.zeros(T, bool); v0[0] = True
    h, _ = run("ring", good["ring"], wrap(psi), g=rate, v=v0)
    assert np.degrees(np.abs(wrap(h - psi)).max()) < 0.05, np.degrees(np.abs(wrap(h - psi)).max())
    for name in CLASSES:
        h, est = run(name, good[name], clean)
        assert rms(h, 250) < 2.0, (name, rms(h, 250))                  # 2) no fault: gyro bias corrected by the sun
        assert est.conf[0] > 0.9
        h, est = run(name, good[name], fs)
        print(f"{name}: no-fault RMS {rms(run(name, good[name], clean)[0], 250):.2f} deg; "
              f"false sun 90 deg RMS after onset {rms(h, 2000):.1f} deg")
        if name in ("ring", "kf"):                                         # 3) gate rejects an abrupt 90 deg false sun
            assert rms(h, 2000) < 15, (name, rms(h, 2000))
    h, _ = run("cf", {"k_sun": 0.5, "clamp_deg": 10.0, "reject_deg": 180.0}, fs)
    assert rms(h, 4000) > 60                                               # without a reject gate the CF follows it
    # 4) batched params == per-row runs
    for name, c in CLASSES.items():
        U = rs.random((3, len(c.PARAM_SPACE)))
        P = sample_params(name, U)
        est = c(3, P)
        for k in range(500):
            hb = est.step(np.full(3, gyro[k]), np.full(3, fs[k]), np.ones(3, bool), dt)
        for i in range(3):
            e1 = c(1, {k: v[i] for k, v in P.items()})
            for k in range(500):
                h1 = e1.step(gyro[k:k + 1], fs[k:k + 1], np.ones(1, bool), dt)
            assert abs(wrap(h1[0] - hb[i])) < 1e-12
    # 5) JS twin (Node) agrees within 1e-9 on faulty inputs with sun dropouts
    vd = rs.random(T) > 0.05
    for name in CLASSES:
        for p in (good[name], mid[name]):
            dh, dc = js_parity(name, p, gyro, fs, vd, dt)
            assert dh <= 1e-9 and dc <= 1e-9, (name, dh, dc)
    print("heading self-check OK (incl. JS parity <= 1e-9)")
