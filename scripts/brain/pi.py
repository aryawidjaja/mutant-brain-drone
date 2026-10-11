"""Central-complex path integrator, reimplemented from the text of Stone et al. 2017, Curr Biol 27:3069
(https://pmc.ncbi.nlm.nih.gov/articles/PMC6196076/). Their code has no licence and was not used or opened.

Layers (batched over B; identical in app/sim/brain/pi.js), angles CCW from +x (east):
  TL   16 cells, 8 preferred directions twice over: I_TL = cos(theta_TL - theta_h)            (paper, verbatim form)
  CL1  16 cells inhibited by TL ("effectively inverting the polarization response")
  TB1  8 cells: I_TB1,j = (1 - c) r_CL1,j + c sum_i W_ij r_TB1,i(t-1), W_ij = (cos(dtheta) - 1) / 2, c = 0.33  (paper)
  TN2  2 cells (left/right), I_TN = v . [direction theta_h +/- phi_TN], phi_TN = pi/4                (paper)
  CPU4 16 cells (8 per hemisphere), initial charge 0.5, clipped to [0, 1]                           (paper)
Activation r = 1 / (1 + exp(-(a I - b))) per layer (paper's form). The paper's slopes/offsets are in its Figure S5,
which we did not open: ours (A_, B_ below) were chosen so each layer spans its range.

Deviations from the printed equations (stated in the paper and app):
  1. CPU4 update. The paper prints I_CPU4(t) = I_CPU4(t-1) + h (r_TN2 - r_TB1 - k), but its prose says the TB1
     response is "modulating the speed signal from the TN2-neurons". The additive form integrates heading over
     time rather than velocity, so we use the multiplicative form the prose describes:
        dI_i = h dt r_TN2 ((1 - r_TB1,i) - mean_j(1 - r_TB1,j))
     The mean term is a speed-proportional uniform loss (it stops all cells charging together); the paper's
     constant loss k is set to 0 (missions are <= 300 s). h is ours (per second, units differ from the paper).
  2. TN2 is rectified-linear in speed, r_TN2 = clip(I_TN / V_REF, 0, 1); the printed max(0, 2 I_TN - 1) assumes
     the paper's arbitrary velocity units.
  3. Readout. Stone steers with CPU1 cells (CPU4 minus TB1 and pontine input) and never decodes a vector. Our
     steering hub (the arbiter) wants a home drive, so we read the CPU4 population vector directly, rotating
     each hemisphere by its TN offset (+/- pi/4) so that holonomic (sideways) motion is integrated:
        h = G * sum_i [I_L,i e^{i(theta_i + pi/4)} + I_R,i e^{i(theta_i - pi/4)}]
     G (complex: scale and sign) is fitted once by calibrate() and frozen below. CPU1/pontine not modelled.
Inputs: theta_h = compass estimate (sun azimuth + Rng stream 2 noise, plus the false_sun offset), TL columns
compass_half silences the CL1 output of 8 of the 16 compass columns (4 adjacent directions x 2); v = body-frame velocity from optic flow (Rng stream 3 noise, speed_scale fault).
The heading the brain uses elsewhere is the TB1 population vector (so compass faults propagate everywhere).
"""
import numpy as np

N8 = 8
TH8 = np.arange(N8) * np.pi / 4
C_TB1 = 0.33
PHI_TN = np.pi / 4
V_REF = 4.0            # m/s, TN2 saturates here
H_CPU4 = 0.025         # 1/s
SIGMA_COMPASS = np.radians(2.0)
SIGMA_SPEED = 0.05     # m/s per component
A_ = {"TL": 3.0, "CL1": 3.0, "TB1": 5.0}
B_ = {"TL": 0.0, "CL1": -1.5, "TB1": 1.0}
W_TB1 = (np.cos(TH8[:, None] - TH8[None, :]) - 1) / 2
GAIN = complex(-109.81636878389264, 0.0)   # fitted by calibrate(); self-check asserts it still holds


def sig(x, layer):
    return 1.0 / (1.0 + np.exp(-(A_[layer] * x - B_[layer])))


class PathIntegrator:
    def __init__(self, B):
        self.B = B
        self.tb1 = np.zeros((B, N8))
        self.cpu4 = np.full((B, 2, N8), 0.5)       # [hemisphere L/R, direction]
        self.col_mask = np.ones((B, 2 * N8))        # 0 = silenced compass column (CL1 output)

    def step(self, theta_h, v_body, dt):
        """theta_h [B] compass estimate (rad); v_body [B,2] (forward, left) m/s estimate. -> heading estimate [B]."""
        th_tl = np.tile(TH8, 2)
        r_tl = sig(np.cos(th_tl[None, :] - theta_h[:, None]), "TL")
        r_cl1 = sig(-r_tl, "CL1") * self.col_mask
        cl1 = 0.5 * (r_cl1[:, :N8] + r_cl1[:, N8:])
        self.tb1 = sig((1 - C_TB1) * cl1 + C_TB1 * self.tb1 @ W_TB1, "TB1")
        f, l = v_body[:, 0], v_body[:, 1]
        tn = np.stack([np.cos(PHI_TN) * f + np.sin(PHI_TN) * l, np.cos(PHI_TN) * f - np.sin(PHI_TN) * l], 1)
        r_tn = np.clip(tn / V_REF, 0.0, 1.0)                                     # [B,2]
        drive = (1 - self.tb1) - (1 - self.tb1).mean(-1, keepdims=True)          # [B,8]
        self.cpu4 = np.clip(self.cpu4 + H_CPU4 * dt * r_tn[:, :, None] * drive[:, None, :], 0.0, 1.0)
        return self.heading()

    def heading(self):
        """TB1 population vector (CL1 inverts TL, so TB1 peaks opposite the heading: add pi)."""
        z = (self.tb1 * np.exp(1j * TH8)).sum(-1)
        return np.angle(-z)

    def raw_vector(self):
        off = np.stack([TH8 + PHI_TN, TH8 - PHI_TN])                            # [2,8]
        return (self.cpu4 * np.exp(1j * off)).sum((-1, -2))

    def home_vector(self):
        """Decoded home vector [B,2] in the compass frame (m)."""
        z = GAIN * self.raw_vector()
        return np.stack([z.real, z.imag], -1)


def calibrate(T=20.0, dt=0.02, speed=2.0):
    """Fly straight 40 m in 16 headings with perfect inputs; least-squares complex gain home_true / raw."""
    hs = np.arange(16) * np.pi / 8
    pi_ = PathIntegrator(16)
    for _ in range(int(round(T / dt))):
        pi_.step(hs, np.tile([speed, 0.0], (16, 1)), dt)
    raw = pi_.raw_vector()
    true = -speed * T * np.exp(1j * hs)
    return complex((np.conj(raw) * true).sum() / (np.abs(raw) ** 2).sum())


if __name__ == "__main__":
    g = calibrate()
    assert abs(g - GAIN) / abs(GAIN) < 1e-9, f"GAIN stale: fitted {g!r}"
    dt = 0.02
    # 1) heading: TB1 decode tracks the compass within 3 deg
    pi_ = PathIntegrator(4)
    th = np.array([0.3, -2.0, 2.9, 1.2])
    for _ in range(20):
        hd = pi_.step(th, np.zeros((4, 2)), dt)
    assert np.all(np.abs((hd - th + np.pi) % (2 * np.pi) - np.pi) < np.radians(3)), np.degrees(hd)
    # 2) out-and-back loop with turns and sideslip: home error < 5 % of the path length
    rs = np.random.default_rng(0)
    B = 8
    pi_ = PathIntegrator(B)
    pos, yaw = np.zeros((B, 2)), rs.uniform(-np.pi, np.pi, B)
    L = 0.0
    for k in range(2500):                       # 50 s of wandering flight with sideslip
        yaw += rs.normal(0, 0.05, B)
        vb = np.stack([np.full(B, 2.0), 0.5 * np.sin(k * 0.05 + np.arange(B))], 1)
        c, s = np.cos(yaw), np.sin(yaw)
        pos += dt * np.stack([c * vb[:, 0] - s * vb[:, 1], s * vb[:, 0] + c * vb[:, 1]], 1)
        L += dt * np.hypot(*vb.T).mean()
        pi_.step(yaw, vb, dt)
    err = np.linalg.norm(pi_.home_vector() - (-pos), axis=1)
    assert err.max() < 0.05 * L, (err, L, np.linalg.norm(pos, axis=1))
    # 3) compass_half: silencing half the TL columns biases the heading for some directions
    pi_ = PathIntegrator(16)
    pi_.col_mask[:, [0, 1, 2, 3, 8, 9, 10, 11]] = 0
    hs = np.arange(16) * np.pi / 8
    for _ in range(20):
        hd = pi_.step(hs, np.zeros((16, 2)), dt)
    bias = np.abs((hd - hs + np.pi) % (2 * np.pi) - np.pi)
    assert bias.max() > np.radians(10), np.degrees(bias)
    print(f"pi self-check OK; GAIN {GAIN}; loop home error max {err.max():.2f} m over {L:.0f} m path; "
          f"compass_half max bias {np.degrees(bias.max()):.0f} deg")
