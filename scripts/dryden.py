"""MIL-F-8785C low-altitude Dryden turbulence, discretised, batched, seeded by scripts/rng.py stream 1.

Formulas as documented in the MathWorks Aerospace Blockset "Dryden Wind Turbulence Model (Continuous)"
(see results/vehicle_params.json["dryden"] for quotes):
  Lw = h, Lu = Lv = h / (0.177 + 0.000823 h)^1.2      (h in ft)
  sigma_w = 0.1 W20, sigma_u = sigma_v = sigma_w / (0.177 + 0.000823 h)^0.4
  Hu = sigma_u sqrt(2Lu/(pi V)) / (1 + Tu s),  Tu = Lu/V
  Hv = sigma_v sqrt(Lv/(pi V)) (1 + sqrt3 Tv s) / (1 + Tv s)^2   (Hw likewise)
Discretisation (identical in app/sim/dryden.js): white noise n_k = N(0,1) * sqrt(pi/dt) (two-sided PSD pi,
which makes each filter's output variance sigma^2), held over the step; each 1/(1+Ts) stage is the exact
zero-order-hold update x <- a x + (1-a) in, a = exp(-dt/T); the lead (1+sqrt3 T s)/(1+T s) is
sqrt3*in + (1-sqrt3)*lag(in). Draw order per step: u, v, w (one normal() each). "none" draws nothing.
ponytail: u, v, w along world x, y, z (not rotated into the mean-wind direction).
"""
import numpy as np

from rng import Rng

FT = 0.3048
KT = 0.514444
W20_KT = {"none": 0.0, "light": 15.0, "moderate": 30.0, "severe": 45.0}
SQ3 = np.sqrt(3.0)


def dryden_scales(intensity, alt_m, airspeed):
    """(sigma [3] m/s, L [3] m) for u, v, w."""
    h = max(alt_m / FT, 10.0)  # ft, clamped (assumption: formulas degenerate at the ground)
    k = 0.177 + 0.000823 * h
    Lu = h / k ** 1.2 * FT
    sw = 0.1 * W20_KT[intensity] * KT
    su = sw / k ** 0.4
    return np.array([su, su, sw]), np.array([Lu, Lu, h * FT])


class Dryden:
    def __init__(self, B, seeds, intensity="none", mean=(0.0, 0.0, 0.0), alt_m=3.0, airspeed=2.0):
        self.B, self.intensity, self.V = B, intensity, float(airspeed)
        self.mean = np.broadcast_to(np.asarray(mean, float), (B, 3)).copy()
        self.sigma, self.L = dryden_scales(intensity, alt_m, airspeed)
        self.T = self.L / self.V
        self.gain = self.sigma * np.sqrt(np.array([2.0, 1.0, 1.0]) * self.L / (np.pi * self.V))
        self.rng = Rng(np.broadcast_to(np.atleast_1d(seeds), (B,)), stream=1)
        self.x1 = np.zeros((B, 3))  # first lag stage (u, v, w)
        self.x2 = np.zeros((B, 3))  # second lag stage (v, w only; column 0 unused)
        self.wind = self.mean.copy()

    def step(self, dt):
        if self.intensity == "none":
            return self.mean.copy()
        n = np.stack([self.rng.normal() for _ in range(3)], axis=1) * np.sqrt(np.pi / dt)
        a = np.exp(-dt / self.T)
        self.x1 = a * self.x1 + (1.0 - a) * n
        self.x2 = a * self.x2 + (1.0 - a) * self.x1
        turb = np.empty((self.B, 3))
        turb[:, 0] = self.gain[0] * self.x1[:, 0]
        turb[:, 1:] = self.gain[1:] * (SQ3 * self.x1[:, 1:] + (1.0 - SQ3) * self.x2[:, 1:])
        self.wind = self.mean + turb
        return self.wind.copy()


if __name__ == "__main__":
    s, L = dryden_scales("moderate", 3.0, 2.0)
    assert abs(s[2] - 0.1 * 30 * KT) < 1e-12 and s[0] > s[2] and L[0] > L[2]
    assert Dryden(2, [0, 1], "none", (1, 2, 0)).step(0.002).tolist() == [[1, 2, 0], [1, 2, 0]]
    # long run: sample std of each component within 15% of sigma (pooled over 64 seeds, 300 s)
    for inten in ("light", "severe"):
        d = Dryden(64, np.arange(64), inten, (0, 0, 0))
        dt = 0.01
        out = np.array([d.step(dt) for _ in range(30000)])[3000:]
        std = out.std(axis=(0, 1))
        assert np.all(np.abs(std / d.sigma - 1) < 0.15), (inten, std, d.sigma)
        assert np.abs(out.mean(axis=(0, 1))).max() < 0.25 * d.sigma.max()
        print(inten, "sigma", d.sigma.round(3), "sample std", std.round(3), "L", d.L.round(2))
    # determinism: same seed -> same wind; different seed -> different
    a, b, c = Dryden(1, [5], "light"), Dryden(1, [5], "light"), Dryden(1, [6], "light")
    wa, wb, wc = [np.array([x.step(0.002) for _ in range(100)]) for x in (a, b, c)]
    assert np.array_equal(wa, wb) and not np.allclose(wa, wc)
    print("dryden OK")
