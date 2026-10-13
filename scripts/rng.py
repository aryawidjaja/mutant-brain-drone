"""Seeded PRNG shared bit-for-bit with app/sim/rng.js (mulberry32 + Box-Muller), vectorised over a batch.

Every stochastic stream (wind, sensor noise, world layout) takes its own stream id so that
Python and JS draw identical numbers in identical order. Simplification: mulberry32 is not
cryptographic and has a 2^32 period per stream; plenty for 300 s missions.
"""
import numpy as np

M32 = np.uint64(0xFFFFFFFF)


def stream_seed(seed, stream):
    """32-bit start state for (seed, stream); same formula in rng.js."""
    return (int(seed) * 0x9E3779B1 + int(stream) * 0x85EBCA77 + 0x6D2B79F5) & 0xFFFFFFFF


def _imul(a, b):
    return (a.astype(np.uint64) * b.astype(np.uint64)) & M32


class Rng:
    """Batch of independent mulberry32 generators. seeds: int or array of ints (one per batch row)."""

    def __init__(self, seeds, stream=0):
        seeds = np.atleast_1d(np.asarray(seeds, dtype=np.int64))
        self.a = np.array([stream_seed(s, stream) for s in seeds], dtype=np.uint64)

    def uniform(self):
        """One U[0,1) draw per batch row."""
        self.a = (self.a + np.uint64(0x6D2B79F5)) & M32
        t = self.a.copy()
        t = _imul(t ^ (t >> np.uint64(15)), t | np.uint64(1))
        t = (t ^ ((t + _imul(t ^ (t >> np.uint64(7)), t | np.uint64(61))) & M32)) & M32
        return ((t ^ (t >> np.uint64(14))) & M32).astype(np.float64) / 4294967296.0

    def normal(self):
        """One N(0,1) draw per batch row (Box-Muller, cosine branch only; consumes two uniforms)."""
        u1 = 1.0 - self.uniform()  # (0, 1]
        u2 = self.uniform()
        return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


if __name__ == "__main__":
    r = Rng([0, 1, 12345], stream=3)
    u = np.array([r.uniform() for _ in range(2000)])
    assert u.shape == (2000, 3) and (u >= 0).all() and (u < 1).all()
    assert abs(u.mean() - 0.5) < 0.02 and not np.allclose(u[:, 0], u[:, 1])
    n = np.array([Rng([7], 1).normal() for _ in range(1)])
    z = Rng(np.arange(4000), 2).normal()
    assert abs(z.mean()) < 0.06 and abs(z.std() - 1) < 0.06
    # reference values checked against app/sim/rng.js (node app/sim/rng.js prints the same)
    r = Rng([42], 0)
    ref = [float(r.uniform()[0]) for _ in range(3)] + [float(Rng([42], 5).normal()[0])]
    print(" ".join(f"{v:.12f}" for v in ref))
