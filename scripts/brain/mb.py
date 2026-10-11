"""Mushroom-body familiarity memory, Ardin et al. 2016 style (PLoS Comput Biol 12:e1004683).

Rate-based abstraction of their spiking model:
  PN layer   = the preprocessed image (n_in values, see preprocess()).
  KC layer   = n_kc Kenyon cells, each summing k_in PNs drawn uniformly at random (with replacement)
               ("20,000 Kenyon cells (KC), each receiving input from 10 randomly selected vPNs").
               k-winners-take-all keeps the top kc_active_frac (1% -> 200 KCs; Ardin: "around 200").
  KC->MBON   = binary weights, start at 1, set to 0 for every KC active in a learned view
               (one-shot; Ardin: "setting it to zero when KC activity coincides with reward").
  familiarity= 1 - (active KCs whose weight is still 1) / (active KCs). 1 = seen before, ~0 = novel.

JS port notes: the wiring is idx[j, i] = floor(u * n_in) for the uniforms of
`new Rng(seed, 10)` drawn in order j=0..n_kc-1, i=0..k_in-1 (row-major). KC drive = sum of x[idx[j, :]].
Active = drive >= the k-th largest drive (ties may admit a few extra KCs; same rule in JS).
"""
import numpy as np

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rng import Rng, stream_seed  # noqa: E402

MB_STREAM = 10


def preprocess(img, size=(32, 32)):
    """img [..., H, W, 3] uint8 or float (or [..., H, W] grey) -> x [..., h*w] float32.

    grey (channel mean) -> area downsample to `size` (H, W must be multiples) -> rank histogram
    equalisation to (0, 1] -> divide by the L2 norm (Ardin: "normalised by dividing each pixel value by
    the square root of the sum of squares"). Rank equalisation stands in for their CLAHE.
    """
    a = np.asarray(img, dtype=np.float32)
    if a.shape[-1] == 3 and a.ndim >= 3:
        a = a.mean(-1)
    H, W = a.shape[-2:]
    h, w = size
    a = a.reshape(*a.shape[:-2], h, H // h, w, W // w).mean((-3, -1)).reshape(*a.shape[:-2], h * w)
    return equalise(a)


def equalise(a):
    """Rank histogram equalisation along the last axis, then L2 normalisation."""
    r = (np.argsort(np.argsort(a, axis=-1, kind="stable"), axis=-1, kind="stable") + 1).astype(np.float32) / a.shape[-1]
    return r / np.linalg.norm(r, axis=-1, keepdims=True)


def wiring(n_in, n_kc, k_in, seed):
    """[n_kc, k_in] PN indices from scripts/rng.py stream 10 (bit-identical to sequential JS draws).

    mulberry32's state is a counter (state_n = state_0 + n*0x6D2B79F5), so all draws are made at once
    by seeding a batch whose rows sit at successive counter values.
    """
    n = n_kc * k_in
    r = Rng([seed], MB_STREAM)
    r.a = (np.uint64(stream_seed(seed, MB_STREAM)) + np.arange(n, dtype=np.uint64) * np.uint64(0x6D2B79F5)) & np.uint64(0xFFFFFFFF)
    return np.floor(r.uniform() * n_in).astype(np.int64).reshape(n_kc, k_in)


class MushroomBody:
    def __init__(self, n_in, n_kc=20000, k_in=10, kc_active_frac=0.01, seed=0):
        self.n_in, self.n_kc, self.k = n_in, n_kc, max(1, int(round(kc_active_frac * n_kc)))
        self.idx = wiring(n_in, n_kc, k_in, seed)
        W = np.zeros((n_in, n_kc), np.float32)            # dense PN->KC count matrix, for a fast matmul
        np.add.at(W, (self.idx, np.arange(n_kc)[:, None]), 1.0)
        self.W = W
        self.reset()

    def reset(self):
        self.w = np.ones(self.n_kc, bool)                 # KC->MBON weights (1 = not yet depressed)

    def active(self, x):
        """x [..., n_in] -> bool [..., n_kc] (k-winners-take-all)."""
        x = np.asarray(x, np.float32)
        d = x @ self.W
        thr = -np.partition(-d, self.k - 1, axis=-1)[..., self.k - 1:self.k]
        return d >= thr

    def learn(self, x):
        a = self.active(np.atleast_2d(x))
        self.w &= ~a.any(0)

    def familiarity(self, x):
        a = self.active(x)
        return 1.0 - (a & self.w).sum(-1) / a.sum(-1)

    @property
    def nbytes(self):
        return (self.n_kc + 7) // 8                       # 1 bit per KC->MBON synapse; wiring regenerated from seed


if __name__ == "__main__":
    rs = np.random.default_rng(0)
    imgs = rs.integers(0, 256, (6, 64, 64, 3), dtype=np.uint8)
    X = preprocess(imgs)
    assert X.shape == (6, 1024) and np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-5)
    mb = MushroomBody(1024, seed=0)
    assert mb.idx.shape == (20000, 10) and mb.idx.min() >= 0 and mb.idx.max() < 1024
    assert mb.active(X[0]).sum() >= 200 and mb.active(X[0]).sum() < 210
    f0 = mb.familiarity(X)
    assert (f0 == 0).all(), f0                            # nothing learned: everything novel
    mb.learn(X[0])
    f = mb.familiarity(X)
    assert f[0] == 1.0 and (f[1:] < 0.2).all(), f         # learned view ~1, unrelated views low
    mb.learn(X[1:3])
    f = mb.familiarity(X)
    assert (f[:3] == 1).all() and (f[3:] < 0.3).all(), f
    mb.reset(); assert (mb.familiarity(X) == 0).all()
    assert mb.nbytes == 2500
    # determinism + equality with sequential draws (what the JS port does)
    assert (MushroomBody(1024, seed=0).idx == mb.idx).all() and not (MushroomBody(1024, seed=1).idx == mb.idx).all()
    r = Rng([0], MB_STREAM)
    seq = [int(np.floor(r.uniform()[0] * 1024)) for _ in range(25)]
    assert seq == mb.idx.reshape(-1)[:25].tolist()
    print("mb self-check ok; first PN indices (seed 0):", seq[:10], "nbytes", mb.nbytes)
