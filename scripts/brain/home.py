"""Home view memory for the final approach (Study 2 brain, H6). JS twin: app/sim/brain/home.js.

Insect basis: "Many wasps and bees learn the position of their nest relative to nearby visual features during
elaborate 'learning' flights that they perform on leaving the nest" and "often both fly towards and face the nest in
the middle of loops and at the turns of zigzags" (Philippides et al. 2013, J Exp Biol 216:1093); the views recorded
close to the nest are later used to return to it (Collett & Zeil 2018, Curr Biol 28:R984).
Here, during a take-off learning flight the drone backs away from its landing pad in arcs while facing it, and every
look it takes of the pad is stored one-shot in a mushroom body (same Ardin-style MB and input pipeline as the target
memory; its own KC->MBON weights). On return, every visible candidate (the pad and any
object slot) is scored by this home memory; the most familiar candidate above F_HOME "locks" the pad. While a locked
candidate stays visible and at least F_TRACK familiar, the drone's home vector is replaced by the visual vector to it
(perceived bearing in the compass frame, horizontal range from the depression angle and the drone's altitude); when
it drops out of view the last visual correction is kept (home vector + offset), so path-integration drift that was
seen once stays corrected.

steer() returns (bearing_correction, familiarity) as in the v2 interface, plus the approach vector it implies.
"""
import numpy as np

F_HOME = 0.97          # familiarity needed to recognise (lock) the pad: results/v2_home_calib.json (train seeds 100-139)
F_TRACK = 0.485        # familiarity needed to keep using a locked candidate (F_HOME / 2)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class HomeMemory:
    """Batched home memory. act_fn(looks [N, n]) -> active KCs [N, n_kc] bool (the mission's KC coding)."""

    def __init__(self, B, act_fn, n_kc, f_home=F_HOME, f_track=F_TRACK):
        self.B, self.act_fn, self.f_home, self.f_track = B, act_fn, f_home, f_track
        self.w = np.ones((B, n_kc), bool)          # KC->MBON weights (one-shot depression)
        self.n_views = np.zeros(B, int)
        self.locked = np.zeros(B, bool)
        self.corr = np.zeros((B, 2))               # visual vector minus PI home vector, last seen (compass frame, m)

    def learn(self, rows, looks):
        """rows [N] (distinct), looks [N, n] noisy grey looks of the pad."""
        if len(rows):
            self.w[rows] &= ~self.act_fn(looks)
            self.n_views[rows] += 1

    def familiarity(self, rows, looks):
        act = self.act_fn(looks)
        return 1.0 - (act & self.w[rows]).sum(-1) / act.sum(-1)

    def steer(self, looks_now, vis, bear_c, rh, hv):
        """looks_now [B, C, n] candidate looks, vis [B, C], bear_c [B, C] perceived bearing (compass frame, rad),
        rh [B, C] horizontal range (m), hv [B, 2] path-integrator home vector (compass frame, m).
        -> bearing_correction [B] (rad), familiarity [B] (best visible candidate, 0 if none), approach vector [B, 2],
           seen [B] (the visual vector was used this tick)."""
        B, C = vis.shape
        rows = np.arange(B)
        fam = np.zeros((B, C))
        idx = np.argwhere(vis)
        if len(idx):
            fam[idx[:, 0], idx[:, 1]] = self.familiarity(idx[:, 0], looks_now[idx[:, 0], idx[:, 1]])
        best = np.argmax(np.where(vis, fam, -1.0), 1)
        fb = np.where(vis[rows, best], fam[rows, best], 0.0)
        self.locked |= vis[rows, best] & (fb >= self.f_home)
        seen = self.locked & vis[rows, best] & (fb >= self.f_track)
        b, r = bear_c[rows, best], rh[rows, best]
        vec = np.stack([r * np.cos(b), r * np.sin(b)], 1)
        self.corr = np.where(seen[:, None], vec - hv, self.corr)
        ap = np.where(self.locked[:, None], hv + self.corr, hv)
        corr_b = np.where(self.locked, wrap(np.arctan2(ap[:, 1], ap[:, 0]) - np.arctan2(hv[:, 1], hv[:, 0])), 0.0)
        return corr_b, fb, ap, seen


if __name__ == "__main__":
    # toy KC code: a look activates the KCs of its own index set; two distinct looks share no KCs
    def act_fn(x):
        a = np.zeros((len(x), 40), bool)
        for i, v in enumerate(x):
            a[i, int(v[0]) * 4:int(v[0]) * 4 + 4] = True
        return a
    hm = HomeMemory(2, act_fn, 40)
    hm.learn(np.array([0, 1]), np.array([[1.0], [2.0]]))
    assert np.allclose(hm.familiarity(np.array([0, 0]), np.array([[1.0], [3.0]])), [1.0, 0.0])
    hv = np.array([[3.0, 0.0], [3.0, 0.0]])
    looks = np.array([[[1.0], [5.0]], [[7.0], [2.0]]])          # row 0: pad in candidate 0; row 1: pad in candidate 1
    vis = np.array([[True, True], [False, False]])
    bear, rh = np.array([[np.pi / 2, 0.0], [0.0, 0.0]]), np.array([[2.0, 9.0], [4.0, 4.0]])
    cb, fb, ap, seen = hm.steer(looks, vis, bear, rh, hv)
    assert seen.tolist() == [True, False] and hm.locked.tolist() == [True, False]
    assert np.allclose(ap[0], [0.0, 2.0]) and np.allclose(ap[1], hv[1]) and abs(cb[0] - np.pi / 2) < 1e-12 and cb[1] == 0
    # the pad drops out of view: the correction persists (the PI vector moves, the offset is kept)
    cb, fb, ap, seen = hm.steer(looks, np.zeros((2, 2), bool), bear, rh, hv - [1.0, 0.0])
    assert not seen.any() and np.allclose(ap[0], [-1.0, 2.0]) and fb[0] == 0
    print("home self-check OK")
