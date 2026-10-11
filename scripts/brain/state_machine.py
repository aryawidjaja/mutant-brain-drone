"""Engineered state-machine arbiter (the "fair rival" baseline), written blind.

Written from the C2 interface spec only, before any fault data or competing
arbiter code was seen, then frozen (hash logged in docs/LOG.md).

States (viz["state"]):
    0 SEARCH    follow the exploration drive; nothing confirmed yet
    1 APPROACH  steer to the filtered bearing of the committed slot
    2 HOLD      committed slot hidden: steer to its last-known bearing until timeout_s
    3 HOME      home drive active (strength > 0): overrides everything

Engineering templates followed (standard practice, not tuned to any fault):
  * Behaviour-tree priority fallback (Colledanchise & Ogren 2018, "Behavior Trees
    in Robotics and AI", Fallback/Sequence nodes): HOME > keep committed goal
    (APPROACH, with HOLD as its recovery branch) > SEARCH.
  * Nav2-style recovery with a bounded budget (Macenski et al. 2020, Nav2
    BT navigator: on loss, run a recovery behaviour for a limited time, then
    fall back): HOLD_LAST_BEARING with timeout_s, then SEARCH.
  * DAMN-style command arbitration by winner-take-all on filtered votes
    (Rosenblatt 1997) and VFH-style low-pass/hysteresis on the selected
    direction (Borenstein & Koren 1991): per-slot evidence low-pass filter,
    Schmitt-trigger hysteresis (enter_thr / exit_thr), a multiplicative
    switching margin, and k-of-n debouncing of every discrete transition.
  * Track-management practice from multi-target tracking (M-of-N confirmation
    and deletion, Blackman & Popoli 1999): k-of-n confirmation to acquire,
    k-of-n to reject, and a reject cooldown so a dropped look-alike is not
    re-acquired straight away.
  * Evidence-regression check: a committed object whose filtered evidence falls
    below drop_frac of its peak since commitment (or below exit_thr) is
    rejected; this is how a look-alike whose evidence sinks up close is released.

Numpy only, vectorised over the leading batch axis B. Params are scalars shared
by the batch. Uses no randomness (rng accepted for interface parity).
"""
import numpy as np

SEARCH, APPROACH, HOLD, HOME = 0, 1, 2, 3
STATE_NAMES = ("SEARCH", "APPROACH", "HOLD", "HOME")

DEFAULTS = {
    "ev_tau_s": 0.3,        # evidence low-pass time constant
    "bearing_tau_s": 0.1,   # bearing low-pass time constant (circular)
    "enter_thr": 0.5,       # filtered evidence needed to acquire / challenge
    "exit_ratio": 0.7,      # exit_thr = exit_ratio * enter_thr (hysteresis band)
    "debounce_s": 0.2,      # k-of-n window length (n = debounce_s / dt)
    "debounce_frac": 0.7,   # k = ceil(debounce_frac * n)
    "switch_margin": 0.25,  # challenger must exceed (1 + margin) * committed evidence
    "drop_frac": 0.6,       # reject committed slot if evidence < drop_frac * peak
    "timeout_s": 2.0,       # hidden-goal memory timeout (HOLD -> SEARCH)
    "reject_s": 5.0,        # cooldown before a rejected slot may be re-acquired
}


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class StateMachine:
    name = "state_machine"
    PARAM_SPACE = {
        "ev_tau_s": (0.03, 2.0, "log"),
        "bearing_tau_s": (0.02, 1.0, "log"),
        "enter_thr": (0.1, 0.9, "lin"),
        "exit_ratio": (0.3, 0.95, "lin"),
        "debounce_s": (0.04, 1.0, "log"),
        "debounce_frac": (0.5, 1.0, "lin"),
        "switch_margin": (0.05, 1.0, "log"),
        "drop_frac": (0.3, 0.95, "lin"),
        "timeout_s": (0.2, 10.0, "log"),
        "reject_s": (0.5, 30.0, "log"),
    }

    def __init__(self, params: dict, batch: int, rng=None):
        unknown = set(params) - set(DEFAULTS)
        assert not unknown, f"unknown state_machine params: {unknown}"
        self.p = {**DEFAULTS, **params}
        self.B = batch
        self.K = None  # set on first step (K comes from obs)

    def _init(self, K, dt):
        B, p = self.B, self.p
        self.K = K
        self.W = max(1, int(round(p["debounce_s"] / dt)))           # n
        self.k_req = max(1, int(np.ceil(p["debounce_frac"] * self.W)))  # k
        self.exit_thr = p["exit_ratio"] * p["enter_thr"]
        self.t = 0
        self.state = np.full(B, SEARCH, dtype=np.int64)
        self.target = np.full(B, -1, dtype=np.int64)
        self.ev = np.zeros((B, K))
        self.bvec = np.zeros((B, K, 2))
        self.seen = np.zeros((B, K), dtype=bool)
        self.hidden_t = np.zeros((B, K))
        self.reject_t = np.zeros((B, K))
        self.peak = np.zeros(B)
        self.h_acq = np.zeros((B, K, self.W), dtype=bool)
        self.h_sw = np.zeros((B, K, self.W), dtype=bool)
        self.h_bad = np.zeros((B, self.W), dtype=bool)

    def _commit(self, rows, slots):
        """Commit rows to slots: APPROACH, reset peak and target-relative debounce."""
        self.target[rows] = slots
        self.state[rows] = APPROACH
        self.peak[rows] = self.ev[rows, slots]
        self.h_sw[rows] = False
        self.h_bad[rows] = False

    def step(self, obs: dict, dt: float) -> dict:
        o = obs["objects"]
        bear = np.asarray(o["bearing"], float)
        stren = np.asarray(o["strength"], float)
        vis = np.asarray(o["visible"], bool)
        if self.K is None:
            self._init(bear.shape[1], dt)
        p, B, K = self.p, self.B, self.K
        rows = np.arange(B)
        slot = self.t % self.W
        self.t += 1

        # --- per-slot filters (frozen while hidden; first sighting initialises) ---
        a_ev = 1.0 - np.exp(-dt / p["ev_tau_s"])
        a_b = 1.0 - np.exp(-dt / p["bearing_tau_s"])
        first = vis & ~self.seen
        u = np.stack([np.cos(bear), np.sin(bear)], -1)
        self.ev = np.where(vis, np.where(first, stren, self.ev + a_ev * (stren - self.ev)), self.ev)
        self.bvec = np.where(vis[..., None], np.where(first[..., None], u, self.bvec + a_b * (u - self.bvec)), self.bvec)
        self.seen |= vis
        fbear = np.arctan2(self.bvec[..., 1], self.bvec[..., 0])
        self.hidden_t = np.where(vis, 0.0, self.hidden_t + dt)
        self.reject_t = np.maximum(self.reject_t - dt, 0.0)
        ok = vis & (self.reject_t <= 0)

        has_t = self.target >= 0
        tgt = np.where(has_t, self.target, 0)
        ev_t = self.ev[rows, tgt]
        vis_t = vis[rows, tgt] & has_t
        self.peak = np.where(vis_t, np.maximum(self.peak, ev_t), self.peak)

        # --- k-of-n debounce buffers ---
        not_t = np.arange(K)[None, :] != self.target[:, None]
        self.h_acq[:, :, slot] = ok & (self.ev > p["enter_thr"])
        self.h_sw[:, :, slot] = (ok & not_t & has_t[:, None] & (self.ev > p["enter_thr"])
                                 & (self.ev > (1.0 + p["switch_margin"]) * ev_t[:, None]))
        self.h_bad[:, slot] = vis_t & ((ev_t < self.exit_thr) | (ev_t < p["drop_frac"] * self.peak))
        acq = self.h_acq.sum(-1) >= self.k_req
        sw = self.h_sw.sum(-1) >= self.k_req
        bad = self.h_bad.sum(-1) >= self.k_req

        # --- transitions, priority order: HOME > reject > switch > keep/hold > acquire ---
        homing = np.asarray(obs["home"]["strength"], float) > 0
        self.state[~homing & (self.state == HOME)] = SEARCH
        active = ~homing & has_t

        r = active & bad                       # look-alike release
        self.reject_t[rows[r], tgt[r]] = p["reject_s"]
        self.target[r] = -1
        self.state[r] = SEARCH
        acq &= self.reject_t <= 0
        sw &= self.reject_t <= 0

        s = active & ~bad & sw.any(-1)         # clearly stronger challenger takes over
        if s.any():
            best = np.argmax(np.where(sw, self.ev, -np.inf), -1)
            self._commit(rows[s], best[s])

        keep = active & ~bad & ~s
        self.state[keep & vis_t] = APPROACH
        hold = keep & ~vis_t
        self.state[hold] = HOLD
        lost = hold & (self.hidden_t[rows, tgt] > p["timeout_s"])
        self.target[lost] = -1
        self.state[lost] = SEARCH

        a = ~homing & (self.target < 0) & acq.any(-1)  # acquire best confirmed slot
        if a.any():
            best = np.argmax(np.where(acq, self.ev, -np.inf), -1)
            self._commit(rows[a], best[a])
        self.state[homing] = HOME

        # --- heading command ---
        tgt = np.where(self.target >= 0, self.target, 0)
        heading = np.asarray(obs["heading"], float)
        sb = np.asarray(obs["search"]["bearing"], float)
        search_h = np.where(np.asarray(obs["search"]["strength"], float) > 0, sb, heading)
        cmd = np.select(
            [self.state == HOME, (self.state == APPROACH) | (self.state == HOLD)],
            [np.asarray(obs["home"]["bearing"], float), fbear[rows, tgt]],
            search_h)

        ev_t = self.ev[rows, tgt]
        band = max(p["enter_thr"] - self.exit_thr, 1e-9)
        commit = np.select(
            [self.state == HOME, self.state == APPROACH, self.state == HOLD],
            [np.ones(B),
             np.clip((ev_t - self.exit_thr) / band, 0, 1),
             np.clip((ev_t - self.exit_thr) / band, 0, 1)
             * np.clip(1 - self.hidden_t[rows, tgt] / p["timeout_s"], 0, 1)],
            0.0)
        return {"heading_cmd": _wrap(cmd), "commit": commit,
                "viz": {"state": self.state.copy(), "target": self.target.copy(),
                        "evidence": self.ev.copy(), "rejected": self.reject_t > 0}}


if __name__ == "__main__":
    dt = 0.02

    def mk(B, K, bear, stren, vis, home_s=0.0, home_b=0.0, search_b=-2.0):
        f = lambda x, sh: np.broadcast_to(np.asarray(x, float), sh).copy()
        return {"objects": {"bearing": f(bear, (B, K)), "strength": f(stren, (B, K)),
                            "visible": np.broadcast_to(np.asarray(vis, bool), (B, K)).copy()},
                "home": {"bearing": f(home_b, (B,)), "strength": f(home_s, (B,))},
                "search": {"bearing": f(search_b, (B,)), "strength": f(1.0, (B,))},
                "heading": np.zeros(B)}

    def run(sm, n, **kw):
        out = None
        for _ in range(n):
            out = sm.step(mk(**kw), dt)
        return out

    def close(a, b, tol=0.05):
        return np.all(np.abs(_wrap(np.asarray(a) - b)) < tol)

    rng = np.random.default_rng(0)

    # (1) single visible object -> steer to its bearing
    sm = StateMachine({}, 1)
    out = run(sm, 50, B=1, K=1, bear=0.7, stren=0.8, vis=True)
    assert out["viz"]["state"][0] == APPROACH and close(out["heading_cmd"], 0.7), out
    assert out["commit"][0] > 0.9

    # (2) hidden 1 s -> holds last bearing; hidden past timeout (2 s) -> SEARCH
    out = run(sm, 50, B=1, K=1, bear=0.0, stren=0.0, vis=False)
    assert out["viz"]["state"][0] == HOLD and close(out["heading_cmd"], 0.7), out
    out = run(sm, 60, B=1, K=1, bear=0.0, stren=0.0, vis=False)
    assert out["viz"]["state"][0] == SEARCH and close(out["heading_cmd"], -2.0), out
    # brief one-tick dropout while approaching does not change course
    sm = StateMachine({}, 1)
    run(sm, 30, B=1, K=1, bear=0.7, stren=0.8, vis=True)
    out = run(sm, 1, B=1, K=1, bear=0.0, stren=0.0, vis=False)
    assert out["viz"]["state"][0] == HOLD and close(out["heading_cmd"], 0.7)
    out = run(sm, 1, B=1, K=1, bear=0.7, stren=0.8, vis=True)
    assert out["viz"]["state"][0] == APPROACH

    # (3) two noisy objects: stronger wins after debounce; switch when the other is clearly stronger
    sm = StateMachine({}, 1)
    out = sm.step(mk(1, 2, [1.0, -1.0], [0.6, 0.8], True), dt)
    assert out["viz"]["state"][0] == SEARCH, "must not commit before debounce"
    for _ in range(50):
        out = sm.step(mk(1, 2, [1.0, -1.0], np.array([0.6, 0.8]) + rng.normal(0, 0.1, 2), True), dt)
    assert out["viz"]["target"][0] == 1 and close(out["heading_cmd"], -1.0, 0.1), out
    # one-tick spike on slot 0 must not cause a switch
    out = sm.step(mk(1, 2, [1.0, -1.0], [3.0, 0.8], True), dt)
    out = run(sm, 5, B=1, K=2, bear=[1.0, -1.0], stren=[0.6, 0.8], vis=True)
    assert out["viz"]["target"][0] == 1
    for _ in range(100):
        out = sm.step(mk(1, 2, [1.0, -1.0], np.array([1.1, 0.6]) + rng.normal(0, 0.1, 2), True), dt)
    assert out["viz"]["target"][0] == 0 and close(out["heading_cmd"], 1.0, 0.1), out

    # look-alike release: evidence sinks as it is approached -> rejected, not re-acquired during cooldown
    sm = StateMachine({}, 1)
    run(sm, 50, B=1, K=1, bear=0.3, stren=0.8, vis=True)
    for s in np.linspace(0.8, 0.3, 100):
        out = sm.step(mk(1, 1, 0.3, s, True), dt)
    assert out["viz"]["state"][0] == SEARCH and out["viz"]["rejected"][0, 0], out
    out = run(sm, 50, B=1, K=1, bear=0.3, stren=0.8, vis=True)
    assert out["viz"]["state"][0] == SEARCH, "rejected slot re-acquired inside cooldown"
    out = run(sm, 250, B=1, K=1, bear=0.3, stren=0.8, vis=True)
    assert out["viz"]["state"][0] == APPROACH, "slot should be re-acquirable after cooldown"

    # (4) home strength > 0 -> steers home, overriding a committed object
    sm = StateMachine({}, 1)
    run(sm, 50, B=1, K=1, bear=0.7, stren=0.8, vis=True)
    out = run(sm, 1, B=1, K=1, bear=0.7, stren=0.8, vis=True, home_s=1.0, home_b=2.5)
    assert out["viz"]["state"][0] == HOME and close(out["heading_cmd"], 2.5) and out["commit"][0] == 1

    # (5) batch shapes, rows evolve independently; timeout_s param is honoured
    B, K = 3, 4
    sm = StateMachine({"timeout_s": 0.5}, B)
    bear = np.tile([0.5, -0.5, 1.5, -1.5], (B, 1))
    stren = np.array([[0.9, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.9, 0]])
    vis = stren > 0
    home_s = np.array([0.0, 0.0, 1.0])
    out = run(sm, 50, B=B, K=K, bear=bear, stren=stren, vis=vis, home_s=home_s, home_b=3.0)
    assert out["heading_cmd"].shape == (B,) and out["commit"].shape == (B,)
    assert out["viz"]["state"].shape == (B,) and out["viz"]["evidence"].shape == (B, K)
    assert list(out["viz"]["state"]) == [APPROACH, SEARCH, HOME], out["viz"]["state"]
    assert close(out["heading_cmd"], np.array([0.5, -2.0, 3.0]))
    assert np.all((out["commit"] >= 0) & (out["commit"] <= 1))
    out = run(sm, 30, B=B, K=K, bear=bear, stren=0.0, vis=False, home_s=home_s, home_b=3.0)
    assert out["viz"]["state"][0] == SEARCH, "0.6 s hidden > timeout_s=0.5 must drop to SEARCH"

    assert set(StateMachine.PARAM_SPACE) == set(DEFAULTS) and "timeout_s" in StateMachine.PARAM_SPACE
    for k, (lo, hi, _) in StateMachine.PARAM_SPACE.items():
        assert lo <= DEFAULTS[k] <= hi, k
    print(f"state_machine self-check OK ({len(StateMachine.PARAM_SPACE)} params, states {STATE_NAMES})")
