"""Find-and-return mission: batched Python reference (app/sim/mission.js is the batch-1 JS twin).

Protocol (fixed before any mission data; the paper's Protocol and amendments section):
  * Before t = 0 the mushroom body (MB) is loaded with the target "picture": the looks of the target taken on a
    previous visit along a learning flight of brain.learning_flight_s seconds (5 views/s around the target, eye.py
    learning_poses); learning_flight_s = 0 means one photo from 4 m in front. The drone has no map and no GPS.
  * The operator gives a rough vector to the reported spot (a waggle-dance-like instruction): bearing error
    N(0, 8 deg), distance factor 1 + N(0, 0.1) (Rng stream 6). In the swapped-decoy fault the reported spot holds
    the look-alike and the real target sits 8-14 m away (scenario.py).
  * t < 2 s "learning": take off and climb at home (the MB already holds the learning-flight views).
    "search": the search drive flies to the reported spot by path integration, then an outward Archimedean spiral
    (6 m pitch) around it. "approach": the arbiter is committed (commit > 0.5) and steers within 30 deg of a visible object with
    strength > 0.2 (display only).
    Shared stop rule (all arms): within 2 m horizontal of a visible object whose 1-s mean strength > 0.6.
    Then "home": the home drive (PI home vector, strength 1) replaces the object and search drives; landing when the
    decoded home vector is shorter than 1 m. "done" on landing, crash (collision) or max_time_s.
  * Rates: physics 500 Hz (C's Quad + Controller + Dryden); brain/arbiter/PI 50 Hz; eye (visibility, MB looks,
    depth ring, avoidance reflex) 10 Hz, held in between; frames logged at 10 Hz.
  * Steering: heading = arbiter heading_cmd + avoidance yaw offset (descending-neuron level), rate-limited to
    120 deg/s, in the drone's own compass frame; the velocity set-point is V_CRUISE (2 m/s) times the looming
    slow-down along that heading, 1 m/s within 5 m of a candidate object, slower near the estimated home.
  * Obs conventions (frozen state machine): object strength 0..1; hidden slots report their last-seen bearing with
    strength 0; home strength 0 until the stop, then 1. No visual home beacon: landing relies on the PI alone.
  * Familiarity -> evidence: strength = clip((f - F0) / (F1 - F0), 0, 1) for visible objects, 0 otherwise,
    F0 = 0.2, F1 = 0.6 (set on training seeds 0-39 from distance/angle sweeps, before mission data).
Metrics (C7; definitions follow scripts/kt2_five_arms.py where they overlap):
  success_target / wrong_object_stop: the stop was at the target / at a decoy. success_home: landed within 2 m of
  home after a target stop. collisions: 1 if the drone hit an obstacle (the mission ends). time_s: end time.
  path_len_m: horizontal path. home_error_m: horizontal distance to home at the end. max_bank_deg: max tilt.
  energy_proxy: integral of sum_i f_i^1.5 dt (rotor power ~ thrust^1.5, momentum theory; N^1.5 s).
  t_hold_occlusion_s: occlusion starts t_s after the target is first visible within 10 m; time until the steering
  command first leaves +/-20 deg of the hidden target's bearing (capped at the occlusion length); NaN if none.
  t_release_decoy_s: from the first time the target's 1-s mean evidence exceeds 1.5x the decoy's (after the decoy
  was seen) until the steering command is within 20 deg of the target's bearing, capped at 20 s; NaN if no decoy
  or the evidence never turned.
Usage: uv run python scripts/mission.py            (self-check: 10 easy meadow seeds, ring and vote)
       uv run python scripts/mission.py --time     (time a batch of 50 seeds x 300 s, print the extrapolation)
"""
import json, math, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "brain")]
from rng import Rng  # noqa: E402
from provenance import stamp  # noqa: E402
from scenario import load, scenario_hash, strip  # noqa: E402
from world import build_world  # noqa: E402
from quad import Quad, Controller, P as QP, wrap  # noqa: E402
from dryden import Dryden  # noqa: E402
import eye  # noqa: E402
import mb as mbmod  # noqa: E402
from pi import PathIntegrator, SIGMA_COMPASS, SIGMA_SPEED  # noqa: E402
from avoid import reflex, collided  # noqa: E402
from mind import say, steering_slot  # noqa: E402
from arbiters import make, drive_inputs, N_RING  # noqa: E402

DT_B, SUB = 0.02, 10           # brain tick, physics sub-steps (500 Hz)
EYE_EVERY = 5                  # eye at 10 Hz
K_MAX = 4                      # object slots (noise draws are made for K_MAX slots so rows never depend on the batch)
ALT, V_CRUISE, V_NEAR = 2.5, 2.0, 1.0
T_TAKEOFF = 2.0
TURN_MAX = math.radians(120)
STOP_R, STOP_THR, STOP_WIN = 2.0, 0.6, 10      # m, evidence, eye samples (1 s)
F0, F1 = 0.2, 0.6
SEARCH_S = 0.5
SPIRAL_B = 6.0
INSTR_BEAR_SD, INSTR_DIST_SD = math.radians(8.0), 0.1
HOLD_DEG, CROSS_RATIO, REL_CAP = 20.0, 1.5, 20.0
OCC_NEAR = 10.0
N_KC, K_IN, KC_FRAC, MB_SEED = 20000, 10, 0.01, 0
COLUMNS = ["scenario_hash", "seed", "arbiter", "ring_gain", "split", "success_target", "success_home", "time_s",
           "path_len_m", "collisions", "home_error_m", "wrong_object_stop", "max_bank_deg", "energy_proxy",
           "t_hold_occlusion_s", "t_release_decoy_s", "git", "sim_version", "created"]

_WIRING = None


def wiring():
    global _WIRING
    if _WIRING is None:
        _WIRING = mbmod.wiring(eye.LH * eye.LW, N_KC, K_IN, MB_SEED)
    return _WIRING


_W = None


def count_matrix():
    """[n_in, N_KC] PN->KC synapse counts (float32), as in mb.MushroomBody.W."""
    global _W
    if _W is None:
        idx = wiring()
        _W = np.zeros((eye.LH * eye.LW, N_KC), np.float32)
        np.add.at(_W, (idx, np.arange(N_KC)[:, None]), 1.0)
    return _W


def ranks(grey):
    """grey [N, n] -> rank 1..n (int), quantised to 1/4096 first, ties broken by pixel index (stable)."""
    q = np.floor(np.asarray(grey) * 4096 + 0.5)
    o = np.argsort(q, axis=-1, kind="stable")
    r = np.empty_like(o)
    np.put_along_axis(r, o, np.arange(1, q.shape[-1] + 1)[None, :].repeat(len(q), 0), -1)
    return r


def kc_active(rk):
    """rank vectors [N, n] -> active KCs [N, N_KC]. Integer form of mb.MushroomBody.active on mb.equalise inputs
    (KC drive = sum of the ranks of its 10 PNs; the L2 normalisation is a common factor), exact in Python and JS."""
    d = rk.astype(np.float32) @ count_matrix()     # small integers: float32 sums are exact in any order
    k = max(1, int(round(KC_FRAC * N_KC)))
    thr = -np.partition(-d, k - 1, axis=-1)[:, k - 1:k]
    return d >= thr


class Mission:
    def __init__(self, scenarios, log_rows=()):
        self.sc = [load(strip(s)) for s in scenarios]
        self.meta = [s.get("_meta", {}) for s in scenarios]
        B = self.B = len(self.sc)
        br = self.sc[0]["brain"]
        for s in self.sc:
            assert (s["brain"]["arbiter"], s["brain"]["params"], s["brain"]["ring_gain"]) == (br["arbiter"], br["params"], br["ring_gain"]), \
                "a batch shares one arbiter config (run_batch groups them)"
            assert len(s["decoys"]) + 1 <= K_MAX
        self.seeds = np.array([s["seed"] for s in self.sc])
        self.worlds = [build_world(s) for s in self.sc]
        self.S = eye.Scene(self.worlds)
        self.K = self.S.K
        self.home = np.array([s["home"] for s in self.sc], float)
        # streams
        self.r_comp, self.r_speed, self.r_eye, self.r_search = (Rng(self.seeds, k) for k in (2, 3, 4, 6))
        # instruction (stream 6): bearing error, distance factor, initial yaw
        nb, nd, u0 = self.r_search.normal(), self.r_search.normal(), self.r_search.uniform()
        rep = []
        for s in self.sc:
            sw = [d for d in s["decoys"] if d.get("swap")]
            rep.append((sw[0]["pos"] if sw else s["target"]["pos"]))
        rel = np.array(rep, float)[:, :2] - self.home[:, :2]
        ang = np.arctan2(rel[:, 1], rel[:, 0]) + INSTR_BEAR_SD * nb
        dist = np.hypot(rel[:, 0], rel[:, 1]) * (1 + INSTR_DIST_SD * nd)
        self.goal = np.stack([dist * np.cos(ang), dist * np.sin(ang)], 1)        # PI frame (home = origin)
        yaw0 = 2 * np.pi * u0 - np.pi
        # vehicle
        self.quad = Quad(B, pos0=self.home * [1, 1, 0], yaw0=yaw0)
        self.ctrl = Controller(B)
        self.wind_groups = []
        for lvl in sorted({s["wind"]["turbulence"] for s in self.sc}):
            rows = np.array([i for i, s in enumerate(self.sc) if s["wind"]["turbulence"] == lvl])
            mean = np.array([self.sc[i]["wind"]["mean_mps"] for i in rows], float)
            self.wind_groups.append((rows, Dryden(len(rows), self.seeds[rows], lvl, mean, alt_m=ALT, airspeed=V_CRUISE)))
        self.wind = np.zeros((B, 3))
        # brain
        self.pi = PathIntegrator(B)
        self.mbw = np.ones((B, N_KC), bool)
        for lf in sorted({s["brain"]["learning_flight_s"] for s in self.sc}):
            rows = np.array([i for i, s in enumerate(self.sc) if s["brain"]["learning_flight_s"] == lf])
            L = eye.learning_looks(self.S, lf)[rows]                               # [b, V, 192]
            act = kc_active(ranks(L.reshape(-1, L.shape[-1]))).reshape(len(rows), L.shape[1], N_KC)
            self.mbw[rows] &= ~act.any(1)
        params = dict(br["params"])
        if br["arbiter"] == "ring":
            params["ring_gain"] = br["ring_gain"]
        self.arb_name = br["arbiter"]
        self.arb = make(br["arbiter"], params, B)
        # faults
        self.fault = [{f["kind"]: f for f in s["faults"]} for s in self.sc]
        self.false_sun = np.array([math.radians(f["false_sun"]["params"].get("offset_deg", 0)) if "false_sun" in f else 0.0 for f in self.fault])
        self.t_sun = np.array([f["false_sun"]["t_s"] if "false_sun" in f else np.inf for f in self.fault])
        self.t_comp = np.array([f["compass_half"]["t_s"] if "compass_half" in f else np.inf for f in self.fault])
        self.speed_scale = np.array([f["speed_scale"]["params"].get("scale", 1.0) if "speed_scale" in f else 1.0 for f in self.fault])
        self.t_speed = np.array([f["speed_scale"]["t_s"] if "speed_scale" in f else np.inf for f in self.fault])
        self.t_cols = np.array([f["eye_columns"]["t_s"] if "eye_columns" in f else np.inf for f in self.fault])
        self.dead_cols = np.zeros((B, eye.NC), bool)
        for b, f in enumerate(self.fault):
            if "eye_columns" in f:
                p = f["eye_columns"]["params"]
                n = int(math.floor(p.get("frac", 0.25) * eye.NC + 0.5))
                self.dead_cols[b, (p.get("start_col", 0) + np.arange(n)) % eye.NC] = True
        self.occ_delay = np.array([f["occlusion"]["t_s"] if "occlusion" in f else np.nan for f in self.fault])
        self.occ_dur = np.array([f["occlusion"]["params"].get("duration_s", 5.0) if "occlusion" in f else 0.0 for f in self.fault])
        # state
        self.t, self.tick = 0.0, 0
        self.phase = np.zeros(B, int)               # 0 learning, 1 search, 2 approach, 3 home, 4 done
        self.psi_sp = yaw0.copy()                   # steering set-point in the compass frame
        self.heading_est = yaw0.copy()
        self.vis = np.zeros((B, self.K), bool)
        self.bear_rel = np.zeros((B, self.K))
        self.last_bear = np.zeros((B, self.K))      # last-seen perceived bearing (compass frame); hidden slots keep it
        self.rh = np.full((B, self.K), np.inf)
        self.strength = np.zeros((B, self.K))
        self.fam = np.zeros((B, self.K))
        self.hist = np.zeros((B, self.K, STOP_WIN))
        self.ev = np.zeros((B, self.K))
        self.yaw_off, self.speed_fac, self.loom = np.zeros(B), np.ones(B), np.zeros(B)
        self.depth = np.full((B, eye.NC), eye.DEPTH_MAX)
        self.spiral = np.full(B, -1.0)              # < 0: still flying to the reported spot
        self.spiral0 = np.zeros(B)
        self.out = {k: np.zeros(B, bool) for k in ("success_target", "success_home", "wrong_object_stop")}
        self.collisions = np.zeros(B, int)
        self.t_end = np.full(B, np.nan)
        self.end_pos = np.full((B, 2), np.nan)       # horizontal position when the mission ended
        self.path = np.zeros(B); self.energy = np.zeros(B); self.max_bank = np.zeros(B)
        self.t_first_t = np.full(B, np.nan); self.occ_on = np.full(B, np.nan)
        self.hold_alive = np.ones(B, bool); self.t_hold = np.full(B, np.nan)
        self.dec_seen = np.zeros(B, bool); self.t_cross = np.full(B, np.nan); self.t_rel = np.full(B, np.nan)
        self.is_t = self.S.role.astype(bool) & self.S.kvalid
        self.is_d = ~self.S.role.astype(bool) & self.S.kvalid
        self.has_dec = self.is_d.any(1)
        self.last = None
        self.log_rows = list(log_rows)
        self.frames = {b: [] for b in self.log_rows}

    # ---------------------------------------------------------------- one brain tick (0.02 s)
    def step(self):
        B, t, S = self.B, self.t, self.S
        rows = np.arange(B)
        q = self.quad
        yaw = q.euler()[:, 2]
        pos = q.pos
        live = self.phase < 4
        # faults active now
        sun_on = t >= self.t_sun
        cols_on = t >= self.t_cols
        self.pi.col_mask[:] = 1.0
        self.pi.col_mask[t >= self.t_comp] = np.tile(np.r_[np.zeros(4), np.ones(4)], 2)
        # compass (stream 2) and odometry (stream 3)
        theta = yaw + SIGMA_COMPASS * self.r_comp.normal() + np.where(sun_on, self.false_sun, 0.0)
        c, s_ = np.cos(yaw), np.sin(yaw)
        vb = np.stack([c * q.vel[:, 0] + s_ * q.vel[:, 1], -s_ * q.vel[:, 0] + c * q.vel[:, 1]], 1)
        n1, n2 = self.r_speed.normal(), self.r_speed.normal()
        scale = np.where(t >= self.t_speed, self.speed_scale, 1.0)
        vmeas = scale[:, None] * vb + SIGMA_SPEED * np.stack([n1, n2], 1)
        self.heading_est = self.pi.step(theta, vmeas, DT_B)
        hv = self.pi.home_vector()
        dead = np.where(cols_on[:, None], self.dead_cols, False)
        # eye at 10 Hz
        if self.tick % EYE_EVERY == 0:
            self._eye(pos, yaw, vmeas, dead, live)
        bearing = np.where(self.vis, wrap(self.bear_rel + self.heading_est[:, None]), self.last_bear)   # perceived, compass frame
        self.last_bear = bearing
        # drives
        p_hat = -hv
        to_goal = self.goal - p_hat
        arrived = (self.spiral < 0) & (np.hypot(to_goal[:, 0], to_goal[:, 1]) < 3.0)
        self.spiral0 = np.where(arrived, np.arctan2(-to_goal[:, 1], -to_goal[:, 0]), self.spiral0)
        self.spiral = np.where(arrived, 0.0, self.spiral)
        for _ in range(100):                                               # keep the spiral waypoint >= 3 m ahead
            wp = self._spiral_wp()
            near = (self.spiral >= 0) & (np.hypot(*(wp - p_hat).T) < 3.0)
            if not near.any():
                break
            r = SPIRAL_B * self.spiral / (2 * np.pi)
            self.spiral = np.where(near, self.spiral + np.minimum(0.5, 1.0 / np.maximum(r, 1.0)), self.spiral)
        tgt = np.where((self.spiral >= 0)[:, None], self._spiral_wp(), self.goal)
        search_b = np.arctan2(tgt[:, 1] - p_hat[:, 1], tgt[:, 0] - p_hat[:, 0])
        homing = self.phase == 3
        seeking = (self.phase == 1) | (self.phase == 2)
        obs = {"objects": {"bearing": bearing, "strength": np.where(seeking[:, None], self.strength, 0.0),
                           "visible": self.vis & seeking[:, None]},
               "home": {"bearing": np.arctan2(hv[:, 1], hv[:, 0]), "strength": homing.astype(float)},
               "search": {"bearing": search_b, "strength": np.where(seeking | (self.phase == 0), SEARCH_S, 0.0)},
               "heading": self.heading_est}
        res = self.arb.step(obs, DT_B)
        cmd = res["heading_cmd"]
        self.last = (obs, res)
        # metrics on the steering command (KT2 definitions)
        true_b = np.arctan2(S.obj[..., 1] - pos[:, None, 1], S.obj[..., 0] - pos[:, None, 0])
        tb = wrap(true_b - yaw[:, None] + self.heading_est[:, None])         # where the target would be seen
        tk = np.argmax(self.is_t, 1)
        on_t = np.abs(wrap(cmd - tb[rows, tk])) <= math.radians(HOLD_DEG)
        occ = (t >= self.occ_on) & (t < self.occ_on + self.occ_dur)
        lose = occ & self.hold_alive & ~on_t & live
        self.t_hold = np.where(lose, t - self.occ_on, self.t_hold)
        self.hold_alive &= ~lose
        rel_now = ~np.isnan(self.t_cross) & np.isnan(self.t_rel) & on_t & live
        self.t_rel = np.where(rel_now, t - self.t_cross, self.t_rel)
        # phase logic: shared stop rule, landing
        self.phase = np.where((self.phase == 0) & (t >= T_TAKEOFF), 1, self.phase)
        seeking = (self.phase == 1) | (self.phase == 2)
        on_obj = (self.vis & (self.strength > 0.2) & (np.abs(wrap(bearing - cmd[:, None])) < math.radians(30))).any(1)
        self.phase = np.where(seeking, np.where((res["commit"] > 0.5) & on_obj, 2, 1), self.phase)
        cand = self.vis & (self.rh < STOP_R) & (self.ev > STOP_THR) & seeking[:, None]
        st_t, st_d = (cand & self.is_t).any(1), (cand & self.is_d).any(1)
        st_d &= ~st_t
        self.out["success_target"] |= st_t
        self.out["wrong_object_stop"] |= st_d
        self.phase = np.where(st_t | st_d, 3, self.phase)
        landed = (self.phase == 3) & (np.hypot(hv[:, 0], hv[:, 1]) < 1.0)
        hd_true = np.hypot(pos[:, 0] - self.home[:, 0], pos[:, 1] - self.home[:, 1])
        self.out["success_home"] |= landed & self.out["success_target"] & (hd_true < STOP_R)
        crash = live & collided(S, pos)
        self.collisions += crash
        end = live & (landed | crash)
        self.t_end = np.where(end, t, self.t_end)
        self.end_pos = np.where(end[:, None], pos[:, :2], self.end_pos)
        self.phase = np.where(end, 4, self.phase)
        # descending-neuron level: arbiter + reflex, rate-limited, mapped from compass frame to the airframe
        live = self.phase < 4
        want = cmd + self.yaw_off
        self.psi_sp = wrap(self.psi_sp + np.clip(wrap(want - self.psi_sp), -TURN_MAX * DT_B, TURN_MAX * DT_B))
        yaw_sp = wrap(self.psi_sp - self.heading_est + yaw)
        near_obj = (self.vis & (self.rh < 5.0) & (self.strength > 0.3)).any(1)
        v = V_CRUISE * self.speed_fac
        v = np.where(near_obj & seeking, np.minimum(v, V_NEAR), v)
        v = np.where(self.phase == 3, np.minimum(v, np.maximum(0.4, np.hypot(hv[:, 0], hv[:, 1]) / 2)), v)
        v = np.where((self.phase == 0) | ~live, 0.0, v)
        sp = {"vel_xy": v[:, None] * np.stack([np.cos(yaw_sp), np.sin(yaw_sp)], 1),
              "alt": np.where(live, ALT, 0.0), "yaw": np.where(self.phase == 0, yaw, yaw_sp)}
        if self.log_rows and self.tick % EYE_EVERY == 0:
            self._log(res, hv, dead, sun_on, yaw)
        for _ in range(SUB):
            for rows_g, d in self.wind_groups:
                self.wind[rows_g] = d.step(DT_B / SUB)
            m = self.ctrl(q, sp, DT_B / SUB)
            q.step(m, self.wind, DT_B / SUB)
            lv = self.phase < 4
            self.path += lv * np.hypot(q.vel[:, 0], q.vel[:, 1]) * DT_B / SUB
            self.energy += lv * (np.maximum(q.motor, 0) ** 1.5).sum(1) * DT_B / SUB
            R = q.quat
            tilt = np.degrees(np.arccos(np.clip(1 - 2 * (R[:, 1] ** 2 + R[:, 2] ** 2), -1, 1)))
            self.max_bank = np.where(lv, np.maximum(self.max_bank, tilt), self.max_bank)
        self.tick += 1
        self.t = self.tick * DT_B

    def _spiral_wp(self):
        ph = np.maximum(self.spiral, 0.0)
        r = SPIRAL_B * ph / (2 * np.pi)
        a = self.spiral0 + ph
        return self.goal + np.stack([r * np.cos(a), r * np.sin(a)], 1)

    def _eye(self, pos, yaw, vmeas, dead, live):
        S, B, K, t = self.S, self.B, self.K, self.t
        rows = np.arange(B)
        eye_pos = pos.copy()
        occ = ((t >= self.occ_on) & (t < self.occ_on + self.occ_dur))[:, None] & self.is_t
        vis, bear, rh = eye.visibility(S, eye_pos, yaw, t, dead=dead, occluded=occ)
        noise = eye.normals(self.r_eye, K_MAX * eye.LH * eye.LW).reshape(B, K_MAX, -1)[:, :K]
        fam = np.zeros((B, K))
        idx = np.argwhere(vis)
        if len(idx):
            grey, _ = eye.looks(eye.sub_scene(S, idx[:, 0], idx[:, 1]), eye_pos[idx[:, 0]], t)
            g = grey[:, 0] + eye.NOISE * noise[idx[:, 0], idx[:, 1]]
            act = kc_active(ranks(g))
            fam[idx[:, 0], idx[:, 1]] = 1.0 - (act & self.mbw[idx[:, 0]]).sum(-1) / act.sum(-1)
        self.fam = fam
        self.vis, self.rh = vis, rh
        self.bear_rel = wrap(bear - yaw[:, None])
        self.strength = np.where(vis, np.clip((fam - F0) / (F1 - F0), 0.0, 1.0), 0.0)
        self.hist = np.roll(self.hist, 1, -1)
        self.hist[..., 0] = self.strength
        self.ev = self.hist.mean(-1)
        # occlusion onset: t_s after the target is first visible within OCC_NEAR m (KT2: t_first + delay)
        tv = (vis & self.is_t & (rh < OCC_NEAR)).any(1) & ((self.phase == 1) | (self.phase == 2))
        self.t_first_t = np.where(np.isnan(self.t_first_t) & tv, t, self.t_first_t)
        self.occ_on = np.where(np.isnan(self.occ_on) & ~np.isnan(self.occ_delay), self.t_first_t + self.occ_delay, self.occ_on)
        # decoy evidence crossing (KT2): target evidence > 1.5 x decoy evidence after the decoy was seen
        ev_t, ev_d = (self.ev * self.is_t).sum(1), (self.ev * self.is_d).sum(1)
        self.dec_seen |= ev_d > 0
        newc = np.isnan(self.t_cross) & self.dec_seen & (ev_t > CROSS_RATIO * ev_d) & live & (self.phase < 3)
        self.t_cross = np.where(newc, t, self.t_cross)
        # depth ring + reflex (flow noise after the look noise, stream 4)
        fn = eye.normals(self.r_eye, eye.NC)
        self.depth = eye.depth_ring(S, pos, yaw)
        self.yaw_off, self.speed_fac, self.loom = reflex(self.depth, vmeas, fn, dead)

    def _log(self, res, hv, dead, sun_on, yaw):
        q = self.quad
        img = eye.panorama(self.S, q.pos, yaw, self.t, dead=dead,
                           false_sun=np.where(sun_on & (self.false_sun != 0), self.S.sun_az + self.false_sun, np.nan))
        eul = q.euler()
        obs, _ = self.last
        names = ["learning", "search", "approach", "home", "done"]
        for b in self.log_rows:
            vz = res["viz"]
            if "ring" in vz and "drives" in vz:
                act, drv = vz["ring"][b], {k: v[b] for k, v in vz["drives"].items()}
            else:
                ob = {"objects": {k: v[b:b + 1] for k, v in obs["objects"].items()},
                      "home": {k: v[b:b + 1] for k, v in obs["home"].items()},
                      "search": {k: v[b:b + 1] for k, v in obs["search"].items()}}
                drv = {k: v[0] for k, v in drive_inputs(ob, 2.0, 1.0, 1.0, 1.0).items()}
                act = np.zeros(N_RING)
            vis_b = self.vis[b] & self.S.kvalid[b]
            fam = float(self.fam[b][vis_b].max()) if vis_b.any() else 0.0
            slot = steering_slot(res["heading_cmd"][b], obs["objects"]["bearing"][b], self.strength[b], vis_b)
            roles = ["target" if r else "decoy" for r in self.S.role[b]]
            fa = [k for k, f in self.fault[b].items() if self.t >= f["t_s"] and k != "occlusion"]
            if (self.t >= self.occ_on[b]) and (self.t < self.occ_on[b] + self.occ_dur[b]):
                fa.append("occlusion")
            text, truth = say(names[self.phase[b]], float(self.heading_est[b]), hv[b].tolist(), fam, slot, float(yaw[b]),
                              (self.home[b, :2] - q.pos[b, :2]).tolist(), roles, self.rh[b].tolist(), fa)
            self.frames[b].append({
                "t": round(self.t, 6), "phase": names[self.phase[b]],
                "outcome": self.outcome_row(b) if self.phase[b] == 4 else None,
                "quad": {"pos": q.pos[b].tolist(), "vel": q.vel[b].tolist(), "quat": q.quat[b].tolist(),
                         "euler_deg": np.degrees(eul[b]).tolist(), "omega": q.omega[b].tolist(),
                         "motors": (q.motor[b] / QP["f_max"]).tolist()},
                "wind": self.wind[b].tolist(),
                "eye": {"w": eye.NC, "h": eye.NR, "rgb": img[b].reshape(-1).tolist()},
                "familiarity": fam,
                "objects": [{"slot": k, "bearing_deg": math.degrees(obs["objects"]["bearing"][b, k]),
                             "strength": float(self.strength[b, k]), "visible": bool(vis_b[k])}
                            for k in range(self.K) if self.S.kvalid[b, k]],
                "ring": {"n": N_RING, "activity": np.asarray(act).tolist(),
                         "drives": {k: np.asarray(v).tolist() for k, v in drv.items()},
                         "heading_cmd_deg": math.degrees(res["heading_cmd"][b])},
                "arbiter": {"name": self.arb_name, "commit": float(res["commit"][b]),
                            "viz": {k: _jsonable(v, b) for k, v in vz.items() if k not in ("ring", "drives")}},
                "pi": {"home_est": hv[b].tolist(), "home_true": (self.home[b, :2] - q.pos[b, :2]).tolist()},
                "avoid": {"yaw_offset_deg": math.degrees(self.yaw_off[b]), "loom": float(self.loom[b])},
                "mind": {"text": text, "truth": truth},
                "faults_active": fa})

    def finish(self):
        t_end = np.where(np.isnan(self.t_end), self.t, self.t_end)
        started = ~np.isnan(self.occ_on) & (self.occ_dur > 0)
        t_hold = np.where(started & np.isnan(self.t_hold) & (self.occ_on < t_end),
                          np.minimum(self.occ_dur, t_end - self.occ_on), self.t_hold)
        t_rel = np.where(~np.isnan(self.t_cross) & np.isnan(self.t_rel), np.minimum(REL_CAP, t_end - self.t_cross),
                         np.minimum(self.t_rel, REL_CAP))
        pos = np.where(np.isnan(self.end_pos), self.quad.pos[:, :2], self.end_pos)
        return {"success_target": self.out["success_target"].copy(), "success_home": self.out["success_home"].copy(),
                "time_s": t_end, "path_len_m": self.path.copy(), "collisions": self.collisions.copy(),
                "home_error_m": np.hypot(pos[:, 0] - self.home[:, 0], pos[:, 1] - self.home[:, 1]),
                "wrong_object_stop": self.out["wrong_object_stop"].copy(), "max_bank_deg": self.max_bank.copy(),
                "energy_proxy": self.energy.copy(), "t_hold_occlusion_s": np.where(self.occ_dur > 0, t_hold, np.nan),
                "t_release_decoy_s": np.where(self.has_dec, t_rel, np.nan)}

    def outcome_row(self, b):
        o = self.finish()
        return {k: (bool(v[b]) if v.dtype == bool else (None if not np.isfinite(v[b]) else float(v[b]))) for k, v in o.items()}

    def run(self, max_time=None):
        T = max_time or max(s["mission"]["max_time_s"] for s in self.sc)
        Tr = np.array([s["mission"]["max_time_s"] for s in self.sc])
        while self.t < T - 1e-9 and (self.phase < 4).any():
            self.step()
            over = (self.phase < 4) & (self.t >= Tr - 1e-9)
            self.t_end = np.where(over, self.t, self.t_end)
            self.end_pos = np.where(over[:, None], self.quad.pos[:, :2], self.end_pos)
            self.phase = np.where(over, 4, self.phase)
        return self.finish()


def _jsonable(v, b):
    if isinstance(v, dict):
        return {k: _jsonable(x, b) for k, x in v.items()}
    a = np.asarray(v)[b]
    return a.tolist() if isinstance(a, np.ndarray) else (a.item() if hasattr(a, "item") else a)


def run_batch(scenarios, split=None, log_rows=None):
    """scenarios: list of C1 dicts (from scenario.sample_scenario or load). Groups rows by arbiter config and runs
    each group as one batch. -> pandas.DataFrame with the C7 columns (+ fault, severity from _meta when present)."""
    import pandas as pd
    groups = {}
    for i, s in enumerate(scenarios):
        b = load(strip(s))["brain"]
        groups.setdefault(json.dumps([b["arbiter"], b["params"], b["ring_gain"]], sort_keys=True), []).append(i)
    rows = [None] * len(scenarios)
    st = stamp(param_source="scripts/mission.py constants; brain.params from the scenario")
    for idx in groups.values():
        m = Mission([scenarios[i] for i in idx])
        o = m.run()
        for j, i in enumerate(idx):
            s = load(strip(scenarios[i]))
            meta = scenarios[i].get("_meta", {})
            rows[i] = {"scenario_hash": scenario_hash(strip(s)), "seed": s["seed"], "arbiter": s["brain"]["arbiter"],
                       "ring_gain": s["brain"]["ring_gain"], "split": split or meta.get("split", "train"),
                       **{k: (bool(v[j]) if v.dtype == bool else (int(v[j]) if k == "collisions" else float(v[j]))) for k, v in o.items()},
                       "git": st["git"], "sim_version": st["sim_version"], "created": st["created"],
                       "fault": meta.get("fault"), "severity": meta.get("severity")}
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from scenario import sample_scenario
    if "--time" in sys.argv:
        B = 50
        scs = [sample_scenario(s, arbiter="ring", params={"g_home": 3.0}) for s in range(B)]   # faults drawn, meadow 0.3
        t0 = time.time()
        m = Mission(scs)
        t1 = time.time()
        o = m.run()
        t2 = time.time()
        print(f"B={B} x 300 s (sampled faults, meadow 0.3, ring): init {t1 - t0:.1f} s, run {t2 - t1:.1f} s wall; "
              f"batch ends at t={m.t:.0f} s; mean mission end {np.mean(o['time_s']):.0f} s; success {np.mean(o['success_target'] & o['success_home']):.2f}")
        sys.exit(0)
    t0 = time.time()
    res = {}
    for arb in ("ring", "vote"):
        prm = {"g_home": 3.0} if arb == "ring" else {}     # the default ring never turns home 
        scs = [sample_scenario(s, fault="none", arbiter=arb, params=prm, obstacle_density=0.1, max_time_s=150.0) for s in range(10)]
        df = run_batch(scs)
        res[arb] = df
        print(arb, df[["seed", "success_target", "success_home", "time_s", "collisions", "home_error_m", "wrong_object_stop"]].to_string())
    print(f"self-check runs took {time.time() - t0:.0f} s")
    for arb, df in res.items():
        assert set(df.columns) >= set(COLUMNS)
        assert df.success_target.sum() >= 6, (arb, df.success_target.sum())
        assert (df.success_home | ~df.success_target).sum() >= 6 and df.success_home.sum() >= 5, arb
        assert (df.collisions == 0).all() or df.collisions.sum() <= 2
    print("mission self-check OK")
